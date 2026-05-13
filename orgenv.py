import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

# ---------------------------------------------------------------------------
# UAVJSCEnv  —  Hybrid MARL-ADMM Joint Sensing & Communication environment
#
# Architecture:
#   • UAV 0 is the LEADER UAV. During training it hosts the centralized
#     Q-critic Q(s_1..N, a_1..N) and coordinates ADMM consensus across the
#     swarm. UAV 0 is still a full actor-agent — it selects its own actions
#     using its local policy π_θ0(a_0 | s_0) just like every other UAV.
#     The critic role is a training-time responsibility only; at deployment
#     all UAVs execute their distributed actors autonomously.
#   • UAVs 1..N-1 are FOLLOWER UAVs. They maintain local actors, report
#     their R_i / Q_i to the leader for consensus, and receive z^R_i, z^Q_i
#     and dual-variable updates back from the leader.
#   • Each UAV i has its OWN independent channel h_i to the ground receiver.
#     Orthogonal access is assumed so there is no direct uplink interference
#     from other UAVs on UAV i's own link (Section IV-A).
#   • I_i = interference received AT UAV i from OTHER UAVs' concurrent
#     transmissions (side-lobe / sensing leakage). It is interference caused
#     BY others, not BY UAV i.
#   • Phase noise σ²_φ,i and PA nonlinearity a3,i appear in both the
#     impairment-aware SINR (Eqs 11, 15) and as explicit reward penalties
#     (Eq 32) so agents learn to prefer low-impairment operating points.
#   • Clock speed c_i is heterogeneous per UAV; its processing cost ξ_i c_i
#     is penalised in the reward (mentor requirement).
#   • ν_i (utility generation rate) is heterogeneous: UAV 0 (leader) runs
#     at a higher rate, reflecting its added processing responsibility.
#   • z^R_i and z^Q_i are per-agent local copies (mentor correction).
#     The leader computes the global mean and broadcasts it each step.
# ---------------------------------------------------------------------------


class UAVJSCEnv(ParallelEnv):
    metadata = {"name": "uav_jsc_v0"}

    def __init__(
        self,
        num_uavs: int = 3,
        motion_case: str = "all_move",
        fading_on: bool = True,
        horizon: int = 100,
    ):
        # ------------------------------------------------------------------
        # 0) Agent identities and episode length
        # ------------------------------------------------------------------
        self.possible_agents = [f"uav_{i}" for i in range(num_uavs)]
        self.agents = self.possible_agents[:]
        self.horizon = int(horizon)

        # UAV 0 is the designated leader: hosts the centralized critic
        # and coordinates ADMM consensus during training (Section IV-C).
        self.leader_uav = self.possible_agents[0]   # "uav_0"

        # ------------------------------------------------------------------
        # 1) Physical / RF constants  (paper Section VII-A)
        # ------------------------------------------------------------------
        self.max_power   = 1.0        # normalised P_max
        self.noise       = 1e-3       # N0 thermal noise (W)
        self.bandwidth   = 1e6        # B = 1 MHz
        self.channel_dim = 2          # beamforming vector dimension

        # ------------------------------------------------------------------
        # 2) Observation and action spaces
        #
        #   obs_i = [tanh(|h_i|), E_i, σ²_φ,i, σ²_t,i·1e9, a3_i, c_i,
        #            tanh(I_i), tanh(z^R_i/BW), tanh(z^Q_i/10),
        #            x_i/L, y_i/L, t/T]           dim=12
        #
        #   act_i = [P_i, w1_i, w2_i]             dim=3
        # ------------------------------------------------------------------
        self.obs_dim = 12
        self.act_dim = 3

        self.observation_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf,
                          shape=(self.obs_dim,), dtype=np.float32)
            for a in self.agents
        }
        self.action_spaces = {
            a: spaces.Box(
                low =np.array([0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([self.max_power, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for a in self.agents
        }

        # ------------------------------------------------------------------
        # 3) ADMM parameters  (Section V)
        # ------------------------------------------------------------------
        self.rho          = 0.05      # initial penalty parameter ρ
        self.rho_min      = 0.01
        self.rho_max      = 1.0
        self.rho_up       = 1.05
        self.rho_down     = 0.95
        self.rho_ratio    = 5.0       # Boyd et al. §3.4.1
        self.rho_update_period = 5
        self.LAMBDA_MAX_R = 10.0
        self.LAMBDA_MAX_Q = 2.0   # sensing_norm in [0,1] → tighter clip prevents saturation

        self.z_R      = {a: 0.0 for a in self.agents}  # z^R_i per agent
        self.z_Q      = {a: 0.0 for a in self.agents}  # z^Q_i per agent
        self.lambda_R = {a: 0.0 for a in self.agents}  # λ^R_i
        self.lambda_Q = {a: 0.0 for a in self.agents}  # λ^Q_i

        self.res_ema = 0.1
        self.r_ema   = 0.0
        self.s_ema   = 0.0
        self.last_primal_residual = 0.0
        self.last_dual_residual   = 0.0
        self.last_z_R = 0.0
        self.last_z_Q = 0.0
        self.last_jain = 1.0

        # ------------------------------------------------------------------
        # 4) Channel / fading model  (3GPP TDL Rayleigh, Section VII-A)
        # ------------------------------------------------------------------
        self.alpha         = 2.2
        self.d0            = 1.0
        self.PL0           = 1e-4
        self.sigma_sh_db   = 4.0
        self.rho_fade      = 0.97
        self.channel_model = "rayleigh"
        self.K_rician      = 5.0
        self.kappa_t       = 1.0      # timing-jitter sensitivity κ_t (Eq 14)

        # ------------------------------------------------------------------
        # 5) Reward / utility parameters  (Eq 32)
        # ------------------------------------------------------------------
        self.alpha_comm = 1.0   # α_i  comm weight
        self.beta_sense = 0.3    # β_i  sensing weight (Eq 5 — co-equal with rate)

        # ν_i: heterogeneous utility generation rates (Section III-C, mentor).
        # The leader UAV (UAV 0) operates at ν=1.2 — its higher processing
        # clock speed and critic-coordination role justify a higher utility
        # generation rate. Follower UAVs run at ν=1.0.
        self.nu = {a: (1.2 if i == 0 else 1.0)
                   for i, a in enumerate(self.agents)}

        self.mu_power    = 0.35   # μ_i   power penalty (was 0.15 → agents drain battery)
        self.xi_clock    = 0.05   # ξ_i   clock-speed penalty
        self.eta_R       = 3.0    # η_R   rate-fairness penalty (strong — must dominate utility spread)
        self.eta_Q       = 1.5    # η_Q   sensing-fairness penalty
        self.chi_dist    = 0.1    # χ_i   PA-distortion penalty
        self.omega_phase = 0.05   # ω_i   phase-noise penalty
        self.zeta_timing = 0.01   # ζ_i   timing-jitter penalty
        self.dp_w        = 0.02   # power-smoothness shaping
        self.reward_scale = 0.05  # halved — keeps rewards in [-2,2] for stable critic

        # ------------------------------------------------------------------
        # 6) Sensing model  (Eq 4)
        #
        #   Q_i = P_i |g^s_i|² / σ²_s
        #   With sensing_noise=1.0 and g^s_i ~ Rayleigh(1), Q_i ~ O(0-2),
        #   which matches R_tilde = R_i/B ~ O(1-30) in the reward.
        #   The previous value of 1e-3 made Q_i ~ O(1000), completely
        #   overwhelming the rate term and causing power collapse.
        # ------------------------------------------------------------------
        self.sensing_noise = 1.0    # σ²_s  (keeps Q_i ~ O(0-2))

        # ------------------------------------------------------------------
        # 7) Mobility / geometry  (1000×1000 m², Section VII-B)
        # ------------------------------------------------------------------
        self.fading_on   = fading_on
        self.motion_case = motion_case
        self.omega       = 0.05
        self.region_size = 1000.0
        self.center      = np.array([500.0, 500.0], dtype=np.float32)
        self.radius      = 150.0
        self.rx_pos      = np.array([800.0, 500.0], dtype=np.float32)

        # ------------------------------------------------------------------
        # 8) Runtime state (filled in reset)
        # ------------------------------------------------------------------
        self.step_count        = 0
        self.orbit_phase_offset = 0.0   # randomised in reset() each episode
        self.positions     = {a: np.zeros(2, dtype=np.float32) for a in self.agents}
        self.distance      = {a: 0.0 for a in self.agents}
        self.energy        = {a: 1.0 for a in self.agents}
        self.channel       = {a: np.ones(self.channel_dim, dtype=np.complex64)
                              for a in self.agents}
        self.h_fade        = {a: np.ones(self.channel_dim, dtype=np.complex64)
                              for a in self.agents}
        self.shadow_db     = {a: 0.0 for a in self.agents}
        self.phase_noise   = {a: 0.0 for a in self.agents}
        self.timing_jitter_var = {a: 0.0 for a in self.agents}
        self.pa_coeff      = {a: 0.02 for a in self.agents}
        self.clock_speed   = {a: 1.0  for a in self.agents}
        self.sensing_gain  = {a: 1.0  for a in self.agents}
        self.phase         = {a: 0.0  for a in self.agents}
        self.cfo           = {a: 0.0  for a in self.agents}
        self.clock_skew    = {a: 0.0  for a in self.agents}
        self.interference  = {a: 0.0  for a in self.agents}
        self.prev_power    = {a: 0.0  for a in self.agents}
        self.last_rates        = {a: 0.0 for a in self.agents}
        self.last_sensing      = {a: 0.0 for a in self.agents}
        self.last_powers       = {a: 0.0 for a in self.agents}
        self.last_interference = {a: 0.0 for a in self.agents}
        self.last_distortion   = {a: 0.0 for a in self.agents}

        self.reset()

    # ======================================================================
    # reset()
    # ======================================================================
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents     = self.possible_agents[:]
        self.step_count = 0

        # Shadowing drawn once per episode
        self.shadow_db = {
            a: float(np.random.normal(0.0, self.sigma_sh_db))
            for a in self.agents
        }

        # Initialise fading state vectors
        self.h_fade  = {}
        self.channel = {}
        for a in self.agents:
            h_vec = (
                np.random.normal(0.0, 1/np.sqrt(2), size=self.channel_dim)
                + 1j * np.random.normal(0.0, 1/np.sqrt(2), size=self.channel_dim)
            ).astype(np.complex64)
            self.h_fade[a]  = h_vec
            self.channel[a] = h_vec.copy()

        # Physical resources
        self.energy     = {a: 1.0 for a in self.agents}
        self.prev_power = {a: 0.0 for a in self.agents}
        self.interference = {a: 0.0 for a in self.agents}

        # Hardware impairments — heterogeneous, fixed per episode
        self.phase_noise = {
            a: float(np.random.uniform(0.01, 0.30)) for a in self.agents
        }
        self.timing_jitter_var = {
            a: float(np.random.uniform(1e-10, 1e-8)) for a in self.agents
        }
        self.pa_coeff = {
            a: float(np.random.uniform(0.01, 0.05)) for a in self.agents
        }
        # Clock speed c_i (heterogeneous across UAVs — mentor requirement)
        self.clock_speed = {
            a: float(np.random.uniform(0.8, 1.2)) for a in self.agents
        }
        # Rayleigh(0.5): tighter gain spread reduces inter-agent Q variance
        # from ~40x range (scale=1) to ~10x, making ADMM Q-fairness tractable
        self.sensing_gain = {
            a: float(np.random.rayleigh(scale=0.5)) for a in self.agents
        }
        self.phase = {
            a: float(np.random.uniform(-np.pi, np.pi)) for a in self.agents
        }
        self.cfo = {
            a: float(np.random.uniform(-20e-6, 20e-6)) for a in self.agents
        }
        self.clock_skew = {
            a: float(np.random.uniform(-5e-6, 5e-6)) for a in self.agents
        }

        # Geometry: evenly spaced on circle with random initial phase per episode
        # This gives trajectory diversity — otherwise all episodes start and orbit
        # identically (deterministic omega*t), providing zero positional diversity.
        N = len(self.agents)
        self.orbit_phase_offset = float(np.random.uniform(0, 2 * np.pi))
        self.positions = {}
        for i, a in enumerate(self.agents):
            angle = self.orbit_phase_offset + i * (2.0 * np.pi / max(N, 1))
            self.positions[a] = self.center + self.radius * np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32
            )
        for a in self.agents:
            self._update_channel(a)

        # ADMM state
        self.z_R      = {a: 0.0 for a in self.agents}
        self.z_Q      = {a: 0.0 for a in self.agents}
        self.lambda_R = {a: 0.0 for a in self.agents}
        self.lambda_Q = {a: 0.0 for a in self.agents}
        self.r_ema    = 0.0
        self.s_ema    = 0.0
        self.last_primal_residual = 0.0
        self.last_dual_residual   = 0.0
        self.last_z_R  = 0.0
        self.last_z_Q  = 0.0
        self.last_jain = 1.0

        # Diagnostic caches
        self.last_rates        = {a: 0.0 for a in self.agents}
        self.last_sensing      = {a: 0.0 for a in self.agents}
        self.last_powers       = {a: 0.0 for a in self.agents}
        self.last_interference = {a: 0.0 for a in self.agents}
        self.last_distortion   = {a: 0.0 for a in self.agents}

        return self._get_obs(), {}

    # ======================================================================
    # _get_obs()
    # ======================================================================
    def _get_obs(self):
        obs    = {}
        t_norm = self.step_count / max(self.horizon, 1)
        for a in self.agents:
            h_mag   = float(np.linalg.norm(self.channel[a]))
            # z_R is stored in bits/s/Hz (O(1-30)), z_Q in sensing units (O(0-2))
            zR_feat = float(np.tanh(self.z_R[a] / 20.0))   # normalise by ~peak SE
            zQ_feat = float(np.tanh(self.z_Q[a]))    # normalise by ~peak Q
            obs[a]  = np.array([
                float(np.tanh(h_mag)),                        # 0  |h_i| (normalised)
                float(self.energy[a]),                        # 1  E_res,i
                float(self.phase_noise[a]),                   # 2  σ²_φ,i
                float(self.timing_jitter_var[a] * 1e9),       # 3  σ²_t,i (scaled)
                float(self.pa_coeff[a]),                      # 4  a3,i
                float(self.clock_speed[a]),                   # 5  c_i
                float(np.tanh(self.interference[a])),         # 6  I_i (normalised)
                zR_feat,                                      # 7  z^R_i
                zQ_feat,                                      # 8  z^Q_i
                float(self.positions[a][0] / self.region_size),  # 9  x/L
                float(self.positions[a][1] / self.region_size),  # 10 y/L
                float(t_norm),                                # 11 t/T
            ], dtype=np.float32)
        return obs

    # ======================================================================
    # _update_channel()  — AR(1) vector Rayleigh/Rician fading + path loss
    # ======================================================================
    def _update_channel(self, a: str):
        d = max(float(np.linalg.norm(self.positions[a] - self.rx_pos)), 0.5)
        self.distance[a] = d
        pathloss = self.PL0 * (d / self.d0) ** (-self.alpha)
        sh_lin   = 10.0 ** (self.shadow_db[a] / 10.0)

        w_vec = (
            np.random.normal(0, 1/np.sqrt(2), size=self.channel_dim)
            + 1j * np.random.normal(0, 1/np.sqrt(2), size=self.channel_dim)
        ).astype(np.complex64)
        self.h_fade[a] = (self.rho_fade * self.h_fade[a]
                          + np.sqrt(1 - self.rho_fade**2) * w_vec)

        if self.channel_model == "rician":
            K    = float(self.K_rician)
            h_ss = (np.sqrt(K/(K+1)) * np.ones(self.channel_dim, dtype=np.complex64)
                    + np.sqrt(1/(K+1)) * self.h_fade[a])
        else:
            h_ss = self.h_fade[a]

        self.channel[a] = (np.sqrt(pathloss * sh_lin) * h_ss).astype(np.complex64)

    # ======================================================================
    # step()
    # ======================================================================
    def step(self, actions):
        rewards      = {}
        rates        = {}
        sensing      = {}
        powers       = {}
        dps          = {}
        p_dist       = {}
        interference = {}
        beamformers  = {}

        # ------------------------------------------------------------------
        # 1) Oscillator / impairment evolution (phase, CFO, clock skew)
        # ------------------------------------------------------------------
        Ts = 1.0
        for a in self.agents:
            eff_Ts = Ts * (1.0 + self.clock_skew[a])
            self.phase[a] += 2.0 * np.pi * self.cfo[a] * eff_Ts
            self.phase[a] += float(np.random.normal(
                0.0, np.sqrt(max(self.phase_noise[a], 1e-12))))
            self.phase[a] = (self.phase[a] + np.pi) % (2.0 * np.pi) - np.pi

        # ------------------------------------------------------------------
        # 2) Mobility update
        # ------------------------------------------------------------------
        N = len(self.agents)
        for i, a in enumerate(self.agents):
            case = str(self.motion_case)
            if case == "uav0_stationary":  case = "two_move"
            if case == "uav01_stationary": case = "one_move"

            if   case == "all_move":  moving = True
            elif case == "two_move":  moving = (N <= 2) or (a not in {self.agents[0]})
            elif case == "one_move":  moving = (a == self.agents[-1])
            else:                     moving = True

            if moving:
                angle = self.orbit_phase_offset + self.omega * self.step_count + i * (2.0 * np.pi / N)
                self.positions[a] = self.center + self.radius * np.array(
                    [np.cos(angle), np.sin(angle)], dtype=np.float32)

        # ------------------------------------------------------------------
        # 3) Channel update
        # ------------------------------------------------------------------
        for a in self.agents:
            if self.fading_on:
                self._update_channel(a)
            else:
                d  = max(float(np.linalg.norm(self.positions[a] - self.rx_pos)), 0.5)
                self.distance[a] = d
                pl = self.PL0 * (d / self.d0) ** (-self.alpha)
                self.channel[a]  = (np.sqrt(pl)
                                    * np.ones(self.channel_dim, dtype=np.complex64))

        # ------------------------------------------------------------------
        # 4) Parse actions: a_i = {P_i, w_i}  (paper Eq 30)
        # ------------------------------------------------------------------
        for a in self.agents:
            act = np.asarray(actions[a], dtype=np.float32).ravel()
            if act.size < 3:
                raise ValueError(
                    f"{a}: need [P, w1, w2], got size {act.size}")

            p_cmd = float(np.clip(act[0], 0.0, 1.0))
            p = float(np.clip(p_cmd, 0.0, self.max_power * self.energy[a]))

            dp = p - self.prev_power[a]
            self.prev_power[a] = p

            w = act[1:3].astype(np.float64)
            n = np.linalg.norm(w)
            w = w / n if n >= 1e-8 else np.array([1.0, 0.0])

            beamformers[a] = w
            powers[a]      = p
            dps[a]         = dp

        # ------------------------------------------------------------------
        # 5) Energy depletion  (Eq 9)
        # ------------------------------------------------------------------
        for a in self.agents:
            self.energy[a] = max(self.energy[a] - powers[a] / 100.0, 0.0)

        # ------------------------------------------------------------------
        # 6) Impairment-aware SINR, rates, and sensing  (Eqs 1-4, 11, 14, 15)
        #
        #   MENTOR CLARIFICATION (implemented here):
        #   • UAV i's uplink to base station uses channel h_i exclusively.
        #     No direct uplink interference from other UAVs (orthogonal access).
        #   • I_i = interference seen AT UAV i from OTHER UAVs' concurrent
        #     transmissions, modelled as cross-channel leakage:
        #       I_i = Σ_{j≠i}  P_j |h_j^T w_j|²
        #     This is interference RECEIVED by i, not CAUSED by i.
        # ------------------------------------------------------------------
        for a in self.agents:
            h_a = np.asarray(self.channel[a])

            # Impairment attenuation factors (Eqs 11, 14)
            phase_att  = np.exp(-self.phase_noise[a])                   # exp(-σ²_φ,i)
            timing_att = np.exp(-self.kappa_t * self.timing_jitter_var[a])

            # Desired signal: P_i · att · |h_i^T w_i|²
            desired_gain   = float(np.abs(np.vdot(h_a, beamformers[a])) ** 2)
            desired_signal = powers[a] * phase_att * timing_att * desired_gain

            # PA distortion P_dist,i = a3,i · P_i³  (Eq 15)
            p_dist[a] = float(self.pa_coeff[a] * (powers[a] ** 3))

            # Interference I_i from OTHER UAVs (mentor correction)
            I_a = 0.0
            for b in self.agents:
                if b == a:
                    continue
                h_b = np.asarray(self.channel[b])
                I_a += powers[b] * float(np.abs(np.vdot(h_b, beamformers[b])) ** 2)
            interference[a]      = float(I_a)
            self.interference[a] = float(I_a)

            # SINR (Eq 2)
            sinr_a   = desired_signal / (self.noise + I_a + p_dist[a] + 1e-15)
            # Rate R_i = B log2(1 + SINR_i)  (Eq 1)
            rates[a] = float(self.bandwidth * np.log2(1.0 + sinr_a))

            # Sensing Q_i = P_i |g^s_i|² / σ²_s  (Eq 4)
            sensing[a] = float(
                powers[a] * (self.sensing_gain[a] ** 2) / (self.sensing_noise + 1e-15)
            )

        # ------------------------------------------------------------------
        # 7) ADMM consensus & dual update  (Section V, Eqs 26-28)
        #
        #   The LEADER UAV (uav_0) aggregates R_i and Q_i from all followers,
        #   computes the global consensus mean z, broadcasts z^R_i = z^Q_i = z
        #   back to every agent, then each agent performs its own dual update.
        #
        #   IMPORTANT: ADMM operates on NORMALISED quantities:
        #     R̃_i = R_i / B  (bits/s/Hz,  O(1-30))
        #     Q_i  = sensing (O(0-2) with sensing_noise=1.0)
        #   This keeps dual variables and residuals at tractable scale and
        #   prevents lambda from saturating in one step when rates are in bps.
        # ------------------------------------------------------------------
        lambda_R_old = self.lambda_R.copy()
        lambda_Q_old = self.lambda_Q.copy()
        rho = float(self.rho)

        # Normalise rates to bits/s/Hz for ADMM (same units as reward R̃_i)
        rates_norm = {a: rates[a] / self.bandwidth for a in self.agents}

        max_Q = max(max(sensing.values()), 1e-8)
        sensing_norm = {a: sensing[a] / max_Q for a in self.agents}

        # Previous consensus values for dual residual
        z_R_prev = float(np.mean([self.z_R[a] for a in self.agents]))
        z_Q_prev = float(np.mean([self.z_Q[a] for a in self.agents]))

        # MIN-BIASED CONSENSUS (paper Eq 6 — max-min fairness)
        # Standard mean consensus enforces each agent tracks the average.
        # Max-min fairness (Eq 6) requires the WORST agent to improve.
        # We bias the consensus target toward the minimum: z = (mean + min)/2
        # This means agents above z^R feel a penalty pulling them down,
        # while the weakest agent below z^R feels a strong pull upward.
        # All paper equations (26-28) remain valid — only the consensus
        # target is biased, which is a valid choice of the global variable z.
        mean_R = float(np.mean([rates_norm[a]   for a in self.agents]))
        mean_Q = float(np.mean([sensing_norm[a] for a in self.agents]))
        min_R  = float(np.min( [rates_norm[a]   for a in self.agents]))
        min_Q  = float(np.min( [sensing_norm[a] for a in self.agents]))

        # Bias weight γ=0.5: equal mix of mean and min
        # γ=0 → pure mean (standard ADMM), γ=1 → pure min (max-min fairness)
        gamma = 0.5
        z_R_global = (1 - gamma) * mean_R + gamma * min_R
        z_Q_global = (1 - gamma) * mean_Q + gamma * min_Q
        for a in self.agents:
            self.z_R[a] = z_R_global
            self.z_Q[a] = z_Q_global

        # Dual update every step  (Eqs 27-28) — normalised residuals
        for a in self.agents:
            self.lambda_R[a] = float(np.clip(
                lambda_R_old[a] + rho * (rates_norm[a] - self.z_R[a]),
                -self.LAMBDA_MAX_R, self.LAMBDA_MAX_R))

            self.lambda_Q[a] = float(np.clip(
                lambda_Q_old[a] + rho * (sensing_norm[a] - self.z_Q[a]),
                -self.LAMBDA_MAX_Q, self.LAMBDA_MAX_Q))

        # Residuals (normalised)
        primal_R   = float(np.sqrt(np.mean(
            [(rates_norm[a]   - self.z_R[a])**2 for a in self.agents])))
        primal_Q   = float(np.sqrt(np.mean(
            [(sensing_norm[a] - self.z_Q[a])**2 for a in self.agents])))
        primal_res = float(np.sqrt(primal_R**2 + primal_Q**2))
        dual_res   = float(rho * np.sqrt(
            (z_R_global - z_R_prev)**2 + (z_Q_global - z_Q_prev)**2))

        # Adaptive ρ  (Boyd et al. §3.4.1)
        if self.step_count % self.rho_update_period == 0:
            if primal_res > self.rho_ratio * (dual_res + 1e-12):
                self.rho = min(self.rho * self.rho_up,   self.rho_max)
            elif dual_res > self.rho_ratio * (primal_res + 1e-12):
                self.rho = max(self.rho * self.rho_down, self.rho_min)

        # Logging — z_R/z_Q are now in normalised units (bits/s/Hz and sensing)
        self.r_ema = (1 - self.res_ema) * self.r_ema + self.res_ema * primal_res
        self.s_ema = (1 - self.res_ema) * self.s_ema + self.res_ema * dual_res
        self.last_primal_residual = primal_res
        self.last_dual_residual   = dual_res
        self.last_z_R  = z_R_global   # bits/s/Hz
        self.last_z_Q  = z_Q_global   # sensing units
        self.last_jain = self._jain_index(rates)  # Jain on raw bps rates

        self.last_rates        = dict(rates)
        self.last_sensing      = dict(sensing)
        self.last_powers       = dict(powers)
        self.last_interference = dict(interference)
        self.last_distortion   = dict(p_dist)

        # ------------------------------------------------------------------
        # 8) Reward  (Eq 32)
        #
        #   r_i = ν_i(α_i R̃_i + β_i Q_i)
        #         − μ_i P_i                    power penalty
        #         − ξ_i c_i                    clock-speed penalty (mentor)
        #         − η_R (R̃_i − z^R_i)²        rate fairness (both normalised)
        #         − η_Q (Q_i  − z^Q_i)²        sensing fairness
        #         − χ_i P_dist,i               PA distortion penalty
        #         − ω_i σ²_φ,i                 phase-noise penalty
        #         − ζ_i σ²_t,i                 timing-jitter penalty
        #
        #   R̃_i = R_i/B (bits/s/Hz) ~ O(1-30).
        #   z^R_i is stored in bits/s/Hz (normalised ADMM), so units match.
        #   Q_i_raw = raw sensing utility term
        #   Q_i     = sensing_norm[a] used for fairness term
        #   z^Q_i   = ADMM consensus on normalized sensing
        #
        #   Battery shaping term REMOVED: -1*(1-E)^2 was the primary cause
        #   of power collapse — it grew as energy depleted, training agents
        #   to conserve energy by doing nothing.
        # ------------------------------------------------------------------
        for a in self.agents:
            R_tilde  = rates[a] / self.bandwidth
            zR_tilde = self.z_R[a]

            Q_raw  = sensing[a]
            Q_norm = sensing_norm[a]
            zQ_i   = self.z_Q[a]

            # FAIRNESS PENALTY (Eq 32, η_R and η_Q terms)
            # Symmetric quadratic: penalises deviation in BOTH directions.
            # (R̃_i - z^R_i)^2 is large for both high AND low outliers,
            # so all agents converge toward the same operating point.
            # Because z^R is min-biased, all agents converge toward the
            # weakest agent's rate — which is exactly max-min fairness (Eq 6).
            fair_R = self.eta_R * (R_tilde  - zR_tilde) ** 2
            fair_Q = self.eta_Q * (Q_norm   - zQ_i)    ** 2

            # SWARM-LEVEL JAIN BONUS: direct term from Eq 38.
            # All agents share a collective bonus proportional to the
            # current Jain index. This gives every agent an incentive to
            # improve the worst-off agent (raising Jain raises everyone's reward).
            jain_bonus = 2.0 * self.last_jain   # scale to ~O(1) since Jain in [0,1]

            r_i = (
                self.nu[a] * (self.alpha_comm * R_tilde + self.beta_sense * Q_raw)
                - self.mu_power    * powers[a]
                - self.xi_clock    * self.clock_speed[a]
                - fair_R
                - fair_Q
                - self.chi_dist    * p_dist[a]
                - self.omega_phase * self.phase_noise[a]
                - self.zeta_timing * self.timing_jitter_var[a]
                + jain_bonus       # collective fairness reward
            )

            r_i -= self.dp_w * (dps[a] ** 2)
            rewards[a] = float(self.reward_scale * r_i)

        # ------------------------------------------------------------------
        # 9) Termination
        # ------------------------------------------------------------------
        self.step_count += 1
        done = (self.step_count >= self.horizon
                or any(self.energy[a] <= 0.0 for a in self.agents))

        return (
            self._get_obs(),
            rewards,
            {a: done for a in self.agents},
            {a: False for a in self.agents},
            {}
        )

    # ======================================================================
    # Helpers
    # ======================================================================
    def _jain_index(self, rates: dict) -> float:
        vals = np.array(list(rates.values()), dtype=np.float64)
        if vals.size == 0 or np.sum(vals**2) < 1e-15:
            return 1.0
        return float(np.sum(vals)**2 / (len(vals) * np.sum(vals**2)))

    def scenario_string(self) -> str:
        rx_x, rx_y = float(self.rx_pos[0]), float(self.rx_pos[1])
        return (
            f"N={len(self.agents)} | {self.motion_case} | {self.channel_model} | "
            f"pn_max={max(self.phase_noise.values()):.3f} | "
            f"rho={self.rho:.4f} | rx=({rx_x:.0f},{rx_y:.0f})"
        )