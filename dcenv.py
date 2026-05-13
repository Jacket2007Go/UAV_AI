"""
env.py  —  UAVJSCEnv: Hybrid MARL-ADMM Joint Sensing & Communication environment

Proposal architecture mapping
══════════════════════════════════════════════════════════════════════════════
 Step 1  Certificate update          → update_certificates()
 Step 2  AI warm start               → (actors in train.py; actions arrive here)
 Step 3  Certified MM-ADMM refinement→ ADMM block inside step()
 Step 4  Barrier/QP safe execution   → project_safe_actions() / SafeActionProjector
 Step 5  Realized measurement/learn  → logging at end of step(); train.py update

Distributed-critic / rotating-prefect mapping (PDF §2-3)
══════════════════════════════════════════════════════════════════════════════
 select_active_critic()              routes to fixed / round-robin / weighted
 compute_weighted_critic_scores()    implements PDF Eq.(2) Q_i(t)
 critic_duty_count / duty_share      track rotating-prefect duty load
 cert_* variables                    certificate-style fairness memory (Step 1)

Architecture:
  • self.default_leader_uav  = 'uav_0' — fallback and fixed-mode host.
  • self.active_critic_agent = dynamically updated critic host.
  • self.leader_uav kept as alias for backward compatibility.
  • ADMM consensus logic is unchanged; the coordinator role may rotate but
    the mathematical update (Eqs 26-28) is the same regardless of host.
  • Certificate variables are lightweight EMA-based online fairness memory;
    they do NOT constitute formal proofs — they implement the structural
    substrate described in the proposal.
"""

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces


class UAVJSCEnv(ParallelEnv):
    metadata = {"name": "uav_jsc_v0"}

    def __init__(
        self,
        num_uavs: int = 3,
        motion_case: str = "all_move",
        fading_on: bool = True,
        horizon: int = 100,
        critic_mode: str = "fixed",         # 'fixed' | 'round_robin' | 'weighted'
    ):
        # ------------------------------------------------------------------
        # 0) Agent identities and episode length
        # ------------------------------------------------------------------
        self.possible_agents = [f"uav_{i}" for i in range(num_uavs)]
        self.agents          = self.possible_agents[:]
        self.horizon         = int(horizon)

        # ── Rotating-prefect / distributed critic coordinator ──────────────
        # Proposal Step 1 context: the 'default_leader_uav' is the permanent
        # fallback coordinator (uav_0) used in 'fixed' mode and as the
        # safety fallback in round-robin / weighted modes.
        # 'active_critic_agent' is the CURRENT logical critic host and can
        # change each episode or step depending on critic_mode.
        self.critic_mode          = str(critic_mode)
        self.default_leader_uav   = self.possible_agents[0]   # "uav_0"
        self.active_critic_agent  = self.possible_agents[0]   # initialised to uav_0
        # Backward-compat alias (train.py used env.leader_uav)
        self.leader_uav           = self.default_leader_uav

        # Per-agent critic duty tracking (rotating-prefect accounting)
        self.critic_duty_count    = {a: 0 for a in self.agents}
        self.critic_duty_share    = {a: 0.0 for a in self.agents}
        self.critic_selection_step = 0   # incremented each call to select_active_critic

        # Weighted-score cache for diagnostics / logging
        self.last_weighted_scores      = {a: 0.0 for a in self.agents}
        self.last_weighted_components  = {}   # {agent: {E,L,C,S,P,W,Q}}

        # ------------------------------------------------------------------
        # 1) Physical / RF constants  (paper Section VII-A)
        # ------------------------------------------------------------------
        self.max_power   = 1.0
        self.noise       = 1e-3
        self.bandwidth   = 1e6
        self.channel_dim = 2

        # ------------------------------------------------------------------
        # 2) Observation and action spaces
        #
        #   obs_i = [tanh(|h_i|), E_i, σ²_φ,i, σ²_t,i·1e9, a3_i, c_i,
        #            tanh(I_i), tanh(z^R_i/20), tanh(z^Q_i),
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
        self.rho               = 0.1
        self.rho_min           = 0.01
        self.rho_max           = 1.0
        self.rho_up            = 1.05
        self.rho_down          = 0.95
        self.rho_ratio         = 2.0
        self.rho_update_period = 5
        self.LAMBDA_MAX_R      = 10.0
        self.LAMBDA_MAX_Q      = 2.0

        self.z_R      = {a: 0.0 for a in self.agents}
        self.z_Q      = {a: 0.0 for a in self.agents}
        self.lambda_R = {a: 0.0 for a in self.agents}
        self.lambda_Q = {a: 0.0 for a in self.agents}

        self.res_ema  = 0.1
        self.r_ema    = 0.0
        self.s_ema    = 0.0
        self.last_primal_residual = 0.0
        self.last_dual_residual   = 0.0
        self.last_z_R  = 0.0
        self.last_z_Q  = 0.0
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
        self.kappa_t       = 1.0

        # ------------------------------------------------------------------
        # 5) Reward / utility parameters  (Eq 32)
        #
        # IMPORTANT: alpha_comm and beta_sense MUST match orgenv exactly,
        # otherwise the Original (orgenv-based) baseline and the
        # DC/Cert/DC+Cert (dcenv-based) variants are optimising different
        # objective functions and the comparison is no longer an ablation.
        # Previous values (alpha_comm=2.0, beta_sense=0.2) gave a 10:1
        # rate:sensing weighting in dcenv vs 3.3:1 in orgenv, which is
        # why Sum Sensing for DC variants was ~20% below Original — they
        # were literally being trained to deprioritise sensing.
        # ------------------------------------------------------------------
        self.alpha_comm = 1.0    # MUST match orgenv (was 2.0 — bug)
        self.beta_sense = 0.3    # MUST match orgenv (was 0.2 — bug)
        self.nu          = {a: (1.2 if i == 0 else 1.0)
                            for i, a in enumerate(self.agents)}
        self.mu_power    = 0.35
        self.xi_clock    = 0.05
        self.eta_R       = 3.0
        self.eta_Q       = 1.5
        self.chi_dist    = 0.1
        self.omega_phase = 0.05
        self.zeta_timing = 0.01
        self.dp_w        = 0.02
        self.reward_scale = 0.05

        # ------------------------------------------------------------------
        # 6) Sensing model  (Eq 4)
        # ------------------------------------------------------------------
        self.sensing_noise = 1.0

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
        self.step_count         = 0
        self.orbit_phase_offset = 0.0
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
        self.last_rates       = {a: 0.0 for a in self.agents}
        self.last_sensing     = {a: 0.0 for a in self.agents}
        self.last_powers      = {a: 0.0 for a in self.agents}
        self.last_interference = {a: 0.0 for a in self.agents}
        self.last_distortion  = {a: 0.0 for a in self.agents}

        # ------------------------------------------------------------------
        # Step 1 — Certificate state variables
        #
        # These are lightweight EMA-based online fairness certificates.
        # They track per-agent cumulative fairness deficits and load.
        # They are NOT formal Lyapunov certificates — they implement the
        # structural substrate described in the proposal Section III.
        #
        # cert_comm_deficit[a]  : EMA of max(0, z^R_i - R̃_i)
        #                         → how often/how much agent a falls below
        #                           the rate consensus target
        # cert_sense_deficit[a] : EMA of max(0, z^Q_i - Q̃_i)
        #                         → sensing shortfall relative to consensus
        # cert_workload[a]      : EMA of P_i  (power/workload proxy)
        # cert_duty_share[a]    : fraction of steps where a was active critic
        #                         (rotating-prefect duty accounting)
        # ------------------------------------------------------------------
        self.cert_ema           = 0.05   # EMA coefficient for certificate update
        self.cert_comm_deficit  = {a: 0.0 for a in self.agents}
        self.cert_sense_deficit = {a: 0.0 for a in self.agents}
        self.cert_workload      = {a: 0.0 for a in self.agents}
        # cert_duty_share is kept in sync with critic_duty_count / total steps

        self.reset()

    # ======================================================================
    # reset()
    # ======================================================================
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents     = self.possible_agents[:]
        self.step_count = 0

        self.shadow_db = {
            a: float(np.random.normal(0.0, self.sigma_sh_db))
            for a in self.agents
        }

        self.h_fade  = {}
        self.channel = {}
        for a in self.agents:
            h_vec = (
                np.random.normal(0.0, 1/np.sqrt(2), size=self.channel_dim)
                + 1j * np.random.normal(0.0, 1/np.sqrt(2), size=self.channel_dim)
            ).astype(np.complex64)
            self.h_fade[a]  = h_vec
            self.channel[a] = h_vec.copy()

        self.energy      = {a: 1.0 for a in self.agents}
        self.prev_power  = {a: 0.0 for a in self.agents}
        self.interference = {a: 0.0 for a in self.agents}

        self.phase_noise = {
            a: float(np.random.uniform(0.01, 0.30)) for a in self.agents
        }
        self.timing_jitter_var = {
            a: float(np.random.uniform(1e-10, 1e-8)) for a in self.agents
        }
        self.pa_coeff = {
            a: float(np.random.uniform(0.01, 0.05)) for a in self.agents
        }
        self.clock_speed = {
            a: float(np.random.uniform(0.8, 1.2)) for a in self.agents
        }
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

        # Certificate state — reset each episode
        self.cert_comm_deficit  = {a: 0.0 for a in self.agents}
        self.cert_sense_deficit = {a: 0.0 for a in self.agents}
        self.cert_workload      = {a: 0.0 for a in self.agents}

        # Duty tracking is cumulative across episodes by design;
        # reset only if starting fresh (critic_selection_step == 0)
        if self.critic_selection_step == 0:
            self.critic_duty_count = {a: 0   for a in self.agents}
            self.critic_duty_share = {a: 0.0 for a in self.agents}

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
            zR_feat = float(np.tanh(self.z_R[a] / 20.0))
            zQ_feat = float(np.tanh(self.z_Q[a]))
            obs[a]  = np.array([
                float(np.tanh(h_mag)),
                float(self.energy[a]),
                float(self.phase_noise[a]),
                float(self.timing_jitter_var[a] * 1e9),
                float(self.pa_coeff[a]),
                float(self.clock_speed[a]),
                float(np.tanh(self.interference[a])),
                zR_feat,
                zQ_feat,
                float(self.positions[a][0] / self.region_size),
                float(self.positions[a][1] / self.region_size),
                float(t_norm),
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
        """
        Main environment transition.

        Proposal step alignment:
            (Steps 2 & 4 happen in train.py before this call)
            Step 3  Certified MM-ADMM refinement  → ADMM block below
            Step 1  Certificate update            → update_certificates() at end
            Step 5  Realized measurement/learn    → logging caches updated here
        """
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
        #    NOTE: actions arriving here should already be safety-projected
        #    (Step 4 via project_safe_actions / SafeActionProjector in train.py).
        #    The clip/norm below is a defensive fallback only.
        # ------------------------------------------------------------------
        for a in self.agents:
            act = np.asarray(actions[a], dtype=np.float32).ravel()
            if act.size < 3:
                raise ValueError(
                    f"{a}: need [P, w1, w2], got size {act.size}")

            p_cmd = float(np.clip(act[0], 0.0, 1.0))
            p     = float(np.clip(p_cmd, 0.0, self.max_power * self.energy[a]))

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
        # ------------------------------------------------------------------
        for a in self.agents:
            h_a = np.asarray(self.channel[a])

            phase_att  = np.exp(-self.phase_noise[a])
            timing_att = np.exp(-self.kappa_t * self.timing_jitter_var[a])

            desired_gain   = float(np.abs(np.vdot(h_a, beamformers[a])) ** 2)
            desired_signal = powers[a] * phase_att * timing_att * desired_gain

            p_dist[a] = float(self.pa_coeff[a] * (powers[a] ** 3))

            # No inter-UAV interference (model assumption).
            # Each agent's SINR depends only on its own signal, noise and PA distortion.
            I_a = 0.0
            interference[a]      = 0.0
            self.interference[a] = 0.0

            sinr_a   = desired_signal / (self.noise + I_a + p_dist[a] + 1e-15)
            rates[a] = float(self.bandwidth * np.log2(1.0 + sinr_a))

            sensing[a] = float(
                powers[a] * (self.sensing_gain[a] ** 2) / (self.sensing_noise + 1e-15)
            )

        # ------------------------------------------------------------------
        # 7) Step 3 — Certified MM-ADMM Refinement  (Section V, Eqs 26-28)
        #
        #    Proposal role: This is the certified MM-ADMM refinement substrate.
        #    It implements the structural consensus and dual-variable update
        #    from the proposal. It does NOT prove formal convergence guarantees
        #    within a single step; it advances the ADMM iterate toward the
        #    max-min fair consensus point.
        #
        #    Coordinator note: The active critic host (self.active_critic_agent)
        #    is the logical coordinator of this consensus step. In a real
        #    distributed system, self.active_critic_agent would aggregate
        #    R_i/Q_i from all followers and broadcast z. In simulation,
        #    the aggregation is centralised in this method for tractability.
        # ------------------------------------------------------------------
        lambda_R_old = self.lambda_R.copy()
        lambda_Q_old = self.lambda_Q.copy()
        rho = float(self.rho)

        rates_norm = {a: rates[a] / self.bandwidth for a in self.agents}

        max_Q = max(max(sensing.values()), 1e-8)
        sensing_norm = {a: sensing[a] / max_Q for a in self.agents}

        z_R_prev = float(np.mean([self.z_R[a] for a in self.agents]))
        z_Q_prev = float(np.mean([self.z_Q[a] for a in self.agents]))

        # Min-biased consensus (max-min fairness, Eq 6; γ=0.5 blend)
        mean_R = float(np.mean([rates_norm[a]   for a in self.agents]))
        mean_Q = float(np.mean([sensing_norm[a] for a in self.agents]))
        min_R  = float(np.min( [rates_norm[a]   for a in self.agents]))
        min_Q  = float(np.min( [sensing_norm[a] for a in self.agents]))
        gamma  = 0.5
        z_R_raw = (1 - gamma) * mean_R + gamma * min_R
        z_R_global = max(0.01, z_R_raw)
        z_Q_global = (1 - gamma) * mean_Q + gamma * min_Q
        for a in self.agents:
            self.z_R[a] = z_R_global
            self.z_Q[a] = z_Q_global

        # Dual update  (Eqs 27-28)
        for a in self.agents:
            self.lambda_R[a] = float(np.clip(
                lambda_R_old[a] + rho * (rates_norm[a] - self.z_R[a]),
                -self.LAMBDA_MAX_R, self.LAMBDA_MAX_R))

            self.lambda_Q[a] = float(np.clip(
                lambda_Q_old[a] + rho * (sensing_norm[a] - self.z_Q[a]),
                -self.LAMBDA_MAX_Q, self.LAMBDA_MAX_Q))

        # Residuals
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

        self.r_ema = (1 - self.res_ema) * self.r_ema + self.res_ema * primal_res
        self.s_ema = (1 - self.res_ema) * self.s_ema + self.res_ema * dual_res
        self.last_primal_residual = primal_res
        self.last_dual_residual   = dual_res
        self.last_z_R  = z_R_global
        self.last_z_Q  = z_Q_global
        self.last_jain = self._jain_index(rates)

        # ------------------------------------------------------------------
        # 8) Reward  (Eq 32)
        # ------------------------------------------------------------------
        for a in self.agents:
            R_tilde  = rates[a] / self.bandwidth
            zR_tilde = self.z_R[a]
            Q_raw    = sensing[a]
            Q_norm   = sensing_norm[a]
            zQ_i     = self.z_Q[a]

            fair_R     = self.eta_R * (R_tilde - zR_tilde) ** 2
            fair_Q     = self.eta_Q * (Q_norm  - zQ_i)    ** 2
            jain_bonus = 2.0 * self.last_jain

            r_i = (
                self.nu[a] * (self.alpha_comm * R_tilde + self.beta_sense * Q_raw)
                - self.mu_power    * powers[a]
                - self.xi_clock    * self.clock_speed[a]
                - fair_R
                - fair_Q
                - self.chi_dist    * p_dist[a]
                - self.omega_phase * self.phase_noise[a]
                - self.zeta_timing * self.timing_jitter_var[a]
                + jain_bonus
            )
            r_i -= self.dp_w * (dps[a] ** 2)
            rewards[a] = float(self.reward_scale * r_i)

        # ------------------------------------------------------------------
        # 9) Termination
        # ------------------------------------------------------------------
        self.step_count += 1
        done = (self.step_count >= self.horizon
                or any(self.energy[a] <= 0.0 for a in self.agents))

        # Step 5 — Realized measurement caches (used for logging in train.py)
        self.last_rates        = dict(rates)
        self.last_sensing      = dict(sensing)
        self.last_powers       = dict(powers)
        self.last_interference = dict(interference)
        self.last_distortion   = dict(p_dist)

        # Step 1 — Certificate update from realized outcomes
        self.update_certificates(rates_norm, sensing_norm, powers)

        return (
            self._get_obs(),
            rewards,
            {a: done for a in self.agents},
            {a: False for a in self.agents},
            {}
        )

    # ======================================================================
    # Step 1 — Certificate Update
    # ======================================================================
    def update_certificates(
        self,
        rates_norm:   dict,
        sensing_norm: dict,
        powers:       dict,
    ):
        """
        Step 1: Update lightweight certificate-style fairness memory.

        Proposal role: These EMA-based certificate variables serve as
        persistent fairness memory across steps. They are NOT formal
        Lyapunov / barrier certificates; they implement the structural
        substrate of the certificate-bearing fairness layer.

        Updates per agent:
            cert_comm_deficit[a]  ← EMA(max(0, z^R - R̃_i))
            cert_sense_deficit[a] ← EMA(max(0, z^Q - Q̃_i))
            cert_workload[a]      ← EMA(P_i)
            cert_duty_share[a]    ← critic_duty_count[a] / max(total_steps, 1)
        """
        alpha = self.cert_ema
        for a in self.agents:
            # Communication deficit: how far below rate consensus target
            comm_def = max(0.0, self.z_R[a] - rates_norm.get(a, 0.0))
            self.cert_comm_deficit[a] = (
                (1 - alpha) * self.cert_comm_deficit[a] + alpha * comm_def
            )

            # Sensing deficit: how far below sensing consensus target
            sense_def = max(0.0, self.z_Q[a] - sensing_norm.get(a, 0.0))
            self.cert_sense_deficit[a] = (
                (1 - alpha) * self.cert_sense_deficit[a] + alpha * sense_def
            )

            # Workload proxy: smoothed power consumption
            self.cert_workload[a] = (
                (1 - alpha) * self.cert_workload[a]
                + alpha * float(powers.get(a, 0.0))
            )

        # Duty share: fraction of all selection steps where each agent hosted
        total = max(self.critic_selection_step, 1)
        for a in self.agents:
            self.critic_duty_share[a] = self.critic_duty_count.get(a, 0) / total

    # ======================================================================
    # Distributed Critic Selection
    # ======================================================================
    def compute_weighted_critic_scores(self) -> dict:
        """
        Compute per-agent weighted scores Q_i(t)  (PDF Eq. 2).

        Practical mapping to codebase quantities:
            Ẽ_i  ← self.energy[a]               (residual energy, ∈ [0,1])
            L̃_i  ← 1 - critic_duty_share[a]     (low duty = low load)
            C̃_i  ← self.clock_speed[a]          (capability, normed)
            S̃_i  ← ‖self.channel[a]‖            (channel magnitude, normed)
            P̃_i  ← proximity to swarm centroid  (normed)
            W̃_i  ← willingness (energy + duty penalties)

        Returns {agent: Q_i_score} and stores to self.last_weighted_scores.
        """
        eps  = 1e-8
        N    = len(self.agents)
        if N == 0:
            return {}

        # Raw component values
        E_arr = np.array([self.energy.get(a, 1.0)         for a in self.agents])
        L_arr = np.array([1.0 - self.critic_duty_share.get(a, 0.0) for a in self.agents])
        C_arr = np.array([self.clock_speed.get(a, 1.0)    for a in self.agents])
        S_arr = np.array([float(np.linalg.norm(self.channel.get(a, np.ones(self.channel_dim))))
                          for a in self.agents])

        # Proximity: distance from swarm centroid (closer = higher P̃)
        centroid  = np.mean(
            np.stack([self.positions[a] for a in self.agents]), axis=0
        )
        dist_arr  = np.array([
            float(np.linalg.norm(self.positions[a] - centroid))
            for a in self.agents
        ])
        max_dist  = float(dist_arr.max()) + eps
        P_arr     = 1.0 - dist_arr / max_dist   # ∈ [0,1]

        # Clip energy and low-load
        E_arr = np.clip(E_arr, 0.0, 1.0)
        L_arr = np.clip(L_arr, 0.0, 1.0)

        # Min-max normalise C, S within swarm
        def _norm01(v):
            lo, hi = v.min(), v.max()
            return (v - lo) / (hi - lo + eps)

        C_arr = _norm01(C_arr)
        S_arr = _norm01(S_arr)

        # Willingness W̃_i  (reduced for depleted or over-burdened agents)
        # Default alpha weights — can be overridden by setting env attributes
        alpha_E = getattr(self, "_cfg_alpha_E", 0.25)
        alpha_L = getattr(self, "_cfg_alpha_L", 0.20)
        alpha_C = getattr(self, "_cfg_alpha_C", 0.15)
        alpha_S = getattr(self, "_cfg_alpha_S", 0.20)
        alpha_P = getattr(self, "_cfg_alpha_P", 0.10)
        alpha_W = getattr(self, "_cfg_alpha_W", 0.10)
        alpha_F = getattr(self, "_cfg_alpha_F", 0.15)  # fairness-alignment
        e_thr   = getattr(self, "_cfg_energy_thr", 0.10)
        d_thr   = getattr(self, "_cfg_duty_thr",   0.25)

        W_arr = np.ones(N)
        for i, a in enumerate(self.agents):
            en = self.energy.get(a, 1.0)
            ds = self.critic_duty_share.get(a, 0.0)
            if en < e_thr:
                W_arr[i] *= max(0.0, en / (e_thr + eps))
            if ds > d_thr:
                excess = (ds - d_thr) / (1.0 - d_thr + eps)
                W_arr[i] *= max(0.0, 1.0 - excess)
        W_arr = np.clip(W_arr, 0.0, 1.0)

        # F̃_i: fairness-alignment — prefer currently underserved agents
        # Agents with rate below z^R represent the fairness bottleneck;
        # hosting the critic from their viewpoint teaches the value function
        # that unfair allocation is costly, driving policy toward equalisation.
        if alpha_F > 0.0 and hasattr(self, "last_rates") and hasattr(self, "z_R"):
            rates_bw = np.array(
                [self.last_rates.get(a, 0.0) / (self.bandwidth + eps)
                 for a in self.agents]
            )
            zR_arr = np.array([self.z_R.get(a, 0.0) for a in self.agents])
            underservice = np.maximum(0.0, zR_arr - rates_bw)
            F_arr = _norm01(underservice)
        else:
            F_arr = np.zeros(N)

        Q_arr = (alpha_E * E_arr + alpha_L * L_arr + alpha_C * C_arr
               + alpha_S * S_arr + alpha_P * P_arr + alpha_W * W_arr
               + alpha_F * F_arr)

        scores = {a: float(Q_arr[i]) for i, a in enumerate(self.agents)}
        self.last_weighted_scores = scores

        # Store components for diagnostics
        self.last_weighted_components = {
            a: {
                "E": float(E_arr[i]), "L": float(L_arr[i]),
                "C": float(C_arr[i]), "S": float(S_arr[i]),
                "P": float(P_arr[i]), "W": float(W_arr[i]),
                "Q": float(Q_arr[i]),
            }
            for i, a in enumerate(self.agents)
        }
        return scores

    def select_active_critic(self, update_step: int = None,
                             global_step: int = None) -> str:
        """
        Select and update self.active_critic_agent based on critic_mode.

        Modes:
            'fixed'       → always returns self.default_leader_uav   (PDF §2 static)
            'round_robin' → cycles through agents by update_step mod N (PDF §2)
            'weighted'    → argmax Q_i(t) with threshold guard         (PDF §3)

        Also:
            - increments self.critic_selection_step
            - increments duty counter for selected agent
            - updates self.critic_duty_share for all agents

        Returns: name of selected critic host agent.
        """
        # Accept either positional update_step or keyword global_step (train.py compat)
        if global_step is not None and update_step is None:
            update_step = global_step
        if update_step is None:
            update_step = self.critic_selection_step

        N = len(self.agents)
        if N == 0:
            return self.default_leader_uav

        mode = self.critic_mode

        if mode == "fixed":
            selected = self.default_leader_uav

        elif mode == "round_robin":
            # PDF §2, Eq.(1):  i*(t) = t mod N
            idx      = update_step % N
            selected = self.agents[idx]

        elif mode == "weighted":
            # PDF §3, Eqs.(2-3)
            scores = self.compute_weighted_critic_scores()
            e_thr  = getattr(self, "_cfg_energy_thr", 0.10)

            eligible = [a for a in self.agents
                        if self.energy.get(a, 1.0) >= e_thr]

            if not eligible:
                # Fallback: previous active critic, then default
                selected = (self.active_critic_agent
                            if self.active_critic_agent in self.agents
                            else self.default_leader_uav)
            else:
                selected = max(eligible, key=lambda a: scores.get(a, 0.0))

        else:
            # Unknown mode: fall back to fixed
            selected = self.default_leader_uav

        # Update active host and duty tracking
        self.active_critic_agent = selected
        self.critic_duty_count[selected] = self.critic_duty_count.get(selected, 0) + 1
        self.critic_selection_step += 1

        # Refresh duty shares
        total = max(self.critic_selection_step, 1)
        for a in self.agents:
            self.critic_duty_share[a] = self.critic_duty_count.get(a, 0) / total

        return selected

    def configure_fairness_weights(
        self,
        eta_R: float = 3.0,
        eta_Q: float = 1.5,
    ):
        """
        Override ADMM fairness penalty weights after construction.

        Cert variants benefit from tighter consensus enforcement because
        the cert observation features give actors better information to
        act on the stronger constraint without collapsing.

        eta_R : ADMM rate-consensus penalty (default 3.0)
        eta_Q : ADMM sensing-consensus penalty (default 1.5)
        """
        self.eta_R = float(eta_R)
        self.eta_Q = float(eta_Q)

    def configure_weighted_selector(
        self,
        alpha_E: float = 0.25,
        alpha_L: float = 0.20,
        alpha_C: float = 0.15,
        alpha_S: float = 0.20,
        alpha_P: float = 0.10,
        alpha_W: float = 0.10,
        alpha_F: float = 0.15,   # fairness-alignment: prefer underserved agents
        energy_threshold: float = 0.10,
        duty_threshold:   float = 0.25,  # was 0.50 — now fires for N≥4 (1/N≈0.20)
    ):
        """
        Set alpha weights and thresholds for weighted critic selection.
        Call this after constructing the env and before training starts.

        alpha_F : weight for fairness-alignment component F̃_i.
                  Agents currently BELOW the consensus rate target z^R score
                  higher — rotating to underserved agents focuses value
                  estimation on the cost of unfair allocation.
        duty_threshold : lowered from 0.50 → 0.25 so willingness penalty
                  fires for typical N≥4 swarms (expected share = 1/N ≈ 0.20).
        """
        self._cfg_alpha_E    = alpha_E
        self._cfg_alpha_L    = alpha_L
        self._cfg_alpha_C    = alpha_C
        self._cfg_alpha_S    = alpha_S
        self._cfg_alpha_P    = alpha_P
        self._cfg_alpha_W    = alpha_W
        self._cfg_alpha_F    = alpha_F
        self._cfg_energy_thr = energy_threshold
        self._cfg_duty_thr   = duty_threshold

    # ======================================================================
    # Step 4 — Barrier/QP Safe Action Projection
    # ======================================================================
    def project_safe_actions(
        self,
        actions: dict,
        noise_std: float = 0.0,
    ) -> dict:
        """
        Step 4: Project actions to safe feasible set (lightweight approximation).

        Proposal role: Structural substrate of the barrier/QP safe execution layer.
        Enforces hard feasibility bounds on power and beamformer normalization.
        This function is called in train.py between actor warm-start (Step 2)
        and env.step() (Step 3).

        ──────────────────────────────────────────────────────────────────
        EXTENSION POINT — replace body with a true CBF-QP solver:

            from osqp import OSQP  # or cvxpy
            # Solve:  min_a ‖a - a_nominal‖²
            #   s.t.  ∂h/∂x · f(x,a) + α(h(x)) ≥ 0   (control barrier fn)
            #         0 ≤ P_i ≤ P_max · E_i
            #         ‖w_i‖ = 1
        ──────────────────────────────────────────────────────────────────
        """
        safe = {}
        for a in self.agents:
            if a not in actions:
                continue
            act = np.asarray(actions[a], dtype=np.float32).copy()

            # Power: clip to [0, P_max · E_i]
            p_eff  = self.max_power * max(float(self.energy.get(a, 1.0)), 0.0)
            act[0] = float(np.clip(act[0], 0.0, p_eff))

            # Beamformer: clip then unit-normalise
            if act.size > 1:
                w = np.clip(act[1:], -1.0, 1.0)
                n = float(np.linalg.norm(w))
                act[1:] = w / n if n >= 1e-8 else np.array([1.0, 0.0], dtype=np.float32)

            safe[a] = act
        return safe

    # ======================================================================
    # Helpers
    # ======================================================================
    def _jain_index(self, rates: dict) -> float:
        vals = np.array(list(rates.values()), dtype=np.float64)
        if vals.size == 0 or np.sum(vals**2) < 1e-15:
            return 1.0
        return float(np.sum(vals)**2 / (len(vals) * np.sum(vals**2)))

    def get_certificate_summary(self) -> dict:
        """
        Return a summary dict of current certificate states for logging.

        Proposal Step 1 / Step 5 — used by train.py diagnostics.
        """
        return {
            "cert_comm_deficit_mean":  float(np.mean(list(self.cert_comm_deficit.values()))),
            "cert_sense_deficit_mean": float(np.mean(list(self.cert_sense_deficit.values()))),
            "cert_workload_mean":      float(np.mean(list(self.cert_workload.values()))),
            "active_critic":           self.active_critic_agent,
            "duty_share":              dict(self.critic_duty_share),
        }

    def scenario_string(self) -> str:
        rx_x, rx_y = float(self.rx_pos[0]), float(self.rx_pos[1])
        return (
            f"N={len(self.agents)} | {self.motion_case} | {self.channel_model} | "
            f"critic_mode={self.critic_mode} | "
            f"active_critic={self.active_critic_agent} | "
            f"pn_max={max(self.phase_noise.values()):.3f} | "
            f"rho={self.rho:.4f} | rx=({rx_x:.0f},{rx_y:.0f})"
        )
