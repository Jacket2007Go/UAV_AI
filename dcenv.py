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
        # Review §5.1-5.2 — physically calibrated link budget with correct
        # units. P_max = 1 W (30 dBm); PL0 = 1e-4 is free-space path loss at
        # d0 = 1 m for fc ~ 2.4 GHz; thermal-noise PSD N0 = kT*NF
        # (-174 dBm/Hz + 7 dB noise figure ~ 2.0e-20 W/Hz). The SINR
        # denominator uses noise POWER N0*B per the revised formulation.
        #
        # NOTE (review §4.3): the legacy code used a flat noise power of
        # 1e-3 W — roughly 11 orders of magnitude above thermal — which
        # compressed every link to ~1e-7 bps/Hz and produced the
        # "1e-8 bps/MHz" logging artifact the review flagged. With the
        # calibrated budget, spectral efficiency lands in a physically
        # sensible 0-20 bps/Hz range and rates are reported in true bps.
        self.max_power   = 1.0                       # W (30 dBm)
        self.N0          = 2.0e-20                   # W/Hz thermal PSD (incl. NF)
        self.bandwidth   = 1e6                       # Hz
        self.noise       = self.N0 * self.bandwidth  # noise POWER N0*B (W)
        self.channel_dim = 2

        # Normalized rate utility reference (review §4.3 / §5.7):
        #   Rbar_i = eta_i / eta_ref,  eta_i = R_i/B in bps/Hz
        # Rbar is dimensionless and used inside the reward and ADMM
        # consensus so that no dimensionful quantity is mislabeled.
        self.eta_ref     = 20.0                      # bps/Hz reference

        # ------------------------------------------------------------------
        # 2) Observation and action spaces
        #
        #   obs_i = [tanh(|h_i|), E_i, σ²_φ,i, σ²_t,i·1e9, a3_i, c_i,
        #            tanh(I_i), tanh(z^R_i), tanh(z^S_i),
        #            x_i/L, y_i/L, t/T]           dim=12
        #   (z^R is the consensus target on the NORMALIZED rate Rbar_i,
        #    so it lives in ~[0,1] and needs no /20 squash — review §5.7)
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
        # 6) Sensing model — Fisher-information based (review §4.4 / §5.3)
        #
        # Sensing observation: y_i(t) = mu_i(theta, t) + n_i(t),
        # theta = [tau, nu, phi]^T (delay, Doppler, angle).
        # Local FIM:  J_i(t) = (2/sigma_s^2) Re{ (dmu/dtheta)^H (dmu/dtheta) }
        # implemented with dimensionless per-parameter sensitivities so the
        # diagonal entries are mutually comparable:
        #   g_tau : normalized RMS-bandwidth term        (delay info)
        #   g_nu  : normalized RMS-duration term         (Doppler info)
        #   g_phi : geometry-dependent aperture term     (angle info;
        #           degrades with range as (d_ref/d)^2)
        # Off-diagonals are zero under the orthogonal-waveform assumption.
        #
        # Scalar sensing quality S_i(t)  (self.sensing_metric):
        #   'fim_min_eig' : S_i = lambda_min(J_i(t))        [review Eq.]
        #   'crb_trace'   : S_i = 1 / (tr(J_i^-1) + eps)    [review alt.]
        #   'snr'         : legacy proxy P_i |g_i^s|^2 / sigma_s^2
        # The CRB condition tr(J_i^-1) <= eps_CRB is logged per agent.
        # ------------------------------------------------------------------
        self.sensing_noise  = 1.0       # sigma_s^2
        self.sensing_metric = "fim_min_eig"
        self.fim_g_tau      = 1.00      # normalized delay sensitivity
        self.fim_g_nu       = 0.60      # normalized Doppler sensitivity
        self.fim_d_ref      = 300.0     # reference range for angle term (m)
        self.crb_eps        = 1e-9
        self.last_sensing_snr = {a: 0.0 for a in self.agents}
        self.last_fim_diag    = {a: np.zeros(3) for a in self.agents}
        self.last_crb_trace   = {a: 0.0 for a in self.agents}

        # ------------------------------------------------------------------
        # 6b) Energy model  (review §4.5)
        #   E_i^total = E_tx + E_rx + E_move + E_comp + E_sync
        # E_tx/E_rx/E_move are charged inside step(). E_comp (critic
        # hosting, kappa_i C_i f_i^2) and E_sync (model migration,
        # P_tx |psi| / R_{i->j}) are charged by the trainer through
        # charge_critic_compute() / charge_critic_sync().
        # Energies are in normalized battery units (1.0 = full battery,
        # ~100 J equivalent at the legacy P/100 transmit depletion rate).
        # ------------------------------------------------------------------
        self.e_tx_scale        = 1.0 / 100.0  # E_tx = P_i * Ts / 100 (legacy rate)
        self.e_rx_const        = 2e-4          # per-step receive-chain energy
        self.e_move_const      = 1e-4          # per-step propulsion proxy
        self.kappa_comp        = 5e-4          # kappa_i switched-capacitance coeff
        self.cycles_per_update = 1.0           # C_i(t) cycles per critic update (norm.)
        self.psi_bits          = 3.2e7         # |psi| critic size in bits (trainer overrides)
        self.sync_energy_cap   = 0.02          # cap per migration event
        self.battery_joules    = 100.0         # J per 1.0 normalized battery unit
        self.energy_used = {a: {"tx": 0.0, "rx": 0.0, "move": 0.0,
                                "comp": 0.0, "sync": 0.0} for a in self.agents}
        self.last_comp_energy = 0.0
        self.last_sync_energy = 0.0

        # Joint fairness weights (review §5.9): J_JSC = wR*J_R + wS*J_S
        self.omega_R_jain = 0.5
        self.omega_S_jain = 0.5

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
        self.last_sinr_eff      = {a: 0.0 for a in self.agents}  # review §5.2
        self.last_spectral_eff  = {a: 0.0 for a in self.agents}  # eta_i, bps/Hz
        self.last_jain_S   = 1.0   # sensing Jain J_S        (review §5.9)
        self.last_jain_JSC = 1.0   # joint Jain J_JSC        (review §5.9)
        self.last_min_rate    = 0.0   # min_i R_i(t), bps    (review §6.3)
        self.last_min_sensing = 0.0   # min_i S_i(t)         (review §6.3)

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
        self.last_sinr_eff     = {a: 0.0 for a in self.agents}
        self.last_spectral_eff = {a: 0.0 for a in self.agents}
        self.last_sensing_snr  = {a: 0.0 for a in self.agents}
        self.last_fim_diag     = {a: np.zeros(3) for a in self.agents}
        self.last_crb_trace    = {a: 0.0 for a in self.agents}
        self.last_jain_S       = 1.0
        self.last_jain_JSC     = 1.0
        self.last_min_rate     = 0.0
        self.last_min_sensing  = 0.0
        # Energy ledger (review §4.5) is cumulative ACROSS episodes by
        # design — host-imbalance I_E is a whole-run statistic. It is
        # only zeroed on a fresh run (no critic selections yet).
        if self.critic_selection_step == 0:
            self.energy_used = {a: {"tx": 0.0, "rx": 0.0, "move": 0.0,
                                    "comp": 0.0, "sync": 0.0}
                                for a in self.agents}

        return self._get_obs(), {}

    # ======================================================================
    # _get_obs()
    # ======================================================================
    def _get_obs(self):
        obs    = {}
        t_norm = self.step_count / max(self.horizon, 1)
        for a in self.agents:
            h_mag   = float(np.linalg.norm(self.channel[a]))
            # z_R is now a consensus target on the NORMALIZED rate
            # Rbar = eta/eta_ref (review §5.7) and lives in ~[0,1], so the
            # legacy /20 squash (sized for raw bps/MHz) is removed.
            zR_feat = float(np.tanh(self.z_R[a]))
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
        # 5) Energy depletion — explicit ledger (review §4.5)
        #    E_i^total = E_tx + E_rx + E_move (+ E_comp + E_sync from trainer)
        # ------------------------------------------------------------------
        for a in self.agents:
            e_tx   = powers[a] * self.e_tx_scale
            e_rx   = self.e_rx_const
            e_move = self.e_move_const   # ring orbit: every agent propels
            self.energy_used[a]["tx"]   += e_tx
            self.energy_used[a]["rx"]   += e_rx
            self.energy_used[a]["move"] += e_move
            self.energy[a] = max(self.energy[a] - (e_tx + e_rx + e_move), 0.0)

        # ------------------------------------------------------------------
        # 6) Impairment-aware SINR, rates, and sensing  (Eqs 1-4, 11, 14, 15)
        # ------------------------------------------------------------------
        for a in self.agents:
            h_a = np.asarray(self.channel[a])

            # ── Review §5.2 — impairment-aware EFFECTIVE SINR ─────────────
            #  SINR_i^eff = P_i e^{-sigma_phi^2} e^{-kappa_t sigma_t^2}
            #               |h_i^T w_i|^2 / (N0*B + I_i + P_dist,i)
            phase_att  = np.exp(-self.phase_noise[a])              # e^{-s_phi^2}
            timing_att = np.exp(-self.kappa_t * self.timing_jitter_var[a])

            desired_gain   = float(np.abs(np.vdot(h_a, beamformers[a])) ** 2)
            desired_signal = powers[a] * phase_att * timing_att * desired_gain

            # PA nonlinearity y = a1 x + a3 |x|^2 x:
            #   transmit-referred distortion power (kept for the reward
            #   chi_dist term, legacy semantics) ...
            p_dist[a] = float(self.pa_coeff[a] * (powers[a] ** 3))
            #   ... but the distortion that lands in the SINR denominator
            #   propagates through the SAME channel as the signal, so it is
            #   receive-referred here. This gives the physically correct
            #   distortion-limited (EVM-floor) behaviour at high SNR.
            p_dist_rx = p_dist[a] * desired_gain

            # No inter-UAV interference (model assumption).
            I_a = 0.0
            interference[a]      = 0.0
            self.interference[a] = 0.0

            sinr_eff = desired_signal / (self.noise + I_a + p_dist_rx + 1e-30)
            self.last_sinr_eff[a] = float(sinr_eff)

            # ── Review §5.1 — correct units ───────────────────────────────
            #   eta_i(t) = log2(1 + SINR_eff)   [bps/Hz, spectral efficiency]
            #   R_i(t)   = B * eta_i(t)         [bps,    true rate]
            eta_a    = float(np.log2(1.0 + sinr_eff))
            rates[a] = float(self.bandwidth * eta_a)
            self.last_spectral_eff[a] = eta_a

            # ── Review §5.3 — FIM-based sensing quality S_i(t) ────────────
            sensing[a] = self._sensing_quality(a, powers[a])

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

        # Review §5.7: consensus operates on the NORMALIZED rate utility
        # Rbar_i = eta_i / eta_ref  (dimensionless, ~[0,1]) so that the
        # rate and sensing consensus terms are unit-balanced.
        rates_norm = {a: rates[a] / (self.bandwidth * self.eta_ref)
                      for a in self.agents}

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
        # Review §5.9 — fairness on BOTH services + joint index:
        #   J_R(t), J_S(t),  J_JSC = wR*J_R + wS*J_S  (wR + wS = 1)
        self.last_jain     = self._jain_index(rates)      # J_R
        self.last_jain_S   = self._jain_index(sensing)    # J_S
        self.last_jain_JSC = float(self.omega_R_jain * self.last_jain
                                   + self.omega_S_jain * self.last_jain_S)
        # Review §6.3 — minimum-service metrics
        self.last_min_rate    = float(min(rates.values()))    # min_i R_i (bps)
        self.last_min_sensing = float(min(sensing.values()))  # min_i S_i

        # ------------------------------------------------------------------
        # 8) Reward  (Eq 32)
        # ------------------------------------------------------------------
        for a in self.agents:
            # Review §5.7 — unit-balanced reward:
            #   r_i = aR*Rbar_i + aS*Sbar_i - aP*P_i - aC*p_i^cert
            #         - aD*P_dist,i - eta_R(Rbar_i - z_R)^2
            #         - eta_S(Sbar_i - z_S)^2  (+ impairment/clock terms)
            # Rbar and Sbar are BOTH dimensionless and O(1), removing the
            # unit imbalance the review flagged (legacy code mixed raw
            # sensing units with normalized rate inside the utility).
            # The certificate term -aC*p_i^cert is applied in the certenv
            # subclass; here p_i^cert = 0.
            R_bar    = rates_norm[a]          # eta_i/eta_ref     in ~[0,1]
            S_bar    = sensing_norm[a]        # swarm-normalized  in  [0,1]
            zR_i     = self.z_R[a]
            zS_i     = self.z_Q[a]

            fair_R     = self.eta_R * (R_bar - zR_i) ** 2
            fair_S     = self.eta_Q * (S_bar - zS_i) ** 2
            jain_bonus = 2.0 * self.last_jain

            r_i = (
                self.nu[a] * (self.alpha_comm * R_bar + self.beta_sense * S_bar)
                - self.mu_power    * powers[a]
                - self.xi_clock    * self.clock_speed[a]
                - fair_R
                - fair_S
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
    # Review §5.3 — Fisher-information sensing quality
    # ======================================================================
    def _sensing_quality(self, a: str, power: float) -> float:
        """
        Compute scalar sensing quality S_i(t) from the local FIM.

        J_i(t) = 2*SNR_s,i * diag(g_tau, g_nu, g_phi)   (orthogonal-waveform
        assumption => zero off-diagonals), with SNR_s,i = P_i|g_i^s|^2/sigma_s^2
        and a geometry-dependent angle sensitivity g_phi = (d_ref/d_i)^2
        (clipped) — angular information degrades with range.

        Returns (per self.sensing_metric):
          'fim_min_eig' : lambda_min(J_i)          — review S_i = λmin(J_i)
          'crb_trace'   : 1/(tr(J_i^-1) + eps)     — review CRB alternative
          'snr'         : legacy SNR proxy
        Also caches last_sensing_snr / last_fim_diag / last_crb_trace
        so the CRB condition tr(J^-1) <= eps_CRB can be reported.
        """
        snr_s = float(power * (self.sensing_gain[a] ** 2)
                      / (self.sensing_noise + 1e-15))
        self.last_sensing_snr[a] = snr_s

        d     = max(float(self.distance.get(a, self.fim_d_ref)), 1.0)
        g_phi = float(np.clip((self.fim_d_ref / d) ** 2, 0.05, 1.0))

        diag = 2.0 * snr_s * np.array(
            [self.fim_g_tau, self.fim_g_nu, g_phi], dtype=np.float64)
        self.last_fim_diag[a] = diag
        crb_tr = float(np.sum(1.0 / (diag + 1e-30)))
        self.last_crb_trace[a] = crb_tr

        if self.sensing_metric == "snr":
            return snr_s
        if self.sensing_metric == "crb_trace":
            return float(1.0 / (crb_tr + self.crb_eps))
        return float(diag.min())          # 'fim_min_eig' (default)

    # ======================================================================
    # Review §4.5 — critic-hosting and migration energy
    # ======================================================================
    def charge_critic_compute(self, host: str = None) -> float:
        """
        Charge E_i^comp = kappa_i * C_i(t) * f_i^2 to the active critic host
        for one critic update (review §4.5). f_i is the host's normalized
        clock frequency; kappa_i its switched-capacitance coefficient.
        """
        a = host if host is not None else self.active_critic_agent
        if a not in self.agents:
            return 0.0
        f_i = float(self.clock_speed.get(a, 1.0))
        e   = self.kappa_comp * self.cycles_per_update * (f_i ** 2)
        self.energy_used[a]["comp"] += e
        self.energy[a] = max(self.energy[a] - e, 0.0)
        self.last_comp_energy = e
        return e

    def charge_critic_sync(self, old_host: str, new_host: str) -> float:
        """
        Charge E_{i->j}^sync = P_i^tx * |psi| / R_{i->j}(t) to the OLD host
        at a critic migration event (review §4.5). |psi| is the critic model
        size in bits; the inter-host link rate is proxied by the weaker of
        the two hosts' last realized rates. Converted from joules to
        normalized battery units via self.battery_joules and capped.
        """
        if (old_host == new_host or old_host not in self.agents
                or new_host not in self.agents):
            self.last_sync_energy = 0.0
            return 0.0
        r_link = max(min(self.last_rates.get(old_host, 0.0),
                         self.last_rates.get(new_host, 0.0)), 1e3)   # bps
        p_tx   = float(self.last_powers.get(old_host, 0.0)) or 0.5   # W
        e_J    = p_tx * self.psi_bits / r_link                       # joules
        e      = min(e_J / self.battery_joules, self.sync_energy_cap)
        self.energy_used[old_host]["sync"] += e
        self.energy[old_host] = max(self.energy[old_host] - e, 0.0)
        self.last_sync_energy = e
        return e

    def energy_imbalance(self) -> float:
        """
        Review §6.3 energy-imbalance metric:
          I_E = sqrt((1/N) * sum_i (E_i^used - E_avg^used)^2) / (E_avg^used + eps)
        computed over the cumulative per-agent total energy ledger.
        """
        used = np.array([sum(self.energy_used[a].values())
                         for a in self.agents], dtype=np.float64)
        if used.size == 0:
            return 0.0
        mean = float(used.mean())
        return float(np.sqrt(np.mean((used - mean) ** 2)) / (mean + 1e-12))

    def total_comp_energy(self) -> float:
        return float(sum(self.energy_used[a]["comp"] for a in self.agents))

    def total_sync_energy(self) -> float:
        return float(sum(self.energy_used[a]["sync"] for a in self.agents))

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
        Compute per-agent critic-host preference scores H_i(t).

        Review §5.8: the host score is named H_i(t) — NOT Q_i(t) — to avoid
        the notation collision with sensing quality Q_i/S_i and the critic
        network Q_psi:
            H_i(t) = aE*E~ + aL*L~ + aC*C~ + aS*S~ + aP*P~ + aW*W~ (+ aF*F~)

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
                [self.last_rates.get(a, 0.0)
                 / (self.bandwidth * self.eta_ref + eps)
                 for a in self.agents]
            )   # Rbar units — same scale as z_R (review §5.7)
            zR_arr = np.array([self.z_R.get(a, 0.0) for a in self.agents])
            underservice = np.maximum(0.0, zR_arr - rates_bw)
            F_arr = _norm01(underservice)
        else:
            F_arr = np.zeros(N)

        # H_i(t) — weighted-preference host score (review §5.8)
        H_arr = (alpha_E * E_arr + alpha_L * L_arr + alpha_C * C_arr
               + alpha_S * S_arr + alpha_P * P_arr + alpha_W * W_arr
               + alpha_F * F_arr)

        scores = {a: float(H_arr[i]) for i, a in enumerate(self.agents)}
        self.last_weighted_scores = scores

        # Store components for diagnostics ('H' is canonical; 'Q' retained
        # as a deprecated alias so existing plotting scripts don't break)
        self.last_weighted_components = {
            a: {
                "E": float(E_arr[i]), "L": float(L_arr[i]),
                "C": float(C_arr[i]), "S": float(S_arr[i]),
                "P": float(P_arr[i]), "W": float(W_arr[i]),
                "H": float(H_arr[i]), "Q": float(H_arr[i]),
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
            # i*(t) = argmax_{i in E_set(t)} H_i(t)   (review §5.8)
            scores = self.compute_weighted_critic_scores()
            e_thr  = getattr(self, "_cfg_energy_thr",   0.10)
            l_max  = getattr(self, "_cfg_load_max",     1.00)
            r_sync = getattr(self, "_cfg_sync_rate_min", 0.0)

            # Eligibility set (review §5.8):
            #   E_set(t) = { i : E_i >= E_min,
            #                    l_i <= l_max,            (duty-share cap)
            #                    R_{i->swarm} >= R_min^sync }
            # The host-to-swarm sync link rate is proxied by the agent's
            # last realized rate (bps).
            eligible = [a for a in self.agents
                        if self.energy.get(a, 1.0) >= e_thr
                        and self.critic_duty_share.get(a, 0.0) <= l_max
                        and self.last_rates.get(a, 0.0) >= r_sync]

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
        load_max:         float = 1.00,  # l_max duty-share eligibility cap (§5.8)
        sync_rate_min:    float = 0.0,   # R_min^sync link-rate floor, bps (§5.8)
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
        self._cfg_energy_thr    = energy_threshold
        self._cfg_duty_thr      = duty_threshold
        self._cfg_load_max      = load_max
        self._cfg_sync_rate_min = sync_rate_min

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
            "energy_imbalance_IE":     self.energy_imbalance(),
            "comp_energy_total":       self.total_comp_energy(),
            "sync_energy_total":       self.total_sync_energy(),
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
