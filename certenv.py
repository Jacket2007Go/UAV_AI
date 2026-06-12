"""
certenv.py  —  Certificate-Bearing UAV JSC Environment
                (relative-absolute four-gate certificate, review-revised)

Implements the reviewer's revised certificate mathematics:

Hard certificate (review §4.1 / §5.4) — FOUR gates
──────────────────────────────────────────────────
  C_i(t) = 1[eta_i(t) >= R_i^min(t)]            (rate gate, bps/Hz)
         · 1[S_i(t)   >= S_i^min(t)]            (sensing gate, FIM units)
         · 1[E_i(t)   >= E_min]                 (energy gate)
         · prod_{j!=i} 1[d_ij(t) >= d_min]      (absolute safety-distance gate)

Relative-PLUS-absolute floors (review §4.2 / §5.4)
──────────────────────────────────────────────────
  R_i^min(t) = max( R_abs, (1 - tau_R) · median_j eta_j(t) )
  S_i^min(t) = max( S_abs, (1 - tau_S) · median_j S_j(t)  )
The absolute floors R_abs / S_abs prevent a collectively weak swarm from
"certifying" itself: if the whole swarm underperforms, the median collapses
but the absolute floor still fails everyone (mission QoS is enforced).

Deficits and EMA state (review §5.5)
──────────────────────────────────────────────────
  d_i^R(t) = max(0, R_i^min(t) - eta_i(t))
  d_i^S(t) = max(0, S_i^min(t) - S_i(t))
  D_i^R(t) = rho_R · D_i^R(t-1) + (1 - rho_R) · d_i^R(t)
  D_i^S(t) = rho_S · D_i^S(t-1) + (1 - rho_S) · d_i^S(t)
Observation extension (base 12 → 15):
  s_i ← [ s_i, tanh(D_i^R / s_R), tanh(D_i^S / s_S), tanh(w_i / s_W) ]
where w_i is the base env's workload EMA. These EMAs are driven by the
CERTIFICATE FLOORS (not the ADMM consensus targets z_R/z_Q as in the
legacy base-env memory, which is retained separately for diagnostics).

Soft certificate (review §5.6) — default reward shaping
──────────────────────────────────────────────────
  C~_i(t) = sigma(kR·(eta_i - R_i^min)) · sigma(kS·(S_i - S_i^min))
            · sigma(kE·(E_i - E_min)),      sigma(x) = 1/(1+e^-x)
  p_i^cert(t) = lam_R·d_i^R + lam_S·d_i^S
              + lam_E·max(0, E_min - E_i)
              + lam_D·sum_{j!=i} max(0, d_min - d_ij)
  reward:  r_i ← r_i - alpha_C · reward_scale · p_i^cert     (cert_mode='soft')
The penalty scales with VIOLATION MAGNITUDE — more informative than the
legacy constant per-failure penalty, which is retained as cert_mode='hard'
(r_i ← r_i - lam_cert · reward_scale per failed hard certificate) so the
hard-vs-soft ablation in review §6.1 is a one-flag switch. The hard
certificate C_i(t) is ALWAYS evaluated and logged for reporting.

Minimum-service metric (review §6.3)
──────────────────────────────────────────────────
  M_QoS(t) = min_i min( eta_i/R_i^min , S_i/S_i^min )
M_QoS >= 1  ⇔  every agent satisfies both normalized service floors.

Diagnostic state exposed after step()
──────────────────────────────────────────────────
  last_cert_passed[a]        bool   hard four-gate certificate C_i(t)
  last_cert_soft[a]          float  soft certificate C~_i(t) ∈ (0,1)
  last_cert_components[a]    dict   {"comm","sense","energy","dist","prox"}
                                    (prox = legacy relative gate, logged only)
  last_per_agent_rate[a]     float  eta_i(t)  spectral efficiency (bps/Hz)
  last_per_agent_sense[a]    float  S_i(t)    FIM-based sensing quality
  last_per_agent_penalty[a]  float  reward penalty applied this step
  last_per_agent_violation[a]float  d_i^R + d_i^S + energy + distance deficits
  last_floor_R[a], last_floor_S[a]  per-agent active floors
  last_m_qos                 float  M_QoS(t)
  last_min_pair_dist / last_max_pair_dist / last_cert_thresholds
  last_cert_issued_count     int    number of agents with C_i(t) = 1
"""

import numpy as np
from gymnasium import spaces
from dcenv import UAVJSCEnv


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


class CertUAVJSCEnv(UAVJSCEnv):
    """
    Relative-absolute four-gate certificate environment (obs_dim = 15).

    Parameters
    ----------
    cert_tau_R, cert_tau_S : float
        Relative slack for the rate / sensing floors:
        pass requires value >= (1 - tau) * swarm median. Review §4.2 sweeps
        these independently, hence two parameters (legacy cert_tau sets both).
    cert_R_abs : float
        Absolute mission rate floor R_abs in bps/Hz (spectral efficiency).
        Floor used is max(R_abs, relative). 0 disables (relative-only
        ablation, review §6.1).
    cert_S_abs : float
        Absolute mission sensing floor S_abs in S_i(t) units. 0 disables.
    cert_E_min : float
        Energy gate: E_i(t) >= E_min (review §4.1). Fraction of battery.
    cert_d_min : float
        ABSOLUTE safety-distance gate in env units: d_ij >= d_min for all
        j != i (review §4.1/§5.4). The legacy RELATIVE proximity check
        (alpha × max pairwise distance) is still computed and logged as
        'prox' but does not gate.
    cert_mode : str
        'soft' (default, review §5.6): magnitude-scaled penalty p_i^cert.
        'hard': legacy constant penalty lam_cert per failed certificate.
    cert_lambda_R/S/E/D : float
        Soft-penalty weights (per unit deficit). Defaults are calibrated so
        a TYPICAL violation (d^R ~ 0.7 bps/Hz or d^S ~ 0.1·floor_S) costs
        the same order as the legacy constant penalty 0.5·reward_scale,
        keeping soft-vs-hard a like-for-like ablation (review §6.1).
        Deficit ranges with default taus: d^R ≲ 1.5 bps/Hz, d^S ≲ 0.1,
        e ≲ 0.1, d^D in env units (rare given the ring formation).
    cert_kappa_R/S/E : float
        Sigmoid sharpness of the soft certificate C~_i (reporting only).
    cert_alpha_C : float
        alpha_C multiplier on p_i^cert in the reward (review §5.7).
    cert_rho_R, cert_rho_S : float
        EMA persistence for D_i^R, D_i^S (review §5.5).
    cert_lambda : float
        Legacy hard-mode constant penalty (× reward_scale).
    cert_alpha : float
        Legacy relative proximity threshold (logged only).
    cert_R_weight, cert_Q_weight : float
        DEAD parameters retained for CLI compatibility.
    """

    # Observation normalisation scales (s_R, s_S, s_W in review §5.5).
    # s_R is sized for bps/Hz deficits (tau_R * median eta ~ 1.5),
    # s_S for FIM-unit deficits, s_W for the workload EMA.
    CERT_COMM_SCALE  = 1.00
    CERT_SENSE_SCALE = 0.10
    CERT_WORK_SCALE  = 0.50

    def __init__(
        self,
        num_uavs: int = 5,
        motion_case: str = "all_move",
        fading_on: bool = True,
        horizon: int = 100,
        critic_mode: str = "fixed",
        # Relative slacks (review §4.2: separate tau_R / tau_S sweeps)
        cert_tau:   float = None,    # legacy: sets both taus if given
        cert_tau_R: float = 0.25,
        cert_tau_S: float = 0.25,
        # Absolute mission floors (review §4.2)
        cert_R_abs: float = 0.5,     # bps/Hz
        cert_S_abs: float = 0.02,    # FIM sensing units
        # Energy and absolute-distance gates (review §4.1)
        cert_E_min: float = 0.10,
        cert_d_min: float = 114.0,   # env units (≈ paper's prox threshold)
        # Penalty mode and weights (review §5.6)
        cert_mode: str = "soft",
        cert_lambda_R: float = 0.30,
        cert_lambda_S: float = 10.0,
        cert_lambda_E: float = 2.00,
        cert_lambda_D: float = 0.01,
        cert_alpha_C:  float = 1.00,
        cert_kappa_R: float = 4.0,
        cert_kappa_S: float = 40.0,
        cert_kappa_E: float = 40.0,
        # EMA persistence (review §5.5)
        cert_rho_R: float = 0.95,
        cert_rho_S: float = 0.95,
        # Legacy parameters
        cert_lambda: float = 0.5,    # hard-mode constant penalty
        cert_alpha:  float = 0.40,   # relative prox (logged only)
        cert_R_weight: float = 2.0,  # dead — kept for CLI compatibility
        cert_Q_weight: float = 1.0,  # dead — kept for CLI compatibility
    ):
        # Store cert hyperparameters before super().__init__ — _get_obs and
        # any reset hook may run during the parent constructor.
        if cert_tau is not None:          # legacy single-tau call sites
            cert_tau_R = cert_tau_S = float(cert_tau)
        self.cert_tau_R  = float(cert_tau_R)
        self.cert_tau_S  = float(cert_tau_S)
        self.cert_tau    = self.cert_tau_R          # backward-compat alias
        self.cert_R_abs  = float(cert_R_abs)
        self.cert_S_abs  = float(cert_S_abs)
        self.cert_E_min  = float(cert_E_min)
        self.cert_d_min  = float(cert_d_min)
        self.cert_mode   = str(cert_mode)
        self.cert_lambda_R = float(cert_lambda_R)
        self.cert_lambda_S = float(cert_lambda_S)
        self.cert_lambda_E = float(cert_lambda_E)
        self.cert_lambda_D = float(cert_lambda_D)
        self.cert_alpha_C  = float(cert_alpha_C)
        self.cert_kappa_R  = float(cert_kappa_R)
        self.cert_kappa_S  = float(cert_kappa_S)
        self.cert_kappa_E  = float(cert_kappa_E)
        self.cert_rho_R    = float(cert_rho_R)
        self.cert_rho_S    = float(cert_rho_S)
        self.cert_lambda   = float(cert_lambda)
        self.cert_alpha    = float(cert_alpha)
        self.cert_R_weight = cert_R_weight
        self.cert_Q_weight = cert_Q_weight

        # Floor-driven EMA deficit state D_i^R, D_i^S (review §5.5) — must
        # exist before super().__init__ because _get_obs runs in reset().
        self.cert_D_R = {}
        self.cert_D_S = {}

        super().__init__(
            num_uavs=num_uavs,
            motion_case=motion_case,
            fading_on=fading_on,
            horizon=horizon,
            critic_mode=critic_mode,
        )

        # Override obs_dim: base 12 + 3 cert features = 15
        self.obs_dim = 15
        self.observation_spaces = {
            a: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim,), dtype=np.float32,
            )
            for a in self.agents
        }

        self._init_cert_diagnostics()

    # ------------------------------------------------------------------
    # Diagnostic initialisation
    # ------------------------------------------------------------------
    def _init_cert_diagnostics(self):
        """Zero out all diagnostic dicts and the floor-driven EMA state."""
        z = {a: 0.0 for a in self.agents}
        self.cert_D_R = dict(z)                       # D_i^R  (review §5.5)
        self.cert_D_S = dict(z)                       # D_i^S
        self.last_per_agent_rate      = dict(z)       # eta_i (bps/Hz)
        self.last_per_agent_sense     = dict(z)       # S_i
        self.last_per_agent_penalty   = dict(z)
        self.last_per_agent_violation = dict(z)       # review §6.3 magnitude
        self.last_cert_soft           = dict(z)       # C~_i (review §5.6)
        self.last_floor_R             = dict(z)       # R_i^min(t)
        self.last_floor_S             = dict(z)       # S_i^min(t)
        self.last_cert_passed = {a: False for a in self.agents}
        self.last_cert_components = {
            a: {"comm": False, "sense": False, "energy": False,
                "dist": False, "prox": False}
            for a in self.agents
        }
        self.last_min_pair_dist = 0.0
        self.last_max_pair_dist = 0.0
        self.last_cert_thresholds = {"comm": 0.0, "sense": 0.0,
                                     "energy": self.cert_E_min,
                                     "dist": self.cert_d_min, "prox": 0.0}
        self.last_cert_issued_count = 0
        self.last_m_qos = 0.0                          # M_QoS (review §6.3)

    def reset(self, seed=None, options=None):
        """Reset env and re-init diagnostic dicts for the new episode."""
        out = super().reset(seed=seed, options=options)
        self._init_cert_diagnostics()
        return out

    # ------------------------------------------------------------------
    # Extended observation: floor-driven EMA deficit features (review §5.5)
    #   s_i ← [s_i, tanh(D_i^R/s_R), tanh(D_i^S/s_S), tanh(w_i/s_W)]
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict:
        obs = super()._get_obs()
        for a in self.agents:
            f_R = float(np.tanh(self.cert_D_R.get(a, 0.0) / self.CERT_COMM_SCALE))
            f_S = float(np.tanh(self.cert_D_S.get(a, 0.0) / self.CERT_SENSE_SCALE))
            f_W = float(np.tanh(self.cert_workload.get(a, 0.0) / self.CERT_WORK_SCALE))
            obs[a] = np.concatenate(
                [obs[a], np.array([f_R, f_S, f_W], dtype=np.float32)]
            )
        return obs

    # ------------------------------------------------------------------
    # Certificate evaluation — pure function of post-step swarm state
    # ------------------------------------------------------------------
    def _evaluate_certificates(self):
        """
        Evaluate the four-gate hard certificate C_i(t), the soft certificate
        C~_i(t), per-agent deficits d_i^R/d_i^S, the EMA state D_i^R/D_i^S,
        violation magnitudes, and M_QoS. Populates all diagnostic dicts and
        returns (passed, penalties) where penalties are the per-agent reward
        penalties for the configured cert_mode.
        """
        agents = list(self.agents)
        N = len(agents)

        # ─── Component arrays ──────────────────────────────────────────
        # Comm gate operates on SPECTRAL EFFICIENCY eta_i (bps/Hz) so that
        # R_abs is a bandwidth-independent mission floor (review §4.3).
        eta = np.array(
            [self.last_spectral_eff.get(a, 0.0) for a in agents],
            dtype=np.float64,
        )
        sense = np.array(
            [self.last_sensing.get(a, 0.0) for a in agents],
            dtype=np.float64,
        )
        energy = np.array(
            [self.energy.get(a, 1.0) for a in agents], dtype=np.float64
        )

        # ─── Pairwise distances (N x N) ────────────────────────────────
        if N >= 2:
            pos = np.stack([self.positions[a] for a in agents], axis=0)
            diffs = pos[:, None, :] - pos[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            np.fill_diagonal(dists, np.inf)
            nearest = dists.min(axis=1)               # nearest neighbour
            np.fill_diagonal(dists, 0.0)
            max_pair = float(dists.max())
            min_pair = float(nearest.min())
        else:
            dists    = np.zeros((1, 1))
            nearest  = np.array([np.inf])
            max_pair = 0.0
            min_pair = float("inf")

        # ─── Relative-PLUS-absolute floors (review §4.2 / §5.4) ────────
        #   R_i^min = max(R_abs, (1 - tau_R) * median eta)
        #   S_i^min = max(S_abs, (1 - tau_S) * median S)
        med_eta = float(np.median(eta))
        med_s   = float(np.median(sense))
        floor_R = max(self.cert_R_abs, (1.0 - self.cert_tau_R) * med_eta)
        floor_S = max(self.cert_S_abs, (1.0 - self.cert_tau_S) * med_s)
        # Legacy relative proximity threshold (logged only, never gates)
        prox_thr = self.cert_alpha * max_pair if max_pair > 0 else 0.0

        passed, components, soft, penalties = {}, {}, {}, {}
        violations = {}
        n_passed = 0
        m_qos = np.inf
        scale = float(getattr(self, "reward_scale", 1.0))

        for i, a in enumerate(agents):
            # ── Hard four-gate certificate (review §4.1 / §5.4) ─────────
            comm_ok   = bool(eta[i]    >= floor_R)
            sense_ok  = bool(sense[i]  >= floor_S)
            energy_ok = bool(energy[i] >= self.cert_E_min)
            if N >= 2:
                dist_ok = bool(nearest[i] >= self.cert_d_min)
            else:
                dist_ok = True
            prox_ok = bool(nearest[i] >= prox_thr) if N >= 2 else True

            c_i = comm_ok and sense_ok and energy_ok and dist_ok
            passed[a] = c_i
            if c_i:
                n_passed += 1
            components[a] = {"comm": comm_ok, "sense": sense_ok,
                             "energy": energy_ok, "dist": dist_ok,
                             "prox": prox_ok}

            # ── Deficits d_i^R, d_i^S (review §5.5) ─────────────────────
            d_R = max(0.0, floor_R - float(eta[i]))
            d_S = max(0.0, floor_S - float(sense[i]))
            d_E = max(0.0, self.cert_E_min - float(energy[i]))
            if N >= 2:
                row  = dists[i].copy()
                row[i] = np.inf
                d_D = float(np.sum(np.maximum(0.0, self.cert_d_min - row)
                                   [np.isfinite(row)]))
            else:
                d_D = 0.0

            # EMA update: D(t) = rho * D(t-1) + (1 - rho) * d(t)
            self.cert_D_R[a] = (self.cert_rho_R * self.cert_D_R.get(a, 0.0)
                                + (1.0 - self.cert_rho_R) * d_R)
            self.cert_D_S[a] = (self.cert_rho_S * self.cert_D_S.get(a, 0.0)
                                + (1.0 - self.cert_rho_S) * d_S)

            # ── Soft certificate C~_i (review §5.6, reporting) ──────────
            soft[a] = (_sigmoid(self.cert_kappa_R * (float(eta[i])    - floor_R))
                     * _sigmoid(self.cert_kappa_S * (float(sense[i])  - floor_S))
                     * _sigmoid(self.cert_kappa_E * (float(energy[i]) - self.cert_E_min)))

            # ── Penalty (review §5.6) ───────────────────────────────────
            p_cert = (self.cert_lambda_R * d_R
                      + self.cert_lambda_S * d_S
                      + self.cert_lambda_E * d_E
                      + self.cert_lambda_D * d_D)
            violations[a] = d_R + d_S + d_E + d_D
            if self.cert_mode == "hard":
                # Legacy constant per-failure penalty (ablation: §6.1)
                penalties[a] = (self.cert_lambda * scale) if not c_i else 0.0
            else:
                # Soft magnitude-scaled penalty: alpha_C * p_i^cert
                penalties[a] = self.cert_alpha_C * scale * p_cert

            # ── M_QoS contribution (review §6.3) ────────────────────────
            m_i = min(float(eta[i])   / (floor_R + 1e-12),
                      float(sense[i]) / (floor_S + 1e-12))
            m_qos = min(m_qos, m_i)

            self.last_floor_R[a] = floor_R
            self.last_floor_S[a] = floor_S

        # ─── Persist diagnostics ───────────────────────────────────────
        self.last_per_agent_rate      = {a: float(eta[i])
                                         for i, a in enumerate(agents)}
        self.last_per_agent_sense     = {a: float(sense[i])
                                         for i, a in enumerate(agents)}
        self.last_per_agent_violation = {a: float(violations[a]) for a in agents}
        self.last_cert_passed     = passed
        self.last_cert_components = components
        self.last_cert_soft       = soft
        self.last_min_pair_dist   = min_pair if np.isfinite(min_pair) else 0.0
        self.last_max_pair_dist   = max_pair
        self.last_cert_thresholds = {
            "comm":   floor_R,
            "sense":  floor_S,
            "energy": self.cert_E_min,
            "dist":   self.cert_d_min,
            "prox":   prox_thr,
        }
        self.last_cert_issued_count = int(n_passed)
        self.last_m_qos = float(m_qos if np.isfinite(m_qos) else 0.0)
        return passed, penalties

    # ------------------------------------------------------------------
    # step() — base reward, certificate evaluation, penalty application
    # ------------------------------------------------------------------
    def step(self, actions: dict):
        """
        Execute the base dcenv step, evaluate the four-gate certificate over
        the resulting swarm state, then apply the configured penalty:
          cert_mode='soft' : r_i -= alpha_C * reward_scale * p_i^cert
          cert_mode='hard' : r_i -= lam_cert * reward_scale per failed cert
        The hard certificate C_i(t) is evaluated and logged in BOTH modes
        (review §5.6: "hard certificate kept for reporting").
        """
        obs, rewards, dones, truncated, info = super().step(actions)

        passed, penalties = self._evaluate_certificates()

        for a in self.agents:
            pen = float(penalties.get(a, 0.0))
            rewards[a] = float(rewards[a] - pen)
            self.last_per_agent_penalty[a] = pen

        # Observations must reflect the POST-update EMA deficit state
        # (the parent step() built obs before _evaluate_certificates ran).
        obs = self._get_obs()

        return obs, rewards, dones, truncated, info
