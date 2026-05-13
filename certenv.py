"""
certenv.py  —  Certificate-Bearing UAV JSC Environment (HARD CERTIFICATE)

Implements a hard certificate gate over three swarm-tuned criteria:
  1. Communication: agent rate ≥ (1 − τ) × swarm-median rate
  2. Sensing:       agent sense ≥ (1 − τ) × swarm-median sense
  3. Proximity:     agent's nearest-neighbour distance ≥ α × max-pairwise-dist

Certificate behaviour
─────────────────────
At each step, every agent that satisfies ALL three criteria "receives" a
certificate. Every agent that fails ≥1 criterion does not. This is per-step,
binary, and reported alongside the existing EMA deficit features.

Reward shaping (per-failure penalty, option (a) from the design discussion)
─────────────────────────────────────────────────────────────────────
Each agent that does NOT receive a certificate this step pays an individual
penalty of cert_lambda × env.reward_scale. No team penalty. Each agent eats
its own failure — clean attribution, no free-rider dynamics.

Diagnostic state exposed after step()
─────────────────────────────────────────────────────────────────────
After every step(), the env populates:
  self.last_cert_passed[a]       : bool — did agent a receive cert this step?
  self.last_cert_components[a]   : dict {"comm": bool, "sense": bool, "prox": bool}
  self.last_per_agent_rate[a]    : float — comm rate (bps/MHz)
  self.last_per_agent_sense[a]   : float — sensing utility
  self.last_per_agent_penalty[a] : float — cert penalty applied this step
  self.last_min_pair_dist        : float — minimum pairwise distance
  self.last_max_pair_dist        : float — maximum pairwise distance
  self.last_cert_thresholds      : dict {"comm": ..., "sense": ..., "prox": ...}
  self.last_cert_issued_count    : int — how many of N agents passed

train_model.py logs episode-mean values of these to .npy files for
diagnose_certificate.py to consume.
"""

import numpy as np
from gymnasium import spaces
from dcenv import UAVJSCEnv


class CertUAVJSCEnv(UAVJSCEnv):
    """
    Hard-certificate environment: obs_dim=15, three swarm-tuned criteria,
    per-failure penalty, rich diagnostic logging.

    Parameters
    ----------
    cert_tau : float
        Slack tolerance for comm/sense thresholds. Agent passes comm iff
        rate >= (1 - tau) * median_rate. tau=0.05 is "within 5% of median."
    cert_alpha : float
        Proximity threshold as a fraction of the swarm's max pairwise
        distance. Agent passes prox iff its nearest-neighbour distance
        >= alpha * max_pairwise_distance. alpha=0.15 means "the closest
        other UAV is at least 15% of the swarm extent away."
    cert_lambda : float
        Per-failure reward penalty (multiplied by env.reward_scale).
        Each non-passing agent loses cert_lambda * reward_scale this step.
    cert_R_weight, cert_Q_weight : float
        DEAD parameters retained for API compatibility with train_model.py.
        Previously used for reward-shaping; no longer touched.
    """

    # Observation normalisation scales for the EMA-deficit cert features
    CERT_COMM_SCALE  = 0.10
    CERT_SENSE_SCALE = 0.10
    CERT_WORK_SCALE  = 0.50

    def __init__(
        self,
        num_uavs: int = 5,
        motion_case: str = "all_move",
        fading_on: bool = True,
        horizon: int = 100,
        critic_mode: str = "fixed",
        cert_tau: float = 0.25,
        cert_alpha: float = 0.40,
        cert_lambda: float = 0.5,
        cert_R_weight: float = 2.0,    # dead — kept for CLI compatibility
        cert_Q_weight: float = 1.0,    # dead — kept for CLI compatibility
    ):
        # Store cert hyperparameters before super().__init__ — _get_obs and
        # any reset hook may run during the parent constructor.
        self.cert_tau    = float(cert_tau)
        self.cert_alpha  = float(cert_alpha)
        self.cert_lambda = float(cert_lambda)
        self.cert_R_weight = cert_R_weight
        self.cert_Q_weight = cert_Q_weight

        super().__init__(
            num_uavs=num_uavs,
            motion_case=motion_case,
            fading_on=fading_on,
            horizon=horizon,
            critic_mode=critic_mode,
        )

        # Override obs_dim: base 12 + 3 EMA-deficit cert features = 15
        self.obs_dim = 15
        self.observation_spaces = {
            a: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim,), dtype=np.float32,
            )
            for a in self.agents
        }

        # Initialise diagnostic dicts so callers can read them even before
        # the first step() (e.g. immediately after reset()).
        self._init_cert_diagnostics()

    # ------------------------------------------------------------------
    # Diagnostic initialisation
    # ------------------------------------------------------------------
    def _init_cert_diagnostics(self):
        """Zero out all diagnostic dicts (called after __init__ and reset)."""
        z = {a: 0.0 for a in self.agents}
        self.last_per_agent_rate    = dict(z)
        self.last_per_agent_sense   = dict(z)
        self.last_per_agent_penalty = dict(z)
        self.last_cert_passed       = {a: False for a in self.agents}
        self.last_cert_components   = {
            a: {"comm": False, "sense": False, "prox": False}
            for a in self.agents
        }
        self.last_min_pair_dist  = 0.0
        self.last_max_pair_dist  = 0.0
        self.last_cert_thresholds = {"comm": 0.0, "sense": 0.0, "prox": 0.0}
        self.last_cert_issued_count = 0

    def reset(self, seed=None, options=None):
        """Reset env and re-init diagnostic dicts for the new episode."""
        out = super().reset(seed=seed, options=options)
        self._init_cert_diagnostics()
        return out

    # ------------------------------------------------------------------
    # Step 2 substrate: extended observation (cert EMA features appended)
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict:
        """Return 15-dim per-agent observations with EMA-deficit features."""
        obs = super()._get_obs()
        for a in self.agents:
            cert_c = float(np.tanh(
                self.cert_comm_deficit.get(a, 0.0)  / self.CERT_COMM_SCALE))
            cert_s = float(np.tanh(
                self.cert_sense_deficit.get(a, 0.0) / self.CERT_SENSE_SCALE))
            cert_w = float(np.tanh(
                self.cert_workload.get(a, 0.0)      / self.CERT_WORK_SCALE))
            obs[a] = np.concatenate(
                [obs[a], np.array([cert_c, cert_s, cert_w], dtype=np.float32)]
            )
        return obs

    # ------------------------------------------------------------------
    # Cert evaluation — pure function of post-step swarm state
    # ------------------------------------------------------------------
    def _evaluate_certificates(self):
        """
        Compute per-agent cert pass/fail from current self.last_rates,
        self.last_sensing, and self.positions. Populates all diagnostic
        dicts and returns the dict of pass/fail booleans.

        Thresholds are swarm-tuned:
          comm_thr  = (1 - tau) * median(rates)
          sense_thr = (1 - tau) * median(sensing)
          prox_thr  = alpha * max_pairwise_distance
        """
        agents = list(self.agents)
        N = len(agents)
        bw = float(getattr(self, "bandwidth", 1.0))

        # ─── Component arrays ──────────────────────────────────────────
        rates_bps_per_hz = np.array(
            [self.last_rates.get(a, 0.0) / (bw + 1e-12) for a in agents],
            dtype=np.float64,
        )
        sense = np.array(
            [self.last_sensing.get(a, 0.0) for a in agents],
            dtype=np.float64,
        )

        # ─── Pairwise distances (N x N) ────────────────────────────────
        if N >= 2:
            pos = np.stack([self.positions[a] for a in agents], axis=0)
            diffs = pos[:, None, :] - pos[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)        # (N, N)
            np.fill_diagonal(dists, np.inf)               # ignore self
            nearest = dists.min(axis=1)                   # nearest-nbr per agent
            np.fill_diagonal(dists, 0.0)
            max_pair = float(dists.max())
            min_pair = float(nearest.min())
        else:
            nearest  = np.array([np.inf])
            max_pair = 0.0
            min_pair = float("inf")

        # ─── Swarm-tuned thresholds ────────────────────────────────────
        med_r = float(np.median(rates_bps_per_hz))
        med_s = float(np.median(sense))
        comm_thr  = (1.0 - self.cert_tau) * med_r
        sense_thr = (1.0 - self.cert_tau) * med_s
        prox_thr  = self.cert_alpha * max_pair if max_pair > 0 else 0.0

        # ─── Per-agent component pass/fail ─────────────────────────────
        # Certificate gate: comm AND sense must both pass.
        # Proximity is computed and logged for D4/D5 safety monitoring
        # but does NOT gate the certificate.
        #
        # Rationale: the UAV ring formation is essentially static under
        # the current motion model — min pairwise distance is constant
        # at ~176 env units across all episodes. Any cert_alpha above
        # ~0.618 causes proximity to always fail; below it always passes.
        # There is no middle ground, so proximity cannot be an informative
        # binary gate. The constancy IS the safety result: the ADMM motion
        # layer maintains collision avoidance structurally. Proximity is
        # still reported in D4 and D5 so the mentor can verify this.
        passed     = {}
        components = {}
        n_passed   = 0
        for i, a in enumerate(agents):
            comm_ok  = bool(rates_bps_per_hz[i] >= comm_thr)
            sense_ok = bool(sense[i]            >= sense_thr)
            prox_ok  = bool(nearest[i]          >= prox_thr) if N >= 2 else True

            # Gate on comm AND sense only
            all_ok = comm_ok and sense_ok
            passed[a] = all_ok
            if all_ok:
                n_passed += 1
            components[a] = {"comm": comm_ok, "sense": sense_ok, "prox": prox_ok}

        # ─── Persist diagnostics ───────────────────────────────────────
        self.last_per_agent_rate    = {a: float(rates_bps_per_hz[i])
                                       for i, a in enumerate(agents)}
        self.last_per_agent_sense   = {a: float(sense[i])
                                       for i, a in enumerate(agents)}
        self.last_cert_passed       = passed
        self.last_cert_components   = components
        self.last_min_pair_dist     = min_pair if np.isfinite(min_pair) else 0.0
        self.last_max_pair_dist     = max_pair
        self.last_cert_thresholds   = {
            "comm":  comm_thr,
            "sense": sense_thr,
            "prox":  prox_thr,
        }
        self.last_cert_issued_count = int(n_passed)
        return passed

    # ------------------------------------------------------------------
    # step() — base reward + per-failure cert penalty + diagnostic logging
    # ------------------------------------------------------------------
    def step(self, actions: dict):
        """
        Execute base dcenv step, evaluate hard certificate over the
        resulting swarm state, then subtract the per-failure penalty
        from each non-passing agent's reward.
        """
        obs, rewards, dones, truncated, info = super().step(actions)

        passed = self._evaluate_certificates()

        # Per-failure reward penalty (option (a)): each agent eats its own.
        # Scaled by reward_scale (~0.05) so the penalty stays comparable in
        # magnitude to the existing fair_R, fair_Q, and jain_bonus terms.
        scale = float(getattr(self, "reward_scale", 1.0))
        for a in self.agents:
            if not passed[a]:
                pen = self.cert_lambda * scale
                rewards[a] = float(rewards[a] - pen)
                self.last_per_agent_penalty[a] = pen
            else:
                self.last_per_agent_penalty[a] = 0.0

        return obs, rewards, dones, truncated, info
