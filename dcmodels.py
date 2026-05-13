"""
models.py  —  Neural network models + coordinator helpers for Hybrid MARL-ADMM UAV JSC

Proposal architecture mapping
══════════════════════════════════════════════════════════════════════════════
 Step 1  Certificate update           → env.py  update_certificates()
 Step 2  AI warm start                → DeterministicActor  (this file)
 Step 3  Certified MM-ADMM refinement → env.py  step() ADMM block
 Step 4  Barrier/QP safe execution    → SafeActionProjector  (this file)
 Step 5  Realized measurement/learn   → train.py  learning update block

Distributed-critic PDF mapping (Eqs 1-3)
══════════════════════════════════════════════════════════════════════════════
 CriticCoordinatorConfig   holds α weights for Eq.(2)
 WeightedCriticSelector    implements Eq.(2) Q_i(t) and Eq.(3) i*(t)
 Fixed / round-robin modes also available (PDF §2)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ============================================================================
# Step 2 — AI Warm Start: DeterministicActor
# ============================================================================

class DeterministicActor(nn.Module):
    """
    Distributed actor  π_θi(a_i | s_i).

    Proposal role: Step 2 — AI Warm Start.
    The actor produces a *nominal* action proposal from the local observation.
    This proposal is then passed through:
        Step 3  (certified MM-ADMM refinement inside env.step)
        Step 4  (SafeActionProjector before env.step)
    so that the final executed action respects both fairness and safety.

    At *deployment* only the actor is needed; the critic and coordinator are
    training-time constructs.
    """

    def __init__(self, obs_dim: int, w_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.w_dim = w_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 + w_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        p   = torch.sigmoid(out[..., :1])           # power ∈ (0, 1)
        w_raw = out[..., 1:]
        w   = torch.tanh(w_raw)
        w   = w / (torch.norm(w, dim=-1, keepdim=True) + 1e-8)  # unit beamformer
        return torch.cat([p, w], dim=-1)


# ============================================================================
# Shared Critic Network
# ============================================================================

class CentralizedQCritic(nn.Module):
    """
    Centralized Q critic:  Q(s_global, a_joint).

    s_global = concatenation of all agent observations.
    a_joint  = concatenation of all agent actions.

    Proposal role: Shared critic network managed by the CriticCoordinator.
    Under 'fixed' mode it is logically owned by uav_0 (current behaviour).
    Under 'round_robin' / 'weighted' mode the *logical host* rotates while
    the network weights remain in shared memory (single-model approximation —
    see CriticCoordinator docstring for rationale).
    """

    def __init__(self, global_obs_dim: int, joint_act_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim + joint_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s_global: torch.Tensor,
                a_joint: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s_global, a_joint], dim=-1)
        return self.net(x)


# ============================================================================
# Distributed Critic Coordinator — config + selector
# ============================================================================

@dataclass
class CriticCoordinatorConfig:
    """
    Configuration for distributed critic selection.

    Proposal / PDF mapping:
        critic_mode  selects the strategy from PDF §2 (round-robin) or §3 (weighted).
        alpha_*      are the α weights in PDF Eq.(2):
                       Q_i(t) = α_E Ẽ_i + α_L L̃_i + α_C C̃_i
                               + α_S S̃_i + α_P P̃_i + α_W W̃_i
        *_threshold  guard conditions from PDF §3.2 ("meets thresholds?" branch).
    """
    # Selection strategy
    critic_mode: str = "fixed"     # 'fixed' | 'round_robin' | 'weighted'

    # Scoring weights (must sum to ~1.0 for interpretability, not enforced)
    alpha_E: float = 0.25   # residual energy weight
    alpha_L: float = 0.20   # low-load weight
    alpha_C: float = 0.15   # capability (clock speed) weight
    alpha_S: float = 0.20   # channel/state quality weight
    alpha_P: float = 0.10   # proximity weight
    alpha_W: float = 0.10   # willingness weight

    # Threshold guards (PDF §3.2 — "meets thresholds?" gate)
    energy_threshold: float = 0.10   # min energy fraction to be eligible
    duty_threshold:   float = 0.50   # max duty share before willingness is reduced


class WeightedCriticSelector:
    """
    Weighted preference-based critic selection  (PDF §3, Eqs 2-3).

    Practical mapping from current codebase quantities:
    ────────────────────────────────────────────────────────────────────────
    Component   Symbol  Source in codebase
    ────────────────────────────────────────────────────────────────────────
    Energy       Ẽ_i   env.energy[a]            (already ∈ [0,1])
    Low-load     L̃_i   1 - duty_share[a]        (inverse of hosting fraction)
    Capability   C̃_i   env.clock_speed[a]       (normed within swarm)
    State qual.  S̃_i   ‖env.channel[a]‖         (normed within swarm)
    Proximity    P̃_i   1 - dist_to_center/max   (closer to swarm centroid)
    Willingness  W̃_i   starts 1.0, penalised if energy<threshold or overloaded
    ────────────────────────────────────────────────────────────────────────
    All components are normalised to [0,1] before weighting.
    """

    def __init__(self, config: CriticCoordinatorConfig,
                 agents: List[str]):
        self.cfg    = config
        self.agents = agents
        self.N      = len(agents)

    def compute_scores(
        self,
        energy:         Dict[str, float],
        duty_share:     Dict[str, float],
        clock_speed:    Dict[str, float],
        channel_mag:    Dict[str, float],
        dist_to_center: Dict[str, float],
        max_dist:       float,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute per-agent Q_i(t) scores  (PDF Eq. 2).

        Returns
        -------
        scores     : {agent: Q_i}
        components : {agent: {E, L, C, S, P, W, Q}} — for logging
        """
        eps = 1e-8

        E_arr = np.array([energy.get(a, 1.0)      for a in self.agents], dtype=np.float64)
        L_arr = np.array([1.0 - duty_share.get(a, 0.0)  for a in self.agents], dtype=np.float64)
        C_arr = np.array([clock_speed.get(a, 1.0) for a in self.agents], dtype=np.float64)
        S_arr = np.array([channel_mag.get(a, 1.0) for a in self.agents], dtype=np.float64)
        P_arr = np.array(
            [max(0.0, 1.0 - dist_to_center.get(a, 0.0) / (max_dist + eps))
             for a in self.agents], dtype=np.float64
        )

        # Hard-clip energy and low-load to [0, 1]
        E_arr = np.clip(E_arr, 0.0, 1.0)
        L_arr = np.clip(L_arr, 0.0, 1.0)

        # Min-max normalize C, S within the swarm; P is already in [0,1]
        def _norm01(v: np.ndarray) -> np.ndarray:
            lo, hi = v.min(), v.max()
            return (v - lo) / (hi - lo + eps)

        C_arr = _norm01(C_arr)
        S_arr = _norm01(S_arr)
        P_arr = np.clip(P_arr, 0.0, 1.0)

        # Willingness W̃_i  (PDF §3.1: "willing to act as critic")
        W_arr = np.ones(self.N, dtype=np.float64)
        for i, a in enumerate(self.agents):
            en = energy.get(a, 1.0)
            ds = duty_share.get(a, 0.0)
            # Reduce if energy below threshold (proportional reduction)
            if en < self.cfg.energy_threshold:
                W_arr[i] *= max(0.0, en / (self.cfg.energy_threshold + eps))
            # Reduce if over-burdened with critic duty
            if ds > self.cfg.duty_threshold:
                excess = (ds - self.cfg.duty_threshold) / (1.0 - self.cfg.duty_threshold + eps)
                W_arr[i] *= max(0.0, 1.0 - excess)
        W_arr = np.clip(W_arr, 0.0, 1.0)

        # Weighted sum  Q_i(t)  (PDF Eq. 2)
        Q_arr = (self.cfg.alpha_E * E_arr
               + self.cfg.alpha_L * L_arr
               + self.cfg.alpha_C * C_arr
               + self.cfg.alpha_S * S_arr
               + self.cfg.alpha_P * P_arr
               + self.cfg.alpha_W * W_arr)

        scores = {a: float(Q_arr[i]) for i, a in enumerate(self.agents)}
        components = {
            a: {
                "E": float(E_arr[i]), "L": float(L_arr[i]),
                "C": float(C_arr[i]), "S": float(S_arr[i]),
                "P": float(P_arr[i]), "W": float(W_arr[i]),
                "Q": float(Q_arr[i]),
            }
            for i, a in enumerate(self.agents)
        }
        return scores, components

    def select(
        self,
        scores:         Dict[str, float],
        energy:         Dict[str, float],
        prev_active:    str,
        default_agent:  str,
    ) -> str:
        """
        Select i*(t) = argmax_i Q_i(t) subject to threshold checks.

        PDF §3.3 flowchart: compute Q_i → meets thresholds? → assign / fallback.

        Fallback priority:
            1. previous valid critic (continuity)
            2. default_agent (uav_0, backward compat)
        """
        eligible = [a for a in self.agents
                    if energy.get(a, 1.0) >= self.cfg.energy_threshold]

        if not eligible:
            # Fallback: previous host if still in swarm, else default
            return prev_active if prev_active in self.agents else default_agent

        best = max(eligible, key=lambda a: scores.get(a, 0.0))
        return best


# ============================================================================
# Step 4 — Barrier/QP Safe Execution: SafeActionProjector
# ============================================================================

class SafeActionProjector:
    """
    Lightweight safe-action projection layer.

    Proposal role: Step 4 — Barrier/QP Safe Execution (structural substrate).
    This is a lightweight approximation; it enforces hard feasibility bounds
    without formal Lyapunov / CBF certificates.

    Safety constraints enforced:
        1. Power   ∈ [0, P_max · E_i]   (energy-aware upper bound)
        2. Beamformer  ‖w‖ = 1          (unit normalisation)
        3. (Extension point — see comment inside project())

    ─────────────────────────────────────────────────────────────────────────
    EXTENSION POINT — to replace with a true CBF-QP solver:

        from osqp import OSQP  # or cvxpy
        # Solve:
        #   min_a  ‖a - a_nominal‖²
        #   s.t.   ∂h/∂x · f(x,a) + α(h(x)) ≥ 0   (control barrier function)
        #          a_lb ≤ a ≤ a_ub
        # Replace the clip/norm body of project() with the QP solution.
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, max_power: float = 1.0, act_dim: int = 3):
        self.max_power = max_power
        self.act_dim   = act_dim

    def project(
        self,
        actions: Dict[str, np.ndarray],
        energy:  Dict[str, float],
        agents:  List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Project nominal (AI warm-start) actions to the safe feasible set.

        Parameters
        ----------
        actions : nominal actions from actors (Step 2 output + exploration noise)
        energy  : per-agent residual energy fractions
        agents  : ordered list of agent ids

        Returns
        -------
        safe_actions : feasibility-projected actions
        """
        safe = {}
        for a in agents:
            act = np.asarray(actions[a], dtype=np.float32).copy()

            # ── Constraint 1: Power ─────────────────────────────────────────
            e_i       = float(energy.get(a, 1.0))
            p_max_eff = self.max_power * max(e_i, 0.0)
            act[0]    = float(np.clip(act[0], 0.0, p_max_eff))

            # ── Constraint 2: Beamformer unit-norm ──────────────────────────
            if self.act_dim > 1:
                w = np.clip(act[1:], -1.0, 1.0)
                n = float(np.linalg.norm(w))
                act[1:] = (w / n if n >= 1e-8
                           else np.array([1.0, 0.0], dtype=np.float32))

            # ── Constraint 3: EXTENSION POINT ───────────────────────────────
            # Insert CBF / QP solver call here for formal guarantees.
            # Example: check inter-UAV separation, altitude limits, etc.

            safe[a] = act
        return safe
