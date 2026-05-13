"""
train.py  —  Hybrid MARL-ADMM training for UAV JSC
=======================================================
Supports N ∈ {2, 3, 20} UAVs as required by the paper (Section VII-D).

Proposal architecture — 5-step training pipeline per timestep
──────────────────────────────────────────────────────────────────────────
  Step 1  Certificate update          → env.update_certificates()
                                        (called at end of env.step)
  Step 2  AI warm start               → select_actions_warm_start()
                                        actors produce nominal action proposals
  Step 3  Certified MM-ADMM refinement→ inside env.step() ADMM block
  Step 4  Barrier/QP safe execution   → env.project_safe_actions()
                                        or SafeActionProjector before env.step
  Step 5  Realized measurement/learn  → logging + MADDPG critic/actor update

Distributed critic / rotating-prefect mapping (PDF §2-3)
──────────────────────────────────────────────────────────────────────────
  CriticCoordinator        manages active_critic_agent, duty tracking,
                           and provides critic network reference
  critic_mode='fixed'      uav_0 always hosts (backward compatible)
  critic_mode='round_robin'cycles host by global_step mod N (PDF §2)
  critic_mode='weighted'   score-based selection Q_i(t) (PDF §3)

Usage:
    python train.py --num_uavs 3 --critic_mode fixed
    python train.py --num_uavs 3 --critic_mode round_robin
    python train.py --num_uavs 3 --critic_mode weighted
    python train.py --num_uavs 3 --critic_mode weighted \\
        --alpha_E 0.3 --alpha_S 0.25 --energy_threshold 0.15
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dcenv import UAVJSCEnv
from dcmodels import (
    DeterministicActor,
    CentralizedQCritic,
    CriticCoordinatorConfig,
    WeightedCriticSelector,
    SafeActionProjector,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Hybrid MARL-ADMM UAV JSC Training")
parser.add_argument("--num_uavs", type=int, default=3, choices=[2, 3, 20],
                    help="Number of UAVs (paper evaluates 2, 3, 20)")

# ── Distributed critic mode (PDF §2-3) ────────────────────────────────────
parser.add_argument("--critic_mode", type=str, default="fixed",
                    choices=["fixed", "round_robin", "weighted"],
                    help="Critic host selection mode: "
                         "fixed=uav_0 always (default, backward compat); "
                         "round_robin=cyclic rotation (PDF §2); "
                         "weighted=score-based selection (PDF §3 Eq.2)")

# ── Weighted-mode alpha hyperparameters (PDF Eq.2) ────────────────────────
parser.add_argument("--alpha_E", type=float, default=0.25,
                    help="Weight for residual energy component Ẽ_i")
parser.add_argument("--alpha_L", type=float, default=0.20,
                    help="Weight for low-load component L̃_i")
parser.add_argument("--alpha_C", type=float, default=0.15,
                    help="Weight for capability component C̃_i")
parser.add_argument("--alpha_S", type=float, default=0.20,
                    help="Weight for channel/state quality S̃_i")
parser.add_argument("--alpha_P", type=float, default=0.10,
                    help="Weight for proximity component P̃_i")
parser.add_argument("--alpha_W", type=float, default=0.10,
                    help="Weight for willingness component W̃_i")
parser.add_argument("--energy_threshold", type=float, default=0.10,
                    help="Min energy fraction for weighted critic eligibility")
parser.add_argument("--duty_threshold", type=float, default=0.50,
                    help="Max duty share before willingness is penalised")

args     = parser.parse_args()
NUM_UAVS = args.num_uavs

# Episode budgets scaled to swarm size (Section VII)
EPISODE_BUDGET = {2: 500, 3: 600, 20: 1000}
episodes        = EPISODE_BUDGET[NUM_UAVS]
max_steps       = 100
batch_size      = 256
start_steps     = 5000
updates_per_step = 1
gamma           = 0.95
tau             = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] N={NUM_UAVS} UAVs | {episodes} episodes | device={device} | "
      f"critic_mode={args.critic_mode}")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env    = UAVJSCEnv(num_uavs=NUM_UAVS, motion_case="all_move",
                   critic_mode=args.critic_mode)
agents = env.possible_agents[:]
N      = len(agents)

# Configure weighted selector hyperparameters (only active in 'weighted' mode)
env.configure_weighted_selector(
    alpha_E=args.alpha_E,
    alpha_L=args.alpha_L,
    alpha_C=args.alpha_C,
    alpha_S=args.alpha_S,
    alpha_P=args.alpha_P,
    alpha_W=args.alpha_W,
    energy_threshold=args.energy_threshold,
    duty_threshold=args.duty_threshold,
)

obs_dim        = env.observation_spaces[agents[0]].shape[0]
act_dim        = env.action_spaces[agents[0]].shape[0]
global_obs_dim = obs_dim * N
joint_act_dim  = act_dim * N

# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0
        self.s  = np.zeros((max_size, global_obs_dim), dtype=np.float32)
        self.a  = np.zeros((max_size, joint_act_dim),  dtype=np.float32)
        self.r  = np.zeros((max_size, 1),              dtype=np.float32)
        self.s2 = np.zeros((max_size, global_obs_dim), dtype=np.float32)
        self.d  = np.zeros((max_size, 1),              dtype=np.float32)

    def add(self, s, a, r, s2, done):
        self.s [self.ptr] = s
        self.a [self.ptr] = a
        self.r [self.ptr] = r
        self.s2[self.ptr] = s2
        self.d [self.ptr] = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)
        return (
            torch.tensor(self.s [idx], dtype=torch.float32, device=device),
            torch.tensor(self.a [idx], dtype=torch.float32, device=device),
            torch.tensor(self.r [idx], dtype=torch.float32, device=device),
            torch.tensor(self.s2[idx], dtype=torch.float32, device=device),
            torch.tensor(self.d [idx], dtype=torch.float32, device=device),
        )


buffer = ReplayBuffer()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Per-agent distributed actors (Step 2: AI warm start)
actors = {
    a: DeterministicActor(obs_dim=obs_dim, w_dim=act_dim - 1).to(device)
    for a in agents
}
actor_targets = {
    a: DeterministicActor(obs_dim=obs_dim, w_dim=act_dim - 1).to(device)
    for a in agents
}
for a in agents:
    actor_targets[a].load_state_dict(actors[a].state_dict())

actor_opts = {
    a: optim.Adam(actors[a].parameters(), lr=3e-4)
    for a in agents
}

# ---------------------------------------------------------------------------
# CriticCoordinator
# ---------------------------------------------------------------------------
class CriticCoordinator:
    """
    Manages the rotating-prefect / distributed critic host.

    Proposal role: Translates the rotating-critic idea from the PDF into
    training mechanics. Maintains a SINGLE shared critic network in memory
    but tracks which UAV is the current logical host/coordinator.

    Design rationale for single-network approach:
        The paper and PDF describe a physically distributed critic; in
        simulation we keep one network for tractability (shared weights),
        while the host-rotation and duty accounting mirror the proposal
        semantics. Extension to per-agent critic networks would require
        a consensus/averaging step between critic weights (federated critic).

    Attributes:
        critic        : CentralizedQCritic — shared network (logical owner rotates)
        critic_target : target network for Polyak update
        critic_opt    : Adam optimizer
        active_agent  : current logical critic host
        mode          : 'fixed' | 'round_robin' | 'weighted'
        duty_count    : per-agent hosting counts (same as env.critic_duty_count)
    """

    def __init__(self, critic: CentralizedQCritic,
                 critic_target: CentralizedQCritic,
                 critic_opt,
                 default_agent: str,
                 mode: str = "fixed"):
        self.critic        = critic
        self.critic_target = critic_target
        self.critic_opt    = critic_opt
        self.default_agent = default_agent
        self.mode          = mode
        self.active_agent  = default_agent
        self.duty_count    = {a: 0 for a in agents}

    def step(self, env: UAVJSCEnv, global_step: int) -> str:
        """
        Update the active critic host for this step.

        Delegates to env.select_active_critic() which implements all three
        modes using the environment's state (energy, channels, duty shares).

        Returns the name of the newly selected critic host.
        """
        selected       = env.select_active_critic(global_step)
        self.active_agent = selected
        # Mirror duty counts from env (env is the authoritative source)
        self.duty_count = dict(env.critic_duty_count)
        return selected

    def duty_share_str(self) -> str:
        """Format duty shares as a compact string for logging."""
        total = max(sum(self.duty_count.values()), 1)
        parts = [f"{a}={self.duty_count.get(a,0)/total:.2f}" for a in agents]
        return ", ".join(parts)


# Instantiate critic and coordinator
critic_cfg = CriticCoordinatorConfig(
    critic_mode=args.critic_mode,
    alpha_E=args.alpha_E, alpha_L=args.alpha_L,
    alpha_C=args.alpha_C, alpha_S=args.alpha_S,
    alpha_P=args.alpha_P, alpha_W=args.alpha_W,
    energy_threshold=args.energy_threshold,
    duty_threshold=args.duty_threshold,
)

shared_critic        = CentralizedQCritic(global_obs_dim, joint_act_dim).to(device)
shared_critic_target = CentralizedQCritic(global_obs_dim, joint_act_dim).to(device)
shared_critic_target.load_state_dict(shared_critic.state_dict())
shared_critic_opt    = optim.Adam(shared_critic.parameters(), lr=1e-4)

coordinator = CriticCoordinator(
    critic=shared_critic,
    critic_target=shared_critic_target,
    critic_opt=shared_critic_opt,
    default_agent=env.default_leader_uav,
    mode=args.critic_mode,
)

# Safe action projector (Step 4 — barrier/QP safe execution, lightweight)
projector = SafeActionProjector(max_power=env.max_power, act_dim=act_dim)

print(
    f"[train] Default leader UAV: {env.default_leader_uav} | "
    f"critic_mode={args.critic_mode} | "
    f"joint dims: global_obs={global_obs_dim}, joint_act={joint_act_dim}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def soft_update(net, target):
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.mul_(1 - tau).add_(tau * p.data)


def obs_to_global(obs_dict):
    return np.concatenate([obs_dict[a] for a in agents]).astype(np.float32)


def act_dict_to_joint(act_dict):
    return np.concatenate([act_dict[a] for a in agents]).astype(np.float32)


def split_global_obs(S: torch.Tensor):
    """Split batched global obs into per-agent chunks."""
    return [S[:, k * obs_dim:(k + 1) * obs_dim] for k in range(N)]


# ---------------------------------------------------------------------------
# Step 2 — AI Warm Start: actor-based nominal action selection
# ---------------------------------------------------------------------------
def select_actions_warm_start(obs: dict, noise_std: float) -> dict:
    """
    Step 2: AI Warm Start — actors produce nominal action proposals.

    Each distributed actor π_θi(a_i | s_i) takes local observation s_i and
    outputs a nominal action a_i. Gaussian exploration noise is added during
    training. The result is a *proposal* that will be safety-projected
    (Step 4) before being passed to env.step() (Step 3).

    Returns: nominal_actions dict {agent: np.ndarray}
    """
    nominal = {}
    for a in agents:
        o = torch.tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act = actors[a](o).cpu().numpy()[0].astype(np.float32)

        # Add exploration noise
        act += np.random.normal(0.0, noise_std, size=act.shape).astype(np.float32)

        # Soft clip before projector (projector will enforce hard bounds)
        act[0] = float(np.clip(act[0], 0.0, 1.0))
        if act_dim > 1:
            w = np.clip(act[1:], -1.0, 1.0)
            n = np.linalg.norm(w)
            act[1:] = w / n if n >= 1e-8 else np.array([1.0, 0.0], dtype=np.float32)

        nominal[a] = act.astype(np.float32)
    return nominal


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_trajectories_3d(ep_pos, title="UAV Trajectories", save_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    for a, traj in ep_pos.items():
        traj = np.array(traj)
        if len(traj) == 0:
            continue
        t = np.arange(len(traj))
        ax.plot(traj[:, 0], traj[:, 1], t, label=a)
        ax.scatter(traj[0, 0],  traj[0, 1],  0,      marker="o")
        ax.scatter(traj[-1, 0], traj[-1, 1], t[-1],  marker="^")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Time step")
    ax.set_title(title); ax.legend(fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); plt.close()
    else:
        plt.show()


def save_learning_curves(metrics: dict, prefix: str):
    """
    Save one figure + one .npy array per metric.
    Naming convention matches plot_comparison.py expectations.
    """
    for name, vals in metrics.items():
        plt.figure(figsize=(6, 4))
        plt.plot(vals)
        plt.xlabel("Episode"); plt.ylabel(name)
        plt.title(f"{name} — N={NUM_UAVS} UAVs | critic_mode={args.critic_mode}")
        plt.tight_layout()
        plt.savefig(f"{prefix}_{name.replace(' ', '_')}.png", dpi=150)
        plt.close()

        safe_key = (name.replace(' ', '_')
                        .replace('(', '').replace(')', '')
                        .replace('/', '_'))
        np.save(f"{prefix}_{safe_key}.npy", np.array(vals, dtype=np.float32))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0

# Episode-level metrics
all_primal_res   = []
all_dual_res     = []
all_critic_loss  = []
all_sum_rate     = []
all_sum_sensing  = []
all_energy_eff   = []
all_jain         = []
all_cert_comm    = []   # NEW: mean communication deficit certificate
all_cert_sense   = []   # NEW: mean sensing deficit certificate
all_active_critic_hist = []  # NEW: active critic host per episode (for analysis)

for ep in range(episodes):
    # Exploration noise decays 0.20 → 0.02 over training
    noise_std = max(0.02, 0.20 * (0.993 ** ep))

    obs, _ = env.reset()
    done   = False

    ep_r   = {a: 0.0 for a in agents}
    ep_a   = {a: []  for a in agents}
    ep_pos = {a: []  for a in agents}
    ep_E   = {a: []  for a in agents}

    ep_pr = []; ep_dr  = []; ep_rho   = []; ep_jain  = []
    ep_closs = []; ep_sr = []; ep_sq  = []; ep_ee    = []
    ep_cert_comm  = []   # step-level certificate logs
    ep_cert_sense = []

    for t in range(max_steps):
        global_step += 1

        # Critic selection: update once per step for stability.
        coordinator.step(env, global_step)
        active_critic = env.active_critic_agent

        # ──────────────────────────────────────────────────────────────────
        # Step 2: AI Warm Start
        # Actors produce nominal action proposals from local observations.
        # ──────────────────────────────────────────────────────────────────
        nominal_actions = select_actions_warm_start(obs, noise_std)

        for a in agents:
            ep_a[a].append(nominal_actions[a].copy())

        # ──────────────────────────────────────────────────────────────────
        # Step 4: Barrier/QP Safe Execution (lightweight projection)
        # Project nominal actions to feasible set before env transition.
        # Note: The projector applies energy-aware power capping and
        # beamformer normalization. For a true CBF/QP solver, replace
        # projector.project() or env.project_safe_actions() accordingly.
        # ──────────────────────────────────────────────────────────────────
        safe_actions = projector.project(nominal_actions, env.energy, agents)
        # Equivalent alternative using env method:
        # safe_actions = env.project_safe_actions(nominal_actions)

        # ──────────────────────────────────────────────────────────────────
        # Step 3: Certified MM-ADMM Refinement (inside env.step)
        # Step 1: Certificate update (inside env.step at end)
        # Step 5: Realized measurement (logging below)
        # ──────────────────────────────────────────────────────────────────
        next_obs, rewards, dones, _, _ = env.step(safe_actions)

        for a in agents:
            ep_r[a] += float(rewards[a])
        done = all(dones.values())

        # ── Logging ───────────────────────────────────────────────────────
        for a in agents:
            if hasattr(env, "positions"):
                x, y = float(env.positions[a][0]), float(env.positions[a][1])
                ep_pos[a].append((x, y))
            if hasattr(env, "energy"):
                ep_E[a].append(float(env.energy[a]))

        if hasattr(env, "last_primal_residual"):
            ep_pr.append(float(env.last_primal_residual))
        if hasattr(env, "last_dual_residual"):
            ep_dr.append(float(env.last_dual_residual))
        if hasattr(env, "rho"):
            ep_rho.append(float(env.rho))
        if hasattr(env, "last_jain"):
            ep_jain.append(float(env.last_jain))
        if hasattr(env, "last_rates"):
            ep_sr.append(float(sum(env.last_rates[a]   for a in agents)))
        if hasattr(env, "last_sensing"):
            ep_sq.append(float(sum(env.last_sensing[a] for a in agents)))
        if hasattr(env, "last_powers") and hasattr(env, "last_rates"):
            tot_p = float(sum(env.last_powers[a] for a in agents))
            tot_r = float(sum(env.last_rates[a]  for a in agents))
            ep_ee.append(tot_r / (tot_p + 1e-12))

        # Certificate logging (Step 1 / Step 5)
        ep_cert_comm.append(
            float(np.mean(list(env.cert_comm_deficit.values()))))
        ep_cert_sense.append(
            float(np.mean(list(env.cert_sense_deficit.values()))))

        # Store transition
        s       = obs_to_global(obs)
        s2      = obs_to_global(next_obs)
        a_joint = act_dict_to_joint(safe_actions)
        r_team  = float(np.clip(sum(rewards[a] for a in agents), -500.0, 500.0))

        buffer.add(s, a_joint, r_team, s2, done)
        obs = next_obs

        # ──────────────────────────────────────────────────────────────────
        # Step 5: Realized Measurement and Learning
        #
        # The active critic host (coordinator.active_agent) coordinates the
        # value signal used to update all actors. In simulation we use the
        # shared critic network regardless of host; the coordinator semantics
        # make host rotation explicit and log the duty distribution.
        #
        # To extend to truly federated critics (one network per UAV with
        # periodic weight averaging), replace shared_critic with a per-agent
        # critic and add an averaging step after each update.
        # ──────────────────────────────────────────────────────────────────
        if buffer.size >= max(batch_size, start_steps):
            for _ in range(updates_per_step):
                S, A, R, S2, D = buffer.sample(batch_size)

                # Target joint action from target actors on next state S2
                chunks2 = split_global_obs(S2)
                with torch.no_grad():
                    A2    = torch.cat([actor_targets[a](ch)
                                       for a, ch in zip(agents, chunks2)], dim=-1)
                    q_nxt = coordinator.critic_target(S2, A2)
                    y     = R + gamma * (1.0 - D) * q_nxt

                # ── Critic update (logical host: coordinator.active_agent) ─
                q_pred = coordinator.critic(S, A)
                closs  = (q_pred - y).pow(2).mean()
                ep_closs.append(float(closs.item()))

                coordinator.critic_opt.zero_grad()
                closs.backward()
                torch.nn.utils.clip_grad_norm_(
                    coordinator.critic.parameters(), 1.0)
                coordinator.critic_opt.step()

                # ── Actor update (gradient through shared/active critic) ───
                chunks = split_global_obs(S)
                A_pi   = torch.cat([actors[a](ch)
                                    for a, ch in zip(agents, chunks)], dim=-1)
                aloss  = -coordinator.critic(S, A_pi).mean()

                for a in agents:
                    actor_opts[a].zero_grad()
                aloss.backward()
                for a in agents:
                    torch.nn.utils.clip_grad_norm_(actors[a].parameters(), 1.0)
                    actor_opts[a].step()

                # Soft target updates
                for a in agents:
                    soft_update(actors[a], actor_targets[a])
                soft_update(coordinator.critic, coordinator.critic_target)

        if done:
            break

    # Episode-level aggregates
    all_primal_res .append(np.nanmean(ep_pr)    if ep_pr    else np.nan)
    all_dual_res   .append(np.nanmean(ep_dr)    if ep_dr    else np.nan)
    all_critic_loss.append(np.nanmean(ep_closs) if ep_closs else np.nan)
    all_sum_rate   .append(np.nanmean(ep_sr)    if ep_sr    else np.nan)
    all_sum_sensing.append(np.nanmean(ep_sq)    if ep_sq    else np.nan)
    all_energy_eff .append(np.nanmean(ep_ee)    if ep_ee    else np.nan)
    all_jain       .append(np.nanmean(ep_jain)  if ep_jain  else np.nan)
    all_cert_comm  .append(np.nanmean(ep_cert_comm)  if ep_cert_comm  else np.nan)
    all_cert_sense .append(np.nanmean(ep_cert_sense) if ep_cert_sense else np.nan)
    all_active_critic_hist.append(coordinator.active_agent)

    # Print diagnostics every 20 episodes
    if ep % 20 == 0:
        avg_pr  = np.nanmean(ep_pr)    if ep_pr    else float("nan")
        avg_dr  = np.nanmean(ep_dr)    if ep_dr    else float("nan")
        avg_rho = np.nanmean(ep_rho)   if ep_rho   else float("nan")
        avg_j   = np.nanmean(ep_jain)  if ep_jain  else float("nan")
        avg_sr  = np.nanmean(ep_sr)    if ep_sr    else float("nan")
        avg_sq  = np.nanmean(ep_sq)    if ep_sq    else float("nan")
        avg_ee  = np.nanmean(ep_ee)    if ep_ee    else float("nan")
        avg_cert_c = np.nanmean(ep_cert_comm)  if ep_cert_comm  else float("nan")
        avg_cert_s = np.nanmean(ep_cert_sense) if ep_cert_sense else float("nan")

        r_str    = ", ".join([f"{a}={ep_r[a]:.1f}" for a in agents])
        lamR_str = ", ".join([f"{a}={env.lambda_R[a]:.3f}" for a in agents])
        lamQ_str = ", ".join([f"{a}={env.lambda_Q[a]:.3f}" for a in agents])
        act_str  = ", ".join([
            f"{a}=P:{np.mean([u[0] for u in ep_a[a]]):.3f}" for a in agents])

        if all(len(ep_pos[a]) > 0 for a in agents):
            pos_str = ", ".join([
                f"{a}=({ep_pos[a][0][0]:.0f},{ep_pos[a][0][1]:.0f})"
                f"→({ep_pos[a][-1][0]:.0f},{ep_pos[a][-1][1]:.0f})"
                for a in agents])
        else:
            pos_str = "n/a"

        E_str = ", ".join([f"{a}={env.energy[a]:.3f}" for a in agents])

        # Duty share summary
        duty_str = coordinator.duty_share_str()

        # Weighted scores (only meaningful in 'weighted' mode)
        if args.critic_mode == "weighted" and env.last_weighted_scores:
            wscore_str = ", ".join([
                f"{a}={env.last_weighted_scores.get(a, 0.0):.3f}"
                for a in agents])
        else:
            wscore_str = "n/a"

        print(
            f"Ep {ep:4d}/{episodes} | N={NUM_UAVS} | noise={noise_std:.3f} | "
            f"critic_mode={args.critic_mode} | "
            f"active_critic={coordinator.active_agent} | "
            f"duty=[{duty_str}] | "
            f"Rwd: {r_str} | "
            f"PrRes={avg_pr:.3f} | DuRes={avg_dr:.3f} | "
            f"Jain={avg_j:.4f} | rho={avg_rho:.4f} | "
            f"zR={env.last_z_R:.3f} | zQ={env.last_z_Q:.3f} | "
            f"SumRate={avg_sr:.2f} | SumSense={avg_sq:.2f} | EE={avg_ee:.2f} | "
            f"cert_comm={avg_cert_c:.4f} | cert_sense={avg_cert_s:.4f} | "
            f"wscores=[{wscore_str}] | "
            f"lamR=[{lamR_str}] | lamQ=[{lamQ_str}] | "
            f"act=[{act_str}] | pos=[{pos_str}] | E=[{E_str}]"
        )

    # Save trajectory plot every 100 episodes
    if ep % 100 == 0:
        plot_trajectories_3d(
            ep_pos,
            title=f"N={NUM_UAVS} UAVs — Ep {ep} | critic_mode={args.critic_mode}",
            save_path=f"traj_N{NUM_UAVS}_ep{ep}.png",
        )

# ---------------------------------------------------------------------------
# Save learning curves
# ---------------------------------------------------------------------------
prefix = f"curves_N{NUM_UAVS}_{args.critic_mode}"
save_learning_curves({
    "Primal Residual":       all_primal_res,
    "Dual Residual":         all_dual_res,
    "Critic Loss":           all_critic_loss,
    "Sum Rate (bps)":        all_sum_rate,
    "Sum Sensing":           all_sum_sensing,
    "Energy Efficiency":     all_energy_eff,
    "Jain Fairness":         all_jain,
    "Cert Comm Deficit":     all_cert_comm,   # NEW: Step 1 certificate
    "Cert Sense Deficit":    all_cert_sense,  # NEW: Step 1 certificate
}, prefix=prefix)

# Save critic selection history
np.save(f"{prefix}_active_critic_hist.npy",
        np.array(all_active_critic_hist, dtype=object))

print(
    f"\n[train] Done. Curves saved as {prefix}_*.png | "
    f"critic_mode={args.critic_mode} | "
    f"final active_critic={coordinator.active_agent} | "
    f"duty_shares=[{coordinator.duty_share_str()}]"
)
