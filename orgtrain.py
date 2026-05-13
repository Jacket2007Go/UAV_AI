"""
train.py  —  Hybrid MARL-ADMM training for UAV JSC
=======================================================
Supports N ∈ {2, 3, 20} UAVs as required by the paper (Section VII-D).

Architecture: Leader-UAV Centralized-Critic / Distributed-Actor (MADDPG)
─────────────────────────────────────────────────────────────────────────
  LEADER UAV (uav_0)
    • Acts as a full distributed actor — selects its own actions using its
      local policy π_θ0(a_0 | s_0) exactly like every other UAV.
    • ADDITIONALLY hosts the centralized Q-critic Q_ψ(s_1..N, a_1..N)
      during training. This critic observes the joint global state and joint
      action of the entire swarm, giving uav_0 a swarm-wide value signal.
    • Coordinates ADMM consensus: collects R_i / Q_i from all followers,
      computes the global mean z, and broadcasts z^R_i, z^Q_i back so
      every UAV can perform its local dual update (Eqs 26-28).
    • The critic is a TRAINING-TIME role only. At deployment, uav_0 runs
      its distributed actor autonomously; no centralized computation occurs.

  FOLLOWER UAVs (uav_1 … uav_{N-1})
    • Each runs its own distributed actor π_θi(a_i | s_i).
    • Reports R_i and Q_i to uav_0 for consensus each step.
    • Receives z^R_i and z^Q_i from uav_0 and updates λ_i locally.

  ADMM
    • Enforced inside env.step() every timestep.
    • Adaptive ρ schedule follows Boyd et al. §3.4.1.

Episode budgets (Section VII-D):
  N=2  → 500 episodes
  N=3  → 600 episodes
  N=20 → 1000 episodes

Usage:
    python train.py --num_uavs 3
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless runs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orgenv import UAVJSCEnv
from orgmodels import DeterministicActor, CentralizedQCritic

# ---------------------------------------------------------------------------
# CLI argument: --num_uavs N
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Hybrid MARL-ADMM UAV JSC Training")
parser.add_argument("--num_uavs", type=int, default=3,
                    choices=[2, 3, 20],
                    help="Number of UAVs (paper evaluates 2, 3, 20)")
args = parser.parse_args()
NUM_UAVS = args.num_uavs

# Episode budgets scaled to swarm size (Section VII)
EPISODE_BUDGET = {2: 500, 3: 600, 20: 1000}
episodes  = EPISODE_BUDGET[NUM_UAVS]
max_steps = 100          # horizon T per episode
batch_size = 256
start_steps = 5000       # pure random exploration before learning starts
updates_per_step = 1
gamma = 0.95
tau   = 0.01             # soft-update coefficient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] N={NUM_UAVS} UAVs | {episodes} episodes | device={device}")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env    = UAVJSCEnv(num_uavs=NUM_UAVS, motion_case="all_move")
agents = env.possible_agents[:]
N      = len(agents)

obs_dim      = env.observation_spaces[agents[0]].shape[0]
act_dim      = env.action_spaces[agents[0]].shape[0]
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
#
# LEADER UAV (uav_0) owns the centralized Q-critic.
#   leader_critic     Q_ψ(s_1..N, a_1..N)  — training only, not deployed
#   leader_critic_target  — polyak-averaged target network
#
# ALL UAVs (including uav_0) each own a distributed actor.
#   actors[a]         π_θi(a_i | s_i)      — deployed at execution time
#   actor_targets[a]  — polyak-averaged target network
# ---------------------------------------------------------------------------
leader_uav = env.leader_uav   # "uav_0"

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

# Centralized critic — owned by uav_0 (the leader UAV)
leader_critic        = CentralizedQCritic(global_obs_dim, joint_act_dim).to(device)
leader_critic_target = CentralizedQCritic(global_obs_dim, joint_act_dim).to(device)
leader_critic_target.load_state_dict(leader_critic.state_dict())
leader_critic_opt    = optim.Adam(leader_critic.parameters(), lr=1e-4)

print(f"[train] Leader UAV: {leader_uav} hosts centralized critic "
      f"({global_obs_dim}→{joint_act_dim} joint dims)")


def soft_update(net, target):
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.mul_(1 - tau).add_(tau * p.data)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def obs_to_global(obs_dict):
    return np.concatenate([obs_dict[a] for a in agents]).astype(np.float32)


def act_dict_to_joint(act_dict):
    return np.concatenate([act_dict[a] for a in agents]).astype(np.float32)


def split_global_obs(S: torch.Tensor):
    """Split batched global obs into per-agent chunks."""
    return [S[:, k * obs_dim:(k + 1) * obs_dim] for k in range(N)]


def select_actions(obs, noise_std: float):
    """Distributed action selection with Gaussian exploration noise."""
    actions = {}
    for a in agents:
        o = torch.tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act = actors[a](o).cpu().numpy()[0].astype(np.float32)

        act += np.random.normal(0.0, noise_std, size=act.shape).astype(np.float32)

        # Power ∈ [0, 1]
        act[0] = float(np.clip(act[0], 0.0, 1.0))

        # Beamformer: clip then re-normalise
        if act_dim > 1:
            w = np.clip(act[1:], -1.0, 1.0)
            n = np.linalg.norm(w)
            act[1:] = w / n if n >= 1e-8 else np.array([1.0, 0.0], dtype=np.float32)

        actions[a] = act.astype(np.float32)
    return actions


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
    """Save one figure + one .npy array per metric.

    The .npy files are required by plot_comparison.py to generate
    baseline comparison figures (mentor request).  The naming
    convention matches what plot_comparison.py expects:
        curves_N{N}_{key}.npy
    where {key} is the metric name with spaces replaced by underscores
    and parentheses/slashes stripped — e.g. "Sum Rate (bps)" becomes
    "Sum_Rate_bps", matching the key used in baseline scripts.
    """
    for name, vals in metrics.items():
        # --- figure ---
        plt.figure(figsize=(6, 4))
        plt.plot(vals)
        plt.xlabel("Episode"); plt.ylabel(name)
        plt.title(f"{name} — N={NUM_UAVS} UAVs")
        plt.tight_layout()
        plt.savefig(f"{prefix}_{name.replace(' ', '_')}.png", dpi=150)
        plt.close()

        # --- numpy array (needed by plot_comparison.py) ---
        safe_key = (name.replace(' ', '_')
                        .replace('(', '').replace(')', '')
                        .replace('/', '_'))
        np.save(f"{prefix}_{safe_key}.npy", np.array(vals, dtype=np.float32))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0

# Episode-level metrics for learning curves
all_primal_res  = []
all_dual_res    = []
all_critic_loss = []
all_sum_rate    = []
all_sum_sensing = []
all_energy_eff  = []
all_jain        = []

for ep in range(episodes):
    # Exploration noise decays from 0.20 → 0.02 over training
    noise_std = max(0.02, 0.20 * (0.993 ** ep))

    obs, _ = env.reset()
    done   = False

    ep_r      = {a: 0.0 for a in agents}
    ep_a      = {a: []  for a in agents}
    ep_pos    = {a: []  for a in agents}
    ep_E      = {a: []  for a in agents}

    ep_pr   = []; ep_dr   = []; ep_rho  = []; ep_jain = []
    ep_closs = []; ep_sr  = []; ep_sq   = []; ep_ee   = []

    for t in range(max_steps):
        global_step += 1

        actions = select_actions(obs, noise_std)
        for a in agents:
            ep_a[a].append(actions[a].copy())

        next_obs, rewards, dones, _, _ = env.step(actions)

        for a in agents:
            ep_r[a] += float(rewards[a])

        done = all(dones.values())

        # Logging
        for a in agents:
            if hasattr(env, "positions"):
                x, y = float(env.positions[a][0]), float(env.positions[a][1])
                ep_pos[a].append((x, y))
            if hasattr(env, "energy"):
                ep_E[a].append(float(env.energy[a]))

        if hasattr(env, "last_primal_residual"): ep_pr.append(float(env.last_primal_residual))
        if hasattr(env, "last_dual_residual"):   ep_dr.append(float(env.last_dual_residual))
        if hasattr(env, "rho"):                  ep_rho.append(float(env.rho))
        if hasattr(env, "last_jain"):            ep_jain.append(float(env.last_jain))

        if hasattr(env, "last_rates"):
            ep_sr.append(float(sum(env.last_rates[a]   for a in agents)))
        if hasattr(env, "last_sensing"):
            ep_sq.append(float(sum(env.last_sensing[a] for a in agents)))
        if hasattr(env, "last_powers") and hasattr(env, "last_rates"):
            tot_p = float(sum(env.last_powers[a] for a in agents))
            tot_r = float(sum(env.last_rates[a]  for a in agents))
            ep_ee.append(tot_r / (tot_p + 1e-12))

        # Store transition (team reward for joint optimisation)
        s       = obs_to_global(obs)
        s2      = obs_to_global(next_obs)
        a_joint = act_dict_to_joint(actions)
        r_team  = float(np.clip(sum(rewards[a] for a in agents), -500.0, 500.0))

        buffer.add(s, a_joint, r_team, s2, done)
        obs = next_obs

        # ------------------------------------------------------------------
        # Learning update — Hybrid MARL-ADMM (Section VI)
        #
        #   LEADER UAV (uav_0) runs the critic update:
        #     Minimise TD error on the joint state-action value estimate.
        #   ALL UAVs run actor updates:
        #     Each actor's gradient flows through uav_0's shared critic,
        #     so the swarm is coordinated through the leader's value signal.
        #   ADMM consensus:
        #     Handled inside env.step() above; uav_0 coordinates z and λ.
        # ------------------------------------------------------------------
        if buffer.size >= max(batch_size, start_steps):
            for _ in range(updates_per_step):
                S, A, R, S2, D = buffer.sample(batch_size)

                # Target joint action from target actors on next state S2
                chunks2 = split_global_obs(S2)
                with torch.no_grad():
                    A2    = torch.cat([actor_targets[a](ch)
                                       for a, ch in zip(agents, chunks2)], dim=-1)
                    q_nxt = leader_critic_target(S2, A2)
                    y     = R + gamma * (1.0 - D) * q_nxt

                # --- Leader UAV (uav_0) critic update ---
                q_pred = leader_critic(S, A)
                closs  = (q_pred - y).pow(2).mean()
                ep_closs.append(float(closs.item()))

                leader_critic_opt.zero_grad()
                closs.backward()
                torch.nn.utils.clip_grad_norm_(leader_critic.parameters(), 1.0)
                leader_critic_opt.step()

                # --- All UAVs actor update (gradient through leader critic) ---
                chunks = split_global_obs(S)
                A_pi   = torch.cat([actors[a](ch)
                                    for a, ch in zip(agents, chunks)], dim=-1)
                aloss  = -leader_critic(S, A_pi).mean()

                for a in agents:
                    actor_opts[a].zero_grad()
                aloss.backward()
                for a in agents:
                    torch.nn.utils.clip_grad_norm_(actors[a].parameters(), 1.0)
                    actor_opts[a].step()

                # Soft target updates
                for a in agents:
                    soft_update(actors[a], actor_targets[a])
                soft_update(leader_critic, leader_critic_target)

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

    # Print every 20 episodes
    if ep % 20 == 0:
        avg_pr  = np.nanmean(ep_pr)    if ep_pr    else float("nan")
        avg_dr  = np.nanmean(ep_dr)    if ep_dr    else float("nan")
        avg_rho = np.nanmean(ep_rho)   if ep_rho   else float("nan")
        avg_j   = np.nanmean(ep_jain)  if ep_jain  else float("nan")
        avg_sr  = np.nanmean(ep_sr)    if ep_sr    else float("nan")
        avg_sq  = np.nanmean(ep_sq)    if ep_sq    else float("nan")
        avg_ee  = np.nanmean(ep_ee)    if ep_ee    else float("nan")

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

        print(
            f"Ep {ep:4d}/{episodes} | N={NUM_UAVS} | noise={noise_std:.3f} | "
            f"Rwd: {r_str} | "
            f"PrRes={avg_pr:.3f} | DuRes={avg_dr:.3f} | "
            f"Jain={avg_j:.4f} | rho={avg_rho:.4f} | "
            f"zR={env.last_z_R:.3f} | zQ={env.last_z_Q:.3f} | "
            f"SumRate={avg_sr:.2f} | SumSense={avg_sq:.2f} | EE={avg_ee:.2f} | "
            f"lamR=[{lamR_str}] | lamQ=[{lamQ_str}] | "
            f"act=[{act_str}] | pos=[{pos_str}] | E=[{E_str}]"
        )

    # Save trajectory plot every 100 episodes
    if ep % 100 == 0:
        plot_trajectories_3d(
            ep_pos,
            title=f"N={NUM_UAVS} UAVs — Episode {ep} Trajectories",
            save_path=f"traj_N{NUM_UAVS}_ep{ep}.png",
        )

# ---------------------------------------------------------------------------
# Save learning curves
# ---------------------------------------------------------------------------
prefix = f"curves_N{NUM_UAVS}"
save_learning_curves({
    "Primal Residual":   all_primal_res,
    "Dual Residual":     all_dual_res,
    "Critic Loss":       all_critic_loss,
    "Sum Rate (bps)":    all_sum_rate,
    "Sum Sensing":       all_sum_sensing,
    "Energy Efficiency": all_energy_eff,
    "Jain Fairness":     all_jain,
}, prefix=prefix)

print(f"\n[train] Done. Curves saved as {prefix}_*.png")