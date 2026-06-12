"""
train_model.py  —  Unified trainer for all 4 UAV JSC model variants
=====================================================================

Model variants
──────────────────────────────────────────────────────────────────────
  original  orgenv  + fixed critic (uav_0) + 12-dim obs + no cert reward
  cert      certenv + fixed critic         + 15-dim obs + cert reward
  dc        dcenv   + weighted critic      + 12-dim obs + no cert reward
  dccert    certenv + weighted critic      + 15-dim obs + cert reward

Proposal architecture (5-step loop) — all variants share the same loop:
  Step 1  Certificate update   → env.update_certificates() (inside env.step)
  Step 2  AI warm start        → select_actions() — actors produce proposals
  Step 3  Certified MM-ADMM    → inside env.step() ADMM block
  Step 4  Barrier/QP projection→ SafeActionProjector (dc / cert / dccert)
  Step 5  Realized measurement → MADDPG critic + actor update

Results saved to:
  {results_dir}/{model_type}_N{N}_s{seed}_{metric}.npy

Usage:
  python train_model.py --model_type original --num_uavs 3
  python train_model.py --model_type cert     --num_uavs 3 --seed 1
  python train_model.py --model_type dc       --num_uavs 3 --seed 2
  python train_model.py --model_type dccert   --num_uavs 3
  python train_model.py --model_type original --num_uavs 3 --episodes 200
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="UAV JSC Unified Trainer")
parser.add_argument("--model_type", type=str, default="original",
                    choices=["original", "cert", "dc", "dccert"],
                    help="Model variant: original | cert | dc | dccert")
parser.add_argument("--num_uavs", type=int, default=3,
                    help="Swarm size (paper evaluates 2, 3, 5, 20)")
parser.add_argument("--episodes", type=int, default=None,
                    help="Episode budget override (default: paper schedule)")
parser.add_argument("--results_dir", type=str, default="results",
                    help="Output directory for .npy curve files")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility (affects np and torch)")
# Cert hyperparameters (only active for cert / dccert)
parser.add_argument("--cert_R_weight", type=float, default=2.0,
                    help="DEAD parameter retained for compatibility")
parser.add_argument("--cert_Q_weight", type=float, default=1.0,
                    help="DEAD parameter retained for compatibility")
# Certificate hyperparameters (review §4.1/§4.2/§5.4/§5.6)
parser.add_argument("--cert_tau", type=float, default=None,
                    help="Legacy single slack: sets BOTH tau_R and tau_S if given")
parser.add_argument("--cert_tau_R", type=float, default=0.25,
                    help="Relative rate slack tau_R (review §4.2)")
parser.add_argument("--cert_tau_S", type=float, default=0.25,
                    help="Relative sensing slack tau_S (review §4.2)")
parser.add_argument("--cert_R_abs", type=float, default=0.5,
                    help="Absolute mission rate floor R_abs in bps/Hz; 0 = relative-only ablation")
parser.add_argument("--cert_S_abs", type=float, default=0.02,
                    help="Absolute mission sensing floor S_abs; 0 = relative-only ablation")
parser.add_argument("--cert_E_min", type=float, default=0.10,
                    help="Energy gate E_min (review §4.1)")
parser.add_argument("--cert_d_min", type=float, default=114.0,
                    help="Absolute safety-distance gate d_min in env units (review §4.1)")
parser.add_argument("--cert_mode", type=str, default="soft",
                    choices=["soft", "hard"],
                    help="soft: magnitude-scaled penalty p_i^cert (review §5.6); "
                         "hard: legacy constant per-failure penalty")
parser.add_argument("--cert_lambda_R", type=float, default=0.30)
parser.add_argument("--cert_lambda_S", type=float, default=10.0)
parser.add_argument("--cert_lambda_E", type=float, default=2.0)
parser.add_argument("--cert_lambda_D", type=float, default=0.01)
parser.add_argument("--cert_alpha_C",  type=float, default=1.0,
                    help="alpha_C multiplier on p_i^cert in the reward (review §5.7)")
parser.add_argument("--cert_rho_R", type=float, default=0.95,
                    help="EMA persistence rho_R for D_i^R (review §5.5)")
parser.add_argument("--cert_rho_S", type=float, default=0.95,
                    help="EMA persistence rho_S for D_i^S (review §5.5)")
parser.add_argument("--cert_alpha", type=float, default=0.40,
                    help="Legacy relative proximity threshold (logged only)")
parser.add_argument("--cert_lambda", type=float, default=0.5,
                    help="Hard-mode constant penalty multiplier (× reward_scale)")
# Sensing model (review §4.4/§5.3)
parser.add_argument("--sensing_metric", type=str, default="fim_min_eig",
                    choices=["fim_min_eig", "crb_trace", "snr"],
                    help="Scalar sensing quality S_i: lambda_min(FIM), 1/tr(CRB), or legacy SNR proxy")
# Joint fairness weights (review §5.9)
parser.add_argument("--omega_R", type=float, default=0.5,
                    help="J_JSC weight on J_R (omega_R + omega_S = 1)")
parser.add_argument("--omega_S", type=float, default=0.5,
                    help="J_JSC weight on J_S")
# Critic rotation interval K (was hardcoded MIN_TENURE = 200)
parser.add_argument("--rotation_interval", type=int, default=200,
                    help="Minimum consecutive hosting episodes K before re-selection")
# Critic-host eligibility set E_set(t) extras (review §5.8)
parser.add_argument("--load_max", type=float, default=1.0,
                    help="Duty-share eligibility cap l_max")
parser.add_argument("--sync_rate_min", type=float, default=0.0,
                    help="Host-to-swarm sync link-rate floor R_min^sync (bps)")
# Weighted critic hyperparameters (only active for dc / dccert)
parser.add_argument("--alpha_E", type=float, default=0.25)
parser.add_argument("--alpha_L", type=float, default=0.20)
parser.add_argument("--alpha_C", type=float, default=0.15)
parser.add_argument("--alpha_S", type=float, default=0.20)
parser.add_argument("--alpha_P", type=float, default=0.10)
parser.add_argument("--alpha_W", type=float, default=0.10)
parser.add_argument("--energy_threshold", type=float, default=0.10)
parser.add_argument("--duty_threshold",   type=float, default=0.50)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Per-variant configuration
# ---------------------------------------------------------------------------
# env_type  : which environment class to instantiate
# obs_dim   : per-agent observation dimension
# critic_mode: 'fixed' (uav_0 always) or 'weighted' (rotating prefect)
# use_proj  : whether to apply SafeActionProjector (Step 4)
MODEL_CONFIGS = {
    # NOTE (review §4.3/§4.7): the Original baseline now runs on dcenv with
    # critic_mode='fixed' (static critic at uav_0, no projection, no certs).
    # dcenv in fixed mode is architecturally identical to the legacy orgenv
    # baseline, but carries the corrected link budget (thermal noise N0*B,
    # bps rates, FIM sensing, normalized rewards). All four variants must
    # share the same physics or the paired deltas dJ_m = V4 - V1 are
    # meaningless. orgenv.py is retained only as a legacy reference.
    "original": dict(env_type="dc",   obs_dim=12, critic_mode="fixed",    use_proj=False),
    "cert":     dict(env_type="cert", obs_dim=15, critic_mode="fixed",    use_proj=True),
    "dc":       dict(env_type="dc",   obs_dim=12, critic_mode="weighted", use_proj=True),
    "dccert":   dict(env_type="cert", obs_dim=15, critic_mode="weighted", use_proj=True),
}
cfg = MODEL_CONFIGS[args.model_type]

# Episode budgets from the paper (Section VII-D).
# .get() with N=600 fallback so arbitrary N doesn't KeyError.
EPISODE_BUDGET = {2: 500, 3: 600, 5: 1000, 20: 1000}
NUM_UAVS  = args.num_uavs
episodes  = args.episodes if args.episodes is not None else EPISODE_BUDGET.get(NUM_UAVS, 600)
max_steps = 100
batch_size       = 512
start_steps      = 5000
updates_per_step = 1
gamma = 0.95
tau   = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scale critic hidden dimension with swarm size to prevent underfitting.
# At N=20, critic input is 300-dim (240 obs + 60 act). With only 128 hidden
# units that is a 2.3x compression — the network cannot represent Q accurately,
# causing the loss spike seen in Figure 4. Scaling to 512 at N=20 fixes this.
# N≤5 → 128, N=10 → 256, N=20 → 512
CRITIC_HIDDEN = min(128 * max(1, NUM_UAVS // 5), 512)
print(f"[train] critic_hidden_dim={CRITIC_HIDDEN} (scaled for N={NUM_UAVS})")
os.makedirs(args.results_dir, exist_ok=True)

print(f"[train] model={args.model_type} | N={NUM_UAVS} | seed={args.seed} | "
      f"episodes={episodes} | batch={batch_size} | critic_mode={cfg['critic_mode']} | "
      f"obs_dim={cfg['obs_dim']} | device={device}")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if cfg["env_type"] == "org":
    from orgenv import UAVJSCEnv
    env = UAVJSCEnv(num_uavs=NUM_UAVS, motion_case="all_move")

elif cfg["env_type"] == "dc":
    from dcenv import UAVJSCEnv
    env = UAVJSCEnv(num_uavs=NUM_UAVS, motion_case="all_move",
                    critic_mode=cfg["critic_mode"])
    if cfg["critic_mode"] == "weighted":
        env.configure_weighted_selector(
            alpha_E=args.alpha_E, alpha_L=args.alpha_L,
            alpha_C=args.alpha_C, alpha_S=args.alpha_S,
            alpha_P=args.alpha_P, alpha_W=args.alpha_W,
            energy_threshold=args.energy_threshold,
            duty_threshold=args.duty_threshold,
            load_max=args.load_max,
            sync_rate_min=args.sync_rate_min,
        )

elif cfg["env_type"] == "cert":
    from certenv import CertUAVJSCEnv
    env = CertUAVJSCEnv(
        num_uavs=NUM_UAVS, motion_case="all_move",
        critic_mode=cfg["critic_mode"],
        cert_tau      = args.cert_tau,        # legacy: overrides both taus
        cert_tau_R    = args.cert_tau_R,
        cert_tau_S    = args.cert_tau_S,
        cert_R_abs    = args.cert_R_abs,
        cert_S_abs    = args.cert_S_abs,
        cert_E_min    = args.cert_E_min,
        cert_d_min    = args.cert_d_min,
        cert_mode     = args.cert_mode,
        cert_lambda_R = args.cert_lambda_R,
        cert_lambda_S = args.cert_lambda_S,
        cert_lambda_E = args.cert_lambda_E,
        cert_lambda_D = args.cert_lambda_D,
        cert_alpha_C  = args.cert_alpha_C,
        cert_rho_R    = args.cert_rho_R,
        cert_rho_S    = args.cert_rho_S,
        cert_alpha    = args.cert_alpha,
        cert_lambda   = args.cert_lambda,
        cert_R_weight = args.cert_R_weight,   # dead — for compatibility
        cert_Q_weight = args.cert_Q_weight,   # dead — for compatibility
    )
    # NOTE: NO configure_fairness_weights() override here.
    # Previously this used eta_R=6.0, eta_Q=3.0 (2x the default 3.0/1.5)
    # to compensate for cert reward shaping. With the Jain-amp reward
    # removed (certenv.step() now pure pass-through), cert variants must
    # use the SAME eta_R/eta_Q as Original/DC for a fair ablation.
    # Otherwise Cert vs Original is testing two things at once: the cert
    # obs features AND a tighter consensus penalty.
    if cfg["critic_mode"] == "weighted":
        env.configure_weighted_selector(
            alpha_E=args.alpha_E, alpha_L=args.alpha_L,
            alpha_C=args.alpha_C, alpha_S=args.alpha_S,
            alpha_P=args.alpha_P, alpha_W=args.alpha_W,
            energy_threshold=args.energy_threshold,
            duty_threshold=args.duty_threshold,
            load_max=args.load_max,
            sync_rate_min=args.sync_rate_min,
        )

# Sensing metric (review §4.4/§5.3) and joint-fairness weights (review §5.9)
# apply uniformly across all env variants so the comparison stays an ablation.
env.sensing_metric = args.sensing_metric
env.omega_R_jain   = args.omega_R
env.omega_S_jain   = args.omega_S

agents        = env.possible_agents[:]
N             = len(agents)
obs_dim       = cfg["obs_dim"]
act_dim       = env.action_spaces[agents[0]].shape[0]
global_obs_dim = obs_dim * N
joint_act_dim  = act_dim * N

# dcmodels is a strict superset of orgmodels — use it for all variants
from dcmodels import DeterministicActor, CentralizedQCritic, SafeActionProjector

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
        self.w  = np.ones( (max_size, 1),              dtype=np.float32)

    def add(self, s, a, r, s2, done, w=1.0):
        self.s [self.ptr] = s
        self.a [self.ptr] = a
        self.r [self.ptr] = r
        self.s2[self.ptr] = s2
        self.d [self.ptr] = float(done)
        self.w [self.ptr] = float(w)
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
            torch.tensor(self.w [idx], dtype=torch.float32, device=device),
        )


buffer = ReplayBuffer()

# ---------------------------------------------------------------------------
# Models
# Actor: π_θi(a_i | s_i) — per-agent distributed actor (Step 2)
# Critic: Q_ψ(s_global, a_joint) — centralized value estimate (Step 5)
# ---------------------------------------------------------------------------
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

critic        = CentralizedQCritic(global_obs_dim, joint_act_dim, hidden_dim=CRITIC_HIDDEN).to(device)
critic_target = CentralizedQCritic(global_obs_dim, joint_act_dim, hidden_dim=CRITIC_HIDDEN).to(device)
critic_target.load_state_dict(critic.state_dict())
critic_opt    = optim.Adam(critic.parameters(), lr=1e-4)

# Step 4: safe action projector (used by cert / dc / dccert)
projector = SafeActionProjector(max_power=env.max_power, act_dim=act_dim)

# Review §4.5: |psi| = critic model size in bits (float32 weights) for the
# migration energy model E_sync = P_tx * |psi| / R_{i->j}.
if hasattr(env, "psi_bits"):
    env.psi_bits = 32.0 * float(sum(p.numel() for p in critic.parameters()))
    print(f"[train] critic size |psi| = {env.psi_bits/8/1024:.1f} kB "
          f"({env.psi_bits:.3g} bits) — used for E_sync")

print(f"[train] global_obs_dim={global_obs_dim} | joint_act_dim={joint_act_dim} | "
      f"actor params={sum(p.numel() for p in actors[agents[0]].parameters()):,} | "
      f"critic params={sum(p.numel() for p in critic.parameters()):,}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def soft_update(net, target):
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.mul_(1 - tau).add_(tau * p.data)


def obs_to_global(obs_dict: dict) -> np.ndarray:
    return np.concatenate([obs_dict[a] for a in agents]).astype(np.float32)


def act_dict_to_joint(act_dict: dict) -> np.ndarray:
    return np.concatenate([act_dict[a] for a in agents]).astype(np.float32)


def split_global_obs(S: torch.Tensor):
    """Split batched global obs tensor into per-agent observation chunks."""
    return [S[:, k * obs_dim:(k + 1) * obs_dim] for k in range(N)]


def select_actions(obs: dict, noise_std: float) -> dict:
    """
    Step 2 — AI Warm Start.
    Each distributed actor proposes a nominal action from local observation.
    Gaussian exploration noise is added during training.
    """
    actions = {}
    for a in agents:
        o = torch.tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act = actors[a](o).cpu().numpy()[0].astype(np.float32)
        act += np.random.normal(0.0, noise_std, size=act.shape).astype(np.float32)
        # Soft clip before projector
        act[0] = float(np.clip(act[0], 0.0, 1.0))
        if act_dim > 1:
            w = np.clip(act[1:], -1.0, 1.0)
            nw = np.linalg.norm(w)
            act[1:] = w / nw if nw >= 1e-8 else np.array([1.0, 0.0], dtype=np.float32)
        actions[a] = act.astype(np.float32)
    return actions

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0

all_primal_res   = []
all_dual_res     = []
all_critic_loss  = []
all_sum_rate     = []
all_sum_sensing  = []
all_energy_eff   = []
all_jain         = []
all_cert_comm    = []   # cert variants: mean communication deficit
all_cert_sense   = []   # cert variants: mean sensing deficit
# Review §5.9 / §6.3 metrics (all variants)
all_jain_S       = []   # sensing Jain J_S
all_jain_JSC     = []   # joint Jain J_JSC
all_min_rate     = []   # min_i R_i (bps)
all_min_sensing  = []   # min_i S_i
all_sum_eta      = []   # sum spectral efficiency (bps/Hz)
all_energy_imb   = []   # I_E energy-imbalance (end of episode)
all_comp_energy  = []   # cumulative critic-hosting compute energy
all_sync_energy  = []   # cumulative critic-migration sync energy
# Review §5.6 / §6.3 certificate metrics (cert variants; NaN otherwise)
all_m_qos        = []   # M_QoS minimum-service metric
all_cert_viol    = []   # mean violation magnitude (not just pass/fail)
all_cert_soft    = []   # mean soft certificate value C~

# ── Hard-certificate diagnostic logging (cert variants only) ──────────────
# Per-episode aggregates suitable for diagnose_certificate.py:
#   all_cert_issued   : episode-mean count of agents passing cert this step
#   all_cert_pass_rate: same as above but normalised by N (∈ [0, 1])
#   all_cert_comm_pass / sense_pass / prox_pass: episode-mean fraction of
#       agents passing each individual criterion (diagnostic for which
#       criterion is the binding constraint)
#   all_cert_min_dist : episode-mean min pairwise distance
#   all_cert_max_dist : episode-mean max pairwise distance
#   all_cert_thr_*    : episode-mean swarm-tuned thresholds
#   all_per_agent_*   : (episodes, N) arrays — per-agent rate / sense /
#       penalty / pass-fraction. Used for the per-agent bar charts.
all_cert_issued     = []
all_cert_pass_rate  = []
all_cert_comm_pass  = []
all_cert_sense_pass = []
all_cert_prox_pass  = []
all_cert_min_dist   = []
all_cert_max_dist   = []
all_cert_thr_comm   = []
all_cert_thr_sense  = []
all_cert_thr_prox   = []
all_per_agent_rate    = []   # list of (N,) arrays → stacked at save time
all_per_agent_sense   = []
all_per_agent_penalty = []
all_per_agent_pass    = []   # fraction of steps each agent passed cert

# ── K=200 tenure gate for DC variants ─────────────────────────────────────
# Based on the rotation sweep experiment, K=200 produces the best Jain index.
# Tenure gate: critic host is held for a minimum of K episodes before the
# weighted selector is allowed to reconsider (Option A — re-evaluates every
# episode once K is met, resets on switch, increments on re-selection).
MIN_TENURE    = int(args.rotation_interval)   # K (review: swept hyperparameter)
critic_tenure = 0   # episodes current host has served consecutively

for ep in range(episodes):
    # Exploration noise decays 0.20 → 0.02 over training
    noise_std = max(0.02, 0.20 * (0.993 ** ep))

    # --- K=200 tenure-gated critic selection (dc / dccert only) ---
    # Fixed-critic variants (original, cert) skip this block entirely.
    #
    # Rotation policy: the selector is consulted ONLY at tenure boundaries
    # (every MIN_TENURE episodes), not every episode. Once a new host is
    # selected, it serves for at least MIN_TENURE more episodes before
    # the selector runs again.
    #
    # This matches the conversation summary's K=200 finding: "long tenure
    # allows full critic convergence before switching." The previous code
    # consulted the selector every episode after the first tenure expired,
    # which let the host flip-flop on borderline scores and never gave the
    # policy stable training data after the first rotation.
    if cfg["critic_mode"] != "fixed" and hasattr(env, "select_active_critic"):
        prev_critic = env.active_critic_agent
        if critic_tenure < MIN_TENURE:
            # Hold current host — manually advance duty counters
            env.critic_selection_step += 1
            env.critic_duty_count[env.active_critic_agent] = (
                env.critic_duty_count.get(env.active_critic_agent, 0) + 1)
            total = max(env.critic_selection_step, 1)
            for a in agents:
                env.critic_duty_share[a] = (
                    env.critic_duty_count.get(a, 0) / total)
            critic_tenure += 1
        else:
            # Tenure met — selector consulted exactly once, then we lock
            # in the result for another full MIN_TENURE.
            env.select_active_critic(global_step=ep)
            critic_tenure = 0   # always reset, whether host changed or not
            # Review §4.5: charge migration energy E_sync = P_tx|psi|/R_{i->j}
            # to the old host when the critic actually moves.
            if (env.active_critic_agent != prev_critic
                    and hasattr(env, "charge_critic_sync")):
                e_sync = env.charge_critic_sync(prev_critic,
                                                env.active_critic_agent)
                if e_sync > 0:
                    print(f"  [rotate] ep {ep}: critic {prev_critic} -> "
                          f"{env.active_critic_agent} | E_sync={e_sync:.5f}")

    obs, _ = env.reset()
    done   = False

    ep_pr = []; ep_dr  = []; ep_jain = []
    ep_closs = []; ep_sr = []; ep_sq = []; ep_ee = []
    ep_cert_c = []; ep_cert_s = []
    ep_jain_S = []; ep_jain_JSC = []
    ep_min_r  = []; ep_min_s    = []; ep_eta = []
    ep_mqos   = []; ep_viol     = []; ep_soft = []
    # Hard-cert per-step accumulators (only populated if env exposes them)
    ep_issued     = []
    ep_comm_pass  = []
    ep_sense_pass = []
    ep_prox_pass  = []
    ep_min_dist   = []
    ep_max_dist   = []
    ep_thr_c      = []
    ep_thr_s      = []
    ep_thr_p      = []
    # Per-agent per-step accumulators (averaged over the episode at the end)
    ep_pa_rate    = {a: [] for a in agents}
    ep_pa_sense   = {a: [] for a in agents}
    ep_pa_penalty = {a: [] for a in agents}
    ep_pa_pass    = {a: [] for a in agents}
    ep_r = {a: 0.0 for a in agents}

    for t in range(max_steps):
        global_step += 1

        # --- Step 2: AI warm start ---
        nominal_actions = select_actions(obs, noise_std)

        # --- Step 4: Barrier / QP safe projection ---
        if cfg["use_proj"]:
            safe_actions = projector.project(nominal_actions, env.energy, agents)
        else:
            safe_actions = nominal_actions

        # --- Step 3 (ADMM) + Step 1 (cert update): inside env.step ---
        next_obs, rewards, dones, _, _ = env.step(safe_actions)

        for a in agents:
            ep_r[a] += float(rewards[a])
        done = all(dones.values())

        # --- Step 5: logging ---
        if hasattr(env, "last_primal_residual"):
            ep_pr.append(float(env.last_primal_residual))
        if hasattr(env, "last_dual_residual"):
            ep_dr.append(float(env.last_dual_residual))
        if hasattr(env, "last_jain"):
            ep_jain.append(float(env.last_jain))
        if hasattr(env, "last_rates"):
            ep_sr.append(float(sum(env.last_rates[a]  for a in agents)))
        if hasattr(env, "last_sensing"):
            ep_sq.append(float(sum(env.last_sensing[a] for a in agents)))
        if hasattr(env, "last_powers") and hasattr(env, "last_rates"):
            tot_p = float(sum(env.last_powers[a] for a in agents))
            tot_r = float(sum(env.last_rates[a]  for a in agents))
            ep_ee.append(tot_r / (tot_p + 1e-12))
        if hasattr(env, "cert_comm_deficit"):
            ep_cert_c.append(float(np.mean(list(env.cert_comm_deficit.values()))))
        if hasattr(env, "cert_sense_deficit"):
            ep_cert_s.append(float(np.mean(list(env.cert_sense_deficit.values()))))
        # Review §5.9 fairness on both services; §6.3 min-service metrics
        if hasattr(env, "last_jain_S"):
            ep_jain_S.append(float(env.last_jain_S))
            ep_jain_JSC.append(float(env.last_jain_JSC))
        if hasattr(env, "last_min_rate"):
            ep_min_r.append(float(env.last_min_rate))
            ep_min_s.append(float(env.last_min_sensing))
        if hasattr(env, "last_spectral_eff"):
            ep_eta.append(float(sum(env.last_spectral_eff[a] for a in agents)))
        # Review §5.6/§6.3 certificate magnitude metrics (cert variants)
        if hasattr(env, "last_m_qos"):
            ep_mqos.append(float(env.last_m_qos))
            ep_viol.append(float(np.mean(
                [env.last_per_agent_violation[a] for a in agents])))
            ep_soft.append(float(np.mean(
                [env.last_cert_soft[a] for a in agents])))

        # Hard-cert per-step diagnostics (cert variants only).
        # All these attrs are populated by certenv._evaluate_certificates().
        if hasattr(env, "last_cert_passed"):
            ep_issued.append(float(env.last_cert_issued_count))
            comps = env.last_cert_components
            ep_comm_pass .append(float(np.mean([comps[a]["comm"]  for a in agents])))
            ep_sense_pass.append(float(np.mean([comps[a]["sense"] for a in agents])))
            ep_prox_pass .append(float(np.mean([comps[a]["prox"]  for a in agents])))
            ep_min_dist  .append(float(env.last_min_pair_dist))
            ep_max_dist  .append(float(env.last_max_pair_dist))
            ep_thr_c.append(float(env.last_cert_thresholds["comm"]))
            ep_thr_s.append(float(env.last_cert_thresholds["sense"]))
            ep_thr_p.append(float(env.last_cert_thresholds["prox"]))
            for a in agents:
                ep_pa_rate   [a].append(float(env.last_per_agent_rate[a]))
                ep_pa_sense  [a].append(float(env.last_per_agent_sense[a]))
                ep_pa_penalty[a].append(float(env.last_per_agent_penalty[a]))
                ep_pa_pass   [a].append(1.0 if env.last_cert_passed[a] else 0.0)

        # Store transition in replay buffer
        s       = obs_to_global(obs)
        s2      = obs_to_global(next_obs)
        a_joint = act_dict_to_joint(safe_actions)
        r_team  = float(np.clip(sum(rewards[a] for a in agents), -500.0, 500.0))

        # Fairness weight — DC/dccert: upweight transitions where ANY agent
        # in the swarm is underserved (max deficit across all agents).
        # Fixed-critic variants always use w=1.0 so their loss is unchanged.
        #
        # Why team-based instead of host-based:
        # ────────────────────────────────────────
        # The previous design used the active critic host's own deficit:
        #     deficit = max(0, z_R[host] - r_host)
        # This created a discontinuity at every rotation event: when the
        # host changed (e.g. uav_0 → uav_1), the fw scalar jumped because
        # a different agent's z_R/rate were now driving the weight. The
        # replay buffer ended up mixing transitions weighted by different
        # agents' fairness perspectives, the critic gradient direction
        # shifted, and the actor would sometimes collapse into a
        # one-agent-dominates degenerate policy (low Jain, high EE, low
        # sensing) for ~250 episodes before recovering.
        #
        # Team-based weighting (max deficit across the swarm) is invariant
        # to which agent is currently hosting, so rotation events no
        # longer perturb the loss landscape. The fw scalar is a property
        # of the swarm state, not the bookkeeping host.
        if cfg["critic_mode"] != "fixed" and hasattr(env, "active_critic_agent"):
            bw = getattr(env, "bandwidth", 1.0)
            # Compute deficit for every agent, take the worst.
            max_deficit = 0.0
            for a in agents:
                r_a = env.last_rates.get(a, 0.0) / (bw + 1e-12)
                zr_a = env.z_R.get(a, 0.0)
                d_a  = max(0.0, zr_a - r_a)
                if d_a > max_deficit:
                    max_deficit = d_a
            # Scale ceiling with N: at N=20 per-agent rates are ~0.065 bps/MHz
            # so a fixed ceiling of 0.30 gives fw≈1.1 (nearly uniform — no signal).
            # Scaling by N/5 restores fw to the full [1.0, 2.0] range at all swarm sizes.
            ceiling = 0.30 / max(1.0, NUM_UAVS / 5.0)  # 0.30@N=5, 0.075@N=20
            fw = 1.0 + min(max_deficit / ceiling, 1.0)  # w ∈ [1.0, 2.0]
        else:
            fw = 1.0

        buffer.add(s, a_joint, r_team, s2, done, w=fw)
        obs = next_obs

        # --- Step 5: MADDPG critic + actor update ---
        if buffer.size >= max(batch_size, start_steps):
            for _ in range(updates_per_step):
                S, A, R, S2, D, W = buffer.sample(batch_size)

                # Compute TD target using target networks
                chunks2 = split_global_obs(S2)
                with torch.no_grad():
                    A2    = torch.cat([actor_targets[a](ch)
                                       for a, ch in zip(agents, chunks2)], dim=-1)
                    q_nxt = critic_target(S2, A2)
                    y     = R + gamma * (1.0 - D) * q_nxt

                # Fairness-weighted critic loss.
                # W>1 for DC variants when host was underserved → stronger
                # gradient on fairness-violating states. W=1 for all
                # fixed-critic variants (original, cert) — identical to before.
                q_pred = critic(S, A)
                closs  = (W * (q_pred - y).pow(2)).mean()
                ep_closs.append(float(closs.item()))

                critic_opt.zero_grad()
                closs.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()

                # Review §4.5: E_comp = kappa_i C_i f_i^2 charged to the
                # ACTIVE critic host for this update. Under the fixed-critic
                # variants uav_0 pays every update — this is precisely the
                # energy-imbalance mechanism the moving critic removes.
                if hasattr(env, "charge_critic_compute"):
                    env.charge_critic_compute()

                # Actor update (gradient through shared critic)
                chunks = split_global_obs(S)
                A_pi   = torch.cat([actors[a](ch)
                                    for a, ch in zip(agents, chunks)], dim=-1)
                aloss  = -critic(S, A_pi).mean()

                for a in agents:
                    actor_opts[a].zero_grad()
                aloss.backward()
                for a in agents:
                    torch.nn.utils.clip_grad_norm_(actors[a].parameters(), 1.0)
                    actor_opts[a].step()

                # Polyak (soft) target updates
                for a in agents:
                    soft_update(actors[a], actor_targets[a])
                soft_update(critic, critic_target)

        if done:
            break

    # Episode-level aggregates
    all_primal_res .append(np.nanmean(ep_pr)     if ep_pr     else np.nan)
    all_dual_res   .append(np.nanmean(ep_dr)     if ep_dr     else np.nan)
    all_critic_loss.append(np.nanmean(ep_closs)  if ep_closs  else np.nan)
    all_sum_rate   .append(np.nanmean(ep_sr)     if ep_sr     else np.nan)
    all_sum_sensing.append(np.nanmean(ep_sq)     if ep_sq     else np.nan)
    all_energy_eff .append(np.nanmean(ep_ee)     if ep_ee     else np.nan)
    all_jain       .append(np.nanmean(ep_jain)   if ep_jain   else np.nan)
    all_cert_comm  .append(np.nanmean(ep_cert_c) if ep_cert_c else np.nan)
    all_cert_sense .append(np.nanmean(ep_cert_s) if ep_cert_s else np.nan)
    all_jain_S     .append(np.nanmean(ep_jain_S)   if ep_jain_S   else np.nan)
    all_jain_JSC   .append(np.nanmean(ep_jain_JSC) if ep_jain_JSC else np.nan)
    all_min_rate   .append(np.nanmean(ep_min_r)    if ep_min_r    else np.nan)
    all_min_sensing.append(np.nanmean(ep_min_s)    if ep_min_s    else np.nan)
    all_sum_eta    .append(np.nanmean(ep_eta)      if ep_eta      else np.nan)
    all_m_qos      .append(np.nanmean(ep_mqos)     if ep_mqos     else np.nan)
    all_cert_viol  .append(np.nanmean(ep_viol)     if ep_viol     else np.nan)
    all_cert_soft  .append(np.nanmean(ep_soft)     if ep_soft     else np.nan)
    # Review §4.5/§6.3: whole-run energy bookkeeping snapshots
    all_energy_imb .append(env.energy_imbalance()
                           if hasattr(env, "energy_imbalance") else np.nan)
    all_comp_energy.append(env.total_comp_energy()
                           if hasattr(env, "total_comp_energy") else np.nan)
    all_sync_energy.append(env.total_sync_energy()
                           if hasattr(env, "total_sync_energy") else np.nan)

    # Hard-cert per-episode aggregates (NaN when env has no cert hooks)
    if ep_issued:
        all_cert_issued    .append(float(np.mean(ep_issued)))
        all_cert_pass_rate .append(float(np.mean(ep_issued)) / max(NUM_UAVS, 1))
        all_cert_comm_pass .append(float(np.mean(ep_comm_pass)))
        all_cert_sense_pass.append(float(np.mean(ep_sense_pass)))
        all_cert_prox_pass .append(float(np.mean(ep_prox_pass)))
        all_cert_min_dist  .append(float(np.mean(ep_min_dist)))
        all_cert_max_dist  .append(float(np.mean(ep_max_dist)))
        all_cert_thr_comm  .append(float(np.mean(ep_thr_c)))
        all_cert_thr_sense .append(float(np.mean(ep_thr_s)))
        all_cert_thr_prox  .append(float(np.mean(ep_thr_p)))
        all_per_agent_rate   .append(np.array(
            [np.mean(ep_pa_rate[a])    for a in agents], dtype=np.float32))
        all_per_agent_sense  .append(np.array(
            [np.mean(ep_pa_sense[a])   for a in agents], dtype=np.float32))
        all_per_agent_penalty.append(np.array(
            [np.mean(ep_pa_penalty[a]) for a in agents], dtype=np.float32))
        all_per_agent_pass   .append(np.array(
            [np.mean(ep_pa_pass[a])    for a in agents], dtype=np.float32))
    else:
        # Non-cert variant — push NaN so array lengths match all_jain.
        all_cert_issued    .append(np.nan)
        all_cert_pass_rate .append(np.nan)
        all_cert_comm_pass .append(np.nan)
        all_cert_sense_pass.append(np.nan)
        all_cert_prox_pass .append(np.nan)
        all_cert_min_dist  .append(np.nan)
        all_cert_max_dist  .append(np.nan)
        all_cert_thr_comm  .append(np.nan)
        all_cert_thr_sense .append(np.nan)
        all_cert_thr_prox  .append(np.nan)
        nanvec = np.full(NUM_UAVS, np.nan, dtype=np.float32)
        all_per_agent_rate   .append(nanvec)
        all_per_agent_sense  .append(nanvec)
        all_per_agent_penalty.append(nanvec)
        all_per_agent_pass   .append(nanvec)

    if ep % 20 == 0:
        avg_j   = np.nanmean(ep_jain)  if ep_jain  else float("nan")
        avg_sr  = np.nanmean(ep_sr)    if ep_sr    else float("nan")
        avg_ee  = np.nanmean(ep_ee)    if ep_ee    else float("nan")
        avg_pr  = np.nanmean(ep_pr)    if ep_pr    else float("nan")
        avg_cc  = np.nanmean(ep_cert_c) if ep_cert_c else float("nan")
        rwd_str = " ".join([f"{a}={ep_r[a]:.2f}" for a in agents])
        avg_js  = np.nanmean(ep_jain_S) if ep_jain_S else float("nan")
        avg_mq  = np.nanmean(ep_mqos)   if ep_mqos   else float("nan")
        print(f"Ep {ep:4d}/{episodes} | {args.model_type} | N={NUM_UAVS} | "
              f"J_R={avg_j:.4f} | J_S={avg_js:.4f} | Rate={avg_sr:.3g}bps | "
              f"EE={avg_ee:.3g} | PrRes={avg_pr:.3f} | M_QoS={avg_mq:.3f} | "
              f"noise={noise_std:.3f} | Rwd:[{rwd_str}]")

# ---------------------------------------------------------------------------
# Save curves to results directory
# ---------------------------------------------------------------------------
prefix = os.path.join(args.results_dir, f"{args.model_type}_N{NUM_UAVS}_s{args.seed}")

curves = {
    "Primal_Residual":    all_primal_res,
    "Dual_Residual":      all_dual_res,
    "Critic_Loss":        all_critic_loss,
    "Sum_Rate_bps":       all_sum_rate,
    "Sum_Sensing":        all_sum_sensing,
    "Energy_Efficiency":  all_energy_eff,
    "Jain_Fairness":      all_jain,
    "Cert_Comm_Deficit":  all_cert_comm,
    "Cert_Sense_Deficit": all_cert_sense,
    # Review §5.9 / §6.3 — fairness on both services + minimum service
    "Jain_Sensing":       all_jain_S,
    "Jain_JSC":           all_jain_JSC,
    "Min_Rate_bps":       all_min_rate,
    "Min_Sensing":        all_min_sensing,
    "Sum_Spectral_Eff":   all_sum_eta,
    # Review §4.5 / §6.3 — energy accounting
    "Energy_Imbalance":   all_energy_imb,
    "Comp_Energy_Cum":    all_comp_energy,
    "Sync_Energy_Cum":    all_sync_energy,
    # Review §5.6 / §6.3 — certificate magnitude metrics
    "M_QoS":              all_m_qos,
    "Cert_Violation_Mag": all_cert_viol,
    "Cert_Soft_Value":    all_cert_soft,
    # Hard-cert diagnostics (NaN-filled for non-cert variants)
    "Cert_Issued_Count":  all_cert_issued,
    "Cert_Pass_Rate":     all_cert_pass_rate,
    "Cert_Comm_Pass":     all_cert_comm_pass,
    "Cert_Sense_Pass":    all_cert_sense_pass,
    "Cert_Prox_Pass":     all_cert_prox_pass,
    "Cert_Min_Dist":      all_cert_min_dist,
    "Cert_Max_Dist":      all_cert_max_dist,
    "Cert_Thr_Comm":      all_cert_thr_comm,
    "Cert_Thr_Sense":     all_cert_thr_sense,
    "Cert_Thr_Prox":      all_cert_thr_prox,
}

# Per-agent matrices saved separately because they are 2D (episodes, N)
per_agent_curves = {
    "PerAgent_Rate":    all_per_agent_rate,
    "PerAgent_Sense":   all_per_agent_sense,
    "PerAgent_Penalty": all_per_agent_penalty,
    "PerAgent_Pass":    all_per_agent_pass,
}
for key, vals in per_agent_curves.items():
    arr = np.stack(vals, axis=0).astype(np.float32) if vals else np.zeros((0, NUM_UAVS), dtype=np.float32)
    np.save(f"{prefix}_{key}.npy", arr)

for key, vals in curves.items():
    arr = np.array(vals, dtype=np.float32)

    # ── .npy  (needed by plot_comparisons.py) ──────────────────────────
    np.save(f"{prefix}_{key}.npy", arr)

    # ── .png  (immediate visual per metric) ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(arr))

    # Raw (translucent) + 20-episode running average (solid)
    ax.plot(x, arr, alpha=0.25, linewidth=0.8, color="#1565C0")
    w = min(20, max(1, len(arr) // 10))
    kernel = np.ones(w) / w
    smooth = np.convolve(
        np.pad(arr, (w // 2, w // 2), mode="edge"), kernel, mode="valid"
    )[: len(arr)]
    ax.plot(x, smooth, linewidth=2.0, color="#1565C0",
            label=f"smoothed (w={w})")

    metric_label = key.replace("_", " ")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(
        f"{metric_label}  —  {args.model_type} | N={NUM_UAVS} UAVs",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    png_path = f"{prefix}_{key}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

print(f"\n[train] Done. Curves saved to {prefix}_*.npy  and  {prefix}_*.png")
print(f"[train] Final Jain={np.nanmean(all_jain[-50:]):.4f} | "
      f"Rate={np.nanmean(all_sum_rate[-50:]):.4f} | "
      f"EE={np.nanmean(all_energy_eff[-50:]):.4f}")
