"""
run_all.py  —  Master runner: train all 4 model variants, then plot

Trains all 4 variants sequentially (or a subset via --models), saves
.npy curves to results/, then calls plot_comparisons.py to generate
all 6 figures in figures/.

Usage:
  # Full paper run (500/600/1000 episodes depending on N)
  python run_all.py --num_uavs 3

  # Quick smoke-test (200 episodes each)
  python run_all.py --num_uavs 3 --episodes 200

  # Only specific models
  python run_all.py --num_uavs 3 --models original cert

  # Skip training, re-generate plots from existing results
  python run_all.py --num_uavs 3 --skip_training

  # Full paper schedule for all swarm sizes
  python run_all.py --num_uavs 2  &&
  python run_all.py --num_uavs 3  &&
  python run_all.py --num_uavs 20

Output tree:
  results/
    original_N3_Jain_Fairness.npy
    original_N3_Sum_Rate_bps.npy
    ...  (9 metrics × 4 models = 36 files per N)
  figures/
    fig1_original_N3.png
    fig2_original_vs_certificate_model_N3.png
    fig3_distributed_critic_vs_dc_plus_certificate_N3.png
    fig4_original_vs_distributed_critic_N3.png
    fig5_certificate_model_vs_dc_plus_certificate_N3.png
    fig6_original_vs_dc_plus_certificate_N3.png
"""

import argparse
import subprocess
import sys
import os
import time

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train all 4 UAV JSC variants + plot")
parser.add_argument("--num_uavs", type=int, default=3,
                    help="Swarm size to train and evaluate")
parser.add_argument("--episodes", type=int, default=None,
                    help="Override episode budget for all models "
                         "(default: paper schedule 500/600/1000)")
parser.add_argument("--models", nargs="+",
                    choices=["original", "cert", "dc", "dccert"],
                    default=["original", "cert", "dc", "dccert"],
                    help="Which model variants to train (default: all 4)")
parser.add_argument("--skip_training", action="store_true",
                    help="Skip all training, only regenerate plots")
parser.add_argument("--results_dir", type=str, default="results",
                    help="Directory where .npy curves are saved/read")
parser.add_argument("--figures_dir", type=str, default="figures",
                    help="Directory where figures are saved")
parser.add_argument("--smooth", type=int, default=30,
                    help="Plot curve smoothing window (episodes)")
parser.add_argument("--num_seeds", type=int, default=3,
                    help="Number of random seeds to train per model variant")
# Cert hyperparameters (forwarded to train_model.py)
parser.add_argument("--cert_R_weight", type=float, default=2.0)
parser.add_argument("--cert_Q_weight", type=float, default=1.0)
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)
os.makedirs(args.figures_dir, exist_ok=True)

PAPER_BUDGET = {2: 500, 3: 600, 5: 1000, 20: 1000}

# ---------------------------------------------------------------------------
# Model order: original first (clean baseline), then cert, dc, dccert
# ---------------------------------------------------------------------------
ORDERED = ["original", "cert", "dc", "dccert"]
models_to_run = [m for m in ORDERED if m in args.models]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model_type: str, seed: int):
    cmd = [
        sys.executable, "train_model.py",
        "--model_type",  model_type,
        "--num_uavs",    str(args.num_uavs),
        "--results_dir", args.results_dir,
        "--seed",        str(seed),
        "--cert_R_weight", str(args.cert_R_weight),
        "--cert_Q_weight", str(args.cert_Q_weight),
    ]
    if args.episodes is not None:
        cmd += ["--episodes", str(args.episodes)]

    ep_budget = args.episodes or PAPER_BUDGET.get(args.num_uavs, 800)
    print(f"\n{'='*65}")
    print(f"  Training: {model_type:<12} | N={args.num_uavs} | seed={seed} | "
          f"{ep_budget} episodes")
    print(f"{'='*65}")

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [WARNING] {model_type} s{seed} training exited with code "
              f"{result.returncode} — continuing.")
    else:
        print(f"  [done] {model_type} s{seed} | elapsed={elapsed/60:.1f} min")
    return result.returncode == 0


if not args.skip_training:
    successes = []
    for seed in range(args.num_seeds):
        for mt in models_to_run:
            ok = train(mt, seed)
            successes.append((mt, seed, ok))

    print(f"\n{'='*65}")
    print("  Training summary:")
    for mt, seed, ok in successes:
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"    {mt:<12}  s{seed}  {status}")
    print(f"{'='*65}")
else:
    print("[run_all] --skip_training set — loading existing results.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"  Generating 6 comparison figures  (N={args.num_uavs})")
print(f"{'='*65}")

plot_cmd = [
    sys.executable, "plot_comparisons.py",
    "--num_uavs",    str(args.num_uavs),
    "--results_dir", args.results_dir,
    "--output_dir",  args.figures_dir,
    "--smooth",      str(args.smooth),
    "--num_seeds",   str(args.num_seeds),
]
plot_result = subprocess.run(plot_cmd, check=False)

if plot_result.returncode == 0:
    print(f"\n[run_all] ✓ Complete.")
    print(f"  Results : {os.path.abspath(args.results_dir)}/")
    print(f"  Figures : {os.path.abspath(args.figures_dir)}/")
else:
    print(f"\n[run_all] plot_comparisons.py exited with error "
          f"(code {plot_result.returncode}).")
    print("  Check that results/*.npy files exist and re-run with "
          "--skip_training to regenerate plots only.")
    