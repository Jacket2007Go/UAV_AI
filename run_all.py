"""
run_all.py  —  Master runner: train all 4 model variants, plot, and report stats

Trains all 4 variants sequentially (or a subset via --models) over
--num_seeds independent seeds (default 10, review §4.7), saves .npy
curves to results/, calls plot_comparisons.py for the 6 figures, then
writes a statistical summary (mean ± CI95 across seeds + paired
deltas ΔJ_m = V4,m − V1,m) to results/stats_summary_N{N}.{csv,txt}.

Usage:
  # Full paper run (500/600/1000 episodes depending on N, 10 seeds)
  python run_all.py --num_uavs 3

  # Quick smoke-test (200 episodes, 2 seeds)
  python run_all.py --num_uavs 3 --episodes 200 --num_seeds 2

  # Only specific models
  python run_all.py --num_uavs 3 --models original cert

  # Skip training, re-generate plots + stats from existing results
  python run_all.py --num_uavs 3 --skip_training

  # Review §6.1 ablations (examples):
  python run_all.py --num_uavs 5 --cert_mode hard                 # hard vs soft
  python run_all.py --num_uavs 5 --cert_R_abs 0 --cert_S_abs 0    # relative-only floors
  python run_all.py --num_uavs 5 --rotation_interval 50           # K sweep

Output tree:
  results/
    {model}_N{N}_s{seed}_{Metric}.npy     (34 metrics × 4 models × seeds)
    stats_summary_N{N}.csv                (machine-readable)
    stats_summary_N{N}.txt                (paper-ready table)
  figures/
    fig1..fig6 comparison plots
"""

import argparse
import subprocess
import sys
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train all 4 UAV JSC variants + plot + stats")
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
                    help="Skip all training, only regenerate plots + stats")
parser.add_argument("--results_dir", type=str, default="results",
                    help="Directory where .npy curves are saved/read")
parser.add_argument("--figures_dir", type=str, default="figures",
                    help="Directory where figures are saved")
parser.add_argument("--smooth", type=int, default=30,
                    help="Plot curve smoothing window (episodes)")
parser.add_argument("--num_seeds", type=int, default=10,
                    help="Independent seeds per variant (review §4.7: ≥10)")
parser.add_argument("--stats_window", type=int, default=100,
                    help="Final-episode window for converged-performance stats")

# ---- Hyperparameters forwarded verbatim to train_model.py ------------------
# default=None ⇒ not passed ⇒ train_model.py's own default applies.
FORWARDED = [
    # legacy cert weights (dead in env, kept for CLI compat)
    ("--cert_R_weight", float), ("--cert_Q_weight", float),
    # certificate floors / gates (review §4.1–§4.2, §5.4)
    ("--cert_tau", float), ("--cert_tau_R", float), ("--cert_tau_S", float),
    ("--cert_R_abs", float), ("--cert_S_abs", float),
    ("--cert_E_min", float), ("--cert_d_min", float),
    # soft/hard certificate + penalty weights (review §5.6)
    ("--cert_mode", str),
    ("--cert_lambda_R", float), ("--cert_lambda_S", float),
    ("--cert_lambda_E", float), ("--cert_lambda_D", float),
    ("--cert_alpha_C", float),
    ("--cert_rho_R", float), ("--cert_rho_S", float),
    ("--cert_alpha", float), ("--cert_lambda", float),
    # sensing model + joint fairness (review §4.4, §5.9)
    ("--sensing_metric", str), ("--omega_R", float), ("--omega_S", float),
    # critic-host rotation / eligibility (review §5.8)
    ("--rotation_interval", int), ("--load_max", float), ("--sync_rate_min", float),
]
for flag, typ in FORWARDED:
    parser.add_argument(flag, type=typ, default=None)

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
    ]
    if args.episodes is not None:
        cmd += ["--episodes", str(args.episodes)]
    for flag, _typ in FORWARDED:
        val = getattr(args, flag.lstrip("-").replace("-", "_"))
        if val is not None:
            cmd += [flag, str(val)]

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
        status = "OK" if ok else "FAILED"
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
if plot_result.returncode != 0:
    print(f"\n[run_all] plot_comparisons.py exited with error "
          f"(code {plot_result.returncode}) — continuing to stats.")

# ===========================================================================
# Statistical summary  (review §4.7)
# ===========================================================================
# For each (variant, metric): per-seed converged value = nan-mean over the
# final --stats_window episodes. Across seeds we report
#       mean ± CI95,   CI95 = 1.96 · s / sqrt(n)        (n = #valid seeds)
# Paired effect of the full architecture:
#       ΔJ_m = J_{V4(dccert),m} − J_{V1(original),m}    (same seed m)
# reported as mean Δ ± CI95 with a paired t statistic
#       t = Δ̄ / (s_Δ / sqrt(n)),                        df = n − 1
# (p-value via scipy if installed, else omitted).
# ===========================================================================

STAT_METRICS = [
    # (curve key, pretty name, higher_is_better)
    ("Jain_Fairness",      "Jain J_R (rate)",          True),
    ("Jain_Sensing",       "Jain J_S (sensing)",       True),
    ("Jain_JSC",           "Jain J_JSC (joint)",       True),
    ("Sum_Rate_bps",       "Sum rate [bps]",           True),
    ("Sum_Spectral_Eff",   "Sum spectral eff [b/s/Hz]",True),
    ("Min_Rate_bps",       "min_i R_i [bps]",          True),
    ("Min_Sensing",        "min_i S_i",                True),
    ("Energy_Efficiency",  "Energy efficiency",        True),
    ("Energy_Imbalance",   "Energy imbalance I_E",     False),
    ("M_QoS",              "QoS margin M_QoS",         True),
    ("Cert_Pass_Rate",     "Cert pass rate",           True),
    ("Cert_Violation_Mag", "Cert violation magnitude", False),
    ("Primal_Residual",    "ADMM primal resid r^k",    False),
    ("Dual_Residual",      "ADMM dual resid s^k",      False),
]

def _final_window_mean(path: str, window: int):
    """nan-mean of the last `window` episodes; None if file missing/empty."""
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.size == 0:
        return None
    tail = arr[-min(window, arr.size):].astype(np.float64)
    with np.errstate(all="ignore"):
        v = np.nanmean(tail)
    return None if np.isnan(v) else float(v)

def _mean_ci(vals):
    """(mean, ci95, n) over a list of floats."""
    v = np.asarray(vals, dtype=np.float64)
    n = v.size
    if n == 0:
        return (np.nan, np.nan, 0)
    m = float(np.mean(v))
    if n == 1:
        return (m, np.nan, 1)
    s = float(np.std(v, ddof=1))
    return (m, 1.96 * s / np.sqrt(n), n)

print(f"\n{'='*65}")
print(f"  Statistical summary  (N={args.num_uavs}, window={args.stats_window} eps, "
      f"seeds 0..{args.num_seeds-1})")
print(f"{'='*65}")

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None

per_seed = {}   # per_seed[model][metric] = {seed: value}
for mt in ORDERED:
    per_seed[mt] = {}
    for key, _pn, _hib in STAT_METRICS:
        d = {}
        for seed in range(args.num_seeds):
            p = os.path.join(args.results_dir,
                             f"{mt}_N{args.num_uavs}_s{seed}_{key}.npy")
            v = _final_window_mean(p, args.stats_window)
            if v is not None:
                d[seed] = v
        per_seed[mt][key] = d

csv_lines = ["section,metric,model,mean,ci95,n_seeds,delta_mean,delta_ci95,t_stat,p_value"]
txt_lines = [f"Statistical summary — N={args.num_uavs}, "
             f"final-{args.stats_window}-episode means, {args.num_seeds} seeds",
             f"CI95 = 1.96*s/sqrt(n);  Δ = dccert − original (paired by seed)",
             "=" * 100]

for key, pretty, hib in STAT_METRICS:
    rows = []
    for mt in ORDERED:
        vals = list(per_seed[mt][key].values())
        m, ci, n = _mean_ci(vals)
        rows.append((mt, m, ci, n))

    # Paired Δ = dccert − original on common seeds
    d_o, d_4 = per_seed["original"][key], per_seed["dccert"][key]
    common = sorted(set(d_o) & set(d_4))
    deltas = [d_4[s] - d_o[s] for s in common]
    dm, dci, dn = _mean_ci(deltas)
    if dn >= 2 and np.std(deltas, ddof=1) > 0:
        t_stat = dm / (np.std(deltas, ddof=1) / np.sqrt(dn))
        p_val  = (float(2 * _scipy_stats.t.sf(abs(t_stat), df=dn - 1))
                  if _scipy_stats is not None else np.nan)
    else:
        t_stat, p_val = np.nan, np.nan

    arrow = "↑ better" if hib else "↓ better"
    txt_lines.append(f"\n{pretty}  ({key}, {arrow})")
    for mt, m, ci, n in rows:
        if n == 0:
            txt_lines.append(f"  {mt:<10} —  (no data)")
        else:
            ci_s = f"± {ci:.4g}" if np.isfinite(ci) else "(single seed)"
            txt_lines.append(f"  {mt:<10} {m:.6g} {ci_s}   [n={n}]")
        csv_lines.append(f"per_model,{key},{mt},"
                         f"{m if n else ''},{ci if n else ''},{n},,,,")
    if dn:
        t_s = f"{t_stat:.3f}" if np.isfinite(t_stat) else ""
        p_s = f"{p_val:.4g}" if np.isfinite(p_val) else ""
        ci_s = f"± {dci:.4g}" if np.isfinite(dci) else ""
        txt_lines.append(f"  Δ(dccert−orig) {dm:+.6g} {ci_s}   "
                         f"[n={dn}, t={t_s}" + (f", p={p_s}" if p_s else "") + "]")
        csv_lines.append(f"paired_delta,{key},dccert-original,,,,"
                         f"{dm},{dci if np.isfinite(dci) else ''},{t_s},{p_s}")

stats_csv = os.path.join(args.results_dir, f"stats_summary_N{args.num_uavs}.csv")
stats_txt = os.path.join(args.results_dir, f"stats_summary_N{args.num_uavs}.txt")
with open(stats_csv, "w") as f:
    f.write("\n".join(csv_lines) + "\n")
with open(stats_txt, "w") as f:
    f.write("\n".join(txt_lines) + "\n")

print("\n".join(txt_lines))
print(f"\n[run_all] Stats written to:\n  {stats_csv}\n  {stats_txt}")
print(f"\n[run_all] Complete.")
print(f"  Results : {os.path.abspath(args.results_dir)}/")
print(f"  Figures : {os.path.abspath(args.figures_dir)}/")
