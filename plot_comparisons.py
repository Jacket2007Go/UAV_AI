"""
plot_comparisons.py  —  Generate all 6 comparison figures

Loads per-model .npy curve files from results/ and produces:

  Figure 1 : Original model — standalone performance (all 6 metrics)
  Figure 2 : Original  vs  Certificate model
  Figure 3 : Distributed Critic  vs  DC + Certificate
  Figure 4 : Original  vs  Distributed Critic
  Figure 5 : Certificate  vs  DC + Certificate
  Figure 6 : Original  vs  DC + Certificate

Usage:
  python plot_comparisons.py --num_uavs 3
  python plot_comparisons.py --num_uavs 3 --results_dir results --output_dir figures --smooth 20
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate 6 UAV JSC comparison figures")
parser.add_argument("--num_uavs",    type=int, default=3)
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("--output_dir",  type=str, default="figures")
parser.add_argument("--smooth",      type=int, default=30,
                    help="Running-average window size for curve smoothing")
parser.add_argument("--num_seeds",   type=int, default=3,
                    help="Number of seeds to average over (looks for _s0_, _s1_, ... files)")
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
N = args.num_uavs

# ---------------------------------------------------------------------------
# Model display metadata
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "original": "Original (Fixed Critic, No Cert)",
    "cert":     "Certificate Model (Fixed Critic)",
    "dc":       "Distributed Critic (No Cert)",
    "dccert":   "DC + Certificate",
}
MODEL_COLORS = {
    "original": "#1565C0",   # deep blue
    "cert":     "#2E7D32",   # deep green
    "dc":       "#E65100",   # deep orange
    "dccert":   "#AD1457",   # deep pink
}
MODEL_LS = {
    "original": "-",
    "cert":     "--",
    "dc":       "-.",
    "dccert":   ":",
}
MODEL_LW = {
    "original": 2.5,
    "cert":     2.5,
    "dc":       2.5,
    "dccert":   2.8,
}

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------
METRIC_LABEL = {
    "Jain_Fairness":     "Jain Fairness Index",
    "Sum_Rate_bps":      "Sum Rate (bps)",
    "Energy_Efficiency": "Energy Efficiency (bps/W)",
    "Critic_Loss":       "Critic Loss",
    "Primal_Residual":   "Primal Residual",
    "Dual_Residual":     "Dual Residual",
    "Sum_Sensing":       "Sum Sensing Utility",
    "Cert_Comm_Deficit": "Cert. Comm. Deficit (EMA)",
    "Cert_Sense_Deficit":"Cert. Sense Deficit (EMA)",
}

# Metrics to display in standalone (Fig 1) and comparison (Figs 2-6) layouts
FIG1_METRICS  = [
    "Jain_Fairness", "Sum_Rate_bps", "Energy_Efficiency",
    "Critic_Loss",   "Primal_Residual", "Dual_Residual",
]
COMP_METRICS  = [
    "Jain_Fairness", "Sum_Rate_bps", "Energy_Efficiency",
    "Critic_Loss",   "Primal_Residual", "Sum_Sensing",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_one(path: str) -> np.ndarray | None:
    """Load a single .npy file, interpolating interior NaNs. Returns None if missing."""
    if not os.path.exists(path):
        return None
    v = np.load(path).astype(np.float64)
    mask = np.isnan(v)
    if mask.any() and (~mask).any():
        idx = np.where(~mask)[0]
        v[mask] = np.interp(np.where(mask)[0], idx, v[idx])
    return v


def load(model_type: str, metric: str) -> np.ndarray | None:
    """
    Load and average across up to num_seeds seeded runs.
    Tries {model_type}_N{N}_s{seed}_{metric}.npy for each seed.
    Falls back to unseeded {model_type}_N{N}_{metric}.npy when no seeded
    files exist (backward compatibility with single-run results).
    Returns element-wise mean across available seeds, or None if nothing found.
    """
    seed_arrays = []
    for s in range(args.num_seeds):
        path = os.path.join(args.results_dir,
                            f"{model_type}_N{N}_s{s}_{metric}.npy")
        v = _load_one(path)
        if v is not None:
            seed_arrays.append(v)

    if seed_arrays:
        min_len = min(len(v) for v in seed_arrays)
        clipped = np.stack([v[:min_len] for v in seed_arrays], axis=0)
        return np.nanmean(clipped, axis=0)

    # Fallback: unseeded single-run file
    fallback = os.path.join(args.results_dir, f"{model_type}_N{N}_{metric}.npy")
    return _load_one(fallback)


def running_avg(v: np.ndarray, w: int) -> np.ndarray:
    """Causal running average — no future leakage."""
    if v is None or len(v) < 2:
        return v
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        lo = max(0, i - w + 1)
        out[i] = np.nanmean(v[lo: i + 1])
    return out


def rolling_std(v: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling standard deviation."""
    if v is None or len(v) < 2:
        return np.zeros_like(v)
    out = np.zeros_like(v)
    for i in range(len(v)):
        lo = max(0, i - w + 1)
        out[i] = np.nanstd(v[lo: i + 1])
    return out


def plot_metric_on_ax(
    ax,
    data_dict: dict,
    metric: str,
    show_legend: bool = True,
    smooth_w: int = 20,
):
    """
    Draw one metric panel comparing multiple models.
    data_dict: {model_type: ndarray | None}

    Plain-line plot style (no bands, no horizontal late-training lines):
      - Thin translucent raw trace shows the actual per-episode values
      - Smoothed running-average line on top (smooth_w-episode window)
    """
    any_data = False
    curves = {}

    for mtype, vals in data_dict.items():
        if vals is None:
            continue
        any_data = True
        x = np.arange(len(vals))
        s = running_avg(vals, smooth_w)
        curves[mtype] = (x, vals, s)

    # Determine y-axis range from smoothed curves only (not raw spikes)
    if any_data:
        all_s = np.concatenate([s for _, _, s in curves.values()
                                 if s is not None])
        valid = all_s[~np.isnan(all_s)]
        if len(valid):
            ylo = float(np.nanpercentile(valid, 2))
            yhi = float(np.nanpercentile(valid, 98))
            pad = max((yhi - ylo) * 0.15, 1e-6)
            ax.set_ylim(ylo - pad, yhi + pad)

    # Y-limits from smoothed data — clip raw values to same range so spikes
    # don't compress the interesting part of the plot.
    ylim = ax.get_ylim()

    for mtype, (x, vals, s) in curves.items():
        col = MODEL_COLORS[mtype]
        lw  = MODEL_LW[mtype]
        ls  = MODEL_LS[mtype]

        # Clip raw trace to smoothed y-range before plotting so outlier spikes
        # stay invisible rather than squishing the smoothed curves.
        clipped = np.clip(vals, ylim[0], ylim[1])
        ax.plot(x, clipped, color=col, alpha=0.10, linewidth=0.6)

        # Smoothed line
        ax.plot(x, s, color=col, linestyle=ls, linewidth=lw,
                label=MODEL_LABELS[mtype])

    if not any_data:
        ax.text(0.5, 0.5, "No data found\n(run training first)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#999999",
                bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#DDDDDD"))

    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=9)
    ax.set_title(METRIC_LABEL.get(metric, metric), fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.20, linestyle="--")
    ax.tick_params(labelsize=8)
    if show_legend and any_data:
        ax.legend(fontsize=8, loc="best", framealpha=0.88)


def shared_legend(fig, model_types: list, ncol: int = None):
    """Attach a shared legend at the top of a figure."""
    handles = [
        Line2D([0], [0],
               color=MODEL_COLORS[mt],
               linestyle=MODEL_LS[mt],
               linewidth=MODEL_LW[mt],
               label=MODEL_LABELS[mt])
        for mt in model_types
    ]
    nc = ncol or min(len(model_types), 4)
    fig.legend(handles=handles,
               loc="upper center", ncol=nc, fontsize=9,
               framealpha=0.9, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, 1.02))


def _style_figure(fig, title: str, fig_num: int):
    fig.suptitle(
        f"Figure {fig_num}  —  {title}  (N={N} UAVs)",
        fontsize=13, fontweight="bold", y=1.06,
    )

# ---------------------------------------------------------------------------
# Figure 1: Original model — standalone (all 6 metrics, 2×3 grid)
# ---------------------------------------------------------------------------
def figure1_original():
    fig = plt.figure(figsize=(15, 9))
    _style_figure(fig, "Original Model — Standalone Performance", 1)

    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    for ax, metric in zip(axes, FIG1_METRICS):
        data = load("original", metric)
        plot_metric_on_ax(ax, {"original": data}, metric,
                          show_legend=False, smooth_w=args.smooth)

    # Single-model legend entry
    shared_legend(fig, ["original"], ncol=1)
    fig.tight_layout()

    path = os.path.join(args.output_dir, f"fig1_original_N{N}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Figure 1 saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Generic comparison figure (Figs 2 – 6)
# ---------------------------------------------------------------------------
def comparison_figure(
    fig_num: int,
    title: str,
    model_types: list,
    metrics: list = None,
):
    if metrics is None:
        metrics = COMP_METRICS

    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(15, 5 * nrows),
                             squeeze=False)
    _style_figure(fig, title, fig_num)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        data_dict = {mt: load(mt, metric) for mt in model_types}
        # Legend only on first panel
        plot_metric_on_ax(ax, data_dict, metric,
                          show_legend=False, smooth_w=args.smooth)

    # Hide any unused subplots
    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    shared_legend(fig, model_types)

    fig.tight_layout()
    safe_title = (title.lower()
                  .replace(" ", "_")
                  .replace("+", "plus")
                  .replace("/", "_")
                  .replace("(", "").replace(")", ""))
    path = os.path.join(args.output_dir, f"fig{fig_num}_{safe_title}_N{N}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Figure {fig_num} saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Generate all 6 figures
# ---------------------------------------------------------------------------
print(f"\n[plot] Generating 6 comparison figures | N={N} | "
      f"results_dir={args.results_dir}")
print("-" * 60)

saved = []
saved.append(figure1_original())

saved.append(comparison_figure(
    2, "Original vs Certificate Model",
    ["original", "cert"],
))

saved.append(comparison_figure(
    3, "Distributed Critic vs DC + Certificate",
    ["dc", "dccert"],
))

saved.append(comparison_figure(
    4, "Original vs Distributed Critic",
    ["original", "dc"],
))

saved.append(comparison_figure(
    5, "Certificate Model vs DC + Certificate",
    ["cert", "dccert"],
))

saved.append(comparison_figure(
    6, "Original vs DC + Certificate",
    ["original", "dccert"],
))

print("-" * 60)
print(f"[plot] All 6 figures saved to: {os.path.abspath(args.output_dir)}/")

# ---------------------------------------------------------------------------
# Optional: summary table printed to stdout
# ---------------------------------------------------------------------------
print("\n[plot] Final 50-episode mean performance summary:")
header = f"{'Model':<28} {'Jain':>8} {'Rate(bps)':>12} {'EE(bps/W)':>12}"
print(header)
print("-" * len(header))
for mt in ["original", "cert", "dc", "dccert"]:
    jain = load(mt, "Jain_Fairness")
    rate = load(mt, "Sum_Rate_bps")
    ee   = load(mt, "Energy_Efficiency")
    j_m  = f"{np.nanmean(jain[-50:]):.4f}"  if jain is not None else "n/a"
    r_m  = f"{np.nanmean(rate[-50:]):.4f}"  if rate is not None else "n/a"
    e_m  = f"{np.nanmean(ee[-50:]):.4f}"    if ee   is not None else "n/a"
    print(f"  {MODEL_LABELS[mt]:<26} {j_m:>8} {r_m:>12} {e_m:>12}")
    