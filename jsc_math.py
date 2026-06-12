"""
jsc_math.py  —  Shared mathematical primitives for the revised formulation
===========================================================================

Implements the reviewer-required mathematics (review §4-§6) in one place so
orgenv.py, dcenv.py, certenv.py, and train_model.py stay consistent:

  §4.4 / §5.3  Fisher-information sensing model     → fim_sensing_quality()
  §5.6         Soft certificate sigmoid             → sigmoid()
  §5.9         Jain triplet J_R, J_S, J_JSC         → jain(), joint_jain()
  §6.3         Energy-imbalance metric I_E          → energy_imbalance()
  §6.3         Minimum-service metric M_QoS         → min_service_metric()
  §4.7         95% confidence interval               → ci95()

All functions are pure (no env state) and numpy-only.
"""

import numpy as np


# ============================================================================
# §4.4 / §5.3 — Fisher-information sensing model
# ============================================================================
def fim_sensing_quality(
    power: float,
    gain_sq: float,
    sigma_s2: float,
    beta_rms: float,
    T_rms: float,
    A_ang: float,
    J_ref: float = 1.0,
    eps: float = 1e-12,
):
    """
    FIM-based sensing quality (review §5.3), replacing the SNR proxy
    Q_i = P_i |g^s_i|² / σ²_s with an estimation-theoretic quantity.

    Model
    -----
    Sensing observation:  y_i(t) = μ_i(θ, t) + n_i(t),   θ = [τ, ν, φ]^T
    (delay, Doppler, angle).  Under the standard matched-filter /
    orthogonal-parameter approximation the FIM is diagonal:

        J_i(t) = (2 / σ_s²) Re{ (∂μ/∂θ)^H (∂μ/∂θ) }
               ≈ 2 · SNR_s,i · diag( 8π² β_rms²,  8π² T_rms²,  A_ang ),

    where
        SNR_s,i  = P_i |g^s_i|² / σ_s²       (sensing SNR — the old proxy),
        β_rms    = RMS waveform bandwidth (normalised units) — delay accuracy,
        T_rms    = RMS waveform duration (normalised units) — Doppler accuracy,
        A_ang    = effective array-aperture term — angle accuracy.

    Scalar quality (review §5.3, first option):

        S_i(t) = λ_min(J_i(t)) / J_ref            (dimensionless, normalised)

    and the CRB-side diagnostic (review §4.4):

        CRB_i(t) = tr(J_i(t)^{-1})    [so the cert condition tr(J^{-1}) ≤ ε_CRB
                                       is checkable from the returned value].

    Returns
    -------
    (S_i, crb_trace, snr_s) :
        S_i        λ_min(J)/J_ref  — use as the sensing-quality metric S_i(t)
        crb_trace  tr(J^{-1})      — CRB diagnostic (np.inf if J singular)
        snr_s      P|g|²/σ²        — legacy SNR proxy, kept for comparison
    """
    snr_s = float(power) * float(gain_sq) / (float(sigma_s2) + eps)

    # Diagonal FIM entries (orthogonal-parameter approximation)
    j_tau = 2.0 * snr_s * 8.0 * np.pi**2 * float(beta_rms) ** 2
    j_nu  = 2.0 * snr_s * 8.0 * np.pi**2 * float(T_rms) ** 2
    j_phi = 2.0 * snr_s * float(A_ang)
    diag  = np.array([j_tau, j_nu, j_phi], dtype=np.float64)

    lam_min = float(diag.min())
    S_i     = lam_min / (float(J_ref) + eps)

    if np.all(diag > eps):
        crb_trace = float(np.sum(1.0 / diag))
    else:
        crb_trace = float("inf")

    return S_i, crb_trace, snr_s


# ============================================================================
# §5.9 — Jain fairness triplet
# ============================================================================
def jain(values) -> float:
    """Jain index J = (Σx)² / (N Σx²) over a dict or array of non-neg values."""
    if isinstance(values, dict):
        vals = np.array(list(values.values()), dtype=np.float64)
    else:
        vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0 or float(np.sum(vals**2)) < 1e-15:
        return 1.0
    return float(np.sum(vals) ** 2 / (vals.size * np.sum(vals**2)))


def joint_jain(j_R: float, j_S: float,
               omega_R: float = 0.5, omega_S: float = 0.5) -> float:
    """J_JSC(t) = ω_R J_R(t) + ω_S J_S(t),  ω_R + ω_S = 1  (review §5.9)."""
    return float(omega_R * j_R + omega_S * j_S)


# ============================================================================
# §6.3 — Energy-imbalance metric
# ============================================================================
def energy_imbalance(energy_used, eps: float = 1e-12) -> float:
    """
    I_E = sqrt( (1/N) Σ_i (E_i^used − E_avg^used)² ) / (E_avg^used + ε)

    Coefficient-of-variation of per-UAV cumulative energy used (review §6.3).
    """
    if isinstance(energy_used, dict):
        e = np.array(list(energy_used.values()), dtype=np.float64)
    else:
        e = np.asarray(energy_used, dtype=np.float64)
    if e.size == 0:
        return 0.0
    avg = float(np.mean(e))
    return float(np.sqrt(np.mean((e - avg) ** 2)) / (avg + eps))


# ============================================================================
# §6.3 — Minimum-service metric
# ============================================================================
def min_service_metric(rates: dict, sensing: dict,
                       r_min: dict, s_min: dict,
                       eps: float = 1e-12) -> float:
    """
    M_QoS(t) = min_i min( R_i/R_i^min , S_i/S_i^min ).

    M_QoS ≥ 1 ⟺ every agent satisfies both normalised service floors.
    Floors of 0 (disabled absolute floor and zero median) contribute ratio
    = +inf for that component, i.e. that component never binds.
    """
    worst = float("inf")
    for a in rates:
        rm = float(r_min.get(a, 0.0))
        sm = float(s_min.get(a, 0.0))
        ratio_r = rates[a]   / rm if rm > eps else float("inf")
        ratio_s = sensing[a] / sm if sm > eps else float("inf")
        worst = min(worst, ratio_r, ratio_s)
    return worst if np.isfinite(worst) else 1.0


# ============================================================================
# §5.6 — Soft certificate
# ============================================================================
def sigmoid(x: float) -> float:
    """σ(x) = 1 / (1 + e^{-x}), numerically clipped."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


# ============================================================================
# §4.7 — Statistics
# ============================================================================
def ci95(samples) -> tuple:
    """
    (mean, half_width) with CI_95% = x̄ ± 1.96 s/√n  (review §4.7).
    s uses ddof=1 (sample standard deviation).
    """
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(x[0]), float("nan")
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(n))
