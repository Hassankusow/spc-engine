"""
Statistical Process Control (SPC) Engine
Implements X-bar/R control charts, Western Electric rules,
and process capability indices (Cp, Cpk, Pp, Ppk).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from datetime import datetime


# ─── Control Limits ──────────────────────────────────────────────────────────

def compute_control_limits(data: pd.Series) -> dict:
    """Compute mean and 3-sigma UCL/LCL for an X-bar control chart."""
    mean = data.mean()
    std = data.std(ddof=1)
    return {
        "mean": mean,
        "ucl": mean + 3 * std,
        "lcl": mean - 3 * std,
        "ucl_1s": mean + std,
        "lcl_1s": mean - std,
        "ucl_2s": mean + 2 * std,
        "lcl_2s": mean - 2 * std,
        "std": std,
    }


# ─── Process Capability ───────────────────────────────────────────────────────

def compute_capability(data: pd.Series, usl: float, lsl: float) -> dict:
    """
    Compute Cp, Cpk (short-term) and Pp, Ppk (long-term) capability indices.
    Cp/Cpk use within-subgroup sigma; Pp/Ppk use overall sigma.
    """
    mean = data.mean()
    sigma_overall = data.std(ddof=1)

    # Short-term sigma estimated via average moving range
    moving_range = data.diff().abs().dropna()
    d2 = 1.128  # control chart constant for n=2
    sigma_within = moving_range.mean() / d2

    cp  = (usl - lsl) / (6 * sigma_within)
    cpk = min((usl - mean) / (3 * sigma_within), (mean - lsl) / (3 * sigma_within))
    pp  = (usl - lsl) / (6 * sigma_overall)
    ppk = min((usl - mean) / (3 * sigma_overall), (mean - lsl) / (3 * sigma_overall))

    return {"Cp": round(cp, 4), "Cpk": round(cpk, 4),
            "Pp": round(pp, 4), "Ppk": round(ppk, 4),
            "mean": round(mean, 4), "sigma_within": round(sigma_within, 4)}


# ─── Western Electric Rules ───────────────────────────────────────────────────

def apply_western_electric_rules(data: pd.Series, limits: dict) -> pd.DataFrame:
    """
    Detect all 8 Western Electric SPC rule violations.
    Returns a DataFrame of violations with index, rule number, and value.
    """
    v = data.values
    mean  = limits["mean"]
    ucl   = limits["ucl"]
    lcl   = limits["lcl"]
    u1s   = limits["ucl_1s"]
    l1s   = limits["lcl_1s"]
    u2s   = limits["ucl_2s"]
    l2s   = limits["lcl_2s"]

    violations = []

    def flag(i, rule):
        violations.append({"index": i, "rule": rule, "value": v[i]})

    for i in range(len(v)):
        # Rule 1: 1 point beyond 3σ
        if v[i] > ucl or v[i] < lcl:
            flag(i, 1)

        if i >= 1:
            # Rule 6: 4 of 5 consecutive points beyond 1σ on same side
            if i >= 4:
                seg = v[i-4:i+1]
                if sum(x > u1s for x in seg) >= 4 or sum(x < l1s for x in seg) >= 4:
                    flag(i, 6)

        if i >= 7:
            seg = v[i-7:i+1]
            # Rule 2: 9 consecutive points same side of mean
            if all(x > mean for x in seg) or all(x < mean for x in seg):
                flag(i, 2)
            # Rule 3: 6 consecutive points steadily increasing or decreasing
            if all(seg[j] < seg[j+1] for j in range(7)) or all(seg[j] > seg[j+1] for j in range(7)):
                flag(i, 3)
            # Rule 4: 14 alternating up/down (check on i>=13)
        if i >= 13:
            seg = v[i-13:i+1]
            alternating = all(
                (seg[j] < seg[j+1]) != (seg[j+1] < seg[j+2])
                for j in range(12)
            )
            if alternating:
                flag(i, 4)

        if i >= 2:
            seg = v[i-2:i+1]
            # Rule 5: 3 of 3 consecutive points beyond 2σ on same side
            if sum(x > u2s for x in seg) == 3 or sum(x < l2s for x in seg) == 3:
                flag(i, 5)

        if i >= 7:
            seg = v[i-7:i+1]
            # Rule 7: 15 consecutive points within 1σ of mean (stratification)
        if i >= 14:
            seg = v[i-14:i+1]
            if all(l1s < x < u1s for x in seg):
                flag(i, 7)

        # Rule 8: 8 consecutive points beyond 1σ on either side (no points near mean)
        if i >= 7:
            seg = v[i-7:i+1]
            if all(x > u1s or x < l1s for x in seg):
                flag(i, 8)

    return pd.DataFrame(violations).drop_duplicates(subset=["index", "rule"])


# ─── Charting ────────────────────────────────────────────────────────────────

def plot_control_chart(data: pd.Series, parameter: str,
                       usl: float = None, lsl: float = None,
                       save_path: str = None):
    """Plot X-bar control chart with violations highlighted."""
    limits = compute_control_limits(data)
    violations = apply_western_electric_rules(data, limits)
    violated_idx = set(violations["index"].tolist())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data.values, color="steelblue", linewidth=1, marker="o",
            markersize=3, label="Measurements")

    # Highlight violations
    if not violations.empty:
        ax.scatter(violations["index"], violations["value"],
                   color="red", zorder=5, s=60, label="WE Violation")

    ax.axhline(limits["mean"], color="green",  linestyle="--", linewidth=1.2, label=f'Mean={limits["mean"]:.3f}')
    ax.axhline(limits["ucl"],  color="red",    linestyle="--", linewidth=1,   label=f'UCL={limits["ucl"]:.3f}')
    ax.axhline(limits["lcl"],  color="red",    linestyle="--", linewidth=1,   label=f'LCL={limits["lcl"]:.3f}')
    ax.axhline(limits["ucl_2s"], color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(limits["lcl_2s"], color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(limits["ucl_1s"], color="gold",   linestyle=":", linewidth=0.8)
    ax.axhline(limits["lcl_1s"], color="gold",   linestyle=":", linewidth=0.8)

    if usl: ax.axhline(usl, color="purple", linestyle="-.", linewidth=1, label=f"USL={usl}")
    if lsl: ax.axhline(lsl, color="purple", linestyle="-.", linewidth=1, label=f"LSL={lsl}")

    ax.set_title(f"X-bar Control Chart — {parameter}  |  Violations: {len(violations)}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─── Reporting ───────────────────────────────────────────────────────────────

def generate_report(data: pd.Series, parameter: str,
                    usl: float, lsl: float) -> dict:
    """Generate a full SPC report for a parameter."""
    limits = compute_control_limits(data)
    capability = compute_capability(data, usl, lsl)
    violations = apply_western_electric_rules(data, limits)

    report = {
        "parameter": parameter,
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(data),
        "control_limits": {k: round(v, 4) for k, v in limits.items()},
        "capability": capability,
        "violations": violations.to_dict(orient="records"),
        "n_violations": len(violations),
        "in_control": len(violations) == 0,
        "capable": capability["Cpk"] >= 1.33,
    }
    return report


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # Simulate 150 samples of a process parameter (e.g., gate oxide thickness in nm)
    n = 150
    base = np.random.normal(loc=10.0, scale=0.15, size=n)

    # Inject a process shift at sample 100 (simulate equipment drift)
    base[100:] += 0.4

    data = pd.Series(base, name="gate_oxide_thickness_nm")

    usl, lsl = 10.6, 9.4

    print("=== SPC Report ===")
    report = generate_report(data, "gate_oxide_thickness_nm", usl, lsl)
    print(json.dumps({k: v for k, v in report.items() if k != "violations"}, indent=2))
    print(f"\nViolations detected: {report['n_violations']}")
    print(f"Process capable (Cpk >= 1.33): {report['capable']}")

    plot_control_chart(data, "gate_oxide_thickness_nm", usl=usl, lsl=lsl,
                       save_path="spc_gate_oxide.png")
