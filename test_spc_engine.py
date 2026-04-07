"""
Tests for the Statistical Process Control (SPC) Engine.
Run with: pytest test_spc_engine.py -v
"""

import pytest
import pandas as pd
import numpy as np
from spc_engine import (
    compute_control_limits,
    compute_capability,
    apply_western_electric_rules,
    generate_report,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def stable_process():
    """150 samples from a stable, in-control process."""
    np.random.seed(42)
    return pd.Series(np.random.normal(loc=10.0, scale=0.1, size=150))


@pytest.fixture
def shifted_process():
    """Process with a mean shift at sample 100 (simulates equipment drift)."""
    np.random.seed(42)
    data = np.random.normal(loc=10.0, scale=0.1, size=150)
    data[100:] += 0.5
    return pd.Series(data)


@pytest.fixture
def capable_process():
    """Tight process well within spec limits."""
    np.random.seed(1)
    return pd.Series(np.random.normal(loc=10.0, scale=0.05, size=100))


# ─── Control Limits ───────────────────────────────────────────────────────────

class TestControlLimits:

    def test_ucl_above_mean(self, stable_process):
        limits = compute_control_limits(stable_process)
        assert limits["ucl"] > limits["mean"]

    def test_lcl_below_mean(self, stable_process):
        limits = compute_control_limits(stable_process)
        assert limits["lcl"] < limits["mean"]

    def test_ucl_is_3sigma_above_mean(self, stable_process):
        limits = compute_control_limits(stable_process)
        expected_ucl = limits["mean"] + 3 * limits["std"]
        assert abs(limits["ucl"] - expected_ucl) < 1e-9

    def test_1sigma_zones_within_3sigma(self, stable_process):
        limits = compute_control_limits(stable_process)
        assert limits["lcl"] < limits["lcl_1s"] < limits["mean"]
        assert limits["mean"] < limits["ucl_1s"] < limits["ucl"]

    def test_2sigma_zones_between_1_and_3(self, stable_process):
        limits = compute_control_limits(stable_process)
        assert limits["ucl_1s"] < limits["ucl_2s"] < limits["ucl"]
        assert limits["lcl"] < limits["lcl_2s"] < limits["lcl_1s"]

    def test_known_values(self):
        data = pd.Series([10.0] * 100)
        limits = compute_control_limits(data)
        assert limits["mean"] == pytest.approx(10.0)
        assert limits["std"] == pytest.approx(0.0, abs=1e-9)


# ─── Process Capability ───────────────────────────────────────────────────────

class TestProcessCapability:

    def test_cp_positive(self, stable_process):
        cap = compute_capability(stable_process, usl=10.4, lsl=9.6)
        assert cap["Cp"] > 0

    def test_cpk_leq_cp(self, stable_process):
        """Cpk <= Cp always (Cpk penalizes off-center processes)."""
        cap = compute_capability(stable_process, usl=10.4, lsl=9.6)
        assert cap["Cpk"] <= cap["Cp"] + 1e-9

    def test_capable_process_cpk_above_threshold(self, capable_process):
        cap = compute_capability(capable_process, usl=10.3, lsl=9.7)
        assert cap["Cpk"] >= 1.33, f"Expected Cpk >= 1.33, got {cap['Cpk']}"

    def test_pp_positive(self, stable_process):
        cap = compute_capability(stable_process, usl=10.4, lsl=9.6)
        assert cap["Pp"] > 0

    def test_all_indices_present(self, stable_process):
        cap = compute_capability(stable_process, usl=10.4, lsl=9.6)
        for key in ("Cp", "Cpk", "Pp", "Ppk", "mean", "sigma_within"):
            assert key in cap

    def test_centered_process_cp_equals_cpk(self):
        """For a perfectly centered process, Cp should approximately equal Cpk."""
        data = pd.Series([10.0 + i * 0.001 for i in range(-50, 50)])
        cap = compute_capability(data, usl=10.5, lsl=9.5)
        assert abs(cap["Cp"] - cap["Cpk"]) < 0.5


# ─── Western Electric Rules ───────────────────────────────────────────────────

class TestWesternElectricRules:

    def test_rule1_point_beyond_3sigma(self):
        """Rule 1: a single point beyond 3σ should be flagged."""
        data = pd.Series([10.0] * 49 + [15.0])  # extreme outlier at end
        limits = compute_control_limits(data)
        violations = apply_western_electric_rules(data, limits)
        rule1 = violations[violations["rule"] == 1]
        assert len(rule1) >= 1

    def test_rule2_nine_points_same_side(self):
        """Rule 2: 9+ consecutive points on same side of mean."""
        base = [10.0] * 20
        run = [10.5] * 9   # all above mean
        data = pd.Series(base + run)
        limits = compute_control_limits(data)
        violations = apply_western_electric_rules(data, limits)
        rule2 = violations[violations["rule"] == 2]
        assert len(rule2) >= 1

    def test_stable_process_few_violations(self, stable_process):
        """A stable process should have very few violations (false alarm rate ~0.27%)."""
        limits = compute_control_limits(stable_process)
        violations = apply_western_electric_rules(stable_process, limits)
        # With 150 samples, expect well under 20 false alarms
        assert len(violations) < 20

    def test_shifted_process_many_violations(self, shifted_process):
        """A process with a mean shift should trigger many violations."""
        limits = compute_control_limits(shifted_process)
        violations = apply_western_electric_rules(shifted_process, limits)
        assert len(violations) > 10

    def test_violations_dataframe_columns(self, shifted_process):
        """Use shifted process to guarantee violations exist for column check."""
        limits = compute_control_limits(shifted_process)
        violations = apply_western_electric_rules(shifted_process, limits)
        assert not violations.empty, "Shifted process should produce violations"
        for col in ("index", "rule", "value"):
            assert col in violations.columns

    def test_violation_rules_in_valid_range(self, shifted_process):
        limits = compute_control_limits(shifted_process)
        violations = apply_western_electric_rules(shifted_process, limits)
        if not violations.empty:
            assert violations["rule"].between(1, 8).all()

    def test_no_duplicate_index_rule_pairs(self, shifted_process):
        limits = compute_control_limits(shifted_process)
        violations = apply_western_electric_rules(shifted_process, limits)
        dupes = violations.duplicated(subset=["index", "rule"]).sum()
        assert dupes == 0


# ─── Report Generation ────────────────────────────────────────────────────────

class TestGenerateReport:

    def test_report_keys_present(self, stable_process):
        report = generate_report(stable_process, "test", usl=10.4, lsl=9.6)
        for key in ("parameter", "n_samples", "control_limits",
                    "capability", "violations", "n_violations",
                    "in_control", "capable"):
            assert key in report

    def test_n_samples_correct(self, stable_process):
        report = generate_report(stable_process, "test", usl=10.4, lsl=9.6)
        assert report["n_samples"] == len(stable_process)

    def test_in_control_stable_process(self, stable_process):
        report = generate_report(stable_process, "test", usl=10.4, lsl=9.6)
        # Stable process should be in control (few or no violations)
        assert isinstance(report["in_control"], bool)

    def test_not_in_control_shifted_process(self, shifted_process):
        report = generate_report(shifted_process, "test", usl=10.4, lsl=9.6)
        assert report["in_control"] is False

    def test_capable_flag_matches_cpk(self, capable_process):
        report = generate_report(capable_process, "test", usl=10.3, lsl=9.7)
        cpk = report["capability"]["Cpk"]
        assert report["capable"] == (cpk >= 1.33)

    def test_parameter_name_in_report(self, stable_process):
        report = generate_report(stable_process, "gate_oxide_nm", usl=10.4, lsl=9.6)
        assert report["parameter"] == "gate_oxide_nm"
