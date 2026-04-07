# Statistical Process Control (SPC) Engine

[![Tests](https://github.com/Hassankusow/spc-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Hassankusow/spc-engine/actions/workflows/tests.yml)

A Python-based SPC monitoring system for semiconductor manufacturing process data. Implements industry-standard control charting, Western Electric rule detection, and process capability analysis used in fab environments.

---

## Features

- **X-bar Control Charts** — dynamic 3-sigma UCL/LCL with 1σ and 2σ zone lines
- **All 8 Western Electric Rules** — full rule set for out-of-control signal detection
- **Process Capability Indices** — Cp, Cpk (short-term) and Pp, Ppk (long-term)
- **Short-term sigma** estimated via average moving range (AIAG SPC reference method)
- **Lot-level ANOVA** — isolates between-lot vs. within-lot variation sources
- **Automated reports** — JSON output flagging violations by rule, index, and value
- **Process shift simulation** — built-in demo injects equipment drift at sample 100

---

## Tech Stack

Python, Pandas, NumPy, SciPy, Matplotlib

---

## Usage

```bash
pip install -r requirements.txt
python spc_engine.py
```

---

## Example Output

```json
{
  "parameter": "gate_oxide_thickness_nm",
  "n_samples": 150,
  "capability": {
    "Cp": 1.3322,
    "Cpk": 1.0636,
    "Pp": 0.834,
    "Ppk": 0.6658,
    "mean": 10.121,
    "sigma_within": 0.1501
  },
  "n_violations": 67,
  "in_control": false,
  "capable": false
}
```

The demo simulates 150 samples of gate oxide thickness (nm) with a process shift injected at sample 100 — mimicking real equipment drift. The engine correctly flags the shift via Western Electric rules and shows Cpk dropping below 1.33.

---

## API Reference

| Function | Description |
|----------|-------------|
| `compute_control_limits(data)` | Returns mean, UCL, LCL, and 1σ/2σ zone limits |
| `compute_capability(data, usl, lsl)` | Returns Cp, Cpk, Pp, Ppk, mean, sigma_within |
| `apply_western_electric_rules(data, limits)` | Returns DataFrame of all violations with rule number and index |
| `plot_control_chart(data, parameter, usl, lsl)` | Saves X-bar chart PNG with violations highlighted |
| `generate_report(data, parameter, usl, lsl)` | Full JSON report combining all analysis |

---

## Western Electric Rules Implemented

| Rule | Description |
|------|-------------|
| 1 | 1 point beyond 3σ |
| 2 | 9 consecutive points same side of mean |
| 3 | 6 consecutive points trending up or down |
| 4 | 14 alternating up/down points |
| 5 | 3 of 3 consecutive points beyond 2σ same side |
| 6 | 4 of 5 consecutive points beyond 1σ same side |
| 7 | 15 consecutive points within 1σ (stratification) |
| 8 | 8 consecutive points beyond 1σ on either side |

---

## Author

**Hassan Abdi**
[GitHub](https://github.com/Hassankusow) | [LinkedIn](https://linkedin.com/in/hassan-abdi-119357267)
