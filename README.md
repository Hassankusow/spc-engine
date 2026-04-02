# Statistical Process Control (SPC) Engine

A Python-based SPC monitoring system for semiconductor manufacturing process data.

## Features
- X-bar control charts with dynamic 3-sigma UCL/LCL recalculation
- All 8 Western Electric rule detections for out-of-control signal identification
- Process capability indices: Cp, Cpk (short-term) and Pp, Ppk (long-term)
- Automated violation reports with rule type, sample index, and value
- Short-term sigma estimation via average moving range (AIAG method)

## Usage
```bash
pip install -r requirements.txt
python spc_engine.py
```

## Example Output
Generates an X-bar control chart PNG and a JSON report flagging SPC violations and capability status.
