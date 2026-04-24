"""
data_loader.py

Handles all data ingestion for the multi-asset portfolio analysis.
Loads monthly manager returns, benchmark returns, the risk-free rate,
and Strategic Asset Allocation (SAA) weights from the /data folder.

Also defines the Tactical Asset Allocation (TAA) weights used to
compute the total fund return.

Main entry point: load_all()
"""

import pandas as pd
from pathlib import Path

# ── TAA Weights (hardcoded from assignment brief) ─────────────────────────────
# TAA is not supplied as a CSV — only the brief's table — so it lives in code.
TAA_WEIGHTS = {
    "AUS_EQ":      0.35,
    "INTL_EQ":     0.35,
    "Bonds":       0.15,
    "Real_Estate": 0.05,
    "PE_VC":       0.10,
}

# ── Name mappings ─────────────────────────────────────────────────────────────
# The CSVs on disk use short lowercase stems (aus_eq, intl_eq, bonds, re, pevc).
# We use cleaner display names everywhere else in the codebase.
# These two dicts translate between the two worlds.

FILE_STEMS = {
    "AUS_EQ":      "aus_eq",
    "INTL_EQ":     "intl_eq",
    "Bonds":       "bonds",
    "Real_Estate": "re",
    "PE_VC":       "pevc",
}

# saa_weight.csv uses different sleeve codes than our display names.
# This dict converts the CSV's codes into our standard names.
SAA_CSV_RENAME = {
    "AUS_EQ":  "AUS_EQ",
    "INTL_EQ": "INTL_EQ",
    "BONDS":   "Bonds",
    "RE":      "Real_Estate",
    "PEVC":    "PE_VC",
}