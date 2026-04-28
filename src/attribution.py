"""
attribution.py

Brinson performance attribution for the multi-asset portfolio.

The module decomposes each sleeve's active return into 2 components:    
    - Allocation effect = the impact of over/underweighing the asset classes relative to the benchmark 
    - Selection effect = the contribution from manager performance versus benchmarks

Effects are calculated on a monthly basis and then aggregated into a full-sample summary across all five sleeves.

Core Functions:
    - brinson_attribution
    - full_attribution

Two summary helpers (sleeve_summary, all_sleeves_summary) bundle the
metrics into a tidy DataFrame for the report.
"""

import numpy as np
import pandas as p