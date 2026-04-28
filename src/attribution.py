"""
attribution.py

Brinson performance attribution for the multi-asset portfolio.

The module decomposes each sleeve's active return into 2 components:    
    - Allocation effect = the impact of over/underweighing the asset classes relative to the benchmark 
    - Selection effect = the contribution from manager performance versus benchmarks

Effects are calculated on a monthly basis and then aggregated into a full-sample summary across all five sleeves.

Core Functions:
    - brinson_attribution   (computes monthly allocation and selection effects for an individual sleeve)
    - full_attribution      (applies the Brinson framework across all sleeves and returns a consolidated summary DataFrame)

Two summary helpers (sleeve_summary, all_sleeves_summary) bundle the
metrics into a tidy DataFrame for the report.
"""

import numpy as np
import pandas as pd

def attribution_summary(
    sleeve_name: str,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    taa_weights: float,
    saa_weights: float,
) -> dict:
    """
    Computes Brinson performance attribution for a single sleeve and
    returns the results as a labelled dictionary.

    Args:
        sleeve_name:        Identifier for the asset class (e.g. 'AUS_EQ').
        portfolio_returns:  Monthly manager returns for the sleeve.
        benchmark_returns:  Monthly benchmark returns for the sleeve.
        taa_weights:        Tactical Asset Allocation (TAA) weight for the sleeve.
        saa_weights:        Strategic Asset Allocation (SAA) weight for the sleeve.

    Returns:
        Dictionary containing attribution metrics for the sleeve,
        keyed by display name.
    """

    # ------ Allocation Effect -------
    allocation = (taa_weights - saa_weights) * benchmark_returns

    # ------ Selection Effect -------
    selection = (portfolio_returns - benchmark_returns) * saa_weights

    # ------ Total -------
    total = allocation + selection

    return {
        'Sleeve':            sleeve_name,
        'Allocation Effect': allocation.sum(),
        'Selection Effect':  selection.sum(),
        'Total':             total.sum(),
    }

def all_sleeves_attribution(
    manager_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    sleeves: list,
    taa_weights: dict,
    saa_weights: dict,
) -> pd.DataFrame:
    """
    Applies Brinson framework across all sleeves and
    returns a consolidated summary DataFrame.

    Args:
        sleeves:            List of sleeve names 
        manager_returns:    Monthly manager returns for the sleeve.
        benchmark_returns:  Monthly benchmark returns for the sleeve.
        taa_weights:        Tactical Asset Allocation (TAA) weight for the sleeve.
        saa_weights:        Strategic Asset Allocation (SAA) weight for the sleeve.

    Returns:
        DataFrame containing attribution metrics for all sleeves.
    """
    rows = [] 
    for sleeve in sleeves: 
        result = attribution_summary(
            sleeve_name=sleeve,
            portfolio_returns=manager_returns[sleeve],
            benchmark_returns=benchmark_returns[sleeve],
            taa_weights=taa_weights[sleeve],
            saa_weights=saa_weights[sleeve]
        )
        rows.append(result)  

    return pd.DataFrame(rows).set_index('Sleeve')  