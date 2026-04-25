"""
performance.py

Performance and risk metrics for the multi-asset portfolio.

Each metric is a small pure function that takes a return series (and
sometimes a benchmark or risk-free series) and returns a number.

Implements:
    - annualised_return        (geometric)
    - annualised_volatility    (sqrt-of-time scaled)
    - sharpe_ratio
    - active_return
    - tracking_error
    - information_ratio
    - max_drawdown
    - wealth_index             (helper for charts)

Two summary helpers (sleeve_summary, all_sleeves_summary) bundle the
metrics into a tidy DataFrame for the report.
"""

import numpy as np
import pandas as pd

# ── Single-series metrics ─────────────────────────────────────────────────────

def annualised_return(monthly_returns: pd.Series) -> float:
    """
    Geometric annualised return from a series of monthly returns.

    Formula: prod(1 + r_t) ^ (12/n) - 1
    where n is the number of monthly observations.

    Args:
        monthly_returns: monthly decimal returns (e.g. 0.012 = +1.2%).

    Returns:
        Annualised return as a decimal.
    """
    n = len(monthly_returns)
    growth_factor = (1 + monthly_returns).prod()
    return growth_factor ** (12 / n) - 1

def annualised_volatility(monthly_returns: pd.Series) -> float:
    """
    Annualised volatility from monthly returns. Scales monthly standard
    deviation by sqrt(12) (square-root-of-time rule).

    Formula: std(r) * sqrt(12)

    Args:
        monthly_returns: monthly decimal returns.

    Returns:
        Annualised volatility as a decimal.
    """
    return monthly_returns.std() * np.sqrt(12)

def max_drawdown(monthly_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of cumulative wealth.

    Formula: min over t of (W_t - peak(W)) / peak(W)
    where W_t is the cumulative wealth index.

    Args:
        monthly_returns: monthly decimal returns.

    Returns:
        Max drawdown as a (negative) decimal. e.g. -0.25 means -25%.
    """
    wealth = (1 + monthly_returns).cumprod()    # Growth-of-$1 path
    rolling_peak = wealth.cummax()              # Highest value seen so far
    drawdown = (wealth - rolling_peak) / rolling_peak
    return drawdown.min()

def wealth_index(monthly_returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """
    Cumulative wealth index from monthly returns. Useful for plotting
    'growth of $1' charts.

    Formula: W_t = start_value * cumprod(1 + r_t)

    Args:
        monthly_returns: monthly decimal returns.
        start_value:     starting wealth (default 1.0).

    Returns:
        Series of cumulative wealth values over time.
    """
    return start_value * (1 + monthly_returns).cumprod()

# ── Manager-vs-benchmark metrics ──────────────────────────────────────────────

def sharpe_ratio(monthly_returns: pd.Series, monthly_rf: pd.Series) -> float:
    """
    Sharpe ratio: excess return per unit of total volatility.

    Formula: (R_ann - Rf_ann) / sigma_ann

    Both the portfolio return and the risk-free rate are annualised
    geometrically before subtraction so the numerator is consistent.

    Args:
        monthly_returns: monthly portfolio returns.
        monthly_rf:      monthly risk-free rate series.

    Returns:
        Sharpe ratio (dimensionless).
    """
    portfolio_ann = annualised_return(monthly_returns)
    rf_ann        = annualised_return(monthly_rf)
    vol_ann       = annualised_volatility(monthly_returns)
    return (portfolio_ann - rf_ann) / vol_ann

def active_return(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Annualised active return — the annual amount by which the portfolio
    outperformed (or underperformed) its benchmark.

    Formula: mean(r_P - r_B) * 12

    We use arithmetic annualisation here (not geometric) because this
    return pairs with tracking error to form the Information Ratio. Both
    sides of that ratio must use the same annualisation method to be
    mathematically consistent.

    Args:
        portfolio_returns: monthly portfolio returns.
        benchmark_returns: monthly benchmark returns.

    Returns:
        Active return as a decimal.
    """
    active = portfolio_returns - benchmark_returns
    return active.mean() * 12