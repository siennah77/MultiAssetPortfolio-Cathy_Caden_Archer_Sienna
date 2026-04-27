import pandas as pd
import numpy as np


# Core portfolio metrics 

def _annualised_return(returns: pd.Series) -> float:
    """
    This calculates the annualised return using geometric compounding.

    Args: 
        returns: Series of monthly returns.

    Returns:
        Annualised return (float).
    """
    returns = returns.dropna()
    return (1 + returns).prod() ** (12 / len(returns)) - 1

def _annualised_volatility(returns: pd.Series) -> float:
    """
    Computes annualised volatility from monthly returns.

    Args:
        returns: Series of monthly returns.

    Returns:
        Annualised standard deviation (float).
    """
    returns = returns.dropna()
    return returns.std()*np.sqrt(12)


def _max_drawdown(returns: pd.Series) -> float:
    """
    This calculates the maximum drawdown (peak-to-trough loss).

    Args:
        returns: Series of monthly returns.

    Returns:
        Maximum drawdown (negative float).
    """
    returns = returns.dropna()

    wealth_index = (1 + returns).cumprod()
    previous_peak = wealth_index.cummax()
    drawdown = (wealth_index - previous_peak) / previous_peak

    return drawdown.min()