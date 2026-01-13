"""
Utility functions for optimization problems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_feasibility(A_ub, b_ub, A_eq, b_eq, x):
    """
    Check if solution satisfies constraints.

    Parameters
    ----------
    A_ub : array-like
        Inequality constraint matrix
    b_ub : array-like
        Inequality bounds
    A_eq : array-like
        Equality constraint matrix
    b_eq : array-like
        Equality bounds
    x : array-like
        Solution to check

    Returns
    -------
    feasible : bool
        Whether solution is feasible
    """
    tol = 1e-6

    # Check inequality constraints
    if A_ub is not None and b_ub is not None:
        if not np.all(np.dot(A_ub, x) <= b_ub + tol):
            return False

    # Check equality constraints
    if A_eq is not None and b_eq is not None:
        if not np.allclose(np.dot(A_eq, x), b_eq, atol=tol):
            return False

    return True


def plot_efficient_frontier(frontier_df, risk_free_rate=0.0):
    """
    Plot portfolio efficient frontier.

    Parameters
    ----------
    frontier_df : pd.DataFrame
        Efficient frontier data with 'return' and 'risk' columns
    risk_free_rate : float
        Risk-free rate for capital allocation line
    """
    plt.figure(figsize=(10, 6))

    # Plot frontier
    plt.plot(frontier_df['risk'], frontier_df['return'], 'b-', linewidth=2, label='Efficient Frontier')

    # Plot capital allocation line (tangency portfolio)
    if len(frontier_df) > 0:
        sharpe_ratios = (frontier_df['return'] - risk_free_rate) / frontier_df['risk']
        max_sharpe_idx = sharpe_ratios.idxmax()
        tangency = frontier_df.loc[max_sharpe_idx]

        plt.scatter(tangency['risk'], tangency['return'], c='red', s=100, marker='*',
                   label='Max Sharpe Ratio', zorder=5)

        # Capital allocation line
        cal_x = np.linspace(0, frontier_df['risk'].max(), 100)
        cal_y = risk_free_rate + (tangency['return'] - risk_free_rate) / tangency['risk'] * cal_x
        plt.plot(cal_x, cal_y, 'r--', alpha=0.5, label='Capital Allocation Line')

    plt.xlabel('Risk (Volatility)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def generate_random_portfolios(mean_returns, cov_matrix, n_portfolios=10000):
    """
    Generate random portfolios for comparison.

    Parameters
    ----------
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix
    n_portfolios : int
        Number of random portfolios to generate

    Returns
    -------
    portfolios : pd.DataFrame
        Random portfolio returns and risks
    """
    n_assets = len(mean_returns)
    results = []

    for _ in range(n_portfolios):
        # Random weights
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        # Portfolio metrics
        port_return = np.dot(weights, mean_returns)
        port_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        results.append({
            'return': port_return,
            'risk': port_risk,
            'weights': weights
        })

    return pd.DataFrame(results)


def format_weights(weights, asset_names=None, threshold=0.01):
    """
    Format portfolio weights for display.

    Parameters
    ----------
    weights : array-like
        Portfolio weights
    asset_names : list, optional
        Asset names
    threshold : float
        Minimum weight to display

    Returns
    -------
    formatted : pd.DataFrame
        Formatted weights
    """
    if asset_names is None:
        asset_names = [f"Asset {i+1}" for i in range(len(weights))]

    df = pd.DataFrame({
        'Asset': asset_names,
        'Weight': weights
    })

    # Filter small weights
    df = df[df['Weight'] >= threshold].copy()
    df = df.sort_values('Weight', ascending=False)
    df['Weight %'] = (df['Weight'] * 100).round(2)

    return df


def sensitivity_to_returns(optimizer, base_returns, variations=[-0.02, -0.01, 0, 0.01, 0.02]):
    """
    Analyze portfolio sensitivity to return assumptions.

    Parameters
    ----------
    optimizer : PortfolioOptimizer
        Portfolio optimizer instance
    base_returns : array-like
        Base case expected returns
    variations : list
        Return variations to test

    Returns
    -------
    sensitivity : pd.DataFrame
        Portfolio metrics for each variation
    """
    results = []

    for var in variations:
        # Adjust returns
        adj_returns = base_returns + var
        temp_optimizer = optimizer.__class__(optimizer.returns)
        temp_optimizer.mean_returns = adj_returns

        # Optimize
        weights = temp_optimizer.maximize_sharpe_ratio()
        metrics = temp_optimizer.get_portfolio_metrics(weights)
        metrics['return_adjustment'] = var

        results.append(metrics)

    return pd.DataFrame(results)
