"""
Portfolio Optimization Example

Demonstrates modern portfolio theory optimization techniques.
"""

import sys
sys.path.append('..')
from src.portfolio_optimizer import PortfolioOptimizer
import pandas as pd
import numpy as np


def generate_sample_returns(n_assets=5, n_periods=252):
    """Generate sample return data."""
    np.random.seed(42)

    # Simulated daily returns
    mean_returns = np.random.uniform(0.0001, 0.0015, n_assets)
    cov_matrix = np.random.uniform(0.0001, 0.0003, (n_assets, n_assets))
    cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(cov_matrix, 0.0004)  # Set variances

    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)

    asset_names = [f"Stock_{chr(65+i)}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=asset_names)


def main():
    """
    Example: Optimize portfolio of 5 stocks.
    """

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    # Generate sample data
    returns = generate_sample_returns()

    print(f"\nHistorical Returns (annualized):")
    annual_returns = returns.mean() * 252
    for asset, ret in annual_returns.items():
        print(f"  {asset}: {ret*100:.2f}%")

    # Create optimizer
    optimizer = PortfolioOptimizer(returns)

    # 1. Maximum Sharpe Ratio
    print("\n" + "-" * 60)
    print("1. MAXIMUM SHARPE RATIO PORTFOLIO")
    print("-" * 60)

    weights_sharpe = optimizer.maximize_sharpe_ratio(risk_free_rate=0.02/252)
    metrics_sharpe = optimizer.get_portfolio_metrics(weights_sharpe, risk_free_rate=0.02/252)

    print("\nOptimal Weights:")
    for i, (asset, weight) in enumerate(zip(returns.columns, weights_sharpe)):
        if weight > 0.01:
            print(f"  {asset}: {weight*100:.2f}%")

    print(f"\nExpected Annual Return: {metrics_sharpe['return']*252*100:.2f}%")
    print(f"Annual Volatility: {metrics_sharpe['volatility']*np.sqrt(252)*100:.2f}%")
    print(f"Sharpe Ratio: {metrics_sharpe['sharpe']*np.sqrt(252):.2f}")

    # 2. Minimum Variance
    print("\n" + "-" * 60)
    print("2. MINIMUM VARIANCE PORTFOLIO")
    print("-" * 60)

    weights_minvar = optimizer.minimize_variance()
    metrics_minvar = optimizer.get_portfolio_metrics(weights_minvar)

    print("\nOptimal Weights:")
    for asset, weight in zip(returns.columns, weights_minvar):
        if weight > 0.01:
            print(f"  {asset}: {weight*100:.2f}%")

    print(f"\nExpected Annual Return: {metrics_minvar['return']*252*100:.2f}%")
    print(f"Annual Volatility: {metrics_minvar['volatility']*np.sqrt(252)*100:.2f}%")

    # 3. Risk Parity
    print("\n" + "-" * 60)
    print("3. RISK PARITY PORTFOLIO")
    print("-" * 60)

    weights_rp = optimizer.risk_parity()
    metrics_rp = optimizer.get_portfolio_metrics(weights_rp)

    print("\nOptimal Weights:")
    for asset, weight in zip(returns.columns, weights_rp):
        print(f"  {asset}: {weight*100:.2f}%")

    print(f"\nExpected Annual Return: {metrics_rp['return']*252*100:.2f}%")
    print(f"Annual Volatility: {metrics_rp['volatility']*np.sqrt(252)*100:.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
