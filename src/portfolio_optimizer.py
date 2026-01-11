"""
Portfolio Optimization

Modern portfolio theory implementations including mean-variance optimization,
risk parity, and maximum Sharpe ratio portfolios.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Portfolio optimization using modern portfolio theory.
    
    Implements various portfolio construction strategies:
    - Mean-variance optimization (Markowitz)
    - Maximum Sharpe ratio
    - Minimum variance
    - Risk parity
    """
    
    def __init__(self, returns):
        """
        Initialize optimizer with historical returns.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns (dates x assets)
        """
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(self.mean_returns)
        
    def maximize_sharpe_ratio(self, risk_free_rate=0.0, constraints=None):
        """
        Find portfolio with maximum Sharpe ratio.
        
        Parameters
        ----------
        risk_free_rate : float
            Risk-free rate for Sharpe calculation
        constraints : dict, optional
            Additional constraints (e.g., {'long_only': True})
        
        Returns
        -------
        weights : array
            Optimal portfolio weights
        """
        # Define optimization problem
        w = cp.Variable(self.n_assets)
        
        # Portfolio return and risk
        port_return = self.mean_returns.values @ w
        port_variance = cp.quad_form(w, self.cov_matrix.values)
        
        # Objective: maximize (return - rf) / sqrt(variance)
        # Equivalent to maximizing return for target variance
        objective = cp.Maximize(port_return - risk_free_rate)
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,  # fully invested
            port_variance <= 1  # variance constraint (normalized)
        ]
        
        if constraints is None or constraints.get('long_only', True):
            constraints_list.append(w >= 0)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        # Rescale to maximize Sharpe
        weights = w.value
        weights = weights / np.sum(np.abs(weights))  # normalize
        
        return weights
    
    def minimize_variance(self, target_return=None, long_only=True):
        """
        Find minimum variance portfolio.
        
        Parameters
        ----------
        target_return : float, optional
            Target return constraint
        long_only : bool
            Whether to restrict to long positions only
        
        Returns
        -------
        weights : array
            Optimal portfolio weights
        """
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize variance
        port_variance = cp.quad_form(w, self.cov_matrix.values)
        objective = cp.Minimize(port_variance)
        
        # Constraints
        constraints_list = [cp.sum(w) == 1]
        
        if target_return is not None:
            port_return = self.mean_returns.values @ w
            constraints_list.append(port_return >= target_return)
        
        if long_only:
            constraints_list.append(w >= 0)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        return w.value
    
    def efficient_frontier(self, n_points=50, long_only=True):
        """
        Calculate efficient frontier.
        
        Parameters
        ----------
        n_points : int
            Number of points on frontier
        long_only : bool
            Whether to restrict to long positions
        
        Returns
        -------
        frontier : pd.DataFrame
            Returns, risks, and weights for frontier portfolios
        """
        # Get range of returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_portfolios = []
        
        for target_ret in target_returns:
            try:
                weights = self.minimize_variance(target_return=target_ret, long_only=long_only)
                
                if weights is not None:
                    port_return = np.dot(weights, self.mean_returns)
                    port_risk = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
                    
                    frontier_portfolios.append({
                        'return': port_return,
                        'risk': port_risk,
                        'weights': weights
                    })
            except:
                continue
        
        return pd.DataFrame(frontier_portfolios)
    
    def risk_parity(self):
        """
        Calculate risk parity portfolio (equal risk contribution).
        
        Returns
        -------
        weights : array
            Risk parity portfolio weights
        """
        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset."""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def objective(weights):
            """Minimize difference in risk contributions."""
            risk_contrib = risk_contribution(weights, self.cov_matrix.values)
            return np.sum((risk_contrib - risk_contrib.mean()) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # fully invested
        ]
        
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Initial guess (equal weights)
        w0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    def get_portfolio_metrics(self, weights, risk_free_rate=0.0):
        """
        Calculate portfolio performance metrics.
        
        Parameters
        ----------
        weights : array-like
            Portfolio weights
        risk_free_rate : float
            Risk-free rate
        
        Returns
        -------
        metrics : dict
            Portfolio metrics (return, volatility, Sharpe ratio, etc.)
        """
        weights = np.array(weights)
        
        # Return and risk
        port_return = np.dot(weights, self.mean_returns)
        port_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        port_volatility = np.sqrt(port_variance)
        
        # Sharpe ratio
        sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(self.cov_matrix))
        weighted_vol = np.dot(weights, asset_vols)
        diversification_ratio = weighted_vol / port_volatility if port_volatility > 0 else 0
        
        return {
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe,
            'diversification_ratio': diversification_ratio,
            'weights': weights
        }
    
    def black_litterman(self, market_caps, views, view_confidences, tau=0.05, risk_aversion=2.5):
        """
        Black-Litterman portfolio optimization incorporating market equilibrium and views.
        
        Parameters
        ----------
        market_caps : array-like
            Market capitalizations of assets
        views : dict
            Expected returns views {asset_index: expected_return}
        view_confidences : dict
            Confidence in each view {asset_index: confidence}
        tau : float
            Uncertainty in prior
        risk_aversion : float
            Risk aversion coefficient
        
        Returns
        -------
        weights : array
            Black-Litterman optimal weights
        """
        # Market equilibrium returns (reverse optimization)
        market_weights = market_caps / np.sum(market_caps)
        pi = risk_aversion * np.dot(self.cov_matrix, market_weights)
        
        # Incorporate views
        # Simplified implementation
        posterior_returns = pi.copy()
        
        for asset_idx, view_return in views.items():
            confidence = view_confidences[asset_idx]
            posterior_returns[asset_idx] = (
                confidence * view_return + (1 - confidence) * pi[asset_idx]
            )
        
        # Optimize using posterior returns
        w = cp.Variable(self.n_assets)
        port_return = posterior_returns @ w
        port_variance = cp.quad_form(w, self.cov_matrix.values)
        
        objective = cp.Maximize(port_return - (risk_aversion / 2) * port_variance)
        constraints = [cp.sum(w) == 1, w >= 0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
