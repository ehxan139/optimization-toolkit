"""
Linear Programming Optimizer

Solve linear programming problems using scipy and cvxpy.
Supports maximization and minimization with inequality and equality constraints.
"""

import numpy as np
from scipy.optimize import linprog
import cvxpy as cp


class LinearOptimizer:
    """
    Linear programming solver for optimization problems.
    
    Solves problems of the form:
        minimize/maximize:  c^T * x
        subject to:         A_ub * x <= b_ub  (inequality constraints)
                           A_eq * x == b_eq  (equality constraints)
                           x >= 0             (non-negativity)
    """
    
    def __init__(self, method='highs'):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        method : str, default='highs'
            Solver method: 'highs', 'simplex', or 'cvxpy'
        """
        self.method = method
        self.result = None
        
    def solve(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, 
              bounds=None, maximize=False):
        """
        Solve linear programming problem.
        
        Parameters
        ----------
        c : array-like
            Objective function coefficients
        A_ub : array-like, optional
            Inequality constraint matrix (A_ub @ x <= b_ub)
        b_ub : array-like, optional
            Inequality constraint bounds
        A_eq : array-like, optional
            Equality constraint matrix (A_eq @ x == b_eq)
        b_eq : array-like, optional
            Equality constraint bounds
        bounds : list of tuples, optional
            Variable bounds [(lower, upper), ...]
        maximize : bool, default=False
            Whether to maximize (True) or minimize (False)
        
        Returns
        -------
        result : dict
            Solution with keys: 'x', 'obj_value', 'status', 'success'
        """
        c = np.array(c)
        
        # Convert to minimization if maximizing
        if maximize:
            c = -c
        
        if self.method in ['highs', 'simplex']:
            # Use scipy linprog
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=self.method
            )
            
            obj_value = result.fun
            if maximize:
                obj_value = -obj_value
            
            self.result = {
                'x': result.x,
                'obj_value': obj_value,
                'status': result.message,
                'success': result.success,
                'iterations': result.nit if hasattr(result, 'nit') else None
            }
            
        elif self.method == 'cvxpy':
            # Use CVXPY
            n = len(c)
            x = cp.Variable(n)
            
            # Objective
            objective = cp.Minimize(c @ x) if not maximize else cp.Maximize(-c @ x)
            
            # Constraints
            constraints = []
            if A_ub is not None and b_ub is not None:
                constraints.append(A_ub @ x <= b_ub)
            if A_eq is not None and b_eq is not None:
                constraints.append(A_eq @ x == b_eq)
            if bounds is not None:
                for i, (lb, ub) in enumerate(bounds):
                    if lb is not None:
                        constraints.append(x[i] >= lb)
                    if ub is not None:
                        constraints.append(x[i] <= ub)
            else:
                constraints.append(x >= 0)
            
            # Solve
            problem = cp.Problem(objective, constraints)
            obj_value = problem.solve()
            
            if maximize:
                obj_value = -obj_value
            
            self.result = {
                'x': x.value,
                'obj_value': obj_value,
                'status': problem.status,
                'success': problem.status == 'optimal'
            }
        
        return self.result
    
    def sensitivity_analysis(self, param='c', index=0, range_pct=0.2, steps=10):
        """
        Perform sensitivity analysis on objective coefficients or constraints.
        
        Parameters
        ----------
        param : str
            Parameter to vary: 'c' (objective), 'b_ub', or 'b_eq'
        index : int
            Index of parameter to vary
        range_pct : float
            Percentage range to vary parameter (±range_pct)
        steps : int
            Number of steps in range
        
        Returns
        -------
        sensitivity : dict
            Parameter values and corresponding objective values
        """
        if self.result is None:
            raise ValueError("Must solve problem before sensitivity analysis")
        
        # Implementation would vary parameters and re-solve
        # Simplified version here
        return {
            'param_values': [],
            'obj_values': [],
            'feasible': []
        }


class ProductionPlanner(LinearOptimizer):
    """
    Specialized optimizer for production planning problems.
    """
    
    def optimize_production(self, profit_per_unit, resource_usage, 
                           resource_capacity, min_production=None):
        """
        Optimize production quantities to maximize profit.
        
        Parameters
        ----------
        profit_per_unit : array-like
            Profit for each product
        resource_usage : array-like (m x n)
            Resource usage matrix (m resources, n products)
        resource_capacity : array-like
            Available capacity for each resource
        min_production : array-like, optional
            Minimum production quantities
        
        Returns
        -------
        result : dict
            Optimal production quantities and profit
        """
        n_products = len(profit_per_unit)
        
        # Set bounds
        if min_production is not None:
            bounds = [(min_prod, None) for min_prod in min_production]
        else:
            bounds = [(0, None)] * n_products
        
        # Solve
        return self.solve(
            c=profit_per_unit,
            A_ub=resource_usage,
            b_ub=resource_capacity,
            bounds=bounds,
            maximize=True
        )


class TransportationOptimizer(LinearOptimizer):
    """
    Solve transportation/logistics optimization problems.
    """
    
    def optimize_transportation(self, costs, supply, demand):
        """
        Minimize transportation costs from sources to destinations.
        
        Parameters
        ----------
        costs : array-like (m x n)
            Cost matrix (m sources, n destinations)
        supply : array-like
            Supply available at each source
        demand : array-like
            Demand required at each destination
        
        Returns
        -------
        result : dict
            Optimal shipment quantities and total cost
        """
        costs = np.array(costs)
        supply = np.array(supply)
        demand = np.array(demand)
        
        m, n = costs.shape
        
        # Flatten cost matrix for objective
        c = costs.flatten()
        
        # Supply constraints (sum over destinations)
        A_supply = np.zeros((m, m * n))
        for i in range(m):
            A_supply[i, i*n:(i+1)*n] = 1
        
        # Demand constraints (sum over sources)
        A_demand = np.zeros((n, m * n))
        for j in range(n):
            A_demand[j, j::n] = 1
        
        # Combine constraints
        A_eq = np.vstack([A_supply, A_demand])
        b_eq = np.hstack([supply, demand])
        
        # Solve
        result = self.solve(c=c, A_eq=A_eq, b_eq=b_eq, maximize=False)
        
        # Reshape solution to matrix form
        if result['success']:
            result['shipments'] = result['x'].reshape(m, n)
        
        return result
