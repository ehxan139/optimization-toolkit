"""
Integer Programming Optimizer

Solve integer and mixed-integer programming problems for discrete optimization.
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import cvxpy as cp


class IntegerOptimizer:
    """
    Integer and mixed-integer programming solver.
    
    Solves problems with integer decision variables for problems like:
    - Facility location
    - Assignment problems
    - Knapsack problems
    - Project selection
    """
    
    def __init__(self):
        self.result = None
        
    def solve_knapsack(self, values, weights, capacity):
        """
        Solve 0-1 knapsack problem.
        
        Parameters
        ----------
        values : array-like
            Value of each item
        weights : array-like
            Weight of each item
        capacity : float
            Knapsack capacity
        
        Returns
        -------
        result : dict
            Selected items and total value
        """
        n = len(values)
        values = np.array(values)
        weights = np.array(weights)
        
        # Decision variables (binary)
        x = cp.Variable(n, boolean=True)
        
        # Objective: maximize value
        objective = cp.Maximize(values @ x)
        
        # Constraints: weight <= capacity
        constraints = [weights @ x <= capacity]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        optimal_value = problem.solve(solver=cp.GLPK_MI)
        
        self.result = {
            'selected_items': x.value,
            'total_value': optimal_value,
            'total_weight': np.sum(weights * x.value),
            'status': problem.status,
            'success': problem.status == 'optimal'
        }
        
        return self.result
    
    def solve_assignment(self, cost_matrix):
        """
        Solve assignment problem (Hungarian algorithm).
        
        Assign n workers to n jobs minimizing total cost.
        
        Parameters
        ----------
        cost_matrix : array-like (n x n)
            Cost of assigning worker i to job j
        
        Returns
        -------
        result : dict
            Optimal assignment and total cost
        """
        from scipy.optimize import linear_sum_assignment
        
        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        total_cost = cost_matrix[row_ind, col_ind].sum()
        
        # Create assignment matrix
        n = len(cost_matrix)
        assignment = np.zeros((n, n), dtype=int)
        assignment[row_ind, col_ind] = 1
        
        self.result = {
            'assignment_matrix': assignment,
            'worker_to_job': dict(zip(row_ind, col_ind)),
            'total_cost': total_cost,
            'success': True
        }
        
        return self.result
    
    def solve_facility_location(self, fixed_costs, variable_costs, demand, capacity):
        """
        Solve facility location problem.
        
        Decide which facilities to open and how to assign customers.
        
        Parameters
        ----------
        fixed_costs : array-like
            Fixed cost of opening each facility
        variable_costs : array-like (m x n)
            Cost of serving customer j from facility i
        demand : array-like
            Demand from each customer
        capacity : array-like
            Capacity of each facility
        
        Returns
        -------
        result : dict
            Facilities to open and customer assignments
        """
        fixed_costs = np.array(fixed_costs)
        variable_costs = np.array(variable_costs)
        demand = np.array(demand)
        capacity = np.array(capacity)
        
        m, n = variable_costs.shape  # m facilities, n customers
        
        # Decision variables
        y = cp.Variable(m, boolean=True)  # facility open/closed
        x = cp.Variable((m, n), boolean=True)  # customer assignments
        
        # Objective: minimize total cost
        fixed_cost_total = fixed_costs @ y
        variable_cost_total = cp.sum(cp.multiply(variable_costs, x))
        objective = cp.Minimize(fixed_cost_total + variable_cost_total)
        
        # Constraints
        constraints = []
        
        # Each customer assigned to exactly one facility
        for j in range(n):
            constraints.append(cp.sum(x[:, j]) == 1)
        
        # Facility capacity constraints
        for i in range(m):
            constraints.append(demand @ x[i, :] <= capacity[i] * y[i])
        
        # Can only assign to open facilities
        for i in range(m):
            for j in range(n):
                constraints.append(x[i, j] <= y[i])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        optimal_cost = problem.solve(solver=cp.GLPK_MI)
        
        self.result = {
            'open_facilities': y.value,
            'assignments': x.value,
            'total_cost': optimal_cost,
            'status': problem.status,
            'success': problem.status == 'optimal'
        }
        
        return self.result
    
    def solve_project_selection(self, npvs, costs, budget, dependencies=None):
        """
        Select optimal set of projects given budget constraint.
        
        Parameters
        ----------
        npvs : array-like
            Net present value of each project
        costs : array-like
            Cost of each project
        budget : float
            Available budget
        dependencies : list of tuples, optional
            (i, j) means project j requires project i
        
        Returns
        -------
        result : dict
            Selected projects and total NPV
        """
        n = len(npvs)
        npvs = np.array(npvs)
        costs = np.array(costs)
        
        # Decision variables
        x = cp.Variable(n, boolean=True)
        
        # Objective: maximize NPV
        objective = cp.Maximize(npvs @ x)
        
        # Constraints
        constraints = [costs @ x <= budget]
        
        # Dependency constraints
        if dependencies is not None:
            for i, j in dependencies:
                constraints.append(x[j] <= x[i])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        optimal_npv = problem.solve(solver=cp.GLPK_MI)
        
        self.result = {
            'selected_projects': x.value,
            'total_npv': optimal_npv,
            'total_cost': np.sum(costs * x.value),
            'status': problem.status,
            'success': problem.status == 'optimal'
        }
        
        return self.result


class SchedulingOptimizer(IntegerOptimizer):
    """
    Workforce and resource scheduling optimizer.
    """
    
    def optimize_shift_scheduling(self, costs, requirements, max_hours=None):
        """
        Assign workers to shifts minimizing cost.
        
        Parameters
        ----------
        costs : array-like (n_workers x n_shifts)
            Cost of assigning worker i to shift j
        requirements : array-like
            Number of workers required for each shift
        max_hours : array-like, optional
            Maximum hours for each worker
        
        Returns
        -------
        result : dict
            Optimal shift assignments
        """
        costs = np.array(costs)
        requirements = np.array(requirements)
        n_workers, n_shifts = costs.shape
        
        # Decision variables
        x = cp.Variable((n_workers, n_shifts), boolean=True)
        
        # Objective: minimize cost
        objective = cp.Minimize(cp.sum(cp.multiply(costs, x)))
        
        # Constraints
        constraints = []
        
        # Meet shift requirements
        for j in range(n_shifts):
            constraints.append(cp.sum(x[:, j]) >= requirements[j])
        
        # Worker hour limits
        if max_hours is not None:
            for i in range(n_workers):
                constraints.append(cp.sum(x[i, :]) <= max_hours[i])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        optimal_cost = problem.solve(solver=cp.GLPK_MI)
        
        self.result = {
            'assignments': x.value,
            'total_cost': optimal_cost,
            'status': problem.status,
            'success': problem.status == 'optimal'
        }
        
        return self.result
