"""
Production Planning Example

Demonstrates optimization of production quantities to maximize profit
subject to resource constraints.
"""

import sys
sys.path.append('..')
from src.linear_optimizer import ProductionPlanner
import numpy as np


def main():
    """
    Example: A factory produces two products (A and B).
    
    Resources:
    - Labor hours: 100 hours available
    - Machine time: 150 hours available
    - Raw material: 120 units available
    
    Product A:
    - Profit: $40 per unit
    - Requires: 1 labor hour, 2 machine hours, 1 material unit
    
    Product B:
    - Profit: $30 per unit
    - Requires: 1 labor hour, 1 machine hour, 2 material units
    """
    
    print("=" * 60)
    print("PRODUCTION PLANNING OPTIMIZATION")
    print("=" * 60)
    
    # Define problem
    profit_per_unit = [40, 30]  # Product A, Product B
    
    resource_usage = [
        [1, 1],  # Labor hours per unit
        [2, 1],  # Machine hours per unit
        [1, 2],  # Material units per unit
    ]
    
    resource_capacity = [100, 150, 120]  # Available resources
    
    # Create optimizer
    planner = ProductionPlanner()
    
    # Solve
    result = planner.optimize_production(
        profit_per_unit=profit_per_unit,
        resource_usage=resource_usage,
        resource_capacity=resource_capacity
    )
    
    # Display results
    if result['success']:
        print("\n✓ Optimal Solution Found\n")
        print(f"Product A: {result['x'][0]:.2f} units")
        print(f"Product B: {result['x'][1]:.2f} units")
        print(f"\nMaximum Profit: ${result['obj_value']:.2f}")
        
        # Calculate resource utilization
        usage = np.dot(resource_usage, result['x'])
        print("\nResource Utilization:")
        resources = ['Labor', 'Machine', 'Material']
        for i, resource in enumerate(resources):
            pct = (usage[i] / resource_capacity[i]) * 100
            print(f"  {resource}: {usage[i]:.2f} / {resource_capacity[i]} ({pct:.1f}%)")
    else:
        print("No feasible solution found")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
