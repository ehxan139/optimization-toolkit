"""
Resource Allocation Example

Demonstrates workforce scheduling and assignment optimization.
"""

import sys
sys.path.append('..')
from src.integer_optimizer import IntegerOptimizer, SchedulingOptimizer
import numpy as np


def knapsack_example():
    """
    Example: Pack a knapsack with maximum value.

    Items with different values and weights, limited capacity.
    """
    print("\n" + "=" * 60)
    print("KNAPSACK PROBLEM")
    print("=" * 60)

    # Define items
    items = ['Laptop', 'Camera', 'Book', 'Tablet', 'Phone']
    values = [1000, 500, 50, 600, 800]
    weights = [3, 2, 1, 2, 2]
    capacity = 5

    print(f"\nKnapsack Capacity: {capacity} kg")
    print("\nAvailable Items:")
    for item, value, weight in zip(items, values, weights):
        print(f"  {item}: ${value}, {weight}kg")

    # Solve
    optimizer = IntegerOptimizer()
    result = optimizer.solve_knapsack(values, weights, capacity)

    if result['success']:
        print("\n✓ Optimal Solution Found\n")
        print("Selected Items:")
        for i, (item, selected) in enumerate(zip(items, result['selected_items'])):
            if selected > 0.5:  # Binary variable
                print(f"  - {item} (${values[i]}, {weights[i]}kg)")

        print(f"\nTotal Value: ${result['total_value']:.0f}")
        print(f"Total Weight: {result['total_weight']:.1f} kg")


def assignment_example():
    """
    Example: Assign workers to tasks minimizing total cost.
    """
    print("\n" + "=" * 60)
    print("ASSIGNMENT PROBLEM")
    print("=" * 60)

    # Cost matrix (workers x tasks)
    workers = ['Alice', 'Bob', 'Charlie', 'Diana']
    tasks = ['Task1', 'Task2', 'Task3', 'Task4']

    cost_matrix = [
        [15, 10, 9, 10],   # Alice's costs
        [9, 15, 10, 12],   # Bob's costs
        [10, 12, 8, 8],    # Charlie's costs
        [8, 8, 9, 11],     # Diana's costs
    ]

    print("\nCost Matrix:")
    print(f"{'Worker':<10} " + " ".join(f"{t:<8}" for t in tasks))
    for worker, costs in zip(workers, cost_matrix):
        print(f"{worker:<10} " + " ".join(f"${c:<7}" for c in costs))

    # Solve
    optimizer = IntegerOptimizer()
    result = optimizer.solve_assignment(cost_matrix)

    if result['success']:
        print("\n✓ Optimal Assignment Found\n")
        for worker_id, task_id in result['worker_to_job'].items():
            print(f"  {workers[worker_id]} → {tasks[task_id]} (${cost_matrix[worker_id][task_id]})")

        print(f"\nTotal Cost: ${result['total_cost']:.2f}")


def shift_scheduling_example():
    """
    Example: Schedule workers to shifts meeting requirements.
    """
    print("\n" + "=" * 60)
    print("SHIFT SCHEDULING")
    print("=" * 60)

    # Workers and shifts
    workers = ['Worker1', 'Worker2', 'Worker3', 'Worker4', 'Worker5']
    shifts = ['Morning', 'Afternoon', 'Evening', 'Night']

    # Cost of assigning each worker to each shift
    costs = [
        [10, 12, 15, 20],  # Worker1 prefers morning
        [12, 10, 12, 18],  # Worker2 prefers afternoon
        [15, 13, 10, 15],  # Worker3 prefers evening
        [18, 15, 12, 10],  # Worker4 prefers night
        [11, 11, 11, 11],  # Worker5 neutral
    ]

    # Minimum workers needed per shift
    requirements = [2, 2, 2, 1]

    print(f"\nShift Requirements: {dict(zip(shifts, requirements))}")

    # Solve
    optimizer = SchedulingOptimizer()
    result = optimizer.optimize_shift_scheduling(costs, requirements, max_hours=[2, 2, 2, 2, 1])

    if result['success']:
        print("\n✓ Optimal Schedule Found\n")
        for j, shift in enumerate(shifts):
            assigned = [workers[i] for i in range(len(workers)) if result['assignments'][i][j] > 0.5]
            print(f"  {shift}: {', '.join(assigned)}")

        print(f"\nTotal Cost: ${result['total_cost']:.2f}")


def main():
    """Run all examples."""
    knapsack_example()
    assignment_example()
    shift_scheduling_example()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
