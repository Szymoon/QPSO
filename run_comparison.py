import numpy as np
import time
from module.pso import StandardPSO
from module.qpso import QuantumPSO
from module.visualization import ackley, rastrigin, griewank, schwefel, levy, f10_composite

# Configuration parameters
#--------------------------

# Select function to optimize:
# Options: 'ackley', 'rastrigin', 'griewank', 'schwefel', 'levy', 'f10_composite'
function_name = 'f10_composite'  

# Dimension of the problem (30+ recommended for the composite function)
dimensions = 30

# Number of particles in the swarm (more needed for higher dimensions)
num_particles = 50  

# Maximum number of iterations (need more for complex landscapes)
max_iter = 300      

# Number of independent runs for statistical comparison
num_runs = 10

# Function mapping
function_map = {
    'ackley': ackley,                # Smooth function with many small local minima
    'rastrigin': rastrigin,          # Many local minima in regular pattern
    'griewank': griewank,            # Many widespread local minima
    'schwefel': schwefel,            # Global minimum far from next best local minima
    'levy': levy,                    # Highly irregular local minima spacing
    'f10_composite': f10_composite   # Extremely challenging composite function
}

# Select the function
function = function_map[function_name]

# Set appropriate bounds based on the function
if function_name == 'schwefel':
    bounds = (-500, 500)  # Schwefel has a much larger domain
elif function_name == 'levy':
    bounds = (-10, 10)    # Levy is typically evaluated in a smaller domain
elif function_name == 'f10_composite':
    bounds = (-30, 30)    # Composite function has a larger domain
else:
    bounds = (-5, 5)      # Standard bounds for Ackley, Rastrigin, Griewank

# Define the high-dimensional function wrapper for any function
def high_dim_wrapper(func, dims):
    """Create a high-dimensional version of a function"""
    def wrapped_func(x):
        return func(x)
    return wrapped_func

# Create the high-dimensional version of the function
high_dim_function = high_dim_wrapper(function, dimensions)

# Define comparison function
def compare_algorithms(function, num_particles, max_iter, bounds, dimensions, num_runs):
    """
    Compare PSO and QPSO algorithms over multiple runs
    
    Parameters:
    -----------
    function: callable
        The objective function to optimize
    num_particles: int
        Number of particles to use
    max_iter: int
        Maximum number of iterations
    bounds: tuple
        (lower_bound, upper_bound) for search space
    dimensions: int
        Number of dimensions for the problem
    num_runs: int
        Number of independent runs for statistical comparison
    """
    # Initialize result containers
    pso_best_fitness = []
    pso_execution_times = []
    pso_iterations_to_converge = []
    
    qpso_best_fitness = []
    qpso_execution_times = []
    qpso_iterations_to_converge = []
    
    # Convergence threshold - adjust based on function and dimensions
    if function.__name__ == 'f10_composite':
        # For the composite function, convergence is harder, use a larger threshold
        convergence_threshold = 1.0 * dimensions / 10  
    elif function.__name__ == 'schwefel' or function.__name__ == 'levy':
        convergence_threshold = 0.1 * dimensions
    else:
        convergence_threshold = 0.01 * dimensions
    
    # Run multiple independent optimizations
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        
        # Run PSO
        start_time = time.time()
        pso = StandardPSO(function, num_particles, max_iter, bounds, dimensions=dimensions)
        pso_results = pso.optimize()
        pso_time = time.time() - start_time
        
        # Run QPSO
        start_time = time.time()
        qpso = QuantumPSO(function, num_particles, max_iter, bounds, dimensions=dimensions)
        qpso_results = qpso.optimize()
        qpso_time = time.time() - start_time
        
        # Save best fitness
        pso_best_fitness.append(pso_results['global_best_fitness'])
        qpso_best_fitness.append(qpso_results['global_best_fitness'])
        
        # Save execution time
        pso_execution_times.append(pso_time)
        qpso_execution_times.append(qpso_time)
        
        # Calculate iterations to convergence
        pso_history = pso_results['convergence_history']
        for j in range(len(pso_history)):
            if pso_history[j] < convergence_threshold:
                pso_iterations_to_converge.append(j)
                break
        else:
            pso_iterations_to_converge.append(max_iter)
            
        qpso_history = qpso_results['convergence_history']
        for j in range(len(qpso_history)):
            if qpso_history[j] < convergence_threshold:
                qpso_iterations_to_converge.append(j)
                break
        else:
            qpso_iterations_to_converge.append(max_iter)
    
    # Calculate statistics
    pso_avg_fitness = np.mean(pso_best_fitness)
    pso_std_fitness = np.std(pso_best_fitness)
    pso_avg_time = np.mean(pso_execution_times)
    pso_avg_iterations = np.mean(pso_iterations_to_converge)
    
    qpso_avg_fitness = np.mean(qpso_best_fitness)
    qpso_std_fitness = np.std(qpso_best_fitness)
    qpso_avg_time = np.mean(qpso_execution_times)
    qpso_avg_iterations = np.mean(qpso_iterations_to_converge)
    
    # Success rate (reaching convergence threshold)
    pso_success_rate = np.sum(np.array(pso_best_fitness) < convergence_threshold) / num_runs * 100
    qpso_success_rate = np.sum(np.array(qpso_best_fitness) < convergence_threshold) / num_runs * 100
    
    # Print results
    print("\nComparison Results:")
    print("=" * 50)
    print(f"Function: {function.__name__}")
    print(f"Dimensions: {dimensions}")
    print(f"Particles: {num_particles}")
    print(f"Convergence Threshold: {convergence_threshold}")
    print("\nPSO Results:")
    print(f"  Average Best Fitness: {pso_avg_fitness:.6f} ± {pso_std_fitness:.6f}")
    print(f"  Average Execution Time: {pso_avg_time:.6f} seconds")
    print(f"  Average Iterations to Converge: {pso_avg_iterations:.2f}")
    print(f"  Success Rate: {pso_success_rate:.2f}%")
    print("\nQPSO Results:")
    print(f"  Average Best Fitness: {qpso_avg_fitness:.6f} ± {qpso_std_fitness:.6f}")
    print(f"  Average Execution Time: {qpso_avg_time:.6f} seconds")
    print(f"  Average Iterations to Converge: {qpso_avg_iterations:.2f}")
    print(f"  Success Rate: {qpso_success_rate:.2f}%")
    
    # Additional metrics to compare the algorithms
    print("\nComparative Metrics:")
    print("=" * 50)
    
    # Compare fitness improvement
    fitness_improvement = (pso_avg_fitness - qpso_avg_fitness) / pso_avg_fitness * 100
    if fitness_improvement > 0:
        print(f"QPSO improved fitness by {fitness_improvement:.2f}% compared to PSO")
    else:
        print(f"PSO improved fitness by {-fitness_improvement:.2f}% compared to QPSO")
    
    # Compare convergence speed
    if pso_avg_iterations < qpso_avg_iterations and pso_avg_iterations < max_iter:
        print(f"PSO converged {qpso_avg_iterations/pso_avg_iterations:.2f}x faster than QPSO")
    elif qpso_avg_iterations < pso_avg_iterations and qpso_avg_iterations < max_iter:
        print(f"QPSO converged {pso_avg_iterations/qpso_avg_iterations:.2f}x faster than PSO")
    else:
        print("Neither algorithm consistently reached convergence within the iteration limit")
    
    # Compare execution time
    if pso_avg_time < qpso_avg_time:
        print(f"PSO was {qpso_avg_time/pso_avg_time:.2f}x faster in execution time than QPSO")
    else:
        print(f"QPSO was {pso_avg_time/qpso_avg_time:.2f}x faster in execution time than PSO")
    
    # Compare success rate
    print(f"Difference in success rate: {abs(pso_success_rate - qpso_success_rate):.2f}%")
    if pso_success_rate > qpso_success_rate:
        print(f"PSO had a higher success rate by {pso_success_rate - qpso_success_rate:.2f}%")
    else:
        print(f"QPSO had a higher success rate by {qpso_success_rate - pso_success_rate:.2f}%")
    
    return {
        'pso': {
            'avg_fitness': pso_avg_fitness,
            'std_fitness': pso_std_fitness,
            'success_rate': pso_success_rate
        },
        'qpso': {
            'avg_fitness': qpso_avg_fitness,
            'std_fitness': qpso_std_fitness,
            'success_rate': qpso_success_rate
        }
    }

# Run main comparison
print(f"Running PSO vs QPSO comparison on {function_name} function in {dimensions} dimensions")
print(f"Particles: {num_particles}, Max iterations: {max_iter}, Number of runs: {num_runs}")
results = compare_algorithms(high_dim_function, num_particles, max_iter, bounds, dimensions, num_runs)

# Try different dimensionality
print("\n" + "=" * 50)
print("Running comparison with different dimensions:")
dimensions_to_test = [10, 30, 50, 100]  # Test various dimensionality

results_by_dim = {}
for dim in dimensions_to_test:
    if dim == dimensions:  # Skip the dimension we already tested
        continue
    print(f"\nTesting with {dim} dimensions:")
    high_dim_func = high_dim_wrapper(function, dim)
    results_by_dim[dim] = compare_algorithms(high_dim_func, num_particles, max_iter, bounds, dim, 3)

# Print summary of dimensional scaling
print("\n" + "=" * 50)
print("Summary of Scaling with Dimensions:")
print("-" * 50)
print(f"{'Dimensions':<10} {'PSO Fitness':<20} {'QPSO Fitness':<20} {'QPSO Improvement':<20}")
print("-" * 50)

for dim in sorted(results_by_dim.keys()):
    pso_fitness = results_by_dim[dim]['pso']['avg_fitness']
    qpso_fitness = results_by_dim[dim]['qpso']['avg_fitness']
    improvement = (pso_fitness - qpso_fitness) / pso_fitness * 100 if pso_fitness > 0 else 0
    print(f"{dim:<10} {pso_fitness:<20.6f} {qpso_fitness:<20.6f} {improvement:<20.2f}%")