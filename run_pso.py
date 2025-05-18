import numpy as np
import time
from module.pso import StandardPSO
from module.visualization import ackley, rastrigin, griewank, schwefel, levy, AnimatedPSOVisualizer

# Configuration parameters
#--------------------------

# Select function to optimize:
# Options: 'ackley', 'rastrigin', 'griewank', 'schwefel', 'levy'
function_name = 'ackley'  

# Number of particles in the swarm
num_particles = 30  

# Maximum number of iterations
max_iter = 100      

# Animation speed factor (higher = slower animation)
# 1.0 = normal speed, 2.0 = half speed, 0.5 = double speed
speed_factor = 2.5  

# Function mapping
function_map = {
    'ackley': ackley,       # Smooth function with many small local minima
    'rastrigin': rastrigin, # Many local minima in regular pattern (good for QPSO comparison)
    'griewank': griewank,   # Many widespread local minima
    'schwefel': schwefel,   # Global minimum far from next best local minima
    'levy': levy            # Highly irregular local minima spacing
}

# Select the function
function = function_map[function_name]

# Set appropriate bounds based on the function
if function_name == 'schwefel':
    bounds = (-500, 500)  # Schwefel has a much larger domain
elif function_name == 'levy':
    bounds = (-10, 10)    # Levy is typically evaluated in a smaller domain
else:
    bounds = (-5, 5)      # Standard bounds for Ackley, Rastrigin, Griewank

# Create visualizer
visualizer = AnimatedPSOVisualizer(
    function, 
    bounds, 
    f"Standard PSO - {function_name.capitalize()} Function",
    speed_factor=speed_factor
)

# Create PSO optimizer
print(f"Starting Standard PSO optimization on {function_name.capitalize()} function...")
print(f"Particles: {num_particles}, Max iterations: {max_iter}")
start_time = time.time()
pso = StandardPSO(function, num_particles, max_iter, bounds)

# Run optimization with visualization callback
results = pso.optimize(callback=visualizer.callback)

# Calculate execution time
execution_time = time.time() - start_time

# Display results
print("\nStandard PSO Results:")
print("=" * 50)
print(f"Function: {function_name.capitalize()}")
print(f"Global Best Position: {results['global_best_position']}")
print(f"Global Best Fitness: {results['global_best_fitness']:.6f}")
print(f"Execution Time: {execution_time:.6f} seconds")
print(f"Iterations: {results['iterations']}")

# Define the filename for your GIF
gif_filename = f"standard_pso_{function_name}_{num_particles}_particles.gif"

# Save the animation as a GIF
visualizer.save_animation(gif_filename)

# Optionally start animation to display it (can be commented out if you only want to save)
visualizer.start_animation()


