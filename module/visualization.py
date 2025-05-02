import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

def ackley(x):
    """
    Ackley function - a common benchmark for optimization algorithms
    Global minimum at (0,0) with value 0
    
    Parameters:
    -----------
    x: numpy array
        Point coordinates [x, y]
        
    Returns:
    --------
    float
        Function value at point x
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    # Ensure x is a numpy array
    x = np.array(x)
    
    term1 = -a * np.exp(-b * np.sqrt(0.5 * np.sum(x**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x[0]) + np.cos(c * x[1])))
    
    return term1 + term2 + a + np.exp(1)

# Add more challenging benchmark functions where QPSO often outperforms PSO
def rastrigin(x):
    """
    Rastrigin function - highly multimodal with many local minima
    Global minimum at (0,0) with value 0
    """
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def griewank(x):
    """
    Griewank function - many widespread local minima
    Global minimum at (0,0) with value 0
    """
    sum_part = sum([xi**2 for xi in x]) / 4000
    prod_part = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_part - prod_part + 1

def schwefel(x):
    """
    Schwefel function - the global minimum is far from the next best local minima
    Global minimum at (420.9687,...,420.9687) with value 0
    """
    n = len(x)
    return 418.9829 * n - sum([xi * np.sin(np.sqrt(np.abs(xi))) for xi in x])

def levy(x):
    """
    Levy function - another challenging multimodal function
    Global minimum at (1,1) with value 0
    """
    # Convert input to numpy array if it's not already
    x = np.array(x)
    
    # Initialize w with the correct shape
    w = 1 + (x - 1) / 4
    
    # Compute the first term
    term1 = np.sin(np.pi * w[0])**2
    
    # Compute the last term
    term2 = ((w[-1] - 1)**2) * (1 + np.sin(2 * np.pi * w[-1])**2)
    
    # Compute the middle sum term
    sum_term = 0
    for i in range(len(w)-1):
        sum_term += (w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2)
    
    return term1 + sum_term + term2

# Add this to module/visualization.py

def f10_composite(x):
    """
    A challenging composite function combining multiple difficult optimization characteristics:
    - Highly multimodal (many local minima)
    - Rotated landscape
    - Non-separable variables (changing one variable affects fitness of others)
    - Oscillatory component
    
    Global minimum at origin with value 0
    Best tested in 30+ dimensions
    """
    dim = len(x)
    
    # Rastrigin component - provides multimodality
    rastrigin_part = 10 * dim
    for i in range(dim):
        rastrigin_part += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    
    # Rosenbrock component - provides a narrow valley
    rosenbrock_part = 0
    for i in range(dim-1):
        rosenbrock_part += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    
    # Rotated high-conditioning component
    rot_part = 0
    for i in range(dim):
        for j in range(i, dim):
            # Correlation between variables through rotation matrix approximation
            rot_part += 0.5 * x[i] * x[j] * np.cos(i * j / (dim + 1.0))
    
    # Oscillatory component - creates deceptive landscapes
    osc_part = 0
    for i in range(dim):
        osc_part += np.sin(x[i]) * np.sin(i * x[i]**2 / np.pi)**20
    
    # Combine all components with weighting
    result = (0.4 * rastrigin_part / 100.0 + 
              0.2 * rosenbrock_part / 400.0 + 
              0.2 * rot_part + 
              0.2 * osc_part)
    
    return result

def create_function_landscape(function, bounds, n_points=100):
    """
    Create grid data for any 2D function
    
    Parameters:
    -----------
    function: callable
        The function to visualize
    bounds: tuple
        (lower_bound, upper_bound) for search space
    n_points: int
        Number of points along each dimension
        
    Returns:
    --------
    X, Y, Z: numpy arrays
        Grid data for plotting
    """
    x = np.linspace(bounds[0], bounds[1], n_points)
    y = np.linspace(bounds[0], bounds[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values for each point in the grid
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = function([X[i, j], Y[i, j]])
            
    return X, Y, Z

class AnimatedPSOVisualizer:
    def __init__(self, function, bounds, title="Particle Swarm Optimization", speed_factor=1.0):
        """
        Initialize the animated PSO visualizer
        
        Parameters:
        -----------
        function: callable
            The objective function to visualize
        bounds: tuple
            (lower_bound, upper_bound) for the plot
        title: str
            Title of the plot
        speed_factor: float
            Controls the animation speed (higher = slower animation)
            Values > 1 slow down the animation, values < 1 speed it up
        """
        self.function = function
        self.bounds = bounds
        self.title = title
        self.speed_factor = speed_factor
        
        # Base interval is 100ms, adjusted by speed factor
        self.interval = int(100 * speed_factor)
        
        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Create landscape data
        self.X, self.Y, self.Z = create_function_landscape(function, bounds)
        
        # Create contour plot
        self.contour = self.ax.contourf(self.X, self.Y, self.Z, 20, cmap=cm.viridis, alpha=0.6)
        self.fig.colorbar(self.contour, ax=self.ax)
        
        # Initialize particles and best point plots
        self.particles_plot, = self.ax.plot([], [], 'bo', markersize=6, label='Particles')
        self.best_plot, = self.ax.plot([], [], 'r*', markersize=10, label='Global Best')
        
        # Path of the global best
        self.best_path_x = []
        self.best_path_y = []
        self.best_path_plot, = self.ax.plot([], [], 'r--', linewidth=1, label='Best Path')
        
        # Configure plot
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[0], bounds[1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(title)
        self.ax.legend()
        
        # Current iteration text
        self.iteration_text = self.ax.text(
            0.02, 0.98, 'Iteration: 0', transform=self.ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Best fitness text
        self.fitness_text = self.ax.text(
            0.02, 0.94, 'Best Fitness: N/A', transform=self.ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Animation function will be set in start_animation
        self.animation = None
        self.positions_history = []
        self.global_best_history = []
    
    def update_frame(self, frame):
        """
        Update the animation frame
        
        Parameters:
        -----------
        frame: int
            Current frame number
        """
        if frame < len(self.positions_history):
            positions = self.positions_history[frame]
            global_best = self.global_best_history[frame]
            
            # Update particles
            self.particles_plot.set_data(positions[:, 0], positions[:, 1])
            
            # Update global best
            self.best_plot.set_data([global_best[0]], [global_best[1]])
            
            # Update best path
            if frame > 0 and not np.array_equal(global_best, self.global_best_history[frame-1]):
                self.best_path_x.append(global_best[0])
                self.best_path_y.append(global_best[1])
            
            self.best_path_plot.set_data(self.best_path_x, self.best_path_y)
            
            # Update texts
            self.iteration_text.set_text(f'Iteration: {frame}')
            fitness = self.function(global_best)
            self.fitness_text.set_text(f'Best Fitness: {fitness:.6f}')
        
        return self.particles_plot, self.best_plot, self.best_path_plot, self.iteration_text, self.fitness_text
    
    def callback(self, positions, global_best, iteration):
        """
        Callback function to collect data during optimization
        
        Parameters:
        -----------
        positions: numpy array
            Current particle positions
        global_best: numpy array
            Current global best position
        iteration: int
            Current iteration number
        """
        self.positions_history.append(positions)
        self.global_best_history.append(global_best)
    
    def start_animation(self):
        """Start the animation after data collection"""
        if not self.positions_history:
            print("No optimization data collected. Run the optimization first.")
            return
        
        self.animation = FuncAnimation(
            self.fig, self.update_frame, frames=len(self.positions_history),
            interval=self.interval, blit=True, repeat=False
        )
        
        plt.tight_layout()
        plt.show()