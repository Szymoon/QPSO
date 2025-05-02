import numpy as np

class QuantumPSO:
    def __init__(self, objective_function, num_particles, max_iter, bounds, beta_min=0.2, beta_max=1.0, dimensions=2):
        """
        Initialize the Quantum PSO algorithm
        
        Parameters:
        -----------
        objective_function: function
            The function to be minimized
        num_particles: int
            Number of particles in the swarm
        max_iter: int
            Maximum number of iterations
        bounds: tuple
            (lower_bound, upper_bound) for search space
        beta_min: float
            Minimum value of the contraction-expansion coefficient
        beta_max: float
            Maximum value of the contraction-expansion coefficient
        dimensions: int
            Number of dimensions for the problem
        """
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Dimension of the search space
        self.dim = dimensions
        
        # Initialize particle positions (no velocities in QPSO)
        self.positions = np.random.uniform(
            bounds[0], bounds[1], (num_particles, self.dim)
        )
        
        # Initialize fitness, personal best and global best
        self.fitness = np.zeros(num_particles)
        self.personal_best_pos = self.positions.copy()
        self.personal_best_fitness = np.zeros(num_particles) + float('inf')
        
        self.global_best_pos = np.zeros(self.dim)
        self.global_best_fitness = float('inf')
        
        # Tracking variables for optimization progress
        self.current_iteration = 0
        self.convergence_history = []
        
    def evaluate_fitness(self):
        """Evaluate the fitness of all particles"""
        self.fitness = np.array([self.objective_function(p) for p in self.positions])
        
    def update_personal_best(self):
        """Update personal best positions and fitness values"""
        improved_indices = self.fitness < self.personal_best_fitness
        self.personal_best_pos[improved_indices] = self.positions[improved_indices].copy()
        self.personal_best_fitness[improved_indices] = self.fitness[improved_indices]
        
    def update_global_best(self):
        """Update global best position and fitness value"""
        best_particle_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_particle_idx] < self.global_best_fitness:
            self.global_best_pos = self.personal_best_pos[best_particle_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_particle_idx]
    
    def calculate_mean_best_position(self):
        """Calculate the mean best position (mbest) for QPSO"""
        return np.mean(self.personal_best_pos, axis=0)
    
    def update_positions(self):
        """
        Update particle positions using quantum behavior
        This is the key difference between PSO and QPSO
        """
        # Calculate contraction-expansion coefficient (linearly decreasing)
        beta = self.beta_max - (self.beta_max - self.beta_min) * self.current_iteration / self.max_iter
        
        # Calculate mean best position (mbest)
        mbest = self.calculate_mean_best_position()
        
        # Random factors for the quantum model
        u = np.random.random((self.num_particles, self.dim))
        p = np.random.random((self.num_particles, self.dim))
        
        # Compute local attractor points
        p = p * self.personal_best_pos + (1 - p) * self.global_best_pos
        
        # Calculate quantum delta potential well
        delta = np.abs(mbest - self.positions)
        
        # Update positions using quantum state
        # This equation represents the wave function collapse in quantum mechanics
        self.positions = p + ((-1) ** np.round(np.random.random((self.num_particles, self.dim)))) * \
                        beta * delta * np.log(1 / u)
        
        # Ensure particles stay within bounds
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])
        
    def step(self):
        """Perform one iteration of the optimization algorithm"""
        # Update positions using quantum behavior
        self.update_positions()
        
        # Evaluate new positions
        self.evaluate_fitness()
        
        # Update personal and global bests
        self.update_personal_best()
        self.update_global_best()
        
        # Update convergence history
        self.convergence_history.append(self.global_best_fitness)
        
        # Increment iteration counter
        self.current_iteration += 1
        
        return self.positions.copy(), self.global_best_pos.copy(), self.global_best_fitness
        
    def optimize(self, callback=None):
        """
        Run the QPSO optimization process
        
        Parameters:
        -----------
        callback: function
            Optional callback function that takes (positions, iteration) as arguments
            Used for visualization during optimization
            
        Returns:
        --------
        dict
            Dictionary containing optimization results
        """
        # Initial evaluation
        self.evaluate_fitness()
        self.update_personal_best()
        self.update_global_best()
        
        # Initial callback if provided
        if callback:
            callback(self.positions, self.global_best_pos, 0)
        
        # Main QPSO loop
        for i in range(self.max_iter):
            # Perform one iteration
            positions, global_best, fitness = self.step()
            
            # Call the callback function if provided
            if callback:
                callback(positions, global_best, i+1)
                
        return {
            'global_best_position': self.global_best_pos,
            'global_best_fitness': self.global_best_fitness,
            'convergence_history': self.convergence_history,
            'iterations': self.max_iter
        }