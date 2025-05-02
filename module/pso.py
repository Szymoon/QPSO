import numpy as np

class StandardPSO:
    def __init__(self, objective_function, num_particles, max_iter, bounds, w=0.5, c1=1.5, c2=1.5, dimensions=2):
        """
        Initialize the PSO algorithm
        
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
        w: float
            Inertia weight
        c1: float
            Cognitive coefficient (personal best attraction)
        c2: float
            Social coefficient (global best attraction)
        dimensions: int
            Number of dimensions for the problem
        """
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Dimension of the search space
        self.dim = dimensions
        
        # Initialize particle positions and velocities
        # Positions are initialized uniformly within the bounds
        self.positions = np.random.uniform(
            bounds[0], bounds[1], (num_particles, self.dim)
        )
        
        # Velocities are initialized within a range proportional to the bounds
        velocity_range = np.abs(bounds[1] - bounds[0])
        self.velocities = np.random.uniform(
            -velocity_range, velocity_range, (num_particles, self.dim)
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
            
    def update_velocities(self):
        """Update particle velocities using PSO velocity equation"""
        # Random factors for cognitive and social components
        r1 = np.random.random((self.num_particles, self.dim))
        r2 = np.random.random((self.num_particles, self.dim))
        
        # PSO velocity equation
        cognitive_component = self.c1 * r1 * (self.personal_best_pos - self.positions)
        social_component = self.c2 * r2 * (self.global_best_pos - self.positions)
        
        self.velocities = self.w * self.velocities + cognitive_component + social_component
        
    def update_positions(self):
        """Update particle positions based on velocities"""
        self.positions = self.positions + self.velocities
        
        # Ensure particles stay within bounds
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])
    
    def step(self):
        """Perform one iteration of the optimization algorithm"""
        # Update velocities and positions
        self.update_velocities()
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
        Run the PSO optimization process
        
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
        
        # Main PSO loop
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