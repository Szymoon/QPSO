# Quantum-Inspired Particle Swarm Optimization (QPSO)

![Algorithm: PSO](https://img.shields.io/badge/Algorithm-PSO-blue) ![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-green) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

A Python implementation of both Standard Particle Swarm Optimization (PSO) and Quantum-inspired Particle Swarm Optimization (QPSO), with visualization tools and performance comparison capabilities. Demonstrates how QPSO can outperform PSO on complex benchmark functions in 3D and higher-dimensional spaces.

## Table of Contents

- [Introduction](#introduction)
- [Algorithm Comparison](#algorithm-comparison)
- [Benchmark Functions](#benchmark-functions)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Visualization](#results-and-visualization)
- [When to Use QPSO](#when-to-use-qpso)
- [Contributing](#contributing)
- [License](#license)

## Introduction

### Particle Swarm Optimization (PSO)

Particle Swarm Optimization is a population-based stochastic optimization technique inspired by the social behavior of birds flocking or fish schooling. Each particle moves through the search space guided by:

- **Personal best**: the best solution the particle has found.
- **Global best**: the best solution found by the entire swarm.

At each iteration, velocity and position update rules drive exploration and exploitation.

### Quantum-inspired PSO (QPSO)

QPSO enhances standard PSO by applying quantum mechanics principles:

- No explicit velocity; positions follow a quantum probability distribution.
- Uses the mean best position (`mbest`) of the swarm.
- Introduces a contraction–expansion coefficient (β) that controls exploration.

This yields stronger global search capabilities and fewer parameters.

## Algorithm Comparison

| Feature                          | Standard PSO        | Quantum PSO       |
|----------------------------------|---------------------|-------------------|
| Particle Behavior                | Classical mechanics | Quantum states    |
| Position Update                  | Deterministic       | Probabilistic     |
| Velocity                         | Required            | Not required      |
| Exploration                      | Good                | Excellent         |
| Exploitation                     | Excellent           | Good              |
| High-Dimensional Performance     | Degrades faster     | More robust       |
| Number of Parameters             | 3 (w, c1, c2)       | 1 (β)             |

## Benchmark Functions

The repository includes common test functions:

1. **Ackley**
2. **Rastrigin**
3. **Griewank**
4. **Schwefel**
5. **Levy**
6. **f10_composite**  (complex multimodal composite)

## Repository Structure

```
QPSO/
├── module/
│   ├── __init__.py
│   ├── pso.py          # Standard PSO implementation
│   ├── qpso.py         # Quantum PSO implementation
│   └── visualization.py# Visualization and benchmarks
├── run_pso.py          # Run PSO with animation
├── run_qpso.py         # Run QPSO with animation
└── run_comparison.py   # Compare PSO vs QPSO
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/QPSO.git
   cd QPSO
   ```

2. Install dependencies:

   ```bash
   pip install numpy matplotlib
   ```

## Usage

### Running Standard PSO

Edit `run_pso.py` to configure parameters, for example:

```python
function_name = 'ackley'   # choose among ackley, rastrigin, ...
num_particles = 30         # swarm size
max_iter = 100             # iterations
speed_factor = 1.5         # animation speed
```

Then:

```bash
python run_pso.py
```

### Running Quantum PSO

Edit `run_qpso.py` similarly and run:

```bash
python run_qpso.py
```

### Comparing PSO and QPSO

In `run_comparison.py`, set:

```python
function_name = 'f10_composite'
dimensions = 30
num_particles = 50
max_iter = 300
```

Then:

```bash
python run_comparison.py
```

## Results and Visualization

- **3D Animation** of particles on the benchmark surface.
- **Convergence Plot**: best‐so‐far fitness vs. iteration.

## When to Use QPSO

- Highly multimodal landscapes (e.g., Rastrigin)
- High-dimensional problems (≥10D)
- Problems requiring strong global exploration

For simple or low-dimensional problems, Standard PSO may be sufficient.

## Contributing

Feel free to:

- Add new benchmark functions.
- Improve visualizations.
- Optimize algorithms or add variants.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
