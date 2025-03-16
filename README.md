# Graph Coloring Solver Using Genetic Algorithms

## Overview
This project implements genetic algorithms (GA) to solve the Graph Coloring Problem (GCP), which involves assigning colors to graph nodes such that no adjacent nodes share the same color while minimizing the total number of colors used. We explore standard GA implementations alongside adaptive parameter strategies to optimize solution efficiency.

## Key Features
- Standard genetic algorithm implementation with configurable parameters
- Two adaptive parameter strategies:
  - DHM-ILC (Decreasing Hybridization-Increasing Local Changes)
  - ILM-DHC (Increasing Local Mutations-Decreasing Hybridization Changes)
- Comprehensive testing framework for parameter sensitivity analysis
- Visualization tools for solution convergence and genetic diversity

## Repository Structure
- **dataset/**: Graph datasets and GA parameter configurations
  - `graphs/dataset_small/`: Sample graph collections (Bull, Butterfly, Cycle, etc.)
  - `ga_params/`: Parameter sets for experiments
- **gcp_implementation/**: Core implementation
  - `graph.py`: Graph data structure implementation
  - `ga_colorize.py`: Standard GA implementation
  - `ga_colorize_original_adaptative.py`: Adaptive GA implementation
  - `main.py`: Entry point for running a single graph coloring experiment
- **experiment_run_script.py**: Script for batch execution of fixed parameter experiments
- **experiment_run_script_dynamic.py**: Script for adaptive parameter experiments

## Quick Start

### Running a Single Graph Coloring Experiment
```python
# In gcp_implementation/main.py
# Choose one of:

# 1. Standard GA with fixed parameters
colorizer = GAColorizeFixed('../dataset/graphs/dataset_small/BullGraph.col.txt',
                          '../dataset/ga_params/experiment_baseline/baseline.txt')

# 2. Adaptive GA with DHM-ILC strategy
# colorizer = GAColorizeAdaptive('../dataset/graphs/dataset_small/BullGraph.col.txt',
#                              '../dataset/ga_params/experiment_adaptative_DHM-ILC/DHM-ILC.txt')

# 3. Adaptive GA with ILM-DHC strategy
# colorizer = GAColorizeAdaptive('../dataset/graphs/dataset_small/BullGraph.col.txt',
#                              '../dataset/ga_params/experiment_adaptative_ILM-DHC/ILM-DHC.txt')

# Run the algorithm
results = colorizer.run_genetic_algorithm()
```

### Running Batch Experiments
```bash
# For fixed parameter experiments
python experiment_run_script.py

# For adaptive parameter experiments
python experiment_run_script_dynamic.py
```

## Parameter Configuration
Key parameters include:
- `num_runs`: Number of independent GA runs
- `num_generations`: Maximum generations per run
- `num_individuals`: Population size
- `mutation_prob`: Probability of gene mutation
- `crossover_prob`: Probability of crossover
- `max_no_improvement_generations`: Early stopping threshold

## Adaptive Strategies
- **DHM-ILC**: Crossover probability increases while mutation probability decreases with generations, prioritizing exploration early and exploitation later
- **ILM-DHC**: Mutation probability increases while crossover probability decreases with generations, gradually shifting from global to local search

## Contributors (Group 21)
- Andreas Pattichis
- Antonio Carpes
- Steven ten Teije
