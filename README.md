# Solving the Graph Coloring Problem using Genetic Algorithms

## Overview
This project implements a Genetic Algorithm (GA) to solve the Graph Coloring Problem (GCP). The GCP involves assigning colors to the nodes of a graph such that no two adjacent nodes share the same color, with the goal of minimizing the number of colors used. Our implementation focuses on planar graphs and explores the impact of different GA parameters and adaptive strategies on the efficiency and effectiveness of the solution.

## Repository Contents
- `dataset/`: Contains the graph datasets and GA parameters for different experiments.
  - `graphs/`: Datasets for graphs used in the experiments.
    - `dataset_small/`: Example graphs such as `BullGraph.col.txt`, `ButterflyGraph.col.txt`, `Cycle3.col.txt`, etc.
  - `ga_params/`: Different sets of GA parameters for creating various experiments.
    - `experiment_baseline/`
    - `experiment_crossover_prob/`
    - `experiment_max_no_improvement_generations/`
    - `experiment_mutation_prob/`
    - `experiment_num_generations/`
    - `experiment_num_individuals/`
    - `experiment_num_runs/`
    - `experiment_adaptative_DHM-ILC`
    - `experiment_adaptative_ILM-DHC`
    - `create_ga_params_datasets_script.py`: Script to quickly create and adjust GA parameter sets for experiments.
- `gcp_implementation/`: Implementation of the GA for solving the GCP.
  - `ga_colorize.py`: Implementation of the GA with fixed parameters.
  - `ga_colorize_original_adaptative.py`: Implementation of the GA with the dynamic approaches we used.
  - `graph.py`: Contains classes for nodes, edges, and graph representation.
  - `main.py`: Main function to run the GA for a single graph with user-specified GA parameters and dataset input.
- `results/`: Contains results from experiments.
  - `experiments_complete/`: Complete results for each graph uploaded in zip format.
  - `report_plots/`: Plots generated for the report.
- `experiment_run_script.py`: Script for running fixed GA experiments with multiple runs.
- `experiment_run_script_dynamic.py`: Script for running dynamic GA experiments with multiple runs.
- `README.md`: This readme file.

## Running a Single Graph with main.py
To run the GA for a single graph from the dataset with a specific set of GA parameters, follow these steps:

1. **Choose the graph and GA parameters**: Select the graph and the parameter set you want to use for the run. For this example, we will use the Bull Graph and the baseline GA parameters.

    **Bull Graph Dataset (`BullGraph.col.txt`):**
    ```
    c FILE: BullGraph.col
    n 1 2 3 4 5
    e 1 2 1
    e 2 3 1
    e 3 1 1
    e 1 4 1
    e 2 5 1
    ```

    **Baseline GA Parameters (baseline.txt):**
    ```
    num_runs 10
    num_generations 100
    num_individuals 100
    mutation_prob 0.05
    crossover_prob 0.8
    max_no_improvement_generations 30
    ```

2. **Update main.py**: Go to `main.py` in the `gcp_implementation` folder. Add the paths for both the dataset and parameters files you decided to use. Update the colorizer initialization line as shown below:

    ```python
    colorizer = GAColorize('../dataset/graphs/dataset_small/BullGraph.col.txt',
                           '../dataset/ga_params/experiment_baseline/baseline.txt')
    ```

    Additionally, you can choose from the following three options by uncommenting the corresponding lines:

    ```python
    # Uncomment the following lines to run the genetic algorithm with the fixed parameters
    colorizer = GAColorizeFixed('../dataset/graphs/dataset_small/BullGraph.col.txt',
                                       '../dataset/ga_params/experiment_baseline/baseline.txt')
    
    # Uncomment the following lines to run the genetic algorithm with the adaptive parameters (DHM-ILC)
    # colorizer = GAColorizeAdaptive('../dataset/graphs/dataset_small/BullGraph.col.txt',
    #                                     '../dataset/ga_params/experiment_adaptative_DHM-ILC/DHM-ILC.txt')
    
    # Uncomment the following lines to run the genetic algorithm with the adaptive parameters (ILM-DHC)
    # colorizer = GAColorizeAdaptive('../dataset/graphs/dataset_small/BullGraph.col.txt',
    #                                     '../dataset/ga_params/experiment_adaptative_ILM-DHC/ILM-DHC.txt')
    ```

3. **Run main.py**: Execute the `main.py` script to start the GA for the selected graph and parameters:
    ```bash
    python gcp_implementation/main.py
    ```

4. **View the results**: The output will display the results of the GA run, including the best solution found:
    ```
    Graph with 8 nodes and 6 edges.
    Node(1)
    Node(2)
    Node(3)
    Node(4)
    Node(5)
    Node(6)
    Node(7)
    Node(8)
    [1, 2]
    [1, 3]
    [1, 4]
    [4, 5]
    [5, 6]
    [5, 7]
    Starting run number 0
    Population initialized...
    Starting run number 1
    Population initialized...
    Starting run number 2
    Population initialized...
    Starting run number 3
    Population initialized...
    Starting run number 4
    Population initialized...
    Starting run number 5
    Population initialized...
    Starting run number 6
    Population initialized...
    Starting run number 7
    Population initialized...
    Starting run number 8
    Population initialized...
    Starting run number 9
    Population initialized...

    Best solution found by the genetic algorithm:
    Individual (node, color pairs): [(1, 0), (2, 1), (3, 2), (4, 2), (5, 0)]
    Fitness value: 166.66666666666666
    Number of colors used: 3
    ```

The output shows the details of the graph, the initialization of the population for each run, and the best solution found. In this case, the algorithm found a solution using 3 colors for the Bull Graph.

## Running Multiple Experiments
To run the fixed GA experiments with multiple runs, use:
```bash
python experiment_run_script.py
```
## Running Dynamic GA Experiments
To run the dynamic GA experiments with multiple runs, use:
```bash
python experiment_run_script_dynamic.py
```


## Contibutions (Group 21)
- Andreas Pattichis
- Antonio Carpes
- Steven ten Teije
