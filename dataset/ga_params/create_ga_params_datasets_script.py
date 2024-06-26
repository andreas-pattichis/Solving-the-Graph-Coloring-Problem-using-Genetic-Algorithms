import os

# Baseline parameters, considered standard for a variety of GA problems
baseline_params = {
    "num_runs": 10,  # Standard number of runs for reliability
    "num_generations": 100,  # Typical number of generations to allow convergence
    "num_individuals": 100,  # Common population size balancing diversity and computation
    "mutation_prob": 0.05,  # Typical mutation rate providing sufficient diversity
    "crossover_prob": 0.8,  # Common crossover rate encouraging recombination
    "max_no_improvement_generations": 30  # Standard patience for improvement
}

# Parameter variations, exploring the effect of different ranges
parameter_variations = {
    "num_runs": [5, 20],  # Varying number of runs to see effect on reliability
    "num_generations": [25, 50, 150, 200],  # Varying generations to check convergence speed
    "num_individuals": [25, 50, 150, 200],  # Varying population size for diversity vs. computation
    "mutation_prob": [0.001, 0.01, 0.03, 0.1],  # Varying mutation rate for exploration
    "crossover_prob": [0.1, 0.6, 0.9, 0.95],  # Varying crossover rate for recombination effects
    "max_no_improvement_generations": [10, 20, 40, 50]  # Varying patience for improvement
}


# Create experiment folders and files
def create_experiment_files(base_params, param_variations):
    for param, values in param_variations.items():
        folder_name = f"experiment_{param}"
        os.makedirs(folder_name, exist_ok=True)
        for value in values:
            experiment_params = base_params.copy()
            experiment_params[param] = value
            file_name = f"{param}_{value}.txt"
            file_path = os.path.join(folder_name, file_name)
            with open(file_path, 'w') as file:
                for key, val in experiment_params.items():
                    file.write(f"{key} {val}\n")


create_experiment_files(baseline_params, parameter_variations)

print("Experiment files created successfully.")
