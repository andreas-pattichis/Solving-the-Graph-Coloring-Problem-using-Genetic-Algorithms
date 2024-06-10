import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gcp_implementation.graph import Graph
import gcp_implementation.ga_colorize as ga_colorize


def run_experiment(graph_path, ga_params_path):
    colorizer = ga_colorize.GAColorize(graph_path, ga_params_path)
    ga_params = colorizer.ga_params

    generation_times = []
    total_times = []
    fitness_values = []
    best_fitness_per_generation = []
    run_results = []
    stopping_generations = []
    first_4_color_generation = []  # Track the first generation where 4 colors are found
    first_min_color_generation = []  # Track the first generation where the minimum number of colors is found
    colors_per_generation = []  # Track the number of colors used in each generation

    for run_index in range(ga_params['num_runs']):
        start_time = time.time()
        individuals = colorizer.initialize_individuals()
        solutions = [colorizer.find_best_individual(individuals)]
        generation_times_run = []
        best_fitness_gen_run = []
        colors_gen_run = []

        found_4_colors = False
        min_colors = float('inf')
        min_color_generation = -1

        for generation in range(ga_params['num_generations']):
            gen_start_time = time.time()

            individuals = colorizer.select_individuals(individuals)
            individuals = colorizer.apply_crossover(individuals)
            individuals = colorizer.apply_mutation(individuals)

            solutions.append(colorizer.find_best_individual(individuals))

            gen_end_time = time.time()
            generation_times_run.append(gen_end_time - gen_start_time)

            valid_solutions = [solution for solution in solutions if solution[0] is not None]
            if valid_solutions:
                _, best_fitness = max(valid_solutions, key=lambda x: x[1])
                best_fitness_gen_run.append(best_fitness)

                # Track the number of colors
                best_solution = max(valid_solutions, key=lambda x: x[1])[0]
                decoded_solution = colorizer.decode_individual(best_solution)
                num_colors = len(set(color for _, color in decoded_solution))
                colors_gen_run.append(num_colors)

                # Check for 4-color solution
                if num_colors <= 4 and not found_4_colors:
                    first_4_color_generation.append(generation)
                    found_4_colors = True

                # Update the minimum colors
                if num_colors < min_colors:
                    min_colors = num_colors
                    min_color_generation = generation

            else:
                best_fitness_gen_run.append(0)
                colors_gen_run.append(0)

            if colorizer.check_end_conditions(generation, solutions):
                stopping_generations.append(generation)
                break
        else:
            stopping_generations.append(ga_params['num_generations'])

        if not found_4_colors:
            first_4_color_generation.append(-1)
        if min_color_generation != -1:
            first_min_color_generation.append(min_color_generation)
        else:
            first_min_color_generation.append(-1)

        generation_times.append(generation_times_run)
        total_times.append(time.time() - start_time)
        fitness_values.append(best_fitness_gen_run[-1])
        best_fitness_per_generation.append(best_fitness_gen_run)
        colors_per_generation.append(colors_gen_run)

        valid_solutions = [solution for solution in solutions if solution[0] is not None]
        if valid_solutions:
            best_individual, best_fitness = max(valid_solutions, key=lambda x: x[1])
            decoded_solution = colorizer.decode_individual(best_individual)
            num_colors = len(set(col for _, col in decoded_solution))
            run_results.append((decoded_solution, best_fitness, num_colors))
        else:
            run_results.append(([], 0, 0))

    return generation_times, total_times, fitness_values, best_fitness_per_generation, run_results, stopping_generations, first_4_color_generation, first_min_color_generation, colors_per_generation

def plot_results(generation_times, total_times, fitness_values, best_fitness_per_generation, output_dir,
                 experiment_name, graph_name):
    num_runs = len(generation_times)

    # Determine the maximum number of generations
    max_generations = max(len(gen_times) for gen_times in generation_times)

    # Initialize avg_generation_times with zeros
    avg_generation_times = [0] * max_generations
    generation_counts = [0] * max_generations

    for gen_times in generation_times:
        for i, gen_time in enumerate(gen_times):
            avg_generation_times[i] += gen_time
            generation_counts[i] += 1

    # Compute the average by dividing by the number of runs for each generation
    avg_generation_times = [avg_generation_times[i] / generation_counts[i] for i in range(max_generations)]

    avg_total_time = sum(total_times) / num_runs
    avg_fitness = sum(fitness_values) / num_runs

    # Plot Average Generation Time per Generation
    plt.figure()
    plt.plot(range(max_generations), avg_generation_times, label='Avg Generation Time', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Time (s)')
    plt.title(f'Average Generation Time per Generation\nExperiment: {experiment_name}, Graph: {graph_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'avg_generation_time.png'))
    plt.close()

    # Plot Fitness Values per Run
    plt.figure()
    plt.plot(range(num_runs), fitness_values, label='Fitness Value', linewidth=2)
    plt.xlabel('Run')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Values per Run\nExperiment: {experiment_name}, Graph: {graph_name}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fitness_values_per_run.png'), bbox_inches='tight')
    plt.close()

    # Plot Best Fitness Value per Generation Across Runs
    plt.figure()
    for i, run in enumerate(best_fitness_per_generation):
        plt.plot(range(len(run)), run, alpha=0.7, linewidth=2, label=f'Run {i + 1}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Best Fitness Value per Generation Across Runs\nExperiment: {experiment_name}, Graph: {graph_name}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'best_fitness_per_generation.png'), bbox_inches='tight')
    plt.close()


def save_results(filename, run_results, avg_total_time, avg_fitness, experiment_name, graph_name):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Experiment: {experiment_name}\n")
        file.write(f"Graph: {graph_name}\n\n")
        for i, (solution, fitness, num_colors) in enumerate(run_results):
            file.write(f"Run {i + 1}:\n")
            file.write(f"Individual (node, color pairs): {solution}\n")
            file.write(f"Fitness value: {fitness}\n")
            file.write(f"Number of colors used: {num_colors}\n\n")

        file.write(f"Average total time for {len(run_results)} runs: {avg_total_time:.2f} seconds\n")
        file.write(f"Average fitness value over {len(run_results)} runs: {avg_fitness:.2f}\n")


def save_results_csv(filename, generation_times, total_times, fitness_values, avg_total_time, avg_fitness,
                     experiment_name, graph_name, run_results, stopping_generations, best_fitness_per_generation,
                     first_4_color_generation, first_min_color_generation, colors_per_generation):
    num_runs = len(generation_times)
    num_generations = max(len(gen_times) for gen_times in generation_times)

    avg_generation_times = [sum(gen_times) / num_runs for gen_times in zip(*generation_times)]
    best_fitness_per_run = [run[1] for run in run_results]
    avg_num_colors_per_run = [run[2] for run in run_results]

    data = {
        'Experiment': [experiment_name] * num_runs,
        'Graph': [graph_name] * num_runs,
        'Run': list(range(1, num_runs + 1)),
        'Avg Generation Time': [sum(times) / len(times) for times in generation_times],
        'Total Run Time': total_times,
        'Final Fitness Value': fitness_values,
        'Best Fitness Value': best_fitness_per_run,
        'Avg Number of Colors Used': avg_num_colors_per_run,
        'Average Total Time': [avg_total_time] * num_runs,
        'Average Fitness': [avg_fitness] * num_runs,
        'Stopping Generation': stopping_generations,
        'First 4 Color Generation': first_4_color_generation,
        'First Min Color Generation': first_min_color_generation
    }

    # Ensure all lists in `data` are the same length
    min_length = min(len(lst) for lst in data.values())
    for key in data:
        data[key] = data[key][:min_length]

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')

    # Save detailed generation data
    gen_data = {
        'Run': [],
        'Generation': [],
        'Generation Time': [],
        'Best Fitness': [],
        'Num Colors': []
    }
    for run_idx, (gen_times_run, colors_run) in enumerate(zip(generation_times, colors_per_generation)):
        for gen_idx, (gen_time, num_colors) in enumerate(zip(gen_times_run, colors_run)):
            gen_data['Run'].append(run_idx + 1)
            gen_data['Generation'].append(gen_idx + 1)
            gen_data['Generation Time'].append(gen_time)
            gen_data['Best Fitness'].append(best_fitness_per_generation[run_idx][gen_idx])
            gen_data['Num Colors'].append(num_colors)

    gen_df = pd.DataFrame(gen_data)
    gen_filename = filename.replace('.csv', '_generation_data.csv')
    gen_df.to_csv(gen_filename, index=False, encoding='utf-8')



def aggregate_experiment_results(experiment_results_folder):
    aggregate_data = []

    graph_folders = [f.path for f in os.scandir(experiment_results_folder) if f.is_dir()]

    for folder in graph_folders:
        for file in os.listdir(folder):
            if file.endswith("experiment_results.csv"):
                df = pd.read_csv(os.path.join(folder, file))
                if df.empty:
                    # Create a DataFrame with NaN values if the original is empty
                    columns = ['Experiment', 'Graph', 'Run', 'Avg Generation Time', 'Total Run Time', 'Final Fitness Value',
                               'Best Fitness Value', 'Avg Number of Colors Used', 'Average Total Time', 'Average Fitness',
                               'Stopping Generation', 'First 4 Color Generation', 'First Min Color Generation']
                    df = pd.DataFrame(np.nan, index=[0], columns=columns)
                    df['Graph'] = os.path.basename(folder)
                aggregate_data.append(df)

    if aggregate_data:
        all_results = pd.concat(aggregate_data, ignore_index=True)

        summary = all_results.groupby(['Experiment', 'Graph']).agg(
            Avg_Generation_Time=('Avg Generation Time', 'mean'),
            Total_Run_Time=('Total Run Time', 'mean'),
            Final_Fitness_Value=('Final Fitness Value', 'mean'),
            Best_Fitness_Value=('Best Fitness Value', 'mean'),
            Avg_Number_of_Colors_Used=('Avg Number of Colors Used', 'mean'),
            Avg_Total_Time=('Average Total Time', 'mean'),
            Avg_Fitness=('Average Fitness', 'mean'),
            Fitness_Std=('Final Fitness Value', 'std'),
            Avg_Stopping_Generation=('Stopping Generation', 'mean'),
            Avg_First_4_Color_Generation=('First 4 Color Generation', 'mean'),
            Avg_First_Min_Color_Generation=('First Min Color Generation', 'mean')
        ).reset_index()

        overall_avg = summary.mean(numeric_only=True)
        overall_avg['Experiment'] = 'Overall'
        overall_avg['Graph'] = 'Overall'
        summary = pd.concat([summary, pd.DataFrame([overall_avg])], ignore_index=True)

        # Remove averages for the specified columns in the overall row
        summary.loc[summary['Experiment'] == 'Overall', 'Avg_First_4_Color_Generation'] = None
        summary.loc[summary['Experiment'] == 'Overall', 'Avg_First_Min_Color_Generation'] = None

        summary_file = os.path.join(experiment_results_folder, "aggregate_summary.csv")
        summary.to_csv(summary_file, index=False, encoding='utf-8')



def run_all_experiments(ga_params_folder, graphs_folder, results_folder):
    ga_params_name = os.path.basename(ga_params_folder)
    results_folder = os.path.join(results_folder, ga_params_name)
    os.makedirs(results_folder, exist_ok=True)

    for ga_params_file in os.listdir(ga_params_folder):
        ga_params_path = os.path.join(ga_params_folder, ga_params_file)
        experiment_name = os.path.splitext(ga_params_file)[0]
        experiment_results_folder = os.path.join(results_folder, experiment_name)
        os.makedirs(experiment_results_folder, exist_ok=True)

        for graph_file in os.listdir(graphs_folder):
            graph_path = os.path.join(graphs_folder, graph_file)
            graph_name = os.path.splitext(graph_file)[0]
            output_dir = os.path.join(experiment_results_folder, graph_name)
            os.makedirs(output_dir, exist_ok=True)

            generation_times, total_times, fitness_values, best_fitness_per_generation, run_results, stopping_generations, first_4_color_generation, first_min_color_generation, colors_per_generation = run_experiment(
                graph_path, ga_params_path)
            plot_results(generation_times, total_times, fitness_values, best_fitness_per_generation, output_dir,
                         experiment_name, graph_name)

            avg_total_time = sum(total_times) / len(total_times)
            avg_fitness = sum(fitness_values) / len(fitness_values)
            results_filename = os.path.join(output_dir, 'experiment_results.txt')
            save_results(results_filename, run_results, avg_total_time, avg_fitness, experiment_name, graph_name)
            csv_filename = os.path.join(output_dir, 'experiment_results.csv')
            save_results_csv(csv_filename, generation_times, total_times, fitness_values, avg_total_time, avg_fitness,
                             experiment_name, graph_name, run_results, stopping_generations,
                             best_fitness_per_generation, first_4_color_generation, first_min_color_generation, colors_per_generation)

        aggregate_experiment_results(experiment_results_folder)


def main():
    ga_params_folder = 'dataset/ga_params/experiment_baseline'  # Change as needed
    graphs_folder = 'dataset/graphs/dataset_small'  # Change as needed
    results_folder = 'exps/results'  # Change as needed

    run_all_experiments(ga_params_folder, graphs_folder, results_folder)

    # Print that the experiment has finished
    print("All experiments finished.")


if __name__ == '__main__':
    main()
