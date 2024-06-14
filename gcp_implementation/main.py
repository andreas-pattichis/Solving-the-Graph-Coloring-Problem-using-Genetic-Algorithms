from graph import Graph
from gcp_implementation.ga_colorize import GAColorize as GAColorizeFixed
from gcp_implementation.ga_colorize_original_adaptative import GAColorize as GAColorizeAdaptive

if __name__ == '__main__':
    # test the graph implementation
    path = '../dataset/graphs/dataset_small/ExampleGraph.txt'
    graph = Graph(path)
    print(graph)

    for node in graph.nodes.values():
        print(node)

    for edge in graph.edges:
        print(edge)

# Uncomment the following lines to run the genetic algorithm with the fixed parameters
colorizer = GAColorizeFixed('../dataset/graphs/dataset_small/BullGraph.col.txt',
                                   '../dataset/ga_params/experiment_baseline/baseline.txt')

# Uncomment the following lines to run the genetic algorithm with the adaptive parameters (DHM-ILC)
# colorizer = GAColorizeAdaptive('../dataset/graphs/dataset_small/BullGraph.col.txt',
#                                     '../dataset/ga_params/experiment_adaptative_DHM-ILC/DHM-ILC.txt')

# # Uncomment the following lines to run the genetic algorithm with the adaptive parameters (ILM-DHC)
# colorizer = GAColorizeAdaptive('../dataset/graphs/dataset_small/BullGraph.col.txt',
#                                     '../dataset/ga_params/experiment_adaptative_ILM-DHC/ILM-DHC.txt')

results = colorizer.run_genetic_algorithm()
best_fitness = 0
for individual, fitness in results:
    if fitness > best_fitness:
        solution = individual
        best_fitness = fitness

decoded_solution = colorizer.decode_individual(solution)
num_colors = len(set(col for _, col in decoded_solution))

print("\nBest solution found by the genetic algorithm:")
print(f"Individual (node, color pairs): {decoded_solution}")
print(f"Fitness value: {best_fitness}")
print(f"Number of colors used: {num_colors}")
