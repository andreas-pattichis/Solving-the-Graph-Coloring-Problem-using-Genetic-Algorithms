from graph import Graph
import ga_colorize

if __name__ == '__main__':
    # test the graph implementation
    path = 'example_graph.txt'
    graph = Graph(path)
    print(graph)

    for node in graph.nodes.values():
        print(node)

    for edge in graph.edges:
        print(edge)

colorizer = ga_colorize.GAColorize('example_graph.txt', '../dataset/ga_params/baseline_ga_params.txt')

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
