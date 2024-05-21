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

colorizer = ga_colorize.GAColorize('example_graph.txt', 'example_ga_params.txt')

results = colorizer.run_genetic_algorithm()
# print(results)
best_fitness = 0
for individual, fitness in results:
    if fitness > best_fitness:
        solution = individual
        best_fitness = fitness
print(f"Individual: {colorizer.decode_individual(solution)}, fitness: {best_fitness}")
print(f"Number of colors: {len(set(col for _, col in colorizer.decode_individual(solution)))}")
