import math
import random

from gcp_implementation.graph import Graph
from tabulate import tabulate


class GAColorize:
    def __init__(self, graph_filename, ga_params_filename):
        """
        Initialize the GAColorize object. Load the graph and GA parameters from the given files.
        """
        self.graph = Graph(graph_filename)
        self.ga_params = {}
        self.load_ga_params(ga_params_filename)
        self.ga_params['num_nodes'] = len(self.graph.nodes)
        self.ga_params['num_edges'] = len(self.graph.edges)
        self.ga_params['bits_per_individual'] = math.ceil(math.log2(self.ga_params['num_nodes']))
        # self.print_params()

    def load_ga_params(self, path):
        """
        Load the GA parameters from a file. The file should have the following format:
            num_runs [int]
            num_generations [int]
            num_individuals [int]
            mutation_prob [float]
            crossover_prob [float]
            max_no_improvement_generations [int]
        """
        with open(path, mode='r') as f:
            for line in f.readlines():
                parts = line.split()
                param_name = parts[0]
                if param_name in ['num_runs', 'num_generations', 'num_individuals', 'max_no_improvement_generations']:
                    self.ga_params[param_name] = int(parts[1])
                elif param_name in ['mutation_prob', 'crossover_prob']:
                    self.ga_params[param_name] = float(parts[1])

    def print_params(self):
        """
        Print the loaded GA parameters.
        """
        params_table = [[key.replace('_', ' ').capitalize(), value] for key, value in self.ga_params.items()]
        print("Loaded GA parameters:")
        print(tabulate(params_table, headers=["Parameter", "Value"], tablefmt="grid"))

    def encode_individual(self, color_pairs):
        """
        Convert individual into a binary sequence. Each node is represented by a number of bits equal to the number of
        colors. The color of the node is encoded as a binary number. The binary sequences of all nodes are concatenated
        to form the individual.
        """
        result = sum(color << (vertex * self.ga_params['bits_per_individual']) for vertex, color in color_pairs)
        return result

    def decode_individual(self, binary_seq):
        """
        Reverse the encoding of vertices from a binary sequence. Extract the color of each node by using a mask and
        shifting bits. Return a list of pairs (node_id, color).
        """
        color_pairs = []
        mask = (1 << self.ga_params['bits_per_individual']) - 1  # Mask with `bit_size` number of 1s

        for node in range(self.ga_params['num_nodes']):
            color = (binary_seq >> (node * self.ga_params['bits_per_individual'])) & mask
            color_pairs.append((node, color))

        return [(node + 1, color) for node, color in color_pairs]

    def get_color(self, nodes_colors, node):
        for n, color in nodes_colors:
            if n == node:
                return color

    def validate_coloring(self, nodes_colors):
        """
        Check if the given coloring is valid by ensuring that each edge connects vertices of different colors.

        Parameters:
        - nodes_colors: A list of tuples where each tuple contains a node (vertex) and its assigned color.
          Example: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)] means vertex 0 is colored 1, vertex 1 is colored 2, and so
          on.
        """
        edges = self.graph.edges  # Step 1: Retrieve the list of edges from the graph
        # print(edges)
        # print(nodes_colors)

        for node1, node2 in edges:  # Step 2: Iterate over each edge (node1, node2)
            # print(node1)
            # print(node2)
            color1 = self.get_color(nodes_colors, node1)
            color2 = self.get_color(nodes_colors, node2)

            if color1 == color2:  # Step 5: Check if the colors are the same
                return False  # If same color, return False (invalid coloring)

        return True  # Step 6: If all edges checked and valid, return True (valid coloring)

    def initialize_individuals(self):
        """
        Generate the initial population by randomly selecting valid individuals. The initial population at T = 0 consists
        only of valid graphs.
        """
        random.seed()  # Step 1: Initialize a random seed
        individuals = []  # Step 2: Initialize an empty list to store the population

        def create_coloring():
            """Generate a random coloring for the vertices."""
            available_colors = list(range(self.ga_params['num_nodes']))  # List of available colors
            return [(node + 1, random.choice(available_colors)) for node in range(self.ga_params['num_nodes'])]

        while len(individuals) < self.ga_params[
            'num_individuals']:  # Step 3: Loop until the population reaches the desired size
            node_coloring = create_coloring()  # Step 4: Generate a node coloring

            if self.validate_coloring(node_coloring):  # Step 5: Validate the generated coloring
                encoded_individual = self.encode_individual(node_coloring)  # Encode the valid coloring
                individuals.append(encoded_individual)  # Add to population

        return individuals  # Step 6: Return the generated population

    def check_end_conditions(self, gen_num, solutions):
        """
        Check if the end conditions are met for the genetic algorithm. The algorithm will stop if:
        1. The number of generations exceeds the maximum number of generations.
        2. There is no improvement for a certain number of generations.
        """
        no_improvement_detected = False

        # Check if the number of generations exceeds the threshold for checking improvements
        if gen_num > self.ga_params['max_no_improvement_generations']:
            latest_solution = solutions[-1]
            past_solutions = solutions[-self.ga_params['max_no_improvement_generations']:-1]

            # Check if there has been no improvement in the fitness value
            no_improvement_detected = latest_solution[1] <= past_solutions[0][1]

        # Stop if the maximum number of generations is reached or if there is no improvement
        return gen_num == self.ga_params['num_generations'] or no_improvement_detected

    def calculate_individuals_fitness(self, individuals):
        """
        Calculate the fitness of each individual in the population. The fitness is calculated as the number of unique
        colors used in the graph. The lower the number of colors, the better the fitness.
        """
        fitness_scores = []

        for individual in individuals:
            # Decoding the individual
            coloring = self.decode_individual(individual)

            # Calculating the fitness
            unique_colors = set(color for node, color in coloring)
            fitness = 100 * self.ga_params['num_nodes'] / len(unique_colors)

            # Appending the individual and its fitness
            fitness_scores.append((individual, fitness))

        return fitness_scores

    def select_individuals(self, individuals):
        """
        Select individuals from the population based on their fitness. Individuals with higher fitness have a higher
        chance of being selected. The selection is done using the roulette wheel selection method. The selection process
        is repeated until the desired number of individuals is selected.
        """
        individuals_with_fitness = self.calculate_individuals_fitness(individuals)

        fitness_sum = 0
        before_selection = []
        for individual, fitness in individuals_with_fitness:
            fitness_sum += fitness
            before_selection.append((individual, fitness_sum))

        after_selection = []

        for _ in range(self.ga_params['num_individuals']):
            draw = random.randint(0, int(fitness_sum))
            for individual, fitness_margin in before_selection:
                if fitness_margin >= draw:
                    after_selection.append(individual)
                    break

        return after_selection

    def apply_crossover(self, individuals):
        """
        Perform crossover on the selected individuals. The crossover operation is performed with a certain probability
        defined by the crossover probability parameter. If the probability is met, two individuals are selected from the
        population and a crossover point is chosen. The bits before the crossover point are swapped between the two
        individuals to create two new individuals. The new individuals are added to the population.
        """
        pairs = []
        num_pairs = math.floor(self.ga_params['num_individuals'] / 2)  # Number of pairs for crossover

        # Create pairs of individuals for crossover
        for _ in range(num_pairs):
            individual1 = random.choice(individuals)
            individual2 = random.choice(individuals)
            pairs.append((individual1, individual2))

        new_population = []

        # Perform crossover on each pair
        for parent1, parent2 in pairs:
            if random.uniform(0.0, 1.0) < self.ga_params['crossover_prob']:
                crossover_point = random.randint(0, self.ga_params['bits_per_individual'])
                mask = (1 << crossover_point) - 1  # Create mask with bits set up to the crossover point

                # Create new individuals by swapping bits at the crossover point
                offspring1 = (parent1 & mask) | (parent2 & ~mask)
                offspring2 = (parent1 & ~mask) | (parent2 & mask)

                new_population.extend([offspring1, offspring2])
            else:
                new_population.extend([parent1, parent2])

        return new_population

    import random

    def apply_mutation(self, individuals):
        """
        Apply mutation to each individual in the population with a given probability.
        The mutation involves flipping bits of the individual at random positions based on the mutation probability.
        """
        mutated_population = []
        mutation_probability = self.ga_params['mutation_prob']  # Probability of mutation

        for individual in individuals:
            mutated_individual = individual

            # Perform mutation on each bit of the individual
            for bit_position in range(self.ga_params['bits_per_individual']):
                if random.uniform(0.0, 1.0) < mutation_probability:
                    mutated_individual ^= (1 << bit_position)  # Flip the bit at the current position

            mutated_population.append(mutated_individual)

        return mutated_population

    def find_best_individual(self, population):
        """
        Select the best individual from a set of acceptable solutions and return
        the individual along with its fitness. If no acceptable solution is found,
        return None for both the individual and its fitness.
        """
        # Decode all individuals to get their colorings
        colorings = [self.decode_individual(individual) for individual in population]

        # Filter and encode only the acceptable colorings
        acceptable_solutions = [
            self.encode_individual(coloring) for coloring in colorings
            if self.validate_coloring(coloring)
        ]

        if not acceptable_solutions:
            return None, None

        # Choose the best individual from the acceptable solutions
        best_individual = acceptable_solutions[0]
        # Decode the best individual to get its coloring
        best_coloring = self.decode_individual(best_individual)
        # Calculate the fitness of the best individual
        unique_colors = set(color for node, color in best_coloring)
        best_fitness = 100 * self.ga_params['num_nodes'] / len(unique_colors)

        # Return the best individual and its fitness (if not found, returns None)
        return best_individual, best_fitness

    def run_genetic_algorithm(self):
        """
        Run the genetic algorithm to find the best coloring of the graph. The algorithm will run for a certain number of
        generations and return the best individual found. The algorithm will stop if there is no improvement in the best
        individual for a certain number of generations. The algorithm will return the best individual found along with its fitness.
        """
        results = []
        for run_index in range(self.ga_params['num_runs']):
            print(f'Starting run number {run_index}')

            # Initialize the population
            individuals = self.initialize_individuals()
            # print('Initial population:', individuals)
            print('Population initialized...')

            generation = 0
            solutions = [self.find_best_individual(individuals)]

            # Run the genetic algorithm until the stop condition is reached
            while not self.check_end_conditions(generation, solutions):
                # Apply genetic operators
                individuals = self.select_individuals(individuals)
                individuals = self.apply_crossover(individuals)
                individuals = self.apply_mutation(individuals)

                # Track the best solution in the current population
                solutions.append(self.find_best_individual(individuals))
                generation += 1

            # Determine the best solution from all generations
            best_individual, best_fitness = solutions[0]
            # print(f"The best fitness is: {best_fitness}")
            # print(solutions)
            for individual, fitness in solutions:
                if individual:
                    if fitness > best_fitness:
                        best_individual, best_fitness = individual, fitness

            results.append((best_individual, best_fitness))
        return results

# Example usage
# GAColorize('example_graph.txt', 'baseline_ga_params.txt')
