import math
import random

from tabulate import tabulate

from gcp_implementation.graph import Graph


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
        result = 0
        for vertex, color in color_pairs:
            result |= color << ((vertex - 1) * self.ga_params['bits_per_individual'])
        return result

    def decode_individual(self, binary_seq):
        """
        Reverse the encoding of vertices from a binary sequence. Extract the color of each node by using a mask and
        shifting bits. Return a list of pairs (node_id, color).
        """
        color_pairs = []
        mask = (1 << self.ga_params['bits_per_individual']) - 1  # Mask with `bits_per_individual` number of 1s

        for node in range(self.ga_params['num_nodes']):
            color = (binary_seq >> (node * self.ga_params['bits_per_individual'])) & mask
            color_pairs.append((node + 1, color))

        return color_pairs

    def get_color(self, nodes_colors, node):
        for n, color in nodes_colors:
            if n == node:
                return color
        return None

    def validate_coloring(self, nodes_colors):
        """
        Check if the given coloring is valid by ensuring that each edge connects vertices of different colors.

        Parameters:
        - nodes_colors: A list of tuples where each tuple contains a node (vertex) and its assigned color.
        """
        edges = self.graph.edges  # Retrieve the list of edges from the graph

        for node1, node2 in edges:  # Iterate over each edge (node1, node2)
            color1 = self.get_color(nodes_colors, node1)
            color2 = self.get_color(nodes_colors, node2)

            if color1 is None or color2 is None:
                return False  # If any color is not found, the coloring is invalid

            if color1 == color2:  # Check if the colors are the same
                return False  # If same color, return False (invalid coloring)

        return True  # If all edges checked and valid, return True (valid coloring)

    def initialize_individuals(self):
        """
        Generate the initial population by randomly selecting valid individuals. The initial population at T = 0 consists
        only of valid graphs.
        """
        random.seed()  # Initialize a random seed
        individuals = []  # Initialize an empty list to store the population

        def create_coloring():
            """Generate a random coloring for the vertices."""
            available_colors = list(range(self.ga_params['num_nodes']))  # List of available colors
            return [(node + 1, random.choice(available_colors)) for node in range(self.ga_params['num_nodes'])]

        while len(individuals) < self.ga_params[
            'num_individuals']:  # Loop until the population reaches the desired size
            node_coloring = create_coloring()  # Generate a node coloring

            if self.validate_coloring(node_coloring):  # Validate the generated coloring
                encoded_individual = self.encode_individual(node_coloring)  # Encode the valid coloring
                individuals.append(encoded_individual)  # Add to population

        return individuals  # Return the generated population

    def check_end_conditions(self, gen_num, solutions):
        """
        Check if the end conditions are met for the genetic algorithm. The algorithm will stop if:
        1. The number of generations exceeds the maximum number of generations.
        2. The fitness of the first solution in the last 30 generations is the same as the current fitness.
        """
        if gen_num >= self.ga_params['num_generations']:
            return True

        if gen_num > self.ga_params['max_no_improvement_generations']:
            # Get the first and last solution in the last 30 generations
            first_solution = solutions[-self.ga_params['max_no_improvement_generations']]
            # Get the last 30 solutions
            last_solutions = solutions[-self.ga_params['max_no_improvement_generations']:]
            # Filter out None solutions
            last_solutions = [sol for sol in last_solutions if sol is not None and sol[1] is not None]

            if not last_solutions:
                return False

            # From these get the one with the highest fitness
            best_last_solution = max(last_solutions, key=lambda x: x[1])

            # If the fitness of the first solution is the same as the best fitness in the last 30 generations
            if first_solution[1] == best_last_solution[1]:
                return True

        return False

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

    def select_individuals(self, individuals, tournament_size=3):
        """
        Select individuals from the population based on their fitness using tournament selection.
        """
        individuals_with_fitness = self.calculate_individuals_fitness(individuals)

        def tournament_selection():
            selected = random.sample(individuals_with_fitness, tournament_size)
            return max(selected, key=lambda x: x[1])[0]

        selected_individuals = [tournament_selection() for _ in range(self.ga_params['num_individuals'])]
        return selected_individuals

    def apply_crossover(self, individuals, crossover_probability):
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
            if random.uniform(0.0, 1.0) < crossover_probability:
                crossover_point = random.randint(0, self.ga_params['bits_per_individual'])
                mask = (1 << crossover_point) - 1  # Create mask with bits set up to the crossover point

                # Create new individuals by swapping bits at the crossover point
                offspring1 = (parent1 & mask) | (parent2 & ~mask)
                offspring2 = (parent1 & ~mask) | (parent2 & mask)

                new_population.extend([offspring1, offspring2])
            else:
                new_population.extend([parent1, parent2])

        return new_population

    def apply_mutation(self, individuals, mutation_probability):
        """
        Apply mutation to each individual in the population with a given probability.
        The mutation involves flipping bits of the individual at random positions based on the mutation probability.
        """
        mutated_population = []
        # mutation_probability = self.ga_params['mutation_prob']  # Probability of mutation

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
            (self.encode_individual(coloring), coloring) for coloring in colorings
            if self.validate_coloring(coloring)
        ]

        if not acceptable_solutions:
            return None, None

        # Initialize best fitness to a very low value
        best_fitness = float('-inf')
        best_individual = None

        # Iterate over acceptable solutions to find the best one
        for encoded_individual, coloring in acceptable_solutions:
            unique_colors = set(color for node, color in coloring)
            fitness = 100 * self.ga_params['num_nodes'] / len(unique_colors)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = encoded_individual

        # Return the best individual and its fitness
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

            # Determine the appropriate tournament size based on the population size
            population_size = self.ga_params['num_individuals']
            if population_size <= 50:
                tournament_size = 2
            elif population_size <= 150:
                tournament_size = 3
            else:
                tournament_size = 5

            # Run the genetic algorithm until the stop condition is reached
            while not self.check_end_conditions(generation, solutions):
                # Apply genetic operators
                individuals = self.select_individuals(individuals, tournament_size=tournament_size)
                # commented below can be found the parameters in the ILM/DHC mode
                # mutation_probability = generation/self.ga_params['num_generations']
                # crossover_probability = 1 - generation/self.ga_params['num_generations']

                # below can be found the parameters in the DHM/ILC
                crossover_probability = generation / self.ga_params['num_generations']
                mutation_probability = 1 - generation / self.ga_params['num_generations']
                individuals = self.apply_crossover(individuals, crossover_probability)
                individuals = self.apply_mutation(individuals, mutation_probability)

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
