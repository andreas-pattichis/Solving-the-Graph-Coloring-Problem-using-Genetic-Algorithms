class Node:
    def __init__(self, node_id):
        self.id = node_id

    def __repr__(self):
        return f"Node({self.id})"


class Edge:
    def __init__(self, start_node, end_node, weight=1):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight

    def __repr__(self):
        # return f"Edge({self.start_node.id} -> {self.end_node.id}, weight={self.weight})"
        return [self.start_node.id, self.end_node.id]


class Graph:
    def __init__(self, path=None):
        self.nodes = {}
        self.edges = []
        if path:
            self.load_from_file(path)

    def load_from_file(self, path):
        with open(path, mode='r') as f:
            for line in f.readlines():
                parts = line.split()
                if parts[0] == 'e':
                    _, start_id, end_id, weight = parts
                    self.add_edge(int(start_id), int(end_id), int(weight))
                elif parts[0] == 'n':
                    no_of_nodes = len(parts) - 1
                    for i in range(no_of_nodes):
                        self.add_node(int(parts[i + 1]))

    def add_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)

    def add_edge(self, start_id, end_id, weight=1):
        if start_id in self.nodes and end_id in self.nodes and start_id != end_id:
            edge = Edge(self.nodes[start_id], self.nodes[end_id], weight)
            self.edges.append(edge.__repr__())

    def __str__(self):
        return f'Graph with {len(self.nodes)} nodes and {len(self.edges)} edges.'
