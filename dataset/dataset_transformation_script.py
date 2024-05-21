import os


# Function to transform the content of a file
def transform_content(file_content):
    lines = file_content.splitlines()
    transformed_lines = []

    # Extracting the header line and the edges
    header = lines[0]
    edges = lines[1:]

    # Adding the header
    transformed_lines.append(header)

    # Collecting all unique node numbers
    nodes = set()
    for edge in edges:
        _, node1, node2 = edge.split()
        nodes.add(node1)
        nodes.add(node2)

    # Adding the node line in the desired format
    nodes_list = sorted(nodes, key=int)
    transformed_lines.append("n " + " ".join(nodes_list))

    # Adding the edges in the desired format
    for edge in edges:
        _, node1, node2 = edge.split()
        transformed_lines.append(f"e {node1} {node2} 1")

    return "\n".join(transformed_lines)


# Function to process all files in the dataset_steven folder
def process_files(dataset_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(dataset_folder, filename)
            with open(input_file_path, 'r') as file:
                file_content = file.read()

            transformed_content = transform_content(file_content)

            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, 'w') as file:
                file.write(transformed_content)


# Example usage
dataset_folder = 'dataset_steven'
output_folder = 'dataset_small'
process_files(dataset_folder, output_folder)

print("Transformation completed.")
