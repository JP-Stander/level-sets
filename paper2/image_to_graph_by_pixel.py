# %%
from os import listdir, cpu_count
from networkx import Graph, write_graphml
from images.utils import load_image
from graphical_model.utils import graphical_model
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import networkx as nx

import networkx as nx

def image_to_graph(image):
    img_size = 50
    # I assume you have the load_image function defined elsewhere in your code
    img = load_image(image, [img_size, img_size])

    # Create a graph
    G = nx.Graph()

    # Get the dimensions of the image
    rows, cols = img.shape

    # Helper function to check if a pixel is within the image boundaries
    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    node_counter = 0  # Counter for new node IDs
    node_mapping = {}  # Mapping from (r, c) to the new node ID

    # Add nodes and edges to the graph
    for r in range(rows):
        for c in range(cols):
            # Map the (r, c) tuple to the new node ID
            node_mapping[(r, c)] = node_counter
            node_counter += 1

    for r in range(rows):
        for c in range(cols):
            # Add node with intensity attribute using the new node ID
            G.add_node(node_mapping[(r, c)], intensity=img[r, c])

            # Connect the pixel to its 8 neighbors
            directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]

            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if in_bounds(new_r, new_c):
                    G.add_edge(node_mapping[(r, c)], node_mapping[(new_r, new_c)])
    write_graphml(G, f"../graphical_models/pixels/{image.split('/')[-1].split('.')[0]}_graph.graphml")
    

# %%

if __name__ == '__main__':
    images_path = "../dtd/images/"
    images = ["../dtd/images/dotted/" + file for file in listdir("../dtd/images/dotted")]
    images += ["../dtd/images/fibrous/" + file for file in listdir("../dtd/images/fibrous")]
    images = images

    n_cores = cpu_count() - 2
    print(f'Running process on {n_cores} cores')
    # with ProcessPoolExecutor(max_workers=n_cores) as executor:
    #     executor.map(img_to_graph, images)
    with ProcessPoolExecutor() as executor:
        # Setup tqdm
        future_to_image = {executor.submit(image_to_graph, image): image for image in images}
        for future in tqdm(as_completed(future_to_image), total=len(images)):
            try:
                future.result()  # retrieve results if there are any
            except Exception as e:
                print(f"Error processing image: {future_to_image[future]}. Error: {e}")

