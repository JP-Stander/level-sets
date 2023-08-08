# %%
from os import listdir
from networkx import Graph, read_graphml
from images.utils import load_image
from level_sets.utils import get_fuzzy_sets, get_level_sets
import numpy as np
import networkx as nx
import pandas as pd
from scipy.special import entr

# %%

def edge_entropy(G):
    if not nx.is_weighted(G):
        raise ValueError("Graph is not weighted")
    w = list(nx.get_edge_attributes(G, 'weight').values())
    c = np.array(pd.value_counts(w))
    p = c / len(w)
    entropy = -sum(np.log2(p)*p)
    return entropy

def attribute_entropy(G, attribute):
    a = list(nx.get_node_attributes(G, attribute).values())
    c = np.array(pd.value_counts(a))
    p = c / len(a)
    entropy = -sum(np.log2(p)*p)
    return entropy

def degree_entropy(G):
    d = [deg for n, deg in G.degree()]
    c = np.array(pd.value_counts(d))
    p = c / len(d)
    entropy = -sum(np.log2(p)*p)
    return entropy

def image_to_graph(img):

    # Create a graph
    G = nx.Graph()

    # Get the dimensions of the image
    rows, cols = img.shape

    # Helper function to check if a pixel is within the image boundaries
    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    # Add nodes and edges to the graph
    for r in range(rows):
        for c in range(cols):
            # Add node with intensity attribute
            G.add_node((r, c), intensity=img[r, c])

            # Connect the pixel to its 8 neighbors
            directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]

            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if in_bounds(new_r, new_c):
                    G.add_edge((r, c), (new_r, new_c))

    return G

# %%
image_names = ["dotted_0161", "fibrous_0116"]
image_name = image_names[1]
img = load_image(f"../dtd/images/{image_name.split('_')[0]}/{image_name}.jpg", [50,50])
ls = get_level_sets(img, 2)
fs = get_fuzzy_sets(img, 10, 8)
# %%
ls_g = read_graphml(f"../graphical_models/level_sets/{image_name}_graph.graphml")
ls_n_edges = len(ls_g.edges())
ls_degree_entr = degree_entropy(ls_g)
ls_intensity_entr = attribute_entropy(ls_g, 'intensity')

fs_g = read_graphml(f"../graphical_models/fuzzy_sets_10/{image_name}_graph.graphml")
fs_n_edges = len(fs_g.edges())
fs_degree_entr = degree_entropy(fs_g)
fs_intensity_entr = attribute_entropy(fs_g, 'intensity')

og_g = image_to_graph(img)
og_n_edges = len(og_g.edges())
og_degree_entr = degree_entropy(og_g)
og_intensity_entr = attribute_entropy(og_g, 'intensity')

# edge_entropy(ls_g)
print(image_name)
print(f"Number of pixels : {img.shape[0]*img.shape[1]}")
print(f"Number of level-sets: {np.max(ls)}")
print(f"Number of fuzzy-sets {np.max(fs)+1}")

print("Level-sets graph")
print(f"Number of edges: {ls_n_edges}")
print(f"Ratio: {ls_n_edges/og_n_edges}")
print(f"Intensity entropy: {ls_intensity_entr}")
print(f"Ratio: {ls_intensity_entr/og_intensity_entr}")
print(f"Degree entropy: {ls_degree_entr}")
print(f"Ratio: {ls_degree_entr/og_degree_entr}")

print("Fuzzy-sets graph")
print(f"Number of edges: {fs_n_edges}")
print(f"Ratio: {fs_n_edges/og_n_edges}")
print(f"Intensity entropy: {fs_intensity_entr}")
print(f"Ratio: {fs_intensity_entr/og_intensity_entr}")
print(f"Degree entropy: {fs_degree_entr}")
print(f"Ratio: {fs_degree_entr/og_degree_entr}")

print("Image graph")
print(f"Number of edges: {og_n_edges}")
print(f"Intensity entropy: {og_intensity_entr}")
print(f"Degree entropy: {og_degree_entr}")
# %%
