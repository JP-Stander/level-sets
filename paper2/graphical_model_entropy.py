# %%
import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from scipy.linalg import eigh
from scipy.special import entr
import matplotlib.pyplot as plt

# %%


def edge_entropy(G):
    if not nx.is_weighted(G):
        raise ValueError("Graph is not weighted")
    w = nx.get_edge_attributes(G, 'weight').values()
    p = np.bincount(w) / len(w)
    entropy = -np.sum(entr(p))
    return entropy


def spectral_entropy(G):
    L = nx.laplacian_matrix(G).todense()
    eigenvalues = eigh(L, eigvals_only=True)
    p = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(entr(p))
    return entropy


def attribute_entropy(G, attribute):
    a = nx.get_node_attributes(G, attribute).values()
    p = np.bincount(a) / len(a)
    entropy = -np.sum(entr(p))
    return entropy

# %%


path_to_files = "../my_graphical_dataset/img_size_30_spatial_euclidean_attr_cityblock"
all_files = [f for f in os.listdir(path_to_files) if f.endswith('.json')]

for file in all_files[:1]:
    # Read the JSON file
    with open(os.path.join(path_to_files, file)) as f:
        json_data = json.load(f)

    # Create an edge dataframe
    edges_df = pd.DataFrame(json_data['edges'])
    edges_df['from'] = edges_df['edge'].apply(lambda x: x[0])
    edges_df['to'] = edges_df['edge'].apply(lambda x: x[1])
    edges_df['weight'] = edges_df['weight'].apply(lambda x: x[0])
    edges_df = edges_df.drop(columns=['edge'])

    # Create a node attributes dataframe
    features_df = pd.DataFrame.from_dict(json_data['features'], orient='index').reset_index()
    features_df = features_df.rename(columns={"index": "name", 0: "intensity", 1: "size", 2: "compactness", 3: "elongation"})

    # Create an empty graph
    G = nx.Graph()

    # Add nodes and their attributes to the graph
    for i, row in features_df.iterrows():
        G.add_node(row['name'], **row[1:].to_dict())

    # Add edges to the graph
    for i, row in edges_df.iterrows():
        G.add_edge(row['from'], row['to'])
# %%

fig, ax = plt.subplots()
# nx.draw(G, with_labels=True)
# nx.draw_shell(G, with_labels=True, ax=ax)
nx.draw_spring(G, with_labels=True, ax=ax)
plt.show()
