# %%
import os
import sys
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
import json
import warnings
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import networkx as nx
from config import nodes_feature_names, classes, graphs_location, experiment_loc, trim, max_graphlet_size
from subgraph.counter import count_unique_subgraphs
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.model_selection import train_test_split

# %% Build dataset

graph_files = [f"{graphs_location}/{clas}/{file}" for clas in classes for file in os.listdir(f"{graphs_location}/{clas}")]
node_counts = pd.DataFrame(0, index=range(len(graph_files)), columns=nodes_feature_names)
pix_idx = True
# Read graphs
feats = {name: [] for name in classes}
connected_subgraphs = {name: [] for name in classes}
for clas in classes:
    print(f"{clas} is class {classes.index(clas)}")

for i, file in enumerate(tqdm(graph_files)):
    graph = nx.read_graphml(file)
    clas = file.split('/')[-2]
    temp_df = pd.DataFrame(0, index=[0], columns=nodes_feature_names)
    for node, data in graph.nodes(data=True):
        for feature_name in nodes_feature_names:
            temp_df.loc[int(node), feature_name] = data[feature_name]
        if pix_idx is True:
            temp_df.loc[int(node), "pixel_indices"] = data["pixel_indices"]
    subgraph_counts = count_unique_subgraphs(graph, max_graphlet_size)
    sorted_sg = dict(sorted(subgraph_counts.items()))

    # Extract the values in the desired order
    values = []
    for key in sorted_sg.keys():
        sub_dict = sorted_sg[key]
        for sub_key in sorted(sub_dict):
            values.append(sub_dict[sub_key])
    
    # Convert the list of values to a numpy array with shape (9, 1)
    graphlets = np.array(values).reshape(-1, )
    loc = classes.index(clas)
    feats[clas] += [np.array(temp_df)]
    connected_subgraphs[clas] += [graphlets]

if not os.path.exists(experiment_loc):
    os.makedirs(experiment_loc)

train_ratio = 0.8
for key in feats.keys():
    # Get the number of arrays in the list (which corresponds to the number of images)
    num_images = len(feats[key])
    
    # Generate indices and split them into training and testing
    indices = list(range(num_images))
    train_indices, test_indices = train_test_split(indices, test_size=1-train_ratio, random_state=42)
    
    # Split feats
    train_feats = [feats[key][i] for i in train_indices]
    test_feats = [feats[key][i] for i in test_indices]
    
    # Split connected_subgraphs
    train_subgraphs = [connected_subgraphs[key][i] for i in train_indices]
    test_subgraphs = [connected_subgraphs[key][i] for i in test_indices]
    
    # Save training feats
    combined_array_train = np.vstack(train_feats)
    np.save(f"{experiment_loc}/{key}_train_data.npy", combined_array_train)
    indices_train = np.cumsum([0] + [arr.shape[0] for arr in train_feats])
    np.save(f"{experiment_loc}/{key}_train_indices.npy", indices_train)
    
    # Save testing feats
    combined_array_test = np.vstack(test_feats)
    np.save(f"{experiment_loc}/{key}_test_data.npy", combined_array_test)
    indices_test = np.cumsum([0] + [arr.shape[0] for arr in test_feats])
    np.save(f"{experiment_loc}/{key}_test_indices.npy", indices_test)
    
    # Save training subgraphs
    with open(f"{experiment_loc}/{key}_train_subgraphs.pkl", 'wb') as f:
        pickle.dump(train_subgraphs, f)
    
    # Save testing subgraphs
    with open(f"{experiment_loc}/{key}_test_subgraphs.pkl", 'wb') as f:
        pickle.dump(test_subgraphs, f)

# %%
