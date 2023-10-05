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
        temp_df.loc[int(node), "intensity"] = data["intensity"]
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

feats_to_save = feats.copy()
for key in feats_to_save.keys():
    feats_to_save[key] = [arr.tolist() for arr in feats_to_save[key]]
# Save the dictionary
if pix_idx is True:
    with open(f"{experiment_loc}/feats_full.pkl", 'wb') as f:
        pickle.dump(feats_to_save, f)
else:
    with open(f"{experiment_loc}/feats.pkl", 'wb') as f:
        pickle.dump(feats_to_save, f)

subgraphs_to_save = connected_subgraphs.copy()
for key in subgraphs_to_save.keys():
    subgraphs_to_save[key] = [arr.tolist() for arr in subgraphs_to_save[key]]
# Save the dictionary
with open(f"{experiment_loc}/subgraphs.pkl", 'wb') as f:
    pickle.dump(subgraphs_to_save, f)

# %%
