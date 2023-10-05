# %%
import os
import sys
import joblib
import pandas as pd
import numpy as np
from utils import process_sublist
import networkx as nx
from matplotlib import pyplot as plt
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from subgraph.counter import count_unique_subgraphs
from images.utils import load_image
from utils import img_to_graph
from level_sets.metrics import get_metrics
from level_sets.utils import get_fuzzy_sets, get_level_sets
from config import sets_feature_names, images_loc, graphs_location, \
        experiment_loc, results_location, graphlet_names, num_clusters, \
        fs_delta, img_size, edge_delta, fs_connectivity, max_graphlet_size, trim, \
        images_for_inference, classes, trim, nodes_feature_names
import warnings
warnings.simplefilter(action='ignore', category=Warning)
# %%

model = joblib.load(f"{experiment_loc}/logistic_regression_model.pkl")
kmeans = joblib.load(f"{experiment_loc}/kmeans.pkl")
with open(f"{experiment_loc}/contains_zero.txt", "r") as f:
    contains_zero = f.read().splitlines()
contains_zero = ["g"+g for g in contains_zero]
# %%
for clas, img_name in images_for_inference.items():
    img_name_only = img_name.split(".")[0]
    img_file = f"{images_loc}/{clas}/{img_name}"
    graph = img_to_graph(
        img_file,
        "",
        fs_delta,
        img_size,
        fs_connectivity,
        edge_delta,
        True,
        sets_feature_names,
        trim
    )
    img_graph = f"{graphs_location}/{img_name_only}_graph.graphml"
    # graph = nx.read_graphml(img_graph)
    img = load_image(img_file, [img_size, img_size], trim=trim)
    img = img/255

    coefs = {name:0 for name in graphlet_names}
    for i, key in enumerate(coefs.keys()):
        coefs[key] = model.coef_[0,i]
    coefs_plane = np.zeros(img.shape)
    nodes_attrs = []
    pred_df3 = pd.DataFrame(0, index=[0], columns = graphlet_names)
    for node, data in graph.nodes(data=True):
        set_attr = pd.DataFrame()
        for feature_name in nodes_feature_names:
            set_attr.loc[int(node), feature_name] = data[feature_name]
        set_attr.loc[int(node), "intensity"] = data["intensity"]
        g_number = kmeans.predict(set_attr.replace(np.inf, 0))[0]
        pred_df3[f"g{g_number+1}"] += 1
        # coefs_plane[level_sets==ls] = coefs[f"g{g_number+1}"]
        ls_idx = [a for a in eval(data.get("pixel_indices"))] if ")," in data.get("pixel_indices") else [eval(data.get("pixel_indices"))]
        rows, cols = zip(*ls_idx)
        coefs_plane[rows, cols] = coefs[f"g{g_number+1}"] if f"g{g_number+1}" not in contains_zero else None
    subgraph_counts = count_unique_subgraphs(graph, 2)
    sorted_sg = dict(sorted(subgraph_counts.items()))

    # Extract the values in the desired order
    values = []
    for key in sorted_sg.keys():
        sub_dict = sorted_sg[key]
        for sub_key in sorted(sub_dict):
            values.append(sub_dict[sub_key])
    
    # Convert the list of values to a numpy array with shape (9, 1)
    graphlets = np.array(values).reshape(-1, )
    pred_df3['g2_1'] = graphlets
    # Convert the list of values to a numpy array with shape (9, 1)
    yhat = model.predict(pred_df3)
    plt.figure()
    plt.imshow(img, 'gray')
    plt.imshow(
        coefs_plane, alpha=0.5,
        cmap = 'bwr',
        vmin=np.min(model.coef_), 
        vmax=np.max(model.coef_)
    )
    plt.colorbar()
    plt.axis("off")
    plt.savefig(f"{results_location}/{img_name_only}_heatmap.png")

    print(f"Predicted class: {classes[yhat[0]]}({yhat[0]})")
    print(f"Actual class : {clas}")

# %%
