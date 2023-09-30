# %%
import os
import sys
import joblib
import pandas as pd
import numpy as np
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
        fs_delta, img_size, edge_delta, fs_connectivity, max_graphlet_size

# %%

model = joblib.load(f"{experiment_loc}/logistic_regression_model.pkl")
kmeans = joblib.load(f"{experiment_loc}/kmeans.pkl")
# %%
for img_name in ["dotted_0188", "fibrous_0191"]: #, "dotted_0111", "dotted_0180", "fibrous_0191", "fibrous_0108", "fibrous_0116"]:
    clas = img_name.split('_')[0]
    img_file = f"{images_loc}/{clas}/{img_name}.jpg"
    img_graph = f"{graphs_location}/{img_name}_graph.graphml"
    img_to_graph(img_file, graphs_location, fs_delta, img_size, edge_delta)
    img = load_image(img_file, [img_size,img_size])
    img = img/255
    graph = nx.read_graphml(img_graph)

    level_sets = get_fuzzy_sets(img, fs_delta/255, fs_connectivity)
    uls = pd.unique(level_sets.flatten())
    coefs = {name:0 for name in graphlet_names}
    for i, key in enumerate(coefs.keys()):
        coefs[key] = model.coef_[0,i]
    coefs_plane = np.zeros(img.shape)
    pred_df = pd.DataFrame(0, index=[0], columns=graphlet_names)

    for ls in uls:
        set = np.zeros(img.shape)
        set[level_sets==ls] = 1

        set_attr = pd.DataFrame()
        metrics = get_metrics(set, metric_names=sets_feature_names)
        for key in metrics.keys():
            set_attr.loc[0, key] = metrics[key]
        set_attr.loc[0, "intensity"] = img[level_sets==ls][0]

        min_idx = kmeans.predict(set_attr)
        coefs_plane[level_sets==ls] = coefs[f"g{min_idx[0]+1}"]
        pred_df.loc[0, f"g{min_idx[0]+1}"] += 1

    subgraph_counts = count_unique_subgraphs(graph, max_graphlet_size)
    sorted_sg = dict(sorted(subgraph_counts.items()))

    # Extract the values in the desired order
    values = []
    for key in sorted_sg.keys():
        sub_dict = sorted_sg[key]
        for sub_key in sorted(sub_dict):
            values.append(sub_dict[sub_key])

    # Convert the list of values to a numpy array with shape (9, 1)
    pred_df.iloc[0,-9:] = np.array(values).reshape(-1, )
    yhat = model.predict(pred_df)
    coefs_plane[coefs_plane==0] = None
    plt.figure()
    plt.imshow(img, 'gray')
    plt.imshow(
        coefs_plane, alpha=0.5, 
        vmin=np.min(model.coef_), 
        vmax=np.max(model.coef_)
    )
    plt.colorbar()
    plt.axis("off")
    plt.savefig(f"{results_location}/{img_name}_heatmap.png")

    print(f"Predicted class: {'dotted(0)' if yhat[0] == 0 else 'firbous(1)'}")
    print(f"Actual class : {clas}")
# %%
