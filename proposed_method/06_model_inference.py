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
from matplotlib.colors import TwoSlopeNorm
from level_sets.metrics import get_metrics
from level_sets.utils import get_fuzzy_sets, get_level_sets
from config import sets_feature_names, images_loc, graphs_location, \
        experiment_loc, results_location, graphlet_names, num_clusters, \
        fs_delta, img_size, edge_delta, fs_connectivity, max_graphlet_size, trim, \
        images_for_inference, classes, trim, nodes_feature_names
import warnings
warnings.simplefilter(action='ignore', category=Warning)
# %%
for clas in classes:
    print(f"{clas} is class {classes.index(clas)}")

model = joblib.load(f"{experiment_loc}/logistic_regression_model.pkl")
kmeans = joblib.load(f"{experiment_loc}/kmeans.pkl")
with open(f"{experiment_loc}/contains_zero.txt", "r") as f:
    contains_zero = f.read().splitlines()
contains_zero = ["g"+g for g in contains_zero]

# %%

for clas, img_names in images_for_inference.items():
    for img_name in img_names:
        img_name_only = img_name.split(".")[0]
        img_file = f"{images_loc}/{clas}/{img_name}"
        feats = {name: [] for name in classes}
        connected_subgraphs = {name: [] for name in classes}
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

        img = load_image(img_file, [img_size, img_size], trim=trim)
        img = img/255

        coefs = {name:0 for name in graphlet_names}
        for i, key in enumerate(coefs.keys()):
            coefs[key] = model.coef_[0,i]
        coefs_plane = np.zeros(img.shape)
        nodes_attrs = []
        
        temp_df = pd.DataFrame(0, index=[0], columns=nodes_feature_names)
        pred_df4 = pd.DataFrame(0, index=[0], columns = graphlet_names)
        for node, data in graph.nodes(data=True):
            for feature_name in nodes_feature_names:
                temp_df.loc[int(node), feature_name] = data[feature_name]
            temp_df.loc[int(node), "intensity"] = data["intensity"]
            if True is True:
                temp_df.loc[int(node), "pixel_indices"] = data["pixel_indices"]
            g_num = f"g{kmeans.predict(temp_df.loc[int(node),nodes_feature_names].values.reshape(1,-1))[0]+1}"
            pred_df4[g_num] += 1
            ls_idx = [a for a in eval(data.get("pixel_indices"))] if ")," in data.get("pixel_indices") else [eval(data.get("pixel_indices"))]
            rows, cols = zip(*ls_idx)
            coefs_plane[rows, cols] = coefs[g_num] #if g_num not in contains_zero else None
        subgraph_counts = count_unique_subgraphs(graph, max_graphlet_size)
        sorted_sg = dict(sorted(subgraph_counts.items()))
        pred_df4["g2_1"] = subgraph_counts["g2"]["g2_1"]
        pred_df4 = pred_df4.loc[:, graphlet_names[1:]+["g2_1"]]
        # Extract the values in the desired order
        values = []
        for key in sorted_sg.keys():
            sub_dict = sorted_sg[key]
            for sub_key in sorted(sub_dict):
                values.append(sub_dict[sub_key])
        
        graphlets = np.array(values).reshape(-1, )
        loc = classes.index(clas)
        feats[clas] += [np.array(temp_df)]
        connected_subgraphs[clas] += [graphlets]
        
        lists_of_full = []
        # Convert the list of values to a numpy array with shape (9, 1)
        for key in feats.keys():
            sublist_descriptors = [arr[:,:-1] for arr in feats[key]]
            sublist_add_features = connected_subgraphs[key]
            lists_of_full.append(process_sublist(sublist_descriptors, sublist_add_features, kmeans))
        
        # Convert the list of values to a numpy array with shape (9, 1)
        # pred_df3 = 
        pred_df3 = lists_of_full[1][0] if len(lists_of_full[1])!=0 else lists_of_full[0][0]
        yhat = model.predict(pred_df3.reshape(1,-1))
        norm = TwoSlopeNorm(vmin=np.min(model.coef_), vcenter=0, vmax=np.max(model.coef_))
        plt.figure()
        plt.imshow(img, 'gray')
        plt.imshow(
            coefs_plane, alpha=0.5,
            cmap = 'bwr',
            norm=norm
        )
        plt.colorbar()
        plt.axis("off")
        plt.show()
        plt.savefig(f"{results_location}/{img_name_only}_heatmap.png")

        print(f"Predicted class: {classes[yhat[0]]}({yhat[0]})")
        print(f"Actual class : {clas}")
# %%
