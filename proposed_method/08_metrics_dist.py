# %%
import os
import sys
import joblib
import pandas as pd
import numpy as np
from utils import process_sublist
from matplotlib import pyplot as plt
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from subgraph.counter import count_unique_subgraphs
from images.utils import load_image
from utils import img_to_graph
from config import sets_feature_names, images_loc, \
        experiment_loc, graphlet_names, \
        fs_delta, img_size, edge_delta, fs_connectivity, max_graphlet_size, trim, \
        images_for_inference, classes, nodes_feature_names
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

predict_class_metrics = {i: {sign: {metric: [] for metric in sets_feature_names} for sign in ['neg', 'pos', 'zero']} for i in [0,1]}
words_counts = {i: {j :[] for j in ["neg", "pos"]} for i in [0,1]}
for clas, img_names in images_for_inference.items():
    g_nums=[]
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
        nodes_attrs = []
        pos_metrics = {metric: [] for metric in sets_feature_names}
        neg_metrics = {metric: [] for metric in sets_feature_names}
        zero_metrics = {metric: [] for metric in sets_feature_names}
        n_word_count = 0
        p_word_count = 0
        temp_df = pd.DataFrame(0, index=[0], columns=nodes_feature_names)
        pred_df4 = pd.DataFrame(0, index=[0], columns = graphlet_names)
        for node, data in graph.nodes(data=True):
            for feature_name in nodes_feature_names:
                temp_df.loc[int(node), feature_name] = data[feature_name]
            temp_df.loc[int(node), "intensity"] = data["intensity"]
            temp_df.loc[int(node), "pixel_indices"] = data["pixel_indices"]
            g_num = f"g{kmeans.predict(temp_df.loc[int(node),nodes_feature_names].values.reshape(1,-1))[0]+1}"
            g_nums += [g_num]
            pred_df4[g_num] += 1
            node_coef = coefs[g_num]
            signif = g_num not in contains_zero
            if node_coef < 0 and signif:
                n_word_count += 1
                for metric in sets_feature_names:
                    neg_metrics[metric] += [temp_df.loc[int(node), metric]]
            elif node_coef > 0 and signif:
                p_word_count += 1
                for metric in sets_feature_names:
                    pos_metrics[metric] += [temp_df.loc[int(node), metric]]
            else:
                for metric in sets_feature_names:
                    zero_metrics[metric] += [temp_df.loc[int(node), metric]]
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
        y = classes.index(clas)
        # if yhat[0]==y:
        words_counts[y]["neg"] += [n_word_count]
        words_counts[y]["pos"] += [p_word_count]
        for metric in sets_feature_names:
            if y == 1:
                predict_class_metrics[y]["pos"][metric] += pos_metrics[metric]
            if y == 0:
                predict_class_metrics[y]["neg"][metric] += neg_metrics[metric]
            predict_class_metrics[y]["zero"][metric] += zero_metrics[metric]


# %%
# asthma is class 0
# control is class 1
import seaborn as sns
for metric in predict_class_metrics[1]['neg'].keys():
    plt.figure()
    sns.kdeplot(data=predict_class_metrics[0]['neg'][metric], label=classes[0])
    sns.kdeplot(data=predict_class_metrics[1]['pos'][metric], label=classes[1])
    # sns.kdeplot(data=predict_class_metrics[0]['zero'][metric], label="zero")
    plt.legend()
    plt.title(f"Distribution of {metric}")
    plt.savefig(f"{experiment_loc}/comparetive_distplot_{metric}.png")

# %%
