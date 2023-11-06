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
from level_sets.utils import get_fuzzy_sets
from level_sets.metrics import get_metrics
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


predict_class_metrics = {i: {metric: [] for metric in sets_feature_names} for i in classes}
for clas, img_names in images_for_inference.items():
    for img_name in img_names:
        img_name_only = img_name.split(".")[0]
        img_file = f"{images_loc}/{clas}/{img_name}"
        img = load_image(img_file, [img_size, img_size], trim=trim)
        level_sets = get_fuzzy_sets(img, fs_delta, fs_connectivity)
        uni_level_sets = pd.unique(level_sets.flatten())
        for i, ls in enumerate(uni_level_sets):
            subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
            level_set = np.array(level_sets == ls)
            set_value = np.mean([img[s] for s in subset])
            if set_value < 56:
                continue
            metric_results = get_metrics(level_set.astype(int), img_size=[img_size, img_size], metric_names=sets_feature_names)
            for metric in sets_feature_names:
                predict_class_metrics[clas][metric] += [metric_results[metric]]
# %%
# asthma is class 0
# control is class 1
import seaborn as sns
for metric in sets_feature_names:
    plt.figure()
    sns.kdeplot(data=predict_class_metrics[classes[0]][metric], label=classes[0])
    sns.kdeplot(data=predict_class_metrics[classes[1]][metric], label=classes[1])
    plt.legend()
    plt.title(f"Distribution of {metric}")
    # plt.savefig(f"{experiment_loc}/comparetive_distplot_{metric}.png")

# %%
