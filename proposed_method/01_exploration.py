# %% Imports
import os
import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
from config import classes, images_loc, img_size, fs_connectivity, fs_delta, sets_feature_names
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image
from level_sets.metrics import get_metrics
from level_sets.utils import get_fuzzy_sets, get_level_sets

# %%

images = []
for clas in classes:
    images += [f"{images_loc}/{clas}/" + file for file in os.listdir(f"{images_loc}/{clas}")]

# Initialize metrics dictionary
metrics = {clas: [] for clas in classes}

for image in tqdm(images):
    img = load_image(image, [img_size, img_size])
    # level_sets = get_level_sets(img)
    level_sets = get_fuzzy_sets(img, fs_delta, fs_connectivity)
    uni_level_sets = pd.unique(level_sets.flatten())
    for i, ls in enumerate(uni_level_sets):
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        level_set = np.array(level_sets == ls)
        metric_results = get_metrics(level_set.astype(int))
        clas = image.split("/")[-2]
        metrics[clas] += [metric_results]

# %% Plot distributions (optional)
df = pd.DataFrame([(clas, metric, value)
                for clas, metrics_list in metrics.items() 
                for metrics_dict in metrics_list 
                for metric, value in metrics_dict.items()],
            columns=["Class", "Metric", "Value"]
            )
# Now, you can plot using seaborn
for metric in pd.unique(df["Metric"]):
    plt.figure()
    for clas in classes:
        subset = df.loc[(df["Class"] == clas) & (df["Metric"] == metric), "Value"]
        sns.kdeplot(subset, label=clas, fill=True)
    plt.title(f'Distribution of {metric} across classes')
    plt.tight_layout()
    plt.show()

# %% Mean and variance (optional
for clas, _ in metrics.items():
    for metric in pd.unique(df["Metric"]):
        print(
            f'Mean(variance) for metric {metric} for texture {clas}: ' +
            f'{df.loc[(df["Class"] == clas) & (df["Metric"] == metric), "Value"].mean()}' +
            f'({df.loc[(df["Class"] == clas) & (df["Metric"] == metric), "Value"].var()})'
            )

# %% KS-statistic
table_data = []

for metric_name in pd.unique(df["Metric"]):
    values1 = df.loc[(df["Class"] == classes[0]) & (df["Metric"] == metric_name), "Value"]
    values2 = df.loc[(df["Class"] == classes[1]) & (df["Metric"] == metric_name), "Value"]
    ks_stat, p_value = ks_2samp(values1, values2)
    table_data.append((metric_name, ks_stat))

# Sort the table data by KS statistics in descending order
table_data.sort(key=lambda x: x[1], reverse=True)
print(pd.DataFrame(table_data))


# %%
from utils import get_img_nea, make_graph
import networkx as nx
import numpy as np
image = images[0]
sets_feature_names = [
    'compactness',
    'elongation',
    'extent',
    'convexity'
]
n,e,a = get_img_nea(image, d=10, img_size=100, metric_names=sets_feature_names)

#%%
delta = 0.5
print(f"Number of edges greater than {delta}: {np.sum(e.flatten() > delta)/2}")
print(f"Proportion of edges greater than {delta}: {np.mean(e.flatten() > delta)/2}")
#%%
g = make_graph(n, e, a, 0.5)
nx.draw_networkx(
    g,
    with_labels=False,
    edge_color = "gray",
    node_size = 2,
)
# %%
