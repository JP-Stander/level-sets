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
from config import classes, images_loc, img_size, fs_connectivity, fs_delta, sets_feature_names, trim
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image
from level_sets.metrics import get_metrics
from level_sets.utils import get_fuzzy_sets, get_level_sets, intersection_over_union

# %%

images = []
for clas in classes:
    images += [f"{images_loc}/{clas}/" + file for file in os.listdir(f"{images_loc}/{clas}")]

# Initialize metrics dictionary
metrics = {clas: [] for clas in classes}

for image in tqdm(images):
    img = load_image(image, [img_size, img_size], trim=trim)
    # level_sets = get_level_sets(img)
    level_sets = get_fuzzy_sets(img, 0, fs_connectivity)
    uni_level_sets = pd.unique(level_sets.flatten())
    for i, ls in enumerate(uni_level_sets):
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        level_set = np.array(level_sets == ls)
        set_value = np.mean([img[s] for s in subset])
        # if len(subset) <= 2:
        #     continue
        metric_results = get_metrics(level_set.astype(int), img_size=[img_size, img_size])
        clas = image.split("/")[-2]
        metrics[clas] += [metric_results]

# Create dataframe
df = pd.DataFrame([(clas, metric, value)
                for clas, metrics_list in metrics.items() 
                for metrics_dict in metrics_list 
                for metric, value in metrics_dict.items()],
            columns=["Class", "Metric", "Value"]
            )
# %% Now, you can plot using seaborn
for metric in ["compactness", "elongation", "extent", "convexity"]:
    plt.figure()
    for clas in classes:
        subset = df.loc[(df["Class"] == clas) & (df["Metric"] == metric), "Value"]
        sns.kdeplot(subset, label=clas, fill=True)
    plt.title(f'Distribution of {metric} across classes')
    plt.legend()
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
comparer = "iou"
for metric_name in pd.unique(df["Metric"]):
    values1 = df.loc[(df["Class"] == classes[0]) & (df["Metric"] == metric_name), "Value"]
    values2 = df.loc[(df["Class"] == classes[1]) & (df["Metric"] == metric_name), "Value"]

    if comparer == "iou":
        ks_stat = intersection_over_union(values1, values2)
    else:
        ks_stat, p_value = ks_2samp(values1, values2)

    table_data.append((metric_name, ks_stat))

# Sort the table data by KS statistics in descending order
if comparer == "iou":
    table_data.sort(key=lambda x: x[1], reverse=False)
else:
    table_data.sort(key=lambda x: x[1], reverse=True)
print(pd.DataFrame(table_data))

# %%
df_metrics = pd.DataFrame(table_data)
df_metrics = df_metrics.loc[df_metrics[0] != "bbox_area", :]
df_metrics = df_metrics.loc[df_metrics[0] != "area", :]
print(df_metrics.apply(lambda row: f'\\textbf{{{row[0]}}} & {row[1]:.4f} \\\\', axis=1).str.cat(sep='\n'))

# %%
from utils import get_img_nea, make_graph
import networkx as nx
import numpy as np
image = '../../colab_alisa/asthma/A1_PRP+T_40X_03.tif'#images[0]
sets_feature_names = [
    'compactness',
    'elongation',
    'angle',
    'area'
]
n,e,a = get_img_nea(image, d=10, img_size=100, connectivity=fs_connectivity, metric_names=sets_feature_names, trim=trim)

# d=10
# img_size=100
# connectivity=fs_connectivity
# metric_names=sets_feature_names
# trim=trim
#%%
delta = 0.25
print(f"Number of edges greater than {delta}: {np.sum(e.flatten() > delta)/2}")
print(f"Proportion of edges greater than {delta}: {np.mean(e.flatten() > delta)/2}")
#%%
g = make_graph(n, e, a, 0.25)
nx.draw_networkx(
    g,
    with_labels=False,
    edge_color = "gray",
    node_size = 2,
)
# %%
from config import graphs_location
import networkx as nx
import ast
graph_files = [f"{graphs_location}/{clas}/{file}" for clas in classes for file in os.listdir(f"{graphs_location}/{clas}")]
graph = nx.read_graphml(graph_files[0])
image_var = np.zeros((100,100))
def replace_values_in_image(image, data):
    # Convert string representation of pixel indices into a list of tuples
    pixel_indices = [a for a in eval(data.get("pixel_indices"))] if ")," in data.get("pixel_indices") else [eval(data.get("pixel_indices"))]

    for index in pixel_indices:
        image[index] = data["level-set"]

for node, data in gr.nodes(data=True):
    replace_values_in_image(image_var, data)

plt.figure()
plt.imshow(image_var)
plt.show()

#%%
img = load_image('../../colab_alisa/control/C1510_PPP_T_20K_02.tif', [100,100], trim)
return_spp=True
set_type="fuzzy"
fuzzy_cutoff=10
normalise_pixel_index=False
connectivity=8
alpha=0.5
normalise_gray=True
size_proportion=False
ls_spatial_dist="euclidean"
ls_attr_dist="cityblock"
centroid_method="mean"
metric_names = sets_feature_names
# %%

image = images[80]

img = load_image(image, [img_size, img_size], trim=trim)
level_sets = get_fuzzy_sets(img, 15, fs_connectivity)

# Create a figure
plt.figure(figsize=(10, 5))

# First subplot for image1
plt.subplot(1, 2, 1)  # (1 row, 2 columns, first subplot)
plt.imshow(img, cmap='gray')
plt.title('Image')
plt.axis('off')  # to hide axes

# Second subplot for image2
plt.subplot(1, 2, 2)  # (1 row, 2 columns, second subplot)
plt.imshow(level_sets)
plt.title('Level-sets')
plt.axis('off')  # to hide axes

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
# %%
image = images[88]

img = load_image(image, [img_size, img_size], trim=trim)
level_sets = get_fuzzy_sets(img, 10, fs_connectivity)
for ls in pd.unique(level_sets.flatten()):
    set_values = img[level_sets==ls]
    # if set_values.mean() < 56:
    #     level_sets[level_sets==ls] = None
plt.figure()
plt.imshow(img, 'gray')
plt.imshow(
    level_sets, alpha=0.5
)
plt.colorbar()
plt.axis("off")
plt.show()
# %%
pd.value_counts(level_sets.flatten())
# %%
