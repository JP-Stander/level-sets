#%%
from images.utils import load_image
from level_sets.distance_calculator import calculate_min_distance_index
from level_sets.metrics import width_to_height, get_angle, compactness, elongation, perimeter
from level_sets.utils import get_fuzzy_sets, get_level_sets
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from subgraph.counter import count_unique_subgraphs, _reference_subgraphs
from tqdm import tqdm
import cv2
import networkx as nx
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
import xgboost as xgb
warnings.simplefilter(action='ignore', category=Warning)

# %%
def get_metrics(data):
    pixels = [a for a in eval(data.get("pixel_indices"))] if ")," in data.get("pixel_indices") else [eval(data.get("pixel_indices"))]
    img_size = max(max(a[0] for a in pixels), max(a[1] for a in pixels))
    level_set = np.zeros((img_size+1, img_size+1))
    rows, cols = zip(*pixels)
    level_set[rows, cols] = 1
    level_set=level_set.astype(int)
    regions = regionprops(level_set)

    perim = perimeter(level_set)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        extent = region.area / ((maxr - minr) * (maxc - minc))
        convex_image = convex_hull_image(region.image)
        convex_perimeter = np.sum(convex_image)  # Approximation
        convexity = perim / convex_perimeter
    # comp = compactness(level_set)
    # eln = elongation(level_set)

    return extent, convexity #, eln, comp

def image_to_histogram(descriptors, kmeans):
    hist = np.zeros(num_clusters)
    labels = kmeans.predict(descriptors)
    for label in labels:
        hist[label] += 1
    return hist


def process_sublist(sublist_descriptors, sublist_add_features, kmeans):
    # Convert each descriptor to histogram
    histograms = [image_to_histogram(desc, kmeans) for desc in sublist_descriptors]
    
    # Concatenate histograms with the additional features
    full = [np.concatenate((hist, add_feat)) for hist, add_feat in zip(histograms, sublist_add_features)]
    
    return full

# %%
# Start modelling
folders = ["dotted", "fibrous"]#os.listdir("../graphical_models_full/fuzzy_sets_10")[:10] #os.listdir("../graphical_models_full/fuzzy_sets_10")
graph_files = [f"../graphical_models_full/fuzzy_sets_10/{folder}/{dir}" for folder in folders for dir in os.listdir(f"../graphical_models_full/fuzzy_sets_10/{folder}")]
# graph_files = graph_files[:10]+graph_files[-10:]
features_names = ['compactness', 'elongation', 'extent', 'convexity', 'intensity', 'pixel_indices']
node_counts = pd.DataFrame(np.zeros((len(graph_files),len(features_names))), columns=features_names)

# Read graphs
feats = [[] for _ in folders]
connected_subgraphs = [[] for _ in folders]

for i, file in enumerate(tqdm(graph_files)):
    graph = nx.read_graphml(file)
    clas = file.split('/')[-2]
    temp_df = pd.DataFrame(np.zeros((len(graph.nodes()),len(features_names))), columns=features_names)
    for node, data in graph.nodes(data=True):
        temp_df.loc[int(node), 'compactness'] = data['compactness']
        temp_df.loc[int(node), 'elongation'] = data['elongation']
        extent, convexity = get_metrics(data)
        temp_df.loc[int(node), 'extent'] = extent
        temp_df.loc[int(node), 'convexity'] = convexity
        temp_df.loc[int(node), 'intensity'] = data['intensity']
    subgraph_counts = count_unique_subgraphs(graph,4)
    sorted_sg = dict(sorted(subgraph_counts.items()))

    # Extract the values in the desired order
    values = []
    for key in sorted_sg.keys():
        sub_dict = sorted_sg[key]
        for sub_key in sorted(sub_dict):
            values.append(sub_dict[sub_key])

    # Convert the list of values to a numpy array with shape (9, 1)
    graphlets = np.array(values).reshape(-1, )
    loc = folders.index(clas)
    feats[loc] += [np.array(temp_df)]
    connected_subgraphs[loc] += [graphlets]

# %%
flattened_feats = [arr for sublist in feats for arr in sublist]
all_descriptors = np.vstack(flattened_feats)

num_clusters = 10  # (best value seems to be 240)

# Step 1: Create a visual vocabulary
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)

# Step 2: Represent each image as a histogram of visual words
lists_of_full = [
    process_sublist(
        sublist_descriptors,
        sublist_add_features,
        kmeans
    ) for sublist_descriptors, sublist_add_features in zip(
        feats,
        connected_subgraphs
    )
]

X = []
y = []

for class_index, sublist in enumerate(lists_of_full):
    X.extend(sublist)
    y.extend([class_index] * len(sublist))

# Step 3: Train classifiers

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000)
}
accs = {
    "Logistic Regression": []
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        # print("Test accuracy:", clf.score(X_test, y_test))
        accs[name].append(clf.score(X_test, y_test))
print([np.mean(accs[name]) for name in accs.keys()])
# %%

import statsmodels.api as sm
feat_names = [f'g{i+1}' for i in range(10)] + \
    ["g2_1"] + \
    [f"g3_{i+1}" for i in range(2)] + \
    [f"g4_{i+1}" for i in range(6)]

X_lr = [X[i].reshape(-1,1) for i in range(len(X))]
X_lr = np.concatenate(X_lr, axis=1)
X_lr = pd.DataFrame(X_lr.transpose(), columns=feat_names)
# X_lr = X_lr.iloc[:, :10]
# X_lr.iloc[:, :10]
sum_first_10 = X_lr.iloc[:, :10].sum(axis=1)
sum_last = X_lr.iloc[:, -9:].sum(axis=1)
X_lr.iloc[:, :10] = X_lr.iloc[:, :10].div(sum_first_10, axis=0)
X_lr.iloc[:, -9:] = X_lr.iloc[:, -9:].div(sum_last, axis=0)
# X_lr = X_lr*100
X_lr = sm.add_constant(X_lr)
y_lr = [i for i in y]

X_train, X_test = X_lr.iloc[train_index], X_lr.iloc[test_index]
y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
model = sm.Logit(y_train, X_train)
result = model.fit()
print(result.summary())
y_hat = (result.predict(X_test) > 0.5).astype(int)
print(np.mean(y_hat==y_test))
# print(result.summary().as_latex())
# %%

coefficients = result.params
sorted_by_absolute = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
top_5_original_values = sorted_by_absolute
print(top_5_original_values)
# %%
R_coefs = """
g1           0.076019   0.035216   2.159 0.030877 *  
g2           0.272850   0.077669   3.513 0.000443 ***
g3          -0.014538   0.055085  -0.264 0.791844    
g4           0.065788   0.045898   1.433 0.151761    
g5           0.084504   0.035167   2.403 0.016264 *  
g6          -0.041302   0.060590  -0.682 0.495446    
g7           0.072389   0.035259   2.053 0.040066 *  
g8           0.071354   0.067717   1.054 0.292021    
g9           0.374901   0.082935   4.520 6.17e-06 ***
g10          0.126994   0.056859   2.234 0.025516 *  
g2_1        -0.060840   0.020709  -2.938 0.003305 ** 
g3_1         0.009840   0.005617   1.752 0.079790 .  
g3_2         0.019390   0.007253   2.673 0.007512 ** 
g4_1        -0.008785   0.003485  -2.521 0.011714 *  
g4_2        -0.002084   0.002011  -1.037 0.299961    
g4_3        -0.002122   0.009617  -0.221 0.825323    
g4_4         0.008531   0.005182   1.646 0.099678 .  
g4_5         0.002310   0.003980   0.580 0.561670    
g4_6        -0.003001   0.001485  -2.021 0.043251 * 
"""

# Create dictionary from data
lines = R_coefs.strip().split('\n')
coefs = {}
for line in lines:
    parts = line.split()
    variable_name = parts[0]
    coefficient = float(parts[1])
    coefs[variable_name] = coefficient
# %%
def get_metrics(level_set):
    level_set=level_set.astype(int)
    regions = regionprops(level_set)

    perim = perimeter(level_set)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        extent = region.area / ((maxr - minr) * (maxc - minc))
        convex_image = convex_hull_image(region.image)
        convex_perimeter = np.sum(convex_image)  # Approximation
        convexity = perim / convex_perimeter
    comp = compactness(level_set)
    eln = elongation(level_set)

    return extent, convexity , eln, comp

from images.utils import load_image
from level_sets.utils import get_fuzzy_sets
from level_sets.metrics import compactness, elongation, width_to_height, get_angle
img_name = "dotted_0038"
img_name = "fibrous_0121"
img_file = f"../dtd/images/{img_name.split('_')[0]}/{img_name}.jpg"
img_graph = f"../graphical_models_old/fuzzy_sets_10/{img_name}_graph.graphml"

img = load_image(img_file, [50,50])
img = img/255
graph = nx.read_graphml(img_graph)

level_sets = get_fuzzy_sets(img, 10/255, 8)
uls = pd.unique(level_sets.flatten())

coefs_plane = np.zeros(img.shape)
# coefs = result.params.to_dict()
pred_df = pd.DataFrame(0, index=[0], columns=list(X_lr))

for ls in uls:
    set = np.zeros(img.shape)
    set[level_sets==ls] = 1

    set_attr = pd.DataFrame(0, index = [0], columns=["compactness",'elongation','extent','convexity','intensity'])
    extent, convexity, eln, cmp = get_metrics(set)
    set_attr.loc[0, 'compactness'] = cmp
    set_attr.loc[0, 'elongation'] = eln
    set_attr.loc[0, 'extent'] = extent
    set_attr.loc[0, 'convexity'] = convexity
    set_attr.loc[0, 'intensity'] = img[level_sets==ls].mean()
    min_idx = kmeans.predict(set_attr)
    coefs_plane[level_sets==ls] = coefs[f"g{min_idx[0]+1}"]
    pred_df.loc[0, f"g{min_idx[0]+1}"] += 1

subgraph_counts = count_unique_subgraphs(graph,4)
sorted_sg = dict(sorted(subgraph_counts.items()))

# Extract the values in the desired order
values = []
for key in sorted_sg.keys():
    sub_dict = sorted_sg[key]
    for sub_key in sorted(sub_dict):
        values.append(sub_dict[sub_key])

# Convert the list of values to a numpy array with shape (9, 1)
# pred_df.iloc[0,-9:] = np.array(values).reshape(-1, )
yhat = (result.predict(pred_df)[0] > 0.5).astype(int)
plt.figure()
plt.imshow(img, 'gray')
plt.imshow(
    coefs_plane, alpha=0.5, 
    vmin=np.min([a for a in coefs.values()]), 
    vmax=np.max([a for a in coefs.values()])
)
plt.colorbar()
plt.axis("off")
plt.savefig(f"../paper3_results/{img_name}_heatmap.png")

print(f"Predicted class: {yhat}")
# %%
