#%% Imports
from images.utils import load_image
from level_sets.distance_calculator import calculate_min_distance_index
from level_sets.metrics import width_to_height, get_angle, compactness, elongation, perimeter, area
from level_sets.utils import get_fuzzy_sets, get_level_sets
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.utils import resample
from scipy.stats import percentileofscore
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

# %% Define functions
def get_metrics_from_indices(data):
    pixels = [a for a in eval(data.get("pixel_indices"))] if ")," in data.get("pixel_indices") else [eval(data.get("pixel_indices"))]
    img_size = max(max(a[0] for a in pixels), max(a[1] for a in pixels))
    level_set = np.zeros((img_size+1, img_size+1))
    rows, cols = zip(*pixels)
    level_set[rows, cols] = 1
    level_set=level_set.astype(int)
    regions = regionprops(level_set)

    ar = area(level_set)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        extent = region.area / ((maxr - minr) * (maxc - minc))
        convex_image = convex_hull_image(region.image)
        convex_area = np.sum(convex_image)
        convexity = convex_area / ar
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

def get_metrics(level_set):
    level_set=level_set.astype(int)
    regions = regionprops(level_set)

    ar = area(level_set)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        extent = region.area / ((maxr - minr) * (maxc - minc))
        convex_image = convex_hull_image(region.image)
        convex_area = np.sum(convex_image)
        convexity = convex_area / ar
    eln = elongation(level_set)
    comp = compactness(level_set)

    return extent, convexity , eln, comp

# %% Build dataset
graph_location = "../graphical_models_100/fuzzy_sets_10"
folders = ["dotted", "fibrous"]
graph_files = [f"{graph_location}/{folder}/{dir}" for folder in folders for dir in os.listdir(f"{graph_location}/{folder}")]
features_names = ['compactness', 'elongation', 'extent', 'convexity', 'intensity']#, 'pixel_indices']
node_counts = pd.DataFrame(np.zeros((len(graph_files),len(features_names))), columns=features_names)

# Read graphs
feats = [[] for _ in folders]
connected_subgraphs = [[] for _ in folders]
print(f"dotted is class {folders.index('dotted')}")
print(f"fibrous is class {folders.index('fibrous')}")
for i, file in enumerate(tqdm(graph_files)):
    graph = nx.read_graphml(file)
    clas = file.split('/')[-2]
    temp_df = pd.DataFrame(np.zeros((len(graph.nodes()),len(features_names))), columns=features_names)
    for node, data in graph.nodes(data=True):
        temp_df.loc[int(node), 'compactness'] = data['compactness']
        temp_df.loc[int(node), 'elongation'] = data['elongation']
        extent, convexity = get_metrics_from_indices(data)
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

# %% Merge datasets
flattened_feats = [arr for sublist in feats for arr in sublist]
all_descriptors = np.vstack(flattened_feats)
# all_descriptors = all_descriptors[:,:-1]
b_acc = 0

num_clusters = 80  # (best value seems to be 80)
# for num_clusters in tqdm(range(10,250,10)):
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

# % sklearn regression model

clf = LogisticRegression(max_iter=1000, solver='liblinear', penalty="l2")#,penalty=None)

accs = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
    clf.fit(X_train, y_train)
    # print("Test accuracy:", clf.score(X_test, y_test))
    accs.append(clf.score(X_test, y_test))
acc = np.mean(accs)
# if acc > b_acc:
#     b_acc = acc
#     b_n = num_clusters
#     print(f"Best accuracy: {b_acc} for n {b_n}")

#%% Bootstrapping

num_bootstrap_iterations = 1000 # You can adjust this as needed
coef_samples = np.zeros((num_bootstrap_iterations, X[0].shape[0]))
for i in tqdm(range(num_bootstrap_iterations)):
    # Resample the training data with replacement
    X_resampled, y_resampled = resample(X, y, replace=True)
    
    # Fit a logistic regression model on the resampled data
    clf = LogisticRegression(max_iter=1000, solver='liblinear', penalty="l2")
    clf.fit(X_resampled, y_resampled)
    
    # Store the coefficients of the model
    coef_samples[i] = clf.coef_
# %%
confidence_intervals = np.percentile(coef_samples, [5, 95], axis=0)
contains_zero = []
for i, (lower, upper) in enumerate(zip(confidence_intervals[0], confidence_intervals[1])):
    if lower <= 0 <= upper:
        contains_zero.append(i)
coefficients = confidence_intervals[:, :10]

# Calculate mean and 95% confidence interval for each coefficient
means = np.mean(coefficients, axis=0)
conf_int = np.percentile(coefficients, [2.5, 97.5], axis=0)

# Create LaTeX table header
latex_table = "\\begin{tabular}{ccc}\n"
latex_table += "Coefficient & Mean & 95\\% Confidence Interval \\\\\n"
latex_table += "\\hline\n"

# Add rows for each coefficient
for i in range(10):
    coeff_name = f"g_{i + 1}"
    mean_val = means[i]
    conf_int_low, conf_int_high = conf_int[:, i]
    
    # Format values to a reasonable number of decimal places
    mean_val_str = f"{mean_val:.4f}"
    conf_int_str = f"({conf_int_low:.4f}, {conf_int_high:.4f})"
    
    # Add the row to the LaTeX table
    latex_table += f"{coeff_name} & {mean_val_str} & {conf_int_str} \\\\\n"

# Close the LaTeX table
latex_table += "\\end{tabular}"

# Print or save the LaTeX table
print(latex_table)
#%% Fit final model
clf = LogisticRegression(max_iter=1000, solver='liblinear', penalty="l2")
clf.fit(X,y)
# %% visualise
img_name = "dotted_0111" #["dotted_0188", "dotted_0111", ""dotted_0180""]
# img_name = "fibrous_0116"#["fibrous_0191", "fibrous_0108", "fibrous_0116"]
feat_names = [f'g{i+1}' for i in range(num_clusters)] + \
    ["g2_1"] + \
    [f"g3_{i+1}" for i in range(2)] + \
    [f"g4_{i+1}" for i in range(6)]
for img_name in ["dotted_0188", "dotted_0111", "dotted_0180", "fibrous_0191", "fibrous_0108", "fibrous_0116"]:
    img_name = img_name.split(".")[0]
    folder = img_name.split('_')[0]
    img_file = f"../dtd/images/{folder}/{img_name}.jpg"
    img_graph = f"{graph_location}/{folder}/{img_name}_graph.graphml"

    img = load_image(img_file, [100,100])
    img = img/255
    graph = nx.read_graphml(img_graph)

    level_sets = get_fuzzy_sets(img, 10/255, 8)
    uls = pd.unique(level_sets.flatten())
    coefs = {name:0 for name in feat_names}
    for i, key in enumerate(coefs.keys()):
        coefs[key] = clf.coef_[0,i]
    coefs_plane = np.zeros(img.shape)
    pred_df = pd.DataFrame(0, index=[0], columns=feat_names)

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
        coefs_plane[level_sets==ls] = coefs[f"g{min_idx[0]+1}"] if min_idx[0] not in contains_zero else None
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
    pred_df.iloc[0,-9:] = np.array(values).reshape(-1, )
    yhat = clf.predict(pred_df)

    plt.figure()
    plt.imshow(img, 'gray')
    plt.imshow(
        coefs_plane, alpha=0.5, 
        vmin=np.min(clf.coef_),#np.min([a for a in coefs.values()]), 
        vmax=np.max(clf.coef_)#np.max([a for a in coefs.values()])
    )
    plt.colorbar()
    plt.axis("off")
    plt.savefig(f"../paper3_results/{img_name}_heatmap.png")

    print(f"Predicted class: {'dotted(0)' if yhat[0] == 0 else 'firbous(1)'}")
    print(f"Actual class : {folder}")
# %%
print(f"dotted is class {folders.index('dotted')}")
print(f"fibrous is class {folders.index('fibrous')}")
# %%
