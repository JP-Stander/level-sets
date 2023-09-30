# %%
from utils import process_sublist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import warnings
from config import experiment_loc
warnings.simplefilter(action='ignore', category=Warning)

# %%

# Load the features dictionary
with open(f"{experiment_loc}/feats.pkl", 'rb') as f:
    loaded_feats = pickle.load(f)

for key in loaded_feats:
    loaded_feats[key] = [np.array(arr) for arr in loaded_feats[key]]

# Load the subgraphs dictionary
with open(f"{experiment_loc}/subgraphs.pkl", 'rb') as f:
    loaded_subgraphs = pickle.load(f)

for key in loaded_subgraphs: # reshape is quationable
    loaded_subgraphs[key] = [np.array(arr) for arr in loaded_subgraphs[key]]

# %%

flattened_feats = [arr for key in loaded_feats for arr in loaded_feats[key]]
all_descriptors = np.vstack(flattened_feats)
b_acc = 0

for num_clusters in tqdm(range(10, 20, 10)): # Best seems to be 80
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)

    lists_of_full = []

    for key in loaded_feats.keys():
        sublist_descriptors = loaded_feats[key]
        sublist_add_features = loaded_subgraphs[key]
        lists_of_full.append(process_sublist(sublist_descriptors, sublist_add_features, kmeans))

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
    if acc > b_acc:
        b_acc = acc
        b_n = num_clusters
print(f"Best performance for {num_clusters} words with 5-fold average accuracy of {b_acc}")
