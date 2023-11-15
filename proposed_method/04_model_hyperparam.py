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
from config import experiment_loc, classes
from utils import load_data_from_npy
warnings.simplefilter(action='ignore', category=Warning)

#%%

loaded_feats = {key: load_data_from_npy(experiment_loc, key, "train") for key in classes}
loaded_subgraphs = {}

for key in classes:
    # Load training subgraphs for the key
    with open(f"{experiment_loc}/{key}_train_subgraphs.pkl", 'rb') as f:
        train_subgraphs = pickle.load(f)

    # Combine train and test subgraphs for the key
    loaded_subgraphs[key] = train_subgraphs

# %%


flattened_feats = [arr[:,:-1] for key in loaded_feats for arr in loaded_feats[key]]
all_descriptors = np.vstack(flattened_feats)
b_acc = 0

for num_clusters in tqdm(range(10, 110, 10)): # Best seems to be 80
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)

    lists_of_full = []

    for key in loaded_feats.keys():
        sublist_descriptors = [arr[:,:-1] for arr in loaded_feats[key]]
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

# %%
import joblib
b_n=50
kmeans = KMeans(n_clusters=b_n, random_state=0).fit(all_descriptors)
lists_of_full = []

for key in loaded_feats.keys():
    sublist_descriptors = [arr[:,:-1] for arr in loaded_feats[key]]
    sublist_add_features = loaded_subgraphs[key]
    lists_of_full.append(process_sublist(sublist_descriptors, sublist_add_features, kmeans))

X = []
y = []

for class_index, sublist in enumerate(lists_of_full):
    X.extend(sublist)
    y.extend([class_index] * len(sublist))
clf = LogisticRegression(max_iter=1000, solver='liblinear', penalty="l2")
clf.fit(X, y)

test_feats = {key: load_data_from_npy(experiment_loc, key, "test") for key in classes}
test_subgraphs = {}

for key in classes:
    # Load training subgraphs for the key
    with open(f"{experiment_loc}/{key}_test_subgraphs.pkl", 'rb') as f:
        temp_subgraphs = pickle.load(f)

    # Combine train and test subgraphs for the key
    test_subgraphs[key] = temp_subgraphs
lists_of_test = []
for key in test_feats.keys():
    sublist_descriptors = [arr[:,:-1] for arr in test_feats[key]]
    sublist_add_features = test_subgraphs[key]
    lists_of_test.append(process_sublist(sublist_descriptors, sublist_add_features, kmeans))

X_test = []
y_test = []
print(len(y_test))
for class_index, sublist in enumerate(lists_of_test):
    X_test.extend(sublist)
    y_test.extend([class_index] * len(sublist))
tst_acc = clf.score(X_test, y_test)
yhat = clf.predict(X_test)
print(f"Test accuracy(n words: {b_n}): {tst_acc}")
joblib.dump(clf, f'{experiment_loc}/logistic_regression_model.pkl')
joblib.dump(kmeans, f'{experiment_loc}/kmeans.pkl') 

# %%
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, yhat)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
# plt.show()
# %%
