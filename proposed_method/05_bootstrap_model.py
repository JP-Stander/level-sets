# %%
from utils import process_sublist, load_data_from_npy
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from config import num_bootstrap_iterations, experiment_loc, num_clusters, classes
import joblib

# %%

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

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)

lists_of_full = []

for key in loaded_feats.keys():
    sublist_descriptors = [arr[:,:-1] for arr in loaded_feats[key]]
    sublist_add_features = loaded_subgraphs[key]
    lists_of_full.append(process_sublist(sublist_descriptors, sublist_add_features, kmeans))

X = []
y = []
y_label = []

for class_index, sublist in enumerate(lists_of_full):
    X.extend(sublist)
    y.extend([class_index] * len(sublist))
    y_label.extend([classes[class_index]] * len(sublist))


coef_samples = np.zeros((num_bootstrap_iterations, X[0].shape[0]))
intercept_samples = []
for i in tqdm(range(num_bootstrap_iterations)):
    # Resample the training data with replacement
    X_resampled, y_resampled = resample(X, y, replace=True)
    
    # Fit a logistic regression model on the resampled data
    clf = LogisticRegression(max_iter=1000, solver='liblinear', penalty="l2")
    clf.fit(X_resampled, y_resampled)
    
    # Store the coefficients of the model
    coef_samples[i] =  clf.coef_
    intercept_samples.append(clf.intercept_[0])

confidence_intervals = np.percentile(coef_samples, [5, 95], axis=0)
contains_zero = []
coefficients = np.zeros((1, X[0].shape[0]))
for i, (lower, upper) in enumerate(zip(confidence_intervals[0], confidence_intervals[1])):
    if not (lower <= 0 <= upper):
        contains_zero.append(i)
        coefficients[:, i] = np.mean(coef_samples[:,i])

# Calculate mean and 95% confidence interval for each coefficient
conf_int = np.percentile(coef_samples, [2.5, 97.5], axis=0)
print("Coefficients")
print(conf_int.transpose())

conf_int_int = np.percentile(intercept_samples, [2.5, 97.5], axis=0)
if conf_int_int[0] <= 0 <= conf_int_int[1]:
    intercept = 0
else:
    intercept = np.mean(intercept_samples)
print("Intercepts")
print(intercept)

# %%

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
    sublist_add_features = loaded_subgraphs[key]
    lists_of_test.append(process_sublist(sublist_descriptors, sublist_add_features, kmeans))

X_test = []
y_test = []

for class_index, sublist in enumerate(lists_of_test):
    X_test.extend(sublist)
    y_test.extend([class_index] * len(sublist))
tst_acc = clf.score(X_test, y_test)
print(f"Test accuracy: {tst_acc}")

# clf.coef_ = coefficients
# clf.intercept_ = np.array([intercept])
print(f"Model test accuracy: {clf.score(X_test, y_test)}")
joblib.dump(clf, f'{experiment_loc}/logistic_regression_model.pkl')
joblib.dump(kmeans, f'{experiment_loc}/kmeans.pkl') 

with open(f"{experiment_loc}/contains_zero.txt", "w") as f:
    for item in contains_zero:
        f.write("%s\n" % item)

# %%
