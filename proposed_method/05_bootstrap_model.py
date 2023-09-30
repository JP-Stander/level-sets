# %%
from utils import process_sublist
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from config import num_bootstrap_iterations, experiment_loc
import joblib

# %%
# Load the features dictionary
with open(f"{experiment_loc}/feats.pkl", 'rb') as f:
    loaded_feats = pickle.load(f)

for key in loaded_feats:
    loaded_feats[key] = [np.array(arr) for arr in loaded_feats[key]]

# Load the subgraphs dictionary
with open(f"{experiment_loc}/subgraphs.pkl", 'rb') as f:
    loaded_subgraphs = pickle.load(f)

for key in loaded_subgraphs:
    loaded_subgraphs[key] = [np.array(arr) for arr in loaded_subgraphs[key]]

# %%

flattened_feats = [arr for key in loaded_feats for arr in loaded_feats[key]]
all_descriptors = np.vstack(flattened_feats)

num_clusters = 80
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
final_model = clf

final_model.coef_ = coefficients
final_model.intercept_ = np.array([intercept])
joblib.dump(final_model, f'{experiment_loc}/logistic_regression_model.pkl')
joblib.dump(kmeans, f'{experiment_loc}/kmeans.pkl') 


# %%
