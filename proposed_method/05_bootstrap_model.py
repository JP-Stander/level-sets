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
from config import num_bootstrap_iterations, experiment_loc, num_clusters, classes, sets_feature_names, trim
import joblib
import sys
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from level_sets.metrics import get_metrics

# %%
def create_line(shape, angle_degree):

    array = np.zeros(shape)
    center = np.array(shape) // 2
    if angle_degree == 90 or angle_degree == -90:
        array[:, center[1]] = 1
        return array
    angle_rad = np.radians(angle_degree)
    m = np.tan(angle_rad)
    b = center[1] - m * center[0]
    for x in range(shape[0]):
        y = int(m * x + b)
        if 0 <= y < shape[1]:
            array[x, y] = 1
    return array

def create_ball(shape, radius):
    ball = np.zeros(shape)
    center = np.array(ball.shape) // 2
    for x in range(ball.shape[0]):
        for y in range(ball.shape[1]):
            distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if distance < radius:
                ball[x, y] = 1
    return ball
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

kmeans1 = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
kmeans2 = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)

ball = create_ball((20, 20), 7)
ball_metrics = get_metrics(ball, ball.shape, metric_names = sets_feature_names)

line1 = create_line((20, 20), 45)
line_metrics1 = get_metrics(line1, line1.shape, metric_names = sets_feature_names)

line2 = create_line((20, 20), 0)
line_metrics2 = get_metrics(line2, line2.shape, metric_names = sets_feature_names)

centroids = kmeans2.cluster_centers_
new_centroids = np.vstack([np.append(np.array(list(metric.values())), i) for i in [0,1] for metric in [ball_metrics, line_metrics1, line_metrics2]])

all_centroids = np.vstack([centroids, new_centroids])

# Modify the KMeans object
kmeans2.cluster_centers_ = all_centroids
kmeans2.n_clusters = len(all_centroids)

labels1 = kmeans1.predict(all_descriptors)
labels2 = kmeans2.predict(all_descriptors)

#%%
kmeans = kmeans2

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
# print(f"Model test accuracy: {clf.score(X_test, y_test)}")
# joblib.dump(clf, f'{experiment_loc}/logistic_regression_model.pkl')
# joblib.dump(kmeans, f'{experiment_loc}/kmeans.pkl') 

# with open(f"{experiment_loc}/contains_zero.txt", "w") as f:
#     for item in contains_zero:
#         f.write("%s\n" % item)


# %%
