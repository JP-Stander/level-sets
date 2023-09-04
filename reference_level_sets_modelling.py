# %%
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from level_sets.distance_calculator import calculate_min_distance_index
from subgraph.counter import count_unique_subgraphs, _reference_subgraphs

# %%
graph_files = [f"../graphical_models_old/fuzzy_sets_10/{dir}" for dir in os.listdir("../graphical_models_old/level_sets")]
reference_nodes = pd.read_csv("../reference_ls/level_sets.csv")
reference_nodes = reference_nodes[['compactness','elongation','width_to_height','angle','intensity','id']]

max_subgraph_size = 5

features_names = ["class"] + list(reference_nodes["id"]) + [k for i in range(2, max_subgraph_size) for k in _reference_subgraphs[f"g{i+1}"].keys()]
node_counts = pd.DataFrame(np.zeros((len(graph_files),len(features_names))), columns=features_names)

# %%
graphs = []
for i, file in enumerate(graph_files):
    G = nx.read_graphml(file)
    graphs.append(G)
    node_counts.loc[i, "class"] = file.split("/")[-1].split("_")[0]


#%%
reference_nodes_no_id = reference_nodes[['compactness', 'elongation', 'width_to_height', 'angle', 'intensity']].values
for i, graph in enumerate(tqdm(graphs)):
    for node, data in graph.nodes(data=True):
        node_data_values = np.array([
            data['compactness'], 
            data['elongation'], 
            data['width_to_height'], 
            data['angle'], 
            data['intensity']
        ])
        
        min_idx = calculate_min_distance_index(node_data_values, reference_nodes_no_id
        )
        node_counts.loc[i, f"g{min_idx+1}"] = node_counts.loc[i, f"g{min_idx+1}"]+1
    subgraph_counts = count_unique_subgraphs(graph,4)
    subgraph_counts = pd.DataFrame(subgraph_counts).fillna(0).sum(axis=1)
    node_counts.loc[i, subgraph_counts.index] = subgraph_counts.values

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Assuming df is your dataframe

if isinstance(node_counts["class"].iloc[0], str):
    node_counts['class'] = (node_counts['class'] == 'dotted').astype(int)
# Splitting the data into training and test sets (80% train, 20% test)
X = node_counts.drop('class', axis=1)  # Features (g1 to g100)
# X = X.iloc[:,:100]
y = node_counts['class']               # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(objective="binary:logistic", random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

results = {}

# Training, predicting, and storing accuracy for each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy * 100
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    # if name == "Random Forest":
    #     break
# %%
from sklearn.ensemble import VotingClassifier

# Create a list of tuples with model name and its instance
model_list = [(name, clf) for name, clf in classifiers.items()]

# Create the VotingClassifier
voting_clf = VotingClassifier(estimators=model_list, voting='hard')

# Train the voting classifier
voting_clf.fit(X_train, y_train)

# Predict and evaluate
voting_pred = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_pred)
print(f"Voting Ensemble Accuracy: {voting_accuracy * 100:.2f}%")
# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# Initialize meta-learner (you can choose a different one as well)
meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize arrays to store predictions for each base model
meta_features_train = np.zeros((X_train.shape[0], len(classifiers)))
meta_features_test = np.zeros((X_test.shape[0], len(classifiers)))

# K-fold cross-validation for creating meta-features
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    for j, (name, clf) in enumerate(classifiers.items()):
        clone_clf = clone(clf)
        clone_clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        val_pred = clone_clf.predict(X_train.iloc[val_idx])
        meta_features_train[val_idx, j] = val_pred
        test_pred = clone_clf.predict(X_test)
        meta_features_test[:, j] += test_pred

# Fit the meta-learner on meta-features
meta_learner.fit(meta_features_train, y_train)

# Make predictions using the meta-learner
meta_pred = meta_learner.predict(meta_features_test)
meta_accuracy = accuracy_score(y_test, meta_pred)
print(f"Stacking Ensemble Accuracy: {meta_accuracy * 100:.2f}%")

# %%
