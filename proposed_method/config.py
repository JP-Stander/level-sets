import json
# Colab w Alisa
# General configurations
experiment_name = "med_experiment1"
classes = ['asthma', 'control']
images_loc = "../../colab_alisa"
experiment_loc = f"results/{experiment_name}"

# Image configurations
trim = {"bottom": 0.08}

# Fuzzy-sets configurations
fs_delta = 10
fs_connectivity = 8
ds = [fs_delta, 15, 20]#[i for i in range(26)] + [30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 255]
img_size = 100
sets_feature_names = [
    'compactness',
    'elongation',
    'area',
    'angle'
]

# Bag-of-visual-words configurations
num_clusters = 200
max_graphlet_size = 2

# Graphical model configurations
edge_delta = 0.25
nodes_feature_names = sets_feature_names + ['intensity']

graph_locs = {clas: {d: f"{images_loc}/graphical_models/{experiment_name}/graphical_models_{img_size}/fuzzy_sets_{d}/{clas}" for d in ds} for clas in classes}
# graphs_location = f"{images_loc}/graphical_models/graphical_models_{img_size}/fuzzy_sets_{fs_delta}"
graphs_location = f"{images_loc}/graphical_models/{experiment_name}/graphical_models_{img_size}/fuzzy_sets_{fs_delta}"

# Bootstrap configurations
num_bootstrap_iterations = 1000

# Inference configurations
results_location = "../../paper3_results/colab_alisa"
images_for_inference = {
    "asthma": "A5_PRP+T_40X_05.tif",
    "control": "C1509_PPP_T_30K_05.tif"
}

graphlet_names = [f'g{i+1}' for i in range(num_clusters)] + \
    ["g2_1"] #+ \
    # [f"g3_{i+1}" for i in range(2)] + \
    # [f"g4_{i+1}" for i in range(6)]

# variables = ["images_loc", "experiment_name", "fs_delta", "fs_connectivity", "num_bootstrap_iterations",
#             "img_size", "sets_feature_names", "num_clusters", "max_graphlet_size", "edge_delta"]

# experiment_parameters = {var: locals()[var] for var in variables}


# if not os.path.exists(f"{images_loc}/graphical_models//{experiment_name}"):
#     os.makedirs(f"{images_loc}/graphical_models//{experiment_name}")
# else:
#     print(f"the following directory already exists and content may be overwritten \n{images_loc}/graphical_models//{experiment_name}")

# with open(f"{images_loc}/graphical_models//{experiment_name}/experiment_parameter.json", "w") as file:
#     json.dump(experiment_parameters, file)

# DTD usecase
## General configurations
# experiment_name = "experiment3"
# classes = ['dotted', 'fibrous']
# images_loc = "../../dtd/images"
# experiment_loc = f"results/{experiment_name}"
# trim = None

# # Fuzzy-sets configurations
# fs_delta = 10
# fs_connectivity = 8
# ds = [fs_delta]#[i for i in range(26)] + [30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 255]
# img_size = 100
# sets_feature_names = [
#     'compactness',
#     'elongation',
#     'extent',
#     'convexity'
# ]

# # Bag-of-visual-words configurations
# num_clusters = 10 #10, 120
# max_graphlet_size = 2

# # Graphical model configurations
# edge_delta = 0.5
# nodes_feature_names = sets_feature_names + ['intensity']

# graph_locs = {clas: {d: f"../../graphical_models_{img_size}/fuzzy_sets_{d}/{clas}" for d in ds} for clas in classes}
# graphs_location = f"../../graphical_models_{img_size}/fuzzy_sets_{fs_delta}"

# # Bootstrap configurations
# num_bootstrap_iterations = 1000

# # Inference configurations
# results_location = "../../paper3_results"

# ["dotted_0188.jpg", "fibrous_0191.jpg"] #, "dotted_0111", "dotted_0180", "fibrous_0191", "fibrous_0108", "fibrous_0116"]

# graphlet_names = [f'g{i+1}' for i in range(num_clusters)] + \
#     ["g2_1"] #+ \
#     # [f"g3_{i+1}" for i in range(2)] + \
#     # [f"g4_{i+1}" for i in range(6)]
