# General configurations
experiment_name = "experiment3"
classes = ['dotted', 'fibrous']
images_loc = "../../dtd/images"
experiment_loc = f"results/{experiment_name}"

# Fuzzy-sets configurations
fs_delta = 10
fs_connectivity = 8
ds = [fs_delta]#[i for i in range(26)] + [30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 255]
img_size = 100
sets_feature_names = [
    'compactness',
    'elongation',
    'extent',
    'convexity'
]

# Bag-of-visual-words configurations
num_clusters = 80
max_graphlet_size = 2

# Graphical model configurations
edge_delta = 0.5
nodes_feature_names = sets_feature_names + ['intensity']

graph_locs = {clas: {d: f"../../graphical_models_{img_size}/fuzzy_sets_{d}/{clas}" for d in ds} for clas in classes}
graphs_location = f"../../graphical_models_{img_size}/fuzzy_sets_{fs_delta}"

# Bootstrap configurations
num_bootstrap_iterations = 1000

# Inference configurations
results_location = "../../paper3_results"

graphlet_names = [f'g{i+1}' for i in range(num_clusters)] + \
    ["g2_1"] #+ \
    # [f"g3_{i+1}" for i in range(2)] + \
    # [f"g4_{i+1}" for i in range(6)]
