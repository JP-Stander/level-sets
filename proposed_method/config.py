import os
import json

config_name = "med"

# med_exp3: 80
# med_exp4: 100,30
configs = {
    # Colab w Alisa
    "med": {
        # General configurations
        "experiment_name": "med_experiment4",
        "classes": ['asthma', 'control'],
        "images_loc": "../../colab_alisa",
        # Image configurations
        "trim": {"bottom": 0.08},
        "img_size": 100,
        # Fuzzy-sets configurations
        "fs_delta": 15,
        "fs_connectivity": 8,
        "ds": [10, 15, 20],
        "sets_feature_names": [
            'compactness', 'elongation', 'area', 'angle'
        ],
        # Bag-of-visual-words configurations
        "num_clusters": 50, #30
        "max_graphlet_size": 2,
        # Graphical model configurations
        "edge_delta": 0.25,
        # Bootstrap configurations
        "num_bootstrap_iterations": 1000,
        "images_for_inference": {
            "asthma": ["A5_PRP+T_40X_05.tif", "A4_PRP+T_40X_03.tif"],
            "control": ["C1509_PPP_T_30K_05.tif", "Conradie_PPP_T_20K_02.tif"]
        },
        "graphlet_names": ["g2_1"] #+ \
            # [f"g3_{i+1}" for i in range(2)] + \
            # [f"g4_{i+1}" for i in range(6)]
    },
    "dtd": {
        # General configurations
        "experiment_name": "experiment3",
        "classes": ['dotted', 'fibrous'],
        "images_loc": "../../dtd/images",
        # Image configurations
        "trim": None,
        "img_size": 100,
        # Fuzzy-sets configurations
        "fs_delta": 10,
        "fs_connectivity": 8,
        "ds": [10, 15, 20],
        "sets_feature_names": [
            'compactness', 'elongation', 'convexity', 'extent'
        ],
        # Bag-of-visual-words configurations
        "num_clusters": 10, #80
        "max_graphlet_size": 4,
        # Graphical model configurations
        "edge_delta": 0.5,
        # Bootstrap configurations
        "num_bootstrap_iterations": 1000,
        "images_for_inference": {
            "dotted": ["dotted_0188.jpg", "dotted_0111.jpg", "dotted_0180.jpg"],
            "fibrous": ["fibrous_0191.jpg", "fibrous_0108.jpg", "fibrous_0116.jpg"]
        },
        "graphlet_names": ["g2_1"] + \
            [f"g3_{i+1}" for i in range(2)] + \
            [f"g4_{i+1}" for i in range(6)]
    }
}

for key in configs.keys():
    configs[key]["graphlet_names"] += [f'g{i+1}' for i in range(configs[key]["num_clusters"])]

# This makes each key of the dict a variable with value, the value of the key 
globals().update(configs[config_name])

# General configurations
experiment_loc = os.path.join("results", experiment_name, f"fuzzy_set_{fs_delta}")

# Graphical model configurations
nodes_feature_names = sets_feature_names + ['intensity']

graph_locs = {clas: {d: os.path.join(images_loc, "graphical_models", experiment_name, f"graphical_models_{img_size}", f"fuzzy_sets_{d}", clas) for d in ds} for clas in classes}
graphs_location = os.path.join(images_loc, "graphical_models", experiment_name, f"graphical_models_{img_size}", f"fuzzy_sets_{fs_delta}")

# Inference configurations
results_location = os.path.join("..", "..", "paper3_results", config_name)

experiment_parameters = configs[config_name]

directory_to_check = os.path.join(images_loc, "graphical_models", experiment_name)
os.makedirs(directory_to_check, exist_ok=True)

with open(os.path.join(directory_to_check, "experiment_parameter.json"), "w") as file:
    json.dump(experiment_parameters, file)
