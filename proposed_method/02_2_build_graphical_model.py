# %%
import os
from config import classes, images_loc, graph_locs, ds, img_size, sets_feature_names, trim, fs_connectivity, edge_delta
from os import listdir, cpu_count
from utils import img_to_graph
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=Warning)


def get_output_path(image, graph_location):
    return f"{graph_location}/{image.split('/')[-1].split('.')[0]}_graph.graphml"

# %%

images = {
    clas: [f"{images_loc}/{clas}/" + file for file in listdir(f"{images_loc}/{clas}")] for clas in classes
}

for clas in graph_locs.keys():
    for _, graph_loc in graph_locs[clas].items():
        if not os.path.exists(graph_loc):
            os.makedirs(graph_loc)

# Compute the total number of iterations
all_combinations = [(image, graph_locs[clas][d_value], d_value) 
                    for clas in classes 
                    for image in images[clas] 
                    for d_value in ds]

# Step 2: Filter out combinations where the output path exists
unprocessed_combinations = [(image, graph_loc, d_value) 
                            for image, graph_loc, d_value in all_combinations 
                            if not os.path.exists(get_output_path(image, graph_loc))]

for image, graph_loc, d_value in tqdm(unprocessed_combinations):
    img_to_graph(
        image, 
        graph_loc,
        d_value, img_size,
        fs_connectivity,
        edge_delta,
        False,
        sets_feature_names,
        trim
    )


# %%
