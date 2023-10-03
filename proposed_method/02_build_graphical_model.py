# %%
import os
from config import classes, images_loc, graph_locs, ds, img_size, sets_feature_names, trim, fs_connectivity, edge_delta
from os import listdir, cpu_count
from utils import img_to_graph
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# %%

if __name__ == '__main__':
    images = {
        clas: [f"{images_loc}/{clas}/" + file for file in listdir(f"{images_loc}/{clas}")] for clas in classes
    }

    for clas in graph_locs.keys():
        for _, graph_loc in graph_locs[clas].items():
            if not os.path.exists(graph_loc):
                os.makedirs(graph_loc)

    n_cores = cpu_count() - 2
    print(f'Running process on {n_cores} cores')

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Setup tqdm
        future_to_detail = {
            executor.submit(
                img_to_graph,
                image, 
                graph_locs[clas][d_value],
                d_value, img_size,
                fs_connectivity,
                edge_delta,
                False,
                sets_feature_names,
                trim
            ): (image, d_value, clas)
            for clas in classes
            for image in images[clas]
            for d_value in ds 
        }

        for future in tqdm(as_completed(future_to_detail), total=sum(len(image) for _, image in images.items()) * len(ds)):
            try:
                future.result()  # retrieve results if there are any
            except Exception as e:
                image, d_value, clas = future_to_detail[future]
                print(f"Error processing image: {image} in class: {clas} with d_value: {d_value}. Error: {e}")
