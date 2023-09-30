# %%
import os
from os import listdir, cpu_count
from utils import img_to_graph
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# %%


if __name__ == '__main__':
    images_path = "../../dtd/images"
    
    images = [f"{images_path}/dotted/" + file for file in listdir(f"{images_path}/dotted")][:5]
    images += [f"{images_path}/fibrous/" + file for file in listdir(f"{images_path}/fibrous")][:5]
    ds = [10]#[i for i in range(26)] + [30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 255]
    img_size = 99
    
    graph_locs = {d: f"../../graphical_models_{img_size}/fuzzy_sets_{d}" for d in ds}

    for _, graph_loc in graph_locs.items():
        if not os.path.exists(graph_loc):
            os.makedirs(graph_loc)

    n_cores = cpu_count() - 2
    print(f'Running process on {n_cores} cores')

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Setup tqdm
        future_to_detail = {(executor.submit(img_to_graph, image, graph_locs[d_value], d_value, img_size)): (image, d_value) for image in images for d_value in ds}

        for future in tqdm(as_completed(future_to_detail), total=len(images) * len(ds)):
            try:
                future.result()  # retrieve results if there are any
            except Exception as e:
                image, d_value = future_to_detail[future]
                print(f"Error processing image: {image} with d_value: {d_value}. Error: {e}")

# %%
