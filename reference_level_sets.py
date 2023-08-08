
# %%
from level_sets.metrics import elongation, compactness, width_to_height, get_angle
import pandas as pd
import numpy as np
from os import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# %%

def get_neighbors(x, y, connectivity=4):
    if connectivity == 4:
        return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    elif connectivity == 8:
        return [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
                (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)]
    else:
        raise ValueError(f"Only 4 and 8 connectivity are supported. \nProvide connectivity :{connectivity}")

def generate_level_sets(x, y, size, connectivity, current_set, all_sets, grid_width, grid_height):
    if size == 1:
        all_sets.append(current_set.copy())
        return
    
    neighbors = get_neighbors(x, y, connectivity)
    for nx, ny in neighbors:
        if (nx, ny) not in current_set and 0 <= nx < grid_width and 0 <= ny < grid_height:
            current_set.append((nx, ny))
            generate_level_sets(nx, ny, size-1, connectivity, current_set, all_sets, grid_width, grid_height)
            current_set.remove((nx, ny))

def generate_all_level_sets(grid_width, grid_height, size, connectivity=4):
    all_sets = []
    for x in range(grid_width):
        for y in range(grid_height):
            generate_level_sets(x, y, size, connectivity, [(x, y)], all_sets, grid_width, grid_height)
    return [list(fset) for fset in {frozenset(set) for set in all_sets}]

def create_set_characteristics(set, intensities = [0, 255]):
    columns = ["size", "compactness", "elongation", "width_to_height", "angle", "intensity"]
    
    sets_char = pd.DataFrame(np.zeros((len(intensities), len(columns))), columns = columns)

    img = np.zeros((len(set), len(set)))
    rows, cols = zip(*set)
    img[rows, cols] = 1
    
    size = sum(img.flatten()>0)
    comp = compactness(img)
    elon = elongation(img)
    w_t_h = width_to_height(img)
    angle = get_angle(img)

    for i, intensity in enumerate(intensities):
        # sets_char.loc[i, "id"] = 1
        sets_char.loc[i, "size"] = size
        sets_char.loc[i, "compactness"] = comp
        sets_char.loc[i, "elongation"] = elon
        sets_char.loc[i, "width_to_height"] = w_t_h
        sets_char.loc[i, "angle"] = angle
        sets_char.loc[i, "intensity"] = intensity

    return sets_char

def process_sets(sets, n_cores, file_name):
    results_list = []  # Initialize list to store individual results

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        characterise = {executor.submit(create_set_characteristics, set): set for set in sets}
        for character in tqdm(as_completed(characterise), total=len(sets)):
            try:
                result_df = character.result()  # retrieve results
                results_list.append(result_df)
            except Exception as e:
                print(f"Error processing level-set: {characterise[character]}. Error: {e}")

    # Concatenate all the results and save to CSV
    final_df = pd.concat(results_list, ignore_index=True)
    final_df.to_csv(file_name, index=False)

# %%

if __name__ == "__main__":
    n_cores = cpu_count() - 2
    print(f'Running process on {n_cores} cores')
    min_size = 2
    max_size = 8
    sets_4 = [set for i in range(min_size, max_size) for set in generate_all_level_sets(i, i, i, connectivity=4)]
    sets_8 = [set for i in range(min_size, max_size) for set in generate_all_level_sets(i, i, i, connectivity=8)]
    
    print(f"Level level-sets (size 4): {len(sets_4)}\n")
    print(f"Level level-sets (size 8): {len(sets_8)}\n")

    process_sets(sets_4, n_cores, f'../unique_level_sets/sets_{min_size}_to_{max_size}_4conn.csv')
    process_sets(sets_8, n_cores, f'../unique_level_sets/sets_{min_size}_to_{max_size}_8conn.csv')
