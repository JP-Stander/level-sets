
# %%
from images.utils import load_image
from level_sets.metrics import elongation, compactness, width_to_height, major_axis, get_angle
from graphical_model.utils import graphical_model
from level_sets.utils import get_level_sets, get_fuzzy_sets
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
# %%
img_size = 100
img = load_image(
    "../dtd/images/dotted/dotted_0161.jpg",
    [img_size, img_size]
)

img2 = load_image(
    "../dtd/images/fibrous/fibrous_0116.jpg",
    [img_size, img_size]
)

# %%
level_sets = get_level_sets(img, 2)
fuzzy_sets = get_fuzzy_sets(img, 20, 8)

fuzzy_set_size_count = pd.value_counts(fuzzy_sets.flatten())
level_set_size_count = pd.value_counts(level_sets.flatten())
fuzzy_set_size_count = fuzzy_set_size_count[fuzzy_set_size_count < 1000]
level_set_size_count = level_set_size_count[level_set_size_count < 1000]

plt.figure()
sns.histplot(fuzzy_set_size_count, label='Fuzzy sets', bins=10)
sns.histplot(level_set_size_count, label='Level sets', bins=10)
plt.legend()
plt.show()

p = 10
print(f"Proporion level-sets < size {p}: {np.mean(level_set_size_count < p)}")
print(f"Proporion fuzzy-sets < size {p}: {np.mean(fuzzy_set_size_count < p)}")
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

    for intensity in i, intensity in enumerate(intensities):
        # sets_char.loc[i, "id"] = 1
        sets_char.loc[i, "size"] = size
        sets_char.loc[i, "compactness"] = comp
        sets_char.loc[i, "elongation"] = elon
        sets_char.loc[i, "width_to_height"] = w_t_h
        sets_char.loc[i, "angle"] = angle
        sets_char.loc[i, "intensity"] = intensity

    return sets_char

# %%

# Example: Generating all level sets of size 3 using 4-connectivity on a 5x5 grid
sets_4 = generate_all_level_sets(2, 2, 2, connectivity=4)
print("4-connectivity:", sets_4)

# Example: Generating all level sets of size 3 using 8-connectivity on a 5x5 grid
sets_8 = generate_all_level_sets(2, 2, 2, connectivity=8)
print("\n8-connectivity:", sets_8)

# %%
sets_4 = []
sets_8 = []
for i in range(2,8,1):
    sets_4 += generate_all_level_sets(i, i, i, connectivity=4)
    sets_8 += generate_all_level_sets(i, i, i, connectivity=8)
    print(f"{i}\n")
    print(f"Level level-sets (size 4): {len(sets_4)}\n")
    print(f"Level level-sets (size 8): {len(sets_8)}\n")

columns = ["id", "size", "compactness", "elongation", "width_to_height", "angle", "intensity"]
# %%
n_level_sets_8 = len(sets_8)
level_sets_char_8 = pd.DataFrame(np.zeros((n_level_sets_8*2, len(columns))), columns = columns)

for i, set in enumerate(sets_8):
    img = np.zeros((len(set), len(set)))
    rows, cols = zip(*set)
    img[rows, cols] = 1
    size = sum(img.flatten()>0)
    comp = compactness(img)
    elon = elongation(img)
    w_t_h = width_to_height(img)
    angle = get_angle(img)
    # Black set
    level_sets_char_8.loc[i, "id"] = i+1
    level_sets_char_8.loc[i, "size"] = size
    level_sets_char_8.loc[i, "compactness"] = comp
    level_sets_char_8.loc[i, "elongation"] = elon
    level_sets_char_8.loc[i, "width_to_height"] = w_t_h
    level_sets_char_8.loc[i, "angle"] = angle
    level_sets_char_8.loc[i, "intensity"] = 0
    # White set
    level_sets_char_8.loc[i+n_level_sets_8, "id"] = i+n_level_sets_8+1
    level_sets_char_8.loc[i+n_level_sets_8, "size"] = size
    level_sets_char_8.loc[i+n_level_sets_8, "compactness"] = comp
    level_sets_char_8.loc[i+n_level_sets_8, "elongation"] = elon
    level_sets_char_8.loc[i+n_level_sets_8, "width_to_height"] = w_t_h
    level_sets_char_8.loc[i+n_level_sets_8, "angle"] = angle
    level_sets_char_8.loc[i+n_level_sets_8, "intensity"] = 255
# %%
n_level_sets_4 = len(sets_4)
level_sets_char_4 = pd.DataFrame(np.zeros((n_level_sets_4*2, len(columns))), columns = columns)

for i, set in enumerate(sets_4):
    img = np.zeros((len(set), len(set)))
    rows, cols = zip(*set)
    img[rows, cols] = 1
    size = sum(img.flatten()>0)
    comp = compactness(img)
    elon = elongation(img)
    w_t_h = width_to_height(img)
    angle = get_angle(img)
    # Black set
    level_sets_char_4.loc[i, "id"] = i+1
    level_sets_char_4.loc[i, "size"] = size
    level_sets_char_4.loc[i, "compactness"] = comp
    level_sets_char_4.loc[i, "elongation"] = elon
    level_sets_char_4.loc[i, "width_to_height"] = w_t_h
    level_sets_char_4.loc[i, "angle"] = angle
    level_sets_char_4.loc[i, "intensity"] = 0
    # White set
    level_sets_char_4.loc[i+n_level_sets_4, "id"] = i+n_level_sets_4+1
    level_sets_char_4.loc[i+n_level_sets_4, "size"] = size
    level_sets_char_4.loc[i+n_level_sets_4, "compactness"] = comp
    level_sets_char_4.loc[i+n_level_sets_4, "elongation"] = elon
    level_sets_char_4.loc[i+n_level_sets_4, "width_to_height"] = w_t_h
    level_sets_char_4.loc[i+n_level_sets_4, "angle"] = angle
    level_sets_char_4.loc[i+n_level_sets_4, "intensity"] = 255


unique_level_sets_char_8 = level_sets_char_8.drop(columns=['id']).drop_duplicates()
unique_level_sets_char_4 = level_sets_char_4.drop(columns=['id']).drop_duplicates()

