# %%
import os
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
from skimage.measure import regionprops, perimeter, label
from skimage.morphology import convex_hull_image
import pandas as pd
import numpy as np
from tqdm import tqdm
from level_sets.metrics import width_to_height, get_angle, compactness, elongation, perimeter
from level_sets.utils import get_fuzzy_sets, get_level_sets
from images.utils import load_image
# %%

def get_metrics(level_set):
    regions = regionprops(level_set)

    perim = perimeter(level_set)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        bbox_area = (maxr-minr)*(maxc-minc)
        aspect_ratio = (maxc - minc) / (maxr - minr)
        extent = region.area / ((maxr - minr) * (maxc - minc))
        orientation = region.orientation
        comp1 = region.area / (perim ** 2)
        convex_image = convex_hull_image(region.image)
        convex_area = np.sum(convex_image)  # Area of convex hull
        
        convex_perimeter = perimeter(convex_image)  # Better approximation for convex perimeter
        
        # Convexity using perimeter
        convexity_perimeter = perim / convex_perimeter
        # Convexity using area
        convexity_area = region.area / convex_area
    comp2 = compactness(level_set)
    eln = elongation(level_set)
    wth = width_to_height(level_set)
    angle = get_angle(level_set)
    return aspect_ratio, extent, eln, orientation, comp1, comp2, convexity_perimeter, convexity_area, bbox_area, wth, angle

# %%

folders = ['dotted', 'fibrous']
images = []
for folder in folders:
    images += [f"../dtd/images/{folder}/" + file for file in os.listdir(f"../dtd/images/{folder}")]

metrics_keys = [
    "aspect_ratio", "extent", "eln", "orientation",
    "comp1", "comp2", "convexity_perimeter", "convexity_area", "bbox_area", "wth", "angle"
]

# Initialize metrics dictionary
metrics = {folder: {key: [] for key in metrics_keys} for folder in folders}

for image in tqdm(images[:5]):
    img = load_image(image, [50,50])
    level_sets = get_level_sets(img)
    level_sets = get_fuzzy_sets(img, 10, 8)
    uni_level_sets = pd.unique(level_sets.flatten())
    for i, ls in enumerate(uni_level_sets):
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        if len(subset) <= 3:
            continue
        level_set = np.array(level_sets == ls)
        metric_results = get_metrics(level_set.astype(int))
        folder = image.split("/")[-2]
        for key, result in zip(metrics_keys, metric_results):
            metrics[folder][key].append(result)

# %% Plot distributions
data = []
for folder, folder_metrics in metrics.items():
    for metric_name, metric_values in folder_metrics.items():
        for value in metric_values:
            data.append({
                "Folder": folder,
                "Metric": metric_name,
                "Value": value
            })

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)
# Now, you can plot using seaborn
for metric in metrics_keys:
    plt.figure()
    # sns.boxplot(data=df[df['Metric'] == metric], x='Folder', y=metric)
    for folder in folders:
        subset = df[(df['Metric'] == metric) & (df['Folder'] == folder)]
        sns.kdeplot(subset['Value'], label=folder, fill=True)
    plt.title(f'Distribution of {metric} across folders')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% Mean and variance
for folder, folder_metrics in metrics.items():
    for metric_name, metric_values in folder_metrics.items():
        print(f"Mean(variance) for metric {metric_name} for texture {folder}: {np.mean(metric_values)}({np.var(metric_values)})")

# %% KS-statistic
# ks_stats = pd.DataFrame(0, columns=["metric"])



from scipy.stats import ks_2samp
for metric_name in metrics_keys:
    values1 = metrics[folders[0]][metric_name]
    values2 = metrics[folders[1]][metric_name]
    ks_stat, p_value = ks_2samp(values1, values2)
    print(f"KS statistc for {metric_name}: {ks_stat} with p={p_value}")
# %%
metrics = {folder: {key: [] for key in metrics_keys} for folder in folders}

for image in tqdm(images):
    img = load_image(image, [50,50])
    level_sets = get_fuzzy_sets(img, 10, 8)
    uni_level_sets = pd.unique(level_sets.flatten())
    for i, ls in enumerate(uni_level_sets):
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        if len(subset) <= 3:
            continue
        level_set = np.array(level_sets == ls)
        metric_results = get_metrics(level_set.astype(int))
        folder = image.split("/")[-2]
        for key, result in zip(metrics_keys, metric_results):
            metrics[folder][key].append(result)
table_data = []

for metric_name in metrics_keys:
    values1 = metrics[folders[0]][metric_name]
    values2 = metrics[folders[1]][metric_name]
    ks_stat, p_value = ks_2samp(values1, values2)
    table_data.append((metric_name, ks_stat))

# Sort the table data by KS statistics in descending order
table_data.sort(key=lambda x: x[1], reverse=True)

# Generate LaTeX code for the table
latex_table = "\\begin{tabular}{|c|c|}\n"
latex_table += "\\hline\n"
latex_table += "Metric Name & KS Statistic \\\\\n"
latex_table += "\\hline\n"

for metric_name, ks_stat in table_data:
    latex_table += f"{metric_name} & {ks_stat:.4f} \n"

latex_table += "\\hline\n"
latex_table += "\\end{tabular}"

# Print or save the LaTeX table code
print(latex_table)
# %%
