# %% Imports
import os
import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
from config import classes, images_loc, img_size, fs_connectivity, fs_delta, sets_feature_names, trim
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image
from level_sets.metrics import get_metrics
from level_sets.utils import get_fuzzy_sets, get_level_sets

# %%
images_loc = "../../dtd/images"
classes = ["dotted", "fibrous"]
images = []
for clas in classes:
    images += [f"{images_loc}/{clas}/" + file for file in os.listdir(f"{images_loc}/{clas}")]

# Initialize metrics dictionary
metrics = {clas: [] for clas in classes}

for image in tqdm(images):
    img = load_image(image, [img_size, img_size], trim=trim)
    # level_sets = get_level_sets(img)
    level_sets = get_fuzzy_sets(img, 35, fs_connectivity)
    uni_level_sets = pd.unique(level_sets.flatten())
    for i, ls in enumerate(uni_level_sets):
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        if len(subset) <= 1:
            continue
        level_set = np.array(level_sets == ls)
        metric_results = get_metrics(level_set.astype(int), img_size=[img_size, img_size], metric_names=["compactness", "elongation"])
        clas = image.split("/")[-2]
        metrics[clas] += [metric_results]

# %% Plot distributions (optional)
df = pd.DataFrame([(clas, metric, value)
                for clas, metrics_list in metrics.items() 
                for metrics_dict in metrics_list 
                for metric, value in metrics_dict.items()],
            columns=["Class", "Metric", "Value"]
            )

#%%
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, mode
for metric_name in pd.unique(df["Metric"]):
    for class_name in classes:
        values = df.loc[(df["Class"] == class_name) & (df["Metric"] == metric_name), "Value"]
        skewness = skew(values)
        kurt = kurtosis(values, fisher=True)
        mean = np.mean(values)
        median = np.median(values)
        mode_result = mode(values)
        
        print(f"Metric: {metric_name}, Class: {class_name}")
        print(f"Skewness: {skewness:.2f}")
        print(f"Kurtosis: {kurt:.2f}")
        print(f"Mean: {mean:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Mode: {mode_result.mode[0]:.2f} (Count: {mode_result.count[0]})")
        print()
#%%
measures = ['Kurtosis', 'Skewness', 'Mean', 'Median', 'Modality']

latex_table = "\\begin{tabular}{c|cc}\n"
latex_table += "\t\\textbf{Metric} & \\textbf{Dotted} & \\textbf{Fibrous}\\\\\n"
latex_table += "\t\\hline\n"

latex_table = "\\begin{tabular}{c|cc}\n"
latex_table += "\t\\textbf{Metric} & \\textbf{Dotted} & \\textbf{Fibrous}\\\\\n"
latex_table += "\t\\hline\n"

for metric_name in pd.unique(df["Metric"]):
    latex_table += f"\t{metric_name} & & \\\\\n"
    for measure in measures:
        results = []
        for class_name in classes:
            values = df.loc[(df["Class"] == class_name) & (df["Metric"] == metric_name), "Value"]
            if measure == 'Kurtosis':
                val = kurtosis(values, fisher=True)
            elif measure == 'Skewness':
                val = skew(values)
            elif measure == 'Mean':
                val = np.mean(values)
            elif measure == 'Median':
                val = np.median(values)
            elif measure == 'Modality':
                mode_result = mode(values, keepdims=True)
                val = mode_result.mode[0]
            else:
                val = ''
            results.append(f"{val:.2f}")
        latex_table += f"\t{measure} & {results[0]} & {results[1]}\\\\\n"
latex_table += "\\end{tabular}"
print(latex_table)

# %%
from scipy.stats import norm

for metric in ["compactness", "elongation"]:
    data = df[df['Metric'] == metric]

    # Calculate the mean and standard deviation for the 'Value' column
    mean_value = data['Value'].mean()
    std_value = data['Value'].std()
    
    # Create a range of values for plotting the normal distribution
    range_values = np.linspace(data['Value'].min(), data['Value'].max()+0.2, 100)
    
    # Get the PDF of the normal distribution with the same mean and std dev
    normal_distribution = norm.pdf(range_values, mean_value, std_value)

    plt.figure()
    sns.kdeplot(data=data, x='Value', hue='Class', common_norm=False, legend=False)
    plt.plot(range_values, normal_distribution, label='Normal Dist', linestyle='--', c="gray")
    # Add labels and a title
    plt.xlabel(metric.capitalize())
    plt.ylabel('Density')
    # Show the plot
    plt.savefig(f"../../paper3_results/{classes[0]}_v_{classes[1]}_{metric}_with_normal.png")
# %%
from scipy.stats import gaussian_kde
metric = "compactness"
data = df[df['Metric'] == metric]
data1 = data[data["Class"]==classes[0]]["Value"]
data2 = data[data["Class"]==classes[1]]["Value"]
kde1 = gaussian_kde(data1)
kde2 = gaussian_kde(data2)

# Evaluate the PDFs on a grid
x = np.linspace(min(np.min(data1), np.min(data2)), max(np.max(data1), np.max(data2)), 1000)
pdf1 = kde1(x)
pdf2 = kde2(x)

# Calculate the intersection and union
intersection = np.minimum(pdf1, pdf2)
union_ = np.maximum(pdf1, pdf2)
intersection_area = np.trapz(intersection, x)
union_area = np.trapz(union_, x)
# %%

plt.figure()#figsize=(10, 6))
# plt.plot(x, pdf1, label='Honeycomb', color='blue', alpha=0.2)
# plt.plot(x, pdf2, label='Striped', color='green', alpha=0.2)
plt.fill_between(x, 0, intersection, color='red', alpha=0.5, label='Intersection')
plt.plot(x, union_, label='Union', color='purple', linestyle='--')
plt.legend()
plt.savefig(f"../../paper3_results/{classes[0]}_v_{classes[1]}_iou_{metric}.png")

print(intersection_area/union_area)

plt.figure()
sns.kdeplot(data=data, x='Value', hue='Class', common_norm=False, legend=False)
# Add labels and a title
plt.xlabel(metric.capitalize())
plt.ylabel('Density')
# Show the plot
plt.savefig(f"../../paper3_results/{classes[0]}_v_{classes[1]}_{metric}.png")
# %%

