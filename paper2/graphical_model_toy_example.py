# %%
import numpy as np
from matplotlib import pyplot as plt
from graphical_model.utils import graphical_model
from level_sets.utils import get_level_sets

# %%
img = np.array([
    [1,1,1,2,2],
    [4,3,3,2,2],
    [4,3,5,5,6],
    [4,4,5,6,6],
    [5,5,5,6,7]
])

plt.figure()
plt.imshow(img, 'rainbow')
plt.xticks([])
plt.yticks([])
plt.vlines([-0.5, 0.5, 1.5 ,2.5, 3.5, 4.5], -0.5, 4.5, 'black', linewidth=0.5)
plt.hlines([-0.5, 0.5, 1.5 ,2.5, 3.5, 4.5], -0.5, 4.5, 'black', linewidth=0.5)
plt.savefig(f"../paper_results/level_sets.png", bbox_inches='tight')

plt.figure()
plt.imshow(img, 'rainbow')
plt.xticks([])
plt.yticks([])
plt.vlines([-0.5, 0.5, 1.5 ,2.5, 3.5, 4.5], -0.5, 4.5, 'black', linewidth=0.5)
plt.hlines([-0.5, 0.5, 1.5 ,2.5, 3.5, 4.5], -0.5, 4.5, 'black', linewidth=0.5)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        plt.text(j, i, str(img[i, j]), ha='center', va='center', color='black', fontsize=24, fontweight='bold')  # 'red' color for visibility, change as needed

plt.savefig(f"../paper_results/labelled.png", bbox_inches='tight')
# %%

ls = get_level_sets(img)
unique_labels = np.unique(ls)

# Compute average location for each label
avg_locations = []
for label in unique_labels:
    # Get pixel locations for the current label
    locations = np.argwhere(ls == label)
    
    # Compute the average location (centroid)
    avg_location = locations.mean(axis=0)
    avg_locations.append(avg_location)

# Convert avg_locations to NumPy array for easier indexing
avg_locations = np.array(avg_locations)

# Plot the image and average locations
# plt.imshow(ls, cmap='tab20b')  # Use a colormap that can differentiate between the labels

# Scatter plot the centroids with viridis colormap based on label values
sc = plt.scatter(avg_locations[:, 1], avg_locations[:, 0], c=unique_labels, cmap='rainbow', marker='o', zorder=3, s=100)

# Annotate each centroid with its corresponding label
# for i, label in enumerate(unique_labels):
#     plt.annotate(str(label), (avg_locations[i, 1], avg_locations[i, 0]))

plt.xticks([])
plt.yticks([])
plt.vlines([-0.5, 0.5, 1.5 ,2.5, 3.5, 4.5], -0.5, 4.5, 'black', linewidth=0.5)
plt.hlines([-0.5, 0.5, 1.5 ,2.5, 3.5, 4.5], -0.5, 4.5, 'black', linewidth=0.5)
plt.gca().invert_yaxis()
plt.axis('off')
plt.savefig(f"../paper_results/marked_point_pattern.png", bbox_inches='tight')

# %%

n,e,a = graphical_model(
        img/7*255,
        True,
        0.5,
        normalise_gray = True
    )
# %%

plt.figure()

# Plot nodes

# Iterate through edges matrix
for i in range(e.shape[0]):
    for j in range(i+1, e.shape[1]):  # This ensures that we consider only one edge between two nodes.
        if e[i, j] > 0.055:
            plt.plot([n.iloc[i, 0], n.iloc[j, 0]], [n.iloc[i, 1], n.iloc[j, 1]], 'k-', linewidth=e[i, j]*20)  # Plot edge
            mid_x = (n.iloc[i, 0] + n.iloc[j, 0]) / 2
            mid_y = (n.iloc[i, 1] + n.iloc[j, 1]) / 2
            plt.annotate(f"{e[i, j]:.2f}", (mid_x, mid_y))  # Annotate the edge with its weight
plt.scatter(n.iloc[:, 0], n.iloc[:, 1], c=unique_labels, cmap='rainbow', s=100, zorder=2)
plt.xticks([])
plt.yticks([])
plt.gca().invert_yaxis()
plt.savefig(f"../paper_results/graphical_model.png", bbox_inches='tight')

# %%
import networkx as nx
n = np.array(n)
G = nx.Graph()

for i, coord in enumerate(n):
    G.add_node(i, pos=coord)

for i in range(e.shape[0]):
    for j in range(i+1, e.shape[1]):
        if e[i, j] > 0.055:
            G.add_edge(i, j, weight=e[i, j])


# Extract edge weights from the graph
weights = [G[u][v]['weight'] for u, v in G.edges()]

# Format edge labels rounded to 2 decimal places
edge_labels = {(u, v): f"{weight:.2f}" for u, v, weight in G.edges(data='weight')}
sn = 124
# Draw the graph
plt.figure()
nx.draw_networkx(
    G,
    with_labels=False,
    pos=nx.spring_layout(G, seed=sn),  # Using spring_layout for positioning
    node_color=unique_labels,
    edge_color="gray",
    cmap = 'rainbow',
    width=[w * 50 for w in weights]  # Adjust the multiplier as needed for desired thickness
)

nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G, seed=sn), edge_labels=edge_labels, font_size=15)
# plt.axis('off') 
plt.savefig("../paper_results/graphical_model.png")

# %%
