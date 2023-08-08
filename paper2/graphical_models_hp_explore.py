# %%
from graphical_model.utils import graphical_model
from level_sets.utils import get_level_sets
from images.utils import load_image
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
# %%

img = load_image(
    "../dtd/images/dotted/dotted_0161.jpg",
    [50, 50]
)

img2 = load_image(
    "../dtd/images/fibrous/fibrous_0116.jpg",
    [50, 50]
)

nodes, edges, attr = graphical_model(
    img,
    True,
    0.5,
    normalise_gray=True,
    size_proportion=False,
    ls_spatial_dist="euclidean",
    ls_attr_dist="cityblock",
    centroid_method="mean"
)
# %%

G = nx.Graph()

# Add nodes
for index, row in nodes.iterrows():
    G.add_node(index)

for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        if edges[i, j] != 0:
            G.add_edge(i, j, weight=edges[i, j])

for index, row in attr.iterrows():
    for col, attr_value in row.items():
        G.nodes[index][col] = attr_value

# %%

nx.draw(G, with_labels=False)
# %%

node_colors = attr['intensity']
node_sizes = attr['size']
cmap = plt.cm.viridis
nx.draw_networkx(
    G,
    with_labels=False,
    pos=graphviz_layout(G),
    # node_size=3,
    # edgecolors="Red",
    node_color=node_colors,
    node_size=node_sizes,
    cmap=cmap,
    # alpha = 0.5
)
# plt.colorbar(np.array([data['features'][i][0] for i in data['features'] if int(i) in list(G.nodes())]))

norm = plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, orientation='vertical', label='Your Label Here')  # Adjust label as needed

plt.show()
# %%


def is_valid(i, j, matrix):
    # Check if the pixel (i, j) is within the bounds of the matrix
    return 0 <= i < len(matrix) and 0 <= j < len(matrix[0])


def dfs(i, j, matrix, set_id, output, delta, connectivity, reference_pixel_value):
    # Define possible moves based on the connectivity
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        moves += [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    stack = [(i, j)]
    while stack:
        i, j = stack.pop()
        if is_valid(i, j, matrix) and abs(matrix[i][j] - reference_pixel_value) <= delta:
            output[i][j] = set_id
            for move in moves:
                ni, nj = i + move[0], j + move[1]
                if is_valid(ni, nj, matrix) and output[ni][nj] == -1:
                    stack.append((ni, nj))


def get_fuzzy_sets(matrix, delta=0, connectivity=4):
    output = [[-1 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]  # Initialize with -1 (unassigned)
    set_id = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if output[i][j] == -1:
                dfs(i, j, matrix, set_id, output, delta, connectivity, matrix[i][j])
                set_id += 1  # Increment set ID for the next set
    return np.array(output)
# %%
img_size = 150
img = load_image(
    "../dtd/images/dotted/dotted_0161.jpg",
    [img_size, img_size]
)

img2 = load_image(
    "../dtd/images/fibrous/fibrous_0116.jpg",
    [img_size, img_size]
)

f, axarr = plt.subplots(2,3)
axarr[0, 0].imshow(img)
axarr[0, 0].set_title("Dotted image")
axarr[0, 1].imshow(get_level_sets(img, 2))
axarr[0, 1].set_title("Dotted level-sets")
axarr[0, 2].imshow(get_fuzzy_sets(img, 20, 8))
axarr[0, 2].set_title("Dotted fuzzy-sets")
axarr[1, 0].imshow(img2)
axarr[1, 0].set_title("Fibrous image")
axarr[1, 1].imshow(get_level_sets(img2, 2))
axarr[1, 1].set_title("Fibrous level-sets")
axarr[1, 2].imshow(get_fuzzy_sets(img2, 20, 8))
axarr[1, 2].set_title("Fibrous fuzzy-sets")

# Add a title for the entire figure
f.suptitle(f"Image size: {img_size}")

# Remove ticks from all axes
for ax_row in axarr:
    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])

# plt.tight_layout()  # Adjust the spacing between subplots for better layout
plt.show()

# %%

img2 = load_image(
    "../dtd/images/fibrous/fibrous_0116.jpg",
    [img_size,img_size]
)

plt.figure()
plt.imshow(img2, 'gray')
plt.yticks([])
plt.xticks([])
plt.show()
