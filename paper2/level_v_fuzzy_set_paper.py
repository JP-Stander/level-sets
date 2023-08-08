# %%
from images.utils import load_image
from graphical_model.utils import graphical_model
from level_sets.utils import get_level_sets, get_fuzzy_sets
from matplotlib import pyplot as plt
import networkx as nx
# %%
img_size = 50
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

# plt.figure()
# plt.imshow(img, 'gray')
# plt.xticks([])
# plt.yticks([])
# plt.savefig(f"../paper_results/dotted_image_{img_size}.png", bbox_inches='tight')

# plt.figure()
# plt.imshow(level_sets)
# plt.xticks([])
# plt.yticks([])
# plt.savefig(f"../paper_results/dotted_ls_{img_size}.png", bbox_inches='tight')

# plt.figure()
# plt.imshow(fuzzy_sets)
# plt.xticks([])
# plt.yticks([])
# plt.savefig(f"../paper_results/dotted_fs_{img_size}.png", bbox_inches='tight')

# %%
nodes1_ls, edges1_ls, attr1_ls = graphical_model(
    img=img,
    return_spp=True,
    alpha=0.5,
    set_type = "level",
)

nodes1_fs, edges1_fs, attr1_fs = graphical_model(
    img=img,
    return_spp=True,
    alpha=0.5,
    set_type="fuzzy",
    fuzzy_cutoff=10,
)

# %% #############################################################

G1_ls = nx.Graph()
edges1_ls = (edges1_ls > 0.001) * edges1_ls
# Add nodes
for index, row in nodes1_ls.iterrows():
    G1_ls.add_node(index) #, pos=(row['x-coor'], row['y-coor'])
    
for i in range(edges1_ls.shape[0]):
    for j in range(edges1_ls.shape[1]):
        if edges1_ls[i, j] != 0:
            G1_ls.add_edge(i, j, weight=edges1_ls[i, j])

for index, row in attr1_ls.iterrows():
    for col, attr_value in row.items():
        G1_ls.nodes[index][col] = attr_value

node_colors1_ls = attr1_ls['intensity']
node_sizes1_ls = attr1_ls['size']
cmap = plt.cm.viridis
nx.draw_networkx(
    G1_ls,
    with_labels=False,
    # pos=graphviz_layout(G),
    node_color = node_colors1_ls,
    node_size = node_sizes1_ls,
    edge_color = "gray",
    cmap=cmap
)
plt.savefig(f"../paper_results/dotted_ls_graph_{img_size}.png", bbox_inches='tight')
# %%
G1_fs = nx.Graph()
edges1_fs = (edges1_fs>0.001)*edges1_fs
# Add nodes
for index, row in nodes1_fs.iterrows():
    G1_fs.add_node(index) #, pos=(row['x-coor'], row['y-coor'])
    
for i in range(edges1_fs.shape[0]):
    for j in range(edges1_fs.shape[1]):
        if edges1_fs[i, j] != 0:
            G1_fs.add_edge(i, j, weight=edges1_fs[i, j])

for index, row in attr1_fs.iterrows():
    for col, attr_value in row.items():
        G1_fs.nodes[index][col] = attr_value

node_colors1_fs = attr1_fs['intensity']
node_sizes1_fs = attr1_fs['size']
cmap = plt.cm.viridis
nx.draw_networkx(
    G1_fs,
    with_labels=False,
    # pos=graphviz_layout(G1_fs),
    node_color = node_colors1_fs,
    node_size = node_sizes1_fs,
    edge_color = "gray",
    cmap=cmap
)
# plt.savefig(f"../paper_results/dotted_fs_graph_{img_size}.png", bbox_inches='tight')

# %% #############################################################
nodes2_ls, edges2_ls, attr2_ls = graphical_model(
    img=img2,
    return_spp=True,
    alpha=0.5,
    set_type = "level",
)

nodes2_fs, edges2_fs, attr2_fs = graphical_model(
    img=img2,
    return_spp=True,
    alpha=0.5,
    set_type = "fuzzy",
    fuzzy_cutoff = 10,
)

# %% ###############################################################
import networkx as nx
G2_ls = nx.Graph()
edges2_ls = (edges2_ls>0.001)*edges2_ls
# Add nodes
for index, row in nodes2_ls.iterrows():
    G2_ls.add_node(index) #, pos=(row['x-coor'], row['y-coor'])
    
for i in range(edges2_ls.shape[0]):
    for j in range(edges2_ls.shape[1]):
        if edges2_ls[i, j] != 0:
            G2_ls.add_edge(i, j, weight=edges2_ls[i, j])

for index, row in attr2_ls.iterrows():
    for col, attr_value in row.items():
        G2_ls.nodes[index][col] = attr_value

node_colors2_ls = attr2_ls['intensity']
node_sizes2_ls = attr2_ls['size']
cmap = plt.cm.viridis
nx.draw_networkx(
    G2_ls,
    with_labels=False,
    # pos=graphviz_layout(G),
    node_color = node_colors2_ls,
    node_size = node_sizes2_ls,
    edge_color = "gray",
    cmap=cmap
)
# plt.savefig(f"../paper_results/fibrous_ls_graph_{img_size}.png", bbox_inches='tight')
# %%
G2_fs = nx.Graph()
edges2_fs = (edges2_fs>0.001)*edges2_fs
# Add nodes
for index, row in nodes2_fs.iterrows():
    G2_fs.add_node(index) #, pos=(row['x-coor'], row['y-coor'])
    
for i in range(edges2_fs.shape[0]):
    for j in range(edges2_fs.shape[1]):
        if edges2_fs[i, j] != 0:
            G2_fs.add_edge(i, j, weight=edges2_fs[i, j])

for index, row in attr2_fs.iterrows():
    for col, attr_value in row.items():
        G2_fs.nodes[index][col] = attr_value

node_colors2_fs = attr2_fs['intensity']
node_sizes2_fs = attr2_fs['size']
cmap = plt.cm.viridis
nx.draw_networkx(
    G2_fs,
    with_labels=False,
    # pos=graphviz_layout(G_fs),
    node_color = node_colors2_fs,
    node_size = node_sizes2_fs,
    edge_color = "gray",
    cmap=cmap
)
# plt.savefig(f"../paper_results/fibrous_fs_graph_{img_size}.png", bbox_inches='tight')

#%%
# Distplots

import seaborn as sns

plt.figure()
plt.title("Degree distribution (level-sets)")
sns.distplot([degree[1] for degree in G1_ls.degree()], label = "Dotted")
sns.distplot([degree[1] for degree in G2_ls.degree()], label = "Fibrous")
plt.xticks([0,5,10,15])
plt.legend()
plt.savefig(f"../paper_results/distplot_ls_f_vs_d_{img_size}1.png", bbox_inches='tight')

plt.figure()
plt.title("Degree distribution (fuzzy-sets)")
sns.distplot([degree[1] for degree in G1_fs.degree()], label = "Dotted")
sns.distplot([degree[1] for degree in G2_fs.degree()], label = "Fibrous")
plt.xticks([0,5,10,15])
plt.legend()
plt.savefig(f"../paper_results/distplot_fs_f_vs_d_{img_size}1.png", bbox_inches='tight')

#%%
# Attribute dsitributions

for attribute in ['compactness', "elongation", "size"]:
    plt.figure()
    plt.title(f"Distribution of node {attribute} (fuzzy-sets)")
    sns.distplot([data[attribute] for node, data in G1_ls.nodes(data=True)], label = "Dotted")
    sns.distplot([data[attribute] for node, data in G2_ls.nodes(data=True)], label = "Fibrous")
    # plt.xticks([0,5,10,15])
    plt.legend()
    plt.savefig(f"../paper_results/distplot_{attribute}_ls_f_vs_d_{img_size}1.png", bbox_inches='tight')
    
for attribute in ['compactness', "elongation", "size"]:
    plt.figure()
    plt.title(f"Distribution of node {attribute} (fuzzy-sets)")
    sns.distplot([data[attribute] for node, data in G1_fs.nodes(data=True)], label = "Dotted")
    sns.distplot([data[attribute] for node, data in G2_fs.nodes(data=True)], label = "Fibrous")
    # plt.xticks([0,5,10,15])
    plt.legend()
    plt.savefig(f"../paper_results/distplot_{attribute}_fs_f_vs_d_{img_size}1.png", bbox_inches='tight')
    
# %% #############################################################
level_sets2 = get_level_sets(img2, 2)
fuzzy_sets2 = get_fuzzy_sets(img2, 40, 8)

plt.figure()
plt.imshow(img2, 'gray')
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper_results/fibrous_image_{img_size}.png", bbox_inches='tight')

plt.figure()
plt.imshow(level_sets2)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper_results/fibrous_ls_{img_size}.png", bbox_inches='tight')

plt.figure()
plt.imshow(fuzzy_sets2)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper_results/fibrous_fs_{img_size}.png", bbox_inches='tight')
# %%
