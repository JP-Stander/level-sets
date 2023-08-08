# %%
import networkx as nx
import matplotlib.pyplot as plt
import collections
import seaborn as sns
from networkx.drawing import layout
# from networkx.drawing.nx_agraph import graphviz_layout

# %%
# Homogeneous Network
G_reg = nx.cycle_graph(100)  # Ring lattice
degree_sequence_reg = sorted([d for n, d in G_reg.degree()], reverse=True)
degreeCount_reg = collections.Counter(degree_sequence_reg)
deg_reg, cnt_reg = zip(*degreeCount_reg.items())

# Heterogeneous Network
G_sf = nx.barabasi_albert_graph(100, 2)  # Scale-free network
degree_sequence_sf = sorted([d for n, d in G_sf.degree()], reverse=True)
degreeCount_sf = collections.Counter(degree_sequence_sf)
deg_sf, cnt_sf = zip(*degreeCount_sf.items())

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(deg_reg, cnt_reg, width=0.80, color="b")
plt.title("Homogeneous Network")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")

plt.subplot(1, 2, 2)
plt.loglog(deg_sf, cnt_sf, 'bo')
plt.title("Heterogeneous Network")
plt.xlabel("Degree (log)")
plt.ylabel("Number of Nodes (log)")

plt.tight_layout()
plt.show()

# %%
cmap = plt.cm.viridis
plt.figure()
sns.distplot([d - 0.5 for d in degree_sequence_sf], kde=False)
plt.savefig(
    "../paper_results/sf_degree_distplot.png"
)

plt.figure()
nx.draw_networkx(
    G_sf,
    with_labels=False,
    pos=layout.kamada_kawai_layout(G_sf),
    node_size=60,
    edge_color="gray",
    node_color=degree_sequence_sf,
)
norm = plt.Normalize(vmin=min(degree_sequence_sf), vmax=max(degree_sequence_sf))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(
    sm,
    orientation='vertical',
    label='Node Degree',
    spacing='proportional',
)  # Adjust label as needed

plt.savefig("../paper_results/sf_degree_graphplot.png")
# %%

cmap = plt.cm.viridis
plt.figure()
ax = sns.distplot(x=[d for d in degree_sequence_reg], kde=False)
ax.set_xticks([0, 1, 2, 3])
plt.savefig(
    "../paper_results/regular_degree_distplot.png"
)

plt.figure()
nx.draw_networkx(
    G_reg,
    with_labels=False,
    # pos = lo.kamada_kawai_layout(G_reg),
    # pos=lo.spiral_layout(G_reg),
    pos=layout.spring_layout(G_reg),
    node_size=60,
    # edgecolors="Red",
    node_color=degree_sequence_reg,
    edge_color="gray"
    # node_size = [data['features'][i][1]*10 for i in data['features'] if int(i) in list(G.nodes())],
    # cmap=cmap,
    # alpha = 0.5
)
norm = plt.Normalize(vmin=min(degree_sequence_reg), vmax=max(degree_sequence_reg))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, orientation='vertical', label='Node Degree')  # Adjust label as needed
plt.savefig("../paper_results/regular_degree_graphplot.png")
# %%
