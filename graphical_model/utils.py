# %%
# import os
from shapely.geometry import MultiPoint, mapping
# from PIL import Image
# import numpy as np
# os.chdir(r"/home/qxz1djt/projects/phd/level-sets")
# img = Image.open('../mnist/img_16.jpg')
# img = img.resize((10, 10))
# img = np.array(img)

# alpha = 0.5
# normalize_gray = True
# ls_spatial_dist = 'euclidean'
# ls_attr_dist = 'cityblock'
import os
import sys
import numpy as np
import pandas as pd
import igraph as ig

current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_script_directory)
from level_sets.metrics import get_metrics  # noqa: E402
from level_sets.utils import get_level_sets, get_fuzzy_sets  # noqa: E402
import spatio_distance  # noqa: E402

# %%
# img = Image.open('../mnist/img_16.jpg')
# img = img.resize((50, 50))
# img = np.array(img)
# %%
# Defining the image's level-sets as a spatial point pattern
# For now the location of the level-set is the mean location of all pixels in the set
# spp = [id, x, y, set-value, set-size]


def graphical_model(
    img,
    return_spp=False,
    set_type="level",
    fuzzy_cutoff=20,
    normalise_pixel_index=False,
    connectivity=4,
    alpha=0.5,
    normalise_gray=True,
    size_proportion=False,
    ls_spatial_dist="euclidean",
    ls_attr_dist="cityblock",
    centroid_method="mean",
    metric_names = "all"
):

    should_normalise = normalise_gray and img.max() > 1
    level_sets = get_level_sets(img, connectivity=connectivity / 4) if set_type == "level" else \
        get_fuzzy_sets(img, fuzzy_cutoff, connectivity) + 1
    uni_level_sets = pd.unique(level_sets.flatten())
    results = []
    width, height = img.shape

    for ls in uni_level_sets:
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        level_set = (level_sets == ls) * ls
        set_value = np.mean([img[s] for s in subset])
        # if set_value < 56:
        #     continue
        set_size = len(subset) / (width * height) if size_proportion else len(subset)
        points = MultiPoint(subset)
        centroid = points.centroid if centroid_method == "mean" else points.representative_point()
        centroid = mapping(centroid)['coordinates']
        intensity = set_value / 255 if should_normalise else set_value

        metrics = get_metrics(level_set, metric_names=metric_names, img_size=[width, height])
        result = {**metrics, **{
            "level-set": ls,
            "x-coor": centroid[0] / width if normalise_pixel_index else centroid[0],
            "y-coor": centroid[1] / width if normalise_pixel_index else centroid[1],
            "intensity": intensity,
            "size": set_size,
            "pixel_indices": subset
        }}
        results.append(result)

    spp = pd.DataFrame(results)
    nodes = spp.loc[:, ["x-coor", "y-coor"]]
    edges = np.asarray(
        spatio_distance.calculate_distance(
            np.array(spp[["x-coor", "y-coor"]+list(metrics.keys())]),
            ls_spatial_dist,
            ls_attr_dist,
            alpha
        )
    )

    if return_spp:
        return nodes, edges, spp
    return nodes, edges


def calculate_graph_attributes(graph):
    attributes = {}

    # Basic properties
    attributes["number_of_nodes"] = graph.vcount()
    attributes["number_of_edges"] = graph.ecount()
    attributes["is_connected"] = graph.is_connected()
    attributes["diameter"] = graph.diameter()

    # Degree distribution
    degree_sequence = sorted(graph.degree(), reverse=True)
    attributes["maximum_degree"] = degree_sequence[0]
    attributes["minimum_degree"] = degree_sequence[- 1]
    attributes["average_degree"] = sum(degree_sequence) / len(degree_sequence)

    # Centrality measures
    attributes["betweenness_centrality"] = np.mean(graph.betweenness())
    attributes["closeness_centrality"] = np.mean(graph.closeness())
    attributes["eigenvector_centrality"] = np.mean(graph.eigenvector_centrality())

    # Clustering coefficients
    attributes["global_clustering_coefficient"] = graph.transitivity_undirected()
    attributes["average_clustering_coefficient"] = graph.transitivity_avglocal_undirected()

    # Community detection
    communities = graph.community_edge_betweenness(directed=False)
    membership = communities.as_clustering()
    attributes["modularity"] = membership.modularity
    attributes["number_of_communities"] = membership.n

    # Path-related measures
    attributes["average_shortest_path_length"] = graph.average_path_length()
    attributes["shortest_paths"] = np.min(graph.shortest_paths_dijkstra())

    # Assortativity
    attributes["assortativity_coefficient"] = graph.assortativity_degree()

    return attributes


# TODO: Fix complexity
def resize_graph(graph, n):  # noqa: C901
    """
    Resize the input graph g to contain n nodes. If n is less than the number of nodes in g,
    the nodes with the highest edges between them are clustered together. If n is greater
    than the number of nodes in g, the biggest nodes are split up into smaller nodes.
    """
    g = graph.copy()
    # If n is less than the number of nodes in g, cluster nodes together
    if n < g.vcount():
        while g.vcount() > n:
            # Compute the edge betweenness of the graph
            eb = g.edge_betweenness()
            # Find the edge with the highest betweenness
            max_eb = max(eb)
            max_eb_edge = g.es[eb.index(max_eb)]
            # Get the nodes connected by the edge with the highest betweenness
            node1, node2 = max_eb_edge.tuple
            # Cluster the nodes together
            new_node = g.vcount()
            g.add_vertex()
            # Inherit the edges of the nodes being clustered
            for neighbor in g.neighbors(node1):
                if neighbor != node2:
                    g.add_edge(new_node, neighbor)
            for neighbor in g.neighbors(node2):
                if neighbor != node1:
                    g.add_edge(new_node, neighbor)
            # Remove the nodes being clustered
            g.delete_vertices([node1, node2])

    # If n is greater than the number of nodes in g, split the biggest nodes
    elif n > g.vcount():
        while g.vcount() < n:
            g2 = g.copy()
            # Find the node with the highest n attribute
            max_n = max(g.vs["size"])
            max_n_node = g.vs.find(size=max_n)
            # Split the node into two

            g2.add_vertices(2)
            new_nodes = list(set(g2.vs.indices) - set(g.vs.indices))
            for attr in max_n_node.attributes().keys():
                g2.vs[new_nodes[0]][attr] = max_n_node[attr]
                g2.vs[new_nodes[1]][attr] = max_n_node[attr]
            g2.vs[new_nodes[0]]["size"] = max_n // 2
            g2.vs[new_nodes[1]]["size"] = max_n - max_n // 2
            # Inherit the edges of the original node
            for neighbor in g.neighbors(max_n_node):
                g2.add_edge(new_nodes[0], neighbor)
                g2.add_edge(new_nodes[1], neighbor)
            # Add a strong edge between the two new nodes
            g2.add_edge(new_nodes[0], new_nodes[1], strength=10)
            # Remove the original node
            g2.delete_vertices(max_n_node)
            g = g2.copy()
    return g


def find_graphlets(g, k):
    subgraphs = ig.GraphBase.get_subisomorphisms_vf2(g, ig.GraphBase(k, directed=False))

    # count the occurrences of each subgraph in the original graph
    graphlet_counts = [0] * len(subgraphs)
    for i in range(g.vcount()):
        for j in range(i + 1, g.vcount()):
            subgraph = g.subgraph([i, j])
            if subgraph.ecount() == k:
                for idx, sg in enumerate(subgraphs):
                    if subgraph.isomorphic(sg):
                        graphlet_counts[idx] += 1

    # print the resulting counts for each subgraph
    for idx, sg in enumerate(subgraphs):
        print("Graphlet {}: {}".format(idx, graphlet_counts[idx]))

# %%
