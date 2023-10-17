#%%
import os
import sys
import numpy as np
from networkx import Graph, write_graphml
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image
from graphical_model.utils import graphical_model

#%%
def load_data_from_npy(experiment_loc, key, stage="train"):
    if os.path.exists(f"{experiment_loc}/{key}_{stage}_data.npy"):
        data = np.load(f"{experiment_loc}/{key}_{stage}_data.npy", allow_pickle=True)
        indices = np.load(f"{experiment_loc}/{key}_{stage}_indices.npy")
        
        list_of_arrays = [data[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
        
        return list_of_arrays
    else:
        return []

def make_graph(nodes, edges, attrs, d=0.005):
    g = Graph()
    edges = (edges > d) * edges
    #  Add nodes
    for index, row in nodes.iterrows():
        g.add_node(index)

    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j] != 0:
                g.add_edge(i, j, weight=edges[i, j])

    for index, row in attrs.iterrows():
        for col, attr_value in row.items():
            g.nodes[index][col] = attr_value

    for node, data in g.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = ','.join(map(str, value))
    return g

def img_to_graph(image, graph_location, d=10, img_size=100, connectivity=8, edge_cut_off=0.005, return_graph=False, metric_names='all', trim=None):
    img = load_image(
        image,
        [img_size, img_size],
        trim=trim
    )

    nodes_fs, edges_fs, attr_fs = graphical_model(
        img=img,
        return_spp=True,
        alpha=0.5,
        set_type="fuzzy",
        fuzzy_cutoff=d,
        metric_names=metric_names,
        connectivity = connectivity
    )

    g = make_graph(nodes_fs, edges_fs, attr_fs, edge_cut_off)
    if return_graph is True:
        return g
    else:
        write_graphml(g, f"{graph_location}/{image.split('/')[-1].split('.')[0]}_graph.graphml")

def get_img_nea(image, d=10, img_size=100, connectivity=8, metric_names="all", trim=None):
    img = load_image(
        image,
        [img_size, img_size],
        trim = trim
    )

    nodes_fs, edges_fs, attr_fs = graphical_model(
        img=img,
        return_spp=True,
        alpha=0.5,
        set_type="fuzzy",
        fuzzy_cutoff=d,
        metric_names=metric_names,
        connectivity = connectivity
    )
    return nodes_fs, edges_fs, attr_fs

def image_to_histogram(descriptors, kmeans):
    hist = np.zeros(kmeans.n_clusters)
    labels = kmeans.predict(descriptors)
    for label in labels:
        hist[label] += 1
    return hist


def process_sublist(sublist_descriptors, sublist_add_features, kmeans):
    # Convert each descriptor to histogram
    histograms = [image_to_histogram(desc, kmeans) for desc in sublist_descriptors]
    
    # Concatenate histograms with the additional features
    full = [np.concatenate((hist, add_feat)) for hist, add_feat in zip(histograms, sublist_add_features)]
    
    return full