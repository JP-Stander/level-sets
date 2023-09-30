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
def _make_graph(nodes, edges, attrs, d=0.005):
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

def img_to_graph(image, graph_location, d=10, img_size=100):
    img = load_image(
        image,
        [img_size, img_size]
    )

    nodes_fs, edges_fs, attr_fs = graphical_model(
        img=img,
        return_spp=True,
        alpha=0.5,
        set_type="fuzzy",
        fuzzy_cutoff=d,
    )

    g = _make_graph(nodes_fs, edges_fs, attr_fs)
    write_graphml(g, f"{graph_location}/{image.split('/')[-1].split('.')[0]}_graph.graphml")
