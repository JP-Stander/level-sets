#%%
import os
import torch
from networkx import Graph
import networkx as nx
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from tqdm import tqdm
from images.utils import load_image
from graphical_model.utils import graphical_model
from graphical_model_cython import build_graph
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split

class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Graph-level representation
        x = self.fc(x)
        return F.softmax(x, dim=1)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return correct / len(test_graphs)
# %
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

def img_to_graph(image, delta):
    img_size = 50
    img = load_image(
        image,
        [img_size, img_size]
    )

    img_float = img.astype(np.float64)
    nodes_ls, edges_ls, attr_ls = graphical_model(
        img=img_float,
        return_spp=True,
        alpha=0.5,
        set_type="fuzzy",
        fuzzy_cutoff=delta
    )

    g1 = make_graph(nodes_ls, edges_ls, attr_ls)
    return g1
#%

def run_experiment(delta):
    def train(model, optimizer, criterion, train_loader):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    images = [f"../dtd/images/dotted/{file}" for file in os.listdir("../dtd/images/dotted")]
    images += [f"../dtd/images/fibrous/{file}" for file in os.listdir("../dtd/images/fibrous")]
    delta = 10
    graphs = []

    for image in tqdm(images[:10]):
        G = img_to_graph(image, delta)
        node_features = []
        for _, data in G.nodes(data=True):
            features = [data[attr] for attr in data]
            node_features.append(features[3:7])
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_indices = []
        for edge in G.edges():
            source, target = edge
            edge_indices.append([int(source), int(target)])
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        graph_label = image.split("/")[-1].split("_")[0]
        data = Data(x=node_features, edge_index=edge_indices, y=graph_label)
        graphs.append(data)

    # %%Collect all the labels
    labels = [data.y for data in graphs]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    # Update graphs with numerical labels
    for i, data in enumerate(graphs):
        data.y = torch.tensor([encoded_labels[i]], dtype=torch.long)

    # %%
    model = GNN(input_dim=4, output_dim=len(encoder.classes_))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32)

    for epoch in range(10):
        train_loss = train(model, optimizer, criterion, train_loader) 
        # val_loss = validate()
        # print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {1}")

    test_accuracy = test()
    return test_accuracy
    # print(f"Test accuracy is {test_accuracy}")

def main():
    # Set a range for delta
    delta_values = [5, 10]  # modify this as per your requirements

    # Determine the number of cores to use (we'll use 10 cores as you specified)
    num_cores = 4

    # Create a Pool of processes
    with Pool(num_cores) as p:
        accuracies = p.map(run_experiment, delta_values)

    # Combine delta_values and their accuracies into a dictionary
    results = dict(zip(delta_values, accuracies))
    
    print(results)

    best_delta = max(results, key=results.get)
    print(f"The best delta value is {best_delta} with an accuracy of {results[best_delta]}")

if __name__ == "__main__":
    main()
# %%
