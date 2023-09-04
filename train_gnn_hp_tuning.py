#%%
import os
import torch
from networkx import Graph
import networkx as nx
from sklearn.model_selection import KFold
import random
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from torch_geometric.data import Data
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from tqdm import tqdm
from images.utils import load_image
from graphical_model.utils import graphical_model
from graphical_model_cython import build_graph
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def validate(model, criterion, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def test(model, test_loader, test_graphs):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return correct / len(test_graphs)

def run_experiment(delta):

    graph_files = [f"../graphical_models/fuzzy_sets_{delta}/{dir}" for dir in os.listdir(f"../graphical_models/fuzzy_sets_{delta}")]

    graphs = []
    for file in graph_files:
        G = nx.read_graphml(file)
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
        graph_label = file.split("/")[-1].split("_")[0]
        data = Data(x=node_features, edge_index=edge_indices, y=graph_label)
        graphs.append(data)
    # Collect all the labels
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
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_splits = []
    k=1
    accs =[]
    for train_index, test_index in kf.split(graphs):
        train_graphs = [graphs[i] for i in train_index]
        test_graphs = [graphs[i] for i in test_index]

        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=32)
        # Actual training loop
        for epoch in range(100):
            train_loss = train(model, optimizer, criterion, train_loader)

        test_accuracy = test(model, test_loader, test_graphs)
        accs.append(test_accuracy)
    result = {"accuracies": accs}
    with open(f'../graphical_models/results/{delta}.json', 'w') as f:
        json.dump(result, f)
    


if __name__ == "__main__":
    ds = [i for i in range(26)] + [30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 255]
    ds = ds[:5]
    n_cores = cpu_count() - 2
    print(f'Running process on {n_cores} cores')

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Setup tqdm
        future_to_detail = {(executor.submit(run_experiment, d)): (d) for d in ds}

        for future in tqdm(as_completed(future_to_detail), total=len(ds)):
            try:
                future.result()  # retrieve results if there are any
            except Exception as e:
                d = future_to_detail[future]
                print(f"Error processing set with delta {d}. Error: {e}")
