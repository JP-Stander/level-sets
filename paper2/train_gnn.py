# %%
import os
import torch
import networkx as nx
import random
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm
# %%

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

# %%
all_graphs = {}

for dataset_type in ["pixels", "fuzzy_sets_10", "fuzzy_sets_30", "level_sets"]:
    graph_files = [f"../graphical_models/{dataset_type}/{dir}" for dir in os.listdir("../graphical_models/level_sets")]

    graphs_og = []
    for file in graph_files:
        G = nx.read_graphml(file)
        # ... extract node and edge features ...
        node_features = []
        for _, data in G.nodes(data=True):
            features = [data[attr] for attr in data]
            if dataset_type == "pixels":
                node_features.append(features)
            else:
                node_features.append(features[3:7])
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_indices = []
        for edge in G.edges():
            source, target = edge
            edge_indices.append([int(source), int(target)])
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        graph_label = file.split("/")[-1].split("_")[0]
        data = Data(x=node_features, edge_index=edge_indices, y=graph_label)
        graphs_og.append(data)
    # Collect all the labels
    labels = [data.y for data in graphs_og]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    # Update graphs with numerical labels
    for i, data in enumerate(graphs_og):
        data.y = torch.tensor([encoded_labels[i]], dtype=torch.long)
    all_graphs[dataset_type] = graphs_og
# %%

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
        return F.log_softmax(x, dim=1)


for dataset_type in ["fuzzy_sets_30", "fuzzy_sets_10", "level_sets", "pixels"]:
    graphs = all_graphs[dataset_type]

    # Given your data's node features have dimension 4
    if dataset_type == "pixels":
        model = GNN(input_dim=1, output_dim=len(encoder.classes_))
    else:
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
        # val_loader = DataLoader(val_graphs, batch_size=32)
        test_loader = DataLoader(test_graphs, batch_size=32)
        # Actual training loop
        for epoch in range(100):
            train_loss = train()
            # val_loss = validate()
            # print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {1}")

        test_accuracy = test()
        accs.append(test_accuracy)
        # print(f"Test Accuracy for fold {k}: {test_accuracy}")
        k += 1
        break
    print(f"{k-1} fold accuracies for {dataset_type} are {accs}\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n\n")

print(model)
print(optimizer)

# %%

import numpy as np
import matplotlib.pyplot as plt
import time
# Assuming necessary imports for your frameworks are already made

losses = {}

for dataset_type in ["fuzzy_sets_30", "fuzzy_sets_10", "level_sets", "pixels"]:
    graphs = all_graphs[dataset_type]

    # Given your data's node features have dimension 4
    if dataset_type == "pixels":
        model = GNN(input_dim=1, output_dim=len(encoder.classes_))
    else:
        model = GNN(input_dim=4, output_dim=len(encoder.classes_))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    # Split the dataset into train, validation, and test
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:val_size+train_size]
    test_graphs = graphs[val_size+train_size:]

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)
    
    train_losses = []
    val_losses = []

    # Actual training loop
    start_time = time.time()
    for epoch in range(100):
        train_loss = train()
        val_loss = validate()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    end_time = time.time()
    print(f"Training time for {dataset_type}: {end_time-start_time} seconds")
    losses[dataset_type] = {'train': train_losses, 'val': val_losses}

    test_accuracy = test()
    print(f"Test Accuracy for {dataset_type}: {test_accuracy}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n\n")

# %% Plotting the losses
for dataset_type, loss_data in losses.items():
    plt.figure()
    plt.plot(loss_data['train'][2:], label="Train Loss", c="red")
    plt.plot(loss_data['val'][2:], label="Validation Loss", c="green")
    plt.title(f"Loss curves for {dataset_type}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(f"../paper_results/test_val_loss_{dataset_type}")

# %%

plt.figure()
plt.plot(losses["level_sets"]['train'][2:], label="Train Loss", c="red", linestyle="-")
plt.plot(losses["level_sets"]['val'][2:], label="Validation Loss", c="green", linestyle="-")
plt.plot(losses["fuzzy_sets_10"]['train'][2:], label="Train Loss", c="red", linestyle="-.")
plt.plot(losses["fuzzy_sets_10"]['val'][2:], label="Validation Loss", c="green", linestyle="-.")
plt.plot(losses["fuzzy_sets_30"]['train'][2:], label="Train Loss", c="red", linestyle="--")
plt.plot(losses["fuzzy_sets_30"]['val'][2:], label="Validation Loss", c="green", linestyle="--")
plt.title(f"Loss curves for {dataset_type}")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
