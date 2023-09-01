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
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from tqdm import tqdm


# %%
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

# Best
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

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# criterion = torch.nn.CrossEntropyLoss()

# Accuracy for pixels: 0.3958333333333333
# Accuracy for fuzzy_sets_10: 0.8958333333333334 (0.875)
# Accuracy for fuzzy_sets_30: 0.375
# Accuracy for level_sets: 0.7916666666666666

train_size = 0.8
for dataset_type in ["pixels", "fuzzy_sets_10", "fuzzy_sets_30", "level_sets"]:
    graphs = all_graphs[dataset_type]

    # random.shuffle(graphs)
    train_number = int(train_size*len(graphs))
    train_graphs = graphs[:train_number]
    test_graphs = graphs[train_number: ]

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32)

    if dataset_type == "pixels":
        model = GNN(input_dim=1, output_dim=len(encoder.classes_))
    else:
        model = GNN(input_dim=4, output_dim=len(encoder.classes_))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(100)):
        train_loss = train()

    test_accuracy = test()
    print(f"Accuracy for {dataset_type}: {test_accuracy}")
print(model)
print(optimizer)
# %%
