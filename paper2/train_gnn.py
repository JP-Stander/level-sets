# %%
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv, global_mean_pool
# %%
# List all the json files
dataset_type = "fuzzy_sets_30"
for dataset_type in ["fuzzy_sets_30", "fuzzy_sets_10", "level_sets"]:
    graph_files = [f"../graphical_models/{dataset_type}/{dir}" for dir in os.listdir("../graphical_models/level_sets")]
    graphs = []
    for file in graph_files:
        G = nx.read_graphml(file)
        # ... extract node and edge features ...
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
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool

    class GNN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)  # Graph-level representation
            x = self.fc(x)
            return F.softmax(x, dim=1)

    # Given your data's node features have dimension 4
    model = GNN(input_dim=4, hidden_dim=64, output_dim=len(encoder.classes_))

    # %%
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()


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
    print(f"{k-1} fold accuracies for {dataset_type} are {accs}")
    # %%
    # import random

    # # Shuffle and split
    # random.shuffle(graphs)
    # train_size = int(0.8 * len(graphs))
    # val_size = int(0.1 * len(graphs))
    # train_graphs = graphs[:train_size]
    # val_graphs = graphs[train_size:train_size+val_size]
    # test_graphs = graphs[train_size+val_size:]

    # train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_graphs, batch_size=32)
    # test_loader = DataLoader(test_graphs, batch_size=32)

    # train_losses = []
    # val_losses = []
    # test_acc = []

    # for epoch in range(100):
    #     train_loss = train()
    #     val_loss = validate()
    #     test_accuracy = test()

    #     train_losses.append(train_loss)
    #     val_losses.append(val_loss)
    #     test_acc.append(test_accuracy)

    #     print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # test_accuracy = test()
    # print(f"Test Accuracy: {test_accuracy}")

    # # %%
    # epochs = range(1, len(train_losses) + 1)

    # # Plotting
    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, train_losses, 'r', label='Training loss')
    # plt.plot(epochs, val_losses, 'g', label='Validation loss')

    # ax = plt.gca()  # Get the current axis
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.savefig(f"../paper_results/test_val_loss_{dataset_type}.png", bbox_inches='tight')

# %%
