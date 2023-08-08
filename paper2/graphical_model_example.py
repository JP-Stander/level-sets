# %%
import glob
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from level_sets.utils import load_image
from graphical_model.utils import graphical_model, calculate_graph_attributes

img = load_image("../mnist/img_45.jpg")

gm = graphical_model(
    img,
    True,
    0.8,
    True
)
# %%
nodes, edges, spp = gm
# convert edges to matrix
edges_cut = np.where(edges > 0.3, edges, 0) * 3

# create graph
g = ig.Graph.Adjacency(
    edges_cut.tolist(),
    mode='undirected',
    attr='weight',
    loops=False
)

for i in range(spp.shape[0]):
    for attr in list(spp):
        g.vs[i][attr] = spp.loc[i, attr]

# plot graph
layout = [(x, y) for x, y in zip(g.vs['x-coor'], g.vs['y-coor'])]
ig.plot(
    g,
    vertex_size=15,
    vertex_color=[(i, i, i) for i in g.vs['intensity']],
    # layout=layout,
    # vertex_frame_color='black',
    # vertex_frame_width=2,
)
# %%
dataframe = pd.DataFrame()

graph_attr = calculate_graph_attributes(g)
graph_attr_pd = pd.DataFrame.from_dict(graph_attr, orient='index').T
dataframe = pd.concat([dataframe, graph_attr_pd], ignore_index=True)


# %%


images = glob.glob("../mnist/trainingSample/**/*.jpg")
dataframe = pd.DataFrame()

for image in tqdm(images):
    if "Zebra" in image:
        continue
    img = load_image(image, [10, 10])

    gm = graphical_model(
        img,
        True,
        0.2,
        True
    )

    nodes, edges, spp = gm
    # convert edges to matrix
    edges_cut = np.where(edges > 0.3, edges, 0) * 10

    # create graph
    g = ig.Graph.Adjacency(
        edges_cut.tolist(),
        mode='undirected',
        attr='weight',
    )
    g = g.simplify(combine_edges='sum')

    graph_attr = calculate_graph_attributes(g)
    graph_attr_pd = pd.DataFrame.from_dict(graph_attr, orient='index').T
    graph_attr_pd['label'] = image.split('/')[-2]
    dataframe = pd.concat([dataframe, graph_attr_pd], ignore_index=True)

dataframe = dataframe.dropna(axis=1, how='all')
dataframe.to_csv("RectData.csv")
X = dataframe.drop('label', axis=1)
y = dataframe['label']

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a decision tree classifier
clf = DecisionTreeClassifier()

# train the classifier on the training data
clf.fit(X_train, y_train)

# evaluate the classifier on the test data
accuracy = clf.score(X_test, y_test)
print("Accuracy on test set:", accuracy)

# %%
