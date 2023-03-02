# %%
import os
from PIL import Image
import numpy as np
import pandas as pd
from level_sets.utils import get_level_sets, spatio_environ_dependence
from level_sets.metrics import compactness, elongation
from graphical_model.utils import graphical_model
import matplotlib.pyplot as plt

img = Image.open('../mnist/img_16.jpg')
img = img.resize((10,10))
img = np.array(img)

plt.figure()
plt.imshow(img)
plt.show(block=True)

# %%
import igraph as ig
nodes, edges = graphical_model(img)
