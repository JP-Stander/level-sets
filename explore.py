# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from graphical_model.utils import graphical_model


img = Image.open('../mnist/img_16.jpg')
img = img.resize((10, 10))
img = np.array(img)

plt.figure()
plt.imshow(img)
plt.show(block=True)

# %%
nodes, edges = graphical_model(img)
