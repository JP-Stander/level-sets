#%%

from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import seaborn as sns

data = pd.read_csv("subgraphs_coordinates.csv")

images = pd.unique(data['image_name'])
image_name = "dotted_0001"
image_type = image_name.split("_")[0]
image_location = f"../dtd/images/{image_type}/{image_name}.jpg"

img = Image.open(image_location).convert('L')
img = img.resize([10,10])
img = np.array(img)

data = data.loc[data['image_name']==image_name,:]

plt.figure
plt.imshow(img, "gray")
sns.scatterplot(data=data, x='x', y='y', hue='graphlet_name', s=3)

# plt.figure()
# # plt.imshow(img, "gray")
# plt.scatter(
#     x=data['x'], 
#     y=data['y'], 
#     s=3, 
#     c=data["graphlet_name"]
# )
plt.show()
# %%
