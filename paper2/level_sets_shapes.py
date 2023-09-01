# %%
import numpy as np
from matplotlib import pyplot as plt

img = np.array([
    [1,1,1,13,4],
    [1,1,1,4,4],
    [1,1,4,4,9],
    [7,4,4,3,10],
    [7,4,8,3,11]
])

plt.figure()
plt.imshow(img, 'rainbow')
plt.xticks([])
plt.yticks([])
plt.vlines([-0.5, 0.5, 1.5 ,2.5, 3.5, 4.5], -0.5, 4.5, 'black', linewidth=0.5)
plt.hlines([-0.5, 0.5, 1.5 ,2.5, 3.5, 4.5], -0.5, 4.5, 'black', linewidth=0.5)
plt.savefig("..//paper_results//level_sets_shapes_example.png", bbox_inches='tight')

# %%
import os
os.listdir("..//")
# %%

from level_sets.metrics import compactness, elongation

img = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,1,1,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0],
])
print(compactness(img))
print(1/elongation(img))

img = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])
print(compactness(img))
print(1/elongation(img))

    # %%
