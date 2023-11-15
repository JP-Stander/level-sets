#%%
import matplotlib.pyplot as plt

# Create a 3x3 grid of pixels
plt.figure()
for i in range(4):
    plt.plot([0,3], [i, i], color='black', linewidth=2)
for i in range(4):
    plt.plot([i, i], [0,3], color='black', linewidth=2)
plt.fill_between([1, 2], [1, 1], [2, 2], color='green')
plt.fill_between([0, 1], [1, 1], [2, 2], color='blue')
plt.fill_between([2, 3], [1, 1], [2, 2], color='blue')
plt.fill_between([1, 2], [0, 0], [1, 1], color='blue')
plt.fill_between([1, 2], [3, 3], [2, 2], color='blue')
plt.axis('off')
plt.show()


# %%
