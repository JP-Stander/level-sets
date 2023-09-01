# %%
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.manifold import TSNE
import ot #cite package if usefull https://github.com/PythonOT/POT
from level_sets.sampler.sinkhorn import sinkhorn_sampler

# %
# Read in all unique level-sets
X_og = pd.read_csv('../unique_level_sets/sets_2_to_8_8conn.csv')
X = X_og.copy()

# Normalise X
X['size'] /= max(X['size'])
X['width_to_height'] /= max(X['width_to_height'])
X['angle'] = (X['angle'] - min(X['angle'])) / (max(X['angle']) - min(X['angle']))
X['intensity'] /= max(X['intensity'])

m, d = X.shape
n = 5
num_iterations = 2

# Weights
a, b = np.ones((m,)) / m, np.ones((n,)) / n

lambda_param = 0.01  # regularization parameter (figure this out)

# %

# Call the Cython function
Y = sinkhorn_sampler(X.values, n, num_iterations, a, b, lambda_param)


# %%
Y = X.iloc[np.random.choice(m, n, replace=False),:]

for iteration in tqdm(range(num_iterations)):
    # Compute distance matrix
    C = distance_matrix(X, Y)
    
    # Compute transport plan
    P = ot.sinkhorn(a, b, C, lambda_param)
    # Update Y based on transport plan
    Y_new = np.zeros_like(Y)
    for j in range(n):
        Y_new[j] = P[:, j].dot(X) / np.sum(P[:, j])  # weighted average
    if sum(sum(Y-Y_new)) < 0.001:
        break
    Y = Y_new
# %%
