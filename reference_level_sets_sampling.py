# %%
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Load the data
sets = pd.read_csv('../unique_level_sets/sets_2_to_8_8conn.csv')
# sets = sets.iloc[:1000, :]
# Reduce the columns
sets = sets[['size', 'compactness', 'elongation', 'width_to_height', 'angle']]

# %%
# Standardize the dataset
mean = sets.mean(axis=0)
std = sets.std(axis=0)
std_sets = (sets - mean) / std

# Constants
n_clusters = 100
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

## Standardised data
# 1. TSNE transformation of standardised data
tsne_on_std_data = tsne.fit_transform(std_sets)

# 2. K-means on t-SNE results of standardised data
kmeans_tsne_std = kmeans.fit(tsne_on_std_data)
cluster_tsne_std = kmeans_tsne_std.cluster_centers_

# 3. K-means clustering on standardized data, then visualize using t-SNE
kmeans_tsne_unstd = kmeans.fit(std_sets)
std_centroids = kmeans_tsne_unstd.cluster_centers_
clust_std_tsne = tsne.fit_transform(std_centroids)


## Unstandardised data
# 4. TSNE transformation of unstandardised data
tsne_on_unstd_data = tsne.fit_transform(sets)

# 5. K-means on t-SNE results of unstandardised data
kmeans_tsne_unstd = kmeans.fit(tsne_on_unstd_data)
cluster_tsne_unstd = kmeans_tsne_unstd.cluster_centers_

# 6. K-means clustering on unstandardized data, then visualize using t-SNE
kmeans_og_unstd = kmeans.fit(sets)
unstd_centroids = kmeans_og_unstd.cluster_centers_
clust_unstd_tsne = tsne.fit_transform(unstd_centroids)

# %%Plotting
# For t-SNE
plt.figure()
sns.scatterplot(
    x=tsne_on_std_data[:,0], y=tsne_on_std_data[:,1],
    color="red", s=10, alpha=0.3, marker="x",
    label = "TSNE transformation on standardised data"
)
sns.scatterplot(
    x=cluster_tsne_std[:, 0], y=cluster_tsne_std[:, 1],
    color="yellow", s=50, alpha=1, marker = "o",
    label = "KMeans on TSNE transformation of standardised data"
)
sns.scatterplot(
    x=clust_std_tsne[:,0], y=clust_std_tsne[:,1],
    color="orange", s=50, alpha=1, marker = "v",
    label = "Cluster Standardised Data and transform via TSNE"
)
plt.title("t-SNE visualization (Standardised Data)")
plt.show()

plt.figure()
sns.scatterplot(
    x=tsne_on_unstd_data[:,0], y=tsne_on_unstd_data[:,1],
    color="blue", s=10, alpha=0.3, marker="x",
    label = "TSNE transformation on unstandardised data"
)
sns.scatterplot(
    x=cluster_tsne_unstd[:, 0], y=cluster_tsne_unstd[:, 1],
    color="green", s=50, alpha=1, marker = "o",
    label = "KMeans on TSNE transformation of unstandardised data"
)
sns.scatterplot(
    x=clust_unstd_tsne[:,0], y=clust_unstd_tsne[:,1],
    color="purple", s=50, alpha=1, marker = "v",
    label = "Cluster unstandardised Data and transform via TSNE"
)
plt.title("t-SNE visualization (Unstandardised Data)")
plt.show()

# %%
from sklearn.decomposition import PCA

# Standardize the dataset
mean = sets.mean(axis=0)
std = sets.std(axis=0)
std_sets = (sets - mean) / std

# Constants
n_clusters = 100
pca = PCA(n_components=2)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

## Standardised data
# 1. PCA transformation of standardised data
pca_on_std_data = pca.fit_transform(std_sets)

# 2. K-means on PCA results of standardised data
kmeans_pca_std = kmeans.fit(pca_on_std_data)
cluster_pca_std = kmeans_pca_std.cluster_centers_

# 3. K-means clustering on standardized data, then visualize using PCA
kmeans_pca_unstd = kmeans.fit(std_sets)
std_centroids = kmeans_pca_unstd.cluster_centers_
clust_std_pca = pca.fit_transform(std_centroids)


## Unstandardised data
# 4. PCA transformation of unstandardised data
pca_on_unstd_data = pca.fit_transform(sets)

# 5. K-means on PCA results of unstandardised data
kmeans_pca_unstd = kmeans.fit(pca_on_unstd_data)
cluster_pca_unstd = kmeans_pca_unstd.cluster_centers_

# 6. K-means clustering on unstandardized data, then visualize using PCA
kmeans_og_unstd = kmeans.fit(sets)
unstd_centroids = kmeans_og_unstd.cluster_centers_
clust_unstd_pca = pca.fit_transform(unstd_centroids)

# Plotting
# For PCA
plt.figure()
sns.scatterplot(
    x=pca_on_std_data[:,0], y=pca_on_std_data[:,1],
    color="red", s=10, alpha=0.3, marker="x",
    label = "PCA transformation on standardised data"
)
sns.scatterplot(
    x=cluster_pca_std[:, 0], y=cluster_pca_std[:, 1],
    color="yellow", s=50, alpha=1, marker = "o",
    label = "KMeans on PCA transformation of standardised data"
)
sns.scatterplot(
    x=clust_std_pca[:,0], y=clust_std_pca[:,1],
    color="orange", s=50, alpha=1, marker = "v",
    label = "Cluster Standardised Data and transform via PCA"
)
plt.title("PCA visualization (Standardised Data)")
plt.show()

plt.figure()
sns.scatterplot(
    x=pca_on_unstd_data[:,0], y=pca_on_unstd_data[:,1],
    color="blue", s=10, alpha=0.3, marker="x",
    label = "PCA transformation on unstandardised data"
)
sns.scatterplot(
    x=cluster_pca_unstd[:, 0], y=cluster_pca_unstd[:, 1],
    color="green", s=50, alpha=1, marker = "o",
    label = "KMeans on PCA transformation of unstandardised data"
)
sns.scatterplot(
    x=clust_unstd_pca[:,0], y=clust_unstd_pca[:,1],
    color="purple", s=50, alpha=1, marker = "v",
    label = "Cluster unstandardised Data and transform via PCA"
)
plt.title("PCA visualization (Unstandardised Data)")
plt.show()

# %%
