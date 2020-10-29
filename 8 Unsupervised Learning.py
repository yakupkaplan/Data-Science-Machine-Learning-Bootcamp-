# UNSUPERVISED LEARNING

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# EDA

os.getcwd()
os.chdir("C:\\Users\\yakup\\PycharmProjects\\dsmlbc")

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df.head()

df.isnull().sum()
df.info()
df.describe().T


# DATA PREP

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]


# K-MEANS

kmeans = KMeans(n_clusters=4)
k_fit = kmeans.fit(df)
k_fit
dir(k_fit)

k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
df[0:5]

# Visualization of Clusters
k_means = KMeans(n_clusters=2).fit(df)
clusters = k_means.labels_
type(df)
df = pd.DataFrame(df)

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=clusters, s=50, cmap="viridis")
plt.show()

# Show centorids
centroids = k_means.cluster_centers_
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=clusters, s=50, cmap="viridis")
plt.scatter(centroids[:, 0], centroids[:, 1], c="black", s=200, alpha=0.5)
plt.show()

# 3-D Visualization

from mpl_toolkits.mplot3d import Axes3D

kmeans = KMeans(n_clusters=3)
k_fit = kmeans.fit(df)
clusters = k_fit.labels_
centroids = kmeans.cluster_centers_

plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])
plt.show()

# Show centorids
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=clusters)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='#050505', s=1000);
plt.show()

# Finding the optimum number of clusters
kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

ssd

# Elbow Method
plt.plot(K, ssd, "bx-")
plt.xlabel("Total Within Sum of Squares")
plt.title("Elbow Method for Optimal Value of Clusters")
plt.show()


# An automized method:
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k=(2, 20))
visu.fit(df)
visu.show()

kmeans = KMeans(n_clusters=6).fit(df)
clusters = kmeans.labels_

# Creating a dataframe
df = pd.read_csv("datasets/USArrests.csv", index_col=0)
pd.DataFrame({"States": df.index, "Clusters": clusters})
df["cluster_no"] = clusters
df.head()

df["cluster_no"] = df["cluster_no"] + 1

df.groupby("cluster_no").agg({"cluster_no": "count"})
df.groupby("cluster_no").agg(np.mean)

df[df["cluster_no"] == 4]


# HIERARCHICAL CLUSTERING

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

# Scaling
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")

# Plotting dendogram
plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendograms")
plt.xlabel("Observation Units")
plt.ylabel("Distances")
dendrogram(hc_average, leaf_font_size=10)
plt.show()


plt.figure(figsize=(15, 10))
plt.title("Hierarchical Clustering Dendograms")
plt.xlabel("Observation Units")
plt.ylabel("Distances")
dendrogram(hc_complete, truncate_mode="lastp", p=10, show_contracted=True, leaf_font_size=10)
plt.show()


hc_average
dir(hc_average)


# PRINCIPAL COMPONENT ANALYSIS

df = pd.read_csv("datasets/Hitters.csv")
df.head()
df.shape

num_cols = [col for col in df.columns if df[col].dtypes != "O"]
df = df[num_cols]
df.dropna(inplace=True)
df.shape

from sklearn.preprocessing import StandardScaler
df = StandardScaler().fit_transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca_fit = pca.fit_transform(df)
pca_fit

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


# Number of principal component
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Ratio of explained variance")
plt.show()

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
pca.explained_variance_ratio_

pca_fit.shape
df.shape



