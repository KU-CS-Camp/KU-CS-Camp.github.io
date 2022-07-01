# https://www.natasshaselvaraj.com/customer-segmentation-with-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# reading the data frame

df = pd.read_csv('Mall_Customers.csv')

print(df.head())

col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features = df[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features = pd.DataFrame(features, columns = col_names)
print(scaled_features.head())

sum_sq_err = []

for cluster in range(1,10):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(df)
    sum_sq_err.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them

frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':sum_sq_err})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('clustersplot.png')
plt.clf()

# # First, build a model with 4 clusters
#
kmeans = KMeans(n_clusters = 4, init='k-means++')
kmeans.fit(df)

# # Now, print the silhouette score of this model
#
print(silhouette_score(df, kmeans.labels_, metric='euclidean'))
#
clusters = kmeans.fit_predict(df.iloc[:,1:])
df['label'] = clusters

fig = plt.figure(figsize=(21,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.view_init(30, 185)
plt.savefig('clusters1.png')
plt.clf()

pca = PCA(n_components=4)
principalComponents = pca.fit_transform(df)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.savefig('pcafeatures.png')

PCA_components = pd.DataFrame(principalComponents)

inertias = []

for k in range(1, 10):
    model = KMeans(n_clusters=k)
    model.fit(PCA_components.iloc[:,:2])
    inertias.append(model.inertia_)

plt.plot(range(1, 10), inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(range(1, 10))
plt.savefig('clustersafterpca.png')
plt.clf()
#
model = KMeans(n_clusters=4)
model.fit(PCA_components.iloc[:,:2])
#
# # silhouette score
print(silhouette_score(PCA_components.iloc[:,:2], model.labels_, metric='euclidean'))
#
model = KMeans(n_clusters=4)

clusters = model.fit_predict(PCA_components.iloc[:,:2])
df["label"] = clusters

fig = plt.figure(figsize=(21,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.view_init(30, 185)
plt.savefig('clusters2.png')
plt.clf()


df = pd.read_csv('Mall_Customers.csv')
df = df.drop(['CustomerID'],axis=1)

pred = model.predict(PCA_components.iloc[:,:2])
frame = pd.DataFrame(df)
frame['cluster'] = pred
print(frame.head())

avg_df = df.groupby(['cluster'], as_index=False).mean()
print(avg_df)

plt.bar(avg_df['cluster'], avg_df['Age'])
plt.xlabel('Cluster')
plt.ylabel('Age')
plt.xticks(avg_df['cluster'])
plt.savefig('clusters_age.png')
plt.clf()

plt.bar(avg_df['cluster'], avg_df['Annual Income (k$)'])
plt.xlabel('Cluster')
plt.ylabel('Income')
plt.xticks(avg_df['cluster'])
plt.savefig('clusters_income.png')
plt.clf()

plt.bar(avg_df['cluster'], avg_df['Spending Score (1-100)'])
plt.xlabel('Cluster')
plt.ylabel('Spending')
plt.xticks(avg_df['cluster'])
plt.savefig('clusters_score.png')
plt.clf()
