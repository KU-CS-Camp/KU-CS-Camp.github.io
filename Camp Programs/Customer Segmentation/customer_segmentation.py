import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df = # TODO - Load the data from csv file using pandas (pd)

print(df.head())

col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features = df[col_names]
# Here we scale the data points down to a smaller scale
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features = pd.DataFrame(features, columns = col_names)
print(scaled_features.head())

sum_sq_err = []

# Now we need to find the optimal number of clusters
# TODO - Create a loop that iterates 10 times
# Make a KMeans model with n_clusters equal to the loop index (it will act as the number of clusters we are testing)
# Then fit the model to the df data
# Append the inertia_ value of the model to the sum_sq_err array


# The following code will plot the inertia of each trial and save it as a png
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':sum_sq_err})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('num_clusters_inertia.png')
plt.clf()


# TODO - Take a look at the save plot. What is the optimal number of clusters?
# Once you determine that number, make a model the same way you did earlier but with the optimal number of clusters
# Fit the model with df as well


# The following code will print the silhouette coefficient.
# This value tells us the quality of our clusters. The closer to 1 the better
print(silhouette_score(df, kmeans.labels_, metric='euclidean'))

# The following code will show and save a 3D plot of the clusters
clusters = kmeans.fit_predict(df.iloc[:,1:])
df['label'] = clusters
fig = plt.figure(figsize=(21,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.view_init(30, 185)
plt.show()
plt.savefig('clusters1.png')
plt.clf()

# Based on the silhouette coefficient and plot, could we do better?
# Let's try to run PCA to find better clusters
pca = # TODO - Create a PCA instance with n_components=4 for our 4 clusters
principalComponents = # TODO - Fit and transform the PCA on the df data

# The following code will plot the variance of each PCA feature
# Which features account for the majority of the variance?
# We will use those features to fit our model
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()
plt.savefig('pcafeatures.png')
plt.clf()

# New dataframe (df) to use based on PCA components
PCA_components = pd.DataFrame(principalComponents)

inertias = []

# TODO - Repeat your code for testing different numbers of clusters
# You will need to fit the model on the first 2 columns in PCA_components based on their high variance we see in the plot
# This will require slicing the dataframe. Here is the syntax: array[startrow:endrow,startcol:endcol]
# : by itself can be used to reference every row or every column

# Again, a plot of each cluster's inertia will be saved.
plt.plot(range(1, 10), inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(range(1, 10))
plt.savefig('num_clusters_intertia_pca.png')
plt.clf()

# TODO - As previously, choose the optimal number of clusters and create a model.
# Fit the model on the first 2 columns of PCA_components


# Silhouette score will be printed and 3D plot is created/saved
print(silhouette_score(PCA_components.iloc[:,:2], model.labels_, metric='euclidean'))

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
plt.show()
plt.savefig('clusters2.png')


# TODO - Analyze the results!
# The following code will save plots of the average age, income, and speanding of each cluster
# Look at these plots and draw some conclusions for each cluster
# You can even try to create personas for each cluster
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
