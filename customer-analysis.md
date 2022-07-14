---
title: Customer Analysis
layout: default
filename: customer-analysis.md
--- 

## Customer Segmentation and Analysis
Customer segmentation is very important for businesses to understand their target audience. With information from customer segmentation, they can curate the advertisements sent to their audience based on their demographics and interests.  We will be utilizing the unsupervised machine learning algorith k-Means clustering to take unlabelled customer data and assign each point to a cluster. The dataset we will use contains information about mall customers (age, income, spending score).

Download the dataset [here](datasets/Mall_Customers.csv)

### Initial Steps

First, load all of the imports necessary for the project.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
```

***
### Load Data

Load the data from the dataset into a variable named df and print the head of the dataset to ensure it has loaded correctly.

<details markdown="1">

<summary>Check Your Code</summary>

```
df = pd.read_csv('Mall_Customers.csv')
print(df.head())
```

</details>

***
### Optimal Clusters (k value)

Now, we need to find the optimal number of clusters to create with our model. We will look at the inertia values of models create with various k sizes and plot them to determine the best k value.

Steps:

- Initialize an empty array named 'sum_sq_err'
- Create a loop that iterates 10 times
- Make a KMeans model with n_clusters equal to the loop index (it will act as the number of clusters we are testing) and an init parameter set to 'k-means++'
- Then fit the model to the df data
- Append the inertia_ value of the model to the sum_sq_err array

<details markdown="1">

<summary>Check Your Code</summary>

```
sum_sq_err = []

for cluster in range(1,10):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(df)
    sum_sq_err.append(kmeans.inertia_)
```

</details>

Plot the resulting data using the code below. 

```
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':sum_sq_err})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('num_clusters_inertia.png')
plt.clf()
```

Take a look at the saved plot. What is the optimal number of clusters? You can use ‘The Elbow Method’, which means the optimal number of clusters is where the elbow occurs.

***
### Create Optimal Model

Create and fit a new model with your optimal number of clusters. This should look the same as the code you wrote in the previous for loop.

<details markdown="1">

<summary>Check Your Code</summary>

```
kmeans = KMeans(n_clusters = 4, init='k-means++')
kmeans.fit(df)
```

</details>

***
### Silhouette Score
A silhouette score is a metric used to evaluate the quality of clusters created by the k-Means algorithm. Silhouette scores range from -1 to +1. The higher the silhouette score, the better the model. The silhouette score measures the distance between all the data points within the same cluster. The lower this distance, the better the silhouette score. A silhouette score closer to +1 indicates good clustering performance, and a silhouette score closer to -1 indicates a poor clustering model.

```
print(silhouette_score(df, kmeans.labels_, metric='euclidean'))
```

Is this a good score? Let's visualize the model as well.

***
### Visualize the Model

The following code will show and save a 3D plot of the clusters.

```
fig = plt.figure(figsize=(21,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.view_init(30, 185)
plt.savefig('clusters1.png')
plt.clf()
```

How do the clusters look? Do you think the model is performing well based on this graph and the silhouette score?

<details markdown="1">

<summary>My Thoughts</summary>

I don't see great cluster separation since the different color points are overlapping with each other.  Along with a silhouette score that I calculated to be 0.35, tells me the model isn't performing very well.

</details>

***
### Improving the Model

One way to our imporve our model is to perform feature selection.  PCA is a technique that helps us reduce the dimension of a dataset. When we run PCA on a data frame, new components are created. These components explain the maximum variance in the model.

Steps:
1. Create an instance of PCA with 4 n_components
2. Call pca.fit_transform on the dataframe (df)
3. Create a new DataFrame called PCA_components with the results of the previous step

<details markdown="1">

<summary>Check Your Code</summary>

```
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(df)
PCA_components = pd.DataFrame(principalComponents)
```

</details>

To view the variance of each component generated by PCA, use the following code to create a bar graph.

```
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.savefig('pcafeatures.png')
```

Which components have the highest variance? We want to pass the two components with the highest variance to the new model.

***
### Finding New Optimal k

Follow the same procedure to find the optimal number of clusters (k) from earlier. This time you will use the 2 chosen components from PCA to fit the model: ```PCA_components.iloc[:,:2]```

Save a plot of the inertias:

```
plt.plot(range(1, 10), inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(range(1, 10))
plt.savefig('clustersafterpca.png')
plt.clf()
```

***
### Create New Optimal Model

Create a new model using the optimal k and fit the model on the 2 chosen PCA components.

<details markdown="1">

<summary>Check Your Code</summary>

```
model = KMeans(n_clusters=4)
model.fit(PCA_components.iloc[:,:2])
```

</details>

***
### Evaluate New Model

Calculate the silhouette score and visualize the data.

```
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
plt.savefig('clusters2.png')
plt.clf()
```

How does this model compare to the last one?

***
### Cluster Analysis
Now, we can take a look at the characteristics of each cluster and draw some conclusions about them. Let's first reset the dataframe then predict on the model with our chosen PCA components. 

```
df = # Read mall customer csv using pandas
df = df.drop(['CustomerID'],axis=1)

pred = # Call predict using the second model with PCA_components.iloc[:,:2]
frame = pd.DataFrame(df)
# Create a new column in frame and set it to the prediction results
print(frame.head())
```

We can compare attributes of the different clusters by finding the average of all variables across each cluster.

```
avg_df = df.groupby(['cluster'], as_index=False).mean()
print(avg_df)
```

Finally, try your hand at creating bar graphs for each cluster's average attributes (calculated above in avg_df). Don't forget the x and y labels and xticks. Create 3 graphs: cluster vs age, cluster vs income, cluster vs spending score

<details markdown="1">

<summary>Check Your Code</summary>

```
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
```

</details>

What are the attributes of each clusters? How would you describe the persona of the customers in each cluster? What are some adjectives you could use to describe them?

***

This exercise was adapted from [natasshaselvaraj](https://www.natasshaselvaraj.com/customer-segmentation-with-python/)
