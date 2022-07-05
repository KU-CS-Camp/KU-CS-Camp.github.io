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

Load the data from the dataset and print the head of the dataset to ensure it has loaded correctly.

```
df = # Load the data from csv file using pandas (pd) read_csv function
# Print the head of the dataset
```

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
