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
### Standardize Data

```
col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features = df[col_names]
# Here we scale the data points down to a smaller scale
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features = pd.DataFrame(features, columns = col_names)
print(scaled_features.head())
```
