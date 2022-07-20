## Old Faithful Eruption Clustering
One use of k-means clustering is to identifying outliers in data. If you know data points are outliers, you can look into what factors cause them to be different. For instance, we will be looking at the the eruption time and waiting time of Old Faithful in Yellowstone, and if there are outliers in this data, we could look into the other factors causing the change.

Download the dataset [here](/datasets/oldfaithful.csv)

### Add Imports

```
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
```

### Load Data
Load the data into a variable as normal with read_csv()

<details markdown="1">

<summary>Check Your Code</summary>

```
oldfaithful = read_csv('oldfaithful.csv')
```

</details>

### Scale Data
We will need to scale the values of our data down to have a value within 0-1. The following code will take care of that for us:

```
ss = StandardScaler()
oldfaithful = pd.DataFrame(ss.fit_transform(oldfaithful), columns=['TimeEruption','TimeWaiting'])
```

### Initial Plot of Data
To view a plot of the data points, utilize the following code:

```
plt.figure(figsize=(8,6))
plt.scatter(oldfaithful.TimeEruption, oldfaithful.TimeWaiting)
plt.xlabel('Eruption Time (minutes)')
plt.ylabel('Waiting time to next eruption (minutes)')
plt.title('Scatter plot of Eruption Time vs. Waiting Time')
plt.savefig('initialfig.png')
```

### Choose K
One technique for choosing the value of k, or the number of clusters, is by simply looking at a plot of the data. Look at the saved plot you just created and determine what you think the k value (number of clusters) should be for this data.

### Create and Train Model
Now, create a KMeans model with a parameter of the number of clusters (n_clusters) you determined in the previous step.  Train/fit the model on the whole dataset (no need to split into different arrays this time since we do not have labels for a y array).

<details markdown="1">

<summary>Check Your Code</summary>

```
km = KMeans(n_clusters=2)
model = km.fit(oldfaithful)
```
</details>

### Plot Clusters
We can now look at these clusters by plotting the data with different colors.

```
colors=["red","blue"]
plt.figure(figsize=(8,6))
for i in range(np.max(model.labels_)+1):
    plt.scatter(oldfaithful[model.labels_==i].TimeEruption, oldfaithful[model.labels_==i].TimeWaiting, label=i, c=colors[i], alpha=0.5, s=40)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], label='Centers', c="black", s=100)
plt.title("K-Means Clustering of oldfaithful Data",size=20)
plt.xlabel("Eruption Time")
plt.ylabel("Waiting Time")
plt.title('Scatter plot of Eruption Time vs. Waiting Time')
plt.legend()
plt.savefig('clusters.png')
```

###  Define a Distance Function
We want to define a distance function that will be called on each sample of data. This function will calculate the Euclidean distance between a data point and the center of its cluster.

Add the following parameters to the function declaration:
- TimeEruption
- TimeWaiting
- Label

We also need to get the center values for the cluster we are looking at.
```
center_TimeEruption =  model.cluster_centers_[label,0]
center_TimeWaiting =  model.cluster_centers_[label,1]
```

Now, to calculate the Euclidean distance, we set a variable equal to this equation: $\sqrt{(TimeEruption - center_TimeEruption)^2 + (TimeWaiting - center_TimeWaiting)^2}$.
- To code a square root, ``` np.sqrt(equation)```
- Squaring a value can be done like this: ``` (value) ** 2 ```

Finally, we return the distance value from the function and round it at the same time (no need to change this code).
```
return np.round(distance, 3)
```

<details markdown="1">

<summary>Check Your Code</summary>

```
def distance_from_center(TimeEruption, TimeWaiting, label):
    center_TimeEruption =  model.cluster_centers_[label,0]
    center_TimeWaiting =  model.cluster_centers_[label,1]
    distance = np.sqrt((TimeEruption - center_TimeEruption) ** 2 + (TimeWaiting - center_TimeWaiting) ** 2)
    return np.round(distance, 3)
```

</details>

### Add Values to Dataset
Now, let's add both the labels from the model and distances to our dataset variable (this way we have easy access to them).  Set oldfaithful['label'] equal to model.labels_ and oldfaithful['distance'] equal to a function call to distance_from_center with the following parameters: oldfaithful.TimeEruption, oldfaithful.TimeWaiting, oldfaithful.label

<details markdown="1">

<summary>Check Your Code</summary>

```
oldfaithful['label'] = model.labels_
oldfaithful['distance'] = distance_from_center(oldfaithful.TimeEruption, oldfaithful.TimeWaiting, oldfaithful.label)
```

</details>

### Finding the Outliers
It is time to find the outliers. The outliers are the datapoints that lay the farthest away from the center of their cluster. The code below will sort the distances we generated and choose the 10 largest distances as our outliers.

```
outliers_idx = list(oldfaithful.sort_values('distance', ascending=False).head(10).index)
outliers = oldfaithful[oldfaithful.index.isin(outliers_idx)]
```

### Final Plot
Use the following code to plot the outliers.  What do you think are some factors that could caused these outliers? In real-life, looking into these other contributing features could help us in future predictions.

```
plt.figure(figsize=(8,6))
colors=["red","blue","green","orange"]
for i in range(np.max(model.labels_)+1):
    plt.scatter(oldfaithful[model.labels_==i].TimeEruption, oldfaithful[model.labels_==i].TimeWaiting, label=i, c=colors[i], alpha=0.5, s=40)
plt.scatter(outliers.TimeEruption, outliers.TimeWaiting, c='aqua', s=100)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], label='Centers', c="black", s=100)
plt.title("K-Means Clustering of oldfaithful Data",size=20)
plt.xlabel("Annual TimeEruption")
plt.ylabel("TimeWaiting")
plt.title('Scatter plot of Annual TimeEruption vs. TimeWaiting')
plt.legend()
plt.savefig('outliersfig.png)
```


