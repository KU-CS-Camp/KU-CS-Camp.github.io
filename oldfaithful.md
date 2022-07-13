## Old Faithful Eruption Clustering
Another use of k-means clustering is to identifying outliers in data. If you know data points are outliers, you can look into what factors cause them to be different. For instance, we will be looking at the the eruption time and waiting time of Old Faithful in Yellowstone, and if there are outliers in this data, we could look into the other factors causing the change such as weather.

Download the dataset [here](oldfaithful.csv)

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

```
oldfaithful = read_csv('oldfaithful.csv')
```

### Scale Data
We will need to scale the values of our data down to have a value within 0-1. The following code will take care of that for us:

```
ss = StandardScaler()
oldfaithful = pd.DataFrame(ss.fit_transform(oldfaithful), columns=['TimeEruption','TimeWaiting'])
```

### Initial Plot of Data

```
plt.figure(figsize=(8,6))
plt.scatter(oldfaithful.TimeEruption, oldfaithful.TimeWaiting)
plt.xlabel('Eruption Time (minutes)')
plt.ylabel('Waiting time to next eruption (minutes)')
plt.title('Scatter plot of Eruption Time vs. Waiting Time')
plt.savefig('initialfig.png')
```

### Choose K
Another technique for choosing the value of k is by simply looking at a plot of the data. Look at the saved plot you just created and determine what you think the k value should be for this data.

### Create and Train Model
Now, create a KMeans model with the number of clusters you determined in the previous step.  Train/fit the model on the dataset (no need to split into different arrays this time)

```
km = KMeans(n_clusters=2)
model = km.fit(oldfaithful)
```


