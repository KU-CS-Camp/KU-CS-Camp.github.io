---
title: Iris Model Comparison
layout: default
filename: iris.md
--- 

## Iris Model Comparison

### Initial Steps
1. Load all of the libraries you will need
2. Run the script to ensure all libraries are installed correctly (ideally the nothing will happen, and the program will finish)

```
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
```


***
### Load Dataset

Now, we need to load the iris data, and the pandas library can help. Pandas is a data analysis library that we will use throughout this camp.

We have already imported the read_csv() function from pandas at the top of our file. We need to pass the function the name of the data file and the column names associated with the csv file using the following syntax:

```
dataset = read_csv(”filename.csv", names=[‘col1’, col2', ‘col3'])
```

In this case, the column names will be the iris characteristics in this order: 

- sepal-length
- sepal-width
- petal-length
- petal-width
- class
***
### Summarize Data

Let’s take a look at the data using some useful commands that could also be helpful in future projects.

- Dimensions of the dataset - This will give us an idea of how many instances (rows) and attributes (columns) the data contains
     ```dataset.shape```
- Peek at data - This allows us to actually view the data


     ```dataset.head(num_rows)```
- Statistical summary - This produces a summary of each attribute (count, mean, min/max values, etc)


     ```dataset.describe()```
- Class distribution - We can group the data by the class attribute and count how many instances belong to each class
     ```dataset.groupby('class').size()```

***
### Visualize Data

Now, we can visualize our data for an even further understanding with matplotlib.

We can look at the distribution of the input variables using box and whisker plots
```
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
```

Another way to view distribution by creating histograms

```
dataset.hist()
pyplot.show()
```

Do you notice any familiar distributions?
