---
title: Iris Model Comparison
layout: default
filename: iris.md
--- 

## Iris Model Comparison

The “hello world” of machine learning  is the Iris flower data set. This dataset is utilized to identify 3 species of the iris flower: Iris Setosa, Iris Versicolour, and Iris Virginica by the length and the width of the sepals and petals, in centimeters. We are going to test different models to find which has the best accuracy with this dataset and design a machine learning application with that model to identify species of the iris flower.


### Initial Steps
First, we need to load all of the libraries that will be needed. You can run the script to ensure all libraries are installed correctly (ideally the nothing will happen, and the program will finish).

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

***
### Set Up Validtion

In order to know that a model we create is good, we need to set up a method for validation. Scikit-learn or sklearn is a machine learning library that provides us with many useful methods for modeling, metrics, splitting data, etc.

We want to be able to test the accuracy of our model on unseen data, so we will split our dataset and use 80% for training and 20% for validation.

To do this:
- Split the data into input (X) and output (y) arrays
    - Some slicing is required for splitting the arrays
    - Syntax: ```array[start_row:end_row, start_col:end_col]```
    - For both arrays, we can use ':' to include every row
    - In the input array, we want the first 4 iris characteristics, so we use 0:4 to catch the array at index 0,1,2,3
    - The output array is only the class characteristic at index 4
- Then call train_test_split on the arrays

```
array = dataset.values
X = array[start_row:end_row, start_col:end_col]
y = array[start_row:end_row, start_col:end_col]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
```

***
### Initiate Models

We need to find an algorithm that works best on this data, so we will test the following 6 algorithms:
- Linear
    - Logistic Regression (LR)
    - Linear Discriminant Analysis (LDA)
- Nonlinear
    - K-Nearest Neighbors (KNN).
    - Classification and Regression Trees (CART)
    - Gaussian Naive Bayes (NB)
    - Support Vector Machines (SVM)

```
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
```

***
### Create Models and Check Accuracy

Now, we will create a model and check its accuracy

We will perform 10-fold cross validation to estimate model accuracy
- Our dataset will be split into 10 parts
- 9 parts train and 1 part is used for testing, repeated for all combinations

The process:
- Loop through every (name, model) pair we just created
- Inside the loop
    - Create a k-fold instance with 10 parts, 1 as the random state seed, and shuffle set to True
    - Calculate the cross-validation score by passing in the model, arrays, kfold type, and accuracy scoring
    - Add the result and name of the model to an array that will hold all of the model results
    - Print out the accuracy and time of each model

```
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy’)
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
```

You can compare the results using the following boxplot:
```
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
```

***
### Results
What are your results?

Which algorithm did you find to be most accurate?

***
### Make Predictions

I found the most accurate algorithm to be SVM. If you found a different model to be more accurate, feel free to use it for predictions.
   
Now, we want to check the accuracy of our model on the validation set we created in the beginning.

First, we will fit our model on our training arrays, and then we can made predictions on the input validation array.

To do this:
- Create an instance of the most accurate algorithm
- Call model.fit() and pass in the X and Y training arrays
- Store the results of calling model.predict() with the X validation array

```
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
```

***
### Prediction Results

We can use functions from sklearn to calculate the accuracy, confusion matrix, and classification report based on the validation array and generated predictions.

```
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
```

How accurate were your predictions?

***

This exercise was adapted from [Machine Learning Mastery](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)