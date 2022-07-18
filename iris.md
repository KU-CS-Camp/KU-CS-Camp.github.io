---
title: Iris Model Comparison
layout: default
filename: iris.md
--- 

## Iris Model Comparison

The “hello world” of machine learning  is the Iris flower data set. This dataset is utilized to identify 3 species of the iris flower: Iris Setosa, Iris Versicolour, and Iris Virginica by the length and the width of the sepals and petals, in centimeters. We are going to test different models to find which has the best accuracy with this dataset and design a machine learning application with that model to identify species of the iris flower.

Download the dataset [here](datasets/iris.csv) (right click and save it)

For most of this project, you will need to change the parameters passed to functions. For example, func('filename.csv') should be changed to the actual file name of the dataset on your computer.

### Create File and Prepare Terminal
Create a new file with the ending .py , move your downloaded data file into the same folder as the new Python file, run the command below in terminal, and then use cd to change into the folder with your new file and data file.

```
source 2022summercamp/bin/activate
```

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

We have already imported the read_csv() function from pandas at the top of our file. We need to pass the location of the data file (should just be the name of the file if it's in the same directory as your Python file) and the column names associated with the csv file using the following syntax:

```
dataset = read_csv('filename.csv')
```

<details markdown="1">

<summary>Check Your Code</summary>

```
dataset = read_csv('iris.csv')
```

</details>

***
### Summarize Data

Let’s take a look at the data using some useful commands that could also be helpful in future projects.

Use the print() function to print out the results of each of the following commands:
- Dimensions of the dataset - This will give us an idea of how many instances (rows) and attributes (columns) the data contains
     ```dataset.shape```
- Peek at data - This allows us to actually view the data


     ```dataset.head(20)```
- Statistical summary - This produces a summary of each attribute (count, mean, min/max values, etc)


     ```dataset.describe()```
- Class distribution - We can group the data by the class attribute and count how many instances belong to each class
     ```dataset.groupby('class').size()```

***
### Visualize Data

Now, we can visualize our data for an even further understanding with matplotlib.

We can look at the distribution of the input variables using box and whisker plots (no need to change any code)
```
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.savefig('databoxplot.png')
```

Another way to view distribution by creating histograms

```
dataset.hist()
pyplot.savefig('datahist.png')
```

Do you notice any familiar distributions? [hint](https://www.mathsisfun.com/data/standard-normal-distribution.html)

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
- Then call train_test_split on the arrays with the x and y arrays, a test size pf 0.2, and a random state set to 1

```
array = dataset.values
X = array[start_row:end_row, start_col:end_col]
y = array[start_row:end_row, start_col:end_col]
X_train, X_validation, Y_train, Y_validation = train_test_split(x array, y array, test_size= test size, random_state= random state)
```

<details markdown="1">

<summary>Check Your Code</summary>

```
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
```

</details>

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

We will store the 6 models we want to test in an array. Copy this code exactly into your file:
```
models = [LogisticRegression(solver='liblinear', multi_class='ovr'),  LinearDiscriminantAnalysis(),  KNeighborsClassifier(), DecisionTreeClassifier(), GaussianNB(), SVC(gamma='auto')]
```

***
### Create Models and Check Accuracy

Now, we will create a model and check its accuracy

We will perform 10-fold cross validation to estimate model accuracy
- Our dataset will be split into 10 parts
- 9 parts train and 1 part is used for testing, repeated for all combinations

The process:
- Create a loop that goes through every (name, model) pair we just created
- Inside the loop
    - Create a k-fold instance with 10 parts, 1 as the random state seed, and shuffle set to True
    - Calculate the cross-validation score by passing in the model, x and y arrays, cv=kfold, and scoring='accuracy'
    - Print out the accuracy and time of each model

```
results = []
names = []
for model in models:
	kfold = 
	cv_results = 
	results.append(cv_results)
	names.append(str(model))
	print('%s: %f (%f)' % (str(model), cv_results.mean(), cv_results.std()))
```

<details markdown="1">

<summary>Check Your Code</summary>

```
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(str(model))
	print('%s: %f (%f)' % (str(model), cv_results.mean(), cv_results.std()))
```

</details>

You can compare the results using the following boxplot (copy as is):
```
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.savefig('alg-comp.png')
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
- Initiate the most accurate algorithm (check the model array from earlier for syntax)
- Call model.fit() and pass in the X and Y training arrays
- Store the results of calling model.predict() with the X validation array

```
# Make predictions on the validation dataset
model = 
# Fit the model on x and y training sets
predictions = 
```

<details markdown="1">

<summary>Check Your Code</summary>

```
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
```

</details>

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
