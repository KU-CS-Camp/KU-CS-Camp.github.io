---
title: Movie Box Office Predictor
layout: default
filename: box-office.md
--- 

## Movie Box Office Predictor
Can a movie's production budget predict how much revenue it will generate? We'll explore this question using linear regression and see if our model could provide an accurate prediction. The dataset we will use contains data from many popular movies but the titles have been removed for easy use.

Download the dataset [here](datasets/movie_cost_revenue.csv)

### Initial Steps

First, load all of the imports necessary for the project.

```
import pandas
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
```

***
### Load Data

Now, we need to load the movie data, and the pandas library can help. Pandas is a data analysis library that we will use throughout this camp.

We have already imported the read_csv() function from pandas at the top of our file. We need to pass the location of the data file (should just be the name of the file if it's in the same directory as your Python file) and the column names associated with the csv file using the following syntax:

```
dataset = read_csv('filename.csv')
```

<details markdown="1">

<summary>Check Your Code</summary>

```
dataset = read_csv('movie_cost_revenue.csv')
```

</details>

***
### Create X and Y Datasets
Create x and y datasets by using DataFrame from pandas. You can use the following syntax:

```
X = DataFrame(data, columns=['column_name'])
y = DataFrame(data, columns=['column_name'])
```

The feature (x) array will contain the production budget column, and the label (y) array will hold the gross revenue column.

<details markdown="1">

<summary>Check Your Code</summary>

```
X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])
print(X.shape)
```

</details>

To view this data in a scatter plot:

```
plt.figure(figsize=(10,6))
plt.scatter(np.sort(X), np.sort(y), alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.savefig('fig1.png')
plt.clf()
```

***
### Build and Train the Model

Create an instance of the Linear Regression model by calling its class constructor. Then, fit the model with the x and y arrays using model.fit() . 

<details markdown="1">

<summary>Check Your Code</summary>

```
model = LinearRegression()
model.fit(X, y)
```

</details>


Check out the coefficient and intercept of the model after training by printing out model.coef_ and model.intercept_ .

<details markdown="1">

<summary>Check Your Code</summary>

```
print(model.coef_)
print(model.intercept_)
```

</details>

***
### Predict and Plot

Now, it is time to predict with the model over the same x dataset. Call model.predict and pass in the X array as testing values. Store these results in a variable.

<details markdown="1">

<summary>Check Your Code</summary>

```
predict_list = model.predict(X)
```

</details>

We can then plot the predictions to get an idea of how well the model performs (copy this code).

```
plt.figure(figsize=(10,6))
plt.scatter(np.sort(X), np.sort(y), alpha=0.3)
plt.plot(X['production_budget_usd'], predict_list, color='red', linewidth=3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.savefig('fig2.png')
```

***
### Evaluation
Finally, we can call model.score() and pass in the x and y arrays to view the accuracy of the model.

<details markdown="1">

<summary>Check Your Code</summary>

```
model.score(X, y)
```

</details>

How does it perform? Would a different model perform better?

***

This exercise was adapted from [Grokking Machine Learning](https://github.com/edualgo/Grokking-Machine-Learning/tree/main/Notebooks/Movie%20Box%20office%20Prediction)
