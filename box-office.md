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
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
```

***
### Load Data

Load the data using the pandas function read_csv('file.csv') then use variable.describe() to see information about the dataset.

<details markdown="1">

<summary>Check Your Code</summary>

```
data = pandas.read_csv('movie_cost_revenue.csv')
data.describe()
```

</details>

***
### Create X and Y Datasets
Create x and y datasets by using DataFrame from pandas. You can use the following syntax:

```
DataFrame(data, columns=['column_name'])
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

Create an instance of the Linear Regression model then fit it with the x and y arrays. Check out the coefficient and intercept of the model after training by printing out regression.coef_ and regression.intercept_ .

<details markdown="1">

<summary>Check Your Code</summary>

```
model = LinearRegression()
model.fit(X, y)

print(model.coef_)
print(model.intercept_)
```

</details>

***
### Predict and Plot

Now, it is time to predict with the model over the same x dataset. 

<details markdown="1">

<summary>Check Your Code</summary>

```
predict_list = model.predict(X)
```

</details>

We can then plot the predictions to get an idea of how well the model performs.

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
Finally, we can call score() on the model and pass in the x and y arrays to view the accuracy of the model.

<details markdown="1">

<summary>Check Your Code</summary>

```
model.score(X, y)
```

</details>

How does it perform? Would a different model perform better?

***

This exercise was adapted from [Grokking Machine Learning](https://github.com/edualgo/Grokking-Machine-Learning/tree/main/Notebooks/Movie%20Box%20office%20Prediction)
