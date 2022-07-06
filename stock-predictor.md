---
title: Stock Predictor
layout: default
filename: stock-predictor.md
--- 

## Stock Predictor
In this project, we will build a stock predictor for Tesla's stock. We will build this predictor using the neural network model.

Helpful Resources
- [LSTM Networks](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)

Download the dataset [here](datasets/TSLA.csv)

### Initial Steps

First, load all of the imports necessary for the project.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
```

***
### Load Data

Load the data using pandas function read_csv.

<details markdown="1">

<summary>Check Your Code</summary>

```
df = pd.read_csv('TSLA.csv')
```

</details>

To view the number of trading days we are looking at, print out the shape of the data.

<details markdown="1">

<summary>Check Your Code</summary>

```
print(df.shape)
```

</details>

***
### Clean, Split, and Scale Data

For simplicity, we will only focus on one variable, the open price. Set your dataframe equal to the values of the 'Open' column of the dataframe. Also, reshape the dataframe so the number of rows stays the same (you can use -1 to designate this) but there is only one column.

```
df['cloumn'].values
df.reshape(rows, columns)
```

<details markdown="1">

<summary>Check Your Code</summary>

```
df = df['Open'].values
df = df.reshape(-1, 1)
print(df.shape)
```

</details>

Now, split your data into training and testing sets. Use 80% of the data for training and the other 20% for testing. You can use NumPy arrays and slicing to do this.

<details markdown="1">

<summary>Check Your Code</summary>

```
dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])
print(dataset_train.shape)
print(dataset_test.shape)
```

</details>

We want to scale our data values to be within zero and one. Create a MinMaxScaler with a feature range of 0-1. Then, call scaler.fit_transform on the training set and scaler.transform on the testing set.

<details markdown="1">

<summary>Check Your Code</summary>

```
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)
```

</details>

***
### Creating X and Y Datasets
For this project, we will need to create the x and y train/test datasets in a slightly different way.  In this case, the features (x) will be the last 50 stock prices and the label (y) will be the next price. Since we will need to create these datasets for both training and testing, we can define a function to call.

Steps:
1. Define a function that takes a dataset
2. Create two empty lists for x and y 
3. Make a for loop that starts at 50 and ends at the size of the dataframe
4. Append the x array with the values from the index-50 until the index and the value at the index to y array
5. After the loop, use np.array() on the x and y lists before you return those values from the function
6. Call this new function on both the training and testing sets

<details markdown="1">

<summary>Check Your Code</summary>

```
def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)
```

</details>
