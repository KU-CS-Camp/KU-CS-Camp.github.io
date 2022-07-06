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
### Clean and Split Data

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
