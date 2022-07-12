---
title: Stock Predictor
layout: default
filename: stock-predictor.md
--- 

## Stock Predictor
In this project, we will build a stock predictor for Tesla's stock. We will build this predictor using the neural network model.

Helpful Resources
- [LSTM Networks](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
- [LSTMs](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/)

Download the dataset [here](datasets/TSLA.csv)

### Create a New File
Create a new file (and new folder if you would like) for this project. It is important that you run ```source 2022summercamp/bin/activate``` in your terminal since Tensorflow requires a specific version of Python.  Make sure you then navigate into the same folder as your Python file to run it (cd /Desktop/foldername or cd /foldername).

***
### Initial Steps

First, load all of the imports necessary for the project.

```
import pandas as pd
from pandas import read_csv
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
df = read_csv('TSLA.csv')
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
df['column'].values
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

**If these steps are too confusing, look at the code in the 'Check your Code' dropdown as you read through the steps to help with understanding.**

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

***
### Building the Model
To begin, initialize the model as sequential. Now, we can start adding layers:

1. LSTM layer with 96 units for the output's dimensionality, return sequence as true (makes the LSTM layer with three-dimensional input), and an input shape equal to (x_train.shape[1], 1)
2. Dropout layer with a 0.2 dropout fraction (drops 20% of the layers)
3. LSTM layer with 96 units and return sequence as true
4. Dropout layer with a 0.2 dropout fraction
5. LSTM layer with 96 units and return sequence as true
6. Dropout layer with a 0.2 dropout fraction
7. LSTM layer with 96 units
8. Dropout layer with a 0.2 dropout fraction
9. Dense layer with units set to 1 for one value output


<details markdown="1">

<summary>Check Your Code</summary>

```
model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))
```

</details>

Also, we need to reshape our data into a 3D array for use in the LSTM layers.

```
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
```

***
### Compile the Model
The final step before training the model is to compile. We want to use [mean_squared_error](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/) as the loss function since this is a regression problem and [adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) as the optimizer to update network weights iteratively based on training data.

<details markdown="1">

<summary>Check Your Code</summary>

```
model.compile(loss='mean_squared_error', optimizer='adam')
```

</details>

***
### Train the Model
Fit the model with the x and y training sets, 50 epochs (cycles through the full training dataset), and 32 as the batch size (number of training examples used in one iteration).

<details markdown="1">

<summary>Check Your Code</summary>

```
model.fit(x_train, y_train, epochs=50, batch_size=32)
```

</details>

***
### Predict and Visualize
Now, it is time to call predict on the x test set.  Once you have done that, the predictions will need to be inversely transformed with the scaler (scaler.inverse_transform).

<details markdown="1">

<summary>Check Your Code</summary>

```
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
```

</details>

You can use the following code to visualize how well the model performs.

```
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(16,8))
ax.plot(y_test_scaled, color='red', label='Original price')
plt.plot(predictions, color='blue', label='Predicted price')
plt.legend()
plt.savefig('StockPrediction.png')
```

How do the predictions look?

***
### Explore Other Parameter Values
Try out different layers arrangements and epoch/batch_size values and see if you can improve your predictions!

***

This exercise was adapted from [Ahmad Mardeni](https://www.section.io/engineering-education/stock-price-prediction-using-python/)
