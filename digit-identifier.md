## Digit Identifier
In this project, we will create a digit identifier for the MNIST handwritten digits dataset.The dataset includes images such as these:

![1_N-43GwxkmUqzJgbTJs3ocw](https://user-images.githubusercontent.com/108029475/176959309-576deb32-d3cc-4c40-89a5-327601c5afc2.png)

### Initial Steps

First, load all of the imports necessary for the project.

```
import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
```

***
### Load Data

Load the data from the MNIST dataset.

```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

***
### Check Array Shape

Check the shape of the new arrays.

```
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

***
### View Dataset Sample

You can use the following code to view the first four training pictures.

```
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.savefig('plot.png')
```

***
### Format Data and Labels


```
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
```

Normalize the image pixel values by dividing the X arrays by 255.

```
X_train = X_train / 255
X_test = X_test / 255
```

to_categorical is used to transform the label data before passing it to the model.

```
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

The number of classes is equal to the second dimension of the y_test array shape. In this dataset, the number of classes should be 10 since we are classifying 10 digits.

```
num_classes = y_test.shape[1]
```

***
### Next Steps

We will be looking at three different neural network architectures, which will allow us to make a comparison.

- Single Layer Neural Network
- Multi-Layer Neural Network
- Deep Neural Network

***
### Single Layer Neural Network

#### Define the Model

Now, we will define a single-layer neural network with the following characteristics:
- An input layer of 784 neurons (the input dimension), which is equal to the number of pixels calculated earlier
- An output layer of 10 neurons (number of classes)
- The activation function used will be softmax activation ([What is softmax activation?](https://machinelearningmastery.com/softmax-activation-function-with-python/))

Steps:

1. Create an instance of the Sequential model
2. Add a Dense layer to the model with the number of classes, input dimension, and softmax activation

```
model.add(Dense(number of classes, input_dim= input dimension, activation= activation function))
```

<details markdown="1">

<summary>Check Your Answer</summary>

```
model = Sequential()
model.add(Dense(num_classes, input_dim=num_pixels, activation='softmax'))
```

#### Compile the Model
Now that we have defined the first model, we need to compile it. For compiling we have to give a loss function, an optimizer, and a metric as an input.

We will use:

- Loss Function: [Cross-entropy Loss (categorical_crossentropy)](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
- Optimizer: [SGD](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/)
- Metric: Accuracy

```
model.compile(loss='loss function', optimizer='optimizer', metrics=['metrics'])
```

</details>


<details markdown="1">

<summary>Check Your Answer</summary>

```
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

</details>

#### Train/Fit Model
The model is now ready to be trained. To do this, we need to provide training data to the neural network. Also, we have to specify the validation data, the number of epochs, and batch size as parameters ([What is an epoch and batch?](https://www.jigsawacademy.com/blogs/ai-ml/epoch-in-machine-learning)).

```
model.fit(training X, training Y, validation_data=(test X, test Y), epochs= epochs, batch_size= batch size)
```

<details markdown="1">

<summary>Check Your Answer</summary>

```
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
```

</details>

#### Evaluate the Model
Finally, we need to evaluate the model on the testing data:

```
scores = model.evaluate(X_test, y_test)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

### Multi-Layer Neural Network

<details markdown="1">

<summary>Check Your Answer</summary>

```
model = Sequential()
model.add(Dense(500, input_dim=num_pixels, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

scores = model.evaluate(X_test, y_test)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

</details>

### Deep Neural Network

we will add 3 hidden layers having 500, 100 and 50 neurons respectively.

<details markdown="1">

<summary>Check Your Code</summary>

```
model = Sequential()
model.add(Dense(500, input_dim=num_pixels, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

scores = model.evaluate(X_test, y_test)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

</details>
