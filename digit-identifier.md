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

Check the shape of the arrays again after they have been reshaped.

```
num_classes = y_test.shape[1]
```

Single Neural Network
```
model = Sequential()
model.add(Dense(num_classes, input_dim=num_pixels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

scores = model.evaluate(X_test, y_test)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

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
