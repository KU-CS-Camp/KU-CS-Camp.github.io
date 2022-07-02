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

<button onclick="alert(document.getElementById('d1').style.visibility);">Check Code</button>

<div id="d1" style="visibility: 'hidden';">
```
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```
</div>
