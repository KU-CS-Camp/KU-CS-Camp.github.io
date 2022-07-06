---
title: Movie Box Office Predictor
layout: default
filename: box-office.md
--- 

## Movie Box Office Predictor
Can a movie's production budget predict how much revenue it will generate? We'll explore this question using linear regression and see if our model could provide an accurate prediction.

Download the dataset here

### Initial Steps

First, load all of the imports necessary for the project.

```
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
```
