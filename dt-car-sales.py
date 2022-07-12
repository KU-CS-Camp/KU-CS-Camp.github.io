from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = # TODO - Use read_csv to read in the car sales data

# Here I use mapping to change 'Male' to 0 and 'Female'to 1
print(data.Gender)
data.Gender = data.Gender.map({'Male': 0, 'Female': 1})
print(data.Gender)

array = data.values
X = # TODO - Use slicing to create the feature (X) array as we did yesterday. Look at how many features we have to determine the indexes.
y = # TODO - Use slicing to create the output (y) array as we did yesterday
# TODO - Create x and y training arrays as well as x and y testing arrays using train_test_split (again in the same way as yesterday)

model = # TODO - Initialize the decision tree classifier
model = # TODO - Fit the model on the x and y training arrays
predictions = # TODO - Predict with the new model on the x test array

# TODO - Print out the accuracy score, confusion matrix, and classification report in the same way as yesterday

# The following code will generate a visual of the decision tree made by the classifier.
tree.plot_tree(model.fit(X, y))
plt.savefig('tree.png')

# Now, create a user interaction similar to your restaurant from yesterday
# Ask them for a gender, age, and salary then use the model you just created to predict whether they will buy a car.
# Once you have the new information, pass it to the predict function in the following format: [[gendernum, age, income]]
# Print out the result of the prediction.
