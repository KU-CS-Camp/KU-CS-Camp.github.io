# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# TODO - Load the dataset
# Use read_csv(filepath, column_names)
url = "iris.csv"
names = # array of column titles (as string)
dataset = # load data

# TODO - Print the dataset shape

# TODO - Print the first 20 entries in the dataset using head()

# TODO - Print the dataset description

# Class distribution
print(dataset.groupby('class').size())

# TODO - Uncomment the following lines to check out the following plots to see the distribution of the data
# You may want to comment them out again once finished looking at the plots
# Box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()
#pyplot.clf()
# Histograms
#dataset.hist()
#pyplot.show()

array = dataset.values
# TODO - Split the array into x and y arrays
# Use array[start_row:end_row, start_col:end_col]
# Just ':' selects all rows or columns
X = # This array should have all rows and the first 4 columns (these are the characteristics)
y = # This array should have all rows and the last column (these are the labels)

# TODO - Split the data into training and testing sets
# Use train_test_split(x_array, y_array, test_size, random_state)
# We will use 20% for testing, so use 0.2 for test size and set random state to 1
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# These are the models we will be testing on the dataset
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


results = []
names = []
for name, model in models:
        # TODO - For every model, run k-fold cross validation
	kfold = # Call StratifiedKFold() with 10 splits, random state as 1, and shuffle as true
	cv_results = # Use cross_val_score() to generate results. Pass it the model, x and y training sets, kfold as cv, and 'accuracy' for scoring
	# Add the cv_results to the results array as well as the name to the names array
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# TODO - Uncomment the following lines to compare the algorithms. Which algorithm performs the best? We will use it in the next step
# You may want to comment them out again once finished looking at the plots
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()

# TODO - Make predictions on the validation dataset
model = # Create an instance of the model that performed the best
# Fit the model on x and y training sets
predictions = # Use the model to predict the x validation set

# Evaluate the predictions. How did the model perform?
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
