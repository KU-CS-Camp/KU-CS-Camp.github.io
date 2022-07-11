import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

msg = # TODO - Read in data from file with read_csv. Include names=["message", "label", "labelnum"] as a parameter
print(msg.shape())
# TODO - Create X and Y arrays. The feature (X) array should be the message and output (Y) array is the label number
X =
y =

# TODO - Generate x and y training and test arrays using train_test_split on x and y
# Print out their shapes and print the x training array to view the data

# We need to use a count vectorizer to convert a collection of text documents to a matrix of token counts
count_vect = CountVectorizer()
xtrain_dtm = # TODO - Use count vectorizer's fit_transform method on the x training array
xtest_dtm = # TODO - Use count vectorizer's transform method on the x test array
print(count_vect.get_feature_names())
print(xtrain_dtm)

# Time to train the Naive Bayes classifier on training data.
nb = MultinomialNB()
# TODO - Fit the model with xtrain_dtm and y train
predicted = # TODO - Predict with the model on xtest_dtm

# TODO - Print the following metrics with the predictions and y test
print("Accuracy metrics")
print()
print("Confusion matrix")
print()
print("Recall")
print()
print("Precison")
print()
