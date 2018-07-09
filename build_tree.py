import pandas as pd
import numpy as np
from sklearn import tree

'''
    This file will build the Decision tree classifier for the diabetes 
    data using the sklearn package.
'''

# Number of samples used to train the dataset
num_samples = 100

# Read csv file with cleaned data and store in 'df'.
df = pd.read_csv('diabetic_data_cleaned.csv')

# Convert pandas dataframe to numpy array.
X = df.values[:num_samples,:-1]  # each row of X is an individual sample and each column is a feature. 
Y = df.values[:num_samples,-1]  # Y is a column vector of the target variable values.

# Construct tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)


