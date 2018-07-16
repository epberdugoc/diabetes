from sklearn import tree
from sklearn.model_selection import train_test_split
from graphviz import Source

def build_decision_tree( X, Y, cr='gini', md=None, mss=2, msl=1, ts=.25):
	'''
		Input: X = numpy 2d array. Each row is an individual sample and each column 
					is a feature.
			   Y = column vector of the target variables.
			   cr = criterion for splitting. The function used to measure the quality of the 
					split.  Either 'gini' or 'entropy'. Default = 'gini'
			   md = the maximum depth of the tree. If None, then nodes are expanded until all 
					leaves are pure or until all leaves contain less than min_samples_split 
					samples. Default = None
			   mss = The minimum number of samples required to split an internal node.  
					If int, then consider mss as the minimum number.  If float, then mss is a
					percentage and ceil(mss,*n_samples) are the minimum number of samples for 
					each split. Default = 2
			   msl = The minimum number of samples required to be a leaf node.  If int, then
					consider msl as the minimum number.  If float, then msl is a percentage 
					and ceil(msl*n_samples) are the minimum number of samples for each node.
					Default = 1
			   ts = test size. IF float, should be between 0.0 and 1.0 and represent the
					proportion of the dataset to include in the test split.  If int, 
					represents the absolute number of test samples. If None, the value is
					set to the complement of the train size. By default, the value is set
					to 0.25.
	
		Output: clf = DecisionTreeClassifier object after having been trained on a subset
						of the dataset.
				test_features = 2d numpy array.  Columns are features and rows are individual
							samples. This set of features will be used to test the classifier,
							i.e. they were not used to train the classifier, clf
				test_labels = 1d numpy array of target variables corresponding to each sample.
	'''

	# Split the data into a "training set" and "test set".
	train_features, test_features, train_labels, test_labels = train_test_split(X,Y, test_size=ts)	

	# Construct tree
	clf = tree.DecisionTreeClassifier(criterion=cr,max_depth=md,min_samples_split=mss,min_samples_leaf=msl)
	clf.fit(train_features,train_labels)
	
	return clf, test_features, test_labels


###########################################################################################

def confusion_matrix_values( predict_labels, test_labels ):
	'''
		Input: predict_labels = 1D int array of class values predicted by the classifier.
			   test_labels = 1D int array of true class values
		Output: tp = number of true positives
				tn = number of true negatives
				fp = number of false positives
				fn = number of false negatives
	'''
	
	true_pos = 0; true_neg = 0
	false_pos = 0; false_neg = 0

	for i in range( predict_labels.shape[0] ):
		if (predict_labels[i]==1) and ( test_labels[i]==1):
			true_pos += 1
		elif (predict_labels[i]==0) and ( test_labels[i]==0):
			true_neg += 1
		elif (predict_labels[i]==1) and ( test_labels[i]==0):
			false_pos += 1
		else:
			false_neg += 1

	return true_pos, true_neg, false_pos, false_neg

###########################################################################################
