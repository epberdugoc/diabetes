from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from tree_utils import build_random_forest, confusion_matrix_values

'''
    This file will build the Random Forest classifier for the diabetes 
    data using the sklearn package and output a ROC plot for
	different tree depths.
'''

# Read csv file with cleaned data and store in 'df'.
df = read_csv('diabetic_data_cleaned.csv')

# Convert pandas dataframe to numpy array.
X = df.values[:,:-1]  # each row of X is an individual sample and each column is a feature. 
Y = df.values[:,-1]  # Y is a column vector of the target variable values.

# Allocate storage for ROC curves.
num_pts = 100
tpr = np.zeros( num_pts+2, dtype='float64') # True positive rate
fpr = np.zeros( num_pts+2, dtype='float64') # False positive rate
dt = 1. / num_pts # delta threshold. The difference in the threshold between each point.

max_depths = [3,5,10,15,20,None]

for mdepth in max_depths:
	# Build Tree
	clf, test_features, test_labels = build_random_forest(X,Y, md=mdepth, nest=100)

	# Make predictions
	predict_labels = clf.predict( test_features )
	tp, tn, fp, fn = confusion_matrix_values(predict_labels, test_labels)

	predict_proba = clf.predict_proba( test_features )

	for i in range(num_pts+2):
		threshold = i*dt

		# Predicted labels for this threshold
		pl = np.array( list(  map( lambda x: 1 if x[1] >= threshold else 0, predict_proba ) ), dtype='int64' )

		tp, tn, fp, fn = confusion_matrix_values(pl, test_labels)

		tpr[i] = float(tp) / float(tp + fn)
		fpr[i] = float(fp) / float(fp + tn)

	plt.plot( fpr, tpr,linewidth=2,label="max_depth="+str(mdepth) )
plt.legend()
plt.plot([0,1],[0,1], color='black',linestyle='--',linewidth=3 )
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.savefig("rf_roc_md.png",dpi=300)
