import numpy as np
from sklearn import tree
from graphviz import Source

def print_tree_structure( T, filename='output_tree_structure.txt' ):
	'''
		Print out data describing the structure of a Decision Tree.
		Input: DecisionTreeClassifier T
		Output: description of tree structure, in words.
	'''
	# The decision estimator has an attribute called tree_  which stores the entire
	# tree structure and allows access to low level attributes. The binary tree
	# tree_ is represented as a number of parallel arrays. The i-th element of each
	# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
	# Some of the arrays only apply to either leaves or split nodes, resp. In this
	# case the values of nodes of the other type are arbitrary!
	#
	# Among those arrays, we have:
	#   - children_left, id of the left child of the node
	#   - children_right, id of the right child of the node
	#   - feature, feature used for splitting the node
	#   - threshold, threshold value at the node
	#
	
	# Using those arrays, we can parse the tree structure:
	n_nodes = T.tree_.node_count
	children_left = T.tree_.children_left
	children_right = T.tree_.children_right
	feature = T.tree_.feature
	threshold = T.tree_.threshold

	# Open output file.
	text_file = open( filename, 'w')

	# The tree structure can be traversed to compute various properties such
	# as the depth of each node and whether or not it is a leaf.
	node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
	is_leaves = np.zeros(shape=n_nodes, dtype=bool)
	stack = [(0,-1)] 
	while len(stack) > 0:
		node_id, parent_depth = stack.pop()
		node_depth[node_id] = parent_depth+1

		# If we have a test node
		if (children_left[node_id] != children_right[node_id]):
			stack.append((children_left[node_id], parent_depth+1))
			stack.append((children_right[node_id], parent_depth+1))
		else:
			is_leaves[node_id] = True

	text_file.write("The binary tree structure has %s nodes and has "
			"the following tree structure:\n"% n_nodes)

	for i in range(n_nodes):
		if is_leaves[i]:
			text_file.write("%snode=%s leaf node.\n" % (node_depth[i] * "\t", i))
		else:
			text_file.write("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
				  "node %s.\n"
				  % (node_depth[i] * "\t",
					i,
					children_left[i],
					feature[i],
					threshold[i],
					children_right[i],
					))

	text_file.write("\n\n")
	text_file.write("Maximum tree depth is %s\n" % node_depth.max())
	text_file.write("Total number of leaves is %s\n" % sum(is_leaves))
	text_file.close()


##############################################################################


def print_decision_path(T, feature_test, filename='output_decision_path.txt'):
	'''
		Input: DecisionTreeClassifier T
			   feature_test = 2d array of test features.  Each row is an individual
							sample and each column is a feature.
		Output: Description of the decsion path for each feature in 'feature_test'
				Results are output to filename.txt
	'''
	text_file = open(filename, 'w')
	n_nodes = T.tree_.node_count
	feature = T.tree_.feature
	threshold = T.tree_.threshold
	
	# First let's retrieve the decision path of each sample. The decision_path
	# method allows to retrieve the node indicator functions. A non zero element of
	# indicator matrix at the position (i, j) indicates that the sample i goes
	# through the node j.
	node_indicator = T.decision_path(feature_test)

	# Similarly, we can also have the leaves ids reached by each sample.
	leave_id = T.apply( feature_test )

	# Now, it's possible to get the tests that were used to predict a sample or
	# a group of samples.  First let's make it for the sample.
	sample_id = 0
	node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

	text_file.write("Rules used to predict sample %s:\n" % sample_id)
	for node_id in node_index:
		if leave_id[sample_id] != node_id:
			continue

		if (feature_test[sample_id, feature[node_id]] <= threshold[node_id]):
			threshold_sign = "<="
		else:
			threshold_sign = ">="

		text_file.write("decision id node %s: (feature_test[%s,%s] (= %s) %s %s)\n"
				% (node_id,
					sample_id,
					feature[node_id],
					feature_test[sample_id, feature[node_id]],
					threshold_sign,
					threshold[node_id]))

	# For a group of samples we have the following common node.
	sample_ids = [0,1]
	common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids))

	common_node_id = np.arange(n_nodes)[common_nodes]

	text_file.write("\nThe following samples %s share the node %s in the tree\n"
			% (sample_ids, common_node_id))
	text_file.write("It is %s %% of all nodes.\n" % (100*len(common_node_id) / n_nodes,))

	text_file.close()


##############################################################################

def visualize_decision_tree( clf, featureNames, classNames, outName ):
	'''
		Input: clf = DecisionTreeClassifier that has already been fitted.
			   featureNames = List of strings containing the names of all the features.
			   classNames = List of strings containing the names of the target classes.
			   outName = String of output file name.
		Output: PDF file of tree. File name will be outName.pdf
	'''

	dot_data = tree.export_graphviz(clf, out_file=None, feature_names = featureNames, class_names = classNames, rounded=True, special_characters=True)
	graph = Source(dot_data)
	graph.render(outName)

##############################################################################
