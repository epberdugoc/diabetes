import numpy as np
'''
	Print out data describing the structure of a Decision Tree.
	Input: DecisionTreeClassifier T
	Output: description of tree structure, in words.
'''
def print_tree_structure( T ):
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

	print("The binary tree structure has %s nodes and has "
			"the following tree structure:"% n_nodes)

	for i in range(n_nodes):
		if is_leaves[i]:
			print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
		else:
			print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
				  "node %s."
				  % (node_depth[i] * "\t",
					i,
					children_left[i],
					feature[i],
					threshold[i],
					children_right[i],
					))
	print()
