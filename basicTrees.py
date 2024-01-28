import numpy as np

def get_entropy(y):
    """
	y (ndarray): Numpy array
	Returns: entropy (float): Entropy at that node
    """
    if len(y) == 0:
        return 0.

    p_1 = sum(y) / len(y)

    if p_1 == 0 or p_1 == 1:
        return 0.

    return -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)

def split_dataset(X, node_indices, feature):
	"""
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
	"""

	left_indices = []
    right_indices = []

	for i in node_indices:
		if X[i, feature] == 1:
	        left_indices.append(i)
		else:
			right_indices.append(i)

    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    """
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        cost (float):        Cost computed

    """

    left_indices, right_indices = split_dataset(X, node_indices, feature)

    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]

    m = len(node_indices)
    w_left = len(left_indices) / m
    w_right = len(right_indices) / m

    entropy_root = compute_entropy(y_node)
    entropy_left = compute_entropy(y_left)
    entropy_right = compute_entropy(y_right)

    entropy_change = w_left * entropy_left + w_right * entropy_right

    return entropy_root - entropy_change

def get_best_split(X, y, node_indices):
    """
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    num_features = X.shape[1]
    best_feature = -1
    best_info_gain = 0

    for feature_idx in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature_idx)

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature_idx

    return best_feature
