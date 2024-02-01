import numpy as np

def find_closest_centroids(X, centroids):
    """
    Find the closest centroids to each point in X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to find the closest centroids to.

    centroids : array-like, shape (n_centroids, n_features)
        The centroids to find the closest centroids to.

    Returns
    -------
    closest_centroids : array-like, shape (n_samples, n_centroids)
        The closest centroids to each point in X.
    """
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(m):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distances)

    return closest_centroids

def compute_centroids(X,idx,K):
    m,n = X.shape
    centroids = np.zeros((K,n))

    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points,axis=0)
    return centroids
