from sklearn.cluster import DBSCAN
import numpy as np


def create_clusters(points, eps: float = 9, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    # fit model and predict clusters
    yhat = model.fit_predict(points)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    # create scatter plot for samples from each cluster
    result = []
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        result.append(points[row_ix, :][0])
    return result
