from collections import Counter

import numpy as np
import sklearn.metrics as metrics
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score as nmi


def calculate_cost_matrix(C: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm.

    Parameters
    ----------
    C : np.ndarray
        Confusion matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cost matrix.
    """

    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices.

    Parameters
    ----------
    indices : np.ndarray
        Indices of the clusters.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def acc_score_metric(
    cluster_assignments: np.ndarray, y: np.ndarray, n_clusters: int
) -> float:
    """
    Compute the accuracy score of the clustering algorithm.

    Parameters
    ----------
    cluster_assignments : np.ndarray
        Cluster assignments for each data point.
    y : np.ndarray
        Ground truth labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    float
        The computed accuracy score.

    Notes
    -----
    This function takes the `cluster_assignments` which represent the assigned clusters for each data point,
    the ground truth labels `y`, and the number of clusters `n_clusters`. It computes the accuracy score of the
    clustering algorithm by comparing the cluster assignments with the ground truth labels. The accuracy score
    is returned as a floating-point value.
    """

    confusion_matrix = metrics.confusion_matrix(y, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters=n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    print(metrics.confusion_matrix(y, y_pred))
    accuracy = np.mean(y_pred == y)
    return accuracy


def nmi_score_metric(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the normalized mutual information score of the clustering algorithm.

    Parameters
    ----------
    cluster_assignments : np.ndarray
        Cluster assignments for each data point.
    y : np.ndarray
        Ground truth labels.

    Returns
    -------
    float
        The computed normalized mutual information score.

    Notes
    -----
    This function takes the `cluster_assignments` which represent the assigned clusters for each data point
    and the ground truth labels `y`. It computes the normalized mutual information (NMI) score of the clustering
    algorithm. NMI measures the mutual dependence between the cluster assignments and the ground truth labels,
    normalized by the entropy of both variables. The NMI score ranges between 0 and 1, where a higher score
    indicates a better clustering performance. The computed NMI score is returned as a floating-point value.
    """
    return nmi(cluster_assignments, y)


def purity_score_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure the arrays have the same shape
    assert y_pred.size == y_true.size

    # Calculate the cluster sizes
    cluster_sizes = Counter(y_pred)

    # Initialize purity
    purity = 0

    # Iterate through each cluster
    for cluster in set(y_pred):
        # Get the indices of points in this cluster
        cluster_indices = np.where(y_pred == cluster)[0]

        # Count the true labels in this cluster
        true_labels = y_true[cluster_indices]
        cluster_label_distribution = Counter(true_labels)

        # Find the most common label in the cluster
        max_count = max(cluster_label_distribution.values())

        # Add to purity
        purity += max_count

    # Divide by total number of samples
    purity /= len(y_true)

    return purity


def run_evaluate(cluster_assignments, y, n_clusters):
    acc_score = acc_score_metric(cluster_assignments, y, n_clusters=n_clusters)
    nmi_score = nmi_score_metric(cluster_assignments, y)
    purity_score = purity_score_metrics(cluster_assignments, y)
    print(f"ACC: {np.round(acc_score, 3)}")
    print(f"NMI: {np.round(nmi_score, 3)}")
    print(f"PURITY: {np.round(purity_score, 3)}")
    results = {
        "ACC": np.round(acc_score, 3),
        "NMI": np.round(nmi_score, 3),
        "PURITY": np.round(purity_score, 3),
    }
    return results
