r"""Utilities for Bonsai: Gradient-free Graph Distillation for Node Classification
"""

import os
from typing import List, Tuple
from collections import defaultdict
import heapq
import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
import torch
from torch_geometric.data import Data


def wl2rknn(WL_distances: np.ndarray, *, k: int) -> dict:
    r"""Compute rev-k-nn's from WL distances."""
    assert isinstance(WL_distances, np.ndarray)
    nnodes = WL_distances.shape[0]
    knn = []
    # nnodes is len(WL_distances), that is same index as WL_distances.
    for node in tqdm(range(nnodes), desc="Eval KNN", ascii=True, ncols=120):
        topk = np.argpartition(WL_distances[node], k)[:k]
        knn.append(topk)
    rknn = defaultdict(set)
    for node, knn_node in tqdm(enumerate(knn), desc="Eval rKNN", ascii=True, ncols=120):
        _ = [rknn[q].add(node) for q in knn_node]
        del _
    # rknn has as index the same index as WL_distances
    # the index is used in two places, rknn key and rknn values
    return {"rknn": dict(rknn), "knn": np.asarray(knn)}


def select_max_coverage_rknn_celf(rknn_dict: dict) -> List[int]:
    r"""https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf"""
    covered_nodes = set()
    selected_nodes = []

    # Initialize priority queue with initial marginal gains
    pq = [(-len(neighbors), 0, node) for node, neighbors in rknn_dict.items()]
    heapq.heapify(pq)

    iteration = 1

    while pq:
        # Get the node with the highest marginal gain
        neg_gain, last_iteration, node = heapq.heappop(pq)

        # If this node's gain was last calculated in the current iteration, it's the best node
        if last_iteration == iteration - 1:
            # Early stopping: if the max coverage is zero, append remaining nodes and break
            if neg_gain == 0:
                selected_nodes.extend(node for _, _, node in pq)
                selected_nodes.append(node)
                break

            # Add the selected node to the list and update the set of covered nodes
            selected_nodes.append(node)
            covered_nodes.update(rknn_dict[node])
            del rknn_dict[node]
            iteration += 1
        else:
            # Recalculate the marginal gain
            new_gain = len(set(rknn_dict[node]) - covered_nodes)
            heapq.heappush(pq, (-new_gain, iteration - 1, node))

    return selected_nodes


# Set environment variables to limit the number of threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def collect_features_used(tree) -> List[int]:
    r"""A utility to extract features used by a decision tree.
    """
    features_used = set()

    def traverse(node_id):
        node = tree.tree_
        # if node_id == 2:
        if node.feature[node_id] >= 0:
            # Collect the feature used for splitting at this node
            features_used.add(node.feature[node_id])
            # Recursively traverse left and right children
            traverse(node.children_left[node_id])
            traverse(node.children_right[node_id])

    traverse(0)  # Start traversal from the root node
    return sorted(features_used)  # Return sorted list of features used


def transform_features_with_tree(
    data: Data, ego_graph_vmed: List[torch.Tensor], train: List[int]
) -> Tuple[List[torch.Tensor], List[int]]:
    r"""Extracts most useful features with decision tree.
    """
    # Extract features and labels
    features = [
        ego_graph_vmed_sample.numpy() for ego_graph_vmed_sample in ego_graph_vmed
    ]
    labels = data.y[train].numpy()

    # Normalize features if needed, but binary features are already in range [0, 1]
    # scaler = preprocessing.StandardScaler()
    # features_normalized = scaler.fit_transform(features)

    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth=50)
    clf.fit(features, labels)

    # Get all features used for splitting
    features_used = collect_features_used(clf)
    # Create new feature vectors based on the features used in splitting
    # This will be a binary feature vector where the columns
    # corresponding to features_used are retained
    # and all other columns are set to 0

    # Convert to PyTorch tensor
    # ego_graph_vmed = [ego_graph_vmed[i] for i in features_used]
    filtered_ego_graph_vmed = [tensor[features_used] for tensor in ego_graph_vmed]
    return filtered_ego_graph_vmed, features_used
