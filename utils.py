from collections import defaultdict
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import dense_to_sparse
from sklearn.model_selection import train_test_split
import random
import heapq
from tqdm import tqdm



class my_train_dataset(Dataset):
    def __init__(this, datalist):
        super().__init__()
        this.datalist = datalist

    def len(this):
        return len(this.datalist)

    def get(this, idx):
        return this.datalist[idx]


def is_binary(vec):
    if isinstance(vec, list):
        vec = np.asarray(vec)
    return len(np.where(np.logical_and(vec != 0, vec != 1))[0]) == 0


def is_int(vec):
    if isinstance(vec, list):
        vec = np.asarray(vec)
    int_vec = vec.astype(int)
    return np.sum(int_vec - vec) == 0


def size_nx(Glist, sparse=True):
    if not sparse:
        raise NotImplementedError
    nints = 0
    nfloats = 0
    nodes = defaultdict(lambda: 0)
    features = {}
    for g in Glist:
        for node in g.nodes:
            nodes[node] += 1
            features[node] = g.nodes[node]["x"]
        nints += 2 * g.number_of_edges()
    nints += sum(v for v in nodes.values())  # each node will map to these many nodes
    nints += len(nodes)  # one for each key
    nints += len(features)  # one for each key
    node = next(iter(nodes.keys()))
    feature = features[node]
    if not is_binary(feature):
        if is_int(feature):
            for feat in features.values():
                nints += 2*np.where(feat != 0)[0].shape[0]
                # nints += np.where(np.asarray(feat) != 0)[0].shape[0]
        else:
            for feat in features.values():
                nints += 3*np.where(feat != 0)[0].shape[0]
                # nfloats += np.where(np.asarray(feat) != 0)[0].shape[0]
    else:
        for feat in features.values():
            nints += np.where(feat != 0)[0].shape[0]
            # this is a bottleneck
    return nints * 2 + nfloats * 4


def size_pyg(G, sparse=True):
    if not sparse:
        raise NotImplementedError
    nfeats = 0
    mult = 2
    x = G.x[0].numpy()
    if not is_binary(x):
        if is_int(x):
            mult = 3  # need to store int values along with coordinates
        else:
            mult = 4  # need to store float values along with coordinates
            mult = 3
    for x in G.x.numpy():
        nfeats += np.where(x != 0)[0].shape[0]
    nedges = G.edge_index.shape[1]
    return (mult * nfeats + nedges * 2) * 2


def wl2rknn(WLdistances, *, k):
    assert isinstance(WLdistances, np.ndarray)
    nnodes = WLdistances.shape[0]
    knn = []
    # nnodes is len(WLdistances), that is same index as WLdistances.
    for node in tqdm(range(nnodes), desc = " Eval KNN", ascii=True, ncols=120):
        topk = np.argpartition(WLdistances[node], k)[:k]
        knn.append(topk)
    rknn = defaultdict(set)
    for node, knn_node in tqdm(enumerate(knn), desc = " Eval rKNN", ascii=True, ncols=120):
        _ = [rknn[q].add(node) for q in knn_node]
        del _
    # rknn has as index the same index as WL_distances
    # the index is used in two places, rknn key and rknn values
    return {"rknn": dict(rknn), "knn": np.asarray(knn)}


def rknn2nodes(rknn, *, target_size, not_chosen_yet=None, seed_target=None, th=0):
    C = [] if seed_target is None else list(seed_target)
    oglen = len(C)
    th = 0 if seed_target is None else th
    if not_chosen_yet is None:
        sorted_rknn = sorted(rknn.items(), key=lambda x: -len(x[1]))
    else:
        sorted_rknn = sorted(not_chosen_yet.items(), key=lambda x: -len(x[1]))
    not_chosen = []
    for node, rknn_node in sorted_rknn:
        maxel = -float("inf")
        for u in C:
            leninter = len(rknn_node.intersection(rknn[u]))
            denom = min(len(rknn_node), len(rknn[u]))
            if (leninter / denom - maxel) > 0:
                maxel = leninter / denom
        if maxel < th:
            C.append(node)
        else:
            not_chosen.append((node, rknn_node))
    # -------------------------------------------------------------
    if len(C) < target_size and th < 1:
        return rknn2nodes(
            rknn,
            not_chosen_yet=dict(not_chosen),
            th=th + 0.1,
            target_size=target_size,
            seed_target=C,
        )
    # -------------------------------------------------------------
    return tuple(C)

def select_max_coverage_rknn(rknn_dict):
    covered_nodes = set()
    selected_nodes = []

    while rknn_dict:
        # Step 1: Find the node with the maximum number of uncovered nodes in its rknn
        max_node = None
        max_coverage = -1
        
        for node, neighbors in rknn_dict.items():
            # Calculate the number of new nodes this node's rknn would cover
            new_coverage = len(neighbors - covered_nodes)
            
            if new_coverage > max_coverage:
                max_coverage = new_coverage
                max_node = node
            elif new_coverage == max_coverage:
                # In case of tie, break randomly
                if random.choice([True, False]):
                    max_node = node

        # Early stopping: if the max coverage is zero, append remaining nodes and break
        if max_coverage == 0:
            selected_nodes.extend(rknn_dict.keys())
            break

        # Step 2: Add the selected node to the list and update the set of covered nodes
        if max_node is not None:
            selected_nodes.append(max_node)
            covered_nodes.update(rknn_dict[max_node])
            
            # Remove the selected node from the dictionary to avoid reprocessing it
            del rknn_dict[max_node]
    return selected_nodes

# ADD CELF IMPLEMENTATION

def select_max_coverage_rknn_celf(rknn_dict):
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


def graph_to_vmed(ego_graph_data: Data):
    edge_index = ego_graph_data.edge_index
    num_nodes = ego_graph_data.x.shape[0]
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    src, dst = edge_index
    feats = ego_graph_data.x
    nbrs_feats = feats[dst]
    rep = nbrs_feats.mean(0)
    return rep


def nxego2data(ego_graph, *, center_node, ix, total):
    from datetime import datetime

    timestamp = datetime.now().strftime("%H:%M:%S -")
    print(timestamp, f"{ix}/{total}")
    nodes = [n for n in ego_graph.nodes]
    nodes.pop(nodes.index(center_node))
    nodes.insert(0, center_node)
    mapping = {v: k for k, v in enumerate(nodes)}
    adj = torch.tensor(nx.adjacency_matrix(ego_graph, nodelist=nodes).todense())
    ego_graph = nx.relabel_nodes(ego_graph, mapping)
    edge_index, edge_weight = dense_to_sparse(adj)
    features_map = nx.get_node_attributes(ego_graph, name="x")
    features = [features_map[n] for n in range(len(nodes))]
    label = nx.get_node_attributes(ego_graph, name="y")[0]
    target_node = torch.zeros(len(nodes), dtype=torch.bool)
    target_node[0] = True
    data = Data(
        x=torch.tensor(features),
        edge_index=edge_index,
        y=torch.tensor(label),
        target_node=target_node,
    )
    # assert size_nx(ego_graph) == size_pyg(data)
    return data


# +
import os
import numpy as np
from sklearn.decomposition import PCA
import torch

# Set environment variables to limit the number of threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def transform_features_PCA(data, variance_retained=0.95):
    # dataX = data.x
    node_features = data.x.numpy()
    pca = PCA(n_components=variance_retained)
    reduced_feature_vectors = pca.fit_transform(node_features)
    data.x = torch.tensor(reduced_feature_vectors, dtype=torch.float)

    max_val = torch.max(data.x).item()
    min_val = torch.min(data.x).item()
    avg_val = torch.mean(data.x).item()
    std_val = torch.std(data.x).item()

    print(f"Max value: {max_val}")
    print(f"Min value: {min_val}")
    print(f"Average value: {avg_val}")
    print(f"Standard deviation: {std_val}")

    min_val = torch.min(data.x)
    max_val = torch.max(data.x)
    normalized_data = (data.x - min_val) / (max_val - min_val)

    # Replace values between -0.1 and 0.1 with 0
    data.x = torch.where(
    (normalized_data > -0.6) & (normalized_data < 0.6), 
    torch.tensor(0, dtype=torch.float),
    torch.where(
        normalized_data > 0.6,
        torch.tensor(1, dtype=torch.float),
        torch.tensor(-1, dtype=torch.float)
    )
    )

    data.x = torch.tensor(data.x)

    zero_count = torch.sum(data.x == 0).item()

    print(f"Number of zero-valued nodes: {zero_count}")

    return data

# +
import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


def collect_features_used(tree):
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

def transform_features_with_tree(data, ego_graph_vmed, nnodes, train):
    # Extract features and labels
    # features = ego_graph_vmed.numpy()
    # train, test = train_test_split(range(nnodes), test_size=0.2, random_state=42)
    features = [ego_graph_vmed_sample.numpy() for ego_graph_vmed_sample in ego_graph_vmed]
    labels = data.y[train].numpy()
    
    # Normalize features if needed, but binary features are already in range [0, 1]
    # scaler = preprocessing.StandardScaler()
    # features_normalized = scaler.fit_transform(features)

    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth=50) #marji meri
    clf.fit(features, labels)
    predictions = clf.predict(features)
    
    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f'accuracy: {accuracy}')
    
    # Get all features used for splitting
    features_used = collect_features_used(clf)
    print("len features used is: ", len(features_used))
    # Create new feature vectors based on the features used in splitting
    # This will be a binary feature vector where the columns corresponding to features_used are retained
    # and all other columns are set to 0
    # new_feature_vectors = np.zeros_like(features, dtype=int)
    # new_feature_vectors[features_used] = features[features_used]

    # Convert to PyTorch tensor
    # ego_graph_vmed = [ego_graph_vmed[i] for i in features_used]
    filtered_ego_graph_vmed = [tensor[features_used] for tensor in ego_graph_vmed]


    # Count the number of zero-valued nodes
    # zero_count = torch.sum(data.x == 0).item()
    # print(f"Number of zero-valued nodes: {zero_count}")
    depth = clf.get_depth()
    print(f"depth: {depth}")
    return filtered_ego_graph_vmed, features_used
