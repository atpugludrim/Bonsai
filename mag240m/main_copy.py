r"""This is the implementation of Bonsai: Gradient-free Graph Distillation
for Node Classification. Please read README.md
"""

import gc
import copy
import time
import typing as t
import argparse
from pathlib import Path
from collections import defaultdict
import hnswlib

from tqdm import tqdm

import networkx as nx
import numpy as np
from scipy.sparse import coo_array

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from tensorboardX import SummaryWriter

from memory_profiler import profile


from utils import (
    wl2rknn,
    select_max_coverage_rknn_celf,
)
from utils import transform_features_with_tree
from model import GCN, GCN_inductive
from WL_Distance2 import compute_wl_representations

from ogb.lsc import MAG240MDataset
from sklearn.preprocessing import StandardScaler

import tracemalloc
import pickle
import os





ADJ = None
FEAT_MULTIPLIER = 1
GLOBAL_NEIGHBORS_DICT = {}
GLOBAL_FEATS = None
FEAT_LEN = None
SAINT_DATASETS = ["flickr", "ogbn-arxiv", "reddit", "mag240m"]


def log(x: str) -> None:
    r"""Function that used to log its string contents to file. Now, a dummy
    function.
    """
    pass


def load_dataset(dataset_name: str, root: t.Union[str, Path], subset_frac: float = 0.1) -> dict:
    """
    Utility for loading a subset of the MAG240M dataset and re-indexing nodes so edges
    fit into a smaller adjacency matrix.

    Params:
        dataset_name: str - Should be "mag240m".
        root: str | pathlib.Path - Path of the root directory.
        subset_frac: float - Fraction of the dataset to load (e.g., 0.001 for 0.1%).

    Returns:
        dict containing:
          - "data": PyG Data object with x, y, edge_index (reindexed)
          - "scaler": The fitted StandardScaler
    """
    if dataset_name != "mag240m":
        raise ValueError("This function is tailored for the 'mag240m' dataset only.")

    log("Loading MAG240M dataset...")
    dataset = MAG240MDataset(root=root)

    # Total number of papers
    num_papers = dataset.num_papers
    log(f"Total number of papers: {num_papers}")
    print(f"Total number of papers: {num_papers}")

    # Use only labeled training nodes to avoid -1 labels
    rng = np.random.default_rng(seed=42)  # For reproducibility
    train_idx = dataset.get_idx_split('train')  # All labeled training nodes

    # Determine subset size
    if subset_frac <= 0 or subset_frac > 1:
        raise ValueError("subset_frac should be between (0,1].")
    subset_size = int(len(train_idx) * subset_frac)
    if subset_size < 1:
        subset_size = 1

    # Randomly sample a subset of training nodes
    subset_indices = rng.choice(train_idx, size=subset_size, replace=False)

    x_np = dataset.all_paper_feat[subset_indices]
    y_np = dataset.all_paper_label[subset_indices]
    nnodes = x_np.shape[0]
    # breakpoint()

    log(f"Loading a subset of {subset_size} papers ({subset_frac * 100:.4f}%)")
    print(f"Loading a subset of {subset_size} papers ({subset_frac * 100:.4f}%)")

    # breakpoint()
    print("made x_np")

    # Retrieve the 'cites' edges (shape: [2, num_cites])
    edge_index_np = dataset.edge_index('paper', 'cites', 'paper')

    # Filter edges where both source and target are in subset_indices
    subset_set = set(subset_indices)
    mask = np.isin(edge_index_np, subset_indices)
    valid_edges = mask[0] & mask[1]
    edge_index_subset = edge_index_np[:, valid_edges]
    log(f"Number of citation edges in the subset (before reindexing): {edge_index_subset.shape[1]}")
    print(f"Number of citation edges in the subset (before reindexing): {edge_index_subset.shape[1]}")

    # -------------------------
    #  Reindex from old IDs to [0..subset_size-1]
    # -------------------------
    # Build a map from original paper ID -> new index
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(subset_indices)}

    # Reindex the edges
    src_old = edge_index_subset[0, :]
    dst_old = edge_index_subset[1, :]

    # Map each old ID to new ID
    src_new = np.array([old_to_new[old] for old in src_old], dtype=np.int64)
    dst_new = np.array([old_to_new[old] for old in dst_old], dtype=np.int64)

    edge_index_subset_reindexed = np.stack([src_new, dst_new], axis=0)
    log(f"Number of citation edges in the subset (after reindexing): {edge_index_subset_reindexed.shape[1]}")
    print(f"Number of citation edges in the subset (after reindexing): {edge_index_subset_reindexed.shape[1]}")

    # Convert to torch tensors
    x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
    y = torch.from_numpy(np.ascontiguousarray(y_np)).long()
    edge_index = torch.from_numpy(np.ascontiguousarray(edge_index_subset_reindexed)).long()
    # < creating tkipf style adj >
    import scipy.sparse as sp
    from torch_sparse import SparseTensor

    d = np.ones(edge_index.shape[1])
    r = edge_index[0].cpu().numpy()
    c = edge_index[1].cpu().numpy()
    n = nnodes
    adj = sp.csr_matrix((d, (r, c)), shape=(n, n))
    adj = adj.tolil()
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    adj = adj.dot(r_mat_inv)
    adj = adj.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(adj.row).unsqueeze(1)
    sparsecol = torch.LongTensor(adj.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(adj.data)
    adj = torch.sparse.FloatTensor(
        sparseconcat.t(), sparsedata, torch.Size(adj.shape)
    )
    adj = SparseTensor(
        row=adj._indices()[0],
        col=adj._indices()[1],
        value=adj._values(),
        sparse_sizes=adj.size(),
    )
    # < / >

    # Create a PyG Data object
    pyg_data = Data(x=x, edge_index=edge_index, y=y, adj=adj)

    # Feature scaling using StandardScaler
    log("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(x_np)  # Fit on the subset
    x_scaled = scaler.transform(x_np)
    pyg_data.x = torch.from_numpy(x_scaled).float()

    log("MAG240M subset loaded and features scaled.")
    print("MAG240M subset loaded and features scaled.")
    return {"data": pyg_data, "scaler": scaler}


def dist2rknn_sorting(WL_dist: np.ndarray, k: int) -> t.List[int]:
    r"""Converts WL Distance matrix into nodes sorted jointly by rev-k-nn
    and max-coverage.
    Params:
    WL_dist: np.ndarray - a nxn (or nxm) np.ndarray
    k: int - the k in rev-k-nn
    returns: a list of integers representing node ids sorted in order of
             preference.
    """
    rknn_result = wl2rknn(WL_dist, k=k)
    sorted_nodes = select_max_coverage_rknn_celf(rknn_result["rknn"])
    return sorted_nodes


def rknn_sorted2budget_select_merged(
    sorted_nodes: t.List[int], train: t.List[int], target_size: float
) -> t.List[int]:
    r"""Only works when features are binary as of now. PubMed and
    ogbn-arxiv might not work. I have made some adaptations. But
    let's see whether they really work.

    For 'reddit' and 'ogbn-arxiv': Features are dense and float.
    For 'flickr': Features are sparse integers, but not binary.
    For 'PubMed': Features are sparse floats.
    """
    size_selected_nodes = []
    selected_nodes = set()
    selected_edges = set()
    ints_till_now_due_to_nodes = 0
    not_selected_for_n_consecutive_iters = 0

    for _, node in tqdm(
        enumerate(sorted_nodes),
        ascii=True,
        ncols=120,
        total=len(sorted_nodes),
        desc="rknn_sorted2budget_select_merged",
    ):
        candidate_selected_nodes = set()
        candidate_selected_edges = set()

        if train[node] not in selected_nodes:
            candidate_selected_nodes.add(train[node])
        node_adj_set = GLOBAL_NEIGHBORS_DICT[train[node]]

        for nbr_node in node_adj_set:
            if nbr_node not in selected_nodes:
                candidate_selected_nodes.add(nbr_node)
            edge = tuple(sorted([nbr_node, train[node]]))
            if edge not in selected_edges:
                candidate_selected_edges.add(edge)
            nbr_adj_set = GLOBAL_NEIGHBORS_DICT[nbr_node]
            ego_nodes = node_adj_set.intersection(nbr_adj_set)
            for nbr_nbr_node in ego_nodes:
                edge = tuple(sorted([nbr_node, nbr_nbr_node]))
                if edge not in selected_edges:
                    candidate_selected_edges.add(edge)
        num_edges = len(selected_edges) + len(candidate_selected_edges)
        added_nodes = candidate_selected_nodes
        ints_due_to_nodes = 0
        for added_node in added_nodes:
            feat_ints = FEAT_LEN[added_node]
            ints_due_to_nodes += feat_ints
        ints_due_to_nodes *= FEAT_MULTIPLIER
        ints_due_to_nodes += ints_till_now_due_to_nodes
        ints_due_to_edges = num_edges * 2
        total_ints = ints_due_to_nodes + ints_due_to_edges
        size_till_now = total_ints * 2
        if size_till_now < target_size:
            size_selected_nodes.append(node)
            selected_nodes.update(added_nodes)
            selected_edges.update(candidate_selected_edges)
            ints_till_now_due_to_nodes = ints_due_to_nodes
        else:
            not_selected_for_n_consecutive_iters += 1
            if not_selected_for_n_consecutive_iters > 100:
                break
    return size_selected_nodes


def train_backend_inductive(
    model: nn.Module,
    nepochs: int,
    data: Data,
    data_syn: Data,
    splits: dict,
    writer: SummaryWriter,
) -> float:
    r"""Utility that trains a model on the synthetic dataset.
    Params:
    model: nn.Module - the object of the GCN model to train
    nepochs: int - the number of epochs of training
    data: torch_geometric.data.Data - full dataset, used for validation
                                      and testing
    data_syn: torch_geometric.data.Data - synthetic dataset
    splits: dict - a dictionary containing train/val/test splits
    writer: tensorboardX.SummaryWriter - for training logs
    returns: accuracy
    """
    opt = optim.Adam(model.parameters())
    loss_fn = F.nll_loss
    loop = tqdm(range(nepochs), ascii=True, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    d = data
    d = d.to(device, "x", "edge_index", "y", "adj")  #adj uddaya hai yaaha se
    d_syn = data_syn
    d_syn = d_syn.to(device, "x", "adj", "y", "target")
    test = splits["test"]
    val = splits["val"]
    for epoch in loop:
        model.train()
        out = model(d_syn.x, d_syn.adj)
        loss = loss_fn(out[d_syn.target], d_syn.y[d_syn.target])
        opt.zero_grad()
        #breakpoint()
        loss.backward()
        opt.step()
        writer.add_scalar("loss/train", loss.item(), epoch)
        model.eval()
        with torch.no_grad():
            out = model(d.x, d.adj)
            loss = loss_fn(out[val], d.y[val])
            writer.add_scalar("loss/val", loss.item(), epoch)
            preds = out[val].max(1)[1].cpu().numpy()
            acc = accuracy_score(d.y[val].cpu().numpy(), preds)
            writer.add_scalar("acc/val", acc, epoch)
            if acc > best_acc_val:
                best_acc_val = acc
                weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    with torch.no_grad():
        out = model(d.x, d.adj)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    writer.add_scalar("test_acc/test", acc)
    return acc


def train_backend_pyg(
    model: nn.Module,
    nepochs: int,
    data: Data,
    data_syn: Data,
    splits: dict,
    writer: SummaryWriter,
) -> float:
    r"""Utility that trains a model on the synthetic dataset.
    Params:
    model: nn.Module - the object of the GCN model to train
    nepochs: int - the number of epochs of training
    data: torch_geometric.data.Data - full dataset, used for validation
                                      and testing
    data_syn: torch_geometric.data.Data - synthetic dataset
    splits: dict - a dictionary containing train/val/test splits
    writer: tensorboardX.SummaryWriter - for training logs
    returns: accuracy
    """
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loop = tqdm(range(nepochs), ascii=True, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    d = data
    d = d.to(device, "x", "edge_index", "y", "adj")
    d_syn = data_syn
    d_syn = d_syn.to(device, "x", "edge_index", "y", "target", "adj")
    test = splits["test"]
    val = splits["val"]
    from my_profiling import profile

    with profile(False):
        for epoch in loop:
            model.train()
            out = model(d_syn.x, d_syn.edge_index)
            loss = loss_fn(out[d_syn.target], d_syn.y[d_syn.target])
            opt.zero_grad()
            loss.backward()
            opt.step()
            writer.add_scalar("loss/train", loss.item(), epoch)
            model.eval()
            with torch.no_grad():
                out = model(d.x, d.edge_index)
                loss = loss_fn(out[val], d.y[val])
                writer.add_scalar("loss/val", loss.item(), epoch)
                preds = out[val].max(1)[1].cpu().numpy()
                acc = accuracy_score(d.y[val].cpu().numpy(), preds)
                writer.add_scalar("acc/val", acc, epoch)
                if acc > best_acc_val:
                    best_acc_val = acc
                    weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    with torch.no_grad():
        out = model(d.x, d.edge_index)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    writer.add_scalar("test_acc/test", acc)
    return acc


def train_model(
    model_type: str,
    model: nn.Module,
    nepochs: int,
    data: Data,
    data_syn: Data,
    splits: dict,
    writer: SummaryWriter,
) -> float:
    r"""Trains a GCN model.
    Params:
    model_type: str - whether the model is GCN or GCN_inductive (for
                      larger graph_saint datasets)
    model: nn.Module - the corresponding model object to train
    nepochs: int - number of epochs to train the model for
    data: torch_geometric.data.Data - full dataset, used for validation
                                      and testing
    data_syn: torch_geometric.data.Data - synthetic dataset
    splits: dict - a dictionary containing train/val/test splits
    writer: tensorboardX.SummaryWriter - for training logs
    returns: accuracy
    """
    if model_type == "GCN_inductive":
        return train_backend_inductive(model, nepochs, data, data_syn, splits, writer)
    return train_backend_pyg(model, nepochs, data, data_syn, splits, writer)


def size(nnodes: int, nedges: int, feats: int, dtype: str = "int") -> int:
    r"""Utility to compute the size of full dataset.
    Params:
    nnodes: int
    nedges: int
    feats: int - number of features per node
    dtype: str - default="int", a string which can be either "int" or "float"
                 that represents the dtype of the node features
    returns: size as int
    """
    mx = 1 if dtype == "int" else 2 if dtype == "float" else None
    return (nnodes * feats * mx + nedges * 2) * 2


def build_neighborhood_dict_sparse() -> dict:
    r"""Utility to build dictionary of neighborhoods."""
    neighbors = defaultdict()
    for node in tqdm(
        range(ADJ.shape[0]),
        ascii=True,
        total=ADJ.shape[0],
        ncols=120,
        desc="build neighbor",
    ):
        node_nbrs = set(ADJ[[node]].tocoo().col)
        node_nbrs.add(node)
        neighbors[node] = node_nbrs
    return neighbors


def compute_degree_weighted_repr_for_node_orig(node_id: int) -> torch.Tensor:
    r"""Computes WL Representation for node."""

    node_neighbors = GLOBAL_NEIGHBORS_DICT[node_id]
    total_degree = 0
    weighted_sum = torch.zeros_like(GLOBAL_FEATS[node_id])

    for nbr in node_neighbors:
        nbr_neighbors = GLOBAL_NEIGHBORS_DICT[nbr]
        local_degree = (
            len(node_neighbors.intersection(nbr_neighbors)) - 1
        )  # -1 done because both nodes are included nbr and central node, but degree added is just 1
        total_degree += local_degree
        torch.add(
            weighted_sum, torch.mul(GLOBAL_FEATS[nbr], local_degree), out=weighted_sum
        )  # Sum up the weighted features

    if total_degree > 0:
        WL_reps_n = weighted_sum / total_degree  # Element-wise division
    else:
        WL_reps_n = torch.zeros_like(
            GLOBAL_FEATS[node_id]
        )  # Handle cases where there is no local degree [should not occur ideally]

    return WL_reps_n


def repr_to_dist(
    degree_weighted_repr: t.List[torch.Tensor], frac_to_sample: int = 1
) -> np.ndarray:
    r"""Converts WL Representations to WL Distances.
    Params:
    degree_weighted_repr: List[torch.Tensor] - list of WL representations
    frac_to_sample: int - default=1, for sampled rev-k-nn based WL distance
    returns: np.ndarray representing WL distances.
    """
    if frac_to_sample == 1:
        distance_matrix = pairwise_distances(
            np.array(degree_weighted_repr), n_jobs=20
        )
    else:
        n = len(degree_weighted_repr)
        m = min(max(int(frac_to_sample * n), 1), n)
        nodes = np.random.choice(range(n), (m,), replace=False)
        node_repr = [degree_weighted_repr[node] for node in nodes]
        distance_matrix = pairwise_distances(
            np.array(node_repr), np.array(degree_weighted_repr), n_jobs=20
        )
        distance_matrix = np.transpose(distance_matrix)
    return distance_matrix


def match_distribution(
    merged_graph: nx.Graph,
    data: Data,
    train: t.List[int],
    rknn_ranked_nodes: t.List[int],
) -> nx.Graph:
    r"""This function matches the class distribution of merged_graph to
    that of the original class distribution in data as closely as possible.

    Params:
    merged_graph: nx.Graph - the synthetic dataset as a networkx graph
    data: torch_geometric.data.Data - the full dataset
    train: List[int] - representing train nodes
    rknn_ranked_nodes: List[int] - nodes in the rknn-max-coverage sorted
                                   order for insertion
    returns: nx.Graph updated to match class-distribution
    """
    # num_nodes = merged_graph.number_of_nodes()
    num_nodes = sum(
        1 for _, v in merged_graph.nodes(data=True) if v["target"]
    )  # only sum trainable nodes
    # num_nodes = 1793

    labels_train = data.y[train]

    class_counts = torch.bincount(labels_train).float()

    total_train_nodes = len(train)
    class_distribution_scaled = (class_counts / total_train_nodes) * num_nodes

    class_distribution_scaled = class_distribution_scaled.long()

    class_dict = {i: [] for i in range(len(class_distribution_scaled))}

    rknn_ranked_nodes = [train[node] for node in rknn_ranked_nodes]
    # ^^ correction, going from I2 to I1
    for node in rknn_ranked_nodes:
        node_class = data.y[node].item()
        class_dict[node_class].append(node)

    all_nodes_to_add = set()
    for class_label, scaled_count in enumerate(class_distribution_scaled):
        print(f"doing for class {class_label}")
        merged_class_nodes = [
            node
            for node, node_attrs in merged_graph.nodes(data=True)
            if data.y[node].item() == class_label and node_attrs["target"]
        ]

        # kept an error bound of +- 10%
        lower_bound = 0.99 * scaled_count.item()
        upper_bound = 1.01 * scaled_count.item()

        current_count = len(merged_class_nodes)

        if current_count < lower_bound:
            nodes_to_add = class_dict[class_label]
            cnt = 0
            for node in nodes_to_add:
                if node not in merged_graph:
                    merged_graph.add_node(node)
                    all_nodes_to_add.add(node)
                    # Add edges from the node to nodes already in merged graph using data.edge_index
                    # data.edge_index is [2, N] where N is the number of edges
                    # for i in range(data.edge_index.shape[1]):
                    #     u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()

                    #     # If node is u or v, and the other node is in the graph, add the edge
                    #     if node == u and v in merged_graph.nodes:
                    #         if not merged_graph.has_edge(node, v):
                    #             merged_graph.add_edge(node, v)
                    #     elif node == v and u in merged_graph.nodes:
                    #         if not merged_graph.has_edge(node, u):
                    #             merged_graph.add_edge(node, u)
                    cnt += 1
                if cnt >= lower_bound - current_count:
                    break

        elif current_count > upper_bound:  # Need to remove nodes
            merged_class_nodes_sorted = sorted(
                merged_class_nodes,
                key=lambda node: rknn_ranked_nodes.index(node) if node in rknn_ranked_nodes else float('inf')
            )
            nodes_to_remove = merged_class_nodes_sorted[-(current_count - int(upper_bound)):]
            for node in nodes_to_remove:
                if node in merged_graph:
                    merged_graph.remove_node(node)
                else:
                    continue

    P1 = lambda u, v: u in all_nodes_to_add and v in all_nodes_to_add
    P2 = lambda u, v: u in all_nodes_to_add and v in merged_graph.nodes
    P3 = lambda u, v: u in merged_graph.nodes and v in all_nodes_to_add
    # P stands for (boolean) predicate
    # P1: make edges between two nodes that have been added
    # P2 and P3: make edges between one node that was already in the graph, and one that has been added
    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        if P1(u, v) or P2(u, v) or P3(u, v):
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v)
    return merged_graph

@profile
def main():
    tracemalloc.start()

# Code block to profile
    start_snapshot = tracemalloc.take_snapshot()
    r"""The main driver code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_size_frac", required=True, type=float)
    parser.add_argument("--nepochs", default=100, type=int)
    parser.add_argument("--k", default=5, type=int, help="The parameter k in kNN.")
    parser.add_argument("--saved_data_path", default="saved_ours/mag240m-0.0005/data_m_0.9.pt")
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Bool flag: save condensed dataset or not",
    )
    parser.add_argument(
        "--dataset",
        default="cora",
        choices=[
            "cora",
            "citeseer",
            "pubmed",
            "ogbn-products",
            "flickr",
            "ogbn-arxiv",
            "reddit",
            "mag240m"
        ],
    )
    parser.add_argument(
        "--subset_frac",
        type=float,
        default=1,  # Default to 0.1%
        help="Fraction of the MAG240M dataset to load (only applicable for 'mag240m')."
    )
    args = parser.parse_args()
    if args.dataset == "mag240m":
        dataset = load_dataset(args.dataset, root="/DATATWO/datasets/mag240m", subset_frac=args.subset_frac)
    else:
        dataset = load_dataset(args.dataset, root="datasets")
    data = dataset["data"]
    scaler = dataset["scaler"]

    nnodes = data.x.shape[0]
    nfeats = data.x.shape[1]
    nedges = data.edge_index.shape[1]
    t1 = time.perf_counter()
    row, col = data.edge_index
    weights = np.ones(len(row))
    row, col = row.numpy(), col.numpy()
    adj = coo_array((weights, (row, col)), shape=(nnodes, nnodes))
    adj = adj.tocsr()
    t2 = time.perf_counter()
    log(f"sparse {t2 - t1:.2f}s")
    global ADJ
    ADJ = adj
    dtype = (
        "int"
        if args.dataset in ["cora", "citeseer", "flickr"]
        else "float" if args.dataset in ["ogbn-arxiv", "reddit", "pubmed", "mag240m"] else None
    )
    size_full = size(nnodes, nedges, nfeats, dtype)
    target_size = float(f"{args.target_size_frac * size_full:.2f}")
    # nclasses = len(set(data.y.reshape(-1).tolist()))
    nclasses = data.y.max().item() + 1

    print(f"the dataset is {size_full} and we take it to {target_size}")


    train, test = train_test_split(range(nnodes), test_size=0.2, random_state=42)  # overriding the split
    train = np.array(train)

    rng = np.random.RandomState(seed=0)
    idx_train = rng.choice(train, size=int(0.7 * len(train)), replace=False)  # making room for validation set
    idx_val = list(set(range(nnodes)) - set(idx_train).union(set(test)))  # validation set
    splits = {"train": idx_train, "val": idx_val, "test": test}
    train = idx_train
    log("split done")

    log("nbr built")

    degree_weighted_repr = []
    global GLOBAL_NEIGHBORS_DICT, GLOBAL_FEATS, FEAT_LEN, FEAT_MULTIPLIER
    GLOBAL_FEATS = data.x
    t2 = time.perf_counter()
    GLOBAL_NEIGHBORS_DICT = build_neighborhood_dict_sparse()

    t3 = time.perf_counter()
    log(f"dict {t3 - t2:.2f}s")

    t1 = time.perf_counter()

    for node_id in tqdm(train, ncols=120, ascii=True, desc="WL reps"):
        degree_weighted_repr.append(compute_degree_weighted_repr_for_node_orig(node_id))
    degree_weighted_repr = compute_wl_representations(data.x, adj)
    degree_weighted_repr = degree_weighted_repr[train]
    t2 = time.perf_counter()

    log("begin dtree")
    t4 = time.perf_counter()
    # ccp_alphas = [0.00002, 0.00003, 0.00005, 0.00007]
    # for ccp_alpha in ccp_alphas:
    degree_weighted_repr, features_used = transform_features_with_tree(
        data, degree_weighted_repr, train
    )
    t5 = time.perf_counter()
    data.x = data.x[:, features_used]
    GLOBAL_FEATS = data.x
    log(f"end dtree time is {t5 - t4:.2f}s")

    if args.dataset not in ["ogbn-arxiv", "reddit", "flickr", "PubMed"]:
        FEAT_LEN = []
        FEAT_MULTIPLIER = 1
        nfeats = len(features_used)
        for x in range(data.x.shape[0]):
            FEAT_LEN.append(data.x[x].sum().item())
    elif args.dataset in ["PubMed", "flickr"]:
        FEAT_MULTIPLIER = 2 if args.dataset == "flickr" else 3
        FEAT_LEN = []
        nfeats = len(features_used)
        for x in range(data.x.shape[0]):
            nnz = torch.where(data.x[x] == 0, 0, 1).sum().item()
            FEAT_LEN.append(nnz)
    else:
        FEAT_LEN = []
        FEAT_MULTIPLIER = 2
        nfeats = len(features_used)  # data.x[:,features_used].shape[1]
        FEAT_LEN = defaultdict(lambda: nfeats)
        # for x in range(data.x.shape[0]):
        #     _feat_len.append(data.x[x].sum().item())
    log(f"{t2 - t1:.2f}s")

    log("Creating WL now")

    start_time = time.perf_counter()
    snapshot = tracemalloc.take_snapshot()
    snapshot.dump("tracemalloc_snapshot.dump")
    # WL_dist = repr_to_dist(degree_weighted_repr, frac_to_sample=0.05)


    # WL_dist = repr_to_dist(degree_weighted_repr)
    # end_time = time.perf_counter()

    # log("Created WL now")

    # log(f"Time taken: {end_time - start_time:.6f} seconds")

    # sorted_nodes = dist2rknn_sorting(WL_dist, args.k)

    # del WL_dist
    # # Force the garbage collector to free up the memory
    # gc.collect()
    accs = {}
    def hnsw2rknn(degree_weighted_repr, *, k: int) -> dict:
        nnodes, dim = len(degree_weighted_repr), len(degree_weighted_repr[0])
        # Initialize hnswlib index
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=nnodes, ef_construction=150, M=10)
        index.add_items(degree_weighted_repr)
        index.set_ef(35)

        knn = []
        for i in tqdm(range(nnodes), desc="Eval KNN", ascii=True, ncols=100):
            labels, _ = index.knn_query(degree_weighted_repr[i], k=k)
            knn.append(labels[0])
        knn = np.asarray(knn)

        rknn = defaultdict(set)
        for node, neighbors in tqdm(enumerate(knn), desc="Eval rKNN", ascii=True, ncols=100):
            for neighbor in neighbors:
                rknn[neighbor].add(node)

        return {"rknn": dict(rknn)}

    def repr2rknn_sorting(degree_weighted_repr: np.ndarray, k: int) -> t.List[int]:
        rknn_result = hnsw2rknn(degree_weighted_repr, k=k)
        sorted_nodes = select_max_coverage_rknn_celf(rknn_result["rknn"])
        return sorted_nodes

    sorted_nodes = repr2rknn_sorting(degree_weighted_repr, args.k)
    del degree_weighted_repr
    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    snapshot.dump("tracemalloc_snapshot.dump")

    def compute_ogsize_nodes():
        original_nodes = rknn_sorted2budget_select_merged(
            sorted_nodes, train, target_size
        )
        merged_nodes = set()

        for org_id_node in original_nodes:
            train_id_node = train[org_id_node]
            nbr_nodes = GLOBAL_NEIGHBORS_DICT[train_id_node]

            for nbrs in nbr_nodes:
                merged_nodes.add(nbrs)
            merged_nodes.add(train_id_node)

        return len(merged_nodes)

    ogsize = compute_ogsize_nodes()
    snapshot = tracemalloc.take_snapshot()
    snapshot.dump("tracemalloc_snapshot.dump")
    for m in [0.9]:
        log(f"{m = }")
        upscale = 1 + m / (1 - m)
        size_selected_nodes = rknn_sorted2budget_select_merged(
            sorted_nodes, train, target_size * upscale
        )
        log("selected nodes")
        merged_graph = nx.Graph()

        for org_id_node in size_selected_nodes:
            train_id_node = train[org_id_node]
            nbr_nodes = GLOBAL_NEIGHBORS_DICT[train_id_node]
            merged_graph.add_node(train_id_node)
            row_center = adj[[train_id_node]]
            for nbr in nbr_nodes:
                merged_graph.add_node(nbr)
                row_nbr = adj[[nbr]]
                ego = row_center.multiply(row_nbr)
                nbrnbrs = ego.tocoo().col
                for nbrnbr in nbrnbrs:
                    edge = (nbr, nbrnbr)
                    edge = sorted(edge)
                    merged_graph.add_edge(*edge)
                edge = (train_id_node, nbr)
                edge = sorted(edge)
                merged_graph.add_edge(*edge)
        # size_sel_nodes_train_ordered = [train[n] for n in size_selected_nodes]
        print("created nodes")

        for node in merged_graph.nodes:
            merged_graph.nodes[node]["target"] = (
                node in train
            )  # size_sel_nodes_train_ordered
            merged_graph.nodes[node]["x"] = data.x[node].numpy().tolist()
            merged_graph.nodes[node]["y"] = data.y[node]
        print("created edges")

        personalization = {node: 0 for node in merged_graph.nodes()}
        l = 1.0 / len(size_selected_nodes)
        log("before personlisation")
        for node in size_selected_nodes:
            personalization[train[node]] = l
        ppr = sorted(
            nx.pagerank(merged_graph, personalization=personalization).items(),
            key=lambda x: x[1],
        )
        todel = len(ppr) - ogsize
        assert todel >= 0
        nodes_todel = []
        log(f"{todel=}")
        for node, _ in ppr:
            nodes_todel.append(node)
            if len(nodes_todel) >= todel:
                break
        for node in nodes_todel:
            merged_graph.remove_node(node)
        final_graph = match_distribution(merged_graph, data, train, sorted_nodes)
        merged_graph = final_graph
        for node in merged_graph.nodes:
            merged_graph.nodes[node]["target"] = (
                node in train
            )  # size_sel_nodes_train_ordered
            merged_graph.nodes[node]["x"] = data.x[node].numpy().tolist()
            merged_graph.nodes[node]["y"] = data.y[node].item()
        # with open("merged_graph.pkl", "wb") as f:
        #     pickle.dump(merged_graph, f)




        # Directory to save the graph chunks
        # save_dir = "/DATATWO/saved_condensed_mag_new"
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # Set your chunk size (number of nodes per chunk); adjust as needed for your memory constraints.
        # chunk_size = 100000
        #
        # # Get the complete list of nodes from merged_graph
        # all_nodes = list(merged_graph.nodes())
        # num_nodes = len(all_nodes)
        #
        # # Define a helper function to convert node attributes (if they are Torch tensors)
        # def convert_node_attrs(attrs):
        #     new_attrs = {}
        #     for key, value in attrs.items():
        #         if isinstance(value, torch.Tensor):
        #             if value.dim() == 0:
        #                 new_attrs[key] = value.item()
        #             else:
        #                 new_attrs[key] = value.detach().cpu().tolist()
        #         else:
        #             new_attrs[key] = value
        #     return new_attrs
        #
        # # Process the merged_graph in chunks
        # for i in range(0, num_nodes, chunk_size):
        #     chunk_nodes = all_nodes[i: i + chunk_size]
        #
        #     # Create a subgraph for the current chunk and make a copy to allow attribute updates
        #     subgraph = merged_graph.subgraph(chunk_nodes).copy()
        #
        #     # Update node attributes in the subgraph using a safe iteration method
        #     for node, attrs in list(subgraph.nodes(data=True)):
        #         new_attrs = convert_node_attrs(attrs)
        #         # Clear the existing attributes and update with new_attrs
        #         subgraph.nodes[node].clear()
        #         subgraph.nodes[node].update(new_attrs)
        #
        #     # Define the filename for this chunk
        #     chunk_filename = os.path.join(save_dir, f"subgraph_{i // chunk_size}.pkl")
        #     with open(chunk_filename, "wb") as f:
        #         pickle.dump(subgraph, f, protocol=pickle.HIGHEST_PROTOCOL)
        #
        #     # Clean up to free memory before processing the next chunk
        #     del subgraph
        #     gc.collect()
        #
        # print("Graph saved in chunks successfully.")

        merged_data = from_networkx(merged_graph, group_node_attrs=["x"])
        del merged_graph
        gc.collect()
        if args.save:
            save_root = Path("saved_ours")
            save_dir = save_root / f"{args.dataset}-{args.target_size_frac}"
            save_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            save_file = save_dir / f"data_m_{m}.pt"
            torch.save(
                {"data": merged_data, "features_used": features_used}, save_file
            )  # this may have compatibility issues
        model_type = "GCN" if args.dataset not in SAINT_DATASETS else "GCN_inductive"
        # model_type = "GCN"
        # nclasses = 152
       # nfeats = 768
        hidim = 128 if args.dataset not in SAINT_DATASETS else 1024
        model = globals()[model_type](nfeats, nclasses, hidim=hidim)
        writer = SummaryWriter(
            f"tensorboard_logs/{args.dataset}_{model_type}_{args.target_size_frac}"
        )
        #loaded_data = torch.load(args.saved_data_path)
        #merged_data = loaded_data["data"]
        if args.dataset in SAINT_DATASETS:
            # merged_data.x = torch.tensor(scaler(merged_data.x.cpu().numpy()))
            scaler_new = StandardScaler()
            scaler_new.fit(merged_data.x.cpu().numpy())
            merged_data.x = torch.tensor(scaler_new.transform(merged_data.x.cpu().numpy()))

            import scipy.sparse as sp
            from torch_sparse import SparseTensor

            d = np.ones(merged_data.edge_index.shape[1])
            r = merged_data.edge_index[0].cpu().numpy()
            c = merged_data.edge_index[1].cpu().numpy()
            n = merged_data.x.shape[0]
            adj = sp.csr_matrix((d, (r, c)), shape=(n, n))
            adj = adj.tolil()
            adj = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj.sum(1))
            r_inv = np.power(rowsum, -1 / 2).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = sp.diags(r_inv)
            adj = r_mat_inv.dot(adj)
            adj = adj.dot(r_mat_inv)
            adj = adj.tocoo().astype(np.float32)
            sparserow = torch.LongTensor(adj.row).unsqueeze(1)
            sparsecol = torch.LongTensor(adj.col).unsqueeze(1)
            sparseconcat = torch.cat((sparserow, sparsecol), 1)
            sparsedata = torch.FloatTensor(adj.data)
            adj = torch.sparse.FloatTensor(
                sparseconcat.t(), sparsedata, torch.Size(adj.shape)
            )
            adj = SparseTensor(
                row=adj._indices()[0],
                col=adj._indices()[1],
                value=adj._values(),
                sparse_sizes=adj.size(),
            )
            merged_data.adj = adj
        data.x = (
            data.x_normed if hasattr(data, "x_normed") else data.x
        )  # only in case of saint_datasets
        kwargs = {
            "model": model,
            "model_type": model_type,
            "data_syn": merged_data,
            "nepochs": args.nepochs,
            "data": data,
            "splits": splits,
        }
        acc = 0
        var = 0
        nruns = 5
        snapshot = tracemalloc.take_snapshot()
        snapshot.dump("tracemalloc_snapshot.dump")
        for run_num in range(1, nruns + 1):
            from timing import Timer

            writer = SummaryWriter(
                f"logs/bonsai_{args.dataset}_{model_type}_{args.target_size_frac}_{run_num}"
            )
            with Timer(f"Training Bonsai for {args.nepochs} epochs") as timer:
                run_acc = train_model(**kwargs, writer=writer)
            writer.add_scalar("experiment/time", timer.dur)
            delta = run_acc - acc
            acc += (run_acc / run_num) - (acc / run_num)
            delta2 = run_acc - acc
            var += delta * delta2
        std = np.sqrt(var / nruns)
        accs[m] = rf"{acc*100:.2f}\pm {std*100:.2f}"
    for _, v in accs.items():
        print(v)
    end_snapshot = tracemalloc.take_snapshot()
    top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')

    print("[ Memory Usage ]")
    for stat in top_stats[:10]:
        print(stat)


if __name__ == "__main__":
    main()


