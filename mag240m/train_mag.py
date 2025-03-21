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

import scipy.sparse as sp
from torch_sparse import SparseTensor





ADJ = None
FEAT_MULTIPLIER = 1
GLOBAL_NEIGHBORS_DICT = {}
GLOBAL_FEATS = None
FEAT_LEN = None
SAINT_DATASETS = ["flickr", "ogbn-arxiv", "reddit", "mag240m"]


def log(x: str) -> None:
    r"""Function that used to log its string contents to file. Was used in debugging and now, a dummy
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

    dataset = MAG240MDataset(root=root)

    num_papers = dataset.num_papers
    

    rng = np.random.default_rng(seed=42) 
    train_idx = dataset.get_idx_split('train')  

    if subset_frac <= 0 or subset_frac > 1:
        raise ValueError("subset_frac should be between (0,1].")
    subset_size = int(len(train_idx) * subset_frac)
    if subset_size < 1:
        subset_size = 1

    subset_indices = rng.choice(train_idx, size=subset_size, replace=False)

    x_np = dataset.all_paper_feat[subset_indices]
    y_np = dataset.all_paper_label[subset_indices]
    nnodes = x_np.shape[0]

   
    edge_index_np = dataset.edge_index('paper', 'cites', 'paper')

    subset_set = set(subset_indices)
    mask = np.isin(edge_index_np, subset_indices)
    valid_edges = mask[0] & mask[1]
    edge_index_subset = edge_index_np[:, valid_edges]
    
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(subset_indices)}

    src_old = edge_index_subset[0, :]
    dst_old = edge_index_subset[1, :]

    src_new = np.array([old_to_new[old] for old in src_old], dtype=np.int64)
    dst_new = np.array([old_to_new[old] for old in dst_old], dtype=np.int64)

    edge_index_subset_reindexed = np.stack([src_new, dst_new], axis=0)
    
    x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
    y = torch.from_numpy(np.ascontiguousarray(y_np)).long()
    edge_index = torch.from_numpy(np.ascontiguousarray(edge_index_subset_reindexed)).long()
    # < creating tkipf style adj >
    
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

    pyg_data = Data(x=x, edge_index=edge_index, y=y, adj=adj)


    scaler = StandardScaler()
    scaler.fit(x_np)  # Fit on the subset
    x_scaled = scaler.transform(x_np)
    pyg_data.x = torch.from_numpy(x_scaled).float()

    train_set = set(train_idx)
    pyg_data.target = torch.tensor([old_id in train_set for old_id in subset_indices], dtype=torch.bool)

    print("MAG240M subset loaded and features scaled.")
    return {"data": pyg_data, "scaler": scaler}



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
    d = d.to(device, "x", "edge_index", "y", "adj")
    d_syn = data_syn
    d_syn = d_syn.to(device, "x", "adj", "y", "target")
    test = splits["test"]
    val = splits["val"]
    for epoch in loop:
        model.train()
        out = model(d_syn.x, d_syn.adj)
        loss = loss_fn(out[d_syn.target], d_syn.y[d_syn.target])
        opt.zero_grad()
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
    num_nodes = sum(
        1 for _, v in merged_graph.nodes(data=True) if v["target"]
    )  # only sum trainable nodes

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
    parser.add_argument("--saved_data_path", default="saved_ours/mag240m-0.005/data_m_0.9.pt")
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
        default=1,  #Now we load the full dataset
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
    nclasses = data.y.max().item() + 1  #we assume continuous labels of nodes from 0 ... max_val

    print(f"the dataset is {size_full} and we take it to {target_size}")


    train, test = train_test_split(range(nnodes), test_size=0.2, random_state=42)  
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

   
    t2 = time.perf_counter()

    features_used = [0]*768

    log("begin dtree")
    t4 = time.perf_counter()
  
    t5 = time.perf_counter()
    
    GLOBAL_FEATS = data.x
    log(f"end dtree time is {t5 - t4:.2f}s")

    if args.dataset not in ["ogbn-arxiv", "reddit", "flickr", "PubMed"]:
        FEAT_LEN = []
        FEAT_MULTIPLIER = 1
        nfeats = 768
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
    
    accs = {}
          

    for m in [0.9]:
        model_type = "GCN" if args.dataset not in SAINT_DATASETS else "GCN_inductive"
        
        nfeats = 768
        hidim = 128 if args.dataset not in SAINT_DATASETS else 1024
        model = globals()[model_type](nfeats, nclasses, hidim=hidim)
        writer = SummaryWriter(
            f"tensorboard_logs/{args.dataset}_{model_type}_{args.target_size_frac}"
        )
        
        merged_data = data


        ############# add for random graph ################
        sample_frac = 0.03  # 0.5% of nodes
        nnodes = data.x.shape[0]
        subset_size = max(1, int(sample_frac * nnodes))

        # Randomly sample subset indices (using numpy for reproducibility, then convert to tensor)
        subset_indices_np = np.random.choice(nnodes, size=subset_size, replace=False)
        subset_indices = torch.from_numpy(subset_indices_np).long()

        # Create a boolean mask for nodes in the subset
        mask = torch.zeros(nnodes, dtype=torch.bool)
        mask[subset_indices] = True

        # Filter edges: only keep those where both endpoints are in the subset
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        filtered_edge_index = data.edge_index[:, edge_mask]

        # Reindex the nodes in filtered_edge_index so that they run from 0 to subset_size-1
        mapping = torch.full((nnodes,), -1, dtype=torch.long)
        mapping[subset_indices] = torch.arange(subset_size)
        reindexed_edge_index = mapping[filtered_edge_index]

        # Build the new Data object for merged_data with the sampled nodes
        merged_x = data.x[subset_indices]
        merged_y = data.y[subset_indices]

        # Generate the "target" attribute: mark True if the original node index is in the training split
        train_set = set(splits["train"])
        merged_target = torch.tensor(
            [i.item() in train_set for i in subset_indices], dtype=torch.bool
        )

        merged_data = Data(
            x=merged_x,
            y=merged_y,
            edge_index=reindexed_edge_index,
            target=merged_target
        )


        ################# end ###########################

        #breakpoint()
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


