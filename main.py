import gc
import copy
import time
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

import networkx as nx
import numpy as np
from scipy.sparse import coo_array

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.utils import from_networkx

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from tensorboardX import SummaryWriter

from utils import (
    wl2rknn,
    select_max_coverage_rknn_celf,
)
from utils import transform_features_with_tree
from model import GCN, GCN_inductive


_global_neighbors_dict = {}
global_feats = None
_feat_len = None
saint_datasets = ["flickr", "ogbn-arxiv", "reddit"]


def log(x):
    pass


def load_dataset(dataset_name, root):
    global saint_datasets
    if dataset_name in saint_datasets:
        from sklearn.preprocessing import StandardScaler
        from load_saint_dataset import load_saint_dataset
        data = load_saint_dataset(dataset_name, root=root)
        feat_full = data.x.cpu().numpy()
        scaler = StandardScaler()
        scaler.fit(feat_full)
        feat_full = torch.tensor(scaler.transform(feat_full))
        data.x_normed = feat_full
        return {"data": data, "scaler": lambda feat: scaler.transform(feat)}
    else:
        from torch_geometric.datasets import Planetoid
        data = Planetoid(root=root, name=dataset_name)._data
        def nop_scaler(x):
            return x
        return {"data": data, "scaler": nop_scaler}


def dist2rknn_sorting(WL_dist, k):
    rknn_result = wl2rknn(WL_dist, k=k)
    sorted_nodes = select_max_coverage_rknn_celf(rknn_result["rknn"])
    return sorted_nodes


def rknn_sorted2budget_select_merged(sorted_nodes, train, target_size):
    r"""Only works when features are binary as of now. PubMed and
    ogbn-arxiv might not work. I have made some adaptations. But
    let's see whether they really work.

    For 'reddit' and 'ogbn-arxiv': Features are dense and float.
    For 'flickr': Features are sparse integers, but not binary.
    For 'PubMed': Features are sparse floats.
    """
    global _global_neighbors_dict, _feat_len, _feat_multiplier
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
        node_adj_set = _global_neighbors_dict[train[node]]

        for nbr_node in node_adj_set:
            if nbr_node not in selected_nodes:
                candidate_selected_nodes.add(nbr_node)
            edge = tuple(sorted([nbr_node, train[node]]))
            if edge not in selected_edges:
                candidate_selected_edges.add(edge)
            nbr_adj_set = _global_neighbors_dict[nbr_node]
            ego_nodes = node_adj_set.intersection(nbr_adj_set)
            for nbr_nbr_node in ego_nodes:
                edge = tuple(sorted([nbr_node, nbr_nbr_node]))
                if edge not in selected_edges:
                    candidate_selected_edges.add(edge)
        num_edges = len(selected_edges) + len(candidate_selected_edges)
        added_nodes = candidate_selected_nodes
        ints_due_to_nodes = 0
        for added_node in added_nodes:
            feat_ints = _feat_len[added_node]
            ints_due_to_nodes += feat_ints
        ints_due_to_nodes *= _feat_multiplier
        ints_due_to_nodes += ints_till_now_due_to_nodes
        ints_due_to_edges = num_edges * 2
        total_ints = ints_due_to_nodes + ints_due_to_edges
        size = total_ints * 2
        if size < target_size:
            size_selected_nodes.append(node)
            selected_nodes.update(added_nodes)
            selected_edges.update(candidate_selected_edges)
            ints_till_now_due_to_nodes = ints_due_to_nodes
        else:
            not_selected_for_n_consecutive_iters += 1
            if not_selected_for_n_consecutive_iters > 100:
                break
    return size_selected_nodes


def train_backend_inductive(model, nepochs, data, data_syn, splits, writer):
    opt = optim.Adam(model.parameters())
    loss_fn = lambda out, y: F.nll_loss(out, y)
    loop = tqdm(range(nepochs), ascii=True, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    d = data
    d = d.to(device, "x", "adj", "y")
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
        writer.add_scalar('loss/train', loss.item(), epoch)
        model.eval()
        with torch.no_grad():
            out = model(d.x, d.adj)
            loss = loss_fn(out[val], d.y[val])
            writer.add_scalar('loss/val', loss.item(), epoch)
            preds = out[val].max(1)[1].cpu().numpy()
            acc = accuracy_score(d.y[val].cpu().numpy(), preds)
            writer.add_scalar('acc/val', acc, epoch)
            if acc > best_acc_val:
                best_acc_val = acc
                weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    with torch.no_grad():
        out = model(d.x, d.adj)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    writer.add_scalar('test_acc/test', acc)
    return acc


def train_backend_pyg(model, nepochs, data, data_syn, splits, writer):
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loop = tqdm(range(nepochs), ascii=True, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    d = data
    d = d.to(device, "x", "edge_index", "y")
    d_syn = data_syn
    d_syn = d_syn.to(device, "x", "edge_index", "y", "target")
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
            writer.add_scalar('loss/train', loss.item(), epoch)
            model.eval()
            with torch.no_grad():
                out = model(d.x, d.edge_index)
                loss = loss_fn(out[val], d.y[val])
                writer.add_scalar('loss/val', loss.item(), epoch)
                preds = out[val].max(1)[1].cpu().numpy()
                acc = accuracy_score(d.y[val].cpu().numpy(), preds)
                writer.add_scalar('acc/val', acc, epoch)
                if acc > best_acc_val:
                    best_acc_val = acc
                    weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    with torch.no_grad():
        out = model(d.x, d.edge_index)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    writer.add_scalar('test_acc/test', acc)
    return acc


def train_model(model_type, model, nepochs, data, data_syn, splits, writer):
    if model_type == "GCN_inductive":
        return train_backend_inductive(model, nepochs, data, data_syn, splits, writer)
    else:
        return train_backend_pyg(model, nepochs, data, data_syn, splits, writer)


def size(nnodes, nedges, feats, dtype="int"):
    mx = 1 if dtype == "int" else 2 if dtype == "float" else None
    return (nnodes * feats * mx + nedges * 2) * 2


def build_neighborhood_dict_sparse():
    neighbors = defaultdict()
    global _adj
    for node in tqdm(
        range(_adj.shape[0]),
        ascii=True,
        total=_adj.shape[0],
        ncols=120,
        desc="build neighbor",
    ):
        node_nbrs = set(_adj[[node]].tocoo().col)
        node_nbrs.add(node)
        neighbors[node] = node_nbrs
    return neighbors


def compute_degree_weighted_repr_for_node_orig(node_id):
    global _global_neighbors_dict
    global global_feats

    node_neighbors = _global_neighbors_dict[node_id]
    total_degree = 0
    weighted_sum = torch.zeros_like(global_feats[node_id])

    for nbr in node_neighbors:
        nbr_neighbors = _global_neighbors_dict[nbr]
        local_degree = (
            len(node_neighbors.intersection(nbr_neighbors)) - 1
        )  # -1 done because both nodes are included nbr and central node, but degree added is just 1
        total_degree += local_degree
        torch.add(
            weighted_sum, torch.mul(global_feats[nbr], local_degree), out=weighted_sum
        )  # Sum up the weighted features

    if total_degree > 0:
        vmed_n = weighted_sum / total_degree  # Element-wise division
    else:
        vmed_n = torch.zeros_like(
            global_feats[node_id]
        )  # Handle cases where there is no local degree [should not occur ideally]

    return vmed_n


def repr_to_dist(degree_weighted_repr):
    DWD_dist_full_matrix = pairwise_distances(np.array(degree_weighted_repr), n_jobs=20)
    return DWD_dist_full_matrix


def match_distribution(merged_graph, data, train, rknn_ranked_nodes):
    # num_nodes = merged_graph.number_of_nodes()
    num_nodes = sum(1 for _, v in merged_graph.nodes(data=True) if v["target"]) # only sum trainable nodes
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
        merged_class_nodes = [node for node, node_attrs in merged_graph.nodes(data=True) if data.y[node].item() == class_label and node_attrs["target"]]

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
                    cnt +=1
                if cnt >= lower_bound-current_count:
                    break

        elif current_count > upper_bound:  # Need to remove nodes
            merged_class_nodes_sorted = sorted(
                merged_class_nodes, key=lambda node: rknn_ranked_nodes.index(node)
            )
            # Remove nodes starting from the rightmost end
            nodes_to_remove = merged_class_nodes_sorted[- (current_count - int(upper_bound)):]
            for node in nodes_to_remove:
                if node in merged_graph:
                    merged_graph.remove_node(node)
        
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_size_frac", required=True, type=float)
    parser.add_argument("--nepochs", default=100, type=int)
    parser.add_argument("--k", default=5, type=int, help="The parameter `k` in kNN.")
    parser.add_argument("--save", default=False, action='store_true', help='Bool flag: save condensed dataset or not')
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
        ],
    )
    args = parser.parse_args()
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
    global _adj
    _adj = adj
    dtype = (
        "int"
        if args.dataset in ["cora", "citeseer", "flickr"]
        else "float" if args.dataset in ["ogbn-arxiv", "reddit", "pubmed"] else None
    )
    size_full = size(nnodes, nedges, nfeats, dtype)
    target_size = float(f"{args.target_size_frac * size_full:.2f}")
    nclasses = len(set(data.y.reshape(-1).tolist()))

    train, test = train_test_split(range(nnodes), test_size=0.2, random_state=42)
    train = np.array(train)
    
    rng = np.random.RandomState(seed=0)
    idx_train = rng.choice(train, size=int(0.7 * len(train)), replace=False)
    idx_val = list(set(range(nnodes))-set(idx_train).union(set(test)))
    splits = {"train":idx_train, "val":idx_val, "test": test}
    train = idx_train
    log("split done")

    
    log("nbr built")

    degree_weighted_repr = []
    global _global_neighbors_dict, global_feats, _feat_len, _feat_multiplier
    global_feats = data.x
    t2 = time.perf_counter()
    _global_neighbors_dict = build_neighborhood_dict_sparse()

    t3 = time.perf_counter()
    log(f"dict {t3 - t2:.2f}s")
    
    t1 = time.perf_counter()

    for idx, node_id in tqdm(
        enumerate(train), total=len(train), ncols=120, ascii=True, desc="vmed reps"
    ):
        degree_weighted_repr.append(compute_degree_weighted_repr_for_node_orig(node_id))
    t2 = time.perf_counter()

    log("begin dtree")
    t4 = time.perf_counter()
    # ccp_alphas = [0.00002, 0.00003, 0.00005, 0.00007]
    # for ccp_alpha in ccp_alphas:
    degree_weighted_repr, features_used = transform_features_with_tree(
        data, degree_weighted_repr, data.x.shape[0], train
    )
    t5 = time.perf_counter()
    data.x = data.x[:, features_used]
    global_feats = data.x
    log(f"end dtree time is {t5 - t4:.2f}s")

    
    if args.dataset not in ["ogbn-arxiv", "reddit", "flickr", "PubMed"]:
        _feat_len = []
        _feat_multiplier = 1
        nfeats = len(features_used)
        for x in range(data.x.shape[0]):
            _feat_len.append(data.x[x].sum().item())
    elif args.dataset in ["PubMed", "flickr"]:
        _feat_multiplier = 2 if args.dataset == "flickr" else 3
        _feat_len = []
        nfeats = len(features_used)
        for x in range(data.x.shape[0]):
            nnz = torch.where(data.x[x] == 0, 0, 1).sum().item()
            _feat_len.append(nnz)
    else:
        _feat_len = []
        _feat_multiplier = 2
        nfeats = len(features_used)#data.x[:,features_used].shape[1]
        _feat_len = defaultdict(lambda: nfeats)
        # for x in range(data.x.shape[0]):
        #     _feat_len.append(data.x[x].sum().item())
    log(f"{t2 - t1:.2f}s")

    log("Creating WL now")

    start_time = time.perf_counter()
    WL_dist = repr_to_dist(degree_weighted_repr)
    end_time = time.perf_counter()

    log("Created WL now")

    log(f"Time taken: {end_time - start_time:.6f} seconds")


    sorted_nodes = dist2rknn_sorting(WL_dist, args.k)
    accs = {}
    del WL_dist
    # Force the garbage collector to free up the memory
    gc.collect()


    def compute_ogsize_nodes():

        global _global_neighbors_dict  # builds from

        original_nodes = rknn_sorted2budget_select_merged(
            sorted_nodes, train, target_size
        )
        l = len(original_nodes)
        merged_nodes = set()

        for org_id_node in original_nodes:
            train_id_node = train[org_id_node]
            nbr_nodes = _global_neighbors_dict[train_id_node]

            for nbrs in nbr_nodes:
                merged_nodes.add(nbrs)
            merged_nodes.add(train_id_node)

        return len(merged_nodes)

    ogsize = compute_ogsize_nodes()

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
            nbr_nodes = _global_neighbors_dict[train_id_node]
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

        for node in merged_graph.nodes:
            merged_graph.nodes[node]["target"] = node in train#size_sel_nodes_train_ordered
            merged_graph.nodes[node]["x"] = data.x[node].numpy().tolist()
            merged_graph.nodes[node]["y"] = data.y[node]

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
            merged_graph.nodes[node]["target"] = node in train#size_sel_nodes_train_ordered
            merged_graph.nodes[node]["x"] = data.x[node].numpy().tolist()
            merged_graph.nodes[node]["y"] = data.y[node]
        merged_data = from_networkx(merged_graph, group_node_attrs=["x"])
        if args.save:
            save_root = Path("saved_ours")
            save_dir = save_root / f"{args.dataset}-{args.target_size_frac}"
            save_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            save_file = save_dir / f"data_m_{m}.pt"
            torch.save({"data":merged_data, "features_used":features_used}, save_file) # this may have compatibility issues
        model_type = "GCN" if args.dataset not in saint_datasets else "GCN_inductive"
        hidim = 128 if args.dataset not in saint_datasets else 1024
        model = globals()[model_type](nfeats, nclasses, hidim=hidim)
        writer = SummaryWriter(f'tensorboard_logs/{args.dataset}_{model_type}_{args.target_size_frac}')
        if args.dataset in saint_datasets:
            merged_data.x = torch.tensor(scaler(merged_data.x.cpu().numpy()))
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
            adj = torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(adj.shape))
            adj = SparseTensor(
                row=adj._indices()[0],
                col=adj._indices()[1],
                value=adj._values(),
                sparse_sizes=adj.size(),
            )
            merged_data.adj = adj
        data.x = data.x_normed if hasattr(data, "x_normed") else data.x # only in case of saint_datasets
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
        for run_num in range(1, nruns + 1):
            from timing import Timer
            writer = SummaryWriter(f'logs/bonsai_{args.dataset}_{model_type}_{args.target_size_frac}_{run_num}')
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


if __name__ == "__main__":
    main()
