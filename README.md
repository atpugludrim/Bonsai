# Bonsai: Gradient-free Graph Distillation for Node Classification

This is the PyTorch implementation for Bonsai: Gradient-free Graph Distillation for Node Classification

# Download Datasets

For Cora, Citeseer, and Pubmed, the code will directly download them from PyTorch Geometric. For Flickr, Ogbn-arxiv, and Reddit, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). They are available on [this Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) provided by the GraphSAINT team. Download the files and unzip them to `datasets` at the root directory.

# Instructions

1. Please install dependencies from requirements.txt in `python==3.9.19`
2. You can directly run `bash run.sh 0`. Here, `0` is the index of the GPU to run on it. If no index is passed, the code runs on CPU. 
3. The outputs will be saved to `saved_ours` directory. It also contains `train_bonsai.py` script along with `train_all.sh` to train a `GCN`, `GAT`, and `GIN` on the saved outputs to get the results.
