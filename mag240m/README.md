# MAG240M

This is the PyTorch implementation for Bonsai: Gradient-free Graph Distillation for Node Classification for the [MAG240M](https://ogb.stanford.edu/docs/lsc/mag240m/) dataset.

# Download Datasets

Since MAG240M is a large dataset, we downloaded it in a hard drive and had its location specified in `run.sh`. You can also download the dataset, and then tell the code where the directory is by editing `run.sh`. For reference, our `run.sh` points to `/DATATWO/datasets/mag240m` and this directory looks like this:

```
mag240m/
├── mag240m_kddcup2021/
│   ├── mapping/
│   ├── meta.pt
│   ├── processed/
│   │   ├── author___affiliated_with___institution/
│   │   │   └── [680M] edge_index.npy
│   │   ├── author___writes___paper/
│   │   │   └── [5.8G] edge_index.npy
│   │   ├── paper/
│   │   │   ├── [174G] node_feat.npy
│   │   │   ├── [929M] node_label.npy
│   │   │   └── [929M] node_year.npy
│   │   └── paper___cites___paper/
│   │       └── [19G] edge_index.npy
│   ├── raw/
│   ├── RELEASE_v1.txt
│   └── [13G] split_dict.pt
└── [167G] mag240m_kddcup2021.zip

8 directories, 10 files
```

# Instructions

1. Please install dependencies from requirements.txt in `python==3.9.19`
2. You can directly run `bash run.sh 0` after specifying the location of `mag240m`. Here, `0` is the index of the GPU to run on it. If no index is passed, the code runs on CPU. 
3. The outputs will be saved to `saved_ours` directory.

## Notes about dataset size

Although MAG240M contains 240M nodes, only 1.4M of them have node labels. We have only condensed the train set created from these 1.4M nodes. The code can be used to generate a condensed set from the full 240M nodes as well, however, we have not performed that experiment yet.
