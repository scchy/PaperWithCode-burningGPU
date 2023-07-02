# Spatio-Temporal Graph Convolutional Networks

- Paper Link: [arXiv](https://arxiv.org/pdf/1709.04875v4.pdf)
- Author's code: [https://github.com/VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18)
- reference blogs: [Build your first Graph Neural Network model to predict traffic speed in 20 minutes](https://towardsdatascience.com/build-your-first-graph-neural-network-model-to-predict-traffic-speed-in-20-minutes-b593f8f838e5)
- data 
    - [metr-la.h5](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)
    - [sensor_graph](https://github.com/chnsh/DCRNN_PyTorch/tree/pytorch_scratch/data/sensor_graph)

- [dgl docs](https://docs.dgl.ai/install/index.html)

```text
file tree
.
├── data
│   ├── metr-la.h5
│   └── sensor_graph
│       ├── distances_la_2012.csv
│       └── graph_sensor_ids.txt
├── load_data.py
├── sensors2graph.py
└── utils.py
```

## Summary

use historical speed data to predict the speed at a future time step. 
