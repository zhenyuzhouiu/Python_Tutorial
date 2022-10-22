# Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric
# link: https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

# Data
# 1). The attributes/features associated with each node
x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

# 2). The graph connectivity (edge index) should be confined with the COO format,
# i.e. the first list contains the index of the source nodes,
# while the index of target nodes is specified in the second list.
edge_index = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)

# Dataset
# PyG tutorial: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
# PyG provides two different types of dataset classes, InMemoryDataset and Dataset.
# As they indicate literally, the former one is for data that fit in your RAM,
# while the second one is for much larger data.

import torch
from torch_geometric.data import InMemoryDataset, download_url


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # A list of files in the raw_dir which needs to be found in order to skip the download.
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        # A list of files in the processed_dir which needs to be found in order to skip the processing.
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        ...

    def process(self):
        # Processes raw data and saves it into the processed_dir
        # Read data into huge `Data` list:->  torch_geometric.data import Data
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # collate the list into one huge Data object via torch_geometric.data.InMemoryDataset.collate()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# DataLoader

# MessagePassing:-> propagate, message, update
# Implement a SageConv layer

import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels].
        # In my understanding, the N represents n nodes, and in_channels represents the dimension of node feature?
        # edge_index has shape [2, E]
        # '2' means the i and j, while the E represents E edges?

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        # x_j is the neighbour of the x_j, E represent the E edges?


        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding
