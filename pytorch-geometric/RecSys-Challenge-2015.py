# A Read-World Example - RecSys Challenge 2015
# links: https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
# ==================== Data Description
# yoochoose-clicks: at one session, there may be several click events
# yoochoose-buys: after several click events, the clients may buy several items
# yoochoose-test
# ==================== Mission
# 1). predict whether the client will buy item after clicking
# 2). predict which item will be bought
# ==================== Solution
# we can use the same session as a graph (mission 1, graph-level)
# at the session graph, different items represent node (mission 2, nodel-level)
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
# LabelEncoder Methods
# fit(y): Fit label encoder.
# fit_transform(y): Fit label encoder and return encoded labels.
# get_params([deep]): Get parameters for this estimator.
# inverse_transform(y): Transform labels back to original encoding.
# set_params(**params): Set the parameters of this estimator.
# transform(y): Transform labels to normalized encoding.
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import roc_auc_score

np.random.seed(42)

# =================== Preprocessing
df = pd.read_csv('./yoochoose/yoochoose-clicks.dat/yoochoose-clicks.dat', header=None)
df.columns = ['session_id', 'timestamp', 'item_id', 'category']
buy_df = pd.read_csv('./yoochoose/yoochoose-buys.dat/yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

# drop sessions where the number of click events is smaller than 2
# Series.map can use mapping dictionary or function to deal with each row of data
df['valid_session'] = df.session_id.map(df.groupby('session_id')['item_id'].size() > 2)
df = df.loc[df.valid_session].drop('valid_session', axis=1)

# randomly sample a couple of them
sampled_session_id = np.random.choice(df.session_id.unique(), 100, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
# df.nunique() get nunique information for each column
print('average length of session: ' + str(df.groupby('session_id')['item_id'].size().mean()))

item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)

# if the session_id appears on the yoochoose-buys and yoochoose-clicks,
# then at the session_id, client will buy an item or items
df['label'] = df.session_id.isin(buy_df.session_id)


class YooChooseBinaryDataset(InMemoryDataset):
    '''
    if the processed_paths aren't empty
    it will skip the process step
    '''
    def __int__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__int__(root, transform, pre_transform)
        # self.processed_paths is the return path by processed_file_names() function
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        '''
        return path:-> ./raw/......
        '''
        return []

    @property
    def processed_file_names(self):
        '''
        return path:-> ./processed/......
        '''
        return ['./yoochoose/yoochoose-clicks-processed/yoochoose-click-binary-1M-sess.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        # group click events by session_id to a graph
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            # for each session, their item id should be classed start from 0
            # different session will use different parameters
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            # reset_index: drop = True, it will drop old index;
            # drop = False (default), it will not drop old index, however it will change the old index to a new column
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            # [group.session_id == session_id, ['sess_item_id', 'item_id']]
            # it has two conditions, one is the former, another one is the last
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values
            node_features = torch.LongTensor(node_features).unsqueeze(1)
            # using encoder code to link edges
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        # self.processed_paths is returned by processed_file_names()
        torch.save((data, slices), self.processed_paths[0])


dataset = YooChooseBinaryDataset(root='./')

dataset = dataset.shuffle()
train_dataset = dataset[:80]
val_dataset = dataset[80:90]
test_dataset = dataset[90:]

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # N represents number of node
        # in_channels represents node feature dimension
        # E represents number of edge
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # why it needs to add self loop edges
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # size = (N, N)
        return self.propagate(edge_index=edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x has shape [N, in_channels]
        # E represents the number of edge
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        # new_embedding:-> [N, in_channels + out_channels]
        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding


class Net(torch.nn.Module):
    def __init__(self, embed_dim):
        super(Net, self).__init__()
        self.embed_dim = embed_dim

        self.conv1 = SAGEConv(self.embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() + 1, embedding_dim=self.embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x


device = torch.device('cuda')
model = Net(embed_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.BCELoss()


def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return roc_auc_score(labels, predictions)


for epoch in range(1):
    loss = train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))