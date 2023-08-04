import torch
import dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import SAGEConv
from torch.utils.data import DataLoader

def _get_activation_fn(activation):
    """
        Return an activation function given a string
    :param activation:
    :return:
    """

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator):
        super().__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.mlp = MLP(out_feats, n_hidden, n_classes)
        self.layer = nn.ModuleList()
        self.layer.append(SAGEConv(in_feats, n_hidden, aggregator))
        self.act_fun = nn.Sigmoid()

        for i in range(1, n_layers - 1):
            self.layer.append(SAGEConv(n_hidden, n_hidden, aggregator))
        self.layer.append(SAGEConv(n_hidden, out_feats, aggregator))
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, g, blocks, feas):
        h = feas
        for i, (layer, block) in enumerate(zip(self.layer, blocks)):
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)

        out = self.mlp(g, h)
        out = self.act_fun(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_feats * 2, hid_feats)
        self.fc2 = nn.Linear(hid_feats, hid_feats)
        self.fc3 = nn.Linear(hid_feats, out_classes)


    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']

        input_h = torch.cat([h_u, h_v], 1)
        output = self.fc1(input_h)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        score = self.fc3(output)

        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        """
        邻居采样
        :param g: dgl Graph
        :param fanouts: 采样节点数量,
        """
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = torch.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return blocks


if __name__ == '__main__':
    in_feats = 256
    n_hidden = 512
    out_feats = 256
    n_classes = 1
    n_layers = 3
    activation = F.relu
    dropout = 0.1
    aggregator = 'mean'
    fan_out = '5, 10'
    model = GraphSAGE(in_feats, n_hidden, out_feats, n_classes, n_layers, activation, dropout, aggregator)

    view_1_obj = np.array([0, 1, 2])
    view_2_obj = np.array([3, 4])
    view_3_obj = np.array([5, 6, 7])

    src = []
    dst = []
    objs = np.concatenate([view_1_obj, view_2_obj, view_3_obj])
    for src_node_id in objs:
        for dst_node_id in objs:
            if src_node_id == dst_node_id:
                continue
            src.append(src_node_id)
            dst.append(dst_node_id)

    edge_pred_graph = dgl.graph((src, dst))
    edge_pred_graph.ndata['feature'] = torch.randn(8, 256)
    edge_pred_graph.edata['label'] = torch.randint(0, 2, size=(56, ))

    edge_pred_graph.edata['train_mask'] = torch.zeros(56, dtype=torch.bool).bernoulli(0.6)
    edge_pred_graph.edata['val_mask'] = torch.zeros(56, dtype=torch.bool).bernoulli(0.2)

    node_features = edge_pred_graph.ndata["feature"]
    edge_label = edge_pred_graph.edata['label']
    train_mask = edge_pred_graph.edata['train_mask']
    valid_mask = edge_pred_graph.edata['val_mask']



    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # create sampler
    sampler = NeighborSampler(edge_pred_graph, [int(fanout) for fanout in fan_out.split(',')])

    for epoch in range(50):
        pred = model(edge_pred_graph, [edge_pred_graph for i in range(n_layers)], node_features)
        loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()

        # acc = evaluate(model, edge_pred_graph, node_features, edge_label, valid_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"loss: {loss.item()}\t ")



