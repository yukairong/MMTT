import dgl
import dgl.nn as dlgnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dlgnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean'
        )
        self.conv2 = dlgnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean'
        )

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, out_classes):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, out_classes)

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        indices = torch.round(logits)
        # _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

if __name__ == '__main__':
    src = np.random.randint(0, 100, 500)
    dst = np.random.randint(0, 100, 500)
    # 建立无向图
    edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
    # 建立点和边特征,以及边的标签
    edge_pred_graph.ndata['feature'] = torch.randn(100, 10)
    # edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
    edge_pred_graph.edata['label'] = torch.randint(0, 2, size=(1000,))
    # 进行训练、验证和测试集划分
    edge_pred_graph.edata['train_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.6)
    edge_pred_graph.edata['val_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.2)

    node_features = edge_pred_graph.ndata["feature"]
    edge_label = edge_pred_graph.edata['label']
    train_mask = edge_pred_graph.edata['train_mask']
    valid_mask = edge_pred_graph.edata['val_mask']

    model = Model(10, 20, 5, 1)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(1000):
        pred = model(edge_pred_graph, node_features)
        loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()

        acc = evaluate(model, edge_pred_graph, node_features, edge_label, valid_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"loss: {loss.item()}\t acc: {acc}")