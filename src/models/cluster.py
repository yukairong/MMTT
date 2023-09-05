import torch
import numpy as np
import networkx as nx
from sklearn import metrics
from torchvision.utils import draw_bounding_boxes, save_image

from utils.box_ops import get_color


class ClusterDetections:

    def __init__(self, inputs,  targets, graph):
        self.sof_flag = True if inputs.shape[-1] > 1 else False
        if not self.sof_flag:
            sig = torch.nn.Sigmoid()
            self.inputs = sig(inputs)
            self.inputs = inputs.squeeze(1).cpu()
            self.targets = targets.squeeze(1).int().cpu()
        else:
            sof = torch.nn.Softmax(dim=1)
            self.inputs = sof(inputs)
            self.targets = targets.squeeze(1).int().cpu()
        self.graph = graph

        self._cam_info = graph.ndata['cam'].cpu()
        self._act_edges = np.zeros(inputs.shape[0], dtype=np.int32)
        self._n_cams = torch.unique(self._cam_info).shape[0]
        self._n_node = graph.num_nodes()
        self._G = nx.Graph()
        self._G.add_nodes_from([n for n in range(self._n_node)])
        self._G.add_weighted_edges_from(self._filter_edges())

    def _filter_edges(self):
        weight_active_edges = []
        if self.sof_flag:
            preds = torch.argmax(self.inputs, dim=1)
            preds = preds.cpu()
            edges_id = np.where(preds == 1)[0]
        else:
            edges_id = np.where(self.inputs >= 0.5)[0]

        for edge_id, u, v in zip(edges_id, *self.graph.find_edges(edges_id)):
            weight_active_edges.append(
                (int(u), int(v), float(self.inputs[edge_id]))
            )
        self._act_edges[edges_id] = 1
        return weight_active_edges

    def scores(self):
        return (
            metrics.adjusted_rand_score(self.targets, self._act_edges),  # ARI
            metrics.adjusted_mutual_info_score(
                self.targets, self._act_edges),  # AMI
            metrics.accuracy_score(self.targets, self._act_edges),  # ACC
            metrics.homogeneity_score(self.targets, self._act_edges),   # H
            metrics.completeness_score(self.targets, self._act_edges),  # C
            metrics.v_measure_score(self.targets, self._act_edges)  # V-m
        )

    def pruning_and_splitting(self):
        self.pruning()
        self.splitting()

    def pruning(self):
        for cc in nx.connected_components(self._G):
            sub_graph = self._G.subgraph(cc)
            for node_id in cc:
                flow = sub_graph.degree(node_id)
                if flow > self._n_cams - 1:
                    prun_edges = sub_graph.edges(node_id, data=True)
                    min_edge = min(prun_edges, key=lambda x: x[-1]['weight'])
                    u, v = min_edge[:2]
                    self._G.remove_edge(u, v)
                    # Deactivate the deleted edge.
                    self._act_edges[self.graph.edge_ids(u, v)] = 0

    def splitting(self):
        for cc in nx.connected_components(self._G):
            sub_graph = self._G.subgraph(cc)
            for node_id in cc:
                edges = sub_graph.edges(node_id)
                if len(edges) == 0:
                    continue
                edge_uv = np.array([e for e in edges], dtype=np.int32)
                dst_cam = self._cam_info[edge_uv[:, 1]]
                error_cam = np.where(np.bincount(dst_cam) > 1)[0]
                for err_cam_id in error_cam:
                    split_edges = edge_uv[dst_cam == err_cam_id]
                    edge_weight = [sub_graph.get_edge_data(*e.tolist())['weight']
                                   for e in split_edges]
                    u, v = split_edges[np.argmin(edge_weight)]
                    self._G.remove_edge(u, v)
                    # Deactivate the deleted edge.
                    self._act_edges[self.graph.edge_ids(u, v)] = 0
