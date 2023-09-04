import os.path
import time
import json
import os.path as osp

import tqdm
import torch
from loguru import logger
from torchvision.ops import sigmoid_focal_loss
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.mlp import NodeFeatureEncoder, EdgeFeatureEncoder, EdgePredictor
from src.models.mpn import MPN
from src.models import build_model
from src.models.cluster import ClusterDetections
from datasets.gnn_wildtrack import build_gnn_wildtrack
from utils.misc import udf_collate_fn


class GnnTrainer:

    def __init__(self, train_args, track_args, gnn_args):
        self.feature_extractor = None
        self.track_args = track_args
        self.gnn_args = gnn_args
        self.train_args = train_args
        self.node_feature_encoder = NodeFeatureEncoder(self.gnn_args.device)
        self.edge_feature_encoder = EdgeFeatureEncoder(self.gnn_args.device)
        self.mpn = MPN(self.gnn_args.device)
        self.predictor = EdgePredictor(self.gnn_args.device)
        self.metrics_name = ['ARI', 'AMI', 'ACC', 'H', 'C', 'V-m']

        self.load_feature_extractor()
        if not os.path.exists(self.gnn_args.output):
            os.mkdir(self.gnn_args.output)

    def load_feature_extractor(self):
        feature_extractor, _, _ = build_model(self.train_args)
        feature_extractor_checkpoint = torch.load(self.track_args.obj_detect_checkpoint_file,
                                                  map_location=lambda storage, loc: storage)
        feature_extractor_state_dict = feature_extractor_checkpoint['model']
        feature_extractor['track_model'].load_state_dict(feature_extractor_state_dict)
        feature_extractor['track_model'].to('cuda')

        self.feature_extractor = feature_extractor['track_model']

    def load_dataset(self):
        # dataset_mode = ['train', 'eval'] if self.gnn_args.train else ['test']
        # TODO: 需要添加eval
        dataset_mode = ['train', 'eval'] if self.gnn_args.train else ['test']
        if self.gnn_args.wildtrack:
            dataset = [build_gnn_wildtrack(mode, feature_extractor=self.feature_extractor, args=self.train_args)
                       for mode in dataset_mode]
        else:
            raise ValueError("Please assign a valid dataset for training!")

        # logger.info(f"Total graphs for training: {len(dataset[0])} and validating: {len(dataset[1])}"
        #             if self.gnn_args.train else f"Total graphs for testing: {len(dataset[0])}")
        # TODO: 缺少eval数据
        logger.info(f"Total graphs for training: {len(dataset[0])}"
                    if self.gnn_args.train else f"Total graphs for testing: {len(dataset[0])}")
        return dataset

    def train(self):
        output_dir = osp.join(self.gnn_args.output, f'train-{int(time.time())}')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        optim = torch.optim.Adam(
            [{"params": self.node_feature_encoder.parameters()},
             {"params": self.edge_feature_encoder.parameters()},
             {"params": self.mpn.parameters()},
             {"params": self.predictor.parameters()}],
            lr=0.01
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.gnn_args.epochs, eta_min=0.001)
        scheduler_warmup = GradualWarmupScheduler(optim, 1.0, 10, scheduler_cosine)
        # this zero gradient update is needed to avoid a warning message
        optim.zero_grad()
        optim.step()

        # train_set, eval_set = self.load_dataset()
        train_set, eval_set = self.load_dataset()
        train_loader = DataLoader(train_set, self.gnn_args.batch_size, True, collate_fn=udf_collate_fn, drop_last=True)
        eval_loader = DataLoader(eval_set, self.gnn_args.eval_batch_size, collate_fn=udf_collate_fn)
        writer = SummaryWriter(output_dir)
        logger.info("Training begin...")

        for epoch in range(self.gnn_args.epochs):
            self._train_one_epoch(epoch, train_loader, optim, scheduler_warmup, writer)
            self._eval_one_epoch(epoch, eval_loader, writer)
            self._save_one_epoch(epoch, output_dir)
        writer.close()

    def test(self):
        ckpt = torch.load(self.gnn_args.ckpt)
        self.node_feature_encoder.load_state_dict(ckpt['node_feature_encoder'])
        self.edge_feature_encoder.load_state_dict(ckpt['edge_feature_encoder'])
        self.mpn.load_state_dict(ckpt['mpn'])
        self.predictor.load_state_dict(ckpt['predictor'])

        output_dir = osp.join(self.gnn_args.output, f'test-{int(time.time())}')
        visualize_dir = None
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        test_set = self.load_dataset()[0]
        test_loader = DataLoader(test_set, collate_fn=udf_collate_fn)
        scores = self._test_one_epoch(test_loader, ckpt['L'], visualize_dir)
        result = {metric: float(score) for metric, score in zip(self.metrics_name, scores)}
        result['test-seq'] = test_set.seq_names
        with open(osp.join(output_dir, 'result.json'), 'w') as fp:
            json.dump(result, fp)
        logger.info(f"Test result has been saved in {output_dir} successfully")

    def _train_one_epoch(self, epoch: int, dataloader, optimizer, scheduler, writer):
        scheduler.step()
        losses = []
        for i, data in enumerate(dataloader):
            graph_losses = []
            for graph, node_feature, edge_feature, y_true in data:
                x_node = self.node_feature_encoder(node_feature)
                x_edge = self.edge_feature_encoder(edge_feature)
                step_losses = []

                for _ in range(self.gnn_args.max_passing_steps):
                    x_node, x_edge = self.mpn(graph, x_node, x_edge)
                    y_pred = self.predictor(x_edge)

                    step_loss = sigmoid_focal_loss(y_pred, y_true, 0.9, 5, "mean")
                    step_losses.append(step_loss)
                loss = sum(step_losses)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                losses.append(loss_val)
                logger.info(f"epoch=({epoch}/{self.gnn_args.epochs - 1})"
                            f" | [{i + 1}/{len(dataloader)}]"
                            f" | loss={loss_val:.4f}"
                            f" | avg_graph_loss={loss_val / self.gnn_args.batch_size:.4f}")
            avg_loss = sum(losses) / len(losses)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            logger.info(f"finished epoch {epoch}. avg_train_loss={avg_loss:.4f}")

    def _eval_one_epoch(self, epoch: int, dataloader, writer):
        avg_scores = self._test_one_epoch(dataloader, self.gnn_args.max_passing_steps)
        log_string = ""
        for i, metric_name in enumerate(self.metrics_name):
            writer.add_scalar(f"Val/{metric_name}", avg_scores[i], epoch)
            log_string += f"{metric_name}={avg_scores[i]:.4f} | "
        logger.info(f"validation results at epoch={epoch}: {log_string}")

    @torch.no_grad()
    def _test_one_epoch(self, dataloader, max_passing_steps: int, visualize_output_dir=None):
        scores_collector = torch.zeros(len(self.metrics_name), dtype=torch.float32)
        n = 0
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            for graph, node_feature, edge_feature, y_true in data:
                x_node = self.node_feature_encoder(node_feature)
                x_edge = self.edge_feature_encoder(edge_feature)
                for _ in range(max_passing_steps):
                    x_node, x_edge = self.mpn(graph, x_node, x_edge)

                y_pred = self.predictor(x_edge)
                cluster = ClusterDetections(y_pred, y_true, graph)
                cluster.pruning_and_splitting()

                scores = cluster.scores()
                scores_collector += torch.tensor(scores, dtype=torch.float32)
                n += 1
        return scores_collector / n

    def _save_one_epoch(self, epoch: int, output_dir):
        logger.info("Saving model...")
        model_path = osp.join(output_dir,
                              f"gnn_cca_{self.gnn_args.device}_epoch_{epoch}.pth.tar")
        torch.save({
            "node_feature_encoder": self.node_feature_encoder.state_dict(),
            "edge_feature_encoder": self.edge_feature_encoder.state_dict(),
            "mpn": self.mpn.state_dict(),
            "predictor": self.predictor.state_dict(),
            "L": self.gnn_args.max_passing_steps
        }, model_path)
        logger.info(f"Model has been saved in {model_path}.\n")