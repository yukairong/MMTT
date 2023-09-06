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
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.models.mlp import NodeFeatureEncoder, EdgeFeatureEncoder, EdgePredictor
from src.models.mpn import MPN
from src.models import build_model
from src.models.cluster import ClusterDetections
from src.datasets.gnn_wildtrack import build_gnn_wildtrack
from src.utils.misc import udf_collate_fn
from src.models.criterion import FocalLoss
import numpy as np


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
        self.init_trainer()
        if not os.path.exists(self.gnn_args.output):
            os.mkdir(self.gnn_args.output)

    def init_trainer(self):
        self.loss_name = self.gnn_args.loss_name
        self.loss_weight_custom = self.gnn_args.loss_weight_custom
        if self.gnn_args.loss_name == "Focal":
            alfa = torch.tensor([0.95])
            self.criterion = FocalLoss(reduction='mean', alpha=alfa)
            self.criterion_no_reduction = FocalLoss(
                reduction='none', alpha=alfa)
            self.weights = torch.tensor([])

        elif self.gnn_args.loss_name == "BCE_weighted":
            self.criterion = nn.BCEWithLogitsLoss(
                reduction='mean', pos_weight=torch.tensor(self.gnn_args.loss_weight))
            self.criterion_no_reduction = nn.BCEWithLogitsLoss(
                reduction='none', pos_weight=torch.tensor(self.gnn_args.loss_name.loss_weight))

        elif self.gnn_args.loss_name == "CE_weighted":
            if self.gnn_args.loss_weight_custom:
                self.criterion = nn.CrossEntropyLoss(reduction='none')
                self.criterion_no_reduction = nn.CrossEntropyLoss(
                    reduction='none')
                self.weights = torch.tensor([])
            else:
                pos_weight = self.gnn_args.loss_weight
                self.weights = torch.tensor([1., pos_weight]).cuda()
                self.criterion = nn.CrossEntropyLoss(
                    weight=weights, reduction='mean')
                self.criterion_no_reduction = nn.CrossEntropyLoss(
                    weight=weights, reduction='none')

        if self.gnn_args.add_FPR:
            self.FPR_flag = True
        else:
            self.FPR_flag = False

        self.flag_BCE = False
        self.FPR_alpha = self.gnn_args.FPR_alpha

    def load_feature_extractor(self):
        feature_extractor, _, _ = build_model(self.train_args)
        feature_extractor_checkpoint = torch.load(self.track_args.obj_detect_checkpoint_file,
                                                  map_location=lambda storage, loc: storage)
        feature_extractor_state_dict = feature_extractor_checkpoint['model']
        feature_extractor['track_model'].load_state_dict(
            feature_extractor_state_dict)
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
        output_dir = osp.join(self.gnn_args.output,
                              f'train-{int(time.time())}')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        optim = torch.optim.SGD(
            [{"params": self.node_feature_encoder.parameters()},
             {"params": self.edge_feature_encoder.parameters()},
             {"params": self.mpn.parameters()},
             {"params": self.predictor.parameters()}],
            lr=0.01,
            momentum=0.9,
            weight_decay=1.0e-4
        )
        lr_warmup_list = np.linspace(0, 0.01, 5 + 1, endpoint=False)
        lr_warmup_list = lr_warmup_list[1:]

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.gnn_args.epochs)
        scheduler_step = torch.optim.lr_scheduler.StepLR(
            optim, step_size=3, gamma=0.1)

        # optim = torch.optim.SGD(
        #     [{"params": self.node_feature_encoder.parameters()},
        #      {"params": self.edge_feature_encoder.parameters()},
        #      {"params": self.mpn.parameters()},
        #      {"params": self.predictor.parameters()}],
        #     lr=0.01
        # )
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, self.gnn_args.epochs, eta_min=1e-6)
        scheduler_warmup = GradualWarmupScheduler(
            optim, 1.0, 2, scheduler_step)
        # this zero gradient update is needed to avoid a warning message
        optim.zero_grad()
        optim.step()

        # train_set, eval_set = self.load_dataset()
        train_set, eval_set = self.load_dataset()
        train_loader = DataLoader(
            train_set, self.gnn_args.batch_size, True, collate_fn=udf_collate_fn, drop_last=True)
        eval_loader = DataLoader(
            eval_set, self.gnn_args.eval_batch_size, collate_fn=udf_collate_fn)
        writer = SummaryWriter(output_dir)
        logger.info("Training begin...")

        for epoch in range(self.gnn_args.epochs):
            self._train_one_epoch(epoch, train_loader,
                                  optim, scheduler_warmup, writer)
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
        result = {metric: float(score)
                  for metric, score in zip(self.metrics_name, scores)}
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

                y_pred_list = []
                for _ in range(self.gnn_args.max_passing_steps):
                    x_node, x_edge = self.mpn(graph, x_node, x_edge)
                    y_pred = self.predictor(x_edge)

                    y_pred_list.append(y_pred)

            loss, precision1, precision0, precision, loss_class1, loss_class0, list_pred_probs, FPR = self.compute_loss_acc(y_pred_list, y_true, self.criterion, "train",
                                                                                                                            self.criterion_no_reduction, self.flag_BCE, self.FPR_flag,
                                                                                                                            self.FPR_alpha, self.loss_weight_custom, self.loss_name, weights=self.weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lambda: float(loss))
            logger.info(f'Epoch: [{epoch}][{i}/{len(dataloader)}]\t'
                        f'Train Loss {loss.item():.3f} \t'
                        f'Train Acc 1 {np.sum(np.asarray([item for item in precision1])) / len(precision1):.3f} \t'
                        f'Train Acc 0 {np.sum(np.asarray([item for item in precision0])) / len(precision0):.3f} \t'
                        )

            #         step_loss = sigmoid_focal_loss(
            #             y_pred, y_true.reshape(-1, 1), 0.9, 5, "mean")
            #         step_losses.append(step_loss)
            #     graph_loss = sum(step_losses)
            #     graph_losses.append(graph_loss)
            # loss = sum(step_losses)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # loss_val = loss.item()
            # losses.append(loss_val)
        #     logger.info(f"epoch=({epoch}/{self.gnn_args.epochs - 1})"
        #                 f" | [{i + 1}/{len(dataloader)}]"
        #                 f" | loss={loss_val:.4f}"
        #                 f" | avg_graph_loss={loss_val / self.gnn_args.batch_size:.4f}")
        # avg_loss = sum(losses) / len(losses)
        # writer.add_scalar("Loss/train", avg_loss, epoch)
        # logger.info(f"finished epoch {epoch}. avg_train_loss={avg_loss:.4f}")

    def _eval_one_epoch(self, epoch: int, dataloader, writer):
        avg_scores = self._test_one_epoch(
            dataloader, self.gnn_args.max_passing_steps)
        log_string = ""
        for i, metric_name in enumerate(self.metrics_name):
            writer.add_scalar(f"Val/{metric_name}", avg_scores[i], epoch)
            log_string += f"{metric_name}={avg_scores[i]:.4f} | "
        logger.info(f"validation results at epoch={epoch}: {log_string}")

    @torch.no_grad()
    def _test_one_epoch(self, dataloader, max_passing_steps: int, visualize_output_dir=None):
        scores_collector = torch.zeros(
            len(self.metrics_name), dtype=torch.float32)
        n = 0
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            for graph, node_feature, edge_feature, y_true in data:
                x_node = self.node_feature_encoder(node_feature)
                x_edge = self.edge_feature_encoder(edge_feature)
                for _ in range(max_passing_steps):
                    x_node, x_edge = self.mpn(graph, x_node, x_edge)

                y_pred = self.predictor(x_edge)
                cluster = ClusterDetections(
                    y_pred, y_true.reshape(-1, 1), graph)
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

    def compute_loss_acc(self, preds_list, targets, criterion, mode,
                         criterion_no_reduction, flag_BCE, FPR_flag,
                         FPR_alpha, loss_weight_custom, loss_name, weights=None):
        # Define Balancing weight
        labels = targets.view(-1)

        # Compute Weighted BCE
        loss = 0
        loss_class1 = 0
        loss_class0 = 0
        precision_class1 = list()
        precision_class0 = list()
        precision_all = list()

        list_pred_prob = list()
        num_steps = len(preds_list)

        # Compute loss of all the steps and sum them
        # comment FOR CONSIDERING ALL STEPS
        step_ini = 0
        step_end = num_steps

        for step in range(step_ini, step_end):
            # FOR BinaryCE and BinaryCEweighted
            if flag_BCE:
                preds = preds_list[step].view(-1)

                if mode == "train":
                    loss_per_sample = criterion_no_reduction(preds, labels)
                    loss += criterion(preds,  labels)

                    loss_class1 += torch.mean(loss_per_sample[labels == 1])
                    loss_class0 += torch.mean(loss_per_sample[labels == 0])

                else:
                    loss_per_sample = F.binary_cross_entropy_with_logits(
                        preds, labels, reduction='none')

                    loss_class1 += torch.mean(loss_per_sample[labels == 1])
                    loss_class0 += torch.mean(loss_per_sample[labels == 0])

                    loss += F.binary_cross_entropy_with_logits(
                        preds, labels, reduction='mean')

                with torch.no_grad():
                    sig = torch.nn.Sigmoid()
                    preds_prob = sig(preds)
                    list_pred_prob.append(preds_prob.cpu())

            # FOR CEw
            else:
                preds = preds_list[step]
                labels = labels.long()
                predictions = torch.argmax(preds, dim=1)
                FP = torch.sum(predictions[labels == 0])
                TN = (predictions[labels == 0]).shape[0] - FP
                FPR = FP / (FP + TN)

                TP = torch.sum(predictions[labels == 1])
                TPR = TP / (TP + TN)

                if mode == "train":

                    if loss_name != 'Focal':

                        if loss_weight_custom == 'False' or loss_weight_custom == False:

                            loss_per_sample = criterion_no_reduction(
                                preds, labels)
                            loss_per_sample = loss_per_sample / \
                                weights[labels].sum()

                            loss_class1 += torch.sum(
                                loss_per_sample[labels == 1])
                            loss_class0 += torch.sum(
                                loss_per_sample[labels == 0])

                            loss += criterion(preds, labels)

                        else:
                            loss_per_sample = criterion_no_reduction(
                                preds, labels)
                            n_0 = len(labels) - sum(labels)
                            n_1 = sum(labels)
                            w_0b = (n_0 + n_1) / (n_0)
                            w_1b = (n_0 + n_1) / (n_1)
                            w_1 = w_1b / w_0b
                            w_0 = w_0b / w_0b
                            custom_weights = torch.tensor([w_0, w_1])
                            loss_per_sample[labels == 0] = (
                                loss_per_sample[labels == 0] * w_0)
                            loss_per_sample[labels == 1] = (
                                loss_per_sample[labels == 1] * w_1)
                            loss_per_sample = loss_per_sample / \
                                custom_weights[labels].sum()

                            loss_class1 += torch.sum(
                                loss_per_sample[labels == 1])
                            loss_class0 += torch.sum(
                                loss_per_sample[labels == 0])
                            loss += torch.sum(loss_per_sample)
                    else:

                        if loss_weight_custom == True or loss_weight_custom == 'True':
                            n_0 = len(labels) - sum(labels)
                            n_1 = sum(labels)
                            w_0b = (n_0 + n_1) / (2 * n_0)
                            w_1b = (n_0 + n_1) / (2 * n_1)
                            w_1 = w_1b / w_0b
                            w_0 = w_0b / w_0b

                            w_1_f = w_1 / w_1
                            w_0_f = w_0 / w_1

                            alfa = 1 - w_0_f
                            criterion_no_reduction = FocalLoss(
                                reduction='none', alpha=alfa)

                            loss_per_sample = criterion_no_reduction(
                                preds, labels)
                            loss_class1 += torch.mean(
                                loss_per_sample[labels == 1])
                            loss_class0 += torch.mean(
                                loss_per_sample[labels == 0])

                            loss += criterion(preds, labels)

                else:
                    loss_per_sample = criterion_no_reduction(preds, labels)

                    loss_class1 += torch.mean(loss_per_sample[labels == 1])
                    loss_class0 += torch.mean(loss_per_sample[labels == 0])

                    loss += F.cross_entropy(preds, labels, reduction='mean')

                if FPR_flag:
                    loss += float(FPR_alpha) * FPR

                with torch.no_grad():
                    sof = torch.nn.Softmax(dim=1)
                    preds_prob = sof(preds)[:, 1]
                    list_pred_prob.append(preds_prob.cpu())

        # Precision is computed only with last step predictions
        with torch.no_grad():

            if flag_BCE:
                preds = preds_list[-1].view(-1)
                sig = torch.nn.Sigmoid()
                preds_prob = sig(preds)
                predictions = (preds_prob >= 0.5) * 1
            else:
                preds = preds_list[-1]
                sof = torch.nn.Softmax(dim=1)
                preds_prob = sof(preds)

                predictions = torch.argmax(preds, dim=1)

            # Precision class 1
            index_label_1 = np.where(np.asarray(labels.cpu()) == 1)
            sum_successes_1 = np.sum(predictions.cpu().numpy(
            )[index_label_1] == labels.cpu().numpy()[index_label_1])

            if sum_successes_1 == 0:
                precision_class1.append(0)
            else:
                precision_class1.append(
                    (sum_successes_1 / len(labels[index_label_1])) * 100.0)

            # Precision class 0
            index_label_0 = np.where(np.asarray(labels.cpu()) == 0)
            sum_successes_0 = np.sum(predictions.cpu().numpy(
            )[index_label_0] == labels.cpu().numpy()[index_label_0])

            if sum_successes_0 == 0:
                precision_class0.append(0)
            else:
                precision_class0.append(
                    (sum_successes_0 / len(labels[index_label_0])) * 100.0)

            # Precision
            sum_successes = np.sum(
                predictions.cpu().numpy() == labels.cpu().numpy())
            if sum_successes == 0:
                precision_all.append(0)
            else:
                precision_all.append((sum_successes / len(labels)) * 100.0)

        return loss, precision_class1, precision_class0, precision_all, loss_class1, loss_class0, list_pred_prob, FPR
