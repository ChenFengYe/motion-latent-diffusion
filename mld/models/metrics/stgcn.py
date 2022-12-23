from typing import List
import random
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
import os
from .utils import *

from mld.models.architectures import uestc_stgcn


class UESTCMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 cfg,
                 num_labels=40,
                 diversity_times=200,
                 multimodality_times=20,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "matching, fid, and diversity scores"

        self.num_labels = num_labels
        self.diversity_times = diversity_times
        self.multimodality_times = multimodality_times
        datapath = os.path.join(cfg.DATASET.SMPL_PATH, "kintree_table.pkl")
        # init classifier module
        self.stgcn_classifier = uestc_stgcn.STGCN(
            in_channels=6,
            kintree_path=datapath,
            num_class=cfg.DATASET.NCLASSES,
            graph_args={
                "layout": "smpl",
                "strategy": "spatial"
            },
            edge_importance_weighting=True)
        # load pretrianed
        model_path = os.path.join(cfg.MODEL.UESTC_REC_PATH,
                                  "uestc_rot6d_stgcn.tar")
        a2m_checkpoint = torch.load(model_path)
        self.stgcn_classifier.load_state_dict(a2m_checkpoint)
        # freeze params
        self.stgcn_classifier.eval()
        for p in self.stgcn_classifier.parameters():
            p.requires_grad = False

        # add metrics
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.metrics = []
        # Accuracy
        self.add_state("accuracy",
                       default=torch.tensor(0.),
                       dist_reduce_fx="mean")
        self.add_state("gt_accuracy",
                       default=torch.tensor(0.),
                       dist_reduce_fx="mean")
        self.metrics.extend(["accuracy", "gt_accuracy"])
        # Fid
        self.add_state("FID", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.add_state("gt_FID",
                       default=torch.tensor(0.),
                       dist_reduce_fx="mean")
        self.metrics.extend(["FID", "gt_FID"])
        # Diversity
        self.add_state("Diversity",
                       default=torch.tensor(0.),
                       dist_reduce_fx="mean")
        self.add_state("gt_Diversity",
                       default=torch.tensor(0.),
                       dist_reduce_fx="mean")
        self.metrics.extend(["Diversity", "gt_Diversity"])
        # Multimodality
        self.add_state("Multimodality",
                       default=torch.tensor(0.),
                       dist_reduce_fx="mean")
        self.add_state("gt_Multimodality",
                       default=torch.tensor(0.),
                       dist_reduce_fx="mean")
        self.metrics.extend(["Multimodality", "gt_Multimodality"])

        # chached batches
        self.add_state("confusion",
                       torch.zeros(num_labels, num_labels, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("gt_confusion",
                       torch.zeros(num_labels, num_labels, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("label_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # Accuracy
        self.accuracy = torch.trace(self.confusion) / torch.sum(self.confusion)
        self.gt_accuracy = torch.trace(self.gt_confusion) / torch.sum(
            self.gt_confusion)

        # cat all embeddings
        all_labels = torch.cat(self.label_embeddings, axis=0)
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0)
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0)
        all_gtmotions2 = all_gtmotions.clone()[
            torch.randperm(all_gtmotions.shape[0]), :]
        genstats = calculate_activation_statistics(all_genmotions)
        gtstats = calculate_activation_statistics(all_gtmotions)
        gtstats2 = calculate_activation_statistics(all_gtmotions2)

        all_labels = all_labels.cpu()

        # calculate diversity and multimodality
        self.Diversity, self.Multimodality = calculate_diversity_multimodality(
            all_genmotions,
            all_labels,
            self.num_labels,
            diversity_times=self.diversity_times,
            multimodality_times=self.multimodality_times)

        self.gt_Diversity, self.gt_Multimodality = calculate_diversity_multimodality(
            all_gtmotions, all_labels, self.num_labels)

        metrics.update(
            {metric: getattr(self, metric)
             for metric in self.metrics})

        # Compute Fid
        metrics["FID"] = calculate_fid(gtstats, genstats)
        metrics["gt_FID"] = calculate_fid(gtstats, gtstats2)

        return {**metrics}

    def update(
        self,
        label: Tensor,
        recmotion: Tensor,
        gtmotion: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        labels = label.squeeze().long()

        rec_out = self.stgcn_classifier(recmotion)
        gt_out = self.stgcn_classifier(gtmotion)
        # Accuracy
        batch_prob = rec_out["yhat"]
        gt_batch_prob = gt_out["yhat"]
        batch_pred = batch_prob.max(dim=1).indices
        for label, pred in zip(labels, batch_pred):
            self.confusion[label][pred] += 1

        gt_batch_pred = gt_batch_prob.max(dim=1).indices
        for label, pred in zip(labels, gt_batch_pred):
            self.gt_confusion[label][pred] += 1

        # Compute embeddings
        recmotion_embeddings = rec_out["features"]
        gtmotion_embeddings = gt_out["features"]

        # store all texts and motions
        self.label_embeddings.append(labels)
        self.recmotion_embeddings.append(recmotion_embeddings)
        self.gtmotion_embeddings.append(gtmotion_embeddings)
