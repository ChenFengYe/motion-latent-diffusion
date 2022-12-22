from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class UncondMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "fid, kid, and diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = 300

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []

        # KID
        self.add_state("KID_mean",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="mean")
        self.add_state("KID_std",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="mean")
        self.metrics.extend(["KID_mean", "KID_std"])
        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.metrics.append("FID")

        # Diversity
        self.add_state("Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # chached batches
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

        # cat all embeddings
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0).cpu()
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0).cpu()

        # Compute kid

        KID_mean, KID_std = calculate_kid(all_gtmotions, all_genmotions)
        metrics["KID_mean"] = KID_mean
        metrics["KID_std"] = KID_std

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        assert count_seq > self.diversity_times
        print(all_genmotions.shape)
        print(all_gtmotions.shape)
        metrics["Diversity"] = calculate_diversity_np(all_genmotions,
                                                      self.diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(
            all_gtmotions, self.diversity_times)

        return {**metrics}

    def update(
        self,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
        recmotion_embeddings=None,
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        if recmotion_embeddings is not None:
            recmotion_embeddings = torch.flatten(recmotion_embeddings,
                                                 start_dim=1).detach()
            # store all texts and motions
            self.recmotion_embeddings.append(recmotion_embeddings)
        gtmotion_embeddings = torch.flatten(gtmotion_embeddings,
                                            start_dim=1).detach()

        self.gtmotion_embeddings.append(gtmotion_embeddings)
