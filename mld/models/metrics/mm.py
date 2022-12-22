from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class MMMetrics(Metric):
    full_state_update = True

    def __init__(self, mm_num_times=10, dist_sync_on_step=True, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MultiModality scores"

        self.mm_num_times = mm_num_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = ["MultiModality"]
        self.add_state("MultiModality",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")

        # chached batches
        self.add_state("mm_motion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        all_mm_motions = torch.cat(self.mm_motion_embeddings,
                                   axis=0).cpu().numpy()
        metrics['MultiModality'] = calculate_multimodality_np(
            all_mm_motions, self.mm_num_times)

        return {**metrics}

    def update(
        self,
        mm_motion_embeddings: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # store all mm motion embeddings
        self.mm_motion_embeddings.append(mm_motion_embeddings)
