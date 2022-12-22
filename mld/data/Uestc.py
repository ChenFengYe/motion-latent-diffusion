from .base import BASEDataModule
from .a2m import UESTC
import os
import rich.progress
import pickle as pkl


class UestcDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 method_name="vibe",
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "Uestc"

        # if method_name == "vibe":
        #     vibe_data_path = os.path.join(self.hparams.datapath,
        #                                   "vibe_cache_refined.pkl")
        #     with rich.progress.open(
        #             vibe_data_path, "rb",
        #             description="loading uestc vibe data") as f:
        #         vibe_data = pkl.load(f)
        #     self.hparams.update({"vibe_data": vibe_data})
        self.Dataset = UESTC
        self.cfg = cfg

        # self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = 150
        self.njoints = 25
        self.nclasses = 40
        # self.transforms = self._sample_set.transforms
