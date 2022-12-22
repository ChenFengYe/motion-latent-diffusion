from .base import BASEDataModule
from .a2m import HumanAct12Poses
import numpy as np


class Humanact12DataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "HumanAct12"
        self.Dataset = HumanAct12Poses
        self.cfg = cfg
        sample_overrides = {
            "num_seq_max": 2,
            "split": "test",
            "tiny": True,
            "progress_bar": False
        }
        # self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = 150
        self.njoints = 25
        self.nclasses = 12
        # self.transforms = self._sample_set.transforms

    # def mm_mode(self, mm_on=True):
    #     # random select samples for mm
    #     if mm_on:
    #         self.is_mm = True
    #         if self.split == 'train':
    #             self.name_list = self.test_dataset._train[index]
    #         else:
    #             self.name_list = self.test_dataset._test[index]
    #         self.name_list = self.test_dataset.name_list
    #         self.mm_list = np.random.choice(self.name_list,
    #                                         self.cfg.TEST.MM_NUM_SAMPLES,
    #                                         replace=False)
    #         self.test_dataset.name_list = self.mm_list
    #     else:
    #         self.is_mm = False
    #         self.test_dataset.name_list = self.name_list
