from os.path import join as pjoin
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BASEDataModule(pl.LightningDataModule):

    def __init__(self, collate_fn, batch_size: int, num_workers: int):
        super().__init__()

        # self.dataloader_options = {
        #     "batch_size": batch_size, "num_workers": num_workers,"collate_fn": collate_datastruct_and_text}
        self.dataloader_options = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
        }

        # self.collate_fn = collate_fn
        self.persistent_workers = True
        self.is_mm = False
        # need to be overloaded:
        # - self.Dataset
        # - self._sample_set => load only a small subset
        #   There is an helper bellow (get_sample_set)
        # - self.nfeats
        # - self.transforms

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        split_file = pjoin(
            eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
            self.cfg.EVAL.SPLIT + ".txt",
        )
        return self.Dataset(split_file=split_file, **sample_params)

    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                # todo: config name not consistent
                subset = subset.upper() if subset != "val" else "EVAL"
                split = eval(f"self.cfg.{subset}.SPLIT")
                split_file = pjoin(
                    eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
                    eval(f"self.cfg.{subset}.SPLIT") + ".txt",
                )
                self.__dict__[item_c] = self.Dataset(split_file=split_file,
                                                     split=split,
                                                     **self.hparams)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")

    def setup(self, stage=None):
        self.stage = stage
        # Use the getter the first time to load the data
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            persistent_workers=True,
            **self.dataloader_options,
        )

    def predict_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options[
            "batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=True,
            **dataloader_options,
        )

    def val_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.EVAL.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.EVAL.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.val_dataset,
            persistent_workers=True,
            **dataloader_options,
        )

    def test_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options[
            "batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        # dataloader_options["drop_last"] = True
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=True,
            **dataloader_options,
        )
