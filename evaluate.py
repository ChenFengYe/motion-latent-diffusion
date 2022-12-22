import json
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    # create logger
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.MODEL.MODEL_TYPE), str(cfg.NAME))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataset
    datasets = get_datasets(cfg, logger=logger, phase="test")
    npy_datasets = get_datasets(cfg, logger=logger, phase="npy")
    logger.info("datasets module {} initialized".format("".join(cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets[0])
    logger.info("model {} loaded".format(cfg.MODEL.MODEL_TYPE))

    # optimizer
    # metric_monitor = {
    #     "Train_jf": "recons/text2jfeats/train",
    #     "Val_jf": "recons/text2jfeats/val",
    #     "Train_rf": "recons/text2rfeats/train",
    #     "Val_rf": "recons/text2rfeats/val",
    #     "APE root": "Metrics/APE_root",
    #     "APE mean pose": "Metrics/APE_mean_pose",
    #     "AVE root": "Metrics/AVE_root",
    #     "AVE mean pose": "Metrics/AVE_mean_pose"
    # }

    # # callbacks
    # callbacks = [
    #     pl.callbacks.RichProgressBar(),
    #     ProgressLogger(metric_monitor=metric_monitor),
    # ]
    # logger.info("Callbacks initialized")

    # trainer
    # trainer = pl.Trainer(
    #     benchmark=False,
    #     max_epochs=cfg.TRAIN.END_EPOCH,
    #     accelerator=cfg.ACCELERATOR,
    #     devices=list(range(len(cfg.DEVICE))),
    #     default_root_dir=cfg.FOLDER_EXP,
    #     log_every_n_steps=cfg.LOGGER.LOG_EVERY_STEPS,
    #     deterministic=False,
    #     detect_anomaly=False,
    #     enable_progress_bar=True,
    #     logger=None,
    #     callbacks=callbacks,
    # )

    # loading state dict
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    ckpt = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    # calculate metrics
    # metrics = trainer.test(model, datamodule=datasets[0])
    # metrics = trainer.validate(model, datamodule=datasets[0])
    tgt_dir = cfg.TEST.TEST_DIR
    gt_dataset = datasets[0].test_dataset
    npy_dataset = npy_datasets[0].test_dataset
    feats2joints = datasets[0].feats2joints
    name_list = gt_dataset.name_list[gt_dataset.pointer :]
    print(len(name_list))
    name_list = npy_dataset.name_list[npy_dataset.pointer :]
    print(len(name_list))
    for i, keyid in enumerate(track(name_list, f"Processing metrcis evaluation")):
        # reference motions
        # ref_datas = gt_dataset.data_dict[keyid]
        (
            rec_word_emb,
            rec_pos_one_hots,
            rec_texts,
            rec_texts_len,
            rec_features,
            rec_lengths,
            _,
        ) = gt_dataset[i]
        rec_features = torch.tensor(rec_features, device=device).unsqueeze(0).float()
        rec_joints = feats2joints(rec_features)
        rec_batch = {
            "motion": rec_features,
            "text": [rec_texts[0]],
            "length": [rec_lengths],
            "word_embs": torch.tensor(
                rec_word_emb, device=device, dtype=rec_features.dtype
            ).unsqueeze(0),
            "pos_ohot": torch.tensor(
                rec_pos_one_hots, device=device, dtype=rec_features.dtype
            ).unsqueeze(0),
            "text_len": torch.tensor(
                rec_texts_len, device=device, dtype=rec_features.dtype
            ).unsqueeze(0),
        }
        rs_rec = model.eval_gt(rec_batch)
        # reconstructed motions
        (
            _,
            _,
            _,
            _,
            ref_features,
            ref_lengths,
            _,
        ) = npy_dataset[i]
        ref_features = torch.tensor(ref_features, device=device).unsqueeze(0).float()
        ref_joints = feats2joints(ref_features)
        ref_batch = rec_batch.copy()
        ref_batch.update(
            {
                "motion": ref_features,
                "length": [ref_lengths],
            }
        )
        rs_ref = model.eval_gt(ref_batch)
        length = [min(ref_lengths, rec_lengths)]
        for metric in model.metrics_dict:
            if metric == "TemosMetric":
                getattr(model, metric).update(rec_joints, ref_joints, length)
            elif metric == "TM2TMetrics":
                getattr(model, metric).update(
                    rs_rec["lat_t"], rs_rec["lat_m"], rs_ref["lat_m"], length
                )

    metrics = {}
    for metric in model.metrics_dict:
        metrics_dict = getattr(model, metric).compute(sanity_flag=False)
        metrics.update(
            {
                f"Metrics/{metric}": value.item()
                for metric, value in metrics_dict.items()
            }
        )
    print(metrics)

    # save metrics to file
    metric_file = output_dir.parent / f"metrics_{cfg.TIME}.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Testing done, the metrics are saved to {str(metric_file)}")


if __name__ == "__main__":
    main()
