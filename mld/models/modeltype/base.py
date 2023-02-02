import os
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import LightningModule
from mld.models.metrics import ComputeMetrics, MRMetrics, TM2TMetrics, MMMetrics, HUMANACTMetrics, UESTCMetrics, UncondMetrics
from os.path import join as pjoin
from collections import OrderedDict


class BaseModel(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times = []

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if len(self.times) *self.cfg.TEST.BATCH_SIZE % (100) > 0 and len(self.times) > 0:
            print(f"Average time per sample ({self.cfg.TEST.BATCH_SIZE*len(self.times)}): ", np.mean(self.times)/self.cfg.TEST.BATCH_SIZE)
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        if split in ["val", "test"]:
            if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.metrics_dict:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict
            for metric in metrics_dicts:
                metrics_dict = getattr(
                    self,
                    metric).compute(sanity_flag=self.trainer.sanity_checking)
                # reset metrics
                getattr(self, metric).reset()
                dico.update({
                    f"Metrics/{metric}": value.item()
                    for metric, value in metrics_dict.items()
                })
        if split != "test":
            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.current_epoch),
            })
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        # # ToDo
        # # re-write vislization checkpoint?
        # # visualize validation
        # parameters = {"xx",xx}
        # vis_path = viz_epoch(self, dataset, epoch, parameters, module=None,
        #                         folder=parameters["folder"], writer=None, exps=f"_{dataset_val.dataset_name}_"+val_set)
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        self.save_npy(outputs)
        self.cfg.TEST.REP_I = self.cfg.TEST.REP_I + 1

        return self.allsplit_epoch_end("test", outputs)

    def on_save_checkpoint(self, checkpoint):
        # don't save clip to checkpoint
        state_dict = checkpoint['state_dict']
        clip_k = []
        for k, v in state_dict.items():
            if 'text_encoder' in k:
                clip_k.append(k)
        for k in clip_k:
            del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        # restore clip state_dict to checkpoint
        clip_state_dict = self.text_encoder.state_dict()
        new_state_dict = OrderedDict()
        for k, v in clip_state_dict.items():
            new_state_dict['text_encoder.' + k] = v
        for k, v in checkpoint['state_dict'].items():
            if 'text_encoder' not in k:
                new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        # load clip state_dict to checkpoint
        clip_state_dict = self.text_encoder.state_dict()
        new_state_dict = OrderedDict()
        for k, v in clip_state_dict.items():
            new_state_dict['text_encoder.' + k] = v
        for k, v in state_dict.items():
            if 'text_encoder' not in k:
                new_state_dict[k] = v

        super().load_state_dict(new_state_dict, strict)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}

    def configure_metrics(self):
        for metric in self.metrics_dict:
            if metric == "TemosMetric":
                self.TemosMetric = ComputeMetrics(
                    njoints=self.njoints,
                    jointstype=self.cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "TM2TMetrics":
                self.TM2TMetrics = TM2TMetrics(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "MRMetrics":
                self.MRMetrics = MRMetrics(
                    njoints=self.njoints,
                    jointstype=self.cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "HUMANACTMetrics":
                self.HUMANACTMetrics = HUMANACTMetrics(
                    datapath=os.path.join(self.cfg.model.humanact12_rec_path,
                                          "humanact12_gru.tar"),
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    multimodality_times=self.cfg.TEST.MM_NUM_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "UESTCMetrics":
                self.UESTCMetrics = UESTCMetrics(
                    cfg=self.cfg,
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    multimodality_times=self.cfg.TEST.MM_NUM_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "UncondMetrics":
                self.UncondMetrics = UncondMetrics(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            else:
                raise NotImplementedError(
                    f"Do not support Metric Type {metric}")
        if "TM2TMetrics" in self.metrics_dict or "UncondMetrics" in self.metrics_dict:
            self.MMMetrics = MMMetrics(
                mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
            )

    def save_npy(self, outputs):
        cfg = self.cfg
        output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.model_type),
                str(cfg.NAME),
                "samples_" + cfg.TIME,
            ))
        if cfg.TEST.SAVE_PREDICTIONS:
            lengths = [i[1] for i in outputs]
            outputs = [i[0] for i in outputs]
            if cfg.TEST.DATASETS[0].lower() in ["humanml3d", "kit"]:
                keyids = self.trainer.datamodule.test_dataset.name_list
                for i in range(len(outputs)):
                    for bid in range(
                            min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = outputs[i][bid].cpu().numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{cfg.TEST.REP_I}"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
            elif cfg.TEST.DATASETS[0].lower() in ["humanact12", "uestc"]:
                keyids = range(len(self.trainer.datamodule.test_dataset))
                for i in range(len(outputs)):
                    for bid in range(
                            min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = outputs[i][bid].cpu()
                        gen_joints = gen_joints.permute(2, 0,
                                                        1)[:lengths[i][bid],
                                                           ...].numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{cfg.TEST.REP_I}"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
