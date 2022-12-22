import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import psutil

logger = logging.getLogger()


class ProgressLogger(Callback):

    def __init__(self, metric_monitor: dict, precision: int = 3):
        # Metric to monitor
        self.metric_monitor = metric_monitor
        self.precision = precision

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding
        metrics_str = []

        losses_dict = trainer.callback_metrics
        for metric_name, dico_name in self.metric_monitor.items():
            if dico_name in losses_dict:
                metric = losses_dict[dico_name].item()
                metric = metric_format.format(metric)
                metric = f"{metric_name} {metric}"
                metrics_str.append(metric)

        if len(metrics_str) == 0:
            return

        memory = f"Memory {psutil.virtual_memory().percent}%"
        line = line + ": " + "   ".join(metrics_str) + "   " + memory
        logger.info(line)
