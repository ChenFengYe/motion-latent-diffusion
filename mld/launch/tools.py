from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import os


def resolve_cfg_path(cfg: DictConfig):
    working_dir = os.getcwd()
    cfg.working_dir = working_dir
