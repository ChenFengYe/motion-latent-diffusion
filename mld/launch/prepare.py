import os
import warnings
from pathlib import Path

import hydra
from mld.tools.runid import generate_id
from omegaconf import OmegaConf


# Local paths
def code_path(path=""):
    code_dir = hydra.utils.get_original_cwd()
    code_dir = Path(code_dir)
    return str(code_dir / path)


def working_path(path):
    return str(Path(os.getcwd()) / path)


# fix the id for this run
ID = generate_id()


def generate_id():
    return ID


def get_last_checkpoint(path, ckpt_name="last.ckpt"):
    output_dir = Path(hydra.utils.to_absolute_path(path))
    last_ckpt_path = output_dir / "checkpoints" / ckpt_name
    return str(last_ckpt_path)


def get_kitname(load_amass_data: bool, load_with_rot: bool):
    if not load_amass_data:
        return "kit-mmm-xyz"
    if load_amass_data and not load_with_rot:
        return "kit-amass-xyz"
    if load_amass_data and load_with_rot:
        return "kit-amass-rot"


OmegaConf.register_new_resolver("code_path", code_path)
OmegaConf.register_new_resolver("working_path", working_path)
OmegaConf.register_new_resolver("generate_id", generate_id)
OmegaConf.register_new_resolver("absolute_path", hydra.utils.to_absolute_path)
OmegaConf.register_new_resolver("get_last_checkpoint", get_last_checkpoint)
OmegaConf.register_new_resolver("get_kitname", get_kitname)


# Remove warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck*"
)

warnings.filterwarnings(
    "ignore", ".*Our suggested max number of worker in current system is*"
)


# os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "24"
