import importlib


def get_model(cfg, datamodule, phase="train"):
    modeltype = cfg.model.model_type
    if modeltype == "mld":
        return get_module(cfg, datamodule)
    else:
        raise ValueError(f"Invalid model type {modeltype}.")


def get_module(cfg, datamodule):
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="mld.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")
    return Model(cfg=cfg, datamodule=datamodule)
