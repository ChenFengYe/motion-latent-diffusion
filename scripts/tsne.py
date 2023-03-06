import json
import os
from pathlib import Path

from os.path import join as pjoin

import numpy as np
import pytorch_lightning as pl
import torch
from rich.progress import track

from omegaconf import OmegaConf
from mld.data.utils import a2m_collate
from torch.utils.data import DataLoader
from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
import sklearn
from sklearn.manifold import TSNE
# from keras.datasets import mnist
from sklearn.datasets import load_iris
from numpy import reshape
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def data_parse(step: int, latents: np.ndarray, classids: list):
    nsample = 30

    # classids = list(range(0,12))
    nclass = len(classids)
    # (12, 50, 50, 256)
    t_0 = latents[classids,:nsample, step,:]
    t_0 = t_0.reshape(-1, t_0.shape[-1])


    # labels = np.array(list(range(0,nclass)))
    # labels = labels.repeat(nsample)

    labels = np.array(['sit', 'lift_dumbbell', 'turn_steering'])
    labels = labels.repeat(nsample)
    # labels = [['sit']* nsample,['lift_dumbbell']* nsample, ['turn steering wheel']* nsample]
    # labels = labels * nsample

    tsne = TSNE(n_components=2, verbose=0, random_state=123)
    z = tsne.fit_transform(t_0) 
    df = pd.DataFrame()

    # normalize
    z = 1.8*(z-np.min(z,axis=0))/(np.max(z,axis=0)-np.min(z,axis=0)) -0.9

    df["y"] = labels
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    return df

def drawFig(output_dir: str, latents: np.ndarray, classids: list = [8,6,5], steps: list = [0, 15, 35, 49] ):
    ''' 
    Draw the figure of t-SNE
    Parameters:
        output_dir: output directory
        latents: (12, 50, 50, 256)
        steps: list of diffusion steps to draw
        classids: list of class ids
            # 0: "warm_up",
            # 1: "walk",
            # 2: "run",
            # 3: "jump",
            # 4: "drink",
            # 5: "lift_dumbbell",
            # 6: "sit",
            # 7: "eat",
            # 8: "turn steering wheel",
            # 9: "phone",
            # 10: "boxing",
            # 11: "throw",
    '''

    sns.set()

    fig, axs = plt.subplots(1, 4, figsize=(4*3,2.5))

    nclass = len(classids)
    steps.sort(reverse=True)
    for i, step in enumerate(steps):
        df = data_parse(steps[0]-step,latents,classids)
        sns.scatterplot(ax=axs[i], x="comp-1", y="comp-2", hue='y',
                        legend = False if i != len(steps) -1  else True,
                        palette=sns.color_palette("hls", nclass),
                        data=df).set(title=r"t = {}".format(step)) 

        axs[i].set_xlim((-1, 1))
        axs[i].set_ylim((-1, 1))

    plt.legend(loc=[1.1,0.2], title='Action ID')

    plt.tight_layout()
    plt.savefig(pjoin(output_dir, 'TSNE.png'), bbox_inches='tight')
    plt.show()

def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    # create logger
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "tsne_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        #     str(x) for x in cfg.DEVICE)
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))
    subset = 'train'.upper() 
    split = eval(f"cfg.{subset}.SPLIT")
    split_file = pjoin(
                    eval(f"cfg.DATASET.{dataset.name.upper()}.SPLIT_ROOT"),
                    eval(f"cfg.{subset}.SPLIT") + ".txt",
                )
    dataloader = DataLoader(dataset.Dataset(split_file=split_file,split=split,**dataset.hparams),batch_size=8,collate_fn=a2m_collate)

    # create model
    model = get_model(cfg, dataset)
    logger.info("model {} loaded".format(cfg.model.model_type))

    # loading state dict
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval()
    
    # Device
    if cfg.ACCELERATOR == "gpu":
        device = torch.device("cuda")
        model = model.to(device)
    
    # Generate latent codes
    with torch.no_grad():
        labels = torch.tensor(np.array(list(range(0,dataset.nclasses)))).unsqueeze(1).to(device)
        lengths = torch.tensor([60]*dataset.nclasses).to(device)
        z_list = []
        for i in track(range(50),'Generating latent codes'):
            cond_emb = torch.cat((torch.zeros_like(labels), labels))
            # [steps, classes, latent_dim]
            z = model._diffusion_reverse_tsne(cond_emb, lengths)
            z_list.append(z)
        # [samples, steps, classes, latent_dim] -> [classes, samples, steps, latent_dim]
        latents = torch.stack(z_list, dim=0).permute(2,0,1,3).cpu().numpy()
        print(latents.shape)
        
    # Draw figure
    drawFig(output_dir, latents, classids = [8,6,5], steps = [0, 15, 35, 49])
    logger.info("TSNE figure saved to {}".format(output_dir))

if __name__ == "__main__":
    main()
