import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def main():
    """
    get input text
    ToDo skip if user input text in command
    current tasks:
         1 text 2 mtion
         2 motion transfer
         3 random sampling
         4 reconstruction

    ToDo 
    1 use one funtion for all expoert
    2 fitting smpl and export fbx in this file
    3 

    """
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")

    if cfg.DEMO.EXAMPLE:
        # Check txt file input
        # load txt
        from mld.utils.demo_utils import load_example_input

        text, length = load_example_input(cfg.DEMO.EXAMPLE)
        task = "Example"
    elif cfg.DEMO.TASK:
        task = cfg.DEMO.TASK
        text = None
    else:
        # keyborad input
        task = "Keyborad_input"
        text = input("Please enter texts, none for random latent sampling:")
        length = input(
            "Please enter length, range 16~196, e.g. 50, none for random latent sampling:"
        )
        if text:
            motion_path = input(
                "Please enter npy_path for motion transfer, none for skip:")
        # text 2 motion
        if text and not motion_path:
            cfg.DEMO.MOTION_TRANSFER = False
        # motion transfer
        elif text and motion_path:
            # load referred motion
            joints = np.load(motion_path)
            frames = subsample(
                len(joints),
                last_framerate=cfg.DEMO.FRAME_RATE,
                new_framerate=cfg.DATASET.KIT.FRAME_RATE,
            )
            joints_sample = torch.from_numpy(joints[frames]).float()

            features = model.transforms.joints2jfeats(joints_sample[None])
            motion = xx
            # datastruct = model.transforms.Datastruct(features=features).to(model.device)
            cfg.DEMO.MOTION_TRANSFER = True

        # default lengths
        length = 200 if not length else length
        length = [int(length)]
        text = [text]

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create mld model
    total_time = time.time()
    model = get_model(cfg, dataset)

    # ToDo
    # 1 choose task, input motion reference, text, lengths
    # 2 print task, input, output path
    #
    # logger.info(f"Input Text: {text}\nInput Length: {length}\nReferred Motion: {motion_path}")
    # random samlping
    if not text:
        logger.info(f"Begin specific task{task}")

    # debugging
    # vae
    # ToDo Remove this
    # temp loading
    # if cfg.TRAIN.PRETRAINED_VAE:
    #     logger.info("Loading pretrain vae from {}".format(cfg.TRAIN.PRETRAINED_VAE))
    #     ckpt = torch.load(cfg.TRAIN.PRETRAINED_VAE, map_location="cpu")
    #     model.load_state_dict(ckpt["state_dict"], strict=False)

    # /apdcephfs/share_1227775/shingxchen/AIMotion/TMOSTData/exps/actor/ACTOR_1010_vae_feats_kl/checkpoints/epoch=1599.ckpt

    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    # # remove mismatched and unused params
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     old, new = "denoiser.decoder.0.", "denoiser.decoder."
    #     # old1, new1 = "text_encoder.text_model.text_model", "text_encoder.text_model.vision_model"
    #     old1 = "text_encoder.text_model.vision_model"
    #     if k[: len(old)] == old:
    #         name = k.replace(old, new)
    #     # elif k[: len(old)] == old:
    #     #     name = k.replace(old, new)
    #     else:
    #         name = k

    #     new_state_dict[name] = v
    #     # if k.split(".")[0] not in ["text_encoder", "denoiser"]:
    #     #     new_state_dict[k] = v
    # model.load_state_dict(new_state_dict, strict=False)

    model.load_state_dict(state_dict, strict=True)

    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    mld_time = time.time()

    # sample
    with torch.no_grad():
        rep_lst = []    
        rep_ref_lst = []
        texts_lst = []
        # task: input or Example
        if text:
            # prepare batch data
            batch = {"length": length, "text": text}
            
            for rep in range(cfg.DEMO.REPLICATION):
                # text motion transfer
                if cfg.DEMO.MOTION_TRANSFER:
                    joints = model.forward_motion_style_transfer(batch)
                # text to motion synthesis
                else:
                    joints = model(batch)

                # cal inference time
                infer_time = time.time() - mld_time
                num_batch = 1
                num_all_frame = sum(batch["length"])
                num_ave_frame = sum(batch["length"]) / len(batch["length"])

                # upscaling to compare with other methods
                # joints = upsample(joints, cfg.DATASET.KIT.FRAME_RATE, cfg.DEMO.FRAME_RATE)
                nsample = len(joints)
                id = 0
                for i in range(nsample):
                    npypath = str(output_dir /
                                f"{task}_{length[i]}_batch{id}_{i}.npy")
                    with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                        text_file.write(batch["text"][i])
                    np.save(npypath, joints[i].detach().cpu().numpy())
                    logger.info(f"Motions are generated here:\n{npypath}")
                
                if cfg.DEMO.OUTALL:
                    rep_lst.append(joints)
                    texts_lst.append(batch["text"])
                    
                    
            if cfg.DEMO.OUTALL:
                grouped_lst = []
                for n in range(nsample):
                    grouped_lst.append(torch.cat([r[n][None] for r in rep_lst], dim=0)[None]) 
                combinedOut = torch.cat(grouped_lst, dim=0)
                try:
                    # save all motions
                    npypath = str(output_dir / f"{task}_{length[i]}_all.npy")
                    
                    np.save(npypath,combinedOut.detach().cpu().numpy())
                    with open(npypath.replace('npy','txt'),"w") as text_file: 
                        for texts in texts_lst:
                            for text in texts:
                                text_file.write(text)
                                text_file.write('\n')
                    logger.info(f"All reconstructed motions are generated here:\n{npypath}")
                except:
                    raise ValueError("Lengths of motions are different, so we cannot save all motions in one file.")
                    

        # random samlping
        if not text:
            if task == "random_sampling":
                # default text
                text = "random sampling"
                length = 196
                nsample, latent_dim = 500, 256
                batch = {
                    "latent":
                    torch.randn(1, nsample, latent_dim, device=model.device),
                    "length": [int(length)] * nsample,
                }
                # vae random sampling
                joints = model.gen_from_latent(batch)

                # latent diffusion random sampling
                # for i in range(100):
                #     model.condition = 'text_uncond'
                #     joints = model(batch)

                num_batch, num_all_frame, num_ave_frame = 100, 100 * 196, 196
                infer_time = time.time() - mld_time

                # joints = joints.cpu().numpy()

                # upscaling to compare with other methods
                # joints = upsample(joints, cfg.DATASET.KIT.FRAME_RATE, cfg.DEMO.FRAME_RATE)
                for i in range(nsample):
                    npypath = output_dir / \
                        f"{text.split(' ')[0]}_{length}_{i}.npy"
                    np.save(npypath, joints[i].detach().cpu().numpy())
                    logger.info(f"Motions are generated here:\n{npypath}")

            elif task in ["reconstrucion", "text_motion"]:
                for rep in range(cfg.DEMO.REPLICATION):
                    logger.info(f"Replication {rep}")
                    joints_lst = []
                    ref_lst = []
                    for id, batch in enumerate(dataset.test_dataloader()):
                        if task == "reconstrucion":
                            # batch = dataset.collate_fn(batch)
                            batch["motion"] = batch["motion"].to(device)
                            length = batch["length"]
                            joints, joints_ref = model.recon_from_motion(batch)
                        elif task == "text_motion":
                            # del batch["motion"]
                            batch["motion"] = batch["motion"].to(device)
                            joints, joints_ref = model(batch, return_ref=True)

                        nsample = len(joints)
                        length = batch["length"]
                        for i in range(nsample):
                            npypath = str(output_dir /
                                        f"{task}_{length[i]}_batch{id}_{i}_{rep}.npy")
                            np.save(npypath, joints[i].detach().cpu().numpy())
                            # if exps == "text-motion":
                            np.save(
                                npypath.replace(".npy", "_ref.npy"),
                                joints_ref[i].detach().cpu().numpy(),
                            )
                            with open(npypath.replace(".npy", ".txt"),
                                    "w") as text_file:
                                text_file.write(batch["text"][i])
                            logger.info(
                                f"Reconstructed motions are generated here:\n{npypath}"
                            )

            else:
                raise ValueError(
                    f"Not support task {task}, only support random_sampling, reconstrucion, text_motion"
                )

        # ToDo fix time counting
        total_time = time.time() - total_time
        print(f'MLD Infer time - This/Ave batch: {infer_time/num_batch:.2f}')
        print(f'MLD Infer FPS - Total batch: {num_all_frame/infer_time:.2f}')
        print(f'MLD Infer time - This/Ave batch: {infer_time/num_batch:.2f}')
        print(f'MLD Infer FPS - Total batch: {num_all_frame/infer_time:.2f}')
        print(
            f'MLD Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}')
        print(
            f'MLD Infer FPS - {num_all_frame/infer_time:.2f}s')
        print(
            f'MLD Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}')

        # todo no num_batch!!!
        # num_batch=> num_forward
        print(
            f'MLD Infer FPS - time for 100 Poses: {infer_time/(num_batch*num_ave_frame)*100:.2f}'
        )
        print(
            f'Total time spent: {total_time:.2f} seconds (including model loading time and exporting time).'
        )

    if cfg.DEMO.RENDER:
        # plot with lines
        # from mld.data.humanml.utils.plot_script import plot_3d_motion
        # fig_path = Path(str(npypath).replace(".npy",".mp4"))
        # plot_3d_motion(fig_path, joints, title=text, fps=cfg.DEMO.FRAME_RATE)

        # single render
        # from mld.utils.demo_utils import render
        # figpath = render(npypath, cfg.DATASET.JOINT_TYPE,
        #                  cfg_path="./configs/render_cx.yaml")
        # logger.info(f"Motions are rendered here:\n{figpath}")

        from mld.utils.demo_utils import render_batch

        blenderpath = cfg.RENDER.BLENDER_PATH
        render_batch(os.path.dirname(npypath),
                     execute_python=blenderpath,
                     mode="sequence")  # sequence
        logger.info(f"Motions are rendered here:\n{os.path.dirname(npypath)}")


if __name__ == "__main__":
    main()
