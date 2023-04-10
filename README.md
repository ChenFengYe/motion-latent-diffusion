# MLD: Motion Latent Diffusion Models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/executing-your-commands-via-motion-diffusion/motion-synthesis-on-humanml3d)](https://paperswithcode.com/sota/motion-synthesis-on-humanml3d?p=executing-your-commands-via-motion-diffusion)
![Pytorch_lighting](https://img.shields.io/badge/Pytorch_lighting->=1.7-Blue?logo=Pytorch) ![Diffusers](https://img.shields.io/badge/Diffusers->=0.7.2-Red?logo=diffusers)

### [Executing your Commands via Motion Diffusion in Latent Space](https://chenxin.tech/mld)

### [Project Page](https://chenxin.tech/mld) | [Arxiv](https://arxiv.org/abs/2212.04048) - CVPR 2023

Motion Latent Diffusion (MLD) is a **text-to-motion** and **action-to-motion** diffusion model. Our work achieves **state-of-the-art** motion quality and two orders of magnitude **faster** than previous diffusion models on raw motion data.

<p float="center">
  <img src="https://user-images.githubusercontent.com/16475892/209251515-ea88127b-0783-4a88-a8c1-2e478f7210a2.png" width="800" />
</p>

## 🚩 News

- [2023/03/08] add [the script](https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/scripts/tsne.py) for latent space visualization and [the script](https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/scripts/flops.py) for the floating point operations (FLOPs)
- [2023/02/28] **MLD got accepted by CVPR 2023**!
- [2023/02/02] release action-to-motion task, please refer to [the config](https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/configs/config_mld_humanact12.yaml) and [the pre-train model](https://drive.google.com/file/d/1G9O5arldtHvB66OPr31oE_rJG1bH_R39/view)
- [2023/01/18] add a detailed [readme](https://github.com/ChenFengYe/motion-latent-diffusion/tree/main/configs) of the configuration
- [2023/01/09] release [no VAE config](https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/configs/config_novae_humanml3d.yaml) and [pre-train model](https://drive.google.com/file/d/1_mgZRWVQ3jwU43tLZzBJdZ28gvxhMm23/view), you can use MLD framework to train diffusion on raw motion like [MDM](https://github.com/GuyTevet/motion-diffusion-model).
- [2022/12/22] first release, demo, and training for text-to-motion
- [2022/12/08] upload paper and init project, code will be released in two weeks

## ⚡ Quick Start

<details>
  <summary><b>Setup and download</b></summary>
  
### 1. Conda environment

```
conda create python=3.9 --name mld
conda activate mld
```

Install the packages in `requirements.txt` and install [PyTorch 1.12.1](https://pytorch.org/)

```
pip install -r requirements.txt
```

We test our code on Python 3.9.12 and PyTorch 1.12.1.

### 2. Dependencies

Run the script to download dependencies materials:

```
bash prepare/download_smpl_model.sh
bash prepare/prepare_clip.sh
```

For Text to Motion Evaluation

```
bash prepare/download_t2m_evaluators.sh
```

### 3. Pre-train model

Run the script to download the pre-train model

```
bash prepare/download_pretrained_models.sh
```

### 4. (Optional) Download manually

Visit [the Google Driver](https://drive.google.com/drive/folders/1U93wvPsqaSzb5waZfGFVYc4tLCAOmB4C) to download the previous dependencies and model.

</details>

## ▶️ Demo

<details>
  <summary><b>Text-to-motion</b></summary>

We support text file or keyboard input, the generated motions are npy files.
Please check the `configs/asset.yaml` for path config, TEST.FOLDER as output folder.

Then, run the following script:

```
python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt
```

Some parameters:

- `--example=./demo/example.txt`: input file as text prompts
- `--task=text_motion`: generate from the test set of dataset
- `--task=random_sampling`: random motion sampling from noise
- ` --replication`: generate motions for same input texts multiple times
- `--allinone`: store all generated motions in a single npy file with the shape of `[num_samples, num_ replication, num_frames, num_joints, xyz]`

The outputs:

- `npy file`: the generated motions with the shape of (nframe, 22, 3)
- `text file`: the input text prompt
</details>

## 💻 Train your own models

<details>
  <summary><b>Training guidance</b></summary>

### 1. Prepare the datasets

Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup.
We will provide instructions for other datasets soon.

### 2.1. Ready to train VAE model

Please first check the parameters in `configs/config_vae_humanml3d.yaml`, e.g. `NAME`,`DEBUG`.

Then, run the following command:

```
python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
```

### 2.2. Ready to train MLD model

Please update the parameters in `configs/config_mld_humanml3d.yaml`, e.g. `NAME`,`DEBUG`,`PRETRAINED_VAE` (change to your `latest ckpt model path` in previous step)

Then, run the following command:

```
python -m train --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
```

### 3. Evaluate the model

Please first put the tained model checkpoint path to `TEST.CHECKPOINT` in `configs/config_mld_humanml3d.yaml`.

Then, run the following command:

```
python -m test --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml
```

</details>

## 👀 Visualization

<details>
  <summary><b>Render SMPL</b></summary>

### 1. Set up blender - WIP

Refer to [TEMOS-Rendering motions](https://github.com/Mathux/TEMOS) for blender setup, then install the following dependencies.

```
YOUR_BLENDER_PYTHON_PATH/python -m pip install -r prepare/requirements_render.txt
```

### 2. (Optional) Render rigged cylinders

Run the following command using blender:

```
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=YOUR_NPY_FOLDER --mode=video --joint_type=HumanML3D
```

### 2. Create SMPL meshes with:

```
python -m fit --dir YOUR_NPY_FOLDER --save_folder TEMP_PLY_FOLDER --cuda
```

This outputs:

- `mesh npy file`: the generate SMPL vertices with the shape of (nframe, 6893, 3)
- `ply files`: the ply mesh file for blender or meshlab

### 3. Render SMPL meshes

Run the following command to render SMPL using blender:

```
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=YOUR_NPY_FOLDER --mode=video --joint_type=HumanML3D
```

optional parameters:

- `--mode=video`: render mp4 video
- `--mode=sequence`: render the whole motion in a png image.
</details>

## ❓ FAQ

<details>
    <summary><b>Solve foot sliding issue</b></summary>
  
 If your demo results have a severe issue on foot sliding, please take a look to the below. It could happen when ``self.feats2joints`` (use mean and std for de-normalization) is broken. 
 https://github.com/ChenFengYe/motion-latent-diffusion/blob/af507c479d771f62a058b5b6abb51276b36d6c6d/mld/models/modeltype/mld.py#L264
 
</details>

<details>
  <summary><b>Details of training</b></summary>
  
1. **GPUs.** You can indicate the IDs to use all your GPUs.  https://github.com/ChenFengYe/motion-latent-diffusion/blob/6643f175fbcd914312fa5f570e3dc7ab57994075/configs/config_vae_humanml3d.yaml#L4
2.  **Epoch Nums.** 1500~3000 epoch is enough for VAE or MLD. I suggest you use **wandb**(prefer) or **tensorborad** to check FID curve of your training.
3. **Training Speed.** 2000 epoch could cost 1 day for a single GPU, and around 12 hours for 8 GPUs. Training speed also depends on ``VAL_EVERY_STEPS`` (Validation Frequency), DataIO Speed. Your training is a little slow.
https://github.com/ChenFengYe/motion-latent-diffusion/blob/6643f175fbcd914312fa5f570e3dc7ab57994075/configs/config_vae_humanml3d.yaml#L77
4. **Data Log.** Only loss print by default. After validation, more metrics of val will print. More details in wandb (prefer) or tensorborad.
5. **Debug or not.** Please use ``--nodebug`` for all your training.
6. **VAE loading.** Please load your pre-train VAE correctly for the MLD diffusion training.
7. **FID.** FID of validation will drop to 0.5~1 after 1500 epochs for both VAE and MLD training. By default, validation is on test split...https://github.com/ChenFengYe/motion-latent-diffusion/blob/6643f175fbcd914312fa5f570e3dc7ab57994075/configs/config_vae_humanml3d.yaml#L30
</details>

<details>
  <summary><b>Details of motion lengths</b></summary>
Our model is capable of generating motions with arbitrary lengths. To handle different lengths of motions in the same batch, padding and masking are utilized in our motion encoder and decoder. After latent vector <i>z</i> is obtained by diffusion process, motion length <i>L</i> represented as a sequence of positional encodings in the form of sinusoidal functions are also provided to the motion decoder, so our motion decoder is able to generate output with variable target lengths.

</details>

<details>
  <summary><b>MLD-1 VS MLD-7</b></summary>
MLD-7 only works best in evaluating VAE models (Tab. 4), and MLD-1 wins these generation tasks (Tab. 1, 2, 3, 6). In other words, MLD-7 wins the first training stage for the VAE part, while MLD-1 wins the second for the diffusion part. We thought MLD-7 should perform better than MLD-1 in several tasks, but the results differ. The main reason for this downgrade of a larger latent size, we believe, is the small amount of training data. HumanML3D only includes 15k motion sequences, much smaller than billions of images in image generation. MLD-7 could work better when the motion data amount reaches the million level.
</details>

</details>

<details>
  <summary><b>Details of Inference Time</b></summary>
We provide a detailed ablation study with DDIM below. We evaluate the total inference time to generate 2048 motion clips with different diffusion schedules, floating point operations (FLOPs) counted by THOP library, the size of diffusion input, and FID. MLD reduces the computational cost of diffusion models, which is the main reason for faster inference. The iterations of diffusion further widen the gap in computational cost.
<img width="839" alt="image" src="https://user-images.githubusercontent.com/24362526/223096066-79ff5879-d685-4ab9-b85e-9b55613df17b.png">
If you want to test the floating point operations (FLOPs) in your model setting, you can run the following command:

```
python -m scripts.flops --cfg configs/your_config.yaml
```

</details>

<details>
  <summary><b>Latent Space Visualization</b></summary>
We provide Visualization of the t-SNE results on evolved latent codes <i>z</i><sup>t</sup> during the reverse diffusion process (inference) on action-to-motion task below. <i>t</i> is the diffusion step but ordered in the forward diffusion trajectory. <i>z</i><sup>t</sup>=49 is the initial random noise. <i>z</i><sup>t</sup>=0 is our prediction. We sample 30 motions for each action label. From left to right, it shows the evolved latent codes during the inference of diffusion models.
<img width="1110" alt="image" src="https://user-images.githubusercontent.com/24362526/223096486-20c497f2-6f75-43af-a892-9e1215954ca4.png">
If you want to visualize Latent Space in your model setting, you can run the following command:

```
python -m scripts.tsne --cfg configs/your_config.yaml
```

**Note**: This only support action-to-motion models for now.

</details>

**[Details of configuration](./configs)**

## Citation

If you find our code or paper helps, please consider citing:

```bibtex
@inproceedings{chen2023mld,
  title     = {Executing your Commands via Motion Diffusion in Latent Space},
  author    = {Xin, Chen and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Jingyi and Yu, Gang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
}
```

## Acknowledgments

Thanks to [TEMOS](https://github.com/Mathux/TEMOS), [ACTOR](https://github.com/Mathux/ACTOR), [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and [joints2smpl](https://github.com/wangsen1312/joints2smpl), our code is partially borrowing from them.

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.
