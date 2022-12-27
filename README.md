# MLD: Motion Latent Diffusion Models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/executing-your-commands-via-motion-diffusion/motion-synthesis-on-humanml3d)](https://paperswithcode.com/sota/motion-synthesis-on-humanml3d?p=executing-your-commands-via-motion-diffusion)
![Pytorch_lighting](https://img.shields.io/badge/Pytorch_lighting->=1.7-Blue?logo=Pytorch) ![Diffusers](https://img.shields.io/badge/Diffusers->=0.7.2-Red?logo=diffusers)

### [Executing your Commands via Motion Diffusion in Latent Space](https://chenxin.tech/mld)

### [Project Page](https://chenxin.tech/mld) | [Arxiv](https://arxiv.org/abs/2212.04048)

Motion Latent Diffusion (MLD) is a **text-to-motion** and **action-to-motion** diffusion model. Our work achieves **state-of-the-art** motion quality and two orders of magnitude **faster** than previous diffusion models on raw motion data.

<p float="center">
  <img src="https://user-images.githubusercontent.com/16475892/209251515-ea88127b-0783-4a88-a8c1-2e478f7210a2.png" width="800" />
</p>

## 🚩 News

22/Dec/2022 - first release, demo, and training for text-to-motion

08/Dec/2022 - upload paper and init project, code will be released in two weeks

## ⚡ Quick Start


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

The outputs:

- `npy file`: the generated motions with the shape of (nframe, 22, 3)
- `text file`: the input text prompt
</details>

## 💻 Train your own models

<details>
  <summary><b>WIP</b></summary>

### 1. Prepare the datasets

### 2. Ready to train

Please first check the parameters in `configs/config_mld_humanml3d.yaml`, e.g. `NAME`,`DEBUG`.

Then, run the following command:

```
python -m train --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64
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
  <summary><b>Details for Rendering</b></summary>

### 1. Setup blender - WIP

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

## Citation

If you find our code or paper helps, please consider citing:

```
@article{chen2022mld,
  author = {Xin, Chen and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Jingyi and Yu, Gang},
  title = {Executing your Commands via Motion Diffusion in Latent Space},
  journal = {arXiv},
  year = {2022},
}
```

## Acknowledgments

Thanks to [TEMOS](https://github.com/Mathux/TEMOS), [ACTOR](https://github.com/Mathux/ACTOR), [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and [joints2smpl](https://github.com/wangsen1312/joints2smpl), our code is partially borrowing from them.

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.
