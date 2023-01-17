# Configuration

- [Configuration](#configuration)
  - [Experiment Name Explanation](#experiment-name-explanation)
  - [Configuration Explanation](#configuration-explanation)
    - [Asset Path Configuration](#asset-path-configuration)
    - [Network Structure Configuration](#network-structure-configuration)
    - [Experiment Configuration](#experiment-configuration)
    - [Default Configuration](#default-configuration)

## Experiment Name Explanation

Taking `1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01` as an example:

- `1222`: Eperiment date, for managing experiments
- `PELearn`: Ablation study abbreviations, here `PELearn` means we use learnable positional embedding for ablation study
- `Diff`: Stage flag, indicating whether it is the vae phase or the diffusion phase, `VAE` stands for the former and `Diff` stands for the latter
- `Latent1`: Latent size, here Latent1 indicates the latent vector shape is (1,256)
- `MEncDec49`: Numbers of head and layer of transformer-based Motion encoder & decoder
- `MdiffEnc49`: Numbers of head and layer of transformer-based Diffusion denoiser
- `bs64`: Batch size
- `clip`: Text encoder type, clip indicates here we use pretrained clip model as our text encoder
- `uncond75_01`: Classifier-free guidence parameter, here `uncond75_01` indicates the classifier-free guidence scale is 7.5 and probability is 0.1

## Configuration Explanation

We use yaml files for configuration. For training & evaluation, our whole configurations are combined of 4 parts.

[Asset Path Configuration](#asset-path-configuration)

[Network Structure Configuration](#network-structure-configuration)

[Experiment Configuration](#experiment-configuration)

[Default Configuration](#default-configuration)

### Asset Path Configuration

The assest configuration defines the file path of resources like datasets, dependence and so on.

**By default**, the program will use the [configs/asssets.yaml](./asssets.yaml) as asset configuration. You can either directly replace the file path in configs/asssets.yaml with yours or create a new yaml file refer to the annotations in [configs/asssets.yaml](./asssets.yaml) and then in the cli line command add `--cfg_asset` to specify your own yaml file.

### Network Structure Configuration

The network structure configuration defines the network structure settings. Our model ara mainly combined of four parts: Motion VAE, Text Encoder, Diffusion Denoiser, Diffusion Scheduler.

In addition, we use the evaluators from previous work for fair comparision with other motion generation work, so we also need evluators network for evaluation.

In conclusion, our network mainly combines the five components below.

1. Motion VAE
2. Text Encoder
3. Diffusion Denoiser
4. Diffusion Scheduler
5. Evaluators

**By default**, the program will use the yaml files in `configs/modules` folder as the configuration for each part of network. If you want to change the configurations of some part of network, you have two methods:

1. Directly modify the `traget` to use different structure and `params` to modify the parameters.
2. Create a new set of module configurations, you can create 5 new yaml files by refering the annotations in [configs/modules](./modules) files **in subfoler of `configs`**. Then specify the your modules folder name `model.target` in Experiment Configuration. Take [config_novae_humanml3d.yaml](./config_novae_humanml3d.yaml) as an example, we specify the `model.target=modules_novae` which means the experiment model will use the configuration files in `configs/modules_novae`.

### Experiment Configuration

The experiment configuration defines the settings apart from network structure like dataset settings, training settings, evaluation settings and so on.

For more details of the configuration, you can refer to the annotations in [config_mld_humanml3d.yaml](./config_mld_humanml3d.yaml)

### Default Configuration

The default configuration defines the default settings and will be overwritten by the configurations above.

**By default**, the program will use the yaml file `configs/base.yaml` folder as the basic configuration.
