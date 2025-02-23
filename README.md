# SaSPA: Advancing Fine-Grained Classification by Structure and Subject Preserving Augmentation

### 🏆 Accepted to NeurIPS 2024!  
[NeurIPS 2024 Poster](https://neurips.cc/virtual/2024/poster/95527)

![Teaser Image](docs/assets/teaser.png)


[![Github](https://img.shields.io/badge/Github%20webpage-222222.svg?style=for-the-badge&logo=github)](https://eyalmichaeli.github.io/SaSPA-Aug/)
[![ArXiv](https://img.shields.io/badge/ArXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2406.14551)
[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS%202024-%237845CC.svg?style=for-the-badge&logo=NeurIPS&logoColor=white)](https://neurips.cc/virtual/2024/poster/95527)


## Table of Contents
1. [Setup](#setup)
    - [Clone the repo](#clone-the-repo)
    - [Environment](#environment)
2. [Quick Setup](#quick-setup)
    - [Baseline Model](#baseline-model)
    - [Generation and Filtering](#generation-and-filtering)
    - [Training with Augmentations](#training-with-augmentations)
3. [Full Setup](#full-setup)
    - [Datasets](#datasets)
    - [Dataset Splits](#dataset-splits)
    - [Using Weights & Biases (wandb)](#using-weights--biases-wandb)
4. [Adding New Datasets](#adding-new-datasets)
5. [Citation](#citation)



## Setup

#### Clone the repo:
```bash
git clone https://github.com/EyalMichaeli/SaSPA-Aug.git
cd SaSPA-Aug
```


### Environment
```bash
conda env create -f environment.yml
conda activate saspa
```


# Quick Setup
For a quick setup, we will run on the planes dataset. The dataset is downloaded automatically.

## Running the Code
### Baseline Model
Download the `planes.pth` checkpoint from [Google Drive](https://drive.google.com/drive/folders/1Bios3Q4RsXcytsqd0e189C5yF9If06SD?usp=sharing) into `all_utils/checkpoints/planes`

### Generation and Filtering
run `run_aug/run_aug.py`. This script generates the data and at the end filters the data. The augmentations will be saved to `data/FGVC-Aircraft/fgvc-aircraft-2013b/data/aug_data/controlnet/sd_v1.5/canny/gpt-meta_class_prompt_w_sub_class_artistic_prompts_p_0.5_seed_1` together with a log file and after the generation is done, a JSON file that contains the original files paths and their respective augmentations.

### Training with Augmentations
Once you have the JSON file, copy its path to `fgvc/trainings_scripts/consecutive_runs_aug.sh`, under the variable `aug_json`. Then, run with 
```bash
bash fgvc/trainings_scripts/consecutive_runs_aug.sh
```

That's it!
You should see your training start at `<repo_path>/logs/dataset_name/`.  


# Full Setup

### Datasets

- **Aircraft, Cars, and DTD**: Downloaded automatically via torchvision to the local folder `data/<dataset_name>`.
- **CUB**: Download from [Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) to `data/CUB`.
- **CompCars**: Download from [CompCars dataset page](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) to `data/compcars`.

#### Dataset Splits
If the original dataset does not include a validation set, file names splits are provided in `fgvc/datasets_files` and are loaded automatically.

#### Using Weights & Biases (wandb)
In our experiments, we utilize [Weights & Biases (wandb)](https://wandb.ai/site) for training monitoring. The traning script auto-connects to wandb. To disable this, set the DONT_WANDB variable in train.py to True.

## Running the Code

### Training Baseline Model
For our filtering, we need a baseline model trained on the original dataset. We provide with pre-trained checkpoints for each dataset used in our paper in [Google Drive](https://drive.google.com/drive/folders/1Bios3Q4RsXcytsqd0e189C5yF9If06SD?usp=sharing), please either download or train a baseline model, and move the checkpoint to the folder `all_utils/checkpoints/<dataset_name>`.

### Prompts Construction
To create the prompts using GPT-4, follow the instructions in the paper.  
The generated prompts should be in `prompts_engineering/gpt_prompts`, which currently contain our generated prompts.  

### Generation and Filtering
The generation code is located at: `run_aug/run_aug.py`.  
Choose a dataset and ensure that `BASE_MODEL = "blip_diffusion"` and `CONTROLNET = "canny"`. If you don't want to use blip_diffusion, you can use other base models such as sd_v1.5 or sd_xl-turbo (Currently it's set for sd_v1.5 because it's better for the Aircraft dataset, for all other datasets, set `BASE_MODEL = "blip_diffusion"`).
The code will generate augmentations in the folder `<dataset_root>/aug_data` and then will automatically generate a JSON file with the filtered augmentations.  

### Training with Augmentations
Once you have the JSON file, copy its path to `trainings_scripts/consecutive_runs_aug.sh`, under the variable `aug_json`.   
Make sure the correct dataset is specified in the `dataset` variable and fill in the rest of the arguments (GPU ID, run_name, etc.). Currently, the appropriate arguments for training, such as augmentation ratio and traditional augmentation used, are automatically chosen in the script based on the dataset name.
After the script args are ready, run with 
```bash
bash fgvc/trainings_scripts/consecutive_runs_aug.sh
```

That's it!
You should see your training start at `<repo_path>/logs/<dataset_name>/`.  


## Adding New Datasets
To incorporate new datasets into the project, follow these structured steps:
- **Prompt Creation**: Begin by generating and adding new prompts to `prompts_engineering/gpt_prompts`.
- **Dataset Class Development**: Add a new dataset class within `all_utils/dataset_utils.py` to manage dataset-specific functionalities.
- **Dataset Module Implementation**: Add a new Python file in the `fgvc/datasets` folder.
- **Dataset Config**: Establish a new Python file with Hyper-parameters in the `fgvc/configs` folder.
- **Baseline Model Training**: Train a baseline model to ensure the new dataset is correctly integrated and functional. This model will also be used in the filtering process.
- **Follow Standard Procedures**: Proceed with the regular augmentation and training workflows as documented in [Running the Code](#running-the-code).



## Citation
If you find our work useful, we welcome citations:
```markdown
@inproceedings{
    michaeli2024advancing,
    title={Advancing Fine-Grained Classification by Structure and Subject Preserving Augmentation},
    author={Eyal Michaeli and Ohad Fried},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=MNg331t8Tj}
}
```

## Troubleshooting
 - You might need to re-install [PyTorch](https://pytorch.org/get-started/locally/) according to your server.


## Special Thanks
We extend our gratitude to the following resources for their significant contributions to our project:
- **CAL Repository**: Visit [CAL Repo](https://github.com/raoyongming/CAL) for more details.
- **Diffusers Package**: Learn more about the Diffusers package at [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/en/index).


