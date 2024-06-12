# SaSPA: Advancing Fine-Grained Classification by Structure and Subject Preserving Augmentation

![Teaser Image](images/teaser.png)


[![Github](https://img.shields.io/badge/Github%20webpage-222222.svg?style=for-the-badge&logo=github)](link)
[![ArXiv](https://img.shields.io/badge/ArXiv-B31B1B.svg?style=for-the-badge)](your_paper_link_here)


## Table of Contents
1. [Setup](#setup)
    - [Datasets](#datasets)
    - [Environment](#environment)
2. [Running the Code](#running-the-code)
    - [Training Baseline Model](#training-baseline-model)
    - [Prompts Construction](#prompts-construction)
    - [Generation and Filtering](#generation-and-filtering)
    - [Training with Augmentations](#training-with-augmentations)
3. [Adding New Datasets](#adding-new-datasets)
4. [Citation](#citation)

## Setup

### Datasets

- **Aircraft, Cars, and DTD**: Downloadable via torchvision.
- **CUB**: Download from [Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/).
- **CompCars**: Available at [CompCars dataset page](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/).


#### Dataset Splits
If the original dataset does not include a validation set, file names splits are provided in `fgvc/datasets_files` and are loaded automatically.


#### Configuration
After downloading, specify the dataset path in the `.py` file within `fgvc/datasets` and update the dataset class in `all_utils/dataset_utils.py`.


### Environment
```bash
conda env create -f environment.yml
```
you might want to re-install [PyTorch](https://pytorch.org/get-started/locally/) according to your server.


#### And install locally:
```bash
pip install -e .
```

#### Using Weights & Biases (wandb)
In our experiments, we utilize [Weights & Biases (wandb)](https://wandb.ai/site) to streamline and monitor the training process. The training scripts are designed to automatically connect to a wandb account. If you do not have a wandb account or prefer not to use this feature, you can disable wandb logging by setting the `DONT_WANDB` variable in the training script.


## Running the Code

### Training Baseline Model
For our filtering, we need a baseline model trained on the original dataset. We provide with pre-trained checkpoints for each dataset used in our paper in [Google Drive](https://drive.google.com/drive/folders/1Bios3Q4RsXcytsqd0e189C5yF9If06SD?usp=sharing), please download to the folder `all_utils/checkpoints`.

### Prompts Construction
To create the prompts using GPT-4, follow the instructions in the paper.  
The generated prompts should be in `prompts_engineering/gpt_prompts`, which currently contain our generated prompts.  

### Generation and Filtering
The generation code is located at: `run_aug/run_aug.py`.  
To use SaSPA, choose a dataset and ensure that `BASE_MODEL = "blip_diffusion"` and `CONTROLNET = "canny"`.  
The code will generate augmentations and then will automatically generate a JSON file with the filtered augmentations.  

### Training with Augmentations
Once you have the JSON file, copy it to `trainings_scripts/consecutive_runs_aug.sh`, under the variable `aug_json`.   
Make sure the correct dataset is specified in the `dataset` variable. The appropriate arguments for training, such as augmentation ratio and traditional augmentation, are automatically chosen in the script based on the dataset name.

That's it!
You should see your training start at `logs/dataset_name/`.  


## Adding New Datasets
To incorporate new datasets into the project, follow these structured steps:
- **Prompt Creation**: Begin by generating and adding new prompts to `prompts_engineering/gpt_prompts`.
- **Dataset Class Development**: Craft a new dataset class within `all_utils/dataset_utils.py` to manage dataset-specific functionalities.
- **Dataset Module Implementation**: Establish a new Python file in the `fgvc/datasets` folder.
- **Dataset Config**: Establish a new Python file with Hyper-parameters in the `fgvc/configs` folder.
- **Baseline Model Training**: Train a baseline model to ensure the new dataset is correctly integrated and functional. This model will also be used in the filtering process.
- **Follow Standard Procedures**: Proceed with the regular augmentation and training workflows as documented in [Running the Code](#running-the-code).


## Citation
If you find our work useful, we welcome citations:
```markdown
@article{michaeli2024saspa,
  author    = {Michaeli, Eyal and Fried, Ohad},
  title     = {Advancing Fine-Grained Classification by Structure and Subject Preserving Augmentation},
  journal   = {Arxiv},
  year      = {2024},
}
```

## Special Thanks
We extend our gratitude to the following resources for their significant contributions to our project:
- **CAL Repository**: Visit [CAL Repo](https://github.com/raoyongming/CAL) for more details.
- **Diffusers Package**: Learn more about the Diffusers package at [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/en/index).
