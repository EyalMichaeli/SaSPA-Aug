"""
This file implements ALIA part in which we get the unique prompts from the captions of the images in the dataset
We will load the captions json created using blip_utils.py and get the unique prompts
"""
import json
import random
from pathlib import Path


def read_captions_from_json(json_file: str):
    with open(json_file, "r") as f:
        file = json.load(f)
        print(f"Read {len(file)} captions from {json_file}")
        return file
    
def get_unique_prompts(captions_dict: dict):
    prompts = [captions_dict[file_path]["caption"] for file_path in captions_dict]
    unique_prompts = list(set(prompts))
    print(f"Got {len(unique_prompts)} unique prompts")
    return unique_prompts


if __name__ == "__main__":
    dataset_name = "dtd"
    captions_to_use = 200
    captions_dict = read_captions_from_json(f"/mnt/raid/home/eyal_michaeli/git/thesis_utils/prompts_engineering/captions/{dataset_name}_captions.json")
    unique_prompts = get_unique_prompts(captions_dict)
    unique_prompts = random.sample(unique_prompts, captions_to_use)
    output_file = f"/mnt/raid/home/eyal_michaeli/git/thesis_utils/prompts_engineering/ALIA_prompts/chosen_captions/{dataset_name}_unique_prompts.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for prompt in unique_prompts:
            f.write(prompt + "\n")
    print(f"Saved {len(unique_prompts)} to\n {output_file}")
