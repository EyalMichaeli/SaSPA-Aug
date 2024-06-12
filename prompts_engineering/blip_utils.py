import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

import torch
from lavis.models import load_model_and_preprocess
from all_utils.utils import *
from all_utils.dataset_utils import *
from prompts_engineering.txt2sentance_prompts import DATASET_TO_LABEL_DICT


def read_prompts_from_json(json_file: str, dataset_name: str, per_class=False):

    with open(json_file, "r") as f:
        file = json.load(f)
        print(f"Read {len(file)} prompts from {json_file}")
    if per_class:
        return file
    else:
        prompts = []
        for k, v in file.items():
            prompts += v
        return prompts

    
def write_captions_of_a_dataset_to_json(dataset_name: str, image_paths: list, output_file: str, questions: list = []):
    print(f"Writing captions of {dataset_name} to:\n{output_file}")
    print(f"Questions: {questions}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    caption_blip_model, caption_vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    dict_to_save = {}
    for image_path in tqdm(image_paths):
        input_image = Image.open(image_path).convert("RGB")
        image_for_blip = caption_vis_processors["eval"](input_image).unsqueeze(0).to(device)
        image_caption = caption_blip_model.generate({"image": image_for_blip})[0]
        image_for_vqa = vis_processors["eval"](input_image).unsqueeze(0).to(device)
        dict_to_save[image_path] = {"caption": image_caption}

        if questions:
            for q in questions:
                question = txt_processors["eval"](q)
                answer = model.predict_answers(samples={"image": image_for_vqa, "text_input": question}, inference_method="generate")[0]
                dict_to_save[image_path][q] = answer

        if random.random() < 0.01:
            print(f"Example of Caption: {image_caption}")
            if questions:
                print(f"Question: {q}, Answer: {answer}")

    with open(output_file, "w") as f:
        json.dump(dict_to_save, f)

    print(f"Saved to\n {output_file}")

def read_captions_from_json(json_file: str):
    with open(json_file, "r") as f:
        file = json.load(f)
        print(f"Read {len(file)} captions from {json_file}")
        return file


def get_caption_background_time_of_day_given_file_path_and_json(captions_dict: dict, file_path: str):
    file_captions_and_answers = captions_dict[file_path]
    caption = file_captions_and_answers["caption"]
    background = file_captions_and_answers["What is the background?"]
    time_of_day = file_captions_and_answers["is it day or night?"]
    return f"{caption}, background is {background}, {time_of_day}time"


if __name__ == "__main__":
    """
    nohup python /mnt/raid/home/user_name/git/thesis_utils/prompts_engineering/blip_utils.py > /mnt/raid/home/user_name/git/thesis_utils/prompts_engineering/blip_utils.log 2>&1 &
    """
    device = torch.device("cuda:1")
    output_path = Path(__file__).parent / "captions"
    dataset_name = "dtd"
    ds_utils = DS_UTILS_DICT[dataset_name]()
    image_paths = ds_utils.original_images_paths

    write_captions_of_a_dataset_to_json(
        dataset_name=dataset_name,
        image_paths=image_paths,
        output_file=f"{str(output_path)}/{dataset_name}_captions.json"
    )
