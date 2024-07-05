from pathlib import Path
import sys, random
import pickle
import json
from tqdm import tqdm
from all_utils import utils, dataset_utils


def word2sentence(classnames, num=200, save_path='', all_classes=False, must_keywords=[], dataset=""):
    print(f"Generating {num} sentences for each class: {classnames}")
    print(f"Keywords that must be in the sentence: {must_keywords}")

    sentence_dict = {}
    num_skipped_due_to_no_class_in_sentence = 0
    for n in classnames:
        sentence_dict[n] = []
    for cls in tqdm(classnames):
        for i in range(num):
            if all_classes:
                inp = f"{DATASET_TO_LABEL_DICT[dataset][0]}, of type {cls}"
                if i == 0:
                    print(f"Input for class {cls}: {inp}")
            else:
                if dataset == "compcars-parts":
                    inp = f"{cls}"
                else:
                    inp = f"{DATASET_TO_LABEL_DICT[dataset][0]}"

            sentence = nlp([inp], num_return_sequences=1, do_sample=True)
            if i < 2:  # print 2 examples
                print(f" Example Input & Output for class: {cls}:\n input: {inp}\n output: {sentence}")
            # if (not all_classes and any([possible_class in sentence.lower() for possible_class in must_keywords])) or (all_classes and cls in sentence): 
            #---- didnt use this because the strings can be mixed so that boing 737-400 can be plane of type boing sub type 737-400. 
            # so just checking if a reasonable keyword is in the sentence is enough
            if any([possible_class in sentence.lower() for possible_class in must_keywords]):
                sentence_dict[cls].append(sentence)
            else:
                print(f"Neither of the classes {must_keywords} is in the sentence: {sentence}, skipping...")
                num_skipped_due_to_no_class_in_sentence += 1
                continue
    
    print(f"num_skipped_due_to_no_class_in_sentence: {num_skipped_due_to_no_class_in_sentence}")
    print(f"total number of sentences: {sum([len(v) for k, v in sentence_dict.items()])}")
    print(f"total number of classes: {len(sentence_dict)}")

    # remove duplicate
    sampled_dict = {}
    for k, v in tqdm(sentence_dict.items()):
        v_unique = list(set(v))
        # v_unique = random.sample(v_unique, num)
        sampled_dict[k] = v_unique

    # save sampled_dict as a json
    with open(save_path, 'w') as fp:
        json.dump(sampled_dict, fp)


def main(dataset: str, num: int, output_path: str, all_classes: bool):
    assert dataset in dataset_utils.DATASETS_SUPPORTED
    assert num > 0

    Path(output_path).mkdir(parents=True, exist_ok=True)
    save_path = Path(output_path) / f"LE_{num}_{dataset}_all_classes_{all_classes}.json"
    if not all_classes:
        if dataset == "compcars-parts":
            utils_to_use = dataset_utils.CompCarsPartsUtils()
            labels = [utils_to_use.get_basic_prompt(part) for part in range(1, 5)]
        else:
            labels = DATASET_TO_LABEL_DICT[dataset]
    else:
        utils_to_use = dataset_utils.DS_UTILS_DICT[dataset]()
        
        labels = utils_to_use.get_classes()

    print(f"Labels: {labels}")
    print(f"Saving to: \n{save_path}")
    word2sentence(labels, num, save_path, all_classes=all_classes, must_keywords=DATASET_TO_LABEL_DICT[dataset], dataset=dataset)
    print(f"saved to \n{save_path}")


# one of the keywords must be in the sentence produced by the model (e.g. "A car" must be in the sentence for the cars dataset)
DATASET_TO_LABEL_DICT =  {
    "planes": ['airplane', 'plane', 'aircraft', 'jet', 'aircraft'],
    "cars": ['car', 'vehicle', 'automobile', 'auto', 'motorcar'],
    "compcars": ['car', 'vehicle', 'automobile', 'auto', 'motorcar'],
    "compcars-parts": ["car", "vehicle", "automobile", "auto", "motorcar"],
    "cub": ["bird"],
    "dtd": ["texture"]
}


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=3 nohup python /mnt/raid/home/eyal_michaeli/git/thesis_utils/prompts_engineering/txt2sentance_prompts.py  > /mnt/raid/home/eyal_michaeli/git/thesis_utils/prompts_engineering/txt2sentance_prompts.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python /mnt/raid/home/eyal_michaeli/git/thesis_utils/prompts_engineering/txt2sentance_prompts.py
    """
    from keytotext import pipeline

    nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")

    # args
    NUM_PER_CLASS = 30  # number of sentences to generate for each class (e.g. a kia picanto, a toyota corolla, etc.)
    NUM_GENERAL = 200  # number of sentences for the general class (e.g. "A car", "A plane", etc.)
    OUTPUT_PATH = "/mnt/raid/home/eyal_michaeli/git/thesis_utils/prompts_engineering/txt2sentences_prompts"
    DATASET = "dtd" 

    # cll_classes = True means it will create prompts for each subclass. False means it will create prompts for the general class (e.g. "A car", "A plane", etc.)
    # main(DATASET, NUM_GENERAL, OUTPUT_PATH, all_classes=False)
    main(DATASET, NUM_PER_CLASS, OUTPUT_PATH, all_classes=True)
    

    # for dataset in DATASET_TO_LABEL_DICT:
    #     main(dataset, NUM_PER_CLASS, OUTPUT_PATH, True)  # for each class, 30 possible sentences
        # main(dataset, NUM_GENERAL, OUTPUT_PATH, False)
