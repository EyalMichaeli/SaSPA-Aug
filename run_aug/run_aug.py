import logging
import random
import sys
import os
from pathlib import Path
import traceback
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import torchvision.transforms as transforms

try:
    from controlnet_aux import HEDdetector
    from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, UniPCMultistepScheduler, AutoencoderKL
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
    from diffusers.pipelines import BlipDiffusionPipeline, BlipDiffusionControlNetPipeline
    from diffusers.pipelines import StableDiffusionInstructPix2PixPipeline
except ImportError:
    pass

from lavis.models import load_model_and_preprocess as load_lavis_model_and_preprocess

sys.path.append(str(Path(__file__).parent.parent))
from prompts_engineering import blip_utils, ARTISTIC_PROMPTS, IMAGE_VARIATIONS_PROMPTS
from all_utils import utils
from all_utils import dataset_utils

# to ignore a pytorch 2 compile logging:
import torch._dynamo
torch._dynamo.config.suppress_errors = True


assert torch.cuda.is_available(), "CUDA is not available"


torch.tensor([1.0]).cuda()  # this is to initialize cuda, so that the next cuda calls will not be slow. This can also prevent bugs


NEGATIVE_PROMPT = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
MAX_FILENAME_LENGTH = 40  # max length to save in the aug folder, peompt length is also limited to 100 chars (above)
MAX_PROMPT_LENGTH = 150  # max length of prompt to use (above this will be truncated)
DEVICE = "cuda:0"
DEBUG = 0  # wont use pytorch 2.0 compile (it's slow starting) and will use 4 paths only


BASE_MODEL_DICT = {
    "sd_v1.5": "runwayml/stable-diffusion-v1-5",
    "sd_v2.1": "stabilityai/stable-diffusion-2-1-base",
    "sd_xl": ["stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/stable-diffusion-xl-refiner-1.0"],
    "sd_xl-turbo": "stabilityai/sdxl-turbo",
    "blip_diffusion": "Salesforce/blipdiffusion",
    "blip_diffusion-controlnet": "Salesforce/blipdiffusion-controlnet",
    "blip_diffusion-edit": "",  # not implemented in diffusers yet
    "ip2p": "timbrooks/instruct-pix2pix"
}

CONTROLNET_DICT_SD = {
    "canny": "lllyasviel/control_v11p_sd15_canny",   # older (v1.0): "lllyasviel/sd-controlnet-canny",
    "hed": "lllyasviel/sd-controlnet-hed"
}

CONTROLNET_DICT_SD_XL = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    # "hed": "diffusers/controlnet-hed-sdxl-1.0" : doesn't exist yet
}


def random_geometric_transforms(seed, size):
    utils.set_seed(seed)
    crop_size = random.choice([
        (int(size[0] // 1.05), int(size[1] // 1.05)),
        (int(size[0] // 1.1), int(size[1] // 1.1)),
    ])
    transform_list = []
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if random.random() < 0.5:
        transform_list.append(transforms.RandomCrop(size=crop_size, padding=random.choice([10, 30]), pad_if_needed=True, fill=0, padding_mode='constant'))
    if random.random() < 0.5:
        transform_list.append(transforms.RandomAffine(degrees=(1, 3), translate=(0.001, 0.01), scale=(0.98, 0.99)))
    
    random_transforms = transform_list + [transforms.Resize(size, interpolation=Image.BICUBIC)]
    composed_transforms = transforms.Compose(random_transforms)
    return composed_transforms

def apply_random_geometric_transforms(image, seed):
    composed_transforms = random_geometric_transforms(seed, image.size[::-1])
    return composed_transforms(image)


class lavis_edit_model:
    def __init__(self):
        model, vis_preprocess, txt_preprocess = load_lavis_model_and_preprocess("blip_diffusion", "base", device=DEVICE, is_eval=True)
        self.model = model
        self.vis_preprocess = vis_preprocess["eval"]
        self.txt_preprocess = txt_preprocess["eval"]

    def pass_through_lavis_edit(self, source_image, subject_image, seed, prompt, cond_subject, src_subject, tgt_subject, guidance_scale, num_inference_steps, num_inversion_steps=50):
        # source_image = self.vis_preprocess(source_image).unsqueeze(0)
        text_prompt = [self.txt_preprocess(prompt)]
        cond_image = self.vis_preprocess(subject_image).unsqueeze(0).to(DEVICE)
        samples = {
                "cond_images": cond_image,  # reference image
                "cond_subject": cond_subject,  # subject to replace with
                "src_subject": src_subject,  # subject to replace
                "tgt_subject": tgt_subject,    # target subject to generate
                "prompt": text_prompt,
                "raw_image": source_image,
                }
        with torch.no_grad():
            output: Image.Image = self.model.edit(
                        samples,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        num_inversion_steps=num_inversion_steps,
                        neg_prompt=NEGATIVE_PROMPT,
                    )[1]
        return output


def init_pipeline(base_model, controlnet, SDEdit, use_compile=False, sampler="ddim"):
    """
    this function initializes the pipeline, and sets the relevant params.
    base_model: out of "sd_v1.5", "sd_v2.1", "sd_xl", "sd_xl-turbo", "blip_diffusion"
    controlnet: out of None, "canny", "hed"
    SDEdit: if True, will do IMG2IMG edit
    use_compile: if True, will compile the model (relevant only for pytorch 2.0)
    sampler: out of "ddim", "unipcmultistep"
    """
    assert base_model in BASE_MODEL_DICT.keys()
    assert controlnet in CONTROLNET_DICT_SD.keys() or controlnet in CONTROLNET_DICT_SD_XL.keys() or controlnet is None
    assert sampler in ["ddim", "unipcmultistep"]

    base_model_path = BASE_MODEL_DICT[base_model]
    if base_model == "sd_xl":
        base_model_path, refiner_model_path = base_model_path

    # init model
    if controlnet is None:
        if base_model == "sd_xl":
            if SDEdit:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                        refiner_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                    ).to(DEVICE)  # taken from: https://huggingface.co/docs/diffusers/main/en/using-diffusers/img2img
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    base_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                )
        elif base_model == "sd_xl-turbo":
            if SDEdit:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                        base_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                    ).to(DEVICE)
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    base_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                )
        elif base_model in ["sd_v1.5", "sd_v2.1"]:
            if SDEdit:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
        
        elif base_model == "blip_diffusion":
            pipe = BlipDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
    
        elif base_model == "ip2p":
            # used in ALIA: https://github.com/lisadunlap/ALIA/blob/main/editing_methods/instruct_pix2pix_example.py
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)

    else:  # with controlNet
        if base_model == "blip_diffusion":
            assert controlnet == "canny", "blip_diffusion model can only be used with canny controlnet"
            base_model_path = "Salesforce/blipdiffusion-controlnet"  # they have a special base model for controlnet: https://huggingface.co/docs/diffusers/v0.25.1/en/api/pipelines/blip_diffusion#diffusers.BlipDiffusionPipeline
            # controlnet is in the BLIP-diffusion code. no need to load it here
        else:
            controlnet_path = CONTROLNET_DICT_SD[controlnet] if base_model in ["sd_v1.5", "sd_v2.1"] else CONTROLNET_DICT_SD_XL[controlnet]
            controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
            controlnet.to(DEVICE)

        if base_model in ["sd_xl", "sd_xl-turbo"]:
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(DEVICE)
            if SDEdit:
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(    
                    base_model_path, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
                )
                
            else:
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    base_model_path, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
                )
        
        elif base_model in ["sd_v1.5", "sd_v2.1"]:
            if SDEdit:
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(    
                    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
                )
            else:
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
                )

        elif base_model == "blip_diffusion":
            pipe = BlipDiffusionControlNetPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
    
    # pipe.enable_xformers_memory_efficient_attention(): no need for pytorch > 2.0
    if use_compile and not DEBUG and not "blip_diffusion" in base_model:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    if not "blip_diffusion" in base_model:  # for blip_diffusion, the scheduler needs to be the default one
        if sampler == "unipcmultistep":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif sampler == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if base_model == "sd_xl-turbo":
        pipe.upcast_vae()  # https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo#speedup-sdxl-turbo-even-more
        if sampler == "unipcmultistep":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)#, timestep_spacing="trailing")
        elif sampler == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)#, timestep_spacing="trailing")  # timestep_spacing="trailing" is recommended for sdxl-turbo: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/sdxl_turbo

    return pipe


def pass_thorugh_pipe(base_model, pipe, prompt, orig_img, SDEdit, SDEdit_strength, num_inference_steps, generator, guidance_scale, control_cond_scale, 
                      negative_prompt=NEGATIVE_PROMPT, control_image=None, blip_src_category=None, blip_target_category=None):
    pipe_args = {
        "prompt": str(prompt),
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt
    }

    if "blip_diffusion" in base_model:
        pipe_args["reference_image"] = orig_img
        pipe_args["source_subject_category"] = blip_src_category
        pipe_args["target_subject_category"] = blip_target_category
        pipe_args["height"] = orig_img.size[1]
        pipe_args["width"] = orig_img.size[0]
        pipe_args["neg_prompt"] = NEGATIVE_PROMPT
        del pipe_args["negative_prompt"]
    
    if "ip2p" in base_model:
        pipe_args["image_guidance_scale"] = 1.3  # this was used by ALIA for biased planes.
        pipe_args["num_inference_steps"] = 100  # this was used by ALIA for biased planes.
        pipe_args["image"] = orig_img 
        
    if control_image is not None:  # if controlnet is used
        if SDEdit:  
            pipe_args["control_image"] = control_image  # naming is a bit confusing, the param name is different for the 2 pipelines
            pipe_args["controlnet_conditioning_scale"] = control_cond_scale

        elif "blip_diffusion" in base_model:
            pipe_args["condtioning_image"] = control_image
            pipe_args["height"] = control_image.size[1]
            pipe_args["width"] = control_image.size[0]

        else:
            pipe_args["image"] = control_image
            pipe_args["controlnet_conditioning_scale"] = control_cond_scale

    if base_model == "sd_xl" and control_image is None:  # for sd_xl, the denoising_end param is relevant only if controlnet is not used
        pipe_args["denoising_end"] = 1.0

    if SDEdit:
        pipe_args["image"] = orig_img
        pipe_args["strength"] = SDEdit_strength

    output = pipe(**pipe_args)
    return output.images[0]


def main(base_model, controlnet, SDEdit, device="cuda:0"):
    try:
        logging.info(f"Starting {version} with seed {SEED}")
        logging.info(f"PID: {os.getpid()}")
        logging.info(f"base model: {base_model}")
        logging.info(f"controlnet: {controlnet}")
        logging.info(f"dataset: {DATASET}")
        logging.info(f"num per image: {NUM_PER_IMAGE}")

        original_images_paths = ds_utils.original_images_paths
        num_errors = 0  # num times we got OOM error
        
        if PROMPT_WITH_SUB_CLASS:
            logging.info("PROMPT_WITH_SUB_CLASS")
        
        if USE_ARTISTIC_PROMPTS:
            logging.info(f"Using artistic prompts with prob {ARTISTIC_PROMPTS_PROB}")
        if USE_CAMERA_VARIATIONS_PROMPTS:
            logging.info(f"Using camera variations prompts with prob {CAMERA_VAIRATIONS_PROB}")

        if PROMPT_TYPE == "ALIA":
            logging.info("Using ALIA prompts")

        if not PROMPT_TYPE == "captions": 
            with open(prompts_file, "r") as f:
                prompts = f.readlines()
            prompts = [prompt.strip()[:MAX_PROMPT_LENGTH] for prompt in prompts]
            logging.info(f"Read {len(prompts)} prompts from {prompts_file}")

        if controlnet == "hed":
            hed_detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

        blip_src_category, blip_target_category = None, None
        if "blip_diffusion" in base_model:
            blip_src_category = ds_utils.meta_class
            blip_target_category = ds_utils.meta_class
            logging.info(f"Using blip_diffusion with src_category: {blip_src_category}, target_category: {blip_target_category}")

        if BASE_MODEL == "blip_diffusion-edit":
            blip_edit_model = lavis_edit_model()
        else:
            pipe = init_pipeline(base_model, controlnet, SDEdit, use_compile=True if not DEBUG else False).to(device, torch.float16)       
        generator = torch.manual_seed(SEED)
        logging.info("Model loaded")

        if PROMPT_TYPE == "captions":  # on a image basis
            blip_captions_dict = blip_utils.read_captions_from_json(blip_captions)
            logging.info(f"Read {len(blip_captions_dict)} captions from {blip_captions}")

        elif PROMPT_TYPE == "txt2sentence":
            prompts = blip_utils.read_prompts_from_json(prompts_file, dataset_name=DATASET, per_class=False)
            prompts = [prompt[:MAX_PROMPT_LENGTH] for prompt in prompts]
            logging.info(f"Read {len(prompts)} prompts from {prompts_file}")

        elif PROMPT_TYPE == "txt2sentence-per_class":
            class_to_prompts_dict = blip_utils.read_prompts_from_json(prompts_file, dataset_name=DATASET, per_class=True)
            for key, value in class_to_prompts_dict.items():
                class_to_prompts_dict[key] = [prompt[:MAX_PROMPT_LENGTH] for prompt in value]  # truncate to 100 chars 
            logging.info(f"Read prompts json with {len(class_to_prompts_dict[key])} prompts for each class. from {prompts_file}")

        elif PROMPT_TYPE == "gpt-meta_class":
            with open(prompts_file, "r") as f:
                prompts = f.readlines()
            prompts = [prompt.strip()[:MAX_PROMPT_LENGTH] for prompt in prompts]
            logging.info(f"Read {len(prompts)} prompts from {prompts_file}, for meta class {DATASET}")

        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        pbar = tqdm(total=len(original_images_paths))
        if DEBUG:
            if SPECIFIC_FILE_STRs:
                original_images_paths = [path for path in original_images_paths if any([specific_str in path for specific_str in SPECIFIC_FILE_STRs])]
            else:
                original_images_paths = original_images_paths[:4]

        for index, source_image_path in enumerate(original_images_paths):
            image_stem = Path(source_image_path).stem
            logging.info(f"Generating from {image_stem}. Index: {index}/{len(original_images_paths)}")
            
            if PROMPT_TYPE == "captions":
                prompts = [blip_captions_dict[source_image_path]["caption"]] * NUM_PER_IMAGE
                prompts = [prompt[:MAX_PROMPT_LENGTH] for prompt in prompts]  # truncate to MAX_PROMPT_LENGTH chars (sometimes BLIP captions are too long- bug in BLIP)
            
            elif PROMPT_TYPE == "txt2sentence-per_class":
                image_classes_dict_key_to_use = Path(source_image_path).stem if DATASET in ["planes", "cars", "planes_biased"] else source_image_path
                prompts = class_to_prompts_dict[image_classes_dict[image_classes_dict_key_to_use]]

            elif PROMPT_TYPE == "gpt-meta_class":
                pass # nothing special to do here

            orig_img = load_image(source_image_path)
            orig_img = utils.resize_image(np.array(orig_img), RESOLUTION)
            orig_img = Image.fromarray(orig_img)
            original_orig_img = orig_img.copy()  # in blip-diffusion orig_img is used as the reference image, so we need to keep it
            # save image with "_source" in the name
            source_image_output_path = os.path.join(output_folder, f"{image_stem[:MAX_FILENAME_LENGTH]}_source.png")
            orig_img.save(source_image_output_path)
            
            prompts = [prompt[:-1] if prompt[-1] == "." else prompt for prompt in prompts]  # remove the "." at the end of the prompt, if exists
            
            sampled_prompts = np.random.choice(prompts, NUM_PER_IMAGE)

            control_image = None
            for i, prompt in enumerate(sampled_prompts):
                if DATASET == "compcars-parts":   # need to add the car part to the prompt
                    part = source_image_path.split("/")[-2]
                    basic_prompt = ds_utils.get_basic_prompt(part=part)
                    prompt = f"{basic_prompt} {prompt}"

                if USE_ARTISTIC_PROMPTS and ((i % 2 == 0 and ARTISTIC_PROMPTS_PROB == 0.5) or (random.random() < ARTISTIC_PROMPTS_PROB and ARTISTIC_PROMPTS_PROB != 0.5)):
                    # translation to the if: to make it take 1 art prompt when using 2 (or any even number) prompts per image and art_prob = 0.5, i make it take the art prompt every 2nd prompt
                    # if it's not 0.5, then i take as usual with the prob
                    prompt = f"{prompt}, {np.random.choice(ARTISTIC_PROMPTS)}"
                
                elif USE_CAMERA_VARIATIONS_PROMPTS and random.random() < CAMERA_VAIRATIONS_PROB:  # don't combine with artistic prompts. This causes the real proportion of camera variations prompts to be lower than the prob if artistic prompts are used too
                    prompt = f"{prompt}, {np.random.choice(IMAGE_VARIATIONS_PROMPTS)} photo"

                if PROMPT_WITH_SUB_CLASS:
                    if DATASET in ["planes", "planes_biased"]:
                        subclass_str = image_classes_dict[image_stem]  # e.g. "Boing 707-320" 
                        # prompt = f"a {subclass_str} plane {prompt}"
                        prompt = prompt.replace("airplane", f"{subclass_str} airplane")

                    elif DATASET == "cars":
                        subclass_str = image_classes_dict[image_stem] 
                        # prompt = f"a {subclass_str} car {prompt}"
                        prompt = prompt.replace("car", f"{subclass_str} car")

                    elif DATASET == "dtd":
                        subclass_str = image_classes_dict[source_image_path]  
                        prompt = f"{prompt} with a {subclass_str} texture"

                    elif DATASET == "compcars":
                        subclass_str = image_classes_dict[source_image_path]
                        prompt = prompt.replace("car", f"{subclass_str} car")
                    
                    elif DATASET == "compcars-parts":
                        subclass_str = image_classes_dict[source_image_path]
                        prompt = prompt.replace("car", f"{subclass_str} car")
                        
                    elif DATASET == "cub":
                        subclass_str = image_classes_dict[source_image_path]
                        prompt = prompt.replace("bird", f"{subclass_str} bird")

                    else:
                        raise NotImplementedError

                output_path = Path(output_folder) / f"{image_stem[:MAX_FILENAME_LENGTH]}_prompt_{prompt.replace('/', '-')}_{i}.png"   # the {i} is for the case of multiple same prompts per image
                if output_path.exists():
                    logging.info(f"Skipping {output_path} as it already exists")
                    continue
                
                seed_for_transform = index * NUM_PER_IMAGE + i
                if CONTROLNET:
                    if CONTROLNET == "canny":
                        control_image = utils.generate_canny(orig_img, low_threshold=LOW_THRESHOLD_CANNY, high_threshold=HIGH_THRESHOLD_CANNY, image_resolution=RESOLUTION)
                    elif CONTROLNET == "hed":
                        control_image = hed_detector(orig_img)

                    if index < 10 and i == 0:
                        control_image.save(f"{output_folder}/{image_stem[:MAX_FILENAME_LENGTH]}_control.png")

                if "blip_diffusion" in base_model:
                    if STYLE_IMG_FROM_DIFF_IMG:
                        diff_img_path_to_use = random.choice(ds_utils.get_image_path_with_same_class(source_image_path))
                        diff_img = load_image(diff_img_path_to_use)
                        diff_img = utils.resize_image(np.array(diff_img), RESOLUTION)
                        diff_img = Image.fromarray(diff_img)
                        if not BASE_MODEL == "blip_diffusion-edit":
                            orig_img = diff_img
                        # save it with "_subject" in the name
                        subject_image_output_path = os.path.join(output_folder, f"{image_stem[:MAX_FILENAME_LENGTH]}_subject_{i}.png")
                        diff_img.save(subject_image_output_path)
                    else:
                        diff_img = orig_img

                if BASE_MODEL == "blip_diffusion-edit":
                    src_subject = blip_src_category
                    cond_subject = tgt_subject = blip_target_category
                    # basically all three of them are the same, because we want to generate the same category as the source image (usually it's the meta class)
                    output = blip_edit_model.pass_through_lavis_edit(orig_img, diff_img, SEED, prompt, cond_subject, src_subject, tgt_subject, GUIDANCE_SCALE, NUM_INFERENCE_STEPS)
                else:
                    output = pass_thorugh_pipe(base_model, pipe, prompt, orig_img, SDEdit, SDEDIT_STRENGTH, NUM_INFERENCE_STEPS, generator, GUIDANCE_SCALE, 
                                            CONTROLNET_CONDITIONING_SCALE, NEGATIVE_PROMPT, control_image=control_image, 
                                            blip_src_category=blip_src_category, blip_target_category=blip_target_category)
                
                orig_img = original_orig_img  # reset the orig_img to the original one, in case it was changed in the blip-diffusion part

                output.save(output_path)
            
            pbar.update(1)
            # print eta
            # if index % 100 == 0:
            #     elapsed_time = tqdm.format_dict['elapsed']
            #     total_images = len(original_images_paths)
            #     eta = elapsed_time / (index + 1) * (total_images - index - 1)
            #     tqdm.set_postfix_str(f"ETA: {eta}")
            #     logging.info(f"ETA: {eta}")
                

        # done, delete pipe and empty cache
        pbar.close()
        logging.info("Done Generating")
        del pipe
        torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
        sys.exit(0)
    
    # except torch cuda OOM error
    except RuntimeError as e:
        logging.exception(e)
        traceback.print_exc()
        num_errors += 1
        logging.info(f"OutOfMemoryError, path: {source_image_path}. num errors: {num_errors}")
        if num_errors > 20:
            logging.info("Too many OOM errors, exiting")
            sys.exit(0)

    except Exception as e:
        logging.exception(e)
        raise e
    

if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=0 nohup sleep 10m; python run_aug/run_aug.py > run_aug/run_aug.log 2>&1 &
    CUDA_VISIBLE_DEVICES=0 nohup python run_aug/run_aug.py > run_aug/run_aug.log 2>&1 &
    CUDA_VISIBLE_DEVICES=0 python run_aug/run_aug.py
    """
    DEBUG = 0  # wont use pytorch 2.0 compile (it's slow) and will use 4 paths only
    SPECIFIC_FILE_STRs = None # ["0680474", "1376795", "0218850", "1538836", "1393334", "1017972", "0989702", "2057958", "0913940"]  # can be sub str/sub path. relevant only if DEBUG is True
    ############################## generation params ##############################
    version = "v1"  # version of the run, you can add smth to help you remember what you did

    # dataset generation params:
    DATASET = "planes"  
    BASE_MODEL = "sd_v1.5" if DATASET == "planes" else "blip_diffusion"  # out of "sd_v1.5", "sd_v2.1", "sd_xl", "sd_xl-turbo", "blip_diffusion"
    CONTROLNET = "canny"  # "canny"  # out of None, "canny", "hed"
    SDEDIT = 0  # if True, will do IMG2IMG edit. LECF is from scratch, ALIA is 0.5
    NUM_PER_IMAGE = 2 # number of augmentations to generate per image
    SEED = 1
    
    # prompts params:
    PROMPT_TYPE = "gpt-meta_class"  # out of "txt2sentence", "txt2sentence-per_class", "captions", "gpt-meta_class", "ALIA"
    PROMPT_WITH_SUB_CLASS = True  # the prompt will contain the sub class of the image (e.g. "a Boing 707-320 plane ...")
    USE_ARTISTIC_PROMPTS = True if BASE_MODEL == "sd_v1.5" else False  # feel free to change this. if True, will use artistic prompts (e.g. "a painting of a ...") at the end of used prompts
    ARTISTIC_PROMPTS_PROB = 0.5  # the probability to use artistic prompts
    USE_CAMERA_VARIATIONS_PROMPTS = False  # if True, will use camera variations prompts (e.g. "High-Speed, Lens Flare) at the end of used prompts. will not combine artistic prompts with this
    CAMERA_VAIRATIONS_PROB = 0.5  # the probability to use camera variations prompts

    # SD params:
    RESOLUTION = 512  # since SD XL is trained on higher res images, recommended res is 768-1024 for SD XL. read: https://huggingface.co/docs/diffusers/v0.19.2/api/pipelines/stable_diffusion/stable_diffusion_xl. SD XL turbo is trained on 512x512
    GUIDANCE_SCALE = 7.5 # 5-9 is usually used. for SD XL turbo, 0 is used (hardcoded in the code)
    NUM_INFERENCE_STEPS = 30  # 20-50 is usually used. for SD XL turbo, 1-4 steps are used
    # SDEdit (img2img) params
    SDEDIT_STRENGTH = 0.85  # range is 0-1. relevant only if SDEdit is true. for ALIA: 0.5. For RG(LECF): 0.15.

    # ControlNet params:
    LOW_THRESHOLD_CANNY = 120  # relevant only for canny model. 120 is default
    HIGH_THRESHOLD_CANNY = 200 # relevant only for canny model. 200 is default
    CONTROLNET_CONDITIONING_SCALE = 0.75  # The higher, the more the controlnet will affect the output (e.g. edges will be perserved). for SD XL turbo not more than 0.75. I noticed that generally, if using a high cond scale, should not use low num of inf steps

    # BLIP diffusion params:
    STYLE_IMG_FROM_DIFF_IMG = True # if True, will use an image from the same class as the style image (and not the edited image)

    ############################## json creation params ##############################
    LPIPS_MIN = None
    LPIPS_MAX = None
    CLIP_FILTERING_TYPE = None  # out of "per_class" only 
    SEMANTIC_FILTERING = 1
    MODEL_CONFIDENCE_BASED_FILTERING = 1
    ALIA_CONF_FILTERING = 0

    ############################## code start ##############################

    if BASE_MODEL == "sd_xl-turbo":
        GUIDANCE_SCALE = 0
        NUM_INFERENCE_STEPS = 2
        NEGATIVE_PROMPT = None

    if BASE_MODEL == "blip_diffusion-edit":
        NUM_INFERENCE_STEPS = 50  # has to be 50 for p2p edit
        
    if SDEDIT:
        assert NUM_INFERENCE_STEPS * SDEDIT_STRENGTH >= 1

    if DEBUG:
        version = f"{version}_DEBUG"

    assert PROMPT_TYPE in ["txt2sentence", "txt2sentence-per_class", "captions", "gpt-meta_class", "ALIA"]
    assert DATASET in dataset_utils.DATASETS_SUPPORTED
    assert BASE_MODEL in BASE_MODEL_DICT.keys()
    assert CONTROLNET in CONTROLNET_DICT_SD.keys() or CONTROLNET in CONTROLNET_DICT_SD_XL.keys() or CONTROLNET is None
    assert NUM_PER_IMAGE > 0

    utils.set_seed(SEED)
    ds_utils: dataset_utils.BaseUtils = dataset_utils.DS_UTILS_DICT[DATASET]()

    if DATASET == "planes":
        blip_captions = Path("").parent / "prompts_engineering/captions/planes_captions.json"
        if PROMPT_TYPE == "txt2sentence-per_class":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_30_planes_all_classes_True.json"
        elif PROMPT_TYPE == "txt2sentence":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_200_planes_all_classes_False.json"
        elif PROMPT_TYPE == "gpt-meta_class":
            prompts_file = Path("").parent / "prompts_engineering/gpt_prompts" / f"{DATASET}-100-gpt_v1.txt"


    elif DATASET == "cars":
        blip_captions = Path("").parent / "prompts_engineering/captions/cars_captions.json"
        if PROMPT_TYPE == "txt2sentence-per_class":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_30_cars_all_classes_True.json"
        elif PROMPT_TYPE == "txt2sentence":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_200_cars_all_classes_False.json"
        elif PROMPT_TYPE == "gpt-meta_class":
            prompts_file = Path("").parent / "prompts_engineering/gpt_prompts" / f"{DATASET}-100-gpt_v1.txt"


    elif DATASET == "dtd":
        blip_captions = Path("").parent / "prompts_engineering/captions/dtd_captions.json"
        if PROMPT_TYPE == "txt2sentence-per_class":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_30_dtd_all_classes_True.json"
        elif PROMPT_TYPE == "txt2sentence":
            raise NotImplementedError
        elif PROMPT_TYPE == "gpt-meta_class":
            raise NotImplementedError


    elif DATASET == "compcars":
        blip_captions = Path("").parent / "prompts_engineering/captions/cars2_captions.json"
        if PROMPT_TYPE == "txt2sentence-per_class":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_30_compcars_all_classes_True.json"
        elif PROMPT_TYPE == "txt2sentence":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_200_compcars_all_classes_False.json"


    elif DATASET == "compcars-parts":
        blip_captions = Path("").parent / "prompts_engineering/captions/compcars-parts_captions.json"
        if PROMPT_TYPE == "txt2sentence-per_class":
            prompts_file = None  # output doesnt make much sense for this dataset
        elif PROMPT_TYPE == "txt2sentence":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_200_cars_all_classes_False.json"
        elif PROMPT_TYPE == "gpt-meta_class":
            prompts_file = Path("").parent / "prompts_engineering/gpt_prompts" / f"cars-100-gpt_v1.txt"


    elif DATASET == "cub":
        # blip_captions = Path("").parent / "prompts_engineering/captions/cub_captions.json"
        if PROMPT_TYPE == "txt2sentence-per_class":
            prompts_file = None
        elif PROMPT_TYPE == "txt2sentence":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_200_cub_all_classes_False.json"
        elif PROMPT_TYPE == "gpt-meta_class":
            prompts_file = Path("").parent / "prompts_engineering/gpt_prompts" / f"{DATASET}-100-gpt_v1.txt"


    elif DATASET == "planes_biased":
        blip_captions = None
        if PROMPT_TYPE == "txt2sentence-per_class":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_30_planes_all_classes_True.json"
        elif PROMPT_TYPE == "txt2sentence":
            prompts_file = Path("").parent / "prompts_engineering/txt2sentences_prompts/LE_200_planes_all_classes_False.json"
        elif PROMPT_TYPE == "gpt-meta_class":
            prompts_file = Path("").parent / "prompts_engineering/gpt_prompts" / f"planes-100-gpt_v1.txt"

    else:
        raise NotImplementedError
    
    if PROMPT_TYPE == "ALIA":
        prompts_file = Path("").parent / "prompts_engineering/ALIA_prompts/gpt_output/{DATASET}_prompts.txt"

    prompt_str = PROMPT_TYPE
    if PROMPT_WITH_SUB_CLASS:
        prompt_str += "_prompt_w_sub_class"
    if USE_ARTISTIC_PROMPTS:
        prompt_str += f"_artistic_prompts_p_{ARTISTIC_PROMPTS_PROB}"
    if USE_CAMERA_VARIATIONS_PROMPTS:
        prompt_str += f"_camera_variations_p_{CAMERA_VAIRATIONS_PROB}"
    if "blip_diffusion" in BASE_MODEL and STYLE_IMG_FROM_DIFF_IMG:
        prompt_str += "_style_img_from_diff_img"
    
    base_model_folder = f"regular/{BASE_MODEL}"
    if SDEDIT:
        base_model_folder += f"-SDEdit_strength_{SDEDIT_STRENGTH}"

    last_folder_name = f"{version}-res_{RESOLUTION}-num_{NUM_PER_IMAGE}-gs_{GUIDANCE_SCALE}-num_inf_steps_{NUM_INFERENCE_STEPS}"
    if CONTROLNET:
        base_model_folder = base_model_folder.replace("regular/", "controlnet/")
        last_folder_name += f"_controlnet_scale_{CONTROLNET_CONDITIONING_SCALE}"
        if CONTROLNET == "canny":
            last_folder_name += f"_low_{LOW_THRESHOLD_CANNY}_high_{HIGH_THRESHOLD_CANNY}"
        elif CONTROLNET == "hed":
            pass


    output_folder = f"{ds_utils.root_path}/aug_data/{base_model_folder}/{CONTROLNET}/{prompt_str}_seed_{SEED}/images"

    logdir = utils.init_logging(str(Path(output_folder).parent))
    logging.info(f"Log folder: {logdir}")
    ds_utils: dataset_utils.BaseUtils = dataset_utils.DS_UTILS_DICT[DATASET](print_func=logging.info)

    image_classes_dict = ds_utils.get_image_stem_to_class_str_dict() if DATASET in ["planes", "cars", "planes_biased"] else ds_utils.get_image_path_to_class_str_dict()
    logging.info(f"Output folder: {output_folder}")

    # get json path
    aug_json_path = utils.get_aug_json_path(
        augmented_image_folder_path=output_folder,
        lpips_min=LPIPS_MIN,
        lpips_max=LPIPS_MAX,
        clip_filtering=CLIP_FILTERING_TYPE,
        semantic_filtering=SEMANTIC_FILTERING,
        model_confidence_based_filtering=MODEL_CONFIDENCE_BASED_FILTERING,
        alia_conf_filtering=ALIA_CONF_FILTERING,
    )
    logging.info(f"Augmented json path will be at: \n{aug_json_path}")

    main(BASE_MODEL, CONTROLNET, SDEDIT)

    ############################## create json  ##############################

    if SPECIFIC_FILE_STRs and DEBUG:
        logging.info("Exiting because SPECIFIC_FILE_STRs is not empty")
        exit(0)

    json_path = utils.create_json_of_image_name_to_augmented_images_paths(
        DATASET,
        augmented_image_folder_path=output_folder,
        lpips_min=LPIPS_MIN,
        lpips_max=LPIPS_MAX,
        resize=(256, 256),
        clip_filtering=CLIP_FILTERING_TYPE,
        clip_filtering_discount=1,  # 1 is no discount
        semantic_filtering=SEMANTIC_FILTERING,
        model_confidence_based_filtering=MODEL_CONFIDENCE_BASED_FILTERING,
        init_log=False,
        alia_conf_filtering=ALIA_CONF_FILTERING,
    )
