import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
import numpy as np
import torch.nn.functional as F


import pandas as pd
from PIL import Image
import os


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



def process_images_(image: torch.Tensor, device):
    if image.dim() == 3:
        image = image.unsqueeze(0)        

    original_height, original_width = image.shape[-2:]
    area = original_height * original_width

    expected_area = 500000

    if expected_area != area:
        expected_width, expected_height = int(original_width * (expected_area / area) ** 0.5), int(original_height * (expected_area / area) ** 0.5)
        image = F.interpolate(image, (expected_height, expected_width), mode="bicubic", align_corners=False, antialias=True)
        
    raw_img_h, raw_img_w = image.shape[-2:]
    patch_h, patch_w = raw_img_h // 14, raw_img_w // 14

    image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_mean = image_mean.to(device)
    image_std = image_std.to(device)

    image = (image - image_mean) / image_std

    image_14 = F.interpolate(image, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True)
    
    return image_14

    

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)


    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    benchmark_dir = os.path.join(args.directory, 'Questions.csv')
    # Load and read the CSV
    df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
    answers_file = os.path.expanduser(args.answers_file)
    # Check if the directory is specified in the path
    if os.path.dirname(answers_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Now open the file
    ans_file = open(answers_file, "w")

    # Loop through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        # Construct the 'prompts' string

        cur_prompt = row['Question'] + " " + row['Options']
        qs = cur_prompt

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Load the corresponding image
        photo_id = index+1
        image_path = os.path.join(args.directory, 'MMVP Images', f"{photo_id}.jpg")

        # # dino(moge)部分的image processor
        # pil_image = Image.open(image_path).convert('RGB')
        # input_array = np.array(pil_image, dtype=np.float32)

        # input_image = torch.tensor(input_array / 255.0, 
        #                         dtype=torch.float32,
        #                         device=model.device).permute(2, 0, 1)  # HWC -> CHW

        # image_tensor = process_images_(input_image, model.device)

        image = Image.open(image_path)
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": photo_id,
                                   "prompt": cur_prompt,
                                   "answer": row["Correct Answer"], 
                                   "response": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/private_data/MyCode/MMVP/LLaVA/MMVP_Model/based-llava-1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--directory", type=str, default="./MMVP")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="eval/answer1.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
