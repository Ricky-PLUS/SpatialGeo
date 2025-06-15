import argparse
import copy
import json
import math
import os
import re
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def eval_model(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device=args.device)

    with open(args.annotation_file) as f:
        questions = json.load(f)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    for line in tqdm(questions, total=len(questions)):
        question_id = line["id"]
        image_file = line["file_path"]
        text_question = line["text_q"]
        qa_info = line["qa_info"]

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

        conv = conv_templates[args.conv_mode].copy()
        
        if qa_info["category"] == "direction":
            conv.system = conv.system + "The answer should be presented in this format {# o'clock}, for example, {3 o'clock} or {12 o'clock}."
        elif qa_info["type"] == "quantitative" :
            conv.system = conv.system + "The answer should be presented in this format {# meters or feets or inches}, for example, {1.23 meters} or {0.56 feets} or {0.65 inches}."
        else:
            conv.system = conv.system + "The answer should be presented in this format {# big or small or behind or front or left or right or tall or short or wide or thin or below or above}, for example,{A is bigger than B}."
            
        conversations = line["conversations"]

        num_question = len(conversations) // 2
        for i in range(num_question):
            question = conversations[i * 2]["value"]
            
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            stop_str = (
                conv_templates[args.conv_mode].sep
                if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
                else conv_templates[args.conv_mode].sep2
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=128,
                    use_cache=True,
                )

            outputs = outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            ans_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "image": image_file,
                        "question": text_question,
                        "pred": outputs,
                        "gt": conversations[i * 2 + 1]["value"],
                        "model_id": model_name,
                        "qa_info": qa_info,
                    }
                )
                + "\n"
            )
            ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./benchmark/rgptbenchmark")
    parser.add_argument("--annotation-file", type=str, default="./benchmark/rgptbenchmark/rgptBenchmark2.json")
    parser.add_argument("--answers-file", type=str, default="./benchmark/rgptbenchmark/answers/4projllavaanswer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
