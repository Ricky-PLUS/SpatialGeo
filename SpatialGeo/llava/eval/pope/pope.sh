#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./llava-v1.5-7-lora-merged \
    --question-file ./benchmark/eval/pope/llava_pope_test.jsonl \
    --image-folder ./benchmark/eval/pope/COCO_val2014 \
    --answers-file ./llava/eval/pope/answers/llava-v1.5-7b-osd.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./benchmark/eval/pope/coco \
    --question-file ./benchmark/eval/pope/llava_pope_test.jsonl \
    --result-file ./llava/eval/pope/answers/llava-v1.5-7b-osd.jsonl
