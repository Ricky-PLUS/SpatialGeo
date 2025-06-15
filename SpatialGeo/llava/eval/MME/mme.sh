#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./llava-v1.5-7-lora-merged \
    --question-file ./llava/eval/MME/llava_mme.jsonl \
    --image-folder ./benchmark/eval/MME/MME_Benchmark_release_version \
    --answers-file ./llava/eval/MME/answers/llava-v1.5-7b-osd.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./llava/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-osd

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-osd