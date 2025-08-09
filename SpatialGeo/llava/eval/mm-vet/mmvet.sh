#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path ./llava-v1.5-7-lora-merged \
    --question-file ./benchmark/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./benchmark/eval/mm-vet/images \
    --answers-file ./llava/eval/mm-vet/answers/llava-v1.5-7b-osd.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./llava/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./llava/eval/mm-vet/answers/llava-v1.5-7b-osd.jsonl \
    --dst ./llava/eval/mm-vet/results/llava-v1.5-7b-osd.json
