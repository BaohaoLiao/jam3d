#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

MODEL_NAME_OR_PATH=/mnt/nushare2/data/baliao/PLLMs/deepseek/DeepSeek-R1-Distill-Qwen-7B
OUTPUT_DIR=DeepSeek-R1-Distill-Qwen-7B

SPLIT="test"
NUM_TEST_SAMPLE=-1


PROMPT_TYPE="deepseek-r1"
DATA_NAME="aime24,aime25,aimo2,math500_level1,math500_level2,math500_level3,math500_level4,math500_level5"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_dir "./data" \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call 32768 \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.6 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --n_sampling 2 \
    --num_think_chunks 2 \
    --num_answers_per_chunk 2 \
    --max_tokens_per_answer 2048 \
    --answer_temperature 0.6 \
    --answer_top_p 0.95


PROMPT_TYPE="deepseek-r1-choice"
DATA_NAME="gpqa"
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_dir "./data" \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call 32768 \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.6 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --n_sampling 2 \
    --num_think_chunks 2 \
    --num_answers_per_chunk 2 \
    --max_tokens_per_answer 2048 \
    --answer_temperature 0.6 \
    --answer_top_p 0.95