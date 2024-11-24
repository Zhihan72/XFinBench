#!/bin/bash

gpu_id=1
sys_msg="On"
dataset="FinBench"
task="bool"
model="Meta-Llama-3.1-8B-Instruct"
reason_type="CoT"
retri_type="free"
retriever="bm25"
top_k_retr=3
max_token=1024

CUDA_VISIBLE_DEVICES=$gpu_id python /your_path/experiment/Eval_Model.py \
    --dataset $dataset\
    --task $task\
    --model $model\
    --reason_type $reason_type\
    --sys_msg $sys_msg\
    --retri_type $retri_type\
    --retriever $retriever\
    --top_k_retr $top_k_retr\
    --max_token $max_token