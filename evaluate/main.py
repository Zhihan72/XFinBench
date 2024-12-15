import os
import sys
import csv
import json
import random
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from load_retriever import build_retriever
from load_models import build_model
from eval_metrics import get_calcu_error_bool, get_calcu_bool, resp2ans

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='FinBench', type=str, help="FinBench, BizBench, KnowledgeFMATH")
parser.add_argument("--task", default='bool', type=str, help="bool, mcq, calcu")
parser.add_argument("--model", default='gpt-4o', type=str, help="gpt model or local models")
parser.add_argument("--reason_type", default='CoT', type=str, help="CoT, PoT, DA")
parser.add_argument("--sys_msg", default='On', type=str, help="On, Off")
parser.add_argument("--retri_type", default='free', type=str, help="no, free, gold")
parser.add_argument("--retriever", default='bm25', type=str, help="bm25, ada")
parser.add_argument("--top_k_retr", default=3, type=int, help="normally set to 3")
parser.add_argument("--max_token", default=1024, type=int, help="max number of output tokens")
args = parser.parse_args()
arg_dict=args.__dict__

dataset_ = arg_dict['dataset']
task_ = arg_dict['task'] 
model_ = arg_dict['model']
reason_type = arg_dict['reason_type']
sys_msg_bool = 1 if arg_dict['sys_msg']=='On' else 0
retri_type = arg_dict['retri_type']
if retri_type=='free':
    retriever_ = arg_dict['retriever']
else:
    retriever_ = ''
top_k_retr = arg_dict['top_k_retr']
max_token_ = arg_dict['max_token']
project_path = os.environ["PROJECT_PATH"]

# Load Model and Retriever
model = build_model(model_)
retriever = build_retriever(retri_type, top_k_retr)

# Load QA Data
qa_data = pd.read_csv(f'{project_path}/dataset/test_set.csv')
qa_data = qa_data[qa_data['task']==task_]
qa_data['model_response'] = ''
qa_data['model_answer'] = ''
qa_data['model_ans_bool'] = 0
qa_data['model_ans_err5_bool'] = 0
qa_data['execute_time'] = ''

# Load Prompt Template
with open(file=f'{project_path}/evaluate/prompt_template/{task_}_{reason_type}.txt', mode='r', encoding='UTF-8') as fp:
    prompt_ques = fp.read()

if sys_msg_bool==0:
    sys_msg = ''
else:
    with open(file=f'{project_path}/evaluate/prompt_template/{task_}_{reason_type}.txt', mode='r', encoding='UTF-8') as fp:
        sys_msg = fp.read()

# Save with excuting date and time
time_ = datetime.now()
current_time = f"{time_.year}{time_.month}{time_.day}_{time_.hour}{time_.minute}"
save_path = f"{project_path}/results/{dataset_}/{task_}/{model_}_{reason_type}_{retri_type}_{retriever_}_{str(top_k_retr)}_{max_token_}_{current_time}_sys{sys_msg_bool}.csv"
print(f"\n\nSAVE PATH: {save_path}\n")

# Evaluation Start
for idx in qa_data.index.tolist():
    qa_uni_id_ = qa_data.loc[idx]['id']
    question_ = qa_data.loc[idx]['question']
    choi_ = qa_data.loc[idx]['choice']
    ans_ = qa_data.loc[idx]['ground_truth']
    gt_ids = qa_data.loc[idx]['gold_fin_term_id']

    if pd.isnull(qa_data.loc[idx]['figure'])==0:
        fg_nm = qa_data.loc[idx]['figure']
        fg_path = f"./dataset/figures/{fg_nm}"
        if '.png' not in fg_path:
            if 'p'==fg_nm[0] or 's'==fg_nm[0]:
                fg_path = fg_path + '.png'
            else:
                fg_path = fg_path + '.jpg'
    else:
        fg_path = ''

    exce_time = datetime.now()
    qa_data.loc[idx, 'execute_time'] = exce_time

    if 'mcq' in task_:
        prompt_idx = prompt_ques.format(knowledge='',question=question_, choices=choi_)
    else:
        prompt_idx = prompt_ques.format(knowledge='',question=question_)

    try:
        response_, num_token_ = model.get_model_response(sys_msg, prompt_idx, model_, fg_path, sys_msg_bool, max_token_)
    except Exception as e:
        response_ = ""
        num_token_ = 0
        print(f"Error when calling model! {e}")
    
    if reason_type=='CoT' or reason_type=='DA':
        model_ans = resp2ans(task_, response_)
        print(f"model_ans: {model_ans}")
    elif reason_type=='PoT':
        exec_code = response_.split("```python")[-1].split("```")[0] + "\nval_ = solution()"
        if 'scipy' in exec_code:
            exec_code = 'import scipy\n' + exec_code
        if 'math' in exec_code:
            exec_code = 'import math\n' + exec_code
        try:
            exec(exec_code)
            model_ans = locals()['val_']
            # For debug
            if type(model_ans)==str:
                model_ans = 'Therefore, my answer is' + model_ans
                model_ans = resp2ans(task_, model_ans)
        except Exception as e:
            print(f"Error when excuting PoT codes! {e}")
            model_ans = 0
    
    qa_data.loc[idx, 'model_response'] = response_
    if 'calcu' in task_:
        try:
            qa_data.loc[idx, 'model_answer'] = model_ans
            # For debug
            if type(ans_)==str:
                if '-' in ans_:
                    ans_ = float(ans_.replace('-','')) * (-1)
                else:
                    ans_ = float(ans_)
            qa_data.loc[idx, 'model_ans_bool'] = get_calcu_bool(ans_, model_ans)
            qa_data.loc[idx, 'model_ans_err5_bool'] = get_calcu_error_bool(ans_, model_ans)
        except: 
            qa_data.loc[idx, 'model_answer'] = 0
            qa_data.loc[idx, 'model_ans_bool'] = 0
            qa_data.loc[idx, 'model_ans_err5_bool'] = 0          
    else:
        qa_data.loc[idx, 'model_answer'] = model_ans
        qa_data.loc[idx, 'model_ans_bool'] = 1 if ans_==model_ans else 0

qa_data.to_csv(save_path)
if 'calcu' in task_:
    em_bool_list = qa_data['model_ans_bool'].tolist()
    err5_bool_list = qa_data['model_ans_err5_bool'].tolist()
    em_acc = sum(em_bool_list)/len(em_bool_list)
    err5_acc = sum(err5_bool_list)/len(err5_bool_list)
    print(f"\nAcc score:\nem acc: {em_acc}\nerror 0.5% acc: {err5_acc}\n{save_path}\nFinished!\n")
else:
    result_list = qa_data['model_ans_bool'].tolist()
    acc = sum(result_list)/len(result_list)
    print(f"\n\nAcc score: {acc}\n{save_path}\nFinished!\n")