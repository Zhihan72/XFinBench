import os
import sys
import csv
import json
import random
import argparse
import base64
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import MllamaForConditionalGeneration, AutoProcessor
from openai import OpenAI
import anthropic
import google.generativeai as genai
from Retriever import build_retriever

os.environ['OPENAI_API_KEY'] = "your_openai_key"
os.environ["ANTHROPIC_API_KEY"] = "your_antropic_key"
os.environ["GEMINI_API_KEY"] = "your_gemini_key"
os.environ["DEEPSEEK_API_TOKEN"] = "your_deepseek_key"

###################### Para setting ######################

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

# Save with excuting date and time
time_ = datetime.now()
current_time = f"{time_.year}{time_.month}{time_.day}_{time_.hour}{time_.minute}"
save_path = f"./experiment/results/{dataset_}/{task_}/{model_}_{reason_type}_{retri_type}_{retriever_}_{str(top_k_retr)}_{max_token_}_{current_time}_sys{sys_msg_bool}.csv"
print(f"\n\nSAVE PATH: {save_path}\n")

##########################################################

############## Load dataset and retriver #################

# QA Data
qa_path = './dataset/test_set.csv'
qa_data = pd.read_csv(qa_path)
qa_data = qa_data[qa_data['task']==task_]
qa_data['model_response'] = ''
qa_data['model_answer'] = ''
qa_data['model_ans_bool'] = 0
qa_data['model_ans_err5_bool'] = 0
qa_data['execute_time'] = ''

# Prompt Template
with open(file=f'./experiment/prompt_template/{task_}_{reason_type}.txt', mode='r', encoding='UTF-8') as fp:
    prompt_ques = fp.read()

if sys_msg_bool==0:
    sys_msg = ''
else:
    with open(file=f'./experiment/prompt_template/Sys_msg/{reason_type}.txt', mode='r', encoding='UTF-8') as fp:
        sys_msg = fp.read()

##########################################################

###################### Load model #########################

device = 'cuda'
if 'gpt' in model_ or 'o1' in model_:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
elif 'claude' in model_:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
elif 'gemini' in model_:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    client = genai.GenerativeModel(model_name=model_)
elif 'deepseek' in model_:
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_TOKEN"), base_url="https://api.deepseek.com")
elif 'Llama' in model_ and 'Vision' in model_:
    model_path = f"./models/{model_}"
    processor = AutoProcessor.from_pretrained(model_path)
    model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto",)
elif 'Llama' in model_ or 'Mixtral' in model_:
    model_path = f"./models/{model_}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    terminators = [ tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


##########################################################

###################### Functions #########################

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_model(sys_msg, msg, model_, image_pt=''):

    # Prepare message
    if 'o1' in model_:
        if sys_msg_bool==1:
            msg = sys_msg + '\n' + msg
        else:
            msg = msg
        messages = [{"role": "user", "content": msg}]
    elif 'gpt' in model_:
        if sys_msg_bool==1:
            messages = [{"role": "system", "content": sys_msg},]
        else:
            messages = []
        if len(image_pt)==0:
            messages.append({"role": "user", "content": msg})
        else:
            base64_image = encode_image(image_pt)
            messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",}})
            messages.append({"role": "user", "content": msg})
    elif 'claude' in model_:
        if len(image_pt)==0:
            messages = [
                {"role": "user",
                "content": [{"type": "text","text": msg}]}
            ]
        else:
            base64_image = encode_image(image_pt)
            img_type = 'image/png' if '.png' in image_pt else 'image/jpeg'
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image","source": {"type": "base64", "media_type": img_type, "data": base64_image,}},
                        {"type": "text","text": msg}
                        ]
                }
            ]
    elif 'gemini' in model_:
        if sys_msg_bool==1:
            msg = sys_msg + '\n' + msg
        else:
            msg = msg
        if len(image_pt)==0:
            messages = msg
        else:
            image = Image.open(image_pt)
            messages = [msg, image]
    elif 'Llama' in model_ and 'Vision' in model_:
        if sys_msg_bool==1:
            msg = sys_msg + '\n' + msg
        else:
            msg = msg
        if len(image_pt)==0:
            image_pt = f"./dataset/figures/blank_image.png"
        image = Image.open(image_pt)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": msg}
            ]}
        ]
    else:
        if sys_msg_bool==1:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": msg},
            ]
        else:
            messages = [
                {"role": "user", "content": msg},
            ]
    
    # Ask the model to answer questions
    if 'gpt' in model_ or 'o1' in model_:
        completion = client.chat.completions.create(
            model= model_,
            messages=messages,
            max_tokens=max_token_,
        )
        reply = completion.choices[0].message.content
        num_token = str(completion.usage.completion_tokens) + ';' + str(completion.usage.prompt_tokens)
        return reply, num_token
    elif 'deepseek' in model_:
        completion = client.chat.completions.create(
            model= model_,
            messages=messages,
            stream=False,
            max_tokens=max_token_,
        )
        reply = completion.choices[0].message.content
        num_token = str(completion.usage.completion_tokens) + ';' + str(completion.usage.prompt_tokens)
        return reply, num_token
    elif 'claude' in model_:
        if sys_msg_bool==1:
            completion = client.messages.create(
                model=model_,
                system=sys_msg,
                messages=messages,
                max_tokens=max_token_,
            )
        else:
            completion = client.messages.create(
                model=model_,
                messages=messages,
                max_tokens=max_token_,
            )
        reply = completion.content[0].text
        num_token = str(completion.usage.output_tokens) + ';' + str(completion.usage.input_tokens)
        return reply, num_token
    elif 'gemini' in model_:
        completion = client.generate_content(messages, generation_config=genai.types.GenerationConfig(max_output_tokens= max_token_,),)
        reply = completion.text
        num_token = str(completion.usage_metadata.candidates_token_count) + ';' + str(completion.usage_metadata.prompt_token_count)
        return reply, num_token
    elif 'Llama' in model_ and 'Vision' in model_:
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_token_)
        output_ids = outputs[0]
        reply = processor.decode(output_ids)
        num_token = len(output_ids)
        return reply, num_token
    elif 'Llama' in model_ or 'Mixtral' in model_:
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
            ).to(device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_token_,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            )
        output_ids = outputs[0][input_ids.shape[-1]:]
        reply = tokenizer.decode(output_ids, skip_special_tokens=True)
        num_token = len(output_ids)
        return reply, num_token

def get_calcu_error_bool(corr_ans, model_ans):
    try:
        model_ans = float(model_ans)
        if model_ans==0:
            return 0
    except Exception as e:
        print(f"Error in get_calcu_error_bool() when tranferring model_ans to float() type! {e}")
        return 0
    try:
        corr_ans = float(corr_ans)
    except Exception as e:
        print(f"Error in get_calcu_error_bool() when tranferring corr_ans to float() type! {e}")
        return 0
    if abs((model_ans-corr_ans)/model_ans) < 0.005:
        return 1
    elif abs((model_ans*0.01-corr_ans)/(model_ans*0.01)) < 0.005 or abs((model_ans-corr_ans*0.01)/(model_ans)) < 0.005:
        return 1
    else:
        return 0

def get_calcu_bool(corr_ans, model_ans):
    try:
        model_ans = float(model_ans)
    except Exception as e:
        print(f"Error in get_calcu_bool() when tranferring model_ans to float() type! {e}")
        return 0
    try:
        corr_ans = float(corr_ans)
    except Exception as e:
        print(f"Error in get_calcu_bool() when tranferring corr_ans to float() type! {e}")
        return 0
    if model_ans==corr_ans:
        return 1
    else:
        return 0

def resp2ans(task, resp):
    if type(resp)!=str:
        return ''
    end_sent = 'Therefore, my answer is'
    if end_sent not in resp:
        return ''
    if  task=='bool':
        resp = resp.split(end_sent)[-1].split('.')[0]
        if 'TRUE' in resp or 'true' in resp or 'True' in resp or 'correct' in resp:
            return 1
        elif 'FALSE' in resp or 'false' in resp or 'False' in resp or 'incorrect' in resp:
            return 0
        else:
            return ''
    elif task=='mcq':
        resp = resp.split(end_sent)[-1].split('.')[0]
        if 'A' in resp:
            return 'A'
        elif 'B' in resp:
            return 'B'
        elif 'C' in resp:
            return 'C'
        elif 'D' in resp:
            return 'D'
        else:
            return ''
    elif task=='calcu':
        resp = resp.split(end_sent)[-1].strip()
        ans_ = resp.split('[')[-1].split(']')[0].strip()
        ans_1 = ''
        for chr_ in list(ans_):
            if ord(chr_) > 57 or ord(chr_) < 45:
                continue
            ans_1 += chr_
        try:
            ans_2 = float(ans_1)
            return ans_2
        except:
            return ''

##########################################################

###################### Experiment #########################

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
        response_, num_token_ = call_model(sys_msg, prompt_idx, model_)
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