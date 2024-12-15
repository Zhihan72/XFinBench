import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import MllamaForConditionalGeneration, AutoProcessor
from openai import OpenAI
import anthropic
import google.generativeai as genai
import base64
from PIL import Image

class build_model:
    def __init__(self, model_name: str,):
        self.project_path = os.environ["PROJECT_PATH"]
        if 'gpt' in model_name or 'o1' in model_name:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
        elif 'claude' in model_name:
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        elif 'gemini' in model_name:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.client = genai.GenerativeModel(model_name=model_name)
        elif 'deepseek' in model_name:
            self.client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_TOKEN"), base_url="https://api.deepseek.com")
        elif 'Llama' in model_name and 'Vision' in model_name:
            model_path = f"{self.project_path}/model_ckpt/{model_name}"
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto",)
        elif 'Llama' in model_name or 'Mixtral' in model_name:
            model_path = f"{self.project_path}/model_ckpt/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
            self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_model_response(self, sys_msg, msg, model_name, image_pt='', sys_msg_bool=1, max_token_=1024):
        # Prepare message
        if 'o1' in model_name:
            if sys_msg_bool==1:
                msg = sys_msg + '\n' + msg
            else:
                msg = msg
            messages = [{"role": "user", "content": msg}]
        elif 'gpt' in model_name:
            if sys_msg_bool==1:
                messages = [{"role": "system", "content": sys_msg},]
            else:
                messages = []
            if len(image_pt)==0:
                messages.append({"role": "user", "content": msg})
            else:
                base64_image = self.encode_image(image_pt)
                messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",}})
                messages.append({"role": "user", "content": msg})
        elif 'claude' in model_name:
            if len(image_pt)==0:
                messages = [
                    {"role": "user",
                    "content": [{"type": "text","text": msg}]}
                ]
            else:
                base64_image = self.encode_image(image_pt)
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
        elif 'gemini' in model_name:
            if sys_msg_bool==1:
                msg = sys_msg + '\n' + msg
            else:
                msg = msg
            if len(image_pt)==0:
                messages = msg
            else:
                image = Image.open(image_pt)
                messages = [msg, image]
        elif 'Llama' in model_name and 'Vision' in model_name:
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
        if 'gpt' in model_name or 'o1' in model_name:
            completion = self.client.chat.completions.create(
                model= model_name,
                messages=messages,
                max_tokens=max_token_,
            )
            reply = completion.choices[0].message.content
            num_token = str(completion.usage.completion_tokens) + ';' + str(completion.usage.prompt_tokens)
            return reply, num_token
        elif 'deepseek' in model_name:
            completion = self.client.chat.completions.create(
                model= model_name,
                messages=messages,
                stream=False,
                max_tokens=max_token_,
            )
            reply = completion.choices[0].message.content
            num_token = str(completion.usage.completion_tokens) + ';' + str(completion.usage.prompt_tokens)
            return reply, num_token
        elif 'claude' in model_name:
            if sys_msg_bool==1:
                completion = self.client.messages.create(
                    model=model_name,
                    system=sys_msg,
                    messages=messages,
                    max_tokens=max_token_,
                )
            else:
                completion = self.client.messages.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_token_,
                )
            reply = completion.content[0].text
            num_token = str(completion.usage.output_tokens) + ';' + str(completion.usage.input_tokens)
            return reply, num_token
        elif 'gemini' in model_name:
            completion = self.client.generate_content(messages, generation_config=genai.types.GenerationConfig(max_output_tokens= max_token_,),)
            reply = completion.text
            num_token = str(completion.usage_metadata.candidates_token_count) + ';' + str(completion.usage_metadata.prompt_token_count)
            return reply, num_token
        elif 'Llama' in model_name and 'Vision' in model_name:
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(image, input_text, return_tensors="pt").to(model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_token_)
            output_ids = outputs[0]
            reply = self.processor.decode(output_ids)
            num_token = len(output_ids)
            return reply, num_token
        elif 'Llama' in model_name or 'Mixtral' in model_name:
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
                ).cuda()
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_token_,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                )
            output_ids = outputs[0][input_ids.shape[-1]:]
            reply = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            num_token = len(output_ids)
            return reply, num_token