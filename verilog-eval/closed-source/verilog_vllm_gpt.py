# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from verilog_eval.data import write_jsonl, read_problems
import torch
import sys
import os
# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import math
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import re
from copy import deepcopy
from google import genai
from google.genai import types
import time
import certifi
import google.genai.errors
from openai import OpenAI
import httpx
#from huggingface_hub import login
#login(token="hf_XXXXXX")
os.environ["SSL_CERT_FILE"] = certifi.where()
client = OpenAI(api_key="<YOUR API KEY>")
def load_json(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

def batch_generate(batch_ids, batch_prompts, max_retries=20, base_delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            dictionary_list = []
            outputs = []
            for item in batch_prompts:
                output = client.chat.completions.create(
                    model="o3",
                    max_completion_tokens=16384,
                    messages=[
                        {"role": "user", "content": item}
                    ]
                ).choices[0].message.content
                outputs.append(output)
                #print(outputs)
            for batch_id, batch_response in zip(batch_ids, outputs):
                dictionary_list.append(dict(task_id=batch_id, completion=batch_response))
                with open(name_file, "a") as f:
                    f.write(json.dumps(dictionary_list[-1]) + "\n")
            #print(dictionary_list)
            return dictionary_list

        except (Exception) as e:
            wait_time = base_delay * (2 ** (attempt - 1))
            print(f"[Retry {attempt}/{max_retries}] API error: {e}. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"[Error] Unexpected exception: {e}")
            return dict(question=batch_id, full_response="", generated_code="")

    print(f"[Failed] Exceeded max retries for question: {batch_id}")
    return dict(question=batch_id, full_response="", generated_code="")

parser = argparse.ArgumentParser(description="Arg Parse")

parser.add_argument('--bench_type',default="Human", type=str,help="Human by default, can be Machine")
args = parser.parse_args()

num_samples_per_task = 20
batch_size = 20

model_id = "gpt-o3"
if num_samples_per_task == 1:
    sampling_params = SamplingParams(temperature=0.5, max_tokens=8192)
else:
    sampling_params = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=1024)
descri_path = '/verilog-eval/descriptions/VerilogDescription_' + args.bench_type + '.jsonl'
input_path = '/verilog-eval/data/VerilogEval_' + args.bench_type + '.jsonl'
des_data = load_json(descri_path)
input_data = load_json(input_path)

problems = read_problems(input_path)

batch_idx_list = list(range(0, num_samples_per_task+1, batch_size))
if batch_idx_list[-1] != num_samples_per_task:
    batch_idx_list.append(num_samples_per_task)
del batch_idx_list[0]

for description in des_data:
    problems[description['task_id']]['description'] = description['detail_description']
    problems[description['task_id']]['prompt'] = description['detail_description'] + '\n' + problems[description['task_id']]['prompt'] +'\n'

name_file = "/outputs/" + model_id + "-verilog-eval-"+args.bench_type+".jsonl"
os.makedirs(os.path.dirname(name_file), exist_ok=True)
samples = []
for break_idx, task_id in enumerate(problems):
    low_idx = 0
    all_samples_for_task = [problems[task_id]['prompt']]*num_samples_per_task
    for high_idx in batch_idx_list:
        batch_prompts = all_samples_for_task[low_idx:high_idx]
        temp_generations = batch_generate([task_id]*len(batch_prompts), batch_prompts)
        samples = samples + temp_generations
        low_idx = high_idx
    print("Done with idx " + str(break_idx) + "/155")



