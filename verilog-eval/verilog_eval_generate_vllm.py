from vllm import LLM, SamplingParams
from verilog_eval.data import write_jsonl, read_problems
import torch
import sys
import os
import argparse
import math
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from huggingface_hub import login


def load_json(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

def batch_generate(batch_ids, batch_prompts):
    dictionary_list = []
    outputs = llm.generate(batch_prompts, sampling_params)
    for batch_id, batch_response in zip(batch_ids, outputs):
        dictionary_list.append(dict(task_id=batch_id, completion=batch_response.outputs[0].text))
        with open(name_file, "a") as f:
            f.write(json.dumps(dictionary_list[-1]) + "\n")
    return dictionary_list

parser = argparse.ArgumentParser(description="Arg Parse")

# Add an argument for the job_id input, setting the type to string (str)
parser.add_argument("--model_id", type=str, help="HF model name or local path")
parser.add_argument("--gpu",type=str,help="Which number GPU to use, index from 0")
parser.add_argument('--bench_type',default="Human", type=str,help="Human by default, can choose Machine")
parser.add_argument('--hf_token', type=str, help="HF read token")
args = parser.parse_args()
login(token=args.hf_token)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu #select a specific GPU in a multi-GPU CUDA Environment
num_samples_per_task = 20
batch_size = 20

model_id = args.model_id
if num_samples_per_task == 1:
    sampling_params = SamplingParams(temperature=0.5, max_tokens=8192)
else:
    sampling_params = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=1024)
descri_path = '/verilog-eval/descriptions/VerilogDescription_' + args.bench_type + '.jsonl'
input_path = '/verilog-eval/data/VerilogEval_' + args.bench_type + '.jsonl'
des_data = load_json(descri_path)
input_data = load_json(input_path)

llm = LLM(model=model_id, dtype="bfloat16")


problems = read_problems(input_path)

batch_idx_list = list(range(0, num_samples_per_task+1, batch_size))
if batch_idx_list[-1] != num_samples_per_task:
    batch_idx_list.append(num_samples_per_task)
del batch_idx_list[0]

for description in des_data:
    problems[description['task_id']]['description'] = description['detail_description']
    problems[description['task_id']]['prompt'] = description['detail_description'] + '\n' + problems[description['task_id']]['prompt'] +'\n'

name_file = "/outputs/" + args.model_id + "-verilog-eval-"+args.bench_type+".jsonl"
os.makedirs(os.path.dirname(name_file), exist_ok=True)
samples = []
for break_idx, task_id in enumerate(problems):
    # if break_idx >=5:
    #     break
    low_idx = 0
    all_samples_for_task = [problems[task_id]['prompt']]*num_samples_per_task
    for high_idx in batch_idx_list:
        batch_prompts = all_samples_for_task[low_idx:high_idx]
        temp_generations = batch_generate([task_id]*len(batch_prompts), batch_prompts)
        samples = samples + temp_generations
        low_idx = high_idx
    print("Done with idx " + str(break_idx) + "/155")



