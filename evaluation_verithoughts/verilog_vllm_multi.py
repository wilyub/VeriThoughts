# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
import torch
import sys
import os
import argparse
import math
import gc
from tqdm import tqdm
import json
import re
from copy import deepcopy
from huggingface_hub import login

#Extract code from the LLM response
def extract_code_block(text):
    if not text:
        return ""
    pattern = r"CODE BEGIN(.*?)CODE END"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        s = matches[-1]  # take the last match
    else:
        s = ""
    return s

def load_json(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

def batch_generate(batch_ids, batch_prompts, batch_benchmark):
    dictionary_list = []
    outputs = llm.generate(batch_prompts, sampling_params)
    for batch_id, batch_response, bench_data in zip(batch_ids, outputs, batch_benchmark):
        dictionary_list.append(dict(question=batch_id, full_response=batch_response.outputs[0].text, generated_code=extract_code_block(batch_response.outputs[0].text), ground_truth=bench_data['ground_truth']))
    return dictionary_list

parser = argparse.ArgumentParser(description="Arg Parse")

parser.add_argument("--model_id", type=str, help="HF model name")
parser.add_argument("--sample_number", type=int, help="Number of samples per question")
parser.add_argument("--batch_size", type=int, help="Batch size for LLM inference")
parser.add_argument("--reasoning_mode", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
parser.add_argument("--hf_read_token", type=str, help="Hugging Face read access token for gated models")
parser.add_argument("--benchmark_path", type=str, help="Absolute path to the benchmark jsonl")
parser.add_argument("--tensor_parallel_size", type=int, help="Number of GPUs to distribute inference over")
parser.add_argument("--gpu_choice", type=int, help="1 selects gpus [0, 1, 2, 3]. 2 selects gpus [4, 5, 6, 7]")
args = parser.parse_args()

if args.gpu_choice == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
elif args.gpu_choice == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"


login(token=args.hf_read_token)
num_samples_per_task = args.sample_number
batch_size = args.batch_size

model_id = args.model_id
sampling_params = SamplingParams(temperature=0.6, top_p = 0.95, max_tokens=16384)
benchmark_path = args.benchmark_path
benchmark_data = load_json(benchmark_path)
results_path = 'benchmark_results/' + args.model_id + '/' + 'results.jsonl'
os.makedirs(os.path.dirname(results_path), exist_ok=True)
question_list = []
verified_benchmark_list = []
for data in benchmark_data:
    if not data['verified']:
        continue
    for _ in range(num_samples_per_task):
        if args.reasoning_mode:
            question_list.append(data['question'] + "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.<think>\n")
        else:
            question_list.append(data['question'] + "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.\n")
        verified_benchmark_list.append(data)

llm = LLM(model=model_id, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size)


for i in range(0, len(question_list), batch_size):
    batch_benchmark = verified_benchmark_list[i:i+batch_size]
    batch_questions = question_list[i: i+batch_size]
    batch_generations = batch_generate(batch_questions, batch_questions, batch_benchmark)
    for batch_generation in batch_generations:
        with open(results_path, "a") as f:
            f.write(json.dumps(batch_generation) + "\n")
    print("Done with idx: " + str(i+batch_size))
