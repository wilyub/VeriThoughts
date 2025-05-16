import json
import random
import argparse
import os
import re
import subprocess
from math import comb
import numpy as np

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            data.append(item)
    return data

def pass_at_k(c_list, n, k):
    pass_at_k_values = []
    for c in c_list:
        if c == 0:
            pass_at_k_values.append(0.0)
        else:
            if n == k:
                pass_at_k_values.append(1.0 if c > 0 else 0.0)
            else:
                val = 1 - comb(n - c, k) / comb(n, k) if (n - c) >= k else 1.0
                pass_at_k_values.append(val)
    return np.mean(pass_at_k_values)

def create_yosys_files(batch_file_path, initial_code, ground_truth):
    with open(batch_file_path + 'verilog_gen.v', 'w') as f:
        f.write(initial_code)
    modified_module_golden, mod_module_list = rename_modules_and_instantiations(ground_truth)
    with open(batch_file_path + 'verilog_truth.v', 'w') as f:
        f.write(modified_module_golden)
    yosys_stdout_list = []
    for original_module_name in mod_module_list:
        module_name = mod_module_list[original_module_name]
        equivalence_string = f"""
        read_verilog {batch_file_path}verilog_truth.v
        read_verilog {batch_file_path}verilog_gen.v
        prep; proc; opt; memory;
        clk2fflogic;
        miter -equiv -flatten {module_name} {original_module_name} miter
        sat -seq 50 -verify -prove trigger 0 -show-all -show-inputs -show-outputs -set-init-zero miter
        """

        with open(batch_file_path + 'equivalence_check.ys', 'w') as f:
            f.write(equivalence_string)


        shell_command = f"""source {yosys_location}
        stdbuf -o0 yosys -s {batch_file_path}equivalence_check.ys
        """

        full_command = f"bash -i -c '{shell_command}'"
        try:
            result = subprocess.run(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=150
            )
            yosys_stdout_list.append(result.returncode)
        except Exception as e:
            yosys_stdout_list.append( -1)
    return yosys_stdout_list

def rename_modules_and_instantiations(verilog_code):
    # Step 1: Find all module names (including those with parameters using #(...))
    module_pattern = re.compile(r'\bmodule\s+(\w+)\s*(?:#\s*\(.*?\))?\s*\(', re.DOTALL)
    module_names = module_pattern.findall(verilog_code)

    # Step 2: Create a mapping from old to new names
    rename_map = {name: name + '1' for name in module_names}

    # Step 3: Replace module declarations
    def replace_module_decl(match):
        original_name = match.group(1)
        before = match.group(0)
        return before.replace(original_name, rename_map[original_name], 1)

    verilog_code = module_pattern.sub(replace_module_decl, verilog_code)

    # Step 4: Replace module instantiations (word boundaries)
    for old_name, new_name in rename_map.items():
        instantiation_pattern = re.compile(rf'\b{old_name}\b')
        verilog_code = instantiation_pattern.sub(new_name, verilog_code)

    return verilog_code, rename_map

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

def extract_modules(verilog_text):
    # This regex matches everything from 'module' to the nearest 'endmodule'
    pattern = r'\bmodule\b.*?\bendmodule\b'
    matches = re.findall(pattern, verilog_text, re.DOTALL)
    # Combine them into a single string, each separated by a newline
    combined_modules = '\n\n'.join(matches)
    return combined_modules

def parsing_helper(verilog_text):
    if args.reason_mode:
        parse1 = extract_code_block(verilog_text)
        if parse1 == '':
            parse1 = extract_modules(verilog_text)
            if parse1 == '':
                parse1 = verilog_text
    else:
        parse1 = extract_modules(verilog_text)
        if parse1 == '':
            parse1 = verilog_text
    return parse1


parser = argparse.ArgumentParser(description="Arg Parse")

# Add an argument for the job_id input, setting the type to string (str)
parser.add_argument("--results_path", type=str, help="Path of model's results file")
parser.add_argument("--samples_per_question", type=int, help="Number of samples generated per quesiton for pass @ k")
parser.add_argument("--old_data", action="store_true", help="Old data format")
parser.add_argument("--reason_mode", action="store_true", help="Reasoning mode for parsing")
parser.add_argument("--yosys_location", type=str, help="Absolute path to yosys environment.")
parser.add_argument("--benchmark_path", type=str, help="Absolute path to the benchmark jsonl")
args = parser.parse_args()
benchmark_data = load_jsonl(args.benchmark_path)
results_data = load_jsonl(args.results_path)
yosys_location = args.yosys_location

verified_results_data = []
if args.old_data:
    for idx, result in enumerate(results_data):
        benchmark = benchmark_data[idx // args.samples_per_question]
        if benchmark['verified']:
            result['ground_truth'] = benchmark['ground_truth']
            verified_results_data.append(result)
    results_data = verified_results_data
evaluation_path = args.results_path[:-6] + "_evaluation.jsonl"
temporary_files_path = args.results_path[:-6] + "/"
os.makedirs(os.path.dirname(temporary_files_path), exist_ok=True)

for result in results_data:
    result['generated_code'] = parsing_helper(result['full_response'])
    stdout_list = create_yosys_files(temporary_files_path, result['generated_code'], result['ground_truth'])
    module_check_count = 0
    for stdout in stdout_list:
        if stdout == 0:
            module_check_count += 1
    evaluation_dictionary = {}
    evaluation_dictionary['stdout_list'] = stdout_list
    evaluation_dictionary['success'] = True if module_check_count == len(stdout_list) else False
    with open(evaluation_path, "a") as f:
        f.write(json.dumps(evaluation_dictionary) + "\n")

evaluation_data = load_jsonl(evaluation_path)
correct_counts = []
for i in range(0, len(evaluation_data), args.samples_per_question):
    single_question_data = evaluation_data[i:i+args.samples_per_question]
    correct_counter = 0
    for sample in single_question_data:
        if sample['success']:
            correct_counter += 1
    correct_counts.append(correct_counter)

print("pass@1:", pass_at_k(correct_counts, args.samples_per_question, 1))
print("pass@5:", pass_at_k(correct_counts, args.samples_per_question, 5))
print("pass@10:", pass_at_k(correct_counts, args.samples_per_question, 10))
