import subprocess
import os
import time
import argparse
from tqdm import tqdm
import json
import re
from datasets import load_dataset
from google import genai
from google.genai import types
import certifi
import google.genai.errors
from openai import OpenAI
import httpx
import random

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            data.append(item)
    return data

def retry_on_rate_limit(max_retries=100):
    """Decorator to retry functions on rate limit errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except google.genai.errors.ClientError as e:
                    if hasattr(e, 'code') and e.code == 429:  # Rate limit error
                        # Try to extract retry delay from the error
                        retry_delay = 30  # Default delay
                        try:
                            error_dict = e.args[0]
                            if isinstance(error_dict, dict) and 'details' in error_dict.get('error', {}):
                                for detail in error_dict['error']['details']:
                                    if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo' and 'retryDelay' in detail:
                                        delay_str = detail['retryDelay']
                                        if delay_str.endswith('s'):
                                            retry_delay = int(delay_str[:-1])
                                        break
                        except:
                            pass  # If parsing fails, use the default delay
                        
                        # Add jitter to avoid synchronized retries
                        jitter = random.uniform(1, 1.5)
                        wait_time = retry_delay * jitter
                        print(f"Rate limited. Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        attempts += 1
                    elif hasattr(e, 'code') and e.code in [502, 503]:  # Server errors
                        wait_time = 30 * (2 ** attempts)  # Exponential backoff
                        print(f"Server error (code: {e.code}). Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        attempts += 1
                    else:
                        print(f"Unhandled API error: {e}")
                        if attempts < max_retries - 1:
                            wait_time = 60 * (attempts + 1)  # Linear backoff for unknown errors
                            print(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            attempts += 1
                        else:
                            raise  # Raise if max retries exceeded
            
            print(f"Max retries ({max_retries}) exceeded.")
            return None  # Or handle this case as appropriate
        return wrapper
    return decorator

@retry_on_rate_limit(max_retries=100)
def safe_gemini_send(chat, message, config=None):
    """Safely send a message to Gemini with retry handling"""
    if config is None:
        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=0.6,
            topP=0.95,
        )
    return chat.send_message(message, config=config)

#Extract question from the LLM response
def extract_question_block(text):
    if not text:
        return ""
    pattern = r"QUESTION BEGIN(.*?)QUESTION END"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        s = matches[-1]
    else:
        s = ""
    return s
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

# Find the module name and modify it
def modify_module_name(text):
    match = re.search(r'\bmodule\s+(\w+)', text)
    if match:
        module_name = match.group(1)
        new_module_name = module_name + '1'
        # Replace only the first occurrence
        modified_code = text.replace(f'module {module_name}', f'module {new_module_name}', 1)
    else:
        new_module_name = ""
        modified_code = ""
    return modified_code, new_module_name

def deepseek_generation(initial_question):
    #Create DeepSeek Prompt
    ds_prompt = {}
    ds_prompt['role'] = 'user'
    ds_prompt['content'] = initial_question + "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END."+ "\n<think>\n"
    
    initial_code = ""
    while initial_code == "":
        #Inference on DeepSeek
        while True:
            try:
                initial_deepseek_response = ds_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[ds_prompt],
                    max_completion_tokens=8192,
                    temperature=0.6,
                    top_p=0.95,
                    stream=False
                )
                initial_code = extract_code_block(initial_deepseek_response.choices[0].message.reasoning_content + initial_deepseek_response.choices[0].message.content)
                initial_reasoning = initial_deepseek_response.choices[0].message.reasoning_content
                if initial_code == "":
                    message_list = [ds_prompt]
                    message_list.append(initial_deepseek_response.choices[0].message)
                    message_list.append({"role": "user", "content": "I could not find the code. Please put CODE BEGIN at the start of the code and CODE END at the end of the code."})    
                    temp_deepseek_response = ds_client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=message_list,
                        max_completion_tokens=8192,
                        temperature=0.6,
                        top_p=0.95,
                        stream=False
                    )
                    initial_code = extract_code_block(temp_deepseek_response.choices[0].message.reasoning_content + temp_deepseek_response.choices[0].message.content)
                    initial_reasoning = temp_deepseek_response.choices[0].message.reasoning_content
                break
            except Exception as e:
                wait_time = 30 * random.uniform(1, 2)
                time.sleep(wait_time)
    return initial_code, initial_reasoning

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

def gemini_question_generation(prompt_zip, gemini_chat):
    initial_question = ""
    while initial_question == "":
        initial_response = safe_gemini_send(gemini_chat, prompt_zip[0])
        if initial_response is None:
            print("Failed to get a response after multiple retries.")
            return ""
        initial_question = extract_question_block(initial_response.text)
    return initial_question

def gemini_generation(initial_question):
    temp_chat = gemini_client.chats.create(model="gemini-2.0-flash-thinking-exp")
    # Inference to generate first question
    initial_response = safe_gemini_send(temp_chat, initial_question)
    if initial_response is None:
        return ""
    
    initial_code = extract_code_block(initial_response.text)
    if initial_code == "":
        temp_response = safe_gemini_send(
            temp_chat, 
            "I could not find the code. Please put CODE BEGIN at the start of the code and CODE END at the end of the code."
        )
        if temp_response is None:
            return ""
        return extract_code_block(temp_response.text)
    return initial_code

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

parser = argparse.ArgumentParser(description="Arg Parse")

parser.add_argument("--start_idx", type=int, help="Which idx to start from.")
parser.add_argument("--end_idx", type=int, help="Which idx to end on (not inclusive)")
parser.add_argument("--label_start", type=int, help="Original start idx used to determine the save path. Used for restarting failed runs.")
parser.add_argument("--batch_file_path", type=str, help="Absolute path to where you want to store artifacts/generated dataset")
parser.add_argument("--gemini_key", type=str, help="Gemini API access key")
parser.add_argument("--deepseek_key", type=str, help="DeepSeek API access key")
parser.add_argument("--yosys_location", type=str, help="Absolute path to the yosys environment. Should end in /yosys/oss-cad-suite/environment")
args = parser.parse_args()

os.environ["SSL_CERT_FILE"] = certifi.where()
gemini_client = genai.Client(api_key=args.gemini_key) #Gemini Key
ds_client = OpenAI(api_key=args.deepseek_key, http_client=httpx.Client(verify=False), base_url="https://api.deepseek.com") #DeepSeek Key

ds = load_dataset("scale-lab/MetRex")
rtl_list = ds['train']['RTL']
question_prompt = "Write a question whose answer is the following Verilog code. Do not make the question so detailed that someone can effectively copy the code straight from the question. The question needs to leave room for the person reading it to need to think about the answer. Make sure to state the interface in your question. You should specify the inputs and outputs and make sure they have the same names as in the original code. In addition, include the exact name of the module. Please do the same for all modules present in the Verilog code I give you. The beginning of your final question should start with QUESTION BEGIN and the end of your question should end with QUESTION END.\n"
question_list = [question_prompt + rtl['ground_truth'] for rtl in rtl_list if len(rtl['ground_truth']) < 10000]
rtl_list_temp = [rtl_entry['ground_truth'] for rtl_entry in rtl_list if len(rtl_entry['ground_truth']) < 10000]
rtl_list = rtl_list_temp
sorted_pairs = sorted(zip(question_list, rtl_list), key=lambda pair: len(pair[0]))
sorted_strings, sorted_indices = zip(*sorted_pairs)

question_list = list(sorted_strings)
rtl_list = list(sorted_indices)
start_idx = args.start_idx
end_idx = args.end_idx
label_start = args.label_start
question_list = question_list[start_idx:end_idx]
rtl_list = rtl_list[start_idx:end_idx]
batch_file_path = args.batch_file_path + str(label_start) + "_" + str(end_idx) + "/"
name_file = batch_file_path + "gemini_questions.jsonl"
os.makedirs(os.path.dirname(name_file), exist_ok=True)
os.makedirs(os.path.dirname(batch_file_path), exist_ok=True)
yosys_location = args.yosys_location

verilog_golden_file = batch_file_path + "verilog_truth.v"
verilog_gen_file = batch_file_path + "verilog_gen.v"
os.makedirs(os.path.dirname(verilog_golden_file), exist_ok=True)
os.makedirs(os.path.dirname(verilog_gen_file), exist_ok=True)

equivalence_file = batch_file_path + "equivalence_check.ys"
os.makedirs(os.path.dirname(equivalence_file), exist_ok=True)

for prompt_idx, prompt_zip in tqdm(enumerate(zip(question_list, rtl_list))):
    gemini_chat = gemini_client.chats.create(model="gemini-2.0-flash-thinking-exp")
    initial_question = gemini_question_generation(prompt_zip, gemini_chat)
    initial_code, initial_reasoning = deepseek_generation(initial_question)

    stdout_list = create_yosys_files(batch_file_path, initial_code, prompt_zip[1])
    module_check_count = 0
    for stdout in stdout_list:
        if stdout == 0:
            module_check_count += 1

    save_dictionary = {}
    save_dictionary['ground_truth'] = prompt_zip[1]
    save_dictionary['question'] = initial_question
    save_dictionary['generated_verilog'] = initial_code
    save_dictionary['reason_list'] = initial_reasoning
    save_dictionary['verified'] = True if (module_check_count == len(stdout_list)) else False
    with open(name_file, "a") as f:
        f.write(json.dumps(save_dictionary) + "\n")    
