# VeriThoughts: Enabling Automated Verilog Code Generation using Reasoning and Formal Verification
This is the repository for the VeriThoughts Dataset, the first large scale formally verified Verilog reasoning dataset. This repository contains all of the code necessary to generate VeriThoughts as well as our model training and evaluation code.

Our datasets can be found on HuggingFace: [Link](https://huggingface.co/collections/wilyub/verithoughts-datasets-6826de76e798014f05de6c0f)

Our fine-tuned Verilog models can be found on HuggingFace: [Link](https://huggingface.co/collections/nyu-dice-lab/verithoughts-models-681eead7cd13abeb5957baf3)

<p align="center">
  <img src="images/verithoughts.jpg" />
</p>

## Table of Contents

1. [VeriThoughts](#VeriThoughts)
2. [Training](#Training)
3. [Evaluation: VeriThoughts](#evaluation-verithoughts)
4. [Evaluation: Verilog Eval](#verilog-eval-human)

## VeriThoughts
1. Clone and install the repo.
```
git clone https://github.com/wilyub/VeriThoughts.git
cd VeriThoughts/verithoughts
```
2. Install dependencies using the requirements.txt
3. Install Yosys by following the instructions:
```
https://yosyshq.readthedocs.io/projects/yosys/en/0.41/getting_started/installation.html#
```
4. Run the VeriThoughts generation script.
```
python multi-turn-yosys-generation.py --start_idx 0 --end_idx 25090 --label_start 0 --batch_file_path your/file/path --gemini_key your_gemini_key --deepseek_key your_deepseek_key --yosys_location your/yosys/location
```
## Training
1. Clone and install the repo.
```
git clone https://github.com/wilyub/VeriThoughts.git
cd VeriThoughts/training
```
2. Install LlamaFactory by following the instructions:
```
https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#installation
```
3. Install dependencies using the requirements.txt
4. Download all training datasets from HuggingFace and store them in VeriThoughts/training/verilogFinetune/data
5. For each ".sh" file in VeriThoughts/training/verilogFinetune:
```
chmod +x file_name.sh
./file_name.sh
```
## Evaluation: VeriThoughts



## Evaluation: Verilog Eval Human
