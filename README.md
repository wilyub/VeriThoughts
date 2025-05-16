# VeriThoughts: Enabling Automated Verilog Code Generation using Reasoning and Formal Verification
This is the repository for the VeriThoughts Dataset, the first large scale formally verified Verilog reasoning dataset. This repository contains all of the code necessary to generate VeriThoughts as well as our model training and evaluation code.

<p align="center">
  <img src="images/verithoughts.jpg" />
</p>

## Table of Contents

1. [VeriThoughts](#VeriThoughts)
2. [Training](#Training)
3. [Evaluation](#Evaluation)

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
