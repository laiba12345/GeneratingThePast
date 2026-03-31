# Generating the Past: End-Frame Video Diffusion

This project explores reverse video generation using a fine-tuned Video Diffusion Model (VDM). Standard VDMs generate videos forward in time from a starting frame — we flip this: given an end-frame image, our model generates a video of the events that led up to it. We achieve this by fine-tuning the pretrained LTX-Video model (2B parameters) using LoRA on a temporally reversed subset of the Something-Something dataset, focused on gravity-related actions (dropping and falling objects). A <reverse> token is used to condition the model on reverse-order generation. The result is a dual-mode VDM capable of both forward and reverse video generation, with applications in event reconstruction and goal-conditioned video generation.


## Overview

This repo uses:
https://github.com/huggingface/finetrainers

We adapted the original finetrainers code to work with our custom dataset.

---

---

## Setup

### 1. Clone repository

git clone https://github.com/laiba12345/GeneratingThePast.git
cd GeneratingThePast

---

### 2. Install requirements

python -m venv venv  
source venv/bin/activate  

pip install -r requirements.txt

---

## Dataset

Download dataset from:
"https://drive.google.com/file/d/189ZCfMbq0OR_TjGBPAOmsoxo_5-Evk3Z"
""

Place dataset inside the folder:

Dataset/

Dataset also contains the config file used for training. Please change the path if required.

---

## Inference

Download the finetuned weights from the link:
https://drive.google.com/file/d/1xfwJJdaRz1Kj7faPcvYUsr5Kr3stbvt4/view?usp=sharing

Place finetuned weights inside:

outputs/

You may need to change the weights path in test.py to the path where you placed the weights

Add the file path in test.py and run the inference.

---

## Project Structure

project-root/

Dataset/  
 config file  

finetrainers/  
 accelerate_configs/  
 assets/  
 docs/  
 examples/  
 finetrainers/  
 tests/  
 train.py  
 train_smoke.sh  
 test.py  

models/  
 LTX-Video/  

README.md  

---

## LORA fine-tuning 

cd finetrainers  

bash train_smoke.sh  

---


## Notes

This project is based on the original finetrainers repository.

We added:
- data preparation scripts
- Lora fine-tunning and testing scripts
- dataset config file
- minor changes to original training file


