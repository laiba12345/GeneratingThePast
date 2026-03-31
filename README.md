# Finetuning LTX-Video using HuggingFace Finetrainers

## Overview

This repo uses:
https://github.com/huggingface/finetrainers

We adapted the original finetrainers code to work with our custom dataset.

We wrote data handling scripts for our dataset and made small changes to the training and inference scripts while keeping most of the original code unchanged.

---

## Setup

### 1. Clone repository

git clone <your-repo-link>  
cd <repo-name>

---

### 2. Install requirements

python -m venv venv  
source venv/bin/activate  

pip install -r requirements.txt

Example requirements.txt:

torch  
transformers  
diffusers  
accelerate  
datasets  
numpy  
pandas  
tqdm  
pyyaml  

---

## Dataset

Download dataset from:

""

Place dataset inside the folder:

Dataset/

Dataset also contains the config file used for training.

---

## Finetuned Weights

If required, place finetuned weights inside:

models/LTX-Video/

Update paths in config file if needed.

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

## Testing

python test.py  

---

## Notes

This project is based on the original finetrainers repository.

We only added:
- custom data handling scripts
- small modifications to training and inference scripts
- dataset config file

Core implementation remains the same as the original repo.
