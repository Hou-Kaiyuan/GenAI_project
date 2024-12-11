# Exploring the Capabilities of LLMs for IMU-based Human Activity Recognition

## Overview
This repository contains the code and resources (data) for our EECSE6694 Generative AI class project. Our project investigates the capabilities of Large Language Models (LLMs) in understanding and interpreting IMU-based human activity data.

### Team Members
| Name           | Email Address          |
|----------------|----------------|
| Kaiyuan Hou  | kh3119@columbia.edu  | 
| Lilin Xu  | lx2331@columbia.edu  | 

## Introduction
Large Language Models (LLMs) have shown exceptional performance across diverse domains, yet their potential for handling sensor data remains unexplored. In this project, we investigate the use of LLMs for interpreting IMU data and observe that their accuracy under zero-shot settings is close to random guessing. By fine-tuning two models with a rigorously self-collected dataset, we achieve up to a $16\times$ improvement in accuracy on the test set, leveraging both fine-tuning and few-shot learning. In a context-based evaluation, where words are predicted from multiple letters, the models attain nearly 80\% accuracy for words with fewer than five characters. However, our findings reveal that fine-tuning state-of-the-art LLMs can degrade performance due to the domain gap between the fine-tuning dataset and the test set, whereas it proves beneficial for weaker models. Overall, while LLMs show promise, their performance appears to be upper-bounded and falls short of simpler traditional models, it still facing several challenges in practical uses.


---

## How to Run the Project

### 1. Prepare the Environment
Set up the required environment by running:
```bash
conda env create -f environment.yml
conda activate GenAI_project
```

### 2. Prepare API Keys
Create a `.env` file and include the following API keys:
- `OPENAI_API_KEY`
- `POE_API_KEY`
- `TOGETHER_API_KEY`
- `WANDB_API_KEY` (required for fine-tuning)

### 3. Code Structure
- **`preliminary_experiments.py`**: Conducts preliminary studies using the POE API. Results are saved in the `pre_exp` folder.
- **`prepare_dataset.py`**: Creates an instruction-response dataset in the instruct dataset format, saved in the `instruction_dataset` folder.
- **`llama_formatting.py`**: Converts the generated dataset into the llama format.
- **`context_eval.py`**: Executes Experiment 2 as detailed in the project report.
- **`datasets` folder**: Stores the final datasets used for GPT and Llama fine-tuning.
- **`experiments` folder**: Contains code to reproduce Experiment 1.
- **`finetune` folder**: Includes scripts for fine-tuning GPT and Llama models, along with CSV files recording loss metrics.

### 4. Suggested Experiments
We recommend running:
```bash
cd experiments
python exp_pretrained_model_llama8b.py
```
This experiment uses a local model, making it feasible for machines with a GPU of 8GB or more. The backend engine used is **Ollama**. 
How to install Ollama please refer to: https://ollama.com/download

Download the llama3.1 8b by:
```bash
ollama pull llama3.1:8b
```


The figures for experiments (confusion matrix, train loss, and context accuracy) can be obtained by running the code in `report_figures.ipynb`.
---

