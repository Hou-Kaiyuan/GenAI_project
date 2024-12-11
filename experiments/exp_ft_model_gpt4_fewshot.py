"""
experiments with finetuned model (GPT4) and the few-shot settting
"""

import json
from together import Together
import os
import numpy as np
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import shutil
import argparse
import time
import pdb
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ft_model_key='ft:gpt-4o-2024-08-06:personal::AaUNbC4p'
# client = OpenAI()
# model = client.fine_tuning.jobs.retrieve(ft_model_key).fine_tuned_model



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-l', '--letter_list',action='append', help='delimited list input')
    return parser.parse_args()


system_prompt = """
You are an analyst working on motion recognition using accelerometer time-series data. The data consists of acceleration values recorded along three axes: x, y, and z, represented as sequences of numerical values. 
Your task is to infer a letter (among the 26 letters of the alphabet) that the recorded motion represents by analyzing these sequences.\n\nBegin by examining the z-axis data to assess if the motion captured is predominantly in two or three dimensions. Should the z-axis values show minimal variation, you can initially consider the motion to be two-dimensional, allowing you to concentrate on analyzing patterns found in the x and y axes. Conversely, if there is significant fluctuation in the z-axis, it indicates a three-dimensional motion, requiring the inclusion of the z-axis in your analysis.\n\nFocus on identifying key features such as peaks, troughs, transitions, and stable regions across the axes. Use these features to draw connections between the properties of the data and potential structural components or movements characteristic of particular letters. Consider elements such as strokes, curves, and directional shifts that might be essential for forming a letter and how these are reflected in the data.\n\nApproach the data with awareness of potential accelerometer drift and limit your inferences to patterns visible in the raw acceleration data rather than calculating derived measures like velocity or position. Through detailed reasoning and analysis, propose which letter may be represented and articulate the alignment between the observed patterns and the inferred letter shape, without presupposing the outcome from known truths. Your explanation should remain consistent with the data provided and explore all plausible interpretations.
There are several example data, each corresponding to the acceleration data as the user write the letter.
First, inspect these example data, learn and summarize acceleration features related to letters through these examples and how that relates to how the letter is written.

------
{example_prompt}
------
With the learned features, please give the prediction result for followig data.
Following is the data: {user_prompt}
"""



def format_data(file_path):
    imu_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    letter = file_path.split('/')[-2]
    filename = file_path.split('/')[-1].replace('.csv','')
    mode = file_path.split('/')[-4]+'/'+file_path.split('/')[-3]
    data = {}
    data['x'] = ', '.join(map(str, imu_data[:, 5]))
    data['y'] = ', '.join(map(str, imu_data[:, 6]))
    data['z'] = ', '.join(map(str, imu_data[:, 7]))
    return mode, filename, letter, data

exp_pth='../results/gpt4_ft_2d_fewshot/'
test_data_folder='2d/mode1_cut/'
data_pth=f'../IMUData_1127/acc_data/{test_data_folder}/'
test_letter_ls=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
example_num=6

if 'test' not in test_data_folder:
    similarity_pth='../IMUData_1127/acc_data/2d/similarity_train/similarity.csv'
else:
    similarity_pth='../IMUData_1127/acc_data/2d/similarity/similarity.csv'
similarity_data=pd.read_csv(similarity_pth)
sorted_similarity_data = similarity_data.sort_values(by=['test_data_pth', 'similarity']).reset_index(drop=True)


args = parse_args()

if args.letter_list:
    test_letter_ls=args.letter_list

print(test_letter_ls)
    
for letter in test_letter_ls:
    letter_data_pth=data_pth+letter+'/'
    file_ls= sorted(os.listdir(letter_data_pth))
    letter_data_file_ls=[f for f in file_ls if f.endswith('.csv')]
    for file in letter_data_file_ls:
        results=[]
        examples=''
        
        file_path = letter_data_pth+file
        
        print(f'process data for {file_path}')
        sorted_=sorted_similarity_data[sorted_similarity_data['test_data_pth']==file_path]
        for i in range(example_num):
            if 'test' not in test_data_folder:
                idx=i+1 # exceed the first one (the test data itself)
            else:
                idx=i
            row=sorted_.iloc[idx]
            _,example_filename,example_label,example_data=format_data(row['train_data_pth'])
            example_prompt=f"Example for {example_label}: x:{example_data['x']}\ny:{example_data['y']}\nz:{example_data['z']}\n"
            examples=examples+example_prompt
        mode, filename, label, data = format_data(file_path)

        user_prompt = f"x:{data['x']}\ny:{data['y']}\nz:{data['z']}"
        prompt=system_prompt.format(example_prompt=examples, user_prompt=user_prompt)
        input_context = prompt
        '''
        pretained 
        '''
        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input_context}
        ],
        temperature = 0.7,
        
        )
        
        output_text=completion.choices[0].message.content
        
        
        print(output_text+'\n\n\n')
        results_pth=os.path.join(exp_pth,test_data_folder, letter)
        if not os.path.exists(results_pth):
            os.makedirs(results_pth)
        result_file=results_pth+'/{}.json'.format(file)
        results.append({
            'response':output_text
        })
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False,)

