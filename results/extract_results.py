"""
extract letter results from the generated answer
"""
import fastapi_poe as fp
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
# from func_timeout import func_timeout, FunctionTimedOut
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()
model = "gpt-4o"



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-l', '--letter_list',action='append', help='delimited list input')
    return parser.parse_args()


system_prompt = """
You are an analyst tasked with extracting the final result from a paragraph that analyzes a sequence of data and provides a recognition result. The final result is typically one or more uppercase alphabet letters.
If a letter is present, output only the letter(s) in plain text.
If no letter is found, output "None."
Here is the paragraph: {user_prompt}
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

exp_pth='./llama8b_pretrained_2d_fewshot/'
test_data_folder='2d/mode1_cut/'
data_pth=f'../IMUData_1127/acc_data/{test_data_folder}/'
test_letter_ls=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']




args = parse_args()

if args.letter_list:
    test_letter_ls=args.letter_list

print(test_letter_ls)



for letter in test_letter_ls:
    results=[]
    answer_pth=exp_pth+test_data_folder+letter+'/'
    file_ls= sorted(os.listdir(answer_pth))
    answer_file_ls=[f for f in file_ls if f.endswith('.json')]
    for file in answer_file_ls:
        
        
        file_path = answer_pth+file
        print(f'process {file_path}')
        
        with open(file_path, 'r') as file:
            data = json.load(file)
        answer=data[0]["response"]

        prompt=system_prompt.format(user_prompt=answer)
        input_context = prompt
       
        
        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input_context}
        ]
        )
        
        output_text=completion.choices[0].message.content
        
        print(output_text+'\n\n\n')
        results.append([letter,file_path,output_text,answer])
        # time.sleep(15)

    df = pd.DataFrame(results, columns=['letter_gt', 'file_name', 'result', 'generated_answer'])
    df.to_csv(exp_pth.split('/')[-2]+'_'+test_data_folder.split('/')[-2]+'.csv',index=False,mode='a')

