import json
import numpy as np
import os
import time
import re
from typing import List, Dict, Generator
from openai import OpenAI
import logging
import backoff
import pyarrow as pa
import pyarrow.parquet as pq
import pdb

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_single_entry(label, data) -> Dict:
    reasoning_prompt_template = """
    You are an expert in analyzing time-series data from an accelerometer. The data consists of acceleration values along three axes: x, y, and z. Each axis is represented as a sequence of numerical values. Along with the data, you are provided with the correct letter that this data represents. Your task is to analyze the patterns in the acceleration data and explain in detail how and why the data corresponds to the given letter.

    Begin by examining the z-axis data to determine whether the motion is in 2D or 3D. If the z-axis values remain relatively constant, assume the motion is 2D and focus your explanation on the patterns in the x and y axes. If the z-axis values vary significantly, infer that the motion is 3D and incorporate the z-axis into your reasoning. Be mindful of accelerometer drift, which may affect the data, and avoid relying on integrations into velocity or position. Instead, focus directly on the raw acceleration data and its patterns.

    Analyze the data to identify features such as peaks, troughs, transitions, or stable regions, and explain how these features align with the structural components of the provided letter. Describe how the observed motion corresponds to key strokes, curves, or directional changes required to form the letter. Use the provided letter as a reference to guide your reasoning and ensure your explanation is consistent with the accelerometer data.

    Here is the data for analysis:

    x: {x_value}  
    y: {y_value}  
    z: {z_value}

    The provided letter is: {letter}. Explain why the data corresponds to this letter in detail.
    """
    
    reasoning_prompt = reasoning_prompt_template.format(x_value = data['x'],y_value=data['y'],z_value=data['z'],letter =label)
    
    response_step1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": reasoning_prompt}
        ],
        temperature = 0.7,
        max_tokens=4098
    )
    
    reconstruct_prompt = f"""
    Take the following response, which explains how accelerometer data corresponds to the letter {label}. The response is structured by starting with the known letter {label} and then explaining why the data aligns with it. Reconstruct the response so that it first analyzes the data independently, identifying patterns and features, and then concludes that the letter is {label} based on these observations. Avoid referring to the letter {label} until the conclusion. Ensure the revised response maintains clarity, structure, and flow.

    Original Response: {response_step1.choices[0].message.content}

    """
    
    response_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": reconstruct_prompt}
        ],
            temperature = 0.7,
            max_tokens=4098
    )
    output = response_answer.choices[0].message.content
    
    question_convertion = f"""
    You are tasked with transforming an analytical prompt into a dataset-friendly version that encourages inference rather than relying on a known ground truth. Take the original prompt provided below, and rewrite it to remove any direct reference to the correct letter. Ensure the new prompt:
    1. Maintains the context of analyzing accelerometer time-series data for motion recognition.
    2. Instructs the responder to infer the letter based on patterns in the data without pre-supposing what the letter is.
    3. Encourages detailed reasoning by focusing on features such as peaks, troughs, transitions, or stable regions in the x, y, and z axes.
    4. Is diverse enough for use in a dataset while retaining simplicity and clarity.

    Here is the original prompt: {reasoning_prompt}
    Now, rewrite the above prompt to make it suitable for a dataset task by removing the explicit ground truth reference while ensuring it aligns with the requirements outlined above.
    You don't have to include the data input in this revised prompt.
    """
    
    response_input = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": question_convertion}
        ]
    )
    
    instruction = response_input.choices[0].message.content
    entry = {}
    entry['output'] = output
    entry['instruction'] = instruction
    entry['input'] = f"x: {data['x']}\ny: {data['y']}\nz: {data['z']}\n"
    entry['text'] = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction: {entry['instruction']}\n### Input: {entry['input']}\n### Response: {entry['output']}"

    return entry          


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
    

def generate_dataset(base_folder, result_folder, entries_per_file=2):
    data_path = []
    for root, dirs, files in os.walk(base_folder):
        if "3d" in root:
            for file in files:
                if file.endswith(".csv"):
                    data_path.append(os.path.join(root, file))
        elif "2d" in root and "mode1_cut" in root:
            for file in files:
                if file.endswith(".csv"):
                    data_path.append(os.path.join(root, file))
    
    data_path=sorted(data_path)
    for dp in data_path:
        mode, filename, label, data = format_data(dp)
        print(f'current label: {label}, target file : {filename}')
        generated_data=[]
        for j in range(entries_per_file):
            print(f'generate ID: {j}')
            
            entry = generate_single_entry(label, data)
            if entry and all(key in entry for key in ['instruction', 'input', 'output', 'text']):
                logger.info(f"Finished processing one entry")
                generated_data.append(entry)
                
                # yield entry
            else:
                logger.warning(f"Skipped incomplete entry")
            time.sleep(2)
            
        # save results
        result_pth = result_folder + '/' + mode + '/' + label + '/' 
        if not os.path.exists(result_pth):
            os.makedirs(result_pth)
        result_file= result_pth + filename +'.json'
        with open(result_file, 'w') as json_file:
            json.dump(generated_data, json_file, indent=4, ensure_ascii=False,)

def save_data_point_as_parquet(entry: Dict, output_file: str, schema: pa.Schema):
    arrays = [
        pa.array([entry['instruction']]),
        pa.array([entry['input']]),
        pa.array([entry['output']]),
        pa.array([entry['text']])
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    
    if os.path.exists(output_file):
        with pq.ParquetWriter(output_file, schema, use_dictionary=True, compression='snappy') as writer:
            writer.write_table(table)
    else:
        pq.write_table(table, output_file, compression='snappy')

if __name__ == "__main__":
    base_folder = "IMUData_1127/acc_data"
    result_folder = "instruction_dataset/acc_data"
    output_file = "instruction_dataset.parquet"
    
    schema = pa.schema([
        ('instruction', pa.string()),
        ('input', pa.string()),
        ('output', pa.string()),
        ('text', pa.string())
    ])
    
    # for entry in generate_dataset(base_folder, result_folder, entries_per_file=3):
        
    #     save_data_point_as_parquet(entry, output_file, schema)
    
    generate_dataset(base_folder, result_folder, entries_per_file=3)