'''
Preliminary experiments (Section 2 in the paper)
Results are saved at'./pre_exp/'
'''
import fastapi_poe as fp
import asyncio
import pandas as pd
import os
import pdb
import json

from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv('POE_API_KEY')

exp_pth='pre_exp/'



"""
Claude-3-Opus
Claude-3.5-Sonnet
Llama-3-70B-T
Llama-3-8B-T
GPT-4o
"""
model_id='Claude-3-Opus'

prompt_oneshot="""
You are an expert in analyzing acceleration data obtained from accelerometer measurements to identify handwritten uppercase letters.
Your task is to interpret a list of 2-axis acceleration data (X, Y) obtained by accelerometer data collected from a device as a user writes a single uppercase letter on a flat surface.

Key points to consider:
The sequence of accelerations is crucial - it reflects the movement change of writing letters.
Consider the standard way of writing uppercase letters, including starting points and stroke order.
The data represents the acceleration of the writing instrument.
The scale of movement is now directly represented in the data values.
Analyze the data mentally, without generating any code or visualizations.

Analyze the major movement patterns in sequence.
Identify the most likely letter based on these movements.
Explain your reasoning in detail, relating the data to the typical writing motion for that letter.
Be open to correction and ready to re-analyze if your initial interpretation is incorrect.
Remember, the goal is to determine which single uppercase letter was most likely written based on the acceleration data provided. The data now represents the instantaneous acceleration over time. Provide your analysis in text form only, without any code or visual representations.

There are eight example data for letters including 'A', 'B', 'C', 'I', 'L', 'M', 'O' and 'U', each corresponding to the acceleration data as the user write the letter.
First, inspect these example data, learn and summarize acceleration features related to letters through these examples and how that relates to how the letter is written.
------
Example for C: {example_C},
Example for O: {example_O},
Example for L: {example_L},
Example for U: {example_U},
Example for M: {example_M},
Example for B: {example_B},
Example for I: {example_I},
Example for A: {example_A},
------
With the learned features, please give the prediction result for followig data. First, provide a detailed explanation of the possible letter matches from A to Z (all 26 alphabets), including their likelihood or confidence scores. Then, summarize the top 5 most likely guesses for each word.
Data: {test_data}
"""
prompt_zeroshot="""
You are an expert in analyzing acceleration data obtained from accelerometer measurements to identify handwritten uppercase letters.
Your task is to interpret a list of 2-axis acceleration data (X, Y) obtained by accelerometer data collected from a device as a user writes a single uppercase letter on a flat surface.
Key points to consider:
The sequence of accelerations is crucial - it reflects the movement change of writing letters.
Consider the standard way of writing uppercase letters, including starting points and stroke order.
The data represents the acceleration of the writing instrument.
The scale of movement is now directly represented in the data values.
Analyze the data mentally, without generating any code or visualizations.

Analyze the major movement patterns in sequence.
Identify the most likely letter based on these movements.
Explain your reasoning in detail, relating the data to the typical writing motion for that letter.
Be open to correction and ready to re-analyze if your initial interpretation is incorrect.
Remember, the goal is to determine which single uppercase letter was most likely written based on the acceleration data provided. The data now represents the instantaneous acceleration over time. Provide your analysis in text form only, without any code or visual representations.

Please give the prediction result for followig data. First, provide a detailed explanation of the possible letter matches from A to Z, including their likelihood or confidence scores. Then, summarize the top 5 most likely guesses for each word.
Data: {test_data}
"""

"""
Example for C: x: {get_data('C', 'exp1')},
Example for O: {get_data('O', 'exp1')},
Example for L: {get_data('L', 'exp1')},
Example for U: {get_data('U', 'exp1')},
Example for M: {get_data('M', 'exp1')},
Example for B: {get_data('B', 'exp1')},
Example for I: {get_data('I', 'exp1')},
Example for A: {get_data('A', 'exp1')},
"""
def get_data(letter,file_folder):
    file_pth=os.path.join(exp_pth, file_folder, letter)+'.csv'
    file_data=pd.read_csv(file_pth,header=None)
    file_data.columns = ['Index', 'X', 'Y', 'Z']
    x_data=file_data['X'].tolist()
    y_data=file_data['Y'].tolist()
    z_data=file_data['Z'].tolist()
    data_2d={
        'x':x_data,
        'y':y_data
    }
    return data_2d

# Create an asynchronous function to encapsulate the async for loop
async def get_responses(api_key, messages,model_id,temperature):
    # pdb.set_trace()
    response=""""""
    async for partial in fp.get_bot_response(messages=messages, bot_name=model_id, api_key=api_key,temperature=0):
        response=response+partial.text
    return response

def send_request_poe_api(exp_prompt,
                         model_id,temperature,api_key):
    pdb.set_trace()
    message=[fp.ProtocolMessage(role="user", content=exp_prompt)]
    response_text=asyncio.run(get_responses(api_key, message,model_id,temperature))
    return response_text

example_data={
    'C': get_data('C', 'exp1'),
    'O': get_data('O', 'exp1'),
    'L': get_data('L', 'exp1'),
    'U': get_data('U', 'exp1'),
    'M': get_data('M', 'exp1'),
    'B': get_data('B', 'exp1'),
    'I': get_data('I', 'exp1'),
    'A': get_data('A', 'exp1'),
    
}
test_folder='exp2'
#test_mode='oneshot'
test_mode='zeroshot'
test_letter_ls=['C','O','L','U','M','B','I','A']
temperature=0.7
test_time=1

for letter in test_letter_ls:
    print("model_id: {};    letter: {}".format(model_id,letter))
    for time in range(test_time):
        results=[]
        test_data=get_data(letter, test_folder)
        if test_mode=='oneshot':
            exp_prompt=prompt_oneshot.format(example_C=example_data['C'] ,example_O= example_data['O'],example_L= example_data['L'],example_U= example_data['U'],\
                                example_M= example_data['M'] ,example_B= example_data['B'],example_I= example_data['I'],example_A= example_data['A'],\
                                test_data=test_data)
        elif test_mode=='zeroshot':
            exp_prompt=prompt_zeroshot.format(test_data=test_data)
        pdb.set_trace()
        response_text=send_request_poe_api(exp_prompt,  model_id, temperature,api_key)
        print(response_text+'\n\n\n')
        results_pth=os.path.join(exp_pth,'results', test_mode, model_id)
        result_file=results_pth+'/{}.json'.format(letter)
        results.append({
            'response':response_text
        })
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False,)