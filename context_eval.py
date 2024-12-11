import pandas as pd
import numpy as np
import json
import fastapi_poe as fp
import asyncio
import os
from together import Together
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


with open("nouns.txt") as f:
    wordlist = []
    for line in f:
        wordlist.append(line.strip())

results_pth='results/llama8b_ft_2d_fewshot_mode1_cut.csv'
llama_ft_fs_testset=pd.read_csv(results_pth)

word_length = [3, 4, 5, 6]
sample_times = [2, 3, 4, 5]
required_samples_per_length = 200
puzzles = {length: {} for length in word_length}

for word in wordlist:
    word_len = len(word)
    if word_len in word_length:
        if len(puzzles[word_len]) >= required_samples_per_length:
            continue

        puzzles[word_len][word] = {}
        for num_samples in sample_times:
            puzzles[word_len][word][num_samples] = {}  # Initialize nested structure
            
            for letter in word:
                upper_case = letter.upper()
                filtered_samples = []
                max_resample_attempts = 5
                attempts = 0

                # Resample until we get valid data or exceed max attempts
                while not filtered_samples and attempts < max_resample_attempts:
                    raw_samples = llama_ft_fs_testset[llama_ft_fs_testset['letter_gt'] == upper_case].sample(num_samples)['result'].values
                    filtered_samples = [x for x in raw_samples if x is not None and not isinstance(x, float) or not np.isnan(x)]
                    attempts += 1
                
                # Save filtered samples for the letter under the current num_samples
                puzzles[word_len][word][num_samples][letter] = list(set(filtered_samples))

        if len(puzzles[word_len]) >= required_samples_per_length:
            print(f"Required samples reached for word length {word_len}")



client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
model = "gpt-4o"


def send_request_poe_api(exp_prompt, model_id, temperature, api_key):
    message = [fp.ProtocolMessage(role="user", content=exp_prompt)]
    response_text = asyncio.run(get_responses(api_key, message, model_id, temperature))
    return response_text


async def get_responses(api_key, messages, model_id, temperature):
    response = ""
    async for partial in fp.get_bot_response(messages=messages, bot_name=model_id, api_key=api_key, temperature=temperature):
        response += partial.text
    return response


def format_prompt(word_dict):
    prompt = f"I am thinking of a common English word (not an abbreviation) that is {len(word_dict)} letters long.\n"
    prompt += "This word must strictly adhere to the following constraints for each letter position, in the order from left to right:\n"
    
    for i, (letter, candidates) in enumerate(word_dict.items(), start=1):
        if len(candidates) == 1:
            prompt += f"The {ordinal(i)} letter must be '{candidates[0]}'.\n"
        else:
            prompt += f"The {ordinal(i)} letter must be one of {', '.join(map(str, candidates))}.\n"
    
    prompt += (
        "Based on these conditions, provide the single most likely answer. The answer must be:\n"
        "- A valid noun in English.\n"
        "- Strictly adhere to the constraints for all letter positions.\n"
        "What is the word? You just need to give me the final conclusion, do not give me anything other than the single english word."
    )
    return prompt


def ordinal(n):
    """Convert an integer to its ordinal representation."""
    suffix = ['th', 'st', 'nd', 'rd'] + ['th'] * 6
    if 11 <= (n % 100) <= 13: 
        return f"{n}th"
    else:
        return f"{n}{suffix[n % 10]}"


def main():
    puzzles = json.load(open('context/contexts_eval.json'))
    eval_result_path = 'context/results.csv'

    word_length = [3, 4, 5, 6]
    sample_times = [2, 3, 4, 5]

    with open(eval_result_path, 'w') as f:
        f.write("word_len,word,num_samples,response_text\n")
        
        for word_len in word_length:
            for word, value in puzzles[str(word_len)].items():
                for num_samples in sample_times:
                    prompt = format_prompt(puzzles[str(word_len)][word][str(num_samples)])
                    
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature = 0.7,
                        
                        )
                        
                    response_text=completion.choices[0].message.content
                        
                    response_text = response_text.lower()
                    f.write(f"{word_len},{word},{num_samples},{response_text.replace(',', ' ')}\n")
                    f.flush()
                    print(f"Saved: word_len={word_len}, word={word}, num_samples={num_samples}")


if __name__ == "__main__":
    main()
