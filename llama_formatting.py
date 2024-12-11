import json
import os
import time
from together.utils import check_file

dataset_path = "instruction_dataset/acc_data/2d"
new_file_path = f"datasets/acc_data_2d.jsonl"
train_data = []

llama_format = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{model_answer}<|eot_id|>
"""
formatted_data = []


for root, dirs, files in os.walk(dataset_path):
    for file_name in files:
        if file_name.endswith(".json"):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    train_data.append(data)  
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")

print(f"Loaded {len(train_data)} JSON files.")

with open(new_file_path, "w", encoding="utf-8") as new_file:
    for piece in train_data:
        print(piece)
        for trial in piece:
            temp_data = {
                "text": llama_format.format(
                    system_prompt=trial["instruction"],
                    user_question=trial["input"],
                    model_answer=trial["output"],
                )
            }
            new_file.write(json.dumps(temp_data))
            new_file.write("\n")


report = check_file(new_file_path)
assert report["is_check_passed"] == True
