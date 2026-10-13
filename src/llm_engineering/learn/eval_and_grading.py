
"""This script generates a dataset using the Anthropic API and saves it as a JSON file. 
The sample dataset consists of tasks that require Python, JSON, or Regex solutions which are exported as JSON files.
The exported JSON file is saved as "dataset.json" in the "json" directory.
The json file is then imported and passed through an evaluation process using the run_eval function.
The evaluation function takes each task in the json file and runs it through claude.
The responses from Claude are collected and included in the evaluation results.
"""

from llm_engineering.utils.helpers import add_user_message, chat, clean_json, generate_dataset, get_anthropic_client, run_eval, model
import json
from dotenv import load_dotenv
import os
from pathlib import Path
from anthropic import Anthropic
import json

#Initialize the Anthropic client with the API key
client = get_anthropic_client()

dataset = generate_dataset(model = model)
output_path = Path(__file__).resolve().parent.parent / "json" / "dataset.json"
with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=2)

with open(output_path, "r") as f:
    dataset = json.load(f)

results = run_eval(dataset)
print(results)



