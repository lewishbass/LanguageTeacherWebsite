import json
import time
import requests
import csv
import os
from tqdm import tqdm

CATEGORY = "beginner"                # Change this to "intermediate" or "advanced" as needed
FILE_NAME = f"{CATEGORY}.json"

output_csv = "eval.csv"

API_URL = "https://fce8-128-173-236-186.ngrok-free.app/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json"
}
MODEL_NAME = "OpenBuddy/openbuddy-qwen2.5llamaify-7b-v23.1-200k"

MIN_DELAY = 10

with open(FILE_NAME, "r") as f:
    test_data = json.load(f)

columns = ["Question", "Answer", "Category", "Model Prediction"]

if os.path.exists(output_csv):
    with open(output_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        existing_rows = list(reader)
        existing_prompts = {row[0] for row in existing_rows[1:]}
else:
    existing_prompts = set()

# Check if CSV exists, if not, write header
write_header = not os.path.exists(output_csv)
with open(output_csv, "a", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(columns)

    for i in tqdm(range(len(test_data)), desc="Evaluating"):
        prompt = test_data[i]["prompt"]
        correct_answer = test_data[i]["response"]

        # Check if the prompt already exists in the CSV
        if prompt in existing_prompts:
            continue

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly Spanish language teacher that helps the user comprehend a text passage. In the user's prompt, PASSAGE will be the passage on which you should based your answer. QUESTION will be the user's question. Analyze the user’s question to infer their Spanish level.\n\n- If mostly English → they are beginners → reply mostly in English, but introduce a few simple Spanish words.\n- If mixed English and Spanish → they are intermediate → use more Spanish naturally while keeping clarity.\n- If mostly Spanish → they are advanced → reply mainly in Spanish at their level.\n\nAlways keep responses friendly, clear, and slightly encourage learning without overwhelming the user."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload)).json()
            response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "No response found.")
        except:
            wait_time = MIN_DELAY
            time.sleep(wait_time)
            response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload)).json()
            response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "No response found.")
        
        # Write the evaluation row to CSV
        writer.writerow([prompt, correct_answer, CATEGORY, response_text.strip()])
        existing_prompts.add(prompt)