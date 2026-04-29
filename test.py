from src.service import Service
import os
import json
svc=Service()
import requests

def send_notification(message):
    requests.post("https://ntfy.sh/tanit",
                  data=message.encode(encoding='utf-8'))

files="/root/tanit_eval/"
prompts=[
    "What should be the next steps in this case?",
    "what do you think of this case?",
    "what's the problem in this case?",
    "what's the solution of this case?",
    
]
with open(f"{files}answers.json", "w", encoding="utf-8") as output:
    prompt_idx=0
    send_notification("Processing started for all files.")
    for file in os.listdir(files):
        if file.endswith(".json") or file.endswith(".txt"):
            continue
        print(f"Processing file: {file} with prompt: {prompts[prompt_idx]}")
        try:
            result=svc.process(history=[],files_path=[os.path.join(files,file)],audio_path=None,text=prompts[prompt_idx])
        except Exception as e:
            send_notification(f"Error processing file {file} with prompt {prompts[prompt_idx]}: {str(e)}")
            result={"error": str(e),"prompt":prompts[prompt_idx]}
        output.write(json.dumps({file:result})+'\n')
        prompt_idx= (prompt_idx + 1) % len(prompts)
send_notification("Processing completed for all files.")