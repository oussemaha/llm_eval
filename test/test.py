from src.service import Service
import os
import json
svc=Service()

files="/home/oussema/Desktop/llm_eval/test/"
prompts=[
    "What should be the next steps in this case?",
    "what do you think of this case?"
    "what's the problem in this case?",
    "what's the solution of this case?",
    
]
with open("test/answers.json", "w", encoding="utf-8") as output:
    prompt_idx=0
    for file in os.listdir(files):
        result=svc.process(history=[],files_path=[os.path.join(files,file)],audio_path=None,text=prompts[prompt_idx])
        output.write(json.dumps({file:result})+'\n')
    prompt_idx= (prompt_idx + 1) % len(prompts)
