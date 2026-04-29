from src.AI.Agent import Agent
from src.preprocessing.Preprocessor import Preprocessor
import json
from src.AI.Retriever import FAISSRetriever_Tool
from src.AI.TTS import TTS
import requests
import re
import soundfile as sf
import numpy as np

def send_notification(message):
    requests.post("https://ntfy.sh/tanit",
                  data=message.encode(encoding='utf-8'))
class Service:
    def __init__(self):

        self.preprocessor=Preprocessor()
        self.reasoning_agent=Agent(system_prompt_file="assests/system_prompts/reasoning_sys_prompt.md")
        self.retrieval_agent=Agent(system_prompt_file="assests/system_prompts/retrieval_sys_prompt.md")
        self.generic_agent=Agent(system_prompt_file="assests/system_prompts/generic_sys_prompt.md")
        self.classifier_agent=Agent(system_prompt_file="assests/system_prompts/classifier_sys_prompt.md")
        self.summarizer_TTS_agent=Agent(system_prompt_file="assests/system_prompts/summarizer_TTS_sys_prompt.md")
        self.guardrail_agent=Agent(system_prompt_file="assests/system_prompts/guardrails_sys_prompt.md")
        self.retriever=FAISSRetriever_Tool()
        self.tts=TTS()
        send_notification("Service initialized successfully.")

    def process(self,history:list,files_path:list,audio_path:str,text:str):
        message=[]
        audio_transcript=""
        if files_path:
            pre=self.preprocessor.preprocess_docs(files_path)
            message.extend(pre)
        if audio_path:
            audio_transcript = self.preprocessor.preprocess_audio(audio_path)
            message.append({"type":"text","text":f"Audio transcription: {audio_transcript}"})
        if text:
            message.append({"type":"text","text":text})
        
        history.append({"role":"user","content":message})
        #the next part can be implemented using langgraph or just a simple if else
        #for now i will implement it using a simple if else
        history_copy=history.copy()
        classification=self.classifier_agent.run(history_copy)
        
        # Clean up possible markdown wrapper
        cleaned_classification = re.sub(r'^```json\s*', '', classification.strip(), flags=re.IGNORECASE)
        cleaned_classification = re.sub(r'```$', '', cleaned_classification.strip())
        
        try:
            classification=json.loads(cleaned_classification)
        except json.JSONDecodeError as e:
            # Fallback to general agent behavior
            classification = {"is_reasoning_required": False, "retrieval_queries": []}
        llm_response=""
        if classification["is_reasoning_required"]:
            knowledge=[]
            for query in classification["retrieval_queries"]:
                knowledge.append(self.retriever.retrieve(query))
            history.append({"role":"user","content":[{"type":"text","text":f"knowledge: {knowledge}"}]})
            llm_response=self.reasoning_agent.run(history)
            
        else:
            llm_response=self.generic_agent.run(history)
        
        history_copy.append({"role":"assistant","content":llm_response})
        if True: #manually changing it just for now . because we do not have a redy front interface
            tts_script = self.summarizer_TTS_agent.run({"role":"user","content":[{"type":"text","text":f" user_input: {message} and/or the audio transcript: {audio_transcript}\n, llm_output: {llm_response}"}]})
            audio_out=self.tts.generate(tts_script)
            sf.write("audio_output.wav", audio_out[0], 24000)

        return history_copy
        
        
if __name__ == "__main__":
    service=Service()
    response=service.process(history=[],files_path=["/home/oussema/Downloads/Downloads/pdf_test/1.pdf"],audio_path=None,text="what sould be the next steps ?")
    print(response)

    
    with open("example.txt", "w", encoding="utf-8") as file:
        file.write(str(response))

    
    