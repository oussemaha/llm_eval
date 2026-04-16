from src.AI.Agent import Agent
from src.preprocessing.Preprocessor import Preprocessor
import json
from src.AI.Retriever import FAISSRetriever_Tool

class Service:
    def __init__(self):
        self.preprocessor=Preprocessor()
        self.reasoning_agent=Agent(host="http://localhost:2000/v1",apikey="EMPTY",model="Qwen/Qwen3-VL-8B-Instruct",system_prompt_file="assests/system_prompts/reasoning_sys_prompt.txt")
        self.retrieval_agent=Agent(host="http://localhost:2000/v1",apikey="EMPTY",model="Qwen/Qwen3-VL-8B-Instruct",system_prompt_file="assests/system_prompts/retrieval_sys_prompt.txt")
        self.generic_agent=Agent(host="http://localhost:2000/v1",apikey="EMPTY",model="Qwen/Qwen3-VL-8B-Instruct",system_prompt_file="assests/system_prompts/generic_sys_prompt.txt")
        self.classifier_agent=Agent(host="http://localhost:2000/v1",apikey="EMPTY",model="Qwen/Qwen3-VL-8B-Instruct",system_prompt_file="assests/system_prompts/classifier_sys_prompt.txt")
        self.retriever=FAISSRetriever_Tool()

    def process(self,history:list,files_path:list,audio_path:str,text:str):
        message=[]
        if files_path:
            pre=self.preprocessor.preprocess_docs(files_path)
            print(pre)
            message.extend(pre)
        if audio_path:
            message.append(self.preprocessor.preprocess_audio(audio_path))
        if text:
            message.append({"type":"text","text":text})
        
        history.append({"role":"user","content":message})
        #the next part can be implemented using langgraph or just a simple if else
        #for now i will implement it using a simple if else
        classification=self.classifier_agent.run(history)
        classification=json.loads(classification)
        if classification["is_reasoning_required"]:
            knowledge=[]
            for query in classification["retrieval_queries"]:
                knowledge.append(self.retriever.retrieve(query))
            history.append({"role":"user","content":[{"type":"text","text":f"knowledge: {knowledge}"}]})
            return self.reasoning_agent.run(history)
        else:
            return self.generic_agent.run(history)
        
        
if __name__ == "__main__":
    service=Service()
    response=service.process(history=[],files_path=["/home/oussema/Downloads/Downloads/pdf_test/1.pdf"],audio_path=None,text="what sould be the next steps ?")
    print(response)

    
    with open("example.txt", "w", encoding="utf-8") as file:
        file.write(str(response))

    
    