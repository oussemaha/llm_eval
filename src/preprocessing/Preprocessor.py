from concurrent.futures import ThreadPoolExecutor
from src.preprocessing.MineruClient import MineruClient
from src.AI.Agent import Agent
import logging
logger = logging.getLogger(__name__)
class Preprocessor:

    def __init__(self):
        self.mineru_client = MineruClient()
        self.tableAgent=Agent(host="http://localhost:2000/v1",apikey="EMPTY",model="Qwen/Qwen3-VL-8B-Instruct",system_prompt_file="assests/system_prompts/table_sys_prompt.txt")
        self.chartAgent=Agent(host="http://localhost:2000/v1",apikey="EMPTY",model="Qwen/Qwen3-VL-8B-Instruct",system_prompt_file="assests/system_prompts/charts_sys_prompt.txt")
        self.otherAgent=Agent(host="http://localhost:2000/v1",apikey="EMPTY",model="Qwen/Qwen3-VL-8B-Instruct",system_prompt_file="assests/system_prompts/other_sys_prompt.txt")
    
    def process_file(self, i):
        additions = [i] 

        if i.get("type") == "image_url":
            img_type = i["image_url"].get("type")
            url = i["image_url"]["url"]
            message = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]

            # Route to the correct agent
            if img_type == "table":
                result = self.tableAgent.run(history=message)
            elif img_type == "chart":
                result = self.chartAgent.run(history=message)
            else:
                result = self.otherAgent.run(history=message)

            additions.append({"type": "text", "text": f"this is a description of the image {result}"})

        return additions

    
    def preprocess_docs(self,docs:list[str])->list[dict]:
        files_content=self.mineru_client.parse_file(docs)
        logger.info(f"Mineru finished, moving to llms if needed")
        docs_processed = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.process_file, files_content))

        for sublist in results:
            docs_processed.extend(sublist)
                
        return docs_processed
    def preprocess_audio(self,audio_path:str):
        content=[]
        if audio_path:
                # Handle Gradio audio format (tuple of sample_rate, audio_data or file path)
                audio_file = None
                if isinstance(audio_path, tuple):
                    # Gradio returns (sample_rate, numpy_array)
                    sample_rate, audio_data = audio_path
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, audio_data, sample_rate)
                        audio_file = tmp.name
                else:
                    # Direct file path
                    audio_file = audio_path

                if audio_file:
                    try:
                        audio_text = self.audio_processor.speech_to_text(audio_file)
                        content.append(
                            {"type": "text", "text": f"Audio transcription: {audio_text}"}
                        )
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        content.append(
                            {
                                "type": "text",
                                "text": f"[Audio transcription failed: {str(e)}]",
                            }
                        )
        return content
if __name__ == "__main__":
    pre=Preprocessor()
    response=pre.preprocess_docs(["/home/oussema/Downloads/Downloads/pdf_test/1.pdf"])
    with open("example.txt", "w", encoding="utf-8") as file:
        file.write(str(response))









# --- Inside your main method ---
