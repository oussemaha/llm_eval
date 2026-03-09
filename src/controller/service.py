
import os

from src.controller.preprocessing.pdf_preprocessor import PDFPreprocessor
from src.controller.LLM.audio import AudioProcessor
from src.controller.LLM.Agent import Agent
from src.controller.tools.Retriever_tool import FAISSRetriever_Tool
from src.controller.tools.ToolRegistry import  ToolRegistry
from src.controller.tools.web_search_tool import WebSearchTool
from src.controller.preprocessing.Image_Preprocessor import ImagePreprocessor
from dotenv import load_dotenv

import tempfile
import soundfile as sf

class service:
    def __init__(self):
        #.env loading
        load_dotenv()

        #tool registry initialization and tool registration
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_list([WebSearchTool(), FAISSRetriever_Tool()])

        #agent initialization
        self.llm=Agent(model=os.getenv("LLM"),apikey=os.getenv("api_key"),host=os.getenv("base_url"))

        #audio and image preprocessors initialization
        self.audio_processor = AudioProcessor(model_name=os.getenv("audio_model"))
        self.image_preprcessor = ImagePreprocessor()
        
        pass


    def process(self,history:list,text_input:str,audio_path:str,file_path:str):
        content=[]
        
        #text input
        content.append({"type": "text", "text": text_input})

        #audio transcription
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
                    content.append({"type": "text", "text": f"Audio transcription: {audio_text}"})
                except Exception as e:
                    print(f"Audio processing error: {e}")
                    content.append({"type": "text", "text": f"[Audio transcription failed: {str(e)}]"})
        if file_path:
            #image to base64
            if file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')): 
                base64_image = self.image_preprcessor.image_to_base64(file_path)
                content.append({"type": "image_url", "image_url": {"url": base64_image}})

            #pdf to text and images
            else: 
                preprocessor = PDFPreprocessor()
                pdf_data = preprocessor.preprocess(file_path)
                pdf_text = []
                for item in pdf_data:

                    if isinstance(item, str):
                        pdf_text.append(item)
                    else:
                        base64_image = ImagePreprocessor.image_to_base64(item)
                        content.append({"type": "image_url", "image_url": {"url": base64_image}})
                content.append({"type": "text", "text": "\n".join(pdf_text)})

        history=[
            *history,
            {"role": "user", "content": content},
        ]
        
        #response generation
        response = self.llm.run(self.tool_registry,history)
        history.append({
            "role": "assistant",
            "content": response
            })
        return history        
        
        
