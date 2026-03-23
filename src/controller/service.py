import logging
logger = logging.getLogger(__name__)

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
        self.llm=Agent(model=os.getenv("LLM"),apikey=os.getenv("api_key"),host=os.getenv("base_url"),max_steps=100)

        #audio and image preprocessors initialization
        self.audio_processor = AudioProcessor(model_name=os.getenv("audio_model"))
        self.image_preprcessor = ImagePreprocessor()
        self.pdf_preprocessor = PDFPreprocessor()

        pass

    def preprocess(self,history:list,text_input:str,audio_path:str,file_path:str):
        content=[]
        
        #text input
        if text_input:
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
                    logger.error(f"Audio processing error: {e}")
                    content.append({"type": "text", "text": f"[Audio transcription failed: {str(e)}]"})
        if file_path:
            #image to base64
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')): 
                try:
                    base64_image = self.image_preprcessor.image_to_base64(file_path)
                    content.append({"type": "image_url", "image_url": {"url": base64_image}})
                except Exception as e:
                    logger.error(f"Image processing error: {e}")
                    content.append({"type": "text", "text": f"[Image processing failed: {str(e)}]"})

            #pdf to text and images
            else: 
                try:
                    pdf_data = self.pdf_preprocessor.preprocess(file_path)
                    pdf_text = []
                    for item in pdf_data:

                        if isinstance(item, str):
                            pdf_text.append(item)
                        else:
                            base64_image = self.image_preprcessor.image_to_base64(item)
                            content.append({"type": "image_url", "image_url": {"url": base64_image}})
                    content.append({"type": "text", "text": "\n".join(pdf_text)})
                except Exception as e:
                    logger.error(f"PDF processing error: {e}")
                    content.append({"type": "text", "text": f"[PDF processing failed: {str(e)}]"})

        history=[
            *history,
            {"role": "user", "content": content},
        ]
        return history

    def process(self,history:list,text_input:str,audio_path:str,file_path:str):
        
        history=self.preprocess(history,text_input,audio_path,file_path)
        #response generation
        response = self.llm.run(self.tool_registry,history)
        history.append({
            "role": "assistant",
            "content": response
            })
        return history        

    def process_stream(self, history: list, text_input: str, audio_path: str = None, file_path: str = None):
        """Same as process() but streams the final LLM response as text delta chunks (generator)."""
        messages=self.preprocess(history,text_input,audio_path,file_path)

        # Stream the LLM response
        yield from self.llm.run_stream(self.tool_registry, messages)
