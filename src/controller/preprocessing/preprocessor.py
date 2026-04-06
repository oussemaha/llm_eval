from dotenv import load_dotenv
from src.controller.LLM.audio import AudioProcessor
from src.controller.preprocessing.Image_Preprocessor import ImagePreprocessor
from src.controller.preprocessing.pdf_preprocessor import PDFPreprocessor
from src.controller.preprocessing.mineru import MineruClient
import logging

logger = logging.getLogger(__name__)
import os
class Preprocessor:
    
    def __init__(self):
        load_dotenv()
        self.audio_processor = AudioProcessor(model_name=os.getenv("audio_model"))
        self.image_preprcessor = ImagePreprocessor()
        self.pdf_preprocessor = PDFPreprocessor()
        self.mineru_client = MineruClient()
    
    def run(self,text_input: str, audio_path: str, file_path: str) -> list[dict]:
        content = []

        # text input
        if text_input:
            content.append({"type": "text", "text": text_input})

        # audio transcription
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
        if file_path:
            # image to base64
            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                try:
                    base64_image = self.image_preprcessor.image_to_base64(file_path)
                    content.append(
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    )
                except Exception as e:
                    logger.error(f"Image processing error: {e}")
                    content.append(
                        {"type": "text", "text": f"[Image processing failed: {str(e)}]"}
                    )

            # pdf to text and images
            else:
                try:
                    pdf_data = self.pdf_preprocessor.preprocess(file_path)
                    pdf_text = []
                    for item in pdf_data:
                        if isinstance(item, str):
                            pdf_text.append(item)
                        else:
                            base64_image = self.image_preprcessor.image_to_base64(item)
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": base64_image},
                                }
                            )
                    content.append({"type": "text", "text": "\n".join(pdf_text)})
                except Exception as e:
                    logger.error(f"PDF processing error: {e}")
                    content.append(
                        {"type": "text", "text": f"[PDF processing failed: {str(e)}]"}
                    )
        return content
    def run_v2(self,text_input:str, audio_path:str|None, files:list[str]|None) -> list[dict]:
        content=[]
        if files:
            try:
                result=self.mineru_client.parse_file(files,backend="hybrid-http-client",parse_method="auto",return_images=True)
                content.extend(result)  
                for file in files:
                     if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                        try:
                            base64_image = self.image_preprcessor.image_to_base64(file_path)
                            content.append(
                                {"type": "image_url", "image_url": {"url": base64_image}}
                            )
                        except Exception as e:
                            logger.error(f"Image processing error: {e}")
                            content.append(
                                {"type": "text", "text": f"[Image processing failed: {str(e)}]"}
                            ) 
            except Exception as e:
                logger.error(f"file processing error: {e}")
                content.append(
                    {"type": "text", "text": f"[file processing failed: {str(e)}]"}
                )
            
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
        if text_input:
            content.append({"type": "text", "text": text_input})
        return content
        