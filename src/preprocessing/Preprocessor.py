from concurrent.futures import ThreadPoolExecutor
from src.preprocessing.MineruClient import MineruClient
from src.AI.Agent import Agent
from src.AI.STT import AudioProcessor
import tempfile
import soundfile as sf
from pathlib import Path
from PIL import Image
from typing import Union
from io import BytesIO
import base64
import librosa
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)
class Preprocessor:

    def __init__(self):
        self.mineru_client = MineruClient()
        self.tableAgent=Agent(system_prompt_file="assests/system_prompts/table_sys_prompt.md")
        self.chartAgent=Agent(system_prompt_file="assests/system_prompts/charts_sys_prompt.md")
        self.otherAgent=Agent(system_prompt_file="assests/system_prompts/other_sys_prompt.md")
        self.audio_processor=AudioProcessor()

    def _encode_image_and_grayscale(self, image: Union[str, Image.Image]) -> str:
        """
        Encode an image to base64 string.
        
        Args:
            image: File path or PIL Image
            
        Returns:
            Base64 encoded image string with data URL prefix
        """
        if isinstance(image, str):
            # Load image from path
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        img = img.convert('L')  # Convert to grayscale
        # Convert to bytes
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    def _open_file_as_image(self, file_path):
        """
        Opens a file (PDF, TIFF, or image) and returns it as a list of PIL Images.
        Uses different methods for PDF/TIFF vs image files.
    
        Args:
            file_path (str or Path): Path to the file
    
        Returns:
            list: List of PIL.Image.Image objects
            None: If the file can't be opened
        """
        file_path = Path(file_path)
    
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
        # Handle PDF files
        if file_path.suffix.lower() == '.pdf':
            try:
                doc = fitz.open(file_path)
                images = []
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    
                    # Convert to PIL Image
                    img_bytes = pix.tobytes("png")
                    img = Image.open(BytesIO(img_bytes))
                    
                    images.append(img)
                
                doc.close()
                return images
    
            except Exception as e:
                print(f"Error reading PDF {file_path}: {str(e)}")
                return None
        
        # Handle TIFF files
        elif file_path.suffix.lower() in ['.tiff', '.tif']:
            try:
                images = []
                pil_img = Image.open(file_path)
                page_num = 0
                
                while True:
                    # Convert current page to RGB if needed
                    if pil_img.mode not in ('RGB', 'RGBA'):
                        img = pil_img.convert('RGB')
                    else:
                        img = pil_img.copy()
                    
                    images.append(img)
                    page_num += 1
                    
                    try:
                        pil_img.seek(page_num)
                    except EOFError:
                        break
                        
                return images
                    
            except Exception as e:
                print(f"Error reading TIFF {file_path}: {str(e)}")
                return None
                
        elif file_path.suffix.lower() == '.gif':
            try:
                images = []
                pil_img = Image.open(file_path)
                
                # Read first frame
                if pil_img.mode not in ('RGB', 'RGBA'):
                    img = pil_img.convert('RGB')
                else:
                    img = pil_img.copy()
                    
                images.append(img)
                return images
    
            except Exception as e:
                print(f"Error reading GIF {file_path}: {str(e)}")
                return None
        
        # Handle single image files
        else:
            try:
                # Use PIL to open the image
                img = Image.open(file_path)
                
                # Convert to RGB if needed
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                return [img]  # Return as list with single image
    
            except Exception as e:
                print(f"Error reading image {file_path}: {str(e)}")
                return None
    def process_file(self, i):
        additions = []   
        additions.append(i)
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
        print(files_content)
        orig_image = []
        for image in docs:
            pil_images=self._open_file_as_image(image)
            if not pil_images:
                continue
            for img in pil_images:
                image_url = self._encode_image_and_grayscale(img)
                orig_image.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }) 
        docs_processed = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.process_file, files_content))

        for sublist in results:
            docs_processed.extend(sublist)
        if len(orig_image)>0:
            docs_processed.append({"type":"text","text":"the following are the original documents the truth, use them if needed, and verify the pervious extracted text is correct if not base only on the document"})
            docs_processed.extend(orig_image)
        
        return docs_processed
    def preprocess_audio(self,audio_path:str):
        content=[]
        """
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
            """
        if audio_path:
            audio_file = None

            # 1. Standardize the data to mono 16kHz
            if isinstance(audio_path, tuple):
                # Gradio (sr, numpy)
                sr_orig, data = audio_path
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = librosa.to_mono(data.T)
                # Resample to 16kHz (Parakeet requirement)
                data = librosa.resample(data, orig_sr=sr_orig, target_sr=16000)
                sample_rate = 16000
            else:
                # It's a file path - we MUST load and force mono
                data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

            # 2. Save the cleaned mono audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, data, sample_rate)
                audio_file = tmp.name
            if audio_file:
                try:
                    audio_text = self.audio_processor.speech_to_text(audio_file)
                    return audio_text
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    return f"[Audio transcription failed: {str(e)}]"
                    
if __name__ == "__main__":
    pre=Preprocessor()
    response=pre.preprocess_docs(["/home/oussema/Downloads/Downloads/pdf_test/1.pdf"])
    with open("example.txt", "w", encoding="utf-8") as file:
        file.write(str(response))
