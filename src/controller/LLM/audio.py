import os
from dotenv import load_dotenv
import whisper
from pyttsx3 import init as pyttsx3_init
from pathlib import Path

load_dotenv()
class AudioProcessor:
    def __init__(self, model_name: str = "base"):
        """
        Initialize the AudioProcessor with Whisper model for STT.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.whisper_model = whisper.load_model(os.getenv("audio_model", model_name))
        self.tts_engine = pyttsx3_init()
    
    def speech_to_text(self, audio_file_path: str) -> str:
        """
        Convert speech to text using Whisper.
        
        Args:
            audio_file_path: Path to audio file (mp3, wav, m4a, etc.)
        
        Returns:
            Transcribed text
        """
        # Validate file exists and has content
        if not Path(audio_file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        file_size = Path(audio_file_path).stat().st_size
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {audio_file_path}")
        
        try:
            # Use fp16=False to avoid NaN issues on some GPUs
            result = self.whisper_model.transcribe(
                audio_file_path,
                language="en",
                fp16=False  # Prevents numerical instability
            )
            return result["text"]
        except Exception as e:
            # Fallback: try without GPU if CUDA error
            if "cuda" in str(e).lower() or "nan" in str(e).lower():
                print(f"GPU error, retrying on CPU: {e}")
                result = self.whisper_model.transcribe(
                    audio_file_path,
                    language="en",
                    device="cpu"
                )
                return result["text"]
            raise
