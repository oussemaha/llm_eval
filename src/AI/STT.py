import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from functools import lru_cache
import whisper
import nemo.collections.asr as nemo_asr

load_dotenv()
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


@lru_cache(maxsize=4)
def load_whisper_model(model_name: str) -> whisper.Whisper:
    """Load and cache Whisper model to avoid repeated expensive loads."""
    logger.info(f"Loading Whisper model: {model_name}")
    return whisper.load_model(model_name)



class AudioProcessor:
    def __init__(self):
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=os.getenv("nemo_asr_model","nvidia/parakeet-tdt-0.6b-v3"))
    def speech_to_text(self, audio_file_path: str) -> str:
        self._validate_file(audio_file_path)
        try:
            print(audio_file_path)
            transcription = self.asr_model.transcribe([audio_file_path])
            result=transcription[0].text
            print(result)
            return result
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            raise RuntimeError(f"ASR transcription failed: {e}") from e
    def _validate_file(self, path: str) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if p.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {path}")
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{p.suffix}'. Supported: {SUPPORTED_EXTENSIONS}"
            )
        return p


if __name__ == "__main__":
    processor = AudioProcessor()
    test_audio_path = "/root/day1_consultation01_doctor.wav"  # Update with
    text = processor.speech_to_text(test_audio_path
        )
    print(f"Transcribed Text: {text}")
