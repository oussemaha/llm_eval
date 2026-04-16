import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from functools import lru_cache
import whisper

load_dotenv()
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


@lru_cache(maxsize=4)
def load_whisper_model(model_name: str) -> whisper.Whisper:
    """Load and cache Whisper model to avoid repeated expensive loads."""
    logger.info(f"Loading Whisper model: {model_name}")
    return whisper.load_model(model_name)


class AudioProcessor:
    def __init__(self, model_name: str = "base"):
        model_name = os.getenv("AUDIO_MODEL", model_name)
        self.model = load_whisper_model(model_name)

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

    def speech_to_text(self, audio_file_path: str) -> dict:
        """
        Transcribe audio with automatic language detection.

        Returns:
            {
                "text": str,
                "language": str,        # ISO 639-1 code e.g. "fr"
                "language_probs": dict, # full probability map over all languages
                "duration": float,      # audio duration in seconds
            }

        Raises:
            FileNotFoundError: Audio file missing
            ValueError: Empty file or unsupported format
            RuntimeError: Transcription failed after all retries
        """
        self._validate_file(audio_file_path)

        transcribe_kwargs = dict(audio_file_path=audio_file_path, fp16=False)

        try:
            result = self._run_transcribe(audio_file_path, fp16=False)
        except RuntimeError as e:
            raise RuntimeError(f"Transcription failed: {e}") from e

        language = result.get("language", "unknown")
        logger.info(
            f"Transcribed '{audio_file_path}' | lang={language} | chars={len(result['text'])}"
        )
        """
            return {
                "text": result["text"].strip(),
                "language": language,
                "language_probs": result.get("language_probs", {}),
                "duration": result.get("segments", [{}])[-1].get("end", 0.0),
            }
        """
        return result["text"].strip()

    def _run_transcribe(self, audio_file_path: str, fp16: bool) -> dict:
        """Run transcription, falling back to CPU if GPU fails."""
        try:
            return self.model.transcribe(audio_file_path, fp16=fp16)
        except Exception as e:
            if any(kw in str(e).lower() for kw in ("cuda", "nan", "out of memory")):
                logger.warning(f"GPU transcription failed ({e}), retrying on CPU")
                cpu_model = whisper.load_model(
                    self.model.dims.n_mels and "base", device="cpu"
                )
                return cpu_model.transcribe(audio_file_path, fp16=False)
            raise
