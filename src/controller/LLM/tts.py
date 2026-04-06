import os
import logging
import platform
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

SUPPORTED_SAVE_FORMATS = {".mp3", ".wav"}

VoiceGender = Literal["male", "female"]


class TTSProcessor:
    """
    Text-to-Speech processor using pyttsx3 (offline, cross-platform).

    Supports:
      - Speaking text aloud directly
      - Saving speech to .wav or .mp3
      - Voice gender selection
      - Rate and volume control
    """

    def __init__(
        self,
        rate: int = 175,  # words per minute (default human ~150-180)
        volume: float = 1.0,  # 0.0 to 1.0
        voice_gender: VoiceGender = "female",
    ):
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TTS engine: {e}") from e

        self._set_rate(rate)
        self._set_volume(volume)
        self._set_voice(voice_gender)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def speak(self, text: str) -> None:
        """
        Speak text aloud through system audio output.

        Args:
            text: Text to speak

        Raises:
            ValueError: Empty text
            RuntimeError: TTS engine failure
        """
        text = self._validate_text(text)
        logger.info(f"Speaking text ({len(text)} chars)")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            raise RuntimeError(f"TTS speak failed: {e}") from e

    def save_to_file(self, text: str, output_path: str) -> Path:
        """
        Save spoken text to an audio file.

        Args:
            text:        Text to convert
            output_path: Destination path (.wav or .mp3)

        Returns:
            Path to saved file

        Raises:
            ValueError:  Empty text or unsupported format
            RuntimeError: TTS engine or conversion failure
        """
        text = self._validate_text(text)
        output = Path(output_path)

        if output.suffix.lower() not in SUPPORTED_SAVE_FORMATS:
            raise ValueError(
                f"Unsupported format '{output.suffix}'. Supported: {SUPPORTED_SAVE_FORMATS}"
            )

        output.parent.mkdir(parents=True, exist_ok=True)

        # pyttsx3 natively saves as .wav — convert to mp3 if needed
        if output.suffix.lower() == ".mp3":
            return self._save_as_mp3(text, output)

        return self._save_as_wav(text, output)

    def set_rate(self, rate: int) -> None:
        """Update speaking rate (words per minute)."""
        self._set_rate(rate)

    def set_volume(self, volume: float) -> None:
        """Update volume (0.0 to 1.0)."""
        self._set_volume(volume)

    def set_voice(self, gender: VoiceGender) -> None:
        """Switch voice gender."""
        self._set_voice(gender)

    def list_voices(self) -> list[dict]:
        """
        List all available voices on the system.

        Returns:
            List of dicts with keys: id, name, gender, languages
        """
        voices = self.engine.getProperty("voices")
        return [
            {
                "id": v.id,
                "name": v.name,
                "gender": self._infer_gender(v),
                "languages": v.languages,
            }
            for v in voices
        ]

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _set_rate(self, rate: int) -> None:
        if not (50 <= rate <= 400):
            raise ValueError(f"Rate must be between 50 and 400 wpm, got {rate}")
        self.engine.setProperty("rate", rate)

    def _set_volume(self, volume: float) -> None:
        if not (0.0 <= volume <= 1.0):
            raise ValueError(f"Volume must be between 0.0 and 1.0, got {volume}")
        self.engine.setProperty("volume", volume)

    def _set_voice(self, gender: VoiceGender) -> None:
        voices = self.engine.getProperty("voices")
        if not voices:
            logger.warning("No voices found on this system")
            return

        match = next(
            (v for v in voices if gender.lower() in self._infer_gender(v).lower()),
            None,
        )

        if match:
            self.engine.setProperty("voice", match.id)
            logger.info(f"Voice set to: {match.name}")
        else:
            logger.warning(f"No '{gender}' voice found, using system default")

    def _save_as_wav(self, text: str, output: Path) -> Path:
        try:
            self.engine.save_to_file(text, str(output))
            self.engine.runAndWait()
        except Exception as e:
            raise RuntimeError(f"Failed to save WAV: {e}") from e

        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"WAV file was not created or is empty: {output}")

        logger.info(f"Saved WAV to: {output}")
        return output

    def _save_as_mp3(self, text: str, output: Path) -> Path:
        """Save via WAV then convert to MP3 using pydub."""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError(
                "MP3 export requires pydub: pip install pydub. "
                "Also ensure ffmpeg is installed on your system."
            )

        tmp_wav = output.with_suffix(".tmp.wav")
        try:
            self._save_as_wav(text, tmp_wav)
            AudioSegment.from_wav(str(tmp_wav)).export(str(output), format="mp3")
        finally:
            tmp_wav.unlink(missing_ok=True)  # always clean up temp file

        logger.info(f"Saved MP3 to: {output}")
        return output

    @staticmethod
    def _infer_gender(voice) -> str:
        """Infer gender from voice metadata (not standardized across platforms)."""
        name_lower = (voice.name or "").lower()
        id_lower = (voice.id or "").lower()
        combined = f"{name_lower} {id_lower}"
        if any(
            w in combined
            for w in (
                "female",
                "woman",
                "zira",
                "hazel",
                "susan",
                "karen",
                "samantha",
                "victoria",
            )
        ):
            return "female"
        if any(
            w in combined
            for w in ("male", "man", "david", "mark", "daniel", "alex", "fred")
        ):
            return "male"
        return "unknown"

    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text)}")
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        return text


if __name__ == "__main__":
    tts = TTSProcessor(rate=175, volume=1.0, voice_gender="male")

    # Speak aloud
    tts.speak("Hello, how are you?")

    # Save to file
    tts.save_to_file("Hello, how are you?", "output.wav")
    tts.save_to_file("Hello, how are you?", "output.mp3")  # needs pydub + ffmpeg

    # Inspect available voices
    for v in tts.list_voices():
        print(v)
