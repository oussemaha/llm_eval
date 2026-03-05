from typing import TypedDict


class AudioState(TypedDict):
    audio_path: str          # local path to audio file
    transcribed_text: str    # output from Whisper STT