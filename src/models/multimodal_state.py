from typing import Optional, TypedDict
from a import MedDocState
from models.audio_state import AudioState


class MultimodalState(TypedDict):
    # ── Inputs ──────────────────────────────
    text_input:  Optional[str]          # raw text
    audio_process: Optional[AudioState] # audio processing result
    image_process: Optional[MedDocState]        # image analysis result

    # ── Aggregator output ────────────────────
    final_output: Optional[str] 