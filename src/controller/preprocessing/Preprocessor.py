

import operator
from typing import Annotated, TypedDict, Optional
from pathlib import Path

# LangGraph
from langgraph.graph import StateGraph, END

# OpenAI (Whisper + GPT-4o vision)
from openai import OpenAI
from preprocessing.Image_Preprocessor import Image_Preprocessor, MedDocState
from src.models.multimodal_state import MultimodalState


# ─────────────────────────────────────────────
# 1. STATE
# ─────────────────────────────────────────────


        # combined result from aggregator_node


# ─────────────────────────────────────────────
# 2. PREPROCESSOR CLASS
# ─────────────────────────────────────────────

class Preprocessor:
    """
    LangGraph Multimodal Pipeline for processing text, audio, and image inputs.
    """

    def __init__(self, api_key: str = "EMPTY", base_url: str = "http://localhost:8000/v1",
                 vision_model: str = "llava-hf/llava-v1.6-mistral-7b-hf",audio_model: str = "whisper-1"):
        
        """Initialize the preprocessor with OpenAI client and image preprocessor."""

        self.client = OpenAI()   # reads OPENAI_API_KEY from env
        self.image_pre = Image_Preprocessor(api_key=api_key, base_url=base_url, vision_model=vision_model)
        self.image_graph = self.image_pre.build_graph()
        self.app = None  # compiled graph, built lazily

    # ── 2a. Text Node ────────────────────────────
    def text_node(self, state: MultimodalState) -> dict:
        """
        Receives plain text and keeps it clean / ready for downstream use.
        Strips whitespace, collapses blank lines — add any extra processing here.
        """
        raw = state.get("text_input")

        if not raw:
            return {"text_output": None, "errors": ["text_node: no text_input provided"]}

        cleaned = "\n".join(line.rstrip() for line in raw.strip().splitlines())

        print(f"[text_node]  OK  {len(cleaned)} chars received")
        return {"text_output": cleaned}

    # ── 2b. Audio Node (Whisper STT) ─────────────
    def audio_node(self, state: MultimodalState) -> dict:
        """
        Transcribes an audio file using OpenAI Whisper (whisper-1).
        Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
        """
        audio_path = state.get("audio_path")

        if not audio_path:
            return {"transcribed_text": None, "errors": ["audio_node: no audio_path provided"]}

        path = Path(audio_path)
        if not path.exists():
            return {
                "transcribed_text": None,
                "errors": [f"audio_node: file not found -> {audio_path}"],
            }

        print(f"[audio_node]  transcribing {path.name} ...")

        with open(path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )

        transcription = response.strip()
        print(f"[audio_node]  OK  {len(transcription)} chars transcribed")
        return {"transcribed_text": transcription}

    # ── 2c. Image Node ───────────────────────────
    def image_node(self, state: MultimodalState) -> dict:
        """
        Analyses a local image using GPT-4o vision and returns a clear description.
        """
        
        image_graph_result = self.image_graph.invoke(state["image_process"])

        return {"image_process": image_graph_result}

    # ── 2d. Aggregator Node ──────────────────────
    def aggregator_node(self, state: MultimodalState) -> dict:
        """
        Fan-in node — waits for all upstream modality nodes to finish,
        then collects their outputs into a single structured summary.

        This is a plain node (not END), so you can freely wire more nodes
        after it: e.g. aggregator_node → llm_node → storage_node → END
        """
        parts: list[str] = []

        if text := state.get("text_output"):
            parts.append(f"[TEXT]\n{text}")

        if audio := state.get("transcribed_text"):
            parts.append(f"[AUDIO TRANSCRIPT]\n{audio}")

        if image := state.get("image_process"):
            parts.append(f"[IMAGE PROCESS]\n{image}")

        if not parts:
            combined = "(no outputs to aggregate)"
        else:
            combined = "\n\n".join(parts)

        print(f"[aggregator_node]  OK  collected {len(parts)} modality output(s)")
        return {"final_output": combined}

    # ─────────────────────────────────────────────
    # 3. ROUTING  (which modality nodes should run?)
    # ─────────────────────────────────────────────

    def route_inputs(self, state: MultimodalState) -> list[str]:
        """
        Conditional entry point.
        Returns the list of node names to activate based on available inputs.
        All active nodes run in parallel (LangGraph fan-out).
        """
        nodes = []
        if state.get("text_input"):
            nodes.append("text_node")
        if state.get("audio_path"):
            nodes.append("audio_node")
        if state.get("image_path"):
            nodes.append("image_node")

        if not nodes:
            raise ValueError(
                "No inputs provided. Supply at least one of: "
                "text_input, audio_path, image_path"
            )

        print(f"[router]  routing to: {nodes}")
        return nodes

    # ─────────────────────────────────────────────
    # 4. BUILD THE GRAPH
    # ─────────────────────────────────────────────

    def build_graph(self) -> StateGraph:
        """Build and compile the LangGraph pipeline."""
        graph = StateGraph(MultimodalState)

        # Register modality nodes
        graph.add_node("text_node",  self.text_node)
        graph.add_node("audio_node", self.audio_node)
        graph.add_node("image_node", self.image_node)

        # Register the aggregator as a real, named node
        graph.add_node("aggregator_node", self.aggregator_node)

        # Conditional parallel fan-out from START
        graph.set_conditional_entry_point(
            self.route_inputs,
            {
                "text_node":  "text_node",
                "audio_node": "audio_node",
                "image_node": "image_node",
            },
        )

        # All modality nodes fan-in to aggregator_node
        graph.add_edge("text_node",  "aggregator_node")
        graph.add_edge("audio_node", "aggregator_node")
        graph.add_edge("image_node", "aggregator_node")

        # aggregator_node → END  (replace this edge to chain more nodes later)
        graph.add_edge("aggregator_node", END)

        self.app = graph.compile()

    # ─────────────────────────────────────────────
    # 5. PUBLIC API
    # ─────────────────────────────────────────────

    def run_pipeline(
        self,
        text_input:  Optional[str] = None,
        audio_path:  Optional[str] = None,
        image_path:  Optional[str] = None,
    ) -> MultimodalState:
        """
        High-level entry point.
        Pass any combination of inputs; unused modalities stay None.
        Returns the final state with all node outputs + final_output.

        Example
        -------
        preprocessor = Preprocessor()
        result = preprocessor.run_pipeline(
            text_input="Summarise everything",
            audio_path="interview.wav",
            image_path="diagram.png",
        )
        print(result["final_output"])
        """
        if self.app is None:
            self.app = self.build_graph()

        initial_state: MultimodalState = {
            "text_input":         text_input,
            "audio_path":         audio_path,
            "image_path":         image_path,
            "text_output":        None,
            "transcribed_text":   None,
            "image_description":  None,
            "final_output":       None,
            "errors":             [],
        }

        return self.app.invoke(initial_state)


# ─────────────────────────────────────────────
# 6. QUICK SMOKE-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    preprocessor = Preprocessor()

    # ── text only (no API call needed) ───────
    result = preprocessor.run_pipeline(text_input="  Hello,   LangGraph world!  \n\n  ")
    print("\n── Result ───────────────────────────")
    print("final_output :", result["final_output"])
    print("errors       :", result["errors"])

    # ── uncomment to test audio ───────────────
    # result = preprocessor.run_pipeline(audio_path="sample.mp3")
    # print("final_output:", result["final_output"])

    # ── uncomment to test image ───────────────
    # result = preprocessor.run_pipeline(image_path="photo.jpg")
    # print("final_output:", result["final_output"])

    # ── all three at once ─────────────────────
    # result = preprocessor.run_pipeline(
    #     text_input="Analyse everything",
    #     audio_path="interview.wav",
    #     image_path="diagram.png",
    # )
    # print(result["final_output"])