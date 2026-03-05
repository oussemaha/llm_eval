import base64
import json
import httpx
from pathlib import Path
from typing import TypedDict, Literal
from models.image_state import image_state
from openai import OpenAI
from langgraph.graph import StateGraph, END
import re


DocumentType = Literal["table", "lab_report", "handwritten_prescription", "medical_scan", "unknown"]

      # hook for your future logic


class Image_Preprocessor:
    """Medical document preprocessing and classification."""
    
    def __init__(self, api_key: str = "EMPTY", base_url: str = "http://localhost:8000/v1",
                 vision_model: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        """
        Initialize the preprocessor.
        
        Args:
            api_key: OpenAI API key (vLLM doesn't require a real key)
            base_url: vLLM server endpoint
            vision_model: Vision model name to use
        """
        self.prompts = {}
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.vision_model = vision_model
        self.valid_categories: set[DocumentType] = {
            "table", "lab_report", "handwritten_prescription", "medical_scan"
        }
        self.load_prompts()

    def load_prompts(self):
        """Load prompts from files."""
        prompt_files = {
            "table": 'assets/prompts/table_prompt.txt',
            "classification": 'assets/prompts/classification_prompt.txt',
            "lab_report": 'assets/prompts/lab_reports_prompt.txt',
            "handwritten_prescription": 'assets/prompts/Hw_prescriptions_prompt.txt',
            "medical_scan": 'assets/prompts/medical_images_prompt.txt',
            "unknown": 'assets/prompts/unknown_prompt.txt'
        }
        for key, path in prompt_files.items():
            with open(path, 'r', encoding='utf-8') as f:
                self.prompts[key] = f.read().strip()
        return self.prompts

    def encode_image(self, image_path: str) -> tuple[str, str]:
        """Returns (base64_string, mime_type)"""
        suffix = Path(image_path).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png",  ".bmp": "image/bmp",
                    ".tiff": "image/tiff", ".webp": "image/webp"}
        mime = mime_map.get(suffix, "image/jpeg")

        if image_path.startswith("http://") or image_path.startswith("https://"):
            data = httpx.get(image_path).content
        else:
            data = Path(image_path).read_bytes()

        return base64.b64encode(data).decode("utf-8"), mime

    def parse_json(self, response):
        """Extract and parse JSON from response."""
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        data = {}
        try:
            data = json.loads(json_str.strip())        
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response was: {response[:200]}...")
        return data

    def handle_doc(self, state: image_state, prompt) -> str:
        """Process a document with a given prompt."""
        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": state["image_b64"]},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        return raw

    def load_image_node(self, state: image_state) -> image_state:
        """Load image from path or URL."""
        print("📂 Loading image...")
        b64, mime = self.encode_image(state["image_path"])
        state["image_b64"] = f"data:{mime};base64,{b64}"
        return state

    def classify_image_node(self, state: image_state) -> image_state:
        """Classify the image."""
        print("🔍 Classifying image with vLLM...")

        raw = self.handle_doc(state, self.prompts["classification"])
        print(f"📋 Raw response:\n{raw}\n")
        state = self.parse_result(state, raw)
        return state

    def parse_result(self, state: image_state, raw_response: str) -> image_state:
        """Parse classification result from response."""
        print("🧩 Parsing classification result...")
        doc_type: DocumentType = "unknown"
        confidence = "low"

        for line in raw_response.splitlines():
            line = line.strip()
            if line.upper().startswith("CATEGORY:"):
                val = line.split(":", 1)[1].strip().lower()
                if val in self.valid_categories:
                    doc_type = val  # type: ignore
            elif line.upper().startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().lower()

        state["doc_type"] = doc_type
        state["confidence"] = confidence
        print(f"✅ Classified as: {doc_type} (confidence: {confidence})")
        return state

    def route_by_doc_type(self, state: image_state) -> str:
        """Route to handler based on document type."""
        match state["doc_type"]:
            case "table":
                return "table"
            case "lab_report":
                return "lab_report"
            case "handwritten_prescription":
                return "handwritten_prescription"
            case "medical_scan":
                return "medical_scan"
            case _:
                return "unknown"

    def handle_table(self, state: image_state) -> image_state:
        """Handle table documents."""
        res = self.handle_doc(state, self.prompts["table"])
        state["doc_desc"] = res
        print(f"📋 Table description:\n{state['doc_desc']}\n")
        return state

    def handle_lab_report(self, state: image_state) -> image_state:
        """Handle lab report documents."""
        res = self.handle_doc(state, self.prompts["lab_report"])
        state["doc_desc"] = res
        print(f"🧪 [LAB REPORT handler] → description: {state['doc_desc']}")
        return state

    def handle_prescription(self, state: image_state) -> image_state:
        """Handle handwritten prescription documents."""
        res = self.handle_doc(state, self.prompts["handwritten_prescription"])
        state["doc_desc"] = res
        print(f"💊 [PRESCRIPTION handler] → description: {state['doc_desc']}")
        return state

    def handle_scan(self, state: image_state) -> image_state:
        """Handle medical scan documents."""
        res = self.handle_doc(state, self.prompts["medical_scan"])
        state["doc_desc"] = res
        print(f"🩻 [MRI/X-RAY/CT handler] → description: {state['doc_desc']}")
        return state

    def handle_unknown(self, state: image_state) -> image_state:
        """Handle unknown document type."""
        res = self.handle_doc(state, self.prompts["unknown"])
        state["doc_desc"] = res
        print("❓ [UNKNOWN] Could not classify document.")
        return state

    def build_graph(self) -> StateGraph:
        """Build the document processing workflow graph."""
        graph = StateGraph(image_state)

        # ── nodes ──
        graph.add_node("load_image", self.load_image_node)
        graph.add_node("classify", self.classify_image_node)
        graph.add_node("table", self.handle_table)
        graph.add_node("lab_report", self.handle_lab_report)
        graph.add_node("handwritten_prescription", self.handle_prescription)
        graph.add_node("medical_scan", self.handle_scan)
        graph.add_node("unknown", self.handle_unknown)

        # ── linear edges ──
        graph.set_entry_point("load_image")
        graph.add_edge("load_image", "classify")

        # ── conditional routing after parse ──
        graph.add_conditional_edges(
            "classify",
            self.route_by_doc_type,
            {
                "table": "table",
                "lab_report": "lab_report",
                "handwritten_prescription": "handwritten_prescription",
                "medical_scan": "medical_scan",
                "unknown": "unknown",
            },
        )

        # ── all handler nodes → END ──
        for node in ["table", "lab_report", "handwritten_prescription",
                     "medical_scan", "unknown"]:
            graph.add_edge(node, END)

        return graph.compile()


if __name__ == "__main__":
    # Initialize the preprocessor
    preprocessor = Image_Preprocessor()
    app = preprocessor.build_graph()

    initial_state: image_state = {
        "image_path": "sample_xray.jpg",   # ← swap with your image path or URL
        "image_b64": "",
        "doc_type": "unknown",
        "confidence": "",
        "doc_desc": "",
    }

    final_state = app.invoke(initial_state)

    print("\n── Final State ──")
    print(f"  Document Type : {final_state['doc_type']}")
    print(f"  Confidence    : {final_state['confidence']}")
    print(f"  Description   : {final_state['doc_desc']}")