import base64
import json
import httpx
from pathlib import Path
from typing import TypedDict, Literal
from openai import OpenAI
from langgraph.graph import StateGraph, END
import re


client = OpenAI(
    api_key="EMPTY",                        # vLLM doesn't require a real key
    base_url="http://localhost:8000/v1",    # your vLLM server endpoint
)

VISION_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"  # swap with your deployed model
prompts = {}

DocumentType = Literal["table", "lab_report", "handwritten_prescription", "medical_scan", "unknown"]

class MedDocState(TypedDict):
    image_path: str          # local path or URL
    image_b64: str           # base64-encoded image
    doc_type: DocumentType   # classification result
    confidence: str          # high / medium / low
    doc_desc: str           # hook for your future logic

def load_prompts():
    global prompts
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
            prompts[key] = f.read().strip()
    return prompts

def encode_image(image_path: str) -> tuple[str, str]:
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

def parse_json(response):
     # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        # Parse JSON
        data = {}
        try:
            data = json.loads(json_str.strip())        
        except json.JSONDecodeError as e:
            # Log error but continue processing other responses
            print(f"Failed to parse JSON: {e}")
            print(f"Response was: {response[:200]}...")
        return data
def handle_doc(state: MedDocState,prompt) -> MedDocState:
    response = client.chat.completions.create(
        model=VISION_MODEL,
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
        temperature=0.1,   # low temp for deterministic classification
    )

    raw = response.choices[0].message.content.strip()
    return raw

def load_image_node(state: MedDocState) -> MedDocState:
    print("📂 Loading image...")
    b64, mime = encode_image(state["image_path"])
    state["image_b64"] = f"data:{mime};base64,{b64}"
    return state

def classify_image_node(state: MedDocState) -> MedDocState:
    print("🔍 Classifying image with vLLM...")

    raw = handle_doc(state, prompts["classification"])
    print(f"📋 Raw response:\n{raw}\n")
    state = parse_result(state, raw)
    return state

VALID_CATEGORIES: set[DocumentType] = {
    "table", "lab_report", "handwritten_prescription", "medical_scan"
}

def parse_result(state: MedDocState, raw_response: str) -> MedDocState:
    print("🧩 Parsing classification result...")
    doc_type: DocumentType = "unknown"
    confidence = "low"

    for line in raw_response.splitlines():
        line = line.strip()
        if line.upper().startswith("CATEGORY:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in VALID_CATEGORIES:
                doc_type = val  # type: ignore
        elif line.upper().startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip().lower()

    state["doc_type"] = doc_type
    state["confidence"] = confidence
    print(f"✅ Classified as: {doc_type} (confidence: {confidence})")
    return state

def route_by_doc_type(state: MedDocState) -> str:
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



def handle_table(state: MedDocState) -> MedDocState:
    res = handle_doc(state, prompts["table"])
    state["doc_desc"] = res
    print(f"📋 Table description:\n{state['doc_desc']}\n")
    return state

def handle_lab_report(state: MedDocState) -> MedDocState:
    res = handle_doc(state, prompts["lab_report"])
    state["doc_desc"] = res
    print(f"🧪 [LAB REPORT handler] → description: {state['doc_desc']}")
    return state

def handle_prescription(state: MedDocState) -> MedDocState:
    res = handle_doc(state, prompts["handwritten_prescription"])
    state["doc_desc"] = res
    print(f"💊 [PRESCRIPTION handler] → description: {state['doc_desc']}")
    return state

def handle_scan(state: MedDocState) -> MedDocState:
    res = handle_doc(state, prompts["medical_scan"])
    state["doc_desc"] = res
    print(f"🩻 [MRI/X-RAY/CT handler] → description: {state['doc_desc']}")
    return state

def handle_unknown(state: MedDocState) -> MedDocState:
    res = handle_doc(state, prompts["unknown"])
    state["doc_desc"] = res
    print("❓ [UNKNOWN] Could not classify document.")
    return state

def build_graph() -> StateGraph:
    
    graph = StateGraph(MedDocState)

    # ── nodes ──
    graph.add_node("load_image", load_image_node)
    graph.add_node("classify", classify_image_node)
    graph.add_node("table", handle_table)
    graph.add_node("lab_report", handle_lab_report)
    graph.add_node("handwritten_prescription", handle_prescription)
    graph.add_node("medical_scan", handle_scan)
    graph.add_node("unknown", handle_unknown)

    # ── linear edges ──
    graph.set_entry_point("load_image")
    graph.add_edge("load_image",   "classify")

    # ── conditional routing after parse ──
    graph.add_conditional_edges(
        "classify",
        route_by_doc_type,
        {
            "table":                    "table",
            "lab_report":               "lab_report",
            "handwritten_prescription": "handwritten_prescription",
            "medical_scan":         "medical_scan",
            "unknown":                  "unknown",
        },
    )

    # ── all handler nodes → END ──
    for node in ["table", "lab_report", "handwritten_prescription",
                 "medical_scan", "unknown"]:
        graph.add_edge(node, END)

    return graph.compile()



if __name__ == "__main__":
    load_prompts()
    app = build_graph()

    initial_state: MedDocState = {
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