from typing import Literal, TypedDict

DocumentType = Literal["table", "lab_report", "handwritten_prescription", "medical_scan", "unknown"]

class image_state(TypedDict):
    image_path: str          # local path or URL
    image_b64: str           # base64-encoded image
    doc_type: DocumentType   # classification result
    confidence: str          # high / medium / low
    doc_desc: str     