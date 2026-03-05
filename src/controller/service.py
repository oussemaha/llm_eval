



from controller.LLM import llm
from preprocessing.Image_Preprocessor import MedDocState
from preprocessing.Preprocessor import AudioState, MultimodalState, Preprocessor
from retriever.retriever import FAISSRetriever


llm_instance = llm.LLM(vision_model="llava-hf/llava-v1.6-mistral-7b-hf", api_key=None, base_url="http://localhost:8000/v1")

def pipeline(image_path:str,audio_path:str,text_input:str):
    retriever = FAISSRetriever(
        embedding_model="neuml/pubmedbert-base-embeddings",
        index_type="flat",
        persist_dir="assets/retriever_data"
    )
    preprocessor = Preprocessor("EMPTY", "http://localhost:8000/v1", "llava-hf/llava-v1.6-mistral-7b-hf", "whisper-1")

    if image_path is not None:
        image_state:MedDocState = {
            "image_path": image_path,   # ← swap with your image path or URL
            "image_b64": "",
            "doc_type": "unknown",
            "confidence": "",
            "doc_desc": "",
        }
    else:
        image_state = None

    if audio_path is not None:
        audio_state :AudioState= {
            "audio_path": audio_path,
            "transcribed_text": "",
        }
    else:
        audio_state = None

    state:MultimodalState={
        "image_process": image_state,
        "audio_process": audio_state,
        "text_input": text_input,
        "final_output": None,
    }
    preprocessor.build_graph()  # build the graph lazily
    final_state = preprocessor.app.invoke(state)
    retriever_prompt=f"data extracted from image: {final_state['image_process']['doc_desc']}\n\n" \
        f"user question: {text_input}. {final_state['audio_process']['transcribed_text']}\n\n" \
        f"Based on the above information, retrieve relevant medical knowledge to answer the user's question."
    
    knowledge_base=retriever.retrieve(retriever_prompt)    
    final_prompt=f"""user question: {text_input} . {final_state['audio_process']['transcribed_text']}\n\n \   knowledge retrieved from retriever: {knowledge_base}\n\n 
        Based on the above information, provide a comprehensive answer to the user's question.
    """
    llm_response = llm_instance.call_vision(image_path=image_path,user_prompt=final_prompt)
    return llm_response