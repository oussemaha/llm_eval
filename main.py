import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
##PoC with gradio or streamlit
from src.controller.service import service
from src.view.gradio import MultimodalChatUI


svc = service()

def chat_logic(message, audio, files, history):

    history = history or []
    history = svc.process(history=history,text_input=message, audio_path=audio, file_path=files[0] if files else None)
    ui_history = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if isinstance(content, list):
            # Extract text blocks and add an image placeholder if any
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        text_parts.append("[Image attached]")
            ui_history.append({"role": role, "content": "\\n".join(text_parts)})
        else:
            ui_history.append({"role": role, "content": str(content)})
            
    return ui_history, history

if __name__ == "__main__":

    ui = MultimodalChatUI(chat_callback=chat_logic)

    demo = ui.build()
    demo.launch(server_name="0.0.0.0")