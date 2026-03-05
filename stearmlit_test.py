import gradio as gr

# Logo-inspired Palette
PURPLE = "#674EA7"  # Deep background
TEAL = "#2DD4BF"    # Vibrant loop accent
WHITE = "#FFFFFF"   # Contrast/Text

custom_css = f"""
    .gradio-container {{
        background-color: {PURPLE};
    }}
    /* Style the Chatbot bubbles to be fluid and rounded */
    .message-wrap .message {{
        border-radius: 20px !important;
        border: 1px solid {TEAL}44 !important; /* Subtle teal border */
    }}
    /* Highlighting the record and upload buttons */
    button.primary {{
        background: {TEAL} !important;
        color: {PURPLE} !important;
    }}
    /* Customizing the input box to look modern */
    #input-box textarea {{
        border: 2px solid {TEAL} !important;
    }}
"""

def handle_multimodal(message, history):
    """
    Processes incoming text, images, and recorded audio.
    'message' is a dict: {"text": "...", "files": [{"path": "...", "orig_name": "..."}]}
    """
    response = "Input received! "
    
    if message["files"]:
        # Count the types of files (images vs audio)
        file_list = [f['path'].lower() for f in message["files"]]
        audio_count = sum(1 for f in file_list if f.endswith(('.wav', '.mp3', '.m4a')))
        image_count = sum(1 for f in file_list if f.endswith(('.png', '.jpg', '.jpeg')))
        
        details = []
        if image_count: details.append(f"{image_count} image(s)")
        if audio_count: details.append(f"{audio_count} audio recording(s)")
        
        response += f"I see you sent: {' and '.join(details)}."
    
    if message["text"]:
        response += f"\n\nYour message: {message['text']}"
        
    return response

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"### <span style='color:{WHITE}'>Dual-Loop Assistant</span>")
    
    gr.ChatInterface(
        fn=handle_multimodal,
        multimodal=True,
        # The key change: adding 'microphone' to sources
        textbox=gr.MultimodalTextbox(
            sources=["upload", "microphone"], 
            file_types=["image", "audio"],
            placeholder="Type, upload images, or hit the mic to record...",
            elem_id="input-box"
        ),
        examples=[
            {"text": "What is in this image?", "files": []},
            {"text": "Can you transcribe this?", "files": []}
        ],
    )

if __name__ == "__main__":
    demo.launch()