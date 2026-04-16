
import logging

logger = logging.getLogger(__name__)
import gradio as gr


class MultimodalChatUI:
    def __init__(self, chat_callback):
        """
        chat_callback must be a function with signature:

        fn(message, audio, files, history) -> (chatbot_history, new_history)

        For streaming, chat_callback can be a **generator**:

        def streaming_callback(message, audio, files, history):
            partial_history = [..., [user_msg, ""]]
            for token in service.process_stream(history, message):
                partial_history[-1][1] += token
                yield partial_history, partial_history
        """
        self.chat_callback = chat_callback
        self.demo = None

    def build(self):
        # Configure a modern, soft theme
        theme = gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
            font=[
                gr.themes.GoogleFont("Inter"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ],
        ).set(
            body_text_color="*neutral_800",
            background_fill_primary="*neutral_50",
            background_fill_secondary="*neutral_100",
            border_color_accent="*primary_300",
            border_color_primary="*neutral_200",
            color_accent_soft="*primary_50",
            block_background_fill="white",
            block_background_fill_dark="*neutral_900",
            block_label_background_fill="*primary_100",
            block_label_text_color="*primary_600",
            block_title_text_color="*primary_600",
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_600",
            button_primary_text_color="white",
            button_secondary_background_fill="*neutral_200",
            button_secondary_background_fill_hover="*neutral_300",
            button_secondary_text_color="*neutral_800",
        )

        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto;
        }
        .header-title {
            text-align: center;
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #4f46e5, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem !important;
        }
        .header-subtitle {
            text-align: center;
            color: var(--neutral-500);
            font-size: 1.1rem;
            margin-bottom: 2rem !important;
        }
        .chat-panel {
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        """

        self.theme = theme
        self.custom_css = custom_css

        with gr.Blocks(title="TanitAI Assistant") as demo:
            gr.Markdown("# ✨ TanitAI Assistant", elem_classes=["header-title"])
            gr.Markdown(
                "An intelligent multimodal assistant. Ask questions, speak, or upload documents.",
                elem_classes=["header-subtitle"],
            )

            history_state = gr.State([])

            with gr.Row():
                # Chat Interface Column
                with gr.Column(scale=7, elem_classes=["chat-panel"]):
                    chatbot = gr.Chatbot(
                        height=600,
                        show_label=False,
                    )

                    with gr.Row(equal_height=True):
                        text_input = gr.Textbox(
                            placeholder="Type a message...",
                            show_label=False,
                            container=False,  # cleaner look
                            scale=8,
                            min_width=150,
                            autofocus=True,
                        )
                        send_btn = gr.Button(
                            "📤 Send", variant="primary", scale=1, min_width=100
                        )

                # Tools & Uploads Column
                with gr.Column(scale=3):
                    gr.Markdown("### 🧰 Modalities")

                    with gr.Accordion("🎙️ Voice Input", open=True):
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Audio Input",
                            show_label=False,
                        )

                    with gr.Accordion("📎 Attachments", open=True):
                        file_input = gr.File(
                            file_count="multiple",
                            label="Upload Documents / Images",
                            show_label=False,
                        )

                    gr.Markdown("### ⚙️ Actions")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            # --- Interactions ---
            # Send via Enter Key
            text_input.submit(
                fn=self.chat_callback,
                inputs=[text_input, audio_input, file_input, history_state],
                outputs=[chatbot, history_state],
            ).then(
                fn=lambda: ("", None, None),  # Clear inputs after sending
                inputs=None,
                outputs=[text_input, audio_input, file_input],
            )

            # Send via Button Click
            send_btn.click(
                fn=self.chat_callback,
                inputs=[text_input, audio_input, file_input, history_state],
                outputs=[chatbot, history_state],
            ).then(
                fn=lambda: ("", None, None),  # Clear inputs after sending
                inputs=None,
                outputs=[text_input, audio_input, file_input],
            )

            # Clear Chat
            clear_btn.click(
                fn=lambda: ([], [], "", None, None),
                inputs=None,
                outputs=[chatbot, history_state, text_input, audio_input, file_input],
            )

        self.demo = demo
        demo.queue()  # Required for streaming generator callbacks
        return demo

    def launch(self, **kwargs):
        if self.demo is None:
            self.build()
        # Ensure we don't accidentally override the kwargs if provided externally
        if "theme" not in kwargs and hasattr(self, "theme"):
            kwargs["theme"] = self.theme
        if "css" not in kwargs and hasattr(self, "custom_css"):
            kwargs["css"] = self.custom_css

        self.demo.launch(**kwargs)
