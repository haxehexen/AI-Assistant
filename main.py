import os
import app
import gradio as gr
import signal
import threading
import customtkinter as ctk
from PIL import Image


IMAGE_PATH = r"E:\PythonProject\AI Assistant\assets\Persona\AIIReS\AIIReS.png"


# ==========================
# Gradio UI
# ==========================

css_code = """
#chat_column {
    height: 80vh !important; /* Set column height to 80% of the viewport height */
    overflow-wrap: anywhere;
}

#google_api_key input {
    background-color: transparent !important;
}
"""


api_key, cx = app.load_google_keys()


with (gr.Blocks(fill_height=True) as demo):
    demo.unload(lambda: os.kill(os.getpid(), signal.SIGTERM))
    with gr.Row():
        # LEFT COLUMN: Admin Settings
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## ⚙️ AI Assistant Settings")

            avatar_ui = gr.Image(
            value=app.load_avatar_for_profile(app.get_persona()[0][1]),
            label="Avatar",
            type="filepath",
            height=300
            )

            # Use gr.Accordion() to create a collapsible section
            with gr.Accordion("📝 Persona Settings", open=False):
                persona = gr.Dropdown(
                    choices=app.get_persona(),
                    label="Persona",
                    value=app.get_persona()[0][1]
                )

                (
                    username_val,
                    name_val,
                    traits_val,
                    visual_val,
                    tone_val,
                    narrative_val,
                    user_desc_val,
                    template_val
                ) = app.load_data_yaml(persona.value)

                username = gr.Textbox(label="User Name", placeholder=username_val)
                persona_name = gr.Textbox(label="Persona Name", placeholder=name_val)

                traits = gr.Textbox(
                    lines=2,
                    label="Persona Traits",
                    value=traits_val,
                    interactive = True
                )

                visuals = gr.Textbox(
                    lines=2,
                    label="Persona Visuals",
                    value=visual_val,
                    interactive = True
                )

                tone = gr.Textbox(
                    lines=2,
                    label="Persona Tone",
                    value=tone_val,
                    interactive = True
                )

                narrative = gr.Textbox(
                    lines=2,
                    label="Persona Narrative",
                    value=narrative_val,
                    interactive = True
                )

                user_desc = gr.Textbox(
                    lines=2,
                    label="User Description",
                    value=user_desc_val,
                    interactive = True
                )

                template = gr.Textbox(
                    lines=2,
                    label="Template",
                    value=template_val,
                    interactive = True
                )

                with gr.Row():
                    create_new_btn = gr.Button("Create Persona", variant="primary")
                    save_btn = gr.Button("Save Persona", variant="primary")

                with gr.Row():
                    img_btn = gr.Button("Update Image", variant="secondary")
                    delete_btn = gr.Button("Delete Persona", size="sm")

                info = gr.Markdown("Persona information.")

                create_new_btn.click(
                    app.create_yaml_wrapper,
                    [username,
                    persona_name,
                    traits,
                    visuals,
                    tone,
                    narrative,
                    user_desc,
                    template],
                    info)

                img_btn.click(app.save_avatar_for_profile, [avatar_ui, persona], avatar_ui)
                save_btn.click(
                    app.save_yaml_wrapper,
                    [persona,
                    username,
                    persona_name,
                    traits,
                    visuals,
                    tone,
                    narrative,
                    user_desc,
                    template],
                    info)

                delete_btn.click(app.delete_persona_yaml,[persona], info)

            with gr.Accordion("🔐 API Settings", open=False):
                google_api_input = gr.Textbox(
                    label="Google API Key",
                    value=api_key,
                    type="password",
                    elem_id="google_api_key"
                )

                google_cx_input = gr.Textbox(
                    label="Google CX (Search Engine ID)",
                    value=cx,
                    type="password",
                    elem_id="google_api_key"
                )

                save_btn = gr.Button("💾 Save Keys")
                status_output = gr.Textbox(label="Status", interactive=False)

                status_markdown = gr.Markdown(
                    f"""
                    **Current Status**
                    - API Key: `{app.mask(api_key)}`
                    - CX: `{app.mask(cx)}`
                    """
                )

                save_btn.click(
                    fn=app.save_google_keys,
                    inputs=[google_api_input, google_cx_input],
                    outputs=[status_output, status_markdown]
                )

            with gr.Accordion("🖥️ AI Configurations", open=False):
                gguf_btn = gr.Button("Select GGUF (GPT-Generated Unified Format) file")
                gguf_output = gr.Textbox(lines=2, label="Selected GGUF File", value=app.get_gguf_file())
                gguf_btn.click(fn=app.select_gguf_file, outputs=gguf_output)

                db_dir_btn = gr.Button("Select ChromaDB Directory")
                db_dir_output = gr.Textbox(lines=2, label="ChromaDB Directory", value=app.get_vec_db())
                db_dir_btn.click(fn=app.select_db_folder, outputs=db_dir_output)

                llm_config_editor = gr.Textbox(
                    lines=5,
                    label="Edit LLM Configuration",
                    value=app.load_llm_config()[0][0],
                    interactive = True
                )

                save_llm_btn = gr.Button("💾 Save LLM Config")
                save_llm_output = gr.Textbox(label="Status", interactive=False)
                save_llm_markdown = gr.Markdown()

                save_llm_btn.click(
                    fn=app.save_llm_config,
                    inputs=llm_config_editor,
                    outputs=[save_llm_output, save_llm_markdown]
                )

        # RIGHT COLUMN: Chat Interface
        with gr.Column(scale=3):

            chat_display = gr.Chatbot(
                elem_id="chat_column",
                label="Conversation Log",
                height=600
            )
            chat_display.retry(app.retry_last_response, [chat_display, persona], [chat_display])

            user_input = gr.Textbox(label="Your message")

            send_btn = gr.Button("Send")
            send_btn.click(
                fn=app.predict,
                inputs=[user_input, persona],
                outputs=chat_display
            )

            demo.load(
                fn=lambda p: app.context_to_chatbot(app.profile_key(p)) if p else [],
                inputs=persona,
                outputs=chat_display
            )

            persona.change(
                fn=app.load_chatbot_on_open,
                inputs=persona,
                outputs=chat_display
            )

            edit_box = gr.Textbox(label="Edit Message (Press Enter to Save)", visible=False)
            selected_idx = gr.State()

            # When a message is clicked, show the textbox with that text
            chat_display.select(app.load_for_edit, [chat_display], [edit_box, selected_idx])

            # When you press enter in the textbox, update the chatbot
            edit_box.submit(
                app.update_history,
                [chat_display, edit_box, selected_idx, persona],
                [chat_display, edit_box]
            )

        # UI Interactivity
        persona.change(
            fn=lambda f: (
                app.load_avatar_for_profile(f),
                *(app.load_data_yaml(f)),  # The '*' unpacks (username, name) into two separate values
            ),
            inputs=[persona],
            outputs=[avatar_ui,
                     username,
                     persona_name,
                     traits,
                     visuals,
                     tone,
                     narrative,
                     user_desc,
                     template]
        )


# ==========================
# CustomTkinter UI
# ==========================


class AppLauncher:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("AIIReS Launcher")
        self.root.geometry("1200x800")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # --- LEFT COLUMN ---
        self.side = ctk.CTkFrame(self.root, width=400, corner_radius=0)
        self.side.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.my_image = ctk.CTkImage(light_image=Image.open(IMAGE_PATH),
                                     dark_image=Image.open(IMAGE_PATH),
                                     size=(352, 192))

        self.image_label = ctk.CTkLabel(self.side, text="", image=self.my_image)
        self.image_label.grid(row=0, padx=20, pady=20, sticky="n")

        # --- RIGHT COLUMN ---
        self.main = ctk.CTkFrame(self.root, width=700, corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.label = ctk.CTkLabel(self.main, text="Launch Web UI Interface", font=("Arial", 16))
        self.label.pack(pady=20)

        self.button = ctk.CTkButton(self.main, text="Start AI Assistant", command=self.handle_launch)
        self.button.pack(padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.main, text="Status: Ready", text_color="gray")
        self.status_label.pack(pady=10)


    def kill_process(self):
        os.kill(os.getpid(), signal.SIGTERM)


    def gradio_logic(self):
        demo.launch(inbrowser=True, theme=gr.themes.Base(), css=css_code, allowed_paths=[app.ASSETS_PATH])


    def handle_launch(self):
        gradio_thread = threading.Thread(target=self.gradio_logic, daemon=False)
        gradio_thread.start()
        self.root.destroy()


    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    clApp = AppLauncher()
    clApp.run()
