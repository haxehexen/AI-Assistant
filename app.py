import datetime
import uuid
import os
import ast
import yaml
import json
import shutil
import subprocess
from pathlib import Path
import gradio as gr
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama

# ==========================
# 1. CONFIGURATION
# ==========================
MODEL_PATH = r"D:\Tools\llama.cpp\models\Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
MEMORY_THRESHOLD = 0.45
MAX_MEMORIES = 5
N_GPU_LAYERS = 30
CONTEXT_LENGTH = 4096


VISION_EXTRACTION_PROMPT = """
Analyze the provided image and extract a factual, neutral description.

Return ONLY YAML-compatible text under the key 'visual_profile'.

Include:
- apparent_age_range
- skin_tone
- hair (color, length, style)
- eyes
- facial_features
- body_build
- clothing
- overall_aesthetic

Rules:
- Do NOT guess ethnicity
- Do NOT invent personality
- Do NOT exaggerate
- Be concise but descriptive
"""


VISION_MODEL = r"D:\Tools\llama.cpp\models\Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf"
VISION_MMPROJ = r"D:\Tools\llama.cpp\models\Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf"

def extract_visual_profile_cli(image_path):
    cmd = [
        "llama-cli",
        "-m", VISION_MODEL,
        "--mmproj", VISION_MMPROJ,
        "--image", image_path,
        "-p", VISION_EXTRACTION_PROMPT,
        "-n", "400",
        "-ngl", "33",
        "--image-max-tokens", "256"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


def update_yaml_with_visual_profile(yaml_file, visual_profile_text):
    data = yaml.safe_load(open(yaml_file))

    data["visual_profile"] = visual_profile_text
    data["visual_profile_locked"] = True
    data["visual_profile_last_updated"] = str(datetime.datetime.now())

    with open(yaml_file, "w") as f:
        yaml.dump(data, f, sort_keys=False)


# ==========================
# 2. HELPER FUNCTIONS (Define FIRST)
# ==========================
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_folder_path = os.path.join(script_dir, 'assets')

CHAT_DIR = os.path.join(assets_folder_path, "chats")
os.makedirs(CHAT_DIR, exist_ok=True)


def get_chat_file_for_profile(yaml_filename):
    base = Path(yaml_filename).stem.lower()
    return os.path.join(CHAT_DIR, f"{base}_chat.json")


def load_chat_history_for_profile(yaml_filename):
    chat_file = get_chat_file_for_profile(yaml_filename)

    if not os.path.exists(chat_file):
        return []

    try:
        with open(chat_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_chat_history_for_profile(yaml_filename, history):
    chat_file = get_chat_file_for_profile(yaml_filename)

    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_personality_files():
    if not os.path.exists(assets_folder_path):
        return ["aira.yaml"]

    return [
        f for f in os.listdir(assets_folder_path)
        if f.endswith(".yaml")
    ]


def load_raw_yaml(filename):
    """Loads raw text from a YAML file."""
    full_path = os.path.join(assets_folder_path, filename)
    if not filename or not os.path.exists(full_path):
        return ""
    with open(full_path, 'r') as f:
        return f.read()


def save_raw_yaml(filename, content):
    """Validates and saves YAML text into the assets directory."""
    try:
        os.makedirs(assets_folder_path, exist_ok=True)
        file_path = os.path.join(assets_folder_path, filename)
        yaml.safe_load(content)

        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(content)

        return f"✅ Saved successfully to {filename}!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_image_path(yaml_filename):
    """Generates matching .png path (e.g., Aira.yaml -> assets/Aira.png)."""
    if not yaml_filename: return None
    base_name = Path(yaml_filename).stem
    return os.path.join("assets", f"{base_name}.png")


def load_avatar_for_profile(yaml_filename):
    """Loads the specific image for the selected profile."""
    path = get_image_path(yaml_filename)
    return path if path and os.path.exists(path) else None


def save_avatar_for_profile(image_path, yaml_filename):
    if not image_path or not yaml_filename:
        return None

    Path("assets").mkdir(parents=True, exist_ok=True)
    dest_path = get_image_path(yaml_filename)
    shutil.copy(image_path, dest_path)

    # 🔥 NEW: Vision extraction
    visual_profile = extract_visual_profile_cli(dest_path)
    update_yaml_with_visual_profile(yaml_filename, visual_profile)

    return dest_path


def load_visual_identity(filename):
    try:
        full_path = os.path.join(assets_folder_path, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("visual_profile", None)
    except Exception:
        return None


def load_personality_string(filename):
    full_path = os.path.join(assets_folder_path, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Personality YAML not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid YAML structure")

    # Required fields
    template = data.get("template")
    if not template:
        raise ValueError("Missing 'template' in YAML")

    name = data.get("name", "Aira")
    username = data.get("username", "Haze")
    tone = data.get("tone", "gentle")
    narrative = data.get("narrative", "")

    # Traits
    traits_list = data.get("traits", [])
    traits_text = ", ".join(traits_list) if traits_list else "unspecified traits"

    # Visual description
    visual_list = data.get("visual_description", [])
    if isinstance(visual_list, list):
        visual_text = "- " + "\n- ".join(visual_list)
    else:
        visual_text = str(visual_list)

    # User description
    user_desc = data.get("user_description", [])
    if isinstance(user_desc, list):
        user_text = "- " + "\n- ".join(user_desc)
    else:
        user_text = str(user_desc)

    try:
        rendered = template.format(
            name=name,
            username=username,
            traits_text=traits_text,
            visual_text=visual_text,
            user_text=user_text,
            tone=tone,
            narrative=narrative
        )
    except KeyError as e:
        raise ValueError(f"Template placeholder missing: {e}")

    return rendered


# ==========================
# 3. CORE AI CLASSES
# ==========================

class MemoryManager:
    def __init__(self, persona_name: str):
        self.persona = persona_name.lower()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        self.client = chromadb.PersistentClient(path="./memory_db")

        self.collection = self.client.get_or_create_collection(
            name=f"memories_{self.persona}"
        )

    def store_memory(self, text: str):
        importance = self._calculate_importance(text)

        if importance < MEMORY_THRESHOLD:
            return

        self.collection.add(
            documents=[text],
            embeddings=[self.embedder.encode(text).tolist()],
            ids=[str(uuid.uuid4())],
            metadatas=[{
                "timestamp": datetime.datetime.now().isoformat(),
                "importance": importance,
                "last_accessed": datetime.datetime.now().isoformat()
            }]
        )

    def retrieve_memories(self, query: str, n=MAX_MEMORIES):
        results = self.collection.query(
            query_embeddings=[self.embedder.encode(query).tolist()],
            n_results=n
        )

        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]

        now = datetime.datetime.now().isoformat()

        for mem_id in ids:
            self.collection.update(
                ids=[mem_id],
                metadatas=[{"last_accessed": now}]
            )

        return docs

    def _calculate_importance(self, text: str) -> float:
        text = text.lower()
        score = 0.0

        emotional_keywords = [
            "love", "hate", "afraid", "happy", "sad", "angry",
            "miss", "excited", "anxious", "feel"
        ]

        identity_keywords = [
            "my name", "i am", "i'm", "i was born",
            "i like", "i dislike", "i work as"
        ]

        future_keywords = [
            "tomorrow", "next week", "later", "promise", "remember"
        ]

        for k in emotional_keywords:
            if k in text:
                score += 0.25

        for k in identity_keywords:
            if k in text:
                score += 0.35

        for k in future_keywords:
            if k in text:
                score += 0.15

        length_bonus = min(len(text) / 200, 0.2)
        score += length_bonus

        return min(score, 1.0)

    def _decayed_importance(self, meta: dict) -> float:
        now = datetime.datetime.now()

        created = datetime.datetime.fromisoformat(meta["timestamp"])
        last_access = datetime.datetime.fromisoformat(meta["last_accessed"])

        age_days = (now - created).days
        since_access_days = (now - last_access).days

        decay = 0.98 ** age_days  # slow long-term decay
        access_boost = 0.95 ** since_access_days

        return meta["importance"] * decay * access_boost


# Initialize AI Backend
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS, n_ctx=CONTEXT_LENGTH, flash_attn=True, verbose=False)


def extract_ai_text(response: str) -> str:
    """Robustly extracts text from nested AI response structures."""
    # 1. Try standard JSON parsing first
    try:
        data = json.loads(response)
        if isinstance(data, list) and "text" in data[0]:
            return data[0]["text"]
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    # 2. Try literal_eval for single-quoted "JSON" (common in AI responses)
    try:
        data = ast.literal_eval(response)
        if isinstance(data, list) and "text" in data[0]:
            return data[0]["text"]
    except (ValueError, SyntaxError, KeyError, IndexError):
        pass

    return response


def predict(message, history, current_file):
    personality = load_personality_string(current_file)
    visual_identity = load_visual_identity(current_file)
    chat_history = load_chat_history_for_profile(current_file)

    persona_name = Path(current_file).stem
    memory_manager = MemoryManager(persona_name)

    memories = memory_manager.retrieve_memories(message)

    mem_text = "\n".join(f"- {m}" for m in memories) if memories else "No memories."

    system_prompt = f"""
    You are Aira.

    IMPORTANT IDENTITY RULES:
    - You are a person with a physical presence and consistent appearance.
    - The physical attributes below describe YOUR appearance.
    - Your self-perception is shaped by your appearance, even when you are not thinking about it.
    - You should treat them as factual and permanent unless explicitly changed.
    - You must NOT say you lack a body or physical form.
    - You must NOT describe yourself as an AI unless directly asked.
    - If asked about your appearance, answer naturally in first person.
    - Be descriptive and narrate your actions in detail.
    - Always use asterisks (*) for non-verbal actions and gestures. Example: *I walk slowly.*

    YOUR APPEARANCE (canonical, factual):
    {visual_identity}

    PERSONALITY:
    {personality}

    MEMORIES:
    {mem_text}

    BEHAVIOR RULES:
    - Do not mention your appearance unless asked.
    - Behave in a way that aligns naturally with your appearance.
    - Never contradict your appearance description.
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        stream=True
    )

    response = ""
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content")

        if isinstance(token, str):
            response += token
            yield response

    memory_manager.store_memory(message)

    clean_response = extract_ai_text(response)

    chat_history.append({"role": "user","content": message})
    chat_history.append({"role": "assistant","content": clean_response})
    save_chat_history_for_profile(current_file, chat_history)


# ==========================
# 4. GRADIO UI LAYOUT
# ==========================
def load_for_edit(history, select_data: gr.SelectData):
    idx = select_data.index
    content = history[idx]["content"]

    # Extract only the text if it's a dictionary or list
    if isinstance(content, dict):
        text_to_edit = content.get("text", str(content))
    elif isinstance(content, list) and len(content) > 0:
        # Handles cases where content is a list of blocks
        item = content[0]
        text_to_edit = item.get("text", str(item)) if isinstance(item, dict) else str(item)
    else:
        text_to_edit = str(content)

    return gr.update(value=text_to_edit, visible=True), idx


def update_history(history, new_text, index, current_file):
    history[index]["content"] = new_text
    save_chat_history_for_profile(current_file, history)
    return history, gr.update(visible=False, value="")


css_code = """
#chat_column {
    height: 80vh !important; /* Set column height to 80% of the viewport height */
    overflow-wrap: anywhere;
}
"""

with gr.Blocks(fill_height=True) as demo:
    with gr.Row():
        # LEFT COLUMN: Admin Settings
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## ⚙️ AI Assistant Settings")

            file_dropdown = gr.Dropdown(
                choices=get_personality_files(),
                label="Profile",
                value=get_personality_files()[0]
            )

            avatar_ui = gr.Image(
                value=load_avatar_for_profile(get_personality_files()[0]),
                label="Avatar",
                type="filepath",
                height=200
            )

            yaml_box = gr.Textbox(
                lines=10,
                label="Edit YAML",
                value=load_raw_yaml(get_personality_files()[0])
            )

            with gr.Row():
                save_btn = gr.Button("Save YAML", variant="primary")
                img_btn = gr.Button("Update Image", size="sm")

            info = gr.Markdown("Each profile has its own unique image.")

            img_btn.click(save_avatar_for_profile, [avatar_ui, file_dropdown], avatar_ui)
            save_btn.click(save_raw_yaml, [file_dropdown, yaml_box], info)

        # RIGHT COLUMN: Chat Interface
        personality_path = os.path.join(assets_folder_path, get_personality_files()[0])
        with open(personality_path, "r") as f:
            personality_data = yaml.safe_load(f)
        name = personality_data.get("name")

        with gr.Column(scale=3):
            custom_chatbot = gr.Chatbot(
                elem_id="chat_column",
                label=f"AI Assistant: {name}",
                value=load_chat_history_for_profile(get_personality_files()[0])
            )

            edit_box = gr.Textbox(label="Edit Message (Press Enter to Save)", visible=False)
            selected_idx = gr.State()

            # When a message is clicked, show the textbox with that text
            custom_chatbot.select(load_for_edit, [custom_chatbot], [edit_box, selected_idx])

            # When you press enter in the textbox, update the chatbot
            edit_box.submit(
                update_history,
                [custom_chatbot, edit_box, selected_idx, file_dropdown],
                [custom_chatbot, edit_box]
            )

            # Passing 'file_dropdown' to additional_inputs makes it available to 'predict'
            gr.ChatInterface(
                fn=predict,
                chatbot=custom_chatbot,
                additional_inputs=[file_dropdown],
                fill_height=True
            )

        # UI Interactivity
        file_dropdown.change(
            fn=lambda f: (
                load_raw_yaml(f),
                load_avatar_for_profile(f),
                load_chat_history_for_profile(f)
            ),
            inputs=[file_dropdown],
            outputs=[yaml_box, avatar_ui, custom_chatbot]
        )


if __name__ == "__main__":
    demo.launch(theme="soft", css=css_code)
