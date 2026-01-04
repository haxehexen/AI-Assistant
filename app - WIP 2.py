import os
import json
import yaml
import uuid
import shutil
import datetime
import requests
import wikipediaapi
from pathlib import Path

import gradio as gr
from llama_cpp import Llama

# ==========================
# CONFIG
# ==========================

#ASSETS_DIR = Path("assets")
ASSETS_DIR = Path(r"E:\PythonProject\AI Assistant\assets")

GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_CX = "YOUR_SEARCH_ENGINE_ID"

HISTORY_DIR = ASSETS_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

MAX_TURNS = 12   # editable + AI context window
MAX_TOKENS = 3000

# ==========================
# 1. CONFIGURATION
# ==========================
MODEL_PATH = r"D:\Tools\llama.cpp\models\Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
MEMORY_THRESHOLD = 0.45
MAX_MEMORIES = 5
N_GPU_LAYERS = 30
CONTEXT_LENGTH = 4096


def get_personality_files():
    if not os.path.exists(ASSETS_DIR):
        return ["aira.yaml"]

    return [
        f for f in os.listdir(ASSETS_DIR)
        if f.endswith(".yaml")
    ]


def load_raw_yaml(filename):
    """Loads raw text from a YAML file."""
    full_path = os.path.join(ASSETS_DIR, filename)
    if not filename or not os.path.exists(full_path):
        return ""
    with open(full_path, 'r') as f:
        return f.read()


def save_raw_yaml(filename, content):
    """Validates and saves YAML text into the assets directory."""
    try:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        file_path = os.path.join(ASSETS_DIR, filename)
        yaml.safe_load(content)

        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(content)

        return f"✅ Saved successfully to {filename}!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def save_avatar_for_profile(image_path, yaml_filename):
    if not image_path or not yaml_filename:
        return None

    Path(ASSETS_DIR).mkdir(parents=True, exist_ok=True)
    dest_path = get_image_path(yaml_filename)
    shutil.copy(image_path, dest_path)

    # 🔥 NEW: Vision extraction

###    visual_profile = extract_visual_profile_cli(dest_path)
###    update_yaml_with_visual_profile(yaml_filename, visual_profile)

    return dest_path


# ==========================
# HISTORY FILE HELPERS
# ==========================

def profile_key(yaml_file: str) -> str:
    return Path(yaml_file).stem.lower()

def context_path(profile: str) -> Path:
    return HISTORY_DIR / f"{profile}_context.json"

def archive_path(profile: str) -> Path:
    return HISTORY_DIR / f"{profile}_archive.txt"

# --------------------------
# CONTEXT (AI) HISTORY
# --------------------------

def load_context(profile):
    path = context_path(profile)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

def save_context(profile, history):
    context_path(profile).write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def trim_context(history):
    system = [m for m in history if m["role"] == "system"]
    convo = [m for m in history if m["role"] != "system"]
    return system + convo[-MAX_TURNS * 2:]


def count_tokens(llm, messages):
    full_text = "".join([
        chunk["text"] if isinstance(chunk, dict) else chunk
        for m in messages
        for chunk in (m["content"] if isinstance(m["content"], list) else [m["content"]])
    ])

    return len(llm.tokenize(full_text.encode("utf-8")))


def trim_messages(messages, max_tokens=MAX_TOKENS):  # Leave room for response
    system_prompt = messages[0] if messages[0]["role"] == "system" else None
    others = messages[1:] if system_prompt else messages

    while count_tokens(llm, messages) > max_tokens and len(others) > 1:
        others.pop(0)

    return [system_prompt] + others if system_prompt else others


# --------------------------
# ARCHIVE (USER) HISTORY
# --------------------------

def append_archive(profile, msg):
    ts = msg["timestamp"]
    role = msg["role"].upper()
    content = msg["content"]

    with open(archive_path(profile), "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {role}:\n{content}\n\n")


def load_archive_text(profile):
    path = archive_path(profile)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_archive_text(profile, history):
    archive_path = os.path.join(
        ASSETS_DIR, "history", f"{profile}_archive.txt"
    )

    with open(archive_path, "w", encoding="utf-8") as f:
        for msg in history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            # Normalize content to string
            if isinstance(content, dict):
                text = content.get("text", str(content))
            elif isinstance(content, list):
                text = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            else:
                text = str(content)

            f.write(f"{role}: {text}\n\n")


def rewrite_archive_last_window(profile, context):
    """
    Rewrites ONLY the last editable window in the archive.
    Older content remains untouched.
    """
    path = archive_path(profile)
    if not path.exists():
        return

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    marker = "=== EDITABLE WINDOW START ===\n"
    if marker not in lines:
        return

    idx = max(i for i, l in enumerate(lines) if l == marker)
    preserved = lines[:idx + 1]

    rebuilt = []
    for m in context:
        rebuilt.append(
            f"[{m['timestamp']}] {m['role'].upper()}:\n{m['content']}\n\n"
        )

    path.write_text("".join(preserved + rebuilt), encoding="utf-8")


def load_personality_string(filename):
    full_path = os.path.join(ASSETS_DIR, filename)

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


def get_image_path(yaml_filename):
    """Generates matching .png path (e.g., Aira.yaml -> assets/Aira.png)."""
    if not yaml_filename: return None
    base_name = Path(yaml_filename).stem
    return os.path.join(ASSETS_DIR, f"{base_name}.png")


def load_avatar_for_profile(yaml_filename):
    """Loads the specific image for the selected profile."""
    path = get_image_path(yaml_filename)
    return path if path and os.path.exists(path) else None


# ==========================
# MESSAGE HELPERS
# ==========================

def new_message(role, content):
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
    }


# Initialize AI Backend
#llm = Llama(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS, n_ctx=CONTEXT_LENGTH, flash_attn=True, verbose=False)

llm = Llama(model_path=MODEL_PATH,
            n_gpu_layers=33,
            n_ctx=CONTEXT_LENGTH,
            temperature=0.65,
            top_p=0.85,
            top_k=40,
            repeat_penalty=1.25,
            repeat_last_n=256,
            presence_penalty=0.4,
            frequency_penalty=0.3,
            flash_attn=True,
            verbose=False)


# ==========================
#   INTERNET CORE FUNCTIONS
# ==========================

# Initialize the Wikipedia object (User-Agent is REQUIRED in 2026)
wiki = wikipediaapi.Wikipedia(
    user_agent='MyDataProject (contact@example.com)',
    language='en'
)


def extract_wiki_title_llm(user_input: str) -> str:
    prompt = (
        "Extract the most likely Wikipedia article title from the input.\n"
        "Return ONLY the title.\n\n"
        f"Input: {user_input}\n"
        "Title:"
    )

    r = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=32
    )

    return r["choices"][0]["message"]["content"].strip()


def wiki_search(query: str, max_chars=1200):
    print(f"DEBUG: Searching Wikipedia for '{query}'...")
    page = wiki.page(query)

    if not page.exists():
        return "", None

    return page.summary[:max_chars], page.fullurl


def build_prompt(user_query: str, context: str) -> list:
    system_prompt = (
    #    "You are a helpful AI assistant.\n"
        "Use the provided CONTEXT to answer the user question.\n"
        "If the context is insufficient, say you are not sure.\n"
        "Do NOT invent facts unless the user specifically said so."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "system",
            "content": f"CONTEXT:\n{context}"
        })

    messages.append({
        "role": "user",
        "content": user_query
    })

    return messages


def wants_sources(query: str) -> bool:
    triggers = [
        "source", "sources", "reference", "references",
        "link", "links", "citation", "cite", "url"
    ]
    q = query.lower()
    return any(t in q for t in triggers)


def rewrite_search_query_llm(user_input: str) -> str:
    prompt = (
        "Rewrite the following user question into a short factual search query.\n"
        "Use only keywords.\n"
        "Do not answer the question.\n"
        "Do not add new information.\n\n"
        f"User question: {user_input}\n"
        "Search query:"
    )

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=32
    )

    return response["choices"][0]["message"]["content"].strip()


def google_search(query, max_results=3, timeout=10):
    url = "https://www.googleapis.com/customsearch/v1"

    print(f"DEBUG: Searching Google for: {query}")

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": max_results,
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        snippets = []
        sources = []

        for item in data.get("items", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link")

            if snippet:
                snippets.append(f"{title}: {snippet}")
            if link:
                sources.append(link)

        text = "\n".join(snippets)
        print(f"DEBUG: Received {len(data.get('items', []))} results")
        return text, sources

    except Exception as e:
        return "", []


# ==========================
# CORE CHAT LOGIC
# ==========================

def predict(user_input, current_file):
    profile = profile_key(current_file)
    context = load_context(profile)

    if not any(m["role"] == "system" for m in context):
        context.append(
            new_message("system", load_personality_string(current_file))
        )

    runtime_context = []
    sources = []

    google_keywords = ["google"]
    wiki_keywords = ["wiki", "wikipedia"]
    keywords = google_keywords + wiki_keywords

    query = user_input.lower()
    for keyword in keywords:
        query = query.replace(keyword, "")

    if any (keyword in user_input.lower() for keyword in keywords):
        query = rewrite_search_query_llm(user_input)

    if any (keyword in user_input.lower() for keyword in google_keywords):
        google_text, google_url = google_search(query)
        print("GOOGLE TEXT SENT TO LLM:\n", google_text)
    else:
        google_text, google_url = "", None

    if any (keyword in user_input.lower() for keyword in wiki_keywords):
        title = extract_wiki_title_llm(query)
        wiki_text, wiki_url = wiki_search(title)
        print("WIKI TEXT SENT TO LLM:\n", wiki_text)
    else:
        wiki_text, wiki_url = "", None

    if google_text or wiki_text:
        runtime_context.append(
            new_message(
                "system",
                "The following is factual reference material.\n"
                "Do not include links unless explicitly instructed.\n\n"
                + google_text + wiki_text
            )
        )
        if google_url or wiki_url:
            sources.append(google_url)
            sources.append(wiki_url)

    user_msg = new_message("user", user_input)
    context.append(user_msg)

    print("runtime_context: ", runtime_context)

    combined = trim_context(runtime_context + context)
    print("combined: ", combined)

    response = llm.create_chat_completion(
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in combined
        ]
    )["choices"][0]["message"]["content"]

    assistant_msg = new_message("assistant", response)
    context.append(assistant_msg)

    save_context(profile, context)
    append_archive(profile, user_msg)
    append_archive(profile, assistant_msg)

    return context_to_chatbot(profile)


# ==========================
# EDITING (LAST MAX_TURNS ONLY)
# ==========================

def edit_message(profile, msg_id, new_content):
    context = load_context(profile)

    editable_ids = {
        m["id"] for m in context if m["role"] != "system"
    } | {m["id"] for m in context[-MAX_TURNS * 2:]}

    for m in context:
        if m["id"] == msg_id:
            if msg_id not in editable_ids:
                raise ValueError("Message not editable.")
            m["content"] = new_content
            break

    save_context(profile, context)
    rewrite_archive_last_window(profile, context)


def load_chatbot_on_open(current_file):
    if not current_file:
        return []
    return context_to_chatbot(profile_key(current_file))


def context_to_chatbot(profile):
    context = load_context(profile)
    messages = []

    for m in context:
        if m["role"] in ("user", "assistant"):
            messages.append({
                "role": m["role"],
                "content": m["content"]
            })

    return messages


def is_editable_index(history, idx):
    if idx is None:
        return False

    editable_start = max(0, len(history) - (MAX_TURNS * 2))
    return idx >= editable_start


def load_for_edit(history, select_data: gr.SelectData):
    idx = select_data.index

    # 🚫 Block edits outside editable range
    if not is_editable_index(history, idx):
        return (
            gr.update(value="", visible=False),
            gr.update(value=None)
        )

    content = history[idx]["content"]

    # Normalize content
    if isinstance(content, dict):
        text_to_edit = content.get("text", str(content))
    elif isinstance(content, list) and len(content) > 0:
        item = content[0]
        text_to_edit = item.get("text", str(item)) if isinstance(item, dict) else str(item)
    else:
        text_to_edit = str(content)

    return (
        gr.update(value=text_to_edit, visible=True),
        idx
    )


def update_history(history, new_text, idx, persona):
    if idx is None:
        return history, gr.update(visible=False)

    if not is_editable_index(history, idx):
        return history, gr.update(visible=False)

    history[idx]["content"] = new_text

    profile = profile_key(persona)
    save_context(profile, history)
    save_archive_text(profile, history)

    return history, gr.update(value="", visible=False)


def find_last_user_index(history):
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "user":
            return i
    return None


def generate_with_guard(messages, retry=False):
#    temp = 0.45 if retry else 0.65
    safe_messages = trim_messages(messages, max_tokens=3000)

    stream = llm.create_chat_completion(
        messages=safe_messages,
#        temperature=temp,
        stream=True
    )

    response = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        response += delta

    return response


def retry_last_response(history, persona):
    profile = profile_key(persona)

    if not history:
        return history

    last_user_idx = find_last_user_index(history)
    if last_user_idx is None:
        return history

    # ✂️ Remove everything after last user message
    new_context = history[: last_user_idx + 1]

    # 🔒 Optional recontextualization nudge
    new_context.insert(-1, {
        "role": "system",
        "content": (
            "Respond differently. "
        #    "Respond with verbosity. "
        #    "Respond with at least three paragraphs. "
            "Avoid repetition.\n"
            + load_personality_string(persona)
        )
    })

    # 🔁 Generate again (with streaming + guards)
    response = generate_with_guard(new_context)

    # Append regenerated assistant message
    new_context.append({
        "role": "assistant",
        "content": response
    })

    # 💾 Save
    save_context(profile, new_context)
    save_archive_text(profile, new_context)

    return new_context


# ==========================
# UI
# ==========================
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

            persona = gr.Dropdown(
                choices=get_personality_files(),
                label="Persona",
                value=get_personality_files()[0]
            )

            avatar_ui = gr.Image(
            value=load_avatar_for_profile(get_personality_files()[0]),
            label="Avatar",
            type="filepath",
            height=300
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

            img_btn.click(save_avatar_for_profile, [avatar_ui, persona], avatar_ui)
            save_btn.click(save_raw_yaml, [persona, yaml_box], info)

        # RIGHT COLUMN: Chat Interface
        with gr.Column(scale=3):

            # Edit to Chatbot
            chat_display = gr.Chatbot(
                elem_id="chat_column",
                label="Conversation Log",
                height=600
            )
            chat_display.retry(retry_last_response, [chat_display, persona], [chat_display])

            user_input = gr.Textbox(label="Your message")

            send_btn = gr.Button("Send")
            send_btn.click(
                fn=predict,
                inputs=[user_input, persona],
                outputs=chat_display
            )

            demo.load(
                fn=lambda p: context_to_chatbot(profile_key(p)) if p else [],
                inputs=persona,
                outputs=chat_display
            )

            persona.change(
                fn=load_chatbot_on_open,
                inputs=persona,
                outputs=chat_display
            )

            edit_box = gr.Textbox(label="Edit Message (Press Enter to Save)", visible=False)
            selected_idx = gr.State()

            # When a message is clicked, show the textbox with that text
            chat_display.select(load_for_edit, [chat_display], [edit_box, selected_idx])

            # When you press enter in the textbox, update the chatbot
            edit_box.submit(
                update_history,
                [chat_display, edit_box, selected_idx, persona],
                [chat_display, edit_box]
            )

        # UI Interactivity
        persona.change(
            fn=lambda f: (
                load_raw_yaml(f),
                load_avatar_for_profile(f),
            ),
            inputs=[persona],
            outputs=[yaml_box, avatar_ui]
        )


if __name__ == "__main__":
    demo.launch(css=css_code)
