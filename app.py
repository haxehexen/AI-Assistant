import os
import re
import json
import math
import uuid
import gradio as gr
import pickle
import shutil
import chromadb
import requests
import wikipediaapi
import customtkinter as ctk
from typing import List
from pathlib import Path
from tkinter import filedialog, messagebox
from datetime import datetime, timezone
from llama_cpp import Llama
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
from cryptography.fernet import Fernet
from sentence_transformers import SentenceTransformer

EMBED_DIM = 384
MAX_MEMORIES = 6
SIMILARITY_THRESHOLD = 0.92


# ==========================
# GLOBAL VALUES
# ==========================

#ASSETS_DIR = Path("assets")
ASSETS_PATH = Path(r"E:\PythonProject\AI Assistant\assets")
PERSONA_PATH = ASSETS_PATH / "Persona"

LLM_CONFIG = ASSETS_PATH / "llm_config.cfg"

GOOGLE_KEYS = ASSETS_PATH / "google_keys.bin"
KEY_PATH = ASSETS_PATH / "secret.key"

MAX_TURNS = 12   # editable + AI context window
MAX_TOKENS = 3000


# ==========================
# 1. CONFIGURATION
# ==========================

def mask(value):
    if not value:
        return ""
    return value[:0] + "••••••••" + value[-4:]


def get_create_key():
    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as key_file:
            key_file.write(key)
    else:
        with open(KEY_PATH, "rb") as key_file:
            key = key_file.read()
    return Fernet(key)


def load_google_keys():
    if not os.path.exists(GOOGLE_KEYS):
        return "", ""

    fernet = get_create_key()

    with open(GOOGLE_KEYS, "rb") as f:
        encrypted_data = f.read()
        decrypted_data = fernet.decrypt(encrypted_data)
        data = pickle.loads(decrypted_data)
        return data.get("GOOGLE_API_KEY", ""), data.get("GOOGLE_CX", "")


def save_google_keys(api_key, cx):
    fernet = get_create_key()

    data_to_pickle = {
        "GOOGLE_API_KEY": api_key,
        "GOOGLE_CX": cx
    }

    pickled_bytes = pickle.dumps(data_to_pickle)
    encrypted_bytes = fernet.encrypt(pickled_bytes)

    with open(GOOGLE_KEYS, "wb") as f:
        f.write(encrypted_bytes)

    status = "✅ Google API keys saved successfully"

    new_markdown = f"""
        **Current Status**
        - API Key: `{mask(api_key)}`
        - CX: `{mask(cx)}`
        """

    return status, new_markdown


# ==========================
# CORE FILE FUNCTIONS
# ==========================

yaml_ruamel = YAML()
yaml_ruamel.preserve_quotes = True


def get_persona():
    names = []
    for root, dirs, files in os.walk(ASSETS_PATH):
        for file in files:
            if file.endswith(".yaml"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = yaml_ruamel.load(f)

                        if isinstance(data, dict) and "name" in data:
                            names.append((data["name"], file))
                        else:
                            names.append(("", file))
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    return names if names else [("AIIReS", "AIIReS.yaml")]


def load_data_yaml(filename):
    file_path = (PERSONA_PATH / filename).with_suffix("")
    full_path = os.path.join(file_path, filename)

    if not filename or not os.path.exists(full_path):
        return "", "", "", "", "", "", "", ""

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml_ruamel.load(f)

    if not isinstance(data, dict):
        return "", "", "", "", "", "", "", ""

    username = (data['username'])
    name = (data['name'])
    traits = (data['traits'])
    visuals = (data['visual_description'])
    tone = (data['tone'])
    narrative = (data['narrative'])
    user_desc = (data['user_description'])
    template = (data['template'])

    return username, name, traits, visuals, tone, narrative, user_desc, template


def build_persona_data(
    username: str,
    name: str,
    traits_text: str,
    visuals: str,
    tone: str,
    narrative: str,
    user_desc: str,
    template: str,
) -> dict:

    return {
        "username": username.strip(),
        "name": name.strip(),
        "traits": LiteralScalarString(traits_text),
        "visual_description": LiteralScalarString(visuals.strip()),
        "tone": LiteralScalarString(tone.strip()),
        "narrative": LiteralScalarString(narrative.strip()),
        "user_description": LiteralScalarString(user_desc.strip()),
        "template": LiteralScalarString(template.strip()),
    }


def save_persona_yaml(
    *,
    persona_id: str | None,
    username: str,
    name: str,
    traits: str,
    visuals: str,
    tone: str,
    narrative: str,
    user_desc: str,
    template: str,
) -> str:
    try:
        persona_id = persona_id or name + uuid.uuid4().hex
        folder = PERSONA_PATH / persona_id
        folder.mkdir(parents=True, exist_ok=True)

        file_path = folder / f"{persona_id}.yaml"

        data = build_persona_data(
            username,
            name,
            traits,
            visuals,
            tone,
            narrative,
            user_desc,
            template,
        )

        with file_path.open("w", encoding="utf-8") as f:
            yaml_ruamel.dump(data, f)

        return f"✅ Persona '{name}' saved successfully"

    except Exception as e:
        return f"❌ Error saving persona: {e}"


def delete_persona_yaml(filename):
    root = ctk.CTk()
    root.attributes('-topmost', True)

    response = messagebox.askyesno("Delete Confirmation", "Do you want to delete this file?")

    if response:  # If True (Yes was clicked)
        print("File deleted.")
        # Insert your file deletion logic here
    else:  # If False (No was clicked)
        print("Deletion cancelled.")

    root.destroy()
"""
    folder_to_delete = (PERSONA_PATH / filename).with_suffix("")

    try:
        if folder_to_delete.exists():
            shutil.rmtree(folder_to_delete)

        return f"✅ Persona deleted successfully"

    except Exception as e:
        return f"❌ Error! No persona file: {e}"
"""


def create_new_yaml(**kwargs) -> str:
    return save_persona_yaml(persona_id=None, **kwargs)


def save_raw_yaml(filename: str, **kwargs) -> str:
    persona_id = Path(filename).stem
    return save_persona_yaml(persona_id=persona_id, **kwargs)


# Wrapper functions for Gradio
def create_yaml_wrapper(username, name, traits, visuals, tone, narrative, user_desc, template):
    try:
        create_new_yaml(
            username=username,
            name=name,
            traits=traits,
            visuals=visuals,
            tone=tone,
            narrative=narrative,
            user_desc=user_desc,
            template=template
        )

        return f"✅ Persona '{name}' created successfully"

    except Exception as e:
        return f"❌ Error creating persona: {e}"


def save_yaml_wrapper(filename, username, name, traits, visuals, tone, narrative, user_desc, template):
    try:
        save_raw_yaml(
            filename=filename,
            username=username,
            name=name,
            traits=traits,
            visuals=visuals,
            tone=tone,
            narrative=narrative,
            user_desc=user_desc,
            template=template
        )
        return f"✅ Persona '{name}' saved successfully"

    except Exception as e:
        return f"❌ Error saving persona: {e}"


# ==========================
# LLM CONFIG FUNCTIONS
# ==========================

def select_gguf_file():
    root = ctk.CTk()
    root.attributes('-topmost', True)

    gguf_file = filedialog.askopenfilename(
        title="Select GGUF (GPT-Generated Unified Format) file",
        filetypes=[
            ("GGUF file", "*.gguf")
        ]
    )

    root.destroy()

    if not gguf_file:
        return f"❌ Error: No GGUF file selected"
    else:
        with open(LLM_CONFIG, "r", encoding="utf-8") as f:
            data = yaml_ruamel.load(f)

        data["gguf_file"] = "model_path=" + gguf_file

        with open(LLM_CONFIG, "w", encoding="utf-8") as f:
            yaml_ruamel.dump(data, f)

            return "✅ GGUF saved successfully", f"File: {LLM_CONFIG}"


def select_db_folder():
    root = ctk.CTk()
    root.attributes('-topmost', True)

    db_folder = filedialog.askdirectory(
        title="Select ChromaDB Directory"
    )

    root.destroy()

    if not db_folder:
        return f"❌ Error: No ChromaDB directory selected"
    else:
        with open(LLM_CONFIG, "r", encoding="utf-8") as f:
            data = yaml_ruamel.load(f)

        data["vec_db_dir"] = db_folder

        with open(LLM_CONFIG, "w", encoding="utf-8") as f:
            yaml_ruamel.dump(data, f)

            return "✅ ChromaDB directory saved successfully", f"File: {LLM_CONFIG}"


def get_gguf_file():
    with open(LLM_CONFIG, "r", encoding="utf-8") as f:
        data = yaml_ruamel.load(f)

    return data["gguf_file"]


def get_vec_db():
    with open(LLM_CONFIG, "r", encoding="utf-8") as f:
        data = yaml_ruamel.load(f)

    return data["vec_db_dir"]


def save_llm_config(config_text):
    with open(LLM_CONFIG, "r", encoding="utf-8") as f:
        data = yaml_ruamel.load(f)

    if not data["gguf_file"]:
        return f"❌ Error no GGUF file selected"
    else:
        data["llm_settings"] = config_text

        with open(LLM_CONFIG, "w", encoding="utf-8") as f:
            yaml_ruamel.dump(data, f)

        return "✅ LLM Config saved successfully", f"File: {LLM_CONFIG}"


def parse_config_string(cfg_str):
    config_dict = {}
    pattern = re.compile(r"(\w+)\s*=\s*([^,\s#]+)")

    for line in cfg_str.splitlines():
        match = pattern.search(line)
        if match:
            key, val = match.groups()

            if val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False
            else:
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass
            config_dict[key] = val
    return config_dict


def load_llm_config():
    with open(LLM_CONFIG, "r", encoding="utf-8") as f:
        data = yaml_ruamel.load(f)

    llm_settings = data["llm_settings"]
    llm_config = data["gguf_file"] + "\n" + data['llm_settings']

    return [(llm_settings, parse_config_string(llm_config))]


# Initialize AI Backend
config = load_llm_config()[0][1]
llm = Llama(**(load_llm_config()[0][1]))


# ==========================
# CORE AVATAR FUNCTIONS
# ==========================

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def save_avatar_for_profile(image_path: str | Path, filename: str) -> Path:
    source = Path(image_path)

    if not source.exists():
        raise FileNotFoundError(f"Avatar image not found: {source}")

    if source.suffix.lower() not in IMAGE_EXT:
        raise ValueError(f"Unsupported image format: {source.suffix}")

    persona_id = Path(filename).stem
    dest_dir = PERSONA_PATH / persona_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    new_name = f"{persona_id}{source.suffix.lower()}"
    dest_path = dest_dir / new_name

    # Atomic replace if exists
    shutil.copy2(source, dest_path)

    return dest_path

    # 🔥 NEW: Vision extraction / Mothballed
###    visual_profile = extract_visual_profile_cli(dest_path)
###    update_yaml_with_visual_profile(yaml_filename, visual_profile)


def get_image_path(filename):
    if not filename: return None

    base_name = Path(filename).stem
    file_path = os.path.join(PERSONA_PATH, base_name)
    extensions = ['.png', '.jpg', '.jpeg', '.webp']

    for ext in extensions:
        potential_path = os.path.join(file_path, f"{base_name}{ext}")

        if os.path.exists(potential_path):
            return potential_path


def load_avatar_for_profile(filename):
    path = get_image_path(filename)
    return path if path and os.path.exists(path) else None


# ==========================
# HISTORY FILE HELPERS
# ==========================

def profile_key(yaml_file: str) -> str:
    return Path(yaml_file).stem.lower()


def context_path(profile: str) -> Path:
    return PERSONA_PATH / profile / f"{profile}_context.json"


def archive_path(profile: str) -> Path:
    return PERSONA_PATH / profile / f"{profile}__archive.txt"


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
    path = archive_path(profile)

    with open(path, "w", encoding="utf-8") as f:
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
    file_path = (PERSONA_PATH / filename).with_suffix("")
    full_path = os.path.join(file_path, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Personality YAML not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml_ruamel.load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid YAML structure")

    # Required fields
    template = data.get("template")
    if not template:
        raise ValueError("Missing 'template' in YAML")

    name = data.get("name", "AIIReS")
    username = data.get("username", "User")
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
# MESSAGE HELPERS
# ==========================

def new_message(role, content):
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }


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
    api_key, cx = load_google_keys()

    if not api_key or not cx:
        return "❌ Google API Key or CX not set"

    url = "https://www.googleapis.com/customsearch/v1"

    print(f"DEBUG: Searching Google for: {query}")

    params = {
        "key": api_key,
        "cx": cx,
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
# CORE LONG-TERM MEMORY
# ==========================

class MemoryManager:
    def __init__(self, persona_name: str, device="cuda"):
        self.persona = persona_name.lower()

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device if device == "cuda" else "cpu",
            local_files_only=True
        )

        self.client = chromadb.PersistentClient(
            path=f"./chromadb_memory/{self.persona}"
        )

        self.collection_name = f"memory_{self.persona}"

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # ------------------------
    # Public API
    # ------------------------

    def store_memory(self, text: str, mem_type="general"):
        importance = self._calculate_importance(text)
        if importance <= 0.2:
            return

        embedding = self._embed(text)
        now = datetime.utcnow().isoformat()

        existing = self._find_similar(embedding)

        if existing:
            self._update_existing(existing, importance, now)
            return

        metadata = {
            "type": mem_type,
            "importance": importance,
            "created": now,
            "last_accessed": now,
        }

        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )


    def retrieve_memories(self, query: str, n=5, min_score=0.15):
        now = datetime.now(timezone.utc)
        query_embedding = self._embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n * 3  # over-fetch, then decay-filter
        )

        memories = []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        scored = []

        for mem_id, text, meta in zip(ids, docs, metas):
            score = self._decay_score(meta, now)
            if score >= min_score:
                scored.append((score, mem_id, text, meta))

        scored.sort(reverse=True, key=lambda x: x[0])
        scored = scored[:n]

        for _, mem_id, text, meta in scored:
            memories.append(text)
            meta["last_accessed"] = now.isoformat()

            self.collection.update(
                ids=[mem_id],
                metadatas=[meta]
            )

        return memories


    def _decay_score(self, meta, now, decay_lambda=0.05):
        last = datetime.fromisoformat(meta["last_accessed"])

        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)

        age_days = (now - last).total_seconds() / 86400
        importance = meta.get("importance", 0.3)

        return importance * math.exp(-decay_lambda * age_days)


    def prune_memories(self, hard_threshold=0.05):
        now = datetime.now(timezone.utc)
        all_data = self.collection.get()

        for mem_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            if self._decay_score(meta, now) < hard_threshold:
                self.collection.delete(ids=[mem_id])


    # ------------------------
    # Internal Helpers
    # ------------------------

    def _embed(self, text: str) -> List[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()


    def _find_similar(self, embedding, threshold=0.92):
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=1
        )

        if not result["ids"][0]:
            return None

        distance = result["distances"][0][0]
        similarity = 1 - distance

        if similarity < threshold:
            return None

        return {
            "id": result["ids"][0][0],
            "metadata": result["metadatas"][0][0]
        }


    def _update_existing(self, existing, importance, now):
        meta = existing["metadata"]
        meta["importance"] = max(meta.get("importance", 0), importance)
        meta["last_accessed"] = now

        self.collection.update(
            ids=[existing["id"]],
            metadatas=[meta]
        )


    def _calculate_importance(self, text: str) -> float:
        length_factor = min(len(text) / 200, 1.0)
        keyword_bonus = 0.2 if any(
            kw in text.lower()
            for kw in ["important", "remember", "note", "always"]
        ) else 0.0

        return min(1.0, 0.3 + length_factor + keyword_bonus)


def summarize_for_memory(user_input: str, assistant_reply: str) -> str:
    prompt = f"""
You are a memory extraction system for an AI assistant.

Your task:
- Extract ONLY long-term memory worthy information.
- Focus on: identity, preferences, decisions, goals, commitments, corrections.
- Ignore questions, commands, temporary emotions, or general discussion.
- Write ONE concise sentence.
- If nothing important exists, return EXACTLY: NONE

User message:
{user_input}

Assistant reply:
{assistant_reply}

Long-term memory summary:
""".strip()

    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You extract long-term memory for an AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    summary = result["choices"][0]["message"]["content"].strip()

    # Hard safety checks
    if not summary:
        return "NONE"

    if summary.upper() == "NONE":
        return "NONE"

    # Avoid storing questions or instructions
    if "?" in summary:
        return "NONE"

    # Avoid extremely long summaries
    if len(summary) > 300:
        return "NONE"

    return summary


# ==========================
# CORE CHAT LOGIC
# ==========================

def predict(user_input, persona):
    profile = profile_key(persona)
    context = load_context(profile)

    memory = MemoryManager(persona, device="cuda")

    if not any(m["role"] == "system" for m in context):
        context.append(
            new_message("system", load_personality_string(persona))
        )

    runtime_context = []
    sources = []

    # ------------------------
    # Memory retrieval
    # ------------------------
    memories = memory.retrieve_memories(user_input, n=5)

    if memories:
        runtime_context.append(
            new_message(
                "system",
                "The assistant has the following long-term memories:\n"
                + "\n".join(f"- {m}" for m in memories)
            )
        )

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
    else:
        google_text, google_url = "", None

    if any (keyword in user_input.lower() for keyword in wiki_keywords):
        title = extract_wiki_title_llm(query)
        wiki_text, wiki_url = wiki_search(title)
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

    combined = trim_context(runtime_context + context)

    response = llm.create_chat_completion(
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in combined
        ]
    )["choices"][0]["message"]["content"]

    assistant_msg = new_message("assistant", response)
    context.append(assistant_msg)

    # ------------------------
    # Store memory
    # ------------------------
    memory_text = summarize_for_memory(user_input, response)
    if memory_text:
        memory.store_memory(memory_text)

    # Optional decay cleanup
    memory.prune_memories()

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

