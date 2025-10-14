import os
import re
import json
import time
import random
import requests
import faiss
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer, util
from playwright.sync_api import sync_playwright

# === INPUT ===
START_URL = "https://docs.github.com"
TASK = "how to create a composite action and how to use it in reusable workflow"

# === CONFIG ===
OUTPUT_DIR = "ai_filtered_pages"
JSONL_FILE = "scraped_data.jsonl"
MODEL_EMBED = "all-MiniLM-L6-v2"
OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
CHUNK_SIZE = 500
TOP_K = 5

# === UTILS ===
def sanitize_filename(url):
    path = urlparse(url).path.strip("/")
    filename = re.sub(r"[^\w\-_.]", "_", path) or "home"
    return filename[:80] + ".txt"

def is_internal_link(base_url, link):
    base_domain = urlparse(base_url).netloc
    link_domain = urlparse(link).netloc
    return link_domain == "" or link_domain == base_domain

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    content = []

    # Extract headings
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        content.append("\n" + heading.get_text(strip=True))

    # Extract paragraphs
    for para in soup.find_all("p"):
        content.append(para.get_text(strip=True))

    # Extract code blocks
    for code_block in soup.find_all(["pre", "code"]):
        code_text = code_block.get_text(strip=True)
        if code_text and len(code_text.splitlines()) <= 100:  # prevent huge dumps
            content.append("\nbash\n" + code_text + "\n")

    return "\n".join(content)

def get_links_from_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for tag in soup.find_all("a", href=True):
        text = (tag.text or "").strip()
        href = tag["href"]
        full_url = urljoin(base_url, href).split("#")[0]
        if is_internal_link(base_url, full_url) and len(text) > 3:
            links.add((text, full_url))
    return list(links)

# === STEP 1: Crawl relevant pages
def crawl_ai_relevant_pages(start_url, task, output_dir, max_links=15):
    os.makedirs(output_dir, exist_ok=True)
    visited = set()
    model = SentenceTransformer(MODEL_EMBED)
    task_embedding = model.encode(task, convert_to_tensor=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print("\n🔍 Discovering homepage links...")
        try:
            page.goto(start_url, timeout=10000)
            page.wait_for_load_state("networkidle")
            html = page.content()
        except Exception as e:
            print(f"❌ Could not load {start_url}: {e}")
            return

        all_links = get_links_from_html(html, start_url)
        texts = [t for t, _ in all_links]
        embeds = model.encode(texts, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(task_embedding, embeds)[0]
        ranked_links = sorted(zip(sims.tolist(), all_links), reverse=True)[:max_links]
        top_links = [link for _, (_, link) in ranked_links]

        child_candidates = []
        for link in top_links:
            if link in visited:
                continue
            try:
                page.goto(link, timeout=10000)
                page.wait_for_load_state("networkidle")
                html = page.content()
                text = extract_text(html)
                filename = sanitize_filename(link)
                with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                    f.write(f"URL: {link}\n\nTask: {task}\n\n{text}")
                visited.add(link)
                child_candidates += get_links_from_html(html, link)
                time.sleep(random.uniform(1.0, 2.2))
            except Exception as e:
                print(f"⚠️ Skipping {link}: {e}")
                continue

        print("\n🔁 Fetching child relevant links...")
        child_texts = [t for t, _ in child_candidates]
        child_embeds = model.encode(child_texts, convert_to_tensor=True)
        child_sims = util.pytorch_cos_sim(task_embedding, child_embeds)[0]
        child_ranked = sorted(zip(child_sims.tolist(), child_candidates), reverse=True)[:max_links]
        child_top = [link for _, (_, link) in child_ranked]

        for link in child_top:
            if link in visited:
                continue
            try:
                page.goto(link, timeout=10000)
                page.wait_for_load_state("networkidle")
                html = page.content()
                text = extract_text(html)
                filename = sanitize_filename(link)
                with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                    f.write(f"URL: {link}\n\nTask: {task}\n\n{text}")
                visited.add(link)
                time.sleep(random.uniform(1.0, 2.2))
            except Exception as e:
                print(f"⚠️ Skipping {link}: {e}")
                continue

        browser.close()
        print(f"\n✅ Crawled {len(visited)} pages.")

# === STEP 2: Convert to JSONL with chunking
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def extract_meta_and_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    url = re.search(r"URL:\s*(https?://[^\s]+)", content)
    task = re.search(r"Task: (.*)", content)
    body = clean_text(content.split("\n\n", 2)[-1])
    return url.group(1) if url else "unknown", task.group(1) if task else "unspecified", body

def convert_to_jsonl(input_folder, jsonl_file):
    data = []
    for fname in os.listdir(input_folder):
        if not fname.endswith(".txt"):
            continue
        url, task, text = extract_meta_and_text(os.path.join(input_folder, fname))
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                data.append({"url": url, "task": task, "chunk_id": i, "text": chunk})
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")
    print(f"✅ Converted to {jsonl_file} ({len(data)} chunks)")
    return data

# === STEP 3: Embed and FAISS index
def build_index(data):
    model = SentenceTransformer(MODEL_EMBED)
    texts = [entry["text"] for entry in data]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model

# === STEP 4: RAG + Query LLM
def create_messages(task, question, top_chunks):
    context = "\n---\n".join([c["text"] for c in top_chunks])
    return [
        {"role": "system", "content": "You are a coding assistant that only uses provided documentation."},
        {"role": "user", "content": f"Task: {task}\n\nDocs:\n{context}\n\nQuestion: {question}"}
    ]

def search(question, index, model, data):
    query_embed = model.encode([question], convert_to_numpy=True)
    _, indices = index.search(query_embed, TOP_K)
    return [data[i] for i in indices[0]]

def query_ollama(messages):
    response = requests.post(OLLAMA_API, json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    })
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return f"❌ Ollama error: {response.text}"

# === MAIN ===
if __name__ == "__main__":
    print("📥 Step 1: Crawling relevant docs...")
    crawl_ai_relevant_pages(START_URL, TASK, OUTPUT_DIR)

    print("\n📦 Step 2: Converting to JSONL...")
    all_data = convert_to_jsonl(OUTPUT_DIR, JSONL_FILE)

    print("\n📐 Step 3: Embedding and indexing...")
    index, embed_model = build_index(all_data)

    question = TASK  # No input prompt
    top_chunks = search(question, index, embed_model, all_data)
    messages = create_messages(TASK, question, top_chunks)

    print("\n📤 Step 4: Sending to LLM...\n")
    answer = query_ollama(messages)
    print("\n🧠 Answer:\n", answer)
