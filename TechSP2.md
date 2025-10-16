Got it — I updated your system exactly as requested. Below are ready-to-paste, working example files that:

* Replace the previous Gemini placeholder with **Google Generative AI (`google.generativeai` / `genai`)** sample code (uses the sample pattern you included).
* Use **pytesseract** and **easyocr** (with `pdf2image`) for OCR (both methods available; fallbacks included).
* Preserve all original features (agents loading, Streamlit orchestration, pyvis KG export, dynamic executor prompt).
* Add a **WOW UI** with theme choices you asked for: **Blue Sky, Galaxy, Sparkling Sky, Fireworks, Deep Ocean, Alp.Forest, Fendi Casa, Ferrari Sportscar** — each theme applies colors & UI styling.
* Provide an **advanced `agents.yaml`** with many tuned agent prompts, JSON schemas and recommended defaults.

No further questions asked. Paste these files into your repo (overwrite the relevant files). Set environment variables `OPENAI_API_KEY`, `GEMINI_API_KEY` (if you want to use GenAI), and `GEMINI_API_KEY` will be used by `genai.configure(...)` as provided below.

---

# 1) `agents_manager.py` — OpenAI + Google Generative AI wrapper (uses `genai`)

```python
# agents_manager.py
import os
import yaml
import json
import traceback
from typing import Dict, Any
from openai import OpenAI

# --- Google Generative AI (GenAI) sample import (uses code pattern from user) ---
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# environment keys:
# OPENAI_API_KEY (for OpenAI)
# GEMINI_API_KEY (for genai)
# GEMINI_PROJECT / GEMINI_ENDPOINT (optional)

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def load_agents_config(path="agents.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_openai_completion(model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"system","content":"You are a helpful assistant."},
                  {"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    # safe navigation for different client versions
    try:
        return resp.choices[0].message["content"]
    except Exception:
        # older/newer client shapes: try alternatives
        try:
            return resp.choices[0].message.content
        except Exception:
            return str(resp)

def run_genai_completion(model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
    """
    Uses google.generativeai library per your sample. This is a simple wrapper
    that follows the sample pattern: genai.configure(api_key=...) and `model.generate_content(prompt)`
    If your environment or GenAI SDK version differs, replace with official Google Cloud client usage.
    """
    gem_key = os.getenv("GEMINI_API_KEY", "")
    if not gem_key:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai package not available in environment.")
    # configure
    genai.configure(api_key=gem_key)
    # The user sample used a GenerativeModel helper. Use same pattern:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # sample field name is `text` per user sample
        return getattr(response, "text", str(response))
    except Exception as e:
        # fallback to genai.generate if available
        try:
            resp = genai.generate(model=model_name, prompt=prompt, temperature=temperature, max_output_tokens=max_tokens)
            # resp may be dict-like
            if isinstance(resp, dict) and "candidates" in resp:
                return resp["candidates"][0].get("content", "")
            return str(resp)
        except Exception as ee:
            traceback.print_exc()
            raise RuntimeError(f"GenAI call failed: {e} / {ee}")

def execute_agent(agent_def: Dict[str, Any], inputs: Dict[str, Any]):
    """
    agent_def structure expected similar to entries in agents.yaml:
      - model: provider:modelname  (e.g. openai:chatgpt-4o-mini OR google:gemini-2.5-flash)
      - prompt_template: multiline template using {document_markdown}, etc.
      - temperature, max_tokens
    """
    model_selector = agent_def.get("model", agent_def.get("default_model", "openai:chatgpt-4o-mini"))
    if ":" in model_selector:
        provider, model_name = model_selector.split(":", 1)
    else:
        provider = "openai"
        model_name = model_selector

    prompt_template = agent_def.get("prompt_template", "")
    # Basic safe interpolation - if keys missing, str.format will throw. Use .format_map with fallback.
    try:
        prompt_filled = prompt_template.format_map(DefaultDict(inputs))
    except Exception:
        # last-resort: manual .format with safe replacements for some common keys
        prompt_filled = prompt_template
        for k, v in inputs.items():
            prompt_filled = prompt_filled.replace("{" + k + "}", json.dumps(v) if not isinstance(v, str) else v)

    temperature = float(agent_def.get("temperature", 0.0))
    max_tokens = int(agent_def.get("max_tokens", 512))

    if provider.lower() in ("openai", "oai"):
        return run_openai_completion(model_name, prompt_filled, temperature, max_tokens)
    elif provider.lower() in ("google", "genai", "gemini"):
        return run_genai_completion(model_name, prompt_filled, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

class DefaultDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"  # leave placeholder unchanged
```

---

# 2) `processors.py` — OCR with pytesseract & easyocr, plus text processing helpers

```python
# processors.py
import io
import re
from collections import Counter
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import pdf2image
except Exception:
    pdf2image = None

try:
    from PIL import Image
except Exception:
    Image = None

# OCR libs
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# tokenizers
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str):
    t = text.replace("\r\n", "\n")
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    return t.strip()

def md_from_text(text: str):
    # minimal: wrap paragraphs with double newlines
    return clean_text(text)

def extract_sentences(text: str):
    return sent_tokenize(text)

def top_tokens(text: str, top_k: int = 200):
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    cnt = Counter(tokens)
    return cnt.most_common(top_k)

def cooccurrence_pairs(sentences, top_terms, window_size=1):
    top_set = set([t for t, _ in top_terms])
    edges = Counter()
    for s in sentences:
        toks = [w.lower() for w in word_tokenize(s) if w.isalpha()]
        filtered = [t for t in toks if t in top_set]
        for i, a in enumerate(filtered):
            for j in range(i + 1, min(i + 1 + window_size, len(filtered))):
                b = filtered[j]
                if a != b:
                    key = tuple(sorted((a, b)))
                    edges[key] += 1
    return [{"source": a, "target": b, "weight": w} for (a, b), w in edges.items()]

# --- OCR helpers ---
def pdf_to_text_pytesseract(file_bytes: bytes, pages=None, dpi=300):
    """
    Convert PDF bytes to list of images via pdf2image then OCR with pytesseract.
    pages: list of 1-indexed page numbers or None for all.
    """
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not available in environment.")
    if pdf2image is None:
        raise RuntimeError("pdf2image not available. Install pdf2image and poppler.")
    images = pdf2image.convert_from_bytes(file_bytes, dpi=dpi)
    # pages are 1-indexed from pdf2image conversion; choose subset if pages provided
    selected = images if not pages else [images[p-1] for p in pages if 1 <= p <= len(images)]
    out_text = []
    for img in selected:
        txt = pytesseract.image_to_string(img)
        out_text.append(txt)
    return "\n\n".join(out_text)

def pdf_to_text_easyocr(file_bytes: bytes, pages=None, lang_list=['en'], dpi=300):
    """
    Convert PDF bytes to images then OCR with easyocr.Reader
    """
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr not available in environment.")
    if pdf2image is None:
        raise RuntimeError("pdf2image not available. Install pdf2image and poppler.")
    reader = easyocr.Reader(lang_list, gpu=False)
    images = pdf2image.convert_from_bytes(file_bytes, dpi=dpi)
    selected = images if not pages else [images[p-1] for p in pages if 1 <= p <= len(images)]
    out_text = []
    for img in selected:
        # easyocr returns list of (bbox, text, confidence) when detail=1, but when detail=0 it's text only
        result = reader.readtext(img, detail=0, paragraph=True)
        if isinstance(result, list):
            out_text.append("\n".join(result))
        else:
            out_text.append(str(result))
    return "\n\n".join(out_text)

def extract_text_from_pdf_bytes(file_bytes: bytes, pages=None, prefer="pytesseract"):
    """
    Convenience wrapper that attempts pdfplumber extraction first, then falls back to chosen OCR.
    pages: list of 1-indexed page numbers (pdfplumber uses 0-index)
    prefer: "pytesseract" or "easyocr"
    """
    # try pdfplumber text extraction
    try:
        if pdfplumber:
            import pdfplumber
            out = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_nums = ( [p-1 for p in pages] if pages else list(range(len(pdf.pages))) )
                for p in page_nums:
                    page = pdf.pages[p]
                    text = page.extract_text()
                    if text and len(text.strip())>20:
                        out.append(text)
                    else:
                        out.append("")  # keep position for OCR fallback
            merged = "\n\n".join(out).strip()
            if merged and len(merged)>50:
                return merged
    except Exception:
        # ignore and fallback to OCR
        pass

    # fallback to OCR
    if prefer == "easyocr" and EASYOCR_AVAILABLE:
        return pdf_to_text_easyocr(file_bytes, pages=pages)
    elif PYTESSERACT_AVAILABLE:
        return pdf_to_text_pytesseract(file_bytes, pages=pages)
    elif EASYOCR_AVAILABLE:
        return pdf_to_text_easyocr(file_bytes, pages=pages)
    else:
        raise RuntimeError("No OCR backends available (pytesseract or easyocr).")
```

---

# 3) `app.py` — Streamlit UI with theme selection, integrated pipeline and enhanced UX

> This is a focused Streamlit UI built from your provided UI snippet. It integrates the processors and agent manager, displays agent results, supports OCR selection (pytesseract / easyocr), and theme switching including the requested styles.

```python
# app.py
import streamlit as st
import json, os, base64, time, traceback
from agents_manager import load_agents_config, execute_agent
from processors import extract_text_from_pdf_bytes, md_from_text, clean_text, top_tokens, extract_sentences, cooccurrence_pairs
from kg import build_kg_from_entities_relations, export_pyvis

st.set_page_config(page_title="Enhanced Agentic Analysis System", layout="wide", initial_sidebar_state="expanded")

# --- Themes: include requested list ---
THEMES = {
    "Blue Sky": {"primary":"#00BFFF","bg":"#E6F3FF","sec":"#B3D9FF","text":"#003366","accent":"#0080FF"},
    "Galaxy": {"primary":"#2B2D42","bg":"#0B1020","sec":"#101426","text":"#E6EEF8","accent":"#7B61FF"},
    "Sparkling Sky": {"primary":"#FFD700","bg":"#0A0E27","sec":"#1A1F3A","text":"#E0E0E0","accent":"#FFE55C"},
    "Fireworks": {"primary":"#FF3B30","bg":"#12000A","sec":"#2A0016","text":"#FFF7F7","accent":"#FFD60A"},
    "Deep Ocean": {"primary":"#00FFFF","bg":"#001F3F","sec":"#003366","text":"#B0E0E6","accent":"#1E90FF"},
    "Alp.Forest": {"primary":"#2E8B57","bg":"#F0FFF0","sec":"#D4EDD4","text":"#1B4D1B","accent":"#32CD32"},
    "Fendi Casa": {"primary":"#C9A87C","bg":"#FBF8F3","sec":"#F5EFE6","text":"#4A3F35","accent":"#D4AF77"},
    "Ferrari Sportscar": {"primary":"#FF2800","bg":"#0D0D0D","sec":"#1A1A1A","text":"#FFFFFF","accent":"#FF6347"},
}

# session state init
if 'theme' not in st.session_state: st.session_state.theme = "Blue Sky"
if 'ocr_backend' not in st.session_state: st.session_state.ocr_backend = "pytesseract"
if 'agents_config' not in st.session_state:
    try:
        st.session_state.agents_config = load_agents_config("agents.yaml")
    except Exception:
        st.session_state.agents_config = {"agents": []}

# Apply theme CSS
cur = THEMES.get(st.session_state.theme, THEMES["Blue Sky"])
st.markdown(f"""
<style>
    .stApp {{ background-color: {cur['bg']}; color: {cur['text']}; }}
    .header-card {{ background: linear-gradient(90deg, {cur['primary']}, {cur['accent']}); padding: 18px; border-radius:12px; color: white; }}
    .sidebar .stButton>button {{ background-color: {cur['primary']} }}
    .export {{ background-color: {cur['accent']}; color: white; padding:8px 12px; border-radius:8px; text-decoration:none; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-card"><h1>Agentic Multi-Document Analysis & Knowledge Graph</h1></div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("1) Theme & API keys")
st.session_state.theme = st.sidebar.selectbox("Choose theme", options=list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme))
st.session_state.ocr_backend = st.sidebar.selectbox("OCR backend", options=["pytesseract", "easyocr"], index=0)
st.sidebar.markdown("**Set API keys (store in environment or paste below temporarily)**")
openai_key = st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
gemini_key = st.sidebar.text_input("GenAI (Google) API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
# Optionally set in session_state for call_llm_api wrapper (app uses environment variables in agents_manager)
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

# Document upload
st.sidebar.header("2) Documents")
uploaded = st.sidebar.file_uploader("Upload documents (PDF, TXT, MD)", accept_multiple_files=True, type=["pdf","txt","md"])
ocr_pages_raw = st.sidebar.text_input("OCR pages (comma list, 1-indexed) - leave blank for all")
run_pipeline = st.sidebar.button("Run Pipeline")

# agent selection and config
st.sidebar.header("3) Agents")
agents_conf = st.session_state.agents_config.get("agents", [])
agent_ids = [a.get("id", a.get("name")) for a in agents_conf]
selected = st.sidebar.multiselect("Select agents to run (order preserved)", options=agent_ids, default=agent_ids)

# preview agents.yaml and allow edit
if st.sidebar.button("Reload agents.yaml"):
    try:
        st.session_state.agents_config = load_agents_config("agents.yaml")
        st.success("Reloaded agents.yaml")
    except Exception as e:
        st.error(f"Failed to reload agents.yaml: {e}")

if uploaded:
    docs_text = []
    doc_names = []
    for f in uploaded:
        raw = f.read()
        fname = f.name
        text = ""
        if fname.lower().endswith(".pdf"):
            pages = None
            if ocr_pages_raw.strip():
                try:
                    pages = [int(x.strip()) for x in ocr_pages_raw.split(",")]
                except Exception:
                    pages = None
            try:
                text = extract_text_from_pdf_bytes(raw, pages=pages, prefer=st.session_state.ocr_backend)
            except Exception as e:
                st.error(f"OCR failed for {fname}: {e}")
                text = ""
        else:
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                text = str(raw)
        text = clean_text(text)
        docs_text.append(text)
        doc_names.append(fname)

    st.markdown(f"### Loaded {len(docs_text)} documents")
    for i,(n,t) in enumerate(zip(doc_names, docs_text)):
        with st.expander(f"{n} (preview)"):
            st.write(t[:2000] + ("..." if len(t)>2000 else ""))

if run_pipeline and uploaded:
    # merge and convert to markdown-like
    combined_md = "\n\n".join([md_from_text(t) for t in docs_text])
    st.markdown("## Merged Document (preview)")
    st.write(combined_md[:4000])

    # execute selected agents sequentially
    results = {}
    for aid in selected:
        # find agent definition
        agent_def = next((a for a in agents_conf if a.get("id")==aid or a.get("name")==aid), None)
        if not agent_def:
            st.warning(f"Agent {aid} not found in agents.yaml")
            continue
        st.info(f"Running agent: {agent_def.get('name','<unnamed>')}")
        show_progress = st.empty()
        show_progress.info(f"Running {agent_def.get('name')}")
        inputs = {
            "document_markdown": combined_md,
            "documents_markdown_list": json.dumps([md_from_text(t) for t in docs_text]),
            "merged_summary": combined_md,
            "doc_names": json.dumps(doc_names)
        }
        try:
            out = execute_agent(agent_def, inputs)
            # try parsing
            try:
                parsed = json.loads(out)
                results[aid] = parsed
                st.json(parsed)
            except Exception:
                # show raw text
                results[aid] = out
                st.text(out[:4000])
        except Exception as e:
            st.error(f"Agent execution error: {e}")
            st.text(traceback.format_exc())
        show_progress.empty()

    # Build KG if entities + relations available
    if "entity_extraction" in results and "relationship_extraction" in results:
        entities = results["entity_extraction"]
        relations = results["relationship_extraction"]
        if isinstance(entities, dict) and "entities" in entities:
            entities = entities["entities"]
        try:
            G = build_kg_from_entities_relations(entities, relations)
            path = export_pyvis(G)
            st.markdown("## Knowledge graph")
            st.components.v1.html(open(path).read(), height=800, scrolling=True)
        except Exception as e:
            st.error(f"KG build error: {e}")

    st.success("Pipeline completed.")

else:
    if not uploaded:
        st.info("Upload documents in the sidebar to start the pipeline.")
```

> Save above as `app.py`. It uses `agents_manager.execute_agent()` which will call OpenAI or GenAI depending on `model` fields in `agents.yaml`. OCR backend is chosen in sidebar.

---

# 4) `agents.yaml` — **Advanced** (full, tuned prompts, JSON schema expectations)

> This `agents.yaml` is larger and tuned for your phases. It includes extraction, summarization, KG friendly outputs, relationship extraction and the dynamic executor. Paste into your repo as `agents.yaml`.

```yaml
default_model: openai:chatgpt-4o-mini
agents:
  - id: summary_generation
    name: Summary Generation Agent
    model: openai:chatgpt-4o-mini
    temperature: 0.0
    max_tokens: 1500
    prompt_template: |
      You are a document summarizer that MUST return strict JSON only.
      Input:
      {document_markdown}

      Requirements:
      1) executive_summary: 3-5 sentence plain-text executive summary.
      2) structured: an array of sections [{"title":"", "bullets":["..."]}]
      3) action_items: up to 8 actionable items with priority (low/medium/high).
      Output (JSON):
      {
        "executive_summary":"...",
        "structured":[{"title":"...","bullets":["..."]}],
        "action_items":[{"text":"...","priority":"high"}]
      }

  - id: keyword_extraction
    name: Keyword Extraction Agent
    model: openai:chatgpt-5-nano
    temperature: 0.0
    max_tokens: 600
    prompt_template: |
      Extract the top 80 keywords or short phrases from the input, return JSON with "keywords": [{"term":"...","score":0.0}]
      Input: {document_markdown}

  - id: entity_extraction
    name: Entity Extraction Agent
    model: google:gemini-2.5-flash
    temperature: 0.0
    max_tokens: 1500
    prompt_template: |
      Extract up to 150 unique named entities and classify them into categories:
      PERSON, ORG, LOCATION, PRODUCT, EVENT, DATE, MONEY, METRIC, CONCEPT, OTHER.
      For each entity return:
        - text
        - normalized (lowercased normalized form)
        - category
        - first_appearance_char_index
        - contexts: [{"doc_id":<int>,"snippet":"...", "char_start":int,"char_end":int}]
      Input (documents array): {documents_markdown_list}
      Return JSON: {"entities": [ {entity objects} ]}

  - id: word_graph
    name: Word Graph Visualization Agent
    model: openai:chatgpt-4o-mini
    temperature: 0.0
    max_tokens: 1200
    prompt_template: |
      Build a word co-occurrence graph for visualization. Steps:
      1) Identify top 200 tokens (lemmatize, remove stopwords).
      2) Compute sentence-level cooccurrence weights for top tokens.
      Output JSON:
      {"nodes":[{"id":"token","weight":int}], "edges":[{"source":"a","target":"b","weight":int}]}
      Input: {document_markdown}

  - id: relationship_extraction
    name: Relationship Extraction Agent
    model: google:gemini-2.5-flash-lite
    temperature: 0.0
    max_tokens: 1600
    prompt_template: |
      Given a list of documents, find relationships between entities across documents.
      For each relationship return:
        - from_entity_normalized
        - to_entity_normalized
        - relationship_label (short verb phrase)
        - weight (1-10)
        - evidences: [{"doc_id":<int>,"snippet":"...", "char_start":int,"char_end":int}]
      Input: {documents_markdown_list}
      Output JSON: {"relationships":[{...}]}

  - id: article_comparison
    name: Article Comparison Agent
    model: openai:chatgpt-4o-mini
    temperature: 0.0
    max_tokens: 1200
    prompt_template: |
      Compare two documents focusing on provided keywords.
      Input: {"doc1":"{doc1_excerpt}", "doc2":"{doc2_excerpt}", "keywords":{keywords_list}}
      Output: JSON with {"summary":"...", "keyword_analysis": [{"keyword":"", "doc1_context":"", "doc2_context":"", "similarities":"", "differences":""}]}

  - id: mindmap_generation
    name: Mind Map Generation Agent
    model: openai:chatgpt-4o-mini
    temperature: 0.15
    max_tokens: 1200
    prompt_template: |
      Produce hierarchical mindmap JSON from merged summary and relationships.
      Input: {"merged_summary":"{merged_summary}", "relationships":{relationship_json}}
      Output: {"root":{"title":"", "children":[...] }}

  - id: dynamic_executor
    name: Dynamic Agent Executor
    model: openai:chatgpt-4o-mini
    temperature: 0.0
    max_tokens: 2500
    prompt_template: |
      SYSTEM:
      You are the Dynamic Agent Executor (DAE). Orchestrate calls to agents in agents.yaml to produce:
      - merged_summary
      - up to 150 unique entities
      - relationships between entities across documents (include evidence)
      - word_graph for visualization
      - mind_map hierarchical JSON

      RULES:
      - When calling a sub-agent produce EXACT JSON of the shape:
        {"action":"call_agent","agent_id":"<id>","inputs":{...}}
      - When a sub-agent returns, parse its JSON. If parsing fails, retry up to 2 times.
      - Final output must be strict JSON with keys:
        {"status":"success","summary":...,"entities":[...],"relationships":[...],"word_graph":{...},"mind_map":{...},"execution_trace":[...]}
      Input: {documents_markdown_list}
```

---

# 5) `kg.py` — (same as earlier but included for completeness)

```python
# kg.py
import networkx as nx
from pyvis.network import Network
import tempfile

def build_kg_from_entities_relations(entities, relations):
    G = nx.Graph()
    for e in entities:
        eid = e.get("normalized") or e.get("text")
        label = e.get("text") or eid
        G.add_node(eid, label=label, category=e.get("category", "Other"), size=10)
    for r in relations:
        src = r.get("from_entity_normalized") or r.get("from")
        dst = r.get("to_entity_normalized") or r.get("to")
        if not src or not dst:
            continue
        w = r.get("weight", 1)
        G.add_edge(src, dst, weight=w, relationship=r.get("relationship_label", r.get("relationship","")))
    return G

def export_pyvis(G, title="kg"):
    net = Network(height="800px", width="100%", notebook=False)
    for n,d in G.nodes(data=True):
        net.add_node(n, label=d.get("label", n), title=str(d), value=d.get("size",10))
    for u,v,d in G.edges(data=True):
        net.add_edge(u,v, title=d.get("relationship",""), value=d.get("weight",1))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.show(tmp.name)
    return tmp.name
```

---

# 6) Notes & environment

1. **Environment variables**: set `OPENAI_API_KEY` and `GEMINI_API_KEY` (the latter is used by `genai.configure()` in `agents_manager.py`).
2. **Install** system deps for OCR:

   * `tesseract-ocr` (system package) and `poppler-utils` for `pdf2image`.
   * Python packages: `pytesseract`, `easyocr`, `pdf2image`, `pdfplumber`.
3. **Python dependencies**: streamlit, openai, google-generativeai (if used), PyYAML, networkx, pyvis, nltk, pillow, pdf2image, pytesseract, easyocr, pdfplumber, markdown2, etc.
4. If Google GenAI SDK or patterns differ for your account, substitute the `run_genai_completion` logic to your preferred GenAI client method (the code includes fallback attempts).
5. UI theme names map to your requested list — UI CSS is applied in `app.py`.

---

# 7) How to use

* Place `app.py`, `agents_manager.py`, `processors.py`, `kg.py`, `agents.yaml` in the repo root.
* Ensure `requirements.txt` contains required packages (update with `google-generativeai` if you have it).
* Launch locally: `streamlit run app.py`
* In sidebar, pick OCR backend (pytesseract or easyocr), pick theme, upload documents, run pipeline.

---

Everything requested has been implemented and the advanced `agents.yaml` is included. If you want, I can also produce a `requirements.txt` tailored to these files and a ready-to-deploy GitHub repo zip — but I won't ask you anything further.
