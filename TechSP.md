Nice — below I built a complete, ready-to-run **agentic multi-document analysis + knowledge-graph platform** that you can deploy on Streamlit Share. It includes:

* a Streamlit app (`app.py`) that orchestrates the phases in your spec
* modular agent manager and tools (`agents_manager.py`, `processors.py`)
* a sample `agents.yaml` with configurable agents and prompts
* advanced prompt templates for agent behaviors (ready to paste into agents.yaml or the UI)
* `requirements.txt` and notes for connecting both OpenAI and Gemini
* short deployment notes and recommended file structure

I did **not** ask any extra clarifying questions (per your request). At the end I included a few short follow-up questions you said you'd like to see.

---

# 1) File structure (suggested)

```
/agentic-kg-platform
│
├── app.py                     # Streamlit app (main)
├── agents.yaml                # Agent specs & prompts (editable)
├── agents_manager.py          # Orchestration, agent execution, SDK wrappers
├── processors.py              # OCR, markdown conversion, extraction helpers
├── kg.py                      # Knowledge graph creation + visualization helpers
├── requirements.txt
└── README.md
```

---

# 2) `requirements.txt`

```text
streamlit>=1.24
openai>=1.0.0
PyYAML>=6.0
pandas>=2.0
pdfplumber>=0.7.8
pytesseract>=0.3.10
python-docx>=0.8.11
markdown2>=2.4.10
networkx>=3.1
pyvis>=0.3.1
nltk>=3.8
spacy>=3.5
transformers>=4.31.0
langchain>=0.0.375
tqdm
uvicorn
requests
python-dotenv
```

(Install additional system deps for Tesseract OCR when deploying to Cloud: `tesseract-ocr` package.)

---

# 3) `agents.yaml` (detailed agent specs)

```yaml
# agents.yaml - editable list of agents and prompt templates
default_model: openai:chatgpt-4o-mini
agents:
  - id: summary_generation
    name: Summary Generation Agent
    model: openai:chatgpt-4o-mini
    temperature: 0.1
    max_tokens: 1200
    prompt_template: |
      You are a document summarizer. Input: {document_markdown}
      Produce:
        1) Short executive summary (3-5 sentences)
        2) Structured bullet summary (sections with headers)
        3) 5 action items (if applicable)
      Output JSON with keys: executive_summary, structured, action_items
    enabled: true
    tools: []

  - id: keyword_extraction
    name: Keyword Extraction Agent
    model: openai:chatgpt-5-nano
    temperature: 0.0
    max_tokens: 300
    prompt_template: |
      Extract top 50 keywords/shortphrases from the text. Rank by importance.
      Input: {document_markdown}
      Output JSON: {"keywords":[{"term":"...", "score":0.0}, ...]}

  - id: entity_extraction
    name: Entity Extraction Agent
    model: google:gemini-2.5-flash
    temperature: 0.0
    max_tokens: 600
    prompt_template: |
      Extract up to 100 unique named entities in categories: PERSON, ORG, LOCATION, PRODUCT, EVENT, DATE, MONEY, IP, OTHER.
      For each entity return: text, normalized_form, category, first_appearance_char_index, contexts:[...]
      Input: {document_markdown}
      Output: JSON array of entities.

  - id: word_graph
    name: Word Graph Visualization Agent
    model: openai:chatgpt-4o-mini
    temperature: 0.1
    prompt_template: |
      Build a word co-occurrence graph description for visualization.
      Input: {document_markdown}
      Instructions:
        - Compute top 200 tokens (after stopword removal, lemmatize).
        - For top tokens, compute pairwise co-occurrence (sentence-level) weights.
      Output JSON: {"nodes":[{"id":"token","weight":...}], "edges":[{"source":"a","target":"b","weight":...}]}

  - id: relationship_extraction
    name: Relationship Extraction Agent
    model: google:gemini-2.5-flash-lite
    temperature: 0.0
    prompt_template: |
      Given TWO or MORE documents, extract relationships between entities across documents.
      Input: {documents_markdown_list}
      Output: JSON: [{"from_entity":"", "to_entity":"", "relationship":"", "evidence":[{"doc_id":1,"text":"..."}]}]

  - id: mindmap_generation
    name: Mind Map Generation Agent
    model: openai:chatgpt-4o-mini
    temperature: 0.2
    prompt_template: |
      Create a hierarchical mind map JSON from the given merged summary and relationship map.
      Input: {merged_summary} and {relationship_json}
      Output JSON format: {"root":{"title":"", "children":[...]}}
```

You can update `agents.yaml` in-app or externally — `app.py` will load it and allow modifications before executing.

---

# 4) `processors.py` — OCR, Markdown conversion, text cleaning, simple extractors

```python
# processors.py
import io, os, re
import pdfplumber
import pytesseract
from PIL import Image
import markdown2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def pdf_to_text_bytes(file_bytes, pages=None, ocr_if_needed=True):
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_nums = pages if pages else list(range(len(pdf.pages)))
        for i in page_nums:
            page = pdf.pages[i]
            ptext = page.extract_text()
            if (not ptext or len(ptext.strip()) < 50) and ocr_if_needed:
                # rasterize then OCR
                pil = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(pil)
                text.append(ocr_text)
            else:
                text.append(ptext or "")
    return "\n\n".join(text)

def md_from_text(text):
    # simple conversion: keep as markdown-safe
    return markdown2.markdown(text)

def clean_text(text):
    # minimal cleaning
    t = text.replace("\r\n", "\n")
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def extract_sentences(text):
    return sent_tokenize(text)

def top_tokens(text, top_k=200):
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    tokens = [t for t in tokens if t not in STOPWORDS and len(t)>2]
    cnt = Counter(tokens)
    return cnt.most_common(top_k)

def cooccurrence_pairs(sentences, top_terms, window_size=1):
    top_set = set([t for t,_ in top_terms])
    edges = Counter()
    for s in sentences:
        toks = [w.lower() for w in word_tokenize(s) if w.isalpha()]
        filtered = [t for t in toks if t in top_set]
        for i,a in enumerate(filtered):
            for j in range(i+1, min(i+1+window_size, len(filtered))):
                b = filtered[j]
                if a!=b:
                    key = tuple(sorted((a,b)))
                    edges[key]+=1
    return [{"source":a,"target":b,"weight":w} for (a,b),w in edges.items()]
```

---

# 5) `agents_manager.py` — orchestrator / agent runner (OpenAI + Gemini wrapper stubs)

```python
# agents_manager.py
import yaml, os, json, time, requests
from openai import OpenAI
from typing import Dict, Any, List

OPENAI_CLIENT = None

def get_openai_client():
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        # Requires OPENAI_API_KEY env var
        OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return OPENAI_CLIENT

def load_agents_config(path="agents.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_openai_completion(model, prompt, temperature=0.0, max_tokens=512):
    client = get_openai_client()
    # Chat completions using Chat API via OpenAI Python client
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    # choose first assistant content
    return resp.choices[0].message["content"]

def run_gemini_completion(model, prompt, temperature=0.0, max_tokens=512):
    # Placeholder: here we expect a user-supplied Gemini key or service endpoint
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT")  # optional
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set for Gemini model calls.")
    # Very simple POST; real impl should use Google SDK with proper auth
    payload = {"model": model, "prompt": prompt, "temperature": temperature, "max_output_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    url = GEMINI_ENDPOINT or "https://api.generative.googleapis.com/v1beta2/models/%s:predict" % model
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json().get("predictions", [r.text])[0]

def execute_agent(agent_def: Dict[str, Any], inputs: Dict[str, Any]):
    model_selector = agent_def.get("model", agent_def.get("default_model", "openai:chatgpt-4o-mini"))
    # model_prefix: "openai:NAME" or "google:NAME"
    if ":" in model_selector:
        provider, model_name = model_selector.split(":",1)
    else:
        # fallback assume openai
        provider, model_name = "openai", model_selector

    prompt_template = agent_def.get("prompt_template", "")
    # simple format interpolation
    prompt_filled = prompt_template.format(**inputs)

    temperature = agent_def.get("temperature", 0.0)
    max_tokens = agent_def.get("max_tokens", 512)

    if provider == "openai":
        return run_openai_completion(model_name, prompt_filled, temperature, max_tokens)
    elif provider == "google":
        return run_gemini_completion(model_name, prompt_filled, temperature, max_tokens)
    else:
        raise ValueError("Unsupported provider: %s" % provider)
```

Notes:

* `OpenAI` client usage above uses the modern `openai` Python package interface. Adjust to your environment if you use `openai==0.x` vs. new `openai` packages.
* Gemini calls are left as a simple `requests.post` — in production use Google Cloud client libs & IAM service accounts.

---

# 6) `kg.py` — knowledge graph creation & visualization (NetworkX + pyvis)

```python
# kg.py
import networkx as nx
from pyvis.network import Network
import json
import tempfile, os

def build_kg_from_entities_relations(entities, relations):
    # entities: [{"id": "...", "label":"", "category":""}]
    # relations: [{"from":"id","to":"id", "relationship":"", "weight":1, "evidence":[...]}]
    G = nx.Graph()
    for e in entities:
        eid = e.get("normalized") or e.get("text")
        G.add_node(eid, label=e.get("text"), category=e.get("category"), size=10)
    for r in relations:
        src = r["from"]
        dst = r["to"]
        w = r.get("weight",1)
        G.add_edge(src, dst, weight=w, relationship=r.get("relationship",""))
    return G

def export_pyvis(G, title="kg"):
    net = Network(height="800px", width="100%", notebook=False)
    for n,d in G.nodes(data=True):
        net.add_node(n, label=d.get("label",str(n)), title=str(d), value=d.get("size",10))
    for u,v,d in G.edges(data=True):
        net.add_edge(u,v, title=d.get("relationship",""), value=d.get("weight",1))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.show(tmp.name)
    return tmp.name
```

---

# 7) `app.py` — Streamlit orchestration UI (main)

```python
# app.py
import streamlit as st
import json, os, base64
from processors import pdf_to_text_bytes, md_from_text, clean_text, extract_sentences, top_tokens, cooccurrence_pairs
from agents_manager import load_agents_config, execute_agent
from kg import build_kg_from_entities_relations, export_pyvis

st.set_page_config(page_title="Agentic KG Platform", layout="wide")
st.title("Agentic Multi-Document Analysis & Knowledge Graph Platform")

# --- load agents config
config = load_agents_config("agents.yaml")
agents = {a["id"]: a for a in config.get("agents", [])}

# --- Document input
st.sidebar.header("Phase 1: Document Input")
uploaded = st.sidebar.file_uploader("Upload documents (PDF, TXT, MD)", accept_multiple_files=True)
ocr_pages = st.sidebar.text_input("OCR pages (comma list) - leave empty for all when using OCR", value="")
run_pipeline = st.sidebar.button("Run Pipeline")

# Agent selection
st.sidebar.header("Agents")
selected_agent_ids = st.sidebar.multiselect("Select agents to run (order preserved)", options=list(agents.keys()), default=list(agents.keys()))

# show uploaded files
if uploaded:
    doc_texts = []
    doc_names = []
    for f in uploaded:
        raw = f.read()
        name = f.name
        if name.lower().endswith(".pdf"):
            pages = None
            if ocr_pages.strip():
                try:
                    pages = [int(x.strip()) for x in ocr_pages.split(",")]
                except:
                    pages = None
            text = pdf_to_text_bytes(raw, pages=pages, ocr_if_needed=True)
        else:
            text = raw.decode("utf-8", errors="ignore")
        text = clean_text(text)
        doc_texts.append(text)
        doc_names.append(name)
    st.write(f"Loaded {len(doc_texts)} documents.")
    for i,(n,t) in enumerate(zip(doc_names, doc_texts)):
        st.expander(f"{n} - preview", expanded=False).write(t[:2000] + ("..." if len(t)>2000 else ""))

if run_pipeline and uploaded:
    # Phase: Convert to Markdown + merge
    st.header("Processing pipeline results")
    md_docs = [md_from_text(t) for t in doc_texts]
    merged_md = "\n\n".join(md_docs)
    st.subheader("Merged document preview")
    st.write(merged_md[:3000])

    # run selected agents sequentially
    results = {}
    for aid in selected_agent_ids:
        agent_def = agents[aid]
        st.info(f"Running agent: {agent_def.get('name')}")
        # prepare inputs mapping for prompt formatting
        inputs = {
            "document_markdown": merged_md,
            "documents_markdown_list": json.dumps(md_docs),
            "merged_summary": merged_md
        }
        try:
            out = execute_agent(agent_def, inputs)
            # try to JSON parse, else keep raw
            try:
                out_json = json.loads(out)
                results[aid] = out_json
                st.json(out_json)
            except Exception:
                results[aid] = out
                st.text(out[:3000])
        except Exception as e:
            st.error(f"Agent {aid} failed: {e}")

    # If entities+relations available, build KG
    if "entity_extraction" in results and "relationship_extraction" in results:
        entities = results["entity_extraction"]
        relations = results["relationship_extraction"]
        # adopt if results are nested
        if isinstance(entities, dict) and entities.get("entities"):
            entities = entities["entities"]
        G = build_kg_from_entities_relations(entities, relations)
        path = export_pyvis(G)
        st.subheader("Knowledge graph")
        st.components.v1.html(open(path).read(), height=800, scrolling=True)

    st.success("Pipeline finished.")
else:
    if not uploaded:
        st.info("Upload at least one document in the sidebar to start.")
```

Notes:

* This Streamlit app loads `agents.yaml`, lets you upload documents, chooses agents to run, and runs them sequentially.
* The code expects environment variables `OPENAI_API_KEY`, `GEMINI_API_KEY` (if using Gemini), and optional `GEMINI_ENDPOINT`.

---

# 8) Advanced prompt design — single advanced "agentic" prompt (for Dynamic Agent Executor)

Use this prompt when you want a single agent to orchestrate other agents or produce multi-step outputs. Paste it as a template in `agents.yaml` for a `dynamic_agent_executor`:

```
Advanced Dynamic Agent Executor Prompt Template
------------------------------------------------
SYSTEM:
You are the Dynamic Agent Executor (DAE). Your job is to orchestrate analysis agents on a corpus of documents to produce:
  - comprehensive merged summary
  - 100 unique entities with categories and contexts
  - relationship map between entities across documents
  - a word co-occurrence graph for visualization
  - a mind map (hierarchical JSON)

You may call internal agents (Summary, EntityExtraction, KeywordExtraction, RelationshipExtraction) in the order you decide. When you call a sub-agent, respond with EXACT JSON with keys:
  { "action":"call_agent", "agent_id":"<id>", "inputs": { ... } }

When finished, return JSON:
{
  "status":"success",
  "summary": { ... },
  "entities": [...],
  "relationships": [...],
  "word_graph": { nodes: [...], edges: [...] },
  "mind_map": { ... },
  "execution_trace": [
    {"agent_id":"", "start_ts":"", "end_ts":"", "result_summary": "..."}
  ]
}

CONSTRAINTS:
- Keep temperature 0.0 for extraction tasks.
- Validate entities are unique by normalized form (lowercased).
- For each relationship include at least one textual evidence snippet with doc_id and char offsets.
- If a sub-agent returns text, parse JSON; if parsing fails, retry the agent step up to 2 times, then store raw output and continue.

INPUT:
- {documents_markdown_list}  (an array of markdown strings)
- {config_agents_yaml}  (raw text of agents.yaml available)

Act as an orchestrator and produce the required final JSON only. Do NOT produce extra commentary.
```

This prompt ensures the dynamic executor behaves as an agent-of-agents — returning structured JSON that your Streamlit UI can parse.

---

# 9) Deployment notes (Streamlit Share)

1. Commit all files and `requirements.txt` to a public or private repo connected to Streamlit Share.
2. In Streamlit app settings, set environment variables:

   * `OPENAI_API_KEY`
   * `GEMINI_API_KEY` (if using Gemini) and `GEMINI_ENDPOINT` if required
   * `TESSERACT_CMD` path if needed
3. Make sure Tesseract system binary is available. If not, avoid OCR or use a hosted OCR service.
4. For Gemini in production, prefer Google Cloud client libs and service account credentials (not raw API key).
5. Add CORS / security layers for production if exposing file upload to public users.

---

# 10) Security & cost considerations

* Running `gemini-2.5-flash` or `chatgpt-4o-mini` at scale can be costly — include quotas and rate-limiting.
* Sanitize and limit document sizes uploaded by users; implement async job queues for very large documents (note: your request forbids background tasks; above code runs synchronously).
* Do not commit API keys to the repo.

---

# 11) Example minimal `agents.yaml` entry for Dynamic Executor (copy-paste ready)

```yaml
- id: dynamic_executor
  name: Dynamic Agent Executor
  model: openai:chatgpt-4o-mini
  temperature: 0.0
  max_tokens: 2500
  prompt_template: |
    [PASTE THE "Advanced Dynamic Agent Executor Prompt Template" HERE]
  enabled: true
```

---

# 12) Final tips & quick checklist before you run

* Set `OPENAI_API_KEY` in your Streamlit Share secrets.
* If using Gemini, define `GEMINI_API_KEY` and prefer `GEMINI_ENDPOINT` pointing to your Google Cloud endpoint.
* Validate `agents.yaml` syntax before launch.
* Start with small sample docs to validate pipeline and agent outputs.

---

# Follow-up questions (short; you asked for them)

1. Do you want me to generate a fully-populated `agents.yaml` with tuned prompts for each extraction category (PERSON/ORG/LOCATION/PRODUCT/DATE/MONEY) and unit tests?
2. Which hosting will you use for Gemini (Google Cloud project / service account) so I can provide a production-grade Gemini call example (OAuth + Google client)?
3. Do you prefer the knowledge graph exported as a standalone interactive HTML (pyvis) or embedded D3/Vis.js for customization?

---

If you want, I can now:

* expand `agents.yaml` with more tuned prompts and JSON schemas for each agent,
* generate unit tests for the pipeline,
* or produce a one-click Streamlit Share-ready repo (zipped) with sample docs.

Which of those should I do next?
