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
