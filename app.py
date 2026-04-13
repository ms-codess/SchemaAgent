"""
SchemaAgent — Streamlit chat UI.

Wires HybridFusion (router -> schema RAG + doc RAG -> SQL agent) into a
polished chat interface. Upload a SQLite database, optionally add documents,
then ask questions in plain English.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -- Page config (must be first Streamlit call) --------------------------------
st.set_page_config(
    page_title="SchemaAgent",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* App shell */
.stApp { background: #07070F; color: #EEEEF5; }

[data-testid="stSidebar"] {
    background: #0C0C18 !important;
    border-right: 1px solid rgba(109, 40, 217, 0.18) !important;
}

.main .block-container {
    max-width: 820px;
    padding: 1rem 2rem 7rem;
    margin: 0 auto;
}

/* Brand */
.brand { padding: 6px 0 22px; display: flex; align-items: center; gap: 12px; }
.brand-mark {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 900; color: white; flex-shrink: 0;
}
.brand-name { font-size: 17px; font-weight: 700; color: #EEEEF5; letter-spacing: -0.4px; }
.brand-tag  { font-size: 10px; color: #4A4A6A; letter-spacing: 1.2px; text-transform: uppercase; }

/* Sidebar section label */
.sl {
    font-size: 10px; font-weight: 700; color: #5B21B6;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;
}

/* Status pill */
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px 4px 8px; border-radius: 20px;
    font-size: 12px; font-weight: 500; margin-top: 6px;
}
.pill-ok  { background: rgba(16,185,129,.1);  color: #34D399; border: 1px solid rgba(16,185,129,.22); }
.pill-off { background: rgba(99,102,241,.08); color: #6366F1; border: 1px solid rgba(99,102,241,.18); }
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.dot-pulse { animation: dp 2s infinite; }
@keyframes dp { 0%,100%{opacity:1} 50%{opacity:.35} }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(109,40,217,.04) !important;
    border: 1px dashed rgba(109,40,217,.28) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(109,40,217,.55) !important;
}
[data-testid="stFileUploader"] label { color: #6B6B8A !important; font-size: 13px !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(109,40,217,.3) !important;
    color: #8B5CF6 !important; border-radius: 10px !important;
    font-size: 13px !important; font-weight: 500 !important;
    transition: all .18s !important;
}
.stButton > button:hover {
    background: rgba(109,40,217,.12) !important;
    border-color: rgba(109,40,217,.55) !important;
    color: #C4B5FD !important;
}

/* Slider thumb */
[data-testid="stSlider"] [role="slider"] { background: #7C3AED !important; }

/* Welcome screen */
.welcome {
    display: flex; flex-direction: column; align-items: center;
    text-align: center; padding: 64px 20px 40px;
}
.wmark {
    width: 64px; height: 64px; border-radius: 20px;
    background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; font-weight: 900; color: white;
    margin-bottom: 20px;
    box-shadow: 0 12px 40px rgba(124,58,237,.35);
}
.wtitle { font-size: 30px; font-weight: 700; color: #EEEEF5; letter-spacing: -1px; margin-bottom: 10px; }
.wsub   { font-size: 15px; color: #555570; line-height: 1.65; max-width: 440px; margin-bottom: 36px; }
.chips  { display: flex; flex-wrap: wrap; gap: 9px; justify-content: center; max-width: 560px; }
.chip {
    background: rgba(109,40,217,.07); border: 1px solid rgba(109,40,217,.2);
    border-radius: 22px; padding: 7px 15px; font-size: 13px; color: #A78BFA;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important; padding: 2px 0 !important;
}

/* User bubble */
.ub {
    background: linear-gradient(135deg, #6D28D9, #7C3AED);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px; max-width: 78%; margin-left: auto;
    font-size: 15px; line-height: 1.55; color: #F3F0FF;
    box-shadow: 0 4px 20px rgba(109,40,217,.28);
    word-wrap: break-word;
}

/* Assistant bubble */
.ab {
    background: #111120; border: 1px solid rgba(255,255,255,.07);
    border-radius: 4px 18px 18px 18px;
    padding: 16px 20px; max-width: 88%;
    font-size: 15px; line-height: 1.65; color: #DDDDF0;
    box-shadow: 0 4px 24px rgba(0,0,0,.35);
    word-wrap: break-word;
}
.ab p { margin: 0 0 6px; }
.ab p:last-child { margin-bottom: 0; }

/* Route badges */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 9px; border-radius: 20px; font-size: 10px;
    font-weight: 700; letter-spacing: .8px; text-transform: uppercase;
    margin-bottom: 10px;
}
.badge-db  { background: rgba(59,130,246,.1);  color: #60A5FA; border: 1px solid rgba(59,130,246,.22); }
.badge-doc { background: rgba(16,185,129,.1);  color: #34D399; border: 1px solid rgba(16,185,129,.22); }
.badge-hyb { background: rgba(139,92,246,.1);  color: #A78BFA; border: 1px solid rgba(139,92,246,.22); }
.badge-att { background: rgba(245,158,11,.08); color: #FBBF24; border: 1px solid rgba(245,158,11,.18); margin-left: 5px; }

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(255,255,255,.02) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 10px !important; margin-top: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important; color: #6B6B8A !important;
    font-weight: 600 !important; letter-spacing: .4px !important;
}
[data-testid="stExpander"] summary:hover { color: #EEEEF5 !important; }
[data-testid="stExpander"] summary svg { fill: #6B6B8A !important; }

/* Code blocks */
.stCode, pre {
    background: #0A0A16 !important;
    border: 1px solid rgba(109,40,217,.2) !important;
    border-radius: 10px !important;
}

/* DataFrame */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }

/* Chat input */
[data-testid="stChatInput"] > div {
    background: #111120 !important;
    border: 1px solid rgba(109,40,217,.28) !important;
    border-radius: 16px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(109,40,217,.65) !important;
    box-shadow: 0 0 0 3px rgba(109,40,217,.12) !important;
}
[data-testid="stChatInput"] textarea {
    color: #EEEEF5 !important; font-size: 15px !important;
    caret-color: #7C3AED !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3A3A58 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    border-radius: 10px !important;
}

/* Error bubble */
.err {
    background: rgba(239,68,68,.08); border: 1px solid rgba(239,68,68,.2);
    border-radius: 10px; padding: 12px 16px; color: #FCA5A5;
    font-size: 14px; margin-top: 8px;
}

/* Spinner */
[data-testid="stSpinner"] p { color: #555570 !important; font-size: 13px !important; }

/* Divider */
hr { border-color: rgba(255,255,255,.06) !important; margin: 14px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(109,40,217,.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(109,40,217,.55); }
</style>
""", unsafe_allow_html=True)

# -- Session state -------------------------------------------------------------
_DEFAULTS = {
    "messages": [],       # list of {role, content, result}
    "db_path": None,
    "db_id": None,
    "indexed_dbs": set(),
    "indexed_docs": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# -- Cached AI components ------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI components…")
def _load_components():
    import anthropic
    from src.schema import SchemaRAG
    from src.doc_rag import DocRAG
    from src.router import IntentRouter
    from src.fusion import HybridFusion

    client     = anthropic.Anthropic()
    schema_rag = SchemaRAG()
    doc_rag    = DocRAG()
    router     = IntentRouter(client)
    fusion     = HybridFusion(
        client=client, schema_rag=schema_rag, doc_rag=doc_rag, router=router
    )
    return client, schema_rag, doc_rag, fusion


# -- Helpers -------------------------------------------------------------------
def _route_badge(intent: str, attempts: int) -> str:
    cfg = {
        "database": ("badge-db",  "⬡", "database"),
        "document": ("badge-doc", "◻", "document"),
        "hybrid":   ("badge-hyb", "⬢", "hybrid"),
    }
    cls, icon, label = cfg.get(intent, ("badge-hyb", "◇", intent))
    att = (
        f'<span class="badge badge-att">↻ {attempts} attempts</span>'
        if attempts > 1 else ""
    )
    return f'<span class="badge {cls}">{icon} {label}</span>{att}'


def _render_result(result) -> None:
    """Render route badge + answer bubble + collapsible SQL / results / sources."""
    st.markdown(_route_badge(result.intent, result.sql_attempts),
                unsafe_allow_html=True)
    st.markdown(f'<div class="ab">{result.answer}</div>', unsafe_allow_html=True)

    if result.sql:
        with st.expander("⬡  SQL  ·  view query"):
            st.code(result.sql, language="sql")

    if result.sql_rows:
        n = len(result.sql_rows)
        with st.expander(
            f"◻  Results  ·  {n} row{'s' if n != 1 else ''}",
            expanded=True,
        ):
            try:
                st.dataframe(
                    pd.DataFrame(result.sql_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception:
                st.text(str(result.sql_rows[:30]))

    if result.doc_passages:
        src_label = ", ".join(result.sources) if result.sources else "documents"
        with st.expander(f"◻  Sources  ·  {src_label}"):
            st.markdown(
                f'<div style="font-size:13px;color:#8B8BA7;line-height:1.7">'
                f'{result.doc_passages[:900]}</div>',
                unsafe_allow_html=True,
            )


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-mark">◈</div>
        <div>
            <div class="brand-name">SchemaAgent</div>
            <div class="brand-tag">Natural Language SQL</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Database
    st.markdown('<div class="sl">Database</div>', unsafe_allow_html=True)
    db_file = st.file_uploader(
        "db", type=["sqlite", "db", "sqlite3"],
        label_visibility="collapsed", key="db_upload",
    )

    if db_file:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(upload_dir / db_file.name)
        with open(db_path, "wb") as f:
            f.write(db_file.getbuffer())

        db_id = Path(db_file.name).stem
        st.session_state.db_path = db_path
        st.session_state.db_id   = db_id

        if db_id not in st.session_state.indexed_dbs:
            with st.spinner(f"Indexing schema for {db_id}…"):
                try:
                    _, schema_rag, _, _ = _load_components()
                    schema_rag.index(db_path=db_path, db_id=db_id)
                    st.session_state.indexed_dbs.add(db_id)
                except Exception as exc:
                    st.error(f"Schema indexing failed: {exc}")

        st.markdown(
            f'<div class="pill pill-ok">'
            f'<div class="dot dot-pulse"></div>{db_id}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pill pill-off"><div class="dot"></div>No database</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Documents
    st.markdown('<div class="sl">Documents</div>', unsafe_allow_html=True)
    doc_files = st.file_uploader(
        "docs", type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed", key="doc_upload",
    )

    if doc_files:
        doc_dir = Path("data/uploads/docs")
        doc_dir.mkdir(parents=True, exist_ok=True)
        _, _, doc_rag, _ = _load_components()

        for f in doc_files:
            if f.name not in st.session_state.indexed_docs:
                with open(doc_dir / f.name, "wb") as fh:
                    fh.write(f.getbuffer())
                with st.spinner(f"Indexing {f.name}…"):
                    try:
                        doc_rag.index_documents(str(doc_dir))
                        st.session_state.indexed_docs.append(f.name)
                    except Exception as exc:
                        st.error(f"{f.name}: {exc}")

        for name in st.session_state.indexed_docs:
            st.markdown(
                f'<div style="font-size:12px;color:#34D399;padding:2px 0">✓ {name}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Settings
    with st.expander("Settings"):
        top_k_schema = st.slider("Schema tables (top-k)", 1, 10, 5)
        top_k_docs   = st.slider("Doc passages (top-k)",  1, 10, 3)
        max_attempts = st.slider("SQL retries",            1,  3, 3)
        use_hyde     = st.checkbox(
            "HyDE document retrieval", value=False,
            help="One extra LLM call per query; improves doc recall",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -- Main chat area ------------------------------------------------------------

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="wmark">◈</div>
        <div class="wtitle">Ask your database anything</div>
        <div class="wsub">
            Upload a SQLite database in the sidebar, then ask questions
            in plain English. SchemaAgent generates SQL, runs it, and
            explains the results — with document awareness.
        </div>
        <div class="chips">
            <div class="chip">How many records are in each table?</div>
            <div class="chip">Show the top 10 entries by date</div>
            <div class="chip">What's the distribution by category?</div>
            <div class="chip">Find any duplicate values</div>
            <div class="chip">Summarise the key statistics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="◯"):
            st.markdown(f'<div class="ub">{msg["content"]}</div>',
                        unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="◈"):
            if msg.get("result"):
                _render_result(msg["result"])
            else:
                st.markdown(f'<div class="err">⚠ {msg["content"]}</div>',
                            unsafe_allow_html=True)

# Chat input
no_db = st.session_state.db_path is None
placeholder = (
    "Ask a question about your database…"
    if not no_db
    else "Upload a database to get started…"
)

if prompt := st.chat_input(placeholder, disabled=no_db):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="◯"):
        st.markdown(f'<div class="ub">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="◈"):
        with st.spinner("Thinking…"):
            try:
                _, schema_rag, _, fusion = _load_components()
                schema_rag.top_k = top_k_schema

                result = fusion.answer(
                    question=prompt,
                    db_id=st.session_state.db_id or "",
                    db_path=st.session_state.db_path or "",
                    use_hyde=use_hyde,
                    top_k_docs=top_k_docs,
                )

                _render_result(result)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "result": result,
                })

            except Exception as exc:
                err_msg = str(exc)
                st.markdown(f'<div class="err">⚠ {err_msg}</div>',
                            unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "result": None,
                })
