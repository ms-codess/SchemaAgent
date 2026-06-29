"""
SchemaAgent — Streamlit chat UI.

Wires HybridFusion (router -> schema RAG + doc RAG -> SQL agent) into a
polished chat interface. Connect to a database, optionally add documents,
then ask questions in plain English.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="SchemaAgent",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS -----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Backgrounds */
.stApp { background: #07070F !important; }
section[data-testid="stSidebar"] {
    background: #0C0C18 !important;
    border-right: 1px solid rgba(124,58,237,.18) !important;
}
.main .block-container {
    max-width: 840px;
    padding: 0.5rem 2rem 7rem;
    margin: 0 auto;
}

/* Brand */
.brand { padding: 4px 0 20px; display: flex; align-items: center; gap: 12px; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 11px; flex-shrink: 0;
    background: linear-gradient(135deg, #7C3AED, #3B82F6);
    display: flex; align-items: center; justify-content: center;
    font-size: 19px; font-weight: 900; color: #fff;
    box-shadow: 0 4px 16px rgba(124,58,237,.4);
}
.brand-name { font-size: 17px; font-weight: 700; color: #EEEEF5; letter-spacing: -.4px; }
.brand-tag  { font-size: 10px; color: #44446A; letter-spacing: 1.3px; text-transform: uppercase; }

/* Sidebar section label */
.sl {
    font-size: 10px; font-weight: 700; letter-spacing: 1.6px;
    text-transform: uppercase; color: #7C3AED; margin-bottom: 6px;
}

/* Connection tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(124,58,237,.06) !important;
    border-radius: 10px !important; gap: 2px !important;
    padding: 3px !important; border: 1px solid rgba(124,58,237,.15) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #6B6B9A !important;
    border-radius: 8px !important; font-size: 12px !important;
    font-weight: 600 !important; padding: 5px 12px !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(124,58,237,.3) !important; color: #C4B5FD !important;
}

/* Text inputs */
.stTextInput > div > div > input {
    background: rgba(124,58,237,.06) !important;
    border: 1px solid rgba(124,58,237,.25) !important;
    border-radius: 9px !important; color: #EEEEF5 !important; font-size: 13px !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(124,58,237,.6) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #3E3E60 !important; }
.stTextInput label { color: #8B8BA7 !important; font-size: 12px !important; }

/* File uploader */
[data-testid="stFileUploader"] > div {
    background: rgba(124,58,237,.04) !important;
    border: 1px dashed rgba(124,58,237,.28) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] span { color: #6B6B9A !important; }
[data-testid="stFileUploader"] small { color: #44446A !important; }

/* Default buttons (sidebar) */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(124,58,237,.3) !important;
    color: #A78BFA !important; border-radius: 9px !important;
    font-size: 13px !important; font-weight: 500 !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    background: rgba(124,58,237,.15) !important;
    border-color: rgba(124,58,237,.55) !important; color: #C4B5FD !important;
}

/* Connect button */
.btn-connect > button {
    background: linear-gradient(135deg,#7C3AED,#6D28D9) !important;
    border: none !important; color: #fff !important;
    font-weight: 600 !important; font-size: 13px !important;
    border-radius: 9px !important;
    box-shadow: 0 3px 14px rgba(124,58,237,.35) !important;
}
.btn-connect > button:hover {
    box-shadow: 0 5px 20px rgba(124,58,237,.5) !important;
    transform: translateY(-1px) !important;
    background: linear-gradient(135deg,#8B5CF6,#7C3AED) !important;
    color: #fff !important;
}

/* Example chip buttons — only main area, not sidebar */
.main .stButton > button {
    background: rgba(124,58,237,.07) !important;
    border: 1px solid rgba(124,58,237,.22) !important;
    color: #A78BFA !important; border-radius: 22px !important;
    font-size: 13px !important; font-weight: 400 !important;
    padding: 7px 16px !important; width: 100% !important;
    text-align: left !important; transition: all .15s !important;
}
.main .stButton > button:hover {
    background: rgba(124,58,237,.18) !important;
    border-color: rgba(124,58,237,.5) !important;
    color: #C4B5FD !important; transform: translateY(-1px) !important;
}

/* Status pills */
.pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 11px 5px 9px; border-radius: 20px;
    font-size: 12px; font-weight: 500; margin-top: 8px;
}
.pill-ok  { background: rgba(16,185,129,.1);  color: #34D399; border: 1px solid rgba(16,185,129,.22); }
.pill-off { background: rgba(99,102,241,.07); color: #818CF8; border: 1px solid rgba(99,102,241,.18); }
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.dot-pulse { animation: dp 2s ease-in-out infinite; }
@keyframes dp { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Sliders + checkboxes */
.stSlider label { color: #8B8BA7 !important; font-size: 12px !important; }
.stCheckbox label { color: #8B8BA7 !important; font-size: 13px !important; }

/* Sidebar expanders */
section[data-testid="stSidebar"] [data-testid="stExpander"] > div:first-child {
    background: rgba(124,58,237,.05) !important;
    border: 1px solid rgba(124,58,237,.16) !important; border-radius: 9px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #8B8BA7 !important; font-size: 13px !important; font-weight: 600 !important;
}

/* Welcome screen */
.welcome {
    display: flex; flex-direction: column; align-items: center;
    text-align: center; padding: 60px 20px 28px;
}
.wmark {
    width: 68px; height: 68px; border-radius: 22px;
    background: linear-gradient(135deg, #7C3AED, #3B82F6);
    display: flex; align-items: center; justify-content: center;
    font-size: 34px; font-weight: 900; color: white; margin-bottom: 22px;
    box-shadow: 0 14px 44px rgba(124,58,237,.38);
}
.wtitle { font-size: 32px; font-weight: 700; color: #EEEEF5; letter-spacing: -1.2px; margin-bottom: 12px; }
.wsub   { font-size: 15px; color: #EEEEF5; line-height: 1.7; max-width: 460px; margin-bottom: 10px; }
.chips-label {
    font-size: 11px; color: #44446A; letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 14px; margin-top: 24px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important; padding: 3px 0 !important;
}
[data-testid="stChatMessage"] > div { background: transparent !important; }

/* User bubble */
.ub {
    background: linear-gradient(135deg, #6D28D9, #7C3AED);
    border-radius: 18px 18px 4px 18px;
    padding: 13px 18px; max-width: 76%; margin-left: auto;
    font-size: 15px; line-height: 1.55; color: #F0EEFF;
    box-shadow: 0 4px 22px rgba(109,40,217,.30); word-break: break-word;
}

/* Assistant bubble */
.ab {
    background: #111120; border: 1px solid rgba(255,255,255,.08);
    border-radius: 4px 18px 18px 18px;
    padding: 16px 20px; max-width: 88%;
    font-size: 15px; line-height: 1.68; color: #DCDCF0;
    box-shadow: 0 4px 28px rgba(0,0,0,.38); word-break: break-word;
}
.ab strong { color: #EEEEF5; }
.ab code {
    background: rgba(124,58,237,.15); color: #C4B5FD;
    padding: 1px 5px; border-radius: 4px; font-size: 13px;
}

/* Route / meta badges */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 700; letter-spacing: .8px;
    text-transform: uppercase; margin-bottom: 11px;
}
.badge-db  { background:rgba(59,130,246,.1);  color:#60A5FA; border:1px solid rgba(59,130,246,.22); }
.badge-doc { background:rgba(16,185,129,.1);  color:#34D399; border:1px solid rgba(16,185,129,.22); }
.badge-hyb { background:rgba(139,92,246,.1);  color:#A78BFA; border:1px solid rgba(139,92,246,.22); }
.badge-att { background:rgba(245,158,11,.08); color:#FBBF24; border:1px solid rgba(245,158,11,.18); margin-left:5px; }

/* Main expanders */
.main [data-testid="stExpander"] {
    background: rgba(255,255,255,.025) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important; margin-top: 10px !important;
}
.main [data-testid="stExpander"] summary {
    color: #6B6B9A !important; font-size: 12px !important; font-weight: 600 !important;
}
.main [data-testid="stExpander"] summary:hover { color: #EEEEF5 !important; }

/* Context used items */
.ctx-row {
    display: flex; align-items: baseline; gap: 10px;
    font-size: 13px; color: #8B8BA7; padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,.04);
}
.ctx-row:last-child { border-bottom: none; }
.ctx-key { font-size: 10px; font-weight: 700; letter-spacing: .8px;
           text-transform: uppercase; color: #44446A; min-width: 80px; }
.ctx-val { color: #C4B5FD; }

/* Code blocks */
.stCode > div, pre {
    background: #0A0A18 !important;
    border: 1px solid rgba(124,58,237,.22) !important; border-radius: 10px !important;
}

/* Chat input */
[data-testid="stChatInput"] > div {
    background: #111120 !important;
    border: 1px solid rgba(124,58,237,.28) !important;
    border-radius: 16px !important; transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(124,58,237,.65) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,.12) !important;
}
[data-testid="stChatInput"] textarea {
    color: #EEEEF5 !important; font-size: 15px !important;
    caret-color: #7C3AED !important; background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3A3A58 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg,#7C3AED,#6D28D9) !important;
    border-radius: 10px !important;
}

/* Error bubble */
.err {
    background: rgba(239,68,68,.08); border: 1px solid rgba(239,68,68,.22);
    border-radius: 10px; padding: 12px 16px; color: #FCA5A5;
    font-size: 14px; line-height: 1.55;
}

/* Divider */
hr { border-color: rgba(255,255,255,.06) !important; margin: 12px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,.32); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,.55); }
</style>
""", unsafe_allow_html=True)

# -- Session state -------------------------------------------------------------
_DEFAULTS: dict = {
    "messages":          [],
    "db_path":           None,
    "db_id":             None,
    "db_label":          None,
    "indexed_dbs":       set(),
    "indexed_docs":      [],
    "pending_question":  None,   # set by chip buttons
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# -- Cached AI components ------------------------------------------------------
@st.cache_resource(show_spinner="Initialising AI components…")
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
        client=client, schema_rag=schema_rag, doc_rag=doc_rag, router=router,
    )
    return client, schema_rag, doc_rag, fusion


# -- Helpers -------------------------------------------------------------------
def _try_connect(db_path: str) -> str | None:
    """Validate a SQLite path. Returns error string or None."""
    p = Path(db_path)
    if not p.exists():
        return f"File not found: {db_path}"
    try:
        con = sqlite3.connect(db_path)
        con.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        con.close()
    except Exception as exc:
        return str(exc)
    return None


def _index_db(db_path: str, db_id: str) -> str | None:
    """Index schema into ChromaDB. Returns error string or None."""
    try:
        _, schema_rag, _, _ = _load_components()
        schema_rag.index(db_path=db_path, db_id=db_id)
        st.session_state.indexed_dbs.add(db_id)
    except Exception as exc:
        return str(exc)
    return None


def _extract_tables(sql: str) -> list[str]:
    """Best-effort table extraction from SQL using regex."""
    raw = re.findall(
        r'\b(?:FROM|JOIN)\s+[`"]?(\w+)[`"]?',
        sql, re.IGNORECASE,
    )
    seen, out = set(), []
    for t in raw:
        if t.upper() not in ("SELECT", "WHERE", "ON") and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _route_badge(intent: str, attempts: int) -> str:
    cfg = {
        "database": ("badge-db",  "⬡", "database"),
        "document": ("badge-doc", "◻", "document"),
        "hybrid":   ("badge-hyb", "⬢", "hybrid"),
    }
    cls, icon, label = cfg.get(intent, ("badge-hyb", "◇", intent))
    att = (f'<span class="badge badge-att">↻ {attempts} attempts</span>'
           if attempts > 1 else "")
    return f'<span class="badge {cls}">{icon} {label}</span>{att}'


def _render_result(result) -> None:
    """Render badge + answer + SQL + results + sources + context expanders."""
    st.markdown(_route_badge(result.intent, result.sql_attempts),
                unsafe_allow_html=True)
    st.markdown(f'<div class="ab">{result.answer}</div>', unsafe_allow_html=True)

    if result.sql:
        with st.expander("⬡  SQL  ·  view query"):
            st.code(result.sql, language="sql")

    if result.sql_rows:
        n = len(result.sql_rows)
        with st.expander(f"◻  Results  ·  {n} row{'s' if n != 1 else ''}",
                         expanded=True):
            try:
                st.dataframe(pd.DataFrame(result.sql_rows),
                             use_container_width=True, hide_index=True)
            except Exception:
                st.text(str(result.sql_rows[:30]))

    if result.doc_passages:
        src = ", ".join(result.sources) if result.sources else "documents"
        with st.expander(f"◻  Sources  ·  {src}"):
            st.markdown(
                f'<div style="font-size:13px;color:#8B8BA7;line-height:1.75">'
                f'{result.doc_passages[:1000]}</div>',
                unsafe_allow_html=True,
            )

    # Context used
    rows = []
    rows.append(("route", result.intent))
    rows.append(("confidence", result.route_confidence))
    if result.route_reasoning:
        rows.append(("reasoning", result.route_reasoning))
    if result.sql:
        tables = _extract_tables(result.sql)
        if tables:
            rows.append(("tables used", ", ".join(tables)))
    if result.sql_attempts > 1:
        rows.append(("sql attempts", str(result.sql_attempts)))
    if result.sources:
        rows.append(("doc sources", ", ".join(result.sources)))

    ctx_html = "".join(
        f'<div class="ctx-row"><span class="ctx-key">{k}</span>'
        f'<span class="ctx-val">{v}</span></div>'
        for k, v in rows
    )
    with st.expander("◎  Context used"):
        st.markdown(ctx_html, unsafe_allow_html=True)


def _contextual_question(prompt: str, messages: list) -> str:
    """
    Prepend the last 3 Q&A exchanges to the question so the agent can
    resolve follow-up references like 'which of them', 'how about X'.
    """
    pairs = []
    history = [m for m in messages if m["role"] in ("user", "assistant")]
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            q = history[i]["content"]
            a = history[i+1]["content"]
            a_short = a[:300] + ("…" if len(a) > 300 else "")
            pairs.append(f"User: {q}\nAssistant: {a_short}")
            i += 2
        else:
            i += 1

    if not pairs:
        return prompt

    recent = pairs[-3:]  # last 3 exchanges
    context = "\n\n".join(recent)
    return (
        f"Conversation history (for context only — answer the current question):\n"
        f"{context}\n\n"
        f"Current question: {prompt}"
    )


def _process_question(prompt: str) -> None:
    """Append user message, call fusion, append assistant result."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="user"):
        st.markdown(f'<div class="ub">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="assistant"):
        result = None
        err_msg = None

        with st.spinner("Thinking…"):
            try:
                import traceback
                _, schema_rag, _, fusion = _load_components()
                schema_rag.top_k = st.session_state.get("_top_k_schema", 5)
                full_q = _contextual_question(
                    prompt, st.session_state.messages[:-1]
                )
                result = fusion.answer(
                    question=full_q,
                    db_id=st.session_state.db_id or "",
                    db_path=st.session_state.db_path or "",
                    use_hyde=st.session_state.get("_use_hyde", False),
                    top_k_docs=st.session_state.get("_top_k_docs", 3),
                )
            except Exception:
                err_msg = traceback.format_exc()
                result = None

        if result is not None:
            _render_result(result)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "result": result,
            })
        else:
            st.code(err_msg, language="text")
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg.strip().splitlines()[-1],
                "result": None,
            })


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

    # Database connection
    st.markdown('<div class="sl">Connect database</div>', unsafe_allow_html=True)
    tab_upload, tab_path = st.tabs(["Upload file", "Local path"])

    with tab_upload:
        db_file = st.file_uploader(
            "SQLite file", type=["sqlite", "db", "sqlite3"],
            label_visibility="collapsed", key="db_upload",
        )
        if db_file:
            upload_dir = Path("data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved = str(upload_dir / db_file.name)
            with open(saved, "wb") as f:
                f.write(db_file.getbuffer())
            db_id = Path(db_file.name).stem
            if db_id not in st.session_state.indexed_dbs:
                with st.spinner("Indexing schema…"):
                    err = _index_db(saved, db_id)
                if err:
                    st.error(err)
                else:
                    st.session_state.db_path  = saved
                    st.session_state.db_id    = db_id
                    st.session_state.db_label = db_file.name
            else:
                st.session_state.db_path  = saved
                st.session_state.db_id    = db_id
                st.session_state.db_label = db_file.name

    with tab_path:
        path_input = st.text_input(
            "Path to .sqlite file",
            placeholder="data/bird/dev_databases/formula_1/formula_1.sqlite",
            key="path_input",
        )
        st.markdown('<div class="btn-connect">', unsafe_allow_html=True)
        connect_clicked = st.button("Connect", key="btn_path", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if connect_clicked and path_input.strip():
            p = path_input.strip()
            err = _try_connect(p)
            if err:
                st.error(err)
            else:
                db_id = Path(p).stem
                with st.spinner("Indexing schema…"):
                    idx_err = _index_db(p, db_id)
                if idx_err:
                    st.error(idx_err)
                else:
                    st.session_state.db_path  = p
                    st.session_state.db_id    = db_id
                    st.session_state.db_label = Path(p).name
                    st.success("Connected!")

    # Status pill
    if st.session_state.db_path:
        label = st.session_state.db_label or st.session_state.db_id or "database"
        st.markdown(
            f'<div class="pill pill-ok"><div class="dot dot-pulse"></div>{label}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pill pill-off"><div class="dot"></div>No database connected</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Documents
    st.markdown('<div class="sl">Documents</div>', unsafe_allow_html=True)
    doc_files = st.file_uploader(
        "PDF / DOCX / TXT", type=["pdf", "docx", "txt"],
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

    # Settings — store in session_state so _process_question can read them
    with st.expander("Settings"):
        st.session_state["_top_k_schema"] = st.slider("Schema tables (top-k)", 1, 10, 5)
        st.session_state["_top_k_docs"]   = st.slider("Doc passages (top-k)",  1, 10, 3)
        st.session_state["_use_hyde"]     = st.checkbox(
            "HyDE retrieval", value=False,
            help="Generates a hypothetical answer to improve doc recall.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.db_path:
        if st.button("Disconnect database", use_container_width=True):
            st.session_state.db_path  = None
            st.session_state.db_id    = None
            st.session_state.db_label = None
            st.session_state.messages = []
            st.rerun()


# -- Main chat area ------------------------------------------------------------
no_db = st.session_state.db_path is None

@st.cache_data(show_spinner=False)
def _schema_questions(db_path: str, db_id: str) -> list[str]:
    """Generate chip questions from the actual table/column names in the DB."""
    try:
        con = sqlite3.connect(db_path)
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()]
        if not tables:
            return ["How many records are in each table?"]

        counts: dict = {}
        for t in tables[:8]:
            try:
                counts[t] = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except Exception:
                counts[t] = 0

        ranked = sorted(counts, key=counts.get, reverse=True)
        main   = ranked[0]
        cols   = [r[1] for r in con.execute(f'PRAGMA table_info("{main}")').fetchall()]
        con.close()

        qs = [f"How many {main} are in the database?",
              f"Show me the top 10 {main} records"]

        if len(ranked) > 1:
            second = ranked[1]
            qs.append(f"How many {second} are there?")

        num_cols  = [c for c in cols if any(k in c.lower()
                     for k in ("points","score","wins","total","count",
                                "salary","price","amount","rank","rating"))]
        date_cols = [c for c in cols if any(k in c.lower()
                     for k in ("date","year","time","season","month"))]
        name_cols = [c for c in cols if any(k in c.lower()
                     for k in ("name","title","label","forename","surname"))]

        if num_cols:
            qs.append(f"What is the highest {num_cols[0]} in {main}?")
        if date_cols:
            qs.append(f"How many {main} per {date_cols[0]}?")
        if name_cols and len(ranked) > 1:
            qs.append(f"Which {ranked[1]} has the most {main}?")
        if len(ranked) > 2:
            qs.append("How many records are in each table?")

        return qs[:6]
    except Exception:
        return [
            "How many records are in each table?",
            "Show me the first 10 rows",
            "What columns does each table have?",
        ]

# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
      <div class="wmark">◈</div>
      <div class="wtitle">Ask your database anything</div>
      <div class="wsub">
        Connect a SQLite database in the sidebar, then ask questions
        in plain English. SchemaAgent generates SQL, executes it, and
        explains the results with optional document awareness.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if no_db:
        st.markdown(
            '<p style="text-align:center;font-size:12px;color:#44446A;margin-top:4px">'
            'Connect a database in the sidebar to see suggested questions</p>',
            unsafe_allow_html=True,
        )
    else:
        # Generate questions from the actual schema
        questions = _schema_questions(
            st.session_state.db_path, st.session_state.db_id
        )
        # Add doc questions if documents are indexed
        if st.session_state.indexed_docs:
            questions += [
                "Summarise the key rules from the uploaded documents",
                "Which records match the criteria described in the documents?",
            ]

        st.markdown(
            '<p style="text-align:center;font-size:11px;color:#44446A;'
            'letter-spacing:1px;text-transform:uppercase;margin-bottom:14px">'
            'Suggested questions</p>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        for i, q in enumerate(questions):
            with (col_a if i % 2 == 0 else col_b):
                if st.button(q, key=f"chip_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="user"):
            st.markdown(f'<div class="ub">{msg["content"]}</div>',
                        unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="assistant"):
            if msg.get("result"):
                _render_result(msg["result"])
            else:
                st.markdown(f'<div class="err">⚠ {msg["content"]}</div>',
                            unsafe_allow_html=True)

# Handle pending question from chip click
if st.session_state.pending_question and not no_db:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    _process_question(q)
    st.rerun()

# Chat input
placeholder = (
    "Ask a follow-up or a new question…"
    if st.session_state.messages else
    "Ask a question about your database…"
    if not no_db else
    "Connect a database in the sidebar to get started…"
)

if prompt := st.chat_input(placeholder, disabled=no_db):
    _process_question(prompt)
    st.rerun()
