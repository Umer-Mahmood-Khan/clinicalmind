"""
app.py — Streamlit UI for ClinicalMind
----------------------------------------
A conversational interface for the multi-agent clinical reasoning system.

Layout:
  • Sidebar    — PDF upload, document list
  • Main area  — Ethical disclaimer banner, chat interface
  • Per answer — Confidence badge, citations, agent trace expander
  • Footer     — Research-only disclaimer + attribution

Run with:
    streamlit run app.py
"""

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY) before any LangChain imports
load_dotenv()

from orchestrator import run  # noqa: E402  (import after load_dotenv)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CONFIDENCE_COLOURS = {
    "HIGH": ("🟢", "#28a745", "HIGH confidence"),
    "MEDIUM": ("🟡", "#ffc107", "MEDIUM confidence"),
    "LOW": ("🔴", "#dc3545", "LOW confidence"),
}


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClinicalMind — Multi-Agent Clinical Reasoning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — clean clinical aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Ethical disclaimer banner */
    .disclaimer-banner {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    /* Confidence badge */
    .badge-high   { background:#28a745; color:#fff; padding:4px 10px; border-radius:12px; font-weight:600; }
    .badge-medium { background:#ffc107; color:#333; padding:4px 10px; border-radius:12px; font-weight:600; }
    .badge-low    { background:#dc3545; color:#fff; padding:4px 10px; border-radius:12px; font-weight:600; }
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 12px;
        border-top: 1px solid #dee2e6;
        font-size: 0.8rem;
        color: #6c757d;
        text-align: center;
    }
    /* Chat bubbles */
    .user-bubble {
        background: #e8f4fd;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 14px;
        margin: 6px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #f8f9fa;
        border-left: 3px solid #0d6efd;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role", "content", "meta"}


# ---------------------------------------------------------------------------
# Sidebar — PDF management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📂 Document Library")
    st.caption("Upload clinical PDFs (guidelines, studies, protocols)")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs are saved to data/ and ingested into the vector index.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            dest = DATA_DIR / uf.name
            if not dest.exists():
                with open(dest, "wb") as f:
                    shutil.copyfileobj(uf, f)
                st.success(f"Saved: {uf.name}")
            else:
                st.info(f"Already present: {uf.name}")

    # Show currently loaded documents
    st.markdown("---")
    st.subheader("Loaded Documents")
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if pdfs:
        for p in pdfs:
            st.markdown(f"• {p.name}")
    else:
        st.caption("No PDFs loaded yet.")

    # Button to clear the vectorstore and force re-ingestion
    st.markdown("---")
    if st.button("🔄 Re-index Documents"):
        import shutil as _shutil
        vs_path = Path("vectorstore")
        if vs_path.exists():
            _shutil.rmtree(vs_path)
            vs_path.mkdir()
        st.success("Vectorstore cleared. It will be rebuilt on the next query.")
        # Reset cached vectorstore in agents module
        import agents
        agents._VECTORSTORE = None

    st.markdown("---")
    st.caption("ClinicalMind v1.0 | NCAI Pakistan")


# ---------------------------------------------------------------------------
# Main area — Ethical disclaimer banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    ⚕️ <strong>Ethical Disclaimer:</strong>
    ClinicalMind is a <em>research and educational tool only</em>.
    It retrieves and reasons over documents you provide.
    It does <strong>NOT</strong> diagnose, prescribe, or replace clinical judgement.
    Always consult a qualified healthcare professional before making any medical decision.
</div>
""", unsafe_allow_html=True)

st.title("🧠 ClinicalMind")
st.subheader("Multi-Agent Clinical Reasoning System")


# ---------------------------------------------------------------------------
# Chat history rendering
# ---------------------------------------------------------------------------
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">👤 {entry["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        # --- Confidence badge ---
        meta = entry.get("meta", {})
        conf = meta.get("confidence", "MEDIUM").upper()
        emoji, colour, label = CONFIDENCE_COLOURS.get(conf, CONFIDENCE_COLOURS["MEDIUM"])
        badge_class = f"badge-{conf.lower()}"

        st.markdown(
            f'<div class="assistant-bubble">'
            f'<span class="{badge_class}">{emoji} {label}</span><br><br>'
            f'{entry["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # --- Agent trace expander ---
        trace = meta.get("agent_trace", [])
        if trace:
            with st.expander("🔍 View Agent Trace", expanded=False):
                for step in trace:
                    agent_name = step.get("agent", "unknown")
                    agent_output = step.get("output", "")
                    step_conf = step.get("confidence", "")

                    st.markdown(f"**`{agent_name}`**")
                    if step_conf:
                        st.markdown(f"*Confidence assigned: {step_conf}*")
                    st.text_area(
                        label="",
                        value=agent_output,
                        height=120,
                        key=f"trace_{id(step)}",
                        disabled=True,
                    )
                    st.markdown("---")


# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
with st.form("query_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_question = st.text_input(
            "Ask a clinical question",
            placeholder="e.g. What are the first-line treatments for Type 2 diabetes?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Ask 🔍", use_container_width=True)

if submitted and user_question.strip():
    # --- Guard: need PDFs before querying ---
    if not list(DATA_DIR.glob("*.pdf")):
        st.warning("Please upload at least one PDF before asking questions.")
        st.stop()

    # Append user message to history
    st.session_state.chat_history.append(
        {"role": "user", "content": user_question}
    )

    # --- Run the multi-agent pipeline ---
    with st.spinner("Running multi-agent clinical reasoning pipeline…"):
        try:
            result = run(user_question)
        except FileNotFoundError as exc:
            st.error(f"Ingestion error: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"An error occurred: {exc}")
            st.stop()

    final_answer = result.get("final_answer", "No answer generated.")
    confidence = result.get("confidence", "MEDIUM")
    agent_trace = result.get("agent_trace", [])

    # Append assistant response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": final_answer,
        "meta": {
            "confidence": confidence,
            "agent_trace": agent_trace,
        },
    })

    # Rerun to render the new messages
    st.rerun()


# ---------------------------------------------------------------------------
# Clear conversation button
# ---------------------------------------------------------------------------
if st.session_state.chat_history:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    🔬 <strong>ClinicalMind</strong> — Research tool only. Not medical advice. Not for clinical use.<br>
    Built at <strong>NCAI Pakistan</strong> · Powered by GPT-4o-mini + LangGraph + FAISS
</div>
""", unsafe_allow_html=True)
