"""
agents.py — Five Specialised Clinical Reasoning Agents for ClinicalMind
------------------------------------------------------------------------
Each agent is a plain Python function that accepts a LangGraph state dict
and returns a partial state update.  They are wired together in orchestrator.py
using a StateGraph.

Agent pipeline:
  retrieval_agent  → raw evidence chunks from FAISS
  guideline_agent  → protocol / recommendation passages from FAISS
  analysis_agent   → clinical reasoning over the evidence
  critic_agent     → hallucination check + confidence rating
  aggregator_agent → final synthesised answer with citations & disclaimer

All agents use GPT-4o-mini at temperature=0 for deterministic, conservative output.
They NEVER diagnose — they only retrieve, reason, and cite.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ingest import load_vectorstore

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
load_dotenv()

# Shared LLM — temperature=0 keeps reasoning grounded and reproducible
_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Vectorstore is loaded once and reused by every agent that needs it
_VECTORSTORE = None


def _get_vectorstore():
    """Lazy-load the vectorstore so startup is fast even when not querying."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = load_vectorstore()
    return _VECTORSTORE


def _format_chunks(docs: list) -> str:
    """
    Serialise retrieved Document objects into a human-readable block that
    the LLM can cite.  Includes source file name and page number.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] (Source: {source}, Page: {page})\n{doc.page_content.strip()}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Agent 1 — Retrieval Agent
# ---------------------------------------------------------------------------

def retrieval_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Searches the FAISS vectorstore for the top-5 chunks most relevant to the
    user's question.  Returns the raw text with source + page citations so
    downstream agents have a grounded evidence base.
    """
    question = state["question"]
    vs = _get_vectorstore()

    # Similarity search: retrieve 5 chunks (k=5 balances breadth vs. noise)
    docs = vs.similarity_search(question, k=5)
    retrieved_text = _format_chunks(docs)

    # Brief LLM pass to confirm relevance and summarise what was found
    messages = [
        SystemMessage(content=(
            "You are a clinical evidence retrieval specialist. "
            "Your ONLY job is to report what evidence was found in the provided chunks. "
            "Do NOT add medical knowledge not present in the chunks. "
            "Do NOT diagnose. Cite every chunk by its [number]."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Retrieved chunks:\n{retrieved_text}\n\n"
            "Summarise what relevant evidence the chunks contain, citing each by [number]."
        )),
    ]
    response = _LLM.invoke(messages)

    return {
        "retrieved_chunks": retrieved_text,
        "retrieval_summary": response.content,
        "agent_trace": state.get("agent_trace", []) + [
            {"agent": "retrieval_agent", "output": response.content}
        ],
    }


# ---------------------------------------------------------------------------
# Agent 2 — Guideline Agent
# ---------------------------------------------------------------------------

def guideline_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Performs a second FAISS search specifically optimised to surface clinical
    guidelines, protocols, and dosing recommendations.  Uses an augmented
    query to bias retrieval toward guideline language.
    """
    question = state["question"]
    vs = _get_vectorstore()

    # Augment the query with guideline-oriented terms so FAISS scores
    # protocol passages more highly than general background text
    guideline_query = (
        f"{question} clinical guideline protocol recommendation dosage "
        "treatment criteria management"
    )
    docs = vs.similarity_search(guideline_query, k=5)
    guideline_text = _format_chunks(docs)

    messages = [
        SystemMessage(content=(
            "You are a clinical guideline extraction specialist. "
            "Extract ONLY protocol statements, dosing ranges, treatment steps, "
            "and grading recommendations from the provided chunks. "
            "Cite every chunk by its [number]. Do NOT diagnose."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Guideline chunks:\n{guideline_text}\n\n"
            "List the relevant guideline statements and protocols found."
        )),
    ]
    response = _LLM.invoke(messages)

    return {
        "guideline_chunks": guideline_text,
        "guideline_summary": response.content,
        "agent_trace": state.get("agent_trace", []) + [
            {"agent": "guideline_agent", "output": response.content}
        ],
    }


# ---------------------------------------------------------------------------
# Agent 3 — Analysis Agent
# ---------------------------------------------------------------------------

def analysis_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Reasons over the outputs of the retrieval and guideline agents to draw
    clinical connections, identify patterns, and explain what the evidence
    collectively suggests.  Does NOT add external knowledge.
    """
    question = state["question"]
    retrieval_summary = state.get("retrieval_summary", "")
    guideline_summary = state.get("guideline_summary", "")

    messages = [
        SystemMessage(content=(
            "You are a clinical evidence analyst. "
            "Your role is to reason over ONLY the evidence summaries provided. "
            "Identify convergences, gaps, and clinical implications. "
            "Do NOT introduce facts, drugs, or recommendations not present in the summaries. "
            "Do NOT diagnose. Use hedged language (e.g., 'the evidence suggests')."
        )),
        HumanMessage(content=(
            f"Clinical Question: {question}\n\n"
            f"Evidence Summary:\n{retrieval_summary}\n\n"
            f"Guideline Summary:\n{guideline_summary}\n\n"
            "Provide a structured clinical analysis connecting these findings. "
            "Note any gaps or areas where evidence is weak or absent."
        )),
    ]
    response = _LLM.invoke(messages)

    return {
        "analysis": response.content,
        "agent_trace": state.get("agent_trace", []) + [
            {"agent": "analysis_agent", "output": response.content}
        ],
    }


# ---------------------------------------------------------------------------
# Agent 4 — Critic Agent
# ---------------------------------------------------------------------------

def critic_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Reviews the analysis for statements not grounded in the retrieved chunks.
    Flags potential hallucinations and assigns an overall confidence rating:
      HIGH   — all claims traceable to retrieved evidence
      MEDIUM — most claims supported; minor gaps
      LOW    — significant unsupported claims detected
    """
    analysis = state.get("analysis", "")
    retrieved_chunks = state.get("retrieved_chunks", "")
    guideline_chunks = state.get("guideline_chunks", "")

    all_evidence = f"{retrieved_chunks}\n\n{guideline_chunks}"

    messages = [
        SystemMessage(content=(
            "You are a rigorous clinical fact-checker. "
            "Compare the analysis against the raw retrieved chunks. "
            "Identify ANY claim in the analysis that is not directly supported by the chunks. "
            "At the end output a confidence rating: HIGH, MEDIUM, or LOW.\n"
            "Format:\n"
            "GROUNDED CLAIMS: <list>\n"
            "UNSUPPORTED CLAIMS: <list or 'None detected'>\n"
            "CONFIDENCE: <HIGH | MEDIUM | LOW>\n"
            "REASON: <one sentence explaining the rating>"
        )),
        HumanMessage(content=(
            f"Analysis to check:\n{analysis}\n\n"
            f"Source evidence chunks:\n{all_evidence}"
        )),
    ]
    response = _LLM.invoke(messages)

    # Parse confidence level from the structured response
    confidence = "MEDIUM"  # safe default
    for line in response.content.splitlines():
        if line.strip().startswith("CONFIDENCE:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("HIGH", "MEDIUM", "LOW"):
                confidence = val
                break

    return {
        "critique": response.content,
        "confidence": confidence,
        "agent_trace": state.get("agent_trace", []) + [
            {"agent": "critic_agent", "output": response.content, "confidence": confidence}
        ],
    }


# ---------------------------------------------------------------------------
# Agent 5 — Aggregator Agent
# ---------------------------------------------------------------------------

def aggregator_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Synthesises all previous agent outputs into a single, cohesive answer.
    Includes:
      • A concise evidence-based response
      • Inline citations ([1], [2], …) linked to source files and pages
      • The confidence level from the critic
      • A mandatory disclaimer that this is not medical advice
    """
    question = state["question"]
    retrieval_summary = state.get("retrieval_summary", "")
    guideline_summary = state.get("guideline_summary", "")
    analysis = state.get("analysis", "")
    critique = state.get("critique", "")
    confidence = state.get("confidence", "MEDIUM")

    messages = [
        SystemMessage(content=(
            "You are the final synthesiser for a clinical reasoning system. "
            "Produce ONE clear, well-structured answer using ONLY the information provided. "
            "Structure:\n"
            "1. Direct answer (2-3 sentences)\n"
            "2. Supporting evidence (bullet points with inline citations)\n"
            "3. Relevant guideline recommendations\n"
            "4. Caveats or evidence gaps\n"
            "5. Disclaimer: 'This response is for research and informational purposes only. "
            "It is NOT medical advice. Always consult a qualified healthcare professional.'\n\n"
            "Do NOT diagnose. Do NOT invent citations. "
            f"Confidence level of this response: {confidence}"
        )),
        HumanMessage(content=(
            f"Clinical Question: {question}\n\n"
            f"Evidence Summary:\n{retrieval_summary}\n\n"
            f"Guideline Summary:\n{guideline_summary}\n\n"
            f"Clinical Analysis:\n{analysis}\n\n"
            f"Fact-check / Critique:\n{critique}\n\n"
            "Synthesise all of the above into the final answer."
        )),
    ]
    response = _LLM.invoke(messages)

    return {
        "final_answer": response.content,
        "agent_trace": state.get("agent_trace", []) + [
            {"agent": "aggregator_agent", "output": response.content}
        ],
    }
