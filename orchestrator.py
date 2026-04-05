"""
orchestrator.py — Master Orchestrator for ClinicalMind
--------------------------------------------------------
Uses a LangGraph StateGraph to coordinate the five agents in a fixed
sequential pipeline:

    User question
        │
        ▼
    retrieval_agent   — FAISS similarity search (general evidence)
        │
        ▼
    guideline_agent   — FAISS search biased toward protocols/guidelines
        │
        ▼
    analysis_agent    — Clinical reasoning over retrieved evidence
        │
        ▼
    critic_agent      — Hallucination check + confidence rating
        │
        ▼
    aggregator_agent  — Final synthesised answer with citations
        │
        ▼
    Final answer + full agent trace returned to caller

The orchestrator is imported by app.py and can also be run standalone
for command-line testing.
"""

from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()

from agents import (
    aggregator_agent,
    analysis_agent,
    critic_agent,
    guideline_agent,
    retrieval_agent,
)


# ---------------------------------------------------------------------------
# Shared State Schema
# ---------------------------------------------------------------------------
# TypedDict gives LangGraph and type-checkers a clear contract for what lives
# in the state dict as it flows through the graph nodes.

class ClinicalState(TypedDict, total=False):
    # Input
    question: str

    # Retrieval agent outputs
    retrieved_chunks: str
    retrieval_summary: str

    # Guideline agent outputs
    guideline_chunks: str
    guideline_summary: str

    # Analysis agent output
    analysis: str

    # Critic agent outputs
    critique: str
    confidence: str            # "HIGH" | "MEDIUM" | "LOW"

    # Aggregator output
    final_answer: str

    # Accumulated trace from all agents
    agent_trace: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Each node is a plain Python function (agent) that:
      • receives the current state dict
      • returns a partial dict with only the keys it modifies

    LangGraph merges these partial updates into the shared state.
    """
    graph = StateGraph(ClinicalState)

    # Register each agent as a named node
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("guideline", guideline_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("aggregator", aggregator_agent)

    # Wire the sequential pipeline
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "guideline")
    graph.add_edge("guideline", "analysis")
    graph.add_edge("analysis", "critic")
    graph.add_edge("critic", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# Compile once at import time so repeated calls share the same object
_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(question: str) -> dict[str, Any]:
    """
    Run the full multi-agent pipeline for a clinical question.

    Args:
        question: The clinical question from the user.

    Returns:
        A dict containing:
          • "final_answer"  — the synthesised response (str)
          • "confidence"    — HIGH / MEDIUM / LOW (str)
          • "agent_trace"   — list of per-agent dicts with agent name + output
          • all intermediate state keys for inspection
    """
    initial_state: ClinicalState = {
        "question": question,
        "agent_trace": [],
    }

    # invoke() runs the entire graph synchronously and returns the final state
    final_state = _GRAPH.invoke(initial_state)
    return final_state


# ---------------------------------------------------------------------------
# CLI entry-point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What are the first-line treatments for hypertension?"
    print(f"\nQuestion: {question}\n{'─' * 60}")

    result = run(question)

    print("\n=== AGENT TRACE ===")
    for step in result.get("agent_trace", []):
        print(f"\n[{step['agent']}]")
        print(step["output"])
        if "confidence" in step:
            print(f"  → Confidence: {step['confidence']}")

    print(f"\n{'─' * 60}")
    print(f"CONFIDENCE: {result.get('confidence', 'N/A')}")
    print(f"\n=== FINAL ANSWER ===\n{result.get('final_answer', '')}")
