"""
LangGraph Workflow Definition.

Graph topology:

  START
    │
    ▼
  segregator_node          ← classifies every page
    │
    ├──────────────────────┐──────────────────────┐
    ▼                      ▼                      ▼
  id_agent_node    discharge_agent_node   bill_agent_node
    │                      │                      │
    └──────────────────────┴──────────────────────┘
                           │
                           ▼
                    aggregator_node
                           │
                           ▼
                          END

The three agent nodes execute in parallel (LangGraph fan-out).
Each writes to isolated state keys so there are no race conditions.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.agents.aggregator import aggregator_node
from app.agents.bill_agent import bill_agent_node
from app.agents.discharge_agent import discharge_agent_node
from app.agents.id_agent import id_agent_node
from app.agents.segregator import segregator_node
from app.graph.state import ClaimState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph ready for `.invoke()` or `.ainvoke()`.
    """
    builder = StateGraph(ClaimState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("segregator",        segregator_node)
    builder.add_node("id_agent",          id_agent_node)
    builder.add_node("discharge_agent",   discharge_agent_node)
    builder.add_node("bill_agent",        bill_agent_node)
    builder.add_node("aggregator",        aggregator_node)

    # ── Wire edges ────────────────────────────────────────────────────────
    # Entry point
    builder.add_edge(START, "segregator")

    # Fan-out: segregator → all three agents simultaneously
    builder.add_edge("segregator", "id_agent")
    builder.add_edge("segregator", "discharge_agent")
    builder.add_edge("segregator", "bill_agent")

    # Fan-in: all three agents → aggregator
    builder.add_edge("id_agent",        "aggregator")
    builder.add_edge("discharge_agent", "aggregator")
    builder.add_edge("bill_agent",      "aggregator")

    # Terminal
    builder.add_edge("aggregator", END)

    compiled = builder.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled


# Module-level singleton — compiled once at import time
claims_graph = build_graph()