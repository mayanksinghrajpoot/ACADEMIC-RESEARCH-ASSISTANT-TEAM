"""Specialized agents. Each module owns one role in the pipeline."""

from research_assistant.agents.contradiction_agent import ContradictionAgent
from research_assistant.agents.factcheck_agent import FactCheckAgent
from research_assistant.agents.query_agent import QueryAgent
from research_assistant.agents.retrieval_agent import RetrievalAgent
from research_assistant.agents.summary_agent import SummaryAgent
from research_assistant.agents.writer_agent import WriterAgent

__all__ = [
    "ContradictionAgent",
    "FactCheckAgent",
    "QueryAgent",
    "RetrievalAgent",
    "SummaryAgent",
    "WriterAgent",
]
