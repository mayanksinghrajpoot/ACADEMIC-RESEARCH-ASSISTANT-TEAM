# Academic Research Assistant Team

A multi-agent system that turns a research question into a short, sourced literature summary. Every claim that remains in the summary maps to a passage retrieved from a local paper corpus. Unsupported claims are listed separately, and disagreements between sources are reported with both positions.

The coordination layer is [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/) (the successor to Semantic Kernel and AutoGen). Agents run as a workflow pipeline with retrieval-augmented generation and a fact-check critic.

## Setup

Requires Python 3.10+.

```bash
uv sync --extra dev
cp .env.example .env
# set OPENAI_API_KEY in .env
```

`OPENAI_BASE_URL` is optional. Set it when the chat endpoint is an OpenAI-compatible gateway (including Azure OpenAI-compatible routes).

## Run

```bash
uv run python -m research_assistant "Does retrieval-augmented generation reduce unsupported claims in scientific question answering?"
```

The command writes:

- `outputs/<run_id>.md` — themed summary, per-claim sources, contradictions, and dropped claims
- `outputs/<run_id>.json` — the shared dossier, from sub-queries through verdicts

## Agents

| Agent | Module | Responsibility |
| --- | --- | --- |
| Query | `agents/query_agent.py` | Turns the question into search sub-queries over the provided papers |
| Retrieval | `agents/retrieval_agent.py` | Fetches relevant passages for each sub-query |
| Summary | `agents/summary_agent.py` | Writes a structured summary grouped by theme |
| Contradiction | `agents/contradiction_agent.py` | Reports where two sources disagree and presents both positions |
| Fact-check | `agents/factcheck_agent.py` | Verifies each claim against retrieved passages |
| Writer | `agents/writer_agent.py` | Publishes the final summary, keeping only supported claims |

Details of the pipeline, the shared dossier, and the demo corpus are filled in with the implementation.
