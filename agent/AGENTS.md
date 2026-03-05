# AGENT MODULE (agent/)

## OVERVIEW
LangGraph-based agent orchestration with nodes, edges, prompts, and tool wiring.

## STRUCTURE
```
agent/
├── orchestrator_agent.py  # agent creation + middleware
├── graph.py               # graph assembly
├── nodes.py               # node behaviors
├── edges.py               # routing logic
├── graph_state.py         # state schema
├── prompts.py             # prompt templates
├── tools.py               # tool factory
└── schemas.py             # pydantic output schemas
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Agent creation | orchestrator_agent.py | middleware + tool limits |
| Graph wiring | graph.py | node/edge graph compile |
| Node logic | nodes.py | summarize/rewrite/aggregate |
| Prompt text | prompts.py | keep prompts centralized |
| Tool exposure | tools.py | tool wrapper around retriever |
| Output schemas | schemas.py | QueryAnalysis schema |

## CONVENTIONS
- Use `prompts.py` for all prompt text; keep nodes/tools lean.
- Graph nodes are pure functions returning state updates.

## ANTI-PATTERNS
- Avoid embedding prompt strings directly in nodes; update `prompts.py` instead.
