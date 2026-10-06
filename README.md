# genai-capstone

Learning LLMs, embeddings, RAG, vector search and agents by building one engine with two tracks — locally, at $0.

| Week | Theme | Status |
|---|---|---|
| 1 | How LLMs see text | in progress |
| 2 | Inside the transformer | — |
| 3 | Embeddings & vector search | — |
| 4 | RAG v1 + evals | — |
| 5 | Better retrieval | — |
| 6 | Multimodal + LLMs in pipelines | — |
| 7 | Text-to-SQL | — |
| 8 | Tools & agents from scratch | — |
| 9 | LangGraph workflows | — |
| 10 | MCP + guardrails | — |
| 11 | Fine-tuning | — |
| 12 | Production & AWS | — |

## Run it

    uv sync
    uv run pytest              # unit tests — no Ollama needed
    uv run pytest -m ollama    # integration tests — Ollama must be running
