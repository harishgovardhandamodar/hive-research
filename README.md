# Hive Research

Lightweight research knowledge base for **Apple Silicon** using **Ollama** local LLMs and the **Hive** datatype for knowledge graphs.

Born from [KG-HD-Research](https://github.com/anomalyco/KG-HD-Research) — streamlined, zero-Docker version focused on Apple Metal acceleration. Optionally deployable via Docker for ARM64.

## Features

- **arXiv Ingestion** — search, fetch metadata, download PDFs by ID or query
- **LLM Analysis** — extracts tags, concepts, relations, and summaries via Ollama (two-model pipeline: fast + main)
- **Knowledge Graph** — typed directed multigraph using [hive-datatype](https://github.com/anomalyco/hive-datatype) (`paper`, `concept` nodes; 11 relation types)
- **RAG** — local embedding search (Ollama `nomic-embed-text` + numpy cosine similarity) with LLM answer generation and source citations
- **Research Pool** — arXiv topic observatory with 8 default topics, background refresh every 12h, observed/imported paper tracking, Jaccard similarity graph
- **Web Dashboard** — D3.js force-directed graph, arXiv search, paper browser, similarity matrix, RAG chat, pool browse/list/graph views, Apple Silicon GPU monitoring, persistent activity log
- **Obsidian Export** — per-paper markdown notes with YAML frontmatter

## Architecture

```
                             ┌─────────────┐
                             │   Ollama     │
                             │ (localhost)  │
                             └──────┬──────┘
                                    │
┌──────────┐  ┌──────────┐  ┌──────┴──────┐  ┌──────────┐  ┌──────────┐
│  arXiv    │  │  PyMuPDF │  │     LLM     │  │   Hive   │  │   RAG    │
│  Fetcher  │─▶│  Parser  │─▶│  Interface  │─▶│  Graph   │─▶│  Engine  │
└──────────┘  └──────────┘  └─────────────┘  └──────┬───┘  └──────────┘
                                                    │
                                           ┌────────┴────────┐
                                           │    Research      │
                                           │      Pool        │
                                           │ (SQLite + arXiv) │
                                           └────────┬────────┘
                                                    │
                                              ┌─────┴─────┐
                                              │  Server   │
                                              │ (Dashboard)│
                                              └───────────┘
```

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) running locally
- Apple Silicon (Metal GPU acceleration via Ollama — also works on Intel Mac/Linux)
- Models pulled: `llama3.2:3b` (fast), any main model, `nomic-embed-text` (embeddings)

## Installation

### Native (macOS/Linux)

```bash
git clone https://github.com/your-org/hive-research.git
cd hive-research
pip install -e .
```

The project depends on [hive-datatype](https://github.com/anomalyco/hive-datatype) which is auto-discovered from `../hive-datatype` relative to the project root.

### Docker (Apple Silicon ARM64)

```bash
docker compose up --build
# Open http://127.0.0.1:7777
```

This builds a `linux/arm64` image, connects to Ollama on the host via `host.docker.internal:11434`, and persists data in a Docker volume.

Override models:

```bash
OLLAMA_MODEL=gemma4:31b-mlx docker compose up --build
```

## Configuration

Edit `config.yaml`:

```yaml
server:
  host: 127.0.0.1
  port: 7777

ollama:
  base_url: http://localhost:11434
  model: llama3.2:3b          # main model for concept extraction
  fast_model: llama3.2:3b     # fast model for tag extraction
  embed_model: nomic-embed-text # embedding model for RAG
```

Override via environment variables: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_FAST_MODEL`, `OLLAMA_EMBED_MODEL`.

## Usage

### CLI

```bash
# Search arXiv
python -m hive_research search "attention mechanism" -n 5

# Add a paper by arXiv ID
python -m hive_research add 1706.03762

# Search and import multiple papers
python -m hive_research import "graph neural networks" -n 10

# Show knowledge graph stats
python -m hive_research stats

# Paper similarity matrix
python -m hive_research similarity

# Ask a RAG question
python -m hive_research query "What is the main contribution of this paper?"

# Launch web dashboard
python -m hive_research serve

# Verbose logging
python -m hive_research -v add 2409.13004
```

### Web Dashboard

```bash
python -m hive_research serve
python -m hive_research serve --port 7777
# Open http://127.0.0.1:7777
```

| Panel | Description |
|---|---|
| **Graph** | D3 force-directed knowledge graph — drag nodes, hover for tooltips, filter bar |
| **Add** | Add papers by arXiv ID/URL with live activity log |
| **Search** | Search arXiv, browse results, bulk import |
| **Browse** | Browse ingested papers and extracted concepts |
| **Pool** | Research observatory — Browse (topic feed), List (observed papers table), Graph (Jaccard similarity D3), Settings (topic CRUD) |
| **Similarity** | Jaccard paper-paper similarity matrix |
| **Chat** | RAG question-answering with source citations |
| **About** | System info, Ollama status, Apple Silicon GPU detection |
| **Activity Log** | Persistent bottom bar showing server logs in real-time (collapsible) |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/graph` | Knowledge graph in node-link JSON format |
| GET | `/api/stats` | Paper/concept/edge/RAG counts |
| GET | `/api/papers` | List of ingested papers |
| GET | `/api/concepts` | List of extracted concepts |
| GET | `/api/similarity` | Paper-pair similarity matrix |
| GET | `/api/ollama` | Ollama connection & model availability |
| GET | `/api/gpu` | Apple Silicon / Metal GPU detection |
| GET | `/api/logs` | Recent server log entries |
| GET | `/api/pool` | arXiv feed grouped by topic |
| GET | `/api/pool/papers` | Observed papers with import status |
| GET | `/api/pool/graph` | Jaccard similarity graph of pool papers |
| GET | `/api/pool/topics` | List configured topics |
| POST | `/api/add` | Add paper by `{id: "arxiv_id"}` |
| POST | `/api/search` | Search arXiv `{query: "..."}` |
| POST | `/api/import` | Search + import all `{query: "..."}` |
| POST | `/api/query` | RAG question `{question: "..."}` |
| POST | `/api/pool/topics/add` | Add topic `{name, query}` |
| POST | `/api/pool/topics/remove` | Remove topic `{name}` |
| POST | `/api/pool/import` | Import pool paper `{arxiv_id}` |

## Project Structure

```
hive-research/
├── Dockerfile                          # Multi-stage ARM64 Docker build
├── docker-compose.yml                  # Apple Silicon Docker Compose
├── config.yaml                         # Configuration
├── pyproject.toml                      # Package metadata
├── hive_research/
│   ├── __init__.py                     # Package init + hive-datatype resolution
│   ├── __main__.py                     # CLI entry point
│   ├── config.py                       # Config loader (YAML + env vars)
│   ├── arxiv_fetcher.py               # arXiv API client
│   ├── parser.py                       # PyMuPDF text extraction
│   ├── llm.py                          # Ollama REST wrapper
│   ├── graph.py                        # Knowledge graph (wraps HiveGraph)
│   ├── pipeline.py                     # LLM analysis → graph → notes
│   ├── similarity.py                   # Jaccard paper similarity
│   ├── rag.py                          # Local RAG engine
│   ├── pool.py                         # Research Pool (SQLite-backed)
│   ├── logs.py                         # LogCapture handler for dashboard
│   ├── organizer.py                    # Central orchestrator
│   ├── server.py                       # HTTP server + REST API
│   ├── dashboard.html                  # Web dashboard
│   └── tests/
│       ├── __init__.py
│       └── bench_ingestion.py          # End-to-end ingestion benchmark
└── data/                               # Created at runtime
    ├── papers/                         # Downloaded PDFs
    ├── graph/                          # Knowledge graph JSON
    ├── vault/                          # Markdown notes
    ├── rag/                            # Embedding index
    └── pool/                           # SQLite database (pool.db)
```

## Research Pool

The pool is a topic-based arXiv observatory that continuously discovers papers:

- **8 default topics**: Knowledge graphs, Federated learning, AI security, LLM security, AI alignment, Adversarial ML, Graph neural networks, Vision-language models
- **Background refresh**: Daemon thread fetches arXiv for all topics every 12 hours (staggered 4s between topics to avoid rate limits)
- **Local SQLite storage**: All observed papers, topic configs, and feed cache stored in `pool.db`
- **Import tracking**: Papers tracked with `imported` status; "Add to KB" imports into the knowledge graph
- **Pool graph**: Jaccard similarity edges (threshold ≥ 0.12) between papers, colored by topic in D3

## Benchmarks

Run the ingestion benchmark:

```bash
python -m hive_research.tests.bench_ingestion --arxiv 2409.13004
```

Example output on Apple M2 Max (96 GB) with Ollama `llama3.2:3b`:

| Stage | Duration |
|---|---|
| arXiv fetch | 61 ms |
| PDF download (2.3 MB) | 104 ms |
| Text extraction (102K chars) | 177 ms |
| Tag extraction (fast LLM) | 353 ms |
| Concept extraction (main LLM) | 2.3 s |
| Graph population | 1 ms |
| Note writing | 0 ms |
| RAG indexing (35 chunks) | 1.6 s |
| **Total** | **4.6 s** |

## Dependencies

```
arxiv>=2.1.0, PyMuPDF>=1.25.0, requests>=2.32.0,
PyYAML>=6.0.0, numpy>=1.24.0
```

Plus [hive-datatype](https://github.com/anomalyco/hive-datatype) for the knowledge graph data model.

## Acknowledgments

- [KG-HD-Research](https://github.com/anomalyco/KG-HD-Research) — the original containerized research knowledge base
- [hive-datatype](https://github.com/anomalyco/hive-datatype) — the Hive knowledge graph data model
- [Ollama](https://ollama.ai) — local LLM runtime
- [D3.js](https://d3js.org) — graph visualization
