# Hive Research

Lightweight research knowledge base for **Apple Silicon** using **Ollama** local LLMs and the **Hive** datatype for knowledge graphs.

Born from [KG-HD-Research](https://github.com/anomalyco/KG-HD-Research) — a streamlined, zero-Docker version focused on Apple Metal acceleration.

## Features

- **arXiv Ingestion** — search, fetch metadata, download PDFs by ID or query
- **LLM Analysis** — extracts tags, concepts, relations, and summaries via Ollama (two-model pipeline: fast + main)
- **Knowledge Graph** — typed directed multigraph using [hive-datatype](https://github.com/anomalyco/hive-datatype) (`paper`, `concept` nodes; 11 relation types)
- **RAG** — local embedding search (Ollama `nomic-embed-text` + numpy cosine similarity) with LLM answer generation and source citations
- **Web Dashboard** — D3.js force-directed graph, arXiv search, paper browser, similarity matrix, RAG chat, Apple Silicon GPU monitoring
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
└──────────┘  └──────────┘  └─────────────┘  └──────────┘  └──────────┘
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

```bash
git clone https://github.com/your-org/hive-research.git
cd hive-research
pip install -e .
```

The project depends on [hive-datatype](https://github.com/anomalyco/hive-datatype) which is auto-discovered from `../hive-datatype` relative to the project root.

## Configuration

Edit `config.yaml`:

```yaml
ollama:
  base_url: http://localhost:11434
  model: qwen3.6:35b-mlx       # main model for concept extraction
  fast_model: llama3.2:3b      # fast model for tag extraction
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
| **Add** | Add papers by arXiv ID/URL with activity log |
| **Search** | Search arXiv, browse results, bulk import |
| **Browse** | Browse ingested papers and extracted concepts |
| **Similarity** | Jaccard paper-paper similarity matrix |
| **Chat** | RAG question-answering with source citations |
| **About** | System info, Ollama status, Apple Silicon GPU detection |

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
| POST | `/api/add` | Add paper by `{id: "arxiv_id"}` |
| POST | `/api/search` | Search arXiv `{query: "..."}` |
| POST | `/api/import` | Search + import all `{query: "..."}` |
| POST | `/api/query` | RAG question `{question: "..."}` |

## Project Structure

```
hive-research/
├── config.yaml                         # Configuration
├── pyproject.toml                      # Package metadata
├── hive_research/
│   ├── __init__.py                     # Package init + hive-datatype path
│   ├── __main__.py                     # CLI entry point
│   ├── config.py                       # Config loader (YAML + env vars)
│   ├── arxiv_fetcher.py               # arXiv API client
│   ├── parser.py                       # PyMuPDF text extraction
│   ├── llm.py                          # Ollama REST wrapper
│   ├── graph.py                        # Knowledge graph (wraps HiveGraph)
│   ├── pipeline.py                     # LLM analysis → graph → notes
│   ├── similarity.py                   # Jaccard paper similarity
│   ├── rag.py                          # Local RAG engine
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
    └── rag/                            # Embedding index
```

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
