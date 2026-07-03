# Hive Research

Lightweight research knowledge base for **Apple Silicon** using **Ollama** local models and the **Hive** datatype for knowledge graphs.

## Quickstart

```bash
pip install -e .
python -m hive_research search --query "attention is all you need"
python -m hive_research add --id 1706.03762
python -m hive_research serve
```

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) running locally with models pulled
- Apple Silicon (Metal GPU acceleration via Ollama)

## Config

Edit `config.yaml` or set environment variables (`OLLAMA_MODEL`, `OLLAMA_FAST_MODEL`, `OLLAMA_BASE_URL`, etc).
