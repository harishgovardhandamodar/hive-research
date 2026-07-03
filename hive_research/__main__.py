from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Config
from .organizer import Organizer

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_search(args: argparse.Namespace) -> None:
    config = Config()
    org = Organizer(config)
    results = org.search(args.query, max_results=args.max_results)
    if not results:
        print("No results found.")
        return
    for p in results:
        print(f"\n[{p['arxiv_id']}] {p['title']}")
        print(f"    Authors: {p['authors'][:120]}")
        print(f"    Published: {p['published']}")
        print(f"    Categories: {', '.join(p['categories'])}")
        print(f"    Abstract: {p['abstract'][:200]}...")


def cmd_add(args: argparse.Namespace) -> None:
    config = Config()
    org = Organizer(config)
    result = org.add_by_id(args.id)
    print(json.dumps(result, indent=2))


def cmd_import_search(args: argparse.Namespace) -> None:
    config = Config()
    org = Organizer(config)
    results = org.add_by_search(args.query, max_results=args.max_results)
    print(json.dumps(results, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    config = Config()
    org = Organizer(config)
    stats = org.stats()
    print(json.dumps(stats, indent=2))


def cmd_similarity(args: argparse.Namespace) -> None:
    config = Config()
    org = Organizer(config)
    sim = org.similarity()
    if not sim:
        print("No papers to compare.")
        return
    for s in sim[:10]:
        print(f"  {s['source_title'][:50]}  ↔  {s['target_title'][:50]}  = {s['score']:.3f}")


def cmd_query(args: argparse.Namespace) -> None:
    config = Config()
    org = Organizer(config)
    result = org.query_rag(args.question)
    print(f"\nAnswer: {result['answer']}")
    if result.get("sources"):
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['id']}] {s['title']}")


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import run_server

    config = Config()
    org = Organizer(config)
    run_server(org, host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hive Research — lightweight research knowledge base",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Search arXiv")
    p_search.add_argument("query", type=str)
    p_search.add_argument("-n", "--max-results", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    p_add = sub.add_parser("add", help="Add paper by arXiv ID")
    p_add.add_argument("id", type=str)
    p_add.set_defaults(func=cmd_add)

    p_import_ = sub.add_parser("import", help="Search and import papers")
    p_import_.add_argument("query", type=str)
    p_import_.add_argument("-n", "--max-results", type=int, default=10)
    p_import_.set_defaults(func=cmd_import_search)

    p_stats = sub.add_parser("stats", help="Show knowledge graph stats")
    p_stats.set_defaults(func=cmd_stats)

    p_sim = sub.add_parser("similarity", help="Paper similarity matrix")
    p_sim.set_defaults(func=cmd_similarity)

    p_query = sub.add_parser("query", help="Ask a RAG question")
    p_query.add_argument("question", type=str)
    p_query.set_defaults(func=cmd_query)

    p_serve = sub.add_parser("serve", help="Start web server")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
