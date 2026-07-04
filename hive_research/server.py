from __future__ import annotations

import json
import logging
import os
import platform
import re
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import requests

from .logs import get_capture
from .organizer import Organizer

logger = logging.getLogger(__name__)

HTML = Path(__file__).parent / "dashboard.html"


def _json_response(
    handler: BaseHTTPRequestHandler,
    data: Any,
    status: int = 200,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(json.dumps(data).encode())


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.end_headers()
    handler.wfile.write(html.encode())


class RouteHandler(BaseHTTPRequestHandler):
    org: Organizer = None  # type: ignore[assignment]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode() if length else ""

    def _parse_path(self) -> tuple[str, dict[str, str]]:
        parts = self.path.split("?", 1)
        path = parts[0].rstrip("/")
        params: dict[str, str] = {}
        if len(parts) > 1:
            for kv in parts[1].split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    params[k] = urllib.parse.unquote(v)
        return path, params

    def do_GET(self) -> None:
        path, params = self._parse_path()
        if path == "/" or path == "" or path == "/index.html":
            self._serve_dashboard()
        elif path == "/debug/graph":
            self._serve_debug_graph()
        elif path == "/api/graph":
            _json_response(self, self.org.graph_data())
        elif path == "/api/stats":
            _json_response(self, self.org.stats())
        elif path == "/api/similarity":
            paper_ids = data.get("paper_ids", params.get("paper_ids", None))
            algorithm = data.get("algorithm", params.get("algorithm", "combined"))
            if isinstance(paper_ids, str):
                paper_ids = [x.strip() for x in paper_ids.split(",") if x.strip()]
            _json_response(self, self.org.similarity(paper_ids=paper_ids, algorithm=algorithm))
        elif path == "/api/vectors/status":
            _json_response(self, self.org.vectors.status())
        elif path == "/api/vectors/project":
            _json_response(self, self.org.project_vectors())
        elif path == "/api/papers":
            from .pipeline import _sanitize_id
            has_lineage = set()
            for e in self.org.kg._hive.edges:
                if e.relation == "cites":
                    has_lineage.add(e.source)
            papers = []
            vault_dir = self.org.config.vault_dir
            all_nodes = list(self.org.kg.papers) + [n for n in self.org.kg._hive.nodes if n.type == "web"]
            for n in all_nodes:
                safe = _sanitize_id(n.label) or n.id
                notes_file = Path(vault_dir) / safe / "00_notes.md"
                if notes_file.exists():
                    note_path = str(notes_file)
                    note_dir = str(Path(vault_dir) / safe)
                else:
                    legacy = Path(vault_dir) / f"{safe}.md"
                    if legacy.exists():
                        note_path = str(legacy)
                        note_dir = note_path
                    elif n.type == "web":
                        # For web articles, also try lookup matching the graph node ID suffix
                        suffix = n.id.replace("web_", "")
                        for sub in Path(vault_dir).iterdir():
                            if sub.is_dir() and suffix in sub.name:
                                note_dir = str(sub)
                                note_path = str(sub / "00_notes.md") if (sub / "00_notes.md").exists() else ""
                                break
                        else:
                            note_path = ""
                            note_dir = ""
                    else:
                        note_path = ""
                        note_dir = ""
                papers.append({
                    "id": n.id, "title": n.label,
                    "authors": getattr(n, "authors", ""),
                    "published": getattr(n, "published", ""),
                    "affiliations": getattr(n, "affiliations", ""),
                    "note_path": note_path,
                    "note_dir": note_dir if Path(note_dir).exists() else "",
                    "has_lineage": n.id in has_lineage,
                    "has_extra": bool(n.definition and n.definition.startswith("{")),
                })
            _json_response(self, papers)
        elif path == "/api/papers/search":
            q = params.get("q", "").lower()
            papers = [
                {"id": n.id, "title": n.label, "authors": n.authors, "published": n.published, "affiliations": n.affiliations}
                for n in self.org.kg.papers
                if not q or q in n.label.lower() or q in n.authors.lower() or q in n.affiliations.lower()
            ]
            _json_response(self, papers)
        elif path == "/api/concepts":
            concepts = [
                {"id": n.id, "label": n.label, "definition": n.definition}
                for n in self.org.kg.concepts
            ]
            _json_response(self, concepts)
        elif path == "/api/browse":
            papers_dir = str(self.org.config.papers_dir)
            vault_dir = str(self.org.config.vault_dir)
            tree = []
            def _scan(dirpath):
                entries = []
                try:
                    for entry in sorted(os.listdir(dirpath)):
                        full = os.path.join(dirpath, entry)
                        if os.path.isdir(full):
                            files = []
                            for root, _dirs, filenames in os.walk(full):
                                for fn in sorted(filenames):
                                    rel = os.path.relpath(os.path.join(root, fn), full)
                                    if rel.startswith("."):
                                        continue
                                    ext = os.path.splitext(fn)[1].lower()
                                    files.append({"name": rel, "ext": ext})
                            entries.append({"name": entry, "files": files})
                        else:
                            ext = os.path.splitext(entry)[1].lower()
                            if ext in (".pdf", ".md", ".txt", ".py", ".yaml", ".json", ".html", ".csv"):
                                entries.append({"name": entry, "files": [{"name": entry, "ext": ext}]})
                except Exception as exc:
                    logger.warning("Browse scan error for %s: %s", dirpath, exc)
                return entries
            try:
                tree = _scan(papers_dir)
                # Add vault notes (supports both flat .md and subdirectory structure)
                vault_entries = []
                for entry in sorted(os.listdir(vault_dir)):
                    full = os.path.join(vault_dir, entry)
                    if os.path.isdir(full):
                        files = []
                        for root, _dirs, filenames in os.walk(full):
                            for fn in sorted(filenames):
                                rel = os.path.relpath(os.path.join(root, fn), full)
                                if rel.startswith("."):
                                    continue
                                ext = os.path.splitext(fn)[1].lower()
                                files.append({"name": rel, "ext": ext})
                        if files:
                            vault_entries.append({"name": entry, "files": files})
                    elif entry.endswith(".md"):
                        vault_entries.append({"name": entry, "files": [{"name": entry, "ext": ".md"}]})
                if vault_entries:
                    vault_entries.sort(key=lambda e: e["name"])
                    tree.append({"name": "Notes", "files": vault_entries})
            except Exception as e:
                _json_response(self, {"error": str(e)}, 500)
                return
            _json_response(self, {"tree": tree})
        elif path == "/api/read":
            filepath = params.get("path", "")
            if not filepath:
                _json_response(self, {"error": "missing path"}, 400)
                return
            basedirs = [str(self.org.config.papers_dir), str(self.org.config.vault_dir)]
            content = None
            for basedir in basedirs:
                abspath = os.path.normpath(os.path.join(basedir, filepath))
                if abspath.startswith(os.path.normpath(basedir)) and os.path.isfile(abspath):
                    try:
                        with open(abspath, encoding="utf-8") as f:
                            content = f.read()
                        break
                    except Exception:
                        continue
            if content is None:
                # Also try the vault directory separately for "Notes/" prefix
                if filepath.startswith("Notes/"):
                    stripped = filepath[len("Notes/"):]
                    for basedir in basedirs:
                        abspath = os.path.normpath(os.path.join(basedir, stripped))
                        if abspath.startswith(os.path.normpath(basedir)) and os.path.isfile(abspath):
                            try:
                                with open(abspath, encoding="utf-8") as f:
                                    content = f.read()
                                break
                            except Exception:
                                continue
            if content is None:
                _json_response(self, {"error": "file not found"}, 404)
                return
            _json_response(self, {"path": filepath, "content": content})
        elif path == "/api/raw":
            file_path = params.get("path", "")
            if not file_path:
                _json_response(self, {"error": "missing path"}, 400)
                return
            basedirs = [str(self.org.config.papers_dir), str(self.org.config.vault_dir), "."]
            abspath = os.path.normpath(file_path)
            found = os.path.isfile(abspath)
            if not found:
                for basedir in basedirs:
                    abspath = os.path.normpath(os.path.join(basedir, file_path))
                    if os.path.isfile(abspath):
                        found = True
                        break
            if not found and file_path.startswith("Notes/"):
                stripped = file_path[len("Notes/"):]
                for basedir in [str(self.org.config.vault_dir), str(self.org.config.papers_dir)]:
                    abspath = os.path.normpath(os.path.join(basedir, stripped))
                    if os.path.isfile(abspath):
                        found = True
                        break
            if not found:
                _json_response(self, {"error": "not found"}, 404)
                return
            ext = os.path.splitext(abspath)[1].lstrip(".").lower()
            ct = {
                "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml", "pdf": "application/pdf",
                "md": "text/markdown; charset=utf-8",
                "txt": "text/plain; charset=utf-8",
            }.get(ext, "application/octet-stream")
            try:
                if ext in ("png", "jpg", "jpeg", "gif", "svg", "pdf"):
                    with open(abspath, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", ct)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    content = Path(abspath).read_text(encoding="utf-8")
                    _html_response(self, f"<pre style='background:#0a0e17;color:#e2e8f0;padding:20px;font-size:13px;line-height:1.7;white-space:pre-wrap'>{content}</pre>")
            except Exception as e:
                _json_response(self, {"error": str(e)}, 500)
        elif path == "/api/web/list":
            from hive_datatype import NodeType
            web_nodes = [
                {"id": n.id, "title": n.label, "url": next((l.split("URL:")[1].strip() for l in (n.definition or "").split("\n") if l.startswith("URL:")), ""), "summary": n.abstract[:200]}
                for n in self.org.kg._hive.nodes
                if n.type == "web"
            ]
            _json_response(self, web_nodes)
        elif path == "/api/ollama":
            self._handle_ollama_status()
        elif path == "/api/gpu":
            self._handle_gpu_status()
        elif path == "/api/logs":
            n = int(params.get("n", 100))
            _json_response(self, get_capture().get_recent(n))
        elif path == "/api/pool":
            data = self.org.pool.get()
            _json_response(self, data)
        elif path == "/api/pool/papers":
            papers = self.org.pool.get_observed_papers()
            _json_response(self, papers)
        elif path == "/api/pool/graph":
            graph = self.org.pool.get_pool_graph()
            _json_response(self, graph)
        elif path == "/api/pool/topics":
            topics = self.org.pool.get_topics()
            _json_response(self, {"topics": topics})
        else:
            _json_response(self, {"error": "not found"}, 404)

    def _handle_ollama_status(self) -> None:
        base = self.org.config.ollama_base_url
        model = self.org.config.ollama_model
        fast = self.org.config.ollama_fast_model
        embed = self.org.config.ollama_embed_model
        connected = False
        models = []
        try:
            r = requests.get(f"{base}/api/tags", timeout=5)
            if r.status_code == 200:
                connected = True
                models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        _json_response(self, {
            "connected": connected,
            "base_url": base,
            "model": model,
            "fast_model": fast,
            "embed_model": embed,
            "model_available": model in models,
            "fast_available": fast in models,
            "embed_available": embed in models,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        })

    def _handle_gpu_status(self) -> None:
        import subprocess
        info: dict[str, Any] = {"backend": "cpu", "apple_silicon": False, "details": ""}
        in_docker = platform.system() != "Darwin"
        if not in_docker:
            info["backend"] = "metal"
            info["apple_silicon"] = True
            info["container"] = False
            try:
                r = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5,
                )
                info["details"] = r.stdout.strip()
            except Exception:
                info["details"] = "Apple Silicon"
            try:
                r2 = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                info["memory_gb"] = round(int(r2.stdout.strip()) / (1024**3), 1)
            except Exception:
                pass
        else:
            info["container"] = True
            info["backend"] = "ollama"
            info["apple_silicon"] = True
            info["details"] = "Apple Silicon (Metal via Ollama)"
            try:
                r = requests.get(
                    f"{self.org.config.ollama_base_url}/api/tags", timeout=5
                )
                if r.status_code == 200:
                    info["ollama_connected"] = True
            except Exception:
                info["ollama_connected"] = False
        _json_response(self, info)

    def _serve_debug_graph(self) -> None:
        data = self.org.graph_data()
        nodes_json = json.dumps(data)
        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
body{{margin:0;background:#0a0e17;overflow:hidden;font-family:sans-serif}}
#graph{{width:100vw;height:100vh}}
#info{{position:fixed;top:10px;left:10px;color:#94a3b8;font-size:12px;z-index:10;background:rgba(0,0,0,.7);padding:8px 12px;border-radius:6px}}
</style></head>
<body>
<div id="info">Loading graph...</div>
<div id="graph"></div>
<script>
const data = {nodes_json};
const info = document.getElementById('info');
info.textContent = 'Nodes: '+data.nodes.length+', Edges: '+data.links.length;
const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select('#graph').append('svg').attr('width',W).attr('height',H).attr('viewBox',[0,0,W,H]);
const links = data.links.map(d=>({{...d}}));
const nodes = data.nodes.map(d=>({{...d}}));
try {{
const sim = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d=>d.id).distance(130).strength(.3))
  .force('charge', d3.forceManyBody().strength(-250))
  .force('center', d3.forceCenter(W/2, H/2))
  .force('collision', d3.forceCollide(25));
const link = svg.append('g').selectAll('line').data(links).join('line')
  .attr('stroke','#1e3a5f').attr('stroke-width',1.2).attr('stroke-opacity',.6);
const node = svg.append('g').selectAll('g').data(nodes).join('g')
  .call(d3.drag().on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y}})
    .on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y}})
    .on('end',(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}}));
node.filter(d=>d.type==='paper').append('rect')
  .attr('width',14).attr('height',14).attr('x',-7).attr('y',-7).attr('rx',4)
  .attr('fill','#60a5fa').attr('stroke','#60a5fa').attr('stroke-width',2);
node.filter(d=>d.type!=='paper').append('circle')
  .attr('r',7).attr('fill','#c084fc').attr('stroke','#c084fc').attr('stroke-width',2);
node.append('text')
  .text(d=>(d.label||'').substring(0,22))
  .attr('dx',15).attr('dy',4).attr('fill','#94a3b8').attr('font-size','10px').attr('font-family','sans-serif');
sim.on('tick',()=>{{
  link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  node.attr('transform',d=>'translate('+d.x+','+d.y+')');
}});
info.textContent += ' | OK';
}} catch(e) {{ info.textContent += ' | ERROR: '+e.message; }}
</script></body></html>"""
        _html_response(self, html)

    def do_POST(self) -> None:
        path, params = self._parse_path()
        body = self._read_body()
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if path == "/api/add":
            arxiv_id = data.get("id", params.get("id", ""))
            if not arxiv_id:
                _json_response(self, {"error": "missing id"}, 400)
                return
            result = self.org.add_by_id(arxiv_id)
            _json_response(self, result)
        elif path == "/api/search":
            query = data.get("query", params.get("query", ""))
            if not query:
                _json_response(self, {"error": "missing query"}, 400)
                return
            results = self.org.search(query)
            _json_response(self, results)
        elif path == "/api/import":
            query = data.get("query", params.get("query", ""))
            if not query:
                _json_response(self, {"error": "missing query"}, 400)
                return
            results = self.org.add_by_search(query)
            _json_response(self, results)
        elif path == "/api/query":
            question = data.get("question", params.get("question", ""))
            if not question:
                _json_response(self, {"error": "missing question"}, 400)
                return
            result = self.org.query_rag(question)
            _json_response(self, result)
        elif path == "/api/lineage":
            arxiv_id = data.get("arxiv_id", params.get("arxiv_id", ""))
            if not arxiv_id:
                _json_response(self, {"error": "missing arxiv_id"}, 400)
                return
            result = self.org.fetch_lineage(arxiv_id)
            _json_response(self, result)
        elif path == "/api/web/add":
            url = data.get("url", params.get("url", ""))
            if not url:
                _json_response(self, {"error": "missing url"}, 400)
                return
            result = self.org.web.ingest(url)
            _json_response(self, result)
        elif path == "/api/similarity":
            paper_ids = data.get("paper_ids", params.get("paper_ids", None))
            algorithm = data.get("algorithm", params.get("algorithm", "combined"))
            if isinstance(paper_ids, str):
                paper_ids = [x.strip() for x in paper_ids.split(",") if x.strip()]
            _json_response(self, self.org.similarity(paper_ids=paper_ids, algorithm=algorithm))
        elif path == "/api/vectors/compute":
            _json_response(self, self.org.compute_vectors())
        elif path == "/api/vectors/status":
            _json_response(self, self.org.vectors.status())
        elif path == "/api/vectors/project":
            _json_response(self, self.org.project_vectors())
        elif path == "/api/vectors/clear":
            self.org.vectors.clear()
            _json_response(self, {"status": "ok"})
        elif path == "/api/refresh":
            model = data.get("model", params.get("model", None))
            result = self.org.refresh_papers(model=model)
            _json_response(self, result)
        elif path == "/api/papers/refresh":
            paper_id = data.get("paper_id", params.get("paper_id", ""))
            if not paper_id:
                _json_response(self, {"error": "missing paper_id"}, 400)
                return
            model = data.get("model", params.get("model", None))
            result = self.org.refresh_paper(paper_id, model=model)
            _json_response(self, result)
        elif path == "/api/definitions":
            result = self.org.generate_definitions()
            _json_response(self, result)
        elif path == "/api/pool/topics/add":
            name = data.get("name", "")
            query = data.get("query", "")
            if not name or not query:
                _json_response(self, {"error": "missing name or query"}, 400)
                return
            self.org.pool.add_topic(name, query)
            _json_response(self, {"status": "ok"})
        elif path == "/api/pool/topics/remove":
            name = data.get("name", "")
            if not name:
                _json_response(self, {"error": "missing name"}, 400)
                return
            self.org.pool.remove_topic(name)
            _json_response(self, {"status": "ok"})
        elif path == "/api/pool/import":
            arxiv_id = data.get("arxiv_id", "")
            if not arxiv_id:
                _json_response(self, {"error": "missing arxiv_id"}, 400)
                return
            result = self.org.add_by_id(arxiv_id)
            if result.get("status") == "added" or result.get("status") == "exists":
                self.org.pool.mark_imported(arxiv_id)
            _json_response(self, result)
        elif path == "/api/pool/import_batch":
            arxiv_ids = data.get("arxiv_ids", [])
            if not arxiv_ids:
                _json_response(self, {"error": "missing arxiv_ids"}, 400)
                return
            results = []
            for aid in arxiv_ids:
                r = self.org.add_by_id(aid)
                if r.get("status") in ("added", "exists"):
                    self.org.pool.mark_imported(aid)
                results.append({"arxiv_id": aid, "status": r.get("status")})
            _json_response(self, {"results": results})
        else:
            _json_response(self, {"error": "not found"}, 404)

    def _serve_dashboard(self) -> None:
        if HTML.exists():
            _html_response(self, HTML.read_text())
        else:
            _html_response(self, _inline_dashboard())


def run_server(
    org: Organizer,
    host: str = "127.0.0.1",
    port: int = 7777,
) -> None:
    get_capture()
    RouteHandler.org = org
    server = HTTPServer((host, port), RouteHandler)
    logger.info("Server listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.server_close()


def _inline_dashboard() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Hive Research</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0a0e17;color:#e2e8f0;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;text-align:center;padding:20px}
.card{background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:40px;max-width:500px}
h1{color:#60a5fa;font-size:24px;margin:0 0 8px}p{color:#94a3b8;line-height:1.6;font-size:14px}
code{background:#1e293b;padding:2px 6px;border-radius:4px;font-size:13px;color:#c084fc}
</style></head><body>
<div class="card"><h1>Hive Research</h1>
<p>Dashboard file not found. Run with <code>dashboard.html</code> present or use the <code>--inline</code> flag.</p></div></body></html>"""
