from __future__ import annotations

import json
import logging
import mimetypes
import re
from functools import wraps
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable

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
                    params[k] = v
        return path, params

    def do_GET(self) -> None:
        path, params = self._parse_path()
        if path == "/" or path == "" or path == "/index.html":
            self._serve_dashboard()
        elif path == "/api/graph":
            _json_response(self, self.org.graph_data())
        elif path == "/api/stats":
            _json_response(self, self.org.stats())
        elif path == "/api/similarity":
            _json_response(self, self.org.similarity())
        elif path == "/api/papers":
            papers = [
                {"id": n.id, "title": n.label, "authors": n.authors, "published": n.published}
                for n in self.org.kg.papers
            ]
            _json_response(self, papers)
        elif path == "/api/concepts":
            concepts = [
                {"id": n.id, "label": n.label, "definition": n.definition}
                for n in self.org.kg.concepts
            ]
            _json_response(self, concepts)
        else:
            _json_response(self, {"error": "not found"}, 404)

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
    port: int = 8080,
) -> None:
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
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Hive Research</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f5;color:#333;padding:20px}
h1{font-size:1.5rem;margin-bottom:16px;color:#1a1a2e}
.card{background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
button{background:#1a1a2e;color:white;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-size:0.9rem}
button:hover{background:#16213e}
input{width:100%;padding:8px;border:1px solid #ddd;border-radius:4px;margin-bottom:8px;font-size:0.9rem}
#graph svg{width:100%;height:500px;border-radius:4px}
.stat{display:inline-block;margin:8px 16px 8px 0;font-size:0.9rem}
.stat span{font-weight:700;color:#1a1a2e}
#output{background:#1a1a2e;color:#e0e0e0;padding:12px;border-radius:4px;font-family:monospace;font-size:0.85rem;max-height:300px;overflow-y:auto;white-space:pre-wrap;margin-top:8px}
.tabs{display:flex;gap:8px;margin-bottom:12px}
.tab{padding:6px 14px;border-radius:4px;cursor:pointer;background:#eee;font-size:0.85rem}
.tab.active{background:#1a1a2e;color:white}
</style>
</head>
<body>
<h1> Hive Research</h1>
<div class="grid">
<div class="card">
  <h3>Knowledge Graph</h3>
  <div id="graph"><svg></svg></div>
  <div id="stats"></div>
</div>
<div class="card">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('add')">Add Paper</div>
    <div class="tab" onclick="switchTab('search')">Search</div>
    <div class="tab" onclick="switchTab('query')">RAG Query</div>
  </div>
  <div id="tab-add">
    <input id="arxiv-id" placeholder="arXiv ID (e.g. 1706.03762)">
    <button onclick="addPaper()">Add Paper</button>
  </div>
  <div id="tab-search" style="display:none">
    <input id="search-query" placeholder="Search query (e.g. attention mechanism)">
    <button onclick="searchArxiv()">Search</button>
    <button onclick="importSearch()" style="margin-left:8px">Import All</button>
  </div>
  <div id="tab-query" style="display:none">
    <input id="rag-question" placeholder="Ask a question about your papers">
    <button onclick="askRag()">Ask</button>
  </div>
  <div id="output">Ready</div>
</div>
</div>
<script>
let gData = {nodes:[],links:[]};
let simulation = null;

function log(msg){document.getElementById('output').textContent = msg}
function switchTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('[id^="tab-"]').forEach(d=>d.style.display='none');
  document.getElementById('tab-'+name).style.display='block';
  event.target.classList.add('active');
}

async function fetchJSON(url,method='GET',body=null){
  const opts = {method,headers:{'Content-Type':'application/json'}};
  if(body) opts.body = JSON.stringify(body);
  const r = await fetch(url,opts);
  return r.json();
}

async function loadGraph(){
  const data = await fetchJSON('/api/graph');
  gData = data;
  renderGraph(data);
  const stats = await fetchJSON('/api/stats');
  document.getElementById('stats').innerHTML =
    Object.entries(stats).filter(([k])=>k!=='rag').map(([k,v])=>`<div class="stat"><span>${k}:</span> ${v}</div>`).join('');
}
async function addPaper(){
  const id = document.getElementById('arxiv-id').value.trim();
  if(!id) return;
  log('Adding paper '+id+'...');
  const r = await fetchJSON('/api/add','POST',{id});
  log(JSON.stringify(r,null,2));
  loadGraph();
}
async function searchArxiv(){
  const q = document.getElementById('search-query').value.trim();
  if(!q) return;
  log('Searching...');
  const r = await fetchJSON('/api/search','POST',{query:q});
  log(r.map(p=>'['+p.arxiv_id+'] '+p.title).join('\\n'));
}
async function importSearch(){
  const q = document.getElementById('search-query').value.trim();
  if(!q) return;
  log('Importing...');
  const r = await fetchJSON('/api/import','POST',{query:q});
  log(JSON.stringify(r,null,2));
  loadGraph();
}
async function askRag(){
  const q = document.getElementById('rag-question').value.trim();
  if(!q) return;
  log('Thinking...');
  const r = await fetchJSON('/api/query','POST',{question:q});
  log('Answer: '+r.answer+'\\n\\nSources: '+(r.sources||[]).map(s=>s.title).join(', '));
}

function renderGraph(data){
  const svg = d3.select('#graph svg');
  svg.selectAll('*').remove();
  const width = svg.node().parentElement.clientWidth;
  const height = 500;
  svg.attr('viewBox',[0,0,width,height]);

  if(!data.nodes || data.nodes.length === 0){
    svg.append('text').attr('x',width/2).attr('y',height/2).attr('text-anchor','middle').attr('fill','#999').text('No papers yet');
    return;
  }

  const links = data.links.map(d=>({...d}));
  const nodes = data.nodes.map(d=>({...d}));

  const color = d3.scaleOrdinal(d3.schemeSet2);
  simulation = d3.forceSimulation(nodes)
    .force('link',d3.forceLink(links).id(d=>d.id).distance(100))
    .force('charge',d3.forceManyBody().strength(-200))
    .force('center',d3.forceCenter(width/2,height/2))
    .force('collision',d3.forceCollide(30));

  const link = svg.append('g')
    .selectAll('line').data(links).join('line')
    .attr('stroke','#ccc').attr('stroke-width',1.5).attr('stroke-opacity',0.6);

  const node = svg.append('g')
    .selectAll('circle').data(nodes).join('circle')
    .attr('r',8).attr('fill',d=>color(d.group||0))
    .attr('stroke','white').attr('stroke-width',1.5)
    .call(d3.drag()
      .on('start',(e,d)=>{if(!e.active)simulation.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y})
      .on('drag',(e,d)=>{d.fx=e.x;d.fy=e.y})
      .on('end',(e,d)=>{if(!e.active)simulation.alphaTarget(0);d.fx=null;d.fy=null}));

  const label = svg.append('g')
    .selectAll('text').data(nodes).join('text')
    .text(d=>d.label?.substring(0,20)).attr('font-size','10px')
    .attr('dx',12).attr('dy',4).attr('fill','#333');

  node.append('title').text(d=>d.label+(d.definition?'\\n'+d.definition:''));

  simulation.on('tick',()=>{
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
        .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    node.attr('cx',d=>d.x).attr('cy',d=>d.y);
    label.attr('x',d=>d.x).attr('y',d=>d.y);
  });
}
loadGraph();
</script>
</body>
</html>"""
