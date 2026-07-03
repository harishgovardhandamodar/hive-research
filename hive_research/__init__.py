import sys
from pathlib import Path

try:
    from hive_datatype import Node, Edge, HiveGraph, NodeType
except ImportError:
    _p = Path(__file__).resolve().parent.parent.parent / "hive-datatype"
    if _p.exists():
        sys.path.insert(0, str(_p))
