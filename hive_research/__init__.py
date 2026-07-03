import sys
from pathlib import Path

_hive_datatype_path = Path(__file__).resolve().parent.parent.parent / "hive-datatype"
if _hive_datatype_path.exists():
    sys.path.insert(0, str(_hive_datatype_path))
