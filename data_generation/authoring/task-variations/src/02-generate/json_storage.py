"""Compact model evidence while preserving immutable source task bytes."""
from pathlib import Path
import json
def evidence_bytes(path):
    path=Path(path);raw=path.read_bytes()
    if path.name=='task.json' or path.suffix!='.json' and path.name!='manifest': return raw
    try: value=json.loads(raw)
    except ValueError: return raw
    return (json.dumps(value,separators=(",", ":"),ensure_ascii=False,sort_keys=True)+'\n').encode()
def bound_bytes(path,expected):
    from task_variations.common.storage import sha_bytes
    raw=Path(path).read_bytes()
    return raw if sha_bytes(raw)==expected else evidence_bytes(path)
