"""Load selected task outputs directly, without historical path lookups."""
from dataclasses import dataclass
from pathlib import Path
import json
from .storage import task_dir,sha,read,current

@dataclass(frozen=True,order=True)
class TaskRef:
    source_split:str
    puzzle_id:str
@dataclass(frozen=True)
class PuzzleSource:
    puzzle_id:str
    source_split:str
    path:Path
    sha256:str
    data:dict
@dataclass(frozen=True)
class SummarySource:
    puzzle_id:str
    source_split:str
    path:Path
    sha256:str
    text:str
    kind:str='selected'
    upstream_run_dir:Path|None=None
    upstream_prompt_sha256:str|None=None
    upstream_prompt_version:str|None=None
    upstream_model:str|None=None
    upstream_reasoning_effort:str|None=None

class SummaryUnavailableError(ValueError): pass
def resolve_source_split(config,name,requested_split='auto'):
    if requested_split not in ('auto','training','evaluation'): raise ValueError('Invalid split')
    splits=('training','evaluation') if requested_split=='auto' else (requested_split,)
    found=[s for s in splits if (task_dir(config,s,name)/'task.json').is_file()]
    if len(found)!=1: raise ValueError('Task is missing or its split is ambiguous: '+name)
    return found[0]
def load_puzzle(config,name,source_split='auto'):
    split=resolve_source_split(config,name,source_split);p=task_dir(config,split,name)/'task.json'
    from task_variations.generate.visualization import _validate_arc_json
    return PuzzleSource(name,split,p,sha(p),_validate_arc_json(read(p)))
def all_task_refs(config,source_split='all'):
    splits=('training','evaluation') if source_split in ('all','auto') else (source_split,)
    if any(s not in ('training','evaluation') for s in splits): raise ValueError('Invalid split')
    return [TaskRef(s,p.parent.name) for s in splits for p in sorted((config.task_root/s).glob('*/task.json'))]
def load_summary(config,puzzle):
    p=puzzle.path.parent/'summary.xml'
    if not p.is_file(): raise SummaryUnavailableError('Missing summary: '+puzzle.puzzle_id)
    if not current(p.parent,'summarise-task'): raise SummaryUnavailableError('Summary needs updating: '+puzzle.puzzle_id)
    from task_variations.summary.validation import validate_summary_xml
    text=p.read_text();v=validate_summary_xml(text)
    if not v.valid: raise SummaryUnavailableError('; '.join(v.errors))
    record=read(p.parent/'provenance.json')['stages']['summarise-task']
    return SummarySource(puzzle.puzzle_id,puzzle.source_split,p,sha(p),text,
                         upstream_model=record.get('model'),upstream_reasoning_effort=record.get('effort'))
