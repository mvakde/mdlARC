from dataclasses import dataclass
from task_variations.common.data import PuzzleSource,TaskRef,load_puzzle,all_task_refs
from task_variations.common.storage import sha,read
from .validation import validate_summary_xml

@dataclass(frozen=True)
class Demonstration:
    puzzle:PuzzleSource
    summary_path:object
    summary_sha256:str
    summary_text:str
def load_demonstrations(config):
    result=[]
    for spec in config.fixed_demonstrations:
        root=config.summary_examples_root/spec.puzzle_id;task=root/'task.json';summary=root/'summary.xml'
        if sha(task)!=spec.task_sha256 or sha(summary)!=spec.summary_sha256: raise ValueError('Demonstration changed: '+spec.puzzle_id)
        text=summary.read_text()
        if not validate_summary_xml(text).valid: raise ValueError('Invalid demonstration summary')
        puzzle=PuzzleSource(spec.puzzle_id,spec.source_split,task,sha(task),read(task))
        result.append(Demonstration(puzzle,summary,sha(summary),text))
    return tuple(result)
def ensure_not_demo_target(config,name,split):
    if config.dataset_name=='arc2' and (split,name) in {(d.source_split,d.puzzle_id) for d in config.fixed_demonstrations}:
        raise ValueError('A fixed demonstration cannot be an inference target')
