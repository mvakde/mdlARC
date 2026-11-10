"""One dataset config; all storage paths are derived here."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json

PIPELINE = Path(__file__).resolve().parents[2]
REPO = PIPELINE.parents[1]
RESULTS = REPO / 'results/task-variations'
STAGES = ('summarise-task', 'generate-from-summary', 'review-first-generation', 'generate-from-reviews')
FOLDERS = dict(zip(STAGES, ('01-summarise-task', '02-generate-from-summary', '03-review-first-generation', '04-generate-from-reviews')))

@dataclass(frozen=True)
class DemoSpec:
    puzzle_id: str
    source_split: str
    task_sha256: str
    summary_sha256: str

def load_config(path=None, stage='generate-from-summary', results_root=None):
    path = Path(path or PIPELINE/'configs/arc2.json').resolve()
    raw = json.loads(path.read_text())
    if raw.get('schema_version') != 2 or raw.get('dataset') not in ('arc2','arc1-filtered','conceptarc'):
        raise ValueError('Expected a dataset config with schema_version 2')
    if stage not in STAGES:
        raise ValueError('Unknown stage: '+stage)
    root = Path(results_root or RESULTS).resolve()/raw['dataset']
    options = {**raw, **raw.get('stages', {}).get(stage, {})}
    demos = tuple(DemoSpec(x['id'],x['source_split'],x['task_sha256'],x['summary_sha256']) for x in raw['summary_demonstrations'])
    if len(demos)!=6 or len({d.puzzle_id for d in demos})!=6:
        raise ValueError('Exactly six distinct summary demonstrations are required')
    ids = tuple(raw['review_examples'])
    if len(ids)!=6 or len(set(ids))!=6:
        raise ValueError('Exactly six distinct review examples are required')
    if type(raw['repair_turns']) is not int or raw['repair_turns']<0:
        raise ValueError('repair_turns must be a nonnegative integer')
    if type(raw['sample_count']) is not int or raw['sample_count']<1:
        raise ValueError('sample_count must be positive')
    if type(raw['sample_timeout_seconds']) is not int or raw['sample_timeout_seconds']<1:
        raise ValueError('sample_timeout_seconds must be positive')
    config = SimpleNamespace(
        config_path=path,path=path,stage=stage,stage_folder=FOLDERS[stage],dataset_name=raw['dataset'],
        task_root=root,results_root=root.parent,default_model=options['model'],default_reasoning_effort=options['effort'],
        prompt_version=options.get('prompt_version','variation-review-v1'),target_dataset=raw['dataset_label'],
        fixed_demonstrations=demos,manual_summary_ids=tuple(d.puzzle_id for d in demos),
        summary_examples_root=PIPELINE/'examples/summary',example_ids=ids,example_root=PIPELINE/'examples/review',
        prompt_template_path=PIPELINE/'prompts'/('improve.txt' if stage=='generate-from-reviews' else 'generate.txt'),
        instructions=PIPELINE/'prompts/review.md',acceptance_root=PIPELINE/'acceptance' if raw['dataset']=='arc2' else None,
        repair_turns=raw['repair_turns'],sample_count=raw['sample_count'],sample_timeout_seconds=raw['sample_timeout_seconds'],
    )
    config.variation=config
    config.base_config=config
    return config
