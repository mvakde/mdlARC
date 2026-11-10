"""Stage only the current task and the six intended demonstrations."""
from pathlib import Path
import json,sys,shutil
from task_variations.common.storage import sha,read,atomic_write_json,atomic_write_text
from task_variations.generate.json_storage import evidence_bytes
from task_variations.common.data import load_puzzle,load_summary
from task_variations.improve.data import _load_generator
from .inspect_puzzles import format_puzzle

PROMPT='''Write a grounded variation review for the target ARC generator. Read instructions.md
and examples/README.md, study all six worked examples, and explore target/ using files,
compact digit grids, code execution, and useful diagnostics. Read task.txt and the
sample .txt files for grids; use JSON for Python analysis. Do not render or view images.
Write only review.md as the deliverable, with brief seed/pair references supporting
findings. Do not write evidence.md, a demonstration recap, or a separate audit narrative.
Investigate until your review is useful and supported; there is no fixed exploration
sequence or final-answer schema. The staged originals are evidence; use copies for
experiments. Work only inside this workspace. Summarize completion in your final message.
'''
def hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '.git' not in p.parts and 'scratch' not in p.parts}
def copy(source,dest,compact_json=False):
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(evidence_bytes(source) if compact_json else source.read_bytes())
def stage_examples(config,destination):
    pinned=read(config.example_root/'manifest.json')
    for name,h in pinned['files'].items():
        path=config.example_root/name
        path.resolve().relative_to(config.example_root.resolve())
        if path.is_symlink() or sha(path)!=h:
            raise ValueError('Pinned review example changed: '+name)
    for name in config.example_ids:
        source=config.example_root/name;folder=destination/name
        for n in ('task.json','summary.xml','generator_before.py','manual_review.md','generator_after.py'):
            copy(source/n,folder/n,True)
        atomic_write_text(folder/'task.txt',format_puzzle(read(folder/'task.json'),label=name))
        for group in ('samples_before','samples_after'):
            (folder/group).mkdir(parents=True,exist_ok=True)
            for p in sorted((source/group).glob('seed-*.json')):
                d=folder/group/p.name;copy(p,d,True)
                atomic_write_text(d.with_suffix('.txt'),format_puzzle(read(d),label=f'{name}/{group}/{p.name}'))
    atomic_write_text(destination/'README.md','Read each example’s task, summary, first generator, human review and improved generator. Compare the before and after samples. These are demonstrations of review style.\n')
    atomic_write_json(destination/'manifest.json',{'example_ids':list(config.example_ids),'files':hashes(destination)})
def stage_workspace(config,name,split,workspace):
    if config.dataset_name=='arc2' and split=='evaluation' and name in config.example_ids:
        raise ValueError('Pinned human review examples cannot be automatically reviewed')
    task=load_puzzle(config,name,split);summary=load_summary(config,task)
    generator=_load_generator(config,task,summary)
    target=workspace/'target'
    for source,n in [(task.path,'task.json'),(summary.path,'summary.xml'),(generator.generator_path,'generator.py'),
                     (generator.manifest_path,'samples/manifest')]: copy(source,target/n,True)
    # Metadata shown to the model describes this task only and uses local paths.
    atomic_write_json(target/'generator_metadata.json',generator.metadata)
    atomic_write_json(target/'generator_validation.json',generator.validation)
    for seed in generator.sampling_context['completed_seeds']:
        n='seed-%010d.json'%seed;copy(generator.manifest_path.parent/n,target/'samples'/n,True)
        atomic_write_text((target/'samples'/n).with_suffix('.txt'),format_puzzle(read(target/'samples'/n),label=f'target/samples/{n}'))
    atomic_write_text(target/'task.txt',format_puzzle(task.data,label=name))
    atomic_write_json(target/'provenance.json',{'puzzle_id':name,'source_split':split,'sampling':generator.sampling_context})
    stage_examples(config,workspace/'examples');copy(config.instructions,workspace/'instructions.md')
    copy(Path(__file__).with_name('inspect_puzzles.py'),workspace/'tools/inspect_puzzles.py')
    atomic_write_json(workspace/'environment.json',{'python_executable':sys.executable,'examples':list(config.example_ids)})
    atomic_write_text(workspace/'AGENTS.md','Read instructions.md. Explore only the staged target, examples and tools. Do not inspect parent or sibling directories. Write review.md with supporting references; preserve all staged evidence. Use scratch/ for experiments.\n'
        'All JSON and JSONL you create, including generated code, must be compact with no indentation. '
        'Use json.dumps(value, separators=(",", ":")) in Python and JSON.stringify(value) in JavaScript.\n'
        'Bound disk use: estimate bytes, file counts and peak temporary space across tasks, seeds, pairs, steps, workers, retries and retained runs. '
        'Keep scratch diagnostics and logs bounded; avoid duplicate reconstructible payloads. Preserve required evidence and hashes. '
        'Use the existing lossless trace codec for saved execution traces, keeping initial and exceptional before states. Generated code follows these rules too. '
        'Before storage optimization edits, report measured projected net savings, runtime and peak disk cost, including journals, indexes and backups.\n')
    return generator.manifest_path.parent
def validate_outputs(workspace,baseline):
    errors=[]
    for n,h in baseline.items():
        p=workspace/n
        if p.is_symlink() or not p.is_file() or sha(p)!=h: errors.append('Staged evidence changed: '+n)
    p=workspace/'review.md'
    if p.is_symlink() or not p.is_file() or not p.read_text().strip(): errors.append('Missing or empty review.md')
    return {'valid':not errors,'errors':errors,'semantic_verification':'not_automatically_verified'}
