"""Exact selected generator and review evidence, all under the task folder."""
from dataclasses import dataclass
from pathlib import Path
from task_variations.common.data import load_puzzle,load_summary,all_task_refs,TaskRef
from task_variations.common.storage import sha,read,current

class TargetUnavailableError(ValueError):
    def __init__(self,status,message): super().__init__(message);self.status=status
@dataclass(frozen=True)
class ReviewSource:
    path:Path
    sha256:str
    notes_sha256:str
    notes:str
    reviewed:bool
    updated_at:str
    author:str='human'
    source_binding:dict|None=None
@dataclass(frozen=True)
class GeneratorSource:
    manifest_path:Path
    manifest_sha256:str
    run_dir:Path
    generator_path:Path
    code_text:str
    code_sha256:str
    response_path:Path
    response_sha256:str
    metadata_path:Path
    metadata:dict
    validation_path:Path
    validation:dict
    sampling_context:dict
@dataclass(frozen=True)
class RegenerationSource:
    target:object
    summary:object
    review:ReviewSource
    generator:GeneratorSource

def _load_generator(config,target,summary,manifest_path=None,allow_sampling_failure=False):
    task=target.path.parent;code=task/'generator_v1.py'
    if not current(task,'generate-from-summary'): raise TargetUnavailableError('stale_source_generator','First generator needs updating')
    from task_variations.common.sampling import ensure_review_samples
    folder=task/'review-samples'
    if manifest_path is None:
        frozen=task/'llm-inputs/03-review-first-generation/reviewed-samples/manifest'
        manifest_path=frozen if frozen.is_file() and current(task,'review-first-generation') else ensure_review_samples(config,task)
    manifest=read(manifest_path);seeds=manifest.get('completed_seeds',[])
    frozen_hashes=manifest_path.parent/'files.json'
    expected=read(frozen_hashes) if frozen_hashes.exists() else manifest.get('file_sha256',{})
    for filename,h in expected.items():
        path=manifest_path.parent/filename
        path.resolve().relative_to(manifest_path.parent.resolve())
        if path.is_symlink() or sha(path)!=h:
            raise TargetUnavailableError('stale_review','Source sample changed: '+filename)
    if not seeds and not allow_sampling_failure: raise TargetUnavailableError('missing_samples','No source samples')
    text=code.read_text();record=read(task/'provenance.json')['stages']['generate-from-summary']
    inputs=task/'llm-inputs/02-generate-from-summary'
    meta=inputs/('metadata.json' if (inputs/'metadata.json').exists() else 'recorded-inputs.json')
    validation=task/'validation_v1.json'
    if not validation.exists(): validation=inputs/'saved-output/validation.json'
    response=inputs/'saved-output/response.md'
    from task_variations.common.storage import sha_bytes
    normalized=sha_bytes(text.rstrip().encode())
    return GeneratorSource(manifest_path,sha(manifest_path),inputs,code,text,normalized,
        response,sha(response) if response.exists() else '',meta,record,validation,
        read(validation) if validation.exists() else {},
        {'completed_seeds':seeds,'sampling':manifest.get('sampling',{}),'state':manifest.get('state','complete'),
         'sample_count':len(seeds),'failures':manifest.get('failures',[])})

def load_regeneration_source(config,name,source_split='auto'):
    target=load_puzzle(config,name,source_split);summary=load_summary(config,target);task=target.path.parent
    if not current(task,'review-first-generation'): raise TargetUnavailableError('stale_review','Review needs updating')
    path=task/'review.md';record=read(task/'provenance.json')['stages']['review-first-generation']
    notes=path.read_text().strip()
    if not notes: raise TargetUnavailableError('empty_review','Empty review')
    from task_variations.common.storage import sha_bytes
    review=ReviewSource(path,sha(path),sha_bytes(notes.encode()),notes,True,record.get('completed_at',''),record.get('author','human'))
    generator=_load_generator(config,target,summary,allow_sampling_failure=True)
    sample_record=task/'llm-inputs/03-review-first-generation/reviewed-samples/manifest'
    if sample_record.exists():
        manifest=read(sample_record)
        expected=manifest.get('file_sha256',{})
        for filename,h in expected.items():
            p=sample_record.parent/filename
            p.resolve().relative_to(sample_record.parent.resolve())
            if sha(p)!=h: raise TargetUnavailableError('stale_review','Reviewed sample changed: '+filename)
    return RegenerationSource(target,summary,review,generator)
