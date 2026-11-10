"""Manual reviews use the same task folders as automatic reviews."""
from pathlib import Path
from dataclasses import dataclass
from http import HTTPStatus
import json,shutil,threading
from task_variations.common.config import load_config,PIPELINE
from task_variations.common.storage import *
from task_variations.common.data import all_task_refs
from task_variations.common.sampling import ensure_review_samples

class ReviewError(Exception):
    def __init__(self,message,status=HTTPStatus.BAD_REQUEST): super().__init__(message);self.status=status
@dataclass(frozen=True)
class TaskEntry:
    split:str
    puzzle_id:str
    seeds:tuple
class ReviewStore:
    def __init__(self,config_path=None,static_root=None,results_root=None):
        self.config=load_config(config_path,results_root=results_root)
        self.generated_root=self.notes_root=self.config.task_root
        self.static_root=Path(static_root or PIPELINE/'tools/manual-review/static').resolve()
        self._lock=threading.RLock()
    def _task(self,split,name):
        try:
            d=task_dir(self.config,split,name)
            if not (d/'task.json').is_file(): raise ValueError('Unknown task')
            return d
        except ValueError as exc: raise ReviewError(str(exc),HTTPStatus.NOT_FOUND) from exc
    def _samples(self,split,name):
        d=self._task(split,name)
        frozen=d/'llm-inputs/03-review-first-generation/reviewed-samples'
        if frozen.is_dir() and current(d,'review-first-generation'): return frozen
        return d/'review-samples'
    def discover(self,force=False):
        entries=[]
        for ref in all_task_refs(self.config):
            samples=self._samples(ref.source_split,ref.puzzle_id)
            seeds=tuple(int(p.stem[5:]) for p in sorted(samples.glob('seed-*.json')))
            task=self._task(ref.source_split,ref.puzzle_id)
            if current(task,'generate-from-summary'):
                entries.append(TaskEntry(ref.source_split,ref.puzzle_id,seeds))
        return tuple(entries)
    def load_note(self,split,name):
        d=self._task(split,name);p=d/'manual-review.json'
        if p.exists(): return {k:read(p).get(k) for k in ('notes','reviewed','updated_at')}
        p=d/'review.md'
        return {'notes':p.read_text() if p.exists() else '', 'reviewed':current(d,'review-first-generation'),'updated_at':None}
    def catalog_payload(self,selected_split='all',force=False):
        if selected_split not in ('all','training','evaluation'): raise ReviewError('Invalid split')
        tasks=[];counts=dict(all=0,training=0,evaluation=0,reviewed=0,noted=0)
        for e in self.discover(force):
            note=self.load_note(e.split,e.puzzle_id);counts['all']+=1;counts[e.split]+=1
            counts['reviewed']+=int(bool(note['reviewed']));counts['noted']+=int(bool(note['notes'].strip()))
            if selected_split in ('all',e.split): tasks.append({'id':e.puzzle_id,'split':e.split,'seed_count':len(e.seeds),
                'reviewed':note['reviewed'],'has_notes':bool(note['notes'].strip()),'note_preview':' '.join(note['notes'].split())[:100]})
        return {'tasks':tasks,'counts':counts}
    def task_overview(self,split,name):
        d=self._task(split,name);samples=self._samples(split,name)
        if samples==d/'review-samples':
            with task_lock(d): samples=ensure_review_samples(self.config,d).parent
        m=read(samples/'manifest') if (samples/'manifest').exists() else {}
        return {'id':name,'split':split,'seeds':[int(p.stem[5:]) for p in sorted(samples.glob('seed-*.json'))],
                'original':read(d/'task.json'),'notes':self.load_note(split,name),
                'generator':{**m.get('generator',{}),'path':str(d/'generator_v1.py'),
                             'puzzle_id':name,'source_split':split}}
    def seed_task(self,split,name,seed):
        if type(seed) is not int or seed<0: raise ReviewError('Invalid seed')
        d=self._task(split,name);samples=self._samples(split,name)
        if samples==d/'review-samples':
            with task_lock(d):samples=ensure_review_samples(self.config,d).parent
        p=samples/('seed-%010d.json'%seed)
        if not p.is_file(): raise ReviewError('Sample not found',HTTPStatus.NOT_FOUND)
        return {'id':name,'split':split,'seed':seed,'task':read(p)}
    def save_note(self,split,name,payload):
        if not isinstance(payload,dict) or not isinstance(payload.get('notes'),str) or type(payload.get('reviewed')) is not bool:
            raise ReviewError('Expected notes text and reviewed boolean')
        if len(payload['notes'])>100000: raise ReviewError('Review is too long')
        d=self._task(split,name)
        with self._lock,task_lock(d):
            saved={**payload,'updated_at':now()}
            publish(d/'manual-review.json',(json.dumps(saved,sort_keys=True,separators=(",", ":"))+'\n').encode())
            if saved['reviewed'] and saved['notes'].strip():
                source=self._samples(split,name)
                if source==d/'review-samples': source=ensure_review_samples(self.config,d).parent
                inputs=d/'llm-inputs/03-review-first-generation'
                if inputs.exists():
                    archived=d/'saved-outputs/llm-inputs'/(now().replace(':','-')+'-03-review-first-generation')
                    archived.parent.mkdir(parents=True,exist_ok=True)
                    source_relative=source.relative_to(inputs) if source.is_relative_to(inputs) else None
                    inputs.rename(archived)
                    relocate_audit(inputs,archived)
                    if source_relative is not None: source=archived/source_relative
                inputs.mkdir(parents=True)
                for name in dependencies('review-first-generation'):
                    shutil.copy2(d/name,inputs/name)
                shutil.copytree(source,inputs/'reviewed-samples')
                atomic_write_json(inputs/'metadata.json',{'source':'human-review','completed_at':saved['updated_at']})
                atomic_write_text(inputs/'saved-output/review.md',saved['notes'].strip()+'\n')
                atomic_write_json(inputs/'manifest.json',{'source':'exact','files':{
                    str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()}})
                publish(d/'review.md',(saved['notes'].strip()+'\n').encode())
                prov=read(d/'provenance.json');prov['stages']['review-first-generation']={
                    'author':'human','model':None,'effort':None,'dependencies':dependency_hashes(d,'review-first-generation'),
                    'output_sha256':sha(d/'review.md'),'completed_at':saved['updated_at'],
                    'inputs':str(inputs.relative_to(d))}
                prov['files']['review.md']=sha(d/'review.md')
                prov['files'].pop('manual-review.json',None)
                atomic_write_json(d/'provenance.json',prov)
                update_index(self.config.results_root)
        return {k:saved[k] for k in ('notes','reviewed','updated_at')}
    def export_notes(self):
        return {'schema_version':2,'exported_at':now(),'reviews':[{'puzzle_id':e.puzzle_id,'source_split':e.split,
            **self.load_note(e.split,e.puzzle_id)} for e in self.discover()]}
