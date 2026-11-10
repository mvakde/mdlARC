"""Atomic task updates and content hashes. Existing outputs are always preserved."""
from pathlib import Path
import json, hashlib, os, tempfile, fcntl
from contextlib import contextmanager
from datetime import datetime, timezone

def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()
def sha(path):
    from .preview_storage import snapshot_bytes
    return sha_bytes(snapshot_bytes(path))
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def atomic_write_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def atomic_write_json(path,value):
    atomic_write_text(path,json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n')
def preserve(path):
    path=Path(path)
    if path.is_file():
        base=path.parent.parent if path.parent.name in ('samples','review-samples') else path.parent
        dest=base/'saved-outputs'/(sha(path)[:16]+'-'+path.name)
        if dest.exists() and dest.read_bytes()!=path.read_bytes():
            raise ValueError('Existing preserved output changed: '+str(dest))
        if not dest.exists():
            dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes())
        relocate_audit(path,dest)
        return dest

def relocate_audit(source,destination):
    """Keep the original-output audit valid when selected files are replaced."""
    source=Path(source);destination=Path(destination)
    for root in source.parents:
        audit=root/'checks/preservation.json'
        if not audit.is_file():continue
        old=str(source.relative_to(root));new=str(destination.relative_to(root))
        with (root/'.preservation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            value=read(audit);changed=False
            for entry in value['files']:
                if entry['path']==old or entry['path'].startswith(old+'/'):
                    entry['path']=new+entry['path'][len(old):];changed=True
            if changed:atomic_write_json(audit,value)
        break
def publish(path,raw):
    path=Path(path)
    if path.is_file() and path.read_bytes()!=raw: preserve(path)
    atomic_write_text(path,raw.decode())
def task_dir(config,split,name):
    if split not in ('training','evaluation') or not name or Path(name).name!=name or name in ('.','..'):
        raise ValueError('Invalid task name or split')
    path=config.task_root/split/name
    path.resolve().relative_to(config.task_root.resolve())
    return path
@contextmanager
def task_lock(task):
    task.mkdir(parents=True,exist_ok=True)
    with (task/'.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

OUTPUTS={'summarise-task':'summary.xml','generate-from-summary':'generator_v1.py','review-first-generation':'review.md','generate-from-reviews':'generator_v2.py'}
def dependencies(stage):
    return ['task.json']+({'summarise-task':[], 'generate-from-summary':['summary.xml'],
        'review-first-generation':['summary.xml','generator_v1.py'],
        'generate-from-reviews':['summary.xml','generator_v1.py','review.md']}[stage])
def dependency_hashes(task,stage): return {n:sha(task/n) for n in dependencies(stage)}
def current(task,stage,model=None,effort=None):
    if not (task/OUTPUTS[stage]).is_file(): return False
    try:
        record=read(task/'provenance.json').get('stages',{}).get(stage,{})
        if record.get('dependencies')!=dependency_hashes(task,stage): return False
        if record.get('output_sha256')!=sha(task/OUTPUTS[stage]): return False
        return (model is None or record.get('model')==model) and (effort is None or record.get('effort')==effort)
    except (OSError,ValueError): return False
def update_index(results):
    with (results/'.index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        tasks=[]
        for dataset in ('arc2','arc1-filtered','conceptarc'):
            for original in sorted((results/dataset).glob('*/*/task.json')):
                d=original.parent; stages={s:current(d,s) for s in OUTPUTS}
                stage=('complete' if all(stages.values()) else next((s for s in OUTPUTS if not stages[s]),'complete'))
                tasks.append({'dataset':dataset,'source_split':d.parent.name,'puzzle_id':d.name,'original_id':d.name,
                              'path':str(d.relative_to(results)),'stage':stage,'stages':stages})
        counts={ds:sum(t['dataset']==ds for t in tasks) for ds in ('arc2','arc1-filtered','conceptarc')}
        atomic_write_json(results/'index.json',{'schema_version':2,'counts':counts,'tasks':tasks})

def verify(results):
    index=read(results/'index.json');seen=set(); files=0
    for t in index['tasks']:
        d=results/t['path']; d.resolve().relative_to(results.resolve())
        if t['path'] in seen: raise ValueError('Duplicate task '+t['path'])
        seen.add(t['path'])
        prov=read(d/'provenance.json')
        for n,h in prov['files'].items():
            if sha(d/n)!=h: raise ValueError('Output changed: '+str(d/n))
            files+=1
        for manifest in (d/'llm-inputs').glob('*/manifest.json'):
            for n,h in read(manifest).get('files',{}).items():
                p=manifest.parent/n;p.resolve().relative_to(manifest.parent.resolve())
                if p.is_symlink() or sha(p)!=h: raise ValueError('Input snapshot changed: '+str(p))
        for s,ready in t.get('stages',{}).items():
            if ready!=current(d,s): raise ValueError('Stale task index: '+str(d))
    preservation=results/'checks/preservation.json'
    if preservation.exists():
        for item in read(preservation)['files']:
            if sha(results/item['path'])!=item['sha256']: raise ValueError('Preserved output missing: '+item['path'])
    return {'passed':True,'tasks':len(seen),'selected_files':files}
