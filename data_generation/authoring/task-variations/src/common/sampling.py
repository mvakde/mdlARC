"""Bounded subprocess sampling. Samples are published beside their task."""
from pathlib import Path
from types import SimpleNamespace
import json,os,subprocess,sys,tempfile,shutil
from .storage import read,sha,sha_bytes,atomic_write_json,publish,preserve,relocate_audit,now

def sample(config,task,version=2,start_seed=0,count=42,num_train=None,num_test=None,output=None):
    if count<1 or start_seed<0: raise ValueError('count must be positive and seed nonnegative')
    generator=task/('generator_v2.py' if version==2 else 'generator_v1.py')
    if not generator.is_file(): raise ValueError('Missing generator: '+str(generator))
    output=Path(output or task/('samples' if version==2 else 'review-samples'))
    with tempfile.TemporaryDirectory(prefix='arc-sampling-') as temp:
        root=Path(temp);request=root/'request.json';result=root/'result.json'
        atomic_write_json(request,{'generator':str(generator),'task':str(task),'output':str(root/'batch'),
            'start_seed':start_seed,'count':count,'num_train':num_train,'num_test':num_test,'result':str(result)})
        env={**os.environ,'PYTHONPATH':os.pathsep.join(sys.path)}
        process=subprocess.run([sys.executable,'-m','task_variations.common.sample_worker',str(request)],env=env,
            capture_output=True,text=True,timeout=max(60,config.sample_timeout_seconds*count))
        if process.returncode: raise ValueError(process.stderr[-3000:] or process.stdout[-3000:])
        response=read(result)
        if not response.get('ok'): raise ValueError(response.get('error','Sampling failed'))
        batch=root/'batch'
        existing_manifest=output/'manifest'
        if existing_manifest.exists():
            old=read(existing_manifest)
            expected=sha_bytes(generator.read_text().rstrip().encode())
            settings=old.get('sampling',{})
            if old.get('generator',{}).get('sha256')!=expected or settings.get('num_train')!=num_train or settings.get('num_test')!=num_test:
                archived=task/'saved-outputs/sample-batches'/now().replace(':','-')
                archived.parent.mkdir(parents=True,exist_ok=True);output.rename(archived)
                relocate_audit(output,archived)
        output.mkdir(parents=True,exist_ok=True)
        for p in batch.glob('seed-*.json'):
            existing=output/p.name
            publish(existing,p.read_bytes())
        manifest=read(batch/'manifest');manifest['generator']={'path':str(generator.relative_to(task)),
            'sha256':sha_bytes(generator.read_text().rstrip().encode()),'puzzle_id':task.name,'source_split':task.parent.name}
        manifest['task_paths']=[p.name for p in sorted(output.glob('seed-*.json'))]
        manifest['file_sha256']={p.name:sha(p) for p in sorted(output.glob('seed-*.json'))}
        manifest['completed_seeds']=[int(n[5:-5]) for n in manifest['file_sha256']]
        preserve(output/'manifest')
        atomic_write_json(output/'manifest',manifest)
    return output/'manifest'
def ensure_review_samples(config,task):
    path=task/'review-samples/manifest'
    if path.is_file():
        m=read(path)
        if m.get('generator',{}).get('sha256')==sha_bytes((task/'generator_v1.py').read_text().rstrip().encode()):
            return path
    return sample(config,task,version=1,count=config.sample_count)
