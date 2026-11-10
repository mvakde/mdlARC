"""Run generated code in a short-lived host process."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
from .storage import read,sha_bytes,atomic_write_json
from task_variations.generate.visualization import generate_samples,GeneratorRun

def main(path):
    r=read(path);generator=Path(r['generator']);task=Path(r['task']);output=Path(r['output'])
    try:
        g=GeneratorRun(task.name,task.parent.name,task,generator,sha_bytes(generator.read_text().rstrip().encode()),'')
        config=SimpleNamespace(artifact_root=output.parent/'cache')
        generate_samples(config,g,start_seed=r['start_seed'],count=r['count'],num_train=r['num_train'],num_test=r['num_test'],
            source_root=output.parent,dataset_name=output.name,tasks_direct=True,resume=False)
        atomic_write_json(Path(r['result']),{'ok':True})
    except Exception as exc:
        atomic_write_json(Path(r['result']),{'ok':False,'error':str(exc)})
        raise
if __name__=='__main__': main(Path(sys.argv[1]))
