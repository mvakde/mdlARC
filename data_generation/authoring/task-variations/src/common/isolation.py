"""A Git baseline plus a restricted Codex read/write profile for each workspace."""
from pathlib import Path
import os,json,subprocess,sys
from .codex_runtime import codex_binary

def overrides(workspace):
    # No root/home/repository read grant and no broadly readable temp directory.
    # NumPy/Python runtime access is allowed; source tasks and saved outputs are not.
    runtime={str(Path(sys.prefix).resolve()),str(Path(sys.base_prefix).resolve()),
             str(Path(sys.base_prefix)),str(Path(sys.executable).resolve().parent.parent)}
    if sys.platform=='darwin' and Path('/opt/homebrew').is_dir():
        runtime.add('/opt/homebrew')
    fs={':minimal':'read',str(workspace.resolve()):'write'}
    fs.update({p:'read' for p in runtime if not workspace.resolve().is_relative_to(p)})
    fs[str(workspace/'.git')]='read'
    return (
        'default_permissions="task-only"',
        'permissions.task-only.filesystem='+toml_value(fs),
        'permissions.task-only.network.enabled=false',
        'approval_policy="never"',
        'shell_environment_policy.inherit="none"',
        'shell_environment_policy.set='+toml_value({'PATH':os.environ.get('PATH','/usr/bin:/bin'),
            'TMPDIR':str(workspace/'scratch/tmp'),'PYTHONDONTWRITEBYTECODE':'1'}),
        'web_search="disabled"',
    )
def toml_value(value):
    if isinstance(value,dict): return '{'+', '.join(json.dumps(k)+' = '+toml_value(v) for k,v in value.items())+'}'
    return json.dumps(value)
def git_baseline(workspace):
    (workspace/'scratch/tmp').mkdir(parents=True,exist_ok=True)
    (workspace/'.gitignore').write_text('__pycache__/\n*.py[cod]\nscratch/\n')
    for args in [('init','--quiet'),('add','.'),('-c','user.name=ARC Pipeline','-c','user.email=arc@local',
                '-c','commit.gpgsign=false','commit','--quiet','-m','Stage task inputs')]:
        subprocess.run(['git','-C',str(workspace),*args],check=True,capture_output=True,text=True)
    return subprocess.check_output(['git','-C',str(workspace),'rev-parse','HEAD'],text=True).strip()
def probe(workspace,forbidden):
    binary=codex_binary()
    if not binary: raise RuntimeError('Codex CLI is required')
    (workspace/'scratch/tmp').mkdir(parents=True,exist_ok=True)
    config=[]
    for value in overrides(workspace): config+=['-c',value]
    script='''import pathlib,sys
import numpy as np
assert np.arange(3).tolist()==[0,1,2]
assert pathlib.Path("probe.txt").read_text()=="allowed"
try:
 pathlib.Path(sys.argv[1]).read_bytes()
except PermissionError:
 pass
else:
 raise RuntimeError("Isolation failed: another task was readable")
pathlib.Path("scratch/probe-output.txt").write_text("allowed")
'''
    (workspace/'probe.txt').write_text('allowed')
    result=subprocess.run([binary,'sandbox',*config,'-P','task-only','-C',str(workspace),
        sys.executable,'-c',script,str(forbidden)],capture_output=True,text=True)
    if result.returncode: raise RuntimeError('Task isolation check failed: '+result.stderr[-3000:])
    (workspace/'probe.txt').unlink();(workspace/'scratch/probe-output.txt').unlink()
    return {'passed':True,'other_task_read':'denied','workspace_read_write':'allowed','numpy_import':'allowed'}
