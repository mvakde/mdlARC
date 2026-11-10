"""Reuse never allocates or executes, and preserves recovered source/results."""
import json
from types import SimpleNamespace

import pytest

from scripts.colab_reuse import prepare


def setup():
    identity=dict(version=1,session='old',run_name='run',endpoint='vm',kernel_id='kernel',ws_session_id='ws',execution_id='exec')
    status=dict(ok=False,backups_complete=True,error='preflight')
    manifest=dict(finalized=True,session='old',run_name='run')
    files={'mdlARC':None,'mdlARC-source.tar.gz':None,'mdlARC-results-manifest.json':manifest,
        'mdlARC-run-status.json':status,'mdlARC-results.part0000':None,'.mdlarc-drive-control':None}
    mutations=[];aliases={}
    class Local:
        endpoint='vm';kernel_id='kernel'
        def model_copy(self,update):return SimpleNamespace(endpoint=self.endpoint,kernel_id=self.kernel_id,**update)
    local=Local()
    def request(proxy,method,path,**kwargs):
        assert proxy=='proxy'
        name=path.removeprefix('/api/contents/content').lstrip('/')
        if method=='GET':
            if not name:return {'content':[{'path':'/content/'+n} for n in files]}
            if name=='mdlARC/runs/run':
                return {'type':'directory','content':[{'path':'/content/mdlARC/runs/run/'+n,'type':'file'} for n in files.get('run_contents',[])]}
            return {'content':json.dumps(files[name]),'format':'text'}
        assert method=='PATCH'
        new=kwargs['json']['path'].removeprefix('/content/')
        assert new not in files
        files[new]=files.pop(name);mutations.append((name,new))
        return {}
    backend=SimpleNamespace(local_session=lambda:local,assignment=lambda endpoint:'proxy',
        existing_kernel=lambda proxy,kernel:{'id':'kernel','execution_state':'idle'},request=request,
        state=SimpleNamespace(store=SimpleNamespace(get=lambda name:aliases.get(name),add=lambda v:aliases.update({v.name:v}))))
    return backend,identity,status,files,mutations,aliases


def test_reuse_preserves_every_artifact_and_original_vm():
    b,i,s,f,m,a=setup()
    result=prepare(b,i,{'state':'failed'},s,'new')
    assert a['new'].endpoint=='vm' and a['new'].kernel_id=='kernel'
    assert len(m)==6 and all(n.endswith('.completed.old') for n in f)
    assert result['previous_finalized_manifest']['finalized'] is True


def test_reuse_leaves_historical_result_chunks_untouched():
    b,i,s,f,m,a=setup()
    historical='mdlARC-results.part0000.completed.earlier'
    f[historical]=None
    prepare(b,i,{'state':'failed'},s,'new')
    assert historical in f
    assert all(old != historical for old,new in m)
    assert len(m)==6


@pytest.mark.parametrize('problem',['busy','unverified','running','wrong_manifest','changed_status','changed_kernel','alias_exists'])
def test_reuse_rejects_before_any_mutation(problem):
    b,i,s,f,m,a=setup();journal={'state':'failed'}
    if problem=='busy':b.existing_kernel=lambda *a:{'execution_state':'busy'}
    if problem=='unverified':s['backups_complete']=False;f['run_contents']=['checkpoint.pt']
    if problem=='running':journal['state']='running'
    if problem=='wrong_manifest':f['mdlARC-results-manifest.json']['session']='other'
    if problem=='changed_status':f['mdlARC-run-status.json']={**s,'error':'other'}
    if problem=='changed_kernel':i['kernel_id']='other'
    if problem=='alias_exists':a['new']=object()
    with pytest.raises(ValueError):prepare(b,i,journal,s,'new')
    assert not m


def test_failed_initialization_without_checkpoints_can_be_preserved():
    b,i,s,f,m,a=setup();s['backups_complete']=False
    result=prepare(b,i,{'state':'needs_recovery'},s,'new')
    assert result['empty_failed_initialization'] is True
    assert result['previous_backups_complete'] is False and a['new'].endpoint=='vm'


@pytest.mark.parametrize('error_kind',['http','kernel'])
def test_verified_native_setup_failure_can_reuse_without_worker_control(error_kind):
    import requests
    from scripts.colab_connection import KernelUnavailable
    b,i,s,f,m,a=setup()
    s['drive_folder_id']='verified-run-folder'
    del f['.mdlarc-drive-control']
    original=b.request
    def request(proxy,method,path,**kwargs):
        if method=='GET' and path.endswith('/.mdlarc-drive-control'):
            if error_kind=='kernel':raise KernelUnavailable('contents 404')
            response=requests.Response();response.status_code=404
            raise requests.HTTPError(response=response)
        return original(proxy,method,path,**kwargs)
    b.request=request
    result=prepare(b,i,{'state':'failed'},s,'new')
    assert a['new'].endpoint=='vm' and len(m)==5
    assert not any('.mdlarc-drive-control' in name for name in result['preserved'])


def test_missing_worker_control_does_not_hide_a_missing_kernel():
    from scripts.colab_connection import KernelUnavailable
    b,i,s,f,m,a=setup()
    s['drive_folder_id']='verified-run-folder'
    del f['.mdlarc-drive-control']
    original=b.request
    def request(proxy,method,path,**kwargs):
        if method=='GET' and path.endswith('/.mdlarc-drive-control'):
            b.existing_kernel=lambda *args: (_ for _ in ()).throw(KernelUnavailable('gone'))
            raise KernelUnavailable('contents 404')
        return original(proxy,method,path,**kwargs)
    b.request=request
    with pytest.raises(KernelUnavailable):prepare(b,i,{'state':'failed'},s,'new')
    assert not m and not a
