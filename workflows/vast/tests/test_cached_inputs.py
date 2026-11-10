"""Continuation inputs stay on the VM and remain tied to verified identities."""

import json
import os
import io
import tarfile
import time
from types import SimpleNamespace

import pytest

from workflows.vast import common, drive, run as workflow, worker


def test_manual_record_backups_only_suffice_for_the_verified_continuation_parent():
    parent = {'session':'parent','instance_id':57431449}
    plan = {'continuation_kind':'pretrain','continuation_of':parent,
            'policy':{'checkpoint_upload_mode':'manual'}}
    status = {'backups_complete':False,'record_backups_complete':True}
    assert workflow.ancestor_backups_verified(plan,parent,status)
    assert not workflow.ancestor_backups_verified(plan,{'session':'other'},status)
    assert not workflow.ancestor_backups_verified({},parent,status)
    assert not workflow.ancestor_backups_verified({**plan,'continuation_kind':'finetune'},parent,status)
    assert not workflow.ancestor_backups_verified(plan,parent,{'record_backups_complete':False})


@pytest.mark.parametrize('changed_checkpoint', [False, True])
def test_cached_checkpoint_and_data_are_verified_and_linked_without_download(tmp_path, changed_checkpoint):
    parent = tmp_path/'parent'
    run = parent/'code/runs/parent-run'
    checkpoint = run/'inputs/pretrain.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'exact saved model')
    identity = common.hashes(checkpoint)
    data = run/'data/priv_challenges.json'
    data.parent.mkdir()
    data.write_text('{"task":{"train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[3]]}]}}')
    payload = run/'data/instances/payload.json'
    payload.parent.mkdir()
    payload.write_text('{"input":[[1]],"output":[[2]]}')
    common.atomic_json(parent/'launch.json', {'branch':'branch','session':'parent','instance_id':57431449,'run_name':'parent-run'})
    workspace = tmp_path/'child'
    plan = {'branch':'branch','run_name':'child-run','continuation_of':{'branch':'branch','session':'parent','instance_id':57431449,'remote_workspace':str(parent)},
            'reuse_parent_inputs':{'checkpoint_path':'inputs/pretrain.pt','data_files':[
                {'path':data.name,'identity':common.hashes(data)},
                {'path':'instances/payload.json','identity':common.hashes(payload)}]}}
    spec = {'checkpoint':{'drive_id':'model','name':'model.pt','size':identity['size'],'sha256':identity['sha256']},'skip_data_build':True}
    metadata = {'parents':['folder'],'name':'model.pt','size':str(identity['size']),
                'md5Checksum':identity['md5'],'sha256Checksum':identity['sha256']}
    store = SimpleNamespace(branch_id='branch-folder', assert_under=lambda *a: None,
                            metadata=lambda _: metadata, _verify_binary=drive.Drive._verify_binary,
                            download=lambda *a, **kw: pytest.fail('Cached continuation downloaded checkpoint'))
    if changed_checkpoint:
        checkpoint.write_bytes(b'changed model')
        with pytest.raises(ValueError, match='On-VM checkpoint'):
            worker.prepare_inputs(store,workspace,plan,spec)
        return
    inputs = worker.prepare_inputs(store,workspace,plan,spec)
    destination = workspace/'code/runs/child-run/inputs/pretrain.pt'
    assert os.path.samefile(checkpoint,destination)
    assert os.path.samefile(data,workspace/'code/runs/child-run/data'/data.name)
    assert os.path.samefile(payload,workspace/'code/runs/child-run/data/instances/payload.json')
    assert inputs['checkpoint']['sha256'] == identity['sha256']


@pytest.mark.parametrize('pretraining,manual,bad_checkpoint,finetune_resume,evaluation_only', [
    (False,False,False,False,False), (True,False,False,False,False), (True,True,False,False,False), (True,True,True,False,False),
    (False,False,False,True,False), (False,False,True,True,False),
    (False,False,False,False,True), (False,False,True,False,True)])
def test_continuation_reuses_running_vm_and_caps_new_window_by_original_total_budget(tmp_path, monkeypatch, pretraining, manual, bad_checkpoint, finetune_resume, evaluation_only):
    root=tmp_path/'repo'; parent_cache=root/'results/branch/vast_runs/parent'; parent_cache.mkdir(parents=True)
    checkpoint={'drive_id':'model','name':'model.pt','size':123,'sha256':'a'*64}
    original=json.dumps({} if pretraining else {'checkpoint': {'different': 'input'} if finetune_resume or evaluation_only else checkpoint}).encode()
    with tarfile.open(parent_cache/'source.tar.gz','w:gz') as archive:
        info=tarfile.TarInfo('specs/original.json');info.size=len(original);archive.addfile(info,io.BytesIO(original))
    parent={'branch':'branch','session':'parent','instance_id':57431449,'run_name':'parent-run','label':'mine',
            'spec_relative':'specs/original.json','source':common.hashes(parent_cache/'source.tar.gz'),
            'deadline_unix':time.time()-1,'rented_at_unix':time.time()-15*3600,
            'policy':{'backup_grace_seconds':1,'max_runtime_hours':15,'max_total_usd':150,'remote_root':'/workspace','python':'python3',
                      'checkpoint_upload_mode':'manual' if manual else 'automatic'},
            'quote':{'hourly_usd':9,'bandwidth_reserved_usd':2},'ssh_key_path':'key',
            'drive_config':{},'drive_folder_id':'parent-folder','remote_workspace':'/workspace/parent'}
    config=root/'config.json';config.write_text(json.dumps({'run_name':'child','eval_only':evaluation_only,'finetune_only':not pretraining and not evaluation_only,'finetune_resume':finetune_resume,
        'pretrain':{'epochs':38},'skip_data_build':True,'checkpoint':checkpoint}))
    monkeypatch.setattr(workflow,'ROOT',root)
    monkeypatch.setattr(workflow,'load_plan',lambda _: (parent_cache,parent))
    def source(cache,plan,path,*a,**kw):
        path.write_bytes(b'frozen source');return common.hashes(path),config.read_bytes(),[],{'pretrain':{} if pretraining else None,'finetunes':{} if pretraining or evaluation_only else {'dev':{}}},[],{}
    monkeypatch.setattr(workflow,'retry_source',source)
    calls=[];instance={'id':57431449,'label':'mine','actual_status':'running','intended_status':'running','gpu_name':'B300'}
    def label(i,value): calls.append('label');instance['label']=value
    client=SimpleNamespace(instance=lambda _:dict(instance),label=label,
        start=lambda _:pytest.fail('Already running VM was restarted'),stop=lambda _:pytest.fail('B300 was stopped'),
        create=lambda *a:pytest.fail('New rental created'))
    monkeypatch.setattr(workflow,'client_for',lambda _:client)
    output_metadata={'parents':['wrong-folder' if bad_checkpoint else 'parent-folder'],'name':'model.pt','size':'123','sha256Checksum':'a'*64}
    store=SimpleNamespace(metadata=lambda identity: output_metadata if identity=='model' else {},_verify_binary=lambda *a:None,
                          create_run=lambda _:calls.append('folder') or 'child-folder',
                          download=lambda *a,**kw:pytest.fail('Continuation downloaded inputs'))
    monkeypatch.setattr(workflow,'store_for',lambda *a,**kw:store)
    monkeypatch.setattr(workflow,'manifest_from_drive',lambda *a:{'ok':not manual,'finalized':True,'backups_complete':not manual,
        'record_backups_complete':True,'archive':{'id':'archive','name':'results'}})
    monkeypatch.setattr(workflow,'verify_idle_instance',lambda *a:calls.append('idle'))
    reuse={'checkpoint_path':'inputs/pretrain.pt','checkpoint_size':123,'data_files':[],'training_tasks':40,'demonstration_pairs':121}
    monkeypatch.setattr(workflow,'SSH',lambda *a:SimpleNamespace(python=lambda *a,**kw:json.dumps(reuse)))
    monkeypatch.setattr(workflow,'handoff',lambda *a:calls.append('handoff'))
    args=workflow.build_parser().parse_args(['continue','--session','parent','--config',str(config),'--new-session','child',
                                           '--reuse-checkpoint','inputs/pretrain.pt','--runtime-hours','1.5','--detach'])
    if bad_checkpoint:
        with pytest.raises(ValueError,match='checkpoint differs'):
            workflow.continue_run(args)
        assert calls==[]
        return
    assert workflow.continue_run(args)==0
    saved=json.loads((root/'results/branch/vast_runs/child/launch.json').read_text())
    assert saved['reuse_parent_inputs']==reuse and saved['continuation_of']['ok'] is (not manual)
    assert saved['continuation_kind']==('evaluation' if evaluation_only else 'pretrain' if pretraining else 'finetune')
    assert saved['evaluation_only'] is evaluation_only
    funded=parent['rented_at_unix']+(150-2)/9*3600
    assert parent['deadline_unix']<saved['deadline_unix']<=funded
    assert calls==['idle','folder','label','handoff']
