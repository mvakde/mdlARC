"""Prepare a new named run on a fully recovered, idle Colab VM.

Uses bounded file APIs to preserve the previous workspace, never kernel code,
kernel creation, interruption, credential transfer, or VM allocation.
"""
import argparse
import base64
import json
import logging
from pathlib import Path
import re
import time
from urllib.parse import quote

if __package__:
    from . import colab_accounts, colab_connection
else:
    import colab_accounts
    import colab_connection


def content_json(model):
    text = model['content']
    if model.get('format') == 'base64': text = base64.b64decode(text).decode()
    return json.loads(text)


def prepare(backend, identity, journal, status, new_session):
    if journal.get('state') not in ('completed', 'failed', 'needs_recovery'):
        raise ValueError('Reuse requires a terminal run and verified local recovery')
    old_session, old_run = identity['session'], identity['run_name']
    colab_connection.validate_identity(identity, old_session, old_run)
    if old_session == new_session or backend.state.store.get(new_session) is not None:
        raise ValueError('Reuse requires a fresh local session alias')
    local = backend.local_session()
    if local is None or local.endpoint != identity['endpoint'] or local.kernel_id != identity['kernel_id']:
        raise ValueError('Original VM/kernel differs from recovered run')
    proxy = backend.assignment(identity['endpoint'])
    kernel = backend.existing_kernel(proxy, identity['kernel_id'])
    if kernel.get('execution_state') != 'idle':
        raise ValueError('Original kernel is busy; no new run may be submitted')
    def read(path):
        import requests
        for attempt in range(3):
            try:
                return backend.request(proxy, 'GET', '/api/contents/'+quote(path.strip('/'),safe='/'))
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2: raise
                time.sleep(2**attempt)
    manifest = content_json(read('/content/mdlARC-results-manifest.json'))
    if (manifest.get('finalized') is not True or manifest.get('session') != old_session
            or manifest.get('run_name') != old_run):
        raise ValueError('Original finalized result manifest is missing or differs')
    if content_json(read('/content/mdlARC-run-status.json')) != status:
        raise ValueError('Recovered status differs from the VM; recovery is required')
    empty_failed_initialization = False
    if status.get('backups_complete') is not True:
        if status.get('ok') is not False:
            raise ValueError('Unverified successful run cannot be reused')
        # A failure before the backup worker starts has no checkpoint ledger.
        # Inspect filenames/metadata only; never download a checkpoint. Preserve
        # the entire workspace even when it has no output checkpoint to back up.
        pending = ['/content/mdlARC/runs/'+old_run]; inspected = 0
        while pending:
            directory = read(pending.pop())
            if directory.get('type') != 'directory':
                raise ValueError('Failed-run inspection did not return a directory')
            for item in directory['content']:
                inspected += 1
                if inspected > 10000: raise ValueError('Failed-run inspection exceeded its bound')
                if item['type'] == 'directory': pending.append(item['path'])
                elif Path(item['path']).name == 'checkpoint_backups.json':
                    raise ValueError('Existing checkpoint ledger requires backup recovery before reuse')
                elif Path(item['path']).suffix in {'.pt','.pth','.ckpt','.safetensors','.partial'}:
                    raise ValueError('Unverified checkpoint exists; recover its backup before reuse')
        empty_failed_initialization = True
    listing = read('/content')['content']
    names = {Path(item['path']).name for item in listing}
    if status.get('drive_folder_id') and status.get('backups_complete') is True:
        # Hidden queue state may be omitted from a directory listing. A failure
        # during native setup can also precede creation of the backup worker.
        if '.mdlarc-drive-control' not in names:
            import requests
            try:
                read('/content/.mdlarc-drive-control')
            except (requests.HTTPError, colab_connection.KernelUnavailable) as exc:
                if isinstance(exc, requests.HTTPError) and (
                    exc.response is None or exc.response.status_code != 404
                ):
                    raise
                # The connection backend maps a contents 404 to
                # KernelUnavailable too. Confirm the original kernel is still
                # idle before treating this optional control path as absent.
                if backend.existing_kernel(proxy, identity['kernel_id']).get('execution_state') != 'idle':
                    raise ValueError('Original kernel is not idle; no workspace may be reused')
            else:
                names.add('.mdlarc-drive-control')
    preserve = sorted(name for name in names if name in (
        'mdlARC','mdlARC-source.tar.gz','mdlARC-results-manifest.json','mdlARC-run-status.json',
        'mdlARC-results.tar.gz','.mdlarc-drive-control','mdlARC-local-records.tar.gz',
        'mdlARC-local-records-status.json') or re.fullmatch(r'mdlARC-results\.part[0-9]+', name))
    suffix = '.completed.' + old_session
    if not {'mdlARC','mdlARC-source.tar.gz','mdlARC-results-manifest.json'}.issubset(preserve):
        raise ValueError('Original source or completion files are missing')
    if any(name+suffix in names for name in preserve):
        raise ValueError('Previous preservation exists; inspect partial reuse rather than retrying')
    # Preserve source and all result/control artifacts before clearing the live
    # paths. A partial failure remains inspectable and never starts execution.
    for name in preserve:
        path='/content/'+name
        backend.request(proxy,'PATCH','/api/contents/'+quote(path.strip('/'),safe='/'),json={'path':path+suffix})
    remaining={Path(item['path']).name for item in read('/content')['content']}
    if any(name in remaining for name in preserve):
        raise ValueError('Workspace preservation did not finish')
    alias = local.model_copy(update={'name':new_session,'last_execution':None,'running':None,'keep_alive_pid':None})
    backend.state.store.add(alias)
    return dict(previous_session=old_session,session=new_session,endpoint=local.endpoint,kernel_id=local.kernel_id,
                preserved=[name+suffix for name in preserve],previous_finalized_manifest=manifest,
                empty_failed_initialization=empty_failed_initialization,
                previous_backups_complete=status.get('backups_complete'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account',required=True)
    parser.add_argument('--previous',required=True)
    parser.add_argument('--session',required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    logging.getLogger().setLevel(logging.CRITICAL)
    colab_accounts.configure_cli(args.account)
    identity=json.loads(args.identity.read_text());journal=json.loads(args.journal.read_text());status=json.loads(args.status.read_text())
    if colab_accounts.record_account(identity) != args.account or identity['session'] != args.previous:
        raise ValueError('Reuse account/session differs from the saved execution')
    report=prepare(colab_connection.ColabBackend(args.previous),identity,journal,status,args.session)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Recovered workspace preserved; new alias uses the original idle VM/kernel.')


if __name__=='__main__':main()
