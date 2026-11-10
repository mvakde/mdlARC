"""Prepare one task, invoke Codex in isolation, and publish selected outputs."""
from pathlib import Path
import dataclasses,json,shutil,tempfile,sys
from .storage import *
from .config import FOLDERS
from .isolation import git_baseline,overrides,codex_binary,probe

def prepare(config,name,split,workspace):
    from .data import load_puzzle
    task=load_puzzle(config,name,split).path.parent
    stage=config.stage
    if stage=='summarise-task':
        from task_variations.summary.prompt import build_prompt
        build=build_prompt(config,name,split);prompt=build.prompt_text
        (workspace/'AGENTS.md').write_text('Use only the supplied prompt. Do not inspect other folders.\n'
            'All JSON and JSONL you create, including generated code, must be compact with no indentation. '
            'Use json.dumps(value, separators=(",", ":")) in Python and JSON.stringify(value) in JavaScript.\n'
            'Bound disk use: estimate bytes, file counts and peak temporary space across tasks, seeds, pairs, steps, workers, retries and retained runs. '
            'Keep scratch diagnostics and logs bounded; avoid duplicate reconstructible payloads. Preserve required evidence and hashes. '
            'Use the existing lossless trace codec for saved execution traces, keeping initial and exceptional before states. Generated code follows these rules too. '
            'Before storage optimization edits, report measured projected net savings, runtime and peak disk cost, including journals, indexes and backups.\n')
        (workspace/'prompt.txt').write_text(prompt)
    elif stage=='generate-from-summary':
        from task_variations.generate.prompt import build_prompt
        build=build_prompt(config,name,split);prompt=build.prompt_text
        (workspace/'AGENTS.md').write_text('Use only the supplied prompt. Do not inspect other folders.\n'
            'All JSON and JSONL you create, including generated code, must be compact with no indentation. '
            'Use json.dumps(value, separators=(",", ":")) in Python and JSON.stringify(value) in JavaScript.\n'
            'Bound disk use: estimate bytes, file counts and peak temporary space across tasks, seeds, pairs, steps, workers, retries and retained runs. '
            'Keep scratch diagnostics and logs bounded; avoid duplicate reconstructible payloads. Preserve required evidence and hashes. '
            'Use the existing lossless trace codec for saved execution traces, keeping initial and exceptional before states. Generated code follows these rules too. '
            'Before storage optimization edits, report measured projected net savings, runtime and peak disk cost, including journals, indexes and backups.\n')
        (workspace/'prompt.txt').write_text(prompt)
    elif stage=='review-first-generation':
        from task_variations.review.workspace import stage_workspace,PROMPT
        samples=stage_workspace(config,name,split,workspace);prompt=PROMPT
    else:
        from task_variations.improve.prompt import build_prompt
        build=build_prompt(config,name,split);prompt=build.prompt_text
        for n,text in build.workspace_files.items():
            p=workspace/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    commit=git_baseline(workspace)
    baseline={str(p.relative_to(workspace)):sha(p) for p in workspace.rglob('*') if p.is_file() and '.git' not in p.parts and 'scratch' not in p.parts}
    return task,prompt,commit,baseline

def snapshot(workspace,dest,prompt,baseline):
    dest.mkdir(parents=True,exist_ok=False)
    for n in baseline:
        p=dest/'workspace'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((workspace/n).read_bytes())
    atomic_write_text(dest/'prompt.txt',prompt)
    atomic_write_json(dest/'manifest.json',{'source':'exact','files':{'prompt.txt':sha(dest/'prompt.txt'),**{'workspace/'+n:h for n,h in baseline.items()}}})
def _validate(stage,workspace,baseline,commit,response):
    if stage=='summarise-task':
        from task_variations.summary.validation import validate_summary_xml
        v=validate_summary_xml(response);return v.to_dict()
    if stage=='generate-from-summary':
        from task_variations.generate.generator_validation import validate_generator_response
        return validate_generator_response(response).to_dict()
    if stage=='review-first-generation':
        from task_variations.review.workspace import validate_outputs
        return validate_outputs(workspace,baseline)
    from task_variations.improve.workspace import validate_workspace
    result=validate_workspace(workspace,commit,baseline['generator.py']).to_dict()
    coverage=workspace/'scratch/review_coverage.md'
    if coverage.is_symlink() or not coverage.is_file() or not coverage.read_text().strip():
        result['valid']=False;result['errors'].append('Missing or empty scratch/review_coverage.md')
    return result

def invoke(config,workspace,prompt,baseline,commit,inputs):
    from openai_codex import Codex,CodexConfig,ApprovalMode
    settings=overrides(workspace)
    client=CodexConfig(codex_bin=codex_binary(),config_overrides=settings,cwd=str(workspace))
    # Do not select a legacy Sandbox preset: it overrides permission profiles.
    with Codex(client) as codex:
        thread=codex.thread_start(model=config.default_model,cwd=str(workspace),approval_mode=ApprovalMode.deny_all,
            config={'model_reasoning_effort':config.default_reasoning_effort,'default_permissions':'task-only'},ephemeral=True,
            developer_instructions='Work only inside this task workspace. Do not read parent or sibling directories. Never request a sandbox escalation.')
        atomic_write_text(inputs/'developer-instructions.txt','Work only inside this task workspace. Do not read parent or sibling directories. Never request a sandbox escalation.')
        for attempt in range(config.repair_turns+1):
            atomic_write_text(inputs/('turn-%02d-prompt.txt'%(attempt+1)),prompt)
            result=thread.run(prompt);response=result.final_response or ''
            atomic_write_text(inputs/'saved-output'/('turn-%02d-response.md'%(attempt+1)),response)
            if 'completed' not in str(result.status).lower() or getattr(result,'error',None):
                raise RuntimeError('Codex did not complete its turn: '+str(result.status))
            validation=_validate(config.stage,workspace,baseline,commit,response)
            atomic_write_json(inputs/'saved-output/validation.json',validation)
            if validation.get('valid'): return response,validation
            prompt='Continue in this same workspace and repair the deliverable. Preserve all staged evidence. Resolve these validation errors:\n'+json.dumps(validation.get('errors',[]))
        raise ValueError('Model output failed validation: '+json.dumps(validation.get('errors',[])))

def run_task(config,name,split,force=False,dry_run=False,invoker=None):
    from .data import load_puzzle
    task=load_puzzle(config,name,split).path.parent;split=task.parent.name
    with task_lock(task):
        if not force and not dry_run and current(task,config.stage,config.default_model,config.default_reasoning_effort):
            return {'task':name,'stage':config.stage,'state':'already-complete'}
        with tempfile.TemporaryDirectory(prefix='arc-task-') as temp:
            workspace=Path(temp)/'workspace';workspace.mkdir()
            task,prompt,commit,baseline=prepare(config,name,split,workspace)
            pending=task/'llm-inputs'/('.pending-'+config.stage_folder)
            if pending.exists():
                failed=task/'saved-outputs/failed-inputs'/(now().replace(':','-')+'-'+config.stage_folder)
                failed.parent.mkdir(parents=True,exist_ok=True);pending.rename(failed)
            snapshot(workspace,pending,prompt,baseline)
            metadata={'stage':config.stage,'model':config.default_model,'effort':config.default_reasoning_effort,
                      'dependencies':dependency_hashes(task,config.stage),'state':'prepared' if dry_run else 'running',
                      'permissions':list(overrides(workspace)),'prepared_at':now()}
            atomic_write_json(pending/'metadata.json',metadata)
            if dry_run:
                destination=task/'llm-inputs'/('prepared-'+config.stage_folder)
                if destination.exists(): shutil.rmtree(destination)
                pending.rename(destination)
                return {'task':name,'stage':config.stage,'state':'prepared','inputs':str(destination)}
            # Check the actual OS boundary, without making any model call.
            forbidden=next((p for p in config.results_root.glob('*/*/*/task.json') if p.parent!=task),None)
            if forbidden is None: forbidden=task/'task.json'
            try:
                metadata['isolation']=probe(workspace,forbidden)
                response,validation=(invoker or invoke)(config,workspace,prompt,baseline,commit,pending)
                if invoker is not None:
                    validation=_validate(config.stage,workspace,baseline,commit,response)
                if not validation.get('valid'):
                    raise ValueError('Output failed host validation: '+json.dumps(validation.get('errors',[])))
                if config.stage=='summarise-task': output=response.encode()
                elif config.stage=='generate-from-summary':
                    from task_variations.generate.generator_validation import validate_generator_response
                    output=(validate_generator_response(response).extracted_code.rstrip()+'\n').encode()
                elif config.stage=='review-first-generation': output=(workspace/'review.md').read_bytes()
                else: output=(workspace/'generator.py').read_bytes()
                # Preserve any previous selected input package before replacing it.
                dest=task/'llm-inputs'/config.stage_folder
                if dest.exists():
                    saved=task/'saved-outputs/llm-inputs'/(now().replace(':','-')+'-'+config.stage_folder)
                    saved.parent.mkdir(parents=True,exist_ok=True);dest.rename(saved)
                    relocate_audit(dest,saved)
                metadata.update(state='succeeded',completed_at=now(),output_sha256=sha_bytes(output))
                atomic_write_json(pending/'metadata.json',metadata)
                atomic_write_text(pending/'saved-output/response.md',response)
                if config.stage=='review-first-generation':
                    shutil.copytree(workspace/'target/samples',pending/'reviewed-samples')
                    m=read(pending/'reviewed-samples/manifest')
                    m['file_sha256']={p.name:sha(p) for p in (pending/'reviewed-samples').glob('seed-*.json')}
                    atomic_write_json(pending/'reviewed-samples/manifest',m)
                    metadata['author']='automated'
                    atomic_write_json(pending/'metadata.json',metadata)
                manifest=read(pending/'manifest.json')
                manifest['files'].update({str(p.relative_to(pending)):sha(p) for p in pending.rglob('*') if p.is_file() and p!=pending/'manifest.json'})
                atomic_write_json(pending/'manifest.json',manifest)
                pending.rename(dest)
                publish(task/OUTPUTS[config.stage],output)
                if config.stage=='generate-from-reviews':
                    coverage=workspace/'scratch/review_coverage.md'
                    if coverage.is_file(): publish(task/'review_coverage.md',coverage.read_bytes())
                validation_name={'generate-from-summary':'validation_v1.json','generate-from-reviews':'validation_v2.json'}.get(config.stage)
                if validation_name: atomic_write_json(task/validation_name,validation)
                prov=read(task/'provenance.json') if (task/'provenance.json').exists() else {'schema_version':2,'stages':{},'files':{}}
                prov['stages'][config.stage]={k:metadata[k] for k in ('model','effort','dependencies','output_sha256','completed_at')}
                if config.stage=='review-first-generation': prov['stages'][config.stage]['author']='automated'
                prov['files']={p.name:sha(p) for p in task.iterdir() if p.is_file() and p.name not in ('provenance.json','.lock','manual-review.json')}
                atomic_write_json(task/'provenance.json',prov)
                update_index(config.results_root)
                return {'task':name,'stage':config.stage,'state':'succeeded'}
            except Exception as exc:
                atomic_write_json(pending/'error.json',{'error':str(exc),'state':'failed','time':now()})
                raise
