"""Explicit resample releases preserve family lineage and immutable old packages."""
import copy
import json
import zipfile

import pytest
from test_family_data import corpus, composition, task, save
from dataset_building_scripts import family_data as fd


def release(c, family, job, schedule=None):
    base = c.synth/'data/compositions/releases/fixture'
    archive_path = base/'families/one.zip'
    archive_path.parent.mkdir(parents=True)
    records = []
    schedule = schedule or [[0, 2, 1], [1, 4, 3]]
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for seed, a, b in schedule:
            puzzle = {'train': [{'input': [[i,i]], 'output': [[i+1,i+1]]} for i in range(a)],
                      'test': [{'input': [[i+a,i+a]], 'output': [[i+a+1,i+a+1]]} for i in range(b)]}
            raw = fd.encoded(puzzle); name = f'seed-{seed:010d}.json'
            archive.writestr(name, raw)
            records.append({'seed': seed, 'num_train': a, 'num_test': b, 'member': name, 'sha256': fd.sha(raw), 'bytes': len(raw)})
        report = {'valid': True, 'family': family, 'stage': 'author', 'generator_sha256': '0'*64, 'records': records}
        archive.writestr('records.json', fd.encoded(report))
    source = {'stage': 'author', 'parents': ['b', 'b'], 'dataset': 'arc2/evaluation', 'generator_sha256': '0'*64,
              'schedule': schedule, 'seeds': [row[0] for row in schedule]}
    from collections import Counter
    counts = {'families': 1, 'samples': len(schedule), 'distribution': {'demos': dict(Counter(str(x[1]) for x in schedule)), 'queries': dict(Counter(str(x[2]) for x in schedule))}}
    inv = {'schema': 'arc-composition-resample-v1', 'release': 'fixture', 'counts': counts,
           'families': {family: source}, 'source_sha256': {}}
    save(base/'inventory.json', inv)
    summary = {'schema': inv['schema'], 'release': 'fixture', 'ready': True, 'failures': [], 'counts': counts,
               'inventory_sha256': fd.file_sha(base/'inventory.json'), 'families': {family: {
                   'archive': 'families/one.zip', 'archive_bytes': archive_path.stat().st_size,
                   'archive_sha256': fd.file_sha(archive_path)}}}
    h = save(base/'release.json', summary)
    identity = {'path': 'data/compositions/releases/fixture/release.json', 'sha256': h}
    c.raw['pretrain_data']['discovery']['composition_release'] = identity
    return identity, base


def test_resample_exact_members_and_counts_survive_package(corpus, monkeypatch):
    c = corpus; job, family = composition(c, 'selected', ['b', 'b'])
    identity, base = release(c, family, job)
    config = fd.normalize_config(c.raw, c.root)
    catalog, payload = fd.discover(c.root, c.synth, config)
    assert catalog['summary']['ready']
    instances = catalog['families'][family]['instances']
    assert [r['pair_count'] for r in instances] == [3, 7]
    assert all(r['composition_release'] == identity and r['stage'] == 'author' for r in instances)
    # A later accepted source job must not silently join an explicitly selected release.
    composition(c, 'later', ['b', 'b'])
    refreshed, _ = fd.discover(c.root, c.synth, config)
    assert set(refreshed['families']) == set(catalog['families'])
    packaged = fd.package_catalog(refreshed, fd.discover(c.root, c.synth, config)[1], c.root/'assets_tmp/new.zip')
    with zipfile.ZipFile(packaged['archive_path']) as handle:
        saved = json.loads(handle.read('family_catalog.json'))
        assert saved['families'][family]['ancestors'] == ['arc2_eval/b']
        assert [len(json.loads(handle.read(r['path']))['train']) for r in saved['families'][family]['instances']] == [2, 4]
    packaged['archive_path'] = 'assets_tmp/new.zip'
    config['pretrain_data']['catalog'] = packaged
    output = c.root/'run/data'
    fd.build_family_dataset(config, output, root=c.root)
    import variations
    monkeypatch.setattr(variations, 'ROOT', c.root)
    ds = variations.VariationTrainingDataset(json_path=output/'challenges.json', index_path=output/'variation_index.json', seed=42, max_augments=80)
    sampler = variations.VariationBatchSampler(ds, 4, seed=42, variations_per_parent_per_epoch=1, eval_support_repeats_per_epoch=0)
    observed = set()
    for epoch in range(4):
        indices = sampler._indices_for_epoch(epoch)
        chosen = [i for i in indices if i[1] == family]
        assert len({i[2] for i in chosen}) == 1
        assert len(chosen) in (3, 7)
        observed.add(len(chosen))
        assert len(chosen) == ds.variation_parents[family][chosen[0][2]]['pair_count']
    assert observed == {3, 7}



def test_heldout_ancestry_still_excludes_resampled_family(corpus):
    c = corpus; job, family = composition(c, 'selected', ['b', 'b'])
    _, base = release(c, family, job)
    # Consumed parent context is independently followed by ordinary discovery.
    parent = c.synth/'results/task-variations/arc2/evaluation/b/llm-inputs/01/recorded-inputs.json'
    save(parent, {'demonstrations': [{'id': 'c', 'source_split': 'evaluation'}]})
    cat, _ = fd.discover(c.root, c.synth, fd.normalize_config(c.raw, c.root))
    assert family not in cat['families']
    assert cat['excluded'][family] == 'heldout_or_alias_ancestor'


def test_release_pin_and_archive_corruption_block(corpus):
    c = corpus; job, family = composition(c, 'selected', ['b', 'b'])
    identity, base = release(c, family, job)
    config = fd.normalize_config(c.raw, c.root)
    with (base/'families/one.zip').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='archive changed'):
        fd.discover(c.root, c.synth, config)
    invalid = copy.deepcopy(c.raw)
    invalid['pretrain_data']['discovery']['composition_release']['path'] = '../escape/release.json'
    with pytest.raises(ValueError, match='Invalid composition release'):
        fd.normalize_config(invalid, c.root)


def test_incomplete_release_never_becomes_training_data(corpus):
    c = corpus; job, family = composition(c, 'selected', ['b', 'b'])
    identity, base = release(c, family, job)
    summary = json.loads((base/'release.json').read_bytes())
    summary['ready'] = False
    identity['sha256'] = save(base/'release.json', summary)
    with pytest.raises(ValueError, match='Incomplete or corrupt'):
        fd.discover(c.root, c.synth, fd.normalize_config(c.raw, c.root))


def test_authorized_single_seed_omission_is_explicit_and_preserves_family(corpus):
    c=corpus; job,family=composition(c,'selected',['b','b'])
    identity,base=release(c,family,job)
    summary=json.loads((base/'release.json').read_bytes())
    archive_path=base/'families/one.zip'
    with zipfile.ZipFile(archive_path) as z:
        report=json.loads(z.read('records.json'))
        keep=report['records'][0];raw=z.read(keep['member'])
    report['records']=[keep]
    with zipfile.ZipFile(archive_path,'w') as z:
        z.writestr(keep['member'],raw);z.writestr('records.json',fd.encoded(report))
    omission={'family':family,'seed':1,'num_train':4,'num_test':3,'error':'explicit fixture failure'}
    h=save(base/'omissions.json',{'inventory_sha256':summary['inventory_sha256'],'omissions':[omission]})
    summary['schema']='arc-composition-resample-v2'
    summary['requested_counts']=copy.deepcopy(summary['counts'])
    summary['counts']={'families':1,'samples':1,'distribution':{'demos':{'2':1},'queries':{'1':1}}}
    summary['selection_exceptions']={'path':'omissions.json','sha256':h}
    summary['families'][family]['archive_bytes']=archive_path.stat().st_size
    summary['families'][family]['archive_sha256']=fd.file_sha(archive_path)
    identity['sha256']=save(base/'release.json',summary)
    cat,_=fd.discover(c.root,c.synth,fd.normalize_config(c.raw,c.root))
    assert cat['summary']['ready']
    assert len(cat['families'][family]['instances'])==1
    assert cat['families'][family]['sampling_omissions']==[omission]
    assert cat['families'][family]['ancestors']==['arc2_eval/b']


def test_preparation_pins_fresh_run_identity_without_changing_recipe(corpus,monkeypatch):
    import sys
    from scripts import prepare_family_data as prepare
    c=corpus;recipe=c.root/'recipe.json'
    save(recipe,{'run_name':'old-identity','dataset':c.raw})
    before=recipe.read_bytes();output=c.root/'specs/temp specs/new.json'
    monkeypatch.setattr(prepare,'ROOT',c.root)
    monkeypatch.setattr(sys,'argv',['prepare_family_data','--spec',str(recipe),'--synth-root',str(c.synth),
        '--audit-dir',str(c.root/'results/audit'),'--package',str(c.root/'assets_tmp/new.zip'),
        '--output-spec',str(output),'--run-name','fresh-counts-identity'])
    prepare.main()
    result=json.loads(output.read_bytes())
    assert result['run_name']=='fresh-counts-identity'
    assert result['dataset']['pretrain_data']['catalog']['sha256']==fd.file_sha(c.root/'assets_tmp/new.zip')
    assert recipe.read_bytes()==before


def test_two_omissions_require_explicit_family_exception(corpus):
    c=corpus;job,family=composition(c,'selected',['b','b'])
    identity,base=release(c,family,job,[[0,2,1],[1,4,3],[2,3,1]])
    summary=json.loads((base/'release.json').read_bytes())
    path=base/'families/one.zip'
    with zipfile.ZipFile(path) as z:
        report=json.loads(z.read('records.json'));keep=report['records'][0];raw=z.read(keep['member'])
    report['records']=[keep]
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(keep['member'],raw);z.writestr('records.json',fd.encoded(report))
    items=[{'family':family,'seed':seed,'num_train':a,'num_test':b,'error':'fixture'} for seed,a,b in [(1,4,3),(2,3,1)]]
    omissions={'inventory_sha256':summary['inventory_sha256'],'omissions':items}
    summary.update(schema='arc-composition-resample-v2',requested_counts=copy.deepcopy(summary['counts']),
        counts={'families':1,'samples':1,'distribution':{'demos':{'2':1},'queries':{'1':1}}})
    summary['families'][family].update(archive_bytes=path.stat().st_size,archive_sha256=fd.file_sha(path))
    summary['selection_exceptions']={'path':'omissions.json','sha256':save(base/'omissions.json',omissions)}
    identity['sha256']=save(base/'release.json',summary)
    with pytest.raises(ValueError,match='omission limit'):
        fd.discover(c.root,c.synth,fd.normalize_config(c.raw,c.root))
    omissions['default_family_omission_limit']=2
    summary['selection_exceptions']['sha256']=save(base/'omissions.json',omissions)
    identity['sha256']=save(base/'release.json',summary)
    cat,_=fd.discover(c.root,c.synth,fd.normalize_config(c.raw,c.root))
    assert cat['summary']['ready'] and cat['families'][family]['sampling_omissions']==items


def test_preparation_audits_complete_recovery_cache_once(corpus,monkeypatch):
    import sys
    from scripts import prepare_family_data as prepare
    from dataset_building_scripts import family_recovery
    c=corpus;recipe=c.root/'recipe.json'
    save(recipe,{'run_name':'fixture','dataset':c.raw})
    # A missing source file is available only through its exact cached bytes.
    cached=family_recovery.RecoveryCache(c.root/'results/recovery')
    missing=c.synth/'results/task-variations/arc2/training/a/samples/seed-0.json'
    raw=missing.read_bytes();assert cached.save(raw,fd.sha(raw));missing.unlink()
    calls=[];real_discover=prepare.discover
    def checked(*args,**kwargs):
        assert isinstance(kwargs['recovery'],family_recovery.RecoveryCache)
        calls.append(True)
        return real_discover(*args,**kwargs)
    def unnecessary(*args,**kwargs):
        pytest.fail('A complete cache must not trigger recovery or a second audit')
    monkeypatch.setattr(prepare,'ROOT',c.root)
    monkeypatch.setattr(prepare,'discover',checked)
    monkeypatch.setattr(family_recovery,'recover_catalog',unnecessary)
    monkeypatch.setattr(sys,'argv',['prepare_family_data','--spec',str(recipe),'--synth-root',str(c.synth),
        '--audit-dir',str(c.root/'results/audit'),'--recovery-cache',str(c.root/'results/recovery'),
        '--package',str(c.root/'assets_tmp/once.zip'),'--output-spec',str(c.root/'specs/temp specs/once.json')])
    prepare.main()
    assert calls==[True]
    with zipfile.ZipFile(c.root/'assets_tmp/once.zip') as z:
        cat=json.loads(z.read('family_catalog.json'))
    assert cat['summary']['ready'] and cat['summary']['instances_per_epoch']==2
