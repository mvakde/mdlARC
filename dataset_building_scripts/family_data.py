"""Explicit saved-corpus discovery, leakage closure and immutable family packages.

No generation, network access, checkpoint loading, or live inventory refresh occurs here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from .build_variation_dataset import ORIGINAL_FILES, _solved_pairs, _write_outputs
from .task_splits import content_sha256, load_parent_split, validate_split_tasks
from .composition_release import validate_identity, load_release, puzzle_bytes

GROUPS = {"arc2_train": "arc2/training", "arc2_eval": "arc2/evaluation",
          "conceptarc": "conceptarc/training", "arc1_filtered": "arc1-filtered/training"}
SCHEMA = "arc-family-catalog-v1"
SAMPLING = {"unit": "family", "variations_per_parent_per_epoch": 1,
            "seed_schedule": "shuffled_cycle", "target_pairs": "all_solved",
            "embedding_scope": "parent"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def regular(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in str(relative):
        raise ValueError(f"Unsafe data path: {relative}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError(f"Linked data path: {relative}")
    return target


def lineage_closure(nodes, originals, blocked):
    """Return original ancestors or an error; dependency edges are directional."""
    cache = {}
    def visit(key, active):
        if key in active:
            raise ValueError("cyclic lineage")
        if key in cache:
            return cache[key]
        if key not in nodes:
            if key in originals:
                return {key}
            raise ValueError(f"unknown ancestor {key}")
        deps = nodes[key]
        if deps is None:
            raise ValueError(f"unknown lineage {key}")
        leaves = {key} if key in originals else set()
        for dep in deps:
            if dep == key and key in originals:
                continue
            leaves.update(visit(dep, active | {key}))
        if not leaves:
            raise ValueError(f"empty lineage {key}")
        cache[key] = leaves
        return leaves
    accepted, excluded = {}, {}
    for key in sorted(nodes):
        try:
            leaves = visit(key, set())
            if leaves & blocked:
                excluded[key] = "heldout_or_alias_ancestor"
            else:
                accepted[key] = sorted(leaves)
        except ValueError as error:
            excluded[key] = str(error)
    return accepted, excluded


def normalize_config(raw, root):
    """Validate a family recipe without opening payloads or discovering new data."""
    data = dict(raw["pretrain_data"])
    unknown = set(data) - {"kind", "mode", "sources", "sampling", "discovery", "catalog"}
    if unknown:
        raise ValueError(f"Unknown family configuration keys: {sorted(unknown)}")
    if data.get("mode") not in {"development", "full"}:
        raise ValueError("family_sampled mode must be development or full")
    sources = data.get("sources")
    if (not isinstance(sources, list) or not sources or len(sources) != len(set(sources))
            or any(s not in GROUPS for s in sources)):
        raise ValueError("family_sampled requires unique enabled sources")
    sampling = {**SAMPLING, **data.get("sampling", {})}
    if set(sampling) - (set(SAMPLING) | {"deduplicate_variation_inputs"}):
        raise ValueError("Unknown family sampling keys")
    if any(sampling[k] != v for k, v in SAMPLING.items()):
        raise ValueError("family sampling requires one instance, all solved pairs, parent embeddings and shuffled_cycle")
    if type(sampling.get("deduplicate_variation_inputs", True)) is not bool:
        raise ValueError("deduplicate_variation_inputs must be boolean")
    evaluation = dict(raw.get("evaluation", {"source": "arc2_eval"}))
    if set(evaluation) - {"source", "split_manifest"} or evaluation.get("source") != "arc2_eval":
        raise ValueError("family_sampled uses arc2_eval evaluation")
    if data["mode"] == "development":
        manifest, _ = load_parent_split(root, evaluation.get("split_manifest"))
        if manifest["source"] != "arc2_eval":
            raise ValueError("development split must use arc2_eval")
    elif evaluation.get("split_manifest"):
        raise ValueError("full mode cannot declare held-out public families")
    discovery = data.get("discovery", {})
    if set(discovery) - {"composition_pipelines", "accepted_states", "composition_release"} or not {"composition_pipelines", "accepted_states"} <= set(discovery):
        raise ValueError("discovery must explicitly name composition_pipelines and accepted_states")
    for key, choices in (("composition_pipelines", {"v2", "v3"}),
                         ("accepted_states", {"completed", "author_completed"})):
        value = discovery[key]
        if not isinstance(value, list) or not value or len(value) != len(set(value)) or set(value) - choices:
            raise ValueError(f"Invalid discovery.{key}")
    if "composition_release" in discovery:
        validate_identity(discovery["composition_release"])
    if "catalog" in data:
        identity = data["catalog"]
        if set(identity) != {"archive_path", "size", "sha256"}:
            raise ValueError("catalog requires archive_path, size and sha256")
        path = Path(identity["archive_path"])
        if (path.is_absolute() or ".." in path.parts or "\\" in str(path) or not path.parts
                or path.parts[0] != "assets_tmp" or path.suffix != ".zip"
                or type(identity["size"]) is not int or identity["size"] < 1
                or not re.fullmatch("[0-9a-f]{64}", identity["sha256"])):
            raise ValueError("Invalid pinned family catalog")
    if raw.get("style", "pretrain") not in {"pretrain", "finetune"} or raw.get("dataset", "arc2") != "arc2":
        raise ValueError("family_sampled requires arc2 and pretrain/finetune style")
    if any(raw.get(k) not in (None, False) for k in ("prefinesub", "add_conceptarc", "add_filtered", "add_rearc", "add_nvarc_training", "add_nvarc_full", "keep_train_test_separate")):
        raise ValueError("Use family sources for additions")
    data.update(sampling=sampling, eval_support_repeats_per_epoch=0)
    return {"dataset": "arc2", "style": raw.get("style", "pretrain"), "evaluation": evaluation, "pretrain_data": data,
            "add_conceptarc": False, "add_filtered": False, "add_rearc": None,
            "add_nvarc_training": False, "add_nvarc_full": False, "prefinesub": False}


def discover(root, synth, config, *, recovery=None):
    """Fresh, read-only metadata inventory. Missing payloads are listed, never omitted."""
    root, synth = Path(root).resolve(), Path(synth).resolve()
    settings = config["pretrain_data"]
    pins, candidates, nodes, issues = {}, {}, {}, {}
    missing_evidence, corrupt_evidence = [], []
    def read_payload(base, relative, digest):
        path = regular(base, relative)
        if path.is_file():
            return path.read_bytes()
        if recovery is not None:
            raw = recovery.read(digest)
            if raw is not None:
                return raw
        raise FileNotFoundError(path)
    def load(base, rel):
        path = regular(base, rel)
        raw = path.read_bytes()
        pins[("repo:" if base == root else "synth:") + str(rel)] = sha(raw)
        return json.loads(raw)
    release = (load_release(synth, settings["discovery"]["composition_release"], pins)
               if "composition_release" in settings["discovery"] else None)
    all_tasks = {}
    source_hashes = {}
    for source, filename in ORIGINAL_FILES.items():
        tasks = load(root, "assets_tmp/" + filename)
        source_hashes[source] = content_sha256(tasks)
        all_tasks.update({f"{source}/{native}": task for native, task in tasks.items()})
    manifest = split_identity = None
    if settings["mode"] == "development":
        manifest, split_identity = load_parent_split(root, config["evaluation"]["split_manifest"])
        validate_split_tasks(manifest, {k.split("/", 1)[1]: v for k, v in all_tasks.items() if k.startswith("arc2_eval/")})
    heldout = {f"arc2_eval/{tid}" for ids in (manifest["eval_groups"].values() if manifest else []) for tid in ids}
    pair_owners = defaultdict(set)
    for family, task in all_tasks.items():
        for pair in _solved_pairs(task, family):
            pair_owners[content_sha256(pair)].add(family)
    # Exact pair aliases close forward from heldout, including chains through other sources.
    blocked = set(heldout)
    changed = True
    while changed:
        previous = set(blocked)
        for owners in pair_owners.values():
            if owners & blocked:
                blocked.update(owners)
        changed = previous != blocked
    protected_pairs = {digest for digest, owners in pair_owners.items() if owners & blocked}
    def qualify(value, source):
        if value in all_tasks or value.startswith("composition/"):
            return value
        # Mixed-split plans save native qualified IDs, rather than bare IDs in
        # the plan's single dataset. Resolve only explicit known namespaces.
        for namespace, group in GROUPS.items():
            if value.startswith(group + "/"):
                return namespace + "/" + value[len(group) + 1:]
        return f"{source}/{value}"
    def candidate(base, relative, digest, seed, kind):
        if not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Missing/invalid sample hash")
        return {"origin": "repo" if base == root else "synth", "relative": str(relative),
                "sha256": digest, "seed": seed, "kind": kind}
    # Context provenance is attached to the parent family; compositions inherit it recursively.
    for family, task in sorted(all_tasks.items()):
        source, native = family.split("/", 1)
        job = Path("results/task-variations") / GROUPS[source] / native
        mp = job / "samples/manifest"
        nodes[family] = []
        if not (synth / mp).is_file():
            if source in settings["sources"]:
                issues[family] = "missing variation manifest"
            nodes[family] = None
            continue
        m = load(synth, mp)
        generator = m.get('generator', {})
        generator_path = regular(synth, job / generator.get('path', 'MISSING'))
        # The historical GeneratorRun contract hashes code after rstrip(),
        # whereas protected/file manifests hash exact stored bytes.
        if not generator_path.is_file() or generator.get('sha256') not in {
                sha(generator_path.read_bytes()), sha(generator_path.read_text().rstrip().encode())}:
            issues[family] = 'unverified variation generator pin'
        if generator_path.is_file():
            pins['synth:' + str(job / generator['path'])] = sha(generator_path.read_bytes())
        task_snapshot = regular(synth, job / 'task.json')
        if not task_snapshot.is_file() or load(synth, job / 'task.json') != task:
            issues[family] = 'variation original source changed'
        samples = m.get("file_sha256", {})
        paths = m.get("task_paths", [])
        if not samples or set(paths) != set(samples) or m.get("state") != "complete":
            issues[family] = "incomplete variation manifest"
        deps, known = set(), False
        for p in sorted((synth / job).glob("llm-inputs/*/recorded-inputs.json")):
            record = load(synth, p.relative_to(synth))
            known = True
            for demo in record.get("demonstrations", []):
                ds = "arc2_eval" if demo.get("source_split") == "evaluation" else "arc2_train"
                dep = qualify(demo["id"], ds)
                deps.add(dep)
                if dep not in all_tasks or (demo.get("task_sha256") and sha(encoded(all_tasks[dep])) != demo["task_sha256"]
                                            and content_sha256(all_tasks[dep]) != demo["task_sha256"]):
                    # Historical task hashes use original serialization; verify their source when present.
                    ds, tid = dep.split("/", 1)
                    snapshot = Path('results/task-variations') / GROUPS.get(ds, 'unknown') / tid / 'task.json'
                    raw_path = regular(synth, snapshot)
                    if (not raw_path.is_file() or sha(raw_path.read_bytes()) != demo.get("task_sha256")
                            or load(synth, snapshot) != all_tasks.get(dep)):
                        issues[family] = "unverified authoring task hash"
        for p in sorted((synth / job).glob("llm-inputs/*/workspace/environment.json")):
            env = load(synth, p.relative_to(synth))
            known = True
            deps.update(qualify(t, "arc2_eval") for t in env.get("examples", []))
        nodes[family] = sorted({"original/" + family} | {"original/" + d for d in deps}) if known else None
        if source not in settings["sources"]:
            continue
        original = encoded(task)
        candidates[family] = [{"origin": "original", "family": family, "sha256": sha(original),
                               "seed": None, "kind": "original", "pair_count": len(_solved_pairs(task, family))}]
        seen = set()
        for rel, digest in sorted(samples.items()):
            match = re.fullmatch(r"seed-(\d+)\.json", rel)
            if not match or int(match[1]) in seen:
                raise ValueError(f"Invalid variation seed membership: {family}")
            seed = int(match[1]); seen.add(seed)
            candidates[family].append(candidate(synth, job / "samples" / rel, digest, seed, "variation"))
        if sorted(m.get("completed_seeds", [])) != sorted(seen) or len(seen) != m.get("sampling", {}).get("count"):
            issues[family] = "variation seed membership differs from manifest"
    policy = settings["discovery"]
    roots = []
    for pipeline in policy["composition_pipelines"]:
        patterns = ["results/task-composition-v3*"] if pipeline == "v3" else ["results/task-compsition-v2*", "results/task-composition-v2*"]
        for pattern in patterns:
            roots.extend(sorted(synth.glob(pattern)))
    states = {}
    for directory in sorted(set(roots)):
        if not directory.is_dir() or directory.is_symlink() or not (directory / "plan.json").is_file():
            continue
        plan = load(synth, (directory / "plan.json").relative_to(synth))
        dataset = plan.get("dataset", "arc2/evaluation")
        source = next((s for s, group in GROUPS.items() if group == dataset),
                      "arc2_mixed" if dataset == "arc2/mixed" else None)
        planned = {j["name"]: j for j in plan.get("jobs", [])}
        for sp in sorted(directory.glob("*/status.json")):
            job = sp.parent
            status = load(synth, sp.relative_to(synth))
            state = status.get("state")
            family = f"composition/{directory.name}/{source}/{job.name}"
            states[family] = state
            nodes[family] = None
            if release is not None and family not in release:
                continue
            if state not in policy["accepted_states"]:
                continue
            if source == "arc2_mixed":
                if not set(settings["sources"]) & {"arc2_train", "arc2_eval"}:
                    continue
                # A disabled original source cannot enter via a mixed family.
                # Unresolved parents proceed to lineage quarantine, never a guess.
                if any(parent in all_tasks and parent.split("/", 1)[0] not in settings["sources"]
                       for parent in (qualify(p, source) for p in status.get("parents", []))):
                    continue
            elif source not in settings["sources"]:
                continue
            if job.name not in planned or source is None or status.get("parents") != planned[job.name].get("parents") or status.get("dataset", dataset) != dataset:
                issues[family] = "unknown or conflicting composition lineage"
                continue
            stage = ("review-improve" if directory.name.startswith("task-composition-v3") else "improve") if state == "completed" else "author"
            prefix = status.get("samples", "").rstrip("/") + "/" if state == "completed" else "llm-inputs/author/checked-samples/"
            cp = job / "checkpoints.json"
            if not cp.is_file():
                issues[family] = "missing composition checkpoint manifest"
                continue
            checkpoint = load(synth, cp.relative_to(synth))
            sample_pins = {name: h for name, h in checkpoint.get(stage, {}).items()
                           if name.startswith(prefix) and re.fullmatch(r"seed-\d+\.json", name[len(prefix):])}
            deps = {qualify(t, source) for t in status["parents"]}
            context_known = True
            for stage_name in (["author", stage] if stage != "author" else [stage]):
                pd = job / "llm-inputs" / stage_name / "protected.json"
                if not pd.is_file():
                    context_known = False
                    continue
                protected = load(synth, pd.relative_to(synth))
                if not protected or 'environment.json' not in protected:
                    context_known = False
                for label, parent in zip(('A', 'B'), status['parents']):
                    snapshot_name = f'parents/{label}/task.json'
                    snapshot_path = regular(pd.parent / 'inputs', snapshot_name)
                    parent_key = qualify(parent, source)
                    if snapshot_name not in protected:
                        context_known = False
                    elif snapshot_path.is_file() and parent_key in all_tasks and json.loads(snapshot_path.read_bytes()) != all_tasks[parent_key]:
                        context_known = False
                for name, digest in protected.items():
                    if name.startswith("examples/"):
                        parts = Path(name).parts
                        if len(parts) >= 3:
                            deps.add(qualify(parts[1], "arc2_eval"))
                        elif name not in {'examples/README.md', 'examples/manifest.json'}:
                            context_known = False
                    if name == "environment.json" or name.startswith(("parents/", "examples/")):
                        path = regular(pd.parent / "inputs", name)
                        record = {'family': family, 'path': str(path.relative_to(synth)), 'sha256': digest}
                        try:
                            raw = read_payload(synth, path.relative_to(synth), digest)
                        except FileNotFoundError:
                            missing_evidence.append(record)
                            continue
                        if sha(raw) != digest:
                            corrupt_evidence.append(record)
                            context_known = False
                        elif name in {'parents/A/task.json', 'parents/B/task.json'}:
                            parent_key = qualify(status['parents'][0 if name.startswith('parents/A/') else 1], source)
                            if parent_key in all_tasks and json.loads(raw) != all_tasks[parent_key]:
                                context_known = False
                        elif name == 'environment.json':
                            env = json.loads(raw)
                            deps.update(qualify(t, 'arc2_eval') for t in env.get('examples', []))
                        elif name == 'examples/manifest.json':
                            examples = json.loads(raw)
                            deps.update(qualify(t, 'arc2_eval') for t in examples.get('example_ids', []))
                if any(k.startswith("examples/") for k in protected):
                    deps.update(qualify(t, "arc2_eval") for t in planned[job.name].get("review_examples", []))
            # Nontrivial/new pipelines must provide an explicit complete dependency list.
            lp = job / "lineage.json"
            if lp.is_file():
                lineage = load(synth, lp.relative_to(synth))
                if lineage.get("complete") is not True or not isinstance(lineage.get("ancestors"), list):
                    context_known = False
                else:
                    deps.update(lineage["ancestors"])
            nodes[family] = sorted(deps) if context_known else None
            candidates[family] = []
            for rel, digest in sorted(sample_pins.items()):
                item = candidate(synth, (job / rel).relative_to(synth), digest, int(re.search(r"seed-(\d+)", rel)[1]), "composition")
                item["stage"] = stage
                candidates[family].append(item)
            if not sample_pins:
                issues[family] = "no saved composition sample hashes"
            expected_count = status.get('sample_count' if state == 'completed' else 'check_count')
            if expected_count is not None and (type(expected_count) is not int or expected_count != len(sample_pins)):
                issues[family] = 'composition sample membership differs from saved status'
            if release is not None:
                frozen = release[family]['source']
                if frozen['stage'] != stage or frozen['parents'] != status['parents'] or frozen['dataset'] != dataset:
                    raise ValueError('Composition release provenance differs from source: ' + family)
                candidates[family] = release[family]['instances']
    if release is not None and set(release) - set(states):
        raise ValueError('Frozen composition families absent from source inventory')
    for family in issues:
        nodes[family] = None
    accepted, excluded = lineage_closure(nodes, {"original/" + k for k in all_tasks}, {"original/" + k for k in blocked})
    accepted = {k: [v.removeprefix("original/") for v in values] for k, values in accepted.items()}
    selected = {k: v for k, v in sorted(candidates.items()) if k in accepted}
    # Scan every present selected payload. One matching heldout pair taints the entire family and descendants.
    missing, corrupt, duplicates = [], [], set()
    def payload(item):
        if item["origin"] == "original":
            return encoded(all_tasks[item["family"]])
        if "archive_member" in item:
            return puzzle_bytes(synth, item)
        return read_payload(root if item["origin"] == "repo" else synth, item["relative"], item['sha256'])
    def verify_metadata():
        for key, expected in pins.items():
            origin, relative = key.split(':', 1)
            if file_sha(regular(root if origin == 'repo' else synth, relative)) != expected:
                raise ValueError(f'Metadata changed during preparation: {key}; build a fresh inventory')
    payload.verify_metadata = verify_metadata
    for family, instances in selected.items():
        for item in instances:
            label = {"family": family, "sha256": item["sha256"], "path": item.get("relative", family), "kind": item["kind"]}
            try:
                data = payload(item)
            except FileNotFoundError:
                missing.append(label)
                continue
            if sha(data) != item["sha256"]:
                corrupt.append(label)
                continue
            pairs = _solved_pairs(json.loads(data), label["path"])
            item["pair_count"] = len(pairs)
            if any(content_sha256(pair) in protected_pairs for pair in pairs):
                duplicates.add(family)
    if duplicates:
        accepted, excluded = lineage_closure(nodes, {"original/" + k for k in all_tasks}, {"original/" + k for k in blocked})
        accepted = {k: [v.removeprefix("original/") for v in values] for k, values in accepted.items()}
        # A generated node in blocked must also be excluded itself.
        tainted = set(duplicates)
        while True:
            nxt = tainted | {k for k, deps in nodes.items() if deps and set(deps) & tainted}
            if nxt == tainted: break
            tainted = nxt
        for k in tainted:
            accepted.pop(k, None); excluded[k] = "heldout_pair_fingerprint"
        selected = {k: v for k, v in selected.items() if k in accepted}
        missing = [x for x in missing if x["family"] in selected]
        corrupt = [x for x in corrupt if x["family"] in selected]
    missing_evidence = [v for v in missing_evidence if v['family'] in selected]
    # Corrupt lineage in a stage-eligible family blocks packaging even if it was quarantined.
    corrupt_evidence = [v for v in corrupt_evidence if v['family'] in candidates]
    catalog = {"schema": SCHEMA, "mode": settings["mode"], "sources": settings["sources"],
               "sampling": settings["sampling"], "discovery": policy, "source_sha256": source_hashes,
               "parent_split": split_identity, "heldout": sorted(heldout), "aliases_excluded": sorted(blocked - heldout),
               "families": {k: {"ancestors": accepted[k], "instances": v,
                   **({"sampling_omissions": release[k]['omissions']}
                      if release is not None and k in release and release[k]['omissions'] else {})}
                   for k, v in selected.items()},
               "excluded": {k: v for k, v in excluded.items() if k in candidates or k in issues},
               "issues": issues, "metadata_sha256": pins}
    summary = {"discovered_original_families": dict(Counter(k.split('/')[0] for k in all_tasks)),
               "discovered_composition_states": dict(Counter(states.values())),
               "eligible_original_families": dict(Counter(k.split('/')[0] for k in selected if not k.startswith('composition/'))),
               "eligible_composition_families": sum(k.startswith('composition/') for k in selected),
               "eligible_composition_sources": dict(Counter(k.split('/')[2] for k in selected if k.startswith('composition/'))),
               "eligible_composition_stages": dict(Counter(v[0]['stage'] for k, v in selected.items() if k.startswith('composition/') and v)),
               "excluded_composition_families": sum(k.startswith('composition/') for k in catalog['excluded']),
               "excluded_reasons": dict(Counter(v.split(' ', 1)[0] if v.startswith(('unknown ', 'cyclic ', 'empty ')) else v for v in catalog['excluded'].values())),
               "instances_per_epoch": len(selected), "candidate_instances": sum(map(len, selected.values())),
               "missing_payloads": dict(Counter(x['kind'] for x in missing)), "corrupt_payloads": len(corrupt),
               "missing_lineage_payloads": len(missing_evidence), "corrupt_lineage_payloads": len(corrupt_evidence),
               "saved_composition_instances": sum(len(v) for k, v in selected.items() if k.startswith('composition/')),
               "blocking_metadata_issues": {k: excluded.get(k, issues.get(k, "missing family")) for k in all_tasks
                   if k.split("/")[0] in settings["sources"] and k not in blocked and k not in selected},
               "blocking_composition_metadata_issues": {k: v for k, v in issues.items() if k.startswith('composition/')},
               "ready": False}
    summary["ready"] = not (missing or corrupt or missing_evidence or corrupt_evidence or
                            summary["blocking_metadata_issues"] or summary["blocking_composition_metadata_issues"])
    catalog.update(summary=summary, missing=missing, corrupt=corrupt,
                   missing_lineage_payloads=missing_evidence, corrupt_lineage_payloads=corrupt_evidence)
    return catalog, payload


def package_catalog(catalog, payload, destination):
    """Stream an immutable package only after the complete selected corpus passes."""
    if not catalog['summary']['ready']:
        raise ValueError("Full corpus is not ready; recover missing payloads/lineage, then build a fresh inventory")
    payload.verify_metadata()
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Catalog package exists; use a fresh path to refresh inventory")
    packaged = json.loads(json.dumps(catalog))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.families-') as temp:
        pending = Path(temp) / 'package.zip'
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            def write_member(name, raw):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, raw)
            for family, record in packaged['families'].items():
                for item in record['instances']:
                    raw = payload(item)
                    if sha(raw) != item['sha256']:
                        raise ValueError('Payload changed while packaging')
                    name = f"instances/{item['sha256']}.json"
                    if name not in archive.NameToInfo:
                        write_member(name, raw)
                    item['path'] = name
            write_member('family_catalog.json', encoded(packaged))
        payload.verify_metadata()
        pending.rename(destination)
    return {"archive_path": destination.as_posix(), "size": destination.stat().st_size,
            "sha256": file_sha(destination)}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_family_dataset(config, output_dir, *, root):
    """Use only the pinned package. An existing run inventory cannot be refreshed."""
    root, output_dir = Path(root), Path(output_dir)
    settings = config['pretrain_data']
    identity = settings.get('catalog')
    if identity is None:
        raise ValueError('Prepare and pin a complete family catalog before training')
    archive_path = regular(root, identity['archive_path'])
    if archive_path.stat().st_size != identity['size'] or file_sha(archive_path) != identity['sha256']:
        raise ValueError('Family catalog package identity changed')
    # Thousands of tiny files are prohibitively slow on pod network volumes.
    # Payloads are disposable: the pinned ZIP and run inventory stay persistent.
    cache = (Path(tempfile.gettempdir()) / 'mdlarc-family-cache' / identity['sha256']
             if os.environ.get('MDLARC_REMOTE') == '1' else None)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    marker = cache / '.complete.json' if cache else None
    cache_ready = False
    if marker and marker.is_file() and (cache / 'instances').is_dir():
        try:
            cache_ready = json.loads(marker.read_text()) == {'schema_version': 1, 'package': identity}
        except (ValueError, OSError):
            pass
    if cache:
        print(f'{"Reusing" if cache_ready else "Preparing"} pinned family payloads on local disk: {cache}', flush=True)
    old_index = output_dir / 'variation_index.json'
    if old_index.exists():
        old = json.loads(old_index.read_text())
        if old.get('family_catalog') != identity:
            raise ValueError('Immutable run inventory cannot be refreshed; select a new run')
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if len(archive.namelist()) != len(names):
            raise ValueError('Duplicate family package members')
        catalog = json.loads(archive.read('family_catalog.json'))
        if catalog.get('schema') != SCHEMA or not catalog['summary']['ready']:
            raise ValueError('Incomplete or unsupported family catalog')
        for key in ('mode', 'sources', 'sampling', 'discovery'):
            if catalog[key] != settings[key]:
                raise ValueError(f'Pinned family catalog {key} differs from spec')
        manifest = identity_split = None
        evaluation = config['evaluation']
        if settings['mode'] == 'development':
            manifest, identity_split = load_parent_split(root, evaluation['split_manifest'])
        if catalog['parent_split'] != identity_split:
            raise ValueError('Family catalog split changed')
        if old_index.exists():
            if file_sha(output_dir / 'family_catalog.json') != sha(encoded(catalog)):
                raise ValueError('Run family catalog changed')
            if cache is None or cache_ready:
                return {'reused_immutable_catalog': True}
        challenges, solutions = {}, {}
        # Public answers are only separated out for development validation.
        if manifest:
            originals = json.loads((root / 'assets_tmp' / ORIGINAL_FILES['arc2_eval']).read_text())
            validate_split_tasks(manifest, originals)
            for native in [t for ids in manifest['eval_groups'].values() for t in ids]:
                task = originals[native]; key = 'arc2_eval/' + native
                challenges[key] = {'train': task['train'], 'test': [{'input': p['input']} for p in task['test']]}
                solutions[key] = [p['output'] for p in task['test']]
        parents = {}
        for family, record in catalog['families'].items():
            parents[family] = []
            if set(record['ancestors']) & set(catalog['heldout'] + catalog['aliases_excluded']):
                raise ValueError('Heldout ancestor in family package')
            for item in record['instances']:
                name = f"instances/{item['sha256']}.json"
                if not re.fullmatch('[0-9a-f]{64}', item['sha256']) or item.get('path') != name or name not in names:
                    raise ValueError('Invalid family member path')
                relative = (cache or output_dir) / name
                if not cache_ready:
                    raw = archive.read(name)
                    if sha(raw) != item['sha256']:
                        raise ValueError('Family sample hash changed')
                    # The pinned package carries audited counts. Training
                    # verifies each selected cached payload's SHA-256 too.
                    if cache is None and len(_solved_pairs(json.loads(raw), name)) != item['pair_count']:
                        raise ValueError('Family sample pair count changed')
                    if not relative.resolve().is_relative_to((cache or root).resolve()):
                        raise ValueError('Run data must be inside repository')
                    if relative.exists():
                        if relative.read_bytes() != raw:
                            raise ValueError('Run sample changed')
                    else:
                        relative.parent.mkdir(parents=True, exist_ok=True)
                        relative.write_bytes(raw)
                parents[family].append({**item, 'path': str(relative) if cache else relative.relative_to(root).as_posix()})
    if marker and not cache_ready:
        pending = marker.with_name(f'.complete-{os.getpid()}.json')
        pending.write_text(json.dumps({'schema_version': 1, 'package': identity}, sort_keys=True))
        pending.replace(marker)
    index = {'snapshot': 'task-variations-v1', 'archive_sha256': identity['sha256'],
             'sources': settings['sources'], 'original_sources': settings['sources'],
             'sampling': settings['sampling'], 'eval_support_repeats_per_epoch': 0,
             'eval_support_task_ids': [], 'parents': parents,
             'composition_parents': [k for k in parents if k.startswith('composition/')],
             'family_catalog': identity, 'catalog_sha256': sha(encoded(catalog)),
             'evaluation_source': 'arc2_eval', 'data_mode': settings['mode']}
    if identity_split:
        index['parent_split'] = identity_split
    groups = {g: ['arc2_eval/' + t for t in ids] for g, ids in manifest['eval_groups'].items()} if manifest else {}
    _write_outputs(output_dir, {'challenges.json': {k: {'train': []} for k in parents},
                               'priv_challenges.json': challenges, 'solutions.json': solutions,
                               'variation_index.json': index, 'family_catalog.json': catalog}, groups,
                   (root / identity_split['manifest']).read_bytes() if identity_split else None)
    return catalog['summary']
