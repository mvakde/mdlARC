"""Shared-coating optical switches; all inference uses only the queried pixels.

Standard library and NumPy only. Generation and self-tests perform no file I/O.
No execution traces, sample caches, or seed-indexed answer tables are retained.
"""

from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,638447458,242567606,951868303,808143490,296856558,289100408,
976650094,825138967,685105501,965223558,601619498,834986408,705749885,
332927863,405180339,176496537,959360240,648126930,692961532,7212001,
300884838,427782416,783722840,382845967,131304984,782212717,97264057,
353260178,492885610,181301449,474407703,50799300,132077895,33581113,
957278290,467539386,665515311,931962051,385194803,787680526,453625988)
CARDINAL = ((-1,0),(1,0),(0,-1),(0,1))
DIAGONAL = ((-1,-1),(-1,1),(1,-1),(1,1))


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r,c))
            for dr, dc in CARDINAL:
                p = r+dr,c+dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(frozenset(cells))
    return result


def _source(elbow, direction, color):
    r, c = elbow
    dr, dc = direction
    return {"elbow": elbow, "direction": direction, "color": int(color),
            "cells": frozenset(((r,c),(r-dr,c),(r,c-dc)))}


def _port(bar, side, slot, length):
    labels = bar["labels"]
    a = tuple(map(int, np.argwhere(labels == side*length+slot)[0]))
    b = tuple(map(int, np.argwhere(labels == (1-side)*length+slot)[0]))
    normal = a[0]-b[0],a[1]-b[1]
    surface = bar["r"]+a[0],bar["c"]+a[1]
    return (surface[0]+normal[0],surface[1]+normal[1]),normal


def _bar(template, t, row, col):
    arr = _orient(template, t)
    return {"r": row, "c": col, "arr": arr, "t": t,
            "labels": _orient(np.arange(template.size).reshape(template.shape), t),
            "cells": frozenset((row+r,col+c) for r in range(arr.shape[0])
                               for c in range(arr.shape[1]))}


def _parse(a):
    values, counts = np.unique(a, return_counts=True)
    assert counts.max() > a.size/2, "background must be a strict majority"
    bg = int(values[np.argmax(counts)])
    bars, sources, clues = [], [], []
    for cells in _components(a, bg):
        rows, cols = zip(*cells)
        r, c = min(rows),min(cols)
        arr = a[r:max(rows)+1,c:max(cols)+1]
        if len(cells) == 1:
            p = next(iter(cells))
            assert p[0] in (0,a.shape[0]-1) or p[1] in (0,a.shape[1]-1)
            clues.append((p,int(a[p])))
        elif len(cells) == 3 and arr.shape == (2,2):
            assert len({int(a[p]) for p in cells}) == 1
            missing = next(iter(set((r+i,c+j) for i in range(2) for j in range(2))-cells))
            elbow = 2*r+1-missing[0],2*c+1-missing[1]
            sources.append(_source(elbow, (elbow[0]-missing[0],elbow[1]-missing[1]), a[elbow]))
        else:
            assert arr.size == len(cells) and min(arr.shape) == 2
            assert 4 <= max(arr.shape) <= 9 and 2 <= len(np.unique(arr)) <= 4
            bars.append({"r": r,"c": c,"arr": arr.copy(),"cells": cells})
    assert 2 <= len(bars) <= 4 and 1 <= len(sources) <= 5 and clues
    template = bars[0]["arr"]
    if template.shape[0] != 2:
        template = template.T.copy()
    length = template.shape[1]
    forms = [_orient(template,t) for t in range(8)]
    assert len({_key(x) for x in forms}) == 8, "template correspondence must be unique"
    structural = set(map(int,template.flat))
    for bar in bars:
        matches = [t for t in range(8) if np.array_equal(forms[t],bar["arr"])]
        assert len(matches) == 1
        bar["t"] = matches[0]
        bar["labels"] = _orient(np.arange(2*length).reshape(2,length),matches[0])
    assert all(color not in structural and color != bg for _,color in clues)
    occupied = set().union(*(b["cells"] for b in bars),*(s["cells"] for s in sources))
    alignments = _alignments(bars,length,a.shape,occupied)
    code, sides, observations = {},set(),[]
    for p, color in clues:
        matches = alignments.get(p,[])
        assert len(matches) == 1, "ambiguous calibration mark"
        index, side, slot = matches[0]
        assert slot not in code or code[slot] == color, "inconsistent coating observations"
        code[slot] = color
        sides.add(side)
        observations.append((index,side,slot,p,color))
    assert len(sides) == 1 and 1 <= len(code) <= min(4,length-1)
    return bg,bars,sources,code,sides.pop(),observations,length


def _alignments(bars, length, shape, occupied):
    alignments = {}
    for index, bar in enumerate(bars):
        for side in (0,1):
            for slot in range(length):
                p, normal = _port(bar,side,slot,length)
                ray = []
                while _inside(p,shape):
                    ray.append(p)
                    p = p[0]+normal[0],p[1]+normal[1]
                if len(ray) >= 2 and not occupied.intersection(ray):
                    alignments.setdefault(ray[-1],[]).append((index,side,slot))
    return alignments


def _trace(a, bg, bars, sources, code, side, length, *, policy="normal", observations=()):
    """Simulate only L-launched light; code inference is a separate input step.

    Diagnostic policies deliberately omit a mechanism and are never used by solve.
    """
    contacts_at = {}
    for i, bar in enumerate(bars):
        for face in (0,1):
            for slot in range(length):
                p, normal = _port(bar,face,slot,length)
                contacts_at.setdefault(p,[]).append((i,face,slot,normal))
    occupied = set().union(*(b["cells"] for b in bars),*(s["cells"] for s in sources))
    local = {(i,face,slot): color for i,face,slot,_,color in observations}
    output, painted, routes = a.copy(),{},[]
    for source in sources:
        dr, dc = source["direction"]
        r, c = source["elbow"]
        r, c = r+dr,c+dc
        color = source["color"]
        states, hits, path, stop = set(),[],[],False
        while _inside((r,c),a.shape):
            state = r,c,dr,dc
            assert state not in states, "looping optical network"
            states.add(state)
            p = r,c
            assert p not in occupied, "corner or short-face collision"
            candidates = [x for x in contacts_at.get(p,())
                          if x[3][0]*dr+x[3][1]*dc == -1]
            assert len(candidates) <= 1, "simultaneous mirror contacts"
            stop = False
            if candidates:
                i, face, slot, normal = candidates[0]
                active = face == side and slot in code
                new_color = code.get(slot,color)
                if policy == "local":
                    active = (i,face,slot) in local
                    new_color = local.get((i,face,slot),color)
                if policy == "all_ports":
                    active, new_color = True,color
                if policy == "no_reflection":
                    active = False
                if active:
                    color = int(new_color)
                    dr = -dr if normal[0] else dr
                    dc = -dc if normal[1] else dc
                else:
                    stop = True
                hits.append((i,face,slot,p,active,color,normal))
            assert int(a[p]) in (bg,color), "beam would overwrite a calibration mark"
            assert p not in painted or painted[p] == color, "conflicting light colors"
            painted[p] = color
            output[p] = color
            path.append(p)
            if stop:
                break
            r, c = r+dr,c+dc
        assert len(path) >= 2, "unreadably short beam"
        routes.append({"hits": hits,"path": tuple(path),"end": "absorbed" if stop else "edge"})
    return output,routes


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    bg,bars,sources,code,side,observations,length = _parse(a)
    return _trace(a,bg,bars,sources,code,side,length,observations=observations)[0]


def _separate(cells, occupied, shape):
    return (all(_inside(p,shape) for p in cells) and not cells.intersection(occupied)
            and all((r+dr,c+dc) not in occupied for r,c in cells for dr,dc in CARDINAL))


def _paint(shape, bg, bars, sources):
    a = np.full(shape,bg,dtype=np.int8)
    for bar in bars:
        r, c = bar["r"],bar["c"]
        h, w = bar["arr"].shape
        a[r:r+h,c:c+w] = bar["arr"]
    for source in sources:
        for p in source["cells"]:
            a[p] = source["color"]
    return a


def _template(rng, length, colors):
    for _ in range(40):
        a = rng.choice(colors,size=(2,length)).astype(np.int8)
        style = int(rng.integers(4))
        if style == 1:
            a[0] = colors[0]
        elif style == 2:
            a[0] = colors[0]
            a[1] = colors[1]
            a[:,0] = colors[-1]
            a[1,-1] = colors[0]
        elif style == 3:
            a[:,:length//2] = colors[0]
        for i, color in enumerate(colors):
            a[1,i] = color
        if len({_key(_orient(a,t)) for t in range(8)}) == 8:
            return a
    a = np.full((2,length),colors[0],dtype=np.int8)
    a[1,1:] = colors[1]
    for i,color in enumerate(colors[2:],2):
        a[1,i] = color
    return a


def _place_core(rng, template, shape, bg, code, side, source_color, count, demo):
    length = template.shape[1]
    elbow = int(rng.integers(1,shape[0]-1)),int(rng.integers(1,shape[1]-1))
    direction = DIAGONAL[int(rng.integers(4))]
    source = _source(elbow,direction,source_color)
    bars, sources, wanted = [],[source],[]
    anchor = elbow
    active_slots = list(code)
    rng.shuffle(active_slots)
    for index in range(count):
        last = index == count-1
        absorb = last and (demo == 0 or (demo is None and rng.random() < .65))
        face = 1-side if demo == 0 and last else side
        if absorb and face == side:
            slots = [s for s in range(length) if s not in code]
        else:
            slots = active_slots
        options = []
        for t in range(8):
            if demo == 0 and index == 1 and t != (bars[0]["t"] ^ 4):
                continue
            if demo == 1 and index == 1:
                if t//4 != bars[0]["t"]//4 or t % 2 == bars[0]["t"] % 2:
                    continue
            dummy = _bar(template,t,0,0)
            _, normal = _port(dummy,face,0,length)
            if normal[0]*direction[0]+normal[1]*direction[1] != -1:
                continue
            for slot in slots:
                if demo == 1 and index == 1 and slot == wanted[0][2]:
                    continue
                for distance in range(3,12):
                    options.append((t,slot,distance,normal))
        rng.shuffle(options)
        occupied = set().union(source["cells"],*(b["cells"] for b in bars))
        accepted = False
        for t,slot,distance,normal in options[:180]:
            contact = anchor[0]+distance*direction[0],anchor[1]+distance*direction[1]
            dummy = _bar(template,t,0,0)
            local_contact,_ = _port(dummy,face,slot,length)
            candidate = _bar(template,t,contact[0]-local_contact[0],contact[1]-local_contact[1])
            if not _separate(candidate["cells"],occupied,shape):
                continue
            trial_bars = bars+[candidate]
            a = _paint(shape,bg,trial_bars,sources)
            try:
                _,routes = _trace(a,bg,trial_bars,sources,code,side,length)
            except AssertionError:
                continue
            expected = wanted+[(index,face,slot,contact,not absorb)]
            actual = [(i,f,s,p,active) for i,f,s,p,active,_,_ in routes[0]["hits"]]
            if actual != expected:
                continue
            bars, wanted, anchor = trial_bars,expected,contact
            if not absorb:
                direction = (-direction[0],direction[1]) if normal[0] else (direction[0],-direction[1])
            accepted = True
            break
        if not accepted:
            return None
    return bars,sources


def _add_source(rng, bars, sources, code, side, length, shape, bg, kind=None):
    occupied = set().union(*(b["cells"] for b in bars),*(s["cells"] for s in sources))
    options = []
    for index,bar in enumerate(bars):
        for face in (0,1):
            for slot in range(length):
                active = face == side and slot in code
                if kind == "active" and not active:
                    continue
                if kind == "dark" and not (face == side and slot not in code):
                    continue
                if kind == "back" and face == side:
                    continue
                contact,normal = _port(bar,face,slot,length)
                for tangent in (-1,1):
                    d = (-normal[0],tangent) if normal[0] else (tangent,-normal[1])
                    for distance in range(2,10):
                        options.append((index,face,slot,contact,d,distance))
    rng.shuffle(options)
    for i,face,slot,p,d,distance in options[:240]:
        elbow = p[0]-distance*d[0],p[1]-distance*d[1]
        color = int(rng.choice([c for c in range(10) if c != bg]))
        source = _source(elbow,d,color)
        if not _separate(source["cells"],occupied,shape):
            continue
        trial = sources+[source]
        a = _paint(shape,bg,bars,trial)
        try:
            _,routes = _trace(a,bg,bars,trial,code,side,length)
        except AssertionError:
            continue
        if routes[-1]["hits"] and routes[-1]["hits"][0][:4] == (i,face,slot,p):
            return trial
    return None


def _clues(rng, a, bars, sources, code, side, length, routes, pooled):
    occupied = set().union(*(b["cells"] for b in bars),*(s["cells"] for s in sources))
    illuminated = set().union(*(set(r["path"]) for r in routes))
    alignments = _alignments(bars,length,a.shape,occupied)
    options = {}
    for slot in code:
        options[slot] = []
        for p,matches in alignments.items():
            if len(matches) != 1:
                continue
            i,face,s = matches[0]
            if face != side or s != slot or p in illuminated:
                continue
            if any(max(abs(p[0]-q[0]),abs(p[1]-q[1])) <= 1 for q in occupied):
                continue
            options[slot].append((i,p))
        rng.shuffle(options[slot])
        if not options[slot]:
            return None
    order = list(code)
    rng.shuffle(order)
    answer, nodes = [],0

    def search(k):
        nonlocal nodes
        nodes += 1
        if nodes > 200:
            return False
        if k == len(order):
            allocations = [sum(i == j for i,_,_ in answer) for j in range(len(bars))]
            if pooled and max(allocations) == len(code):
                return False
            observed = {(i,s) for i,s,_ in answer}
            remote = any(active and (i,s) not in observed for route in routes
                         for i,_,s,_,active,_,_ in route["hits"])
            return remote
        slot = order[k]
        for index,p in options[slot]:
            if any(max(abs(p[0]-q[0]),abs(p[1]-q[1])) <= 1 for _,_,q in answer):
                continue
            answer.append((index,slot,p))
            if search(k+1):
                return True
            answer.pop()
        return False

    if not search(0):
        return None
    result = a.copy()
    for _,slot,p in answer:
        result[p] = code[slot]
    return result


def _make_pair(rng, count, demo=None):
    # Every retry is in memory; at most four strips, five sources, and 900 cells.
    for attempt in range(160):
        length = int(rng.integers(4,10))
        nstruct = int(rng.integers(2,5))
        palette = list(map(int,rng.permutation(10)))
        bg = palette[0]
        structural = palette[1:1+nstruct]
        nslots = 2 if demo is not None else int(rng.integers(1,min(4,length-1)+1))
        slots = sorted(map(int,rng.choice(length,size=nslots,replace=False)))
        signals = palette[1+nstruct:1+nstruct+nslots]
        if demo is None and nslots > 1 and rng.random() < .3:
            signals[-1] = signals[0]
        code = dict(zip(slots,signals))
        side = int(rng.integers(2))
        template = _template(rng,length,structural)
        low = 17 if count == 2 else 23
        shape = int(rng.integers(low,31)),int(rng.integers(low,31))
        if rng.random() < .2:
            shape = shape[0],shape[0]
        source_color = palette[-1]
        core = _place_core(rng,template,shape,bg,code,side,source_color,count,demo)
        if core is None:
            continue
        bars,sources = core
        kinds = ["active","dark"] if demo == 0 else ["dark"] if demo == 1 else []
        success = True
        for kind in kinds:
            result = _add_source(rng,bars,sources,code,side,length,shape,bg,kind)
            if result is None:
                success = False
                break
            sources = result
        if not success:
            continue
        target = int(rng.integers(len(sources),5))
        while len(sources) < target:
            result = _add_source(rng,bars,sources,code,side,length,shape,bg)
            if result is None:
                break
            sources = result
        a = _paint(shape,bg,bars,sources)
        try:
            _,routes = _trace(a,bg,bars,sources,code,side,length)
        except AssertionError:
            continue
        a = _clues(rng,a,bars,sources,code,side,length,routes,pooled=demo is not None)
        if a is None or np.count_nonzero(a == bg) <= a.size/2:
            continue
        try:
            output = solve(a)
            f = _features(a)
        except AssertionError:
            continue
        if (not f["core_chain"] or f["changes"] < 2 or f["remote_hits"] < 1
                or any(value == 0 for value in f["alternatives"].values())):
            continue
        if demo == 0 and not (f["dark_stops"] and f["back_stops"] and f["reflection_classes"] == 2):
            continue
        if demo == 1 and not (f["max_reflections"] >= 2 and f["normal_axes"] == 2 and f["dark_stops"]):
            continue
        return {"input": a,"output": output}
    raise RuntimeError(f"could not construct a clear {count}-strip scene in 160 attempts")


def _features(a):
    bg,bars,sources,code,side,obs,length = _parse(a)
    out,routes = _trace(a,bg,bars,sources,code,side,length,observations=obs)
    observed = {(i,face,s) for i,face,s,_,_ in obs}
    hits = [hit for route in routes for hit in route["hits"]]
    allocations = tuple(sum(x[0] == i for x in obs) for i in range(len(bars)))
    alternatives = {}
    for policy in ("local","all_ports","no_reflection"):
        try:
            alt,_ = _trace(a,bg,bars,sources,code,side,length,policy=policy,observations=obs)
            alternatives[policy] = int(np.count_nonzero(alt != out))
        except AssertionError:
            alternatives[policy] = -1  # Alternative violates a visible scene constraint.
    return {"shape": a.shape,"background": bg,"length": length,"bars": len(bars),
            "sources": len(sources),"slots": tuple(sorted(code)),"colors": tuple(code[s] for s in sorted(code)),
            "structural_colors": len(np.unique(bars[0]["arr"])),
            "transforms": tuple(b["t"] for b in bars),
            "reflection_classes": len({b["t"]//4 for b in bars}),
            "normal_axes": len({int(n[0] == 0) for _,_,_,_,active,_,n in hits if active}),
            "allocations": allocations,"pooled": max(allocations) < len(code),
            "remote_hits": sum(active and (i,face,s) not in observed for i,face,s,_,active,_,_ in hits),
            "dark_stops": sum(face == side and s not in code for _,face,s,_,active,_,_ in hits if not active),
            "back_stops": sum(face != side for _,face,_,_,active,_,_ in hits if not active),
            "max_contacts": max(len(r["hits"]) for r in routes),
            "core_chain": any(len({h[0] for h in r["hits"]}) == len(bars) for r in routes),
            "max_reflections": max(sum(h[4] for h in r["hits"]) for r in routes),
            "active_slots_hit": len({s for _,_,s,_,active,_,_ in hits if active}),
            "alternatives": alternatives,"changes": int(np.count_nonzero(a != out))}


def _count(value, low, high, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train,2,10,"num_train")
    ne = _count(num_test,1,4,"num_test")
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    puzzle,seen = {"train": [],"test": []},set()
    train_max = 2
    for split,count in (("train",nt),("test",ne)):
        for index in range(count):
            demo = index if split == "train" and index < 2 else None
            nbar = 2 if split == "train" else int(rng.integers(3,5))
            if split == "train" and index >= 2 and rng.random() < .35:
                nbar = 3
            if split == "train":
                train_max = max(train_max,nbar)
            else:
                nbar = max(nbar,train_max+1)
            for retry in range(12):
                pair = _make_pair(rng,nbar,demo)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct scene budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(),{"train": [],"test": []}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            a,b = _grid(pair["input"]),_grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            assert np.array_equal(solve(a),b), "incorrect optical completion"
            f = _features(a)
            assert f["changes"] >= 2 and f["remote_hits"] >= 1
            bg = f["background"]
            assert np.array_equal(a[a != bg],b[a != bg]), "original object changed"
            assert set(map(int,b.flat)) <= set(map(int,a.flat))
            assert f["core_chain"], "a route must encounter every strip"
            assert all(x != 0 for x in f["alternatives"].values()), "concept ablation has no effect"
            analyses[split].append(f)
    train = analyses["train"]
    assert any(f["pooled"] and f["reflection_classes"] == 2 for f in train)
    assert any(f["max_reflections"] >= 2 and f["normal_axes"] == 2 for f in train)
    assert any(f["dark_stops"] for f in train) and any(f["back_stops"] for f in train)
    assert any(f["active_slots_hit"] >= 2 for f in train)
    assert min(f["bars"] for f in analyses["test"]) > max(f["bars"] for f in train)


def _equal(p, q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(a[k],b[k])
               for a,b in zip(p[s],q[s]) for k in ("input","output")) for s in ("train","test"))


def self_test():
    """Bounded: stream 42 puzzles; no retained grids beyond the current puzzle."""
    _rule_regressions()
    features, fingerprints, structures = [],set(),set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train","test"):
            for pair in p[split]:
                a = pair["input"]
                f = _features(a)
                features.append(f)
                digest.update(bytes(a.shape)+a.tobytes())
                roles,normalized = {},[]
                for c in a.flat:
                    normalized.append(roles.setdefault(int(c),len(roles)))
                structures.add(hashlib.sha256(bytes(a.shape)+bytes(normalized)).digest())
                # The optical rule is equivariant to every whole-scene isometry.
                t = (int(seed)+len(features)) % 8
                permutation = np.roll(np.arange(10,dtype=np.int8),len(features) % 10)
                assert np.array_equal(solve(_orient(permutation[a],t)),
                                      _orient(permutation[pair["output"]],t))
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42 and len(structures) >= 190
    assert len({f["shape"] for f in features}) >= 50
    assert {f["length"] for f in features} == set(range(4,10))
    assert {f["bars"] for f in features} == {2,3,4}
    assert {len(f["slots"]) for f in features} == {1,2,3,4}
    assert {f["structural_colors"] for f in features} == {2,3,4}
    assert len({f["sources"] for f in features}) >= 3
    assert {f["background"] for f in features} == set(range(10))
    assert any(len(set(f["colors"])) < len(f["colors"]) for f in features)
    assert any(any(b-a == 1 for a,b in zip(f["slots"],f["slots"][1:])) for f in features)
    original = generate_puzzle(123,2,1)
    assert _equal(original,generate_puzzle(123,2,1))
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(4000+10*nt+ne,nt,ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(3.0,1),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in original[s]] for s in ("train","test")}
    pair = bad["test"][0]
    cell = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][cell] = pair["input"][cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("erased trajectory pixel accepted")


def _rule_regressions():
    # Hand-enumerated pixels: remote coating, immediate contact recoloring,
    # reflection, dark-port absorption, and a remote gate changing a later path.
    a = np.zeros((15,17),dtype=np.int8)
    pattern = np.array(((1,1,2,1,1),(2,1,2,2,1)),dtype=np.int8)
    a[7:9,8:13] = pattern
    a[10:12,0:5] = pattern[:,::-1]
    a[2,5] = a[1,5] = a[2,4] = 6
    a[7,2] = a[6,2] = a[7,1] = 7
    a[0,3],a[0,11] = 4,5
    expected = a.copy()
    for p in ((3,6),(4,7),(5,8)):
        expected[p] = 6
    for p in ((6,9),(5,10),(4,11),(3,12),(2,13),(1,14),(0,15)):
        expected[p] = 4
    expected[8,3] = expected[9,4] = 7
    assert np.array_equal(solve(a),expected)
    # Replace the unused slot-3 observation with a slot-0 coating. Its observed
    # copy is not the copy receiving the formerly absorbed second beam.
    b = a.copy()
    b[0,11],b[0,8] = 0,6
    changed = expected.copy()
    changed[0,11],changed[0,8] = 0,6
    for p in ((9,4),(8,5),(7,6),(6,7),(5,8),(4,9),(3,10),(2,11),(1,12),(0,13)):
        changed[p] = 6
    assert np.array_equal(solve(b),changed)
