"""Self-assembling deflectors: rigid fitting changes a causal branching flow.

Only standard-library modules and NumPy are used. No files, caches, parent
modules, generation metadata, or demonstration labels are consulted by solve.
All diagnostics in self_test are bounded in-memory aggregates.
"""
from itertools import combinations
import hashlib
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RUN_SHORTCUTS = ('one_run', 'ignore_singletons', 'no_edge_detour')
HOST_SEEDS = (0,373264459,124619977,48949740,687673688,677424474,543033598,216400958,94376562,801379357,941208807,759242194,328294174,39757530,115914892,127140029,751481712,519046285,465823687,380383117,352715188,168984975,495977220,181771875,143544727,548887237,255983649,135333707,841759766,898571590,65742080,515352508,629426561,217493402,696314244,487716665,420242467,68579113,568639390,904132426,595666833,853000784)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4, "num_test must be an integer in 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in DIR4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(frozenset(component))
    return result


def _bbox(cells):
    return min(r for r, c in cells), max(r for r, c in cells), min(c for r, c in cells), max(c for r, c in cells)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _transform(regions, op):
    """Four rotations, then four reflections; normalize the whole composite."""
    transformed = []
    for region in regions:
        points = []
        for r, c in region:
            if op >= 4:
                c = -c
            for _ in range(op % 4):
                r, c = c, -r
            points.append((r, c))
        transformed.append(points)
    r0 = min(r for region in transformed for r, c in region)
    c0 = min(c for region in transformed for r, c in region)
    return tuple(frozenset((r-r0, c-c0) for r, c in region) for region in transformed)


def _shape(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _orbit(cells):
    return {_shape(_transform((cells,), op)[0]) for op in range(8)}


def _canonical_shape(cells):
    return min(_orbit(cells))


def _points(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _recognize(inp):
    """Local roles are encoded by stubs and one/two-color components."""
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 5 and np.sum(counts == max(counts)) == 1, "need five colors and a unique dominant background")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for rotation in range(4):
        a = np.rot90(inp, rotation)
        h, w = a.shape
        for guide in map(int, colors):
            if guide == bg:
                continue
            entries = list(map(int, np.flatnonzero(a[0] == guide)))
            if not entries or any(y-x < 2 for x, y in zip(entries, entries[1:])):
                continue
            expected = set()
            for c in entries:
                n = 0
                while n < h and a[n, c] == guide:
                    expected.add((n, c))
                    n += 1
                if not 2 <= n < h:
                    break
            else:
                if expected != set(_points(a == guide)):
                    continue
                objects = _components(_points((a != bg) & (a != guide)))
                mono, keys = [], []
                for cells in objects:
                    cs = tuple(sorted({int(a[p]) for p in cells}))
                    if len(cs) == 1:
                        mono.append((cells, cs[0]))
                    elif len(cs) == 2:
                        regions = tuple(frozenset(p for p in cells if a[p] == color) for color in cs)
                        if not all(len(_components(x)) == 1 for x in regions):
                            break
                        keys.append({"cells": cells, "colors": cs, "regions": regions})
                    else:
                        break
                else:
                    if (keys and len(mono) == 2*len(keys) and len({x[1] for x in mono}) == 1
                            and len({key['colors'] for key in keys}) == 1):
                        wall = mono[0][1]
                        if len({bg, guide, wall, *keys[0]['colors']}) == 5:
                            candidates.append((rotation, a.copy(), bg, guide, wall, mono, keys))
    _require(len(candidates) == 1, "ambiguous input roles or entry direction")
    return candidates[0]


def _fit(a, bg, wall, mono, keys):
    """Find a unique complete colored assembly, including unpowered devices."""
    targets = []
    h, w = a.shape
    fixed = frozenset().union(*(x[0] for x in mono))
    inventory = frozenset().union(*(key['cells'] for key in keys))
    for i, j in combinations(range(len(mono)), 2):
        union = mono[i][0] | mono[j][0]
        r0, r1, c0, c1 = _bbox(union)
        rect = frozenset((r, c) for r in range(r0, r1+1) for c in range(c0, c1+1))
        if frozenset(p for p in rect if a[p] != bg) != union:
            continue
        slot = rect - union
        if slot and len(_components(slot)) == 1:
            targets.append({"parts": frozenset((i, j)), "bbox": (r0,r1,c0,c1), "rect": rect, "slot": slot})
    choices = []
    for ki, key in enumerate(keys):
        moves = []
        for target in targets:
            sr, _, sc, _ = _bbox(target['slot'])
            for op in range(8):
                regions = _transform(key['regions'], op)
                for pi in range(2):
                    pr, _, pc, _ = _bbox(regions[pi])
                    placed = tuple(_shift(x, sr-pr, sc-pc) for x in regions)
                    if placed[pi] != target['slot']:
                        continue
                    handle = placed[1-pi]
                    # Keys must unfold beside or beyond the front, never upstream.
                    if any(r < target['bbox'][0] or r >= h or c < 0 or c >= w for r, c in handle):
                        continue
                    if handle & (target['rect'] | fixed | inventory):
                        continue
                    colors = key['colors'][::-1] if op >= 4 else key['colors']
                    paint = {p: color for region, color in zip(placed, colors) for p in region}
                    moves.append({**target, "key": ki, "op": op, "plug_index": pi,
                                  "regions": placed, "handle": handle, "paint": paint,
                                  "full": frozenset(paint) | target['rect']})
        _require(moves, "a key has no legal shaped slot")
        choices.append(moves)
    results = {}

    def search(index, used, occupied, selected):
        if len(results) > 1:
            return
        if index == len(keys):
            if len(used) == len(mono):
                # Source identity matters: only powered sources disappear.
                # Equal complete canvases cannot justify arbitrarily choosing
                # which of two identical inventory keys will be consumed.
                signature = tuple((move['key'],move['bbox'],tuple(sorted(move['paint'].items())))
                                  for move in selected)
                results.setdefault(signature, selected)
            return
        for move in choices[index]:
            if not move['parts'] & used and not move['full'] & occupied:
                search(index+1, used | move['parts'], occupied | move['full'], selected + [move])

    search(0, frozenset(), frozenset(), [])
    _require(len(results) == 1, "complete colored fitting is ambiguous")
    return next(iter(results.values()))


def _flow(a, bg, guide, wall, moves, keys, mode="normal"):
    """Causal layer simulation. Modes are deliberate alternative-rule probes."""
    h, w = a.shape
    barriers = a == wall
    beam = a == guide
    active = set(map(int, np.flatnonzero(beam[0])))
    powered, events, activations = set(), [], []
    for r in range(1, h):
        for i, move in enumerate(moves):
            r0, r1, c0, c1 = move['bbox']
            trigger = r0 <= r <= r1 and any(c0 <= c <= c1 for c in active)
            if mode == "all_devices":
                trigger = r == 1
            if i not in powered and trigger:
                if mode == "normal":
                    _require(r == r0, "a device is first contacted behind its front face")
                powered.add(i)
                activations.append((r, i))
                if mode != "late_assembly":
                    points = move['rect'] if mode == "no_handle" else move['full']
                    for p in points:
                        barriers[p] = True
        hits, following = {}, set()
        for c in sorted(active):
            if barriers[r, c]:
                lo = hi = c
                while lo > 0 and barriers[r, lo-1]:
                    lo -= 1
                while hi+1 < w and barriers[r, hi+1]:
                    hi += 1
                hits.setdefault((lo, hi), []).append(c)
            else:
                beam[r, c] = True
                following.add(c)
        for hit_index, ((lo, hi), incident) in enumerate(sorted(hits.items())):
            events.append((r, lo, hi, tuple(incident)))
            if mode == "no_branch":
                continue
            if ((mode == 'one_run' and hit_index > 0)
                    or (mode == 'ignore_singletons' and lo == hi)
                    or (mode == 'no_edge_detour' and (lo == 0 or hi == w-1))):
                continue
            for c in range(max(0, lo-1), min(w, hi+2)):
                if not barriers[r-1, c]:
                    beam[r-1, c] = True
            for c in (lo-1, hi+1):
                if 0 <= c < w:
                    following.add(c)
                    beam[r, c] = True
        active = following
    out = a.copy()
    for i in powered:
        for p in keys[moves[i]['key']]['cells']:
            out[p] = bg
    for i in powered:
        for p, color in moves[i]['paint'].items():
            out[p] = color
    out[beam & (out == bg)] = guide
    return out, {"powered": powered, "events": events, "activations": activations, "beam": beam}


def _analyze(input_grid):
    inp = _grid(input_grid)
    rotation, a, bg, guide, wall, mono, keys = _recognize(inp)
    moves = _fit(a, bg, wall, mono, keys)
    out, stats = _flow(a, bg, guide, wall, moves, keys)
    return np.rot90(out, -rotation).copy(), (rotation, a, bg, guide, wall, mono, keys, moves, stats)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _analyze(input_grid)[0]


def _make_shape(rng, forbidden):
    for _ in range(150):
        h, w = int(rng.integers(4, 7)), int(rng.integers(3, 7))
        y = int(rng.integers(1, h-1))
        slot = {(y, 0)}
        for c in range(1, w):
            ny = int(np.clip(y + rng.integers(-1, 2), 1, h-2))
            slot.update((r, c) for r in range(min(y, ny), max(y, ny)+1))
            y = ny
        # Optional thickness gives wider connected separators as well as steps.
        if rng.random() < .35:
            for r, c in list(sorted(slot)):
                if 1 <= r+1 <= h-2 and rng.random() < .3:
                    slot.add((r+1, c))
        rect = frozenset((r,c) for r in range(h) for c in range(w))
        fixed = rect - slot
        parts = _components(fixed)
        if len(parts) != 2 or _bbox(fixed) != (0,h-1,0,w-1):
            continue
        if len(_orbit(slot)) != 8 or _canonical_shape(slot) in forbidden:
            continue
        side = int(rng.integers(2))
        endpoint = min((p for p in slot if p[1] == (0 if side == 0 else w-1)))
        start = endpoint[0], -1 if side == 0 else w
        relationship = int(rng.integers(3))
        size = (int(rng.integers(2, max(3,len(slot)))) if relationship == 0 else
                len(slot) if relationship == 1 else len(slot)+int(rng.integers(1,5)))
        size = min(size, 16)
        handle = {start}
        for _ in range(size-1):
            frontier = sorted({(r+dr,c+dc) for r,c in handle for dr,dc in DIR4
                               if 0 <= r+dr < h+3 and (-4 <= c+dc < 0 if side == 0 else w <= c+dc < w+4)} - handle)
            if not frontier:
                break
            handle.add(frontier[int(rng.integers(len(frontier)))])
        if len(handle) == size:
            return {"rect": rect, "slot": frozenset(slot), "fixed": fixed,
                    "handle": frozenset(handle), "bbox": (0,h-1,0,w-1)}
    raise ValueError("shape construction exhausted")


def _moat(cells):
    return set(cells) | {(r+dr,c+dc) for r,c in cells for dr,dc in DIR4}


def _place(rng, cells, h, w, blocked, top_min=0):
    r0, r1, c0, c1 = _bbox(cells)
    options = [(r,c) for r in range(top_min-r0,h-r1) for c in range(-c0,w-c1)]
    rng.shuffle(options)
    for dr, dc in options:
        placed = _shift(cells, dr, dc)
        if not placed & blocked:
            return dr, dc
    raise ValueError("no empty placement")


def _construct(rng, palette, n, desired_active, parity, entries):
    bg, guide, wall, pc, hc = palette
    h = int(rng.integers(21 if n < 4 else 27, 31))
    w = int(rng.integers(19 if n < 4 else 27, 31))
    a = np.full((h,w), bg, dtype=np.int8)
    blocked, forbidden, moves, keys = set(), set(), [], []
    for i in range(n):
        shape = _make_shape(rng, forbidden)
        forbidden.add(_canonical_shape(shape['slot']))
        full = shape['rect'] | shape['handle']
        dr, dc = _place(rng, full, h, w, blocked, top_min=3)
        rect, slot, handle, fixed = (_shift(shape[name],dr,dc) for name in ('rect','slot','handle','fixed'))
        blocked |= _moat(rect | handle)
        for p in fixed:
            a[p] = wall
        # Source -> destination parity is known from the inverse operation.
        op = int(rng.integers(4)) + 4*(parity if i == 0 else int(rng.integers(2)))
        inverse = (0,3,2,1,4,5,6,7)[op]
        source_regions = _transform((slot,handle), inverse)
        colors = (pc,hc) if rng.random() < .5 else (hc,pc)
        dest_colors = colors[::-1] if op >= 4 else colors
        paint = {p: color for region,color in zip((slot,handle),dest_colors) for p in region}
        moves.append({"bbox": _bbox(rect), "rect": rect, "slot": slot, "handle": handle,
                      "full": rect|handle, "paint": paint, "key": i, "op": op,
                      "regions": (slot,handle), "plug_index": 0})
        keys.append({"regions": source_regions, "colors": colors, "cells": frozenset()})
    # Start over a randomly chosen device, not at a fixed central column.
    first = int(rng.integers(n))
    _, _, c0, c1 = moves[first]['bbox']
    columns = [int(rng.integers(c0,c1+1))]
    if entries == 2:
        candidates = [c for c in range(w) if abs(c-columns[0]) >= 2]
        columns.append(int(rng.choice(candidates)))
    for c in columns:
        a[:int(rng.integers(2,4)),c] = guide
    # Place the inventory after routing, outside the entire guide footprint.
    _, stats = _flow(a,bg,guide,wall,moves,keys)
    powered = stats['powered']
    _require(len(powered) == desired_active and len(powered) < n, "wrong causal reachability")
    _require(any(moves[i]['op']//4 == parity for i in powered), "missing parity witness")
    if desired_active >= 2:
        # The new bridge must affect activation, not merely add a decorative
        # detour beside an otherwise independently solved assembly.
        _require(_flow(a,bg,guide,wall,moves,keys,'no_handle')[1]['powered'] != powered,
                 "the handle does not change a downstream activation")
        _require(_flow(a,bg,guide,wall,moves,keys,'no_branch')[1]['powered'] != powered,
                 "all devices are independently reached by straight entries")
    blocked |= _moat(_points(stats['beam']))
    for i, key in enumerate(keys):
        full = key['regions'][0] | key['regions'][1]
        dr, dc = _place(rng, full,h,w,blocked)
        regions = tuple(_shift(x,dr,dc) for x in key['regions'])
        cells = regions[0] | regions[1]
        blocked |= _moat(cells)
        key['regions'], key['cells'] = regions, cells
        for region,color in zip(regions,key['colors']):
            for p in region:
                a[p] = color
    out, _ = _flow(a,bg,guide,wall,moves,keys)
    return a, out


def _signature(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    nt, nq = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int,rng.choice(10,5,replace=False)))
    mixed_palette, mixed_direction = rng.random() < .5, rng.random() < .5
    direction = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    unrefuted = set(RUN_SHORTCUTS)
    for split, count in (("train",nt),("test",nq)):
        for index in range(count):
            if split == 'train' and index == 0:
                n, wanted, parity, entries = 2,1,0,1
            elif split == 'train' and index == 1:
                n, wanted, parity, entries = 3,2,1,2
            elif split == 'train':
                n = int(rng.integers(2,4))
                wanted, parity, entries = int(rng.integers(1,n)), int(rng.integers(2)), int(rng.integers(1,3))
            else:
                # The first query combines a third activation with the familiar
                # handle-driven two-device cascade. Extra queries may overlap
                # the demonstrated count regime.
                n = 4 if index == 0 else int(rng.choice((3,4),p=(.5,.5)))
                wanted, parity, entries = n-1,int(rng.integers(2)),int(rng.integers(1,3))
            local_palette = tuple(map(int,rng.choice(10,5,replace=False))) if mixed_palette else palette
            rotation = int(rng.integers(4)) if mixed_direction else direction
            for attempt in range(500):
                try:
                    a, out = _construct(rng,local_palette,n,wanted,parity,entries)
                    a, out = np.rot90(a,rotation).copy(),np.rot90(out,rotation).copy()
                    if _signature(a) in seen:
                        continue
                    solved, analysis = _analyze(a)
                    if not np.array_equal(solved,out):
                        continue
                    # Check the bridge: exterior geometry actually changes flow.
                    _, canonical,bg,g,t,mono,keys,moves,stats = analysis
                    if np.array_equal(_flow(canonical,bg,g,t,moves,keys,'no_handle')[0],np.rot90(out,analysis[0])):
                        continue
                    canonical_out = np.rot90(out,analysis[0])
                    distinguished = {mode for mode in unrefuted if not np.array_equal(
                        _flow(canonical,bg,g,t,moves,keys,mode)[0],canonical_out)}
                    if split == 'test' and distinguished:
                        continue
                    if split == 'train':
                        unrefuted -= distinguished
                    seen.add(_signature(a))
                    puzzle[split].append({"input":a,"output":out})
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f"bounded scene construction exhausted: {split}/{index}, devices={n}")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen, train_parities, train_active, train_entries = set(),set(),set(),set()
    unrefuted = set(RUN_SHORTCUTS)
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, "invalid pair schema")
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _require(inp.shape == out.shape and not np.array_equal(inp,out), "invalid output shape or identity")
            _require(_signature(inp) not in seen, "duplicate input")
            seen.add(_signature(inp))
            solved, info = _analyze(inp)
            _require(np.array_equal(solved,out), "incorrect causal assembly or guide")
            rotation,a,bg,g,t,mono,keys,moves,stats = info
            canonical_out = np.rot90(out,rotation)
            _require(0 < len(stats['powered']) < len(moves), "need powered and unpowered devices")
            _require(np.all(canonical_out[a == t] == t) and np.all(canonical_out[a == g] == g), "fixed pixels changed")
            _require(np.count_nonzero(np.isin(a,keys[0]['colors'])) == np.count_nonzero(np.isin(canonical_out,keys[0]['colors'])), "key area not conserved")
            for i,move in enumerate(moves):
                source = keys[move['key']]['cells']
                _require(not source & _points(stats['beam']), "inventory intersects a guide path")
                if i in stats['powered']:
                    _require(all(canonical_out[p] == bg for p in source), "used key was not erased")
                    _require(all(canonical_out[p] == color for p,color in move['paint'].items()), "rigid key changed")
                    if split == 'train':
                        train_parities.add(int(move['op'] >= 4))
                else:
                    _require(all(canonical_out[p] == a[p] for p in source), "unused key moved")
                    _require(all(canonical_out[p] not in keys[0]['colors'] for p in move['slot']), "unpowered slot filled")
            for mode in ('all_devices','no_branch','late_assembly','no_handle'):
                wrong = _flow(a,bg,g,t,moves,keys,mode)[0]
                _require(not np.array_equal(wrong,canonical_out), f"scene does not distinguish {mode}")
            if len(stats['powered']) >= 2:
                for mode in ('no_branch','no_handle'):
                    _require(_flow(a,bg,g,t,moves,keys,mode)[1]['powered'] != stats['powered'],
                             f"cascade activation does not depend on {mode}")
            distinguished = {mode for mode in unrefuted if not np.array_equal(
                _flow(a,bg,g,t,moves,keys,mode)[0],canonical_out)}
            if split == 'train':
                unrefuted -= distinguished
            else:
                _require(not distinguished, "a training-compatible run shortcut changes this test answer")
            if split == 'train':
                train_active.add(len(stats['powered']))
                train_entries.add(int(np.count_nonzero(a[0] == g)))
    _require(train_parities == {0,1}, "training needs active rotation and reflection")
    _require({1,2} <= train_active and {1,2} <= train_entries, "training lacks simple and branching cascade witnesses")


def self_test():
    """42 fixed seeds, streaming aggregates, explicit counts and bad inputs."""
    shapes, slots, handles, operations, palettes, layouts = set(),set(),set(),set(),set(),set()
    active_counts, device_counts, relations, directions = set(),set(),set(),set()
    simultaneous = clipped = multiple_hits = pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        for split in ('train','test'):
            for pair in puzzle[split]:
                _, info = _analyze(pair['input'])
                rotation,a,bg,g,t,mono,keys,moves,stats = info
                pairs += 1
                shapes.add(a.shape)
                directions.add(rotation)
                palettes.add((bg,g,t,*keys[0]['colors']))
                active_counts.add(len(stats['powered']))
                device_counts.add(len(moves))
                normalized = np.zeros(a.shape,dtype=np.int8)
                for index,color in enumerate((bg,g,t,*keys[0]['colors'])):
                    normalized[a == color] = index
                layouts.add(hashlib.sha256(normalized.tobytes()).digest())
                rows = [e[0] for e in stats['events']]
                simultaneous += len(rows) != len(set(rows))
                clipped += any(lo == 0 or hi == a.shape[1]-1 for r,lo,hi,hit in stats['events'])
                multiple_hits += any(len(hit)>1 for r,lo,hi,hit in stats['events'])
                for move in moves:
                    slots.add(_canonical_shape(move['slot']))
                    handles.add(_canonical_shape(move['handle']))
                    operations.add(move['op'])
                    relations.add(int(np.sign(len(move['slot'])-len(move['handle']))))
    _require(len(shapes) >= 35 and len(layouts) == pairs, "structural diversity collapsed")
    _require(len(slots) >= 35 and len(handles) >= 50, "geometry diversity collapsed")
    _require(len(palettes) >= 25 and directions == set(range(4)), "palette/direction diversity collapsed")
    _require(operations == set(range(8)) and relations == {-1,0,1}, "rigid operation or size role missing")
    _require({1,2,3} <= active_counts and {2,3,4} <= device_counts, "causal count diversity collapsed")
    _require(simultaneous and clipped and multiple_hits, "branch interaction coverage missing")
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            generate_puzzle(1234,nt,nq)
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(4,None),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    one,two = generate_puzzle(932,2,1),generate_puzzle(932,2,1)
    _require(all(np.array_equal(x[k],y[k]) for s in one for x,y in zip(one[s],two[s]) for k in x), "nondeterminism")
    damaged = {s:[{k:v.copy() for k,v in p.items()} for p in one[s]] for s in one}
    damaged['test'][0]['output'] = damaged['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted an unchanged answer")
    # Two interchangeable source keys would leave an ambiguous unused key,
    # even though assembling every device gives one identical colored canvas.
    ambiguous = np.zeros((25,25),dtype=np.int8)
    ambiguous[:2,3] = 1
    slot = frozenset(((1,0),(1,1),(2,1),(2,2),(2,3)))
    handle = frozenset(((2,4),(2,5),(3,5)))
    rect = frozenset((r,c) for r in range(4) for c in range(4))
    for dr in (4,14):
        for p in _shift(rect-slot,dr,2):
            ambiguous[p] = 2
        for region,color in ((slot,3),(handle,4)):
            for p in _shift(region,dr,14):
                ambiguous[p] = color
    try:
        solve(ambiguous)
    except ValueError as error:
        _require('ambiguous' in str(error), "duplicate-key fixture failed for an unrelated reason")
    else:
        raise AssertionError("interchangeable source-key assignments were accepted")
    return {"seeds":len(HOST_SEEDS),"pairs":pairs,"shapes":len(shapes),"slots":len(slots),
            "handles":len(handles),"palettes":len(palettes),"operations":sorted(operations),
            "simultaneous_pairs":simultaneous,"clipped_pairs":clipped,"multi_hit_pairs":multiple_hits}
