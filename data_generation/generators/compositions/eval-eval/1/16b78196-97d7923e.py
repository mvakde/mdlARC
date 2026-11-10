"""Ranked zipper docks: assemble compatible strips, rank, dock and activate.

Only Python/NumPy; no files, parent imports, caches or generation state in solve.
"""
from __future__ import annotations

import hashlib
import itertools
import numbers
import numpy as np


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _check(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _check(bool(np.all((0 <= a) & (a <= 9))), "color range")
    return a.astype(np.int8, copy=True)


def _runs(values):
    values = list(map(int, values))
    return [list(group) for _, group in itertools.groupby(values, key=lambda x, c=itertools.count(): x-next(c))]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        components.append(np.array(sorted(cells), dtype=int))
    return components


def _profile(cells):
    origin = cells.min(axis=0)
    local = cells-origin
    width = int(local[:, 1].max())+1
    lo, hi = [], []
    for c in range(width):
        rows = local[local[:, 1] == c, 0]
        _check(len(rows) > 0, "empty piece column")
        _check(np.array_equal(np.sort(rows), np.arange(rows.min(), rows.max()+1)), "nonconvex piece")
        lo.append(int(rows.min()))
        hi.append(int(rows.max())+1)
    return np.array(lo), np.array(hi), local


def _parse_canonical(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _check(np.count_nonzero(counts == counts[bg]) == 1, "background must be unique modal color")
    candidates = []
    for band in range(10):
        if band == bg:
            continue
        full = np.flatnonzero(np.all(g == band, axis=1))
        if len(full) >= 2:
            candidates.append((band, int(full.min()), int(full.max())+1))
    _check(len(candidates) == 1, "one spanning band required")
    band, top, baseline = candidates[0]
    _check(baseline-top >= 3, "band needs key-bearing core")
    _check(np.all(g[top:baseline] != bg), "holes in band core")
    _check(not np.any(g[:top] == band), "projections must face outward")
    profile = []
    for c in range(g.shape[1]):
        rows = np.flatnonzero(g[baseline:, c] == band)
        _check(np.array_equal(rows, np.arange(len(rows))), "band projection not contiguous")
        profile.append(len(rows))
    profile = np.array(profile)
    docks = []
    for run in _runs(np.flatnonzero(profile > 0)):
        start, width = run[0], len(run)
        _check(width >= 3, "socket width")
        key_cells = np.argwhere(g[top:baseline, start:start+width] != band)
        _check(len(key_cells) > 0 and len(set(key_cells[:, 1])) == 1, "one unary key per socket")
        col = int(key_cells[0, 1])
        key_rows = key_cells[:, 0]
        _check(np.array_equal(key_rows, np.arange(1, len(key_cells)+1)), "unary key must follow inner band edge")
        _check(key_rows[-1] < baseline-top-1, "key must be enclosed in core")
        cap = int(g[top+1, start+col])
        _check(np.all(g[top+key_rows, start+col] == cap), "mixed unary key")
        docks.append({'start': start, 'width': width, 'rank': len(key_cells),
                      'cap': cap, 'profile': profile[start:start+width].copy()})
    _check(len(docks) > 0, "no marked sockets")
    controls = {d['cap'] for d in docks}
    _check(len(controls) == len(docks) and not controls & {bg, band}, "distinct socket identities")
    core_expected = np.full_like(g[top:baseline], band)
    for d in docks:
        block = g[top:baseline, d['start']:d['start']+d['width']]
        core_expected[:, d['start']:d['start']+d['width']] = block
    _check(np.array_equal(core_expected, g[top:baseline]), "unmarked core pixels")
    structure = np.zeros_like(g, dtype=bool)
    structure[top:baseline] = True
    structure |= g == band
    pieces = []
    for cells in _components((g != bg) & ~structure):
        colors = np.unique(g[cells[:, 0], cells[:, 1]])
        _check(len(colors) == 1, "inventory fragments must be separated and monochrome")
        color = int(colors[0])
        lo, hi, local = _profile(cells)
        _check(len(hi) >= 3, "fragment width")
        terminal = len(set(hi)) == 1
        _check((color in controls) == terminal, "only terminal fragments carry cap identities")
        pieces.append({'cells': cells, 'color': color, 'lo': lo, 'hi': hi,
                       'local': local, 'terminal': terminal})
    _check(pieces, "empty inventory")
    return {'bg': bg, 'band': band, 'top': top, 'baseline': baseline,
            'docks': docks, 'pieces': pieces, 'structure': structure}


def _decode(inp):
    candidates = []
    for rotation in range(4):
        g = np.rot90(inp, -rotation)
        try:
            scene = _parse_canonical(g)
        except AssertionError:
            continue
        candidates.append((rotation, g, scene))
    _check(len(candidates) == 1, "unique keyed-band orientation required")
    return candidates[0]


def _assemblies(scene, dock, height):
    """All full-width, translation-only chains rooted at this actual socket.

    A chain ends at the first flat, matching-cap terminal. No intended chain,
    seed, scattered position, or output label is used to find candidates.
    """
    pieces = scene['pieces']
    initial = scene['baseline']+dock['profile']
    results = []

    def visit(frontier, used, placements):
        offset = int(frontier.min())
        normalized = frontier-offset
        for i, p in enumerate(pieces):
            if i in used or len(p['lo']) != dock['width']:
                continue
            if not np.array_equal(p['lo'], normalized):
                continue
            upper = p['hi']+offset
            if int(upper.max()) > height:
                continue
            target = p['local']+np.array((offset, dock['start']))
            if np.any(scene['structure'][target[:, 0], target[:, 1]]):
                continue
            next_placements = placements+[(i, target)]
            if p['terminal']:
                if p['color'] == dock['cap']:
                    results.append({'length': int(upper[0])-scene['baseline'],
                                    'placements': next_placements})
            else:
                visit(upper, used | {i}, next_placements)

    visit(initial, set(), [])
    _check(len(results) >= dock['rank'], "ordinal exceeds compatible population")
    _check(len({r['length'] for r in results}) == len(results), "tied compatible lengths")
    return sorted(results, key=lambda r: -r['length'])


def _analyze(inp):
    rotation, g, scene = _decode(_grid(inp))
    for d in scene['docks']:
        d['assemblies'] = _assemblies(scene, d, g.shape[0])
        d['selected'] = d['assemblies'][d['rank']-1]
    return rotation, g, scene


def _free_chains(scene, dock, height):
    """Counterfactual: reconstruct strips while ignoring socket compatibility."""
    pieces = [p for p in scene['pieces'] if len(p['lo']) == dock['width']]
    continuation_profiles = {tuple(p['hi']-p['hi'].min()) for p in pieces if not p['terminal']}
    root_profiles = {tuple(p['lo']) for p in pieces if not p['terminal']
                     and tuple(p['lo']) not in continuation_profiles}
    chains = []
    free_structure = scene['structure'].copy()
    free_structure[scene['baseline']:, dock['start']:dock['start']+dock['width']] = False
    free_scene = dict(scene, structure=free_structure)
    for root in sorted(root_profiles):
        hypothetical = dict(dock, profile=np.array(root)+1, rank=1)
        try:
            chains.extend(_assemblies(free_scene, hypothetical, height))
        except AssertionError:
            continue
    return sorted(chains, key=lambda x: -x['length'])


def _render(g, scene):
    out = g.copy()
    selected_ids, destinations = set(), set()
    for d in scene['docks']:
        for i, target in d['selected']['placements']:
            _check(i not in selected_ids, "selected tabs cannot share a fragment")
            selected_ids.add(i)
            for p in map(tuple, target):
                _check(p not in destinations, "tabs overlap")
                destinations.add(p)
    for i in selected_ids:
        source = scene['pieces'][i]['cells']
        out[source[:, 0], source[:, 1]] = scene['bg']
    for d in scene['docks']:
        for i, target in d['selected']['placements']:
            _check(np.all(out[target[:, 0], target[:, 1]] == scene['bg']), "inventory blocks selected destination")
            out[target[:, 0], target[:, 1]] = d['cap']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    rotation, g, scene = _analyze(input_grid)
    return np.ascontiguousarray(np.rot90(_render(g, scene), rotation))


def _cells_between(lo, hi):
    return np.array(sorted((r, c) for c in range(len(lo)) for r in range(int(lo[c]), int(hi[c]))), dtype=int)


def _shape_pool(width):
    # Nonflat multiheight contours; connected layers are checked separately.
    return [np.array(v, dtype=int) for v in itertools.product(range(3), repeat=width)
            if min(v) == 0 and max(v) > 0]


def _scatter(rng, base, parts, reserved):
    h, w = base.shape
    for _ in range(24):
        g = base.copy()
        forbidden = reserved.copy()
        occupied = g != int(base[0, 0])
        # base[0,0] is background by construction; include diagonal clearance.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r0, r1 = max(0, -dr), min(h, h-dr)
                c0, c1 = max(0, -dc), min(w, w-dc)
                forbidden[r0+dr:r1+dr, c0+dc:c1+dc] |= occupied[r0:r1, c0:c1]
        placed, success = [], True
        order = sorted(range(len(parts)), key=lambda i: -len(parts[i][0])+float(rng.random())*3)
        for i in order:
            cells, color = parts[i]
            cells = cells-cells.min(axis=0)
            ph, pw = cells.max(axis=0)+1
            if ph > h or pw > w:
                success = False
                break
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in cells:
                valid &= ~forbidden[r:r+h-ph+1, c:c+w-pw+1]
            choices = np.argwhere(valid)
            if not len(choices):
                success = False
                break
            target = cells+choices[int(rng.integers(len(choices)))]
            g[target[:, 0], target[:, 1]] = color
            placed.append(target)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    shifted = target+np.array((dr, dc))
                    ok = np.all((shifted >= 0) & (shifted < (h, w)), axis=1)
                    forbidden[shifted[ok, 0], shifted[ok, 1]] = True
        if success:
            return g
    raise ValueError("scatter space exhausted")


def _build_pair(rng, caps, ranks, rotation, two_stage, narrow=False):
    """Construct chains, then require a fresh pixel search to recover exactly them."""
    count = len(caps)
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c not in caps])]
    bg, band = palette[:2]
    bodies = palette[2:]
    width = int(rng.integers(3, 5 if count == 1 else 4))
    pool = _shape_pool(width)
    order = rng.permutation(len(pool))
    # Sometimes equal-width sockets also have identical profiles: cap grouping
    # then matters in addition to exact geometry.
    roots = [pool[int(order[0])]]
    if count == 2:
        roots.append(roots[0] if rng.random() < .4 else pool[int(order[1])])
    excluded = {tuple(p) for p in roots}
    wrong_roots = [p for p in pool if tuple(p) not in excluded]
    rng.shuffle(wrong_roots)
    wrong_roots = wrong_roots[:count]
    connector_excluded = excluded | {tuple(p) for p in wrong_roots}
    connectors = [p for p in pool if tuple(p) not in connector_excluded]
    rng.shuffle(connectors)
    h = int(rng.integers(26 if count == 2 else 23, 31))
    w = int(rng.integers(25 if count == 2 else 19, 31))
    top = int(rng.integers(9, h-13+1))
    baseline = top+5
    base = np.full((h, w), bg, dtype=np.int8)
    base[top:baseline] = band
    starts = []
    choices = list(range(1, w-width))
    for _ in caps:
        start = int(rng.choice(choices))
        starts.append(start)
        choices = [x for x in choices if abs(x-start) >= width+2]
    parts, reserved, expected_chains = [], np.zeros((h, w), dtype=bool), []
    for family, (cap, rank, start, root) in enumerate(zip(caps, ranks, starts, roots)):
        root = root+1
        for c in range(width):
            base[baseline:baseline+root[c], start+c] = band
        base[top+1:top+1+rank, start+width//2] = cap
        # At least 3 competitors teaches a non-extreme second ordinal.
        population = 3 if count == 1 else max(2, rank)
        if count == 1 and not narrow and rng.random() < .25:
            population = 4
        chains = []
        # Depth spacing also admits close lengths and varying fragment depths.
        for j in range(population+1):
            decoy = j == population
            low = root.copy()
            if decoy:
                low = wrong_roots[family]+1
                # A wrong root may never equal ANY true root for this width.
                _check(tuple(low-low.min()) not in excluded, "decoy root collision")
            chain, current = [], low
            layers = 3 if two_stage and family == 0 and (j == 0 or
                        (j < population-1 and rng.random() < .5)) else 2
            for layer in range(layers-1):
                connector = connectors.pop()
                # All adjacent column intervals overlap; no disconnected fingers.
                offset = max(int(np.max(current+1-connector)),
                             int(np.max(np.maximum(current[:-1], current[1:])+1-
                                        np.minimum(connector[:-1], connector[1:]))))
                upper = connector+offset
                cells = _cells_between(current, upper)
                chain.append(len(parts))
                parts.append((cells, int(rng.choice(bodies))))
                current = upper
            end = int(current.max())+1+j
            if decoy:
                end = max(end, max(length for length, _, _ in chains)+1)
            if end > h-baseline:
                raise ValueError("tab too long")
            chain.append(len(parts))
            parts.append((_cells_between(current, np.full(width, end)), cap))
            chains.append((end, chain, not decoy))
        # Distinct lengths are guaranteed by redraw if contour offsets collide.
        lengths = [length for length, _, fits in chains if fits]
        if len(set(lengths)) != len(lengths):
            raise ValueError("length collision")
        if chains[-1][0] <= max(lengths):
            raise ValueError("decoy must be longer")
        reserve_end = max(lengths)
        reserved[baseline:baseline+reserve_end, start:start+width] = True
        expected_chains.append(chains)
    inp = _scatter(rng, base, parts, reserved)
    frequencies = np.bincount(inp.ravel(), minlength=10)
    if frequencies[bg] <= max(frequencies[c] for c in range(10) if c != bg):
        raise ValueError("background density")
    # Independently construct the label from the chosen generated mask chain.
    canonical_out = inp.copy()
    for family, (cap, rank, start, root) in enumerate(zip(caps, ranks, starts, roots)):
        choices = sorted((x for x in expected_chains[family] if x[2]), reverse=True)
        _, chosen, _ = choices[rank-1]
        # Scatter did not retain source IDs: recover each uniquely shaped/color
        # fragment directly, checking one corresponding component per inventory.
        for i in chosen:
            cells, color = parts[i]
            local = cells-cells.min(axis=0)
            matches = [c for c in _components(inp == color)
                       if np.array_equal(c-c.min(axis=0), local)]
            if len(matches) != 1:
                raise ValueError("indistinguishable inventory fragments")
            source = matches[0]
            canonical_out[source[:, 0], source[:, 1]] = bg
            target = cells+np.array((baseline, start))
            canonical_out[target[:, 0], target[:, 1]] = cap
    if bool(rng.integers(2)):
        inp, canonical_out = inp[:, ::-1], canonical_out[:, ::-1]
    inp = np.ascontiguousarray(np.rot90(inp, rotation))
    out = np.ascontiguousarray(np.rot90(canonical_out, rotation))
    predicted = solve(inp)
    if not np.array_equal(predicted, out):
        raise ValueError("construction disagrees with input-only chain search")
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not lo <= value <= hi:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    caps = [int(c) for c in rng.choice(10, 2, replace=False)]
    orientation = int(rng.integers(4))
    shift = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            first_contrast = split == 'train' and index < 2
            double = not first_contrast and ((split == 'test' and shift == 1) or
                                            (split == 'train' and shift != 1 and rng.random() < .35))
            active_caps = caps if double else caps[:1]
            if first_contrast:
                ranks = [index+1]
            else:
                ranks = [int(rng.integers(1, 3)) for _ in active_caps]
            rot = ((orientation+1) % 4 if split == 'test' and shift == 0 else orientation)
            if shift == 2:
                rot = (orientation+(index % 2 if split == 'train' else int(rng.integers(2)))) % 4
            # Three-fragment chains are demonstrated before any test uses them.
            three_parts = (index % 2 == 1) if split == 'train' else bool(rng.integers(2))
            if split == 'test' and shift == 2:
                # A new ordinal, backed by the complete 1,2 progression.
                ranks[0] = 3
            for attempt in range(160):
                try:
                    pair = _build_pair(rng, active_caps, ranks, rot, three_parts)
                except (ValueError, AssertionError):
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded construction budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split types")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= a <= 10 and 1 <= b <= 4 and a > b, "example counts")
    seen, rank_history, cap_history = set(), set(), {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _check(inp.shape == out.shape, "canvas preservation")
            key = (inp.shape, inp.tobytes())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            rotation, g, scene = _analyze(inp)
            actual = np.rot90(out, -rotation)
            _check(np.array_equal(_render(g, scene), actual), "wrong selected assembly")
            _check(np.array_equal(g[scene['structure']], actual[scene['structure']]), "changed band/key")
            selected = {i for d in scene['docks'] for i, _ in d['selected']['placements']}
            for i, p in enumerate(scene['pieces']):
                cells = p['cells']
                if i not in selected:
                    _check(np.all(actual[cells[:, 0], cells[:, 1]] == p['color']), "changed unselected fragment")
                else:
                    _check(np.all(actual[cells[:, 0], cells[:, 1]] == scene['bg']), "selected source not erased")
            for d in scene['docks']:
                assemblies = d['assemblies']
                _check(len(assemblies) >= 2, "no ranking competition")
                _check(all(len(x['placements']) >= 2 for x in assemblies), "assembly concept omitted")
                terminal_ids = {chain['placements'][-1][0] for chain in assemblies}
                cap_fragments = [i for i, p in enumerate(scene['pieces']) if p['color'] == d['cap']]
                _check(any(i not in terminal_ids for i in cap_fragments), "missing same-cap wrong-fit strip")
                free = _free_chains(scene, d, g.shape[0])
                _check(len(free) > len(assemblies), "no complete incompatible strip")
                chosen_ids = tuple(i for i, _ in d['selected']['placements'])
                fitless_ids = tuple(i for i, _ in free[d['rank']-1]['placements'])
                _check(chosen_ids != fitless_ids, "ignoring socket fit gives same selection")
                depth = d['selected']['length']
                rectangle = actual[scene['baseline']:scene['baseline']+depth,
                                   d['start']:d['start']+d['width']]
                _check(np.all((rectangle == scene['band']) | (rectangle == d['cap'])), "tab not rectangular")
                for i, target in d['selected']['placements']:
                    source = scene['pieces'][i]['cells']
                    _check(np.array_equal(source-source.min(axis=0), target-target.min(axis=0)), "nontranslation")
                    _check(np.all(actual[target[:, 0], target[:, 1]] == d['cap']), "activation color")
                if split == 'train':
                    rank_history.add(d['rank'])
                    cap_history.setdefault(d['cap'], set()).add(d['rank'])
                else:
                    _check(d['rank'] in rank_history or set(range(1, d['rank'])) <= rank_history,
                           "unseen ordinal without progression")
    _check({1, 2} <= rank_history, "missing unary teaching contrast")
    _check(any({1, 2} <= history for history in cap_history.values()), "same cap must change rank")


def _signature(puzzle):
    return tuple((s, tuple((np.asarray(p['input']).shape, np.asarray(p['input']).tobytes(),
                            np.asarray(p['output']).tobytes()) for p in puzzle[s])) for s in ('train', 'test'))


def self_test():
    """42 streamed seeds, bounded arrays, meaningful masks and adversarial checks."""
    signatures, shapes, structures, roots, populations, fragment_counts = set(), set(), set(), set(), set(), set()
    rotations, backgrounds, ranks, docks = set(), set(), set(), set()
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest())
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                total_pairs += 1
                inp = pair['input']
                shapes.add(inp.shape)
                mapping, normalized = {}, []
                for c in inp.flat:
                    normalized.append(mapping.setdefault(int(c), len(mapping)))
                structures.add((inp.shape, bytes(normalized)))
                rotation, g, scene = _analyze(inp)
                rotations.add(rotation)
                backgrounds.add(scene['bg'])
                docks.add(len(scene['docks']))
                for d in scene['docks']:
                    roots.add(tuple(d['profile']))
                    populations.add(len(d['assemblies']))
                    ranks.add(d['rank'])
                    fragment_counts.update(len(x['placements']) for x in d['assemblies'])
                _check(np.array_equal(solve(inp.tolist()), pair['output']), "list input solve")
        if seed == 0:
            _check(_signature(puzzle) == _signature(generate_puzzle(seed)), "determinism")
            # Recoloring and rotations must commute with the input-only rule.
            pair = puzzle['train'][1]
            perm = np.array([7, 4, 9, 0, 6, 2, 8, 1, 5, 3], dtype=np.int8)
            for rotation in range(4):
                x = perm[np.rot90(pair['input'], rotation)]
                y = perm[np.rot90(pair['output'], rotation)]
                _check(np.array_equal(solve(x), y), "rotation/palette invariance")
            bad = {s: [dict(p) for p in puzzle[s]] for s in puzzle}
            bad['test'][0] = {'input': puzzle['test'][0]['input'], 'output': puzzle['test'][0]['input']}
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("identity label accepted")
    _check(len(signatures) == 42, "seed diversity")
    _check(len(shapes) >= 35 and len(structures) >= total_pairs*.85, "structural variation")
    _check(rotations == {0, 1, 2, 3} and len(backgrounds) >= 8, "visible role variation")
    _check(docks == {1, 2} and {2, 3, 4} <= populations, "competition variation")
    _check(ranks == {1, 2, 3} and fragment_counts == {2, 3} and len(roots) >= 20, "zipper/rank variation")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(900+a*10+b, a, b)
            _check(len(p['train']) == a and len(p['test']) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {'seeds': 42, 'pairs': total_pairs, 'shapes': len(shapes),
            'structures': len(structures), 'root_profiles': len(roots)}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
