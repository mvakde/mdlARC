"""Phase-locked jigsaw loom. No parent code or files are imported at runtime."""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        root = min(todo)
        todo.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _halo(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([padded[a:a+h, b:b+w]
                                 for a in range(3) for b in range(3)])


def _scene(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > g.size / 2, "unique majority background required"
    clocks = []
    for rotation in range(4):
        view = np.rot90(g, rotation)
        if len(view) < 5:
            continue
        for ordinary in np.unique(view[0]):
            ordinary = int(ordinary)
            if ordinary == bg:
                continue
            cells = np.argwhere(view == ordinary)
            span = np.flatnonzero(view[0] == ordinary)
            if not len(span) or not np.array_equal(span, np.arange(span[0], span[-1]+1)):
                continue
            start, end = int(span[0]), int(span[-1]+1)
            if not (0 < start < end < view.shape[1]):
                continue
            expected = {(r, c) for r in (0, 2) for c in range(start, end)}
            if set(map(tuple, cells)) != expected:
                continue
            high = int(view[4, start])
            if high in (bg, ordinary) or not np.all(view[4, start:end] == high):
                continue
            if not np.all(view[[1, 3], start:end] == bg):
                continue
            clocks.append((rotation, view, ordinary, high, start, end))
    assert len(clocks) == 1, "one edge clock required"
    rotation, g, ordinary, high, start, end = clocks[0]
    bands = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, ordinary, high):
            continue
        mask = g == color
        if not np.all(mask.any(axis=1)):
            continue
        edges = []
        for row in mask:
            cs = np.flatnonzero(row)
            if not np.array_equal(cs, np.arange(cs[0], cs[-1]+1)):
                break
            edges.append((int(cs[0]), int(cs[-1]+1)))
        if len(edges) == len(g):
            edges = np.array(edges)
            if np.all(np.maximum(edges[:-1, 0], edges[1:, 0]) <
                      np.minimum(edges[:-1, 1], edges[1:, 1])):
                bands.append((color, edges))
    assert len(bands) == 1, "one fixed band parallel to sweep required"
    band, edges = bands[0]
    assert not np.any(g[:, start:end] == band), "clock corridor must miss band"
    clock_mask = np.zeros_like(g, dtype=bool)
    clock_mask[[0, 2, 4], start:end] = True
    foreground = (g != bg) & (g != band) & ~clock_mask
    pieces = []
    occupied = (g == band) | clock_mask
    for cells in _components(foreground):
        colors = g[cells[:, 0], cells[:, 1]].copy()
        bodies = set(map(int, colors)) - {high}
        assert len(bodies) == 1 and ordinary not in bodies
        assert np.any(colors == high) and np.any(colors != high)
        assert not np.any(_halo(occupied)[cells[:, 0], cells[:, 1]]), "tiles detached"
        occupied[cells[:, 0], cells[:, 1]] = True
        pieces.append((cells, colors))
    assert pieces, "at least one pinned tile required"
    base = np.full_like(g, bg)
    base[g == band] = band
    base[clock_mask] = g[clock_mask]
    return dict(grid=g, background=bg, ordinary=ordinary, high=high, band=band,
                start=start, end=end, rotation=rotation, edges=edges,
                pieces=pieces, base=base)


def _description(cells, colors, sign):
    """Row-convex tiles: rows are scan distances, columns are outward depth."""
    cross = cells[:, 0] - cells[:, 0].min()
    outward = sign * cells[:, 1]
    lower, upper = [], []
    for row in range(int(cross.max()) + 1):
        values = np.sort(outward[cross == row])
        assert len(values) and np.array_equal(values, np.arange(values[0], values[-1]+1))
        lower.append(int(values[0]))
        upper.append(int(values[-1]+1))
    origin = min(lower)
    return (tuple(v-origin for v in lower), tuple(v-origin for v in upper),
            np.column_stack((cross, outward-origin)), colors)


def _assemblies(info, require_phase=True, limit=2):
    """Enumerate all boundary windows, orders, subsets, sides and global covers.

    Search stops after two different assembled color grids, sufficient to reject
    ambiguity. Identity permutations of visually identical tiles are deduplicated.
    """
    g, edges, pieces = info['grid'], info['edges'], info['pieces']
    height, width = g.shape
    candidates = []
    for sign in (-1, 1):
        desc = [_description(cells, colors, sign) for cells, colors in pieces]
        lookup = {}
        for i, (lo, hi, local, colors) in enumerate(desc):
            lookup.setdefault(lo, []).append(i)
        boundary = 1-edges[:, 0] if sign == -1 else edges[:, 1]
        for span in sorted({len(d[0]) for d in desc}):
            for row in range(height-span+1):
                initial = tuple(map(int, boundary[row:row+span]))

                def extend(profile, used, placements):
                    offset = min(profile)
                    normalized = tuple(x-offset for x in profile)
                    if used and len(set(profile)) == 1:
                        occupied = frozenset((int(r), int(c))
                                             for _, cs in placements for r, c in cs)
                        candidates.append(dict(bits=used, placements=placements,
                                               cells=occupied, side=sign, row=row,
                                               span=span, initial=initial,
                                               outer=profile[0]))
                    for i in lookup.get(normalized, ()):
                        if used & (1 << i):
                            continue
                        lo, hi, local, colors = desc[i]
                        cells = np.column_stack((row+local[:, 0], sign*(offset+local[:, 1])))
                        if np.any(cells[:, 1] < 0) or np.any(cells[:, 1] >= width):
                            continue
                        if np.any((cells[:, 1] >= info['start']) & (cells[:, 1] < info['end'])):
                            continue
                        if np.any(info['base'][cells[:, 0], cells[:, 1]] != info['background']):
                            continue
                        if set(map(tuple, cells)) == set(map(tuple, pieces[i][0])):
                            continue
                        if require_phase and np.any(cells[colors == info['high'], 0] % 6 != 4):
                            continue
                        extend(tuple(offset+x for x in hi), used | (1 << i), placements+[(i, cells)])

                extend(initial, 0, [])
    by_piece = [[] for _ in pieces]
    for candidate in candidates:
        for i in range(len(pieces)):
            if candidate['bits'] & (1 << i):
                by_piece[i].append(candidate)
    solutions = {}
    full = (1 << len(pieces))-1

    def cover(used, occupied, chosen):
        if len(solutions) >= limit:
            return
        if used == full:
            assembled = info['base'].copy()
            for tab in chosen:
                for i, cs in tab['placements']:
                    assembled[cs[:, 0], cs[:, 1]] = pieces[i][1]
            solutions.setdefault(assembled.tobytes(), (assembled, chosen))
            return
        options = [[c for c in by_piece[i] if not c['bits'] & used
                    and not c['cells'] & occupied]
                   for i in range(len(pieces)) if not used & (1 << i)]
        for candidate in min(options, key=len):
            cover(used | candidate['bits'], occupied | candidate['cells'], chosen+[candidate])

    cover(0, frozenset(), [])
    return list(solutions.values())


def _scan(info, assembled):
    result = assembled.copy()
    # Band cells are structural, and never part of the movable foreground.
    payload = (assembled != info['background']) & (assembled != info['band'])
    payload[:, info['start']:info['end']] = False
    result[4::6][payload[4::6]] = info['high']
    for distance in range(len(result)):
        result[distance, info['start']:info['end']] = (
            info['background'] if distance % 2 else
            info['high'] if distance % 6 == 4 else info['ordinary'])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _scene(input_grid)
    solutions = _assemblies(info)
    assert len(solutions) == 1, "phase and packing must specify one assembled color grid"
    return np.rot90(_scan(info, solutions[0][0]), -info['rotation']).copy()


def _profile(rng, span, amplitude):
    mode = int(rng.integers(3))
    if mode == 0:
        p = rng.integers(0, amplitude+1, span)
    elif mode == 1:
        p = np.floor(np.linspace(0, amplitude, span)).astype(int)
        if rng.random() < .5:
            p = p[::-1].copy()
    else:
        p = np.clip(np.cumsum(rng.integers(-1, 2, span)), -amplitude, amplitude)
    p -= p.min()
    if p.max() > amplitude:
        p = np.rint(p * amplitude / p.max()).astype(int)
    if not p.any():
        p[int(rng.integers(span))] = 1
    return p


def _cuts(rng, span, count):
    cuts = [_profile(rng, span, int(rng.integers(1, 3)))]
    for _ in range(count-1):
        p = _profile(rng, span, int(rng.integers(1, 3)))
        old = cuts[-1]
        offset = max(int(np.max(old+1-p)),
                     int(np.max(np.maximum(old[:-1], old[1:])+1-
                                np.minimum(p[:-1], p[1:]))))
        cuts.append(p+offset+int(rng.integers(0, 2)))
    cuts.append(np.full(span, int(cuts[-1].max())+int(rng.integers(2, 4))))
    return cuts


def _slots(rng, spans, sides, height):
    slots = {}
    for side in set(sides):
        items = [(i, active) for i, s in enumerate(sides) if s == side for active in (True, False)]
        length = sum(spans[i] for i, _ in items) + len(items)-1
        if length > height:
            raise ValueError('socket spacing')
        for _ in range(80):
            order = rng.permutation(len(items))
            gaps = rng.multinomial(height-length, np.full(len(items)+1, 1/(len(items)+1)))
            cursor = int(gaps[0])
            trial = {}
            for j, index in enumerate(order):
                i, active = items[int(index)]
                trial[i, active] = cursor
                cursor += spans[i]+1+int(gaps[j+1])
            if all(any((trial[i, True]+r) % 6 == 4 for r in range(spans[i]))
                   and trial[i, True] % 6 != trial[i, False] % 6
                   for i, s in enumerate(sides) if s == side):
                slots.update(trial)
                break
        else:
            raise ValueError('phase-separated sockets')
    return slots


def _scatter(rng, base, tiles, bg, high):
    h, w = base.shape
    for _ in range(16):
        result = base.copy()
        order = sorted(range(len(tiles)), key=lambda i: -len(tiles[i][0])+rng.random()*5)
        for i in order:
            target, colors = tiles[i]
            local = target-target.min(axis=0)
            hh, ww = local.max(axis=0)+1
            valid = np.ones((h-hh+1, w-ww+1), dtype=bool)
            forbidden = _halo(result != bg)
            for r, c in local:
                valid &= ~forbidden[r:r+valid.shape[0], c:c+valid.shape[1]]
            # A clear clock header; scattered pieces may cross the future track.
            valid[:6] = False
            origin = target.min(axis=0)
            if 0 <= origin[0] < len(valid) and 0 <= origin[1] < valid.shape[1]:
                valid[tuple(origin)] = False
            # Every tile's source pin phase differs from its destination phase.
            pin = int(local[colors == high, 0][0])
            valid[(np.arange(len(valid))+pin) % 6 == 4] = False
            choices = np.argwhere(valid)
            if not len(choices):
                break
            cs = local+choices[int(rng.integers(len(choices)))]
            result[cs[:, 0], cs[:, 1]] = colors
        else:
            return result
    raise ValueError('scatter clearance')


def _make_pair(rng, rotation, groups, counts, reuse=False):
    spans = [int(rng.integers(3, 8)) for _ in range(groups)]
    if groups == 2 and rng.random() < .6:
        spans[1] = spans[0]
    sides = [int(rng.choice((-1, 1))) for _ in range(groups)]
    minimum = max(18, max(sum(2*spans[i]+2 for i, s in enumerate(sides) if s == side)-1
                          for side in set(sides)))
    if minimum > 30:
        raise ValueError('too many socket rows')
    height = int(rng.integers(minimum, 31))
    slots = _slots(rng, spans, sides, height)
    cuts = [_cuts(rng, span, n) for span, n in zip(spans, counts)]
    extents = {s: max([2]+[int(p[-1][0]) for p, side in zip(cuts, sides) if side == s])
               for s in (-1, 1)}
    guide_width = int(rng.integers(1, 4))
    guide_side = int(rng.choice((-1, 1)))
    margins = {s: guide_width+3 if s == guide_side else int(rng.integers(0, 4)) for s in (-1, 1)}
    core = int(rng.integers(2, 4))
    minimum_width = max(18, sum(extents.values())+sum(margins.values())+core)
    if minimum_width > 30:
        raise ValueError('outward depth')
    width = int(rng.integers(minimum_width, 31))
    slack = width-(sum(extents.values())+sum(margins.values())+core)
    core_start = margins[-1]+extents[-1]+int(rng.integers(slack+1))
    start = 1 if guide_side == -1 else width-guide_width-1
    end = start+guide_width
    palette = rng.permutation(10).tolist()
    bg, band, ordinary, high = palette[:4]
    base = np.full((height, width), bg, dtype=np.int8)
    edges = {s: rng.integers(0, 3, height) for s in (-1, 1)}
    for i, side in enumerate(sides):
        for active in (True, False):
            r = slots[i, active]
            edges[side][r:r+spans[i]] = cuts[i][0]
    for r in range(height):
        base[r, core_start-edges[-1][r]:core_start+core+edges[1][r]] = band
    base[0, start:end] = base[2, start:end] = ordinary
    base[4, start:end] = high
    target = base.copy()
    tiles = []
    body_count = 1 if reuse else int(rng.integers(1, min(4, sum(counts))+1))
    for i, side in enumerate(sides):
        row = slots[i, True]
        eligible = [r for r in range(spans[i]) if (row+r) % 6 == 4]
        for lo, hi in zip(cuts[i], cuts[i][1:]):
            cs = np.array([(row+r, core_start-1-u if side == -1 else core_start+core+u)
                           for r in range(spans[i]) for u in range(int(lo[r]), int(hi[r]))])
            body = palette[4+len(tiles) % body_count]
            colors = np.full(len(cs), body, dtype=np.int8)
            # Pins can be on different special cross-sections of a seven-row tile.
            pin_row = row+int(rng.choice(eligible))
            choices = np.flatnonzero(cs[:, 0] == pin_row)
            colors[int(rng.choice(choices))] = high
            assert np.all(target[cs[:, 0], cs[:, 1]] == bg)
            target[cs[:, 0], cs[:, 1]] = colors
            tiles.append((cs, colors))
    inp = _scatter(rng, base, tiles, bg, high)
    if np.count_nonzero(inp == bg) <= inp.size / 2:
        raise ValueError('background majority')
    # Author construction supplies the label, and input-only global search checks it.
    inp = np.rot90(inp, rotation).copy()
    info = _scene(inp)
    constructed = np.rot90(target, rotation+info['rotation'])
    solutions = _assemblies(info)
    if len(solutions) != 1 or not np.array_equal(solutions[0][0], constructed):
        raise ValueError('ambiguous or inconsistent assembly')
    if len(_assemblies(info, require_phase=False)) != 2:
        raise ValueError('phase must resolve a real geometric ambiguity')
    out = np.rot90(_scan(info, constructed), -info['rotation']).copy()
    return {'input': inp, 'output': out}


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError('illegal example count')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    rotation, regime = int(rng.integers(4)), int(rng.integers(4))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            groups = (1 if index == 0 else 2) if split == 'train' else int(rng.integers(1, 3))
            counts = [2]+[int(rng.integers(1, 3)) for _ in range(groups-1)]
            if split == 'test':
                counts[0] = 3
            turn = rotation
            if regime == 1 and split == 'test':
                turn += 1
            elif regime == 2 and split == 'test':
                turn += 2
            elif regime == 3:
                turn += index % 4
            for _ in range(160):
                try:
                    pair = _make_pair(rng, turn % 4, groups, counts,
                                      reuse=split == 'train' and index == 1)
                except (ValueError, AssertionError):
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded phase-locked scene proposals exhausted')
    validate_generated_puzzle(result)
    return result


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    info = _scene(inp)
    solutions = _assemblies(info)
    assert len(solutions) == 1
    assembled, tabs = solutions[0]
    expected = np.rot90(_scan(info, assembled), -info['rotation'])
    assert np.array_equal(expected, out), 'incorrect output'
    assert np.array_equal(inp == info['band'], out == info['band'])
    assert len(_assemblies(info, require_phase=False)) == 2, 'B must change the answer'
    occupied = np.zeros_like(assembled, dtype=bool)
    moved, expanded, preserved, nonrectangular = 0, 0, 0, 0
    for tab in tabs:
        for i, target in tab['placements']:
            source, colors = info['pieces'][i]
            assert np.array_equal(target-target.min(axis=0), source-source.min(axis=0))
            assert np.array_equal(assembled[target[:, 0], target[:, 1]], colors)
            assert not occupied[target[:, 0], target[:, 1]].any()
            occupied[target[:, 0], target[:, 1]] = True
            assert not np.array_equal(source, target)
            assert np.all(target[colors == info['high'], 0] % 6 == 4)
            moved += 1
            expanded += int(np.sum((target[:, 0] % 6 == 4) & (colors != info['high'])))
            preserved += int(np.sum((target[:, 0] % 6 != 4) & (colors != info['high'])))
            box = np.prod(source.max(axis=0)-source.min(axis=0)+1)
            nonrectangular += len(source) < box
        # Fixed band cells + tile cells cover the complete tab rectangle.
        sign, initial, outer = tab['side'], tab['initial'], tab['outer']
        inside = min(initial)-1
        for r in range(tab['row'], tab['row']+tab['span']):
            cs = [sign*u for u in range(inside, outer)]
            assert all(0 <= c < assembled.shape[1] for c in cs)
            assert np.all(assembled[r, cs] != info['background'])
    assert moved == len(info['pieces']) and expanded > 0 and preserved > 0
    assert nonrectangular == moved, 'shape matching must be nontrivial'
    # A B-only sweep of the original scattered tiles cannot produce the answer.
    assert not np.array_equal(_scan(info, info['grid']), _scan(info, assembled))
    return info, tabs


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info, tabs = _inspect_pair(pair)
            key = _key(np.asarray(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            if split == 'train':
                teaching.append((info, tabs))
    assert any(len(tabs) == 1 and len(tabs[0]['placements']) >= 2 for _, tabs in teaching)
    assert any(len(tabs) >= 2 for _, tabs in teaching)
    assert any(len({int(next(c for c in colors if c != info['high']))
                    for _, colors in info['pieces']}) < len(info['pieces'])
               for info, _ in teaching), 'teach that colors are not tile identities'


def self_test():
    """42 seeds, streaming metrics, mutation tests and all 30 legal count pairs."""
    shapes, structures, rotations, spans, lengths, layouts, bodies, roles = (set() for _ in range(8))
    pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        for split in ('train', 'test'):
            for a, b in zip(p[split], q[split]):
                assert all(np.array_equal(a[k], b[k]) for k in ('input', 'output'))
                assert np.array_equal(solve(a['input']), a['output'])
                info, tabs = _inspect_pair(a)
                shapes.add(a['input'].shape)
                rotations.add(info['rotation'])
                roles.add((info['background'], info['band'], info['ordinary'], info['high']))
                spans.update(t['span'] for t in tabs)
                lengths.update(len(t['placements']) for t in tabs)
                layouts.add(tuple(sorted(t['side'] for t in tabs)))
                bodies.add(len(info['pieces']))
                palette, norm = {}, []
                for c in a['input'].flat:
                    norm.append(palette.setdefault(int(c), len(palette)))
                structures.add((a['input'].shape, hashlib.sha256(bytes(norm)).digest()))
                pairs += 1
    assert len(shapes) >= 40 and len(structures) >= pairs*.98
    assert rotations == {0, 1, 2, 3} and spans >= {3, 4, 5, 6, 7}
    assert lengths >= {1, 2, 3} and len(layouts) >= 4 and len(bodies) >= 4
    assert len(roles) >= 100
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1700+10*nt+ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    # Solver must be independent of generation and preserve its argument.
    inp = p['test'][0]['input'].copy()
    before = inp.copy()
    answer = solve(inp.tolist())
    assert np.array_equal(inp, before) and np.array_equal(answer, p['test'][0]['output'])
    return {'seeds': 42, 'pairs': pairs, 'input_shapes': len(shapes),
            'color_normalized_structures': len(structures), 'scan_directions': len(rotations),
            'tab_spans': sorted(spans), 'chain_lengths': sorted(lengths),
            'layouts': len(layouts), 'piece_counts': sorted(bodies),
            'role_palettes': len(roles), 'independent_semantic_check': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
