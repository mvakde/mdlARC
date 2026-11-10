"""Disassemble a marker-selected external silhouette into ordered rimmed ribbons.

Standard library + NumPy only. No parent code, files, labels, or RNG in solve.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np

FAST = 1
SLOW = 2                    # Also the single seed color inside the map.
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _flood(allowed, start):
    reached = np.zeros(allowed.shape, bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    pending = deque([start])
    h, w = allowed.shape
    while pending:
        r, c = pending.popleft()
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and allowed[x, y] and not reached[x, y]:
                reached[x, y] = True
                pending.append((x, y))
    return reached


def _components(mask):
    remaining = mask.copy()
    result = []
    while remaining.any():
        p = tuple(map(int, np.argwhere(remaining)[0]))
        part = _flood(remaining, p)
        result.append(part)
        remaining[part] = False
    return result


def _touch(padded, shape, diagonal=True):
    h, w = shape
    result = np.zeros(shape, bool)
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if r or c] if diagonal else N4
    for dr, dc in offsets:
        result |= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
    return result


def _halo(mask, times=1):
    result = mask.copy()
    for _ in range(times):
        result |= _touch(np.pad(result, 1), result.shape)
    return result


def _runs(line):
    padded = np.r_[False, line, False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1).tolist(),
                    np.flatnonzero(np.diff(padded) == -1).tolist()))


def _decode(value):
    grid = _grid(value)
    candidates = []
    for k in range(4):
        oriented = np.rot90(grid, k)
        for up in (False, True):
            row = -1 if up else 0
            edge = oriented[:-1, 0] if up else oriented[1:, 0]
            if not (np.all(oriented[row, 1:] == FAST) and np.all(edge == SLOW)):
                continue
            ink = int(oriented[row, 0])
            body = oriented[:-1, 1:] if up else oriented[1:, 1:]
            colors, counts = np.unique(body, return_counts=True)
            if ink in (FAST, SLOW) or len(colors) != 3 or FAST in colors or ink in colors:
                continue
            if np.count_nonzero(body == SLOW) != 1:
                continue
            bg = int(colors[np.argmax(counts)])
            if bg == SLOW:
                continue
            wall = int(next(c for c in colors if c not in (bg, SLOW)))
            if np.count_nonzero(body == bg) <= np.count_nonzero(body == wall):
                continue
            candidates.append(dict(body=body.copy(), background=bg, wall=wall,
                                   ink=ink, up=up, rotation=k))
    if len(candidates) != 1:
        raise ValueError("The input must have one readable L key and one seeded map")
    return candidates[0]


def _topology(body, wall):
    seed = tuple(map(int, np.argwhere(body == SLOW)[0]))
    selected = _flood(body != wall, seed)
    outside = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))
    silhouette = ~outside[1:-1, 1:-1]
    rim = selected & _touch(outside, body.shape)
    return selected, outside, silhouette, rim


def _boxes(silhouette, up):
    h, _ = silhouette.shape
    if h % 2:
        raise ValueError("The map's slow extent must be even")
    rows = list(range(0, h, 2))
    if up:
        rows.reverse()
    boxes = []
    for r in rows:
        band = silhouette[r:r+2]
        for a, b in _runs(band.any(axis=0)):
            if a % 2 or b % 2 or not band[:, a:b].all():
                raise ValueError("The external silhouette must tile full even-length ribbons")
            boxes.append((r, a, b))
    if not 1 <= len(boxes) <= 15:
        raise ValueError("Expected 1..15 ribbons")
    return boxes


def _stack(blocks, background, left_align=False):
    width = max(b.shape[1] for b in blocks)
    result = np.full((2 * len(blocks), width), background, np.int8)
    for i, block in enumerate(blocks):
        start = 0 if left_align else (width - block.shape[1]) // 2
        result[2*i:2*i+2, start:start+block.shape[1]] = block
    return result


def _assemble(body, silhouette, rim, bg, ink, up):
    painted = body.copy()
    painted[rim & (body == bg)] = ink
    boxes = _boxes(silhouette, up)
    blocks = [painted[r:r+2, a:b].copy() for r, a, b in boxes]
    return _stack(blocks, bg), boxes, blocks


def _analyze(value):
    info = _decode(value)
    body, wall = info['body'], info['wall']
    selected, outside, silhouette, rim = _topology(body, wall)
    out, boxes, blocks = _assemble(body, silhouette, rim, info['background'],
                                   info['ink'], info['up'])
    info.update(selected=selected, outside=outside, silhouette=silhouette,
                rim=rim, boxes=boxes, blocks=blocks, output=out)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles, topology, ribbon cuts and output from this input alone."""
    return _analyze(input_grid)['output'].copy()


def _edge_touch(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _macro_ok(wall):
    if wall.sum() * 2 >= wall.size or not wall.any():
        return False
    a, b, c, d = wall[:-1, :-1], wall[:-1, 1:], wall[1:, :-1], wall[1:, 1:]
    if np.any((a == d) & (b == c) & (a != b)):
        return False
    return all(p.sum() >= 3 and _edge_touch(p) for p in _components(~wall))


def _macro_scene(rng, witness, up):
    if witness:
        h, w = int(rng.integers(7, 10)), int(rng.integers(12, 15))
        cut = int(rng.integers(4, 6)) if witness == 1 else int(rng.integers(3, 5))
        wall = np.zeros((h, w), bool)
        wall[:, cut] = True
        # A finger at the near slow edge creates two runs that merge later.
        lo, hi = (0, cut) if witness == 1 else (cut+1, w)
        finger = int(rng.integers(lo+1, hi-1))
        depth = int(rng.integers(2, 4))
        if up:
            wall[h-depth:h, finger] = True
        else:
            wall[:depth, finger] = True
        # A far-side branch changes the compartment count without controlling
        # the seed-side cross-sections. It is optional, never a semantic role.
        if rng.random() < .55:
            row = int(rng.integers(2, h-2))
            if witness == 1:
                wall[row, cut+1:] = True
            else:
                wall[row, :cut] = True
        selected = _flood(~wall, (h//2, lo))
        return wall, selected
    h, w = int(rng.integers(2, 13)), int(rng.integers(3, 15))
    wall = np.zeros((h, w), bool)
    if w >= 6 and rng.random() < .65:
        cut = int(rng.integers(2, w-2))
        wall[:, cut] = True
        for _ in range(int(rng.integers(0, 3))):
            if h >= 5:
                row = int(rng.integers(1, h-1))
                trial = wall.copy()
                if rng.random() < .5:
                    trial[row, :cut] = True
                else:
                    trial[row, cut+1:] = True
                if _macro_ok(trial):
                    wall = trial
    # Independently positioned intrusions: bars, rectangles and stepped lobes.
    for _ in range(int(rng.integers(1, 6))):
        sh = int(rng.integers(1, max(2, h//2+1)))
        sw = int(rng.integers(1, max(2, w//3+1)))
        r, c = int(rng.integers(h-sh+1)), int(rng.integers(w-sw+1))
        side = int(rng.integers(4))
        if side == 0:
            r = 0
        elif side == 1:
            r = h-sh
        elif side == 2:
            c = 0
        else:
            c = w-sw
        trial = wall.copy()
        trial[r:r+sh, c:c+sw] = True
        if sh > 1 and sw > 1 and rng.random() < .4:
            # The shoulder stays attached to the chosen exterior edge.
            if side == 0:
                trial[r+sh-1, c+sw-1] = wall[r+sh-1, c+sw-1]
            elif side == 1:
                trial[r, c+sw-1] = wall[r, c+sw-1]
        if _macro_ok(trial):
            wall = trial
    if not wall.any():
        wall[0, 0] = True
    parts = _components(~wall)
    selection = int(rng.integers(3))
    index = int(np.argmin([p.sum() for p in parts])) if selection == 0 else (
        int(np.argmax([p.sum() for p in parts])) if selection == 1 else int(rng.integers(len(parts))))
    return wall, parts[index]


def _place_island(rng, walls, region):
    # Solid connected pixel motifs, with two free cells of clearance. They
    # cannot split the selected free region or touch the external boundary.
    safe = region & ~walls
    for _ in range(2):
        safe &= ~_touch(np.pad(~safe, 1, constant_values=True), safe.shape)
    starts = np.argwhere(safe).tolist()
    if not starts:
        return False
    for attempt in range(16):
        sh, sw = (int(rng.integers(1, 5)), int(rng.integers(1, 5))) if attempt < 12 else (1, 1)
        motif = np.ones((sh, sw), bool)
        kind = int(rng.integers(3))
        if kind == 1:
            motif[1:, 1:] = False
        elif kind == 2:
            for row in range(sh):
                motif[row, int(rng.integers(1, sw+1)):] = False
        places = [(r, c) for r, c in starts
                  if r+sh <= walls.shape[0] and c+sw <= walls.shape[1]]
        rng.shuffle(places)
        for r, c in places:
            if not np.all(safe[r:r+sh, c:c+sw][motif]):
                continue
            cells = np.zeros(walls.shape, bool)
            cells[r:r+sh, c:c+sw] = motif
            trial = walls | cells
            if len(_components(region & ~trial)) != 1:
                continue
            walls[cells] = True
            return True
    return False


def _encode(body, bg, ink, up, rotation):
    h, w = body.shape
    grid = np.full((h+1, w+1), bg, np.int8)
    row = -1 if up else 0
    if up:
        grid[:-1, 1:] = body
        grid[:-1, 0] = SLOW
    else:
        grid[1:, 1:] = body
        grid[1:, 0] = SLOW
    grid[row, 1:] = FAST
    grid[row, 0] = ink
    return np.rot90(grid, rotation).copy()


def _make_pair(rng, witness, up, rotation, roles):
    bg, wall_color, ink = roles
    macro_walls, macro_selected = _macro_scene(rng, witness, up)
    walls = np.repeat(np.repeat(macro_walls, 2, axis=0), 2, axis=1)
    silhouette = np.repeat(np.repeat(macro_selected, 2, axis=0), 2, axis=1)
    try:
        _boxes(silhouette, up)
    except ValueError:
        return None
    count = int(rng.integers(1, 4)) if witness else int(rng.integers(0, 4))
    inserted = sum(_place_island(rng, walls, silhouette) for _ in range(count))
    if witness and not inserted:
        return None
    # Objects in unselected compartments are genuine irrelevant distractors.
    others = _components(~walls & ~silhouette)
    for _ in range(int(rng.integers(1 if witness else 0, 4))):
        if others:
            _place_island(rng, walls, others[int(rng.integers(len(others)))])
    free = silhouette & ~walls
    outside_known = np.pad(~silhouette, 1, constant_values=True)
    rim = free & _touch(outside_known, walls.shape)
    positions = np.argwhere(free & (~rim if witness == 1 else rim if witness == 2 else True))
    if not len(positions):
        return None
    seed = tuple(map(int, positions[int(rng.integers(len(positions)))]))
    body = np.where(walls, wall_color, bg).astype(np.int8)
    body[seed] = SLOW
    if np.count_nonzero(body == bg) <= walls.sum():
        return None
    # Labels use the constructive silhouette, not solve's recovered flood.
    output, _, _ = _assemble(body, silhouette, rim, bg, ink, up)
    pair = dict(input=_encode(body, bg, ink, up, rotation), output=output)
    info = _analyze(pair['input'])
    if not np.array_equal(info['silhouette'], silhouette):
        return None
    if witness == 1:
        areas = [p.sum() for p in _components(body != wall_color)]
        if info['selected'].sum() >= max(areas):
            return None
    if witness == 2:
        areas = [p.sum() for p in _components(body != wall_color)]
        if info['selected'].sum() != max(areas):
            return None
    return pair


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    a = np.asarray(grid)
    return (a.shape, tuple(map(int, a.flat)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    palette = [c for c in range(10) if c not in (FAST, SLOW)]
    fixed_roles = tuple(map(int, rng.choice(palette, 3, replace=False)))
    changing_palette = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    first_up = bool(rng.integers(2))
    train_configs = [(first_up, base_rotation), (not first_up, (base_rotation+1) % 4)]
    test_configs = [(not first_up, base_rotation), (first_up, (base_rotation+1) % 4)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            witness = i+1 if split == 'train' and i < 2 else 0
            config = train_configs[i] if witness else (
                train_configs[int(rng.integers(2))] if split == 'train' else test_configs[int(rng.integers(2))])
            roles = tuple(map(int, rng.choice(palette, 3, replace=False))) if changing_palette else fixed_roles
            for attempt in range(160):
                pair = _make_pair(rng, witness, *config, roles)
                if pair is not None and split == 'test':
                    info = _analyze(pair['input'])
                    if len(info['boxes']) < 3 or len({b-a for _, a, b in info['boxes']}) < 2:
                        pair = None
                if pair is not None and witness == 2:
                    try:
                        _training_evidence([_analyze(p['input']) for p in puzzle['train'] + [pair]])
                    except AssertionError:
                        pair = None
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a distinct readable compartment")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(infos):
    # These checks observe returned arrays, not construction flags.
    assert {d['up'] for d in infos} == {False, True}
    assert len({d['rotation'] % 2 for d in infos}) == 2
    assert any(d['selected'].sum() < max(p.sum() for p in _components(d['body'] != d['wall']))
               for d in infos), "Need a smaller selected compartment"
    assert any(d['selected'].sum() == max(p.sum() for p in _components(d['body'] != d['wall']))
               for d in infos), "Need a largest selected compartment"
    holes, diagonal, alternatives = False, False, set()
    for d in infos:
        body, bg, ink, selected = d['body'], d['background'], d['ink'], d['selected']
        # Contouring every obstruction is a plausible but different rule.
        all_rim = selected & _touch(np.pad(~selected, 1, constant_values=True), body.shape)
        alt, _, _ = _assemble(body, d['silhouette'], all_rim, bg, ink, d['up'])
        holes |= not np.array_equal(alt, d['output'])
        diagonal |= bool(np.any(d['rim'] & ~_touch(d['outside'], body.shape, False) & (body == bg)))
        blocks, boxes = d['blocks'], d['boxes']
        lengths = [b.shape[1] for b in blocks]
        if max(lengths) > lengths[0]:
            alternatives.add('first_max')
        if not np.array_equal(_stack(blocks, bg, True), d['output']):
            alternatives.add('left_align')
        if len({_key(b) for b in blocks}) < len(blocks):
            alternatives.add('deduplicate')
        orders = {
            'slow_reverse': sorted(range(len(boxes)), key=lambda i: (-boxes[i][0] if not d['up'] else boxes[i][0], boxes[i][1])),
            'fast_reverse': sorted(range(len(boxes)), key=lambda i: (-boxes[i][0] if d['up'] else boxes[i][0], -boxes[i][1])),
            'fast_major': sorted(range(len(boxes)), key=lambda i: (boxes[i][1], -boxes[i][0] if d['up'] else boxes[i][0])),
        }
        for name, order in orders.items():
            if not np.array_equal(_stack([blocks[i] for i in order], bg), d['output']):
                alternatives.add(name)
        if not np.array_equal(_stack([b[::-1] for b in blocks], bg), d['output']):
            alternatives.add('thickness_reflection')
    assert holes, "Need unoutlined enclosed-island evidence"
    assert diagonal, "Need an eight-neighbor-only external rim cell"
    assert {'first_max', 'left_align', 'deduplicate', 'slow_reverse', 'fast_reverse',
            'fast_major', 'thickness_reflection'} <= alternatives, alternatives


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_infos = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _analyze(inp)
            assert np.array_equal(d['output'], out)
            body, selected, silhouette = d['body'], d['selected'], d['silhouette']
            assert all(_edge_touch(p) for p in _components(body != d['wall'])), "No nested free rooms"
            assert np.all(body[silhouette & ~selected] == d['wall']), "Only solid islands inside the outline"
            assert np.count_nonzero(out == SLOW) == 1
            assert FAST not in out and np.any(out == d['ink'])
            assert out.shape == (2*len(d['boxes']), max(b-a for _, a, b in d['boxes']))
            # Each copied island pixel occurs once, without repainting it.
            assert np.count_nonzero(out == d['wall']) == np.count_nonzero(body[silhouette] == d['wall'])
            assert not np.array_equal(inp, out)
            if split == 'train':
                train_infos.append(d)
    _training_evidence(train_infos)


def _serial(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
                       for s in ('train', 'test')}, separators=(',', ':'))


def self_test():
    signatures, shapes, layouts, palettes, ribbon_counts, widths, island_counts = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_serial(puzzle))
        if seed < 3:
            assert _serial(puzzle) == _serial(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _analyze(pair['input'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                layouts.add((d['silhouette'].shape, d['silhouette'].tobytes(), tuple(d['boxes'])))
                palettes.add((d['background'], d['wall'], d['ink']))
                ribbon_counts.add(len(d['boxes']))
                widths.add(pair['output'].shape[1])
                island_counts.add(len(_components(d['silhouette'] & ~d['selected'])))
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input'])
                # Whole-input rotations preserve the normalized inventory.
                if seed < 4:
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(before, k)), pair['output'])
    assert len(signatures) == 42 and len(layouts) >= 120 and len(shapes) >= 90
    assert len(palettes) >= 30 and len(ribbon_counts) >= 10 and len(widths) >= 10
    assert {0, 1, 2} <= island_counts
    # Count contract, including every legal combination, and strict bad types.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    _metamorphic_tests()
    corrupted = generate_puzzle(901, 2, 1)
    g = corrupted['test'][0]['output']
    g[0, 0] = (int(g[0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt label accepted")


def _metamorphic_tests():
    # Hand-constructed fixture: a two-run notch, detached island, another room.
    bg, wall, ink = 0, 3, 4
    body = np.zeros((12, 20), np.int8)
    body[:, 12:14] = wall
    body[:4, 4:6] = wall
    body[6:8, 6:8] = wall
    body[9, 2] = SLOW
    inp = _encode(body, bg, ink, False, 0)
    d = _analyze(inp)
    assert len(d['boxes']) == 8 and d['output'].shape == (16, 12)
    assert d['silhouette'][6:8, 6:8].all() and not d['rim'][5:9, 5:9].any()
    # Known first two ribbons: near outer edge above a notch, centered.
    assert np.array_equal(d['output'][:2], np.array([[0,0,0,0,4,4,4,4,0,0,0,0],
                                                   [0,0,0,0,4,0,0,4,0,0,0,0]], np.int8))
    moved = body.copy()
    moved[9, 2], moved[10, 8] = bg, SLOW
    md = _analyze(_encode(moved, bg, ink, False, 0))
    assert np.array_equal(md['silhouette'], d['silhouette']) and np.array_equal(md['rim'], d['rim'])
    erased = body.copy()
    erased[2:4, 16:18] = wall
    assert np.array_equal(solve(_encode(erased, bg, ink, False, 0)), d['output'])
    no_island = body.copy()
    no_island[6:8, 6:8] = bg
    nd = _analyze(_encode(no_island, bg, ink, False, 0))
    assert np.array_equal(nd['rim'], d['rim']) and nd['boxes'] == d['boxes']
    attached = body.copy()
    attached[:8, 6:8] = wall
    ad = _analyze(_encode(attached, bg, ink, False, 0))
    assert len(ad['boxes']) != len(d['boxes']) and not np.array_equal(ad['output'], d['output'])
    across = body.copy()
    across[9, 2], across[6, 16] = bg, SLOW
    assert not np.array_equal(solve(_encode(across, bg, ink, False, 0)), d['output'])
    # Any permutation of non-guide colors is equivariant, including black roles.
    mapping = np.array([8, 1, 2, 0, 9, 5, 6, 7, 3, 4], np.int8)
    assert np.array_equal(solve(mapping[inp]), mapping[d['output']])


if __name__ == '__main__':
    self_test()
    print('self_test passed')
