"""Companion-controlled printing: shape multiplicity sets silhouette scale.

Standard library and NumPy only. No parent code, examples, or latent solver state.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


DIR4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r, c) != (0, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int64, copy=False)


def _crop(mask):
    rr, cc = np.nonzero(mask)
    if not len(rr):
        raise ValueError("Empty glyph")
    return np.asarray(mask, dtype=bool)[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _key(mask):
    return (mask.shape, mask.astype(np.uint8).tobytes())


def _rotation_key(mask):
    return min(_key(np.rot90(mask, k)) for k in range(4))


def _reflection_key(mask):
    return min(_rotation_key(mask), _rotation_key(np.fliplr(mask)))


def _components(grid, background, directions=DIR8):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == background:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in directions:
                    ny, nx = y+dy, x+dx
                    if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                            and int(grid[ny, nx]) == color):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            top, left = min(y for y, x in cells), min(x for y, x in cells)
            mask = np.zeros((max(y for y, x in cells)-top+1,
                             max(x for y, x in cells)-left+1), dtype=bool)
            for y, x in cells:
                mask[y-top, x-left] = True
            objects.append(dict(top=top, left=left, color=color, mask=mask))
    return sorted(objects, key=lambda o: (o['top'], o['left']))


def _connected(mask, directions=DIR4):
    cells = set(map(tuple, np.argwhere(mask)))
    todo, seen = [min(cells)], {min(cells)}
    while todo:
        y, x = todo.pop()
        for dy, dx in directions:
            q = (y+dy, x+dx)
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return seen == cells


def _background(grid):
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if not np.all(border == border[0]):
        raise ValueError("The input needs an uninterrupted blank border")
    return int(border[0])


def _roles(grid, matching='rotation', directions=DIR8):
    bg = _background(grid)
    objects = _components(grid, bg, directions)
    signature = {
        'rotation': lambda o: _rotation_key(o['mask']),
        'exact': lambda o: _key(o['mask']),
        'reflection': lambda o: _reflection_key(o['mask']),
        'color': lambda o: o['color'],
    }[matching]
    keys = [signature(o) for o in objects]
    frequency = Counter(keys)
    singles = [o for o, key in zip(objects, keys) if frequency[key] == 1]
    repeat_keys = [key for key, count in frequency.items() if count > 1]
    if len(singles) != 1 or len(repeat_keys) != 1:
        raise ValueError("Expected one singleton and one repeated shape class")
    copies = [o for o, key in zip(objects, keys) if key == repeat_keys[0]]
    return bg, objects, singles[0], copies


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, scale, phase, orientation and inks from this input only."""
    grid = _grid(input_grid)
    bg, _, carrier, copies = _roles(grid)
    k = len(copies)
    master = copies[0]  # Objects are ordered by bounding-box top, then left.
    ch, cw = carrier['mask'].shape
    th, tw = master['mask'].shape
    h, w = k*ch, k*cw
    if not 2 <= k <= 4 or h > 30 or w > 30:
        raise ValueError("Class count or resulting canvas is outside the family")
    out = np.full((h, w), bg, dtype=np.int64)
    # This cell-loop path is deliberately separate from construction's repeat/tile.
    for r in range(h):
        for c in range(w):
            if carrier['mask'][r//k, c//k]:
                out[r, c] = (master['color'] if master['mask'][r % th, c % tw]
                             else carrier['color'])
    return out


def _render(carrier, master, k, bg):
    support = np.repeat(np.repeat(carrier['mask'], k, axis=0), k, axis=1)
    h, w = support.shape
    th, tw = master['mask'].shape
    tiled = np.tile(master['mask'], ((h+th-1)//th, (w+tw-1)//tw))[:h, :w]
    out = np.full((h, w), bg, dtype=np.int64)
    out[support] = carrier['color']
    out[support & tiled] = master['color']
    return out


def _random_mask(rng, limit):
    """Connected masks, ranging from simple geometry to arbitrary local growth."""
    kind = int(rng.integers(8))
    h, w = (int(rng.integers(2, limit+1)) for _ in range(2))
    mask = np.zeros((h, w), dtype=bool)
    if kind == 0:
        mask[:] = True
        if h > 2 and w > 2 and rng.random() < .7:
            mask[1:-1, 1:-1] = False
    elif kind == 1:
        mask[:, 0] = True
        mask[-1] = True
    elif kind == 2:
        mask[int(rng.integers(h)), :] = True
        mask[:, int(rng.integers(w))] = True
    elif kind == 3:
        mask = np.ones((1, int(rng.integers(1, limit+1))), dtype=bool)
    elif kind == 4:
        # Four-connected branch with a strictly diagonal attachment.
        mask[0, 0] = True
        mask[1:, 1] = True
        mask[-1, 1:] = True
        if min(h, w) > 3 and rng.random() < .5:
            mask[1:-1, 1:-1] = True
    elif kind == 5:
        # Staircase: can contain only diagonal connections or thick turns.
        for r in range(h):
            c = min(w-1, r*w//h)
            mask[r, c] = True
            if r and abs(c-min(w-1, (r-1)*w//h)) > 1:
                mask[r, min(w-1, (r-1)*w//h):c+1] = True
    else:
        directions = DIR4 if kind == 6 else DIR8
        count = int(rng.integers(3, min(28, limit*limit)+1))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        while len(cells) < min(count, h*w):
            frontier = sorted({(y+dy, x+dx) for y, x in cells for dy, dx in directions
                               if 0 <= y+dy < h and 0 <= x+dx < w} - cells)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        for p in cells:
            mask[p] = True
    return _crop(np.rot90(mask, int(rng.integers(4))))


def _shape(rng, limit, predicate):
    for _ in range(300):
        mask = _random_mask(rng, limit)
        if _connected(mask, DIR8) and predicate(mask):
            return mask
    # Enumerate genuine connected glyphs, not stored puzzle examples.
    for bits in range(1, 512):
        mask = _crop(np.array([(bits >> j) & 1 for j in range(9)],
                              dtype=bool).reshape(3, 3))
        if _connected(mask, DIR8) and predicate(mask):
            return mask
    raise RuntimeError("No compatible shape in the bounded search")


def _place(rng, objects, tie=False, nonleft=False):
    """Random whole-box clearance; a small shelf arrangement guarantees fit."""
    for attempt in range(36):
        minimum_h = max(12, max(o['mask'].shape[0] for o in objects)+3)
        minimum_w = max(12, max(o['mask'].shape[1] for o in objects)+3)
        if tie:
            minimum_w = max(minimum_w, objects[0]['mask'].shape[1]
                            + objects[1]['mask'].shape[1]+3)
        h, w = int(rng.integers(minimum_h, 31)), int(rng.integers(minimum_w, 31))
        blocked = np.zeros((h, w), dtype=bool)
        placed = []

        def put(obj, top, left):
            oh, ow = obj['mask'].shape
            if not (1 <= top and 1 <= left and top+oh < h and left+ow < w):
                return False
            if blocked[top:top+oh, left:left+ow].any():
                return False
            obj['top'], obj['left'] = top, left
            blocked[top-1:top+oh+1, left-1:left+ow+1] = True
            placed.append(obj)
            return True

        remaining = list(range(len(objects)))
        if tie:
            a, b = objects[:2]
            ah, aw = a['mask'].shape
            bh, bw = b['mask'].shape
            gap = 1
            top = int(rng.integers(1, h-max(ah, bh)))
            left = int(rng.integers(1, w-aw-bw-gap))
            if rng.random() < .5:
                put(a, top, left)
                put(b, top, left+aw+gap)
            else:
                put(b, top, left)
                put(a, top, left+bw+gap)
            remaining = remaining[2:]
        order = sorted(remaining, key=lambda i: -objects[i]['mask'].size)
        for i in order:
            oh, ow = objects[i]['mask'].shape
            for _ in range(90):
                if put(objects[i], int(rng.integers(1, h-oh)),
                       int(rng.integers(1, w-ow))):
                    break
            else:
                break
        if len(placed) == len(objects):
            repeats = sorted(objects[:-1], key=lambda o: (o['top'], o['left']))
            if not nonleft or repeats[0] is not min(repeats, key=lambda o: o['left']):
                return h, w
    # At most five <=7x7 glyphs fit in two rows and three columns with borders.
    mh = max(o['mask'].shape[0] for o in objects)
    mw = max(o['mask'].shape[1] for o in objects)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)]
    if nonleft:
        slots = [(0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]
    if not tie and not nonleft:
        rng.shuffle(slots)
    h, w = 2*mh+3, 3*mw+4
    for obj, (r, c) in zip(objects, slots):
        obj['top'], obj['left'] = 1+r*(mh+1), 1+c*(mw+1)
    return h, w


def _sample_pair(rng, k, limit, witness=None):
    mirror = witness == 'mirror' or (witness is None and rng.random() < .18)
    diagonal = witness == 'ordinary'
    base = _shape(rng, limit, lambda m: (
        (not witness or (not m.all() and min(m.shape) >= 2))
        and (not mirror or _rotation_key(m) != _rotation_key(np.fliplr(m)))))
    turns = [int(rng.integers(4)) for _ in range(k)]
    if witness:
        turns[1] = (turns[0]+1) % 4
    repeats = [dict(mask=np.rot90(base, turn).copy()) for turn in turns]
    if mirror:
        fabric = _crop(np.rot90(np.fliplr(base), int(rng.integers(4))))
    else:
        fabric = _shape(rng, limit, lambda m: (
            _rotation_key(m) != _rotation_key(base)
            and _reflection_key(m) != _reflection_key(base)
            and (not diagonal or not _connected(m, DIR4))))
    carrier = dict(mask=fabric)
    objects = repeats + [carrier]
    h, w = _place(rng, objects, tie=witness == 'mirror', nonleft=diagonal)
    bg = int(rng.integers(10))
    palette = [int(c) for c in rng.permutation(10) if c != bg]
    palette = palette[:int(rng.integers(2, min(6, k+2)+1))]
    for obj in objects:
        obj['color'] = int(rng.choice(palette))
    ordered = sorted(repeats, key=lambda o: (o['top'], o['left']))
    if witness:
        ordered[0]['color'], carrier['color'] = palette[0], palette[1]
        ordered[1]['color'] = palette[1]  # Reuse across roles, but visible print.
        ordered[-1]['color'] = palette[1]
    master = ordered[0]
    inp = np.full((h, w), bg, dtype=np.int64)
    for obj in objects:
        oh, ow = obj['mask'].shape
        patch = inp[obj['top']:obj['top']+oh, obj['left']:obj['left']+ow]
        patch[obj['mask']] = obj['color']
    return dict(input=inp, output=_render(carrier, master, k, bg))


def _fingerprint(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _alternative(grid, name):
    """Concrete shortcut hypotheses, used only to curate/check demonstrations."""
    matching = {'rotation_only': 'exact', 'no_reflection': 'reflection',
                'no_color': 'color'}.get(name, 'rotation')
    try:
        bg, objects, carrier, copies = _roles(
            grid, matching, DIR4 if name == 'eight_connected' else DIR8)
    except ValueError:
        return np.full((1, 1), _background(grid), dtype=np.int64)
    master = copies[0]
    k = len(copies)
    if name == 'largest_is_not_carrier':
        carrier = min(objects, key=lambda o: (-int(o['mask'].sum()), o['top'], o['left']))
        master = next(o for o in objects if o is not carrier)
    if name == 'scale_not_two':
        k = 2
    elif name == 'scale_not_three':
        k = 3
    elif name == 'scale_not_area':
        k = int(master['mask'].sum())
    elif name == 'scale_not_object_total':
        k = len(objects)
    if name == 'master_not_last':
        master = copies[-1]
    elif name == 'master_not_leftmost':
        master = min(copies, key=lambda o: (o['left'], o['top']))
    elif name == 'tie_left_first':
        master = min(copies, key=lambda o: (o['top'], -o['left']))
    elif name == 'master_not_canonical':
        mask = min((np.rot90(master['mask'], turn).copy() for turn in range(4)), key=_key)
        master = dict(master, mask=mask)
    ch, cw = carrier['mask'].shape
    th, tw = master['mask'].shape
    h, w = k*ch, k*cw
    if not 1 <= h <= 30 or not 1 <= w <= 30:
        return np.full((1, 1), bg, dtype=np.int64)
    rr, cc = np.indices((h, w))
    support = carrier['mask'][rr//k, cc//k]
    if name == 'carrier_not_tiled':
        support = carrier['mask'][rr % ch, cc % cw]
    tiled = master['mask'][(rr+(name == 'phase_row')) % th,
                           (cc+(name == 'phase_col')) % tw]
    if name == 'native_not_resized':
        tiled = master['mask'][rr*th//h, cc*tw//w]
    out = np.full((h, w), bg, dtype=np.int64)
    out[support] = carrier['color']
    if name == 'ignore_stencil':
        return out
    out[tiled if name == 'clipping' else (support & tiled)] = master['color']
    if name == 'transparent':
        out[support & ~tiled] = bg
    return out


ALTERNATIVES = frozenset((
    'rotation_only', 'no_reflection', 'no_color', 'eight_connected',
    'largest_is_not_carrier', 'scale_not_two', 'scale_not_three',
    'scale_not_area', 'scale_not_object_total', 'native_not_resized',
    'carrier_not_tiled', 'clipping', 'transparent', 'phase_row', 'phase_col',
    'master_not_last', 'master_not_leftmost', 'master_not_canonical',
    'tie_left_first', 'ignore_stencil',
))


def _evidence(pair):
    inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
    evidence = {name for name in ALTERNATIVES
                if not np.array_equal(out, _alternative(inp, name))}
    bg, _, carrier, copies = _roles(inp)
    master = copies[0]
    if np.any(out == carrier['color']) and master['color'] != carrier['color']:
        evidence.add('visible_carrier')
    if np.any(out == master['color']) and master['color'] != carrier['color']:
        evidence.add('visible_print')
    return evidence


REQUIRED = ALTERNATIVES | {'visible_carrier', 'visible_print'}
TEST_EVIDENCE = frozenset((
    'rotation_only', 'native_not_resized', 'carrier_not_tiled',
    'clipping', 'transparent', 'visible_carrier', 'visible_print',
))


def _check_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_count(num_train, 2, 10, 'num_train')
    _check_count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Both semantic lessons are present even at the two-demonstration boundary.
    # Their glyphs, geometry, palettes and positions are new draws each time.
    for _ in range(160):
        first = _sample_pair(rng, 2, int(rng.integers(3, 6)), 'mirror')
        second = _sample_pair(rng, 3, int(rng.integers(3, 6)), 'ordinary')
        if (REQUIRED <= (_evidence(first) | _evidence(second))
                and first['input'].shape != second['input'].shape
                and first['output'].shape != second['output'].shape):
            break
    else:
        raise RuntimeError("Bounded demonstration search found insufficient evidence")
    puzzle = {'train': [first, second], 'test': []}
    seen = {_fingerprint(p['input']) for p in puzzle['train']}

    def append(split, k):
        for _ in range(80):
            pair = _sample_pair(rng, k, int(rng.integers(3, 8)))
            if _fingerprint(pair['input']) in seen:
                continue
            if split == 'test' and pair['output'].shape in training_shapes:
                continue
            if split == 'test' and not TEST_EVIDENCE <= _evidence(pair):
                continue
            seen.add(_fingerprint(pair['input']))
            puzzle[split].append(pair)
            return
        raise RuntimeError("Bounded distinct-example search exhausted")

    while len(puzzle['train']) < num_train:
        append('train', int(rng.choice([2, 3])))
    training_shapes = {p['output'].shape for p in puzzle['train']}
    # A count-four extrapolation plus optional other novel count/shape combinations.
    append('test', 4)
    while len(puzzle['test']) < num_test:
        append('test', int(rng.choice([2, 3, 4])))
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp = np.asarray(pair['input'])
    bg, objects, carrier, copies = _roles(inp)
    master = copies[0]
    return dict(background=bg, k=len(copies), input_shape=inp.shape,
                output_shape=np.asarray(pair['output']).shape,
                carrier_shape=carrier['mask'].shape, period=master['mask'].shape,
                carrier_key=_rotation_key(carrier['mask']),
                stencil_key=_rotation_key(master['mask']),
                diagonal=any(not _connected(o['mask'], DIR4) for o in objects),
                mirror=_reflection_key(carrier['mask']) == _reflection_key(master['mask']),
                palette=len({o['color'] for o in objects}),
                ink_overlap=any(o['color'] == carrier['color'] for o in copies),
                tie=sum(o['top'] == master['top'] for o in copies) > 1)


def validate_generated_puzzle(puzzle):
    """Reconstruct semantics and teaching evidence from the returned grids."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _fingerprint(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            bg, objects, carrier, copies = _roles(inp)
            k, master = len(copies), copies[0]
            assert 2 <= k <= 4 and len(objects) == k+1
            occupied = set()
            for obj in objects:
                assert max(obj['mask'].shape) <= 7
                assert _connected(obj['mask'], DIR8)
                points = {(obj['top']+int(y), obj['left']+int(x))
                          for y, x in np.argwhere(obj['mask'])}
                halo = {(y+dy, x+dx) for y, x in occupied
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                assert not points & halo, "Distinct glyphs need blank clearance"
                occupied |= points
            support = np.repeat(np.repeat(carrier['mask'], k, 0), k, 1)
            assert out.shape == support.shape, "Multiplicity must control both axes"
            assert np.array_equal(out != bg, support), "Silhouette or clipping changed"
            assert set(np.unique(out)) <= {bg, carrier['color'], master['color']}
            assert np.array_equal(out, _render(carrier, master, k, bg)), "Wrong print"
            assert np.array_equal(out, solve(inp)), "Input-only reconstruction disagrees"
            features[split].append(_features(pair))
    evidence = set().union(*(_evidence(p) for p in puzzle['train']))
    assert REQUIRED <= evidence, f"Missing training evidence: {REQUIRED-evidence}"
    train, test = features['train'], features['test']
    assert {f['k'] for f in train} == {2, 3}
    assert any(f['k'] == 4 for f in test), "Missing multiplicity extrapolation"
    train_shapes = {f['output_shape'] for f in train}
    assert len(train_shapes) >= 2
    assert all(f['output_shape'] not in train_shapes for f in test)
    assert all(TEST_EVIDENCE <= _evidence(pair) for pair in puzzle['test'])
    assert any(f['mirror'] for f in train) and any(not f['mirror'] for f in train)
    assert any(f['diagonal'] for f in train)


def _puzzle_key(puzzle):
    return tuple((_fingerprint(np.asarray(p['input'])), _fingerprint(np.asarray(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed audit, legal counts, semantic witnesses and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    fs = [_features(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    metrics = dict(seeds=42, pairs=len(fs),
                   input_shapes=len({f['input_shape'] for f in fs}),
                   output_shapes=len({f['output_shape'] for f in fs}),
                   carriers=len({f['carrier_key'] for f in fs}),
                   stencils=len({f['stencil_key'] for f in fs}),
                   periods=len({f['period'] for f in fs}),
                   backgrounds=sorted({f['background'] for f in fs}),
                   palettes=sorted({f['palette'] for f in fs}),
                   multiplicities=sorted({f['k'] for f in fs}))
    assert metrics['input_shapes'] >= 50 and metrics['output_shapes'] >= 30
    assert metrics['carriers'] >= 45 and metrics['stencils'] >= 45
    assert metrics['periods'] >= 15 and metrics['backgrounds'] == list(range(10))
    assert metrics['multiplicities'] == [2, 3, 4]
    assert any(f['carrier_shape'][0] != f['carrier_shape'][1] for f in fs)
    assert any(1 in f['carrier_shape'] for f in fs)
    assert any(max(f['carrier_shape']) == 7 for f in fs)
    for split in ('train', 'test'):
        sf = [_features(pair) for p in puzzles for pair in p[split]]
        assert any(f['mirror'] for f in sf) and any(not f['mirror'] for f in sf)
        assert any(f['diagonal'] for f in sf) and any(f['ink_overlap'] for f in sf)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(10000+10*a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Palette renaming is equivariant, including the visible background role.
    pair = puzzles[0]['train'][0]
    mapping = np.array([7, 2, 9, 0, 6, 1, 8, 4, 3, 5])
    assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
    # Extra blank margins do not affect classification, reading order, or printing.
    small = next(pair for p in puzzles for pair in p['train']
                 if max(pair['input'].shape) <= 28)
    padded = np.pad(small['input'], 1, constant_values=_background(small['input']))
    assert np.array_equal(solve(padded), small['output'])
    # Bad labels are caught by support and overwrite invariants.
    bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return metrics


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
