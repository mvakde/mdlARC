"""Nested collectors: local color ownership, component counting and stack addition.

No files, parent imports, hidden puzzle parameters, or training data are used by
solve. Generation labels are drawn from explicit scene descriptors; the parser
checks them from the returned pixels. Run self_test() for bounded diagnostics.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=False)


def _parts(a, bg):
    seen, result = set(), []
    for y, x in zip(*np.nonzero(a != bg)):
        start = (int(y), int(x))
        if start in seen:
            continue
        color, cells, todo = int(a[start]), [], [start]
        seen.add(start)
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]
                        and (nr, nc) not in seen and int(a[nr, nc]) == color):
                    seen.add((nr, nc))
                    todo.append((nr, nc))
        cells = tuple(sorted(cells))
        rs, cs = zip(*cells)
        box = (min(rs), min(cs), max(rs), max(cs))
        mask = tuple((r-box[0], c-box[1]) for r, c in cells)
        result.append(dict(color=color, cells=cells, box=box, mask=mask))
    return result


def _perimeter(box):
    t, l, b, r = box
    return tuple((y, x) for y in range(t, b+1) for x in range(l, r+1)
                 if y in (t, b) or x in (l, r))


def _inside(inner, outer):
    t, l, b, r = inner
    T, L, B, R = outer
    return T < t <= b < B and L < l <= r < R


def _area(box):
    t, l, b, r = box
    return (b-t+1)*(r-l+1)


def _owner(box, color, frames):
    matches = [i for i, f in enumerate(frames)
               if f['color'] == color and _inside(box, f['box'])]
    return min(matches, key=lambda i: _area(frames[i]['box'])) if matches else None


def _draw(a, cells, color):
    for r, c in cells:
        a[r, c] = color


def _translated(mask, y, x):
    return tuple((y+dy, x+dx) for dy, dx in mask)


def _parse_upright(a):
    """Canonical view: all reservoirs start at their lower left corners."""
    values, frequencies = np.unique(a, return_counts=True)
    _require(np.count_nonzero(frequencies == frequencies.max()) == 1, "background tie")
    bg = int(values[np.argmax(frequencies)])
    parts = _parts(a, bg)
    frames = [p for p in parts if min(p['box'][2]-p['box'][0]+1,
                                     p['box'][3]-p['box'][1]+1) >= 8
              and p['cells'] == _perimeter(p['box'])]
    _require(bool(frames), "no collector frames")
    frame_ids = {id(f) for f in frames}
    others = [p for p in parts if id(p) not in frame_ids]
    # Frames form a forest of strict inclusions, never overlapping rectangles.
    for i, f in enumerate(frames):
        for g in frames[i+1:]:
            t, l, b, r = f['box']
            T, L, B, R = g['box']
            separate = b < T or B < t or r < L or R < l
            _require(separate or _inside(f['box'], g['box'])
                     or _inside(g['box'], f['box']), "crossing frames")
    # All components have an eight-neighbor moat. This includes future growth.
    labels = np.full(a.shape, -1, dtype=int)
    for i, p in enumerate(parts):
        for cell in p['cells']:
            labels[cell] = i
    for r, c in zip(*np.nonzero(labels >= 0)):
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            y, x = r+dr, c+dc
            if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
                _require(labels[y, x] in (-1, labels[r, c]), "touching components")
    frame_colors = {f['color'] for f in frames}
    used, inks = set(), set()
    for f in frames:
        t, l, b, r = f['box']
        candidates = [p for p in others if p['box'][1] == l+2
                      and p['box'][2] == b-2 and _inside(p['box'], f['box'])
                      and p['color'] not in frame_colors]
        _require(len(candidates) == 1, "missing corner reservoir")
        bottom = candidates[0]
        h = bottom['box'][2]-bottom['box'][0]+1
        w = bottom['box'][3]-bottom['box'][1]+1
        _require(1 <= h <= 2 and 1 <= w <= 3, "reservoir token dimensions")
        stack = sorted([p for p in others if p['color'] == bottom['color']
                        and p['box'][1] == l+2 and p['mask'] == bottom['mask']
                        and _inside(p['box'], f['box'])], key=lambda p: p['box'][0])
        _require(len(stack) >= 2 and stack[-1] is bottom, "visible reservoir repetition")
        starts = [p['box'][0] for p in stack]
        pitches = {v-u for u, v in zip(starts, starts[1:])}
        _require(len(pitches) == 1, "inconsistent pitch")
        pitch = pitches.pop()
        _require(h < pitch <= h+2, "reservoir pitch")
        _require(starts[0] >= t+2, "reservoir crosses wall")
        _require(bottom['color'] not in inks, "reservoir inks must be distinct")
        inks.add(bottom['color'])
        for p in stack:
            _require(id(p) not in used, "shared reservoir")
            used.add(id(p))
        f.update(ink=bottom['color'], token=bottom['mask'], pitch=pitch,
                 x=l+2, y=bottom['box'][0], initial=len(stack), stack=stack,
                 capacity=1+(bottom['box'][0]-(t+2))//pitch)
    objects = [p for p in others if id(p) not in used]
    for p in objects:
        _require(p['color'] not in inks, "reservoir ink used as loose input object")
        _require(1 <= len(p['cells']) <= 8, "loose object size")
        for f in frames:
            t, l, b, r = f['box']
            w = 1+max(x for y, x in f['token'])
            if _inside(p['box'], f['box']):
                _require(p['box'][1] >= l+w+4, "object in reserved reservoir lane")
        p['owner'] = _owner(p['box'], p['color'], frames)
    for i, f in enumerate(frames):
        t, l, b, r = f['box']
        owned = [p for p in objects if p['owner'] == i]
        f['contribution'] = len(owned)
        f['source_area'] = sum(len(p['cells']) for p in owned)
        _require(f['initial']+len(owned) <= f['capacity'], "collector overflow")
        # Check every proposed new token against all existing components.
        for k in range(f['initial'], f['initial']+len(owned)):
            cells = _translated(f['token'], f['y']-k*f['pitch'], f['x'])
            for y, x in cells:
                _require(t < y < b and l < x < r, "growth outside frame")
                _require(np.all(a[max(0, y-1):y+2, max(0, x-1):x+2] == bg),
                         "growth collision")
    return dict(background=bg, frames=frames, objects=objects)


def _inspect(a):
    candidates = []
    for rotation in range(4):
        view = np.rot90(a, rotation)
        try:
            d = _parse_upright(view)
        except (AssertionError, IndexError, ValueError):
            continue
        candidates.append((rotation, view, d))
    _require(len(candidates) == 1, "input must identify one growth orientation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count each loose component once, at its smallest matching enclosure."""
    a = _grid(input_grid)
    rotation, view, d = _inspect(a)
    out = view.copy()
    for p in d['objects']:
        if p['owner'] is not None:
            _draw(out, p['cells'], d['frames'][p['owner']]['ink'])
    for f in d['frames']:
        for k in range(f['initial'], f['initial']+f['contribution']):
            _draw(out, _translated(f['token'], f['y']-k*f['pitch'], f['x']), f['ink'])
    return np.rot90(out, -rotation).copy()


def _box(rng, t, l, h, w):
    return (t, l, t+h-1, l+w-1)


def _layout(rng, topology):
    """Continuous geometry within five small forest topologies; <=3 frames."""
    if topology in ('chain2', 'chain3'):
        h = int(rng.integers(27, 30))
        w = 29 if topology == 'chain3' else int(rng.integers(21, 30))
        t, l = int(rng.integers(31-h)), int(rng.integers(31-w))
        boxes = [_box(rng, t, l, h, w)]
        depth = 3 if topology == 'chain3' else 2
        for i in range(1, depth):
            T, L, B, R = boxes[-1]
            boxes.append((T+int(rng.integers(2, 4)), L+8,
                          B-int(rng.integers(2, 4)), R-2))
    elif topology == 'branch':
        h, w = 29, int(rng.integers(22, 30))
        t, l = int(rng.integers(2)), int(rng.integers(31-w))
        first_h = int(rng.integers(10, 13))
        gap = int(rng.integers(1, 3))
        boxes = [_box(rng, t, l, h, w),
                 _box(rng, t+2, l+8, first_h, w-10),
                 (t+2+first_h+gap, l+8, t+h-3, l+w-3)]
    elif topology == 'forest':
        w1, w2 = int(rng.integers(9, 14)), int(rng.integers(9, 14))
        gap = int(rng.integers(1, min(4, 30-w1-w2)+1))
        left = int(rng.integers(31-w1-w2-gap))
        boxes = []
        for l, w in ((left, w1), (left+w1+gap, w2)):
            h = int(rng.integers(18, 29))
            boxes.append(_box(rng, int(rng.integers(31-h)), l, h, w))
    else:
        h, w = int(rng.integers(17, 30)), int(rng.integers(10, 27))
        boxes = [_box(rng, int(rng.integers(31-h)), int(rng.integers(31-w)), h, w)]
    height = max(b for t, l, b, r in boxes)+1
    width = max(r for t, l, b, r in boxes)+1
    height += int(rng.integers(min(2, 30-height)+1))
    width += int(rng.integers(min(2, 30-width)+1))
    return boxes, (height, width)


def _mask(rng, area):
    # Free growth in a 4x3 patch, independent of number of source components.
    cells = {(int(rng.integers(4)), int(rng.integers(3)))}
    while len(cells) < area:
        frontier = sorted({(r+dr, c+dc) for r, c in cells
                           for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                           if 0 <= r+dr < 4 and 0 <= c+dc < 3}-cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    t, l = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-t, c-l) for r, c in cells))


def _place_object(rng, a, blocked, frames, color, area, owner, physical=None):
    # At most 14 masks (last four are narrow fallbacks), no disk or retry logs.
    for attempt in range(14):
        mask = _mask(rng, area) if attempt < 10 else tuple((k, 0) for k in range(area))
        mh, mw = 1+max(r for r, c in mask), 1+max(c for r, c in mask)
        if mh > a.shape[0] or mw > a.shape[1]:
            continue
        available = np.ones((a.shape[0]-mh+1, a.shape[1]-mw+1), dtype=bool)
        for dr, dc in mask:
            available &= ~blocked[dr:dr+available.shape[0], dc:dc+available.shape[1]]
        candidates = []
        for y, x in zip(*np.nonzero(available)):
            box = (int(y), int(x), int(y)+mh-1, int(x)+mw-1)
            if _owner(box, color, frames) != owner:
                continue
            enclosing = [i for i, f in enumerate(frames) if _inside(box, f['box'])]
            innermost = min(enclosing, key=lambda i: _area(frames[i]['box'])) if enclosing else None
            if physical is not None and innermost != physical:
                continue
            # A source never straddles a frame's rectangle or enters a lane.
            valid = True
            for f in frames:
                T, L, B, R = f['box']
                if _inside(box, f['box']) and box[1] < L+f['tw']+4:
                    valid = False
                if not _inside(box, f['box']):
                    for dy, dx in mask:
                        if T <= y+dy <= B and L <= x+dx <= R:
                            valid = False
            if valid:
                candidates.append((int(y), int(x)))
        if not candidates:
            continue
        y, x = candidates[int(rng.integers(len(candidates)))]
        cells = _translated(mask, y, x)
        _draw(a, cells, color)
        for r, c in cells:
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
        return cells
    return None


def _build(rng, topology, lesson, anchor, rotation):
    boxes, shape = _layout(rng, topology)
    n = len(boxes)
    palette = [int(x) for x in rng.permutation([c for c in range(10) if c != anchor])]
    bg, inks = palette[0], palette[1:1+n]
    categories = [anchor]+[int(rng.choice([anchor]+palette[n+1:n+4])) for _ in range(n-1)]
    if lesson == 0:
        categories = [anchor, anchor]
    elif lesson == 1:
        categories = [anchor, palette[n+1]]
    # Deeper test: outer/inner share a color; middle has a different color.
    if topology == 'chain3':
        categories = [anchor, palette[n+1], anchor]
    frames = []
    for i, (box, ink, color) in enumerate(zip(boxes, inks, categories)):
        t, l, b, r = box
        h = b-t+1
        th = 1 if h < 20 or topology == 'chain3' else int(rng.integers(1, 3))
        tw = int(rng.integers(1, 3))
        token = tuple((y, x) for y in range(th) for x in range(tw)
                      if not (th == tw == 2 and y == 0 and x == 1 and rng.random() < .5))
        # Draw the random notch once, then copy the resulting connected mask.
        pitch = th+int(rng.integers(1, 3))
        y = b-1-th
        capacity = 1+(y-(t+2))//pitch
        initial = int(rng.integers(2, min(3, capacity)+1))
        contribution = int(rng.integers(min(3, capacity-initial)+1))
        if lesson in (0, 1):
            contribution = (1, 2)[i] if lesson == 0 else (2, 1)[i]
            initial = 2 if (i+lesson) % 2 == 0 else 3
            if initial+contribution > capacity:
                return None
        frames.append(dict(box=box, color=color, ink=ink, token=token, tw=tw,
                           pitch=pitch, y=y, x=l+2, initial=initial,
                           contribution=contribution, capacity=capacity))
    if not any(f['contribution'] for f in frames):
        possible = [f for f in frames if f['capacity'] > f['initial']]
        if not possible:
            return None
        possible[int(rng.integers(len(possible)))]['contribution'] = 1
    a = np.full(shape, bg, dtype=np.int8)
    out = a.copy()
    blocked = np.zeros(shape, dtype=bool)
    for f in frames:
        t, l, b, r = f['box']
        border = _perimeter(f['box'])
        _draw(a, border, f['color'])
        _draw(out, border, f['color'])
        for y, x in border:
            blocked[max(0, y-1):y+2, max(0, x-1):x+2] = True
        blocked[t:b+1, l:l+f['tw']+4] = True
        for k in range(f['initial']+f['contribution']):
            cells = _translated(f['token'], f['y']-k*f['pitch'], f['x'])
            _draw(out, cells, f['ink'])
            if k < f['initial']:
                _draw(a, cells, f['ink'])
    requests = []
    for i, f in enumerate(frames):
        for j in range(f['contribution']):
            area = int(rng.integers(1, 7))
            physical = None
            if lesson == 0:
                area = 4 if i == 0 else (2, 4)[j]
                physical = i
            elif lesson == 1:
                area = (1, 3)[j] if i == 0 else 2
                physical = 1 if i == 0 and j == 0 else i
            requests.append((f['color'], area, i, physical))
    if lesson == 1:
        # A child-category object outside that child is unowned, despite a
        # matching frame elsewhere. Ancestor A objects inside B bypass B.
        requests.append((categories[1], 2, None, 0))
    else:
        # Unmatched components use the same geometry sampler; optional noise is
        # semantic, never a different species of decoration.
        other = [c for c in range(10) if c not in [bg]+inks+categories]
        for _ in range(int(rng.integers(3))):
            requests.append((int(rng.choice(other)), int(rng.integers(1, 5)), None, None))
    order = sorted(range(len(requests)), key=lambda i: (-requests[i][1], float(rng.random())))
    for index in order:
        color, area, owner, physical = requests[index]
        cells = _place_object(rng, a, blocked, frames, color, area, owner, physical)
        if cells is None:
            return None
        _draw(out, cells, frames[owner]['ink'] if owner is not None else color)
    return {'input': np.rot90(a, rotation).copy(),
            'output': np.rot90(out, rotation).copy()}


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                    ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    anchor, base_rotation = int(rng.integers(10)), int(rng.integers(4))
    result, seen = {'train': [], 'test': []}, set()
    for index in range(num_train+num_test):
        train = index < num_train
        lesson = index if index < 2 else None
        topology = 'chain2' if lesson is not None else str(rng.choice(
            ['single', 'forest', 'chain2', 'branch'] if train
            else ['chain3', 'branch']))
        if index == num_train:
            topology = 'chain3'
        rotation = (base_rotation+index) % 4 if lesson is not None else int(rng.integers(4))
        # Bounded rejection only for geometric packing/unique input, not labels.
        for attempt in range(64):
            pair = _build(rng, topology, lesson, anchor, rotation)
            if pair is None:
                continue
            key = (pair['input'].shape, pair['input'].tobytes())
            if key in seen:
                continue
            # This also excludes accidental alternative orientation parses.
            try:
                expected = solve(pair['input'])
            except AssertionError:
                continue
            _require(np.array_equal(expected, pair['output']), 'descriptor/parser disagreement')
            break
        else:
            raise AssertionError(f'bounded scene packing exhausted: {topology}/{lesson}')
        seen.add(key)
        result['train' if train else 'test'].append(pair)
    validate_generated_puzzle(result)
    return result


def _features(pair):
    rotation, view, d = _inspect(_grid(pair['input']))
    frames, objects = d['frames'], d['objects']
    for f in frames:
        f['depth'] = 1+sum(_inside(f['box'], g['box']) for g in frames)
    owned = [p for p in objects if p['owner'] is not None]
    return dict(rotation=rotation, shape=view.shape, background=d['background'],
                frames=frames, objects=objects, owned=owned,
                bypass=any(any(_inside(p['box'], g['box']) and
                               _area(g['box']) < _area(frames[p['owner']]['box'])
                               for g in frames) for p in owned),
                shadow=any(sum(_inside(p['box'], f['box']) and f['color'] == p['color']
                               for f in frames) > 1 for p in owned),
                local_discard=any(p['owner'] is None and p['color'] in
                                  {f['color'] for f in frames} for p in objects))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'canvas changed')
            key = (a.shape, a.tobytes())
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            _require(np.array_equal(solve(a), b), 'wrong transfer/highlight/preservation')
            _require(not np.array_equal(a, b), 'identity pair')
            f = _features(pair)
            _require(bool(f['owned']), 'no arithmetic transfer')
            features.append(f)
    tr = features[:len(puzzle['train'])]
    _require(any(f['shadow'] for f in tr), 'missing matching-inner-frame witness')
    _require(any(f['bypass'] for f in tr), 'missing mismatching-inner-frame witness')
    _require(any(f['local_discard'] for f in tr), 'missing enclosure/color conjunction witness')
    operands = [(f['initial'], f['contribution'], f['source_area'])
                for d in tr for f in d['frames']]
    _require(len({d for d, n, a in operands}) >= 2, 'initial stack counts do not vary')
    _require({1, 2} <= {n for d, n, a in operands}, 'count progression missing')
    _require(any(n != d for d, n, a in operands), 'doubling shortcut')
    _require(any(n != a for d, n, a in operands), 'area shortcut')
    _require(any(n != nn and a == aa for d, n, a in operands
                 for dd, nn, aa in operands), 'same area/different count evidence missing')
    _require(any(n == nn and a != aa for d, n, a in operands
                 for dd, nn, aa in operands), 'same count/different area evidence missing')
    _require(len({f['rotation'] for f in tr}) >= 2, 'orientation teaching missing')
    te = features[len(puzzle['train']):]
    _require(any(max(f['depth'] for f in d['frames']) >= 3 for d in te), 'test topology shift missing')


def self_test():
    """42 seeds, streaming metrics and metamorphic checks; no saved payloads."""
    shapes, masks, palettes, rotations, topologies, digests = set(), set(), set(), set(), set(), set()
    counts, flags = set(), dict(shadow=0, bypass=0, local_discard=0)
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                digest.update(bytes(a.shape)+a.tobytes()+pair['output'].tobytes())
                f = _features(pair)
                shapes.add(a.shape)
                rotations.add(f['rotation'])
                palettes.add(tuple(sorted(set(int(v) for v in a.flat))))
                topologies.add(tuple(sorted(g['depth'] for g in f['frames'])))
                for g in f['frames']:
                    masks.add(g['token'])
                    counts.add((g['initial'], g['contribution']))
                for key in flags:
                    flags[key] += f[key]
                # Equivariance to arbitrary color permutation and rotation.
                lut = np.roll(np.arange(10, dtype=np.int8), seed % 9+1)
                _require(np.array_equal(solve(lut[np.rot90(a)]),
                                       lut[np.rot90(pair['output'])]), 'equivariance')
        digests.add(digest.digest())
    _require(len(digests) == 42, 'cross-seed duplication')
    _require(len(shapes) >= 30 and rotations == set(range(4)), 'geometry diversity')
    _require(len(masks) >= 5 and len(palettes) >= 15, 'geometry/palette diversity')
    _require(len(topologies) >= 5 and len(counts) >= 8, 'topology/count diversity')
    _require(min(flags.values()) >= 42, 'interaction coverage')
    p, q = generate_puzzle(91, 2, 1), generate_puzzle(91, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p
                 for x, y in zip(p[s], q[s]) for k in x), 'determinism')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            x = generate_puzzle(73, a, b)
            _require(len(x['train']) == a and len(x['test']) == b, 'explicit counts')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (False, 1), (4.0, 1), (4, True), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Perturb the added tally, source highlighting, frame, or ignored object.
    sample = generate_puzzle(0, 2, 1)
    for defect in ('addition', 'highlight', 'wall', 'ignored'):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
        pair = broken['train'][1]
        rot, view, d = _inspect(pair['input'])
        out = np.rot90(pair['output'], rot).copy()
        if defect == 'addition':
            f = next(f for f in d['frames'] if f['contribution'])
            cell = _translated(f['token'], f['y']-f['initial']*f['pitch'], f['x'])[0]
        elif defect == 'highlight':
            cell = next(p for p in d['objects'] if p['owner'] is not None)['cells'][0]
        elif defect == 'wall':
            cell = d['frames'][0]['cells'][0]
        else:
            cell = next(p for p in d['objects'] if p['owner'] is None)['cells'][0]
        out[cell] = (int(out[cell])+1) % 10
        pair['output'] = np.rot90(out, -rot).copy()
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('accepted mutation: '+defect)
    return {'seeds': 42, 'distinct_puzzles': len(digests), 'canvas_shapes': len(shapes),
            'reservoir_masks': len(masks), 'palettes': len(palettes),
            'forest_depth_signatures': len(topologies), 'operand_pairs': len(counts),
            'interaction_pairs': flags}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
