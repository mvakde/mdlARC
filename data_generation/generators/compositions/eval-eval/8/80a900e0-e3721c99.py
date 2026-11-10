"""Topological rail repair. Standard library and NumPy only; no saved state."""
from __future__ import annotations

import hashlib
import json

import numpy as np


DIRS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    neighbors = EIGHT if diagonal else ORTH
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo = [first]
        cells = {first}
        while todo:
            r, c = todo.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    todo.append(p)
        result.append(cells)
    return result


def _mask(cells):
    rc = np.asarray(sorted(cells), dtype=int)
    lo, hi = rc.min(axis=0), rc.max(axis=0) + 1
    a = np.zeros(tuple(hi - lo), dtype=bool)
    a[tuple((rc - lo).T)] = True
    return a


def _holes(cells):
    """Exterior padding prevents canvas edges from becoming imaginary walls."""
    a = np.pad(_mask(cells), 1)
    return [s for s in _components(~a)
            if all(0 < r < a.shape[0] - 1 and 0 < c < a.shape[1] - 1 for r, c in s)]


def _count(cells):
    return len(_holes(cells))


def _cavity_pixels(cells):
    lo = np.asarray(sorted(cells), dtype=int).min(axis=0)
    return {tuple(lo + np.asarray(p) - 1) for hole in _holes(cells) for p in hole}


def _paint(grid, cells, color):
    if cells:
        grid[tuple(np.asarray(sorted(cells)).T)] = color


def _diamond(center, radius):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius}


def _cap(center, radius, direction):
    r, c = center
    vr, vc = direction
    return [(r + vr * (k + 1), c + vc * (radius - k + 1)) for k in range(radius + 1)]


def _rays(cap, direction, shape):
    vr, vc = direction
    paths = []
    for r, c in (cap[0], cap[-1]):
        path = []
        r, c = r + vr, c + vc
        while 0 <= r < shape[0] and 0 <= c < shape[1]:
            path.append((r, c))
            r, c = r + vr, c + vc
        paths.append(path)
    return paths


def _touch(a, b):
    return any((r + dr, c + dc) in b for r, c in a for dr, dc in ((0, 0),) + EIGHT)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _parse_canonical(grid, separator):
    """Parse a candidate whose legend is above one complete separator row."""
    h, w = grid.shape
    assert 4 <= separator <= h - 8
    sep = int(grid[separator, 0])
    work = grid[separator + 1:]
    colors, frequencies = np.unique(work, return_counts=True)
    assert len(colors) == 3
    bg = int(colors[frequencies.argmax()])
    assert frequencies.max() > work.size // 2 and sep != bg
    assert np.all(grid[separator - 1] == bg)
    panel = grid[:separator]
    entries = []
    mapping = {}
    for color in np.unique(panel):
        color = int(color)
        if color == bg:
            continue
        assert color != sep and color not in colors
        parts = _components(panel == color, True)
        assert len(parts) == 1
        k = _count(parts[0])
        assert k not in mapping
        mapping[k] = color
        entries.append((min(parts[0]), k, color))
    assert 1 <= len(mapping) <= 5
    candidates = []
    for struct in (int(c) for c in colors if c != bg):
        query = next(int(c) for c in colors if c not in (bg, struct))
        if sep in (struct, query):
            continue
        pixels = set(map(tuple, np.argwhere(work == struct)))
        emitters = []
        occupied = set()
        overlapping = False
        # A structural color must be exactly a union of complete outlines.
        for radius in (2, 3, 4):
            for r in range(radius, work.shape[0] - radius):
                for c in range(radius, w - radius):
                    diamond = _diamond((r, c), radius)
                    if diamond <= pixels:
                        overlapping |= bool(occupied & diamond)
                        occupied |= diamond
                        emitters.append(((r, c), radius))
        if overlapping or not emitters or occupied != pixels:
            continue
        caps, rails, directions = [], set(), []
        valid = True
        for center, radius in emitters:
            active = 0
            for direction in DIRS:
                cap = _cap(center, radius, direction)
                if not all(0 <= r < work.shape[0] and 0 <= c < w for r, c in cap):
                    continue
                if all(work[p] == query for p in cap):
                    active += 1
                    paths = _rays(cap, direction, work.shape)
                    new = {p for path in paths for p in path}
                    if new & pixels or new & rails:
                        valid = False
                    rails |= new
                    caps.append(set(cap))
                    directions.append(direction)
            if not active:
                valid = False
        if not valid:
            continue
        original = set(map(tuple, np.argwhere(work == query)))
        material = original | rails
        canvas = np.zeros(work.shape, dtype=bool)
        _paint(canvas, material, True)
        objects = _components(canvas, True)
        if len(objects) != len(caps):
            continue
        if not all(sum(cap <= obj for cap in caps) == 1 for obj in objects):
            continue
        if not all(any(cap <= obj for obj in objects) for cap in caps):
            continue
        # A class is made of actual empty regions, not an enclosure containing
        # another device or a reference diamond. Nested objects are excluded.
        if any(_cavity_pixels(obj) & (occupied | (material - obj)) for obj in objects):
            continue
        candidates.append(dict(background=bg, query=query, struct=struct, mapping=mapping,
                               entries=entries, emitters=emitters, caps=caps, rails=rails,
                               original=original, objects=objects, work=work,
                               separator=separator, directions=directions))
    assert len(candidates) == 1
    return candidates[0]


def _parse(grid):
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(grid, rotation)
        for r in range(4, canonical.shape[0] - 7):
            if np.all(canonical[r] == canonical[r, 0]):
                try:
                    parsed = _parse_canonical(canonical, r)
                except (AssertionError, StopIteration):
                    continue
                candidates.append((rotation, canonical, parsed))
    assert len(candidates) == 1, 'input needs exactly one legend/emitter interpretation'
    return candidates[0]


def _apply(canonical, parsed, complete=True):
    out = canonical.copy()
    target = out[parsed['separator'] + 1:]
    if complete:
        objects = parsed['objects']
    else:
        objects = _components(parsed['work'] == parsed['query'], True)
    for cells in objects:
        _paint(target, cells, parsed['mapping'].get(_count(cells), parsed['background']))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    rotation, canonical, parsed = _parse(grid)
    return np.rot90(_apply(canonical, parsed), -rotation).copy()


def _beam(rng, center, radius, direction, shape, bars=None, loop=None, broken=False, stepped=False):
    cap = _cap(center, radius, direction)
    if not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cap):
        return None
    paths = _rays(cap, direction, shape)
    length = min(map(len, paths))
    if length < 3:
        return None
    vr, vc = direction

    def pos(k, t):
        return (cap[0][0] + vr * (k + t), cap[0][1] + vc * (t - k))

    if bars is None:
        bars = int(rng.integers(0, min(3, (length + 1) // 3) + 1))
    if loop is None:
        loop = bool(length >= 5 and rng.random() < .4)
    if bars == 2:
        times = [2, 5]
    elif bars == 1:
        times = [int(rng.integers(3, min(length, 5) + 1))]
    elif bars == 3:
        times = [2, 5, 8]
    else:
        times = []
    if times and max(times) > length:
        return None
    sparse = set(cap)
    for t in times:
        # A transverse path with a one-step bay is still a crosswall, not a tally glyph.
        previous = t
        for k in range(radius + 1):
            current = t + int(stepped and 0 < k < radius)
            sparse.add(pos(k, current))
            if current != previous:
                sparse.add(pos(k, previous))
            previous = current
    if broken:
        choices = [t for t in range(2, length + 1) if all(abs(t - u) >= 2 for u in times)]
        if choices:
            t = int(rng.choice(choices))
            gap = int(rng.integers(1, radius))
            sparse.update(pos(k, t) for k in range(radius + 1) if k != gap)
    if loop:
        t = min(length, max(4, (length + 2) // 2))
        r, c = pos(-2, t)
        # The missing far corner is a diagonal escape only; the cavity stays enclosed.
        ring = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if abs(dr) == 1 or abs(dc) == 1}
        if rng.random() < .5:
            ring.discard((r - vr, c + vc))
        sparse |= ring
    if not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in sparse):
        return None
    rails = {p for path in paths for p in path}
    completed = sparse | rails
    if len(_components(_mask(completed), True)) != 1 or _count(completed) > 5:
        return None
    return dict(sparse=sparse, rails=rails, complete=completed, holes=_count(completed),
                bars=bars, loop=loop, stepped=stepped, direction=direction, radius=radius)


def _geometry(rng, shape, teaching=None):
    """Finite free placement. Teaching cases carry explicit structural contrasts."""
    desired = 1 if teaching is not None else int(rng.integers(1, 4))
    all_beams, all_diamonds, emitters = [], set(), []
    for index in range(desired):
        accepted = False
        for _ in range(100):
            radius = int(rng.integers(2, 5))
            if teaching is not None:
                margin = radius + 6
                if min(shape) <= 2 * margin:
                    continue
                center = tuple(int(rng.integers(margin, n - margin)) for n in shape)
                dirs = [DIRS[i] for i in ((0, 2) if teaching == 0 else (1, 3, 0))]
            else:
                center = tuple(int(rng.integers(radius, n - radius)) for n in shape)
                count = int(rng.integers(1, (5 if desired == 1 else 3)))
                dirs = [DIRS[int(i)] for i in rng.choice(4, count, replace=False)]
            diamond = _diamond(center, radius)
            if _touch(diamond, all_diamonds):
                continue
            beams = []
            for j, direction in enumerate(dirs):
                if teaching == 0:
                    beam = _beam(rng, center, radius, direction, shape, bars=1, loop=j == 1)
                elif teaching == 1:
                    beam = _beam(rng, center, radius, direction, shape, bars=(j + 1 if j < 2 else 0),
                                 loop=False, broken=j == 0, stepped=False)
                else:
                    beam = _beam(rng, center, radius, direction, shape,
                                 broken=bool(rng.random() < .65), stepped=bool(rng.random() < .6))
                if beam is None:
                    break
                beams.append(beam)
            if len(beams) != len(dirs):
                continue
            expected = [1, 2] if teaching == 0 else [0, 1, 2]
            if teaching is not None and sorted(b['holes'] for b in beams) != expected:
                continue
            existing = set().union(*(b['complete'] for b in all_beams)) if all_beams else set()
            new = set()
            good = True
            for b in beams:
                if b['complete'] & (diamond | all_diamonds) or _touch(b['complete'], existing | new):
                    good = False
                new |= b['complete']
            if not good or new & all_diamonds or diamond & existing:
                continue
            all_beams.extend(beams)
            all_diamonds |= diamond
            emitters.append((center, radius))
            accepted = True
            break
        if not accepted and teaching is not None:
            return None
    if not all_beams:
        return None
    return all_diamonds, all_beams, emitters


def _glyph(k, color, bg, rng):
    a = np.full((3, max(3, 2 * k + 1)), color, dtype=np.int8)
    if k:
        a[1, 1:2 * k:2] = bg
    if rng.random() < .3:
        a = np.repeat(a, 2, axis=0)
    return a


def _make_pair(rng, teaching, rotation, test=False):
    for _ in range(70):
        # Keys are selected after geometry, but decoded only from visible glyphs.
        shape = (int(rng.integers(22, 25)) if teaching is not None else int(rng.integers(15, 25)),
                 int(rng.integers(23, 31)))
        result = _geometry(rng, shape, teaching)
        if result is None:
            continue
        diamonds, beams, _ = result
        counts = sorted({b['holes'] for b in beams})
        if teaching == 0:
            keys = [1, 2]
        elif teaching == 1:
            keys = [1, 2, int(rng.choice([3, 4]))]
        else:
            keep = int(rng.choice([k for k in counts if k > 0] or counts))
            keys = [keep]
            pool = [k for k in range(6) if k != keep]
            keys += [int(k) for k in rng.choice(pool, int(rng.integers(0, 3)), replace=False)]
        # Non-sorted spatial key order is visible training evidence against rank codes.
        rng.shuffle(keys)
        roles = list(map(int, rng.permutation(10)))
        bg, query, struct, sep = roles[:4]
        mapping = dict(zip(keys, roles[4:]))
        glyphs = [_glyph(k, mapping[k], bg, rng) for k in keys]
        width = max(shape[1], sum(g.shape[1] for g in glyphs) + len(glyphs) + 1)
        panel_height = max(g.shape[0] for g in glyphs) + 2
        if width > 30 or panel_height + 1 + shape[0] > 30:
            continue
        # Geometry retains its sampled work width, avoiding any latent ray endpoints.
        if width != shape[1]:
            continue
        grid = np.full((panel_height + 1 + shape[0], width), bg, dtype=np.int8)
        grid[panel_height] = sep
        offset = int(rng.integers(width - sum(g.shape[1] + 1 for g in glyphs)))
        for g in glyphs:
            r = int(rng.integers(1, panel_height - g.shape[0]))
            patch = grid[r:r + g.shape[0], offset:offset + g.shape[1]]
            patch[:] = g
            offset += g.shape[1] + 1
        work = grid[panel_height + 1:]
        _paint(work, diamonds, struct)
        for b in beams:
            _paint(work, b['sparse'], query)
        # Construct labels from the planned contours; solve reconstructs from pixels.
        out = grid.copy()
        target = out[panel_height + 1:]
        for b in beams:
            _paint(target, b['complete'], mapping.get(b['holes'], bg))
        pair = {'input':np.rot90(grid, rotation).copy(), 'output':np.rot90(out, rotation).copy()}
        try:
            turn, canonical, parsed = _parse(pair['input'])
            assert np.array_equal(np.rot90(_apply(canonical, parsed), -turn), pair['output'])
            assert len(parsed['objects']) == len(beams)
            if test:
                # Test recombines beyond the two minimal straight-wall demonstrations.
                assert any(b['holes'] in mapping and b['holes'] > 0 for b in beams)
                assert any(b['stepped'] and b['bars'] and b['holes'] in mapping for b in beams) or len(beams) >= 3
            assert np.any(pair['input'] != pair['output'])
        except AssertionError:
            continue
        return pair
    raise RuntimeError('bounded scene search exhausted')


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('counts must be integers')
        if not low <= value <= high:
            raise ValueError('count outside legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    puzzle = {'train':[], 'test':[]}
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            teaching = i if split == 'train' and i < 2 else None
            rot = rotation if teaching is not None else int(rng.integers(4))
            for _ in range(12):
                pair = _make_pair(rng, teaching, rot, test=split == 'test')
                signature = (pair['input'].shape, pair['input'].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded unique scene search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_directions, classes = set(), set(), set()
    erased = topology_created = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            signature = (inp.shape, inp.tobytes())
            assert signature not in seen
            seen.add(signature)
            rotation, canonical, p = _parse(inp)
            assert np.array_equal(np.rot90(_apply(canonical, p), -rotation), out)
            cout = np.rot90(out, rotation)
            work_out = cout[p['separator'] + 1:]
            # Only completed material may change; markers and all cavity cells survive.
            material = set().union(*p['objects'])
            allowed = np.zeros(p['work'].shape, dtype=bool)
            _paint(allowed, material, True)
            assert np.array_equal(cout[:p['separator'] + 1], canonical[:p['separator'] + 1])
            assert np.array_equal(work_out[~allowed], p['work'][~allowed])
            for obj in p['objects']:
                k = _count(obj)
                color = p['mapping'].get(k, p['background'])
                assert all(work_out[cell] == color for cell in obj)
                if split == 'train':
                    if k in p['mapping']:
                        classes.add(k)
                        for cap, direction in zip(p['caps'], p['directions']):
                            if cap <= obj and len((obj & p['rails']) - p['original']) >= 4:
                                train_directions.add(direction)
                        fragments = _components(_mask(obj & p['original']), True)
                        before = sum(_count(fragment) for fragment in fragments)
                        topology_created |= k > before
                    else:
                        erased = True
            if split == 'train':
                # Ablation: classifying fragments before assembly changes the answer.
                assert not np.array_equal(_apply(canonical, p, complete=False), cout)
    assert len(classes) >= 2 and erased and topology_created and train_directions == set(DIRS)


def self_test():
    """Bounded 42-seed sweep, topology fixtures, determinism, and negative checks."""
    assert _count({(0, 1), (1, 0), (1, 2), (2, 1)}) == 1
    assert _count({(0, 1), (1, 0), (2, 1)}) == 0
    fingerprints, shapes, radii, masks, keysets, populations, counts, areas = set(), set(), set(), set(), set(), set(), set(), set()
    varied = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        local_shapes = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                fingerprint.update(bytes(inp.shape) + inp.tobytes() + pair['output'].tobytes())
                shapes.add(inp.shape)
                local_shapes.add(inp.shape)
                _, canonical, p = _parse(inp)
                radii.update(r for _, r in p['emitters'])
                for center, radius in p['emitters']:
                    masks.add(tuple(direction for direction in DIRS
                                    if set(_cap(center, radius, direction)) in p['caps']))
                keysets.add(tuple(sorted(p['mapping'])))
                populations.add(len(p['emitters']))
                for obj in p['objects']:
                    holes = _holes(obj)
                    counts.add(len(holes))
                    areas.update(map(len, holes))
                assert np.array_equal(solve(inp.copy()), pair['output'])
                # Omitting B leaves neutral rails and cannot reproduce any colored answer.
                neutral = canonical.copy()
                _paint(neutral[p['separator'] + 1:], p['rails'], p['query'])
                assert not np.array_equal(neutral, _apply(canonical, p))
        fingerprints.add(fingerprint.hexdigest())
        varied += len(local_shapes) > 1
    assert len(fingerprints) == 42 and len(shapes) >= 25 and varied == 42
    assert radii == {2, 3, 4} and len(populations) >= 2
    assert len(masks) >= 8 and len(keysets) >= 12 and len(counts) >= 4 and len(areas) >= 12
    p = generate_puzzle(123, 2, 1)
    q = generate_puzzle(123, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            _counts(a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s:[{k:v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return {'seeds':42, 'puzzles':len(fingerprints), 'shapes':len(shapes),
            'radii':sorted(radii), 'emitter_counts':sorted(populations), 'cap_subsets':len(masks),
            'key_subsets':len(keysets), 'cavity_counts':sorted(counts), 'cavity_areas':len(areas)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
