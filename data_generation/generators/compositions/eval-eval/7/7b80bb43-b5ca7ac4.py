"""Mended bulkhead docks. Standard library and NumPy; no external evidence at runtime.

Wall reconstruction adapts the visible geometric criterion in parent A v2;
packing uses the stable-order interpretation of parent B v2. Neither parent
generator is imported or invoked. Generation retains one puzzle at a time.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=True)


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong(mask):
    result = np.zeros_like(mask)
    for m, s in ((mask, result), (mask.T, result.T)):
        if m.shape[1] < 3:
            continue
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for offset in range(3):
            s[:, offset:offset + triples.shape[1]] |= triples
    return result


def _gaps(mask):
    result = []
    for m, vertical in ((mask, False), (mask.T, True)):
        if m.shape[1] < 5:
            continue
        bracket = m[:, :-4] & m[:, 4:]
        bracket &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(bracket):
            y, x = int(y), int(x)
            result.append((x, y, 1, 0) if vertical else (y, x, 0, 1))
    return result


def _repair(mask):
    """Simultaneous, input-only repair; diagonal evidence is never iterated."""
    strong = _strong(mask)
    added, anchors = np.zeros_like(mask), np.zeros_like(mask)
    evidence = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                cells = [(ar+k*sr, ac+k*sc) for k in (1, 2)]
                if (all(_at(mask, rr, cc) and not strong[rr, cc] for rr, cc in cells)
                        and not _at(mask, ar+3*sr, ac+3*sc)):
                    matches.append(cells)
        if matches:
            if len(matches) != 1:
                raise ValueError("ambiguous wall repair")
            evidence.append(((r, c, dr, dc), matches[0]))
            for k in (1, 2, 3):
                added[r+k*dr, c+k*dc] = True
            anchors[r, c] = anchors[r+4*dr, c+4*dc] = True
    weak = mask & ~_strong(mask | added) & ~anchors
    diagonal = np.zeros_like(mask)
    for dc in (-1, 1):
        a = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        b = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        linked = a & b
        if dc == 1:
            diagonal[:-1, :-1] |= linked
            diagonal[1:, 1:] |= linked
        else:
            diagonal[:-1, 1:] |= linked
            diagonal[1:, :-1] |= linked
    clean = (mask & ~diagonal) | added
    return clean, added, mask & ~clean, evidence


def _band(a, b):
    return max(a['r'], b['r']) < min(a['r']+a['h'], b['r']+b['h'])


def _overlap(a, b):
    return _band(a, b) and max(a['c'], b['c']) < min(a['c']+a['w'], b['c']+b['w'])


def _decode(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg]*2 <= grid.size:
        raise ValueError("background must occupy a strict majority")
    axes = []
    for transpose in (False, True):
        g = grid.T if transpose else grid
        a, b = int(g[0, 0]), int(g[0, -1])
        if (a != b and bg not in (a, b)
                and np.all(g[:, 0] == a) and np.all(g[:, -1] == b)):
            axes.append((transpose, g, a, b))
    if len(axes) != 1:
        raise ValueError("two opposite dock edges must identify one axis")
    transpose, g, low, high = axes[0]
    covered = g == bg
    covered[:, 0] = covered[:, -1] = True
    objects = []
    height, width = g.shape
    for r in range(height):
        for c in range(1, width-1):
            frame = int(g[r, c])
            if covered[r, c] or frame not in (low, high):
                continue
            if r+2 >= height or c+2 >= width-1:
                raise ValueError("incomplete cargo frame")
            bottom, right = r+2, c+2
            while bottom < height and g[bottom, c+1] != frame:
                bottom += 1
            while right < width-1 and g[r+1, right] != frame:
                right += 1
            if bottom == height or right >= width-1:
                raise ValueError("open cargo frame")
            patch = g[r:bottom+1, c:right+1]
            if (not np.all(patch[0] == frame) or not np.all(patch[-1] == frame)
                    or not np.all(patch[:, 0] == frame) or not np.all(patch[:, -1] == frame)
                    or np.any(patch[1:-1, 1:-1] == frame)
                    or np.any(patch[1:-1, 1:-1] == bg)
                    or np.any(covered[r:bottom+1, c:right+1])):
                raise ValueError("cargo is not a distinct complete rectangle")
            covered[r:bottom+1, c:right+1] = True
            objects.append(dict(r=r, c=c, h=bottom-r+1, w=right-c+1,
                                d=int(frame == high), patch=patch.copy()))
    remaining = ~covered
    colors = np.unique(g[remaining])
    if not objects or len(colors) != 1 or colors[0] in (bg, low, high):
        raise ValueError("expected cargo and one distinct wall color")
    wall_color = int(colors[0])
    if any(np.any(o['patch'] == wall_color) for o in objects):
        raise ValueError("wall color cannot be cargo payload")
    return transpose, g, bg, low, high, wall_color, remaining, objects


def _pack(objects, walls):
    """Nearest reachable static barrier, then all stable same-class predecessors."""
    result = [dict(o) for o in objects]
    width = walls.shape[1]
    for d in (0, 1):
        order = sorted((i for i, o in enumerate(objects) if o['d'] == d),
                       key=lambda i: ((objects[i]['c'] if d == 0
                                       else -objects[i]['c']-objects[i]['w']),
                                      objects[i]['r']))
        settled = []
        for i in order:
            o = result[i]
            occupied_columns = np.flatnonzero(walls[o['r']:o['r']+o['h']].any(axis=0))
            if d == 0:
                barriers = occupied_columns[occupied_columns < o['c']]
                target = int(barriers[-1])+1 if len(barriers) else 1
                o['c'] = max([target] + [p['c']+p['w'] for p in settled if _band(o, p)])
                if o['c'] > objects[i]['c']:
                    raise ValueError("cargo would move away from its dock")
            else:
                barriers = occupied_columns[occupied_columns >= o['c']+o['w']]
                target = (int(barriers[0]) if len(barriers) else width-1)-o['w']
                o['c'] = min([target] + [p['c']-o['w'] for p in settled if _band(o, p)])
                if o['c'] < objects[i]['c']:
                    raise ValueError("cargo would move away from its dock")
            if (not 1 <= o['c'] <= width-1-o['w']
                    or walls[o['r']:o['r']+o['h'], o['c']:o['c']+o['w']].any()):
                raise ValueError("cargo intersects a wall or dock")
            settled.append(o)
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i+1:]):
        raise ValueError("independently packed classes overlap")
    return result


def _simulate(objects, walls):
    """Separate unit-step check: cargo cannot jump walls or its own class."""
    result = [dict(o) for o in objects]
    height, width = walls.shape
    for d in (0, 1):
        group = [o for o in result if o['d'] == d]
        occupied = walls.copy()
        occupied[:, 0] = occupied[:, -1] = True
        for o in group:
            occupied[o['r']:o['r']+o['h'], o['c']:o['c']+o['w']] = True
        for _ in range(width*len(group)+1):
            moved = False
            for o in group:
                r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
                occupied[r:r+h, c:c+w] = False
                nc = c + (1 if d else -1)
                if 1 <= nc <= width-1-w and not occupied[r:r+h, nc:nc+w].any():
                    o['c'] = nc
                    moved = True
                occupied[r:r+h, o['c']:o['c']+w] = True
            if not moved:
                break
        else:
            raise AssertionError("physical packing failed to converge")
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i+1:]):
        raise ValueError("physical final groups collide")
    return result


def _render(shape, bg, low, high, line, walls, objects):
    grid = np.full(shape, bg, dtype=np.int8)
    grid[walls] = line
    grid[:, 0], grid[:, -1] = low, high
    occupied = walls.copy()
    occupied[:, 0] = occupied[:, -1] = True
    for o in objects:
        r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
        if occupied[r:r+h, c:c+w].any():
            raise ValueError("render collision")
        grid[r:r+h, c:c+w] = o['patch']
        occupied[r:r+h, c:c+w] = True
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    transpose, g, bg, low, high, line, noisy, objects = _decode(input_grid)
    walls = _repair(noisy)[0]
    out = _render(g.shape, bg, low, high, line, walls, _pack(objects, walls))
    return out.T.copy() if transpose else out


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _bulkheads(rng, h, w, count, teaching):
    clean = np.zeros((h, w), dtype=bool)
    if teaching:
        columns = [int(rng.integers(14 if teaching == 'chain' else 10, w-7))]
    else:
        # Spacing protects the two-cell deviations and permits variable chambers.
        columns = list(map(int, rng.choice(np.arange(6, w-5, 6), count, replace=False)))
    repairs, openings = [], []
    for col in columns:
        top = int(rng.integers(0, 3))
        bottom = int(rng.integers(h-3, h))
        clean[top:bottom+1, col] = True
        # Orthogonal stubs make each bulkhead an A-like multi-axis scaffold.
        for row in (top, bottom):
            side = int(_choose(rng, (-1, 1)))
            length = int(rng.integers(1, 4))
            a, b = sorted((col, col+side*length))
            clean[row, a:b+1] = True
        first = int(rng.integers(top+3, min(top+6, bottom-9)))
        second = int(rng.integers(first+6, bottom-3))
        if not teaching and rng.integers(2):
            first, second = second, first
        clean[second:second+3, col] = False
        repairs.append((first, col))
        openings.append((second, col))
    noisy = clean.copy()
    for row, col in repairs:
        noisy[row:row+3, col] = False
        endpoint = int(_choose(rng, (0, 4)))
        anchor = row-1+endpoint
        toward = 1 if endpoint == 0 else -1
        side = int(_choose(rng, (-1, 1)))
        for k in (1, 2):
            noisy[anchor+k*toward, col+k*side] = True
    # Optional detached oblique clutter, accepted only under full reconstruction.
    for _ in range(int(rng.integers(3))):
        for _ in range(20):
            r, c = int(rng.integers(h-2)), int(rng.integers(2, w-4))
            dc = int(_choose(rng, (-1, 1)))
            length = int(_choose(rng, (2, 3)))
            cells = [(r+k, c+k*dc) for k in range(length)]
            if any(clean[rr, cc] or noisy[rr, cc] for rr, cc in cells):
                continue
            trial = noisy.copy()
            for rr, cc in cells:
                trial[rr, cc] = True
            try:
                if np.array_equal(_repair(trial)[0], clean):
                    noisy = trial
                    break
            except ValueError:
                pass
    if not np.array_equal(_repair(noisy)[0], clean):
        raise ValueError("bulkheads have accidental interacting evidence")
    return noisy, clean, repairs, openings


def _appearance(rng, objects, p, shared=False):
    for i, o in enumerate(objects):
        frame = p['high'] if o['d'] else p['low']
        h, w = o['h'], o['w']
        palette = p['payloads']
        interior = np.full((h-2, w-2), _choose(rng, palette), dtype=np.int8)
        if shared and i < 2:
            interior[:] = palette[0]
        elif interior.size > 1 and rng.random() < p['patterns']:
            a, b = map(int, rng.choice(palette, 2, replace=False))
            rr, cc = np.indices(interior.shape)
            pattern = int(rng.integers(4))
            bits = rr % 2 if pattern == 0 else cc % 2 if pattern == 1 else (rr+cc) % 2
            if pattern == 3:
                bits = rng.integers(2, size=interior.shape)
            interior[:] = np.where(bits, a, b)
        patch = np.full((h, w), frame, dtype=np.int8)
        patch[1:-1, 1:-1] = interior
        o['patch'] = patch


def _make_pair(rng, p, purpose, test):
    for attempt in range(80):
        h = int(rng.integers(20, 31))
        w = int(rng.integers(25 if test and p['shift'] == 'bulkheads' else 23, 31))
        count = (3 if test and p['shift'] == 'bulkheads'
                 else 1 if purpose else int(rng.integers(1, 3)))
        try:
            noisy, clean, repairs, openings = _bulkheads(rng, h, w, count, purpose)
            objects = []
            if purpose == 'gate':
                row, col = repairs[0]
                gap, _ = openings[0]
                # Same contents under opposite frames; a closed/open/tight contrast.
                objects = [dict(r=row, c=col+4, h=3, w=3, d=0),
                           dict(r=row, c=2, h=3, w=3, d=1),
                           dict(r=gap, c=col+4, h=3, w=3, d=0),
                           dict(r=gap-1, c=2, h=4, w=3, d=1)]
            elif purpose == 'chain':
                col = repairs[0][1]
                ow = 3
                top = int(rng.integers(3, 8))
                first = int(rng.integers(2, col-3*ow))
                objects = [dict(r=top+2*i, c=first+3*i, h=3, w=ow, d=0)
                           for i in range(3)]
                # Reflect the complete canonical scene for a high-directed chain.
                if rng.integers(2):
                    noisy, clean = noisy[:, ::-1].copy(), clean[:, ::-1].copy()
                    objects = [dict(o, c=w-o['c']-o['w'], d=1) for o in objects]
            else:
                requested = int(rng.integers(3, 10))
                big = test and p['shift'] == 'cargo'
                for _ in range(requested):
                    for _ in range(60):
                        oh = int(rng.integers(3, 6 if big else 5))
                        ow = int(rng.integers(5, 7) if big else rng.integers(3, 5))
                        o = dict(r=int(rng.integers(h-oh+1)), c=int(rng.integers(1, w-ow)),
                                 h=oh, w=ow, d=int(rng.integers(2)))
                        if ((noisy | clean)[o['r']:o['r']+oh, o['c']:o['c']+ow].any()
                                or any(_overlap(o, q) for q in objects)):
                            continue
                        try:
                            _pack(objects+[o], clean)
                        except ValueError:
                            continue
                        objects.append(o)
                        break
            if not objects:
                continue
            _appearance(rng, objects, p, purpose == 'gate')
            settled = _pack(objects, clean)
            inp = _render((h, w), p['bg'], p['low'], p['high'], p['line'], noisy, objects)
            out = _render((h, w), p['bg'], p['low'], p['high'], p['line'], clean, settled)
            transpose = p['axis'] if purpose else bool(rng.integers(2))
            # Transposition is visible from the dock edges, never latent.
            if transpose:
                inp, out = inp.T.copy(), out.T.copy()
            _analyze(inp, out)
            return {'input': inp, 'output': out}
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    n = _integer(num_train, 'num_train', 2, 10)
    t = _integer(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    p = dict(bg=palette[0], line=palette[1], low=palette[2], high=palette[3],
             payloads=palette[4:], axis=bool(rng.integers(2)),
             shift=str(_choose(rng, ('bulkheads', 'cargo'))),
             patterns=float(_choose(rng, (0.0, 0.4, 0.9))))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for i in range(count):
            purpose = ('gate' if i == 0 else 'chain' if i == 1 else '') if split == 'train' else ''
            for _ in range(20):
                pair = _make_pair(rng, p, purpose, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded unique-input sampling exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _destinations(objects):
    return tuple(o['c'] for o in objects)


def _edge_first(objects, walls):
    """A deliberately wrong competitor that jumps same-class blockers."""
    result = [dict(o) for o in objects]
    width = walls.shape[1]
    for d in (0, 1):
        order = sorted((i for i, o in enumerate(objects) if o['d'] == d),
                       key=lambda i: objects[i]['c'] if d == 0 else -objects[i]['c']-objects[i]['w'])
        settled = []
        for i in order:
            source = objects[i]
            choices = range(1, source['c']+1) if d == 0 else range(width-1-source['w'], source['c']-1, -1)
            for c in choices:
                o = dict(source, c=c)
                # Keep the correct static chamber; only the overtaking differs.
                lo, hi = min(c, source['c']), max(c, source['c'])+o['w']
                if walls[o['r']:o['r']+o['h'], lo:hi].any():
                    continue
                if not any(_overlap(o, q) for q in settled):
                    result[i] = o
                    settled.append(o)
                    break
    return result


def _analyze(inp, out):
    transpose, g, bg, low, high, line, noisy, objects = _decode(inp)
    walls, added, removed, evidence = _repair(noisy)
    if not evidence or not added.any() or not removed.any():
        raise AssertionError('each scene must exhibit wall repair and deletion')
    expected_objects = _simulate(objects, walls)
    expected = _render(g.shape, bg, low, high, line, walls, expected_objects)
    actual = _grid(out).T if transpose else _grid(out)
    assert np.array_equal(expected, actual), 'wrong repair, position, frame, or payload'
    assert _destinations(_pack(objects, walls)) == _destinations(expected_objects)
    # Distinguish closing a passage from merely erasing its displaced fragments.
    no_closure = walls & ~added
    def differs(other_walls):
        try:
            return _destinations(_pack(objects, other_walls)) != _destinations(expected_objects)
        except ValueError:
            return True
    all_closed = walls.copy()
    for r, c, dr, dc in _gaps(walls):
        for k in (1, 2, 3):
            all_closed[r+k*dr, c+k*dc] = True
    passage, tight = False, False
    for r, c, dr, dc in _gaps(walls):
        if dc:  # Generator apertures are perpendicular to canonical motion.
            continue
        for source, dest in zip(objects, expected_objects):
            lo, hi = source['r'], source['r']+source['h']
            spans_gap = max(lo, r+1) < min(hi, r+4)
            crossing = ((source['c'] > c and dest['c']+dest['w'] <= c)
                        or (source['c']+source['w'] <= c and dest['c'] > c))
            passage |= crossing and r+1 <= lo and hi <= r+4
            stops_beside = dest['c'] == c+1 or dest['c']+dest['w'] == c
            tight |= spans_gap and source['h'] > 3 and stops_beside
    return dict(shape=tuple(np.asarray(inp).shape), axis=int(transpose), roles=(bg, line, low, high),
                count=len(objects), footprints=tuple(sorted({(o['h'], o['w']) for o in objects})),
                repairs=len(evidence), apertures=len(_gaps(walls)),
                wall_columns=int(np.count_nonzero(walls.any(axis=0))),
                closure_changes_cargo=differs(no_closure),
                fill_all_changes_cargo=differs(all_closed),
                motion_changes_cargo=any(a['c'] != b['c'] for a, b in zip(objects, expected_objects)),
                stable_witness=_destinations(_edge_first(objects, walls)) != _destinations(expected_objects),
                directions=tuple(sorted({o['d'] for o in objects})),
                moving_directions=tuple(sorted({a['d'] for a, b in zip(objects, expected_objects)
                                                if a['c'] != b['c']})),
                passage_witness=bool(passage), clearance_witness=bool(tight),
                pattern_count=sum(len(np.unique(o['patch'][1:-1, 1:-1])) > 1 for o in objects),
                travel=tuple(abs(a['c']-b['c']) for a, b in zip(objects, expected_objects)),
                shared_payload=any(a['d'] != b['d'] and a['patch'].shape == b['patch'].shape
                                   and np.array_equal(a['patch'][1:-1, 1:-1], b['patch'][1:-1, 1:-1])
                                   for a in objects for b in objects))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train, roles = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _analyze(inp, out)
            roles.add(info['roles'])
            if split == 'train':
                train.append(info)
    assert len(roles) == 1, 'puzzle palette roles must be consistent'
    assert any(i['closure_changes_cargo'] for i in train), 'A must affect cargo destinations'
    assert any(i['motion_changes_cargo'] for i in train), 'B must cause cargo movement'
    assert any(i['fill_all_changes_cargo'] for i in train), 'teach unsupported apertures'
    assert any(i['stable_witness'] for i in train), 'teach no same-class overtaking'
    assert any(i['shared_payload'] for i in train), 'separate direction from payload'
    assert set(d for i in train for d in i['directions']) == {0, 1}
    assert set(d for i in train for d in i['moving_directions']) == {0, 1}
    assert any(i['passage_witness'] and i['clearance_witness'] for i in train), 'teach full-footprint clearance'


def self_test():
    """42 seeds streamed in memory; no files, corpora, or execution traces saved."""
    signatures, shapes, roles, counts, footprints, axes = set(), set(), set(), set(), set(), set()
    repairs, patterns, walls, travels = set(), 0, set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _analyze(pair['input'], pair['output'])
                for key in ('input', 'output'):
                    digest.update(str(pair[key].shape).encode())
                    digest.update(pair[key].tobytes())
                shapes.add(info['shape'])
                roles.add(info['roles'])
                counts.add(info['count'])
                footprints.update(info['footprints'])
                axes.add(info['axis'])
                repairs.add(info['repairs'])
                walls.add(info['wall_columns'])
                travels.update(info['travel'])
                patterns += info['pattern_count']
                assert np.array_equal(solve(pair['input']), pair['output'])
                # All rotations/reflections are visibly labeled by their docks.
                for k in range(4):
                    for flip in (False, True):
                        a, b = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                        if flip:
                            a, b = a[:, ::-1], b[:, ::-1]
                        assert np.array_equal(solve(a), b)
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 25 and len(roles) >= 30
    assert len(counts) >= 5 and len(footprints) >= 8 and axes == {0, 1}
    assert len(repairs) >= 3 and len(walls) >= 4 and patterns > 0
    assert 0 in travels and max(travels) >= 10
    for n in range(2, 11):
        for t in range(1, min(n-1, 4)+1):
            validate_generated_puzzle(generate_puzzle(100+n*5+t, n, t))
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(7), generate_puzzle(7)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # Both major failure modes must be rejected, including partial repair.
    for mutation in ('identity', 'omit_repair', 'damage_payload', 'overtake'):
        p = generate_puzzle(0, 2, 1)
        pair = p['train'][1 if mutation == 'overtake' else 0]
        tr, g, bg, low, high, line, noisy, objects = _decode(pair['input'])
        clean, added, _, evidence = _repair(noisy)
        if mutation == 'identity':
            bad = g.copy()
        elif mutation == 'omit_repair':
            bad = (pair['output'].T if tr else pair['output']).copy()
            (r, c, dr, dc), _ = evidence[0]
            for k in (1, 2, 3):
                bad[r+k*dr, c+k*dc] = bg
        elif mutation == 'damage_payload':
            bad = (pair['output'].T if tr else pair['output']).copy()
            o = _pack(objects, clean)[0]
            bad[o['r']+1, o['c']+1] = bg
        else:
            bad = _render(g.shape, bg, low, high, line, clean, _edge_first(objects, clean))
        pair['output'] = bad.T.copy() if tr else bad
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f'validator accepted {mutation}')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
