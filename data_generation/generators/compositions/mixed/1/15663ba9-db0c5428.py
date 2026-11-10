"""Literal implementation of the saved 3x3 contour-ring proposal.

IMPORTANT: a valid 3x3 contour has no concave bends. This module does not
claim to realize the proposal's full two-label contribution from parent A.
See composition.md and scratch/contour_feasibility.json for the obstruction.
No files, execution traces, or per-seed payloads are written by this module.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


CONVEX = 4
CONCAVE = 2
DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if (r, c) != (0, 0))
SIDES = ((0, 1), (1, 2), (2, 1), (1, 0))
INPUT_COLORS = tuple(c for c in range(10) if c not in (CONVEX, CONCAVE))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors must be 0..9")
    return a


def _classes(mask):
    """Trace a simple pixel cycle; signed turns are relative to its interior.

    -1 = unoccupied, 0 = straight, 1 = convex, 2 = concave. The helper is
    general, but the composition's 3x3 domain makes class 2 unreachable.
    """
    cells = set(zip(*np.nonzero(mask)))
    _require(len(cells) >= 8, "a contour must enclose an interior cell")
    adjacent = {}
    for r, c in sorted(cells):
        adjacent[r, c] = sorted(q for q in ((r-1, c), (r+1, c),
                                           (r, c-1), (r, c+1)) if q in cells)
        _require(len(adjacent[r, c]) == 2, "contour branches or has an endpoint")
    start = min(cells)
    path = [start]
    seen = {start}
    previous, current = start, adjacent[start][0]
    while current != start:
        _require(current not in seen, "contour is not a simple cycle")
        path.append(current)
        seen.add(current)
        following = next(q for q in adjacent[current] if q != previous)
        previous, current = current, following
    _require(seen == cells, "tile contains multiple cycles")
    blocked = np.pad(mask.astype(bool), 1)
    outside = np.zeros(blocked.shape, dtype=bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]
                    and not blocked[nr, nc] and not outside[nr, nc]):
                outside[nr, nc] = True
                stack.append((nr, nc))
    _require(bool(np.any(~blocked & ~outside)), "contour has no bounded interior")
    area = sum(c * path[(i+1) % len(path)][0]
               - path[(i+1) % len(path)][1] * r
               for i, (r, c) in enumerate(path))
    _require(area != 0, "zero-area contour")
    result = np.full(mask.shape, -1, dtype=np.int8)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i-1]
        nr, nc = path[(i+1) % len(path)]
        turn = (c-pc)*(nr-r) - (r-pr)*(nc-c)
        result[r, c] = 0 if turn == 0 else (1 if turn*area > 0 else 2)
    _require(int(np.count_nonzero(result == 1))
             - int(np.count_nonzero(result == 2)) == 4,
             "invalid total turning angle")
    return result


def _analyze(value):
    a = _grid(value)
    _require(min(a.shape) >= 15, "the complete expansion needs a 15x15 footprint")
    _require(not np.any((a == CONVEX) | (a == CONCAVE)),
             "corner-label colors are reserved and absent from inputs")
    bg = int(a[0, 0])
    occupied = np.argwhere(a != bg)
    _require(len(occupied) == 64, "expected eight eight-cell contour tiles")
    top, left = map(int, occupied.min(axis=0))
    bottom, right = map(int, occupied.max(axis=0))
    _require((bottom-top, right-left) == (8, 8), "expected a 9x9 source ring")
    _require(3 <= top <= a.shape[0]-12 and 3 <= left <= a.shape[1]-12,
             "outward copies would cross the canvas boundary")
    mask = np.zeros(a.shape, dtype=bool)
    for dr, dc in DIRECTIONS:
        r, c = top+3*(dr+1), left+3*(dc+1)
        mask[r:r+3, c:c+3] = True
        mask[r+1, c+1] = False
    _require(np.array_equal(a != bg, mask), "malformed contour ring or noise")
    colors, counts = np.unique(a[mask], return_counts=True)
    carrier = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > 32, "carrier must be a strict foreground majority")
    tiles = []
    for dr, dc in DIRECTIONS:
        r, c = top+3*(dr+1), left+3*(dc+1)
        tile = a[r:r+3, c:c+3]
        _require(bool(np.all(tile[np.ix_([0, 2], [0, 2])] == carrier)),
                 "all contour bends must initially use the shared carrier")
        tiles.append(tile)
    return dict(background=bg, carrier=carrier, top=top, left=left, tiles=tiles)


def _recolor(tile, background):
    classes = _classes(tile != background)
    result = tile.copy()
    result[classes == 1] = CONVEX
    result[classes == 2] = CONCAVE
    return result


def _finish(a, info, label=True, expand=True):
    """Compose at tile level; no parent whole-puzzle generator is invoked."""
    out = a.copy()
    r, c = info['top']+3, info['left']+3
    center = np.full((3, 3), info['carrier'], dtype=a.dtype)
    for (dr, dc), original in zip(DIRECTIONS, info['tiles']):
        tile = _recolor(original, info['background']) if label else original
        sr, sc = r+3*dr, c+3*dc
        out[sr:sr+3, sc:sc+3] = tile
        if expand:
            center[1+dr, 1+dc] = tile[1-dr, 1-dc]
            tr, tc = r+6*dr, c+6*dc
            out[tr:tr+3, tc:tc+3] = tile[::-1, ::-1]
    if expand:
        out[r:r+3, c:c+3] = center
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read geometry, label bends, and sample/echo the labeled tile contents."""
    a = _grid(input_grid)
    return _finish(a, _analyze(a))


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                    ('num_test', num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_))
                 and isinstance(value, (int, np.integer))
                 and low <= int(value) <= high,
                 f'{name} must be an integer in {low}..{high}')
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _new_input(rng, background, profile):
    h, w = map(int, rng.integers(15, 31, size=2))
    top = int(rng.integers(3, h-11))
    left = int(rng.integers(3, w-11))
    palette = [int(v) for v in rng.permutation([c for c in INPUT_COLORS
                                               if c != background])]
    carrier, details = palette[0], palette[1:]
    a = np.full((h, w), background, dtype=np.int8)
    if profile == 'witness':
        counts = [2]*8
    elif profile == 'train':
        counts = list(map(int, rng.integers(0, 4, size=8)))
        # Every additional demonstration includes simpler and richer motifs.
        positions = rng.permutation(8)[:4]
        for position, count in zip(positions, range(4)):
            counts[int(position)] = count
    else:
        counts = list(map(int, rng.integers(0, 5, size=8)))
        special = int(rng.integers(8))
        counts[special] = 4
        if sum(counts) == 32:
            counts[(special+1) % 8] = 3
    for i, ((dr, dc), count) in enumerate(zip(DIRECTIONS, counts)):
        tile = np.full((3, 3), carrier, dtype=np.int8)
        tile[1, 1] = background
        if profile == 'witness':
            first = SIDES.index((1-dr, 1-dc)) if dr*dc == 0 else int(rng.integers(4))
            second = (first + int(rng.choice((-1, 1)))) % 4
            tile[SIDES[first]] = details[i % len(details)]
            tile[SIDES[second]] = details[(i+1) % len(details)]
        else:
            chosen = rng.permutation(4)[:count]
            # Repeated colors and independent colors are both admissible.
            common = details[int(rng.integers(len(details)))]
            for side in chosen:
                value = common if rng.random() < .4 else details[int(rng.integers(len(details)))]
                tile[SIDES[int(side)]] = value
        r, c = top+3*(dr+1), left+3*(dc+1)
        a[r:r+3, c:c+3] = tile
    return a


def _unique_input(a, seen):
    if _key(a) not in seen:
        return a
    # At most 13 previous inputs. Sixteen distinct canvases guarantee success
    # without unbounded rejection sampling, retained retries, or extra files.
    info = _analyze(a)
    ring = a[info['top']:info['top']+9, info['left']:info['left']+9]
    for h in range(15, 31):
        candidate = np.full((h, 30), info['background'], dtype=np.int8)
        candidate[3:12, 3:12] = ring
        if _key(candidate) not in seen:
            return candidate
    raise RuntimeError('bounded uniqueness fallback exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_))
             and isinstance(seed, (int, np.integer)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    fixed_background = bool(rng.random() < .3)
    background = int(rng.choice(INPUT_COLORS))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', n), ('test', m)):
        for i in range(count):
            bg = background if fixed_background else int(rng.choice(INPUT_COLORS))
            profile = ('witness' if i == 0 else 'train') if split == 'train' else 'test'
            a = _unique_input(_new_input(rng, bg, profile), seen)
            seen.add(_key(a))
            puzzle[split].append({'input': a, 'output': solve(a)})
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _views(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _check_demonstrations(puzzle):
    """Check query-relevant D4 and sample-position ambiguity using training.

    This is a bounded alternative-rule check, not an independent ARC solver.
    All diagonal corners coincide after labeling; their position ambiguity is
    unavoidable but harmless on every member of this fixed-size family.
    """
    operations = [set(range(8)) for _ in DIRECTIONS]
    positions = [set(range(9)) for _ in DIRECTIONS]
    for pair in puzzle['train']:
        a, b = np.asarray(pair['input']), np.asarray(pair['output'])
        info = _analyze(a)
        r, c = info['top']+4, info['left']+4
        for i, ((dr, dc), raw) in enumerate(zip(DIRECTIONS, info['tiles'])):
            tile = _recolor(raw, info['background'])
            target = b[r+6*dr-1:r+6*dr+2, c+6*dc-1:c+6*dc+2]
            operations[i].intersection_update(k for k, view in enumerate(_views(tile))
                                               if np.array_equal(view, target))
            positions[i].intersection_update(j for j in range(9)
                                              if tile.flat[j] == b[r+dr, c+dc])
    _require(all(operations) and all(positions), 'contradictory demonstration geometry')
    for pair in puzzle['test']:
        info = _analyze(pair['input'])
        for i, raw in enumerate(info['tiles']):
            tile = _recolor(raw, info['background'])
            views = _views(tile)
            _require(len({views[k].tobytes() for k in operations[i]}) == 1,
                     'training leaves a query-relevant half-turn ambiguity')
            _require(len({int(tile.flat[j]) for j in positions[i]}) == 1,
                     'training leaves a query-relevant sample ambiguity')


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             'puzzle must contain exactly train/test')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')),
             'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    for pair in puzzle['train'] + puzzle['test']:
        _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                 'pair must contain exactly input/output')
        a, b = _grid(pair['input']), _grid(pair['output'])
        _require(a.shape == b.shape, 'dimensions changed')
        _require(_key(a) not in seen, 'duplicate input')
        seen.add(_key(a))
        info = _analyze(a)
        r, c = info['top']+4, info['left']+4
        expected = a.copy()
        expected[r, c] = info['carrier']
        # Scalar coordinate checks use the only admissible contour's local
        # corner/side geometry, independently of solve's trace and slice builder.
        for dr, dc in DIRECTIONS:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    sr, sc = r+3*dr+dy, c+3*dc+dx
                    value = CONVEX if dy*dx != 0 else int(a[sr, sc])
                    expected[sr, sc] = value
                    expected[r+6*dr-dy, c+6*dc-dx] = value
            sample = CONVEX if dr*dc != 0 else int(a[r+2*dr, c+2*dc])
            expected[r+dr, c+dc] = sample
        _require(np.array_equal(b, expected), 'incorrect labeling, sampling, echo, or preservation')
        _require(int(np.count_nonzero(a != b)) == 105, 'expected exactly 105 changed cells')
        _require(int(np.count_nonzero(b != info['background'])) == 137,
                 'expected 137 occupied output cells')
        _require(int(np.count_nonzero(b == CONVEX)) == 68, 'incorrect convex-label propagation')
        _require(not np.any(b == CONCAVE), 'a 3x3 contour cannot have concave bends')
        _require(set(map(int, np.unique(b))) == set(map(int, np.unique(a))) | {CONVEX},
                 'incorrect output palette')
        _require(int(np.count_nonzero(b != _finish(a, info, label=False))) == 68,
                 'omitting parent A must change 68 transported/original bend cells')
        _require(int(np.count_nonzero(b != _finish(a, info, expand=False))) == 73,
                 'omitting parent B must leave 73 required foreground cells blank')
    _check_demonstrations(puzzle)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        h.update(split.encode())
        for pair in puzzle[split]:
            for name in ('input', 'output'):
                a = np.asarray(pair[name])
                h.update(bytes(a.shape))
                h.update(a.astype(np.int8, copy=False).tobytes())
    return h.digest()


def _expect_invalid(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid data were accepted')


def self_test():
    """42 streaming seeds, aggregate diagnostics only; no disk writes."""
    signatures, shapes, layouts, motifs, backgrounds, carriers = (set() for _ in range(6))
    side_counts, sample_patterns = set(), set()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pair_count += 1
                a = pair['input']
                info = _analyze(a)
                shapes.add(a.shape)
                layouts.add((a.shape, info['top'], info['left']))
                backgrounds.add(info['background'])
                carriers.add(info['carrier'])
                ring = a[info['top']:info['top']+9, info['left']:info['left']+9]
                roles = {info['background']: 0, info['carrier']: 1}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in ring.flat)
                motifs.add(hashlib.sha256(normalized).digest())
                counts = [sum(int(t[p]) != info['carrier'] for p in SIDES) for t in info['tiles']]
                side_counts.update(counts)
                if split == 'train':
                    assert max(counts) <= 3
                else:
                    assert max(counts) == 4
                r, c = info['top']+4, info['left']+4
                values = [int(pair['output'][r+dr, c+dc]) for dr, dc in DIRECTIONS]
                roles = {}
                sample_patterns.add(tuple(roles.setdefault(v, len(roles)) for v in values))
                saved = a.copy()
                assert np.array_equal(solve(a), pair['output'])
                assert np.array_equal(a, saved), 'solve mutated its input'
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 70 and len(layouts) >= 150 and len(motifs) >= 150
    assert backgrounds == carriers == set(INPUT_COLORS)
    # Four forced-convex diagonal samples leave only Bell(4)=15 possible
    # equality patterns among the four cardinal samples.
    assert side_counts == set(range(5)) and len(sample_patterns) >= 10
    legal_counts = 0
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(1200+n*10+m, n, m)
            assert len(p['train']) == n and len(p['test']) == m
            legal_counts += 1
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                  (2.0, 1), (True, 1), (4, False), (None, 1), ('4', 1)):
        _expect_invalid(lambda: generate_puzzle(0, n, m))
    # Exhaust the 512 masks instead of claiming nonexistent concave examples.
    accepted = []
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        try:
            classes = _classes(mask)
        except ValueError:
            continue
        accepted.append(bits)
        assert np.count_nonzero(classes == 1) == 4 and not np.any(classes == 2)
    assert accepted == [495]
    # The general helper recognizes concavity, but this 5x5 shape is NOT a
    # composition tile and is never passed to solve as a family member.
    larger = np.array([[int(c) for c in row] for row in
                       ('11111', '10001', '10111', '10100', '11100')], dtype=bool)
    classes = _classes(larger)
    assert np.count_nonzero(classes == 1) == 5 and np.count_nonzero(classes == 2) == 1
    base = generate_puzzle(0)
    for point_type in ('bend', 'straight', 'sample', 'carrier', 'echo', 'unused'):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        pair = broken['test'][0]
        info = _analyze(pair['input'])
        r, c = info['top']+4, info['left']+4
        point = {'bend': (r-4, c-4), 'straight': (r-4, c-3),
                 'sample': (r-1, c-1), 'carrier': (r, c),
                 'echo': (r-7, c-7), 'unused': (r-7, c-4)}[point_type]
        pair['output'][point] = (int(pair['output'][point])+1) % 10
        _expect_invalid(lambda: validate_generated_puzzle(broken))
    for coordinate in ((0, 0), (1, 1)):
        info = _analyze(base['test'][0]['input'])
        bad = base['test'][0]['input'].copy()
        bad[info['top']+coordinate[0], info['left']+coordinate[1]] = info['background'] if coordinate == (0, 0) else info['carrier']
        _expect_invalid(lambda: solve(bad))
    # Exercise the full footprint against every canvas edge and extreme aspect ratios.
    info = _analyze(base['test'][0]['input'])
    ring = base['test'][0]['input'][info['top']:info['top']+9, info['left']:info['left']+9]
    for h, w in ((15, 15), (15, 30), (30, 15), (30, 30)):
        for top, left in ((3, 3), (h-12, w-12)):
            a = np.full((h, w), info['background'], dtype=np.int8)
            a[top:top+9, left:left+9] = ring
            b = solve(a)
            assert b.shape == a.shape and np.count_nonzero(a != b) == 105
    # JSON/list callers and NumPy integer counts use the same semantics.
    listed = {s: [{k: v.tolist() for k, v in p.items()} for p in base[s]] for s in base}
    validate_generated_puzzle(listed)
    generate_puzzle(np.int64(9), np.int64(2), np.int64(1))
    return {'seeds': 42, 'pairs': pair_count, 'distinct_puzzles': len(signatures),
            'grid_shapes': len(shapes), 'layouts': len(layouts),
            'normalized_color_motifs': len(motifs), 'center_equality_patterns': len(sample_patterns),
            'legal_counts': legal_counts, 'admissible_3x3_masks': len(accepted),
            'concave_family_examples': 0,
            'limitation': 'Fixed 3x3 contours cannot demonstrate the concave label.'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
