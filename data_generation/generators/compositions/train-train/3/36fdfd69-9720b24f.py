"""Open molds turn camouflaged rectangular parcels out across their mouths.

Only standard-library and NumPy code. The solver reads geometry and colors from
one input; generation records are used only to construct independent labels.
"""
from __future__ import annotations

from collections import deque
import json
import math
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        group.append(q)
                        stack.append(q)
        groups.append(group)
    return groups


def _runs(row):
    cs = np.flatnonzero(row)
    return np.split(cs, np.flatnonzero(np.diff(cs) > 1) + 1) if len(cs) else []


def _recognize(wall):
    """Recognize scanline-convex two-arm walls, rather than a template catalog."""
    found = []
    for turns in range(4):
        a = np.rot90(wall, turns)
        h, w = a.shape
        if h < 4 or w < 4:
            continue
        sides, cavity = [], np.zeros_like(a)
        for r in range(h - 1):
            runs = _runs(a[r])
            if (len(runs) != 2 or any(not 1 <= len(v) <= 2 for v in runs)
                    or runs[1][0] - runs[0][-1] < 3):
                break
            sides.append((int(runs[0][0]), int(runs[1][-1])))
            if r:
                cavity[r, runs[0][-1] + 1:runs[1][0]] = True
        if len(sides) != h - 1:
            continue
        bottom = _runs(a[-1])
        if (len(bottom) != 1 or len(bottom[0]) < 4
                or (int(bottom[0][0]), int(bottom[0][-1])) != sides[-1]
                or any(abs(x[0] - y[0]) > 1 or abs(x[1] - y[1]) > 1
                       for x, y in zip(sides, sides[1:]))):
            continue
        # The mouth gap itself is a boundary, never part of the hollow.
        mouth = np.zeros_like(a)
        lip_runs = _runs(a[0])
        mouth[0, lip_runs[0][-1] + 1:lip_runs[1][0]] = True
        found.append((turns, np.rot90(cavity, -turns), np.rot90(mouth, -turns)))
    if len(found) != 1:
        raise ValueError("wall does not have one identifiable mouth")
    return found[0]


def _flood_hollow(wall, mouth):
    """Separate topological check: close the mouth, flood from outside."""
    blocked = np.pad(wall | mouth, 1)
    outside = np.zeros_like(blocked)
    outside[0, 0] = True
    todo = deque([(0, 0)])
    while todo:
        r, c = todo.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]
                    and not blocked[nr, nc] and not outside[nr, nc]):
                outside[nr, nc] = True
                todo.append((nr, nc))
    return (~blocked & ~outside)[1:-1, 1:-1]


def _reflect(points, axis, line):
    result = np.array(points, dtype=int, copy=True)
    result[:, axis] = 2 * line - result[:, axis]
    return result


def _parse(x):
    """Infer camouflage, walls, membership, marker rectangles and exit geometry."""
    colors = [int(v) for v in np.unique(x) if v not in (0, 2)]
    answers = []
    if not np.any(x == 0) or not np.any(x == 2):
        raise ValueError("background and parcel markers must be present")
    for camouflage in colors:
        records, used_hollows, destinations = [], np.zeros_like(x, bool), np.zeros_like(x, bool)
        walls = np.zeros_like(x, bool)
        try:
            for color in colors:
                if color == camouflage:
                    continue
                for group in _components(x == color):
                    pts = np.array(group)
                    top, left = pts.min(0)
                    bottom, right = pts.max(0) + 1
                    local = np.zeros((bottom - top, right - left), bool)
                    local[pts[:, 0] - top, pts[:, 1] - left] = True
                    turns, cavity, mouth = _recognize(local)
                    hollow = np.zeros_like(x, bool)
                    hollow[top:bottom, left:right] = cavity
                    mouth_global = np.zeros_like(x, bool)
                    mouth_global[top:bottom, left:right] = mouth
                    if (np.any(used_hollows & hollow) or np.any(x[mouth_global] != 0)
                            or np.any(~np.isin(x[hollow], (0, 2, camouflage)))):
                        raise ValueError("invalid hollow or mouth contents")
                    markers = np.argwhere(hollow & (x == 2))
                    if len(markers) < 2:
                        raise ValueError("each mold needs a parcel")
                    r0, c0 = markers.min(0)
                    r1, c1 = markers.max(0) + 1
                    parcel = np.zeros_like(x, bool)
                    parcel[r0:r1, c0:c1] = True
                    if (r1 - r0 < 2 or c1 - c0 < 2 or np.any(parcel & ~hollow)
                            or np.any(~np.isin(x[parcel], (2, camouflage)))
                            or not np.any(x[parcel] == camouflage)):
                        raise ValueError("markers must bound a solid camouflaged rectangle")
                    axis = turns % 2
                    line = (int(top) if turns == 0 else int(right - 1) if turns == 1
                            else int(bottom - 1) if turns == 2 else int(left))
                    source = np.argwhere(parcel)
                    dest = _reflect(source, axis, line)
                    if (np.any(dest < 0) or np.any(dest >= np.array(x.shape))
                            or np.any(x[dest[:, 0], dest[:, 1]] != 0)
                            or np.any(destinations[dest[:, 0], dest[:, 1]])):
                        raise ValueError("exit parcel must fit in empty, unshared space")
                    destinations[dest[:, 0], dest[:, 1]] = True
                    used_hollows |= hollow
                    walls[pts[:, 0], pts[:, 1]] = True
                    records.append(dict(color=color, turns=turns, axis=axis, line=line,
                                        wall=local, mouth=mouth, cavity=cavity,
                                        box=(int(top), int(left), int(bottom), int(right)),
                                        hollow=hollow, parcel=parcel, source=source, dest=dest))
            if not records or np.any(used_hollows & walls) or np.any(destinations & used_hollows):
                raise ValueError("molds must have separate hollows and exits")
        except ValueError:
            continue
        answers.append((camouflage, records, used_hollows, destinations))
    if len(answers) != 1:
        raise ValueError("input must uniquely identify camouflage and open molds")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    x = _grid(input_grid)
    camouflage, records, hollows, _ = _parse(x)
    y = x.copy()
    y[hollows] = 0
    for record in records:
        src, dst = record['source'], record['dest']
        values = x[src[:, 0], src[:, 1]]
        y[dst[:, 0], dst[:, 1]] = np.where(values == camouflage, record['color'], 2)
    return y


def _pattern(rng, h, w, asymmetric):
    for _ in range(60):
        mode = int(rng.integers(5))
        m = np.zeros((h, w), bool)
        if mode == 0:
            m[0, 0] = m[-1, -1] = True
            if rng.random() < .5:
                m = m[:, ::-1].copy()
        elif mode == 1:
            for r, c in ((0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                         (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)):
                m[r, c] = True
        elif mode == 2:
            m = rng.random((h, w)) < float(rng.uniform(.25, .75))
            m[0, 0] = m[-1, -1] = True
        elif mode == 3:
            m[0] = True
            m[:, -1] = True
        else:
            m = np.indices((h, w)).sum(0) % 2 == int(rng.integers(2))
        if (m.any() and not m.all() and (not asymmetric or
                (not np.array_equal(m, m[::-1]) and not np.array_equal(m, m[:, ::-1])))):
            return m
    m[:] = False
    m[0, 0] = m[-1, -1] = True
    return m


def _module(rng, count, rotation, force_asymmetric, teaching_witness=False):
    """Construct geometry and a parcel before rotating the entire source/exit site."""
    for _ in range(100):
        h = int(rng.integers(5 if teaching_witness else 4,
                             11 if count == 1 else 8 if count == 2 else 7))
        gap = int(rng.integers(3, 11 if count == 1 else 8))
        lefts, rights = [h], [h + gap + 1]
        straight = rng.random() < .30 and not teaching_witness
        for r in range(1, h - 1):
            a = lefts[-1] + (0 if straight else int(rng.choice((-1, 0, 0, 1))))
            b = rights[-1] + (0 if straight else int(rng.choice((-1, 0, 0, 1))))
            if b - a < 4 or b - a > 12:
                a, b = lefts[-1], rights[-1]
            lefts.append(a)
            rights.append(b)
        wall = np.zeros((h, max(rights) + 1), bool)
        for r, (a, b) in enumerate(zip(lefts, rights)):
            ta = 1 + (b - a >= 6 and rng.random() < .23)
            tb = 1 + (b - a - ta >= 5 and rng.random() < .23)
            wall[r, a:a + ta] = True
            wall[r, b - tb + 1:b + 1] = True
        wall[-1, lefts[-1]:rights[-1] + 1] = True
        occupied_cols = np.flatnonzero(wall.any(0))
        wall = wall[:, occupied_cols[0]:occupied_cols[-1] + 1]
        try:
            turns, cavity, mouth = _recognize(wall)
        except ValueError:
            continue
        if turns != 0 or not np.array_equal(cavity, _flood_hollow(wall, mouth)):
            continue
        if teaching_witness and len(set(zip(lefts, rights))) == 1:
            continue
        options = []
        for r0 in range(2 if teaching_witness else 1, h - 2):
            for r1 in range(r0 + 2, h):
                usable = np.flatnonzero(cavity[r0:r1].all(0))
                if len(usable) < 2:
                    continue
                for c0 in usable:
                    for c1 in range(int(c0) + 2, int(usable[-1]) + 2):
                        if cavity[r0:r1, c0:c1].all() and (r1-r0)*(c1-c0) < cavity.sum():
                            options.append((r0, r1, int(c0), c1))
        if not options:
            continue
        r0, r1, c0, c1 = options[int(rng.integers(len(options)))]
        shape = (2 * h - 1, wall.shape[1])
        masks = {name: np.zeros(shape, bool) for name in ('wall', 'cavity', 'mouth', 'parcel', 'markers', 'dest')}
        for name, mask in (('wall', wall), ('cavity', cavity), ('mouth', mouth)):
            masks[name][h - 1:] = mask
        masks['parcel'][h - 1 + r0:h - 1 + r1, c0:c1] = True
        masks['markers'][h - 1 + r0:h - 1 + r1, c0:c1] = _pattern(rng, r1-r0, c1-c0, force_asymmetric)
        masks['dest'] = masks['parcel'][::-1].copy()
        masks['reflected_markers'] = masks['markers'][::-1].copy()
        return {name: np.rot90(mask, rotation).copy() for name, mask in masks.items()}
    raise RuntimeError("could not construct a mold")


def _pack(rng, shapes):
    n = len(shapes)
    if n == 1:
        h = shapes[0][0] + int(rng.integers(0, 7))
        w = shapes[0][1] + int(rng.integers(0, 7))
    else:
        lower = max(16 if n == 2 else 20 if n == 3 else 23,
                    math.ceil(math.sqrt(sum(h*w for h, w in shapes))))
        h = int(rng.integers(max(lower, max(s[0] for s in shapes)), 31))
        w = int(rng.integers(max(lower, max(s[1] for s in shapes)), 31))
    if h > 30 or w > 30:
        return None
    order = sorted(range(n), key=lambda i: -shapes[i][0] * shapes[i][1])
    for _ in range(20):
        placed, locations = [], [None] * n
        for i in order:
            mh, mw = shapes[i]
            options = [(r, c) for r in range(h-mh+1) for c in range(w-mw+1)]
            rng.shuffle(options)
            for r, c in options:
                if all(r+mh < a or b < r or c+mw < d or e < c for a, b, d, e in placed):
                    placed.append((r, r+mh, c, c+mw))
                    locations[i] = (r, c)
                    break
            if locations[i] is None:
                break
        if all(p is not None for p in locations):
            return (h, w), locations
    return None


def _make_pair(rng, count, rotations, camouflage, require_decoy):
    for _ in range(100):
        modules = [_module(rng, count, rotation, i == 0, require_decoy and i == 0)
                   for i, rotation in enumerate(rotations)]
        packing = _pack(rng, [m['wall'].shape for m in modules])
        if packing is None:
            continue
        shape, positions = packing
        x = np.zeros(shape, np.int8)
        masks = {name: np.zeros(shape, bool) for name in ('wall', 'cavity', 'mouth', 'parcel', 'markers', 'dest')}
        records = []
        wall_palette = [c for c in range(1, 10) if c not in (2, camouflage)]
        for index, (m, (r, c)) in enumerate(zip(modules, positions)):
            sl = np.s_[r:r+m['wall'].shape[0], c:c+m['wall'].shape[1]]
            choices = [v for v in wall_palette if not (require_decoy and index == 1 and v == records[0][2])]
            color = int(rng.choice(choices))
            for name in masks:
                masks[name][sl] |= m[name]
            x[sl][m['wall']] = color
            records.append((sl, m, color))
        free = ~(masks['wall'] | masks['mouth'] | masks['dest'])
        # Ranked random field plus short patches: camouflage can touch walls and parcels.
        score = rng.random(shape)
        for _ in range(int(rng.integers(1, 6))):
            r, c = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
            score[r:r+int(rng.integers(1, 4)), c:c+int(rng.integers(2, 6))] += .5
        density = float(rng.uniform(.08, .43))
        x[free & (score > 1-density)] = camouflage
        x[masks['parcel']] = camouflage
        x[masks['markers']] = 2
        # Deliberate loose material distinguishes a rectangle from the whole hollow.
        for sl, m, _ in records:
            loose = np.argwhere(m['cavity'] & ~m['parcel'])
            r, c = loose[int(rng.integers(len(loose)))]
            x[sl][r, c] = camouflage
        if require_decoy:
            sl, m, _ = records[0]
            lo, hi = np.argwhere(m['wall']).min(0), np.argwhere(m['wall']).max(0) + 1
            within_box = np.zeros_like(m['wall'])
            within_box[lo[0]:hi[0], lo[1]:hi[1]] = True
            corners = np.argwhere(within_box & ~(m['wall'] | m['cavity'] | m['mouth'] | m['dest']))
            if not len(corners):
                continue
            r, c = corners[int(rng.integers(len(corners)))]
            x[sl][r, c] = camouflage
        # An uncontained solid marked rectangle is a direct containment counterexample.
        eligible = ~(masks['wall'] | masks['cavity'] | masks['mouth'] | masks['dest'])
        options = [(r, c) for r in range(shape[0]-1) for c in range(shape[1]-1)
                   if eligible[r:r+2, c:c+2].all()]
        if require_decoy and not options:
            continue
        if options and (require_decoy or rng.random() < .65):
            r, c = options[int(rng.integers(len(options)))]
            x[r:r+2, c:c+2] = camouflage
            if rng.random() < .5:
                x[r, c] = x[r+1, c+1] = 2
            else:
                x[r+1, c] = x[r, c+1] = 2
        # Add an exterior camouflage witness. Unique color-role interpretation is
        # checked below from the returned grid, rather than assumed from this mark.
        outside_points = np.argwhere(eligible)
        if not len(outside_points):
            continue
        r, c = outside_points[int(rng.integers(len(outside_points)))]
        if x[r, c] != 2:
            x[r, c] = camouflage
        y = x.copy()
        y[masks['cavity']] = 0
        for sl, m, color in records:
            y[sl][m['dest']] = color
            y[sl][m['reflected_markers']] = 2
        try:
            inferred, parsed, _, _ = _parse(x)
        except ValueError:
            continue
        if inferred != camouflage or len(parsed) != count or not np.array_equal(solve(x), y):
            continue
        return {'input': x, 'output': y}
    raise RuntimeError("could not construct a distinct interpretable scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not lo <= int(value) <= hi:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    base = int(rng.integers(4))
    camouflage_palette = [int(c) for c in rng.permutation([1, 3, 4, 5, 6, 7, 8, 9])]
    first_directions = [base, (base + 1) % 4]
    rng.shuffle(first_directions)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            if split == 'train':
                n = 1 if index == 0 else int(rng.choice((2, 2, 3))) if index == 1 else int(rng.choice((1, 1, 2, 2, 3, 4)))
                direction = first_directions[index] if index < 2 else int(rng.choice(first_directions))
                rotations = [direction] * n
                camouflage = camouflage_palette[index % 2] if index < 2 else int(rng.choice(camouflage_palette))
            else:
                n = int(rng.choice((1, 2, 2, 3, 4)))
                directions = [(base + 2) % 4, (base + 3) % 4]
                rotations = [directions[i % 2] for i in range(n)]
                rng.shuffle(rotations)
                camouflage = int(rng.choice(camouflage_palette))
            for _ in range(30):
                pair = _make_pair(rng, n, rotations, camouflage, split == 'train' and index == 1)
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not produce a new input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nv = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    seen, train_directions, test_directions = set(), set(), set()
    decoy_witness, asymmetric_witness, depth_witness, corner_witness = False, False, False, False
    train_wall_colors = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            x, y = _grid(pair['input']), _grid(pair['output'])
            assert x.shape == y.shape and _key(x) not in seen
            seen.add(_key(x))
            camouflage, records, hollows, exits = _parse(x)
            assert 1 <= len(records) <= 4
            assert np.all(y[hollows] == 0)
            assert np.array_equal(x[~(hollows | exits)], y[~(hollows | exits)])
            assert np.all(x[exits] == 0) and np.all(y[exits] != 0)
            assert np.any((x == camouflage) & ~hollows & ~exits)
            for rec in records:
                assert np.array_equal(_flood_hollow(rec['wall'], rec['mouth']), rec['cavity'])
                src, dst = rec['source'], rec['dest']
                expected = np.where(x[src[:, 0], src[:, 1]] == 2, 2, rec['color'])
                assert np.array_equal(y[dst[:, 0], dst[:, 1]], expected)
                assert np.any(rec['hollow'] & ~rec['parcel'] & (x == camouflage))
                assert np.count_nonzero(x[rec['parcel']] == 2) == np.count_nonzero(y[dst[:, 0], dst[:, 1]] == 2)
                lo, hi = src.min(0), src.max(0) + 1
                pattern = x[lo[0]:hi[0], lo[1]:hi[1]] == 2
                axis = rec['axis']
                if split == 'train':
                    asymmetric_witness |= not np.array_equal(pattern, np.flip(pattern, axis=axis))
                    depth_witness |= np.min(np.abs(src[:, axis] - rec['line'])) > 1
                    train_wall_colors.add(rec['color'])
                    top, left, bottom, right = rec['box']
                    corner_witness |= bool(np.any((x[top:bottom, left:right] == camouflage)
                                                   & ~rec['cavity']))
                (train_directions if split == 'train' else test_directions).add(rec['turns'])
            if split == 'train':
                for r in range(x.shape[0] - 1):
                    for c in range(x.shape[1] - 1):
                        block = x[r:r+2, c:c+2]
                        marks = block == 2
                        decoy_witness |= bool(not hollows[r:r+2, c:c+2].any()
                                              and np.isin(block, (2, camouflage)).all()
                                              and marks.sum() >= 2 and (block == camouflage).any()
                                              and marks.any(0).all() and marks.any(1).all())
            assert np.array_equal(solve(x), y)
    assert len(train_directions) == 2 and train_directions.isdisjoint(test_directions)
    assert decoy_witness and asymmetric_witness and depth_witness and corner_witness
    assert len(train_wall_colors) >= 2


def self_test():
    """Bounded survey of 42 seeds, plus counts, ablations and equivariance."""
    shapes, wall_shapes, parcel_shapes, counts, colors, fingerprints = set(), set(), set(), set(), set(), set()
    feature_counts = dict(stepped=0, thick=0, two_markers=0, disconnected=0,
                          depth_gap=0, edge=0, reflected_not_translated=0)
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(tuple(_key(v['input']) for s in ('train', 'test') for v in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                x, y = pair['input'], pair['output']
                camo, records, hollows, exits = _parse(x)
                shapes.add(x.shape)
                counts.add(len(records))
                colors.add(camo)
                # Removing B's emptying leaves loose material in the hollow.
                no_emptying = y.copy()
                for rec in records:
                    loose = rec['hollow'] & ~rec['parcel']
                    no_emptying[loose] = x[loose]
                assert not np.array_equal(no_emptying, y)
                # Removing A's parcel selection ejects loose material too.
                no_rectangle = y.copy()
                for rec in records:
                    loose = np.argwhere(rec['hollow'] & ~rec['parcel'] & (x == camo))
                    dst = _reflect(loose, rec['axis'], rec['line'])
                    assert np.all(dst >= 0) and np.all(dst < np.array(x.shape))
                    no_rectangle[dst[:, 0], dst[:, 1]] = rec['color']
                assert not np.array_equal(no_rectangle, y)
                # Rotating an entire input must rotate its answer as well.
                if seed < 3:
                    for turns in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(x, turns)), np.rot90(y, turns))
                for rec in records:
                    a = np.rot90(rec['wall'], rec['turns'])
                    side_runs = [_runs(row) for row in a[:-1]]
                    feature_counts['stepped'] += len({(int(r[0][0]), int(r[1][-1])) for r in side_runs}) > 1
                    feature_counts['thick'] += any(len(v) == 2 for runs in side_runs for v in runs)
                    pts = rec['source']
                    lo, hi = pts.min(0), pts.max(0) + 1
                    pattern = x[lo[0]:hi[0], lo[1]:hi[1]] == 2
                    feature_counts['two_markers'] += pattern.sum() == 2
                    feature_counts['disconnected'] += len(_components(pattern)) > 1
                    feature_counts['depth_gap'] += min(abs(int(v) - rec['line']) for v in pts[:, rec['axis']]) > 1
                    feature_counts['reflected_not_translated'] += not np.array_equal(pattern, np.flip(pattern, axis=rec['axis']))
                    top, left, bottom, right = rec['box']
                    feature_counts['edge'] += top == 0 or left == 0 or bottom == x.shape[0] or right == x.shape[1]
                    wall_shapes.add((a.shape, a.tobytes()))
                    parcel_shapes.add(tuple(hi - lo))
    assert len(fingerprints) == 42 and len(shapes) >= 35
    assert len(wall_shapes) >= 80 and len(parcel_shapes) >= 15
    assert counts == {1, 2, 3, 4} and len(colors) == 8
    assert all(v > 0 for v in feature_counts.values()), feature_counts
    for nt in range(2, 11):
        for nv in range(1, min(4, nt-1) + 1):
            p = generate_puzzle(700 + nt * 10 + nv, nt, nv)
            assert (len(p['train']), len(p['test'])) == (nt, nv)
    for nt, nv in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(9123), generate_puzzle(9123)
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(p[s], q[s]) for k in ('input', 'output'))
    return dict(seeds=42, distinct_puzzles=len(fingerprints), canvas_shapes=len(shapes),
                wall_geometries=len(wall_shapes), parcel_shapes=len(parcel_shapes),
                mold_counts=sorted(counts), camouflage_colors=sorted(colors),
                features={k: int(v) for k, v in feature_counts.items()})


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
