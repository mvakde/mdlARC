"""Contour rubbings: localize a transformed crop of a region's external rim.

The two framed input objects supply all information. Nothing in solve depends
on a seed, a generated reference, or other examples. Black always means paper.
"""
from __future__ import annotations

import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 728111203, 736757921, 126381828, 445026667, 502516281,
              93082964, 414081905, 24477572, 247358001, 16216702, 514280305,
              814311670, 951317059, 19672409, 417136462, 813296381, 73570554,
              452879471, 105795012, 561750401, 933268643, 49397377, 250097403,
              791896383, 220877121, 470392952, 902129484, 550859053, 963593691,
              636846093, 304069213, 185525638, 428692388, 409265802, 13312545,
              916223282, 589693619, 670972199, 742115444, 121091710, 332109841)


def _d4(a, k):
    """0..3 rotations; 4..7 reflection followed by a rotation."""
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4).copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(d < 1 or d > 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, sides 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _flood(allowed, start):
    seen = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return seen
    seen[start] = True
    stack = [start]
    h, w = allowed.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and allowed[x, y] and not seen[x, y]:
                seen[x, y] = True
                stack.append((x, y))
    return seen


def _components(mask):
    remaining = mask.copy()
    result = []
    while remaining.any():
        start = tuple(np.argwhere(remaining)[0])
        part = _flood(remaining, start)
        result.append(part)
        remaining[part] = False
    return result


def _near(mask, diagonal=True):
    h, w = mask.shape
    pad = np.pad(mask, 1)
    result = np.zeros_like(mask)
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if r or c] if diagonal else N4
    for dr, dc in offsets:
        result |= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _region_info(walls):
    result = []
    for region in _components(~walls):
        outside = _flood(np.pad(~region, 1, constant_values=True), (0, 0))
        # Padding is important: the edge of the unframed map is external.
        rim = _near(outside)[1:-1, 1:-1] & region
        orth = _near(outside, False)[1:-1, 1:-1] & region
        result.append(dict(region=region, rim=rim, exterior=outside[1:-1, 1:-1],
                           diagonal=rim & ~orth))
    return result


def _bbox(mask):
    rr, cc = np.nonzero(mask)
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _parse(grid):
    """A frame color's global bounding box is exactly its rectangle."""
    colors, counts = np.unique(grid[grid != 0], return_counts=True)
    if len(colors) != 3 or np.count_nonzero(counts == 1) != 1:
        raise ValueError("need two colored frames and one uniquely colored marker")
    marker = int(colors[counts == 1][0])
    mr, mc = (int(x) for x in np.argwhere(grid == marker)[0])
    boxes = []
    for color in colors[counts != 1]:
        color = int(color)
        r0, r1, c0, c1 = _bbox(grid == color)
        tile = grid[r0:r1, c0:c1]
        if (min(tile.shape) < 5 or not np.all(tile[[0, -1], :] == color)
                or not np.all(tile[:, [0, -1]] == color)):
            raise ValueError("invalid frame")
        boxes.append((color, (r0, r1, c0, c1), tile[1:-1, 1:-1].copy(),
                      r0 < mr < r1 - 1 and c0 < mc < c1 - 1))
    if sum(b[3] for b in boxes) != 1:
        raise ValueError("the marker must be inside exactly one frame")
    ink, qbox, patch, _ = next(b for b in boxes if b[3])
    wall, mbox, reference, _ = next(b for b in boxes if not b[3])
    if not set(np.unique(reference)) <= {0, wall} or not set(np.unique(patch)) <= {0, ink, marker}:
        raise ValueError("unexpected colors inside a frame")
    occupied = np.zeros(grid.shape, bool)
    for _, (r0, r1, c0, c1), _, _ in boxes:
        if occupied[r0:r1, c0:c1].any():
            raise ValueError("frames overlap")
        occupied[r0:r1, c0:c1] = True
    if np.any(grid[~occupied] != 0):
        raise ValueError("nonblack cells outside frames")
    if not (np.any(patch == 0) and np.any(patch == ink)):
        raise ValueError("rubbing must show ink and paper")
    roles = np.zeros(patch.shape, np.int8)
    roles[patch == ink] = 1
    roles[patch == marker] = 2
    return dict(walls=reference == wall, patch=roles, wall=wall, ink=ink,
                marker=marker, map_box=mbox, query_box=qbox)


def _matches(infos, patch):
    """Return distinct (region, transferred marker) addresses, not latent ties."""
    addresses = {}
    for k in range(8):
        q = _d4(patch, k)
        h, w = q.shape
        valid = q != 2
        mr, mc = (int(x) for x in np.argwhere(q == 2)[0])
        for j, info in enumerate(infos):
            field = info['rim']
            H, W = field.shape
            if h > H or w > W or h * w >= H * W:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(field, (h, w))
            same = np.all(windows[..., valid] == (q == 1)[valid], axis=-1)
            rr, cc = np.nonzero(same)
            for r, c in zip(rr, cc):
                seed = int(r + mr), int(c + mc)
                if info['region'][seed]:
                    addresses.setdefault((j, *seed), (k, int(r), int(c), h, w))
    return addresses


def _render(walls, info, seed, wall, ink, marker):
    result = np.zeros(walls.shape, np.int8)
    near_region = _near(info['region'], False)
    for component in _components(walls):
        if np.any(component & near_region):
            result[component] = wall
    result[info['rim']] = ink
    result[seed] = marker
    return result


def _analyze(grid):
    data = _parse(grid)
    infos = _region_info(data['walls'])
    matches = _matches(infos, data['patch'])
    if len(matches) != 1:
        raise ValueError("rubbing must determine exactly one region and marker address")
    address, placement = next(iter(matches.items()))
    j, r, c = address
    data.update(infos=infos, selected=j, seed=(r, c), placement=placement)
    return data


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _analyze(_grid(input_grid))
    return _render(data['walls'], data['infos'][data['selected']], data['seed'],
                   data['wall'], data['ink'], data['marker'])


def _map_ok(walls):
    if not .07 < walls.mean() < .45:
        return False
    a, b, c, d = walls[:-1, :-1], walls[:-1, 1:], walls[1:, :-1], walls[1:, 1:]
    if np.any((a == d) & (b == c) & (a != b)):
        return False
    regions = _components(~walls)
    return (2 <= len(regions) <= 5 and all(m.sum() >= 18 and
            (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()) for m in regions))


def _add_shape(rng, walls, region, edge):
    """Place separated solid rectangles, bars, Ls, or ragged masses."""
    H, W = walls.shape
    for _ in range(50):
        h, w = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
        if (r == 0 or c == 0 or r + h == H or c + w == W) != edge:
            continue
        shape = np.ones((h, w), bool)
        kind = int(rng.integers(4))
        if kind == 1:
            shape[1:, 1:] = False
        elif kind == 2:
            for row in range(h):
                shape[row, int(rng.integers(1, w + 1)):] = False
        shape = _d4(shape, int(rng.choice([0, 2, 4, 6])))
        sh, sw = shape.shape
        if r + sh > H or c + sw > W:
            continue
        obj = np.zeros_like(walls)
        obj[r:r + sh, c:c + sw] = shape
        halo = _near(obj) | obj
        if not edge:
            halo |= _near(halo)
            if np.any(obj[[0, 1, -2, -1], :]) or np.any(obj[:, [0, 1, -2, -1]]):
                continue
        if np.any(halo & ~region):
            continue
        trial = walls | obj
        if _map_ok(trial) and len(_components(~trial)) == len(_components(~walls)):
            return trial
    return walls


def _make_map(rng):
    H, W = int(rng.integers(13, 18)), int(rng.integers(16, 23))
    walls = np.zeros((H, W), bool)
    cut = int(rng.integers(5, W - 6))
    # A connected stair divider; independently located steps may reverse.
    previous = cut
    stops = sorted({0, H, *[int(x) for x in rng.integers(2, H - 2, size=int(rng.integers(1, 5)))]})
    for start, stop in zip(stops, stops[1:]):
        col = int(np.clip(cut + rng.integers(-1, 2), 5, W - 7))
        walls[start:stop, col] = True
        if start:
            walls[start, min(previous, col):max(previous, col) + 1] = True
        previous = col
    # A parallel division plus a branch of one compartment can make four
    # compartments. Longer branches can partition several free runs.
    if rng.random() < .22 and W >= 20:
        walls[:, W - 5] = True
    if rng.random() < .45:
        row = int(rng.integers(4, H - 4))
        free = np.flatnonzero(~walls[row])
        runs = [a for a in np.split(free, np.flatnonzero(np.diff(free) > 1) + 1) if len(a) >= 4]
        trial = walls.copy()
        chosen = [runs[int(rng.integers(len(runs)))]]
        if rng.random() < .2:
            chosen = runs
        for run in chosen:
            trial[row, run] = True
        if _map_ok(trial):
            walls = trial
    if not _map_ok(walls):
        return None
    # Add wall lobes without adding compartments.
    for _ in range(int(rng.integers(1, 6))):
        r, c = np.argwhere(walls)[int(rng.integers(walls.sum()))]
        trial = walls.copy()
        trial[max(0, r - 1):min(H, r + 2), max(0, c - 1):min(W, c + 2)] = True
        if _map_ok(trial) and len(_components(~trial)) == len(_components(~walls)):
            walls = trial
    for region in _components(~walls):
        for _ in range(int(rng.integers(1, 4))):
            walls = _add_shape(rng, walls, region & ~walls, True)
        for _ in range(int(rng.integers(1, 4))):
            walls = _add_shape(rng, walls, region & ~walls, False)
    infos = _region_info(walls)
    if not any(np.any(walls & ~i['exterior']) and i['diagonal'].any() for i in infos):
        return None
    return walls


def _fits(map_shape, query_shape):
    H, W = map_shape
    h, w = query_shape
    return (H + h + 5 <= 30 and max(W, w) + 2 <= 30) or (
        W + w + 5 <= 30 and max(H, h) + 2 <= 30)


def _frame(interior, color):
    return np.pad(interior, 1, constant_values=color)


def _pack(rng, walls, patch, palette):
    wall, ink, marker = palette
    ref = _frame(np.where(walls, wall, 0).astype(np.int8), wall)
    query = _frame(np.choose(patch, [0, ink, marker]).astype(np.int8), ink)
    options = []
    if ref.shape[0] + query.shape[0] + 1 <= 30:
        options.append(0)
    if ref.shape[1] + query.shape[1] + 1 <= 30:
        options.append(1)
    axis = int(rng.choice(options))
    objects = [ref, query]
    if rng.random() < .5:
        objects.reverse()
    size = [max(ref.shape[0], query.shape[0]), max(ref.shape[1], query.shape[1])]
    size[axis] = ref.shape[axis] + query.shape[axis] + 1
    # Independent gaps, edge alignments, and spare margins; never over 30.
    size = [n + int(rng.integers(0, min(3, 30 - n) + 1)) for n in size]
    canvas = np.zeros(size, np.int8)
    first = int(rng.integers(size[axis] - sum(o.shape[axis] for o in objects)))
    gap = int(rng.integers(1, size[axis] - first - sum(o.shape[axis] for o in objects) + 1))
    pos = first
    for obj in objects:
        off = int(rng.integers(size[1 - axis] - obj.shape[1 - axis] + 1))
        r, c = (pos, off) if axis == 0 else (off, pos)
        canvas[r:r + obj.shape[0], c:c + obj.shape[1]] = obj
        pos += obj.shape[axis] + gap
    return canvas


def _pair(rng, base, palette, split, index, seen):
    map_rotation = int(rng.integers(8))
    walls = _d4(base, map_rotation)
    infos = _region_info(walls)
    sizes = [int(i['region'].sum()) for i in infos]
    # First two examples demonstrate opposite size choices and marker categories.
    target = int(np.argmin(sizes)) if index == 0 and split == 'train' else (
        int(np.argmax(sizes)) if index == 1 and split == 'train' else int(rng.integers(len(infos))))
    marker_mode = 'interior' if split == 'train' and index == 0 else (
        'rim' if split == 'train' and index == 1 else 'any')
    # Two nontrivial isometries already teach rotations and reflection at 2/1.
    transform = 1 if split == 'train' and index == 0 else (
        4 if split == 'train' and index == 1 else int(rng.integers(8)))
    H, W = walls.shape
    low, high = (.26, .60) if split == 'train' else (.10, .25)
    shapes = [(h, w) for h in range(3, H + 1) for w in range(3, W + 1)
              if low <= h * w / (H * W) <= high and _fits(walls.shape, _d4(np.zeros((h, w)), transform).shape)]
    if not shapes:
        return None
    info = infos[target]
    for _ in range(750):
        h, w = shapes[int(rng.integers(len(shapes)))]
        r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
        selected = info['region'][r:r + h, c:c + w]
        rim = info['rim'][r:r + h, c:c + w]
        allowed = selected & (~rim if marker_mode == 'interior' else rim if marker_mode == 'rim' else True)
        positions = np.argwhere(allowed)
        if not len(positions) or rim.sum() < 4 or not np.any(~rim):
            continue
        mr, mc = (int(x) for x in positions[int(rng.integers(len(positions)))])
        patch = rim.astype(np.int8)
        patch[mr, mc] = 2
        patch = _d4(patch, transform)
        matches = _matches(infos, patch)
        seed = r + mr, c + mc
        if len(matches) != 1 or (target, *seed) not in matches:
            continue
        if split == 'train' and index < 2:
            # Avoid a symmetric rubbing hiding the advertised isometry.
            if next(iter(matches.values()))[0] != (3 if index == 0 else 4):
                continue
        if np.count_nonzero(info['rim']) <= np.count_nonzero(rim):
            continue
        inp = _pack(rng, walls, patch, palette)
        if _key(inp) in seen:
            continue
        seen.add(_key(inp))
        out = _render(walls, info, seed, *palette)
        return {'input': inp, 'output': out}
    return None


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("num_train must be 2..10 and num_test 1..4, both integers")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(int(x) for x in rng.choice(np.arange(1, 10), 3, replace=False))
    for _ in range(80):
        base = _make_map(rng)
        if base is None:
            continue
        puzzle, seen = {'train': [], 'test': []}, set()
        failed = False
        for split, count in (('train', num_train), ('test', num_test)):
            for index in range(count):
                pair = _pair(rng, base, palette, split, index, seen)
                if pair is None:
                    failed = True
                    break
                puzzle[split].append(pair)
            if failed:
                break
        if failed:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except (AssertionError, ValueError):
            continue
        rng.shuffle(puzzle['train'])
        rng.shuffle(puzzle['test'])
        return puzzle
    raise RuntimeError("bounded map/unique-rubbing search exhausted")


def _features(data, output):
    walls = data['walls']
    info = data['infos'][data['selected']]
    _, r, c, h, w = data['placement']
    support = np.zeros(walls.shape, bool)
    support[r:r + h, c:c + w] = True
    islands = walls & ~info['exterior']
    quiet_hole_neighbors = _near(islands) & info['region'] & ~info['rim']
    erased = walls & (output == 0)
    # Ablation: an ink rubbing is not an exact crop of raw structural pixels.
    raw = [dict(rim=walls, region=~walls)]
    raw_addresses = _matches(raw, data['patch'])
    local_seed = tuple(int(x) for x in np.argwhere(data['patch'] == 2)[0])
    local_wrong = True
    if (local_seed[0] < walls.shape[0] and local_seed[1] < walls.shape[1]
            and not walls[local_seed]):
        local_info = next(i for i in data['infos'] if i['region'][local_seed])
        local_out = _render(walls, local_info, local_seed, data['wall'], data['ink'], data['marker'])
        local_wrong = not np.array_equal(local_out, output)
    return dict(area=h * w / walls.size, selected=int(info['region'].sum()),
                largest=max(int(i['region'].sum()) for i in data['infos']),
                smallest=min(int(i['region'].sum()) for i in data['infos']),
                hole=bool((quiet_hole_neighbors & (output == 0)).any()), erased=bool(erased.any()),
                diagonal=bool((info['diagonal'] & (output == data['ink'])).any()),
                rim_marker=bool(info['rim'][data['seed']]),
                outside_crop=bool((info['rim'] & ~support).any()),
                raw_shortcut=(0, *data['seed']) not in raw_addresses,
                local_shortcut=local_wrong,
                compartments=len(data['infos']), islands=len(_components(islands)),
                bends=int(np.count_nonzero(np.diff(walls.astype(int), axis=0))),
                patch_shape=data['patch'].shape, map_shape=walls.shape,
                transform=data['placement'][0])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, palette, references, train_features, test_features = set(), set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            data = _analyze(inp)
            assert _map_ok(data['walls'])
            assert out.shape == data['walls'].shape
            assert np.array_equal(out, solve(inp))
            palette.add((data['wall'], data['ink'], data['marker']))
            references.add(min(_key(_d4(data['walls'], k)) for k in range(8)))
            info = data['infos'][data['selected']]
            assert np.count_nonzero(out == data['marker']) == 1
            assert out[data['seed']] == data['marker']
            assert np.all(out[info['rim'] & (out != data['marker'])] == data['ink'])
            assert np.all(out[~info['region'] & ~data['walls']] == 0)
            f = _features(data, out)
            assert f['outside_crop']
            (train_features if split == 'train' else test_features).append(f)
    assert len(palette) == len(references) == 1
    # Coverage checks are sampling guarantees, not additional solve conventions.
    assert any(f['selected'] == f['smallest'] < f['largest'] for f in train_features)
    assert any(f['selected'] == f['largest'] > f['smallest'] for f in train_features)
    assert any(f['hole'] for f in train_features)
    assert any(f['erased'] for f in train_features)
    assert any(f['diagonal'] for f in train_features)
    assert {f['rim_marker'] for f in train_features} == {False, True}
    assert any(f['raw_shortcut'] for f in train_features)
    assert any(f['local_shortcut'] for f in train_features)
    assert {3, 4} <= {f['transform'] for f in train_features}
    assert min(f['area'] for f in train_features) > max(f['area'] for f in test_features)


def _signature(puzzle):
    return tuple((s, _key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    corpus = [generate_puzzle(s) for s in HOST_SEEDS]
    assert len({_signature(p) for p in corpus}) == 42
    assert _signature(corpus[0]) == _signature(generate_puzzle(HOST_SEEDS[0]))
    references, records, palettes = set(), [], set()
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _analyze(_grid(pair['input']))
                references.add(min(_key(_d4(data['walls'], k)) for k in range(8)))
                palettes.add((data['wall'], data['ink'], data['marker']))
                records.append(_features(data, pair['output']))
        # Transformation and palette covariance use inputs alone.
        pair = puzzle['test'][0]
        for k in (1, 4):
            assert np.array_equal(solve(_d4(pair['input'], k)), _d4(pair['output'], k))
        before = pair['input'].copy()
        solve(before)
        assert np.array_equal(before, pair['input'])
        if before.shape[0] < 30:
            assert np.array_equal(solve(np.pad(before, ((0, 1), (0, 0)))), pair['output'])
        renaming = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 1], np.int8)
        assert np.array_equal(solve(renaming[before]), renaming[pair['output']])
    assert len(references) == 42
    assert len({f['map_shape'] for f in records}) >= 20
    assert len({f['patch_shape'] for f in records}) >= 30
    assert len({f['bends'] for f in records}) >= 20
    assert len(palettes) >= 30
    assert set(range(8)) == {f['transform'] for f in records}
    assert {2, 3, 4} <= {f['compartments'] for f in records}
    assert any(f['islands'] >= 2 for f in records)
    # Exact hand-derived contour regression: island is absent from the rubbing.
    walls = np.zeros((9, 12), bool)
    walls[:, 5] = True
    walls[0:2, 1:3] = True
    walls[4, 2] = True
    walls[4:6, 9:11] = True
    infos = _region_info(walls)
    left = next(i for i in infos if i['region'][3, 1])
    assert not left['rim'][4, 1] and not left['rim'][3, 2]
    assert left['rim'][2, 3] and left['diagonal'][2, 3]
    out = _render(walls, left, (4, 1), 1, 2, 3)
    assert out[4, 2] == 1 and np.all(out[4:6, 9:11] == 0) and out[4, 1] == 3
    expected = np.array([[int(c) for c in row] for row in (
        '211221000000', '211221000000', '222221000000',
        '200021000000', '231021000000', '200021000000',
        '200021000000', '200021000000', '222221000000')], np.int8)
    assert np.array_equal(out, expected)
    rubbing = left['rim'][:7, :5].astype(np.int8)
    rubbing[4, 1] = 2
    fixture_rng = np.random.default_rng(123)
    for k in range(8):
        inp = _pack(fixture_rng, walls, _d4(rubbing, k), (1, 2, 3))
        assert np.array_equal(solve(inp), expected)
    # A new enclosed island changes only retained structure, not the rubbing.
    extra_island = walls.copy()
    extra_island[6, 2] = True
    extra_expected = expected.copy()
    extra_expected[6, 2] = 1
    assert np.array_equal(solve(_pack(fixture_rng, extra_island, rubbing, (1, 2, 3))), extra_expected)
    # Removing far-side distractors leaves the entire selected answer unchanged.
    no_distractor = walls.copy()
    no_distractor[4:6, 9:11] = False
    assert np.array_equal(solve(_pack(fixture_rng, no_distractor, rubbing, (1, 2, 3))), expected)
    # Moving the query annotation moves the answer dot, preserving rim geometry.
    moved = rubbing.copy()
    moved[4, 1], moved[2, 3] = 0, 2
    moved_expected = expected.copy()
    moved_expected[4, 1], moved_expected[2, 3] = 0, 3
    assert np.array_equal(solve(_pack(fixture_rng, walls, moved, (1, 2, 3))), moved_expected)
    attached = walls.copy()
    attached[0:4, 2] = True
    new_left = next(i for i in _region_info(attached) if i['region'][3, 1])
    assert new_left['rim'][4, 1]
    attached_matches = _matches(_region_info(attached), rubbing)
    assert not any((r, c) == (4, 1) for _, r, c in attached_matches)
    # Two equally sized rectangles produce conflicting rubbing addresses.
    symmetric = np.zeros((9, 11), bool)
    symmetric[:, 5] = True
    ambiguous = np.array([[1, 1, 1], [1, 2, 0], [1, 0, 0]], np.int8)
    try:
        solve(_pack(fixture_rng, symmetric, ambiguous, (1, 2, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous rubbing accepted")
    # Corrupted labels must not pass.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in corpus[0][s]] for s in corpus[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    for a, b in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(12345, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")


if __name__ == '__main__':
    self_test()
    print('self_test: passed (42 seeds)')
