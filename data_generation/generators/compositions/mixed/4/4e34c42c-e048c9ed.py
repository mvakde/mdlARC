"""Register overlapping observation windows, then label complete horizontal bars.

Only standard Python and NumPy are used. No files, examples, seeds, or generation
coordinates are consulted by solve(). The visible marker anchors translation.
"""
from itertools import combinations
import hashlib
import numpy as np


_LENGTHS = (2, 3, 4, 5)
_BACKGROUNDS = (0, 2, 3, 5, 7, 8)  # Never hide digits 1, 4, 9, or 6.


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _groups(points):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], {p}
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _box(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points)+1, max(c for r, c in points)+1)


def _border(r0, c0, r1, c1):
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)
            if r in (r0, r1-1) or c in (c0, c1-1)}


def _objects(cells, marker_color):
    """Parse separated 2x2 signatures, monochrome bar pieces, and the marker."""
    keys, bars, markers = {}, [], []
    for group in _groups(cells):
        r0, c0, r1, c1 = _box(group)
        colors = {cells[p] for p in group}
        if len(group) == 1 and colors == {marker_color}:
            markers.append(next(iter(group)))
        elif len(group) == 4 and (r1-r0, c1-c0) == (2, 2) and len(colors) >= 3:
            _require(marker_color not in colors, "marker color used in a signature")
            pattern = tuple(cells[r, c] for r in range(r0, r1) for c in range(c0, c1))
            _require(pattern not in keys, "same signature at two positions in one view")
            keys[pattern] = (r0, c0)
        else:
            _require(r1-r0 == 1 and len(colors) == 1 and len(group) == c1-c0,
                     "unrecognized interior object")
            _require(marker_color not in colors, "marker color used on a bar")
            bars.append((r0, c0, c1-c0, next(iter(colors))))
    return keys, bars, markers


def _read_windows(inp):
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[counts.argmax()])
    _require(int(counts.max())*2 > inp.size, "background must occupy over half the input")
    _require(background in _BACKGROUNDS, "background would hide an arithmetic label")
    frames = []
    for color in values:
        if color == background:
            continue
        for group in _groups(zip(*np.nonzero(inp == color))):
            r0, c0, r1, c1 = _box(group)
            if r1-r0 >= 4 and c1-c0 >= 4 and group == _border(r0, c0, r1, c1):
                frames.append((r0, c0, r1, c1, int(color)))
    frames.sort()
    _require(2 <= len(frames) <= 6, "need two to six observation frames")
    frame_colors = {f[4] for f in frames}
    _require(len(frame_colors) == 1, "frames must share a color")
    marker_color = next(iter(frame_colors))
    occupied = np.zeros(inp.shape, dtype=bool)
    windows = []
    for r0, c0, r1, c1, _ in frames:
        _require(not occupied[r0:r1, c0:c1].any(), "overlapping input frames")
        # The outer rectangles have at least a background row/column between them.
        for other in windows:
            a, b, d, e = other['frame']
            _require(r1 < a or d < r0 or c1 < b or e < c0, "input frames touch")
        occupied[r0:r1, c0:c1] = True
        patch = inp[r0+1:r1-1, c0+1:c1-1].copy()
        cells = {(int(r), int(c)): int(patch[r, c])
                 for r, c in zip(*np.nonzero(patch != background))}
        keys, pieces, markers = _objects(cells, marker_color)
        _require(bool(keys), "every observation needs an exact signature")
        windows.append({'frame': (r0, c0, r1, c1), 'patch': patch,
                        'cells': cells, 'keys': keys, 'pieces': pieces, 'markers': markers})
    _require(bool(np.all(inp[~occupied] == background)), "foreground outside frames")
    markers = [(i, p) for i, w in enumerate(windows) for p in w['markers']]
    _require(len(markers) == 1, "exactly one interior reference marker is required")
    return background, marker_color, windows, markers[0]


def _register(inp):
    background, marker_color, windows, (anchor, marker_local) = _read_windows(inp)
    occurrences = {}
    for i, w in enumerate(windows):
        for pattern, point in w['keys'].items():
            occurrences.setdefault(pattern, []).append((i, point))
    adjacency = [[] for _ in windows]
    for copies in occurrences.values():
        for (i, a), (j, b) in combinations(copies, 2):
            delta = a[0]-b[0], a[1]-b[1]
            adjacency[i].append((j, delta))
            adjacency[j].append((i, (-delta[0], -delta[1])))
    ar, ac, _, _ = windows[anchor]['frame']
    origins = {anchor: (ar+1, ac+1)}
    stack = [anchor]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in adjacency[i]:
            candidate = origins[i][0]+dr, origins[i][1]+dc
            if j in origins:
                _require(origins[j] == candidate, "inconsistent exact signatures")
            else:
                origins[j] = candidate
                stack.append(j)
    _require(len(origins) == len(windows), "disconnected registration graph")
    # All matching keys impose equations, so a connected consistent graph has
    # exactly one placement once the visible marker window is held stationary.
    observed, union = {}, {}
    for i, w in enumerate(windows):
        dr, dc = origins[i]
        h, width = w['patch'].shape
        _require(0 <= dr and 0 <= dc and dr+h <= inp.shape[0]
                 and dc+width <= inp.shape[1], "registered view crosses the canvas boundary")
        for r in range(h):
            for c in range(width):
                p, color = (r+dr, c+dc), int(w['patch'][r, c])
                _require(p not in observed or observed[p] == color, "overlap disagrees, including blank context")
                observed[p] = color
                if color != background:
                    union[p] = color
    keys, bars, markers = _objects(union, marker_color)
    marker = marker_local[0]+origins[anchor][0], marker_local[1]+origins[anchor][1]
    _require(markers == [marker], "assembled marker is not unique")
    _require(1 <= len(bars) <= 6, "need one to six assembled bars")
    _require(len({b[0] for b in bars}) == len(bars), "two bars share an assembled row")
    key_rows = {r+d for r, c in keys.values() for d in (0, 1)}
    for r, c, length, color in bars:
        _require(length in _LENGTHS, "assembled bar length must be 2..5")
        _require(r != marker[0] and r not in key_rows, "bar row has another role")
        _require(c+length < marker[1] or c > marker[1]+1, "bar needs a blank gap from reference column")
        _require((r, marker[1]) not in union, "label would overwrite material")
    base = np.full_like(inp, background)
    for point, color in union.items():
        base[point] = color
    incomplete, visible_lengths = [], []
    for r, c, length, color in bars:
        visible = [n for i, w in enumerate(windows) for rr, cc, n, v in w['pieces']
                   if rr+origins[i][0] == r and v == color
                   and c <= cc+origins[i][1] < c+length]
        visible_lengths.append(max(visible, default=0))
        incomplete.append(visible_lengths[-1] < length)
    return {'background': background, 'marker_color': marker_color, 'windows': windows,
            'origins': origins, 'marker': marker, 'bars': bars, 'base': base,
            'keys': keys, 'incomplete': incomplete, 'visible_lengths': visible_lengths}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution, also valid for other coherent covers in this family."""
    info = _register(_grid(input_grid))
    out = info['base'].copy()
    for r, c, length, color in info['bars']:
        out[r, info['marker'][1]] = (length-1)**2 % 10
    return out


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                    ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name+' must be an integer')
        if not low <= value <= high:
            raise ValueError(name+' is outside its legal range')
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _scene(rng, lengths, colors, background, frame_color, geometry, layout):
    n = len(lengths)
    width = int(rng.integers(16 if geometry == 'mixed' else 14, 18))
    # Keep the interior singleton disconnected from its observation frame.
    reference = width//2 if geometry == 'mixed' else (width-2 if geometry == 'left' else 1)
    if layout == 'three':
        cuts = [3, width-5]
    elif geometry == 'mixed':
        cuts = [int(rng.choice([3, width-5]))]
    else:
        cuts = [int(rng.integers(3, width-5))]
    # Unequal vertical gaps keep rows geometric rather than an ordinal code.
    upper_count = max(1, n//2)
    rows = [3]
    for _ in range(upper_count-1):
        rows.append(rows[-1]+int(rng.integers(2, 4)))
    middle = rows[-1]+2
    lower_rows = [middle+3] if n > upper_count else []
    for _ in range(n-upper_count-1):
        lower_rows.append(lower_rows[-1]+int(rng.integers(2, 4)))
    rows += lower_rows
    height = max(rows[-1]+1, middle+2)+int(rng.integers(0, 3))
    scene = np.full((height, width), background, dtype=np.int8)
    if layout in ('two', 'three'):
        key_positions = [(r, c) for r in (0, middle) for c in cuts]
    else:
        key_positions = [(0, cuts[0])]
        if layout in ('grid', 'right_full'):
            key_positions.append((middle, 1))
        if layout in ('grid', 'left_full'):
            key_positions.append((middle, width-3))
    palette = [x for x in range(10) if x not in (background, frame_color)]
    patterns = set()
    for r, c in key_positions:
        for _ in range(64):
            k = int(rng.integers(3, 5))
            chosen = [int(v) for v in rng.choice(palette, k, replace=False)]
            indices = list(range(k))+[int(rng.integers(k)) for _ in range(4-k)]
            pattern = tuple(chosen[int(i)] for i in rng.permutation(indices))
            if pattern not in patterns:
                patterns.add(pattern)
                scene[r:r+2, c:c+2] = np.array(pattern, dtype=np.int8).reshape(2, 2)
                break
        else:
            raise ValueError("bounded signature sampling exhausted")
    records = []
    for i, (r, length, color) in enumerate(zip(rows, lengths, colors)):
        candidates = [c for c in range(width-length+1)
                      if c+length < reference or c > reference+1]
        if i == 0:
            # Both overlapping views see strictly less than the full first bar.
            candidates = [c for c in candidates if any(c == cut-1 for cut in cuts)]
        elif i == 1 and geometry == 'mixed':
            # Even a two-demonstration puzzle visibly teaches both sides.
            first_left = records[0][1] < reference
            candidates = [c for c in candidates if (c < reference) != first_left]
        _require(bool(candidates), "no compatible bar position")
        c = int(rng.choice(candidates))
        scene[r, c:c+length] = color
        records.append((r, c, length, color))
    cut = cuts[0]
    if layout == 'two':
        views = [(0, 0, height, cut+2), (0, cut, height, width)]
    elif layout == 'three':
        views = [(0, 0, height, cuts[0]+2), (0, cuts[0], height, cuts[1]+2),
                 (0, cuts[1], height, width)]
    elif layout == 'grid':
        views = [(0, 0, middle+2, cut+2), (0, cut, middle+2, width),
                 (middle, 0, height, cut+2), (middle, cut, height, width)]
    elif layout == 'left_full':
        views = [(0, 0, height, cut+2), (0, cut, middle+2, width),
                 (middle, cut, height, width)]
    else:
        views = [(0, 0, middle+2, cut+2), (middle, 0, height, cut+2),
                 (0, cut, height, width)]
    # The singleton travels with an unambiguous carrier window. It is not an
    # unsupported registration feature. Leave a gap from every carrier edge.
    marker_options = []
    for r in range(1, height-1):
        containing = [i for i, (a, b, d, e) in enumerate(views)
                      if a <= r < d and b <= reference < e]
        if len(containing) != 1 or r in rows:
            continue
        a, b, d, e = views[containing[0]]
        if not (a < r < d-1 and b < reference < e-1):
            continue
        if np.all(scene[r-1:r+2, max(0, reference-1):reference+2] == background):
            marker_options.append((r, containing[0]))
    _require(bool(marker_options), "no isolated marker position")
    marker_row, anchor = marker_options[int(rng.integers(len(marker_options)))]
    scene[marker_row, reference] = frame_color
    return scene, views, anchor, records


def _pack(rng, scene, views, anchor, background, frame_color):
    """Bounded shelf placement; no pixels or logs from failed trials are saved."""
    shapes = [(d-a+2, e-b+2) for a, b, d, e in views]
    n = len(views)
    partitions = {2: [(2,), (1, 1)], 3: [(3,), (1, 2), (2, 1)],
                  4: [(4,), (2, 2), (1, 3), (3, 1)]}[n]
    for attempt in range(192):
        order = [int(x) for x in rng.permutation(n)]
        partition = partitions[int(rng.integers(len(partitions)))]
        axis = int(rng.integers(2))
        positions, offset, at = {}, 0, 0
        gap = int(rng.integers(1, 3))
        for count in partition:
            members = order[at:at+count]
            at += count
            depth = max(shapes[i][axis] for i in members)
            across = 0
            for i in members:
                point = [0, 0]
                point[axis] = offset+int(rng.integers(depth-shapes[i][axis]+1))
                point[1-axis] = across
                positions[i] = tuple(point)
                across += shapes[i][1-axis]+gap
            offset += depth+gap
        ar, ac = positions[anchor]
        va, vb, _, _ = views[anchor]
        scene_top, scene_left = ar+1-va, ac+1-vb
        r0 = min(0, scene_top)
        c0 = min(0, scene_left)
        r1 = max(scene_top+scene.shape[0], max(positions[i][0]+shapes[i][0] for i in range(n)))
        c1 = max(scene_left+scene.shape[1], max(positions[i][1]+shapes[i][1] for i in range(n)))
        min_h, min_w = r1-r0, c1-c0
        if min_h > 30 or min_w > 30:
            continue
        foreground = sum(2*h+2*w-4 + int(np.count_nonzero(scene[a:d, b:e] != background))
                         for (h, w), (a, b, d, e) in zip(shapes, views))
        sizes = [(h, w) for h in range(min_h, min(30, min_h+5)+1)
                 for w in range(min_w, min(30, min_w+5)+1) if h*w > 2*foreground]
        if not sizes:
            continue
        h, w = sizes[int(rng.integers(len(sizes)))]
        dr = -r0+int(rng.integers(h-min_h+1))
        dc = -c0+int(rng.integers(w-min_w+1))
        inp = np.full((h, w), background, dtype=np.int8)
        for i, (a, b, d, e) in enumerate(views):
            top, left = positions[i][0]+dr, positions[i][1]+dc
            ph, pw = shapes[i]
            inp[top:top+ph, left:left+pw] = frame_color
            inp[top+1:top+ph-1, left+1:left+pw-1] = scene[a:d, b:e]
        origin = scene_top+dr, scene_left+dc
        return inp, origin
    raise ValueError("bounded packer could not fit observation windows")


def _make_pair(rng, lengths, colors, background, frame_color, geometry, layout):
    # Draw a complete intended scene first. The expected answer is rendered from
    # this construction, never obtained by calling solve on the packed input.
    scene, views, anchor, records = _scene(rng, lengths, colors, background,
                                          frame_color, geometry, layout)
    inp, (top, left) = _pack(rng, scene, views, anchor, background, frame_color)
    out = np.full_like(inp, background)
    out[top:top+scene.shape[0], left:left+scene.shape[1]] = scene
    reference = int(np.argwhere(scene == frame_color)[0, 1])
    for r, c, length, color in records:
        out[top+r, left+reference] = (length-1)**2 % 10
    # Registration is checked against the independently rendered construction.
    _require(np.array_equal(solve(inp), out), "construction and reconstruction disagree")
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    background = int(rng.choice(_BACKGROUNDS))
    frame_color = int(rng.choice([c for c in range(10) if c != background]))
    a, b = [int(x) for x in rng.choice([c for c in range(10)
                                      if c not in (background, frame_color)], 2, replace=False)]
    mixed = bool(rng.integers(2))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            if split == 'train' and i == 0:
                lengths, colors, layout = [4, 2, 3], [a, a, a], 'two'
            elif split == 'train' and i == 1:
                lengths, colors = [5, 4, 2], [a, b, b]
                layout = str(rng.choice(['left_full', 'right_full']))
            else:
                n = int(rng.integers(1 if split == 'train' else 2, 6))
                lengths = [int(rng.choice([4, 5]))]+[int(rng.choice(_LENGTHS)) for _ in range(n-1)]
                palette = [c for c in range(10) if c not in (background, frame_color)]
                colors = [int(rng.choice(palette)) for _ in lengths]
                layout = (str(rng.choice(['two', 'three', 'left_full', 'right_full']))
                          if split == 'train' and i > 2 else 'two')
                if split == 'test':
                    layout = str(rng.choice(['grid', 'left_full', 'right_full']))
            geometry = 'mixed' if mixed else ('left' if i % 2 == 0 else 'right')
            if split == 'test' and not mixed:
                geometry = str(rng.choice(['left', 'right']))
            for attempt in range(48):
                try:
                    pair = _make_pair(rng, lengths, colors, background, frame_color, geometry, layout)
                except ValueError:
                    continue
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded generation exhausted its attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair keys")
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, "canvas dimensions changed")
    info = _register(inp)
    expected = info['base'].copy()
    for r, c, length, color in info['bars']:
        expected[r, info['marker'][1]] = (length-1)**2 % 10
    _require(np.array_equal(out, expected), "wrong registered material or arithmetic labels")
    _require(np.count_nonzero(out != info['base']) == len(info['bars']), "one visible label per bar required")
    _require(any(info['incomplete']), "registration must complete at least one bar")
    _require(any((visible-1)**2 % 10 != (bar[2]-1)**2 % 10
                 for bar, visible in zip(info['bars'], info['visible_lengths'])),
             "clipped-piece measurement must change an arithmetic label")
    return info


def validate_generated_puzzle(puzzle):
    """Check schema, visible reconstruction, causal dependence, and teaching."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(_grid(pair['input']))
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            infos[split].append(info)
    _require(len({i['background'] for group in infos.values() for i in group}) == 1,
             "generated puzzle changes background convention")
    _require(len({i['marker_color'] for group in infos.values() for i in group}) == 1,
             "generated puzzle changes frame/marker color")
    by_length, by_color = {}, {}
    for info in infos['train']:
        for r, c, length, color in info['bars']:
            by_length.setdefault(length, set()).add(color)
            by_color.setdefault(color, set()).add(length)
    _require(set(by_length) == set(_LENGTHS), "training must demonstrate lengths 2..5")
    _require(any(len(v) >= 2 for v in by_length.values()), "fixed length needs two source colors")
    _require(any(len(v) >= 2 for v in by_color.values()), "fixed source color needs two lengths")
    _require(any(len({p[0] for p in info['origins'].values()}) >= 2 for info in infos['train']),
             "training must teach partial-height registration")
    for info in infos['test']:
        _require(all(length in by_length for r, c, length, color in info['bars']), "untaught test length")
        _require(len({p[0] for p in info['origins'].values()}) >= 2,
                 "test must combine partial-height observations")
    train_sides = {-1 if c < info['marker'][1] else 1
                   for info in infos['train'] for r, c, length, color in info['bars']}
    test_sides = {-1 if c < info['marker'][1] else 1
                  for info in infos['test'] for r, c, length, color in info['bars']}
    _require(test_sides <= train_sides, "projection side must be demonstrated")
    return None


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, count boundaries, deterministic and negative checks.

    Holds at most two puzzles at once; no disk I/O, logs, caches, or traces.
    """
    digests, shapes, layouts, keys, backgrounds, frames, counts = set(), set(), set(), set(), set(), set(), set()
    side_sets, row_offsets = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "list input failed")
                shapes.add(pair['input'].shape)
                backgrounds.add(info['background'])
                frames.add(info['marker_color'])
                counts.add(len(info['bars']))
                keys.update(info['keys'])
                layouts.add(tuple((r-info['marker'][0], c-info['marker'][1], n)
                                  for r, c, n, color in info['bars']))
                side_sets.add(tuple(sorted({-1 if c < info['marker'][1] else 1
                                            for r, c, n, color in info['bars']})))
                row_offsets.add(len({r for r, c in info['origins'].values()}))
    _require(len(digests) == 42, "duplicate seeds")
    _require(len(shapes) >= 30 and len(layouts) >= 80 and len(keys) >= 100, "insufficient structural variation")
    _require(backgrounds == set(_BACKGROUNDS) and len(frames) >= 8, "insufficient palette-role variation")
    _require(counts >= {1, 2, 3, 4, 5} and (-1, 1) in side_sets, "bar count/side coverage missing")
    _require(max(row_offsets) >= 2, "no two-dimensional registration")
    _require(_puzzle_digest(generate_puzzle(17)) == _puzzle_digest(generate_puzzle(17)), "nondeterministic")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(42+10*nt+nq, nt, nq)
            _require((len(puzzle['train']), len(puzzle['test'])) == (nt, nq), "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0)
    for split in ('train', 'test'):
        for p in puzzle[split]:
            for axis in (0, 1):
                _require(np.array_equal(solve(np.flip(p['input'], axis)), np.flip(p['output'], axis)),
                         "whole-example reflection changed the rule")
    pair = puzzle['train'][1]
    info = _inspect_pair(pair)
    # A: labeling clipped pieces, even after their rows are known, is wrong.
    wrong = info['base'].copy()
    for (r, c, length, color), visible_length in zip(info['bars'], info['visible_lengths']):
        wrong[r, info['marker'][1]] = (visible_length-1)**2 % 10
    _require(not np.array_equal(wrong, pair['output']), "registration omission did not change labels")
    # B: replace the required conversion by raw length, or omit the labels.
    wrong = info['base'].copy()
    for r, c, length, color in info['bars']:
        wrong[r, info['marker'][1]] = length
    _require(not np.array_equal(wrong, pair['output']), "raw-length shortcut survived")
    _require(not np.array_equal(info['base'], pair['output']), "arithmetic omission survived")
    bad_pair = {'input': pair['input'], 'output': wrong}
    try:
        _inspect_pair(bad_pair)
    except ValueError:
        pass
    else:
        raise AssertionError("bad labels accepted")
    # Pixel-level malformed inputs must not be silently registered or guessed.
    first = puzzle['train'][0]
    meta = _inspect_pair(first)
    no_marker = first['input'].copy()
    no_marker[meta['marker']] = meta['background']  # Anchor remains at input position.
    inconsistent = first['input'].copy()
    window = meta['windows'][0]
    kr, kc = next(iter(window['keys'].values()))
    fr, fc, _, _ = window['frame']
    point = fr+1+kr, fc+1+kc
    inconsistent[point] = next(c for c in range(10)
                               if c not in (meta['background'], meta['marker_color'], int(inconsistent[point])))
    for bad_input in (no_marker, inconsistent):
        try:
            solve(bad_input)
        except ValueError:
            pass
        else:
            raise AssertionError("missing marker or inconsistent overlap accepted")

