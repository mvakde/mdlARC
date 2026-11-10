"""Cyclic seam jigsaw: joint translation-only exact cover and cyclic path matching.

The solver reads only the supplied grid. No parent code or generated examples are
imported. A chamber's empty terrain defines a dry staircase; the legend constrains
the colors of the tiles that must cover it, rather than repainting those tiles.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, excluded=()):
    remaining = {tuple(map(int, p)) for p in np.argwhere(~np.isin(a, list(excluded)))}
    result = []
    while remaining:
        first = min(remaining)
        color = int(a[first])
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and int(a[p]) == color:
                    remaining.remove(p)
                    stack.append(p)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _orient(a, code):
    return np.rot90(a[:, ::-1] if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -code % 4)
    return (b[:, ::-1] if code >= 4 else b).copy()


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _trace(mask, start):
    """Maximal right/up legs, counting every visited cell exactly once."""
    directions = ((0, 1), (-1, 0))
    current, d = start, 0
    path, collisions, legs = [], [], []
    while True:
        r, c = current
        dr, dc = directions[d]
        leg = []
        while _inside(mask.shape, (r, c)) and mask[r, c]:
            leg.append((r, c))
            r, c = r+dr, c+dc
        if not leg:
            raise ValueError("empty staircase leg")
        path.extend(leg)
        legs.append(len(leg))
        if not _inside(mask.shape, (r, c)):
            end = "edge"
            break
        collisions.append((r, c))
        d = 1-d
        dr, dc = directions[d]
        current = leg[-1][0]+dr, leg[-1][1]+dc
        if not _inside(mask.shape, current) or not mask[current]:
            end = "blocked"
            break
    return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(legs), end=end)


def _step_trace(mask, start):
    """Separate cell-step check for bends and collision/edge termination."""
    r, c = start
    up = False
    path, collisions, legs = [], [], [0]
    while True:
        assert mask[r, c] and (r, c) not in path
        path.append((r, c))
        legs[-1] += 1
        ahead = (r-1, c) if up else (r, c+1)
        if not _inside(mask.shape, ahead):
            end = "edge"
            break
        if mask[ahead]:
            r, c = ahead
            continue
        collisions.append(ahead)
        turn = (r, c+1) if up else (r-1, c)
        if not _inside(mask.shape, turn) or not mask[turn]:
            end = "blocked"
            break
        up = not up
        legs.append(0)
        r, c = turn
    return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(legs), end=end)


def _canonical_decode(a, bg):
    comps = _components(a, (bg,))
    frames = []
    for cells, color in comps:
        r0, c0 = map(int, cells.min(axis=0))
        r1, c1 = map(int, cells.max(axis=0))
        if r0 < 8 or r1-r0 < 5 or c1-c0 < 6:
            continue
        block = a[r0:r1+1, c0:c1+1]
        if not (np.all(block[[0, -1], :] == color)
                and np.all(block[:, [0, -1]] == color)):
            continue
        special = np.argwhere(~np.isin(block, (bg, color)))
        if special.shape != (1, 2) or tuple(special[0]) != (r1-r0-1, 1):
            continue
        if np.count_nonzero(block == bg) < 8:
            continue
        frames.append((r0, c0, r1, c1, color))
    if len(frames) != 1:
        raise ValueError("no unique framed chamber in this orientation")
    r0, c0, r1, c1, terrain = frames[0]
    swatches = []
    for cells, color in comps:
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        if (len(cells) == 4 and tuple(hi-lo) == (1, 1) and hi[0] <= 6):
            swatches.append((int(lo[0]), int(lo[1]), color, cells))
    rows = {s[0] for s in swatches if s[0] in (1, 2)}
    if len(rows) != 1:
        raise ValueError("legend must be one aligned row above the chamber")
    row = rows.pop()
    legend = sorted([s for s in swatches if s[0] == row], key=lambda s: s[1])
    markers = [s for s in swatches if s[0] == row+3]
    if not (2 <= len(legend) <= 6 and len(markers) == 1
            and len(swatches) == len(legend)+1):
        raise ValueError("invalid program/marker swatches")
    if any(b[1]-s[1] < 3 for s, b in zip(legend, legend[1:])):
        raise ValueError("legend entries must be separated")
    word = tuple(s[2] for s in legend)
    marker = markers[0][2]
    start = (r1-1, c0+1)
    if (a[start] != word[0] or len(set(word)) < 2 or marker in word
            or {bg, terrain, marker} & set(word) or marker in (bg, terrain)):
        raise ValueError("invalid visible color roles/start")
    annotation = {tuple(p) for s in legend+markers for p in s[3]}
    header = {tuple(p) for p in np.argwhere(a[:7] != bg)}
    if annotation != header:
        raise ValueError("unexpected header content")
    base = a.copy()
    for p in annotation:
        base[p] = bg
    base[start] = bg
    pieces = []
    for cells, color in _components(base, (bg, terrain)):
        if color not in word or np.any(cells[:, 0] < 7):
            raise ValueError("a tile has an invalid color or location")
        if np.any((cells[:, 0] >= r0) & (cells[:, 0] <= r1)
                  & (cells[:, 1] >= c0) & (cells[:, 1] <= c1)):
            raise ValueError("movable tiles must be outside the chamber")
        pieces.append((cells, color))
        base[cells[:, 0], cells[:, 1]] = bg
    if not 3 <= len(pieces) <= 9:
        raise ValueError("expected 3..9 detached connected tiles")
    mask = np.zeros_like(a, dtype=bool)
    mask[r0+1:r1, c0+1:c1] = base[r0+1:r1, c0+1:c1] == bg
    if sum(len(p) for p, _ in pieces) != int(mask.sum()):
        raise ValueError("tile area must exactly equal the chamber's empty area")
    if not mask[start] or not mask[start[0], start[1]+1]:
        raise ValueError("start needs an inward neighbor")
    record = _trace(mask, start)
    if any(base[p] != terrain for p in record['collisions']):
        raise ValueError("collisions must hit fixed terrain or the frame")
    return dict(grid=a, background=bg, terrain=terrain, marker=marker,
                frame=(r0, c0, r1, c1), word=word, start=start,
                annotation=annotation, base=base, mask=mask,
                pieces=pieces, record=record)


def _decode(value):
    a = _grid(value)
    bg = int(a[0, 0])
    if not (np.all(a[[0, -1], :] == bg) and np.all(a[:, [0, -1]] == bg)):
        raise ValueError("the canvas must have an unbroken background rim")
    candidates = []
    for code in range(8):
        try:
            info = _canonical_decode(_unorient(a, code), bg)
        except ValueError:
            continue
        info['orientation'] = code
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("input has no unique semantic orientation")
    return candidates[0]


def _packing(info, use_cycle=True, limit=2):
    """Enumerate translations and search all exact covers, deduplicating colors.

    Without the cycle we still preserve the visible start's first color. Thus the
    ablation is stronger than simply forgetting every annotation.
    """
    mask, pieces = info['mask'], info['pieces']
    positions = [tuple(map(int, p)) for p in np.argwhere(mask)]
    index = {p: j for j, p in enumerate(positions)}
    full = (1 << len(positions))-1
    required = {p: info['word'][j % len(info['word'])]
                for j, p in enumerate(info['record']['path'])} if use_cycle else {
                    info['start']: info['word'][0]}
    r0, c0, r1, c1 = info['frame']
    options, by_cell, by_piece, equivalent = [], [[] for _ in positions], [], {}
    for i, (source, color) in enumerate(pieces):
        local = source-source.min(axis=0)
        identity = (color, tuple(map(tuple, local)))
        equivalent[i] = next((j for j in range(i) if equivalent.get(('id', j)) == identity), None)
        equivalent[('id', i)] = identity
        ph, pw = local.max(axis=0)+1
        own = []
        for r in range(r0+1, r1-int(ph)+1):
            for c in range(c0+1, c1-int(pw)+1):
                target = local+(r, c)
                cells = [tuple(map(int, p)) for p in target]
                if any(p not in index or (p in required and required[p] != color) for p in cells):
                    continue
                bits = sum(1 << index[p] for p in cells)
                k = len(options)
                options.append((i, bits, target))
                own.append(k)
                for p in cells:
                    by_cell[index[p]].append(k)
        by_piece.append(own)
    solutions = {}
    active_limit = 1

    def search(used, occupied, chosen):
        if len(solutions) >= active_limit:
            return
        if occupied == full:
            if used != (1 << len(pieces))-1:
                return
            out = info['base'].copy()
            placements = []
            for k in chosen:
                i, _, target = options[k]
                out[target[:, 0], target[:, 1]] = pieces[i][1]
                placements.append((i, target))
            solutions.setdefault(out.tobytes(), (out, placements))
            return
        available = []
        valid = {}
        for i in range(len(pieces)):
            if used & (1 << i):
                continue
            valid[i] = [k for k in by_piece[i] if not options[k][1] & occupied]
            if not valid[i]:
                return
            # Identical same-colored masks are indistinguishable in the answer.
            previous = equivalent[i]
            if previous is None or used & (1 << previous):
                available.append(valid[i])
        missing = full ^ occupied
        while missing:
            bit = missing & -missing
            j = bit.bit_length()-1
            candidates = [k for k in by_cell[j]
                          if not used & (1 << options[k][0]) and not options[k][1] & occupied
                          and (equivalent[options[k][0]] is None
                               or used & (1 << equivalent[options[k][0]]))]
            # Equivalent pieces may be interchangeable; choosing their earliest
            # unused copy at this cell is always sufficient.
            if not candidates:
                return
            available.append(candidates)
            missing ^= bit
        for k in min(available, key=len):
            i, bits, _ = options[k]
            search(used | (1 << i), occupied | bits, chosen+[k])

    # First obtain one cover. For colored-output uniqueness, another answer must
    # contain a placement that paints at least one cell differently from this
    # reference. Force every such placement in turn. This is exhaustive over
    # different colored answers without enumerating equivalent same-color tilings.
    search(0, 0, [])
    if not solutions or limit == 1:
        return list(solutions.values())
    reference = next(iter(solutions.values()))[0]
    active_limit = limit
    for k, (i, bits, target) in enumerate(options):
        if len(solutions) >= limit:
            break
        if equivalent[i] is not None:
            continue  # An identical copy can be renamed to the first copy.
        if np.any(reference[target[:, 0], target[:, 1]] != pieces[i][1]):
            search(1 << i, bits, [k])
    return list(solutions.values())


def _answer(info):
    solutions = _packing(info)
    if len(solutions) != 1:
        raise ValueError("the combined shape/program constraints must give one colored answer")
    assembled, placements = solutions[0]
    out = assembled.copy()
    for p in info['record']['collisions']:
        out[p] = info['marker']
    return _orient(out, info['orientation']), placements


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer(_decode(input_grid))[0]


def _coverage(info):
    word, record = info['word'], info['record']
    n = len(record['path'])
    actual = [word[j % len(word)] for j in range(n)]
    unique = tuple(dict.fromkeys(word))
    reset, consume, phase = [], [], 0
    for leg_index, length in enumerate(record['legs']):
        reset.extend(word[j % len(word)] for j in range(length))
        consume.extend(word[(phase+j+leg_index) % len(word)] for j in range(length))
        phase += length
    evidence = set()
    if n >= 2*len(word):
        evidence.add('cycling')
    if actual != [unique[j % len(unique)] for j in range(n)]:
        evidence.add('multiplicity')
    if actual != reset:
        evidence.add('bend_phase')
    if actual != consume:
        evidence.add('marker_phase')
    if any(len(cells) < int(np.prod(cells.max(axis=0)-cells.min(axis=0)+1))
           for cells, _ in info['pieces']):
        evidence.add('shape')
    if int(info['mask'].sum()) > n+3:
        evidence.add('off_seam')
    solutions = _packing(info)
    if len(solutions) == 1:
        owner = {tuple(p): i for i, cells in solutions[0][1] for p in cells}
        wrong, phase, previous = [], -1, None
        for p in record['path']:
            if owner[p] != previous:
                phase += 1
                previous = owner[p]
            wrong.append(word[phase % len(word)])
        if actual != wrong:
            evidence.add('cell_clock')
    return evidence


_REQUIRED = {'cycling', 'multiplicity', 'bend_phase', 'marker_phase', 'shape', 'off_seam', 'cell_clock'}


def _gap(rng, h, w):
    """Independent interval profiles create broad chambers and staircase bays."""
    knots = np.unique(np.r_[0, rng.choice(np.arange(1, h-1),
                                        size=min(h-2, 3), replace=False), h-1])
    left = np.rint(np.interp(np.arange(h), knots,
                            rng.integers(0, max(2, w-2), len(knots)))).astype(int)
    right = np.rint(np.interp(np.arange(h), knots,
                             rng.integers(2, w, len(knots)))).astype(int)
    left[-1] = 0
    right[-1] = int(rng.integers(2, w-1))
    right = np.maximum(right, left+1)
    right = np.minimum(right, w-1)
    left = np.minimum(left, right-1)
    for r in range(h-2, -1, -1):
        left[r] = min(left[r], right[r+1])
        right[r] = max(right[r], left[r+1])
    gap = np.zeros((h+2, w+2), dtype=bool)
    for r in range(h):
        gap[r+1, left[r]+1:right[r]+2] = True
    return gap


def _partition(rng, gap, record, word):
    path = record['path']
    colors = [word[j % len(word)] for j in range(len(path))]
    # Two equal masks in different colors touch the seam at one cell apiece and
    # extend OFF it. The cyclic constraint decides the color of each whole tile;
    # arbitrary geometric assembly followed by seam repainting cannot do that.
    stencils = [((0, 0), (dr, dc)) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    stencils += [((0, 0), (-1, 0), (-1, s)) for s in (-1, 1)]
    stencils += [((0, 0), (0, s), (-1, s)) for s in (-1, 1)]
    stencils += [((0, 0), (-1, 0), (-2, 0))]
    path_set = set(path)
    choices = []
    for stencil in stencils:
        footprints = {}
        for j in range(1, len(path)):
            r, c = path[j]
            cells = {(r+dr, c+dc) for dr, dc in stencil}
            if (all(_inside(gap.shape, p) and gap[p] for p in cells)
                    and cells & path_set == {path[j]}):
                footprints[j] = cells
        pairs = [(a, b) for a, b in combinations(footprints, 2)
                 if colors[a] != colors[b] and not footprints[a] & footprints[b]]
        if pairs:
            choices.append((footprints, pairs))
    if not choices:
        raise ValueError("need two equal seam-addressed masks with off-seam cells")
    footprints, pairs = choices[int(rng.integers(len(choices)))]
    a, b = pairs[int(rng.integers(len(pairs)))]
    protected = footprints[a] | footprints[b]
    labels = np.full(gap.shape, -1, dtype=int)
    tile_colors = [colors[a], colors[b]]
    for i, j in enumerate((a, b)):
        for p in footprints[j]:
            labels[p] = i
    for j, p in enumerate(path):
        if p in protected:
            continue
        if j and path[j-1] not in protected and colors[j] == colors[j-1]:
            labels[p] = labels[path[j-1]]
        else:
            labels[p] = len(tile_colors)
            tile_colors.append(colors[j])
    if len(tile_colors) > 9:
        raise ValueError("too many seam runs")
    while np.any(gap & (labels < 0)):
        frontier = {}
        for r, c in np.argwhere(gap & (labels < 0)):
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                i = int(labels[p])
                if i >= 2:
                    frontier.setdefault(i, set()).add((int(r), int(c)))
        if not frontier:
            p = tuple(map(int, np.argwhere(gap & (labels < 0))[0]))
            labels[p] = len(tile_colors)
            tile_colors.append(int(rng.choice(word)))
            if len(tile_colors) > 9:
                raise ValueError("too many disconnected off-seam regions")
            continue
        i = int(rng.choice(sorted(frontier)))
        choices = sorted(frontier[i])
        labels[choices[int(rng.integers(len(choices)))]] = i
    pieces = [(np.argwhere(labels == i), color) for i, color in enumerate(tile_colors)]
    if not any(len(cells) < int(np.prod(cells.max(axis=0)-cells.min(axis=0)+1)) for cells, _ in pieces):
        raise ValueError("need complementary nonrectangular masks")
    return pieces


def _dilate(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[r:r+mask.shape[0], c:c+mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def _scatter(rng, inp, pieces, forbidden):
    for _ in range(16):
        out = inp.copy()
        blocked = forbidden.copy()
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0])+float(rng.random())*.5)
        success = True
        for i in order:
            cells, color = pieces[i]
            local = cells-cells.min(axis=0)
            ph, pw = map(int, local.max(axis=0)+1)
            nh, nw = out.shape[0]-ph+1, out.shape[1]-pw+1
            valid = np.ones((nh, nw), dtype=bool)
            for r, c in local:
                valid &= ~blocked[r:r+nh, c:c+nw]
            choices = np.argwhere(valid)
            if not len(choices):
                success = False
                break
            placed = local+choices[int(rng.integers(len(choices)))]
            out[placed[:, 0], placed[:, 1]] = color
            occupied = np.zeros_like(blocked)
            occupied[placed[:, 0], placed[:, 1]] = True
            blocked |= _dilate(occupied)
        if success:
            return out
    raise ValueError("not enough clear scattering space")


def _sample_word(rng, witness, expanded):
    if witness:
        return (0, 0, 1)
    n = int(rng.integers(2, 7 if expanded else 6))
    d = int(rng.integers(2, min(n, 4)+1))
    word = list(range(d))+list(map(int, rng.integers(d, size=n-d)))
    rng.shuffle(word)
    return tuple(word)


def _make_pair(rng, orientation, witness=False, expanded=False):
    for _ in range(500):
        h = int(rng.integers(4, 9 if expanded else 8))
        w = int(rng.integers(5, 11 if expanded else 10))
        gap = _gap(rng, h, w)
        record = _trace(gap, (h, 1))
        n = len(record['path'])
        if not 6 <= n <= 18 or int(gap.sum()) < n+4:
            continue
        raw_word = _sample_word(rng, witness, expanded)
        palette = tuple(map(int, rng.permutation(10)))
        bg, terrain, marker = palette[:3]
        word = tuple(palette[3+x] for x in raw_word)
        try:
            tiles = _partition(rng, gap, record, word)
        except ValueError:
            continue
        scene_h = int(rng.integers(max(18, h+12), 31))
        scene_w = int(rng.integers(max(20, 3*len(word)+3, 2*w+4), 31))
        r0 = int(rng.integers(8, scene_h-h-2))
        c0 = int(rng.integers(1, scene_w-w-2))
        base = np.full((scene_h, scene_w), bg, dtype=np.int8)
        base[r0:r0+h+2, c0:c0+w+2] = np.where(gap, bg, terrain)
        inp = base.copy()
        row = int(rng.integers(1, 3))
        slack = scene_w-(3*len(word)-1)-2
        gaps = np.ones(len(word)-1, dtype=int)
        for j in rng.permutation(len(gaps)):
            extra = int(rng.integers(0, min(2, slack)+1))
            gaps[j] += extra
            slack -= extra
        col = int(rng.integers(1, slack+2))
        columns = [col]
        for gap_size in gaps:
            columns.append(columns[-1]+2+int(gap_size))
        for col, color in zip(columns, word):
            inp[row:row+2, col:col+2] = color
        mc = int(rng.integers(1, scene_w-2))
        inp[row+3:row+5, mc:mc+2] = marker
        forbidden = np.zeros_like(inp, dtype=bool)
        forbidden[:8, :] = True
        forbidden[[0, -1], :] = True
        forbidden[:, [0, -1]] = True
        chamber = np.zeros_like(forbidden)
        chamber[r0:r0+h+2, c0:c0+w+2] = True
        forbidden |= _dilate(chamber)
        try:
            inp = _scatter(rng, inp, tiles, forbidden)
        except ValueError:
            continue
        start = (r0+h, c0+1)
        inp[start] = word[0]
        target = base.copy()
        for cells, color in tiles:
            translated = cells+(r0, c0)
            target[translated[:, 0], translated[:, 1]] = color
        for r, c in record['collisions']:
            target[r+r0, c+c0] = marker
        pair = {'input': _orient(inp, orientation), 'output': _orient(target, orientation)}
        try:
            info = _decode(pair['input'])
            expected, placements = _answer(info)
            if not np.array_equal(expected, pair['output']):
                continue
            if not _addressed_twins(info, placements):
                continue
            if witness and not _REQUIRED <= _coverage(info):
                continue
        except ValueError:
            continue
        return pair
    raise RuntimeError("bounded chamber/tile sampler exhausted")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    base_orientation = int(rng.integers(8))
    schedule = int(rng.integers(3))
    witness_slot = int(rng.integers(nt))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            orientation = base_orientation
            if schedule == 0 and split == 'test':
                orientation = 4*(base_orientation//4)+(base_orientation+1)%4
            elif schedule == 2:
                orientation = int(rng.integers(8))
            expanded = split == 'test' and schedule != 0
            for _ in range(12):
                pair = _make_pair(rng, orientation, split == 'train' and i == witness_slot, expanded)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
            seen.add(_key(pair['input']))
            result[split].append(pair)
    # In growth/recombination schedules, demand a visible change in topology,
    # program length, or tile count rather than relying on a sampling label.
    if schedule != 0:
        train_features = {_shift_feature(_decode(p['input'])) for p in result['train']}
        for i in range(ne):
            for _ in range(30):
                if _shift_feature(_decode(result['test'][i]['input'])) not in train_features:
                    break
                orientation = base_orientation if schedule == 1 else int(rng.integers(8))
                pair = _make_pair(rng, orientation, expanded=True)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    result['test'][i] = pair
            else:
                raise RuntimeError("could not construct the observed train/test shift")
    validate_generated_puzzle(result)
    return result


def _shift_feature(info):
    return len(info['word']), len(info['pieces']), len(info['record']['legs'])


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected, placements = _answer(info)
            assert np.array_equal(out, expected), "wrong assembled colors, shape, or collision markers"
            assert info['record'] == _step_trace(info['mask'], info['start'])
            canonical = _unorient(out, info['orientation'])
            assert len(placements) == len(info['pieces'])
            source_occupied = np.zeros_like(info['grid'], dtype=bool)
            r0, c0, r1, c1 = info['frame']
            source_occupied[r0:r1+1, c0:c1+1] = True
            for p in info['annotation']:
                source_occupied[p] = True
            for source, _ in info['pieces']:
                assert not np.any(_dilate(source_occupied)[source[:, 0], source[:, 1]])
                source_occupied[source[:, 0], source[:, 1]] = True
            occupied = set()
            for i, target in placements:
                source, color = info['pieces'][i]
                delta = target.min(axis=0)-source.min(axis=0)
                assert set(map(tuple, source+delta)) == set(map(tuple, target))
                assert np.any(delta != 0)
                cells = set(map(tuple, target))
                assert not cells & occupied
                occupied |= cells
                assert np.all(canonical[target[:, 0], target[:, 1]] == color)
            assert occupied == set(map(tuple, np.argwhere(info['mask'])))
            r0, c0, r1, c1 = info['frame']
            assert not np.any(canonical[r0:r1+1, c0:c1+1] == info['background'])
            for j, p in enumerate(info['record']['path']):
                assert canonical[p] == info['word'][j % len(info['word'])]
            terrain_cells = set(map(tuple, np.argwhere(info['base'] == info['terrain'])))
            collisions = set(info['record']['collisions'])
            assert set(map(tuple, np.argwhere(canonical == info['marker']))) == collisions
            for p in terrain_cells-collisions:
                assert canonical[p] == info['terrain']
            assert len(_packing(info, use_cycle=False)) == 2, "program must resolve a real packing ambiguity"
            assert _addressed_twins(info, placements), "cycle must address whole tiles beyond the seam"
            if split == 'train':
                evidence |= _coverage(info)
    assert _REQUIRED <= evidence, f"missing demonstration evidence: {_REQUIRED-evidence}"


def _pattern(word):
    roles = {}
    return tuple(roles.setdefault(c, len(roles)) for c in word)


def _addressed_twins(info, placements):
    path = set(info['record']['path'])
    for (i, a), (j, b) in combinations(placements, 2):
        shape_a = set(map(tuple, a-a.min(axis=0)))
        shape_b = set(map(tuple, b-b.min(axis=0)))
        if (shape_a == shape_b and len(a) >= 2 and info['pieces'][i][1] != info['pieces'][j][1]
                and info['start'] not in set(map(tuple, a)) | set(map(tuple, b))
                and set(map(tuple, a)) & path and set(map(tuple, b)) & path
                and set(map(tuple, a))-path and set(map(tuple, b))-path):
            return True
    return False


def self_test() -> dict:
    """Bounded checks on 42 seeds, meaningful ablations, and explicit counts."""
    shapes, structures, geometries, words, counts, orientations = set(), set(), set(), set(), set(), set()
    puzzle_keys = set()
    features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = tuple((_key(p['input']), _key(p['output'])) for p in puzzle['train']+puzzle['test'])
        assert signature == tuple((_key(p['input']), _key(p['output'])) for p in again['train']+again['test'])
        puzzle_keys.add(signature)
        for p in puzzle['train']+puzzle['test']:
            info = _decode(p['input'])
            shapes.add(p['input'].shape)
            roles = {}
            structures.add((p['input'].shape, tuple(roles.setdefault(int(c), len(roles)) for c in p['input'].flat)))
            r0, c0, r1, c1 = info['frame']
            geometries.add(_key(info['mask'][r0:r1+1, c0:c1+1]))
            words.add(_pattern(info['word']))
            counts.add(len(info['pieces']))
            orientations.add(info['orientation'])
            features.append((len(info['record']['legs']), len(info['record']['path']), int(info['mask'].sum())))
            # Ignoring A gives only a thin painted seam, leaving the complementary
            # off-seam cells empty. It cannot equal a filled chamber.
            thin = info['base'].copy()
            for j, cell in enumerate(info['record']['path']):
                thin[cell] = info['word'][j % len(info['word'])]
            assert not np.array_equal(_orient(thin, info['orientation']), p['output'])
    assert len(puzzle_keys) == 42
    assert len(shapes) >= 35 and len(structures) >= 190 and len(geometries) >= 70
    assert orientations == set(range(8))
    assert len(words) >= 12 and len(counts) >= 4
    assert {f[0] for f in features} >= {1, 2, 3, 4}
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for counts_bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                       (4.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, *counts_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("identity labels accepted")
    # A boundary regression: available terrain around the corner is irrelevant
    # once a forward step leaves the canvas.
    fixture = np.zeros((4, 8), dtype=bool)
    fixture[2, :] = True
    fixture[1, 7] = True
    assert _trace(fixture, (2, 0)) == _step_trace(fixture, (2, 0))
    assert _trace(fixture, (2, 0))['path'] == tuple((2, c) for c in range(8))
    return dict(seeds=42, pairs=210, distinct_shapes=len(shapes),
                color_normalized_inputs=len(structures), chamber_geometries=len(geometries),
                word_patterns=len(words), tile_counts=sorted(counts),
                orientations=sorted(orientations), leg_counts=sorted({f[0] for f in features}),
                independent_solver=False)
