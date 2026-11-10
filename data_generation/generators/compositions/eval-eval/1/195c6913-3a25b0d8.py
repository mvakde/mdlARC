"""End-anchored chamber embroidery. Standard library and NumPy only."""
from collections import Counter
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return int(num_train), int(num_test)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask)))
    parts = []
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, part = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
                    part.append(q)
        parts.append(np.asarray(part, dtype=int))
    return parts


def _box(points):
    return tuple(map(int, points.min(0))) + tuple(map(int, points.max(0) + 1))


def _rooms(grid, wall):
    padded = np.pad(grid != wall, 1, constant_values=True)
    return [p - 1 for p in _components(padded)
            if np.all(p > 0) and np.all(p < np.array(padded.shape) - 1)]


def _relations(rooms):
    boxes = [_box(p) for p in rooms]
    return [[tuple(-1 if a[d + 2] <= b[d] else 1 if b[d + 2] <= a[d] else 0
                   for d in (0, 1)) for b in boxes] for a in boxes]


def _match(donor_rooms, target_rooms, colors):
    """Match the complete visible spatial graph, requiring color agreement."""
    a, b = _relations(donor_rooms), _relations(target_rooms)
    n = len(a)
    assert n == len(b)
    signatures_a = [Counter(row) for row in a]
    signatures_b = [Counter(row) for row in b]
    options = [[j for j in range(n) if signatures_a[i] == signatures_b[j]]
               for i in range(n)]
    answers, mappings = [], []

    def search(mapping, used):
        if len(mapping) == n:
            anchors = [None] * n
            for i, j in mapping.items():
                anchors[j] = colors[i]
            if answers:
                assert anchors == answers[0], "ambiguous transferred anchors"
            else:
                answers.append(anchors)
            mappings.append(dict(mapping))
            return
        candidates = {i: [j for j in options[i] if j not in used and all(
            a[i][k] == b[j][v] and a[k][i] == b[v][j]
            for k, v in mapping.items())] for i in range(n) if i not in mapping}
        i = min(candidates, key=lambda x: len(candidates[x]))
        for j in candidates[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    assert answers, "no relation-preserving chamber correspondence"
    return answers[0], mappings


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _trace(mask, start):
    """Maximal-leg production implementation. Bend endpoints count once."""
    path, legs = [], []
    current, direction = start, 0
    for _ in range(sum(mask.shape) + 1):
        dr, dc = ((0, 1), (-1, 0))[direction]
        r, c = current
        leg = []
        while _inside(mask.shape, (r, c)) and mask[r, c]:
            leg.append((r, c))
            r, c = r + dr, c + dc
        assert leg
        path.extend(leg)
        legs.append(len(leg))
        if not _inside(mask.shape, (r, c)):
            break
        direction = 1 - direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        current = leg[-1][0] + dr, leg[-1][1] + dc
        if not _inside(mask.shape, current) or not mask[current]:
            break
    else:
        raise AssertionError("nonterminating staircase")
    return tuple(path), tuple(legs)


def _walk(mask, start):
    """Separate cell-step implementation for semantic cross-checks."""
    r, c = start
    north = False
    points, lengths = [], [0]
    while True:
        assert mask[r, c] and (r, c) not in points
        points.append((r, c))
        lengths[-1] += 1
        q = (r - 1, c) if north else (r, c + 1)
        if not _inside(mask.shape, q):
            break
        if mask[q]:
            r, c = q
            continue
        q = (r, c + 1) if north else (r - 1, c)
        if not _inside(mask.shape, q) or not mask[q]:
            break
        north = not north
        lengths.append(0)
        r, c = q
    return tuple(points), tuple(lengths)


def _decode(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= x <= 30 for x in grid.shape)
    assert np.all((0 <= grid) & (grid <= 9))
    bg = int(grid[0, 0])
    figures, swatches = [], []
    for p in _components(grid != bg, diagonal=True):
        r, c, R, C = _box(p)
        values = np.unique(grid[tuple(p.T)])
        if (R - r, C - c) == (2, 2) and len(p) == 4 and len(values) == 1:
            swatches.append((r, c, int(values[0])))
        else:
            crop = np.full((R - r, C - c), bg, dtype=np.int8)
            crop[p[:, 0] - r, p[:, 1] - c] = grid[tuple(p.T)]
            figures.append((crop, (r, c, R, C)))
    assert len(figures) == 2 and 3 <= len(swatches) <= 6
    swatches.sort(key=lambda p: p[1])
    assert len({r for r, c, color in swatches}) == 1
    assert all(v[1] - u[1] >= 3 for u, v in zip(swatches, swatches[1:]))
    assert swatches[0][0] + 2 < min(b[0] for g, b in figures)
    word = tuple(color for r, c, color in swatches)
    word_colors = set(word)
    donors = [(g, b) for g, b in figures if np.any(np.isin(g, list(word_colors)))]
    targets = [(g, b) for g, b in figures if not np.any(np.isin(g, list(word_colors)))]
    assert len(donors) == len(targets) == 1
    donor, db = donors[0]
    target, tb = targets[0]
    common = (set(map(int, donor.flat)) & set(map(int, target.flat))) - {bg}
    assert len(common) == 1
    wall = common.pop()
    marker_set = set(map(int, target.flat)) - {bg, wall}
    assert len(marker_set) == 1
    marker = marker_set.pop()
    assert not {bg, wall, marker} & word_colors
    dr, tr = _rooms(donor, wall), _rooms(target, wall)
    assert 3 <= len(dr) == len(tr) <= 7
    colors = []
    for room in dr:
        values = np.unique(donor[tuple(room.T)])
        assert len(room) >= 4 and len(values) == 1 and int(values[0]) in word_colors
        colors.append(int(values[0]))
    assert len(set(colors)) >= 2
    assert all(word.count(color) == 1 for color in colors)
    assert sum(len(p) for p in dr) == np.count_nonzero((donor != bg) & (donor != wall))
    starts, masks, records = [], [], []
    for room in tr:
        values = target[tuple(room.T)]
        assert len(room) >= 4 and np.count_nonzero(values == marker) == 1
        assert set(map(int, values)) == {bg, marker}
        start = tuple(map(int, room[np.flatnonzero(values == marker)[0]]))
        mask = np.zeros(target.shape, dtype=bool)
        mask[tuple(room.T)] = True
        r, c = start
        assert c > 0 and not mask[r, c - 1] and mask[r, c + 1]
        record = _trace(mask, start)
        assert record == _walk(mask, start)
        assert len(record[0]) >= 3
        starts.append(start)
        masks.append(mask)
        records.append(record)
    anchors, mappings = _match(dr, tr, colors)
    return dict(bg=bg, wall=wall, marker=marker, donor=donor, target=target,
                donor_box=db, target_box=tb, dr=dr, tr=tr, word=word,
                colors=colors, anchors=anchors, mappings=mappings, starts=starts,
                masks=masks, records=records)


def _paint(info, word=None, anchors=None, mode="correct"):
    word = info['word'] if word is None else tuple(word)
    anchors = info['anchors'] if anchors is None else anchors
    out = info['target'].copy()
    for room, anchor, (path, legs) in zip(info['tr'], anchors, info['records']):
        if mode == "uniform":
            out[tuple(room.T)] = anchor
            continue
        j, n = word.index(anchor), len(path)
        phase = j if mode == "start" else (j - n + 1) % len(word)
        if mode == "bend":
            # Each leg separately satisfies the anchor at its own final cell.
            values = [word[(j - length + 1 + i) % len(word)]
                      for length in legs for i in range(length)]
        elif mode == "reverse":
            values = [word[(j + n - 1 - i) % len(word)] for i in range(n)]
        else:
            values = [word[(phase + i) % len(word)] for i in range(n)]
        for p, color in zip(path, values):
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every convention, phase anchor, and route from this input alone."""
    return _paint(_decode(input_grid))


def _coverage(info):
    correct = _paint(info)
    variants = {
        'omit_A': _paint(info, mode='uniform'),
        'omit_B': _paint(info, anchors=[info['anchors'][0]] * len(info['tr'])),
        'scan_order': _paint(info, anchors=info['colors']),
        'start_anchor': _paint(info, mode='start'),
        'deduplicate': _paint(info, word=tuple(dict.fromkeys(info['word']))),
        'bend_reset': _paint(info, mode='bend'),
        'reverse': _paint(info, mode='reverse'),
    }
    evidence = {name for name, answer in variants.items() if not np.array_equal(answer, correct)}
    if any(len(path) >= 2 * len(info['word']) for path, legs in info['records']):
        evidence.add('cycling')
    # Three legs require a genuinely stepped route, beyond a rectangular L.
    if any(len(legs) >= 3 for path, legs in info['records']):
        evidence.add('geometry')
    return evidence


_REQUIRED = {'omit_A', 'omit_B', 'scan_order', 'start_anchor', 'deduplicate',
             'bend_reset', 'reverse', 'cycling', 'geometry'}


def _plan(rng, count, witness=False):
    if witness:
        if count == 3:
            return np.array([[1, 0, 0], [0, 0, 2]])
        plan = np.array([[1, 0, 0], [0, 0, 2], [3, 4, 5]])
        plan[plan >= count] = count - 1
        return plan
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3)]
    shapes = [s for s in shapes if s[0] * s[1] >= count]
    shape = shapes[int(rng.integers(len(shapes)))]
    plan = np.arange(np.prod(shape)).reshape(shape)
    # Corner omissions give irregular exterior silhouettes when space permits.
    for r, c in rng.permutation(np.array([[0, 0], [0, shape[1]-1],
                                         [shape[0]-1, 0], [shape[0]-1, shape[1]-1]])):
        if len(set(plan.flat) - {-1}) > count and rng.random() < .28:
            trial = plan.copy()
            trial[r, c] = -1
            if len(_components(trial >= 0)) == 1:
                plan = trial
    while len(set(plan.flat) - {-1}) > count:
        edges = [(int(plan[r, c]), int(plan[r + dr, c + dc]))
                 for r, c in np.argwhere(plan >= 0) for dr, dc in ((1, 0), (0, 1))
                 if r + dr < shape[0] and c + dc < shape[1]
                 and plan[r + dr, c + dc] >= 0 and plan[r, c] != plan[r + dr, c + dc]]
        a, b = edges[int(rng.integers(len(edges)))]
        plan[plan == b] = a
    old = plan.copy()
    for new, label in enumerate(sorted(set(plan.flat) - {-1})):
        plan[old == label] = new
    return plan


def _geometry(rng, plan, target=False, witness=False, simple=False):
    heights = rng.integers(2, 5 if target else 4, size=plan.shape[0])
    widths = rng.integers(2, 5 if target else 4, size=plan.shape[1])
    if simple:
        heights[:] = 3 if target else 2
        widths[:] = 3 if target else 2
    ys, xs = np.r_[0, np.cumsum(heights + 1)], np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in np.argwhere(plan >= 0):
        label = plan[r, c]
        labels[ys[r]+1:ys[r+1], xs[c]+1:xs[c+1]] = label
        if r and plan[r-1, c] == label:
            labels[ys[r], xs[c]+1:xs[c+1]] = label
        if c and plan[r, c-1] == label:
            labels[ys[r]+1:ys[r+1], xs[c]] = label
        if r and c and np.all(plan[r-1:r+1, c-1:c+1] == label):
            labels[ys[r], xs[c]] = label
    occupied = labels >= 0
    pad = np.pad(occupied, 1)
    near = np.zeros(labels.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= pad[dr:dr+labels.shape[0], dc:dc+labels.shape[1]]
    labels[(labels < 0) & near] = -2
    # Inset one small upper-left room only in the target: spatial relations
    # stay fixed, while flood-fill discovery order reverses with room zero.
    if witness and target:
        points = np.argwhere(labels == 1)
        r = int(points[:, 0].min())
        labels[r, labels[r] == 1] = -2
    if not simple:
        for label in sorted(set(plan.flat) - {-1}):
            if witness and label in (0, 1):
                continue
            # Insets/notches are wall additions, never holes or disconnections.
            for _ in range(int(rng.integers(0, 4))):
                points = np.argwhere(labels == label)
                r, c, R, C = _box(points)
                candidate = labels.copy()
                if rng.random() < .45:
                    side = int(rng.integers(4))
                    if side == 0:
                        candidate[r, candidate[r] == label] = -2
                    elif side == 1:
                        candidate[R-1, candidate[R-1] == label] = -2
                    elif side == 2:
                        candidate[candidate[:, c] == label, c] = -2
                    else:
                        candidate[candidate[:, C-1] == label, C-1] = -2
                else:
                    rr, cc = [(r, c), (r, C-1), (R-1, c), (R-1, C-1)][int(rng.integers(4))]
                    if candidate[rr, cc] == label:
                        candidate[rr, cc] = -2
                mask = candidate == label
                if mask.sum() >= 4 and len(_components(mask)) == 1:
                    labels = candidate
    r, c, R, C = _box(np.argwhere(labels != -1))
    return labels[r:R, c:C]


def _starts(rng, labels, witness=False):
    starts, records = {}, {}
    for label in sorted(set(labels.flat) - {-2, -1}):
        mask = labels == label
        options = []
        for r, c in np.argwhere(mask):
            if c > 0 and not mask[r, c-1] and c+1 < mask.shape[1] and mask[r, c+1]:
                record = _walk(mask, (int(r), int(c)))
                if len(record[0]) >= 3:
                    options.append(((int(r), int(c)), record))
        assert options
        if witness and label == 0:
            options.sort(key=lambda p: (len(p[1][1]), len(p[1][0])), reverse=True)
            start, record = options[0]
        else:
            start, record = options[int(rng.integers(len(options)))]
        starts[label], records[label] = start, record
    return starts, records


def _render(labels, bg, wall, colors=None, starts=None, marker=None):
    grid = np.full(labels.shape, bg, dtype=np.int8)
    grid[labels == -2] = wall
    if colors is not None:
        for label, color in enumerate(colors):
            grid[labels == label] = color
    if starts is not None:
        for p in starts.values():
            grid[p] = marker
    return grid


def _embed(rng, donor, target, word, bg):
    a, b = np.argwhere(donor != bg), np.argwhere(target != bg)
    delta = (a[:, None, :] - b[None, :, :]).reshape(-1, 2)
    forbidden = np.zeros((65, 65), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            q = delta + (32 + dr, 32 + dc)
            forbidden[q[:, 0], q[:, 1]] = True
    dh, dw = donor.shape
    th, tw = target.shape
    header = int(rng.integers(4, 6))
    groups = {k: [] for k in ('side', 'stack', 'diagonal', 'interleaved')}
    for r in range(-th-2, dh+3):
        for c in range(-tw-2, dw+3):
            h, w = max(dh, r+th)-min(0, r), max(dw, c+tw)-min(0, c)
            if h + header > 30 or w > 30 or forbidden[r+32, c+32]:
                continue
            ro, co = min(dh, r+th) > max(0, r), min(dw, c+tw) > max(0, c)
            group = 'interleaved' if ro and co else 'side' if ro else 'stack' if co else 'diagonal'
            groups[group].append((r, c, h, w))
    available = [g for g in groups if groups[g]]
    assert available
    group = available[int(rng.integers(len(available)))]
    r, c, h, w = groups[group][int(rng.integers(len(groups[group])))]
    gaps = rng.integers(1, 3, len(word)-1)
    legend_width = 2*len(word) + int(gaps.sum())
    w = max(w, legend_width+2)
    extra_h = int(rng.integers(0, min(3, 30-header-h)+1))
    extra_w = int(rng.integers(0, min(3, 30-w)+1))
    H, W = header+h+extra_h, w+extra_w
    inp = np.full((H, W), bg, dtype=np.int8)
    top = header + int(rng.integers(extra_h+1))
    left = int(rng.integers(W-(max(dw, c+tw)-min(0, c))+1))
    for grid, rr, cc in ((donor, top-min(0, r), left-min(0, c)),
                         (target, top+max(0, r), left+max(0, c))):
        region = inp[rr:rr+grid.shape[0], cc:cc+grid.shape[1]]
        mask = grid != bg
        region[mask] = grid[mask]
    lr = int(rng.integers(0, header-2))
    lc = int(rng.integers(1, W-legend_width+1))
    for i, color in enumerate(word):
        inp[lr:lr+2, lc:lc+2] = color
        if i < len(gaps):
            lc += 2 + int(gaps[i])
    # A swatch never reaches column zero, even on the top row.
    assert inp[0, 0] == bg
    return inp


def _word(rng, palette, length, repeat):
    if repeat:
        anchor_count = int(rng.integers(2, min(4, length-2)+1))
        anchors = list(map(int, palette[3:3+anchor_count]))
        spare = int(palette[3+anchor_count])
        word = anchors + [spare] * (length-anchor_count)
    else:
        word = list(map(int, palette[3:3+length]))
        anchor_count = int(rng.integers(2, min(4, length)+1))
        anchors = list(map(int, rng.choice(word, anchor_count, replace=False)))
    rng.shuffle(word)
    return tuple(word), anchors


def _pair(rng, count, length, witness=False):
    for attempt in range(96):
        simple = attempt >= 80
        palette = rng.permutation(10)
        bg, wall, marker = map(int, palette[:3])
        repeat = witness or (length >= 4 and rng.random() < .5)
        word, anchor_pool = _word(rng, palette, length, repeat)
        # Witness geometry has a long staircase and a discovery-order change.
        use_witness = witness or (simple and count <= 6)
        plan = _plan(rng, count, use_witness)
        donor_labels = _geometry(rng, plan, witness=use_witness, simple=simple)
        target_labels = _geometry(rng, plan, target=True, witness=use_witness, simple=simple)
        colors = [anchor_pool[i % len(anchor_pool)] for i in range(count)]
        rng.shuffle(colors)
        if use_witness:
            # Rooms zero and one are the pair whose scan order swaps.
            colors[0], colors[1] = anchor_pool[:2]
        try:
            starts, records = _starts(rng, target_labels, use_witness)
            donor = _render(donor_labels, bg, wall, colors)
            target = _render(target_labels, bg, wall, starts=starts, marker=marker)
            expected = target.copy()
            for label, (path, legs) in records.items():
                j, n = word.index(colors[label]), len(path)
                for i, p in enumerate(path):
                    expected[p] = word[(j-n+1+i) % len(word)]
            inp = _embed(rng, donor, target, word, bg)
            info = _decode(inp)
            assert np.array_equal(_paint(info), expected)
            if witness:
                assert _REQUIRED <= _coverage(info)
            return {'input': inp, 'output': expected}
        except (AssertionError, IndexError):
            continue
    raise AssertionError("bounded chamber construction exhausted")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    curriculum = int(rng.integers(2))  # 0: longer programs; 1: more chambers
    witness_index = int(rng.integers(nt))
    max_train_rooms = 6 if curriculum == 0 else 5
    counts = list(map(int, rng.integers(3, max_train_rooms+1, nt)))
    if counts[0] == counts[1]:
        counts[1] = 3 + (counts[0]-2) % (max_train_rooms-2)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', nt), ('test', ne)):
        for index in range(number):
            witness = split == 'train' and index == witness_index
            if split == 'train':
                count = counts[index]
                length = 4 if curriculum == 0 and witness else int(rng.integers(3, 5 if curriculum == 0 else 7))
                if witness:
                    length = max(4, length)
            else:
                count = int(rng.integers(3, 7)) if curriculum == 0 else int(rng.integers(6, 8))
                length = int(rng.integers(5, 7)) if curriculum == 0 else int(rng.integers(3, 7))
            for _ in range(12):
                pair = _pair(rng, count, length, witness)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError("could not construct distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, training_counts = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(_paint(info), out)
            assert out.shape == info['target'].shape
            # Exact frame preservation, including thick walls and crop exterior.
            assert np.array_equal(out == info['wall'], info['target'] == info['wall'])
            footprint = np.zeros(out.shape, dtype=bool)
            for anchor, mask, start, (path, legs) in zip(info['anchors'], info['masks'],
                                                       info['starts'], info['records']):
                assert (path, legs) == _walk(mask, start) == _trace(mask, start)
                assert len(set(path)) == len(path) and sum(legs) == len(path)
                assert out[path[-1]] == anchor
                j = info['word'].index(anchor)
                expected = [info['word'][(j-len(path)+1+i) % len(info['word'])]
                            for i in range(len(path))]
                assert [int(out[p]) for p in path] == expected
                assert all(not footprint[p] for p in path)
                for p in path:
                    footprint[p] = True
            assert np.array_equal(out != info['target'], footprint)
            assert np.array_equal(out[~footprint], info['target'][~footprint])
            assert info['marker'] not in out
            if split == 'train':
                evidence |= _coverage(info)
                training_counts.add(len(info['tr']))
    assert len(training_counts) >= 2
    assert _REQUIRED <= evidence, _REQUIRED - evidence


def self_test():
    # Local tracer regressions include a real three-leg stair and canvas exit.
    mask = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0, 0]], dtype=bool)
    expected = ((2, 0), (2, 1), (2, 2), (2, 3), (1, 3), (1, 4), (1, 5))
    assert _trace(mask, (2, 0)) == _walk(mask, (2, 0)) == (expected, (4, 1, 2))
    mask = np.ones((3, 6), dtype=bool)
    assert _trace(mask, (2, 0))[0] == tuple((2, c) for c in range(6))
    shapes, structures, words, roles, rooms, legs_seen = set(), set(), set(), set(), set(), set()
    scan_changes = reused = growth = shrink = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s]
                             for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add(pair['input'].shape)
                structures.add((info['target'].shape, (info['target'] == info['wall']).tobytes()))
                equality = {}
                words.add(tuple(equality.setdefault(c, len(equality)) for c in info['word']))
                roles.add((info['bg'], info['wall'], info['marker']))
                rooms.add(len(info['tr']))
                legs_seen |= {len(legs) for path, legs in info['records']}
                scan_changes += 'scan_order' in _coverage(info)
                reused += len(set(info['colors'])) < len(info['colors'])
                for i, j in info['mappings'][0].items():
                    growth += len(info['tr'][j]) > len(info['dr'][i])
                    shrink += len(info['tr'][j]) < len(info['dr'][i])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(structures) >= 150 and len(roles) >= 100
    assert len(words) >= 12 and rooms == {3, 4, 5, 6, 7}
    assert {1, 2, 3} <= legs_seen
    assert scan_changes >= 42 and reused >= 50 and growth >= 100 and shrink >= 30
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        a, b = generate_puzzle(813, nt, ne), generate_puzzle(813, nt, ne)
        assert all(_key(p[k]) == _key(q[k]) for s in ('train', 'test')
                   for p, q in zip(a[s], b[s]) for k in ('input', 'output'))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A wrong anchor, a wall edit, and a wrong timing program must be rejected.
    for corruption in ('wall', 'start', 'uniform'):
        puzzle = generate_puzzle(99, 2, 1)
        pair = puzzle['test'][0]
        info = _decode(pair['input'])
        if corruption == 'wall':
            pair['output'] = pair['output'].copy()
            p = tuple(np.argwhere(info['target'] == info['wall'])[0])
            pair['output'][p] = info['bg']
        else:
            pair['output'] = _paint(info, mode=corruption)
        assert not np.array_equal(pair['output'], _paint(info))
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted output accepted")
