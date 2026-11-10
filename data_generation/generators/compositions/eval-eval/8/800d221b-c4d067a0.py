"""Consensus quilts: local ballots, port-weighted spatial votes, and registration.

Only standard Python and NumPy are used. Construction records never enter solve.
"""
from functools import lru_cache
from itertools import product

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_HOST_SEEDS = (0, 471139710, 885127668, 320903598, 700563190, 275195041,
               673970734, 562322330, 934735228, 104538459, 815959580,
               629608447, 897454262, 481648593, 42354666, 74355936,
               75064020, 421829046, 632634008, 627739972, 702014046,
               646700309, 583497551, 417065687, 816735581, 717757873,
               912374167, 489663574, 332721175, 341806591, 558044429,
               990402266, 910131850, 592073896, 4285921, 286005872,
               39459140, 430916312, 379150688, 708715430, 550284405, 470020811)


def _check_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _DIRS:
                ny, nx = y + dy, x + dx
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append(cells)
    return result


def _box(cells):
    p = np.asarray(cells)
    r, c = map(int, p.min(axis=0))
    b, e = map(int, p.max(axis=0) + 1)
    return r, c, b, e


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _frame_in_component(grid, cells, scaffold):
    """The frame is a hollow rectangle; branches cannot supply another ring."""
    r0, c0, r1, c1 = _box(cells)
    runs = {}
    for r in range(r0, r1):
        row = grid[r, c0:c1] == scaffold
        starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
        ends = np.flatnonzero(row & ~np.r_[row[1:], False]) + 1
        for start, end in zip(starts, ends):
            if end - start >= 5:
                runs.setdefault((int(start) + c0, int(end) + c0), []).append(r)
    frames = []
    for (c, e), rows in runs.items():
        for r, bottom in product(rows, repeat=2):
            b = bottom + 1
            if b - r < 5:
                continue
            if (np.all(grid[r:b, c] == scaffold)
                    and np.all(grid[r:b, e - 1] == scaffold)
                    and not np.any(grid[r + 1:b - 1, c + 1:e - 1] == scaffold)):
                frames.append((r, c, b, e))
    assert len(frames) == 1, "each conduit component must have exactly one hollow frame"
    return frames[0]


def _read_card(grid, cells, scaffold, bg, signals):
    r, c, b, e = _frame_in_component(grid, cells, scaffold)
    inside = grid[r + 1:b - 1, c + 1:e - 1]
    tiles = _components(inside != bg)
    assert len(tiles) >= 4
    boxes = [_box(tile) for tile in tiles]
    ys = sorted({box[0] for box in boxes})
    xs = sorted({box[1] for box in boxes})
    th, tw = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]
    assert min(len(ys), len(xs)) >= 2 and th * tw >= 3
    assert ys[0] == xs[0] == 0
    assert ys[-1] + th == inside.shape[0] and xs[-1] + tw == inside.shape[1]
    dy, dx = ys[1] - ys[0], xs[1] - xs[0]
    assert dy > th and dx > tw
    assert all(y == i * dy for i, y in enumerate(ys))
    assert all(x == j * dx for j, x in enumerate(xs))
    assert len(tiles) == len(ys) * len(xs)
    majorities = np.empty((len(ys), len(xs)), dtype=np.int8)
    firsts = majorities.copy()
    raw = np.zeros_like(majorities, dtype=int)
    covered = np.zeros_like(inside, dtype=bool)
    for tile, box in zip(tiles, boxes):
        y, x, bottom, right = box
        assert bottom - y == th and right - x == tw and len(tile) == th * tw
        block = inside[y:bottom, x:right]
        counts = [int(np.count_nonzero(block == s)) for s in signals]
        assert sum(counts) == th * tw and min(counts) > 0 and counts[0] != counts[1]
        i, j = ys.index(y), xs.index(x)
        majorities[i, j] = int(counts[1] > counts[0])
        firsts[i, j] = int(block[0, 0] == signals[1])
        raw[i, j] = counts[1] - counts[0]
        covered[y:bottom, x:right] = True
    assert np.all(inside[~covered] == bg)
    ports = [(r - 1, x) for x in range(c + 1, e - 1)]
    ports += [(b, x) for x in range(c + 1, e - 1)]
    ports += [(y, c - 1) for y in range(r + 1, b - 1)]
    ports += [(y, e) for y in range(r + 1, b - 1)]
    ports = [p for p in ports if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]
             and grid[p] == scaffold]
    assert 1 <= len(ports) <= 4
    # Corner attachments do not count as ports and are prohibited.
    for p in ((r - 1, c), (r - 1, e - 1), (b, c), (b, e - 1),
              (r, c - 1), (b - 1, c - 1), (r, e), (b - 1, e)):
        if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]:
            assert grid[p] != scaffold
    return dict(frame=(r, c, b, e), cells=cells, ports=ports, weight=len(ports),
                majorities=majorities, firsts=firsts, raw=raw, tile=(th, tw),
                mini_gutters=(dy - th, dx - tw))


def _interpret_cards(grid, bg, scaffold, signals):
    components = _components(grid == scaffold)
    assert 3 <= len(components) <= 5
    cards = [_read_card(grid, cells, scaffold, bg, signals) for cells in components]
    shape = cards[0]['majorities'].shape
    assert all(card['majorities'].shape == shape for card in cards)
    outside = (grid != bg) & (grid != scaffold)
    for card in cards:
        r, c, b, e = card['frame']
        outside[r:b, c:e] = False
    squares = []
    for cells in _components(outside):
        r, c, b, e = _box(cells)
        side = b - r
        assert side >= 2 and e - c == side and len(cells) == side * side
        color = int(grid[r, c])
        assert color in signals and np.all(grid[r:b, c:e] == color)
        squares.append((r, c, side, color))
    assert 2 <= len(squares) <= max(shape)
    if len({s[0] for s in squares}) == 1:
        axis = 1
    else:
        assert len({s[1] for s in squares}) == 1
        axis = 0
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    assert all(s[2] == side for s in squares)
    diffs = np.diff([s[axis] for s in squares])
    stride = int(diffs[0])
    assert stride > side and np.all(diffs == stride)
    weights = np.array([card['weight'] for card in cards])
    votes = np.stack([card['majorities'] for card in cards])
    score = np.tensordot(weights, 2 * votes.astype(int) - 1, axes=1)
    plan = np.where(score == 0, bg, np.where(score > 0, signals[1], signals[0])).astype(np.int8)
    sequence = tuple(s[3] for s in squares)
    length = len(squares)
    matches = []
    for y in range(shape[0] - (length - 1 if axis == 0 else 0)):
        for x in range(shape[1] - (length - 1 if axis == 1 else 0)):
            if tuple(int(plan[y + (axis == 0) * k, x + (axis == 1) * k])
                     for k in range(length)) == sequence:
                matches.append((y, x))
    assert len(matches) == 1, "the voted sequence must determine one origin"
    match = matches[0]
    base = (squares[0][0] - match[0] * stride, squares[0][1] - match[1] * stride)
    target = (base[0], base[1], base[0] + (shape[0] - 1) * stride + side,
              base[1] + (shape[1] - 1) * stride + side)
    assert 0 <= target[0] < target[2] <= grid.shape[0]
    assert 0 <= target[1] < target[3] <= grid.shape[1]
    # The whole rendered rectangle, including empty sites and gutters, is clear.
    for card in cards:
        assert not _overlap(target, _box(card['cells']))
    sites = {(match[0] + (axis == 0) * k, match[1] + (axis == 1) * k) for k in range(length)}
    region = grid[target[0]:target[2], target[1]:target[3]].copy()
    expected = grid.copy()
    added = []
    for (y, x), value in np.ndenumerate(plan):
        yy, xx = base[0] + y * stride, base[1] + x * stride
        block = grid[yy:yy + side, xx:xx + side]
        assert np.all(block == (value if (y, x) in sites else bg))
        if (y, x) in sites:
            region[y * stride:y * stride + side, x * stride:x * stride + side] = bg
        if value != bg:
            expected[yy:yy + side, xx:xx + side] = value
            if (y, x) not in sites:
                added.append((y, x))
    assert np.all(region == bg) and added
    return expected, dict(background=bg, scaffold=scaffold, signals=signals, cards=cards,
                          plan=plan, score=score, votes=votes, weights=weights, axis=axis,
                          match=match, base=base, side=side, stride=stride, sites=sites,
                          added=added, target=target, sequence=sequence)


def _recover(input_grid):
    grid = _check_grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 4 and np.count_nonzero(counts == counts.max()) == 1
    bg = int(colors[np.argmax(counts)])
    interpretations = []
    for scaffold in colors:
        scaffold = int(scaffold)
        if scaffold == bg:
            continue
        signals = tuple(int(s) for s in colors if s not in (bg, scaffold))
        try:
            interpretations.append(_interpret_cards(grid, bg, scaffold, signals))
        except AssertionError:
            continue
    assert len(interpretations) == 1, "input must have one geometric color-role interpretation"
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the consensus and registration using only the supplied pixels."""
    return _recover(input_grid)[0]


def _asymmetric(plan):
    transforms = [np.flip(plan, 0), np.flip(plan, 1), np.flip(plan, (0, 1))]
    if plan.shape[0] == plan.shape[1]:
        transforms += [plan.T, np.rot90(plan), np.rot90(plan, 3), np.flip(plan.T, (0, 1))]
    return all(not np.array_equal(plan, t) for t in transforms)


@lru_cache(maxsize=None)
def _states(weights):
    bits = np.array(list(product((0, 1), repeat=len(weights))), dtype=np.int8)
    signs = 2 * bits.astype(int) - 1
    score = signs @ np.array(weights)
    plain = signs.sum(axis=1)
    weighted_witness = np.flatnonzero((score * plain < 0))
    ties = np.flatnonzero(score == 0)
    return bits, score, weighted_witness, ties


@lru_cache(maxsize=None)
def _profiles(count, max_weight, kind):
    profiles = []
    for weights in product(range(1, max_weight + 1), repeat=count):
        _, _, wrong, ties = _states(weights)
        if kind == 'weighted' and len(wrong) == 0:
            continue
        if kind == 'tie' and len(ties) == 0:
            continue
        profiles.append(weights)
    return tuple(profiles)


@lru_cache(maxsize=None)
def _options(kind, side_filter=0):
    """Enumerate feasible layouts rather than maintaining a scene-template bank."""
    result = []
    for count in range(3, 6):
        for ph, pw in product(range(2, 5), repeat=2):
            if kind in ('tie', 'test') and pw < 3:
                continue
            for th, tw in ((1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)):
                for mg in (1, 2):
                    fh, fw = (ph - 1) * (th + mg) + th + 2, (pw - 1) * (tw + mg) + tw + 2
                    max_weight = min(4, fh - 2)
                    profile_kind = kind if kind in ('weighted', 'tie') else 'general'
                    if not _profiles(count, max_weight, profile_kind):
                        continue
                    for wire in (1, 2, 3):
                        for columns in range(1, count + 1):
                            rows = (count + columns - 1) // columns
                            # One background cell separates card envelopes.
                            board = (rows * fh + rows - 1, columns * (fw + wire) + columns - 1)
                            for side in (2, 3, 4):
                                if side_filter and side != side_filter:
                                    continue
                                for gutter in (1, 2, 3):
                                    target = ((ph - 1) * (side + gutter) + side,
                                              (pw - 1) * (side + gutter) + side)
                                    for separation in (0, 1):
                                        required = [max(board[d], target[d]) for d in (0, 1)]
                                        required[separation] = board[separation] + target[separation] + 1
                                        if max(required) <= 30:
                                            result.append((count, ph, pw, th, tw, mg, wire, columns,
                                                           side, gutter, separation, fh, fw, board, target))
    assert result
    return tuple(result)


def _sample_votes(rng, shape, weights, kind):
    bits, scores, wrong, ties = _states(tuple(weights))
    h, w = shape
    for _ in range(300):
        indices = rng.integers(len(bits), size=shape)
        forced = None
        if kind in ('weighted', 'tie'):
            forced = (int(rng.integers(h)), int(rng.integers(w)))
            indices[forced] = int(rng.choice(wrong if kind == 'weighted' else ties))
        local = None
        if kind in ('weighted', 'tie'):
            locations = [(r, c) for r in range(h) for c in range(w) if (r, c) != forced]
            local = locations[int(rng.integers(len(locations)))]
            # All cards agree here; looking at one tile pixel reverses the vote.
            value = int(scores[indices[forced]] < 0) if kind == 'weighted' else int(rng.integers(2))
            indices[local] = len(bits) - 1 if value else 0
        values = scores[indices]
        plan = np.where(values == 0, -1, (values > 0).astype(int)).astype(np.int8)
        if not _asymmetric(plan) or not {0, 1} <= set(map(int, plan.flat)):
            continue
        choices = []
        for length in range(2, w + 1):
            if kind in ('tie', 'test') and length == w:
                continue
            matches = {}
            for r in range(h):
                for c in range(w - length + 1):
                    seq = tuple(map(int, plan[r, c:c + length]))
                    if -1 not in seq:
                        matches.setdefault(seq, []).append((r, c))
            for positions in matches.values():
                if len(positions) != 1:
                    continue
                match = positions[0]
                sites = {(match[0], match[1] + k) for k in range(length)}
                if forced in sites or local in sites:
                    continue
                if kind == 'tie' and match == (0, 0):
                    continue
                rest = {int(plan[r, c]) for r in range(h) for c in range(w) if (r, c) not in sites}
                if not {0, 1} <= rest:
                    continue
                choices.append((match, length))
        if choices:
            # Tests favor completion above and below, when geometry allows it.
            preferred = [v for v in choices if 0 < v[0][0] < h - 1]
            if kind == 'test' and preferred and rng.random() < .8:
                choices = preferred
            match, length = choices[int(rng.integers(len(choices)))]
            return bits[indices].transpose(2, 0, 1), plan, match, length, forced, local
    return None


def _place(rng, board, target, separation):
    other = 1 - separation
    gap = int(rng.integers(1, min(3, 30 - board[separation] - target[separation]) + 1))
    needed = [max(board[d], target[d]) for d in (0, 1)]
    needed[separation] = board[separation] + target[separation] + gap
    shape = tuple(int(rng.integers(n, min(30, n + 4) + 1)) for n in needed)
    if rng.random() < .15:
        shape = (30, 30)
    elif rng.random() < .2:
        shape = (max(shape), max(shape))
    origin, base = [0, 0], [0, 0]
    margin = int(rng.integers(shape[separation] - needed[separation] + 1))
    if rng.integers(2):
        origin[separation] = margin
        base[separation] = margin + board[separation] + gap
    else:
        base[separation] = margin
        origin[separation] = margin + target[separation] + gap
    origin[other] = int(rng.integers(shape[other] - board[other] + 1))
    base[other] = int(rng.integers(shape[other] - target[other] + 1))
    return shape, tuple(origin), tuple(base)


def _make_pair(rng, roles, kind, physical_axis, side_filter=0):
    options = _options(kind, side_filter)
    for _ in range(150):
        option = options[int(rng.integers(len(options)))]
        count, ph, pw, th, tw, mg, wire, cols, side, gutter, sep, fh, fw, board, target = option
        profile_kind = kind if kind in ('weighted', 'tie') else 'general'
        profiles = _profiles(count, min(4, fh - 2), profile_kind)
        weights = profiles[int(rng.integers(len(profiles)))]
        sampled = _sample_votes(rng, (ph, pw), weights, kind)
        if sampled is None:
            continue
        votes, abstract, match, length, forced, local = sampled
        shape, origin, base = _place(rng, board, target, sep)
        bg, scaffold, s0, s1 = roles
        signals = (s0, s1)
        inp = np.full(shape, bg, dtype=np.int8)
        # Randomly assign logical cards to physical slots, leaving the last row
        # ragged when appropriate. Card ordering never affects the vote.
        slot_order = rng.permutation(count)
        for slot, owner in enumerate(slot_order):
            owner = int(owner)
            r = origin[0] + (slot // cols) * (fh + 1)
            c = origin[1] + (slot % cols) * (fw + wire + 1)
            inp[r:r + fh, c] = inp[r:r + fh, c + fw - 1] = scaffold
            inp[r, c:c + fw] = inp[r + fh - 1, c:c + fw] = scaffold
            port_rows = rng.choice(np.arange(1, fh - 1), size=weights[owner], replace=False)
            for row in port_rows:
                extent = int(rng.integers(1, wire + 1))
                inp[r + int(row), c + fw:c + fw + extent] = scaffold
            for y, x in product(range(ph), range(pw)):
                majority = signals[int(votes[owner, y, x])]
                minority = signals[1 - int(votes[owner, y, x])]
                area = th * tw
                nminor = ((area - 1) // 2 if (y, x) == forced and kind == 'weighted'
                          else int(rng.integers(1, (area - 1) // 2 + 1)))
                tile = np.full(area, majority, dtype=np.int8)
                if (y, x) == local:
                    positions = np.r_[0, rng.permutation(np.arange(1, area))[:nminor - 1]]
                else:
                    positions = rng.permutation(area)[:nminor]
                tile[positions] = minority
                yy, xx = r + 1 + y * (th + mg), c + 1 + x * (tw + mg)
                inp[yy:yy + th, xx:xx + tw] = tile.reshape(th, tw)
        plan = np.where(abstract == -1, bg, np.where(abstract == 0, s0, s1)).astype(np.int8)
        stride = side + gutter
        for k in range(length):
            y, x = match[0], match[1] + k
            yy, xx = base[0] + y * stride, base[1] + x * stride
            inp[yy:yy + side, xx:xx + side] = plan[y, x]
        # Ground truth is rendered from sampled ballots and weights, before
        # reconstruction sees any pixels. It does not call solve to get labels.
        out = inp.copy()
        for (y, x), value in np.ndenumerate(plan):
            if value != bg:
                yy, xx = base[0] + y * stride, base[1] + x * stride
                out[yy:yy + side, xx:xx + side] = value
        if physical_axis == 0:
            inp, out = inp.T.copy(), out.T.copy()
        for axis in (0, 1):
            if rng.integers(2):
                inp, out = np.flip(inp, axis).copy(), np.flip(out, axis).copy()
        try:
            reconstructed, info = _recover(inp)
            assert np.array_equal(reconstructed, out)
            assert _asymmetric(info['plan'])
            m = _measure(info)
            if kind == 'weighted':
                assert m['weight_witness'] and m['raw_witness'] and m['local_witness']
            if kind == 'tie':
                assert m['holes'] and m['local_witness'] and info['match'] != (0, 0)
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError('bounded consensus-quilt construction exhausted')


def _measure(info):
    cards, votes, weights = info['cards'], info['votes'], info['weights']
    plain = (2 * votes.astype(int) - 1).sum(axis=0)
    raw = np.stack([c['raw'] for c in cards]).sum(axis=0)
    first = np.tensordot(weights, 2 * np.stack([c['firsts'] for c in cards]).astype(int) - 1, axes=1)
    score = info['score']
    added = info['added']
    return dict(plan_shape=info['plan'].shape, cards=len(cards),
                weights=tuple(sorted(map(int, weights))), tiles=tuple(sorted(c['tile'] for c in cards)),
                holes=int(np.count_nonzero(score == 0)),
                weight_witness=any(score[p] * plain[p] < 0 for p in added),
                raw_witness=any(score[p] * raw[p] < 0 for p in added),
                local_witness=any(score[p] * first[p] < 0 for p in added),
                novel=all(not np.array_equal(info['plan'],
                          np.where(c['majorities'], info['signals'][1], info['signals'][0])) for c in cards),
                side=info['side'], gutter=info['stride'] - info['side'], axis=info['axis'],
                length=len(info['sequence']), match=info['match'],
                roles=(info['background'], info['scaffold'], *info['signals']))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer between {low} and {high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable = bool(rng.random() < .25)
    axis_mode = int(rng.integers(3))  # vertical, horizontal, or demonstrated mixed
    axes = [axis_mode] if axis_mode < 2 else [0, 1]
    train_axes = [axes[i % len(axes)] for i in range(int(num_train))]
    rng.shuffle(train_axes)
    first_side = int(rng.choice((2, 3)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(int(count)):
            kind = ('weighted' if i == 0 else 'tie' if i == 1 else 'general') if split == 'train' else 'test'
            side_filter = (first_side if i == 0 else 5 - first_side if i == 1 else 0) if split == 'train' else 0
            axis = train_axes[i] if split == 'train' else int(rng.choice(axes))
            for _ in range(32):
                roles = palette if stable else tuple(map(int, rng.choice(10, 4, replace=False)))
                pair = _make_pair(rng, roles, kind, axis, side_filter)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('could not construct a distinct input')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, measures, teaching = set(), {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(out, expected), 'incorrect quilt or changed source pixel'
            assert _asymmetric(info['plan'])
            added_colors = {int(info['plan'][p]) for p in info['added']}
            assert added_colors == set(info['signals'])
            # Input evidence is immutable, including frames, minority votes,
            # routes, and supplied squares. Only background can be added to.
            assert np.array_equal(out[inp != info['background']], inp[inp != info['background']])
            for (r, c), color in np.ndenumerate(info['plan']):
                if color == info['background']:
                    y, x = info['base'][0] + r * info['stride'], info['base'][1] + c * info['stride']
                    assert np.all(out[y:y + info['side'], x:x + info['side']] == color)
            measures[split].append(_measure(info))
            if split == 'train':
                teaching.add(info['axis'])
    train = measures['train']
    assert any(m['weight_witness'] and m['raw_witness'] for m in train)
    assert any(m['holes'] and m['novel'] and m['match'] != (0, 0) for m in train)
    assert any(m['local_witness'] for m in train)
    assert len({m['side'] for m in train}) >= 2
    assert {m['axis'] for m in measures['test']} <= teaching


def self_test():
    """Bounded structural, semantic-ablation, determinism, and diversity checks."""
    # A literal semantic fixture expands beyond the sampler: different tile
    # geometries per card and exits on all four sides. The expected consensus
    # and registration are stated directly, rather than obtained from solve.
    fixture = np.zeros((28, 29), dtype=np.int8)
    literal_cards = (
        (1, 1, 1, 3, [[0, 1, 0], [1, 0, 1]], [(0, 3), (3, 14)]),
        (8, 2, 2, 2, [[0, 0, 0], [1, 1, 1]], [(15, 5)]),
        (18, 1, 2, 3, [[0, 1, 0], [0, 1, 1]], [(20, 0)]),
    )
    for r, c, th, tw, decisions, ports in literal_cards:
        fh, fw = 2 * th + 3, 3 * tw + 4
        fixture[r:r + fh, c] = fixture[r:r + fh, c + fw - 1] = 5
        fixture[r, c:c + fw] = fixture[r + fh - 1, c:c + fw] = 5
        for p in ports:
            fixture[p] = 5
        for y, x in product(range(2), range(3)):
            majority = (2, 7)[decisions[y][x]]
            minority = (7, 2)[decisions[y][x]]
            yy, xx = r + 1 + y * (th + 1), c + 1 + x * (tw + 1)
            fixture[yy:yy + th, xx:xx + tw] = majority
            fixture[yy, xx] = minority
    fixture[2:4, 21:23] = 7
    fixture[2:4, 24:26] = 2
    expected_fixture = fixture.copy()
    literal_consensus = [[2, 7, 2], [7, 0, 7]]
    for y, x in product(range(2), range(3)):
        value = literal_consensus[y][x]
        if value:
            yy, xx = 2 + 3 * y, 18 + 3 * x
            expected_fixture[yy:yy + 2, xx:xx + 2] = value
    assert np.array_equal(solve(fixture), expected_fixture)
    for axis in (0, 1):
        assert np.array_equal(solve(np.flip(fixture, axis)), np.flip(expected_fixture, axis))
    assert np.array_equal(solve(fixture.T), expected_fixture.T)
    # Omitting registration shifts the rendering three columns to the right,
    # even though the correct voted blueprint has been supplied explicitly.
    wrong_origin = fixture.copy()
    for y, x in product(range(2), range(3)):
        value = literal_consensus[y][x]
        if value:
            yy, xx = 2 + 3 * y, 21 + 3 * x
            wrong_origin[yy:yy + 2, xx:xx + 2] = value
    assert not np.array_equal(wrong_origin, expected_fixture)
    # A local 2:2 tie is invalid; the valid global tie above remains a hole.
    tied_ballot = fixture.copy()
    tied_ballot[9, 4] = 7
    try:
        solve(tied_ballot)
    except AssertionError:
        pass
    else:
        raise AssertionError('local ballot tie accepted')
    signatures, shapes, structures, records = set(), set(), set(), []
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(p['input']), _key(p['output']))
                          for s in ('train', 'test') for p in puzzle[s])
        signatures.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, info = _recover(pair['input'])
                assert np.array_equal(predicted, pair['output'])
                records.append((split, _measure(info)))
                shapes.add(pair['input'].shape)
                roles, next_role = {}, 0
                normalized = []
                for value in pair['input'].flat:
                    value = int(value)
                    if value not in roles:
                        roles[value] = next_role
                        next_role += 1
                    normalized.append(roles[value])
                structures.add((pair['input'].shape, tuple(normalized)))
    assert len(signatures) == 42 and len(shapes) >= 20 and len(structures) >= 150
    all_m = [m for _, m in records]
    assert {m['cards'] for m in all_m} == {3, 4, 5}
    assert {m['axis'] for m in all_m} == {0, 1}
    assert {m['side'] for m in all_m} == {2, 3, 4}
    assert len({m['plan_shape'] for m in all_m}) >= 6
    assert len({m['weights'] for m in all_m}) >= 12
    assert len({m['tiles'] for m in all_m}) >= 4
    assert len({m['match'] for m in all_m}) >= 6
    assert len({m['roles'] for m in all_m}) >= 30
    assert any(m['length'] >= 3 for m in all_m)
    assert any(m['holes'] >= 2 for m in all_m)
    assert sum(m['novel'] for m in all_m) >= 42
    for seed, tr, te in ((91, 2, 1), (92, 10, 4), (93, 6, 3)):
        p = generate_puzzle(seed, tr, te)
        assert len(p['train']) == tr and len(p['test']) == te
    p, q = generate_puzzle(913), generate_puzzle(913)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # A direct algebra fixture exercises both tie and minority-card override.
    bits, scores, wrong, ties = _states((3, 1, 1))
    assert len(wrong) == 2 and len(ties) == 0
    assert np.array_equal(scores, (2 * bits.astype(int) - 1) @ np.array([3, 1, 1]))
    assert len(_states((2, 1, 1))[3]) == 2
    # Negative output mutations: a filled tie site and an altered source vote.
    hole_pair = next(pair for pair in p['train'] if _measure(_recover(pair['input'])[1])['holes'])
    info = _recover(hole_pair['input'])[1]
    y, x = map(int, np.argwhere(info['score'] == 0)[0])
    row, col = info['base'][0] + y * info['stride'], info['base'][1] + x * info['stride']
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    slot = next(i for i, pair in enumerate(p['train']) if pair is hole_pair)
    damaged['train'][slot]['output'][row, col] = info['signals'][0]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('filled tie accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    r, c, _, _ = info['cards'][0]['frame']
    damaged['train'][slot]['output'][r + 1, c + 1] = info['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed ballot accepted')


if __name__ == '__main__':
    self_test()
    print('Consensus-quilt self-test passed (42 seeds).')
