"""Deduplicate framed socket liners, then exact-cover them with unchanged pieces.

Only NumPy and the standard library are used. No file I/O, generation metadata,
or seed-dependent convention enters solve(). Coordinates in the implementation
are normalized to a horizontal band; a vertical scene is transposed once.
"""

from itertools import product
import hashlib

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "ARC colors required")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        color = int(a[first])
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and a[p] == color:
                    remaining.remove(p)
                    stack.append(p)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _scene(value):
    original = _grid(value)
    bg = int(original[0, 0])
    _require(all(int(original[r, c]) == bg for r in (0, -1) for c in (0, -1)), "background corners")
    axes = []
    for transpose in (False, True):
        a = original.T if transpose else original
        rows = [r for r in range(a.shape[0]) if a[r, 0] != bg and np.all(a[r] == a[r, 0])]
        if rows:
            axes.append((transpose, a, rows))
    _require(len(axes) == 1, "one solid spanning band required")
    transpose, a, band_rows = axes[0]
    band = int(a[band_rows[0], 0])
    b0, b1 = min(band_rows), max(band_rows)+1
    _require(band_rows == list(range(b0, b1)), "contiguous band core")
    _require(np.count_nonzero(a == band) == (b1-b0)*a.shape[1], "band color reserved")

    def row_at(side, u):
        return b0-1-u if side == -1 else b1+u

    frame_options = []
    for color in set(map(int, a.flat)) - {bg, band}:
        posts = {}
        valid = True
        for side in (-1, 1):
            near = row_at(side, 0)
            if not 0 <= near < a.shape[0]:
                continue
            columns = np.flatnonzero(a[near] == color)
            if len(columns) % 2:
                valid = False
            data = []
            for c in columns:
                depth = 0
                while 0 <= row_at(side, depth) < a.shape[0] and a[row_at(side, depth), c] == color:
                    depth += 1
                if depth < 4:
                    valid = False
                data.append((int(c), depth))
            if data:
                posts[side] = data
        if valid and posts:
            frame_options.append((color, posts))
    _require(len(frame_options) == 1, "one frame-marker color required")
    frame, posts = frame_options[0]
    reserved = np.zeros_like(a, dtype=bool)
    frame_mask = np.zeros_like(a, dtype=bool)
    corridors = []
    for side, data in sorted(posts.items()):
        for (left, depth), (right, other_depth) in zip(data[::2], data[1::2]):
            _require(depth == other_depth and right-left >= 4, "paired frame posts")
            rows = np.array([row_at(side, u) for u in range(depth)])
            frame_mask[rows, left] = True
            frame_mask[rows, right] = True
            _require(not np.any(reserved[np.ix_(rows, np.arange(left, right+1))]), "disjoint frames")
            reserved[np.ix_(rows, np.arange(left, right+1))] = True
            raw = a[np.ix_(rows, np.arange(left+1, right))].copy()
            _require(not np.any(np.isin(raw, [band, frame])), "liner colors distinct from fixed structure")
            heights = []
            for column in raw.T:
                occupied = np.flatnonzero(column != bg)
                _require(1 <= len(occupied) <= 3 and np.array_equal(occupied, np.arange(len(occupied))), "bottom-connected liner columns of height 1..3")
                heights.append(len(occupied))
            keys = [tuple(map(int, column)) for column in raw.T]
            retained = list(dict.fromkeys(keys))
            _require(3 <= len(retained) <= 7 and len(retained) < len(keys), "redundant corridor with 3..7 unique columns")
            compact = np.array(retained, dtype=np.int8).T
            corridors.append(dict(side=side, left=left, right=right, depth=depth, rows=rows,
                                  raw=raw, compact=compact, keys=keys, heights=heights))
    _require(np.array_equal(frame_mask, a == frame), "frame color reserved")
    payload_mask = (a != bg) & (a != band) & ~reserved
    pieces = _components(a, payload_mask)
    _require(1 <= len(pieces) <= 8, "one to eight detached pieces")
    liner_colors = set(int(v) for cor in corridors for v in cor['raw'].flat) - {bg}
    payload_colors = {color for _, color in pieces}
    _require(not payload_colors & (liner_colors | {bg, band, frame}), "disjoint visible color roles")
    for cells, _ in pieces:
        _require(np.all((cells[:, 0] < b0-1) | (cells[:, 0] > b1)), "sources detached from band")
    return dict(a=a, transpose=transpose, bg=bg, band=band, band_rows=(b0, b1), frame=frame,
                frame_mask=frame_mask, corridors=corridors, pieces=pieces, reserved=reserved)


def _prepare(info, compact=True):
    """Erase sources; B acts on the liners, never on the pieces or fixed band."""
    base = info['a'].copy()
    for cells, _ in info['pieces']:
        base[cells[:, 0], cells[:, 1]] = info['bg']
    targets = []
    for cor in info['corridors']:
        rows, left = cor['rows'], cor['left']+1
        data = cor['compact'] if compact else cor['raw']
        if compact:
            base[np.ix_(rows, np.arange(left, cor['right']))] = info['bg']
            base[np.ix_(rows, np.arange(left, left+data.shape[1]))] = data
        holes = [(int(rows[u]), left+c) for u, c in np.argwhere(data == info['bg'])]
        targets.extend(holes)
    return base, sorted(targets)


def _exact_covers(base, targets, pieces, limit=2):
    """All translation candidates, then cell-and-object exact cover.

    Uniqueness is by colored output, so permutations of identical same-colored
    components do not create false ambiguity. No full-width-chain assumption is
    imposed by the solver.
    """
    if not targets:
        return []
    index = {p: j for j, p in enumerate(targets)}
    by_cell = [[] for _ in targets]
    by_piece = [[] for _ in pieces]
    candidates = []
    low = np.min(np.array(targets), axis=0)
    high = np.max(np.array(targets), axis=0)
    for i, (source, color) in enumerate(pieces):
        shape = source-source.min(axis=0)
        h, w = shape.max(axis=0)+1
        for r in range(int(low[0]), int(high[0]-h+2)):
            for c in range(int(low[1]), int(high[1]-w+2)):
                cells = shape+np.array([r, c])
                positions = [index.get(tuple(p)) for p in cells]
                if any(p is None for p in positions):
                    continue
                if np.array_equal(cells, source):
                    continue
                bits = sum(1 << p for p in positions)
                k = len(candidates)
                candidates.append((i, bits, cells, color))
                by_piece[i].append(k)
                for p in positions:
                    by_cell[p].append(k)
    full_cells = (1 << len(targets))-1
    full_pieces = (1 << len(pieces))-1
    solutions = {}

    def visit(used, filled, chosen):
        if len(solutions) >= limit:
            return
        if used == full_pieces:
            if filled == full_cells:
                answer = base.copy()
                placements = []
                for k in chosen:
                    i, _, cells, color = candidates[k]
                    answer[cells[:, 0], cells[:, 1]] = color
                    placements.append((i, cells))
                solutions.setdefault(answer.tobytes(), (answer, placements))
            return

        def available(options):
            return [k for k in options if not (used & (1 << candidates[k][0]) or filled & candidates[k][1])]

        options = None
        for i in range(len(pieces)):
            if not used & (1 << i):
                choices = available(by_piece[i])
                if not choices:
                    return
                if options is None or len(choices) < len(options):
                    options = choices
        remaining = full_cells ^ filled
        while remaining:
            bit = remaining & -remaining
            p = bit.bit_length()-1
            remaining -= bit
            choices = available(by_cell[p])
            if not choices:
                return
            if len(choices) < len(options):
                options = choices
                if len(options) == 1:
                    break
        for k in options:
            i, bits, _, _ = candidates[k]
            visit(used | (1 << i), filled | bits, chosen+[k])

    visit(0, 0, [])
    return list(solutions.values())


def _solve_info(grid):
    info = _scene(grid)
    base, targets = _prepare(info)
    answers = _exact_covers(base, targets, info['pieces'])
    _require(len(answers) == 1, "compacted sockets need one distinct colored exact cover")
    answer, placements = answers[0]
    info.update(base=base, targets=targets, placements=placements)
    return (answer.T if info['transpose'] else answer).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _solve_info(input_grid)[0]


def _raw_cover_count(info):
    """A-only ablation for the single-corridor teaching example.

    Search all full-width raw boundary windows, leaving other liner detail
    unused, as in A's socket search. Return at most two colored answers.
    """
    if len(info['corridors']) != 1:
        return 0
    widths = {int(cells[:, 1].max()-cells[:, 1].min()+1) for cells, _ in info['pieces']}
    if len(widths) != 1:
        return 0
    width = widths.pop()
    cor = info['corridors'][0]
    base = info['a'].copy()
    for cells, _ in info['pieces']:
        base[cells[:, 0], cells[:, 1]] = info['bg']
    solutions = set()
    for start in range(cor['raw'].shape[1]-width+1):
        holes = [(int(cor['rows'][u]), cor['left']+1+start+c)
                 for u, c in np.argwhere(cor['raw'][:, start:start+width] == info['bg'])]
        for answer, _ in _exact_covers(base, holes, info['pieces']):
            solutions.add(answer.tobytes())
            if len(solutions) >= 2:
                return 2
    return len(solutions)


def _liner_tokens(rng, width, colors):
    # A one-cell color difference must survive dedup even at equal height.
    first = (colors[0], colors[0])
    second = (colors[0], colors[1])
    pool = [p for h in range(1, 4) for p in product(colors, repeat=h) if p not in (first, second)]
    tokens = [first, second]
    third = (int(rng.choice(colors)),) if rng.random() < .5 else tuple(map(int, rng.choice(colors, size=3)))
    tokens.append(third)
    pool = [p for p in pool if p != third]
    for j in rng.choice(len(pool), size=width-3, replace=False):
        tokens.append(pool[int(j)])
    rng.shuffle(tokens)
    return tokens


def _cuts(rng, root, count):
    cuts = [np.array(root, dtype=int)]
    width = len(root)
    for _ in range(count-1):
        previous = cuts[-1]
        shape = rng.integers(0, 3, size=width)
        if np.all(shape == shape[0]):
            shape[0] = (shape[0]+1) % 3
        offset = max(int(np.max(previous+1-shape)),
                     int(np.max(np.maximum(previous[:-1], previous[1:])+1-
                                np.minimum(shape[:-1], shape[1:]))))
        cuts.append(shape+offset+int(rng.integers(0, 2)))
    cuts.append(np.full(width, max(4, int(cuts[-1].max())+int(rng.integers(1, 3)))))
    return cuts


def _indices(rng, heights, witness, short, test):
    m = len(heights)
    for _ in range(40):
        result = list(range(m))
        selected = int(rng.integers(0, m-2))
        insertion = int(rng.integers(selected+2, m))
        result.insert(insertion, selected)
        if [heights[i] for i in result[:m]] != list(heights):
            break
    else:
        return None
    if witness:
        result += list(range(m))*2
    else:
        extra = int(rng.integers(1, (3 if short else m+2)))
        tail = list(map(int, rng.integers(0, m, size=extra)))
        if test:
            # Additional duplicates appear before new types; no fixed crop works.
            for _ in range(2):
                position = int(rng.integers(1, len(result)))
                result.insert(position, int(rng.choice(result[:position])))
        result += tail
    return result


def _dilate(mask):
    result = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, -dr), min(h, h-dr)
            c0, c1 = max(0, -dc), min(w, w-dc)
            result[r0+dr:r1+dr, c0+dc:c1+dc] |= mask[r0:r1, c0:c1]
    return result


def _scatter(rng, base, pieces, bg, region):
    r0, r1 = region
    normalized = [(cells-cells.min(axis=0), color) for cells, color in pieces]
    for _ in range(24):
        a = base.copy()
        forbidden = _dilate(a != bg)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0])+rng.random()*4)
        for i in order:
            shape, color = normalized[i]
            h, w = map(int, shape.max(axis=0)+1)
            if r1-r0 < h or a.shape[1]-2 < w:
                break
            valid = np.ones((r1-r0-h+1, a.shape[1]-w-1), dtype=bool)
            for r, c in shape:
                valid &= ~forbidden[r0+r:r0+r+valid.shape[0], 1+c:1+c+valid.shape[1]]
            choices = np.argwhere(valid)
            if not len(choices):
                break
            origin = choices[int(rng.integers(len(choices)))]+np.array([r0, 1])
            target = shape+origin
            a[target[:, 0], target[:, 1]] = color
            occupied = np.zeros_like(a, dtype=bool)
            occupied[target[:, 0], target[:, 1]] = True
            forbidden |= _dilate(occupied)
        else:
            return a
    raise ValueError("source scatter space")


def _make_pair(rng, widths, counts, witness, test, transpose):
    palette = list(map(int, rng.permutation(10)))
    bg, band, frame, g0, g1 = palette[:5]
    descriptions = []
    for width, count in zip(widths, counts):
        tokens = _liner_tokens(rng, width, (g0, g1))
        cuts = _cuts(rng, list(map(len, tokens)), count)
        ids = _indices(rng, list(map(len, tokens)), witness, len(widths) > 1, test)
        if ids is None:
            raise ValueError("early-repeat profile witness")
        descriptions.append((tokens, cuts, ids))
    required_width = 2+sum(len(ids)+2 for _, _, ids in descriptions)+len(widths)-1
    if required_width > 30:
        raise ValueError("raw corridor width")
    width = int(rng.integers(max(16, required_width), 31))
    core = int(rng.integers(1, 4))
    depth = max(int(cuts[-1][0]) for _, cuts, _ in descriptions)
    if depth+core+10 > 30:
        raise ValueError("target depth")
    # Space is bounded before allocating any grids or scatter attempts.
    source_height = int(rng.integers(max(8, 15-depth//2), 31-depth-core-2))
    height = source_height+core+depth+2
    side = int(rng.choice([-1, 1]))
    b0 = depth+1 if side == -1 else source_height+1
    b1 = b0+core
    base = np.full((height, width), bg, dtype=np.int8)
    base[b0:b1] = band
    answer = base.copy()
    start = int(rng.integers(1, width-required_width+2))
    pieces = []
    total = sum(counts)
    color_count = int(rng.integers(2, min(5, total)+1)) if total >= 2 else 1
    color_order = list(map(int, rng.permutation(total)))
    for tokens, cuts, ids in descriptions:
        d = int(cuts[-1][0])
        rows = np.array([b0-1-u if side == -1 else b1+u for u in range(d)])
        right = start+len(ids)+1
        for grid in (base, answer):
            grid[rows, start] = frame
            grid[rows, right] = frame
        for j, i in enumerate(ids):
            base[rows[:len(tokens[i])], start+1+j] = tokens[i]
        for j, token in enumerate(tokens):
            answer[rows[:len(token)], start+1+j] = token
        for lo, hi in zip(cuts, cuts[1:]):
            cells = np.array([(int(rows[u]), start+1+c) for c in range(len(tokens))
                              for u in range(int(lo[c]), int(hi[c]))], dtype=int)
            color = palette[5+color_order[len(pieces)] % color_count]
            _require(np.all(answer[cells[:, 0], cells[:, 1]] == bg), "construction overlap")
            answer[cells[:, 0], cells[:, 1]] = color
            pieces.append((cells, color))
        start = right+2
    source_region = (b1+1, height-1) if side == -1 else (1, b0-1)
    inp = _scatter(rng, base, pieces, bg, source_region)
    if transpose:
        inp, answer = inp.T.copy(), answer.T.copy()
    predicted, info = _solve_info(inp)
    if not np.array_equal(predicted, answer):
        raise ValueError("construction differs from input-derived exact cover")
    if witness and _raw_cover_count(info) < 2:
        raise ValueError("raw sockets do not expose ambiguity")
    return {'input': inp.copy(), 'output': answer.copy()}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    orientation = int(rng.integers(3))  # H, V, or mixed; no hidden solving rule.
    shift = int(rng.integers(2))  # Width extrapolation or longer chains.
    seen = set()
    puzzle = {'train': [], 'test': []}
    for split, count in (('train', train_count), ('test', test_count)):
        for i in range(count):
            test = split == 'test'
            witness = not test and i == 0
            transpose = orientation == 1 or (orientation == 2 and (i % 2 == 1 if not test else bool(rng.integers(2))))
            for _ in range(100):
                if witness:
                    widths = [int(rng.integers(3, 6))]
                    counts = [2]
                elif not test and i == 1:
                    widths = [int(rng.integers(3, 5))]*2
                    counts = [int(rng.integers(1, 3)) for _ in widths]
                elif test and shift == 0:
                    widths = [int(rng.integers(6, 8))]
                    counts = [int(rng.integers(2, 4))]
                else:
                    number = int(rng.choice([1, 2]))
                    widths = [int(rng.integers(3, 5 if number == 2 else 6)) for _ in range(number)]
                    if number == 2 and rng.random() < .65:
                        widths[1] = widths[0]
                    counts = [int(rng.integers(1, 4)) for _ in widths]
                    if test:
                        counts[0] = 4
                try:
                    pair = _make_pair(rng, widths, counts, witness, test, transpose)
                except (ValueError, AssertionError):
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape and not np.array_equal(inp, out), "nontrivial same-canvas transform")
    predicted, info = _solve_info(inp)
    _require(np.array_equal(predicted, out), "incorrect output")
    a = out.T if info['transpose'] else out
    _require(np.array_equal(inp == info['band'], out == info['band']), "fixed band changed")
    _require(np.array_equal(inp == info['frame'], out == info['frame']), "fixed frames changed")
    permitted = info['reserved'].copy()
    for cells, _ in info['pieces']:
        permitted[cells[:, 0], cells[:, 1]] = True
        _require(np.all(a[cells[:, 0], cells[:, 1]] == info['bg']), "source not erased")
    _require(np.all(a[~permitted] == info['a'][~permitted]), "unaffected cells changed")
    used = set()
    for i, destination in info['placements']:
        _require(i not in used, "piece used twice")
        used.add(i)
        source, color = info['pieces'][i]
        delta = destination.min(axis=0)-source.min(axis=0)
        _require(np.any(delta != 0) and set(map(tuple, source+delta)) == set(map(tuple, destination)), "piece not translated intact")
        _require(np.all(a[destination[:, 0], destination[:, 1]] == color), "payload recolored")
    _require(len(used) == len(info['pieces']), "piece omitted")
    for cor in info['corridors']:
        m = cor['compact'].shape[1]
        cols = np.arange(cor['left']+1, cor['left']+1+m)
        rect = a[np.ix_(cor['rows'], cols)]
        guide = cor['compact'] != info['bg']
        _require(np.array_equal(rect[guide], cor['compact'][guide]), "retained liner columns altered")
        _require(np.all(rect != info['bg']), "rectangle not complete")
        _require(np.all(a[np.ix_(cor['rows'], np.arange(cor['left']+1+m, cor['right']))] == info['bg']), "discarded-column area not cleared")
    # B alone leaves precisely the target cells empty; A must supply them.
    _require(len(info['targets']) > 0 and not np.array_equal(info['base'], a), "A contributes no change")
    return info


def _evidence(info):
    early = separated = crop_fails = near = False
    for cor in info['corridors']:
        keys = cor['keys']
        last_new = max(keys.index(key) for key in set(keys))
        early |= any(keys[j] in keys[:j] for j in range(1, last_new))
        separated |= any(keys[j] in keys[:max(0, j-1)] and keys[j] != keys[j-1] for j in range(2, last_new+1))
        compact = cor['compact']
        crop_fails |= not np.array_equal(cor['raw'][:, :compact.shape[1]], compact)
        vectors = [tuple(map(int, col)) for col in compact.T]
        near |= any(sum(x != y for x, y in zip(v, w)) == 1 for j, v in enumerate(vectors) for w in vectors[j+1:])
    return early, separated, crop_fails, near


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, axes, test_axes = set(), set(), set()
    train_infos = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = _key(np.asarray(pair['input']))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == 'train':
                axes.add(info['transpose'])
                train_infos.append(info)
            else:
                test_axes.add(info['transpose'])
    _require(test_axes <= axes, "test orientation not demonstrated")
    _require(all(any(info['transpose'] == axis and all(_evidence(info)) for info in train_infos) for axis in axes), "missing vector/order witness")
    _require(any(_raw_cover_count(info) >= 2 for info in train_infos), "no raw alternative-cover witness")
    _require(any(len(info['corridors']) == 2 for info in train_infos), "joint socket matching not demonstrated")


def _fingerprint(puzzle):
    return tuple(tuple((_key(np.asarray(pair['input'])), _key(np.asarray(pair['output']))) for pair in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """42 seeds, streamed metrics, no saved grids, traces, logs, or retries."""
    # Colored-output uniqueness, rather than object permutations, is essential
    # when detached components reuse a color.
    fixture = np.zeros((3, 4), dtype=np.int8)
    dots = [(np.array([[0, 0]]), 2), (np.array([[2, 3]]), 3)]
    _require(len(_exact_covers(fixture, [(1, 1), (1, 2)], dots)) == 2, "ambiguous colors not detected")
    dots[1] = (dots[1][0], 2)
    _require(len(_exact_covers(fixture, [(1, 1), (1, 2)], dots)) == 1, "identical-color permutations counted twice")
    shapes, structures, roles, widths, depths, piece_counts, frame_counts = (set() for _ in range(7))
    regimes, fingerprints = set(), set()
    reuse = equal_width = same_height_distinct = 0
    pair_total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(hashlib.sha256(repr(_fingerprint(puzzle)).encode()).hexdigest())
        regimes.add(tuple(sorted({_scene(pair['input'])['transpose'] for pair in puzzle['train']})))
        local_shapes = set()
        for pair in puzzle['train']+puzzle['test']:
            pair_total += 1
            info = _scene(pair['input'])
            shapes.add(pair['input'].shape)
            local_shapes.add(pair['input'].shape)
            roles.add((info['bg'], info['band'], info['frame']))
            mapping = {}
            normalized = bytes(mapping.setdefault(int(v), len(mapping)) for v in pair['input'].flat)
            structures.add((pair['input'].shape, hashlib.sha256(normalized).digest()))
            frame_counts.add(len(info['corridors']))
            piece_counts.add(len(info['pieces']))
            reuse += len({color for _, color in info['pieces']}) < len(info['pieces'])
            ww = [cor['compact'].shape[1] for cor in info['corridors']]
            equal_width += len(ww) > len(set(ww))
            widths.update(ww)
            depths.update(cor['depth'] for cor in info['corridors'])
            for cor in info['corridors']:
                hs = [np.count_nonzero(col != info['bg']) for col in cor['compact'].T]
                same_height_distinct += len(hs) > len(set(hs))
            before = pair['input'].copy()
            _require(np.array_equal(solve(before), pair['output']) and np.array_equal(before, pair['input']), "solve purity")
        _require(len(local_shapes) >= 2, "within-puzzle geometry does not vary")
    _require(len(fingerprints) == 42 and len(structures) >= pair_total*.95, "structural diversity")
    _require(len(shapes) >= 40 and len(roles) >= 40 and widths == set(range(3, 8)), "shape/role/width coverage")
    _require(regimes == {(False,), (True,), (False, True)}, "orientation coverage")
    _require(frame_counts == {1, 2} and len(depths) >= 6 and len(piece_counts) >= 4, "packing variation")
    _require(reuse >= 20 and equal_width >= 42 and same_height_distinct >= pair_total, "color identity or full-vector evidence")
    _require(_fingerprint(generate_puzzle(17)) == _fingerprint(generate_puzzle(17)), "determinism")
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(1000+10*train+test, train, test)
            _require(len(p['train']) == train and len(p['test']) == test, "explicit counts")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (3, True), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    info = _scene(p['train'][0]['input'])
    _require(_raw_cover_count(info) == 2, "omitting B no longer changes the geometric answer")
    base, _ = _prepare(info)
    out = p['train'][0]['output'].T if info['transpose'] else p['train'][0]['output']
    _require(not np.array_equal(base, out), "omitting A has no effect")
    broken = {s: [dict(input=x['input'].copy(), output=x['output'].copy()) for x in p[s]] for s in p}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("identity label accepted")
    return dict(seeds=42, pairs=pair_total, shapes=len(shapes), structures=len(structures),
                role_assignments=len(roles), widths=sorted(widths), depths=sorted(depths),
                piece_counts=sorted(piece_counts), reused_color_pairs=reuse, equal_width_pairs=equal_width)
