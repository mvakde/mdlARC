"""Ordinal loom: transfer a stencil in keyed length-rank coordinates.

No file access, stored examples, generation state, or parent imports are used.
Generation paints its labels from logical offsets; solving independently reads
frames, keys and bars from pixels. All diagnostics run in bounded memory.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= array) & (array <= 9))), "colors must be 0..9")
    return array.astype(np.int8, copy=False)


def _signature(grid):
    return grid.shape, hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest()


def _frame_boxes(grid):
    """Frame identities are reserved within an input, never literal constants."""
    found = []
    for color in np.unique(grid):
        points = np.argwhere(grid == color)
        top, left = points.min(axis=0)
        bottom, right = points.max(axis=0) + 1
        tile = grid[top:bottom, left:right]
        h, w = tile.shape
        if h < 9 or w < 9:
            continue
        mask = tile == color
        if not (mask[0].all() and mask[-1].all()
                and mask[:, 0].all() and mask[:, -1].all()):
            continue
        rows = np.flatnonzero(mask[1:-1].all(axis=1)) + 1
        cols = np.flatnonzero(mask[:, 1:-1].all(axis=0)) + 1
        if len(rows) != 1 or len(cols) != 1:
            continue
        r, c = int(rows[0]), int(cols[0])
        expected = np.zeros_like(mask)
        expected[[0, -1], :] = True
        expected[:, [0, -1]] = True
        expected[r, :] = True
        expected[:, c] = True
        if np.array_equal(mask, expected):
            found.append((int(top), int(left), h, w, int(color), r, c))
    return found


def _unary(values, color, neutral):
    occupied = np.flatnonzero(values != neutral)
    _require(len(occupied) > 0, "missing unary key")
    _require(np.array_equal(occupied, np.arange(len(occupied))), "broken unary key")
    _require(bool(np.all(values[occupied] == color)), "mixed unary key")
    return len(occupied)


def _read_comb(region, cap, neutral):
    """Read vertical capped bars; a transpose handles horizontal row bars."""
    depth, span = region.shape
    records, payloads = [], set()
    for position in range(span):
        values = region[:, position]
        occupied = np.flatnonzero(values != neutral)
        if len(occupied) == 0:
            continue
        start = int(occupied[0])
        _require(np.array_equal(occupied, np.arange(start, depth)),
                 "bar must be contiguous and baseline aligned")
        _require(len(occupied) >= 3, "bar needs caps and payload")
        _require(int(values[start]) == int(values[-1]) == cap, "mismatched caps")
        payload = int(values[start + 1])
        _require(payload not in (neutral, cap)
                 and bool(np.all(values[start + 1:-1] == payload)),
                 "bar payload must be uniform and distinct")
        payloads.add(payload)
        records.append((position, len(occupied)))
    _require(3 <= len(records) <= 5, "comb population must be 3..5")
    _require(len({length for _, length in records}) == len(records), "tied lengths")
    _require(len(payloads) == 1, "comb payloads must agree")
    rank_to_position = tuple(position for position, _ in sorted(records,
                                                               key=lambda x: -x[1]))
    return dict(records=tuple(records), rank_to_position=rank_to_position,
                position_to_rank={p: i for i, p in enumerate(rank_to_position)},
                payload=next(iter(payloads)))


def _parse_board(grid, box):
    top, left, h, w, frame, r, c = box
    tile = grid[top:top + h, left:left + w]
    _require(5 <= r - 1 and 5 <= c - 1 and h <= 15 and w <= 15,
             "invalid board proportions")
    _require(h - r - 2 >= 3 and w - c - 2 >= 3, "missing plotting area")
    neutral = int(tile[1, 1])
    row_color, col_color = int(tile[1, 2]), int(tile[2, 1])
    _require(len({neutral, frame, row_color, col_color}) == 4, "erased board roles")
    row_key = _unary(tile[1, 2:c], row_color, neutral)
    col_key = _unary(tile[2:r, 1], col_color, neutral)
    key = np.full((r - 1, c - 1), neutral, dtype=np.int8)
    key[0, 1:row_key + 1] = row_color
    key[1:col_key + 1, 0] = col_color
    _require(np.array_equal(key, tile[1:r, 1:c]), "unrecognized key quadrant")
    columns = _read_comb(tile[1:r, c + 1:-1], col_color, neutral)
    rows = _read_comb(tile[r + 1:-1, 1:c].T, row_color, neutral)
    _require(columns['payload'] == rows['payload'], "axis payloads must agree")
    payload = rows['payload']
    _require(payload not in (neutral, frame, row_color, col_color), "erased payload role")
    _require(row_key <= len(rows['records']) and col_key <= len(columns['records']),
             "requested ordinal does not exist")
    core = tile[r + 1:-1, c + 1:-1]
    _require(set(map(int, np.unique(core))) <= {neutral, payload, row_color, col_color},
             "unknown stencil role")
    return dict(box=box, tile=tile, core=core, rows=rows, columns=columns,
                neutral=neutral, payload=payload, frame=frame, row_color=row_color,
                col_color=col_color, keys=(row_key, col_key))


def _connected(points):
    remaining = set(points)
    if not remaining:
        return False
    todo = deque([remaining.pop()])
    while todo:
        r, c = todo.popleft()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in remaining:
                remaining.remove(p)
                todo.append(p)
    return not remaining


def _read_stencil(reference):
    core, bg = reference['core'], reference['neutral']
    row_inverse = reference['rows']['position_to_rank']
    col_inverse = reference['columns']['position_to_rank']
    origin_r, origin_c = (x - 1 for x in reference['keys'])
    marker_position = (reference['rows']['rank_to_position'][origin_r],
                       reference['columns']['rank_to_position'][origin_c])
    markers = np.argwhere(core == reference['payload'])
    _require(len(markers) == 1 and tuple(markers[0]) == marker_position,
             "reference marker must register the keyed ranks")
    roles = {reference['payload']: 0, reference['row_color']: 1, reference['col_color']: 2}
    stencil = []
    for r, c in np.argwhere(core != bg):
        _require(int(r) in row_inverse and int(c) in col_inverse,
                 "stencil cell is not a bar intersection")
        stencil.append((row_inverse[int(r)] - origin_r,
                        col_inverse[int(c)] - origin_c, roles[int(core[r, c])]))
    _require(4 <= len(stencil) <= 10, "stencil size must be 4..10")
    _require({role for _, _, role in stencil} == {0, 1, 2}, "both motif roles required")
    _require(any(r for r, _, _ in stencil) and any(c for _, c, _ in stencil),
             "stencil must use both axes")
    _require(_connected((r, c) for r, c, _ in stencil), "stencil must connect in rank space")
    return tuple(sorted(stencil))


def _transfer(stencil, target, *, keys=None):
    """Pixel-reading solver path: does not use generation geometry records."""
    out = np.full(target['core'].shape, target['neutral'], dtype=np.int8)
    kr, kc = target['keys'] if keys is None else keys
    row_map = target['rows']['rank_to_position']
    col_map = target['columns']['rank_to_position']
    colors = (target['payload'], target['row_color'], target['col_color'])
    for dr, dc, role in stencil:
        r, c = kr - 1 + dr, kc - 1 + dc
        _require(0 <= r < len(row_map) and 0 <= c < len(col_map),
                 "stencil transfer would leave the finite rank domain")
        out[row_map[r], col_map[c]] = colors[role]
    return out


def _parse_canonical(grid):
    boxes = _frame_boxes(grid)
    _require(len(boxes) == 2, "expected exactly two reserved frame identities")
    boards = [_parse_board(grid, box) for box in boxes]
    _require(len({b['neutral'] for b in boards}) == 1, "board neutrals must agree")
    _require(len({b['payload'] for b in boards}) == 1, "board payloads must agree")
    frames = {b['frame'] for b in boards}
    other_roles = {b[k] for b in boards for k in ('neutral', 'payload', 'row_color', 'col_color')}
    _require(not (frames & other_roles), "frame identities must be reserved")
    first, second = [{b['row_color'], b['col_color']} for b in boards]
    _require(first == second or not (first & second), "cap palettes must be shared or remapped")
    if first == second:
        _require(boards[0]['row_color'] == boards[1]['row_color'], "shared axis roles must agree")
    exterior = np.ones(grid.shape, dtype=bool)
    for top, left, h, w, *_ in boxes:
        _require(bool(np.all(exterior[top:top + h, left:left + w])), "boards overlap")
        exterior[top:top + h, left:left + w] = False
    bg = boards[0]['neutral']
    _require(bool(np.all(grid[exterior] == bg)), "unexpected exterior cells")
    blank = [bool(np.all(b['core'] == bg)) for b in boards]
    _require(sum(blank) == 1, "need a completed stencil and one blank plotting area")
    target_index = blank.index(True)
    target, reference = boards[target_index], boards[1 - target_index]
    stencil = _read_stencil(reference)
    out = _transfer(stencil, target)
    return dict(reference=reference, target=target, stencil=stencil, answer=out,
                board_order=target_index, neutral=bg)


def _decode(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(grid, -rotation)
        try:
            decoded = _parse_canonical(canonical)
        except AssertionError:
            continue
        candidates.append((rotation, decoded))
    _require(len(candidates) == 1, "scene needs a unique common key orientation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct only the blank plotting rectangle, entirely from its input."""
    rotation, decoded = _decode(input_grid)
    return np.ascontiguousarray(np.rot90(decoded['answer'], rotation))


def _positions(rng, count, budget):
    extra = int(rng.integers(0, min(3, budget) + 1))
    gaps = rng.multinomial(extra, [1 / (count + 1)] * (count + 1))
    positions, cursor = [], int(gaps[0])
    for i in range(count):
        positions.append(cursor)
        cursor += 1 + int(gaps[i + 1])
    return tuple(positions), count + extra


def _permutation(rng, count, anchor, scramble):
    if not scramble:
        return tuple(range(count))
    order = [int(x) for x in rng.permutation(count)]
    # The selected ordinal must visibly differ from the corresponding spatial
    # ordinal. This provides a teaching witness, not an additional solve rule.
    if order.index(anchor) == anchor:
        other = (anchor + int(rng.integers(1, count))) % count
        order[anchor], order[other] = order[other], order[anchor]
    return tuple(order)


def _geometry(rng):
    nr, nc = map(int, rng.integers(3, 6, size=2))
    left_depth = int(rng.integers(nr + 2, 13 - nc))
    top_depth = int(rng.integers(nc + 2, 13 - nr))
    rows, rh = _positions(rng, nr, 12 - top_depth - nr)
    cols, cw = _positions(rng, nc, 12 - left_depth - nc)
    row_lengths = tuple(sorted(map(int, rng.choice(np.arange(3, left_depth + 1), nr,
                                                  replace=False)), reverse=True))
    col_lengths = tuple(sorted(map(int, rng.choice(np.arange(3, top_depth + 1), nc,
                                                  replace=False)), reverse=True))
    return dict(nr=nr, nc=nc, rows=rows, cols=cols, rh=rh, cw=cw,
                left_depth=left_depth, top_depth=top_depth,
                row_lengths=row_lengths, col_lengths=col_lengths)


def _origin(rng, reference, target, target_keys):
    keys = []
    for size_key, target_key in zip(('nr', 'nc'), target_keys):
        n, m = reference[size_key], target[size_key]
        options = [k for k in range(1, n + 1)
                   if min(k - 1, target_key - 1) + min(n - k, m - target_key) >= 1]
        keys.append(int(rng.choice(options)))
    return tuple(keys)


def _new_stencil(rng, reference, target, ref_keys, target_keys):
    limits = []
    for size_key, a, b in zip(('nr', 'nc'), ref_keys, target_keys):
        limits.append((-min(a - 1, b - 1), min(reference[size_key] - a, target[size_key] - b)))
    (lo_r, hi_r), (lo_c, hi_c) = limits
    dr = int(rng.choice([x for x in (-1, 1) if lo_r <= x <= hi_r]))
    dc = int(rng.choice([x for x in (-1, 1) if lo_c <= x <= hi_c]))
    cells = {(0, 0): 0, (dr, 0): 1, (0, dc): 2}
    area = (hi_r - lo_r + 1) * (hi_c - lo_c + 1)
    desired = int(rng.integers(4, min(10, area) + 1))
    while len(cells) < desired:
        frontier = set()
        for r, c in cells:
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if lo_r <= p[0] <= hi_r and lo_c <= p[1] <= hi_c and p not in cells:
                    frontier.add(p)
        point = sorted(frontier)[int(rng.integers(len(frontier)))]
        cells[point] = int(rng.integers(1, 3))
    return tuple((r, c, role) for (r, c), role in sorted(cells.items()))


def _paint_from_geometry(geometry, keys, stencil, colors):
    """Construction-only label painter, independent of the pixel parser."""
    bg, payload, _, row_color, col_color = colors
    out = np.full((geometry['rh'], geometry['cw']), bg, dtype=np.int8)
    row_lookup = dict(zip(geometry['row_order'], geometry['rows']))
    col_lookup = dict(zip(geometry['col_order'], geometry['cols']))
    for dr, dc, role in stencil:
        position = (row_lookup[keys[0] - 1 + dr], col_lookup[keys[1] - 1 + dc])
        out[position] = (payload, row_color, col_color)[role]
    return out


def _draw_board(geometry, keys, colors, stencil=None):
    bg, payload, frame, row_color, col_color = colors
    l, t = geometry['left_depth'], geometry['top_depth']
    r, c = t + 1, l + 1
    board = np.full((t + geometry['rh'] + 3, l + geometry['cw'] + 3), bg, dtype=np.int8)
    board[[0, -1], :] = frame
    board[:, [0, -1]] = frame
    board[r, :] = frame
    board[:, c] = frame
    board[1, 2:keys[0] + 2] = row_color
    board[2:keys[1] + 2, 1] = col_color
    for y, rank in zip(geometry['rows'], geometry['row_order']):
        length = geometry['row_lengths'][rank]
        start = l - length + 1
        row = r + 1 + y
        board[row, start:l + 1] = payload
        board[row, start] = board[row, l] = row_color
    for x, rank in zip(geometry['cols'], geometry['col_order']):
        length = geometry['col_lengths'][rank]
        start = t - length + 1
        column = c + 1 + x
        board[start:t + 1, column] = payload
        board[start, column] = board[t, column] = col_color
    if stencil is not None:
        board[r + 1:-1, c + 1:-1] = _paint_from_geometry(geometry, keys, stencil, colors)
    return board


def _assemble(rng, boards, bg, axis):
    order = [int(i) for i in rng.permutation(2)]
    boards = [boards[i] for i in order]
    along = sum(b.shape[axis] for b in boards)
    gap = int(rng.integers(0, min(2, 30 - along) + 1))
    margin = int(rng.integers(0, min(2, 30 - along - gap) + 1))
    before = int(rng.integers(margin + 1))
    cross = max(b.shape[1 - axis] for b in boards) + int(rng.integers(3))
    shape = (along + gap + margin, cross) if axis == 0 else (cross, along + gap + margin)
    canvas = np.full(shape, bg, dtype=np.int8)
    cursor = before
    for board in boards:
        offset = int(rng.integers(cross - board.shape[1 - axis] + 1))
        row, col = (cursor, offset) if axis == 0 else (offset, cursor)
        canvas[row:row + board.shape[0], col:col + board.shape[1]] = board
        cursor += board.shape[axis] + gap
    return canvas


def _make_pair(rng, style, mode, *, reference_scramble=False):
    reference, target = _geometry(rng), _geometry(rng)
    target_keys = (1, 2) if mode == 'rows' else ((2, 1) if mode == 'columns'
                                               else tuple(map(int, rng.integers(1, 3, size=2))))
    ref_keys = _origin(rng, reference, target, target_keys)
    for geometry, keys, is_target in ((reference, ref_keys, False), (target, target_keys, True)):
        scramble_rows = mode in ('rows', 'both') if is_target else reference_scramble
        scramble_cols = mode in ('columns', 'both') if is_target else reference_scramble
        geometry['row_order'] = _permutation(rng, geometry['nr'], keys[0] - 1, scramble_rows)
        geometry['col_order'] = _permutation(rng, geometry['nc'], keys[1] - 1, scramble_cols)
    palette = style['palette'] if rng.random() >= style['variation'] else rng.permutation(10)
    bg, payload, frame_r, frame_t, row_r, col_r, row_t, col_t = map(int, palette[:8])
    if rng.integers(2):
        row_t, col_t = row_r, col_r
    ref_colors = (bg, payload, frame_r, row_r, col_r)
    target_colors = (bg, payload, frame_t, row_t, col_t)
    stencil = _new_stencil(rng, reference, target, ref_keys, target_keys)
    boards = [_draw_board(reference, ref_keys, ref_colors, stencil),
              _draw_board(target, target_keys, target_colors)]
    out = _paint_from_geometry(target, target_keys, stencil, target_colors)
    axis = int(rng.integers(2))
    canvas = _assemble(rng, boards, bg, axis)
    rotation = int(rng.integers(4))
    return {'input': np.ascontiguousarray(np.rot90(canvas, rotation)),
            'output': np.ascontiguousarray(np.rot90(out, rotation))}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    a = _count(num_train, 2, 10, 'num_train')
    b = _count(num_test, 1, 4, 'num_test')
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    style = dict(palette=rng.permutation(10), variation=float(rng.choice([0., .5, 1.])))
    modes = ['rows', 'columns']
    rng.shuffle(modes)
    modes += [str(rng.choice(['rows', 'columns'])) for _ in range(a - 2)]
    puzzle, seen, motifs = {'train': [], 'test': []}, set(), set()
    for split, schedule in (('train', modes), ('test', ['both'] * b)):
        for index, mode in enumerate(schedule):
            for _ in range(64):
                pair = _make_pair(rng, style, mode, reference_scramble=(index >= 2 and bool(rng.integers(2))))
                signature = _signature(pair['input'])
                # This parse is a construction guard, never the source of labels.
                _, decoded = _decode(pair['input'])
                motif = decoded['stencil']
                if signature in seen or (split == 'train' and index < 2 and motif in motifs):
                    continue
                seen.add(signature)
                motifs.add(motif)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _teaching_checks(analyses):
    row_witness = col_witness = copy_witness = False
    row_ordinals, col_ordinals, motifs = set(), set(), set()
    modes = set()
    for _, data in analyses:
        target, reference = data['target'], data['reference']
        kr, kc = target['keys']
        row_ordinals.add(kr)
        col_ordinals.add(kc)
        motifs.add(data['stencil'])
        row_positions = [p for p, _ in target['rows']['records']]
        col_positions = [p for p, _ in target['columns']['records']]
        row_witness |= target['rows']['rank_to_position'][kr - 1] != row_positions[kr - 1]
        col_witness |= target['columns']['rank_to_position'][kc - 1] != col_positions[kc - 1]
        row_ordered = target['rows']['rank_to_position'] == tuple(row_positions)
        col_ordered = target['columns']['rank_to_position'] == tuple(col_positions)
        modes.add((row_ordered, col_ordered))
        mapping = {reference['neutral']: target['neutral'], reference['payload']: target['payload'],
                   reference['row_color']: target['row_color'], reference['col_color']: target['col_color']}
        recolored = np.array([[mapping[int(x)] for x in row] for row in reference['core']], dtype=np.int8)
        copy_witness |= not np.array_equal(recolored, data['answer'])
    _require({1, 2} <= row_ordinals and {1, 2} <= col_ordinals,
             "training must contrast both unary origins")
    _require(row_witness and col_witness, "training must refute spatial ordinal selection on both axes")
    _require({(False, True), (True, False)} <= modes, "training must isolate both axis permutations")
    _require(copy_witness, "training must refute direct recolored copying")
    _require(len(motifs) >= 2, "training must expose different reference stencils")


def validate_generated_puzzle(puzzle):
    """Check pixel semantics first, then the deliberately narrow curriculum."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            rotation, decoded = _decode(inp)
            expected = np.rot90(decoded['answer'], rotation)
            _require(np.array_equal(out, expected), "incorrect stencil transport")
            _require(int(np.count_nonzero(out != decoded['neutral'])) == len(decoded['stencil']),
                     "lost or invented stencil cells")
            analyses[split].append((rotation, decoded))
    _teaching_checks(analyses['train'])
    for _, data in analyses['test']:
        target = data['target']
        _require(all(target['keys'][axis] in {d['target']['keys'][axis]
                                             for _, d in analyses['train']} for axis in (0, 1)),
                 "test unary values must have target demonstrations")
        for name in ('rows', 'columns'):
            _require(target[name]['rank_to_position'] != tuple(p for p, _ in target[name]['records']),
                     "tests must combine the two demonstrated permutations")


def _fixture():
    """Small exact rule witness; used only in self-test, never in solve."""
    base = dict(nr=5, nc=5, rows=tuple(range(5)), cols=tuple(range(5)), rh=5, cw=5,
                left_depth=7, top_depth=7, row_lengths=(7, 6, 5, 4, 3),
                col_lengths=(7, 6, 5, 4, 3), row_order=tuple(range(5)), col_order=tuple(range(5)))
    target = dict(base, row_order=(2, 0, 4, 1, 3), col_order=(3, 1, 4, 0, 2))
    stencil = ((0, 0, 0), (0, 1, 2), (1, 0, 1), (1, 1, 1))
    ref = _draw_board(base, (2, 2), (0, 2, 4, 6, 7), stencil)
    tar = _draw_board(target, (3, 3), (0, 2, 5, 1, 3))
    return np.hstack((ref, tar))


def _diagnostics():
    grid = _fixture()
    expected = np.array([[int(c) for c in row] for row in
                         ('30002', '00000', '00000', '00000', '10001')], dtype=np.int8)
    _require(np.array_equal(solve(grid), expected), "exact ordinal-coordinate diagnostic")
    # A: alter only the informative reference blueprint, keeping target fixed.
    changed_reference = grid.copy()
    changed_reference[11, 11] = 7  # reference ranks (3,3), row role -> column role
    reference_answer = solve(changed_reference)
    expected_reference = expected.copy()
    expected_reference[4, 0] = 3
    _require(np.array_equal(reference_answer, expected_reference), "reference counterfactual")
    changed_shape = grid.copy()
    changed_shape[9, 10] = 6  # add a reference arm at rank offset (-1,0)
    expected_shape = expected.copy()
    expected_shape[3, 4] = 1
    _require(np.array_equal(solve(changed_shape), expected_shape), "reference geometry counterfactual")
    # B: alter only a target unary key, preserving every bar and the stencil.
    changed_key = grid.copy()
    changed_key[1, 19] = 0  # row unary 3 -> 2 in right-hand board
    key_answer = solve(changed_key)
    _require(not np.array_equal(key_answer, expected), "unary key counterfactual")
    _require(tuple(np.argwhere(key_answer == 2)[0]) == (3, 4), "keyed origin must move by rank")
    # B: swap only two target row-bar lengths. The answer must follow rank,
    # even though physical rows, unary keys and the reference are unchanged.
    changed_lengths = grid.copy()
    changed_lengths[9, 16:23], changed_lengths[13, 16:23] = (
        grid[13, 16:23].copy(), grid[9, 16:23].copy())
    length_answer = solve(changed_lengths)
    _require(not np.array_equal(length_answer, expected), "length-ranking counterfactual")
    _require(tuple(np.argwhere(length_answer == 2)[0]) == (4, 4), "origin must follow changed lengths")
    _, decoded = _decode(grid)
    ref = decoded['reference']
    ascending_origin = (ref['rows']['rank_to_position'][::-1][ref['keys'][0] - 1],
                        ref['columns']['rank_to_position'][::-1][ref['keys'][1] - 1])
    _require(ref['core'][ascending_origin] != ref['payload'], "ascending ranks contradict visible origin")
    palette = np.array([9, 4, 7, 0, 5, 8, 1, 3, 6, 2], dtype=np.int8)
    for reflection in (False, True):
        x = np.fliplr(grid) if reflection else grid
        y = np.fliplr(expected) if reflection else expected
        for rotation in range(4):
            transformed = np.rot90(x, rotation)
            answer = np.rot90(y, rotation)
            _require(np.array_equal(solve(transformed), answer), "dihedral geometry invariance")
            _require(np.array_equal(solve(palette[transformed]), palette[answer]), "palette invariance")
    # Change one reference marker into a neutral cell: invalid inputs must not
    # silently obtain a convenient answer.
    broken = grid.copy()
    broken[10, 10] = 0
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("unregistered reference accepted")


def self_test():
    """Stream 42 seeds; retain hashes/features, never a corpus of payloads."""
    shapes, palettes, motifs, populations, lengths, ranks = set(), set(), set(), set(), set(), set()
    orders, rotations, signatures = set(), set(), set()
    spacers = remapped = shortest = disconnected = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pairs += 1
                rotation, data = _decode(pair['input'])
                rotations.add(rotation)
                shapes.add(('input', pair['input'].shape))
                shapes.add(('output', pair['output'].shape))
                palettes.add(tuple(map(int, np.unique(pair['input']))))
                motifs.add(data['stencil'])
                orders.add(data['board_order'])
                a, b = data['reference'], data['target']
                remapped += a['row_color'] != b['row_color']
                disconnected += not _connected(map(tuple, np.argwhere(pair['output'] != data['neutral'])))
                for board in (a, b):
                    for axis, name in enumerate(('rows', 'columns')):
                        comb = board[name]
                        n = len(comb['records'])
                        populations.add(n)
                        ranks.add(board['keys'][axis])
                        shortest += board['keys'][axis] == n
                        lengths.add(tuple(sorted(length for _, length in comb['records'])))
                        spacers += board['core'].shape[axis] > n
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), "fresh input-only solve")
    _require(len(signatures) == 42, "seed diversity")
    _require(len(shapes) >= 25 and len(motifs) >= 80 and len(lengths) >= 15, "structural variation")
    _require(populations == {3, 4, 5} and {1, 2, 3} <= ranks, "population/ordinal coverage")
    _require(rotations == {0, 1, 2, 3} and orders == {0, 1}, "layout/orientation variation")
    _require(len(palettes) >= 10 and min(spacers, remapped, shortest, disconnected) > 0,
             "spacer/palette/shortest/disconnected coverage")
    first = generate_puzzle(3198304, 2, 1)
    again = generate_puzzle(3198304, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for split in first for p, q in zip(first[split], again[split])
                 for k in ('input', 'output')), "whole-puzzle determinism")
    legal_count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(538841416, a, b)
            _require((len(puzzle['train']), len(puzzle['test'])) == (a, b), "explicit counts")
            legal_count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 0.5, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    _diagnostics()
    return dict(valid=True, seeds=42, pairs=pairs, legal_count_calls=legal_count_calls,
                grid_shapes=len(shapes), ordinal_stencils=len(motifs), length_sets=len(lengths),
                palettes=len(palettes), populations=sorted(populations), key_values=sorted(ranks),
                spacer_axes=spacers, remapped_pairs=remapped, shortest_origins=shortest,
                physically_disconnected_outputs=disconnected, independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
