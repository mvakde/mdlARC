"""Two signals branch around bars inside marker-selected permeable territories.

The control map and arena are coordinate-aligned. All solution information is
visible; sampling preferences never enter solve(). Only NumPy and stdlib are used.
"""
from __future__ import annotations

import json
import numpy as np


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _canonical(a, direction):
    return (a, np.flipud(a), a.T, np.fliplr(a).T)[direction]


def _actual(a, direction):
    return np.ascontiguousarray((a, np.flipud(a), a.T, np.fliplr(a.T))[direction])


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, queue = [start], [start]
        while queue:
            r, c = queue.pop()
            for dr, dc in offsets:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    queue.append(p)
        result.append(cells)
    return result


def _cut_info(cut):
    """Recognize a simple digital path, then flood its two four-connected sides."""
    h, w = cut.shape
    cells = set(map(tuple, np.argwhere(cut)))
    _require(len(cells) >= 3, "cut too short")
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if (dr or dc) and q in cells:
                    if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                        continue  # Do not shortcut an ordinary right-angle bend.
                    neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p, neighbors in graph.items() if len(neighbors) == 1]
    _require(len(ends) == 2 and all(1 <= len(q) <= 2 for q in graph.values()),
             "cut must be a path without branches or loops")
    boundary = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(boundary(p) for p in ends), "cut endpoints must touch boundary")
    _require(all(not boundary(p) for p in cells if p not in ends), "extra boundary contact")
    order, last, p = [], None, min(ends)
    while p is not None and p not in order:
        order.append(p)
        following = [q for q in graph[p] if q != last]
        last, p = p, following[0] if following else None
    _require(len(order) == len(cells), "disconnected cut")
    parts = _components(~cut)
    _require(len(parts) == 2, "cut must separate exactly two regions")
    regions, sites = [], []
    near = cut.copy()
    for r, c in cells:
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    for part in parts:
        mask = np.zeros(cut.shape, dtype=bool)
        for p in part:
            mask[p] = True
        regions.append(mask)
        sites.append([tuple(map(int, p)) for p in np.argwhere(mask & ~near)])
    _require(all(sites), "each side needs a separated marker site")
    steps = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(order, order[1:])]
    bends = sum(a != b for a, b in zip(steps, steps[1:]))
    return regions, sites, bends


def _pack(control, arena, frame):
    h, w = control.shape
    if 2 * w + 3 <= 30:
        a = np.full((h + 2, 2 * w + 3), frame, dtype=np.int8)
        a[1:-1, 1:w + 1] = control
        a[1:-1, w + 2:-1] = arena
    else:
        _require(2 * h + 3 <= 30 and w + 2 <= 30, "panels exceed canvas")
        a = np.full((2 * h + 3, w + 2), frame, dtype=np.int8)
        a[1:h + 1, 1:-1] = control
        a[h + 2:-1, 1:-1] = arena
    return a


def _decode(value):
    a = _grid(value)
    height, width = a.shape
    _require(height >= 5 and width >= 5, "invalid two-panel frame")
    frame = int(a[0, 0])
    _require(np.all(a[[0, -1], :] == frame) and np.all(a[:, [0, -1]] == frame),
             "broken frame")
    columns = [int(c) for c in range(1, width - 1) if np.all(a[:, c] == frame)]
    rows = [int(r) for r in range(1, height - 1) if np.all(a[r, :] == frame)]
    _require(len(columns) + len(rows) == 1, "need one full divider")
    if columns:
        middle = columns[0]
        _require(2 * middle == width - 1, "unequal side-by-side panels")
        control_slice = (slice(1, -1), slice(1, middle))
        arena_slice = (slice(1, -1), slice(middle + 1, -1))
    else:
        middle = rows[0]
        _require(2 * middle == height - 1, "unequal stacked panels")
        control_slice = (slice(1, middle), slice(1, -1))
        arena_slice = (slice(middle + 1, -1), slice(1, -1))
    control, arena = a[control_slice], a[arena_slice]
    _require(frame not in control and frame not in arena, "frame color inside panel")
    values, counts = np.unique(control, return_counts=True)
    _require(len(values) == 3 and np.sum(counts == max(counts)) == 1, "unreadable map palette")
    bg = int(values[np.argmax(counts)])
    markers = [int(c) for c, n in zip(values, counts) if n == 1 and c != bg]
    _require(len(markers) == 1, "map needs one singleton marker color")
    preferred = markers[0]
    cut_color = next(int(c) for c in values if c not in (bg, preferred))
    _require(len({frame, bg, preferred, cut_color}) == 4, "map roles must differ")
    marker = tuple(map(int, np.argwhere(control == preferred)[0]))
    cut = control == cut_color
    regions, sites, bends = _cut_info(cut)
    side = 0 if regions[0][marker] else 1
    _require(marker in sites[side], "marker touches cut")
    colors = set(map(int, np.unique(arena)))
    _require(len(colors) == 4 and {bg, preferred} <= colors and not colors & {frame, cut_color},
             "arena needs background, two signals and one obstacle color")
    candidates = []
    for other in sorted(colors - {bg, preferred}):
        obstacle = next(c for c in colors if c not in (bg, preferred, other))
        for direction in range(4):
            board = _canonical(arena, direction)
            entries, lengths = {}, {}
            good = True
            for color in (preferred, other):
                cols = list(map(int, np.flatnonzero(board[0] == color)))
                if not cols:
                    good = False
                    break
                mask = np.zeros(board.shape, dtype=bool)
                lengths[color] = []
                for c in cols:
                    rows = np.flatnonzero(board[:, c] == color)
                    if not 2 <= len(rows) < board.shape[0] or not np.array_equal(rows, np.arange(len(rows))):
                        good = False
                        break
                    mask[:len(rows), c] = True
                    lengths[color].append(len(rows))
                if not good or not np.array_equal(mask, board == color):
                    good = False
                    break
                entries[color] = cols
            if good:
                candidates.append((direction, other, obstacle, entries, lengths))
    _require(len(candidates) == 1, "signal roles or entry direction ambiguous")
    direction, other, obstacle, entries, lengths = candidates[0]
    board = _canonical(arena, direction).copy()
    seam = _canonical(cut, direction)
    marked = _canonical(regions[side], direction)
    allowed = {preferred: marked | seam, other: (~marked) | seam}
    # ~marked includes the seam; the two non-seam sides are complementary.
    for color in (preferred, other):
        _require(np.all(allowed[color][board == color]), "entry stub outside its own territory")
    return dict(input=a, control=control, arena=arena, arena_slice=arena_slice, board=board, bg=bg, frame=frame,
                cut_color=cut_color, preferred=preferred, other=other, obstacle=obstacle,
                marker=marker, side=side, cut=cut, seam=seam, marked=marked,
                allowed=allowed, direction=direction, entries=entries, lengths=lengths,
                bends=bends, sites=sites, regions=regions)


def _run(d, mode="rule"):
    """Keep ray states separate from painted cells; seam overlap is not a new ray.

    Counterfactual modes are only used to check demonstrations, never to label them.
    """
    board, bg, wall = d['board'], d['bg'], d['obstacle']
    first, second = d['preferred'], d['other']
    h, w = board.shape
    allowed = d['allowed']
    if mode in ("unrestricted", "postmask"):
        allowed = {c: np.ones((h, w), dtype=bool) for c in (first, second)}
    elif mode == "reverse_marker":
        allowed = {first: d['allowed'][second], second: d['allowed'][first]}
    traces = {c: board == c for c in (first, second)}
    active = {c: set(d['entries'][c]) for c in (first, second)}
    events, rejections, two_flanks, merges, clipped = [], 0, 0, 0, 0
    for r in range(1, h):
        for color in (first, second):
            following = set()
            handled = set()
            for c in sorted(active[color]):
                # Source stubs are immutable injections; valid rule inputs put
                # them inside their assigned side. Reverse-marker is faulty.
                if not allowed[color][r, c]:
                    rejections += 1
                    continue
                if board[r, c] != wall:
                    traces[color][r, c] = True
                    following.add(c)
                    continue
                left = right = c
                while left > 0 and board[r, left - 1] == wall:
                    left -= 1
                while right + 1 < w and board[r, right + 1] == wall:
                    right += 1
                exits = []
                suppressed = mode == 'one_per_layer' and handled and (left, right) not in handled
                handled.add((left, right))
                if mode != "no_branch" and not suppressed:
                    for step, target in ((-1, left - 1), (1, right + 1)):
                        reached = True
                        for x in range(c + step, target + step, step):
                            if not 0 <= x < w:
                                clipped += 1
                                reached = False
                                break
                            if not allowed[color][r - 1, x] or board[r - 1, x] == wall:
                                rejections += 1
                                if mode == 'jump_detour':
                                    continue
                                reached = False
                                break
                            traces[color][r - 1, x] = True
                        if reached and allowed[color][r, target] and board[r, target] != wall:
                            traces[color][r, target] = True
                            merges += target in following
                            following.add(target)
                            exits.append(target)
                        elif reached:
                            rejections += 1
                two_flanks += len(exits) == 2
                events.append((r, left, right, color, c, tuple(exits)))
            active[color] = following
    if mode == "postmask":
        for color in (first, second):
            traces[color] &= d['allowed'][color]
            traces[color] |= board == color
    if mode == "outline_all":
        # A local, wrong outlining rule: add every bar's upstream span for
        # either permitted color, even if no incoming signal ever reaches it.
        for r in range(1, h):
            c = 0
            while c < w:
                if board[r, c] != wall:
                    c += 1
                    continue
                left = c
                while c + 1 < w and board[r, c + 1] == wall:
                    c += 1
                for color in (first, second):
                    for x in range(max(0, left - 1), min(w, c + 2)):
                        if allowed[color][r - 1, x] and board[r - 1, x] != wall:
                            traces[color][r - 1, x] = True
                c += 1
    result = board.copy()
    blank = board == bg
    # The marked signal wins wherever BOTH newly drawn traces use the seam.
    priority = (first, second) if mode == "reverse_priority" else (second, first)
    for color in priority:
        result[blank & traces[color]] = color
    reached = {(r, a, b) for r, a, b, _, _, _ in events}
    all_runs = set()
    for r in range(h):
        c = 0
        while c < w:
            if board[r, c] != wall:
                c += 1
                continue
            left = c
            while c + 1 < w and board[r, c + 1] == wall:
                c += 1
            all_runs.add((r, left, c))
            c += 1
    overlap = blank & traces[first] & traces[second]
    stats = dict(events=events, reached=reached, inert=all_runs - reached,
                 two_flanks=two_flanks, rejections=rejections, merges=merges, clipped=clipped,
                 overlap=int(overlap.sum()), trace_masks=traces)
    return result, stats


def _whole_output(d, board):
    out = d['input'].copy()
    out[d['arena_slice']] = _actual(board, d['direction'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, topology, local permeability and every ray from input only."""
    d = _decode(input_grid)
    board, _ = _run(d)
    return _whole_output(d, board)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _flags(d):
    correct, stats = _run(d)
    flags = {name for name in ("no_branch", "unrestricted", "postmask", "reverse_marker",
                              "reverse_priority", "outline_all", "jump_detour", "one_per_layer")
             if not np.array_equal(_run(d, name)[0], correct)}
    return flags, stats


def _path(h, w, columns, rows):
    cut = np.zeros((h, w), dtype=bool)
    last, c = 0, columns[0]
    for row, new in zip(rows, columns[1:]):
        cut[last:row + 1, c] = True
        cut[row, min(c, new):max(c, new) + 1] = True
        last, c = row, new
    cut[last:, c] = True
    return cut


def _sample_cut(rng, h, w, kind, steps=0):
    k = int(rng.integers(3, w - 3))
    if kind == 'vertical':
        return _path(h, w, [k], []), k
    if kind == 'diagonal':
        end = int(rng.integers(2, w - 2))
        cols = np.rint(np.linspace(k, end, h)).astype(int)
        cut = np.zeros((h, w), dtype=bool)
        cut[np.arange(h), cols] = True
        return cut, k
    if kind == 'elbow':
        row = int(rng.integers(6, h - 2))
        cut = np.zeros((h, w), dtype=bool)
        cut[:row + 1, k] = True
        if rng.random() < .5:
            cut[row, :k + 1] = True
        else:
            cut[row, k:] = True
        return cut, k
    rows = sorted(map(int, rng.choice(np.arange(5, h - 2, 3), steps, replace=False)))
    columns = [k]
    for _ in rows:
        choices = [x for x in range(2, w - 2) if x != columns[-1]]
        columns.append(int(rng.choice(choices)))
    return _path(h, w, columns, rows), k


def _stub_columns(rng, available, count):
    cols = []
    for c in map(int, rng.permutation(available)):
        if all(abs(c - x) >= 2 for x in cols):
            cols.append(c)
            if len(cols) == count:
                break
    return sorted(cols)


def _make_input(rng, palette, direction, side, kind, shift, witness=None):
    bg, frame, cut_color, wall, first, second = palette
    if witness == 'seam':
        h, w = int(rng.integers(12, 20)), int(rng.integers(9, 14))
        k = int(rng.integers(4, w - 4))
        cut = _path(h, w, [k], [])
    elif witness == 'return':
        h, w = int(rng.integers(16, 22)), int(rng.integers(9, 14))
        k = int(rng.integers(4, w - 4))
        distance = int(rng.integers(2, min(3, w - 3 - k) + 1))
        t1 = int(rng.integers(7, h - 7))
        t2 = int(rng.integers(t1 + 3, h - 3))
        cut = _path(h, w, [k, k + distance, k], [t1, t2])
    else:
        if shift == 'scale' and kind == 'test':
            h, w = int(rng.integers(24, 29)), int(rng.integers(10, 14))
        else:
            # Scale-transfer puzzles keep ALL their non-witness training
            # arenas compact, rather than accidentally teaching the test scale.
            high = 22 if shift == 'scale' and kind == 'train' else 29
            h, w = int(rng.integers(16, high)), int(rng.integers(9, 14))
        if kind == 'test' and shift == 'bends':
            h = max(20, h)
        family = 'stepped' if kind == 'test' and shift == 'bends' else str(rng.choice(['vertical', 'diagonal', 'elbow', 'stepped']))
        steps = int(rng.integers(3, 5)) if kind == 'test' and shift == 'bends' else int(rng.integers(1, 3))
        cut, k = _sample_cut(rng, h, w, family, steps)
        if kind == 'test' and shift == 'combination':
            # Return to an earlier boundary position, now with several entry
            # signals and a fresh obstacle network instead of one isolated
            # victim. This combines the two different training witnesses.
            k = int(rng.integers(3, w - 3))
            new = int(rng.choice([c for c in range(2, w - 2) if c != k]))
            rows = sorted(map(int, rng.choice(np.arange(5, h - 2, 3), 2, replace=False)))
            cut = _path(h, w, [k, new, k], rows)
    regions, sites, _ = _cut_info(cut)
    # In canonical coordinates the component containing top-left is index 0.
    marker = sites[side][int(rng.integers(len(sites[side])))]
    marked = regions[side]
    color_left = first if marked[0, 0] else second
    color_right = second if color_left == first else first
    control = np.full((h, w), bg, dtype=np.int8)
    control[cut] = cut_color
    control[marker] = first
    arena = np.full((h, w), bg, dtype=np.int8)
    if witness == 'seam':
        extent_left = int(rng.integers(1, k - 1))
        extent_right = int(rng.integers(1, w - k - 2))
        left, right = k - extent_left, k + extent_right
        c1 = int(rng.integers(left, k))
        c2 = int(rng.integers(k + 1, right + 1))
        n1, n2 = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        arena[:n1, c1], arena[:n2, c2] = color_left, color_right
        row = int(rng.integers(max(n1, n2) + 1, min(h - 4, max(n1, n2) + 4)))
        arena[row, left:right + 1] = wall
        # Interior singleton hits visibly produce two children in their territory.
        arena[row + 2, left - 1] = wall
        if rng.random() < .65:
            arena[row + 2, right + 1] = wall
        # Two separate descendant hits at one depth share the middle flank.
        arena[row + 4, left - 2] = wall
        arena[row + 4, left] = wall
        inert_col = next(c for c in rng.permutation(w) if c not in (c1, c2))
        arena[1, inert_col] = wall
    elif witness == 'return':
        c1, victim = k - 2, k + 1
        arena[:2, c1], arena[:2, victim] = color_left, color_right
        row = int(rng.integers(3, t1 - 1))
        arena[row, c1 - 1:c1 + 2] = wall
        # An unreached upstream obstacle interrupts the left detour. It remains
        # a blocker rather than becoming a bridge to a disconnected flank.
        arena[row - 1, c1 - 1] = wall
        # The signal on the squeezed side dies, then its old side opens again.
        # A full simulation followed by masking invents a ghost tail and hit here.
        arena[t2 + 2, victim:victim + 3] = wall
    else:
        count = int(rng.integers(2, 4)) if kind == 'test' and shift == 'combination' else int(rng.integers(1, 4))
        stable = np.logical_and.reduce([~cut[r] for r in range(4)])
        left_slots = [c for c in range(k) if stable[c] and all(regions[0][r, c] for r in range(4))]
        right_slots = [c for c in range(k + 1, w) if stable[c] and all(regions[1][r, c] for r in range(4))]
        _require(left_slots and right_slots, "no unambiguous entry slots")
        for slots, color in ((left_slots, color_left), (right_slots, color_right)):
            for c in _stub_columns(rng, slots, count):
                arena[:int(rng.integers(2, 5)), c] = color
        # Compose bars at freely selected depths. Reached and inert status is
        # determined by signals, never by the bar's size or color.
        rows = list(map(int, rng.choice(np.arange(5, h), int(rng.integers(2, min(8, h - 4))), replace=False)))
        for row in rows:
            for _ in range(int(rng.integers(1, 4))):
                length = int(rng.integers(1, w + 1))
                left = int(rng.integers(w - length + 1))
                arena[row, left:left + length] = wall
        if rng.random() < .5:
            c = int(rng.integers(w))
            if arena[1, c] == bg:
                arena[1, c] = wall
    return _pack(_actual(control, direction), _actual(arena, direction), frame)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    fixed_palette = bool(rng.random() < .45)
    direction = int(rng.integers(4))
    mixed_direction = bool(rng.random() < .5)
    first_side = int(rng.integers(2))
    shift = str(rng.choice(['bends', 'scale', 'combination'], p=[.5, .25, .25]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = ('seam' if i == 0 else 'return') if split == 'train' and i < 2 else None
            side = (first_side if i == 0 else 1 - first_side) if split == 'train' and i < 2 else int(rng.integers(2))
            for attempt in range(200):
                local_palette = palette if fixed_palette or (split == 'train' and i == 0) else tuple(map(int, rng.choice(10, 6, replace=False)))
                local_direction = (direction + i + int(split == 'test')) % 4 if mixed_direction else direction
                try:
                    inp = _make_input(rng, local_palette, local_direction, side,
                                      'test' if split == 'test' else 'train', shift, witness)
                    d = _decode(inp)
                    flags, stats = _flags(d)
                except AssertionError:
                    continue
                if _key(inp) in seen or not stats['events']:
                    continue
                if witness == 'seam' and not ({'no_branch', 'unrestricted', 'reverse_marker', 'reverse_priority', 'outline_all', 'one_per_layer'} <= flags and stats['two_flanks'] and stats['merges']):
                    continue
                if witness == 'return' and not ({'postmask', 'jump_detour'} <= flags and stats['inert']):
                    continue
                if split == 'test' and not (
                    len({r for r, *_ in stats['events']}) >= 2
                    and stats['two_flanks'] and {'no_branch', 'unrestricted'} <= flags
                ):
                    continue
                if split == 'test' and shift == 'combination' and not (
                    sum(len(v) for v in d['entries'].values()) >= 3 and 'postmask' in flags
                ):
                    continue
                result, _ = _run(d)
                if np.array_equal(result, d['board']):
                    continue
                puzzle[split].append({'input': inp, 'output': _whole_output(d, result)})
                seen.add(_key(inp))
                break
            else:
                raise RuntimeError(f"bounded construction exhausted for {split}/{i}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, taught, sides = set(), set(), set()
    branching = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "output shape changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            result, stats = _run(d)
            _require(np.array_equal(out, _whole_output(d, result)), "incorrect signal propagation")
            _require(np.all(out[inp != d['bg']] == inp[inp != d['bg']]), "immutable pixel changed")
            _require(np.all(np.isin(out[out != inp], [d['preferred'], d['other']])), "non-signal paint added")
            _require(not np.array_equal(inp, out), "identity example")
            for color in (d['preferred'], d['other']):
                # Newly painted colors must stay in their own side or on seam.
                actual = _canonical(out[d['arena_slice']], d['direction'])
                _require(not np.any((actual == color) & ~d['allowed'][color]), "signal leaked into wrong side")
            _require(np.all((stats['trace_masks'][d['preferred']] & stats['trace_masks'][d['other']]) <= d['seam']),
                     "different signals overlap away from seam")
            if split == 'train':
                flags, _ = _flags(d)
                taught |= flags
                sides.add(d['side'])
                branching |= stats['two_flanks'] > 0
    _require(taught == {'no_branch', 'unrestricted', 'postmask', 'reverse_marker', 'reverse_priority', 'outline_all', 'jump_detour', 'one_per_layer'},
             "training fails to distinguish a specified shortcut")
    _require(sides == {0, 1} and branching, "training needs both marked sides and two-way branching")


def self_test():
    """Bounded 42-seed coverage, local regressions, metamorphisms, and corruption."""
    # Hand-calculated labels anchor the local semantics independently of _run.
    cut = _path(9, 9, [4], [])
    control = np.zeros((9, 9), dtype=np.int8)
    control[cut], control[0, 0] = 2, 4
    arena = np.zeros((9, 9), dtype=np.int8)
    arena[:2, 2], arena[:2, 6] = 4, 5
    arena[3, 2:7], arena[5, 1], arena[7, 0], arena[7, 2] = 3, 3, 3, 3
    expected_rows = ('004000500', '004000500', '044445550', '043333350',
                     '444000050', '434000050', '444400050', '343400050', '040400050')
    expected = np.array([[int(c) for c in row] for row in expected_rows], dtype=np.int8)
    _require(np.array_equal(solve(_pack(control, arena, 1)), _pack(control, expected, 1)),
             "hand branching/priority/merge/edge regression failed")
    cut = _path(11, 9, [4, 6, 4], [4, 7])
    control = np.zeros((11, 9), dtype=np.int8)
    control[cut], control[0, 0] = 2, 4
    arena = np.zeros((11, 9), dtype=np.int8)
    arena[:2, 2], arena[:2, 5], arena[9, 5] = 4, 5, 3
    expected = np.zeros((11, 9), dtype=np.int8)
    expected[:, 2], expected[:5, 5], expected[9, 5] = 4, 5, 3
    _require(np.array_equal(solve(_pack(control, arena, 1)), _pack(control, expected, 1)),
             "a killed signal resurrected after territory returned")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, structures, palettes, directions, bends, counts, lengths = set(), set(), set(), set(), set(), set(), set()
    features = dict(two_flanks=0, rejections=0, overlap=0, inert=0, clipped=0, merges=0)
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        _require(all(_key(p['input']) == _key(q['input']) and _key(p['output']) == _key(q['output'])
                     for s in puzzle for p, q in zip(puzzle[s], generate_puzzle(seed)[s])), "nondeterminism")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                board, stats = _run(d)
                _require(np.array_equal(solve(pair['input']), pair['output']), "input-only solver mismatch")
                shapes.add(pair['input'].shape)
                role_map = {c: i for i, c in enumerate((d['bg'], d['frame'], d['cut_color'], d['obstacle'], d['preferred'], d['other']))}
                norm = np.vectorize(role_map.__getitem__, otypes=[np.int8])(pair['input'])
                structures.add(_key(norm))
                palettes.add(tuple(role_map))
                directions.add(d['direction'])
                bends.add(d['bends'])
                counts.add(tuple(len(d['entries'][c]) for c in (d['preferred'], d['other'])))
                lengths.update(b - a + 1 for _, a, b, *_ in stats['events'])
                for name in features:
                    features[name] += bool(stats[name])
        # Reflect and quarter-turn BOTH interiors, preserving their ordered roles.
        pair = puzzle['train'][0]
        d = _decode(pair['input'])
        answer_board = pair['output'][d['arena_slice']]
        for transform in (lambda x: np.fliplr(x), lambda x: np.rot90(x)):
            inp = _pack(transform(d['control']), transform(d['arena']), d['frame'])
            expected = _pack(transform(d['control']), transform(answer_board), d['frame'])
            _require(np.array_equal(solve(inp), expected), "geometric equivariance failed")
        # A bijective palette change cannot alter routing.
        mapping = np.random.default_rng(seed + 101).permutation(10)
        _require(np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']]), "palette equivariance failed")
    _require(len(shapes) >= 30 and len(structures) >= 150, "structural variation collapsed")
    _require(len(palettes) >= 30 and directions == set(range(4)), "visible roles collapsed")
    _require({0, 4, 6, 8} <= bends and len(counts) >= 4 and len(lengths) >= 8, "geometric variation collapsed")
    _require(all(n > 0 for n in features.values()), "missing family interaction")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(73, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['train'][0]['output'][0, 0] = (int(bad['train'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt frame accepted")
    # Persist only compact descriptive measurements, not solver state.
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in p),
                grid_shapes=len(shapes), normalized_structures=len(structures),
                palettes=len(palettes), directions=sorted(directions), bends=sorted(bends),
                entry_counts=sorted(counts), reached_lengths=sorted(lengths), features=features)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
