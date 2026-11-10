"""Parity printer: routed binary-tile exchange, with input-only solving.

No files, parent modules, network access, latent conventions, or saved examples
are needed at runtime. Tests and diagnostics retain no execution traces.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np

_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 185316391, 854824724, 587358551, 900609164, 379122890,
              999547895, 363674686, 326571401, 292948694, 604497837,
              948298861, 290716463, 95726481, 135904245, 416733772,
              148102269, 109062216, 949494054, 825912581, 775134636,
              357028625, 147027313, 848057239, 63241959, 207603469,
              447733798, 224759542, 673009108, 366029058, 369733714,
              317974103, 270246769, 930319347, 566293846, 311348233,
              655471360, 836286180, 708181052, 54563344, 929107327,
              454573659)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape),
             "grids must be two dimensional and bounded by 30")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _rectangles(board, colors):
    """Read same-color components as rectangles; repeated colors are allowed."""
    h, w = board.shape
    seen = set()
    rectangles = []
    for r in range(h):
        for c in range(w):
            color = int(board[r, c])
            if color not in colors or (r, c) in seen:
                continue
            todo = [(r, c)]
            seen.add((r, c))
            cells = []
            while todo:
                y, x = todo.pop()
                cells.append((y, x))
                for dy, dx in _DIRECTIONS:
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and
                            (yy, xx) not in seen and board[yy, xx] == color):
                        seen.add((yy, xx))
                        todo.append((yy, xx))
            ys, xs = zip(*cells)
            top, bottom, left, right = min(ys), max(ys) + 1, min(xs), max(xs) + 1
            _require(len(cells) == (bottom - top) * (right - left),
                     "each blocker component must be a solid rectangle")
            rectangles.append((top, bottom, left, right, color))
    return rectangles


def _trace(board, source, turns, *, mapping=None, straight=False, one_turn=False):
    """Trace four independent simple rays on the macro lattice.

    A collision changes heading without entering its obstacle. A corner is
    visited once. The array read for collisions is always the original board.
    Each branch has a distinct coverage mask, so overlap order cannot matter.
    """
    h, w = board.shape
    masks = np.zeros((4, h, w), dtype=bool)
    axes = np.zeros((4, 2, h, w), dtype=bool)
    headings = np.full((4, h, w), -1, dtype=np.int8)
    events, chains = [], []
    for branch in range(4):
        r, c = source
        direction = branch
        path = {(r, c)}
        states = set()
        chain = []
        just_turned = False
        while True:
            state = (r, c, direction)
            _require(state not in states, "cyclic ray")
            states.add(state)
            masks[branch, r, c] = True
            axes[branch, direction % 2, r, c] = True
            headings[branch, r, c] = direction
            dy, dx = _DIRECTIONS[direction]
            rr, cc = r + dy, c + dx
            if not (0 <= rr < h and 0 <= cc < w):
                break
            value = int(board[rr, cc])
            if value in turns:
                if straight or (one_turn and chain):
                    break
                _require(not just_turned, "turning corner has no clear outgoing cell")
                turn = turns[value]
                if mapping is not None:
                    turn = mapping[turn]
                events.append((branch, direction, value, r, c))
                chain.append((direction, value, r, c))
                direction = (direction + turn) % 4
                just_turned = True
            else:
                _require((rr, cc) not in path, "a ray revisits its own path or source")
                r, c = rr, cc
                path.add((r, c))
                just_turned = False
        chains.append(tuple(chain))
    masks[:, source[0], source[1]] = False
    axes[:, :, source[0], source[1]] = False
    return dict(masks=masks, axes=axes, headings=headings, counts=masks.sum(axis=0),
                events=tuple(events), chains=tuple(chains))


def _decode(input_grid):
    grid = _grid(input_grid)
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "missing top key frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom key frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(grid) - 1, "invalid key height")
    band = grid[1:bottom]
    _require(np.all(band == band[0]), "key bars must span their interiors")
    row = band[0]
    _require(row[0] == frame and row[-1] == frame, "missing key walls")
    runs = []
    c = 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        runs.append((start, c))
    _require(2 <= len(runs) <= 5 and all(b - a >= 3 for a, b in runs),
             "need 2..5 visibly separated keys")
    background = int(row[runs[0][0] + 1])
    _require(background != frame, "background and frame differ")
    turns = {}
    widths = []
    for a, b in runs:
        _require(np.all(row[a + 1:b - 1] == background), "inconsistent key ground")
        lv, rv = int(row[a]), int(row[b - 1])
        _require((lv == background) != (rv == background), "ambiguous side bar")
        color, turn = (rv, 1) if lv == background else (lv, -1)
        _require(color not in (background, frame) and color not in turns,
                 "each blocker color needs one unique key")
        turns[color] = turn
        widths.append(b - a + 2)
    field = grid[bottom + 1:].copy()
    _require(not np.any(field == frame), "frame is reserved for the key band")
    tile_colors = set(map(int, np.unique(field))) - set(turns) - {background}
    _require(len(tile_colors) == 2, "the source needs two unkeyed content colors")
    points = np.argwhere(np.isin(field, list(tile_colors)))
    top, left = map(int, points.min(axis=0))
    last_r, last_c = map(int, points.max(axis=0))
    n = last_r - top + 1
    _require(2 <= n <= 4 and last_c - left + 1 == n and len(points) == n * n,
             "source must be one complete binary square, size 2..4")
    _require(top % n == 0 and left % n == 0 and
             all(x % n == 0 for x in field.shape), "source and field must register to the tile lattice")
    tile = field[top:top + n, left:left + n].copy()
    source = (top // n, left // n)
    h, w = field.shape[0] // n, field.shape[1] // n
    _require(min(h, w) >= 5, "the macro field must be at least 5 by 5")
    board = np.empty((h, w), dtype=np.int8)
    for r in range(h):
        for c in range(w):
            block = field[r * n:(r + 1) * n, c * n:(c + 1) * n]
            if (r, c) == source:
                board[r, c] = min(tile_colors)
            else:
                value = int(block[0, 0])
                _require(np.all(block == value) and (value == background or value in turns),
                         "non-source macro cells must be blank or solid blockers")
                board[r, c] = value
    rectangles = _rectangles(board, turns)
    _require(rectangles, "at least one blocker is needed")
    return dict(grid=grid, field=field, board=board, source=source, tile=tile,
                n=n, turns=turns, background=background, frame=frame,
                header_height=bottom + 1, key_widths=tuple(widths), rectangles=rectangles)


def _print(info, trace, mode="parity"):
    out = info["field"].copy()
    tile, n = info["tile"], info["n"]
    a, b = map(int, np.unique(tile))
    exchanged = a + b - tile.astype(int)
    for r, c in np.argwhere(trace["counts"] > 0):
        count = int(trace["counts"][r, c])
        if mode == "union":
            stamp = exchanged
        elif mode == "original":
            stamp = tile
        elif mode == "erase_even" and count % 2 == 0:
            stamp = np.full(tile.shape, info["background"], dtype=int)
        elif mode == "rotate":
            branch = int(np.flatnonzero(trace["masks"][:, r, c])[0])
            direction = int(trace["headings"][branch, r, c])
            stamp = np.rot90(exchanged if count % 2 else tile, -direction)
        elif mode == "solid":
            stamp = np.full(tile.shape, a, dtype=int)
        else:
            stamp = exchanged if count % 2 else tile
        out[r * n:(r + 1) * n, c * n:(c + 1) * n] = stamp
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell using only this input's visible geometry/key."""
    info = _decode(input_grid)
    return _print(info, _trace(info["board"], info["source"], info["turns"]))


def _route_features(board, source, turns, trace, background=0):
    masks, axes, counts = trace["masks"], trace["axes"], trace["counts"]
    crossing, shared_run = False, False
    for i in range(4):
        for j in range(i):
            crossing |= bool(np.any(axes[i, 0] & axes[j, 1]) or
                             np.any(axes[i, 1] & axes[j, 0]))
            for axis in range(2):
                shared = axes[i, axis] & axes[j, axis]
                shared_run |= bool(np.any(shared[:-1] & shared[1:]) if axis == 0 else
                                   np.any(shared[:, :-1] & shared[:, 1:]))
    headings = {side: {d for _, d, v, _, _ in trace["events"] if turns[v] == side}
                for side in (-1, 1)}
    sr, sc = source
    off_axis = any(r != sr and c != sc for _, _, _, r, c in trace["events"])
    return dict(max_multiplicity=int(counts.max()), odd=bool(np.any(counts % 2)),
                even=bool(np.any((counts > 0) & (counts % 2 == 0))),
                untouched=bool(np.any((counts == 0) & (board == background))),
                crossing=crossing, shared_run=shared_run, headings=headings,
                max_turns=max(map(len, trace["chains"])), off_axis=off_axis)


def _sample_scene(rng, n, density):
    header = int(rng.integers(3, 7))
    max_h = (30 - header) // n
    h = int(rng.integers(5, max_h + 1))
    w = int(rng.integers(max(5, (10 + n - 1) // n), 30 // n + 1))
    key_count = int(rng.integers(2, min(5, n * w // 5) + 1))
    turns = {color: int(rng.choice((-1, 1))) for color in range(2, key_count + 2)}
    source = (int(rng.integers(h)), int(rng.integers(w)))
    board = np.zeros((h, w), dtype=np.int8)
    board[source] = 1
    target = int(rng.integers(2 if density else 1, 11 if density else 8))
    for _ in range(target):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        # Separate choices of transverse/long extent include bars, squares,
        # compact blocks, boundary pieces and larger rectangles.
        if rng.random() < .3:
            hh, ww = 1, int(rng.integers(1, w + 1))
        elif rng.random() < .4:
            hh, ww = int(rng.integers(1, h + 1)), 1
        else:
            hh, ww = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        end_r, end_c = min(h, r + hh), min(w, c + ww)
        color = int(rng.choice(list(turns)))
        if np.any(board[r:end_r, c:end_c] != 0):
            continue
        # Keep components with the same color spatially distinguishable.
        if np.any(board[max(0, r - 1):min(h, end_r + 1),
                        max(0, c - 1):min(w, end_c + 1)] == color):
            continue
        board[r:end_r, c:end_c] = color
    return board, source, turns, header


def _tile(rng, n, density):
    for _ in range(64):
        bits = rng.random((n, n)) < density
        if bits.any() and not bits.all() and all(
                not np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            return bits
    bits = np.zeros((n, n), dtype=bool)
    bits[0, 0] = True
    return bits


def _render(rng, board, source, turns, header_height, n, tile_density):
    colors = rng.permutation(10)
    background, frame, ca, cb = map(int, colors[:4])
    key_colors = {v: int(colors[4 + i]) for i, v in enumerate(turns)}
    palette = {0: background, 1: ca, **key_colors}
    coarse = np.array([[palette[int(v)] for v in row] for row in board], dtype=np.int8)
    field = np.repeat(np.repeat(coarse, n, axis=0), n, axis=1)
    tile = np.where(_tile(rng, n, tile_density), ca, cb).astype(np.int8)
    sr, sc = source
    field[sr * n:(sr + 1) * n, sc * n:(sc + 1) * n] = tile
    width = field.shape[1]
    count = len(turns)
    widths = np.full(count, 5, dtype=int)
    if width > 5 * count:
        widths += rng.multinomial(width - 5 * count, np.full(count, 1 / count))
    header = np.full((header_height, width), background, dtype=np.int8)
    header[[0, -1]] = frame
    col = 0
    for v, size in zip(rng.permutation(list(turns)), widths):
        header[:, col] = frame
        header[:, col + size - 1] = frame
        bar = col + 1 if turns[int(v)] == -1 else col + size - 2
        header[1:-1, bar] = key_colors[int(v)]
        col += int(size)
    inp = np.vstack((header, field))
    actual_turns = {key_colors[v]: t for v, t in turns.items()}
    actual_board = np.array([[palette[int(v)] for v in row] for row in board], dtype=np.int8)
    info = dict(field=field, tile=tile, n=n, background=background)
    trace = _trace(actual_board, source, actual_turns)
    return {"input": inp, "output": _print(info, trace)}


def _headings_union(a, b):
    return {side: set(a.get(side, ())) | set(b.get(side, ())) for side in (-1, 1)}


def _make_pair(rng, n, params, *, test=False, lesson=0, previous_headings=None):
    # Trials are transient tiny arrays; no candidate bank or retry logs persist.
    for _ in range(5000):
        board, source, turns, header = _sample_scene(rng, n, params["density"])
        if not np.any(board > 1):
            continue
        try:
            trace = _trace(board, source, turns)
        except AssertionError:
            continue
        f = _route_features(board, source, turns, trace)
        if not (f["odd"] and f["untouched"]):
            continue
        if test:
            if f["max_multiplicity"] < 3:
                continue
        elif f["max_multiplicity"] > 2:
            continue
        if lesson:
            if not (f["even"] and f["off_axis"] and f["max_turns"] >= 2 and
                    all(f["headings"].values())):
                continue
            if lesson == 1 and not f["shared_run"]:
                continue
            if lesson == 2:
                combined = _headings_union(previous_headings or {}, f["headings"])
                if not f["crossing"] or min(map(len, combined.values())) < 2:
                    continue
        pair = _render(rng, board, source, turns, header, n, params["tile_density"])
        return pair, f["headings"]
    raise RuntimeError("bounded scene search exhausted; no silent fixed-layout fallback")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    params = dict(density=bool(rng.integers(2)), tile_density=float(rng.uniform(.25, .75)))
    sizes = rng.choice((2, 3, 4), size=2, replace=False)
    puzzle = {"train": [], "test": []}
    seen = set()
    headings = {}
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            n = int(sizes[i]) if split == "train" and i < 2 else int(rng.choice((2, 3, 4)))
            lesson = i + 1 if split == "train" and i < 2 else 0
            for _ in range(8):
                pair, local_heads = _make_pair(rng, n, params, test=split == "test",
                                               lesson=lesson, previous_headings=headings)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
            headings = _headings_union(headings, local_heads)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(infos, outputs):
    headings = {}
    features = []
    modes = ("union", "original", "erase_even", "rotate", "solid")
    rejected = {name: False for name in modes}
    route_rejected = {name: False for name in ("straight", "one_turn")}
    fits = {(a, b): True for a in (-1, 1) for b in (-1, 1)}
    for info, out in zip(infos, outputs):
        trace = _trace(info["board"], info["source"], info["turns"])
        f = _route_features(info["board"], info["source"], info["turns"], trace,
                            info["background"])
        features.append(f)
        headings = _headings_union(headings, f["headings"])
        for mode in modes:
            rejected[mode] |= not np.array_equal(_print(info, trace, mode), out)
        for mode in route_rejected:
            wrong = _trace(info["board"], info["source"], info["turns"],
                           straight=mode == "straight", one_turn=mode == "one_turn")
            route_rejected[mode] |= not np.array_equal(_print(info, wrong), out)
        for a, b in fits:
            if not fits[(a, b)]:
                continue
            try:
                wrong = _trace(info["board"], info["source"], info["turns"], mapping={-1: a, 1: b})
                fits[(a, b)] &= np.array_equal(_print(info, wrong), out)
            except AssertionError:
                fits[(a, b)] = False
    _require(min(map(len, headings.values())) >= 2, "both relative turns need multiple headings")
    _require(all(rejected.values()), "training leaves a printing shortcut unchallenged")
    _require(all(route_rejected.values()), "training leaves a routing shortcut unchallenged")
    _require([key for key, value in fits.items() if value] == [(-1, 1)],
             "training must uniquely reveal both key meanings")
    _require(any(f["shared_run"] for f in features) and any(f["crossing"] for f in features),
             "training must show both crossings and shared runs")
    _require(any(f["off_axis"] and f["max_turns"] >= 2 for f in features),
             "recursive off-axis routing must be demonstrated")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen = set()
    training, training_outputs = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _key(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _decode(inp)
            trace = _trace(info["board"], info["source"], info["turns"])
            expected = _print(info, trace)
            _require(np.array_equal(out, expected), "output differs from routed parity printing")
            n, sr, sc = info["n"], *info["source"]
            _require(np.array_equal(out[sr*n:(sr+1)*n, sc*n:(sc+1)*n], info["tile"]),
                     "emitter changed")
            # Direct block invariants check preservation and both content roles.
            for r in range(info["board"].shape[0]):
                for c in range(info["board"].shape[1]):
                    block = out[r*n:(r+1)*n, c*n:(c+1)*n]
                    count = int(trace["counts"][r, c])
                    if not count:
                        _require(np.array_equal(block, info["field"][r*n:(r+1)*n, c*n:(c+1)*n]),
                                 "unvisited cell or blocker changed")
                    else:
                        _require(set(map(int, np.unique(block))) == set(map(int, np.unique(info["tile"]))),
                                 "printed blocks must retain both tile colors")
            _require(np.any(trace["counts"] % 2), "no odd witness")
            blank = info["board"] == info["background"]
            _require(np.any(blank & (trace["counts"] == 0)), "no transparency witness")
            if split == "train":
                _require(trace["counts"].max() <= 2, "training multiplicity exceeds its curriculum")
                training.append(info)
                training_outputs.append(out)
            else:
                _require(trace["counts"].max() >= 3, "test must generalize overlap multiplicity")
    _require(len({info["n"] for info in training}) >= 2, "training needs scale variation")
    _evidence(training, training_outputs)


def pair_features(pair):
    """Compact review measurements, reconstructed from pixels only."""
    info = _decode(pair["input"])
    trace = _trace(info["board"], info["source"], info["turns"])
    counts = trace["counts"]
    rects = info["rectangles"]
    hits = {v for _, _, v, _, _ in trace["events"]}
    return dict(tile_size=info["n"], macro_shape=info["board"].shape,
                output_shape=info["field"].shape, source=info["source"],
                key_count=len(info["turns"]), blocker_count=len(rects),
                blocker_shapes=tuple((b-a, d-c) for a, b, c, d, _ in rects),
                unused_keys=len(set(info["turns"]) - {v for *_, v in rects}),
                repeated_colors=len(rects) > len({v for *_, v in rects}),
                unhit_colors=len({v for *_, v in rects} - hits),
                header_height=info["header_height"], key_widths=info["key_widths"],
                palette_size=len(np.unique(info["grid"])),
                background=info["background"], frame=info["frame"],
                max_multiplicity=int(counts.max()),
                multiplicity_cells=tuple(int(np.count_nonzero(counts == k)) for k in range(1, 5)),
                branch_turns=tuple(map(len, trace["chains"])),
                layout_hash=hashlib.sha256(repr((info["board"].shape, info["source"],
                    tuple(x[:4] for x in rects), tuple(info["turns"].values()))).encode()).hexdigest())


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                shape, data = _key(np.asarray(pair[name]))
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def _fixture_checks():
    """Small hand-traced regression cases, never generation templates.

    At (2,4), north hits a left-turn stop immediately and joins west.
    Both go west to column 1 then north to the boundary. East goes to
    column 5 and turns south. A right-turn stop below the source makes
    south join the westward pair; removing that stop frees south instead.
    """
    rows = ("0000000", "2003333", "2000102", "0030202", "0030000", "0000000")
    board = np.array([[int(v) for v in row] for row in rows], dtype=np.int8)
    triple = ("0300000", "0300000", "0333010", "0000010", "0000010", "0000010")
    double = ("0200000", "0200000", "0222010", "0000110", "0000110", "0000110")
    for remove, text in ((False, triple), (True, double)):
        scene = board.copy()
        if remove:
            scene[3, 4] = 0
        expected_counts = np.array([[int(v) for v in row] for row in text], dtype=int)
        traced = _trace(scene, (2, 4), {2: 1, 3: -1})
        _require(np.array_equal(traced["counts"], expected_counts), "hand-traced multiplicity differs")
        for n in (2, 3, 4):
            pair = _render(np.random.default_rng(730 + n), scene, (2, 4),
                           {2: 1, 3: -1}, 3, n, .5)
            info = _decode(pair["input"])
            expected = info["field"].copy()
            colors = sorted(map(int, np.unique(info["tile"])))
            # Cell-by-cell fixture prediction uses only the manually written
            # counts, not the tracer, branch masks, or the printing helper.
            for r in range(expected.shape[0]):
                for c in range(expected.shape[1]):
                    m = int(expected_counts[r // n, c // n])
                    if m:
                        value = int(info["tile"][r % n, c % n])
                        expected[r, c] = colors[1 - colors.index(value)] if m % 2 else value
            _require(np.array_equal(solve(pair["input"]), expected), "fixture pixel prediction differs")
            # A missing tile pixel may not be silently completed.
            damaged = pair["input"].copy()
            damaged[info["header_height"] + 2*n, 4*n] = info["background"]
            try:
                _decode(damaged)
            except AssertionError:
                pass
            else:
                raise AssertionError("incomplete source square accepted")
    # Two adjacent stops at a corner have no defined outgoing free step.
    corner = np.zeros((7, 7), dtype=np.int8)
    corner[3, 3] = 1
    corner[2, 3] = corner[3, 4] = 2
    # This four-stop route returns to the source, violating simple paths.
    loop = np.zeros((7, 7), dtype=np.int8)
    loop[3, 3] = 1
    for r, c in ((3, 6), (6, 5), (5, 2), (2, 3)):
        loop[r, c] = 2
    for scene in (corner, loop):
        try:
            _trace(scene, (3, 3), {2: 1})
        except AssertionError:
            pass
        else:
            raise AssertionError("undefined corner/self-revisiting route accepted")


def self_test():
    """42 fixed seeds; transient pairs and small aggregate sets, no disk writes."""
    _fixture_checks()
    seen, sizes, shapes, layouts, counts, keys, heights = (set() for _ in range(7))
    palettes, test_multiplicities = set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        seen.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = pair_features(pair)
                sizes.add(f["tile_size"])
                shapes.add(tuple(f["output_shape"]))
                layouts.add(f["layout_hash"])
                counts.add(f["blocker_count"])
                keys.add(f["key_count"])
                heights.add(f["header_height"])
                palettes.add(f["palette_size"])
                if split == "test":
                    test_multiplicities.add(f["max_multiplicity"])
                _require(np.array_equal(solve(np.asarray(pair["input"]).tolist()), pair["output"]),
                         "detached input-only solve failed")
    _require(len(seen) == 42 and sizes == {2, 3, 4}, "insufficient seed/scale variation")
    _require(len(shapes) >= 30 and len(layouts) >= 180, "insufficient structural variation")
    _require(len(counts) >= 6 and len(keys) >= 3 and len(heights) >= 3,
             "insufficient object/key/header variation")
    _require(len(palettes) >= 3 and 3 in test_multiplicities, "insufficient palette/overlap variation")
    p = generate_puzzle(17, 2, 1)
    _require(_fingerprint(p) == _fingerprint(generate_puzzle(17, 2, 1)), "nondeterministic generation")
    # Semantic color renaming must commute with the transformation.
    permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0])
    for pair in p["train"] + p["test"]:
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "solver depends on literal colors")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(1000 + 10*nt + nq, nt, nq)
            _require(len(q["train"]) == nt and len(q["test"]) == nq, "wrong explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{name: grid.copy() for name, grid in pair.items()} for pair in p[s]]
               for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged label accepted")
    return {"seeds": 42, "distinct_puzzles": len(seen), "output_shapes": len(shapes),
            "layouts": len(layouts), "tile_sizes": sorted(sizes),
            "blocker_counts": sorted(counts), "key_counts": sorted(keys),
            "header_heights": sorted(heights), "palette_sizes": sorted(palettes),
            "test_max_multiplicities": sorted(test_multiplicities)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
