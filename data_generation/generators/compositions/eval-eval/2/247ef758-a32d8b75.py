"""Coordinate-addressed, exchanged binary punches with parity cancellation.

Only standard-library Python and NumPy are used. No parent generator, seed or
training labels participate in solve(). Card geometry and palette roles are
fully visible. Generation, solve and validation use different painting paths.
"""
from __future__ import annotations

import numpy as np


_NEIGHBORS = ((0, 1), (1, 2), (2, 1), (1, 0))
_SHORTCUTS = ("regular_spacing", "all_marks", "a_only", "no_exchange", "fill_holes",
              "rotate_tile", "no_turn", "reverse_turn", "top_left", "union",
              "single_hit", "paint_ink")


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if low is not None and not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape), "grid bounds"
    assert np.issubdtype(a.dtype, np.integer), "integer colors required"
    assert np.all((0 <= a) & (a <= 9)), "color bounds"
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.uint8).tobytes()


def _fields(panel, paper, frame):
    """Decode complete, bounded fields using blank gaps, not fixed slices."""
    assert np.all(panel[0] == paper) and np.all(panel[-1] == paper)
    used = np.any(panel != paper, axis=1)
    bands = []
    r = 0
    while r < len(panel):
        if not used[r]:
            r += 1
            continue
        top = r
        while r < len(panel) and used[r]:
            r += 1
        cols = np.flatnonzero(np.any(panel[top:r] != paper, axis=0))
        left, right = int(cols[0]), int(cols[-1]) + 1
        box = panel[top:r, left:right]
        assert min(box.shape) >= 3
        assert np.all(box[0] == frame) and np.all(box[-1] == frame)
        assert np.all(box[:, 0] == frame) and np.all(box[:, -1] == frame)
        assert np.all(panel[top:r, :left] == paper)
        assert np.all(panel[top:r, right:] == paper)
        bands.append((box[1:-1, 1:-1].copy(), (top + 1, left + 1)))
    assert len(bands) == 3, "tile, stencil, direction fields required"
    return bands


def _decode_side(inp, d, side):
    if side == "left":
        panel, board, origin = inp[:, :d], inp[:, d + 1:], d + 1
    else:
        board, panel, origin = inp[:, :d], inp[:, d + 1:], 0
    assert panel.shape[1] >= 5 and board.shape[1] >= 7
    paper, separator = int(panel[0, 0]), int(inp[0, d])
    frame = int(board[0, 0])
    assert len({paper, separator, frame}) == 3
    assert all(int(board[r, c]) == frame for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    fields = _fields(panel, paper, frame)
    tile, stencil, direction = (x[0] for x in fields)
    assert tile.shape[0] == tile.shape[1] and len(tile) in (3, 5)
    values = set(map(int, np.unique(tile)))
    assert len(values) == 2 and paper in values, "complete binary tile"
    ink = next(iter(values - {paper}))
    sv = set(map(int, np.unique(stencil))) - {paper}
    assert len(sv) == 1
    signal = next(iter(sv))
    assert len({paper, ink, frame, separator, signal}) == 5
    assert all(2 <= x <= 6 for x in stencil.shape)
    mask = stencil == signal
    assert mask[0].any() and mask[-1].any()
    assert mask[:, 0].any() and mask[:, -1].any(), "tight stencil support"
    assert direction.shape == (3, 3)
    points = set(map(tuple, np.argwhere(direction != paper)))
    turns = [k for k, p in enumerate(_NEIGHBORS) if points == {(1, 1), p}]
    assert len(turns) == 1 and np.all(direction[direction != paper] == signal)
    turn = turns[0]
    assert np.array_equal(board[0], board[-1])
    assert np.array_equal(board[:, 0], board[:, -1])
    edge = np.concatenate((board[0], board[-1], board[:, 0], board[:, -1]))
    assert set(map(int, np.unique(edge))) <= {frame, signal, ink}
    assert set(map(int, np.unique(board[1:-1, 1:-1]))) <= {paper, ink}
    rows = np.flatnonzero(board[1:-1, 0] == signal) + 1
    cols = np.flatnonzero(board[0, 1:-1] == signal) + 1
    expected_shape = mask.shape[::-1] if turn % 2 else mask.shape
    assert (len(rows), len(cols)) == expected_shape, "station/stencil rank mismatch"
    radius = len(tile) // 2
    assert np.all(rows - radius >= 1) and np.all(rows + radius < len(board) - 1)
    assert np.all(cols - radius >= 1) and np.all(cols + radius < board.shape[1] - 1)
    return dict(board=board.copy(), panel=panel.copy(), origin=origin, divider=d,
                side=side, paper=paper, ink=ink, frame=frame, separator=separator,
                signal=signal, tile=tile, stencil=mask, turn=turn,
                rows=rows, cols=cols, field_origins=tuple(x[1] for x in fields))


def _decode(input_grid):
    inp = _grid(input_grid)
    assert min(inp.shape) >= 7
    candidates = []
    for d in range(5, inp.shape[1] - 5):
        if not np.all(inp[:, d] == inp[0, d]):
            continue
        for side in ("left", "right"):
            try:
                candidates.append(_decode_side(inp, d, side))
            except (AssertionError, IndexError, ValueError):
                continue
    assert len(candidates) == 1, "input must have one readable card and board"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible input; XOR translated punches on the original board."""
    p = _decode(input_grid)
    mask = np.rot90(p["stencil"], -p["turn"])
    punch = p["tile"] == p["paper"]  # ink after exchanging both roles
    n, parity = len(punch), np.zeros(p["board"].shape, dtype=bool)
    rad = n // 2
    for i, j in np.argwhere(mask):
        r, c = int(p["rows"][i]), int(p["cols"][j])
        parity[r-rad:r+rad+1, c-rad:c+rad+1] ^= punch
    out = p["board"].copy()
    out[parity] = p["paper"] + p["ink"] - out[parity]
    return out


def _inverse_counts(p, behavior="normal"):
    """Diagnostic/validator: query each canvas cell from each station address.

    Rotation is recovered by inverse original-stencil indices. This does not
    call solve, rotate a mask, or use the generation painting routine.
    """
    board, tile, source = p["board"], p["tile"], p["stencil"]
    k, rows, cols = p["turn"], p["rows"], p["cols"]
    if behavior == "no_turn":
        k = 0
    elif behavior == "reverse_turn":
        k = (-k) % 4
    sh, sw = source.shape
    rh, rw = (sw, sh) if k % 2 else (sh, sw)
    n, rad = len(tile), len(tile) // 2
    if behavior == "regular_spacing":
        rows = np.arange(len(rows)) * n + rows[0]
        cols = np.arange(len(cols)) * n + cols[0]
    elif behavior == "all_marks":
        rows = np.flatnonzero(board[1:-1, 0] != p["frame"]) + 1
        cols = np.flatnonzero(board[0, 1:-1] != p["frame"]) + 1
    yy, xx = np.indices(board.shape)
    counts = np.zeros(board.shape, dtype=np.int16)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if i >= rh or j >= rw:
                continue  # malformed rival dimensions do not add fake stations
            if k == 0:
                sy, sx = i, j
            elif k == 1:
                sy, sx = sh - 1 - j, i
            elif k == 2:
                sy, sx = sh - 1 - i, sw - 1 - j
            else:
                sy, sx = j, sw - 1 - i
            if behavior not in ("fill_holes", "a_only") and not source[sy, sx]:
                continue
            ty, tx = yy - r + rad, xx - c + rad
            if behavior == "top_left":
                ty, tx = yy - r, xx - c
            inside = ((ty >= 0) & (ty < n) & (tx >= 0) & (tx < n)
                      & (yy >= 1) & (yy < len(board)-1)
                      & (xx >= 1) & (xx < board.shape[1]-1))
            ty, tx = np.clip(ty, 0, n-1), np.clip(tx, 0, n-1)
            if behavior == "rotate_tile":
                for _ in range(p["turn"]):
                    ty, tx = n-1-tx, ty
            hit = tile[ty, tx] == (p["ink"] if behavior in ("no_exchange", "a_only") else p["paper"])
            counts += inside & hit
    return counts


def _diagnostic_prediction(p, behavior="normal"):
    counts = _inverse_counts(p, behavior)
    out = p["board"].copy()
    active = counts > 0 if behavior == "union" else counts == 1 if behavior == "single_hit" else counts % 2 == 1
    if behavior == "paint_ink":
        out[active] = p["ink"]
    else:
        out[active] = p["paper"] + p["ink"] - out[active]
    return out


def _mask(rng, h, w, witness):
    yy, xx = np.indices((h, w))
    style = int(rng.integers(6))
    if witness or style == 0:
        m = rng.random((h, w)) < rng.uniform(.28, .82)
    elif style == 1:
        m = (yy == 0) | (yy == h-1) | (xx == 0) | (xx == w-1)
    elif style == 2:
        m = np.ones((h, w), bool)
    elif style == 3:
        m = (yy + xx) % int(rng.integers(2, 4)) == 0
    elif style == 4:
        m = (yy == int(rng.integers(h))) | (xx == int(rng.integers(w)))
    else:
        m = np.zeros((h, w), bool)
        c = int(rng.integers(w))
        for r in range(h):
            nc = int(rng.integers(w))
            m[r, min(c, nc):max(c, nc)+1] = True
            c = nc
    # Four independently chosen extrema determine a tight support.
    m[0, int(rng.integers(w))] = True
    m[-1, int(rng.integers(w))] = True
    m[int(rng.integers(h)), 0] = True
    m[int(rng.integers(h)), -1] = True
    return m


def _tile_bits(rng, n, witness):
    for _ in range(40):
        bits = rng.random((n, n)) < rng.uniform(.23, .77)
        if bits.any() and not bits.all():
            if not witness or all(not np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
                return bits
    bits = np.zeros((n, n), bool)
    bits[0, 1] = True
    return bits


def _stations(rng, length, n, count, witness):
    low, high = 1+n//2, length-2-n//2
    span = high-low
    assert span >= count-1
    if witness and count >= 3:
        # Teach both overlap and unequal spacing, leaving station origin free.
        for _ in range(100):
            gaps = rng.integers(1, n+2, size=count-1)
            if gaps.sum() <= span and gaps.min() < n <= gaps.max():
                break
        else:
            gaps = np.ones(count-1, int)
            gaps[-1] = n
        assert gaps.sum() <= span
        start = int(rng.integers(low, high-int(gaps.sum())+1))
        return np.r_[start, start+np.cumsum(gaps)].astype(int)
    if witness:
        gap = int(rng.integers(1, min(n-1, span)+1))
        start = int(rng.integers(low, high-gap+1))
        return np.array([start, start+gap], int)
    mode = int(rng.integers(3))
    if mode == 0:
        # A contiguous subset of a randomly located short interval.
        width = min(span+1, count+int(rng.integers(0, n+2)))
        start = int(rng.integers(low, high-width+2))
        positions = rng.choice(np.arange(start, start+width), count, replace=False)
    else:
        positions = rng.choice(np.arange(low, high+1), count, replace=False)
    return np.sort(positions).astype(int)


def _canvas(rng, h, w, paper, ink, kind):
    yy, xx = np.indices((h, w))
    if kind == "paper":
        bits = np.zeros((h, w), bool)
    else:
        mode = int(rng.integers(5))
        if mode == 0:
            bits = rng.random((h, w)) < rng.uniform(.1, .55)
        elif mode == 1:
            bits = (yy // int(rng.integers(2, 7))) % 2 == 1
        elif mode == 2:
            bits = (xx // int(rng.integers(2, 6))) % 2 == 1
        elif mode == 3:
            bits = (yy + int(rng.integers(1, 4))*xx) // int(rng.integers(2, 7)) % 2 == 1
        else:
            bits = np.zeros((h, w), bool)
            for _ in range(int(rng.integers(1, 5))):
                r, c = int(rng.integers(h)), int(rng.integers(w))
                bits[r:r+int(rng.integers(2, h+1)), c:c+int(rng.integers(2, w+1))] = True
    return np.where(bits, ink, paper).astype(int)


def _assemble(rng, tile, target, turn, rows, cols, canvas, palette, pwidth, side, inert, early_inert=False):
    paper, ink, frame, separator, signal = map(int, palette)
    source = np.rot90(target, turn)  # inverse of the requested clockwise turn
    h, iw = canvas.shape
    board = np.full((h, iw+2), frame, int)
    board[1:-1, 1:-1] = canvas[1:-1]  # canvas has full height; margins are frame
    board[rows, 0] = board[rows, -1] = signal
    board[0, cols] = board[-1, cols] = signal
    for axis, centers in ((0, rows), (1, cols)):
        length = h if axis == 0 else iw+2
        free = np.array([v for v in range(1, length-1) if v not in centers], int)
        count = min(inert, len(free))
        selected = list(map(int, rng.choice(free, count, replace=False)))
        if early_inert and count:
            # Inserting an inert mark before the first station changes ranks
            # for a color-blind decoder, rather than only adding unused tails.
            selected[0] = int(rng.choice(free[free < min(centers)]))
            selected = list(dict.fromkeys(selected))
        for v in selected:
            if axis == 0:
                board[v, 0] = board[v, -1] = ink
            else:
                board[0, v] = board[-1, v] = ink
    direction = np.full((3, 3), paper, int)
    direction[1, 1] = signal
    direction[_NEIGHBORS[turn]] = signal
    contents = (tile, np.where(source, signal, paper), direction)
    heights = [len(a)+2 for a in contents]
    spare = h - sum(heights) - 4
    assert spare >= 0
    slack = rng.multinomial(spare, [.25]*4)
    panel = np.full((h, pwidth), paper, int)
    top = 1+int(slack[0])
    for index, content in enumerate(contents):
        bh, bw = np.array(content.shape)+2
        left = int(rng.integers(pwidth-bw+1))
        panel[top:top+bh, left:left+bw] = frame
        panel[top+1:top+bh-1, left+1:left+bw-1] = content
        top += bh + 1 + int(slack[index+1])
    divider = np.full((h, 1), separator, int)
    inp = np.concatenate((panel, divider, board) if side == "left" else (board, divider, panel), axis=1)
    # Forward label construction uses records rather than parsing the input.
    counts = np.zeros(board.shape, np.int16)
    rad, punch = len(tile)//2, tile == paper
    for i, j in np.argwhere(target):
        r, c = int(rows[i]), int(cols[j])
        counts[r-rad:r+rad+1, c-rad:c+rad+1] += punch
    out = board.copy()
    odd = counts % 2 == 1
    out[odd] = paper+ink-out[odd]
    return {"input": inp, "output": out}


def _candidate(rng, params, split, index, turn, side, fallback=False):
    witness = split == "train" and index < 2
    n = 3 if params["shift"] == "size" and split == "train" else (
        5 if params["shift"] == "size" else int(rng.choice([3, 5])))
    if witness and index == 0:
        n = 3
    nr, nc = map(int, rng.integers(2, 5, size=2))
    if witness:
        if index == 0:
            nr = int(rng.integers(3, 5))
        else:
            nc = int(rng.integers(3, 5))
    elif split == "test" and params["shift"] == "count":
        if rng.random() < .5:
            nr = int(rng.integers(5, 7))
        else:
            nc = int(rng.integers(5, 7))
    target = _mask(rng, nr, nc, witness)
    bits = _tile_bits(rng, n, witness)
    source_shape = target.shape[::-1] if turn % 2 else target.shape
    min_p = max(n+2, source_shape[1]+2, 5)
    pwidth = int(rng.integers(min_p, max(min_p, 9)+1))
    max_iw = 30-pwidth-3
    min_iw = max(n+nc-1, 16 if witness else 12)
    if min_iw > max_iw:
        return None
    iw = int(rng.integers(min_iw, max_iw+1))
    min_h = n + source_shape[0] + 13
    h = int(rng.integers(max(min_h, 20 if witness else 18), 31))
    palette = rng.choice(10, 5, replace=False)
    paper, ink = map(int, palette[:2])
    tile = np.where(bits, ink, paper)
    rows = _stations(rng, h, n, nr, witness)
    cols = _stations(rng, iw+2, n, nc, witness)
    canvas = _canvas(rng, h, iw, paper, ink,
                     "paper" if witness and index == 0 else "varied" if witness else params["canvas"])
    inert = 1 if witness and index == 1 else int(rng.integers(0, 3))
    if fallback:
        # Explicit finite witness construction, still recolored and positioned
        # by the supplied RNG. All dimensions fit for either card side/turn.
        n, nr, nc, h, iw, pwidth = 3, 3, 3, 25, 16, 7
        bits = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]], bool)
        target = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 0]], bool)
        tile = np.where(bits, ink, paper)
        rows, cols = np.array([5, 6, 12]), np.array([4, 5, 11])
        yy, xx = np.indices((h, iw))
        canvas = np.full((h, iw), paper, int) if index == 0 else np.where((yy+xx)%3 == 0, ink, paper)
        inert = int(index == 1)
    return _assemble(rng, tile, target, turn, rows, cols, canvas, palette, pwidth, side, inert,
                     early_inert=witness and index == 1)


def _witness_ok(pair, index):
    p = _decode(pair["input"])
    out, counts = pair["output"], _inverse_counts(p)
    if p["stencil"].all() or np.array_equal(p["stencil"], np.rot90(p["stencil"], 2)):
        return False
    if not np.any((counts > 0) & (counts % 2 == 0)):
        return False
    if index == 0:
        required = tuple(x for x in _SHORTCUTS if x not in ("paint_ink", "all_marks"))
    else:
        required = ("paint_ink", "union", "no_exchange", "fill_holes", "regular_spacing", "all_marks")
        if not np.any((counts % 2 == 1) & (p["board"] == p["ink"])):
            return False
    return all(not np.array_equal(_diagnostic_prediction(p, v), out) for v in required)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train", 2, 10), _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed % (1 << 64))
    params = dict(shift=str(rng.choice(["count", "size"])),
                  canvas=str(rng.choice(["paper", "varied", "varied"])))
    quarter, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    first_side = str(rng.choice(["left", "right"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = split == "train" and index < 2
            turn = quarter if witness and index == 0 else even if witness else (
                (-quarter) % 4 if split == "test" and index == 0 else int(rng.integers(4)))
            side = (first_side if index == 0 else ("right" if first_side == "left" else "left")) if witness else str(rng.choice(["left", "right"]))
            for attempt in range(129):
                pair = _candidate(rng, params, split, index, turn, side,
                                  fallback=witness and attempt == 128)
                if pair is None:
                    continue
                if witness and not _witness_ok(pair, index):
                    continue
                if np.array_equal(_decode(pair["input"])["board"], pair["output"]):
                    continue
                break
            else:
                raise AssertionError("bounded construction exhausted")
            if _key(pair["input"]) in seen:
                originals = {k: a.copy() for k, a in pair.items()}
                p = _decode(pair["input"])
                # Twenty distinct bijections exceed all thirteen prior inputs.
                for offset in range(20):
                    mapping = np.arange(10)
                    if offset >= 10:
                        mapping[p["paper"]], mapping[p["ink"]] = p["ink"], p["paper"]
                    mapping = (mapping + offset % 10) % 10
                    pair = {k: mapping[a] for k, a in originals.items()}
                    if _key(pair["input"]) not in seen:
                        break
                else:
                    raise AssertionError("distinct recolorings exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate visible geometry, labels, local invariants and training evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_decoded, test_decoded = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            p = _decode(inp)
            counts = _inverse_counts(p)
            assert out.shape == p["board"].shape, "return only framed canvas"
            assert np.array_equal(out, _diagnostic_prediction(p)), "incorrect parity label"
            assert np.array_equal(out[[0, -1]], p["board"][[0, -1]])
            assert np.array_equal(out[:, [0, -1]], p["board"][:, [0, -1]])
            assert not counts[[0, -1]].any() and not counts[:, [0, -1]].any()
            assert not np.array_equal(out, p["board"]), "generated pair must be informative"
            assert set(map(int, np.unique(out[1:-1, 1:-1]))) <= {p["paper"], p["ink"]}
            if split == "train":
                train_decoded.append((p, out))
            else:
                test_decoded.append(p)
    for behavior in _SHORTCUTS:
        assert any(not np.array_equal(_diagnostic_prediction(p, behavior), out)
                   for p, out in train_decoded), f"under-demonstrated rule: {behavior}"
    assert any(np.any((p["board"] == p["ink"]) & (_inverse_counts(p) % 2 == 1))
               for p, _ in train_decoded), "ink erasure witness"
    assert {p["side"] for p, _ in train_decoded} == {"left", "right"}
    assert any(p["turn"] % 2 for p, _ in train_decoded), "quarter-turn witness"
    assert any(np.any(p["board"][0] == p["ink"]) and np.any(p["board"][:, 0] == p["ink"])
               for p, _ in train_decoded), "color-matching witness"
    count_shift = (all(max(len(p["rows"]), len(p["cols"])) <= 4 for p, _ in train_decoded)
                   and all(max(len(p["rows"]), len(p["cols"])) >= 5 for p in test_decoded))
    size_shift = (all(len(p["tile"]) == 3 for p, _ in train_decoded)
                  and all(len(p["tile"]) == 5 for p in test_decoded))
    assert count_shift or size_shift, "visible train/test scale shift required"


def _puzzle_key(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def _replace_board(inp, p, board):
    out = inp.copy()
    out[:, p["origin"]:p["origin"]+board.shape[1]] = board
    return out


def _local_perturbation_test(inp):
    """One impression's finite difference equals its punch, regardless of hits."""
    p = _decode(inp)
    old = solve(inp)
    source = p["stencil"]
    chosen = None
    for r, c in np.ndindex(source.shape):
        changed = source.copy()
        changed[r, c] = ~changed[r, c]
        if changed[0].any() and changed[-1].any() and changed[:, 0].any() and changed[:, -1].any():
            chosen = (r, c)
            break
    assert chosen is not None
    panel = p["panel"].copy()
    top, left = p["field_origins"][1]
    r, c = chosen
    panel[top+r, left+c] = p["paper"] if source[r, c] else p["signal"]
    altered = inp.copy()
    start = 0 if p["side"] == "left" else p["divider"]+1
    altered[:, start:start+panel.shape[1]] = panel
    delta = np.zeros(source.shape, bool)
    delta[r, c] = True
    i, j = np.argwhere(np.rot90(delta, -p["turn"]))[0]
    rad, n = len(p["tile"])//2, len(p["tile"])
    y, x = p["rows"][i]-rad, p["cols"][j]-rad
    expected_delta = np.zeros(old.shape, bool)
    expected_delta[y:y+n, x:x+n] = p["tile"] == p["paper"]
    assert np.array_equal(solve(altered) != old, expected_delta)
    # Move one visible station: remove its old impressions, add its new ones.
    for axis, stations in ((0, p["rows"]), (1, p["cols"])):
        length = old.shape[axis]
        for rank, station in enumerate(stations):
            for step in (-1, 1):
                value = int(station+step)
                border_value = p["board"][value, 0] if axis == 0 and 0 <= value < length else (
                    p["board"][0, value] if axis == 1 and 0 <= value < length else p["signal"])
                if not (1+rad <= value <= length-2-rad) or border_value != p["frame"]:
                    continue
                replacement = stations.copy()
                replacement[rank] = value
                if len(set(map(int, replacement))) != len(stations) or np.any(np.diff(replacement) <= 0):
                    continue
                board = p["board"].copy()
                if axis == 0:
                    board[station, 0] = board[station, -1] = p["frame"]
                    board[value, 0] = board[value, -1] = p["signal"]
                else:
                    board[0, station] = board[-1, station] = p["frame"]
                    board[0, value] = board[-1, value] = p["signal"]
                mask = np.rot90(source, -p["turn"])
                diff = np.zeros(old.shape, bool)
                for i, j in np.argwhere(mask):
                    if (i if axis == 0 else j) != rank:
                        continue
                    y, x = int(p["rows"][i]), int(p["cols"][j])
                    for displacement in (0, step):
                        sy = y+displacement if axis == 0 else y
                        sx = x+displacement if axis == 1 else x
                        diff[sy-rad:sy+rad+1, sx-rad:sx+rad+1] ^= p["tile"] == p["paper"]
                new = solve(_replace_board(inp, p, board))
                assert np.array_equal((new != old)[1:-1, 1:-1], diff[1:-1, 1:-1])
                return
    raise AssertionError("witness offers no legal station perturbation")


def self_test():
    """Bounded 42-seed variation audit and semantic mutation checks."""
    signatures, shapes, tiles, masks, layouts, sizes, turns = set(), set(), set(), set(), set(), set(), set()
    palette_roles, multiplicities, gaps, max_hits = set(), set(), set(), 0
    curricula = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        first_test = _decode(puzzle["test"][0]["input"])
        curricula.add("count" if max(len(first_test["rows"]), len(first_test["cols"])) >= 5 else "size")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                before = inp.copy()
                p = _decode(inp)
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, before), "solve mutated input"
                shapes.add(inp.shape)
                tiles.add(_key(p["tile"] == p["paper"]))
                masks.add(_key(p["stencil"]))
                layouts.add(p["side"])
                sizes.add(len(p["tile"]))
                turns.add(p["turn"])
                palette_roles.add(tuple(p[k] for k in ("paper", "ink", "frame", "separator", "signal")))
                multiplicities.add((len(p["rows"]), len(p["cols"])))
                gaps.update(map(int, np.diff(p["rows"])))
                gaps.update(map(int, np.diff(p["cols"])))
                max_hits = max(max_hits, int(_inverse_counts(p).max()))
        _local_perturbation_test(puzzle["train"][0]["input"])
        # A palette bijection preserves all local semantics, including color 0.
        perm = np.random.default_rng(seed+400).permutation(10)
        pair = puzzle["test"][0]
        assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    assert len(signatures) == 42 and len(shapes) >= 25
    assert len(tiles) >= 80 and len(masks) >= 60 and len(palette_roles) >= 150
    assert layouts == {"left", "right"} and sizes == {3, 5} and turns == {0, 1, 2, 3}
    assert len(multiplicities) >= 12 and len(gaps) >= 7 and max_hits >= 4
    assert curricula == {"count", "size"}
    # Exercise every legal explicit count pair, and both default/partial calls.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    assert len(generate_puzzle(8, num_train=6)["train"]) == 6
    assert len(generate_puzzle(8, num_test=2)["test"]) == 2
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(17)
    damaged = {s: [{k: a.copy() for k, a in pair.items()} for pair in puzzle[s]] for s in puzzle}
    damaged["test"][0]["output"][1, 1] = (damaged["test"][0]["output"][1, 1]+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt answer accepted")
    for behavior in _SHORTCUTS:
        for index, pair in enumerate(puzzle["train"]):
            wrong = _diagnostic_prediction(_decode(pair["input"]), behavior)
            if np.array_equal(wrong, pair["output"]):
                continue
            damaged = {s: [{k: a.copy() for k, a in x.items()} for x in puzzle[s]] for s in puzzle}
            damaged["train"][index]["output"] = wrong
            try:
                validate_generated_puzzle(damaged)
            except AssertionError:
                break
            raise AssertionError(f"shortcut label accepted: {behavior}")
        else:
            raise AssertionError(f"missing shortcut witness: {behavior}")
    # The conservative witness must work for either quarter and even direction.
    rng = np.random.default_rng(99)
    for index, allowed in ((0, (1, 3)), (1, (0, 2))):
        for turn in allowed:
            pair = _candidate(rng, {"shift": "count", "canvas": "varied"}, "train", index,
                              turn, "left", fallback=True)
            assert _witness_ok(pair, index), "fallback witness not informative"
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                tile_patterns=len(tiles), stencil_patterns=len(masks),
                palettes=len(palette_roles), station_count_pairs=len(multiplicities),
                gap_values=len(gaps), maximum_hits=max_hits)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
