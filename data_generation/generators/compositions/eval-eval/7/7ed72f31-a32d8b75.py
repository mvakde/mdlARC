"""Symmetry interference prints, using a coarse stencil and a binary swatch.

No parent generator, examples, files, seed, or training state is used by solve.
Construction stamps forwards; solve/validation reconstruct by inverse coordinates.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _marker(board, color):
    cells = np.argwhere(board == color)
    if len(cells) == 1:
        return "P", tuple(map(int, cells[0]))
    lo, hi = cells.min(0), cells.max(0)
    if lo[0] == hi[0] and len(cells) == hi[1] - lo[1] + 1:
        return "H", tuple(map(int, lo))
    if lo[1] == hi[1] and len(cells) == hi[0] - lo[0] + 1:
        return "V", tuple(map(int, lo))
    raise ValueError("The board needs one straight marker or one point marker")


def _map(r, c, kind, center):
    ar, ac = center
    return (2 * ar - r if kind in ("H", "P") else r,
            2 * ac - c if kind in ("V", "P") else c)


def _turn(tile, kind):
    if kind == "H":
        return tile[::-1, :]
    if kind == "V":
        return tile[:, ::-1]
    return tile[::-1, ::-1]


def _decode(input_grid):
    a = _grid(input_grid)
    exterior = int(a[0, 0])
    if not (np.all(a[0] == exterior) and np.all(a[-1] == exterior)
            and np.all(a[:, 0] == exterior) and np.all(a[:, -1] == exterior)):
        raise ValueError("A uniform exterior border must surround both frames")
    components = _components(a != exterior)
    if len(components) != 2:
        raise ValueError("Expected exactly two separated framed rectangles")
    fields, frames = [], []
    for component in components:
        points = np.array(component)
        lo, hi = points.min(0), points.max(0) + 1
        box = a[lo[0]:hi[0], lo[1]:hi[1]]
        if len(component) != box.size or min(box.shape) < 4:
            raise ValueError("Each field must be a complete framed rectangle")
        frame = int(box[0, 0])
        if not (np.all(box[0] == frame) and np.all(box[-1] == frame)
                and np.all(box[:, 0] == frame) and np.all(box[:, -1] == frame)):
            raise ValueError("Broken frame")
        interior = box[1:-1, 1:-1].copy()
        if np.any(interior == frame):
            raise ValueError("Frame color is structural and absent from field interiors")
        fields.append(interior)
        frames.append(frame)
    if frames[0] != frames[1]:
        raise ValueError("Both frames must share one structural color")
    tiles = [f for f in fields if len(np.unique(f)) == 2]
    boards = [f for f in fields if len(np.unique(f)) == 3]
    if len(tiles) != 1 or len(boards) != 1:
        raise ValueError("The binary swatch and three-color board must be distinct")
    tile, board = tiles[0], boards[0]
    n = len(tile)
    if tile.shape != (n, n) or not 2 <= n <= 5:
        raise ValueError("Swatch must be a complete binary square of side 2..5")
    colors = list(map(int, np.unique(tile)))
    markers = set(map(int, np.unique(board))) - set(colors)
    if len(markers) != 1 or not set(colors) <= set(map(int, np.unique(board))):
        raise ValueError("Board and swatch must share the same binary palette")
    marker = markers.pop()
    counts = [int(np.count_nonzero(board == color)) for color in colors]
    if counts[0] == counts[1]:
        raise ValueError("The carrier role must be the strictly more frequent board color")
    off = colors[int(counts[1] > counts[0])]
    on = colors[1 - colors.index(off)]
    kind, center = _marker(board, marker)
    if any(x * n > 30 for x in board.shape) or min(board.shape) < 5:
        raise ValueError("Board dimensions must be at least 5 and expand within 30")
    pegs = set(map(tuple, np.argwhere(board == on).tolist()))
    h, w = board.shape
    for r, c in pegs:
        if (kind == "H" and r == center[0]) or (kind == "V" and c == center[1]):
            raise ValueError("No peg may lie on the infinite extension of a line axis")
        rr, cc = _map(r, c, kind, center)
        if not (0 <= rr < h and 0 <= cc < w) or board[rr, cc] == marker:
            raise ValueError("Every reflected peg must fit clear of markers")
    return dict(tile=tile, board=board, n=n, off=off, on=on, marker=marker,
                kind=kind, center=center, exterior=exterior, frame=frames[0])


def _inverse(info, behavior="normal"):
    """Reconstruct a pixel from its source and inverse-reflected macro address."""
    board, n = info["board"], info["n"]
    h, w = board.shape
    kind, center = info["kind"], info["center"]
    rr, cc = np.indices((h * n, w * n))
    mr, mc, tr, tc = rr // n, cc // n, rr % n, cc % n
    source = board[mr, mc] == info["on"]
    ir, ic = _map(mr, mc, kind, center)
    inside = (ir >= 0) & (ic >= 0) & (ir < h) & (ic < w)
    image = inside & (board[np.clip(ir, 0, h - 1), np.clip(ic, 0, w - 1)] == info["on"])
    tile_bits = info["tile"] == info["on"]
    sr = n - 1 - tr if kind in ("H", "P") else tr
    sc = n - 1 - tc if kind in ("V", "P") else tc
    positive = tile_bits[tr, tc]
    negative = ~tile_bits[sr, sc]
    if behavior == "no_symmetry":
        image = np.zeros_like(image)
    elif behavior == "no_tile":
        positive = negative = np.ones_like(source)
    elif behavior == "no_exchange":
        negative = tile_bits[sr, sc]
    elif behavior == "exchange_both":
        positive = ~tile_bits[tr, tc]
    elif behavior == "no_tile_transform":
        negative = ~tile_bits[tr, tc]
    first, second = source & positive, image & negative
    bits = first | second if behavior == "union" else first ^ second
    if behavior == "overwrite":
        bits = np.where(image, negative, first)
    if behavior == "macro_cancel":
        bits &= ~(source & image)
    out = np.where(bits, info["on"], info["off"])
    out[board[mr, mc] == info["marker"]] = info["marker"]
    return out.astype(int)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only decoding and inverse-coordinate interference reconstruction."""
    return _inverse(_decode(input_grid))


def _forward(board, tile, off, on, marker):
    """Generation path: stamp each positive tile and its exchanged symmetric image."""
    kind, center = _marker(board, marker)
    n = len(tile)
    bits = np.zeros(tuple(x * n for x in board.shape), dtype=bool)
    positive = tile == on
    negative = ~_turn(positive, kind)
    for r, c in np.argwhere(board == on):
        r, c = int(r), int(c)
        rr, cc = _map(r, c, kind, center)
        bits[r*n:(r+1)*n, c*n:(c+1)*n] ^= positive
        bits[rr*n:(rr+1)*n, cc*n:(cc+1)*n] ^= negative
    out = np.where(bits, on, off).astype(int)
    out[np.repeat(np.repeat(board == marker, n, axis=0), n, axis=1)] = marker
    return out


def _facts(info):
    board, on = info["board"], info["on"]
    pegs = set(map(tuple, np.argwhere(board == on).tolist()))
    images = {_map(r, c, info["kind"], info["center"]) for r, c in pegs}
    overlap = pegs & images
    bits = info["tile"] == on
    double_bits = bits ^ ~_turn(bits, info["kind"])
    axis_cells = np.argwhere(board == info["marker"])
    axis_coord = 1 if info["kind"] == "H" else 0
    overhang = info["kind"] != "P" and any(
        p[axis_coord] < axis_cells[:, axis_coord].min()
        or p[axis_coord] > axis_cells[:, axis_coord].max() for p in pegs)
    return dict(kind=info["kind"], board_shape=board.shape, tile_size=info["n"],
                source_count=len(pegs), reciprocal_pairs=len(overlap)//2,
                single_orbits=len(pegs-images), marker_length=len(axis_cells),
                overhang=bool(overhang), tile_witness=bool(
                    double_bits.any() and not double_bits.all()),
                empty_macros=board.size-len(pegs | images)-len(axis_cells),
                tile_bits=(bits.shape, bits.tobytes()),
                stencil=(board.shape, (board == on).tobytes()))


def _tile(rng, n, off, on, kind, witness):
    for _ in range(100):
        r, c = np.indices((n, n))
        style = int(rng.integers(6))
        if style == 0:
            bits = rng.random((n, n)) < rng.uniform(.18, .82)
        elif style == 1:
            bits = (r == int(rng.integers(n))) | (c == int(rng.integers(n)))
        elif style == 2:
            bits = (r + c) % int(rng.integers(2, 4)) == 0
        elif style == 3:
            bits = (r == 0) | (c == 0) | (r == n - 1) | (c == n - 1)
            bits[int(rng.integers(n)), int(rng.integers(n))] ^= True
        elif style == 4:
            bits = (r < int(rng.integers(1, n))) & (c < int(rng.integers(1, n)))
        else:
            bits = np.zeros((n, n), bool)
            for row in range(n):
                bits[row, int(rng.integers(n))] = True
        collision = bits ^ ~_turn(bits, kind)
        if bits.any() and not bits.all() and (not witness or (collision.any() and not collision.all())):
            return np.where(bits, on, off).astype(int)
    # A single ink cell always has both equal and unequal reflected locations.
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, on, off).astype(int)


def _orbits(h, w, kind, center):
    orbits = set()
    for r in range(h):
        for c in range(w):
            if (kind == "H" and r == center[0]) or (kind == "V" and c == center[1]):
                continue
            q = _map(r, c, kind, center)
            if 0 <= q[0] < h and 0 <= q[1] < w and q != (r, c):
                orbits.add(tuple(sorted(((r, c), q))))
    return sorted(orbits)


def _stencil(rng, h, w, kind, center, marker_cells, test):
    orbits = _orbits(h, w, kind, center)
    budget = (h*w - len(marker_cells) - 1)//2  # carrier strictly outnumbers pegs
    cap = min(4, (budget-2)//2, len(orbits)-3)
    reciprocal = int(rng.integers(2, cap+1)) if test else 1
    single_max = min(5, budget - 2*reciprocal, len(orbits)-reciprocal-1)
    singles = int(rng.integers(2, single_max+1))
    rr, cc = np.indices((h, w))
    style = int(rng.integers(6))
    ar, ac = int(rng.integers(h)), int(rng.integers(w))
    if style == 0:
        preference = rng.random((h, w))
    elif style == 1:
        preference = ((rr == ar) | (cc == ac)).astype(float)
    elif style == 2:
        preference = (np.abs((rr-ar)-(cc-ac)) <= 1).astype(float)
    elif style == 3:
        preference = ((rr == 0) | (cc == 0) | (rr == h-1) | (cc == w-1)).astype(float)
    elif style == 4:
        preference = 1 / (1 + abs(rr-ar) + abs(cc-ac))
    else:
        preference = ((rr % 2 == 0) & (cc % 2 == 0)).astype(float)
    scores = [max(preference[p], preference[q]) + float(rng.random())*.35 for p, q in orbits]
    selected = [orbits[i] for i in sorted(range(len(orbits)), key=lambda i: -scores[i])[:reciprocal+singles]]
    double_indices = set(map(int, rng.choice(len(selected), reciprocal, replace=False)))
    pegs = set()
    for i, (p, q) in enumerate(selected):
        if i in double_indices:
            pegs.update((p, q))
        elif preference[p] == preference[q]:
            pegs.add((p, q)[int(rng.integers(2))])
        else:
            pegs.add(p if preference[p] > preference[q] else q)
    return pegs


def _frame(interior, color):
    return np.pad(interior, 1, constant_values=color)


def _layout(rng, board, tile, frame, exterior):
    boxes = [_frame(board, frame), _frame(tile, frame)]
    if rng.integers(2):
        boxes.reverse()
    vertical = bool(rng.integers(2))
    if vertical:
        boxes = [b.T for b in boxes]
    gap = int(rng.integers(1, 5))
    width = sum(b.shape[1] for b in boxes) + gap + 2 + int(rng.integers(4))
    height = max(len(b) for b in boxes) + 2 + int(rng.integers(5))
    a = np.full((height, width), exterior, int)
    spare = width - sum(b.shape[1] for b in boxes) - gap - 2
    left = 1 + int(rng.integers(spare+1))
    for box in boxes:
        top = int(rng.integers(1, height - len(box)))
        a[top:top+len(box), left:left+box.shape[1]] = box
        left += box.shape[1] + gap
    return a.T.copy() if vertical else a


def _make_pair(rng, kind, test, witness):
    n = int(rng.integers(2, 6))
    bound = min(12, 30//n)
    h, w = int(rng.integers(5, bound+1)), int(rng.integers(5, bound+1))
    # Centers may be offset, but leave at least depth two on either side.
    ar, ac = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
    center = (ar, ac)
    if kind == "H":
        length = int(rng.integers(2, w+1))
        start = int(rng.integers(w-length+1))
        marker_cells = [(ar, c) for c in range(start, start+length)]
    elif kind == "V":
        length = int(rng.integers(2, h+1))
        start = int(rng.integers(h-length+1))
        marker_cells = [(r, ac) for r in range(start, start+length)]
    else:
        marker_cells = [center]
    off, on, marker, frame, exterior = map(int, rng.choice(10, 5, replace=False))
    board = np.full((h, w), off, int)
    for p in marker_cells:
        board[p] = marker
    for p in _stencil(rng, h, w, kind, center, marker_cells, test):
        board[p] = on
    tile = _tile(rng, n, off, on, kind, witness)
    return dict(input=_layout(rng, board, tile, frame, exterior),
                output=_forward(board, tile, off, on, marker))


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    line = str(rng.choice(["H", "V"]))
    kinds = [line, "P"]
    if nt >= 3 and rng.random() < .6:
        kinds.append("V" if line == "H" else "H")
    plan = list(rng.permutation(kinds))
    while len(plan) < nt:
        plan.append(str(rng.choice(kinds)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            kind = str(plan[i]) if split == "train" else str(rng.choice(kinds))
            for _ in range(20):
                pair = _make_pair(rng, kind, split == "test", split == "train" and i < len(kinds))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not produce a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


_ALTERNATIVES = ("no_symmetry", "no_tile", "no_exchange", "exchange_both",
                 "no_tile_transform", "union", "overwrite", "macro_cancel")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _count(len(puzzle["train"]), "num_train", 2, 10)
    _count(len(puzzle["test"]), "num_test", 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, trained, witnesses, refuted = set(), set(), set(), set()
    train_collisions = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            info, predicted = _decode(inp), solve(inp)
            assert np.array_equal(out, predicted), "Wrong interference output"
            facts = _facts(info)
            assert facts["reciprocal_pairs"] >= 1 and facts["single_orbits"] >= 2
            assert facts["empty_macros"] >= 1, "No transparent stencil hole"
            # Entire expanded marker blocks survive; output has exactly three roles.
            assert np.count_nonzero(out == info["marker"]) == facts["marker_length"] * info["n"]**2
            assert set(map(int, np.unique(out))) == {info["off"], info["on"], info["marker"]}
            if split == "train":
                trained.add(info["kind"])
                train_collisions.append(facts["reciprocal_pairs"])
                if facts["tile_witness"]:
                    witnesses.add(info["kind"])
                for variant in _ALTERNATIVES:
                    if not np.array_equal(_inverse(info, variant), out):
                        refuted.add(variant)
            else:
                assert info["kind"] in trained, "Test requires an undemonstrated marker kind"
                assert facts["reciprocal_pairs"] > max(train_collisions), "Missing collision-count shift"
    assert "P" in trained and trained & {"H", "V"}
    assert trained <= witnesses, "Each operation needs a nonuniform overlap witness"
    assert refuted == set(_ALTERNATIVES), "Training permits a listed shortcut"


def self_test():
    measurements, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        signature = tuple((_key(x["input"]), _key(x["output"])) for s in p for x in p[s])
        assert signature == tuple((_key(x["input"]), _key(x["output"])) for s in q for x in q[s])
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"].copy()
                info = _decode(inp)
                facts = _facts(info)
                measurements.append(facts)
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, pair["input"]), "solve modified its input"
                board, tile = info["board"], info["tile"]
                # Validate the three local stamp cases, including both bits of collisions.
                sources = set(map(tuple, np.argwhere(board == info["on"]).tolist()))
                images = {_map(r, c, info["kind"], info["center"]) for r, c in sources}
                negative = np.where(_turn(tile, info["kind"]) == info["on"], info["off"], info["on"])
                collision = np.where((tile == info["on"]) ^ (negative == info["on"]), info["on"], info["off"])
                for cells, expected in ((sources-images, tile), (images-sources, negative), (sources & images, collision)):
                    for r, c in cells:
                        n = info["n"]
                        assert np.array_equal(pair["output"][r*n:(r+1)*n, c*n:(c+1)*n], expected)
    assert len(signatures) == 42
    assert {m["kind"] for m in measurements} == {"H", "V", "P"}
    assert {m["tile_size"] for m in measurements} == {2, 3, 4, 5}
    assert len({m["board_shape"] for m in measurements}) >= 25
    assert len({m["tile_bits"] for m in measurements}) >= 45
    assert len({m["stencil"] for m in measurements}) >= 180
    assert sum(m["overhang"] for m in measurements) >= 15
    assert {m["reciprocal_pairs"] for m in measurements} >= {1, 2, 3}
    # Hand-calculated overlap fingerprints: T=00/01.
    t = np.array([[0, 0], [0, 1]], int)
    for kind, expected in (("H", [[1, 0], [1, 0]]), ("V", [[1, 1], [0, 0]]), ("P", [[0, 1], [1, 0]])):
        b = np.zeros((5, 5), int)
        if kind == "H":
            b[2, 1:3] = 2
        elif kind == "V":
            b[1:3, 2] = 2
        else:
            b[2, 2] = 2
        for pos in ((0, 0), (4, 4) if kind == "P" else ((4, 0) if kind == "H" else (0, 4)), (1, 1), (3, 4)):
            b[pos] = 1
        a = np.full((9, 14), 9, int)
        a[1:8, 1:8] = _frame(b, 8)
        a[1:5, 9:13] = _frame(t, 8)
        out = solve(a)
        assert np.array_equal(out[:2, :2], np.array(expected)), kind
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(2026, nt, nq)
        assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    damaged = generate_puzzle(0)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt label accepted")
    return dict(seeds=42, pairs=len(measurements), distinct_puzzles=len(signatures),
                marker_kinds=sorted({m["kind"] for m in measurements}),
                tile_sizes=sorted({m["tile_size"] for m in measurements}),
                board_shapes=len({m["board_shape"] for m in measurements}),
                binary_swatches=len({m["tile_bits"] for m in measurements}),
                stencil_structures=len({m["stencil"] for m in measurements}),
                overhanging_line_examples=sum(m["overhang"] for m in measurements),
                reciprocal_pair_counts=sorted({m["reciprocal_pairs"] for m in measurements}))
