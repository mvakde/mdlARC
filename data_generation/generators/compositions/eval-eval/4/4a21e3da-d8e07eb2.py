"""Keys that open cuts: exact lookup certifies a spatial cut in the glyph cargo.

Only standard-library modules and NumPy are used. No parent imports or fixtures.
"""
from __future__ import annotations

import itertools
import numpy as np


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} is outside the permitted range")
    return value


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("a grid must be a two-dimensional integer array")
    if not all(1 <= v <= 30 for v in a.shape) or not np.all((a >= 0) & (a <= 9)):
        raise ValueError("grid bounds or colors are invalid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _trim(a, mask):
    rr, cc = np.nonzero(mask)
    if not len(rr):
        raise ValueError("empty token")
    return a[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


def _orient(a, orientation):
    # Transpose and quarter-turns give all eight whole-scene symmetries.
    return np.rot90(a.T if orientation >= 4 else a, orientation % 4).copy()


def _geometry(n, sh, sw):
    return (n + 1) * sh + 4, (n + 1) * sw + 4, sh + 2, sw + 2


def _complete(order, n, side):
    pos = set(p for p in order if p is not None)
    if len(pos) != n:
        return None
    coordinates = {p[1] if side == "top" else p[0] for p in pos}
    if len(coordinates) != 1:
        return None
    return next(iter(coordinates))


def _failure_kind(order, n, side):
    pos = set(p for p in order if p is not None)
    if _complete(order, n, side) is not None:
        return "complete"
    if len(pos) < n:
        return "partial"
    if pos in ({(i, i) for i in range(n)}, {(i, n - 1 - i) for i in range(n)}):
        return "diagonal"
    other = "left" if side == "top" else "top"
    if _complete(order, n, other) is not None:
        return "orthogonal"
    counts = [sum((r, c) in pos for r in range(n)) for c in range(n)] if side == "top" else [
        sum((r, c) in pos for c in range(n)) for r in range(n)]
    return "near" if max(counts) == n - 1 else "scatter"


def _parse_canonical(g):
    bg = int(g[0, 0])
    if not (np.all(g[[0, -1], :] == bg) and np.all(g[:, [0, -1]] == bg)):
        raise ValueError("the outside margin must be background")
    candidates = []
    for n, sh, sw in itertools.product((3, 4), (4, 5), (4, 5)):
        h, w, r0, c0 = _geometry(n, sh, sw)
        if g.shape != (h, w):
            continue
        r1, c1 = r0 + n * sh, c0 + n * sw
        frame = int(g[r0 - 1, c0 - 1])
        if frame == bg:
            continue
        if not (np.all(g[[r0 - 1, r1], c0 - 1:c1 + 1] == frame)
                and np.all(g[r0 - 1:r1 + 1, [c0 - 1, c1]] == frame)):
            continue
        expected = np.full_like(g, bg)
        expected[[r0 - 1, r1], c0 - 1:c1 + 1] = frame
        expected[r0 - 1:r1 + 1, [c0 - 1, c1]] = frame
        board, lookup, trimmed = {}, {}, set()
        valid = True
        for r, c in itertools.product(range(n), repeat=2):
            rs, cs = slice(r0 + r * sh, r0 + (r + 1) * sh), slice(c0 + c * sw, c0 + (c + 1) * sw)
            block = g[rs, cs]
            mask = block != bg
            if (not mask.any() or len(np.unique(block[mask])) != 1
                    or frame in block[mask] or np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]])):
                valid = False
                break
            cropped = _trim(block, mask)
            if _key(block) in lookup or _key(cropped) in trimmed:
                valid = False
                break
            # Consistent placement; offset alone is never token identity.
            rr, cc = np.nonzero(mask)
            if rr.min() != 1 + (sh - 2 - cropped.shape[0]) // 2 or cc.min() != 1 + (sw - 2 - cropped.shape[1]) // 2:
                valid = False
                break
            board[r, c] = block.copy()
            lookup[_key(block)] = (r, c)
            trimmed.add(_key(cropped))
            expected[rs, cs] = block
        if not valid:
            continue
        queries = {}
        for side in ("top", "left"):
            order = []
            for j in range(n):
                rs = slice(1, 1 + sh) if side == "top" else slice(r0 + j * sh, r0 + (j + 1) * sh)
                cs = slice(c0 + j * sw, c0 + (j + 1) * sw) if side == "top" else slice(1, 1 + sw)
                block = g[rs, cs]
                if np.all(block == bg):
                    order.append(None)
                elif _key(block) in lookup:
                    order.append(lookup[_key(block)])
                else:
                    valid = False
                    break
                expected[rs, cs] = block
            if not valid:
                break
            occupied = [p for p in order if p is not None]
            if not occupied or len(occupied) != len(set(occupied)):
                valid = False
                break
            queries[side] = order
        if not valid or not np.array_equal(g, expected):
            continue
        col, row = _complete(queries["top"], n, "top"), _complete(queries["left"], n, "left")
        candidates.append(dict(grid=g, bg=bg, frame=frame, n=n, sh=sh, sw=sw,
                               r0=r0, c0=c0, board=board, queries=queries,
                               row=row, col=col, flags=(col is not None, row is not None)))
    if len(candidates) != 1:
        raise ValueError("no unique framed glyph/key lattice")
    return candidates[0]


def _analyze(input_grid):
    g = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(g, rotation))
        except ValueError:
            continue
        info["rotation"] = rotation
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("query sides must identify one scene orientation")
    return candidates[0]


def _regions(info, force_flags=None):
    n, sh, sw, r0, c0 = (info[k] for k in ("n", "sh", "sw", "r0", "c0"))
    cargo = info["grid"][r0:r0 + n * sh, c0:c0 + n * sw]
    # Mutation diagnostics can force a cut even when a key set does not certify it.
    top, left = info["flags"] if force_flags is None else force_flags
    def guessed(side, known):
        if known is not None:
            return known
        axis = 1 if side == "top" else 0
        counts = [sum(p is not None and p[axis] == j for p in info["queries"][side]) for j in range(n)]
        return int(np.argmax(counts))
    ac = guessed("top", info["col"]) * sw + (sw - 1) // 2 if top else None
    ar = guessed("left", info["row"]) * sh + (sh - 1) // 2 if left else None
    cells = [(int(r), int(c), int(cargo[r, c])) for r, c in np.argwhere(cargo != info["bg"])]
    fixed, groups = [], {}
    for r, c, color in cells:
        if (ar is not None and r == ar) or (ac is not None and c == ac):
            fixed.append((r, c, color))
        elif top and left:
            corner = (r > ar, c > ac)  # False = top/left, True = bottom/right.
            groups.setdefault(corner, []).append((r, c, color))
        elif top:
            groups.setdefault((False, c > ac), []).append((r, c, color))
        elif left:
            groups.setdefault((r > ar, False), []).append((r, c, color))
    return cargo, cells, fixed, groups, ar, ac


def _move(cells, corner, shape):
    if not cells:
        return []
    rr, cc, _ = zip(*cells)
    dr = shape[0] - 1 - max(rr) if corner[0] else -min(rr)
    dc = shape[1] - 1 - max(cc) if corner[1] else -min(cc)
    return [(r + dr, c + dc, color) for r, c, color in cells]


def _render_solution(info, *, force_flags=None, nearest=False, separate=False, keep_opposite=False):
    cargo, cells, fixed, groups, ar, ac = _regions(info, force_flags)
    if ar is None and ac is None:
        return np.rot90(info["grid"], -info["rotation"]).copy()
    result = np.full_like(cargo, info["bg"])
    for r, c, color in fixed:
        result[r, c] = color
    if ac is not None:
        hits = [r for r, c, _ in cells if c == ac]
        if not hits and force_flags is None:
            raise ValueError("active vertical cut must intersect ink")
        stop = (min(hits) if nearest else max(hits)) if hits else -1
        for r in range(stop + 1):
            if result[r, ac] == info["bg"]:
                result[r, ac] = info["frame"]
    if ar is not None:
        hits = [c for r, c, _ in cells if r == ar]
        if not hits and force_flags is None:
            raise ValueError("active horizontal cut must intersect ink")
        stop = (min(hits) if nearest else max(hits)) if hits else -1
        for c in range(stop + 1):
            if result[ar, c] == info["bg"]:
                result[ar, c] = info["frame"]
    for corner, group in groups.items():
        if ar is not None and ac is not None and corner == (True, True) and not keep_opposite:
            continue
        parts = _components(group) if separate else [group]
        for part in parts:
            for r, c, color in _move(part, corner, result.shape):
                if not separate and result[r, c] != info["bg"]:
                    raise ValueError("translated partition collides with fixed ink or guide")
                result[r, c] = color
    output = info["grid"].copy()
    r0, c0 = info["r0"], info["c0"]
    output[r0:r0 + cargo.shape[0], c0:c0 + cargo.shape[1]] = result
    return np.rot90(output, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every convention from the displayed input; never read generation state."""
    return _render_solution(_analyze(input_grid))


def _components(cells):
    remaining = {(r, c): color for r, c, color in cells}
    result = []
    while remaining:
        start = min(remaining)
        stack, part = [start], []
        color = remaining.pop(start)
        part.append((*start, color))
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    color = remaining.pop(p)
                    part.append((*p, color))
                    stack.append(p)
        result.append(part)
    return result


def _bank(gh, gw):
    bank = {}
    for code in range(1, 1 << (gh * gw)):
        mask = np.array([(code >> j) & 1 for j in range(gh * gw)], bool).reshape(gh, gw)
        crop = _trim(mask, mask)
        centered = np.zeros((gh, gw), bool)
        r, c = (gh - crop.shape[0]) // 2, (gw - crop.shape[1]) // 2
        centered[r:r + crop.shape[0], c:c + crop.shape[1]] = crop
        bank[_key(crop)] = centered
    return list(bank.values())


def _token(mask, color, bg):
    block = np.full((mask.shape[0] + 2, mask.shape[1] + 2), bg, np.int8)
    block[1:-1, 1:-1][mask] = color
    return block


def _make_board(rng, n, sh, sw, colors, line_col):
    bg, frame, *inks = colors
    bank = _bank(sh - 2, sw - 2)
    # Three explicit confusions: identical shape/different ink; identical ink/different shape.
    special = np.indices((sh - 2, sw - 2)).sum(0) % 2 == 0
    solid = np.ones_like(special)
    chosen_row = int(rng.integers(n))
    a = (chosen_row, line_col)
    outside = [(r, c) for r, c in itertools.product(range(n), repeat=2) if c != line_col]
    rng.shuffle(outside)
    b, d = outside[:2]
    board = {a: _token(special, inks[0], bg), b: _token(special, inks[1], bg),
             d: _token(solid, inks[0], bg)}
    used = {_key(block) for block in board.values()}
    combos = [(mask, ink) for mask in bank for ink in inks
              if _key(_token(mask, ink, bg)) not in used]
    rng.shuffle(combos)
    for p in itertools.product(range(n), repeat=2):
        if p not in board:
            mask, color = combos.pop()
            board[p] = _token(mask, color, bg)
    return board, a, b


def _query(rng, n, side, kind, index):
    line = [(r, index) for r in range(n)] if side == "top" else [(index, c) for c in range(n)]
    if kind == "complete":
        selected = line
    elif kind == "near":
        missing = int(rng.integers(n))
        outside = [p for p in itertools.product(range(n), repeat=2) if p not in line]
        selected = line[:missing] + line[missing + 1:] + [outside[int(rng.integers(len(outside)))]]
    elif kind == "partial":
        count = int(rng.integers(1, n))
        selected = [line[int(j)] for j in rng.choice(n, count, replace=False)]
    elif kind == "diagonal":
        selected = [(j, j) for j in range(n)]
        if rng.random() < .5:
            selected = [(r, n - 1 - c) for r, c in selected]
    elif kind == "orthogonal":
        selected = [(index, c) for c in range(n)] if side == "top" else [(r, index) for r in range(n)]
    elif kind == "scatter":
        choices = [list(v) for v in itertools.combinations(itertools.product(range(n), repeat=2), n)
                   if _complete(v, n, side) is None and _failure_kind(v, n, side) == "scatter"]
        if not choices:
            return _query(rng, n, side, "near", index)
        selected = choices[int(rng.integers(len(choices)))]
    else:
        raise ValueError("unknown query type")
    order = list(selected)
    rng.shuffle(order)
    slots = sorted(int(j) for j in rng.choice(n, len(order), replace=False))
    result = [None] * n
    for slot, position in zip(slots, order):
        result[slot] = position
    return result


def _make_input(n, sh, sw, colors, board, top, left):
    h, w, r0, c0 = _geometry(n, sh, sw)
    bg, frame = colors[:2]
    g = np.full((h, w), bg, np.int8)
    g[[r0 - 1, r0 + n * sh], c0 - 1:c0 + n * sw + 1] = frame
    g[r0 - 1:r0 + n * sh + 1, [c0 - 1, c0 + n * sw]] = frame
    for (r, c), block in board.items():
        g[r0 + r * sh:r0 + (r + 1) * sh, c0 + c * sw:c0 + (c + 1) * sw] = block
    for j, p in enumerate(top):
        if p is not None:
            g[1:1 + sh, c0 + j * sw:c0 + (j + 1) * sw] = board[p]
    for j, p in enumerate(left):
        if p is not None:
            g[r0 + j * sh:r0 + (j + 1) * sh, 1:1 + sw] = board[p]
    return g


def _feasible(g):
    info = _analyze(g)
    cargo, cells, fixed, groups, ar, ac = _regions(info)
    if ar is not None and ac is not None:
        if set(groups) != set(itertools.product((False, True), repeat=2)):
            return False
    elif ar is not None or ac is not None:
        if len(groups) != 2:
            return False
    for axis, value in ((0, ac), (1, ar)):
        if value is not None:
            hits = [p[axis] for p in cells if p[1 - axis] == value]
            slot = info["sh"] if axis == 0 else info["sw"]
            if len({v // slot for v in hits}) < 2:
                return False
    try:
        _render_solution(info)
    except ValueError:
        return False
    return True


def _colors(rng):
    return tuple(int(c) for c in rng.choice(10, int(rng.integers(4, 7)), replace=False))


def _edge_index(rng, n, slot):
    # For a two-pixel glyph, the center toward the query side is its first pixel.
    # The first board line would leave no ink on the outer side of that axis.
    return n - 1 if slot == 4 else int(rng.choice((0, n - 1)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    n = int(rng.choice((3, 4)))
    sh, sw = (int(v) for v in rng.choice((4, 5), 2))
    colors = _colors(rng)
    orientation = int(rng.integers(8))
    varied_layout = nt >= 4 and rng.random() < .5
    varied_palette = nt >= 3 and rng.random() < .5
    varied_rotation = nt >= 4 and rng.random() < .45
    col, row = _edge_index(rng, n, sw), _edge_index(rng, n, sh)
    # The first two lessons are a deliberate one-key contrast on identical cargo.
    for _ in range(128):
        board, a, b = _make_board(rng, n, sh, sw, colors, col)
        top = _query(rng, n, "top", "complete", col)
        left = _query(rng, n, "left", "complete", row)
        g0 = _make_input(n, sh, sw, colors, board, top, left)
        replacement = [b if p == a else p for p in top]
        g1 = _make_input(n, sh, sw, colors, board, replacement, left)
        if _feasible(g0) and _feasible(g1):
            break
    else:
        raise RuntimeError("failed to construct the bounded core contrast")
    puzzle, seen = {"train": [], "test": []}, set()

    def append(g, split, orient):
        inp = _orient(g, orient)
        key = _key(inp)
        if key in seen:
            return False
        pair = {"input": inp, "output": solve(inp)}
        puzzle[split].append(pair)
        seen.add(key)
        return True

    append(g0, "train", orientation)
    append(g1, "train", orientation)
    schedule = [(True, False)] if nt >= 3 else []
    if nt >= 4:
        schedule.append((False, False))
    while len(schedule) < nt - 2:
        schedule.append(tuple(bool(v) for v in rng.integers(0, 2, 2)))
    failure_options = ["near", "partial", "diagonal", "orthogonal", "scatter"]
    for index, flags in enumerate(schedule):
        for attempt in range(128):
            nn = int(rng.choice((3, 4))) if varied_layout else n
            ss, tt = (int(v) for v in rng.choice((4, 5), 2)) if varied_layout else (sh, sw)
            cc = _colors(rng) if varied_palette else colors
            ci, ri = _edge_index(rng, nn, tt), _edge_index(rng, nn, ss)
            bb, _, _ = _make_board(rng, nn, ss, tt, cc, ci)
            kinds = ["complete" if flag else str(rng.choice(failure_options)) for flag in flags]
            qt, ql = _query(rng, nn, "top", kinds[0], ci), _query(rng, nn, "left", kinds[1], ri)
            g = _make_input(nn, ss, tt, cc, bb, qt, ql)
            orient = (orientation // 4) * 4 + int(rng.integers(4)) if varied_rotation else orientation
            if _feasible(g) and append(g, "train", orient):
                break
        else:
            raise RuntimeError("failed to construct distinct training cargo")
    train_infos = [_analyze(p["input"]) for p in puzzle["train"]]
    # Use canonical layouts and flags of witnessed scenes, including transposition.
    for test_index in range(nq):
        for attempt in range(128):
            pool = [i for i in train_infos if any(i["flags"])] if test_index == 0 else train_infos
            witness = pool[int(rng.integers(len(pool)))]
            nn, ss, tt = witness["n"], witness["sh"], witness["sw"]
            cc = _colors(rng) if varied_palette else (witness["bg"], witness["frame"], *sorted({
                int(v) for block in witness["board"].values() for v in np.unique(block) if v != witness["bg"]}))
            ci, ri = int(rng.integers(1, nn - 1)), int(rng.integers(1, nn - 1))
            bb, _, _ = _make_board(rng, nn, ss, tt, cc, ci)
            flags = witness["flags"]
            kinds = []
            for side, flag in zip(("top", "left"), flags):
                taught = sorted({_failure_kind(i["queries"][side], i["n"], side) for i in train_infos
                                 if _failure_kind(i["queries"][side], i["n"], side) != "complete"})
                kinds.append("complete" if flag else str(rng.choice(taught)))
            qt, ql = _query(rng, nn, "top", kinds[0], ci), _query(rng, nn, "left", kinds[1], ri)
            g = _make_input(nn, ss, tt, cc, bb, qt, ql)
            # Witness is already canonical; only rotate it back (do not transpose twice).
            orient = int(rng.integers(4)) if varied_rotation else (-witness["rotation"]) % 4
            if _feasible(g) and append(g, "test", orient):
                break
        else:
            raise RuntimeError("failed to construct distinct test cargo")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    puzzle["test"] = [puzzle["test"][int(i)] for i in rng.permutation(nq)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _lookup_mutant(info, kind):
    clone = dict(info)
    def identity(block):
        if kind == "shape":
            return _key(block != info["bg"])
        return int(block[block != info["bg"]][0])
    queries = {}
    for side, order in info["queries"].items():
        matched = set()
        for p in order:
            if p is not None:
                matched.update(q for q, b in info["board"].items() if identity(b) == identity(info["board"][p]))
        queries[side] = list(sorted(matched))
    clone["col"] = _complete(queries["top"], info["n"], "top")
    clone["row"] = _complete(queries["left"], info["n"], "left")
    clone["flags"] = (clone["col"] is not None, clone["row"] is not None)
    return _render_solution(clone)


def pair_features(input_grid):
    info = _analyze(input_grid)
    cargo, cells, fixed, groups, ar, ac = _regions(info)
    deleted = groups.get((True, True), []) if ar is not None and ac is not None else []
    retained = [v for corner, v in groups.items() if not (ar is not None and ac is not None and corner == (True, True))]
    return dict(n=info["n"], slot=(info["sh"], info["sw"]), shape=info["grid"].shape,
                flags=info["flags"], cut=(info["row"], info["col"]), bg=info["bg"], frame=info["frame"],
                ink_count=len(set(color for _, _, color in cells)), ink_cells=len(cells),
                fixed=len(fixed), deleted=len(deleted),
                partition_sizes=[len(v) for v in retained],
                components=[len(_components(v)) for v in retained],
                failures=[_failure_kind(info["queries"][s], info["n"], s) for s in ("top", "left")],
                board_key=_key(cargo), rotation=info["rotation"])


def validate_generated_puzzle(puzzle):
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert _feasible(inp)
            assert np.array_equal(solve(inp), out)
            canonical_out = np.rot90(out, info["rotation"])
            cargo, cells, fixed, groups, ar, ac = _regions(info)
            r0, c0 = info["r0"], info["c0"]
            bay_out = canonical_out[r0:r0 + cargo.shape[0], c0:c0 + cargo.shape[1]]
            exterior = np.ones(info["grid"].shape, bool)
            exterior[r0:r0 + cargo.shape[0], c0:c0 + cargo.shape[1]] = False
            assert np.array_equal(canonical_out[exterior], info["grid"][exterior])
            for r, c, color in fixed:
                assert bay_out[r, c] == color
            deleted = groups.get((True, True), []) if ar is not None and ac is not None else []
            for color in set(v for _, _, v in cells):
                assert np.count_nonzero(bay_out == color) == sum(v == color for _, _, v in cells) - sum(
                    v == color for _, _, v in deleted)
            for corner, group in groups.items():
                if ar is not None and ac is not None and corner == (True, True):
                    continue
                moved = _move(group, corner, cargo.shape)
                for r, c, color in moved:
                    assert bay_out[r, c] == color
                rr, cc, _ = zip(*moved)
                assert (max(rr) == cargo.shape[0] - 1 if corner[0] else min(rr) == 0)
                assert (max(cc) == cargo.shape[1] - 1 if corner[1] else min(cc) == 0)
            infos[split].append(info)
    tr, te = infos["train"], infos["test"]
    assert (True, True) in {i["flags"] for i in tr}
    assert any(sum(i["flags"]) == 1 for i in tr)
    assert {i["flags"] for i in te} <= {i["flags"] for i in tr}
    assert {i["n"] for i in te} <= {i["n"] for i in tr}
    assert any(any(i["flags"]) for i in te), "the test split must exercise an interior cut"
    for info in tr:
        assert all(index is None or index in (0, info["n"] - 1) for index in (info["row"], info["col"]))
    for info in te:
        assert all(index is None or 0 < index < info["n"] - 1 for index in (info["row"], info["col"]))
    for side in ("top", "left"):
        taught = {_failure_kind(i["queries"][side], i["n"], side) for i in tr}
        assert {_failure_kind(i["queries"][side], i["n"], side) for i in te} <= taught
    # Pixel-derived same-cargo contrast, without seed or construction metadata.
    contrast = False
    for a, b in itertools.combinations(tr, 2):
        ca = _regions(a)[0]
        cb = _regions(b)[0]
        if a["bg"] == b["bg"] and a["frame"] == b["frame"] and np.array_equal(ca, cb):
            if sorted((sum(a["flags"]), sum(b["flags"]))) == [1, 2]:
                contrast = True
    assert contrast, "missing same-cargo key contrast"
    # Every training collection defeats ablations of both inherited mechanisms.
    mutants = [False] * 6
    for pair, info in zip(puzzle["train"], tr):
        out = np.asarray(pair["output"])
        choices = (info["grid"], _render_solution(info, force_flags=(True, True)),
                   _render_solution(info, nearest=True), _render_solution(info, separate=True),
                   _lookup_mutant(info, "shape"), _lookup_mutant(info, "color"))
        # First mutant leaves cargo in place; restore displayed orientation.
        choices = (np.rot90(choices[0], -info["rotation"]), *choices[1:])
        for j, mutant in enumerate(choices):
            mutants[j] |= not np.array_equal(mutant, out)
    assert all(mutants), "training does not distinguish a simpler alternative"


def self_test():
    """Bounded 42-seed diversity, counts, determinism, metamorphism, and damage checks."""
    records, puzzle_keys = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        q = generate_puzzle(seed)
        assert all(np.array_equal(a[k], b[k]) for split in p for a, b in zip(p[split], q[split]) for k in a)
        puzzle_keys.add(tuple(_key(pair["input"]) for s in p for pair in p[s]))
        records.extend((split, pair_features(pair["input"])) for split in p for pair in p[split])
    assert len(puzzle_keys) == 42
    fs = [f for _, f in records]
    assert {f["n"] for f in fs} == {3, 4}
    assert len({f["slot"] for f in fs}) == 4
    assert len({f["shape"] for f in fs}) == 7
    assert len({f["board_key"] for f in fs}) >= 130
    assert len({f["bg"] for f in fs}) >= 8
    assert len({f["frame"] for f in fs}) >= 8
    assert {f["ink_count"] for f in fs} >= {2, 3, 4}
    assert {f["flags"] for f in fs} == set(itertools.product((False, True), repeat=2))
    assert {k for f in fs for k in f["failures"]} >= {"complete", "near", "partial", "diagonal", "orthogonal"}
    assert max(c for f in fs for c in f["components"]) >= 5
    assert len({tuple(f["partition_sizes"]) for f in fs}) >= 80
    assert {f["rotation"] for f in fs} == set(range(4))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(9000 + 10 * train + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("accepted illegal counts")
    base = generate_puzzle(0)
    for orient in range(8):
        for split in base:
            for pair in base[split]:
                assert np.array_equal(solve(_orient(pair["input"], orient)), _orient(pair["output"], orient))
    for defect in ("duplicate", "pinned_ink", "exterior"):
        damaged = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        if defect == "duplicate":
            damaged["train"][1] = {k: v.copy() for k, v in damaged["train"][0].items()}
        else:
            pair = next(p for p in damaged["train"] if all(_analyze(p["input"])["flags"]))
            info = _analyze(pair["input"])
            out = np.rot90(pair["output"], info["rotation"])
            if defect == "pinned_ink":
                r, c, _ = _regions(info)[2][0]
                out[r + info["r0"], c + info["c0"]] = info["bg"]
            else:
                out[0, 0] = info["frame"]
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted damage")


if __name__ == "__main__":
    self_test()
    print("42-seed self-test passed")
