"""Pinwheel typesetter: a visible stencil dictionary drives four stateful paths.

Only NumPy and the standard library are needed. No file I/O occurs during
generation, solving, validation, or self_test. JSON, if written by a caller,
should use json.dumps(value, separators=(",", ":")).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _right(d):
    return d[1], -d[0]


def _path_positions(length, radial, stages):
    """Infinite-plane prefix, including off-canvas excursions and short jogs."""
    radius = max(2, length)
    dy, dx = radial
    for distance in range(1, radius + 1):
        yield distance * dy, distance * dx
    y, x = radius * dy, radius * dx
    forward = _right(radial)
    for k in range(stages):
        for (vy, vx), count in ((forward, 4*k + 2),
                                (_right(forward), radius - 2),
                                (forward, radius)):
            for _ in range(count):
                y, x = y + vy, x + vx
                yield y, x
        forward = _right(forward)


def _routes(shape, center, length):
    """Return ordered visible cells with their actual outgoing unit tangent."""
    h, w = shape
    cy, cx = center
    distance = max(cy, cx, h-1-cy, w-1-cx)
    # Four stages move each strand out by eight. Extra cycles cover re-entry.
    stages = 4 * (distance // 8 + 2)
    result, seen = [], set()
    for radial in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        positions = iter(_path_positions(length, radial, stages))
        previous = next(positions)
        visible = []
        for step, following in enumerate(positions):
            y, x = previous[0] + cy, previous[1] + cx
            if 0 <= y < h and 0 <= x < w:
                assert (y, x) not in seen, "pinwheel strands must not intersect"
                seen.add((y, x))
                visible.append((y, x, following[0]-previous[0],
                                following[1]-previous[1], step))
            previous = following
        result.append(visible)
    return result


def _analytic_domain(shape, center, length):
    """Closed-form interval membership, used only to cross-check path geometry."""
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy, xx = yy - center[0], xx - center[1]
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    result = np.zeros(shape, dtype=bool)
    for _ in range(4):
        result |= (xx == 0) & (yy >= -radius) & (yy <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, ((y, x), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (yy-y)*dy + (xx-x)*dx
                across = (yy-y)*dx - (xx-x)*dy
                run = 16*q + 4*s + 2
                result |= (across == 0) & (along >= 0) & (along <= run)
                result |= (along == run) & (across >= 0) & (across <= radius-2)
                result |= (across == radius-2) & (along >= run) & (along <= run+radius)
        yy, xx = xx, -yy
    return result


def _parse(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) <= a.size // 2:
        raise ValueError("background must be a strict majority")
    stripes = [c for c in range(a.shape[1])
               if a[0, c] != bg and np.all(a[:, c] == a[0, c])]
    if len(stripes) != 1:
        raise ValueError("expected one full-height separator")
    separator = stripes[0]
    spine = int(a[0, separator])
    left, right = a[:, :separator], a[:, separator+1:]
    if left.shape[1] < right.shape[1]:
        legend, board, side = left, right.copy(), "left"
    else:
        board, legend, side = left.copy(), right, "right"
    size = legend.shape[1] - 1
    h, w = board.shape
    if size not in (2, 3) or max(h*size, w*size) > 30 or min(h, w) < 7:
        raise ValueError("invalid legend width or print dimensions")
    book, covered = {}, np.zeros_like(legend, dtype=bool)
    for row in np.flatnonzero(legend[:, 0] != bg):
        row = int(row)
        code = int(legend[row, 0])
        if row + size > h or code in book:
            raise ValueError("duplicate or truncated dictionary card")
        tile = legend[row:row+size, 1:]
        ink_colors = set(map(int, np.unique(tile))) - {bg}
        if len(ink_colors) != 1:
            raise ValueError("each stencil must have one ink")
        ink = ink_colors.pop()
        mask = tile == ink
        if not (mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any()):
            raise ValueError("stencil bounds must be tight")
        book[code] = (ink, mask.copy())
        covered[row, 0] = True
        covered[row:row+size, 1:] |= mask
    if not 1 <= len(book) <= 3 or not np.array_equal(covered, legend != bg):
        raise ValueError("unexplained dictionary material")
    codes, inks = set(book), {v[0] for v in book.values()}
    if (len(inks) != len(book) or codes & inks
            or {bg, spine} & (codes | inks) or bg == spine):
        raise ValueError("color roles must be disjoint")
    if not set(map(int, np.unique(board))) <= {bg, spine} | codes | inks:
        raise ValueError("unexplained board color")
    gray = np.argwhere(board == spine)
    if len(gray) == 0 or len(gray) % 4:
        raise ValueError("expected four equal neutral spokes")
    length = len(gray) // 4 + 1
    center_float = gray.mean(axis=0)
    center = tuple(map(int, center_float))
    if not np.array_equal(center_float, center):
        raise ValueError("nonintegral hollow-cross center")
    cy, cx = center
    if not (length <= cy < h-length and length <= cx < w-length):
        raise ValueError("cross must be fully visible")
    expected = np.zeros_like(board, dtype=bool)
    for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        if int(board[cy+dy, cx+dx]) not in codes:
            raise ValueError("each printhead needs an initial cartridge")
        for d in range(2, length+1):
            expected[cy+d*dy, cx+d*dx] = True
    if not np.array_equal(expected, board == spine) or board[center] != bg:
        raise ValueError("malformed hollow cross")
    return {"board": board, "book": book, "bg": bg, "spine": spine,
            "size": size, "length": length, "center": center, "side": side}


_QUARTER_TURNS = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}


def _print(board, book, bg, center, length, mode="normal"):
    size = next(iter(book.values()))[1].shape[0]
    out = np.full((board.shape[0]*size, board.shape[1]*size), bg, dtype=np.int8)
    for strand in _routes(board.shape, center, length):
        state = int(board[strand[0][0], strand[0][1]])
        for y, x, dy, dx, _ in strand:
            code = int(board[y, x])
            if mode != "initial" and code in book:
                state = code
            if mode == "local" and code not in book:
                continue
            ink, mask = book[state]
            turns = 0 if mode == "unrotated" else _QUARTER_TURNS[(dy, dx)]
            out[y*size:(y+1)*size, x*size:(x+1)*size][np.rot90(mask, turns)] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the input's own vocabulary and execute its four directed tapes."""
    info = _parse(input_grid)
    return _print(info["board"], info["book"], info["bg"],
                  info["center"], info["length"])


def _connected(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    stack = [remaining.pop()]
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = y+dy, x+dx
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
    return not remaining


def _mask_pool(size):
    pool = []
    for bits in range(1, 1 << (size*size)):
        mask = np.array([(bits >> n) & 1 for n in range(size*size)],
                        dtype=bool).reshape(size, size)
        if not (mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        if _connected(mask) and int(mask.sum()) <= size*size-1:
            pool.append(mask)
    return pool


def _encode(board, book, bg, spine, side, rng):
    h, w = board.shape
    size = next(iter(book.values()))[1].shape[0]
    legend = np.full((h, size+1), bg, dtype=np.int8)
    codes = list(book)
    rng.shuffle(codes)
    slack = h - len(codes)*size - (len(codes)-1)
    gaps = rng.multinomial(slack, np.full(len(codes)+1, 1/(len(codes)+1)))
    row = int(gaps[0])
    for i, code in enumerate(codes):
        ink, mask = book[code]
        legend[row, 0] = code
        legend[row:row+size, 1:][mask] = ink
        row += size + 1 + int(gaps[i+1])
    stripe = np.full((h, 1), spine, dtype=np.int8)
    return np.concatenate((legend, stripe, board) if side == "left"
                          else (board, stripe, legend), axis=1)


def _make_pair(rng, size, book, backgrounds, spine, length, kind, allow_painted=True):
    """Bounded randomized construction; all semantics are encoded in the grid."""
    codes = list(book)
    min_h = max(8, len(codes)*size+len(codes)-1, 2*length+1)
    min_w = max(8, 2*length+1)
    for attempt in range(80):
        if kind == "test":
            if size == 2:
                h, w = map(int, rng.integers(14, 16, size=2))
            else:
                h, w = map(int, rng.integers(9, 11, size=2))
                if h == w == 9:
                    h = 10
        else:
            high = 13 if size == 2 else 9
            safe = kind in ("simple", "jog")
            h = int(rng.integers(max(min_h, 2*length+3 if safe else 0), high+1))
            w = int(rng.integers(max(min_w, 2*length+3 if safe else 0), high+1))
        padding = 1 if kind in ("simple", "jog") else 0
        cy = int(rng.integers(length+padding, h-length-padding))
        cx = int(rng.integers(length+padding, w-length-padding))
        center = cy, cx
        bg = int(rng.choice(backgrounds))
        board = np.full((h, w), bg, dtype=np.int8)
        initial = codes + [int(rng.choice(codes)) for _ in range(4-len(codes))]
        rng.shuffle(initial)
        for (dy, dx), code in zip(((-1, 0), (0, 1), (1, 0), (0, -1)), initial):
            board[cy+dy, cx+dx] = code
            for d in range(2, length+1):
                board[cy+d*dy, cx+d*dx] = spine
        routes = _routes(board.shape, center, length)
        domain = {(y, x) for arm in routes for y, x, *_ in arm}
        switches = []
        if kind == "jog":
            # First inward jog: its successor has smaller Manhattan radius.
            choices = []
            for arm in routes:
                for i, p in enumerate(arm[:-2]):
                    y, x, *_ = p
                    ny, nx, *_ = arm[i+1]
                    if (abs(ny-cy)+abs(nx-cx) < abs(y-cy)+abs(x-cx)
                            and max(abs(y-cy), abs(x-cx)) > 1):
                        choices.append((y, x))
                        break
            if not choices:
                continue
            switches.append(choices[int(rng.integers(len(choices)))])
        number = (0 if kind == "simple" else int(rng.integers(2, 4))
                  if kind == "jog" else int(rng.integers(4, 8))
                  if kind == "test" else int(rng.integers(1, 5)))
        candidates = [(y, x) for arm in routes for i, (y, x, *_) in enumerate(arm)
                      if i >= length and i < len(arm)-3 and board[y, x] == bg]
        rng.shuffle(candidates)
        for q in candidates if number else []:
            if q not in switches and all(max(abs(q[0]-p[0]), abs(q[1]-p[1])) > 1
                                         for p in switches):
                switches.append(q)
            if len(switches) >= number:
                break
        if len(switches) < number:
            continue
        switch_set = set(switches)
        for arm in routes:
            state = int(board[arm[0][0], arm[0][1]])
            for y, x, *_ in arm:
                if (y, x) in switch_set:
                    state = int(rng.choice([c for c in codes if c != state]))
                    board[y, x] = state
        # Same vocabulary between strands is inert. Keep the hollow center clear.
        unused = [(y, x) for y in range(h) for x in range(w)
                  if (y, x) not in domain and (y, x) != center]
        rng.shuffle(unused)
        for y, x in unused[:int(rng.integers(1, 4))]:
            board[y, x] = int(rng.choice(codes))
        # Optional already painted stencil; ink is material, never a command.
        if kind != "simple" and rng.random() < .7 and allow_painted:
            code = int(rng.choice(codes))
            ink, mask = book[code]
            locations = [(y, x) for y in range(h-size+1) for x in range(w-size+1)]
            rng.shuffle(locations)
            for y, x in locations:
                block = board[y:y+size, x:x+size]
                if np.all(block == bg) and not (y <= cy < y+size and x <= cx < x+size):
                    block[mask] = ink
                    break
        side = str(rng.choice(["left", "right"]))
        inp = _encode(board, book, bg, spine, side, rng)
        if np.count_nonzero(inp == bg) <= inp.size // 2:
            continue
        out = _print(board, book, bg, center, length)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded placement budget exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    size = int(rng.choice([2, 3]))
    count = int(rng.integers(2, 4)) if size == 2 else 2
    palette = list(map(int, rng.permutation(10)))
    backgrounds, spine = palette[:2], palette[2]
    codes, inks = palette[3:3+count], palette[3+count:3+2*count]
    pool = _mask_pool(size)
    choices = rng.choice(len(pool), count, replace=False)
    book = {c: (ink, pool[int(i)].copy()) for c, ink, i in zip(codes, inks, choices)}
    train_plan = [(2, "simple"), (3, "jog")]
    for _ in range(nt-2):
        train_plan.append((int(rng.choice([2, 3, 4] if size == 2 else [2, 3])), "ordinary"))
    taught_lengths = sorted({p[0] for p in train_plan})
    puzzle, used, painted_taught = {"train": [], "test": []}, set(), False
    plan = [("train", l, kind) for l, kind in train_plan]
    plan += [("test", int(rng.choice(taught_lengths)), "test") for _ in range(nq)]
    for split, length, kind in plan:
        for _ in range(12):
            pair = _make_pair(rng, size, book, backgrounds, spine, length, kind,
                              allow_painted=(split == "train" or painted_taught))
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    board = _parse(pair["input"])["board"]
                    painted_taught |= bool(np.isin(board, inks).any())
                break
        else:
            raise RuntimeError("distinct-input budget exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _book_key(book):
    return tuple((code, ink, mask.shape, mask.tobytes())
                 for code, (ink, mask) in sorted(book.items()))


def _facts(info):
    board, book, center, length = (info[k] for k in ("board", "book", "center", "length"))
    routes = _routes(board.shape, center, length)
    domain, switches, decreases, directions = set(), 0, 0, set()
    for arm in routes:
        state = int(board[arm[0][0], arm[0][1]])
        for i, (y, x, dy, dx, _) in enumerate(arm):
            domain.add((y, x))
            directions.add((dy, dx))
            code = int(board[y, x])
            if code in book and code != state:
                switches += 1
                state = code
                if i+1 < len(arm):
                    ny, nx = arm[i+1][:2]
                    decreases += int(abs(ny-center[0])+abs(nx-center[1])
                                     < abs(y-center[0])+abs(x-center[1]))
    off = [(int(y), int(x)) for y, x in np.argwhere(np.isin(board, list(book)))
           if (int(y), int(x)) not in domain]
    inks = {ink for ink, _ in book.values()}
    return {"switches": switches, "jog_changes": decreases, "off": off,
            "domain": domain, "directions": directions,
            "painted": int(np.count_nonzero(np.isin(board, list(inks))))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, dictionary, common_spine, all_info = set(), None, None, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            book_key = _book_key(info["book"])
            masks = [mask for _, mask in info["book"].values()]
            assert len({mask.tobytes() for mask in masks}) == len(masks)
            assert all(_connected(mask) and not mask.all()
                       and len({np.rot90(mask, k).tobytes() for k in range(4)}) == 4
                       for mask in masks), "stencils must expose exact shape and orientation"
            if dictionary is None:
                dictionary = book_key
                common_spine = info["spine"]
            assert dictionary == book_key, "dictionary changes within a puzzle"
            assert common_spine == info["spine"], "spine color changes within a puzzle"
            board, size, bg = info["board"], info["size"], info["bg"]
            assert out.shape == (board.shape[0]*size, board.shape[1]*size)
            assert np.array_equal(solve(inp), out), "incorrect input-only prediction"
            facts = _facts(info)
            cy, cx = info["center"]
            assert {int(board[cy+dy, cx+dx]) for dy, dx in
                    ((-1, 0), (0, 1), (1, 0), (0, -1))} == set(info["book"])
            domain_mask = np.zeros(board.shape, dtype=bool)
            for q in facts["domain"]:
                domain_mask[q] = True
            assert np.array_equal(domain_mask, _analytic_domain(
                board.shape, info["center"], info["length"])), "path geometry drift"
            blocks = out.reshape(board.shape[0], size, board.shape[1], size)
            printed = np.any(blocks != bg, axis=(1, 3))
            assert np.array_equal(printed, domain_mask), "one nonempty tile per strand cell"
            assert not printed[info["center"]], "hollow center must stay blank"
            assert set(map(int, np.unique(out))) <= {bg} | {ink for ink, _ in info["book"].values()}
            assert facts["off"], "generated examples must expose off-strand rejection"
            assert not np.array_equal(out, _print(board, info["book"], bg,
                                                 info["center"], info["length"], "local"))
            assert not np.array_equal(out, _print(board, info["book"], bg,
                                                 info["center"], info["length"], "unrotated"))
            all_info[split].append((info, facts))
    trained_lengths = {i["length"] for i, _ in all_info["train"]}
    assert {2, 3} <= trained_lengths
    assert any(f["switches"] == 0 for _, f in all_info["train"])
    assert any(f["jog_changes"] > 0 for _, f in all_info["train"]), "teach traversal rather than radius order"
    assert (not any(f["painted"] for _, f in all_info["test"])
            or any(f["painted"] for _, f in all_info["train"])), "painted material must be taught"
    assert any(not np.array_equal(
        _print(i["board"], i["book"], i["bg"], i["center"], i["length"], "initial"),
        _print(i["board"], i["book"], i["bg"], i["center"], i["length"]))
        for i, _ in all_info["train"]), "must demonstrate cartridge replacement"
    for info, facts in all_info["test"]:
        assert info["length"] in trained_lengths, "test length must be taught"
        assert facts["switches"] >= 4, "test combines multiple taught state changes"
        assert not np.array_equal(_print(info["board"], info["book"], info["bg"],
                                         info["center"], info["length"], "initial"),
                                  _print(info["board"], info["book"], info["bg"],
                                         info["center"], info["length"]))


def self_test():
    """Bounded, streaming checks of 42 seeds; returns compact aggregate evidence."""
    # A small explicit interaction fixture protects outgoing orientation and
    # persistence through the inward jog, rather than just comparing renderers.
    board = np.zeros((9, 9), dtype=np.int8)
    book = {1: (2, np.array([[1, 1], [1, 0]], dtype=bool)),
            3: (4, np.array([[1, 1], [0, 1]], dtype=bool))}
    for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        board[4+dy, 4+dx] = 1
        for d in (2, 3):
            board[4+d*dy, 4+d*dx] = 6
    board[1, 6] = board[3, 5] = 3
    fixture = solve(_encode(board, book, 0, 6, "left", np.random.default_rng(0)))
    assert np.array_equal(fixture[2:4, 12:14], [[0, 4], [4, 4]])
    assert np.array_equal(fixture[4:6, 12:14], [[4, 4], [0, 4]])
    assert np.array_equal(fixture[2:4, 10:12], [[2, 2], [2, 0]])
    assert np.array_equal(fixture[6:8, 8:10], [[2, 0], [2, 2]])
    assert np.array_equal(fixture[8:10, 10:12], [[2, 2], [2, 0]])
    assert not fixture[6:8, 10:12].any() and not fixture[8:10, 8:10].any()
    metrics = Counter()
    shapes, books, signatures, layouts, lengths, sides, sizes = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                facts = _facts(info)
                board, center, length = info["board"], info["center"], info["length"]
                shapes.add(board.shape)
                books.add(_book_key(info["book"]))
                lengths.add(length)
                sizes.add(info["size"])
                sides.add(info["side"])
                layouts.add((board.shape, center, length, facts["switches"], tuple(facts["off"])))
                signature.update(repr(_key(pair["input"])).encode())
                metrics["pairs"] += 1
                metrics["switches"] += facts["switches"]
                metrics["off_strand_markers"] += len(facts["off"])
                metrics["painted_pairs"] += int(facts["painted"] > 0)
                metrics["jog_change_pairs"] += int(facts["jog_changes"] > 0)
                metrics["edge_crosses"] += int(min(center[0]-length, center[1]-length,
                    board.shape[0]-1-center[0]-length, board.shape[1]-1-center[1]-length) == 0)
        signatures.add(signature.digest())
    assert len(signatures) == 42 and len(books) == 42
    assert sizes == {2, 3} and sides == {"left", "right"} and {2, 3, 4} <= lengths
    assert len(shapes) >= 30 and len(layouts) >= 180
    assert metrics["painted_pairs"] > 30 and metrics["jog_change_pairs"] >= 42
    assert metrics["edge_crosses"] > 10
    # Check all supported jog scales, covariance and clipping/re-entry separately.
    for length in range(2, 7):
        for shape, center in (((30, 30), (length, length)),
                              ((22, 29), (21-length, 28-length))):
            routes = _routes(shape, center, length)
            mask = np.zeros(shape, dtype=bool)
            for arm in routes:
                for y, x, *_ in arm:
                    mask[y, x] = True
            assert np.array_equal(mask, _analytic_domain(shape, center, length))
            large = _analytic_domain((90, 90), (45, 45), length)
            y, x = 45-center[0], 45-center[1]
            assert np.array_equal(mask, large[y:y+shape[0], x:x+shape[1]])
            assert np.array_equal(np.rot90(mask), _analytic_domain(
                shape[::-1], (shape[1]-1-center[1], center[0]), length))
    assert _analytic_domain((22, 22), (9, 9), 3)[21, 21]
    # Reproducibility, count boundaries and rejection of label corruption.
    first, second = generate_puzzle(991, 10, 4), generate_puzzle(991, 10, 4)
    assert all(np.array_equal(a[k], b[k]) for s in first
               for a, b in zip(first[s], second[s]) for k in ("input", "output"))
    validate_generated_puzzle(generate_puzzle(992, 2, 1))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(994, 2, 1)
    out = bad["test"][0]["output"]
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted test label accepted")
    for mutation in ("erase_tile", "ignore_changes"):
        bad = generate_puzzle(995, 2, 1)
        pair = bad["test"][0]
        info = _parse(pair["input"])
        if mutation == "erase_tile":
            y, x = _routes(info["board"].shape, info["center"], info["length"])[0][0][:2]
            size = info["size"]
            pair["output"][y*size:(y+1)*size, x*size:(x+1)*size] = info["bg"]
        else:
            pair["output"] = _print(info["board"], info["book"], info["bg"],
                                     info["center"], info["length"], "initial")
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{mutation} was accepted")
    # Input counterfactuals distinguish the inherited concepts without labels.
    p = generate_puzzle(996, 2, 1)["test"][0]
    info = _parse(p["input"])
    facts = _facts(info)
    start = info["size"]+2 if info["side"] == "left" else 0
    changed = p["input"].copy()
    y, x = facts["off"][0]
    changed[y, start+x] = info["bg"]
    assert np.array_equal(solve(changed), p["output"]), "off-strand commands are inert"
    arm = _routes(info["board"].shape, info["center"], info["length"])[0]
    y, x = arm[0][:2]
    changed = p["input"].copy()
    changed[y, start+x] = next(c for c in info["book"] if c != int(changed[y, start+x]))
    assert not np.array_equal(solve(changed), p["output"]), "cartridge identity matters"
    changed = p["input"].copy()
    code = next(iter(info["book"]))
    legend_start = 0 if info["side"] == "left" else info["board"].shape[1]+1
    row = int(np.flatnonzero(changed[:, legend_start] == code)[0])
    # Rotate one visible master. This remains a valid input vocabulary and
    # changes all its impressions without changing the strand geometry.
    size = info["size"]
    block = changed[row:row+size, legend_start+1:legend_start+1+size]
    block[:] = np.rot90(block).copy()
    assert not np.array_equal(solve(changed), p["output"]), "stencil shape matters"
    metrics.update({"seeds": 42, "distinct_dictionaries": len(books),
                    "board_shapes": len(shapes), "layouts": len(layouts)})
    return dict(metrics)
