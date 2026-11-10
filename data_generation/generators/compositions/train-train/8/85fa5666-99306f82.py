"""Corner keys turn clockwise, then pour into their quadrant of the frame.

Composition of 85fa5666 (four diagonal corner identities around a core move one
corner clockwise) and 99306f82 (a diagonal colour key leading into a frame corner
becomes nested rectangular layers, far-end colour outermost, last colour
saturating the remaining core).

Family rule (see task_spec.md):
* Every object is a one-cell hollow rectangular frame whose interior has even
  height and even width.  Up to four ordered colour keys sit on the outward
  diagonals of its four outer corners (a key is a diagonal run of distinct
  colours starting at the cell diagonally outside the corner).
* Every key turns one corner clockwise (TL->TR->BR->BL->TL), keeping its order
  along the diagonal measured from the frame.
* The interior is split by its midlines into four equal quadrants.  The key
  now standing at a corner pours into that corner's quadrant: an interior cell
  at boundary depth d (distance to the nearest interior edge) receives key
  colour min(d, k-1), counting from the key's far end.  A quadrant whose corner
  receives no key stays background.
"""
from __future__ import annotations

import numpy as np

TL, TR, BR, BL = 0, 1, 2, 3
CORNER_NAMES = ("TL", "TR", "BR", "BL")
OUTWARD = ((-1, -1), (-1, 1), (1, 1), (1, -1))
SHIFTS = ("scale", "crowd", "aspect", "sparse", "mixed")
MAX_KEY = 5


class _Retry(Exception):
    """Internal signal: resample the current pair or puzzle."""


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------
# Geometry shared by the solver, the validator and the generator.
# --------------------------------------------------------------------------
def _corner(frame, c):
    top, left, bottom, right = frame
    return ((top, left), (top, right), (bottom, right), (bottom, left))[c]


def _diagonal(frame, c, n):
    """Cells of a length-n key at corner c, nearest the frame first."""
    r, col = _corner(frame, c)
    dr, dc = OUTWARD[c]
    return [(r + dr * (i + 1), col + dc * (i + 1)) for i in range(n)]


def _quadrant(i, j, hi, wi):
    top = 2 * i < hi
    left = 2 * j < wi
    if top:
        return TL if left else TR
    return BL if left else BR


def _ring_depth(i, j, hi, wi):
    return min(i, j, hi - 1 - i, wi - 1 - j)


def _corner_depth(q, i, j, hi, wi):
    """Chebyshev distance from the quadrant's own corner (shortcut variant)."""
    a = i if q in (TL, TR) else hi - 1 - i
    b = j if q in (TL, BL) else wi - 1 - j
    return max(a, b)


# --------------------------------------------------------------------------
# Input-only parsing.
# --------------------------------------------------------------------------
def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_frames(grid, bg):
    """One-cell hollow rectangles (4-connected, at least 4x4) of any colour."""
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    frames = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                            and int(grid[ny, nx]) == color):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)
            fh, fw = bottom - top + 1, right - left + 1
            if fh < 4 or fw < 4 or len(cells) != 2 * (fh + fw) - 4:
                continue
            if all(y in (top, bottom) or x in (left, right) for y, x in cells):
                frames.append((top, left, bottom, right))
    return sorted(frames)


def _read_keys(grid, bg, frame):
    """Four keys (TL, TR, BR, BL), each ordered from its far end to the frame."""
    h, w = grid.shape
    keys = []
    for c in range(4):
        r, col = _corner(frame, c)
        dr, dc = OUTWARD[c]
        r, col = r + dr, col + dc
        run = []
        while 0 <= r < h and 0 <= col < w and int(grid[r, col]) != bg:
            run.append(int(grid[r, col]))
            r, col = r + dr, col + dc
        keys.append(tuple(reversed(run)))
    return keys


CLOCKWISE = (BL, TL, TR, BR)  # corner c receives the key that stood at CLOCKWISE[c]


def _apply(grid, *, source=CLOCKWISE, order="far", tail="saturate", depth="ring",
           fill="quadrant"):
    """The family rule (default arguments) or a named shortcut variant."""
    grid = np.asarray(grid).astype(int)
    bg = _background(grid)
    frames = _find_frames(grid, bg)
    keys = {frame: _read_keys(grid, bg, frame) for frame in frames}
    out = grid.copy()
    h, w = grid.shape
    for frame in frames:
        for c in range(4):
            for cell in _diagonal(frame, c, len(keys[frame][c])):
                out[cell] = bg
    for frame in frames:
        moved = [keys[frame][source[c]] for c in range(4)]
        for c in range(4):
            key = moved[c]
            for i, (r, col) in enumerate(_diagonal(frame, c, len(key))):
                if 0 <= r < h and 0 <= col < w:
                    out[r, col] = key[len(key) - 1 - i]
        top, left, bottom, right = frame
        hi, wi = bottom - top - 1, right - left - 1
        for i in range(hi):
            for j in range(wi):
                q = _quadrant(i, j, hi, wi)
                key = moved[TL] if fill == "single" else moved[q]
                if not key:
                    continue
                if order == "near":
                    key = key[::-1]
                d = (_ring_depth(i, j, hi, wi) if depth == "ring"
                     else _corner_depth(q, i, j, hi, wi))
                n = len(key)
                if tail == "saturate":
                    color = key[min(d, n - 1)]
                elif tail == "cycle":
                    color = key[d % n]
                elif tail == "flat_far":
                    color = key[0]
                elif tail == "flat_near":
                    color = key[-1]
                else:
                    raise ValueError(tail)
                out[top + 1 + i, left + 1 + j] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Rotate every frame's corner keys clockwise and pour them into quadrants."""
    return _apply(input_grid)


SHORTCUTS = {
    "no_turn": {"source": (TL, TR, BR, BL)},
    "counterclockwise": {"source": (TR, BR, BL, TL)},
    "half_turn": {"source": (BR, BL, TL, TR)},
    "mirror_left_right": {"source": (TR, TL, BL, BR)},
    "mirror_top_bottom": {"source": (BL, BR, TR, TL)},
    "transpose": {"source": (TL, BL, BR, TR)},
    "anti_transpose": {"source": (BR, TR, TL, BL)},
    "near_colour_outermost": {"order": "near"},
    "cycle_instead_of_saturate": {"tail": "cycle"},
    "flat_far_colour": {"tail": "flat_far"},
    "flat_near_colour": {"tail": "flat_near"},
    "corner_distance_layers": {"depth": "corner"},
    "one_key_whole_frame": {"fill": "single"},
}


# --------------------------------------------------------------------------
# Strict structural analysis used by the validator and self-test.
# --------------------------------------------------------------------------
def _analyze(grid):
    grid = np.asarray(grid)
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    _check(len(counts) == 1 or counts[order[0]] > counts[order[1]],
           "background must be the unique most frequent colour")
    bg = int(values[order[0]])
    frames = _find_frames(grid, bg)
    _check(frames, "every input needs at least one frame")
    frame_colors = {int(grid[f[0], f[1]]) for f in frames}
    _check(len(frame_colors) == 1, "all frames of a pair share one colour")
    frame_color = frame_colors.pop()
    owned = {}
    info = []
    for frame in frames:
        top, left, bottom, right = frame
        hi, wi = bottom - top - 1, right - left - 1
        _check(hi >= 2 and wi >= 2 and hi % 2 == 0 and wi % 2 == 0,
               "frame interiors must have even height and width")
        _check(np.all(grid[top + 1:bottom, left + 1:right] == bg),
               "frame interiors must be blank")
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if r in (top, bottom) or c in (left, right):
                    owned[(r, c)] = frame
        keys = _read_keys(grid, bg, frame)
        present = [k for k in keys if k]
        _check(present, "each frame needs at least one key")
        qh, qw = hi // 2, wi // 2
        for c, key in enumerate(keys):
            _check(len(set(key)) == len(key), "key colours must be distinct")
            _check(frame_color not in key and bg not in key,
                   "key colours must differ from frame and background")
            _check(len(key) <= min(qh, qw, MAX_KEY),
                   "every key must fit inside one quadrant")
            for cell in _diagonal(frame, c, len(key)):
                _check(cell not in owned, "key cells overlap another object")
                owned[cell] = frame
        _check(len(set(present)) == len(present), "keys of one frame must differ")
        _check(len({k[0] for k in present}) == len(present),
               "outermost key colours of one frame must differ")
        # Moved keys must stay on the canvas.
        for c in range(4):
            key = keys[(c - 1) % 4]
            for r, col in _diagonal(frame, c, len(key)):
                _check(0 <= r < h and 0 <= col < w, "a turned key leaves the canvas")
        info.append({"frame": frame, "keys": keys, "qh": qh, "qw": qw})
    nonbg = {(int(r), int(c)) for r, c in np.argwhere(grid != bg)}
    _check(nonbg == set(owned), "stray cells outside frames and keys")
    # Footprints (frame plus incoming and outgoing keys) must be separated.
    boxes = []
    for item in info:
        frame, keys = item["frame"], item["keys"]
        cells = [(frame[0], frame[1]), (frame[2], frame[3])]
        for c in range(4):
            for key in (keys[c], keys[(c - 1) % 4]):
                cells.extend(_diagonal(frame, c, len(key)))
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        boxes.append((min(rows), min(cols), max(rows), max(cols)))
        item["box"] = boxes[-1]
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            r0, c0, r1, c1 = boxes[a]
            s0, d0, s1, d1 = boxes[b]
            _check(r1 + 1 < s0 or s1 + 1 < r0 or c1 + 1 < d0 or d1 + 1 < c0,
                   "frame footprints must be separated by background")
    return {"background": bg, "frame_color": frame_color, "frames": info,
            "shape": (h, w)}


# --------------------------------------------------------------------------
# Generation.
# --------------------------------------------------------------------------
def _validate_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")


def _roles(rng):
    background = 0 if rng.random() < 0.4 else int(rng.integers(1, 10))
    frame = int(rng.choice([c for c in range(10) if c != background]))
    return background, frame


def _config(rng):
    shift = SHIFTS[int(rng.integers(len(SHIFTS)))]
    return {
        "shift": shift,
        "palette": "constant" if rng.random() < 0.5 else "variable",
        "roles": _roles(rng),
        "missing": shift == "sparse" or rng.random() < 0.3,
        "multi": shift == "crowd" or rng.random() < 0.45,
        # Scale tests use 4-5 colour keys, so their training shows 3-colour layering.
        "train_max_len": 3 if shift == "scale" else int(rng.choice((2, 3, 4), p=(0.3, 0.4, 0.3))),
        "square_only": shift == "aspect",
    }


def _quad_dims(rng, low, high, shape):
    if shape == "square":
        q = int(rng.integers(low, high + 1))
        return q, q
    if shape == "rect":
        for _ in range(50):
            a, b = int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
            if a != b:
                return a, b
        return low, high if high != low else low + 1
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _plan_frame(rng, cfg, role):
    """Return (qh, qw, lengths[4]) where a length of 0 means no key."""
    tmax = cfg["train_max_len"]
    shift = cfg["shift"]
    shape = "square" if cfg["square_only"] else ("square" if rng.random() < 0.45 else "any")
    force_long = 0  # minimum length one key must reach
    saturation = False
    drop = 0
    if role == "anchor":
        # All four keys, one of them layered (3 colours when the puzzle allows
        # it) inside a quadrant deep enough for its last colour to saturate.
        force_long, saturation = min(3, tmax), True
        qh, qw = _quad_dims(rng, force_long + 1, 4 if shift == "crowd" else 5, shape)
        cap = min(qh, qw, tmax)
    elif role == "train":
        low = 1 if rng.random() < 0.2 else 2
        qh, qw = _quad_dims(rng, low, 4 if shift == "crowd" else 5, shape)
        cap = min(qh, qw, tmax)
        if cfg["missing"] and rng.random() < 0.35:
            drop = int(rng.integers(1, 3))
    elif role == "missing_demo":
        qh, qw = _quad_dims(rng, 2, 4, shape)
        cap = min(qh, qw, tmax)
        drop = int(rng.integers(1, 3))
        force_long = 2
    elif role == "test":
        if shift == "scale":
            qh, qw = _quad_dims(rng, 4, 6, "any" if rng.random() < 0.6 else "square")
            cap = min(qh, qw, MAX_KEY)
            force_long = 4
        elif shift == "crowd":
            qh, qw = _quad_dims(rng, 2, 3, "any")
            cap = min(qh, qw, tmax)
            force_long = 2
        elif shift == "aspect":
            a = int(rng.integers(2, 4))
            b = int(rng.integers(max(2 * a, 4), 8))
            qh, qw = (a, b) if rng.random() < 0.5 else (b, a)
            cap = min(qh, qw, tmax)
            force_long = 2
        elif shift == "sparse":
            qh, qw = _quad_dims(rng, 2, 5, shape)
            cap = min(qh, qw, tmax)
            drop = int(rng.choice((1, 2, 3), p=(0.45, 0.4, 0.15)))
            force_long = 2
        else:
            qh, qw = _quad_dims(rng, 2, 6, shape)
            cap = min(qh, qw, tmax)
            if cfg["missing"] and rng.random() < 0.4:
                drop = int(rng.integers(1, 3))
            force_long = 2
    elif role == "test_small":
        qh, qw = _quad_dims(rng, 1, 3, "any")
        cap = min(qh, qw, tmax)
    else:
        raise ValueError(role)
    if force_long > cap:
        raise _Retry()
    present = list(range(4))
    if drop:
        absent = rng.choice(4, size=drop, replace=False)
        present = [c for c in present if c not in set(int(x) for x in absent)]
    lengths = [0, 0, 0, 0]
    weights = np.arange(1, cap + 1, dtype=float)
    for c in present:
        lengths[c] = int(rng.choice(np.arange(1, cap + 1), p=weights / weights.sum()))
    if force_long:
        c = present[int(rng.integers(len(present)))]
        hi = cap if not saturation else min(cap, min(qh, qw) - 1)
        if hi < force_long:
            raise _Retry()
        lengths[c] = int(rng.integers(force_long, hi + 1))
    return qh, qw, lengths


def _pick_keys(rng, lengths, available):
    for _ in range(200):
        keys = []
        for n in lengths:
            keys.append(tuple(int(x) for x in rng.choice(available, size=n, replace=False))
                        if n else ())
        present = [k for k in keys if k]
        if len({k[0] for k in present}) == len(present) and len(set(present)) == len(present):
            return keys
    raise _Retry()


def _extents(qh, qw, keys):
    room = [max(len(keys[c]), len(keys[(c - 1) % 4])) for c in range(4)]
    up = max(room[TL], room[TR])
    down = max(room[BR], room[BL])
    lft = max(room[TL], room[BL])
    rgt = max(room[TR], room[BR])
    return up, down, lft, rgt


def _make_pair(rng, cfg, roles_list, palette):
    background, frame_color = palette
    available = [c for c in range(10) if c not in (background, frame_color)]
    for _ in range(60):
        plans = []
        try:
            for role in roles_list:
                qh, qw, lengths = _plan_frame(rng, cfg, role)
                keys = _pick_keys(rng, lengths, available)
                plans.append((qh, qw, keys))
        except _Retry:
            continue
        boxes = []
        for qh, qw, keys in plans:
            up, down, lft, rgt = _extents(qh, qw, keys)
            boxes.append((up, down, lft, rgt, 2 * qh + 2, 2 * qw + 2))
        need_h = max(b[0] + b[1] + b[4] for b in boxes)
        need_w = max(b[2] + b[3] + b[5] for b in boxes)
        if need_h > 30 or need_w > 30:
            continue
        n = len(plans)
        for _ in range(12):
            slack = 7 if n == 1 else 14
            height = int(rng.integers(need_h, min(30, need_h + slack) + 1))
            width = int(rng.integers(need_w, min(30, need_w + slack) + 1))
            if n > 1:
                height = max(height, min(30, need_h + int(rng.integers(4, 14))))
                width = max(width, min(30, need_w + int(rng.integers(4, 14))))
            placed = []
            ok = True
            for (qh, qw, keys), (up, down, lft, rgt, fh, fw) in zip(plans, boxes):
                spots = []
                rows = range(up, height - down - fh + 1)
                cols = range(lft, width - rgt - fw + 1)
                if not len(rows) or not len(cols):
                    ok = False
                    break
                for _ in range(80):
                    top = int(rng.integers(rows.start, rows.stop))
                    left = int(rng.integers(cols.start, cols.stop))
                    box = (top - up, left - lft, top + fh - 1 + down, left + fw - 1 + rgt)
                    if all(box[2] + 1 < o[0] or o[2] + 1 < box[0]
                           or box[3] + 1 < o[1] or o[3] + 1 < box[1]
                           for o in (p[1] for p in placed)):
                        spots.append((top, left, box))
                        break
                if not spots:
                    ok = False
                    break
                top, left, box = spots[0]
                placed.append(((top, left, top + fh - 1, left + fw - 1, keys), box))
            if not ok:
                continue
            grid = np.full((height, width), background, dtype=int)
            for (top, left, bottom, right, keys), _box in placed:
                grid[top, left:right + 1] = frame_color
                grid[bottom, left:right + 1] = frame_color
                grid[top:bottom + 1, left] = frame_color
                grid[top:bottom + 1, right] = frame_color
                frame = (top, left, bottom, right)
                for c, key in enumerate(keys):
                    for i, cell in enumerate(_diagonal(frame, c, len(key))):
                        grid[cell] = key[len(key) - 1 - i]
            try:
                _analyze(grid)
            except AssertionError:
                continue
            return {"input": grid, "output": solve(grid)}
    raise _Retry()


def _train_roles(rng, cfg, num_train):
    roles = [["anchor"]]
    support = []
    if cfg["multi"]:
        support = ["missing_demo" if cfg["missing"] else "train", "train"]
    elif cfg["missing"]:
        support = ["missing_demo"]
    if support:
        roles.append(support)
    while len(roles) < num_train:
        count = 2 if cfg["multi"] and rng.random() < 0.5 else 1
        roles.append(["train"] * count)
    order = rng.permutation(len(roles))
    return [roles[int(i)] for i in order]


def _test_roles(rng, cfg, num_test):
    result = []
    for _ in range(num_test):
        shift = cfg["shift"]
        if shift == "crowd":
            result.append(["test", "test_small", "test_small"])
        elif shift == "scale":
            result.append(["test"])
        elif cfg["multi"] and rng.random() < 0.6:
            result.append(["test", "test_small" if shift != "aspect" else "test"])
        else:
            result.append(["test"])
    return result


def _build(rng, cfg, num_train, num_test):
    puzzle = {"train": [], "test": []}
    seen = set()
    plan = (("train", _train_roles(rng, cfg, num_train)),
            ("test", _test_roles(rng, cfg, num_test)))
    for split, role_lists in plan:
        for roles in role_lists:
            palette = cfg["roles"] if cfg["palette"] == "constant" else _roles(rng)
            for _ in range(20):
                pair = _make_pair(rng, cfg, roles, palette)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise _Retry()
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _validate_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    cfg = _config(rng)
    last = None
    for _ in range(40):
        try:
            puzzle = _build(rng, cfg, int(num_train), int(num_test))
            validate_generated_puzzle(puzzle)
            return puzzle
        except (_Retry, AssertionError) as error:
            last = error
            continue
    raise RuntimeError(f"could not generate a puzzle for seed {seed}: {last}")


# --------------------------------------------------------------------------
# Validation.
# --------------------------------------------------------------------------
def _grid(value, label):
    array = np.asarray(value)
    _check(array.ndim == 2, f"{label} must be two-dimensional")
    _check(1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30,
           f"{label} must be 1..30 on each side")
    _check(np.issubdtype(array.dtype, np.integer), f"{label} must contain integers")
    _check(int(array.min()) >= 0 and int(array.max()) <= 9, f"{label} colours must be 0..9")
    return array.astype(int)


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must have exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _check(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
           "illegal pair counts")
    seen = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   f"{label} must have exactly input and output")
            inp = _grid(pair["input"], label + ".input")
            out = _grid(pair["output"], label + ".output")
            _check(inp.shape == out.shape, f"{label} must keep the canvas size")
            key = (inp.shape, inp.tobytes())
            _check(key not in seen, "inputs must be distinct within a puzzle")
            seen.add(key)
            info = _analyze(inp)
            _check(np.array_equal(out, solve(inp)), f"{label} output violates the rule")
            _check(not np.array_equal(inp, out), f"{label} is an identity pair")
            differs = {name: not np.array_equal(out, _apply(inp, **kwargs))
                       for name, kwargs in SHORTCUTS.items()}
            facts[split].append({"info": info, "differs": differs})
    # Training must expose every ingredient that the tests rely on.
    train_frames = [f for p in facts["train"] for f in p["info"]["frames"]]
    test_frames = [f for p in facts["test"] for f in p["info"]["frames"]]
    _check(any(all(f["keys"]) for f in train_frames),
           "training needs a frame with all four keys")
    _check(any(2 <= len(f["keys"][(c - 1) % 4]) < min(f["qh"], f["qw"])
               for f in train_frames for c in range(4)),
           "training needs a layered key whose last colour visibly saturates")
    longest_train = max(len(k) for f in train_frames for k in f["keys"])
    longest_test = max(len(k) for f in test_frames for k in f["keys"])
    _check(longest_train >= min(3, longest_test),
           "tests with 3+ colour keys need 3-colour layering in training")
    if any(not all(f["keys"]) for f in test_frames):
        _check(any(not all(f["keys"]) for f in train_frames),
               "tests with missing keys need a training frame with a missing key")
    if any(len(p["info"]["frames"]) > 1 for p in facts["test"]):
        _check(any(len(p["info"]["frames"]) > 1 for p in facts["train"]),
               "tests with several frames need a multi-frame training pair")
    for name in SHORTCUTS:
        _check(any(p["differs"][name] for p in facts["train"]),
               f"training does not refute the shortcut {name}")
    for p in facts["test"]:
        _check(all(p["differs"][name] for name in (
            "no_turn", "counterclockwise", "flat_far_colour", "flat_near_colour",
            "near_colour_outermost")),
               "each test must depend on both the turn and the layering")


# --------------------------------------------------------------------------
# Self-test.
# --------------------------------------------------------------------------
def _signature(puzzle):
    return tuple((split, tuple((np.asarray(p["input"]).tobytes(), np.asarray(p["output"]).tobytes(),
                                np.asarray(p["input"]).shape) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test() -> None:
    # A hand-built pair pins down the exact rule.
    grid = np.zeros((14, 14), dtype=int)
    top, left, bottom, right = 3, 3, 10, 10  # 8x8 frame, 6x6 interior
    grid[top, left:right + 1] = grid[bottom, left:right + 1] = 5
    grid[top:bottom + 1, left] = grid[top:bottom + 1, right] = 5
    frame = (top, left, bottom, right)
    keys = [(1, 2), (3,), (4, 6, 7), (8,)]
    for c, key in enumerate(keys):
        for i, cell in enumerate(_diagonal(frame, c, len(key))):
            grid[cell] = key[len(key) - 1 - i]
    out = solve(grid)
    _check(tuple(out[1, 1:3]) == (0, 0) and out[2, 2] == 8, "TL receives the BL key")
    _check(out[2, 11] == 2 and out[1, 12] == 1, "TR receives the TL key in order")
    _check(out[11, 11] == 3 and out[12, 12] == 0, "BR receives the TR key")
    _check((out[11, 2], out[12, 1], out[13, 0]) == (7, 6, 4), "BL receives the BR key")
    interior = out[4:10, 4:10]
    expected = np.array([
        [8, 8, 8, 1, 1, 1],
        [8, 8, 8, 2, 2, 1],
        [8, 8, 8, 2, 2, 1],
        [4, 6, 7, 3, 3, 3],
        [4, 6, 6, 3, 3, 3],
        [4, 4, 4, 3, 3, 3],
    ])
    _check(np.array_equal(interior, expected), "quadrant layering is wrong")
    _check(np.array_equal(out[top, left:right + 1], grid[top, left:right + 1]), "frame changed")

    # Count handling.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7, a, b)
            _check(len(p["train"]) == a and len(p["test"]) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(3, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    _check(_signature(generate_puzzle(123, 5, 2)) == _signature(generate_puzzle(123, 5, 2)),
           "generation is not deterministic")

    # Variation over 42 seeds.
    signatures, shapes, quads, lengths = set(), set(), set(), set()
    frame_counts, palettes, shifts = set(), set(), set()
    missing_tests = multi_tests = rect_tests = longer_tests = 0
    variable_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        local_palettes = set()
        train_len = 0
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(np.asarray(pair["input"]))
                shapes.add(info["shape"])
                frame_counts.add(len(info["frames"]))
                local_palettes.add((info["background"], info["frame_color"]))
                for f in info["frames"]:
                    quads.add((f["qh"], f["qw"]))
                    lengths.update(len(k) for k in f["keys"])
                    if split == "train":
                        train_len = max(train_len, max(len(k) for k in f["keys"]))
                if split == "test":
                    missing_tests += any(not all(f["keys"]) for f in info["frames"])
                    multi_tests += len(info["frames"]) > 1
                    rect_tests += any(f["qh"] != f["qw"] for f in info["frames"])
                    longer_tests += any(max(len(k) for k in f["keys"]) > train_len
                                        for f in info["frames"])
        palettes.update(local_palettes)
        variable_palette += len(local_palettes) > 1
        shifts.add(_config(np.random.default_rng(seed))["shift"])
    _check(len(signatures) == 42, "seeds repeat puzzles")
    _check(len(shapes) >= 40, "canvas sizes lack variety")
    _check(len(quads) >= 20, "frame shapes lack variety")
    _check({0, 1, 2, 3, 4}.issubset(lengths), "key lengths lack variety")
    _check({1, 2, 3}.issubset(frame_counts), "frame counts lack variety")
    _check(len(palettes) >= 25 and variable_palette >= 8, "palettes lack variety")
    _check(shifts == set(SHIFTS), "train/test shift modes are missing")
    _check(missing_tests >= 5 and multi_tests >= 6 and rect_tests >= 10 and longer_tests >= 4,
           "test variation is too narrow")
