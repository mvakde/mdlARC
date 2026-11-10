"""Match adjacent guides, orient the complete crop, then mask uniform rows.

Only NumPy and the standard library are used. No generation state is read by
solve(). Diagnostics are returned in memory; no grids, traces or logs are saved.
"""
from __future__ import annotations

import hashlib
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
POSITIVE = 5
BACKGROUND = 0
ROTATIONS = (
    np.array(((1, 0), (0, 1)), dtype=int),
    np.array(((0, -1), (1, 0)), dtype=int),
    np.array(((-1, 0), (0, -1)), dtype=int),
    np.array(((0, 1), (-1, 0)), dtype=int),
)
MATRICES = ROTATIONS + tuple(m @ np.diag((1, -1)) for m in ROTATIONS)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and a.dtype.kind in "iu", "grid must be 2D integers")
    _require(all(1 <= n <= 30 for n in a.shape), "grid dimensions outside 1..30")
    _require(np.all((a >= 0) & (a <= 9)), "illegal color")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8, copy=False).tobytes()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _adjacent(i):
    return i, (i + 1) % 4


def _side(a, side):
    return (a[0, :], a[:, -1], a[-1, :], a[:, 0])[side]


def _transform(a, code):
    """Construction path: rotations, optionally preceded by left/right flip."""
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _side_map(code):
    return tuple(NORMALS.index(tuple(MATRICES[code] @ np.array(n))) for n in NORMALS)


def _arrangement(outer, inner, reverse):
    targets = _adjacent(outer)[::(-1 if reverse else 1)]
    codes = [code for code in range(8)
             if all(_side_map(code)[s] == t for s, t in zip(_adjacent(inner), targets))]
    _require(len(codes) == 1, "side correspondence not unique")
    return codes[0]


def _components(a, color):
    visited = np.zeros(a.shape, dtype=bool)
    for rr, cc in np.argwhere(a == color):
        start = int(rr), int(cc)
        if visited[start]:
            continue
        stack, cells = [start], []
        visited[start] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in NORMALS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]
                        and not visited[nr, nc] and a[nr, nc] == color):
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        yield cells


def _decode(input_grid):
    a = _grid(input_grid)
    H, W = a.shape
    _require(min(H, W) >= 9, "canvas too small for this family")
    _require(not np.any(a == POSITIVE), "5 is reserved for the output marker")
    outer = {}
    for s in range(4):
        core = _side(a, s)[1:-1]
        if len(core) and int(core[0]) != BACKGROUND and np.all(core == core[0]):
            outer[s] = int(core[0])
    _require(len(outer) == 2 and len(set(outer.values())) == 2,
             "need two distinct complete outer controls")
    _require(any(set(outer) == set(_adjacent(i)) for i in range(4)),
             "outer controls must be adjacent")
    bars = []
    for color in outer.values():
        candidates = []
        for cells in _components(a, color):
            if any(r in (0, H - 1) or c in (0, W - 1) for r, c in cells):
                continue
            rows = sorted({r for r, _ in cells})
            cols = sorted({c for _, c in cells})
            if len(rows) == 1 and len(cols) >= 4 and cols == list(range(cols[0], cols[-1] + 1)):
                candidates.append(("H", rows[0], cols[0], cols[-1], color))
            elif len(cols) == 1 and len(rows) >= 4 and rows == list(range(rows[0], rows[-1] + 1)):
                candidates.append(("V", cols[0], rows[0], rows[-1], color))
            else:
                raise AssertionError("an interior marker component is not a guide")
        _require(len(candidates) == 1, "each control needs exactly one inner guide")
        bars.extend(candidates)
    horizontal = [b for b in bars if b[0] == "H"]
    vertical = [b for b in bars if b[0] == "V"]
    _require(len(horizontal) == len(vertical) == 1, "guides must be perpendicular")
    _, hr, c, ce, hc = horizontal[0]
    _, vc, r, re, vc_color = vertical[0]
    h, w = re - r + 1, ce - c + 1
    _require(hr in (r - 1, re + 1) and vc in (c - 1, ce + 1),
             "guide endpoints do not define adjacent rectangle sides")
    guides = {0 if hr == r - 1 else 2: hc,
              3 if vc == c - 1 else 1: vc_color}
    _require(r >= 2 and c >= 2 and r + h <= H - 2 and c + w <= W - 2,
             "payload must be separated from outer controls")
    p = a[r:r + h, c:c + w].copy()
    _require(not set(np.unique(p)) & set(outer.values()), "payload contains guide colors")
    r0, r1 = r - (0 in guides), r + h + (2 in guides)
    c0, c1 = c - (3 in guides), c + w + (1 in guides)
    _require(r0 >= 2 and c0 >= 2 and r1 <= H - 2 and c1 <= W - 2,
             "guides need a background moat")
    ring = np.concatenate((a[r0 - 1, c0 - 1:c1 + 1], a[r1, c0 - 1:c1 + 1],
                           a[r0:r1, c0 - 1], a[r0:r1, c1]))
    _require(np.all(ring == BACKGROUND), "guide rectangle lacks a clear moat")
    for s in range(4):
        segment = (a[r - 1, c:c + w], a[r:r + h, c + w],
                   a[r + h, c:c + w], a[r:r + h, c - 1])[s]
        _require(np.all(segment == guides.get(s, BACKGROUND)), "unexpected guide-side material")
    matrix = np.zeros((2, 2), dtype=int)
    for s, color in guides.items():
        target = next(t for t in outer if outer[t] == color)
        matrix += np.outer(NORMALS[target], NORMALS[s])
    codes = [i for i, m in enumerate(MATRICES) if np.array_equal(matrix, m)]
    _require(len(codes) == 1, "no unique dihedral transform")
    return dict(payload=p, bounds=(r, c, h, w), guides=guides, outer=outer,
                matrix=matrix, code=codes[0])


def _oriented(info):
    """Solver path: signed coordinate movement from the two matched normals."""
    p = info["payload"]
    coords = info["matrix"] @ np.indices(p.shape).reshape(2, -1)
    coords -= coords.min(axis=1, keepdims=True)
    a = np.empty(tuple(coords.max(axis=1) + 1), dtype=p.dtype)
    a[coords[0], coords[1]] = p.ravel()
    return a


def _row_output(a):
    mask = np.all(a == a[:, :1], axis=1)
    return np.repeat(np.where(mask, POSITIVE, BACKGROUND).astype(np.int8)[:, None],
                     a.shape[1], axis=1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the entire answer from this input only, without seed or examples."""
    return _row_output(_oriented(_decode(input_grid)))


def _palette(rng):
    colors = rng.permutation([c for c in range(1, 10) if c != POSITIVE]).tolist()
    return dict(markers=colors[:2], noise=colors[2], payload=colors[3:])


def _training_payload(rng, shape, colors):
    """Sparse exceptions: both row and column predicates are informative.

    Three singleton columns expose first/interior/last cell comparisons after
    a quarter-turn. A fourth column with two exceptions makes the two teaching
    answers have different negative counts (four columns versus five rows).
    Blank side columns force full guide-defined extraction, not an ink crop.
    """
    h, w = shape
    row_mid = _pick(rng, [r for r in range(1, h - 1) if 2 * r != h - 1])
    rows = [0, row_mid, h - 1]
    for _ in range(40):
        cols = sorted(int(x) for x in rng.choice(np.arange(1, w - 1), size=3, replace=False))
        if set(cols) != {w - 1 - c for c in cols}:
            break
    else:
        cols = [1, 2, w - 2]
    rng.shuffle(cols)
    a = np.zeros(shape, dtype=np.int8)
    for r, c in zip(rows, cols):
        a[r, c] = int(_pick(rng, colors))
    extra_col = int(_pick(rng, [c for c in range(1, w - 1) if c not in cols]))
    extra_rows = rng.choice([r for r in range(1, h - 1) if r != row_mid], size=2, replace=False)
    for r in extra_rows:
        a[int(r), extra_col] = int(_pick(rng, colors))
    # Keep the column mask asymmetric so opposite quarter-turn directions are
    # distinguishable even after the colors have been discarded.
    active = np.any(a != 0, axis=0)
    if np.array_equal(active, active[::-1]):
        candidates = [c for c in range(1, w - 1) if not active[c]]
        replacement = next(c for c in candidates if
                           set(cols + [c]) != {w - 1 - x for x in cols + [c]})
        a[:, replacement] = a[:, extra_col]
        a[:, extra_col] = 0
    return a


def _mixed_row(rng, width, colors, style=None):
    a, b = (int(x) for x in rng.choice(colors, size=2, replace=False))
    style = style or _pick(rng, ("exception", "alternating", "runs", "random", "balanced"))
    row = np.full(width, a, dtype=np.int8)
    if style == "exception":
        row[int(rng.integers(width))] = b
    elif style == "alternating":
        row[1::2] = b
    elif style == "runs":
        row[int(rng.integers(1, width)):] = b
    elif style == "balanced":
        row[width // 2:] = b
        rng.shuffle(row)
    else:
        row[:] = rng.choice(colors, size=width)
        if np.all(row == row[0]):
            row[-1] = b if row[0] == a else a
    return row


def _random_payload(rng, shape, code, palette, category):
    h, w = shape
    colors = [BACKGROUND] + palette["payload"][:int(rng.integers(1, 6))]
    if category == "cross":
        baseline = int(_pick(rng, colors))
        p = np.full(shape, baseline, dtype=np.int8)
        # Reserve at least one full row and column; modify a variable submatrix.
        rr = rng.choice(h, size=int(rng.integers(1, h)), replace=False)
        cc = rng.choice(w, size=int(rng.integers(1, w)), replace=False)
        alternate = [v for v in colors if v != baseline]
        for r in rr:
            for c in cc:
                if rng.random() < .65:
                    p[int(r), int(c)] = int(_pick(rng, alternate))
        p[int(rr[0]), int(cc[0])] = int(_pick(rng, alternate))
        return p
    # Construct in the eventual row frame, then use the inverse symmetry.
    out_shape = _transform(np.empty(shape, dtype=np.int8), code).shape
    oh, ow = out_shape
    p = np.empty(out_shape, dtype=np.int8)
    probability = float(rng.uniform(.2, .8))
    mask = rng.random(oh) < probability
    if category == "all_positive":
        mask[:] = True
    elif category == "all_negative":
        mask[:] = False
    else:
        mask[int(rng.integers(oh))] = True
        mask[(int(np.flatnonzero(mask)[0]) + 1) % oh] = False
    for r in range(oh):
        if mask[r]:
            p[r] = int(_pick(rng, colors))
        elif r and not mask[r - 1] and rng.random() < .15:
            p[r] = p[r - 1]
        else:
            p[r] = _mixed_row(rng, ow, colors)
    inverse = next(i for i, m in enumerate(MATRICES)
                   if np.array_equal(m, MATRICES[code].T))
    return _transform(p, inverse)


def _frame_coordinate(shape, side, pos, depth=0):
    H, W = shape
    return ((depth, pos), (pos, W - 1 - depth),
            (H - 1 - depth, pos), (pos, depth))[side]


def _render(rng, p, outer_index, inner_index, reverse, palette, geometry=None):
    h, w = p.shape
    inner = _adjacent(inner_index)
    targets = _adjacent(outer_index)[::(-1 if reverse else 1)]
    guides = dict(zip(inner, palette["markers"]))
    outer = dict(zip(targets, palette["markers"]))
    if geometry is None:
        H, W = int(rng.integers(h + 5, 31)), int(rng.integers(w + 5, 31))
        if rng.random() < .1:
            H = W = 30
        r = int(rng.integers(2 + (0 in inner), H - h - 1 - (2 in inner)))
        c = int(rng.integers(2 + (3 in inner), W - w - 1 - (1 in inner)))
    else:
        H, W, r, c = geometry
    a = np.zeros((H, W), dtype=np.int8)
    protected = np.zeros_like(a, dtype=bool)
    r0, r1 = r - (0 in inner), r + h + (2 in inner)
    c0, c1 = c - (3 in inner), c + w + (1 in inner)
    protected[r0 - 1:r1 + 1, c0 - 1:c1 + 1] = True
    if rng.random() < .7:
        for s in range(4):
            if s in outer:
                continue
            step = int(rng.integers(2, 7))
            for pos in range(int(rng.integers(1, step)), len(_side(a, s)) - 1, step):
                a[_frame_coordinate(a.shape, s, pos)] = palette["noise"]
    for s in rng.permutation(list(outer)):
        _side(a, int(s))[:] = outer[int(s)]
    if rng.random() < .55:
        for s, color in outer.items():
            for _ in range(int(rng.integers(1, 4))):
                pos = int(rng.integers(2, len(_side(a, s)) - 2))
                for depth in range(1, int(rng.integers(2, 4))):
                    cell = _frame_coordinate(a.shape, s, pos, depth)
                    if protected[cell] or a[cell] not in (BACKGROUND, color):
                        break
                    a[cell] = color
    a[r:r + h, c:c + w] = p
    for s, color in guides.items():
        (a[r - 1, c:c + w], a[r:r + h, c + w],
         a[r + h, c:c + w], a[r:r + h, c - 1])[s][:] = color
    code = _arrangement(outer_index, inner_index, reverse)
    return {"input": a, "output": _row_output(_transform(p, code))}


def _swap_controls(pair):
    a = pair["input"].copy()
    info = _decode(a)
    colors = list(info["outer"].values())
    replacements = []
    for color, replacement in zip(colors, colors[::-1]):
        for component in _components(a, color):
            if any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1) for r, c in component):
                replacements.extend((r, c, replacement) for r, c in component)
    for r, c, replacement in replacements:
        a[r, c] = replacement
    return {"input": a, "output": solve(a)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    held_outer = int(rng.integers(4))
    training_outers = [i for i in range(4) if i != held_outer]
    first = _pick(rng, [(o, i, rev) for o in training_outers for i in range(4)
                        for rev in (False, True) if _arrangement(o, i, rev) in (1, 3)])
    palette = _palette(rng)
    h, w = int(rng.integers(7, 14)), int(rng.integers(7, 14))
    if h == w:
        w = 7 if h != 7 else 8
    p = _training_payload(rng, (h, w), palette["payload"])
    a = _render(rng, p, *first, palette)
    b = _swap_controls(a)
    info = _decode(b["input"])
    # A bijective recoloring of only the payload: uniform nonzero rows teach
    # the same positive label as the first demonstration's uniform zero rows.
    lookup = np.arange(10, dtype=np.int8)
    base = int(_pick(rng, palette["payload"]))
    lookup[0], lookup[base] = base, 0
    r, c, h, w = info["bounds"]
    b["input"][r:r + h, c:c + w] = lookup[p]
    b["output"] = solve(b["input"])
    puzzle = {"train": [a, b], "test": []}
    if nt >= 3:
        # Exact causal contrast: only border-connected control identities swap.
        puzzle["train"].append(_swap_controls(a))
    used = {_key(pair["input"]) for pair in puzzle["train"]}
    test_answers = set()
    for split, count in (("train", nt - len(puzzle["train"])), ("test", ne)):
        for index in range(count):
            for attempt in range(64):
                o = int(_pick(rng, training_outers)) if split == "train" else held_outer
                i, rev = int(rng.integers(4)), bool(rng.integers(2))
                code = _arrangement(o, i, rev)
                shape = (int(rng.integers(4, 26)), int(rng.integers(4, 26)))
                if rng.random() < .18:
                    shape = (shape[0], shape[0])
                pal = _palette(rng)
                category = _pick(rng, ("mixed", "mixed", "mixed", "cross", "cross",
                                       "all_positive", "all_negative"))
                # The first test is a nontrivial recombination of known mechanisms.
                if split == "test" and index == 0:
                    category = _pick(rng, ("mixed", "cross"))
                payload = _random_payload(rng, shape, code, pal, category)
                pair = _render(rng, payload, o, i, rev, pal)
                if _key(pair["input"]) in used:
                    continue
                if split == "test" and _key(pair["output"]) in test_answers:
                    continue
                if split == "test" and index == 0:
                    # Identity and left/right reflection necessarily share
                    # their row mask. Keep these valid family members too.
                    if code not in (0, 4) and np.array_equal(pair["output"], _row_output(payload)):
                        continue
                    if len(np.unique(pair["output"])) != 2:
                        continue
                break
            else:
                raise RuntimeError("bounded construction exhausted")
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            if split == "test":
                test_answers.add(_key(pair["output"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _exception_style(row):
    values, counts = np.unique(row, return_counts=True)
    if len(values) != 2 or sorted(counts.tolist()) != [1, len(row) - 1]:
        return None
    j = int(np.flatnonzero(row == values[int(np.argmin(counts))])[0])
    return "first" if j == 0 else "last" if j == len(row) - 1 else "interior"


def _equality_pattern(a):
    labels = {}
    return tuple(labels.setdefault(int(v), len(labels)) for v in a.flat)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
    seen, records, styles, uniform_colors = set(), {"train": [], "test": []}, set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _decode(a)
            oriented = _oriented(info)
            # Separate paths agree about every transformed cell before masking.
            _require(np.array_equal(oriented, _transform(info["payload"], info["code"])),
                     "coordinate and array transforms disagree")
            _require(out.shape == oriented.shape, "wrong crop extent")
            _require(set(np.unique(out)) <= {BACKGROUND, POSITIVE}, "output not a binary mask")
            for row, result in zip(oriented, out):
                expected = POSITIVE if len(set(int(v) for v in row)) == 1 else BACKGROUND
                _require(np.all(result == expected), "wrong whole-row equality replacement")
                if split == "train":
                    if expected == POSITIVE:
                        uniform_colors.add(int(row[0]))
                    else:
                        style = _exception_style(row)
                        if style:
                            styles.add(style)
            records[split].append(info)
    train, test = records["train"], records["test"]
    _require(styles == {"first", "interior", "last"}, "missing full-row mismatch contrasts")
    _require(0 in uniform_colors and len(uniform_colors) >= 2, "need uniform zero and nonzero positives")
    _require(any(any(np.all(_side(t["payload"], s) == 0) for s in range(4)) for t in train),
             "training must demonstrate blank-edge crop extent")
    x, y = train[:2]
    _require(x["bounds"] == y["bounds"] and x["guides"] == y["guides"]
             and set(x["outer"]) == set(y["outer"]) and x["outer"] != y["outer"],
             "missing correspondence contrast")
    _require(_equality_pattern(x["payload"]) == _equality_pattern(y["payload"]),
             "teaching payloads must have matching equality structure")
    _require(not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"]),
             "correspondence contrast must affect the answer")
    negative_counts = {int(np.count_nonzero(np.any(np.asarray(p["output"]) == BACKGROUND, axis=1)))
                       for p in puzzle["train"][:2]}
    _require(len(negative_counts) == 2, "teaching needs different negative-row counts")
    _require(all(set(t["outer"]) not in [set(a["outer"]) for a in train] for t in test),
             "tests must hold out an outer corner")
    _require(any(not np.array_equal(_row_output(t["payload"]), pair["output"])
                 for t, pair in zip(train, puzzle["train"])), "omitting orientation must change training")
    _require(all(not np.array_equal(_oriented(t), pair["output"])
                 for s in ("train", "test") for t, pair in zip(records[s], puzzle[s])),
             "omitting row classification must change answers")
    _require(len({_key(np.asarray(pair["output"])) for pair in puzzle["test"]}) == ne,
             "multiple tests must have distinct answers")
    if nt >= 3:
        z = train[2]
        _require(np.array_equal(x["payload"], z["payload"]) and x["guides"] == z["guides"]
                 and x["bounds"] == z["bounds"] and x["outer"] != z["outer"],
                 "missing exact control-only contrast")
        _require(np.array_equal(puzzle["train"][1]["output"], puzzle["train"][2]["output"]),
                 "payload recoloring must not change equality mask")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def self_test():
    """Bounded streaming checks on 42 seeds; return compact aggregate evidence."""
    shapes, payload_shapes, masks, palettes, fingerprints = set(), set(), set(), set(), set()
    transforms = {"train": set(), "test": set()}
    aspect = {"train": set(), "test": set()}
    extremes = set()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        if seed < 3:
            _require(_same(puzzle, generate_puzzle(seed)), "nondeterministic seed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total += 1
                a, out = pair["input"], pair["output"]
                before = a.copy()
                _require(np.array_equal(solve(a), out), "input-only solver mismatch")
                _require(np.array_equal(a, before), "solve mutated its input")
                info = _decode(a)
                p = info["payload"]
                shapes.add(a.shape)
                payload_shapes.add(p.shape)
                palettes.add(tuple(np.unique(a).tolist()))
                transforms[split].add(info["code"])
                aspect[split].add(int(np.sign(p.shape[0] - p.shape[1])))
                mask = tuple(bool(v) for v in np.all(_oriented(info) == _oriented(info)[:, :1], axis=1))
                masks.add(mask)
                if all(mask):
                    extremes.add("positive")
                if not any(mask):
                    extremes.add("negative")
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
        fingerprints.add(digest.hexdigest())
    _require(len(fingerprints) == 42 and len(shapes) >= 50 and len(payload_shapes) >= 50,
             "insufficient structural variation")
    _require(transforms["train"] == transforms["test"] == set(range(8)), "missing D4 coverage")
    _require(aspect["train"] == aspect["test"] == {-1, 0, 1}, "missing aspect ratios")
    _require(len(masks) >= 50 and len(palettes) >= 20, "insufficient masks or palettes")
    _require(extremes == {"positive", "negative"}, "missing extreme masks")
    for nt in range(2, 11):
        for ne in range(1, min(nt, 5)):
            p = generate_puzzle(900 + 10 * nt + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "counts changed")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    good = generate_puzzle(0)
    for kind in ("cell", "extent", "guide", "controls"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if kind == "cell":
            pair["output"][0, 0] = 0 if pair["output"][0, 0] == 5 else 5
        elif kind == "extent":
            pair["output"] = pair["output"][1:].copy()
        elif kind == "guide":
            r, c, h, w = info["bounds"]
            s = next(iter(info["guides"]))
            pos = ((r - 1, c + w // 2), (r + h // 2, c + w),
                   (r + h, c + w // 2), (r + h // 2, c - 1))[s]
            pair["input"][pos] = 0
        else:
            pair["input"] = _swap_controls(pair)["input"]
            # Retain the old answer: choose a mutation that changes its value.
            if np.array_equal(solve(pair["input"]), pair["output"]):
                continue
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError(f"mutation accepted: {kind}")
    return dict(seeds=42, pairs=total, canvas_shapes=len(shapes), payload_shapes=len(payload_shapes),
                masks=len(masks), palettes=len(palettes),
                train_transforms=sorted(transforms["train"]), test_transforms=sorted(transforms["test"]),
                extreme_masks=sorted(extremes))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
