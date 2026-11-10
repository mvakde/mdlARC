"""Synthetic puzzles that highlight complete blank rows and columns."""

import hashlib
import numpy as np


_SOURCE_HASHES = {
    "78809a88f7648b85cc4bf0e3abd936162ae1db690acdbc3e83b308a907dd7029",
    "95ea11617949fe6acc3efe8f61ca33d7e32cc11457fd9976b5879708a8d4a727",
    "f0621a1bd4c730a96bb804bb1517f252479cba631164b53f48fcf02ef4e009e3",
    "f5e174bc716c70a5616915e3e403d6eacc6c5d71a79c82aa8ffbf09b10189972",
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} is out of bounds")
    return value


def _counts(rng, num_train, num_test):
    nt, ns = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _pair_hash(inp, out):
    a, b = np.asarray(inp, dtype=np.uint8), np.asarray(out, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _candidate_blank(inp):
    scores = {}
    for c in np.unique(inp):
        n = int(np.count_nonzero(np.all(inp == c, axis=0)) + np.count_nonzero(np.all(inp == c, axis=1)))
        if n:
            scores[int(c)] = n
    if not scores:
        return None
    maximum = max(scores.values())
    winners = [c for c, n in scores.items() if n == maximum]
    return winners[0] if len(winners) == 1 else None


def _shape(rng, nr, nc, regime, train_shapes):
    if regime == "target_test" and train_shapes and rng.random() < 0.16:
        h, w = train_shapes[int(rng.integers(len(train_shapes)))]
        if h >= nr + 3 and w >= nc + 3:
            return h, w
    mode = int(rng.choice(6, p=[.20, .22, .20, .15, .15, .08]))
    if mode == 0:
        h, w = int(rng.integers(5, 11)), int(rng.integers(5, 11))
    elif mode == 1:
        h, w = int(rng.integers(10, 22)), int(rng.integers(10, 22))
    elif mode == 2:
        h, w = int(rng.integers(21, 31)), int(rng.integers(21, 31))
    elif mode == 3:
        h, w = int(rng.integers(5, 13)), int(rng.integers(16, 31))
    elif mode == 4:
        h, w = int(rng.integers(16, 31)), int(rng.integers(5, 13))
    else:
        h = w = int(rng.integers(5, 31))
    return max(h, nr + 3), max(w, nc + 3)


def _positions(rng, length, count):
    if count == 0:
        return np.empty(0, dtype=int)
    mode = int(rng.integers(4))
    if mode == 0:
        edge = int(rng.choice([0, length - 1]))
        other = rng.choice([i for i in range(length) if i != edge], count - 1, replace=False)
        return np.sort(np.r_[edge, other])
    if mode == 1 and count >= 2:
        first = int(rng.integers(length - 1))
        other = rng.choice([i for i in range(length) if i not in (first, first + 1)], count - 2, replace=False)
        return np.sort(np.r_[first, first + 1, other])
    return np.sort(rng.choice(length, count, replace=False))


def _texture(rng, shape, rows, cols, blank, palette, near_axis, solid):
    h, w = shape
    grid = np.full(shape, blank, dtype=np.int8)
    ar = np.array([i for i in range(h) if i not in rows], dtype=int)
    ac = np.array([i for i in range(w) if i not in cols], dtype=int)
    density = float(rng.uniform(.30, .82))
    style = int(rng.integers(3))
    if style == 0:
        filled = rng.random((len(ar), len(ac))) < density
    elif style == 1:
        coarse = rng.random(((h + 2) // 3, (w + 2) // 3))
        patch = coarse[np.ix_(ar // 3, ac // 3)]
        filled = .7 * patch + .3 * rng.random(patch.shape) < density
    else:
        rr, cc = np.meshgrid(ar, ac, indexing="ij")
        pattern = (rr // int(rng.integers(2, 5)) + cc // int(rng.integers(2, 5))) % 2 == 0
        filled = rng.random(pattern.shape) < np.where(pattern, min(.95, density + .25), max(.1, density - .25))
    colors = rng.choice(palette, size=filled.shape)
    grid[np.ix_(ar, ac)] = np.where(filled, colors, blank)

    solid_row = solid_col = None
    if solid and len(rows) and not len(cols):
        solid_row = int(rng.choice(ar))
        grid[solid_row, :] = int(rng.choice(palette))
    if solid and len(cols) and not len(rows):
        solid_col = int(rng.choice(ac))
        grid[:, solid_col] = int(rng.choice(palette))

    near_row = near_col = None
    if near_axis == "row":
        near_row = int(rng.choice(ar[ar != solid_row] if solid_row is not None else ar))
        crossing_color = int(grid[near_row, solid_col]) if solid_col is not None else None
        grid[near_row, ac] = blank
        picks = list(rng.choice(ac, size=int(rng.integers(1, 3)), replace=False))
        if solid_col is not None:
            picks[0] = solid_col
        grid[near_row, picks] = rng.choice(palette, size=len(picks))
        if solid_col is not None:
            grid[near_row, solid_col] = crossing_color
    else:
        near_col = int(rng.choice(ac[ac != solid_col] if solid_col is not None else ac))
        crossing_color = int(grid[solid_row, near_col]) if solid_row is not None else None
        grid[ar, near_col] = blank
        picks = list(rng.choice(ar, size=int(rng.integers(1, 3)), replace=False))
        if solid_row is not None:
            picks[0] = solid_row
        grid[picks, near_col] = rng.choice(palette, size=len(picks))
        if solid_row is not None:
            grid[solid_row, near_col] = crossing_color

    for r in ar:
        if not np.any(grid[r, ac] != blank):
            choices = ac[ac != near_col] if near_col is not None else ac
            grid[r, int(rng.choice(choices))] = int(rng.choice(palette))
    for c in ac:
        if not np.any(grid[ar, c] != blank):
            choices = ar[ar != near_row] if near_row is not None else ar
            grid[int(rng.choice(choices)), c] = int(rng.choice(palette))
    for r in ar:
        if r != solid_row and np.all(grid[r] != blank):
            choices = ac[ac != near_col] if near_col is not None else ac
            grid[r, int(rng.choice(choices))] = blank
    for c in ac:
        if c != solid_col and np.all(grid[:, c] != blank):
            choices = ar[ar != near_row] if near_row is not None else ar
            grid[int(rng.choice(choices)), c] = blank
    # For multicolor pairs, place missing colors in distinct ordinary cells.
    available = [(int(r), int(c)) for r in ar if r != near_row for c in ac if c != near_col and r != solid_row and c != solid_col]
    rng.shuffle(available)
    for color in palette:
        if color not in grid:
            if not available:
                raise ValueError("no room for all texture colors")
            r, c = available.pop()
            grid[r, c] = color
    return grid


def _line_counts(rng, kind):
    if kind == "horizontal":
        return int(rng.integers(1, 5)), 0
    if kind == "vertical":
        return 0, int(rng.integers(1, 5))
    return int(rng.integers(1, 5)), int(rng.integers(1, 5))


def _make_pair(rng, kind, regime, highlight, blank, palette, train_shapes, force_solid):
    nr, nc = _line_counts(rng, kind)
    solid = kind != "crossing" and (force_solid or rng.random() < .23)
    if solid:
        nr = max(nr, 2) if nr else nr
        nc = max(nc, 2) if nc else nc
    shape = _shape(rng, nr, nc, regime, train_shapes)
    rows, cols = _positions(rng, shape[0], nr), _positions(rng, shape[1], nc)
    near_axis = str(rng.choice(["row", "column"]))
    inp = _texture(rng, shape, rows, cols, blank, palette, near_axis, solid)
    if _candidate_blank(inp) != blank:
        raise ValueError("ambiguous blank color")
    mask = np.all(inp == blank, axis=1)[:, None] | np.all(inp == blank, axis=0)[None, :]
    out = inp.copy()
    out[mask] = highlight
    return {"input": inp, "output": out}


def _solid(inp, blank):
    return any(c != blank and (np.any(np.all(inp == c, axis=0)) or np.any(np.all(inp == c, axis=1))) for c in np.unique(inp))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    nt, ns = _counts(rng, num_train, num_test)
    highlight = int(rng.integers(10))
    colors = [c for c in range(10) if c != highlight]
    solo = str(rng.choice(["horizontal", "vertical"]))
    other = "vertical" if solo == "horizontal" else "horizontal"
    train_kinds = [solo, "crossing"] if nt == 2 else [solo, other, "crossing"]
    train_kinds += [str(rng.choice(["horizontal", "vertical", "crossing"], p=[.28, .28, .44])) for _ in range(nt - len(train_kinds))]
    rng.shuffle(train_kinds)
    test_kinds = [str(rng.choice(["horizontal", "vertical", "crossing"], p=[.31, .31, .38])) for _ in range(ns)]
    backgrounds = [int(rng.choice(colors))]
    backgrounds.append(int(rng.choice([c for c in colors if c != backgrounds[0]])))
    backgrounds += [int(rng.choice(colors)) for _ in range(nt - 2)]
    puzzle = {"train": [], "test": []}
    train_shapes, train_textures, used = [], set(), set()
    for split, kinds in (("train", train_kinds), ("test", test_kinds)):
        for i, kind in enumerate(kinds):
            if split == "train":
                blank = backgrounds[i]
                solid_demo = any(_solid(p["input"], backgrounds[j]) for j, p in enumerate(puzzle["train"]))
                force_solid = kind == solo and not solid_demo
                multicolor_demo = any(len(np.unique(p["input"])) >= 3 for p in puzzle["train"])
                multicolor = (kind == "crossing" and not multicolor_demo) or rng.random() < .35
            else:
                blank = int(rng.choice(backgrounds)) if rng.random() < .55 else int(rng.choice(colors))
                force_solid = False
                multicolor = rng.random() < .47
            choices = [c for c in colors if c != blank]
            ncolors = int(rng.integers(2, 4)) if multicolor else 1
            if split == "test" and train_textures and rng.random() < .55:
                reuse = sorted(train_textures & set(choices))
                first = int(rng.choice(reuse)) if reuse else int(rng.choice(choices))
                palette = [first] + [int(c) for c in rng.choice([c for c in choices if c != first], ncolors - 1, replace=False)]
            else:
                palette = [int(c) for c in rng.choice(choices, ncolors, replace=False)]
            for _ in range(32):
                pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63)))
                try:
                    pair = _make_pair(pair_rng, kind, split, highlight, blank, palette, train_shapes, force_solid)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in used and _pair_hash(pair["input"], pair["output"]) not in _SOURCE_HASHES:
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            used.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_shapes.append(pair["input"].shape)
                train_textures.update(int(c) for c in np.unique(pair["input"]) if c != blank)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
        _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "invalid grid dimensions")
        _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
        _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid ARC color")
    _require(inp.shape == out.shape, "input/output dimensions differ")
    changed = inp != out
    _require(bool(np.any(changed)), "pair has no change")
    old, new = np.unique(inp[changed]), np.unique(out[changed])
    _require(len(old) == len(new) == 1, "changed colors are inconsistent")
    blank, highlight = int(old[0]), int(new[0])
    _require(highlight not in inp, "highlight color occurs in input")
    _require(_candidate_blank(inp) == blank, "blank color is ambiguous")
    rows = np.flatnonzero(np.all(inp == blank, axis=1))
    cols = np.flatnonzero(np.all(inp == blank, axis=0))
    _require(len(rows) + len(cols) >= 1, "no complete blank lines")
    mask = np.zeros(inp.shape, dtype=bool)
    mask[rows, :] = True
    mask[:, cols] = True
    expected = inp.copy()
    expected[mask] = highlight
    _require(np.array_equal(out, expected), "not exactly the full-line transformation")
    _require(np.any((inp == blank) & ~mask), "ordinary blank gaps absent")
    _require(np.any(inp != blank), "texture absent")
    _require(_pair_hash(inp, out) not in _SOURCE_HASHES, "exact source pair copied")
    active_rows = np.flatnonzero(~np.all(inp == blank, axis=1))
    active_cols = np.flatnonzero(~np.all(inp == blank, axis=0))
    near = np.any(np.sum(inp[active_rows] != blank, axis=1) <= 2) or np.any(np.sum(inp[:, active_cols] != blank, axis=0) <= 2)
    return {"blank": blank, "highlight": highlight, "rows": rows, "cols": cols, "near": bool(near), "solid": _solid(inp, blank), "multicolor": len(np.unique(inp)) >= 3}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    records, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            records[split].append(_inspect_pair(pair))
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input or pair")
            seen.add(key)
    line_colors = {r["highlight"] for split in records.values() for r in split}
    _require(len(line_colors) == 1, "highlight color is not puzzle-wide")
    _require(any(len(r["rows"]) for r in records["train"]), "horizontal lines not taught")
    _require(any(len(r["cols"]) for r in records["train"]), "vertical lines not taught")
    _require(any(r["solid"] for r in records["train"]), "solid texture not demonstrated")
    _require(any(r["multicolor"] for r in records["train"]), "multicolor texture not demonstrated")
    _require(len({r["blank"] for r in records["train"]}) >= 2, "blank role does not vary")
    _require(all(r["near"] for r in records["train"]), "near-blank line not demonstrated")
    highlight = next(iter(line_colors))
    for pair in puzzle["test"]:
        inp = pair["input"].copy()
        blank = _candidate_blank(inp)
        prediction = inp.copy()
        prediction[np.all(inp == blank, axis=1), :] = highlight
        prediction[:, np.all(inp == blank, axis=0)] = highlight
        _require(np.array_equal(prediction, pair["output"]), "test not inferable from training")


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "cross-seed puzzles repeat")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack range")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack range")
    shapes, layouts, highlights, arrangements = set(), set(), set(), set()
    edge = adjacent = near = solid = multi = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                r = _inspect_pair(pair)
                rows, cols = tuple(r["rows"]), tuple(r["cols"])
                shapes.add(inp.shape)
                layouts.add((inp.shape, rows, cols))
                highlights.add(r["highlight"])
                arrangements.add((len(rows), len(cols)))
                edge += any(x in (0, inp.shape[0] - 1) for x in rows) or any(x in (0, inp.shape[1] - 1) for x in cols)
                adjacent += any(b - a == 1 for a, b in zip(rows, rows[1:])) or any(b - a == 1 for a, b in zip(cols, cols[1:]))
                near += r["near"]
                solid += r["solid"]
                multi += r["multicolor"]
    _require(len(shapes) >= 65 and len(layouts) >= 170, "geometry diversity weak")
    _require(len(highlights) >= 8 and len(arrangements) >= 12, "colors or line counts lack range")
    _require(min(edge, adjacent, near, solid, multi) >= 10, "review feature coverage weak")
    _require(_signature(generate_puzzle(123456, 7, 3)) == _signature(generate_puzzle(123456, 7, 3)), "explicit generation nondeterministic")
    _require(_signature(generate_puzzle(98765)) == _signature(generate_puzzle(98765)), "automatic generation nondeterministic")
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(5000 + nt * 10 + ns, nt, ns)
        _require((len(p["train"]), len(p["test"])) == (nt, ns), "explicit counts ignored")
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(6000 + (nt or 0) * 10 + (ns or 0), nt, ns)
        _require((nt is None or len(p["train"]) == nt) and (ns is None or len(p["test"]) == ns), "partial counts ignored")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7000, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
