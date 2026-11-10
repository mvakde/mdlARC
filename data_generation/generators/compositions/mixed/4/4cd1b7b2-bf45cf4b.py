"""A color-selected Latin completion becomes a mask for exact stamp copying.

Only NumPy and the standard library are used. Generation never reads evidence,
writes files, or calls a parent generator. The solver uses only the input grid.
"""
from functools import lru_cache
from itertools import combinations, permutations
import hashlib

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


@lru_cache(maxsize=1)
def _latin_catalogue():
    """All 576 order-four Latin squares, independent of any sampled answer."""
    rows = list(permutations(range(4)))
    answers = []
    for a in rows:
        for b in rows:
            if any(a[c] == b[c] for c in range(4)):
                continue
            for d in rows:
                if any(d[c] in (a[c], b[c]) for c in range(4)):
                    continue
                e = tuple(next(x for x in range(4) if x not in (a[c], b[c], d[c]))
                          for c in range(4))
                if len(set(e)) == 4:
                    answers.append((a, b, d, e))
    result = np.asarray(answers, dtype=np.int8)
    result.flags.writeable = False
    return result


def _matching_latin(clues):
    # Missing cells are -1. No block, diagonal, arithmetic or cyclic constraints.
    catalogue = _latin_catalogue()
    return catalogue[np.all((clues[None] < 0) | (catalogue == clues[None]), axis=(1, 2))]


@lru_cache(maxsize=1)
def _mask_catalogue():
    masks = np.eye(4, dtype=bool)[np.asarray(list(permutations(range(4))))]
    masks.flags.writeable = False
    return masks


def _binary_alternatives(board, active):
    """Count placements after forgetting identities of nonselected symbols."""
    masks = _mask_catalogue()
    return int(np.sum(np.all((board[None] < 0) | (masks == (board == active)[None]),
                             axis=(1, 2))))


def _transforms(a):
    return [f(np.rot90(a, k)) for k in range(4) for f in (lambda x: x, np.fliplr)]


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in _transforms(a)[1:])


def _bbox(binary):
    r, c = np.nonzero(binary)
    if len(r) == 0:
        raise ValueError("missing encoding")
    return int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1


def _crop(a, box):
    r0, r1, c0, c1 = box
    return a[r0:r1, c0:c1].copy()


def _separated(a, b):
    # End coordinates are exclusive: equality would mean no blank separator.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _read_input(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if max(counts) * 2 <= a.size or min(a.shape) < 6:
        raise ValueError("the canvas background and framed board must be evident")
    windows = np.lib.stride_tricks.sliding_window_view(a, (6, 6))
    corner = windows[:, :, 0, 0]
    rings = corner != bg
    for side in (windows[:, :, 0, :], windows[:, :, -1, :],
                 windows[:, :, :, 0], windows[:, :, :, -1]):
        rings &= np.all(side == corner[:, :, None], axis=2)
    candidates = []
    for r, c in np.argwhere(rings):
        r, c = int(r), int(c)
        active = int(a[r, c])
        clues = a[r + 1:r + 5, c + 1:c + 5].copy()
        symbols = tuple(sorted(set(map(int, clues.flat)) - {bg}))
        if len(symbols) != 4 or active not in symbols or not np.any(clues == bg):
            continue
        normalized = np.full((4, 4), -1, dtype=np.int8)
        for i, color in enumerate(symbols):
            normalized[clues == color] = i
        completions = _matching_latin(normalized)
        if len(completions) != 1:
            continue
        board_box = (r, r + 6, c, c + 6)
        remaining = a != bg
        remaining[r:r + 6, c:c + 6] = False
        if not np.any(remaining):
            continue
        stamp_box = _bbox(remaining)
        if not _separated(board_box, stamp_box):
            continue
        stamp = _crop(a, stamp_box)
        payload = set(map(int, stamp.flat)) - {bg}
        if (not all(2 <= n <= 7 for n in stamp.shape) or len(payload) < 2
                or payload.intersection(symbols)):
            continue
        latin = np.asarray(symbols, dtype=np.int8)[completions[0]]
        candidates.append(dict(background=bg, active=active, symbols=symbols,
                               clues=clues, latin=latin, mask=latin == active,
                               stamp=stamp, board_box=board_box, stamp_box=stamp_box))
    if len(candidates) != 1:
        raise ValueError("expected exactly one framed Latin board and separate stamp")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the board, select its frame color, then expand with the stamp."""
    info = _read_input(input_grid)
    stamp, mask = info["stamp"], info["mask"]
    sh, sw = stamp.shape
    r, c = np.indices((4 * sh, 4 * sw))
    return np.where(mask[r // sh, c // sw], stamp[r % sh, c % sw],
                    info["background"]).astype(np.int8)


def _expand(mask, stamp, bg):
    """Generation writes blocks; solve uses index arithmetic instead."""
    sh, sw = stamp.shape
    output = np.full((4 * sh, 4 * sw), bg, dtype=np.int8)
    for r, c in np.argwhere(mask):
        output[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = stamp
    return output


def _visible_box_repeat(board):
    for r in (0, 2):
        for c in (0, 2):
            values = [int(x) for x in board[r:r + 2, c:c + 2].flat if x >= 0]
            if len(values) > len(set(values)):
                return True
    return False


def _make_board(rng, active, kind, low, high, witness=False):
    """Delete clues only while the actual Latin completion stays unique."""
    catalogue = _latin_catalogue()
    for _ in range(160):
        square = catalogue[int(rng.integers(len(catalogue)))].copy()
        active_cells = list(map(tuple, np.argwhere(square == active)))
        if witness and not _asymmetric(square == active):
            continue
        holes = np.zeros((4, 4), dtype=bool)
        if kind == "intersection":
            rectangles = []
            for (r0, c0), (r1, c1) in combinations(active_cells, 2):
                if square[r0, c1] != square[r1, c0]:
                    rectangles.append((r0, r1, c0, c1))
            if not rectangles:
                continue
            r0, r1, c0, c1 = rectangles[int(rng.integers(len(rectangles)))]
            holes[np.ix_((r0, r1), (c0, c1))] = True
        else:
            # A 2- or 3-cycle admits two binary masks but can have just one
            # four-symbol completion. Three-cycles also cover Klein squares;
            # restricting this teaching device to rectangles would exclude them.
            length = int(rng.integers(2, 4))
            cycle = [active_cells[int(i)] for i in rng.choice(4, size=length, replace=False)]
            for j, (r, c) in enumerate(cycle):
                holes[r, c] = True
                holes[r, cycle[(j + 1) % length][1]] = True
            if kind == "row":
                holes[cycle[0][0], :] = True
            elif kind == "column":
                holes[:, cycle[0][1]] = True
        initial = square.copy()
        initial[holes] = -1
        if (holes.sum() > high or len(set(map(int, initial[~holes]))) != 4
                or len(_matching_latin(initial)) != 1):
            continue
        target = int(rng.integers(low, high + 1))
        preferences = rng.random((4, 4))
        if kind == "cluster":
            r, c = rng.integers(4, size=2)
            rr, cc = np.indices((4, 4))
            preferences += .25 * (abs(rr - r) + abs(cc - c))
        for flat in np.argsort(preferences, axis=None):
            if holes.sum() >= target:
                break
            r, c = divmod(int(flat), 4)
            if holes[r, c]:
                continue
            holes[r, c] = True
            trial = square.copy()
            trial[holes] = -1
            if (len(set(map(int, trial[~holes]))) != 4
                    or len(_matching_latin(trial)) != 1):
                holes[r, c] = False
        board = square.copy()
        board[holes] = -1
        hidden_active = int(np.sum(holes & (square == active)))
        hidden_inactive = int(np.sum(holes & (square != active)))
        if (not low <= holes.sum() <= high or hidden_active not in (2, 3)
                or hidden_inactive < 2 or len(set(map(int, board[~holes]))) != 4
                or len(_matching_latin(board)) != 1
                or _binary_alternatives(board, active) < 2):
            continue
        if witness and not _visible_box_repeat(board):
            continue
        return square, board
    raise RuntimeError("bounded Latin-mask construction exhausted")


def _stamp_shape(rng, regime, test, witness, previous):
    low = 3 if witness else 2
    options = [(h, w) for h in range(low, 8) for w in range(low, 8)]
    if regime == "stamp_scale":
        options = [(h, w) for h, w in options
                   if (min(h, w) >= 5 if test else max(h, w) <= 4)]
    elif regime == "aspect":
        options = [(h, w) for h, w in options if (h < w if test else h > w)]
    if previous is not None:
        options = [s for s in options if s != previous]
    return options[int(rng.integers(len(options)))]


def _make_stamp(rng, shape, bg, payload, witness):
    h, w = shape
    rr, cc = np.indices(shape)
    family = "random" if witness else str(rng.choice(
        ["bands", "rings", "diagonal", "tile", "symmetric", "random"]))
    hole_kind = "witness" if witness else str(rng.choice(
        ["none", "none", "interior", "edge", "multiple", "rectangle"]))
    n = len(payload)
    for _ in range(100):
        if family == "bands":
            labels = (rr if rng.integers(2) else cc) // int(rng.integers(1, 3))
        elif family == "rings":
            labels = np.minimum.reduce([rr, cc, h - rr - 1, w - cc - 1])
        elif family == "diagonal":
            labels = int(rng.integers(1, n + 1)) * rr + cc
        elif family == "tile":
            tile = rng.integers(n, size=(int(rng.integers(1, min(h, 3) + 1)),
                                        int(rng.integers(1, min(w, 3) + 1))))
            labels = tile[rr % tile.shape[0], cc % tile.shape[1]]
        elif family == "symmetric":
            table = rng.integers(n, size=((h + 1) // 2, (w + 1) // 2))
            labels = table[np.minimum(rr, h - 1 - rr), np.minimum(cc, w - 1 - cc)]
        else:
            labels = rng.integers(n, size=shape)
        stamp = np.asarray(payload, dtype=np.int8)[(labels + int(rng.integers(n))) % n]
        holes = np.zeros(shape, dtype=bool)
        if hole_kind in ("witness", "interior") and h >= 3 and w >= 3:
            holes[int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))] = True
        if hole_kind in ("witness", "edge"):
            edge = [(r, c) for r in range(h) for c in range(w)
                    if r in (0, h - 1) or c in (0, w - 1)]
            holes[edge[int(rng.integers(len(edge)))]] = True
        elif hole_kind == "multiple":
            holes.flat[rng.choice(h * w, size=int(rng.integers(1, max(2, h*w//3))),
                                 replace=False)] = True
        elif hole_kind == "rectangle" and h >= 4 and w >= 4:
            r0, c0 = int(rng.integers(1, h - 2)), int(rng.integers(1, w - 2))
            holes[r0:int(rng.integers(r0 + 2, h)),
                  c0:int(rng.integers(c0 + 2, w))] = True
        stamp[holes] = bg
        fg = stamp != bg
        if (len(set(map(int, stamp.flat)) - {bg}) < 2
                or not all(np.any(side) for side in (fg[0], fg[-1], fg[:, 0], fg[:, -1]))):
            continue
        if witness and not _asymmetric(stamp):
            continue
        return stamp.copy()
    # A deterministic, shape-general asymmetric fallback; also keeps both holes.
    stamp = np.full(shape, payload[0], dtype=np.int8)
    stamp[0, 1] = payload[1]
    if witness:
        stamp[1, 1] = bg
        stamp[-1, 0] = bg
    else:
        stamp[-1, -1] = payload[1]
    return stamp


def _place(rng, board, active, stamp, bg):
    framed = np.full((6, 6), active, dtype=np.int8)
    framed[1:5, 1:5] = board
    horizontal, reverse = bool(rng.integers(2)), bool(rng.integers(2))
    first, second = (stamp, framed) if reverse else (framed, stamp)
    gap, rim = int(rng.integers(1, 6)), int(rng.integers(2))
    fh, fw = first.shape
    sh, sw = second.shape
    base_h = (max(fh, sh) if horizontal else fh + sh + gap) + 2 * rim
    base_w = (fw + sw + gap if horizontal else max(fw, sw)) + 2 * rim
    h = base_h + int(rng.integers(min(30 - base_h, 9) + 1))
    w = base_w + int(rng.integers(min(30 - base_w, 9) + 1))
    foreground = np.count_nonzero(framed != bg) + np.count_nonzero(stamp != bg)
    while h * w <= 2 * foreground:
        if h < 30 and (h <= w or w == 30):
            h += 1
        else:
            w += 1
    if horizontal:
        fc = rim + int(rng.integers(w - base_w + 1))
        sc = fc + fw + gap
        fr = rim + int(rng.integers(h - fh - 2 * rim + 1))
        sr = rim + int(rng.integers(h - sh - 2 * rim + 1))
    else:
        fr = rim + int(rng.integers(h - base_h + 1))
        sr = fr + fh + gap
        fc = rim + int(rng.integers(w - fw - 2 * rim + 1))
        sc = rim + int(rng.integers(w - sw - 2 * rim + 1))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[fr:fr + fh, fc:fc + fw] = first
    grid[sr:sr + sh, sc:sc + sw] = second
    return grid


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["clue_sparsity", "stamp_scale", "axis", "aspect"]))
    palette = rng.permutation(10).astype(np.int8)
    varied_palette = bool(rng.integers(2))
    active_order = rng.permutation(4)
    puzzle, seen = {"train": [], "test": []}, set()
    previous_shape = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            test, witness = split == "test", split == "train" and i == 0
            # The first two demonstrations reuse an alphabet but query different
            # symbols. Subsequent examples may exchange all literal color roles.
            colors = rng.permutation(10).astype(np.int8) if varied_palette and (test or i >= 2) else palette
            bg, symbols = int(colors[0]), colors[1:5]
            active_index = int(active_order[i % 4]) if not test else int(rng.integers(4))
            active = int(symbols[active_index])
            k = int(rng.integers(2, 6))
            payload = colors[5:5 + k]
            if witness:
                kind, low, high = "intersection", 4, 4
            elif regime == "axis":
                kind, low, high = ("column" if test else "row"), 6, 9
            elif regime == "clue_sparsity":
                kind = str(rng.choice(["random", "cluster"]))
                low, high = (8, 11) if test else (4, 7)
            else:
                kind = str(rng.choice(["random", "cluster", "row", "column"]))
                low, high = 5, 10
            shape = _stamp_shape(rng, regime, test, witness, previous_shape)
            for _ in range(24):
                square, partial = _make_board(rng, active_index, kind, low, high, witness)
                board = np.full((4, 4), bg, dtype=np.int8)
                board[partial >= 0] = symbols[partial[partial >= 0]]
                stamp = _make_stamp(rng, shape, bg, payload, witness)
                inp = _place(rng, board, active, stamp, bg)
                key = (inp.shape, inp.tobytes())
                if key in seen:
                    continue
                # Reject the rare competing frame inside a stamp; never select
                # an interpretation using the stored generated answer.
                try:
                    info = _read_input(inp)
                except ValueError:
                    continue
                out = _expand(square == active_index, stamp, bg)
                if not np.array_equal(info["latin"], symbols[square]):
                    raise AssertionError("visible clues changed the sampled completion")
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                previous_shape = shape
                break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_edge_hole(info):
    holes = info["stamp"] == info["background"]
    return any(np.any(side) for side in (holes[0], holes[-1], holes[:, 0], holes[:, -1]))


def validate_generated_puzzle(puzzle):
    """Validate returned grids, unique recovery, parent necessity and witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    information, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "repeated input"
            seen.add(key)
            info = _read_input(inp)
            information[split].append(info)
            bg, active = info["background"], info["active"]
            clues, latin, stamp = info["clues"], info["latin"], info["stamp"]
            holes, mask = clues == bg, info["mask"]
            assert 4 <= holes.sum() <= 11
            assert np.sum(holes & mask) in (2, 3), "Latin completion must add active sites"
            assert np.sum(holes & ~mask) >= 2, "blanks cannot all mean active"
            assert _binary_alternatives(np.where(holes, -1, clues), active) >= 2, (
                "other Latin-symbol identities must resolve a binary placement ambiguity")
            assert np.array_equal(clues[~holes], latin[~holes])
            assert all(set(map(int, row)) == set(info["symbols"]) for row in list(latin) + list(latin.T))
            assert np.all(mask.sum(axis=0) == 1) and np.all(mask.sum(axis=1) == 1)
            assert int(mask.sum()) == 4
            sh, sw = stamp.shape
            assert out.shape == (4 * sh, 4 * sw)
            for r in range(4):
                for c in range(4):
                    block = out[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
                    assert np.array_equal(block, stamp) if mask[r, c] else np.all(block == bg)
            assert not set(map(int, out.flat)).intersection(info["symbols"])
            assert set(map(int, out.flat)) == set(map(int, stamp.flat)) | {bg}
            # Concrete counterfactuals omitting either parent contribution.
            visible_only = _expand(clues == active, stamp, bg)
            assert not np.array_equal(visible_only, out)
            assert not np.array_equal(_expand((clues == active) | holes, stamp, bg), out)
            assert out.shape != latin.shape, "without B the Latin grid has the wrong dimensions"
            assert np.array_equal(solve(inp), out)
    train, test = information["train"], information["test"]
    assert len({(i["clues"] == i["background"]).tobytes() for i in train}) >= 2
    assert len({i["stamp"].shape for i in train}) >= 2
    groups = {}
    for info in train:
        groups.setdefault((info["background"], info["symbols"]), set()).add(info["active"])
    assert any(len(queries) >= 2 for queries in groups.values()), "show a changing query on one alphabet"
    assert any(_asymmetric(i["mask"]) for i in train), "expose mask orientation"
    assert any(_asymmetric(i["stamp"]) for i in train), "expose stamp orientation"
    assert any(_has_edge_hole(i) and np.any(i["stamp"][1:-1, 1:-1] == i["background"])
               for i in train), "show retained interior and boundary background cells"
    assert any(_visible_box_repeat(np.where(i["clues"] == i["background"], -1, i["clues"]))
               for i in train), "exclude a Sudoku sub-box rule visibly"
    assert any(not np.any((i["clues"] == i["background"]).sum(axis=0) == 1)
               and not np.any((i["clues"] == i["background"]).sum(axis=1) == 1)
               for i in train), "show a case requiring more than line-single filling"
    # Infer the observable shift instead of consulting a sampled regime label.
    sparse_shift = min(np.sum(i["clues"] == i["background"]) for i in test) > max(
        np.sum(i["clues"] == i["background"]) for i in train)
    scale_shift = min(min(i["stamp"].shape) for i in test) > max(max(i["stamp"].shape) for i in train)
    aspect_shift = all(i["stamp"].shape[0] > i["stamp"].shape[1] for i in train) and all(
        i["stamp"].shape[0] < i["stamp"].shape[1] for i in test)
    axis_shift = any(np.any(np.all(i["clues"] == i["background"], axis=1)) for i in train) and all(
        np.any(np.all(i["clues"] == i["background"], axis=0)) for i in test)
    assert sparse_shift or scale_shift or aspect_shift or axis_shift, "missing structural train/test shift"


def _canonical(a, bg):
    mapping = {int(bg): 0}
    return a.shape, tuple(mapping.setdefault(int(x), len(mapping)) for x in a.flat)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed semantic/diversity audit, without saving puzzle payloads."""
    assert _latin_catalogue().shape == (576, 4, 4)
    patterns, masks, stamps, shapes, palettes, holes, layouts, signatures = (set() for _ in range(8))
    rectangles = rings = box_repeats = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _read_input(pair["input"])
                b, s = i["board_box"], i["stamp_box"]
                layouts.add("left" if b[3] < s[2] else "right" if s[3] < b[2]
                            else "above" if b[1] < s[0] else "below")
                patterns.add(_canonical(i["latin"], i["background"]))
                masks.add(i["mask"].tobytes())
                stamps.add(_canonical(i["stamp"], i["background"]))
                shapes.add(i["stamp"].shape)
                palettes.add((i["background"], i["symbols"], i["active"]))
                holes.add(int(np.sum(i["clues"] == i["background"])))
                rectangles += int(i["stamp"].shape[0] != i["stamp"].shape[1])
                rings += int(_has_edge_hole(i))
                box_repeats += int(_visible_box_repeat(np.where(
                    i["clues"] == i["background"], -1, i["clues"])))
    assert len(signatures) == 42
    assert len(patterns) >= 20 and len(masks) >= 16 and len(stamps) >= 100
    assert len(shapes) >= 24 and len(palettes) >= 70 and len(holes) >= 6
    assert layouts == {"left", "right", "above", "below"}
    assert rectangles >= 80 and rings >= 42 and box_repeats >= 42
    assert _digest(generate_puzzle(100)) == _digest(generate_puzzle(100))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(9000 + 10 * train + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # Input-only invariance under role changes, translations and transposition.
    pair = generate_puzzle(0)["train"][0]
    info = _read_input(pair["input"])
    before = pair["input"].copy()
    assert np.array_equal(solve(before), pair["output"])
    assert np.array_equal(before, pair["input"])
    perm = np.array([9, 3, 0, 7, 1, 8, 2, 6, 4, 5], dtype=np.int8)
    assert np.array_equal(solve(perm[before]), perm[pair["output"]])
    assert np.array_equal(solve(before.T), pair["output"].T)
    if max(before.shape) <= 28:
        padded = np.pad(before, 1, constant_values=info["background"])
        assert np.array_equal(solve(padded), pair["output"])
    # A different visible frame query must cause a different mask and answer.
    changed = before.copy()
    r0, r1, c0, c1 = info["board_box"]
    other = next(c for c in info["symbols"] if c != info["active"])
    changed[r0, c0:c1] = changed[r1 - 1, c0:c1] = other
    changed[r0:r1, c0] = changed[r0:r1, c1 - 1] = other
    assert not np.array_equal(solve(changed), pair["output"])
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in ps]
           for s, ps in generate_puzzle(1).items()}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("a damaged output was accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "latin_patterns": len(patterns), "active_masks": len(masks),
            "stamp_patterns": len(stamps), "stamp_shapes": len(shapes),
            "palette_queries": len(palettes), "blank_counts": sorted(holes),
            "layouts": sorted(layouts), "rectangular_stamps": rectangles,
            "edge_hole_pairs": rings, "visible_non_sudoku_pairs": box_repeats}
