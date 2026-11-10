"""Registered fragments restore bars whose adjacency squares label a column.

No parent module is imported. Generation constructs a scene, divides its pixels,
and stages two patches; solving parses only the staged input. Diagnostics are
streamed in self_test, with no retained puzzles, JSON, logs, or execution traces.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


LENGTHS = (2, 3, 4)
BACKGROUNDS = (0, 2, 3, 5, 6, 7, 8)
HOST_SEEDS = (0, 578517727, 633484945, 701431516, 992310517,
              617169657, 422940904, 788050542, 805325569, 419037709,
              225924193, 103744418, 703114297, 248984672, 276269572,
              384811012, 60376261, 270517854, 215988141, 579742139,
              392293642, 683268621, 34735962, 28613156, 967302666,
              750295617, 645006159, 497899577, 677020949, 111304322,
              923394247, 903650264, 654859244, 199246332, 730329514,
              763226235, 521429057, 103789098, 441395707, 116578302,
              796361453, 8966069)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return grid


def _key(grid):
    return (tuple(grid.shape), grid.astype(np.uint8, copy=False).tobytes())


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _counts(rng, num_train, num_test):
    # Explicit None is supported as a convenience; omitted counts remain 4/1.
    nt = None if num_train is None else _count(num_train, "num_train", 2, 10)
    ns = None if num_test is None else _count(num_test, "num_test", 1, 4)
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(ns + 1, 11))
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _runs(cells):
    """Four-connected components; used for the two straight guide bars."""
    todo, parts = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        part, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _bbox(cells):
    return (min(y for y, x in cells), max(y for y, x in cells),
            min(x for y, x in cells), max(x for y, x in cells))


def _shift(cells, dy=0, dx=0):
    return {(y + dy, x + dx) for y, x in cells}


def _parse(grid):
    """Recover every role and patch owner without outputs or generation state.

    Each patch lies to the right of its full-height guide. Among guides spanning
    a pixel's row, the closest guide on its left owns it. Patch rectangles must
    be disjoint with at least one blank row/column between them. Reading order
    selects the anchor, so no connectedness assumption on payload is needed.
    """
    grid = _array(grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 4 or counts.max() * 2 <= grid.size:
        raise ValueError("four colors and a strict-majority background required")
    bg = int(colors[np.argmax(counts)])
    by = {int(c): {tuple(map(int, p)) for p in np.argwhere(grid == c)}
          for c in colors if c != bg}
    refs = [c for c, pixels in by.items() if len(pixels) == 1]
    if len(refs) != 1:
        raise ValueError("one distinct-color reference singleton required")
    ref = refs[0]
    choices = []
    for guide, pixels in by.items():
        if guide == ref:
            continue
        parts = _runs(pixels)
        if len(parts) != 2 or len(parts[0]) != len(parts[1]) or len(parts[0]) < 5:
            continue
        guides = []
        for part in parts:
            t, b, l, r = _bbox(part)
            if l != r or b - t + 1 != len(part):
                break
            guides.append((t, l, b))
        if len(guides) != 2:
            continue
        guides.sort()  # top first; at equal tops, left first
        fg = next(c for c in by if c not in (guide, ref))
        patches = [set(), set()]
        reference_owner = None
        valid = True
        for color in (fg, ref):
            for y, x in by[color]:
                owners = [(c, i) for i, (t, c, b) in enumerate(guides)
                          if t <= y <= b and x >= c + 2]
                if not owners:
                    valid = False
                    break
                _, i = max(owners)
                t, c, _ = guides[i]
                patches[i].add((y - t, x - c))
                if color == ref:
                    reference_owner = i
            if not valid:
                break
        if not valid or any(not p for p in patches):
            continue
        boxes = []
        for i, (t, c, b) in enumerate(guides):
            boxes.append((t, b, c, c + max(x for y, x in patches[i])))
        a, b = boxes
        if not (a[1] + 1 < b[0] or b[1] + 1 < a[0]
                or a[3] + 1 < b[2] or b[3] + 1 < a[2]):
            continue
        # Check the payload does not accidentally occupy another patch's box.
        ownership_ok = all(not (boxes[1-i][0] <= t+y <= boxes[1-i][1]
                                and boxes[1-i][2] <= c+x <= boxes[1-i][3])
                           for i, (t, c, _) in enumerate(guides)
                           for y, x in patches[i])
        if not ownership_ok:
            continue
        ry, rx = next(iter(by[ref]))
        t, c, _ = guides[reference_owner]
        ref_relative = (ry - t, rx - c - reference_owner)
        anchor = {p for p in patches[0] if (p[0] + guides[0][0],
                                           p[1] + guides[0][1]) not in by[ref]}
        mover = {p for p in patches[1] if (p[0] + guides[1][0],
                                          p[1] + guides[1][1]) not in by[ref]}
        if not anchor or not mover:
            continue
        choices.append({"background": bg, "foreground": fg, "guide_color": guide,
                        "reference_color": ref, "guides": tuple(guides),
                        "anchor": anchor, "mover": _shift(mover, dx=-1),
                        "reference": ref_relative, "reference_owner": reference_owner,
                        "boxes": tuple(boxes)})
    if len(choices) != 1:
        raise ValueError("guide/payload assignment must be unique")
    return choices[0]


def _restored(parsed):
    """Analyze the registered union, including its independent row witnesses."""
    a, b = parsed["anchor"], parsed["mover"]
    merged = a | b
    ry, rx = parsed["reference"]
    if (ry, rx) in merged or any(y == ry for y, x in merged):
        raise ValueError("the reference needs its own row")
    rows = sorted({y for y, x in merged})
    bars = []
    for y in rows:
        xs = sorted(x for yy, x in merged if yy == y)
        if (len(xs) not in LENGTHS or xs != list(range(xs[0], xs[-1] + 1))
                or not (xs[-1] <= rx - 2 or xs[0] >= rx + 2)):
            raise ValueError("union must contain separated bars beside the reference column")
        left = {x for yy, x in a if yy == y}
        right = {x for yy, x in b if yy == y}
        if not left or not right:
            raise ValueError("both patches must contribute to every bar")
        bars.append({"row": y, "start": xs[0], "length": len(xs),
                     "a": frozenset(left), "b": frozenset(right),
                     "overlap": len(left & right)})
    if not 1 <= len(bars) <= 5 or any(y2 - y1 < 2 for y1, y2 in zip(rows, rows[1:])):
        raise ValueError("one to five bars on separated rows required")
    if any(abs(y - ry) < 2 for y in rows):
        raise ValueError("reference row must be separated from bar rows")
    if min(x for y, x in merged | {(ry, rx)}) < 2:
        raise ValueError("data must stay clear of the registered guides")
    return merged, bars


def _render(parsed, merged, bars, digit_mode="square", length_mode="union"):
    ref = parsed["reference"]
    t, b, l, r = _bbox(merged | {ref})
    out = np.full((b - t + 1, r - l + 1), parsed["background"], dtype=np.int8)
    for y, x in merged:
        out[y - t, x - l] = parsed["foreground"]
    out[ref[0] - t, ref[1] - l] = parsed["reference_color"]
    if digit_mode == "none":
        return out
    for bar in bars:
        length = bar["length"]
        if length_mode == "largest":
            length = max(len(bar["a"]), len(bar["b"]))
        elif length_mode == "sum":
            length = len(bar["a"]) + len(bar["b"])
        adjacency = length - 1
        if digit_mode == "square":
            code = adjacency * adjacency % 10
        elif digit_mode == "adjacency":
            code = adjacency % 10
        elif digit_mode == "length_square":
            code = length * length % 10
        elif digit_mode == "color":
            code = parsed["foreground"]
        else:
            raise ValueError("unknown diagnostic mode")
        out[bar["row"] - t, ref[1] - l] = code
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Register the input patches, union their pixels, then label restored bars."""
    parsed = _parse(input_grid)
    merged, bars = _restored(parsed)
    return _render(parsed, merged, bars)


def _split_bar(rng, length, mode):
    """Pixel masks in completed-bar coordinates; their union is the full bar."""
    if mode == "disjoint":
        cut = int(rng.integers(1, length))
        a, b = set(range(cut)), set(range(cut, length))
    elif mode == "overlap" and length >= 3:
        # Both omit an end, but overlap internally: sums cannot measure union.
        a, b = set(range(length - 1)), set(range(1, length))
    elif mode == "weave" and length >= 3:
        # The first patch has an internal hole; sample overlap independently.
        a, b = set(range(0, length, 2)), set(range(1, length, 2))
        if rng.random() < .65:
            b.add(int(rng.choice(sorted(a))))
    else:
        a, b = {0}, {1}
    if rng.integers(2):
        a, b = {length - 1 - x for x in a}, {length - 1 - x for x in b}
    if rng.integers(2):
        a, b = b, a
    return a, b


def _stage(rng, payloads, height, palette):
    widths = [max(x for y, x in p) + 1 for p in payloads]
    candidates = []
    for gap in range(1, 5):
        for jitter in range(-3, 4):
            # First patch is always first in guide reading order.
            candidates.append(((0, max(0, -jitter)),
                               (height + gap, max(0, jitter)), "vertical"))
            if jitter >= 0:
                candidates.append(((0, 0), (jitter, widths[0] + gap), "horizontal"))
            candidates.append(((0, 0), (height + gap, widths[0] + gap), "diagonal"))
            candidates.append(((0, widths[1] + gap), (height + gap, 0), "diagonal"))
    feasible = []
    for p0, p1, stage in candidates:
        h = max(p0[0], p1[0]) + height
        w = max(p0[1] + widths[0], p1[1] + widths[1])
        if h <= 30 and w <= 30:
            feasible.append((p0, p1, stage, h, w))
    # Choose a feasible staging category first, avoiding vertical-only bias from
    # its greater number of jitter combinations. All loops have fixed bounds.
    stages = sorted({p[2] for p in feasible})
    stage = str(rng.choice(stages))
    options = [p for p in feasible if p[2] == stage]
    p0, p1, _, h, w = options[int(rng.integers(len(options)))]
    pads = [int(rng.integers(0, 4)) for _ in range(4)]
    pads[0] = min(pads[0], 30 - h)
    pads[1] = min(pads[1], 30 - h - pads[0])
    pads[2] = min(pads[2], 30 - w)
    pads[3] = min(pads[3], 30 - w - pads[2])
    bg, fg, gc, rc = palette
    grid = np.full((h + pads[0] + pads[1], w + pads[2] + pads[3]), bg, dtype=np.int8)
    for i, (y0, x0) in enumerate((p0, p1)):
        y0 += pads[0]
        x0 += pads[2]
        grid[y0:y0 + height, x0] = gc
        for (y, x), is_ref in payloads[i].items():
            grid[y0 + y, x0 + x] = rc if is_ref else fg
    return grid


def _make_pair(rng, palette, sides_mode, lengths, modes):
    n = len(lengths)
    # The guide spans every local row. Active rows have one or two intervening
    # blank rows; the reference may precede, follow, or lie between the bars.
    ref_slot = int(rng.integers(n + 1))
    gaps = [2] * n
    for _ in range(int(rng.integers(0, min(2, 12 - 2 * n) + 1))):
        gaps[int(rng.integers(n))] += 1
    active = [0]
    for gap in gaps:
        active.append(active[-1] + gap)
    span = active[-1] + 1
    height = int(rng.integers(max(5, span), min(13, max(5, span) + 3) + 1))
    top = int(rng.integers(height - span + 1))
    active = [y + top for y in active]
    ref_row = active.pop(ref_slot)
    if sides_mode == "mixed":
        sides = [int(rng.choice((-1, 1))) for _ in lengths]
        if n >= 2:
            sides[:2] = [-1, 1]
            rng.shuffle(sides)
    else:
        sides = [-1 if sides_mode == "left" else 1] * n
    left_max = max([l for l, s in zip(lengths, sides) if s == -1], default=0)
    right_max = max([l for l, s in zip(lengths, sides) if s == 1], default=0)
    ref_col = 2 + (left_max + 1 + int(rng.integers(0, 3)) if left_max else 0)
    right_capacity = right_max + 1 + int(rng.integers(0, 3)) if right_max else 0
    payloads = [{}, {}]
    completed = set()
    for row, length, side, mode in zip(active, lengths, sides, modes):
        if side == -1:
            gap = int(rng.integers(1, ref_col - length - 1))
            start = ref_col - gap - length
        else:
            gap = int(rng.integers(1, right_capacity - length + 1))
            start = ref_col + gap + 1
        a, b = _split_bar(rng, length, mode)
        for i, mask in enumerate((a, b)):
            for x in mask:
                payloads[i][(row, start + x + i)] = False
        completed.update((row, start + x) for x in range(length))
    owner = int(rng.integers(2))
    payloads[owner][(ref_row, ref_col + owner)] = True
    inp = _stage(rng, payloads, height, palette)
    # Expected output is constructed directly from the intended completed scene,
    # not by invoking solve. Parsing later checks every sampled split/staging.
    t, bottom, l, r = _bbox(completed | {(ref_row, ref_col)})
    out = np.full((bottom - t + 1, r - l + 1), palette[0], dtype=np.int8)
    for y, x in completed:
        out[y - t, x - l] = palette[1]
    out[ref_row - t, ref_col - l] = palette[3]
    for row, length in zip(active, lengths):
        out[row - t, ref_col - l] = (length - 1) ** 2 % 10
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    nt, ns = _counts(rng, num_train, num_test)
    background = int(rng.choice(BACKGROUNDS))
    colors = [c for c in range(10) if c != background]
    fixed = tuple(int(c) for c in rng.choice(colors, 3, replace=False))
    changing_palette = bool(rng.integers(2))
    sides_mode = str(rng.choice(("left", "right", "mixed", "mixed")))
    puzzle, used = {"train": [], "test": []}, set()
    first_fg = None
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            if changing_palette:
                roles = tuple(int(c) for c in rng.choice(colors, 3, replace=False))
                if split == "train" and index == 1 and roles[0] == first_fg:
                    # A deterministic swap makes the varying convention visible
                    # even with the minimum two demonstrations.
                    roles = (roles[1], roles[0], roles[2])
            else:
                roles = fixed
            palette = (background,) + roles
            if split == "train" and index == 0:
                first_fg = roles[0]
                lengths = [int(x) for x in rng.permutation(LENGTHS)]
                modes = ["disjoint"] * 3
            elif split == "train" and index == 1:
                n = int(rng.integers(2, 6))
                lengths = [3, 4] + [int(rng.choice(LENGTHS)) for _ in range(n - 2)]
                rng.shuffle(lengths)
                modes = ["overlap"] * n
            else:
                n = int(rng.integers(1, 6))
                lengths = [int(rng.choice(LENGTHS)) for _ in range(n)]
                modes = [str(rng.choice(("disjoint", "overlap"))) for _ in range(n)]
                if split == "test":
                    lengths[0] = int(rng.choice((3, 4)))
                    modes[0] = "weave"
            for _ in range(32):
                pair = _make_pair(rng, palette, sides_mode, lengths, modes)
                key = _key(pair["input"])
                if key in used:
                    continue
                info = _inspect_pair(pair)
                if split == "test" and not info["internal_hole"]:
                    continue
                used.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded unique construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = (_array(pair[k]) for k in ("input", "output"))
    parsed = _parse(inp)
    merged, bars = _restored(parsed)
    expected = _render(parsed, merged, bars)
    _require(np.array_equal(out, expected), "registration/union/crop/digit mismatch")
    blank = _render(parsed, merged, bars, "none")
    _require(np.count_nonzero(blank != out) == len(bars), "one visible label per restored bar")
    _require(len(merged) == len(parsed["anchor"]) + len(parsed["mover"])
             - len(parsed["anchor"] & parsed["mover"]), "union pixel conservation")
    fg = parsed["foreground"]
    # Codes may equal foreground: compare the unlabeled scene for conservation.
    _require(np.count_nonzero(blank == fg) == len(merged), "original data changed")
    _require(np.count_nonzero(blank == parsed["reference_color"]) == 1, "reference lost")
    _require(all(np.any(blank[edge] != parsed["background"])
                 for edge in ((0, slice(None)), (-1, slice(None)),
                              (slice(None), 0), (slice(None), -1))), "crop is not tight")
    boxes = parsed["boxes"]
    xs = boxes[0][3] < boxes[1][2] or boxes[1][3] < boxes[0][2]
    ys = boxes[0][1] < boxes[1][0] or boxes[1][1] < boxes[0][0]
    stage = "diagonal" if xs and ys else "horizontal" if xs else "vertical"
    hole = lambda cells: max(cells) - min(cells) + 1 > len(cells)
    return {"parsed": parsed, "bars": bars, "stage": stage,
            "internal_hole": any(hole(b[k]) for b in bars for k in ("a", "b")),
            "overlap": sum(b["overlap"] for b in bars),
            "lengths": {b["length"] for b in bars},
            "sides": {-1 if b["start"] < parsed["reference"][1] else 1 for b in bars},
            "ablation_a": not np.array_equal(out, _render(parsed, merged, bars, length_mode="largest")),
            "sum_wrong": not np.array_equal(out, _render(parsed, merged, bars, length_mode="sum")),
            "ablation_b": not np.array_equal(out, blank)}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    infos, keys = {"train": [], "test": []}, set()
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact pair schema")
            inp, out = (_array(pair[k]) for k in ("input", "output"))
            _require(_key(inp) not in keys, "duplicate input")
            keys.add(_key(inp))
            info = _inspect_pair(pair)
            _require(info["ablation_a"] and info["ablation_b"], "both parents must affect each answer")
            infos[split].append(info)
    all_infos = infos["train"] + infos["test"]
    _require(len({i["parsed"]["background"] for i in all_infos}) == 1, "background changes within puzzle")
    _require(set.union(*(i["lengths"] for i in infos["train"])) == set(LENGTHS), "teach all codes")
    _require(any(i["overlap"] and i["sum_wrong"] for i in infos["train"]), "teach union rather than addition")
    _require(any(not i["overlap"] for i in infos["train"]), "teach complementary assembly")
    _require(all(not i["internal_hole"] for i in infos["train"]), "training shift changed")
    _require(all(i["internal_hole"] for i in infos["test"]), "test must recombine internally gapped shards")
    # Registration is also taught: coincident guides or the opposite adjacent
    # order must not reproduce all training scenes after a tight crop.
    for delta in (-1, 1, 2):
        differences = []
        for info in infos["train"]:
            p = info["parsed"]
            merged, bars = _restored(p)
            mutant = dict(p)
            if p["reference_owner"] == 1:
                y, x = p["reference"]
                mutant["reference"] = (y, x + delta)
            wrong_cells = p["anchor"] | _shift(p["mover"], dx=delta)
            differences.append(not np.array_equal(_render(p, merged, bars, "none"),
                                                  _render(mutant, wrong_cells, bars, "none")))
        _require(any(differences), "registration offset not demonstrated")
    train_sides = set.union(*(i["sides"] for i in infos["train"]))
    _require(set.union(*(i["sides"] for i in infos["test"])) <= train_sides, "teach projection sides")
    palettes = [tuple(i["parsed"][k] for k in ("foreground", "guide_color", "reference_color"))
                for i in infos["train"]]
    test_palettes = [tuple(i["parsed"][k] for k in ("foreground", "guide_color", "reference_color"))
                     for i in infos["test"]]
    _require(len(set(palettes)) > 1 or set(test_palettes) <= set(palettes), "teach palette variation")
    # Use observed training digits as a lookup, independently of the arithmetic
    # expression, to confirm that test codes require no numerical extrapolation.
    learned = {}
    for pair, info in zip(puzzle["train"], infos["train"]):
        p = info["parsed"]
        merged, bars = _restored(p)
        t, _, l, _ = _bbox(merged | {p["reference"]})
        for bar in bars:
            code = int(np.asarray(pair["output"])[bar["row"] - t, p["reference"][1] - l])
            _require(learned.setdefault(bar["length"], code) == code, "inconsistent training digits")
    for pair, info in zip(puzzle["test"], infos["test"]):
        p = info["parsed"]
        merged, bars = _restored(p)
        prediction = _render(p, merged, bars, "none")
        t, _, l, _ = _bbox(merged | {p["reference"]})
        for bar in bars:
            prediction[bar["row"] - t, p["reference"][1] - l] = learned[bar["length"]]
        _require(np.array_equal(prediction, pair["output"]), "untaught test code")
    for mode, length_mode in (("adjacency", "union"), ("length_square", "union"),
                              ("color", "union"), ("square", "largest"), ("square", "sum")):
        _require(any(not np.array_equal(np.asarray(pair["output"]),
                                       _render(info["parsed"], *_restored(info["parsed"]), mode, length_mode))
                     for pair, info in zip(puzzle["train"], infos["train"])),
                 "training leaves a simpler arithmetic/measurement shortcut")


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def self_test() -> dict:
    """Bounded 42-seed, streaming semantic/diversity and mutation checks.

    At most two puzzles (<=28 grids apiece, <=25,200 int8 grid bytes each) are
    live for repeat checks. No files, subprocesses, traces, or network are used.
    """
    # Literal regression constructed by hand: one disjoint length-two bar,
    # overlapping length-three/four bars, and a reference in the moving patch.
    # This exercises the parser, guide offset, union, crop, and all three codes
    # without invoking any of the scene-construction helpers.
    fixture_input = ("702000000", "700000000", "702200000", "700000000",
                     "702220000", "700000000", "700000000", "000000000",
                     "000000000", "700020000", "700000000", "700022000",
                     "700000000", "700022200", "700000000", "700000008")
    fixture_output = ("220001", "000000", "222004", "000000", "222209",
                      "000000", "000008")
    array_rows = lambda rows: np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    _require(np.array_equal(solve(array_rows(fixture_input)), array_rows(fixture_output)),
             "hand-checked registration/arithmetic regression failed")
    fingerprints, shapes, out_shapes, structures, palettes = set(), set(), set(), set(), set()
    stages, heights, counts, sides, owners, backgrounds = set(), set(), set(), set(), set(), set()
    overlaps = holes = pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic seed")
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                p = info["parsed"]
                grid = pair["input"]
                _require(np.array_equal(solve(grid.tolist()), pair["output"]), "list input solve failed")
                before = grid.copy()
                solve(grid)
                _require(np.array_equal(grid, before), "solver mutated input")
                shapes.add(tuple(grid.shape))
                out_shapes.add(tuple(pair["output"].shape))
                stages.add(info["stage"])
                heights.add(p["guides"][0][2] - p["guides"][0][0] + 1)
                counts.add(len(info["bars"]))
                sides.update(info["sides"])
                owners.add(p["reference_owner"])
                backgrounds.add(p["background"])
                palettes.add(tuple(p[k] for k in ("background", "foreground", "guide_color", "reference_color")))
                # Hash normalized structures, retaining no duplicate grid payloads.
                role = {p["background"]: 0, p["foreground"]: 1,
                        p["guide_color"]: 2, p["reference_color"]: 3}
                normalized = bytes(role[int(c)] for c in grid.ravel())
                structures.add((tuple(grid.shape), hashlib.sha256(normalized).digest()))
                overlaps += info["overlap"] > 0
                holes += info["internal_hole"]
                pairs += 1
                # A single wrong label is a semantic mutation, not just a schema
                # violation: the validator must reject it.
                bar = info["bars"][0]
                merged, _ = _restored(p)
                t, _, l, _ = _bbox(merged | {p["reference"]})
                wrong = pair["output"].copy()
                y, x = bar["row"] - t, p["reference"][1] - l
                wrong[y, x] = (int(wrong[y, x]) + 1) % 10
                try:
                    _inspect_pair({"input": grid, "output": wrong})
                except AssertionError:
                    pass
                else:
                    raise AssertionError("wrong arithmetic accepted")
    _require(len(fingerprints) == 42, "duplicate seed puzzles")
    _require(len(shapes) >= 45 and len(out_shapes) >= 25, "insufficient shape variation")
    _require(len(structures) >= 180 and len(palettes) >= 35, "insufficient structural/palette variation")
    _require(stages == {"horizontal", "vertical", "diagonal"}, "missing staging geometry")
    _require(heights >= {5, 7, 9, 11, 13} and counts == set(range(1, 6)), "missing guide/count variation")
    _require(sides == {-1, 1} and owners == {0, 1} and len(backgrounds) == 7, "missing role variation")
    _require(overlaps >= 42 and holes == 42, "missing union/shift evidence")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(10000 + nt * 10 + ns, nt, ns)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "counts ignored")
    for nt, ns in ((None, None), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(71, nt, ns)
        _require(nt is None or len(puzzle["train"]) == nt, "partial train count ignored")
        _require(ns is None or len(puzzle["test"]) == ns, "partial test count ignored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), ("4", 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(shapes),
            "output_shapes": len(out_shapes), "structures": len(structures),
            "palettes": len(palettes), "guide_lengths": sorted(heights),
            "bar_counts": sorted(counts), "staging": sorted(stages),
            "overlap_pairs": overlaps, "internally_gapped_tests": holes}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
