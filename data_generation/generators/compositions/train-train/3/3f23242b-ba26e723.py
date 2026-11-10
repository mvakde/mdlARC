"""Scaffolds remove dominoes from a visibly calibrated rail rhythm.

Only NumPy and the standard library are used. No parent code is imported.
All conventions needed by solve are present in the single input.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    todo = {tuple(map(int, x)) for x in np.argwhere(mask)}
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for pt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if pt in todo:
                    todo.remove(pt)
                    stack.append(pt)
        result.append(sorted(cells))
    return result


def _parse_canonical(a, background):
    """Recognize a left-to-right rail; reject the other three rotations."""
    candidates = []
    for q, row in enumerate(a):
        occupied = np.flatnonzero(row != background)
        if len(occupied) < 13:
            continue
        l, end = int(occupied[0]), int(occupied[-1])
        if len(occupied) != end - l + 1 or not (1 <= q < a.shape[0] - 1):
            continue
        values, counts = np.unique(row[l:end + 1], return_counts=True)
        if len(values) != 2 or sorted(map(int, counts)) != [2, end - l - 1]:
            continue
        highlight = int(values[np.argmin(counts)])
        structure = int(values[np.argmax(counts)])
        marks = np.flatnonzero(row[l:end + 1] == highlight).tolist()
        if marks[0] != 0 or marks[1] not in (3, 4, 5):
            continue
        p, n = marks[1], end - l + 1
        sides, rail_mask = [], np.zeros(a.shape, dtype=bool)
        valid = True
        for i in range(n):
            c, color = l + i, int(a[q, l + i])
            teeth = [s for s in (-1, 1) if a[q + s, c] == color]
            if len(teeth) != 1 or a[q - teeth[0], c] != background:
                valid = False
                break
            sides.append(teeth[0])
            rail_mask[q, c] = rail_mask[q + teeth[0], c] = True
        if not valid or set(sides) != {-1, 1}:
            continue
        residual = (a != background) & ~rail_mask
        # Markers have colors different from both rail roles. They are isolated
        # bars, so their shape and color identify them without a learned palette.
        if np.any(residual & np.isin(a, [structure, highlight])):
            continue
        anchors = []
        for cells in _components(residual):
            rs = {r for r, _ in cells}
            cs = [c for _, c in cells]
            colors = {int(a[r, c]) for r, c in cells}
            if len(rs) != 1 or len(colors) != 1 or len(cells) not in (1, 3):
                valid = False
                break
            r = next(iter(rs))
            c = (min(cs) + max(cs)) // 2
            half = len(cells) // 2 + 2
            side = -1 if r < q else 1
            cap = r + 2 * side
            lo, hi = c - half, c + half
            if (max(cs) - min(cs) + 1 != len(cells) or
                    not 2 <= abs(r - q) <= 4 or
                    not 0 <= cap < a.shape[0] or
                    lo < l + p + 1 or hi >= end):
                valid = False
                break
            anchors.append({"r": r, "c": c, "half": half, "side": side,
                            "color": next(iter(colors)), "lo": lo, "hi": hi,
                            "length": len(cells)})
        if not valid or not 1 <= len(anchors) <= 4:
            continue
        anchors.sort(key=lambda x: x["lo"])
        if any(b["lo"] - a0["hi"] < 2 for a0, b in zip(anchors, anchors[1:])):
            continue
        blocked = np.zeros(n, dtype=bool)
        for anchor in anchors:
            blocked[anchor["lo"] - l:anchor["hi"] - l + 1] = True
        free = np.flatnonzero(~blocked).tolist()
        selected = free[::p]
        # The calibration must be the first two unoccluded beats.
        if selected[:2] != [0, p] or len(selected) < 3:
            continue
        candidates.append({"a": a, "background": background, "structure": structure,
                           "highlight": highlight, "q": q, "l": l, "n": n,
                           "period": p, "sides": tuple(sides), "anchors": anchors,
                           "blocked": blocked, "free": free, "selected": selected})
    return candidates


def _parse(input_grid):
    original = _grid(input_grid)
    values, counts = np.unique(original, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    background = int(values[np.argmax(counts)])
    candidates = []
    for turns in range(4):
        for info in _parse_canonical(np.rot90(original, turns).copy(), background):
            info["turns"] = turns
            candidates.append(info)
    _require(len(candidates) == 1, "input must identify one directed toothed rail")
    return candidates[0]


def _render(info, selected=None, frames=True):
    """Render in the input's canonical coordinates, then undo normalization."""
    a, q, l, n = info["a"], info["q"], info["l"], info["n"]
    bg, structure, highlight = (info[x] for x in ("background", "structure", "highlight"))
    out = a.copy()
    out[q, :] = structure  # A's common foundation reaches both canvas edges.
    for i in info["selected"] if selected is None else selected:
        out[q, l + i] = out[q + info["sides"][i], l + i] = highlight
    if frames:
        for anchor in info["anchors"]:
            r, c, side, color, lo, hi = (anchor[x] for x in ("r", "c", "side", "color", "lo", "hi"))
            # Omit covered dominoes, including teeth on the opposite side.
            out[q - 1, lo:hi + 1] = bg
            out[q + 1, lo:hi + 1] = bg
            out[q, lo:hi + 1] = color
            cap = r + 2 * side
            out[cap, lo:hi + 1] = color
            out[r + side, c] = color
            for rr in range(min(q, cap) + 1, max(q, cap)):
                out[rr, lo] = out[rr, hi] = structure
    return np.rot90(out, -info["turns"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer colors, direction, period, markers and blocked beats from input."""
    return _render(_parse(input_grid))


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             f"{name} must be an integer")
    value = int(value)
    _require(low <= value <= high, f"{name} must be in {low}..{high}")
    return value


def _partition(rng, total, bins):
    return list(map(int, rng.multinomial(total, np.full(bins, 1 / bins))))


def _build(rng, period, palette, count, rotation, witness=False, simple=False):
    bg, structure, highlight, marker0, marker1 = palette
    if witness:
        lengths = [3 if period == 5 else 1]
    elif count == 2 and simple:
        lengths = [1, 3]
        rng.shuffle(lengths)
    else:
        lengths = list(map(int, rng.choice([1, 3], size=count)))
        while (period + 1 + sum(x + 4 for x in lengths) + count > 30 or
               sum(x + 4 for x in lengths) + 2 * period + 1 > 30):
            lengths[lengths.index(3)] = 1
    widths = [x + 4 for x in lengths]
    tail = period + 1 if witness else 1
    minimum = max(period + 1 + sum(widths) + count - 1 + tail,
                  sum(widths) + (3 if witness else 2) * period + 1)
    n = int(rng.integers(max(17, minimum), 31))
    slack = n - minimum
    gaps = _partition(rng, slack, count + 1)
    # First witness starts between beats, so restarting the counter differs.
    if witness:
        choices = [x for x in range(period + 1, n - widths[0] - tail + 1)
                   if x % period != 0]
        start = int(rng.choice(choices))
    else:
        start = period + 1 + gaps[0]
    intervals = []
    for j, width in enumerate(widths):
        intervals.append((start, start + width - 1))
        start += width + 1 + (gaps[j + 1] if j + 1 < count else 0)
    if witness:
        sides = [-1 if bool(rng.integers(2)) else 1]
        depths = [2]
    else:
        sides = [(-1 if j % 2 == 0 else 1) for j in range(count)]
        if rng.integers(2):
            sides = [-x for x in sides]
        if not simple and rng.random() < .35:
            sides = [int(rng.choice([-1, 1])) for _ in range(count)]
        depths = [3 + (j % 2) if simple else int(rng.integers(2, 5)) for j in range(count)]
    top = max([1] + [d + 2 for s, d in zip(sides, depths) if s == -1])
    bottom = max([1] + [d + 2 for s, d in zip(sides, depths) if s == 1])
    outer_top, outer_bottom = map(int, rng.integers(0, 3, size=2))
    left, right = map(int, rng.integers(0, min(3, 31 - n), size=2))
    while n + left + right > 30:
        if right:
            right -= 1
        else:
            left -= 1
    q = top + outer_top
    a = np.full((top + bottom + 1 + outer_top + outer_bottom, n + left + right), bg, dtype=np.int8)
    tooth_sides = [int(rng.choice([-1, 1]))]
    for i in range(1, n):
        repeat = not witness and i > 1 and rng.random() < .15
        if simple and i == 1:
            repeat = True  # The second demonstration visibly breaks alternation.
        if len(tooth_sides) >= 3 and len(set(tooth_sides[-3:])) == 1:
            repeat = False
        tooth_sides.append(tooth_sides[-1] if repeat else -tooth_sides[-1])
    for i, side in enumerate(tooth_sides):
        color = highlight if i in (0, period) else structure
        a[q, left + i] = a[q + side, left + i] = color
    for j, ((lo, hi), length, side, depth) in enumerate(zip(intervals, lengths, sides, depths)):
        c, r = left + (lo + hi) // 2, q + side * depth
        color = marker0 if witness or j % 2 == 0 else marker1
        a[r, c - length // 2:c + length // 2 + 1] = color
    return np.rot90(a, rotation).copy()


def _absolute_selected(info):
    return [i for i in range(info["n"]) if not info["blocked"][i] and i % info["period"] == 0]


def _reset_selected(info):
    result, rank = [], 0
    for i in range(info["n"]):
        if info["blocked"][i]:
            rank = 0
        else:
            if rank % info["period"] == 0:
                result.append(i)
            rank += 1
    return result


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    period = int(rng.choice([3, 4, 5]))
    palette = tuple(map(int, rng.choice(10, size=5, replace=False)))
    rotation = int(rng.integers(4))
    rotations = [rotation, (rotation + 1) % 4] if rng.random() < .65 else [rotation]
    puzzle, used = {"train": [], "test": []}, set()
    for split, total in (("train", nt), ("test", ne)):
        for index in range(total):
            witness = split == "train" and index == 0
            simple = split == "train" and index == 1
            if witness:
                count = 1
            elif simple:
                count = 2
            elif split == "train":
                count = int(rng.choice([1, 2]))
            elif index == 0:
                count = int(rng.choice([3, 4] if period < 5 else [3]))
            else:
                count = int(rng.choice([1, 2, 3, 4] if period < 5 else [1, 2, 3]))
            rot = rotations[index % len(rotations)] if split == "train" else int(rng.choice(rotations))
            for _ in range(200):
                inp = _build(rng, period, palette, count, rot, witness, simple)
                info = _parse(inp)
                if _key(inp) in used or info["selected"] == _absolute_selected(info):
                    continue
                if witness and info["selected"] == _reset_selected(info):
                    continue
                out = _render(info)
                puzzle[split].append({"input": inp, "output": out})
                used.add(_key(inp))
                break
            else:
                raise RuntimeError("could not sample an informative unique rail")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            infos[split].append(info)
            _require(np.array_equal(out, _render(info)), "output disagrees with the rule")
            canonical = np.rot90(out, info["turns"])
            q, l = info["q"], info["l"]
            # Check actual cell relations as well as renderer agreement.
            for anchor in info["anchors"]:
                r, c, lo, hi, s, col = (anchor[x] for x in ("r", "c", "lo", "hi", "side", "color"))
                _require(np.all(canonical[q, lo:hi + 1] == col), "foundation segment erased")
                _require(np.all(canonical[r + 2 * s, lo:hi + 1] == col), "cap incomplete")
                _require(canonical[r + s, c] == col, "cap projection missing")
                k = anchor["length"] // 2
                _require(np.all(canonical[r, c - k:c + k + 1] == col), "marker not preserved")
                _require(canonical[q + s, lo] == info["structure"] and
                         canonical[q + s, hi] == info["structure"], "supports fail to reach foundation")
                _require(np.all(canonical[q - s, lo:hi + 1] == info["background"]), "opposite teeth not occluded")
            selected = set(info["selected"])
            for i in info["free"]:
                col = info["highlight"] if i in selected else info["structure"]
                _require(canonical[q, l + i] == canonical[q + info["sides"][i], l + i] == col,
                         "exposed domino is not recolored as a whole")
            _require(np.all(canonical[q, :l] == info["structure"]) and
                     np.all(canonical[q, l + info["n"]:] == info["structure"]), "line does not reach edges")
            _require(info["selected"] != _absolute_selected(info), "geometry does not influence cadence")
            _require(not np.array_equal(out, _render(info, frames=False)), "A ablation did not change output")
            # Keep geometry, but omit B's continuation beyond the calibration.
            _require(not np.array_equal(out, _render(info, selected=[0, info["period"]])),
                     "B ablation did not change output")
    all_infos = infos["train"] + infos["test"]
    conventions = {(i["background"], i["structure"], i["highlight"], i["period"]) for i in all_infos}
    _require(len(conventions) == 1, "puzzle conventions changed")
    train = infos["train"]
    _require(len(train[0]["anchors"]) == 1 and len(train[1]["anchors"]) == 2, "missing scaffold demonstrations")
    _require(len(train[0]["selected"]) >= 4, "isolated demonstration needs two new beats")
    _require(any(a == b for a, b in zip(train[1]["sides"], train[1]["sides"][1:])),
             "second demonstration must expose tooth-side repetition")
    _require(train[0]["selected"] != _reset_selected(train[0]), "training permits rhythm reset")
    _require({a["length"] for i in train for a in i["anchors"]} == {1, 3}, "training lacks marker width examples")
    _require({a["side"] for i in train for a in i["anchors"]} == {-1, 1}, "training lacks both rail sides")
    _require({abs(a["r"] - i["q"]) for i in train for a in i["anchors"]} == {2, 3, 4}, "training lacks support heights")
    _require(len(infos["test"][0]["anchors"]) > max(len(i["anchors"]) for i in train), "missing count shift")
    trained_rotations = {i["turns"] for i in train}
    trained_colors = {a["color"] for i in train for a in i["anchors"]}
    _require(all(i["turns"] in trained_rotations for i in infos["test"]), "test orientation undemonstrated")
    _require(all(a["color"] in trained_colors for i in infos["test"] for a in i["anchors"]), "new test marker color")
    return None


def _signature(puzzle):
    return tuple((s, _key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded family checks, 42 seeds, counts, determinism, and mutation rejection."""
    # A short rail outside the generator's length schedule: boundary cap,
    # long same-side tooth run, background 9, structural color 0, and margins.
    edge = np.full((6, 18), 9, dtype=np.int8)
    for index in range(13):
        side = -1 if index < 8 else 1
        color = 3 if index in (0, 3) else 0
        edge[4, 3 + index] = edge[4 + side, 3 + index] = color
    edge[2, 9] = 5
    result = solve(edge)
    _require(np.flatnonzero(result[4] == 3).tolist() == [3, 6, 14], "short rail counts margins or covered cells")
    _require(np.all(result[0, 7:12] == 5) and np.all(result[4, 7:12] == 5),
             "boundary scaffold clipped")
    _require(np.all(result[5, 7:12] == 9), "covered opposite teeth survived")
    _require(np.all(result[4, :3] == 0) and np.all(result[4, 16:] == 0), "margin extension failed")
    for turns in range(4):
        for reflected in (False, True):
            inp = np.fliplr(edge) if reflected else edge
            out = np.fliplr(result) if reflected else result
            _require(np.array_equal(solve(np.rot90(inp, turns)), np.rot90(out, turns)),
                     "rule is not covariant under rotation/reflection")
    # Three different marker colors and all scaffolds on the same rail side
    # are legal even though generation uses two marker colors per puzzle.
    multicolor = np.full((8, 30), 0, dtype=np.int8)
    for index in range(30):
        side = -1 if index % 2 else 1
        color = 6 if index in (0, 3) else 4
        multicolor[6, index] = multicolor[6 + side, index] = color
    multicolor[4, 6] = 2
    multicolor[3, 14:17] = 5
    multicolor[2, 23] = 7
    result = solve(multicolor)
    _require(np.flatnonzero(result[6] == 6).tolist() == [0, 3, 11, 26, 29], "multicolor foundations alter rank incorrectly")
    _require(result[6, 6] == 2 and result[6, 15] == 5 and result[6, 23] == 7,
             "shared line erased a differently colored foundation")
    shapes, signatures, structures, palettes = set(), set(), set(), set()
    periods, rotations, counts, lengths, depths, margins = set(), set(), set(), set(), set(), set()
    runs = False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "not reproducible")
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                frozen = inp.copy()
                _require(np.array_equal(solve(inp), pair["output"]) and np.array_equal(inp, frozen), "solver mutated input")
                i = _parse(inp)
                shapes.add(inp.shape)
                periods.add(i["period"])
                rotations.add(i["turns"])
                counts.add(len(i["anchors"]))
                lengths.update(a["length"] for a in i["anchors"])
                depths.update(abs(a["r"] - i["q"]) for a in i["anchors"])
                margins.add((i["l"] > 0, i["l"] + i["n"] < i["a"].shape[1]))
                palettes.add((i["background"], i["structure"], i["highlight"]))
                structures.add((i["n"], i["sides"], tuple((a["r"] - i["q"], a["lo"] - i["l"], a["length"]) for a in i["anchors"])))
                runs |= any(a == b for a, b in zip(i["sides"], i["sides"][1:]))
    _require(len(signatures) == 42 and len(shapes) >= 40 and len(structures) >= 150, "structural variation collapsed")
    _require(periods == {3, 4, 5} and rotations == {0, 1, 2, 3}, "period/direction coverage missing")
    _require(counts == {1, 2, 3, 4} and lengths == {1, 3} and depths == {2, 3, 4}, "scaffold geometry coverage missing")
    _require(len(palettes) >= 30 and len(margins) == 4 and runs, "palette/margin/tooth variation missing")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345 + nt * 10 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "counts not honored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0)
    original = puzzle["train"][0]["output"].copy()
    i = _parse(puzzle["train"][0]["input"])
    targets = [(i["q"], i["anchors"][0]["lo"]),
               (i["anchors"][0]["r"] + 2 * i["anchors"][0]["side"], i["anchors"][0]["c"]),
               (i["q"] + i["sides"][i["selected"][-1]], i["l"] + i["selected"][-1])]
    for cell in targets:
        corrupted = np.rot90(original, i["turns"]).copy()
        corrupted[cell] = i["background"]
        puzzle["train"][0]["output"] = np.rot90(corrupted, -i["turns"]).copy()
        try:
            validate_generated_puzzle(puzzle)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt foundation/cap/domino accepted")
    puzzle["train"][0]["output"] = original
    validate_generated_puzzle(puzzle)
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "structures": len(structures), "palettes": len(palettes),
            "periods": sorted(periods), "directions": sorted(rotations),
            "scaffold_counts": sorted(counts), "marker_lengths": sorted(lengths),
            "support_depths": sorted(depths)}
