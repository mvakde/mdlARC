"""Repair casts: periodic evidence reveals the objects that are routed.

Only Python's standard library and NumPy are used. Generation labels come from
the clean construction; solve rediscovers the repair and all routing geometry
from the query's pixels. No parent module or saved example is imported.
"""
from itertools import combinations
import json

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _count(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, np.integer)) and low <= int(value) <= high,
             f"{name} must be an integer in {low}..{high}")
    return int(value)


def _components(labels, diagonal=False):
    """Components of a sparse coordinate -> intended-color picture."""
    unseen = set(labels)
    result = []
    neighbors = _EIGHT if diagonal else _FOUR
    while unseen:
        start = min(unseen)
        color = int(labels[start])
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in unseen and labels[p] == color:
                    unseen.remove(p)
                    todo.append(p)
        result.append({"color": color, "cells": tuple(sorted(cells))})
    return result


def _repair_motif(motif, frame, accent):
    """All strict-majority, sparse-error explanations must agree on pixels.

    At least three copies of each phase are required. Redundant multiples of
    a primitive period are harmless, but equally supported different repairs
    are not resolved by an arbitrary shortest-period tie-break.
    """
    cross, length = motif.shape
    _require(set(map(int, np.unique(motif))) == {frame, accent}, "motif must be binary")
    limit = max(1, motif.size // 8)
    candidates = []
    for period in range(1, length // 3 + 1):
        unit = np.empty((cross, period), dtype=np.int8)
        errors, supported = 0, True
        for phase in range(period):
            values = motif[:, phase::period]
            votes = np.count_nonzero(values == accent, axis=1)
            copies = values.shape[1]
            if np.any(2 * votes == copies):
                supported = False
                break
            unit[:, phase] = np.where(2 * votes > copies, accent, frame)
            errors += int(np.minimum(votes, copies - votes).sum())
        if not supported or errors > limit or len(np.unique(unit)) != 2:
            continue
        repaired = unit[:, np.arange(length) % period]
        candidates.append((period, repaired, errors))
    _require(bool(candidates), "no supported periodic repair")
    answer = candidates[0][1]
    _require(all(np.array_equal(answer, c[1]) for c in candidates), "ambiguous periodic repair")
    return answer.copy(), candidates[0][0]


def _cast(repaired_grid, components, bands):
    out = repaired_grid.copy()
    lookup = {b["route"]: b for b in bands}
    occupied, targets = set(), []
    for comp in components:
        _require(comp["color"] in lookup, "missing destination color")
        band = lookup[comp["color"]]
        shift = band["bottom"] - max(r for r, c in comp["cells"])
        cells = {(r + shift, c) for r, c in comp["cells"]}
        _require(all(band["top"] <= r <= band["bottom"] and
                     0 < c < out.shape[1] - 1 for r, c in cells), "cast does not fit")
        _require(not cells & occupied, "casts overlap")
        for p in cells:
            out[p] = band["rail"]
        occupied.update(cells)
        targets.append(cells)
    return out, targets


def _parse_canonical(grid):
    """The canonical scene has its ribbons at the top and trays below."""
    h, w = grid.shape
    _require(h >= 9 and w >= 12, "scene too small")
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "missing near-side frame")
    end = 0
    while end + 1 < h and grid[end + 1, 0] == frame and grid[end + 1, -1] == frame:
        end += 1
    _require(4 <= end < h - 2 and np.all(grid[end] == frame), "invalid framed ribbon region")
    background = int(grid[1, 1])
    _require(background != frame, "padding must differ from frame")
    dividers = np.flatnonzero(np.all(grid[:end + 1] == frame, axis=1)).tolist()
    _require(2 <= len(dividers) <= 4 and dividers[0] == 0 and dividers[-1] == end,
             "need one to three ribbons")
    _require(all(b - a >= 4 for a, b in zip(dividers, dividers[1:])), "ribbon too thin")
    source_mask = np.zeros_like(grid, dtype=bool)
    source_mask[:end + 1] = True
    motif_mask = np.zeros_like(grid, dtype=bool)
    structure_mask = np.zeros_like(grid, dtype=bool)
    structure_mask[:end + 1, [0, -1]] = True
    structure_mask[dividers, :] = True
    repaired = grid.copy()
    motifs, accents, labels = [], [], {}
    length = w - 4
    for first, last in zip(dividers, dividers[1:]):
        rect = (slice(first + 2, last - 1), slice(2, w - 2))
        motif = grid[rect]
        cross = motif.shape[0]
        _require(1 <= cross <= 8 and length > cross, "invalid ribbon geometry")
        nonframe = [int(c) for c in np.unique(motif) if c != frame]
        _require(len(nonframe) == 1 and nonframe[0] != background, "invalid accent")
        accent = nonframe[0]
        _require(accent not in accents, "ribbon accents must be distinct")
        clean, period = _repair_motif(motif, frame, accent)
        repaired[rect] = clean
        motif_mask[rect] = True
        for r, c in np.argwhere(motif != clean):
            p = (first + 2 + int(r), 2 + int(c))
            labels[p] = int(clean[r, c])
        accents.append(accent)
        motifs.append({"top": first + 2, "cross": cross, "period": period,
                       "accent": accent, "clean": clean, "noisy": motif.copy()})
    padding_mask = source_mask & ~(motif_mask | structure_mask)
    _require(np.all(grid[padding_mask] == background), "padding must be uniform")
    _require(np.all(grid[structure_mask] == frame), "damaged frame")
    _require(bool(labels), "scene must contain a correction")
    _require(np.all(grid[end + 1] == background), "missing ribbon/tray separator")
    bands, row = [], end + 1
    while row < h:
        if np.all(grid[row] == background):
            row += 1
            continue
        rail, route = int(grid[row, 0]), int(grid[row, 1])
        _require(rail != route and background not in (rail, route) and
                 grid[row, -1] == rail and np.all(grid[row, 1:-1] == route), "invalid tray")
        top = row
        while row + 1 < h and np.array_equal(grid[row + 1], grid[top]):
            row += 1
        bands.append({"top": top, "bottom": row, "route": route, "rail": rail})
        row += 1
        _require(row == h or np.all(grid[row] == background), "trays need separators")
    _require(len(bands) == len(accents) + 1 and
             {b["route"] for b in bands} == {frame, *accents}, "tray routes must match ribbon colors")
    _require(len({b["route"] for b in bands}) == len(bands), "duplicate route")
    components = _components(labels)
    output, targets = _cast(repaired, components, bands)
    return {"canonical": grid, "frame": frame, "background": background,
            "end": end, "motifs": motifs, "bands": bands, "labels": labels,
            "components": components, "targets": targets, "repaired": repaired,
            "expected": output, "motif_mask": motif_mask,
            "source_mask": source_mask, "length": length}


def _analyze_input(value):
    grid = _grid(value)
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(grid, rotation).copy()
        try:
            a = _parse_canonical(canonical)
        except ValueError:
            continue
        a["rotation"] = rotation
        a["output"] = np.rot90(a["expected"], -rotation).copy()
        candidates.append(a)
    _require(len(candidates) == 1, "scene needs one visible direction and interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the repaired pattern, discrepancy components and trays from input."""
    return _analyze_input(input_grid)["output"]


def _shape(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _irregular(comp):
    s = _shape(comp["cells"])
    return len(s) < (1 + max(r for r, c in s)) * (1 + max(c for r, c in s))


def _independent(a):
    comps = a["components"]
    return any(x["color"] == y["color"] and
               max(r for r, c in x["cells"]) != max(r for r, c in y["cells"])
               for i, x in enumerate(comps) for y in comps[i + 1:])


def _color_witness(a):
    return len({int(color) for p, color in a["labels"].items()
                if a["canonical"][p] == a["frame"] and color != a["frame"]}) >= 2


def _reverse_witness(a):
    return any(color == a["frame"] for color in a["labels"].values())


def _color_contact(a):
    labels = a["labels"]
    return any((r + dr, c + dc) in labels and labels[(r + dr, c + dc)] != color
               for (r, c), color in labels.items() for dr, dc in _FOUR)


def _same_connectivity(a):
    return {c["cells"] for c in a["components"]} == {
        c["cells"] for c in _components(a["labels"], True)}


def _shortcut_differs(a, shortcut):
    if shortcut == "observed_color":
        labels = {p: int(a["canonical"][p]) for p in a["labels"]}
        comps = _components(labels)
    elif shortcut == "whole_color":
        comps = [{"color": color, "cells": tuple(p for p, c in a["labels"].items() if c == color)}
                 for color in sorted(set(a["labels"].values()))]
    elif shortcut == "bounding_boxes":
        comps = []
        for comp in a["components"]:
            cells = comp["cells"]
            box = tuple((r, c) for r in range(min(r for r, c in cells), max(r for r, c in cells) + 1)
                        for c in range(min(c for r, c in cells), max(c for r, c in cells) + 1))
            comps.append({"color": comp["color"], "cells": box})
    else:
        raise ValueError("unknown shortcut")
    try:
        alternative, _ = _cast(a["repaired"], comps, a["bands"])
    except ValueError:
        return True
    return not np.array_equal(alternative, a["expected"])


def _layout(rng, count, lesson, length, reserve=False):
    # Reserve enough depth for any error component in each ribbon. The common
    # frame-color tray must accommodate the thickest ribbon.
    choices = []
    for crosses in np.ndindex(*(8,) * count):
        crosses = tuple(x + 1 for x in crosses)
        if lesson == "color" and crosses[0] < 2:
            continue
        if lesson == "independent" and count == 1 and crosses[0] < 3:
            continue
        if 2 + 4 * count + 2 * sum(crosses) + max(crosses) <= 30:
            choices.append(crosses)
    if reserve:
        crosses = (3,) if count == 1 else (2,) + (1,) * (count - 1)
    else:
        crosses = choices[int(rng.integers(len(choices)))]
    palette = list(map(int, rng.permutation(10)))
    background, frame = palette[:2]
    accents = palette[2:count + 2]
    routes = [frame] + accents
    rails = [int(rng.choice([c for c in range(10) if c not in (background, route)]))
             for route in routes]
    order = list(map(int, rng.permutation(count + 1)))
    depths = [max(crosses)] + list(crosses)
    gaps = [1] * (count + 1)
    end = sum(c + 3 for c in crosses)
    margin = 0
    budget = 30 - (end + 1 + sum(depths) + sum(gaps))
    for _ in range(int(rng.integers(budget + 1))):
        kind = int(rng.integers(3))
        if kind == 0:
            depths[int(rng.integers(count + 1))] += 1
        elif kind == 1 and min(gaps) < 3:
            eligible = [i for i, g in enumerate(gaps) if g < 3]
            gaps[int(rng.choice(eligible))] += 1
        else:
            margin += 1
    grid = np.full((end + 1 + sum(depths) + sum(gaps) + margin, length + 4),
                   background, dtype=np.int8)
    grid[:end + 1, [0, -1]] = frame
    motifs, cursor = [], 0
    grid[0] = frame
    for cross, accent in zip(crosses, accents):
        last = cursor + cross + 3
        grid[last] = frame
        motifs.append({"top": cursor + 2, "cross": cross, "accent": accent})
        cursor = last
    bands, cursor = [], end + 1
    for gap, i in zip(gaps, order):
        cursor += gap
        depth, route, rail = depths[i], routes[i], rails[i]
        b = {"top": cursor, "bottom": cursor + depth - 1, "route": route, "rail": rail}
        grid[cursor:cursor + depth] = route
        grid[cursor:cursor + depth, [0, -1]] = rail
        bands.append(b)
        cursor += depth
    return grid, motifs, bands, frame, background, end


def _make_units(rng, motifs, frame, length, reserve):
    periods = []
    for i, m in enumerate(motifs):
        if reserve:
            period = 3 if i == 0 else 2
            bits = (np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8)[:m["cross"]]
                    if i == 0 else np.tile([1, 0], (m["cross"], 1)))
        else:
            available = list(range(2, min(6, length // 4) + 1))
            if i == 1 and len(available) > 1:
                available.remove(periods[0])
            period = int(rng.choice(available))
            bits = rng.integers(0, 2, (m["cross"], period), dtype=np.int8)
            # Row shifts and repeated row words give structured units as well
            # as free binary masks. No unit encodes a destination index.
            if rng.random() < .35:
                bits = np.stack([np.roll(bits[0], int(rng.integers(period)))
                                 for _ in range(m["cross"])])
            if len(np.unique(bits)) != 2:
                return False
        unit = np.where(bits, m["accent"], frame).astype(np.int8)
        m["period"] = period
        m["clean"] = unit[:, np.arange(length) % period]
        m["cap"] = max(1, m["clean"].size // 8)
        periods.append(period)
    return True


def _select_errors(rng, motifs, frame, lesson, reserve):
    errors = [set() for _ in motifs]
    phases = [set() for _ in motifs]

    def add(i, r, c):
        m = motifs[i]
        phase = (r, c % m["period"])
        if len(errors[i]) >= m["cap"] or phase in phases[i]:
            return False
        errors[i].add((r, c))
        phases[i].add(phase)
        return True

    if reserve:
        if lesson == "color":
            for r, c in ((0, 0), (0, 1), (1, 0)):
                add(0, r, c)
            for i in range(1, len(motifs)):
                add(i, 0, 2 + 2 * i)
        else:
            add(0, 0, 2)
            add(0, 1, 4) if len(motifs) == 1 else add(1, 0, 5)
            add(0, 0, 1)
        return errors

    if lesson == "color":
        clean, accent = motifs[0]["clean"], motifs[0]["accent"]
        patches = []
        for r in range(clean.shape[0] - 1):
            for c in range(clean.shape[1] - 1):
                points = [(y, x) for y, x in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
                          if clean[y, x] == accent]
                patches.extend(combinations(points, 3))
        if not patches:
            return None
        for r, c in patches[int(rng.integers(len(patches)))]:
            if not add(0, r, c):
                return None
        for i, m in enumerate(motifs[1:], 1):
            points = np.argwhere(m["clean"] == m["accent"])
            r, c = points[int(rng.integers(len(points)))]
            add(i, int(r), int(c))
    elif lesson == "independent":
        points = [(i, int(r), int(c)) for i, m in enumerate(motifs)
                  for r, c in np.argwhere(m["clean"] == frame)]
        rng.shuffle(points)
        chosen = None
        for x in points[:40]:
            for y in points:
                i, r, c = x
                j, s, d = y
                if (abs(c - d) >= 3 and motifs[i]["top"] + r != motifs[j]["top"] + s and
                        (i != j or (r, c % motifs[i]["period"]) != (s, d % motifs[j]["period"]))):
                    chosen = (x, y)
                    break
            if chosen:
                break
        if not chosen:
            return None
        for i, r, c in chosen:
            if not add(i, r, c):
                return None
        # Make the inverse-color lesson also show a touching multicolor error
        # region. It must split by intended color, not by error occupancy alone.
        accent_points = sorted({(i, r + dr, c + dc) for i, points in enumerate(errors)
                                for r, c in points for dr, dc in _FOUR
                                if 0 <= r + dr < motifs[i]["cross"] and
                                0 <= c + dc < motifs[i]["clean"].shape[1] and
                                motifs[i]["clean"][r + dr, c + dc] == motifs[i]["accent"]})
        rng.shuffle(accent_points)
        if not any(add(i, r, c) for i, r, c in accent_points):
            return None

    for i, m in enumerate(motifs):
        cap = min(m["cap"], 9)
        if lesson == "color":
            goal = int(rng.integers(len(errors[i]), cap + 1))
        elif lesson == "independent":
            goal = min(cap, len(errors[i]) + int(rng.integers(3)))
        else:
            goal = 0 if rng.random() < .15 else int(rng.integers(1, cap + 1))
        clean = m["clean"]
        for _ in range(60):
            if len(errors[i]) >= goal:
                break
            frontier = sorted({(r + dr, c + dc) for r, c in errors[i] for dr, dc in _FOUR
                               if 0 <= r + dr < clean.shape[0] and 0 <= c + dc < clean.shape[1]
                               and clean[r + dr, c + dc] == clean[r, c]})
            if frontier and rng.random() < .65:
                r, c = frontier[int(rng.integers(len(frontier)))]
            else:
                r, c = int(rng.integers(clean.shape[0])), int(rng.integers(clean.shape[1]))
            if lesson == "color" and clean[r, c] != m["accent"]:
                continue
            add(i, r, c)
    if not any(errors):
        add(0, 0, 0)
    return errors


def _build_pair(rng, count, lesson, length, turn, reserve=False):
    clean_grid, motifs, bands, frame, background, end = _layout(rng, count, lesson, length, reserve)
    if not _make_units(rng, motifs, frame, length, reserve):
        return None
    errors = _select_errors(rng, motifs, frame, lesson, reserve)
    if errors is None:
        return None
    labels = {}
    for m, points in zip(motifs, errors):
        clean_grid[m["top"]:m["top"] + m["cross"], 2:-2] = m["clean"]
        for r, c in points:
            labels[(m["top"] + r, c + 2)] = int(m["clean"][r, c])
    noisy = clean_grid.copy()
    for m, points in zip(motifs, errors):
        for r, c in points:
            noisy[m["top"] + r, c + 2] = m["accent"] if m["clean"][r, c] == frame else frame
    try:
        # Ground truth uses construction evidence, not periodic recovery.
        expected, _ = _cast(clean_grid, _components(labels), bands)
        a = _parse_canonical(noisy)
        if not np.array_equal(a["expected"], expected) or not _same_connectivity(a):
            return None
        if any(m["period"] < 2 for m in a["motifs"]):
            return None
        if lesson == "color" and not (_color_witness(a) and any(_irregular(c) for c in a["components"])):
            return None
        if lesson == "color" and len({m["period"] for m in a["motifs"]}) < 2:
            return None
        if lesson == "independent" and not (_independent(a) and _reverse_witness(a) and _color_contact(a)):
            return None
        pair = {"input": np.rot90(noisy, turn).copy(), "output": np.rot90(expected, turn).copy()}
        if not np.array_equal(solve(pair["input"]), pair["output"]):
            return None
        return pair
    except ValueError:
        return None


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train = _count(num_train, 2, 10, "num_train")
    test = _count(num_test, 1, 4, "num_test")
    _require(train > test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and
             int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    turns = list(map(int, rng.permutation(4)))
    # A held-out direction follows two demonstrated directions. Other puzzles
    # keep directions familiar and extrapolate ribbon length instead.
    holdout = bool(rng.random() < .6)
    fixed = bool(rng.random() < .5)
    first_count = int(rng.choice([2, 3]))
    second_count = int(rng.choice([c for c in (1, 3) if c != first_count]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", train), ("test", test)):
        for index in range(n):
            lesson = ("color" if index == 0 else "independent" if index == 1 else None) if split == "train" else None
            count = first_count if lesson == "color" else second_count if lesson else int(rng.integers(1, 4))
            if holdout:
                turn = turns[index % 2] if split == "train" else turns[2 + index % 2]
                length = int(rng.integers(12, 27))
            else:
                turn = turns[0] if fixed else turns[index % 2]
                length = int(rng.integers(12, 21) if split == "train" else rng.integers(21, 27))
            pair = None
            for _ in range(120):
                candidate = _build_pair(rng, count, lesson, length, turn)
                if candidate is not None and _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            if pair is None:
                # Finite, explicit reserve for sparse-noise rejection. It uses
                # the same family and lessons, with fresh geometry and roles.
                for serial in range(16):
                    candidate = _build_pair(rng, count, lesson, length, turn, reserve=True)
                    if candidate is not None and _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            _require(pair is not None, "bounded construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count(len(puzzle["train"]), 2, 10, "training count")
    _count(len(puzzle["test"]), 1, 4, "test count")
    _require(len(puzzle["train"]) > len(puzzle["test"]), "invalid count relation")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze_input(inp)
            canonical_out = np.rot90(out, a["rotation"])
            _require(np.array_equal(canonical_out, a["expected"]), "incorrect repair/cast output")
            _require(_same_connectivity(a), "diagonal-only discrepancy grouping is excluded")
            changed = a["canonical"] != canonical_out
            allowed = np.zeros_like(changed)
            for p in a["labels"]:
                allowed[p] = True
            for target in a["targets"]:
                for p in target:
                    allowed[p] = True
            _require(np.array_equal(changed, allowed), "unexpected change or invisible correction")
            _require(sum(map(len, a["targets"])) == len(a["labels"]), "repair/cast cell conservation")
            _require(np.array_equal(canonical_out[:a["end"] + 1], a["repaired"][:a["end"] + 1]),
                     "ribbon repair not preserved")
            _require(not np.array_equal(a["repaired"], a["expected"]), "A-only shortcut succeeds")
            analyses[split].append(a)
    train = analyses["train"]
    _require(any(_color_witness(a) and len({m["period"] for m in a["motifs"]}) >= 2 and
                 any(_irregular(c) for c in a["components"]) and a["frame"] not in a["labels"].values()
                 for a in train), "missing intended-color/period/mask/unused-tray lesson")
    _require(any(_shortcut_differs(a, "observed_color") for a in train), "observed-color routing not distinguished")
    _require(any(_independent(a) and _reverse_witness(a) and _color_contact(a) for a in train),
             "missing independent/reverse/multicolor lesson")
    _require(any(_shortcut_differs(a, "whole_color") for a in train), "whole-color grouping not distinguished")
    _require(any(_shortcut_differs(a, "bounding_boxes") for a in train), "mask preservation not distinguished")
    _require(len({len(a["motifs"]) for a in train}) >= 2, "training lacks ribbon-count variation")
    directions = {a["rotation"] for a in train}
    if any(a["rotation"] not in directions for a in analyses["test"]):
        _require(len(directions) >= 2, "orientation transfer needs varied training")
    else:
        _require(min(a["length"] for a in analyses["test"]) > max(a["length"] for a in train),
                 "familiar directions require a length shift")


def measure_puzzle(puzzle):
    """Small input-derived descriptors for reproducible variation inspection."""
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _analyze_input(pair["input"])
            records.append({"split": split, "shape": list(np.asarray(pair["input"]).shape),
                            "direction": a["rotation"], "ribbons": len(a["motifs"]),
                            "length": a["length"], "crosses": [m["cross"] for m in a["motifs"]],
                            "periods": [m["period"] for m in a["motifs"]],
                            "depths": [b["bottom"] - b["top"] + 1 for b in a["bands"]],
                            "errors": len(a["labels"]), "components": len(a["components"]),
                            "masks": [list(map(list, _shape(c["cells"]))) for c in a["components"]],
                            "unused": len(a["bands"]) - len(set(a["labels"].values())),
                            "irregular": sum(_irregular(c) for c in a["components"]),
                            "reuse": sum(b["rail"] in {x["route"] for x in a["bands"]} for b in a["bands"]),
                            "partial": any(a["length"] % m["period"] for m in a["motifs"]),
                            "independent": _independent(a), "contact": _color_contact(a)})
    return records


def _equal(left, right):
    return all(len(left[s]) == len(right[s]) and
               all(np.array_equal(x[k], y[k]) for x, y in zip(left[s], right[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed measurement, meaningful regressions and API checks."""
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        records.extend(measure_puzzle(p))
        signatures.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
    assert len(signatures) == 42
    assert {r["direction"] for r in records} == set(range(4))
    assert {r["ribbons"] for r in records} == {1, 2, 3}
    assert len({tuple(r["shape"]) for r in records}) >= 40
    assert len({r["length"] for r in records}) >= 10
    assert len({c for r in records for c in r["crosses"]}) >= 6
    assert {p for r in records for p in r["periods"]} >= {2, 3, 4, 5, 6}
    assert len({tuple(r["depths"]) for r in records}) >= 30
    masks = {tuple(map(tuple, m)) for r in records for m in r["masks"]}
    assert len(masks) >= 20
    assert len({r["errors"] for r in records}) >= 8
    for split in ("train", "test"):
        rs = [r for r in records if r["split"] == split]
        assert any(r["partial"] for r in rs)
        assert any(r["unused"] for r in rs)
        assert any(r["irregular"] for r in rs)
        assert any(r["independent"] for r in rs)
        assert any(r["reuse"] for r in rs)
    p = generate_puzzle(91, 7, 3)
    assert _equal(p, generate_puzzle(91, 7, 3))
    # Every explicit legal count combination is also checked by verify.py.
    for train, test in ((2, 1), (10, 1), (10, 4)):
        q = generate_puzzle(900 + train + test, train, test)
        assert len(q["train"]) == train and len(q["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reserve construction is exercised, rather than left as untested code.
    for count, lesson in ((2, "color"), (3, "color"), (1, "independent"), (3, "independent"), (2, None)):
        for length in (12, 17, 26):
            for turn in range(4):
                pair = _build_pair(np.random.default_rng(700 + count), count, lesson, length, turn, True)
                assert pair is not None and np.array_equal(solve(pair["input"]), pair["output"])
    # Equivariance follows the visible scene, not a seed or literal palette.
    sample = generate_puzzle(101, 2, 1)["train"][0]
    original = sample["input"].copy()
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(original, turn)), np.rot90(sample["output"], turn))
    permutation = np.random.default_rng(11).permutation(10)
    assert np.array_equal(solve(permutation[original]), permutation[sample["output"]])
    assert np.array_equal(sample["input"], original), "solve mutated its argument"
    # A literal 9x12 scene exercises the solver's smallest specified ribbon
    # length, which the ordinary sampler does not use. Only one cell moves
    # into the accent tray; the common-color tray remains untouched.
    small_input = ("222222222222", "200000000002", "205222525202",
                   "200000000002", "222222222222", "000000000000",
                   "722222222227", "000000000000", "855555555558")
    small_output = list(small_input)
    small_output[2] = "205252525202"
    small_output[8] = "855585555558"
    literal = lambda rows: np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    assert np.array_equal(solve(literal(small_input)), literal(small_output))
    # Explicit irregular mask, independent displacement and real collision.
    tray = {"top": 5, "bottom": 7, "route": 1, "rail": 2}
    canvas = np.ones((8, 10), dtype=np.int8)
    pieces = [{"color": 1, "cells": ((0, 2), (0, 3), (1, 2))},
              {"color": 1, "cells": ((3, 6),)}]
    out, targets = _cast(canvas, pieces, [tray])
    assert targets == [{(6, 2), (6, 3), (7, 2)}, {(7, 6)}]
    assert out[7, 3] == 1, "bounding-box filling lost an L's notch"
    pieces[1]["cells"] = ((3, 3),)
    _, overlapping_boxes = _cast(canvas, pieces, [tray])
    assert overlapping_boxes[1] == {(7, 3)}, "valid bounding-box overlap rejected"
    pieces[1]["cells"] = ((3, 2),)
    try:
        _cast(canvas, pieces, [tray])
    except ValueError:
        pass
    else:
        raise AssertionError("real cast collision accepted")
    # Majority ties and inconsistent non-motif output cells are rejected.
    try:
        _repair_motif(np.array([[1, 2, 1, 2, 2, 1]]), 1, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("undersupported periodic evidence accepted")
    ambiguous = np.array([[1] * 12, [2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1]], dtype=np.int8)
    try:
        _repair_motif(ambiguous, 1, 2)
    except ValueError as e:
        assert str(e) == "ambiguous periodic repair"
    else:
        raise AssertionError("competing repairs accepted")
    broken = generate_puzzle(102, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("bad label accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({tuple(r["shape"]) for r in records}),
            "error_masks": len(masks), "ribbon_counts": [1, 2, 3],
            "periods": sorted({p for r in records for p in r["periods"]}),
            "directions": sorted({r["direction"] for r in records})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
