"""Negative-space registration at a compass rail, followed by cavity projection.

Only Python's standard library and NumPy are used. The solver enumerates visible
D4 poses and exact covers of target pixels; it never uses construction metadata.
"""
from __future__ import annotations

import numpy as np


STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in STEPS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    todo.append(p)
        result.append(cells)
    return result


def _transform(cells, t):
    moved = []
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        moved.append((r, c, int(v)))
    r0, c0 = min(r for r, c, v in moved), min(c for r, c, v in moved)
    return {(r - r0, c - c0): v for r, c, v in moved}


def _shift(cells, r0, c0):
    return {(r + r0, c + c0): v for (r, c), v in cells.items()}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _profile(cells):
    """All colored cells are walls, irrespective of their color."""
    h = 1 + max(r for r, c in cells)
    w = 1 + max(c for r, c in cells)
    mask = np.zeros((h, w), dtype=bool)
    for p in cells:
        mask[p] = True
    holes = [x for x in _components(~mask)
             if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in x)]
    active = [x for x in holes if any(c == w - 2 for r, c in x)]
    key = set().union(*active) if active else set()
    all_holes = set().union(*holes) if holes else set()
    naive = {r for r in range(h) if w >= 3 and not mask[r, w - 2]}
    return {"mask": mask, "holes": holes, "active": active, "key": key,
            "all_holes": all_holes, "rows": {r for r, c in key},
            "naive_rows": naive}


def _labels_match(cells, sides):
    h, w = 1 + max(r for r, c in cells), 1 + max(c for r, c in cells)
    identities = set(sides)
    directions = []
    for (r, c), value in cells.items():
        if value not in identities:
            continue
        on = [i for i, yes in enumerate((r == 0, c == w - 1, r == h - 1, c == 0)) if yes]
        if len(on) != 1 or sides[on[0]] != value:
            return False
        directions.append(on[0])
    return len(set(directions)) >= 2


def _parse(grid):
    """Canonical view has the screen to the right of the unique full rail."""
    possibilities = []
    for turn in range(4):
        a = np.rot90(grid, turn).copy()
        h, w = a.shape
        if h < 5:
            continue
        for rail in range(4, w - 1):
            right, bg = int(a[0, rail]), int(a[0, -1])
            if right == bg or not np.all(a[:, rail] == right):
                continue
            if not np.all(a[:, rail + 1:] == bg):
                continue
            top, bottom, left = int(a[0, 0]), int(a[-1, 0]), int(a[1, 0])
            sides = (top, right, bottom, left)
            if len(set((bg,) + sides)) != 5:
                continue
            if not (np.all(a[0, :rail] == top) and np.all(a[-1, :rail] == bottom)
                    and np.all(a[1:-1, 0] == left)):
                continue
            possibilities.append((a, turn, bg, rail, sides))
    _require(len(possibilities) == 1, "compass rail/screen not unique")
    a, turn, bg, rail, sides = possibilities[0]
    interior = np.zeros(a.shape, dtype=bool)
    interior[1:-1, 1:rail] = True
    sources, target_groups, reg_colors, body_colors = [], [], set(), set()
    for group in _components((a != bg) & interior):
        cells = {p: int(a[p]) for p in group}
        values = set(cells.values())
        if len(values) == 1:
            target_groups.append(group)
            reg_colors.update(values)
        else:
            sources.append(_transform(cells, 0))
            body_colors.update(values - set(sides))
    _require(sources and target_groups and len(reg_colors) == 1, "missing pieces/marks")
    reg = next(iter(reg_colors))
    _require(len(body_colors) == 1, "body role is ambiguous")
    body = next(iter(body_colors))
    _require(len(set((bg, body, reg) + sides)) == 7, "roles must be distinct")
    for source in sources:
        _require(set(source.values()) - set(sides) == {body}, "marks cannot be embedded walls")
        label_colors = set(source.values()) & set(sides)
        _require(len(label_colors) >= 2, "piece lacks compass evidence")
    targets = set().union(*target_groups)
    return {"grid": a, "turn": turn, "bg": bg, "rail": rail, "sides": sides,
            "body": body, "reg": reg, "sources": sources, "targets": targets}


def _candidates(source, info, use_labels=True):
    h = info["grid"].shape[0]
    result, seen = [], set()
    for t in range(8):
        cells = _transform(source, t)
        if use_labels and not _labels_match(cells, info["sides"]):
            continue
        profile = _profile(cells)
        if not profile["key"]:
            continue
        ph, pw = profile["mask"].shape
        col = info["rail"] - pw
        if col < 1:
            continue
        for row in range(1, h - ph):
            key = {(r + row, c + col) for r, c in profile["key"]}
            if not key <= info["targets"]:
                continue
            placed = _shift(cells, row, col)
            if placed.keys() & info["targets"]:
                continue
            sig = tuple(sorted(placed.items()))
            if sig in seen:
                continue
            seen.add(sig)
            result.append({"cells": placed, "key": key, "profile": profile,
                           "row": row, "col": col, "transform": t})
    return result


def _solutions(info, use_labels=True, limit=2):
    choices = [_candidates(s, info, use_labels) for s in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    results = {}

    def visit(depth, covered, occupied, placements):
        if len(results) >= limit:
            return
        if depth == len(order):
            if covered == info["targets"]:
                sig = tuple(sorted(occupied.items()))
                results[sig] = list(placements)
            return
        for candidate in choices[order[depth]]:
            if covered & candidate["key"] or occupied.keys() & candidate["cells"].keys():
                continue
            visit(depth + 1, covered | candidate["key"],
                  {**occupied, **candidate["cells"]}, placements + [candidate])

    visit(0, set(), {}, [])
    return list(results.values())


def _render(info, placements):
    out = info["grid"].copy()
    out[1:-1, 1:info["rail"]] = info["bg"]
    rows = set()
    for p in placements:
        for cell, color in p["cells"].items():
            out[cell] = color
        rows.update(r + p["row"] for r in p["profile"]["rows"])
    for row in rows:
        out[row, info["rail"] + 1:] = info["reg"]
    return np.rot90(out, -info["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique output using only the input's walls, compass and marks."""
    info = _parse(_grid(input_grid))
    answers = _solutions(info)
    _require(len(answers) == 1, "negative registration has zero or multiple outputs")
    return _render(info, answers[0])


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _piece(rng, palette, kind, cap, symmetric_key=False):
    bg, body, reg, top, right, bottom, left = palette
    chambers = ["active"]
    if kind == "open":
        chambers += ["open"]
    elif kind == "decoy":
        chambers += ["decoy"]
    elif kind == "combined":
        chambers += ["open", "decoy"]
    elif kind == "multi":
        chambers += ["active"] * int(rng.integers(1, 3))
    elif kind == "random":
        chambers += [str(rng.choice(["active", "open", "decoy"]))
                     for _ in range(int(rng.integers(0, 3)))]
    while 2 * len(chambers) + 1 > cap:
        chambers.pop()
    rng.shuffle(chambers)
    lengths = [1] * len(chambers)
    if symmetric_key:
        lengths[chambers.index("active")] = 2
    spare = cap - (sum(lengths) + len(chambers) + 1)
    for _ in range(int(rng.integers(max(0, spare) + 1))):
        eligible = [i for i, x in enumerate(lengths) if x < 3]
        if eligible:
            lengths[int(rng.choice(eligible))] += 1
    h = sum(lengths) + len(chambers) + 1
    w = int(rng.integers(4, 10))
    mask = np.ones((h, w), dtype=bool)
    row = 1
    for kind_here, length in zip(chambers, lengths):
        if kind_here == "open":
            mask[row:row + length, :w - 1] = False
        else:
            end = w - 2 if kind_here == "active" else int(rng.integers(1, w - 2))
            start = int(rng.integers(1, end + 1))
            for rr in range(row, row + length):
                edge = start
                if not symmetric_key and length > 1 and rng.random() < .5:
                    edge = int(rng.integers(1, end + 1))
                mask[rr, edge:end + 1] = False
        row += length + 1
    # Labels are recolored wall cells on noncorner bounding edges. Their exact
    # positions do not encode direction; equality to boundary identities does.
    selected = [(int(rng.integers(4)))]
    selected += [(selected[0] + 1) % 4]
    if rng.random() < .3:
        selected += [d for d in range(4) if d not in selected and rng.random() < .55]
    sides = (top, right, bottom, left)
    labels = {}
    for d in selected:
        points = ([(0, c) for c in range(1, w - 1)] if d == 0 else
                  [(r, w - 1) for r in range(1, h - 1)] if d == 1 else
                  [(h - 1, c) for c in range(1, w - 1)] if d == 2 else
                  [(r, 0) for r in range(1, h - 1)])
        points = [p for p in points if mask[p]]
        rng.shuffle(points)
        for p in points[:int(rng.integers(1, min(3, len(points)) + 1))]:
            labels[p] = sides[d]
    # Erode a few outline cells without changing any closed cavity or key.
    old_holes = _profile({p: body for p in zip(*np.nonzero(mask))})["all_holes"]
    for _ in range(int(rng.integers(1, 9))):
        occupied = sorted(p for p in zip(*np.nonzero(mask)) if p not in labels)
        p = occupied[int(rng.integers(len(occupied)))]
        if not (p[0] in (0, h - 1) or p[1] in (0, w - 1)):
            continue
        trial = mask.copy()
        trial[p] = False
        if not (trial[0].any() and trial[-1].any() and trial[:, 0].any() and trial[:, -1].any()):
            continue
        if len(_components(trial)) != 1:
            continue
        cells = {p: body for p in zip(*np.nonzero(trial))}
        if _profile(cells)["all_holes"] == old_holes:
            mask = trial
    cells = {p: body for p in zip(*np.nonzero(mask))}
    cells.update(labels)
    _require(_labels_match(cells, sides), "constructor label error")
    return cells


def _construct(rng, palette, kinds, scene_turn, teaching=None):
    count = len(kinds)
    cap = min(11, (27 - (count - 1)) // count)
    pieces = [_piece(rng, palette, kind, cap, teaching == "rotation" and i == 0)
              for i, kind in enumerate(kinds)]
    rng.shuffle(pieces)
    heights = [1 + max(r for r, c in p) for p in pieces]
    extent = max(max(1 + max(r for r, c in p), 1 + max(c for r, c in p)) for p in pieces)
    minimum_h = max(sum(heights) + count - 1 + 2, extent + 4)
    if minimum_h > 30:
        return None
    h = int(rng.integers(minimum_h, min(30, minimum_h + 5) + 1))
    work = int(rng.integers(max(13, extent + 5), 24))
    rail = work + 1
    screen = int(rng.integers(1, min(8, 29 - rail) + 1))
    bg, body, reg, top, right, bottom, left = palette
    base = np.full((h, rail + 1 + screen), bg, dtype=np.int8)
    base[:, rail] = right
    base[0, :rail], base[-1, :rail] = top, bottom
    base[1:-1, 0] = left
    extras = rng.multinomial(h - 2 - sum(heights) - count + 1,
                             np.full(count + 1, 1 / (count + 1)))
    finals, targets, row = [], set(), 1 + int(extras[0])
    for i, piece in enumerate(pieces):
        width = 1 + max(c for r, c in piece)
        final = _shift(piece, row, rail - width)
        profile = _profile(piece)
        targets.update((r + row, c + rail - width) for r, c in profile["key"])
        finals.append(final)
        row += heights[i] + 1 + int(extras[i + 1])
    inp = base.copy()
    for p in targets:
        inp[p] = reg
    blocked = _halo(targets)
    for i, piece in enumerate(pieces):
        transforms = list(map(int, rng.permutation(8)))
        if i == 0 and teaching == "rotation":
            transforms = [t for t in transforms if 0 < t < 4]
        elif i == 0 and teaching == "reflection":
            transforms = [t for t in transforms if t >= 4]
        placed = None
        for t in transforms:
            transformed = _transform(piece, t)
            ph = 1 + max(r for r, c in transformed)
            pw = 1 + max(c for r, c in transformed)
            origins = [(r, c) for r in range(1, h - ph)
                       for c in range(1, rail - pw + 1)]
            rng.shuffle(origins)
            for r0, c0 in origins:
                trial = _shift(transformed, r0, c0)
                if not (trial.keys() & blocked):
                    placed = trial
                    break
            if placed is not None:
                break
        if placed is None:
            return None
        for p, v in placed.items():
            inp[p] = v
        blocked.update(_halo(placed))
    out = base.copy()
    ray_rows = set()
    for final in finals:
        for p, v in final.items():
            out[p] = v
    ray_rows.update(r for r, c in targets)
    for r in ray_rows:
        out[r, rail + 1:] = reg
    inp, out = np.rot90(inp, scene_turn).copy(), np.rot90(out, scene_turn).copy()
    try:
        info = _parse(inp)
        answers = _solutions(info)
        if len(answers) != 1 or not np.array_equal(_render(info, answers[0]), out):
            return None
        if teaching == "rotation" and len(_solutions(info, False)) < 2:
            return None
    except AssertionError:
        return None
    return {"input": inp, "output": out}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(["count", "orientation", "combined"]))
    base_turn = int(rng.integers(4))
    shared_palette = list(map(int, rng.permutation(10)[:7]))
    fixed_palette = bool(rng.random() < .4)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            teaching = ("rotation" if index == 0 else "reflection" if index == 1 else None) if split == "train" else None
            palette = shared_palette if fixed_palette else list(map(int, rng.permutation(10)[:7]))
            if regime == "orientation":
                turn = base_turn if split == "train" else (base_turn + int(rng.choice([1, 3]))) % 4
            else:
                turn = int(rng.integers(4))
            if regime == "count":
                count = int(rng.integers(1, 3)) if split == "train" else 3
            elif teaching == "rotation":
                count = 1
            else:
                count = int(rng.integers(1, 4))
            if teaching == "rotation":
                kinds = ["open"] + ["active"] * (count - 1)
            elif teaching == "reflection":
                kinds = ["decoy"] + ["active"] * (count - 1)
            elif split == "test" and regime == "combined":
                count = int(rng.integers(1, 3))
                kinds = ["combined"] + ["multi"] * (count - 1)
            else:
                pool = ["active", "open", "decoy", "multi"] if regime == "combined" else ["active", "open", "decoy", "multi", "random"]
                kinds = [str(rng.choice(pool)) for _ in range(count)]
            for _ in range(150):
                pair = _construct(rng, palette, kinds, turn, teaching)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "canvas must be preserved")
    info = _parse(inp)
    answers = _solutions(info)
    _require(len(answers) == 1, "ambiguous or impossible key assignment")
    placements = answers[0]
    _require(np.array_equal(out, _render(info, placements)), "wrong output")
    canonical_out = np.rot90(out, info["turn"])
    _require(all(canonical_out[p] == info["bg"] for p in info["targets"]), "marks must become empty holes")
    rows, open_contrast, decoy, multirow, hole_shapes = set(), False, False, False, set()
    for p in placements:
        profile = p["profile"]
        _require(max(c for r, c in p["cells"]) == info["rail"] - 1, "not rail aligned")
        _require(_labels_match(_transform(p["cells"], 0), info["sides"]), "bad compass pose")
        _require(len(_components(profile["mask"])) == 1, "disconnected wall")
        for hole in profile["active"]:
            _require({r for r, c in hole} == {r for r, c in hole if c == profile["mask"].shape[1] - 2},
                     "partial leading contact outside sampled domain")
        for r, c in profile["all_holes"]:
            _require(canonical_out[r + p["row"], c + p["col"]] == info["bg"], "cavity filled")
        rows.update(r + p["row"] for r in profile["rows"])
        open_contrast |= bool(profile["naive_rows"] - profile["rows"])
        decoy |= bool(profile["all_holes"] - profile["key"])
        multirow |= any(len({r for r, c in hole}) >= 2 for hole in profile["active"])
        hole_shapes.update(tuple(sorted(hole)) for hole in profile["holes"])
    for r in range(canonical_out.shape[0]):
        expected = info["reg"] if r in rows else info["bg"]
        _require(np.all(canonical_out[r, info["rail"] + 1:] == expected), "incorrect signal rows")
    return {"info": info, "placements": placements, "open": open_contrast, "decoy": decoy,
            "multirow": multirow, "hole_shapes": hole_shapes,
            "compass_needed": len(_solutions(info, False)) > 1,
            "transforms": {p["transform"] for p in placements}}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "bad counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            record = _inspect_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                training.append(record)
    _require(any(x["open"] for x in training), "no enclosure contrast")
    _require(any(x["decoy"] for x in training), "no extreme-edge contrast")
    _require(any(x["multirow"] for x in training), "no full cavity-row projection")
    _require(any(x["compass_needed"] for x in training), "compass can be ignored")
    _require(any(x["transforms"] & {1, 2, 3} for x in training), "no demonstrated rotation")
    _require(any(x["transforms"] & {4, 5, 6, 7} for x in training), "no demonstrated reflection")


def self_test():
    """Bounded 42-seed audit, equivariance, count limits, and corruption checks."""
    signatures, dimensions, piece_shapes, hole_shapes, normalized, counts, turns = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(np.asarray(p[k])) for s in ("train", "test")
                             for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                rec = _inspect_pair(pair)
                info = rec["info"]
                dimensions.add(pair["input"].shape)
                counts.add(len(info["sources"]))
                turns.add(info["turn"])
                hole_shapes.update(rec["hole_shapes"])
                for p in rec["placements"]:
                    piece_shapes.add(p["profile"]["mask"].shape)
                lookup = {info["bg"]: 0, info["body"]: 1, info["reg"]: 2}
                lookup.update({c: i + 3 for i, c in enumerate(info["sides"])})
                normalized.add((pair["input"].shape, bytes(lookup[int(v)] for v in pair["input"].flat)))
        # A complete scene rotation and an arbitrary bijective palette change
        # commute with the input-only rule, including its output signal color.
        pair = puzzle["test"][0]
        turn = seed % 4
        _require(np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn)), "rotation equivariance")
        lut = np.random.default_rng(seed + 10000).permutation(10)
        _require(np.array_equal(solve(lut[pair["input"]]), lut[pair["output"]]), "palette equivariance")
    _require(len(signatures) == 42, "seed diversity collapsed")
    _require(len(dimensions) >= 35 and len(piece_shapes) >= 20, "geometry diversity collapsed")
    _require(len(hole_shapes) >= 60 and len(normalized) >= 180, "structural diversity collapsed")
    _require(counts == {1, 2, 3} and turns == {0, 1, 2, 3}, "count/orientation diversity collapsed")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(12345 + a * 10 + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    p = generate_puzzle(77, 2, 1)
    q = generate_puzzle(77, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ("input", "output")), "nondeterministic seed")
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted answer")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts)")
