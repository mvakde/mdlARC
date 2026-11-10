"""Contour-probe printing: layer codes control rigid stencil docking."""
from collections import deque
import numpy as np


_CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _CARDINAL:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    comp.add(p)
                    stack.append(p)
        result.append(comp)
    return result


def _depth(plate, diagonal=False):
    """Boundary cells have depth zero; diagonal=True is an ablation only."""
    neighbors = _EIGHT if diagonal else _CARDINAL
    depths = {}
    queue = deque()
    for r, c in sorted(plate):
        if any((r + dr, c + dc) not in plate for dr, dc in neighbors):
            depths[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            p = r + dr, c + dc
            if p in plate and p not in depths:
                depths[p] = depths[r, c] + 1
                queue.append(p)
    return depths


def _transform(cells, transform):
    raw = {}
    for (r, c), v in cells.items():
        if transform >= 4:
            c = -c
        for _ in range(transform % 4):
            r, c = c, -r
        raw[r, c] = v
    r0 = min(r for r, _ in raw)
    c0 = min(c for _, c in raw)
    return {(r - r0, c - c0): v for (r, c), v in raw.items()}


def _translate(cells, origin):
    return {(r + origin[0], c + origin[1]): v for (r, c), v in cells.items()}


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _arc_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _parse(input_grid):
    grid = _arc_grid(input_grid)
    code = []
    for row in grid:
        if 1 <= int(row[0]) <= 6 and np.all(row == row[0]):
            code.append(int(row[0]))
        else:
            break
    h = len(code)
    if not 3 <= h <= 7 or h + 1 >= grid.shape[0] or not np.all(grid[h] == 0):
        raise ValueError("missing top-to-bottom layer code and blank separator")
    work = grid[h + 1:]
    candidates = _components(zip(*np.nonzero((work == 8) | (work == 9))))
    boards = [p for p in candidates if any(work[q] == 8 for q in p)]
    if len(boards) != 1:
        raise ValueError("expected one connected blank plate")
    plate = boards[0]
    if any(work[p] not in (8, 9) for p in plate):
        raise ValueError("plate contains nonblank cells")
    depths = _depth(plate)
    if max(depths.values()) != h - 1:
        raise ValueError("code must contain exactly one row per erosion depth")
    field = {p: code[d] for p, d in depths.items()}
    targets = _components(p for p in plate if work[p] == 9)
    active = set(zip(*np.nonzero(work))) - plate
    sources = [{p: int(work[p]) for p in comp} for comp in _components(active)]
    if not sources or len(sources) != len(targets):
        raise ValueError("need one composite stamp per registration target")
    if not 1 <= len(sources) <= 4 or len(set(code)) < 2:
        raise ValueError("need 1..4 stamps and at least two layer colors")
    for source in sources:
        values = set(source.values())
        if not {7, 9} <= values or not values <= set(code) | {7, 9}:
            raise ValueError("invalid stamp roles")
        if len(values & set(code)) < 2:
            raise ValueError("stamp needs probes of at least two colors")
        if len(_components(p for p, v in source.items() if v == 9)) != 1:
            raise ValueError("registration marks must be connected")
        marks = [p for p, v in source.items() if v == 9]
        probes = [p for p, v in source.items() if v <= 6]
        if not 1 <= len(marks) <= 4 or not 3 <= len(probes) <= 6:
            raise ValueError("invalid registration or probe count")
        if sum(v == 7 for v in source.values()) < 3:
            raise ValueError("a stamp needs at least three neutral printing cells")
        if any(all((p[0] + dr, p[1] + dc) in source for dr, dc in _CARDINAL)
               for p in probes):
            raise ValueError("probes must lie on the stamp perimeter")
    return {"grid": grid, "work": work, "code": code, "offset": h + 1,
            "plate": plate, "depths": depths, "field": field,
            "targets": targets, "sources": sources}


def _candidates(source, info, use_probes=True):
    result, seen = [], set()
    for t in range(8):
        raw = _transform(source, t)
        marks = {p for p, v in raw.items() if v == 9}
        first = min(marks)
        for index, target in enumerate(info["targets"]):
            if len(marks) != len(target):
                continue
            start = min(target)
            delta = start[0] - first[0], start[1] - first[1]
            placed = _translate(raw, delta)
            if {p for p, v in placed.items() if v == 9} != target:
                continue
            if not set(placed) <= info["plate"]:
                continue
            if use_probes and any(v <= 6 and info["field"][p] != v
                                  for p, v in placed.items()):
                continue
            printed = {p: 9 if v == 9 else info["field"][p] for p, v in placed.items()}
            key = index, tuple(sorted(placed.items()))
            if key not in seen:
                seen.add(key)
                result.append({"target": index, "raw": placed, "printed": printed})
    return result


def _solutions(info, use_probes=True, limit=2):
    choices = [_candidates(s, info, use_probes) for s in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    answers = {}

    def search(i, used, pixels, certificates):
        if len(answers) >= limit:
            return
        if i == len(order):
            key = tuple(sorted(pixels.items()))
            answers.setdefault(key, (dict(pixels), list(certificates)))
            return
        for option in choices[order[i]]:
            if option["target"] in used or pixels.keys() & option["printed"].keys():
                continue
            search(i + 1, used | {option["target"]},
                   {**pixels, **option["printed"]}, certificates + [(order[i], option)])
    search(0, set(), {}, [])
    return list(answers.values())


def _render(info, pixels):
    out = info["grid"].copy()
    work = out[info["offset"]:]
    work[:] = 0
    for p in info["plate"]:
        work[p] = 8
    for p, v in pixels.items():
        work[p] = v
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the plate, contour field, assignment and D4 poses from input only."""
    info = _parse(input_grid)
    answers = _solutions(info)
    if len(answers) != 1:
        raise ValueError("stencils and probes do not determine one printed output")
    return _render(info, answers[0][0])


def _choice(rng, values):
    values = list(values)
    return values[int(rng.integers(len(values)))]


def _sample_plate(rng, layers, kind, count):
    minimum = 2 * layers - 1
    wh = int(rng.integers(max(17, minimum + 2), 30 - layers))
    ww = int(rng.integers(26 if count >= 3 else 22, 31))
    for _ in range(60):
        short = minimum + int(rng.integers(2))
        long = int(rng.integers(short, min(short + 5, wh - 1, ww - 7)))
        ph, pw = (short, long) if rng.random() < .5 else (long, short)
        if ph > wh - 2:
            continue
        cells = {(r, c) for r in range(ph) for c in range(pw)}
        if kind == "cut":
            for corner in rng.choice(4, size=int(rng.integers(1, 4)), replace=False):
                size = int(rng.integers(1, layers + 1))
                cells -= {(r, c) for r in range(ph) for c in range(pw)
                          if (r if corner < 2 else ph - 1 - r)
                          + (c if corner % 2 == 0 else pw - 1 - c) < size}
        elif kind == "notch":
            side = int(rng.integers(4))
            length = ph if side % 2 else pw
            start = int(rng.integers(2, length - 2))
            span = int(rng.integers(1, min(4, length - start - 1)))
            reach = int(rng.integers(1, layers))
            cells -= {(r, c) for r in range(ph) for c in range(pw)
                      if ((side == 0 and r < reach and start <= c < start + span)
                          or (side == 1 and c >= pw - reach and start <= r < start + span)
                          or (side == 2 and r >= ph - reach and start <= c < start + span)
                          or (side == 3 and c < reach and start <= r < start + span))}
        if len(_components(cells)) != 1 or max(_depth(cells).values()) != layers - 1:
            continue
        row = int(rng.integers(1, wh - ph))
        columns = [c for c in range(1, ww - pw)
                   if max(c, ww - pw - c) >= 6]
        if not columns:
            continue
        col = _choice(rng, columns)
        return wh, ww, {(r + row, c + col) for r, c in cells}
    return None


def _shape(rng, count, mark_size):
    h = int(rng.integers(3, 6 if count >= 3 else 8))
    w = int(rng.integers(3, 6 if count >= 3 else 8))
    if h * w < mark_size + 7:
        w = 4
    points = {(int(rng.integers(h)), int(rng.integers(w)))}
    size = int(rng.integers(max(9, mark_size + 7), h * w + 1))
    for _ in range(size - 1):
        front = sorted({(r + dr, c + dc) for r, c in points for dr, dc in _CARDINAL
                        if 0 <= r + dr < h and 0 <= c + dc < w} - points)
        if not front:
            break
        points.add(_choice(rng, front))
    return set(_transform({p: 7 for p in points}, 0))


def _marks(rng, points, count):
    result = {_choice(rng, sorted(points))}
    while len(result) < count:
        front = sorted({(r + dr, c + dc) for r, c in result for dr, dc in _CARDINAL}
                       & points - result)
        if not front:
            return None
        result.add(_choice(rng, front))
    return result


def _sample_code(rng, layers, teaching, deep):
    colors = [int(c) for c in rng.permutation(np.arange(1, 7))]
    if teaching == "rotate":
        return [colors[0], colors[0], colors[1], colors[2]]
    if teaching == "reflect":
        return colors[:layers]
    k = int(rng.integers(2, min(layers, 6) + 1))
    if deep:
        # The held-out deepest layer has a new visible color; its use in a
        # neutral printing cell is checked below, rather than merely requested.
        k = max(3, k)
        prefix = [colors[i % (k - 1)] for i in range(layers - 1)]
        rng.shuffle(prefix)
        return prefix + [colors[k - 1]]
    if rng.random() < .25:
        # Straightforward thin layers remain a substantial part of the family.
        return [colors[i % min(layers, 6)] for i in range(layers)]
    code = [int(c) for c in rng.choice(colors[:k], size=layers)]
    positions = rng.choice(layers, k, replace=False)
    for p, c in zip(positions, colors[:k]):
        code[int(p)] = c
    return code


def _construct(rng, layers, count, kind, mark_min, mark_max, teaching, deep=False):
    scene = _sample_plate(rng, layers, kind, count)
    if scene is None:
        return None
    wh, ww, plate = scene
    code = _sample_code(rng, layers, teaching, deep)
    depths = _depth(plate)
    field = {p: code[d] for p, d in depths.items()}
    diagonal_field = {p: code[d] for p, d in _depth(plate, diagonal=True).items()}
    r0, r1 = min(r for r, _ in plate), max(r for r, _ in plate)
    c0, c1 = min(c for _, c in plate), max(c for _, c in plate)
    rectangle_field = {p: code[min(p[0] - r0, r1 - p[0], p[1] - c0, c1 - p[1])]
                       for p in plate}
    collapsed = [c for i, c in enumerate(code) if i == 0 or code[i - 1] != c]
    repeated_marks = int(rng.integers(1, min(mark_max, 2) + 1)) if teaching == "reflect" else None
    finals, targets, blocked = [], [], set()
    for piece_index in range(count):
        found = False
        for _ in range(90):
            mark_size = (1 if teaching == "rotate" and piece_index == 0 else repeated_marks
                         if repeated_marks is not None else int(rng.integers(mark_min, mark_max + 1)))
            points = _shape(rng, count, mark_size)
            marks = _marks(rng, points, mark_size)
            if marks is None:
                continue
            edge = {p for p in points if any((p[0] + dr, p[1] + dc) not in points for dr, dc in _CARDINAL)} - marks
            probe_count = min(len(edge), int(rng.integers(3, 7)), len(points) - len(marks) - 3)
            if probe_count < 3:
                continue
            positions = sorted(edge)
            probes = {positions[int(j)] for j in rng.choice(len(positions), probe_count, replace=False)}
            if len(probes) < 3:
                continue
            origins = [(r, c) for r in range(wh) for c in range(ww)
                       if all((r + pr, c + pc) in plate and (r + pr, c + pc) not in blocked
                              for pr, pc in points)]
            rng.shuffle(origins)
            for origin in origins[:45]:
                raw = _translate({p: 9 if p in marks else 7 for p in points}, origin)
                for pr, pc in probes:
                    q = pr + origin[0], pc + origin[1]
                    raw[q] = field[q]
                if len({v for v in raw.values() if v <= 6}) < 2:
                    continue
                if len({field[p] for p, v in raw.items() if v == 7}) < 2:
                    continue
                if deep and piece_index == 0 and not any(
                    v == 7 and depths[p] == layers - 1 for p, v in raw.items()):
                    continue
                if teaching == "reflect" and piece_index == 0 and not any(
                    v == 7 and field[p] != rectangle_field[p] for p, v in raw.items()):
                    continue
                if teaching == "reflect" and piece_index == 0 and not any(
                    v == 7 and field[p] != diagonal_field[p] for p, v in raw.items()):
                    continue
                if teaching == "rotate" and piece_index == 0 and not any(
                    v == 7 and field[p] != collapsed[min(depths[p], len(collapsed) - 1)]
                    for p, v in raw.items()):
                    continue
                target = {p for p, v in raw.items() if v == 9}
                info = {"targets": [target], "plate": plate, "field": field}
                options = _candidates(raw, info)
                if len(options) != 1:
                    continue
                if teaching == "rotate" and piece_index == 0 and len(_candidates(raw, info, False)) < 2:
                    continue
                if teaching == "reflect" and piece_index == 0 and any(
                    _signature(_transform(raw, t)) == _signature(raw) for t in (4, 5, 6, 7)):
                    continue
                finals.append(raw)
                targets.append(target)
                blocked |= _halo(raw)
                found = True
                break
            if found:
                break
        if not found:
            return None
    blocked = _halo(plate)
    sources = []
    for i, final in enumerate(finals):
        transforms = list(int(t) for t in rng.permutation(8))
        if teaching and i == 0:
            transforms = [t for t in transforms if (1 <= t <= 3 if teaching == "rotate" else t >= 4)]
        found = False
        for t in transforms:
            raw = _transform(final, t)
            ph = 1 + max(r for r, _ in raw)
            pw = 1 + max(c for _, c in raw)
            origins = [(r, c) for r in range(wh - ph + 1) for c in range(ww - pw + 1)
                       if all((r + pr, c + pc) not in blocked for pr, pc in raw)]
            if origins:
                source = _translate(raw, _choice(rng, origins))
                sources.append(source)
                blocked |= _halo(source)
                found = True
                break
        if not found:
            return None
    inp = np.zeros((layers + 1 + wh, ww), dtype=np.int8)
    inp[:layers] = np.asarray(code, dtype=np.int8)[:, None]
    work = inp[layers + 1:]
    for p in plate:
        work[p] = 8
    for target in targets:
        for p in target:
            work[p] = 9
    for source in sources:
        for p, v in source.items():
            work[p] = v
    expected_pixels = {p: 9 if v == 9 else field[p] for final in finals for p, v in final.items()}
    info = _parse(inp)
    answers = _solutions(info)
    if len(answers) != 1 or answers[0][0] != expected_pixels:
        return None
    if teaching == "rotate" and len(_solutions(info, False)) != 2:
        return None
    if teaching and not any(_motion(info["sources"][i], option["raw"]) == teaching
                            for i, option in answers[0][1]):
        return None
    # Check the certificate actually retained by pixel-map deduplication. A
    # different equivalent assignment must not erase a teaching witness.
    certified_neutral = [p for _, option in answers[0][1]
                         for p, v in option["raw"].items() if v == 7]
    if teaching == "rotate" and not any(field[p] != collapsed[min(depths[p], len(collapsed) - 1)]
                                        for p in certified_neutral):
        return None
    if teaching == "reflect" and not any(field[p] != rectangle_field[p] for p in certified_neutral):
        return None
    if teaching == "reflect" and not any(field[p] != diagonal_field[p] for p in certified_neutral):
        return None
    return {"input": inp, "output": _render(info, expected_pixels)}


def _counts(num_train, num_test):
    for v, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                or not low <= int(v) <= high):
            raise ValueError(f"{name} is not a legal count")
    a, b = int(num_train), int(num_test)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    a, b = _counts(num_train, num_test)
    regime = _choice(rng, ("depth", "count", "stencil"))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", a), ("test", b)):
        for i in range(number):
            teaching = ("rotate" if i == 0 else "reflect" if i == 1 else None) if split == "train" else None
            if split == "train":
                layers = 4 if i == 0 else 5 if i == 1 else int(rng.integers(3, 7))
                count = 1 if i == 0 or layers == 3 else 2 if i == 1 else int(rng.integers(1, 3))
                kind = "rect" if i == 0 else "notch" if i == 1 else _choice(rng, ("rect", "cut", "notch"))
                mark_max = 2 if regime == "stencil" else 4
                mark_min = 1
            else:
                layers = 7 if regime == "depth" else int(rng.integers(6, 8)) if regime == "count" else int(rng.integers(4, 7))
                count = int(rng.integers(3, 5)) if regime == "count" else int(rng.integers(1, 3))
                kind = _choice(rng, ("rect", "cut", "notch"))
                mark_max = 4
                mark_min = 3 if regime == "stencil" else 1
            for attempt in range(120):
                pair = _construct(rng, layers, count, kind, mark_min, mark_max, teaching,
                                  deep=split == "test" and regime == "depth")
                if pair is not None and _grid_key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError(f"could not construct {split} {i}: {layers}/{count}/{kind}")
            seen.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _motion(source, final):
    if _signature(source) == _signature(final):
        return "translate"
    if any(_signature(_transform(source, t)) == _signature(final) for t in range(1, 4)):
        return "rotate"
    return "reflect"


def _unanchored_prints(source, info, limit=2):
    """Ablation: remove registration targets, retaining probe/depth checks."""
    found = set()
    wh, ww = info["work"].shape
    for t in range(8):
        raw = _transform(source, t)
        h = 1 + max(r for r, _ in raw)
        w = 1 + max(c for _, c in raw)
        for r in range(wh - h + 1):
            for c in range(ww - w + 1):
                placed = _translate(raw, (r, c))
                if not placed.keys() <= info["plate"]:
                    continue
                if any(v <= 6 and info["field"][p] != v for p, v in placed.items()):
                    continue
                printed = tuple(sorted((p, 9 if v == 9 else info["field"][p])
                                       for p, v in placed.items()))
                found.add(printed)
                if len(found) >= limit:
                    return found
    return found


def _features(pair):
    inp, out = _arc_grid(pair["input"]), _arc_grid(pair["output"])
    info = _parse(inp)
    answers = _solutions(info)
    assert len(answers) == 1, "input is ambiguous or unsatisfiable"
    pixels, certificates = answers[0]
    assert np.array_equal(out, _render(info, pixels)), "wrong contour print"
    assert not np.array_equal(inp, out)
    work_out = out[info["offset"]:]
    assert np.array_equal(inp[:info["offset"]], out[:info["offset"]]), "legend changed"
    assert not np.any(work_out == 7), "unprinted neutral cells remain"
    assert np.all(work_out[[p[0] for p in info["plate"]], [p[1] for p in info["plate"]]] != 0)
    outside = np.ones(info["work"].shape, dtype=bool)
    for p in info["plate"]:
        outside[p] = False
    assert not np.any(work_out[outside]), "old stamps were not completely erased"
    assert {p for p, v in pixels.items() if v == 9} == set().union(*info["targets"])
    assert len(_components(pixels)) == len(info["sources"]), "printed masks touch or split"
    r0, r1 = min(r for r, _ in info["plate"]), max(r for r, _ in info["plate"])
    c0, c1 = min(c for _, c in info["plate"]), max(c for _, c in info["plate"])
    collapsed = [c for i, c in enumerate(info["code"]) if i == 0 or c != info["code"][i - 1]]
    diagonal_field = {p: info["code"][d] for p, d in _depth(info["plate"], diagonal=True).items()}
    repeat_sensitive = contour_sensitive = diagonal_sensitive = False
    masks, patterns, moves, neutral_depths, probe_counts, ink_colors = [], [], [], [], [], []
    for source_index, option in certificates:
        source = info["sources"][source_index]
        raw = option["raw"]
        assert len(source) == len(raw)
        assert _signature(raw) in {_signature(_transform(source, t)) for t in range(8)}
        assert len(set(option["printed"])) == len(source), "mask geometry was changed"
        moves.append(_motion(source, raw))
        norm = _transform(source, 0)
        masks.append(tuple(sorted(norm)))
        marks = {p: 9 for p, v in norm.items() if v == 9}
        patterns.append(min(tuple(sorted(_transform(marks, t))) for t in range(8)))
        probes = [p for p, v in raw.items() if v <= 6]
        assert all(v == 9 or option["printed"][p] == info["field"][p] for p, v in raw.items())
        assert all(option["printed"][p] == raw[p] for p in probes), "probe colors not preserved"
        neutral = [p for p, v in raw.items() if v == 7]
        colors = {info["field"][p] for p in neutral}
        assert len(colors) >= 2, "neutral body does not expose the depth field"
        ink_colors.append(len(colors))
        neutral_depths.extend(info["depths"][p] for p in neutral)
        probe_counts.append(len(probes))
        repeat_sensitive |= any(info["field"][p] != collapsed[min(info["depths"][p], len(collapsed) - 1)]
                                for p in neutral)
        contour_sensitive |= any(info["field"][p] != info["code"][min(p[0] - r0, r1 - p[0],
                                                                     p[1] - c0, c1 - p[1])]
                                 for p in neutral)
        diagonal_sensitive |= any(info["field"][p] != diagonal_field[p] for p in neutral)
    sizes = [len(t) for t in info["targets"]]
    # Ignore literal colors when measuring structures, but retain geometry and
    # code equality patterns. These measurements are descriptive, not a proof.
    roles = {}
    structure = tuple(tuple(roles.setdefault(int(v), len(roles)) for v in row) for row in inp)
    board_norm = tuple(sorted((r - r0, c - c0) for r, c in info["plate"]))
    return {"shape": inp.shape, "layers": len(info["code"]), "code": tuple(info["code"]),
            "palette": len(set(info["code"])), "count": len(info["sources"]),
            "register_sizes": sizes, "patterns": patterns, "masks": masks,
            "probe_counts": probe_counts, "moves": moves, "ink_colors": ink_colors,
            "max_print_depth": max(neutral_depths), "repeat_sensitive": repeat_sensitive,
            "contour_sensitive": contour_sensitive,
            "diagonal_sensitive": diagonal_sensitive,
            "probe_needed": len(_solutions(info, False)) >= 2,
            "registration_needed": (len(info["sources"]) == 1 and
                                    len(_unanchored_prints(info["sources"][0], info)) >= 2),
            "same_count_targets": len(set(sizes)) < len(sizes),
            "board": board_norm, "structure": structure,
            "gap_masks": sum(len(m) < (1 + max(r for r, _ in m)) *
                             (1 + max(c for _, c in m)) for m in masks)}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _grid_key(_arc_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            records[split].append(_features(pair))
    training = records["train"]
    assert len({r["layers"] for r in training}) >= 2
    assert any(r["repeat_sensitive"] and r["probe_needed"] and "rotate" in r["moves"]
               for r in training), "missing multiplicity/compass/rotation demonstration"
    assert any(r["contour_sensitive"] and r["diagonal_sensitive"]
               and r["same_count_targets"] and "reflect" in r["moves"]
               for r in training), "missing notch/equal-count/reflection demonstration"
    assert any(r["registration_needed"] for r in training), "registration has no demonstrated role"
    assert len({r["palette"] for r in training}) >= 2
    for test in records["test"]:
        assert (test["layers"] > max(r["layers"] for r in training)
                or test["count"] > max(r["count"] for r in training)
                or min(test["register_sizes"]) > max(n for r in training for n in r["register_sizes"])), \
            "test has no documented structural shift"
        if test["layers"] > max(r["layers"] for r in training):
            # Count-shift tests can also incidentally grow the plate; only the
            # dedicated depth curriculum guarantees printing the deepest row.
            assert test["max_print_depth"] < test["layers"]


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity audit plus invariants and adversarial checks."""
    records, signatures, colors = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            assert np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"])
            records.append(_features(pair))
            colors.update(int(c) for c in np.unique(pair["input"]))
    assert len(signatures) == 42
    assert len({r["structure"] for r in records}) >= 150
    assert len({r["board"] for r in records}) >= 60
    assert len({m for r in records for m in r["masks"]}) >= 100
    assert {r["layers"] for r in records} == set(range(3, 8))
    assert {r["count"] for r in records} == {1, 2, 3, 4}
    assert {n for r in records for n in r["register_sizes"]} == {1, 2, 3, 4}
    assert {m for r in records for m in r["moves"]} == {"translate", "rotate", "reflect"}
    assert colors == set(range(10))
    assert {r["palette"] for r in records} >= {2, 3, 4, 5, 6}
    assert sum(r["gap_masks"] for r in records) >= 100
    assert sum(r["contour_sensitive"] for r in records) >= 42
    assert sum(r["diagonal_sensitive"] for r in records) >= 42
    assert sum(r["probe_needed"] for r in records) >= 42
    assert sum(r["registration_needed"] for r in records) >= 42
    assert _puzzle_key(generate_puzzle(8675309, 5, 2)) == _puzzle_key(generate_puzzle(8675309, 5, 2))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5),
                 (None, None), (None, 1), (4, None), (-1, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Direct geometry fixtures: the field is rectangular-ring depth on a square
    # and follows a notch through the interior, instead of the bounding box.
    square = {(r, c) for r in range(7) for c in range(7)}
    assert all(d == min(r, c, 6 - r, 6 - c) for (r, c), d in _depth(square).items())
    notched = square - {(r, 3) for r in range(3)}
    assert _depth(notched)[3, 3] == 0 and _depth(notched)[3, 2] == 1
    # Empty cells transform as absence; never fill a stamp's bounding rectangle.
    ring = {(r, c): 7 for r in range(3) for c in range(3) if (r, c) != (1, 1)}
    assert all(len(_transform(ring, t)) == 8 and (1, 1) not in _transform(ring, t) for t in range(8))
    # A full input fixture combines a preserved enclosed gap with a symmetric
    # stamp: several D4 identifiers give one legal printed answer.
    fixture = np.zeros((20, 18), dtype=np.int8)
    fixture[:4] = np.asarray([1, 2, 3, 4], dtype=np.int8)[:, None]
    plate = {(r + 1, c + 1) for r in range(7) for c in range(7)}
    work = fixture[5:]
    for p in plate:
        work[p] = 8
    symmetric = {(0, 0): 7, (0, 1): 1, (0, 2): 7, (1, 0): 7,
                 (1, 2): 7, (2, 0): 3, (2, 1): 9, (2, 2): 3}
    assert len({_signature(_transform(symmetric, t)) for t in range(8)}) == 4
    final = _translate(symmetric, (1, 3))
    source = _translate(_transform(symmetric, 1), (10, 10))
    for p, v in final.items():
        if v == 9:
            work[p] = v
    for p, v in source.items():
        work[p] = v
    info = _parse(fixture)
    field = info["field"]
    expected = _render(info, {p: 9 if v == 9 else field[p] for p, v in final.items()})
    assert np.array_equal(solve(fixture), expected) and expected[7, 4] == 8
    # A deliberately corrupted printed body must fail semantic validation.
    bad = generate_puzzle(0, 2, 1)
    pair = bad["test"][0]
    info = _parse(pair["input"])
    pixels, _ = _solutions(info)[0]
    cell = next(p for p, v in pixels.items() if v != 9)
    pair["output"][cell[0] + info["offset"], cell[1]] = 8
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted contour print accepted")
    return {"seeds": 42, "pairs": len(records),
            "structures": len({r["structure"] for r in records}),
            "plate_shapes": len({r["board"] for r in records}),
            "stamp_masks": len({m for r in records for m in r["masks"]}),
            "probe_dependent_pairs": sum(r["probe_needed"] for r in records),
            "contour_dependent_pairs": sum(r["contour_sensitive"] for r in records),
            "diagonal_sensitive_pairs": sum(r["diagonal_sensitive"] for r in records),
            "repeat_sensitive_pairs": sum(r["repeat_sensitive"] for r in records)}
