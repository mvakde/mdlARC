"""Reassemble a split ruler: destination-ordered, two-arm calibration.

No parent code, saved examples, files, or generator state are consulted by solve.
Generation draws labels from its construction; validation decodes returned cells.
All diagnostics are in memory; this module writes no artifacts.
"""
from __future__ import annotations

import hashlib
import numpy as np

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer) and bool(np.all((a >= 0) & (a <= 9))),
          "grid must contain integer colors 0..9")
    return a


def _counts(a, b):
    for value, low, high in ((a, 2, 10), (b, 1, 4)):
        _need(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
              and low <= value <= high, "illegal example count")
    _need(a > b, "num_train must exceed num_test")
    return int(a), int(b)


def _components(grid, background):
    todo = set(map(tuple, np.argwhere(grid != background).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBORS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _decode(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    _need(int(max(counts)) > grid.size / 2, "background must occupy over half the input")
    anchors, carriers = {}, []
    for cells in _components(grid, background):
        p = min(cells)
        boundary = p[0] in (0, h - 1) or p[1] in (0, w - 1)
        if len(cells) == 1 and boundary:
            _need(p[0] not in (0, h - 1) or p[1] not in (0, w - 1), "corner marker")
            anchors.setdefault(int(grid[p]), []).append(p)
        else:
            carriers.append(cells)
    _need(len(carriers) >= 3 and len(carriers) % 2 == 1, "need an odd rod family")
    _need(len(anchors) == len(carriers) + 1, "need one rail identity and one address per rod")
    guides = {}
    for color, points in anchors.items():
        rows, cols = {p[0] for p in points}, {p[1] for p in points}
        row_guide = len(rows) == 1 and all(c in (0, w - 1) for _, c in points)
        col_guide = len(cols) == 1 and all(r in (0, h - 1) for r, _ in points)
        _need(row_guide != col_guide, "markers do not define one guide")
        _need(len(points) in (1, 2), "too many guide endpoints")
        axis = 0 if row_guide else 1
        guides[color] = (axis, next(iter(rows if row_guide else cols)))
    presence = {color: sum(any(int(grid[p]) == color for p in cells) for cells in carriers)
                for color in anchors}
    shared = [color for color, count in presence.items() if count == len(carriers)]
    _need(len(shared) == 1, "rail identity must occur in every rod")
    rail = shared[0]
    axial, target_axial = guides[rail]
    cross = 1 - axial
    private = set(anchors) - {rail}
    _need(all(presence[k] == 1 and guides[k][0] == cross for k in private),
          "addresses must be private and perpendicular to the rail")
    _need(len({guides[k][1] for k in private}) == len(private), "tied destination addresses")
    rods = []
    for cells in carriers:
        tags = [p for p in cells if int(grid[p]) in private]
        stems = [p for p in cells if int(grid[p]) == rail]
        _need(len(tags) == len(stems) == 1, "rod needs exactly two registration cells")
        pivot, stem = tags[0], stems[0]
        key = int(grid[pivot])
        _need(stem[axial] == pivot[axial] and abs(stem[cross] - pivot[cross]) == 1,
              "rail tag must protrude one cell perpendicular to the rod")
        body = cells - {pivot, stem}
        _need(body and all(p[cross] == pivot[cross] and int(grid[p]) not in anchors for p in body),
              "arms must be straight and use non-marker colors")
        sides = []
        for sign in (-1, 1):
            arm = sorted([p for p in body if sign * (p[axial] - pivot[axial]) > 0],
                         key=lambda p: abs(p[axial] - pivot[axial]))
            _need(arm and [abs(p[axial] - pivot[axial]) for p in arm] == list(range(1, len(arm) + 1)),
                  "each arm must be nonempty and contiguous")
            colors = {int(grid[p]) for p in arm}
            _need(len(colors) == 1, "each arm must be monochrome")
            sides.append((len(arm), colors.pop()))
        _need(sum(x[0] for x in sides) == len(body), "unexpected rod cells")
        dest = list(pivot)
        dest[axial], dest[cross] = target_axial, guides[key][1]
        rods.append({"key": key, "pivot": pivot, "stem_delta": stem[cross] - pivot[cross],
                     "target": tuple(dest), "lengths": tuple(s[0] for s in sides),
                     "colors": tuple(s[1] for s in sides)})
    rods.sort(key=lambda rod: rod["target"][cross])
    return {"shape": grid.shape, "background": background, "anchors": anchors,
            "rail": rail, "axial": axial, "cross": cross, "rods": rods}


def _paint_model(model, lengths=None, dock=True):
    """Render decoded geometry; optional arguments are used only for ablations."""
    out = np.full(model["shape"], model["background"], dtype=np.int8)
    occupied = set()

    def put(p, color):
        _need(all(0 <= p[i] < out.shape[i] for i in (0, 1)), "target leaves the canvas")
        _need(p not in occupied, "contradictory target overlap")
        occupied.add(p)
        out[p] = color

    for color, points in model["anchors"].items():
        for p in points:
            put(p, color)
    a, q = model["axial"], model["cross"]
    for rod in model["rods"]:
        p = rod["target"] if dock else rod["pivot"]
        put(p, rod["key"])
        stem = list(p)
        stem[q] += rod["stem_delta"]
        put(tuple(stem), model["rail"])
        spans = rod["lengths"] if lengths is None else lengths
        for index, sign in enumerate((-1, 1)):
            for step in range(1, spans[index] + 1):
                cell = list(p)
                cell[a] += sign * step
                put(tuple(cell), rod["colors"][index])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Dock the rods and copy both arm spans of the middle destination rod."""
    model = _decode(input_grid)
    reference = model["rods"][len(model["rods"]) // 2]["lengths"]
    return _paint_model(model, reference)


def _gaps(rng, n, minimum, ceiling):
    gaps = [minimum] * (n - 1)
    extra = int(rng.integers(ceiling - sum(gaps) + 1))
    for _ in range(extra):
        gaps[int(rng.integers(n - 1))] += 1
    return gaps


def _offset_positions(rng, gaps, size):
    span = sum(gaps)
    start = int(rng.integers(2, size - span - 2))
    return [start] + [start + int(v) for v in np.cumsum(gaps)]


def _construct_pair(rng, n, reference, witness, transform):
    """Construct a label directly, without invoking solve or its decoder."""
    palette = list(map(int, rng.permutation(10)))
    bg, rail = palette[:2]
    keys, payload = palette[2:n + 2], palette[n + 2:]
    mid = n // 2
    spans = [list(reference) for _ in range(n)]
    others = [i for i in range(n) if i != mid]
    rng.shuffle(others)
    if witness:
        # Equal total length can conceal unequal left/right calibration.
        spans[others[0]] = [reference[0] - 1, reference[1] + 1]
        spans[others[1]] = [reference[0] + 1, reference[1] - 1]
        for i in others[2:]:
            spans[i] = [int(rng.integers(1, v)) for v in reference]
    else:
        mode = int(rng.integers(4))
        for i in others:
            for arm, ref in enumerate(reference):
                if mode == 0:
                    spans[i][arm] = int(rng.integers(1, ref + 1))
                elif mode == 1:
                    spans[i][arm] = int(rng.integers(ref, min(12, ref + 3) + 1))
                else:
                    spans[i][arm] = int(rng.integers(1, min(12, ref + 3) + 1))
        # A real two-arm change; source-middle will never impersonate the ruler.
        for i in others:
            if spans[i] == list(reference):
                arm = int(rng.integers(2))
                spans[i][arm] = reference[arm] - 1
    source_gaps = _gaps(rng, n, 4, 25)
    if witness:
        target_gaps = [3] * (n - 1)
        target_gaps[int(rng.choice([0, n - 2]))] += 7
    else:
        target_gaps = _gaps(rng, n, 3, 25)
    height_min = max(sum(source_gaps), sum(target_gaps)) + 5
    h = int(rng.integers(height_min, 31))
    source_rows = _offset_positions(rng, source_gaps, h)
    target_rows = _offset_positions(rng, target_gaps, h)
    max_left = max(s[0] for s in spans)
    max_right = max(s[1] for s in spans)
    max_width = max_left + max_right + 1
    w = int(rng.integers(max_width + 5, 31))
    rail_col = int(rng.integers(max_left + 2, w - max_right - 2))
    inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    anchors = []
    rail_sides = int(rng.integers(3))  # top, bottom, or both
    for row in ((0,) if rail_sides == 0 else (h - 1,) if rail_sides == 1 else (0, h - 1)):
        anchors.append(((row, rail_col), rail))
    for i, row in enumerate(target_rows):
        sides = int(rng.integers(3))
        for col in ((0,) if sides == 0 else (w - 1,) if sides == 1 else (0, w - 1)):
            anchors.append(((row, col), keys[i]))
    for p, color in anchors:
        inp[p] = out[p] = color
    order = list(map(int, rng.permutation(n)))
    if order[mid] == mid:
        order[0], order[mid] = order[mid], order[0]
    arm_palette = [tuple(map(int, rng.choice(payload, size=2))) for _ in range(n)]
    if witness:
        # Teach local side-specific colors even in the minimum two-demo puzzle.
        color_choices = [(x, y) for x in payload for y in payload
                         if x != y and (x, y) != arm_palette[mid]]
        arm_palette[others[0]] = color_choices[int(rng.integers(len(color_choices)))]
    for source_index, i in enumerate(order):
        row = source_rows[source_index]
        left, right = spans[i]
        stem_sign = int(rng.choice([-1, 1]))
        possible = []
        for col in range(left + 2, w - right - 2):
            cells = {(row, c) for c in range(col - left, col + right + 1)} | {(row + stem_sign, col)}
            if all(inp[r, c] == bg and all(not (0 <= r + dr < h and 0 <= c + dc < w)
                                          or inp[r + dr, c + dc] == bg for dr, dc in _NEIGHBORS)
                   for r, c in cells):
                possible.append(col)
        _need(possible, "construction has no separated source position")
        col = int(rng.choice(possible))
        arm_colors = arm_palette[i]
        for grid, r0, c0, lengths in ((inp, row, col, spans[i]),
                                     (out, target_rows[i], rail_col, reference)):
            grid[r0, c0] = keys[i]
            grid[r0 + stem_sign, c0] = rail
            for sign, length, color in zip((-1, 1), lengths, arm_colors):
                for step in range(1, length + 1):
                    grid[r0, c0 + sign * step] = color
    turns, mirror = transform
    pair = {}
    for name, grid in (("input", inp), ("output", out)):
        grid = np.rot90(grid, turns)
        if mirror:
            grid = np.fliplr(grid)
        pair[name] = grid.copy()
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = int(rng.integers(4))
    common_orientation = bool(rng.integers(2))
    puzzle, seen, training_references = {"train": [], "test": []}, set(), set()
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            witness = split == "train" and index == 0
            n = 5 if witness else 3 if split == "train" and index == 1 else int(rng.choice([3, 5, 7]))
            if split == "train":
                choices = [(x, y) for x in range(2, 6) for y in range(2, 6)
                           if x != y and (x, y) not in training_references
                           and (y, x) not in training_references]
                if not choices:
                    choices = [(x, y) for x in range(2, 6) for y in range(2, 6) if x != y]
            else:
                choices = [(x, y) for x in range(6, 10) for y in range(2, 6)]
                choices += [(y, x) for x, y in choices]
            reference = choices[int(rng.integers(len(choices)))]
            turns = base if common_orientation else (base + index) % 4
            # Tests reverse the demonstrated rod axis, or transfer to its quarter turn.
            if split == "test":
                turns = (base + (2 if common_orientation else 1) + index) % 4
            transform = (turns, bool(rng.integers(2)))
            for _ in range(24):
                pair = _construct_pair(rng, n, reference, witness, transform)
                signature = _signature(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input sampling exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
            if split == "train":
                training_references.add(reference)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(model):
    rods, cross = model["rods"], model["cross"]
    middle = len(rods) // 2
    ref = rods[middle]["lengths"]
    source_middle = sorted(rods, key=lambda r: r["pivot"][cross])[middle]
    coords = [r["target"][cross] for r in rods]
    distances = [abs(2 * x - coords[0] - coords[-1]) for x in coords]
    nearest = min(range(len(rods)), key=lambda i: distances[i])
    median = tuple(sorted(r["lengths"][j] for r in rods)[middle] for j in (0, 1))
    return {"count": len(rods), "reference": ref, "axis": model["axial"],
            "background": model["background"], "shape": model["shape"],
            "source_middle_trap": source_middle["lengths"] != ref,
            "nonmedian": median != ref,
            "geometric_trap": distances.count(distances[nearest]) == 1 and rods[nearest]["lengths"] != ref,
            "equal_total_trap": any(sum(r["lengths"]) == sum(ref) and r["lengths"] != ref for r in rods),
            "local_color_witness": any(r["colors"][0] != r["colors"][1]
                                       and r["colors"] != rods[middle]["colors"] for r in rods),
            "extends": any(any(r["lengths"][j] < ref[j] for j in (0, 1)) for r in rods),
            "trims": any(any(r["lengths"][j] > ref[j] for j in (0, 1)) for r in rods),
            "moves": sum(r["pivot"] != r["target"] for r in rods),
            "gaps": tuple(y - x for x, y in zip(coords, coords[1:]))}


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, stats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "canvas changed")
            key = _signature(inp)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            model = _decode(inp)
            rods = model["rods"]
            _need(len(rods) in (3, 5, 7), "unsupported generated rod count")
            ref = rods[len(rods) // 2]["lengths"]
            expected = _paint_model(model, ref)
            _need(np.array_equal(expected, out), "wrong docking or two-arm calibration")
            f = _features(model)
            _need(f["moves"] and f["source_middle_trap"], "registration lacks a reference-routing witness")
            _need(any(r["lengths"] != ref for r in rods), "normalization does nothing")
            _need(not np.array_equal(_paint_model(model), out), "omitting B did not change the answer")
            # Strong A ablation: even with correct docking, source-middle calibration is wrong.
            wrong_ref = sorted(rods, key=lambda r: r["pivot"][model["cross"]])[len(rods) // 2]["lengths"]
            try:
                wrong = _paint_model(model, wrong_ref)
            except ValueError:  # The alternative may exceed a boundary.
                wrong = None
            _need(wrong is None or not np.array_equal(wrong, out), "omitting routed ordering did not change answer")
            stats[split].append(f)
    train = stats["train"]
    _need(len({f["count"] for f in train}) >= 2, "training needs two rod counts")
    _need(len({f["reference"] for f in train}) >= 2, "training needs distinct split rulers")
    _need(any(all(f[k] for k in ("nonmedian", "geometric_trap", "equal_total_trap", "extends", "trims",
                                "local_color_witness"))
              for f in train), "missing informative calibration demonstration")
    _need(all(max(f["reference"]) <= 5 for f in train)
          and all(max(f["reference"]) >= 6 for f in stats["test"]), "missing target-span extrapolation")


def self_test():
    """Bounded, file-free 42-seed audit, semantic mutations, and metamorphisms."""
    shapes, structures, counts, references, backgrounds, axes = set(), set(), set(), set(), set(), set()
    fingerprints = set()
    summaries = {"seeds": 42, "pairs": 0, "source_middle_traps": 0,
                 "nonmedian": 0, "geometric_traps": 0, "equal_total_traps": 0}
    rng = np.random.default_rng(90123)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                model = _decode(inp)
                f = _features(model)
                shapes.add(inp.shape)
                counts.add(f["count"])
                references.add(f["reference"])
                backgrounds.add(f["background"])
                axes.add(f["axis"])
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in inp.ravel())
                structures.add(hashlib.sha256(str(inp.shape).encode() + normalized).digest())
                summaries["pairs"] += 1
                for key, feature in (("source_middle_traps", "source_middle_trap"),
                                     ("nonmedian", "nonmedian"), ("geometric_traps", "geometric_trap"),
                                     ("equal_total_traps", "equal_total_trap")):
                    summaries[key] += int(f[feature])
                # Exact color and dihedral equivariance, one pair per seed.
                if split == "train" and pair is puzzle[split][0]:
                    palette = rng.permutation(10)
                    for turns in range(4):
                        for mirror in (False, True):
                            x, y = np.rot90(inp, turns), np.rot90(out, turns)
                            if mirror:
                                x, y = np.fliplr(x), np.fliplr(y)
                            _need(np.array_equal(solve(palette[x]), palette[y]), "equivariance failed")
                    # Perturb only visible evidence, without regenerating a pair.
                    ruler = model["rods"][len(model["rods"]) // 2]
                    shortened = inp.copy()
                    tip = list(ruler["pivot"])
                    tip[model["axial"]] -= ruler["lengths"][0]
                    shortened[tuple(tip)] = model["background"]
                    _need(not np.array_equal(solve(shortened), out), "visible ruler is ignored")
                    # Permute only two border addresses. The reference must be rerouted.
                    other = model["rods"][0]
                    rerouted = inp.copy()
                    for old, new in ((ruler["key"], other["key"]), (other["key"], ruler["key"])):
                        for p in model["anchors"][old]:
                            rerouted[p] = new
                    _need(not np.array_equal(solve(rerouted), out), "visible address order is ignored")
        fingerprints.add(digest.digest())
        if seed < 3:
            repeated = generate_puzzle(seed)
            _need(all(np.array_equal(p[k], q[k]) for s in puzzle
                      for p, q in zip(puzzle[s], repeated[s]) for k in p), "nondeterminism")
    _need(len(fingerprints) == 42 and len(structures) >= 200 and len(shapes) >= 50, "structural diversity")
    _need(counts == {3, 5, 7} and axes == {0, 1} and backgrounds == set(range(10)), "role/geometry coverage")
    _need(len(references) >= 20, "reference diversity")
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            generate_puzzle(777 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(99, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    summaries.update({"grid_shapes": len(shapes), "color_normalized_structures": len(structures),
                      "references": len(references), "rod_counts": sorted(counts), "backgrounds": sorted(backgrounds)})
    return summaries
