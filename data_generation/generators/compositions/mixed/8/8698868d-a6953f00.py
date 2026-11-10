"""Cavity-matched inset mosaic, followed by a parity-selected 2x2 crop.

Only NumPy and the standard library are used.  No parent module, saved puzzle,
seed, or generator state is consulted by solve().  See task_spec.md for scope.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CAPACITY = {6: 2, 7: 5, 8: 7, 9: 8, 10: 8}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= d <= 30 for d in array.shape), "grid bounds")
    _require(np.issubdtype(array.dtype, np.integer), "integer grids required")
    _require(bool(np.all((array >= 0) & (array <= 9))), "color bounds")
    return array.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                neighbor = (r + dr, c + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _crop(mosaic):
    h, w = mosaic.shape
    _require(h % 2 == w % 2, "mosaic sides must have equal parity")
    return mosaic[:2, :2].copy() if w % 2 else mosaic[:2, -2:].copy()


def _interpret(grid, background):
    objects = sorted(_components(grid != background), key=len, reverse=True)
    _require(len(objects) >= 3, "need a mosaic and at least two motifs")
    boxes = [_box(cells) for cells in objects]
    r0, r1, c0, c1 = boxes[0]
    shapes = {(b - a, d - c) for a, b, c, d in boxes[1:]}
    _require(len(shapes) == 1, "common motif dimensions")
    s, sw = next(iter(shapes))
    _require(4 <= s <= 8 and s == sw, "square motifs of side 4..8")
    t = s + 2
    mosaic = grid[r0:r1, c0:c1]
    h, w = mosaic.shape
    _require(h % t == w % t == 0 and h % 2 == w % 2, "mosaic partition/parity")
    rows, cols = h // t, w // t
    _require(rows * cols == len(objects) - 1, "one motif per tile")
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, g, j in boxes[i + 1:]:
            _require(b < e or f < a or d < g or j < c, "object boxes need a neutral gap")
    motifs, motif_boxes = {}, {}
    for a, b, c, d in boxes[1:]:
        motif = grid[a:b, c:d]
        colors = np.unique(motif[motif != background])
        _require(len(colors) == 1, "binary motif")
        _require(np.all(_perimeter(motif) == colors[0]), "closed motif perimeter")
        k = len(_components(motif == background))
        _require(k not in motifs, "cavity counts must be unique")
        motifs[k], motif_boxes[k] = motif.copy(), (a, b, c, d)
    counts, carriers, clues = [], [], []
    assembled = np.empty_like(mosaic)
    for i in range(rows * cols):
        rr, cc = (i // cols) * t, (i % cols) * t
        tile = mosaic[rr:rr + t, cc:cc + t]
        carrier = int(tile[0, 0])
        _require(carrier != background and np.all(_perimeter(tile) == carrier), "carrier border")
        _require(set(map(int, np.unique(tile))) <= {background, carrier}, "binary tile")
        clue = tile[1:-1, 1:-1] == background
        _require(not np.any(clue[:-1] & clue[1:]) and not np.any(clue[:, :-1] & clue[:, 1:]),
                 "clues must be four-isolated")
        _require(len(_components(tile != background)) == 1, "connected carrier")
        k = int(clue.sum())
        _require(k in motifs and int(motifs[k][0, 0]) != carrier, "visible unique match")
        result = np.full((t, t), carrier, dtype=np.int8)
        result[1:-1, 1:-1] = np.where(motifs[k] == background, carrier, motifs[k])
        assembled[rr:rr + t, cc:cc + t] = result
        counts.append(k)
        carriers.append(carrier)
        clues.append(clue.copy())
    _require(len(set(counts)) == len(counts) and set(counts) == set(motifs), "count bijection")
    return {
        "background": background, "tile_size": t, "layout": (rows, cols),
        "mosaic_box": boxes[0], "mosaic_input": mosaic.copy(), "assembled": assembled,
        "motifs": motifs, "motif_boxes": motif_boxes, "counts": tuple(counts),
        "carriers": tuple(carriers), "clues": clues,
    }


def _decode(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for background in map(int, np.unique(grid)):
        try:
            candidates.append(_interpret(grid, background))
        except (AssertionError, IndexError, ValueError):
            continue
    _require(len(candidates) == 1, "input needs exactly one valid scene interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the inset mosaic from input alone, then crop by its parity."""
    return _crop(_decode(input_grid)["assembled"])


def _isolated(rng, size, count):
    """Independent clue/cavity seeds, with connected surrounding foreground."""
    if count == 0:
        return np.zeros((size, size), dtype=bool)
    for _ in range(12):
        mask = np.zeros((size, size), dtype=bool)
        for index in rng.permutation(size * size):
            r, c = divmod(int(index), size)
            if any(0 <= r + dr < size and 0 <= c + dc < size and mask[r + dr, c + dc]
                   for dr, dc in _STEPS):
                continue
            mask[r, c] = True
            if len(_components(np.pad(~mask, 1, constant_values=True))) != 1:
                mask[r, c] = False
                continue
            if int(mask.sum()) == count:
                return mask
    if size <= 3:
        cells = [(r, c) for r in range(size) for c in range(size) if (r + c) % 2 == 0]
    elif size == 4:
        cells = [(0, 0), (0, 2), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]
    else:
        cells = ([(r, c) for r in (0, size - 1) for c in range(0, size, 2)]
                 + [(r, c) for r in range(2, size - 2, 2) for c in (0, size - 1)])
    _require(len(cells) >= count, "independent-set capacity")
    mask = np.zeros((size, size), dtype=bool)
    for index in rng.permutation(len(cells))[:count]:
        mask[cells[int(index)]] = True
    _require(len(_components(np.pad(~mask, 1, constant_values=True))) == 1, "fallback connectivity")
    return mask


def _cavities(rng, side, count):
    if count == 0:
        return np.zeros((side, side), dtype=bool)
    n = side - 2
    mode = int(rng.integers(4))
    if count == 1 and mode == 0:
        return np.pad(np.ones((n, n), dtype=bool), 1)
    if count == 1 and mode == 1:
        mask = np.zeros((n, n), dtype=bool)
        r0, c0 = int(rng.integers(n)), int(rng.integers(n))
        mask[r0:int(rng.integers(r0 + 1, n + 1)), c0:int(rng.integers(c0 + 1, n + 1))] = True
        return np.pad(mask, 1)
    mask = np.pad(_isolated(rng, n, count), 1)
    target = int(rng.integers(count, n * n + 1))
    for _ in range(3):
        for index in rng.permutation(n * n):
            if int(mask.sum()) >= target:
                return mask
            r, c = divmod(int(index), n)
            r, c = r + 1, c + 1
            if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _STEPS):
                continue
            trial = mask.copy()
            trial[r, c] = True
            if len(_components(trial)) == count and len(_components(~trial)) == 1:
                mask = trial
    return mask


def _free_origins(boxes, height, width):
    rr, cc = np.indices((31 - height, 31 - width))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, h, w in boxes:
        free &= (rr + height < r) | (rr > r + h) | (cc + width < c) | (cc > c + w)
    return np.argwhere(free)


def _fallback_pack(config):
    t, rows, cols = config
    boxes = [(0, 0, rows * t, cols * t)]
    for _ in range(rows * cols):
        free = _free_origins(boxes, t - 2, t - 2)
        if not len(free):
            return None
        r, c = map(int, free[0])
        boxes.append((r, c, t - 2, t - 2))
    return boxes


_CONFIGS = tuple(
    (t, r, c) for t in range(6, 11)
    for r, c in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
    if r * t <= 30 and c * t <= 30 and (r * t) % 2 == (c * t) % 2
    and r * c <= _CAPACITY[t] + 1 and _fallback_pack((t, r, c)) is not None
)


def _place(rng, mosaic, motifs, background, full_canvas=False):
    h, w = mosaic.shape
    s = len(motifs[0])
    boxes = None
    for _ in range(12):
        r = _choice(rng, (0, 30 - h, int(rng.integers(31 - h))))
        c = _choice(rng, (0, 30 - w, int(rng.integers(31 - w))))
        trial = [(r, c, h, w)]
        for _ in motifs:
            free = _free_origins(trial, s, s)
            if not len(free):
                break
            rr, cc = map(int, free[int(rng.integers(len(free)))])
            trial.append((rr, cc, s, s))
        if len(trial) == len(motifs) + 1:
            boxes = trial
            break
    if boxes is None:
        boxes = _fallback_pack((s + 2, h // (s + 2), w // (s + 2)))
    _require(boxes is not None, "packing configuration")
    canvas = np.full((30, 30), background, dtype=np.int8)
    objects = [mosaic] + [motifs[int(i)] for i in rng.permutation(len(motifs))]
    for (r, c, bh, bw), obj in zip(boxes, objects):
        canvas[r:r + bh, c:c + bw] = obj
    if full_canvas or rng.random() < 0.2:
        return canvas
    lo_r, lo_c = min(b[0] for b in boxes), min(b[1] for b in boxes)
    hi_r, hi_c = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    lo_r = int(rng.integers(max(0, lo_r - 4), lo_r + 1))
    lo_c = int(rng.integers(max(0, lo_c - 4), lo_c + 1))
    hi_r = int(rng.integers(hi_r, min(30, hi_r + 4) + 1))
    hi_c = int(rng.integers(hi_c, min(30, hi_c + 4) + 1))
    return canvas[lo_r:hi_r, lo_c:hi_c].copy()


def _pair(rng, config, witness=False):
    t, rows, cols = config
    n, s = rows * cols, t - 2
    if witness:
        counts = [0, 1, 2]
        counts += list(map(int, rng.choice(np.arange(3, _CAPACITY[t] + 1), n - 3, replace=False)))
    else:
        counts = list(map(int, rng.choice(_CAPACITY[t] + 1, n, replace=False)))
    order = list(map(int, rng.permutation(counts)))
    selected = 0 if (t * cols) % 2 else cols - 1
    if witness:
        selected_count = int(rng.choice([1, 2]))
        index = order.index(selected_count)
        order[selected], order[index] = order[index], order[selected]
    background = int(rng.integers(10))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != background])))
    pool = palette[:int(rng.integers(max(2, n if witness else 2), 10))]
    if witness:
        foregrounds = list(map(int, rng.choice(pool, n, replace=False)))
    else:
        foregrounds = list(map(int, rng.choice(pool, n, replace=True)))
        if len(set(foregrounds)) == 1:
            other = (selected + 1) % n
            foregrounds[other] = _choice(rng, [c for c in pool if c != foregrounds[selected]])
    carriers = [_choice(rng, [c for c in pool if c != foregrounds[i]]) for i in range(n)]
    motifs, masks = {}, {}
    for i, k in enumerate(order):
        mask = _cavities(rng, s, k)
        if witness and k == 1:
            mask = np.pad(np.ones((s - 2, s - 2), dtype=bool), 1)
        if witness and k == 2:
            mask = np.zeros((s, s), dtype=bool)
            mask[1, 1] = mask[2, 2] = mask[2, 3] = True
            mask = np.rot90(mask, int(rng.integers(4)))
            if rng.random() < 0.5:
                mask = mask.T
        motif = np.full((s, s), foregrounds[i], dtype=np.int8)
        motif[mask] = background
        motifs[k], masks[k] = motif, mask
    mosaic, assembled = np.empty((rows * t, cols * t), dtype=np.int8), np.empty((rows * t, cols * t), dtype=np.int8)
    for i, k in enumerate(order):
        r, c = (i // cols) * t, (i % cols) * t
        tile = np.full((t, t), carriers[i], dtype=np.int8)
        clues = _isolated(rng, s, k)
        tile[1:-1, 1:-1][clues] = background
        mosaic[r:r + t, c:c + t] = tile
        tile[:] = carriers[i]
        tile[1:-1, 1:-1] = np.where(masks[k], carriers[i], motifs[k])
        assembled[r:r + t, c:c + t] = tile
    return {"input": _place(rng, mosaic, list(motifs.values()), background, witness),
            "output": _crop(assembled)}


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one deterministic puzzle; explicit counts never change the rule."""
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    odd = [conf for conf in _CONFIGS if (conf[0] * conf[2]) % 2]
    even = [conf for conf in _CONFIGS if not (conf[0] * conf[2]) % 2]
    first = _choice(rng, odd)
    second = _choice(rng, [conf for conf in even if conf[0] % 2 and conf[1:] == (2, 2)])
    configs = [first, second]
    if num_train >= 4:
        configs.append(_choice(rng, [conf for conf in odd if conf[0] != first[0]]))
        configs.append(_choice(rng, [conf for conf in even if conf[0] % 2 == 0]))
    while len(configs) < num_train:
        configs.append(_choice(rng, _CONFIGS))
    result, used = {"train": [], "test": []}, set()
    for i, config in enumerate(configs):
        pair = _pair(rng, config, witness=i < 2)
        pair = _fresh(pair, used)
        result["train"].append(pair)
    test_parities = [1, 0] if num_test >= 2 else [int(rng.integers(2))]
    while len(test_parities) < num_test:
        test_parities.append(int(rng.integers(2)))
    rng.shuffle(test_parities)
    for parity in test_parities:
        eligible = odd if parity else even
        novel = [conf for conf in eligible if conf not in configs]
        config = _choice(rng, novel or eligible)
        result["test"].append(_fresh(_pair(rng, config), used))
    result["train"] = [result["train"][int(i)] for i in rng.permutation(num_train)]
    return result


def _fresh(pair, used):
    if _key(pair["input"]) not in used:
        used.add(_key(pair["input"]))
        return pair
    # At most 13 earlier inputs, but 90 distinct images of two present colors.
    # This finite fallback preserves every structural training witness.
    present = list(map(int, np.unique(pair["input"])))
    source = present[:2] + [c for c in range(10) if c not in present[:2]]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            target = [a, b] + [c for c in range(10) if c not in (a, b)]
            lookup = np.empty(10, dtype=np.int8)
            lookup[source] = target
            changed = {k: lookup[v] for k, v in pair.items()}
            key = _key(changed["input"])
            if key not in used:
                used.add(key)
                return changed
    raise AssertionError("finite uniqueness bound violated")


def _area_crop(analysis):
    """Competing assignment: rank motifs by neutral area instead of cavities."""
    bg, motifs = analysis["background"], analysis["motifs"]
    areas = {k: int((motif == bg).sum()) for k, motif in motifs.items()}
    if len(set(areas.values())) < len(areas):
        return None
    assigned = dict(zip(sorted(motifs), sorted(motifs, key=areas.get)))
    out = analysis["assembled"].copy()
    t, (_, cols) = analysis["tile_size"], analysis["layout"]
    for i, k in enumerate(analysis["counts"]):
        r, c = (i // cols) * t, (i % cols) * t
        carrier = analysis["carriers"][i]
        out[r:r + t, c:c + t] = carrier
        motif = motifs[assigned[k]]
        out[r + 1:r + t - 1, c + 1:c + t - 1] = np.where(motif == bg, carrier, motif)
    return _crop(out)


def _validate(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(grid)
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            analysis = _decode(grid)
            mosaic = analysis["assembled"]
            _require(out.shape == (2, 2) and np.array_equal(out, _crop(mosaic)), "incorrect parity crop")
            _require(not np.array_equal(out, _crop(analysis["mosaic_input"])), "matching is invisible")
            _require(not np.array_equal(mosaic[:2, :2], mosaic[:2, -2:]), "parity is invisible")
            bg = analysis["background"]
            _require(bg not in out and len(np.unique(out)) == 2, "inset corner color roles")
            analyses[split].append(analysis)
    training = analyses["train"]
    _require({x["assembled"].shape[1] % 2 for x in training} == {0, 1}, "teach both parity branches")
    _require(any(x["tile_size"] % 2 != x["assembled"].shape[1] % 2 for x in training), "tile parity shortcut")
    _require(any(np.asarray(p["input"]).shape[0] % 2 != x["assembled"].shape[0] % 2
                 for p, x in zip(puzzle["train"], training)), "canvas parity shortcut")
    _require(any(_area_crop(x) is not None and not np.array_equal(_area_crop(x), _crop(x["assembled"]))
                 for x in training), "training permits area-ranking shortcut")
    _require(any(len(_components(m == x["background"], True)) < k
                 for x in training for k, m in x["motifs"].items()), "four/eight connectivity witness")
    _require(any(k and not np.array_equal(clue, x["motifs"][k] == x["background"])
                 for x in training for k, clue in zip(x["counts"], x["clues"])), "clue geometry shortcut")
    return analyses


def validate_generated_puzzle(puzzle):
    """Validate schema, unique parsing, exact assembly/crop and teaching witnesses."""
    _validate(puzzle)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(array.shape))
                digest.update(array.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds and a finite count matrix; write no files or traces."""
    fingerprints, shapes, configs, backgrounds, masks, normalized, count_sets = (set() for _ in range(7))
    pairs = repeated_carriers = multi_cell = parity_mismatch = 0
    positional_failures = area_failures = novel_tests = 0
    test_parities, selected_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        records = _validate(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        train_configs = {(x["tile_size"], *x["layout"]) for x in records["train"]}
        for split in ("train", "test"):
            for pair, x in zip(puzzle[split], records[split]):
                pairs += 1
                inp = pair["input"]
                shapes.add(inp.shape)
                configs.add((x["tile_size"], *x["layout"]))
                backgrounds.add(x["background"])
                count_sets.add(tuple(sorted(x["counts"])))
                repeated_carriers += len(set(x["carriers"])) < len(x["carriers"])
                parity_mismatch += inp.shape[0] % 2 != x["assembled"].shape[0] % 2
                index = 0 if x["assembled"].shape[1] % 2 else x["layout"][1] - 1
                spatial_order = sorted(x["motif_boxes"], key=lambda k: (x["motif_boxes"][k][0], x["motif_boxes"][k][2]))
                positional_failures += int(x["motifs"][spatial_order[index]][0, 0]) != int(x["motifs"][x["counts"][index]][0, 0])
                area_prediction = _area_crop(x)
                area_failures += area_prediction is not None and not np.array_equal(area_prediction, pair["output"])
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                normalized.add((inp.shape, hashlib.sha256(norm).hexdigest()))
                for k, motif in x["motifs"].items():
                    mask = motif == x["background"]
                    masks.add((mask.shape, mask.tobytes()))
                    multi_cell += any(len(cells) > 1 for cells in _components(mask))
                if split == "test":
                    novel_tests += (x["tile_size"], *x["layout"]) not in train_configs
                    parity = x["assembled"].shape[1] % 2
                    test_parities.add(parity)
                    index = 0 if parity else x["layout"][1] - 1
                    selected_counts.add(x["counts"][index])
                    original = inp.copy()
                    _require(np.array_equal(solve(inp), pair["output"]), "input-only solve")
                    _require(np.array_equal(inp, original), "solve mutated its input")
        if seed == 0:
            _require(_fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle), "determinism")
    _require(len(fingerprints) == 42 and len(normalized) >= 180, "structural variation")
    _require(len(shapes) >= 15 and len(configs) >= 12 and len(backgrounds) >= 8, "geometry/color variation")
    _require(len(masks) >= 70 and len(count_sets) >= 20, "cavity/count variation")
    _require(test_parities == {0, 1} and len(selected_counts) >= 5, "test selection variation")
    _require(repeated_carriers >= 10 and multi_cell >= 40 and parity_mismatch >= 42, "combined variation")
    _require(positional_failures >= 30 and area_failures >= 42 and novel_tests == 42, "shortcut/shift evidence")
    legal_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            _validate(generate_puzzle(314159, a, b))
            legal_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Counterfactuals prove matching changes a retained pixel at constant parity.
    puzzle = generate_puzzle(7, 2, 1)
    pair = puzzle["train"][0]
    x = _decode(pair["input"])
    parity = x["assembled"].shape[1] % 2
    index = 0 if parity else x["layout"][1] - 1
    k = x["counts"][index]
    a, b, c, d = x["motif_boxes"][k]
    changed = pair["input"].copy()
    motif = changed[a:b, c:d]
    old_color = int(motif[0, 0])
    new_color = next(color for color in range(10) if color not in (old_color, x["background"], x["carriers"][index]))
    motif[motif == old_color] = new_color
    _require(not np.array_equal(solve(changed), pair["output"]), "motif color has no effect")
    # Change ONLY cavity patterns in the final comparison: exchange two count
    # addresses while preserving all motif colors, positions and dimensions.
    # First normalize carriers to a spare color so both assignments stay visible.
    baseline = pair["input"].copy()
    spare = next(color for color in range(10)
                 if color != x["background"] and all(color != int(m[0, 0]) for m in x["motifs"].values()))
    r0, r1, c0, c1 = x["mosaic_box"]
    region = baseline[r0:r1, c0:c1]
    region[region != x["background"]] = spare
    swapped = baseline.copy()
    other = next(count for count in x["motifs"] if count != k)
    for destination, source in ((k, other), (other, k)):
        a, b, c, d = x["motif_boxes"][destination]
        target = swapped[a:b, c:d]
        target[:] = int(x["motifs"][destination][0, 0])
        target[x["motifs"][source] == x["background"]] = x["background"]
    _require(not np.array_equal(solve(baseline), solve(swapped)), "cavity assignment has no retained effect")
    # The validator must reject a wrong label and missing odd demonstrations.
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    even_only = {"train": [], "test": puzzle["test"]}
    even_pair = next(p for p in puzzle["train"] if _decode(p["input"])["assembled"].shape[1] % 2 == 0)
    even_only["train"] = [even_pair, {k: (v + 1) % 10 for k, v in even_pair.items()}]
    try:
        validate_generated_puzzle(even_only)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing odd demonstration accepted")
    as_lists = {s: [{k: v.tolist() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    validate_generated_puzzle(as_lists)
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "configurations": len(configs),
            "backgrounds": len(backgrounds), "cavity_masks": len(masks),
            "count_sets": len(count_sets), "normalized_inputs": len(normalized),
            "repeated_carrier_pairs": repeated_carriers, "multi_cell_motifs": multi_cell,
            "canvas_parity_mismatches": parity_mismatch, "test_parities": sorted(test_parities),
            "selected_test_counts": sorted(selected_counts), "legal_count_calls": legal_calls,
            "position_shortcut_failures": positional_failures, "area_shortcut_failures": area_failures,
            "tests_with_novel_geometry": novel_tests, "cavity_swap_counterfactual": True}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
