"""Color-set loom: object membership controls the multiplicity of each slice.

No parent modules, files, examples, seeds, or generation metadata are used by
solve. Generation and self-tests retain no files or execution traces.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells = [start]
        for r, c in cells:
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    cells.append(p)
        result.append(cells)
    return result


def _read_panels(g, cuts, divider, transposed):
    a, b, c = cuts
    if not (0 < a < b - 1 < c - 2 < g.shape[1] - 3):
        raise ValueError("empty panel")
    source, records = g[:, :a], g[:, a + 1:b]
    p, s = g[:, b + 1:c], g[:, c + 1:]
    if not (np.all(p == p.flat[0]) and np.all(s == s.flat[0]) and p.flat[0] != s.flat[0]):
        raise ValueError("invalid swatches")
    bg = int(source[0, 0])
    if not (np.all(source[0] == bg) and np.all(source[-1] == bg)
            and np.all(source[:, 0] == bg) and np.all(source[:, -1] == bg)):
        raise ValueError("source needs a background border")
    occupied = np.argwhere(source != bg)
    if not len(occupied):
        raise ValueError("empty stencil")
    lo, hi = occupied.min(axis=0), occupied.max(axis=0) + 1
    tile = source[lo[0]:hi[0], lo[1]:hi[1]].copy()
    queries = [frozenset(map(int, np.unique(col[col != bg]))) for col in tile.T]
    if any(len(q) != 3 for q in queries):
        raise ValueError("each source slice needs three identities")
    objects = []
    boxes = []
    for cells in _components(records != bg):
        coords = np.array(cells)
        low, high = coords.min(axis=0), coords.max(axis=0) + 1
        r0, c0 = map(int, low)
        r1, c1 = map(int, high)
        if len(cells) != (r1 - r0) * (c1 - c0):
            raise ValueError("record is not a filled rectangle")
        box = (r0, r1, c0, c1)
        if any(not (r1 < x0 or x1 < r0 or c1 < y0 or y1 < c0)
               for x0, x1, y0, y1 in boxes):
            raise ValueError("record boxes need a background gap")
        boxes.append(box)
        values, counts = np.unique(records[r0:r1, c0:c1], return_counts=True)
        objects.append({"box": box, "colors": frozenset(map(int, values)),
                        "counts": {int(v): int(n) for v, n in zip(values, counts)}})
    if not objects:
        raise ValueError("no records")
    # All label identities are distinct from a common, visibly dominant filler.
    # Membership itself does not require deciding which non-query color fills.
    label_union = set().union(*queries)
    common = set.intersection(*(set(o["colors"]) for o in objects)) - label_union
    totals = {v: sum(o["counts"].get(v, 0) for o in objects) for v in common}
    if not totals:
        raise ValueError("missing common filler")
    best = max(totals.values())
    modes = [v for v in totals if totals[v] == best]
    if len(modes) != 1:
        raise ValueError("ambiguous filler")
    filler = modes[0]
    for obj in objects:
        obj["labels"] = obj["colors"] - {filler}
        obj["marks"] = sum(n for v, n in obj["counts"].items() if v != filler)
        obj["matches"] = tuple(q <= obj["colors"] for q in queries)
    weights = tuple(sum(obj["matches"][i] for obj in objects) for i in range(len(queries)))
    if not (1 <= sum(weights) <= 15):
        raise ValueError("empty or oversized weave")
    return {"transposed": transposed, "background": bg, "divider": divider,
            "primary": int(p.flat[0]), "secondary": int(s.flat[0]),
            "filler": filler, "tile": tile, "queries": queries, "weights": weights,
            "objects": objects, "cuts": cuts,
            "crop_box": (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])),
            "panel_shape": records.shape}


def _decode(grid):
    grid = _grid(grid)
    candidates = []
    for transposed in (False, True):
        g = grid.T if transposed else grid
        for divider in range(10):
            cuts = tuple(map(int, np.flatnonzero(np.all(g == divider, axis=0))))
            if len(cuts) != 3 or np.count_nonzero(g == divider) != 3 * g.shape[0]:
                continue
            try:
                candidates.append(_read_panels(g, cuts, divider, transposed))
            except ValueError:
                continue
    _require(len(candidates) == 1, "input needs exactly one four-panel interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read local roles, then expand each source slice by its containment count."""
    d = _decode(input_grid)
    tile, weights = d["tile"], d["weights"]
    out = np.full((tile.shape[0], 2 * sum(weights) - 1), d["secondary"], dtype=np.int8)
    # Solver writes by positions; generation below constructs concatenated strips.
    cursor = 0
    for col, weight in enumerate(weights):
        for _ in range(weight):
            out[tile[:, col] != d["background"], cursor] = d["primary"]
            cursor += 2
    return out.T.copy() if d["transposed"] else out


def _render(tile, bg, primary, secondary, weights):
    columns = [np.where(tile[:, i:i + 1] != bg, primary, secondary).astype(np.int8)
               for i, n in enumerate(weights) for _ in range(n)]
    gap = np.full((tile.shape[0], 1), secondary, dtype=np.int8)
    return np.concatenate([piece for i, col in enumerate(columns)
                           for piece in ((col,) if i == len(columns) - 1 else (col, gap))], axis=1)


def _tile(rng, height, queries, bg):
    # No connectivity or opposite-corner condition: the complete multicolor
    # stencil, including gaps and disconnected pieces, supplies the tight crop.
    for _ in range(200):
        masks = []
        for _q in queries:
            mask = np.zeros(height, dtype=bool)
            mask[rng.choice(height, int(rng.integers(3, height)), replace=False)] = True
            masks.append(mask)
        mask = np.stack(masks, axis=1)
        if not mask[0].any() or not mask[-1].any():
            continue
        if len({m.tobytes() for m in mask.T}) != len(queries):
            continue
        if np.array_equal(mask, mask[::-1]):
            continue
        break
    else:
        raise AssertionError("stencil sampling exhausted")
    tile = np.full(mask.shape, bg, dtype=np.int8)
    for i, q in enumerate(queries):
        positions = np.flatnonzero(mask[:, i])
        labels = list(rng.permutation(sorted(q)))
        labels += list(rng.choice(sorted(q), len(positions) - 3))
        rng.shuffle(labels)
        tile[positions, i] = labels
    return tile


def _cuts(box, minimum=12):
    r0, r1, c0, c1 = box
    choices = []
    for gap in (1, 2, 3):
        for r in range(r0 + 1, r1 - gap):
            if (r - r0) * (c1 - c0) >= minimum and (r1 - r - gap) * (c1 - c0) >= minimum:
                choices.append(((r0, r, c0, c1), (r + gap, r1, c0, c1)))
        for c in range(c0 + 1, c1 - gap):
            if (r1 - r0) * (c - c0) >= minimum and (r1 - r0) * (c1 - c - gap) >= minimum:
                choices.append(((r0, r1, c0, c), (r0, r1, c + gap, c1)))
    return choices


def _layout(rng, h, w, count):
    # At most 16 random partitions. A two-cell-high-band fallback is guaranteed
    # by h >= 3*count-1 and w >= 9. No search artifacts are retained.
    for _ in range(16):
        leaves = [(0, h, 0, w)]
        while len(leaves) < count:
            options = [(i, _cuts(b)) for i, b in enumerate(leaves)]
            options = [(i, cuts) for i, cuts in options if cuts]
            if not options:
                break
            i, cuts = options[int(rng.integers(len(options)))]
            chosen = cuts[int(rng.integers(len(cuts)))]
            leaves.pop(i)
            leaves.extend(chosen)
        if len(leaves) == count:
            break
    else:
        leaves = [(3 * i, 3 * i + 2, 0, w) for i in range(count)]
    rng.shuffle(leaves)
    boxes = []
    for r0, r1, c0, c1 in leaves:
        sizes = [(a, b) for a in range(1, r1 - r0 + 1)
                 for b in range(1, c1 - c0 + 1) if a * b >= 12]
        a, b = sizes[int(rng.integers(len(sizes)))]
        top = r0 + int(rng.integers(r1 - r0 - a + 1))
        left = c0 + int(rng.integers(c1 - c0 - b + 1))
        boxes.append((top, top + a, left, left + b))
    return boxes


def _package(labels, index):
    a, b, c, d, e, f = labels
    if index == 0:
        return [[a, b, c], [a, b, d], [b, c, e],
                [a, c, d, e, a, e], [a, b, c, e, a, e]]
    return [[a, b, c], [a, b, c, e, a, e], [a, b, d, d],
            [a, b, c, d, e], [c, e, f, f], []]


def _free_records(rng, labels, queries):
    count = int(rng.integers(2, 9))
    # Favor both sparse and rich sets. Crucially, the number and size of marks
    # do not determine membership; incomplete sets may be richer than matches.
    for _ in range(40):
        marks = []
        for _obj in range(count):
            size = int(rng.choice(7, p=[.07, .06, .13, .32, .24, .14, .04]))
            selected = list(map(int, rng.choice(labels, size, replace=False)))
            if selected:
                selected += list(map(int, rng.choice(selected, int(rng.integers(0, 3)))))
            marks.append(selected)
        weights = [sum(set(q) <= set(m) for m in marks) for q in queries]
        if 1 <= sum(weights) <= 15:
            return marks
    # A bounded fallback remains an ordinary member of the same rule.
    return [list(queries[0])] + [[] for _ in range(count - 1)]


def _make_pair(rng, palette, swatches, tile, marks, transposed):
    bg, divider, filler = palette[:3]
    primary, secondary = swatches
    weights = [sum(set(col[col != bg]) <= set(m) for m in marks) for col in tile.T]
    _require(1 <= sum(weights) <= 15, "output budget exceeded")
    count = len(marks)
    h_min = max(tile.shape[0] + 2, 3 * count - 1, 10)
    h = int(rng.integers(h_min, min(30, h_min + 7) + 1))
    source_w = tile.shape[1] + int(rng.integers(2, 5))
    swatch_widths = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
    max_w = 30 - source_w - 3 - sum(swatch_widths)
    _require(max_w >= 9, "input budget exceeded")
    record_w = int(rng.integers(9, max_w + 1))
    source = np.full((h, source_w), bg, dtype=np.int8)
    top = int(rng.integers(1, h - tile.shape[0]))
    left = int(rng.integers(1, source_w - tile.shape[1]))
    source[top:top + tile.shape[0], left:left + tile.shape[1]] = tile
    records = np.full((h, record_w), bg, dtype=np.int8)
    boxes = _layout(rng, h, record_w, count)
    for (r0, r1, c0, c1), tokens in zip(boxes, marks):
        area = (r1 - r0) * (c1 - c0)
        records[r0:r1, c0:c1] = filler
        for position, color in zip(rng.choice(area, len(tokens), replace=False), rng.permutation(tokens)):
            r, c = divmod(int(position), c1 - c0)
            records[r0 + r, c0 + c] = color
    line = np.full((h, 1), divider, dtype=np.int8)
    swatches = [np.full((h, width), color, dtype=np.int8)
                for color, width in zip(swatches, swatch_widths)]
    inp = np.concatenate([source, line, records, line, swatches[0], line, swatches[1]], axis=1)
    out = _render(tile, bg, primary, secondary, weights)
    if transposed:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _counts(num_train, num_test):
    for v, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError("pair counts must be legal integers")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    bg, divider, _filler = palette[:3]
    labels = palette[3:9]
    a, b, c, d, e, f = labels
    teaching_queries = [(a, b, c), (a, b, d), (c, e, f)]
    teaching_tile = _tile(rng, int(rng.integers(5, 10)), teaching_queries, bg)
    swatch_choices = [v for v in range(10) if v != divider]
    teaching_swatches = tuple(map(int, rng.choice(swatch_choices, 2, replace=False)))
    mixed = bool(rng.integers(2))
    base = bool(rng.integers(2))
    directions = [base, not base] if mixed else [base]
    all_queries = list(itertools.combinations(labels, 3))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_height_max = teaching_tile.shape[0]
    for split, amount in (("train", nt), ("test", nq)):
        for index in range(amount):
            for attempt in range(16):
                teaching = split == "train" and index < 2
                if teaching:
                    tile = teaching_tile
                    marks = _package(labels, index)
                    swatches = teaching_swatches
                    transposed = directions[index % len(directions)]
                else:
                    axial = int(rng.integers(2, 6))
                    queries = [all_queries[int(k)] for k in rng.choice(len(all_queries), axial, replace=False)]
                    height = (int(rng.integers(5, 11)) if split == "train" else
                              int(rng.integers(train_height_max + 1, min(22, train_height_max + 8) + 1)))
                    tile = _tile(rng, height, queries, bg)
                    marks = _free_records(rng, labels, queries)
                    swatches = tuple(map(int, rng.choice(swatch_choices, 2, replace=False)))
                    transposed = directions[int(rng.integers(len(directions)))]
                pair = _make_pair(rng, palette, swatches, tile, marks, transposed)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise AssertionError("distinct-input sampling exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_height_max = max(train_height_max, tile.shape[0])
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(infos):
    # Check evidence reconstructed from actual cells, not package identifiers.
    _require(any(0 in d["weights"] and max(d["weights"]) >= 2 for d in infos), "missing zero and repetition witness")
    _require(any(min(d["weights"]) >= 1 and len(set(d["weights"])) >= 3 for d in infos), "missing nonuniform positive counts")
    _require(any(_key(a["tile"]) == _key(b["tile"])
                 and sum(x != y for x, y in zip(a["weights"], b["weights"])) >= 2
                 for a in infos for b in infos), "same stencil must respond to changed records")
    _require(any(sum(o["matches"]) >= 2 for d in infos for o in d["objects"]), "missing shared-record interaction")
    _require(any(not o["labels"] for d in infos for o in d["objects"]), "missing filler-only record")
    witnessed = False
    for d in infos:
        for query in d["queries"]:
            yes = [o for o in d["objects"] if query <= o["labels"]]
            no = [o for o in d["objects"] if not query <= o["labels"]]
            if not yes or not no:
                continue
            if not all(any(o["labels"] & query == query - {q} for o in no) for q in query):
                continue
            exact = any(o["marks"] == 3 for o in yes)
            tied = any(o["marks"] == 3 and len(o["labels"]) == 3 for o in no)
            rich = any(o["marks"] > 3 and len(o["labels"]) > 3 for o in no)
            extra = any(o["labels"] - query and any(n > 1 for v, n in o["counts"].items()
                                                     if v != d["filler"]) for o in yes)
            witnessed |= exact and tied and rich and extra
    _require(witnessed, "training does not distinguish identities, extras, repetitions and all three requirements")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            expected = solve(inp)
            _require(np.array_equal(out, expected), "wrong slice expansion")
            canonical = out.T if d["transposed"] else out
            _require(canonical.shape == (d["tile"].shape[0], 2 * sum(d["weights"]) - 1), "wrong output extent")
            _require(np.all(canonical[:, 1::2] == d["secondary"]), "separator is not one secondary cell wide")
            _require(set(map(int, np.unique(out))) <= {d["primary"], d["secondary"]}, "output palette")
            _require(np.count_nonzero(canonical == d["primary"]) ==
                     sum(n * int(np.count_nonzero(d["tile"][:, i] != d["background"]))
                         for i, n in enumerate(d["weights"])), "foreground mass is not conserved under multiplicity")
            infos[split].append(d)
    _training_evidence(infos["train"])
    roles = {(d["background"], d["divider"], d["filler"]) for ds in infos.values() for d in ds}
    _require(len(roles) == 1, "structural palette must be shared within one puzzle")
    train_dirs = {d["transposed"] for d in infos["train"]}
    max_height = max(d["tile"].shape[0] for d in infos["train"])
    _require(all(d["transposed"] in train_dirs for d in infos["test"]), "unshown orientation")
    _require(all(d["tile"].shape[0] > max_height for d in infos["test"]), "missing cross-axis stencil shift")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = _grid(pair[role])
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def _metamorphic(puzzle):
    decoded = [(p, _decode(p["input"])) for p in puzzle["train"]]
    pair, d = next((p, d) for p, d in decoded if d["weights"] == (2, 1, 0)
                   and any(_key(other["tile"]) == _key(d["tile"])
                           and other["weights"] == (3, 2, 1) for _, other in decoded))
    inp, answer = pair["input"], pair["output"]
    view = inp.T if d["transposed"] else inp
    # Permuting cells inside each stencil slice changes neither its mask nor query.
    changed = view.copy()
    r0, r1, c0, c1 = d["crop_box"]
    for col in range(c0, c1):
        positions = np.flatnonzero(changed[r0:r1, col] != d["background"])
        changed[r0 + positions, col] = changed[r0 + positions[::-1], col]
    _require(np.array_equal(solve(changed.T if d["transposed"] else changed), answer), "query positions accidentally matter")
    # Moving all marks within each rectangle also leaves every multiplicity intact.
    changed = view.copy()
    record_start = d["cuts"][0] + 1
    for obj in d["objects"]:
        x0, x1, y0, y1 = obj["box"]
        part = changed[x0:x1, record_start + y0:record_start + y1]
        part[:] = part[::-1, ::-1].copy()
    _require(np.array_equal(solve(changed.T if d["transposed"] else changed), answer), "record geometry accidentally matters")
    # Remove one whole matching record. Its area is irrelevant; every source
    # query it satisfies loses exactly ONE copy (two axial cells, including gap).
    obj = next(o for o in d["objects"] if any(o["matches"]))
    changed = view.copy()
    x0, x1, y0, y1 = obj["box"]
    changed[x0:x1, record_start + y0:record_start + y1] = d["background"]
    altered = solve(changed.T if d["transposed"] else changed)
    _require(altered.shape[int(not d["transposed"])] ==
             answer.shape[int(not d["transposed"])] - 2 * sum(obj["matches"]), "object counting failed")
    # Change only one query from the first recipe to the second recipe.
    # The teaching zero-profile scene has multiplicities 2,1,0.
    changed = view.copy()
    first, second = d["queries"][:2]
    old, new = next(iter(first - second)), next(iter(second - first))
    col = changed[r0:r1, c0]
    col[col == old] = new
    altered = solve(changed.T if d["transposed"] else changed)
    _require(not np.array_equal(altered, answer), "query identity has no causal effect")


def _boundary_probes():
    """Hand-constructed members beyond sampler templates, checked cell-for-cell."""
    active = np.array([0, 1, 2, 3, 0], dtype=np.int8)
    inactive = np.array([3, 4, 5, 0, 3], dtype=np.int8)
    tile = np.stack([active, inactive], axis=1)
    one = np.zeros((7, 9), dtype=np.int8)
    one[1:3, 1:5] = 8
    one[1, 1:4] = [1, 2, 3]
    two = np.zeros_like(one)
    two[1, 1:5] = [1, 2, 3, 8]
    two[3, 1:5] = [1, 2, 3, 8]
    three = two.copy()
    three[5, 1:5] = [1, 2, 3, 8]
    # one and two have exactly the same total occupied record area (8).
    _require(np.count_nonzero(one) == np.count_nonzero(two), "area probe setup")
    full = np.array([1, 1, 2, 3, 3], dtype=np.int8)
    cases = [(tile, one, 1, 1, 4), (tile, two, 2, 1, 4),
             (np.stack([active, inactive, active], axis=1), one, 2, 1, 4),
             (np.stack([full] * 5, axis=1), three, 15, 0, 5)]
    for source_tile, records, total, first_row, last_row in cases:
        source = np.pad(source_tile, 1)
        line = np.full((7, 1), 9, dtype=np.int8)
        primary = np.full((7, 1), 7, dtype=np.int8)
        secondary = np.zeros((7, 1), dtype=np.int8)
        inp = np.concatenate([source, line, records, line, primary, line, secondary], axis=1)
        expected = np.zeros((5, 2 * total - 1), dtype=np.int8)
        expected[first_row:last_row, ::2] = 7
        _require(np.array_equal(solve(inp), expected), "crop, zero-slice, equal-area, or budget boundary failed")
        _require(np.array_equal(solve(inp.T), expected.T), "transposed boundary failed")


def self_test():
    """42 streamed seeds; aggregates and SHA-256 only, no saved payloads."""
    signatures, templates, profiles, input_shapes, output_shapes = set(), set(), set(), set(), set()
    directions, record_counts, query_sets, swatches = set(), set(), set(), set()
    zero_slices = shared_records = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        _metamorphic(puzzle)
        if seed < 3:
            _require(_signature(generate_puzzle(seed)) == _signature(puzzle), "nondeterministic puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                tile = d["tile"] != d["background"]
                templates.add((tile.shape, tile.tobytes()))
                profiles.add(d["weights"])
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                directions.add(d["transposed"])
                record_counts.add(len(d["objects"]))
                query_sets.update(d["queries"])
                swatches.add((d["primary"], d["secondary"]))
                zero_slices += d["weights"].count(0)
                shared_records += sum(sum(o["matches"]) >= 2 for o in d["objects"])
                pairs += 1
    _require(len(signatures) == 42, "seeds collapsed")
    _require(len(templates) >= 100 and len(profiles) >= 35, "geometry or multiplicity variation collapsed")
    _require(len(input_shapes) >= 35 and len(output_shapes) >= 40, "shape variation collapsed")
    _require(directions == {False, True} and len(record_counts) >= 6, "layout variation collapsed")
    _require(len(query_sets) >= 50 and len(swatches) >= 35, "palette or recipe variation collapsed")
    _require(zero_slices >= 42 and shared_records >= 42, "missing coupled interactions")
    _boundary_probes()
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9201, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"].flat[0] = (int(bad["test"][0]["output"].flat[0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "binary_stencils": len(templates), "multiplicity_profiles": len(profiles),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "record_counts": sorted(record_counts), "query_sets": len(query_sets),
            "swatch_pairs": len(swatches), "zero_slices": zero_slices,
            "records_matching_multiple_slices": shared_records,
            "hand_constructed_boundary_checks": 8,
            "verification_scope": "generator-owned invariants and metamorphic checks"}
