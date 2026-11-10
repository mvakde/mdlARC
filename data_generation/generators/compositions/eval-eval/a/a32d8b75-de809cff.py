"""Occluded mirror loom: rectangle defects become a reflected tile stencil.

All answer conventions are visible or fixed by the family. No parent modules,
example tables, files, or generation state are used by solve.
"""
from itertools import permutations

import numpy as np


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = remaining.pop()
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _bounds(cells):
    a = np.asarray(cells)
    lo, hi = a.min(axis=0), a.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _area(a, b):
    return (max(0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0, min(a[3], b[3]) - max(a[1], b[1])))


def _paint(shape, background, boxes):
    clean = np.full(shape, background, dtype=int)
    owner = np.full(shape, -1, dtype=int)
    for i, (r0, c0, r1, c1, color) in enumerate(boxes):
        clean[r0:r1, c0:c1] = color
        owner[r0:r1, c0:c1] = i
    return clean, owner


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(int, copy=True)


def _decode(input_grid):
    a = _grid(input_grid)
    background = int(a[0, 0])
    assert all(np.all(edge == background)
               for edge in (a[0], a[-1], a[:, 0], a[:, -1])), "missing outer margin"
    candidates = []
    # A frame color consists exactly of three separate rectangular perimeters.
    # Checking all occurrences prevents fill patterns from posing as a frame.
    for color in np.unique(a):
        if color == background:
            continue
        groups = _components(a == color)
        if len(groups) != 3:
            continue
        fields, valid = [], True
        for cells in groups:
            r0, c0, r1, c1 = _bounds(cells)
            h, w = r1 - r0, c1 - c0
            box = a[r0:r1, c0:c1]
            if (min(h, w) < 3 or len(cells) != 2 * h + 2 * w - 4
                    or not all(np.all(edge == color)
                               for edge in (box[0], box[-1], box[:, 0], box[:, -1]))):
                valid = False
                break
            fields.append((box[1:-1, 1:-1].copy(), (r0, c0, r1, c1)))
        if not valid:
            continue
        swatches = [f for f in fields if f[0].shape == (1, 1)]
        tiles = [f for f in fields if f[0].shape[0] == f[0].shape[1]
                 and 2 <= f[0].shape[0] <= 4]
        scenes = [f for f in fields if min(f[0].shape) >= 5]
        if len(swatches) != 1 or len(tiles) != 1 or len(scenes) != 1:
            continue
        tile, tile_bounds = tiles[0]
        scene, scene_bounds = scenes[0]
        reference = int(swatches[0][0][0, 0])
        fills = tuple(map(int, np.unique(tile)))
        if (len(fills) != 2 or reference not in fills or background in fills
                or color in fills or not set(np.unique(scene)) <= {background, *fills}
                or not all(np.any(scene == f) for f in fills)):
            continue
        occupied = np.zeros(a.shape, bool)
        for _, (r0, c0, r1, c1) in fields:
            occupied[r0:r1, c0:c1] = True
        if not np.all(a[~occupied] == background):
            continue
        if any(d * len(tile) > 30 for d in scene.shape):
            continue
        candidates.append(dict(scene=scene, tile=tile, reference=reference,
                               background=background, frame=int(color), fills=fills,
                               tile_bounds=tile_bounds, scene_bounds=scene_bounds))
    assert len(candidates) == 1, "frames/roles are not uniquely readable"
    return candidates[0]


def _propose_boxes(scene, fills):
    boxes = []
    for color in fills:
        for cells in _components(scene == color):
            if len(cells) <= 2:  # isolated ordinary noise
                continue
            assert len(cells) >= 5, "unsupported medium-size noise component"
            b = (*_bounds(cells), color)
            assert min(b[2] - b[0], b[3] - b[1]) >= 3
            boxes.append(b)
    assert 2 <= len(boxes) <= 4 and {b[4] for b in boxes} == set(fills)
    assert all(not _area(a, b) for i, a in enumerate(boxes)
               for b in boxes[:i] if a[4] == b[4]), "same-fill boxes overlap"
    return sorted(boxes)


def _scene_evidence(observed, clean, owner, boxes, background):
    if np.count_nonzero(clean == background) < 5:
        return False
    if np.count_nonzero(observed != clean) > max(6, int(.18 * clean.size)):
        return False
    for i, b in enumerate(boxes):
        cells = np.argwhere(owner == i)
        if len(cells) < 7 or _bounds(cells) != b[:4]:
            return False
        if np.count_nonzero((owner == i) & (observed == b[4])) < .70 * len(cells):
            return False
    return True


def _recover(scene, background, fills):
    """Infer bounds from surviving large components; fit every covering order.

    The family preserves connected fill evidence and all four occupied extrema.
    Small wrong-fill components/speckles do not determine rectangle bounds.
    Equivalent hidden orders are allowed; different tied visible scenes are not.
    """
    boxes = _propose_boxes(scene, fills)
    candidates = {}
    for order in permutations(boxes):
        clean, owner = _paint(scene.shape, background, order)
        if not _scene_evidence(scene, clean, owner, order, background):
            continue
        cost = int(np.count_nonzero(scene != clean))
        key = clean.tobytes()
        if key not in candidates:
            candidates[key] = (cost, clean, owner, order)
    assert candidates, "no supported clean rectangle scene"
    ranked = sorted(candidates.values(), key=lambda v: v[0])
    best = ranked[0]
    assert all(c[0] >= best[0] + 2 for c in ranked[1:]), "ambiguous covering order"
    return best[1], best[2], best[3]


def _targets(clean, owner, boxes, holes, behavior="normal"):
    targets = []
    for r, c in holes:
        i = int(owner[r, c])
        assert i >= 0
        r0, c0, r1, c1, fill = boxes[i]
        if behavior == "no_reflection":
            y, x = r, c
        elif behavior == "canvas_reflection":
            y, x = clean.shape[0] - 1 - r, clean.shape[1] - 1 - c
        else:
            y, x = r0 + r1 - 1 - r, c0 + c1 - 1 - c
        if behavior == "ignore_occlusion" or owner[y, x] == i:
            targets.append((y, x, fill))
    return targets


def _forward(clean, owner, boxes, holes, tile, reference, behavior="normal"):
    """Construction and ablations: reflect centers and copy blocks explicitly."""
    n = len(tile)
    fills = tuple(map(int, np.unique(tile)))
    exchanged = fills[0] + fills[1] - tile
    if behavior == "no_exchange":
        exchanged = tile
    if behavior == "rotate_tile":
        exchanged = np.rot90(exchanged, 2)
    out = np.repeat(np.repeat(clean, n, axis=0), n, axis=1)
    for r, c, fill in _targets(clean, owner, boxes, holes, behavior):
        other = fills[0] + fills[1] - fill
        patch = np.where(exchanged == reference, fill, other)
        out[r * n:(r + 1) * n, c * n:(c + 1) * n] = patch
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only reconstruction with inverse lookup of the reflected stencil."""
    d = _decode(input_grid)
    scene, tile, bg = d["scene"], d["tile"], d["background"]
    clean, owner, boxes = _recover(scene, bg, d["fills"])
    holes = (scene == bg) & (owner >= 0)
    stencil = np.zeros(scene.shape, bool)
    # A destination looks across its own rectangle to find an observed erasure.
    # The source must have the same visible owner; otherwise it was occluded.
    for i, (r0, c0, r1, c1, _) in enumerate(boxes):
        reflected = holes[r0:r1, c0:c1][::-1, ::-1]
        reflected_owner = owner[r0:r1, c0:c1][::-1, ::-1]
        stencil[r0:r1, c0:c1] |= (reflected & (reflected_owner == i)
                                 & (owner[r0:r1, c0:c1] == i))
    n = len(tile)
    expanded = np.repeat(np.repeat(clean, n, 0), n, 1)
    active = np.repeat(np.repeat(stencil, n, 0), n, 1)
    # In the exchanged tile, original non-reference cells take the local fill.
    local = np.tile(tile != d["reference"], scene.shape)
    opposite = d["fills"][0] + d["fills"][1] - expanded
    expanded[active & ~local] = opposite[active & ~local]
    return expanded


def _sample_scene(rng, shape, count, needs_edge):
    h, w = shape
    for _ in range(600):
        boxes = []
        for k in range(count):
            for _ in range(80):
                rh = int(rng.integers(3, h))
                rw = int(rng.integers(3, w))
                r = int(rng.integers(h - rh + 1))
                c = int(rng.integers(w - rw + 1))
                b = (r, c, r + rh, c + rw, 1 + k % 2)
                if boxes:
                    overlap = _area(b, boxes[-1])
                    cap = min(rh * rw, _area(boxes[-1], boxes[-1]))
                    if not 1 <= overlap <= .45 * cap:
                        continue
                    if any(_area(b, old) for old in boxes[:-1]):
                        continue
                boxes.append(b)
                break
            else:
                break
        if len(boxes) != count:
            continue
        boxes = [boxes[int(i)] for i in rng.permutation(count)]
        clean, owner = _paint(shape, 0, boxes)
        if not _scene_evidence(clean, clean, owner, boxes, 0):
            continue
        if any(len(_components(owner == i)) != 1 for i in range(count)):
            continue
        kept, hidden, edges = [], [], []
        for r, c in np.argwhere(owner >= 0):
            r, c = int(r), int(c)
            i = int(owner[r, c])
            b = boxes[i]
            y, x = b[0] + b[2] - 1 - r, b[1] + b[3] - 1 - c
            if (r, c) == (y, x):
                continue
            if owner[y, x] == i:
                kept.append((r, c))
                if r in (0, h - 1) or c in (0, w - 1):
                    edges.append((r, c))
            else:
                hidden.append((r, c))
        if not hidden or {int(clean[p]) for p in kept} != {1, 2}:
            continue
        if needs_edge and not edges:
            continue
        return clean, owner, boxes, kept, hidden, edges
    raise RuntimeError("could not sample recoverable rectangle geometry")


def _evidence_preserved(observed, clean, owner, boxes):
    try:
        proposed = _propose_boxes(observed, (1, 2))
        return sorted(boxes) == proposed and _scene_evidence(observed, clean, owner, boxes, 0)
    except AssertionError:
        return False


def _corrupt(rng, clean, owner, boxes, kept, hidden, edges, needs_edge, clustered):
    observed = clean.copy()
    holes = []

    def add_hole(p):
        if p in holes:
            return False
        trial = observed.copy()
        trial[p] = 0
        if not _evidence_preserved(trial, clean, owner, boxes):
            return False
        observed[p] = 0
        holes.append(p)
        return True

    def add_from(pool):
        for j in rng.permutation(len(pool)):
            if add_hole(pool[int(j)]):
                return True
        return False

    # Each pair contains both a surviving and an occluded reflection. The
    # teaching pairs also show both contextual fill roles and boundary defects.
    if not add_from(hidden):
        return None
    for fill in (1, 2):
        if not add_from([p for p in kept if clean[p] == fill]):
            return None
    if needs_edge and not add_from(edges):
        return None
    if clustered:
        adjacent = [(r + dr, c + dc) for r, c in holes
                    for dr, dc in ((1, 0), (0, 1), (1, 1), (-1, 1))
                    if 0 <= r + dr < len(clean) and 0 <= c + dc < clean.shape[1]
                    and clean[r + dr, c + dc] == clean[r, c]
                    and (r + dr, c + dc) not in holes]
        if not add_from(adjacent):
            return None
    target = int(rng.integers(len(holes), max(len(holes) + 1, min(11, int((owner >= 0).sum() * .14) + 1))))
    pool = kept + hidden
    for j in rng.permutation(len(pool)):
        if len(holes) >= target:
            break
        add_hole(pool[int(j)])
    # A real wrong-fill defect with a surviving, unused reflected target makes
    # treating every mismatch as a stencil site observably wrong.
    destinations = {(r, c) for r, c, _ in _targets(clean, owner, boxes, holes)}
    flips = []
    for j in rng.permutation(len(kept)):
        p = kept[int(j)]
        if p in holes:
            continue
        b = boxes[int(owner[p])]
        dest = (b[0] + b[2] - 1 - p[0], b[1] + b[3] - 1 - p[1])
        if dest in destinations:
            continue
        trial = observed.copy()
        trial[p] = 3 - clean[p]
        if _evidence_preserved(trial, clean, owner, boxes):
            observed[p] = trial[p]
            flips.append(p)
            break
    if not flips:
        return None
    for _ in range(int(rng.integers(0, 3))):
        p = kept[int(rng.integers(len(kept)))]
        if p in holes or p in flips:
            continue
        trial = observed.copy()
        trial[p] = 3 - clean[p]
        if _evidence_preserved(trial, clean, owner, boxes):
            observed[p] = trial[p]
            flips.append(p)
    exterior = [tuple(map(int, p)) for p in np.argwhere(owner < 0)]
    speckles = []
    for j in rng.permutation(len(exterior)):
        p = exterior[int(j)]
        trial = observed.copy()
        trial[p] = int(rng.integers(1, 3))
        if _evidence_preserved(trial, clean, owner, boxes):
            observed[p] = trial[p]
            speckles.append(p)
            if len(speckles) >= int(rng.integers(1, 4)):
                break
    if not speckles:
        return None
    try:
        reconstructed, _, _ = _recover(observed, 0, (1, 2))
    except AssertionError:
        return None
    if not np.array_equal(reconstructed, clean):
        return None
    return observed, holes


def _make_tile(rng, n, fills, witness):
    for _ in range(60):
        style = int(rng.integers(4))
        rr, cc = np.indices((n, n))
        if style == 0:
            bits = rng.random((n, n)) < rng.uniform(.2, .8)
        elif style == 1:
            bits = rr <= cc
        elif style == 2:
            bits = (rr + cc) % 2 == 0
        else:
            bits = np.zeros((n, n), bool)
            bits[int(rng.integers(n)), int(rng.integers(n))] = True
        bits = np.rot90(bits, int(rng.integers(4)))
        if not bits.any() or bits.all():
            continue
        if witness and any(np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            continue
        return np.where(bits, fills[0], fills[1]).astype(int)
    raise RuntimeError("could not sample an informative binary tile")


def _frame(content, color):
    return np.pad(content, 1, constant_values=color)


def _board(rng, observed, tile, reference, background, frame):
    fields = [_frame(tile, frame), _frame(np.array([[reference]], int), frame)]
    if rng.integers(2):
        fields.reverse()
    gap = int(rng.integers(1, 4))
    header_width = sum(f.shape[1] for f in fields) + gap
    header_height = max(f.shape[0] for f in fields)
    scene_field = _frame(observed, frame)
    width = max(header_width, scene_field.shape[1]) + int(rng.integers(2, 6))
    top_gap, mid_gap, bottom_gap = map(int, rng.integers(1, 4, 3))
    height = top_gap + header_height + mid_gap + scene_field.shape[0] + bottom_gap
    board = np.full((height, width), background, int)
    left = int(rng.integers(1, width - header_width))
    for field in fields:
        row = top_gap + int(rng.integers(header_height - len(field) + 1))
        board[row:row + len(field), left:left + field.shape[1]] = field
        left += field.shape[1] + gap
    row = top_gap + header_height + mid_gap
    left = int(rng.integers(1, width - scene_field.shape[1]))
    board[row:row + len(scene_field), left:left + scene_field.shape[1]] = scene_field
    return board


def _make_pair(rng, n, shape, count, needs_edge, clustered, witness):
    for _ in range(100):
        clean, owner, boxes, kept, hidden, edges = _sample_scene(rng, shape, count, needs_edge)
        damaged = _corrupt(rng, clean, owner, boxes, kept, hidden, edges, needs_edge, clustered)
        if damaged is None:
            continue
        observed, holes = damaged
        palette = np.asarray(rng.choice(10, 4, replace=False), int)
        bg, frame, f1, f2 = map(int, palette)
        mapping = np.array([bg, f1, f2])
        actual_clean, actual_input = mapping[clean], mapping[observed]
        actual_boxes = [(*b[:4], int(mapping[b[4]])) for b in boxes]
        tile = _make_tile(rng, n, (f1, f2), witness)
        reference = int(rng.choice([f1, f2]))
        output = _forward(actual_clean, owner, actual_boxes, holes, tile, reference)
        if witness:
            alternatives = ("no_exchange", "no_reflection", "canvas_reflection",
                            "rotate_tile", "ignore_occlusion")
            if any(np.array_equal(output, _forward(actual_clean, owner, actual_boxes,
                                                   holes, tile, reference, v)) for v in alternatives):
                continue
        board = _board(rng, actual_input, tile, reference, bg, frame)
        k = int(rng.integers(4))
        pair = {"input": np.rot90(board, k).copy(), "output": np.rot90(output, k).copy()}
        try:
            prediction = solve(pair["input"])
        except AssertionError:
            continue
        if np.array_equal(prediction, pair["output"]):
            return pair
    raise RuntimeError(f"bounded informative pair construction failed: tile={n}, scene={shape}, "
                       f"rectangles={count}, edge={needs_edge}, cluster={clustered}")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            witness = split == "train" and i < 2
            if split == "train" and i == 0:
                n, low, high, count = 2, 9, 14, 2
            elif split == "train" and i == 1:
                n, low, high, count = 3, 8, 11, 3
            elif split == "test" and i == 0:
                n, low, high, count = 4, 7, 8, int(rng.choice([2, 3]))
            else:
                n = int(rng.choice([2, 3])) if split == "train" else int(rng.choice([3, 4]))
                low, high = {2: (9, 16), 3: (7, 11), 4: (6, 8)}[n]
                count = int(rng.choice([2, 3]))
            shape = (int(rng.integers(low, high)), int(rng.integers(low, high)))
            # Six-cell axes leave room for two evidenced rectangles, but not
            # the full three-rectangle/noise/cluster teaching combination.
            if n == 4 and min(shape) == 6:
                count = 2
            if n == 2 and min(shape) >= 12 and i >= 2 and rng.random() < .45:
                count = 4
            needs_edge = (split == "train" and i == 1) or split == "test"
            clustered = (split == "train" and i == 1) or rng.random() < .45
            if shape == (6, 6):
                clustered = False  # seven mandatory corruptions exceed the evidence budget
            for _ in range(8):
                pair = _make_pair(rng, n, shape, count, needs_edge, clustered, witness)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _analysis(pair):
    d = _decode(pair["input"])
    clean, owner, boxes = _recover(d["scene"], d["background"], d["fills"])
    holes = [tuple(map(int, p)) for p in np.argwhere((owner >= 0) & (d["scene"] == d["background"]))]
    targets = _targets(clean, owner, boxes, holes)
    wrong = [tuple(map(int, p)) for p in np.argwhere((owner >= 0)
             & (d["scene"] != clean) & (d["scene"] != d["background"]))]
    d.update(clean=clean, owner=owner, boxes=boxes, holes=holes, targets=targets, wrong=wrong)
    return d


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, teaching = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _analysis(pair)
            n = len(d["tile"])
            assert out.shape == tuple(n * s for s in d["scene"].shape)
            assert d["holes"] and d["targets"]
            assert len(d["targets"]) < len(d["holes"]), "no occlusion witness"
            assert d["wrong"], "missing ordinary-error witness"
            assert np.any((d["owner"] < 0) & (d["scene"] != d["background"]))
            assert len({(r, c) for r, c, _ in d["targets"]}) == len(d["targets"])
            assert np.array_equal(solve(inp), out)
            forward = _forward(d["clean"], d["owner"], d["boxes"], d["holes"], d["tile"], d["reference"])
            assert np.array_equal(forward, out), "forward/inverse disagreement"
            # All exterior macroblocks are clean background, even at speckles.
            reshaped = out.reshape(d["scene"].shape[0], n, d["scene"].shape[1], n).transpose(0, 2, 1, 3)
            assert np.all(reshaped[d["owner"] < 0] == d["background"])
            if split == "train":
                teaching.append((d, out))
    for v in ("no_exchange", "no_reflection", "canvas_reflection", "rotate_tile", "ignore_occlusion"):
        assert any(not np.array_equal(out, _forward(d["clean"], d["owner"], d["boxes"],
                   d["holes"], d["tile"], d["reference"], v)) for d, out in teaching), f"untaught distinction: {v}"
    assert any({f for _, _, f in d["targets"]} == set(d["fills"]) for d, _ in teaching)
    assert any(any(p[0] in (0, len(d["scene"]) - 1) or p[1] in (0, d["scene"].shape[1] - 1)
                   for p in d["holes"]) for d, _ in teaching)
    assert any(any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                   for i, p in enumerate(d["holes"]) for q in d["holes"][:i]) for d, _ in teaching)
    assert any(not np.array_equal(out, _forward(d["clean"], d["owner"], d["boxes"],
                   d["holes"] + d["wrong"], d["tile"], d["reference"])) for d, out in teaching), "all-errors shortcut survives"
    train_sizes = {len(d["tile"]) for d, _ in teaching}
    assert {2, 3} <= train_sizes and 4 not in train_sizes
    assert len(_decode(puzzle["test"][0]["input"])["tile"]) == 4


def self_test():
    """Bounded semantic probes, diversity across 42 seeds, and count contracts."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, signatures, patterns, geometries = [], set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _analysis(pair)
                features.append((split, len(d["tile"]), d["scene"].shape, len(d["boxes"]),
                                 len(d["holes"]), len(d["targets"]), d["background"], d["frame"],
                                 d["reference"], d["tile_bounds"][:2], pair["input"].shape))
                patterns.add((d["tile"].shape, (d["tile"] == d["reference"]).tobytes()))
                geometries.add((d["scene"].shape, (d["owner"] >= 0).tobytes(),
                                tuple(b[:4] for b in d["boxes"])))
    assert len(signatures) == 42
    assert {f[1] for f in features} == {2, 3, 4}
    assert {f[3] for f in features} == {2, 3, 4}
    assert len({f[2] for f in features}) >= 35
    assert len({f[-1] for f in features}) >= 45
    assert len(geometries) >= 150 and len(patterns) >= 30
    assert len({f[6] for f in features}) == 10 and len({f[7] for f in features}) == 10
    assert len({f[9] for f in features}) >= 20
    assert len({f[4] for f in features}) >= 5
    assert tuple(_key(p[k]) for s in ("train", "test") for p in puzzles[7][s]
                 for k in ("input", "output")) == tuple(_key(p[k]) for s in ("train", "test")
                 for p in generate_puzzle(7)[s] for k in ("input", "output"))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(31415 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Small hand-derived geometry: one rear hole disappears behind the front
    # rectangle; a second reflects to (2,1) and a front hole to (3,4).
    boxes = [(0, 0, 4, 4, 1), (2, 2, 6, 6, 2)]
    clean, owner = _paint((7, 7), 0, boxes)
    holes = [(0, 0), (1, 2), (4, 3)]
    tile = np.array([[1, 1], [1, 2]])
    expected = np.repeat(np.repeat(clean, 2, 0), 2, 1)
    expected[4:6, 2:4] = [[2, 2], [2, 1]]
    expected[6:8, 8:10] = [[1, 1], [1, 2]]
    assert np.array_equal(_forward(clean, owner, boxes, holes, tile, 1), expected)
    # Input-only solver commutes with rotations of the whole framed board.
    pair = puzzles[0]["train"][0]
    for k in (1, 2, 3):
        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (damaged["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds plus all legal count combinations")
