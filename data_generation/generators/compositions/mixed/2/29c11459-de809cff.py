"""Recover rectangle cross-sections, bisect them, and annotate their erasures.

Only NumPy and the standard library are used. There is no file I/O, cached
scene, output lookup, or random choice in solve(). See task_spec.md for the
fixed horizontal convention, odd spans, and simultaneous overlay priority.
"""

from itertools import permutations
import hashlib

import numpy as np


SEPARATOR = 5
HIGHLIGHT = 8


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(a, color):
    """Four-connected observed components, including small noise components."""
    visited = a != color
    groups = []
    h, w = a.shape
    for r, c in np.argwhere(a == color):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        cells = []
        while stack:
            p, q = stack.pop()
            cells.append((p, q))
            for u, v in ((p - 1, q), (p + 1, q), (p, q - 1), (p, q + 1)):
                if 0 <= u < h and 0 <= v < w and not visited[u, v]:
                    visited[u, v] = True
                    stack.append((u, v))
        groups.append(np.asarray(cells, dtype=int))
    return groups


def _propose_boxes(a, background):
    # Three surviving cells on a side distinguish a true extent from a sparse
    # bump. Generation retains at least four clean visible cells on each side.
    boxes = []
    for color in np.unique(a):
        color = int(color)
        if color == background:
            continue
        for cells in _components(a, color):
            if len(cells) < 18:
                continue
            rows = np.flatnonzero(np.bincount(cells[:, 0], minlength=a.shape[0]) >= 3)
            cols = np.flatnonzero(np.bincount(cells[:, 1], minlength=a.shape[1]) >= 3)
            if len(rows) and len(cols):
                boxes.append((int(rows[0]), int(cols[0]), int(rows[-1]) + 1,
                              int(cols[-1]) + 1, color))
    return tuple(sorted(boxes))


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))


def _paint(shape, background, ordered):
    a = np.full(shape, background, dtype=np.int8)
    for r0, c0, r1, c1, color in ordered:
        a[r0:r1, c0:c1] = color
    return a


def _geometry_ok(boxes, shape):
    if not 2 <= len(boxes) <= 4 or len({b[4] for b in boxes}) != 2:
        return False
    degree = [0] * len(boxes)
    for i, b in enumerate(boxes):
        r0, c0, r1, c1, color = b
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            return False
        if min(r1 - r0, c1 - c0) < 5:
            return False
        for j, other in enumerate(boxes[:i]):
            area = _overlap(b, other)
            if area:
                if color == other[4] or area < 4:
                    return False
                degree[i] += 1
                degree[j] += 1
    # An alternating spatial path; painting order is not tied to that path.
    return sorted(degree) == [1, 1] + [2] * (len(boxes) - 2)


def _sections(clean, background):
    sections = []
    for row in range(clean.shape[0]):
        occupied = clean[row] != background
        edges = np.diff(np.pad(occupied.astype(np.int8), (1, 1)))
        for left, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            left, right = int(left), int(stop) - 1
            _require((right - left) % 2 == 0, "Even cross-section has no single midpoint")
            sections.append((row, left, right, (left + right) // 2,
                             int(clean[row, left]), int(clean[row, right])))
    return sections


def _scene_ok(clean, background, boxes):
    if not _geometry_ok(boxes, clean.shape):
        return False
    if np.count_nonzero(clean == background) < max(12, clean.size // 10):
        return False
    # Each visible rectangle remains connected and exposes its true extent.
    if _propose_boxes(clean, background) != tuple(sorted(boxes)):
        return False
    for r0, c0, r1, c1, color in boxes:
        v = clean[r0:r1, c0:c1] == color
        if v.sum() < max(22, .35 * v.size):
            return False
        if min(v[0].sum(), v[-1].sum(), v[:, 0].sum(), v[:, -1].sum()) < 4:
            return False
    try:
        sections = _sections(clean, background)
    except AssertionError:
        return False
    return any(a != b for _, _, _, _, a, b in sections)


def _recover(input_grid):
    """Fit input-only proposals and reject competing visible scenes.

    This is a bounded geometric fitter, not an exhaustive rectangle-cover
    search. Its evidence envelope is specified in task_spec.md.
    """
    inp = _grid(input_grid)
    colors = tuple(map(int, np.unique(inp)))
    _require(len(colors) == 3 and SEPARATOR not in colors and HIGHLIGHT not in colors,
             "Input must contain background and two fills, excluding 5 and 8")
    candidates = {}
    for background in colors:
        boxes = _propose_boxes(inp, background)
        if not _geometry_ok(boxes, inp.shape):
            continue
        for ordered in permutations(boxes):
            clean = _paint(inp.shape, background, ordered)
            cost = int(np.count_nonzero(inp != clean))
            if cost > max(8, .12 * inp.size) or not _scene_ok(clean, background, boxes):
                continue
            if any(np.count_nonzero((inp[r0:r1, c0:c1] == color) &
                                   (clean[r0:r1, c0:c1] == color)) <
                   .85 * np.count_nonzero(clean[r0:r1, c0:c1] == color)
                   for r0, c0, r1, c1, color in boxes):
                continue
            candidates[(background, clean.tobytes())] = (cost, clean, background, ordered)
    _require(candidates, "No well-supported rectangle reconstruction")
    ranked = sorted(candidates.values(), key=lambda x: (x[0], x[2], x[1].tobytes()))
    cost, clean, background, boxes = ranked[0]
    _require(all(x[0] > cost + 1 for x in ranked[1:]), "Ambiguous visible covering order")
    # Probe local alternative extents against the original observations. Even
    # boundaries are probed too, so parity cannot rescue weak boundary evidence.
    for i, box in enumerate(boxes):
        for side in range(4):
            for delta in (-1, 1):
                trial = list(boxes)
                changed = list(box)
                changed[side] += delta
                r0, c0, r1, c1, _ = changed
                if not (0 <= r0 < r1 <= inp.shape[0] and 0 <= c0 < c1 <= inp.shape[1]):
                    continue
                trial[i] = tuple(changed)
                other = _paint(inp.shape, background, trial)
                if not np.array_equal(other, clean):
                    _require(np.count_nonzero(inp != other) > cost + 1,
                             "Insufficient straight-boundary evidence")
    fills = tuple(c for c in colors if c != background)
    return clean, background, fills, tuple(boxes)


def _bisect(clean, background):
    out = np.full_like(clean, background)
    for row, left, right, mid, a, b in _sections(clean, background):
        out[row, left:mid] = a
        out[row, mid] = SEPARATOR
        out[row, mid + 1:right + 1] = b
    return out


def _neighbors(shape, point):
    row, col = point
    return [(r, c) for r in range(max(0, row - 1), min(shape[0], row + 2))
            for c in range(max(0, col - 1), min(shape[1], col + 2)) if (r, c) != point]


def _overlay(base, clean, centers, fills):
    demands = {}
    center_set = set(centers)
    for p in centers:
        opposite = fills[1] if int(clean[p]) == fills[0] else fills[0]
        for q in _neighbors(clean.shape, p):
            if q in center_set:
                continue
            _require(q not in demands or demands[q] == opposite, "Conflicting non-center rings")
            demands[q] = opposite
    out = base.copy()
    for q, color in demands.items():
        out[q] = color
    for p in centers:
        out[p] = HIGHLIGHT
    return out


def _analyze(inp):
    clean, background, fills, boxes = _recover(inp)
    centers = [tuple(map(int, p)) for p in np.argwhere((clean != background) & (inp == background))]
    _require(centers, "At least one erasure is required")
    base = _bisect(clean, background)
    expected = _overlay(base, clean, centers, fills)
    return {"clean": clean, "background": background, "fills": fills, "boxes": boxes,
            "centers": centers, "base": base, "expected": expected}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from one input, under fixed family conventions."""
    return _analyze(_grid(input_grid))["expected"]


def _sample_scene(rng, shape, count):
    """Free coordinate/size proposals; no parent whole-puzzle generator is used."""
    parity = int(rng.integers(2))
    boxes = []
    h, w = shape
    max_side = min(18 if count == 2 else 14 if count == 3 else 12, min(shape) - 1)
    widths = list(range(5, max_side + 1, 2))
    for _ in range(220):
        rh = int(rng.integers(5, max_side + 1))
        rw = int(rng.choice(widths))
        starts = list(range(parity, w - rw + 1, 2))
        if not starts:
            continue
        r0 = int(rng.integers(h - rh + 1))
        c0 = int(rng.choice(starts))
        b = (r0, c0, r0 + rh, c0 + rw, 1 + len(boxes) % 2)
        if boxes:
            area = _overlap(b, boxes[-1])
            if not 4 <= area <= .48 * min(rh * rw, (boxes[-1][2] - boxes[-1][0]) *
                                           (boxes[-1][3] - boxes[-1][1])):
                continue
            if any(_overlap(b, other) for other in boxes[:-1]):
                continue
        boxes.append(b)
        if len(boxes) == count:
            ordered = [boxes[int(i)] for i in rng.permutation(count)]
            clean = _paint(shape, 0, ordered)
            return (clean, tuple(ordered)) if _scene_ok(clean, 0, ordered) else None
    return None


def _fallback_scene(rng, shape, count):
    # A guaranteed size-compatible alternating-offset chain. This is only a
    # bounded last resort, still varying margins, widths, order and reflection.
    h, w = shape
    rh, step = (7, 4) if count == 2 else (8, 5)
    total = rh + step * (count - 1)
    rstart = int(rng.integers(h - total + 1))
    rw = int(rng.choice([v for v in (9, 11) if v + 4 <= w]))
    left = int(rng.integers(w - rw - 4 + 1))
    boxes = [(rstart + step * i, left + 4 * (i % 2), rstart + step * i + rh,
              left + 4 * (i % 2) + rw, 1 + i % 2) for i in range(count)]
    if rng.integers(2):
        boxes.reverse()
    clean = _paint(shape, 0, boxes)
    _require(_scene_ok(clean, 0, boxes), "Fallback geometry failed")
    return clean, tuple(boxes)


def _centers(rng, clean, mode, target):
    base = _bisect(clean, 0)
    points = [tuple(map(int, p)) for p in np.argwhere(clean != 0)]
    rng.shuffle(points)
    selected = []

    def add(p):
        if p in selected or clean[p] == 0:
            return False
        if any(clean[p] != clean[q] and max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 2
               for q in selected):
            return False
        # Small compatible clusters; retain dense evidence for reconstruction.
        if sum(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in selected) >= 3:
            return False
        selected.append(p)
        return True

    if mode == 0:
        # A changed territory next to the midpoint distinguishes original-fill
        # rings from complements computed after the midpoint repainting.
        options = [p for p in points if base[p] not in (int(clean[p]), SEPARATOR)
                   and any(base[q] == SEPARATOR for q in _neighbors(clean.shape, p))]
        if not options or not add(options[0]):
            return None
    elif mode == 1:
        # Erase an actual row endpoint and an adjacent same-fill neighbor.
        ends = {(r, c) for r, l, t, _, _, _ in _sections(clean, 0) for c in (l, t)}
        options = [p for p in points if p in ends]
        found = False
        for p in options:
            near = [q for q in _neighbors(clean.shape, p) if clean[q] == clean[p]]
            rng.shuffle(near)
            if near:
                add(p)
                add(near[0])
                found = True
                break
        if not found:
            return None
    # Both complementary directions appear in each of the two demonstrations.
    if mode in (0, 1):
        for color in (1, 2):
            if any(clean[p] == color for p in selected):
                continue
            if not any(add(p) for p in points if clean[p] == color):
                return None
    for p in points:
        if len(selected) >= target:
            break
        if add(p) and rng.random() < .30:
            near = [q for q in _neighbors(clean.shape, p) if clean[q] == clean[p]]
            rng.shuffle(near)
            for q in near[:1]:
                if len(selected) < target:
                    add(q)
    return selected if len(selected) >= min(target, 3) else None


def _corrupt(rng, clean, centers, demonstrate_noise=False):
    inp = clean.copy()
    used = set(centers)
    for p in centers:
        inp[p] = 0
    # Sparse wrong-fill noise, including seam bumps. Never mark these as holes.
    candidates = [tuple(map(int, p)) for p in np.argwhere(clean != 0) if tuple(map(int, p)) not in used]
    rng.shuffle(candidates)
    seam = lambda p: any(clean[q] in (1, 2) and clean[q] != clean[p]
                         for q in _neighbors(clean.shape, p))
    if rng.random() < .65:
        candidates.sort(key=lambda p: not seam(p))
    wrong = []
    target = int(rng.integers(1 if demonstrate_noise else 0, 4))
    for p in candidates:
        if len(wrong) >= target:
            break
        if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 2 for q in wrong):
            continue
        inp[p] = 3 - int(clean[p])
        wrong.append(p)
    # Exterior speckles stay isolated from dense components. This leaves their
    # interpretation visible without making a protected corner a solver cue.
    outside = [tuple(map(int, p)) for p in np.argwhere(clean == 0)
               if all(clean[q] == 0 for q in _neighbors(clean.shape, tuple(map(int, p))))]
    rng.shuffle(outside)
    speckles = []
    int_target = int(rng.integers(1 if demonstrate_noise else 0, 5))
    for p in outside:
        if len(speckles) >= int_target:
            break
        if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in speckles):
            continue
        inp[p] = int(rng.integers(1, 3))
        speckles.append(p)
    return inp


def _make_pair(rng, count, mode, is_test, palette):
    minimum = 14 if count == 2 else 18 if count == 3 else 24
    maximum = 30 if is_test else 28
    shape = tuple(int(rng.integers(minimum, maximum + 1)) for _ in range(2))
    for attempt in range(128):
        scene = _sample_scene(rng, shape, count) if attempt < 112 else _fallback_scene(rng, shape, count)
        if scene is None:
            continue
        clean, boxes = scene
        target = int(rng.integers(5, 14) if is_test else rng.integers(3, 8))
        centers = _centers(rng, clean, mode, target)
        if centers is None:
            continue
        inp = _corrupt(rng, clean, centers, demonstrate_noise=(mode == 1))
        if mode == 1 and not np.any((clean == 0) & (inp != 0)):
            continue
        mapping = np.asarray(palette, dtype=np.int8)
        actual, actual_clean = mapping[inp], mapping[clean]
        try:
            a = _analyze(actual)
            if not np.array_equal(a["clean"], actual_clean):
                continue
            out = _overlay(_bisect(actual_clean, palette[0]), actual_clean, centers, tuple(palette[1:]))
            if not np.array_equal(a["expected"], out):
                continue
            witnesses = _witnesses(a)
            if not witnesses["a_changes"]:
                continue
            if mode == 0 and not (witnesses["both_fills"] and witnesses["changed_territory"]
                                  and witnesses["ring_over_separator"]):
                continue
            if mode == 1 and not (witnesses["adjacent"] and witnesses["erased_endpoint"]):
                continue
            return {"input": actual, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("Bounded construction failed to retain unambiguous evidence")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    available = [c for c in range(10) if c not in (SEPARATOR, HIGHLIGHT)]
    palette = tuple(map(int, rng.choice(available, 3, replace=False)))
    palette_mode = int(rng.integers(3))
    train_max = int(rng.integers(2, 4))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, total in (("train", nt), ("test", ne)):
        for index in range(total):
            count = (train_max + 1 if split == "test" else
                     2 if index == 0 else train_max if index == 1 else int(rng.integers(2, train_max + 1)))
            mode = index if split == "train" and index < 2 else 2
            local_palette = palette
            if palette_mode == 1:
                local_palette = (int(rng.choice([c for c in available if c not in palette[1:]])), *palette[1:])
            elif palette_mode == 2:
                local_palette = tuple(map(int, rng.choice(available, 3, replace=False)))
            for _ in range(8):
                pair = _make_pair(rng, count, mode, split == "test", local_palette)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("Could not produce distinct inputs")
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(a):
    clean, bg, centers, fills, base, out = (a[k] for k in
        ("clean", "background", "centers", "fills", "base", "expected"))
    ends = {(r, c) for r, l, t, _, _, _ in _sections(clean, bg) for c in (l, t)}
    ordinary = _overlay(clean, clean, centers, fills)
    return {"a_changes": not np.array_equal(out, ordinary),
            "b_changes": not np.array_equal(out, base),
            "both_fills": {int(clean[p]) for p in centers} == set(fills),
            "erased_endpoint": any(p in ends for p in centers),
            "adjacent": any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                            for i, p in enumerate(centers) for q in centers[:i]),
            "changed_territory": any(base[p] not in (int(clean[p]), SEPARATOR) for p in centers),
            "ring_over_separator": any(base[q] == SEPARATOR and q not in centers
                                       for p in centers for q in _neighbors(clean.shape, p)),
            "ring_into_background": any(clean[q] == bg for p in centers for q in _neighbors(clean.shape, p)),
            "clipped": any(p[0] in (0, clean.shape[0] - 1) or p[1] in (0, clean.shape[1] - 1)
                           for p in centers)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Illegal pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "Dimensions must be preserved")
            signature = _key(inp)
            _require(signature not in seen, "Duplicate input")
            seen.add(signature)
            a = _analyze(inp)
            _require(np.array_equal(out, a["expected"]), "Output does not follow input-inferred rule")
            _require(set(map(tuple, np.argwhere(out == HIGHLIGHT))) == set(a["centers"]),
                     "Highlights must mark exactly the original-background erasures")
            w = _witnesses(a)
            a["has_wrong_fill"] = bool(np.any((a["clean"] != a["background"]) &
                (inp != a["background"]) & (inp != a["clean"])))
            a["has_speckle"] = bool(np.any((a["clean"] == a["background"]) &
                (inp != a["background"])))
            _require(w["a_changes"] and w["b_changes"], "A parent contribution is invisible")
            # Row construction is checked before marker overlays. The overlays
            # may legitimately overwrite a midpoint or even an endpoint.
            for row, l, r, m, left, right in _sections(a["clean"], a["background"]):
                _require(m - l == r - m and a["base"][row, m] == SEPARATOR, "Unequal halves")
                _require(np.all(a["base"][row, l:m] == left) and
                         np.all(a["base"][row, m + 1:r + 1] == right), "Wrong endpoint ownership")
            analyses[split].append((a, w))
    training = [w for _, w in analyses["train"]]
    _require(any(w["both_fills"] and w["changed_territory"] and w["ring_over_separator"]
                 for w in training), "Training lacks interaction/complement evidence")
    _require(any(w["adjacent"] and w["erased_endpoint"] for w in training),
             "Training lacks center precedence and recovered endpoint evidence")
    _require(any(a["has_wrong_fill"] and a["has_speckle"] for a, _ in analyses["train"]),
             "Training does not distinguish ordinary noise from erasures")
    _require(min(len(a["boxes"]) for a, _ in analyses["test"]) >
             max(len(a["boxes"]) for a, _ in analyses["train"]), "Missing rectangle-count shift")


def self_test():
    """Bounded, in-memory 42-seed sweep plus targeted semantic rejection probes."""
    metrics = {"shapes": set(), "layouts": set(), "palettes": set(), "backgrounds": set(),
               "counts": set(), "puzzles": set(), "holes": set()}
    flags = {k: 0 for k in ("clipped", "adjacent", "changed_territory", "ring_over_separator",
                            "ring_into_background", "erased_endpoint")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                a = _analyze(inp)
                clean, bg, base, centers = (a[k] for k in ("clean", "background", "base", "centers"))
                metrics["shapes"].add(inp.shape)
                metrics["layouts"].add((inp.shape, tuple(sorted(b[:4] for b in a["boxes"]))))
                metrics["palettes"].add((bg, *a["fills"]))
                metrics["backgrounds"].add(bg)
                metrics["counts"].add(len(a["boxes"]))
                metrics["holes"].add(len(centers))
                digest.update(inp.tobytes())
                for k, v in _witnesses(a).items():
                    if k in flags:
                        flags[k] += int(v)
                # Direct per-cell distance implementation of midpoint filling,
                # separate from slicing in _bisect.
                check = np.full_like(clean, bg)
                for row, left, right, _, ca, cb in _sections(clean, bg):
                    for col in range(left, right + 1):
                        dl, dr = col - left, right - col
                        check[row, col] = ca if dl < dr else cb if dr < dl else SEPARATOR
                _require(np.array_equal(check, base), "Nearest-endpoint distance check failed")
                # A second ring implementation uses shifted boolean masks,
                # rather than the production composer's coordinate dictionary.
                mask_centers = np.zeros(clean.shape, dtype=bool)
                for p in centers:
                    mask_centers[p] = True
                masks = []
                for color in a["fills"]:
                    padded = np.pad(mask_centers & (clean == color), 1)
                    ring = np.zeros(clean.shape, dtype=bool)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr or dc:
                                ring |= padded[1 + dr:1 + dr + clean.shape[0],
                                               1 + dc:1 + dc + clean.shape[1]]
                    masks.append(ring)
                _require(not np.any(masks[0] & masks[1] & ~mask_centers), "Incompatible ring masks")
                check = base.copy()
                check[masks[0]] = a["fills"][1]
                check[masks[1]] = a["fills"][0]
                check[mask_centers] = HIGHLIGHT
                _require(np.array_equal(check, pair["output"]), "Shift-mask ring check failed")
                # Reverse iteration must not introduce a ring painting order.
                _require(np.array_equal(_overlay(base, clean, centers[::-1], a["fills"]), pair["output"]),
                         "Output depends on center iteration order")
        metrics["puzzles"].add(digest.digest())
    _require(len(metrics["puzzles"]) == 42, "Seed diversity failed")
    _require(len(metrics["shapes"]) >= 65 and len(metrics["layouts"]) >= 180, "Insufficient geometric diversity")
    _require(metrics["counts"] == {2, 3, 4} and len(metrics["backgrounds"]) == 8, "Role/count diversity failed")
    _require(len(metrics["palettes"]) >= 30 and len(metrics["holes"]) >= 8, "Palette/damage diversity failed")
    _require(all(flags.values()), "A documented interaction feature was never sampled")
    first = generate_puzzle(71, 2, 1)
    again = generate_puzzle(71, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s])
                 for k in ("input", "output")), "Not deterministic")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(103 + nt * 5 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "Explicit counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect output accepted")
    clean = np.ones((7, 9), dtype=np.int8)
    clean[:, 5:] = 2
    try:
        _overlay(clean, clean, [(3, 4), (3, 5)], (1, 2))
    except AssertionError:
        pass
    else:
        raise AssertionError("Conflicting ring demands accepted")
    try:
        _sections(np.ones((1, 6), dtype=np.int8), 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("Even cross-section accepted")
    # Common endpoint parity is only the sampler's sufficient construction,
    # not a hidden solver convention: this nested cross-section breaks it.
    boxes = ((0, 0, 8, 13, 1), (4, 3, 14, 10, 2), (11, 7, 20, 16, 1))
    clean = _paint((22, 20), 0, boxes)
    inp = clean.copy()
    centers = [(2, 2), (10, 4)]
    for p in centers:
        inp[p] = 0
    _require(np.array_equal(solve(inp), _overlay(_bisect(clean, 0), clean, centers, (1, 2))),
             "Solver depended on common start-coordinate parity")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(metrics["puzzles"]),
            "input_shapes": len(metrics["shapes"]), "rectangle_layouts": len(metrics["layouts"]),
            "palettes": len(metrics["palettes"]), "backgrounds": sorted(metrics["backgrounds"]),
            "rectangle_counts": sorted(metrics["counts"]), "hole_counts": sorted(metrics["holes"]),
            "witness_pairs": flags,
            "checks": ["input-only reconstruction", "equal-distance midpoint rule", "parent ablations",
                       "center-order independence", "all legal explicit counts", "illegal counts",
                       "label mutation rejection", "conflict rejection", "even-span rejection"]}

