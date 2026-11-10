"""Compact exact column copies, then highlight the unique largest empty rectangle.

The fixed color roles are background 0 and output-only highlight 6. No parent
generator, file, training example, or random state is consulted by solve().
Generation and self_test write no files and keep only one puzzle at a time.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


BACKGROUND = 0
HIGHLIGHT = 6
BLOCKERS = (1, 2, 3, 4, 5, 7, 8, 9)
CHECK_SEEDS = (0, 72944522, 482140221, 969995100, 183103957, 399997997,
               390462843, 957203428, 615707682, 708825251, 162093422,
               991355243, 798776237, 502912684, 311451262, 116558107,
               982862, 391342364, 998205116, 280864828, 948697521,
               905248289, 130246993, 179008336, 566827990, 322986586,
               875065783, 190840112, 163169520, 943055918, 642768003,
               208452775, 637573365, 77384663, 431139499, 804032739,
               857166354, 761961175, 681672988, 565234115, 387410823,
               658537390)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need 2..10 train, 1..4 test, and num_train > num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape):
        raise ValueError("grid must be two-dimensional with dimensions 1..30")
    if not np.issubdtype(grid.dtype, np.integer) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return grid.astype(np.int8, copy=False)


def _decode(grid):
    """Read exact whole columns and return their first-occurrence identities."""
    n, width = grid.shape
    if not (3 <= n <= 10 and width == 3 * n):
        raise ValueError("input must be an n by 3n horizontal strip, 3 <= n <= 10")
    if HIGHLIGHT in grid or BACKGROUND not in grid:
        raise ValueError("0 is required background; 6 is reserved for output")
    identities, indices, columns = {}, [], []
    for column in grid.T:
        key = tuple(int(c) for c in column)
        if key not in identities:
            identities[key] = len(columns)
            columns.append(column)
        indices.append(identities[key])
    if len(columns) != n or set(Counter(indices).values()) != {3}:
        raise ValueError("need exactly n distinct columns, each appearing three times")
    return np.stack(columns, axis=1).copy(), indices


def _runs(values):
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(values)


def _rectangles(mask):
    """Every row span's full empty runs; a maximum cannot be narrower."""
    height, width = mask.shape
    rectangles = []
    for top in range(height - 1):
        common = mask[top].copy()
        for bottom in range(top + 1, height):
            common &= mask[bottom]
            for left, right in _runs(common):
                if right - left >= 2:
                    h, w = bottom - top + 1, right - left
                    rectangles.append((h * w, top, left, h, w))
    return rectangles


def _best(mask):
    candidates = _rectangles(mask)
    area = max((rect[0] for rect in candidates), default=0)
    return area, [rect[1:] for rect in candidates if rect[0] == area]


def _unique_rectangle(mask):
    _, winners = _best(mask)
    if len(winners) != 1:
        raise ValueError("latent square must have a unique largest background rectangle >= 2x2")
    return winners[0]


def _mask(shape, rectangle):
    top, left, h, w = rectangle
    result = np.zeros(shape, dtype=bool)
    result[top:top + h, left:left + w] = True
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    latent, _ = _decode(grid)
    rectangle = _unique_rectangle(latent == BACKGROUND)
    output = latent.copy()
    output[_mask(latent.shape, rectangle)] = HIGHLIGHT
    return output


def _exhaustive_best(mask):
    """Validation enumerates all rectangles using summed areas, including subruns."""
    height, width = mask.shape
    sums = np.pad(mask.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, winners = 0, []
    for top in range(height - 1):
        for bottom in range(top + 2, height + 1):
            for left in range(width - 1):
                for right in range(left + 2, width + 1):
                    area = (bottom - top) * (right - left)
                    if area < best:
                        continue
                    count = sums[bottom, right] - sums[top, right] - sums[bottom, left] + sums[top, left]
                    if count == area:
                        rectangle = (top, left, bottom - top, right - left)
                        if area > best:
                            best, winners = area, [rectangle]
                        else:
                            winners.append(rectangle)
    return best, winners


def _features(latent, rectangle):
    empty = latent == BACKGROUND
    selected = _mask(latent.shape, rectangle)
    area = int(selected.sum())
    others = []
    overlap = False
    for candidate in _rectangles(empty):
        candidate_mask = _mask(latent.shape, candidate[1:])
        if np.any(candidate_mask & ~selected):
            others.append(candidate)
            overlap |= bool(np.any(candidate_mask & selected))
    remaining = empty & ~selected
    longest = max([b - a for line in list(remaining) + list(remaining.T)
                   for a, b in _runs(line)] + [0])
    start = (rectangle[0], rectangle[1])
    visited, stack = {start}, [start]
    while stack:
        row, col = stack.pop()
        for y, x in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (0 <= y < len(empty) and 0 <= x < len(empty[0])
                    and empty[y, x] and (y, x) not in visited):
                visited.add((y, x))
                stack.append((y, x))
    top, left, h, w = rectangle
    return {"area": area, "shape": (h, w), "position": (top, left),
            "overlap": overlap, "attached": len(visited) > area,
            "long_line": longest > area, "longest_unchanged_line": longest,
            "runner_up": max((rect[0] for rect in others), default=0),
            "wider_loser": any(rect[4] > w for rect in others),
            "taller_loser": any(rect[3] > h for rect in others),
            "interior": top > 0 and left > 0 and top + h < len(empty) and left + w < len(empty),
            "edge": top == 0 or left == 0 or top + h == len(empty) or left + w == len(empty)}


def _raw_search_differs(strip, indices, rectangle):
    """Compare semantic cells, ignoring the trivial raw-vs-compact shape change."""
    _, winners = _best(strip == BACKGROUND)
    if len(winners) != 1:
        return False
    top, left, h, w = winners[0]
    footprint = np.zeros((strip.shape[0], strip.shape[0]), dtype=bool)
    for identity in indices[left:left + w]:
        footprint[top:top + h, identity] = True
    return not np.array_equal(footprint, _mask(footprint.shape, rectangle))


def _order(rng, n, style):
    if style == "blocks":
        forward = list(range(n))
        return forward + (forward if rng.integers(2) else forward[::-1]) + (forward if rng.integers(2) else forward[::-1])
    if style == "grouped":
        return [i for i in range(n) for _ in range(3)]
    # Stable introduction order, with at least one repeat before the last new type.
    order = []
    for identity in range(n):
        order.append(identity)
        if identity < n - 1 and rng.random() < 0.42:
            order.append(identity)
    if len(order) == n:
        position = int(rng.integers(1, n))
        order.insert(position, int(rng.integers(position)))
    remaining = [i for i in range(n) for _ in range(3 - order.count(i))]
    rng.shuffle(remaining)
    order.extend(remaining)
    return order


def _plant(rng, n, profile):
    """Propose empty geometry; the entire finished mask decides acceptance."""
    if profile == "line":
        # A non-square winner in every training set rules out largest-square
        # selection, even when only two demonstrations were requested.
        h, w = ((2, 3), (3, 2))[int(rng.integers(2))]
    elif profile == "overlap":
        shapes = [(h, w) for h, w in ((3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4))
                  if h < n and w <= n]
        h, w = shapes[int(rng.integers(len(shapes)))]
    elif profile == "rival":
        h, w = ((3, 3), (2, 4), (4, 2), (3, 4), (4, 3))[int(rng.integers(5))]
    else:
        h = int(rng.integers(2, n))
        w = int(rng.integers(2, n + 1))
    top, left = int(rng.integers(n - h + 1)), int(rng.integers(n - w + 1))
    rectangle = (top, left, h, w)
    empty = _mask((n, n), rectangle)
    if profile == "overlap":
        extensions = []
        if top > 0:
            extensions.append((top - 1, left, 1, w - 1))
        if top + h < n:
            extensions.append((top + h, left + 1, 1, w - 1))
        if left > 0:
            extensions.append((top, left - 1, h - 1, 1))
        if left + w < n:
            extensions.append((top + 1, left + w, h - 1, 1))
        if not extensions:
            return None
        extension = extensions[int(rng.integers(len(extensions)))]
        empty |= _mask(empty.shape, extension)
    if profile == "line":
        length = int(rng.integers(h * w + 1, n + 1))
        if rng.integers(2):
            y, x = int(rng.integers(n)), int(rng.integers(n - length + 1))
            empty[y, x:x + length] = True
        else:
            y, x = int(rng.integers(n - length + 1)), int(rng.integers(n))
            empty[y:y + length, x] = True
    if profile == "rival":
        rivals = [(a, b) for a in range(2, n) for b in range(2, n + 1)
                  if max(4, 0.65 * h * w) <= a * b < h * w]
        if not rivals:
            return None
        a, b = rivals[int(rng.integers(len(rivals)))]
        y, x = int(rng.integers(n - a + 1)), int(rng.integers(n - b + 1))
        empty |= _mask(empty.shape, (y, x, a, b))
    if profile != "simple":
        empty |= rng.random((n, n)) < float(rng.uniform(0.02, 0.24))
    return empty, rectangle


def _make_pair(rng, n, profile, palette, style, require_causal, seen):
    # Bounded attempts, no logged rejects and no retained run payloads.
    for _ in range(1600):
        planted = _plant(rng, n, profile)
        if planted is None:
            continue
        empty, rectangle = planted
        _, winners = _best(empty)
        if winners != [rectangle]:
            continue
        latent = rng.choice(palette, size=(n, n)).astype(np.int8)
        latent[empty] = BACKGROUND
        # No synthetic identity stripe: distinctions come from actual cell texture.
        if len({tuple(int(c) for c in col) for col in latent.T}) != n:
            continue
        if require_causal and len({tuple(col) for col in empty.T}) == n:
            # Two differently colored vectors share an emptiness mask. This
            # makes exact colors necessary for A although irrelevant to B.
            continue
        features = _features(latent, rectangle)
        if profile == "overlap" and not (features["overlap"] and features["attached"]):
            continue
        if profile == "line" and not features["long_line"]:
            continue
        if profile == "rival" and features["runner_up"] < 0.65 * features["area"]:
            continue
        for _order_attempt in range(32 if require_causal else 1):
            indices = _order(rng, n, style)
            strip = latent[:, indices].copy()
            if require_causal and not _raw_search_differs(strip, indices, rectangle):
                continue
            key = (strip.shape, strip.tobytes())
            if key in seen:
                continue
            seen.add(key)
            output = latent.copy()
            output[_mask((n, n), rectangle)] = HIGHLIGHT
            return {"input": strip, "output": output}
    raise RuntimeError(f"bounded sampler exhausted for {n}/{profile}/{style}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Fixed 0/6 roles enable input-only solving; blocker identities and subsets vary.
    palette = rng.choice(BLOCKERS, size=int(rng.integers(2, 9)), replace=False)
    plans = [(int(rng.integers(6, 9)), "overlap", "interleaved", True),
             (int(rng.integers(7, 9)), "line", str(rng.choice(("interleaved", "grouped"))), False)]
    for _ in range(num_train - 2):
        n = int(rng.integers(3, 9))
        profile = str(rng.choice(("simple", "free", "overlap", "rival"))) if n >= 6 else str(rng.choice(("simple", "free")))
        style = str(rng.choice(("blocks", "grouped", "interleaved")))
        plans.append((n, profile, style, False))
    rng.shuffle(plans)
    seen = set()
    puzzle = {"train": [], "test": []}
    for n, profile, style, causal in plans:
        # Each pair can use a different subset; at least two blockers help keep
        # wide empty rectangles compatible with distinct full-height columns.
        active = rng.choice(palette, size=int(rng.integers(2, len(palette) + 1)), replace=False)
        puzzle["train"].append(_make_pair(rng, n, profile, active, style, causal, seen))
    for _ in range(num_test):
        n = int(rng.integers(9, 11))
        profile = str(rng.choice(("simple", "free", "overlap", "rival", "line")))
        active = rng.choice(palette, size=int(rng.integers(2, len(palette) + 1)), replace=False)
        puzzle["test"].append(_make_pair(rng, n, profile, active,
                                          str(rng.choice(("interleaved", "grouped", "blocks"))), False, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    strip, output = _grid(pair["input"]), _grid(pair["output"])
    latent, indices = _decode(strip)
    area, winners = _exhaustive_best(latent == BACKGROUND)
    _require(len(winners) == 1, "largest latent rectangle is absent or tied")
    rectangle = winners[0]
    _require(_best(latent == BACKGROUND) == (area, winners), "rectangle enumerators disagree")
    _require(output.shape == latent.shape, "output must be the latent square")
    selected = _mask(latent.shape, rectangle)
    _require(np.array_equal(output != latent, selected), "must change exactly the winning rectangle")
    _require(np.all(output[selected] == HIGHLIGHT), "incorrect highlight")
    _require(np.array_equal(output[~selected], latent[~selected]), "changed outside winner")
    _require(np.array_equal(solve(strip), output), "solve mismatch")
    features = _features(latent, rectangle)
    features.update({"n": len(latent), "early_repeat": not np.array_equal(strip[:, :len(latent)], latent),
                     "causal": _raw_search_differs(strip, indices, rectangle),
                     "color_identity": len({tuple(col) for col in (latent == BACKGROUND).T}) < len(latent),
                     "blockers": tuple(int(c) for c in np.unique(strip) if c != BACKGROUND),
                     "layout": tuple(indices)})
    return features


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split type")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            source = _grid(pair["input"])
            key = (source.shape, source.tobytes())
            _require(key not in seen, "repeated input")
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    _require(any(i["early_repeat"] and i["causal"] for i in train), "missing causal/crop training witness")
    _require(any(i["overlap"] and i["attached"] for i in train), "missing rectangle-vs-component witness")
    _require(any(i["long_line"] for i in train), "missing one-dimensional distractor witness")
    _require(any(i["shape"][0] != i["shape"][1] for i in train), "missing non-square winner")
    _require(any(i["color_identity"] for i in train), "missing full-color column identity witness")
    _require(all(i["n"] > max(j["n"] for j in train) for i in infos["test"]), "missing size shift")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes((len(puzzle[split]),)))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = _grid(pair[role])
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 fixed seeds, all count combinations, geometry and counterfactual checks.

    Returns a compact descriptive summary. This is generator-owned verification,
    not independent assessment of ARC inferability. Nothing is written to disk.
    """
    fingerprints, normalized, shapes, target_shapes, palettes, layouts = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprint = _digest(puzzle)
        _require(fingerprint == _digest(generate_puzzle(seed)), "nondeterminism")
        fingerprints.add(fingerprint)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                totals["pairs"] += 1
                for name in ("overlap", "attached", "long_line", "causal", "early_repeat", "color_identity", "wider_loser", "taller_loser", "interior", "edge"):
                    totals[name] += int(info[name])
                shapes.add(info["n"])
                target_shapes.add(info["shape"])
                palettes.add(info["blockers"])
                layouts.add((info["n"], info["layout"]))
                latent, _ = _decode(_grid(pair["input"]))
                # Background and highlight roles remain distinct from blocker names.
                roles = {BACKGROUND: 0, HIGHLIGHT: 1}
                normalized.add((latent.shape, tuple(roles.setdefault(int(c), len(roles)) for c in latent.flat)))
                _require(not np.array_equal(latent, pair["output"]), "omitting B must change answer")
    _require(len(fingerprints) == 42, "duplicate puzzles across seeds")
    _require(shapes == set(range(3, 11)), "latent sizes did not span 3..10")
    _require(len(target_shapes) >= 20 and len(palettes) >= 20 and len(layouts) >= 100,
             "geometry, palette, or column-order coverage too narrow")
    _require(len(normalized) >= 180, "too few distinct color-normalized latent patterns")
    for name in ("overlap", "attached", "long_line", "causal", "early_repeat", "color_identity", "wider_loser", "taller_loser", "interior", "edge"):
        _require(totals[name] >= 10, f"insufficient {name} coverage")
    legal_counts = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(20261023 + train * 10 + test, train, test)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "count mismatch")
            legal_counts += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicitly test that ties and an empty candidate set fail closed.
    tied = np.zeros((5, 5), dtype=bool)
    tied[:2, :2] = True
    tied[3:, 3:] = True
    for invalid in (tied, np.eye(5, dtype=bool)):
        try:
            _unique_rectangle(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous rectangle accepted")
    return {"seeds": len(CHECK_SEEDS), "distinct_puzzles": len(fingerprints),
            "legal_count_combinations": legal_counts, "latent_sizes": sorted(shapes),
            "target_shapes": sorted(target_shapes), "blocker_palettes": len(palettes),
            "column_layouts": len(layouts), "normalized_latent_patterns": len(normalized),
            "features": dict(totals), "semantic_verification": "generator-owned checks only"}
