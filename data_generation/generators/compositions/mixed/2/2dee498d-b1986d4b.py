"""Stable whole-column compression reveals a rank-wise square inventory.

Only NumPy and the standard library are used. Generation never reads evidence
files; solve has no random or generation state. Diagnostics are not written here.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _counts(num_train, num_test):
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    return nt, nq


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _columns(a):
    return [tuple(int(x) for x in a[:, c]) for c in range(a.shape[1])]


def _deduplicate(a):
    seen, keep = set(), []
    for c, vector in enumerate(_columns(a)):
        if vector not in seen:
            seen.add(vector)
            keep.append(c)
    return a[:, keep].copy()


def _components(a, bg):
    """Maximal four-connected components, not every embedded subsquare."""
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    for y in range(h):
        for x in range(w):
            color = int(a[y, x])
            if seen[y, x] or color == bg:
                continue
            stack, cells = [(y, x)], []
            seen[y, x] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= rr < h and 0 <= cc < w and not seen[rr, cc]
                            and int(a[rr, cc]) == color):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            r0 = min(r for r, c in cells)
            r1 = max(r for r, c in cells) + 1
            c0 = min(c for r, c in cells)
            c1 = max(c for r, c in cells) + 1
            result.append((color, r0, c0, r1 - r0, c1 - c0, len(cells)))
    return result


def _inventory(a):
    """Read local roles from a visible panel, also usable for ablations."""
    hist = np.bincount(a.ravel(), minlength=10)
    bg = int(hist.argmax())
    if np.count_nonzero(hist == hist[bg]) != 1:
        raise ValueError("background must be the unique most frequent panel color")
    components = _components(a, bg)
    classes, boxes = {}, []
    for color, r, c, hh, ww, area in components:
        if hh == ww and hh >= 2 and area == hh * ww:
            if hh in classes and classes[hh][0] != color:
                raise ValueError("each size must have one color")
            old = classes.get(hh, (color, 0))
            classes[hh] = (color, old[1] + 1)
            boxes.append((hh, color, r, c))
    if not classes or len({v[0] for v in classes.values()}) != len(classes):
        raise ValueError("missing or ambiguous square classes")
    payload = {v[0] for v in classes.values()}
    rest = set(int(x) for x in np.unique(a)) - payload - {bg}
    if len(rest) != 1:
        raise ValueError("one nonsquare noise color is required")
    noise = rest.pop()
    color_side = {color: side for side, (color, count) in classes.items()}
    fragments = 0
    for color, r, c, hh, ww, area in components:
        if color == noise:
            if hh == ww and hh >= 2 and area == hh * ww:
                raise ValueError("noise forms a competing square")
        elif not (hh == ww and hh >= 2 and area == hh * ww):
            side = color_side[color]
            edge = r == 0 or c == 0 or r + hh == a.shape[0] or c + ww == a.shape[1]
            if not (edge and area == hh * ww and hh != ww and max(hh, ww) == side):
                raise ValueError("payload fragment must be a visibly cropped edge rectangle")
            fragments += 1
    return {"background": bg, "noise": noise, "classes": classes,
            "boxes": boxes, "fragments": fragments}


def _profile(facts):
    return tuple((s, n) for s, (_, n) in sorted(facts["classes"].items()))


def _render(facts, always_all=False):
    classes = facts["classes"]
    ranks = max(count for color, count in classes.values())
    active = [sorted((s for s, (_, count) in classes.items()
                      if always_all or count >= rank), reverse=True)
              for rank in range(1, ranks + 1)]
    width = sum(sides[0] + 1 for sides in active)
    result = np.full((max(classes) + 1, width), facts["background"], dtype=np.int8)
    cursor = 0
    for sides in active:
        for side in sides:
            result[:side, cursor:cursor + side] = classes[side][0]
        cursor += sides[0] + 1
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from exact input columns and visible objects."""
    source = _grid(input_grid)
    panel = _deduplicate(source)
    if panel.shape[0] != panel.shape[1]:
        raise ValueError("stable column deduplication must recover a square panel")
    result = _render(_inventory(panel))
    if not all(1 <= d <= 30 for d in result.shape):
        raise ValueError("inventory exceeds the ARC output size limit")
    return result


def _reserve(mask, r, c, hh, ww):
    n = mask.shape[0]
    mask[max(0, r - 1):min(n, r + hh + 1), max(0, c - 1):min(n, c + ww + 1)] = True


def _pack(n, bg, classes, rng, horizontal_margin):
    items = [(s, color) for s, color, count in classes for _ in range(count)]
    for attempt in range(18):
        a = np.full((n, n), bg, dtype=np.int8)
        reserved = np.zeros((n, n), dtype=bool)
        rng.shuffle(items)
        order = sorted(items, reverse=True)
        for side, color in order:
            positions = [(r, c) for r in range(n - side + 1)
                         for c in range(horizontal_margin, n - side + 1 - horizontal_margin)
                         if not reserved[r:r + side, c:c + side].any()]
            if not positions:
                break
            r, c = positions[int(rng.integers(len(positions)))]
            a[r:r + side, c:c + side] = color
            _reserve(reserved, r, c, side, side)
        else:
            return a, reserved
    return None


def _fragment(a, reserved, classes, rng):
    n = len(a)
    choices = []
    for side, color, count in classes:
        for depth in range(1, side):
            for p in range(n - side + 1):
                choices.extend(((0, p, depth, side, color), (n - depth, p, depth, side, color),
                                (p, 0, side, depth, color), (p, n - depth, side, depth, color)))
    rng.shuffle(choices)
    for r, c, hh, ww, color in choices:
        if not reserved[r:r + hh, c:c + ww].any():
            a[r:r + hh, c:c + ww] = color
            return True
    return False


def _noise_square(a, r, c, noise):
    n = len(a)
    return any(np.all(a[y:y + 2, x:x + 2] == noise)
               for y in (r - 1, r) for x in (c - 1, c)
               if 0 <= y < n - 1 and 0 <= x < n - 1)


def _mark_columns(base, bg, noise, rng):
    """Noise is ignored only AFTER vector identity has done its work."""
    n = len(base)
    for attempt in range(48):
        a = base.copy()
        available = np.argwhere(a == bg)
        rng.shuffle(available)
        target = int(rng.integers(max(n, n * n // 10), max(n, n * n // 5) + 1))
        placed = 0
        for rr, cc in available:
            r, c = int(rr), int(cc)
            a[r, c] = noise
            if _noise_square(a, r, c, noise):
                a[r, c] = bg
            else:
                placed += 1
            if placed == target:
                break
        if len(set(_columns(a))) == n:
            return a
    return None


def _indices(n, rng, style):
    if style == "triple_reflected":
        return list(range(n)) + list(range(n - 1, -1, -1)) + list(range(n))
    repeats = [2] * n
    # Unequal multiplicity is visible, never needed to infer the rule.
    extra = int(rng.integers(0, min(n, 30 - 2 * n) + 1))
    for j in rng.choice(n, size=extra, replace=False):
        repeats[int(j)] += 1
    if style == "blocks":
        result = list(range(n))
        for wave in range(1, max(repeats)):
            tail = [i for i in range(n) if repeats[i] > wave]
            if int(rng.integers(2)):
                tail.reverse()
            result.extend(tail)
        return result
    result = list(range(n))
    early = 1 if style == "early" else int(rng.integers(2, min(5, n - 1)))
    # Insert copies of already discovered vectors, including separated copies.
    selected = [int(x) for x in rng.choice(n - 2, size=early, replace=False)]
    for j in sorted(selected, reverse=True):
        insertion = int(rng.integers(j + 2, n))
        result.insert(insertion, j)
    tail = [j for j in range(n) for _ in range(repeats[j] - result.count(j))]
    if style == "early":
        tail.sort(reverse=True)
    else:
        rng.shuffle(tail)
    return result + tail


def _shortcut_differs(a, expected):
    """Wrong rule: crop the first n columns, then perform square nesting."""
    crop = a[:, :a.shape[0]]
    try:
        return not np.array_equal(_render(_inventory(crop)), expected)
    except ValueError:
        return True


def _make_pair(rng, role, allowed_sides=None, forbidden_profiles=()):
    for attempt in range(160):
        if role == "first":
            n = int(rng.integers(9, 11))
            sides, counts, style = (2, 3), (3, 2), "triple_reflected"
        elif role == "second":
            n = int(rng.integers(11, 14))
            sides = (2, 3, int(rng.choice((4, 5))))
            counts, style = (1, 2, 1), "early"
        else:
            n = int(rng.integers(8, 15))
            pool = tuple(allowed_sides or (2, 3, 4, 5))
            k = int(rng.integers(2, min(4, len(pool)) + 1))
            sides = tuple(sorted(int(x) for x in rng.choice(pool, size=k, replace=False)))
            counts = tuple(int(x) for x in rng.integers(1, 5, size=k))
            style = "test" if role == "test" else str(rng.choice(("early", "blocks", "test")))
        if max(counts) < 2 or tuple(zip(sides, counts)) in forbidden_profiles:
            continue
        if sum(s * s * count for s, count in zip(sides, counts)) > .43 * n * n:
            continue
        if sum(max(s for s, count in zip(sides, counts) if count >= rank) + 1
               for rank in range(1, max(counts) + 1)) > 30:
            continue
        palette = [int(x) for x in rng.permutation(10)]
        bg, noise = palette[:2]
        classes = [(s, color, count) for s, color, count in zip(sides, palette[2:], counts)]
        packed = _pack(n, bg, classes, rng, int(role == "first"))
        if packed is None:
            continue
        base, reserved = packed
        if role != "first" and rng.random() < .35:
            _fragment(base, reserved, classes, rng)
        panel = _mark_columns(base, bg, noise, rng)
        if panel is None:
            continue
        try:
            facts = _inventory(panel)
        except ValueError:
            continue
        intended = {s: (color, count) for s, color, count in classes}
        if facts["classes"] != intended or facts["background"] != bg or facts["noise"] != noise:
            continue
        a = panel[:, _indices(n, rng, style)].copy()
        expected = _render(facts)
        if role == "first":
            # Reflection joins two equal edge vectors. Reject accidental noise
            # squares across that seam so the no-dedup witness has exactly 3x
            # the payload inventory, with no competing noise interpretation.
            try:
                raw = _inventory(a)
            except ValueError:
                continue
            if raw["classes"] != {s: (color, 3 * count) for s, color, count in classes}:
                continue
        if style not in ("blocks", "triple_reflected") and not _shortcut_differs(a, expected):
            continue
        _require(np.array_equal(_deduplicate(a), panel), "construction lost canonical order")
        output = solve(a)
        _require(np.array_equal(output, expected), "visible inventory differs from intended inventory")
        return {"input": a, "output": output}
    raise RuntimeError(f"bounded construction attempts exhausted for {role}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen, profiles, vocabulary = set(), set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            role = ("first" if index == 0 else "second" if index == 1 else "other") if split == "train" else "test"
            for retry in range(8):
                pair = _make_pair(rng, role, vocabulary if split == "test" else None,
                                  profiles if split == "test" else ())
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    facts = _inventory(_deduplicate(pair["input"]))
                    if split == "train":
                        profiles.add(_profile(facts))
                        vocabulary.update(facts["classes"])
                    break
            else:
                raise RuntimeError("bounded distinct-input retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(source) not in seen, "duplicate input")
            seen.add(_key(source))
            panel = _deduplicate(source)
            n = len(panel)
            _require(8 <= n <= 14 and panel.shape == (n, n), "canonical square dimensions")
            vectors = _columns(source)
            _require(all(2 <= vectors.count(v) <= 3 for v in set(vectors)), "column multiplicities")
            facts = _inventory(panel)
            _require(2 <= len(facts["classes"]) <= 4, "size class count")
            _require(set(facts["classes"]) <= {2, 3, 4, 5}, "square sides")
            _require(all(1 <= count <= 4 for color, count in facts["classes"].values()), "inventory counts")
            _require(np.array_equal(target, solve(source)), "output mismatch")
            _require(np.all(target[-1] == facts["background"]), "bottom padding")
            cursor = 0
            for rank in range(1, max(count for color, count in facts["classes"].values()) + 1):
                side = max(s for s, (color, count) in facts["classes"].items() if count >= rank)
                _require(np.all(target[:, cursor + side] == facts["background"]), "rank separator")
                cursor += side + 1
            _require(cursor == target.shape[1], "rank width")
            (training if split == "train" else testing).append((facts, source, target))
    first, a, target = training[0]
    n = len(a)
    panel = _deduplicate(a)
    _require(_profile(first) == ((2, 3), (3, 2)), "first training inventory")
    _require(np.array_equal(a, np.concatenate((panel, panel[:, ::-1], panel), axis=1)), "reflected witness")
    raw = _inventory(a)
    _require(all(raw["classes"][s] == (color, 3 * count)
                 for s, (color, count) in first["classes"].items()), "dedup ablation must triple counts")
    _require(not np.array_equal(_render(raw), target), "A concept has no effect")
    second, a, target = training[1]
    _require(tuple(count for s, count in _profile(second)) == (1, 2, 1), "second training inventory")
    _require(_shortcut_differs(a, target), "crop shortcut survives training")
    adjacent = a[:, [j for j in range(a.shape[1]) if j == 0 or not np.array_equal(a[:, j], a[:, j - 1])]]
    _require(adjacent.shape[1] > len(a), "adjacent dedup shortcut survives")
    for facts, a, target in training[:2]:
        _require(not np.array_equal(_render(facts, always_all=True), target), "B threshold has no effect")
    profiles = {_profile(facts) for facts, a, target in training}
    vocabulary = set().union(*(set(facts["classes"]) for facts, a, target in training))
    for facts, a, target in testing:
        _require(_profile(facts) not in profiles, "test inventory must recombine counts")
        _require(set(facts["classes"]) <= vocabulary, "test uses an undemonstrated size")
        _require(_shortcut_differs(a, target), "test should generalize early interleaving")


def _fingerprint(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test():
    """Bounded in-memory study; return small reproducible coverage statistics."""
    shapes, profiles, sides, classes, roles, layouts, structures = (set() for _ in range(7))
    fingerprints = set()
    fragments = edges = unequal = crop_failures = tied = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        if seed in (0, 17, 41):
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, target = pair["input"], pair["output"]
                panel = _deduplicate(a)
                facts = _inventory(panel)
                shapes.add(a.shape)
                profiles.add(_profile(facts))
                sides.update(facts["classes"])
                classes.add(len(facts["classes"]))
                roles.add((facts["background"], facts["noise"], tuple(sorted(facts["classes"].items()))))
                layouts.add(tuple(sorted(facts["boxes"])))
                fragments += facts["fragments"] > 0
                edges += any(r == 0 or c == 0 or r + side == len(panel) or c + side == len(panel)
                             for side, color, r, c in facts["boxes"])
                counts = [count for color, count in facts["classes"].values()]
                tied += len(set(counts)) < len(counts)
                vc = _columns(a)
                unequal += len({vc.count(v) for v in set(vc)}) > 1
                crop_failures += _shortcut_differs(a, target)
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)
                structures.add((a.shape, normalized))
                # Permuting literal colors must commute with the whole rule.
                permutation = np.array([9, 3, 7, 0, 5, 2, 8, 1, 6, 4], dtype=np.int8)
                _require(np.array_equal(solve(permutation[a]), permutation[target]), "color equivariance")
                # Labels must be invariant to redundancy placement when first order is fixed.
                alternate = np.concatenate((panel, panel[:, ::-1]), axis=1)
                _require(np.array_equal(solve(alternate), target), "redundancy affected inventory")
                # A half turn preserves distinct columns. A quarter turn can
                # turn legitimate repeated rows into duplicate columns, which
                # is outside this family's input encoding contract.
                rotated = np.rot90(panel, 2).copy()
                rotated_strip = np.concatenate((rotated, rotated[:, ::-1]), axis=1)
                _require(np.array_equal(solve(rotated_strip), target), "object positions affected inventory")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(17000 + 10 * nt + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (4.0, 1), (4, True), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # A direct three-class witness checks overwrite direction and exhaustion.
    facts = {"background": 0, "classes": {2: (4, 1), 3: (7, 2), 5: (9, 1)}}
    expected = np.zeros((6, 10), dtype=np.int8)
    expected[:5, :5] = 9
    expected[:3, :3] = 7
    expected[:2, :2] = 4
    expected[:3, 6:9] = 7
    _require(np.array_equal(_render(facts), expected), "rank overwrite or padding regression")
    # Direct boundary fixture: count complete edge squares, ignore the 1x5
    # edge fragment, and do not count sub-squares within the 5x5 component.
    base = np.zeros((12, 12), dtype=np.int8)
    base[:2, :2] = 4
    base[2:5, 5:8] = 7
    base[8:11, 2:5] = 7
    base[7:12, 7:12] = 9
    base[0:1, 7:12] = 9
    fixture = _mark_columns(base, 0, 1, np.random.default_rng(831))
    _require(fixture is not None, "boundary fixture markers")
    read = _inventory(fixture)
    _require(read["classes"] == facts["classes"] and read["fragments"] == 1, "edge inventory regression")
    _require(np.array_equal(solve(np.concatenate((fixture, fixture), axis=1)), expected),
             "boundary fixture solution")
    _require(len(fingerprints) == 42 and len(shapes) >= 15 and len(profiles) >= 25, "variation too narrow")
    _require(sides == {2, 3, 4, 5} and classes == {2, 3, 4}, "size coverage")
    _require(len(layouts) >= 150 and len(structures) >= 200 and len(roles) >= 100, "structural coverage")
    _require(fragments >= 10 and edges >= 40 and unequal >= 40 and crop_failures >= 100 and tied >= 40,
             "family feature coverage")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "inventory_profiles": len(profiles),
            "size_classes": sorted(classes), "side_lengths": sorted(sides),
            "layouts": len(layouts), "color_normalized_inputs": len(structures),
            "fragment_pairs": fragments, "edge_square_pairs": edges,
            "unequal_column_multiplicity_pairs": unequal, "crop_defeating_pairs": crop_failures,
            "tied_count_pairs": tied, "independent_solver": False}
