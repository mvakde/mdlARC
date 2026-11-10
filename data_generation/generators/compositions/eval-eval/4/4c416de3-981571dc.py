"""Open corner-shaped apertures in shutters over a doubly reflected tapestry.

Shape comes from a maximal visible probe stencil; color comes from the global
tapestry orbit. No parent module, saved sample, seed or label enters solve().
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np


DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
MAX_OFFSET = 3
MAX_RADIUS = 4


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, comp = [start], {start}
        while todo:
            y, x = todo.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    todo.append(q)
        result.append(comp)
    return result


def _connected(pattern):
    seen = {min(pattern)}
    todo = list(seen)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                q = (u + du, v + dv)
                if q in pattern and q not in seen:
                    seen.add(q)
                    todo.append(q)
    return seen == set(pattern)


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _corners(rect):
    t, l, b, r = rect
    return ((t, l), (t, r), (b, l), (b, r))


def _point(rect, q, offset):
    y, x = _corners(rect)[q]
    dy, dx = DIRS[q]
    u, v = offset
    return y + dy * u, x + dx * v


def _orbit(point, shape, axes):
    y, x = point
    h, w = shape
    a, b = axes
    rows, cols = {y}, {x}
    if 0 <= a - y < h:
        rows.add(a - y)
    if 0 <= b - x < w:
        cols.add(b - x)
    return tuple(sorted((r, c) for r in rows for c in cols))


def _all_orbits(shape, axes):
    seen, orbits = set(), []
    for y in range(shape[0]):
        for x in range(shape[1]):
            if (y, x) not in seen:
                orbit = _orbit((y, x), shape, axes)
                seen.update(orbit)
                orbits.append(orbit)
    return orbits


def _read_scene(inp):
    """Infer all three structural colors and shutters from intact ring geometry."""
    candidates = []
    for color in np.unique(inp):
        comps = _components(tuple(map(int, p)) for p in np.argwhere(inp == color))
        rects = []
        for comp in comps:
            ys, xs = zip(*comp)
            rect = min(ys), min(xs), max(ys), max(xs)
            if rect[2] - rect[0] < 6 or rect[3] - rect[1] < 6 or comp != _outline(rect):
                break
            rects.append(rect)
        else:
            interior = [int(inp[y, x]) for t, l, b, r in rects
                        for y in range(t + 1, b) for x in range(l + 1, r)]
            counts = Counter(interior)
            if len(counts) != 2 or int(color) in counts:
                continue
            opaque, probe = [k for k, _ in counts.most_common()]
            if counts[opaque] <= counts[probe]:
                continue
            # Role colors do not occur as tapestry payload, anywhere.
            region = np.zeros(inp.shape, dtype=bool)
            for t, l, b, r in rects:
                region[t:b + 1, l:r + 1] = True
            if np.any(np.isin(inp[~region], (opaque, probe))):
                continue
            occurrences = []
            valid = True
            for fid, rect in enumerate(rects):
                t, l, b, r = rect
                local = {}
                for y, x in np.argwhere(inp[t + 1:b, l + 1:r] == probe):
                    y, x = int(y) + t + 1, int(x) + l + 1
                    distances = [(y - cy) ** 2 + (x - cx) ** 2 for cy, cx in _corners(rect)]
                    best = min(distances)
                    if distances.count(best) != 1:
                        valid = False
                        break
                    q = distances.index(best)
                    cy, cx = _corners(rect)[q]
                    dy, dx = DIRS[q]
                    offset = (y - cy) * dy, (x - cx) * dx
                    if not all(1 <= v <= MAX_RADIUS for v in offset):
                        valid = False
                        break
                    local.setdefault(q, set()).add(offset)
                occurrences.extend((fid, q, frozenset(p)) for q, p in sorted(local.items()))
            if valid and len(occurrences) >= 2:
                candidates.append({"rectangles": rects, "frame": int(color),
                                   "opaque": opaque, "probe": probe,
                                   "hidden": region, "occurrences": occurrences})
    if len(candidates) != 1:
        raise ValueError("Missing or ambiguous shutter/role interpretation")
    return candidates[0]


def _apertures(scene, rotation=False):
    occurrences = scene["occurrences"]
    normalized = [(fid, q, frozenset((v, u) for u, v in p)
                                    if rotation and q in (1, 2) else p)
                  for fid, q, p in occurrences]
    maximal = {p for _, _, p in normalized if all(s <= p for _, _, s in normalized)}
    if len(maximal) != 1:
        raise ValueError("No complete compatible stencil")
    stencil = maximal.pop()
    if not 3 <= len(stencil) <= 10 or not _connected(stencil):
        raise ValueError("Stencil must have 3..10 eight-connected cells")
    targets = set()
    for fid, q, _ in normalized:
        rect = scene["rectangles"][fid]
        local = {(v, u) for u, v in stencil} if rotation and q in (1, 2) else stencil
        for offset in local:
            point = _point(rect, q, offset)
            t, l, b, r = rect
            if not (t < point[0] < b and l < point[1] < r):
                raise ValueError("Aperture crosses shutter outline")
            distances = [(point[0] - cy) ** 2 + (point[1] - cx) ** 2
                         for cy, cx in _corners(rect)]
            if distances.count(min(distances)) != 1 or distances[q] != min(distances):
                raise ValueError("Completed stencil has ambiguous corner")
            if point in targets:
                raise ValueError("Apertures overlap")
            targets.add(point)
    return stencil, targets


def _infer_axes(inp, scene):
    visible = ~scene["hidden"]
    result = []
    for dimension in (0, 1):
        a = inp if dimension == 0 else inp.T
        v = visible if dimension == 0 else visible.T
        length, width = a.shape
        candidates = []
        # Family folds have at most three unmatched rows or columns at an edge.
        for fold in range(length - 1 - MAX_OFFSET, length + MAX_OFFSET):
            left = np.array([i for i in range(length) if i < fold - i < length], dtype=int)
            right = fold - left
            usable = v[left] & v[right]
            if np.count_nonzero(usable) < max(8, width):
                continue
            if np.all(a[left][usable] == a[right][usable]):
                candidates.append(fold)
        if len(candidates) != 1:
            raise ValueError("Surviving tapestry does not identify unique mirror folds")
        result.append(candidates[0])
    return tuple(result)


def _prediction(inp, rotation=False):
    scene = _read_scene(inp)
    stencil, targets = _apertures(scene, rotation)
    axes = _infer_axes(inp, scene)
    out = inp.copy()
    for point in targets:
        colors = {int(inp[p]) for p in _orbit(point, inp.shape, axes) if not scene["hidden"][p]}
        if len(colors) != 1:
            raise ValueError("Aperture has no unique surviving tapestry witness")
        out[point] = colors.pop()
    return out, scene, stencil, targets, axes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the queried grid; structural roles are inferred geometrically."""
    return _prediction(_grid(input_grid))[0]


def _sample_stencil(rng, radius, asymmetric):
    for _ in range(100):
        size = int(rng.integers(3, min(10, radius * radius) + 1))
        start = tuple(map(int, rng.integers(1, radius + 1, 2)))
        p = {start}
        while len(p) < size:
            frontier = sorted({(u + du, v + dv) for u, v in p
                               for du in (-1, 0, 1) for dv in (-1, 0, 1)
                               if 1 <= u + du <= radius and 1 <= v + dv <= radius} - p)
            p.add(frontier[int(rng.integers(len(frontier)))])
        if not asymmetric or p != {(v, u) for u, v in p}:
            return frozenset(p)
    # The asymmetric domain always contains this connected triangle.
    return frozenset({(1, 1), (1, 2), (2, 2)})


def _pack_shutters(rng, shape, radius, requested):
    h, w = shape
    rects = []
    for _ in range(requested):
        for _ in range(70):
            small = 2 * radius + 3
            hi_h, hi_w = min(h, small + 8), min(w, small + 8)
            sh, sw = int(rng.integers(small, hi_h + 1)), int(rng.integers(small, hi_w + 1))
            t, l = int(rng.integers(h - sh + 1)), int(rng.integers(w - sw + 1))
            box = (t, l, t + sh - 1, l + sw - 1)
            if all(box[2] + 2 <= q[0] or q[2] + 2 <= box[0]
                   or box[3] + 2 <= q[1] or q[3] + 2 <= box[1] for q in rects):
                rects.append(box)
                break
    return sorted(rects)


def _diagonal_only(point, shape, axes, hidden):
    y, x = point
    h, w = shape
    a, b = axes
    return bool(0 <= a - y < h and 0 <= b - x < w
            and hidden[a - y, x] and hidden[y, b - x]
            and not hidden[a - y, b - x])


def _cloth(rng, shape, axes, colors, style):
    orbits = _all_orbits(shape, axes)
    bh, bw = map(int, rng.integers(2, 5, 2))
    shift = int(rng.integers(len(colors)))
    out = np.empty(shape, dtype=np.int8)
    patch = rng.integers(len(colors), size=shape)
    for orbit in orbits:
        y, x = orbit[0]
        if style == 0:
            k = int(rng.integers(len(colors)))
        elif style == 1:
            k = (y // bh + x // bw + shift) % len(colors)
        elif style == 2:
            k = ((y + 2 * x) // bw + shift) % len(colors)
        elif style == 3:
            k = (max(abs(y - axes[0] // 2), abs(x - axes[1] // 2)) // bh + shift) % len(colors)
        else:
            k = int(patch[y // bh, x // bw])
        if rng.random() < .12:
            k = int(rng.integers(len(colors)))
        for point in orbit:
            out[point] = colors[k]
    return out


def _build_pair(rng, roles, pool, held_color, hard, witness, test):
    frame, opaque, probe = roles
    for attempt in range(500):
        radius = int(rng.choice((2, 2, 3) if hard else (2, 2, 3, 4)))
        lo = 22 if hard else max(14, 2 * radius + 7)
        shape = tuple(map(int, rng.integers(lo, 31, 2)))
        axes = tuple(n - 1 + int(rng.integers(-3, 4)) for n in shape)
        stencil = _sample_stencil(rng, radius, witness)
        requested = int(rng.integers(2, 5) if hard else rng.integers(1, 5))
        rects = _pack_shutters(rng, shape, radius, requested)
        hidden = np.zeros(shape, dtype=bool)
        for t, l, b, r in rects:
            hidden[t:b + 1, l:r + 1] = True
        viable, diagonal = [], []
        for fid, rect in enumerate(rects):
            for q in range(4):
                points = [_point(rect, q, s) for s in stencil]
                if all(any(not hidden[p] for p in _orbit(s, shape, axes)) for s in points):
                    viable.append((fid, q))
                    if any(_diagonal_only(p, shape, axes, hidden) for p in points):
                        diagonal.append((fid, q))
        if len(viable) < (3 if hard else 2) or (hard and not diagonal):
            continue
        adjacent = [(c, (c[0], c[1] ^ k)) for c in viable for k in (1, 2)
                    if (c[0], c[1] ^ k) in viable]
        if witness and not adjacent:
            continue
        rng.shuffle(viable)
        count = int(rng.integers(3 if hard else 2, min(9, len(viable)) + 1))
        selected = viable[:count]
        if hard:
            selected = list(dict.fromkeys([diagonal[int(rng.integers(len(diagonal)))]] + selected))
        if witness:
            ref, partial_witness = adjacent[int(rng.integers(len(adjacent)))]
            selected = list(dict.fromkeys([ref, partial_witness] + selected))
        else:
            ref = selected[int(rng.integers(len(selected)))]
        # At least one corner is intentionally left unopened.
        if len(selected) == 4 * len(rects):
            removable = [p for p in selected if p != ref and (not witness or p != partial_witness)
                         and (not hard or p not in diagonal or sum(c in diagonal for c in selected) > 1)]
            if not removable:
                continue
            selected.remove(removable[-1])
        palette_size = int(rng.integers(2, min(6, len(pool)) + 1))
        colors = list(map(int, rng.choice(pool, palette_size, replace=False)))
        if held_color is not None:
            if held_color in colors:
                k = colors.index(held_color)
                colors[0], colors[k] = colors[k], colors[0]
            else:
                colors[0] = held_color
        cloth = _cloth(rng, shape, axes, colors, int(rng.integers(5)))
        # Two matching local offsets must have different globally determined
        # values: copying the reference's recovered color map is insufficient.
        differing = [(s, c) for s in sorted(stencil) for c in selected if c != ref
                     and _orbit(_point(rects[ref[0]], ref[1], s), shape, axes)
                     != _orbit(_point(rects[c[0]], c[1], s), shape, axes)]
        if not differing:
            continue
        offset, other = differing[int(rng.integers(len(differing)))]
        for c, color in ((ref, colors[0]), (other, colors[1])):
            for point in _orbit(_point(rects[c[0]], c[1], offset), shape, axes):
                cloth[point] = color
        inp = cloth.copy()
        for rect in rects:
            t, l, b, r = rect
            inp[t:b + 1, l:r + 1] = opaque
            for p in _outline(rect):
                inp[p] = frame
        out = inp.copy()
        for fid, q in selected:
            for s in stencil:
                p = _point(rects[fid], q, s)
                out[p] = cloth[p]
            offsets = sorted(stencil)
            rng.shuffle(offsets)
            if (fid, q) == ref:
                keep = offsets
            elif witness and (fid, q) == partial_witness:
                keep = offsets[:len(offsets) - 1]
                distinctive = min(set(stencil) - {(v, u) for u, v in stencil})
                if distinctive not in keep:
                    keep[-1] = distinctive
            elif not test and rng.random() < .15:
                keep = offsets  # Multiple complete probes are allowed.
            else:
                lo_keep = 1 if test else max(1, len(stencil) // 2)
                hi_keep = max(1, len(stencil) // 2) if test else len(stencil) - 1
                keep = offsets[:int(rng.integers(lo_keep, hi_keep + 1))]
            for s in keep:
                inp[_point(rects[fid], q, s)] = probe
        try:
            pred, scene, recovered, targets, inferred = _prediction(inp)
            if not np.array_equal(pred, out) or recovered != stencil or inferred != axes:
                continue
            if not any(p != stencil for _, _, p in scene["occurrences"]):
                continue
            if witness:
                try:
                    alternative = _prediction(inp, True)[0]
                except ValueError:
                    alternative = None
                if alternative is not None and np.array_equal(alternative, out):
                    continue
        except ValueError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("Bounded shutter construction exhausted; no semantic fallback")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    held = int(rng.integers(10))
    available_roles = [c for c in range(10) if c != held]
    stable = bool(rng.random() < .55)
    shared_roles = tuple(map(int, rng.choice(available_roles, 3, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            roles = shared_roles if stable else tuple(map(int, rng.choice(available_roles, 3, replace=False)))
            pool = [c for c in range(10) if c not in roles and (split == "test" or c != held)]
            hard = split == "test" or i == 1 or bool(rng.random() < .3)
            for _ in range(20):
                pair = _build_pair(rng, roles, pool, held if split == "test" else None,
                                   hard, split == "train" and i == 0, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    # Witness obligations do not impose visible pair-index conventions.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    out, scene, stencil, targets, axes = _prediction(inp)
    diagonal = sum(_diagonal_only(p, inp.shape, axes, scene["hidden"]) for p in targets)
    reference = next((fid, q) for fid, q, p in scene["occurrences"] if p == stencil)
    ref_map = {s: int(out[_point(scene["rectangles"][reference[0]], reference[1], s)]) for s in stencil}
    copy_wrong = any(int(out[_point(scene["rectangles"][fid], q, s)]) != ref_map[s]
                     for fid, q, _ in scene["occurrences"] for s in stencil)
    return {"shape": inp.shape, "rectangles": len(scene["rectangles"]),
            "frame_shapes": [(b - t + 1, r - l + 1) for t, l, b, r in scene["rectangles"]],
            "stencil": tuple(sorted(stencil)), "stencil_size": len(stencil),
            "footprint": (max(u for u, _ in stencil) - min(u for u, _ in stencil) + 1,
                          max(v for _, v in stencil) - min(v for _, v in stencil) + 1),
            "offset_anchor": (1, 1) not in stencil,
            "symmetric": stencil == {(v, u) for u, v in stencil},
            "marked": len(scene["occurrences"]), "diagonal_only": diagonal,
            "axes": axes, "offsets": tuple(a - n + 1 for a, n in zip(axes, inp.shape)),
            "roles": (scene["frame"], scene["opaque"], scene["probe"]),
            "partial_counts": [len(p) for _, _, p in scene["occurrences"] if p != stencil],
            "references": sum(p == stencil for _, _, p in scene["occurrences"]),
            "copy_reference_wrong": copy_wrong,
            "targets": len(targets), "unopened": int(scene["hidden"].sum()) - len(targets)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, trained_colors = set(), set()
    train_diagonal = convention_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            pred, scene, stencil, targets, axes = _prediction(inp)
            assert np.array_equal(pred, out), "Input-only prediction disagrees"
            mask = np.zeros(inp.shape, dtype=bool)
            for p in targets:
                mask[p] = True
            assert np.array_equal(inp != out, mask), "Only complete aperture support may change"
            assert np.all(np.isin(inp[mask], (scene["opaque"], scene["probe"])))
            assert not np.any(np.isin(out[mask], tuple(scene[k] for k in ("frame", "opaque", "probe"))))
            assert len(np.unique(out[mask])) >= 2, "Aperture contents must be multicolor"
            assert np.array_equal(inp[~mask], out[~mask])
            assert any(p != stencil for _, _, p in scene["occurrences"]), "No partial-copy witness"
            assert len(scene["occurrences"]) < 4 * len(scene["rectangles"]), "No unmarked corner"
            assert np.any((inp == scene["opaque"]) & ~mask), "Opening all cover would be equivalent"
            f = _features(inp)
            assert f["copy_reference_wrong"], "Reference-local colors happen to solve every corner"
            # Check agreement of every visible orbit, including one-fold fringes.
            for orbit in _all_orbits(inp.shape, axes):
                visible = {int(inp[p]) for p in orbit if not scene["hidden"][p]}
                assert len(visible) <= 1
            if split == "train":
                train_diagonal |= f["diagonal_only"] > 0
                trained_colors.update(map(int, inp.flat))
                trained_colors.update(map(int, out.flat))
                try:
                    alternative = _prediction(inp, True)[0]
                except ValueError:
                    convention_witness = True
                else:
                    convention_witness |= not np.array_equal(alternative, out)
            else:
                assert f["diagonal_only"] > 0, "Test must require both folds"
                assert all(n <= len(stencil) // 2 for n in f["partial_counts"])
                assert set(map(int, out[mask])) - trained_colors, "No visible-witness novel-color shift"
    assert train_diagonal, "Training must teach both folds"
    assert convention_witness, "Training does not distinguish reflection from rotation"


def _signature(puzzle):
    return tuple(_key(np.asarray(p[k])) for split in ("train", "test")
                 for p in puzzle[split] for k in ("input", "output"))


def self_test():
    """Bounded 42-seed study, count checks, and semantic mutation regressions."""
    signatures, features = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = _grid(pair["input"])
                original = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, original), "Solver mutated its input"
                features.append(_features(inp))
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len({f["shape"] for f in features}) >= 45
    assert len({f["stencil"] for f in features}) >= 35
    assert {f["rectangles"] for f in features} >= {1, 2, 3}
    assert len({s for f in features for s in f["frame_shapes"]}) >= 30
    assert len({f["stencil_size"] for f in features}) >= 6
    assert any(f["offset_anchor"] for f in features)
    assert any(f["symmetric"] for f in features) and any(not f["symmetric"] for f in features)
    assert any(f["references"] > 1 for f in features)
    assert any(f["footprint"][0] != f["footprint"][1] for f in features)
    assert {a % 2 for f in features for a in f["axes"]} == {0, 1}
    assert len({f["offsets"] for f in features}) >= 25
    assert all(len({f["roles"][k] for f in features}) == 10 for k in range(3))
    assert any(30 in f["shape"] for f in features)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(901 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    sample = generate_puzzle(71, 2, 1)
    inp = sample["test"][0]["input"]
    _, scene, _, targets, _ = _prediction(inp)
    for point in (min(targets), tuple(map(int, np.argwhere(~scene["hidden"])[0]))):
        bad = {s: [{k: np.asarray(a).copy() for k, a in p.items()} for p in sample[s]] for s in sample}
        bad["test"][0]["output"][point] = (int(bad["test"][0]["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted output accepted")
    # A rotated scene stays within the family and is solved from its own pixels.
    for turns in (1, 2, 3):
        assert np.array_equal(solve(np.rot90(inp, turns)), np.rot90(sample["test"][0]["output"], turns))
    return {"seeds": 42, "pairs": len(features), "shapes": len({f["shape"] for f in features}),
            "stencils": len({f["stencil"] for f in features}),
            "frame_counts": sorted({f["rectangles"] for f in features}),
            "diagonal_pairs": sum(f["diagonal_only"] > 0 for f in features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
