"""Torn periodic fabric: register, mend, and corner-dye four fragments.

Only NumPy and the standard library are used. No files are read by this module.
The solver searches visible assemblies and periodic explanations jointly.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4, "num_test must be an integer in 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and bool(np.all((0 <= a) & (a <= 9))), "integer colors 0..9 required")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in unseen:
                    unseen.remove(point)
                    todo.append(point)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _parse(grid):
    a = _grid(grid)
    bg = int(a[0, 0])
    _require(all(bool(np.all(edge == bg)) for edge in (a[0], a[-1], a[:, 0], a[:, -1])), "a blank background border is required")
    comps = _components(a != bg)
    _require(len(comps) == 4, "exactly four separated fragments required")
    for x, y in itertools.combinations(comps, 2):
        _require(int(np.abs(x[:, None] - y[None, :]).max(axis=2).min()) >= 2, "fragments need a background-cell gap")
    colors, counts = np.unique(a[a != bg], return_counts=True)
    key_colors = colors[counts == 1]
    tones = colors[counts > 1]
    _require(len(key_colors) == 4 and len(tones) == 2, "four singleton key colors and two fabric tones required")
    points = np.argwhere(np.isin(a, key_colors))
    r, c = points.min(axis=0)
    _require(set(map(tuple, points)) == {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}, "key must be a 2 by 2 patch")
    anchors = [i for i, coords in enumerate(comps) if (coords == (r, c)).all(axis=1).any()]
    _require(len(anchors) == 1, "key must lie inside one fragment")
    anchor = anchors[0]
    comps = [comps[anchor]] + [v for i, v in enumerate(comps) if i != anchor]
    origin = comps[0].min(axis=0)
    base = int(a[tuple(origin)])
    _require(base in tones, "anchor's top-left bounding-box corner must show a fabric tone")
    ink = int(next(v for v in tones if v != base))
    pieces = []
    for coords in comps:
        low, high = coords.min(axis=0), coords.max(axis=0)+1
        box = np.full(tuple(high-low), -2, dtype=np.int8)
        values = a[coords[:, 0], coords[:, 1]]
        bits = np.where(values == base, 0, np.where(values == ink, 1, -1))
        local = coords-low
        box[local[:, 0], local[:, 1]] = bits
        pieces.append(box)
    _require(np.count_nonzero(pieces[0] == -1) == 4 and all(not (p == -1).any() for p in pieces[1:]), "all key cells must belong to the anchor")
    total = sum(len(x) for x in comps)
    n = int(np.sqrt(total))
    _require(8 <= n <= 14 and n*n == total, "fragment area must be an 8..14 square")
    return {"grid": a, "background": bg, "base": base, "ink": ink,
            "key": a[r:r+2, c:c+2].copy(), "pieces": pieces, "side": n}


def _placement(box, n, corner, rotation):
    view = np.rot90(box, rotation)
    h, w = view.shape
    if h > n or w > n:
        return None
    row = 0 if corner < 2 else n-h
    col = 0 if corner % 2 == 0 else n-w
    coords = np.argwhere(view != -2)
    rr, cc = coords[:, 0]+row, coords[:, 1]+col
    indices = rr*n+cc
    mask = sum(1 << int(i) for i in indices)
    corners = (0, n-1, n*(n-1), n*n-1)
    if [i for i, x in enumerate(corners) if mask & (1 << x)] != [corner]:
        return None
    return (mask, indices, view[coords[:, 0], coords[:, 1]], rotation, corner)


def _assemblies(parsed):
    """All corner packings; texture orientations are never deduplicated away."""
    n, pieces = parsed["side"], parsed["pieces"]
    anchor = _placement(pieces[0], n, 0, 0)
    if anchor is None:
        return
    tables = {}
    for i in range(1, 4):
        for corner in range(1, 4):
            candidates, seen = [], set()
            for turn in range(4):
                candidate = _placement(pieces[i], n, corner, turn)
                if candidate is not None:
                    sig = (candidate[0], candidate[2].tobytes())
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append(candidate)
            tables[i, corner] = candidates
    full = (1 << (n*n))-1

    def walk(corner, remaining, occupied, placed):
        if corner == 4:
            if occupied != full:
                return
            observed = np.empty(n*n, dtype=np.int8)
            owner = np.empty(n*n, dtype=np.int8)
            rotations = []
            for item in placed:
                observed[item[1]] = item[2]
                owner[item[1]] = item[4]
                rotations.append(item[3])
            yield observed.reshape(n, n), owner.reshape(n, n), tuple(rotations)
            return
        for piece in remaining:
            for candidate in tables[piece, corner]:
                if occupied & candidate[0] == 0:
                    yield from walk(corner+1, tuple(i for i in remaining if i != piece),
                                    occupied | candidate[0], placed+[candidate])

    yield from walk(1, (1, 2, 3), anchor[0], [anchor])


def _periodic_repairs(observed):
    """Strict-majority vector periods in either axis, with >=3 known votes."""
    n = len(observed)
    cap = max(1, n*n//16)
    answers = []
    for axis in (0, 1):  # 0 repeats columns; 1 repeats rows.
        view = observed if axis == 0 else observed.T
        for period in range(1, n//3+1):
            unit = np.empty((n, period), dtype=np.int8)
            cost, valid = 0, True
            for phase in range(period):
                group = view[:, phase::period]
                zeros, ones = (group == 0).sum(axis=1), (group == 1).sum(axis=1)
                if ((zeros+ones < 3) | (zeros == ones)).any():
                    valid = False
                    break
                cost += int(np.minimum(zeros, ones).sum())
                if cost > cap:
                    valid = False
                    break
                unit[:, phase] = ones > zeros
            if valid:
                repaired = unit[:, np.arange(n) % period]
                answers.append((cost, repaired if axis == 0 else repaired.T, axis, period))
    return answers


def _analyze(input_grid):
    parsed = _parse(input_grid)
    best, outputs, repairs = None, {}, {}
    geometry = set()
    assembly_count = 0
    for observed, owner, rotations in _assemblies(parsed):
        assembly_count += 1
        geometry.add(owner.tobytes())
        token = observed.tobytes()
        if token not in repairs:
            repairs[token] = _periodic_repairs(observed)
        for cost, clean, axis, period in repairs[token]:
            # The anchor corner visibly defines ground and is never an error.
            if clean[0, 0] != 0:
                continue
            if best is not None and cost > best:
                continue
            out = np.where(clean == 0, parsed["base"], parsed["key"].reshape(-1)[owner]).astype(np.int8)
            if best is None or cost < best:
                best, outputs = cost, {}
            outputs.setdefault(out.tobytes(), (out, clean.copy(), owner.copy(), observed.copy(), axis, period, rotations))
    _require(best is not None, "no supported periodic square assembly")
    _require(len(outputs) == 1, "minimum-repair assembly has multiple colored outputs")
    out, clean, owner, observed, axis, period, rotations = next(iter(outputs.values()))
    parsed.update(output=out, clean=clean, owner=owner, observed=observed,
                  axis=axis, period=period, errors=best, rotations=rotations,
                  geometry_owners=len(geometry), assemblies=assembly_count)
    return parsed


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique minimum-flip periodic assembly, dyed by corner key."""
    return _analyze(input_grid)["output"].copy()


def _band_partition(rng, n, style):
    """Two complementary pairs; the pair opposite the anchor can reverse.

    Its 180-degree reversal is a genuine alternative geometric packing.
    Profiles supply rectangles, steps, narrow spines, and deep interlocks.
    """
    split = int(rng.integers(3, n-2))
    owner = np.empty((n, n), dtype=np.int8)
    for lo, hi, left, right in ((0, split, 0, 1), (split, n, 2, 3)):
        cut = int(rng.integers(2, n-1))
        for r in range(lo, hi):
            if style == 1:
                cut = int(np.clip(cut+int(rng.integers(-1, 2)), 1, n-1))
            elif style == 2:
                cut = int(rng.integers(1, n))
            elif style == 3:
                cut = 1 + ((cut-1+int(rng.choice([-1, 1]))) % (n-1))
            owner[r, :cut], owner[r, cut:] = left, right
    owner = np.rot90(owner, int(rng.integers(4)))
    labels = [int(owner[r, c]) for r, c in ((0, 0), (0, n-1), (n-1, 0), (n-1, n-1))]
    return np.argsort(labels).astype(np.int8)[owner]


def _grown_partition(rng, n):
    """Connected corner growth gives cuts without a complete straight seam."""
    owner = np.full((n, n), -1, dtype=np.int8)
    corners = ((0, 0), (0, n-1), (n-1, 0), (n-1, n-1))
    frontier = []
    for label, (r, c) in enumerate(corners):
        owner[r, c] = label
        frontier.append((r, c))
    while frontier:
        index = int(rng.integers(len(frontier)))
        r, c = frontier[index]
        choices = [(rr, cc) for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                   if 0 <= rr < n and 0 <= cc < n and owner[rr,cc] == -1]
        if not choices:
            frontier.pop(index)
            continue
        rr, cc = choices[int(rng.integers(len(choices)))]
        owner[rr, cc] = owner[r, c]
        frontier.append((rr, cc))
    return owner


def _key_locations(owner):
    n = len(owner)
    return [(r, c) for r in range(n-1) for c in range(n-1)
            if (r, c) != (0, 0) and (owner[r:r+2, c:c+2] == 0).all()]


def _scatter(rng, boxes, bg):
    """Free placement first, then enumerate all bounded shelf groupings."""
    if rng.random() < 0.55:
        h, w = int(rng.integers(23, 31)), int(rng.integers(23, 31))
        blocked = np.zeros((h, w), dtype=bool)
        positions = [None]*4
        for i in rng.permutation(4):
            coords = np.argwhere(boxes[i] != -2)
            bh, bw = boxes[i].shape
            for _ in range(40):
                r, c = int(rng.integers(1, h-bh)), int(rng.integers(1, w-bw))
                rr, cc = coords[:, 0]+r, coords[:, 1]+c
                if blocked[rr, cc].any():
                    continue
                positions[i] = (r, c)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        blocked[rr+dr, cc+dc] = True
                break
            else:
                break
        if all(p is not None for p in positions):
            canvas = np.full((h, w), bg, dtype=np.int8)
            for box, (r, c) in zip(boxes, positions):
                view = canvas[r:r+box.shape[0], c:c+box.shape[1]]
                view[box != -2] = box[box != -2]
            return canvas
    layouts = []
    for order in itertools.permutations(range(4)):
        for cuts in range(8):
            groups, group = [], []
            for j, i in enumerate(order):
                group.append(i)
                if j == 3 or (cuts >> j) & 1:
                    groups.append(group)
                    group = []
            heights = [max(boxes[i].shape[0] for i in g) for g in groups]
            widths = [sum(boxes[i].shape[1] for i in g)+len(g)-1 for g in groups]
            h, w = sum(heights)+len(groups)+1, max(widths)+2
            if h <= 30 and w <= 30:
                layouts.append((groups, heights, widths, h, w))
    if not layouts:
        return None
    groups, heights, widths, h, w = layouts[int(rng.integers(len(layouts)))]
    # Independently vary blank margins without exceeding 30.
    extra_h, extra_w = int(rng.integers(min(4, 31-h))), int(rng.integers(min(4, 31-w)))
    canvas = np.full((h+extra_h, w+extra_w), bg, dtype=np.int8)
    r = 1 + int(rng.integers(extra_h+1))
    for group, height, width in zip(groups, heights, widths):
        c = int(rng.integers(1, canvas.shape[1]-width))
        for i in group:
            box = boxes[i]
            rr = r + int(rng.integers(height-box.shape[0]+1))
            view = canvas[rr:rr+box.shape[0], c:c+box.shape[1]]
            view[box != -2] = box[box != -2]
            c += box.shape[1]+1
        r += height+1
    return canvas


def _make_pair(rng, n, period, axis, style, require_ambiguity, turns):
    # Bounded rejection over fresh combinatorial cuts and binary units.
    for attempt in range(160):
        owner = (_grown_partition(rng, n) if style == 4 and attempt < 80
                 else _band_partition(rng, n, style % 4))
        locations = _key_locations(owner)
        if not locations or any(np.count_nonzero(owner == i) < 6 for i in range(4)):
            continue
        unit = rng.integers(0, 2, size=(n, period), dtype=np.int8)
        clean = unit[:, np.arange(n) % period]
        if axis:
            clean = clean.T.copy()
        clean ^= clean[0, 0]
        if any(np.count_nonzero((owner == i) & (clean == 1)) < 3 for i in range(4)):
            continue
        key_r, key_c = locations[int(rng.integers(len(locations)))]
        observed = clean.copy()
        observed[key_r:key_r+2, key_c:key_c+2] = -1
        # Each altered phase-coordinate group receives at most one flip.
        # Force both false additions and false deletions in every example.
        used_groups, flipped = set(), []
        error_count = int(rng.integers(2, 5))
        for bit in [0, 1] + [int(rng.integers(2)) for _ in range(error_count-2)]:
            candidates = []
            for r, c in np.argwhere(observed == bit):
                group = (int(c), int(r) % period) if axis else (int(r), int(c) % period)
                if (r, c) != (0, 0) and group not in used_groups:
                    candidates.append((int(r), int(c), group))
            if not candidates:
                break
            r, c, group = candidates[int(rng.integers(len(candidates)))]
            observed[r, c] ^= 1
            used_groups.add(group)
            flipped.append((r, c))
        if len(flipped) != error_count:
            continue
        palette = [int(x) for x in rng.choice(10, 7, replace=False)]
        bg, base, ink, *key_values = palette
        key = np.array(key_values, dtype=np.int8).reshape(2, 2)
        painted = np.where(observed == 0, base, ink).astype(np.int8)
        painted[key_r:key_r+2, key_c:key_c+2] = key
        boxes = []
        for i in range(4):
            coords = np.argwhere(owner == i)
            low, high = coords.min(axis=0), coords.max(axis=0)+1
            box = np.where(owner[low[0]:high[0], low[1]:high[1]] == i,
                           painted[low[0]:high[0], low[1]:high[1]], -2)
            boxes.append(np.rot90(box, turns[i]))
        canvas = _scatter(rng, boxes, bg)
        if canvas is None:
            continue
        expected = np.where(clean == 0, base, key.reshape(-1)[owner]).astype(np.int8)
        try:
            analysis = _analyze(canvas)
        except ValueError:
            continue
        if (analysis["errors"] != error_count or not np.array_equal(analysis["output"], expected)
                or analysis["period"] != period or analysis["axis"] != axis
                or require_ambiguity and analysis["geometry_owners"] < 2):
            continue
        return {"input": canvas, "output": expected}
    raise RuntimeError("bounded joint-assembly sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    first_axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(num_train+num_test):
        split = "train" if i < num_train else "test"
        if i == 0:
            n, period, axis, style = int(rng.choice([8, 10, 12])), 2, first_axis, 0
        elif i == 1:
            n, period, axis, style = 12, 3, 1-first_axis, int(rng.choice([1, 2, 3]))
        elif split == "test":
            period = int(rng.choice([2, 3]))
            n = int(rng.choice([11, 13])) if period == 2 else int(rng.choice([13, 14]))
            axis, style = int(rng.integers(2)), int(rng.choice([1, 2, 3]))
        else:
            period = int(rng.choice([2, 3]))
            n = int(rng.choice([8, 10, 12, 14])) if period == 2 else 12
            axis, style = int(rng.integers(2)), int(rng.integers(5))
        turns = [0]+([1, 2, 3] if i < 2 else [int(x) for x in rng.integers(0, 4, size=3)])
        for _ in range(8):
            pair = _make_pair(rng, n, period, axis, style, i < 2 or split == "test", turns)
            token = (pair["input"].shape, pair["input"].tobytes())
            if token not in seen:
                seen.add(token)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("distinct-input sampling exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            token = (a.shape, a.tobytes())
            _require(token not in seen, "duplicate input")
            seen.add(token)
            info = _analyze(a)
            _require(np.array_equal(out, info["output"]), "output differs from joint input-only solution")
            _require(2 <= info["errors"] <= 4, "generated examples need 2..4 flips")
            _require(set(np.unique(out)) == {info["base"], *map(int, info["key"].flat)}, "all four corner dyes must be visible")
            known = info["observed"] >= 0
            wrong = known & (info["observed"] != info["clean"])
            _require((wrong & (info["clean"] == 0)).any() and (wrong & (info["clean"] == 1)).any(), "both flip directions required")
            # Explicit ablation: copying assembled observations preserves errors.
            no_repair = np.where(info["observed"] == 0, info["base"], info["key"].reshape(-1)[info["owner"]])
            _require(not np.array_equal(no_repair, out), "periodic repair must affect the answer")
            no_key = np.where(info["clean"] == 0, info["base"], info["ink"])
            _require(not np.array_equal(no_key, out), "corner key must affect the answer")
            if split == "train":
                training.append(info)
            else:
                _require(info["side"] % info["period"] != 0, "test must end in a partial repetition")
                _require(info["geometry_owners"] >= 2, "test must need texture to resolve geometric ambiguity")
    _require({x["axis"] for x in training[:2]} == {0, 1}, "first two demonstrations must expose both axes")
    _require({x["period"] for x in training[:2]} == {2, 3}, "first two demonstrations must expose distinct periods")
    _require(all(x["side"] % x["period"] == 0 and x["geometry_owners"] >= 2 for x in training[:2]), "initial demonstrations require complete cycles and ambiguous geometry")
    return None


def self_test():
    """42 seeds, bounded in-memory measurements, count coverage and negatives."""
    shapes, sides, periods, axes, errors, palettes, cuts, area_profiles = set(), set(), set(), set(), set(), set(), set(), set()
    signatures, ambiguous, pairs = set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                digest.update(pair["input"].tobytes())
                shapes.add(pair["input"].shape)
                sides.add(a["side"])
                periods.add(a["period"])
                axes.add(a["axis"])
                errors.add(a["errors"])
                palettes.add((a["background"], a["base"], a["ink"]))
                cuts.add((a["side"], a["owner"].tobytes()))
                area_profiles.add(tuple(sorted(int((a["owner"] == j).sum()) for j in range(4))))
                ambiguous += a["geometry_owners"] > 1
                pairs += 1
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42 and len(shapes) >= 35 and len(cuts) >= 100, "insufficient structural variation")
    _require(periods == {2, 3} and axes == {0, 1} and errors == {2, 3, 4}, "period, axis or noise coverage missing")
    _require(len(sides) >= 6 and len(palettes) >= 50 and len(area_profiles) >= 100, "scale, palette or area variation missing")
    _require(ambiguous >= 126, "joint registration not sufficiently exposed")
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(700+tr*10+te, tr, te)
            _require(len(p["train"]) == tr and len(p["test"]) == te, "explicit counts ignored")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(0), generate_puzzle(0)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "nondeterminism")
    a["test"][0]["output"][0, 0] = (int(a["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(a)
    except ValueError:
        pass
    else:
        raise AssertionError("bad output accepted")
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(shapes), "output_sides": sorted(sides),
            "periods": sorted(periods), "axes": sorted(axes), "flip_counts": sorted(errors),
            "palettes": len(palettes), "cuts": len(cuts), "area_profiles": len(area_profiles),
            "geometrically_ambiguous_pairs": ambiguous}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
