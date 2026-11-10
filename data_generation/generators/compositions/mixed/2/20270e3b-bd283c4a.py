"""Register transparent colored fragments, then frequency-pack their union.

The solver reads only pixels. Generation constructs colored masks directly; it
does not import, execute, or compose either parent's whole-puzzle generator.
All experiments are bounded and self_test retains aggregate measurements only.
"""
from collections import Counter
import hashlib
import json

import numpy as np


_DIRECTIONS = tuple((major, outer, inner)
                    for major in ("column", "row")
                    for outer in (False, True) for inner in (False, True))
_CORRECT_DIRECTION = ("column", False, True)
_ASSEMBLY_MODELS = tuple((seam, blend, guides)
                        for seam in (-1, 0, 1)
                        for blend in ("union", "xor", "anchor", "mover",
                                      "anchor_opaque", "mover_opaque")
                        for guides in ("transparent", "erase", "paint"))
_CORRECT_ASSEMBLY = (-1, "union", "transparent")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= n <= 30 for n in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=True):
    remaining, result = set(cells), []
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (diagonal or not (dr and dc))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _hist(grid):
    return Counter(int(c) for c in np.asarray(grid).flat)


def _rank(hist):
    return tuple(sorted(hist, key=lambda c: (-hist[c], c)))


def _render(cells, background):
    if not cells:
        raise ValueError("empty union")
    t, b, l, r = _box(cells)
    a = np.full((b - t + 1, r - l + 1), background, dtype=np.int8)
    for (y, x), color in cells.items():
        a[y - t, x - l] = color
    return a


def _pack(rectangle, direction=_CORRECT_DIRECTION, ranking="frequency"):
    counts = _hist(rectangle)
    if ranking == "frequency":
        colors = _rank(counts)
    elif ranking == "ascending":
        colors = sorted(counts, key=lambda c: (counts[c], c))
    elif ranking == "numeric":
        colors = sorted(counts)
    else:
        raise ValueError("unknown ranking")
    stream = [c for c in colors for _ in range(counts[c])]
    h, w = rectangle.shape
    major, reverse_outer, reverse_inner = direction
    rows = list(range(h))
    cols = list(range(w))
    if major == "column":
        if reverse_outer:
            cols.reverse()
        if reverse_inner:
            rows.reverse()
        positions = ((r, c) for c in cols for r in rows)
    else:
        if reverse_outer:
            rows.reverse()
        if reverse_inner:
            cols.reverse()
        positions = ((r, c) for r in rows for c in cols)
    result = np.empty_like(rectangle)
    for (r, c), color in zip(positions, stream):
        result[r, c] = color
    return result


def _parse(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if not (np.all(grid[0] == background) and np.all(grid[-1] == background)
            and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)):
        raise ValueError("the outer border must identify the background")
    occupied = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    fragments = _components(occupied)
    if len(fragments) != 2:
        raise ValueError("expected two separated eight-connected fragments")
    candidates = []
    for marker in map(int, np.unique(grid)):
        if marker == background:
            continue
        bars = _components({tuple(map(int, p)) for p in np.argwhere(grid == marker)}, False)
        if len(bars) != 2 or len(bars[0]) != len(bars[1]):
            continue
        length = len(bars[0])
        if not 2 <= length <= 6:
            continue
        for axis in (0, 1):
            convert = lambda p: (p[axis], p[1 - axis])
            oriented = [{convert(p) for p in bar} for bar in bars]
            if any(len({r for r, c in bar}) != 1
                   or max(c for r, c in bar) - min(c for r, c in bar) + 1 != length
                   for bar in oriented):
                continue
            records = []
            for fragment in fragments:
                own = [i for i, bar in enumerate(bars) if bar <= fragment]
                if len(own) != 1:
                    break
                bar = oriented[own[0]]
                r0, _, c0, _ = _box(bar)
                cells = {}
                for p in fragment - bars[own[0]]:
                    r, c = convert(p)
                    cells[r - r0, c - c0] = int(grid[p])
                negative = any((-1, c) in cells for c in range(length))
                positive = any((1, c) in cells for c in range(length))
                if negative == positive:
                    break
                records.append((negative, cells))
            if len(records) != 2 or records[0][0] == records[1][0]:
                continue
            records.sort(key=lambda record: not record[0])
            candidates.append({"a": records[0][1], "b": records[1][1],
                               "background": background, "marker": marker,
                               "length": length, "axis": axis})
    if len(candidates) != 1:
        raise ValueError("registration marker and orientation must be unique")
    return candidates[0]


def _assemble(parsed, seam=-1, blend="union", guides="transparent"):
    a = parsed["a"]
    b = {(r + seam, c): color for (r, c), color in parsed["b"].items()}
    shared = set(a) & set(b)
    if blend == "union" and any(a[p] != b[p] for p in shared):
        raise ValueError("conflicting foreground colors at an overlap")
    if blend == "union":
        merged = dict(a)
        merged.update(b)
    elif blend == "xor":
        merged = {p: color for p, color in (list(a.items()) + list(b.items())) if p not in shared}
    elif blend in ("anchor", "mover"):
        merged = dict(a if blend == "anchor" else b)
    elif blend in ("anchor_opaque", "mover_opaque"):
        front, back = (a, b) if blend == "anchor_opaque" else (b, a)
        t, bot, l, r = _box(front)
        merged = {p: color for p, color in back.items()
                  if not (t <= p[0] <= bot and l <= p[1] <= r)}
        merged.update(front)
    else:
        raise ValueError("unknown blend")
    markers = {(r, c) for r in (0, seam) for c in range(parsed["length"])}
    if guides == "erase":
        merged = {p: color for p, color in merged.items() if p not in markers}
    elif guides == "paint":
        merged.update({p: parsed["marker"] for p in markers})
    rectangle = _render(merged, parsed["background"])
    if parsed["axis"]:
        rectangle = rectangle.T.copy()
    return rectangle, merged, shared, markers


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer registration, deduplicate, crop, count, and rasterize from input only."""
    parsed = _parse(input_grid)
    rectangle, _, _, _ = _assemble(parsed)
    counts = _hist(rectangle)
    if len(set(counts.values())) != len(counts):
        raise ValueError("the family excludes tied union frequencies")
    return _pack(rectangle)


def _surviving_assembly_models(pairs):
    """Finite teaching check using demonstration pixels and labels only.

    Alternative models are allowed tied counts with a numeric fallback: rejecting
    them must rely on their actual predictions, not our no-tie generation rule.
    A color conflict makes transparent colored superposition undefined.
    """
    surviving = set(_ASSEMBLY_MODELS)
    for pair in pairs:
        parsed = _parse(pair["input"])
        for model in tuple(surviving):
            try:
                prediction = _pack(_assemble(parsed, *model)[0])
            except ValueError:
                surviving.remove(model)
            else:
                if not np.array_equal(prediction, pair["output"]):
                    surviving.remove(model)
    return surviving


def _valid_masks(a, b, ga, gb, length):
    return (not (a & ga or b & gb)
            and len(_components(a | ga)) == len(_components(b | gb)) == 1
            and any(p in a for p in ((next(iter(ga))[0] - 1, c) for _, c in ga))
            and any(p in b for p in ((next(iter(gb))[0] + 1, c) for _, c in gb))
            and all((r + 1, c) not in a for r, c in ga)
            and all((r - 1, c) not in b for r, c in gb))


def _geometry(rng, h, w, length, kind, collision, holes):
    g = int(rng.integers(2, h - 2))
    c0 = int(rng.integers(1, w - length))
    ga = {(g, c) for c in range(c0, c0 + length)}
    gb = {(g - 1, c) for c in range(c0, c0 + length)}
    if kind == "terraces":
        boundary = rng.integers(0, h - 1, size=w)
        boundary[c0:c0 + length] = g - 1
        depth = rng.integers(1, 4, size=w) if collision else np.zeros(w, dtype=int)
        depth[c0:c0 + length] = 0
        # Two exterior columns guarantee a substantial shared-cell witness.
        if collision:
            boundary[0] = boundary[-1] = max(2, g - 1)
            depth[0] = depth[-1] = 3
        a = {(r, c) for c in range(w) for r in range(int(boundary[c]) + 1)}
        b = {(r, c) for c in range(w)
             for r in range(max(0, int(boundary[c] + 1 - depth[c])), h)}
    elif kind == "frame":
        a = {(r, c) for r in range(h) for c in range(w)
             if r in (0, h - 1) or c in (0, w - 1)}
        a.update((g - 1, c) for c in range(c0 + length))
        left = int(rng.integers(1, c0 + 1))
        right = int(rng.integers(c0 + length - 1, w - 1))
        bottom = h - 1 if collision else h - 2
        filled = bool(rng.integers(2))
        b = {(r, c) for r in range(g, bottom + 1) for c in range(left, right + 1)
             if filled or r in (g, bottom) or c in (left, right)}
    else:  # Hollow/open stroke motifs with optional crossing side arms.
        a = {(r, c) for r in range(g) for c in range(w)
             if r in (0, g - 1) or c in (0, w - 1)}
        b = {(r, c) for r in range(g, h) for c in range(w)
             if r in (g, h - 1) or c in (0, w - 1)}
        for c in range(1, w - 1):
            if rng.random() < .35:
                a.update((r, c) for r in range(int(rng.integers(g))))
            if rng.random() < .35:
                b.update((r, c) for r in range(int(rng.integers(g, h)), h))
        if collision:
            side = 0 if rng.integers(2) else w - 1
            a.update((r, side) for r in range(g, min(h, g + int(rng.integers(1, 4)))))
            b.update((r, side) for r in range(max(0, g - int(rng.integers(1, 4))), g))
    a -= ga
    b -= gb
    if holes:
        # A real uncovered guide, while other backing/guide positions survive.
        missing = (g - 1, int(rng.integers(c0, c0 + length)))
        a.discard(missing)
        b.discard(missing)
        # Interior holes can remove material from either or both masks.
        candidates = sorted((a | b) - ga - gb)
        candidates = [p for p in candidates if 0 < p[0] < h - 1 and 0 < p[1] < w - 1]
        if candidates:
            for index in rng.choice(len(candidates), size=min(len(candidates), int(rng.integers(1, 5))), replace=False):
                a.discard(candidates[int(index)])
                b.discard(candidates[int(index)])
    if not _valid_masks(a, b, ga, gb, length):
        return None
    return a, b, ga, gb


def _color_union(rng, a, b, background, colors, witness, axis):
    cells, overlap = sorted(a | b), sorted(a & b)
    t, bot, l, r = _box(cells)
    blank = (bot - t + 1) * (r - l + 1) - len(cells)
    k, n = len(colors), len(cells)
    if n < k * (k + 1) // 2:
        return None
    for _ in range(120):
        profile = int(rng.integers(4)) if not witness else 0
        alpha = (12., 1.5, .45, 3.)[profile]
        counts = np.sort(rng.multinomial(n - k, rng.dirichlet(np.full(k, alpha))) + 1)[::-1]
        if len(set(map(int, counts))) != k or (blank and blank in counts):
            continue
        ranked_colors = list(map(int, rng.permutation(colors)))
        stream = [c for count, c in zip(counts, ranked_colors) for _ in range(int(count))]
        if witness:
            # Boost raw rank two with repeated cells; deduplication reverses it.
            repeated = min(len(overlap), int(counts[1]))
            chosen = overlap[:repeated]
            inventory = Counter(stream)
            inventory[ranked_colors[1]] -= repeated
            rest = [c for c in ranked_colors for _ in range(inventory[c])]
            rng.shuffle(rest)
            mapping = {p: ranked_colors[1] for p in chosen}
            mapping.update(zip([p for p in cells if p not in mapping], rest))
        else:
            mode = int(rng.integers(4))
            if mode == 0:
                rng.shuffle(stream)
                order = cells
            elif mode == 1:
                order = sorted(cells, key=lambda p: (p[1], -p[0]))
            elif mode == 2:
                ar, ac = int(rng.integers(t, bot + 1)), int(rng.integers(l, r + 1))
                order = sorted(cells, key=lambda p: (abs(p[0] - ar) + abs(p[1] - ac), p))
            else:
                tile = int(rng.integers(2, 4))
                order = sorted(cells, key=lambda p: (p[0] // tile, p[1] // tile, p))
            mapping = dict(zip(order, stream))
        rectangle = _render(mapping, background)
        if axis:
            rectangle = rectangle.T.copy()
        packed = _pack(rectangle)
        if witness:
            raw = Counter(mapping[p] for p in a)
            raw.update(mapping[p] for p in b)
            union_fg = Counter(mapping.values())
            matches = [d for d in _DIRECTIONS if np.array_equal(_pack(rectangle, d), packed)]
            if (_rank(raw) == _rank(union_fg) or matches != [_CORRECT_DIRECTION]
                    or np.array_equal(_pack(rectangle, ranking="numeric"), packed)
                    or np.array_equal(rectangle, packed)):
                continue
        return mapping, rectangle, packed
    return None


def _stage(rng, a, b, ga, gb, mapping, background, marker, mode, axis):
    groups = []
    for mask, guide in ((a, ga), (b, gb)):
        t, bot, l, r = _box(mask | guide)
        data = {(y - t, x - l): mapping[y, x] for y, x in sorted(mask)}
        data.update({(y - t, x - l): marker for y, x in sorted(guide)})
        groups.append((data, bot - t + 1, r - l + 1))
    if rng.integers(2):
        groups.reverse()
    (one, h1, w1), (two, h2, w2) = groups
    gap = int(rng.integers(1, 4))
    if mode == "horizontal":
        offsets = [(0, 0), (int(rng.integers(-2, 3)), w1 + gap)]
    elif mode == "vertical":
        offsets = [(0, 0), (h1 + gap, int(rng.integers(-2, 3)))]
    else:
        offsets = [(0, 0), (h1 + gap, w1 + gap)]
    scene = {}
    for (data, _, _), (dr, dc) in zip(groups, offsets):
        scene.update({(r + dr, c + dc): color for (r, c), color in data.items()})
    tight = _render(scene, background)
    if any(n > 28 for n in tight.shape):
        return None
    top = int(rng.integers(1, min(3, 30 - tight.shape[0])))
    left = int(rng.integers(1, min(3, 30 - tight.shape[1])))
    bottom = int(rng.integers(1, min(3, 31 - tight.shape[0] - top)))
    right = int(rng.integers(1, min(3, 31 - tight.shape[1] - left)))
    grid = np.pad(tight, ((top, bottom), (left, right)), constant_values=background)
    return grid.T.copy() if axis else grid


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _make_pair(rng, profile, seen, teaching_pair=None):
    previous_rank = None
    if teaching_pair is not None:
        previous = _assemble(_parse(teaching_pair["input"]))[1]
        previous_rank = _rank(Counter(previous.values()))
    for _ in range(180):
        length = profile["length"]
        lo, hi = profile["size"]
        h = int(rng.integers(lo, hi + 1))
        w = int(rng.integers(max(length + 2, lo), max(length + 2, hi) + 1))
        geometry = _geometry(rng, h, w, length, profile["kind"],
                             profile["collision"], profile["holes"])
        if geometry is None:
            continue
        a, b, ga, gb = geometry
        if profile["witness"] and len(a & b) < 4:
            continue
        coloring = _color_union(rng, a, b, profile["background"], profile["colors"],
                                profile["witness"], profile["axis"])
        if coloring is None:
            continue
        mapping, rectangle, output = coloring
        if previous_rank == _rank(Counter(mapping.values())):
            continue
        grid = _stage(rng, a, b, ga, gb, mapping, profile["background"],
                      profile["marker"], profile["stage"], profile["axis"])
        if grid is None or _signature(grid) in seen:
            continue
        try:
            parsed = _parse(grid)
            reconstructed, merged, shared, markers = _assemble(parsed)
        except ValueError:
            continue
        if not np.array_equal(reconstructed, rectangle):
            continue
        if profile["holes"] and (not (markers - set(merged)) or not (markers & set(merged))):
            continue
        pair = {"input": grid, "output": output}
        if teaching_pair is not None and _surviving_assembly_models(
                [teaching_pair, pair]) != {_CORRECT_ASSEMBLY}:
            continue
        return pair, (a, b, ga, gb, mapping, profile.copy())
    raise RuntimeError("bounded fragment construction exhausted")


def _contrast(rng, source, seen):
    a, b, ga, gb, mapping, profile = source
    a, b = set(a), set(b)
    # Remove redundant ownership only; the colored union is unchanged.
    for index in rng.permutation(len(a & b)):
        p = sorted(a & b)[int(index)]
        aa = a - {p}
        if _valid_masks(aa, b, ga, gb, profile["length"]):
            a = aa
            break
        bb = b - {p}
        if _valid_masks(a, bb, ga, gb, profile["length"]):
            b = bb
            break
    for _ in range(16):
        mode = "vertical" if profile["stage"] == "horizontal" else "horizontal"
        grid = _stage(rng, a, b, ga, gb, mapping, profile["background"], profile["marker"], mode, profile["axis"])
        if grid is None or _signature(grid) in seen:
            continue
        try:
            parsed = _parse(grid)
            rectangle, _, _, _ = _assemble(parsed)
        except ValueError:
            continue
        expected = _render(mapping, profile["background"])
        if profile["axis"]:
            expected = expected.T.copy()
        if np.array_equal(rectangle, expected):
            return {"input": grid, "output": _pack(expected)}
    raise RuntimeError("bounded controlled contrast exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(("length", "size", "placement", "axis_length")))
    background, marker = map(int, rng.choice(10, size=2, replace=False))
    palette = [c for c in range(10) if c not in (background, marker)]
    content = list(map(int, rng.choice(palette, size=3, replace=False)))
    axis0 = int(rng.integers(2))
    mixed_axis = shift == "axis_length" or bool(rng.integers(2))
    changing_palette = bool(rng.integers(2))
    puzzle, seen, first_source = {"train": [], "test": []}, set(), None
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            if split == "train" and index == 2:
                pair = _contrast(rng, first_source, seen)
            else:
                first = split == "train" and index == 0
                second = split == "train" and index == 1
                train = split == "train"
                length = (2 if index % 2 == 0 else 4) if train else int(rng.choice((2, 4)))
                if shift == "length" and not train:
                    length = 3
                elif shift not in ("length", "axis_length") and not first and not second:
                    length = int(rng.integers(2, 7))
                axis = (axis0 + index % 2) % 2 if mixed_axis else axis0
                if shift == "axis_length":
                    axis = (axis0 + int(length == 4) + int(not train)) % 2
                bg, mk, cols = background, marker, content
                if changing_palette and not first and not second:
                    roles = list(map(int, rng.choice(10, size=7, replace=False)))
                    bg, mk = roles[:2]
                    cols = roles[2:2 + int(rng.integers(2, 6))]
                size = (6, 9)
                if shift == "size":
                    size = (6, 7) if train else (9, 11)
                stage = ("horizontal" if index % 2 == 0 else "vertical") if train else str(rng.choice(("horizontal", "vertical", "diagonal")))
                if shift == "placement" and not train:
                    stage = "diagonal"
                profile = {"length": length, "size": size, "axis": axis,
                           "background": bg, "marker": mk, "colors": cols,
                           "stage": stage, "kind": "terraces" if first else str(rng.choice(("terraces", "frame", "strokes"))),
                           "collision": True if first else bool(rng.integers(2)),
                           "holes": not first, "witness": first}
                teaching_pair = puzzle["train"][0] if second else None
                pair, source = _make_pair(rng, profile, seen, teaching_pair)
                if first:
                    first_source = source
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    before, after = _grid(pair["input"]), _grid(pair["output"])
    parsed = _parse(before)
    rectangle, merged, shared, markers = _assemble(parsed)
    bg = parsed["background"]
    assert len(set(_hist(rectangle).values())) == len(_hist(rectangle)), "tied union frequencies"
    assert after.shape == rectangle.shape
    assert _hist(after) == _hist(rectangle), "wrong union inventory"
    assert np.array_equal(after, _pack(rectangle)), "wrong frequency raster"
    assert parsed["marker"] not in _hist(after), "guide color survived"
    assert len(merged) == len(parsed["a"]) + len(parsed["b"]) - len(shared)
    assert np.count_nonzero(after != bg) == len(merged)
    raw = Counter(parsed["a"].values())
    raw.update(parsed["b"].values())
    content = Counter(merged.values())
    return {"parsed": parsed, "rectangle": rectangle, "overlap": len(shared),
            "holes": int(np.count_nonzero(rectangle == bg)),
            "covered": len(markers & set(merged)), "uncovered": len(markers - set(merged)),
            "raw_rank_changed": _rank(raw) != _rank(content), "raw": raw}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _signature(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _features(pair)
            if split == "train":
                train_features.append(info)
    first, second = train_features[:2]
    assert first["overlap"] >= 4 and first["raw_rank_changed"]
    assert second["holes"] and second["uncovered"] and second["covered"]
    ranks = [_rank(Counter(_assemble(info["parsed"])[1].values()))
             for info in (first, second)]
    assert ranks[0] != ranks[1], "fixed literal color priority survives"
    assert _surviving_assembly_models(puzzle["train"]) == {_CORRECT_ASSEMBLY}, (
        "demonstrations do not expose assembly conventions")
    directions = [d for d in _DIRECTIONS if all(np.array_equal(
        _pack(info["rectangle"], d), pair["output"])
        for info, pair in zip(train_features, puzzle["train"]))]
    assert directions == [_CORRECT_DIRECTION], "demonstrations do not expose traversal"
    assert any(not np.array_equal(info["rectangle"], pair["output"])
               for info, pair in zip(train_features, puzzle["train"]))
    for ranking in ("ascending", "numeric"):
        assert any(not np.array_equal(_pack(info["rectangle"], ranking=ranking), pair["output"])
                   for info, pair in zip(train_features, puzzle["train"])), "ranking shortcut survives"
    if len(train_features) >= 3:
        assert np.array_equal(puzzle["train"][0]["output"], puzzle["train"][2]["output"])


def self_test():
    """42 seeds, streamed metrics, explicit counts and targeted semantic mutants."""
    shapes, output_shapes, palettes, lengths, axes, structures = set(), set(), set(), set(), set(), set()
    overlaps, blanks, raw_changes, partials, dominant_colors = set(), set(), 0, set(), set()
    fingerprints, rejected_mutants = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before, after = pair["input"], pair["output"]
                info = _features(pair)
                p = info["parsed"]
                assert np.array_equal(solve(before.tolist()), after)
                # Explicit scalar raster check, distinct from the packing routine.
                counts = Counter(map(int, info["rectangle"].flat))
                stream = [c for c in sorted(counts, key=lambda c: -counts[c]) for _ in range(counts[c])]
                expected = np.empty_like(after)
                for k, c in enumerate(stream):
                    expected[after.shape[0] - 1 - k % after.shape[0], k // after.shape[0]] = c
                assert np.array_equal(after, expected)
                shapes.add(before.shape)
                output_shapes.add(after.shape)
                palettes.add(len(counts))
                axes.add(p["axis"])
                lengths.add(p["length"])
                overlaps.add(info["overlap"])
                blanks.add(info["holes"])
                raw_changes += info["raw_rank_changed"]
                dominant_colors.add(_rank(counts)[0])
                partials.update(int(x) % after.shape[0] for x in np.cumsum([counts[c] for c in _rank(counts)])[:-1])
                roles, normalized = {}, []
                for c in before.flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                structures.add(hashlib.sha256(bytes(normalized)).digest())
                digest.update(before.tobytes())
                digest.update(after.tobytes())
                # Each named assembly omission must be exposed somewhere in training.
                if split == "train":
                    for name, seam, blend, guides in (
                            ("same_line", 0, "union", "transparent"),
                            ("opposite_seam", 1, "union", "transparent"),
                            ("xor", -1, "xor", "transparent"),
                            ("one_fragment", -1, "anchor", "transparent"),
                            ("erase_guides", -1, "union", "erase"),
                            ("paint_guides", -1, "union", "paint"),
                            ("opaque_a", -1, "anchor_opaque", "transparent"),
                            ("opaque_b", -1, "mover_opaque", "transparent")):
                        try:
                            wrong = _pack(_assemble(p, seam, blend, guides)[0])
                        except ValueError:
                            rejected_mutants.add(name)
                        else:
                            if not np.array_equal(wrong, after):
                                rejected_mutants.add(name)
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(output_shapes) >= 20 and len(structures) >= 150
    assert axes == {0, 1} and lengths >= {2, 3, 4, 5, 6}
    assert 0 in overlaps and max(overlaps) >= 10
    assert 0 in blanks and len(blanks) >= 12
    assert len(palettes) >= 3 and raw_changes >= 42
    assert len(partials) >= 6 and len(dominant_colors) >= 8
    assert len(rejected_mutants) == 8
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(20261023 + nt * 7 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4., 1), (4, False), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(12345, 10, 4), generate_puzzle(12345, 10, 4)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "normalized_inputs": len(structures),
            "palette_sizes": sorted(palettes), "bar_lengths": sorted(lengths),
            "overlap_counts": sorted(overlaps), "hole_counts": sorted(blanks),
            "raw_rank_changes": raw_changes, "assembly_mutants_exposed": sorted(rejected_mutants)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
