"""Joint positive-clue selection and exact-cover completion of a tile-class map.

Only standard-library modules and NumPy are used. Geometry caches contain tile
placements, never generated examples, answers, seeds, or palette conventions.
"""
from functools import lru_cache
import json

import numpy as np


_BASE_SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
)
_MAX_COVERS = 96  # Sampling bound only; solve has no unconditioned cover cap.


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected an integer 1..30 by 1..30 color grid")
    return g.astype(np.int8, copy=False)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _shapes(symmetry):
    result = []
    for cells in _BASE_SHAPES:
        m = np.zeros((2, 2), dtype=np.int8)
        for p in cells:
            m[p] = 1
        m = np.rot90(m, symmetry % 4)
        if symmetry >= 4:
            m = np.fliplr(m)
        result.append(tuple(map(tuple, np.argwhere(m).tolist())))
    return tuple(result)


@lru_cache(maxsize=256)
def _placements(h, w, shapes):
    result = []
    for r in range(h - 1):
        for c in range(w - 1):
            for kind, shape in enumerate(shapes):
                cells = tuple((r + dr) * w + c + dc for dr, dc in shape)
                bits = sum(1 << p for p in cells)
                result.append((kind, r, c, cells, bits))
    return tuple(result)


def _bits(mask):
    return sum(1 << int(p) for p in np.flatnonzero(mask))


def _covers(mask, shapes, clues=None, limit=2):
    """Enumerate covers, optionally enforcing class labels at positive cells.

    A missing L corner is unrestricted. In particular, it can belong to another
    tile. Color clues are checked at every occupied tile cell before search.
    """
    h, w = mask.shape
    occupied = _bits(mask)
    if not occupied:
        return []
    if clues is not None and np.any((clues >= 0) & ~mask):
        return []
    placements = _placements(h, w, shapes)
    available = []
    by_cell = [[] for _ in range(h * w)]
    for item in placements:
        kind, _, _, cells, bits = item
        if bits & occupied != bits:
            continue
        if clues is not None and any(clues.flat[p] not in (-1, kind) for p in cells):
            continue
        j = len(available)
        available.append(item)
        for p in cells:
            by_cell[p].append(j)
    positions = list(map(int, np.flatnonzero(mask)))
    answers = []

    def search(used, chosen):
        if len(answers) >= limit:
            return
        if used == occupied:
            answers.append(tuple(available[j] for j in chosen))
            return
        best = None
        for p in positions:
            if used & (1 << p):
                continue
            options = [j for j in by_cell[p] if not (used & available[j][4])]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        for j in best:
            search(used | available[j][4], chosen + (j,))
            if len(answers) >= limit:
                return

    search(0, ())
    return answers


def _labels(shape, cover):
    labels = np.full(shape, -1, dtype=np.int8)
    for kind, _, _, cells, _ in cover:
        for p in cells:
            labels.flat[p] = kind
    return labels


def _frame(g, bg):
    found = []
    for value in np.unique(g):
        color = int(value)
        if color == bg:
            continue
        points = np.argwhere(g == color)
        lo, hi = points.min(axis=0), points.max(axis=0) + 1
        r, c = map(int, lo)
        hh, ww = map(int, hi - lo)
        if hh < 8 or ww < 8:
            continue
        actual = g[r:r + hh, c:c + ww] == color
        perimeter = np.zeros((hh, ww), bool)
        perimeter[0, :] = perimeter[-1, :] = True
        perimeter[:, 0] = perimeter[:, -1] = True
        if np.array_equal(actual, perimeter):
            found.append((color, (r, c, hh, ww)))
    if len(found) != 1:
        raise ValueError("expected exactly one one-cell rectangular border")
    return found[0]


def _separate(a, b):
    r, c, h, w = a
    y, x, hh, ww = b
    return r + h < y or y + hh < r or c + w < x or x + ww < c


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("the outer canvas edge must be background")
    border, frame = _frame(g, bg)
    r, c, fh, fw = frame
    outside = np.ones(g.shape, bool)
    outside[r:r + fh, c:c + fw] = False
    legends, catalogs, boxes = [], [], [frame]
    for color in map(int, np.unique(g[outside])):
        if color == bg:
            continue
        pts = np.argwhere((g == color) & outside)
        lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
        y, x = map(int, lo)
        h, w = map(int, hi - lo)
        box = (y, x, h, w)
        patch = g[y:y + h, x:x + w]
        if not set(map(int, np.unique(patch))) <= {bg, color}:
            raise ValueError("object bounding boxes must contain a single ink")
        if not all(_separate(box, old) for old in boxes):
            raise ValueError("object bounding boxes require a background gap")
        boxes.append(box)
        mask = patch == color
        if (h, w) == (2, 2) and int(mask.sum()) in (3, 4):
            legends.append((color, tuple(map(tuple, np.argwhere(mask).tolist()))))
        else:
            if h < 3 or w < 3:
                raise ValueError("catalog grids must span at least 3 by 3")
            if np.any((g == color) & ~outside):
                raise ValueError("catalog ink may not appear inside the frame")
            catalogs.append((color, mask))
    if len(legends) != 4 or not 2 <= len(catalogs) <= 4:
        raise ValueError("expected four legend tiles and two to four candidates")
    shapes = tuple(item[1] for item in legends)
    if len(set(shapes)) != 4 or sorted(map(len, shapes)) != [3, 3, 3, 4]:
        raise ValueError("legend must show a square and three distinct L orientations")
    colors = tuple(item[0] for item in legends)
    if len({m.shape for _, m in catalogs}) != 1:
        raise ValueError("candidate silhouettes must share a bounding grid")
    h, w = catalogs[0][1].shape
    if (fh - 2) % h or (fw - 2) % w:
        raise ValueError("frame and catalog dimensions do not align")
    k = (fh - 2) // h
    if k != (fw - 2) // w or k < 2:
        raise ValueError("magnification must be a common integer at least two")
    interior = g[r + 1:r + fh - 1, c + 1:c + fw - 1]
    clues = np.full((h, w), -1, dtype=np.int8)
    for y in range(h):
        for x in range(w):
            block = interior[y * k:(y + 1) * k, x * k:(x + 1) * k]
            value = int(block[0, 0])
            if not np.all(block == value) or value not in (bg,) + colors:
                raise ValueError("clues must be aligned uniform blocks in legend colors")
            if value != bg:
                clues[y, x] = colors.index(value)
    if not np.any(clues >= 0):
        raise ValueError("at least one positive class clue is required")
    return dict(grid=g, bg=bg, border=border, frame=frame, shapes=shapes,
                colors=colors, catalogs=catalogs, clues=clues, scale=k, boxes=boxes)


def _matching(info):
    matches = []
    for i, (_, mask) in enumerate(info["catalogs"]):
        for cover in _covers(mask, info["shapes"], info["clues"], limit=2):
            matches.append((i, cover))
            if len(matches) == 2:
                return matches
    return matches


def _render(info, labels):
    _, _, h, w = info["frame"]
    result = np.full((h, w), info["bg"], dtype=np.int8)
    result[0, :] = result[-1, :] = info["border"]
    result[:, 0] = result[:, -1] = info["border"]
    small = np.full(labels.shape, info["bg"], dtype=np.int8)
    for kind, color in enumerate(info["colors"]):
        small[labels == kind] = color
    k = info["scale"]
    result[1:-1, 1:-1] = np.repeat(np.repeat(small, k, axis=0), k, axis=1)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the grammar, scale, candidate, and cover entirely from input."""
    info = _parse(input_grid)
    matches = _matching(info)
    if len(matches) != 1:
        raise ValueError("positive class clues must select one candidate and one exact cover")
    i, cover = matches[0]
    return _render(info, _labels(info["catalogs"][i][1].shape, cover))


def _sample_mask(rng, h, w, shapes, inventory):
    placements = _placements(h, w, shapes)
    full = (1 << (h * w)) - 1
    for _ in range(90):
        used = 0
        order = list(inventory)
        rng.shuffle(order)
        for kind in order:
            options = [p for p in placements if p[0] == kind and not (used & p[4])]
            if not options:
                break
            p = options[int(rng.integers(len(options)))]
            used |= p[4]
        else:
            if used == full:
                continue
            mask = np.array([(used >> p) & 1 for p in range(h * w)], dtype=bool).reshape(h, w)
            if mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any():
                return mask
    return None


def _minimal_clues(rng, records, target_index, allowed):
    """Greedy distinguishing clues followed by deletion to irredundancy."""
    target = records[target_index][2]
    positions = list(map(int, np.flatnonzero(allowed)))
    rng.shuffle(positions)
    remaining = [i for i in range(len(records)) if i != target_index]
    chosen = []
    while remaining:
        scores = [(sum(records[i][2].flat[p] != target.flat[p] for i in remaining), p)
                  for p in positions if p not in chosen]
        if not scores or max(s for s, _ in scores) == 0:
            return None
        best = max(s for s, _ in scores)
        pool = [p for s, p in scores if s == best]
        p = pool[int(rng.integers(len(pool)))]
        chosen.append(p)
        remaining = [i for i in remaining if records[i][2].flat[p] == target.flat[p]]
    rng.shuffle(chosen)
    for p in chosen[:]:
        other = [q for q in chosen if q != p]
        if all(i == target_index or any(records[i][2].flat[q] != target.flat[q] for q in other)
               for i in range(len(records))):
            chosen.remove(p)
    clues = np.full(target.shape, -1, dtype=np.int8)
    for p in chosen:
        clues.flat[p] = target.flat[p]
    return clues


def _make_logic(rng, h, w, tile_count, count, shapes, intro):
    for _ in range(160):
        inventory = list(range(4)) if intro else list(map(int, rng.integers(4, size=tile_count)))
        if intro:
            inventory += list(map(int, rng.integers(4, size=tile_count - 4)))
        mask = _sample_mask(rng, h, w, shapes, inventory)
        if mask is None:
            continue
        covers = _covers(mask, shapes, limit=_MAX_COVERS + 1)
        if not covers or len(covers) > _MAX_COVERS or (intro and len(covers) != 1):
            continue
        if not intro and len(covers) < 2:
            continue
        masks, all_covers = [mask], [covers]
        for _ in range(120):
            n = max(2, tile_count + int(rng.integers(-1, 2)))
            inv = list(map(int, rng.integers(4, size=n)))
            m = _sample_mask(rng, h, w, shapes, inv)
            if m is None or any(np.array_equal(m, old) for old in masks):
                continue
            cc = _covers(m, shapes, limit=_MAX_COVERS + 1)
            if not cc or len(cc) > _MAX_COVERS:
                continue
            masks.append(m)
            all_covers.append(cc)
            if len(masks) == count:
                break
        if len(masks) != count:
            continue
        records = [(i, cover, _labels((h, w), cover))
                   for i, group in enumerate(all_covers) for cover in group]
        targets = list(range(len(covers)))
        rng.shuffle(targets)
        for target in targets:
            for distractor in rng.permutation(np.arange(1, count)):
                # All clues fall on this distractor too: occupancy cannot pick
                # a catalog entry even when every positive coordinate is used.
                clues = _minimal_clues(rng, records, target, mask & masks[int(distractor)])
                if clues is None:
                    continue
                nc = int(np.count_nonzero(clues >= 0))
                if nc == 0 or nc > 5 or (not intro and nc < 2):
                    continue
                labels = records[target][2]
                if len(np.unique(labels[labels >= 0])) < 2 or nc >= int(mask.sum()):
                    continue
                # Some examples give a more complete partial map. Extra clues
                # are still positive, and in the joint regime no single clue
                # is allowed to select the answer by itself.
                if rng.random() < .40:
                    extra = [p for p in np.flatnonzero(mask & masks[int(distractor)] & (clues < 0))
                             if intro or sum(rec[2].flat[p] == labels.flat[p]
                                             for rec in records) >= 2]
                    rng.shuffle(extra)
                    for p in extra[:min(int(rng.integers(1, 4)), int(mask.sum()) - nc - 1)]:
                        clues.flat[p] = labels.flat[p]
                return masks, clues, labels
    raise RuntimeError("could not build a joint candidate/cover clue problem")


def _free_origins(blocked, h, w):
    integral = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _pack(rng, patches, bg):
    """Free rectangular packing with no fixed panels or catalog order."""
    largest_h = max(p.shape[0] for p in patches)
    largest_w = max(p.shape[1] for p in patches)
    for _ in range(160):
        height = int(rng.integers(max(largest_h + 2, 19), 31))
        width = int(rng.integers(max(largest_w + 2, 19), 31))
        blocked = np.zeros((height, width), bool)
        blocked[0, :] = blocked[-1, :] = True
        blocked[:, 0] = blocked[:, -1] = True
        grid = np.full((height, width), bg, dtype=np.int8)
        # Frame first, then silhouettes in random order, then tiny legend tiles.
        order = [0] + list(range(1, len(patches) - 4))
        tail = order[1:]
        rng.shuffle(tail)
        order[1:] = tail
        legends = list(range(len(patches) - 4, len(patches)))
        rng.shuffle(legends)
        order += legends
        for i in order:
            patch = patches[i]
            h, w = patch.shape
            options = _free_origins(blocked, h, w)
            if not len(options):
                break
            r, c = map(int, options[int(rng.integers(len(options)))])
            grid[r:r + h, c:c + w] = patch
            blocked[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
        else:
            # Tighten unused exterior space with independently varying padding.
            pts = np.argwhere(grid != bg)
            lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
            r0 = max(0, int(lo[0]) - int(rng.integers(1, 4)))
            c0 = max(0, int(lo[1]) - int(rng.integers(1, 4)))
            r1 = min(height, int(hi[0]) + int(rng.integers(1, 4)))
            c1 = min(width, int(hi[1]) + int(rng.integers(1, 4)))
            return grid[r0:r1, c0:c1].copy()
    raise RuntimeError("could not pack frame, silhouettes, and legend under 30 by 30")


def _compose(rng, shapes, palette, intro, test, scale):
    # Inventories grow in tests; geometry and scale remain visible input roles.
    bg, border = palette[:2]
    class_colors = palette[2:6]
    for _ in range(100):
        h, w = map(int, rng.integers(3, 8, size=2))
        n = 4 if intro else int(rng.integers(6, 10) if test else rng.integers(4, 8))
        # Bounds express packing density, not a dimension/inventory recipe.
        # They allow tall, wide, square, sparse, and dense silhouettes.
        if not (.48 <= (3.25 * n) / (h * w) <= .90):
            continue
        if intro and h * w > 25:
            continue
        count = int(rng.integers(2, 5))
        # Larger frames leave less room for a large catalog.
        k = scale
        if max(h * k + 2, w * k + 2) > 26:
            continue
        try:
            masks, clues, labels = _make_logic(rng, h, w, n, count, shapes, intro)
        except RuntimeError:
            continue
        frame = np.full((h * k + 2, w * k + 2), bg, dtype=np.int8)
        frame[0, :] = frame[-1, :] = border
        frame[:, 0] = frame[:, -1] = border
        small = np.full((h, w), bg, dtype=np.int8)
        for kind, color in enumerate(class_colors):
            small[clues == kind] = color
        frame[1:-1, 1:-1] = np.repeat(np.repeat(small, k, 0), k, 1)
        inks = list(palette[6:])
        rng.shuffle(inks)
        patches = [frame] + [np.where(m, ink, bg).astype(np.int8) for m, ink in zip(masks, inks)]
        for shape, color in zip(shapes, class_colors):
            patch = np.full((2, 2), bg, dtype=np.int8)
            for p in shape:
                patch[p] = color
            patches.append(patch)
        try:
            grid = _pack(rng, patches, bg)
        except RuntimeError:
            continue
        info = dict(frame=(0, 0, *frame.shape), bg=bg, border=border,
                    colors=class_colors, scale=k)
        # Expected output is rendered from the selected construction cover.
        # The public solver independently recovers it from the packed input.
        return {"input": grid, "output": _render(info, labels)}
    raise RuntimeError("unable to construct a pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    shapes = _shapes(int(rng.integers(8)))
    palette = tuple(map(int, rng.permutation(10)))
    scales = list(map(int, rng.choice([2, 3, 4], size=2, replace=False)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            k = scales[i % 2] if split == "train" else scales[int(rng.integers(2))]
            for _ in range(10):
                pair = _compose(rng, shapes, palette, intro=split == "train" and i == 0,
                                test=split == "test", scale=k)
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    info = _parse(pair["input"])
    output = _grid(pair["output"])
    matches = _matching(info)
    assert len(matches) == 1, "joint cover must be unique"
    selected, cover = matches[0]
    mask = info["catalogs"][selected][1]
    labels = _labels(mask.shape, cover)
    expected = _render(info, labels)
    assert np.array_equal(output, expected), "wrong class completion or frame crop"
    positive = info["clues"] >= 0
    nc = int(positive.sum())
    assert 0 < nc < int(mask.sum()), "must complete at least one occupied block"
    assert np.all(labels[positive] == info["clues"][positive]), "clue overwritten"
    binary_matches = sum(bool(m[positive].all()) for _, m in info["catalogs"])
    assert binary_matches >= 2, "occupancy-only matching must be insufficient"
    assert len({_key(m) for _, m in info["catalogs"]}) == len(info["catalogs"])
    assert all(_covers(m, info["shapes"], limit=1) for _, m in info["catalogs"]), "untileable candidate"
    unconditioned = _covers(mask, info["shapes"], limit=_MAX_COVERS + 1)
    assert 1 <= len(unconditioned) <= _MAX_COVERS
    # Record whether each clue alone still admits another candidate or cover.
    # The advanced demonstrations and tests require this joint-clue property.
    joint = True
    for r, c in np.argwhere(positive):
        single = dict(info)
        single["clues"] = np.full(info["clues"].shape, -1, dtype=np.int8)
        single["clues"][r, c] = info["clues"][r, c]
        joint &= len(_matching(single)) >= 2
    # Candidate ink never appears in the framed answer.
    assert not (set(map(int, np.unique(output))) & {color for color, _ in info["catalogs"]})
    kinds = {item[0] for item in cover}
    assert len(kinds) >= 2
    info.update(selected=selected, cover=cover, labels=labels, mask=mask,
                ambiguity=len(unconditioned), clue_count=nc, tile_count=len(cover), kinds=kinds, joint=joint)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    conventions = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                _grid(value)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _inspect(pair)
            analyses[split].append(info)
            conventions.add((info["bg"], info["border"],
                             tuple(sorted(zip(info["shapes"], info["colors"])))))
    assert len(conventions) == 1, "grammar and palette must be consistent within a puzzle"
    first = analyses["train"][0]
    assert first["ambiguity"] == 1 and len(first["kinds"]) == 4
    assert len({x["scale"] for x in analyses["train"]}) >= 2
    for info in analyses["train"][1:] + analyses["test"]:
        assert info["ambiguity"] >= 2 and info["clue_count"] >= 2 and info["joint"]
    for info in analyses["test"]:
        assert info["tile_count"] > first["tile_count"], "test must recombine larger inventory"
        assert info["scale"] in {x["scale"] for x in analyses["train"]}


def _fingerprint(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded, 42-seed diagnostics plus all legal explicit count requests."""
    signatures, geometries, layouts, palettes, scales = set(), set(), set(), set(), set()
    clue_counts, counts, ambiguities, inventories, orientations = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect(pair)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                geometries.add((info["mask"].shape, info["scale"], len(info["catalogs"])))
                layouts.add((pair["input"].shape, tuple(info["boxes"])))
                palettes.add((info["bg"], info["border"], info["colors"]))
                scales.add(info["scale"])
                clue_counts.add(info["clue_count"])
                counts.add(info["tile_count"])
                ambiguities.add(info["ambiguity"])
                inventories.add(tuple(sum(p[0] == kind for p in info["cover"]) for kind in range(4)))
                orientations.add(frozenset(info["shapes"]))
    assert len(signatures) == 42 and len(geometries) >= 20
    assert len(layouts) >= 150 and len(palettes) >= 35
    assert scales == {2, 3, 4} and len(clue_counts) >= 2
    assert len(counts) >= 4 and len(ambiguities) >= 3
    assert len(inventories) >= 30 and len(orientations) == 4
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(10000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    return {"seeds": 42, "puzzles": len(signatures), "geometry_combinations": len(geometries),
            "layouts": len(layouts), "palettes": len(palettes),
            "tile_counts": sorted(counts), "cover_counts": sorted(ambiguities),
            "clue_counts": sorted(clue_counts), "inventories": len(inventories),
            "omitted_orientations": len(orientations)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
