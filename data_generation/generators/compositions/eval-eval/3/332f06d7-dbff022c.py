"""Square-brush flood: a directed contour key colors motion-swept cavities.

Only the standard library and NumPy are used. No parent code or saved samples
are loaded. See task_spec.md for the input-visible conventions and curriculum.
"""
from collections import deque
import json

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "color range")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    steps = _EIGHT if diagonal else _FOUR
    for p in map(tuple, np.argwhere(mask)):
        if seen[p]:
            continue
        seen[p] = True
        queue = [p]
        for r, c in queue:
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(tuple((int(r), int(c)) for r, c in queue))
    return result


def _square(cells):
    rr, cc = zip(*cells)
    h, w = max(rr) - min(rr) + 1, max(cc) - min(cc) + 1
    if h == w and len(cells) == h * w:
        return (min(rr), min(cc)), h
    return None


def _reach(valid, start):
    _require(valid[start], "brush start must be a legal whole-square placement")
    seen = np.zeros(valid.shape, dtype=bool)
    seen[start] = True
    queue = deque([start])
    h, w = valid.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in _FOUR:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and valid[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    return seen


def _sweep(region, start, k):
    """Erode by the footprint, explore its placements, then union footprints."""
    valid = np.lib.stride_tricks.sliding_window_view(region, (k, k)).all(axis=(-1, -2))
    reachable = _reach(valid, start)
    swept = np.zeros(region.shape, dtype=bool)
    for r, c in np.argwhere(reachable):
        swept[r:r + k, c:c + k] = True
    return swept, reachable, valid


def _read_input(inp):
    """Recover roles, enclosure and tools solely from the untouched input."""
    colors, counts = np.unique(inp, return_counts=True)
    _require(np.sum(counts == counts.max()) == 1, "background must be uniquely most frequent")
    bg = int(colors[np.argmax(counts)])
    h, w = inp.shape
    keys = []
    for cells in _components(inp != bg, diagonal=True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != sh * sw or len({int(inp[p]) for p in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and 3 <= sw <= 6:
            if r0 == 0:
                lanes.append(("top", inp[r0, c0:c1 + 1], inp[r1, c0:c1 + 1]))
            if r1 == h - 1:
                lanes.append(("bottom", inp[r1, c0:c1 + 1], inp[r0, c0:c1 + 1]))
        if sw == 2 and 3 <= sh <= 6:
            if c0 == 0:
                lanes.append(("left", inp[r0:r1 + 1, c0], inp[r0:r1 + 1, c1]))
            if c1 == w - 1:
                lanes.append(("right", inp[r0:r1 + 1, c1], inp[r0:r1 + 1, c0]))
        for side, outer, inner in lanes:
            source, fill = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(source)) == len(source):
                keys.append({"side": side, "mapping": dict(zip(source, fill)),
                             "sources": source, "fills": fill, "cells": cells})
    _require(len(keys) == 1, "one isolated unambiguous edge key required")
    key = keys[0]
    key_mask = np.zeros(inp.shape, dtype=bool)
    for p in key["cells"]:
        key_mask[p] = True
    key_colors = set(key["sources"]) | set(key["fills"])
    candidates = []
    ink = {}
    for color in colors:
        color = int(color)
        if color == bg:
            continue
        groups = _components((inp == color) & ~key_mask, diagonal=True)
        if groups:
            ink[color] = groups
            squares = [_square(c) for c in groups]
            if color not in key_colors and all(s is not None for s in squares):
                candidates.append((color, squares))
    _require(len(candidates) == 1, "one key-absent color made entirely of square brushes required")
    brush_color, squares = candidates[0]
    _require(all(1 <= k <= 4 for p, k in squares), "brush sides must be 1..4")
    brush_mask = inp == brush_color
    empty = (inp == bg) | brush_mask
    regions = []
    seed_regions = {}
    for cells in _components(empty):
        mask = np.zeros(inp.shape, dtype=bool)
        for p in cells:
            mask[p] = True
        edge = any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        boundary = set()
        for r, c in cells:
            for dr, dc in _FOUR:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not empty[rr, cc]:
                    boundary.add((rr, cc))
        seeds = [(p, k) for p, k in squares if mask[p]]
        for p, k in seeds:
            _require(mask[p[0]:p[0] + k, p[1]:p[1] + k].all(), "split brush")
            seed_regions[p] = len(regions)
        owner = None
        if not edge:
            _require(not any(key_mask[p] for p in boundary), "key cannot bound a cavity")
            owners = {int(inp[p]) for p in boundary}
            _require(len(owners) == 1, "closed cavity must have one contour color")
            owner = next(iter(owners))
            _require(len(seeds) <= 1, "at most one brush in each closed cavity")
        regions.append({"mask": mask, "edge": edge, "owner": owner,
                        "seeds": seeds, "boundary": boundary})
    _require(len(seed_regions) == len(squares), "every brush must belong to a visible region")
    # Non-tool colors must contain genuine contour ink, not orphan square tokens.
    _require(all(any(_square(c) is None for c in groups)
                 for color, groups in ink.items() if color != brush_color), "orphan solid ink")
    return {"background": bg, "brush": brush_color, "squares": squares,
            "key": key, "key_mask": key_mask, "regions": regions}


def _run(inp):
    info = _read_input(inp)
    out = inp.copy()
    mapping = info["key"]["mapping"]
    records = []
    for region in info["regions"]:
        record = dict(region)
        active = not region["edge"] and bool(region["seeds"]) and region["owner"] in mapping
        record.update(active=active, partial=False, swept_count=0, reachable_count=0,
                      branch=False, remote_fit=False)
        if active:
            start, k = region["seeds"][0]
            swept, reach, valid = _sweep(region["mask"], start, k)
            paint = swept & (inp == info["background"])
            out[paint] = mapping[region["owner"]]
            record.update(swept=swept, reachable=reach, valid=valid,
                          partial=bool(np.any(region["mask"] & ~swept)),
                          swept_count=int(paint.sum()), reachable_count=int(reach.sum()),
                          remote_fit=bool(np.any(valid & ~reach)))
            degree = np.zeros(reach.shape, dtype=np.int8)
            degree[1:] += reach[:-1]
            degree[:-1] += reach[1:]
            degree[:, 1:] += reach[:, :-1]
            degree[:, :-1] += reach[:, 1:]
            record["branch"] = bool(np.any(reach & (degree >= 3)))
        records.append(record)
    info["regions"] = records
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Preserve ink; paint the swept background of eligible seeded cavities."""
    return _run(_grid(input_grid))[0]


def _alternative(inp, policy):
    """Explicit weaker hypotheses for demonstration checks, not generation."""
    info = _read_input(inp)
    out = inp.copy()
    mapping = info["key"]["mapping"]
    for region in info["regions"]:
        if region["edge"] or region["owner"] not in mapping:
            continue
        if not region["seeds"]:
            if policy == "seedless":
                out[region["mask"] & (inp == info["background"])] = mapping[region["owner"]]
            continue
        start, k = region["seeds"][0]
        swept = region["mask"] if policy == "point" else _sweep(region["mask"], start, k)[0]
        fill = region["owner"] if policy == "source" else mapping[region["owner"]]
        if policy == "recursive":
            visited = set()
            while fill in mapping and fill not in visited:
                visited.add(fill)
                fill = mapping[fill]
        out[swept & (inp == info["background"])] = fill
    return out


def _dilate(mask, diagonal=False):
    result = mask.copy()
    h, w = mask.shape
    for dr, dc in (_EIGHT if diagonal else _FOUR):
        r0, r1 = max(0, dr), min(h, h + dr)
        c0, c1 = max(0, dc), min(w, w + dc)
        result[r0:r1, c0:c1] |= mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return result


def _object(cavity, color, start=None, k=1, opening=False, thick=False):
    pad = 2 if thick else 1
    interior = np.pad(cavity, pad)
    extent = _dilate(interior)
    if thick:
        extent = _dilate(extent)
    patch = np.zeros(interior.shape, dtype=np.int8)
    patch[extent & ~interior] = color
    if opening:
        # Remove a straight ray through the shell, from an extreme cavity cell.
        r, c = min(map(tuple, np.argwhere(interior)))
        patch[:r, c] = 0
    if start is not None:
        r, c = start[0] + pad, start[1] + pad
        _require(interior[r:r + k, c:c + k].all(), "constructed brush outside chamber")
        patch[r:r + k, c:c + k] = 1
    return patch


def _transform(rng, patch, turns=None, reflect=None):
    if turns is None:
        turns = int(rng.integers(4))
    if reflect is None:
        reflect = bool(rng.integers(2))
    patch = np.rot90(patch, turns)
    return np.array(np.fliplr(patch) if reflect else patch, copy=True)


def _neck_contrast(rng, color):
    large = int(rng.integers(2, 4))
    q = large + int(rng.integers(2, 4))
    gap = int(rng.integers(1, 5))
    width = int(rng.integers(1, large))
    offset = int(rng.integers(q - width + 1))
    mask = np.zeros((q, 2 * q + gap), dtype=bool)
    mask[:, :q] = mask[:, q + gap:] = True
    mask[offset:offset + width, q:q + gap] = True
    # Both masks and initial top-left coordinates agree; only footprint differs.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    return [_transform(rng, _object(mask, color, (1, 1), k), turns, reflect)
            for k in (1, large)]


def _graph_cavity(rng, k, wanted, cross=False):
    """Grow connected chambers, independent necks, optional loops and spurs."""
    q = k + 2
    gap = int(rng.integers(1, 3)) if q <= 4 else 1
    if cross:
        nr = nc = 3
        nodes = [(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)]
        links = [(nodes[0], p) for p in nodes[1:]]
        start_node = nodes[0]
    else:
        nr, nc = (3, 3) if wanted >= 5 else ((1, 2) if wanted == 2 else (2, 2))
        start_node = (int(rng.integers(nr)), int(rng.integers(nc)))
        nodes, links = [start_node], []
        while len(nodes) < wanted:
            options = [(p, (p[0] + dr, p[1] + dc)) for p in nodes for dr, dc in _FOUR
                       if 0 <= p[0] + dr < nr and 0 <= p[1] + dc < nc
                       and (p[0] + dr, p[1] + dc) not in nodes]
            a, b = options[int(rng.integers(len(options)))]
            nodes.append(b)
            links.append((a, b))
        if rng.random() < .45:
            existing = {frozenset((a, b)) for a, b in links}
            for a in nodes:
                for b in nodes:
                    if sum(abs(a[i] - b[i]) for i in (0, 1)) == 1 and frozenset((a, b)) not in existing:
                        if rng.random() < .45:
                            links.append((a, b))
                        existing.add(frozenset((a, b)))
    mask = np.zeros((nr * q + (nr - 1) * gap, nc * q + (nc - 1) * gap), dtype=bool)
    for r, c in nodes:
        r, c = r * (q + gap), c * (q + gap)
        mask[r:r + q, c:c + q] = True
    for i, (a, b) in enumerate(links):
        width = int(rng.integers(1, q + 1))
        if cross:
            width = 1 if i == 0 else int(rng.integers(k, q + 1))
        offset = int(rng.integers(q - width + 1))
        if a[0] == b[0]:
            r = a[0] * (q + gap) + offset
            lo, hi = sorted((a[1], b[1]))
            mask[r:r + width, lo * (q + gap) + q:hi * (q + gap)] = True
        else:
            c = a[1] * (q + gap) + offset
            lo, hi = sorted((a[0], b[0]))
            mask[lo * (q + gap) + q:hi * (q + gap), c:c + width] = True
    # A one-cell dead end may be visible inside an unused lattice connection.
    if not cross and rng.random() < .45:
        choices = [(a, (a[0] + dr, a[1] + dc)) for a in nodes for dr, dc in _FOUR
                   if 0 <= a[0] + dr < nr and 0 <= a[1] + dc < nc
                   and (a[0] + dr, a[1] + dc) not in nodes]
        if choices:
            a, b = choices[int(rng.integers(len(choices)))]
            r, c = a[0] * (q + gap), a[1] * (q + gap)
            mid = int(rng.integers(q))
            if a[0] == b[0]:
                col = c + q if b[1] > a[1] else c - 1
                mask[r + mid, col] = True
            else:
                row = r + q if b[0] > a[0] else r - 1
                mask[row, c + mid] = True
    points = np.argwhere(mask)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    start = (start_node[0] * (q + gap) + 1 - int(lo[0]),
             start_node[1] * (q + gap) + 1 - int(lo[1]))
    return mask[lo[0]:hi[0], lo[1]:hi[1]], start


def _small_object(rng, color, seeded=True, opening=False, k=1):
    q = k + 2
    kind = int(rng.integers(3)) if k == 1 and not opening else 0
    if kind == 1:
        # A diagonal contour genuinely closes a four-connected cavity.
        rr, cc = np.indices((5, 5))
        mask = abs(rr - 2) + abs(cc - 2) <= 2
        start = (2, 2)
    else:
        mask = np.ones((q, q + int(rng.integers(2))), dtype=bool)
        if kind == 2:
            mask[-1, -1] = False
        start = (1, 1)
    return _transform(rng, _object(mask, color, start if seeded else None, k, opening))


def _key_and_palette(rng, lesson):
    pool = list(map(int, rng.permutation(np.arange(2, 10))))
    n = int(rng.integers(3, 7))
    sources, outside = pool[:n], pool[n:]
    fills = list(map(int, rng.choice(sources + outside[:1], n)))
    if lesson == "contrast":
        fills[0], fills[1] = sources[1], outside[0]
    elif lesson == "evidence":
        fills[0], fills[1] = outside[0], sources[1]
    else:
        # Extra scenes need no fixed active entry or destination type. Permit
        # their primary chamber to exercise chains, self-maps and shared fills.
        if outside[0] not in fills:
            fills[-1] = outside[0]
    mapping = dict(zip(sources, fills))
    # Two colors always remain outside the source lane; one is absent from both.
    return mapping, sources, outside


def _pack(rng, patches, mapping):
    """Free placement with local key isolation; a finite shelf fallback."""
    entries = list(mapping.items())
    entries = [entries[i] for i in rng.permutation(len(entries))]
    mh, mw = max(a.shape[0] for a in patches), max(a.shape[1] for a in patches)
    area = sum(a.size for a in patches) + 2 * len(entries)
    for attempt in range(36):
        if attempt >= 24:
            h = w = 30
        else:
            h = int(rng.integers(max(8, mh), 31))
            w = int(rng.integers(max(8, mw, len(entries)), 31))
            if h * w < area * 1.2:
                continue
        grid = np.zeros((h, w), dtype=np.int8)
        offset = int(rng.integers(w - len(entries) + 1))
        for j, (source, fill) in enumerate(entries):
            grid[0, offset + j], grid[1, offset + j] = source, fill
        blocked = _dilate(grid != 0, diagonal=True)
        for i in sorted(range(len(patches)), key=lambda j: -patches[j].size):
            patch = patches[i]
            ph, pw = patch.shape
            possible = np.argwhere(~np.lib.stride_tricks.sliding_window_view(blocked, (ph, pw)).any(axis=(-1, -2)))
            if len(possible) == 0:
                break
            border = possible[(possible[:, 0] == 0) | (possible[:, 1] == 0)
                              | (possible[:, 0] + ph == h) | (possible[:, 1] + pw == w)]
            choices = border if len(border) and rng.random() < .35 else possible
            r, c = choices[int(rng.integers(len(choices)))]
            grid[r:r + ph, c:c + pw] = patch
            blocked[max(0, r - 1):min(h, r + ph + 1), max(0, c - 1):min(w, c + pw + 1)] = True
        else:
            return grid
    grid = np.zeros((30, 30), dtype=np.int8)
    for j, (source, fill) in enumerate(entries):
        grid[0, j], grid[1, j] = source, fill
    r, c, shelf = 3, 0, 0
    for patch in sorted(patches, key=lambda a: (-a.shape[0], -a.shape[1])):
        ph, pw = patch.shape
        if c + pw > 30:
            r, c, shelf = r + shelf + 1, 0, 0
        _require(r + ph <= 30, "shelf fallback exceeds bounds")
        grid[r:r + ph, c:c + pw] = patch
        c += pw + 1
        shelf = max(shelf, ph)
    return grid


def _make_pair(rng, lesson, side, k=2, wanted=4):
    mapping, sources, outside = _key_and_palette(rng, lesson)
    if lesson == "contrast":
        patches = _neck_contrast(rng, sources[0])
    elif lesson == "evidence":
        cavity, start = _graph_cavity(rng, 2, 5, cross=True)
        patches = [_transform(rng, _object(cavity, sources[0], start, 2))]
        patches += [_small_object(rng, sources[1]),
                    _small_object(rng, sources[0], opening=True, k=2),
                    _small_object(rng, outside[int(rng.integers(2))]),
                    _small_object(rng, sources[0], seeded=False)]
    else:
        cavity, start = _graph_cavity(rng, k, wanted)
        primary_color = int(rng.choice(sources))
        patches = [_transform(rng, _object(cavity, primary_color, start, k,
                                          thick=bool(k <= 2 and wanted <= 4 and rng.random() < .2)))]
        for _ in range(int(rng.integers(0, 4))):
            role = int(rng.integers(4))
            color = int(rng.choice(sources)) if role != 2 else int(rng.choice(outside))
            patches.append(_small_object(rng, color, seeded=role != 3, opening=role == 1))
    grid = _pack(rng, patches, mapping)
    # Rotate the whole scene to give the requested boundary-to-inward direction.
    grid = np.rot90(grid, {"top": 0, "left": 1, "bottom": 2, "right": 3}[side]).copy()
    palette = rng.permutation(10).astype(np.int8)
    grid = palette[grid]
    out, info = _run(grid)
    _require(any(r["active"] and r["swept_count"] > 0 for r in info["regions"]), "empty fill lesson")
    return {"input": grid, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(name + " must be an integer in " + str((lo, hi)))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    sides = list(map(str, rng.permutation(["top", "left", "bottom", "right"])))
    shift = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    specs = [("train", "contrast", sides[0], 2, 2),
             ("train", "evidence", sides[1], 2, 5)]
    for i in range(num_train - 2):
        specs.append(("train", "free", str(rng.choice(sides[:2])), int(rng.integers(1, 4)), int(rng.integers(2, 6))))
    for i in range(num_test):
        k = 4 if shift == 0 else int(rng.integers(1, 4))
        wanted = int(rng.integers(6, 10)) if shift == 1 else int(rng.integers(2, 5))
        side = sides[2 + i % 2] if shift == 2 else str(rng.choice(sides))
        specs.append(("test", "free", side, k, wanted))
    for split, lesson, side, k, wanted in specs:
        for _ in range(48):
            pair = _make_pair(rng, lesson, side, k, wanted)
            signature = _fingerprint(pair["input"])
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("distinct input construction exhausted")
    puzzle["train"] = [puzzle["train"][i] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _region_signature(mask):
    points = np.argwhere(mask)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return _fingerprint(mask[lo[0]:hi[0], lo[1]:hi[1]])


def _features(inp, info):
    mapping = info["key"]["mapping"]
    active = [r for r in info["regions"] if r["active"]]
    closed = [r for r in info["regions"] if not r["edge"]]
    diagonal = False
    for r in active:
        boundary = np.zeros(inp.shape, dtype=bool)
        for p in r["boundary"]:
            boundary[p] = True
        diagonal |= len(_components(boundary)) > 1
    return {"shape": inp.shape, "side": info["key"]["side"],
            "background": info["background"], "brush": info["brush"],
            "key_length": len(mapping), "brush_sizes": tuple(k for p, k in info["squares"]),
            "brush_count": len(info["squares"]), "cavity_count": len(closed),
            "partial": sum(r["partial"] for r in active),
            "full": sum(not r["partial"] for r in active),
            "remote_fit": any(r["remote_fit"] for r in active),
            "branch": any(r["branch"] for r in active), "diagonal": diagonal,
            "open_seeded": any(r["edge"] and r["seeds"] for r in info["regions"]),
            "unlisted_seeded": any(r["seeds"] and r["owner"] not in mapping for r in closed),
            "unseeded_mapped": any(not r["seeds"] and r["owner"] in mapping for r in closed),
            "chain": any(mapping[r["owner"]] in mapping and mapping[mapping[r["owner"]]] != mapping[r["owner"]]
                         for r in active),
            "self_map": any(mapping[r["owner"]] == r["owner"] for r in active),
            "painted": sum(r["swept_count"] for r in active)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, features, training = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape preservation")
            signature = _fingerprint(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            expected, info = _run(inp)
            _require(np.array_equal(out, expected), "incorrect sweep or changed scenery")
            _require(np.array_equal(inp[inp != info["background"]], out[inp != info["background"]]), "ink preservation")
            _require(np.array_equal(inp[info["key_mask"]], out[info["key_mask"]]), "key preservation")
            _require(np.any(inp != out), "each example must demonstrate painting")
            f = _features(inp, info)
            features.append(f)
            if split == "train":
                training.append((inp, out, info, f))
    for policy in ("point", "source", "recursive", "seedless"):
        _require(any(not np.array_equal(_alternative(inp, policy), out)
                     for inp, out, info, f in training), "missing counterexample to " + policy)
    for feature in ("branch", "diagonal", "open_seeded", "unlisted_seeded", "unseeded_mapped", "chain", "self_map", "remote_fit"):
        _require(any(f[feature] for inp, out, info, f in training), "missing training witness: " + feature)
    contrast = False
    for inp, out, info, f in training:
        active = [r for r in info["regions"] if r["active"]]
        for a in active:
            for b in active:
                contrast |= (a["owner"] == b["owner"] and _region_signature(a["mask"]) == _region_signature(b["mask"])
                             and a["seeds"][0][1] == 1 and b["seeds"][0][1] > 1
                             and not a["partial"] and b["partial"] and b["remote_fit"])
    _require(contrast, "same-geometry size contrast required")
    _require(len({f["side"] for inp, out, info, f in training}) >= 2, "two key directions must be demonstrated")


def _puzzle_signature(puzzle):
    return tuple(_fingerprint(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))


def self_test():
    """Bounded 42-seed checks plus literal geometry and mutation witnesses."""
    # Hand-specified dumbbell, not a sampled path: a 2-square cannot cross
    # row 5's one-cell neck, despite the remote 4x4 room fitting it.
    for k in (1, 2):
        inp = np.zeros((12, 18), dtype=np.int8)
        inp[0, 2:5], inp[1, 2:5] = [2, 3, 4], [3, 5, 4]
        cavity = np.zeros(inp.shape, dtype=bool)
        cavity[4:8, 2:6] = cavity[4:8, 9:13] = True
        cavity[5, 6:9] = True
        inp[_dilate(cavity) & ~cavity] = 2
        inp[5:5 + k, 3:3 + k] = 1
        expected = inp.copy()
        painted = cavity.copy() if k == 1 else np.zeros(inp.shape, dtype=bool)
        if k == 2:
            painted[4:8, 2:6] = True
        expected[painted & (inp == 0)] = 3
        _require(np.array_equal(solve(inp), expected), "literal bottleneck witness")
    # Pixel enclosure and square enclosure differ: this one-cell opening leaks
    # background, although a two-square remains trapped inside the left room.
    inp[3, 3] = 0
    trapped, _, _ = _sweep((inp == 0) | (inp == 1), (5, 3), 2)
    _require(not trapped[0].any() and not trapped[-1].any()
             and not trapped[:, 0].any() and not trapped[:, -1].any(), "opening must trap the square")
    _require(np.array_equal(solve(inp), inp), "an open chamber must not paint even with a trapped brush")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, "cross-seed puzzle repetition")
    shapes, structures, records = set(), set(), []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                expected, info = _run(inp)
                _require(np.array_equal(out, expected), "solver disagreement")
                # Rotations, reflection and color bijection preserve the rule.
                transformed = (np.rot90(np.fliplr(inp)) + 3) % 10
                target = (np.rot90(np.fliplr(out)) + 3) % 10
                _require(np.array_equal(solve(transformed), target), "role/orientation equivariance")
                f = _features(inp, info)
                f["split"] = split
                records.append(f)
                shapes.add(inp.shape)
                palette, normalized = {}, []
                for value in inp.flat:
                    normalized.append(palette.setdefault(int(value), len(palette)))
                structures.add((inp.shape, tuple(normalized)))
    _require(len(shapes) >= 35 and len(structures) >= 150, "geometry diversity")
    _require({f["side"] for f in records} == {"top", "left", "bottom", "right"}, "key sides")
    _require({f["background"] for f in records} == set(range(10)), "background palette")
    _require({f["brush"] for f in records} == set(range(10)), "tool palette")
    _require({k for f in records for k in f["brush_sizes"]} == {1, 2, 3, 4}, "size variation")
    _require({f["key_length"] for f in records} == {3, 4, 5, 6}, "key length variation")
    _require(len({f["cavity_count"] for f in records}) >= 4, "cavity count variation")
    _require(any(f["partial"] for f in records if f["split"] == "test"), "partial test fills")
    _require(any(f["full"] for f in records if f["split"] == "test"), "full test fills")
    _require(_puzzle_signature(puzzles[0]) == _puzzle_signature(generate_puzzle(0)), "seed reproducibility")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(73, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject wrong paint, changed key and erased size marker in returned labels.
    for mutation in ("paint", "key", "brush"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        pair = bad["test"][0]
        info = _read_input(pair["input"])
        if mutation == "paint":
            p = tuple(np.argwhere(pair["input"] != pair["output"])[0])
        elif mutation == "key":
            p = info["key"]["cells"][0]
        else:
            p = info["squares"][0][0]
        pair["output"][p] = (int(pair["output"][p]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + mutation + " mutation")
    return {"seeds": 42, "pairs": len(records), "canvas_shapes": len(shapes),
            "color_normalized_inputs": len(structures), "brush_sizes": [1, 2, 3, 4],
            "key_lengths": [3, 4, 5, 6],
            "semantic_checks": "generator-owned; no independent solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
