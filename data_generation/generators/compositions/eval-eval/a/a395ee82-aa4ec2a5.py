"""Rooted chamber loom: anchored stencils, collective enclosure, rooted filling.

All semantic information is recovered from the input. The sampler deliberately
teaches small, disconnected networks before testing a larger joined network.
"""
from __future__ import annotations

from collections import deque
import itertools
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "illegal grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(((0 <= a) & (a <= 9)).all()), "illegal color")
    return a.astype(np.int8, copy=False)


def _regions(mask, neighbors=N4):
    """Deterministic component labels and coordinate lists."""
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    regions = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        label = len(regions)
        labels[r, c] = label
        cells, queue = [], deque([(r, c)])
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for dr, dc in neighbors:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                    labels[nr, nc] = label
                    queue.append((nr, nc))
        regions.append(cells)
    return labels, regions


def _holes(mask, neighbors=N4):
    # Padding means the canvas edge never acts as an extra wall.
    p = np.pad(mask, 1, constant_values=False)
    exterior = np.zeros_like(p)
    exterior[0, 0] = True
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < p.shape[0] and 0 <= cc < p.shape[1] and not p[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                queue.append((rr, cc))
    return (~p & ~exterior)[1:-1, 1:-1]


def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _clear_diagonals(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not bool(np.any((a == d) & (b == c) & (a != b)))


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in (
        np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2),
        mask.T, np.flipud(mask).T, np.fliplr(mask).T, np.rot90(mask, 2).T))


def _separated_runs(mask):
    return any(len(ix := np.flatnonzero(row)) > 1 and np.any(np.diff(ix) > 1)
               for rows in (mask, mask.T) for row in rows)


def _decode(input_grid):
    inp = _grid(input_grid)
    values, counts = np.unique(inp, return_counts=True)
    _require(len(values) == 3, "expected background, stencil, and map colors")
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be uniquely dominant")
    bg = int(values[np.argmax(counts)])
    candidates = []
    comps = {}
    for color in values:
        if color == bg:
            continue
        comps[int(color)] = _regions(inp == color)[1]
        sizes = sorted(map(len, comps[int(color)]))
        if len(sizes) == 2 and sizes[0] == 1 and sizes[1] >= 3:
            candidates.append(int(color))
    _require(len(candidates) == 1, "stencil and anchor must be identifiable")
    ink = candidates[0]
    marker = next(int(v) for v in values if v not in (bg, ink))
    glyph = next(c for c in comps[ink] if len(c) > 1)
    anchor_pixel = next(c[0] for c in comps[ink] if len(c) == 1)
    _require(all(len(c) == 1 for c in comps[marker]), "ordinary map points must be isolated")
    coords = np.asarray(glyph)
    tr, tc = map(int, coords.min(axis=0))
    br, bc = map(int, coords.max(axis=0))
    mask = inp[tr:br + 1, tc:bc + 1] == ink
    h, w = mask.shape
    _require(min(h, w) >= 2 and not _holes(mask).any(), "stencil must be open or solid, with no cavity")
    _require(_clear_diagonals(mask), "ambiguous stencil diagonal")
    points = [anchor_pixel] + [c[0] for c in comps[marker]]
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    _require(br + 1 < pr or er + 1 < tr or bc + 1 < pc or ec + 1 < tc,
             "map and stencil bounding boxes need a blank separating row or column")
    _require(all((r - pr) % 2 == 0 and (c - pc) % 2 == 0 for r, c in points), "map pitch is two")
    layout = np.zeros(((er - pr) // 2 + 1, (ec - pc) // 2 + 1), bool)
    for r, c in points:
        layout[(r - pr) // 2, (c - pc) // 2] = True
    anchor = ((anchor_pixel[0] - pr) // 2, (anchor_pixel[1] - pc) // 2)
    top, left = tr - anchor[0] * h, tc - anchor[1] * w
    _require(top >= 0 and left >= 0 and top + h * layout.shape[0] <= inp.shape[0]
             and left + w * layout.shape[1] <= inp.shape[1], "expanded assembly must fit without clipping")
    wall = np.zeros(inp.shape, bool)
    stamp = np.kron(layout, mask)
    wall[top:top + stamp.shape[0], left:left + stamp.shape[1]] = stamp
    labels, components = _regions(wall)
    mr, mc = np.argwhere(mask)[0]
    root = int(labels[tr + mr, tc + mc])
    root_body = labels == root
    root_holes = _holes(root_body)
    _require(not np.any(root_holes & wall), "nested wall networks are outside the family")
    return dict(background=bg, ink=ink, marker=marker, mask=mask, layout=layout,
                anchor=anchor, template_origin=(tr, tc), assembly_origin=(top, left),
                map_points=points, wall=wall, labels=labels, components=components,
                root=root, root_body=root_body, root_holes=root_holes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct and inspect only the network containing the anchored copy."""
    m = _decode(input_grid)
    out = np.full(m["wall"].shape, m["background"], dtype=np.int8)
    out[m["wall"]] = m["ink"]
    if m["root_holes"].any():
        out[m["root_body"]] = m["marker"]
        out[m["root_holes"]] = m["ink"]
    return out


def _choice(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _grow(rng, h, w, count, square=False):
    mask = np.zeros((h, w), bool)
    if square:
        r, c = int(rng.integers(h - 1)), int(rng.integers(w - 1))
        mask[r:r + 2, c:c + 2] = True
    else:
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
    for _ in range(count - int(mask.sum())):
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in N4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    candidate = mask.copy()
                    candidate[rr, cc] = True
                    if _clear_diagonals(candidate):
                        frontier.add((rr, cc))
        if not frontier:
            break
        mask[_choice(rng, sorted(frontier))] = True
    return _crop(mask)


def _orient(rng, mask):
    m = np.rot90(mask, int(rng.integers(4)))
    return np.fliplr(m).copy() if rng.integers(2) else m.copy()


def _template(rng, previous=None, teaching=0):
    """Sample geometry; teaching stencils exhibit collective micro-cavities."""
    options = [(h, w) for h in range(2, 6) for w in range(2, 6)
               if (previous is None or (h != previous[0] and w != previous[1]))
               and (teaching != 1 or h != w)
               and (teaching != 2 or min(h, w) >= 3)]
    for attempt in range(80):
        h, w = _choice(rng, options)
        style = int(rng.integers(5))
        if style in (0, 1):
            mask = np.zeros((h, w), bool)
            mask[:, :int(rng.integers(1, w))] = True
            mask[-int(rng.integers(1, h)):, :] = True
            if style == 1:
                mask[:int(rng.integers(1, h - 1)) if h > 2 else 1, :] = True
        else:
            mask = _grow(rng, h, w, int(rng.integers(h + w - 1, h * w + 1)))
            if mask.shape != (h, w):
                continue
        mask = _orient(rng, mask)
        if previous is not None and (mask.shape[0] == previous[0] or mask.shape[1] == previous[1]):
            continue
        mask |= _holes(mask)
        joins = bool(np.any(mask[0] & mask[-1]) and np.any(mask[:, 0] & mask[:, -1]))
        if not joins or not _clear_diagonals(mask):
            continue
        tiled = np.tile(mask, (2, 2))
        ring = np.ones((3, 3), bool)
        ring[1, 1] = False
        if not _clear_diagonals(tiled) or not _clear_diagonals(np.kron(ring, mask)):
            continue
        if teaching:
            if not _holes(tiled).any():
                continue
            if teaching == 1 and (not _asymmetric(mask) or mask.shape[0] == mask.shape[1]):
                continue
            if teaching == 2 and not _separated_runs(mask):
                continue
        return mask
    # Finite construction fallback, preserving the teaching predicates.
    if teaching == 2:
        h, w = next((a, b) for a in (3, 4, 5) for b in (3, 4, 5)
                    if previous is None or (a != previous[0] and b != previous[1]))
        mask = np.ones((h, w), bool)
        mask[1:-1, 1:] = False
    else:
        h, w = (3, 4)
        mask = np.zeros((h, w), bool)
        mask[:, 0] = True
        mask[-1] = True
    return mask


def _small_cluster(rng, hollow=False):
    if hollow:
        h, w = _choice(rng, [(2, 2), (2, 3), (3, 2)])
        return _grow(rng, h, w, int(rng.integers(4, h * w + 1)), square=True)
    h, w = _choice(rng, [(1, 1), (1, 3), (3, 1), (2, 3), (3, 2)])
    return _grow(rng, h, w, int(rng.integers(1, min(6, h * w) + 1)))


def _training_layout(rng, mask, teaching, previous_anchor=None):
    # Two freely oriented clusters, separated by one or two blank lattice bands.
    # Small groups can have branches and offsets; they are not output panels.
    for _ in range(100):
        a = _small_cluster(rng, hollow=teaching > 0)
        b = np.ones((1, 1), bool) if teaching == 2 else _small_cluster(rng, hollow=teaching == 1)
        gap = int(rng.integers(1, 3))
        height = max(a.shape[0], b.shape[0]) + int(rng.integers(2))
        layout = np.zeros((height, a.shape[1] + gap + b.shape[1]), bool)
        ra = int(rng.integers(height - a.shape[0] + 1))
        rb = int(rng.integers(height - b.shape[0] + 1))
        layout[ra:ra + a.shape[0], :a.shape[1]] = a
        layout[rb:rb + b.shape[0], a.shape[1] + gap:] = b
        layout = _crop(_orient(rng, layout))
        if any(d * s > 28 for d, s in zip(layout.shape, mask.shape)):
            continue
        wall = np.kron(layout, mask)
        if not _clear_diagonals(wall):
            continue
        labels, comps = _regions(wall)
        cavity_counts = [len(_regions(_holes(labels == i))[1]) for i in range(len(comps))]
        mr, mc = np.argwhere(mask)[0]
        choices = []
        for r, c in np.argwhere(layout):
            label = labels[r * mask.shape[0] + mr, c * mask.shape[1] + mc]
            hollow = cavity_counts[label] > 0
            if teaching and hollow != (teaching == 1):
                continue
            if teaching == 1 and cavity_counts[label] < 2:
                continue
            anchor = int(r), int(c)
            if previous_anchor is not None and (anchor[0] == previous_anchor[0] or anchor[1] == previous_anchor[1]):
                continue
            if teaching == 1 and anchor == ((layout.shape[0] - 1) // 2, (layout.shape[1] - 1) // 2):
                continue
            choices.append(anchor)
        if choices and len(comps) >= 2:
            return layout, _choice(rng, choices)
    # Fits every stencil size <=5, and contrasts both anchor coordinates.
    if teaching == 2:
        base = np.array([[1, 1], [1, 1], [0, 0], [0, 1]], bool)
        choices = []
        for turns in range(4):
            for flip in (False, True):
                layout = np.rot90(base, turns)
                if flip:
                    layout = np.fliplr(layout)
                _, components = _regions(layout)
                anchor = next(c[0] for c in components if len(c) == 1)
                if previous_anchor is None or all(a != b for a, b in zip(anchor, previous_anchor)):
                    choices.append((layout.copy(), anchor))
        layout, anchor = choices[0]
    else:
        layout = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0],
                           [1, 1, 1], [1, 1, 1]], bool)
        anchor = (0, 0)
    return layout, anchor


def _test_layout(rng, mask, require_hollow):
    h, w = mask.shape
    options = [(a, b) for a in range(3, min(6, 28 // h) + 1)
               for b in range(3, min(6, 28 // w) + 1)]
    for _ in range(100):
        a, b = _choice(rng, options)
        count = int(rng.integers(7, min(20, a * b) + 1))
        layout = _grow(rng, a, b, count, square=require_hollow)
        wall = np.kron(layout, mask)
        if int(layout.sum()) < 7 or not _clear_diagonals(wall):
            continue
        if require_hollow and not _holes(wall).any():
            continue
        return layout, _choice(rng, [tuple(map(int, p)) for p in np.argwhere(layout)])
    # A 3x3 lattice ring has a macro-cavity for every supported joining stencil.
    layout = np.ones((3, 3), bool)
    layout[1, 1] = False
    _require(_holes(np.kron(layout, mask)).any() or not require_hollow, "test construction failed")
    return layout, (0, 0)


def _place(rng, mask, layout, anchor, colors):
    h, w = mask.shape
    ah, aw = h * layout.shape[0], w * layout.shape[1]
    ph, pw = 2 * layout.shape[0] - 1, 2 * layout.shape[1] - 1
    tr, tc = anchor[0] * h, anchor[1] * w
    positions = []
    for r in range(-ph - 1, ah + 2):
        for c in range(-pw - 1, aw + 2):
            if not (r + ph < tr or tr + h < r or c + pw < tc or tc + w < c):
                continue
            lo_r, lo_c = min(0, r), min(0, c)
            bh, bw = max(ah, r + ph) - lo_r, max(aw, c + pw) - lo_c
            if bh <= 30 and bw <= 30:
                positions.append((r, c, lo_r, lo_c, bh, bw))
    _require(bool(positions), "no legal map placement")
    if rng.random() < .5:
        best = min(p[4] * p[5] for p in positions)
        positions = [p for p in positions if p[4] * p[5] <= best * 1.35]
    pr, pc, lo_r, lo_c, bh, bw = _choice(rng, positions)
    gh, gw = int(rng.integers(bh, 31)), int(rng.integers(bw, 31))
    top, left = int(rng.integers(gh - bh + 1)) - lo_r, int(rng.integers(gw - bw + 1)) - lo_c
    bg, ink, marker = colors
    inp = np.full((gh, gw), bg, np.int8)
    rr, cc = np.nonzero(mask)
    inp[top + tr + rr, left + tc + cc] = ink
    for r, c in np.argwhere(layout):
        inp[top + pr + 2 * r, left + pc + 2 * c] = ink if (r, c) == anchor else marker
    return inp


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    fixed_palette = rng.random() < .3
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    previous_dims = previous_anchor = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = index + 1 if split == "train" and index < 2 else 0
            mask = _template(rng, previous_dims if teaching == 2 else None, teaching)
            if split == "train":
                layout, anchor = _training_layout(rng, mask, teaching, previous_anchor if teaching == 2 else None)
            else:
                layout, anchor = _test_layout(rng, mask, require_hollow=index == 0)
            colors = palette if fixed_palette else tuple(map(int, rng.choice(10, 3, replace=False)))
            for _ in range(32):
                inp = _place(rng, mask, layout, anchor, colors)
                if _key(inp) not in seen:
                    break
            else:
                # Palette is visible, so a finite role permutation is semantic.
                # 720 triples exceed the maximum fourteen requested inputs.
                for colors in itertools.permutations(range(10), 3):
                    remap = np.arange(10, dtype=np.int8)
                    m = _decode(inp)
                    remap[[m["background"], m["ink"], m["marker"]]] = colors
                    candidate = remap[inp]
                    if _key(candidate) not in seen:
                        inp = candidate
                        break
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": solve(inp)})
            if teaching == 1:
                previous_dims, previous_anchor = mask.shape, anchor
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    m = _decode(inp)
    wall, labels = m["wall"], m["labels"]
    holes = []
    all_holes = np.zeros_like(wall)
    for i in range(len(m["components"])):
        hole = _holes(labels == i)
        _require(not np.any(hole & wall) and not np.any(hole & all_holes), "nested or shared cavity")
        _require(np.array_equal(hole, _holes(labels == i, N8)), "diagonal-only cavity leak")
        all_holes |= hole
        holes.append(hole)
    _require(_clear_diagonals(wall), "ambiguous assembled diagonal")
    _require(len(m["components"]) == len(_regions(wall, N8)[1]), "diagonal-only wall contact")
    _require(np.array_equal(all_holes, _holes(wall)), "cavity ownership needs a unique network")
    nr, nc = m["assembly_origin"]
    h, w = m["mask"].shape
    mr, mc = np.argwhere(m["mask"])[0]
    copy_labels = [int(labels[nr + r * h + mr, nc + c * w + mc])
                   for r, c in np.argwhere(m["layout"])]
    root_copies = copy_labels.count(m["root"])
    micro = any(m["root_holes"][nr + r * h:nr + (r + 1) * h,
                               nc + c * w:nc + (c + 1) * w].any()
                for r, c in np.argwhere(m["layout"]))
    return dict(m, holes=holes, all_holes=all_holes, root_copies=root_copies,
                copy_counts=tuple(copy_labels.count(i) for i in range(len(holes))),
                root_hollow=bool(m["root_holes"].any()), micro_cavity=micro,
                decoy_hollow=any(hole.any() for i, hole in enumerate(holes) if i != m["root"]),
                cavity_count=len(_regions(m["root_holes"])[1]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    features, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            m = _features(inp)
            # Explicit semantic zones check whole-network propagation, cavity
            # filling, unselected networks, and every remaining blank cell.
            expected = np.full_like(inp, m["background"])
            expected[m["wall"]] = m["ink"]
            if m["root_hollow"]:
                expected[m["root_body"]] = m["marker"]
                expected[m["root_holes"]] = m["ink"]
            _require(np.array_equal(out, expected), "wrong wall/cavity/background pixels")
            _require(not np.array_equal(inp, out), "identity pair")
            _require(int(m["wall"].sum()) == int(m["mask"].sum() * m["layout"].sum()), "copies overlap or lose mask pixels")
            features[split].append(m)
    a, b = features["train"][:2]
    _require(a["root_hollow"] and a["cavity_count"] >= 2 and a["micro_cavity"] and a["decoy_hollow"],
             "first teaching scene needs multiple emergent and unselected cavities")
    _require(not b["root_hollow"] and b["decoy_hollow"] and _separated_runs(b["mask"]), "second teaching scene needs an open root and closed decoy")
    _require(_asymmetric(a["mask"]) and a["mask"].shape[0] != a["mask"].shape[1], "exact rectangular stencil copying needs evidence")
    _require(all(x != y for x, y in zip(a["mask"].shape, b["mask"].shape)), "both pitches need to vary in training")
    _require(a["anchor"][0] != b["anchor"][0] and a["anchor"][1] != b["anchor"][1], "both anchor coordinates need to vary")
    _require(all(len(m["components"]) >= 2 and max(m["copy_counts"]) <= 6 for m in features["train"]), "training must use small separate networks")
    _require(all(len(m["components"]) == 1 and m["root_copies"] >= 7 for m in features["test"]), "test must join a larger network")
    _require(features["test"][0]["root_hollow"], "first test must exercise both parent concepts")
    return None


def self_test():
    """Bounded 42-seed semantic diversity and counterfactual regression suite."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, masks, layouts, anchors, roots, cavities, palettes = (set() for _ in range(8))
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in puzzle:
            for pair in puzzle[split]:
                m = _features(pair["input"])
                records.append(m)
                shapes.add(pair["input"].shape)
                masks.add(_key(m["mask"]))
                layouts.add(_key(m["layout"]))
                anchors.add(m["anchor"])
                roots.add(m["root_copies"])
                cavities.add(m["cavity_count"])
                palettes.add((m["background"], m["ink"], m["marker"]))
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list input solve mismatch")
    _require(len(signatures) == 42 and len(shapes) >= 35, "puzzle/canvas diversity missing")
    _require(len(masks) >= 35 and len(layouts) >= 55 and len(anchors) >= 15, "semantic geometric diversity missing")
    _require(len(roots) >= 8 and len(cavities) >= 4 and len(palettes) >= 35, "network/topology/palette variation missing")
    _require(any(m["mask"].all() for m in records), "solid stencil variant missing")
    _require(any(_holes(m["layout"]).any() for m in records), "macro-cavity variation missing")
    _require(any(m["cavity_count"] >= 3 for m in records), "multiple chambers missing")
    a, b = generate_puzzle(1234, 2, 1), generate_puzzle(1234, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "nondeterministic puzzle")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(700 + nt * 10 + nq, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # B ablation: simply copying all walls misses the selected body and fill.
    pair = puzzles[0]["train"][0]
    m = _features(pair["input"])
    plain = np.where(m["wall"], m["ink"], m["background"])
    _require(not np.array_equal(plain, pair["output"]), "topology contribution is dispensable")
    # Root ablation: every hollow network filled would alter the closed decoy.
    decoy = next(h for i, h in enumerate(m["holes"]) if i != m["root"] and h.any())
    _require(np.all(pair["output"][decoy] == m["background"]), "closed decoy was filled")
    # Move only the exceptional point; both possible assemblies fit the canvas.
    mask = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], bool)
    layout = np.array([[1, 1, 0, 1], [1, 1, 0, 0]], bool)
    fixture = np.zeros((18, 24), np.int8)
    fixture[6:9, 12:15][mask] = 2
    for r, c in np.argwhere(layout):
        fixture[12 + 2 * r, 2 + 2 * c] = 2 if (r, c) == (0, 0) else 7
    prediction = solve(fixture)
    swapped = fixture.copy()
    swapped[12, 2], swapped[12, 8] = 7, 2
    other_prediction = solve(swapped)
    _require(not np.array_equal(prediction, other_prediction), "color-defined root/anchor has no effect")
    _require(not _decode(swapped)["root_holes"].any(), "root swap should select the open singleton")
    # Malformed labels must be caught; no external semantic oracle is claimed.
    corrupt = {s: [{k: np.array(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    rr, cc = np.argwhere(m["root_holes"])[0]
    corrupt["train"][0]["output"][rr, cc] = m["background"]
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted corrupted cavity fill")
