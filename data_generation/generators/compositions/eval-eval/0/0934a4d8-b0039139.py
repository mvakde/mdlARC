"""Mirror-window census printing: input-only geometry, topology, and stamping.

The window is both the source of the stamp and a selector of WHOLE objects.
No parent generator, file, seed, or demonstration is consulted by solve().
"""
from __future__ import annotations

import copy
import json
from collections import deque

import numpy as np


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "illegal grid shape")
    _check(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _check(bool(np.all((0 <= a) & (a <= 9))), "illegal grid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask, diagonal=False):
    """Return coordinate sets; unlike color objects, all foreground is joined."""
    todo = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    groups = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, queue = {start}, deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    queue.append(p)
        groups.append(group)
    return groups


def _bbox(mask):
    points = np.argwhere(mask)
    _check(len(points) > 0, "empty stencil")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _crop(mask):
    r, s, c, d = _bbox(mask)
    return mask[r:s, c:d].copy()


def _decode(input_grid):
    """Locate the mask and establish the unique centered reflection from pixels.

    Two possible axial reflections are checked in full, with masked pairs
    ignored. Every masked mate must be visible. No output is used to choose
    an axis, identify a color, or determine the bounding boxes.
    """
    a = _grid(input_grid)
    corners = [int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])]
    _check(len(set(corners)) == 1, "corners must identify the background")
    bg = corners[0]
    colors = set(map(int, np.unique(a)))
    _check(len(colors) == 3, "expected background, scene ink, and window ink")
    rectangles = []
    for color in colors - {bg}:
        box = _bbox(a == color)
        r, s, c, d = box
        if (s-r)*(d-c) == int(np.count_nonzero(a == color)) and (s-r)*(d-c) >= 2:
            rectangles.append((color, box))
    _check(len(rectangles) == 1, "window must be the unique solid rectangular color")
    pigment, box = rectangles[0]
    ink = next(iter(colors - {bg, pigment}))
    erased = a == pigment
    candidates = []
    for axis in (0, 1):
        mate = np.flip(a, axis=axis)
        mate_erased = np.flip(erased, axis=axis)
        if np.any(erased & mate_erased):
            continue
        visible = ~erased & ~mate_erased
        if not np.array_equal(a[visible], mate[visible]):
            continue
        rr, cc = np.indices(a.shape)
        coord = rr if axis == 0 else cc
        nonfixed = coord != a.shape[axis]-1-coord
        if np.count_nonzero(visible & nonfixed & (a == ink)) < 4:
            continue
        restored = a.copy()
        restored[erased] = mate[erased]
        candidates.append((axis, restored))
    _check(len(candidates) == 1, "reflection direction must be uniquely witnessed")
    axis, full = candidates[0]
    r, s, c, d = box
    window = full[r:s, c:d] == ink
    tile = _crop(window)
    groups = _components(full == ink)
    selected = [g for g in groups if any(erased[p] for p in g)]
    _check(len(selected) > 0, "window must intersect foreground objects")
    pieces = []
    for group in selected:
        intersection = np.zeros(a.shape, dtype=bool)
        for point in group:
            intersection[point] = erased[point]
        pieces.append(len(_components(intersection)))
    return {"array": a, "background": bg, "ink": ink, "pigment": pigment,
            "bbox": box, "axis": axis, "full": full, "window": window,
            "tile": tile, "groups": groups, "selected": selected,
            "count": len(selected), "erased": erased, "pieces": tuple(pieces)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore, select whole objects, and print the cropped window design."""
    d = _decode(input_grid)
    tile, axis, n = d["tile"], d["axis"], d["count"]
    shape = list(tile.shape)
    length = tile.shape[axis]
    shape[axis] = n * (length + 1) - 1
    _check(max(shape) <= 30, "printing exceeds the grid budget")
    out = np.full(shape, d["background"], dtype=np.int8)
    # Indexed rendering deliberately differs from the constructor's concatenation.
    for r in range(shape[0]):
        for c in range(shape[1]):
            coord = r if axis == 0 else c
            local = coord % (length + 1)
            if local < length:
                tr, tc = (local, c) if axis == 0 else (r, local)
                if tile[tr, tc]:
                    out[r, c] = d["pigment"]
    return out


def _grow(h, w, area, rng):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            x, y = p
            if 0 <= x < h and 0 <= y < w and not mask[p]:
                frontier.add(p)
    return mask


def _shape(h, w, rng, area=None):
    if area is not None:
        return _grow(h, w, area, rng)
    mode = int(rng.integers(5))
    if mode == 0:
        return np.ones((h, w), dtype=bool)
    if mode == 1 and min(h, w) >= 3:
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
        return mask
    if mode == 2:
        mask = np.zeros((h, w), dtype=bool)
        mask[::2] = True
        for r in range(1, h, 2):
            mask[r, 0 if (r//2) % 2 else w-1] = True
        return mask
    return _grow(h, w, int(rng.integers(1, h*w+1)), rng)


def _positive_partition(total, count, rng):
    values = np.ones(count, dtype=int)
    for _ in range(total-count):
        values[int(rng.integers(count))] += 1
    return list(map(int, values))


def _render_label(full, box, ink, pigment, bg, axis, count):
    """Construction label from the unoccluded scene and designed object count."""
    r, s, c, d = box
    tile = np.where(_crop(full[r:s, c:d] == ink), pigment, bg).astype(np.int8)
    gap_shape = list(tile.shape)
    gap_shape[axis] = 1
    gap = np.full(gap_shape, bg, dtype=np.int8)
    pieces = []
    for i in range(count):
        if i:
            pieces.append(gap)
        pieces.append(tile)
    return np.concatenate(pieces, axis=axis)


def _make_pair(rng, n, fingers, axis, palette, area, seen):
    """Build arbitrary connected objects in disjoint bands, then reflect them.

    'fingers' lists the selected objects whose joining spine is outside the
    window. All other selected objects are wholly hidden. Mirrored copies and
    extra objects supply restoration evidence, never an independent counter.
    """
    bg, ink, pigment = palette
    for _attempt in range(100):
        compact = _attempt >= 60
        axial_max = min(11, 31//n-1)
        width = int(rng.integers(3, min(5, axial_max)+1)) if area is not None else int(rng.integers(1, axial_max+1))
        pad = int(rng.integers(0, 3))
        mask_width = width + pad
        shared_center = bool(rng.integers(2))
        half_w = int(rng.integers(max(6, mask_width+2), 16))
        bridge_right = bool(rng.integers(2))
        if bridge_right:
            # Outside spine may reach the mirror seam and join its reflected copy.
            qleft = int(rng.integers(0, half_w-mask_width))
        else:
            qleft = int(rng.integers(1, half_w-mask_width+(0 if shared_center else 1)))
        qright = qleft + mask_width
        active_left = qright-width if bridge_right else qleft
        active_right = active_left+width
        spine = qright if bridge_right else qleft-1
        descriptors, bridge_area = [], 0
        for f in fingers:
            height = 2*f-1 + (0 if compact else int(rng.integers(0, 3)))
            rows = np.linspace(0, height-1, f).astype(int)
            lengths = [width] + [int(rng.integers(1, width+1)) for _ in range(f-1)]
            rng.shuffle(lengths)
            bridge_area += sum(lengths)
            descriptors.append({"h": height, "rows": rows, "lengths": lengths})
        ordinary_count = n-len(fingers)
        if area is not None:
            if area-bridge_area < ordinary_count:
                continue
            areas = _positive_partition(area-bridge_area, ordinary_count, rng)
        else:
            areas = [None] * ordinary_count
        for amount in areas:
            height = ((amount+width-1)//width + int(rng.integers(0, 2))
                      if amount is not None else int(rng.integers(1, 3 if compact or n >= 5 else 6)))
            descriptors.append({"h": height, "amount": amount})
        rng.shuffle(descriptors)
        gaps = [1 if compact else int(rng.integers(1, 3)) for _ in range(n-1)]
        selected_h = sum(d["h"] for d in descriptors) + sum(gaps)
        distractors = [(int(rng.integers(1, 3 if compact else 5)), int(rng.integers(1, half_w+1)))
                       for _ in range(1 if compact else int(rng.integers(1, 3)))]
        distractor_h = sum(h for h, _ in distractors) + 1*(len(distractors)-1)
        outer_top, outer_bottom = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        outside_gap = int(rng.integers(1, 4))
        height = outer_top + selected_h + outside_gap + distractor_h + outer_bottom
        if height > 30:
            continue
        distractor_first = bool(rng.integers(2))
        start = outer_top + (distractor_h+outside_gap if distractor_first else 0)
        dstart = outer_top if distractor_first else start+selected_h+outside_gap
        top_room = outside_gap-1 if distractor_first else outer_top
        bottom_room = outer_bottom if distractor_first else outside_gap-1
        qtop = start - int(rng.integers(top_room+1))
        qbottom = start+selected_h+int(rng.integers(bottom_room+1))
        half = np.zeros((height, half_w), dtype=bool)
        r0 = start
        for i, desc in enumerate(descriptors):
            h = desc["h"]
            if "rows" in desc:
                half[r0:r0+h, spine] = True
                for row, length in zip(desc["rows"], desc["lengths"]):
                    if bridge_right:
                        half[r0+row, active_right-length:spine+1] = True
                    else:
                        half[r0+row, spine:active_left+length] = True
            else:
                amount = desc["amount"]
                sw = width if amount is not None else int(rng.integers(1, width+1))
                obj = _shape(h, sw, rng, amount)
                c0 = active_left + int(rng.integers(width-sw+1))
                half[r0:r0+h, c0:c0+sw] = obj
            r0 += h + (gaps[i] if i < n-1 else 0)
        r0 = dstart
        for dh, dw in distractors:
            obj = _shape(dh, dw, rng)
            c0 = int(rng.integers(half_w-dw+1))
            half[r0:r0+dh, c0:c0+dw] = obj
            r0 += dh+1
        reflected = half[:, -2::-1] if shared_center else half[:, ::-1]
        field = np.concatenate((half, reflected), axis=1)
        full = np.where(field, ink, bg).astype(np.int8)
        box = qtop, qbottom, qleft, qright
        if bool(rng.integers(2)):
            full = full[:, ::-1].copy()
            box = qtop, qbottom, full.shape[1]-qright, full.shape[1]-qleft
        if axis == 0:
            full = full.T.copy()
            r, s, c, d = box
            box = c, d, r, s
        r, s, c, d = box
        inp = full.copy()
        inp[r:s, c:d] = pigment
        if _key(inp) in seen:
            continue
        try:
            record = _decode(inp)
        except AssertionError:
            continue
        if record["axis"] != axis or record["count"] != n:
            continue
        if len(_components(record["window"])) != n + sum(f-1 for f in fingers):
            continue
        # The two training witnesses teach target orientation and real cropping.
        if area is not None:
            if int(record["window"].sum()) != area:
                continue
            if record["window"].shape[axis] == n:
                continue
            if len(record["groups"]) == 2*n:
                continue
            if np.array_equal(record["tile"], np.flip(record["tile"], axis=axis)):
                continue
            if record["tile"].shape == record["window"].shape:
                continue
        out = _render_label(full, box, ink, pigment, bg, axis, n)
        return {"input": inp, "output": out}
    raise AssertionError("bounded scene construction exhausted")


def _count_arg(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count_arg(num_train, "num_train", 2, 10)
    nq = _count_arg(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_axis = int(rng.integers(2))
    mixed = nt >= 3 and rng.random() < .6
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    recolor = rng.random() < .7
    witness_counts = list(map(int, rng.choice([2, 3, 4], 2, replace=False)))
    witness_area = int(rng.integers(12, 20))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for i in range(amount):
            witness = split == "train" and i < 2
            axis = base_axis
            if mixed and ((split == "train" and i >= 2) or split == "test"):
                axis = 1-base_axis if split == "train" and i == 2 else int(rng.integers(2))
            if witness:
                n, fingers, area = witness_counts[i], [2+i], witness_area
            elif split == "train":
                # A repeated count in the second orientation breaks an axis/count shortcut.
                n = witness_counts[0] if i == 2 else int(rng.integers(1, 7))
                fingers, area = ([] if i == 2 else [int(rng.integers(2, 4))]), None
            else:
                n = int(rng.integers(1, 7))
                fingers = [4] if n == 1 else ([2, 2] if rng.random() < .5 else [3, 2])
                area = None
            pair_palette = tuple(map(int, rng.choice(10, 3, replace=False))) if recolor else palette
            pair = _make_pair(rng, n, fingers, axis, pair_palette, area, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Input-derived semantic checks, plus evidence required of a demonstration set."""
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _check(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            d = _decode(a)
            _check(np.array_equal(solve(a), out), "incorrect printed census")
            _check(np.array_equal(d["full"], np.flip(d["full"], axis=d["axis"])), "restoration failed")
            _check(len(d["groups"]) == len(_components(d["full"] == d["ink"], True)), "ambiguous connectivity")
            _check(len(_components(d["window"])) == len(_components(d["window"], True)), "ambiguous window fragments")
            _check(set(map(int, np.unique(out))) <= {d["background"], d["pigment"]}, "wrong output roles")
            if split == "train":
                training.append(d)
    _check(len({d["count"] for d in training}) > 1, "need distinct object counts")
    _check(any(int(a["window"].sum()) == int(b["window"].sum()) and a["count"] != b["count"]
               for a in training for b in training), "need equal-area/different-count witnesses")
    _check(any(len(_components(d["window"])) > d["count"] for d in training), "need outside reconnection witness")
    _check(any(len(d["groups"]) > d["count"] for d in training), "need unselected objects")
    _check(any(len(d["groups"]) != 2*d["count"] for d in training), "need a witness against halving the global census")
    _check(len({len(_components(d["window"]))-d["count"] for d in training}) > 1, "constant fragment-offset shortcut")
    _check(any(any(all(d["erased"][p] for p in g) for g in d["selected"]) for d in training), "need a fully hidden selected object")
    _check(any(d["window"].shape != d["tile"].shape for d in training), "need tight cropping witness")
    _check(any(d["window"].shape[d["axis"]] != d["count"] for d in training), "window thickness is not the count")
    _check(any(not np.array_equal(d["tile"], np.flip(d["tile"], axis=d["axis"])) for d in training), "need target orientation witness")


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity, all legal counts, and semantic mutations."""
    signatures, shapes, tiles, topologies, palettes = set(), set(), set(), set(), set()
    roles = [set(), set(), set()]
    axes, parities, counts, placements = set(), set(), set(), set()
    edge_cases = seam_joins = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        if seed in (0, 7, 41):
            _check(_signature(p) == _signature(generate_puzzle(seed)), "nondeterminism")
        train_surplus = []
        for split in ("train", "test"):
            for pair in p[split]:
                d = _decode(pair["input"])
                _check(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                shapes.add((pair["input"].shape, pair["output"].shape))
                tiles.add(_key(d["tile"]))
                topologies.add(tuple(sorted(len(g) for g in d["selected"])))
                palette = d["background"], d["ink"], d["pigment"]
                palettes.add(palette)
                for role, color in zip(roles, palette):
                    role.add(color)
                axes.add(d["axis"])
                parities.add((d["axis"], d["full"].shape[d["axis"]] % 2))
                counts.add(d["count"])
                r, s, c, e = d["bbox"]
                placements.add((r, s, c, e))
                edge_cases += r == 0 or c == 0 or s == d["full"].shape[0] or e == d["full"].shape[1]
                seam_joins += any(any(point[d["axis"]] < d["full"].shape[d["axis"]]//2 for point in g)
                                  and any(point[d["axis"]] >= d["full"].shape[d["axis"]]//2 for point in g)
                                  for g in d["selected"])
                surplus = len(_components(d["window"])) - d["count"]
                if split == "train":
                    train_surplus.append(surplus)
                else:
                    _check(surplus >= 2, "missing fragmentation shift")
                    _check(sum(x > 1 for x in d["pieces"]) >= 2 or max(d["pieces"]) >= 4,
                           "test should combine reconnections or extend a familiar fan")
        _check({1, 2} <= set(train_surplus), "training should distinguish fixed fragment offsets")
    _check(len(signatures) == 42, "puzzle diversity")
    _check(len(shapes) >= 100 and len(tiles) >= 150, "geometric diversity")
    _check(len(topologies) >= 80 and len(palettes) >= 70, "object/palette diversity")
    _check(all(len(r) == 10 for r in roles), "palette roles should span all colors")
    _check(axes == {0, 1} and len(parities) == 4 and counts == set(range(1, 7)), "axis/parity/count coverage")
    _check(len(placements) >= 150 and edge_cases >= 10 and seam_joins >= 5, "placement/connectivity coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(81723, nt, nq)
            _check(len(p["train"]) == nt and len(p["test"]) == nq, "ignored counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    corrupt = copy.deepcopy(p)
    corrupt["test"][0]["output"].flat[0] = (int(corrupt["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    # A transpose and a color bijection preserve the entire family, including
    # the print axis. Neither operation supplies solve with latent parameters.
    perm = np.array([7, 1, 9, 0, 6, 5, 2, 8, 3, 4], dtype=np.int8)
    transformed = {s: [{k: perm[np.asarray(v).T].copy() for k, v in pair.items()}
                       for pair in p[s]] for s in p}
    validate_generated_puzzle(transformed)
    # A joining path above the window (rather than beside it) is a valid family
    # member even though the sampler's bands use side spines.
    half = np.zeros((15, 12), dtype=bool)
    half[2, 2:6] = True
    half[2:10, 2] = True
    half[2:8, 5] = True
    half[6:9, 7] = True
    half[12:14, 8:11] = True
    full = np.concatenate((half, half[:, ::-1]), axis=1).astype(np.int8)
    masked = full.copy()
    masked[4:10, 1:8] = 2
    d = _decode(masked)
    _check(d["count"] == 2 and sorted(d["pieces"]) == [1, 2], "whole object selection fixture")
    expected = _render_label(full, (4, 10, 1, 8), 1, 2, 0, 1, 2)
    _check(np.array_equal(solve(masked), expected), "non-band path not supported")
    # The rule and solver support fitting counts beyond the sampler's 1..6.
    half = np.zeros((19, 6), dtype=bool)
    half[2:15:2, 2] = True
    half[16:18, 4] = True
    full = np.concatenate((half, half[:, ::-1]), axis=1).astype(np.int8)
    masked = full.copy()
    masked[1:16, 2:3] = 2
    _check(_decode(masked)["count"] == 7, "general census count")
    _check(np.array_equal(solve(masked), _render_label(full, (1, 16, 2, 3), 1, 2, 0, 1, 7)),
           "solver depends on generation count envelope")
    # A window covering both members of a mirror orbit cannot be reconstructed.
    invalid = np.array([[0, 1, 0, 0, 1, 0], [0, 1, 2, 2, 1, 0],
                        [0, 0, 2, 2, 0, 0], [0, 1, 0, 0, 1, 0]], dtype=np.int8)
    try:
        solve(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("erased orbit accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shape_pairs": len(shapes),
            "stencils": len(tiles), "selected_size_lists": len(topologies),
            "palettes": len(palettes), "edge_windows": int(edge_cases), "seam_joined": int(seam_joins)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
