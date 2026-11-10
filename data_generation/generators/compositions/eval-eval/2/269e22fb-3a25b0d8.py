"""Inspection windows select whole rooms across a deformed floorplan.

All reference information is in the input. Generation uses labelled rooms to
construct answers; solving uses only exact wall fragments and spatial relations.
No parent generator, saved pair, seed table, or training label is used by solve.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _transform(grid, index):
    """0..3 are quarter turns; 4..7 reflect left/right, then turn."""
    return np.rot90(np.fliplr(grid) if index >= 4 else grid, index % 4).copy()


def _components(mask, diagonal=False):
    h, w = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        visited[r, c] = True
        stack, part = [(r, c)], []
        while stack:
            y, x = stack.pop()
            part.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.array(part, dtype=int))
    return result


def _box(points):
    y, x = points.min(axis=0)
    Y, X = points.max(axis=0) + 1
    return int(y), int(x), int(Y), int(X)


def _rooms(grid, wall):
    """Four-connected enclosed non-wall regions, in discovery order."""
    h, w = grid.shape
    return [p for p in _components(grid != wall)
            if np.all(p > 0) and np.all(p < np.array([h - 1, w - 1]))]


def _relations(rooms):
    boxes = [_box(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else 1 if b[k + 2] <= a[k] else 0
                   for k in (0, 1)) for b in boxes] for a in boxes]


def _room_mapping(donor_rooms, target_rooms):
    """Unique compass-relation isomorphism; no areas, colors, or room ranks."""
    a, b = _relations(donor_rooms), _relations(target_rooms)
    n = len(a)
    if n != len(b):
        raise ValueError("different room counts")
    signatures_a = [Counter(row) for row in a]
    signatures_b = [Counter(row) for row in b]
    domains = [[j for j in range(n) if signatures_a[i] == signatures_b[j]] for i in range(n)]
    answers = []

    def search(mapping, used):
        if len(answers) > 1:
            return
        if len(mapping) == n:
            answers.append(dict(mapping))
            return
        options = {i: [j for j in domains[i] if j not in used and all(
            a[i][k] == b[j][v] for k, v in mapping.items())]
            for i in range(n) if i not in mapping}
        i = min(options, key=lambda k: (len(options[k]), k))
        for j in options[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    if len(answers) != 1:
        raise ValueError("room correspondence must be unique")
    return answers[0]


def _locations(reference, patch):
    h, w = patch.shape
    if h > reference.shape[0] or w > reference.shape[1]:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(reference, (h, w))
    return [(int(r), int(c)) for r, c in np.argwhere(np.all(windows == patch, axis=(-2, -1)))]


def _locate(reference, patch):
    """All D4 registrations, deduplicating identical geometric supports."""
    candidates = set()
    for i in range(8):
        p = _transform(patch, i)
        candidates.update((r, c, *p.shape) for r, c in _locations(reference, p))
    if len(candidates) != 1:
        raise ValueError("inspection fragment does not have one location")
    return next(iter(candidates))


def _extract(input_grid):
    inp = np.asarray(input_grid)
    if inp.ndim != 2 or not np.issubdtype(inp.dtype, np.integer):
        raise ValueError("expected an integer grid")
    if not all(1 <= d <= 30 for d in inp.shape) or np.any((inp < 0) | (inp > 9)):
        raise ValueError("grid outside ARC bounds")
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    if len(np.unique(border)) != 1:
        raise ValueError("expected a uniform background border")
    bg = int(border[0])
    frames = []
    for color in np.unique(inp):
        if color == bg:
            continue
        y, x, Y, X = _box(np.argwhere(inp == color))
        if min(Y - y, X - x) < 4:
            continue
        mask = np.zeros(inp.shape, dtype=bool)
        mask[y, x:X] = mask[Y - 1, x:X] = True
        mask[y:Y, x] = mask[y:Y, X - 1] = True
        if np.array_equal(inp == color, mask):
            frames.append((int(color), (y, x, Y, X)))
    if len(frames) != 1:
        raise ValueError("expected one uniquely colored rectangular swatch frame")
    frame, (y, x, Y, X) = frames[0]
    tile = inp[y + 1:Y - 1, x + 1:X - 1].copy()
    wall_colors = set(map(int, tile.flat)) - {bg}
    if len(wall_colors) != 1 or bg not in tile:
        raise ValueError("the swatch must contain both wall and background")
    wall = wall_colors.pop()
    if wall == frame:
        raise ValueError("wall and swatch frame have different roles")
    remaining = inp != bg
    remaining[y:Y, x:X] = False
    parts = _components(remaining, diagonal=True)
    if len(parts) != 2:
        raise ValueError("expected donor and target outside the swatch")
    objects = []
    for part in parts:
        r, c, R, C = _box(part)
        crop = np.full((R - r, C - c), bg, dtype=np.int8)
        crop[part[:, 0] - r, part[:, 1] - c] = inp[tuple(part.T)]
        objects.append(crop)
    targets = [o for o in objects if set(map(int, o.flat)) == {bg, wall}]
    donors = [o for o in objects if len(set(map(int, o.flat)) - {bg, wall}) >= 2]
    if len(targets) != 1 or len(donors) != 1:
        raise ValueError("expected a colored donor and a blank target")
    donor, target = donors[0], targets[0]
    dr, tr = _rooms(donor, wall), _rooms(target, wall)
    if not 4 <= len(dr) <= 8 or len(tr) != len(dr):
        raise ValueError("expected four to eight corresponding rooms")
    colors = []
    covered = np.zeros(donor.shape, dtype=bool)
    for room in dr:
        values = np.unique(donor[tuple(room.T)])
        if len(values) != 1 or int(values[0]) in (bg, wall, frame):
            raise ValueError("donor rooms must be uniformly colored")
        colors.append(int(values[0]))
        covered[tuple(room.T)] = True
    if not np.array_equal(covered, (donor != bg) & (donor != wall)):
        raise ValueError("all donor payload belongs to rooms")
    if not all(np.all(target[tuple(room.T)] == bg) for room in tr):
        raise ValueError("target rooms must be blank")
    return bg, wall, frame, tile, donor, target, dr, tr, colors


def _analyze(input_grid):
    info = _extract(input_grid)
    bg, wall, frame, tile, donor, target, dr, tr, colors = info
    mapping = _room_mapping(dr, tr)
    r, c, h, w = _locate(donor == wall, tile == wall)
    selected = [i for i, p in enumerate(dr) if np.any(
        (p[:, 0] >= r) & (p[:, 0] < r + h) & (p[:, 1] >= c) & (p[:, 1] < c + w))]
    output = target.copy()
    for i in selected:
        output[tuple(tr[mapping[i]].T)] = colors[i]
    return output, info, mapping, selected, (r, c, h, w)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recognize the window, then transfer the selected whole-room colors."""
    return _analyze(input_grid)[0]


def _partition(rng, count):
    choices = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3)]
    choices = [s for s in choices if s[0] * s[1] >= count]
    shape = choices[int(rng.integers(len(choices)))]
    plan = np.arange(np.prod(shape), dtype=int).reshape(shape)
    while len(np.unique(plan)) > count:
        edges = [(int(plan[r, c]), int(plan[r + dy, c + dx]))
                 for r in range(shape[0]) for c in range(shape[1])
                 for dy, dx in ((1, 0), (0, 1))
                 if r + dy < shape[0] and c + dx < shape[1]
                 and plan[r, c] != plan[r + dy, c + dx]]
        a, b = edges[int(rng.integers(len(edges)))]
        plan[plan == b] = a
    old = plan.copy()
    for i, label in enumerate(np.unique(old)):
        plan[old == label] = i
    return plan


def _embed_plan(plan, heights, widths):
    ys, xs = np.r_[0, np.cumsum(heights + 1)], np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in np.ndindex(plan.shape):
        label = plan[r, c]
        labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = label
        if r and plan[r - 1, c] == label:
            labels[ys[r], xs[c] + 1:xs[c + 1]] = label
        if c and plan[r, c - 1] == label:
            labels[ys[r] + 1:ys[r + 1], xs[c]] = label
        if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == label):
            labels[ys[r], xs[c]] = label
    return labels


def _render(rng, plan):
    # Each coarse axis is embedded independently; merged cells form bent rooms.
    heights = rng.integers(1, 4, size=plan.shape[0])
    widths = rng.integers(1, 4, size=plan.shape[1])
    while heights.sum() + len(heights) + 1 > 13:
        heights[int(rng.choice(np.flatnonzero(heights > 1)))] -= 1
    while widths.sum() + len(widths) + 1 > 13:
        widths[int(rng.choice(np.flatnonzero(widths > 1)))] -= 1
    labels = _embed_plan(plan, heights, widths)
    for label in np.unique(plan):
        # Boundary terraces change local room starts, even their scan order.
        for _ in range(int(rng.integers(1, 4))):
            points = np.argwhere(labels == label)
            if len(points) <= 2:
                break
            y, x, Y, X = _box(points)
            direction = int(rng.integers(6))
            cut = np.zeros(labels.shape, dtype=bool)
            if direction == 0:
                cut[y, :] = True
            elif direction == 1:
                cut[Y - 1, :] = True
            elif direction == 2:
                cut[:, x] = True
            elif direction == 3:
                cut[:, X - 1] = True
            else:
                p = [(y, x), (y, X - 1), (Y - 1, x), (Y - 1, X - 1)][int(rng.integers(4))]
                cut[p] = True
            trial = (labels == label) & ~cut
            if trial.any() and len(_components(trial)) == 1:
                labels[(labels == label) & cut] = -1
    return labels


def _paint(labels, bg, wall, colors, selected):
    result = np.full(labels.shape, bg, dtype=np.int8)
    result[labels < 0] = wall
    for label in selected:
        result[labels == label] = colors[label]
    return result


def _find_window(rng, labels, colors):
    H, W = labels.shape
    for _ in range(160):
        h = int(rng.integers(3, min(11, H) + 1))
        w = int(rng.integers(3, min(11, W) + 1))
        if h == H and w == W:
            continue
        r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
        crop = labels[r:r + h, c:c + w]
        selected = sorted(set(map(int, crop.flat)) - {-1})
        if not 2 <= len(selected) < len(colors) or len({colors[i] for i in selected}) < 2:
            continue
        # A partially observed room distinguishes intersection from containment.
        if not any(np.count_nonzero(crop == i) < np.count_nonzero(labels == i) for i in selected):
            continue
        patch = crop < 0
        if len({_key(_transform(patch, k)) for k in range(8)}) != 8:
            continue
        try:
            found = _locate(labels < 0, patch)
        except ValueError:
            continue
        if found == (r, c, h, w):
            return patch, selected
    raise ValueError("could not find a selective identifying window")


def _pack(rng, objects, bg):
    """Random free placement in a bounded canvas, then a guaranteed shelf fit."""
    for _ in range(24):
        canvas = np.full((30, 30), bg, dtype=np.int8)
        occupied = np.zeros((30, 30), dtype=bool)
        good = True
        for index in rng.permutation(3):
            obj = objects[int(index)]
            h, w = obj.shape
            placed = False
            for _ in range(36):
                r, c = int(rng.integers(1, 30 - h)), int(rng.integers(1, 30 - w))
                if not occupied[r - 1:r + h + 1, c - 1:c + w + 1].any():
                    canvas[r:r + h, c:c + w] = obj
                    occupied[r:r + h, c:c + w] = True
                    placed = True
                    break
            if not placed:
                good = False
                break
        if good:
            break
    else:
        canvas = np.full((30, 30), bg, dtype=np.int8)
        order = list(rng.permutation(3))
        for k, (r, c) in zip(order, ((1, 1), (1, 15), (15, 1))):
            obj = objects[int(k)]
            canvas[r:r + obj.shape[0], c:c + obj.shape[1]] = obj
    # Retain a one-cell border, with independently variable extra margins.
    points = np.argwhere(canvas != bg)
    y, x, Y, X = _box(points)
    y = max(0, y - 1 - int(rng.integers(3)))
    x = max(0, x - 1 - int(rng.integers(3)))
    Y = min(30, Y + 1 + int(rng.integers(3)))
    X = min(30, X + 1 + int(rng.integers(3)))
    return canvas[y:Y, x:X].copy()


def _fallback_pair(rng, count, probe_orientation):
    """A finite procedural construction, also a scan-order counterexample.

    Unequal top-room starts identify the crop. The two selected top rooms swap
    discovery order after deformation. This is a generation fallback, never a
    reference image or a branch in the solver.
    """
    cols = (count + 1) // 2
    plan = np.arange(2 * cols).reshape(2, cols)
    if count % 2:
        plan[1, -1] = plan[0, -1]
        old = plan.copy()
        for i, label in enumerate(np.unique(old)):
            plan[old == label] = i
    d = _embed_plan(plan, np.array([3, 3]), np.full(cols, 2))
    t = d.copy()
    d[1, 1:3] = -1
    t[1, 4:6] = -1
    palette = list(map(int, rng.permutation(10)))
    bg, wall, frame = palette[:3]
    colors = [palette[3], palette[4]]
    colors += [palette[3 + int(rng.integers(min(7, count)))] for _ in range(count - 2)]
    donor = _paint(d, bg, wall, colors, range(count))
    target = _paint(t, bg, wall, colors, [])
    expected = _paint(t, bg, wall, colors, [0, 1])
    patch = _transform(d[:3, :7] < 0, probe_orientation)
    tile = np.where(patch, wall, bg).astype(np.int8)
    swatch = np.pad(tile, 1, constant_values=frame)
    inp = _pack(rng, [donor, target, swatch], bg)
    pair = {"input": inp, "output": expected}
    m = _measure(pair)
    assert m["scan_witness"]
    return pair


def _make_pair(rng, count, probe_orientation, require_scan_witness=False, attempts=240):
    for attempt in range(attempts):
        plan = _partition(rng, count)
        d, t = _render(rng, plan), _render(rng, plan)
        common_orientation = int(rng.integers(8))
        d, t = _transform(d, common_orientation), _transform(t, common_orientation)
        if d.shape == t.shape and np.array_equal(d, t):
            continue
        if len(_components(d < 0, diagonal=True)) != 1 or len(_components(t < 0, diagonal=True)) != 1:
            continue
        dr = [np.argwhere(d == i) for i in range(count)]
        tr = [np.argwhere(t == i) for i in range(count)]
        if _relations(dr) != _relations(tr):
            continue
        try:
            _room_mapping(dr, tr)
        except ValueError:
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, wall, frame = palette[:3]
        richness = int(rng.integers(2, min(7, count) + 1))
        colors = palette[3:3 + richness]
        colors += [colors[int(rng.integers(richness))] for _ in range(count - richness)]
        rng.shuffle(colors)
        try:
            patch, selected = _find_window(rng, d, colors)
        except ValueError:
            continue
        donor = _paint(d, bg, wall, colors, range(count))
        target = _paint(t, bg, wall, colors, [])
        expected = _paint(t, bg, wall, colors, selected)
        # Geometric deformation, not same-sized cell copying, must matter.
        if _key(expected) == _key(_paint(d, bg, wall, colors, selected)):
            continue
        if require_scan_witness:
            d_order = sorted(range(count), key=lambda i: tuple(dr[i][0]))
            t_order = sorted(range(count), key=lambda i: tuple(tr[i][0]))
            rank_answer = target.copy()
            for i, j in zip(d_order, t_order):
                if i in selected:
                    rank_answer[t == j] = colors[i]
            if np.array_equal(rank_answer, expected):
                continue
        # Framing is a visible datatype cue; frame and wall roles never mix.
        tile = np.where(_transform(patch, probe_orientation), wall, bg).astype(np.int8)
        swatch = np.pad(tile, 1, constant_values=frame)
        inp = _pack(rng, [donor, target, swatch], bg)
        # Answers were constructed from room labels; check visible inference.
        try:
            if np.array_equal(solve(inp), expected):
                return {"input": inp, "output": expected}
        except ValueError:
            continue
    return _fallback_pair(rng, count, probe_orientation)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= value <= high:
        raise ValueError(name + " outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # Hold out one factor, rather than bundling all novelty into every test.
    count_shift = bool(rng.integers(2))
    maximum = 6 if count_shift else 8
    counts = [int(rng.integers(4, maximum + 1))]
    counts.append(int(rng.choice([c for c in range(4, maximum + 1) if c != counts[0]])))
    counts += list(map(int, rng.integers(4, maximum + 1, size=nt - 2)))
    counts += list(map(int, rng.integers(7 if count_shift else 4, 9, size=nq)))
    orientations = [int(rng.choice([1, 3])), 4]
    orientations += list(map(int, rng.choice([0, 1, 3, 4] if not count_shift else range(8), nt - 2)))
    orientations += list(map(int, rng.choice(range(8) if count_shift else [2, 5, 6, 7], nq)))
    result, seen = {"train": [], "test": []}, set()
    for i, (count, orientation) in enumerate(zip(counts, orientations)):
        for _ in range(12):
            pair = _make_pair(rng, count, orientation, require_scan_witness=(i == 1))
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            # Ninety different visible (background, wall) role assignments
            # exceed the thirteen previously used inputs. Recolor bijectively.
            bg, wall, frame = _extract(pair["input"])[:3]
            old = [bg, wall, frame] + [c for c in range(10) if c not in (bg, wall, frame)]
            found = False
            for new_bg in range(10):
                for new_wall in range(10):
                    if new_bg == new_wall:
                        continue
                    new = [new_bg, new_wall] + [c for c in range(10) if c not in (new_bg, new_wall)]
                    lookup = np.empty(10, dtype=np.int8)
                    lookup[old] = new
                    recolored = {k: lookup[v] for k, v in pair.items()}
                    key = _key(recolored["input"])
                    if key not in seen:
                        pair, found = recolored, True
                        break
                if found:
                    break
            assert found
        seen.add(key)
        result["train" if i < nt else "test"].append(pair)
    return result


def _measure(pair):
    predicted, info, mapping, selected, window = _analyze(pair["input"])
    bg, wall, frame, tile, donor, target, dr, tr, colors = info
    assert np.array_equal(predicted, pair["output"]), "incorrect answer"
    assert 2 <= len(selected) < len(dr), "selection must be nontrivial"
    assert len({colors[i] for i in selected}) >= 2, "show multicolor transfer"
    assert len({_key(_transform(tile, k)) for k in range(8)}) == 8
    r, c, h, w = window
    orientation = next(k for k in range(8) if np.array_equal(
        tile == wall, _transform((donor == wall)[r:r + h, c:c + w], k)))
    partial = [i for i in selected if sum(r <= y < r + h and c <= x < c + w for y, x in dr[i]) < len(dr[i])]
    assert partial, "show propagation from a partial room"
    changed = np.zeros(target.shape, dtype=bool)
    all_filled, rank_answer, area_answer = target.copy(), target.copy(), target.copy()
    for i in range(len(dr)):
        all_filled[tuple(tr[mapping[i]].T)] = colors[i]
        if i in selected:
            changed[tuple(tr[mapping[i]].T)] = True
    for i, j in zip(range(len(dr)), range(len(tr))):
        if i in selected:
            rank_answer[tuple(tr[j].T)] = colors[i]
    for i, j in zip(sorted(range(len(dr)), key=lambda k: (len(dr[k]), k)),
                    sorted(range(len(tr)), key=lambda k: (len(tr[k]), k))):
        if i in selected:
            area_answer[tuple(tr[j].T)] = colors[i]
    assert np.array_equal(predicted == wall, target == wall), "preserve walls"
    assert np.array_equal(predicted != target, changed), "fill exactly selected interiors"
    assert not np.array_equal(predicted, all_filled), "omitting A must change the answer"
    # Omitting B: return the selected donor, without transporting its geometry.
    donor_selection = donor.copy()
    for i, room in enumerate(dr):
        if i not in selected:
            donor_selection[tuple(room.T)] = bg
    assert _key(donor_selection) != _key(predicted), "deformation must affect the answer"
    contained_answer = target.copy()
    for i in selected:
        if i not in partial:
            contained_answer[tuple(tr[mapping[i]].T)] = colors[i]
    assert not np.array_equal(contained_answer, predicted)
    return {"rooms": len(dr), "selected": len(selected), "palette": (bg, wall, frame),
            "donor_shape": donor.shape, "target_shape": target.shape, "tile_shape": tile.shape,
            "orientation": orientation,
            "window_fraction": h * w / donor.size, "partial_rooms": len(partial),
            "scan_witness": not np.array_equal(rank_answer, predicted),
            "area_witness": not np.array_equal(area_answer, predicted),
            "signs": tuple(sorted({int(np.sign(len(tr[mapping[i]]) - len(dr[i]))) for i in range(len(dr))})),
            "bent_rooms": sum(len(p) != (_box(p)[2] - _box(p)[0]) * (_box(p)[3] - _box(p)[1]) for p in dr + tr),
            "reference": _key(donor == wall)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                a = np.asarray(grid)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= d <= 30 for d in a.shape) and np.all((0 <= a) & (a <= 9))
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            m = _measure(pair)
            if split == "train":
                training.append(m)
    assert len({m["rooms"] for m in training}) >= 2
    assert any(m["scan_witness"] for m in training), "demonstrations must refute scan-order matching"


def self_test():
    """Stream 42 seeds; retain only small diversity sets, not a puzzle corpus."""
    signatures, references, shapes, roles, counts, selects, signs = (set() for _ in range(7))
    bent = scan = area = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                for name in ("input", "output"):
                    a = np.asarray(pair[name], dtype=np.int8)
                    digest.update(bytes(a.shape) + a.tobytes())
                m = _measure(pair)
                references.add(m["reference"])
                shapes.add((m["donor_shape"], m["target_shape"], m["tile_shape"]))
                roles.add(m["palette"])
                counts.add(m["rooms"])
                selects.add(m["selected"])
                signs.update(m["signs"])
                bent += m["bent_rooms"] > 0
                scan += m["scan_witness"]
                area += m["area_witness"]
                pairs += 1
        signatures.add(digest.digest())
        tm = [_measure(p) for p in puzzle["train"]]
        qm = [_measure(p) for p in puzzle["test"]]
        count_shift = max(m["rooms"] for m in tm) <= 6 and min(m["rooms"] for m in qm) >= 7
        orientation_shift = ({m["orientation"] for m in tm} <= {0, 1, 3, 4}
                             and {m["orientation"] for m in qm} <= {2, 5, 6, 7})
        assert count_shift or orientation_shift
    assert len(signatures) == 42
    assert len(references) >= 150 and len(shapes) >= 100 and len(roles) >= 100
    assert counts == set(range(4, 9)) and len(selects) >= 4 and signs == {-1, 0, 1}
    assert bent >= 60 and scan >= 42 and area >= 100
    a, b = generate_puzzle(947, 2, 1), generate_puzzle(947, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    unchanged = a["test"][0]["input"].copy()
    solve(a["test"][0]["input"])
    assert np.array_equal(unchanged, a["test"][0]["input"])
    for n in range(4, 9):
        for orientation in range(8):
            _make_pair(np.random.default_rng(800 + n * 8 + orientation), n,
                       orientation, require_scan_witness=True, attempts=0)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(300 + nt * 10 + nq, nt, nq)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nq
            validate_generated_puzzle(puzzle)
    for nt, nq in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)]:
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(5, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    return {"seeds": 42, "pairs": pairs, "references": len(references),
            "geometry_combinations": len(shapes), "role_palettes": len(roles),
            "bent_pairs": bent, "scan_counterexamples": scan, "area_counterexamples": area}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
