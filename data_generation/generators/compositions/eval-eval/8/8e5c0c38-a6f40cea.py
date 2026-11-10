"""Witness-connected mirror reconstruction through one opaque window.

All inference uses visible pixels only. Generation renders a complete clean
layer before hiding it; the solver never sees that construction layer.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= d <= 30 for d in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return array


def _points(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _neighbors(points):
    return {(y + dy, x + dx) for y, x in points
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))}


def _components(points):
    todo = set(points)
    while todo:
        first = min(todo)
        todo.remove(first)
        group, queue = {first}, deque([first])
        while queue:
            y, x = queue.popleft()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    queue.append(p)
        yield group


def _window(grid):
    """The sole dedicated-color rectangular border encloses a uniform fill."""
    bg = int(grid[0, 0])
    if any(int(grid[p]) != bg for p in
           ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("the four canvas corners must show the background")
    frames = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        cells = _points(grid == color)
        t, b, l, r = _bbox(cells)
        if b - t < 2 or r - l < 2 or cells != _perimeter((t, b, l, r)):
            continue
        inner = grid[t + 1:b, l + 1:r]
        if np.all(inner == inner[0, 0]) and int(inner[0, 0]) != color:
            frames.append(((t, b, l, r), color, int(inner[0, 0])))
    if len(frames) != 1:
        raise ValueError("expected exactly one dedicated-color opaque frame")
    box, frame, fill = frames[0]
    t, b, l, r = box
    mask = np.zeros(grid.shape, dtype=bool)
    mask[t:b + 1, l:r + 1] = True
    return bg, frame, fill, box, mask


def _read_figure(grid, color, mask):
    """Score visible twins, reject contradictions, then propagate witnesses.

    A missing pixel under the mask is unknown. A missing pixel elsewhere is a
    contradiction. Candidate components need at least one fully visible orbit.
    """
    visible = (grid == color) & ~mask
    width = grid.shape[1]
    scores = np.zeros(2 * width - 1, dtype=int)
    for row in visible.astype(np.int16):
        scores += np.convolve(row, row)
    winners = np.flatnonzero(scores == scores.max())
    if len(winners) != 1 or int(scores.max()) < 2:
        raise ValueError("figure needs a unique axis with visible twin evidence")
    axis = int(winners[0])
    candidates, anchors, rejected = set(), set(), set()
    observed = _points(visible)
    off_axis = 0
    for p in sorted(observed):
        y, x = p
        q = (y, axis - x)
        if not 0 <= q[1] < width:
            rejected.add(p)
        elif mask[q]:
            candidates.update((p, q))
        elif q in observed:
            candidates.update((p, q))
            anchors.update((p, q))
            off_axis += int(q != p)
        else:
            rejected.add(p)
    if off_axis < 2:
        raise ValueError("axis-only dots do not establish the direction")
    kept, orphans = set(), set()
    for group in _components(candidates):
        if group & anchors:
            kept.update(group)
        else:
            orphans.update(group)
    return {"axis2": axis, "score": int(scores.max()), "observed": observed,
            "anchors": anchors, "candidates": candidates, "kept": kept,
            "orphans": orphans, "rejected": rejected}


def _read(grid):
    bg, frame, fill, box, mask = _window(grid)
    colors = sorted(set(map(int, np.unique(grid[~mask]))) - {bg})
    if not colors or frame in colors or (fill != bg and fill in colors):
        raise ValueError("window and figure color roles must be disjoint")
    t, b, l, r = box
    output = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    figures, assigned = {}, set()
    for color in colors:
        info = _read_figure(grid, color, mask)
        hidden = {p for p in info["kept"] if t < p[0] < b and l < p[1] < r}
        if hidden & assigned:
            raise ValueError("different figures cannot claim one hidden cell")
        for y, x in hidden:
            output[y - t - 1, x - l - 1] = color
        assigned.update(hidden)
        info["hidden"] = hidden
        figures[color] = info
    return output, {"background": bg, "frame": frame, "fill": fill,
                    "box": box, "mask": mask, "figures": figures}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the trusted reconstructed strict interior, using only the input."""
    return _read(_grid(input_grid))[0]


def _reflect(points, width):
    return {(y, width - 1 - x) for y, x in points}


def _shape(rng, height, width, box, side, parity, island_demo=False):
    """A connected cage with a spine, varied rungs, and a false attached branch.

    Construct in a left-approaching coordinate system, then optionally flip.
    The false branch is attached through one visibly contradicted mirror orbit;
    its far end faces the mask and becomes an unanchored component after pruning.
    """
    t, b, l, r = box
    if side:
        l, r = width - 1 - r, width - 1 - l
    offset = int(parity)  # odd axis2: one central column on each side of the axis
    k_choices = [k for k in (3, 4) if l >= 2 * k + 3 - offset]
    if not k_choices:
        return None
    k = int(rng.choice(k_choices))
    a = l - k
    minimum = k + 3 - offset
    maximum = min(a, k + 5 - offset, r - 1 - a - offset)
    if minimum > maximum:
        return None
    radius = int(rng.integers(minimum, maximum + 1))
    inner = k - 2 - offset
    h = int(rng.integers(5, min(11, height - 2) + 1))
    top = int(rng.integers(1, height - h))
    bottom = top + h - 1
    rows = [y for y in range(top + 2, bottom - 1) if t < y < b]
    if not rows:
        return None
    decoy_row = int(rng.choice(rows))

    def twins(y, distance):
        return {(y, a - distance), (y, a + offset + distance)}

    style = int(rng.integers(4))
    levels = [radius] * h
    if style in (1, 2):
        for index in range(1, h - 1):
            if abs(top + index - decoy_row) >= 2:
                levels[index] = int(rng.integers(max(inner + 2, radius - 3), radius + 1))
    core = set()
    for y in range(top, bottom + 1):
        spine = inner if y == decoy_row or style == 0 else int(rng.integers(inner + 1))
        for d in range(spine + 1):
            core.update(twins(y, d))
        level = levels[y - top]
        previous = levels[max(0, y - top - 1)]
        for d in range(min(level, previous), max(level, previous) + 1):
            core.update(twins(y, d))
    caps = {top, bottom}
    # Additional rectangular crossbars and stepped lobes, away from the decoy.
    free_rows = [y for y in range(top + 1, bottom)
                 if abs(y - decoy_row) >= 2]
    for y in free_rows:
        if rng.random() < (.8 if style == 3 else .35):
            caps.add(y)
    for y in caps:
        for d in range(levels[y - top] + 1):
            core.update(twins(y, d))
    # Short symmetric twigs thicken some contours without prescribing a
    # rectangle template. Keep the trap's three-row neighborhood untouched.
    if style == 2:
        for y in free_rows:
            for d in range(inner + 1, levels[y - top]):
                cells = twins(y, d)
                if cells & _neighbors(core) and rng.random() < .4:
                    core.update(cells)
    # A smaller independently witnessed component defeats "keep only the
    # largest connected object". Its answer uses the same reflection axis.
    island_rows = [y for y in range(t + 1, b)
                   if 0 < y < height - 1 and (y <= top - 2 or y >= bottom + 2)]
    if island_demo or rng.random() < .28:
        if not island_rows:
            return None
        y = int(rng.choice(island_rows))
        island_radius = int(rng.integers(minimum, radius + 1))
        for d in range(island_radius + 1):
            core.update(twins(y, d))
    # One-sided visible additions; their missing twins just before the frame
    # are known blanks, while the branch tip's twin is genuinely unknown.
    decoy = {(decoy_row, a - d) for d in range(inner + 1, radius - 1)}
    axis = 2 * a + offset
    if side:
        core, decoy = _reflect(core, width), _reflect(decoy, width)
        axis = 2 * (width - 1) - axis
    return core, decoy, axis


def _render_visible(under, bg, frame, fill, box, noise):
    grid = under.copy()
    for color, cells in noise.items():
        for p in cells:
            grid[p] = color
    t, b, l, r = box
    grid[t:b + 1, l:r + 1] = fill
    for p in _perimeter(box):
        grid[p] = frame
    return grid


def _make_pair(rng, count, background, parity_demo=None, fill_matches=None, island_demo=False):
    for _attempt in range(96):
        height, width = int(rng.integers(22, 31)), int(rng.integers(26, 31))
        t = int(rng.integers(1, 5))
        b = height - 1 - int(rng.integers(1, 5))
        l = int(rng.integers(9, 12))
        r = width - 1 - int(rng.integers(9, 12))
        if r - l < 4:
            continue
        box = (t, b, l, r)
        others = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
        frame, distinct_fill = others[:2]
        colors = others[2:2 + count]
        if fill_matches is None:
            fill_matches_here = bool(rng.integers(2))
        else:
            fill_matches_here = fill_matches
        fill = background if fill_matches_here else distinct_fill
        under = np.full((height, width), background, dtype=np.int8)
        mask = np.zeros((height, width), dtype=bool)
        mask[t:b + 1, l:r + 1] = True
        occupied, cores, noise, axes = set(), {}, {}, {}
        side_order = [int(rng.integers(2)) for _ in range(count)]
        if count >= 2:
            side_order[:2] = [0, 1]
            rng.shuffle(side_order)
        success = True
        for index, color in enumerate(colors):
            placed = False
            parity = (parity_demo if parity_demo is not None and index == 0
                      else int(rng.integers(2)))
            for _ in range(180):
                result = _shape(rng, height, width, box, side_order[index], parity,
                                island_demo and index == 0)
                if result is None:
                    continue
                core, decoy, axis = result
                # The first figure supplies the contradiction/ghost witness;
                # others can be clean rather than obeying a fixed noise quota.
                noisy = index == 0 or rng.random() < .75
                if not noisy:
                    decoy = set()
                if (core | decoy) & occupied:
                    continue
                visible_decoy = {p for p in decoy if not mask[p]}
                probe = np.full(under.shape, background, dtype=np.int8)
                for p in core | visible_decoy:
                    probe[p] = color
                try:
                    info = _read_figure(probe, color, mask)
                except ValueError:
                    continue
                hidden = {p for p in core if t < p[0] < b and l < p[1] < r}
                if (info["axis2"] != axis or info["kept"] != core or not hidden):
                    continue
                if noisy and (not info["rejected"] or not any(
                        t < y < b and l < x < r for y, x in info["orphans"])):
                    continue
                # The raw attached decoy must be a real A-omission trap.
                if noisy:
                    raw = {p for p in core | decoy if not mask[p]}
                    raw_group = next(g for g in _components(raw) if g & visible_decoy)
                    if not (raw_group & info["anchors"]):
                        continue
                placed = True
                cores[color], noise[color], axes[color] = core, visible_decoy, axis
                occupied.update(core | visible_decoy)
                for p in core:
                    under[p] = color
                break
            if not placed:
                success = False
                break
        if not success:
            continue
        # A distant additive speck moves the raw color bounding box. The axis
        # must come from reflected agreement rather than bounding-box centering.
        first_color = colors[0]
        far_x = width - 1 if axes[first_color] < 2 * l else 0
        far_rows = [y for y in range(1, height - 1) if (y, far_x) not in occupied]
        rng.shuffle(far_rows)
        for y in far_rows:
            p = (y, far_x)
            probe = _render_visible(under, background, frame, fill, box,
                                    {first_color: noise[first_color] | {p}})
            try:
                info = _read_figure(probe, first_color, mask)
            except ValueError:
                continue
            if info["axis2"] == axes[first_color] and info["kept"] == cores[first_color]:
                noise[first_color].add(p)
                occupied.add(p)
                break
        else:
            continue
        # Optional detached contradictory specks and short attached additions.
        # These do not define the mechanism; every object already has its trap.
        for color in colors:
            axis = axes[color]
            candidates = []
            near = _neighbors(cores[color])
            for p in sorted(near):
                y, x = p
                q = (y, axis - x)
                if (0 < y < height - 1 and 0 <= x < width and p not in occupied
                        and not mask[p] and 0 <= q[1] < width and not mask[q]
                        and q not in cores[color] and q not in noise[color] and p != q):
                    candidates.append(p)
            if candidates:
                for j in rng.permutation(len(candidates))[:int(rng.integers(0, 4))]:
                    p = candidates[int(j)]
                    trial = noise[color] | {p}
                    probe = _render_visible(under, background, frame, fill, box, {color: trial})
                    try:
                        info = _read_figure(probe, color, mask)
                    except ValueError:
                        continue
                    if info["axis2"] == axis and info["kept"] == cores[color]:
                        noise[color].add(p)
                        occupied.add(p)
        grid = _render_visible(under, background, frame, fill, box, noise)
        output = under[t + 1:b, l + 1:r].copy()
        output[output == background] = fill
        try:
            predicted, scene = _read(grid)
            if not np.array_equal(predicted, output):
                continue
            if any(scene["figures"][c]["kept"] != cores[c] for c in colors):
                continue
        except ValueError:
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded scene construction could not place the requested figures")


def _key(grid):
    return (grid.shape, grid.astype(np.uint8).tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    backgrounds = [int(c) for c in rng.permutation(10)]
    first_parity = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, length in (("train", num_train), ("test", num_test)):
        for index in range(length):
            count = (1 if index == 0 else 2 if index == 1 else int(rng.integers(1, 3))) if split == "train" else 3
            parity_demo = (first_parity if index == 0 else 1 - first_parity) if split == "train" and index < 2 else None
            fill_matches = (index == 0) if split == "train" and index < 2 else None
            for _ in range(12):
                pair = _make_pair(rng, count, backgrounds[len(seen) % 10], parity_demo,
                                  fill_matches, split == "train" and index == 0)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not avoid duplicate inputs")
    return puzzle


def _measure(pair):
    output, scene = _read(np.asarray(pair["input"]))
    t, b, l, r = scene["box"]
    fill = scene["fill"]
    naive = np.full(output.shape, fill, dtype=np.int8)
    longest_relay = 0
    for color, info in scene["figures"].items():
        # A shortcut that trusts every exterior pixel facing the unknown area.
        for y, x in info["candidates"]:
            if t < y < b and l < x < r:
                naive[y - t - 1, x - l - 1] = color
        distances = {p: 0 for p in info["anchors"]}
        queue = deque(sorted(info["anchors"]))
        while queue:
            p = queue.popleft()
            for q in _neighbors({p}) & info["kept"]:
                if q not in distances:
                    distances[q] = distances[p] + 1
                    queue.append(q)
        longest_relay = max(longest_relay, max((distances[p] for p in info["hidden"]), default=0))
    return output, scene, {"shape": tuple(np.asarray(pair["input"]).shape),
                          "output_shape": output.shape, "count": len(scene["figures"]),
                          "background": scene["background"], "fill_matches": fill == scene["background"],
                          "parities": tuple(sorted(i["axis2"] % 2 for i in scene["figures"].values())),
                          "left": sum(i["axis2"] < 2 * l for i in scene["figures"].values()),
                          "relay": longest_relay,
                          "false_hidden": int(np.count_nonzero(naive != output)),
                          "hidden": int(np.count_nonzero(output != fill)),
                          "rejected": sum(len(i["rejected"]) for i in scene["figures"].values()),
                          "components": tuple(len(list(_components(i["kept"]))) for i in scene["figures"].values()),
                          "clean": sum(not i["rejected"] and not i["orphans"] for i in scene["figures"].values()),
                          "bbox_wrong": sum((_bbox(i["observed"])[2] + _bbox(i["observed"])[3]) != i["axis2"]
                                            for i in scene["figures"].values())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, metrics = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, truth = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, scene, info = _measure(pair)
            assert np.array_equal(predicted, truth), "wrong reconstructed crop"
            assert info["hidden"] > 0 and info["false_hidden"] > 0
            assert info["relay"] >= 2 and info["rejected"] > 0
            assert info["bbox_wrong"] > 0
            t, bottom, l, r = scene["box"]
            mask = scene["mask"]
            assert truth.shape == (bottom - t - 1, r - l - 1)
            for color, figure in scene["figures"].items():
                axis = figure["axis2"]
                kept = figure["kept"]
                assert kept and all((y, axis - x) in kept for y, x in kept)
                assert figure["hidden"]
                assert not (figure["rejected"] & kept)
                # Every inferred covered pixel has an actually visible witness.
                assert all(not mask[y, axis - x] and int(inp[y, axis - x]) == color
                           for y, x in kept if mask[y, x])
                # Deletion must disconnect an apparently attached false branch.
                if figure["orphans"]:
                    assert any((group & figure["anchors"]) and (group & figure["orphans"])
                               for group in _components(figure["observed"]))
            metrics[split].append(info)
    training, testing = metrics["train"], metrics["test"]
    assert training[0]["count"] == 1 and training[1]["count"] == 2
    assert training[0]["fill_matches"] and not training[1]["fill_matches"]
    assert max(training[0]["components"]) >= 2
    assert {p for info in training[:2] for p in info["parities"]} == {0, 1}
    assert len({info["background"] for info in training}) >= 2
    assert all(info["count"] <= 2 for info in training)
    assert all(info["count"] == 3 for info in testing)


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, no saved grids, bounded aggregate metrics and mutation checks."""
    signatures, shapes, output_shapes, structures, backgrounds = set(), set(), set(), set(), set()
    fills, parities, approaches = set(), set(), set()
    totals = {"pairs": 0, "hidden_cells": 0, "false_hidden_cells": 0,
              "rejected_visible_cells": 0, "max_relay": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        if seed < 3:
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, scene, f = _measure(pair)
                shapes.add(f["shape"])
                output_shapes.add(f["output_shape"])
                backgrounds.add(f["background"])
                fills.add(f["fill_matches"])
                parities.update(f["parities"])
                approaches.add((f["count"], f["left"]))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair["input"].ravel())
                structures.add(hashlib.sha256(bytes(pair["input"].shape) + normalized).hexdigest())
                totals["pairs"] += 1
                totals["hidden_cells"] += f["hidden"]
                totals["false_hidden_cells"] += f["false_hidden"]
                totals["rejected_visible_cells"] += f["rejected"]
                totals["max_relay"] = max(totals["max_relay"], f["relay"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                for color, figure in scene["figures"].items():
                    # Explicit set enumeration cross-checks the convolution axis.
                    cells = figure["observed"]
                    scores = [sum((y, a - x) in cells for y, x in cells)
                              for a in range(2 * pair["input"].shape[1] - 1)]
                    assert scores.count(max(scores)) == 1
                    assert scores.index(max(scores)) == figure["axis2"]
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 35 and len(output_shapes) >= 30
    assert backgrounds == set(range(10)) and fills == {False, True} and parities == {0, 1}
    assert {(1, 0), (1, 1), (2, 1), (3, 1), (3, 2)} <= approaches
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(10000 + 10 * a + b, a, b)
            validate_generated_puzzle(puzzle)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    bad = generate_puzzle(987, 2, 1)
    out = bad["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted a corrupt output")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "color_normalized_structures": len(structures), **totals}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
