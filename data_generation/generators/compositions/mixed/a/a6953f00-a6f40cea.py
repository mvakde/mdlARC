"""Reconstruct occluded loops, then query a corner using canvas-side parity.

Scene rendering and the input reader are separate. No scene descriptors, seeds,
labels, parent modules, or generation-time caches are used by solve().
"""

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not np.issubdtype(array.dtype, np.integer)
            or any(d < 1 or d > 30 for d in array.shape)
            or np.any(array < 0) or np.any(array > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return array


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _bbox(points):
    yy, xx = zip(*points)
    return min(yy), max(yy), min(xx), max(xx)


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _components(points):
    pending = set(points)
    groups = []
    while pending:
        first = min(pending)
        pending.remove(first)
        group, stack = {first}, [first]
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in pending:
                    pending.remove(q)
                    stack.append(q)
                    group.add(q)
        groups.append(group)
    return groups


def _filled_frames(grid):
    """Uniform rectangular components bounded by a complete foreign border."""
    bg = int(grid[0, 0])
    n = grid.shape[0]
    result = []
    for color in np.unique(grid):
        points = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
        for group in _components(points):
            t, b, l, r = _bbox(group)
            if len(group) != (b - t + 1) * (r - l + 1):
                continue
            if min(t, l) == 0 or max(b, r) == n - 1:
                continue
            box = (t - 1, b + 1, l - 1, r + 1)
            border = {int(grid[p]) for p in _perimeter(box)}
            if bg not in border and int(color) not in border:
                result.append({"box": box, "fill": int(color),
                               "colors": border, "area": len(group)})
    return sorted(result, key=lambda f: (-f["area"], f["box"]))


def _phase(grid, visible):
    """A solid is a two-parity map with equal values; alternation has two."""
    values, counts = {}, [0, 0]
    for y, x in sorted(visible):
        parity, color = (y + x) % 2, int(grid[y, x])
        if parity in values and values[parity] != color:
            return None
        values[parity] = color
        counts[parity] += 1
    if min(counts) < 2:
        return None
    return values


def _reconstruct(input_grid):
    """Read all geometry and phases before taking either candidate crop."""
    grid = _grid(input_grid)
    n = grid.shape[0]
    _require(grid.shape == (n, n), "canvas must be square")
    bg = int(grid[0, 0])
    _require(np.all(grid[0] == bg) and np.all(grid[-1] == bg)
             and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg),
             "outer border must be background")
    frames = _filled_frames(grid)
    _require(bool(frames), "no filled main frame")
    main = frames[0]
    _require(len(frames) == 1 or main["area"] > frames[1]["area"],
             "largest filled frame must be unique")
    t, b, l, r = main["box"]
    _require(b - t >= 3 and r - l >= 3, "main interior must be at least 2x2")
    # A fully visible background-filled loop is transparent. Other small filled
    # rectangles are opaque distractors, disjoint from all middle perimeters.
    opaque = [f for f in frames if f is main or f["fill"] != bg]
    masked = set()
    for frame in opaque:
        cells = _area(frame["box"])
        _require(not masked & cells, "opaque masks overlap")
        masked.update(cells)
    fragments = _components({tuple(map(int, p)) for p in np.argwhere(grid != bg)} - masked)
    _require(len(fragments) <= 12, "at most twelve visible fragments are supported")
    candidates = []
    # A rectangle with all four extrema exposed has at most two components
    # outside one rectangular occluder. Small opaque distractors never cut loops.
    for i, first in enumerate(fragments):
        for j in range(i, len(fragments)):
            visible = first if i == j else first | fragments[j]
            box = _bbox(visible)
            y0, y1, x0, x1 = box
            if y1 - y0 < 2 or x1 - x0 < 2:
                continue
            full = _perimeter(box)
            if full - masked != visible:
                continue
            phase = _phase(grid, visible)
            if phase is None:
                continue
            candidates.append({"parts": frozenset((i, j)), "box": box,
                               "full": full, "phase": phase,
                               "visible": visible, "hidden": full & _area(main["box"])})
    covers = []

    def cover(remaining, chosen, occupied_halo):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for candidate in candidates:
            parts = candidate["parts"]
            if (first in parts and parts <= remaining
                    and not candidate["full"] & occupied_halo):
                cover(remaining - parts, chosen + [candidate],
                      occupied_halo | _halo(candidate["full"]))

    cover(set(range(len(fragments))), [], set())
    _require(len(covers) == 1, "perimeter geometry is ambiguous or incomplete")
    loops = covers[0]
    middle_colors = {c for loop in loops for c in loop["phase"].values()}
    forbidden = {bg, main["fill"]} | set().union(*(f["colors"] for f in opaque))
    forbidden.update(f["fill"] for f in opaque)
    _require(middle_colors.isdisjoint(forbidden), "middle and opaque palette roles overlap")
    _require(any(any(t < y < b and l < x < r for y, x in loop["hidden"])
                 for loop in loops), "at least one loop must enter the strict interior")
    restored = np.full((b - t - 1, r - l - 1), main["fill"], dtype=np.int8)
    for loop in loops:
        for y, x in loop["hidden"]:
            if t < y < b and l < x < r:
                restored[y - t - 1, x - l - 1] = loop["phase"][(y + x) % 2]
    return restored, {"main": main, "loops": loops, "background": bg,
                      "opaque_count": len(opaque)}


def _corner(restored, side):
    return (restored[:2, :2] if side % 2 else restored[:2, -2:]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    restored, _ = _reconstruct(input_grid)
    return _corner(restored, np.asarray(input_grid).shape[0])


def _rand(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _target_box(rng, main, n, left):
    """A sampled loop whose hidden turn or side meets one top 2x2 query."""
    t, b, l, r = main
    center = (l + r) // 2
    limit = center - l - 1 if left else r - center - 1
    shape = _rand(rng, 0, 3)  # turn, horizontal side, vertical side, single cell
    if shape == 1 and limit >= 3:
        inset = _rand(rng, 3, limit)
    else:
        inset = 1 if shape == 3 else _rand(rng, 1, min(2, limit))
    inner = l + inset if left else r - inset
    outer = _rand(rng, 1, l - 2) if left else _rand(rng, r + 2, n - 2)
    if rng.random() < .65:
        y0 = _rand(rng, 1, t - 2)
        y1 = (_rand(rng, t + 3, b - 1) if shape == 2
              else t + (1 if shape == 3 else _rand(rng, 1, 2)))
    else:
        y0 = t + (1 if shape == 3 else _rand(rng, 1, 2))
        y1 = _rand(rng, b + 2, n - 2)
    return y0, y1, min(inner, outer), max(inner, outer)


def _random_box(rng, n):
    yy = sorted(int(v) for v in rng.choice(np.arange(1, n - 1), 2, replace=False))
    xx = sorted(int(v) for v in rng.choice(np.arange(1, n - 1), 2, replace=False))
    return yy[0], yy[1], xx[0], xx[1]


def _geometry(rng, n, count, two_targets, require_extra=False):
    h, w = _rand(rng, 4, min(14, n - 8)), _rand(rng, 6, min(16, n - 8))
    t, l = _rand(rng, 3, n - h - 5), _rand(rng, 3, n - w - 5)
    main = (t, t + h + 1, l, l + w + 1)
    mask, inside = _area(main), _area((t + 1, t + h, l + 1, l + w))
    boxes, occupied = [], set()

    def accept(box, crossing):
        y0, y1, x0, x1 = box
        if y1 - y0 < 2 or x1 - x0 < 2:
            return False
        full = _perimeter(box)
        visible = full - mask
        if (not visible or _bbox(visible) != box or bool(full & inside) != crossing
                or full & _halo(occupied)):
            return False
        if not crossing and full & mask:
            return False
        # Every alternating phase must be repeatedly observable.
        if min(sum((y + x) % 2 == p for y, x in visible) for p in (0, 1)) < 2:
            return False
        boxes.append(box)
        occupied.update(full)
        return True

    target_sides = [True, False] if two_targets else [bool(n % 2)]
    for left in target_sides:
        if not accept(_target_box(rng, main, n, left), True):
            return None
    for _ in range(count - len(boxes)):
        for _ in range(650):
            if accept(_random_box(rng, n), True):
                break
        else:
            return None
    # Optional complete transparent loops are small, exterior, and regular.
    if rng.random() < .65:
        for _ in range(70):
            hh, ww = _rand(rng, 3, 5), _rand(rng, 3, 5)
            y, x = _rand(rng, 1, n - hh - 1), _rand(rng, 1, n - ww - 1)
            box = (y, y + hh - 1, x, x + ww - 1)
            if (hh - 2) * (ww - 2) < h * w and accept(box, False):
                break
    extras = []
    if require_extra or rng.random() < .6:
        blocked = _halo(occupied | mask)
        for _ in range(80):
            hh, ww = _rand(rng, 3, 4), _rand(rng, 3, 5)
            y, x = _rand(rng, 1, n - hh - 1), _rand(rng, 1, n - ww - 1)
            box = (y, y + hh - 1, x, x + ww - 1)
            if not _area(box) & blocked:
                extras.append(box)
                break
    if require_extra and not extras:
        return None
    return main, boxes, extras


def _render(rng, n, geometry, distinct_fill, force_alternating):
    main, boxes, extras = geometry
    colors = [int(c) for c in rng.permutation(10)]
    bg, fill_color = colors[:2]
    frame_count = 1 + int(rng.random() < .4)
    frame_colors, middle = colors[2:2 + frame_count], colors[2 + frame_count:]
    fill = fill_color if distinct_fill else bg
    under = np.full((n, n), bg, dtype=np.int8)
    for index, box in enumerate(boxes):
        alternating = (force_alternating and index < 2) or rng.random() < .6
        a, c = [int(v) for v in rng.choice(middle, size=2, replace=False)]
        if not alternating:
            c = a
        phase = _rand(rng, 0, 1)
        for y, x in _perimeter(box):
            under[y, x] = (a, c)[(y + x + phase) % 2]
    t, b, l, r = main
    # Truth is constructed from complete middle perimeters, not from solve().
    truth = under[t + 1:b, l + 1:r].copy()
    truth[truth == bg] = fill
    grid = under.copy()
    for box in [main] + extras:
        y0, y1, x0, x1 = box
        grid[y0 + 1:y1, x0 + 1:x1] = fill if box == main else fill_color
        phase = _rand(rng, 0, 1)
        for y, x in _perimeter(box):
            grid[y, x] = frame_colors[(y + x + phase) % frame_count]
    return grid, truth


def _scene(rng, n, count, *, twin=False, distinct_fill=True):
    for _ in range(100):
        geometry = _geometry(rng, n, count, two_targets=twin or count > 1,
                             require_extra=twin)
        if geometry is None:
            continue
        grid, truth = _render(rng, n, geometry, distinct_fill, force_alternating=twin)
        try:
            restored, scene = _reconstruct(grid)
            _require(np.array_equal(restored, truth), "full hidden layer differs from renderer")
            _require(len(scene["loops"]) == len(geometry[1]), "loop count changed")
            left, right = truth[:2, :2], truth[:2, -2:]
            _require(not np.array_equal(left, right), "corner candidates agree")
            if twin:
                _require(np.count_nonzero(left != scene["main"]["fill"]) >= 2
                         and np.count_nonzero(right != scene["main"]["fill"]) >= 2,
                         "paired demonstrations must sample two hidden cells per corner")
            else:
                _require(np.any(_corner(truth, n) != scene["main"]["fill"]),
                         "selected query must depend on completion")
            return grid, truth
        except AssertionError:
            continue
    raise RuntimeError("bounded scene sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()

    def append(split, grid, truth):
        key = (grid.shape, grid.tobytes())
        _require(key not in seen, "duplicate generated input")
        seen.add(key)
        puzzle[split].append({"input": grid, "output": _corner(truth, grid.shape[0])})

    # Same content, changed original square-canvas parity. Both branches query
    # genuinely hidden alternating content. Padded cells carry no new markers.
    # The shortest training set must distinguish retaining the window fill
    # from replacing it with the exterior background. Larger sets show both.
    fill_mode = True if num_train == 2 else bool(_rand(rng, 0, 1))
    base = _rand(rng, 16, 28)
    bases = [base, int(rng.choice([v for v in range(16, 29) if abs(v - base) >= 4]))]
    for block in range(2 if num_train >= 4 else 1):
        n = bases[block]
        grid, truth = _scene(rng, n, 2, twin=True, distinct_fill=fill_mode ^ bool(block))
        padded = np.pad(grid, ((0, 1), (0, 1)), constant_values=int(grid[0, 0]))
        pair = [grid, padded]
        if _rand(rng, 0, 1):
            pair.reverse()
        for g in pair:
            append("train", g, truth)
    while len(puzzle["train"]) < num_train:
        n, count = _rand(rng, 16, 29), _rand(rng, 1, 3)
        distinct_fill = (not fill_mode if num_train == 3
                         else bool(_rand(rng, 0, 1)))
        grid, truth = _scene(rng, n, count, distinct_fill=distinct_fill)
        append("train", grid, truth)
    test_parities = [0, 1]
    rng.shuffle(test_parities)
    for index in range(num_test):
        parity = test_parities[index % 2] if num_test >= 2 else _rand(rng, 0, 1)
        n = int(rng.choice([s for s in range(20, 31) if s % 2 == parity]))
        count = 3 + int(rng.random() < .3)
        grid, truth = _scene(rng, n, count, distinct_fill=bool(_rand(rng, 0, 1)))
        append("test", grid, truth)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    restored, scene = _reconstruct(grid)
    t, b, l, r = scene["main"]["box"]
    loops = scene["loops"]
    selected = {(t + 1 + y, (l + 1 if grid.shape[0] % 2 else r - 2) + x)
                for y in range(2) for x in range(2)}
    crossing = [loop for loop in loops
                if any(t < y < b and l < x < r for y, x in loop["hidden"])]
    return {"side": grid.shape[0], "main_box": (t, b, l, r),
            "interior_shape": restored.shape, "fill": scene["main"]["fill"],
            "background": scene["background"], "frame_colors": tuple(sorted(scene["main"]["colors"])),
            "crossing_count": len(crossing), "loop_count": len(loops),
            "opaque_count": scene["opaque_count"],
            "alternating_count": sum(loop["phase"][0] != loop["phase"][1] for loop in crossing),
            "sampled_alternating": any(loop["phase"][0] != loop["phase"][1]
                                       and {(y + x) % 2 for y, x in loop["hidden"] & selected}
                                       == {0, 1} for loop in loops),
            "split_loops": sum(len(_components(loop["visible"])) == 2 for loop in loops),
            "spanning": any((box[0] < t and box[1] > b) or (box[2] < l and box[3] > r)
                            for box in [loop["box"] for loop in crossing]),
            "nested": any(a[0] < z[0] < z[1] < a[1] and a[2] < z[2] < z[3] < a[3]
                          for a in [loop["box"] for loop in loops]
                          for z in [loop["box"] for loop in loops]),
            "layout": tuple(sorted(loop["box"] for loop in loops))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts")
    seen, witnesses, phase_witnesses = set(), set(), set()
    train_fills, train_opaque_counts = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            key = (grid.shape, grid.astype(np.int8).tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            restored, scene = _reconstruct(grid)
            _require(out.shape == (2, 2) and np.array_equal(out, _corner(restored, grid.shape[0])),
                     "incorrect reconstructed parity crop")
            # Two explicit ablations, assessed from pixels rather than descriptors.
            _require(not np.array_equal(restored[:2, :2], restored[:2, -2:]),
                     "fixed-corner shortcut is untested")
            _require(np.any(out != scene["main"]["fill"]), "completion does not affect answer")
            if split == "train":
                witnesses.add(grid.shape[0] % 2)
                train_fills.add(scene["main"]["fill"] != scene["background"])
                train_opaque_counts.append(scene["opaque_count"])
                if _features(grid)["sampled_alternating"]:
                    phase_witnesses.add(grid.shape[0] % 2)
            else:
                _require(3 <= _features(grid)["crossing_count"] <= 4,
                         "tests must combine three or four hidden loops")
    _require(witnesses == {0, 1} and phase_witnesses == {0, 1},
             "training must show both parity branches sampling alternating hidden cells")
    _require(True in train_fills and all(n >= 2 for n in train_opaque_counts[:2]),
             "training must demonstrate fill retention and largest-frame selection")
    if nt >= 3:
        _require(train_fills == {False, True}, "training must show both fill identities")
    if nt >= 4:
        _require(all(len({np.asarray(p["input"]).shape[0] for p in puzzle["train"]
                          if np.asarray(p["input"]).shape[0] % 2 == parity}) >= 2
                     for parity in (0, 1)), "multiple scales per parity must be demonstrated")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(array.shape))
                h.update(array.tobytes())
    return h.hexdigest()


def self_test():
    """Streaming 42-seed checks: no trace files, sample dumps, or retained runs."""
    hashes, sizes, layouts, shapes, palettes, patches = set(), set(), set(), set(), set(), set()
    backgrounds, fills, frame_sizes, counts, parity_mismatches = set(), set(), set(), set(), set()
    extras = spanning = nested = split_loops = pair_count = 0
    for seed in range(42):
        a = 2 + seed % 9
        b = 1 + (seed // 9) % min(4, a - 1)
        puzzle = generate_puzzle(seed, a, b)
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                grid, out = pair["input"], pair["output"]
                _require(np.array_equal(solve(grid), out), "input-only solve disagrees")
                f = _features(grid)
                sizes.add(f["side"])
                layouts.add((f["side"], f["main_box"], f["layout"]))
                shapes.add(f["interior_shape"])
                palettes.add(tuple(int(c) for c in np.unique(grid)))
                mapping = {}
                patches.add(tuple(mapping.setdefault(int(c), len(mapping)) for c in out.flat))
                backgrounds.add(f["background"])
                fills.add(f["fill"] == f["background"])
                frame_sizes.add(len(f["frame_colors"]))
                counts.add(f["crossing_count"])
                parity_mismatches.add((f["side"] % 2, f["interior_shape"][1] % 2))
                extras += f["opaque_count"] > 1
                spanning += f["spanning"]
                nested += f["nested"]
                split_loops += f["split_loops"]
    _require(len(hashes) == 42 and len(layouts) >= 110, "geometric diversity collapsed")
    _require(len(sizes) >= 12 and len(shapes) >= 35 and len(palettes) >= 25,
             "size, aspect, or palette diversity collapsed")
    _require(len(patches) >= 7, "tiny outputs lack structural variety")
    _require(backgrounds == set(range(10)) and fills == {False, True}
             and frame_sizes == {1, 2}, "role variation collapsed")
    _require(counts >= {1, 2, 3, 4} and len(parity_mismatches) == 4,
             "counts or canvas/interior parity independence collapsed")
    _require(extras > 0 and spanning > 0 and nested > 0 and split_loops > 0,
             "broader loop structures absent")
    # All legal explicit count combinations, with deterministic bounded seeds.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(42000 + a * 10 + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count mismatch")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_digest(generate_puzzle(817)) == _digest(generate_puzzle(817)), "nondeterminism")
    # Padding flips parity while keeping the complete reconstructed layer fixed.
    puzzle = generate_puzzle(919)
    grid = puzzle["train"][0]["input"]
    if grid.shape[0] < 30:
        padded = np.pad(grid, ((0, 1), (0, 1)), constant_values=int(grid[0, 0]))
        a, _ = _reconstruct(grid)
        b, _ = _reconstruct(padded)
        _require(np.array_equal(a, b) and not np.array_equal(solve(grid), solve(padded)),
                 "canvas-parity intervention failed")
    # Swapping the observed colors of a sampled alternating loop must swap its
    # restored cells too. Coordinates, mask, canvas parity and other loops stay
    # fixed. This checks phase causality beyond comparing the two implementations.
    restored, scene = _reconstruct(grid)
    t, b, l, r = scene["main"]["box"]
    col = l + 1 if grid.shape[0] % 2 else r - 2
    selected = {(t + 1 + y, col + x) for y in range(2) for x in range(2)}
    loop = next(loop for loop in scene["loops"]
                if loop["phase"][0] != loop["phase"][1] and loop["hidden"] & selected)
    changed, expected = grid.copy(), restored.copy()
    for y, x in loop["visible"]:
        changed[y, x] = loop["phase"][1 - (y + x) % 2]
    for y, x in loop["hidden"]:
        if t < y < b and l < x < r:
            expected[y - t - 1, x - l - 1] = loop["phase"][1 - (y + x) % 2]
    _require(np.array_equal(_reconstruct(changed)[0], expected)
             and np.array_equal(solve(changed), _corner(expected, grid.shape[0]))
             and not np.array_equal(solve(changed), solve(grid)),
             "visible phase intervention failed")
    # All ten colors are freely renameable, including zero and background.
    rename = np.array([7, 0, 9, 2, 6, 1, 8, 4, 5, 3], dtype=np.int8)
    _require(np.array_equal(solve(rename[grid]), rename[solve(grid)]),
             "literal color identities affected the answer")
    puzzle["test"][0]["output"][0, 0] = (int(puzzle["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(hashes),
            "canvas_sides": sorted(sizes), "layouts": len(layouts),
            "interior_shapes": len(shapes), "palettes": len(palettes),
            "normalized_output_patterns": len(patches), "crossing_counts": sorted(counts),
            "with_smaller_opaque_frames": extras, "with_spanning_loops": spanning,
            "with_nested_loops": nested, "split_visible_loops": split_loops,
            "legal_count_combinations": 30,
            "interventions": ["canvas_padding", "visible_phase_swap", "color_renaming",
                              "corrupted_label_rejection"]}


if __name__ == "__main__":
    self_test()
    print("self_test passed")
