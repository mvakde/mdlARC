"""Competitive pinwheel fields: stamp placement, directed growth, arrival races.

Only NumPy and the standard library are used.  Generation renders segment
intervals; solve walks the paths one pixel at a time.  Neither reads parent
files, example labels, seeds, or generation parameters to interpret an input.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np


_INF = 10**9
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be integer rectangles, dimensions 1..30, colors 0..9")
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _bbox(binary):
    rr, cc = np.nonzero(binary)
    if not len(rr):
        raise ValueError("empty encoding")
    return int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())


def _separated(a, b):
    return (a[1] + 1 < b[0] or b[1] + 1 < a[0]
            or a[3] + 1 < b[2] or b[3] + 1 < a[2])


def _read_input(input_grid):
    """Recover the framed placement board and the four-phase cross stamp."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) <= grid.size // 2:
        raise ValueError("background must be a strict majority")
    candidates = []
    for frame in map(int, colors):
        if frame == bg:
            continue
        r0, r1, c0, c1 = _bbox(grid == frame)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        patch = grid[r0:r1+1, c0:c1+1]
        rim = np.ones(patch.shape, dtype=bool)
        rim[1:-1, 1:-1] = False
        if not np.all(patch[rim] == frame) or np.any(patch[~rim] == frame):
            continue
        board = patch[1:-1, 1:-1]
        signal_colors = set(map(int, np.unique(board))) - {bg}
        if len(signal_colors) != 1:
            continue
        signal = signal_colors.pop()
        if signal == frame:
            continue
        inside = np.zeros(grid.shape, dtype=bool)
        inside[r0:r1+1, c0:c1+1] = True
        outside_support = (grid != bg) & ~inside
        if not np.any(outside_support):
            continue
        sb = _bbox(outside_support)
        if not _separated((r0, r1, c0, c1), sb):
            continue
        sr0, sr1, sc0, sc1 = sb
        stamp = grid[sr0:sr1+1, sc0:sc1+1].copy()
        n = stamp.shape[0]
        if stamp.shape != (n, n) or n not in (3, 5, 7):
            continue
        length = n // 2
        expected = np.full(stamp.shape, bg, dtype=int)
        phases = []
        for dy, dx in _DIRECTIONS:
            color = int(stamp[length+dy, length+dx])
            phases.append(color)
            for d in range(1, length+1):
                expected[length+dy*d, length+dx*d] = color
        if (len(set(phases)) != 4 or set(phases) & {bg, frame, signal}
                or not np.array_equal(stamp, expected)):
            continue
        mask = board == signal
        if (not all(3 <= side <= 10 for side in mask.shape)
                or any(side*n > 30 for side in mask.shape)):
            continue
        # Colors used by the board cannot leak into the stamp or blank canvas.
        if np.any((grid == signal) & ~inside):
            continue
        candidates.append({"background": bg, "frame": frame, "signal": signal,
                           "mask": mask.copy(), "stamp": stamp, "length": length,
                           "phase_colors": tuple(phases), "board_box": (r0, r1, c0, c1),
                           "stamp_box": sb})
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous board and cross stamp")
    return candidates[0]


def _centers(info):
    length = info["length"]
    n = 2*length+1
    return [(int(r)*n+length, int(c)*n+length)
            for r, c in np.argwhere(info["mask"])]


def _walk_times(shape, center, length, phase, *, hand=1, first_exit=False):
    """Arrival clock of one arm; the clock advances outside the canvas too."""
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    times = np.full(shape, _INF, dtype=np.int64)
    y, x, clock = cy, cx, 0
    dy, dx = _DIRECTIONS[phase]
    stopped = False

    def travel(vy, vx, steps):
        nonlocal y, x, clock, stopped
        if stopped:
            return
        for _ in range(steps):
            y += vy
            x += vx
            clock += 1
            if 0 <= y < h and 0 <= x < w:
                times[y, x] = min(clock, int(times[y, x]))
            elif first_exit:
                stopped = True
                return

    travel(dy, dx, radius)
    dy, dx = hand*dx, -hand*dy
    distance = max(cy, cx, h-1-cy, w-1-cx)
    # Every four stages move the next envelope outward by eight pixels.
    # Two surplus cycles include all possible later in-bounds re-entries.
    for k in range(4*(distance//8+2)):
        travel(dy, dx, 4*k+2)
        travel(hand*dx, -hand*dy, radius-2)
        travel(dy, dx, radius)
        dy, dx = hand*dx, -hand*dy
    return times


def _interval_times(shape, center, length, phase):
    """Closed-form stage positions and clocks; no endpoint walking or bitmaps.

    Segment starts are excluded so a shared endpoint is not a second arrival.
    This separate calculation is used for generation and label validation.
    """
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy -= center[0]
    xx -= center[1]
    for _ in range(phase):
        yy, xx = -xx, yy  # inverse clockwise rotation, into north-arm coordinates
    result = np.full(shape, _INF, dtype=np.int64)
    radial = (xx == 0) & (yy <= -1) & (yy >= -radius)
    result[radial] = -yy[radial]
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    for q in range(bound//8+2):
        starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                  (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
        for s, ((y, x), (dy, dx)) in enumerate(zip(
                starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
            k = 4*q+s
            run = 4*k+2
            start_clock = radius + 2*k*(k+radius-1)
            along = (yy-y)*dy + (xx-x)*dx
            across = (yy-y)*dx - (xx-x)*dy
            segments = (
                ((across == 0) & (along >= 1) & (along <= run), start_clock+along),
                ((along == run) & (across >= 1) & (across <= radius-2),
                 start_clock+run+across),
                ((across == radius-2) & (along >= run+1) & (along <= run+radius),
                 start_clock+radius-2+along),
            )
            for members, clock in segments:
                result[members] = np.minimum(result[members], clock[members])
    return result


def _offer(best, count, color_grid, score, color):
    less = score < best
    tied = (score == best) & (score < _INF)
    count[less] = 1
    count[tied] += 1
    color_grid[less] = color
    best[less] = score[less]


def _field(info, method, *, diagnostics=False):
    """The simultaneous race is commutative in source and arm enumeration."""
    n = 2*info["length"]+1
    shape = tuple(side*n for side in info["mask"].shape)
    bg = info["background"]
    best = np.full(shape, _INF, dtype=np.int64)
    count = np.zeros(shape, dtype=np.int64)
    color_grid = np.full(shape, bg, dtype=np.int64)
    centers = _centers(info)
    if diagnostics:
        yy, xx = np.indices(shape)
        near_best = best.copy()
        near_count = count.copy()
        near_color = color_grid.copy()
        overwrite = color_grid.copy()
    for center in centers:
        if diagnostics:
            distance = abs(yy-center[0]) + abs(xx-center[1])
        for phase, color in enumerate(info["phase_colors"]):
            time = method(shape, center, info["length"], phase)
            _offer(best, count, color_grid, time, color)
            if diagnostics:
                score = np.where(time < _INF, distance, _INF)
                _offer(near_best, near_count, near_color, score, color)
                overwrite[time < _INF] = color
    ties = count > 1
    output = np.where(count == 1, color_grid, bg)
    # A seed's hollow origin has clock zero with no ink, so stays background.
    for center in centers:
        output[center] = bg
        ties[center] = False
    result = {"output": output.astype(np.int8), "time": best,
              "ties": ties, "coverage": best < _INF}
    if diagnostics:
        nearest = np.where(near_count == 1, near_color, bg)
        for center in centers:
            nearest[center] = overwrite[center] = bg
        result.update({"nearest": nearest.astype(np.int8),
                       "overwrite": overwrite.astype(np.int8),
                       "nearest_differences": int(np.count_nonzero(output != nearest)),
                       "overwrite_differences": int(np.count_nonzero(output != overwrite))})
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone, using the family's fixed clockwise rule."""
    return _field(_read_input(input_grid), _walk_times)["output"]


def _stamp_only(info):
    n = info["stamp"].shape[0]
    h, w = info["mask"].shape
    result = np.full((h*n, w*n), info["background"], dtype=np.int8)
    for r, c in np.argwhere(info["mask"]):
        result[r*n:(r+1)*n, c*n:(c+1)*n] = info["stamp"]
    return result


def _stamp(length, bg, phases):
    n = 2*length+1
    result = np.full((n, n), bg, dtype=np.int8)
    for (dy, dx), color in zip(_DIRECTIONS, phases):
        for d in range(1, length+1):
            result[length+dy*d, length+dx*d] = color
    return result


def _pack(rng, mask, stamp, bg, signal, frame):
    board = np.full((mask.shape[0]+2, mask.shape[1]+2), frame, dtype=np.int8)
    board[1:-1, 1:-1] = np.where(mask, signal, bg)
    horizontal = bool(rng.integers(2))
    reverse = bool(rng.integers(2))
    gap = int(rng.integers(1, 5))
    rim = int(rng.integers(2))
    bh, bw = board.shape
    sh, sw = stamp.shape
    base_h = (max(bh, sh) if horizontal else bh+sh+gap) + 2*rim
    base_w = (bw+sw+gap if horizontal else max(bw, sw)) + 2*rim
    h = base_h + int(rng.integers(1, 6))
    w = base_w + int(rng.integers(1, 6))
    foreground = int(np.count_nonzero(board != bg) + np.count_nonzero(stamp != bg))
    while h*w <= 2*foreground:
        if h <= w:
            h += 1
        else:
            w += 1
    assert h <= 30 and w <= 30
    if horizontal:
        start = rim + int(rng.integers(w-base_w+1))
        bc, sc = ((start+sw+gap, start) if reverse else (start, start+bw+gap))
        br = rim + int(rng.integers(h-bh-2*rim+1))
        sr = rim + int(rng.integers(h-sh-2*rim+1))
    else:
        start = rim + int(rng.integers(h-base_h+1))
        br, sr = ((start+sh+gap, start) if reverse else (start, start+bh+gap))
        bc = rim + int(rng.integers(w-bw-2*rim+1))
        sc = rim + int(rng.integers(w-sw-2*rim+1))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[br:br+bh, bc:bc+bw] = board
    grid[sr:sr+sh, sc:sc+sw] = stamp
    return grid


def _rank_mask(rng, shape, count, kind):
    """Exactly count sources; priorities provide structural, not color, variety."""
    h, w = shape
    rr, cc = np.indices(shape)
    jitter = rng.random(shape)
    if kind == "cluster":
        y, x = int(rng.integers(h)), int(rng.integers(w))
        score = abs(rr-y)+abs(cc-x)+jitter*.8
    elif kind == "diagonal":
        score = abs(rr/max(1, h-1)-cc/max(1, w-1))*10+jitter
    elif kind == "ring":
        score = np.minimum.reduce((rr, cc, h-1-rr, w-1-cc))*3+jitter
    elif kind == "stripes":
        period = int(rng.integers(2, 4))
        score = (rr % period)*3+jitter
    elif kind == "islands":
        score = ((rr % 2)+(cc % 2))*3+jitter
    elif kind == "row":
        score = rr*3+jitter
    else:
        score = jitter
    mask = np.zeros(shape, dtype=bool)
    mask.flat[np.argsort(score.ravel())[:count]] = True
    turns = int(rng.integers(4))
    return np.rot90(mask, turns).copy()


def _info(mask, length, palette):
    return {"mask": mask, "length": length, "background": int(palette[0]),
            "signal": int(palette[1]), "frame": int(palette[2]),
            "phase_colors": tuple(map(int, palette[3:7])),
            "stamp": _stamp(length, int(palette[0]), palette[3:7])}


def _multi_witness(rng, length, palette):
    """A diagonal pair exposes cancellation and travel time versus proximity."""
    maximum = 30//(2*length+1)
    minimum = 5 if length <= 2 else 3
    offset = 2 if length <= 2 else 1
    for _ in range(32):
        shape = tuple(map(int, rng.integers(minimum, maximum+1, size=2)))
        mask = np.zeros(shape, dtype=bool)
        y = int(rng.integers(shape[0]-offset))
        x = int(rng.integers(shape[1]-offset))
        mask[y, x] = mask[y+offset, x+offset] = True
        mask = np.rot90(mask, int(rng.integers(4))).copy()
        info = _info(mask, length, palette)
        field = _field(info, _interval_times, diagnostics=True)
        if (np.any(field["ties"]) and field["nearest_differences"] > 0
                and field["overwrite_differences"] > 0):
            return info, field
    # Checked, translation-free fallback; no arbitrary exception to the rule.
    mask = np.zeros((minimum, minimum), dtype=bool)
    mask[0, 0] = mask[offset, offset] = True
    info = _info(mask, length, palette)
    field = _field(info, _interval_times, diagnostics=True)
    assert np.any(field["ties"]) and field["nearest_differences"] > 0
    return info, field


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    varied_palette = bool(rng.random() < .8)
    lengths = list(map(int, rng.choice([1, 2, 3], size=2, replace=False)))
    if num_train >= 3:
        lengths.append(next(n for n in (1, 2, 3) if n not in lengths))
    while len(lengths) < num_train:
        lengths.append(int(rng.integers(1, 4)))
    seen_lengths = sorted(set(lengths))
    test_lengths = list(map(int, rng.choice(seen_lengths, size=num_test)))
    if num_test > 1:
        test_lengths[1] = int(rng.choice([l for l in seen_lengths if l != test_lengths[0]]))
    puzzle = {"train": [], "test": []}
    used = set()
    max_train_sources = 2
    for index, length in enumerate(lengths+test_lengths):
        split = "train" if index < num_train else "test"
        pair_palette = rng.permutation(10) if varied_palette else palette
        # In varied puzzles, the first two demonstrations exchange literal roles.
        if varied_palette and index == 0:
            palette = pair_palette.copy()
        if varied_palette and index == 1:
            pair_palette = np.roll(palette, 1)
        for _ in range(32):
            if index == 1:
                info, field = _multi_witness(rng, length, pair_palette)
            else:
                maximum = 30//(2*length+1)
                minimum = 4 if length == 1 else 3
                shape = tuple(map(int, rng.integers(minimum, maximum+1, size=2)))
                if index == 0:
                    mask = np.zeros(shape, dtype=bool)
                    # Mix centered and edge sources; a single source shows the
                    # four colored phases without any competing emitter.
                    if rng.random() < .5:
                        y, x = (shape[0]//2, shape[1]//2)
                    else:
                        y, x = (int(rng.choice([0, shape[0]-1])),
                                int(rng.choice([0, shape[1]-1])))
                    mask[y, x] = True
                else:
                    low = 2 if split == "train" else max_train_sources+1
                    high = min(shape[0]*shape[1], 6 if split == "train" else max_train_sources+6)
                    count = int(rng.integers(low, high+1))
                    kind = str(rng.choice(["cluster", "diagonal", "ring", "stripes",
                                           "islands", "row", "random"]))
                    mask = _rank_mask(rng, shape, count, kind)
                info = _info(mask, length, pair_palette)
                field = _field(info, _interval_times)
            grid = _pack(rng, info["mask"], info["stamp"], info["background"],
                         info["signal"], info["frame"])
            key = _key(grid)
            if key in used:
                continue
            used.add(key)
            puzzle[split].append({"input": grid, "output": field["output"]})
            if split == "train":
                max_train_sources = max(max_train_sources, int(info["mask"].sum()))
            break
        else:
            raise RuntimeError("failed to make a distinct input")
    rng.shuffle(puzzle["train"])
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantics, motif retention, competitions, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            info = _read_input(a)
            field = _field(info, _interval_times, diagnostics=(split == "train"))
            assert np.array_equal(b, field["output"]), "incorrect arrival-race output"
            assert info["frame"] not in b and info["signal"] not in b
            assert set(map(int, np.unique(b))) == {info["background"], *info["phase_colors"]}
            assert not np.array_equal(b, _stamp_only(info)), "B must add winding cells"
            n, l = 2*info["length"]+1, info["length"]
            assert b.shape == tuple(n*s for s in info["mask"].shape)
            for center in _centers(info):
                assert b[center] == info["background"], "source origins stay hollow"
                for (dy, dx), color in zip(_DIRECTIONS, info["phase_colors"]):
                    for d in range(1, l+1):
                        assert b[center[0]+dy*d, center[1]+dx*d] == color, "stamp spokes survive"
            records[split].append((info, field))
    train = records["train"]
    assert any(int(i["mask"].sum()) == 1 for i, _ in train), "show an isolated winding"
    witnesses = [(i, f) for i, f in train if int(i["mask"].sum()) >= 2
                 and np.any(f["ties"]) and f["nearest_differences"] > 0
                 and f["overwrite_differences"] > 0]
    assert witnesses, "show ties, collisions, and travel time versus proximity"
    info, _ = witnesses[0]
    first_only = dict(info, mask=np.zeros(info["mask"].shape, dtype=bool))
    first_only["mask"][tuple(np.argwhere(info["mask"])[0])] = True
    assert not np.array_equal(_field(first_only, _interval_times)["output"],
                              _field(info, _interval_times)["output"]), "A must matter"
    lengths = {i["length"] for i, _ in train}
    max_count = max(int(i["mask"].sum()) for i, _ in train)
    for info, _ in records["test"]:
        assert info["length"] in lengths, "do not test an unexplained new jog length"
        assert int(info["mask"].sum()) > max_count, "test must increase the number of emitters"
    return None


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and (diagonal or abs(dy)+abs(dx) == 1)]
    count = 0
    while remaining:
        stack = [remaining.pop()]
        count += 1
        while stack:
            y, x = stack.pop()
            for dy, dx in offsets:
                cell = y+dy, x+dx
                if cell in remaining:
                    remaining.remove(cell)
                    stack.append(cell)
    return count


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed variation audit plus geometric and negative regressions."""
    # Literal clock witnesses distinguish a jog from an ordinary square spiral.
    times = _walk_times((30, 30), (12, 12), 3, 0)
    for (dy, dx), clock in (((-3, 0), 3), ((-3, 2), 5), ((-2, 2), 6),
                            ((-2, 5), 9), ((4, 5), 15), ((4, 4), 16), ((7, 4), 19)):
        assert times[12+dy, 12+dx] == clock
    for length in (1, 2, 3):
        for shape, center in (((30, 30), (1, 1)), ((23, 29), (21, 27)),
                              ((11, 17), (5, 9))):
            for phase in range(4):
                a = _walk_times(shape, center, length, phase)
                b = _interval_times(shape, center, length, phase)
                assert np.array_equal(a, b), "walk/interval arrival clocks disagree"
                big = _interval_times((80, 80), (40, 40), length, phase)
                top, left = 40-center[0], 40-center[1]
                assert np.array_equal(a, big[top:top+shape[0], left:left+shape[1]])
    assert np.array_equal(_walk_times((30, 30), (15, 15), 1, 0),
                          _walk_times((30, 30), (15, 15), 2, 0))
    assert _walk_times((30, 30), (1, 1), 1, 0)[0, 5] == 7
    assert _walk_times((30, 30), (1, 1), 1, 0, first_exit=True)[0, 5] == _INF
    measurements = Counter()
    shapes, boards, masks, lengths, palettes, layouts, signatures = [set() for _ in range(7)]
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                field = _field(info, _interval_times)
                shape = pair["output"].shape
                shapes.add(shape)
                boards.add(info["mask"].shape)
                masks.add(_key(info["mask"]))
                lengths.add(info["length"])
                palettes.add((info["background"], info["signal"], info["frame"], *info["phase_colors"]))
                a, b = info["board_box"], info["stamp_box"]
                layout = ("left" if a[3] < b[2] else "right") if (
                    a[3]+1 < b[2] or b[3]+1 < a[2]) else ("above" if a[1] < b[0] else "below")
                layouts.add(layout)
                measurements["pairs"] += 1
                measurements["tie_pixels"] += int(field["ties"].sum())
                measurements["pairs_with_ties"] += int(np.any(field["ties"]))
                measurements["disconnected_masks"] += int(_components(info["mask"], True) > 1)
                measurements["diagonal_only_masks"] += int(_components(info["mask"]) > 1
                                                            and _components(info["mask"], True) == 1)
                measurements["rectangular_boards"] += int(shape[0] != shape[1])
                measurements["edge_sources"] += int(any(np.any(side) for side in (
                    info["mask"][0], info["mask"][-1], info["mask"][:, 0], info["mask"][:, -1])))
                measurements["full_boards"] += int(np.all(info["mask"]))
                measurements["sources_"+split] += int(info["mask"].sum())
    assert len(signatures) == 42
    assert lengths == {1, 2, 3}
    assert len(shapes) >= 25 and len(boards) >= 20 and len(masks) >= 100
    assert len(palettes) >= 50
    assert layouts == {"left", "right", "above", "below"}
    for name in ("tie_pixels", "disconnected_masks", "diagonal_only_masks",
                 "rectangular_boards", "edge_sources", "full_boards"):
        assert measurements[name] > 0, name
    assert max(max(s) for s in shapes) == 30
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for counts in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(321, *counts)
        validate_generated_puzzle(p)
        assert tuple(map(len, (p["train"], p["test"]))) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Changing a label and filling an actual simultaneous-arrival gap are rejected.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad output accepted")
    multi = next(p for p in puzzles[0]["train"] if np.any(
        _field(_read_input(p["input"]), _interval_times)["ties"]))
    info = _read_input(multi["input"])
    field = _field(info, _interval_times)
    y, x = np.argwhere(field["ties"])[0]
    assert multi["output"][y, x] == info["background"]
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    target = next(p for p in damaged["train"] if np.array_equal(p["input"], multi["input"]))
    target["output"][y, x] = info["phase_colors"][0]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("filled cancellation accepted")
    # Input-only solving also handles unsampled dense members, rotated layouts,
    # and arbitrary bijective reassignments of the visible color roles.
    for length in (1, 2, 3):
        dense = _info(np.ones((30//(2*length+1),)*2, dtype=bool), length, np.arange(10))
        packed = _pack(np.random.default_rng(100+length), dense["mask"], dense["stamp"],
                       dense["background"], dense["signal"], dense["frame"])
        assert np.array_equal(solve(packed), _field(dense, _interval_times)["output"])
    for pair in puzzles[0]["train"]:
        assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
        recolor = np.array([6, 8, 1, 5, 9, 0, 4, 2, 7, 3])
        assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
    isolated = next(p for p in puzzles[0]["train"] if _read_input(p["input"])["mask"].sum() == 1)
    isolated_info = _read_input(isolated["input"])
    reflected_turns = _field(isolated_info, lambda shape, center, length, phase:
                             _walk_times(shape, center, length, phase, hand=-1))["output"]
    assert not np.array_equal(isolated["output"], reflected_turns), "show clockwise handedness"
    # Simultaneous aggregation must be unaffected by enumeration order.
    first = _field(info, _interval_times)["output"]
    best = np.full(first.shape, _INF, dtype=np.int64)
    count = np.zeros(first.shape, dtype=np.int64)
    colors = np.full(first.shape, info["background"], dtype=np.int64)
    for center in reversed(_centers(info)):
        for phase in reversed(range(4)):
            _offer(best, count, colors, _interval_times(first.shape, center, info["length"], phase),
                   info["phase_colors"][phase])
    reverse = np.where(count == 1, colors, info["background"])
    for center in _centers(info):
        reverse[center] = info["background"]
    assert np.array_equal(first, reverse)
    return {"seeds": 42, **dict(measurements), "output_shapes": len(shapes),
            "board_shapes": len(boards), "mask_patterns": len(masks),
            "role_palettes": len(palettes), "layouts": sorted(layouts),
            "lengths": sorted(lengths), "independent_verification": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
