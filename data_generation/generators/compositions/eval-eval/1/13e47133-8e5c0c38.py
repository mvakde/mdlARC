"""Echo-front mosaics: supported mirror sources emit competing cyclic waves.

Only NumPy and the standard library are used. No parent module, artifact, seed,
training example, or generator state is consulted by solve().
"""
from collections import deque
import hashlib
import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _decode(value):
    a = _grid(value)
    if a.shape[0] < 5 or a.shape[1] < 7:
        raise ValueError("missing palette header and field")
    bg, separator = int(a[0, -1]), int(a[1, 0])
    if bg == separator or not np.all(a[1] == separator):
        raise ValueError("second row must be a solid separator")
    stop = np.flatnonzero(a[0] == bg)
    length = int(stop[0]) if len(stop) else 0
    cycle = tuple(map(int, a[0, :length]))
    if not (2 <= length <= 6 and len(set(cycle)) == length
            and separator not in cycle and np.all(a[0, length:] == bg)):
        raise ValueError("invalid ordered palette")
    body = a[2:]
    colors = tuple(c for c in cycle if np.any(body == c))
    if not colors or not set(map(int, np.unique(body))) <= {bg, *cycle}:
        raise ValueError("field contains an unkeyed color or no sources")
    return a, body, bg, separator, cycle, colors


def _scores(mask):
    # At doubled axis a, row self-convolution counts exactly the occupied
    # cells whose reflected position a-c is occupied, including axis cells.
    return sum((np.convolve(row, row) for row in mask.astype(np.int16)),
               np.zeros(2 * mask.shape[1] - 1, dtype=np.int32))


def _support(mask):
    scores = _scores(mask)
    winners = np.flatnonzero(scores == scores.max())
    if len(winners) != 1:
        raise ValueError("reflection support must have one maximum")
    axis = int(winners[0])
    reflected = axis - np.arange(mask.shape[1])
    valid = (0 <= reflected) & (reflected < mask.shape[1])
    keep = np.zeros_like(mask)
    keep[:, valid] = mask[:, valid] & mask[:, reflected[valid]]
    return keep, axis


def _distance(mask, manhattan=False):
    """Distance to a set; Chebyshev is outward eight-neighbor contour depth."""
    rr, cc = np.indices(mask.shape)
    answer = np.full(mask.shape, 100, dtype=np.int16)
    for r, c in np.argwhere(mask):
        dr, dc = np.abs(rr - r), np.abs(cc - c)
        np.minimum(answer, dr + dc if manhattan else np.maximum(dr, dc), out=answer)
    return answer


def _paint(masks, colors, cycle, bg, *, manhattan=False, advance=True, clear_ties=True):
    ds = np.stack([_distance(m, manhattan) for m in masks])
    distance = ds.min(axis=0)
    owner = ds.argmin(axis=0)
    tied = (ds == distance).sum(axis=0) > 1
    phases = np.array([cycle.index(c) for c in colors], dtype=np.int16)
    indices = phases[owner] + (distance if advance else 0)
    out = np.asarray(cycle, dtype=np.int8)[indices % len(cycle)]
    if clear_ties:
        out[tied] = bg
    return out, distance, tied, owner


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, body, bg, _, cycle, colors = _decode(input_grid)
    masks = [_support(body == c)[0] for c in colors]
    out = a.copy()
    out[2:] = _paint(masks, colors, cycle, bg)[0]
    return out


def _checked_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= value <= high:
        raise ValueError(name + " out of range")
    return int(value)


def _core(rng, parity, count, hollow=False, small=False):
    """Mirror geometry with changing profiles, holes, stems and islands."""
    if hollow:
        h = int(rng.integers(4, 8))
        w = int(rng.choice([5, 7] if parity == 0 else [4, 6]))
        family = 0
    else:
        upper = 5 if small else (7 if count >= 4 else 9)
        h = int(rng.integers(2, upper + 1))
        widths = [w for w in range(2, upper + 1) if (w - 1) % 2 == parity]
        w = int(rng.choice(widths))
        family = int(rng.integers(5))
    m = np.zeros((h, w), dtype=bool)
    if family == 0:
        m[0] = m[-1] = True
        m[:, 0] = m[:, -1] = True
    else:
        half = (w + 1) // 2
        level = int(rng.integers(half))
        for r in range(h):
            if family == 1:
                level = int(np.clip(level + rng.integers(-1, 2), 0, half - 1))
            elif family == 2:
                level = int(rng.integers(half))
            elif family == 3:
                level = min(half - 1, r, h - r - 1)
            else:
                level = half - 1 if r % 3 == 0 else int(rng.integers(half))
            for c in range(half):
                d = half - c - 1
                present = d <= level and (family in (1, 3) or d == level or rng.random() < .45)
                if present:
                    m[r, c] = m[r, w - c - 1] = True
        # An actual full extent, without forcing the middle row to be widest.
        r = int(rng.integers(h))
        m[r, 0] = m[r, -1] = True
        if family == 4 and h >= 5:
            m[int(rng.integers(1, h - 1))] = False
    rr, cc = np.nonzero(m)
    return m[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


def _place(rng, shape, count, first):
    occupied = np.zeros(shape, dtype=bool)
    cores, axes = [], []
    for i in range(count):
        parity = i % 2 if first else int(rng.integers(2))
        for attempt in range(100):
            m = _core(rng, parity, count, hollow=first and i == 0,
                      small=attempt >= 60)
            h, w = m.shape
            r = int(rng.integers(shape[0] - h + 1))
            c = int(rng.integers(shape[1] - w + 1))
            if rng.random() < .2:
                if rng.integers(2):
                    r = int(rng.choice([0, shape[0] - h]))
                else:
                    c = int(rng.choice([0, shape[1] - w]))
            full = np.zeros(shape, dtype=bool)
            full[r:r + h, c:c + w] = m
            if np.any(full & occupied) or (occupied.sum() + full.sum()) > .38 * occupied.size:
                continue
            occupied |= full
            cores.append(full)
            axes.append(2 * c + w - 1)
            break
        else:
            return None
    return cores, axes


def _add_noise(rng, core, axis, occupied):
    h, w = core.shape
    mask = core.copy()
    scores = _scores(mask)
    support = int(core.sum())
    rows, cols = np.nonzero(core)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    distance = _distance(core)
    amount = max(1, min(10, int(round(support * rng.uniform(.12, .42)))))
    for index in range(amount):
        candidates = []
        for r, c in np.argwhere(~occupied):
            r, c = int(r), int(c)
            reflected = axis - c
            if 2 * c == axis or (0 <= reflected < w and mask[r, reflected]):
                continue
            candidates.append((r, c))
        rng.shuffle(candidates)
        # Outside-column noise exposes why a noisy bounding-box midpoint is
        # not an axis estimator. Later noise mixes protrusions, holes and dots.
        if index == 0:
            candidates.sort(key=lambda p: (not (p[1] < c0 or p[1] > c1),
                                           not (distance[p] <= 2)))
        else:
            local = bool(rng.random() < .8)
            if local:
                candidates.sort(key=lambda p: not (distance[p] <= 2))
        for r, c in candidates[:160]:
            updated = scores.copy()
            updated[2 * c] += 1
            for old_c in np.flatnonzero(mask[r]):
                updated[c + int(old_c)] += 2
            if updated[axis] != support or np.count_nonzero(updated >= support) != 1:
                continue
            mask[r, c] = occupied[r, c] = True
            scores = updated
            break
        else:
            break
    return mask


def _features(pair):
    a, body, bg, separator, cycle, colors = _decode(pair["input"])
    out = _grid(pair["output"])
    assert a.shape == out.shape and np.array_equal(a[:2], out[:2])
    raw = [body == c for c in colors]
    supported = [_support(m) for m in raw]
    masks, axes = [x[0] for x in supported], [x[1] for x in supported]
    predicted, distance, tied, owner = _paint(masks, colors, cycle, bg)
    assert np.array_equal(out[2:], predicted), "incorrect wave mosaic"
    noise = np.logical_or.reduce([m & ~k for m, k in zip(raw, masks)])
    no_filter = _paint(raw, colors, cycle, bg)[0]
    no_cycle = _paint(masks, colors, cycle, bg, advance=False)[0]
    l1 = _paint(masks, colors, cycle, bg, manhattan=True)[0]
    rectangle_masks = []
    midpoint_wrong = False
    holes, islands, axis_only, edge = 0, 0, 0, False
    for raw_m, core, axis in zip(raw, masks, axes):
        rr, cc = np.nonzero(core)
        rectangle = np.zeros_like(core)
        rectangle[rr.min():rr.max() + 1, cc.min():cc.max() + 1] = True
        rectangle_masks.append(rectangle)
        raw_c = np.nonzero(raw_m)[1]
        midpoint_wrong |= int(raw_c.min() + raw_c.max()) != axis
        holes += int(_has_hole(core))
        axis_only += int(any(row.sum() == 1 for row in core))
        islands += int(_component_count(core) > 1)
        edge |= bool(core[0].any() or core[-1].any() or core[:, 0].any() or core[:, -1].any())
        assert np.all(predicted[core] == body[core]), "retained seeds must keep their phase"
    rectangle_out = _paint(rectangle_masks, colors, cycle, bg)[0]
    return {"shape": tuple(a.shape), "background": bg, "separator": separator,
            "cycle_length": len(cycle), "count": len(colors), "axes": axes,
            "parities": {axis % 2 for axis in axes}, "noise": int(noise.sum()),
            "nonlocal_b": int(np.sum((no_filter != predicted) & ~noise)),
            "a_witness": int(np.sum(no_cycle != predicted)),
            "metric_witness": int(np.sum(l1 != predicted)),
            "box_witness": int(np.sum(rectangle_out != predicted)),
            "tie_cells": int(tied.sum()), "wrap_cells": int(np.sum((distance >= len(cycle)) & ~tied)),
            "midpoint_wrong": midpoint_wrong, "holes": holes,
            "islands": islands, "axis_only": axis_only, "edge": edge,
            "masks": masks, "raw": raw, "distance": distance, "tied": tied,
            "owner": owner, "colors": colors, "cycle": cycle}


def _component_count(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    count = 0
    while remaining:
        todo = [remaining.pop()]
        count += 1
        while todo:
            r, c = todo.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
    return count


def _has_hole(mask):
    """An empty four-connected component enclosed inside the retained ink."""
    rr, cc = np.nonzero(mask)
    local = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    empty = set(map(tuple, np.argwhere(~local).tolist()))
    boundary = {p for p in empty if p[0] in (0, local.shape[0] - 1)
                or p[1] in (0, local.shape[1] - 1)}
    frontier = list(boundary)
    exterior = set(boundary)
    while frontier:
        r, c = frontier.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in empty and p not in exterior:
                exterior.add(p)
                frontier.append(p)
    return bool(empty - exterior)


def _informative(info):
    return (info["noise"] > 0 and info["nonlocal_b"] >= 3
            and info["a_witness"] >= 5 and info["metric_witness"] >= 3
            and info["tie_cells"] > 0 and info["wrap_cells"] > 0)


def _pair(rng, shape, count, length, palette, first):
    bg, separator, *all_colors = palette
    cycle = tuple(all_colors[:length])
    colors = tuple(int(c) for c in rng.choice(cycle, count, replace=False))
    for attempt in range(60):
        placed = _place(rng, shape, count, first)
        if placed is None:
            continue
        cores, axes = placed
        occupied = np.logical_or.reduce(cores)
        raw = [_add_noise(rng, core, axis, occupied) for core, axis in zip(cores, axes)]
        inp = np.full((shape[0] + 2, shape[1]), bg, dtype=np.int8)
        inp[0, :length] = cycle
        inp[1] = separator
        for color, mask in zip(colors, raw):
            inp[2:][mask] = color
        out = inp.copy()
        out[2:] = _paint(cores, colors, cycle, bg)[0]
        pair = {"input": inp, "output": out}
        info = _features(pair)
        if count == 1:
            okay = info["noise"] > 0 and info["nonlocal_b"] >= 3 and info["a_witness"] >= 5
        else:
            okay = _informative(info)
        if first:
            okay &= info["parities"] == {0, 1} and info["box_witness"] > 0 and info["midpoint_wrong"]
        okay &= np.sum(inp[2:] == bg) > inp[2:].size / 2
        if okay:
            return pair
    raise RuntimeError("bounded construction could not supply informative waves")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _checked_count("num_train", num_train, 2, 10)
    ne = _checked_count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    shapes = set()
    # Rotate all ten literal colors through observable roles. This is solely
    # balanced sampling, never a hidden inference convention.
    role_order = list(map(int, rng.permutation(10)))
    for i in range(nt + ne):
        first = i == 0
        count = 2 if first else 3 if i == 1 else int(rng.integers(1, 4)) if i < nt else int(rng.integers(4, 6))
        length = int(rng.integers(max(2, count), 7))
        if first:
            length = int(rng.integers(2, 4))
        possible = [(h, w) for h in range(18 if first else 12, 29)
                    for w in range(18 if first else 12, 31) if (h, w) not in shapes]
        shape = possible[int(rng.integers(len(possible)))]
        shapes.add(shape)
        bg = role_order[i % 10]
        sep = role_order[(i + 1) % 10]
        other = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, sep)])))
        pair = _pair(rng, shape, count, length, [bg, sep, *other], first)
        puzzle["train" if i < nt else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _features(pair)
            assert np.array_equal(solve(inp), out)
            assert np.sum(inp[2:] == info["background"]) > inp[2:].size / 2
            assert 1 <= info["count"] <= 5
            assert info["noise"] > 0 and info["nonlocal_b"] >= 3 and info["a_witness"] >= 5
            if info["count"] > 1:
                assert _informative(info)
            assert info["count"] <= 3 if split == "train" else 4 <= info["count"] <= 5
            analyses[split].append(info)
    first = analyses["train"][0]
    assert first["count"] == 2 and analyses["train"][1]["count"] == 3
    assert first["parities"] == {0, 1} and first["box_witness"] > 0 and first["midpoint_wrong"]
    assert len({a["background"] for a in analyses["train"]}) >= 2
    assert min(a["count"] for a in analyses["test"]) > max(a["count"] for a in analyses["train"])


def _bfs_distance(mask):
    """Check the direct metric computation by simultaneous eight-neighbor growth."""
    distance = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque(map(tuple, np.argwhere(mask).tolist()))
    distance[mask] = 0
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and distance[nr, nc] < 0:
                    distance[nr, nc] = distance[r, c] + 1
                    queue.append((nr, nc))
    return distance


def self_test():
    """42 streaming puzzles, semantic counterfactuals, diversity and mutations."""
    shapes, counts, lengths, backgrounds, separators, parities, signatures = (set() for _ in range(7))
    metrics = {"pairs": 0, "holes": 0, "islands": 0, "axis_only": 0, "edge": 0,
               "nonlocal_b_cells": 0, "tie_cells": 0, "wrap_cells": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _features(pair)
                shapes.add(info["shape"])
                counts.add(info["count"])
                lengths.add(info["cycle_length"])
                backgrounds.add(info["background"])
                separators.add(info["separator"])
                parities.update(info["parities"])
                digest.update(pair["input"].tobytes())
                metrics["pairs"] += 1
                for name in ("holes", "islands", "axis_only", "edge", "tie_cells", "wrap_cells"):
                    metrics[name] += int(info[name])
                metrics["nonlocal_b_cells"] += info["nonlocal_b"]
                body = pair["input"][2:]
                for color, raw, core, axis in zip(info["colors"], info["raw"], info["masks"], info["axes"]):
                    cells = set(map(tuple, np.argwhere(raw).tolist()))
                    scores = [sum((r, a - c) in cells for r, c in cells)
                              for a in range(2 * body.shape[1] - 1)]
                    assert scores.count(max(scores)) == 1 and scores.index(max(scores)) == axis
                    supported = {(r, c) for r, c in cells if (r, axis - c) in cells}
                    assert supported == set(map(tuple, np.argwhere(core).tolist()))
                    assert np.array_equal(_distance(core), _bfs_distance(core))
        signatures.add(digest.digest())
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                       for k in ("input", "output"))
    assert len(signatures) == 42 and len(shapes) >= 70
    assert counts == {1, 2, 3, 4, 5} and lengths == {2, 3, 4, 5, 6}
    assert backgrounds == separators == set(range(10)) and parities == {0, 1}
    assert all(metrics[k] >= 10 for k in ("holes", "islands", "axis_only", "edge"))
    # Deliberate support tie: do not invent a leftmost/integral tie-break.
    tie = np.zeros((3, 8), dtype=bool)
    tie[0, 1] = tie[2, 6] = True
    try:
        _support(tie)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous source accepted")
    bad = generate_puzzle(81, 2, 1)
    info = _features(bad["test"][0])
    r, c = np.argwhere(info["tied"])[0]
    bad["test"][0]["output"][r + 2, c] = info["cycle"][0]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("filled collision ridge accepted")
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal explicit counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "source_counts": sorted(counts), "cycle_lengths": sorted(lengths),
            "backgrounds": sorted(backgrounds), "separators": sorted(separators), **metrics}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
