"""Keyed strip canvas followed by the union of all empty 3x3 windows.

All color roles are visible. The origin doubles as a highlight swatch; the
fast arm is parallel to the common long axis of the full two-thick strips.
No parent modules, examples, files, random state, or training labels are read
by solve. Generation and self_test write no artifacts.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask):
    """Four-connected pixel sets, also used for nonrectangular blank regions."""
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, points = [first], [first]
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    points.append(p)
        result.append(points)
    return result


def _bbox(points):
    return (min(p[0] for p in points), max(p[0] for p in points)+1,
            min(p[1] for p in points), max(p[1] for p in points)+1)


def _parse(input_grid):
    """Recover key, axis, lane order, and properly rotated strips from input."""
    a = _grid(input_grid)
    h, w = a.shape
    candidates = []
    if min(h, w) < 5:
        raise ValueError("A separated strip scene needs an L key and clear margins")
    for bottom in (False, True):
        for right in (False, True):
            aligned = a[::-1] if bottom else a
            aligned = aligned[:, ::-1] if right else aligned
            origin, horizontal, vertical, bg = map(int, (
                aligned[0, 0], aligned[0, 1], aligned[1, 0], aligned[-1, -1]))
            if len({origin, horizontal, vertical, bg}) != 4:
                continue
            if (np.all(aligned[0, 1:] == horizontal)
                    and np.all(aligned[1:, 0] == vertical)
                    and np.all(aligned[-1, 1:] == bg)
                    and np.all(aligned[1:, -1] == bg)):
                candidates.append((bottom, right, origin, horizontal, vertical, bg))
    if len(candidates) != 1:
        raise ValueError("The outer border must determine exactly one L key")
    bottom, right, origin, horizontal, vertical, bg = candidates[0]
    mask = a != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes = []
    for points in _components(mask):
        r0, r1, c0, c1 = _bbox(points)
        if len(points) != (r1-r0)*(c1-c0):
            raise ValueError("Each payload component must be a full rectangle")
        if not (1+int(not bottom) <= r0 < r1 <= h-1-int(bottom)
                and 1+int(not right) <= c0 < c1 <= w-1-int(right)):
            raise ValueError("Payload must have a blank margin from every border")
        if np.any(a[r0:r1, c0:c1] == origin):
            raise ValueError("The origin/highlight color is reserved for the key")
        boxes.append((int(r0), int(r1), int(c0), int(c1)))
    if not 1 <= len(boxes) <= 15:
        raise ValueError("Expected 1..15 strips")
    horizontal_ok = all(b[1]-b[0] == 2 and b[3]-b[2] >= 2 for b in boxes)
    vertical_ok = all(b[3]-b[2] == 2 and b[1]-b[0] >= 2 for b in boxes)
    if horizontal_ok == vertical_ok:
        raise ValueError("A common long axis, witnessed by a nonsquare strip, is required")
    axis = int(vertical_ok)
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    lanes = {}
    for b in boxes:
        lanes.setdefault(b[2] if axis else b[0], []).append(b)
    positions = sorted(lanes)
    if any(y-x < 3 for x, y in zip(positions, positions[1:])):
        raise ValueError("Lane bands must be aligned and separated")
    ordered, blocks, counts = [], [], []
    for v in sorted(lanes, reverse=right if axis else bottom):
        group = sorted(lanes[v], key=lambda b: b[0] if axis else b[2],
                       reverse=bottom if axis else right)
        intervals = sorted((b[0], b[1]) if axis else (b[2], b[3]) for b in group)
        if any(y[0]-x[1] < 1 for x, y in zip(intervals, intervals[1:])):
            raise ValueError("Strips within a lane need a blank separator")
        counts.append(len(group))
        for r0, r1, c0, c1 in group:
            blocks.append(np.rot90(a[r0:r1, c0:c1], rotation).copy())
            ordered.append((r0, r1, c0, c1))
    lengths = [b.shape[1] for b in blocks]
    if len({n % 2 for n in lengths}) != 1:
        raise ValueError("All strip lengths must have the same parity for exact centering")
    return {"blocks": blocks, "boxes": ordered, "lengths": lengths,
            "counts": counts, "axis": axis, "bottom": bottom, "right": right,
            "background": bg, "highlight": origin,
            "fast": vertical if axis else horizontal,
            "slow": horizontal if axis else vertical}


def _stack(blocks, background):
    width = max(b.shape[1] for b in blocks)
    canvas = np.full((2*len(blocks), width), background, dtype=np.int8)
    for i, block in enumerate(blocks):
        left = (width-block.shape[1])//2
        canvas[2*i:2*i+2, left:left+block.shape[1]] = block
    return canvas


def _window_union(canvas, background, size=3):
    """Detect on the unchanged canvas, then take the union simultaneously."""
    h, w = canvas.shape
    mask = np.zeros((h, w), dtype=bool)
    if min(h, w) < size:
        return mask, []
    blank = canvas == background
    counts = np.zeros((h-size+1, w-size+1), dtype=np.int16)
    for dr in range(size):
        for dc in range(size):
            counts += blank[dr:dr+h-size+1, dc:dc+w-size+1]
    starts = [tuple(map(int, p)) for p in np.argwhere(counts == size*size)]
    for r, c in starts:
        mask[r:r+size, c:c+size] = True
    return mask, starts


def _finish(blocks, background, highlight):
    canvas = _stack(blocks, background)
    mask, _ = _window_union(canvas, background)
    result = canvas.copy()
    result[mask] = highlight
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _parse(input_grid)
    return _finish(d["blocks"], d["background"], d["highlight"])


def _gaps(rng, occupied, count, extent=None):
    minimum = occupied+count+1
    if extent is None:
        room = 29-minimum
        extra = int(rng.integers((min(room, 5) if rng.random() < .6 else room)+1))
        extent = minimum+extra
    extra = extent-minimum
    weights = rng.dirichlet(np.full(count+1, float(rng.choice([.4, 1., 3.]))))
    return (1+rng.multinomial(extra, weights)).tolist(), extent


def _partition(rng, lengths, witness):
    """Sample contiguous lane groups by exact finite dynamic programming."""
    n = len(lengths)
    fits = {}
    for start in range(n):
        fits[start] = [end for end in range(start+1, n+1)
                       if sum(lengths[start:end])+(end-start)+1 <= 29]
    ways = {(n, 0): 1}
    for start in range(n-1, -1, -1):
        for lanes in range(1, min(9, n-start)+1):
            ways[start, lanes] = sum(ways.get((end, lanes-1), 0) for end in fits[start])
    choices = [k for k in range(2 if witness else 1, min(9, n)+1)
               if ways.get((0, k), 0) and (not witness or k < n)]
    if not choices:
        return None
    weights = np.array([3. if 2 <= k <= 5 else 1. for k in choices])
    k = int(rng.choice(choices, p=weights/weights.sum()))
    groups, start = [], 0
    while k:
        ends = [end for end in fits[start] if ways.get((end, k-1), 0)]
        weights = np.array([ways[end, k-1] for end in ends], dtype=float)
        end = int(rng.choice(ends, p=weights/weights.sum()))
        groups.append(lengths[start:end])
        start, k = end, k-1
    return groups


def _lengths(rng, kind):
    base = int(rng.choice([2, 3]))
    pmax = int(rng.integers(4 if kind == "step" else 3, 13))
    width = base+2*pmax
    if kind in ("exact", "step"):
        # Coverage motifs are surrounded and reversed in variable ways; the
        # ordinary branch below samples unrestricted padding sequences.
        run = [3, 3] if kind == "exact" else [int(rng.integers(4, pmax+1))]*2+[3]
        thin = int(rng.integers(1, 3))
        run = [thin]+run if rng.random() < .5 else run+[thin]
        isolated = [0, int(rng.integers(3, pmax+1)), 0]
        padding = run+isolated if rng.random() < .5 else isolated+run
        if rng.random() < .5:
            padding.reverse()
        # Extra end blocks do not destroy the protected run or isolated strip.
        extras = list(map(int, rng.integers(0, pmax+1, size=int(rng.integers(5)))))
        padding = extras+padding if rng.random() < .5 else padding+extras
    elif kind == "negative":
        n = int(rng.integers(3, 16))
        if rng.random() < .5:
            padding = list(map(int, rng.integers(0, 3, size=n)))
        else:
            padding = [int(rng.integers(3, pmax+1)) if i % 2 else 0 for i in range(n)]
        padding[int(rng.integers(n))] = 0
    else:
        n = int(rng.integers(3, 16))
        padding = list(map(int, rng.integers(0, pmax+1, size=n)))
        padding[int(rng.integers(n))] = 0
        if rng.random() < .7:
            i = int(rng.integers(n-1))
            padding[i:i+2] = list(map(int, rng.integers(3, pmax+1, size=2)))
            # Retain a real longest strip outside the short run.
            candidates = [j for j in range(n) if j not in (i, i+1)]
            padding[int(rng.choice(candidates))] = 0
    return [width-2*p for p in padding]


def _asymmetric(block):
    return all(not np.array_equal(block, variant) for variant in
               (block[::-1], block[:, ::-1], block[::-1, ::-1]))


def _payloads(rng, lengths, pool, witness):
    palette = rng.choice(pool, size=int(rng.integers(3 if witness else 1, len(pool)+1)),
                         replace=False)
    all_solid = not witness and rng.random() < .3
    blocks = []
    for length in lengths:
        previous = [b for b in blocks if b.shape[1] == length]
        if previous and rng.random() < .2:
            block = previous[int(rng.integers(len(previous)))].copy()
        else:
            mode = 0 if all_solid else int(rng.integers(4))
            if mode == 0:
                block = np.full((2, length), rng.choice(palette), dtype=np.int8)
            elif mode == 1:
                block = np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1)
            elif mode == 2:
                period = int(rng.integers(1, min(length, 4)+1))
                block = np.tile(rng.choice(palette, size=(2, period)), (1, length))[:, :length]
            else:
                block = rng.choice(palette, size=(2, length))
        blocks.append(block.astype(np.int8))
    if witness:
        i = int(rng.integers(len(blocks)))
        # Three distinct corner values force row, column and half-turn evidence.
        block = rng.choice(palette, size=blocks[i].shape).astype(np.int8)
        block[0, 0], block[0, -1], block[1, 0] = palette[:3]
        blocks[i] = block
        assert _asymmetric(block)
    return blocks


def _paint(rng, groups, blocks, configuration, roles):
    axis, bottom, right = configuration
    bg, highlight, fast, slow = roles
    minimum = max(sum(g)+len(g)+1 for g in groups)
    spare = 29-minimum
    extra = int(rng.integers((min(spare, 5) if rng.random() < .6 else spare)+1))
    fast_extent = minimum+extra
    slow_gaps, slow_extent = _gaps(rng, 2*len(groups), len(groups))
    shape = ((fast_extent+1, slow_extent+1) if axis else (slow_extent+1, fast_extent+1))
    h, w = shape
    a = np.full(shape, bg, dtype=np.int8)
    kr, kc = h-1 if bottom else 0, w-1 if right else 0
    a[kr, :] = slow if axis else fast
    a[:, kc] = fast if axis else slow
    a[kr, kc] = highlight
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    index, v = 0, slow_gaps[0]
    for lane, group in enumerate(groups):
        gaps, _ = _gaps(rng, sum(group), len(group), fast_extent)
        u = gaps[0]
        for j, length in enumerate(group):
            if axis:
                r0 = h-1-u-length if bottom else 1+u
                c0 = w-1-v-2 if right else 1+v
            else:
                r0 = h-1-v-2 if bottom else 1+v
                c0 = w-1-u-length if right else 1+u
            physical = np.rot90(blocks[index], -rotation)
            a[r0:r0+physical.shape[0], c0:c0+physical.shape[1]] = physical
            u += length+gaps[j+1]
            index += 1
        v += 2+slow_gaps[lane+1]
    return {"input": a, "output": _finish(blocks, bg, highlight)}


def _alternative_orders(d):
    groups, start = [], 0
    for count in d["counts"]:
        groups.append(list(range(start, start+count)))
        start += count
    fast_reverse = [i for group in groups for i in group[::-1]]
    slow_reverse = [i for group in groups[::-1] for i in group]
    coord = 0 if d["axis"] else 2
    reverse = d["bottom"] if d["axis"] else d["right"]
    fast_major = sorted(range(len(d["blocks"])), key=lambda i: d["boxes"][i][coord], reverse=reverse)
    physical = sorted(range(len(d["blocks"])), key=lambda i: d["boxes"][i][:3:2])
    return {"reverse_fast": fast_reverse, "reverse_lanes": slow_reverse,
            "fast_major": fast_major, "physical_order": physical}


def _witness_layout(d):
    bg, hl = d["background"], d["highlight"]
    expected = _finish(d["blocks"], bg, hl)
    mask, _ = _window_union(_stack(d["blocks"], bg), bg)
    changed_selection = False
    for name, order in _alternative_orders(d).items():
        if name == "physical_order":
            continue
        blocks = [d["blocks"][i] for i in order]
        if np.array_equal(expected, _finish(blocks, bg, hl)):
            return False
        other, _ = _window_union(_stack(blocks, bg), bg)
        changed_selection |= not np.array_equal(mask, other)
    return changed_selection


def _make_pair(rng, kind, configuration, roles):
    witness = kind in ("exact", "step")
    pool = [c for c in range(10) if c not in roles[:2]]
    for _ in range(96):
        lengths = _lengths(rng, kind)
        groups = _partition(rng, lengths, witness)
        if groups is None:
            continue
        blocks = _payloads(rng, lengths, pool, witness)
        pair = _paint(rng, groups, blocks, configuration, roles)
        d = _parse(pair["input"])
        if witness and not _witness_layout(d):
            continue
        return pair
    raise RuntimeError("Bounded construction could not find an informative packed scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, size=4, replace=False)))
    configurations = [(axis, bottom, right) for axis in (0, 1)
                      for bottom in (False, True) for right in (False, True)]
    first = configurations[int(rng.integers(8))]
    second = (1-first[0], not first[1], not first[2])
    held = [c for c in configurations if c not in (first, second)]
    query = held[int(rng.integers(len(held)))]
    available = [c for c in configurations if c != query]
    train_configs = [first, second]+[available[int(rng.integers(7))] for _ in range(num_train-2)]
    test_configs = [query]+[configurations[int(rng.integers(8))] for _ in range(num_test-1)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, configs in (("train", train_configs), ("test", test_configs)):
        for i, config in enumerate(configs):
            if split == "train" and i < 2:
                kind = ("exact", "step")[i]
            else:
                kind = "negative" if rng.random() < .16 else "ordinary"
            for _ in range(16):
                pair = _make_pair(rng, kind, config, roles)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded generation exhausted distinct input attempts")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _blank_evidence(canvas, mask, background):
    partial_component = nonrectangular_selection = False
    for points in _components(canvas == background):
        selected = [p for p in points if mask[p]]
        if selected:
            partial_component |= len(selected) < len(points)
            r0, r1, c0, c1 = _bbox(selected)
            nonrectangular_selection |= len(selected) < (r1-r0)*(c1-c0)
    return partial_component, nonrectangular_selection


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, training, query_configs = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(a)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            d = _parse(a)
            bg, hl = d["background"], d["highlight"]
            roles.add((bg, hl, d["fast"], d["slow"]))
            canvas = _stack(d["blocks"], bg)
            mask, starts = _window_union(canvas, bg)
            assert out.shape == (2*len(d["blocks"]), max(d["lengths"]))
            assert np.array_equal(out != canvas, mask), "Changed footprint is not the window union"
            assert np.all(out[mask] == hl) and np.all(canvas[mask] == bg)
            assert np.array_equal(out[~mask], canvas[~mask]), "Non-target cells changed"
            # A second, literal local check avoids relying on a mask count alone.
            for r in range(canvas.shape[0]-2):
                for c in range(canvas.shape[1]-2):
                    empty = bool(np.all(canvas[r:r+3, c:c+3] == bg))
                    assert empty == ((r, c) in starts)
            assert np.array_equal(solve(a), out)
            if split == "train":
                training.append((d, canvas, mask, starts))
            else:
                query_configs.append((d["axis"], d["bottom"], d["right"]))
    assert len(roles) == 1, "Palette roles must be consistent within a puzzle"
    assert {d["axis"] for d, _, _, _ in training} == {0, 1}
    assert {d["bottom"] for d, _, _, _ in training} == {False, True}
    assert {d["right"] for d, _, _, _ in training} == {False, True}
    train_configs = {(d["axis"], d["bottom"], d["right"]) for d, _, _, _ in training}
    assert query_configs[0] not in train_configs, "First query must combine the orientation factors anew"
    for name in ("reverse_fast", "reverse_lanes", "fast_major", "physical_order"):
        assert any(not np.array_equal(
            _finish([d["blocks"][i] for i in _alternative_orders(d)[name]],
                    d["background"], d["highlight"]),
            _finish(d["blocks"], d["background"], d["highlight"]))
                   for d, _, _, _ in training), f"Unresolved order shortcut: {name}"
    assert any(_witness_layout(d) for d, _, _, _ in training), "Key must affect blank selection"
    assert all(any(d["axis"] == axis and any(_asymmetric(b) for b in d["blocks"])
                   for d, _, _, _ in training) for axis in (0, 1))
    assert any(mask.any() and int(mask.sum()) < 9*len(starts)
               for _, _, mask, starts in training), "Need overlapping windows"
    assert any(_blank_evidence(c, m, d["background"])[0] for d, c, m, _ in training)
    assert any(_blank_evidence(c, m, d["background"])[1] for d, c, m, _ in training)
    for size in (2, 4):
        assert any(not np.array_equal(mask, _window_union(canvas, d["background"], size)[0])
                   for d, canvas, mask, _ in training), f"Unresolved window size {size}"


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds, retain only hashes and small feature sets, never grids."""
    signatures, input_shapes, output_shapes, lengths, counts = set(), set(), set(), set(), set()
    configs, palettes, lane_populations, masks, structures = set(), set(), set(), set(), set()
    negative = duplicate = odd = wide = step = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                d = _parse(a)
                input_shapes.add(a.shape)
                output_shapes.add(out.shape)
                lengths.update(d["lengths"])
                counts.add(len(d["blocks"]))
                configs.add((d["axis"], d["bottom"], d["right"]))
                palettes.add((d["background"], d["highlight"], d["fast"], d["slow"]))
                lane_populations.add(tuple(d["counts"]))
                canvas = _stack(d["blocks"], d["background"])
                mask, _ = _window_union(canvas, d["background"])
                masks.add(hashlib.sha256(bytes(mask.shape)+mask.tobytes()).hexdigest())
                normalized, next_role = [], {}
                for color in a.flat:
                    normalized.append(next_role.setdefault(int(color), len(next_role)))
                structures.add(hashlib.sha256(bytes(a.shape)+bytes(normalized)).hexdigest())
                negative += not mask.any()
                duplicate += len({_signature(b) for b in d["blocks"]}) < len(d["blocks"])
                odd += max(d["lengths"]) % 2
                wide += max(d["lengths"]) >= 20
                step += _blank_evidence(canvas, mask, d["background"])[1]
                # Whole-scene proper rotations preserve the normalized answer.
                if split == "test":
                    for k in range(4):
                        rotated = np.rot90(a, k).copy()
                        before = rotated.copy()
                        assert np.array_equal(solve(rotated), out)
                        assert np.array_equal(rotated, before), "solve mutated its input"
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(input_shapes) >= 60 and len(output_shapes) >= 40
    assert len(configs) == 8 and len(palettes) >= 30
    assert len(counts) >= 10 and max(counts) == 15
    assert len(lengths) >= 20 and max(lengths) >= 26
    assert len(lane_populations) >= 60 and len(masks) >= 90
    assert negative and duplicate and odd and wide and step
    # Exhaust all 30 legal explicit count combinations, retaining one at a time.
    count_calls = 0
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            puzzle = generate_puzzle(7000+count_calls, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
            count_calls += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Exact spatial fixtures: threshold, union overlap, fringe, and step.
    for padding in ([3, 3, 1, 0, 3, 0], [4, 4, 3, 1, 0, 4, 0]):
        blocks = [np.full((2, 10-2*p), 5, dtype=np.int8) for p in padding]
        canvas = _stack(blocks, 0)
        mask, starts = _window_union(canvas, 0)
        assert len(starts) > 1 and mask.sum() < 9*len(starts)
        expected = np.zeros_like(mask)
        end = 4 if len(padding) == 6 else 6
        for r in range(end):
            p = padding[r//2]
            expected[r, :p] = True
            expected[r, 10-p:] = True
        assert np.array_equal(mask, expected)
        assert _blank_evidence(canvas, mask, 0)[0]
    # Solver coverage beyond sampled motifs: one/two strips, minimal odd width,
    # maximum physical width, and the maximum two-row output population.
    fixture_rng = np.random.default_rng(100)
    for group_lengths in ([[3]], [[2, 8]], [[27]], [[8], [2]*7, [2]*7]):
        blocks = []
        for group in group_lengths:
            for length in group:
                block = np.full((2, length), 5, dtype=np.int8)
                block[0, 0], block[0, -1], block[1, 0] = 3, 6, 7
                blocks.append(block)
        expected = _finish(blocks, 0, 4)
        for axis in (0, 1):
            for bottom in (False, True):
                for right in (False, True):
                    pair = _paint(fixture_rng, group_lengths, blocks, (axis, bottom, right), (0, 4, 1, 2))
                    assert np.array_equal(solve(pair["input"]), expected)
    # Highlight follows the visible origin under an arbitrary palette bijection.
    p = generate_puzzle(42, 2, 1)
    permutation = np.array([6, 2, 9, 0, 8, 4, 1, 7, 5, 3], dtype=np.int8)
    for pair in p["train"]+p["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    bad = generate_puzzle(43, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Wrong output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "configurations": len(configs), "palettes": len(palettes),
            "strip_counts": sorted(counts), "length_values": len(lengths),
            "lane_populations": len(lane_populations), "window_footprints": len(masks),
            "color_normalized_inputs": len(structures), "negative_pairs": int(negative),
            "duplicate_payload_pairs": int(duplicate), "odd_pairs": int(odd),
            "wide_pairs": int(wide), "stepped_union_pairs": int(step),
            "legal_count_calls": count_calls, "independent_semantic_check": False}
