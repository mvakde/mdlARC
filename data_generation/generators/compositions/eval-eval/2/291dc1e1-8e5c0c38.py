"""Mirror loom: reflection in the ordinal coordinates of keyed strip fragments.

Only the first dimension of a (fragment, thickness-row, centered-column) tensor
is reflected. This is deliberately not ordinary reflection of the output pixels.
Generation constructs symmetric coordinate orbits; prediction uses Boolean masks.
No files, seeds, examples, or generator state are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid in colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(g):
    """The doubled origin tick identifies the fast arm without a color legend."""
    candidates = []
    if min(g.shape) < 5:
        raise ValueError("no room for the coordinate key")
    for bottom in (False, True):
        for right in (False, True):
            a = g[::-1] if bottom else g
            a = a[:, ::-1] if right else a
            origin, bg = int(a[0, 0]), int(a[-1, -1])
            if not (np.all(a[-1, 1:] == bg) and np.all(a[1:, -1] == bg)):
                continue
            for axis in (0, 1):
                fast_arm = a[1:, 0] if axis else a[0, 1:]
                slow_arm = a[0, 1:] if axis else a[1:, 0]
                fast, carrier = int(fast_arm[1]), int(slow_arm[0])
                if len({origin, bg, fast, carrier}) != 4:
                    continue
                if (fast_arm[0] == origin and np.all(fast_arm[1:] == fast)
                        and np.all(slow_arm == carrier)):
                    candidates.append((axis, bottom, right, bg, carrier, origin, fast))
    if len(candidates) != 1:
        raise ValueError("the input must determine exactly one ticked L-key")
    return candidates[0]


def _rectangles(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    boxes = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, cells = [p], [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    todo.append(q)
        r0, c0 = map(min, zip(*cells))
        r1, c1 = (max(x)+1 for x in zip(*cells))
        if len(cells) != (r1-r0)*(c1-c0):
            raise ValueError("every fragment must be a fully occupied rectangle")
        boxes.append((r0, r1, c0, c1))
    return boxes


def _assemble(blocks, bg, carrier, centered=True):
    width = max(b.shape[1] for b in blocks)
    tensor = np.full((len(blocks), 2, width), bg, dtype=np.int8)
    for i, b in enumerate(blocks):
        start = (width-b.shape[1])//2 if centered else 0
        tensor[i, :, start:start+b.shape[1]] = np.where(b == carrier, bg, b)
    return tensor


def _read(input_grid):
    g = _grid(input_grid)
    config = _key(g)
    axis, bottom, right, bg, carrier, _, _ = config
    mask = g != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes = _rectangles(mask)
    if not 1 <= len(boxes) <= 15:
        raise ValueError("there must be 1..15 fragment slots")
    lanes = {}
    for box in boxes:
        r0, r1, c0, c1 = box
        if not (0 < r0 < r1 < g.shape[0] and 0 < c0 < c1 < g.shape[1]):
            raise ValueError("fragments require a background margin")
        if (c1-c0 if axis else r1-r0) != 2:
            raise ValueError("fragments must be two cells thick")
        if (r1-r0 if axis else c1-c0) < 2:
            raise ValueError("fragment length must be at least two")
        lanes.setdefault(c0 if axis else r0, []).append(box)
    if any(b-a < 3 for a, b in zip(sorted(lanes), sorted(lanes)[1:])):
        raise ValueError("lane bands must be separated and unstaggered")
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    blocks, positions, counts = [], [], []
    for v in sorted(lanes, reverse=right if axis else bottom):
        group = sorted(lanes[v], key=lambda b: b[0] if axis else b[2],
                       reverse=bottom if axis else right)
        counts.append(len(group))
        for box in group:
            r0, r1, c0, c1 = box
            blocks.append(np.rot90(g[r0:r1, c0:c1], rotation).copy())
            # Coordinates measured from the key, independent of physical axes.
            u = (g.shape[0]-r1 if bottom else r0) if axis else (g.shape[1]-c1 if right else c0)
            positions.append((u, len(counts)-1))
    lengths = [b.shape[1] for b in blocks]
    if len({n % 2 for n in lengths}) != 1:
        raise ValueError("fragment lengths must permit exact centering")
    tensor = _assemble(blocks, bg, carrier)
    if not np.any(tensor != bg):
        raise ValueError("at least one payload color is required")
    return {"config": config, "background": bg, "carrier": carrier,
            "blocks": blocks, "lengths": lengths, "counts": counts,
            "positions": positions, "tensor": tensor}


def _supported(mask, axis2):
    n = len(mask)
    reflected = axis2 - np.arange(n)
    valid = (reflected >= 0) & (reflected < n)
    keep = np.zeros_like(mask, dtype=bool)
    keep[valid] = mask[valid] & mask[reflected[valid]]
    return keep


def _filter(tensor, bg):
    output = np.full_like(tensor, bg)
    axes = {}
    for color in sorted(set(map(int, tensor.flat)) - {bg}):
        mask = tensor == color
        scores = [int(_supported(mask, a).sum()) for a in range(2*len(tensor)-1)]
        winners = np.flatnonzero(np.array(scores) == max(scores))
        if len(winners) != 1:
            raise ValueError("a payload color has tied maximum-support slot axes")
        axis = int(winners[0])
        output[_supported(mask, axis)] = color
        axes[color] = axis
    return output, axes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the mirror-supported transcript, using only the supplied input."""
    d = _read(input_grid)
    out, _ = _filter(d["tensor"], d["background"])
    return out.reshape(2*len(out), out.shape[-1]).copy()


def _could_explain(raw, expected, bg):
    """Can ANY maximum-axis tie choices in an alternative explain the label?"""
    for color in (set(map(int, raw.flat)) | set(map(int, expected.flat))) - {bg}:
        mask, target = raw == color, expected == color
        supports = [_supported(mask, a) for a in range(2*len(raw)-1)]
        maximum = max(int(s.sum()) for s in supports)
        if not any(int(s.sum()) == maximum and np.array_equal(s, target) for s in supports):
            return False
    return True


def _witnesses(d, cleaned):
    raw, bg = d["tensor"], d["background"]
    groups, start = [], 0
    for count in d["counts"]:
        groups.append(list(range(start, start+count)))
        start += count
    orders = {
        "reverse_fast": [i for group in groups for i in group[::-1]],
        "reverse_lanes": [i for group in groups[::-1] for i in group],
        "fast_major": sorted(range(len(raw)), key=lambda i: d["positions"][i]),
    }
    flags = {name: not _could_explain(raw[order], cleaned, bg) for name, order in orders.items()}
    # Check a stronger interaction than a different output permutation: a
    # competing traversal must actually change which SOURCE marks are kept.
    flags["order_changes_support"] = False
    for order in orders.values():
        try:
            alternative, _ = _filter(raw[order], bg)
        except ValueError:
            continue
        restored = alternative[np.argsort(order)]
        flags["order_changes_support"] |= not np.array_equal(restored, cleaned)
    _, axes = _filter(raw, bg)
    lane_of = [lane for lane, group in enumerate(groups) for _ in group]
    flags["cross_lane_support"] = any(
        0 <= axis-i < len(raw) and lane_of[i] != lane_of[axis-i]
        and np.any(cleaned[i] == color)
        for color, axis in axes.items() for i in range(len(raw)))
    flags["omit_B"] = not np.array_equal(raw, cleaned)
    left = _assemble(d["blocks"], bg, d["carrier"], centered=False)
    flags["left_align"] = not _could_explain(left, cleaned, bg)
    # Ordinary pixel reflection reverses the two thickness rows as well.
    flags["pixel_reflection"] = not _could_explain(
        raw.reshape(2*len(raw), -1), cleaned.reshape(2*len(raw), -1), bg)
    central = np.full_like(raw, bg)
    for color in set(map(int, raw.flat)) - {bg}:
        central[_supported(raw == color, len(raw)-1)] = color
    flags["one_central_axis"] = not np.array_equal(central, cleaned)
    # An empty end slot makes both dropping empty fragments and cropping the
    # surviving foreground directly disagree with an observed demonstration.
    empty = np.all(cleaned == bg, axis=(1, 2))
    flags["preserve_empty_slots"] = bool(empty[0] or empty[-1])
    return flags


def _gaps(rng, occupied, count, extent):
    spare = extent - occupied - count - 1
    if spare < 0:
        raise ValueError("packing budget exhausted")
    return 1 + rng.multinomial(spare, rng.dirichlet(np.ones(count+1)))


def _layout(rng, n, odd, range_mode, witness):
    base = 3 if odd else 2
    capacity = 28//(base+1)
    low = max(2 if witness else 1, (n+capacity-1)//capacity)
    high = min(4 if witness else 7, n//2 if witness else n)
    lane_count = int(rng.integers(low, high+1))
    counts = [1]*lane_count
    preferences = rng.dirichlet(np.ones(lane_count))
    for _ in range(n-lane_count):
        eligible = [i for i, count in enumerate(counts) if count < capacity]
        p = preferences[eligible]
        counts[int(rng.choice(eligible, p=p/p.sum()))] += 1
    ceiling = base + (2 if range_mode == "narrow" else 8 if range_mode == "wide" else 6)
    lengths, groups = [], []
    equal = not witness and range_mode != "wide" and rng.random() < .16
    for count in counts:
        group = [base]*count if equal else list(map(int, rng.choice(np.arange(base, ceiling+1, 2), count)))
        while sum(group)+count > 28:
            i = int(rng.choice([j for j, length in enumerate(group) if length > base]))
            group[i] -= 2
        groups.append(group)
    if witness and len({v for group in groups for v in group}) == 1:
        choices = [(i, j) for i, group in enumerate(groups) for j in range(len(group))
                   if sum(group)+len(group)+2 <= 28]
        i, j = choices[int(rng.integers(len(choices)))]
        groups[i][j] += 2
    if range_mode == "wide" and max(v for group in groups for v in group) < base+6:
        choices = [(i, j) for i, group in enumerate(groups) for j in range(len(group))
                   if sum(group)-group[j]+base+6+len(group) <= 28]
        if choices:
            i, j = choices[int(rng.integers(len(choices)))]
            groups[i][j] = base+6
    min_fast = max(sum(group)+len(group)+1 for group in groups)
    fast_extent = int(rng.integers(min_fast, min(29, min_fast+6)+1))
    slow_extent = int(rng.integers(3*lane_count+1, min(29, 3*lane_count+8)+1))
    slow_gaps = _gaps(rng, 2*lane_count, lane_count, slow_extent)
    placements, v = [], int(slow_gaps[0])
    for lane, group in enumerate(groups):
        gaps = _gaps(rng, sum(group), len(group), fast_extent)
        u = int(gaps[0])
        for j, length in enumerate(group):
            lengths.append(length)
            placements.append((v, u, length))
            u += length + int(gaps[j+1])
        v += 2 + int(slow_gaps[lane+1])
    return lengths, placements, (slow_extent, fast_extent)


def _set_axis(cells, n):
    """Coordinate-orbit construction check, separate from prediction masks."""
    scores = []
    for a in range(2*n-1):
        keep = {p for p in cells if (a-p[0], p[1], p[2]) in cells}
        scores.append((len(keep), a, keep))
    maximum = max(item[0] for item in scores)
    winners = [item for item in scores if item[0] == maximum]
    if len(winners) != 1:
        raise ValueError("ambiguous construction")
    return winners[0][1], winners[0][2]


def _payload(rng, lengths, bg, carrier, witness):
    n, width = len(lengths), max(lengths)
    allowed = set()
    for i, length in enumerate(lengths):
        left = (width-length)//2
        allowed.update((i, r, c) for r in range(2) for c in range(left, left+length))
    color_count = 2 if witness else int(rng.integers(1, 4))
    colors = list(map(int, rng.choice([c for c in range(10) if c not in (bg, carrier)], color_count, replace=False)))
    candidates = [a for a in range(1, 2*n-2)
                  if sum(0 <= a-i < n and i < a-i for i in range(n)) >= 2]
    if witness:
        axes = [int(rng.choice([a for a in candidates if a % 2 == p])) for p in (0, 1)]
        rng.shuffle(axes)
    else:
        axes = list(map(int, rng.choice(candidates, color_count, replace=False)))
    core, occupied, per_color = np.full((n, 2, width), bg, dtype=np.int8), set(), {}
    for color, axis in zip(colors, axes):
        orbits = [(p, (axis-p[0], p[1], p[2])) for p in sorted(allowed-occupied)
                  if p[0] < axis-p[0] and (axis-p[0], p[1], p[2]) in allowed-occupied]
        rng.shuffle(orbits)
        distinct_pairs = sorted({(p[0], q[0]) for p, q in orbits})
        if len(distinct_pairs) < 2:
            raise ValueError("not enough independent reflected slot pairs")
        chosen_pairs = rng.choice(len(distinct_pairs), 2, replace=False)
        selected = set()
        for j in chosen_pairs:
            pair = distinct_pairs[int(j)]
            options = [(p, q) for p, q in orbits if (p[0], q[0]) == pair]
            p, q = options[int(rng.integers(len(options)))]
            selected.update((p, q))
        cap = max(2, min(9, len(allowed)//(color_count*6)))
        target = int(rng.integers(2, cap+1))
        for p, q in orbits:
            if len(selected) >= 2*target:
                break
            selected.update((p, q))
        if axis % 2 == 0:
            singles = sorted(p for p in allowed-occupied if 2*p[0] == axis)
            if singles:
                for j in rng.choice(len(singles), int(rng.integers(1 if witness else 0, min(3, len(singles))+1)), replace=False):
                    selected.add(singles[int(j)])
        occupied.update(selected)
        per_color[color] = selected
        for p in selected:
            core[p] = color
    raw = core.copy()
    for color, axis in zip(colors, axes):
        cells = set(per_color[color])
        amount = int(rng.integers(1 if witness else 0, 6))
        if not witness and rng.random() < .2:
            amount = 0
        choices = sorted(allowed-occupied)
        rng.shuffle(choices)
        added = 0
        for p in choices:
            if added >= amount:
                break
            trial = cells | {p}
            try:
                winner, supported = _set_axis(trial, n)
            except ValueError:
                continue
            if winner == axis and supported == per_color[color]:
                cells.add(p)
                raw[p] = color
                occupied.add(p)
                added += 1
        if witness and added == 0:
            raise ValueError("a demonstration needs unmatched marks")
        winner, supported = _set_axis(cells, n)
        if winner != axis or supported != per_color[color]:
            raise ValueError("corruption changed the intended core")
    blocks = []
    for i, length in enumerate(lengths):
        start = (width-length)//2
        block = raw[i, :, start:start+length].copy()
        block[block == bg] = carrier
        blocks.append(block)
    return blocks, core


def _paint(blocks, placements, shape, config, roles):
    axis, bottom, right = config
    bg, carrier, origin, fast = roles
    interior = np.full(shape, bg, dtype=np.int8)
    reverse_thickness = (bottom == right) if axis else (bottom != right)
    for block, (v, u, length) in zip(blocks, placements):
        interior[v:v+2, u:u+length] = block[::-1] if reverse_thickness else block
    grid = np.full((shape[0]+1, shape[1]+1), bg, dtype=np.int8)
    grid[1:, 1:] = interior
    grid[0, 0], grid[0, 1:], grid[1:, 0] = origin, fast, carrier
    grid[0, 1] = origin
    if axis:
        grid = grid.T
    if bottom:
        grid = grid[::-1]
    if right:
        grid = grid[:, ::-1]
    return grid.copy()


def _make_pair(rng, config, n, range_mode, witness, odd):
    for _ in range(80):
        bg, carrier, origin, fast = map(int, rng.choice(10, 4, replace=False))
        lengths, placements, shape = _layout(rng, n, odd, range_mode, witness)
        try:
            blocks, core = _payload(rng, lengths, bg, carrier, witness)
            inp = _paint(blocks, placements, shape, config, (bg, carrier, origin, fast))
            out = core.reshape(2*n, max(lengths)).copy()
            if witness:
                d = _read(inp)
                if not all(_witnesses(d, core).values()):
                    continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded construction exhausted; no altered rule or hidden fallback")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    configs = [(a, b, c) for a in (0, 1) for b in (False, True) for c in (False, True)]
    first = configs[int(rng.integers(8))]
    second = (1-first[0], not first[1], not first[2])
    shift = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index < 2
            pool = [first, second] if split == "train" and shift == 0 else configs
            if split == "test" and shift == 0:
                pool = [c for c in configs if c not in (first, second)]
            config = (first if index == 0 else second) if witness else pool[int(rng.integers(len(pool)))]
            n = int(rng.integers(6 if witness else 5,
                                 8 if split == "train" and shift == 1 else 13))
            if split == "test" and shift == 1:
                n = int(rng.integers(8, 13))
            range_mode = ("narrow" if split == "train" else "wide") if shift == 2 else "ordinary"
            odd = index % 2 if witness else int(rng.integers(2))
            for _ in range(12):
                pair = _make_pair(rng, config, n, range_mode, witness, odd)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate resampling exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, demonstrated = set(), set()
    flags = {name: False for name in ("reverse_fast", "reverse_lanes", "fast_major",
             "omit_B", "left_align", "pixel_reflection", "one_central_axis",
             "order_changes_support", "cross_lane_support", "preserve_empty_slots")}
    training_configs, train_length_parities, query_parities = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            d = _read(inp)
            core, axes = _filter(d["tensor"], d["background"])
            assert np.array_equal(core.reshape(2*len(core), -1), out), "incorrect mirror transcript"
            raw, bg = d["tensor"], d["background"]
            assert np.all((core == raw) | (core == bg)), "deletion-only payload rule"
            for color, axis in axes.items():
                # Independent coordinate-set confirmation of the input-only masks.
                cells = set(map(tuple, np.argwhere(raw == color).tolist()))
                winner, supported = _set_axis(cells, len(raw))
                assert winner == axis and supported == set(map(tuple, np.argwhere(core == color).tolist()))
                assert np.array_equal(_supported(core == color, axis), core == color)
                if split == "train" and np.any((raw == color) & (core != color)):
                    demonstrated.add(axis % 2)
                if split == "test":
                    query_parities.add(axis % 2)
            if split == "train":
                training_configs.add(d["config"][:3])
                train_length_parities.add(d["lengths"][0] % 2)
                for name, present in _witnesses(d, core).items():
                    flags[name] |= present
    assert query_parities <= demonstrated, "query axis parity lacks a noisy demonstration"
    assert demonstrated == {0, 1} and train_length_parities == {0, 1}
    assert {c[0] for c in training_configs} == {0, 1}
    assert {c[1] for c in training_configs} == {False, True}
    assert {c[2] for c in training_configs} == {False, True}
    assert all(flags.values()), f"training leaves an alternative unresolved: {flags}"


def _puzzle_hash(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds plus small boundary/mutation fixtures; store no corpus."""
    fingerprints, shapes, configurations, counts, widths, palettes = (set() for _ in range(6))
    lane_vectors, axis_values, noise_totals, core_sizes, length_vectors = (set() for _ in range(5))
    empty_slots = reused_key_colors = clean_colors = 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_puzzle_hash(puzzle))
        if seed < 3:
            assert _puzzle_hash(generate_puzzle(seed)) == _puzzle_hash(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                d = _read(inp)
                core, axes = _filter(d["tensor"], d["background"])
                bg = d["background"]
                shapes.add((inp.shape, out.shape))
                configurations.add(d["config"][:3])
                counts.add(len(core))
                widths.add(out.shape[1])
                palettes.add(tuple(sorted(axes)))
                lane_vectors.add(tuple(d["counts"]))
                length_vectors.add(tuple(d["lengths"]))
                axis_values.update(axes.values())
                noise_totals.add(int(np.sum(d["tensor"] != core)))
                core_sizes.add(int(np.sum(core != bg)))
                empty_slots += int(np.sum(np.all(core == bg, axis=(1, 2))))
                reused_key_colors += bool(set(axes) & set(d["config"][5:]))
                clean_colors += sum(np.array_equal(d["tensor"] == c, core == c) for c in axes)
    assert len(fingerprints) == 42 and len(shapes) >= 100
    assert len(configurations) == 8 and counts == set(range(5, 13))
    assert len(widths) >= 6 and len(lane_vectors) >= 35 and len(length_vectors) >= 150
    assert len(palettes) >= 40 and len(axis_values) >= 10
    assert len(noise_totals) >= 8 and len(core_sizes) >= 15
    assert empty_slots and reused_key_colors and clean_colors
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+10*a+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = generate_puzzle(72, 2, 1)
    d = _read(bad["test"][0]["input"])
    output = bad["test"][0]["output"]
    r, c = np.argwhere(output != d["background"])[0]
    output[r, c] = d["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("deleted supported cell was accepted")
    # Direct family members beyond the sampled 5..12-slot range, including
    # ties, a self-supported axis slot, and a carrier-only empty fragment.
    for n in (1, 3, 15):
        blocks = [np.array([[2, 6, 6, 6], [6, 6, 6, 6]], dtype=np.int8) for _ in range(n)]
        if n > 1:
            blocks[n//2][:] = 6
        counts_local = [min(5, n-i) for i in range(0, n, 5)]
        placements, start = [], 0
        for lane, count in enumerate(counts_local):
            for j in range(count):
                placements.append((1+3*lane, 1+5*j, 4))
            start += count
        shape = (3*len(counts_local)+1, 5*max(counts_local)+1)
        for config in sorted(configurations):
            inp = _paint(blocks, placements, shape, config, (0, 6, 8, 1))
            expected = np.full((2*n, 4), 0, dtype=np.int8)
            for i, block in enumerate(blocks):
                expected[2*i:2*i+2] = np.where(block == 6, 0, block)
            assert np.array_equal(solve(inp), expected)
    tied = np.zeros((4, 2, 2), dtype=np.int8)
    tied[0, 0, 0] = tied[1, 1, 1] = 2
    try:
        _filter(tied, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous axes were accepted")
    # The specification's worked example is an explicit expected subset,
    # independent of random construction and its selected orbit parameters.
    raw = np.zeros((6, 2, 6), dtype=np.int8)
    expected = raw.copy()
    points = {2: [(0, 0, 1), (4, 0, 1), (1, 1, 2), (3, 1, 2), (2, 0, 4)],
              3: [(0, 1, 0), (5, 1, 0), (1, 0, 3), (4, 0, 3), (2, 1, 4), (3, 1, 4)]}
    for color, cells in points.items():
        for p in cells:
            raw[p] = expected[p] = color
    raw[5, 1, 5], raw[0, 0, 5] = 2, 3
    clean, axes = _filter(raw, 0)
    assert axes == {2: 4, 3: 5} and np.array_equal(clean, expected)
    order = [0, 3, 1, 4, 2, 5]
    wrong, _ = _filter(raw[order], 0)
    assert not np.array_equal(wrong[np.argsort(order)], expected)
    blocks = [np.where(block == 0, 6, block).astype(np.int8) for block in raw]
    placements = [(1+3*(i//3), 1+7*(i % 3), 6) for i in range(6)]
    for config in sorted(configurations):
        inp = _paint(blocks, placements, (8, 23), config, (0, 6, 8, 1))
        assert np.array_equal(solve(inp), expected.reshape(12, 6))
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "shape_pairs": len(shapes), "configurations": len(configurations),
            "fragment_counts": sorted(counts), "widths": sorted(widths),
            "lane_vectors": len(lane_vectors), "length_vectors": len(length_vectors),
            "payload_palettes": len(palettes), "axis2_values": sorted(axis_values),
            "noise_totals": sorted(noise_totals), "core_sizes": len(core_sizes),
            "empty_output_slots": empty_slots, "key_color_payload_pairs": reused_key_colors,
            "clean_payload_colors": clean_colors,
            "semantic_checks": "author-owned; no independent or blind solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
