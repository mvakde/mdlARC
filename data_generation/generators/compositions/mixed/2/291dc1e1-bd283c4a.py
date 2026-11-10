"""L-key strip compaction followed by stable frequency-ranked column packing.

Ties use first appearance in the normalized, centered rectangle. This explicit
completion of proposal.json makes strip order observable when counts tie.
Ordering and rotation cannot change the histogram itself; the stronger saved
dependency claim is impossible. See composition.md for that limitation.
Only the standard library and NumPy are used; generation writes no files.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    boxes = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], [start]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    cells.append(q)
        r0, c0 = map(int, np.min(cells, axis=0))
        r1, c1 = (int(v)+1 for v in np.max(cells, axis=0))
        if len(cells) != (r1-r0)*(c1-c0):
            raise ValueError("strip contains a hole or is not a full rectangle")
        boxes.append((r0, r1, c0, c1))
    return boxes


def _read_input(input_grid):
    """Read visible key and payload only. Guide 1 is fast, guide 2 is slow."""
    g = _grid(input_grid)
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            aligned = g[::-1] if bottom else g
            aligned = aligned[:, ::-1] if right else aligned
            if min(g.shape) < 5:
                continue
            origin, horizontal, vertical, bg = map(int, (
                aligned[0, 0], aligned[0, 1], aligned[1, 0], aligned[-1, -1]))
            if (len({origin, horizontal, vertical, bg}) == 4
                    and {horizontal, vertical} == {1, 2}
                    and np.all(aligned[0, 1:] == horizontal)
                    and np.all(aligned[1:, 0] == vertical)
                    and np.all(aligned[-1, 1:] == bg)
                    and np.all(aligned[1:, -1] == bg)):
                candidates.append((bottom, right, horizontal == 2, bg, origin))
    if len(candidates) != 1:
        raise ValueError("expected one L-key with fast arm 1 and lane arm 2")
    bottom, right, axis, bg, origin = candidates[0]
    mask = g != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes = _components(mask)
    if not 1 <= len(boxes) <= 15:
        raise ValueError("expected 1..15 strips")
    lanes = {}
    for box in boxes:
        r0, r1, c0, c1 = box
        if not (r0 >= 1 and c0 >= 1 and r1 < g.shape[0] and c1 < g.shape[1]):
            raise ValueError("strips need clear margins on every canvas edge")
        if (c1-c0 if axis else r1-r0) != 2 or (r1-r0 if axis else c1-c0) < 2:
            raise ValueError("strip must be two cells thick along the slow axis")
        lanes.setdefault(c0 if axis else r0, []).append(box)
    offsets = sorted(lanes)
    if any(b-a < 3 for a, b in zip(offsets, offsets[1:])):
        raise ValueError("lane bands overlap or are staggered")
    slow_reverse, fast_reverse = (right, bottom) if axis else (bottom, right)
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    blocks, counts, positions = [], [], []
    for lane in sorted(lanes, reverse=slow_reverse):
        ordered = sorted(lanes[lane], key=lambda b: b[0] if axis else b[2],
                         reverse=fast_reverse)
        counts.append(len(ordered))
        for r0, r1, c0, c1 in ordered:
            blocks.append(np.rot90(g[r0:r1, c0:c1], rotation).copy())
            u = ((g.shape[0]-r1 if bottom else r0) if axis
                 else (g.shape[1]-c1 if right else c0))
            v = ((g.shape[1]-c1 if right else c0) if axis
                 else (g.shape[0]-r1 if bottom else r0))
            positions.append((u, v))
    lengths = [b.shape[1] for b in blocks]
    if len({n % 2 for n in lengths}) != 1:
        raise ValueError("all strip lengths must share parity for exact centering")
    return {"blocks": blocks, "lane_counts": counts, "positions": positions,
            "lengths": lengths, "background": bg, "origin": origin,
            "configuration": (int(axis), bottom, right)}


def _rectangle(blocks, background):
    width = max(b.shape[1] for b in blocks)
    result = np.full((2*len(blocks), width), background, dtype=np.int8)
    for i, b in enumerate(blocks):
        left = (width-b.shape[1]) // 2
        result[2*i:2*i+2, left:left+b.shape[1]] = b
    return result


def _ranking(rectangle):
    counts, first = {}, {}
    for i, value in enumerate(rectangle.flat):
        c = int(value)
        counts[c] = counts.get(c, 0) + 1
        first.setdefault(c, i)
    order = sorted(counts, key=lambda c: (-counts[c], first[c]))
    return order, counts


def _pack(rectangle):
    colors, counts = _ranking(rectangle)
    stream = np.array([c for c in colors for _ in range(counts[c])], dtype=np.int8)
    h, w = rectangle.shape
    return stream.reshape(w, h)[:, ::-1].T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Normalize the keyed strips, then pack their stable frequency ranking."""
    d = _read_input(input_grid)
    return _pack(_rectangle(d["blocks"], d["background"]))


def _alternatives(d):
    """Explicit counterfactuals for the inherited mechanisms, not extra rules."""
    blocks, groups, start = d["blocks"], [], 0
    for count in d["lane_counts"]:
        groups.append(list(range(start, start+count)))
        start += count
    orders = {
        "lane_reverse": [i for group in groups[::-1] for i in group],
        "fast_reverse": [i for group in groups for i in group[::-1]],
        "fast_major": sorted(range(len(blocks)), key=lambda i: d["positions"][i]),
    }
    bg = d["background"]
    results = {name: _pack(_rectangle([blocks[i] for i in order], bg))
               for name, order in orders.items()}
    for name, transformed in (
            ("flip_rows", [b[::-1] for b in blocks]),
            ("flip_columns", [b[:, ::-1] for b in blocks]),
            ("rotate_180", [b[::-1, ::-1] for b in blocks])):
        results[name] = _pack(_rectangle(transformed, bg))
    rectangle = _rectangle(blocks, bg)
    order, counts = _ranking(rectangle)
    encounter = list(dict.fromkeys(map(int, rectangle.flat)))
    h, w = rectangle.shape
    stream = [c for c in encounter for _ in range(counts[c])]
    results["encounter_only"] = np.array(stream, dtype=np.int8).reshape(w, h)[:, ::-1].T
    numeric = sorted(counts, key=lambda c: (-counts[c], c))
    stream = [c for c in numeric for _ in range(counts[c])]
    results["numeric_ties"] = np.array(stream, dtype=np.int8).reshape(w, h)[:, ::-1].T
    results["omit_B"] = rectangle
    return results


def _traversals(rectangle):
    colors, counts = _ranking(rectangle)
    stream = [c for c in colors for _ in range(counts[c])]
    h, w = rectangle.shape
    results = []
    for row_major in (False, True):
        for reverse_major in (False, True):
            for reverse_minor in (False, True):
                nr, nc = (h, w) if row_major else (w, h)
                major = range(nr-1, -1, -1) if reverse_major else range(nr)
                minor = range(nc-1, -1, -1) if reverse_minor else range(nc)
                positions = [(a, b) if row_major else (b, a)
                             for a in major for b in minor]
                out = np.empty_like(rectangle)
                for position, color in zip(positions, stream):
                    out[position] = color
                results.append(out)
    return results


def _distribute_gaps(rng, occupied, items, extent):
    spare = extent - occupied - items - 1
    if spare < 0:
        raise ValueError("layout exceeds the canvas budget")
    weights = rng.dirichlet(np.full(items+1, rng.choice([.4, 1., 3.])))
    return 1 + rng.multinomial(spare, weights)


def _layout(rng, witness=False, test=False):
    base = int(rng.choice([2, 3], p=[.72, .28]))
    n = int(rng.integers(5, 13) if test and witness else
            rng.integers(4, 10) if witness else rng.integers(1, 16))
    capacity = 28 // (base+1)
    low = max(2 if witness else 1, (n+capacity-1)//capacity)
    high = min(9, n//2 if witness else n)
    lanes = int(rng.integers(low, min(high, 4)+1) if witness
                else rng.integers(low, high+1))
    counts = [2 if witness else 1]*lanes
    for _ in range(n-sum(counts)):
        eligible = [i for i, c in enumerate(counts) if c < capacity]
        counts[int(rng.choice(eligible))] += 1
    equal = not witness and rng.random() < .24
    common_limit = min((28-c)//c for c in counts)
    common = base + 2*int(rng.integers((common_limit-base)//2+1))
    lengths = []
    for count in counts:
        if equal:
            group = [common]*count
        else:
            extra = (28-count-count*base)//2
            units = int(rng.integers(extra+1))
            additions = rng.multinomial(units, rng.dirichlet(np.ones(count)))
            group = [base+2*int(x) for x in additions]
            rng.shuffle(group)
        lengths.append(group)
    if witness and len({v for group in lengths for v in group}) == 1:
        choices = [(j, k) for j, group in enumerate(lengths)
                   for k in range(len(group)) if sum(group)+len(group)+3 <= 29]
        j, k = choices[int(rng.integers(len(choices)))]
        lengths[j][k] += 2
    slow_min = 3*lanes+1
    fast_min = max(sum(group)+len(group)+1 for group in lengths)
    slow_extent = int(rng.integers(slow_min, 30))
    fast_extent = int(rng.integers(fast_min, 30))
    slow_gaps = _distribute_gaps(rng, 2*lanes, lanes, slow_extent)
    placements, flat_lengths, v = [], [], int(slow_gaps[0])
    for j, group in enumerate(lengths):
        gaps = _distribute_gaps(rng, sum(group), len(group), fast_extent)
        u = int(gaps[0])
        for k, length in enumerate(group):
            placements.append((v, u, length))
            flat_lengths.append(length)
            u += length+int(gaps[k+1])
        v += 2+int(slow_gaps[j+1])
    return flat_lengths, placements, (slow_extent, fast_extent), counts


def _paint(blocks, placements, shape, configuration, bg, origin):
    axis, bottom, right = configuration
    interior = np.full(shape, bg, dtype=np.int8)
    # Correct thickness so the net transform of each block is a proper rotation.
    reverse_thickness = (bottom == right) if axis else (bottom != right)
    for block, (v, u, length) in zip(blocks, placements):
        interior[v:v+2, u:u+length] = block[::-1] if reverse_thickness else block
    grid = np.full((shape[0]+1, shape[1]+1), bg, dtype=np.int8)
    grid[1:, 1:] = interior
    grid[0, 0], grid[0, 1:], grid[1:, 0] = origin, 1, 2
    if axis:
        grid = grid.T
    if bottom:
        grid = grid[::-1]
    if right:
        grid = grid[:, ::-1]
    return grid.copy()


def _payloads(rng, lengths, palette, tied, mode):
    total, k = 2*sum(lengths), len(palette)
    if mode == "solid" and not tied:
        blocks = []
        for length in lengths:
            previous = [b for b in blocks if b.shape[1] == length]
            if previous and rng.random() < .35:
                blocks.append(previous[int(rng.integers(len(previous)))].copy())
            else:
                blocks.append(np.full((2, length), rng.choice(palette), dtype=np.int8))
        return blocks
    if tied:
        # Three equal-frequency colors give varied stable-order witnesses.
        f = int(rng.integers(1, (total-(k-3))//3+1))
        tail = rng.multinomial(total-3*f-(k-3), rng.dirichlet(np.ones(k-3)))+1
        counts = np.concatenate(([f]*3, tail))
    else:
        counts = rng.multinomial(total-k, rng.dirichlet(np.full(k, rng.choice([.5, 2., 8.]))))+1
    stream = np.repeat(palette, counts).astype(np.int8)
    if mode == "scatter":
        rng.shuffle(stream)
    elif mode == "ribbons":
        stream = np.roll(stream, int(rng.integers(total)))
        stream = stream[np.argsort(np.arange(total) % int(rng.integers(2, 8)), kind="stable")]
    elif mode == "patches":
        # Permuting chunks keeps the sampled inventory exactly.
        chunks = np.array_split(stream, int(rng.integers(3, min(18, total)+1)))
        rng.shuffle(chunks)
        stream = np.concatenate(chunks)
    else:
        rng.shuffle(stream)
    blocks, start = [], 0
    for length in lengths:
        blocks.append(stream[start:start+2*length].reshape(2, length).copy())
        start += 2*length
    return blocks


def _new_pair(rng, configuration, witness=None, test=False, avoid_bg=None, avoid_top=None):
    local = [c for c in range(10) if c not in (1, 2, avoid_bg)]
    bg = int(rng.choice(local))
    origin = int(rng.choice([c for c in range(10) if c not in (1, 2, bg)]))
    # Layout retries allow geometry and histogram constraints to combine.
    for layout_attempt in range(12):
        lengths, placements, shape, lane_counts = _layout(rng, witness is not None, test)
        total = 2*sum(lengths)
        for _ in range(160):
            tied = witness is not None or rng.random() < .50
            low = 4 if tied else 2
            k = int(rng.integers(low, min(8, total, 7 if witness else 8)+1))
            palette = rng.choice([c for c in range(10) if c != bg], size=k, replace=False)
            mode = "scatter" if witness else str(rng.choice(["scatter", "ribbons", "patches", "solid"]))
            blocks = _payloads(rng, lengths, palette, tied, mode)
            rectangle = _rectangle(blocks, bg)
            output = _pack(rectangle)
            d = {"blocks": blocks, "lane_counts": lane_counts, "background": bg,
                 "positions": [(u, v) for v, u, length in placements]}
            if witness:
                alternatives = _alternatives(d)
                names = (("lane_reverse", "fast_reverse", "fast_major") if witness == "lanes"
                         else ("flip_rows", "flip_columns", "rotate_180"))
                names += ("omit_B", "encounter_only", "numeric_ties")
                if any(np.array_equal(output, alternatives[name]) for name in names):
                    continue
                if sum(np.array_equal(output, x) for x in _traversals(rectangle)) != 1:
                    continue
                if avoid_top is not None and _ranking(rectangle)[0][0] == avoid_top:
                    continue
            grid = _paint(blocks, placements, shape, configuration, bg, origin)
            return {"input": grid, "output": output}
    raise RuntimeError("bounded construction could not realize the requested witness")


def _reverse_key_lanes(input_grid):
    """Controlled contrast: unchanged payload and histogram, reversed lane key."""
    d = _read_input(input_grid)
    axis, bottom, right = d["configuration"]
    g, bg = np.array(input_grid, copy=True), d["background"]
    g[-1 if bottom else 0, :] = bg
    g[:, -1 if right else 0] = bg
    bottom, right = (bottom, not right) if axis else (not bottom, right)
    r, c = (-1 if bottom else 0), (-1 if right else 0)
    g[r, :], g[:, c] = (2 if axis else 1), (1 if axis else 2)
    g[r, c] = d["origin"]
    return g


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    configurations = [(a, b, c) for a in (0, 1) for b in (False, True) for c in (False, True)]
    first = configurations[int(rng.integers(8))]
    second = (1-first[0], not first[1], not first[2])
    # Reserve one unseen combination even when all ten demonstrations are requested.
    reversed_first = ((first[0], first[1], not first[2]) if first[0]
                      else (first[0], not first[1], first[2]))
    held = [c for c in configurations if c not in (first, second, reversed_first)]
    held = held[int(rng.integers(len(held)))]
    training_pool = [c for c in configurations if c != held]
    pair0 = _new_pair(rng, first, "lanes")
    d0 = _read_input(pair0["input"])
    top0 = _ranking(_rectangle(d0["blocks"], d0["background"]))[0][0]
    pair1 = _new_pair(rng, second, "rotation", avoid_bg=d0["background"], avoid_top=top0)
    puzzle = {"train": [pair0, pair1], "test": []}
    seen = {_signature(p["input"]) for p in puzzle["train"]}
    for index in range(2, num_train):
        if index == 2:
            g = _reverse_key_lanes(pair0["input"])
            # Avoid putting the held-out configuration in training.
            if _read_input(g)["configuration"] != held:
                pair = {"input": g, "output": solve(g)}
            else:
                pair = _new_pair(rng, training_pool[int(rng.integers(len(training_pool)))])
        else:
            pair = _new_pair(rng, training_pool[int(rng.integers(len(training_pool)))])
        for _ in range(20):
            if _signature(pair["input"]) not in seen:
                break
            pair = _new_pair(rng, training_pool[int(rng.integers(len(training_pool)))])
        else:
            raise RuntimeError("could not generate a distinct input")
        seen.add(_signature(pair["input"]))
        puzzle["train"].append(pair)
    for index in range(num_test):
        config = held if index == 0 else configurations[int(rng.integers(8))]
        for _ in range(20):
            pair = _new_pair(rng, config, "lanes" if index == 0 else None, test=True)
            if _signature(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("could not generate a distinct test input")
        seen.add(_signature(pair["input"]))
        puzzle["test"].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(g)
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            d = _read_input(g)
            rectangle = _rectangle(d["blocks"], d["background"])
            assert out.shape == rectangle.shape
            assert np.array_equal(np.bincount(rectangle.ravel(), minlength=10),
                                  np.bincount(out.ravel(), minlength=10)), "inventory changed"
            assert np.array_equal(out, _pack(rectangle)), "wrong stable frequency raster"
            payload_cells = sum(b.size for b in d["blocks"])
            assert int(np.count_nonzero(out == d["background"])) == rectangle.size-payload_cells
            assert np.array_equal(solve(g.copy()), out)
            assert not np.array_equal(g, out)
            flags = {name: not np.array_equal(out, alt) for name, alt in _alternatives(d).items()}
            record = (d, rectangle, flags)
            (training if split == "train" else testing).append(record)
    for name in ("lane_reverse", "fast_reverse", "fast_major", "flip_rows", "flip_columns", "rotate_180",
                 "omit_B", "encounter_only", "numeric_ties"):
        assert any(flags[name] for d, rect, flags in training), f"no training witness against {name}"
    assert any(max(d["lengths"]) > min(d["lengths"]) for d, rect, flags in training)
    assert any(sum(np.array_equal(_pack(rect), alt) for alt in _traversals(rect)) == 1
               for d, rect, flags in training), "raster direction is ambiguous"
    assert len({_ranking(rect)[0][0] for d, rect, flags in training}) >= 2
    assert len({d["configuration"][0] for d, rect, flags in training}) == 2
    # Change only the slow-direction border key, not a single payload cell.
    # This must change a demonstrated tied rank, while conserving the inventory.
    assert any(not np.array_equal(solve(_reverse_key_lanes(pair["input"])), pair["output"])
               for pair in puzzle["train"]), "the L-key has no demonstrated causal effect"
    train_configs = {d["configuration"] for d, rect, flags in training}
    assert testing[0][0]["configuration"] not in train_configs, "missing orientation combination shift"
    assert testing[0][2]["lane_reverse"] and testing[0][2]["fast_reverse"]


def _digest(puzzle):
    # Exact, bounded signature; never retain all generated grids for diversity.
    h = hashlib.sha256()
    for s in ("train", "test"):
        for pair in puzzle[s]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streaming seeds, causal counterfactuals, counts and malformed cases."""
    signatures, shapes, configs, layouts, palettes, backgrounds, origins = (set() for _ in range(7))
    widths, populations, tied_pairs = set(), set(), 0
    rekey_changes = untied_invariants = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                d = _read_input(g)
                rect = _rectangle(d["blocks"], d["background"])
                colors, counts = _ranking(rect)
                # Direct coordinate assertion independent of reshape packing.
                cursor = 0
                for color in colors:
                    for _ in range(counts[color]):
                        assert out[out.shape[0]-1-cursor % out.shape[0], cursor // out.shape[0]] == color
                        cursor += 1
                tied = len(set(counts.values())) != len(counts)
                tied_pairs += tied
                if not tied:
                    reversed_rect = _rectangle(d["blocks"][::-1], d["background"])
                    assert np.array_equal(out, _pack(reversed_rect))
                    untied_invariants += 1
                rekey = _reverse_key_lanes(g)
                out2 = solve(rekey)
                assert out2.shape == out.shape
                assert np.array_equal(np.bincount(out.ravel(), minlength=10),
                                      np.bincount(out2.ravel(), minlength=10))
                rekey_changes += not np.array_equal(out, out2)
                for rotation in range(4):
                    assert np.array_equal(solve(np.rot90(g, rotation)), out)
                shapes.add((g.shape, out.shape))
                configs.add(d["configuration"])
                layouts.add((tuple(d["lane_counts"]), tuple(d["lengths"])))
                palettes.add(len(counts))
                backgrounds.add(d["background"])
                origins.add(d["origin"])
                widths.add(out.shape[1])
                populations.add(len(d["blocks"]))
    assert len(signatures) == 42 and len(shapes) >= 100 and len(layouts) >= 100
    assert len(configs) == 8 and len(palettes) >= 4 and len(widths) >= 10
    assert len(populations) >= 10 and len(backgrounds) == len(origins) == 8
    assert tied_pairs >= 126 and rekey_changes >= 84 and untied_invariants >= 1
    # Every legal count combination is exercised without retaining prior puzzles.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(4200+nt*10+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    # Fixed hand-computable stable tie regression, independent of random layouts.
    blocks = [np.array([[4, 3], [5, 5]], dtype=np.int8),
              np.array([[3, 4, 5, 5], [5, 5, 5, 5]], dtype=np.int8)]
    rect = _rectangle(blocks, 8)
    assert _ranking(rect)[0] == [5, 8, 4, 3]
    assert _ranking(_rectangle(blocks[::-1], 8))[0] == [5, 8, 3, 4]
    assert np.array_equal(_pack(rect), np.array([[5, 5, 8, 3], [5, 5, 8, 3],
                                               [5, 5, 8, 4], [5, 5, 8, 4]], dtype=np.int8))
    # Construct boundary members directly, including widths and populations
    # that need not occur naturally in a small random seed suite.
    for lengths, lane_counts in (([2]*15, [9, 6]), ([3, 7, 3], [1, 2]),
                                 ([2, 4, 26], [1, 1, 1]), ([27], [1])):
        blocks = [(np.arange(2*n).reshape(2, n)+i) % 3+3
                  for i, n in enumerate(lengths)]
        placements, start, v, fast_extent = [], 0, 1, 0
        for count in lane_counts:
            u = 1
            for length in lengths[start:start+count]:
                placements.append((v, u, length))
                u += length+1
            fast_extent = max(fast_extent, u)
            start += count
            v += 3
        for axis in (0, 1):
            for bottom in (False, True):
                for right in (False, True):
                    g = _paint(blocks, placements, (v, fast_extent),
                               (axis, bottom, right), 8, 0)
                    assert np.array_equal(solve(g), _pack(_rectangle(blocks, 8)))
    # Malformed geometry must be rejected using the input alone.
    square = np.full((2, 2), 3, dtype=np.int8)
    mixed = _paint([square, np.full((2, 3), 4, dtype=np.int8)],
                   [(1, 1, 2), (4, 1, 3)], (7, 5), (0, False, False), 8, 0)
    hole = _paint([square], [(1, 1, 2)], (4, 4), (0, False, False), 8, 0)
    hole[2, 2] = 8
    stagger = _paint([square, square], [(1, 1, 2), (2, 5, 2)],
                     (6, 8), (0, False, False), 8, 0)
    for malformed in (mixed, hole, stagger, np.zeros((4, 4), dtype=int)):
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed input accepted")
    # A bad answer must be rejected.
    bad = generate_puzzle(99)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "shape_pairs": len(shapes), "layouts": len(layouts), "configurations": len(configs),
            "palette_sizes": sorted(palettes), "output_widths": sorted(widths),
            "strip_counts": sorted(populations), "tied_pairs": tied_pairs,
            "rekey_changes": rekey_changes, "untied_order_invariants": untied_invariants}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
