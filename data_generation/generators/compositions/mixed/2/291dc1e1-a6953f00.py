"""L-key traversal followed by canvas-parity cropping of each rotated strip.

No parent modules, files, training examples, or generation state are used by solve.
The long dimension of at least one strip visibly identifies the fast key arm.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= v <= 30 for v in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, a.tobytes()


def _components(mask):
    """Four-connected components of occupancy, independent of payload colors."""
    remaining = set(zip(*np.nonzero(mask)))
    boxes = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, points = [first], [first]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    points.append(q)
        r0, c0 = map(int, np.min(points, axis=0))
        r1, c1 = (int(v)+1 for v in np.max(points, axis=0))
        _require(len(points) == (r1-r0)*(c1-c0), "strip has a hole or irregular outline")
        boxes.append((r0, r1, c0, c1))
    return boxes


def _read(input_grid):
    """Determine the unique key, axis, ordered boxes, and rotations from input."""
    g = _grid(input_grid)
    n = g.shape[0]
    _require(g.shape == (n, n) and n >= 6, "square canvas of side 6..30 required")
    keys = []
    for bottom in (False, True):
        for right in (False, True):
            aligned = g[::-1] if bottom else g
            aligned = aligned[:, ::-1] if right else aligned
            origin, horizontal, vertical, bg = map(int, (
                aligned[0, 0], aligned[0, 1], aligned[1, 0], aligned[-1, -1]))
            if len({origin, horizontal, vertical, bg}) != 4:
                continue
            if (np.all(aligned[0, 1:] == horizontal)
                    and np.all(aligned[1:, 0] == vertical)
                    and np.all(aligned[-1, 1:] == bg)
                    and np.all(aligned[1:, -1] == bg)):
                keys.append((bottom, right, origin, horizontal, vertical, bg))
    _require(len(keys) == 1, "one unambiguous L-key required")
    bottom, right, origin, horizontal, vertical, bg = keys[0]
    mask = g != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes = _components(mask)
    _require(1 <= len(boxes) <= 15, "one through fifteen strips required")
    axes = set()
    for r0, r1, c0, c1 in boxes:
        _require(r0 >= 1+int(not bottom) and r1 <= n-1-int(bottom)
                 and c0 >= 1+int(not right) and c1 <= n-1-int(right),
                 "one background cell must separate payload from every border")
        h, w = r1-r0, c1-c0
        _require(min(h, w) == 2, "strips must be two cells thick")
        if h != w:
            axes.add(int(h > w))
    _require(len(axes) == 1, "at least one elongated strip, all with the same fast axis")
    axis = axes.pop()  # 0 horizontal, 1 vertical
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    records = []
    for box in boxes:
        r0, r1, c0, c1 = box
        _require((c1-c0 if axis else r1-r0) == 2, "inconsistent strip thickness")
        # Distances from the origin borders give positive slow/fast coordinates.
        vr = n-r1 if bottom else r0
        uc = n-c1 if right else c0
        v, u = (uc, vr) if axis else (vr, uc)
        block = np.rot90(g[r0:r1, c0:c1], rotation).copy()
        records.append((v, u, box, block))
    records.sort(key=lambda x: (x[0], x[1]))
    lanes = sorted({x[0] for x in records})
    _require(all(b-a >= 3 for a, b in zip(lanes, lanes[1:])), "overlapping or staggered lanes")
    for v in lanes:
        lane = [x for x in records if x[0] == v]
        _require(all(b[1] >= a[1]+a[3].shape[1]+1 for a, b in zip(lane, lane[1:])),
                 "strips must be separated within each lane")
    return {"n": n, "axis": axis, "bottom": bottom, "right": right,
            "roles": (bg, origin, horizontal, vertical), "rotation": rotation,
            "records": records, "blocks": [x[3] for x in records],
            "counts": [sum(x[0] == v for x in records) for v in lanes]}


def _crop_stack(blocks, odd):
    return np.concatenate([b[:, :2] if odd else b[:, -2:] for b in blocks], axis=0).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Rotate first; crop each strip using the ORIGINAL square side parity."""
    d = _read(input_grid)
    return _crop_stack(d["blocks"], d["n"] % 2)


def _gaps(rng, occupied, count, n, near=None):
    # Two cells from origin (key then background); at least one at other gaps.
    gaps = np.ones(count+1, dtype=int)
    gaps[0] = 2 if near is None else near
    spare = n-occupied-int(gaps.sum())
    _require(spare >= 0, "packing budget")
    weights = rng.dirichlet(np.full(count+1, float(rng.choice([.4, 1., 3.]))))
    if near is not None:
        weights[0] = 0
        weights /= weights.sum()
    gaps += rng.multinomial(spare, weights)
    return gaps.tolist()


def _layout(rng, witness, maximum, total=None):
    if total is None:
        total = int(rng.integers(4, 8) if witness else rng.integers(1, 16))
    if witness:
        lane_count = int(rng.integers(2, min(3, total//2)+1))
        counts = [2]*lane_count
    else:
        lo = (total+8)//9
        choices = np.arange(lo, min(total, 9)+1)
        weights = np.array([3. if 2 <= x <= 4 else 1. for x in choices])
        lane_count = int(rng.choice(choices, p=weights/weights.sum()))
        counts = [1]*lane_count
    for _ in range(total-sum(counts)):
        eligible = [i for i, c in enumerate(counts) if c < 9]
        counts[int(rng.choice(eligible))] += 1
    ceiling = int(rng.integers(4, 10) if witness else rng.integers(3, maximum-2))
    groups = []
    for count in counts:
        lengths = rng.integers(2, ceiling+1, size=count).tolist()
        budget = maximum-count-2
        while sum(lengths) > budget:
            candidates = [i for i, x in enumerate(lengths) if x > 2]
            lengths[int(rng.choice(candidates))] -= 1
        groups.append(lengths)
    if max(x for group in groups for x in group) == 2:
        eligible = [i for i, group in enumerate(groups)
                    if sum(group)+len(group)+2 < maximum]
        groups[int(rng.choice(eligible))][0] = 3
    if witness:
        # Both length parities occur; centering's shared-parity restriction is gone.
        flat = [x for group in groups for x in group]
        if len({x % 2 for x in flat}) == 1:
            group = groups[int(rng.integers(lane_count))]
            i = int(rng.integers(len(group)))
            group[i] = group[i]-1 if group[i] > 2 else 3
    minimum = max(3*lane_count+2, max(sum(g)+len(g)+2 for g in groups))
    n = int(rng.integers(minimum, maximum+1))
    vgaps = _gaps(rng, 2*lane_count, lane_count, n)
    # Aligned lane starts in the witnesses make lane-major vs fast-major observable.
    near = None
    if witness:
        spare = min(n-sum(g)-len(g)-2 for g in groups)
        near = 2+int(rng.integers(spare+1))
    placements, lengths, v = [], [], vgaps[0]
    for j, group in enumerate(groups):
        ugaps = _gaps(rng, sum(group), len(group), n, near)
        u = ugaps[0]
        for i, length in enumerate(group):
            placements.append((v, u, length))
            lengths.append(length)
            u += length+ugaps[i+1]
        v += 2+vgaps[j+1]
    return n, placements, lengths


def _asymmetric(b):
    return all(not np.array_equal(b, other) for other in (b[::-1], b[:, ::-1], b[::-1, ::-1]))


def _block(rng, length, palette):
    mode = int(rng.integers(5))
    if mode == 0:
        return np.full((2, length), rng.choice(palette), dtype=np.int8)
    if mode == 1:
        return np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1).astype(np.int8)
    if mode == 2:
        return np.repeat(rng.choice(palette, size=(1, length)), 2, axis=0).astype(np.int8)
    if mode == 3:
        period = int(rng.integers(1, min(4, length)+1))
        return np.tile(rng.choice(palette, size=(2, period)), (1, length))[:, :length].astype(np.int8)
    return rng.choice(palette, size=(2, length)).astype(np.int8)


def _payloads(rng, lengths, bg, witness):
    pool = [c for c in range(10) if c != bg]
    palette = rng.choice(pool, size=int(rng.integers(3 if witness else 1, 10)), replace=False)
    if witness:
        for _ in range(128):
            blocks = [rng.choice(palette, size=(2, length)).astype(np.int8) for length in lengths]
            ends = [[b[:, :2] for b in blocks], [b[:, -2:] for b in blocks]]
            if (all(len({_signature(b) for b in end}) == len(blocks) for end in ends)
                    and all(any(_asymmetric(b) for b in end) for end in ends)
                    and any(not np.array_equal(b[:, :2], b[:, -2:][::-1, ::-1]) for b in blocks)
                    and not np.array_equal(_crop_stack(blocks, True), _crop_stack(blocks, False))):
                return blocks
        raise RuntimeError("bounded witness sampling exhausted")
    blocks = []
    solid = rng.random() < .18
    for length in lengths:
        repeated = [b for b in blocks if b.shape[1] == length]
        if repeated and rng.random() < .25:
            block = repeated[int(rng.integers(len(repeated)))].copy()
        elif solid:
            block = np.full((2, length), rng.choice(palette), dtype=np.int8)
        else:
            block = _block(rng, length, palette)
        blocks.append(block)
    return blocks


def _render(n, placements, blocks, config, roles):
    axis, bottom, right = config
    bg, origin, fast, slow = roles
    grid = np.full((n, n), bg, dtype=np.int8)
    rkey, ckey = (n-1 if bottom else 0), (n-1 if right else 0)
    grid[rkey, :] = slow if axis else fast
    grid[:, ckey] = fast if axis else slow
    grid[rkey, ckey] = origin
    k = (3 if bottom else 1) if axis else (2 if right else 0)
    for (v, u, length), block in zip(placements, blocks):
        h, w = (length, 2) if axis else (2, length)
        rd, cd = (u, v) if axis else (v, u)
        r0 = n-rd-h if bottom else rd
        c0 = n-cd-w if right else cd
        grid[r0:r0+h, c0:c0+w] = np.rot90(block, -k)
    # Ground truth comes from pre-placement payloads, independently of the reader.
    return {"input": grid, "output": _crop_stack(blocks, n % 2)}


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    all_configs = [(a, b, r) for a in (0, 1) for b in (False, True) for r in (False, True)]
    # The first demonstration cannot accidentally make the key dispensable.
    first_pool = [c for c in all_configs if c[0] == 1 or c[2]]
    first = first_pool[int(rng.integers(len(first_pool)))]
    second = (1-first[0], not first[1], not first[2])
    puzzle = {"train": [], "test": []}
    used = set()
    familiar_configs = [first]
    previous_side = None
    for config in ([first, second] if num_train >= 4 else [first]):
        n, placements, lengths = _layout(rng, True, 29 if previous_side is None else 26)
        if previous_side is not None:
            # Two disjoint adjacent-size pairs give two sizes of each parity.
            # The spare far margin does not change any distances from the key.
            options = [x for x in range(n, 30)
                       if x % 2 == previous_side % 2 and x != previous_side]
            n = int(rng.choice(options))
        previous_side = n
        roles = tuple(map(int, rng.choice(10, size=4, replace=False)))
        blocks = _payloads(rng, lengths, roles[0], True)
        for side in (n, n+1):
            pair = _render(side, placements, blocks, config, roles)
            used.add(_signature(pair["input"]))
            puzzle["train"].append(pair)
        if config not in familiar_configs:
            familiar_configs.append(config)
    for split, target in (("train", num_train), ("test", num_test)):
        while len(puzzle[split]) < target:
            if split == "test" and num_train >= 4:
                pool = [c for c in all_configs if c not in familiar_configs]
            else:
                pool = familiar_configs
            config = pool[int(rng.integers(len(pool)))]
            target_count = None
            if split == "test" and num_train < 4:
                familiar_counts = {len(p["output"])//2 for p in puzzle["train"]}
                target_count = int(rng.choice([x for x in range(1, 16) if x not in familiar_counts]))
            for _ in range(64):
                n, placements, lengths = _layout(rng, False, 30, target_count)
                roles = tuple(map(int, rng.choice(10, size=4, replace=False)))
                blocks = _payloads(rng, lengths, roles[0], False)
                pair = _render(n, placements, blocks, config, roles)
                key = _signature(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input sampling exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternatives(d):
    """Concrete ablations; used for evidence checks, never for solving."""
    blocks, records, odd = d["blocks"], d["records"], d["n"] % 2
    groups, start = [], 0
    for count in d["counts"]:
        groups.append(list(range(start, start+count)))
        start += count
    orders = {
        "reverse_fast": [i for group in groups for i in group[::-1]],
        "reverse_lanes": [i for group in groups[::-1] for i in group],
        "fast_major": sorted(range(len(blocks)), key=lambda i: (records[i][1], records[i][0])),
        "ignore_key_order": sorted(range(len(blocks)), key=lambda i: records[i][2]),
    }
    result = {name: _crop_stack([blocks[i] for i in order], odd) for name, order in orders.items()}
    result.update({"always_left": _crop_stack(blocks, True),
                   "always_right": _crop_stack(blocks, False),
                   "strip_parity": np.concatenate([b[:, :2] if b.shape[1] % 2 else b[:, -2:] for b in blocks]),
                   "reflect_rows": _crop_stack([b[::-1] for b in blocks], odd),
                   "reflect_columns": _crop_stack([b[:, ::-1] for b in blocks], odd)})
    physical = [np.rot90(b, -d["rotation"]) for b in blocks]
    result["skip_rotation"] = np.concatenate([b[:2, :2] if odd else b[:2, -2:] for b in physical])
    return result


def _twin_signature(d):
    return (d["axis"], d["bottom"], d["right"], d["roles"],
            tuple((v, u, _signature(b)) for v, u, _, b in d["records"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "list splits required")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, queries = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(g)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _read(g)
            expected = _crop_stack(d["blocks"], d["n"] % 2)
            _require(out.shape == (2*len(d["blocks"]), 2), "output must stack one 2x2 per strip")
            _require(np.array_equal(out, expected), "wrong ordered parity crop")
            _require(set(map(int, out.flat)) <= set(map(int, g.flat)), "invented output color")
            if split == "train":
                training.append(d)
            else:
                queries.append(d)
    _require({d["n"] % 2 for d in training} == {0, 1}, "both canvas parities need training")
    twins = []
    for i, a in enumerate(training):
        for b in training[i+1:]:
            if abs(a["n"]-b["n"]) == 1 and _twin_signature(a) == _twin_signature(b):
                _require(not np.array_equal(_crop_stack(a["blocks"], True),
                                            _crop_stack(a["blocks"], False)), "uninformative parity twins")
                twins.append((a, b))
    _require(len(twins) >= (2 if len(training) >= 4 else 1), "matched parity demonstrations missing")
    if len(training) >= 4:
        _require({d["axis"] for d in training} == {0, 1}, "both axes need demonstrations")
        for parity in (0, 1):
            _require(len({d["n"] for d in training if d["n"] % 2 == parity}) >= 2,
                     "parity must appear at multiple scales")
    distinguished = set()
    for d in training:
        expected = _crop_stack(d["blocks"], d["n"] % 2)
        distinguished.update(name for name, out in _alternatives(d).items()
                             if not np.array_equal(out, expected))
    _require(distinguished == set(_alternatives(training[0])), "training leaves a core shortcut unresolved")
    if len(training) >= 4:
        familiar = {(d["axis"], d["bottom"], d["right"]) for d in training}
        _require(all((d["axis"], d["bottom"], d["right"]) not in familiar for d in queries),
                 "tests must hold out a key/axis combination")
    else:
        familiar = {len(d["blocks"]) for d in training}
        _require(all(len(d["blocks"]) not in familiar for d in queries), "test count shift missing")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed sweep; metrics only, no saved grids or execution traces."""
    digests, layouts, sides, lengths, configs, populations, backgrounds, palettes = (set() for _ in range(8))
    duplicates = uniform = mixed_lengths = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = _puzzle_digest(p)
        _require(digest == _puzzle_digest(generate_puzzle(seed)), "seed reproducibility")
        digests.add(digest)
        for split in ("train", "test"):
            for pair in p[split]:
                d = _read(pair["input"])
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "input-only solution")
                sides.add(d["n"])
                lengths.update(b.shape[1] for b in d["blocks"])
                configs.add((d["axis"], d["bottom"], d["right"]))
                populations.add(tuple(d["counts"]))
                backgrounds.add(d["roles"][0])
                palettes.add(tuple(np.unique(pair["input"])))
                layouts.add((d["n"], d["axis"], d["bottom"], d["right"],
                             tuple((v, u, b.shape[1]) for v, u, _, b in d["records"])))
                duplicates += len({_signature(b) for b in d["blocks"]}) < len(d["blocks"])
                uniform += any(len(np.unique(b)) == 1 for b in d["blocks"])
                mixed_lengths += len({b.shape[1] % 2 for b in d["blocks"]}) == 2
    _require(len(digests) == 42 and len(layouts) >= 110, "structural diversity")
    _require(len(configs) == 8 and len(sides) >= 16 and len(lengths) >= 14, "geometric diversity")
    _require(len(populations) >= 25 and len(backgrounds) == 10 and len(palettes) >= 20, "role/count diversity")
    _require(duplicates > 0 and uniform > 0 and mixed_lengths >= 42, "simple and mixed patterns")
    # Known physical strips: these expected crops are written by hand, not rendered.
    for n in (9, 10):
        g = np.full((n, n), 0, dtype=np.int8)
        g[0, :] = 1
        g[:, -1] = 2
        g[0, -1] = 6
        g[2:4, 2:6] = [[3, 4, 5, 6], [7, 8, 9, 3]]
        expected = [[3, 9], [6, 5]] if n % 2 else [[8, 7], [4, 3]]
        _require(np.array_equal(solve(g), expected), "leftward proper rotation regression")
        g = np.full((n, n), 0, dtype=np.int8)
        g[0, :] = 2
        g[:, 0] = 1
        g[0, 0] = 6
        g[2:6, 2:4] = [[3, 4], [5, 6], [7, 8], [9, 3]]
        expected = [[4, 6], [3, 5]] if n % 2 else [[8, 3], [7, 9]]
        _require(np.array_equal(solve(g), expected), "downward proper rotation regression")
    # Every direction, maximum length, mixed lengths, duplicates, fifteen outputs.
    for config in ((a, b, r) for a in (0, 1) for b in (False, True) for r in (False, True)):
        for n, lens in ((6, [3]), (7, [4]), (29, [26]), (30, [27]), (30, [3]+[2]*14)):
            placements = ([(2, 2, lens[0])] if len(lens) == 1 else
                          [(2+3*(i//5), 2+4*(i % 5), length) for i, length in enumerate(lens)])
            blocks = [np.tile(np.array([[3], [4]], dtype=np.int8), (1, length)) for length in lens]
            pair = _render(n, placements, blocks, config, (0, 6, 1, 2))
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "boundary/configuration case")
    # Rotating a whole square preserves parity and normalized output. Recoloring
    # is equivariant, including recoloring zero, the background, and the guides.
    recolor = np.array([8, 4, 0, 9, 3, 7, 2, 1, 6, 5], dtype=np.int8)
    for pair in generate_puzzle(0)["train"]:
        for k in range(4):
            transformed = recolor[np.rot90(pair["input"], k)]
            _require(np.array_equal(solve(transformed), recolor[pair["output"]]), "rotation/recolor equivariance")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(135)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    # Reject ambiguous all-square scenes and irregular occupancy before labels.
    square = _render(8, [(2, 2, 2)], [np.full((2, 2), 3, dtype=np.int8)],
                     (0, False, False), (0, 6, 1, 2))["input"]
    hole = _render(8, [(2, 2, 4)], [np.full((2, 4), 3, dtype=np.int8)],
                   (0, False, False), (0, 6, 1, 2))["input"]
    hole[2, 3] = 0
    for invalid in (square, hole):
        try:
            solve(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid input geometry accepted")
    return {"seeds": 42, "distinct_puzzles": len(digests), "layouts": len(layouts),
            "canvas_sides": sorted(sides), "strip_lengths": sorted(lengths),
            "configurations": len(configs), "lane_populations": len(populations),
            "palettes": len(palettes), "pairs_with_duplicates": duplicates,
            "pairs_with_solid_strips": uniform, "pairs_with_mixed_length_parity": mixed_lengths}
