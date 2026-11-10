"""Consensus mosaics: parity-corrected, shape-indexed pixel ballots.

Only NumPy and the standard library are used. No files are read or written.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(frozenset(part))
    return result


def _normal(cells):
    cells = tuple(cells)
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _point(p, op):
    r, c = p
    if op >= 4:
        c = -c
    for _ in range(op % 4):
        r, c = c, -r
    return r, c


def _transform_map(values, op, swap=False):
    """Transform the complete object, never its two regions separately."""
    mapped = {_point(p, op): int(v) for p, v in values.items()}
    r0 = min(r for r, c in mapped)
    c0 = min(c for r, c in mapped)
    palette = sorted(set(mapped.values()))
    exchange = dict(zip(palette, reversed(palette))) if swap else {}
    return {(r - r0, c - c0): exchange.get(v, v)
            for (r, c), v in mapped.items()}


@lru_cache(maxsize=512)
def _orbit(shape):
    return tuple(_normal(_point(p, op) for p in shape) for op in range(8))


def _canonical(cells):
    return min(_orbit(_normal(cells)))


def _extent(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _array(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected an integer 1..30 by 1..30 color grid")
    return grid


def _decode(input_grid):
    grid = _array(input_grid)
    frames = []
    for color in np.unique(grid):
        for component in _components(map(tuple, np.argwhere(grid == color))):
            if len(component) < 16:
                continue
            r0, c0 = min(r for r, c in component), min(c for r, c in component)
            r1, c1 = max(r for r, c in component), max(c for r, c in component)
            box = grid[r0:r1 + 1, c0:c1 + 1]
            if min(box.shape) < 4:
                continue
            others = set(map(int, np.unique(box))) - {int(color)}
            if (len(others) == 1 and np.all(box[0] == color)
                    and np.all(box[-1] == color) and np.all(box[:, 0] == color)
                    and np.all(box[:, -1] == color)
                    and np.count_nonzero(grid == color) == len(component)):
                frames.append((int(color), others.pop(), (r0, c0, r1, c1)))
    if len(frames) != 1:
        raise ValueError("expected one connected rectangular carrier")
    carrier, bg, bbox = frames[0]
    r0, c0, r1, c1 = bbox
    box = grid[r0:r1 + 1, c0:c1 + 1]
    cavities = _components(map(tuple, np.argwhere(box == bg)))
    for cavity in cavities:
        for r, c in cavity:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if (0 <= p[0] < box.shape[0] and 0 <= p[1] < box.shape[1]
                            and box[p] == bg and p not in cavity):
                        raise ValueError("cavities must have a carrier moat")
    exterior = grid.copy()
    exterior[r0:r1 + 1, c0:c1 + 1] = bg
    keys = defaultdict(list)
    for component in _components(map(tuple, np.argwhere(exterior != bg))):
        for r, c in component:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]
                            and grid[p] != bg and p not in component):
                        raise ValueError("ballots need a background moat")
        colors = sorted({int(exterior[p]) for p in component})
        if len(colors) != 2 or carrier in colors:
            raise ValueError("each exterior ballot must have exactly two colors")
        if any(len(_components(p for p in component if int(exterior[p]) == col)) != 1
               for col in colors):
            raise ValueError("each ballot color region must be connected")
        local = _transform_map({p: int(exterior[p]) for p in component}, 0)
        shape = _canonical(component)
        keys[shape].append(local)
    cavity_shapes = {_canonical(cavity) for cavity in cavities}
    if not cavities or set(keys) != cavity_shapes:
        raise ValueError("every cavity class needs ballots, and every ballot needs a cavity")
    return grid, box, carrier, bg, bbox, cavities, keys


def _aligned_ballots(cavity, keys, policy=(False, True)):
    shape = _normal(cavity)
    cavity_parity = shape not in _orbit(_canonical(cavity))[:4]
    ballots, parities = [], []
    for key in keys:
        matches = {}
        for op in range(8):
            exchange = cavity_parity if policy == "cavity" else bool(policy[int(op >= 4)])
            fitted = _transform_map(key, op, exchange)
            if tuple(sorted(fitted)) == shape:
                signature = tuple(fitted[p] for p in shape)
                matches[signature] = op >= 4
        if len(matches) != 1:
            raise ValueError("a ballot must have one unambiguous colored fit")
        signature, parity = next(iter(matches.items()))
        ballots.append(signature)
        parities.append(parity)
    return ballots, parities


def _plurality(ballots):
    result = []
    for column in zip(*ballots):
        counts = Counter(column)
        high = max(counts.values())
        winners = [color for color, n in counts.items() if n == high]
        if len(winners) != 1:
            raise ValueError("tied cell ballot")
        result.append(winners[0])
    return tuple(result)


def _solve(input_grid, policy=(False, True), features=False):
    grid, box, carrier, bg, bbox, cavities, keys = _decode(input_grid)
    out = box.copy()
    records = []
    for cavity in cavities:
        signature = _canonical(cavity)
        ballots, parities = _aligned_ballots(cavity, keys[signature], policy)
        winner = _plurality(ballots)
        r0, c0 = min(r for r, c in cavity), min(c for r, c in cavity)
        for p, color in zip(_normal(cavity), winner):
            out[r0 + p[0], c0 + p[1]] = color
        if features:
            records.append({"shape": signature, "voters": len(ballots),
                            "parities": tuple(parities), "synthesis": winner not in ballots,
                            "source_parities": tuple(tuple(sorted(key)) not in _orbit(signature)[:4]
                                                     for key in keys[signature]),
                            "mixed": len(set(winner)) > 1,
                            "nonmajority": any(max(Counter(col).values()) * 2 <= len(col)
                                               for col in zip(*ballots))})
    if features:
        return out, {"carrier": carrier, "background": bg, "bbox": bbox,
                     "classes": len(keys), "cavities": len(cavities), "records": records}
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every ballot and output solely from the supplied input grid."""
    return _solve(input_grid)


@lru_cache(maxsize=256)
def _partitions(shape):
    """All two-connected-region bipartitions; bounded to generated <=10 cells."""
    n = len(shape)
    partitions = []
    for mask in range(1, 1 << (n - 1)):
        a = {shape[i] for i in range(n) if mask & (1 << i)}
        b = set(shape) - a
        if len(_components(a)) == len(_components(b)) == 1:
            bits = tuple(int(p in a) for p in shape)
            partitions.extend((bits, tuple(1 - x for x in bits)))
    return tuple(partitions)


def _hole_free(shape):
    h, w = _extent(shape)
    exterior = {(r, c) for r in range(-1, h + 1) for c in range(-1, w + 1)} - set(shape)
    return len(_components(exterior)) == 1


def _shape_pool(rng):
    pool = set()
    # All connected subsets of a 3x3 box give many exact-geometry alternatives
    # with the same area/box, including cycles needed for emergent mosaics.
    cells = tuple((r, c) for r in range(3) for c in range(3))
    for n in (6, 7, 8):
        for subset in combinations(cells, n):
            shape = _canonical(subset)
            if (len(_components(shape)) == 1 and len(set(_orbit(shape))) == 8
                    and _hole_free(shape)):
                pool.add(shape)
    # New shapes grow for every puzzle; these are not a fixed catalog.
    for _ in range(24):
        shape = {(0, 0)}
        for _ in range(int(rng.integers(5, 10))):
            frontier = sorted({(r + dr, c + dc) for r, c in shape for dr, dc in DIR4} - shape)
            shape.add(frontier[int(rng.integers(len(frontier)))])
        shape = _canonical(shape)
        if max(_extent(shape)) <= 5 and len(set(_orbit(shape))) == 8 and _hole_free(shape):
            pool.add(shape)
    return tuple(sorted(pool))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_ballots(rng, shape, colors, count, synthesis):
    parts = _partitions(shape)
    core_count = 5 if count == 7 else count
    for _ in range(160):
        chosen = [_choose(rng, parts) for _ in range(core_count)]
        ballots = [tuple(colors[bit] for bit in pattern) for pattern in chosen]
        if count == 7:
            pattern = _choose(rng, parts)
            ballots += [tuple(colors[2 + bit] for bit in pattern),
                        tuple(colors[3 - bit] for bit in pattern)]
        winner = _plurality(ballots)
        rich_ok = count != 7 or any(max(Counter(col).values()) * 2 <= count
                                    for col in zip(*ballots))
        if rich_ok and len(set(winner)) > 1 and (not synthesis or winner not in ballots):
            return [{p: color for p, color in zip(shape, ballot)} for ballot in ballots]
    return None


def _positions(available, shape):
    h, w = _extent(shape)
    rows, cols = available.shape[0] - h + 1, available.shape[1] - w + 1
    if rows < 1 or cols < 1:
        return np.empty((0, 2), dtype=int)
    valid = np.ones((rows, cols), dtype=bool)
    for r, c in shape:
        valid &= available[r:r + rows, c:c + cols]
    return np.argwhere(valid)


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = False


def _construct_pair(rng, shapes, teaching=False, rich=False, repeat=False):
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, accents = palette[0], palette[1], palette[2:]
    ballot_groups = []
    for i, shape in enumerate(shapes):
        count = 7 if rich and i == 0 else int(rng.choice((3, 5)))
        colors = accents[i:i + 4]
        ballots = _make_ballots(rng, shape, colors, count, synthesis=teaching and i == 0)
        if ballots is None:
            return None
        ballot_groups.append(ballots)
    cavities, keys = [], []
    for i, (shape, ballots) in enumerate(zip(shapes, ballot_groups)):
        # Two handed versions of one cavity read the same ballots differently.
        ops = [int(rng.integers(4))]
        if i == 0 and (teaching or repeat):
            ops.append(int(rng.integers(4, 8)))
        elif rng.random() < .5:
            ops[0] += 4
        for op in ops:
            fitted = [_transform_map(ballot, op, op >= 4) for ballot in ballots]
            cell_order = tuple(sorted(fitted[0]))
            winner = _plurality([tuple(ballot[p] for p in cell_order) for ballot in fitted])
            cavities.append(dict(zip(cell_order, winner)))
        for ballot in ballots:
            # Reuse one silhouette with opposite source handedness in the two
            # teaching scenes. The relative key-to-cavity fit controls swapping.
            op = (int(rng.integers(4)) + 4 * (teaching == 2) if teaching
                  else int(rng.integers(8)))
            keys.append(_transform_map(ballot, op, op >= 4))
    rng.shuffle(cavities)
    rng.shuffle(keys)
    # Rectangle shape and cavity layout are sampled independently of the split.
    for attempt in range(36):
        if attempt >= 24:
            sh, sw = 10, 16
        else:
            area = sum((_extent(c)[0] + 1) * (_extent(c)[1] + 1) for c in cavities)
            sh = int(rng.integers(7, 14))
            sw = max(7, int(np.ceil(area * 1.7 / (sh - 2))) + 2)
            sw += int(rng.integers(0, 4))
            if sw > 20:
                continue
        board_in = np.full((sh, sw), carrier, dtype=np.int8)
        board_out = board_in.copy()
        available = np.ones((sh, sw), dtype=bool)
        available[[0, -1], :] = False
        available[:, [0, -1]] = False
        failed = False
        for cavity in sorted(cavities, key=lambda c: -len(c)):
            positions = _positions(available, cavity)
            if not len(positions):
                failed = True
                break
            y, x = map(int, _choose(rng, positions))
            absolute = [(y + r, x + c) for r, c in cavity]
            for (r, c), color in cavity.items():
                board_in[y + r, x + c] = bg
                board_out[y + r, x + c] = color
            _block(available, absolute)
        if failed:
            continue
        height = 30 if attempt >= 12 else int(rng.integers(max(sh + 6, 23), 31))
        width = 30 if attempt >= 12 else int(rng.integers(max(sw + 5, 23), 31))
        y = int(rng.integers(height - sh + 1))
        x = int(rng.integers(width - sw + 1))
        grid = np.full((height, width), bg, dtype=np.int8)
        grid[y:y + sh, x:x + sw] = board_in
        available = np.ones(grid.shape, dtype=bool)
        _block(available, [(r, c) for r in range(y, y + sh) for c in range(x, x + sw)])
        for key in sorted(keys, key=lambda k: -max(_extent(k))):
            positions = _positions(available, key)
            if not len(positions):
                failed = True
                break
            ky, kx = map(int, _choose(rng, positions))
            for (r, c), color in key.items():
                grid[ky + r, kx + c] = color
            _block(available, [(ky + r, kx + c) for r, c in key])
        if not failed:
            # A global rotation is geometry variation, not an extra color swap.
            turns = int(rng.integers(4))
            return {"input": np.rot90(grid, turns).copy(),
                    "output": np.rot90(board_out, turns).copy()}
    return None


def _legal_counts(num_train, num_test):
    for n in (num_train, num_test):
        if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = _shape_pool(rng)
    # Compact same-area/same-box classes make exact shape discrimination visible.
    compact = [s for s in pool if _extent(s) == (3, 3) and len(s) <= 8]
    anchor = _choose(rng, compact)
    companions = [s for s in compact if len(s) == len(anchor) and s != anchor]
    if not companions:
        anchor = next(s for s in compact if sum(len(t) == len(s) for t in compact) >= 2)
        companions = [s for s in compact if len(s) == len(anchor) and s != anchor]
    companion = _choose(rng, companions)
    puzzle, seen, train_shapes = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            teaching = (i + 1) if split == "train" and i < 2 else 0
            if teaching:
                shapes = [anchor, companion] if i == 0 else [anchor]
            elif split == "test":
                unseen = [s for s in pool if s not in train_shapes]
                # Novel exact geometry, often combined with a familiar class.
                shapes = [_choose(rng, unseen)]
                if rng.random() < .65:
                    shapes.append(_choose(rng, sorted(train_shapes)))
            else:
                indices = rng.choice(len(pool), int(rng.integers(1, 3)), replace=False)
                shapes = [pool[int(j)] for j in indices]
            for retry in range(28):
                pair = _construct_pair(rng, shapes, teaching=teaching,
                                       rich=(i == 1 or rng.random() < .3),
                                       repeat=split == "test" or rng.random() < .25)
                if pair is None:
                    continue
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_shapes.update(shapes)
                    break
            else:
                raise RuntimeError("bounded pair construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _different_policy(pair, policy):
    try:
        return not np.array_equal(_solve(pair["input"], policy), pair["output"])
    except ValueError:
        return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _legal_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            signature = inp.shape, inp.astype(np.int8).tobytes()
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            predicted, data = _solve(inp, features=True)
            assert np.array_equal(predicted, out), "incorrect mosaic or crop"
            grid, box, carrier, bg, bbox, cavities, keys = _decode(inp)
            assert np.all(out[box == carrier] == carrier), "carrier changed"
            assert all(len(set(_orbit(_normal(c)))) == 8 for c in cavities)
            assert all(bg not in key.values() and carrier not in key.values()
                       for group in keys.values() for key in group)
            assert all(record["mixed"] for record in data["records"])
            records[split].append(data)
    # Both mechanisms alter visible answers even at the legal 2/1 minimum.
    assert any(r["synthesis"] for d in records["train"] for r in d["records"]), "no consensus evidence"
    for policy in ((False, False), (True, True), (True, False)):
        assert any(_different_policy(pair, policy) for pair in puzzle["train"]), "parity rule not taught"
    assert any(_different_policy(pair, "cavity") for pair in puzzle["train"]), "absolute handedness shortcut survives"
    source_history = defaultdict(set)
    for data in records["train"]:
        for record in data["records"]:
            source_history[record["shape"]].update(record["source_parities"])
    assert any(parities == {False, True} for parities in source_history.values()), "relative handedness not taught"
    assert any(r["nonmajority"] for d in records["train"] for r in d["records"])
    train_shapes = {r["shape"] for d in records["train"] for r in d["records"]}
    assert all(any(r["shape"] not in train_shapes for r in d["records"])
               for d in records["test"]), "test must transfer to unseen geometry"


def self_test():
    """42 seeds, in memory; no traces, files, worker pool, or unbounded retries."""
    shapes, dimensions, crops, ballots, backgrounds, palettes, signatures = set(), set(), set(), set(), set(), set(), set()
    synthesis = nonmajority = reflected = preserved = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        test = puzzle["test"][0]
        # Scene reflection preserves RELATIVE transformation parity; arbitrary
        # color relabeling must also preserve the same geometric rule.
        assert np.array_equal(solve(np.fliplr(test["input"])), np.fliplr(test["output"]))
        permutation = np.random.default_rng(seed + 1000).permutation(10)
        assert np.array_equal(solve(permutation[test["input"]]), permutation[test["output"]])
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                prediction, data = _solve(pair["input"], features=True)
                assert np.array_equal(prediction, pair["output"])
                dimensions.add(pair["input"].shape)
                crops.add(pair["output"].shape)
                backgrounds.add(data["background"])
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                for r in data["records"]:
                    shapes.add(r["shape"])
                    ballots.add(r["voters"])
                    synthesis += r["synthesis"]
                    nonmajority += r["nonmajority"]
                    reflected += any(r["parities"])
                    preserved += not all(r["parities"])
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(dimensions) >= 25 and len(crops) >= 35
    assert len(backgrounds) == 10 and len(palettes) >= 20
    assert ballots == {3, 5, 7}
    assert min(synthesis, nonmajority, reflected, preserved) >= 42
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(42, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    try:
        _plurality([(1, 2), (2, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("tied ballot accepted")
