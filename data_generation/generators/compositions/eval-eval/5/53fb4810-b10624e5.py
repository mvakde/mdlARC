"""Mirror-ribbon interference: exact scale, directed cyclic words, consensus.

Only NumPy and the standard library are used. No files are read or written by
generation, solving, validation, or self_test. All roles come from input pixels.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid dimensions"
    assert np.issubdtype(g.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= g) & (g <= 9)), "invalid color"
    return g.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _square(points):
    rs, cs = zip(*points)
    r, c = min(rs), min(cs)
    s = max(rs)-r+1
    assert max(cs)-c+1 == s and len(points) == s*s, "expected a solid square"
    return r, c, s


def _slice(hub, axis, sign, distance):
    r, c, s = hub
    if axis == 0:
        return (r+s-1+distance if sign > 0 else r-distance, slice(c, c+s))
    return (slice(r, r+s), c+s-1+distance if sign > 0 else c-distance)


def _distance(shape, hub, axis, sign):
    r, c, s = hub
    p = (r, c)[axis]
    return shape[axis]-p-s if sign > 0 else p


def _signs(source, destination):
    differences = [2*destination[i]+destination[2]-2*source[i]-source[2] for i in (0, 1)]
    assert all(differences), "receiver centers cannot lie on a reference axis"
    return tuple(1 if x > 0 else -1 for x in differences)


def _decode(g):
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])), "input has an empty border"
    groups = _components(g != bg)
    sources = [p for p in groups if len({int(g[x]) for x in p}) > 1]
    receivers = [p for p in groups if len({int(g[x]) for x in p}) == 1]
    assert len(sources) == 1 and 1 <= len(receivers) <= 5, "one reference and 1..5 receiver squares"
    colors = {int(g[min(p)]) for p in receivers}
    assert len(colors) == 1
    anchor = colors.pop()
    source_cells = sources[0]
    source = _square({p for p in source_cells if int(g[p]) == anchor})
    hubs = [source] + [_square(p) for p in sorted(receivers, key=min)]
    r, c, s = source
    claimed = {(y, x) for y in range(r, r+s) for x in range(c, c+s)}
    words, directions = [], []
    for axis in (0, 1):
        candidates = []
        for sign in (-1, 1):
            values = g[_slice(source, axis, sign, 1)]
            if np.all((values != bg) & (values != anchor)):
                candidates.append(sign)
        assert len(candidates) == 1, "one full-width arm per axis"
        sign = candidates[0]
        word = []
        for d in range(1, _distance(g.shape, source, axis, sign)+1):
            index = _slice(source, axis, sign, d)
            values = g[index]
            if np.all(values == bg):
                break
            assert np.all((values != bg) & (values != anchor)), "partial or malformed seed slice"
            word.append(values.copy())
            mask = np.zeros(g.shape, dtype=bool)
            mask[index] = True
            claimed.update(map(tuple, np.argwhere(mask).tolist()))
        assert word
        words.append(np.array(word, dtype=np.int8))
        directions.append(sign)
    assert claimed == source_cells, "reference is exactly a square plus two rectangular swatches"
    for hub in hubs[1:]:
        _signs(source, hub)
    return dict(shape=g.shape, background=bg, anchor=anchor, hubs=hubs,
                words=words, directions=tuple(directions), seed_cells=source_cells-{
                    (y, x) for y in range(r, r+s) for x in range(c, c+s)})


def _scale_word(word, source_side, target_side):
    """Exact rational rasterization; color boundaries may never be rounded."""
    length = len(word)*target_side
    assert length % source_side == 0, "nonintegral tile endpoint"
    expanded = np.repeat(np.repeat(word, target_side, axis=0), target_side, axis=1)
    cells = expanded.reshape(length//source_side, source_side, target_side, source_side)
    lo, hi = cells.min(axis=(1, 3)), cells.max(axis=(1, 3))
    assert np.array_equal(lo, hi), "nonintegral internal color boundary"
    return lo


def _ray_records(data, *, mirror=True, longitudinal_scale=True):
    source = data["hubs"][0]
    for i, hub in enumerate(data["hubs"]):
        signs = data["directions"] if i == 0 or not mirror else _signs(source, hub)
        for axis in (0, 1):
            word = _scale_word(data["words"][axis], source[2], hub[2])
            if not longitudinal_scale and i:
                # Deliberately wrong alternative: fit the transverse dimension
                # but retain the source's longitudinal pixel clock.
                native_length = len(data["words"][axis])
                word = word[np.arange(native_length)*len(word)//native_length]
            if mirror and i and signs[1-axis] != data["directions"][1-axis]:
                word = word[:, ::-1]
            yield i, hub, axis, signs[axis], word


def _prediction(data, *, finite=False, mirror=True, longitudinal_scale=True, collision="consensus", prefix_period=False):
    shape, bg = data["shape"], data["background"]
    counts = np.zeros(shape, dtype=np.int8)
    first = np.full(shape, -1, dtype=np.int8)
    conflict = np.zeros(shape, dtype=bool)
    axis_counts = np.zeros((2, *shape), dtype=np.int8)
    foreign_seed = False
    for i, hub, axis, sign, word in _ray_records(data, mirror=mirror, longitudinal_scale=longitudinal_scale):
        if prefix_period:
            p = next(p for p in range(1, len(word)+1)
                     if all(np.array_equal(row, word[j % p]) for j, row in enumerate(word)))
            word = word[:p]
        maximum = _distance(shape, hub, axis, sign)
        if finite:
            maximum = min(maximum, len(word))
        for d in range(1, maximum+1):
            index = _slice(hub, axis, sign, d)
            values = word[(d-1) % len(word)]
            old = first[index]
            conflict[index] |= (old >= 0) & (old != values)
            first[index] = np.where(old < 0, values, old)
            counts[index] += 1
            axis_counts[(axis, *index)] += 1
            if i and not foreign_seed:
                mask = np.zeros(shape, dtype=bool)
                mask[index] = True
                foreign_seed = any(mask[p] for p in data["seed_cells"])
    result = np.full(shape, bg, dtype=np.int8)
    visible = counts > 0
    if collision == "consensus":
        visible &= ~conflict
    elif collision == "erase_all":
        visible &= counts == 1
    else:
        assert collision == "first"
    result[visible] = first[visible]
    # Original anchors and swatches are immutable. Legal geometry ensures no
    # foreign ray ever encounters them; this restoration also defines ablations.
    for r, c, s in data["hubs"]:
        result[r:r+s, c:c+s] = data["anchor"]
    src = data["hubs"][0]
    for axis in (0, 1):
        for d, values in enumerate(data["words"][axis], 1):
            result[_slice(src, axis, data["directions"][axis], d)] = values
    return result, dict(counts=counts, conflict=conflict, axis_counts=axis_counts,
                        first=first, foreign_seed=foreign_seed)


def _geometry_ok(data):
    """No hub/seed occlusion and no parallel overlap; perpendicular crossings allowed."""
    shape = data["shape"]
    anchors = np.zeros(shape, dtype=bool)
    for r, c, s in data["hubs"]:
        if not (1 <= r and 1 <= c and r+s < shape[0] and c+s < shape[1]):
            return False
        anchors[r:r+s, c:c+s] = True
    axis_masks = np.zeros((2, *shape), dtype=bool)
    for i, hub, axis, sign, word in _ray_records(data):
        mask = np.zeros(shape, dtype=bool)
        for d in range(1, _distance(shape, hub, axis, sign)+1):
            mask[_slice(hub, axis, sign, d)] = True
        if np.any(mask & anchors) or np.any(mask & axis_masks[axis]):
            return False
        if i and any(mask[p] for p in data["seed_cells"]):
            return False
        axis_masks[axis] |= mask
    return True


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; no generator state, seeds, or examples."""
    g = _grid(input_grid)
    data = _decode(g)
    assert _geometry_ok(data), "rays must avoid anchors, foreign swatches, and parallel overlap"
    return _prediction(data)[0]


def _source_cells(hub, directions, words):
    r, c, s = hub
    cells = {(y, x) for y in range(r, r+s) for x in range(c, c+s)}
    seeds = set()
    for axis in (0, 1):
        for d in range(1, len(words[axis])+1):
            if axis == 0:
                y = r+s-1+d if directions[axis] > 0 else r-d
                seeds.update((y, x) for x in range(c, c+s))
            else:
                x = c+s-1+d if directions[axis] > 0 else c-d
                seeds.update((y, x) for y in range(r, r+s))
    return cells | seeds, seeds


def _make_input(data):
    g = np.full(data["shape"], data["background"], dtype=np.int8)
    for r, c, s in data["hubs"]:
        g[r:r+s, c:c+s] = data["anchor"]
    for axis in (0, 1):
        for d, values in enumerate(data["words"][axis], 1):
            g[_slice(data["hubs"][0], axis, data["directions"][axis], d)] = values
    return g


def _separated(hub, occupied):
    r, c, s = hub
    return not any((y, x) in occupied for y in range(r-1, r+s+1) for x in range(c-1, c+s+1))


def _interesting(data, require_witness):
    expected, info = _prediction(data)
    overlap = info["counts"] == 2
    if not np.any(overlap):
        return False
    if require_witness:
        if not (np.any(overlap & info["conflict"]) and np.any(overlap & ~info["conflict"])):
            return False
        for options in (dict(finite=True), dict(mirror=False), dict(longitudinal_scale=False), dict(prefix_period=True)):
            if np.array_equal(expected, _prediction(data, **options)[0]):
                return False
    return True


def _layout(rng, words, sides, bg, anchor, require_witness, required_flips=None):
    source_side = words[0].shape[1]
    for attempt in range(36):
        low = 25 if len(sides) >= 4 else 18
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31))) if attempt < 24 else (30, 30)
        directions = tuple(int(rng.choice([-1, 1])) for _ in range(2))
        ranges = [(1+len(words[a]) if directions[a] < 0 else 1,
                   shape[a]-source_side-1-(len(words[a]) if directions[a] > 0 else 0)) for a in (0, 1)]
        if any(lo > hi for lo, hi in ranges):
            continue
        source = (*(int(rng.integers(lo, hi+1)) for lo, hi in ranges), source_side)
        occupied, seeds = _source_cells(source, directions, words)
        data = dict(shape=shape, background=bg, anchor=anchor, hubs=[source],
                    words=words, directions=directions, seed_cells=seeds)
        for receiver_index, side in enumerate(sides):
            positions = [(r, c, side) for r in range(1, shape[0]-side) for c in range(1, shape[1]-side)]
            order = rng.permutation(len(positions))
            for index in order[:180]:
                hub = positions[int(index)]
                if not _separated(hub, occupied):
                    continue
                try:
                    signs = _signs(source, hub)
                except AssertionError:
                    continue
                if required_flips is not None:
                    desired = tuple(directions[a]*(-1 if required_flips[receiver_index][a] else 1) for a in (0, 1))
                    if signs != desired:
                        continue
                data["hubs"].append(hub)
                if _geometry_ok(data):
                    r, c, s = hub
                    occupied.update((y, x) for y in range(r, r+s) for x in range(c, c+s))
                    break
                data["hubs"].pop()
            else:
                break
        if len(data["hubs"]) == len(sides)+1 and _interesting(data, require_witness):
            return data
    return _fallback_layout(words, sides, bg, anchor, require_witness, required_flips)


def _fallback_layout(words, sides, bg, anchor, require_witness, required_flips=None):
    """Finite spacious escape layout, varying crossing phases, without RNG retries."""
    source = (14, 14, words[0].shape[1])
    _, seeds = _source_cells(source, (1, 1), words)
    for dr in range(4):
        for dc in range(4):
            positions = [(2+dr, 8+dc), (8+dr, 2+dc), (2, 19), (8, 25), (24, 3)]
            if required_flips and required_flips[0] == (False, True):
                positions = [(18+dr, 2+dc), (25, 8+dc), (2, 20)]
            data = dict(shape=(30, 30), background=bg, anchor=anchor,
                        hubs=[source]+[(r, c, s) for (r, c), s in zip(positions, sides)],
                        directions=(1, 1), words=words, seed_cells=seeds)
            original, _ = _source_cells(source, (1, 1), words)
            valid = True
            for hub in data["hubs"][1:]:
                if not _separated(hub, original):
                    valid = False
                    break
                r, c, s = hub
                original.update((y, x) for y in range(r, r+s) for x in range(c, c+s))
            if valid and _geometry_ok(data) and _interesting(data, require_witness):
                return data
    raise AssertionError("finite layout exhausted")


def _sample_words(rng, m, unit, periods, palette, witness):
    words = []
    a, b = palette[:2]
    for axis, period in enumerate(periods):
        word = rng.choice(palette, size=(period, m)).astype(np.int8)
        if witness:
            # An entire ABA word, with a visible transverse asymmetry. Repeating
            # a shorter nondividing prefix and ignoring reflection both fail.
            word = np.array([[a, b], [b, a], [a, b]] if axis == 0 else
                            [[b, a], [a, a], [b, a]], dtype=np.int8)
        else:
            style = int(rng.integers(4))
            if style == 0:
                word[:] = rng.choice(palette, size=(period, 1))
            elif style == 1:
                word[:] = rng.choice(palette, size=(1, m))
            if word.size >= 2:
                word.flat[0], word.flat[-1] = a, b
        words.append(np.repeat(np.repeat(word, unit, axis=0), unit, axis=1))
    return words


def _unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    data = _decode(pair["input"])
    old = [data["background"], data["anchor"]]
    old += [c for c in range(10) if c not in old]
    # Ninety bijections give ninety different inputs, exceeding the at most
    # thirteen already accepted pairs. No extra payloads or logs are retained.
    for bg in range(10):
        for anchor in range(10):
            if bg == anchor:
                continue
            new = [bg, anchor] + [c for c in range(10) if c not in (bg, anchor)]
            mapping = np.zeros(10, dtype=np.int8)
            mapping[old] = new
            candidate = {k: mapping[v] for k, v in pair.items()}
            if _key(candidate["input"]) not in used:
                return candidate
    raise AssertionError("finite recoloring space exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # more receivers / longer cyclic words / new exact scales
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = split == "train" and i == 0
            required_flips = ([(True, True)]*2 if witness else
                              [(False, True), (False, True), (True, False)] if split == "train" and i == 1 else None)
            colors = [int(c) for c in rng.permutation(10)]
            bg, anchor, palette = colors[0], colors[1], colors[2:2+int(rng.integers(2, 6))]
            if witness:
                m, unit, periods, sides = 2, int(rng.integers(1, 3)), (3, 3), [2, 4]
            elif split == "train" and i == 1:
                m, unit, periods = (1, 2, (3, 2)) if regime == 2 else (2, 1, (2, 3))
                sides = [2, 4, 1] if regime == 2 else [4, 2, 2]
            else:
                m = int(rng.choice([1, 2]))
                unit = int(rng.integers(1, 3))
                if regime == 1:
                    m = 2
                maximum_period = 5 if split == "test" and regime == 1 else 3
                periods = tuple(int(rng.integers(1, maximum_period+1)) for _ in range(2))
                if split == "test" and regime == 1:
                    periods = (int(rng.integers(4, 6)), periods[1])
                    unit = 1
                n = int(rng.integers(4, 6)) if split == "test" and regime == 0 else int(rng.integers(2, 4))
                if split == "test" and regime == 2:
                    m, unit = 1, int(rng.integers(2, 4))
                    periods = tuple(min(p, 3) for p in periods)
                    sides = [3, 4] + [int(rng.choice([1, 2, 3, 4])) for _ in range(n-2)]
                else:
                    choices = [2, 4] if m == 2 else [1, 2, 4] if regime == 2 and split == "train" else [1, 2, 3, 4]
                    sides = [int(rng.choice(choices)) for _ in range(n)]
            words = _sample_words(rng, m, unit, periods, palette, witness)
            if split == "test" and regime == 1:
                word = words[0]
                primitive = next(p for p in range(1, len(word)+1) if len(word) % p == 0
                                 and all(np.array_equal(row, word[j % p]) for j, row in enumerate(word)))
                if primitive != len(word):
                    word[:] = palette[0]
                    word[-1, -1] = palette[1]
            data = _layout(rng, words, sides, bg, anchor, witness, required_flips)
            pair = _unique({"input": _make_input(data), "output": _prediction(data)[0]}, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    data = _decode(inp)
    assert _geometry_ok(data), "invalid ray collision with protected material"
    expected, info = _prediction(data)
    assert np.array_equal(out, expected), "incorrect phase, scale, reflection, or interference"
    original = inp != data["background"]
    assert np.array_equal(out[original], inp[original]), "original material changed"
    assert np.any(out != inp), "there must be visible continuation"
    assert info["counts"].max() <= 2 and not info["foreign_seed"]
    overlap = info["counts"] == 2
    assert np.any(overlap), "include an interaction between receivers"
    assert np.all(info["axis_counts"][:, overlap] == 1), "every overlap is perpendicular"
    assert np.all(out[overlap & info["conflict"]] == data["background"])
    assert np.all(out[overlap & ~info["conflict"]] == info["first"][overlap & ~info["conflict"]])
    for axis in (0, 1):
        # End at the actual edge, including a partial last motif.
        assert any(_distance(inp.shape, h, axis, sign) > len(word)
                   for _, h, a, sign, word in _ray_records(data) if a == axis)
    return data, info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, witnessed, frames = set(), False, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            data, info = _analyze_pair(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            if split == "train":
                for hub in data["hubs"][1:]:
                    signs = _signs(data["hubs"][0], hub)
                    frames.add(tuple(signs[a] != data["directions"][a] for a in (0, 1)))
                overlap = info["counts"] == 2
                both = np.any(overlap & info["conflict"]) and np.any(overlap & ~info["conflict"])
                expected = np.asarray(pair["output"])
                alternatives = (dict(finite=True), dict(mirror=False), dict(longitudinal_scale=False), dict(prefix_period=True),
                                dict(collision="first"), dict(collision="erase_all"))
                witnessed |= bool(both and all(not np.array_equal(expected, _prediction(data, **a)[0]) for a in alternatives))
    assert witnessed, "training must distinguish repetition, reflection, scaling, agreement, and cancellation"
    assert frames >= {(True, True), (False, True), (True, False)}, "teach each reflection, including single-axis cases"


def review_features(pair):
    """Small pixel-derived measurements; no latent curriculum flags."""
    data, info = _analyze_pair(pair)
    b = data["hubs"][0][2]
    ratios = sorted({(h[2]//int(np.gcd(h[2], b)), b//int(np.gcd(h[2], b))) for h in data["hubs"][1:]})
    overlap = info["counts"] == 2
    period_lengths, clipped, signs = set(), 0, set()
    for _, hub, axis, sign, word in _ray_records(data):
        period_lengths.add(len(word))
        signs.add((axis, sign))
        clipped += _distance(data["shape"], hub, axis, sign) % len(word) != 0
    return dict(shape=list(data["shape"]), receivers=len(data["hubs"])-1,
                source_side=b, sides=sorted({h[2] for h in data["hubs"]}),
                roles=[data["background"], data["anchor"]], ratios=[list(x) for x in ratios],
                source_lengths=[len(w) for w in data["words"]], lengths=sorted(period_lengths),
                directions=[list(x) for x in sorted(signs)], clipped_rays=clipped,
                agreements=int(np.count_nonzero(overlap & ~info["conflict"])),
                cancellations=int(np.count_nonzero(overlap & info["conflict"])),
                crossings=int(np.count_nonzero(overlap)),
                normalized_lengths=[(len(w)//int(np.gcd(len(w), b)), b//int(np.gcd(len(w), b))) for w in data["words"]])


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded: 42 streaming puzzles, small semantic fixtures, and count edges."""
    features, signatures, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        if seed < 2:
            assert _puzzle_digest(generate_puzzle(seed)) == _puzzle_digest(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                f = review_features(pair)
                features.append(f)
                structures.add((tuple(f["shape"]), f["receivers"], tuple(f["sides"]),
                                tuple(f["source_lengths"]), f["crossings"]))
    assert len(signatures) == 42 and len(structures) >= 100
    assert len({tuple(f["shape"]) for f in features}) >= 40
    assert {f["receivers"] for f in features} == {2, 3, 4, 5}
    assert {f["source_side"] for f in features} >= {1, 2, 3, 4}
    assert {s for f in features for s in f["sides"]} >= {1, 2, 3, 4}
    assert {tuple(r) for f in features for r in f["ratios"]} >= {(1, 2), (1, 1), (2, 1), (3, 2)}
    assert {d for f in features for d in map(tuple, f["directions"])} == {(0, -1), (0, 1), (1, -1), (1, 1)}
    assert len({tuple(f["roles"]) for f in features}) >= 50
    assert any(f["clipped_rays"] for f in features)
    assert any(f["cancellations"] > 8 and f["agreements"] > 2 for f in features)
    # Exact rational scale and the ABA whole-word convention are observable
    # numeric facts, not comparisons of solve with itself.
    word = np.array([[2, 3], [3, 2], [2, 3]], dtype=np.int8)
    large = _scale_word(word, 2, 4)
    assert large.tolist() == [[2, 2, 3, 3], [2, 2, 3, 3], [3, 3, 2, 2],
                              [3, 3, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]]
    assert word[3 % len(word)].tolist() == [2, 3]
    # Hand-calculated crossing: two reflected clocks disagree in five cells
    # and agree in three. This expected rectangle is not produced by solve.
    hword = np.array([[3, 2], [2, 2], [3, 2]], dtype=np.int8)
    _, seeds = _source_cells((14, 14, 2), (1, 1), [word, hword])
    fixture = dict(shape=(30, 30), background=0, anchor=1,
                   hubs=[(14, 14, 2), (2, 8, 2), (8, 2, 4)],
                   words=[word, hword], directions=(1, 1), seed_cells=seeds)
    assert _geometry_ok(fixture)
    expected = _prediction(fixture)[0]
    assert expected[2:4, 2:6].tolist() == [[0, 0, 2, 2], [3, 0, 0, 0]]
    assert expected[0, 8:10].tolist() == [2, 3]
    assert np.array_equal(solve(_make_input(fixture)), expected)
    for unit in (1, 2):
        words = [np.repeat(np.repeat(w, unit, axis=0), unit, axis=1) for w in (word, hword)]
        for n in (2, 3, 4, 5):
            data = _fallback_layout(words, [2, 4, 2, 4, 2][:n], 0, 1, True)
            pair = {"input": _make_input(data), "output": _prediction(data)[0]}
            _analyze_pair(pair)
    try:
        _scale_word(word, 2, 3)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous half-pixel boundary accepted")
    for a, b in ((2, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(6701, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(904, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("altered output accepted")
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                structures=len(structures), shapes=len({tuple(f["shape"]) for f in features}))
