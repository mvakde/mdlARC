"""Selective radial echoes: a sampled micro-stencil chooses whole-tile copies.

Only Python's standard library and NumPy are used. No reference files, saved
examples, seed, or generation metadata are consulted by solve().
"""
from itertools import combinations

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and i == result[-1][1] + 1:
            result[-1] = result[-1][0], i
        else:
            result.append((i, i))
    return result


def _lattice(a):
    divider = int(a[0, 0])
    rr = _runs(np.flatnonzero(np.all(a == divider, axis=1)))
    cc = _runs(np.flatnonzero(np.all(a == divider, axis=0)))
    _require(len(rr) >= 2 and len(cc) >= 2, "missing complete dividers")
    _require(rr[0][0] == cc[0][0] == 0 and
             rr[-1][1] == a.shape[0] - 1 and cc[-1][1] == a.shape[1] - 1,
             "the lattice must have a complete outer border")
    widths = {e - s + 1 for s, e in rr + cc}
    _require(len(widths) == 1, "divider thickness must be regular")
    thickness = widths.pop()
    _require(thickness in (1, 2), "divider thickness must be 1 or 2")
    for groups in (rr, cc):
        _require(all(groups[i + 1][0] - groups[i][1] - 1 == 3
                     for i in range(len(groups) - 1)), "tiles must be 3x3")
    nr, nc = len(rr) - 1, len(cc) - 1
    _require(5 <= nr <= 7 and 5 <= nc <= 7, "coarse axes must be 5..7")
    tiles = np.empty((nr, nc, 3, 3), dtype=np.int8)
    for r in range(nr):
        for c in range(nc):
            y, x = rr[r][1] + 1, cc[c][1] + 1
            tiles[r, c] = a[y:y + 3, x:x + 3]
    _require(not np.any(tiles == divider), "divider color occurs inside a tile")
    return tiles, divider, thickness


def _analyze(input_grid):
    """Recover ordinary tile, ring, dilation, samples, and selected class."""
    a = _grid(input_grid)
    tiles, divider, thickness = _lattice(a)
    nr, nc = tiles.shape[:2]
    records = {}
    for r in range(nr):
        for c in range(nc):
            records.setdefault(tiles[r, c].tobytes(), []).append((r, c))
    ranked = sorted(records.values(), key=len, reverse=True)
    _require(len(ranked) > 1 and len(ranked[0]) > len(ranked[1]),
             "ordinary tile must be uniquely modal")
    ordinary_positions = set(ranked[0])
    ordinary = tiles[ranked[0][0]].copy()
    ordinary_colors = set(map(int, np.unique(ordinary)))
    _require(2 <= len(ordinary_colors) <= 3, "ordinary tile needs 2 or 3 colors")
    occupied = {(r, c) for r in range(nr) for c in range(nc)} - ordinary_positions
    candidates = []
    for r in range(1, nr - 1):
        for c in range(1, nc - 1):
            ring_positions = {(r + dr, c + dc) for dr, dc in DIRECTIONS}
            if (r, c) not in ordinary_positions or not ring_positions <= occupied:
                continue
            sources = tuple(tiles[r + dr, c + dc].copy() for dr, dc in DIRECTIONS)
            source_colors, counts = np.unique(np.array(sources), return_counts=True)
            carrier = int(source_colors[np.argmax(counts)])
            if int(counts.max()) <= 36 or ordinary_colors & set(map(int, source_colors)):
                continue
            if any(not np.any(t == carrier) for t in sources):
                continue
            samples = tuple(int(t[1 - dr, 1 - dc])
                            for t, (dr, dc) in zip(sources, DIRECTIONS))
            if carrier in samples:
                continue
            for scale in (2, 3):
                if not (scale <= r < nr - scale and scale <= c < nc - scale):
                    continue
                echo_positions = {(r + scale * dr, c + scale * dc): i
                                  for i, (dr, dc) in enumerate(DIRECTIONS)}
                outer = occupied - ring_positions
                if not outer or not outer <= set(echo_positions):
                    continue
                clues = frozenset(echo_positions[p] for p in outer)
                clue_colors = {samples[i] for i in clues}
                if len(clue_colors) != 1:
                    continue
                selected_color = clue_colors.pop()
                selected = frozenset(i for i, s in enumerate(samples) if s == selected_color)
                if not clues < selected or len(selected) == 8:
                    continue
                if any(not np.array_equal(tiles[p], sources[echo_positions[p]][::-1, ::-1])
                       for p in outer):
                    continue
                candidates.append(dict(center=(r, c), scale=scale, sources=sources,
                                       samples=samples, selected=selected, clues=clues,
                                       selected_color=selected_color, carrier=carrier))
    _require(len(candidates) == 1, "input must determine one ring, scale, and color class")
    info = candidates[0]
    info.update(tiles=tiles, divider=divider, thickness=thickness,
                ordinary=ordinary, lattice=(nr, nc), shape=a.shape)
    return info


def _slice(info, r, c):
    t = info["thickness"]
    y, x = t + r * (3 + t), t + c * (3 + t)
    return slice(y, y + 3), slice(x, x + 3)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill the sampled hub and complete the clue-selected outward stencil."""
    a = _grid(input_grid)
    f = _analyze(a)
    out = a.copy()
    r, c = f["center"]
    hub = np.full((3, 3), f["carrier"], dtype=np.int8)
    for (dr, dc), color in zip(DIRECTIONS, f["samples"]):
        hub[1 + dr, 1 + dc] = color
    out[_slice(f, r, c)] = hub
    for i in f["selected"] - f["clues"]:
        dr, dc = DIRECTIONS[i]
        out[_slice(f, r + f["scale"] * dr, c + f["scale"] * dc)] = (
            f["sources"][i][::-1, ::-1])
    return out


def _dihedral(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _check_pair(pair):
    """Check scalar coordinates, preservation, palette, and both ablations.

    This does not call solve or use the slice-based copying operation above.
    """
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "output shape changed")
    f = _analyze(a)
    r, c = f["center"]
    t, step = f["thickness"], 3 + f["thickness"]
    cy, cx = t + r * step, t + c * step
    assigned = np.zeros(a.shape, dtype=bool)
    for y in range(3):
        for x in range(3):
            direction = (y - 1, x - 1)
            value = (f["carrier"] if direction == (0, 0) else
                     f["samples"][DIRECTIONS.index(direction)])
            _require(int(b[cy + y, cx + x]) == value, "wrong center sample")
            assigned[cy + y, cx + x] = True
    for i in f["selected"] - f["clues"]:
        dr, dc = DIRECTIONS[i]
        sy, sx = cy + dr * step, cx + dc * step
        dy, dx = cy + f["scale"] * dr * step, cx + f["scale"] * dc * step
        for y in range(3):
            for x in range(3):
                _require(int(b[dy + y, dx + x]) == int(a[sy + 2 - y, sx + 2 - x]),
                         "wrong whole-tile half-turn")
                assigned[dy + y, dx + x] = True
    _require(np.array_equal(a[~assigned], b[~assigned]), "source, clues, or unused cells changed")
    _require(int(np.count_nonzero(a != b)) == 9 * (1 + len(f["selected"] - f["clues"])),
             "every destination must replace a full ordinary tile")
    _require(np.array_equal(np.unique(a), np.unique(b)), "palette changed")
    # Without A's class stencil, copying all eight fills a forbidden direction.
    _require(len(f["selected"]) < 8, "no selective-stencil witness")
    # Without B's spatial sampling, using the carrier for the hub loses eight cells.
    _require(all(s != f["carrier"] for s in f["samples"]), "no spatial-sampling witness")
    return f


def _training_coverage(pairs):
    """Resolve sampling and all per-direction D4 alternatives using new cells."""
    positions = [set(range(9)) for _ in DIRECTIONS]
    operations = [set(range(8)) for _ in DIRECTIONS]
    selected_counts, selected_colors, masks, scales = set(), set(), set(), set()
    for pair in pairs:
        f = _check_pair(pair)
        b = _grid(pair["output"])
        r, c = f["center"]
        hub = b[_slice(f, r, c)]
        selected_counts.add(len(f["selected"]))
        selected_colors.add(f["selected_color"])
        masks.add(tuple(sorted(f["selected"])))
        scales.add(f["scale"])
        for i, (dr, dc) in enumerate(DIRECTIONS):
            tile = f["sources"][i]
            positions[i].intersection_update(j for j in range(9)
                                            if int(tile.flat[j]) == int(hub[1 + dr, 1 + dc]))
            if i in f["selected"] - f["clues"]:
                target = b[_slice(f, r + f["scale"] * dr, c + f["scale"] * dc)]
                operations[i].intersection_update(j for j, view in enumerate(_dihedral(tile))
                                                 if np.array_equal(target, view))
    _require(all(p == {(1 - dr) * 3 + 1 - dc}
                 for p, (dr, dc) in zip(positions, DIRECTIONS)),
             "training must isolate every inward sample coordinate")
    _require(all(ops == {2} for ops in operations), "training must teach a half-turn in every direction")
    _require(3 in selected_counts and 7 in selected_counts,
             "training must expose sparse and dense selected stencils")
    _require(len(selected_colors) >= 2, "training must rule out a fixed literal target color")
    return masks, scales


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid pair counts")
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check_pair(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
    masks, scales = _training_coverage(puzzle["train"])
    for pair in puzzle["test"]:
        f = _analyze(pair["input"])
        _require(tuple(sorted(f["selected"])) not in masks, "query must transfer to an unseen stencil")
        _require(f["scale"] in scales, "query scale must have a demonstration")


def _ordinary(rng, colors):
    count = len(colors)
    labels = np.concatenate((np.arange(count), rng.integers(count, size=9 - count)))
    rng.shuffle(labels)
    return np.array(colors, dtype=np.int8)[labels.reshape(3, 3)]


def _tile(rng, carrier, sample, inward, details, witness):
    tile = np.full((3, 3), carrier, dtype=np.int8)
    tile[inward] = sample
    if witness:
        auxiliary = int(rng.choice([v for v in details if v != sample]))
        choices = []
        for p in range(9):
            if p == inward[0] * 3 + inward[1]:
                continue
            trial = tile.copy()
            trial.flat[p] = auxiliary
            if len({v.tobytes() for v in _dihedral(trial)}) == 8:
                choices.append(trial)
        return choices[int(rng.integers(len(choices)))]
    count = int(rng.choice(6, p=(.20, .22, .23, .18, .10, .07)))
    available = [p for p in range(9) if p != inward[0] * 3 + inward[1]]
    rng.shuffle(available)
    mode = int(rng.integers(3))
    if mode == 1:  # Prefer a line, then extend with scattered marks.
        axis, line = int(rng.integers(2)), int(rng.integers(3))
        available.sort(key=lambda p: (p // 3 if axis == 0 else p % 3) != line)
    elif mode == 2:  # Grow a local patch from the sample, then add marks.
        available.sort(key=lambda p: abs(p // 3 - inward[0]) + abs(p % 3 - inward[1]))
    common = int(rng.choice(details))
    for p in available[:count]:
        tile.flat[p] = int(rng.choice(details)) if mode == 0 else common
    return tile


def _layout(rng, scale):
    thickness = 2 if scale == 2 and rng.random() < .22 else 1
    minimum = 2 * scale + 1
    maximum = (30 - thickness) // (3 + thickness)
    nr, nc = map(int, rng.integers(minimum, maximum + 1, size=2))
    r, c = int(rng.integers(scale, nr - scale)), int(rng.integers(scale, nc - scale))
    return thickness, nr, nc, r, c


def _paint(ordinary, sources, carrier, samples, selected, clues, geometry, divider):
    thickness, nr, nc, r, c, scale = geometry
    step = 3 + thickness
    a = np.full((nr * step + thickness, nc * step + thickness), divider, dtype=np.int8)
    for y in range(nr):
        for x in range(nc):
            sy, sx = thickness + y * step, thickness + x * step
            a[sy:sy + 3, sx:sx + 3] = ordinary
    def put(y, x, tile):
        sy, sx = thickness + y * step, thickness + x * step
        a[sy:sy + 3, sx:sx + 3] = tile
    for i, (dr, dc) in enumerate(DIRECTIONS):
        put(r + dr, c + dc, sources[i])
        if i in clues:
            put(r + scale * dr, c + scale * dc, sources[i][::-1, ::-1])
    # Labels are independently recovered from a, including selected and scale.
    # Check construction parameters against that recovery before making output.
    f = _analyze(a)
    _require(f["selected"] == frozenset(selected) and f["scale"] == scale and
             f["carrier"] == carrier and f["samples"] == tuple(samples),
             "construction and visible recovery disagree")
    return {"input": a, "output": solve(a)}


def _new_pair(rng, selected, clues, scale, witness=False, avoid_color=None):
    palette = list(map(int, rng.permutation(10)))
    divider, carrier = palette[:2]
    ordinary_count = 2 if witness or rng.random() < .72 else 3
    ordinary_colors = palette[2:2 + ordinary_count]
    detail_colors = palette[2 + ordinary_count:]
    sample_count = 2 if witness else int(rng.integers(2, min(4, 9 - len(selected)) + 1))
    sample_colors = detail_colors[:sample_count]
    if sample_colors[0] == avoid_color:
        sample_colors[0], sample_colors[1] = sample_colors[1], sample_colors[0]
    samples = [sample_colors[0]] * 8
    other = [i for i in range(8) if i not in selected]
    labels = list(range(1, sample_count))
    labels += list(map(int, rng.integers(1, sample_count, size=len(other) - len(labels))))
    rng.shuffle(labels)
    for i, label in zip(other, labels):
        samples[i] = sample_colors[label]
    sources = [_tile(rng, carrier, samples[i], (1 - dr, 1 - dc), detail_colors, witness)
               for i, (dr, dc) in enumerate(DIRECTIONS)]
    # Permit dense individual tiles while retaining a strict global carrier majority.
    extras = [(i, p) for i, tile in enumerate(sources) for p in range(9)
              if p != (1 - DIRECTIONS[i][0]) * 3 + 1 - DIRECTIONS[i][1]
              and int(tile.flat[p]) != carrier]
    rng.shuffle(extras)
    for i, p in extras[23:]:
        sources[i].flat[p] = carrier
    ordinary = _ordinary(rng, ordinary_colors)
    geometry = (*_layout(rng, scale), scale)
    return _paint(ordinary, sources, carrier, samples, selected, clues, geometry, divider)


def _counterfactual(rng, pair):
    f = _analyze(pair["input"])
    groups = {}
    for i, value in enumerate(f["samples"]):
        groups.setdefault(value, []).append(i)
    candidates = [g for v, g in groups.items() if v != f["selected_color"] and len(g) >= 2]
    selected = candidates[int(rng.integers(len(candidates)))]
    count = int(rng.integers(1, len(selected)))
    clues = frozenset(map(int, rng.choice(selected, count, replace=False)))
    geometry = (f["thickness"], *f["lattice"], *f["center"], f["scale"])
    return _paint(f["ordinary"], f["sources"], f["carrier"], f["samples"],
                  selected, clues, geometry, f["divider"])


def _distinct(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    f = _analyze(pair["input"])
    first = [f["divider"], f["carrier"]]
    order = first + [v for v in range(10) if v not in first]
    # Ninety distinguishable divider/carrier assignments; at most 13 inputs used.
    for divider in range(10):
        for carrier in range(10):
            if divider == carrier:
                continue
            target = [divider, carrier] + [v for v in range(10) if v not in (divider, carrier)]
            mapping = np.empty(10, dtype=np.int8)
            mapping[order] = target
            a = mapping[pair["input"]]
            if _key(a) not in seen:
                return {"input": a, "output": mapping[pair["output"]]}
    raise RuntimeError("distinct palette construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and
                 isinstance(value, (int, np.integer)) and low <= int(value) <= high,
                 f"{name} must be an integer in {low}..{high}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    scale_plan = ((2, 2), (2, 3), (3, 3))[int(rng.choice(3, p=(.30, .45, .25)))]
    # Three selected directions, then seven, with two shared directions. Clue
    # choices ensure every direction is NEWLY copied in at least one anchor.
    sparse = frozenset(map(int, rng.choice(8, 3, replace=False)))
    order = list(sparse)
    rng.shuffle(order)
    u, v, w = order
    dense = frozenset(set(range(8)) - {w})
    first = _new_pair(rng, sparse, {u}, scale_plan[0], witness=True)
    first_color = _analyze(first["input"])["selected_color"]
    second = _new_pair(rng, dense, {v}, scale_plan[1], witness=True, avoid_color=first_color)
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    if n >= 3:
        third = _distinct(_counterfactual(rng, first), seen)
        puzzle["train"].append(third)
        seen.add(_key(third["input"]))
    masks = {tuple(sorted(_analyze(p["input"])["selected"])) for p in puzzle["train"]}
    for split, count in (("train", n - len(puzzle["train"])), ("test", m)):
        for _ in range(count):
            size = int(rng.integers(2, 8))
            choices = [s for s in combinations(range(8), size)
                       if split == "train" or s not in masks]
            if not choices:
                # Even ten demonstrations cannot exhaust the full mask space.
                choices = [s for k in range(2, 8) for s in combinations(range(8), k)
                           if s not in masks]
                selected_tuple = choices[int(rng.integers(len(choices)))]
                size = len(selected_tuple)
            else:
                selected_tuple = choices[int(rng.integers(len(choices)))]
            selected = frozenset(selected_tuple)
            clue_count = int(rng.integers(1, size))
            clues = frozenset(map(int, rng.choice(sorted(selected), clue_count, replace=False)))
            scale = int(rng.choice(scale_plan))
            pair = _distinct(_new_pair(rng, selected, clues, scale), seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                masks.add(tuple(sorted(selected)))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((split, tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                              for p in puzzle[split])) for split in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity, semantic, corruption, and count checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "seed diversity collapsed")
    infos = [_check_pair(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    _require({f["scale"] for f in infos} == {2, 3}, "scale coverage missing")
    _require({f["thickness"] for f in infos} == {1, 2}, "divider coverage missing")
    _require({len(f["selected"]) for f in infos} == set(range(2, 8)), "stencil density gap")
    _require(len({tuple(sorted(f["selected"])) for f in infos}) >= 60, "stencil diversity collapsed")
    _require(len({(f["shape"], f["center"], f["scale"], f["thickness"]) for f in infos}) >= 20,
             "geometry variation collapsed")
    _require(len({_key(f["ordinary"]) for f in infos}) >= 80, "ordinary pattern variation collapsed")
    _require({f["divider"] for f in infos} == set(range(10)) and
             {f["carrier"] for f in infos} == set(range(10)), "palette role coverage missing")
    _require(any(len(set(f["samples"])) >= 4 for f in infos), "multiple stencil classes missing")
    _require(any(len(f["selected"] - f["clues"]) == 1 for f in infos) and
             any(len(f["selected"] - f["clues"]) >= 5 for f in infos), "omission variation missing")
    _require(any(np.count_nonzero(tile == sample) > 1 for f in infos
                 for tile, sample in zip(f["sources"], f["samples"])), "repeated sample colors missing")
    _require(any(np.count_nonzero(tile == f["carrier"]) < 5 for f in infos
                 for tile in f["sources"]), "dense individual tiles missing")
    _require(_signature(generate_puzzle(12345, 5, 3)) ==
             _signature(generate_puzzle(12345, 5, 3)), "nondeterministic puzzle")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(4200 + 10 * n + m, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "wrong explicit counts")
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    p = puzzles[0]
    pair = p["test"][0]
    a, b = pair["input"], pair["output"]
    f = _analyze(a)
    r, c = f["center"]
    copied = next(iter(f["selected"] - f["clues"]))
    dr, dc = DIRECTIONS[copied]
    source_slice = _slice(f, r + dr, c + dc)
    dest_slice = _slice(f, r + f["scale"] * dr, c + f["scale"] * dc)
    # Scalar checks must reject errors in each semantically distinct region.
    for where in (_slice(f, r, c), source_slice, dest_slice, (slice(0, 1), slice(0, 1))):
        bad = b.copy()
        view = bad[where]
        view.flat[0] = (int(view.flat[0]) + 1) % 10
        try:
            _check_pair({"input": a, "output": bad})
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    damaged = a.copy()
    damaged[0, 0] = (int(damaged[0, 0]) + 1) % 10
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged divider accepted")
    # A compact controlled ablation: the newly copied tile differs without B's
    # half-turn, and every unselected direction differs without A's stencil.
    anchor = next(pair for pair in p["train"]
                  if len(_analyze(pair["input"])["selected"]) == 7 and
                  all(len({v.tobytes() for v in _dihedral(tile)}) == 8
                      for tile in _analyze(pair["input"])["sources"]))
    af = _analyze(anchor["input"])
    _require(all(not np.array_equal(af["sources"][i], af["sources"][i][::-1, ::-1])
                 for i in af["selected"] - af["clues"]), "half-turn ablation has no witness")
    # Count rules fail: sparse anchor selects 3 versus 5; dense selects 7 versus 1.
    sparse_f = next(_analyze(pair["input"]) for pair in p["train"]
                    if len(_analyze(pair["input"])["selected"]) == 3 and
                    len(set(_analyze(pair["input"])["samples"])) == 2)
    _require(len(set(sparse_f["samples"])) == 2, "sparse anchor must reject largest-class shortcut")
    # An admissible input beyond the sampler's four-class preference: all ten
    # colors, six sample classes, repeated samples, and symmetric source tiles.
    ordinary = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    samples = (2, 3, 4, 2, 5, 6, 7, 3)
    sources = []
    for (dr, dc), sample in zip(DIRECTIONS, samples):
        tile = np.full((3, 3), 8, dtype=np.int8)
        tile[1 - dr, 1 - dc] = tile[1 + dr, 1 + dc] = sample
        sources.append(tile)
    wide = _paint(ordinary, sources, 8, samples, {0, 3}, {0}, (1, 7, 6, 3, 2, 2), 9)
    wf = _check_pair(wide)
    _require(len(set(wf["samples"])) == 6 and len(np.unique(wide["input"])) == 10,
             "input-only solver did not support richer visible classes")
    # A conflicting class clue and a missing-clue input must both be rejected.
    conflict = wide["input"].copy()
    conflict[_slice(wf, 1, 2)] = sources[1][::-1, ::-1]
    absent = wide["input"].copy()
    absent[_slice(wf, 1, 0)] = ordinary
    for invalid in (conflict, absent):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent or absent class clues accepted")
    return {"seeds_checked": 42, "pairs_checked": len(infos),
            "stencils": len({tuple(sorted(f["selected"])) for f in infos}),
            "layouts": len({(f["shape"], f["center"], f["scale"], f["thickness"]) for f in infos})}
