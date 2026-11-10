"""Clockwork refraction: a striped surface controls four-way trace geometry.

All semantic choices needed by solve are visible in its input. Zero is a
barrier; four other colors are a pair-local base/trace palette. See task_spec.md.
Only standard-library modules and NumPy are used. No parent code is imported.
"""
import numpy as np


# NE, SE, SW, NW: a cardinal direction followed by its clockwise neighbor.
DIRECTIONS = (((-1, 0), (0, 1)), ((0, 1), (1, 0)),
              ((1, 0), (0, -1)), ((0, -1), (-1, 0)))


def walk(mask, seeds, blockers=(), *, run=2, clockwise=True, reset=True,
         medium="terrain"):
    """Trace original sources against an immutable surface and barrier set.

    Optional arguments are used only for counterfactual validation; the family
    always uses run=2, clockwise=True, reset=True and medium='terrain'.
    Each branch keeps its own phase, even when branches meet.
    """
    blockers = set(blockers)
    h, w = mask.shape
    paths = []
    for source, seed in enumerate(seeds):
        for heading, (a, b) in enumerate(DIRECTIONS):
            if not clockwise:
                a, b = b, a
            p, phase = seed, 0
            cells, departures = [], []
            for _ in range(h + w + 1):
                stripe = bool(mask[p])
                stairs = medium == "stairs" or (medium == "terrain" and stripe)
                step = (a if phase < run else b) if stairs else (
                    a[0] + b[0], a[1] + b[1])
                q = (p[0] + step[0], p[1] + step[1])
                if not (0 <= q[0] < h and 0 <= q[1] < w):
                    stop = "edge"
                    break
                if q in blockers:
                    stop = "barrier"
                    break
                cells.append(q)
                departures.append((stripe, phase, step))
                if stairs:
                    phase = (phase + 1) % (2 * run)
                # Departure controls this move. Arriving on field clears the
                # clock, so the next stripe visit begins with the first run.
                if reset and medium == "terrain" and not bool(mask[q]):
                    phase = 0
                p = q
            else:
                raise AssertionError("monotone path failed to leave canvas")
            paths.append(dict(source=source, heading=heading, cells=cells,
                              departures=departures, stop=stop, next=q))
    return paths


def union(shape, seeds, paths):
    result = np.zeros(shape, bool)
    for p in seeds:
        result[p] = True
    for path in paths:
        for p in path["cells"]:
            result[p] = True
    return result


def metrics(mask, seeds, blockers):
    paths = walk(mask, seeds, blockers)
    traced = union(mask.shape, seeds, paths)
    cycles, longest, enter_exit = set(), 0, False
    visits = {}
    for path in paths:
        streak = 0
        roles = [bool(mask[seeds[path["source"]]])]
        roles += [bool(mask[p]) for p in path["cells"]]
        compressed = [v for i, v in enumerate(roles) if i == 0 or v != roles[i - 1]]
        enter_exit |= any(compressed[i:i + 3] == [False, True, False]
                          for i in range(len(compressed) - 2))
        for stripe, phase, step in path["departures"]:
            streak = streak + 1 if stripe else 0
            longest = max(longest, streak)
            if streak >= 4:
                cycles.add(path["heading"])
        for p in path["cells"]:
            if p not in seeds:
                visits.setdefault(p, set()).add(path["source"])
    def changed(**kwargs):
        other = union(mask.shape, seeds, walk(mask, seeds, blockers, **kwargs))
        return bool(np.any((other != traced) & ~barrier_mask))
    barrier_mask = np.zeros(mask.shape, bool)
    for p in blockers:
        barrier_mask[p] = True
    free = union(mask.shape, seeds, walk(mask, seeds))
    local_color = any(all(bool(mask[seeds[s]]) != bool(mask[p]) for s in owners)
                      for p, owners in visits.items())
    return dict(paths=paths, traced=traced, cycles=cycles, longest=longest,
                enter_exit=enter_exit,
                overlap=sum(len(s) > 1 for s in visits.values()),
                local_color=local_color,
                stopped=any(p["stop"] == "barrier" for p in paths),
                immediate=any(p["stop"] == "barrier" and not p["cells"] for p in paths),
                blocking=bool(np.any((free != traced) & ~barrier_mask)),
                diagonal=changed(medium="diagonal"),
                always_stairs=changed(medium="stairs"),
                run_one=changed(run=1), reverse=changed(clockwise=False),
                reset=changed(reset=False))


def groups(axis):
    starts = np.flatnonzero(axis & ~np.r_[False, axis[:-1]])
    ends = np.flatnonzero(axis & ~np.r_[axis[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def axis_bands(rng, length, count, *, wide=False, demo=False):
    axis = np.zeros(length, bool)
    if count == 0:
        return axis
    if demo:
        start = int(rng.integers(3, length - 6))
        axis[start:start + 4] = True
        return axis
    budget = (length - 1) // 2
    width = int(rng.integers(5, 7)) if wide else 1
    count = min(count, budget - width + 1)
    widths = [width] + [1] * (count - 1)
    for _ in range(budget - sum(widths)):
        choices = [i for i, n in enumerate(widths) if n < (6 if wide and i == 0 else 4)]
        if choices and rng.random() < 0.75:
            widths[int(rng.choice(choices))] += 1
    rng.shuffle(widths)
    free = length - sum(widths) - count + 1
    cuts = [0] + sorted(int(x) for x in rng.integers(0, free + 1, size=count)) + [free]
    gaps = np.diff(cuts)
    pos = int(gaps[0])
    for i, width in enumerate(widths):
        axis[pos:pos + width] = True
        pos += width + 1 + int(gaps[i + 1])
    return axis


def layout(rng, mode):
    low, high = (21, 28) if mode == "demo" else (18, 28) if mode == "overlap" else (
        (14, 31) if mode == "test" else (10, 31))
    h, w = map(int, rng.integers(low, high, size=2))
    if rng.random() < 0.2:
        w = h
    orientation = int(rng.integers(2)) if mode == "demo" else (
        2 if mode == "overlap" else int(rng.integers(3)))
    nr = int(rng.integers(1, 4)) if orientation in (0, 2) else 0
    nc = int(rng.integers(1, 4)) if orientation in (1, 2) else 0
    if mode in ("demo", "overlap"):
        nr, nc = int(nr > 0), int(nc > 0)
    wide_axis = int(rng.choice(([0] if nr else []) + ([1] if nc else [])))
    rows = axis_bands(rng, h, nr, wide=mode == "test" and wide_axis == 0,
                      demo=mode == "demo")
    cols = axis_bands(rng, w, nc, wide=mode == "test" and wide_axis == 1,
                      demo=mode == "demo")
    return rows, cols, rows[:, None] | cols[None, :]


def sample_seeds(rng, mask, count, mode):
    h, w = mask.shape
    allowed = np.ones(mask.shape, bool)
    if mode == "demo":
        allowed[:2] = allowed[-2:] = False
        allowed[:, :2] = allowed[:, -2:] = False
    pools = [list(map(tuple, np.argwhere(allowed & (mask == role)).tolist()))
             for role in (False, True)]
    seeds = [pool[int(rng.integers(len(pool)))] for pool in pools]
    while len(seeds) < count:
        candidates = []
        if mode == "overlap" or rng.random() < 0.6:
            candidates = list({p for path in walk(mask, seeds)
                               for p in path["cells"][2:-2] if p not in seeds})
            candidates.sort()
        if not candidates:
            candidates = [tuple(p) for p in np.argwhere(np.ones(mask.shape, bool)).tolist()
                          if tuple(p) not in seeds]
        seeds.append(candidates[int(rng.integers(len(candidates)))])
    return seeds


def sample_blockers(rng, mask, seeds, mode):
    paths = walk(mask, seeds)
    eligible = [p["cells"] for p in paths if len(p["cells"]) >= 6]
    if not eligible:
        raise ValueError("no path for a witnessed stop")
    rng.shuffle(eligible)
    blockers = set()
    for cells in eligible:
        index = int(rng.integers(2, len(cells) - 2))
        if cells[index] not in seeds and cells[index + 2] not in seeds:
            blockers.update((cells[index], cells[index + 2]))
            break
    if mode == "overlap":
        starts = [p["cells"][0] for p in paths if p["cells"] and p["cells"][0] not in seeds]
        blockers.add(starts[int(rng.integers(len(starts)))])
    target = max(len(blockers), int(round(mask.size * rng.uniform(0.008, 0.06))))
    candidates = [tuple(p) for p in np.argwhere(np.ones(mask.shape, bool)).tolist()
                  if tuple(p) not in seeds and tuple(p) not in blockers]
    rng.shuffle(candidates)
    for p in candidates:
        if len(blockers) >= target:
            break
        blockers.add(p)
        if rng.random() < 0.3:
            dr, dc = DIRECTIONS[int(rng.integers(4))][0]
            q = (p[0] + dr, p[1] + dc)
            if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                    and q not in seeds):
                blockers.add(q)
    return blockers


def decode(grid):
    grid = np.asarray(grid)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid")
    values, counts = np.unique(grid[grid != 0], return_counts=True)
    if len(values) != 4 or not np.any(grid == 0):
        raise ValueError("four nonzero colors and zero barriers are required")
    order = np.argsort(-counts)
    if counts[order[1]] <= counts[order[2]]:
        raise ValueError("base colors must be more frequent than seed colors")
    bases = [int(values[i]) for i in order[:2]]
    traces = [int(values[i]) for i in order[2:]]
    candidates = []
    for field, stripe in (bases, bases[::-1]):
        seen_field = grid == field
        rows, cols = ~seen_field.any(axis=1), ~seen_field.any(axis=0)
        if (not (rows.any() or cols.any()) or 2 * rows.sum() >= len(rows)
                or 2 * cols.sum() >= len(cols)):
            continue
        if np.any(np.sum(seen_field, axis=1)[~rows] < 2) or np.any(
                np.sum(seen_field, axis=0)[~cols] < 2):
            continue
        if len(groups(rows)) > 3 or len(groups(cols)) > 3 or any(
                b - a + 1 > 6 for a, b in groups(rows) + groups(cols)):
            continue
        mask = rows[:, None] | cols[None, :]
        base = np.where(mask, stripe, field).astype(np.int8)
        intact = (grid == field) | (grid == stripe)
        if not np.array_equal(grid[intact], base[intact]):
            continue
        mapping = {}
        for trace in traces:
            roles = np.unique(mask[grid == trace])
            if len(roles) != 1:
                break
            mapping[bool(roles[0])] = trace
        if set(mapping) != {False, True}:
            continue
        seeds = list(map(tuple, np.argwhere(np.isin(grid, traces)).tolist()))
        if not 2 <= len(seeds) <= 6:
            continue
        blockers = set(map(tuple, np.argwhere(grid == 0).tolist()))
        candidates.append(dict(mask=mask, base=base, mapping=mapping,
                               seeds=seeds, blockers=blockers, rows=rows, cols=cols,
                               colors=(field, stripe, mapping[False], mapping[True])))
    if len(candidates) != 1:
        raise ValueError("the surface and local color mapping must be unique")
    return candidates[0]


def draw(mask, seeds, blockers, colors):
    field, stripe, ft, st = colors
    inp = np.where(mask, stripe, field).astype(np.int8)
    for p in seeds:
        inp[p] = st if mask[p] else ft
    for p in blockers:
        inp[p] = 0
    traced = union(mask.shape, seeds, walk(mask, seeds, blockers))
    out = inp.copy()
    for role, trace in ((False, ft), (True, st)):
        out[traced & (mask == role)] = trace
    return dict(input=inp, output=out)


def make_pair(rng, mode, colors):
    for attempt in range(600):
        rows, cols, mask = layout(rng, mode)
        count = 2 if mode == "demo" else 3 if mode == "overlap" else int(
            rng.integers(2, 7 if mode == "test" else 6))
        seeds = sample_seeds(rng, mask, count, mode)
        try:
            blockers = sample_blockers(rng, mask, seeds, mode)
            pair = draw(mask, seeds, blockers, colors)
            info = decode(pair["input"])
            if not np.array_equal(info["mask"], mask) or info["colors"] != colors:
                continue
        except ValueError:
            continue
        m = metrics(mask, seeds, blockers)
        ok = all(m[k] for k in ("blocking", "diagonal", "always_stairs", "run_one", "reverse", "local_color"))
        changed = pair["input"] != pair["output"]
        ok &= all(np.any(changed & (pair["output"] == c)) for c in colors[2:])
        if mode == "demo":
            ok &= m["cycles"] == set(range(4)) and m["longest"] >= 8 and m["enter_exit"]
        if mode == "overlap":
            ok &= m["overlap"] >= 2 and m["immediate"] and m["reset"]
        if ok:
            return pair, m, attempt + 1
    raise RuntimeError("bounded construction exhausted for " + mode)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the underlay, freeze the original sources, then trace their union."""
    grid = np.asarray(input_grid)
    info = decode(grid)
    paths = walk(info["mask"], info["seeds"], info["blockers"])
    traced = union(grid.shape, info["seeds"], paths)
    output = grid.copy()
    for role in (False, True):
        output[traced & (info["mask"] == role)] = info["mapping"][role]
    return output


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate deterministic, distinct examples with a held-out band-width shift."""
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    first_colors = tuple(int(x) for x in rng.choice(np.arange(1, 10), 4, replace=False))
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == 0:
                mode, colors = "demo", first_colors
            elif split == "train" and index == 1:
                f, s, ft, st = first_colors
                # Cycle all four roles. The old stripe remains a base but
                # now maps to the old field color, directly refuting a fixed
                # numeric base-to-trace mapping across examples.
                mode, colors = "overlap", (s, st, f, ft)
            else:
                mode = "test" if split == "test" else "random"
                colors = tuple(int(x) for x in rng.choice(np.arange(1, 10), 4, replace=False))
            for _ in range(20):
                pair, _, _ = make_pair(rng, mode, colors)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not make a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    for grid in (inp, out):
        assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    info = decode(inp)
    mask, seeds, blockers = info["mask"], info["seeds"], info["blockers"]
    m = metrics(mask, seeds, blockers)
    expected_change = m["traced"].copy()
    for p in seeds:
        expected_change[p] = False
    assert np.array_equal(inp != out, expected_change)
    assert np.array_equal(out[inp == 0], inp[inp == 0])
    assert np.array_equal(out[~m["traced"]], inp[~m["traced"]])
    for role in (False, True):
        assert np.all(out[m["traced"] & (mask == role)] == info["mapping"][role])
        assert np.any(expected_change & (mask == role))
    assert np.array_equal(solve(inp.copy()), out)

    # Inspect every branch, including branches later hidden by union overlap.
    for path in m["paths"]:
        a, b = DIRECTIONS[path["heading"]]
        previous = seeds[path["source"]]
        assert len(set(path["cells"])) == len(path["cells"])
        assert len(path["cells"]) <= sum(inp.shape)
        for p, (stripe, phase, step) in zip(path["cells"], path["departures"]):
            assert p not in blockers and (p[0] - previous[0], p[1] - previous[1]) == step
            assert stripe == bool(mask[previous])
            assert step == ((a if phase < 2 else b) if stripe else
                            (a[0] + b[0], a[1] + b[1]))
            previous = p
        q = path["next"]
        if path["stop"] == "barrier":
            assert q in blockers
        else:
            assert not (0 <= q[0] < inp.shape[0] and 0 <= q[1] < inp.shape[1])
    # These witnesses stop either inherited contribution from being cosmetic.
    assert all(m[k] for k in ("blocking", "diagonal", "always_stairs", "run_one", "reverse", "local_color"))
    return info, m


def validate_generated_puzzle(puzzle):
    """Check schema, recoverability, path invariants, ablations and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            info, m = _check_pair(pair)
            infos[split].append((info, m))
            widths = [b - a + 1 for a, b in groups(info["rows"]) + groups(info["cols"])]
            if split == "train":
                assert max(widths) <= 4
            else:
                assert max(widths) >= 5
    first, fm = infos["train"][0]
    second, sm = infos["train"][1]
    assert bool(first["rows"].any()) != bool(first["cols"].any())
    assert second["rows"].any() and second["cols"].any()
    f, s, ft, st = first["colors"]
    assert second["colors"] == (s, st, f, ft)
    assert fm["cycles"] == set(range(4)) and fm["longest"] >= 8 and fm["enter_exit"]
    assert sm["overlap"] >= 2 and sm["immediate"] and sm["reset"]
    assert len(first["seeds"]) == 2 and len(second["seeds"]) == 3


def _mechanics_test():
    # A tiny input-only model exercises precise departure timing and phase reset.
    mask = np.zeros((16, 18), bool)
    mask[7:9] = True
    mask[:, 6:9] = True
    seeds = [(3, 4), (7, 6)]
    blockers = {(0, 0), (2, 5), (1, 6), (0, 7)}
    pair = draw(mask, seeds, blockers, (1, 2, 3, 4))
    assert np.array_equal(solve(pair["input"]), pair["output"])
    immediate = walk(mask, seeds, blockers)[0]
    assert immediate["cells"] == [] and immediate["next"] == (2, 5)
    # With the destination clear, two flank barriers do not close a diagonal.
    clear = np.zeros((8, 8), bool)
    path = walk(clear, [(4, 3)], {(3, 3), (4, 4)})[0]
    assert path["cells"][0] == (3, 4)
    stripe = np.ones((10, 10), bool)
    path = walk(stripe, [(6, 1)])[0]
    assert path["cells"][:8] == [(5, 1), (4, 1), (4, 2), (4, 3),
                                  (3, 3), (2, 3), (2, 4), (2, 5)]
    # An intersection stays stripe; there is no reset at an internal band edge.
    path = walk(mask, [(10, 6)])[0]
    assert path["cells"][:4] == [(9, 6), (8, 6), (8, 7), (8, 8)]
    # A valid union can completely cover a base color. The surface still
    # comes from the input; retaining that base color in the output is not a
    # transformation invariant.
    thin = np.zeros((10, 10), bool)
    thin[0] = True
    sources = [(0, 0), (0, 3), (0, 6), (0, 9), (5, 5)]
    pair = draw(thin, sources, {(7, 7)}, (1, 2, 3, 4))
    assert np.all(pair["output"][0] == 4) and not np.any(pair["output"] == 2)
    _check_pair(pair)


def self_test():
    """Bounded 42-seed structural survey plus explicit count/mechanics checks.

    The returned dictionary is descriptive evidence, not independent approval.
    """
    _mechanics_test()
    shapes, palettes, widths, orientations, counts, signatures = (set() for _ in range(6))
    normalized_inputs, barriers, positions = set(), set(), set()
    witnesses = {k: 0 for k in ("blocking", "diagonal", "always_stairs", "run_one",
                                "reverse", "reset", "immediate", "enter_exit", "local_color")}
    edge_bands = later_barriers = smaller_tests = larger_tests = 0
    max_overlap = max_streak = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        train_areas = [np.asarray(p["input"]).size for p in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = decode(grid)
                m = metrics(info["mask"], info["seeds"], info["blockers"])
                shapes.add(grid.shape)
                palettes.add(info["colors"])
                counts.add(len(info["seeds"]))
                barriers.add(len(info["blockers"]))
                positions.update((grid.shape, p) for p in info["seeds"])
                orientation = (bool(info["rows"].any()), bool(info["cols"].any()))
                orientations.add(orientation)
                for axis in (info["rows"], info["cols"]):
                    intervals = groups(axis)
                    widths.update(b - a + 1 for a, b in intervals)
                    edge_bands += any(a == 0 or b == len(axis) - 1 for a, b in intervals)
                renaming = {0: 0}
                normalized_inputs.add((grid.shape, tuple(renaming.setdefault(int(c), len(renaming))
                                                        for c in grid.flat)))
                for k in witnesses:
                    witnesses[k] += bool(m[k])
                max_overlap = max(max_overlap, m["overlap"])
                max_streak = max(max_streak, m["longest"])
                for path in walk(info["mask"], info["seeds"]):
                    indices = [i for i, p in enumerate(path["cells"]) if p in info["blockers"]]
                    later_barriers += len(indices) >= 2
                if split == "test":
                    smaller_tests += grid.size < max(train_areas)
                    larger_tests += grid.size > max(train_areas)
    assert len(signatures) == 42 and len(shapes) >= 65
    assert len(normalized_inputs) == 210 and len(palettes) >= 100
    assert orientations == {(True, False), (False, True), (True, True)}
    assert widths == set(range(1, 7)) and counts == set(range(2, 7))
    assert len(barriers) >= 20 and len(positions) >= 300
    assert all(witnesses.values()) and edge_bands and later_barriers
    assert smaller_tests and larger_tests and max_overlap >= 10 and max_streak >= 12
    legal_count_checks = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            legal_count_checks += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(999, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=210, shapes=len(shapes), palettes=len(palettes),
                normalized_inputs=len(normalized_inputs), seed_counts=sorted(counts),
                band_widths=sorted(widths), edge_band_axes=edge_bands,
                multi_barrier_routes=later_barriers, smaller_tests=smaller_tests,
                larger_tests=larger_tests, max_overlap=max_overlap, max_stripe_streak=max_streak,
                witnesses=witnesses, legal_count_checks=legal_count_checks)
