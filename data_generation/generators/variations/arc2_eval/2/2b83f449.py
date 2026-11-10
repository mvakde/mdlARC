import numpy as np


_PAL_KEYS = ("background", "track", "bead", "input_divider", "output_divider")
_DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))

# Literal source regressions/exclusions, never used to sample a layout.
_SOURCE_CASES = ((('3888888088888888883', '0007770007770000000', '3888888888888888883', '0000077700007770000', '3888888888888888883', '0077700777000077700', '3888888888888888883', '0000777000777000000', '3888888888888880883', '0000000777000000000', '3888888888888888883', '0000777000007770000', '8808888808888888888'), ('8888688088688888888', '0008680008680000000', '3388686888688688888', '0000086800008680000', '8886886868888686888', '0086800868000086800', '3386868868868886833', '0000868000868000000', '3888868868868830883', '0000000868000000000', '8888868868888688888', '0000868000008680000', '8803868803888688883')), (('3888888888888883', '0007770000000000', '3888888888888883', '0000000000077700', '3888888888888883', '0000077700000000', '3888888888888883', '0077700000777000', '3888888888888883', '0000007770000000', '8888888888888888'), ('8888688888888888', '0008680000000000', '3388688888886888', '0000000000086800', '8888886888886833', '0000086800000000', '8886886888868888', '0086800000868000', '3386888688868833', '0000008680000000', '3888888688888883')), (('38888888888888883', '00000000077700000', '38888888888888883', '00000000000077700', '38888880888888883', '00000000000000000', '38888888888888883', '00000077700000000', '38888888888888883', '00077700000000000', '38888888880888883', '00000000000000000', '38888888888888883', '00000000077700000', '38888888888888883', '00000077700000000', '38888888888888883', '00777000000777000', '88888880888888888'), ('88888888886888888', '00000000086800000', '33888888886886888', '00000000000086800', '38888880388886833', '00000000000000000', '88888886888888888', '00000086800000000', '88886886888888833', '00086800000000000', '33886888830888883', '00000000000000000', '88888888886888888', '00000000086800000', '88888886886888833', '00000086800000000', '88868886888868888', '00868000000868000', '38868830388868883')))


def _source_pairs():
    return [(np.array([[int(c) for c in row] for row in a], dtype=np.int8),
             np.array([[int(c) for c in row] for row in b], dtype=np.int8))
            for a, b in _SOURCE_CASES]


def _palette_tuple(palette):
    return tuple(int(palette[key]) for key in _PAL_KEYS)


def _parse_input(inp, palette):
    """Parse a cropped horizontal network, accepting agreeing local writes."""
    h, w = inp.shape
    bg, track, bead, divider, turned = _palette_tuple(palette)
    assert h >= 3 and h % 2 == 1 and w >= 7
    assert set(map(int, np.unique(inp))) <= {bg, track, bead, divider}
    assert np.all(np.isin(inp[::2], [bg, track, bead]))
    assert np.all(np.isin(inp[1::2], [bg, divider]))
    assert np.all(np.isin(inp[::2, [0, w - 1]], [track, bead]))
    centers, per_separator, breaks, writes = [], [], [], {}
    overlaps = 0
    for row in range(1, h, 2):
        xs = np.flatnonzero(inp[row] == divider).tolist()
        starts = [x for x in xs if x - 1 not in xs]
        row_centers = []
        for start in starts:
            assert start >= 1 and start + 3 < w
            assert xs[xs.index(start):xs.index(start) + 3] == list(range(start, start + 3))
            assert start + 3 not in xs
            c = start + 1
            assert np.all(np.isin(inp[row - 1, c - 1:c + 2], [track, bead]))
            assert np.all(np.isin(inp[row + 1, c - 1:c + 2], [track, bead]))
            assert inp[row - 1, c] == track and inp[row + 1, c] == track
            for r in range(row - 1, row + 2):
                for x in range(c - 1, c + 2):
                    value = turned if x == c else track
                    if (r, x) in writes:
                        assert writes[r, x] == value
                        overlaps += 1
                    writes[r, x] = value
            centers.append((row, c))
            row_centers.append(c)
        per_separator.append(row_centers)
    assert centers
    for row in range(0, h, 2):
        xs = np.flatnonzero(inp[row] == bg).tolist()
        assert all(2 <= x <= w - 3 for x in xs)
        assert all(b - a >= 3 for a, b in zip(xs, xs[1:]))
        breaks.extend((row, x) for x in xs)
    structural = inp.copy()
    structural[structural == bead] = track
    for cell, value in writes.items():
        structural[cell] = value
    return structural, centers, breaks, per_separator, overlaps


def _neighbors(
    mask: np.ndarray,
    row: int,
    col: int,
) -> list[tuple[int, int]]:
    height, width = mask.shape
    result = []

    for dr, dc in _DIRS:
        rr = row + dr
        cc = col + dc
        if (
            0 <= rr < height
            and 0 <= cc < width
            and bool(mask[rr, cc])
        ):
            result.append((rr, cc))

    return result


def _trace_paths(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    paths = []

    for start_row in range(height):
        for start_col in range(width):
            if (
                not bool(mask[start_row, start_col])
                or bool(seen[start_row, start_col])
            ):
                continue

            component = []
            queue = [(start_row, start_col)]
            seen[start_row, start_col] = True
            queue_index = 0

            while queue_index < len(queue):
                row, col = queue[queue_index]
                queue_index += 1
                component.append((row, col))

                for rr, cc in _neighbors(mask, row, col):
                    if not bool(seen[rr, cc]):
                        seen[rr, cc] = True
                        queue.append((rr, cc))

            endpoints = []

            for row, col in component:
                degree = len(_neighbors(mask, row, col))
                assert degree in (1, 2)
                if degree == 1:
                    endpoints.append((row, col))

            assert len(endpoints) == 2
            endpoints.sort()

            ordered = [endpoints[0]]
            previous = None
            current = endpoints[0]

            while current != endpoints[1]:
                choices = [
                    cell
                    for cell in _neighbors(mask, *current)
                    if cell != previous
                ]
                assert len(choices) == 1
                next_cell = choices[0]
                ordered.append(next_cell)
                previous, current = current, next_cell

            assert len(ordered) == len(component)
            paths.append(ordered)

    paths.sort(key=lambda path: (path[0], path[-1]))
    return paths


def _analyze(inp, palette):
    structural, centers, breaks, rows, overlaps = _parse_input(inp, palette)
    paths = _trace_paths(structural == palette["track"])
    bead = palette["bead"]
    bg = palette["background"]
    bead_cells = set(map(tuple, np.argwhere(inp == bead)))
    endpoints = set()
    for path in paths:
        # Every nonlevel path has a single direction along the gravity axis.
        # This excludes cycles as well as intermediate gravitational traps.
        steps = {b[0] - a[0] for a, b in zip(path, path[1:]) if a[0] != b[0]}
        assert len(steps) <= 1
        for r, c in (path[0], path[-1]):
            assert r % 2 == 0
            assert c in (0, inp.shape[1] - 1) or inp[r, c - 1] == bg or inp[r, c + 1] == bg
            endpoints.add((r, c))
    assert bead_cells <= endpoints
    return {"structural": structural, "centers": centers, "breaks": breaks,
            "per_separator": rows, "overlaps": overlaps, "paths": paths,
            "input_beads": bead_cells}


def _settle(inp, palette, info, sign=1):
    answer = info["structural"].copy()
    bead = palette["bead"]
    counts, target_cells = [], set()
    single = collisions = levels = settled = empty = 0
    for path in info["paths"]:
        beads = info["input_beads"] & {path[0], path[-1]}
        count = len(beads)
        counts.append(count)
        if not count:
            empty += 1
            continue
        if path[0][0] == path[-1][0]:
            targets = beads
            levels += 1
        else:
            ordered = path if sign * path[0][0] > sign * path[-1][0] else path[::-1]
            targets = set(ordered[:count])
            collisions += count == 2
            single += count == 1 and targets != beads
            settled += targets == beads
        target_cells.update(targets)
        for cell in targets:
            answer[cell] = bead
    assert np.count_nonzero(answer == bead) == len(info["input_beads"])
    details = dict(info)
    details.update(bead_counts=counts, output_beads=target_cells,
                   moved=target_cells != info["input_beads"], single=single,
                   collisions=collisions, levels=levels, settled=settled, empty=empty)
    return answer, details


def _inspect_input(inp, output_color):
    """Recover visible roles, lane axis and margins using the input alone."""
    colors = set(map(int, np.unique(inp)))
    assert len(colors) == 4 and output_color not in colors
    candidates = []
    for bg in sorted(colors):
        locations = np.argwhere(inp != bg)
        r0, c0 = locations.min(axis=0)
        r1, c1 = locations.max(axis=0) + 1
        crop = inp[r0:r1, c0:c1]
        for axis in (0, 1):
            core = crop if axis == 0 else crop.T
            h, w = core.shape
            if h < 3 or h % 2 != 1 or w < 7:
                continue
            divider_colors = set(map(int, np.unique(core[1::2]))) - {bg}
            if len(divider_colors) != 1:
                continue
            divider = next(iter(divider_colors))
            rest = colors - {bg, divider}
            if len(rest) != 2:
                continue
            for track in sorted(rest):
                bead = next(iter(rest - {track}))
                pal = dict(zip(_PAL_KEYS, (bg, track, bead, divider, output_color)))
                try:
                    info = _analyze(core, pal)
                except AssertionError:
                    continue
                candidates.append((core, pal, (int(r0), int(r1), int(c0), int(c1)), axis, info))
    assert len(candidates) == 1, "input roles/axis must be geometrically identifiable"
    return candidates[0]


def _restore(core, shape, palette, box, axis):
    answer = np.full(shape, palette["background"], dtype=np.int8)
    r0, r1, c0, c1 = box
    answer[r0:r1, c0:c1] = core if axis == 0 else core.T
    return answer


def _learn(train_pairs):
    output_color = None
    directions = set(_DIRS)
    inspections = []
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        new_colors = set(map(int, np.unique(out))) - set(map(int, np.unique(inp)))
        assert len(new_colors) == 1
        new = next(iter(new_colors))
        assert output_color is None or output_color == new
        output_color = new
        core, pal, box, axis, info = _inspect_input(inp, new)
        matching = set()
        for sign in (-1, 1):
            expected, details = _settle(core, pal, info, sign)
            if np.array_equal(out, _restore(expected, inp.shape, pal, box, axis)):
                matching.add((sign, 0) if axis == 0 else (0, sign))
        directions &= matching
        inspections.append((core, pal, box, axis, info))
    assert len(directions) == 1, "training must demonstrate a unique gravity direction"
    return output_color, next(iter(directions)), inspections


def solve(train_pairs, test_input):
    """Prediction uses only demonstrations and a query input, never its label."""
    color, gravity, _ = _learn(train_pairs)
    core, pal, box, axis, info = _inspect_input(test_input, color)
    sign = gravity[axis]
    assert sign in (-1, 1)
    answer, _ = _settle(core, pal, info, sign)
    return _restore(answer, test_input.shape, pal, box, axis)


def _build_base(lanes, width, centers, breaks, palette):
    grid = np.full((2 * lanes - 1, width), palette["background"], dtype=np.int8)
    grid[::2] = palette["track"]
    for row, c in centers:
        grid[row, c - 1:c + 2] = palette["input_divider"]
    for cell in breaks:
        grid[cell] = palette["background"]
    return grid


def _sample_centers(rng, lanes, width, woven, richness):
    rows = []
    for i in range(lanes - 1):
        previous = rows[-1] if rows else []
        if woven:
            # Interlace consecutive separator rows. Internal gaps are filled;
            # optional exterior dividers grow/shrink the routing throughout.
            internal = []
            for a, b in zip(previous, previous[1:]):
                internal.append(int(rng.integers(a + 2, b - 1)))
            chosen = internal
            if not previous:
                chosen = [int(rng.integers(2, width - 2))]
            else:
                limit = min((width - 1) // 4, lanes - 1 - i)
                exteriors = [(2, previous[0] - 2), (previous[-1] + 2, width - 3)]
                for j in rng.permutation(2):
                    lo, hi = exteriors[int(j)]
                    if lo <= hi and len(chosen) < limit and rng.random() < .65:
                        chosen.append(int(rng.integers(lo, hi + 1)))
            rows.append(sorted(chosen))
        else:
            maximum = min(6, (width - 1) // 4)
            wanted = int(rng.integers(0, maximum + 1)) if richness > .6 else int(rng.choice([0, 1, 1, 2]))
            chosen = []
            for c in rng.permutation(np.arange(2, width - 2)):
                c = int(c)
                if len(chosen) >= wanted:
                    break
                if all(abs(c - p) >= 2 for p in previous) and all(abs(c - p) >= 4 for p in chosen):
                    chosen.append(c)
            rows.append(sorted(chosen))
    if not any(rows):
        rows[int(rng.integers(lanes - 1))] = [int(rng.integers(2, width - 2))]
    return [(2 * i + 1, c) for i, row in enumerate(rows) for c in row]


def _required_breaks(rng, lanes, centers):
    breaks = []
    for lane in range(lanes):
        r = 2 * lane
        ports = sorted((c, dr) for dr, c in centers if abs(dr - r) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            # Two ports pointing the same way make a U turn. Cut that lane
            # segment so both ports lead to exposed terminals instead.
            if side_a == side_b:
                assert b - a >= 4
                breaks.append((r, int(rng.integers(a + 2, b - 1))))
    return breaks


def _break_candidates(lanes, width, centers, breaks):
    return [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(2, width - 2)
            if all(abs(c - x) >= 2 for y, x in centers if abs(y - r) == 1)
            and all(abs(c - x) >= 3 for y, x in breaks if y == r)]


def _layout(rng, lanes, width, palette, richness, require_level):
    centers = _sample_centers(rng, lanes, width, rng.random() < .42, richness)
    breaks = _required_breaks(rng, lanes, centers)
    extra = int(rng.choice([0, 0, 0, 1, 2, 3, 5]))
    for _ in range(extra):
        pool = _break_candidates(lanes, width, centers, breaks)
        if not pool:
            break
        breaks.append(pool[int(rng.integers(len(pool)))])
    base = _build_base(lanes, width, centers, breaks, palette)
    info = _analyze(base, palette)
    if require_level and not any(p[0][0] == p[-1][0] for p in info["paths"]):
        # Split an outer segment to exhibit level retention without a trap.
        pool = _break_candidates(lanes, width, centers, breaks)
        rng.shuffle(pool)
        for cell in pool:
            proposed = breaks + [cell]
            trial = _build_base(lanes, width, centers, proposed, palette)
            trial_info = _analyze(trial, palette)
            if any(p[0][0] == p[-1][0] for p in trial_info["paths"]):
                return trial, trial_info
        return None
    return base, info


def _choose_beads(inp, info, rng, mode, require_single, require_double, require_level):
    paths = info["paths"]
    vertical = [i for i, p in enumerate(paths) if p[0][0] != p[-1][0]]
    level = [i for i, p in enumerate(paths) if p[0][0] == p[-1][0]]
    assert vertical
    if require_level and not level:
        return None
    chosen = set()
    if mode == "source":
        for path in paths:
            chosen.update(cell for cell in (path[0], path[-1])
                          if cell[1] in (0, inp.shape[1] - 1) and cell[0] < inp.shape[0] - 1)
    else:
        probability = {"sparse": (.06, .25), "moderate": (.3, .65),
                       "dense": (.8, 1.), "settled": (.2, .8)}[mode]
        p = float(rng.uniform(*probability))
        for path in paths:
            endpoints = (path[0], path[-1])
            if mode == "settled" and endpoints[0][0] != endpoints[1][0]:
                endpoints = (max(endpoints),)
            chosen.update(cell for cell in endpoints if rng.random() < p)
    available = list(rng.permutation(vertical))
    if require_single:
        path = paths[int(available.pop())]
        chosen.difference_update((path[0], path[-1]))
        chosen.add(min(path[0], path[-1]))
    if require_double:
        path = paths[int(available.pop())]
        chosen.update((path[0], path[-1]))
    if require_level:
        path = paths[int(rng.choice(level))]
        chosen.add(path[0] if rng.random() < .5 else path[-1])
    if not chosen:
        path = paths[int(rng.integers(len(paths)))]
        chosen.add(max(path[0], path[-1]))
    return chosen


def _generate_pair(rng, regime, palette, size, density, richness, rotation,
                   require_single=False, require_double=False, require_level=False):
    assert regime in ("source_train", "target_test")
    for attempt in range(40):
        if size == "small":
            lanes, width = int(rng.integers(2, 5)), int(rng.integers(7, 17))
        elif size == "large":
            lanes, width = int(rng.integers(8, 16)), int(rng.integers(17, 31))
        else:
            lanes, width = int(rng.integers(2, 16)), int(rng.integers(7, 31))
        if require_level:
            width = max(9, width)
        layout = _layout(rng, lanes, width, palette, richness, require_level)
        if layout is not None:
            break
    else:
        # Bounded constructive fallback with both a level segment and two
        # unequal-endpoint paths; it has no dependence on rejection luck.
        lanes, width = 3, 11
        base = _build_base(lanes, width, [(1, 3), (3, 5)], [(0, 7)], palette)
        layout = (base, _analyze(base, palette))
    inp, info = layout
    beads = _choose_beads(inp, info, rng, density, require_single, require_double, require_level)
    assert beads
    inp = inp.copy()
    for cell in beads:
        inp[cell] = palette["bead"]
    info = _analyze(inp, palette)
    out, _ = _settle(inp, palette, info)
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    margins = []
    for dimension in inp.shape:
        room = min(4, 30 - dimension) if rng.random() < .45 else 0
        total = int(rng.integers(room + 1))
        before = int(rng.integers(total + 1))
        margins.append((before, total - before))
    inp = np.pad(inp, margins, constant_values=palette["background"])
    out = np.pad(out, margins, constant_values=palette["background"])
    return np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()


def _validate_count(
    name: str,
    value,
    low: int,
    high: int,
):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    value = int(value)

    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )

    return value


def _resolve_counts(
    seed: int,
    num_train,
    num_test,
) -> tuple[int, int]:
    num_train = _validate_count(
        "num_train",
        num_train,
        2,
        10,
    )
    num_test = _validate_count(
        "num_test",
        num_test,
        1,
        4,
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        base_train = 2 + (seed // 4) % 9
        num_train = max(base_train, num_test + 1)

    elif num_train is None:
        span = 10 - num_test
        num_train = num_test + 1 + seed % span

    elif num_test is None:
        highest_legal_test = min(4, num_train - 1)
        num_test = 1 + seed % highest_legal_test

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return int(num_train), int(num_test)


def _puzzles_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False

    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False

        for pair_a, pair_b in zip(
            a[split],
            b[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False

            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    pieces = []

    for split in ("train", "test"):
        for pair in puzzle[split]:
            pieces.append((
                split,
                pair["input"].shape,
                pair["input"].tobytes(),
                pair["output"].tobytes(),
            ))

    return tuple(pieces)



def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    output_color = int(rng.integers(10))
    colors = [c for c in range(10) if c != output_color]
    fixed_palette = list(map(int, rng.choice(colors, 4, replace=False)))
    vary_palette = rng.random() < .7
    rotation = int(rng.choice(4, p=[.45, .2, .15, .2]))
    shift = int(rng.choice(5, p=[.5, .15, .15, .1, .1]))
    demonstrations = list(map(int, rng.permutation(train_count)))
    richness = float(rng.random())
    puzzle = {"train": [], "test": []}
    seen = set()
    source_inputs = {(a.shape, a.tobytes()) for a, _ in _source_pairs()}
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for attempt in range(30):
                role_colors = list(map(int, pair_rng.choice(colors, 4, replace=False))) if vary_palette else fixed_palette
                palette = dict(zip(_PAL_KEYS, role_colors + [output_color]))
                size = "all"
                if shift in (1, 2):
                    small = (split == "train") == (shift == 1)
                    size = "small" if small else "large"
                density = str(pair_rng.choice(["sparse", "moderate", "dense", "source", "settled"],
                                              p=[.22, .28, .23, .19, .08]))
                if shift in (3, 4):
                    dense = (split == "train") == (shift == 3)
                    density = str(pair_rng.choice(["dense", "source"] if dense else ["sparse", "moderate"]))
                single = split == "train" and index == demonstrations[0]
                double = split == "train" and index == demonstrations[1]
                inp, out = _generate_pair(pair_rng,
                    "source_train" if split == "train" else "target_test", palette,
                    size, density, .5 * richness + .5 * float(pair_rng.random()), rotation,
                    require_single=single, require_double=double, require_level=single)
                key = (inp.shape, inp.tobytes())
                if key not in seen and key not in source_inputs:
                    seen.add(key)
                    puzzle[split].append({"input": inp, "output": out})
                    break
            else:
                raise AssertionError("failed to construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    signatures = set()
    source_pairs = _source_pairs()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            signature = (inp.shape, inp.astype(np.int8).tobytes())
            assert signature not in signatures
            signatures.add(signature)
            assert not any(np.array_equal(inp, a) and np.array_equal(out, b) for a, b in source_pairs)
    color, gravity, train_info = _learn(puzzle["train"])
    coverage = {"single": False, "collisions": False, "levels": False}
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            inp, out = pair["input"], pair["output"]
            inspected = train_info[i] if split == "train" else _inspect_input(inp, color)
            core, pal, box, axis, info = inspected
            # Whole-puzzle rotation supplies one visible lane axis and one
            # demonstrated gravity direction. Pair colors and margins vary.
            sign = gravity[axis]
            assert sign in (-1, 1)
            expected, details = _settle(core, pal, info, sign)
            assert np.array_equal(out, _restore(expected, inp.shape, pal, box, axis))
            if split == "train":
                for feature in coverage:
                    coverage[feature] |= bool(details[feature])
    assert all(coverage.values()), "training needs motion, packing and level retention"


def review_features(puzzle):
    """Reconstruct measurable review axes from the returned grids."""
    color, gravity, _ = _learn(puzzle["train"])
    records = []
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            inp = pair["input"]
            core, pal, box, axis, info = _inspect_input(inp, color)
            _, details = _settle(core, pal, info, gravity[axis])
            counts = [len(row) for row in info["per_separator"]]
            beads = info["input_beads"]
            endpoints = {cell for path in info["paths"] for cell in (path[0], path[-1])}
            low_row = core.shape[0] - 1 if gravity[axis] > 0 else 0
            source_beads = {cell for cell in endpoints if cell[1] in (0, core.shape[1] - 1) and cell[0] != low_row}
            normalized = np.zeros(core.shape, dtype=np.int8)
            for code, key in enumerate(_PAL_KEYS[:4]):
                normalized[core == pal[key]] = code
            records.append({
                "split": split, "pair": i, "shape": tuple(map(int, inp.shape)),
                "lanes": (core.shape[0] + 1) // 2, "width": core.shape[1],
                "axis": axis, "gravity": gravity, "palette": _palette_tuple(pal),
                "margins": int(inp.size - core.size), "even_canvas": any(n % 2 == 0 for n in inp.shape),
                "dividers": len(info["centers"]), "separator_counts": counts,
                "blank_positions": [j for j, n in enumerate(counts) if n == 0],
                "busy_rows": sum(n >= 2 for n in counts), "overlaps": info["overlaps"],
                "breaks": len(info["breaks"]), "paths": len(info["paths"]),
                "beads": len(beads), "occupancy": len(beads) / len(endpoints),
                "source_placement": beads == source_beads,
                "interior_beads": sum(c not in (0, core.shape[1] - 1) for r, c in beads),
                "no_break_multi": not info["breaks"] and max(counts) >= 2,
                "connected_final_multi": counts[-1] >= 2 and len(counts) > 1 and counts[-2] >= 1,
                "single": details["single"], "collisions": details["collisions"],
                "levels": details["levels"], "settled": details["settled"],
                "empty": details["empty"], "settled_only": not details["moved"],
                "layout": (core.shape, normalized.tobytes()),
            })
    return records


def self_test() -> None:
    # These are regressions, not generation templates. The first case includes
    # the four overlapping writes and the second is a break-free busy network.
    sources = _source_pairs()
    source_train = [{"input": a, "output": b} for a, b in sources[:2]]
    for inp, expected in sources:
        assert np.array_equal(solve(source_train, inp), expected)
    palette = dict(zip(_PAL_KEYS, (0, 8, 3, 7, 6)))
    compact = _build_base(3, 9, [(1, 2), (3, 4)], [], palette)
    compact[0, 0] = 3
    compact[2, 0] = 3
    info = _analyze(compact, palette)
    assert info["overlaps"] == 1
    assert _settle(compact, palette, info)[1]["collisions"] == 1
    # Conflicting writes, branching aligned dividers, cycles and traps fail.
    invalid_layouts = [(3, 9, [(1, 2), (3, 3)], []),
                       (3, 9, [(1, 3), (3, 3)], []),
                       (2, 11, [(1, 3), (1, 7)], []),
                       (2, 11, [(1, 3), (1, 7)], [(0, 5)])]
    for lanes, width, centers, breaks in invalid_layouts:
        try:
            _analyze(_build_base(lanes, width, centers, breaks, palette), palette)
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid topology accepted")
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzles_equal(generate_puzzle(314159, 6, 3), generate_puzzle(314159, 6, 3))
    assert not _puzzles_equal(automatic[0], automatic[1])
    for nt, nq in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(800 + (nt or 0) + (nq or 0), nt, nq)
        assert nt is None or len(puzzle["train"]) == nt
        assert nq is None or len(puzzle["test"]) == nq
        validate_generated_puzzle(puzzle)
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(999, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    all_records, within = [], []
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        # Predictions receive training copies and a test input only.
        train = [{k: v.copy() for k, v in pair.items()} for pair in puzzle["train"]]
        predictions = [solve(train, pair["input"].copy()) for pair in puzzle["test"]]
        assert all(np.array_equal(pred, pair["output"]) for pred, pair in zip(predictions, puzzle["test"]))
        records = review_features(puzzle)
        all_records.extend(records)
        within.append(records)
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert len({_puzzle_signature(p) for p in automatic}) == 42
    assert len({r["layout"] for r in all_records}) >= 150
    assert len({r["shape"] for r in all_records}) >= 70
    assert len({r["palette"] for r in all_records}) >= 60
    for role in range(5):
        assert {r["palette"][role] for r in all_records} == set(range(10))
    assert {r["gravity"] for r in all_records} == set(_DIRS)
    assert {r["lanes"] for r in all_records} == set(range(2, 16))
    assert min(r["width"] for r in all_records) == 7 and max(r["width"] for r in all_records) == 30
    assert max(r["beads"] for r in all_records) >= 24
    assert len({r["breaks"] for r in all_records}) >= 8
    assert len({r["dividers"] for r in all_records}) >= 15
    for split in ("train", "test"):
        rows = [r for r in all_records if r["split"] == split]
        assert {0, 1, 2, 3} <= {c for r in rows for c in r["separator_counts"]}
        for key in ("overlaps", "no_break_multi", "connected_final_multi", "interior_beads",
                    "source_placement", "settled_only", "margins", "even_canvas"):
            assert any(r[key] for r in rows), (split, key)
        assert any(r["busy_rows"] >= 2 for r in rows)
        assert any(r["dividers"] == 1 for r in rows)
        assert any(r["breaks"] == 0 for r in rows)
        assert any(0 in r["blank_positions"] for r in rows)
        assert any(len(r["separator_counts"]) - 1 in r["blank_positions"] for r in rows)
        assert any(any(0 < j < len(r["separator_counts"]) - 1 for j in r["blank_positions"]) for r in rows)
        assert any(r["collisions"] >= 4 for r in rows)
        assert any(r["lanes"] <= 4 and r["occupancy"] >= .75 for r in rows)
        assert any(r["lanes"] >= 10 and r["occupancy"] <= .3 and r["dividers"] >= 6 for r in rows)
    assert sum(len({r["palette"] for r in rows}) > 1 for rows in within) >= 20
    assert sum(len({r["shape"] for r in rows}) > 1 for rows in within) >= 40
    assert sum(len({r["beads"] for r in rows}) >= 3 for rows in within) >= 25
    for feature in ("occupancy", "dividers", "breaks"):
        assert sum(len({r[feature] for r in rows}) > 1 for rows in within) >= 35
    density_shifts = []
    for rows in within:
        means = [np.mean([r["occupancy"] for r in rows if r["split"] == split])
                 for split in ("train", "test")]
        density_shifts.append(means[1] - means[0])
    assert min(density_shifts) < -.25 and max(density_shifts) > .25
    assert any(min(r["lanes"] for r in rows if r["split"] == "test") >
               max(r["lanes"] for r in rows if r["split"] == "train") for rows in within)
    assert any(max(r["lanes"] for r in rows if r["split"] == "test") <
               min(r["lanes"] for r in rows if r["split"] == "train") for rows in within)
    assert len({r["pair"] for r in all_records if r["split"] == "train" and r["single"] and r["levels"]}) >= 4
    # Validation must reject a changed query label.
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in automatic[0][s]] for s in ("train", "test")}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect query output accepted")
