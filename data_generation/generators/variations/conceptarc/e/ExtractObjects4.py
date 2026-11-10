import numpy as np


# Exact source inputs are excluded; their canvas sizes are valid.
_SOURCE_INPUTS = frozenset((
    "000000000000000/222222222222222/000000000000000/222222222222222/000000000000000/222222222222222/022222000000000/222222222222222/022222000000000/222222222222222/000000000000000/222222222222222/000000000000000/222222222222222/000000000000000",
    "000000/666666/000000/666666/066660/666666/066660/666666/000000/666666/000000/666666/000000/666666",
    "0000000000000000/4444444444444444/0000000000044400/4444444444444444/0000000000000000/4444444444444444",
    "88888888888/00000000000/88888888888/00000000000/88888888888/00888888800/88888888888/00888888800/88888888888/00888888800/88888888888",
    "000000/555555/055000/555555/000000/555555",
    "30303030303030/30303030303030/30333330303030/30333330303030/30333330303030/30333330303030",
))


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("invalid pair count")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("training count must exceed test count")
    return int(num_train), int(num_test)


def _grid_text(grid):
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _signature(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _build(n, m, orientation, phase, start, width, along, length, bg, ink):
    view = np.full((n, m), bg, dtype=np.int8)
    view[phase::2, :] = ink
    view[start:start + width, along:along + length] = ink
    inp = view if orientation == "h" else view.T.copy()
    shape = (width, length) if orientation == "h" else (length, width)
    return inp, np.full(shape, ink, dtype=np.int8)


def _decode(inp):
    """Enumerate every single striped rectangle consistent with the input."""
    colors = np.unique(inp)
    assert len(colors) == 2
    matches = []
    for orientation in ("h", "v"):
        view = inp if orientation == "h" else inp.T
        n, m = view.shape
        for phase in (0, 1):
            stripes = np.arange(phase, n, 2)
            blanks = np.arange(1 - phase, n, 2)
            if len(stripes) < 2 or len(blanks) < 2:
                continue
            for ink in colors:
                ink = int(ink)
                bg = int(colors[0] if colors[1] == ink else colors[1])
                if not np.all(view[stripes] == ink):
                    continue
                changed, spans = [], []
                unchanged = 0
                for band in blanks:
                    positions = np.flatnonzero(view[band] == ink)
                    if len(positions) == 0:
                        unchanged += 1
                    elif (len(positions) >= 2 and
                          positions[-1] - positions[0] + 1 == len(positions)):
                        changed.append(int(band))
                        spans.append((int(positions[0]), int(positions[-1]) + 1))
                    else:
                        break
                else:
                    if not changed or not unchanged or len(set(spans)) != 1:
                        continue
                    first, last = changed[0], changed[-1]
                    if (changed != list(range(first, last + 1, 2)) or
                            first < 1 or last + 1 >= n):
                        continue
                    lo, hi = spans[0]
                    expected = np.full(view.shape, bg, dtype=inp.dtype)
                    expected[phase::2, :] = ink
                    expected[first - 1:last + 2, lo:hi] = ink
                    if not np.array_equal(expected, view):
                        continue
                    width = last - first + 3
                    matches.append({
                        "orientation": orientation, "phase": phase,
                        "bg": bg, "ink": ink,
                        "shape": ((width, hi - lo) if orientation == "h"
                                  else (hi - lo, width)),
                        "perp_start": first - 1, "perp_stop": last + 2,
                        "perp_n": n, "long_start": lo, "long_stop": hi,
                        "long_n": m, "changed_bands": len(changed),
                        "unchanged_bands": unchanged,
                    })
    assert len(matches) == 1, "input does not identify one striped rectangle"
    return matches[0]


def _dimension(rng, minimum):
    lo, hi = ((minimum, 9), (9, 19), (19, 31))[int(rng.integers(3))]
    return int(rng.integers(lo, hi))


def _pick_bucket(rng, candidates, key, bins, weights):
    available = [i for i, (lo, hi) in enumerate(bins)
                 if any(lo <= key(item) <= hi for item in candidates)]
    probs = np.array([weights[i] for i in available], dtype=float)
    probs /= probs.sum()
    bucket = available[int(rng.choice(len(available), p=probs))]
    lo, hi = bins[bucket]
    subset = [item for item in candidates if lo <= key(item) <= hi]
    return subset[int(rng.integers(len(subset)))]


def _geometry(rng, regime, phase):
    n, m = _dimension(rng, 5), _dimension(rng, 4)
    choices = []
    blank_count = len(range(1 - phase, n, 2))
    for start in range(phase, n, 2):
        for stop in range(start + 3, n + 1, 2):
            changed = (stop - start - 1) // 2
            if blank_count - changed >= 1:
                edge = ("first" if start == 0 else
                        "last" if stop == n else "interior")
                choices.append((start, stop - start, edge))
    if not choices:
        return None
    edge_weights = ((0.56, 0.22, 0.22) if regime == "source_train"
                    else (0.42, 0.29, 0.29))
    edge_names = ("interior", "first", "last")
    available = [i for i, name in enumerate(edge_names)
                 if any(item[2] == name for item in choices)]
    probs = np.array([edge_weights[i] for i in available], dtype=float)
    probs /= probs.sum()
    edge = edge_names[available[int(rng.choice(len(available), p=probs))]]
    choices = [item for item in choices if item[2] == edge]
    weights = ((0.40, 0.35, 0.25) if regime == "source_train"
               else (0.28, 0.34, 0.38))
    start, width, _ = _pick_bucket(
        rng, choices, lambda item: item[1],
        ((3, 5), (7, 11), (13, 29)), weights)
    length = _pick_bucket(
        rng, list(range(2, m + 1)), lambda item: item,
        ((2, 4), (5, 9), (10, 30)),
        ((0.38, 0.37, 0.25) if regime == "source_train"
         else (0.30, 0.34, 0.36)))
    if int(rng.integers(10)) == 0:
        length = m
    if length == m:
        along = 0
    else:
        positions = list(range(m - length + 1))
        edge_weights = ((0.54, 0.23, 0.23) if regime == "source_train"
                        else (0.40, 0.30, 0.30))
        groups = (positions[1:-1], positions[:1], positions[-1:])
        available = [i for i, group in enumerate(groups) if group]
        probs = np.array([edge_weights[i] for i in available], dtype=float)
        probs /= probs.sum()
        group = groups[available[int(rng.choice(len(available), p=probs))]]
        along = group[int(rng.integers(len(group)))]
    return n, m, start, width, along, length


def _orientation(rng, profile, section):
    if profile == 0:
        return "h"
    if profile == 1:
        return "v"
    if profile == 2:
        return "h" if section == "train" else "v"
    if profile == 3:
        return "v" if section == "train" else "h"
    return "h" if int(rng.integers(2)) == 0 else "v"


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    profile = int(rng.integers(5))
    varied_background = bool(rng.integers(2))
    first_bg = int(rng.integers(10))
    first_ink = int(rng.choice([c for c in range(10) if c != first_bg]))
    seen = set()
    puzzle = {"train": [], "test": []}
    for index in range(ntrain + ntest):
        section = "train" if index < ntrain else "test"
        for attempt in range(240):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            if index == 0:
                bg, ink = first_bg, first_ink
            elif index == 1 and varied_background:
                bg = first_ink
                ink = (first_bg if int(pair_rng.integers(2)) == 0 else
                       int(pair_rng.choice([c for c in range(10) if c != bg])))
            else:
                bg = (int(pair_rng.integers(10)) if varied_background
                      else first_bg)
                inks = [c for c in range(10) if c != bg and
                        (index != 1 or c != first_ink)]
                ink = int(pair_rng.choice(inks))
            orientation = _orientation(pair_rng, profile, section)
            phase = int(pair_rng.integers(2))
            regime = "source_train" if section == "train" else "target_test"
            geometry = _geometry(pair_rng, regime, phase)
            if geometry is None:
                continue
            n, m, start, width, along, length = geometry
            inp, out = _build(n, m, orientation, phase, start, width,
                              along, length, bg, ink)
            signature = _signature(inp)
            if signature in seen or _grid_text(inp) in _SOURCE_INPUTS:
                continue
            try:
                decoded = _decode(inp)
            except AssertionError:
                continue
            if decoded["shape"] != out.shape or decoded["ink"] != ink:
                continue
            if index == 1 and out.shape == puzzle["train"][0]["output"].shape:
                continue
            break
        else:
            raise AssertionError("could not draw a distinct recoverable pair")
        seen.add(signature)
        puzzle[section].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    decoded = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            insig, outsig = _signature(inp), _signature(out)
            assert insig not in seen_inputs
            assert (insig, outsig) not in seen_pairs
            assert _grid_text(inp) not in _SOURCE_INPUTS
            seen_inputs.add(insig)
            seen_pairs.add((insig, outsig))
            d = _decode(inp)
            assert out.shape == d["shape"] and np.all(out == d["ink"])
            assert not np.array_equal(inp, out)
            assert d["shape"][0] >= 2 and d["shape"][1] >= 2
            assert d["changed_bands"] >= 1 and d["unchanged_bands"] >= 1
            decoded[section].append(d)
    assert decoded["train"][0]["shape"] != decoded["train"][1]["shape"]
    all_bg = {d["bg"] for section in decoded.values() for d in section}
    train_bg = {d["bg"] for d in decoded["train"]}
    if len(all_bg) > 1:
        assert len(train_bg) > 1


def _same_puzzle(a, b):
    return (all(len(a[s]) == len(b[s]) for s in ("train", "test")) and
            all(np.array_equal(x[k], y[k])
                for s in ("train", "test") for x, y in zip(a[s], b[s])
                for k in ("input", "output")))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    signatures, layouts, output_shapes = set(), set(), set()
    orientations, phases, edges, color_roles = set(), set(), set(), set()
    odd_sides = small_sides = large_sides = full_spans = corners = 0
    mixed_orientation = varied_background = mixed_phase = 0
    orientation_profiles = set()
    axis_parities = set()
    first_slots = {("train", 0): set(), ("train", 1): set(),
                   ("test", 0): set()}
    role_swaps = source_sized = big_train = edge_train = 0
    small_test = interior_test = 0
    combinations = set()
    source_shapes = {(15, 15), (14, 6), (6, 16),
                     (11, 11), (6, 6), (6, 14)}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        ds = {s: [_decode(p["input"]) for p in puzzle[s]]
              for s in ("train", "test")}
        orientation_profiles.add((tuple(sorted({d["orientation"] for d in ds["train"]})),
                                  tuple(sorted({d["orientation"] for d in ds["test"]}))))
        all_d = ds["train"] + ds["test"]
        role_swaps += ds["train"][1]["bg"] == ds["train"][0]["ink"]
        big_train += any(max(d["shape"]) >= 13 for d in ds["train"])
        edge_train += any(d["perp_start"] == 0 or
                          d["perp_stop"] == d["perp_n"] for d in ds["train"])
        small_test += any(max(d["shape"]) <= 5 for d in ds["test"])
        interior_test += any(d["perp_start"] > 0 and
                             d["perp_stop"] < d["perp_n"] for d in ds["test"])
        mixed_orientation += len({d["orientation"] for d in all_d}) > 1
        mixed_phase += len({d["phase"] for d in all_d}) > 1
        varied_background += len({d["bg"] for d in all_d}) > 1
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test")
                             for p in puzzle[s]))
        for section in ("train", "test"):
            for index, (pair, d) in enumerate(zip(puzzle[section], ds[section])):
                inp, out = pair["input"], pair["output"]
                assert out.shape == d["shape"] and np.all(out == d["ink"])
                assert _grid_text(inp) not in _SOURCE_INPUTS
                layouts.add((inp.shape, d["orientation"], d["phase"],
                             d["perp_start"], d["long_start"], out.shape))
                output_shapes.add(out.shape)
                orientations.add((section, d["orientation"]))
                phases.add((section, d["phase"]))
                color_roles.add((d["bg"], d["ink"]))
                axis_parities.add((d["long_stop"] - d["long_start"]) % 2)
                source_sized += inp.shape in source_shapes
                if (section, index) in first_slots:
                    first_slots[(section, index)].add(
                        (d["orientation"], d["phase"], d["shape"]))
                odd_sides += any(side % 2 for side in inp.shape)
                small_sides += min(inp.shape) <= 6
                large_sides += max(inp.shape) >= 25
                full_spans += d["long_stop"] - d["long_start"] == d["long_n"]
                perp_edge = (d["perp_start"] == 0 or
                             d["perp_stop"] == d["perp_n"])
                axis_edge = (d["long_start"] == 0 or
                             d["long_stop"] == d["long_n"])
                corners += perp_edge and axis_edge
                combinations.add((d["orientation"], d["phase"],
                                  perp_edge, axis_edge,
                                  d["shape"][0] >= 7,
                                  d["shape"][1] >= 7))
                edges.add((section, "first" if d["perp_start"] == 0 else
                           "last" if d["perp_stop"] == d["perp_n"] else
                           "interior"))
                edges.add((section, "axis_first" if d["long_start"] == 0 else
                           "axis_last" if d["long_stop"] == d["long_n"] else
                           "axis_interior"))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(layouts) >= 130
    assert len(output_shapes) >= 45 and len(color_roles) >= 35
    assert orientations == {(s, o) for s in ("train", "test") for o in ("h", "v")}
    assert phases == {(s, p) for s in ("train", "test") for p in (0, 1)}
    assert all((s, e) in edges for s in ("train", "test")
               for e in ("first", "last", "interior", "axis_first",
                         "axis_last", "axis_interior"))
    assert axis_parities == {0, 1}
    assert min(odd_sides, small_sides, large_sides, full_spans, corners) >= 5
    assert min(mixed_orientation, varied_background, mixed_phase) >= 10
    assert min(role_swaps, source_sized, big_train, edge_train,
               small_test, interior_test) >= 4
    assert len(combinations) >= 25
    assert all(len(values) >= 20 for values in first_slots.values())
    assert (("h",), ("h",)) in orientation_profiles
    assert (("v",), ("v",)) in orientation_profiles
    assert (("h",), ("v",)) in orientation_profiles
    assert (("v",), ("h",)) in orientation_profiles
    for counts in ((2, 1), (10, 1), (10, 4),
                   (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(173, *counts)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(173, *counts))
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (3, 3)):
        try:
            generate_puzzle(7, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
