from __future__ import annotations
import numpy as np

def _ok(value, message):
    if not value:
        raise AssertionError(message)

def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value

def _counts(seed, rng, num_train, num_test):
    tr = _count("num_train", num_train, 2, 10)
    te = _count("num_test", num_test, 1, 4)
    if tr is None and te is None:
        te = 1 + seed % 4
        tr = te + 1 + (seed // 4) % (10 - te)
    elif tr is None:
        tr = int(rng.integers(te + 1, 11))
    elif te is None:
        te = int(rng.integers(1, min(4, tr - 1) + 1))
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    return tr, te

def _base(n, row, orientation):
    return row if orientation == "tl" else n - 1 - row

def _render(n, bg, orientation, profiles):
    grid = np.full((n, n), bg, dtype=np.int8)
    for row in range(n):
        base = _base(n, row, orientation)
        for offset, color in profiles[row % len(profiles)].items():
            col = base + offset
            if 0 <= col < n:
                grid[row, col] = color
    return grid

def _infer(grid):
    n = grid.shape[0]
    colors, counts = np.unique(grid, return_counts=True)
    _ok(len(colors) >= 2 and sum(counts == max(counts)) == 1, "background unclear")
    bg = int(colors[np.argmax(counts)])
    foreground = np.argwhere(grid != bg)
    residual = {o: max(abs(int(c) - _base(n, int(r), o)) for r, c in foreground)
                for o in ("tl", "anti")}
    orientation = min(residual, key=residual.get)
    other = "anti" if orientation == "tl" else "tl"
    _ok(residual[orientation] <= 2 and residual[orientation] + 2 < residual[other],
        "orientation unclear")
    for period in range(1, min(4, n // 2) + 1):
        profiles = [{} for _ in range(period)]
        valid = True
        for phase in range(period):
            rows = list(range(phase, n, period))
            if len(rows) < 2 or any(
                not any(0 <= _base(n, r, orientation) + offset < n for r in rows)
                for offset in range(-2, 3)
            ):
                valid = False
                break
            for offset in range(-2, 3):
                seen = {int(grid[r, _base(n, r, orientation) + offset]) for r in rows
                        if 0 <= _base(n, r, orientation) + offset < n}
                if len(seen) != 1:
                    valid = False
                    break
                color = seen.pop()
                if color != bg:
                    profiles[phase][offset] = color
            if not valid:
                break
        if valid and all(profiles) and np.array_equal(_render(n, bg, orientation, profiles), grid):
            support = tuple(sorted({x for p in profiles for x in p}))
            _ok(0 in support and support == tuple(range(support[0], support[-1] + 1)),
                "band not anchored")
            return dict(background=bg, orientation=orientation, period=period,
                        profiles=profiles, support=support, width=len(support), size=n,
                        colors=frozenset(int(c) for c in colors))
    raise AssertionError("profiles not uniquely recoverable")

_ENVELOPES = {
    1: [(0,)],
    2: [(-1, 0), (0, 1)],
    3: [(-2, -1, 0), (-1, 0, 1), (0, 1, 2)],
    4: [(-2, -1, 0, 1), (-1, 0, 1, 2)],
    5: [(-2, -1, 0, 1, 2)],
}

def _letters(rng, period):
    if period == 1:
        return [0]
    if period == 2:
        return [0, 1]
    if period == 3 and rng.random() < .45:
        return [0, 0, 1]
    if period == 4 and rng.random() < .45:
        return [0, 1, 0, 2]
    return list(range(period))

def _profiles(rng, width, period, motif):
    letters = _letters(rng, period)
    envelope = _ENVELOPES[width][int(rng.integers(len(_ENVELOPES[width])))]
    supports = [envelope]
    if width >= 3 and max(letters) >= 1 and rng.random() < .36:
        # Adjacent phases can occupy opposite sides of the diagonal even
        # when neither phase fills the whole envelope.
        if width == 3:
            envelope = (-1, 0, 1)
        supports = [envelope[:-1], envelope[1:]]
    for _ in range(max(letters)):
        if len(supports) >= max(letters) + 1:
            break
        variants = [envelope]
        if width > 1:
            variants += [envelope[1:], envelope[:-1]]
        if width > 2:
            variants += [tuple(x for x in envelope if x != 0),
                         envelope[::2], envelope[1::2]]
        supports.append(variants[int(rng.integers(len(variants)))])
    slots = [(letter, offset) for letter, support in enumerate(supports)
             for offset in support]
    assignments = [motif[int(rng.integers(len(motif)))] for _ in slots]
    for index, color in zip(rng.permutation(len(slots)), motif):
        assignments[int(index)] = color
    unique = [{} for _ in supports]
    for (letter, offset), color in zip(slots, assignments):
        unique[letter][offset] = color
    return [dict(unique[letter]) for letter in letters]

def _palette(rng, width, forced):
    bg = int(rng.integers(10))
    if bg == forced:
        bg = (bg + 1) % 10
    count = int(rng.integers(1 if width <= 3 else 2, 5))
    available = [c for c in range(10) if c != bg and c != forced]
    colors = list(int(c) for c in rng.choice(available, count - (forced is not None),
                                               replace=False))
    if forced is not None:
        colors.append(forced)
    rng.shuffle(colors)
    return bg, tuple(colors)

def _minimum_size(width, period):
    return max(5, 2 * period, {1: 5, 2: 5, 3: 7, 4: 8, 5: 9}[width])

def _make_pair(rng, orientation, width, period, size, forced):
    for _ in range(120):
        bg, motif = _palette(rng, width, forced)
        if width == 1 and period > 1 and len(motif) < 2:
            continue
        profiles = _profiles(rng, width, period, motif)
        inp = _render(size, bg, orientation, profiles)
        try:
            info = _infer(inp)
        except AssertionError:
            continue
        if (info["orientation"] == orientation and info["period"] == period
                and info["width"] == width and info["colors"] == frozenset((bg,) + motif)):
            return {"input": inp, "output": _render(2 * size, bg, orientation, profiles)}
    raise RuntimeError("cannot make recoverable pair")

def _signature(grid):
    return tuple(grid.shape), bytes(grid.astype(np.int8).tobytes())

_SOURCE_INPUTS = (
    ("25000", "52000", "00250", "00520", "00002"),
    ("410000", "132000", "023100", "001410", "000132", "000023"),
    ("00000001", "00000060", "00000100", "00006000", "00010000", "00600000", "01000000", "60000000"),
    ("0000000066", "0000000486", "0000006640", "0000048600", "0000664000", "0004860000", "0066400000", "0486000000", "6640000000", "8600000000"),
)

def _is_source(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS

def _normalized(pair):
    info = _infer(pair["input"])
    mapping = {}
    phases = []
    for profile in info["profiles"]:
        phase = []
        for offset, color in sorted(profile.items()):
            if color not in mapping:
                mapping[color] = len(mapping)
            phase.append((offset, mapping[color]))
        phases.append(tuple(phase))
    return info["orientation"], tuple(phases)

def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    # The shared rule is continuation; colors and profile geometry are local.
    thick_orientation = "tl" if rng.random() < .5 else "anti"
    thin_orientation = ("anti" if thick_orientation == "tl" else "tl") if rng.random() < .68 else thick_orientation
    specs = [(thick_orientation, int(rng.integers(3, 6)), int(rng.integers(2, 4))),
             (thin_orientation, 1, int(rng.integers(2, 5)))]
    for _ in range(train_count - 2):
        specs.append(("tl" if rng.random() < .5 else "anti",
                      int(rng.integers(1, 6)), int(rng.integers(1, 5))))
    rng.shuffle(specs)
    scale_mode = int(rng.integers(5))
    used_inputs = set()
    used_normalized = set()
    def add(orientation, width, period, forced=None, target=False):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        for attempt in range(160):
            minimum = _minimum_size(width, period)
            if scale_mode == 0:
                low, high = (max(10, minimum), 16) if target else (minimum, 10)
            elif scale_mode == 1:
                low, high = (minimum, 10) if target else (max(10, minimum), 16)
            else:
                low, high = minimum, 16
            size = int(pair_rng.integers(low, high))
            pair = _make_pair(pair_rng, orientation, width, period, size, forced)
            signature = _signature(pair["input"])
            if signature in used_inputs or _is_source(pair["input"]):
                continue
            normalized = _normalized(pair)
            if normalized in used_normalized and attempt < 120:
                continue
            used_inputs.add(signature)
            used_normalized.add(normalized)
            return pair
        raise RuntimeError("cannot make distinct pair")
    train = [add(*spec) for spec in specs]
    train_info = [_infer(p["input"]) for p in train]
    orientations = tuple(sorted({i["orientation"] for i in train_info}))
    seen_colors = {int(c) for p in train for c in np.unique(p["input"])}
    new_colors = tuple(c for c in range(10) if c not in seen_colors)
    novel = bool(new_colors) and rng.random() < .55
    test = []
    for index in range(test_count):
        orientation = orientations[int(rng.integers(len(orientations)))]
        width = int(rng.integers(1, 6))
        period = int(rng.integers(1, 5))
        forced = int(rng.choice(new_colors)) if novel and index == 0 else None
        test.append(add(orientation, width, period, forced, True))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle

def validate_generated_puzzle(puzzle: dict) -> None:
    _ok(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    train, test = puzzle["train"], puzzle["test"]
    _ok(isinstance(train, list) and isinstance(test, list), "split types")
    _ok(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
        "pair counts")
    inputs = set()
    infos = []
    for pair in train + test:
        _ok(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            _ok(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                np.issubdtype(grid.dtype, np.integer), "grid type")
            _ok(all(1 <= d <= 30 for d in grid.shape) and
                bool(np.all((grid >= 0) & (grid <= 9))), "grid bounds")
        _ok(inp.shape[0] == inp.shape[1] and out.shape == (2 * inp.shape[0],) * 2,
            "canvas doubling")
        signature = _signature(inp)
        _ok(signature not in inputs, "duplicate input")
        inputs.add(signature)
        _ok(not _is_source(inp), "source copy")
        info = _infer(inp)
        expected = _render(2 * info["size"], info["background"],
                           info["orientation"], info["profiles"])
        _ok(np.array_equal(out, expected), "wrong continuation")
        _ok(set(int(c) for c in np.unique(out)) == set(info["colors"]), "colors changed")
        infos.append(info)
    train_info = infos[:len(train)]
    test_info = infos[len(train):]
    _ok(any(i["width"] >= 3 and i["period"] >= 2 for i in train_info),
        "training lacks thick repeated band")
    _ok(any(i["width"] == 1 for i in train_info), "training lacks thin band")
    known = {i["orientation"] for i in train_info}
    _ok(all(i["orientation"] in known for i in test_info),
        "test orientation unseen in training")

def _puzzle_signature(puzzle):
    return tuple(_signature(p["input"]) for split in ("train", "test") for p in puzzle[split])

def self_test() -> None:
    train_counts, test_counts, backgrounds = set(), set(), set()
    widths, periods, sizes, layouts = set(), set(), set(), set()
    orientations, signatures = set(), set()
    palette_changes = shape_changes = novel_colors = smaller_tests = 0
    repeated_phases = missing_centers = 0
    varying_support = mixed_lengths = left_right = wide = clipped = 0
    aab = abac = larger_tests = source_like = reverse = 0
    thin_tl = thin_anti = thick_anti = test_tl = 0
    normalized_repeats = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = [_infer(p["input"]) for p in puzzle["train"] + puzzle["test"]]
        cut = len(puzzle["train"])
        normalized_repeats += len({_normalized(p) for p in puzzle["train"] + puzzle["test"]}) < len(infos)
        palette_changes += len({(i["background"], i["colors"]) for i in infos}) > 1
        shape_changes += len({(i["orientation"], i["support"], i["period"]) for i in infos}) > 1
        orientations.add(tuple(sorted({i["orientation"] for i in infos[:cut]})))
        train_colors = set().union(*(set(i["colors"]) for i in infos[:cut]))
        novel_colors += any(set(i["colors"]) - train_colors for i in infos[cut:])
        smaller_tests += min(i["size"] for i in infos[cut:]) <= max(i["size"] for i in infos[:cut])
        larger_tests += min(i["size"] for i in infos[cut:]) > max(i["size"] for i in infos[:cut])
        thick_anti += any(i["orientation"] == "anti" and i["width"] >= 3 for i in infos[:cut])
        thin_tl += any(i["orientation"] == "tl" and i["width"] == 1 for i in infos[:cut])
        thin_anti += any(i["orientation"] == "anti" and i["width"] == 1 for i in infos[:cut])
        test_tl += any(i["orientation"] == "tl" for i in infos[cut:])
        source_like += (any(i["orientation"] == "tl" and i["width"] >= 3 for i in infos[:cut])
                        and any(i["orientation"] == "anti" and i["width"] == 1 for i in infos[:cut])
                        and any(i["orientation"] == "anti" and i["width"] >= 3 for i in infos[cut:]))
        reverse += (any(i["orientation"] == "anti" and i["width"] >= 3 for i in infos[:cut])
                    and any(i["orientation"] == "tl" and i["width"] == 1 for i in infos[:cut])
                    and any(i["orientation"] == "tl" and i["width"] >= 3 for i in infos[cut:]))
        for info in infos:
            backgrounds.add(info["background"])
            widths.add(info["width"])
            periods.add(info["period"])
            sizes.add(info["size"])
            layouts.add((info["size"], info["orientation"], info["support"],
                         tuple(tuple(sorted(p)) for p in info["profiles"])))
            repeated_phases += len({tuple(sorted(p.items())) for p in info["profiles"]}) < info["period"]
            missing_centers += any(0 not in p for p in info["profiles"])
            supports = [frozenset(p) for p in info["profiles"]]
            varying_support += len(set(supports)) > 1
            mixed_lengths += {2, 3}.issubset({len(p) for p in info["profiles"]})
            left_right += (info["width"] >= 3 and any(
                min(a) < 0 and max(a) <= 0 and min(b) >= 0 and max(b) > 0
                for a in supports for b in supports))
            wide += info["width"] >= 4
            aab += (info["period"] == 3 and info["profiles"][0] == info["profiles"][1]
                    and info["profiles"][1] != info["profiles"][2])
            abac += (info["period"] == 4 and info["profiles"][0] == info["profiles"][2]
                     and info["profiles"][0] != info["profiles"][1]
                     and info["profiles"][1] != info["profiles"][3])
            clipped += any(not 0 <= _base(info["size"], row, info["orientation"]) + offset < info["size"]
                           for row in range(info["size"])
                           for offset in info["profiles"][row % info["period"]])
    _ok(len(signatures) >= 40 and len(train_counts) >= 4 and len(test_counts) >= 3,
        "count or puzzle diversity")
    _ok(len(backgrounds) == 10 and widths == set(range(1, 6))
        and periods == set(range(1, 5)), "color, width, or period coverage")
    _ok(len(sizes) >= 9 and len(layouts) >= 80 and len(orientations) >= 2,
        "structural diversity")
    _ok(palette_changes >= 25 and shape_changes >= 40 and novel_colors >= 3
        and smaller_tests >= 15 and repeated_phases >= 5 and missing_centers >= 3,
        "review coverage")
    _ok(varying_support >= 50 and mixed_lengths >= 10 and left_right >= 5
        and wide >= 30 and clipped >= 80 and aab >= 5 and abac >= 5,
        "profile geometry and rhythm coverage")
    _ok(larger_tests >= 5 and source_like >= 5 and reverse >= 5
        and min(thin_tl, thin_anti, thick_anti, test_tl) >= 5
        and normalized_repeats <= 8, "regime and lesson coverage")
    for seed, tr, te in ((2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
                         (2004, 5, 2), (2101, None, 1), (2102, None, 4),
                         (2103, 2, None), (2104, 10, None)):
        puzzle = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(puzzle)
        if tr is not None:
            _ok(len(puzzle["train"]) == tr, "train count ignored")
        if te is not None:
            _ok(len(puzzle["test"]) == te, "test count ignored")
    _ok(_puzzle_signature(generate_puzzle(3001, 7, 3)) ==
        _puzzle_signature(generate_puzzle(3001, 7, 3)), "not deterministic")
    _ok(len({_puzzle_signature(generate_puzzle(s, 4, 2)) for s in range(3100, 3112)}) >= 11,
        "cross-seed repetition")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(4001, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {arguments}")
