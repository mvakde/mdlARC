import numpy as np


def _validate_count_argument(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    num_train = _validate_count_argument("num_train", num_train, 2, 10)
    num_test = _validate_count_argument("num_test", num_test, 1, 4)
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _array_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def _draw(height, width, row, starts, widths, colors, spans, bg, carrier):
    grid = np.full((height, width), bg, dtype=np.int8)
    grid[row, :] = carrier
    for x, w, color, (lo, hi) in zip(starts, widths, colors, spans):
        grid[row + lo:row + hi + 1, x:x + w] = color
    return grid


def _mode(values):
    counts = np.bincount(values.ravel(), minlength=10)
    color = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[color]) == 1
    return color


def _analyze_input(grid):
    """Recover roles and slots from cells, including touching color runs."""
    bg = _mode(grid)
    rows = np.flatnonzero(np.all(grid != bg, axis=1))
    cols = np.flatnonzero(np.all(grid != bg, axis=0))
    assert len(rows) + len(cols) == 1, "carrier must be geometrically unique"
    vertical = bool(len(cols))
    canonical = grid.T if vertical else grid
    row = int(cols[0] if vertical else rows[0])
    line = canonical[row]
    carrier = _mode(line)
    assert carrier != bg
    assert np.count_nonzero(canonical == carrier) == np.count_nonzero(line == carrier)
    assert np.count_nonzero(line == carrier) >= 3
    rebuilt = np.full(canonical.shape, bg, dtype=np.int8)
    rebuilt[row] = carrier
    records = []
    x = 0
    while x < len(line):
        color = int(line[x])
        end = x + 1
        while end < len(line) and line[end] == color:
            end += 1
        if color != carrier:
            assert color != bg
            occupied = np.flatnonzero(np.any(canonical[:, x:end] != bg, axis=1))
            top, bottom = int(occupied[0]), int(occupied[-1])
            assert top <= row <= bottom
            assert np.all(canonical[top:bottom + 1, x:end] == color)
            rebuilt[top:bottom + 1, x:end] = color
            records.append((x, end - x, color, top - row, bottom - row))
        x = end
    assert 2 <= len(records) <= 8
    assert len({r[2] for r in records}) >= 2
    assert np.array_equal(rebuilt, canonical), "unexplained or nonsolid cells"
    templates = {}
    for _, _, _, lo, hi in records:
        length = hi - lo + 1
        assert templates.setdefault(length, (lo, hi)) == (lo, hi), "ambiguous offset tie"
    return {"background": bg, "carrier": carrier, "vertical": vertical,
            "row": row, "shape": canonical.shape, "records": records}


def _check_pair(pair):
    """Relational check without the construction's span sorter or painter."""
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    a, b = _analyze_input(inp), _analyze_input(out)
    for key in ("background", "carrier", "vertical", "row", "shape"):
        assert a[key] == b[key]
    assert [r[:3] for r in a["records"]] == [r[:3] for r in b["records"]]
    remaining = [r[3:] for r in a["records"]]
    previous = 0
    for record in b["records"]:
        span = record[3:]
        length = span[1] - span[0] + 1
        assert length >= previous, "output lengths must be nondecreasing"
        assert span in remaining, "the complete input interval must transfer"
        remaining.remove(span)
        previous = length
    assert not remaining
    return a, b


def _offset_kinds(span):
    lo, hi = span
    length = hi - lo + 1
    kinds = set()
    if length % 2 == 0 and abs(lo + hi) == 1:
        kinds.add("even_upper" if lo + hi < 0 else "even_lower")
    if abs(lo + hi) > 1:
        kinds.add("offset_upper" if lo + hi < 0 else "offset_lower")
    if length >= 3 and (lo == 0 or hi == 0):
        kinds.add("one_lower" if lo == 0 else "one_upper")
    return kinds


def _pair_features(pair):
    a, b = _check_pair(pair)
    records, result = a["records"], b["records"]
    lengths = [r[4] - r[3] + 1 for r in records]
    widths = [r[1] for r in records]
    colors = [r[2] for r in records]
    spans = [r[3:] for r in records]
    final = [r[3:] for r in result]
    order = ("sorted" if lengths == sorted(lengths) else
             "decreasing" if lengths == sorted(lengths, reverse=True) else
             "nearly_sorted" if sum(x != y for x, y in zip(lengths, sorted(lengths))) == 2
             else "permuted")
    area_spans = [records[i][3:] for i in sorted(range(len(records)),
                  key=lambda i: lengths[i] * widths[i])]
    used, moved = set(), set()
    for span, target in zip(spans, final):
        used.update(_offset_kinds(span))
        if span != target:
            moved.update(_offset_kinds(span))
    gaps = [right[0] - left[0] - left[1] for left, right in zip(records, records[1:])]
    return {"count": len(records), "widths": tuple(widths), "lengths": tuple(lengths),
            "gaps": tuple(gaps), "mixed_widths": len(set(widths)) > 1,
            "touching": 0 in gaps, "repeated_accents": len(set(colors)) < len(colors),
            "palette": tuple(sorted(set(colors))), "ties": len(set(lengths)) < len(lengths),
            "unit": 1 in lengths, "even_count": sum(n % 2 == 0 for n in lengths),
            "order": order, "area_disagrees": area_spans != final,
            "reversal_disagrees": list(reversed(spans)) != final,
            "changed": spans != final, "offsets": tuple(sorted(used)),
            "moved_offsets": tuple(sorted(moved)), "vertical": a["vertical"],
            "background": a["background"], "carrier": a["carrier"],
            "carrier_fraction": a["row"] / (a["shape"][0] - 1),
            "shape": tuple(int(n) for n in pair["input"].shape),
            "layout": (a["vertical"], a["shape"], a["row"],
                       tuple((r[0], r[1], r[3], r[4]) for r in records))}


def _lengths_and_spans(rng, n, policy, cap, witness):
    if policy == "odd":
        cap = max(cap, 2 * n - 1) if witness else cap
        available = list(range(1, min(27, cap) + 1, 2))
    else:
        cap = min(13, cap) if policy == "free" else min(27, cap)
        cap = max(cap, n + 2, 6) if witness else max(cap, 4)
        available = list(range(1, cap + 1))
    special = []
    if witness and policy != "odd":
        even = [x for x in available if x % 2 == 0 and (policy != "free" or x >= 4)]
        special = [int(x) for x in rng.choice(even, size=2, replace=False)]
    if witness:
        rest = [x for x in available if x not in special]
        lengths = special + [int(x) for x in rng.choice(rest, n - len(special), replace=False)]
    else:
        distinct = min(n, len(available))
        if n >= 3 and rng.random() < 0.45:
            distinct = int(rng.integers(2, distinct + (distinct < n)))
        selected = [int(x) for x in rng.choice(available, distinct, replace=False)]
        lengths = selected + [int(x) for x in rng.choice(selected, n - distinct)]
    templates = {}
    for length in set(lengths):
        if policy == "odd" or (length % 2 and (policy != "free" or rng.random() < 0.3)):
            lo = -(length // 2)
        elif policy == "balanced":
            lo = -(length // 2) + int(rng.integers(0, 2))
        else:
            lo = -int(rng.integers(0, length))
        templates[length] = (lo, lo + length - 1)
    if special:
        upper, lower = special
        if policy == "balanced":
            templates[upper] = (-upper // 2, upper // 2 - 1)
            templates[lower] = (-lower // 2 + 1, lower // 2)
        else:
            templates[upper], templates[lower] = (1 - upper, 0), (0, lower - 1)
    ordered = sorted(lengths)
    if witness:
        for _ in range(24):
            rng.shuffle(lengths)
            if (lengths != ordered and lengths[::-1] != ordered and
                    all(lengths.index(x) != ordered.index(x) for x in special)):
                break
        else:
            lengths = ordered[1:] + ordered[:1]
    else:
        mode = int(rng.choice(4, p=[0.55, 0.22, 0.17, 0.06]))
        lengths = ordered.copy()
        if mode == 0:
            rng.shuffle(lengths)
        elif mode == 1:
            swaps = [i for i in range(n - 1) if lengths[i] != lengths[i + 1]]
            i = int(rng.choice(swaps))
            lengths[i], lengths[i + 1] = lengths[i + 1], lengths[i]
        elif mode == 2:
            lengths.reverse()
    return lengths, [templates[x] for x in lengths]


def _sample_pair(rng, config, split, vertical, witness=0, count=None, area=False):
    # Size tendencies overlap deliberately; fitting can expand either dimension.
    regime = config["scale"]
    small = ((regime == "grow" and split == "source_train") or
             (regime == "shrink" and split == "target_test"))
    large = ((regime == "grow" and split == "target_test") or
             (regime == "shrink" and split == "source_train"))
    compact = not witness and not large and rng.random() < 0.24
    low, high = (7, 10) if compact else (7, 16) if small else (20, 30) if large else (7, 30)
    height, width = [int(x) for x in rng.integers(low, high + 1, size=2)]
    square = rng.random() < (0.65 if compact else 0.28)
    if not small and not compact and rng.random() < 0.08:
        height = width = 30
        square = True
    n = count or int(rng.integers(2, 4 if compact else 9))
    policy = config["spans"]
    # Free-span puzzles show both even biases in one witness and strong,
    # one-sided intervals in the other. All are visibly reassigned.
    if policy == "free" and witness == 1:
        policy = "balanced"
    if not witness and rng.random() < 0.28:
        policy = "odd"
    lengths, spans = _lengths_and_spans(rng, n, policy, max(5, height - 2), witness)
    if config["fixed_roles"]:
        bg, carrier = config["roles"]
        pool = [c for c in range(10) if c not in (bg, carrier)]
    else:
        pool = list(range(10))
    if rng.random() < 0.45:
        palette = config["palette"][:min(n, len(config["palette"]))]
    else:
        size = int(rng.integers(2, min(n, 6) + 1))
        palette = [int(x) for x in rng.choice(pool, size, replace=False)]
    if not config["fixed_roles"]:
        bg, carrier = [int(x) for x in rng.choice([c for c in range(10) if c not in palette], 2, replace=False)]
    colors = list(palette) + [int(x) for x in rng.choice(palette, n - len(palette))]
    rng.shuffle(colors)
    uniform = rng.random() > config["complexity"] or compact
    widths = ([int(rng.integers(1, 4))] * n if uniform else
              [int(x) for x in rng.integers(1, 6, size=n)])
    if compact:
        widths = [1] * n
    locked = {}
    if area:
        values = sorted(set(lengths))
        short, long = min(zip(values, values[1:]), key=lambda p: p[1] / p[0])
        narrow, wide = lengths.index(long), lengths.index(short)
        locked = {narrow: 1, wide: long // short + 1}
        for i, w in locked.items():
            widths[i] = w
    spacing = float(rng.choice([0.0, 0.4, 0.85]))
    gaps = [0] + [int(colors[i] == colors[i + 1] or rng.random() >= spacing)
                  for i in range(n - 1)] + [0]

    def minimum_exposure():
        # A unit bar's color cannot be mistaken for the exposed carrier.
        by_color = [sum(w for w, c in zip(widths, colors) if c == color) for color in palette]
        return max(3, max(by_color) + 1, sum(gaps))

    while sum(widths) + minimum_exposure() > 30:
        reducible = [i for i in range(n) if widths[i] > locked.get(i, 1)]
        i = max(reducible, key=lambda j: widths[j])
        widths[i] -= 1
    width = max(width, sum(widths) + minimum_exposure())
    min_span, max_span = min(s[0] for s in spans), max(s[1] for s in spans)
    height = max(height, max_span - min_span + 3)
    assert height <= 30
    if square:
        height = width = max(height, width)
    extra = width - sum(widths) - sum(gaps)
    eligible = [i for i in range(n + 1) if i in (0, n) or gaps[i]]
    allocation = rng.multinomial(extra, [1 / len(eligible)] * len(eligible))
    for i, amount in zip(eligible, allocation):
        gaps[i] += int(amount)
    starts, position = [], gaps[0]
    for i in range(n):
        starts.append(position)
        position += widths[i] + gaps[i + 1]
    row = int(rng.integers(1 - min_span, height - 1 - max_span))
    inp = _draw(height, width, row, starts, widths, colors, spans, bg, carrier)
    assigned = [spans[int(i)] for i in np.argsort(lengths, kind="stable")]
    out = _draw(height, width, row, starts, widths, colors, assigned, bg, carrier)
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _source_pairs():
    return [{"input": np.array([[int(c) for c in row] for row in inp.split()], dtype=np.int8),
             "output": np.array([[int(c) for c in row] for row in out.split()], dtype=np.int8)}
            for inp, out in _SOURCE_ROWS]


def _avoid_duplicate(pair, seen, source_keys):
    key = (_array_key(pair["input"]), _array_key(pair["output"]))
    if key[0] not in seen and key not in source_keys:
        return pair
    # 90 distinct role assignments and fewer than 18 exclusions guarantee a
    # terminating fallback. Geometry and all slot-color relationships survive.
    a = _analyze_input(pair["input"])
    old = [a["background"], a["carrier"]] + sorted({r[2] for r in a["records"]})
    for bg in range(10):
        for carrier in range(10):
            if bg == carrier:
                continue
            new = [bg, carrier] + [c for c in range(10) if c not in (bg, carrier)]
            mapping = np.arange(10, dtype=np.int8)
            for x, y in zip(old, new):
                mapping[x] = y
            candidate = {role: mapping[pair[role]] for role in ("input", "output")}
            key = (_array_key(candidate["input"]), _array_key(candidate["output"]))
            if key[0] not in seen and key not in source_keys:
                return candidate
    raise AssertionError("finite coloring fallback exhausted")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(rng, num_train, num_test)
    complexity = float(rng.choice([0.25, 0.65, 0.9], p=[0.18, 0.55, 0.27]))
    roles = [int(x) for x in rng.choice(10, 2, replace=False)]
    pool = [c for c in range(10) if c not in roles]
    config = {"complexity": complexity,
              "spans": str(rng.choice(["odd", "balanced", "free"], p=[0.25, 0.45, 0.3])),
              "scale": str(rng.choice(["grow", "shrink", "overlap"], p=[0.25, 0.2, 0.55])),
              "fixed_roles": bool(rng.random() < 0.2), "roles": roles,
              "palette": [int(x) for x in rng.choice(pool, int(rng.integers(2, 4)), replace=False)]}
    mode = str(rng.choice(["horizontal", "vertical", "mixed", "shift"], p=[0.3, 0.2, 0.3, 0.2]))
    if mode in ("horizontal", "vertical"):
        train_axes, test_axes = [mode == "vertical"] * nt, [mode == "vertical"] * nq
    else:
        train_axes = [False, True] + [bool(rng.integers(0, 2)) for _ in range(nt - 2)]
        if mode == "shift":
            target = bool(rng.integers(0, 2))
            train_axes[2:] = [not target] * (nt - 2)
            test_axes = [target] * nq
        else:
            test_axes = [bool(rng.integers(0, 2)) for _ in range(nq)]
            if nq > 1:
                test_axes[:2] = [False, True]
    # Compositional witnesses establish direction, full interval transfer and
    # count variation even with two demonstrations. Shuffle their display order.
    first_count = int(rng.integers(3, 9))
    second_count = int(rng.choice([n for n in range(3, 9) if n != first_count]))
    puzzle = {"train": [], "test": []}
    seen = set()
    source_keys = {(_array_key(p["input"]), _array_key(p["output"])) for p in _source_pairs()}
    for split, regime, axes in (("train", "source_train", train_axes), ("test", "target_test", test_axes)):
        for i, vertical in enumerate(axes):
            witness = i + 1 if split == "train" and i < 2 else 0
            count = [first_count, second_count][i] if witness else None
            pair = _sample_pair(rng, config, regime, vertical, witness, count, area=witness == 1)
            pair = _avoid_duplicate(pair, seen, source_keys)
            puzzle[split].append(pair)
            seen.add(_array_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    source = {(_array_key(p["input"]), _array_key(p["output"])) for p in _source_pairs()}
    seen, pairs, features = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = (_array_key(pair["input"]), _array_key(pair["output"]))
            assert key[0] not in seen and key not in pairs and key not in source
            seen.add(key[0])
            pairs.add(key)
            features[split].append(_pair_features(pair))
    train = features["train"]
    assert len({f["count"] for f in train}) >= 2
    assert sum(f["changed"] for f in train) >= 2
    assert any(f["area_disagrees"] for f in train), "training must rule out area sorting"
    assert any(f["changed"] and f["reversal_disagrees"] for f in train)
    demonstrated_axes = {f["vertical"] for f in train if f["changed"] and f["reversal_disagrees"]}
    demonstrated_offsets = {kind for f in train for kind in f["moved_offsets"]}
    for f in features["test"]:
        assert f["vertical"] in demonstrated_axes
        assert set(f["offsets"]) <= demonstrated_offsets, "test interval convention lacks demonstration"


def predict_from_training(train_pairs, test_input):
    """Prediction interface with no access to test labels or generation state."""
    axis_evidence, offset_evidence = set(), set()
    for pair in train_pairs:
        f = _pair_features(pair)
        if f["changed"]:
            axis_evidence.add(f["vertical"])
        offset_evidence.update(f["moved_offsets"])
    a = _analyze_input(test_input)
    assert a["vertical"] in axis_evidence
    intervals = [r[3:] for r in a["records"]]
    assert {k for span in intervals for k in _offset_kinds(span)} <= offset_evidence
    canonical = test_input.T if a["vertical"] else test_input
    out = np.full_like(canonical, a["background"])
    out[a["row"]] = canonical[a["row"]]
    # Selection by rank is independent of construction's argsort-and-draw path.
    for x, width, color, _, _ in a["records"]:
        shortest = min(intervals, key=lambda span: span[1] - span[0])
        intervals.remove(shortest)
        for offset in range(shortest[0], shortest[1] + 1):
            for col in range(x, x + width):
                out[a["row"] + offset, col] = color
    return out.T.copy() if a["vertical"] else out


def self_test() -> None:
    # Frozen source cells include touching bars, mixed widths, ties, and the
    # original even interval. No generated source-sized canvas is prohibited.
    for pair in _source_pairs():
        _check_pair(pair)
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzle_key(automatic[0]) == _puzzle_key(generate_puzzle(0))
    assert len({_puzzle_key(p) for p in automatic}) == 42
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    measured = {s: [] for s in ("train", "test")}
    within = []
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        rows = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _pair_features(pair)
                measured[split].append(f)
                rows.append(f)
                predicted = predict_from_training(puzzle["train"], pair["input"].copy())
                assert np.array_equal(predicted, pair["output"])
        within.append(rows)
    all_features = measured["train"] + measured["test"]
    assert len({f["shape"] for f in all_features}) >= 70
    assert len({f["layout"] for f in all_features}) >= 0.9 * len(all_features)
    assert {f["background"] for f in all_features} == set(range(10))
    assert {f["carrier"] for f in all_features} == set(range(10))
    assert {w for f in all_features for w in f["widths"]} >= {1, 2, 3, 4, 5}
    assert (30, 30) in {f["shape"] for f in all_features}
    assert any(f["shape"][0] == f["shape"][1] <= 10 for f in all_features)
    assert max(f["carrier_fraction"] for f in all_features) > 0.8
    assert min(f["carrier_fraction"] for f in all_features) < 0.2
    for split, rows in measured.items():
        assert {f["count"] for f in rows} == set(range(2, 9)), split
        assert {f["order"] for f in rows} == {"sorted", "decreasing", "nearly_sorted", "permuted"}
        assert {l for f in rows for l in f["lengths"]} >= {1, 2, 3, 4, 5, 6}
        assert {f["vertical"] for f in rows} == {False, True}
        for key in ("mixed_widths", "touching", "repeated_accents", "ties", "unit", "area_disagrees"):
            assert {f[key] for f in rows} == {False, True}, (split, key)
        assert any(f["even_count"] >= 2 for f in rows)
        assert {k for f in rows for k in f["offsets"]} == {
            "even_upper", "even_lower", "offset_upper", "offset_lower", "one_upper", "one_lower"}
        assert any(f["mixed_widths"] and f["touching"] and f["repeated_accents"] and f["offsets"] for f in rows)
    assert sum(len({f["background"] for f in rows}) > 1 for rows in within) >= 20
    assert sum(len({f["carrier"] for f in rows}) > 1 for rows in within) >= 20
    assert sum(len({f["vertical"] for f in rows}) > 1 for rows in within) >= 10
    assert any({f["palette"] for f in [_pair_features(p) for p in puzzle["train"]]} &
               {f["palette"] for f in [_pair_features(p) for p in puzzle["test"]]} for puzzle in automatic)
    means = [(np.mean([p["input"].size for p in puzzle["train"]]),
              np.mean([p["input"].size for p in puzzle["test"]])) for puzzle in automatic]
    assert any(a > b * 1.4 for a, b in means) and any(b > a * 1.4 for a, b in means)
    rejected = {"area": False, "reversal": False, "recenter": False}
    for puzzle in automatic:
        for pair in puzzle["train"]:
            a = _analyze_input(pair["input"])
            rs = a["records"]
            for mode in rejected:
                if rejected[mode]:
                    continue
                if mode == "area":
                    spans = [r[3:] for r in sorted(rs, key=lambda r: r[1] * (r[4] - r[3] + 1))]
                elif mode == "reversal":
                    spans = [r[3:] for r in rs[::-1]]
                else:
                    spans = [(-((r[4] - r[3] + 1) // 2), (r[4] - r[3]) // 2)
                             for r in sorted(rs, key=lambda r: r[4] - r[3])]
                if any(a["row"] + lo < 0 or a["row"] + hi >= a["shape"][0] for lo, hi in spans):
                    continue
                out = _draw(*a["shape"], a["row"], [r[0] for r in rs], [r[1] for r in rs],
                            [r[2] for r in rs], spans, a["background"], a["carrier"])
                if a["vertical"]:
                    out = out.T.copy()
                if np.array_equal(out, pair["output"]):
                    continue
                try:
                    _check_pair({"input": pair["input"], "output": out})
                except AssertionError:
                    rejected[mode] = True
                else:
                    raise AssertionError(f"accepted {mode} mutation")
    assert all(rejected.values())
    cases = [(2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None)]
    for i, (nt, nq) in enumerate(cases):
        p = generate_puzzle(9001 + i, nt, nq)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert nq is None or len(p["test"]) == nq
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(9001 + i, nt, nq))
    for nt, nq in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (4.5, 1)]:
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")


# Exact source cells, used only for regression and content exclusion.
_SOURCE_ROWS = (
    (
        '8888888888888 8888888888888 8588888888888 8588882888888 8588882788888 8588882718888 6566662716666 8588882718888 8588882788888 8588882888888 8588888888888 8888888888888 8888888888888',
        '8888888888888 8888888888888 8888888818888 8888888718888 8888882718888 8588882718888 6566662716666 8588882718888 8888882718888 8888888718888 8888888818888 8888888888888 8888888888888',
    ),
    (
        '888888888888888 888888888888888 888888888888988 888878888888988 888878888888988 884878588888988 884878581888988 664676561666966 884878581888988 884878588888988 888878888888988 888878888888988 888888888888988 888888888888888 888888888888888',
        '888888888888888 888888888888888 888888888888988 888888881888988 888888881888988 888878581888988 884878581888988 664676561666966 884878581888988 888878581888988 888888881888988 888888881888988 888888888888988 888888888888888 888888888888888',
    ),
    (
        '8888888 8818888 4818187 4616167 4818187 8818888 8888888',
        '8888888 8888887 4818187 4616167 4818187 8888887 8888888',
    ),
    (
        '88888888888888888 88888888888888888 88888888888888888 98888888888888888 98888888888888888 98555888888888888 98555882228888888 98555882228888888 96555662226666667 98555882228888888 98555882228888888 98555888888888888 98888888888888888 88888888888888888 88888888888888888 88888888888888888 88888888888888888',
        '88888888888888888 88888888888888888 88888888888888888 88888888888888887 88888888888888887 88888882228888887 88555882228888887 88555882228888887 96555662226666667 88555882228888887 88555882228888887 88888882228888887 88888888888888887 88888888888888888 88888888888888888 88888888888888888 88888888888888888',
    ),
)
