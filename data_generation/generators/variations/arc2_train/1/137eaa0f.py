import numpy as np


# The four source pairs are protected against exact reproduction.
_SOURCE_SIGNATURES = {
    (("00000000000", "00000000000", "06000000000", "00507570000", "00000000000", "00000000000", "00000000000", "00000005000", "00220033300", "00500000000", "00000000000"), ("622", "757", "333")),
    (("00000000000", "00000066000", "00050005000", "00440000000", "00000000000", "00000000000", "00000000000", "00000070000", "00000570000", "00000000000", "00000000000"), ("667", "057", "440")),
    (("00000000000", "00000000000", "01100000000", "15000000000", "00000520000", "00000020000", "00000000000", "00000000000", "00000005000", "00000099000", "00000000000"), ("011", "152", "992")),
    (("00000000000", "00000000900", "00000009500", "40000009000", "05400000000", "00000000000", "00000008000", "00000050000", "00000000000", "00050000000", "00022000000"), ("498", "954", "922")),
}
_RING = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _lines(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
                raise ValueError(name + " is outside its legal range")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed + 3 * (seed // 4)) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _partition(rng, regime):
    # The positions are selected first. Cuts partition those very positions,
    # so assembly is exact by construction, with no repair or movement rule.
    if rng.random() < .09:
        edges = [(a, b) for a in _RING for b in _RING
                 if a < b and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
        chosen = []
        used = set()
        for index in rng.permutation(len(edges)):
            edge = edges[int(index)]
            if not any(cell in used for cell in edge):
                chosen.append(tuple(sorted(edge)))
                used.update(edge)
                if len(chosen) == 3:
                    rng.shuffle(chosen)
                    return chosen
    if regime == "source_train":
        n = int(rng.choice(np.arange(3, 9), p=[.08, .14, .19, .23, .20, .16]))
    else:
        n = int(rng.choice(np.arange(3, 9), p=[.04, .08, .14, .21, .24, .29]))
    k = int(rng.integers(2, n + 1))
    if regime == "target_test" and n >= 5 and rng.random() < .36:
        k = int(rng.integers(4, n + 1))
    bar = None
    if n >= k + 2 and rng.random() < .34:
        bars = [tuple(sorted(((-1, -1), (-1, 0), (-1, 1)))),
                tuple(sorted(((1, -1), (1, 0), (1, 1)))),
                tuple(sorted(((-1, -1), (0, -1), (1, -1)))),
                tuple(sorted(((-1, 1), (0, 1), (1, 1))))]
        bar = bars[int(rng.integers(0, len(bars)))]
        available = [cell for cell in _RING if cell not in bar]
        occupied = list(bar) + [tuple(available[int(i)]) for i in
                                rng.choice(len(available), size=n - 3, replace=False)]
    else:
        occupied = [tuple(_RING[int(i)]) for i in rng.choice(8, size=n, replace=False)]
    rng.shuffle(occupied)
    if bar is not None:
        occupied = list(bar) + [cell for cell in occupied if cell not in bar]
        if n - 3 == k - 1:
            sizes = [3] + [1] * (k - 1)
        else:
            cuts = sorted(int(i) for i in rng.choice(np.arange(1, n - 3), size=k - 2, replace=False))
            sizes = [3] + [b - a for a, b in zip([0] + cuts, cuts + [n - 3])]
    elif rng.random() < .23 and n - k + 1 >= 3:
        sizes = [n - k + 1] + [1] * (k - 1)
        rng.shuffle(sizes)
    else:
        cuts = sorted(int(i) for i in rng.choice(np.arange(1, n), size=k - 1, replace=False))
        sizes = [b - a for a, b in zip([0] + cuts, cuts + [n])]
    pieces = []
    start = 0
    for size in sizes:
        pieces.append(tuple(sorted(occupied[start:start + size])))
        start += size
    rng.shuffle(pieces)
    return pieces


def _canonical_partition(pieces):
    variants = []
    for flip in (False, True):
        for turns in range(4):
            transformed = []
            for piece in pieces:
                cells = []
                for r, c in piece:
                    if flip:
                        c = -c
                    for _ in range(turns):
                        r, c = c, -r
                    cells.append((r, c))
                transformed.append(tuple(sorted(cells)))
            variants.append(tuple(sorted(transformed)))
    return min(variants)


def _colors(rng, k, background, marker):
    available = [c for c in range(10) if c not in (background, marker)]
    if k > 1 and rng.random() < .44:
        palette_size = int(rng.integers(1, min(k - 1, len(available)) + 1))
        palette = [int(c) for c in rng.choice(available, size=palette_size, replace=False)]
        result = [int(palette[int(i)]) for i in rng.integers(0, palette_size, size=k)]
        result[0] = result[1]
        rng.shuffle(result)
        return result
    return [int(c) for c in rng.choice(available, size=k, replace=k > len(available))]


def _dimensions(rng, k):
    mode = int(rng.integers(0, 5))
    if mode == 0:
        h, w = int(rng.integers(6, 12)), int(rng.integers(6, 12))
    elif mode == 1:
        h, w = int(rng.integers(10, 21)), int(rng.integers(10, 21))
    elif mode == 2:
        h, w = int(rng.integers(18, 31)), int(rng.integers(18, 31))
    elif mode == 3:
        h, w = int(rng.integers(6, 12)), int(rng.integers(17, 31))
        if rng.random() < .5:
            h, w = w, h
    else:
        h, w = 30, int(rng.integers(9, 31))
        if rng.random() < .5:
            h, w = w, h
    # Many groups need enough room to remain visually separate.
    if k >= 6 and min(h, w) < 9:
        h = max(h, 10)
        w = max(w, 10)
    return h, w


def _scatter(rng, pieces, colors, background, marker):
    edge_mode = int(rng.choice(3, p=[.38, .27, .35]))  # edge, interior, free
    order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]))
    for attempt in range(44):
        h, w = _dimensions(rng, len(pieces)) if attempt < 38 else (30, 30)
        groups = [None] * len(pieces)
        markers = [None] * len(pieces)
        failed = False
        for step, index in enumerate(order):
            piece = pieces[index]
            all_candidates = []
            for r in range(h):
                for c in range(w):
                    cells = ((r, c),) + tuple((r + dr, c + dc) for dr, dc in piece)
                    if not all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                        continue
                    if edge_mode == 1 and not all(0 < rr < h - 1 and 0 < cc < w - 1 for rr, cc in cells):
                        continue
                    if edge_mode == 0 and step == 0 and not any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in cells):
                        continue
                    all_candidates.append((r, c, cells))
            if not all_candidates:
                failed = True
                break
            chosen = None
            for j in rng.permutation(len(all_candidates)):
                r, c, cells = all_candidates[int(j)]
                if all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2
                       for old in groups if old is not None for a in cells for b in old):
                    chosen = (r, c, cells)
                    break
            if chosen is None:
                failed = True
                break
            markers[index] = (chosen[0], chosen[1])
            groups[index] = chosen[2]
        if failed:
            continue
        inp = np.full((h, w), background, dtype=np.int8)
        for index, ((r, c), piece, color) in enumerate(zip(markers, pieces, colors)):
            inp[r, c] = marker
            for dr, dc in piece:
                inp[r + dr, c + dc] = color
        return inp
    raise RuntimeError("could not scatter the complete fragments")


def _make_pair(rng, regime, background, marker):
    pieces = _partition(rng, regime)
    colors = _colors(rng, len(pieces), background, marker)
    out = np.full((3, 3), background, dtype=np.int8)
    out[1, 1] = marker
    for piece, color in zip(pieces, colors):
        for r, c in piece:
            assert out[1 + r, 1 + c] == background
            out[1 + r, 1 + c] = color
    inp = _scatter(rng, pieces, colors, background, marker)
    return {"input": inp, "output": out}, _canonical_partition(pieces)


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= x <= 30 for x in grid.shape)
    assert 0 <= int(grid.min()) and int(grid.max()) <= 9


def _analyze_input(inp, marker):
    counts = np.bincount(inp.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert background != marker
    assert counts[background] > max(int(v) for i, v in enumerate(counts) if i != background)
    marks = [tuple(map(int, pos)) for pos in np.argwhere(inp == marker)]
    assert 2 <= len(marks) <= 8
    remaining = {tuple(map(int, pos)) for pos in np.argwhere(inp != background)}
    components = []
    while remaining:
        component = set()
        frontier = [next(iter(remaining))]
        while frontier:
            cell = frontier.pop()
            if cell not in remaining:
                continue
            remaining.remove(cell)
            component.add(cell)
            frontier.extend(other for other in remaining
                            if max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) == 1)
        components.append(component)
    assert len(components) == len(marks)
    group_cells = []
    payload = []
    for mark in marks:
        owners = [group for group in components if mark in group]
        assert len(owners) == 1
        group = owners[0]
        assert sum(pos in group for pos in marks) == 1
        group_cells.append(group)
        payload.append(sorted(group - {mark}))
    assert all(payload)
    assert all(len({int(inp[cell]) for cell in part}) == 1 for part in payload)
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            assert min(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                       for a in group_cells[i] for b in group_cells[j]) >= 2
    pieces = [tuple(sorted((r - mark[0], c - mark[1]) for r, c in part))
              for mark, part in zip(marks, payload)]
    colors = [int(inp[part[0]]) for part in payload]
    cells = [cell for piece in pieces for cell in piece]
    assert len(cells) <= 8 and len(set(cells)) == len(cells)
    assert all(cell in _RING for cell in cells)
    out = np.full((3, 3), background, dtype=np.int8)
    out[1, 1] = marker
    for piece, color in zip(pieces, colors):
        for r, c in piece:
            out[1 + r, 1 + c] = color
    return {"background": background, "marker": marker, "pieces": pieces,
            "colors": colors, "output": out, "markers": marks, "groups": group_cells,
            "occupied": len(cells), "partition": _canonical_partition(pieces),
            "reused_color": len(set(colors)) < len(colors),
            "min_separation": min(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                                  for i in range(len(marks)) for j in range(i + 1, len(marks))
                                  for a in group_cells[i] for b in group_cells[j])}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    first = puzzle["train"][0]
    assert isinstance(first, dict) and set(first) == {"input", "output"}
    _check_grid(first["output"])
    assert first["output"].shape == (3, 3)
    marker = int(first["output"][1, 1])
    inputs, pairs = set(), set()
    train_patterns = set()
    train_backgrounds = set()
    train_rule_examples = 0
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            assert out.shape == (3, 3) and inp.shape != out.shape
            assert int(out[1, 1]) == marker
            ikey, pkey = _key(inp), (_key(inp), _key(out))
            assert ikey not in inputs and pkey not in pairs
            inputs.add(ikey)
            pairs.add(pkey)
            assert (_lines(inp), _lines(out)) not in _SOURCE_SIGNATURES
            analysis = _analyze_input(inp, marker)
            assert np.array_equal(analysis["output"], out)
            if section == "train":
                train_patterns.add(analysis["partition"])
                train_backgrounds.add(analysis["background"])
                train_rule_examples += analysis["occupied"] < 8 and max(map(len, analysis["pieces"])) >= 3
            else:
                assert analysis["partition"] not in train_patterns
    assert len(train_backgrounds) >= 2
    assert train_rule_examples >= 1


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, ns = _counts(seed, rng, num_train, num_test)
    marker = int(rng.integers(0, 10))
    backgrounds = [int(c) for c in rng.permutation([c for c in range(10) if c != marker])]
    puzzle = {"train": [], "test": []}
    seen_inputs, train_patterns = set(), set()
    for section, count, regime in (("train", nt, "source_train"), ("test", ns, "target_test")):
        for index in range(count):
            background = backgrounds[index % len(backgrounds)]
            if section == "test":
                background = backgrounds[(nt + index) % len(backgrounds)]
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(100):
                pair, pattern = _make_pair(pair_rng, regime, background, marker)
                if section == "train" and index == 0 and not (sum(map(len, pattern)) < 8 and max(map(len, pattern)) >= 3):
                    continue
                if _key(pair["input"]) in seen_inputs or (_lines(pair["input"]), _lines(pair["output"])) in _SOURCE_SIGNATURES:
                    continue
                if section == "test" and pattern in train_patterns:
                    continue
                seen_inputs.add(_key(pair["input"]))
                puzzle[section].append(pair)
                if section == "train":
                    train_patterns.add(pattern)
                break
            else:
                raise RuntimeError("could not create a distinct pair")
        if section == "train":
            rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[t], y[t]) for x, y in zip(a[s], b[s]) for t in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(12345)
    assert _equal(a, generate_puzzle(12345))
    b = generate_puzzle(314159, 6, 2)
    assert _equal(b, generate_puzzle(314159, 6, 2))
    assert not _equal(b, generate_puzzle(314160, 6, 2))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(5000 + i, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, roles, shapes, partitions = set(), set(), set(), set(), set()
    group_counts, occupancies, size_patterns, backgrounds, holes = set(), set(), set(), set(), set()
    reused = edges = interiors = close = large = compact = full = 0
    dominoes = bars = bends = diagonals = disconnected = square = narrow = 0
    combined_compact = combined_full = 0
    within_occupancy = within_groups = within_backgrounds = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        marker = int(puzzle["train"][0]["output"][1, 1])
        roles.add(marker)
        puzzle_occupancy, puzzle_groups, puzzle_backgrounds = set(), set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                a = _analyze_input(inp, marker)
                # Independent cell-by-cell registration check, with no
                # access to generation-time pieces, shifts, or metadata.
                reconstruction = np.full((3, 3), a["background"], dtype=np.int8)
                reconstruction[1, 1] = marker
                for mark, group in zip(a["markers"], a["groups"]):
                    for r, c in group:
                        if (r, c) != mark:
                            rr, cc = 1 + r - mark[0], 1 + c - mark[1]
                            assert 0 <= rr < 3 and 0 <= cc < 3
                            assert reconstruction[rr, cc] == a["background"]
                            reconstruction[rr, cc] = inp[r, c]
                assert np.array_equal(reconstruction, out)
                total += 1
                shapes.add(inp.shape)
                partitions.add(a["partition"])
                group_counts.add(len(a["markers"]))
                occupancies.add(a["occupied"])
                puzzle_groups.add(len(a["markers"]))
                puzzle_occupancy.add(a["occupied"])
                puzzle_backgrounds.add(a["background"])
                sizes = tuple(sorted(map(len, a["pieces"])))
                size_patterns.add(sizes)
                holes.add(frozenset(_RING) - frozenset(cell for piece in a["pieces"] for cell in piece))
                backgrounds.add(a["background"])
                reused += a["reused_color"]
                close += a["min_separation"] == 2
                large += max(map(len, a["pieces"])) >= 5
                dominoes += sizes == (2, 2, 2)
                bars += any(len(piece) >= 3 and (len({r for r, c in piece}) == 1 or
                              len({c for r, c in piece}) == 1) for piece in a["pieces"])
                bends += any(len(piece) >= 3 and len({r for r, c in piece}) > 1 and
                             len({c for r, c in piece}) > 1 for piece in a["pieces"])
                diagonals += any(abs(r) == abs(c) == 1 for piece in a["pieces"] for r, c in piece)
                disconnected += any(len(piece) > 1 and any(
                    all(abs(r - rr) + abs(c - cc) != 1 for rr, cc in piece if (rr, cc) != (r, c))
                    for r, c in piece) for piece in a["pieces"])
                h, w = inp.shape
                compact += max(h, w) <= 11
                full += h == 30 or w == 30
                square += h == w
                narrow += max(h, w) >= 2 * min(h, w)
                combined_compact += max(h, w) <= 11 and a["reused_color"] and a["min_separation"] == 2
                combined_full += (h == 30 or w == 30) and max(map(len, a["pieces"])) >= 5
                edge = any(r in (0, h - 1) or c in (0, w - 1) for group in a["groups"] for r, c in group)
                edges += edge
                interiors += not edge
        within_occupancy += len(puzzle_occupancy) > 1
        within_groups += len(puzzle_groups) > 1
        within_backgrounds += len(puzzle_backgrounds) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(roles) >= 8 and len(backgrounds) == 10
    assert group_counts == set(range(2, 9))
    assert occupancies == set(range(3, 9))
    assert len(shapes) >= 70 and len(partitions) >= 70 and len(size_patterns) >= 30
    assert len(holes) >= 30
    assert reused >= 30 and close >= 10 and large >= 5
    assert compact >= 3 and full >= 20 and edges >= 20 and interiors >= 20
    assert dominoes >= 5 and bars >= 10 and bends >= 20 and diagonals >= 30 and disconnected >= 20
    assert square >= 10 and narrow >= 20 and combined_compact >= 2 and combined_full >= 1
    assert within_occupancy >= 35 and within_groups >= 35 and within_backgrounds == 42
