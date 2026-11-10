import numpy as np


_ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SOURCE = (
    (("500000", "003200", "000000", "030002", "020000", "500300", "000000"),
     ("500000", "008000", "000000", "080002", "000000", "500300", "000000")),
    (("320", "000", "050"), ("800", "000", "050")),
    (("0000020", "3000003", "5023000", "0000000", "0000020", "3200030", "0005000"),
     ("0000020", "3000003", "5008000", "0000000", "0000000", "8000080", "0005000")),
    (("000020005", "020000320", "030000000", "000050002", "500200000", "000000030", "530005020"),
     ("000020005", "000000800", "080000000", "000050002", "500200000", "000000080", "530005000")),
)
_COUNT_OPTIONS = tuple((a, b) for a in range(2, 11) for b in range(1, min(4, a - 1) + 1))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, seed, num_train, num_test):
    a = _count("num_train", num_train, 2, 10)
    b = _count("num_test", num_test, 1, 4)
    if a is None and b is None:
        a, b = _COUNT_OPTIONS[int(seed) % len(_COUNT_OPTIONS)]
    elif a is None:
        a = int(rng.integers(max(2, b + 1), 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _neighbors(p, height, width, deltas=_ORTHO):
    r, c = p
    for dr, dc in deltas:
        q = (r + dr, c + dc)
        if 0 <= q[0] < height and 0 <= q[1] < width:
            yield q


def _cross_degrees(cells, height, width):
    for p, role in cells.items():
        if role not in (1, 2):
            continue
        opposite = 3 - role
        if sum(cells.get(q) == opposite for q in _neighbors(p, height, width)) > 1:
            return False
    return True


def _can_add(cells, additions, height, width):
    if any(p in cells or not (0 <= p[0] < height and 0 <= p[1] < width)
           for p in additions):
        return False
    trial = cells | additions
    return _cross_degrees(trial, height, width)


def _try_motif(rng, cells, height, width, motif, orientation):
    if motif == "chain":
        pattern = ((0, 0, 1), (0, 1, 1), (0, 2, 2))
    else:
        pattern = ((0, 0, 1), (0, 1, 2), (1, 0, 1), (1, 1, 2))
    if orientation == "V":
        pattern = tuple((c, r, role) for r, c, role in pattern)
    if bool(rng.integers(0, 2)):
        maximum = max(c for _, c, _ in pattern)
        pattern = tuple((r, maximum - c, role) for r, c, role in pattern)
    if bool(rng.integers(0, 2)):
        maximum = max(r for r, _, _ in pattern)
        pattern = tuple((maximum - r, c, role) for r, c, role in pattern)
    ph = max(r for r, _, _ in pattern) + 1
    pw = max(c for _, c, _ in pattern) + 1
    candidates = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
    rng.shuffle(candidates)
    for r, c in candidates:
        additions = {(r + dr, c + dc): role for dr, dc, role in pattern}
        if _can_add(cells, additions, height, width):
            cells.update(additions)
            return True
    return False


def _add_domino(rng, cells, reserved, locked, height, width, orientation=None):
    candidates = []
    orientations = (orientation,) if orientation else ("H", "V")
    for direction in orientations:
        dr, dc = (0, 1) if direction == "H" else (1, 0)
        for r in range(height - dr):
            for c in range(width - dc):
                p, q = (r, c), (r + dr, c + dc)
                for e, t in ((p, q), (q, p)):
                    additions = {e: 1, t: 2}
                    if p in reserved or q in reserved:
                        continue
                    if _can_add(cells, additions, height, width):
                        trial = cells | additions
                        if any(
                            any(trial.get(n) == 3 - cells[point]
                                for n in _neighbors(point, height, width))
                            for point in locked
                        ):
                            continue
                        same_role_contacts = sum(
                            cells.get(n) == role
                            for point, role in additions.items()
                            for n in _neighbors(point, height, width)
                        )
                        candidates.append((same_role_contacts, additions))
    if not candidates:
        return False
    if rng.random() < 0.38 and any(score for score, _ in candidates):
        candidates = [item for item in candidates if item[0]]
    cells.update(candidates[int(rng.integers(len(candidates)))][1])
    return True


def _add_near_miss(rng, cells, reserved, locked, height, width, kind):
    candidates = []
    for r in range(height):
        for c in range(width):
            p = (r, c)
            if kind == "diagonal":
                offsets = _DIAG
            else:
                offsets = ((-2, 0), (2, 0), (0, -2), (0, 2))
            for dr, dc in offsets:
                q = (r + dr, c + dc)
                if not (0 <= q[0] < height and 0 <= q[1] < width):
                    continue
                middle = (r + dr // 2, c + dc // 2) if kind == "gap" else None
                if middle is not None and (middle in cells or middle in reserved):
                    continue
                if p in reserved or q in reserved:
                    continue
                additions = {p: 1, q: 2}
                if _can_add(cells, additions, height, width):
                    trial = cells | additions
                    if all(sum(trial.get(n) == 3 - role for n in _neighbors(point, height, width)) == 0
                           for point, role in additions.items()):
                        candidates.append((additions, middle))
    if not candidates:
        return False
    additions, middle = candidates[int(rng.integers(len(candidates)))]
    cells.update(additions)
    locked.update(additions)
    if middle is not None:
        reserved.add(middle)
    return True


def _add_single(rng, cells, reserved, height, width, role):
    candidates = []
    for r in range(height):
        for c in range(width):
            p = (r, c)
            if p in reserved or not _can_add(cells, {p: role}, height, width):
                continue
            if any(cells.get(n) == 3 - role for n in _neighbors(p, height, width)):
                continue
            candidates.append(p)
    if not candidates:
        return False
    cells[candidates[int(rng.integers(len(candidates)))]] = role
    return True


def _matched_positions(cells, height, width):
    matches = []
    for p, role in cells.items():
        if role == 1:
            targets = [q for q in _neighbors(p, height, width) if cells.get(q) == 2]
            if len(targets) == 1:
                matches.append((p, targets[0]))
    return matches


def _shape(rng, rich, split):
    if rich:
        dimensions = (int(rng.integers(8, 13)), int(rng.integers(8, 14)))
    else:
        probabilities = (0.35, 0.45, 0.16, 0.04) if split == "train" else (0.27, 0.42, 0.24, 0.07)
        band = int(rng.choice(4, p=probabilities))
        bounds = ((3, 6), (6, 12), (11, 21), (20, 29))[band]
        dimensions = (int(rng.integers(*bounds)), int(rng.integers(*bounds)))
        if band >= 1 and rng.random() < 0.35:
            narrow = int(rng.integers(3, 8))
            dimensions = (dimensions[0], narrow)
    if bool(rng.integers(0, 2)):
        dimensions = dimensions[::-1]
    return dimensions


def _make_pair(rng, roles, background, marker, plan):
    for attempt in range(18):
        height, width = _shape(rng, plan["rich"], plan["split"])
        cells = {}
        reserved = set()
        locked = set()
        motif = plan["motif"]
        if motif and not _try_motif(rng, cells, height, width, motif, plan["orientation"]):
            continue
        initial_matches = 0 if motif is None else (2 if motif == "block" else 1)
        if plan["no_match"]:
            desired_matches = 0
        elif plan["rich"]:
            desired_matches = int(rng.integers(initial_matches, max(initial_matches + 1, 5)))
        else:
            upper = max(initial_matches + 1, min(9, max(2, (height * width) // 9 + 1)))
            desired_matches = int(rng.integers(max(1, initial_matches), upper))
        if plan["clean"]:
            desired_matches = max(1, initial_matches)
        if plan["diagonal"] and not _add_near_miss(rng, cells, reserved, locked, height, width, "diagonal"):
            continue
        if plan["gap"] and not _add_near_miss(rng, cells, reserved, locked, height, width, "gap"):
            continue
        failed = False
        for index in range(desired_matches - initial_matches):
            required_orientation = plan["orientation"] if index == 0 and initial_matches == 0 else None
            if not _add_domino(rng, cells, reserved, locked, height, width, required_orientation):
                failed = True
                break
        if failed:
            continue
        if plan["no_match"]:
            extra_e, extra_t = 1, 1
        elif plan["clean"]:
            extra_e, extra_t = 0, 0
        elif plan["rich"]:
            extra_e, extra_t = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        else:
            extra_e, extra_t = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        for role, count in ((1, extra_e), (2, extra_t)):
            for _ in range(count):
                if not _add_single(rng, cells, reserved, height, width, role):
                    failed = True
                    break
        if failed:
            continue
        if plan["rich"]:
            marker_count = int(rng.integers(1, 5))
        elif plan["clean"]:
            marker_count = 0 if rng.random() < 0.7 else 1
        else:
            marker_count = int(rng.integers(0, 6))
        if plan["must_marker"]:
            marker_count = max(1, marker_count)
        if plan["matched_marker"]:
            marker_count = max(1, marker_count)
        if marker_count:
            if plan["matched_marker"]:
                endpoints = [point for match in _matched_positions(cells, height, width) for point in match]
                candidates = sorted({q for p in endpoints for q in _neighbors(p, height, width)
                                     if q not in cells and q not in reserved})
                if not candidates:
                    continue
                cells[candidates[int(rng.integers(len(candidates)))]] = 3
                marker_count -= 1
            for _ in range(marker_count):
                candidates = [(r, c) for r in range(height) for c in range(width)
                              if (r, c) not in cells and (r, c) not in reserved]
                if not candidates:
                    failed = True
                    break
                cells[candidates[int(rng.integers(len(candidates)))]] = 3
        if failed or len(cells) * 2 >= height * width:
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        palette = {1: roles[0], 2: roles[1], 3: marker}
        for point, role in cells.items():
            input_grid[point] = palette[role]
        output_grid = _rewrite(input_grid, background, roles[0], roles[1], roles[2])
        if output_grid is not None and (plan["no_match"] or not np.array_equal(input_grid, output_grid)):
            return input_grid, output_grid
    raise RuntimeError("could not construct the requested layout")


def _matches(grid, eraser, target):
    height, width = grid.shape
    matches = []
    target_degrees = {}
    for row, col in np.argwhere(grid == eraser):
        p = (int(row), int(col))
        contacts = [q for q in _neighbors(p, height, width) if int(grid[q]) == target]
        if len(contacts) > 1:
            return None
        if contacts:
            q = contacts[0]
            target_degrees[q] = target_degrees.get(q, 0) + 1
            matches.append((p, q))
    if any(degree > 1 for degree in target_degrees.values()):
        return None
    return matches


def _rewrite(grid, background, eraser, target, highlight):
    matches = _matches(grid, eraser, target)
    if matches is None:
        return None
    output = grid.copy()
    for e, t in matches:
        output[e] = background
        output[t] = highlight
    return output


def _key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _source_copy(inp, out):
    rows_in = tuple("".join(str(int(v)) for v in row) for row in inp)
    rows_out = tuple("".join(str(int(v)) for v in row) for row in out)
    return (rows_in, rows_out) in _SOURCE


def _mode(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert int(np.count_nonzero(counts == counts.max())) == 1
    return int(values[int(np.argmax(counts))])


def _infer_roles(train):
    erasers, targets, highlights = set(), set(), set()
    backgrounds, markers = set(), set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        bg = _mode(inp)
        backgrounds.add(bg)
        removed = (inp != out) & (out == bg)
        colored = (inp != out) & (out != bg)
        assert np.any(removed) and np.any(colored)
        erasers.update(int(v) for v in np.unique(inp[removed]))
        targets.update(int(v) for v in np.unique(inp[colored]))
        highlights.update(int(v) for v in np.unique(out[colored]))
    assert len(erasers) == len(targets) == len(highlights) == 1
    e, t, h = next(iter(erasers)), next(iter(targets)), next(iter(highlights))
    assert len({e, t, h}) == 3
    for pair in train:
        inp = pair["input"]
        bg = _mode(inp)
        assert bg not in (e, t, h)
        extras = set(int(v) for v in np.unique(inp)) - {bg, e, t}
        assert len(extras) <= 1 and h not in extras
        markers.update(extras)
    assert markers.isdisjoint({e, t, h}) and markers.isdisjoint(backgrounds)
    return e, t, h, backgrounds, markers


def _stats(grid, bg, e, t, marker_colors):
    height, width = grid.shape
    matches = _matches(grid, e, t)
    assert matches is not None
    matched_e = {a for a, _ in matches}
    matched_t = {b for _, b in matches}
    e_points = {(int(r), int(c)) for r, c in np.argwhere(grid == e)}
    t_points = {(int(r), int(c)) for r, c in np.argwhere(grid == t)}
    diagonal = sum(any(q in t_points for q in _neighbors(p, height, width, _DIAG))
                   for p in e_points - matched_e)
    gap = 0
    for r, c in e_points - matched_e:
        for dr, dc in _ORTHO:
            q = (r + 2 * dr, c + 2 * dc)
            middle = (r + dr, c + dc)
            if q in t_points - matched_t and 0 <= middle[0] < height and 0 <= middle[1] < width and int(grid[middle]) == bg:
                gap += 1
    matched_marker = any(
        int(grid[q]) in marker_colors
        for match in matches for p in match for q in _neighbors(p, height, width)
    )
    same_role = any(
        int(grid[q]) == role
        for role, points in ((e, e_points), (t, t_points))
        for p in points for q in _neighbors(p, height, width)
    )
    return {
        "matches": len(matches),
        "horizontal": sum(a[0] == b[0] for a, b in matches),
        "vertical": sum(a[1] == b[1] for a, b in matches),
        "unmatched_e": len(e_points - matched_e),
        "unmatched_t": len(t_points - matched_t),
        "diagonal": diagonal,
        "gap": gap,
        "matched_marker": matched_marker,
        "same_role": same_role,
        "markers": sum(int(np.count_nonzero(grid == color)) for color in marker_colors),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    colors = [int(v) for v in rng.permutation(10)]
    roles = tuple(colors[:3])
    varied_palette = bool(rng.integers(0, 2))
    backgrounds = colors[3:5] if varied_palette else colors[3:4]
    markers = colors[5:7] if varied_palette else colors[5:6]
    train_backgrounds = [backgrounds[i % len(backgrounds)] for i in range(train_count)]
    train_markers = [markers[i % len(markers)] for i in range(train_count)]
    rng.shuffle(train_backgrounds)
    rng.shuffle(train_markers)
    orientations = ["H", "V"] + [
        "H" if bool(rng.integers(0, 2)) else "V" for _ in range(train_count - 2)
    ]
    rng.shuffle(orientations)
    coverage_index = int(rng.integers(train_count))
    coverage_motif = "chain" if bool(rng.integers(0, 2)) else "block"
    puzzle = {"train": [], "test": []}
    seen = set()
    trained_motifs = {coverage_motif}
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            if split == "train":
                rich = index == coverage_index
                motif = coverage_motif if rich else (
                    ("chain" if bool(pair_rng.integers(0, 2)) else "block")
                    if pair_rng.random() < 0.24 else None
                )
                if motif:
                    trained_motifs.add(motif)
                clean = not rich and pair_rng.random() < 0.34
                plan = {
                    "split": split, "rich": rich, "motif": motif, "clean": clean,
                    "orientation": orientations[index], "no_match": False,
                    "diagonal": rich or (not clean and pair_rng.random() < 0.30),
                    "gap": rich or (not clean and pair_rng.random() < 0.30),
                    "matched_marker": rich or (not clean and pair_rng.random() < 0.28),
                    "must_marker": varied_palette or index == coverage_index,
                }
                bg, marker = train_backgrounds[index], train_markers[index]
            else:
                clean = pair_rng.random() < 0.28
                no_match = pair_rng.random() < 0.10
                motif = None if clean or no_match or pair_rng.random() >= 0.45 else (
                    sorted(trained_motifs)[int(pair_rng.integers(len(trained_motifs)))]
                )
                plan = {
                    "split": split, "rich": False, "motif": motif, "clean": clean and not no_match,
                    "orientation": "H" if bool(pair_rng.integers(0, 2)) else "V",
                    "no_match": no_match,
                    "diagonal": not clean and pair_rng.random() < 0.36,
                    "gap": not clean and pair_rng.random() < 0.36,
                    "matched_marker": not clean and not no_match and pair_rng.random() < 0.34,
                    "must_marker": False,
                }
                bg = backgrounds[int(pair_rng.integers(len(backgrounds)))]
                marker = markers[int(pair_rng.integers(len(markers)))]
            for _ in range(12):
                try:
                    inp, out = _make_pair(pair_rng, roles, bg, marker, plan)
                except RuntimeError:
                    continue
                if _key(inp) in seen or _source_copy(inp, out):
                    continue
                seen.add(_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))
        assert inp.shape == out.shape
        key = _key(inp), _key(out)
        assert key[0] not in seen_inputs and key not in seen_pairs
        seen_inputs.add(key[0])
        seen_pairs.add(key)
        assert not _source_copy(inp, out)
    e, t, h, backgrounds, markers = _infer_roles(puzzle["train"])
    training_stats = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            bg = _mode(inp)
            assert bg in backgrounds and bg not in (e, t, h)
            assert int(np.count_nonzero(inp == bg)) * 2 > inp.size
            extras = set(int(v) for v in np.unique(inp)) - {bg, e, t}
            assert extras <= markers and len(extras) <= 1
            assert h not in set(int(v) for v in np.unique(inp))
            expected = _rewrite(inp, bg, e, t, h)
            assert expected is not None and np.array_equal(expected, out)
            stats = _stats(inp, bg, e, t, markers)
            if split == "train":
                assert stats["matches"] >= 1
                training_stats.append(stats)
            elif stats["matches"] == 0:
                assert np.array_equal(inp, out)
    assert any(s["horizontal"] for s in training_stats)
    assert any(s["vertical"] for s in training_stats)
    assert any(s["same_role"] for s in training_stats)
    assert any(s["diagonal"] and s["gap"] and s["matched_marker"] and
               s["unmatched_e"] and s["unmatched_t"] for s in training_stats)
    assert markers


def _equal_puzzles(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
            for x, y in zip(a[split], b[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, signatures, palettes = set(), set(), set()
    test_stats = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        e, t, h, backgrounds, markers = _infer_roles(puzzle["train"])
        palettes.add((e, t, h, tuple(sorted(backgrounds)), tuple(sorted(markers))))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                shapes.add((split, inp.shape))
                signatures.add(_key(inp))
                bg = _mode(inp)
                assert np.array_equal(_rewrite(inp, bg, e, t, h), out)
                if split == "test":
                    test_stats.append((_stats(inp, bg, e, t, markers), inp.shape))
    assert len(shapes) >= 40 and len(signatures) >= 110 and len(palettes) >= 24
    assert any(s["matches"] == 0 for s, _ in test_stats)
    assert any(s["matches"] == 1 and s["markers"] == 0 for s, _ in test_stats)
    assert any(shape[0] > shape[1] for _, shape in test_stats)
    assert any(max(shape) >= 20 for _, shape in test_stats)
    assert any(s["same_role"] and s["matched_marker"] for s, _ in test_stats)
    for args in ((314159, None, None), (271828, 7, 3)):
        assert _equal_puzzles(generate_puzzle(*args), generate_puzzle(*args))
    for seed, a, b in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                       (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (3, 3), (3, 4), (4, 4),
                 (True, 1), (2.0, 1), (2, False), (2, 1.0)):
        try:
            generate_puzzle(999, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
