import numpy as np


_SOURCE_PAIR_ROWS = {
    (
        (
            "77478677877657797",
            "77448622833655797",
            "22478672837675099",
            "77448622833655079",
            "77478677877657079",
        ),
        ("77897", "77897", "22899", "77879", "77879"),
    ),
    (
        (
            "77188677777647777",
            "88188622944640011",
            "87177622944640071",
            "88777677774670011",
            "77777677774677717",
        ),
        ("77777", "88911", "87971", "88711", "77717"),
    ),
    (
        (
            "77777677777677777",
            "79317677777677777",
            "77777678567677777",
            "77777677777674027",
            "77777677777677777",
        ),
        ("77777", "79777", "77577", "77727", "77777"),
    ),
    (
        (
            "44997688777622777",
            "47997678777672799",
            "44772688799622579",
            "74772687279672799",
            "44772688799622777",
        ),
        ("44777", "47799", "44779", "74299", "44777"),
    ),
    (
        (
            "77182677777677777",
            "77182677777677743",
            "77782677797677543",
            "77772677893677543",
            "77777677893677773",
        ),
        ("77177", "77173", "77793", "77793", "77793"),
    ),
    (
        (
            "71833369922776577777",
            "71777369772776788777",
            "71777369772776778777",
            "11777369772796778887",
            "77777767772796777787",
            "77777767772296777775",
        ),
        ("712277", "717277", "717277", "117277", "777277", "777225"),
    ),
}


def _is_int_value(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _is_int_value(num_train):
            raise ValueError("num_train must be an integer or None")
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        if not _is_int_value(num_test):
            raise ValueError("num_test must be an integer or None")
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        span = 10 - num_test
        num_train = num_test + 1 + (seed // 4) % span
        return num_train, num_test

    if num_train is None:
        lower = max(2, int(num_test) + 1)
        num_train = lower + seed % (11 - lower)
        return num_train, int(num_test)

    maximum_test = min(4, num_train - 1)
    num_test = 1 + seed % maximum_test
    return num_train, num_test


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _grid_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _is_source_pair(pair: dict) -> bool:
    return (_grid_rows(pair["input"]), _grid_rows(pair["output"])) in _SOURCE_PAIR_ROWS


def _components(panel: np.ndarray, background: int) -> list[dict]:
    n = panel.shape[0]
    seen = np.zeros(panel.shape, dtype=bool)
    found = []
    for row in range(n):
        for col in range(n):
            color = int(panel[row, col])
            if color == background or seen[row, col]:
                continue
            stack = [(row, col)]
            cells = []
            seen[row, col] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n and not seen[rr, cc] and int(panel[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            cells.sort()
            found.append({"color": color, "cells": tuple(cells),
                          "min_col": min(c for _, c in cells),
                          "max_col": max(c for _, c in cells)})
    assert len(found) == 3
    found.sort(key=lambda x: x["min_col"])
    assert all(found[i]["max_col"] < found[i + 1]["min_col"] for i in (0, 1))
    return found


def _analyze_pair(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    n = inp.shape[0]
    assert 5 <= n <= 9 and inp.shape == (n, 3 * n + 2) and out.shape == (n, n)
    divider = int(inp[0, n])
    assert np.all(inp[:, n] == divider) and np.all(inp[:, 2 * n + 1] == divider)
    panels = [inp[:, :n], inp[:, n + 1:2 * n + 1], inp[:, 2 * n + 2:]]
    values, counts = np.unique(np.concatenate([p.flat for p in panels]), return_counts=True)
    assert np.sum(counts == counts.max()) == 1
    background = int(values[np.argmax(counts)])
    assert divider != background
    table = [_components(p, background) for p in panels]
    return {"n": n, "background": background, "divider": divider,
            "panels": panels, "components": table}


def _compose(analysis: dict, ranks: tuple[int, int, int]) -> np.ndarray | None:
    n, bg = analysis["n"], analysis["background"]
    out = np.full((n, n), bg, dtype=np.int8)
    occupied = np.zeros((n, n), dtype=bool)
    for panel_index, rank in enumerate(ranks):
        obj = analysis["components"][panel_index][rank]
        for row, col in obj["cells"]:
            if occupied[row, col] and int(out[row, col]) != obj["color"]:
                return None
            out[row, col] = obj["color"]
            occupied[row, col] = True
    return out


def _matching_choices(analysis: dict, output: np.ndarray) -> list[tuple[int, int, int]]:
    matches = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                guess = _compose(analysis, (a, b, c))
                if guess is not None and np.array_equal(guess, output):
                    matches.append((a, b, c))
    return matches


def _touches(panel: np.ndarray, background: int) -> bool:
    for a, b in ((panel[:, :-1], panel[:, 1:]), (panel[:-1, :], panel[1:, :])):
        if np.any((a != background) & (b != background) & (a != b)):
            return True
    return False


def _features(analysis: dict) -> dict:
    table = analysis["components"]
    sizes = [len(obj["cells"]) for panel in table for obj in panel]
    colors = [[obj["color"] for obj in panel] for panel in table]
    densities = [sum(len(obj["cells"]) for obj in panel) / analysis["n"] ** 2 for panel in table]
    return {"all_points": all(size == 1 for size in sizes),
            "aligned_points": all(size == 1 for size in sizes) and all(
                len({row for obj in panel for row, _ in obj["cells"]}) == 1
                for panel in table),
            "mixed_points": any(size == 1 for size in sizes) and any(size > 1 for size in sizes),
            "dense": max(densities) >= 0.56,
            "density_max": max(densities),
            "density_min": min(densities),
            "repeat_panels": tuple(i for i, row in enumerate(colors) if row[0] == row[2]),
            "touch_panels": tuple(i for i, panel in enumerate(analysis["panels"]) if _touches(panel, analysis["background"])),
            "divider_object": any(analysis["divider"] in row for row in colors),
            "selected_recur": any(colors[i][i] in [colors[p][j] for p in range(3) for j in range(3) if p != j] for i in range(3)),
            "wide_object": any(obj["max_col"] > obj["min_col"] for panel in table for obj in panel)}


def _intervals(rng: np.random.Generator, n: int, profile: str,
               contact: str, point_mask: tuple[bool, bool, bool]) -> list[tuple[int, int]]:
    if profile == "points":
        columns = sorted(int(x) for x in rng.choice(n, size=3, replace=False))
        return [(c, 1) for c in columns]
    minimum_gaps = [0, 1 if contact == "none" else 0, 1 if contact == "none" else 0, 0]
    if profile == "dense":
        used = n if contact != "none" else n - 2
    else:
        used = int(rng.integers(3, n - sum(minimum_gaps) + 1))
    if all(point_mask):
        used = 3
    widths = [1, 1, 1]
    recipients = [i for i in range(3) if not point_mask[i]]
    for _ in range(used - 3):
        if not recipients:
            break
        widths[int(rng.choice(recipients))] += 1
    gaps = minimum_gaps[:]
    allowed = [0, 1, 2, 3]
    if contact in ("left", "both"):
        allowed.remove(1)
    if contact in ("right", "both"):
        allowed.remove(2)
    for _ in range(n - used - sum(gaps)):
        gaps[int(rng.choice(allowed))] += 1
    starts = []
    cursor = gaps[0]
    for i in range(3):
        starts.append((cursor, widths[i]))
        cursor += widths[i] + gaps[i + 1]
    assert cursor == n
    return starts


def _grow(rng: np.random.Generator, cells: set[tuple[int, int]], n: int,
          start: int, width: int) -> bool:
    edge = set()
    for row, col in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = row + dr, col + dc
            if 0 <= r < n and start <= c < start + width and (r, c) not in cells:
                edge.add((r, c))
    if not edge:
        return False
    choices = sorted(edge)
    cells.add(choices[int(rng.integers(len(choices)))])
    return True


def _geometry(rng: np.random.Generator, n: int, profile: str, contact: str,
              point_mask: tuple[bool, bool, bool]) -> list[set[tuple[int, int]]]:
    intervals = _intervals(rng, n, profile, contact, point_mask)
    shapes = []
    last_row = None
    for i, (start, width) in enumerate(intervals):
        row = last_row if i == 1 and contact in ("left", "both") or i == 2 and contact in ("right", "both") else int(rng.integers(n))
        cells = {(row, start)}
        for col in range(start + 1, start + width):
            row = max(0, min(n - 1, row + int(rng.integers(-1, 2))))
            cells.add((row, col))
            if (row - 1, col - 1) in cells:
                cells.add((row - 1, col))
            if (row + 1, col - 1) in cells:
                cells.add((row + 1, col))
        shapes.append(cells)
        last_row = row
    base = sum(map(len, shapes))
    capacity = sum((1 if point_mask[i] else n * width) for i, (_, width) in enumerate(intervals))
    if profile == "dense":
        target = int(rng.integers(int(np.ceil(.56 * n * n)), int(np.ceil(.71 * n * n)) + 1))
    elif profile == "mixed":
        target = int(rng.integers(int(np.ceil(.28 * n * n)), int(np.ceil(.55 * n * n)) + 1))
    else:
        target = int(rng.integers(base, max(base, int(np.ceil(.46 * n * n))) + 1))
    target = min(capacity, max(base, target))
    while sum(map(len, shapes)) < target:
        candidates = [i for i in range(3) if not point_mask[i] and len(shapes[i]) < n * intervals[i][1]]
        if not candidates:
            break
        i = int(rng.choice(candidates))
        _grow(rng, shapes[i], n, *intervals[i])
    return shapes


def _colors(rng: np.random.Generator, background: int, divider: int,
            profile: str, force_collision: bool) -> list[list[int]]:
    palette = [c for c in range(10) if c != background]
    mode = int(rng.choice(4, p=[.14, .40, .31, .15]))
    if profile == "points" and float(rng.random()) < .55:
        mode = 0
    if mode == 0:
        values = [int(x) for x in rng.permutation(palette)]
        matrix = [values[i * 3:i * 3 + 3] for i in range(3)]
    else:
        pool = palette if mode in (1, 2) else [int(x) for x in rng.choice(palette, size=int(rng.integers(3, 6)), replace=False)]
        matrix = []
        for _ in range(3):
            a = int(rng.choice(pool))
            b = int(rng.choice([c for c in pool if c != a]))
            c = int(rng.choice([c for c in pool if c != b]))
            matrix.append([a, b, c])
        if mode == 2:
            for i in rng.choice(3, size=int(rng.integers(1, 3)), replace=False):
                matrix[int(i)][2] = matrix[int(i)][0]
    if force_collision and not any(divider in row for row in matrix):
        p = int(rng.integers(3))
        j = int(rng.integers(3))
        matrix[p][j] = divider
        for neighbor in (j - 1, j + 1):
            if 0 <= neighbor < 3 and matrix[p][neighbor] == divider:
                matrix[p][neighbor] = int(rng.choice([c for c in palette if c != divider]))
    return matrix


def _make_pair(rng: np.random.Generator, n: int, background: int, divider: int,
               preferred: str | None = None) -> dict:
    for _ in range(350):
        profile = preferred if preferred is not None else str(rng.choice(["open", "mixed", "dense", "points"], p=[.29, .30, .31, .10]))
        contact = str(rng.choice(["none", "left", "right", "both", "free"], p=[.22, .18, .18, .12, .30]))
        point_masks = []
        for _panel in range(3):
            if profile == "points":
                mask = (True, True, True)
            elif profile == "mixed":
                mask = tuple(bool(x) for x in rng.random(3) < .28)
            else:
                mask = (False, False, False)
            point_masks.append(mask)
        if profile == "mixed" and not any(any(x) for x in point_masks):
            p, j = int(rng.integers(3)), int(rng.integers(3))
            mask = list(point_masks[p]); mask[j] = True; point_masks[p] = tuple(mask)
        shapes = [_geometry(rng, n, profile, contact, point_masks[p]) for p in range(3)]
        selected = [shapes[p][p] for p in range(3)]
        if any(selected[i] & selected[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            continue
        matrix = _colors(rng, background, divider, profile, float(rng.random()) < .24)
        panels = []
        for p in range(3):
            panel = np.full((n, n), background, dtype=np.int8)
            for j in range(3):
                for row, col in shapes[p][j]:
                    panel[row, col] = matrix[p][j]
            panels.append(panel)
        div = np.full((n, 1), divider, dtype=np.int8)
        inp = np.concatenate([panels[0], div, panels[1], div, panels[2]], axis=1)
        out = np.full((n, n), background, dtype=np.int8)
        for p in range(3):
            for row, col in selected[p]:
                out[row, col] = matrix[p][p]
        pair = {"input": inp, "output": out}
        if _is_source_pair(pair):
            continue
        try:
            analysis = _analyze_pair(pair)
            if analysis["background"] != background or _matching_choices(analysis, out) != [(0, 1, 2)]:
                continue
            if profile == "points" and not _features(analysis)["all_points"]:
                continue
            if profile == "dense" and not _features(analysis)["dense"]:
                continue
            if contact == "none" and _features(analysis)["touch_panels"]:
                continue
        except AssertionError:
            continue
        return pair
    raise RuntimeError("could not construct pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_int_value(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_roles = bool(rng.integers(2))
    first_roles = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_sizes = []
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(100):
                if fixed_roles or split == "train" and index == 0:
                    background, divider = first_roles
                else:
                    background, divider = (int(x) for x in rng.choice(10, size=2, replace=False))
                    if split == "train" and index == 1 and (background == first_roles[0] or divider == first_roles[1]):
                        continue
                if split == "train":
                    n = 5 if index == 0 else int(rng.choice([5, 6, 7, 8], p=[.48, .29, .17, .06]))
                else:
                    n = int(rng.choice(train_sizes)) if float(rng.random()) < .38 else int(rng.choice([5, 6, 7, 8, 9], p=[.29, .29, .23, .13, .06]))
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                first_profile = str(rng.choice(["open", "mixed", "dense"], p=[.30, .40, .30])) if split == "train" and index == 0 else None
                pair = _make_pair(pair_rng, n, background, divider, first_profile)
                key = _grid_key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_sizes.append(n)
                break
            else:
                raise RuntimeError("could not construct unique puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key); seen_pairs.add(pair_key)
            assert not _is_source_pair(pair)
            analysis = _analyze_pair(pair)
            assert np.array_equal(_compose(analysis, (0, 1, 2)), pair["output"])
            assert _matching_choices(analysis, pair["output"]) == [(0, 1, 2)]
            analyses[split].append(analysis)
    assert analyses["train"][0]["n"] == 5
    compatible_rules = {
        (a, b, c) for a in range(3) for b in range(3) for c in range(3)
    }
    for pair, analysis in zip(puzzle["train"], analyses["train"]):
        compatible_rules &= set(_matching_choices(analysis, pair["output"]))
    assert compatible_rules == {(0, 1, 2)}
    learned_rule = next(iter(compatible_rules))
    for pair, analysis in zip(puzzle["test"], analyses["test"]):
        predicted = _compose(analysis, learned_rule)
        assert predicted is not None and np.array_equal(predicted, pair["output"])
    train_roles = {(a["background"], a["divider"]) for a in analyses["train"]}
    all_roles = train_roles | {(a["background"], a["divider"]) for a in analyses["test"]}
    if len(all_roles) > 1:
        assert len(train_roles) > 1
        assert len({a["background"] for a in analyses["train"]}) > 1
        assert len({a["divider"] for a in analyses["train"]}) > 1
    # Each demonstrated panel has three visible ranked objects and the same diagonal choice.
    # The structural role rule makes a novel literal color in a test input observable.


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_grid_key(p["train"][0]["input"]) for p in puzzles}) == 42
    role_changes = constant_roles = dense = all_points = aligned_points = mixed = collisions = wide = 0
    repeat_locations, touch_locations, sizes, train_sizes = set(), set(), set(), set()
    no_repeat_contact_test = first_selected_recur = first_not_recur = 0
    larger_train = all_tests_trained = larger_test_nonlast = 0
    varied_points = varied_repeats = varied_contacts = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_size_set = {_analyze_pair(p)["n"] for p in puzzle["train"]}
        larger_train += len(train_size_set) > 1
        all_tests_trained += all(_analyze_pair(p)["n"] in train_size_set for p in puzzle["test"])
        larger_test_nonlast += any(_analyze_pair(p)["n"] > max(train_size_set) for p in puzzle["test"][:-1])
        role_changes += len({(_analyze_pair(p)["background"], _analyze_pair(p)["divider"]) for p in puzzle["train"]}) > 1
        constant_roles += len({(_analyze_pair(p)["background"], _analyze_pair(p)["divider"]) for split in ("train", "test") for p in puzzle[split]}) == 1
        all_pair_features = [_features(_analyze_pair(p)) for split in ("train", "test") for p in puzzle[split]]
        varied_points += len({f["all_points"] for f in all_pair_features}) > 1
        varied_repeats += len({bool(f["repeat_panels"]) for f in all_pair_features}) > 1
        varied_contacts += len({bool(f["touch_panels"]) for f in all_pair_features}) > 1
        if all_pair_features[0]["selected_recur"]: first_selected_recur += 1
        else: first_not_recur += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair)
                f = _features(analysis)
                dense += f["dense"]; all_points += f["all_points"]; aligned_points += f["aligned_points"]; mixed += f["mixed_points"]
                collisions += f["divider_object"]; wide += f["wide_object"]
                repeat_locations.update(f["repeat_panels"]); touch_locations.update(f["touch_panels"])
                sizes.add(analysis["n"])
                if split == "train": train_sizes.add(analysis["n"])
                else: no_repeat_contact_test += not f["repeat_panels"] and not f["touch_panels"]
                assert _matching_choices(analysis, pair["output"]) == [(0, 1, 2)]
    assert role_changes >= 8 and constant_roles >= 8 and larger_train >= 8 and all_tests_trained >= 4
    assert larger_test_nonlast >= 1
    assert varied_points >= 8 and varied_repeats >= 8 and varied_contacts >= 8
    assert first_selected_recur >= 5 and first_not_recur >= 5
    assert dense >= 12 and all_points >= 3 and aligned_points >= 1 and mixed >= 12 and collisions >= 12 and wide >= 20
    assert repeat_locations == {0, 1, 2} and touch_locations == {0, 1, 2}
    assert no_repeat_contact_test >= 1 and len(sizes) >= 4 and len(train_sizes) >= 2
    for seed, nt, ns in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 6, 2),
                         (110, None, 1), (111, None, 4), (112, 2, None), (113, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    assert _puzzles_equal(generate_puzzle(700, 5, 2), generate_puzzle(700, 5, 2))
    assert not _puzzles_equal(generate_puzzle(700, 5, 2), generate_puzzle(701, 5, 2))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(900, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
