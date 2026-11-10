import numpy as np


_DIRECTIONS = ("down", "up", "right", "left")

_SOURCE_PAIR_TEXTS = {
    (
        "080008080080/080008080080/080008080080/080008080080/"
        "000000000000/022002220000/000000000000/000000000000/"
        "000000000000/000000000000",
        "080008080080/080008080080/080008080080/080008080080/"
        "880088088080/822082228080/800080008080/800080008080/"
        "800080008080/800080008080",
    ),
    (
        "080808080808/080808080808/080808080808/080808080808/"
        "000000000000/000002220000/000000000000/000000000000/"
        "000000000000/000000000000",
        "080808080808/080808080808/080808080808/080808080808/"
        "080888088808/080882228808/080880008808/080880008808/"
        "080880008808/080880008808",
    ),
    (
        "808000800808/808000800808/808000800808/808000800808/"
        "000000000000/002202220000/000000000000/000000000000/"
        "000000000000/000000000000",
        "808000800808/808000800808/808000800808/808000800808/"
        "888000888808/882202228808/880000008808/880000008808/"
        "880000008808/880000008808",
    ),
    (
        "0800080008/0800080008/0800080008/0000000000/0000222000/"
        "0000000000/0000000000/0000000000/0000000000",
        "0800080008/0800080008/0800080008/0800088808/0800222808/"
        "0800000808/0800000808/0800000808/0800000808",
    ),
    (
        "080080080080080/080080080080080/080080080080080/"
        "080080080080080/000022000022220/000000000000000/"
        "000000000000000/000000000000000/000000000000000/"
        "000000000000000",
        "080080080080080/080080080080080/080080080080080/"
        "080880080880088/080822080822228/080800080800008/"
        "080800080800008/080800080800008/080800080800008/"
        "080800080800008",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _valid_count(value, low: int, high: int) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and low <= int(value) <= high
    )


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None and not _valid_count(num_train, 2, 10):
        raise ValueError("num_train must be an integer from 2 through 10")

    if num_test is not None and not _valid_count(num_test, 1, 4):
        raise ValueError("num_test must be an integer from 1 through 4")

    if num_train is None and num_test is None:
        index = abs(int(seed))
        num_test = 1 + index % 4
        first_train = max(2, num_test + 1)
        number_of_choices = 11 - first_train
        num_train = first_train + (index // 4) % number_of_choices

    elif num_train is None:
        num_test = int(num_test)
        first_train = max(2, num_test + 1)
        num_train = int(rng.integers(first_train, 11))

    elif num_test is None:
        num_train = int(num_train)
        largest_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test + 1))

    else:
        num_train = int(num_train)
        num_test = int(num_test)

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _detour_target(lane: int, start: int, end: int) -> int:
    lower, upper = start - 1, end + 1
    return lower if lane - lower < upper - lane else upper


def _from_oriented(grid: np.ndarray, direction: str) -> np.ndarray:
    if direction == "down":
        return grid.copy()
    if direction == "up":
        return grid[::-1, :].copy()
    if direction == "right":
        return grid.T.copy()
    if direction == "left":
        return grid[::-1, :].T.copy()
    raise ValueError("unknown direction")


def _to_oriented(grid: np.ndarray, direction: str) -> np.ndarray:
    if direction == "down":
        return grid.copy()
    if direction == "up":
        return grid[::-1, :].copy()
    if direction == "right":
        return grid.T.copy()
    if direction == "left":
        return grid.T[::-1, :].copy()
    raise ValueError("unknown direction")


def _components(mask: np.ndarray) -> list[tuple[int, int]]:
    result = []
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        if start is not None and (not active or i == len(mask) - 1):
            end = i if active else i - 1
            result.append((start, end))
            start = None
    return result


def _grid_signature(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _pair_signature(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _grid_signature(input_grid), _grid_signature(output_grid)


def _grid_text(grid: np.ndarray) -> str:
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _source_copy(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return (_grid_text(input_grid), _grid_text(output_grid)) in _SOURCE_PAIR_TEXTS


def _interpret_oriented(grid: np.ndarray, direction: str) -> dict | None:
    """Recover source, barriers, and route using the visible input alone."""
    a = _to_oriented(grid, direction)
    length, width = a.shape
    if length < 7 or width < 8:
        return None
    colors, counts = np.unique(a[0], return_counts=True)
    if len(colors) != 2:
        return None
    background = int(colors[int(np.argmax(counts))])
    guide = int(colors[int(np.argmin(counts))])
    if counts[0] == counts[1]:
        return None
    all_colors = set(int(x) for x in np.unique(a))
    if len(all_colors) != 3:
        return None
    barrier = next(iter(all_colors - {background, guide}))
    band = 0
    while band < length and np.array_equal(a[band], a[0]):
        band += 1
    if not 2 <= band <= 6 or band >= length - 3:
        return None
    guides = tuple(int(x) for x in np.flatnonzero(a[0] == guide))
    if not 2 <= len(guides) <= 10 or any(y - x < 2 for x, y in zip(guides, guides[1:])):
        return None
    if np.any(a[band:] == guide) or np.any(a[:band] == barrier):
        return None
    if np.any((a[band:] != background) & (a[band:] != barrier)):
        return None
    if np.any(a[-2:] == barrier):
        return None
    obstacles = []
    barrier_rows = []
    for row in range(band, length - 2):
        parts = _components(a[row] == barrier)
        if parts:
            barrier_rows.append(row)
        for start, end in parts:
            if start < 1 or end > width - 2 or not 2 <= end - start + 1 <= 6:
                return None
            obstacles.append((row, start, end))
    if not 1 <= len(obstacles) <= 5:
        return None
    if any(y - x < 2 for x, y in zip(barrier_rows, barrier_rows[1:])):
        return None
    output = a.copy()
    occupied = np.zeros(a.shape, dtype=bool)
    hit_obstacles = set()
    targets = set()
    routes = []
    for lane in guides:
        first = next(((index, item) for index, item in enumerate(obstacles)
                      if item[1] <= lane <= item[2]), None)
        if first is None:
            path = [(row, lane) for row in range(band, length)]
            routes.append((lane, None, None))
        else:
            index, (row, start, end) = first
            if row < band:
                return None
            if start - 1 in guides or end + 1 in guides:
                return None
            if np.any(a[row:, start - 1] == barrier) or np.any(a[row:, end + 1] == barrier):
                return None
            target = _detour_target(lane, start, end)
            if target in guides or target in targets:
                return None
            if np.any(a[row:, target] == barrier):
                return None
            if np.any(a[band:row, lane] == barrier):
                return None
            low, high = sorted((lane, target))
            path = ([(s, lane) for s in range(band, row)]
                    + [(row - 1, col) for col in range(low, high + 1)]
                    + [(s, target) for s in range(row, length)])
            hit_obstacles.add(index)
            targets.add(target)
            routes.append((lane, index, target))
        for r, c in set(path):
            if a[r, c] == barrier:
                return None
            if occupied[r, c] and r >= band - 1:
                return None
            if a[r, c] == guide and c != lane:
                return None
            occupied[r, c] = True
            output[r, c] = guide
    if not hit_obstacles or np.array_equal(a, output):
        return None
    # Every barrier is a distinct component; an unhit one is allowed.
    return {
        "direction": direction, "background": background, "guide": guide,
        "barrier": barrier, "band": band, "length": length, "width": width,
        "guides": guides, "obstacles": tuple(obstacles),
        "routes": tuple(routes), "hit_obstacles": frozenset(hit_obstacles),
        "output": _from_oriented(output, direction),
    }


def _interpret_input(grid: np.ndarray) -> dict | None:
    candidates = []
    for direction in _DIRECTIONS:
        interpretation = _interpret_oriented(grid, direction)
        if interpretation is not None:
            candidates.append(interpretation)
    return candidates[0] if len(candidates) == 1 else None


def _features(info: dict) -> dict:
    obstacles = info["obstacles"]
    hit = info["hit_obstacles"]
    guides = info["guides"]
    routes = info["routes"]
    hit_counts = [sum(ob[1] <= lane <= ob[2] and route[1] == i
                      for lane, route in zip(guides, routes))
                  for i, ob in enumerate(obstacles)]
    ties = []
    clear_tie = False
    oriented_output = _to_oriented(info["output"], info["direction"])
    for lane, index, target in routes:
        if index is not None:
            row, start, end = obstacles[index]
            if lane - (start - 1) == (end + 1) - lane:
                ties.append((lane, index, target))
                if ((oriented_output[row - 1, lane - 1] == info["guide"])
                        != (oriented_output[row - 1, lane + 1] == info["guide"])):
                    clear_tie = True
    gaps = [obstacles[i][0] - info["band"] for i in hit]
    same_row_separations = [right[1] - left[2] - 1
                            for left, right in zip(obstacles, obstacles[1:])
                            if left[0] == right[0] and left[2] < right[1]]
    return {
        "tie": bool(ties), "clear_tie": clear_tie,
        "multi": any(n >= 2 for n in hit_counts),
        "straight": any(route[1] is None for route in routes),
        "all_hit": all(route[1] is not None for route in routes),
        "unhit": len(hit) < len(obstacles),
        "staggered": len({row for row, _, _ in obstacles}) > 1,
        "one_barrier": len(obstacles) == 1,
        "narrow": 1 in same_row_separations,
        "gaps": tuple(gaps),
        "lengths": tuple(end - start + 1 for _, start, end in obstacles),
        "hit_counts": tuple(hit_counts),
    }


def _sample_guides(rng: np.random.Generator, width: int, count: int) -> tuple[int, ...]:
    positions = np.sort(rng.choice(width - count + 1, size=count, replace=False))
    return tuple(int(x) for x in positions + np.arange(count))


def _barrier_candidates(width: int, guides: tuple[int, ...], mission: str) -> list[tuple]:
    result = []
    guide_set = set(guides)
    for size in range(2, min(6, width - 2) + 1):
        for start in range(1, width - size):
            end = start + size - 1
            hits = tuple(g for g in guides if start <= g <= end)
            if not 1 <= len(hits) <= 2:
                continue
            targets = tuple(_detour_target(g, start, end) for g in hits)
            if (len(set(targets)) != len(targets)
                    or start - 1 in guide_set or end + 1 in guide_set):
                continue
            if mission == "tie" and not any(g - (start - 1) == (end + 1) - g for g in hits):
                continue
            if mission == "multi" and len(hits) != 2:
                continue
            if mission == "no_tie" and any(g - (start - 1) == (end + 1) - g for g in hits):
                continue
            result.append((start, end, hits, targets))
    return result


def _compatible(candidate: tuple, row: int, selected: list[tuple]) -> bool:
    start, end, hits, targets = candidate
    for old_row, old_start, old_end, old_hits, old_targets in selected:
        if set(hits) & set(old_hits) or set(targets) & set(old_targets):
            return False
        if row == old_row and not (end + 1 < old_start or old_end + 1 < start):
            return False
        if row != old_row and abs(row - old_row) < 2:
            return False
    return True


def _build_pair(rng: np.random.Generator, regime: str, mission: str,
                direction: str, palette: tuple[int, int, int]) -> tuple | None:
    width = int(rng.integers(9, 24 if regime == "source_train" else 27))
    band = int(rng.integers(2, 6))
    gap = int(rng.integers(0, 5))
    reach = int(rng.integers(3, 11 if regime == "source_train" else 13))
    length = band + gap + reach
    if length > 29:
        return None
    max_guides = min(9, (width - 1) // 2)
    guide_count = int(rng.integers(2, max_guides + 1))
    guides = _sample_guides(rng, width, guide_count)
    first_candidates = _barrier_candidates(width, guides, mission if mission in ("tie", "multi", "no_tie") else "ordinary")
    if not first_candidates:
        return None
    first = first_candidates[int(rng.integers(len(first_candidates)))]
    selected = [(band + gap, *first)]
    target_count = int(rng.choice((1, 2, 2, 3, 3, 4)))
    if mission in ("narrow", "staggered", "all_hit", "unhit"):
        target_count = max(2, target_count)
    candidates = _barrier_candidates(width, guides, "no_tie" if mission == "no_tie" else "ordinary")
    for _ in range(target_count - 1):
        possibilities = []
        for candidate in candidates:
            for row in range(band + gap, length - 2):
                if not _compatible(candidate, row, selected):
                    continue
                if mission == "narrow" and len(selected) == 1:
                    if row != selected[0][0] or min(abs(candidate[0] - selected[0][2]),
                                                      abs(selected[0][1] - candidate[1])) != 2:
                        continue
                if mission == "staggered" and len(selected) == 1 and row == selected[0][0]:
                    continue
                possibilities.append((row, *candidate))
        if not possibilities:
            break
        selected.append(possibilities[int(rng.integers(len(possibilities)))])
    if mission in ("narrow", "staggered", "all_hit") and len(selected) < 2:
        return None
    a = np.full((length, width), palette[0], dtype=np.int8)
    a[:band, list(guides)] = palette[1]
    for row, start, end, _, _ in selected:
        if np.any(a[row, start:end + 1] != palette[0]):
            return None
        a[row, start:end + 1] = palette[2]
    if mission == "unhit":
        choices = []
        for row in range(band + gap, length - 2):
            if any(abs(row - old[0]) < 2 for old in selected):
                continue
            for size in range(2, 5):
                for start in range(1, width - size):
                    end = start + size - 1
                    if not any(start <= g <= end for g in guides):
                        choices.append((row, start, end))
        if not choices:
            return None
        row, start, end = choices[int(rng.integers(len(choices)))]
        a[row, start:end + 1] = palette[2]
    displayed = _from_oriented(a, direction)
    info = _interpret_input(displayed)
    if info is None or info["direction"] != direction:
        return None
    features = _features(info)
    if mission in features and not features[mission]:
        return None
    if mission == "tie" and not features["clear_tie"]:
        return None
    if mission == "no_tie" and features["tie"]:
        return None
    return displayed, info["output"], info


def _generate_pair(rng: np.random.Generator, regime: str, mission: str,
                   direction: str, palette: tuple[int, int, int]) -> tuple:
    for _ in range(700):
        result = _build_pair(rng, regime, mission, direction, palette)
        if result is not None and not _source_copy(result[0], result[1]):
            return result
    raise RuntimeError(f"could not construct {regime} {mission} pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    anchor_palette = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    anchor_direction = _DIRECTIONS[int(rng.integers(4))]
    mixed_palette = bool(rng.integers(0, 4))
    mixed_direction = bool(rng.integers(0, 4))
    tie_mode = bool(rng.integers(0, 4))
    missions = (["tie", "multi"] if tie_mode else ["no_tie", "multi"]) + ["ordinary"] * (train_count - 2)
    extras = ("ordinary", "narrow", "staggered", "all_hit", "unhit", "one_barrier")
    for i in range(2, train_count):
        missions[i] = extras[int(rng.integers(len(extras)))]
    rng.shuffle(missions)
    test_missions = [extras[int(rng.integers(len(extras)))] if tie_mode else "no_tie"
                     for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    tie_directions = set()
    for split, assigned in (("train", missions), ("test", test_missions)):
        regime = "source_train" if split == "train" else "target_test"
        for index, mission in enumerate(assigned):
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(pair_seed)
            for _ in range(100):
                if mixed_direction:
                    direction = _DIRECTIONS[int(pair_rng.integers(4))]
                else:
                    direction = anchor_direction
                if mixed_palette:
                    palette = tuple(int(x) for x in pair_rng.choice(10, size=3, replace=False))
                else:
                    palette = anchor_palette
                try:
                    input_grid, output_grid, info = _generate_pair(
                        pair_rng, regime, mission, direction, palette)
                except RuntimeError:
                    continue
                signature = _grid_signature(input_grid)
                if signature in seen:
                    continue
                if not tie_mode and _features(info)["tie"]:
                    continue
                if split == "test" and _features(info)["tie"] and direction not in tie_directions:
                    continue
                break
            else:
                raise RuntimeError("could not generate unique recoverable pair")
            seen.add(signature)
            puzzle[split].append({"input": input_grid, "output": output_grid})
            if split == "train" and _features(info)["clear_tie"]:
                tie_directions.add(direction)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")), "split lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_infos, test_infos = [], []
    for split, infos in (("train", train_infos), ("test", test_infos)):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "grid colors")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "shape or change")
            insig, pairsig = _grid_signature(inp), _pair_signature(inp, out)
            _require(insig not in seen_inputs and pairsig not in seen_pairs, "duplicate pair")
            seen_inputs.add(insig)
            seen_pairs.add(pairsig)
            _require(not _source_copy(inp, out), "copied source pair")
            info = _interpret_input(inp)
            _require(info is not None, "input has ambiguous or invalid geometry")
            _require(np.array_equal(info["output"], out), "routing differs from visible-input rule")
            infos.append(info)
    train_features = [_features(info) for info in train_infos]
    _require(any(f["multi"] for f in train_features), "independent multi-guide routing missing")
    train_tie_directions = {info["direction"] for info, f in zip(train_infos, train_features) if f["clear_tie"]}
    for info in test_infos:
        if _features(info)["tie"]:
            _require(info["direction"] in train_tie_directions, "test tie lacks orientation witness")


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(not _puzzles_equal(puzzles[0], puzzles[1]), "cross-seed difference")
    for counts in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(8100 + 10 * counts[0] + counts[1], *counts)
        _require((len(p["train"]), len(p["test"])) == counts, "explicit counts")
    for i, (nt, nv) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(8300 + i, nt, nv)
        _require((nt is None or len(p["train"]) == nt) and
                 (nv is None or len(p["test"]) == nv), "partial counts")
    p = generate_puzzle(7001, 6, 3)
    _require(_puzzles_equal(p, generate_puzzle(7001, 6, 3)), "determinism")
    for nt, nv in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(991, nt, nv)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    infos = [_interpret_input(pair["input"]) for p in puzzles
             for split in ("train", "test") for pair in p[split]]
    fs = [_features(info) for info in infos]
    _require(len({i["direction"] for i in infos}) == 4, "orientation diversity")
    _require(len({tuple(pair["input"].shape) for p in puzzles
                  for split in ("train", "test") for pair in p[split]}) >= 30, "size diversity")
    _require(len({(i["background"], i["guide"], i["barrier"]) for i in infos}) >= 25, "palette diversity")
    _require({0, 1, 2}.issubset({g for f in fs for g in f["gaps"]}), "gap diversity")
    for property_name in ("narrow", "staggered", "all_hit", "unhit", "one_barrier", "multi"):
        _require(any(f[property_name] for f in fs), f"missing {property_name} examples")
    _require(len({len(i["guides"]) for i in infos}) >= 5, "guide density diversity")
    _require(len({len(i["obstacles"]) for i in infos}) >= 4, "barrier count diversity")
    _require(len({size for f in fs for size in f["lengths"]}) >= 4, "barrier length diversity")
    _require(any(len({i["direction"] for i in [_interpret_input(pair["input"]) for split in ("train", "test") for pair in p[split]]}) > 1 for p in puzzles), "within-puzzle directions")
    _require(any(len({(_interpret_input(pair["input"])["background"], _interpret_input(pair["input"])["guide"], _interpret_input(pair["input"])["barrier"]) for split in ("train", "test") for pair in p[split]}) > 1 for p in puzzles), "within-puzzle palettes")
    puzzle_info_groups = [[_interpret_input(pair["input"])
                           for split in ("train", "test") for pair in p[split]]
                          for p in puzzles]
    _require(sum(len({gap for i in group for gap in _features(i)["gaps"]}) > 1
                 for group in puzzle_info_groups) >= 30, "within-puzzle gap diversity")
    _require(sum(len({len(i["obstacles"]) for i in group}) > 1
                 for group in puzzle_info_groups) >= 25, "within-puzzle barrier diversity")
    _require(sum(len({len(i["guides"]) for i in group}) > 1
                 for group in puzzle_info_groups) >= 25, "within-puzzle guide diversity")
    _require(any(not any(_features(_interpret_input(pair["input"]))["tie"]
                         for split in ("train", "test") for pair in p[split])
                 for p in puzzles), "missing no-tie puzzles")
    _require(any(len({_interpret_input(pair["input"])["direction"]
                      for split in ("train", "test") for pair in p[split]}) == 1
                 for p in puzzles), "missing uniform-direction puzzles")
    _require(any(len({tuple(_interpret_input(pair["input"])[k]
                             for k in ("background", "guide", "barrier"))
                      for split in ("train", "test") for pair in p[split]}) == 1
                 for p in puzzles), "missing uniform-palette puzzles")
    _require(all(any(_features(_interpret_input(pair["input"]))["multi"]
                     for pair in p["train"]) for p in puzzles),
             "multi-guide evidence missing")
    for split in ("train", "test"):
        split_infos = [_interpret_input(pair["input"]) for p in puzzles
                       for pair in p[split]]
        split_features = [_features(i) for i in split_infos]
        _require({0, 1, 2, 3}.issubset({gap for f in split_features
                                         for gap in f["gaps"]}),
                 f"{split} gap diversity")
        _require(any(f["one_barrier"] for f in split_features)
                 and any(len(i["obstacles"]) >= 3 for i in split_infos),
                 f"{split} barrier-count diversity")
        _require(any(f["all_hit"] for f in split_features)
                 and any(f["straight"] for f in split_features),
                 f"{split} hit diversity")
        _require(any(f["staggered"] for f in split_features)
                 and any(f["unhit"] for f in split_features)
                 and any(f["narrow"] for f in split_features),
                 f"{split} obstacle-layout diversity")
    for p in puzzles:
        for split in ("train", "test"):
            for pair in p[split]:
                info = _interpret_input(pair["input"])
                a, b = pair["input"], pair["output"]
                _require(np.array_equal(a[a == info["barrier"]], b[a == info["barrier"]]),
                         "barrier changed")
                _require(np.array_equal(a[a == info["guide"]], b[a == info["guide"]]),
                         "existing guide changed")
                _require(bool(np.all(b[a != b] == info["guide"])
                              and np.all(a[a != b] == info["background"])),
                         "non-background cell changed")
