"""Canonical orthogonal lines recovered from independent band expansion.

No parent generator is imported. Answers are constructed as latent arrangements;
solve observes only the supplied input. All diagnostics stay in memory.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


# Input fingerprints of both originals and both selected variations. These are
# only a generation exclusion list; solve never reads it or any saved example.
_PARENT_INPUTS = frozenset("""
06e48453767fce82f754d45e007a59742bb611a2d1fbca5bb49009d9cf260ff0
114db8251b5e71f2968154cbcb64f6b2f527013fad09e4c90732a7085dbed4c3
1c2d4149f1012c0d4eeb8c8566b7accf643654ed2411f93be87ea5fda5e8377f
1df01b0449ec03449c2ceffeaf4a1c4da6690e052058e3d39e6a682ac9b72c79
23676a4656dc0ffc41eca13b4920a7708d6234af51beab779149ffebb20776e2
34010743088c49f34b217345831679188b850635fd0d12ab97b0349b5c9aa74f
3f8399862bbff4ee0f9a111fa4c9639418a7d2d872cd5bdda236e8e65f483b65
4b0ed0b892056f04ecfaae8ad486f632db405e6a3f3c2c72635c4baaf6042cad
4de4f0c610e46c7ecdc32b35f0342691787ac544c30ffa3c580efe07a3957d3d
6439552185cb29412719a66da60400489041d2429ecb50f12f7cd086bfcefcd7
80f470f3f300c392e12dd126da653d9e6c1b9b133aea78d0b3ea487bb1b17014
85e50d1ca14c8a75b3eea7cf64fbcfb15d9e26501c6196e635848f802ece1b09
8c004e0c955221107407b62a2ab8ae7a56c33377d0b650865f43417ecc32d30a
903288eaca8a3babdda5e7728a0a8e46886dfea178615a73bb4fd362c3fa3afa
a0b9b5355e6485ddccc6f796a76634dc7057abe38bdfbd71c73c5949acbdc4ce
acf063ae053337f55aa51b94e70579b20b426722c6ad20e42598e8c393e8fc49
d420745db94e5fe3d22d6cf6a049b06fb423f7ef8e9d36b2c30772d53167a3df
d4a655409cfc1c34c18aadd3fb5f625ae14148aa260d62bd8be98003c6d2d5b8
dbc037cd3a612ba3f07b42c97712b8a5b63012f39b5b5646851877dc90da9676
f292ab7c7488a9e18e7137ddce326e81e3b50ee4e8aefd3b94a7a0cda75cbaf6
""".split())

_TEST_SEEDS = (
    0, 352036594, 337061493, 237556334, 464130203, 560578850,
    803181785, 617444568, 143737064, 521252940, 459554563,
    736743228, 786146220, 33559530, 148348852, 14791332,
    208252850, 177073716, 671292441, 326204414, 480187195,
    56546191, 456188857, 916079092, 248628346, 116105490,
    249743225, 961413506, 629386181, 848559120, 207787755,
    685727989, 213892749, 799330337, 61570343, 15230749,
    460308480, 317276558, 509895941, 741851755, 63148517, 414912715,
)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its permitted range")
    return value


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= side <= 30 for side in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return array


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Keep one literal representative of each maximal equal full-line run."""
    grid = _grid(input_grid)
    rows = np.r_[True, np.any(grid[1:, :] != grid[:-1, :], axis=1)]
    columns = np.r_[True, np.any(grid[:, 1:] != grid[:, :-1], axis=0)]
    return grid[np.ix_(rows, columns)].copy()


def _fingerprint(grid):
    array = np.asarray(grid, dtype=np.int8)
    return hashlib.sha256(bytes(array.shape) + array.tobytes()).hexdigest()


def _bands(grid, axis):
    """Loop-based full-line partition used by validation, independent of masks."""
    vectors = grid if axis == 0 else grid.T
    starts = [0]
    for index in range(1, len(vectors)):
        if not np.array_equal(vectors[index - 1], vectors[index]):
            starts.append(index)
    edges = starts + [len(vectors)]
    lengths = tuple(b - a for a, b in zip(edges, edges[1:]))
    return starts, lengths


def _expand(grid, rows, columns):
    return np.repeat(np.repeat(grid, rows, axis=0), columns, axis=1)


def _canonical(rng, h, v, kind, forbidden_background=None):
    backgrounds = [x for x in range(10) if x != forbidden_background]
    background = int(rng.choice(backgrounds))
    palette = rng.permutation([x for x in range(10) if x != background])
    if kind == "anchor":
        horizontal, vertical = palette[:h].copy(), palette[h:h + v].copy()
        # Demonstrate spatial order independently of numerical color order.
        if np.all(horizontal[:-1] < horizontal[1:]):
            horizontal = horizontal[::-1].copy()
    elif kind == "repeat":
        horizontal = np.full(h, palette[0], dtype=np.int8)
        vertical = np.full(v, palette[1], dtype=np.int8)
    else:
        color_mode = int(rng.integers(5))
        if color_mode == 0 and h + v <= 9:
            colors = palette[:h + v].copy()
        else:
            count = int(rng.integers(1, min(9, h + v) + 1))
            colors = rng.choice(palette[:count], size=h + v).copy()
        horizontal, vertical = colors[:h].copy(), colors[h:].copy()
        if color_mode == 1:
            if h >= 2:
                horizontal[-1] = horizontal[0]
            elif v >= 2:
                vertical[-1] = vertical[0]
        elif color_mode == 2:
            vertical[0] = horizontal[0]
        elif color_mode == 3:
            horizontal[:] = palette[0]
            vertical[:] = palette[int(rng.integers(2))]

    mode = int(rng.integers(6))
    if mode < 2:
        bits = np.full((h, v), mode, dtype=np.int8)
    elif mode == 2:
        bits = np.repeat(rng.integers(2, size=(h, 1)), v, axis=1)
    elif mode == 3:
        bits = np.repeat(rng.integers(2, size=(1, v)), h, axis=0)
    elif mode == 4:
        bits = (np.indices((h, v)).sum(axis=0) + int(rng.integers(2))) % 2
    else:
        bits = rng.integers(2, size=(h, v))
    if kind in ("anchor", "repeat"):
        # A four-crossing cycle defeats any global color-priority ordering.
        phase = int(rng.integers(2))
        bits[:2, :2] = np.array([[0, 1], [1, 0]]) ^ phase
    if kind == "repeat":
        # Preserve genuinely repeated colored signatures, as well as gaps.
        if h >= 3:
            bits[-1, :] = bits[0, :]
        if v >= 3:
            bits[:, -1] = bits[:, 0]

    result = np.full((2 * h + 1, 2 * v + 1), background, dtype=np.int8)
    result[1::2, 0::2] = horizontal[:, None]
    result[0::2, 1::2] = vertical[None, :]
    result[1::2, 1::2] = np.where(bits, vertical[None, :], horizontal[:, None])
    return result


def _add_units(rng, runs, positions, extra, cap=6):
    # At most 27 iterations: no unbounded rejection and no large temporaries.
    for _ in range(extra):
        eligible = [p for p in positions if runs[p] < cap]
        if not eligible:
            break
        runs[int(rng.choice(eligible))] += 1


def _runs(rng, length, active=True, anchor=False, wide=False):
    runs = np.ones(length, dtype=np.int64)
    if not active:
        return tuple(int(x) for x in runs)
    if wide:
        position = int(rng.integers(length))
        neighbor = position + 1 if position == 0 else position - 1
        runs[position] = int(rng.integers(9, min(18, 31 - length) + 1))
        positions = [i for i in range(length) if i not in (position, neighbor)]
        extra = int(rng.integers(31 - int(runs.sum())))
        _add_units(rng, runs, positions, extra)
    elif anchor:
        line = int(rng.choice(np.arange(1, length, 2)))
        gap = int(rng.choice(np.arange(0, length, 2)))
        runs[[line, gap]] = 2
        locked = int(rng.choice([i for i in range(length) if i not in (line, gap)]))
        positions = [i for i in range(length) if i != locked]
        extra = int(rng.integers(min(30 - int(runs.sum()), 2 * length) + 1))
        _add_units(rng, runs, positions, extra)
    else:
        mode = int(rng.integers(8))
        if mode == 0:
            runs[:] = int(rng.integers(2, min(6, 30 // length) + 1))
        else:
            if mode == 3:
                positions = [0, length - 1]
            elif mode == 4:
                positions = list(range(1, length - 1))
            elif mode == 5:
                positions = list(range(1, length, 2))
            elif mode == 6:
                positions = list(range(0, length, 2))
            else:
                positions = list(range(length))
            maximum = min(30 - length, 5 * len(positions))
            extra = int(rng.integers(1, maximum + 1))
            if mode in (3, 4):
                bulk = int(rng.choice(positions))
                runs[bulk] += min(5, extra)
                extra -= min(5, extra)
            _add_units(rng, runs, positions, extra)
            if mode in (1, 2):
                runs.sort()
                if mode == 2:
                    runs = runs[::-1]
    return tuple(int(x) for x in runs)


def _make_pair(rng, h, v, kind="free", wide=False, forbidden_background=None):
    output = _canonical(rng, h, v, kind, forbidden_background)
    if kind == "anchor":
        rows = _runs(rng, 2 * h + 1, anchor=True)
        columns = _runs(rng, 2 * v + 1, anchor=True)
    elif wide:
        axis = int(rng.integers(2))
        rows = _runs(rng, 2 * h + 1, active=axis == 0 or rng.random() < .8,
                     wide=axis == 0)
        columns = _runs(rng, 2 * v + 1, active=axis == 1 or rng.random() < .8,
                        wide=axis == 1)
    else:
        operation = int(rng.choice(3, p=[.72, .14, .14]))
        rows = _runs(rng, 2 * h + 1, active=operation != 2)
        columns = _runs(rng, 2 * v + 1, active=operation != 1)
    return {"input": _expand(output, rows, columns), "output": output}


def _distinct_pair(pair, used, forbidden_background):
    key = _fingerprint(pair["input"])
    if key not in used and key not in _PARENT_INPUTS:
        return pair, key
    # Recolor only on an exact collision. At most 34 candidates can be blocked:
    # 13 previous inputs + 20 parent inputs. There are at least 81 allowed maps.
    background = int(pair["output"][0, 0])
    values = [background] + [int(x) for x in np.unique(pair["input"]) if x != background]
    for new_background in range(10):
        if new_background == forbidden_background:
            continue
        available = [c for c in range(10) if c != new_background]
        for tail in itertools.permutations(available, len(values) - 1):
            lookup = np.arange(10, dtype=np.int8)
            lookup[values] = (new_background,) + tail
            candidate = {role: lookup[grid] for role, grid in pair.items()}
            key = _fingerprint(candidate["input"])
            if key not in used and key not in _PARENT_INPUTS:
                return candidate, key
    raise RuntimeError("exhausted injective recolorings")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first = (int(rng.integers(2, 4)), int(rng.integers(2, 4)))
    second = (int(rng.integers(3, 5)), int(rng.integers(2, 5)))
    if rng.random() < .5:
        second = second[::-1]
    if second == first:
        second = (4, second[1])
    train_shapes = [first, second]
    train_shapes += [(int(rng.integers(1, 6)), int(rng.integers(1, 6)))
                     for _ in range(num_train - 2)]
    max_h = max(h for h, _ in train_shapes)
    max_v = max(v for _, v in train_shapes)
    modes = [int(x) for x in rng.permutation(4)[:num_test]]
    tests = []
    for mode in modes:
        if mode == 0:
            shape = (max_h + 1, int(rng.choice([v for _, v in train_shapes])))
        elif mode == 1:
            shape = (int(rng.choice([h for h, _ in train_shapes])), max_v + 1)
        elif mode == 2:
            shape = (max_h + 1, max_v + 1)
        else:
            shape = (int(rng.integers(1, max_h + 1)), int(rng.integers(1, max_v + 1)))
        tests.append((shape, mode == 3))

    puzzle, used = {"train": [], "test": []}, set()
    for index, (h, v) in enumerate(train_shapes):
        forbidden = int(puzzle["train"][0]["output"][0, 0]) if index == 1 else None
        kind = ("anchor", "repeat")[index] if index < 2 else "free"
        pair = _make_pair(rng, h, v, kind, forbidden_background=forbidden)
        pair, key = _distinct_pair(pair, used, forbidden)
        puzzle["train"].append(pair)
        used.add(key)
    for (h, v), wide in tests:
        pair, key = _distinct_pair(_make_pair(rng, h, v, wide=wide), used, None)
        puzzle["test"].append(pair)
        used.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    source, output = _grid(pair["input"]), _grid(pair["output"])
    row_starts, rows = _bands(source, 0)
    column_starts, columns = _bands(source, 1)
    retained = source[np.ix_(row_starts, column_starts)]
    _require(np.array_equal(retained, output), "literal representative values changed")
    _require(np.array_equal(_expand(output, rows, columns), source), "expansion inverse")
    _require(np.array_equal(solve(source), output), "solve disagrees")
    _require(source.shape != output.shape, "unchanged generated pair")
    _require(np.array_equal(np.unique(source), np.unique(output)), "lost a color")
    height, width = output.shape
    _require(height >= 3 and width >= 3 and height % 2 == width % 2 == 1, "line dimensions")
    h, v = (height - 1) // 2, (width - 1) // 2
    background = int(output[0, 0])
    horizontal = tuple(int(x) for x in output[1::2, 0])
    vertical = tuple(int(x) for x in output[0, 1::2])
    _require(np.all(output[0::2, 0::2] == background), "spacing signature")
    _require(all(c != background for c in horizontal + vertical), "invisible line")
    _require(np.all(output[1::2, 0::2] == np.asarray(horizontal)[:, None]), "horizontal signature")
    _require(np.all(output[0::2, 1::2] == np.asarray(vertical)[None, :]), "vertical signature")
    owners, by_pair = set(), {}
    for i, hc in enumerate(horizontal):
        for j, vc in enumerate(vertical):
            crossing = int(output[2 * i + 1, 2 * j + 1])
            _require(crossing in (hc, vc), "illegal crossing color")
            if hc != vc:
                owner = int(crossing == vc)
                owners.add(owner)
                by_pair.setdefault((hc, vc), set()).add(owner)
    repeated_colored = (any(np.array_equal(output[a], output[b])
                            for a in range(1, height, 2) for b in range(a + 2, height, 2))
                        or any(np.array_equal(output[:, a], output[:, b])
                               for a in range(1, width, 2) for b in range(a + 2, width, 2)))
    return {"h": h, "v": v, "background": background, "rows": rows, "columns": columns,
            "distinct": len(set(horizontal + vertical)) == h + v,
            "mixed": owners == {0, 1}, "pair_conflict": any(len(x) == 2 for x in by_pair.values()),
            "repeated_colored": repeated_colored,
            "parallel_reuse": len(set(horizontal)) < h or len(set(vertical)) < v,
            "cross_reuse": bool(set(horizontal) & set(vertical)),
            "all_h": owners == {0}, "all_v": owners == {1}}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "counts")
    analyses, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze_pair(pair)
            key = _fingerprint(pair["input"])
            _require(key not in seen and key not in _PARENT_INPUTS, "duplicate or parent input")
            seen.add(key)
            analyses[split].append(analysis)
    train, test = analyses["train"], analyses["test"]
    _require(len({(x["h"], x["v"]) for x in train}) >= 2, "training shape variation")
    _require(len({x["background"] for x in train}) >= 2, "local background convention")
    _require(all(max(x["rows"] + x["columns"]) <= 6 for x in train), "training band cap")
    _require(any(x["distinct"] and x["mixed"]
                 and all(max(x[axis][1::2]) > 1 and max(x[axis][0::2]) > 1
                         and len(set(x[axis])) > 1 for axis in ("rows", "columns"))
                 for x in train), "missing two-axis thick-line/thick-gap demonstration")
    _require(any(x["pair_conflict"] and x["repeated_colored"] for x in train),
             "missing separated colored repeats or color-priority counterexample")
    max_h, max_v = max(x["h"] for x in train), max(x["v"] for x in train)
    _require(len({(x["h"], x["v"]) for x in test}) == b, "test inventory diversity")
    for x in test:
        _require(x["h"] > max_h or x["v"] > max_v
                 or max(x["rows"] + x["columns"]) >= 9, "missing test shift")


def _puzzle_digest(puzzle):
    data = "".join(_fingerprint(p[role]) for split in ("train", "test")
                   for p in puzzle[split] for role in ("input", "output"))
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def _rejected_shortcuts(pair):
    """Counterfactual checks, not an independent solver or parent ablation.

    In particular B's full compression rule already preserves all crossings;
    repainting them tests a weaker alternative, not removal of parent A.
    """
    source, output = _grid(pair["input"]), _grid(pair["output"])
    row_starts, rows = _bands(source, 0)
    column_starts, columns = _bands(source, 1)
    candidates = {"identity": source, "rows_only": source[row_starts, :],
                  "columns_only": source[:, column_starts]}
    # Normalize only the spacing bands, leaving thick line bands unchanged.
    indices = []
    for starts, lengths in ((row_starts, rows), (column_starts, columns)):
        indices.append([p for i, (start, length) in enumerate(zip(starts, lengths))
                        for p in (range(start, start + length) if i % 2 else [start])])
    candidates["gaps_only"] = source[np.ix_(*indices)]

    # Remove equal vectors globally rather than only within consecutive runs.
    unique_rows, unique_columns = [], []
    for i in range(source.shape[0]):
        if not any(np.array_equal(source[i], source[j]) for j in unique_rows):
            unique_rows.append(i)
    for i in range(source.shape[1]):
        if not any(np.array_equal(source[:, i], source[:, j]) for j in unique_columns):
            unique_columns.append(i)
    candidates["global_unique"] = source[np.ix_(unique_rows, unique_columns)]

    retained = source[np.ix_(row_starts, column_starts)]
    horizontal, vertical = retained[1::2, 0, None], retained[None, 0, 1::2]
    for name, values in (("horizontal_priority", horizontal),
                         ("vertical_priority", vertical),
                         ("larger_color_priority", np.maximum(horizontal, vertical)),
                         ("smaller_color_priority", np.minimum(horizontal, vertical))):
        candidate = retained.copy()
        candidate[1::2, 1::2] = values
        candidates[name] = candidate
    return {name for name, candidate in candidates.items()
            if not np.array_equal(candidate, output)}


def self_test() -> dict:
    """Bounded: 42 specified seeds, 30 legal count pairs, no file writes."""
    shapes, palettes, inputs, layouts, row_runs, column_runs, puzzles = (set() for _ in range(7))
    metrics = dict(pairs=0, one_axis=0, uniform_axis=0, wide_thin=0,
                   all_horizontal=0, all_vertical=0, mixed=0, parallel_reuse=0,
                   cross_reuse=0, grow_h=0, grow_v=0, thickness_shift=0)
    shortcuts = {name: 0 for name in ("identity", "rows_only", "columns_only", "gaps_only",
                                    "global_unique", "horizontal_priority", "vertical_priority",
                                    "larger_color_priority", "smaller_color_priority")}
    for seed in _TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "determinism")
        puzzles.add(_puzzle_digest(puzzle))
        train = [_analyze_pair(p) for p in puzzle["train"]]
        rejected = set().union(*(_rejected_shortcuts(p) for p in puzzle["train"][:2]))
        _require(rejected == set(shortcuts), "training leaves a listed shortcut viable")
        for name in rejected:
            shortcuts[name] += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, output = pair["input"], pair["output"]
                x = _analyze_pair(pair)
                shapes.add(output.shape)
                inputs.add(source.shape)
                palettes.add(tuple(int(c) for c in np.unique(output)))
                roles = {}
                normalized = np.array([roles.setdefault(int(c), len(roles))
                                       for c in source.ravel()], dtype=np.int8).reshape(source.shape)
                layouts.add(_fingerprint(normalized))
                row_runs.add(x["rows"])
                column_runs.add(x["columns"])
                metrics["pairs"] += 1
                metrics["one_axis"] += (max(x["rows"]) == 1) != (max(x["columns"]) == 1)
                metrics["uniform_axis"] += any(len(set(x[axis])) == 1 and max(x[axis]) > 1
                                               for axis in ("rows", "columns"))
                metrics["wide_thin"] += any(min(a, b) == 1 and max(a, b) >= 9
                                             for runs in (x["rows"], x["columns"])
                                             for a, b in zip(runs, runs[1:]))
                for name, prop in (("all_horizontal", "all_h"), ("all_vertical", "all_v"),
                                   ("mixed", "mixed"), ("parallel_reuse", "parallel_reuse"),
                                   ("cross_reuse", "cross_reuse")):
                    metrics[name] += bool(x[prop])
                if split == "test":
                    metrics["grow_h"] += x["h"] > max(a["h"] for a in train)
                    metrics["grow_v"] += x["v"] > max(a["v"] for a in train)
                    metrics["thickness_shift"] += max(x["rows"] + x["columns"]) >= 9
                original = source.copy()
                _require(np.array_equal(solve(source), output), "solve values")
                _require(np.array_equal(source, original), "solve mutated input")
                _require(np.array_equal(solve(output), output), "idempotence")
                _require(np.array_equal(solve(source.T).T, output), "axis symmetry")
                _require(np.array_equal(solve(9 - source), 9 - output), "palette equivariance")
    _require(len(puzzles) == 42, "seed diversity")
    for count, minimum, label in ((len(shapes), 15, "output shapes"),
                                   (len(inputs), 70, "input shapes"),
                                   (len(layouts), 170, "normalized structures"),
                                   (len(palettes), 30, "palettes"),
                                   (len(row_runs), 50, "row schedules"),
                                   (len(column_runs), 50, "column schedules")):
        _require(count >= minimum, f"insufficient {label}: {count}")
    for name in ("one_axis", "uniform_axis", "wide_thin", "all_horizontal",
                 "all_vertical", "cross_reuse", "grow_h", "grow_v", "thickness_shift"):
        _require(metrics[name] > 0, f"uncovered variation: {name}")

    legal = 0
    for train_count in range(2, 11):
        for test_count in range(1, min(4, train_count - 1) + 1):
            puzzle = generate_puzzle(817, train_count, test_count)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count), "explicit counts")
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4., 1), (4, 1.), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Input-only edge cases beyond the line-arrangement sampling envelope.
    for grid, expected in (([[7]], [[7]]),
                            ([[1, 1, 2, 2, 1]], [[1, 2, 1]]),
                            ([[1], [1], [2], [1]], [[1], [2], [1]]),
                            (np.full((30, 30), 4), [[4]]),
                            ([[1, 1], [1, 2], [1, 1]], [[1, 1], [1, 2], [1, 1]])):
        _require(np.array_equal(solve(np.asarray(grid)), np.asarray(expected)), "edge case")
    # Maximum canonical line inventory, beyond the sampler's six-line cap.
    canonical = _canonical(np.random.default_rng(519), 14, 14, "repeat")
    rows, columns = [2] + [1] * 28, [1] * 28 + [2]
    _require(np.array_equal(solve(_expand(canonical, rows, columns)), canonical),
             "full-domain boundary inventory")
    # Exercise the collision fallback without relying on a random collision.
    pair = generate_puzzle(0)["train"][0]
    replacement, key = _distinct_pair(pair, {_fingerprint(pair["input"])}, None)
    _require(key != _fingerprint(pair["input"]), "collision fallback")
    _analyze_pair(replacement)
    return {"seeds": len(_TEST_SEEDS), "legal_counts": legal, "distinct_puzzles": len(puzzles),
            "output_shapes": len(shapes), "input_shapes": len(inputs),
            "normalized_structures": len(layouts), "palettes": len(palettes),
            "row_schedules": len(row_runs), "column_schedules": len(column_runs),
            "training_shortcuts_rejected": shortcuts, **metrics}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
