"""Conductive cross switchboards: boundary-selected contact windows and relays.

Only Python's standard library and NumPy are used. No parent code or puzzle is
loaded at runtime. The input alone specifies every color and geometric role.
"""
from collections import deque
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= x <= 30 for x in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _edge(line, background):
    positions = np.flatnonzero(line != background)
    if not len(positions) or np.any(np.diff(positions) != 1):
        raise ValueError("edge segment must be nonempty and contiguous")
    color = int(line[positions[0]])
    if not np.all(line[positions] == color):
        raise ValueError("edge segment must be monochromatic")
    return (int(positions[0]), int(positions[-1]) + 1), color


def _parse(input_grid):
    """Find the unique orientation, divider, bars, and colored singleton labels."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must have unique maximum frequency")
    background = int(colors[np.argmax(counts)])
    candidates = []
    for transpose in (False, True):
        plane = grid.T if transpose else grid
        p, q = plane.shape
        if p < 5 or q < 3:
            continue
        try:
            first, first_color = _edge(plane[0], background)
            second, second_color = _edge(plane[-1], background)
        except ValueError:
            continue
        if first_color == second_color or first[1] - first[0] == second[1] - second[0]:
            continue
        common = max(first[0], second[0]), min(first[1], second[1])
        if common[0] >= common[1]:
            continue
        for divider in range(2, p - 2):
            line = plane[divider]
            separator = int(line[0])
            if separator in (background, first_color, second_color) or not np.all(line == separator):
                continue
            interior = plane.copy()
            interior[[0, -1], :] = background
            interior[divider, :] = background
            coordinates = np.argwhere(interior != background)
            n = len(coordinates)
            if not 2 <= n <= 8:
                continue
            markers = sorted((int(plane[r, c]), int(r), int(c)) for r, c in coordinates)
            if (len({x[0] for x in markers}) != n
                    or len({x[1] for x in markers}) != n
                    or len({x[2] for x in markers}) != n
                    or separator in {x[0] for x in markers}
                    or not {first_color, second_color} <= {x[0] for x in markers}):
                continue
            longer_first = first[1] - first[0] > second[1] - second[0]
            source_color = first_color if longer_first else second_color
            source = next(i for i, x in enumerate(markers) if x[0] == source_color)
            candidates.append(dict(plane=plane, transpose=transpose, background=background,
                                   separator=separator, divider=divider, first=first,
                                   second=second, common=common, longer_first=longer_first,
                                   markers=markers, source=source))
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous switchboard reading")
    return candidates[0]


def _contact_graph(info):
    """Off-diagonal cross intersections conduct only inside the B window."""
    markers = info["markers"]
    rows = np.array([x[1] for x in markers])
    cols = np.array([x[2] for x in markers])
    row_inside = rows < info["divider"] if info["longer_first"] else rows > info["divider"]
    col_inside = (info["common"][0] <= cols) & (cols < info["common"][1])
    contacts = row_inside[:, None] & col_inside[None, :]
    graph = contacts | contacts.T
    np.fill_diagonal(graph, False)
    return graph


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    graph = _contact_graph(info)
    active = np.zeros(len(info["markers"]), dtype=bool)
    active[info["source"]] = True
    while True:
        expanded = active | np.any(graph[active], axis=0)
        if np.array_equal(active, expanded):
            break
        active = expanded
    plane = info["plane"]
    output = plane.copy()
    hits = np.zeros(plane.shape, dtype=np.int8)
    ink = np.full(plane.shape, info["background"], dtype=np.int8)
    for enabled, (color, row, col) in zip(active, info["markers"]):
        if not enabled:
            continue
        cross = np.zeros(plane.shape, dtype=bool)
        cross[row, :] = True
        cross[1:-1, col] = True
        cross[info["divider"], :] = False
        hits += cross
        ink[cross] = color
    output[hits == 1] = ink[hits == 1]
    output[hits >= 2] = info["separator"]
    return output.T.copy() if info["transpose"] else output


def _reach_by_crossings(markers, source, divider, first_side, common):
    """Construction/check path: enumerate two crossing coordinates per pair."""
    graph = [set() for _ in markers]
    for i, (_, r1, c1) in enumerate(markers):
        for j in range(i):
            _, r2, c2 = markers[j]
            for row, col in ((r1, c2), (r2, c1)):
                on_side = row < divider if first_side else row > divider
                if on_side and common[0] <= col < common[1]:
                    graph[i].add(j)
                    graph[j].add(i)
                    break
    distances = {source: 0}
    queue = deque([source])
    while queue:
        i = queue.popleft()
        for j in sorted(graph[i]):
            if j not in distances:
                distances[j] = distances[i] + 1
                queue.append(j)
    return graph, distances


def _paint_known(plane, markers, active, divider, separator):
    """Cellwise constructor; does not call solve or its vectorized painter."""
    output = plane.copy()
    for row in range(1, len(plane) - 1):
        if row == divider:
            continue
        for col in range(plane.shape[1]):
            inks = [color for i, (color, mr, mc) in enumerate(markers)
                    if i in active and (row == mr or col == mc)]
            if inks:
                output[row, col] = inks[0] if len(inks) == 1 else separator
    return output


def _types(rng, n, profile, width):
    # Bits indicate row in selected half / column in common interval.
    patterns = {"row": [(1, 0), (0, 1), (1, 0), (0, 0)],
                "column": [(0, 1), (1, 0), (0, 1), (0, 0)],
                "isolated": [(0, 0), (1, 0), (0, 1), (1, 0)],
                "both": [(1, 1), (1, 0), (0, 1), (0, 0)],
                "direct": [(1, 0), (0, 1), (0, 0), (0, 0)]}
    types = patterns[profile].copy()
    while len(types) < n:
        choices = [(0, 0), (1, 0)]
        if sum(t[1] for t in types) < width:
            choices += [(0, 1), (1, 1)]
        types.append(choices[int(rng.integers(len(choices)))])
    return types


def _intervals(rng, q, width, relation):
    if relation == "partial":
        starts = [i for i in range(1, q - width) if max(i, q - width - i) >= 2]
        start = int(rng.choice(starts))
        extensions = [(a, b) for a in range(1, start + 1)
                      for b in range(1, q - width - start + 1) if a != b]
        a, b = extensions[int(rng.integers(len(extensions)))]
        left, right = (start - a, start + width), (start, start + width + b)
        longer, shorter = (left, right) if a > b else (right, left)
    else:
        start = int(rng.integers(q - width + 1))
        shorter = start, start + width
        extensions = [(a, b) for a in range(start + 1)
                      for b in range(q - start - width + 1) if a + b > 0]
        if rng.random() < 0.2:
            a, b = start, q - start - width
        else:
            a, b = extensions[int(rng.integers(len(extensions)))]
        longer = start - a, start + width + b
    return longer, shorter, (start, start + width)


def _make_pair(rng, n, profile, width, relation, first_side, transpose):
    types = _types(rng, n, profile, width)
    nrow = sum(t[0] for t in types)
    ncol = sum(t[1] for t in types)
    selected_gap = max(2, nrow) + int(rng.integers(0, 8))
    other_gap = max(2, n - nrow) + int(rng.integers(0, 8))
    top_gap, bottom_gap = (selected_gap, other_gap) if first_side else (other_gap, selected_gap)
    p, divider = top_gap + bottom_gap + 3, top_gap + 1
    minimum = max(8, n + 2, width + n - ncol + 2, width + 4)
    upper = min(30, minimum + int(rng.integers(2, 16)))
    q = int(rng.integers(minimum, upper + 1))
    longer, shorter, common = _intervals(rng, q, width, relation)
    first, second = (longer, shorter) if first_side else (shorter, longer)
    row_true = list(range(1, divider)) if first_side else list(range(divider + 1, p - 1))
    row_false = list(range(divider + 1, p - 1)) if first_side else list(range(1, divider))
    col_true = list(range(*common))
    col_false = [c for c in range(q) if c not in col_true]
    pools = [row_false, row_true, col_false, col_true]
    for pool in pools:
        rng.shuffle(pool)
    palette = [int(c) for c in rng.choice(10, n + 2, replace=False)]
    background, separator = palette[:2]
    marker_colors = palette[2:]
    # Put an isolated label in an exclusive projection. Across the first two
    # demonstrations this rejects either whole bar, as well as their union.
    forced_index = 0 if profile == "isolated" else 3
    forced_col = None
    if relation == "partial":
        containing, excluding = (longer, shorter) if profile == "isolated" else (shorter, longer)
        exclusive = [c for c in range(*containing) if not excluding[0] <= c < excluding[1]]
        forced_col = int(rng.choice(exclusive))
        pools[2].remove(forced_col)
    markers = []
    for i, t in enumerate(types):
        row = pools[t[0]].pop()
        col = forced_col if i == forced_index and forced_col is not None else pools[2 + t[1]].pop()
        markers.append((marker_colors[i], row, col))
    # The longer bar labels marker 0. The shorter bar labels an unpowered
    # marker, or a member of the disconnected component in isolated cases.
    short_index = 1 if profile == "isolated" else 3
    long_color, short_color = marker_colors[0], marker_colors[short_index]
    first_color, second_color = (long_color, short_color) if first_side else (short_color, long_color)
    plane = np.full((p, q), background, dtype=np.int8)
    plane[divider, :] = separator
    plane[0, first[0]:first[1]] = first_color
    plane[-1, second[0]:second[1]] = second_color
    for color, row, col in markers:
        plane[row, col] = color
    _, distances = _reach_by_crossings(markers, 0, divider, first_side, common)
    output = _paint_known(plane, markers, distances, divider, separator)
    if rng.random() < 0.5:
        plane, output = plane[:, ::-1].copy(), output[:, ::-1].copy()
    if transpose:
        plane, output = plane.T.copy(), output.T.copy()
    return {"input": plane, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    narrow_shift = bool(rng.integers(2))
    base_orientation, first_side = bool(rng.integers(2)), bool(rng.integers(2))
    profiles = ["row", "isolated"]
    if num_train >= 3:
        profiles.append("column")
    if num_train >= 4:
        profiles.append("both" if rng.random() < 0.5 else "direct")
    while len(profiles) < num_train:
        profiles.append(str(rng.choice(["row", "column", "both", "direct", "isolated"])))
    puzzle, used = {"train": [], "test": []}, set()
    trained_max, train_widths = 4, []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                profile = profiles[index]
                low = 2 if narrow_shift or profile in ("column", "both") else 1
                width = int(rng.integers(low, 6 if narrow_shift else 4))
                n = 4 if index < 2 else int(rng.integers(4, 9))
                trained_max = max(trained_max, n)
                train_widths.append(width)
                relation = "partial" if index < 2 else ("contained" if index < 4 else str(rng.choice(["partial", "contained"])))
                side = first_side if index % 2 == 0 else not first_side
                orientation = base_orientation if index < 2 else not base_orientation if index == 2 else bool(rng.integers(2))
            else:
                width = (1 if narrow_shift else int(rng.integers(max(train_widths) + 1, 8))) if index == 0 else int(rng.integers(1, 8))
                choices = ["row", "direct", "isolated"] + (["column", "both"] if width >= 2 else [])
                profile = "row" if index == 0 else str(rng.choice(choices))
                n = int(rng.integers(4, trained_max + 1))
                relation = "partial" if num_train == 2 else str(rng.choice(["partial", "contained"]))
                side = bool(rng.integers(2))
                orientation = base_orientation if num_train == 2 else bool(rng.integers(2))
            for _ in range(50):
                pair = _make_pair(rng, n, profile, width, relation, side, orientation)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct switchboards")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    _require(grid.shape == output.shape, "dimensions changed")
    info = _parse(grid)
    plane, expected_plane = info["plane"], output.T if info["transpose"] else output
    graph, distances = _reach_by_crossings(info["markers"], info["source"], info["divider"],
                                         info["longer_first"], info["common"])
    expected = _paint_known(plane, info["markers"], distances, info["divider"], info["separator"])
    _require(np.array_equal(expected_plane, expected), "incorrect activation or cross painting")
    _require(np.array_equal(solve(grid), output), "input-only solver disagrees")
    _require(np.all(output[grid != info["background"]] == grid[grid != info["background"]]),
             "an existing object was overwritten")
    _require(np.any(output != grid), "no expansion")
    active_count = len(distances)
    # Distinct marker axes give exactly two off-diagonal intersections per
    # active unordered pair; the protected divider is on no marker row.
    new_knots = np.count_nonzero((expected_plane == info["separator"]) & (plane != info["separator"]))
    _require(new_knots == active_count * (active_count - 1), "wrong knot count")
    for i, (_, row, col) in enumerate(info["markers"]):
        if i not in distances:
            _require(not (graph[i] & distances.keys()), "activation stopped before closure")
        _require(expected_plane[row, col] == plane[row, col], "marker lost")
    info.update(graph=graph, distances=distances, active_count=active_count)
    return info


def _ablation_changes(info):
    """Witness that the demonstrations reject specific simpler rules."""
    markers, root = info["markers"], info["source"]
    actual = set(info["distances"])
    p, q = info["plane"].shape
    first, second = info["first"], info["second"]
    shorter = second if info["longer_first"] else first
    union = min(first[0], second[0]), max(first[1], second[1])
    def reached(divider, side, common, source=root):
        return set(_reach_by_crossings(markers, source, divider, side, common)[1])
    side, divider, common = info["longer_first"], info["divider"], info["common"]
    short_color = int(info["plane"][-1 if side else 0, shorter[0]])
    short_root = next(i for i, x in enumerate(markers) if x[0] == short_color)
    return {
        "source_only": actual != {root},
        "one_hop": actual != ({root} | info["graph"][root]),
        "all_crosses": len(actual) != len(markers),
        "ignore_side": actual != reached(p, True, common),
        "opposite_side": actual != reached(divider, not side, common),
        "shorter_projection": actual != reached(divider, side, shorter),
        "longer_projection": actual != reached(divider, side, first if side else second),
        "union_projection": actual != reached(divider, side, union),
        "shorter_source": actual != reached(divider, not side, common, short_root),
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    t, q = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= t <= 10 and 1 <= q <= 4 and t > q, "invalid counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(info)
    train = records["train"]
    _require(any(max(x["distances"].values()) >= 2 for x in train), "no relay demonstration")
    _require(any(x["active_count"] == 1 and any(x["graph"]) for x in train),
             "need an unpowered connected component demonstration")
    _require({x["longer_first"] for x in train} == {False, True}, "longer side never changes")
    _require(all(any(i not in x["distances"] for i in range(len(x["markers"]))) for x in train),
             "training must expose inactive markers")
    ablations = [_ablation_changes(x) for x in train]
    _require(all(any(record[name] for record in ablations) for name in ablations[0]),
             "demonstrations do not distinguish a required simpler alternative")
    maximum = max(len(x["markers"]) for x in train)
    _require(all(len(x["markers"]) <= maximum for x in records["test"]), "undemonstrated marker count")
    widths = [x["common"][1] - x["common"][0] for x in train]
    test_width = records["test"][0]["common"][1] - records["test"][0]["common"][0]
    _require(test_width < min(widths) or test_width > max(widths), "missing aperture-width shift")


_SEEDS = (0, 40077829, 920037902, 590559564, 351912155, 418323660, 335027946,
          330522273, 188123253, 538339597, 397543522, 364300886, 90625162,
          24438304, 755278894, 336556344, 987805248, 242923700, 427378883,
          672136151, 708941074, 62801586, 839953617, 301883419, 929344457,
          424509613, 708050601, 724458577, 910270869, 451223493, 705420996,
          863738178, 732465363, 257148704, 409942560, 422723275, 47435061,
          605660270, 948930536, 385422938, 857920632, 865838880)


def _signature(puzzle):
    return tuple((split, _key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded semantic, metamorphic, count, reproducibility and diversity checks."""
    signatures, shapes, layouts, marker_counts = set(), set(), set(), set()
    active_counts, widths, orientations, palettes, shifts = (set() for _ in range(5))
    relay_pairs = isolated_pairs = 0
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterminism")
        signatures.add(_signature(puzzle))
        train_widths = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, info["divider"], info["first"], info["second"],
                             tuple((r, c) for _, r, c in info["markers"])))
                marker_counts.add(len(info["markers"]))
                active_counts.add(info["active_count"])
                widths.add(info["common"][1] - info["common"][0])
                orientations.add(info["transpose"])
                palettes.add((info["background"], info["separator"]))
                relay_pairs += max(info["distances"].values()) >= 2
                isolated_pairs += info["active_count"] == 1
                if split == "train":
                    train_widths.append(info["common"][1] - info["common"][0])
        width = _parse(puzzle["test"][0]["input"])["common"]
        shifts.add("narrow" if width[1] - width[0] < min(train_widths) else "wide")
        # Geometry and colors carry roles, so all dihedral transforms and any
        # bijective recoloring must commute with solve, even beyond the sampler.
        pair = puzzle["train"][0]
        for k in range(4):
            for flip in (False, True):
                x, y = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                if flip:
                    x, y = x[:, ::-1], y[:, ::-1]
                _require(np.array_equal(solve(x), y), "dihedral equivariance failed")
        permutation = np.random.default_rng(seed + 100).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "palette equivariance failed")
    _require(len(signatures) == 42 and len(layouts) >= 190, "layout diversity collapsed")
    _require(len(shapes) >= 65 and len(palettes) >= 45, "geometry/palette diversity collapsed")
    _require(marker_counts >= {4, 5, 6, 7, 8}, "marker-count diversity collapsed")
    _require(active_counts >= {1, 2, 3} and len(widths) >= 6, "network/aperture diversity collapsed")
    _require(orientations == {False, True} and len(shifts) == 2, "orientation/shift diversity collapsed")
    _require(relay_pairs >= 80 and isolated_pairs >= 42, "relay/isolation coverage collapsed")
    # Valid family members beyond the sampler: 2/3 labels, with the shorter
    # bar's label powered too. No demonstration schedule is needed by solve.
    for n in (2, 3):
        plane = np.zeros((11, 9), dtype=np.int8)
        plane[5, :] = 9
        plane[0, 1:7], plane[-1, 4:8] = 1, 2
        markers = [(1, 2, 0), (2, 7, 4), (3, 3, 8)][:n]
        for color, row, col in markers:
            plane[row, col] = color
        expected = _paint_known(plane, markers, set(range(n)), 5, 9)
        _require(np.array_equal(solve(plane), expected), "small switchboard support failed")
    for t in range(2, 11):
        for q in range(1, min(4, t - 1) + 1):
            generate_puzzle(123456 + 10 * t + q, t, q)
    for t, q in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (4, True), (None, 1), (4, None)):
        try:
            generate_puzzle(0, t, q)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {t}/{q}")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, dihedral and palette checks")
