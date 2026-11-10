"""Split self-portraits: midpoint territories carrying global source color layers.

Only NumPy and the standard library are used. No parent code or stored examples
are imported. The solver needs only the supplied input.
"""
from __future__ import annotations

import numpy as np


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 2D integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=False)


def _lines(source, background, axis):
    view = source if axis == "H" else source.T
    result, blanks = [], 0
    for lane, row in enumerate(view):
        positions = np.flatnonzero(row != background)
        if len(positions) == 0:
            blanks += 1
            continue
        if len(positions) != 2:
            return None
        a, z = map(int, positions)
        left, right = int(row[a]), int(row[z])
        if z - a not in (2, 4) or left == right:
            return None
        result.append((lane, a, z, left, right, (a + z) // 2))
    return tuple(result) if result and blanks else None


def _analyze(value):
    source = _grid(value)
    if not all(2 <= n <= 5 for n in source.shape):
        raise ValueError("The source must have sides 2..5")
    colors, counts = np.unique(source, return_counts=True)
    maximum = counts.max()
    if np.count_nonzero(counts == maximum) != 1:
        raise ValueError("Background must be uniquely most frequent")
    background = int(colors[int(np.argmax(counts))])
    candidates = [(axis, lines) for axis in ("H", "V")
                  if (lines := _lines(source, background, axis)) is not None]
    if len(candidates) != 1:
        raise ValueError("Endpoint geometry must determine exactly one axis")
    axis, lines = candidates[0]
    return {"source": source, "background": background, "axis": axis,
            "lines": lines, "extent": source.shape[1 if axis == "H" else 0]}


def _coordinate(axis, lane, position):
    return (lane, position) if axis == "H" else (position, lane)


def _layer(source, color, background):
    return np.where(source == color, source, background).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve every source satisfying the family constraints, without latent roles."""
    info = _analyze(input_grid)
    source, background = info["source"], info["background"]
    h, w = source.shape
    output = np.full((h * h, w * w), background, dtype=np.int8)
    layers = {int(c): _layer(source, int(c), background)
              for c in np.unique(source) if int(c) != background}
    for lane, a, z, left, right, middle in info["lines"]:
        for position in range(a, z + 1):
            r, c = _coordinate(info["axis"], lane, position)
            tile = (source if position == middle else
                    layers[left if position < middle else right])
            output[r*h:(r+1)*h, c*w:(c+1)*w] = tile
    return output


def _macro_maps(info):
    """Coarse ownership and whole-picture centers, used by construction/checks."""
    owners = np.full(info["source"].shape, info["background"], dtype=np.int8)
    centers = np.zeros(owners.shape, dtype=bool)
    view = owners if info["axis"] == "H" else owners.T
    cview = centers if info["axis"] == "H" else centers.T
    for lane, a, z, left, right, middle in info["lines"]:
        view[lane, a:middle] = left
        view[lane, middle+1:z+1] = right
        cview[lane, middle] = True
    return owners, centers


def _construct(source):
    # Construction uses Boolean outer products; solve uses direct block copying.
    info = _analyze(source)
    owners, centers = _macro_maps(info)
    background = info["background"]
    h, w = source.shape
    output = np.full((h*h, w*w), background, dtype=np.int8)
    for color in np.unique(source):
        if int(color) != background:
            output[np.kron(owners == color, source == color)] = color
    for r, c in np.argwhere(centers):
        output[r*h:(r+1)*h, c*w:(c+1)*w] = source
    return output


def _key(grid):
    grid = np.ascontiguousarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _checked_integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"Illegal {name}")
    return int(value)


def _sample_source(rng, background, palette, axis, split, index, regime):
    if split == "train":
        extent = 5
        count = 1 if index == 0 else (2 if regime == "density" else
                                    int(rng.integers(2, 4)))
        if index > 1:
            count = int(rng.integers(1, 3 if regime == "density" else 5))
    else:
        extent = int(rng.integers(3, 6 if regime == "density" else 5))
        count = int(rng.integers(3, 5)) if regime == "density" else int(rng.integers(2, 5))
    lane_count = int(rng.integers(max(2, count + 1), 6))
    lanes = sorted(map(int, rng.choice(lane_count, count, replace=False)))
    source = np.full((lane_count, extent), background, dtype=np.int8)
    local_colors = [int(c) for c in rng.permutation(palette)]
    for j, lane in enumerate(lanes):
        if split == "train" and index == 0:
            length = 5
        elif split == "train" and index == 1:
            length = 5 if j == 0 else 3 if j == 1 else int(rng.choice([3, 5]))
        else:
            length = int(rng.choice([3, 5])) if extent == 5 else 3
        start = int(rng.integers(extent - length + 1))
        if index == 1 and split == "train":
            # Three colors, and a layer represented in at least two lanes.
            assignments = ((local_colors[0], local_colors[1]),
                           (local_colors[2], local_colors[0]),
                           (local_colors[1], local_colors[2]))
            left, right = assignments[j]
        elif j == 0:
            left, right = local_colors[:2]
        elif j == 1:
            left, right = local_colors[2], local_colors[0]
        elif rng.random() < 0.6:
            left, right = ((local_colors[1], local_colors[0]) if rng.random() < 0.5
                           else (local_colors[0], local_colors[1]))
        else:
            left, right = map(int, rng.choice(palette, 2, replace=False))
        source[lane, start], source[lane, start+length-1] = left, right
    return source if axis == "H" else source.T.copy()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_integer(seed, "seed", 0)
    num_train = _checked_integer(num_train, "num_train", 2, 10)
    num_test = _checked_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(10)]
    background = colors[0]
    palette = colors[1:1+int(rng.integers(3, 7))]
    regime = str(rng.choice(["density", "compression"]))
    orientation = str(rng.choice(["H", "V", "mixed"], p=[0.3, 0.3, 0.4]))
    first_axis = str(rng.choice(["H", "V"])) if orientation == "mixed" else orientation
    variable_background = bool(rng.random() < 0.3)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if orientation != "mixed":
                axis = first_axis
            elif split == "train" and index < 2:
                axis = first_axis if index == 0 else ("V" if first_axis == "H" else "H")
            else:
                axis = str(rng.choice(["H", "V"]))
            local_background = (int(rng.choice([c for c in range(10) if c not in palette]))
                                if variable_background else background)
            for _ in range(128):
                source = _sample_source(rng, local_background, palette, axis,
                                        split, index, regime)
                try:
                    _analyze(source)
                except ValueError:
                    continue
                k = _key(source)
                if k in seen:
                    continue
                puzzle[split].append({"input": source, "output": _construct(source)})
                seen.add(k)
                break
            else:
                raise RuntimeError("Could not sample a distinct unambiguous source")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair schema")
    info = _analyze(pair["input"])
    source, background = info["source"], info["background"]
    output = _grid(pair["output"])
    h, w = source.shape
    _check(output.shape == (h*h, w*w), "Output dimensions must be squared")
    owners, centers = _macro_maps(info)
    full_count, active_count = 0, 0
    for r in range(h):
        for c in range(w):
            tile = output[r*h:(r+1)*h, c*w:(c+1)*w]
            owner = int(owners[r, c])
            if centers[r, c]:
                expected = source
            elif owner != background:
                expected = _layer(source, owner, background)
            else:
                expected = np.full(source.shape, background, dtype=np.int8)
            _check(np.array_equal(tile, expected), "Incorrect full, layer, or blank block")
            full_count += int(np.array_equal(tile, source))
            active_count += int(np.any(tile != background))
    _check(full_count == len(info["lines"]), "Need exactly one whole copy per pair")
    _check(active_count == sum(z-a+1 for _, a, z, _, _, _ in info["lines"]),
           "Active block count must equal total span lengths")
    for lane, a, z, left, right, middle in info["lines"]:
        _check(middle-a == z-middle, "Territories have unequal lengths")
        for position, color in ((a, left), (z, right)):
            r, c = _coordinate(info["axis"], lane, position)
            _check(int(owners[r, c]) == color, "Endpoint ownership changed")
    return info


def _ablation(source, mode):
    """Counterfactual rules for evidence checks, never used to solve or generate."""
    info = _analyze(source)
    b, axis = info["background"], info["axis"]
    h, w = source.shape
    out = np.full((h*h, w*w), b, dtype=np.int8)
    for lane, a, z, left, right, middle in info["lines"]:
        for position in range(a, z+1):
            r, c = _coordinate(axis, lane, position)
            owner = left if position < middle else right
            if mode == "no_A":
                if position not in (a, z):
                    continue
                tile = _layer(source, left if position == a else right, b)
            elif mode == "no_B":
                tile = source if position == middle else np.full(source.shape, owner, dtype=np.int8)
            elif mode == "full_everywhere":
                tile = source
            elif mode == "local_layers":
                tile = source.copy() if position == middle else np.full(source.shape, b, dtype=np.int8)
                if position != middle:
                    er, ec = _coordinate(axis, lane, a if position < middle else z)
                    tile[er, ec] = owner
            elif mode == "pair_center":
                tile = (np.where((source == left) | (source == right), source, b)
                        if position == middle else _layer(source, owner, b))
            else:
                raise ValueError("Unknown ablation")
            out[r*h:(r+1)*h, c*w:(c+1)*w] = tile
    return out


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    n, t = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= n <= 10 and 1 <= t <= 4 and n > t, "Illegal pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            k = _key(info["source"])
            _check(k not in seen, "Repeated input")
            seen.add(k)
            infos[split].append(info)
    first, second = infos["train"][:2]
    _check(len(first["lines"]) == 1 and first["extent"] == 5
           and first["lines"][0][2] - first["lines"][0][1] == 4,
           "First demonstration must expose actual propagation")
    _check(len(second["lines"]) >= 2 and second["extent"] == 5,
           "Second demonstration must expose multiple lanes")
    _check({z-a for _, a, z, _, _, _ in second["lines"]} == {2, 4},
           "Second demonstration needs both span scales")
    second_colors = [int(c) for c in second["source"].flat if int(c) != second["background"]]
    _check(len(set(second_colors)) >= 3 and len(second_colors) > len(set(second_colors)),
           "Need a shared global color and a third color")
    train_axes = {i["axis"] for i in infos["train"]}
    _check({i["axis"] for i in infos["test"]} <= train_axes, "Undemonstrated axis")
    if len(train_axes) == 2:
        _check(first["axis"] != second["axis"], "Both axes need introductory evidence")
    _check(all(i["extent"] == 5 for i in infos["train"]), "Training extent drift")
    density_shift = (all(len(i["lines"]) <= 2 for i in infos["train"])
                     and all(len(i["lines"]) >= 3 for i in infos["test"]))
    compression_shift = all(i["extent"] in (3, 4) for i in infos["test"])
    _check(density_shift or compression_shift, "Missing structural train/test shift")
    for mode in ("no_A", "no_B", "full_everywhere", "local_layers", "pair_center"):
        _check(not np.array_equal(_ablation(second["source"], mode),
                                  np.asarray(puzzle["train"][1]["output"])),
               f"Second demonstration does not distinguish {mode}")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded rule, robustness, count, and 42-seed diversity checks."""
    fixture = np.array([[2, 0, 0, 0, 3], [0, 0, 4, 0, 2], [0, 0, 0, 0, 0]], dtype=np.int8)
    macro = ((2, 2, -1, 3, 3), (0, 0, 4, -1, 2), (0, 0, 0, 0, 0))
    expected = np.zeros((9, 25), dtype=np.int8)
    for r in range(3):
        for c in range(5):
            for u in range(3):
                for v in range(5):
                    if macro[r][c] == -1 or fixture[u, v] == macro[r][c]:
                        expected[3*r+u, 5*c+v] = fixture[u, v]
    _check(np.array_equal(solve(fixture), expected), "Hand-written fixture failed")
    _check(np.array_equal(solve(fixture.T), expected.T), "Transpose equivariance failed")
    permutation = np.array([7, 9, 0, 5, 2, 8, 3, 6, 4, 1], dtype=np.int8)
    _check(np.array_equal(solve(permutation[fixture]), permutation[expected]),
           "Palette equivariance failed")
    # An even-width, short-span source outside the introductory sampler layout.
    short = np.array([[0, 2, 0, 3], [0, 0, 0, 0]], dtype=np.int8)
    _check(np.array_equal(solve(short), _construct(short)), "General short-span member failed")
    ambiguous = np.array([[2, 0, 3], [0, 0, 0], [4, 0, 5]], dtype=np.int8)
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous axis accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(704+a*10+b, a, b)
            _check((len(p["train"]), len(p["test"])) == (a, b), "Counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    for seed in (-1, True, 0.5, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal seed accepted")

    shapes, structures, signatures, backgrounds, endpoint_colors = set(), set(), set(), set(), set()
    axes, counts, lengths, extents, regimes = set(), set(), set(), set(), set()
    features = {name: 0 for name in ("reused", "internal", "asymmetric", "adjacent", "edge", "mixed")}
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = _puzzle_key(p)
        _check(signature == _puzzle_key(generate_puzzle(seed)), "Nondeterministic seed")
        signatures.add(signature)
        paxes = set()
        regimes.add("compression" if all(_analyze(x["input"])["extent"] < 5
                                         for x in p["test"]) else "density")
        for split in ("train", "test"):
            for pair in p[split]:
                info = _analyze(pair["input"])
                source, b, axis = info["source"], info["background"], info["axis"]
                _check(np.array_equal(solve(source), pair["output"]), "Solver disagreement")
                _check(np.array_equal(solve(source.T), np.asarray(pair["output"]).T),
                       "Generated transpose equivariance failed")
                shapes.add(source.shape)
                backgrounds.add(b)
                endpoint_colors.update(int(c) for c in np.unique(source) if int(c) != b)
                axes.add(axis)
                paxes.add(axis)
                counts.add(len(info["lines"]))
                extents.add(info["extent"])
                roles = {b: 0}
                structures.add((source.shape, tuple(roles.setdefault(int(c), len(roles))
                                                    for c in source.flat)))
                nonbackground = source[source != b]
                features["reused"] += int(len(nonbackground) > len(np.unique(nonbackground)))
                lane_ids = [x[0] for x in info["lines"]]
                features["adjacent"] += int(any(y-x == 1 for x, y in zip(lane_ids, lane_ids[1:])))
                lanes = source.shape[0 if axis == "H" else 1]
                features["edge"] += int(0 in lane_ids or lanes-1 in lane_ids)
                for _, a, z, _, _, _ in info["lines"]:
                    lengths.add(z-a+1)
                    features["internal"] += int(a > 0 or z < info["extent"]-1)
                    features["asymmetric"] += int(a != info["extent"]-1-z)
        features["mixed"] += int(len(paxes) == 2)
    _check(len(signatures) == 42 and len(structures) >= 70, "Insufficient structural diversity")
    _check(len(shapes) >= 10 and axes == {"H", "V"}, "Insufficient geometry diversity")
    _check(counts == {1, 2, 3, 4} and lengths == {3, 5} and extents == {3, 4, 5},
           "Missing count/scale variation")
    _check(len(backgrounds) >= 8 and len(endpoint_colors) == 10, "Insufficient palette variation")
    _check(regimes == {"density", "compression"} and all(features.values()),
           "Missing structural feature or shift")
    p = generate_puzzle(0)
    as_lists = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in p[s]]
                for s in ("train", "test")}
    validate_generated_puzzle(as_lists)
    # Deliberately alter a half tile, a center, and a blank block.
    info = _analyze(p["train"][1]["input"])
    owners, centers = _macro_maps(info)
    b = info["background"]
    coordinates = [tuple(np.argwhere(owners != b)[0]), tuple(np.argwhere(centers)[0]),
                   tuple(np.argwhere((owners == b) & ~centers)[0])]
    h, w = info["source"].shape
    for r, c in coordinates:
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]]
               for s in ("train", "test")}
        bad["train"][1]["output"][r*h, c*w] = (int(bad["train"][1]["output"][r*h, c*w])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus explicit-count and rule checks)")
