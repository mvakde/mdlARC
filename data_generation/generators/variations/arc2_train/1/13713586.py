"""Boundary-guided projection of parallel one-cell bars."""

import hashlib
import numpy as np


_EDGES = ("top", "bottom", "left", "right")
_SOURCE_PAIR_HASHES = {
    "8b9a79ad4dda663b8b860ecb5ff6df1188353f00cfb558c64d5e5fcbb36046f5",
    "fc2b9487a88cc2e9b2683b2c9e2aa2223233f7e357acb47fbbab8b151a3c73d4",
    "470442e49920ae8e5988e5600c885a4cd95e0259aa093849825a6265d051b5db",
    "5a27e46ede957d6423939ecc52fdb02bf880f71c1f2cbd4d7afd42325467db34",
}
_SOURCE_REGRESSIONS = (
    (
        "00000000000000005/00000000000200005/00000000000200005/"
        "00000000000000005/00000030000000005/00000030000000005/"
        "00000030000000005/00700030000000005/00700000000000005/"
        "00700000000000005/00700000000000005/00000000000000005/"
        "00000000000000005/00000000000000005/00000000000000005/"
        "00000000000000005",
        "ae13b510ee17dce3756e17df055f52d1f497bef3e3b295222665d5efdf392323",
    ),
    (
        "000000000000000/000000000000000/000000000000000/000000444440000/"
        "000000000000000/000000000000000/888800000000000/000000000000000/"
        "000000000000000/000000000000000/000000000000000/000000033300000/"
        "000000000000000/000000000000000/000000000000000/555555555555555",
        "870866d92758822ce5ee043e809e60a58e2520c6e0982cb72131e25054c485a2",
    ),
    (
        "50000000000000000/50000000030000000/50000000030000000/"
        "50000000030000000/50000000030000000/50000000030006000/"
        "50000000030006000/50000000000006000/50000000000006000/"
        "50000000000006000/50000020000006000/50000020000006000/"
        "50000000000006000/50000000000000000/50000000000000000/"
        "50000000000000000/50000000000000000",
        "0982fbd3222245fe472896663bd27686b7c68c266dc7a033e491668793900eb3",
    ),
    (
        "555555555555555555/000000000000000000/000000000000000000/"
        "000000000000000000/000000000000000000/000000000000008880/"
        "000000000000000000/444400000000000000/000000000000000000/"
        "000000000000000000/000000000000000000/000000002222200000/"
        "000000000000000000/000000000000000000/000000000000000000/"
        "003333333330000000/000000000000000000/000000000000000000/"
        "000000000000000000",
        "3a060529af0d0bfc3598a56d016ffa3e065fdad326a4da3fdea6629ccdfd1207",
    ),
)


def _signature(grid):
    return (tuple(grid.shape), grid.tobytes())


def _pair_hash(pair):
    a = pair["input"].astype(np.uint8, copy=False)
    b = pair["output"].astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _structural_signature(grid):
    info = _parse_input(grid)
    bars = tuple(sorted((bar["distance"], bar["start"], bar["end"])
                        for bar in info["bars"]))
    return (grid.shape, info["edge"], bars)


def _resolve_counts(rng, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer or None")
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")
    if num_test is None and num_train is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _to_top(grid, edge):
    if edge == "top":
        return grid
    if edge == "bottom":
        return grid[::-1, :]
    if edge == "left":
        return grid.T
    return grid.T[::-1, :]


def _from_top(grid, edge):
    if edge == "top":
        return grid
    if edge == "bottom":
        return grid[::-1, :]
    if edge == "left":
        return grid.T
    return grid[::-1, :].T


def _parse_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) >= 3
    assert np.count_nonzero(counts == counts.max()) == 1
    background = int(colors[np.argmax(counts)])
    h, w = grid.shape
    sides = (("top", grid[0, :]), ("bottom", grid[-1, :]),
             ("left", grid[:, 0]), ("right", grid[:, -1]))
    candidates = [(edge, int(line[0])) for edge, line in sides
                  if int(line[0]) != background and np.all(line == line[0])]
    assert len(candidates) == 1
    edge, boundary = candidates[0]
    top = _to_top(grid, edge)
    p, lateral = top.shape
    assert p >= 4 and lateral >= 2
    assert np.all(top[0] == boundary)
    assert np.count_nonzero(top == boundary) == lateral
    assert np.all(top[-1] == background)
    bars = []
    for distance in range(1, p - 1):
        row = top[distance]
        x = 0
        while x < lateral:
            if row[x] == background:
                x += 1
                continue
            color = int(row[x])
            assert color != boundary
            start = x
            while x + 1 < lateral and int(row[x + 1]) == color:
                x += 1
            end = x
            bars.append({"color": color, "distance": distance,
                         "start": start, "end": end})
            x += 1
            assert x == lateral or int(row[x]) == background
    assert bars
    for i, first in enumerate(bars):
        for second in bars[i + 1:]:
            overlap = max(first["start"], second["start"]) <= min(first["end"], second["end"])
            if first["distance"] == second["distance"]:
                assert not overlap
            if first["color"] == second["color"] and abs(first["distance"] - second["distance"]) == 1:
                assert not overlap
    return {"background": background, "boundary": boundary,
            "edge": edge, "bars": bars}


def _transform_grid(input_grid):
    info = _parse_input(input_grid)
    top = _to_top(input_grid, info["edge"]).copy()
    for bar in sorted(info["bars"], key=lambda b: b["distance"], reverse=True):
        top[1:bar["distance"] + 1, bar["start"]:bar["end"] + 1] = bar["color"]
    return _from_top(top, info["edge"]).copy()


def _ray_scan(input_grid):
    """Independent cell-wise solution, used to check the painted construction."""
    info = _parse_input(input_grid)
    top = _to_top(input_grid, info["edge"])
    result = top.copy()
    for row in range(1, top.shape[0] - 1):
        for col in range(top.shape[1]):
            rays = [bar for bar in info["bars"] if bar["distance"] >= row
                    and bar["start"] <= col <= bar["end"]]
            if rays:
                result[row, col] = min(rays, key=lambda bar: bar["distance"])["color"]
    return _from_top(result, info["edge"]).copy()


def _overlap(first, second):
    return max(first[0], second[0]) <= min(first[1], second[1])


def _sample_length(rng, lateral):
    band = int(rng.integers(0, 5))
    if band == 0:
        return 1
    if band == 1:
        return int(rng.integers(2, min(lateral, 4) + 1))
    if band == 2:
        return int(rng.integers(1, lateral + 1))
    if band == 3:
        return int(rng.integers(max(1, lateral // 2), lateral + 1))
    return int(rng.integers(max(1, lateral - 2), lateral + 1))


def _random_span(rng, lateral, region=None):
    lo, hi = (0, lateral - 1) if region is None else region
    size = hi - lo + 1
    length = min(size, _sample_length(rng, lateral))
    start = int(rng.integers(lo, hi - length + 2))
    return (start, start + length - 1)


def _anchor_span(rng, lo, hi, anchor):
    size = hi - lo + 1
    length = min(size, _sample_length(rng, size))
    lower = max(lo, anchor - length + 1)
    upper = min(anchor, hi - length + 1)
    start = int(rng.integers(lower, upper + 1))
    return (start, start + length - 1)


def _disjoint_spans(rng, lateral, count):
    # One cell of background separates neighboring bars.
    extra = lateral - (2 * count - 1)
    weights = rng.multinomial(extra, np.full(2 * count + 1, 1 / (2 * count + 1)))
    lengths = [1 + int(weights[2 * i + 1]) for i in range(count)]
    gaps = [1 + int(weights[2 * i + 2]) for i in range(count - 1)]
    pos = int(weights[0])
    spans = []
    for i, length in enumerate(lengths):
        spans.append((pos, pos + length - 1))
        pos += length + (gaps[i] if i < count - 1 else 0)
    if rng.random() < 0.5:
        spans.reverse()
    return spans


def _sample_spans(rng, lateral, count, evidence, regime):
    modes = ["dense", "separated", "chain", "mixed"]
    if not evidence and lateral >= 2 * count - 1:
        modes.append("disjoint")
    mode = modes[int(rng.integers(0, len(modes)))]
    if mode == "disjoint":
        spans = _disjoint_spans(rng, lateral, count)
    elif mode == "chain" and count >= 3 and lateral >= 2 * count + 1:
        step = int(rng.integers(2, max(3, (lateral - 1) // (count - 1))))
        while (count - 1) * step + 3 > lateral:
            step -= 1
        half = max(1, (step + 1) // 2)
        width = (count - 1) * step + 2 * half + 1
        if width > lateral:
            half = 1
            width = (count - 1) * step + 3
        offset = int(rng.integers(0, lateral - width + 1))
        spans = [(offset + i * step, offset + i * step + 2 * half)
                 for i in range(count)]
    elif mode == "separated" and count >= 3 and lateral >= 6:
        split = int(rng.integers(2, lateral - 2))
        regions = ((0, split - 1), (split + 1, lateral - 1))
        if rng.random() < 0.5:
            regions = regions[::-1]
        anchor = int(rng.integers(regions[0][0], regions[0][1] + 1))
        spans = [_anchor_span(rng, *regions[0], anchor) for _ in range(2)]
        spans += [_random_span(rng, lateral, regions[1]) for _ in range(count - 2)]
    elif mode == "dense":
        anchor = int(rng.integers(0, lateral))
        spans = [_anchor_span(rng, 0, lateral - 1, anchor) for _ in range(count)]
    else:
        spans = [_random_span(rng, lateral) for _ in range(count)]
        if evidence or (count >= 2 and rng.random() < 0.45):
            anchor = int(rng.integers(spans[0][0], spans[0][1] + 1))
            spans[1] = _anchor_span(rng, 0, lateral - 1, anchor)
    if evidence and not _overlap(spans[0], spans[1]):
        anchor = int(rng.integers(spans[0][0], spans[0][1] + 1))
        spans[1] = _anchor_span(rng, 0, lateral - 1, anchor)
    if count >= (3 if evidence else 2) and rng.random() < 0.13 and mode != "disjoint":
        j = int(rng.integers(2 if evidence else 0, count))
        i = int(rng.integers(0, count - 1))
        if i >= j:
            i += 1
        spans[j] = spans[i]
    return spans


def _sample_distances(rng, perpendicular, spans, evidence):
    count = len(spans)
    max_distance = perpendicular - 2
    # Selecting compressed positions gives distinct rows with one blank row between.
    base = np.sort(rng.choice(np.arange(1, max_distance - count + 2),
                              size=count, replace=False))
    distances = [int(value + i) for i, value in enumerate(base)]
    if rng.random() < 0.22:
        shift = distances[0] - 1
        distances = [d - shift for d in distances]
    elif rng.random() < 0.18:
        shift = max_distance - distances[-1]
        distances = [d + shift for d in distances]
    if evidence:
        # The core bars are the nearest two, so their contested pixels are visible.
        ordered = distances[:]
        distances = [ordered[0], ordered[1]] + ordered[2:]
    else:
        rng.shuffle(distances)
        candidates = [(i, j) for i in range(count) for j in range(i + 1, count)
                      if spans[i][1] + 1 < spans[j][0] or spans[j][1] + 1 < spans[i][0]]
        if candidates and rng.random() < 0.22:
            i, j = candidates[int(rng.integers(0, len(candidates)))]
            distances[j] = distances[i]
    if max(distances) == 1:
        distances[0] = 2
    return distances


def _sample_dimension(rng, regime):
    if regime == "source_train":
        weights = (0.20, 0.43, 0.29, 0.08)
    else:
        weights = (0.15, 0.31, 0.39, 0.15)
    band = int(rng.choice(4, p=weights))
    ranges = ((6, 11), (11, 21), (21, 30), (30, 31))
    lo, hi = ranges[band]
    return int(rng.integers(lo, hi))


def _make_pair(rng, regime, edge, background, boundary, core, evidence_index):
    h = _sample_dimension(rng, regime)
    w = _sample_dimension(rng, regime)
    perpendicular, lateral = (h, w) if edge in ("top", "bottom") else (w, h)
    limit = min(8, (perpendicular - 2) // 2)
    minimum = 2 if evidence_index is not None else 1
    assert limit >= minimum
    if regime == "target_test" and limit >= 3:
        roll = rng.random()
        if roll < 0.10:
            count = 1
        elif roll < 0.22:
            count = 2
        else:
            count = int(rng.integers(3, limit + 1))
    else:
        count = int(rng.integers(minimum, limit + 1))
    spans = _sample_spans(rng, lateral, count, evidence_index is not None, regime)
    distances = _sample_distances(rng, perpendicular, spans, evidence_index is not None)
    if evidence_index == 1:
        distances[0], distances[1] = distances[1], distances[0]
    available = [c for c in range(10) if c not in (background, boundary)]
    colors = [core[0], core[1]] if evidence_index is not None else []
    while len(colors) < count:
        index = len(colors)
        overlapping = [colors[i] for i in range(index) if _overlap(spans[index], spans[i])]
        separated = [colors[i] for i in range(index) if not _overlap(spans[index], spans[i])]
        roll = rng.random()
        if separated and roll < 0.24:
            colors.append(int(rng.choice(separated)))
        elif overlapping and roll < 0.43:
            colors.append(int(rng.choice(overlapping)))
        else:
            colors.append(int(rng.choice(available)))
    top = np.full((perpendicular, lateral), background, dtype=np.int8)
    top[0, :] = boundary
    for (start, end), distance, color in zip(spans, distances, colors):
        assert np.all(top[distance, start:end + 1] == background)
        top[distance, start:end + 1] = color
    input_grid = _from_top(top, edge).copy()
    output_grid = _transform_grid(input_grid)
    assert not np.array_equal(input_grid, output_grid)
    return {"input": input_grid, "output": output_grid}


def _orientation_plan(rng, train_count, test_count):
    mode = int(rng.integers(0, 4))
    if mode == 0:
        held = _EDGES[int(rng.integers(0, 4))]
        pool = [edge for edge in _EDGES if edge != held]
        train = [pool[i % 3] for i in range(train_count)]
        rng.shuffle(train)
        test = [held] * test_count
    elif mode == 1:
        shared = _EDGES[int(rng.integers(0, 4))]
        train = [shared] + [_EDGES[int(rng.integers(0, 4))] for _ in range(train_count - 1)]
        test = [shared] + [_EDGES[int(rng.integers(0, 4))] for _ in range(test_count - 1)]
    elif mode == 2:
        train = [_EDGES[i % 4] for i in range(train_count)]
        rng.shuffle(train)
        test = [_EDGES[i % 4] for i in range(test_count)]
        rng.shuffle(test)
    else:
        uniform = _EDGES[int(rng.integers(0, 4))]
        train = [uniform] * train_count
        test = [uniform] * test_count
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    core = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
    shared_palette = rng.random() < 0.36
    if shared_palette:
        roles = [c for c in range(10) if c not in core]
        common = [int(x) for x in rng.choice(roles, size=2, replace=False)]
    train_edges, test_edges = _orientation_plan(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    used = set()
    used_structures = set()
    for part, edges in (("train", train_edges), ("test", test_edges)):
        for index, edge in enumerate(edges):
            evidence = index if part == "train" and index < 2 else None
            for _ in range(32):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                if shared_palette:
                    background, boundary = common
                else:
                    role_pool = [c for c in range(10) if evidence is None or c not in core]
                    background, boundary = (int(x) for x in pair_rng.choice(role_pool, size=2, replace=False))
                pair = _make_pair(pair_rng, "source_train" if part == "train" else "target_test",
                                  edge, background, boundary, core, evidence)
                signature = _signature(pair["input"])
                structure = _structural_signature(pair["input"])
                if (signature not in used and structure not in used_structures
                        and _pair_hash(pair) not in _SOURCE_PAIR_HASHES):
                    used.add(signature)
                    used_structures.add(structure)
                    puzzle[part].append(pair)
                    break
            else:
                raise AssertionError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _reversed_precedence(first, second):
    for a in first["bars"]:
        for b in first["bars"]:
            if a["color"] == b["color"] or a["distance"] == b["distance"]:
                continue
            if not _overlap((a["start"], a["end"]), (b["start"], b["end"])):
                continue
            for c in second["bars"]:
                for d in second["bars"]:
                    if c["color"] == a["color"] and d["color"] == b["color"]:
                        if _overlap((c["start"], c["end"]), (d["start"], d["end"])):
                            if (a["distance"] < b["distance"]) != (c["distance"] < d["distance"]):
                                return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    seen_structures = set()
    parsed = {"train": [], "test": []}
    for part in ("train", "test"):
        for pair in puzzle[part]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert set(np.unique(inp)) == set(np.unique(out))
            input_sig = _signature(inp)
            pair_sig = (input_sig, _signature(out))
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            structure = _structural_signature(inp)
            assert structure not in seen_structures
            seen_structures.add(structure)
            assert _pair_hash(pair) not in _SOURCE_PAIR_HASHES
            info = _parse_input(inp)
            assert np.array_equal(out, _ray_scan(inp))
            assert np.array_equal(out, _transform_grid(inp))
            perpendicular = inp.shape[0] if info["edge"] in ("top", "bottom") else inp.shape[1]
            assert len(info["bars"]) <= min(8, (perpendicular - 2) // 2)
            parsed[part].append(info)
    assert _reversed_precedence(parsed["train"][0], parsed["train"][1])
    # Semantic colors are geometric roles, so palettes may change from pair to pair.
    for part in ("train", "test"):
        for info in parsed[part]:
            assert info["background"] != info["boundary"]
            assert all(bar["color"] not in (info["background"], info["boundary"])
                       for bar in info["bars"])


def _puzzle_signature(puzzle):
    return tuple((_signature(pair["input"]), _signature(pair["output"]))
                 for part in ("train", "test") for pair in puzzle[part])


def self_test() -> None:
    for encoded, output_hash in _SOURCE_REGRESSIONS:
        rows = encoded.split("/")
        source_input = np.array([[int(cell) for cell in row] for row in rows], dtype=np.int8)
        answer = _ray_scan(source_input)
        assert np.array_equal(answer, _transform_grid(source_input))
        digest = hashlib.sha256(bytes(answer.shape) + answer.astype(np.uint8).tobytes()).hexdigest()
        assert digest == output_hash
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, layouts, edges, counts, backgrounds, boundaries = set(), set(), set(), set(), set(), set()
    split_counts = {"train": set(), "test": set()}
    split_colors = {"train": set(), "test": set()}
    split_distances = {"train": set(), "test": set()}
    modes = {"side_touch": 0, "singleton": 0, "adjacent": 0, "same_color": 0,
             "same_distance": 0, "disjoint": 0, "partial": 0, "identical_span": 0,
             "full_span": 0, "near_full": 0, "rectangle": 0, "axis30": 0,
             "chain_without_common": 0, "isolated_with_overlap": 0,
             "same_color_overlap": 0, "same_color_separated": 0,
             "side_touch_and_partial": 0}
    puzzle_modes = {"held_out": 0, "shared_edge": 0, "mixed_both": 0,
                    "uniform_edge": 0, "shared_roles": 0,
                    "unseen_test_color": 0}
    varying_roles = 0
    mixed_counts = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        info_list = []
        for part in ("train", "test"):
            for pair in puzzle[part]:
                inp = pair["input"]
                info = _parse_input(inp)
                info_list.append(info)
                shapes.add(inp.shape)
                edges.add(info["edge"])
                backgrounds.add(info["background"])
                boundaries.add(info["boundary"])
                counts.add(len(info["bars"]))
                split_counts[part].add(len(info["bars"]))
                split_colors[part].add(len({b["color"] for b in info["bars"]}))
                split_distances[part].add(len({b["distance"] for b in info["bars"]}))
                geometry = tuple(sorted((b["distance"], b["start"], b["end"]) for b in info["bars"]))
                layouts.add((inp.shape, info["edge"], geometry))
                lateral = inp.shape[1] if info["edge"] in ("top", "bottom") else inp.shape[0]
                modes["side_touch"] += any(b["start"] == 0 or b["end"] == lateral - 1 for b in info["bars"])
                modes["full_span"] += any(b["start"] == 0 and b["end"] == lateral - 1 for b in info["bars"])
                modes["near_full"] += any(b["end"] - b["start"] + 1 >= lateral - 2 for b in info["bars"])
                modes["rectangle"] += max(inp.shape) >= 2 * min(inp.shape)
                modes["axis30"] += 30 in inp.shape
                modes["singleton"] += any(b["start"] == b["end"] for b in info["bars"])
                modes["adjacent"] += any(b["distance"] == 1 for b in info["bars"])
                bars = info["bars"]
                overlap_count = sum(_overlap((a["start"], a["end"]), (b["start"], b["end"]))
                                    for i, a in enumerate(bars) for b in bars[i + 1:])
                modes["disjoint"] += len(bars) >= 2 and overlap_count == 0
                modes["partial"] += 0 < overlap_count < len(bars) * (len(bars) - 1) // 2
                modes["same_color"] += any(a["color"] == b["color"] for i, a in enumerate(bars) for b in bars[i + 1:])
                modes["same_color_overlap"] += any(
                    a["color"] == b["color"] and _overlap((a["start"], a["end"]), (b["start"], b["end"]))
                    for i, a in enumerate(bars) for b in bars[i + 1:])
                modes["same_color_separated"] += any(
                    a["color"] == b["color"] and not _overlap((a["start"], a["end"]), (b["start"], b["end"]))
                    for i, a in enumerate(bars) for b in bars[i + 1:])
                modes["same_distance"] += any(a["distance"] == b["distance"] for i, a in enumerate(bars) for b in bars[i + 1:])
                modes["identical_span"] += any((a["start"], a["end"]) == (b["start"], b["end"])
                                               for i, a in enumerate(bars) for b in bars[i + 1:])
                graph = [set() for _ in bars]
                for i, a in enumerate(bars):
                    for j in range(i + 1, len(bars)):
                        if _overlap((a["start"], a["end"]), (bars[j]["start"], bars[j]["end"])):
                            graph[i].add(j)
                            graph[j].add(i)
                common = max(b["start"] for b in bars) <= min(b["end"] for b in bars)
                modes["chain_without_common"] += len(bars) >= 3 and overlap_count >= 2 and not common and all(graph)
                modes["isolated_with_overlap"] += overlap_count > 0 and any(not neighbors for neighbors in graph)
                modes["side_touch_and_partial"] += (
                    any(b["start"] == 0 or b["end"] == lateral - 1 for b in bars) and
                    0 < overlap_count < len(bars) * (len(bars) - 1) // 2)
        varying_roles += len({(x["background"], x["boundary"]) for x in info_list}) > 1
        mixed_counts += len({len(x["bars"]) for x in info_list}) > 1
        train_info = [_parse_input(pair["input"]) for pair in puzzle["train"]]
        test_info = [_parse_input(pair["input"]) for pair in puzzle["test"]]
        train_edges = {x["edge"] for x in train_info}
        test_edges = {x["edge"] for x in test_info}
        puzzle_modes["held_out"] += train_edges.isdisjoint(test_edges) and len(test_edges) == 1
        puzzle_modes["shared_edge"] += bool(train_edges & test_edges)
        puzzle_modes["mixed_both"] += len(train_edges) > 1 and len(test_edges) > 1
        puzzle_modes["uniform_edge"] += len(train_edges | test_edges) == 1
        puzzle_modes["shared_roles"] += len({(x["background"], x["boundary"]) for x in info_list}) == 1
        train_colors = {b["color"] for x in train_info for b in x["bars"]}
        puzzle_modes["unseen_test_color"] += any(b["color"] not in train_colors for x in test_info for b in x["bars"])
    assert edges == set(_EDGES) and len(shapes) >= 75 and len(layouts) >= 250
    assert len(counts) >= 6 and len(backgrounds) == 10 and len(boundaries) == 10
    assert varying_roles >= 15 and mixed_counts >= 25
    assert all(value >= 3 for value in modes.values()), modes
    assert all(len(split_counts[part]) >= 6 for part in ("train", "test"))
    assert all(len(split_colors[part]) >= 5 for part in ("train", "test"))
    assert all(len(split_distances[part]) >= 6 for part in ("train", "test"))
    assert all(value >= 3 for value in puzzle_modes.values()), puzzle_modes
    assert _puzzle_signature(generate_puzzle(7319, 7, 3)) == _puzzle_signature(generate_puzzle(7319, 7, 3))
    assert _puzzle_signature(generate_puzzle(91)) == _puzzle_signature(generate_puzzle(91))
    assert _puzzle_signature(generate_puzzle(101)) != _puzzle_signature(generate_puzzle(102))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(8000 + i, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(9000 + i, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(123, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {kwargs}")
