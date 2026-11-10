"""Compositional generators for the detached-key and rectangular-panel task."""

import hashlib

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_AXES = {
    "both": _DIRS,
    "vertical": _DIRS[:2],
    "horizontal": _DIRS[2:],
}
_SOURCE_DIGESTS = {
    "6a45829c01da69f5ae458ddd40113206ce9ef2de151a618c9ba289a97343b486",
    "fdde6cad7855b914b8b6d2d570927c1796fbd11c6c5fbc821106a4b8e63d6524",
    "98cbe0fc02b54587d004251b99085e1ca7219d06bb37cebd52de8a129729f437",
    "e14fe6a7580491b9b65c4bf3858cc05b685c8acbf3b7c5082bea13641bbd4d0c",
}


def _checked_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = None if num_train is None else _checked_count(num_train, "num_train", 2, 10)
    test = None if num_test is None else _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            component = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr, dc in _DIRS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            found.append(component)
    return found


def _guide_axes(motif, center, seed, fill, background):
    """Find opposing two-cell arms around a possible key anchor."""
    r, c = center
    observed = []
    for name, vectors in (("vertical", _AXES["vertical"]),
                          ("horizontal", _AXES["horizontal"])):
        values = [motif.get((r + n * dr, c + n * dc))
                  for dr, dc in vectors for n in (1, 2)]
        if values[0] is not None and len(set(values)) == 1:
            value = values[0]
            if value not in (seed, fill, background):
                observed.append((name, value))
    if not observed or len({value for name, value in observed}) != 1:
        return (), None
    return tuple(name for name, value in observed), observed[0][1]


def _inspect_input(grid):
    height, width = grid.shape
    background = int(grid[0, 0])
    assert all(int(grid[r, c]) == background for r, c in
               ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)))
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    colors = [c for c in range(10) if c != background and counts[c]]
    assert len(colors) >= 3
    fill = max(colors, key=lambda c: int(counts[c]))
    fill_components = _components(grid == fill)
    assert 1 <= len(fill_components) <= 4
    boxes = []
    inside = np.zeros(grid.shape, dtype=bool)
    for component in fill_components:
        rows, cols = zip(*component)
        box = (min(rows), max(rows) + 1, min(cols), max(cols) + 1)
        r0, r1, c0, c1 = box
        assert r1 - r0 >= 5 and c1 - c0 >= 5
        assert not inside[r0:r1, c0:c1].any()
        inside[r0:r1, c0:c1] = True
        assert np.all(grid[r0:r1, c0:c1] != background)
        boxes.append(box)
    boxes.sort()
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        for s0, s1, d0, d1 in boxes[i + 1:]:
            assert r1 < s0 or s1 < r0 or c1 < d0 or d1 < c0
    panel_seeds = []
    seed_colors = set()
    for r0, r1, c0, c1 in boxes:
        sub = grid[r0:r1, c0:c1]
        anomalies = set(int(x) for x in np.unique(sub) if x != fill)
        assert len(anomalies) == 1
        seed_colors.update(anomalies)
        seed_coords = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(sub != fill)]
        assert seed_coords
        panel_seeds.append(seed_coords)
    assert len(seed_colors) == 1
    seed = next(iter(seed_colors))
    assert seed not in (background, fill)
    key_mask = (grid != background) & ~inside
    key_components = _components(key_mask)
    assert len(key_components) == 1
    key_coords = key_components[0]
    key_cells = {(r, c): int(grid[r, c]) for r, c in key_coords}
    for r, c in key_coords:
        for dr, dc in _DIRS:
            rr, cc = r + dr, c + dc
            assert not (0 <= rr < height and 0 <= cc < width and inside[rr, cc])
    candidates = []
    for point, color in key_cells.items():
        if color != seed:
            continue
        axes, guide = _guide_axes(key_cells, point, seed, fill, background)
        if axes:
            candidates.append((point, axes, guide))
    assert len(candidates) == 1
    center, axes, guide = candidates[0]
    assert guide is not None
    directions = tuple(vector for axis in axes for vector in _AXES[axis])
    motif = {(r - center[0], c - center[1]): color for (r, c), color in key_cells.items()}
    assert motif[(0, 0)] == seed
    assert all(abs(dr) <= 2 and abs(dc) <= 2 for dr, dc in motif)
    assert all(color not in (background, fill) for color in motif.values())
    arm_offsets = {(n * dr, n * dc) for dr, dc in directions for n in (1, 2)}
    assert all(motif[offset] == guide for offset in arm_offsets)
    assert all(color != guide for offset, color in motif.items() if offset not in arm_offsets)
    return {
        "background": background, "fill": fill, "seed": seed, "guide": guide,
        "boxes": boxes, "panel_seeds": panel_seeds, "key_coords": key_coords,
        "center": center, "motif": motif, "directions": directions,
        "axes": "both" if len(axes) == 2 else axes[0],
        "arm_offsets": arm_offsets,
    }


def _box_assignments(box, seeds, motif, directions, guide):
    r0, r1, c0, c1 = box
    arm_offsets = {(n * dr, n * dc) for dr, dc in directions for n in (1, 2)}
    writes = {}

    def put(point, color):
        old = writes.get(point)
        if old is not None and old != color:
            raise AssertionError("conflicting stamps or ray through a seed")
        writes[point] = color

    for seed_r, seed_c in seeds:
        put((seed_r, seed_c), motif[(0, 0)])
    for seed_r, seed_c in seeds:
        for (dr, dc), color in motif.items():
            if (dr, dc) in arm_offsets:
                continue
            row, col = seed_r + dr, seed_c + dc
            if not (r0 <= row < r1 and c0 <= col < c1):
                raise AssertionError("decoration must fit within its panel")
            put((row, col), color)
        for dr, dc in directions:
            row, col = seed_r + dr, seed_c + dc
            while r0 <= row < r1 and c0 <= col < c1:
                put((row, col), guide)
                row += dr
                col += dc
    return writes


def _expected_from_inspection(grid, facts):
    output = grid.copy()
    for point in facts["key_coords"]:
        output[point] = facts["background"]
    for box, seeds in zip(facts["boxes"], facts["panel_seeds"]):
        for point, color in _box_assignments(
            box, seeds, facts["motif"], facts["directions"], facts["guide"]
        ).items():
            output[point] = color
    return output


def _inspect_pair(pair):
    facts = _inspect_input(pair["input"])
    expected = _expected_from_inspection(pair["input"], facts)
    assert np.array_equal(pair["output"], expected)
    assert not np.array_equal(pair["input"], pair["output"])
    return facts


def _sample_roles(rng, previous):
    if previous is not None and rng.random() < 0.22:
        return previous
    return tuple(int(x) for x in rng.choice(10, size=4, replace=False))


def _decoration_pool(orientation):
    if orientation == "vertical":
        return [(r, c) for r in range(-2, 3) for c in (-1, 1)]
    if orientation == "horizontal":
        return [(r, c) for r in (-1, 1) for c in range(-2, 3)]
    return [(r, c) for r in range(-2, 3) for c in range(-2, 3)
            if r != 0 and c != 0 and (abs(r) == 1 or abs(c) == 1)]


def _sample_motif(rng, orientation, roles, decor_mode, richness):
    background, fill, seed, guide = roles
    motif = {(0, 0): seed}
    directions = _AXES[orientation]
    for dr, dc in directions:
        for n in (1, 2):
            motif[(n * dr, n * dc)] = guide
    pool = _decoration_pool(orientation)
    maximum = min(len(pool), (3, 6, 10)[richness])
    target = int(rng.integers(0 if decor_mode == "free" else 1, maximum + 1))
    symmetry = int(rng.integers(0, 4))
    picked = set()
    for index in rng.permutation(len(pool)):
        if len(picked) >= target:
            break
        r, c = pool[int(index)]
        orbit = {(r, c)}
        if symmetry == 1:
            orbit.add((-r, c))
        elif symmetry == 2:
            orbit.add((r, -c))
        elif symmetry == 3:
            orbit.update({(-r, c), (r, -c), (-r, -c)})
        picked.update(orbit)
    if picked:
        other = [c for c in range(10) if c not in (background, fill, seed, guide)]
        rng.shuffle(other)
        distinct = min(len(picked), int(rng.integers(1, 4)))
        colors = other[:distinct]
        if decor_mode == "shared" or (decor_mode == "free" and rng.random() < 0.35):
            colors[0] = seed
        points = list(picked)
        rng.shuffle(points)
        for index, point in enumerate(points):
            motif[point] = colors[index] if index < len(colors) else colors[int(rng.integers(len(colors)))]
    return motif, directions


def _slot_dimensions(rng, count):
    choices = {
        1: ((1, 2), (2, 1)),
        2: ((1, 3), (3, 1), (2, 2)),
        3: ((2, 2), (1, 4), (4, 1)),
        4: ((2, 3), (3, 2)),
    }
    rows, cols = choices[count][int(rng.integers(len(choices[count])))]
    min_h, min_w = rows * 6 + rows - 1, cols * 6 + cols - 1
    height = 30 if rng.random() < 0.18 else int(rng.integers(max(13, min_h), 31))
    width = 30 if rng.random() < 0.18 else int(rng.integers(max(13, min_w), 31))
    return rows, cols, height, width


def _partition(rng, total, parts):
    extra = total - parts * 6 - (parts - 1)
    assert extra >= 0
    weights = rng.dirichlet(np.ones(parts))
    lengths = [6 + int(x) for x in rng.multinomial(extra, weights)]
    starts = []
    cursor = 0
    for length in lengths:
        starts.append((cursor, cursor + length))
        cursor += length + 1
    return starts


def _contains_corner(box, height, width):
    r0, r1, c0, c1 = box
    return any(r0 <= r < r1 and c0 <= c < c1 for r, c in
               ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)))


def _place_in_slot(rng, slot, size_range, height, width):
    sr0, sr1, sc0, sc1 = slot
    min_h, min_w, max_h, max_w = size_range
    max_h = min(max_h, sr1 - sr0)
    max_w = min(max_w, sc1 - sc0)
    if min_h > max_h or min_w > max_w:
        raise AssertionError("slot too small")
    for _ in range(80):
        box_h = int(rng.integers(min_h, max_h + 1))
        box_w = int(rng.integers(min_w, max_w + 1))
        r0 = int(rng.integers(sr0, sr1 - box_h + 1))
        c0 = int(rng.integers(sc0, sc1 - box_w + 1))
        box = (r0, r0 + box_h, c0, c0 + box_w)
        if not _contains_corner(box, height, width):
            return box
    raise AssertionError("could not keep the corners clear")


def _layout(rng, panel_count, motif):
    rows, cols, height, width = _slot_dimensions(rng, panel_count)
    row_slots = _partition(rng, height, rows)
    col_slots = _partition(rng, width, cols)
    slots = [(r0, r1, c0, c1) for r0, r1 in row_slots for c0, c1 in col_slots]
    order = [int(x) for x in rng.permutation(len(slots))]
    key_slot = slots[order[0]]
    panel_slots = [slots[i] for i in order[1:panel_count + 1]]
    drs = [point[0] for point in motif]
    dcs = [point[1] for point in motif]
    key_h, key_w = max(drs) - min(drs) + 1, max(dcs) - min(dcs) + 1
    key_box = _place_in_slot(rng, key_slot, (key_h, key_w, key_h, key_w), height, width)
    key_center = (key_box[0] - min(drs), key_box[2] - min(dcs))
    boxes = [_place_in_slot(rng, slot, (5, 5, slot[1] - slot[0], slot[3] - slot[2]),
                            height, width) for slot in panel_slots]
    boxes.sort()
    return height, width, key_center, boxes


def _candidate_seeds(box, motif, directions):
    r0, r1, c0, c1 = box
    arms = {(n * dr, n * dc) for dr, dc in directions for n in (1, 2)}
    local = [offset for offset in motif if offset not in arms]
    min_dr = min(dr for dr, _ in local)
    max_dr = max(dr for dr, _ in local)
    min_dc = min(dc for _, dc in local)
    max_dc = max(dc for _, dc in local)
    low_r = r0 + max(1, -min_dr)
    high_r = r1 - 1 - max(1, max_dr)
    low_c = c0 + max(1, -min_dc)
    high_c = c1 - 1 - max(1, max_dc)
    return [(r, c) for r in range(low_r, high_r + 1)
            for c in range(low_c, high_c + 1)]


def _sample_seeds(rng, box, motif, directions, guide):
    candidates = _candidate_seeds(box, motif, directions)
    if not candidates:
        raise AssertionError("panel has no valid seed position")
    rng.shuffle(candidates)
    r0, r1, c0, c1 = box
    capacity = min(6, max(1, min(r1 - r0, c1 - c0) // 3 + int((r1-r0)*(c1-c0) >= 150)))
    target = int(rng.integers(1, capacity + 1))
    chosen = []
    for point in candidates:
        if len(chosen) >= target:
            break
        if any(abs(point[0]-old[0]) + abs(point[1]-old[1]) < 3 for old in chosen):
            continue
        try:
            _box_assignments(box, chosen + [point], motif, directions, guide)
        except AssertionError:
            continue
        chosen.append(point)
    if not chosen:
        raise AssertionError("no compatible seeds")
    return chosen


def _has_long_ray(facts):
    for box, seeds in zip(facts["boxes"], facts["panel_seeds"]):
        r0, r1, c0, c1 = box
        for row, col in seeds:
            for dr, dc in facts["directions"]:
                if dr == -1 and row - r0 >= 3:
                    return True
                if dr == 1 and r1 - 1 - row >= 3:
                    return True
                if dc == -1 and col - c0 >= 3:
                    return True
                if dc == 1 and c1 - 1 - col >= 3:
                    return True
    return False


def _has_clipped_guide(facts):
    for box, seeds in zip(facts["boxes"], facts["panel_seeds"]):
        r0, r1, c0, c1 = box
        for row, col in seeds:
            for dr, dc in facts["directions"]:
                if (dr == -1 and row - r0 == 1 or
                    dr == 1 and r1 - 1 - row == 1 or
                    dc == -1 and col - c0 == 1 or
                    dc == 1 and c1 - 1 - col == 1):
                    return True
    return False


def _make_pair(rng, orientation, roles, decor_mode, richness,
               force_long_ray, force_clipped_guide):
    background, fill, seed, guide = roles
    for _ in range(100):
        motif, directions = _sample_motif(rng, orientation, roles, decor_mode, richness)
        panel_count = int(rng.choice((1, 2, 3, 4), p=(0.16, 0.39, 0.33, 0.12)))
        try:
            height, width, center, boxes = _layout(rng, panel_count, motif)
            seeds = [_sample_seeds(rng, box, motif, directions, guide) for box in boxes]
            grid = np.full((height, width), background, dtype=np.int8)
            for box, group in zip(boxes, seeds):
                r0, r1, c0, c1 = box
                grid[r0:r1, c0:c1] = fill
                for point in group:
                    grid[point] = seed
            for (dr, dc), color in motif.items():
                grid[center[0] + dr, center[1] + dc] = color
            facts = _inspect_input(grid)
            if force_long_ray and not _has_long_ray(facts):
                continue
            if force_clipped_guide and not _has_clipped_guide(facts):
                continue
            output = _expected_from_inspection(grid, facts)
            pair = {"input": grid, "output": output}
            _inspect_pair(pair)
            return pair
        except AssertionError:
            continue
    raise RuntimeError("could not construct a valid panel and key scene")


def _orientation_plan(rng, train_count, test_count):
    regime = int(rng.integers(0, 5))
    if regime == 0:  # source-like: vertical learned, horizontal held out
        train = ["both", "vertical"]
        test = ["horizontal"] * test_count
        extras = ("both", "vertical")
    elif regime == 1:  # rotated counterpart
        train = ["both", "horizontal"]
        test = ["vertical"] * test_count
        extras = ("both", "horizontal")
    elif regime == 2:  # mixed orientations in both splits
        train = ["vertical", "horizontal"]
        test = ["both" if i == 0 else ("vertical", "horizontal")[i % 2]
                for i in range(test_count)]
        extras = ("both", "vertical", "horizontal")
    elif regime == 3:  # an orientation can remain consistent across the task
        same = ("both", "vertical", "horizontal")[int(rng.integers(3))]
        train = [same, same]
        test = [same] * test_count
        extras = (same,)
    else:  # recombine demonstrated horizontal and both-axis behavior
        train = ["both", "horizontal"]
        test = [("horizontal", "both", "vertical")[i % 3] for i in range(test_count)]
        extras = ("both", "horizontal", "vertical")
    train.extend(extras[int(rng.integers(len(extras)))]
                 for _ in range(train_count - 2))
    rng.shuffle(train)
    return train, test


def _pair_digest(pair):
    a = np.asarray(pair["input"], dtype=np.int8)
    b = np.asarray(pair["output"], dtype=np.int8)
    return hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest() if a.shape == (30, 30) else None


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    train_axes, test_axes = _orientation_plan(rng, train_count, test_count)
    # The learned convention is the same visible rule in every pair. The
    # scene regime controls which compositions training demonstrates.
    richness = int(rng.integers(0, 3))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_roles = None
    for section, orientations in (("train", train_axes), ("test", test_axes)):
        for index, orientation in enumerate(orientations):
            for _ in range(30):
                roles = _sample_roles(rng, previous_roles)
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                pair = _make_pair(pair_rng, orientation, roles,
                                  ("shared", "distinct")[index] if section == "train" and index < 2 else "free",
                                  richness, section == "train" and index == 0,
                                  section == "train" and index == 1)
                key = _grid_key(pair["input"])
                if key in seen or _pair_digest(pair) in _SOURCE_DIGESTS:
                    continue
                seen.add(key)
                previous_roles = roles
                puzzle[section].append(pair)
                break
            else:
                raise RuntimeError("could not obtain a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    train_facts = []
    test_facts = []
    for section, destination in (("train", train_facts), ("test", test_facts)):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            destination.append(_inspect_pair(pair))
    # Training shows literal motif copying, ray continuation, and key erasure.
    assert any(f["seed"] in (color for offset, color in f["motif"].items()
                              if offset != (0, 0) and offset not in f["arm_offsets"])
               for f in train_facts)
    assert any(any(color != f["seed"] for offset, color in f["motif"].items()
                   if offset != (0, 0) and offset not in f["arm_offsets"])
               for f in train_facts)
    assert all(f["axes"] in _AXES for f in train_facts + test_facts)
    assert any(_has_long_ray(facts) for facts in train_facts)
    assert any(_has_clipped_guide(facts) for facts in train_facts)
    # Each test is independently determined by its visible key and panels.
    for pair in puzzle["test"]:
        inferred = _expected_from_inspection(pair["input"], _inspect_input(pair["input"]))
        assert np.array_equal(inferred, pair["output"])


def _puzzles_equal(first, second):
    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False
        for a, b in zip(first[section], second[section]):
            if not np.array_equal(a["input"], b["input"]):
                return False
            if not np.array_equal(a["output"], b["output"]):
                return False
    return True


def self_test() -> None:
    a = generate_puzzle(991, 6, 2)
    b = generate_puzzle(991, 6, 2)
    assert _puzzles_equal(a, b)
    assert not _puzzles_equal(a, generate_puzzle(992, 6, 2))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(2000 + i, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + i, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts = set(), set()
    shapes, palettes, motifs, layouts = set(), set(), set(), set()
    orientations, panels, seed_counts = set(), set(), set()
    shared_decor = distinct_decor = clipped_guide = 0
    multi_seed_panel = many_seed_panel = unequal_panels = 0
    plain_vertical = plain_horizontal = 0
    puzzle_signatures = set()
    within_shape = within_palette = within_orientation = 0
    occupancy_motifs = set()
    scene = {name: 0 for name in (
        "full_canvas", "compact", "large_panel", "single_panel", "four_panels",
        "panel_edge", "key_edge", "multicolor_decor", "decorated_multi_seed",
        "shared_and_clipped", "tall", "wide", "new_test_roles", "repeated_roles",
        "source_like", "reverse", "uniform", "mixed_splits", "many_colors",
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        seen_shapes, seen_palettes, seen_axes = set(), set(), set()
        signature = []
        train_axes, test_axes, train_roles, test_roles = [], [], [], []
        puzzle_colors = set()
        for pair in puzzle["train"] + puzzle["test"]:
            facts = _inspect_pair(pair)
            signature.append(_grid_key(pair["input"]))
            shape = pair["input"].shape
            colors = (facts["background"], facts["fill"], facts["seed"], facts["guide"])
            seen_shapes.add(shape)
            seen_palettes.add(colors)
            seen_axes.add(facts["axes"])
            if len(train_axes) < len(puzzle["train"]):
                train_axes.append(facts["axes"])
                train_roles.append(colors)
            else:
                test_axes.append(facts["axes"])
                test_roles.append(colors)
            puzzle_colors.update(int(x) for x in pair["input"].flat)
            shapes.add(shape)
            palettes.add(colors)
            orientations.add(facts["axes"])
            panels.add(len(facts["boxes"]))
            motifs.add((facts["axes"], tuple(sorted(facts["motif"].items()))))
            layouts.add((shape, tuple(facts["boxes"])))
            seed_counts.update(len(group) for group in facts["panel_seeds"])
            arm = facts["arm_offsets"]
            decor = {offset: color for offset, color in facts["motif"].items()
                     if offset not in arm and offset != (0, 0)}
            occupancy_motifs.add((facts["axes"], tuple(sorted(decor))))
            shared_decor += int(facts["seed"] in decor.values())
            distinct_decor += int(any(color != facts["seed"] for color in decor.values()))
            plain_vertical += int(facts["axes"] == "vertical" and not decor)
            plain_horizontal += int(facts["axes"] == "horizontal" and not decor)
            multi_seed_panel += int(any(len(group) >= 2 for group in facts["panel_seeds"]))
            many_seed_panel += int(any(len(group) >= 3 for group in facts["panel_seeds"]))
            unequal_panels += int(len({(r1-r0, c1-c0) for r0, r1, c0, c1 in facts["boxes"]}) > 1)
            scene["full_canvas"] += int(shape == (30, 30))
            scene["compact"] += int(max(shape) <= 18)
            scene["large_panel"] += int(any(max(r1-r0, c1-c0) >= 17 for r0, r1, c0, c1 in facts["boxes"]))
            scene["single_panel"] += int(len(facts["boxes"]) == 1)
            scene["four_panels"] += int(len(facts["boxes"]) == 4)
            scene["panel_edge"] += int(any(r0 == 0 or c0 == 0 or r1 == shape[0] or c1 == shape[1]
                                            for r0, r1, c0, c1 in facts["boxes"]))
            key_rows = [r for r, c in facts["key_coords"]]
            key_cols = [c for r, c in facts["key_coords"]]
            scene["key_edge"] += int(min(key_rows) == 0 or min(key_cols) == 0 or
                                     max(key_rows) == shape[0] - 1 or max(key_cols) == shape[1] - 1)
            scene["multicolor_decor"] += int(len(set(decor.values())) >= 2)
            scene["decorated_multi_seed"] += int(bool(decor) and any(len(g) >= 2 for g in facts["panel_seeds"]))
            scene["shared_and_clipped"] += int(facts["seed"] in decor.values() and _has_clipped_guide(facts))
            scene["tall"] += int(shape[0] >= 1.5 * shape[1])
            scene["wide"] += int(shape[1] >= 1.5 * shape[0])
            for box, group in zip(facts["boxes"], facts["panel_seeds"]):
                r0, r1, c0, c1 = box
                clipped_guide += int(any((sr-r0 == 1 or r1-1-sr == 1 or
                                          sc-c0 == 1 or c1-1-sc == 1) for sr, sc in group))
        puzzle_signatures.add(tuple(signature))
        within_shape += int(len(seen_shapes) > 1)
        within_palette += int(len(seen_palettes) > 1)
        within_orientation += int(len(seen_axes) > 1)
        scene["many_colors"] += int(len(puzzle_colors) >= 7)
        scene["new_test_roles"] += int(any(role not in train_roles for role in test_roles))
        scene["repeated_roles"] += int(len(set(train_roles + test_roles)) < len(train_roles + test_roles))
        scene["source_like"] += int("both" in train_axes and "vertical" in train_axes and
                                     "horizontal" not in train_axes and set(test_axes) == {"horizontal"})
        scene["reverse"] += int("both" in train_axes and "horizontal" in train_axes and
                                 "vertical" not in train_axes and set(test_axes) == {"vertical"})
        scene["uniform"] += int(len(set(train_axes + test_axes)) == 1)
        scene["mixed_splits"] += int(len(set(train_axes)) > 1 and len(set(test_axes)) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40
    assert len(shapes) >= 40 and len(palettes) >= 40
    assert len(motifs) >= 35 and len(layouts) >= 80
    assert len(occupancy_motifs) >= 40
    assert orientations == {"both", "vertical", "horizontal"}
    assert panels == {1, 2, 3, 4}
    assert max(seed_counts) >= 3
    assert shared_decor >= 4 and distinct_decor >= 4 and clipped_guide >= 4
    assert multi_seed_panel >= 8 and many_seed_panel >= 2 and unequal_panels >= 8
    assert plain_vertical >= 1 and plain_horizontal >= 1
    assert within_shape >= 30 and within_palette >= 30 and within_orientation >= 8
    assert scene["full_canvas"] >= 2 and scene["compact"] >= 1
    assert scene["large_panel"] >= 10 and scene["single_panel"] >= 5 and scene["four_panels"] >= 5
    assert scene["panel_edge"] >= 5 and scene["key_edge"] >= 5
    assert scene["multicolor_decor"] >= 10 and scene["decorated_multi_seed"] >= 10
    assert scene["shared_and_clipped"] >= 3
    assert scene["tall"] >= 5 and scene["wide"] >= 5
    assert scene["many_colors"] >= 30 and scene["new_test_roles"] >= 15
    assert scene["repeated_roles"] >= 10
    assert scene["source_like"] >= 2 and scene["reverse"] >= 2
    assert scene["uniform"] >= 2 and scene["mixed_splits"] >= 2
