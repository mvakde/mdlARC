"""Compositional instances of one-way, cell-centered mirror repair."""

import hashlib
import numpy as np


_SOURCE_PAIRS = {
    ("efc634b7031b017334ac3aa8028603d73d32a7035490dad21215ef517f4fa8bd", "989e04832795ab6bbd0c069c857510988793e8dc959e13fd4bcba26f240d4b55"),
    ("8cb21aebf25139cc9902bb12247fbe970d8e9c85baf79ac7a2d2cb44aba63c26", "5bdd1f7acd97e556ae1df244beb3ee3128a0fe178845f577662ac05e96af56aa"),
    ("f4e1ab46f8508b13bb1211497f0687525a663d60a4b87ec2560adddcb52e62f6", "fa4e105916435d7800db0793e75ada0e1f0d64095bd93556629fb264e174efce"),
}
_PROFILES = (("h", "upper"), ("h", "lower"), ("v", "left"), ("v", "right"))


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _connected(mask):
    points = np.argwhere(mask)
    if not len(points):
        return False
    start = tuple(int(v) for v in points[0])
    seen = {start}
    queue = [start]
    while queue:
        r, c = queue.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and mask[p] and p not in seen:
                seen.add(p)
                queue.append(p)
    return len(seen) == len(points)


def _template(rng, depth, length, stripe_start, stripe_length, background, body, guide):
    # Overlapping contiguous bands make a connected silhouette. Later carving
    # creates irregular open or enclosed holes without a periodic hole phase.
    mask = np.zeros((depth, length), dtype=bool)
    left, right = 0, length
    for d in range(depth):
        if d:
            previous_left, previous_right = left, right
            left = int(np.clip(left + rng.integers(-1, 3), 0, length - 2))
            right = int(np.clip(right + rng.integers(-2, 2), left + 2, length))
            if right <= left + 1:
                right = left + 2
            left = min(left, previous_right - 1)
            right = max(right, previous_left + 1, left + 2)
        mask[d, left:right] = True
    # Leave a firm connection to the axis and at least one cell at maximum depth.
    candidates = [(int(r), int(c)) for r, c in np.argwhere(mask) if r > 0 or
                  not stripe_start <= c < stripe_start + stripe_length]
    rng.shuffle(candidates)
    rate = float(rng.choice((0.0, 0.06, 0.13, 0.22, 0.30)))
    desired = min(int(round(rate * int(mask.sum()))), max(0, int(mask.sum()) - 4))
    removed = 0
    for r, c in candidates:
        if removed >= desired:
            break
        if r == 0 and c in (0, length - 1):
            continue
        mask[r, c] = False
        if not mask[-1].any() or not np.any(mask[0, stripe_start:stripe_start + stripe_length]) or not _connected(mask):
            mask[r, c] = True
        else:
            removed += 1
    result = np.full((depth, length), background, dtype=np.int8)
    result[mask] = body
    # Details are isolated from the central stripe and each other, so the
    # marker has exactly one straight run of three or more cells per object.
    if depth >= 2:
        detail_candidates = [(int(r), int(c)) for r, c in np.argwhere(mask) if r >= 1]
        rng.shuffle(detail_candidates)
        wanted = int(rng.choice((0, 0, 1, 1, 2, 3)))
        details = []
        if wanted:
            for r, c in detail_candidates:
                if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in details):
                    details.append((r, c))
                    if len(details) >= wanted:
                        break
        for pos in details:
            result[pos] = guide
    return result


def _corrupt(rng, template, background, body, guide, regime):
    result = template.copy()
    occupied = [tuple(int(x) for x in p) for p in np.argwhere(template != background)]
    holes = [tuple(int(x) for x in p) for p in np.argwhere(template == background)]
    rng.shuffle(occupied)
    rng.shuffle(holes)
    severity = str(rng.choice(("near", "moderate", "heavy"), p=(0.23, 0.49, 0.28) if regime == "target_test" else (0.32, 0.48, 0.20)))
    fraction = {"near": (0.06, 0.17), "moderate": (0.18, 0.36), "heavy": (0.37, 0.62)}[severity]
    omissions = max(1, min(len(occupied) - 2, int(round(len(occupied) * rng.uniform(*fraction)))))
    for pos in occupied[:omissions]:
        result[pos] = background
    mode = str(rng.choice(("omission", "mixed", "color"), p=(0.37, 0.39, 0.24)))
    additions = 0
    wrong = 0
    if mode == "mixed" and holes and omissions >= 2:
        additions = min(len(holes), omissions - 1, int(rng.integers(1, 4)))
        for pos in holes[:additions]:
            result[pos] = guide if int(rng.integers(3)) == 0 else body
    if mode in ("mixed", "color"):
        remaining = occupied[omissions:]
        wrong = min(len(remaining), int(rng.integers(1, 4)))
        for pos in remaining[:wrong]:
            result[pos] = body if template[pos] == guide else guide
    assert np.count_nonzero(result != background) < np.count_nonzero(template != background)
    return result, (omissions, additions, wrong)


def _figure(rng, slot_height, slot_width, profile, background, body, guide, regime):
    orientation, authority = profile
    along_capacity = slot_width if orientation == "h" else slot_height
    across_capacity = slot_height if orientation == "h" else slot_width
    length_limit = min(10, along_capacity)
    depth_limit = min(5, (across_capacity - 1) // 2)
    length = max(int(rng.integers(3, length_limit + 1)) for _ in range(2))
    depth = max(int(rng.integers(1, depth_limit + 1)) for _ in range(2))
    # Shorter stripes are common, including the source's three-cell example.
    stripe_length = int(rng.integers(3, length + 1))
    stripe_start = int(rng.integers(0, length - stripe_length + 1))
    template = _template(rng, depth, length, stripe_start, stripe_length, background, body, guide)
    shape = (2 * depth + 1, length) if orientation == "h" else (length, 2 * depth + 1)
    def compose(damaged):
        original = np.full(shape, background, dtype=np.int8)
        repaired = original.copy()
        if orientation == "h":
            original[depth, stripe_start:stripe_start + stripe_length] = guide
            repaired[depth, stripe_start:stripe_start + stripe_length] = guide
            for d in range(1, depth + 1):
                values = template[d - 1]
                repaired[depth - d] = values
                repaired[depth + d] = values
                original[depth - d] = values if authority == "upper" else damaged[d - 1]
                original[depth + d] = values if authority == "lower" else damaged[d - 1]
        else:
            original[stripe_start:stripe_start + stripe_length, depth] = guide
            repaired[stripe_start:stripe_start + stripe_length, depth] = guide
            for d in range(1, depth + 1):
                values = template[d - 1]
                repaired[:, depth - d] = values
                repaired[:, depth + d] = values
                original[:, depth - d] = values if authority == "left" else damaged[d - 1]
                original[:, depth + d] = values if authority == "right" else damaged[d - 1]
        return original, repaired

    for _ in range(12):
        damaged, damage = _corrupt(rng, template, background, body, guide, regime)
        original, repaired = compose(damaged)
        if _runs(original, guide) == _runs(repaired, guide):
            return original, repaired, damage
    damaged = template.copy()
    damaged[tuple(int(v) for v in np.argwhere(template != background)[0])] = background
    original, repaired = compose(damaged)
    return original, repaired, (1, 0, 0)


def _slots(rng, count):
    if count == 1:
        rows, cols = 1, 1
    elif count == 2:
        rows, cols = ((1, 2), (2, 1), (2, 2))[int(rng.integers(3))]
    elif count == 3:
        rows, cols = ((1, 3), (3, 1), (2, 2))[int(rng.integers(3))]
    elif count == 4:
        rows, cols = ((1, 4), (4, 1), (2, 2), (2, 3))[int(rng.integers(4))]
    else:
        rows, cols = ((2, 3), (3, 2))[int(rng.integers(2))]
    locations = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(locations)
    return rows, cols, locations[:count]


def _partition(rng, total, parts):
    values = [5] * parts
    for _ in range(total - 5 * parts):
        values[int(rng.integers(parts))] += 1
    return values


def _layout(rng, count, regime):
    rows, cols, assignments = _slots(rng, count)
    min_height = 5 * rows + 2 * (rows - 1) + 2
    min_width = 5 * cols + 2 * (cols - 1) + 2
    max_side = 22 if count == 1 else 27 if count == 2 else 30
    height = int(rng.integers(max(9, min_height), max_side + 1))
    width = int(rng.integers(max(9, min_width), max_side + 1))
    if rng.random() < 0.12:
        familiar_sizes = [(h, w) for h, w in ((14, 14), (17, 19), (24, 28))
                          if min_height <= h <= max_side and min_width <= w <= max_side]
        if familiar_sizes:
            height, width = familiar_sizes[int(rng.integers(len(familiar_sizes)))]
    # Modest size bias for test, with broad overlap between regimes.
    if regime == "target_test" and int(rng.integers(3)) == 0:
        height = max(height, min(30, min_height + 4))
        width = max(width, min(30, min_width + 4))
    row_sizes = _partition(rng, height - 2 * (rows - 1) - 2, rows)
    col_sizes = _partition(rng, width - 2 * (cols - 1) - 2, cols)
    row_starts, col_starts = [], []
    offset = 1
    for size in row_sizes:
        row_starts.append(offset)
        offset += size + 2
    offset = 1
    for size in col_sizes:
        col_starts.append(offset)
        offset += size + 2
    return height, width, [(row_starts[r], col_starts[c], row_sizes[r], col_sizes[c]) for r, c in assignments]


def _pair(rng, regime, index, params, required_profile, forbid_lower=False):
    count = int(rng.choice((1, 2, 3, 4) if regime == "source_train" else (1, 2, 3, 4, 5), p=(0.29, 0.31, 0.25, 0.15) if regime == "source_train" else (0.12, 0.18, 0.23, 0.27, 0.20)))
    height, width, slots = _layout(rng, count, regime)
    background = int(rng.choice(params["backgrounds"]))
    original = np.full((height, width), background, dtype=np.int8)
    repaired = original.copy()
    profiles = [_PROFILES[int(rng.integers(4))] for _ in range(count)]
    if forbid_lower:
        profiles = [("h", "upper") if profile == ("h", "lower") else profile for profile in profiles]
    if required_profile is not None:
        profiles[0] = required_profile
    # The first two training examples use disjoint body subsets. Thus their
    # only color with a straight marker in common is the shared guide.
    palette = params["bodies"][:2] if regime == "source_train" and index == 0 else params["bodies"][2:] if regime == "source_train" and index == 1 else params["bodies"]
    for slot, profile in zip(slots, profiles):
        top, left, sh, sw = slot
        body = int(rng.choice(palette))
        figure_input, figure_output, _ = _figure(rng, sh, sw, profile, background, body, params["guide"], regime)
        fh, fw = figure_input.shape
        dr = int(rng.integers(sh - fh + 1))
        dc = int(rng.integers(sw - fw + 1))
        rr, cc = top + dr, left + dc
        original[rr:rr + fh, cc:cc + fw] = figure_input
        repaired[rr:rr + fh, cc:cc + fw] = figure_output
    return {"input": original, "output": repaired}


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_hash(grid):
    return hashlib.sha256(bytes([grid.shape[0], grid.shape[1]]) + grid.astype(np.uint8).tobytes()).hexdigest()


def _components(mask):
    unseen = set((int(r), int(c)) for r, c in np.argwhere(mask))
    result = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        queue = [start]
        while queue:
            r, c = queue.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    comp.add(p)
                    queue.append(p)
        result.append(comp)
    return result


def _runs(grid, color):
    result = []
    for orientation in ("h", "v"):
        lines = grid if orientation == "h" else grid.T
        for fixed, line in enumerate(lines):
            start = 0
            while start < len(line):
                if line[start] != color:
                    start += 1
                    continue
                end = start + 1
                while end < len(line) and line[end] == color:
                    end += 1
                if end - start >= 3:
                    result.append((orientation, fixed, start, end - start))
                start = end
    return result


def _inspect_pair(pair, guide):
    source, target = pair["input"], pair["output"]
    assert source.shape == target.shape and not np.array_equal(source, target)
    background = int(target[0, 0])
    assert background != guide
    assert all(int(grid[r, c]) == background for grid in (source, target)
               for r in (0, -1) for c in (0, -1))
    components = _components(target != background)
    assert 1 <= len(components) <= 5
    boxes = []
    facts = []
    claimed = np.zeros(target.shape, dtype=bool)
    for comp in components:
        rr = [p[0] for p in comp]
        cc = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1
        assert not claimed[r0:r1, c0:c1].any()
        claimed[r0:r1, c0:c1] = True
        boxes.append((r0, r1, c0, c1))
        out = target[r0:r1, c0:c1]
        inp = source[r0:r1, c0:c1]
        colors = set(int(v) for v in np.unique(out)) - {background, guide}
        assert len(colors) == 1
        axes = _runs(out, guide)
        assert len(axes) == 1
        assert _runs(inp, guide) == axes
        orientation, fixed, start, length = axes[0]
        if orientation == "h":
            assert fixed > 0 and fixed == out.shape[0] - 1 - fixed
            axis_input, axis_output = inp[fixed], out[fixed]
            a_out = out[:fixed][::-1]
            b_out = out[fixed + 1:]
            a_in = inp[:fixed][::-1]
            b_in = inp[fixed + 1:]
            names = ("upper", "lower")
        else:
            assert fixed > 0 and fixed == out.shape[1] - 1 - fixed
            axis_input, axis_output = inp[:, fixed], out[:, fixed]
            a_out = out[:, :fixed][:, ::-1]
            b_out = out[:, fixed + 1:]
            a_in = inp[:, :fixed][:, ::-1]
            b_in = inp[:, fixed + 1:]
            names = ("left", "right")
        assert np.array_equal(axis_input, axis_output)
        assert np.array_equal(a_out, b_out)
        if np.array_equal(a_in, a_out):
            assert not np.array_equal(b_in, b_out)
            good, bad, truth, authority = a_in, b_in, b_out, names[0]
        else:
            assert np.array_equal(b_in, b_out)
            good, bad, truth, authority = b_in, a_in, a_out, names[1]
        good_count = int(np.count_nonzero(good != background))
        bad_count = int(np.count_nonzero(bad != background))
        assert good_count > bad_count
        omissions = int(np.count_nonzero((bad == background) & (truth != background)))
        additions = int(np.count_nonzero((bad != background) & (truth == background)))
        wrong = int(np.count_nonzero((bad != background) & (truth != background) & (bad != truth)))
        assert omissions >= 1 and omissions > additions
        facts.append({"orientation": orientation, "authority": authority, "length": length, "depth": fixed,
                      "span": out.shape[1] if orientation == "h" else out.shape[0],
                      "density": good_count / (fixed * (out.shape[1] if orientation == "h" else out.shape[0])),
                      "omissions": omissions, "additions": additions, "wrong": wrong,
                      "body": next(iter(colors)), "center": (r0 + (r1 - 1) / 2, c0 + (c1 - 1) / 2)})
    assert np.all(source[~claimed] == background)
    assert np.all(target[~claimed] == background)
    return {"shape": target.shape, "background": background, "figures": facts}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    all_pairs = puzzle["train"] + puzzle["test"]
    for pair in all_pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
        key = _grid_key(pair["input"])
        pair_key = (key, _grid_key(pair["output"]))
        assert key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        assert (_source_hash(pair["input"]), _source_hash(pair["output"])) not in _SOURCE_PAIRS
    candidate = set(range(10))
    for pair in all_pairs:
        candidate &= {int(v) for v in np.unique(pair["output"]) if _runs(pair["output"], int(v))}
    valid = []
    for color in candidate:
        try:
            info = [_inspect_pair(pair, color) for pair in all_pairs]
        except (AssertionError, ValueError, IndexError):
            continue
        valid.append((color, info))
    assert len(valid) == 1
    _, info = valid[0]
    train_info = info[:len(puzzle["train"])]
    test_info = info[len(puzzle["train"]):]
    train_orientations = {f["orientation"] for x in train_info for f in x["figures"]}
    assert train_orientations == {"h", "v"}
    assert all(1 <= len(x["figures"]) <= 4 for x in train_info)
    assert all(1 <= len(x["figures"]) <= 5 for x in test_info)
    assert all(set(f["orientation"] for f in x["figures"]) <= train_orientations for x in test_info)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    colors = [int(x) for x in rng.permutation(10)]
    params = {"guide": colors[0], "bodies": colors[1:5], "backgrounds": colors[5:]}
    withhold_lower = bool(rng.integers(0, 3) == 0)
    horizontal = ("h", "upper") if withhold_lower else ("h", "lower")
    vertical = _PROFILES[2 + int(rng.integers(2))]
    mandatory = [horizontal, vertical]
    rng.shuffle(mandatory)
    puzzle = {"train": [], "test": []}
    for regime, count, destination in (("source_train", train_count, puzzle["train"]),
                                       ("target_test", test_count, puzzle["test"])):
        for index in range(count):
            child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            required = mandatory[index] if regime == "source_train" and index < 2 else ("h", "lower") if regime == "target_test" and index == 0 and withhold_lower else None
            destination.append(_pair(child, regime, index, params, required,
                                     forbid_lower=withhold_lower and regime == "source_train"))
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = []
    train_counts, test_counts, backgrounds, guides = set(), set(), set(), set()
    shapes, counts, profiles, lengths, depths, short_axes = set(), set(), set(), set(), set(), 0
    damage_modes = set()
    near, heavy, solid_bands, guide_details, reused_bodies, enclosed_holes = 0, 0, 0, 0, 0, 0
    varying_backgrounds, varying_counts, lower_taught, lower_withheld = 0, 0, 0, 0
    source_dimensions = 0
    densities = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        guide = next(color for color in range(10) if all(_runs(p["output"], color) for p in puzzle["train"] + puzzle["test"]) and all(_inspect_pair(p, color) for p in puzzle["train"] + puzzle["test"]))
        guides.add(guide)
        puzzle_backgrounds, puzzle_counts = set(), set()
        train_profiles, test_profiles = set(), set()
        for split in ("train", "test"):
          for pair in puzzle[split]:
            info = _inspect_pair(pair, guide)
            backgrounds.add(info["background"])
            puzzle_backgrounds.add(info["background"])
            shapes.add(info["shape"])
            source_dimensions += info["shape"] in ((14, 14), (17, 19), (24, 28))
            counts.add(len(info["figures"]))
            puzzle_counts.add(len(info["figures"]))
            reused_bodies += len({f["body"] for f in info["figures"]}) < len(info["figures"])
            for f in info["figures"]:
                (train_profiles if split == "train" else test_profiles).add((f["orientation"], f["authority"]))
                profiles.add((f["orientation"], f["authority"]))
                lengths.add(f["length"])
                depths.add(f["depth"])
                densities.add("sparse" if f["density"] < 0.60 else "dense" if f["density"] > 0.85 else "middle")
                short_axes += f["length"] < f["span"]
                damage_modes.add((f["additions"] > 0, f["wrong"] > 0))
                changes = f["omissions"] + f["additions"] + f["wrong"]
                near += changes <= 2
                heavy += changes >= 10
            for component in _components(pair["output"] != info["background"]):
                rows = [p[0] for p in component]
                cols = [p[1] for p in component]
                crop = pair["output"][min(rows):max(rows) + 1, min(cols):max(cols) + 1]
                orientation, fixed, _, length = _runs(crop, guide)[0]
                bands = [crop[r] for r in range(crop.shape[0]) if r != fixed] if orientation == "h" else [crop[:, c] for c in range(crop.shape[1]) if c != fixed]
                solid_bands += any(np.all(band != info["background"]) for band in bands)
                guide_details += int(np.count_nonzero(crop == guide) > length)
                half = crop[:fixed] if orientation == "h" else crop[:, :fixed].T
                for r in range(1, half.shape[0] - 1):
                    for c in range(1, half.shape[1] - 1):
                        if half[r, c] == info["background"] and all(half[rr, cc] != info["background"] for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
                            enclosed_holes += 1
        varying_backgrounds += len(puzzle_backgrounds) > 1
        varying_counts += len(puzzle_counts) > 1
        lower_taught += ("h", "lower") in train_profiles
        lower_withheld += ("h", "lower") not in train_profiles and ("h", "lower") in test_profiles
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 5 and len(guides) >= 5
    assert len(shapes) >= 30 and counts == {1, 2, 3, 4, 5}
    assert profiles == set(_PROFILES) and {3, 4, 5} <= lengths and {1, 2, 3, 4, 5} <= depths
    assert short_axes > 50 and len(damage_modes) >= 3
    assert near > 30 and heavy > 20 and solid_bands > 100 and guide_details > 50 and enclosed_holes > 10
    assert reused_bodies > 10 and varying_backgrounds > 10 and varying_counts > 10
    assert lower_taught > 5 and lower_withheld > 5 and source_dimensions > 0
    assert densities == {"sparse", "middle", "dense"}
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 6, 3)):
        p = generate_puzzle(seed, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for kwargs in ({"num_test": 1}, {"num_test": 4}, {"num_train": 2}, {"num_train": 10}):
        validate_generated_puzzle(generate_puzzle(110, **kwargs))
    assert _equal(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3))
    assert not _equal(generate_puzzle(777, 7, 3), generate_puzzle(778, 7, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_test": 1.5}):
        try:
            generate_puzzle(900, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
