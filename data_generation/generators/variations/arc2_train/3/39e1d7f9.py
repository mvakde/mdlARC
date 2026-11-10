"""Generators for colored-stencil completion on a regular ARC lattice."""

import hashlib

import numpy as np


ORTHOGONAL = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBORS = ORTHOGONAL + DIAGONAL
SOURCE_DIGESTS = frozenset({
    "a0731b3b91841c49e6c55f860be5c6f683d934fd955fb9c1e6b4ff19b19aa1ae",
    "14007680696ee123101534f700989edf0704d7a05cd24109099cb0c1cc56f93d",
    "020269129f1e78384d78f0956918c6b48b4ba4c3952034a3c9b99edfff22b74a",
    "0d8e2c59b4168ad90ae256072be9f3f8a99cd54bb05e80fa1d210e30fe7d2acc",
})


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _inside(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols


def _clipped(center, offsets, rows, cols):
    return any(
        not _inside(center[0] + dr, center[1] + dc, rows, cols)
        for dr, dc in offsets
    )


def _render(macro, tile_h, tile_w, divider):
    rows, cols = macro.shape
    grid = np.full(
        (rows * (tile_h + 1) - 1, cols * (tile_w + 1) - 1),
        divider,
        dtype=np.int8,
    )
    for r in range(rows):
        for c in range(cols):
            grid[
                r * (tile_h + 1):r * (tile_h + 1) + tile_h,
                c * (tile_w + 1):c * (tile_w + 1) + tile_w,
            ] = macro[r, c]
    return grid


def _lattice(grid):
    """Parse the visible divider lines without generation metadata."""
    height, width = grid.shape
    possibilities = []
    for tile_h in range(1, 6):
        if (height + 1) % (tile_h + 1):
            continue
        rows = (height + 1) // (tile_h + 1)
        if rows < 4:
            continue
        line_rows = [r * (tile_h + 1) - 1 for r in range(1, rows)]
        for tile_w in range(1, 6):
            if (width + 1) % (tile_w + 1):
                continue
            cols = (width + 1) // (tile_w + 1)
            if cols < 4:
                continue
            line_cols = [c * (tile_w + 1) - 1 for c in range(1, cols)]
            divider = int(grid[line_rows[0], line_cols[0]])
            if not np.all(grid[line_rows, :] == divider):
                continue
            if not np.all(grid[:, line_cols] == divider):
                continue
            macro = np.empty((rows, cols), dtype=np.int8)
            valid = True
            for r in range(rows):
                for c in range(cols):
                    block = grid[
                        r * (tile_h + 1):r * (tile_h + 1) + tile_h,
                        c * (tile_w + 1):c * (tile_w + 1) + tile_w,
                    ]
                    color = int(block[0, 0])
                    if color == divider or not np.all(block == color):
                        valid = False
                        break
                    macro[r, c] = color
                if not valid:
                    break
            if valid:
                possibilities.append((macro, tile_h, tile_w, divider))
    assert len(possibilities) == 1
    return possibilities[0]


def _input_solution(grid):
    """Infer the single complete colored stencil from the input alone."""
    macro, tile_h, tile_w, divider = _lattice(grid)
    rows, cols = macro.shape
    colors, counts = np.unique(macro, return_counts=True)
    maximum = int(np.max(counts))
    assert int(np.sum(counts == maximum)) == 1
    background = int(colors[int(np.argmax(counts))])
    occupied = {
        (int(r), int(c))
        for r, c in np.argwhere(macro != background)
    }
    solutions = []
    for seed in map(int, colors):
        if seed == background:
            continue
        centers = {
            (int(r), int(c))
            for r, c in np.argwhere(macro == seed)
        }
        if not 2 <= len(centers) <= 6:
            continue
        supports = occupied - centers
        if not 3 <= len(supports) <= 8:
            continue
        for prototype in sorted(centers):
            if not (0 < prototype[0] < rows - 1 and 0 < prototype[1] < cols - 1):
                continue
            if any(
                max(abs(r - prototype[0]), abs(c - prototype[1])) != 1
                for r, c in supports
            ):
                continue
            stencil = {
                (r - prototype[0], c - prototype[1]): int(macro[r, c])
                for r, c in supports
            }
            if any(
                _inside(center[0] + dr, center[1] + dc, rows, cols)
                and (center[0] + dr, center[1] + dc) in supports
                for center in centers - {prototype}
                for dr, dc in NEIGHBORS
            ):
                continue
            output = macro.copy()
            valid = True
            shared = False
            placed = set(supports)
            for center in sorted(centers - {prototype}):
                for (dr, dc), color in stencil.items():
                    r, c = center[0] + dr, center[1] + dc
                    if not _inside(r, c, rows, cols):
                        continue
                    if int(output[r, c]) not in (background, color):
                        valid = False
                        break
                    if (r, c) in placed:
                        shared = True
                    placed.add((r, c))
                    output[r, c] = color
                if not valid:
                    break
            if valid and not np.array_equal(output, macro):
                solutions.append({
                    "macro_input": macro,
                    "macro_output": output,
                    "tile_h": tile_h,
                    "tile_w": tile_w,
                    "divider": divider,
                    "background": background,
                    "seed": seed,
                    "prototype": prototype,
                    "targets": tuple(sorted(centers - {prototype})),
                    "stencil": stencil,
                    "shared": shared,
                })
    assert len(solutions) == 1
    return solutions[0]


def _signature(grid):
    return grid.shape, grid.tobytes()


def _source_digest(first, second):
    return hashlib.sha256(
        bytes(first.shape) + first.astype(np.int8).tobytes()
        + second.astype(np.int8).tobytes()
    ).hexdigest()


def _motif(rng, style, section):
    if style == "source_like" and section == "train":
        kind = str(rng.choice(["plus", "ring"]))
        palette_size = 1
    elif style == "source_like" and section == "test" and rng.random() < 0.75:
        kind = "ring"
        palette_size = 2
    elif style == "simple":
        kind = str(rng.choice(["plus", "ring"]))
        palette_size = 1
    else:
        kind = str(rng.choice(["plus", "ring", "asymmetric"], p=[0.25, 0.25, 0.5]))
        palette_size = int(rng.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1]))

    if kind == "plus":
        offsets = list(ORTHOGONAL)
    elif kind == "ring":
        offsets = list(NEIGHBORS)
    else:
        subset_size = int(rng.integers(3, 8))
        offsets = [NEIGHBORS[int(i)] for i in rng.choice(8, subset_size, replace=False)]
        offsets.sort()
    palette_size = min(palette_size, len(offsets))
    colors = [int(x) for x in rng.permutation(10)[:3 + palette_size]]
    background, divider, seed = colors[:3]
    supports = colors[3:]
    if kind == "ring" and palette_size == 2 and rng.random() < 0.45:
        if rng.random() < 0.5:
            assignments = [0 if offset in ORTHOGONAL else 1 for offset in offsets]
        else:
            assignments = [1 if offset in ORTHOGONAL else 0 for offset in offsets]
    else:
        assignments = [int(x) for x in rng.integers(0, palette_size, len(offsets))]
        for i, index in enumerate(rng.permutation(len(offsets))[:palette_size]):
            assignments[int(index)] = i
    stencil = {offset: supports[assignment] for offset, assignment in zip(offsets, assignments)}
    return background, divider, seed, stencil


def _shape(rng, style, section, reuse=None):
    if reuse is not None:
        rows, cols = reuse
        options = [
            (h, w)
            for h in range(1, 5)
            for w in range(1, 5)
            if rows * (h + 1) - 1 <= 30
            and cols * (w + 1) - 1 <= 30
        ]
        h, w = options[int(rng.integers(len(options)))]
        return rows, cols, h, w
    if style == "source_like" and section == "test" and rng.random() < 0.7:
        h = w = 2
    else:
        h = int(rng.integers(1, 5))
        w = int(rng.integers(1, 5))
    rows = int(rng.integers(4, min(15, (31 // (h + 1))) + 1))
    cols = int(rng.integers(4, min(15, (31 // (w + 1))) + 1))
    return rows, cols, h, w


def _layout(rng, rows, cols, stencil, desired, edge_policy, overlap_policy):
    offsets = tuple(stencil)
    prototypes = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]
    rng.shuffle(prototypes)
    for prototype in prototypes:
        prototype_support = {
            (prototype[0] + dr, prototype[1] + dc)
            for dr, dc in offsets
        }
        candidates = []
        for r in range(rows):
            for c in range(cols):
                center = (r, c)
                if center == prototype:
                    continue
                if any(
                    max(abs(r - sr), abs(c - sc)) <= 1
                    for sr, sc in prototype_support
                ):
                    continue
                clipped = _clipped(center, offsets, rows, cols)
                if edge_policy == "interior" and clipped:
                    continue
                if edge_policy == "clipped" and not clipped:
                    continue
                candidates.append(center)
        rng.shuffle(candidates)
        chosen = []
        paint = {}
        for _ in range(desired):
            legal = []
            for center in candidates:
                if center in chosen:
                    continue
                if any(
                    max(abs(center[0] - other[0]), abs(center[1] - other[1])) <= 1
                    for other in chosen
                ):
                    continue
                additions = {}
                good = True
                shared = False
                for (dr, dc), color in stencil.items():
                    pos = (center[0] + dr, center[1] + dc)
                    if not _inside(*pos, rows, cols):
                        continue
                    if pos in prototype_support:
                        good = False
                        break
                    if pos in paint:
                        if paint[pos] != color:
                            good = False
                            break
                        shared = True
                    additions[pos] = color
                if good and additions:
                    legal.append((center, additions, shared))
            if not legal:
                break
            if overlap_policy and chosen:
                overlap = [item for item in legal if item[2]]
                if overlap:
                    legal = overlap
            center, additions, _ = legal[int(rng.integers(len(legal)))]
            chosen.append(center)
            paint.update(additions)
        if len(chosen) == desired:
            return prototype, chosen
    return None


def _make_pair(rng, style, section, reuse=None):
    for _ in range(180):
        rows, cols, tile_h, tile_w = _shape(rng, style, section, reuse)
        background, divider, seed, stencil = _motif(rng, style, section)
        if style == "source_like" and section == "test" and rng.random() < 0.65:
            desired = int(rng.integers(2, 5))
        else:
            desired = int(rng.integers(1, 5))
        edge_policy = str(rng.choice(["interior", "clipped", "any"], p=[0.3, 0.3, 0.4]))
        overlap_policy = bool(rng.random() < 0.35)
        placement = _layout(rng, rows, cols, stencil, desired, edge_policy, overlap_policy)
        if placement is None:
            continue
        prototype, targets = placement
        macro = np.full((rows, cols), background, dtype=np.int8)
        macro[prototype] = seed
        for (dr, dc), color in stencil.items():
            macro[prototype[0] + dr, prototype[1] + dc] = color
        for center in targets:
            macro[center] = seed
        if int(np.sum(macro == background)) <= int(np.max([
            np.sum(macro == color) for color in np.unique(macro) if color != background
        ])):
            continue
        input_grid = _render(macro, tile_h, tile_w, divider)
        try:
            solution = _input_solution(input_grid)
        except AssertionError:
            continue
        output_grid = _render(
            solution["macro_output"], tile_h, tile_w, divider
        )
        if _source_digest(input_grid, output_grid) in SOURCE_DIGESTS:
            continue
        return {"input": input_grid, "output": output_grid}
    raise ValueError("could not construct an unambiguous pair")


def _features(pair):
    analysis = _input_solution(pair["input"])
    stencil = analysis["stencil"]
    offsets = set(stencil)
    if offsets == set(ORTHOGONAL):
        geometry = "plus"
    elif offsets == set(NEIGHBORS):
        geometry = "ring"
    else:
        geometry = "asymmetric"
    rows, cols = analysis["macro_input"].shape
    clipped = [
        _clipped(center, stencil, rows, cols)
        for center in analysis["targets"]
    ]
    return {
        "roles": (analysis["background"], analysis["divider"], analysis["seed"]),
        "shape": (rows, cols, analysis["tile_h"], analysis["tile_w"]),
        "geometry": geometry,
        "colors": len(set(stencil.values())),
        "targets": len(analysis["targets"]),
        "clipped": any(clipped),
        "all_clipped": all(clipped),
        "all_interior": not any(clipped),
        "overlap": analysis["shared"],
        "stencil": tuple(sorted(stencil.items())),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    features = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            first, second = pair["input"], pair["output"]
            for grid in (first, second):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert 0 <= int(np.min(grid)) <= int(np.max(grid)) <= 9
            assert first.shape == second.shape
            input_key = _signature(first)
            pair_key = (input_key, _signature(second))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert _source_digest(first, second) not in SOURCE_DIGESTS
            analysis = _input_solution(first)
            expected = _render(
                analysis["macro_output"],
                analysis["tile_h"],
                analysis["tile_w"],
                analysis["divider"],
            )
            assert np.array_equal(expected, second)
            assert not np.array_equal(first, second)
            features.append(_features(pair))
    # Every pair visibly supplies its own palette and oriented stencil.
    # The puzzle-level convention is the unchanged lattice-copying operation.
    assert len(set(feature["roles"] for feature in features)) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    train_count, test_count = _counts(rng, num_train, num_test)
    # Only the copy rule is shared. Each input displays its own color roles
    # and exact oriented stencil, as in the original source puzzle.
    style = str(rng.choice(["mixed", "simple", "source_like"], p=[0.6, 0.2, 0.2]))
    puzzle = {"train": [], "test": []}
    used = set()
    macro_reuse = None
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            accepted = None
            for _ in range(40):
                reuse = macro_reuse if (section == "train" and index == 1 and rng.random() < 0.45) else None
                pair = _make_pair(rng, style, section, reuse)
                signature = _signature(pair["input"])
                if signature not in used:
                    accepted = pair
                    used.add(signature)
                    break
            if accepted is None:
                raise ValueError("could not construct unique puzzle inputs")
            puzzle[section].append(accepted)
            if section == "train" and index == 0:
                macro = _input_solution(accepted["input"])["macro_input"]
                macro_reuse = macro.shape
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(first, second):
    return all(
        len(first[section]) == len(second[section])
        and all(
            np.array_equal(a[name], b[name])
            for a, b in zip(first[section], second[section])
            for name in ("input", "output")
        )
        for section in ("train", "test")
    )


def self_test() -> None:
    observations = []
    train_counts, test_counts = set(), set()
    signatures = set()
    split_features = {"train": [], "test": []}
    role_colors = {"background": set(), "divider": set(), "seed": set(), "support": set()}
    macro_target_counts = {}
    asymmetric_offsets = set()
    within = {
        "simple": 0, "mono_train_multi_test": 0, "multi_train": 0,
        "mono_test": 0, "same_macro_new_tile": 0, "mixed_clipping": 0,
    }
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(
            _signature(pair["input"])
            for section in ("train", "test")
            for pair in puzzle[section]
        ))
        features = {
            section: [_features(pair) for pair in puzzle[section]]
            for section in ("train", "test")
        }
        for section in ("train", "test"):
            observations.extend(features[section])
            split_features[section].extend(features[section])
        current = features["train"] + features["test"]
        within["simple"] += all(
            f["geometry"] in ("plus", "ring") and f["colors"] == 1
            for f in current
        )
        within["mono_train_multi_test"] += (
            all(f["colors"] == 1 for f in features["train"])
            and any(f["colors"] > 1 for f in features["test"])
        )
        within["multi_train"] += any(f["colors"] > 1 for f in features["train"])
        within["mono_test"] += any(f["colors"] == 1 for f in features["test"])
        within["same_macro_new_tile"] += any(
            len({f["shape"][2:] for f in current if f["shape"][:2] == shape}) > 1
            for shape in {f["shape"][:2] for f in current}
        )
        within["mixed_clipping"] += any(
            f["clipped"] and not f["all_clipped"] for f in current
        )
        for f in current:
            for role, color in zip(("background", "divider", "seed"), f["roles"]):
                role_colors[role].add(color)
            role_colors["support"].update(color for _, color in f["stencil"])
            macro_target_counts.setdefault(f["shape"][:2], set()).add(f["targets"])
            if f["geometry"] == "asymmetric":
                asymmetric_offsets.add(tuple(offset for offset, _ in f["stencil"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42
    assert len(set(x["shape"] for x in observations)) >= 25
    assert len(asymmetric_offsets) >= 20
    assert all(len(colors) == 10 for colors in role_colors.values())
    assert sum(len(counts) > 1 for counts in macro_target_counts.values()) >= 10
    assert {x["geometry"] for x in observations} == {"plus", "ring", "asymmetric"}
    assert {x["colors"] for x in observations} >= {1, 2, 3}
    assert {x["targets"] for x in observations} == {1, 2, 3, 4}
    assert any(x["all_interior"] for x in observations)
    assert any(x["all_clipped"] for x in observations)
    assert any(x["clipped"] and not x["all_clipped"] for x in observations)
    assert any(x["overlap"] for x in observations)
    assert any(not x["overlap"] for x in observations)
    assert any(x["roles"][0] == 0 for x in observations)
    assert any(x["shape"][2] != x["shape"][3] for x in observations)
    assert any(x["shape"][0] != x["shape"][1] for x in observations)
    assert any(
        x["geometry"] == "asymmetric" and x["colors"] > 1
        and x["shape"][2] != x["shape"][3] and x["clipped"]
        for x in observations
    )
    assert any(
        x["colors"] > 1 and (
            len({dict(x["stencil"])[offset] for offset in ORTHOGONAL
                 if offset in dict(x["stencil"])}) > 1
            or len({dict(x["stencil"])[offset] for offset in DIAGONAL
                    if offset in dict(x["stencil"])}) > 1
        )
        for x in observations
    )
    for section in ("train", "test"):
        assert {x["geometry"] for x in split_features[section]} == {
            "plus", "ring", "asymmetric"
        }
        assert {x["colors"] == 1 for x in split_features[section]} == {True, False}
    assert all(value > 0 for value in within.values())
    assert _equal(generate_puzzle(314159, 10, 4), generate_puzzle(314159, 10, 4))
    assert not _equal(generate_puzzle(0), generate_puzzle(1))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(300 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(500 + (train or 0) + (test or 0), train, test))
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": True},
        {"num_test": 2.0},
    ):
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid counts: {arguments}")
