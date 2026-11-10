import hashlib

import numpy as np


# SHA-256 of the four original (input bytes || output bytes) pairs.  Exact
# content is excluded; their dimensions, palette and motifs remain legal.
_SOURCE_PAIR_DIGESTS = {
    "13f08223ed6fd85d4416c47f59c11908c90e8e0d184a1cb4676052eae5f780a1",
    "ec72a273bb5ecd3e38b80d4f50c64c30477f36090c77fe7ecaaf21cfbe320c64",
    "184681ae76cab211fbbafd03bbe541093e834c68943a415702a29e12bca633e5",
    "6d456bb8fee5af5ef3d6e80d0cb1624c747f1bb16b39e5fd7d374e612112cd5a",
}


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _frame_components(grid, background, structure):
    mask = grid == structure
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = grid.shape
    frames = []

    for start_r in range(height):
        for start_c in range(width):
            if not mask[start_r, start_c] or seen[start_r, start_c]:
                continue

            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)

            assert r1 - r0 >= 2
            assert c1 - c0 >= 2

            perimeter = set()
            for c in range(c0, c1 + 1):
                perimeter.add((r0, c))
                perimeter.add((r1, c))
            for r in range(r0, r1 + 1):
                perimeter.add((r, c0))
                perimeter.add((r, c1))

            assert set(cells) == perimeter
            assert np.all(
                grid[r0 + 1 : r1, c0 + 1 : c1] == background
            )

            frames.append((r0, r1, c0, c1))

    return sorted(frames)


def _analyze_input(grid):
    values = np.unique(grid)
    assert len(values) == 2
    candidates = []
    for background, structure in (
        (int(values[0]), int(values[1])),
        (int(values[1]), int(values[0])),
    ):
        try:
            frames = _frame_components(grid, background, structure)
        except AssertionError:
            continue
        if len(frames) >= 3:
            candidates.append((background, structure, frames))
    assert len(candidates) == 1
    return candidates[0]


def _candidate_edges(grid, frames, background):
    edges = []

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            a, b = frames[i], frames[j]

            if a[3] < b[2] - 1 or b[3] < a[2] - 1:
                left, right = (a, b) if a[3] < b[2] else (b, a)
                lo = max(left[0] + 1, right[0] + 1)
                hi = min(left[1] - 1, right[1] - 1)
                gap = right[2] - left[3] - 1

                clear = (
                    gap >= 1
                    and lo <= hi
                    and np.all(
                        grid[
                            lo : hi + 1,
                            left[3] + 1 : right[2],
                        ]
                        == background
                    )
                )

                if clear:
                    edges.append((gap, i, j, "h", lo, hi))

            if a[1] < b[0] - 1 or b[1] < a[0] - 1:
                top, bottom = (a, b) if a[1] < b[0] else (b, a)
                lo = max(top[2] + 1, bottom[2] + 1)
                hi = min(top[3] - 1, bottom[3] - 1)
                gap = bottom[0] - top[1] - 1

                clear = (
                    gap >= 1
                    and lo <= hi
                    and np.all(
                        grid[
                            top[1] + 1 : bottom[0],
                            lo : hi + 1,
                        ]
                        == background
                    )
                )

                if clear:
                    edges.append((gap, i, j, "v", lo, hi))

    return sorted(
        edges,
        key=lambda edge: (
            edge[0],
            edge[1],
            edge[2],
            edge[3],
        ),
    )


def _minimum_spanning_tree(number_of_vertices, edges):
    parent = list(range(number_of_vertices))
    rank = [0] * number_of_vertices

    def find(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(a, b):
        a, b = find(a), find(b)

        if a == b:
            return False

        if rank[a] < rank[b]:
            a, b = b, a

        parent[b] = a

        if rank[a] == rank[b]:
            rank[a] += 1

        return True

    chosen = []

    for edge in edges:
        if union(edge[1], edge[2]):
            chosen.append(edge)

    if len(chosen) != number_of_vertices - 1:
        return None, False

    adjacency = [[] for _ in range(number_of_vertices)]
    chosen_keys = set()

    for edge in chosen:
        weight, a, b, orientation = edge[:4]
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))
        chosen_keys.add((a, b, orientation))

    unique = True

    for edge in edges:
        weight, source, target, orientation = edge[:4]

        if (source, target, orientation) in chosen_keys:
            continue

        stack = [(source, -1, 0)]
        path_maximum = None

        while stack:
            vertex, previous, maximum = stack.pop()

            if vertex == target:
                path_maximum = maximum
                break

            for neighbor, tree_weight in adjacency[vertex]:
                if neighbor != previous:
                    stack.append(
                        (
                            neighbor,
                            vertex,
                            max(maximum, tree_weight),
                        )
                    )

        assert path_maximum is not None
        assert weight >= path_maximum

        if weight == path_maximum:
            unique = False

    return chosen, unique


def _gap_cells(edge, frames):
    a, b = frames[edge[1]], frames[edge[2]]
    cells = set()

    if edge[3] == "h":
        left, right = (a, b) if a[3] < b[2] else (b, a)

        for r in range(edge[4], edge[5] + 1):
            for c in range(left[3] + 1, right[2]):
                cells.add((r, c))
    else:
        top, bottom = (a, b) if a[1] < b[0] else (b, a)

        for r in range(top[1] + 1, bottom[0]):
            for c in range(edge[4], edge[5] + 1):
                cells.add((r, c))

    return cells


def _corridors_do_not_cross_candidates(chosen, edges, frames):
    chosen_keys = {
        (edge[1], edge[2], edge[3])
        for edge in chosen
    }
    footprints = [
        _gap_cells(edge, frames)
        for edge in edges
    ]

    for i, edge in enumerate(edges):
        if (edge[1], edge[2], edge[3]) not in chosen_keys:
            continue

        for j in range(len(edges)):
            if (
                i != j
                and footprints[i].intersection(footprints[j])
            ):
                return False

    return True


def _render_corridors(
    grid,
    frames,
    chosen,
    background,
    structure,
):
    output = grid.copy()

    for edge in chosen:
        a, b = frames[edge[1]], frames[edge[2]]
        lo, hi = edge[4], edge[5]

        if edge[3] == "h":
            left, right = (a, b) if a[3] < b[2] else (b, a)

            output[
                lo : hi + 1,
                left[3] : right[2] + 1,
            ] = background
            output[
                lo,
                left[3] : right[2] + 1,
            ] = structure
            output[
                hi,
                left[3] : right[2] + 1,
            ] = structure
        else:
            top, bottom = (a, b) if a[1] < b[0] else (b, a)

            output[
                top[1] : bottom[0] + 1,
                lo : hi + 1,
            ] = background
            output[
                top[1] : bottom[0] + 1,
                lo,
            ] = structure
            output[
                top[1] : bottom[0] + 1,
                hi,
            ] = structure

    return output


def _paint_frames(frames):
    grid = np.zeros((30, 30), dtype=np.int8)
    for r0, r1, c0, c1 in frames:
        grid[r0, c0 : c1 + 1] = 1
        grid[r1, c0 : c1 + 1] = 1
        grid[r0 : r1 + 1, c0] = 1
        grid[r0 : r1 + 1, c1] = 1
    return grid


def _grow_frames(rng, number_of_frames):
    """Grow a spatial graph by placing each room beside an earlier room."""
    cap = 11 if number_of_frames <= 5 else 9 if number_of_frames <= 7 else 7

    def side():
        # Independent heights and widths create both long narrow and broad rooms.
        if rng.random() < 0.16:
            return int(rng.integers(8, min(12, cap + 2) + 1))
        return int(rng.integers(3, cap + 1))

    height, width = side(), side()
    r0 = int(rng.integers(4, 27 - height))
    c0 = int(rng.integers(4, 27 - width))
    frames = [(r0, r0 + height - 1, c0, c0 + width - 1)]

    for _ in range(1, number_of_frames):
        placed = False
        for _ in range(240):
            parent = frames[int(rng.integers(len(frames)))]
            height, width = side(), side()
            gap = int(rng.choice((1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8)))
            direction = int(rng.integers(4))

            if direction < 2:
                low = max(0, parent[0] - height + 3)
                high = min(30 - height, parent[1] - 2)
                if low > high:
                    continue
                r0 = int(rng.integers(low, high + 1))
                r1 = r0 + height - 1
                if direction == 0:
                    c0 = parent[3] + gap + 1
                    c1 = c0 + width - 1
                else:
                    c1 = parent[2] - gap - 1
                    c0 = c1 - width + 1
            else:
                low = max(0, parent[2] - width + 3)
                high = min(30 - width, parent[3] - 2)
                if low > high:
                    continue
                c0 = int(rng.integers(low, high + 1))
                c1 = c0 + width - 1
                if direction == 2:
                    r0 = parent[1] + gap + 1
                    r1 = r0 + height - 1
                else:
                    r1 = parent[0] - gap - 1
                    r0 = r1 - height + 1

            if min(r0, c0) < 0 or max(r1, c1) >= 30:
                continue
            if any(
                r0 <= b[1] + 1 and r1 >= b[0] - 1
                and c0 <= b[3] + 1 and c1 >= b[2] - 1
                for b in frames
            ):
                continue
            frames.append((r0, r1, c0, c1))
            placed = True
            break
        if not placed:
            return None

    grid = _paint_frames(frames)
    if rng.random() >= 0.14:
        top = min(frame[0] for frame in frames)
        bottom = max(frame[1] for frame in frames)
        left = min(frame[2] for frame in frames)
        right = max(frame[3] for frame in frames)
        top = max(0, top - int(rng.integers(4)))
        bottom = min(29, bottom + int(rng.integers(4)))
        left = max(0, left - int(rng.integers(4)))
        right = min(29, right + int(rng.integers(4)))
        grid = grid[top : bottom + 1, left : right + 1]

    grid = np.rot90(grid, int(rng.integers(4)))
    if rng.random() < 0.5:
        grid = np.fliplr(grid)
    return grid.copy()


def _pair_digest(input_grid, output_grid):
    if input_grid.shape != (30, 30):
        return None
    return hashlib.sha256(
        input_grid.astype(np.int8).tobytes()
        + output_grid.astype(np.int8).tobytes()
    ).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.astype(np.int16, copy=False).tobytes()


def _inspect_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    background, structure, frames = _analyze_input(input_grid)
    assert 3 <= len(frames) <= 9
    assert set(map(int, np.unique(output_grid))) == {background, structure}

    edges = _candidate_edges(input_grid, frames, background)
    chosen, unique = _minimum_spanning_tree(len(frames), edges)
    assert chosen is not None and unique
    assert _corridors_do_not_cross_candidates(chosen, edges, frames)
    expected = _render_corridors(
        input_grid, frames, chosen, background, structure
    )
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    assert _pair_digest(input_grid, output_grid) not in _SOURCE_PAIR_DIGESTS

    degrees = [0] * len(frames)
    for edge in chosen:
        degrees[edge[1]] += 1
        degrees[edge[2]] += 1
    return {
        "palette": (background, structure),
        "frames": frames,
        "edges": edges,
        "chosen": chosen,
        "degrees": degrees,
    }


def _generate_pair(rng, regime, palette, target):
    for _ in range(700):
        if regime == "source_train":
            number_of_frames = int(rng.integers(5 if target == "mixed" else 4, 9))
        else:
            number_of_frames = int(rng.integers(3, 10))
        stencil = _grow_frames(rng, number_of_frames)
        if stencil is None:
            continue
        frames = _frame_components(stencil, 0, 1)
        if len(frames) != number_of_frames:
            continue
        edges = _candidate_edges(stencil, frames, 0)
        chosen, unique = _minimum_spanning_tree(number_of_frames, edges)
        if chosen is None or not unique:
            continue
        if not _corridors_do_not_cross_candidates(chosen, edges, frames):
            continue

        widths = [edge[5] - edge[4] + 1 for edge in chosen]
        surplus = len(edges) - number_of_frames + 1
        if target == "mixed" and (
            {edge[3] for edge in chosen} != {"h", "v"}
            or min(widths) > 2 or max(widths) < 3
        ):
            continue
        if target == "cycle" and surplus < 1:
            continue
        if target == "tree" and surplus != 0:
            continue
        if target == "hard" and not any(
            edge not in chosen and edge[0] < max(c[0] for c in chosen)
            for edge in edges
        ):
            continue

        output = _render_corridors(stencil, frames, chosen, 0, 1)
        if np.array_equal(stencil, output):
            continue
        background, structure = palette
        recolored_input = np.where(stencil == 0, background, structure).astype(np.int8)
        recolored_output = np.where(output == 0, background, structure).astype(np.int8)
        if _pair_digest(recolored_input, recolored_output) in _SOURCE_PAIR_DIGESTS:
            continue
        return recolored_input, recolored_output
    raise ValueError("could not construct an unambiguous pair")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys, pair_keys = set(), set()
    train_infos, test_infos = [], []
    for regime, infos in (("train", train_infos), ("test", test_infos)):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in input_keys and pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            infos.append(_inspect_pair(pair["input"], pair["output"]))

    # The first demonstration shows both directions and both passage types;
    # another demonstrates rejection of an otherwise valid cycle-closing link.
    train_edges = [edge for info in train_infos for edge in info["chosen"]]
    assert {edge[3] for edge in train_edges} == {"h", "v"}
    widths = [edge[5] - edge[4] + 1 for edge in train_edges]
    assert min(widths) <= 2 and max(widths) >= 3
    assert any(len(info["edges"]) > len(info["frames"]) - 1 for info in train_infos)
    # The palette is read from each input's frame/background roles.  A changed
    # test palette therefore introduces no hidden mapping to learn.
    assert all(info["palette"][0] != info["palette"][1]
               for info in train_infos + test_infos)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    variable_palette = bool(rng.integers(2))
    base_palette = tuple(map(int, rng.choice(10, size=2, replace=False)))
    puzzle = {"train": [], "test": []}
    seen_inputs = set()

    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if not variable_palette or (regime == "train" and index == 0):
                palette = base_palette
            else:
                palette = tuple(map(int, rng.choice(10, size=2, replace=False)))
                if regime == "train" and index == 1:
                    while palette == base_palette:
                        palette = tuple(map(int, rng.choice(10, size=2, replace=False)))

            if regime == "train":
                target = "mixed" if index == 0 else "cycle" if index == 1 else (
                    rng.choice(("free", "tree", "cycle", "hard"), p=(0.4, 0.25, 0.25, 0.1))
                )
            else:
                target = rng.choice(
                    ("free", "tree", "cycle", "hard"),
                    p=(0.35, 0.3, 0.25, 0.1),
                )
            for _ in range(12):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                input_grid, output_grid = _generate_pair(
                    pair_rng, "source_train" if regime == "train" else "target_test",
                    palette, target,
                )
                key = _grid_key(input_grid)
                if key not in seen_inputs:
                    seen_inputs.add(key)
                    puzzle[regime].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise ValueError("could not generate distinct pairs")

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    if set(a) != set(b):
        return False
    return all(
        np.array_equal(x[k], y[k])
        for regime in ("train", "test")
        for x, y in zip(a[regime], b[regime])
        for k in ("input", "output")
    ) and all(len(a[r]) == len(b[r]) for r in ("train", "test"))


def _distractor_counts(grid, info):
    """Count apparent alignments that are ineligible for distinct reasons."""
    background = info["palette"][0]
    frames = info["frames"]
    near = blocked = 0
    for i, a in enumerate(frames):
        for b in frames[i + 1:]:
            if a[3] < b[2] - 1 or b[3] < a[2] - 1:
                left, right = (a, b) if a[3] < b[2] else (b, a)
                lo = max(left[0] + 1, right[0] + 1)
                hi = min(left[1] - 1, right[1] - 1)
                if max(left[0], right[0]) <= min(left[1], right[1]) and lo > hi:
                    near += 1
                elif lo <= hi and not np.all(
                    grid[lo:hi + 1, left[3] + 1:right[2]] == background
                ):
                    blocked += 1
            if a[1] < b[0] - 1 or b[1] < a[0] - 1:
                top, bottom = (a, b) if a[1] < b[0] else (b, a)
                lo = max(top[2] + 1, bottom[2] + 1)
                hi = min(top[3] - 1, bottom[3] - 1)
                if max(top[2], bottom[2]) <= min(top[3], bottom[3]) and lo > hi:
                    near += 1
                elif lo <= hi and not np.all(
                    grid[top[1] + 1:bottom[0], lo:hi + 1] == background
                ):
                    blocked += 1
    return near, blocked


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(314159), generate_puzzle(314159))
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                            (None, 1), (None, 4),
                                            (2, None), (10, None), (5, 2))):
        puzzle = generate_puzzle(1000 + offset, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": False, "num_test": 1}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")

    train_counts, test_counts, palettes, shapes, layouts = set(), set(), set(), set(), set()
    all_inputs, puzzle_keys = set(), set()
    side_lengths, overlaps, gaps, surpluses, frame_counts = set(), set(), set(), set(), set()
    varied_puzzles = simpler_tests = larger_tests = tree_pairs = cycle_pairs = 0
    near_pairs = blocked_pairs = hard_pairs = long_narrow_pairs = 0
    combination_pairs = full_size_pairs = small_pairs = chains = forks = 0
    all_solid_pairs = all_hollow_pairs = 0
    varied_frames = varied_graphs = varied_shapes = varied_widths = 0
    larger_test_graphs = smaller_test_graphs = 0
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(tuple(
            (_grid_key(p["input"]), _grid_key(p["output"]))
            for regime in ("train", "test") for p in puzzle[regime]
        ))
        infos = []
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                total_pairs += 1
                info = _inspect_pair(pair["input"], pair["output"])
                infos.append((regime, info))
                all_inputs.add(_grid_key(pair["input"]))
                palettes.add(info["palette"])
                shapes.add(pair["input"].shape)
                frame_counts.add(len(info["frames"]))
                frame_geometry = tuple(sorted((r1-r0+1, c1-c0+1)
                                              for r0,r1,c0,c1 in info["frames"]))
                graph_geometry = tuple(sorted((e[3],e[0],e[5]-e[4]+1)
                                              for e in info["edges"]))
                layouts.add((len(info["frames"]), frame_geometry, graph_geometry))
                side_lengths.update(side for dims in frame_geometry for side in dims)
                overlaps.update(e[5]-e[4]+1 for e in info["chosen"])
                gaps.update(e[0] for e in info["chosen"])
                surplus = len(info["edges"])-len(info["frames"])+1
                surpluses.add(surplus)
                tree_pairs += surplus == 0
                cycle_pairs += surplus > 0
                near, blocked = _distractor_counts(pair["input"], info)
                near_pairs += near > 0
                blocked_pairs += blocked > 0
                hard_pairs += any(
                    e not in info["chosen"] and e[0] < max(c[0] for c in info["chosen"])
                    for e in info["edges"]
                )
                long_narrow = any(max(dims) >= 10 and min(dims) <= 5
                                  for dims in frame_geometry)
                long_narrow_pairs += long_narrow
                combination_pairs += surplus > 0 and long_narrow and any(
                    e[0] == 1 for e in info["chosen"]
                )
                full_size_pairs += pair["input"].shape == (30, 30)
                small_pairs += min(pair["input"].shape) <= 20
                chains += max(info["degrees"]) == 2
                forks += max(info["degrees"]) >= 3
                all_solid_pairs += max(e[5]-e[4]+1 for e in info["chosen"]) <= 2
                all_hollow_pairs += min(e[5]-e[4]+1 for e in info["chosen"]) >= 3
        varied_puzzles += len({x[1]["palette"] for x in infos}) > 1
        varied_frames += len({len(x[1]["frames"]) for x in infos}) > 1
        varied_graphs += len({len(x[1]["edges"])-len(x[1]["frames"])+1
                              for x in infos}) > 1
        varied_shapes += len({pair["input"].shape for regime in ("train", "test")
                              for pair in puzzle[regime]}) > 1
        varied_widths += len({max(e[5]-e[4]+1 for e in x[1]["chosen"])
                              for x in infos}) > 1
        train_max = max(len(x[1]["frames"]) for x in infos if x[0] == "train")
        simpler_tests += any(len(x[1]["frames"]) <= train_max for x in infos if x[0] == "test")
        larger_tests += any(len(x[1]["frames"]) > train_max for x in infos if x[0] == "test")
        train_surpluses = [len(x[1]["edges"])-len(x[1]["frames"])+1
                          for x in infos if x[0] == "train"]
        test_surpluses = [len(x[1]["edges"])-len(x[1]["frames"])+1
                         for x in infos if x[0] == "test"]
        larger_test_graphs += max(test_surpluses) > max(train_surpluses)
        smaller_test_graphs += min(test_surpluses) < min(train_surpluses)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42 and len(all_inputs) >= int(0.98 * total_pairs)
    assert len(palettes) >= 15 and len(shapes) >= 25 and len(layouts) >= 100
    assert min(side_lengths) == 3 and max(side_lengths) >= 11
    assert 1 in overlaps and max(overlaps) >= 6 and 1 in gaps
    assert 0 in surpluses and max(surpluses) >= 2
    assert len(frame_counts) >= 6 and tree_pairs >= 20 and cycle_pairs >= 20
    assert varied_puzzles >= 12 and simpler_tests >= 12
    assert min(varied_frames, varied_graphs, varied_shapes, varied_widths) >= 30
    assert larger_test_graphs >= 1 and smaller_test_graphs >= 1
    assert larger_tests >= 1 and full_size_pairs >= 10 and small_pairs >= 10
    assert near_pairs >= 30 and blocked_pairs >= 30 and hard_pairs >= 10
    assert long_narrow_pairs >= 20 and combination_pairs >= 10
    assert chains >= 20 and forks >= 20
    assert all_solid_pairs >= 20 and all_hollow_pairs >= 1
