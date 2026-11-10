"""Seed-controlled corner reflections for ARC grids."""

import hashlib

import numpy as np


_QUADRANTS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SOURCE_HASHES = {
    ((20, 20), "cde8bd7f0e06a02ebda44d46d41e535c8e115bde9c2e5f9830c34d7c3cf35327"),
    ((30, 30), "151d8d808f4639bc5da6e2a6c9b5b8f539e2cac6811dffa85e09173e176cd288"),
    ((14, 14), "c77090a68bc2b1445aaa0f8f31ea48ee21b822e0a98416691177b18a3468fe6c"),
    ((24, 24), "74ad52e95cfdcf5d005614dd7bb2b9793c6145b3acbe2ae365e3ad6e4aa1cdd8"),
}


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _corner(jr, jc, quadrant):
    sy, sx = quadrant
    return jr - 1 if sy < 0 else jr, jc - 1 if sx < 0 else jc


def _cell(jr, jc, quadrant, dr, dc):
    sy, sx = quadrant
    return (jr - 1 - dr if sy < 0 else jr + dr,
            jc - 1 - dc if sx < 0 else jc + dc)


def _path(cells, end_row, end_col, rng):
    row = col = 0
    while row < end_row or col < end_col:
        moves = []
        if row < end_row:
            moves.append((1, 0))
        if col < end_col:
            moves.append((0, 1))
        if row < end_row and col < end_col:
            moves.extend(((1, 1), (1, 1)))
        dr, dc = moves[int(rng.integers(len(moves)))]
        row += dr
        col += dc
        cells.add((row, col))


def _motif(rng, height, width):
    # Both extrema and the junction cell are present, but the two cells
    # immediately along the junction axes are deliberately optional.
    cells = {(0, 0)}
    _path(cells, height - 1, int(rng.integers(width)), rng)
    _path(cells, int(rng.integers(height)), width - 1, rng)
    band = int(rng.integers(3))
    low, high = ((0.25, 0.45), (0.40, 0.68), (0.65, 0.91))[band]
    target = max(3, len(cells), int(round(float(rng.uniform(low, high)) * height * width)))
    target = min(target, height * width)
    while len(cells) < target:
        frontier = set()
        for row, col in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (row + dr, col + dc)
                    if (0 <= neighbor[0] < height and 0 <= neighbor[1] < width
                            and neighbor not in cells):
                        frontier.add(neighbor)
        choices = sorted(frontier)
        cells.add(choices[int(rng.integers(len(choices)))])
    return tuple(sorted(cells))


def _seed_quadrants(source, rng, regime, force_full=False):
    sy, sx = source
    other = ((sy, -sx), (-sy, sx), (-sy, -sx))
    if force_full:
        mask = 7
    else:
        weights = ((0.11, 0.11, 0.11, 0.16, 0.12, 0.12, 0.27)
                   if regime == "source_train" else
                   (0.09, 0.09, 0.13, 0.21, 0.12, 0.12, 0.24))
        mask = int(rng.choice(np.arange(1, 8), p=weights))
    return tuple(other[index] for index in range(3) if mask & (1 << index))


def _group_specs(rng, group_count, regime, force_full):
    specs = []
    signatures = set()
    for index in range(group_count):
        side_max = 3 if group_count == 4 else 5
        for _ in range(50):
            mh = int(rng.integers(2, side_max + 1))
            mw = int(rng.integers(2, side_max + 1))
            motif = _motif(rng, mh, mw)
            signature = (mh, mw, motif)
            if signature not in signatures:
                signatures.add(signature)
                break
        else:
            raise RuntimeError("motif sampling exhausted")
        source = _QUADRANTS[int(rng.integers(4))]
        seeds = _seed_quadrants(source, rng, regime, force_full and index == 0)
        specs.append({
            "mh": mh, "mw": mw, "motif": motif, "source": source,
            "seeds": seeds,
        })
    return specs


def _separated(a, b):
    at, ab, al, ar = a
    bt, bb, bl, br = b
    return ab + 1 < bt or bb + 1 < at or ar + 1 < bl or br + 1 < al


def _place(rng, specs):
    sizes = [(2 * spec["mh"], 2 * spec["mw"]) for spec in specs]
    area = sum(h * w for h, w in sizes)
    scale = int(np.ceil(np.sqrt(area) * float(rng.uniform(0.95, 1.30))))
    min_h = max(2 * max(spec["mh"] for spec in specs), min(scale, 30), 8)
    min_w = max(2 * max(spec["mw"] for spec in specs), min(scale, 30), 8)
    order = sorted(range(len(specs)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for attempt in range(36):
        height = int(rng.integers(min_h, 31)) if attempt < 28 else 30
        width = int(rng.integers(min_w, 31)) if attempt < 28 else 30
        rectangles = {}
        for index in order:
            gh, gw = sizes[index]
            locations = [(top, left)
                         for top in range(height - gh + 1)
                         for left in range(width - gw + 1)]
            for choice in rng.permutation(len(locations)):
                top, left = locations[int(choice)]
                rect = (top, top + gh - 1, left, left + gw - 1)
                if all(_separated(rect, prior) for prior in rectangles.values()):
                    rectangles[index] = rect
                    break
            else:
                break
        if len(rectangles) == len(specs):
            return height, width, rectangles
    raise RuntimeError("could not place separated reflection groups")


def _render_pair(rng, specs):
    height, width, rectangles = _place(rng, specs)
    background = int(rng.integers(10))
    grid = np.full((height, width), background, dtype=np.int8)
    output = grid.copy()
    available = [color for color in range(10) if color != background]
    for index, spec in enumerate(specs):
        top, _, left, _ = rectangles[index]
        jr = top + spec["mh"]
        jc = left + spec["mw"]
        colors = [int(color) for color in
                  rng.choice(available, size=1 + len(spec["seeds"]), replace=False)]
        for dr, dc in spec["motif"]:
            row, col = _cell(jr, jc, spec["source"], dr, dc)
            assert grid[row, col] == background
            grid[row, col] = output[row, col] = colors[0]
        for quadrant, color in zip(spec["seeds"], colors[1:]):
            row, col = _corner(jr, jc, quadrant)
            assert grid[row, col] == background
            grid[row, col] = color
            for dr, dc in spec["motif"]:
                row, col = _cell(jr, jc, quadrant, dr, dc)
                assert output[row, col] in (background, color)
                output[row, col] = color
    return {"input": grid, "output": output}


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_hash(pair):
    data = (pair["input"].astype(np.int8, copy=False).tobytes()
            + pair["output"].astype(np.int8, copy=False).tobytes())
    return pair["input"].shape, hashlib.sha256(data).hexdigest()


def _background(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > grid.size // 2, "background is not identifiable"
    return background


def _components(grid, background):
    occupied = set(map(tuple, np.argwhere(grid != background)))
    components = []
    while occupied:
        seed = occupied.pop()
        component = {seed}
        stack = [seed]
        while stack:
            row, col = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = row + dr, col + dc
                    if neighbor in occupied:
                        occupied.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        components.append(component)
    return components


def _connected(cells):
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        row, col = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = row + dr, col + dc
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return not remaining


def _parse_group(grid, component, background):
    by_color = {}
    for row, col in component:
        by_color.setdefault(int(grid[row, col]), []).append((row, col))
    sources = [color for color, cells in by_color.items() if len(cells) >= 3]
    assert len(sources) == 1, "one substantial source is required"
    source_color = sources[0]
    source_cells = by_color[source_color]
    assert _connected(source_cells)
    assert all(len(cells) == 1 for color, cells in by_color.items() if color != source_color)
    assert 2 <= len(by_color) <= 4
    rows = [row for row, _ in source_cells]
    cols = [col for _, col in source_cells]
    candidates = []
    for source in _QUADRANTS:
        sy, sx = source
        jr = max(rows) + 1 if sy < 0 else min(rows)
        jc = max(cols) + 1 if sx < 0 else min(cols)
        if _corner(jr, jc, source) not in source_cells:
            continue
        motif = tuple(sorted((jr - 1 - row if sy < 0 else row - jr,
                              jc - 1 - col if sx < 0 else col - jc)
                             for row, col in source_cells))
        if any(dr < 0 or dc < 0 for dr, dc in motif):
            continue
        mh = max(dr for dr, _ in motif) + 1
        mw = max(dc for _, dc in motif) + 1
        if not (2 <= mh <= 5 and 2 <= mw <= 5):
            continue
        rect = (jr - mh, jr + mh - 1, jc - mw, jc + mw - 1)
        if not (0 <= rect[0] <= rect[1] < grid.shape[0]
                and 0 <= rect[2] <= rect[3] < grid.shape[1]):
            continue
        seeds = {}
        for color, cells in by_color.items():
            if color == source_color:
                continue
            positions = [q for q in _QUADRANTS if _corner(jr, jc, q) == cells[0]]
            if len(positions) != 1 or positions[0] == source:
                break
            seeds[positions[0]] = color
        else:
            if len(seeds) != len(by_color) - 1:
                continue
            if any(int(grid[_corner(jr, jc, q)]) !=
                   (source_color if q == source else seeds.get(q, background))
                   for q in _QUADRANTS):
                continue
            candidates.append({
                "source": source, "source_color": source_color,
                "junction": (jr, jc), "motif": motif, "size": (mh, mw),
                "seeds": seeds, "rect": rect,
            })
    assert len(candidates) == 1, "source and junction must be unambiguous"
    return candidates[0]


def _analyze_pair(pair):
    grid, output = pair["input"], pair["output"]
    assert grid.shape == output.shape
    background = _background(grid)
    components = _components(grid, background)
    assert 1 <= len(components) <= 4
    groups = [_parse_group(grid, component, background) for component in components]
    for i, first in enumerate(groups):
        for second in groups[i + 1:]:
            assert _separated(first["rect"], second["rect"])
    expected = grid.copy()
    for group in groups:
        jr, jc = group["junction"]
        for quadrant, color in group["seeds"].items():
            for dr, dc in group["motif"]:
                row, col = _cell(jr, jc, quadrant, dr, dc)
                assert expected[row, col] in (background, color)
                expected[row, col] = color
    assert not np.array_equal(grid, output)
    assert np.array_equal(expected, output), "reflected copies or colors differ"
    return background, groups


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_groups = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            key = _grid_key(pair["input"])
            pair_key = key, _grid_key(pair["output"])
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert _source_hash(pair) not in _SOURCE_HASHES
            _, groups = _analyze_pair(pair)
            if section == "train":
                train_groups.extend(groups)
    # One completed four-quadrant group demonstrates the side and diagonal
    # maps. All other colors are local instructions at the same junction.
    assert any(len(group["seeds"]) == 3 for group in train_groups)


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
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count, regime in (("train", train_count, "source_train"),
                                   ("test", test_count, "target_test")):
        for pair_index in range(count):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                group_count = int(pair_rng.choice(
                    np.arange(1, 5), p=(0.24, 0.36, 0.30, 0.10)))
                specs = _group_specs(pair_rng, group_count, regime,
                                     section == "train" and pair_index == 0)
                pair = _render_pair(pair_rng, specs)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(np.array_equal(x[part], y[part])
               for section in ("train", "test")
               for x, y in zip(a[section], b[section])
               for part in ("input", "output"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    backgrounds, sizes, motifs, patterns, orientations = set(), set(), set(), set(), set()
    group_counts, margins, layouts, palette_roles = set(), set(), set(), set()
    sparse = elbow_free = diagonal_only = 0
    all_puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        all_puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                bg, groups = _analyze_pair(pair)
                backgrounds.add(bg)
                sizes.add(grid.shape)
                group_counts.add(len(groups))
                layouts.add((grid.shape, tuple(sorted(group["rect"] for group in groups))))
                source_colors = {group["source_color"] for group in groups}
                seed_colors = {color for group in groups for color in group["seeds"].values()}
                if source_colors & seed_colors:
                    palette_roles.add("reused")
                if not source_colors & seed_colors and len(groups) > 1:
                    palette_roles.add("independent")
                for group in groups:
                    mh, mw = group["size"]
                    motif = set(group["motif"])
                    motifs.add((mh, mw, group["motif"]))
                    orientations.add(group["source"])
                    sy, sx = group["source"]
                    mask = sum(1 << i for i, q in enumerate(((sy, -sx), (-sy, sx), (-sy, -sx)))
                               if q in group["seeds"])
                    patterns.add(mask)
                    sparse += len(motif) * 2 <= mh * mw
                    elbow_free += not {(1, 0), (0, 1)}.issubset(motif)
                    diagonal_only += mask == 4
                output = pair["output"]
                margins.add(any(np.any(edge != bg) for edge in
                                (output[0], output[-1], output[:, 0], output[:, -1])))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 5 and len(sizes) >= 20
    assert len(motifs) >= 80 and len(layouts) >= 100
    assert group_counts == {1, 2, 3, 4}
    assert orientations == set(_QUADRANTS) and patterns == set(range(1, 8))
    assert margins == {False, True} and palette_roles == {"reused", "independent"}
    assert sparse >= 10 and elbow_free >= 50 and diagonal_only >= 10
    assert not _same_puzzle(all_puzzles[0], all_puzzles[1])
    assert _same_puzzle(generate_puzzle(7319, 7, 3), generate_puzzle(7319, 7, 3))
    for seed, nt, ns in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1101, None, 1), (1102, None, 4),
                         (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1201, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count did not raise ValueError")
