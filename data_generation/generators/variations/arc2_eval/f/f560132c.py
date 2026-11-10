"""Four scattered pieces, one visible corner-color key, one unique square.

Construction and validation are separate: accepted cuts make candidate outputs;
validation segments the returned input and exhaustively packs its footprints.
All learned conventions are fixed by this family, while presentation varies.
"""
import hashlib
import heapq
import itertools
import numpy as np

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))

_SOURCE_DIGESTS = frozenset(('fe4935c3afe539756f4ebf8c0e6e58a321609b3f1acff6b85fd33541fb342632', '7d022919bff3171442e913ece12656ae6810359ff6656f153adfd671b07ce95a', 'df7d35e0d63f08b2053344570f21023926285b20a12b5d7a6c5e2f977892cc24'))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_integer_argument(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _validate_count_argument(name: str, value, low: int, high: int) -> None:
    if value is None:
        return
    if not _is_integer_argument(value):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count_argument("num_train", num_train, 2, 10)
    _validate_count_argument("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        train_count = int(num_train)
        test_count = int(num_test)
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if num_train is None and num_test is None:
        test_count = 1 + seed % 4
        low_train = max(2, test_count + 1)
        train_count = low_train + (seed // 4) % (11 - low_train)
        return train_count, test_count

    if num_train is None:
        test_count = int(num_test)
        low_train = max(2, test_count + 1)
        train_count = low_train + seed % (11 - low_train)
        return train_count, test_count

    train_count = int(num_train)
    max_test = min(4, train_count - 1)
    test_count = 1 + seed % max_test
    return train_count, test_count


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    array = np.asarray(coords, dtype=np.int16)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 2:
        raise ValueError("a shape must contain coordinate pairs")
    array = array - array.min(axis=0)
    return tuple(
        sorted((int(row), int(col)) for row, col in array.tolist())
    )


def _rotate_shape(
    shape: tuple[tuple[int, int], ...],
    quarter_turns: int,
) -> tuple[tuple[int, int], ...]:
    result = np.asarray(shape, dtype=np.int16)

    for _ in range(quarter_turns % 4):
        height = int(result[:, 0].max()) + 1
        result = np.column_stack(
            (result[:, 1], height - 1 - result[:, 0])
        )
        result = result - result.min(axis=0)

    return tuple(
        sorted((int(row), int(col)) for row, col in result.tolist())
    )


def _shape_dimensions(
    shape: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    return (
        max(row for row, _ in shape) + 1,
        max(col for _, col in shape) + 1,
    )


def _shape_mask(
    shape: tuple[tuple[int, int], ...],
    side: int,
    row_offset: int = 0,
    col_offset: int = 0,
) -> int:
    mask = 0
    for row, col in shape:
        rr = row + row_offset
        cc = col + col_offset
        if not (0 <= rr < side and 0 <= cc < side):
            raise ValueError("shape lies outside the square")
        mask |= 1 << (rr * side + cc)
    return mask


def _coordinate_mask(coords, side: int) -> int:
    mask = 0
    for row, col in np.asarray(coords, dtype=np.int16):
        mask |= 1 << (int(row) * side + int(col))
    return mask


def _corner_candidates(shape, side, corner):
    # Occupying an extreme corner fixes the translation of a bounding box.
    candidates = {}
    corner_indices = (0, side - 1, side * (side - 1), side * side - 1)
    for rotation in range(4):
        rotated = _rotate_shape(shape, rotation)
        height, width = _shape_dimensions(rotated)
        if height > side or width > side:
            continue
        row = 0 if corner < 2 else side - height
        col = 0 if corner % 2 == 0 else side - width
        mask = _shape_mask(rotated, side, row, col)
        hits = [bool(mask & (1 << i)) for i in corner_indices]
        if sum(hits) == 1 and hits[corner]:
            # First rotation is the smallest actual demand, including symmetry.
            candidates.setdefault(mask, rotation)
    return list(candidates.items())


def _solve_square(
    shapes: list[tuple[tuple[int, int], ...]],
    side: int,
    limit: int = 2,
) -> list[dict]:
    if len(shapes) != 4:
        return []
    if sum(len(shape) for shape in shapes) != side * side:
        return []
    if (0, 0) not in shapes[0]:
        return []

    anchor_height, anchor_width = _shape_dimensions(shapes[0])
    if anchor_height > side or anchor_width > side:
        return []

    anchor_mask = _shape_mask(shapes[0], side)
    if sum(bool(anchor_mask & (1 << i)) for i in (0, side-1, side*(side-1), side*side-1)) != 1:
        return []
    full_mask = (1 << (side * side)) - 1

    candidate_table = {
        (piece, corner): _corner_candidates(
            shapes[piece],
            side,
            corner,
        )
        for piece in range(1, 4)
        for corner in range(1, 4)
    }

    solutions: list[dict] = []
    seen_colored_squares = {}
    masks: list[int | None] = [anchor_mask, None, None, None]
    piece_for_corner: list[int | None] = [0, None, None, None]
    rotations: list[int | None] = [0, None, None, None]

    def search(
        corner: int,
        remaining_pieces: tuple[int, ...],
        occupied: int,
    ) -> None:
        if len(solutions) >= limit:
            return

        if corner == 4:
            if occupied == full_mask:
                answer = {
                    'masks': tuple(masks),
                    'piece_for_corner': tuple(piece_for_corner),
                    'rotations': tuple(rotations),
                }
                previous = seen_colored_squares.get(tuple(masks))
                if previous is None:
                    seen_colored_squares[tuple(masks)] = len(solutions)
                    solutions.append(answer)
                else:
                    # Congruent pieces may be interchangeable without changing
                    # the colored answer. Measure the least rotation needed.
                    score = lambda a: (sum(k != 0 for k in a['rotations']), sum(a['rotations']), a['rotations'])
                    if score(answer) < score(solutions[previous]):
                        solutions[previous] = answer
            return

        for piece in remaining_pieces:
            for candidate_mask, rotation in candidate_table[
                (piece, corner)
            ]:
                if occupied & candidate_mask:
                    continue

                masks[corner] = candidate_mask
                piece_for_corner[corner] = piece
                rotations[corner] = rotation
                next_remaining = tuple(
                    item for item in remaining_pieces if item != piece
                )
                search(
                    corner + 1,
                    next_remaining,
                    occupied | candidate_mask,
                )

                if len(solutions) >= limit:
                    return

    search(1, (1, 2, 3), anchor_mask)
    return solutions


def _weighted_corner_partition(
    rng: np.random.Generator,
    side: int,
    roughness: float,
) -> np.ndarray:
    weights = 0.8 + roughness * rng.random((side, side))
    sources = (
        (0, 0),
        (0, side - 1),
        (side - 1, 0),
        (side - 1, side - 1),
    )
    all_distances = []

    for source_row, source_col in sources:
        distances = np.full((side, side), np.inf, dtype=np.float64)
        distances[source_row, source_col] = 0.0
        heap = [(0.0, source_row, source_col)]

        while heap:
            distance, row, col = heapq.heappop(heap)
            if distance > distances[row, col]:
                continue

            for dr, dc in _NEIGHBORS:
                rr = row + dr
                cc = col + dc
                if not (0 <= rr < side and 0 <= cc < side):
                    continue

                edge_cost = 0.5 * (
                    weights[row, col] + weights[rr, cc]
                )
                candidate = distance + edge_cost

                if candidate < distances[rr, cc]:
                    distances[rr, cc] = candidate
                    heapq.heappush(heap, (candidate, rr, cc))

        all_distances.append(distances)

    return np.argmin(
        np.stack(all_distances, axis=0),
        axis=0,
    ).astype(np.int8)


def _key_block_locations(owner: np.ndarray) -> list[tuple[int, int]]:
    side = owner.shape[0]
    locations = []

    for row in range(side - 1):
        for col in range(side - 1):
            if np.all(owner[row : row + 2, col : col + 2] == 0):
                locations.append((row, col))

    return locations


def _is_irregular(
    shape: tuple[tuple[int, int], ...],
) -> bool:
    height, width = _shape_dimensions(shape)
    return len(shape) < height * width


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _normalized_layout_key(
    output: np.ndarray,
    key: np.ndarray,
) -> tuple:
    normalized = np.zeros(output.shape, dtype=np.int8)

    for corner, color in enumerate(key.reshape(-1), start=1):
        normalized[output == int(color)] = corner

    return _grid_key(normalized)


def _connected_components(
    nonbackground: np.ndarray,
) -> list[np.ndarray]:
    height, width = nonbackground.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for start_row in range(height):
        for start_col in range(width):
            if (
                not nonbackground[start_row, start_col]
                or visited[start_row, start_col]
            ):
                continue

            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            cells = []

            while stack:
                row, col = stack.pop()
                cells.append((row, col))

                for dr, dc in _NEIGHBORS:
                    rr = row + dr
                    cc = col + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if (
                        nonbackground[rr, cc]
                        and not visited[rr, cc]
                    ):
                        visited[rr, cc] = True
                        stack.append((rr, cc))

            components.append(
                np.asarray(cells, dtype=np.int16)
            )

    return components


def _reconstruct_from_solution(
    side: int,
    key: np.ndarray,
    solution: dict,
) -> np.ndarray:
    reconstructed = np.full(
        (side, side),
        -1,
        dtype=np.int16,
    )
    flat_key = key.reshape(-1)

    for corner, mask in enumerate(solution["masks"]):
        color = int(flat_key[corner])
        for index in range(side * side):
            if (mask >> index) & 1:
                reconstructed[
                    index // side,
                    index % side,
                ] = color

    _require(
        np.all(reconstructed >= 0),
        "solution did not cover the output square",
    )
    return reconstructed.astype(np.int8)


def _balanced_partition(rng, side):
    """Exchange equal numbers of cells along seams without changing areas."""
    half = side // 2
    owner = np.empty((side, side), dtype=np.int8)
    owner[:half,:half], owner[:half,half:] = 0, 1
    owner[half:,:half], owner[half:,half:] = 2, 3
    corners = {(0,0),(0,side-1),(side-1,0),(side-1,side-1)}
    seams = [(0,1),(2,3)] + [tuple(int(v) for v in rng.choice(4,2,replace=False)) for _ in range(int(rng.integers(2,side+1)))]
    for a,b in seams:
        for _ in range(12):
            boundary = {a: [], b: []}
            for r,c in np.argwhere((owner == a) | (owner == b)):
                if (r,c) in corners:
                    continue
                label = int(owner[r,c])
                other = b if label == a else a
                if any(0 <= r+dr < side and 0 <= c+dc < side and owner[r+dr,c+dc] == other for dr,dc in _NEIGHBORS):
                    boundary[label].append((r,c))
            if not boundary[a] or not boundary[b]:
                break
            x = boundary[a][int(rng.integers(len(boundary[a])))]
            y = boundary[b][int(rng.integers(len(boundary[b])))]
            owner[x], owner[y] = b, a
            if all(len(_connected_components(owner == label)) == 1 for label in (a,b)) and _key_block_locations(owner):
                break
            owner[x], owner[y] = a, b
    return owner


def _seam_partition(rng, side):
    """Independent wandering cuts, including smooth steps and straight edges."""
    if side % 2 == 0 and rng.random() < 0.3:
        return _balanced_partition(rng, side)
    top = int(rng.integers(2, side - 2))
    left_top = int(rng.integers(2, side - 1))
    left_bottom = int(rng.integers(1, side - 1))
    owner = np.empty((side, side), dtype=np.int8)
    rough = float(rng.choice([0.0, 0.25, 0.7, 1.0]))
    for r in range(side):
        if r < top:
            if r >= 2 and rng.random() < rough:
                left_top = int(np.clip(left_top + rng.choice([-1, 1]), 1, side - 1))
            cut = left_top
            owner[r, :cut], owner[r, cut:] = 0, 1
        else:
            if rng.random() < rough:
                left_bottom = int(np.clip(left_bottom + rng.choice([-1, 1]), 1, side - 1))
            cut = left_bottom
            owner[r, :cut], owner[r, cut:] = 2, 3
    # Small cross-seam exchanges remove the otherwise mandatory horizontal cut.
    for _ in range(int(rng.integers(0, side + 1))):
        r = int(rng.integers(1, side - 1))
        c = int(rng.integers(1, side - 1))
        neighbors = [int(owner[r + dr, c + dc]) for dr, dc in _NEIGHBORS]
        if rng.random() < 0.7:
            owner[r, c] = int(rng.choice(neighbors))
    return owner


def _carved_partition(rng, side):
    """Carve three independently profiled corner pieces out of a large fourth.

    A corner piece has a full outer spine and variable row widths. The
    remaining piece can wrap around recesses and reach three or four edges.
    Connectivity and uniqueness are checked after carving, not assumed.
    """
    large = int(rng.integers(4))
    owner = np.full((side, side), large, dtype=np.int8)
    for corner in rng.permutation([i for i in range(4) if i != large]):
        corner = int(corner)
        height = int(rng.integers(2, side - 1))
        max_width = int(rng.integers(2, side - 1))
        style = int(rng.integers(4))
        width = int(rng.integers(2 if corner == 0 else 1, max_width + 1))
        for depth in range(height):
            if style == 0:  # rectangle, including small simple corner pieces
                w = width
            elif style == 1:  # staircase
                w = max(1, max_width - depth * max_width // height)
            elif style == 2:  # deep recess with a returning arm
                w = max_width if depth in (0, 1, height - 1) else int(rng.integers(1, max_width))
            else:  # narrow necks and irregular lobes
                width = int(np.clip(width + rng.integers(-2, 3), 1, max_width))
                w = width
            if corner == 0 and depth < 2:
                w = max(2, w)
            for x in range(w):
                r = depth if corner < 2 else side - 1 - depth
                c = x if corner % 2 == 0 else side - 1 - x
                if owner[r, c] != large:
                    return None
                owner[r, c] = corner
    return owner


def _partition_shapes(owner, demonstrate=False):
    if owner is None:
        return None
    side = len(owner)
    if tuple(int(owner[r, c]) for r, c in ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))) != (0, 1, 2, 3):
        return None
    shapes = []
    for corner in range(4):
        coords = np.argwhere(owner == corner)
        if len(coords) < (6 if corner == 0 else 3):
            return None
        if len(_connected_components(owner == corner)) != 1:
            return None
        shapes.append(_normalize_shape(coords))
    if not _key_block_locations(owner):
        return None
    if demonstrate and len({_rotate_shape(shapes[0], k) for k in range(4)}) != 4:
        return None
    canonical = [min(_rotate_shape(s, k) for k in range(4)) for s in shapes]
    if demonstrate and canonical[0] in canonical[1:]:
        return None
    if demonstrate == 2 and (any(len({_rotate_shape(s, k) for k in range(4)}) != 4 for s in shapes) or len(set(canonical)) != 4):
        return None
    solutions = _solve_square(shapes, side)
    expected = tuple(_coordinate_mask(np.argwhere(owner == i), side) for i in range(4))
    if len(solutions) != 1 or solutions[0]['masks'] != expected:
        return None
    return shapes


def _fallback_partition(side, demonstrate):
    # A finite, deterministic search over cut positions and one-cell tabs.
    # This is only a last resort; normal construction is stochastic/compositional.
    for y in range(2, side - 1):
        for x in range(2, side - 1):
            for z in range(1, side - 1):
                for tab in range(1, side - 1):
                    owner = np.empty((side, side), dtype=np.int8)
                    owner[:y, :x], owner[:y, x:] = 0, 1
                    owner[y:, :z], owner[y:, z:] = 2, 3
                    owner[y-1, tab] = owner[y, tab]
                    # Break the two remaining rectangles' rotational symmetries.
                    owner[1, x] = 0
                    owner[side-2, z] = 2
                    shapes = _partition_shapes(owner, demonstrate)
                    if shapes is not None:
                        return owner, shapes
    raise AssertionError('no fallback partition for supported side')


def _sample_partition(rng, side, kind, demonstrate):
    for attempt in range(180):
        selected = kind if attempt < 130 else int(rng.integers(3))
        if selected == 0:
            owner = _weighted_corner_partition(rng, side, float(rng.uniform(0.3, 5.0)))
        elif selected == 1:
            owner = _seam_partition(rng, side)
        else:
            owner = _carved_partition(rng, side)
        shapes = _partition_shapes(owner, demonstrate)
        if shapes is not None:
            return owner, shapes
    return _fallback_partition(side, demonstrate)


def _shelf_layouts(rng, shapes, gap):
    """Enumerate all ordered shelf groupings, with either axis as the shelf."""
    layouts = []
    for transpose in (False, True):
        dims = [tuple(reversed(_shape_dimensions(s))) if transpose else _shape_dimensions(s) for s in shapes]
        for order in itertools.permutations(range(4)):
            for cuts in range(8):
                groups, group = [], []
                for j, piece in enumerate(order):
                    group.append(piece)
                    if j == 3 or (cuts >> j) & 1:
                        groups.append(group)
                        group = []
                widths = [sum(dims[i][1] for i in g) + gap*(len(g)-1) for g in groups]
                heights = [max(dims[i][0] for i in g) for g in groups]
                height, width = sum(heights) + gap*(len(groups)-1), max(widths)
                if height > 30 or width > 30:
                    continue
                offsets, row = [None]*4, 0
                for g, h, w in zip(groups, heights, widths):
                    col = int(rng.integers(width - w + 1))
                    for i in g:
                        rr = row + int(rng.integers(h - dims[i][0] + 1))
                        offsets[i] = (col, rr) if transpose else (rr, col)
                        col += dims[i][1] + gap
                    row += h + gap
                layout_shape = (width, height) if transpose else (height, width)
                layouts.append((offsets, layout_shape, len(groups) == 1))
    return layouts


def _free_layout(rng, shapes):
    # Bounding boxes may overlap; the actual footprints retain an empty halo.
    height, width = (int(rng.integers(19, 31)), int(rng.integers(19, 31)))
    order = list(rng.permutation(4))
    forbidden = np.zeros((height, width), dtype=bool)
    offsets = [None]*4
    for i in order:
        shape = np.asarray(shapes[i])
        h, w = _shape_dimensions(shapes[i])
        if h > height or w > width:
            return None
        for _ in range(120):
            r, c = int(rng.integers(height-h+1)), int(rng.integers(width-w+1))
            rr, cc = shape[:, 0]+r, shape[:, 1]+c
            if forbidden[rr, cc].any():
                continue
            offsets[i] = (r, c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    forbidden[np.clip(rr+dr, 0, height-1), np.clip(cc+dc, 0, width-1)] = True
            break
        else:
            return None
    return offsets


def _scatter(rng, shapes, layout_style):
    offsets = _free_layout(rng, shapes) if layout_style == 0 else None
    if offsets is None:
        gap = int(rng.choice([1, 1, 2, 3, 4]))
        layouts = _shelf_layouts(rng, shapes, gap)
        if not layouts:
            layouts = _shelf_layouts(rng, shapes, 1)
        # Row and column examples are selected when they physically fit.
        linear = [v for v in layouts if v[2]]
        pool = linear if layout_style == 1 and linear else layouts
        offsets = pool[int(rng.integers(len(pool)))][0]
    mins = np.min(offsets, axis=0)
    offsets = [(r-int(mins[0]), c-int(mins[1])) for r, c in offsets]
    extent = [max(offsets[i][a] + _shape_dimensions(s)[a] for i, s in enumerate(shapes)) for a in (0, 1)]
    pads = []
    for length in extent:
        room = 30 - length
        # Inset, single-edge, and two-edge contact are all possible.
        if room >= 2 and rng.random() < 0.65:
            before = int(rng.integers(1, min(5, room)))
            after = int(rng.integers(1, min(6, room-before+1)))
        else:
            before = int(rng.integers(min(5, room)+1))
            after = int(rng.integers(min(5, room-before)+1))
        pads.append((before, after))
    shape = tuple(extent[a] + sum(pads[a]) for a in (0, 1))
    offsets = [(r+pads[0][0], c+pads[1][0]) for r, c in offsets]
    return offsets, shape


def _sample_schedule(rng, train_count, test_count):
    """Puzzle-level emphasis controls ranges, never a hidden solving rule."""
    total = train_count + test_count
    regime = int(rng.integers(6))  # growth, overlap, equal, shrink, mixed, simple
    fixed_side = int(rng.integers(6, 14))
    rotation_free = regime == 5 or rng.random() < 0.12
    rough_weight = float(rng.uniform(0.2, 0.75))
    plan = []
    for i in range(total):
        train = i < train_count
        if regime == 0:
            side = int(rng.integers(6, 11) if train else rng.integers(11, 15))
        elif regime == 2:
            side = fixed_side
        elif regime == 3:
            side = int(rng.integers(10, 15) if train else rng.integers(6, 10))
        elif regime == 5:
            side = int(rng.integers(6, 10))
        else:
            side = int(rng.integers(6, 15))
            if regime == 4 and i < 2:
                side = int(rng.integers(6, 9) if i == 0 else rng.integers(12, 15))
        kind = int(rng.choice([0, 1, 2], p=[(1-rough_weight)/2, (1-rough_weight)/2, rough_weight]))
        if regime == 5:
            kind = int(rng.integers(2))
        rotations = [0, 0, 0, 0]
        demonstrate = (1 if rotation_free else 2) if i == 0 else 0
        if demonstrate == 2:
            rotations[1:] = [int(x) for x in rng.permutation([1, 2, 3])]
        elif not rotation_free:
            count = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.25, 0.20, 0.25]))
            chosen = rng.choice([1, 2, 3], size=count, replace=False)
            repeat = int(rng.integers(1, 4)) if rng.random() < 0.5 else None
            for piece in chosen:
                rotations[int(piece)] = repeat if repeat is not None else int(rng.integers(1, 4))
        plan.append((side, kind, rotations, demonstrate, int(rng.integers(3))))
    return plan


def _sample_keys(rng, total):
    first = rng.choice(10, 4, replace=False).astype(np.int8)
    keys = [first]
    for i in range(1, total):
        mode = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.20, 0.30]))
        if i == 1 and mode == 0:
            mode = 1
        values = keys[int(rng.integers(len(keys)))].copy()
        if mode == 1:  # A partial repeat, including fixed positions across trains.
            j = int(rng.integers(4))
            values[j] = int(rng.choice([c for c in range(10) if c not in values]))
        elif mode == 2:
            a, b = rng.choice(4, 2, replace=False)
            values[a], values[b] = values[b], values[a]
        elif mode == 3:
            values = rng.choice(10, 4, replace=False).astype(np.int8)
        if i == 1 and np.array_equal(values, first):
            values = np.roll(values, 1)
        keys.append(values)
    return [k.reshape(2, 2) for k in keys]


def _generate_pair(rng, regime, plan, key):
    _require(regime in ('source_train', 'target_test'), 'unknown regime')
    side, kind, rotations, demonstrate, layout = plan
    owner, shapes = _sample_partition(rng, side, kind, demonstrate)
    oriented = [_rotate_shape(s, k) for s, k in zip(shapes, rotations)]
    key_row, key_col = _key_block_locations(owner)[int(rng.integers(len(_key_block_locations(owner))))]
    available = [c for c in range(10) if c not in key]
    bg = 0 if 0 in available and rng.random() < 0.4 else int(rng.choice(available))
    carrier = int(rng.choice([c for c in available if c != bg]))
    carrier_count = int(rng.choice([1, 2, 3, 4], p=[0.28, 0.25, 0.20, 0.27]))
    # Other carriers can coincide with key colors; only the anchor must be distinct.
    pool = [carrier] + [int(c) for c in rng.choice([c for c in range(10) if c not in (bg, carrier)], carrier_count-1, replace=False)]
    colors = [carrier] + pool[1:] + [int(rng.choice(pool)) for _ in range(4-carrier_count)]
    colors[1:] = [colors[i] for i in rng.permutation([1, 2, 3])]
    def paint(offsets, canvas):
        grid = np.full(canvas, bg, dtype=np.int8)
        for i, s in enumerate(oriented):
            coords = np.asarray(s) + offsets[i]
            grid[coords[:, 0], coords[:, 1]] = colors[i]
        r, c = offsets[0]
        grid[r+key_row:r+key_row+2, c+key_col:c+key_col+2] = key
        return grid

    for _ in range(20):
        offsets, canvas = _scatter(rng, oriented, layout)
        grid = paint(offsets, canvas)
        border = _border(grid)
        if np.count_nonzero(border == bg) > len(border)//2:
            return {'input': grid, 'output': key.reshape(-1)[owner].astype(np.int8)}
    # Guaranteed fallback: two shelves plus a blank top and left border.
    # Each footprint is at most 14 by 14, so this fits within 30 by 30.
    dims = [_shape_dimensions(s) for s in oriented]
    row2 = 2 + max(dims[0][0], dims[1][0])
    offsets = [(1,1),(1,dims[0][1]+2),(row2,1),(row2,dims[2][1]+2)]
    canvas = (row2+max(dims[2][0],dims[3][0]), max(dims[0][1]+dims[1][1],dims[2][1]+dims[3][1])+2)
    return {'input': paint(offsets, canvas), 'output': key.reshape(-1)[owner].astype(np.int8)}


def _border(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _parse_input(grid):
    border = _border(grid)
    background = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == background) > len(border)//2, 'background must be readable on the border')
    components = _connected_components(grid != background)
    _require(len(components) == 4, 'input must have four disconnected footprints')
    for x, y in itertools.combinations(components, 2):
        _require(np.max(np.abs(x[:,None,:] - y[None,:,:]), axis=2).min() >= 2, 'footprints need a background-cell gap')
    anchors, uniform = [], []
    for coords in components:
        values = grid[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        if len(colors) == 1:
            uniform.append((coords, int(colors[0])))
            continue
        _require(len(colors) == 5 and sum(counts == 1) == 4 and sum(counts > 1) == 1, 'anchor must have one carrier and four singleton key cells')
        points = coords[np.isin(values, colors[counts == 1])]
        r, c = points.min(axis=0)
        _require(set(map(tuple, points)) == {(r,c), (r+1,c), (r,c+1), (r+1,c+1)}, 'key must be an ordered 2 by 2 block')
        anchors.append((coords, int(colors[counts > 1][0]), grid[r:r+2, c:c+2].copy(), (int(r-coords[:,0].min()), int(c-coords[:,1].min()))))
    _require(len(anchors) == 1 and len(uniform) == 3, 'exactly one anchor and three plain pieces required')
    coords, carrier, key, key_location = anchors[0]
    shapes = [_normalize_shape(coords)] + [_normalize_shape(v[0]) for v in uniform]
    total = sum(map(len, shapes))
    side = int(total ** 0.5)
    _require(side*side == total, 'total piece area must be square')
    _require(carrier not in key and background not in key, 'anchor palette is ambiguous')
    return {'side': side, 'key': key, 'shapes': shapes, 'background': background,
            'carriers': [carrier] + [v[1] for v in uniform],
            'components': [coords] + [v[0] for v in uniform], 'key_location': key_location}


def _solve_input(grid):
    analysis = _parse_input(grid)
    solutions = _solve_square(analysis['shapes'], analysis['side'])
    _require(len(solutions) == 1, 'visible pieces must have exactly one colored square packing')
    analysis['solution'] = solutions[0]
    analysis['reconstructed'] = _reconstruct_from_solution(analysis['side'], analysis['key'], solutions[0])
    return analysis


def _pair_digest(pair):
    digest = hashlib.sha256()
    for role in ('input', 'output'):
        grid = pair[role].astype(np.int8)
        digest.update(bytes(grid.shape))
        digest.update(grid.tobytes())
    return digest.hexdigest()


def _analyze_pair(input_grid, output_grid):
    analysis = _solve_input(input_grid)
    _require(np.array_equal(analysis['reconstructed'], output_grid), 'output must equal the input-only keyed packing')
    return analysis


def predict_from_training(train_pairs, test_input):
    """Infer the corner/key convention from demonstrations, then pack input.

    This is an internal prediction check, not a protected reference solver.
    Test labels and generation state are not arguments or consulted globals.
    """
    mappings = set(itertools.permutations(range(4)))
    anchor_conventions = set(itertools.product(range(4), range(4)))
    demonstrated_rotations = {0}
    for pair in train_pairs:
        a = _solve_input(pair['input'])
        out = pair['output']
        _require(out.shape == (a['side'], a['side']), 'training square area mismatch')
        corner_colors = [int(out[r, c]) for r,c in ((0,0),(0,-1),(-1,0),(-1,-1))]
        key = a['key'].reshape(-1)
        mappings = {p for p in mappings if all(corner_colors[i] == key[p[i]] for i in range(4))}
        _require(np.array_equal(out, a['reconstructed']), 'inconsistent training packing')
        target_shapes = [_normalize_shape(np.argwhere(out == color)) for color in corner_colors]
        anchor_conventions = {(c,k) for c,k in anchor_conventions if _rotate_shape(a['shapes'][0],k) == target_shapes[c]}
        demonstrated_rotations.update(a['solution']['rotations'])
    _require(mappings == {(0,1,2,3)}, 'training must establish spatial key order')
    _require(anchor_conventions == {(0,0)}, 'training must establish the unrotated top-left anchor')
    test = _solve_input(test_input)
    _require(set(test['solution']['rotations']) <= demonstrated_rotations, 'training must demonstrate required asymmetric rotations')
    return test['reconstructed'].copy()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _is_integer_argument(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(seed)
    plans = _sample_schedule(rng, train_count, test_count)
    keys = _sample_keys(rng, train_count + test_count)
    puzzle, seen_inputs, seen_layouts = {'train': [], 'test': []}, set(), set()
    for i, (plan, key) in enumerate(zip(plans, keys)):
        split = 'train' if i < train_count else 'test'
        regime = 'source_train' if split == 'train' else 'target_test'
        for _ in range(100):
            pair = _generate_pair(rng, regime, plan, key)
            signature = _grid_key(pair['input'])
            layout = _normalized_layout_key(pair['output'], key)
            if signature in seen_inputs or layout in seen_layouts or _pair_digest(pair) in _SOURCE_DIGESTS:
                continue
            seen_inputs.add(signature)
            seen_layouts.add(layout)
            puzzle[split].append(pair)
            break
        else:
            raise AssertionError('unable to produce distinct cuts')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train','test'}, 'exactly train and test required')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test']), 'illegal pair counts')
    seen, layouts, train_keys = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input','output'}, 'pair schema mismatch')
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a two dimensional NumPy array')
                _require(np.issubdtype(grid.dtype, np.integer), 'grid must have integer dtype')
                _require(all(1 <= d <= 30 for d in grid.shape) and bool(np.all((grid >= 0) & (grid <= 9))), 'grid bounds or colors invalid')
            signature = _grid_key(pair['input'])
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            _require(_pair_digest(pair) not in _SOURCE_DIGESTS, 'exact source pair copied')
            _require(not np.array_equal(pair['input'], pair['output']), 'input cannot already be the output')
            analysis = _analyze_pair(pair['input'], pair['output'])
            layout = _normalized_layout_key(pair['output'], analysis['key'])
            _require(layout not in layouts, 'cuts must vary within a puzzle')
            layouts.add(layout)
            if split == 'train':
                train_keys.add(tuple(analysis['key'].flat))
    _require(len(train_keys) >= 2, 'training must demonstrate the visible legend can change')
    train_copies = [{k: v.copy() for k,v in p.items()} for p in puzzle['train']]
    for pair in puzzle['test']:
        predicted = predict_from_training(train_copies, pair['input'].copy())
        _require(np.array_equal(predicted, pair['output']), 'training-only prediction failed')


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(np.array_equal(a[k],b[k]) for a,b in zip(first[s],second[s]) for k in ('input','output')) for s in ('train','test'))


def _has_neck(coords):
    """A one-cell neck is an articulation vertex of the footprint graph."""
    cells = set(map(tuple, coords.tolist()))
    arrival, low = {}, {}
    found = False

    def visit(cell, parent):
        nonlocal found
        arrival[cell] = low[cell] = len(arrival)
        children = 0
        for dr, dc in _NEIGHBORS:
            neighbor = (cell[0]+dr, cell[1]+dc)
            if neighbor not in cells or neighbor == parent:
                continue
            if neighbor in arrival:
                low[cell] = min(low[cell], arrival[neighbor])
            else:
                children += 1
                visit(neighbor, cell)
                low[cell] = min(low[cell], low[neighbor])
                if parent is not None and low[neighbor] >= arrival[cell]:
                    found = True
        if parent is None and children > 1:
            found = True

    visit(min(cells), None)
    return found


def measure_pair(pair):
    """Features reconstructed from visible grids, not sampled plan parameters."""
    a = _analyze_pair(pair['input'], pair['output'])
    out, grid = pair['output'], pair['input']
    key = a['key'].reshape(-1)
    areas, fills, edges, thin, recesses, rectangles = [], [], [], 0, 0, 0
    necks, staircase = 0, 0
    for color in key:
        mask = out == color
        coords = np.argwhere(mask)
        r0,c0 = coords.min(axis=0)
        r1,c1 = coords.max(axis=0)+1
        box = mask[r0:r1,c0:c1]
        area = len(coords)
        areas.append(area)
        necks += int(_has_neck(coords))
        fills.append(area / box.size)
        edges.append(sum(bool(edge.any()) for edge in (mask[0],mask[-1],mask[:,0],mask[:,-1])))
        rectangles += int(box.all())
        thin += int(bool((box.sum(axis=0) == 1).any() or (box.sum(axis=1) == 1).any()))
        for view in (box, box.T):
            for end in (0, -1):
                boundary = [int(np.flatnonzero(row)[end]) for row in view]
                run, previous = 0, 0
                for step in np.diff(boundary):
                    run = run+1 if abs(step) == 1 and step == previous else int(abs(step) == 1)
                    previous = step
                    staircase = max(staircase, run)
            for row in view:
                occupied = np.flatnonzero(row)
                if len(occupied) > 1:
                    # Long empty intervals inside a footprint's row/column
                    # measure returning arms around an open recess.
                    recesses = max(recesses, int(np.diff(occupied).max(initial=1))-1)
    bounds = []
    for comp in a['components']:
        mins,maxs = comp.min(axis=0), comp.max(axis=0)+1
        bounds.append((int(mins[0]),int(maxs[0]),int(mins[1]),int(maxs[1])))
    row = (all(x[3] <= y[2] or y[3] <= x[2] for x,y in itertools.combinations(bounds,2))
           and max(x[0] for x in bounds) < min(x[1] for x in bounds))
    col = (all(x[1] <= y[0] or y[1] <= x[0] for x,y in itertools.combinations(bounds,2))
           and max(x[2] for x in bounds) < min(x[3] for x in bounds))
    quadrant = False
    for top in itertools.combinations(range(4),2):
        bottom = [i for i in range(4) if i not in top]
        if max(bounds[i][1] for i in top) > min(bounds[i][0] for i in bottom):
            continue
        for left_top in top:
            for left_bottom in bottom:
                left = [left_top,left_bottom]
                right = [i for i in range(4) if i not in left]
                if max(bounds[i][3] for i in left) <= min(bounds[i][2] for i in right):
                    quadrant = True
    bg = a['background']
    contacts = [bool((edge != bg).any()) for edge in (grid[0], grid[-1], grid[:,0], grid[:,-1])]
    gaps = []
    for x,y in itertools.combinations(a['components'],2):
        gaps.append(int(np.max(np.abs(x[:,None,:]-y[None,:,:]),axis=2).min())-1)
    rotations = tuple(int(x) for x in a['solution']['rotations'][1:])
    reading_order = sorted(range(4), key=lambda i: (bounds[i][0],bounds[i][2]))
    corner_for_piece = {piece:corner for corner,piece in enumerate(a['solution']['piece_for_corner'])}
    return {
        'side': a['side'], 'canvas': tuple(grid.shape), 'areas': tuple(areas),
        'area_ratio': max(areas)/min(areas), 'largest_corner': int(np.argmax(areas)),
        'min_fill': min(fills), 'max_edges': max(edges), 'rectangles': rectangles,
        'thin_pieces': thin, 'neck_pieces': necks, 'staircase_run': staircase,
        'recess_depth': recesses, 'distinct_areas': len(set(areas)),
        'row': row, 'column': col, 'quadrant': quadrant, 'contacts': tuple(contacts),
        'min_gap': min(gaps), 'background': bg, 'carrier_count': len(set(a['carriers'])),
        'black_carrier': 0 in a['carriers'],
        'input_colors': len(np.unique(grid)), 'key': tuple(int(v) for v in key),
        'key_location': a['key_location'], 'rotations': rotations,
        'reading_order': tuple(corner_for_piece[i] for i in reading_order),
        'rotated_pieces': sum(v != 0 for v in rotations),
        'layout': _normalized_layout_key(out, a['key']),
    }


def self_test() -> None:
    a = generate_puzzle(314159, 6, 2)
    _require(_puzzles_equal(a,generate_puzzle(314159,6,2)), 'determinism failed')
    _require(not _puzzles_equal(a,generate_puzzle(314160,6,2)), 'seeds must change puzzles')
    for i,(tr,te) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p = generate_puzzle(7000+i,tr,te)
        validate_generated_puzzle(p)
        _require(tr is None or len(p['train']) == tr, 'requested train count ignored')
        _require(te is None or len(p['test']) == te, 'requested test count ignored')
    for tr,te in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1),(2.0,1)):
        try:
            generate_puzzle(72,tr,te)
        except ValueError:
            continue
        raise AssertionError('invalid counts must raise ValueError')
    counts = [set(),set()]
    rows, split_rows = [], {'train': [], 'test': []}
    regimes, varied_geometry, varied_canvases, varied_backgrounds = set(),0,0,0
    partial_keys, repeated_keys, combined, seen = 0,0,0,set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts[0].add(len(puzzle['train']))
        counts[1].add(len(puzzle['test']))
        local = []
        for split in ('train','test'):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                rows.append(m);local.append(m);split_rows[split].append(m)
                sig = _grid_key(pair['input'])
                _require(sig not in seen, 'cross-seed duplicate input')
                seen.add(sig)
                _require(m['min_gap'] >= 1, 'pieces need background separation')
                combined += int(m['area_ratio'] >= 4 and m['background'] != 0 and m['carrier_count'] < 4 and not m['quadrant'])
        train_sides = [x['output'].shape[0] for x in puzzle['train']]
        for pair in puzzle['test']:
            side = pair['output'].shape[0]
            regimes.add('larger' if side > max(train_sides) else 'smaller' if side < min(train_sides) else 'equal' if side in train_sides else 'overlap')
        varied_geometry += len({(m['area_ratio'],m['min_fill'],m['max_edges']) for m in local}) > 1
        varied_canvases += len({m['canvas'] for m in local}) > 1
        varied_backgrounds += len({m['background'] for m in local}) > 1
        keys = [m['key'] for m in local]
        repeated_keys += len(set(keys)) < len(keys)
        partial_keys += any(0 < sum(u==v for u,v in zip(x,y)) < 4 for x,y in itertools.combinations(keys,2))
    _require(len(counts[0]) >= 4 and len(counts[1]) >= 3, 'automatic counts lack variety')
    _require(len({r['canvas'] for r in rows}) >= 70 and len({r['side'] for r in rows}) == 9, 'dimensions lack variety')
    _require(len({r['layout'] for r in rows}) > len(rows)*0.8, 'normalized cuts lack variety')
    _require(varied_geometry >= 40 and varied_canvases >= 40 and varied_backgrounds >= 30, 'pair-level properties collapsed')
    _require(regimes == {'larger','smaller','equal','overlap'}, 'size regimes missing')
    _require(max(r['area_ratio'] for r in rows) >= 5.17 and sum(r['area_ratio'] >= 4 for r in rows) >= 20, 'unequal pieces missing')
    _require({r['largest_corner'] for r in rows if r['area_ratio'] >= 4} == {0,1,2,3}, 'large role correlated with anchor')
    _require(min(r['min_fill'] for r in rows) <= 0.52 and sum(r['max_edges'] >= 3 for r in rows) >= 15, 'interlocking reach missing')
    _require(sum(r['recess_depth'] >= 3 for r in rows) >= 10 and sum(r['thin_pieces'] > 0 for r in rows) >= 20, 'deep recesses or narrow arms missing')
    _require(sum(r['neck_pieces'] > 0 for r in rows) >= 20 and sum(r['staircase_run'] >= 3 for r in rows) >= 15, 'necks or staircases missing')
    _require(any(r['rectangles'] >= 2 for r in rows) and any(r['distinct_areas'] <= 2 for r in rows), 'simpler cuts filtered away')
    _require(sum(r['distinct_areas'] == 1 and r['rectangles'] < 4 for r in rows) >= 4, 'equal-area irregular cuts missing')
    _require(sum(r['row'] for r in rows) >= 8 and sum(r['column'] for r in rows) >= 8, 'linear layouts missing')
    _require(sum(not r['quadrant'] for r in rows) >= len(rows)//3, 'quadrant layout still dominant')
    _require(len({r['reading_order'] for r in rows}) == 24, 'input position is correlated with output corner')
    _require(sum(not any(r['contacts']) for r in rows) >= 30 and all(any(r['contacts'][i] for r in rows) for i in range(4)), 'inset/edge diversity missing')
    _require(len({r['background'] for r in rows}) == 10 and any(0 in r['key'] for r in rows), 'palette roles fixed')
    _require({r['carrier_count'] for r in rows} == {1,2,3,4} and len({r['input_colors'] for r in rows}) >= 4, 'palette sharing missing')
    _require(partial_keys >= 30 and repeated_keys >= 15 and len({r['key_location'] for r in rows}) >= 20, 'key variation or repetition missing')
    _require(combined >= 5, 'compatible variation axes do not combine')
    for split, records in split_rows.items():
        _require({r['rotated_pieces'] for r in records} == {0,1,2,3}, split+' lacks actual rotation spectrum')
        _require(any(sorted(r['rotations']) == [0,2,2] for r in records), split+' lacks repeated half-turn case')
        _require({v for r in records for v in r['rotations']} == {0,1,2,3}, split+' lacks quarter-turn coverage')
    # Corruption checks make sure reconstruction and convention checks are live.
    damaged = {s:[{k:v.copy() for k,v in p.items()} for p in a[s]] for s in a}
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect test output was accepted')
