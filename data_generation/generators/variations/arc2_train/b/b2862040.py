import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = {
    "f259b8a536d4ac4c1a0a1e780a677c7224c881b7c97bc5df7f60f47c374b5353",
    "e50ab45962147a10e0c93e2569319ad7bc59fda7ca2d2852c1cc4ca39b0bda4c",
    "c79daeb3a3287cd24230b6b296b36635ebeca41ae07ed97cc4f1262e6e67db06",
    "3e7c0ce30f231501bdc5e9bc739ba95f86f37325b4b831ebe54e5db8184e7378",
    "18f7b54768b4ffa0088a42e66a4558bafdf16ce1dba7dbbebfb45b20f84078a5",
}

_DIRS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)


def _validate_count_arg(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> None:
    if value is None:
        return

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count_arg(num_train, "num_train", 2, 10)
    _validate_count_arg(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        test_count = 1 + (int(seed) % 4)
        choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + ((int(seed) // 4) % choices)
        )
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(
            rng.integers(
                1,
                min(4, train_count - 1) + 1,
            )
        )
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def _components_of_color(
    grid: np.ndarray,
    color: int,
) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) != color:
                continue

            seen[row, col] = True
            stack = [(row, col)]
            component = []

            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))

                for delta_row, delta_col in _DIRS:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col

                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and not seen[next_row, next_col]
                        and int(grid[next_row, next_col]) == color
                    ):
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))

            components.append(component)

    return components


def _component_hole_count(
    component: list[tuple[int, int]],
) -> int:
    min_row = min(row for row, _ in component)
    max_row = max(row for row, _ in component)
    min_col = min(col for _, col in component)
    max_col = max(col for _, col in component)

    box_height = max_row - min_row + 1
    box_width = max_col - min_col + 1

    wall = np.zeros(
        (box_height + 2, box_width + 2),
        dtype=bool,
    )

    for row, col in component:
        wall[
            row - min_row + 1,
            col - min_col + 1,
        ] = True

    reached = np.zeros_like(wall)
    reached[0, 0] = True
    stack = [(0, 0)]

    while stack:
        row, col = stack.pop()

        for delta_row, delta_col in _DIRS:
            next_row = row + delta_row
            next_col = col + delta_col

            if (
                0 <= next_row < wall.shape[0]
                and 0 <= next_col < wall.shape[1]
                and not wall[next_row, next_col]
                and not reached[next_row, next_col]
            ):
                reached[next_row, next_col] = True
                stack.append((next_row, next_col))

    holes = 0

    for row in range(1, box_height + 1):
        for col in range(1, box_width + 1):
            if wall[row, col] or reached[row, col]:
                continue

            holes += 1
            reached[row, col] = True
            stack = [(row, col)]

            while stack:
                current_row, current_col = stack.pop()

                for delta_row, delta_col in _DIRS:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col

                    if (
                        0 <= next_row < wall.shape[0]
                        and 0 <= next_col < wall.shape[1]
                        and not wall[next_row, next_col]
                        and not reached[next_row, next_col]
                    ):
                        reached[next_row, next_col] = True
                        stack.append((next_row, next_col))

    return holes


def _component_mask(
    component: list[tuple[int, int]],
) -> np.ndarray:
    min_row = min(row for row, _ in component)
    max_row = max(row for row, _ in component)
    min_col = min(col for _, col in component)
    max_col = max(col for _, col in component)

    mask = np.zeros(
        (
            max_row - min_row + 1,
            max_col - min_col + 1,
        ),
        dtype=bool,
    )

    for row, col in component:
        mask[row - min_row, col - min_col] = True

    return mask


def _analyze_input(
    grid: np.ndarray,
    foreground: int,
) -> list[dict]:
    result = []

    for component in _components_of_color(
        grid,
        foreground,
    ):
        result.append(
            {
                "coords": component,
                "hole_count": _component_hole_count(
                    component
                ),
                "mask": _component_mask(component),
            }
        )

    return result


def _is_plain_rectangle_ring(
    component: list[tuple[int, int]],
) -> bool:
    mask = _component_mask(component)
    height, width = mask.shape

    if height < 3 or width < 3:
        return False

    perimeter = np.zeros_like(mask)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    return bool(np.array_equal(mask, perimeter))


def _has_peelable_extension(
    component: list[tuple[int, int]],
) -> bool:
    points = set(component)

    degrees = {
        point: sum(
            (
                point[0] + delta_row,
                point[1] + delta_col,
            )
            in points
            for delta_row, delta_col in _DIRS
        )
        for point in points
    }

    stack = [
        point
        for point, degree in degrees.items()
        if degree <= 1
    ]
    removed = set()

    while stack:
        point = stack.pop()

        if point in removed or degrees[point] > 1:
            continue

        removed.add(point)

        for delta_row, delta_col in _DIRS:
            neighbor = (
                point[0] + delta_row,
                point[1] + delta_col,
            )

            if (
                neighbor in degrees
                and neighbor not in removed
            ):
                degrees[neighbor] -= 1

                if degrees[neighbor] <= 1:
                    stack.append(neighbor)

    return 0 < len(removed) < len(points)


def _apply_rule(
    grid: np.ndarray,
    foreground: int,
    highlight: int,
) -> np.ndarray:
    output = grid.copy()

    for component in _components_of_color(
        grid,
        foreground,
    ):
        if _component_hole_count(component) > 0:
            for row, col in component:
                output[row, col] = highlight

    return output


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))

    return np.ascontiguousarray(
        mask[
            rows[0] : rows[-1] + 1,
            cols[0] : cols[-1] + 1,
        ]
    )


def _random_orient(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.rot90(
        mask,
        int(rng.integers(0, 4)),
    )

    if int(rng.integers(0, 2)):
        result = np.fliplr(result)

    return _crop_mask(result)


def _points(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]


def _mask_is_connected(mask: np.ndarray) -> bool:
    points = set(_points(mask))
    if not points:
        return False
    seen = {next(iter(points))}
    stack = list(seen)
    while stack:
        row, col = stack.pop()
        for dr, dc in _DIRS:
            point = (row + dr, col + dc)
            if point in points and point not in seen:
                seen.add(point)
                stack.append(point)
    return seen == points


def _boundary_of_cavity(cavity: np.ndarray) -> np.ndarray:
    padded = np.pad(cavity, 1)
    wall = np.zeros_like(padded)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            source = padded[max(0, -dr):padded.shape[0]-max(0, dr),
                            max(0, -dc):padded.shape[1]-max(0, dc)]
            target = wall[max(0, dr):wall.shape[0]-max(0, -dr),
                          max(0, dc):wall.shape[1]-max(0, -dc)]
            target |= source
    wall &= ~padded
    return _crop_mask(wall)


def _add_extensions(mask: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    # Each arm starts on an outer extreme and runs into free space. A second
    # segment can turn, making bent and multi-ended branches.
    result = np.pad(mask, 8)
    points = _points(result)
    for _ in range(count):
        side = int(rng.integers(0, 4))
        if side == 0:
            extreme = min(r for r, _ in points)
            anchors = [(r, c) for r, c in points if r == extreme]
            dr, dc = -1, 0
        elif side == 1:
            extreme = max(r for r, _ in points)
            anchors = [(r, c) for r, c in points if r == extreme]
            dr, dc = 1, 0
        elif side == 2:
            extreme = min(c for _, c in points)
            anchors = [(r, c) for r, c in points if c == extreme]
            dr, dc = 0, -1
        else:
            extreme = max(c for _, c in points)
            anchors = [(r, c) for r, c in points if c == extreme]
            dr, dc = 0, 1
        row, col = anchors[int(rng.integers(len(anchors)))]
        for _ in range(int(rng.integers(1, 5))):
            row += dr
            col += dc
            result[row, col] = True
        if rng.random() < 0.55:
            if dr:
                turn = (-1 if rng.random() < 0.5 else 1, 0)
                turn = (0, turn[0])
            else:
                turn = (-1 if rng.random() < 0.5 else 1, 0)
            for _ in range(int(rng.integers(1, 4))):
                row += turn[0]
                col += turn[1]
                result[row, col] = True
    return _crop_mask(result)


def _make_closed_mask(rng: np.random.Generator, scale: str,
                      style: str = 'mixed') -> np.ndarray:
    for _ in range(50):
        if style == 'tiny':
            cavity = np.ones((1, 1), dtype=bool)
            mask = np.ones((3, 3), dtype=bool)
            mask[1, 1] = False
        else:
            upper = {'small': 4, 'medium': 5, 'large': 6}[scale]
            h = int(rng.integers(2, upper + 1))
            w = int(rng.integers(2, upper + 2))
            cavity = np.zeros((h + 4, w + 4), dtype=bool)
            cavity[2:2+h, 2:2+w] = True
            if style in ('complex', 'mixed') and (style == 'complex' or rng.random() < 0.60):
                side = int(rng.integers(4))
                if side == 0:
                    start = int(rng.integers(2, w + 1))
                    cavity[1, start:start+int(rng.integers(1, w + 2 - start))] = True
                elif side == 1:
                    start = int(rng.integers(2, w + 1))
                    cavity[h+2, start:start+int(rng.integers(1, w + 2 - start))] = True
                elif side == 2:
                    start = int(rng.integers(2, h + 1))
                    cavity[start:start+int(rng.integers(1, h + 2 - start)), 1] = True
                else:
                    start = int(rng.integers(2, h + 1))
                    cavity[start:start+int(rng.integers(1, h + 2 - start)), w+2] = True
            if style == 'complex' or (style == 'mixed' and rng.random() < 0.22):
                # A second disjoint cavity shares a connected outer wall.
                h2 = int(rng.integers(2, min(5, upper) + 1))
                w2 = int(rng.integers(2, min(5, upper) + 1))
                combined = np.zeros((max(cavity.shape[0], h2 + 4), cavity.shape[1] + w2 + 2), dtype=bool)
                combined[:cavity.shape[0], :cavity.shape[1]] = cavity
                combined[2:2+h2, cavity.shape[1]-1:cavity.shape[1]-1+w2] = True
                cavity = combined
            mask = _boundary_of_cavity(cavity)
        if style == 'complex':
            arms = int(rng.integers(2, 4))
        elif style == 'plain' or style == 'tiny':
            arms = 0
        else:
            arms = int(rng.integers(0, 4))
        if arms:
            mask = _add_extensions(mask, rng, arms)
        mask = _random_orient(mask, rng)
        if (_mask_is_connected(mask)
                and _component_hole_count(_points(mask)) > 0
                and (style != 'complex' or _has_peelable_extension(_points(mask)))):
            return mask
    raise AssertionError('closed mask construction failed')


def _make_open_mask(rng: np.random.Generator, scale: str,
                    style: str = 'mixed') -> np.ndarray:
    for _ in range(50):
        if style == 'large' or (style == 'mixed' and rng.random() < 0.65):
            upper = {'small': 6, 'medium': 8, 'large': 10}[scale]
            h = int(rng.integers(4, upper + 1))
            w = int(rng.integers(4, upper + 1))
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
            side = int(rng.integers(4))
            breach = int(rng.integers(1, 3))
            if side == 0:
                start = int(rng.integers(1, w - breach))
                mask[0, start:start+breach] = False
            elif side == 1:
                start = int(rng.integers(1, w - breach))
                mask[-1, start:start+breach] = False
            elif side == 2:
                start = int(rng.integers(1, h - breach))
                mask[start:start+breach, 0] = False
            else:
                start = int(rng.integers(1, h - breach))
                mask[start:start+breach, -1] = False
            if rng.random() < 0.45:
                mask = _add_extensions(mask, rng, int(rng.integers(1, 3)))
        else:
            h = int(rng.integers(3, {'small': 5, 'medium': 7, 'large': 9}[scale] + 1))
            w = int(rng.integers(3, {'small': 6, 'medium': 8, 'large': 10}[scale] + 1))
            mask = np.zeros((h, w), dtype=bool)
            kind = int(rng.integers(4))
            if kind == 0:
                mask[h//2, :] = True
                mask[:, w//2] = True
            elif kind == 1:
                mask[:, 0] = True
                mask[-1, :] = True
                mask[:h//2+1, -1] = True
            elif kind == 2:
                mask[0, :] = True
                mask[:, w//2] = True
                mask[-1, w//2:] = True
            else:
                mask[:, 0] = True
                mask[-1, :] = True
                mask[0, :w//2+1] = True
        mask = _random_orient(mask, rng)
        if _mask_is_connected(mask) and _component_hole_count(_points(mask)) == 0:
            return mask
    raise AssertionError('open mask construction failed')


def _place_masks(rng: np.random.Generator, height: int, width: int,
                 masks: list[np.ndarray], background: int, foreground: int) -> np.ndarray | None:
    if any(m.shape[0] > height or m.shape[1] > width for m in masks):
        return None
    order = sorted(range(len(masks)), key=lambda i: masks[i].size, reverse=True)
    for _ in range(36):
        boxes = np.zeros((height, width), dtype=bool)
        placements = {}
        for i in order:
            mh, mw = masks[i].shape
            candidates = []
            for r in range(height - mh + 1):
                for c in range(width - mw + 1):
                    r0, r1 = max(0, r-1), min(height, r+mh+1)
                    c0, c1 = max(0, c-1), min(width, c+mw+1)
                    if not np.any(boxes[r0:r1, c0:c1]):
                        candidates.append((r, c))
            if not candidates:
                break
            edges = [(r,c) for r,c in candidates if r == 0 or c == 0 or r+mh == height or c+mw == width]
            if edges and rng.random() < 0.42:
                r,c = edges[int(rng.integers(len(edges)))]
            else:
                r,c = candidates[int(rng.integers(len(candidates)))]
            placements[i] = (r,c)
            boxes[r:r+mh, c:c+mw] = True
        if len(placements) != len(masks):
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for i,(r,c) in placements.items():
            m = masks[i]
            grid[r:r+m.shape[0], c:c+m.shape[1]][m] = foreground
        return grid
    return None


def _sample_shape(rng: np.random.Generator, regime: str,
                  pair_index: int) -> tuple[int, int, str]:
    if regime == 'source_train' and pair_index == 0:
        scale = 'medium'
        h = int(rng.integers(19, 24))
        w = int(rng.integers(20, 26))
    else:
        weights = (0.35, 0.43, 0.22) if regime == 'source_train' else (0.25, 0.36, 0.39)
        scale = ('small', 'medium', 'large')[int(rng.choice(3, p=weights))]
        ranges = {'small': ((11, 17), (12, 19)),
                  'medium': ((16, 24), (16, 25)),
                  'large': ((22, 30), (22, 30))}
        hr, wr = ranges[scale]
        h = int(rng.integers(*hr))
        w = int(rng.integers(*wr))
    if rng.random() < 0.35:
        h,w = w,h
    return h,w,scale


def _pair_roles(rng: np.random.Generator, highlight: int,
                base: tuple[int, int], varied: bool, index: int) -> tuple[int, int]:
    if not varied:
        return base
    choices = [c for c in range(10) if c != highlight]
    if index == 0:
        return base
    for _ in range(20):
        roles = tuple(int(x) for x in rng.choice(choices, 2, replace=False))
        if roles != base:
            return roles
    raise AssertionError('palette choice failed')


def _generate_pair(rng: np.random.Generator, regime: str, pair_index: int,
                   background: int, foreground: int, highlight: int) -> dict:
    for attempt in range(65):
        height,width,scale = _sample_shape(rng, regime, pair_index)
        capacity = {'small': 1, 'medium': 2, 'large': 3}[scale]
        if regime == 'source_train' and pair_index == 0:
            selected_count = 1
            open_count = 1
        elif regime == 'source_train':
            selected_count = int(rng.integers(1, capacity+1))
            open_count = int(rng.integers(1, min(3, capacity+1)+1))
        else:
            selected_count = 0 if rng.random() < 0.09 else int(rng.integers(1, capacity+1))
            open_count = int(rng.integers(1, min(4, capacity+2)+1))
        masks = []
        for i in range(selected_count):
            if regime == 'source_train' and pair_index == 0:
                style = 'complex'
            elif regime == 'source_train' and pair_index == 1 and i == 0:
                style = 'tiny'
            else:
                style = 'mixed'
            masks.append(_make_closed_mask(rng, scale, style))
        for i in range(open_count):
            style = 'large' if i == 0 and (regime == 'source_train' and pair_index == 0 or regime == 'target_test' and rng.random() < 0.6) else 'mixed'
            masks.append(_make_open_mask(rng, scale, style))
        grid = _place_masks(rng, height, width, masks, background, foreground)
        if grid is None or np.count_nonzero(grid == background) <= grid.size // 2:
            continue
        info = _analyze_input(grid, foreground)
        if len(info) != len(masks) or sum(x['hole_count'] > 0 for x in info) != selected_count:
            continue
        return {'input': grid, 'output': _apply_rule(grid, foreground, highlight)}
    raise ValueError(f'could not construct {regime} pair')


def _canonical_grid(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), grid.astype(np.uint8, copy=False).tobytes())


def _source_pair_hash(pair: dict) -> str:
    a = pair['input'].astype(np.uint8, copy=False)
    b = pair['output'].astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _pair_profile(pair: dict, highlight: int | None = None) -> tuple:
    a,b = pair['input'],pair['output']
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[int(np.argmin(counts))])
    assert np.max(counts) > a.size // 2
    changed = a != b
    new_colors = set(int(x) for x in np.unique(b)) - set(int(x) for x in colors)
    if highlight is None:
        assert len(new_colors) == 1
        highlight = new_colors.pop()
    else:
        assert new_colors <= {highlight}
    assert highlight not in (background, foreground)
    assert np.array_equal(_apply_rule(a, foreground, highlight), b)
    info = _analyze_input(a, foreground)
    selected = [x for x in info if x['hole_count'] > 0]
    preserved = [x for x in info if x['hole_count'] == 0]
    assert len(info) == len(selected) + len(preserved)
    assert preserved
    assert bool(changed.any()) == bool(selected)
    return (background, foreground, highlight), info, selected, preserved


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, pairs = set(),set()
    highlight = None
    train_complex = False
    train_plain = False
    train_open = False
    train_palettes = []
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input','output'}
            a,b = pair['input'],pair['output']
            for grid in (a,b):
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype,np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape
            key = _canonical_grid(a)
            pairkey = (key,_canonical_grid(b))
            assert key not in inputs and pairkey not in pairs
            inputs.add(key); pairs.add(pairkey)
            assert _source_pair_hash(pair) not in _SOURCE_PAIR_HASHES
            roles, info, selected, preserved = _pair_profile(pair, highlight)
            highlight = roles[2]
            if split == 'train':
                assert selected and preserved
                train_palettes.append(roles[:2])
                train_complex |= any(not _is_plain_rectangle_ring(x['coords']) and _has_peelable_extension(x['coords']) for x in selected)
                train_plain |= any(_is_plain_rectangle_ring(x['coords']) for x in selected)
                train_open |= bool(preserved)
            else:
                assert len(selected) <= 3
                assert preserved
    assert train_complex and train_plain and train_open
    assert highlight is not None
    # The output role is learned from every training output. Input roles are
    # legible independently as the dominant background and common object color.
    assert all(highlight not in palette for palette in train_palettes)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count,test_count = _resolve_counts(seed, rng, num_train, num_test)
    colors = [int(x) for x in rng.choice(10, size=3, replace=False)]
    base = (colors[0],colors[1])
    highlight = colors[2]
    varied = bool(rng.random() < 0.55)
    puzzle = {'train': [], 'test': []}
    seen = set()
    index = 0
    for split,count,regime in (('train',train_count,'source_train'), ('test',test_count,'target_test')):
        for pair_index in range(count):
            for _ in range(12):
                roles = _pair_roles(rng, highlight, base, varied, index)
                index += 1
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _generate_pair(child_rng, regime, pair_index, *roles, highlight)
                key = _canonical_grid(pair['input'])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise ValueError('could not construct unique puzzle pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k],b[k]) for a,b in zip(first[s],second[s]) for k in ('input','output'))
        for s in ('train','test'))


def self_test() -> None:
    a = generate_puzzle(1234)
    assert _puzzles_equal(a, generate_puzzle(1234))
    assert not _puzzles_equal(a, generate_puzzle(1235))
    for n,m in ((2,1),(10,1),(10,4),(7,3)):
        p = generate_puzzle(2000+n*10+m,n,m)
        assert len(p['train']) == n and len(p['test']) == m
        validate_generated_puzzle(p)
    for kwargs in ({'num_test':1},{'num_test':4},{'num_train':2},{'num_train':10}):
        validate_generated_puzzle(generate_puzzle(3000+len(kwargs)+sum(v for v in kwargs.values()),**kwargs))
    for kwargs in ({'num_train':1},{'num_train':11},{'num_test':0},{'num_test':5},
                   {'num_train':2,'num_test':2},{'num_train':3,'num_test':4}):
        try:
            generate_puzzle(4000,**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {kwargs}')
    train_counts,test_counts,shapes,layouts,palettes = set(),set(),set(),set(),set()
    edge_pairs, variable_puzzles, fixed_puzzles = 0,0,0
    large_open_pairs, nonrect_pairs, multiple_hole_pairs = 0,0,0
    selected_edge_pairs, open_edge_pairs, three_end_pairs = 0,0,0
    combined_pairs, varied_selection_puzzles, varied_shape_puzzles = 0,0,0
    selected_counts,open_counts = set(),set()
    small_tests,large_tests,zero_tests = 0,0,0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p['train'])); test_counts.add(len(p['test']))
        roles_seen, shapes_seen, selections_seen = set(),set(),set()
        for split in ('train','test'):
            for pair in p[split]:
                a = pair['input']
                roles, info, selected, preserved = _pair_profile(pair, _pair_profile(p['train'][0])[0][2])
                roles_seen.add(roles[:2]); palettes.add(roles)
                shapes_seen.add(a.shape); selections_seen.add(len(selected))
                shapes.add(a.shape); layouts.add(_canonical_grid(a))
                selected_counts.add(len(selected)); open_counts.add(len(preserved))
                if any(r in (0,a.shape[0]-1) or c in (0,a.shape[1]-1) for x in info for r,c in x['coords']):
                    edge_pairs += 1
                selected_edge_pairs += any(r in (0,a.shape[0]-1) or c in (0,a.shape[1]-1) for x in selected for r,c in x['coords'])
                open_edge_pairs += any(r in (0,a.shape[0]-1) or c in (0,a.shape[1]-1) for x in preserved for r,c in x['coords'])
                if selected and preserved and max(len(x['coords']) for x in preserved) >= min(len(x['coords']) for x in selected):
                    large_open_pairs += 1
                    combined_pairs += any(r in (0,a.shape[0]-1) or c in (0,a.shape[1]-1) for x in info for r,c in x['coords'])
                if any(not _is_plain_rectangle_ring(x['coords']) for x in selected):
                    nonrect_pairs += 1
                multiple_hole_pairs += any(x['hole_count'] >= 2 for x in selected)
                for item in selected:
                    points = set(item['coords'])
                    ends = sum(sum((r+dr,c+dc) in points for dr,dc in _DIRS) == 1 for r,c in points)
                    if ends >= 3:
                        three_end_pairs += 1
                        break
                if split == 'test':
                    small_tests += max(a.shape) <= 18
                    large_tests += min(a.shape) >= 22
                    zero_tests += not selected
        variable_puzzles += len(roles_seen) > 1
        fixed_puzzles += len(roles_seen) == 1
        varied_shape_puzzles += len(shapes_seen) > 1
        varied_selection_puzzles += len(selections_seen) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(layouts) >= 120 and len(palettes) >= 20
    assert edge_pairs >= 15 and variable_puzzles >= 8
    assert large_open_pairs >= 12 and nonrect_pairs >= 25
    assert len(selected_counts) >= 4 and len(open_counts) >= 3
    assert small_tests >= 3 and large_tests >= 5 and zero_tests >= 1
    assert fixed_puzzles >= 5 and varied_shape_puzzles >= 35
    assert varied_selection_puzzles >= 20
    assert selected_edge_pairs >= 20 and open_edge_pairs >= 20
    assert multiple_hole_pairs >= 20 and three_end_pairs >= 12
    assert combined_pairs >= 10
