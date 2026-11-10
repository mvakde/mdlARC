import numpy as np


# Rejecting an original input is stronger than merely rejecting an exact
# original input/output pair, since the intended output is deterministic.
_SOURCE_INPUT_CODES = frozenset({
    (
        "0696606369666990/"
        "9906600936669906/"
        "6090060660300600/"
        "9669996369966366/"
        "6600669003000009/"
        "9960090039300090/"
        "3644444600090009/"
        "9043304006009693/"
        "9049394900390093/"
        "6946604396060930/"
        "3349004906000600/"
        "0046394609090000/"
        "9944444990990006"
    ),
    (
        "388038808031118803838/"
        "330053038033811813183/"
        "151311830038301088550/"
        "530822222222221103003/"
        "013320080333320080331/"
        "800821000303120008010/"
        "115023303308121083100/"
        "008823351030021050301/"
        "010025130131128805038/"
        "833325080308823300338/"
        "111522222222220081300/"
        "333088083088303081010/"
        "800330830330133180038/"
        "515183508338180003005/"
        "131013105033808388800/"
        "533338801108513008310/"
        "313380380318318113810/"
        "038330130385303103008/"
        "383013801381011858311/"
        "151331533113508811808/"
        "130133100158353803838/"
        "313080800131108851018/"
        "331031880051881335358"
    ),
    (
        "250030020000003535/"
        "200202222225303205/"
        "055888888888888500/"
        "202800533322508255/"
        "503830055552058333/"
        "055835020305308025/"
        "522832550530508000/"
        "000852525022228205/"
        "505805250000338005/"
        "300823230050508320/"
        "350832505000558002/"
        "330888888888888020/"
        "500303352500000500/"
        "252522000520203030/"
        "022220020233202525/"
        "300005300022502320/"
        "002505030000233523"
    ),
    (
        "0008118008080008/"
        "0108810112811202/"
        "0088118811808001/"
        "1010801810118880/"
        "8083333333333302/"
        "1083208111010300/"
        "0883810828128318/"
        "1083820201181388/"
        "0803018811818321/"
        "1003018808020381/"
        "0883088288888388/"
        "1113802000088380/"
        "1803028812002381/"
        "8113333333333302/"
        "8100008801288818/"
        "8100118012810120/"
        "8082808201818188"
    ),
})


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validate_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        # This schedule guarantees broad count diversity over consecutive seeds.
        num_test = 1 + seed % 4
        span = 10 - num_test
        num_train = num_test + 1 + (seed // 4) % span
    elif num_train is None:
        assert num_test is not None
        span = 10 - num_test
        num_train = num_test + 1 + seed % span
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    assert num_train is not None
    assert num_test is not None

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _grid_code(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.dtype.str, grid.tobytes()


def _puzzle_key(puzzle: dict) -> tuple:
    result = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            result.append((
                split,
                _array_key(pair["input"]),
                _array_key(pair["output"]),
            ))
    return tuple(result)


def _find_frames(
    grid: np.ndarray,
    stop_after: int | None = None,
) -> list[tuple[int, int, int, int, int]]:
    """Find all monochromatic rectangular outlines with nonempty interiors."""
    height, width = grid.shape
    frames: list[tuple[int, int, int, int, int]] = []

    if height < 3 or width < 3:
        return frames

    for raw_color in np.unique(grid):
        color = int(raw_color)
        mask = grid == color

        row_prefix = np.concatenate(
            (
                np.zeros((height, 1), dtype=np.int16),
                np.cumsum(mask, axis=1, dtype=np.int16),
            ),
            axis=1,
        )
        column_prefix = np.concatenate(
            (
                np.zeros((1, width), dtype=np.int16),
                np.cumsum(mask, axis=0, dtype=np.int16),
            ),
            axis=0,
        )

        for top in range(height - 2):
            for bottom in range(top + 2, height):
                side_height = bottom - top + 1
                column_counts = (
                    column_prefix[bottom + 1] - column_prefix[top]
                )
                vertical_columns = np.flatnonzero(
                    column_counts == side_height
                )

                if vertical_columns.size < 2:
                    continue

                for left_index in range(vertical_columns.size - 1):
                    left = int(vertical_columns[left_index])

                    for right_index in range(
                        left_index + 1, vertical_columns.size
                    ):
                        right = int(vertical_columns[right_index])
                        if right - left < 2:
                            continue

                        side_width = right - left + 1
                        top_count = (
                            row_prefix[top, right + 1]
                            - row_prefix[top, left]
                        )
                        if top_count != side_width:
                            continue

                        bottom_count = (
                            row_prefix[bottom, right + 1]
                            - row_prefix[bottom, left]
                        )
                        if bottom_count != side_width:
                            continue

                        frames.append(
                            (top, bottom, left, right, color)
                        )
                        if (
                            stop_after is not None
                            and len(frames) >= stop_after
                        ):
                            return frames

    return frames



def _perimeter(frame):
    top, bottom, left, right, _ = frame
    return 2 * ((bottom - top) + (right - left))


def _select_marker(grid):
    """A complete border must dominate all incidental complete outlines.

    Comparable outlines are rejected, never resolved by position or color.
    The source's tiny 0 loop is subordinate to its large 8 border.
    """
    frames = _find_frames(grid, stop_after=200)
    if not frames or len(frames) == 200:
        return None
    frames.sort(key=_perimeter, reverse=True)
    if len(frames) > 1 and _perimeter(frames[0]) <= 2 * _perimeter(frames[1]):
        return None
    return frames[0]


def _sample_dimension(rng, high=28):
    bands = ((1, 3), (4, 10), (11, 20), (21, 28))
    weights = np.array((0.20, 0.35, 0.30, 0.15), dtype=float)
    available = [i for i, (low, _) in enumerate(bands) if low <= high]
    probabilities = weights[available] / sum(weights[available])
    band = bands[int(rng.choice(available, p=probabilities))]
    return int(rng.integers(band[0], min(band[1], high) + 1))


def _sample_shape(rng, placement, marker_mode):
    high = 26 if placement in ('interior', 'edge') else 28
    if marker_mode in ('exterior', 'both'):
        high = min(high, 22)
    if marker_mode in ('interior', 'both'):
        return (int(rng.integers(5, min(20, high) + 1)),
                int(rng.integers(5, min(20, high) + 1)))
    return _sample_dimension(rng, high), _sample_dimension(rng, high)


def _sample_extra(rng, maximum, minimum):
    if maximum < minimum:
        raise ValueError('layout has insufficient margin')
    if maximum == minimum:
        return minimum
    if rng.random() < 0.35:
        return maximum
    return int(rng.integers(minimum, maximum + 1))


def _choose_layout(rng, shape, placement):
    fh, fw = shape[0] + 2, shape[1] + 2
    mh, mw = 30 - fh, 30 - fw
    if placement == 'interior':
        eh = _sample_extra(rng, mh, 2)
        ew = _sample_extra(rng, mw, 2)
        top = int(rng.integers(1, eh))
        left = int(rng.integers(1, ew))
    elif placement == 'edge':
        eh = _sample_extra(rng, mh, 2)
        ew = _sample_extra(rng, mw, 2)
        side = int(rng.integers(0, 4))
        top = 0 if side == 0 else eh if side == 1 else int(rng.integers(1, eh))
        left = 0 if side == 2 else ew if side == 3 else int(rng.integers(1, ew))
    elif placement == 'corner':
        eh = _sample_extra(rng, mh, 0)
        ew = _sample_extra(rng, mw, 0)
        top = 0 if rng.random() < 0.5 else eh
        left = 0 if rng.random() < 0.5 else ew
    elif placement == 'span':
        if rng.random() < 0.5:
            eh = 0
            ew = _sample_extra(rng, mw, 0)
        else:
            ew = 0
            eh = _sample_extra(rng, mh, 0)
        top = 0 if eh == 0 else int(rng.integers(0, eh + 1))
        left = 0 if ew == 0 else int(rng.integers(0, ew + 1))
    elif placement == 'whole':
        eh = ew = top = left = 0
    else:
        raise ValueError('unknown placement')
    return fh + eh, fw + ew, top, top + fh - 1, left, left + fw - 1


def _sample_pair_palette(rng, frame_color):
    other = np.array([c for c in range(10) if c != frame_color], dtype=np.int8)
    if frame_color != 0 and rng.random() < 0.45:
        other = other[other != 0]
    k = int(rng.integers(1, min(len(other), 8) + 1))
    return (frame_color,) + tuple(int(x) for x in rng.choice(other, k, replace=False))


def _region_colors(rng, nonframe):
    chosen = tuple(c for c in nonframe if rng.random() < 0.72)
    return chosen or (int(rng.choice(nonframe)),)


def _texture(rng, height, width, colors):
    colors = np.asarray(colors, dtype=np.int8)
    n = len(colors)
    mode = int(rng.integers(0, 7))
    rows, cols = np.indices((height, width))
    if n == 1:
        grid = np.full((height, width), colors[0], dtype=np.int8)
    elif mode == 0:
        grid = rng.choice(colors, size=(height, width)).astype(np.int8)
    elif mode == 1:
        dominant = int(rng.integers(n))
        probability = float(rng.uniform(0.30, 0.68))
        probs = np.full(n, (1 - probability) / (n - 1))
        probs[dominant] = probability
        grid = rng.choice(colors, size=(height, width), p=probs).astype(np.int8)
    elif mode == 2:
        step_r = int(rng.integers(1, n))
        step_c = int(rng.integers(1, n))
        indices = (rows * step_r + cols * step_c + int(rng.integers(n))) % n
        grid = colors[indices].copy()
        mutations = rng.random((height, width)) < rng.uniform(0.05, 0.35)
        grid[mutations] = rng.choice(colors, size=int(mutations.sum()))
    elif mode == 3:
        bh, bw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        indices = (rows // bh + 2 * (cols // bw) + int(rng.integers(n))) % n
        grid = colors[indices].copy()
        mutations = rng.random((height, width)) < rng.uniform(0.04, 0.27)
        grid[mutations] = rng.choice(colors, size=int(mutations.sum()))
    elif mode == 4:
        grid = np.full((height, width), int(rng.choice(colors)), dtype=np.int8)
        count = int(rng.integers(2, max(3, (height * width) // 9 + 2)))
        for _ in range(count):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            rh, cw = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            grid[r:min(height, r + rh), c:min(width, c + cw)] = int(rng.choice(colors))
    elif mode == 5:
        grid = np.full((height, width), int(rng.choice(colors)), dtype=np.int8)
        flecks = rng.random((height, width)) < rng.uniform(0.03, 0.18)
        grid[flecks] = rng.choice(colors, size=int(flecks.sum()))
    else:
        grid = rng.choice(colors, size=(height, width)).astype(np.int8)
        for _ in range(int(rng.integers(2, 10))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            length = int(rng.integers(2, 9))
            color = int(rng.choice(colors))
            if rng.random() < 0.5:
                grid[r, c:min(width, c + length)] = color
            else:
                grid[r:min(height, r + length), c] = color
    return grid


def _add_fragments(rng, grid, colors):
    h, w = grid.shape
    for _ in range(int(rng.integers(0, 4))):
        if h < 2 or w < 2:
            break
        r, c = int(rng.integers(h)), int(rng.integers(w))
        rh = int(rng.integers(2, min(h, 8) + 1))
        cw = int(rng.integers(2, min(w, 8) + 1))
        color = int(rng.choice(colors))
        grid[r:min(h, r + rh), c] = color
        if rng.random() < 0.75:
            grid[r, c:min(w, c + cw)] = color
        if rng.random() < 0.45:
            grid[min(h - 1, r + rh - 1), c:min(w, c + cw)] = color


def _overlay_frame(grid, top, bottom, left, right, color):
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color


def _add_tiny_loop(rng, grid, bounds, colors):
    top, bottom, left, right = bounds
    if bottom - top < 6 or right - left < 6:
        return False
    r = int(rng.integers(top + 2, bottom - 2))
    c = int(rng.integers(left + 2, right - 2))
    color = int(rng.choice(colors))
    _overlay_frame(grid, r, r + 2, c, c + 2, color)
    return True


def _add_marker_speckles(rng, grid, bounds, color, mode):
    if mode == 'none':
        return True
    top, bottom, left, right = bounds
    h, w = grid.shape
    rows, cols = np.indices((h, w))
    inside = ((rows >= top + 2) & (rows <= bottom - 2) &
              (cols >= left + 2) & (cols <= right - 2))
    outside = ((rows < top - 1) | (rows > bottom + 1) |
               (cols < left - 1) | (cols > right + 1))
    outside &= ((rows < top) | (rows > bottom) |
                (cols < left) | (cols > right))
    for target in (('interior', inside), ('exterior', outside)):
        if mode not in (target[0], 'both'):
            continue
        eligible = np.argwhere(target[1])
        if not len(eligible):
            return False
        rng.shuffle(eligible)
        placed = 0
        for row, col in eligible:
            row, col = int(row), int(col)
            window = grid[max(0, row-1):min(h, row+2), max(0, col-1):min(w, col+2)]
            if np.any(window == color):
                continue
            grid[row, col] = color
            placed += 1
            if placed >= int(rng.integers(1, 4)):
                break
        if not placed:
            return False
    return True


def _make_pair(rng, regime, shape, placement, frame_color, marker_mode, loop_requested):
    if regime not in ('source_train', 'target_test'):
        raise ValueError('unknown regime')
    palette = _sample_pair_palette(rng, frame_color)
    nonframe = palette[1:]
    interior_colors = _region_colors(rng, nonframe)
    exterior_colors = _region_colors(rng, nonframe)
    for attempt in range(48):
        h, w, top, bottom, left, right = _choose_layout(rng, shape, placement)
        bounds = top, bottom, left, right
        grid = _texture(rng, h, w, exterior_colors)
        _add_fragments(rng, grid, exterior_colors)
        payload = _texture(rng, shape[0], shape[1], interior_colors)
        _add_fragments(rng, payload, interior_colors)
        grid[top + 1:bottom, left + 1:right] = payload
        if loop_requested:
            _add_tiny_loop(rng, grid, bounds, nonframe)
        _overlay_frame(grid, top, bottom, left, right, frame_color)
        if not _add_marker_speckles(rng, grid, bounds, frame_color, marker_mode):
            continue
        marker = _select_marker(grid)
        if marker == (top, bottom, left, right, frame_color):
            return {'input': grid, 'output': grid[top + 1:bottom, left + 1:right].copy()}
    # A deterministic checkerboard has no other complete monochromatic border.
    for _ in range(100):
        h, w, top, bottom, left, right = _choose_layout(rng, shape, placement)
        rows, cols = np.indices((h, w))
        fallback_colors = nonframe if len(nonframe) >= 2 else (
            nonframe[0], next(c for c in range(10) if c != nonframe[0] and c != frame_color))
        grid = np.asarray(fallback_colors, dtype=np.int8)[(rows + cols) % len(fallback_colors)].copy()
        _overlay_frame(grid, top, bottom, left, right, frame_color)
        if not _add_marker_speckles(rng, grid, (top, bottom, left, right), frame_color, marker_mode):
            continue
        if _select_marker(grid) == (top, bottom, left, right, frame_color):
            return {'input': grid, 'output': grid[top + 1:bottom, left + 1:right].copy()}
    raise RuntimeError('failed to construct salient frame')


def _observed_nonframe_colors(puzzle):
    result = set()
    for pair in puzzle['train']:
        frame = _select_marker(pair['input'])
        result.update(int(c) for c in np.unique(pair['input']) if int(c) != frame[4])
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Geometry and strict interior copying are the only shared convention.
    frame_order = tuple(int(c) for c in rng.permutation(np.arange(10)))
    train_colors = [frame_order[0], frame_order[1]]
    train_colors += [int(rng.choice(frame_order)) for _ in range(num_train - 2)]
    placements = ['interior', 'edge']
    placement_options = ('interior', 'edge', 'corner', 'span', 'whole')
    placements += [str(rng.choice(placement_options, p=(.25, .25, .2, .2, .1)))
                   for _ in range(num_train - 2)]
    assignments = list(zip(train_colors, placements))
    rng.shuffle(assignments)
    extension = str(rng.choice(('none', 'interior', 'exterior', 'both'), p=(.35, .22, .22, .21)))
    train_modes = ['none'] * num_train
    if extension != 'none':
        needed = ['interior', 'exterior'] if extension == 'both' else [extension]
        for i, mode in enumerate(needed):
            train_modes[i] = mode
    rng.shuffle(train_modes)
    puzzle = {'train': [], 'test': []}
    for i, ((color, placement), mode) in enumerate(zip(assignments, train_modes)):
        if mode in ('interior', 'both') and placement in ('span', 'whole'):
            placement = 'interior'
        if mode in ('exterior', 'both') and placement == 'whole':
            placement = 'edge'
        if mode != 'none' and placement == 'edge':
            shape = (int(rng.integers(5, 18)), int(rng.integers(5, 18)))
        else:
            shape = _sample_shape(rng, placement, mode)
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        pair = _make_pair(pair_rng, 'source_train', shape, placement, color,
                          mode, bool(rng.random() < .25))
        puzzle['train'].append(pair)
    train_frames = [_select_marker(p['input'])[4] for p in puzzle['train']]
    nonframe_seen = sorted(_observed_nonframe_colors(puzzle) - set(train_frames))
    train_shapes = [p['output'].shape for p in puzzle['train']]
    for i in range(num_test):
        role = str(rng.choice(('familiar', 'swap', 'new'), p=(.36, .44, .20)))
        if role == 'familiar':
            color = int(rng.choice(train_frames))
        elif role == 'swap' and nonframe_seen:
            color = int(rng.choice(nonframe_seen))
        else:
            color = int(rng.integers(0, 10))
        placement = str(rng.choice(placement_options, p=(.28, .25, .21, .21, .05)))
        mode = 'none' if extension == 'none' else str(rng.choice(('none', extension)))
        if mode in ('interior', 'both') and placement in ('span', 'whole'):
            placement = 'interior'
        if mode in ('exterior', 'both') and placement == 'whole':
            placement = 'edge'
        size_regime = str(rng.choice(('familiar', 'novel', 'large'), p=(.27, .43, .30)))
        if size_regime == 'familiar':
            eligible = [s for s in train_shapes if placement != 'interior' or max(s) <= 26]
            shape = tuple(eligible[int(rng.integers(len(eligible)))]) if eligible else _sample_shape(rng, placement, mode)
        elif size_regime == 'large' and mode == 'none' and placement != 'interior':
            shape = (_sample_dimension(rng, 28), int(rng.integers(20, 29)))
            if rng.random() < .5:
                shape = shape[::-1]
        else:
            shape = _sample_shape(rng, placement, mode)
        if mode in ('interior', 'both'):
            shape = (max(5, min(24, shape[0])), max(5, min(24, shape[1])))
        if mode in ('exterior', 'both'):
            shape = (min(22, shape[0]), min(22, shape[1]))
        if placement == 'edge':
            shape = (min(26, shape[0]), min(26, shape[1]))
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        puzzle['test'].append(_make_pair(pair_rng, 'target_test', shape, placement,
                                         color, mode, bool(rng.random() < .35)))
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    grid, output = pair['input'], pair['output']
    marker = _select_marker(grid)
    assert marker is not None
    top, bottom, left, right, color = marker
    assert np.array_equal(output, grid[top + 1:bottom, left + 1:right])
    return {'marker': marker, 'palette': frozenset(int(x) for x in np.unique(grid)),
            'shape': output.shape,
            'boundary': top == 0 or left == 0 or bottom == grid.shape[0] - 1 or right == grid.shape[1] - 1,
            'corner': (top == 0 or bottom == grid.shape[0] - 1) and (left == 0 or right == grid.shape[1] - 1),
            'span': top == 0 and bottom == grid.shape[0] - 1 or left == 0 and right == grid.shape[1] - 1,
            'frame_interior': bool(np.any(output == color)),
            'frame_exterior': bool(np.any(grid[:top] == color) or np.any(grid[bottom + 1:] == color) or
                                   np.any(grid[top:bottom + 1, :left] == color) or
                                   np.any(grid[top:bottom + 1, right + 1:] == color))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            key = _array_key(pair['input'])
            assert key not in seen
            seen.add(key)
            assert pair['input'].shape != pair['output'].shape
            assert _grid_code(pair['input']) not in _SOURCE_INPUT_CODES
            analyses[split].append(_analyze_pair(pair))
    assert len({a['marker'][4] for a in analyses['train']}) >= 2
    assert any(a['boundary'] for a in analyses['train'])
    assert any(not a['boundary'] for a in analyses['train'])
    train_inside = any(a['frame_interior'] for a in analyses['train'])
    train_outside = any(a['frame_exterior'] for a in analyses['train'])
    assert all(not a['frame_interior'] or train_inside for a in analyses['test'])
    assert all(not a['frame_exterior'] or train_outside for a in analyses['test'])


def self_test() -> None:
    # The original third training input contains a subordinate 0 loop.
    source = next(code for code in _SOURCE_INPUT_CODES if code.startswith('2500300'))
    source_grid = np.array([[int(c) for c in row] for row in source.split('/')], dtype=np.int8)
    assert len(_find_frames(source_grid)) > 1
    assert _select_marker(source_grid) == (2, 11, 3, 14, 8)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    for seed, nt, ne in ((7001, 2, 1), (7002, 10, 1), (7003, 10, 4),
                         (8101, None, 1), (8102, None, 4), (8103, 2, None), (8104, 10, None)):
        p = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(p)
        assert nt is None or len(p['train']) == nt
        assert ne is None or len(p['test']) == ne
    for args in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                 {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(9000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    all_pairs = [pair for p in puzzles for split in ('train', 'test') for pair in p[split]]
    analyses = [_analyze_pair(pair) for pair in all_pairs]
    train_analyses = [_analyze_pair(pair) for p in puzzles for pair in p['train']]
    test_analyses = [_analyze_pair(pair) for p in puzzles for pair in p['test']]
    assert len({a['shape'] for a in analyses}) >= 100
    assert len({pair['input'].shape for pair in all_pairs}) >= 100
    assert len({tuple(sorted(a['palette'])) for a in analyses}) >= 40
    assert any(0 not in a['palette'] for a in analyses)
    assert any(a['shape'][0] % 2 != a['shape'][1] % 2 for a in train_analyses)
    assert any(min(a['shape']) == 1 for a in analyses)
    assert any(a['boundary'] for a in test_analyses)
    assert any(a['corner'] for a in test_analyses)
    assert any(a['span'] for a in analyses)
    assert any(a['frame_interior'] for a in train_analyses)
    assert any(a['frame_exterior'] for a in train_analyses)
    assert any(a['frame_interior'] for a in test_analyses)
    assert any(a['frame_exterior'] for a in test_analyses)
    assert all(len({_analyze_pair(q)['palette'] for s in ('train', 'test')
                    for q in p[s]}) > 1 for p in puzzles)
    assert any(a['shape'] == (1, 1) for a in analyses)
    assert any(a['shape'][0] <= 3 and a['shape'][1] >= 12 for a in analyses)
    assert any(a['shape'][1] <= 3 and a['shape'][0] >= 12 for a in analyses)
    assert any(min(a['shape']) >= 20 for a in analyses)
    assert any(pair['input'].shape == (a['shape'][0] + 2, a['shape'][1] + 2)
               for pair, a in zip(all_pairs, analyses))
    assert sum(len(_find_frames(pair['input'], stop_after=200)) > 1
               for pair in all_pairs) >= 10
    assert any(not a['frame_interior'] and not a['frame_exterior'] for a in analyses)
    assert any(a['frame_interior'] and a['corner'] for a in test_analyses)
    familiar_shape_tests = 0
    familiar_color_tests = 0
    swapped_color_tests = 0
    payload_only = 0
    noise_only = 0
    for p in puzzles:
        train_shapes = {q['output'].shape for q in p['train']}
        train_colors = {_analyze_pair(q)['marker'][4] for q in p['train']}
        train_nonframe = set().union(*(
            set(_analyze_pair(q)['palette']) - {_analyze_pair(q)['marker'][4]}
            for q in p['train']))
        for q in p['test']:
            analysis = _analyze_pair(q)
            familiar_shape_tests += analysis['shape'] in train_shapes
            familiar_color_tests += analysis['marker'][4] in train_colors
            swapped_color_tests += analysis['marker'][4] in train_nonframe - train_colors
        for split in ('train', 'test'):
            for q in p[split]:
                top, bottom, left, right, color = _analyze_pair(q)['marker']
                grid = q['input']
                exterior = np.concatenate((
                    grid[:top].ravel(), grid[bottom + 1:].ravel(),
                    grid[top:bottom + 1, :left].ravel(),
                    grid[top:bottom + 1, right + 1:].ravel()))
                outside_colors = set(int(v) for v in np.unique(exterior))
                inside_colors = set(int(v) for v in np.unique(q['output']))
                payload_only += bool(inside_colors - outside_colors - {color})
                noise_only += bool(outside_colors - inside_colors - {color})
    assert familiar_shape_tests >= 5
    assert familiar_color_tests >= 5 and swapped_color_tests >= 5
    assert payload_only >= 20 and noise_only >= 20
