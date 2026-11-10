"""Guide-defined extraction with local color correspondence under all eight symmetries."""


import numpy as np


_SIDES = ("T", "R", "B", "L")


_ADJACENT_SIDE_PAIRS = {frozenset((_SIDES[i], _SIDES[(i + 1) % 4])) for i in range(4)}


_SIDE_MAPS = {
    "identity": {
        "T": "T", "R": "R", "B": "B", "L": "L",
    },
    "r90": {
        "T": "L", "R": "T", "B": "R", "L": "B",
    },
    "r180": {
        "T": "B", "R": "L", "B": "T", "L": "R",
    },
    "r270": {
        "T": "R", "R": "B", "B": "L", "L": "T",
    },
    "flip_lr": {
        "T": "T", "R": "L", "B": "B", "L": "R",
    },
    "flip_ud": {
        "T": "B", "R": "R", "B": "T", "L": "L",
    },
    "transpose": {
        "T": "L", "R": "B", "B": "R", "L": "T",
    },
    "anti_transpose": {
        "T": "R", "R": "T", "B": "L", "L": "B",
    },
}


_AXIS_SWAPPING_TRANSFORMS = {
    "r90", "r270", "transpose", "anti_transpose",
}


def _validate_optional_count(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer count")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer count")


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_optional_count("num_train", num_train)
    _validate_optional_count("num_test", num_test)

    if num_train is not None:
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        token = seed + seed // 4
        num_train = minimum_train + token % span
    elif num_train is None:
        assert num_test is not None
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + seed % span
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    assert num_train is not None
    assert num_test is not None

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _apply_transform(array: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        result = array
    elif name == "r90":
        result = np.rot90(array, 1)
    elif name == "r180":
        result = np.rot90(array, 2)
    elif name == "r270":
        result = np.rot90(array, 3)
    elif name == "flip_lr":
        result = np.fliplr(array)
    elif name == "flip_ud":
        result = np.flipud(array)
    elif name == "transpose":
        result = array.T
    elif name == "anti_transpose":
        result = np.rot90(array.T, 2)
    else:
        raise AssertionError(f"unknown transformation: {name}")

    return np.asarray(result).copy()


def _side_values(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "T":
        return grid[0, :]
    if side == "R":
        return grid[:, -1]
    if side == "B":
        return grid[-1, :]
    if side == "L":
        return grid[:, 0]
    raise AssertionError(f"unknown side: {side}")


def _write_side(grid: np.ndarray, side: str, color: int) -> None:
    if side == "T":
        grid[0, :] = color
    elif side == "R":
        grid[:, -1] = color
    elif side == "B":
        grid[-1, :] = color
    elif side == "L":
        grid[:, 0] = color
    else:
        raise AssertionError(f"unknown side: {side}")


def _paint_guide(
    grid: np.ndarray,
    side: str,
    color: int,
    row: int,
    col: int,
    height: int,
    width: int,
) -> None:
    if side == "T":
        grid[row - 1, col:col + width] = color
    elif side == "R":
        grid[row:row + height, col + width] = color
    elif side == "B":
        grid[row + height, col:col + width] = color
    elif side == "L":
        grid[row:row + height, col - 1] = color
    else:
        raise AssertionError(f"unknown side: {side}")


def _color_components(
    grid: np.ndarray,
    color: int,
) -> list:
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []

    rows, cols = np.where(grid == color)
    for start_row, start_col in zip(rows, cols):
        start = (int(start_row), int(start_col))
        if seen[start]:
            continue

        stack = [start]
        seen[start] = True
        component = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))

            for delta_row, delta_col in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ):
                next_row = row + delta_row
                next_col = col + delta_col

                if not (
                    0 <= next_row < height
                    and 0 <= next_col < width
                ):
                    continue
                if seen[next_row, next_col]:
                    continue
                if int(grid[next_row, next_col]) != color:
                    continue

                seen[next_row, next_col] = True
                stack.append((next_row, next_col))

        components.append(component)

    return components


def _straight_component_description(
    component: list,
    grid_shape: tuple[int, int],
    color: int,
) -> dict | None:
    height, width = grid_shape

    if any(
        row == 0
        or row == height - 1
        or col == 0
        or col == width - 1
        for row, col in component
    ):
        return None

    rows = sorted({row for row, _ in component})
    cols = sorted({col for _, col in component})

    if len(rows) == 1 and len(component) >= 4:
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        return {
            "orientation": "H",
            "row": rows[0],
            "start": cols[0],
            "end": cols[-1],
            "length": len(cols),
            "color": color,
        }

    if len(cols) == 1 and len(component) >= 4:
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return {
            "orientation": "V",
            "col": cols[0],
            "start": rows[0],
            "end": rows[-1],
            "length": len(rows),
            "color": color,
        }

    return None


def _extract_guides_and_payload(
    grid: np.ndarray,
    marker_colors: frozenset,
) -> tuple[dict, tuple[int, int, int, int]]:
    descriptions = []

    for color in sorted(marker_colors):
        straight_components = []

        for component in _color_components(grid, int(color)):
            description = _straight_component_description(
                component,
                grid.shape,
                int(color),
            )
            if description is not None:
                straight_components.append(description)

        assert len(straight_components) == 1
        descriptions.append(straight_components[0])

    horizontal = [
        item
        for item in descriptions
        if item["orientation"] == "H"
    ]
    vertical = [
        item
        for item in descriptions
        if item["orientation"] == "V"
    ]

    assert len(horizontal) == 1
    assert len(vertical) == 1

    horizontal = horizontal[0]
    vertical = vertical[0]

    h_row = int(horizontal["row"])
    h_start = int(horizontal["start"])
    h_end = int(horizontal["end"])
    v_col = int(vertical["col"])
    v_start = int(vertical["start"])
    v_end = int(vertical["end"])

    payload_height = int(vertical["length"])
    payload_width = int(horizontal["length"])
    matches = []

    # Top and left guides.
    if v_start == h_row + 1 and v_col == h_start - 1:
        matches.append((
            h_row + 1,
            h_start,
            payload_height,
            payload_width,
            {
                "T": int(horizontal["color"]),
                "L": int(vertical["color"]),
            },
        ))

    # Top and right guides.
    if v_start == h_row + 1 and v_col == h_end + 1:
        matches.append((
            h_row + 1,
            h_start,
            payload_height,
            payload_width,
            {
                "T": int(horizontal["color"]),
                "R": int(vertical["color"]),
            },
        ))

    # Bottom and left guides.
    if v_end == h_row - 1 and v_col == h_start - 1:
        matches.append((
            v_start,
            h_start,
            payload_height,
            payload_width,
            {
                "B": int(horizontal["color"]),
                "L": int(vertical["color"]),
            },
        ))

    # Bottom and right guides.
    if v_end == h_row - 1 and v_col == h_end + 1:
        matches.append((
            v_start,
            h_start,
            payload_height,
            payload_width,
            {
                "B": int(horizontal["color"]),
                "R": int(vertical["color"]),
            },
        ))

    assert len(matches) == 1
    row, col, height, width, guide_by_side = matches[0]
    assert frozenset(guide_by_side) in _ADJACENT_SIDE_PAIRS

    return guide_by_side, (
        int(row),
        int(col),
        int(height),
        int(width),
    )


def _adjacent_payload_segment(
    grid: np.ndarray,
    side: str,
    row: int,
    col: int,
    height: int,
    width: int,
) -> np.ndarray:
    if side == "T":
        return grid[row - 1, col:col + width]
    if side == "R":
        return grid[row:row + height, col + width]
    if side == "B":
        return grid[row + height, col:col + width]
    if side == "L":
        return grid[row:row + height, col - 1]
    raise AssertionError(f"unknown side: {side}")


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = grid.astype(np.uint8, copy=False)
    return grid.shape, normalized.tobytes()


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != set(second):
        return False

    for split_name in ("train", "test"):
        if len(first[split_name]) != len(second[split_name]):
            return False

        for pair_a, pair_b in zip(
            first[split_name],
            second[split_name],
        ):
            if set(pair_a) != set(pair_b):
                return False
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False
            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


# Complete source contents, used only for exact-copy exclusion; dimensions remain legal.
_SOURCE_CONTENTS = ((((22, 22), '4444444444444444444444604000440000000400400560000000000000000000006000000000000000000005600011111110111400000566000111111811140000006600111111181114000005600000111118111400000060000011111811140000006000111111181114000005660011111118111400000060001111111811140000056000111111181114000000600011111228221400000060001111121111140000006000188888111314000000660011111211131400000560001111121333140000056600111111111114000000600066666666666000000060000000000000000000056055000505055000550055'), ((11, 15), '111111111111111133312111111111131112111111111131118888888880111112111111111122822111111111111811111111111111811111111111111811111111111111811111100111111111111100101')), (((22, 22), '5505005000550500050551000000000000000000000150000000000000000000110000000000000000000001500000000000000000000100000000000000000000115003222222222200000001500324222228820000000100032222222222000000015003222222222200000001000322222222220000000100032222222222000000115003242444244200000011000324242424220000000100032444244422000000010003222222222200000011000011111111110000000150000000000000000000010000000000000000000001500000000000000000000103000330000300030330013333333333333333333333'), ((10, 10), '2222222222282222422228222244422222222242222222444222222242222222224442222222224224222244422222222222')), (((22, 22), '2005050055005000505005220000000000000000000020000000000000000000052000000000000000000000200022222000000000000020048888800000000000052004898880000000000005200489888000000000000022048366000000000000052204888800000000000000200000000000000000000020000000000000000000052000000000000000000005220000000000000000000020000000000000000000002000000000000000000005200000000000000000000020000000000000000000052200000000000000000000200000000400000000000520044040044004000440454444444444444444444444'), ((5, 5), '8880088868888688993888888')), (((29, 29), '2222222222222222222222222222220200202220202022020002022222500000000200000000000000002030000000000000000000000000000350024700000446444444000400033000247400044464666664744000330002474444444646444647400000350024744444446444446474000003500247464666664444464744000330002474664444444446647440000350024744444444444444474000033000247777777777777777744000030002444444444444444444440000300024889494949494949494400033500248894949494949494944000335002444444444444444444440000300003333333333333333333300003500000000000000000000000000330000000000000000000000000003350000000000000000000000000003500000000000000000000000000035500000000000000000000000000350000000000000000000000000003500000000000000000000000000330000000000000000000000000033350000000000000000000000000033500000000000000000000000000030000000000000000000000000003350550505550050550050050055003'), ((20, 12), '444444444444777777774884044444474884004466474994004446474444004464474994044464474444444464474994444464474444666664474994444444474444466444474994464444474444464444474994464446474444466666474994044444474444077777774994044444444444440044044444')))


_SOURCE_KEYS = {tuple((shape, bytes(int(c) for c in cells)) for shape, cells in pair) for pair in _SOURCE_CONTENTS}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _palette(rng):
    colors = [int(c) for c in rng.permutation(10)]
    return dict(background=colors[0], markers=colors[1:3], fill=colors[3],
                noise=colors[4], motifs=colors[5:5 + int(rng.integers(1, 6))])


def sample_puzzle_parameters(rng):
    # These are sampling regimes, never a hidden interpretation codebook.
    return dict(regime=_pick(rng, ('mixed', 'mixed', 'familiar', 'held_outer',
                                   'held_combination', 'source_shift')),
                stable=bool(rng.random() < .18), palette=_palette(rng),
                blank_edges=bool(rng.random() < .75),
                detail=float(rng.uniform(.15, 1)),
                held_outer=int(rng.integers(4)), held_inner=int(rng.integers(4)))


def _adjacent(index):
    return (_SIDES[index], _SIDES[(index + 1) % 4])


def _arrangement(outer, inner, reverse):
    targets = outer[::-1] if reverse else outer
    mapping = dict(zip(inner, targets))
    matches = [name for name, m in _SIDE_MAPS.items()
               if all(m[side] == target for side, target in mapping.items())]
    assert len(matches) == 1
    return matches[0]


def _sample_shape(rng, regime, split):
    if regime == 'source_shift':
        if split == 'source_train':
            return int(rng.integers(4, 11)), int(rng.integers(4, 12))
        if rng.random() < .25:
            return 12, 20
        return int(rng.integers(12, 23)), int(rng.integers(12, 23))
    if rng.random() < .14:
        return _pick(rng, ((5, 5), (10, 10), (11, 15), (20, 12), (4, 7)))
    low, high = _pick(rng, ((4, 9), (6, 17), (12, 26)))
    h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
    aspect = int(rng.integers(4))
    if aspect == 0:
        w = h
    elif aspect == 1:
        h = int(rng.integers(4, min(w + 1, 10)))
    elif aspect == 2:
        w = int(rng.integers(4, min(h + 1, 10)))
    return h, w


def _asymmetric(payload):
    return len({_grid_key(_apply_transform(payload, t)) for t in _SIDE_MAPS}) == 8


def _make_payload(rng, shape, palette, blank_sides, witness, detail):
    h, w = shape
    bg, fill, colors = palette['background'], palette['fill'], palette['motifs']
    allowed = np.ones(shape, dtype=bool)
    blank_sides = list(blank_sides)
    if blank_sides and min(shape) >= 8 and rng.random() < .25:
        blank_sides.append(str(_pick(rng, [s for s in _SIDES if s not in blank_sides])))
    for side in blank_sides:
        thickness = 2 if min(shape) >= 10 and rng.random() < .3 else 1
        if side == 'T':
            allowed[:thickness, :] = False
        elif side == 'B':
            allowed[-thickness:, :] = False
        elif side == 'L':
            allowed[:, :thickness] = False
        else:
            allowed[:, -thickness:] = False
    # Corners are ordinary content, sometimes deliberately all fill or missing.
    corner_mode = int(rng.integers(3))
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    motif_allowed = allowed.copy()
    if corner_mode == 1:
        for pos in corners:
            motif_allowed[pos] = False
    positions = [tuple(int(v) for v in p) for p in np.argwhere(motif_allowed)]
    density = float(rng.uniform(.06, .18 + .27 * detail))
    budget = max(len(colors), int(len(positions) * density))
    payload = np.full(shape, fill, dtype=np.int8)
    # Ordered strokes retain bends/patches/repeats even when a density budget
    # truncates the last operation. Each operation has fresh local geometry.
    strokes = []
    styles = rng.permutation(5)
    for index in range(int(rng.integers(1, 5))):
        style = int(styles[index])
        color = int(_pick(rng, colors))
        r, c = _pick(rng, positions)
        cells = []
        if style == 0:  # Off-center line, often with a perpendicular bend.
            dr, dc = _pick(rng, ((1, 0), (-1, 0), (0, 1), (0, -1)))
            for _ in range(int(rng.integers(2, max(h, w) + 1))):
                cells.append((r, c)); r += dr; c += dc
            r -= dr; c -= dc
            dr, dc = dc, dr
            for _ in range(int(rng.integers(1, max(h, w) // 2 + 2))):
                r += dr; c += dc; cells.append((r, c))
        elif style == 1:  # Connected patch with random branches.
            cells = [(r, c)]
            for _ in range(int(rng.integers(3, max(5, h * w // 3)))):
                r, c = _pick(rng, cells)
                dr, dc = _pick(rng, ((1, 0), (-1, 0), (0, 1), (0, -1)))
                cells.append((min(h - 1, max(0, r + dr)),
                              min(w - 1, max(0, c + dc))))
        elif style == 2:  # Repeated teeth, with a displaced local detail.
            horizontal = bool(rng.integers(2))
            step = int(rng.integers(2, 5))
            for t in range(int(rng.integers(step)), w if horizontal else h, step):
                for depth in range(int(rng.integers(1, 4))):
                    cells.append((r + depth, t) if horizontal else (t, c + depth))
            cells.append((r - 1, c + 1))
        elif style == 3:  # Scattered pattern.
            cells = [positions[int(i)] for i in rng.permutation(len(positions))[:budget]]
        else:  # Incomplete rectangular outline.
            rr, cc = int(rng.integers(h)), int(rng.integers(w))
            r0, r1 = sorted((r, rr)); c0, c1 = sorted((c, cc))
            cells = ([(r0, x) for x in range(c0, c1 + 1)] +
                     [(x, c1) for x in range(r0, r1 + 1)] +
                     [(r1, x) for x in range(c0 + 1, c1)])
        strokes.extend((pos, color) for pos in cells
                       if 0 <= pos[0] < h and 0 <= pos[1] < w and motif_allowed[pos])
    # Bounded completion also makes one-color, sparse motifs available.
    strokes.extend((positions[int(i)], int(_pick(rng, colors)))
                   for i in rng.permutation(len(positions)))
    painted = set()
    for pos, color in strokes:
        if len(painted) >= budget:
            break
        payload[pos] = color
        painted.add(pos)
    # Ensure the sampled motif palette is visible without any fixed anchors.
    chosen = sorted(painted)
    for color, index in zip(colors, rng.permutation(len(chosen))):
        payload[chosen[int(index)]] = color
    if rng.random() < .5:
        available = [p for p in positions if p not in painted]
        for i in rng.permutation(len(available))[:int(len(available) * rng.uniform(.01, .09))]:
            payload[available[int(i)]] = bg
    if corner_mode == 2 and rng.random() < .6:
        for i in rng.permutation(4)[:int(rng.integers(1, 4))]:
            payload[corners[int(i)]] = bg
    payload[~allowed] = bg
    if not witness and not blank_sides and rng.random() < .4:
        # Once training establishes the rule, an occasional simpler symmetric
        # motif is legitimate; its guides still uniquely encode orientation.
        if rng.random() < .5:
            payload[:, w // 2:] = np.fliplr(payload)[:, w // 2:]
        else:
            payload[h // 2:, :] = np.flipud(payload)[h // 2:, :]
    for side in _SIDES:
        if side not in blank_sides and np.all(_side_values(payload, side) == bg):
            # Incidental holes must not silently introduce an untrained crop
            # extension in a puzzle sampled without blank-edge demonstrations.
            line = _side_values(payload, side)
            line[int(rng.integers(1, len(line) - 1))] = fill
    if witness and not _asymmetric(payload):
        # A unique accent away from every reflection axis breaks residual
        # symmetry. Its position is sampled, never a protected corner code.
        accent = colors[-1]
        payload[payload == accent] = fill
        candidates = [p for p in positions if 2 * p[0] != h - 1
                      and 2 * p[1] != w - 1
                      and (h != w or (p[0] != p[1] and p[0] + p[1] != h - 1))]
        payload[_pick(rng, candidates)] = accent
    assert not witness or _asymmetric(payload)
    return payload


def _edge_coordinate(shape, side, position, depth=0):
    h, w = shape
    return {'T': (depth, position), 'R': (position, w - 1 - depth),
            'B': (h - 1 - depth, position), 'L': (position, depth)}[side]


def _paint_frame(rng, grid, outer, palette, protected, detail):
    bg, noise = palette['background'], palette['noise']
    decoration = _pick(rng, ('clean', 'light', 'regular', 'irregular', 'busy'))
    if detail < .3:
        decoration = _pick(rng, ('clean', 'light', 'regular'))
    for side in _SIDES:
        n = len(_side_values(grid, side))
        if side not in outer and decoration != 'clean':
            if decoration == 'regular':
                step = int(rng.integers(2, 7))
                marks = np.arange(n) % step == int(rng.integers(step))
            else:
                probability = {'light': .12, 'irregular': .38, 'busy': .78}[decoration]
                marks = rng.random(n) < probability
            # Even >50% occupied borders stay fragmented, with run length <=3.
            run = 0
            for p in range(1, n - 1):
                run = run + 1 if marks[p] else 0
                if run >= 4:
                    run = 0; marks[p] = False
                if marks[p]:
                    grid[_edge_coordinate(grid.shape, side, p)] = noise
    for side in rng.permutation(list(outer)):
        _write_side(grid, str(side), outer[str(side)])
    if decoration == 'clean':
        return
    for side, color in outer.items():
        n = len(_side_values(grid, side))
        for p in rng.permutation(np.arange(2, n - 2))[:int(rng.integers(1, max(2, n // 3)))]:
            depth = int(rng.integers(1, 4))
            for d in range(1, depth + 1):
                pos = _edge_coordinate(grid.shape, side, int(p), d)
                if protected[pos] or grid[pos] not in (bg, color):
                    break
                grid[pos] = color
                if rng.random() < .25:
                    shoulder = _edge_coordinate(grid.shape, side, int(p) + 1, d)
                    if not protected[shoulder] and grid[shoulder] == bg:
                        grid[shoulder] = color
        # Optional single interference pixel; complete spans still dominate.
        if decoration in ('irregular', 'busy') and rng.random() < .35:
            p = int(rng.integers(2, n - 2))
            grid[_edge_coordinate(grid.shape, side, p)] = noise
    if decoration in ('busy', 'irregular', 'light'):
        # Small nearby specks can be inward of the outermost edge.
        for _ in range(int(rng.integers(1, 9))):
            side = str(_pick(rng, _SIDES))
            n = len(_side_values(grid, side))
            pos = _edge_coordinate(grid.shape, side, int(rng.integers(1, n - 1)),
                                   int(rng.integers(1, 4)))
            if not protected[pos] and grid[pos] == bg:
                grid[pos] = noise


def _generate_pair(rng, split, parameters, outer_index, inner_index, reverse,
                   blank_sides=(), witness=False, palette=None):
    assert split in ('source_train', 'target_test')
    palette = palette or (parameters['palette'] if parameters['stable'] else _palette(rng))
    inner, outer = _adjacent(inner_index), _adjacent(outer_index)
    transform = _arrangement(outer, inner, reverse)
    h, w = _sample_shape(rng, parameters['regime'], split)
    H, W = int(rng.integers(h + 5, 31)), int(rng.integers(w + 5, 31))
    canvas_mode = float(rng.random())
    if canvas_mode < .10:
        H = W = 30
    elif canvas_mode < .28:
        H = W = max(H, W)
    row = int(rng.integers(2 + ('T' in inner), H - h - 1 - ('B' in inner)))
    col = int(rng.integers(2 + ('L' in inner), W - w - 1 - ('R' in inner)))
    payload = _make_payload(rng, (h, w), palette, blank_sides, witness, parameters['detail'])
    grid = np.full((H, W), palette['background'], dtype=np.int8)
    guides = dict(zip(inner, palette['markers']))
    controls = {_SIDE_MAPS[transform][side]: color for side, color in guides.items()}
    # Reserve the payload, both guides, and a full one-cell background moat.
    protected = np.zeros(grid.shape, dtype=bool)
    r0, r1 = row - ('T' in inner), row + h + ('B' in inner)
    c0, c1 = col - ('L' in inner), col + w + ('R' in inner)
    protected[r0 - 1:r1 + 1, c0 - 1:c1 + 1] = True
    _paint_frame(rng, grid, controls, palette, protected, parameters['detail'])
    grid[row:row + h, col:col + w] = payload
    for side, color in guides.items():
        _paint_guide(grid, side, color, row, col, h, w)
    return {'input': grid, 'output': _apply_transform(payload, transform)}


def _longest_run(values, color):
    best = run = 0
    for value in values:
        run = run + 1 if int(value) == color else 0
        best = max(best, run)
    return best


def _decode_input(grid):
    # First find near-complete outer lines, without assuming the globally most
    # frequent color is the background (large payloads can dominate a canvas).
    candidates = {}
    for side in _SIDES:
        core = _side_values(grid, side)[1:-1]
        values, counts = np.unique(core, return_counts=True)
        for color, count in zip(values, counts):
            if count >= len(core) - 1 and _longest_run(core, int(color)) >= max(4, len(core) // 2):
                candidates[side] = int(color)
    matches = []
    for index in range(4):
        sides = _adjacent(index)
        if any(s not in candidates for s in sides):
            continue
        outer = {s: candidates[s] for s in sides}
        if len(set(outer.values())) != 2:
            continue
        try:
            guides, bounds = _extract_guides_and_payload(grid, frozenset(outer.values()))
            r, c, h, w = bounds
            assert r >= 2 and c >= 2 and r + h <= grid.shape[0] - 2 and c + w <= grid.shape[1] - 2
            payload = grid[r:r + h, c:c + w].copy()
            assert not set(np.unique(payload)) & set(outer.values())
            r0, r1 = r - ('T' in guides), r + h + ('B' in guides)
            c0, c1 = c - ('L' in guides), c + w + ('R' in guides)
            ring = np.concatenate((grid[r0 - 1, c0 - 1:c1 + 1], grid[r1, c0 - 1:c1 + 1],
                                   grid[r0:r1, c0 - 1], grid[r0:r1, c1]))
            assert len(np.unique(ring)) == 1
            bg = int(ring[0])
            assert bg not in outer.values()
            for side in _SIDES:
                segment = _adjacent_payload_segment(grid, side, *bounds)
                assert np.all(segment == guides.get(side, bg))
            # Only the one rectangle may contain payload colors. Noise remains
            # near the frame and cannot define an alternative payload.
            outside = np.ones(grid.shape, dtype=bool)
            outside[r:r + h, c:c + w] = False
            assert not set(np.unique(grid[outside])) & (set(np.unique(payload)) - {bg})
            for color in set(np.unique(grid[outside])) - set(outer.values()) - {bg}:
                rr, cc = np.where((grid == color) & outside)
                assert np.all(np.minimum.reduce((rr, cc, grid.shape[0] - 1 - rr,
                                                 grid.shape[1] - 1 - cc)) <= 3)
            transforms = [t for t, mapping in _SIDE_MAPS.items()
                          if all(outer.get(mapping[s]) == color for s, color in guides.items())]
            assert len(transforms) == 1
            matches.append(dict(background=bg, outer_by_side=outer, guide_by_side=guides,
                                bounds=bounds, payload=payload, transform=transforms[0]))
        except AssertionError:
            continue
    assert len(matches) == 1, 'input must have exactly one complete adjacent control/guide correspondence'
    return matches[0]


def _coordinate_prediction(info):
    # Independent of the named array-transform implementation: form the two
    # signed image axes from corresponding side normals, then move every cell.
    normals = {'T': (-1, 0), 'R': (0, 1), 'B': (1, 0), 'L': (0, -1)}
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in info['guide_by_side'].items():
        target = next(s for s, c in info['outer_by_side'].items() if c == color)
        matrix += np.outer(normals[target], normals[side])
    payload = info['payload']
    positions = np.indices(payload.shape).reshape(2, -1)
    target = matrix @ positions
    target -= target.min(axis=1, keepdims=True)
    result = np.empty(tuple(target.max(axis=1) + 1), dtype=payload.dtype)
    result[target[0], target[1]] = payload.ravel()
    return result


def solve(train_pairs, test_input):
    """Predict from visible grids only; no test labels or generation parameters."""
    for pair in train_pairs:
        assert np.array_equal(_coordinate_prediction(_decode_input(pair['input'])), pair['output'])
    return _coordinate_prediction(_decode_input(test_input))


def _pair_measurements(pair):
    info = _decode_input(pair['input'])
    payload, bg = info['payload'], info['background']
    values, counts = np.unique(payload[payload != bg], return_counts=True)
    fill = int(values[np.argmax(counts)])
    blank = tuple(s for s in _SIDES if np.all(_side_values(payload, s) == bg))
    blank_widths = []
    for side in _SIDES:
        lines = payload if side in ('T', 'B') else payload.T
        if side in ('B', 'R'):
            lines = lines[::-1]
        width = 0
        for line in lines:
            if not np.all(line == bg):
                break
            width += 1
        blank_widths.append(width)
    r, c, h, w = info['bounds']; H, W = pair['input'].shape
    sides = lambda d: tuple(s for s in _SIDES if s in d)
    outer, guides = info['outer_by_side'], info['guide_by_side']
    corner = tuple(int(payload[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    aspect = 'square' if h == w else 'tall' if h > w else 'wide'
    markers = set(outer.values())
    noise = set(np.unique(pair['input'])) - set(np.unique(payload)) - markers - {bg}
    noise_positions = [(int(rr), int(cc)) for color in noise
                       for rr, cc in zip(*np.where(pair['input'] == color))]
    noise_depth = max((min(rr, cc, H - 1 - rr, W - 1 - cc) for rr, cc in noise_positions), default=0)
    border_noise = max((sum(int(v) in noise for v in _side_values(pair['input'], s)[1:-1]) /
                        (len(_side_values(pair['input'], s)) - 2) for s in _SIDES if s not in outer), default=0)
    depths = []
    for side, color in outer.items():
        for component in _color_components(pair['input'], color):
            if not any(rr in (0, H - 1) or cc in (0, W - 1) for rr, cc in component):
                continue
            depths.extend({'T': rr, 'B': H - 1 - rr, 'L': cc, 'R': W - 1 - cc}[side]
                          for rr, cc in component)
    motifs = (payload != fill) & (payload != bg)
    # Actual motif connectivity, independent of which drawing operators ran.
    connectivity = len(_color_components(motifs.astype(np.int8), 1))
    return dict(transform=info['transform'], outer=sides(outer), inner=sides(guides),
                correspondence=tuple(next(s for s in _SIDES if outer.get(s) == guides[g]) for g in sides(guides)),
                input_shape=(H, W), payload_shape=(h, w), aspect=aspect,
                output_shape=tuple(pair['output'].shape), margins=(r, c, H - r - h, W - c - w),
                background=bg, markers=tuple(sorted(markers)), fill=fill,
                motif_colors=tuple(int(x) for x in np.unique(payload) if x not in (bg, fill)),
                noise=tuple(int(x) for x in sorted(noise)), corner=corner,
                fill_corners=sum(x == fill for x in corner), blank_edges=blank,
                blank_widths=tuple(blank_widths),
                background_cells=int(np.count_nonzero(payload == bg)),
                motif_density=float(np.mean(motifs)), motif_components=connectivity,
                noise_depth=noise_depth, border_noise_fraction=border_noise,
                protrusion_depth=max(depths, default=0), asymmetric=_asymmetric(payload))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs, information = set(), set(), {}
    for split in ('train', 'test'):
        information[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            ikey, okey = _grid_key(pair['input']), _grid_key(pair['output'])
            key = (ikey, okey)
            assert ikey != okey and ikey not in seen_inputs and key not in seen_pairs
            assert key not in _SOURCE_KEYS
            seen_inputs.add(ikey); seen_pairs.add(key)
            info = _decode_input(pair['input'])
            expected = _coordinate_prediction(info)
            assert np.array_equal(expected, pair['output']), 'payload cells or orientation changed'
            assert np.array_equal(np.sort(info['payload'].ravel()), np.sort(pair['output'].ravel()))
            information[split].append(info)
    train, test = information['train'], information['test']
    # Training defeats an occupied-side lookup and demonstrates both rotation
    # and reflection, even at the legal minimum of two demonstrations.
    contrasts = [(a, b) for i, a in enumerate(train) for b in train[i + 1:]
                 if set(a['outer_by_side']) == set(b['outer_by_side'])
                 and set(a['guide_by_side']) == set(b['guide_by_side'])
                 and a['transform'] != b['transform']
                 and _asymmetric(a['payload']) and _asymmetric(b['payload'])]
    assert contrasts, 'training needs an asymmetric correspondence reversal'
    rotations = {'r90', 'r270'}
    reflections = {'flip_lr', 'flip_ud', 'transpose', 'anti_transpose'}
    assert any(a['transform'] in rotations and _asymmetric(a['payload']) for a in train)
    assert any(a['transform'] in reflections for a in train)
    has_blank = lambda a: any(np.all(_side_values(a['payload'], s) == a['background']) for s in _SIDES)
    assert not any(has_blank(a) for a in test) or any(has_blank(a) for a in train)
    # Literal palette equality is deliberately not a convention. Matching the
    # repeated local identities is the same rule in both named split regimes.
    for pair, info in zip(puzzle['test'], test):
        assert np.array_equal(_coordinate_prediction(info), pair['output'])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    nt, ne = _resolve_counts(int(seed), num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    copied_contrast = parameters['stable']
    regime = parameters['regime']
    held_outer = 0 if regime == 'source_shift' else parameters['held_outer']
    held_inner = parameters['held_inner']
    train_outer = [i for i in range(4) if regime not in ('held_outer', 'source_shift') or i != held_outer]
    arrangements = [(o, i, rev) for o in train_outer for i in range(4) for rev in (False, True)
                    if not (regime == 'held_combination' and (o, i) == (held_outer, held_inner))]
    teaching_rotations = ('r90', 'r270') if nt == 2 else ('r90', 'r180', 'r270')
    first = _pick(rng, [a for a in arrangements if _arrangement(_adjacent(a[0]), _adjacent(a[1]), a[2]) in teaching_rotations])
    specs = [first, (first[0], first[1], not first[2])]
    specs.extend(_pick(rng, arrangements) for _ in range(nt - 2))
    if not any(_arrangement(_adjacent(o), _adjacent(i), rev) in ('r90', 'r270') for o, i, rev in specs):
        specs[2] = _pick(rng, [a for a in arrangements if _arrangement(_adjacent(a[0]), _adjacent(a[1]), a[2]) in ('r90', 'r270')])
    learned = {_arrangement(_adjacent(o), _adjacent(i), rev) for o, i, rev in specs}
    tests = [(o, i, rev) for o in range(4) for i in range(4) for rev in (False, True)
             if (regime not in ('held_outer', 'source_shift') or o == held_outer)
             and (regime != 'held_combination' or (o, i) == (held_outer, held_inner))
             and (regime != 'familiar' or _arrangement(_adjacent(o), _adjacent(i), rev) in learned)]
    # A new side arrangement accompanies the contrast, including 2/1 puzzles.
    novel = [a for a in tests if a[:2] != first[:2]]
    specs.extend([_pick(rng, novel or tests)] + [_pick(rng, tests) for _ in range(ne - 1)])
    blank_train = str(_pick(rng, _SIDES))
    blank_test = str(_pick(rng, [s for s in _SIDES if s != blank_train]))
    puzzle = {'train': [], 'test': []}
    used = set()
    for index, (outer, inner, reverse) in enumerate(specs):
        split = 'source_train' if index < nt else 'target_test'
        blank = ()
        if parameters['blank_edges'] and (index in (0, nt) or rng.random() < .35):
            blank = (blank_train if index < nt else blank_test,)
        palette = parameters['palette'] if parameters['stable'] else _palette(rng)
        # In variable-palette puzzles the two teaching pairs visibly change all
        # color roles, including palette size, without changing the local rule.
        if index == 1 and not parameters['stable']:
            previous = first_palette
            order = [int(c) for c in rng.permutation(10)]
            shift = int(rng.integers(1, 10))
            color_map = {order[j]: order[(j + shift) % 10] for j in range(10)}
            palette = dict(background=color_map[previous['background']],
                           markers=[color_map[c] for c in previous['markers']],
                           fill=color_map[previous['fill']], noise=color_map[previous['noise']])
            free = [c for c in order if c not in (palette['background'], palette['fill'], palette['noise'], *palette['markers'])]
            count = _pick(rng, [n for n in range(1, 6) if n != len(previous['motifs'])])
            palette['motifs'] = free[:count]
        if index == 0:
            first_palette = palette
        witness = index < 2 or rng.random() < .85
        if index < nt and _arrangement(_adjacent(outer), _adjacent(inner), reverse) in ('r90', 'r270'):
            witness = True
        pair = _generate_pair(rng, split, parameters, outer, inner, reverse, blank,
                              witness=witness, palette=palette)
        if index == 1 and copied_contrast:
            pair = {k: v.copy() for k, v in puzzle['train'][0].items()}
            info = _decode_input(pair['input'])
            colors = list(info['outer_by_side'].values())
            replacements = []
            H, W = pair['input'].shape
            for color, replacement in zip(colors, colors[::-1]):
                for component in _color_components(pair['input'], color):
                    if any(r in (0, H - 1) or c in (0, W - 1) for r, c in component):
                        replacements.extend((pos, replacement) for pos in component)
            for pos, color in replacements:
                pair['input'][pos] = color
            pair['output'] = _apply_transform(info['payload'], _arrangement(_adjacent(outer), _adjacent(inner), reverse))
        # Bounded duplicate fallback: identical inputs imply identical local
        # decoding, so a translated legal color permutation creates a new pair.
        for shift in range(10):
            candidate = {k: ((v.astype(int) + shift) % 10).astype(np.int8) for k, v in pair.items()}
            key = (_grid_key(candidate['input']), _grid_key(candidate['output']))
            if key[0] not in used and key not in _SOURCE_KEYS:
                pair = candidate
                break
        else:
            # More than ten pairs is possible. A half-turn of the whole input
            # and output preserves matching, and provides ten more candidates.
            for shift in range(10):
                candidate = {k: ((np.rot90(v, 2).astype(int) + shift) % 10).astype(np.int8) for k, v in pair.items()}
                key = (_grid_key(candidate['input']), _grid_key(candidate['output']))
                if key[0] not in used and key not in _SOURCE_KEYS:
                    pair = candidate
                    break
            else:
                raise AssertionError('duplicate fallback exhausted')
        used.add(_grid_key(pair['input']))
        puzzle['train' if index < nt else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    measurements, puzzles = [], []
    counts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        counts.add((len(puzzle['train']), len(puzzle['test'])))
        per_puzzle = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _pair_measurements(pair)
                m['split'] = split
                measurements.append(m); per_puzzle.append(m)
        # Contrastive learning, local palettes, and changing sizes are checked
        # on decoded grids, never on sampled regime or operator metadata.
        assert len({m['transform'] for m in per_puzzle}) >= 2
        assert len({m['input_shape'] for m in per_puzzle}) >= 2
        assert len({m['payload_shape'] for m in per_puzzle}) >= 2
        assert len({m['outer'] for m in per_puzzle}) >= 2 or len({m['inner'] for m in per_puzzle}) >= 2
        train_only = [{k: v.copy() for k, v in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            prediction = solve(train_only, pair['input'].copy())
            assert np.array_equal(prediction, pair['output'])
        if seed < 3:
            assert _puzzles_equal(puzzle, generate_puzzle(seed))
    assert len({a for a, _ in counts}) >= 4 and len({b for _, b in counts}) >= 3
    assert len({tuple(_grid_key(p['input']) for p in puzzle['train']) for puzzle in puzzles}) == 42
    distinct = lambda key: {m[key] for m in measurements}
    assert distinct('transform') == set(_SIDE_MAPS)
    for split in ('train', 'test'):
        assert {m['transform'] for m in measurements if m['split'] == split} == set(_SIDE_MAPS)
        assert {m['aspect'] for m in measurements if m['split'] == split} == {'square', 'tall', 'wide'}
    assert len(distinct('input_shape')) >= 100 and len(distinct('payload_shape')) >= 80
    assert (30, 30) in distinct('input_shape')
    assert {(5, 5), (10, 10), (11, 15), (20, 12)} <= distinct('output_shape')
    assert len(distinct('margins')) >= 150 and len(distinct('corner')) >= 80
    assert distinct('fill_corners') == set(range(5))
    assert any(m['background_cells'] == 0 for m in measurements)
    assert any(not m['asymmetric'] for m in measurements)
    assert any(len(m['blank_edges']) >= 2 for m in measurements)
    assert any(2 in m['blank_widths'] for m in measurements)
    assert {s for m in measurements for s in m['blank_edges']} == set(_SIDES)
    assert {len(m['motif_colors']) for m in measurements} >= {1, 2, 3, 4, 5}
    assert len(distinct('motif_components')) >= 15
    assert min(m['motif_density'] for m in measurements) < .1
    assert max(m['motif_density'] for m in measurements) > .3
    assert distinct('noise_depth') >= {0, 1, 2, 3}
    assert distinct('protrusion_depth') >= {0, 1, 2, 3}
    assert any(m['border_noise_fraction'] >= .5 for m in measurements)
    assert any(not m['noise'] and not m['protrusion_depth'] for m in measurements)
    combinations = {(m['transform'], m['aspect'], bool(m['blank_edges']),
                     m['input_shape'][0] == m['input_shape'][1]) for m in measurements}
    assert len(combinations) >= 55
    variable, stable, larger, not_larger, familiar, held, copied = 0, 0, 0, 0, 0, 0, 0
    for puzzle in puzzles:
        train = [_pair_measurements(p) for p in puzzle['train']]
        test = [_pair_measurements(p) for p in puzzle['test']]
        roles = {(m['background'], m['markers'], m['fill']) for m in train + test}
        variable += len(roles) > 1; stable += len(roles) == 1
        if len(roles) > 1:
            assert train[0]['background'] != train[1]['background']
            assert train[0]['fill'] != train[1]['fill']
        area = lambda m: m['payload_shape'][0] * m['payload_shape'][1]
        larger += all(area(t) > max(area(a) for a in train) for t in test)
        not_larger += any(area(t) <= max(area(a) for a in train) for t in test)
        familiar += {t['transform'] for t in test} <= {t['transform'] for t in train}
        held += not ({t['outer'] for t in test} & {t['outer'] for t in train})
        a, b = (_decode_input(p['input']) for p in puzzle['train'][:2])
        copied += (a['bounds'] == b['bounds'] and np.array_equal(a['payload'], b['payload'])
                   and a['guide_by_side'] == b['guide_by_side'])
    assert variable >= 25 and stable >= 1
    assert larger >= 1 and not_larger >= 20 and familiar >= 1 and held >= 1
    assert copied >= 1
    # Source regression tests reject the former false semantic restrictions:
    # two motif colors / no background, 50%-occupied dotted side, inward speck,
    # and depth-two protrusions all remain decodable.
    for source in _SOURCE_CONTENTS:
        inp, out = [np.array([int(c) for c in cells], dtype=np.int8).reshape(shape)
                    for shape, cells in source]
        assert np.array_equal(_coordinate_prediction(_decode_input(inp)), out)
    for requested in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(197, *requested)
        validate_generated_puzzle(p)
        assert _puzzles_equal(p, generate_puzzle(197, *requested))
        assert requested[0] is None or len(p['train']) == requested[0]
        assert requested[1] is None or len(p['test']) == requested[1]
    for a in range(2, 11):
        for b in range(1, min(a, 5)):
            validate_generated_puzzle(generate_puzzle(611 + 13 * a + b, a, b))
    for requested in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4),
                      (True, 1), (3.5, 1), (3, False)):
        try:
            generate_puzzle(1, *requested)
        except ValueError:
            continue
        raise AssertionError('illegal counts were accepted')
    # Mutations must be rejected: wrong output cells, a truncated answer, a
    # broken guide, and an undemonstrated blank-edge extension.
    original = puzzles[0]
    mutations = []
    for kind in ('cell', 'crop', 'guide'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = bad['test'][0]
        if kind == 'cell':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        elif kind == 'crop':
            pair['output'] = pair['output'][1:, :].copy()
        else:
            info = _decode_input(pair['input'])
            r, c, h, w = info['bounds']
            side = next(iter(info['guide_by_side']))
            line = _adjacent_payload_segment(pair['input'], side, r, c, h, w)
            line[len(line) // 2] = info['background']
        mutations.append(bad)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    for pair in bad['train']:
        info = _decode_input(pair['input'])
        payload = info['payload']
        vals, freq = np.unique(payload[payload != info['background']], return_counts=True)
        payload[payload == info['background']] = int(vals[np.argmax(freq)])
        r, c, h, w = info['bounds']
        pair['input'][r:r + h, c:c + w] = payload
        pair['output'] = _coordinate_prediction(info)
    assert any(_pair_measurements(p)['blank_edges'] for p in bad['test'])
    mutations.append(bad)
    for bad in mutations:
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('invalid task was accepted')
