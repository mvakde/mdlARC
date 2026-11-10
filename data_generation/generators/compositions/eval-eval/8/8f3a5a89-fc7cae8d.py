"""Compass-woven rims: topology chooses the compass; a swatch weaves the rim.

Only NumPy and the standard library are used. No files, examples, or generation
state are read by solve(). Diagnostics in self_test() are streamed and not saved.
"""

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
N4 = tuple(NORMALS.values())
BACKGROUND, WALL, SEED = 0, 5, 9
_ROT = np.array([[0, -1], [1, 0]], dtype=int)
_FLIP = np.array([[1, 0], [0, -1]], dtype=int)
MATRICES = tuple(np.linalg.matrix_power(_ROT, k) for k in range(4)) + tuple(
    _FLIP @ np.linalg.matrix_power(_ROT, k) for k in range(4)
)


def _apply(array, operation):
    result = np.rot90(array, operation % 4)
    return (np.fliplr(result) if operation >= 4 else result).copy()


def _adjacent(corner):
    return (SIDES[corner], SIDES[(corner + 1) % 4])


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "invalid color"
    return a


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells, stack = [], [(r, c)]
        seen[r, c] = True
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for dr, dc in N4:
                nr, nc = x + dr, y + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _flood(allowed, start):
    h, w = allowed.shape
    reached = np.zeros(allowed.shape, bool)
    assert allowed[start]
    reached[start] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                stack.append((nr, nc))
    return reached


def _dilate(mask, diagonal=False):
    h, w = mask.shape
    pad = np.pad(mask, 1)
    result = np.zeros_like(mask)
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)] if diagonal else N4
    for dr, dc in offsets:
        result |= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _exterior(target):
    return _flood(np.pad(~target, 1, constant_values=True), (0, 0))


def _rims(target):
    exterior = _exterior(target)
    rim = _dilate(exterior, True)[1:-1, 1:-1] & target
    orthogonal = _dilate(exterior)[1:-1, 1:-1] & target
    return exterior[1:-1, 1:-1], rim, orthogonal


def _segment(grid, bounds, side):
    r, c, h, w = bounds
    if side == "T":
        return grid[r - 1, c:c + w]
    if side == "R":
        return grid[r:r + h, c + w]
    if side == "B":
        return grid[r + h, c:c + w]
    return grid[r:r + h, c - 1]


def _decode_swatch(grid):
    """The isolated hollow wall frame fixes extents, including repeated edges."""
    candidates = []
    for cells in _components(grid == WALL):
        rr, cc = zip(*cells)
        r, c, h, w = min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1
        if not (6 <= h <= 9 and 6 <= w <= 9 and r > 0 and c > 0
                and r + h < grid.shape[0] and c + w < grid.shape[1]):
            continue
        if len(cells) != 2 * (h + w) - 4:
            continue
        frame = grid[r:r + h, c:c + w]
        if not (np.all(frame[[0, -1], :] == WALL) and np.all(frame[:, [0, -1]] == WALL)
                and not np.any(frame[1:-1, 1:-1] == WALL)):
            continue
        bounds = (r + 2, c + 2, h - 4, w - 4)
        guides, valid = {}, True
        for side in SIDES:
            line = _segment(grid, bounds, side)
            if len(np.unique(line)) != 1:
                valid = False
                break
            color = int(line[0])
            if color != BACKGROUND:
                guides[side] = color
        corners = frame[np.ix_([1, h - 2], [1, w - 2])]
        if not (valid and np.all(corners == BACKGROUND) and len(guides) == 2
                and len(set(guides.values())) == 2
                and any(set(guides) == set(_adjacent(k)) for k in range(4))
                and not set(guides.values()) & {BACKGROUND, WALL, SEED}):
            continue
        tile = frame[2:-2, 2:-2].copy()
        if set(np.unique(tile)) & ({BACKGROUND, WALL, SEED} | set(guides.values())):
            continue
        key_mask = np.zeros(grid.shape, bool)
        key_mask[r:r + h, c:c + w] = True
        candidates.append(dict(tile=tile, guides=guides, key_mask=key_mask, frame=(r, c, h, w)))
    assert len(candidates) == 1, "need one isolated, guide-bearing swatch frame"
    return candidates[0]


def _toward_target(target, side):
    """For each wall position, is its inward neighbor in the selected region?"""
    dr, dc = NORMALS[side]
    h, w = target.shape
    pad = np.pad(target, 1)
    return pad[1 - dr:1 - dr + h, 1 - dc:1 - dc + w]


def _operation(guides, controls):
    matches = []
    for i, matrix in enumerate(MATRICES):
        if all(any(color == other and tuple(matrix @ NORMALS[side]) == NORMALS[target]
                   for target, other in controls.items()) for side, color in guides.items()):
            matches.append(i)
    assert len(matches) == 1, "two identities must determine one symmetry"
    return matches[0]


def _analyze(input_grid):
    grid = _grid(input_grid)
    info = _decode_swatch(grid)
    markers = np.argwhere(grid == SEED)
    assert len(markers) == 1, "need one seed"
    origin = tuple(int(v) for v in markers[0])
    target = _flood((grid == BACKGROUND) | (grid == SEED), origin)
    external, rim, orthogonal = _rims(target)
    neighbors = {s: _toward_target(target, s) for s in SIDES}
    degree = sum(m.astype(int) for m in neighbors.values())
    controls, gate_masks = {}, {}
    for color in info["guides"].values():
        mask = (grid == color) & external & (degree > 0)
        cells = np.argwhere(mask)
        assert len(cells) >= 2, "missing external boundary tab"
        faces = [s for s in SIDES if np.all(neighbors[s][mask]) and np.all(degree[mask] == 1)]
        assert len(faces) == 1, "tab must have one unambiguous inward face"
        side = faces[0]
        assert side not in controls
        axis = 1 if side in ("T", "B") else 0
        assert len(set(cells[:, 1 - axis])) == 1
        assert sorted(cells[:, axis]) == list(range(int(cells[:, axis].min()), int(cells[:, axis].max()) + 1))
        controls[side], gate_masks[color] = int(color), mask
    assert any(set(controls) == set(_adjacent(k)) for k in range(4)), "tabs must face adjacent directions"
    operation = _operation(info["guides"], controls)
    info.update(grid=grid, origin=origin, target=target, external=external,
                rim=rim, orthogonal=orthogonal, controls=controls,
                gate_masks=gate_masks, operation=operation)
    return info


def _weave(info, tile=None, mask=None, origin=None):
    tile = _apply(info["tile"], info["operation"]) if tile is None else tile
    mask = info["rim"] if mask is None else mask
    origin = info["origin"] if origin is None else origin
    output = info["grid"].copy()
    rr, cc = np.where(mask & (output == BACKGROUND))
    output[rr, cc] = tile[(rr - origin[0]) % tile.shape[0], (cc - origin[1]) % tile.shape[1]]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive selection, compass, swatch, phase, and rim only from this input."""
    return _weave(_analyze(input_grid))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _asymmetric(tile):
    return len({_signature(_apply(tile, i)) for i in range(8)}) == 8


def _primitive(tile):
    return all(not np.array_equal(tile, np.roll(tile, shift, axis=axis))
               for axis in (0, 1) for shift in range(1, tile.shape[axis]))


def _make_tile(rng, colors):
    for _ in range(64):
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        count = int(rng.integers(3, min(5, h * w) + 1))
        tile = rng.choice(colors[:count], size=(h, w)).astype(np.int8)
        # Sometimes introduce repeated stripes, a solid patch, or a bent stroke.
        style = int(rng.integers(4))
        if style == 1:
            tile[:, int(rng.integers(w))] = colors[0]
        elif style == 2:
            tile[int(rng.integers(h)), :] = colors[1]
        elif style == 3:
            tile[:max(1, h // 2), :max(1, w // 2)] = colors[0]
        if len(np.unique(tile)) >= 3 and _asymmetric(tile) and _primitive(tile):
            return tile
    # Guaranteed asymmetric primitive pattern; termination is independent of luck.
    return np.array([[colors[0], colors[0], colors[1]],
                     [colors[2], colors[3], colors[1]]], dtype=np.int8)


def _rectangle_locations(available, h, w, clearance=1):
    # At most 900 candidates and one integral image; no retry payloads retained.
    pad = clearance
    blocked = np.pad((~available).astype(int), pad, constant_values=1)
    integral = np.pad(blocked, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    H, W = available.shape
    result = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0, c0, r1, c1 = r, c, r + h + 2 * pad, c + w + 2 * pad
            if integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0] == 0:
                result.append((r, c))
    return result


def _base_scene(rng, selection, side, rotation, open_scene):
    h = int(rng.integers(20 if selection else 18, 31))
    w = int(rng.integers(27 if selection else (18 if open_scene else 25), 31))
    grid = np.zeros((h, w), np.int8)
    grid[[0, -1], :] = WALL
    grid[:, [0, -1]] = WALL
    if not open_scene:
        thickness = int(rng.integers(1, 3))
        total = w - 2 - thickness
        if selection == "small":
            width = int(rng.integers(11, (total - 1) // 2 + 1))
        elif selection == "large":
            width = total - int(rng.integers(5, 8))
        else:
            width = int(rng.integers(11, total - 4))
        divider = 1 + width if side == 0 else w - 1 - width - thickness
        grid[:, divider:divider + thickness] = WALL
        for _ in range(int(rng.integers(1, 5))):
            r = int(rng.integers(2, h - 3))
            length, depth = int(rng.integers(1, 5)), int(rng.integers(1, 4))
            c0, c1 = (divider - depth, divider) if side == 0 else (divider + thickness, divider + thickness + depth)
            grid[r:min(h - 1, r + length), c0:c1] = WALL
        # Branches change the number and shapes of excluded rooms independently.
        branch_rows = rng.permutation(np.arange(4, h - 4))[::5][:int(rng.integers(3))]
        for r in branch_rows:
            if side == 0:
                grid[r, divider + thickness:w - 1] = WALL
            else:
                grid[r, 1:divider] = WALL
        selected_half = np.indices(grid.shape)[1] < divider if side == 0 else np.indices(grid.shape)[1] >= divider + thickness
    else:
        selected_half = np.ones(grid.shape, bool)
    # Attached masses create concavities and diagonal-only rim positions.
    for _ in range(int(rng.integers(1, 5))):
        edge = int(rng.integers(4))
        depth, length = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        if edge in (0, 2):
            c = int(rng.integers(2, w - length - 1))
            rr = slice(0, depth + 1) if edge == 0 else slice(h - depth - 1, h)
            grid[rr, c:c + length] = WALL
        else:
            r = int(rng.integers(2, h - length - 1))
            cc = slice(0, depth + 1) if edge == 3 else slice(w - depth - 1, w)
            grid[r:r + length, cc] = WALL
    regions = _components((grid == BACKGROUND) & selected_half)
    if not regions:
        return None
    cells = max(regions, key=len)
    target = np.zeros(grid.shape, bool)
    rr, cc = zip(*cells)
    target[rr, cc] = True
    return _apply(grid, rotation), _apply(target, rotation)


def _paint_swatch(grid, position, tile, guides):
    r, c = position
    h, w = tile.shape
    frame = np.zeros((h + 4, w + 4), dtype=np.int8)
    frame[[0, -1], :] = WALL
    frame[:, [0, -1]] = WALL
    frame[2:-2, 2:-2] = tile
    for side, color in guides.items():
        _segment(frame, (2, 2, h, w), side)[:] = color
    grid[r:r + h + 4, c:c + w + 4] = frame


def _paint_tab(grid, target, external, side, color, rng):
    neighbors = {s: _toward_target(target, s) for s in SIDES}
    degree = sum(m.astype(int) for m in neighbors.values())
    allowed = (grid == WALL) & external & neighbors[side] & (degree == 1)
    axis = 1 if side in ("T", "B") else 0
    choices = []
    for length in range(2, 5):
        for r, c in np.argwhere(allowed):
            r, c = int(r), int(c)
            cells = [(r + (axis == 0) * k, c + (axis == 1) * k) for k in range(length)]
            if all(x < grid.shape[0] and y < grid.shape[1] and allowed[x, y] for x, y in cells):
                choices.append(cells)
    if not choices:
        return False
    for pos in _pick(rng, choices):
        grid[pos] = color
    return True


def _render_scene(rng, tile, ids, source_corner, target_corner, reverse,
                  selection, side, rotation, marker_mode, open_scene=False):
    base = _base_scene(rng, selection, side, rotation, open_scene)
    if base is None:
        return None
    grid, region = base
    guides = dict(zip(_adjacent(source_corner), ids))
    positions = _rectangle_locations(region, tile.shape[0] + 4, tile.shape[1] + 4)
    if not positions:
        return None
    key_position = _pick(rng, positions)
    _paint_swatch(grid, key_position, tile, guides)
    available = region & (grid == BACKGROUND)
    # A second, solid island carries a convincing but topologically invalid key.
    ih, iw = int(rng.integers(4, 6)), int(rng.integers(4, 6))
    positions = _rectangle_locations(available, ih, iw)
    if not positions:
        return None
    ir, ic = _pick(rng, positions)
    grid[ir:ir + ih, ic:ic + iw] = WALL
    for s, color in zip(_adjacent(int(rng.integers(4))), ids):
        if s == "T":
            grid[ir, ic + 1:ic + iw - 1] = color
        elif s == "B":
            grid[ir + ih - 1, ic + 1:ic + iw - 1] = color
        elif s == "L":
            grid[ir + 1:ir + ih - 1, ic] = color
        else:
            grid[ir + 1:ir + ih - 1, ic + iw - 1] = color
    for _ in range(int(rng.integers(4))):
        ih, iw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        places = _rectangle_locations(region & (grid == BACKGROUND), ih, iw)
        if places:
            r, c = _pick(rng, places)
            shape = np.ones((ih, iw), bool)
            if ih > 1 and iw > 1 and rng.random() < .5:
                shape[1:, 1:] = False
            patch = grid[r:r + ih, c:c + iw]
            patch[shape] = WALL
    # Remote matching marks are preserved, but cannot orient the selected room.
    far = ~region & (grid == BACKGROUND)
    for _ in range(int(rng.integers(3))):
        places = _rectangle_locations(far & (grid == BACKGROUND), 2, 3)
        if places:
            r, c = _pick(rng, places)
            grid[r:r + 2, c:c + 3] = WALL
            grid[r, c:c + 2] = _pick(rng, ids)
    start = tuple(int(v) for v in np.argwhere(region & (grid == BACKGROUND))[0])
    target = _flood(grid == BACKGROUND, start)
    external, rim, _ = _rims(target)
    controls = dict(zip(_adjacent(target_corner), ids[::-1] if reverse else ids))
    for normal, color in controls.items():
        if not _paint_tab(grid, target, external, normal, color, rng):
            return None
    candidates = np.argwhere(target & (~rim if marker_mode == "interior" else rim if marker_mode == "rim" else target))
    if not len(candidates):
        return None
    operation = _operation(guides, controls)
    oriented = _apply(tile, operation)
    # Teach marker-relative phase, avoiding a coincident canvas-origin phase.
    reference_phase = ((key_position[0] + 2) % oriented.shape[0],
                       (key_position[1] + 2) % oriented.shape[1])
    shifted = [tuple(int(v) for v in p) for p in candidates
               if (int(p[0]) % oriented.shape[0], int(p[1]) % oriented.shape[1])
               not in ((0, 0), reference_phase)]
    origin = _pick(rng, shifted or [tuple(int(v) for v in p) for p in candidates])
    grid[origin] = SEED
    info = _analyze(grid)
    if not np.all(~info["external"][info["key_mask"]]):
        return None
    output = _weave(info)
    f = _features(info)
    if selection and f["selection"] != selection:
        return None
    if not (f["hole_gap"] and f["diagonal"] and f["orientation_matters"]
            and f["phase_matters"] and f["reference_phase_matters"]):
        return None
    return {"input": grid, "output": output}


def _features(info):
    grid, target, rim = info["grid"], info["target"], info["rim"]
    free = ((grid == BACKGROUND) | (grid == SEED)) & ~info["key_mask"]
    areas = [len(c) for c in _components(free)]
    area = int(target.sum())
    selection = "only" if len(areas) == 1 else "small" if area < max(areas) else "large" if area > min(areas) else "equal"
    all_boundary = _dilate(np.pad(~target, 1, constant_values=True), True)[1:-1, 1:-1] & target
    output = _weave(info)
    swapped = {s: next(c for c in info["controls"].values() if c != color)
               for s, color in info["controls"].items()}
    wrong_op = _operation(info["guides"], swapped)
    tab_colors = set(info["guides"].values())
    decoys = sum(int(np.count_nonzero((grid == c) & ~info["external"] & ~info["key_mask"])) for c in tab_colors)
    return dict(shape=grid.shape, tile_shape=info["tile"].shape, operation=info["operation"],
                source=tuple(s for s in SIDES if s in info["guides"]),
                target_normals=tuple(s for s in SIDES if s in info["controls"]),
                correspondence=tuple((s, next(t for t, c in info["controls"].items() if c == color))
                                     for s, color in sorted(info["guides"].items())),
                compartments=len(areas), selection=selection, area=area,
                marker_on_rim=bool(rim[info["origin"]]),
                diagonal=int(np.count_nonzero(rim & ~info["orthogonal"] & (grid == BACKGROUND))),
                hole_gap=int(np.count_nonzero(all_boundary & ~rim & (grid == BACKGROUND))),
                island_tabs=decoys, rim_cells=int(np.count_nonzero(rim & (grid == BACKGROUND))),
                orientation_matters=not np.array_equal(output, _weave(info, tile=_apply(info["tile"], wrong_op))),
                identity_differs=not np.array_equal(output, _weave(info, tile=info["tile"])),
                phase_matters=not np.array_equal(output, _weave(info, origin=(0, 0))),
                reference_phase_matters=not np.array_equal(output, _weave(info, origin=(info["frame"][0] + 2, info["frame"][1] + 2))),
                islands=len(_components(~target & ~info["external"] & ~info["key_mask"])))


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    pool = [int(c) for c in rng.permutation([1, 2, 3, 4, 6, 7, 8])]
    ids, inks = pool[:2], pool[2:]
    first_tile = _make_tile(rng, inks)
    held = int(rng.integers(4))
    train_corners = [k for k in range(4) if k != held]
    arrangements = [(s, t, rev) for s in train_corners for t in range(4) for rev in (False, True)
                    if _operation(dict(zip(_adjacent(s), ids)), dict(zip(_adjacent(t), ids[::-1] if rev else ids))) in (1, 3)]
    first = _pick(rng, arrangements)
    first_side, first_rotation = int(rng.integers(2)), int(rng.integers(8))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train" and i < 2:
                source, target_corner, reverse = first[0], first[1], first[2] ^ bool(i)
                tile = first_tile
                selection, side = ("small", first_side) if i == 0 else ("large", 1 - first_side)
                rotation, mode, open_scene = first_rotation, "interior" if i == 0 else "rim", False
            else:
                source = held if split == "test" else int(_pick(rng, train_corners))
                target_corner, reverse = int(rng.integers(4)), bool(rng.integers(2))
                tile = _make_tile(rng, inks)
                selection, side = None, int(rng.integers(2))
                rotation, mode = int(rng.integers(8)), str(_pick(rng, ("rim", "interior", "any")))
                open_scene = bool(rng.random() < .22)
            for _ in range(64):
                pair = _render_scene(rng, tile, ids, source, target_corner, reverse,
                                     selection, side, rotation, mode, open_scene)
                if pair is not None and _signature(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantic invariants and teaching witnesses, without generation data."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records, palettes = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _signature(inp) not in seen, "duplicate input"
            seen.add(_signature(inp))
            info = _analyze(inp)
            assert _asymmetric(info["tile"]) and _primitive(info["tile"])
            assert not np.any(info["target"] & info["key_mask"])
            assert not np.any(info["external"] & info["key_mask"])
            assert np.array_equal(out, _weave(info)), "wrong rim, texture, symmetry, or phase"
            changed = inp != out
            assert np.array_equal(changed, info["rim"] & (inp == BACKGROUND))
            assert out[info["origin"]] == SEED
            assert np.array_equal(inp[~changed], out[~changed])
            f = _features(info)
            assert f["hole_gap"] and f["diagonal"] and f["island_tabs"] >= 4
            assert f["orientation_matters"] and f["phase_matters"] and f["reference_phase_matters"]
            palettes.add(tuple(sorted(info["guides"].values())))
            records[split].append((info, f))
    assert len(palettes) == 1, "tab palette is shared within a puzzle"
    train = records["train"]
    assert any(f["selection"] == "small" for _, f in train)
    assert any(f["selection"] == "large" for _, f in train)
    assert any(not f["marker_on_rim"] for _, f in train)
    assert any(f["marker_on_rim"] for _, f in train)
    assert any(f["operation"] in (1, 3) and f["identity_differs"] for _, f in train)
    assert any(f["operation"] >= 4 for _, f in train)
    contrasts = [(a, b) for j, (a, fa) in enumerate(train) for b, fb in train[j + 1:]
                 if a["guides"] == b["guides"] and set(a["controls"]) == set(b["controls"])
                 and a["controls"] != b["controls"] and np.array_equal(a["tile"], b["tile"])]
    assert contrasts, "need a same-swatch correspondence reversal"
    # A fixed corner selects different sides in the two teaching witnesses.
    # Check room centroids rather than the hidden construction's side labels.
    assert any(np.dot(np.argwhere(a["target"]).mean(0) / np.array(a["grid"].shape) - .5,
                         np.argwhere(b["target"]).mean(0) / np.array(b["grid"].shape) - .5) < 0
               for a, b in contrasts)
    train_sources = {f["source"] for _, f in train}
    assert all(f["source"] not in train_sources for _, f in records["test"]), "missing guide-arrangement shift"
    return True


def self_test():
    """42 seeds, count boundaries, cell-level invariants and counterfactuals.

    Keeps only bounded aggregate sets and the current puzzle. No files or traces
    are written. The count sweep discards each result before constructing another.
    """
    shapes, tiles, operations, structures = set(), set(), {s: set() for s in ("train", "test")}, set()
    compartments, island_counts, marker_modes, identities = set(), set(), set(), set()
    puzzle_hashes = set()
    import hashlib
    # Check the named NumPy operations against signed coordinate movement.
    probe = np.array([[1, 2, 3], [4, 6, 7]], dtype=np.int8)
    for operation, matrix in enumerate(MATRICES):
        positions = matrix @ np.indices(probe.shape).reshape(2, -1)
        positions -= positions.min(axis=1, keepdims=True)
        moved = np.empty(tuple(positions.max(axis=1) + 1), np.int8)
        moved[positions[0], positions[1]] = probe.ravel()
        assert np.array_equal(moved, _apply(probe, operation))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        local_shapes, local_structure = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, before), "solve mutated input"
                info = _analyze(inp)
                f = _features(info)
                shapes.add(f["shape"]); tiles.add(f["tile_shape"])
                operations[split].add(f["operation"])
                compartments.add(f["compartments"]); island_counts.add(f["islands"])
                marker_modes.add(f["marker_on_rim"]); identities.add(tuple(sorted(info["guides"].values())))
                # Role-normalized obstacle shape, not just literal colors.
                role = (inp != BACKGROUND).astype(np.uint8)
                key = (inp.shape, hashlib.sha256(role.tobytes()).digest())
                structures.add(key); local_structure.add(key); local_shapes.add(inp.shape)
                digest.update(bytes(inp.shape)); digest.update(inp.tobytes())
        puzzle_hashes.add(digest.digest())
        assert len(local_shapes) >= 2 and len(local_structure) == 5
        if seed < 3:
            repeated = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], repeated[s]) for k in ("input", "output"))
            del repeated
        # Swapping only true controls changes the pattern but not its support.
        info = _analyze(puzzle["train"][0]["input"])
        swapped = info["grid"].copy()
        colors = list(info["gate_masks"])
        for old, new in zip(colors, colors[::-1]):
            swapped[info["gate_masks"][old]] = new
        alt = _analyze(swapped)
        assert np.array_equal(info["target"], alt["target"])
        assert np.array_equal(info["rim"], alt["rim"])
        assert not np.array_equal(_weave(info)[info["rim"]], _weave(alt)[info["rim"]])
        # Changing enclosed decoy identities leaves the entire answer unchanged
        # except those preserved input cells. Hole topology ignores these tabs.
        decoy = (np.isin(info["grid"], colors) & ~info["external"] & ~info["key_mask"])
        changed = info["grid"].copy()
        for old, new in zip(colors, colors[::-1]):
            changed[decoy & (info["grid"] == old)] = new
        expected = _weave(info)
        expected[decoy] = changed[decoy]
        assert np.array_equal(solve(changed), expected)
        # Eight-neighbor contact and outside flooding each make a difference.
        assert not np.array_equal(_weave(info), _weave(info, mask=info["orthogonal"]))
        all_rim = _dilate(np.pad(~info["target"], 1, constant_values=True), True)[1:-1, 1:-1] & info["target"]
        assert not np.array_equal(_weave(info), _weave(info, mask=all_rim))
    assert len(puzzle_hashes) == 42 and len(structures) >= 200
    assert len(shapes) >= 45 and len(tiles) >= 8 and len(identities) >= 12
    assert operations["train"] == set(range(8)) and operations["test"] == set(range(8))
    assert {1, 2, 3, 4} <= compartments and len(island_counts) >= 3
    assert marker_modes == {False, True}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(719 + 31 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
            del p
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = 1
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "tile_shapes": len(tiles), "obstacle_structures": len(structures),
            "train_symmetries": sorted(operations["train"]), "test_symmetries": sorted(operations["test"]),
            "compartments": sorted(compartments), "island_counts": sorted(island_counts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
