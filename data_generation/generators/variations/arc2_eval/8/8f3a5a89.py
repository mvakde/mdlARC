import numpy as np

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


_SOURCE_FIXTURES = (('8888881888118888\n8888881888118888\n8888881888118888\n8888881888888888\n8888881888888888\n8888881888888888\n8888881881188888\n8888881881188888\n8888881881188888\n8888881888888888\n8888881888888888\n8888881888888888\n8888881888881111\n8888881888881111\n8888881888881111\n6888881888881111', '7777771888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n7888871888888888\n6777771888888888'), ('8888118888\n8888118888\n8888118881\n8888888881\n1888888881\n8888888888\n8888888888\n8888888888\n8888881888\n6888811888', '7777117777\n7887117877\n7887117871\n7787777871\n1788888871\n7788888877\n7888888887\n7888877787\n7888771787\n6777711777'), ('88888888888818888888\n88888888888818888888\n88888888888818888888\n18888888888818888888\n11888888888818888888\n11188888888888888888\n11118888888888888888\n11111888888888888888\n11188888888888888888\n88888888881888888888\n88888888811888888888\n88888888811888888888\n88888888888888888888\n88888888888888888888\n88888888888888111111\n88888888888888111111\n88888888888888111111\n88888888888888111111\n88888888888888111111\n68888888888888111111', '77777777777717777777\n78888888888717888887\n77888888888717888887\n17788888888717888887\n11778888888717888887\n11177888888777888887\n11117788888888888887\n11111788888888888887\n11177788888888888887\n77778888881888888887\n78888888811888888887\n78888888811888888887\n78888888888888888887\n78888888888887777777\n78888888888887111111\n78888888888887111111\n78888888888887111111\n78888888888887111111\n78888888888887111111\n67777777777777111111'), ('888888888818\n888888888818\n888888888818\n888888888818\n888888811118\n888888811118\n888888881118\n888888888818\n888888888818\n888888888111\n888888888111\n681111888111', '777777777718\n788888888718\n788888888718\n788888777718\n788888711118\n788888711118\n788888771118\n788888877718\n788888887718\n788888887111\n777777787111\n671111777111'))

_SOURCE_INPUT_SIGNATURES = {tuple(inp.splitlines()) for inp, _ in _SOURCE_FIXTURES}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            stack = [(r, c)]
            component = []

            while stack:
                x, y = stack.pop()
                component.append((x, y))

                for dr, dc in _N4:
                    nx, ny = x + dr, y + dc
                    if (
                        0 <= nx < h
                        and 0 <= ny < w
                        and mask[nx, ny]
                        and not seen[nx, ny]
                    ):
                        seen[nx, ny] = True
                        stack.append((nx, ny))

            result.append(component)

    return result


def _flood(
    passable: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray:
    h, w = passable.shape
    reached = np.zeros((h, w), dtype=bool)

    if not passable[start]:
        return reached

    reached[start] = True
    stack = [start]

    while stack:
        r, c = stack.pop()

        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and passable[nr, nc]
                and not reached[nr, nc]
            ):
                reached[nr, nc] = True
                stack.append((nr, nc))

    return reached


def _outside_complement(target: np.ndarray) -> np.ndarray:
    h, w = target.shape
    allowed = np.ones((h + 2, w + 2), dtype=bool)
    allowed[1:h + 1, 1:w + 1] = ~target
    return _flood(allowed, (0, 0))


def _analyze_input(
    grid: np.ndarray,
    background: int,
    structure: int,
    marker: int,
    contour: int,
) -> dict:
    h, w = grid.shape
    marker_positions = np.argwhere(grid == marker)
    assert len(marker_positions) == 1

    sr, sc = (int(x) for x in marker_positions[0])
    passable = (grid == background) | (grid == marker)
    target = _flood(passable, (sr, sc))
    outside_padded = _outside_complement(target)

    contour_mask = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(target & (grid == background)):
        r, c = int(r), int(c)
        if outside_padded[r:r + 3, c:c + 3].any():
            contour_mask[r, c] = True

    struct_components = _components(grid == structure)
    preserved = []

    for component in struct_components:
        keep = False

        for r, c in component:
            for dr, dc in _N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and target[nr, nc]:
                    keep = True
                    break
            if keep:
                break

        preserved.append(keep)

    expected = grid.copy()

    for component, keep in zip(struct_components, preserved):
        if not keep:
            for r, c in component:
                expected[r, c] = background

    expected[contour_mask] = contour
    expected[sr, sc] = marker

    return {
        "target": target,
        "outside_padded": outside_padded,
        "contour_mask": contour_mask,
        "struct_components": struct_components,
        "preserved": preserved,
        "expected": expected,
        "seed": (sr, sc),
    }


def _materialize(
    role_grid: np.ndarray,
    parameters: tuple[int, int, int, int],
) -> np.ndarray:
    background, structure, marker, _ = parameters
    grid = np.full(role_grid.shape, background, dtype=np.int8)
    grid[role_grid == 1] = structure
    grid[role_grid == 2] = marker
    return grid


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _input_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.dtype.str, grid.tobytes()


def _pair_key(pair: dict) -> tuple:
    return (
        _input_key(pair["input"]),
        _input_key(pair["output"]),
    )


def _validate_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)

    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )

    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train = _validate_count(
        "num_train",
        num_train,
        2,
        10,
    )
    test = _validate_count(
        "num_test",
        num_test,
        1,
        4,
    )

    if train is None and test is None:
        test = 1 + seed % 4
        minimum_train = max(2, test + 1)
        train = (
            minimum_train
            + (seed // 4) % (11 - minimum_train)
        )
    elif train is None:
        minimum_train = max(2, test + 1)
        train = int(
            rng.integers(minimum_train, 11)
        )
    elif test is None:
        maximum_test = min(4, train - 1)

        if maximum_test < 1:
            raise ValueError(
                "no legal num_test for the requested num_train"
            )

        test = int(
            rng.integers(1, maximum_test + 1)
        )

    if train <= test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train, test


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if not (
                np.array_equal(
                    pair_a["input"],
                    pair_b["input"],
                )
                and np.array_equal(
                    pair_a["output"],
                    pair_b["output"],
                )
            ):
                return False

    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            _input_key(pair["input"]),
            _input_key(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def _roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    background = int(values[np.argmax(counts)])
    singleton = values[counts == 1]
    assert len(singleton) == 1
    marker = int(singleton[0])
    structure = int(next(v for v in values if v not in (background, marker)))
    assert np.count_nonzero(grid == background) > np.count_nonzero(grid == structure) > 1
    return background, structure, marker


def _dilate(mask, diagonal=False):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    result = mask.copy()
    offsets = ((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)) if diagonal else _N4
    for dr, dc in offsets:
        result |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _spread(allowed, reached):
    """Synchronous dilation, independent of the stack-based label constructor."""
    reached = reached & allowed
    while True:
        expanded = _dilate(reached) & allowed
        if np.array_equal(expanded, reached):
            return reached
        reached = expanded


def _predict(train_pairs, test_input):
    """Learn only the new color from demonstrations; inspect all input roles."""
    introduced = set()
    for pair in train_pairs:
        new = set(np.unique(pair['output'])) - set(np.unique(pair['input']))
        assert len(new) == 1
        introduced.update(new)
    assert len(introduced) == 1
    contour = int(next(iter(introduced)))
    grid = test_input.copy()
    bg, st, marker = _roles(grid)
    assert contour not in (bg, st, marker)
    selected = _spread(grid != st, grid == marker)
    complement = np.pad(~selected, 1, constant_values=True)
    border = np.zeros_like(complement)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    exterior = _spread(complement, border)
    rim = _dilate(exterior, True)[1:-1, 1:-1] & selected & (grid == bg)
    kept_structure = _spread(grid == st, _dilate(selected) & (grid == st))
    result = grid.copy()
    result[(grid == st) & ~kept_structure] = bg
    result[rim] = contour
    return result


def _checkerboard(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return bool(np.any((a == d) & (b == c) & (a != b)))


def _regions(role):
    result = []
    for cells in _components(role == 0):
        mask = np.zeros(role.shape, dtype=bool)
        rr, cc = zip(*cells)
        mask[rr, cc] = True
        result.append(mask)
    return result


def _geometry_ok(role):
    if _checkerboard(role == 1) or np.count_nonzero(role == 1) * 2 >= role.size:
        return False
    regions = _regions(role)
    return bool(regions) and all(
        int(m.sum()) >= 8 and (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        for m in regions
    )


def _shape(rng, max_h, max_w):
    """Solid, connected motifs; irregular shapes have a shared filled spine."""
    h = int(rng.integers(1, max_h + 1))
    w = int(rng.integers(1, max_w + 1))
    kind = int(rng.integers(0, 5))
    shape = np.ones((h, w), dtype=bool)
    if kind == 1:
        shape = np.ones((1, w), dtype=bool)
    elif kind == 2:
        shape = np.ones((h, 1), dtype=bool)
    elif kind == 3:
        shape[1:, 1:] = False
    elif kind == 4:
        for r in range(h):
            shape[r, int(rng.integers(1, w + 1)):] = False
    return np.rot90(shape, int(rng.integers(4))).copy()


def _place_object(rng, role, region, edge, roomy=False):
    h, w = role.shape
    max_h = max(1, min(8, h // (3 if roomy else 4)))
    max_w = max(1, min(8, w // (3 if roomy else 4)))
    # Try progressively smaller motifs, ending with an exhaustive singleton pass.
    for attempt in range(8):
        shape = _shape(rng, max_h, max_w) if attempt < 7 else np.ones((1, 1), bool)
        sh, sw = shape.shape
        locations = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)
                     if (r == 0 or c == 0 or r + sh == h or c + sw == w) == edge]
        rng.shuffle(locations)
        for r, c in locations:
            if not np.all(region[r:r + sh, c:c + sw][shape]):
                continue
            cells = np.zeros(role.shape, bool)
            cells[r:r + sh, c:c + sw] = shape
            halo = _dilate(cells, True)
            if not edge:
                halo = _dilate(halo, True)
                # Leave two cells to the exterior rim and to other structures.
                if r < 2 or c < 2 or r + sh > h - 2 or c + sw > w - 2:
                    continue
            if not np.all(region[halo]):
                continue
            role[cells] = 1
            region[cells] = False
            return True
    return False


def _dimensions(rng, modest, witness):
    if witness:
        return int(rng.integers(16, 23 if modest else 31)), int(rng.integers(19, 24 if modest else 31))
    if modest:
        return int(rng.integers(8, 20)), int(rng.integers(8, 20))
    draw = float(rng.random())
    if draw < .10:
        return 30, 30
    if draw < .30:
        return int(rng.integers(7, 13)), int(rng.integers(24, 31))
    return int(rng.integers(7, 31)), int(rng.integers(7, 31))


def _walls(rng, h, w, kind, count, witness):
    role = np.zeros((h, w), np.int8)
    if count == 1:
        return role
    margin = 6 if witness else 3
    center = int(rng.integers(margin, w - margin))
    thick = int(rng.integers(1, 3)) if w - center > margin else 1
    if kind == 'stair' and count == 2:
        low, high = max(3, center - 3), min(w - 4, center + 3)
        cuts = sorted(set([0, h] + [int(x) for x in rng.integers(2, h - 1, size=int(rng.integers(1, 7)))]))
        previous = center
        for start, stop in zip(cuts, cuts[1:]):
            col = int(rng.integers(low, high + 1))
            role[start:stop, col:col + thick] = 1
            if start:
                role[start, min(col, previous):max(col, previous) + thick] = 1
            previous = col
    else:
        role[:, center:center + thick] = 1
        # Guillotine splits create parallel dividers or branches, never enclosed rooms.
        rectangles = [(0, h, 0, center), (0, h, center + thick, w)]
        while len(rectangles) < count:
            candidates = [(i, axis) for i, (r0, r1, c0, c1) in enumerate(rectangles)
                          for axis, extent in ((0, r1-r0), (1, c1-c0)) if extent >= 9]
            if not candidates:
                break
            i, axis = candidates[int(rng.integers(len(candidates)))]
            r0, r1, c0, c1 = rectangles.pop(i)
            if axis == 0:
                cut = int(rng.integers(r0 + 3, r1 - 3))
                role[cut, c0:c1] = 1
                rectangles.extend([(r0, cut, c0, c1), (cut + 1, r1, c0, c1)])
            else:
                cut = int(rng.integers(c0 + 3, c1 - 3))
                role[r0:r1, cut] = 1
                rectangles.extend([(r0, r1, c0, cut), (r0, r1, cut + 1, c1)])
    if kind in ('lobed', 'stair'):
        for _ in range(int(rng.integers(1, 7))):
            cells = np.argwhere(role == 1)
            r, c = cells[int(rng.integers(len(cells)))]
            rh, cw = int(rng.integers(1, max(2, h//4))), int(rng.integers(1, max(2, w//6)))
            r0, r1 = max(0, int(r) - rh//2), min(h, int(r) + rh)
            c0, c1 = max(0, int(c) - cw), min(w, int(c) + cw + 1)
            trial = role.copy()
            trial[r0:r1, c0:c1] = 1
            if _geometry_ok(trial) and len(_regions(trial)) == len(_regions(role)):
                role = trial
    return role


def _marker_candidates(region, role, mode):
    h, w = role.shape
    exterior = _outside_complement(region)
    rim = _dilate(exterior, True)[1:-1, 1:-1] & region
    edge = np.zeros_like(region)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    corners = np.zeros_like(region)
    corners[0, 0] = corners[0, -1] = corners[-1, 0] = corners[-1, -1] = True
    masks = {'corner': region & corners, 'edge': region & edge & ~corners,
             'wall': rim & ~edge, 'interior': region & ~rim}
    return np.argwhere(masks[mode])


def _scene(rng, modest, kind, selection, marker_mode, witness, rotation, reflect):
    h, w = _dimensions(rng, modest, witness)
    max_regions = 3 if modest else 5
    count = int(rng.integers(2 if witness else 1, max_regions + 1))
    if w < 9:
        count = 1
    role = _walls(rng, h, w, kind, count, witness)
    if not _geometry_ok(role):
        return None
    regions = _regions(role)
    if witness and len(regions) < 2:
        return None
    areas = [int(m.sum()) for m in regions]
    index = int(np.argmin(areas)) if selection == 'small' else int(np.argmax(areas)) if selection == 'large' else int(rng.integers(len(regions)))
    target = regions[index]
    islands = int(rng.integers(1 if witness == 1 else 0, 3 if modest else 5))
    erased = int(rng.integers(1 if witness == 1 else 0, 4 if modest else 6))
    intrusions = int(rng.integers(1 if witness == 1 or count == 1 else 0, 4 if modest else 6))
    # Reserve hole evidence before consuming edge space.
    for _ in range(islands):
        _place_object(rng, role, target, False, not modest)
    others = [m for i, m in enumerate(regions) if i != index]
    for _ in range(erased):
        if others:
            _place_object(rng, role, others[int(rng.integers(len(others)))], bool(rng.integers(2)), not modest)
    for _ in range(intrusions):
        _place_object(rng, role, target, True, not modest)
    if np.count_nonzero(role) < 2 or not _geometry_ok(role):
        return None
    candidates = _marker_candidates(target, role, marker_mode)
    if not len(candidates):
        if witness:
            return None
        candidates = np.argwhere(target)
    r, c = candidates[int(rng.integers(len(candidates)))]
    role[r, c] = 2
    role = np.rot90(role, rotation)
    if reflect:
        role = np.fliplr(role)
    return role.copy()


def _features(pair):
    grid, output = pair['input'], pair['output']
    bg, st, marker = _roles(grid)
    introduced = set(np.unique(output)) - set(np.unique(grid))
    assert len(introduced) == 1
    contour = int(next(iter(introduced)))
    info = _analyze_input(grid, bg, st, marker, contour)
    target, outside = info['target'], info['outside_padded'][1:-1, 1:-1]
    regions = _components(grid != st)
    sizes = [len(c) for c in regions]
    area = int(target.sum())
    h, w = grid.shape
    r, c = info['seed']
    center = np.argwhere(target).mean(axis=0)
    displacement = ((center[0]+.5)/h-.5, (center[1]+.5)/w-.5)
    side = ('top' if displacement[0] < 0 else 'bottom') if abs(displacement[0]) > abs(displacement[1]) else ('left' if displacement[1] < 0 else 'right')
    padded = info['outside_padded']
    orth = np.zeros_like(target)
    for dr, dc in _N4:
        orth |= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
    rim = _dilate(padded, True)[1:-1, 1:-1] & target
    category = 'corner' if r in (0,h-1) and c in (0,w-1) else 'edge' if r in (0,h-1) or c in (0,w-1) else 'wall' if rim[r,c] else 'interior'
    islands, erased, intrusions, island_areas, object_shapes, bends, orientations = 0, 0, 0, [], [], 0, set()
    for cells, keep in zip(info['struct_components'], info['preserved']):
        rr, cc = zip(*cells)
        island = not any(outside[x,y] for x,y in cells)
        if island:
            islands += 1
            island_areas.append(len(cells))
        elif not keep:
            erased += 1
        spans_h = min(rr) == 0 and max(rr) == h-1
        spans_w = min(cc) == 0 and max(cc) == w-1
        if spans_h:
            orientations.add('vertical')
        if spans_w:
            orientations.add('horizontal')
        if keep and not island and not spans_h and not spans_w:
            intrusions += 1
        local = np.zeros((max(rr)-min(rr)+1, max(cc)-min(cc)+1), np.int8)
        for x,y in cells:
            local[x-min(rr),y-min(cc)] = 1
        object_shapes.append((local.shape, local.tobytes()))
        # Number of changes along either exposed side of each spanning wall.
        if spans_h:
            spans = [(min(y for x,y in cells if x==row), max(y for x,y in cells if x==row)) for row in range(h)]
            bends += sum(a != b for a,b in zip(spans,spans[1:]))
        elif spans_w:
            spans = [(min(x for x,y in cells if y==col), max(x for x,y in cells if y==col)) for col in range(w)]
            bends += sum(a != b for a,b in zip(spans,spans[1:]))
    return {'shape':grid.shape, 'roles':(bg,st,marker,contour), 'compartments':len(regions),
            'selected': 'only' if len(sizes)==1 else 'small' if area < max(sizes) else 'equal' if sizes.count(area)>1 else 'large',
            'balance':area/sum(sizes), 'near_equal':len(sizes)>1 and min(sizes)/max(sizes) >= .8,
            'corners':tuple(i for i,v in enumerate((target[0,0],target[0,-1],target[-1,0],target[-1,-1])) if v),
            'side':side, 'marker':category, 'marker_on_rim':bool(rim[r,c]),
            'islands':islands, 'erased':erased, 'intrusions':intrusions,
            'island_areas':island_areas, 'object_shapes':object_shapes,
            'orientation':tuple(sorted(orientations)) or ('open',), 'bends':bends,
            'full_divider':bool(np.any(np.all(grid==st,axis=0)) or np.any(np.all(grid==st,axis=1))),
            'diagonal':int(np.count_nonzero(rim & ~orth & (grid==bg))),
            'hole_evidence':bool(islands and np.any(_dilate((grid==st)&~outside,True)&(grid==bg)&(output==bg)))}


def _fallback(index, witness, rotation, reflect):
    """Constructive termination path, including every required teaching witness."""
    h, w = 20 + index % 5, 25 + index // 5
    role = np.zeros((h,w), np.int8)
    role[:,9] = 1
    # Both sides contain a spaced island and a singleton edge intrusion.
    role[7:9,3:5] = role[10:13,16:19] = 1
    role[0,4] = role[-1,19] = 1
    role[3:5,23:25] = role[-4:-2,0:2] = 1
    role[h//2,4] = 2 if witness == 1 else 0
    if witness != 1:
        role[0,w-1] = 2
    role = np.rot90(role,rotation)
    return np.fliplr(role).copy() if reflect else role.copy()


def _fallback_pair(index, witness, parameters, seen, first_side, first_corners=()):
    # Each fallback region offers more marker positions than the maximum
    # thirteen previous inputs. Enumerating them guarantees a fresh input.
    for rotation in range(4):
        role = _fallback(index, witness, rotation, False)
        start = tuple(int(v) for v in np.argwhere(role == 2)[0])
        region = _flood(role != 1, start)
        exterior = _outside_complement(region)
        rim = _dilate(exterior, True)[1:-1, 1:-1] & region
        positions = [start] + [tuple(int(v) for v in cell) for cell in
                              np.argwhere(region & (~rim if witness == 1 else rim))]
        for position in positions:
            trial = role.copy()
            trial[start], trial[position] = 0, 2
            inp = _materialize(trial, parameters)
            if _input_key(inp) in seen:
                continue
            pair = {'input': inp, 'output': _analyze_input(inp, *parameters)['expected']}
            f = _features(pair)
            if witness == 2 and (f['side'] == first_side or set(f['corners']) & set(first_corners)):
                break
            return pair, f
    raise AssertionError('fallback marker positions unexpectedly exhausted')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed,rng,num_train,num_test)
    contour = int(rng.integers(10))
    pool = [c for c in range(10) if c != contour]
    palette = tuple(int(x) for x in rng.choice(pool,3,replace=False))
    # These are sampling preferences, never hidden transformation conventions.
    color_mode = int(rng.integers(3))
    regime = int(rng.integers(5))
    modest = regime == 0
    base_rotation = int(rng.integers(4))
    puzzle = {'train':[], 'test':[]}
    seen, first_side, first_corners = set(), None, ()
    for split, count in (('train',ntrain),('test',ntest)):
        for index in range(count):
            witness = index+1 if split=='train' and index<2 else 0
            if color_mode == 0:
                roles = palette
            elif color_mode == 1:
                choices = [v for v in pool if v != palette[2]]
                roles = tuple(int(x) for x in rng.choice(choices,2,replace=False)) + (palette[2],)
            else:
                roles = tuple(int(x) for x in rng.choice(pool,3,replace=False))
            if witness == 2 and color_mode and roles == _roles(puzzle['train'][0]['input']):
                roles = (roles[1],roles[0],roles[2])
            parameters = roles + (contour,)
            for attempt in range(25):
                rotation = base_rotation if regime==2 else ((base_rotation+(split=='test'))%4 if regime==3 else int(rng.integers(4)))
                reflect = bool(rng.integers(2))
                if regime==1:
                    kind = 'straight' if split=='train' else str(rng.choice(['lobed','stair']))
                elif modest:
                    kind = str(rng.choice(['straight','straight','lobed']))
                else:
                    kind = str(rng.choice(['straight','lobed','stair']))
                selection = 'small' if witness==1 else 'large' if witness==2 else str(rng.choice(['small','large','any']))
                mode = 'interior' if witness==1 else str(rng.choice(['corner','edge','wall'])) if witness==2 else str(rng.choice(['corner','edge','wall','interior']))
                role = _scene(rng,modest,kind,selection,mode,witness,rotation,reflect)
                if role is None:
                    continue
                inp = _materialize(role,parameters)
                pair = {'input':inp,'output':_analyze_input(inp,*parameters)['expected']}
                f = _features(pair)
                if witness==1 and not (f['selected']=='small' and f['erased'] and f['hole_evidence'] and f['diagonal']):
                    continue
                if witness==2 and not (f['selected']=='large' and f['side']!=first_side and f['marker_on_rim']):
                    continue
                if witness == 2 and set(f['corners']) & set(first_corners):
                    continue
                if _input_key(inp) not in seen and _rows_signature(inp) not in _SOURCE_INPUT_SIGNATURES:
                    break
            else:
                pair, f = _fallback_pair(index, witness, parameters, seen, first_side, first_corners)
                inp = pair['input']
            if witness == 1:
                first_side = f['side']
                first_corners = f['corners']
            seen.add(_input_key(inp))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features, colors = set(), [], set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            for g in pair.values():
                assert isinstance(g,np.ndarray) and g.ndim==2 and np.issubdtype(g.dtype,np.integer)
                assert all(1<=d<=30 for d in g.shape) and np.all((g>=0)&(g<=9))
            inp,out = pair['input'],pair['output']
            assert inp.shape==out.shape and not np.array_equal(inp,out)
            key = (inp.shape,tuple(int(v) for v in inp.flat))
            assert key not in seen
            seen.add(key)
            assert _rows_signature(inp) not in _SOURCE_INPUT_SIGNATURES
            bg,st,marker = _roles(inp)
            assert not _checkerboard(inp==st)
            # Avoid nested regions; every free compartment meets the canvas.
            assert _geometry_ok(np.where(inp==st,1,0))
            assert np.array_equal(out, _predict(puzzle['train'],inp.copy()))
            f = _features(pair)
            colors.add(f['roles'][3])
            if split=='train':
                features.append(f)
    assert len(colors)==1
    assert any(f['erased'] for f in features)
    assert any(f['hole_evidence'] for f in features)
    assert any(f['diagonal'] for f in features)
    assert any(f['selected']=='small' for f in features)
    assert any(f['selected']=='large' for f in features)
    assert len({f['side'] for f in features if f['compartments']>1}) >= 2
    assert not set.intersection(*(set(f['corners']) for f in features))
    assert any(f['marker']=='interior' for f in features)
    assert any(f['marker_on_rim'] for f in features)



def _source_regressions():
    examples = [{'input':np.array([[int(c) for c in row] for row in inp.splitlines()], np.int8),
                 'output':np.array([[int(c) for c in row] for row in out.splitlines()], np.int8)}
                for inp,out in _SOURCE_FIXTURES]
    for pair in examples:
        assert np.array_equal(_analyze_input(pair['input'],8,1,6,7)['expected'],pair['output'])
        assert np.array_equal(_predict(examples[:3],pair['input'].copy()),pair['output'])
    # The source's open/no-island scene and full-column test are explicitly legal.
    assert _features(examples[1])['islands'] == 0
    assert _features(examples[1])['compartments'] == 1
    assert _features(examples[3])['full_divider']
    return examples


def _copied_grid_regressions(examples):
    for pair in examples:
        for rotation in range(4):
            for reflect in (False,True):
                inp,out = np.rot90(pair['input'],rotation).copy(),np.rot90(pair['output'],rotation).copy()
                if reflect:
                    inp,out = np.fliplr(inp).copy(),np.fliplr(out).copy()
                assert np.array_equal(_predict(examples[:3],inp),out)
                assert np.array_equal(_analyze_input(inp,8,1,6,7)['expected'],out)
    grid = np.full((18,24),8,np.int8)
    grid[:,10] = 1
    grid[17,0] = 6
    base = _predict(examples[:3],grid.copy())
    moved = grid.copy()
    moved[17,0],moved[7,4] = 8,6
    predicted = _predict(examples[:3],moved.copy())
    expected = base.copy()
    expected[17,0],expected[7,4] = 7,6
    assert np.array_equal(predicted,expected)
    # A roomy internal island is preserved, with exactly the same exterior rim.
    island = moved.copy()
    island[7:10,4:7] = 1
    island[7,4],island[4,4] = 1,6
    expected = _predict(examples[:3],moved.copy())
    expected[7,4] = 8
    expected[7:10,4:7] = 1
    expected[4,4] = 6
    assert np.array_equal(_predict(examples[:3],island.copy()),expected)
    assert not np.any(_predict(examples[:3],island)[6:11,3:8] == 7)
    # Attaching the island to the outside changes it into a contoured intrusion.
    attached = island.copy()
    attached[:8,6] = 1
    attached_out = _predict(examples[:3],attached)
    assert attached_out[10,4] == 7 and attached_out[10,6] == 7
    # Three independent far-side components include an edge block and an L.
    distractors = moved.copy()
    distractors[1:3,14:17] = 1
    distractors[7:10,18] = 1
    distractors[9,18:21] = 1
    distractors[15:18,22:24] = 1
    assert np.array_equal(_predict(examples[:3],distractors.copy()),predicted)
    across = distractors.copy()
    across[7,4],across[7,14] = 8,6
    across[10:12,3:5] = 1
    across_out = _predict(examples[:3],across.copy())
    assert np.all(across_out[10:12,3:5] == 8)
    assert np.all(across_out[1:3,14:17] == 1)
    assert across_out[0,23] == 7 and across_out[0,0] == 8
    assert across_out[7,14] == 6
    # A further divider makes three compartments without a new transformation.
    multi = distractors.copy()
    multi[:,21] = 1
    multi[5:7,6:8] = 1
    multi[11:13,3:5] = 1
    for inp in (moved,island,attached,distractors,across,multi):
        before = inp.copy()
        out = _predict(examples[:3],inp)
        assert np.array_equal(inp,before)
        assert np.array_equal(out,_analyze_input(inp,8,1,6,7)['expected'])


def self_test() -> None:
    examples = _source_regressions()
    _copied_grid_regressions(examples)
    fallback, seen = {'train': [], 'test': []}, set()
    side = None
    for index, witness in enumerate((1, 2, 0)):
        pair, f = _fallback_pair(index, witness, (8, 1, 6, 7), seen, side)
        fallback['train' if witness else 'test'].append(pair)
        seen.add(_input_key(pair['input']))
        if witness == 1:
            side = f['side']
    validate_generated_puzzle(fallback)
    # Exercise collision handling even if normal sampling never needs fallback.
    for _ in range(14):
        pair, _ = _fallback_pair(0, 1, (8, 1, 6, 7), seen, None)
        assert _input_key(pair['input']) not in seen
        seen.add(_input_key(pair['input']))
    records, puzzles, signatures = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        fs = [_features(pair) for s in ('train','test') for pair in puzzle[s]]
        records.extend(fs)
        # Measure selection, placement, and shape diversity within each puzzle.
        assert len({f['selected'] for f in fs}) >= 2
        assert len({f['marker'] for f in fs}) >= 2
        assert len({f['side'] for f in fs}) >= 2
        assert len({f['shape'] for f in fs}) >= 2
        for pair in puzzle['test']:
            training = [{k:v.copy() for k,v in p.items()} for p in puzzle['train']]
            assert np.array_equal(_predict(training,pair['input'].copy()),pair['output'])
    assert len(signatures)==42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({f['shape'] for f in records}) >= 80
    assert min(min(f['shape']) for f in records) <= 8
    assert (30,30) in {f['shape'] for f in records}
    assert any(max(f['shape'])/min(f['shape']) >= 3 for f in records)
    assert {1,2,3,4}.issubset({f['compartments'] for f in records})
    assert {0,1,2,3}.issubset({f['islands'] for f in records})
    assert {0,1,2,3}.issubset({f['erased'] for f in records})
    assert max(f['intrusions'] for f in records) >= 3
    assert {'corner','edge','wall','interior'} == {f['marker'] for f in records}
    assert any(f['near_equal'] for f in records)
    assert len({f['bends'] for f in records}) >= 8
    assert len({f['orientation'] for f in records}) >= 4
    assert any(f['full_divider'] and f['bends'] for f in records)
    assert any(f['compartments']==1 and not f['islands'] and f['intrusions'] for f in records)
    assert any(f['compartments']>=3 and f['islands']>=2 and f['erased']>=2 for f in records)
    assert len({shape for f in records for shape in f['object_shapes']}) >= 150
    assert max(a for f in records for a in f['island_areas']) >= 12
    palette_counts = [len({_roles(pair['input']) for s in p for pair in p[s]}) for p in puzzles]
    assert min(palette_counts)==1 and max(palette_counts)>=4
    assert all(len({f['roles'][i] for f in records})==10 for i in range(4))
    fixed = generate_puzzle(123456,7,3)
    assert _puzzles_equal(fixed,generate_puzzle(123456,7,3))
    assert not _puzzles_equal(fixed,generate_puzzle(123457,7,3))
    for index,(nt,nq) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        puzzle = generate_puzzle(1000+index,nt,nq)
        assert nt is None or len(puzzle['train'])==nt
        assert nq is None or len(puzzle['test'])==nq
        validate_generated_puzzle(puzzle)
    for nt,nq in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1),(2.5,1)):
        try:
            generate_puzzle(999,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal pair counts accepted')
    corrupted = {s:[{k:v.copy() for k,v in pair.items()} for pair in fixed[s]] for s in fixed}
    g = corrupted['test'][0]['output']
    g[0,0] = (int(g[0,0])+1)%10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
