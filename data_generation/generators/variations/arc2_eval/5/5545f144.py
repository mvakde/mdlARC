import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

_SOURCE_SERIALIZED = {
    (
        "11111111311111111311111111/14111141314111111314111111/"
        "11141411311111111311111111/11111141311111111311111111/"
        "11111111311414111311111111/11141141311141141311111111/"
        "11111111311111111311111111/11111111311111111311111414/"
        "14111141314111141314111141/11111111311111111311111111",
        "11111111/14111111/41411111/11111111/11111111/"
        "11111111/11111111/11111111/11111111/11111111",
    ),
    (
        "000000200000020003002000000/030030203003020003302000000/"
        "000000233300020003002000000/000000200000020000002000333/"
        "000030200003020000302000030/030000200000020000002000000/"
        "333030200003020000302000030/000000200000020000002000000",
        "000000/000000/000000/000000/000000/000333/000030/000000",
    ),
    (
        "5555565555554555556555555/5655555555554555555555555/"
        "5566556555654555555555555/5655555555554555555555555/"
        "5555555555554555555655555/5565655655654556566555555/"
        "5555555555554555555655555/5555565555554555556555555/"
        "5555555555554555555555555/5565565555554556556555555/"
        "5555555555554555555555555/5555555555554555555555555",
        "555556555555/555556555555/555565655555/555555555555/"
        "555555555555/555555555555/555555555555/555555555555/"
        "555555555555/555555555555/555555555555/555555555555",
    ),
    (
        "666666666666666/666666686666866/666666666666666/"
        "666866666866866/666666666666666/666666686666666/"
        "666686888686666/666666868666666/686866666666666/"
        "666666666666666/666666666666666/666666666666666/"
        "666666666666666/686666866686666/666666666666666",
        "666666666666666/666666666666666/666666666666666/"
        "666666666666666/666666666666666/666668866666666/"
        "666688666666666/666668866666666/666666666666666/"
        "666666666666666/666666666666666/666666666666666/"
        "666666666666666/666666666666666/666666666666666",
    ),
}


def _rotate_offset(offset, turns):
    row, col = offset
    for _ in range(turns % 4):
        row, col = col, -row
    return row, col


def _arrow(span, extension, solid, tails):
    """Compose a right-facing head, extended tip, and optional hollow rear."""
    cells = {(0, -i) for i in range(extension)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, -extension + 1 - depth) for r in rows)
    for length in range(1, tails + 1):
        cells.update((r, -extension + 1 - span - length) for r in (-span, span))
    return tuple(sorted(cells))


_MASKS = tuple(dict.fromkeys(
    _arrow(span, extension, solid, tails)
    for span in (1, 2) for extension in (1, 2)
    for solid in (False, True) for tails in (0, 1, 2)
))
_SOURCE_MASKS = (_arrow(1, 1, False, 0), _arrow(1, 1, True, 0),
                 _arrow(1, 2, False, 0), _arrow(1, 1, True, 1))


def _cells(mask, tip, direction):
    return {(tip[0] + dr, tip[1] + dc)
            for dr, dc in (_rotate_offset(p, direction - 1) for p in mask)}


def _in_bounds(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _direction_between(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr == 0) != (dc == 0), "references must be distinct and orthogonal"
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _decode_route(tip, waypoints):
    """Require ONE occupied ray, even when one candidate is straight ahead."""
    remaining, points, directions = set(waypoints), [tip], []
    while remaining:
        rays = {}
        for q in remaining:
            if q[0] == tip[0] or q[1] == tip[1]:
                direction = _direction_between(tip, q)
                rays.setdefault(direction, []).append(q)
        assert len(rays) == 1, "route has a junction or a disconnected suffix"
        direction, candidates = next(iter(rays.items()))
        tip = min(candidates, key=lambda q: abs(q[0]-tip[0])+abs(q[1]-tip[1]))
        points.append(tip)
        directions.append(direction)
        remaining.remove(tip)
    return points, directions


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        component = {remaining.pop()}
        stack = list(component)
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _identify_mask(cells):
    """Infer an axis of symmetry and its single projecting tip, not a size ID."""
    matches = []
    for tip in sorted(cells):
        for direction in range(4):
            offsets = {_rotate_offset((r-tip[0], c-tip[1]), 1-direction)
                       for r, c in cells}
            if any(c > 0 for r, c in offsets):
                continue
            if {p for p in offsets if p[1] == 0} != {(0, 0)}:
                continue
            if max(abs(r) for r, c in offsets) == 0:
                continue
            if {( -r, c) for r, c in offsets} != offsets:
                continue
            matches.append((tuple(sorted(offsets)), tip, direction))
    assert len(matches) == 1, "mover needs a unique pointed, symmetric geometry"
    return matches[0]


def _split_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) in (2, 3)
    if len(colors) == 2:
        boards, axis, divider = [grid], "single", None
    else:
        candidates = []
        for color in colors:
            for dim in (0, 1):
                lines = np.flatnonzero(np.all(grid == color, axis=1-dim)).tolist()
                if not 1 <= len(lines) <= 3:
                    continue
                ends = [-1] + lines + [grid.shape[dim]]
                lengths = [b-a-1 for a, b in zip(ends, ends[1:])]
                if min(lengths) <= 0 or len(set(lengths)) != 1:
                    continue
                boards = [grid[a+1:b, :] if dim == 0 else grid[:, a+1:b]
                          for a, b in zip(ends, ends[1:])]
                if all(color not in b for b in boards):
                    candidates.append((boards, "vertical" if dim == 0 else "horizontal", int(color)))
        assert len(candidates) == 1, "dividers must separate equal boards uniquely"
        boards, axis, divider = candidates[0]
    colors, counts = np.unique(boards[0], return_counts=True)
    assert len(colors) == 2 and max(counts) > boards[0].size // 2
    bg = int(colors[int(np.argmax(counts))])
    ink = int(colors[int(np.argmin(counts))])
    assert all(set(np.unique(b)) == {bg, ink} for b in boards)
    return boards, bg, ink, divider, axis


def _render_board(shape, bg, ink, mask, tip, direction, waypoints):
    result = np.full(shape, bg, dtype=np.int8)
    mover = _cells(mask, tip, direction)
    assert _in_bounds(mover, shape)
    # Rendering is an overlay: in the source test one unshown intermediate
    # footprint covers a future waypoint. Its logical route suffix survives.
    for r, c in mover | set(waypoints):
        result[r, c] = ink
    return result


def _solve_input(grid):
    """Read only the input. Later snapshots are matched, never re-segmented."""
    boards, bg, ink, divider, axis = _split_input(grid)
    components = _components(set(zip(*np.where(boards[0] == ink))))
    movers = [c for c in components if len(c) > 1]
    assert len(movers) == 1, "first snapshot must separate mover and singletons"
    mask, tip, initial = _identify_mask(movers[0])
    waypoints = {next(iter(c)) for c in components if len(c) == 1}
    assert len(waypoints) >= 2
    points, directions = _decode_route(tip, waypoints)
    assert directions[0] == initial, "first leg must follow the visible tip"
    states = []
    for i, point in enumerate(points):
        direction = initial if i == 0 else directions[i-1]
        states.append(_render_board(boards[0].shape, bg, ink, mask,
                                    point, direction, points[i+1:]))
    indices = []
    for board in boards:
        matches = [i for i, state in enumerate(states[:-1]) if np.array_equal(board, state)]
        assert len(matches) == 1, "snapshot does not preserve the rigid mover and route suffix"
        assert not indices or matches[0] > indices[-1], "snapshots must advance in time"
        indices.append(matches[0])
    assert indices[0] == 0
    return states[-1], dict(mask=mask, points=points, directions=directions,
                           snapshots=indices, axis=axis, bg=bg, ink=ink,
                           divider=divider, shape=boards[0].shape)


def _extend_options(points, footprints, mask, shape):
    """Adding a future point may not create an alternative ray at ANY past tip."""
    options = {}
    r, c = points[-1]
    for direction, (dr, dc) in enumerate(_DIRS):
        initial_cells = _cells(mask, points[0], direction) if len(points) == 1 else None
        if initial_cells is not None and not _in_bounds(initial_cells, shape):
            continue
        past_cells = footprints if initial_cells is None else [initial_cells]
        for distance in range(2, max(shape)):
            q = (r + dr*distance, c + dc*distance)
            new_cells = _cells(mask, q, direction)
            if not _in_bounds(new_cells, shape):
                continue
            if any(max(abs(q[0]-a), abs(q[1]-b)) <= 1
                   for cells in past_cells for a, b in cells):
                continue
            valid = True
            for p, nxt in zip(points[:-1], points[1:]):
                if p[0] == q[0] or p[1] == q[1]:
                    cross = (q[0]-p[0])*(nxt[1]-p[1]) - (q[1]-p[1])*(nxt[0]-p[0])
                    dot = (q[0]-nxt[0])*(nxt[0]-p[0]) + (q[1]-nxt[1])*(nxt[1]-p[1])
                    if cross or dot <= 0:
                        valid = False
                        break
            if valid:
                options.setdefault(direction, []).append((q, new_cells))
    return options


def _grow_route(rng, shape, mask, target, minimum, need_turn):
    best = None
    straight_weight = float(rng.uniform(0.25, 2.0))
    for _ in range(28):
        points = [(int(rng.integers(shape[0])), int(rng.integers(shape[1])))]
        footprints, directions = [], []
        while len(points) <= target:
            options = _extend_options(points, footprints, mask, shape)
            if not options:
                break
            choices = list(options)
            weights = np.array([straight_weight if directions and d == directions[-1] else 1.0
                                for d in choices])
            d = int(rng.choice(choices, p=weights/weights.sum()))
            q, cells = options[d][int(rng.integers(len(options[d])))]
            if not footprints:
                footprints.append(_cells(mask, points[0], d))
            footprints.append(cells)
            points.append(q)
            directions.append(d)
        if len(directions) >= minimum and (not need_turn or directions[0] != directions[-1]):
            if best is None or len(points) > len(best):
                best = points
            if len(directions) == target:
                break
    return best


def _combine(boards, axis, divider):
    if axis == "single":
        return boards[0].copy()
    dim = 0 if axis == "vertical" else 1
    bar_shape = (1, boards[0].shape[1]) if dim == 0 else (boards[0].shape[0], 1)
    bar = np.full(bar_shape, divider, dtype=np.int8)
    pieces = []
    for b in boards:
        if pieces:
            pieces.append(bar)
        pieces.append(b)
    return np.concatenate(pieces, axis=dim)


def _make_pair(rng, palette, axis, mask, target, need_turn, small=False, compact=False,
               large=False, avoid_shape=None):
    count = 1 if axis == "single" else (2 if compact else int(rng.integers(2, 5)))
    depth = -min(c for r, c in mask)
    span = max(r for r, c in mask)
    # Large arrows need room to turn as well as merely fit in a panel.
    if count > 1 and (30-count+1)//count < depth+span+3:
        count = 2
    narrow_max = 30 if count == 1 else (30-count+1)//count
    narrow_min = min(narrow_max, max(6, depth+span+3))
    if large:
        narrow_min = max(narrow_min, 15)
    if small:
        narrow_max = min(narrow_max, 14)
    minimum = max(count, 8 if large else 4 if need_turn else 2)
    target = max(target, minimum)
    route = None
    for attempt in range(3):
        narrow = int(rng.integers(narrow_min, narrow_max+1))
        long_min = max(8, 2*span+3, depth+5)
        if large:
            long_min = max(long_min, 15)
        long_max = 14 if small else 30
        long = int(rng.integers(long_min, long_max+1))
        if not small and rng.random() < 0.18:
            long = 30
        shape = (narrow, long) if axis == "vertical" else (long, narrow)
        if axis == "single" and rng.random() < 0.5:
            shape = shape[::-1]
        if shape == avoid_shape:
            dim = 1 if axis == "vertical" else 0
            shape = tuple((n+1 if n<30 else n-1) if j==dim else n for j,n in enumerate(shape))
        route = _grow_route(rng, shape, mask, target, minimum, need_turn)
        if route is not None:
            break
    if route is None:
        # A bounded constructive fallback: a short bend followed by a straight
        # suffix on a long board. It has one occupied ray at every reference.
        shape = (30, max(narrow_min, depth+span+3))
        route = [(span, depth), (span, depth+2)]
        route += [(span+3+2*i, depth+2) for i in range(minimum-1)]
        if axis == "vertical":
            shape = shape[::-1]
            route = [(c, r) for r, c in route]
        if shape == avoid_shape:
            dim = 0 if axis == "vertical" else 1
            shape = tuple(n+1 if j==dim else n for j,n in enumerate(shape))
    directions = [_direction_between(a, b) for a, b in zip(route, route[1:])]
    # Use full-board translations, including edge placements, without cropping.
    occupied = set(route)
    for i, p in enumerate(route):
        occupied |= _cells(mask, p, directions[max(0, i-1)])
    r0, r1 = min(r for r, c in occupied), max(r for r, c in occupied)
    c0, c1 = min(c for r, c in occupied), max(c for r, c in occupied)
    dr = int(rng.integers(-r0, shape[0]-r1))
    dc = int(rng.integers(-c0, shape[1]-c1))
    route = [(r+dr, c+dc) for r, c in route]
    if count == 1:
        indices = [0]
    elif rng.random() < 0.35:
        indices = list(range(count))
    else:
        indices = [0] + sorted(int(i) for i in rng.choice(np.arange(1, len(route)-1), count-1, replace=False))
    bg, ink, divider = palette
    boards = [_render_board(shape, bg, ink, mask, route[i], directions[max(0, i-1)], route[i+1:])
              for i in indices]
    inp = _combine(boards, axis, divider)
    # Render by moving the actual sampled cells. The independent input decoder
    # below checks this construction without using the route or mask parameters.
    out = _render_board(shape, bg, ink, mask, route[-1], directions[-1], [])
    return {"input": inp, "output": out}


def _validate_optional_count(value, name, low, high):
    if value is not None and (isinstance(value, (bool, np.bool_)) or
                             not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}, or None")


def _resolve_counts(rng, num_train, num_test):
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1) + 1 if num_train is not None else 5))
    if num_train is None:
        num_train = int(rng.integers(num_test+1, 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid_fingerprint(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _serialize_grid(grid):
    return "/".join("".join(str(int(value)) for value in row) for row in grid)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Puzzle-level choices govern the teaching regime, never hidden commands.
    regime = int(rng.choice(3, p=[0.25, 0.60, 0.15]))  # source shift, mixed, single
    shared_palette = bool(rng.random() < 0.20)
    palette = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    primary_axis = "horizontal" if regime == 0 or rng.random() < 0.55 else "vertical"
    train_axes = []
    for i in range(train_count):
        if regime == 2:
            axis = "single"
        elif regime == 0 or i == 0:
            axis = primary_axis
        elif i == 1:
            axis = "single"
        else:
            axis = str(rng.choice(["single", "horizontal", "vertical"]))
        train_axes.append(axis)
    demonstrated_axes = sorted(set(train_axes) - {"single"})
    test_axes = []
    for i in range(test_count):
        if regime != 1:
            axis = "single"
        elif i == 0:
            axis = str(rng.choice(demonstrated_axes))
        elif i == 1:
            axis = "single"
        else:
            axis = str(rng.choice(["single"] + demonstrated_axes))
        test_axes.append(axis)
    puzzle, seen, first_mask = {"train": [], "test": []}, set(), None
    first_info = None
    for split, axes in (("train", train_axes), ("test", test_axes)):
        for i, axis in enumerate(axes):
            pool = _SOURCE_MASKS if (regime == 0 and split == "train") or rng.random() < 0.5 else _MASKS
            if split == "train" and i == 1:
                pool = tuple(m for m in pool if m != first_mask)
            mask = pool[int(rng.integers(len(pool)))]
            if split == "train" and i == 0:
                first_mask = mask
            if not shared_palette:
                choices = [p for p in range(10) if p != (palette[0] if i else -1)]
                bg = int(rng.choice(choices))
                rest = rng.choice([c for c in range(10) if c != bg], 2, replace=False)
                palette = (bg, int(rest[0]), int(rest[1]))
            teach = split == "train" and i == 0
            short = split == "train" and i == 1
            target = int(rng.integers(5, 10) if teach else rng.integers(2, 4) if short else rng.integers(2, 21))
            for attempt in range(12):
                pair = _make_pair(rng, palette, axis, mask, target, teach,
                                  small=(regime == 0 and split == "train"), compact=short,
                                  large=(regime == 0 and split == "test"),
                                  avoid_shape=first_info["shape"] if short else None)
                key = _grid_fingerprint(pair["input"])
                if key in seen or (_serialize_grid(pair["input"]), _serialize_grid(pair["output"])) in _SOURCE_SERIALIZED:
                    continue
                expected, info = _solve_input(pair["input"])
                assert np.array_equal(expected, pair["output"])
                if teach and info["directions"][0] == info["directions"][-1]:
                    continue
                if short and (len(info["directions"]) == len(first_info["directions"]) or info["shape"] == first_info["shape"]):
                    continue
                break
            else:
                # At least 90 role assignments versus at most 13 earlier inputs
                # and four source pairs: this fallback is finite and guaranteed.
                done = False
                for bg in range(10):
                    for ink in range(10):
                        if bg == ink:
                            continue
                        divider = next(c for c in range(10) if c not in (bg, ink))
                        color_map = np.arange(10, dtype=np.int8)
                        for old, new in zip(palette, (bg, ink, divider)):
                            color_map[old] = new
                        recolored = {k: color_map[v] for k,v in pair.items()}
                        source_key = tuple(_serialize_grid(recolored[k]) for k in ("input", "output"))
                        if _grid_fingerprint(recolored["input"]) not in seen and source_key not in _SOURCE_SERIALIZED:
                            pair, done = recolored, True
                            break
                    if done:
                        break
                assert done
                _, info = _solve_input(pair["input"])
            if teach:
                first_info = info
            seen.add(_grid_fingerprint(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_fingerprint(pair["input"])
            assert key not in seen
            seen.add(key)
            assert (_serialize_grid(pair["input"]), _serialize_grid(pair["output"])) not in _SOURCE_SERIALIZED
            expected, info = _solve_input(pair["input"])
            assert np.array_equal(expected, pair["output"]), "incorrect terminal rigid shape"
            assert not np.array_equal(pair["input"], pair["output"])
            # New examples deliberately keep all remaining singletons visibly
            # separated at every waypoint, including unshown intermediate states.
            for i, tip in enumerate(info["points"]):
                cells = _cells(info["mask"], tip, info["directions"][max(0, i-1)])
                assert all(max(abs(q[0]-r), abs(q[1]-c)) > 1
                           for q in info["points"][i+1:] for r,c in cells)
            analyses[split].append(info)
    train, test = analyses["train"], analyses["test"]
    assert len({a["mask"] for a in train}) >= 2, "training must demonstrate shape preservation"
    assert len({len(a["directions"]) for a in train}) >= 2
    assert len({a["shape"] for a in train}) >= 2
    assert any(a["directions"][0] != a["directions"][-1] for a in train), "training must show rotation"
    demonstrated = {a["axis"] for a in train}
    assert all(a["axis"] == "single" or a["axis"] in demonstrated for a in test)
    return analyses


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid puzzle: {exc}") from exc


def _route_signature(points):
    variants = []
    for reflection in (1, -1):
        for turns in range(4):
            offsets = [_rotate_offset((r-points[0][0], reflection*(c-points[0][1])), turns)
                       for r,c in points]
            variants.append(tuple(offsets))
    return min(variants)


def measure_pair(pair):
    """Review measurements reconstructed from grids, not generation metadata."""
    expected, a = _solve_input(pair["input"])
    assert np.array_equal(expected, pair["output"])
    points, ds, mask = a["points"], a["directions"], a["mask"]
    turns = [(b-a)%4 for a,b in zip(ds, ds[1:]) if a != b]
    distances = [abs(b[0]-a[0])+abs(b[1]-a[1]) for a,b in zip(points,points[1:])]
    terminal = _cells(mask, points[-1], ds[-1])
    margin = min(min(r,c,a["shape"][0]-1-r,a["shape"][1]-1-c) for r,c in terminal)
    legs = []
    for d, length in zip(ds, distances):
        if legs and legs[-1][0] == d:
            legs[-1] = (d, legs[-1][1]+length)
        else:
            legs.append((d, length))
    winding = set()
    for i in range(len(legs)-3):
        segment = legs[i:i+4]
        senses = {(y[0]-x[0])%4 for x,y in zip(segment,segment[1:])}
        if len(senses) == 1 and senses <= {1,3}:
            differences = [segment[j+2][1]-segment[j][1] for j in (0,1)]
            winding.add("inward" if max(differences)<0 else "outward" if min(differences)>0 else "winding")
    return dict(board_shape=a["shape"], snapshots=len(a["snapshots"]), axis=a["axis"],
                snapshot_gaps=tuple(b-a for a,b in zip(a["snapshots"],a["snapshots"][1:])),
                mask=mask, area=len(mask), waypoints=len(ds), distances=tuple(distances),
                turns=len(turns), mixed_turns=1 in turns and 3 in turns,
                straight_steps=sum(x==y for x,y in zip(ds,ds[1:])), initial=ds[0], final=ds[-1],
                same_direction=ds[0]==ds[-1], terminal_margin=margin,
                palette=(a["bg"],a["ink"],a["divider"]), route=_route_signature(points),
                endpoint=points[-1], winding=tuple(sorted(winding)))


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_fingerprint(p["input"]),_grid_fingerprint(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    # Exact source regressions include the disconnected chevron and the later
    # diagonally touching singleton. Outputs are used only for comparison.
    for inp, out in _SOURCE_SERIALIZED:
        inp = np.array([[int(c) for c in row] for row in inp.split("/")], dtype=np.int8)
        out = np.array([[int(c) for c in row] for row in out.split("/")], dtype=np.int8)
        expected, _ = _solve_input(inp)
        assert np.array_equal(expected, out)
    metrics = {"train": [], "test": []}
    counts, keys, palette_variety, regimes = [set(),set()], set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(_puzzle_key(puzzle))
        for i,s in enumerate(("train", "test")):
            counts[i].add(len(puzzle[s]))
            metrics[s].extend(measure_pair(p) for p in puzzle[s])
        rows = [measure_pair(p) for p in puzzle["train"]+puzzle["test"]]
        assert len({m["mask"] for m in rows}) >= 2
        assert len({m["waypoints"] for m in rows}) >= 2
        palette_variety.append(len({m["palette"][:2] for m in rows}))
        regimes.add((tuple(sorted({measure_pair(p)["axis"] for p in puzzle["train"]})),
                     tuple(sorted({measure_pair(p)["axis"] for p in puzzle["test"]}))))
    assert len(counts[0]) >= 4 and len(counts[1]) >= 3 and len(keys) >= 40
    assert sum(v>1 for v in palette_variety) >= 25 and sum(v==1 for v in palette_variety) >= 2
    assert len(regimes) >= 5
    for split, rows in metrics.items():
        assert {m["initial"] for m in rows} == set(range(4))
        assert {m["final"] for m in rows} == set(range(4))
        assert len({m["route"] for m in rows}) >= 45
        assert len({m["mask"] for m in rows}) >= 10
        assert len({m["waypoints"] for m in rows}) >= 10
        assert min(m["waypoints"] for m in rows) == 2 and max(m["waypoints"] for m in rows) >= 15
        assert sum(m["mixed_turns"] for m in rows) >= 10
        assert sum(m["same_direction"] for m in rows) >= 10
        assert any(m["turns"] == 0 for m in rows)
        assert sum(len(set(m["distances"])) > 1 for m in rows) >= len(rows)//2
        assert any(m["straight_steps"] >= 2 for m in rows)
        assert any("inward" in m["winding"] for m in rows)
        assert any("outward" in m["winding"] for m in rows)
        assert all({m["palette"][i] for m in rows if m["palette"][i] is not None} == set(range(10))
                   for i in range(3))
        assert {m["axis"] for m in rows} == {"single", "horizontal", "vertical"}
        assert {m["snapshots"] for m in rows} == {1,2,3,4}
        assert any(30 in m["board_shape"] for m in rows)
        assert any(m["terminal_margin"] <= 1 for m in rows) and any(m["terminal_margin"] >= 3 for m in rows)
        assert any(1 in m["snapshot_gaps"] for m in rows)
        assert any(any(g>1 for g in m["snapshot_gaps"]) for m in rows)
        assert all(any(m["area"] == n for m in rows) for n in (3,4,6))
        assert any(len({m["mask"] for m in rows if m["area"] == n}) > 1 for n in range(3,15))
        assert any(len(m["mask"])>6 and m["mixed_turns"] and len(set(m["distances"]))>1
                   and m["board_shape"][0]!=m["board_shape"][1] for m in rows)
    for i, requested in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p = generate_puzzle(101+i, *requested)
        validate_generated_puzzle(p)
        for s,n in zip(("train","test"),requested):
            assert n is None or len(p[s]) == n
    assert _puzzle_key(generate_puzzle(500,7,3)) == _puzzle_key(generate_puzzle(500,7,3))
    assert _puzzle_key(generate_puzzle(500,7,3)) != _puzzle_key(generate_puzzle(501,7,3))
    for a,b in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1),(2,1.5)):
        try:
            generate_puzzle(900,a,b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Negative tests: branching, corrupted labels, altered snapshots, and schema.
    try:
        _decode_route((3,3), {(3,5),(5,3)})
    except AssertionError:
        pass
    else:
        raise AssertionError("junction accepted")
    p = generate_puzzle(123,3,1)
    broken = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0,0] = (int(broken["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    p = generate_puzzle(0,3,1)
    panel, bg, ink, divider, axis = _split_input(p["train"][0]["input"])
    assert axis != "single"
    repeated = _combine([panel[0],panel[0]],axis,divider)
    try:
        _solve_input(repeated)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-advancing snapshots accepted")
