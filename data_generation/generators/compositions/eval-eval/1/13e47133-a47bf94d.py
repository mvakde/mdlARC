"""Wired contour weaving: input-only geometric decoding, no parent imports.

Only NumPy and the standard library are used. Generation retains one puzzle at
a time, writes no artifacts, and makes at most 32 geometry proposals per pair.
"""
from collections import Counter, deque
from itertools import permutations
import hashlib
import json

import numpy as np


_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        cells, queue = [first], deque([first])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _D4:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    queue.append(q)
        part = np.zeros((h, w), dtype=bool)
        rr, cc = zip(*cells)
        part[rr, cc] = True
        result.append(part)
    return result


def _touches_edge(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _depth(mask, diagonal=True):
    """Input solver: BFS inward from all boundary cells, including hole edges."""
    h, w = mask.shape
    dirs = _D8 if diagonal else _D4
    padded = np.pad(mask, 1)
    interior = mask.copy()
    for dr, dc in dirs:
        interior &= padded[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
    depth = np.full(mask.shape, -1, dtype=np.int8)
    boundary = mask & ~interior
    depth[boundary] = 0
    queue = deque(map(tuple, np.argwhere(boundary)))
    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and depth[nr, nc] < 0:
                depth[nr, nc] = depth[r, c] + 1
                queue.append((nr, nc))
    return depth


def _erosion(mask):
    """Construction: repeated Boolean erosion, separate from the BFS solver."""
    h, w = mask.shape
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int8)
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in _D8:
            inner &= padded[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
        result[current & ~inner] = level
        current = inner
        level += 1
    return result


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _corners(mask, depth, length):
    options = []
    for r, c in np.argwhere(mask & (depth == 0)):
        r, c = int(r), int(c)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if _inside(mask, r - dr, c) or _inside(mask, r, c - dc):
                    continue
                if not all(_inside(mask, r + a, c + b) for a, b in ((dr, 0), (0, dc), (dr, dc))):
                    continue
                if all(_inside(mask, r + k * dr, c + k * dc)
                       and depth[r + k * dr, c + k * dc] == k for k in range(length)):
                    options.append(tuple((r + k * dr, c + k * dc) for k in range(length)))
    return options


def _decode_key(grid, mask, depth, background):
    cues = set(map(tuple, np.argwhere(mask & (grid != background))))
    assert len(cues) in (1, 2), "one explicit corner program per chamber"
    candidates = [p for p in _corners(mask, depth, len(cues)) if set(p) == cues]
    assert candidates, "key is not a consecutive inward corner diagonal"
    programs = {tuple(int(grid[p]) for p in path) for path in candidates}
    assert len(programs) == 1
    program = next(iter(programs))
    assert len(set(program)) == len(program), "two-entry programs must be primitive"
    return program


def _trace(grid, background, wall, rooms):
    room_map = np.full(grid.shape, -1, dtype=np.int8)
    for i, room in enumerate(rooms):
        room_map[room] = i
    field = (room_map < 0) & (grid != background) & (grid != wall)
    visible = set(map(tuple, np.argwhere(field)))
    colors = set(map(int, np.unique(grid[field])))
    assert 1 <= len(colors) <= 2
    graph = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _D4
                 if (p[0] + dr, p[1] + dc) in visible] for p in visible}
    ends = sorted(p for p in graph if len(graph[p]) == 1)
    crosses = {p for p in graph if len(graph[p]) == 4}
    assert len(ends) == len(rooms)
    assert all(len(edges) in (1, 2, 4) for edges in graph.values())
    carrier_colors = {int(grid[p]) for p in visible - crosses}
    assert len(carrier_colors) == 1
    carrier = carrier_colors.pop()
    crossing_colors = {int(grid[p]) for p in crosses}
    assert len(crossing_colors) == (1 if crosses else 0)
    cross = next(iter(crossing_colors), None)
    assert cross is None or cross != carrier
    assert colors == ({carrier} if cross is None else {carrier, cross})
    chamber_of, ports, normals = {}, [None] * len(rooms), [None] * len(rooms)
    h, w = grid.shape
    for p in ends:
        neighbor = graph[p][0]
        dr, dc = p[0] - neighbor[0], p[1] - neighbor[1]
        a, b = (p[0] + dr, p[1] + dc), (p[0] + 2 * dr, p[1] + 2 * dc)
        assert 0 <= b[0] < h and 0 <= b[1] < w
        assert grid[a] == wall and room_map[b] >= 0, "endpoint does not enter a chamber"
        room = int(room_map[b])
        assert ports[room] is None, "chamber has multiple ports"
        chamber_of[p], ports[room], normals[room] = room, p, (dr, dc)
    assert all(p is not None for p in ports)
    matching, seen, routes = [-1] * len(rooms), set(), []
    used_ends = set()
    for start in ends:
        if start in used_ends:
            continue
        previous, current = None, start
        route, states = [], set()
        while True:
            state = (previous, current)
            assert state not in states, "closed or self-repeating cable"
            states.add(state)
            route.append(current)
            seen.add(current)
            neighbors = graph[current]
            if previous is not None and len(neighbors) == 1:
                break
            if len(neighbors) == 4:
                following = (2 * current[0] - previous[0], 2 * current[1] - previous[1])
                assert following in neighbors
            else:
                following = next(p for p in neighbors if p != previous)
            previous, current = current, following
        a, b = chamber_of[start], chamber_of[current]
        assert a != b and matching[a] == matching[b] == -1
        matching[a], matching[b] = b, a
        used_ends.update((start, current))
        routes.append(route)
    assert seen == visible, "unattached routing fragment or closed wire"
    labels = {int(c) for room in rooms for c in np.unique(grid[room])} - {background}
    assert labels.isdisjoint({wall, carrier, cross})
    return dict(matching=matching, ports=ports, normals=normals, routes=routes,
                crossings=crosses, carrier=carrier, cross=cross)


def _parse(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    order = np.argsort(-counts, kind="stable")
    background = int(order[0])
    assert counts[background] > counts[order[1]], "background must be locally identifiable"
    interpretations = []
    # Every legal outline has at least 24 cells; sparse key colors cannot be walls.
    for color in map(int, order[1:]):
        if counts[color] < 24:
            continue
        rooms = [m for m in _components(grid != color) if not _touches_edge(m)]
        if not (2 <= len(rooms) <= 6 and len(rooms) % 2 == 0):
            continue
        try:
            depths, keys = [], []
            for room in rooms:
                assert int(room.sum()) >= 20
                depth = _depth(room)
                assert depth.max() >= 1
                key = _decode_key(grid, room, depth, background)
                depths.append(depth)
                keys.append(key)
            info = _trace(grid, background, color, rooms)
            info.update(grid=grid, background=background, wall=color, rooms=rooms,
                        depths=depths, keys=keys)
            interpretations.append(info)
        except (AssertionError, StopIteration):
            continue
    assert len(interpretations) == 1, f"expected one interpretation, found {len(interpretations)}"
    return interpretations[0]


def _paint(info, mode="weave", matching=None):
    result = info["grid"].copy()
    matching = info["matching"] if matching is None else matching
    for i, (room, depth, own) in enumerate(zip(info["rooms"], info["depths"], info["keys"])):
        peer = info["keys"][matching[i]]
        dd = depth[room].astype(int)
        if mode == "box":
            rr, cc = np.nonzero(room)
            dd = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        elif mode == "l1":
            dd = _depth(room, diagonal=False)[room].astype(int)
        if mode == "concat":
            program = np.array(own + peer, dtype=np.int8)
            values = program[dd % len(program)]
        else:
            if mode == "omit_b":
                peer = own
            k = dd if mode == "wrong_index" else dd // 2
            values = np.where(dd % 2 == 0, np.array(own)[k % len(own)], np.array(peer)[k % len(peer)])
        result[room] = values
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, programs, depths and pairings from this input alone."""
    return _paint(_parse(input_grid))


def _positional(info):
    groups = {}
    for i, normal in enumerate(info["normals"]):
        groups.setdefault(normal, []).append(i)
    if len(groups) != 2:
        return None
    a, b = sorted(groups)
    if a != tuple(-x for x in b) or len(groups[a]) != len(groups[b]):
        return None
    axis = 1 if a[0] else 0
    left = sorted(groups[a], key=lambda i: info["ports"][i][axis])
    right = sorted(groups[b], key=lambda i: info["ports"][i][axis])
    result = [-1] * len(info["rooms"])
    for i, j in zip(left, right):
        result[i], result[j] = j, i
    return result


def _witnesses(info, out):
    flags = {name: not np.array_equal(_paint(info, name), out)
             for name in ("omit_b", "box", "l1", "wrong_index", "concat")}
    positional = _positional(info)
    flags["positional"] = positional is not None and not np.array_equal(_paint(info, matching=positional), out)
    return flags


def _has_island(info):
    for room in info["rooms"]:
        rr, cc = np.nonzero(room)
        local = info["grid"][rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        if any(not _touches_edge(m) for m in _components(local == info["wall"])):
            return True
    return False


def _transfers(target, rng):
    """Move permutation cycles through one vacant track; at most n+cycles moves."""
    n = len(target)
    state = list(range(n)) + [None]
    moves = []
    for j in map(int, rng.permutation(n)):
        if state[j] == target[j]:
            continue
        saved = state[j]
        moves.append((j, n))
        state[n], state[j] = state[j], None
        vacant = j
        while target[vacant] != saved:
            old = state.index(target[vacant])
            moves.append((old, vacant))
            state[vacant], state[old] = state[old], None
            vacant = old
        moves.append((n, vacant))
        state[vacant], state[n] = state[n], None
    assert state[:n] == target
    return moves


def _room_shape(rng, height, width, notch, island=False):
    wall = np.zeros((height, width), dtype=bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    if notch:
        # Side notches leave both facing port walls intact.
        work = wall if rng.integers(2) else wall[:, ::-1]
        a = int(rng.integers(2, max(3, height - 5)))
        b = int(rng.integers(a + 2, height - 2))
        shift = int(rng.integers(1, min(3, width - 6)))
        work[a + 1:b, 0] = False
        work[a, :shift + 1] = work[b, :shift + 1] = True
        work[a:b + 1, shift] = True
    if island:
        # A small solid obstacle, away from the boundary and port column.
        r = int(rng.integers(3, height - 3))
        c = int(rng.integers(3, width - 3))
        wall[r, c] = True
        if rng.random() < .4 and c + 1 < width - 2:
            wall[r, c + 1] = True
    regions = [m for m in _components(~wall) if not _touches_edge(m)]
    assert len(regions) == 1
    region = regions[0]
    assert region[1, width // 2] and region[-2, width // 2]
    return wall, region


def _geometry(rng, n, moves, lesson):
    if n == 1:
        slot_widths = [int(rng.integers(19 if lesson == 1 else 13, 28))]
        gap = 0
    elif n == 2:
        slot_widths = [int(rng.integers(11, 14)) for _ in range(2)]
        gap = int(rng.integers(1, min(4, 31 - sum(slot_widths))))
    else:
        slot_widths, gap = [9, 9, 9], 1
    base_width = sum(slot_widths) + gap * (n - 1)
    left = int(rng.integers(min(2, 30 - base_width) + 1))
    right = int(rng.integers(min(2, 30 - base_width - left) + 1))
    width = base_width + left + right
    centers, slots, x = [], [], left
    for sw in slot_widths:
        centers.append(x + sw // 2)
        slots.append((x, sw))
        x += sw + gap
    minimum = 2 * len(moves) + 3 if moves else 5
    base_height = 11 if lesson == 1 else 9
    available = 30 - minimum - 2 * base_height
    assert available >= 0
    top_add = int(rng.integers(min(2, available) + 1)); available -= top_add
    bottom_add = int(rng.integers(min(2, available) + 1)); available -= bottom_add
    top, bottom = base_height + top_add, base_height + bottom_add
    extras = int(rng.integers(available + 1))
    pieces = [2, 2] + [2] * max(0, len(moves) - 1)
    for _ in range(extras):
        pieces[int(rng.integers(len(pieces)))] += 1
    lead, tail, gaps = pieces[0], pieces[1], pieces[2:]
    corridor = lead + tail + sum(gaps) + 1
    spare = 30 - (top + bottom + corridor)
    upper_margin = int(rng.integers(min(2, spare) + 1)); spare -= upper_margin
    lower_margin = int(rng.integers(min(2, spare) + 1))
    shape = (upper_margin + top + corridor + bottom + lower_margin, width)
    return shape, slots, centers, top, bottom, upper_margin, lead, tail, gaps


def _paths(centers, spare, start, end, moves, lead, gaps):
    columns = centers + [spare]
    state = list(range(len(centers))) + [None]
    paths = [[(start, c)] for c in centers]
    previous, row = start, start + lead
    for k, (old, new) in enumerate(moves):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        owner = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[owner].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = row
        if k < len(gaps):
            row += gaps[k]
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    return paths, state[:-1]


def _make_pair(rng, palette, lesson, is_test, orientation, allow_islands=True):
    n = 2 if lesson == 0 else 1 if lesson == 1 else int(rng.integers(2 if is_test else 1, 4))
    choices = list(permutations(range(n)))
    if lesson == 0 or is_test:
        choices = [p for p in choices if p != tuple(range(n))]
    target = list(choices[int(rng.integers(len(choices)))])
    moves = _transfers(target, rng)
    if lesson is None and not is_test and not moves and rng.random() < .4:
        lane = int(rng.integers(n))
        moves = [(lane, n), (n, lane)]
    shape, slots, centers, top, bottom, margin, lead, tail, gaps = _geometry(rng, n, moves, lesson)
    bg, border, carrier, cross = palette[:4]
    colors = palette[4:]
    inp = np.full(shape, bg, dtype=np.int8)
    masks, depths = [], []
    lengths = [int(rng.integers(1, 3)) for _ in range(2 * n)]
    if lesson == 0:
        lengths = [1] * 4
    elif lesson == 1:
        lengths = [2, 1] if rng.integers(2) else [1, 2]
    if is_test:
        lengths[0] = lengths[n + target.index(0)] = 2
    programs = [tuple(map(int, rng.choice(colors, length, replace=False))) for length in lengths]
    if lesson == 0:
        values = list(map(int, rng.choice(colors, 4, replace=False)))
        programs = [(v,) for v in values]
    elif lesson == 1:
        values = list(map(int, rng.choice(colors, 3, replace=False)))
        programs = [tuple(values[:lengths[0]]), tuple(values[lengths[0]:])]
    elif is_test:
        values = list(map(int, rng.choice(colors, 4, replace=False)))
        programs[0], programs[n + target.index(0)] = tuple(values[:2]), tuple(values[2:])
    start = margin + top
    end = start + lead + tail + sum(gaps)
    for bank, max_height in enumerate((top, bottom)):
        for i, ((slot_left, sw), center) in enumerate(zip(slots, centers)):
            index = bank * n + i
            height = max_height if lesson == 1 else int(rng.integers(9, max_height + 1))
            width = int(rng.integers(max(17 if lesson == 1 else 9, sw - 3), sw + 1))
            x0 = center - width // 2
            y0 = start - height if bank == 0 else end + 1
            # Three-lane length-two query objects need an intact 7x7 core.
            protected = is_test and index in (0, n + target.index(0))
            notch = lesson == 1 or (not protected and rng.random() < .65)
            island = lesson is None and allow_islands and not protected and width >= 11 and height >= 10 and rng.random() < .3
            wall, room = _room_shape(rng, height, width, notch, island)
            full = np.zeros(shape, dtype=bool)
            full[y0:y0 + height, x0:x0 + width] = room
            depth = _erosion(full)
            if lesson == 1:
                assert depth.max() >= 4
            if protected:
                assert depth.max() >= 3
            inp[y0:y0 + height, x0:x0 + width][wall] = border
            key_paths = _corners(full, depth, lengths[index])
            assert key_paths
            path = key_paths[int(rng.integers(len(key_paths)))]
            for p, value in zip(path, programs[index]):
                inp[p] = value
            masks.append(full)
            depths.append(depth)
    spare = centers[-1] + int(rng.integers(3, min(6, shape[1] - centers[-1] - 1) + 1))
    paths, final_state = _paths(centers, spare, start, end, moves, lead, gaps)
    assert final_state == target
    visits = Counter(p for path in paths for p in path)
    for p, times in visits.items():
        assert times in (1, 2) and inp[p] == bg
        inp[p] = cross if times == 2 else carrier
    matching = [-1] * (2 * n)
    for j, i in enumerate(final_state):
        matching[i], matching[n + j] = n + j, i
    # Ground truth uses the constructed pairing and erosion, never solve().
    out = inp.copy()
    for i, (mask, depth, own) in enumerate(zip(masks, depths, programs)):
        remote = programs[matching[i]]
        for d in range(int(depth.max()) + 1):
            program = own if d % 2 == 0 else remote
            out[mask & (depth == d)] = program[(d // 2) % len(program)]
    if rng.integers(2):
        inp, out = inp[:, ::-1], out[:, ::-1]
    inp, out = np.rot90(inp, orientation).copy(), np.rot90(out, orientation).copy()
    info = _parse(inp)
    assert np.array_equal(_paint(info), out)
    if lesson == 0:
        assert _witnesses(info, out)["positional"] and info["crossings"]
    elif lesson == 1:
        flags = _witnesses(info, out)
        assert all(flags[name] for name in ("box", "l1", "wrong_index", "concat"))
    return {"input": inp, "output": out}, _has_island(info)


def _check_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count("num_train", num_train, 2, 10)
    nq = _check_count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.permutation(10)))
    offset = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    train_island = False
    for index in range(nt + nq):
        is_test = index >= nt
        lesson = index if index < 2 else None
        background = backgrounds[index % 10]
        palette = [background] + list(map(int, rng.permutation([c for c in range(10) if c != background])))
        orientation = (offset + index) % 4 if lesson is not None else int(rng.integers(4))
        for attempt in range(32):
            try:
                pair, island = _make_pair(rng, palette, lesson, is_test, orientation,
                                          allow_islands=not is_test or train_island)
                key = (pair["input"].shape, pair["input"].tobytes())
                assert key not in seen
                break
            except AssertionError:
                if attempt == 31:
                    raise
        seen.add(key)
        puzzle["test" if is_test else "train"].append(pair)
        if not is_test:
            train_island |= island
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, lessons, training_flags = set(), [], set()
    train_island = False
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            info = _parse(inp)
            assert np.array_equal(_paint(info), out), "input-derived weaving disagrees"
            interior = np.logical_or.reduce(info["rooms"])
            assert np.array_equal(inp[~interior], out[~interior]), "noninterior corruption"
            assert all(np.array_equal(d, _erosion(m)) for d, m in zip(info["depths"], info["rooms"]))
            if split == "train":
                flags = _witnesses(info, out)
                training_flags.update(k for k, v in flags.items() if v)
                lessons.append(info)
                train_island |= _has_island(info)
            else:
                assert not _has_island(info) or train_island, "query island lacks a training demonstration"
                assert len(info["rooms"]) in (4, 6)
                positional = _positional(info)
                assert positional is not None and positional != info["matching"]
                assert any(len(info["keys"][i]) == len(info["keys"][j]) == 2
                           and info["depths"][i].max() >= 3 and info["depths"][j].max() >= 3
                           for i, j in enumerate(info["matching"]))
    assert {"omit_b", "box", "l1", "wrong_index", "concat", "positional"} <= training_flags
    first, second = lessons[:2]
    assert len(first["rooms"]) == 4 and first["crossings"]
    assert len({k[0] for k in first["keys"]}) == 4 and all(len(k) == 1 for k in first["keys"])
    assert len(second["rooms"]) == 2 and sorted(map(len, second["keys"])) == [1, 2]
    assert all(d.max() >= 4 for d in second["depths"])
    first_horizontal = first["normals"][0][0] == 0
    second_horizontal = second["normals"][0][0] == 0
    assert first_horizontal != second_horizontal


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = _grid(pair[role])
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming 42-seed coverage; no saved grids or unbounded retries."""
    signatures, shapes, structural, palettes, matches, axes = (set() for _ in range(6))
    room_counts, key_lengths, wall_islands, notched, crosses = set(), set(), 0, 0, 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                pairs += 1
                shapes.add(pair["input"].shape)
                palettes.add((info["background"], info["wall"], info["carrier"], info["cross"]))
                room_counts.add(len(info["rooms"]))
                key_lengths.update(map(len, info["keys"]))
                axes.add(int(info["normals"][0][0] == 0))
                crosses += len(info["crossings"])
                matches.add(tuple(info["matching"]))
                silhouette = np.logical_or.reduce(info["rooms"])
                structural.add((silhouette.shape, silhouette.tobytes()))
                for room in info["rooms"]:
                    rr, cc = np.nonzero(room)
                    notched += int(room.sum()) < (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
                    # A wall component wholly inside the room's bounding box is a hole.
                    local = pair["input"][rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                    wall_islands += any(not _touches_edge(m) for m in _components(local == info["wall"]))
    assert len(signatures) == 42 and pairs == 210
    assert room_counts == {2, 4, 6} and key_lengths == {1, 2} and axes == {0, 1}
    assert len(shapes) >= 35 and len(structural) >= 150 and len(palettes) >= 100
    assert len(matches) >= 6 and crosses >= 100 and notched >= 100 and wall_islands >= 1
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9001, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _digest(generate_puzzle(101, 2, 1)) == _digest(generate_puzzle(101, 2, 1))
    assert _digest(generate_puzzle(101, 2, 1)) != _digest(generate_puzzle(102, 2, 1))
    broken = generate_puzzle(777, 2, 1)
    info = _parse(broken["test"][0]["input"])
    p = next(iter(info["crossings"]))
    broken["test"][0]["output"][p] = info["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed crossing accepted")
    return {"seeds": 42, "pairs": pairs, "shapes": len(shapes),
            "silhouettes": len(structural), "role_assignments": len(palettes),
            "matching_signatures": len(matches), "crossings": crosses,
            "nonrectangular_chambers": int(notched), "wall_islands": int(wall_islands)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
