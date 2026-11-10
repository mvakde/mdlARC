"""Corner-relay mosaics: shuffled executable rectangles jointly tile a stencil.

Only NumPy and the standard library are required. No execution traces, caches,
grids, or logs are written by this module. See task_spec.md for the full rule.
"""
from itertools import permutations

import numpy as np


_PORTS = ((0, 0), (0, 1), (1, 0), (1, 1))
_MAX_NODES = 20000


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    while cells:
        first = min(cells)
        cells.remove(first)
        stack, component = [first], [first]
        while stack:
            y, x = stack.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in cells:
                    cells.remove(p)
                    stack.append(p)
                    component.append(p)
        result.append(component)
    return result


def _tile_ports(tile, port):
    """The four normalized two-dot glyphs: LL, LR, RL, RR."""
    h, w = tile.shape
    if h < 2 or w < 2:
        raise ValueError("Tiles need two distinct rows and columns")
    dots = set(map(tuple, np.argwhere(tile == port)))
    if len(dots) != 2:
        raise ValueError("A tile must have exactly two port dots")
    top = [x for y, x in dots if y == 0 and x in (0, w - 1)]
    bottom = [x for y, x in dots if y == h - 1 and x in (0, w - 1)]
    if len(top) != 1 or len(bottom) != 1:
        raise ValueError("Ports must be one upper and one lower corner")
    return top[0], bottom[0]


def _extract(tray, background, port):
    tiles = []
    for component in _components(tray != background):
        ys, xs = zip(*component)
        y, x = min(ys), min(xs)
        tile = tray[y:max(ys) + 1, x:max(xs) + 1].copy()
        if tile.size != len(component) or np.any(tile == background):
            raise ValueError("Tray objects must be completely filled rectangles")
        _tile_ports(tile, port)
        tiles.append(tile)
    if not tiles:
        raise ValueError("Empty tray")
    return tiles


def _relay(info, limit=2, available=None):
    """Exhaust distinct rendered answers, grouping interchangeable physical copies.

    The next rectangle occupies a wholly new band of rows. Covering every
    vacancy in that band is a consequence of the rule, not a layout heuristic.
    Only genuinely impossible suffixes enter the dead-state memo: an already
    seen rendered completion still counts as a feasible suffix.
    """
    board, mask, port = info["board"], info["mask"], info["port"]
    tiles = info["tiles"] if available is None else available
    groups, lookup = [], {}
    for tile in tiles:
        k = _key(tile)
        if k not in lookup:
            lookup[k] = len(groups)
            en, ex = _tile_ports(tile, port)
            groups.append({"tile": tile, "entry": en, "exit": ex, "count": 0})
        groups[lookup[k]]["count"] += 1
    counts = [g["count"] for g in groups]
    row_width = mask.sum(axis=1)
    row_left = np.argmax(mask, axis=1)
    start, finish = info["start"], info["finish"]
    rendered = board.copy()
    outcomes, dead = {}, set()
    nodes = 0

    def visit(row, col, path):
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES:
            raise ValueError("Relay search exceeded its finite node budget")
        if row == finish[0]:
            if col != finish[1]:
                return False
            k = _key(rendered)
            if k not in outcomes:
                outcomes[k] = {"output": rendered.copy(), "path": tuple(path)}
            return True
        state = row, col, tuple(counts)
        if state in dead or row > finish[0]:
            return False
        feasible = False
        for i, g in enumerate(groups):
            if not counts[i]:
                continue
            tile = g["tile"]
            h, w = tile.shape
            x = col - g["entry"]
            if (x < 0 or x + w > mask.shape[1] or row + h > finish[0]
                    or not np.all(row_width[row:row + h] == w)
                    or not np.all(row_left[row:row + h] == x)):
                continue
            counts[i] -= 1
            rendered[row:row + h, x:x + w] = tile
            feasible |= visit(row + h, x + g["exit"], path + [(i, row, x)])
            rendered[row:row + h, x:x + w] = board[row:row + h, x:x + w]
            counts[i] += 1
            if len(outcomes) >= limit:
                return True
        if not feasible:
            dead.add(state)
        return feasible

    visit(start[0] + 1, start[1], [])
    return list(outcomes.values()), groups, nodes


def _interpret(grid):
    """Recover panels, local colors, terminals, tiles, and answer from input only."""
    a = _grid(grid)
    candidates = []
    for transposed in (False, True):
        canonical = a.T if transposed else a
        h, w = canonical.shape
        for cut in range(2, w - 2):
            separator = int(canonical[0, cut])
            if (not np.all(canonical[:, cut] == separator)
                    or np.count_nonzero(canonical == separator) != h):
                continue
            tray, board = canonical[:, :cut], canonical[:, cut + 1:]
            colors = [int(c) for c in np.unique(board)]
            if len(colors) != 3:
                continue
            absent = [c for c in colors if not np.any(tray == c)]
            if len(absent) != 1:
                continue
            board_color = absent[0]
            for port in colors:
                if port == board_color:
                    continue
                markers = np.argwhere(board == port)
                if markers.shape != (2, 2) or markers[0, 0] == markers[1, 0]:
                    continue
                background = next(c for c in colors if c not in (port, board_color))
                mask = board == background
                ys = np.flatnonzero(mask.any(axis=1))
                if (len(ys) == 0 or ys[0] != markers[0, 0] + 1
                        or ys[-1] != markers[1, 0] - 1
                        or len(ys) != ys[-1] - ys[0] + 1):
                    continue
                # A relay advances downward through one interval per row.
                if any(np.count_nonzero(mask[y]) != np.flatnonzero(mask[y])[-1]
                       - np.flatnonzero(mask[y])[0] + 1 for y in ys):
                    continue
                try:
                    tiles = _extract(tray, background, port)
                except ValueError:
                    continue
                info = {"board": board.copy(), "tray": tray.copy(), "tiles": tiles,
                        "mask": mask, "background": background, "port": port,
                        "board_color": board_color, "separator": separator,
                        "start": tuple(map(int, markers[0])),
                        "finish": tuple(map(int, markers[1])),
                        "transposed": transposed}
                # Do not conceal an unresolved interpretation by dropping it.
                outcomes, groups, nodes = _relay(info)
                if len(outcomes) > 1:
                    raise ValueError("More than one rendered relay completion")
                if outcomes:
                    info.update(outcomes[0])
                    info.update(groups=groups, nodes=nodes)
                    candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("Input must identify one panel split and one completion")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve without seed, demonstrations, generator state, or stored examples."""
    info = _interpret(input_grid)
    out = info["output"]
    return (out.T if info["transposed"] else out).copy()


def _body(rng, shape, palette, port, glyph, asymmetric=False):
    h, w = shape
    mode = int(rng.integers(5))
    tile = np.full(shape, int(rng.choice(palette)), dtype=np.int8)
    yy, xx = np.indices(shape)
    if len(palette) > 1:
        second = int(rng.choice([c for c in palette if c != int(tile[0, 0])]))
        if mode == 1:
            tile[yy % 2 == 1] = second
        elif mode == 2:
            tile[xx % 2 == 1] = second
        elif mode == 3:
            tile[(yy + xx) % 2 == 1] = second
        elif mode == 4:
            tile = rng.choice(palette, size=shape).astype(np.int8)
        if asymmetric:
            tile.fill(palette[0])
            tile[0, (1 - glyph[0]) * (w - 1)] = palette[1]
    tile[0, glyph[0] * (w - 1)] = port
    tile[-1, glyph[1] * (w - 1)] = port
    return tile


def _route_only_witness(info):
    """Find another in-bounds ordering that reaches the terminals but misses holes.

    This is an ablation, deliberately ignoring the stencil mask. It examines at
    most 120 orders of a four/five-tile training example, never test permutations.
    """
    chosen = [info["groups"][g]["tile"] for g, _, _ in info["path"]]
    if len(chosen) > 5:
        return False
    for order in permutations(range(len(chosen))):
        out = info["board"].copy()
        row, col = info["start"]
        row += 1
        legal = True
        for i in order:
            tile = chosen[i]
            h, w = tile.shape
            en, ex = _tile_ports(tile, info["port"])
            x = col - en
            if x < 0 or x + w > out.shape[1] or row + h > info["finish"][0]:
                legal = False
                break
            out[row:row + h, x:x + w] = tile
            row, col = row + h, x + ex
        if (legal and (row, col) == info["finish"]
                and not np.array_equal(out, info["output"])):
            return True
    return False


def _geometry_swap(info):
    """Construct two different patterns that ordinary exact covering can exchange."""
    path, groups = info["path"], info["groups"]
    for i, (g, y, x) in enumerate(path):
        first = groups[g]["tile"]
        for g2, y2, x2 in path[i + 1:]:
            second = groups[g2]["tile"]
            if first.shape == second.shape and not np.array_equal(first, second):
                h, w = first.shape
                alternate = info["output"].copy()
                alternate[y:y + h, x:x + w] = second
                alternate[y2:y2 + h, x2:x2 + w] = first
                assert np.array_equal(alternate[~info["mask"]],
                                      info["board"][~info["mask"]])
                return not np.array_equal(alternate, info["output"])
    return False


def _scatter(rng, tiles, height, max_width, background):
    """Independent source positions; no source ordering, rotation, or shelf rule."""
    if max_width < max(t.shape[1] for t in tiles):
        return None
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    # At most eight layouts, with all legal coordinates considered at each step.
    for attempt in range(8):
        lower = max(max(t.shape[1] for t in tiles), max_width - 4)
        width = max_width if attempt == 7 else int(rng.integers(lower, max_width + 1))
        occupied = np.zeros((height, width), dtype=bool)
        positions = []
        for i in order:
            th, tw = tiles[i].shape
            legal = []
            for y in range(height - th + 1):
                for x in range(width - tw + 1):
                    if not occupied[max(0, y - 1):min(height, y + th + 1),
                                    max(0, x - 1):min(width, x + tw + 1)].any():
                        legal.append((y, x))
            if not legal:
                break
            y, x = legal[int(rng.integers(len(legal)))]
            occupied[y:y + th, x:x + tw] = True
            positions.append((i, y, x))
        if len(positions) == len(tiles):
            tray = np.full((height, width), background, dtype=np.int8)
            for i, y, x in positions:
                th, tw = tiles[i].shape
                tray[y:y + th, x:x + tw] = tiles[i]
            return tray
    return None


def _build_pair(rng, count, roles, transposed, surplus, repeat=False, teaching=False):
    background, board_color, separator, port = roles
    available_colors = [c for c in range(10) if c not in roles]
    for attempt in range(160):
        dims = [tuple(map(int, rng.integers((2, 2), (5, 6)))) for _ in range(count)]
        glyphs = list(_PORTS)
        glyphs.extend(_PORTS[int(rng.integers(4))] for _ in range(count - 4))
        rng.shuffle(glyphs)
        if teaching:
            # An equal-width joined rectangle hides this boundary. Its LL->LR
            # or RR->RL relay cannot be recovered by shape matching alone.
            mirror = int(rng.integers(2))
            motif = [(mirror, mirror), (mirror, 1 - mirror)]
            other = [g for g in _PORTS if g not in motif]
            rng.shuffle(other)
            insert = int(rng.integers(3))
            glyphs = other[:insert] + motif + other[insert:]
            a, b = insert, insert + 1
        else:
            pairs = [(i, j) for i in range(count) for j in range(i + 1, count)
                     if glyphs[i] != glyphs[j]]
            a, b = pairs[int(rng.integers(len(pairs)))]
        dims[b] = dims[a]
        if repeat:
            dims[-1], glyphs[-1] = dims[0], glyphs[0]
            # Keep all four command types after installing the identical copy.
            if set(glyphs) != set(_PORTS):
                continue
        palette_count = int(rng.integers(2 if teaching else 1, 7))
        palette = [int(c) for c in rng.choice(available_colors, palette_count, replace=False)]
        tiles = [_body(rng, d, palette, port, g, teaching and i == 0)
                 for i, (d, g) in enumerate(zip(dims, glyphs))]
        if repeat:
            tiles[-1] = tiles[0].copy()
        # Repeated copies are not tied to the last position in the relay.
        if repeat:
            order = list(map(int, rng.permutation(count)))
            tiles = [tiles[i] for i in order]
        upper, lower = map(int, rng.integers(0, 3, size=2))
        row, col = upper + 1, 0
        rects = []
        for tile in tiles:
            th, tw = tile.shape
            en, ex = _tile_ports(tile, port)
            x = col - en
            rects.append((row, x, th, tw))
            row, col = row + th, x + ex
        height = row + 1 + lower
        if height > 30:
            continue
        left = min(x for y, x, h, w in rects)
        right = max(x + w for y, x, h, w in rects)
        lm, rm = map(int, rng.integers(0, 4, size=2))
        width = right - left + lm + rm
        if not 3 <= width <= 14:
            continue
        shift = lm - left
        board = np.full((height, width), board_color, dtype=np.int8)
        start, finish = (upper, shift), (row, col + shift)
        board[start] = board[finish] = port
        intended = board.copy()
        for (y, x, th, tw), tile in zip(rects, tiles):
            board[y:y + th, x + shift:x + shift + tw] = background
            intended[y:y + th, x + shift:x + shift + tw] = tile
        info = {"board": board, "mask": board == background, "port": port,
                "tiles": tiles.copy(), "start": start, "finish": finish}
        try:
            outcomes, groups, _ = _relay(info)
        except ValueError:
            continue
        if len(outcomes) != 1 or not np.array_equal(outcomes[0]["output"], intended):
            continue
        info.update(outcomes[0])
        info["groups"] = groups
        if not _geometry_swap(info) or (teaching and not _route_only_witness(info)):
            continue
        # Surplus rectangles have a genuine local geometric fit: they are the
        # same size as a selected tile, but carry a different corner command.
        target_surplus = int(rng.integers(1, 3)) if surplus else 0
        for _ in range(16):
            if len(info["tiles"]) == count + target_surplus:
                break
            source = tiles[int(rng.integers(count))]
            en, ex = _tile_ports(source, port)
            glyph = (int(en > 0), int(ex > 0))
            alternatives = [g for g in _PORTS if g != glyph]
            g = alternatives[int(rng.integers(3))]
            decoy = _body(rng, source.shape, palette, port, g)
            trial = info["tiles"] + [decoy]
            try:
                answers, _, _ = _relay(info, available=trial)
            except ValueError:
                continue
            if len(answers) == 1 and np.array_equal(answers[0]["output"], intended):
                info["tiles"] = trial
        if len(info["tiles"]) != count + target_surplus:
            continue
        tray = _scatter(rng, info["tiles"], height, 29 - width, background)
        if tray is None:
            continue
        inp = np.concatenate((tray, np.full((height, 1), separator, dtype=np.int8),
                              board), axis=1)
        if transposed:
            inp, intended = inp.T, intended.T
        try:
            parsed = _interpret(inp)
        except ValueError:
            continue
        placed_glyphs = [(parsed["groups"][g]["entry"],
                          parsed["groups"][g]["exit"],
                          parsed["groups"][g]["tile"].shape) for g, _, _ in parsed["path"]]
        raster_glyphs = [(*_tile_ports(t, port), t.shape) for t in parsed["tiles"]]
        if teaching and placed_glyphs == raster_glyphs:
            continue
        if not np.array_equal(solve(inp), intended):
            continue
        return {"input": inp.copy(), "output": intended.copy()}
    raise RuntimeError("Bounded construction could not find an unambiguous relay")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    allow_surplus = bool(rng.integers(2))
    phase = bool(rng.integers(2))
    result, seen = {"train": [], "test": []}, set()
    previous = None
    for i in range(nt + nq):
        training = i < nt
        count = (4 if i == 0 else 5 if i == 1 else int(rng.integers(4, 6))) if training else int(rng.integers(6, 9))
        transposed = phase if i == 0 else not phase if i == 1 else bool(rng.integers(2))
        surplus = allow_surplus and (i == 0 or bool(rng.integers(2)))
        for retry in range(8):
            roles = tuple(map(int, rng.choice(10, 4, replace=False)))
            if previous is not None and roles == previous:
                continue
            pair = _build_pair(rng, count, roles, transposed, surplus,
                               repeat=i == 1, teaching=i == 0)
            k = _key(pair["input"])
            if k not in seen:
                seen.add(k)
                previous = roles
                result["train" if training else "test"].append(pair)
                break
        else:
            raise RuntimeError("Could not construct a distinct input")
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, features = set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        assert _key(inp) not in seen
        seen.add(_key(inp))
        info = _interpret(inp)
        canonical_out = out.T if info["transposed"] else out
        assert np.array_equal(canonical_out, info["output"])
        assert np.array_equal(canonical_out[~info["mask"]], info["board"][~info["mask"]])
        assert not np.any(canonical_out[info["mask"]] == info["background"])
        assert not np.any(canonical_out[info["mask"]] == info["board_color"])
        used = [g for g, _, _ in info["path"]]
        assert all(used.count(g) <= group["count"] for g, group in enumerate(info["groups"]))
        assert _geometry_swap(info), "Every pair must defeat geometry-only packing"
        glyphs = {(int(info["groups"][g]["entry"] > 0),
                   int(info["groups"][g]["exit"] > 0)) for g in used}
        assert glyphs == set(_PORTS)
        features.append((info, len(used), len(info["tiles"]) - len(used)))
    training, testing = features[:nt], features[nt:]
    assert max(n for _, n, _ in training) < min(n for _, n, _ in testing)
    assert all(n in (4, 5) for _, n, _ in training)
    assert all(6 <= n <= 8 for _, n, _ in testing)
    assert {i["transposed"] for i, _, _ in training} == {False, True}
    assert _route_only_witness(training[0][0]), "Training must defeat relay-only ordering"
    assert any(len({g for g, _, _ in info["path"]}) < len(info["path"])
               for info, _, _ in training), "Show finite repeated physical copies"
    assert not any(n for _, _, n in testing) or any(n for _, _, n in training)
    assert len({(i["background"], i["board_color"], i["separator"], i["port"])
                for i, _, _ in features}) >= 2


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed diversity, semantic ablations, inventory, and API checks."""
    signatures, shapes, silhouettes, roles, dimensions, glyphs, patterns = (set() for _ in range(7))
    lengths, orientations, surplus_counts = set(), set(), set()
    hidden_boundaries = repeated = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        for pair in p["train"] + p["test"]:
            info = _interpret(pair["input"])
            assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
            shapes.add(pair["input"].shape)
            silhouettes.add(_key(info["mask"]))
            roles.add((info["background"], info["board_color"], info["separator"], info["port"]))
            lengths.add(len(info["path"]))
            orientations.add(info["transposed"])
            surplus_counts.add(len(info["tiles"]) - len(info["path"]))
            profile = [(x, info["groups"][g]["tile"].shape[1]) for g, y, x in info["path"]]
            hidden_boundaries += any(a == b for a, b in zip(profile, profile[1:]))
            repeated += len(set(g for g, _, _ in info["path"])) < len(info["path"])
            for g in info["groups"]:
                dimensions.add(g["tile"].shape)
                glyphs.add((bool(g["entry"]), bool(g["exit"])))
                patterns.add(_key(g["tile"]))
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(silhouettes) >= 160
    assert len(roles) >= 150 and len(dimensions) == 12 and len(patterns) >= 300
    assert glyphs == set(_PORTS) and lengths == {4, 5, 6, 7, 8}
    assert orientations == {False, True} and surplus_counts == {0, 1, 2}
    assert hidden_boundaries >= 42 and repeated >= 42
    assert _signature(generate_puzzle(731)) == _signature(generate_puzzle(731))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # Scarcity and pixel ambiguity: one physical LL tile cannot run twice.
    t = np.array([[9, 2], [9, 3]], dtype=np.int8)
    board = np.full((6, 2), 8, dtype=np.int8)
    board[0, 0] = board[5, 0] = 9
    board[1:5] = 0
    info = {"board": board, "mask": board == 0, "port": 9,
            "start": (0, 0), "finish": (5, 0), "tiles": [t]}
    assert not _relay(info)[0]
    info["tiles"] = [t, t.copy()]
    assert len(_relay(info)[0]) == 1
    u = t.copy()
    u[0, 1] = 4
    info["tiles"] = [t, u]
    assert len(_relay(info)[0]) == 2
    # Damage an output body cell and ensure public validation catches it.
    p = generate_puzzle(20261022)
    damaged = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    info = _interpret(damaged["test"][0]["input"])
    r, c = map(int, np.argwhere(info["mask"])[0])
    if info["transposed"]:
        r, c = c, r
    damaged["test"][0]["output"][r, c] = info["background"]
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Semantic corruption accepted")
