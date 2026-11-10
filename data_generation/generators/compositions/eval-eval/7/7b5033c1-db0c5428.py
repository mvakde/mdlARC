"""Matching-port relay: a single ordered sweep through a folded tile chain.

No parent generator, examples, or sampling metadata are used by solve().
Only Python's standard library and NumPy are required.
"""

import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PORTS = ((0, 1), (1, 2), (2, 1), (1, 0))
CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return (a.shape, tuple(map(int, a.flat)))


def _parse(value):
    """Recover blocks and the connector graph from visible pixels only."""
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > a.size / 2, "background must be a strict majority")
    h, w = a.shape
    _require(h >= 3 and w >= 3, "no complete tiles")
    occupied = a != bg
    full = occupied[:-2, :-2].copy()
    for dy in range(3):
        for dx in range(3):
            full &= occupied[dy:dy + h - 2, dx:dx + w - 2]
    origins = [tuple(map(int, q)) for q in np.argwhere(full)]
    _require(2 <= len(origins) <= 9, "expected 2..9 complete 3x3 tiles")
    centers = [(r + 1, c + 1) for r, c in origins]
    tile_mask = np.zeros(a.shape, dtype=bool)
    tiles = []
    for r, c in origins:
        _require(not tile_mask[r:r + 3, c:c + 3].any(), "ambiguous overlapping blocks")
        tile_mask[r:r + 3, c:c + 3] = True
        tiles.append(a[r:r + 3, c:c + 3].copy())
    loose = occupied & ~tile_mask
    loose_values = np.unique(a[loose])
    _require(len(loose_values) == 1, "expected one connector color")
    marker = int(loose_values[0])
    carrier = int(tiles[0][1, 1])
    _require(carrier != marker, "carrier and connector must differ")
    for t in tiles:
        _require(t[1, 1] == carrier, "tile centers must share a carrier")
        _require(not np.any(t == marker), "connector color occurs inside a tile")
        _require(all(t[p] != carrier for p in PORTS), "all four ports must be details")
    graph = {i: [] for i in range(len(tiles))}
    covered = np.zeros(a.shape, dtype=bool)
    pitches = []
    for i, (r, c) in enumerate(centers):
        for j in range(i + 1, len(centers)):
            y, x = centers[j]
            delta = abs(y - r) + abs(x - c)
            if (y != r and x != c) or not 4 <= delta <= 6:
                continue
            dy, dx = (int(np.sign(y - r)), int(np.sign(x - c)))
            bridge = [(r + k * dy, c + k * dx) for k in range(2, delta - 1)]
            if all(a[p] == marker for p in bridge):
                graph[i].append(j)
                graph[j].append(i)
                _require(not any(covered[p] for p in bridge), "connectors intersect")
                for p in bridge:
                    covered[p] = True
                pitches.append(delta)
    _require(np.array_equal(covered, loose), "unassigned connector pixels or noise")
    ends = [i for i, ns in graph.items() if len(ns) == 1]
    _require(len(ends) == 2 and all(1 <= len(ns) <= 2 for ns in graph.values()),
             "tile graph must be one unbranched chain")
    _require(centers[ends[0]][0] != centers[ends[1]][0], "endpoint row tie")
    current = min(ends, key=lambda i: centers[i][0])
    order, previous = [], None
    while True:
        _require(current not in order, "cycle")
        order.append(current)
        choices = [j for j in graph[current] if j != previous]
        if not choices:
            break
        _require(len(choices) == 1, "branch")
        previous, current = current, choices[0]
    _require(len(order) == len(tiles), "disconnected chain")
    ordered_centers = [centers[i] for i in order]
    directions = []
    for p, q in zip(ordered_centers, ordered_centers[1:]):
        d = (int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1])))
        directions.append(DIRS.index(d))
    return dict(grid=a, background=bg, carrier=carrier, marker=marker,
                centers=ordered_centers, tiles=[tiles[i] for i in order],
                directions=directions, tile_mask=tile_mask, pitches=pitches)


def _transform(tile, operation=2):
    if operation == 2:
        return tile[::-1, ::-1].copy()
    return np.rot90(tile if operation < 4 else tile[:, ::-1], operation % 4).copy()


def _sweep(tiles, directions, policy="dynamic", operation=2, reverse=False):
    """Array implementation. Alternative policies are used only for diagnostics."""
    current = [t.copy() for t in tiles]
    edges = list(range(len(directions)))
    if reverse:
        edges.reverse()
    for i in edges:
        d = directions[i]
        source = tiles if policy == "static" else current
        matched = source[i][PORTS[d]] == source[i + 1][PORTS[(d + 2) % 4]]
        if policy == "always":
            matched = True
        if not matched:
            if policy == "stop":
                break
            continue
        left, right = current[i], current[i + 1]
        current[i], current[i + 1] = _transform(right, operation), _transform(left, operation)
    return current


def _paint(info, tiles):
    answer = info["grid"].copy()
    for (r, c), tile in zip(info["centers"], tiles):
        answer[r - 1:r + 2, c - 1:c + 2] = tile
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Visit links from the higher endpoint; match current ports and swap half-turns."""
    info = _parse(input_grid)
    return _paint(info, _sweep(info["tiles"], info["directions"]))


def _audit(info):
    """Scalar source-coordinate invariant, without mutating tile arrays.

    Each position carries an original tile identity and a half-turn parity.
    A swap exchanges identities and toggles both parities. This also measures
    whether current-port comparison actually differs from static comparison.
    """
    original = info["tiles"]
    tokens = [(i, 0) for i in range(len(original))]
    events = []

    def read(token, point):
        identity, parity = token
        y, x = point
        return int(original[identity][2 - y if parity else y, 2 - x if parity else x])

    for i, d in enumerate(info["directions"]):
        p, q = PORTS[d], PORTS[(d + 2) % 4]
        left, right = read(tokens[i], p), read(tokens[i + 1], q)
        static = int(original[i][p]) == int(original[i + 1][q])
        matched = left == right
        events.append(dict(edge=i, direction=d, left=left, right=right, match=matched,
                           static_match=static, identities=(tokens[i][0], tokens[i + 1][0])))
        if matched:
            (a, ap), (b, bp) = tokens[i], tokens[i + 1]
            tokens[i], tokens[i + 1] = (b, 1 - bp), (a, 1 - ap)
    return tokens, events


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "canvas dimensions changed")
    info = _parse(a)
    _require(np.array_equal(a[~info["tile_mask"]], b[~info["tile_mask"]]),
             "background or connectors changed")
    tokens, events = _audit(info)
    for (r, c), (identity, parity) in zip(info["centers"], tokens):
        tile = info["tiles"][identity]
        for y in range(3):
            for x in range(3):
                _require(b[r - 1 + y, c - 1 + x] == tile[2 - y if parity else y,
                                                       2 - x if parity else x],
                         "incorrect relayed tile cell")
    _require(np.array_equal(np.bincount(a.ravel(), minlength=10),
                           np.bincount(b.ravel(), minlength=10)), "color counts changed")
    info["events"] = events
    return info


def _different(info, pair, **options):
    repeat = options.pop("repeat", False)
    result = _sweep(info["tiles"], info["directions"], **options)
    if repeat:
        result = _sweep(result, info["directions"], **options)
    return not np.array_equal(_paint(info, result), pair["output"])


def _corner_witness(info, pair):
    output = _grid(pair["output"])
    return any(info["grid"][r - 1 + y, c - 1 + x] != output[r - 1 + y, c - 1 + x]
               for r, c in info["centers"] for y, x in CORNERS)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(info["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    train = list(zip(infos["train"], puzzle["train"]))
    # Teaching checks concern answer differences, rather than merely motifs.
    for options in (dict(policy="static"), dict(policy="always"), dict(policy="stop"),
                    dict(reverse=True), dict(repeat=True)):
        _require(any(_different(i, p, **options) for i, p in train),
                 "demonstrations do not distinguish " + str(options))
    for operation in range(8):
        if operation != 2:
            _require(any(_different(i, p, operation=operation) for i, p in train),
                     "demonstrations leave a global tile-transform ambiguity")
    _require(any(any(e["match"] != e["static_match"] for e in i["events"]) for i, _ in train),
             "no current-state port witness")
    _require(any(_corner_witness(i, p) for i, p in train), "full-payload transport is not shown")
    if any(min(r for r, _ in i["centers"]) < i["centers"][0][0] for i in infos["test"]):
        _require(any(min(r for r, _ in i["centers"]) < i["centers"][0][0]
                     and _different(i, p, reverse=True) for i, p in train),
                 "above-endpoint query lacks teaching")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
        if not low <= int(value) <= high:
            raise ValueError("num_train must be 2..10; num_test must be 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _fallback(n, above):
    if above:
        points = [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2),
                  (2, 3), (3, 3), (4, 3)]
    elif n == 2:
        points = [(0, 0), (1, 0)]
    else:
        points = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0), (2, -1),
                  (3, -1), (4, -1), (4, 0)]
    return points[:n]


def _path(rng, n, allow_above, need_above=False, first_bend=False):
    """Grow an induced macro-grid walk, then orient by endpoint row."""
    for _ in range(100):
        points, occupied = [(0, 0)], {(0, 0)}
        for _ in range(n - 1):
            candidates = []
            for dy, dx in DIRS:
                q = (points[-1][0] + dy, points[-1][1] + dx)
                if q not in occupied and sum((q[0] + a, q[1] + b) in occupied
                                             for a, b in DIRS) == 1:
                    candidates.append(q)
            if not candidates:
                break
            q = candidates[int(rng.integers(len(candidates)))]
            points.append(q)
            occupied.add(q)
        if len(points) != n or points[0][0] == points[-1][0]:
            continue
        if points[0][0] > points[-1][0]:
            points.reverse()
        above = min(r for r, _ in points) < points[0][0]
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
        if (above and not allow_above) or (need_above and not above):
            continue
        if first_bend and steps[0] == steps[1]:
            continue
        if max(r for r, _ in points) - min(r for r, _ in points) > 4:
            continue
        if max(c for _, c in points) - min(c for _, c in points) > 4:
            continue
        return points
    points = _fallback(n, need_above)
    if rng.random() < .5:
        points = [(r, -c) for r, c in points]
    return points


def _tile(rng, carrier, details, corner_range):
    t = np.full((3, 3), carrier, dtype=np.int8)
    # A shared palette can recur freely; matching is positional, never unique-color based.
    style = int(rng.integers(3))
    ports = rng.choice(details, size=4)
    if style == 0:
        ports[:] = ports[0]
    elif style == 1:
        ports[2], ports[3] = ports[0], ports[1]
    for point, color in zip(PORTS, ports):
        t[point] = color
    count = int(rng.integers(corner_range[0], corner_range[1] + 1))
    chosen = rng.permutation(4)[:count]
    for j in chosen:
        t[CORNERS[int(j)]] = int(rng.choice(details))
    return t


def _embed(rng, points, tiles, background, marker):
    pr, pc = map(int, rng.integers(4, 7, size=2))
    mr, mc = min(r for r, _ in points), min(c for _, c in points)
    centers = [(1 + pr * (r - mr), 1 + pc * (c - mc)) for r, c in points]
    bh, bw = max(r for r, _ in centers) + 2, max(c for _, c in centers) + 2
    _require(bh <= 30 and bw <= 30, "construction does not fit")
    compact = rng.random() < .65
    h = min(30, bh + int(rng.integers(0, 5))) if compact else int(rng.integers(bh, 31))
    w = min(30, bw + int(rng.integers(0, 5))) if compact else int(rng.integers(bw, 31))
    occupied = 9 * len(points)
    for p, q in zip(centers, centers[1:]):
        occupied += abs(q[0] - p[0]) + abs(q[1] - p[1]) - 3
    # Background readability is a visible invariant, including border contact.
    while h * w <= 2 * occupied:
        if (h <= w and h < 30) or w == 30:
            h += 1
        else:
            w += 1
    top, left = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    if rng.random() < .45:
        if rng.random() < .5:
            top = int(rng.choice((0, h - bh)))
        else:
            left = int(rng.choice((0, w - bw)))
    centers = [(r + top, c + left) for r, c in centers]
    a = np.full((h, w), background, dtype=np.int8)
    for (r, c), t in zip(centers, tiles):
        a[r - 1:r + 2, c - 1:c + 2] = t
    for p, q in zip(centers, centers[1:]):
        dy, dx = int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1]))
        distance = abs(q[0] - p[0]) + abs(q[1] - p[1])
        for k in range(2, distance - 1):
            a[p[0] + k * dy, p[1] + k * dx] = marker
    return a, centers


def _sample_pair(rng, n, allow_above, need_above, kind, roles, corners, relay_length=2):
    points = _path(rng, n, allow_above, need_above, first_bend=kind == "cascade")
    directions = [DIRS.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(points, points[1:])]
    if roles is None:
        palette = list(map(int, rng.permutation(10)))
        background, carrier, marker = palette[:3]
    else:
        background, carrier, marker = roles
        palette = [background, carrier, marker] + list(map(int, rng.permutation(
            [c for c in range(10) if c not in roles])))
    details = palette[3:3 + int(rng.integers(3, 8))]
    tiles = [_tile(rng, carrier, details, corners) for _ in range(n)]
    # This asymmetric two-link witness couples rotation, direction and mutable state.
    if kind == "cascade":
        d0, d1 = directions[:2]
        a, b, c = details[:3]
        for p in PORTS:
            tiles[0][p] = c
        tiles[0][PORTS[d0]] = a
        tiles[0][PORTS[(d1 + 2) % 4]] = b
        tiles[1][PORTS[d1]] = c
        # One copied corner in a guaranteed exchange teaches full-payload transport,
        # including the regime whose training tiles otherwise have 0..1 corners.
        for p in CORNERS:
            tiles[0][p] = carrier
            tiles[1][p] = carrier
        tiles[1][CORNERS[int(rng.integers(4))]] = c
    targets = list(map(bool, rng.integers(0, 2, size=n - 1)))
    if kind == "cascade":
        targets[:2] = [True, True]
        targets[2:] = [False] * (n - 3)
    elif kind == "barrier":
        targets[:2] = [False, True]
        targets[2:] = [False] * (n - 3)
    elif kind == "relay":
        targets[:relay_length] = [True] * relay_length
    elif kind == "short":
        # Optional training cases cannot hide a longer relay demonstration.
        for i in range(2, len(targets)):
            if targets[i - 2] and targets[i - 1]:
                targets[i] = False
    work = [t.copy() for t in tiles]
    for i, (d, matched) in enumerate(zip(directions, targets)):
        value = int(work[i][PORTS[d]])
        incoming = PORTS[(d + 2) % 4]
        other = value if matched else int(rng.choice([c for c in details if c != value]))
        tiles[i + 1][incoming] = other
        work[i + 1][incoming] = other
        if matched:
            work[i], work[i + 1] = _transform(work[i + 1]), _transform(work[i])
    grid, centers = _embed(rng, points, tiles, background, marker)
    # Ground truth uses the constructed walk and tile state; solve must rediscover both.
    output = grid.copy()
    for (r, c), tile in zip(centers, work):
        output[r - 1:r + 2, c - 1:c + 2] = tile
    return dict(input=grid, output=output)


def _distinct(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # At least 90 distinct role assignments, versus at most 13 forbidden inputs.
    info = _parse(pair["input"])
    order = [info["background"], info["carrier"], info["marker"]]
    order += [c for c in range(10) if c not in order]
    for bg in range(10):
        for carrier in range(10):
            if bg == carrier:
                continue
            target = [bg, carrier] + [c for c in range(10) if c not in (bg, carrier)]
            mapping = np.empty(10, dtype=np.int8)
            mapping[order] = target
            grid = mapping[pair["input"]]
            if _key(grid) not in seen:
                return dict(input=grid, output=mapping[pair["output"]])
    raise AssertionError("distinct-input fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    allow_above = bool(rng.integers(2))
    regime = str(rng.choice(("length", "relay", "motif")))
    roles = tuple(map(int, rng.permutation(10)[:3])) if rng.random() < .35 else None
    train_corners = (0, 1) if regime == "motif" else (0, int(rng.integers(2, 5)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n), ("test", m)):
        for index in range(count):
            teaching = split == "train" and index < 2
            if teaching:
                kind = "cascade" if index == 0 else "barrier"
                above = allow_above and index == 0
                size = 6 if above else int(rng.integers(4, 7))
            elif split == "train":
                kind, above = "short", False
                size = int(rng.integers(2, 7))
            else:
                kind, above = "relay", False
                size = int(rng.integers(8, 10)) if regime == "length" else int(rng.integers(5, 10))
            corner_range = (2, 4) if split == "test" and regime == "motif" else train_corners
            relay_length = int(rng.integers(3, size)) if split == "test" and regime == "relay" else 2
            # Bounded retries ensure the two explicit teaching patterns affect outputs.
            pair = None
            for _ in range(40):
                candidate = _sample_pair(rng, size, allow_above, above, kind, roles,
                                         corner_range, relay_length)
                info = _check_pair(candidate)
                if kind == "cascade" and not all(_different(info, candidate, **opts) for opts in
                    [dict(policy="static"), dict(reverse=True)] +
                    [dict(operation=k) for k in range(8) if k != 2]):
                    continue
                if kind == "barrier" and not all(_different(info, candidate, policy=p)
                                                  for p in ("always", "stop")):
                    continue
                if split == "test" and np.array_equal(candidate["input"], candidate["output"]):
                    continue
                pair = _distinct(candidate, seen)
                break
            _require(pair is not None, "bounded teaching construction failed")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _check_pair(pair)
    centers, ds, events = info["centers"], info["directions"], info["events"]
    run = longest = 0
    for event in events:
        run = run + 1 if event["match"] else 0
        longest = max(longest, run)
    corners = tuple(sum(t[p] != info["carrier"] for p in CORNERS) for t in info["tiles"])
    rows = sorted({r for r, _ in centers})
    cols = sorted({c for _, c in centers})
    macro = tuple((rows.index(r), cols.index(c)) for r, c in centers)
    colors, structure = {}, []
    for t in info["tiles"]:
        for v in t.flat:
            structure.append(colors.setdefault(int(v), len(colors)))
    return dict(shape=info["grid"].shape, nodes=len(centers), directions=tuple(ds),
                turns=sum(a != b for a, b in zip(ds, ds[1:])), macro=macro,
                background=info["background"], carrier=info["carrier"], marker=info["marker"],
                pitch=tuple(sorted(set(info["pitches"]))), corners=corners,
                structure=tuple(structure), swaps=sum(e["match"] for e in events),
                longest_relay=longest, feedback=any(e["match"] != e["static_match"] for e in events),
                above=min(r for r, _ in centers) < centers[0][0],
                border=any(r in (1, info["grid"].shape[0] - 2) or
                           c in (1, info["grid"].shape[1] - 2) for r, c in centers))


def self_test():
    """Bounded 42-seed semantic, teaching, determinism and diversity checks."""
    records, signatures, regimes = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(all(np.array_equal(solve(p["input"]), p["output"])
                     for split in puzzle.values() for p in split), "solve disagrees")
        signature = tuple(_key(p[k]) for s in ("train", "test")
                          for p in puzzle[s] for k in ("input", "output"))
        signatures.add(signature)
        train = [_features(p) for p in puzzle["train"]]
        tests = [_features(p) for p in puzzle["test"]]
        records.extend(train + tests)
        if min(f["nodes"] for f in tests) > max(f["nodes"] for f in train):
            regimes.add("length")
        if min(f["longest_relay"] for f in tests) > max(f["longest_relay"] for f in train):
            regimes.add("relay")
        if min(min(f["corners"]) for f in tests) > max(max(f["corners"]) for f in train):
            regimes.add("motif")
        _require(any(f["feedback"] for f in train), "A/B interaction absent in training")
    _require(len(signatures) == 42, "puzzles repeated across seeds")
    _require(regimes == {"length", "relay", "motif"}, "train/test shift modes collapsed")
    _require({f["nodes"] for f in records} == set(range(2, 10)), "node count range collapsed")
    _require(all({f[k] for f in records} == set(range(10)) for k in
                 ("background", "carrier", "marker")), "literal role colors omitted")
    _require(len({f["shape"] for f in records}) >= 60, "canvas variation collapsed")
    _require(len({f["macro"] for f in records}) >= 80, "route geometry variation collapsed")
    _require(len({f["structure"] for f in records}) >= 150, "tile structure variation collapsed")
    _require({p for f in records for p in f["pitch"]} == {4, 5, 6}, "bridge lengths omitted")
    _require(sum(f["above"] for f in records) >= 12, "above-endpoint teaching missing")
    _require(sum(f["border"] for f in records) >= 30, "border placements missing")
    _require(any(0 in f["corners"] and max(f["corners"]) > 0 for f in records), "mixed decoration absent")
    _require(any(1 in f["directions"] and 3 in f["directions"] for f in records), "horizontal reversals absent")
    # Every legal count combination, and concrete illegal types as well as ranges.
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(901, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "counts ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(17), generate_puzzle(17)
    _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
                 for k in ("input", "output")), "nondeterminism")
    # Reject a wrong payload and a changed connector, not just a bad schema.
    info = _parse(p["test"][0]["input"])
    for point in (info["centers"][0], tuple(map(int, np.argwhere(
                  (info["grid"] == info["marker"]))[0]))):
        bad = {s: [{k: v.copy() for k, v in a.items()} for a in p[s]] for s in p}
        bad["test"][0]["output"][point] = (int(bad["test"][0]["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("output mutation accepted")
    # Tied endpoints, a cycle, and a branch are structurally different failures.
    malformed_routes = (
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)],
        [(0, 1), (1, 1), (1, 0), (1, 1), (1, 2)],
    )
    for route in malformed_routes:
        t = np.ones((3, 3), dtype=np.int8)
        for port in PORTS:
            t[port] = 3
        malformed, _ = _embed(np.random.default_rng(312), route,
                              [t.copy() for _ in route], 0, 2)
        try:
            solve(malformed)
        except AssertionError:
            continue
        raise AssertionError("malformed topology accepted")
    # Explicitly exercise a valid no-match chain; solve must permit identity members.
    identity = _sample_pair(np.random.default_rng(203), 2, False, False, "short", None, (0, 0))
    ii = _parse(identity["input"])
    d = ii["directions"][0]
    t = [a.copy() for a in ii["tiles"]]
    old = int(t[0][PORTS[d]])
    t[1][PORTS[(d + 2) % 4]] = next(c for c in range(10) if c not in
        (ii["background"], ii["carrier"], ii["marker"], old))
    a = _paint(ii, t)
    _require(np.array_equal(solve(a), a), "identity chain unsupported")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                shapes=len({f["shape"] for f in records}),
                macro_routes=len({f["macro"] for f in records}),
                tile_structures=len({f["structure"] for f in records}), shifts=sorted(regimes))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
