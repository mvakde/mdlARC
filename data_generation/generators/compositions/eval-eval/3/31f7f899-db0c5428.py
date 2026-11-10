"""Channel-sorted radial echoes: an input-only, visibly keyed ARC family.

Eight possible angular slots are ordered N, NE, E, SE, S, SW, W, NW.
The center-facing pixel of each card is its channel. Within a channel, sort
spoke lengths while keeping each slot's card, stem color, and stem thickness.
Fill the tiny hub map and attach a half-turned card at each new stem endpoint.
"""
from __future__ import annotations

import numpy as np


DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _offsets(width):
    # Even thickness has one extra cell on the negative perpendicular side.
    return range(-(width // 2), (width - 1) // 2 + 1)


def _stem_points(direction, radius, length, width):
    dr, dc = DIRECTIONS[direction]
    return [(k * dr + t * dc, k * dc - t * dr)
            for k in range(radius + 2, radius + 2 + length)
            for t in _offsets(width)]


def _views(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _singleton_components(a, bg):
    """The hub is the only singleton under eight-neighbor connectivity."""
    points = set(map(tuple, np.argwhere(a != bg)))
    singletons = []
    while points:
        first = points.pop()
        todo, size = [first], 0
        while todo:
            r, c = todo.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (r + dy, c + dx)
                    if q in points:
                        points.remove(q)
                        todo.append(q)
        if size == 1:
            singletons.append(first)
    return singletons


def _analyze(value):
    a = _grid(value)
    bg = int(a[0, 0])
    hubs = _singleton_components(a, bg)
    _require(len(hubs) == 1, "expected one isolated hub pixel")
    hr, hc = map(int, hubs[0])
    hub = int(a[hr, hc])
    _require(1 <= hr < a.shape[0] - 1 and 1 <= hc < a.shape[1] - 1,
             "hub map must fit")
    candidates = []
    for radius in (3, 4, 5):
        rebuilt = np.full_like(a, bg)
        rebuilt[hr, hc] = hub
        cards = []
        valid = True
        for direction, (dr, dc) in enumerate(DIRECTIONS):
            r, c = hr + radius * dr, hc + radius * dc
            if not (1 <= r < a.shape[0] - 1 and 1 <= c < a.shape[1] - 1):
                continue
            tile = a[r - 1:r + 2, c - 1:c + 2]
            if np.all(tile == bg):
                continue
            if np.any(tile == bg) or np.any(tile == hub):
                valid = False
                break
            counts = np.bincount(tile.ravel(), minlength=10)
            base = int(np.argmax(counts))
            channel = int(tile[1 - dr, 1 - dc])
            if (counts[base] < 5 or channel == base or
                    tile[1, 1] != base or tile[1 + dr, 1 + dc] != base):
                valid = False
                break
            length = 0
            for k in range(radius + 2, 12):
                y, x = hr + k * dr, hc + k * dc
                if not (0 <= y < a.shape[0] and 0 <= x < a.shape[1]) or a[y, x] != base:
                    break
                length += 1
            if not 1 <= length <= 10 - radius:
                valid = False
                break
            present = []
            for t in (-1, 0, 1):
                k = radius + 2
                y, x = hr + k * dr + t * dc, hc + k * dc - t * dr
                if 0 <= y < a.shape[0] and 0 <= x < a.shape[1] and a[y, x] == base:
                    present.append(t)
            widths = [w for w in (1, 2, 3) if list(_offsets(w)) == present]
            if len(widths) != 1:
                valid = False
                break
            width = widths[0]
            rebuilt[r - 1:r + 2, c - 1:c + 2] = tile
            for y, x in _stem_points(direction, radius, length, width):
                yy, xx = hr + y, hc + x
                if not (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]):
                    valid = False
                    break
                rebuilt[yy, xx] = base
            cards.append(dict(direction=direction, base=base, channel=channel,
                              length=length, width=width, tile=tile.copy()))
        channel_colors = {c["channel"] for c in cards}
        base_colors = {c["base"] for c in cards}
        if (valid and 3 <= len(cards) <= 8 and 1 <= len(channel_colors) <= 4
                and not channel_colors & base_colors and np.array_equal(rebuilt, a)):
            # A candidate must have enough space for its actual sorted echoes.
            assigned = _assigned_lengths(cards)
            for card, length in zip(cards, assigned):
                dr, dc = DIRECTIONS[card["direction"]]
                distance = radius + 3 + length
                r, c = hr + distance * dr, hc + distance * dc
                valid &= 1 <= r < a.shape[0] - 1 and 1 <= c < a.shape[1] - 1
            if valid:
                candidates.append(dict(grid=a, background=bg, hub=hub, center=(hr, hc),
                                       radius=radius, cards=cards))
    _require(len(candidates) == 1, "source radius or card geometry is ambiguous")
    return candidates[0]


def _assigned_lengths(cards, order=tuple(range(8)), metric="length", group="channel"):
    """A local numerical helper, with no stored puzzle state."""
    positions = {d: i for i, d in enumerate(order)}
    answer = [c["length"] for c in cards]
    for channel in {c[group] for c in cards}:
        indices = sorted((i for i, c in enumerate(cards) if c[group] == channel),
                         key=lambda i: positions[cards[i]["direction"]])
        def score(i):
            factor = cards[i]["width"] if metric == "area" else 1
            if metric == "euclidean":
                dr, dc = DIRECTIONS[cards[i]["direction"]]
                factor = float(np.sqrt(dr * dr + dc * dc))
            return cards[i]["length"] * factor
        ranked = sorted(indices, key=score)
        for destination, source in zip(indices, ranked):
            answer[destination] = cards[source]["length"]
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every role and length from the input; use only the fixed rule."""
    info = _analyze(input_grid)
    a, cards = info["grid"], info["cards"]
    hr, hc = info["center"]
    radius, bg = info["radius"], info["background"]
    out = a.copy()
    # Clear old stems, without moving or modifying the source cards.
    for card in cards:
        for y, x in _stem_points(card["direction"], radius, card["length"], card["width"]):
            out[hr + y, hc + x] = bg
    # Select order statistics directly from each visible channel's multiset.
    remaining = {}
    for card in cards:
        remaining.setdefault(card["channel"], []).append(card["length"])
    for card in cards:  # Parser supplies the fixed N-clockwise order.
        values = remaining[card["channel"]]
        length = min(values)
        values.remove(length)
        dr, dc = DIRECTIONS[card["direction"]]
        out[hr + dr, hc + dc] = card["channel"]
        for y, x in _stem_points(card["direction"], radius, length, card["width"]):
            out[hr + y, hc + x] = card["base"]
        distance = radius + 3 + length
        r, c = hr + distance * dr, hc + distance * dc
        out[r - 1:r + 2, c - 1:c + 2] = card["tile"][::-1, ::-1]
    return out


def _make_tile(rng, direction, base, channel, bg, hub, witness=False, repeat=False):
    dr, dc = DIRECTIONS[direction]
    inward = (1 - dr, 1 - dc)
    reserved = {inward, (1, 1), (1 + dr, 1 + dc)}
    positions = [p for p in np.ndindex(3, 3) if p not in reserved]
    auxiliary = [c for c in range(10) if c not in (bg, hub, base, channel)]
    if witness:
        # Two differently colored, noncollinear marks distinguish all eight D4 views.
        possibilities = []
        for p in positions:
            tile = np.full((3, 3), base, dtype=np.int8)
            tile[inward] = channel
            tile[p] = auxiliary[0]
            if len({v.tobytes() for v in _views(tile)}) == 8:
                possibilities.append(tile)
        tile = possibilities[int(rng.integers(len(possibilities)))].copy()
        tile[tile == auxiliary[0]] = int(rng.choice(auxiliary))
        if repeat:
            extras = [p for p in positions if tile[p] == base]
            rng.shuffle(extras)
            for p in extras[:int(rng.integers(1, 3))]:
                tile[p] = channel
        return tile
    tile = np.full((3, 3), base, dtype=np.int8)
    tile[inward] = channel
    count = int(rng.choice(4, p=[.25, .3, .3, .15]))
    rng.shuffle(positions)
    color = int(rng.choice([channel] + auxiliary))
    for p in positions[:count]:
        tile[p] = color if rng.random() < .7 else int(rng.choice([channel] + auxiliary))
    return tile


def _plan_grid(rng, radius, cards, bg, hub):
    """Constructor paints a coordinate plan, independently of the solver parser."""
    inputs, outputs = {(0, 0): hub}, {(0, 0): hub}
    targets = _assigned_lengths(cards)
    for card, length in zip(cards, targets):
        dr, dc = DIRECTIONS[card["direction"]]
        for y, x in np.ndindex(3, 3):
            point = (radius * dr + y - 1, radius * dc + x - 1)
            inputs[point] = outputs[point] = int(card["tile"][y, x])
        for point in _stem_points(card["direction"], radius, card["length"], card["width"]):
            _require(point not in inputs, "input shapes overlap")
            inputs[point] = card["base"]
        outputs[(dr, dc)] = card["channel"]
        for point in _stem_points(card["direction"], radius, length, card["width"]):
            _require(point not in outputs, "output stem overlaps a source card")
            outputs[point] = card["base"]
        distance = radius + 3 + length
        for y, x in np.ndindex(3, 3):
            point = (distance * dr + y - 1, distance * dc + x - 1)
            _require(point not in outputs, "echoes overlap another shape")
            outputs[point] = int(card["tile"][2 - y, 2 - x])
    points = set(inputs) | set(outputs)
    # The complete 3x3 hub map must fit, including missing-direction background.
    points |= {(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)}
    low = [min(p[k] for p in points) for k in (0, 1)]
    high = [max(p[k] for p in points) for k in (0, 1)]
    shape, origin = [], []
    for lo, hi in zip(low, high):
        extent = hi - lo + 1
        _require(extent <= 30, "construction exceeds ARC bounds")
        extra = int(rng.integers(0, min(5, 30 - extent) + 1))
        before = int(rng.integers(extra + 1))
        shape.append(extent + extra)
        origin.append(before - lo)
    pair = {name: np.full(shape, bg, dtype=np.int8) for name in ("input", "output")}
    for name, cells in (("input", inputs), ("output", outputs)):
        for (r, c), color in cells.items():
            pair[name][r + origin[0], c + origin[1]] = color
    _require(pair["input"][0, 0] == bg, "background corner unavailable")
    return pair


def _sample_pair(rng, config, radius, directions, witness=0, repeat_direction=None,
                 test=False, ensure_change=False):
    n = len(directions)
    colors = [int(c) for c in rng.permutation(10)]
    bg, hub = config["roles"] if config["fixed_roles"] else colors[:2]
    pool = [c for c in colors if c not in (bg, hub)]
    maximum_groups = min(4, n - int(ensure_change))
    ng = 2 if witness == 1 else 3 if witness == 2 else int(rng.integers(1, maximum_groups + 1))
    group_colors = pool[:ng]
    bases = pool[ng:]
    labels = {}
    if witness == 1:
        middle = int(rng.choice([d for d in directions if d not in (0, 7)]))
        labels = {d: int(d not in (0, middle, 7)) for d in directions}
    else:
        shuffled = list(directions)
        rng.shuffle(shuffled)
        for j, d in enumerate(shuffled):
            labels[d] = j % ng
    cap = 10 - radius
    if config["shift"] == "length" and not test:
        cap = min(cap, 5)
    lengths = [int(rng.integers(1, cap + 1)) for _ in directions]
    widths = [int(rng.integers(1, 4)) for _ in directions]
    if witness == 1:
        indices = [i for i, d in enumerate(directions) if labels[d] == 0]
        # The N arm is longer in grid steps than NW but shorter in Euclidean
        # distance. This demonstrates one common metric for cardinal/diagonal arms.
        low, medium, high = ((2, 3, 4), (2, 4, 5), (3, 4, 5))[int(rng.integers(3))]
        values = [high, low, medium]
        for i, length in zip(indices, values):
            lengths[i] = length
            widths[i] = 3 if length == low else 1 if length == high else int(rng.integers(1, 4))
        other = [i for i, d in enumerate(directions) if labels[d] == 1]
        lengths[other[0]], lengths[other[-1]] = 5, 1
    elif witness == 2:
        repeated = next(g for g in range(ng) if sum(labels[d] == g for d in directions) >= 2)
        indices = [i for i, d in enumerate(directions) if labels[d] == repeated]
        lengths[indices[0]], lengths[indices[-1]] = 4, 1
    if ensure_change:
        repeated_group = next(g for g in range(ng) if sum(labels[d] == g for d in directions) >= 2)
        query_indices = [i for i, d in enumerate(directions) if labels[d] == repeated_group]
        lengths[query_indices[0]], lengths[query_indices[-1]] = cap, 1
    elif test and config["shift"] == "length":
        lengths[int(rng.integers(n))] = cap
    cards = []
    for i, d in enumerate(directions):
        base = bases[0] if witness == 1 or config["common_base"] else int(rng.choice(bases))
        if witness == 2:
            base = bases[i % len(bases)]
            if i == indices[0]:
                base = bases[0]
            elif i == indices[-1]:
                base = bases[1]
        channel = group_colors[labels[d]]
        tile = _make_tile(rng, d, base, channel, bg, hub,
                          witness=bool(witness), repeat=d == repeat_direction)
        cards.append(dict(direction=d, base=base, channel=channel, length=lengths[i],
                          width=widths[i], tile=tile))
    return _plan_grid(rng, radius, cards, bg, hub)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be {low}..{high}")
    return int(value)


def _distinct(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # At most 13 preceding inputs forbid only 13 of the 90 role permutations.
    info = _analyze(pair["input"])
    old = [info["background"], info["hub"]] + [c for c in range(10)
            if c not in (info["background"], info["hub"])]
    for bg in range(10):
        for hub in range(10):
            if bg == hub:
                continue
            new = [bg, hub] + [c for c in range(10) if c not in (bg, hub)]
            mapping = np.empty(10, dtype=np.int8)
            mapping[old] = new
            candidate = {k: mapping[a] for k, a in pair.items()}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise AssertionError("finite palette fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["radius", "count", "length"]))
    radius = int(rng.integers(3, 5 if shift != "count" else 6))
    config = dict(shift=shift, roles=tuple(map(int, rng.choice(10, 2, replace=False))),
                  fixed_roles=bool(rng.random() < .3), common_base=bool(rng.random() < .25))
    n1 = int(rng.integers(5, 9))
    first = sorted([0, 7] + list(map(int, rng.choice(range(1, 7), n1 - 2, replace=False))))
    n2 = int(rng.choice([n for n in range(5, 9) if n != n1]))
    missing = [d for d in range(8) if d not in first]
    second = sorted(missing + list(map(int, rng.choice(first, n2 - len(missing), replace=False))))
    repeated = int(rng.choice([d for d in second if d in first]))
    train = [_sample_pair(rng, config, radius, first, witness=1),
             _sample_pair(rng, config, radius, second, witness=2, repeat_direction=repeated)]
    for _ in range(nt - 2):
        n = int(rng.integers(5 if shift == "count" else 3, 9))
        directions = sorted(map(int, rng.choice(8, n, replace=False)))
        train.append(_sample_pair(rng, config, radius, directions))
    tests = []
    for i in range(nq):
        n = int(rng.integers(3, 5 if shift == "count" else 9))
        directions = sorted(map(int, rng.choice(8, n, replace=False)))
        tests.append(_sample_pair(rng, config, radius + int(shift == "radius"), directions,
                                  test=True, ensure_change=i == 0))
    rng.shuffle(train)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, pairs in (("train", train), ("test", tests)):
        for pair in pairs:
            pair = _distinct(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    """Check rank/multiset relations and scalar destination coordinates."""
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "canvas changed")
    info = _analyze(a)
    hr, hc = info["center"]
    radius, bg, cards = info["radius"], info["background"], info["cards"]
    expected = {(hr, hc): info["hub"]}
    targets = []
    for card in cards:
        # Rank via repeated removal, rather than construction's argsort mapping.
        group = [c for c in cards if c["channel"] == card["channel"]]
        rank = sum(c["direction"] < card["direction"] for c in group)
        values = [c["length"] for c in group]
        for _ in range(rank):
            values.remove(min(values))
        length = min(values)
        targets.append(length)
        dr, dc = DIRECTIONS[card["direction"]]
        expected[hr + dr, hc + dc] = card["channel"]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                expected[hr + radius * dr + dy, hc + radius * dc + dx] = int(card["tile"][dy + 1, dx + 1])
                distance = radius + 3 + length
                expected[hr + distance * dr - dy, hc + distance * dc - dx] = int(card["tile"][dy + 1, dx + 1])
        for k in range(radius + 2, radius + 2 + length):
            for t in range(-(card["width"] // 2), (card["width"] - 1) // 2 + 1):
                expected[hr + k * dr + t * dc, hc + k * dc - t * dr] = card["base"]
    for r, c in np.ndindex(a.shape):
        _require(b[r, c] == expected.get((r, c), bg), "incorrect output cell or preservation")
    _require(set(map(int, np.unique(a))) == set(map(int, np.unique(b))), "palette changed")
    info["targets"] = targets
    return info


def _evidence(train):
    positions, operations = [set(range(9)) for _ in range(8)], [set(range(8)) for _ in range(8)]
    orders = [(tuple((start + sign * k) % 8 for k in range(8)))
              for sign in (1, -1) for start in range(8)]
    survivors = set(range(16))
    features = []
    for pair in train:
        info = _check_pair(pair)
        cards = info["cards"]
        b = np.asarray(pair["output"])
        hr, hc = info["center"]
        for card, length in zip(cards, info["targets"]):
            d, tile = card["direction"], card["tile"]
            dr, dc = DIRECTIONS[d]
            sample = int(b[hr + dr, hc + dc])
            distance = info["radius"] + 3 + length
            r, c = hr + distance * dr, hc + distance * dc
            target = b[r - 1:r + 2, c - 1:c + 2]
            positions[d].intersection_update(i for i in range(9) if tile.flat[i] == sample)
            operations[d].intersection_update(i for i, v in enumerate(_views(tile)) if np.array_equal(v, target))
        survivors.intersection_update(i for i, order in enumerate(orders)
                                      if _assigned_lengths(cards, order) == info["targets"])
        lengths = [c["length"] for c in cards]
        reverse = lengths.copy()
        for g in {c["channel"] for c in cards}:
            indices = [i for i, c in enumerate(cards) if c["channel"] == g]
            for i, value in zip(indices, [lengths[j] for j in indices[::-1]]):
                reverse[i] = value
        features.append(dict(info=info, changed=lengths != info["targets"],
                             area=_assigned_lengths(cards, metric="area") != info["targets"],
                             euclidean=_assigned_lengths(cards, metric="euclidean") != info["targets"],
                             global_sort=sorted(lengths) != info["targets"],
                             reverse=reverse != info["targets"],
                             modal=_assigned_lengths(cards, group="base") != info["targets"]))
    return positions, operations, [orders[i] for i in sorted(survivors)], features


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "bad example counts")
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
    positions, operations, orders, features = _evidence(puzzle["train"])
    _require(orders == [tuple(range(8))], "angular ordering lacks training evidence")
    _require(len({len(f["info"]["cards"]) for f in features}) >= 2, "training counts do not vary")
    _require(sum(f["changed"] for f in features) >= 2, "two sorting demonstrations required")
    for name in ("area", "euclidean", "global_sort", "reverse", "modal"):
        _require(any(f[name] for f in features), f"training permits {name} shortcut")
    _require(any(len({c["base"] for c in f["info"]["cards"]}) > 1 and f["changed"]
                 for f in features), "fixed destination colors lack evidence")
    _require(any(np.count_nonzero(c["tile"] == c["channel"]) > 1
                 for f in features for c in f["info"]["cards"]), "repeated sampling color not shown")
    _require(any([c["length"] for c in _analyze(p["input"])["cards"]] !=
                 _assigned_lengths(_analyze(p["input"])["cards"]) for p in puzzle["test"]),
             "at least one query must need sorting")
    for pair in puzzle["test"]:
        info = _analyze(pair["input"])
        for card in info["cards"]:
            d, tile = card["direction"], card["tile"]
            _require(bool(positions[d]) and bool(operations[d]), "direction not demonstrated")
            _require({int(tile.flat[i]) for i in positions[d]} == {card["channel"]},
                     "query sample location is ambiguous")
            _require(all(np.array_equal(_views(tile)[i], tile[::-1, ::-1]) for i in operations[d]),
                     "query half-turn is ambiguous")


def self_test():
    """Bounded 42-seed diversity, interface, ablation, and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity collapsed")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterministic")
    shapes, layouts, structures, counts, widths, radii, channels, roles = (set() for _ in range(8))
    ties = empty_directions = repeated = mixed_bases = 0
    shifts = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = {}
        for split in ("train", "test"):
            infos[split] = []
            for pair in puzzle[split]:
                info = _check_pair(pair)
                infos[split].append(info)
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solver disagrees")
                cards = info["cards"]
                shapes.add(tuple(np.asarray(pair["input"]).shape))
                layouts.add((info["radius"], info["center"], tuple(
                    (c["direction"], c["width"], c["length"]) for c in cards)))
                normalized = {}
                structures.add(tuple(normalized.setdefault(int(v), len(normalized))
                                     for v in np.asarray(pair["input"]).flat))
                counts.add(len(cards))
                widths.update(c["width"] for c in cards)
                radii.add(info["radius"])
                channels.add(len({c["channel"] for c in cards}))
                roles.add((info["background"], info["hub"]))
                ties += any(len([c["length"] for c in cards if c["channel"] == g]) >
                            len({c["length"] for c in cards if c["channel"] == g})
                            for g in {c["channel"] for c in cards})
                empty_directions += len(cards) < 8
                repeated += any(np.count_nonzero(c["tile"] == c["channel"]) > 1 for c in cards)
                mixed_bases += len({c["base"] for c in cards}) > 1
        if max(i["radius"] for i in infos["test"]) > max(i["radius"] for i in infos["train"]):
            shifts.add("radius")
        if max(len(i["cards"]) for i in infos["test"]) < min(len(i["cards"]) for i in infos["train"]):
            shifts.add("count")
        if max(c["length"] for i in infos["test"] for c in i["cards"]) > max(
                c["length"] for i in infos["train"] for c in i["cards"]):
            shifts.add("length")
    _require(counts == set(range(3, 9)), "slot counts omitted")
    _require(widths == {1, 2, 3} and radii == {3, 4, 5}, "geometry variation missing")
    _require(channels == {1, 2, 3, 4}, "channel variation missing")
    _require(len(shapes) >= 40 and len(layouts) >= 150 and len(structures) >= 150, "structure diversity low")
    _require(len(roles) >= 40 and min(ties, empty_directions, repeated, mixed_bases) >= 20,
             "role or structural combinations missing")
    _require(shifts == {"radius", "count", "length"}, "test shifts omitted")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(4242, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Corrupt each functional part, including old-stem erasure and unused background.
    base = puzzles[0]
    pair = base["test"][0]
    info = _analyze(pair["input"])
    hr, hc = info["center"]
    card = info["cards"][0]
    dr, dc = DIRECTIONS[card["direction"]]
    length = _assigned_lengths(info["cards"])[0]
    echo = info["radius"] + 3 + length
    points = [(hr + dr, hc + dc), (hr + info["radius"] * dr, hc + info["radius"] * dc),
              (hr + echo * dr, hc + echo * dc), (0, 0),
              (hr + (info["radius"] + 2) * dr, hc + (info["radius"] + 2) * dc)]
    erased = np.argwhere((pair["input"] != info["background"]) &
                         (pair["output"] == info["background"]))
    _require(len(erased) > 0, "seed-0 query must exercise vacated stem erasure")
    points.append(tuple(map(int, erased[0])))
    for point in points:
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in base[s]] for s in base}
        bad["test"][0]["output"][point] = (int(bad["test"][0]["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("validator accepted output corruption")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "shapes": len(shapes), "layouts": len(layouts), "structures": len(structures),
            "slot_counts": sorted(counts), "widths": sorted(widths), "radii": sorted(radii),
            "channel_counts": sorted(channels), "role_pairs": len(roles), "shifts": sorted(shifts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
