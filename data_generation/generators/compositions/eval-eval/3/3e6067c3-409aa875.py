"""Rendezvous weaving: a color itinerary through projected collision hubs.

No parent code is imported. All solving conventions are fixed and input-visible.
"""
from itertools import combinations

import numpy as np


DISTANCE = 5
DIRECTIONS = tuple((dr, dc) for dr in (-1, 0, 1)
                   for dc in (-1, 0, 1) if dr or dc)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return frozenset((r + dy, c + dx) for dy, dx in offsets)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            neighbors = {(r + dy, c + dx) for dy in (-1, 0, 1)
                         for dx in (-1, 0, 1)} & remaining
            remaining -= neighbors
            comp |= neighbors
            stack.extend(neighbors)
        result.append(frozenset(comp))
    return result


def _square(hub):
    r, c, label, count = hub
    radius = count - 1
    return frozenset((y, x) for y in range(r - radius, r + radius + 1)
                     for x in range(c - radius, c + radius + 1))


def _band(a, b):
    """Source-width ribbon strictly between the two virtual square edges."""
    r, c, _, n = a
    y, x, _, m = b
    ra, rb = n - 1, m - 1
    if r == y and abs(x - c) > ra + rb + 1:
        lo, hi = (c + ra + 1, x - rb) if c < x else (x + rb + 1, c - ra)
        return frozenset((yy, xx) for yy in range(r - ra, r + ra + 1)
                         for xx in range(lo, hi))
    if c == x and abs(y - r) > ra + rb + 1:
        lo, hi = (r + ra + 1, y - rb) if r < y else (y + rb + 1, r - ra)
        return frozenset((yy, xx) for yy in range(lo, hi)
                         for xx in range(c - ra, c + ra + 1))
    return frozenset()


def _graph(hubs):
    squares = [_square(h) for h in hubs]
    graph, bands = [[] for _ in hubs], {}
    for i, a in enumerate(hubs):
        for j, b in enumerate(hubs):
            if i == j:
                continue
            band = _band(a, b)
            if band and not any(band & square for square in squares):
                graph[i].append(j)
                bands[i, j] = band
    return graph, bands


def _routes(hubs, key, limit=2):
    graph, bands = _graph(hubs)
    candidates = [[i for i, h in enumerate(hubs) if h[2] == color]
                  for color in key]
    answers = []

    def visit(path, painted):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if i in path:
                continue
            if path:
                if i not in graph[path[-1]]:
                    continue
                band = bands[path[-1], i]
                if band & painted:
                    continue
            else:
                band = frozenset()
            visit(path + [i], painted | band)

    visit([], frozenset())
    return answers


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in
               (a[0], a[-1], a[:, 0], a[:, -1])), "Blank outer frame required"
    tokens, arrows = [], []
    for comp in _components(a != bg):
        if len(comp) == 1:
            tokens.append(next(iter(comp)))
            continue
        assert len(comp) == 3, "Only isolated three-cell arrowheads and key tokens"
        assert len({int(a[p]) for p in comp}) == 1
        matches = [(tip, direction) for tip in sorted(comp) for direction in DIRECTIONS
                   if _arrow(tip, direction) == comp]
        assert len(matches) == 1, "Arrow decomposition must be unique"
        tip, direction = matches[0]
        arrows.append((tip, direction, int(a[tip]), comp))
    assert arrows and 3 <= len(tokens) <= 9
    tokens.sort(key=lambda p: p[1])
    key_y = tokens[0][0]
    assert all(y == key_y for y, x in tokens)
    assert all(b[1] - p[1] == 2 for p, b in zip(tokens, tokens[1:]))
    assert key_y >= max(y for _, _, _, cells in arrows for y, x in cells) + 2
    key = [int(a[p]) for p in tokens]
    grouped = {}
    for tip, direction, color, cells in arrows:
        endpoint = tuple(tip[k] + DISTANCE * direction[k] for k in (0, 1))
        r, c = endpoint
        assert 1 <= r < key_y - 1 and 1 <= c < a.shape[1] - 1
        assert int(a[endpoint]) == bg, "Every rendezvous center is initially blank"
        grouped.setdefault(endpoint, []).append(color)
    hubs = []
    for (r, c), colors in sorted(grouped.items()):
        assert len(set(colors)) == 1, "Coincident shots must agree on their label"
        assert 1 <= len(colors) <= 3
        hub = (r, c, colors[0], len(colors))
        assert all(1 <= y < key_y - 1 and 1 <= x < a.shape[1] - 1
                   for y, x in _square(hub))
        hubs.append(hub)
    assert 3 <= len(hubs) <= 10
    squares = [_square(h) for h in hubs]
    assert all(not (squares[i] & squares[j]) for i in range(len(hubs))
               for j in range(i))
    assert len(set(key)) >= 2
    return dict(grid=a, background=bg, key=key, key_y=key_y,
                hubs=hubs, arrows=arrows)


def _paint(a, bg, hubs, route):
    out = a.copy()
    strokes = [(_square(hubs[i]), hubs[i][2]) for i in route]
    strokes += [(_band(hubs[i], hubs[j]), hubs[i][2])
                for i, j in zip(route, route[1:])]
    occupied = set()
    for cells, color in strokes:
        assert not occupied.intersection(cells), "Painted regions must not overlap"
        occupied.update(cells)
        for p in cells:
            if int(a[p]) == bg:
                out[p] = color
    return out


def _decode(input_grid):
    info = _parse(input_grid)
    routes = _routes(info["hubs"], info["key"])
    assert len(routes) == 1, "The key must select one distinct-hub itinerary"
    info["route"] = routes[0]
    expected = _paint(info["grid"], info["background"], info["hubs"], routes[0])
    return expected, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input pixels alone, without examples or generator state."""
    return _decode(input_grid)[0]


def _random_path(rng, hubs, length, both_axes):
    graph, bands = _graph(hubs)
    budget = 1500

    def visit(path, painted):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            axes = {hubs[i][0] == hubs[j][0] for i, j in zip(path, path[1:])}
            return path if not both_axes or len(axes) == 2 else None
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            if j in path or painted & bands[path[-1], j]:
                continue
            answer = visit(path + [j], painted | bands[path[-1], j])
            if answer is not None:
                return answer
        return None

    for i in rng.permutation(len(hubs)):
        answer = visit([int(i)], frozenset())
        if answer is not None:
            return answer
    return None


def _coordinates(rng, n, upper):
    available = [c for c in combinations(range(3, upper - 3), n)
                 if all(b - a >= 7 for a, b in zip(c, c[1:]))]
    if not available:
        return None
    return available[int(rng.integers(len(available)))]


def _place_arrows(rng, hubs, field_shape):
    centers = {(h[0], h[1]) for h in hubs}
    choices = []
    for r, c, label, count in hubs:
        candidates = []
        for dr, dc in DIRECTIONS:
            tip = (r - DISTANCE * dr, c - DISTANCE * dc)
            cells = _arrow(tip, (dr, dc))
            if (all(1 <= y < field_shape[0] - 1 and 1 <= x < field_shape[1] - 1
                    for y, x in cells) and not (cells & centers)):
                candidates.append((tip, (dr, dc), label, cells))
        groups = list(combinations(candidates, count))
        groups = [group for group in groups
                  if len({p for obj in group for p in obj[3]}) == 3 * count
                  and all(not any((r + dy, c + dx) in b[3]
                                  for r, c in a[3] for dy in (-1, 0, 1)
                                  for dx in (-1, 0, 1))
                          for a, b in combinations(group, 2))]
        if not groups:
            return None
        choices.append([groups[int(i)] for i in rng.permutation(len(groups))])
    order = sorted(range(len(hubs)), key=lambda i: len(choices[i]))
    budget = 5000

    def visit(position, selected, forbidden):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if position == len(order):
            return selected
        for group in choices[order[position]]:
            cells = {p for obj in group for p in obj[3]}
            if cells & forbidden:
                continue
            expanded = {(r + dy, c + dx) for r, c in cells
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
            answer = visit(position + 1, selected + list(group), forbidden | expanded)
            if answer is not None:
                return answer
        return None

    return visit(0, [], set())


def _sample_pair(rng, kind, stable_background):
    length = 3 if kind == "intro" else 4 if kind == "contrast" else (
        int(rng.integers(6, 8)) if kind == "test" else int(rng.integers(3, 6)))
    unused = 0 if kind == "intro" else (
        int(rng.integers(1, 3)) if kind == "contrast" else int(rng.integers(0, 3)))
    n = length + unused
    for attempt in range(500):
        height = int(rng.integers(24 if kind == "test" else 18, 31))
        width = int(rng.integers(23 if kind == "test" else 17, 31))
        field_height = height - int(rng.integers(3, 6))
        shapes = [(nr, nc) for nr in range(1, 4) for nc in range(1, 5)
                  if n <= nr * nc <= n + 4 and (nr > 1 or kind != "contrast")]
        nr, nc = shapes[int(rng.integers(len(shapes)))]
        ys, xs = _coordinates(rng, nr, field_height), _coordinates(rng, nc, width)
        if ys is None or xs is None:
            continue
        positions = [(y, x) for y in ys for x in xs]
        positions = sorted(positions[int(i)] for i in rng.choice(len(positions), n, replace=False))
        counts = [int(x) for x in rng.choice((1, 1, 2, 2, 3), n)]
        # Required count-to-size contrasts occur on the actual selected route.
        hubs = [(r, c, 1, count) for (r, c), count in zip(positions, counts)]
        path = _random_path(rng, hubs, length, kind in ("contrast", "test"))
        if path is None:
            continue
        required = (1, 2) if kind == "intro" else (1, 2, 3) if kind == "contrast" else ()
        if not set(required) <= {counts[i] for i in path[:-1]}:
            continue
        bg = int(rng.integers(10)) if stable_background is None else stable_background
        palette = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
        palette = palette[:int(rng.integers(2, min(7, n) + 1))]
        assigned = None
        for _ in range(80):
            labels = [int(rng.choice(palette)) for i in hubs]
            key = [labels[i] for i in path]
            if len(set(key)) < 2 or not any(a != b for a, b in zip(key, key[1:])):
                continue
            if kind == "contrast":
                distractors = [i for i in range(n) if i not in path]
                labels[distractors[0]] = int(rng.choice(key))
            current = [(r, c, labels[i], count) for i, (r, c, _, count) in enumerate(hubs)]
            routes = _routes(current, key)
            if len(routes) == 1 and routes[0] == tuple(path):
                assigned = current, key
                break
        if assigned is None:
            continue
        hubs, key = assigned
        arrows = _place_arrows(rng, hubs, (field_height, width))
        if arrows is None:
            continue
        if kind in ("intro", "contrast"):
            types = {int(bool(dr and dc)) for _, (dr, dc), _, _ in arrows}
            if len(types) != 2:
                continue
        if kind == "contrast":
            stroke_colors = {p: hubs[i][2] for i in path for p in _square(hubs[i])}
            stroke_colors.update({p: hubs[i][2] for i, j in zip(path, path[1:])
                                  for p in _band(hubs[i], hubs[j])})
            if not any(p in stroke_colors and label != stroke_colors[p]
                       for _, _, label, cells in arrows for p in cells):
                continue
        a = np.full((height, width), bg, dtype=np.int8)
        for _, _, label, cells in arrows:
            for p in cells:
                a[p] = label
        key_y = int(rng.integers(field_height + 1, height - 1))
        key_x = int(rng.integers(1, width - 2 * len(key) + 1))
        for i, color in enumerate(key):
            a[key_y, key_x + 2 * i] = color
        # Construct labels geometrically, then re-read the result during validation.
        out = _paint(a, bg, hubs, path)
        return {"input": a, "output": out}
    raise RuntimeError("Bounded geometric construction exhausted")


def _counts(num_train, num_test):
    for name, count, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(count) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    stable_background = int(rng.integers(10)) if rng.random() < .35 else None
    specs = ["intro", "contrast"] + ["train"] * (nt - 2)
    specs = [specs[int(i)] for i in rng.permutation(nt)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, kinds in (("train", specs), ("test", ["test"] * nq)):
        for kind in kinds:
            for _ in range(20):
                pair = _sample_pair(rng, kind, stable_background)
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            signature = _signature(a)
            assert signature not in seen
            seen.add(signature)
            expected, info = _decode(a)
            assert np.array_equal(expected, out), "Wrong virtual stations or ribbons"
            assert not np.array_equal(a, out)
            foreground = a != info["background"]
            assert np.array_equal(a[foreground], out[foreground]), "Arrowheads and key are stencils"
            assert all(out[h[:2]] == h[2] for i, h in enumerate(info["hubs"])
                       if i in info["route"])
            assert all(out[h[:2]] == info["background"] for i, h in enumerate(info["hubs"])
                       if i not in info["route"])
            features[split].append(info)
    training = features["train"]
    assert {info["hubs"][i][3] for info in training for i in info["route"][:-1]} == {1, 2, 3}, \
        "Teach all outgoing multiplicities"
    assert any(len(info["route"]) < len(info["hubs"]) and
               any(h[2] in info["key"] for i, h in enumerate(info["hubs"])
                   if i not in info["route"]) for info in training), "Teach ordered exclusion"
    assert any(len({info["hubs"][i][0] == info["hubs"][j][0]
                    for i, j in zip(info["route"], info["route"][1:])}) == 2
               for info in training), "Teach both ribbon axes"
    assert all(len(info["key"]) <= 5 for info in training)
    assert all(6 <= len(info["key"]) <= 7 for info in features["test"])
    assert any(any(info["grid"][p] != info["background"] and
                   info["grid"][p] != info["hubs"][i][2]
                   for i, j in zip(info["route"], info["route"][1:])
                   for p in _band(info["hubs"][i], info["hubs"][j]))
               or any(info["grid"][p] != info["background"] and
                      info["grid"][p] != info["hubs"][i][2]
                      for i in info["route"] for p in _square(info["hubs"][i]))
               for info in training), "Teach foreground preservation within strokes"


def _regression_tests():
    # Explicit geometry checks do not rely on random-constructor metadata.
    a = np.zeros((30, 30), dtype=np.int8)
    specs = [((8, 3), (0, 1), 2), ((13, 8), (-1, 0), 2),
             ((3, 21), (1, 0), 3), ((16, 21), (1, 0), 4),
             ((21, 26), (0, -1), 4), ((16, 26), (1, -1), 4),
             ((8, 13), (-1, -1), 3)]
    for tip, direction, color in specs:
        for p in _arrow(tip, direction):
            a[p] = color
    a[28, 2:7:2] = (2, 3, 4)
    out, info = _decode(a)
    assert info["hubs"] == [(3, 8, 3, 1), (8, 8, 2, 2),
                            (8, 21, 3, 1), (21, 21, 4, 3)]
    assert info["route"] == (1, 2, 3)
    assert out[3, 8] == 0                    # Same-label false candidate.
    assert out[7, 10] == out[8, 10] == out[9, 10] == 2
    assert out[6, 10] == 0                   # Exactly source width three.
    assert out[10, 21] == 3 and out[10, 20] == 0  # Source one, target five.
    assert out[20, 20] == out[23, 23] == 4
    assert out[8, 13] == 3 and out[16, 21] == 4   # Two visible stencils.
    reversed_key = a.copy()
    reversed_key[28, 2:7:2] = (4, 3, 2)
    reverse = solve(reversed_key)
    assert reverse[8, 10] == 3 and reverse[7, 10] == 0
    assert reverse[10, 20] == 4
    assert not np.array_equal(out, reverse)
    # Foreground preservation does not depend on a shooter's hub selection.
    assert np.array_equal(out[a != 0], a[a != 0])
    for direction in DIRECTIONS:
        tip = (10, 10)
        comp = _arrow(tip, direction)
        matches = [(t, d) for t in comp for d in DIRECTIONS if _arrow(t, d) == comp]
        assert matches == [(tip, direction)]
        endpoint = tuple(tip[k] + 5 * direction[k] for k in (0, 1))
        assert max(abs(endpoint[k] - tip[k]) for k in (0, 1)) == 5
    # Input-only solving also works on smaller valid scenes outside the sampler.
    compact = np.zeros((17, 18), dtype=np.int8)
    compact_specs = [((7, 2), (0, 1), 2), ((2, 13), (1, 0), 3),
                     ((12, 8), (0, 1), 4)]
    for tip, direction, color in compact_specs:
        for p in _arrow(tip, direction):
            compact[p] = color
    compact[15, 2:7:2] = (2, 3, 4)
    small = solve(compact)
    assert small[7, 7] == 2 and small[7, 10] == 2
    assert small[9, 13] == 3 and small[12, 13] == 4


def self_test():
    """Bounded 42-seed audit plus count, ablation, and preservation regressions."""
    _regression_tests()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, layouts, directions, counts, lengths, palettes = set(), set(), set(), set(), set(), set()
    occlusions = repeats = unused = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                expected, info = _decode(pair["input"])
                assert np.array_equal(expected, pair["output"])
                assert np.array_equal(solve(pair["input"].tolist()), expected)
                hubs, route = info["hubs"], info["route"]
                shapes.add(expected.shape)
                origin = (min(h[0] for h in hubs), min(h[1] for h in hubs))
                layouts.add(tuple((h[0] - origin[0], h[1] - origin[1], h[3]) for h in hubs))
                directions.update(d for _, d, _, _ in info["arrows"])
                counts.add(len(info["arrows"]))
                lengths.add(len(route))
                palettes.add((info["background"], tuple(sorted(set(info["key"])))))
                repeats += len(set(info["key"])) < len(info["key"])
                unused += len(hubs) > len(route)
                strokes = set().union(*[_square(hubs[i]) for i in route],
                                      *[_band(hubs[i], hubs[j]) for i, j in zip(route, route[1:])])
                occlusions += any(pair["input"][p] != info["background"] for p in strokes)
                # Collapse all collisions to singleton hubs: the answer must change
                # whenever a selected collision is present (center is always blank).
                collapsed = [(r, c, color, 1) for r, c, color, n in hubs]
                if any(hubs[i][3] > 1 for i in route):
                    alternate = _paint(pair["input"], info["background"], collapsed, route)
                    assert not np.array_equal(alternate, expected)
                # Ignoring the key's exclusion would materialize an unused center.
                if len(hubs) > len(route):
                    i = next(i for i in range(len(hubs)) if i not in route)
                    assert expected[hubs[i][:2]] == info["background"]
    assert len(shapes) >= 25 and len(layouts) >= 100
    assert directions == set(DIRECTIONS)
    assert len(counts) >= 12 and lengths == {3, 4, 5, 6, 7}
    assert len(palettes) >= 70 and repeats >= 50 and unused >= 50 and occlusions >= 20
    repeated = generate_puzzle(0)
    assert all(np.array_equal(puzzles[0][s][i][k], repeated[s][i][k])
               for s in ("train", "test") for i in range(len(puzzles[0][s])) for k in ("input", "output"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    damaged = {s: [{k: a.copy() for k, a in pair.items()} for pair in puzzles[0][s]]
               for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, input-only solving")
