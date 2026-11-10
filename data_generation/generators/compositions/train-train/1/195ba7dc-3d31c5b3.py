"""Welded inks: union supplies connectivity; a visible key supplies dominance.

Only standard-library modules and NumPy are used. No parent generator is imported.
"""
from collections import Counter, deque
import json
import numpy as np


HOST_SEEDS = (0, 194971236, 421905426, 820228050, 145021329, 699318276,
              272171398, 834793559, 788825074, 964527611, 651233671, 167170604,
              490447699, 396421742, 594458747, 377162950, 864997316, 979028245,
              345847765, 691561746, 777718159, 586195391, 846516435, 966974607,
              18345940, 582974879, 718167698, 518218921, 724384998, 324575912,
              329827840, 480404847, 746391574, 927116087, 661958933, 813509215,
              443861169, 690971085, 383800758, 657675883, 83643763, 777126164)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    """Return orthogonal components, with an eight-neighbor counterfactual option."""
    h, w = mask.shape
    unseen = set(zip(*np.nonzero(mask)))
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue, group = [start], []
        while queue:
            r, c = queue.pop()
            group.append((int(r), int(c)))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if 0 <= p[0] < h and 0 <= p[1] < w and p in unseen:
                    unseen.remove(p)
                    queue.append(p)
        groups.append(group)
    return groups


def _parse(input_grid):
    a = _grid(input_grid)
    if a.shape[0] < 3:
        raise ValueError("missing key, divider, or maps")
    separator = int(a[1, 0])
    if separator == 0 or not np.all(a[1] == separator):
        raise ValueError("row below key must be a complete nonzero divider")
    key = [int(c) for c in a[0] if c != 0]
    if (not 3 <= len(key) <= 5 or len(set(key)) != len(key) or
            separator in key or list(a[0, :len(key)]) != key or
            np.any(a[0, len(key):] != 0)):
        raise ValueError("key must be a left-aligned list of 3..5 distinct inks")
    data = a[2:]
    candidates = []
    if data.shape[1] % 2:
        middle = data.shape[1] // 2
        if middle and np.all(data[:, middle] == separator):
            candidates.append((data[:, :middle], data[:, middle + 1:], "side"))
    if data.shape[0] % 2:
        middle = data.shape[0] // 2
        if middle and np.all(data[middle] == separator):
            candidates.append((data[:middle], data[middle + 1:], "stack"))
    allowed = {0, *key}
    candidates = [(p, q, layout) for p, q, layout in candidates
                  if p.shape == q.shape and set(np.unique(p)) <= allowed
                  and set(np.unique(q)) <= allowed]
    if len(candidates) != 1:
        raise ValueError("maps must be equal, aligned, and uniquely separated")
    first, second, layout = candidates[0]
    return first, second, tuple(key), separator, layout


def _fuse(first, second, key, *, diagonal=False, majority=False, minority=False):
    ranks = {c: i for i, c in enumerate(key)}
    out = np.zeros(first.shape, dtype=np.int8)
    for cells in _components((first != 0) | (second != 0), diagonal):
        colors = [int(panel[r, c]) for r, c in cells
                  for panel in (first, second) if panel[r, c] != 0]
        if majority or minority:
            counts = Counter(colors)
            winner = min(counts, key=lambda c: ((-1 if majority else 1) * counts[c], ranks[c]))
        else:
            winner = min(colors, key=ranks.__getitem__)
        for r, c in cells:
            out[r, c] = winner
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the two maps and the visible left-to-right priority key."""
    first, second, key, _, _ = _parse(input_grid)
    return _fuse(first, second, key)


def _pixel_overlay(first, second, key):
    out = np.zeros(first.shape, dtype=np.int8)
    for color in reversed(key):
        out[(first == color) | (second == color)] = color
    return out


def _sheetwise(first, second, key):
    empty = np.zeros_like(first)
    return _pixel_overlay(_fuse(first, empty, key), _fuse(empty, second, key), key)


def _normalize(groups):
    r0 = min(r for g in groups for r, _ in g)
    c0 = min(c for g in groups for _, c in g)
    return [[(r - r0, c - c0) for r, c in g] for g in groups]


def _transform(groups, rng):
    groups = [[(r, -c) for r, c in g] for g in groups] if rng.random() < .5 else groups
    for _ in range(int(rng.integers(4))):
        groups = [[(c, -r) for r, c in g] for g in groups]
    return _normalize(groups)


def _shape(rng, large=False, tiny=False):
    """Connected silhouettes: bars, bends, trees, frames, blocks, stairs, combs."""
    if tiny:
        return _transform([[(0, c) for c in range(int(rng.integers(1, 4)))]], rng)[0]
    mode = int(rng.integers(7))
    h = int(rng.integers(4, 8) if large else rng.integers(2, 5))
    w = int(rng.integers(5, 11) if large else rng.integers(3, 6))
    if mode == 0:
        cells = [(0, c) for c in range(int(rng.integers(10, 20) if large else rng.integers(5, 9)))]
    elif mode == 1:
        cells = [(r, 0) for r in range(h)] + [(h - 1, c) for c in range(1, w)]
    elif mode == 2:
        h = max(3, h)
        cells = [(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)]
    elif mode == 3:
        target = int(rng.integers(18, 36) if large else rng.integers(6, 12))
        cells_set = {(h // 2, w // 2)}
        while len(cells_set) < min(target, h * w):
            frontier = sorted({(r + dr, c + dc) for r, c in cells_set
                               for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells_set)
            cells_set.add(frontier[int(rng.integers(len(frontier)))])
        cells = sorted(cells_set)
    elif mode == 4:
        cells = [(r, c) for r in range(h if large else 2) for c in range(w)]
    elif mode == 5:
        n = int(rng.integers(5, 9) if large else rng.integers(3, 5))
        cells = [(r, c) for r in range(n) for c in (r, r + 1)]
    else:
        cells = [(0, c) for c in range(w)] + [(r, c) for c in range(0, w, 2) for r in range(1, h)]
    cells = set(cells)
    while len(cells) < 6:
        r, c = max(cells, key=lambda p: (p[1], p[0]))
        cells.add((r, c + 1))
    return _transform([sorted(cells)], rng)[0]


def _place(groups, occupied, rng):
    """Pack a module anywhere, allowing internal diagonal contact only."""
    h, w = occupied.shape
    mh = 1 + max(r for g in groups for r, _ in g)
    mw = 1 + max(c for g in groups for _, c in g)
    if mh > h or mw > w:
        return None
    forbidden = occupied.copy()
    forbidden[1:] |= occupied[:-1]
    forbidden[:-1] |= occupied[1:]
    forbidden[:, 1:] |= occupied[:, :-1]
    forbidden[:, :-1] |= occupied[:, 1:]
    offsets = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    for index in rng.permutation(len(offsets)):
        dr, dc = offsets[int(index)]
        placed = [[(r + dr, c + dc) for r, c in g] for g in groups]
        if not any(forbidden[r, c] for g in placed for r, c in g):
            for g in placed:
                for r, c in g:
                    occupied[r, c] = True
            return placed
    return None


def _distances(cells, start):
    cells = set(cells)
    distances, queue = {start: 0}, deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            p = (r + dr, c + dc)
            if p in cells and p not in distances:
                distances[p] = distances[r, c] + 1
                queue.append(p)
    return distances


def _paint(first, second, cells, colors, rng, rare=None):
    """Split a silhouette into fragments; dominant ink is a small remote donor.

    Some mixed objects isolate a dominant donor on one sheet; the other sheet
    supplies its neighbors. Other objects make the dominant ink abundant, so
    rarity is not a substitute for priority.
    """
    panels = (first, second)
    rare = bool(rng.integers(2)) if rare is None else rare
    base = (colors[1:] if rare else colors[:1]) if len(colors) > 1 else colors
    for r, c in cells:
        state = int(rng.choice((1, 2, 3), p=(.4, .4, .2)))
        for i, panel in enumerate(panels):
            if state & (1 << i):
                panel[r, c] = int(rng.choice(base))
    if len(colors) == 1:
        return
    if not rare:
        sites = [cells[int(i)] for i in rng.permutation(len(cells))[:len(colors) - 1]]
        for color, p in zip(colors[1:], sites):
            q = int(rng.integers(2))
            panels[q][p] = color
            if color == colors[-1]:
                panels[1 - q][p] = 0
        return
    cellset = set(cells)
    degree = {p: sum((p[0] + dr, p[1] + dc) in cellset
                     for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))) for p in cells}
    endpoints = sorted(p for p in cells if degree[p] == min(degree.values()))
    donor = endpoints[int(rng.integers(len(endpoints)))]
    distance = _distances(cells, donor)
    remote = max(sorted(cells), key=distance.__getitem__)
    assert distance[remote] >= 2
    q = int(rng.integers(2))
    panels[q][donor] = colors[0]
    panels[1 - q][donor] = colors[-1]
    for p in cells:
        if distance[p] == 1:
            panels[q][p] = 0
            panels[1 - q][p] = colors[-1]
    panels[q][remote] = colors[-1]
    panels[1 - q][remote] = 0
    # Preserve every requested lower ink at a different non-donor coordinate.
    sites = [p for p in cells if p not in (donor, remote)]
    for color, index in zip(colors[1:-1], rng.permutation(len(sites))):
        p = sites[int(index)]
        active = 1 - q if distance[p] == 1 else (0 if first[p] else 1)
        panels[active][p] = color


def _assemble(first, second, key, separator, layout):
    h, w = first.shape
    if layout == "side":
        data = np.concatenate((first, np.full((h, 1), separator, np.int8), second), axis=1)
    else:
        data = np.concatenate((first, np.full((1, w), separator, np.int8), second), axis=0)
    header = np.zeros((1, data.shape[1]), np.int8)
    header[0, :len(key)] = key
    return np.concatenate((header, np.full_like(header, separator), data), axis=0)


def _make_pair(rng, key, separator, layout, comparisons, duel, test, duel_rare=True):
    # Descriptors are component geometry plus its participating inks; output is
    # built from these descriptors, independently of the input parser/solver.
    for attempt in range(80):
        if layout == "side":
            h = int(rng.integers(18, 29) if test else rng.integers(11, 18))
            w = int(rng.integers(10, 15))
        else:
            h = int(rng.integers(10, 14))
            w = int(rng.integers(23, 31) if test else rng.integers(12, 23))
        modules = []
        if test:
            # New conjunction: at least three ranks inhabit one welded object.
            n = int(rng.integers(3, len(key) + 1))
            selected = sorted(rng.choice(len(key), n, replace=False))
            modules.append(([_shape(rng, large=True)], [tuple(key[int(i)] for i in selected)], None))
            for _ in range(int(rng.integers(0, 5))):
                n = int(rng.integers(1, len(key) + 1))
                indices = sorted(rng.choice(len(key), n, replace=False))
                modules.append(([_shape(rng, large=rng.random() < .45)],
                                [tuple(key[int(i)] for i in indices)], None))
        else:
            for a, b in comparisons:
                modules.append(([_shape(rng)], [(key[a], key[b])], None))
            if duel is not None:
                colors = tuple(c for c in key if c in duel)
                modules.append(([_shape(rng)], [colors], duel_rare))
            for _ in range(int(rng.integers(0, 3))):
                indices = sorted(rng.choice(len(key), int(rng.integers(1, 3)), replace=False))
                modules.append(([_shape(rng)], [tuple(key[int(i)] for i in indices)], None))
        # A low-only object proves that dominance is local to a welded component.
        diagnostic = duel is not None and duel_rare
        if diagnostic or rng.random() < .40:
            modules.append(([_shape(rng, tiny=True)], [(key[-1],)], None))
        # Two different components touch diagonally but must keep distinct inks.
        if diagnostic or rng.random() < .25:
            a, b = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            groups = _transform([[(0, c) for c in range(a)],
                                 [(1, a + c) for c in range(b)]], rng)
            modules.append((groups, [(key[0],), (key[-1],)], None))
        if not modules:
            modules.append(([_shape(rng)], [(key[int(rng.integers(len(key)))],)], None))
        modules.sort(key=lambda m: (1 + max(r for g in m[0] for r, _ in g)) *
                     (1 + max(c for g in m[0] for _, c in g)), reverse=True)
        occupied = np.zeros((h, w), bool)
        first = np.zeros((h, w), np.int8)
        second, output = np.zeros_like(first), np.zeros_like(first)
        success = True
        for groups, color_groups, rare in modules:
            placed = _place(groups, occupied, rng)
            if placed is None:
                success = False
                break
            for cells, colors in zip(placed, color_groups):
                _paint(first, second, cells, colors, rng, rare)
                for r, c in cells:
                    output[r, c] = colors[0]
        if success:
            return {"input": _assemble(first, second, key, separator, layout), "output": output}
    raise RuntimeError("could not pack separated silhouettes")


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    count = int(rng.integers(3, 6))
    palette = [int(c) for c in rng.permutation(np.arange(1, 10))]
    separator, base = palette[0], tuple(palette[1:count + 1])
    mode = str(rng.choice(("side", "stack", "mixed")))
    assignments = [[] for _ in range(num_train)]
    slots = [int(i) for i in rng.permutation(num_train)]
    for i in range(count - 1):
        assignments[slots[i % num_train]].append((i, i + 1))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            if split == "train" and index == 0:
                order = base
            elif split == "train" and index == 1:
                order = base[::-1]
            else:
                order = tuple(int(c) for c in rng.permutation(base))
            layout = ("side" if index % 2 == 0 else "stack") if mode == "mixed" else mode
            for _ in range(30):
                pair = _make_pair(rng, order, separator, layout,
                                  assignments[index] if split == "train" else [],
                                  (base[0], base[-1]) if split == "train" and index < 2 else None,
                                  split == "test", duel_rare=index == 0)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analysis(first, second, key, out):
    ranks = {c: i for i, c in enumerate(key)}
    comparisons, decisions, counts, diameters = set(), [], [], []
    for cells in _components((first != 0) | (second != 0)):
        colors = {int(p[r, c]) for r, c in cells for p in (first, second) if p[r, c]}
        indices = tuple(sorted(ranks[c] for c in colors))
        if len(indices) == 2 and indices[1] == indices[0] + 1:
            comparisons.add(indices)
        decisions.append((frozenset(colors), int(out[cells[0]])))
        counts.append(len(colors))
        diameters.append(max(_distances(cells, cells[0]).values()))
        winner = min(colors, key=ranks.__getitem__)
        _require(all(out[p] == winner for p in cells), "component did not take its dominant ink")
    _require(np.array_equal(out != 0, (first != 0) | (second != 0)), "union silhouette changed")
    return comparisons, decisions, counts, diameters


def validate_generated_puzzle(puzzle):
    """Check actual grids, semantic witnesses, visible conventions, and sampling shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, convention, comparisons, decisions = set(), None, set(), {}
    evidence = dict(pixel=False, sheetwise=False, diagonal=False, majority=False, minority=False,
                    omit_first=False, omit_second=False, global_color=False)
    states, train_color_counts, test_color_counts = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            first, second, key, separator, _ = _parse(inp)
            current = (frozenset(key), separator)
            if convention is None:
                convention = current
            _require(current == convention, "puzzle palette changed")
            _require(out.shape == first.shape, "output size differs from map size")
            _require(np.array_equal(solve(inp), out), "input-only solver disagrees")
            comp, choice, ncolors, _ = _analysis(first, second, key, out)
            if split == "train":
                comparisons.update(comp)
                train_color_counts.extend(ncolors)
                for colors, winner in choice:
                    decisions.setdefault(colors, set()).add(winner)
                states.update(np.unique((first != 0).astype(int) + 2 * (second != 0)))
                zero = np.zeros_like(first)
                alternatives = {
                    "pixel": _pixel_overlay(first, second, key),
                    "sheetwise": _sheetwise(first, second, key),
                    "diagonal": _fuse(first, second, key, diagonal=True),
                    "majority": _fuse(first, second, key, majority=True),
                    "minority": _fuse(first, second, key, minority=True),
                    "omit_first": _fuse(zero, second, key),
                    "omit_second": _fuse(first, zero, key),
                    "global_color": np.where((first != 0) | (second != 0), key[0], 0),
                }
                for name, alternative in alternatives.items():
                    evidence[name] |= not np.array_equal(out, alternative)
            else:
                test_color_counts.extend(ncolors)
                _require(max(ncolors) >= 3, "test lacks multi-ink fusion")
    k = len(convention[0])
    _require(comparisons == {(i, i + 1) for i in range(k - 1)}, "missing rank comparisons")
    _require(states == {0, 1, 2, 3}, "training does not expose OR truth table")
    _require(all(evidence.values()), "training permits a documented shortcut")
    _require(any(len(winners) > 1 for colors, winners in decisions.items() if len(colors) == 2),
             "training does not demonstrate changing key order")
    _require(max(train_color_counts) == 2 and max(test_color_counts) >= 3,
             "missing pairwise-to-multiple-ink shift")
    return None


def _fingerprint(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed sweep, count boundaries, invariance probes, and diversity."""
    def rows(*lines):
        return np.array([[int(c) for c in line] for line in lines], np.int8)

    probes = [
        (rows("2"), rows("4"), rows("2")),
        (rows("20007"), rows("04440"), rows("22222")),
        (rows("200"), rows("477"), rows("222")),
        (rows("20", "07"), rows("00", "00"), rows("20", "07")),
        (rows("24077"), rows("00000"), rows("22077")),
        (rows("22222"), rows("00007"), rows("22222")),
        (rows("222", "202", "222"), rows("000", "000", "000"),
         rows("222", "202", "222")),
        (rows("247"), rows("247"), rows("222")),
        (rows("777", "777"), rows("000", "020"), rows("222", "222")),
        (rows("000", "000"), rows("000", "000"), rows("000", "000")),
    ]
    for first, second, expected in probes:
        for layout in ("side", "stack"):
            if layout == "stack" and first.shape[1] < 3:
                continue
            inp = _assemble(first, second, (2, 4, 7), 9, layout)
            before = inp.copy()
            _require(np.array_equal(solve(inp), expected), "handcrafted rule probe")
            _require(np.array_equal(inp, before), "solver mutated input")
    reversed_key = _assemble(rows("20007"), rows("04440"), (7, 4, 2), 9, "side")
    _require(np.array_equal(solve(reversed_key), rows("77777")), "key reversal ignored")
    shapes, layouts, palettes, keys, sizes, inks, components = set(), set(), set(), set(), set(), set(), set()
    fingerprints, normalized = set(), set()
    holes = edges = overlaps = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                first, second, key, separator, layout = _parse(pair["input"])
                shapes.add(first.shape)
                layouts.add(layout)
                palettes.add((separator, frozenset(key)))
                keys.add(key)
                inks.add(len(key))
                groups = _components((first != 0) | (second != 0))
                components.add(len(groups))
                sizes.update(len(g) for g in groups)
                roles = {0: 0, separator: 1, **{c: i + 2 for i, c in enumerate(key)}}
                normalized.add((pair["input"].shape,
                                tuple(roles[int(c)] for c in pair["input"].flat)))
                support = (first != 0) | (second != 0)
                edges += bool(np.any(support[0]) or np.any(support[-1]) or
                              np.any(support[:, 0]) or np.any(support[:, -1]))
                holes += any(all(r not in (0, support.shape[0] - 1) and
                                 c not in (0, support.shape[1] - 1) for r, c in g)
                             for g in _components(~support))
                overlaps += bool(np.any((first != 0) & (second != 0) & (first != second)))
                # Sheet exchange is a symmetry; key rank relabeling is equivariant.
                swapped = _assemble(second, first, key, separator, layout)
                _require(np.array_equal(solve(swapped), pair["output"]), "sheet order affected fusion")
                relabel = np.arange(10)
                perm = np.roll(np.array(key), 1)
                for old, new in zip(key, perm):
                    relabel[old] = new
                _require(np.array_equal(solve(relabel[pair["input"]]), relabel[pair["output"]]),
                         "literal color affected priority")
    _require(len(fingerprints) == 42 and len(normalized) >= 200, "structural diversity")
    _require(len(shapes) >= 50 and layouts == {"side", "stack"}, "geometry diversity")
    _require(inks == {3, 4, 5} and len(palettes) >= 35 and len(keys) >= 100, "palette/rank diversity")
    _require(len(sizes) >= 20 and len(components) >= 5, "object variation")
    _require(holes > 0 and edges > 0 and overlaps > 0, "holes, boundaries, or overlaps absent")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(42 + 10 * train + test, train, test)
            _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test, "counts ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4., 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "output_shapes": len(shapes),
            "normalized_inputs": len(normalized), "palettes": len(palettes),
            "priority_keys": len(keys), "component_sizes": len(sizes),
            "component_counts": sorted(components), "hole_examples": holes,
            "boundary_examples": edges, "different_ink_overlap_examples": overlaps}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
