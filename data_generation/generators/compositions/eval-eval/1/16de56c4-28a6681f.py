"""Perforated levees: periodic drilling changes the topology of liquid flow.

The input contains the entire program: paired plugs encode holes, not liquid.
No parent generator, file, training example, or seed is consulted by solve().
"""

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _parse(input_grid):
    g = np.asarray(input_grid)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(1 <= n <= 30 for n in g.shape)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an ARC integer grid")
    h, w = g.shape
    if h < 8 or w < 9:
        raise ValueError("missing levee enclosure")
    bg, wall = int(g[0, 0]), int(g[-2, 1])
    f = h - 2
    if (bg == wall or not np.all(g[[0, -1]] == bg)
            or not np.all(g[:, [0, -1]] == bg)
            or not np.all(g[f, 1:-1] == wall)
            or not np.all(g[1:f + 1, [1, w - 2]] == wall)):
        raise ValueError("expected blank margin around a U-shaped enclosure")
    divs = [c for c in range(2, w - 2) if g[1, c] != bg]
    if not divs:
        raise ValueError("no marked levee")
    bounds = [1] + divs + [w - 2]
    if any(b - a < 3 for a, b in zip(bounds, bounds[1:])):
        raise ValueError("chambers need at least two free columns")
    off_divider = np.ones(g.shape, dtype=bool)
    off_divider[:, divs] = False
    hues = np.unique(g[off_divider & (g != bg) & (g != wall)])
    if len(hues) != 1:
        raise ValueError("liquid is the unique free-space foreground hue")
    water = int(hues[0])
    signals = []
    for c in divs:
        line = g[1:f, c]
        if np.any((line == bg) | (line == water)):
            raise ValueError("an input levee must be solid, including its plugs")
        colors, counts = np.unique(line[line != wall], return_counts=True)
        repeated = colors[counts == 2]
        if len(repeated) != 1 or np.any(counts > 2):
            raise ValueError("each levee needs one unambiguous equal plug pair")
        ref = int(repeated[0])
        a, b = map(int, np.flatnonzero(g[1:f, c] == ref) + 1)
        period = b - a
        extras = [r for r in range(1, f) if int(g[r, c]) not in (wall, ref)]
        aligned = [r for r in extras if (r - a) % period == 0]
        if len(aligned) > 1 or any(a < r < b for r in aligned):
            raise ValueError("ambiguous terminal")
        t = aligned[0] if aligned else None
        lo, hi = (1, f - 1) if t is None else (min(a, t), max(b, t))
        holes = tuple(r for r in range(lo, hi + 1) if (r - a) % period == 0)
        signals.append({"col": c, "a": a, "b": b, "period": period,
                        "terminal": t, "holes": holes,
                        "noise": tuple(r for r in extras if r != t)})
    return g.astype(np.int8, copy=True), bg, wall, water, signals, bounds


def _open(g, bg, signals, mode="normal"):
    out = g.copy()
    for s in signals:
        if mode == "visible_only":
            rows = [s["a"], s["b"]] + ([] if s["terminal"] is None else [s["terminal"]])
        elif mode == "ignore_terminal":
            rows = [r for r in range(1, g.shape[0] - 2) if (r - s["a"]) % s["period"] == 0]
        else:
            rows = s["holes"]
        out[list(rows), s["col"]] = bg
    return out


class _LiquidState:
    """Incrementally track connected pools and their dry boundaries.

    A pour only adds water, so union/find avoids reflooding each existing pool
    for every delivered unit. All state belongs to this one input simulation.
    """

    def __init__(self, dry, bg, water):
        self.cells = np.asarray(dry).tolist()
        self.h, self.w = len(self.cells), len(self.cells[0])
        self.bg, self.water = bg, water
        self.parent, self.edges, self.sizes = {}, {}, {}

    def find(self, index):
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while index != root:
            following = self.parent[index]
            self.parent[index] = root
            index = following
        return root

    def neighbors(self, row, col):
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < self.h and 0 <= c < self.w:
                yield r, c

    def put(self, row, col):
        index = row * self.w + col
        assert self.cells[row][col] == self.bg
        self.cells[row][col] = self.water
        self.parent[index] = index
        self.sizes[index] = 1
        self.edges[index] = {r * self.w + c for r, c in self.neighbors(row, col)
                             if self.cells[r][c] == self.bg}
        for r, c in self.neighbors(row, col):
            if self.cells[r][c] != self.water:
                continue
            first, second = self.find(index), self.find(r * self.w + c)
            if first == second:
                continue
            if self.sizes[first] < self.sizes[second]:
                first, second = second, first
            self.parent[second] = first
            self.sizes[first] += self.sizes.pop(second)
            self.edges[first].update(self.edges.pop(second))
        self.edges[self.find(index)].discard(index)

    def frontier(self, row, col, source_col, reverse):
        boundary = self.edges[self.find(row * self.w + col)]
        if not boundary:
            raise ValueError("overfilled sealed pool")
        index = min(boundary, key=lambda i: (-(i // self.w), abs(i % self.w - source_col),
                                             -(i % self.w) if reverse else i % self.w))
        return divmod(index, self.w)


def _drop(state, bg, water, row, col, reverse=False):
    """A unit falls; a supported unit spreads to a downward outlet.

    Nearest/leftmost ordering is only an implementation order. Family members
    must have complete stable levels and agree under all four pouring orders.
    """
    cells = state.cells
    h, w = state.h, state.w
    row, col = int(row), int(col)
    fed = False
    if cells[row][col] == water:
        row, col = state.frontier(row, col, col, reverse)
        fed = True
    elif cells[row][col] != bg:
        raise ValueError("source inside solid terrain")
    for _ in range(h * w):
        if row + 1 < h and cells[row + 1][col] == bg:
            row += 1
            fed = False
            continue
        if row + 1 < h and cells[row + 1][col] == water and not fed:
            row, col = state.frontier(row + 1, col, col, reverse)
            fed = True
            continue
        left = right = col
        while left > 0 and cells[row][left - 1] in (bg, water):
            left -= 1
        while right + 1 < w and cells[row][right + 1] in (bg, water):
            right += 1
        exits = [] if row + 1 == h else [c for c in range(left, right + 1)
                                               if cells[row + 1][c] == bg]
        if not exits:
            state.put(row, col)
            return
        col = min(exits, key=lambda c: (abs(c - col), -c if reverse else c))
        row += 1
        fed = False
    raise ValueError("flow did not terminate")


def _flow(drilled, bg, water, reverse=False, top_first=False):
    points = sorted(map(tuple, np.argwhere(drilled == water)),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    out = _LiquidState(np.where(drilled == water, bg, drilled), bg, water)
    for r, c in points:
        _drop(out, bg, water, r, c, reverse)
    return np.asarray(out.cells, dtype=np.int8)


def _stable(out, bg, water):
    """No liquid has a dry side neighbor or dry cell immediately below it.

    This local closure is equivalent to complete, supported basin levels;
    it excludes all partially filled levels and unresolved horizontal splits.
    """
    wet, dry = out == water, out == bg
    return not (np.any(wet[:-1] & dry[1:])
                or np.any(wet[:, :-1] & dry[:, 1:])
                or np.any(wet[:, 1:] & dry[:, :-1]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    g, bg, _, water, signals, _ = _parse(input_grid)
    return _flow(_open(g, bg, signals), bg, water)


def _frame(rng, widths, height, bg, wall, shaped=False):
    width = sum(widths) + len(widths) - 1 + 4
    f = height - 2
    g = np.full((height, width), bg, dtype=np.int8)
    g[1:f + 1, 1:width - 1] = wall
    bounds, col = [1], 2
    for i, span in enumerate(widths):
        if not shaped:
            profile = np.full(span, f, dtype=int)
        else:
            style = int(rng.integers(5))
            depths = np.arange(span)
            if style == 0:
                profile = np.full(span, f - int(rng.integers(4)))
            elif style == 1:
                profile = f - (depths * int(rng.integers(1, 4)) // max(1, span - 1))
            elif style == 2:
                profile = f - np.minimum(3, np.abs(depths - span // 2))
            elif style == 3:
                profile = f - rng.integers(0, 4, span)
            else:
                profile = f - np.minimum(3, span // 2 - np.abs(depths - span // 2))
            if rng.integers(2):
                profile = profile[::-1]
        for c, floor in zip(range(col, col + span), profile):
            g[1:int(floor), c] = bg
        col += span
        if i < len(widths) - 1:
            bounds.append(col)
            col += 1
    bounds.append(width - 2)
    return g, bounds


def _mark(g, rng, col, a, period, terminal, palette, noise=False):
    f = g.shape[0] - 2
    order = list(map(int, rng.permutation(palette)))
    b = a + period
    g[[a, b], col] = order[0]
    if terminal is not None:
        g[terminal, col] = order[1]
    options = [r for r in range(1, f) if (r - a) % period]
    if noise and options:
        g[int(_pick(rng, options)), col] = order[2]


def _supply(g, rng, bg, water, bounds, allocations):
    inp = g.copy()
    for index, n in enumerate(allocations):
        if not n:
            continue
        left, right = bounds[index:index + 2]
        mask = g == bg
        mask[:2] = False
        mask[-2:] = False
        mask[:, :left + 1] = False
        mask[:, right:] = False
        cells = np.argwhere(mask)
        if n > len(cells):
            return None
        anchor = cells[int(rng.integers(len(cells)))]
        dr = np.abs(cells[:, 0] - anchor[0])
        dc = np.abs(cells[:, 1] - anchor[1])
        jitter = rng.random(len(cells))
        style = int(rng.integers(6))
        scores = (dr + dc + jitter, jitter, cells[:, 0] + jitter,
                  -cells[:, 0] + jitter, 4 * dr + dc + jitter,
                  4 * dc + dr + jitter)[style]
        chosen = cells[np.argsort(scores)[:n]]
        inp[chosen[:, 0], chosen[:, 1]] = water
    return inp


def _confirmed(inp, expected, bg, water):
    if inp is None:
        return False
    if np.argwhere(inp == water)[:, 0].sum() >= np.argwhere(expected == water)[:, 0].sum():
        return False
    g, _, _, _, sig, _ = _parse(inp)
    drilled = _open(g, bg, sig)
    return (_stable(expected, bg, water)
            and all(np.array_equal(_flow(drilled, bg, water, rev, top), expected)
                    for rev in (False, True) for top in (False, True)))


def _witness(rng, roles, bounded):
    bg, wall, water, palette = roles
    if not bounded:
        p = int(rng.integers(2, 6))
        a = p + int(rng.integers(2, p + 1))
        heights = [h for h in range(max(17, 4 * p + 6), 31)
                   if (h - 3 - a) % p != 0]
        height = int(_pick(rng, heights))
        widths = [int(rng.integers(4, 8)), int(rng.integers(2, 6))]
        g, bounds = _frame(rng, widths, height, bg, wall, shaped=True)
        # Even a minimum-count puzzle teaches an irregular supporting floor.
        f = height - 2
        g[1:f, 2:4] = bg
        g[f - 1, 2] = wall
        t = None
    else:
        p = 1
        height = int(rng.integers(17, 30))
        widths = [int(rng.integers(4, 8)), int(rng.integers(2, 5))]
        g, bounds = _frame(rng, widths, height, bg, wall)
        a = int(rng.integers(2, 6))
        t = int(rng.integers(max(a + 3, height - 9), height - 5))
    _mark(g, rng, bounds[1], a, p, t, palette, noise=not bounded)
    # Read just the planted wall geometry to construct analytic target masks.
    f = height - 2
    col = bounds[1]
    target = g.copy()
    lo, hi = (1, f - 1) if t is None else (a, t)
    holes = [r for r in range(lo, hi + 1) if (r - a) % p == 0]
    target[holes, col] = bg
    if not bounded:
        # The known target is the entire connected basin below an open port
        # above all floor ridges. Isolated holes buried in solid stay dry.
        top = max(r for r in holes if r <= f - 4)
        stack, cells = [(top, 2)], {(top, 2)}
        while stack:
            r, c = stack.pop()
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (top <= rr < f and 1 < cc < g.shape[1] - 2
                        and target[rr, cc] == bg and (rr, cc) not in cells):
                    cells.add((rr, cc))
                    stack.append((rr, cc))
        for r, c in cells:
            target[r, c] = water
    else:
        depth = int(rng.integers(1, f - t - 1))
        target[t + 1:f, 2:col] = water
        target[f - depth:f, col + 1:-2] = water
    n = int(np.count_nonzero(target == water))
    for _ in range(20):
        inp = _supply(g, rng, bg, water, bounds, [n, 0])
        if _confirmed(inp, target, bg, water):
            break
    else:
        # High-first supplies always lower gravitational potential here.
        inp = g.copy()
        cells = np.argwhere((g == bg) & (np.indices(g.shape)[1] > 1)
                            & (np.indices(g.shape)[1] < col)
                            & (np.indices(g.shape)[0] >= 2)
                            & (np.indices(g.shape)[0] < f))
        chosen = cells[:n]
        inp[chosen[:, 0], chosen[:, 1]] = water
        assert _confirmed(inp, target, bg, water)
    if rng.integers(2):
        inp, target = inp[:, ::-1].copy(), target[:, ::-1].copy()
    return {"input": inp, "output": target}


def _random_marks(g, rng, bounds, palette, test):
    f = g.shape[0] - 2
    for col in bounds[1:-1]:
        p = int(rng.integers(1, min(6, (f - 2) // 2) + 1))
        bounded = bool(rng.integers(2))
        choices = []
        for a in range(1, f - p):
            phase = list(range(1 + (a - 1) % p, f, p))
            if len(phase) < 3:
                continue
            if not bounded:
                choices.append((a, None))
            else:
                for t in phase:
                    if a <= t <= a + p:
                        continue
                    lo, hi = min(a, t), max(a + p, t)
                    span = [r for r in phase if lo <= r <= hi]
                    if len(span) >= 4 and (not test or max(span) >= f - 7):
                        choices.append((a, t))
        if not choices:
            a, t = 1, None
        else:
            a, t = _pick(rng, choices)
        _mark(g, rng, col, a, p, t, palette, noise=rng.random() < .45)


def _network(rng, roles, test):
    bg, wall, water, palette = roles
    count = int(rng.integers(2, 5) if test else rng.integers(1, 4))
    widths = list(map(int, rng.integers(2, 6, count + 1)))
    source = int(rng.integers(count + 1))
    widths[source] = int(rng.integers(5, 8))
    if sum(widths) + count + 4 > 30:
        return None
    height = int(rng.integers(15, 31))
    g, bounds = _frame(rng, widths, height, bg, wall, shaped=True)
    _random_marks(g, rng, bounds, palette, test)
    probe = g.copy()
    # A temporary source makes parsing possible; no output depends on it.
    probe[2, bounds[source] + 1] = water
    _, _, _, _, sig, _ = _parse(probe)
    dry = _open(g, bg, sig)
    state = _LiquidState(dry, bg, water)
    sources = [source]
    if rng.random() < .35:
        others = [i for i in range(count + 1) if i != source]
        sources.append(int(_pick(rng, others)))
    capacities = [int(np.count_nonzero(g[2:-2, bounds[i] + 1:bounds[i + 1]] == bg))
                  for i in range(count + 1)]
    allocation = [0] * (count + 1)
    candidates = []
    for volume in range(1, min(150, sum(capacities[i] for i in sources)) + 1):
        options = [i for i in sources if allocation[i] < capacities[i]]
        if not options:
            break
        index = int(_pick(rng, options))
        allocation[index] += 1
        c = (bounds[index] + bounds[index + 1]) // 2
        try:
            _drop(state, bg, water, 2, c)
        except ValueError:
            break
        view = np.asarray(state.cells, dtype=np.int8)
        if volume < 3 or not _stable(view, bg, water):
            continue
        wet_chambers = sum(np.any(view[:, l + 1:r] == water)
                           for l, r in zip(bounds, bounds[1:]))
        if test and wet_chambers < 3:
            continue
        candidates.append((view.copy(), allocation.copy()))
    rng.shuffle(candidates)
    for target, allocations in candidates[:12]:
        for _ in range(4):
            inp = _supply(g, rng, bg, water, bounds, allocations)
            if _confirmed(inp, target, bg, water):
                return {"input": inp, "output": target}
    return None


def _joined_fallback(rng, roles, test):
    """A constructive low-port network, used only after bounded sampling."""
    bg, wall, water, palette = roles
    count = int(rng.integers(2, 5) if test else rng.integers(1, 4))
    widths = [int(rng.integers(2, 5)) for _ in range(count + 1)]
    widths[0] = 6
    height = int(rng.integers(22, 31))
    g, bounds = _frame(rng, widths, height, bg, wall)
    f = height - 2
    for c in bounds[1:-1]:
        p = int(rng.integers(2, 6))
        a = f - 1 - 3 * p
        _mark(g, rng, c, a, p, f - 1 if rng.integers(2) else None,
              palette, noise=bool(rng.integers(2)))
    n = sum(widths) + count
    inp = g.copy()
    cells = [(r, c) for r in range(2, f) for c in range(2, bounds[1])]
    for r, c in cells[:n]:
        inp[r, c] = water
    target = g.copy()
    # All phases meet in the last row above the floor.
    _, _, _, _, signals, _ = _parse(inp)
    target = _open(target, bg, signals)
    target[f - 1, 2:-2] = water
    assert _confirmed(inp, target, bg, water)
    if rng.integers(2):
        inp, target = inp[:, ::-1].copy(), target[:, ::-1].copy()
    return {"input": inp, "output": target}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if not _integer(value) or not low <= int(value) <= high:
            raise ValueError(name + " has an illegal count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    roles = (colors[0], colors[1], colors[2], colors[3:])
    puzzle, seen = {"train": [], "test": []}, set()
    witnesses = [False, True]
    rng.shuffle(witnesses)
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            for attempt in range(32):
                if split == "train" and index < 2:
                    pair = _witness(rng, roles, witnesses[index])
                elif attempt < 24:
                    pair = _network(rng, roles, split == "test")
                else:
                    pair = _joined_fallback(rng, roles, split == "test")
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct examples")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair, adversarial=False):
    inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    g, bg, wall, water, signals, bounds = _parse(inp)
    assert out.shape == inp.shape and np.issubdtype(out.dtype, np.integer)
    assert np.all((0 <= out) & (out <= 9))
    assert np.all(out[[0, -1]] == bg) and np.all(out[:, [0, -1]] == bg)
    drilled = _open(g, bg, signals)
    fixed = (drilled != bg) & (drilled != water)
    assert np.array_equal(out[fixed], drilled[fixed]), "a nondrilled wall/plug moved"
    assert np.array_equal(fixed, (out != bg) & (out != water)), "unexpected solid changes"
    assert np.count_nonzero(inp == water) == np.count_nonzero(out == water) > 0
    assert _stable(out, bg, water), "partial water level or leaking pool"
    assert np.array_equal(_flow(drilled, bg, water), out)
    assert np.argwhere(out == water)[:, 0].sum() > np.argwhere(inp == water)[:, 0].sum()
    if adversarial:
        for reverse in (False, True):
            for top in (False, True):
                assert np.array_equal(_flow(drilled, bg, water, reverse, top), out)
    known_only = _flow(_open(g, bg, signals, "visible_only"), bg, water)
    unbounded = _flow(_open(g, bg, signals, "ignore_terminal"), bg, water)
    outside = np.ones(g.shape, dtype=bool)
    outside[:, bounds] = False
    wet_counts = [int(np.count_nonzero(out[:, l + 1:r] == water))
                  for l, r in zip(bounds, bounds[1:])]
    source_counts = [int(np.count_nonzero(inp[:, l + 1:r] == water))
                     for l, r in zip(bounds, bounds[1:])]
    surfaces = [int(np.flatnonzero(np.any(out[:, l + 1:r] == water, axis=1))[0])
                for l, r in zip(bounds, bounds[1:]) if np.any(out[:, l + 1:r] == water)]
    return {"roles": (bg, wall, water), "barriers": len(signals),
            "periods": tuple(s["period"] for s in signals),
            "terminal_count": sum(s["terminal"] is not None for s in signals),
            "directions": tuple(0 if s["terminal"] is None else
                                1 if s["terminal"] > s["b"] else -1 for s in signals),
            "noise": sum(len(s["noise"]) for s in signals),
            "stop": any(s["terminal"] is not None and any(
                r not in s["holes"] for r in range(1, g.shape[0] - 2)
                if (r - s["a"]) % s["period"] == 0) for s in signals),
            "phase_boundary_witness": any(s["terminal"] is None
                and 1 < s["holes"][0] < s["a"]
                and s["b"] < s["holes"][-1] < g.shape[0] - 3 for s in signals),
            "no_A_changes_water": bool(np.any(((known_only == water) != (out == water)) & outside)),
            "terminal_changes_water": bool(np.any(((unbounded == water) != (out == water)) & outside)),
            "wet_chambers": sum(n > 0 for n in wet_counts),
            "source_chambers": sum(n > 0 for n in source_counts),
            "different_levels": len(set(surfaces)) > 1,
            "volume": int(np.count_nonzero(out == water)),
            "lateral": bool(np.any((inp == water).sum(axis=0) != (out == water).sum(axis=0)))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            stats = _inspect(pair)
            roles.add(stats["roles"])
            if split == "train":
                training.append(stats)
            else:
                assert stats["barriers"] >= 2 and stats["wet_chambers"] >= 3
    assert len(roles) == 1, "physical color roles must agree within a puzzle"
    assert any(s["terminal_count"] == 0 for s in training)
    assert any(s["stop"] and s["terminal_changes_water"] for s in training)
    assert any(s["no_A_changes_water"] for s in training)
    assert any(s["phase_boundary_witness"] for s in training)
    assert any(s["different_levels"] for s in training)
    assert any(s["noise"] for s in training), "off-phase preservation must be demonstrated"
    assert any(1 in s["periods"] for s in training)
    assert any(max(s["periods"]) > 1 for s in training)


def self_test():
    """Bounded structural, causal, symmetry, and 42-seed variation checks."""
    # Hand-calculated fixtures. Neither targets nor terrain use the sampler,
    # parser, phase completion, or flow engine to determine their coordinates.
    for bounded in (False, True):
        inp = np.zeros((14, 11), dtype=np.int8)
        inp[1:13, [1, 9]] = 5
        inp[12, 1:10] = 5
        inp[1:12, 5] = 5
        inp[2:6, 2:5] = 1
        if bounded:
            inp[[2, 4], 5] = 2
            inp[8, 5] = 3
            inp[7, 5] = 4
            expected = inp.copy()
            expected[expected == 1] = 0
            expected[[2, 4, 6, 8], 5] = 0
            expected[9:12, 2:5] = 1
            expected[11, 6:9] = 1
        else:
            inp[2:8, 2:5] = 1
            inp[8, 2] = 1
            inp[[3, 6], 5] = 2
            inp[10, 5] = 4
            expected = inp.copy()
            expected[expected == 1] = 0
            expected[[3, 6, 9], 5] = 0
            expected[9:12, 2:5] = 1
            expected[9:12, 6:9] = 1
            expected[9, 5] = 1
        assert np.array_equal(solve(inp), expected)
        _inspect({"input": inp, "output": expected}, adversarial=True)
    records, shapes, layouts, palettes, signatures = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                stats = _inspect(pair, adversarial=True)
                records.append(stats)
                g = pair["input"]
                shapes.add(g.shape)
                palettes.add(stats["roles"])
                bg, wall, water = stats["roles"]
                normalized = np.where(g == bg, 0, np.where(g == wall, 1,
                                      np.where(g == water, 2, 3))).astype(np.int8)
                layouts.add(_key(normalized))
                assert np.array_equal(solve(g), pair["output"])
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 50 and len(palettes) >= 35
    assert {1, 2, 3, 4} <= {s["barriers"] for s in records}
    assert {1, 2, 3, 4, 5, 6} <= {p for s in records for p in s["periods"]}
    assert {-1, 0, 1} <= {d for s in records for d in s["directions"]}
    assert sum(s["no_A_changes_water"] for s in records) >= 42
    assert sum(s["terminal_changes_water"] for s in records) >= 42
    assert sum(s["different_levels"] for s in records) >= 42
    assert sum(s["source_chambers"] > 1 for s in records) >= 10
    assert len({s["volume"] for s in records}) >= 40
    # Color permutation and horizontal reflection preserve all semantics.
    p = generate_puzzle(314159, 2, 1)
    perm = np.array([7, 2, 9, 0, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    for split in p.values():
        for pair in split:
            assert np.array_equal(solve(perm[pair["input"]][:, ::-1]),
                                  perm[pair["output"]][:, ::-1])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(2026 + 10 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first = generate_puzzle(8128, 2, 1)
    second = generate_puzzle(8128, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first
               for a, b in zip(first[s], second[s]) for k in ("input", "output"))


if __name__ == "__main__":
    self_test()
