"""Contact elections: support-dependent admission to exact-shape pluralities.

Only standard-library modules and NumPy are used. No parent data is loaded.
The solver reconstructs everything from the queried input.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json

import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normal(cells):
    cells = tuple(cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r - y), int(c - x)) for r, c in cells))


@lru_cache(maxsize=4096)
def _variants(raw, reflections=True):
    result = set()
    for mirror in range(2 if reflections else 1):
        cur = _normal((r, -c if mirror else c) for r, c in raw)
        for _ in range(4):
            result.add(cur)
            cur = _normal((c, -r) for r, c in cur)
    return tuple(sorted(result))


def _canonical(cells):
    return min(_variants(_normal(cells)))


def _size(raw):
    return max(r for r, c in raw) + 1, max(c for r, c in raw) + 1


def _straight(cells):
    return (len(cells) >= 2 and
            (len({r for r, c in cells}) == 1 or len({c for r, c in cells}) == 1))


def _components(mask):
    seen, result = set(), []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        start = int(r), int(c)
        if start in seen:
            continue
        seen.add(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _NEIGHBORS:
                p = y + dy, x + dx
                if (0 <= p[0] < h and 0 <= p[1] < w and
                        p not in seen and mask[p]):
                    seen.add(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _hole_free(raw):
    h, w = _size(raw)
    mask = np.ones((h + 2, w + 2), dtype=bool)
    for r, c in raw:
        mask[r + 1, c + 1] = False
    return len(_components(mask)) == 1


@lru_cache(maxsize=1)
def _pool():
    # Grow geometric classes, not a color-coded or seed-specific dictionary.
    current, result = {((0, 0),)}, set()
    for area in range(2, 8):
        next_shapes = set()
        for shape in sorted(current):
            frontier = {(r + dy, c + dx) for r, c in shape
                        for dy, dx in _NEIGHBORS} - set(shape)
            for cell in sorted(frontier):
                candidate = _canonical(shape + (cell,))
                if max(_size(candidate)) <= 4:
                    next_shapes.add(candidate)
        current = next_shapes
        if area >= 4:
            result.update(s for s in current if not _straight(s) and _hole_free(s))
    return tuple(sorted(result))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _translate(piece, d):
    color, cells = piece
    return color, tuple((r + d, c) for r, c in cells)


def _settle(pieces, shape, fixed):
    """Greatest simultaneously moving set; no ordering or latent supports."""
    result = list(pieces)
    h, w = shape
    for _ in range(h * (len(pieces) + 1)):
        occupied = [set(cells) for color, cells in result]
        targets = [{(r + 1, c) for r, c in cells} for cells in occupied]
        moving = {i for i, target in enumerate(targets)
                  if not target & fixed and
                  all(0 <= r < h and 0 <= c < w for r, c in target)}
        while moving:
            stationary = set(range(len(result))) - moving
            blocked = {i for i in moving
                       if any(targets[i] & occupied[j] for j in stationary)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_translate(p, 1) if i in moving else p
                  for i, p in enumerate(result)]
    raise ValueError("settlement did not converge")


def _decode(value):
    grid = _grid(value)
    if grid.shape[0] < 5 or not np.all(grid[1] == grid[1, 0]):
        raise ValueError("missing blank separator")
    bg = int(grid[1, 0])
    controls = [int(v) for v in grid[0] if v != bg]
    if not 1 <= len(controls) <= 3 or len(set(controls)) != len(controls):
        raise ValueError("invalid selectors")
    region = grid.copy()
    region[:2] = bg
    rectangles = []
    for color in np.unique(region):
        if color == bg:
            continue
        for cells in _components(region == color):
            y, x = min(r for r, c in cells), min(c for r, c in cells)
            b, e = max(r for r, c in cells), max(c for r, c in cells)
            box = grid[y:b + 1, x:e + 1]
            if (min(box.shape) >= 3 and np.any(box == bg) and
                    set(map(int, np.unique(box))) == {bg, int(color)} and
                    np.all(box[0] == color) and np.all(box[-1] == color) and
                    np.all(box[:, 0] == color) and np.all(box[:, -1] == color) and
                    np.count_nonzero(region == color) == len(cells)):
                rectangles.append((int(color), y, x, b, e, cells))
    if len(rectangles) != 1:
        raise ValueError("tablet must be unique")
    carrier, y, x, b, e, cells = rectangles[0]
    if y < 3 or carrier in controls:
        raise ValueError("invalid tablet position or color")
    cavities = _components(grid[y:b + 1, x:e + 1] == bg)
    classes = {_canonical(c) for c in cavities}
    if not cavities or any(len(c) < 3 or _straight(c) or not _hole_free(_normal(c)) for c in cavities):
        raise ValueError("invalid cavity shapes")
    region[y:b + 1, x:e + 1] = bg
    pieces = []
    for color in np.unique(region):
        if color == bg:
            continue
        for component in _components(region == color):
            if any(r >= y or not x <= c <= e for r, c in component):
                raise ValueError("pieces must be above the tablet and within its width")
            if not _straight(component):
                if (len(component) < 3 or _canonical(component) not in classes or
                        not _hole_free(_normal(component)) or color in controls):
                    raise ValueError("invalid ballot")
            pieces.append((int(color), component))
    if not pieces:
        raise ValueError("empty construction")
    rod_colors = {color for color, c in pieces if _straight(c)}
    ballot_colors = {color for color, c in pieces if not _straight(c)}
    if rod_colors & ballot_colors:
        raise ValueError("rod and ballot color roles must be separate")
    fixed = set(cells)
    if _settle(pieces, grid.shape, fixed) != pieces:
        raise ValueError("input must be gravity-stable")
    survivors = [p for p in pieces if p[0] not in controls]
    settled = _settle(survivors, grid.shape, fixed)
    eligible = [not _straight(c) and any(r == y - 1 and x <= q <= e for r, q in c)
                for color, c in settled]
    return dict(grid=grid, background=bg, carrier=carrier, bbox=(y, x, b, e),
                cavities=cavities, classes=classes, fixed=fixed, controls=tuple(controls),
                pieces=pieces, survivors=survivors, settled=settled, eligible=eligible)


def _signature(cells, mode):
    raw = _normal(cells)
    if mode == "area":
        return len(raw)
    if mode == "bbox":
        return tuple(sorted(_size(raw)))
    if mode == "rotation":
        return min(_variants(raw, False))
    if mode == "translation":
        return raw
    return _canonical(raw)


def _tally(data, mode="correct"):
    counts = defaultdict(Counter)
    y, x, b, e = data["bbox"]
    for before, after, admitted in zip(data["survivors"], data["settled"], data["eligible"]):
        color, cells = after
        if _straight(cells):
            continue
        if mode == "all":
            admitted = True
        elif mode == "initial":
            admitted = any(r == y - 1 for r, c in before[1])
        elif mode == "movers":
            admitted = before[1] != cells
        if admitted:
            counts[_signature(cells, mode)][color] += 1
    winners = {}
    for cells in data["cavities"]:
        s = _signature(cells, mode)
        ballot = counts[s]
        high = max(ballot.values(), default=0)
        winning = [color for color, count in ballot.items() if count == high]
        if len(winning) != 1:
            raise ValueError("missing or tied eligible ballot")
        winners[s] = winning[0]
    output = data["grid"][y:b + 1, x:e + 1].copy()
    for cells in data["cavities"]:
        for r, c in cells:
            output[r, c] = winners[_signature(cells, mode)]
    return output, dict(counts), winners


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct visible pieces, settle them, admit contacts, tally, crop."""
    return _tally(_decode(input_grid))[0]


def _choice(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _tablet(rng, classes, width, bg, carrier, forced=None):
    height = int(rng.integers(max(6, max(max(_size(s)) for s in classes) + 2), 10))
    available = np.ones((height, width), dtype=bool)
    available[[0, -1]] = False
    available[:, [0, -1]] = False
    grid = np.full((height, width), carrier, dtype=np.int8)
    holes = []
    requests = list(classes) + [_choice(rng, classes) for _ in range(int(rng.integers(0, 6)))]
    for i, s in enumerate(requests):
        variants = (forced[s],) if i < len(classes) and forced and s in forced else _variants(s)
        candidates = []
        for raw in variants:
            h, w = _size(raw)
            for y in range(1, height - h):
                for x in range(1, width - w):
                    cells = tuple((y + r, x + c) for r, c in raw)
                    if all(available[p] for p in cells):
                        candidates.append(cells)
        if not candidates:
            if i < len(classes):
                return None
            continue
        cells = _choice(rng, candidates)
        holes.append((s, cells))
        for r, c in cells:
            grid[r, c] = bg
            available[max(0, r - 1):min(height, r + 2),
                      max(0, c - 1):min(width, c + 2)] = False
    if len(_components(grid == carrier)) != 1:
        return None
    return grid, holes


def _raw_for_width(rng, shape, max_width=4):
    possibilities = [s for s in _variants(shape) if _size(s)[1] <= max_width]
    return _choice(rng, possibilities)


def _add(pieces, color, cells):
    cells = tuple(sorted(cells))
    if any(set(cells) & set(other) for c, other in pieces):
        raise ValueError("construction overlap")
    pieces.append((int(color), cells))


def _drop_static(raw, col, pieces, fixed, shape):
    candidate = (0, tuple((2 + r, col + c) for r, c in raw))
    occupied = fixed | {p for color, cells in pieces for p in cells}
    if set(candidate[1]) & occupied:
        raise ValueError("insufficient stack headroom")
    for _ in range(shape[0]):
        target = _translate(candidate, 1)
        if any(r >= shape[0] for r, c in target[1]) or set(target[1]) & occupied:
            return candidate[1]
        candidate = target
    raise ValueError("no stack landing")


def _build_scene(rng, classes, units, palette, controls, shared=(), forced=None,
                 teaching=False):
    bg, carrier, red, blue, keep = palette[:5]
    widths = [_size(u["raw"])[1] for u in units]
    required = sum(widths) + len(units) - 1
    if required > 29:
        return None
    gaps = [1] * (len(units) - 1)
    for _ in range(int(rng.integers(0, min(4, 30 - required) + 1))):
        if gaps:
            gaps[int(rng.integers(len(gaps)))] += 1
    content_width = sum(widths) + sum(gaps)
    slack = int(rng.integers(0, min(4, 30 - content_width) + 1))
    width = max(content_width + slack, max(max(_size(s)) for s in classes) + 2)
    inner_left = int(rng.integers(width - content_width + 1))
    outside = int(rng.integers(0, min(3, 30 - width) + 1))
    left = int(rng.integers(outside + 1))
    canvas_width = width + outside
    tablet = _tablet(rng, classes, width, bg, carrier, forced)
    if tablet is None:
        return None
    box, holes = tablet
    max_top = 30 - box.shape[0]
    top = int(rng.integers(18 if teaching else 13, max_top + 1))
    margin = int(rng.integers(0, min(2, 30 - top - box.shape[0]) + 1))
    shape = top + box.shape[0] + margin, canvas_width
    grid = np.full(shape, bg, dtype=np.int8)
    grid[top:top + box.shape[0], left:left + width] = box
    fixed = {(top + r, left + c) for r, c in np.argwhere(box == carrier)}
    starts, col = [], left + inner_left
    for i, w in enumerate(widths):
        starts.append(col)
        col += w + (gaps[i] if i < len(gaps) else 0)
    pieces, planned = [], []
    shared_rows = {}
    if shared:
        a, b = shared
        height = int(rng.integers(2, 7))
        row = top - height - 1
        leg_col = starts[a] + widths[a]
        _add(pieces, keep, ((r, leg_col) for r in range(row + 1, top)))
        _add(pieces, red, ((row, c) for c in range(starts[a], starts[b] + widths[b])))
        shared_rows = {a: row - 1, b: row - 1}
    for i, unit in enumerate(units):
        raw, kind, x = unit["raw"], unit["kind"], starts[i]
        low = max(r for r, c in raw)
        feet = sorted(c for r, c in raw if r == low)
        foot = x + _choice(rng, feet)
        if i in shared_rows:
            bottom = shared_rows[i]
        elif kind == "floor":
            bottom = top - 1
        elif kind in ("release", "permanent", "dual"):
            height = int(rng.integers(2, 8))
            bottom = top - height - 1
            if kind == "dual":
                if len(feet) < 2:
                    return None
                for c, color in zip((feet[0], feet[-1]), (red, blue)):
                    _add(pieces, color, ((r, x + c) for r in range(top - height, top)))
            else:
                color = red if kind == "release" else keep
                _add(pieces, color, ((r, foot) for r in range(top - height, top)))
        elif kind == "partial":
            short, extra = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            bottom = top - short - extra - 1
            _add(pieces, keep, ((r, foot) for r in range(top - short, top)))
            _add(pieces, red, ((r, foot) for r in range(top - short - extra, top - short)))
        elif kind == "landing":
            height = int(rng.integers(2, 7))
            row = top - height - 1
            _add(pieces, red, ((r, foot) for r in range(row + 1, top)))
            _add(pieces, keep, ((row, c) for c in range(x, x + widths[i])))
            bottom = row - 1
        else:
            raise ValueError("unknown construction kind")
        cells = tuple((bottom - low + r, x + c) for r, c in raw)
        _add(pieces, unit["color"], cells)
        admitted = kind in ("floor", "release", "shared") or (kind == "dual" and blue in controls)
        planned.append((unit["shape"], unit["color"], admitted))
    # Stacked ballots are mechanical pieces, but never direct contacts.
    for i, unit in enumerate(units):
        if "stack" not in unit:
            continue
        raw, color = unit["stack"]
        if _size(raw)[1] > widths[i]:
            return None
        if unit.get("stack_shelf"):
            # A surviving rod moves with the lower ballot and supports the
            # excluded upper ballot: all piece types share the same physics.
            bar = tuple((0, c) for c in range(widths[i]))
            shelf_cells = _drop_static(bar, starts[i], pieces, fixed, shape)
            _add(pieces, keep, shelf_cells)
        cells = _drop_static(raw, starts[i] + int(rng.integers(widths[i] - _size(raw)[1] + 1)),
                             pieces, fixed, shape)
        _add(pieces, color, cells)
        planned.append((_canonical(raw), color, False))
    for color, cells in pieces:
        for cell in cells:
            if cell[0] < 2:
                return None
            grid[cell] = color
    columns = rng.choice(canvas_width, len(controls), replace=False)
    for c, color in zip(columns, controls):
        grid[0, c] = color
    try:
        data = _decode(grid)
        if set(data["pieces"]) != set(pieces):
            return None  # Same-color contacts must not conceal piece identity.
        actual = Counter((_canonical(cells), color, admitted)
                         for (color, cells), admitted in zip(data["settled"], data["eligible"])
                         if not _straight(cells))
        if actual != Counter(planned):
            return None
        ballots = defaultdict(Counter)
        for s, color, admitted in planned:
            if admitted:
                ballots[s][color] += 1
        winners = {}
        for s in classes:
            high = max(ballots[s].values(), default=0)
            best = [c for c, n in ballots[s].items() if n == high]
            if len(best) != 1:
                return None
            winners[s] = best[0]
        output = box.copy()
        for s, cells in holes:
            for cell in cells:
                output[cell] = winners[s]
        if not np.array_equal(_tally(data)[0], output):
            return None
        return {"input": grid, "output": output}
    except ValueError:
        return None


def _teaching(rng):
    pool = _pool()
    candidates = []
    for s in pool:
        if len(s) > 6 or _normal((r, -c) for r, c in s) in _variants(s, False):
            continue
        companions = [t for t in pool if t != s and len(t) == len(s)
                      and sorted(_size(t)) == sorted(_size(s))]
        raws = [raw for raw in _variants(s) if _size(raw)[1] <= 3 and
                sum(r == _size(raw)[0] - 1 for r, c in raw) >= 2]
        if companions and raws:
            candidates.append((s, companions, raws))
    for _ in range(100):
        anchor, companions, raws = _choice(rng, candidates)
        companion = _choice(rng, companions)
        raw = _choice(rng, raws)
        mirror = _normal((r, -c) for r, c in raw)
        companion_raw = _raw_for_width(rng, companion, 3)
        palette = list(map(int, rng.permutation(10)))
        red, blue = palette[2:4]
        win, rival, third, other = palette[5:9]
        def unit(s, color, form, kind):
            return dict(shape=s, color=color, raw=form, kind=kind)
        winners = [unit(anchor, win, mirror, "shared"), unit(anchor, win, mirror, "shared")]
        winners[0]["stack"] = raw, rival
        winners[0]["stack_shelf"] = True
        singles = [unit(anchor, rival, raw, "floor"), unit(anchor, third, mirror, "floor"),
                   unit(anchor, rival, raw, "dual"), unit(anchor, rival, raw, "dual"),
                   unit(companion, other, companion_raw, "floor")]
        # Keep the shared shelf pair together but put that group anywhere.
        blocks = [winners] + [[u] for u in singles]
        rng.shuffle(blocks)
        units = [u for block in blocks for u in block]
        shared = tuple(i for i, u in enumerate(units) if u["kind"] == "shared")
        pair = _build_scene(rng, (anchor, companion), units, palette, (red,), shared,
                            {anchor: raw}, teaching=True)
        if pair is None:
            continue
        second_input = pair["input"].copy()
        free = np.flatnonzero(second_input[0] == palette[0])
        second_input[0, int(_choice(rng, free))] = blue
        second = {"input": second_input, "output": solve(second_input)}
        if _fails(pair, "area") and _fails(pair, "bbox") and _fails(pair, "rotation"):
            return [pair, second]
    raise RuntimeError("bounded teaching construction failed")


def _optional(rng, classes, test=False):
    for _ in range(100):
        palette = list(map(int, rng.permutation(10)))
        red, blue, keep = palette[2:5]
        ballot_colors = palette[5:9]
        controls = [red] + ([blue] if rng.random() < .5 else [])
        if rng.random() < .16:
            controls.append(palette[9])  # A visibly absent construction identity.
        units = []
        for s in classes:
            counts = _choice(rng, ((1,), (2, 1), (2, 1, 1), (3, 2)))
            if len(classes) == 3 or sum(len(u["raw"]) for u in units) > 20:
                counts = (1,) if rng.random() < .65 else (2, 1)
            colors = [int(c) for c in rng.choice(ballot_colors, len(counts), replace=False)]
            for color, n in zip(colors, counts):
                for _ in range(n):
                    units.append(dict(shape=s, color=color, raw=_raw_for_width(rng, s),
                                      kind="release" if rng.random() < .65 else "floor"))
        if len(units) > 6:
            continue
        for _ in range(int(rng.integers(0, 3))):
            s = _choice(rng, classes)
            units.append(dict(shape=s, color=_choice(rng, ballot_colors), raw=_raw_for_width(rng, s),
                              kind=_choice(rng, ("permanent", "partial", "landing"))))
        rng.shuffle(units)
        eligible_indices = [i for i, u in enumerate(units) if u["kind"] in ("release", "floor")]
        shared = ()
        choices = [(i, i + 1) for i in eligible_indices if i + 1 in eligible_indices]
        if choices and rng.random() < (.7 if test else .4):
            shared = _choice(rng, choices)
            for i in shared:
                units[i]["kind"] = "shared"
        for i in rng.permutation(len(units))[:int(rng.integers(0, 3))]:
            u = units[int(i)]
            possible = [raw for s in classes for raw in _variants(s)
                        if _size(raw)[1] <= _size(u["raw"])[1]]
            if possible:
                raw = _choice(rng, possible)
                color = _choice(rng, [c for c in ballot_colors if c != u["color"]])
                u["stack"] = raw, color
        try:
            pair = _build_scene(rng, classes, units, palette, tuple(controls), shared)
        except ValueError:
            pair = None
        if pair is not None:
            return pair
    raise RuntimeError("bounded optional construction failed")


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError("illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    num_train, num_test = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    train = _teaching(rng)
    seen = {_key(p["input"]) for p in train}
    pool = _pool()
    for _ in range(num_train - 2):
        for attempt in range(20):
            k = int(rng.integers(1, 4))
            classes = tuple(pool[int(i)] for i in rng.choice(len(pool), k, replace=False))
            pair = _optional(rng, classes)
            if _key(pair["input"]) not in seen:
                train.append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            raise RuntimeError("duplicate training inputs")
    known = set().union(*(_decode(p["input"])["classes"] for p in train))
    unseen = [s for s in pool if s not in known]
    tests = []
    for _ in range(num_test):
        for attempt in range(20):
            k = int(rng.integers(1, 4))
            novel = _choice(rng, unseen)
            others = [s for s in pool if s != novel]
            classes = (novel,) + tuple(others[int(i)] for i in rng.choice(len(others), k - 1, replace=False))
            pair = _optional(rng, classes, test=True)
            if _key(pair["input"]) not in seen:
                tests.append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            raise RuntimeError("duplicate test inputs")
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _fails(pair, mode, data=None):
    try:
        answer = _tally(_decode(pair["input"]) if data is None else data, mode)[0]
    except ValueError:
        return True
    return not np.array_equal(answer, np.asarray(pair["output"]))


def _redundant_reversal(analyses):
    """Find the actual shared scaffold and two jointly necessary removals."""
    for small in analyses:
        for large in analyses:
            if (len(large["controls"]) != 2 or
                    not set(small["controls"]) < set(large["controls"]) or
                    small["grid"].shape != large["grid"].shape or
                    not np.array_equal(small["grid"][1:], large["grid"][1:]) or
                    small["winners"] == large["winners"]):
                continue
            old = {p: admitted for p, admitted in zip(small["survivors"], small["eligible"])}
            new = {p: admitted for p, admitted in zip(large["survivors"], large["eligible"])}
            gained = [p for p in old if not _straight(p[1]) and not old[p] and new.get(p)]
            if len(gained) < 2:
                continue
            independently_blocked = True
            for command in large["controls"]:
                survivors = [p for p in large["pieces"] if p[0] != command]
                settled = _settle(survivors, large["grid"].shape, large["fixed"])
                positions = dict(zip(survivors, settled))
                independently_blocked &= all(positions[p] == p for p in gained)
            if independently_blocked:
                return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            data = _decode(inp)
            answer, ballots, winners = _tally(data)
            assert np.array_equal(answer, out), "incorrect contact election or crop"
            y, x, b, e = data["bbox"]
            carrier_mask = inp[y:b + 1, x:e + 1] == data["carrier"]
            assert np.all(out[carrier_mask] == data["carrier"])
            assert set(ballots) == data["classes"]
            assert all(n > 0 for counts in ballots.values() for n in counts.values())
            # Rigidity, unchanged columns, no upward movement, no overlap, rest.
            occupied = set()
            for before, after in zip(data["survivors"], data["settled"]):
                assert before[0] == after[0] and _normal(before[1]) == _normal(after[1])
                assert [c for r, c in before[1]] == [c for r, c in after[1]]
                assert after[1][0][0] >= before[1][0][0]
                assert not occupied & set(after[1])
                occupied.update(after[1])
            assert _settle(data["settled"], inp.shape, data["fixed"]) == data["settled"]
            data["ballots"], data["winners"] = ballots, winners
            analyses[split].append(data)
    train = puzzle["train"]
    for mode in ("all", "initial", "movers", "area", "bbox", "rotation", "translation"):
        assert any(_fails(p, mode, d) for p, d in zip(train, analyses["train"])), "missing teaching witness: " + mode
    # Require real changed answers, not merely undefined alternative tallies.
    for mode in ("all", "area", "bbox"):
        valid_ablation = False
        for p, d in zip(train, analyses["train"]):
            try:
                valid_ablation |= not np.array_equal(_tally(d, mode)[0], p["output"])
            except ValueError:
                pass
        assert valid_ablation, "parent ablation must change a valid answer: " + mode
    reflected_witness = False
    for d in analyses["train"]:
        for cavity in d["cavities"]:
            rotations = _variants(_normal(cavity), False)
            votes = Counter(color for (color, cells), admitted in zip(d["settled"], d["eligible"])
                            if admitted and _normal(cells) in rotations)
            high = max(votes.values(), default=0)
            best = [c for c, n in votes.items() if n == high]
            if len(best) == 1 and best[0] != d["winners"][_canonical(cavity)]:
                reflected_witness = True
    assert reflected_witness, "reflection must change an actual cavity color"
    assert any(max(c.values()) * 2 <= sum(c.values()) and len(c) >= 3
               for d in analyses["train"] for c in d["ballots"].values())
    history = defaultdict(set)
    for d in analyses["train"]:
        for s, color in d["winners"].items():
            history[s].add(color)
    assert any(len(colors) > 1 for colors in history.values()), "missing winner reversal"
    assert _redundant_reversal(analyses["train"]), "missing jointly released counterfactual election"
    known = set(history)
    assert all(d["classes"] - known for d in analyses["test"]), "test needs a locally supported novel shape"
    assert any(len(d["controls"]) >= 2 for d in analyses["train"])
    assert any(any(a != b and not _straight(a[1]) and not admitted for a, b, admitted in
                   zip(d["survivors"], d["settled"], d["eligible"]))
               for d in analyses["train"]), "must demonstrate moving exclusion"
    assert any(any(a != b and _straight(a[1]) for a, b in zip(d["survivors"], d["settled"]))
               for d in analyses["train"]), "must demonstrate surviving rod motion"


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _features(pair):
    d = _decode(pair["input"])
    _, counts, _ = _tally(d)
    distances = [b[1][0][0] - a[1][0][0] for a, b in zip(d["survivors"], d["settled"])]
    voters = [(i, p) for i, p in enumerate(d["settled"]) if not _straight(p[1])]
    return dict(shape=tuple(d["grid"].shape), crop=tuple(pair["output"].shape),
                classes=len(d["classes"]), vocabulary=d["classes"], pieces=len(d["pieces"]),
                cavities=len(d["cavities"]), admitted=sum(d["eligible"]),
                excluded=sum(not d["eligible"][i] for i, p in voters),
                moving_excluded=sum(distances[i] > 0 and not d["eligible"][i] for i, p in voters),
                distances=tuple(sorted(set(distances))), commands=len(d["controls"]),
                ballots=tuple(sorted(tuple(sorted(c.values(), reverse=True)) for c in counts.values())),
                background=d["background"], carrier=d["carrier"],
                rod_lengths=tuple(len(c) for color, c in d["pieces"] if _straight(c)))


def _mechanics_test():
    fixed = {(9, c) for c in range(8)}
    # Removing only one equal-height post keeps this rigid L ballot suspended.
    left = (1, ((6, 2), (7, 2), (8, 2)))
    right = (2, ((6, 3), (7, 3), (8, 3)))
    lower = (3, ((4, 2), (5, 2), (5, 3)))
    upper = (4, ((2, 2), (3, 2), (3, 3)))
    assert _settle([left, right, lower, upper], (10, 8), fixed) == [left, right, lower, upper]
    assert _settle([right, lower, upper], (10, 8), fixed) == [right, lower, upper]
    result = _settle([lower, upper], (10, 8), fixed)
    assert result == [_translate(lower, 3), _translate(upper, 3)]
    assert any(r == 8 for r, c in result[0][1]) and not any(r == 8 for r, c in result[1][1])


def self_test():
    """Bounded 42-seed diversity and family-specific checks; returns measurements."""
    _mechanics_test()
    records, fingerprints = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        records.extend(_features(pair) for split in ("train", "test") for pair in p[split])
    assert len(fingerprints) == 42
    shapes, crops = {f["shape"] for f in records}, {f["crop"] for f in records}
    vocabulary = set().union(*(f["vocabulary"] for f in records))
    assert len(shapes) >= 30 and len(crops) >= 20 and len(vocabulary) >= 40
    assert {f["classes"] for f in records} == {1, 2, 3}
    assert len({f["pieces"] for f in records}) >= 10
    assert len({f["admitted"] for f in records}) >= 5
    assert len({f["cavities"] for f in records}) >= 5
    assert len({f["background"] for f in records}) == 10
    assert len({f["carrier"] for f in records}) == 10
    assert {f["commands"] for f in records} >= {1, 2, 3}
    assert sum(f["moving_excluded"] > 0 for f in records) >= 42
    assert sum(len(set(f["distances"]) - {0}) >= 2 for f in records) >= 20
    assert len({f["ballots"] for f in records}) >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9173, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    assert _fingerprint(generate_puzzle(12345)) == _fingerprint(generate_puzzle(12345))
    assert _fingerprint(generate_puzzle(12345)) != _fingerprint(generate_puzzle(12346))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    p = generate_puzzle(0)
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted corrupt carrier")
    return dict(seeds=42, pairs=len(records), input_shapes=len(shapes), crop_shapes=len(crops),
                shape_classes=len(vocabulary), piece_counts=sorted({f["pieces"] for f in records}),
                admitted_counts=sorted({f["admitted"] for f in records}),
                cavity_counts=sorted({f["cavities"] for f in records}),
                moving_exclusion_pairs=sum(f["moving_excluded"] > 0 for f in records),
                unequal_fall_pairs=sum(len(set(f["distances"]) - {0}) >= 2 for f in records),
                ballot_patterns=len({f["ballots"] for f in records}))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
