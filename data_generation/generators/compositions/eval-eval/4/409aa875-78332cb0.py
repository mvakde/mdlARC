"""Projected-port docking: arrows determine a chain and open its seams.

Only NumPy and the standard library are used.  The solver reads one input;
generation builds labels from the intended links, then validates from pixels.
"""

from collections import Counter
from itertools import combinations, permutations

import numpy as np


DISTANCE = 5
VECTORS = ((-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return (a.shape, a.tobytes())


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _arrow(tip, vector):
    r, c = tip
    dr, dc = vector
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return (tuple(tip), tuple(vector), tuple(sorted((r+a, c+b) for a, b in offsets)))


def _endpoint(obj, distance=DISTANCE):
    return tuple(a + distance*b for a, b in zip(obj[0], obj[1]))


def _separate(obj, occupied):
    return all((r+a, c+b) not in occupied for r, c in obj[2]
               for a in (-1, 0, 1) for b in (-1, 0, 1))


def _read_arrows(panel, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(panel != bg)}
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            new = {(r+a, c+b) for a in (-1, 0, 1) for b in (-1, 0, 1)} & remaining
            remaining -= new
            part |= new
            todo.extend(sorted(new))
        if len(part) != 3 or len({int(panel[p]) for p in part}) != 1:
            raise ValueError("each isolated component must be a monochrome three-cell arrow")
        matches = [_arrow(t, v) for t in sorted(part) for v in VECTORS
                   if set(_arrow(t, v)[2]) == part]
        if len(matches) != 1:
            raise ValueError("ambiguous arrow geometry")
        objects.append(matches[0])
    if not objects:
        raise ValueError("every panel must contain arrows")
    return objects


def _partition(grid):
    """Detect full divider lines; no catalog of sizes or layouts is consulted."""
    candidates = []
    h, w = grid.shape
    for color in np.unique(grid):
        rows = list(map(int, np.flatnonzero(np.all(grid == color, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(grid == color, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        hs = [b-a-1 for a, b in zip(rr, rr[1:])]
        ws = [b-a-1 for a, b in zip(cc, cc[1:])]
        if min(hs+ws) < 3 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        panels = [grid[r+1:rr[i+1], c+1:cc[j+1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 4 or any(np.any(p == color) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        bg = next(iter(corners))
        try:
            arrows = [_read_arrows(p, bg) for p in panels]
        except ValueError:
            continue
        candidates.append(dict(panels=panels, arrows=arrows, bg=bg,
                               divider=int(color), layout=(len(hs), len(ws))))
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous equal-panel partition")
    return candidates[0]


def _orders(signatures):
    """Find up to two complete chains, including constrained repeated ports."""
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(order, left):
        if len(answers) == 2:
            return
        if not left:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        port = signatures[order[-1]][1]
        if not port:
            return
        for i in left:
            if signatures[i][0] == port:
                visit(order + [i], tuple(j for j in left if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _decode(grid):
    parsed = _partition(grid)
    h, w = parsed["panels"][0].shape
    possibilities = []
    for axis in (1, 0):
        signatures = []
        valid = True
        for panel, objects in zip(parsed["panels"], parsed["arrows"]):
            oriented = panel if axis == 1 else panel.T
            # Keep the non-chain edges clear, including every corner.
            if np.any(oriented[0] != parsed["bg"]) or np.any(oriented[-1] != parsed["bg"]):
                valid = False
                break
            sides = [set(), set()]
            for obj in objects:
                r, c = _endpoint(obj)
                t, x, span, length = (r, c, h, w) if axis == 1 else (c, r, w, h)
                if not 0 <= t < span or x not in (-1, length):
                    valid = False
                    break
                sides[int(x == length)].add(t)
            if not valid:
                break
            signatures.append(tuple(tuple(sorted(s)) for s in sides))
        if valid:
            orders = _orders(signatures)
            if len(orders) > 1:
                raise ValueError("projected ports admit multiple complete orders")
            if orders:
                possibilities.append((axis, signatures, orders[0]))
    if len(possibilities) != 1:
        raise ValueError("projected ports must specify one axis and one complete chain")
    axis, signatures, order = possibilities[0]
    if len(order)*(parsed["panels"][0].shape[axis]+1)-1 > 30:
        raise ValueError("assembled chain exceeds 30 cells")
    parsed.update(axis=axis, signatures=signatures, order=order)
    return parsed


def _layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    grid = np.full((nr*(h+1)-1, nc*(w+1)-1), divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        grid[r*(h+1):r*(h+1)+h, c*(w+1):c*(w+1)+w] = panels[index]
    return grid


def _assemble(parsed, order=None):
    order = parsed["order"] if order is None else order
    shape = (1, len(order)) if parsed["axis"] == 1 else (len(order), 1)
    return _layout(parsed["panels"], order, shape, parsed["divider"])


def _counts(parsed, order=None, distance=DISTANCE):
    order = parsed["order"] if order is None else order
    axis = parsed["axis"]
    step = parsed["panels"][0].shape[axis]+1
    counts = Counter()
    for slot, index in enumerate(order):
        for obj in parsed["arrows"][index]:
            point = list(_endpoint(obj, distance))
            point[axis] += slot*step
            counts[tuple(point)] += 1
    return counts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode virtual ports, dock panels, and erase simultaneous seam impacts."""
    parsed = _decode(_grid(input_grid))
    out = _assemble(parsed)
    for point, count in _counts(parsed).items():
        if (not all(0 <= p < n for p, n in zip(point, out.shape))
                or out[point] != parsed["divider"] or count < 2):
            raise ValueError("every shot must meet an opposing shot on a separator")
        out[point] = parsed["bg"]
    return out


def _options(h, w, side, row):
    dc = -1 if side == 0 else 1
    end_c = -1 if side == 0 else w
    result = []
    for dr in (0, -1, 1):
        obj = _arrow((row-DISTANCE*dr, end_c-DISTANCE*dc), (dr, dc))
        if all(1 <= r < h-1 and 0 <= c < w for r, c in obj[2]):
            result.append(obj)
    return result


def _pack(h, w, incoming, outgoing, rng=None, need_diagonal=False):
    requirements = [(side, row) for side, sig in enumerate((incoming, outgoing)) for row in sig]
    choices = [_options(h, w, side, row) for side, row in requirements]
    if any(not items for items in choices):
        return None
    if rng is not None:
        for items in choices:
            rng.shuffle(items)
    # Fewest choices first improves the bounded (at most 3^6) search.
    choices.sort(key=len)

    def visit(i, selected, occupied):
        if i == len(choices):
            if need_diagonal and not any(o[1][0] for o in selected):
                return None
            return selected
        for obj in choices[i]:
            if _separate(obj, occupied):
                answer = visit(i+1, selected+[obj], occupied | set(obj[2]))
                if answer is not None:
                    return answer
        return None

    return visit(0, [], set())


def _pool(h, sizes):
    # One-side feasibility is independent of tile width. Width six separates
    # the two firing banks, so it also gives a guaranteed construction fallback.
    return [s for size in sizes for s in combinations(range(h), size)
            if _pack(h, 6, (), s) is not None]


def _links_and_objects(rng, n, h, w, witness, multiplicity_witness=False):
    sizes = (1,) if multiplicity_witness else (2,) if witness else _choose(rng, ((1,), (1, 2), (2,), (1, 2, 3)))
    pool = _pool(h, sizes)
    if multiplicity_witness:
        pool = [s for s in pool if len(_options(h, w, 1, s[0])) >= 2
                and any(o[1][0] == 0 for o in _options(h, w, 1, s[0]))]
    if len(pool) < n-1:
        return None
    for _ in range(60):
        indices = rng.choice(len(pool), n-1, replace=False)
        links = [pool[int(i)] for i in indices]
        # At most one self-loop at a signature; complete-order search is still
        # mandatory. The teaching witness keeps distinct equal-size signatures.
        if not witness and n >= 3 and rng.random() < .30:
            j = int(rng.integers(n-2))
            links[j+1] = links[j]
        sigs = list(zip([()]+links, links+[()]))
        if len(_orders(sigs)) != 1:
            continue
        objects = [_pack(h, w, a, b, rng) for a, b in sigs]
        if any(items is None for items in objects):
            continue
        if witness and not any(o[1][0] for items in objects for o in items):
            for i, (a, b) in enumerate(sigs):
                alternative = _pack(h, w, a, b, rng, need_diagonal=True)
                if alternative is not None:
                    objects[i] = alternative
                    break
            else:
                continue
        if multiplicity_witness:
            # One singleton handshake with two shooters on one side and one on
            # the other explicitly separates sets from endpoint multiplicities.
            i = int(rng.integers(2))
            side = 1-i
            objects[i] = [next(o for o in _options(h, w, side, links[0][0]) if o[1][0] == 0)]
            occupied = {p for o in objects[i] for p in o[2]}
            extra = [o for o in _options(h, w, side, links[0][0]) if _separate(o, occupied)]
            if not extra:
                continue
            objects[i].append(_choose(rng, extra))
            return links, objects
        # Add extra shooters to already occupied virtual ports: equality is of
        # occupied positions, never of colors or endpoint multiplicities.
        for i, (a, b) in enumerate(sigs):
            occupied = {p for o in objects[i] for p in o[2]}
            possibilities = [o for side, sig in enumerate((a, b)) for row in sig
                             for o in _options(h, w, side, row)]
            rng.shuffle(possibilities)
            for o in possibilities:
                if rng.random() < .24 and _separate(o, occupied):
                    objects[i].append(o)
                    occupied.update(o[2])
        return links, objects
    return None


def _make_pair(rng, axis, span_shift=False, witness=False, monochrome=False, multiplicity_witness=False):
    n = 4 if witness else 2 if multiplicity_witness else int(rng.integers(2, 5))
    if witness:
        h = int(rng.integers(12, 15))
    elif multiplicity_witness:
        h = int(rng.integers(10, 17))
    elif span_shift:
        h = int(rng.integers(17, 25))
    else:
        h = int(rng.integers(8, 17))
    widths = list(range(6, min(14, 31//n-1)+1))
    w = 6 if witness else _choose(rng, widths)
    made = _links_and_objects(rng, n, h, w, witness, multiplicity_witness)
    if made is None:
        # Geometric packing can be impossible for overlapping banks. Changing
        # the sampled width is a construction fallback, never a solving rule.
        w = 6
        made = _links_and_objects(rng, n, h, w, witness, multiplicity_witness)
    if made is None:
        raise RuntimeError("bounded geometric construction failed")
    links, objects = made
    palette = list(map(int, rng.permutation(10)))
    bg, divider = palette[:2]
    arrow_colors = palette[2:]
    panels = []
    shared = _choose(rng, arrow_colors)
    for items in objects:
        p = np.full((h, w), bg, dtype=np.int8)
        local_color = _choose(rng, arrow_colors)
        color_mode = int(rng.integers(2))
        for obj in items:
            color = shared if monochrome else local_color if color_mode else _choose(rng, arrow_colors)
            for point in obj[2]:
                p[point] = color
        panels.append(p)
    # Label construction uses the planned join sets, not solve or decoded order.
    output = _layout(panels, list(range(n)), (1, n), divider)
    for j, sig in enumerate(links, 1):
        output[list(sig), j*(w+1)-1] = bg
    if axis == 0:
        panels = [p.T.copy() for p in panels]
        output = output.T.copy()
    ph, pw = panels[0].shape
    layouts = [(nr, n//nr) for nr in range(1, n+1) if n % nr == 0
               and nr*(ph+1)-1 <= 30 and (n//nr)*(pw+1)-1 <= 30]
    layout = _choose(rng, layouts)
    order = list(range(n))
    kind = int(rng.integers(5))
    if kind == 0:
        order.reverse()
    elif kind == 1:
        order[0], order[-1] = order[-1], order[0]
    elif kind == 2:
        a, b = map(int, rng.choice(n, 2, replace=False))
        order[a], order[b] = order[b], order[a]
    elif kind == 3:
        rng.shuffle(order)
    # Identity relayouts occur, but essential evidence always changes order.
    if witness and order == list(range(n)):
        order[0], order[-1] = order[-1], order[0]
    return {"input": _layout(panels, order, layout, divider), "output": output}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fresh_palette(pair, seen):
    """A finite distinct-input fallback; palettes never encode the rule."""
    colors = list(map(int, np.unique(pair["input"])))
    for assignment in permutations(range(10), len(colors)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = assignment
        candidate = {role: mapping[grid] for role, grid in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("palette space exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    base_axis = int(rng.integers(2))
    shift = int(rng.integers(3))  # new axis, larger transverse span, or both
    plans = [(base_axis, False, True, False, False), (base_axis, False, False, True, True)]
    plans += [(base_axis if shift != 1 else int(rng.integers(2)), False, False,
               bool(rng.integers(2)), False) for _ in range(num_train-2)]
    rng.shuffle(plans)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, specs in (("train", plans), ("test", [
            (1-base_axis if shift != 1 else int(rng.integers(2)), shift != 0,
             False, bool(rng.integers(2)), False) for _ in range(num_test)])):
        for axis, large, witness, mono, multiplicity_witness in specs:
            for _ in range(20):
                pair = _make_pair(rng, axis, large, witness, mono, multiplicity_witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                # At most thirteen previous inputs can exclude assignments;
                # with at least three colors there are at least 720 available.
                pair = _fresh_palette(pair, seen)
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _reading_order_shortcut(parsed):
    order = tuple(range(len(parsed["panels"])))
    out = _assemble(parsed, order)
    for point, count in _counts(parsed, order).items():
        if (count >= 2 and all(0 <= p < n for p, n in zip(point, out.shape))
                and out[point] == parsed["divider"]):
            out[point] = parsed["bg"]
    return out


def _features(inp):
    d = _decode(inp)
    ordered_sigs = [d["signatures"][i] for i in d["order"]]
    links = [b for a, b in ordered_sigs[:-1]]
    counts = _counts(d)
    all_objects = [o for items in d["arrows"] for o in items]
    h, w = d["panels"][0].shape
    side_counts = [Counter(_endpoint(o) for o in items) for items in d["arrows"]]
    return dict(n=len(d["panels"]), tile=(h, w), axis=d["axis"], layout=d["layout"],
                bg=d["bg"], divider=d["divider"], arrows=len(all_objects),
                order=d["order"], links=tuple(links), holes=len(counts),
                max_convergence=max(counts.values()),
                directions=tuple(sorted({o[1] for o in all_objects})),
                diagonal=any(a*b != 0 for a, b in (o[1] for o in all_objects)),
                cardinal=any(a*b == 0 for a, b in (o[1] for o in all_objects)),
                sizes=tuple(sorted({len(s) for s in links})),
                repeated=len(set(links)) < len(links),
                unequal_multiplicity=any(side_counts[a][(-1, t) if d["axis"] == 0 else (t, -1)] !=
                                         side_counts[b][(h, t) if d["axis"] == 0 else (t, w)]
                                         for b, a in zip(d["order"], d["order"][1:])
                                         for t in d["signatures"][b][1]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, teaching, test_features = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            d = _decode(inp)
            expected = solve(inp)
            assert np.array_equal(out, expected), "incorrect output"
            assembled = _assemble(d)
            assert out.shape == assembled.shape
            count = _counts(d)
            hole_cells = set(count)
            changed = {tuple(map(int, p)) for p in np.argwhere(out != assembled)}
            assert changed == hole_cells and hole_cells
            assert all(count[p] >= 2 and assembled[p] == d["divider"] and out[p] == d["bg"] for p in changed)
            # Verify each complete panel matrix is translated intact, regardless
            # of missing separator pixels in the output.
            h, w = d["panels"][0].shape
            for slot, index in enumerate(d["order"]):
                r, c = (0, slot*(w+1)) if d["axis"] == 1 else (slot*(h+1), 0)
                assert np.array_equal(out[r:r+h, c:c+w], d["panels"][index])
            assert np.array_equal(inp, _layout(d["panels"], range(len(d["panels"])), d["layout"], d["divider"]))
            if split == "train":
                f = _features(inp)
                teaching.append((f, not np.array_equal(out, _reading_order_shortcut(d))))
            else:
                test_features.append(_features(inp))
    assert any(diff for f, diff in teaching), "teach assembly before seam impacts"
    assert any(f["diagonal"] for f, diff in teaching), "teach transverse displacement"
    assert any(f["cardinal"] for f, diff in teaching), "teach cardinal arrow geometry"
    assert any(f["n"] == 2 and f["sizes"] == (1,) and f["unequal_multiplicity"]
               and f["max_convergence"] == 3 for f, diff in teaching), "teach occupied sets, not equal multiplicities"
    assert any(f["n"] == 4 and len(set(f["links"])) == 3 and f["sizes"] == (2,)
               and f["order"] != tuple(range(4)) for f, diff in teaching), "teach exact signatures, not cardinality"
    train_axes = {f["axis"] for f, diff in teaching}
    test_axes = {f["axis"] for f in test_features}
    axis_shift = len(train_axes) == 1 and not train_axes & test_axes
    span_shift = (max(f["tile"][1-f["axis"]] for f, diff in teaching) <= 16
                  and min(f["tile"][1-f["axis"]] for f in test_features) >= 17)
    assert axis_shift or span_shift, "require an actual axis or transverse-span shift"


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed coverage, geometric regressions, and mutation rejection."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[17]) == _puzzle_key(generate_puzzle(17))
    records = [_features(pair["input"]) for p in puzzles for s in ("train", "test") for pair in p[s]]
    assert {r["n"] for r in records} == {2, 3, 4}
    assert {r["axis"] for r in records} == {0, 1}
    assert len({r["tile"] for r in records}) >= 25
    assert len({r["layout"] for r in records}) >= 5
    assert len({r["arrows"] for r in records}) >= 10
    assert len({r["links"] for r in records}) >= 100
    assert {r["bg"] for r in records} == set(range(10))
    assert {r["divider"] for r in records} == set(range(10))
    assert {d for r in records for d in r["directions"]} == set(VECTORS)
    assert {s for r in records for s in r["sizes"]} == {1, 2, 3}
    assert any(r["repeated"] for r in records)
    assert any(r["max_convergence"] >= 3 for r in records)
    assert any(r["unequal_multiplicity"] for r in records)
    # A forced self-loop is legal; two swappable loops are ambiguous.
    a, b = (2,), (6,)
    assert len(_orders([((), a), (a, a), (a, b), (b, ())])) == 1
    assert len(_orders([((), a), (a, a), (a, a), (a, ())])) == 2
    # Tip-relative diagonal projection differs from snapping to the tip's row.
    o = _arrow((8, 1), (-1, 1))
    assert _endpoint(o) == (3, 6)
    # Transverse boundary openings and a signature larger than the sampler's
    # three-position preference remain valid, input-only family members.
    panels = [np.full((14, 7), 0, dtype=np.int8) for _ in range(3)]
    fixtures = [[_arrow((5, 2), (-1, 1))],
                [_arrow((5, 4), (-1, -1)), _arrow((8, 2), (1, 1))],
                [_arrow((8, 4), (1, -1))]]
    for p, items in zip(panels, fixtures):
        for o in items:
            for point in o[2]:
                p[point] = 3
    inp = _layout(panels, [2, 0, 1], (1, 3), 8)
    expected = _layout(panels, [0, 1, 2], (1, 3), 8)
    expected[0, 7] = expected[13, 15] = 0
    assert np.array_equal(solve(inp), expected)
    panels = [np.full((30, 6), 0, dtype=np.int8) for _ in range(2)]
    rows = list(range(3, 28, 4))
    for i, p in enumerate(panels):
        for row in rows:
            o = _arrow((row, 1 if i == 0 else 4), (0, 1 if i == 0 else -1))
            for point in o[2]:
                p[point] = 4+i
    inp = _layout(panels, [1, 0], (1, 2), 7)
    expected = _layout(panels, [0, 1], (1, 2), 7)
    expected[rows, 6] = 0
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(solve(inp.T), expected.T)
    # All orthogonal symmetries preserve the rule, including terminal reversal.
    pair = puzzles[0]["train"][0]
    fresh = _fresh_palette(pair, {_key(pair["input"])})
    assert _key(fresh["input"]) != _key(pair["input"])
    assert np.array_equal(solve(fresh["input"]), fresh["output"])
    for k in range(4):
        for flip in (False, True):
            inp, out = (np.rot90(pair[r], k) for r in ("input", "output"))
            if flip:
                inp, out = np.fliplr(inp), np.fliplr(out)
            assert np.array_equal(solve(inp), out)
    # Omission of A: intact assembly lacks holes. Omission of B: keeping reading
    # order gives a different label on the teaching witness in every puzzle.
    for p in puzzles:
        assert any(not np.array_equal(_reading_order_shortcut(_decode(q["input"])), q["output"])
                   for q in p["train"])
        for q in p["train"] + p["test"]:
            assert not np.array_equal(_assemble(_decode(q["input"])), q["output"])
    broken = {s: [{r: q[r].copy() for r in q} for q in puzzles[0][s]] for s in puzzles[0]}
    d = _decode(broken["test"][0]["input"])
    hole = next(iter(_counts(d)))
    broken["test"][0]["output"][hole] = d["divider"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing seam opening was accepted")
    for counts in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(91, *counts)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(91, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(9, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(records), "tile_shapes": len({r["tile"] for r in records}),
            "link_patterns": len({r["links"] for r in records}),
            "arrow_counts": sorted({r["arrows"] for r in records}),
            "max_convergence": max(r["max_convergence"] for r in records)}
