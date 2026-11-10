"""Thread-count printer: geometric card assembly changes object identity.

All answer conventions are visible in the input. The printer's start card
contains the stencil; its end card contains two ordered color swatches. Cards
are translated, and matching edge contacts are welded before counting threads.
Only Python's standard library and NumPy are used.
"""

from itertools import combinations, permutations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while cells:
        start = min(cells)
        cells.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in cells:
                    cells.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _crop(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty stencil")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _signature(mask):
    return (tuple(map(int, np.flatnonzero(mask[:, 0]))),
            tuple(map(int, np.flatnonzero(mask[:, -1]))))


def _orders(signatures, by_size=False):
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(order, remaining):
        if not by_size and len(answers) >= 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        outgoing = signatures[order[-1]][1]
        if not outgoing:
            return
        for i in remaining:
            incoming = signatures[i][0]
            match = (len(incoming) == len(outgoing) if by_size else incoming == outgoing)
            if match:
                visit(order + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _extract_cards(grid):
    """Infer equal cards, their background, and the full-line divider locally."""
    h, w = grid.shape
    candidates = []
    for d in np.unique(grid):
        rows = np.flatnonzero(np.all(grid == d, axis=1)).tolist()
        cols = np.flatnonzero(np.all(grid == d, axis=0)).tolist()
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        hs = [b - a - 1 for a, b in zip(rr, rr[1:])]
        ws = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(hs + ws) < 3 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        cards = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                 for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(cards) <= 6 or any(np.any(p == d) for p in cards):
            continue
        corners = {int(p[r, c]) for p in cards
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) == 1:
            candidates.append(dict(cards=cards, background=corners.pop(), divider=int(d),
                                   layout=(len(hs), len(ws))))
    _require(len(candidates) == 1, "one equal-card partition required")
    return candidates[0]


def _decode(grid):
    data = _extract_cards(grid)
    cards, bg = data["cards"], data["background"]
    axes = []
    for axis in (1, 0):
        ps = [p if axis == 1 else p.T for p in cards]
        if any(np.any(p[0] != bg) or np.any(p[-1] != bg) for p in ps):
            continue
        signatures = [_signature(p != bg) for p in ps]
        orders = _orders(signatures)
        if orders:
            _require(len(orders) == 1, "ambiguous complete chain")
            axes.append((axis, ps, signatures, orders[0]))
    _require(len(axes) == 1, "ports must determine one directed chain and axis")
    axis, ps, signatures, order = axes[0]
    wire_colors = {int(p[r, c]) for p in ps for c in (0, p.shape[1] - 1)
                   for r in range(p.shape[0]) if p[r, c] != bg}
    masks = [np.isin(p, tuple(wire_colors)) for p in ps]
    _require(all(len(np.unique(p[m])) == 1 for p, m in zip(ps, masks)),
             "each card has one wire color")
    _require(all(_signature(m) == s for m, s in zip(masks, signatures)), "wire port roles")
    for mask in masks:
        parts = _components(mask)
        _require(len(parts) == len(_components(mask, True)), "diagonal local ambiguity")
        _require(all(any(c in (0, mask.shape[1] - 1) for r, c in part) for part in parts),
                 "every wire fragment must reach a port")
    start, end = ps[order[0]], ps[order[-1]]
    extra = [set(map(int, np.unique(p))) - wire_colors - {bg} for p in ps]
    _require(len(extra[order[0]]) == 1 and len(extra[order[-1]]) == 2,
             "terminal stencil and two swatches required")
    _require(all(not extra[i] for i in order[1:-1]), "payload only at terminals")
    ink = next(iter(extra[order[0]]))
    _require(ink not in extra[order[-1]], "six distinct role categories")
    stencil = start == ink
    tile = _crop(stencil)
    _require(len(_components(stencil)) == 1, "connected stencil required")
    swatches = []
    for color in extra[order[-1]]:
        coords = np.argwhere(end == color)
        lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
        _require(np.all(end[lo[0]:hi[0], lo[1]:hi[1]] == color), "rectangular swatches")
        swatches.append((int(lo[1]), int(hi[1]), color, (int(hi[0]-lo[0]), int(hi[1]-lo[1]))))
    swatches.sort()
    _require(swatches[0][1] < swatches[1][0], "swatches ordered along the chain axis")
    data.update(axis=axis, oriented=ps, signatures=signatures, order=order,
                wires=masks, wire_colors=wire_colors, ink=ink, tile=tile,
                primary=swatches[0][2], secondary=swatches[1][2],
                swatch_shapes=(swatches[0][3], swatches[1][3]),
                stencil_area=int(stencil.sum()))
    return data


def _weld(masks, order):
    """Virtual assembly: dividers become blank, except for matching wire bridges."""
    parts = []
    for j, i in enumerate(order):
        if j:
            previous = masks[order[j-1]][:, -1]
            incoming = masks[i][:, 0]
            _require(np.array_equal(previous, incoming), "seam signature mismatch")
            parts.append(previous[:, None])
        parts.append(masks[i])
    return np.concatenate(parts, axis=1)


def _terminal_counts(masks, order):
    parts = _components(_weld(masks, order))
    width = masks[0].shape[1]
    last = (len(masks)-1) * (width+1)
    touch = [(any(c < width for r, c in part), any(c >= last for r, c in part))
             for part in parts]
    return (sum(a for a, b in touch), sum(b for a, b in touch),
            sum(a and b for a, b in touch))


def _union_count(masks, order, rank=False, overlap=False):
    """Count equivalence classes of fragments; also measure redundant contacts.

    This checks the physical welding by a different calculation. rank=True is
    the documented wrong rule matching bundles by size rather than positions.
    overlap=True is the documented wrong rule retaining input reading order.
    """
    labels, total = [], 0
    for mask in masks:
        ids = np.full(mask.shape, -1, dtype=int)
        for group in _components(mask):
            for p in group:
                ids[p] = total
            total += 1
        labels.append(ids)
    parent = list(range(total))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    redundant = contacts = 0
    for i, j in zip(order, order[1:]):
        a, b = labels[i][:, -1], labels[j][:, 0]
        if rank:
            _require(np.count_nonzero(a >= 0) == np.count_nonzero(b >= 0), "rank sizes")
            matches = zip(a[a >= 0], b[b >= 0])
        else:
            if not overlap:
                _require(np.array_equal(a >= 0, b >= 0), "union seam mismatch")
            matches = ((a[r], b[r]) for r in range(len(a)) if a[r] >= 0 and b[r] >= 0)
        for x, y in matches:
            contacts += 1
            x, y = root(int(x)), root(int(y))
            redundant += x == y
            parent[x] = y
    return len({root(i) for i in range(total)}), total, contacts, redundant


def _count(data):
    return len(_components(_weld(data["wires"], data["order"])))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from one input; no generator state or labels."""
    data = _decode(_grid(input_grid))
    tile, n = data["tile"], _count(data)
    h, w = tile.shape
    _require(n * (w + 1) - 1 <= 30, "printed strip exceeds grid budget")
    out = np.full((h, n * (w + 1) - 1), data["secondary"], dtype=np.int8)
    for c in range(out.shape[1]):
        local = c % (w + 1)
        if local < w:
            out[tile[:, local], c] = data["primary"]
    return out if data["axis"] == 1 else out.T.copy()


def _make_wire(h, w, incoming, outgoing, rng, split_probability=.6, groups=None):
    roots = sorted(set(incoming) | set(outgoing))
    if groups is None:
        cuts = [i for i in range(1, len(roots))
                if roots[i] - roots[i-1] >= 2 and rng.random() < split_probability]
        groups = [roots[a:b] for a, b in zip([0] + cuts, cuts + [len(roots)])]
    mask = np.zeros((h, w), dtype=bool)
    for j, group in enumerate(groups):
        hub = int(rng.integers(1, w-1))
        lo, hi = min(group), max(group)
        mask[lo:hi+1, hub] = True
        for r in group:
            if r in incoming:
                mask[r, :hub+1] = True
            if r in outgoing:
                mask[r, hub:] = True
        # Give each group its own interior band, with a full blank row between
        # bands. Connected growth adds bodies, bends and occasionally holes.
        r0 = min(roots) if j == 0 else (max(groups[j-1]) + lo) // 2 + 1
        r1 = h-1 if j == len(groups)-1 else (hi + min(groups[j+1])) // 2
        allowed = {(r, c) for r in range(r0, r1) for c in range(1, w-1)}
        body = {tuple(p) for p in np.argwhere(mask) if r0 <= p[0] < r1}
        target = int(len(allowed) * rng.uniform(.05, .85))
        for _ in range(max(0, target - len(body & allowed))):
            frontier = sorted({q for r, c in body for q in
                               ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                               if q in allowed and q not in body})
            if not frontier:
                break
            p = _choose(rng, frontier)
            body.add(p)
            mask[p] = True
    _require(_signature(mask) == (tuple(incoming), tuple(outgoing)), "construction ports")
    return mask


def _links(n, rows, rng, repeated=False):
    pool = [s for k in (1, 2, 3) for s in combinations(rows, k)]
    for _ in range(80):
        if repeated and n >= 3:
            links = [_choose(rng, pool) for _ in range(n-2)]
            position = int(rng.integers(len(links)))
            links.insert(position, links[position])
        else:
            links = [_choose(rng, pool) for _ in range(n-1)]
        if len(_orders(list(zip([()] + links, links + [()])))) == 1:
            return links
    # Distinct signatures always yield a uniquely directed chain.
    return [pool[i] for i in rng.choice(len(pool), n-1, replace=False)]


def _network(n, h, w, upper, rng, witness=None):
    rows = list(range(upper + 2, h-1))
    if witness == "assembly":
        # Three equal-size, distinct joins admit a wrong cardinality-only order.
        # Retain an input whose true topology makes that wrong order observable.
        pool = list(combinations(rows, 3))
        for _ in range(256):
            links = [pool[i] for i in rng.choice(len(pool), 3, replace=False)]
            kinds = {_kind(s) for s in links}
            if not {"thick", "separated"} <= kinds:
                continue
            masks = [_make_wire(h, w, () if i == 0 else links[i-1],
                                () if i == 3 else links[i], rng) for i in range(4)]
            count, _, _, redundant = _union_count(masks, range(4))
            wrong = _union_count(masks, (0, 2, 1, 3), rank=True)[0]
            spanning = _terminal_counts(masks, range(4))[2]
            terminal_local = (len(_components(masks[0])), len(_components(masks[-1])))
            if (2 <= count <= 4 and count != wrong and redundant > 0
                    and spanning < count and count not in terminal_local):
                return masks
        # Bounded constructive fallback, parametrized by sampled size and bodies.
        a, b, c, d, e = rows[:5]
        links = [(a, c, e), (c, d, e), (a, d, e)]
        groups = [[(a,), (c, e)], [(a,), (c, d, e)],
                  [(a,), (c, d, e)], [(a,), (d, e)]]
        return [_make_wire(h, w, () if i == 0 else links[i-1],
                           () if i == 3 else links[i], rng, groups=groups[i]) for i in range(4)]
    if witness == "one":
        # A singleton signature and a three-contact signature feed one thread.
        signatures = [_choose(rng, list(combinations(rows, k))) for k in (1, 3)]
        rng.shuffle(signatures)
        return [_make_wire(h, w, () if i == 0 else signatures[i-1],
                           () if i == 2 else signatures[i], rng, split_probability=0)
                for i in range(3)]
    links = _links(n, rows, rng, repeated=rng.random() < .35)
    masks = []
    for i in range(n):
        incoming = () if i == 0 else links[i-1]
        outgoing = () if i == n-1 else links[i]
        if incoming and outgoing and rng.random() < .25:
            mask = _separate_sides(h, w, incoming, outgoing, rng)
        else:
            mask = _make_wire(h, w, incoming, outgoing, rng,
                              split_probability=float(rng.uniform(.15, .9)))
        masks.append(mask)
    return masks


def _kind(signature):
    if len(signature) == 1:
        return "single"
    return "thick" if signature[-1] - signature[0] + 1 == len(signature) else "separated"


def _separate_sides(h, w, incoming, outgoing, rng):
    """A card can end one thread and begin another at the same transverse row."""
    mask = np.zeros((h, w), dtype=bool)
    blank = int(rng.integers(2, w-2))
    for r in incoming:
        depth = int(rng.integers(2, blank+1))
        mask[r, :depth] = True
    for r in outgoing:
        depth = int(rng.integers(2, w-blank))
        mask[r, w-depth:] = True
    return mask


def _stamp(h, w, rng, witness=False):
    # Pruning generates connected tight masks without a forced corner path.
    mask = np.ones((h, w), dtype=bool)
    target = max(h+w-1, int(h*w*rng.uniform(.35, 1)))
    for index in rng.permutation(h*w):
        if int(mask.sum()) <= target:
            break
        r, c = divmod(int(index), w)
        mask[r, c] = False
        if len(_components(mask)) != 1 or _crop(mask).shape != (h, w):
            mask[r, c] = True
    if witness and (mask.all() or any(np.array_equal(mask, _transform(mask, k)) for k in range(1, 8))):
        mask[:] = False
        mask[0] = True
        mask[:, 0] = True
        mask[-1, 1] = True
    return mask


def _transform(mask, k):
    return np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4)


def _layout(cards, permutation, shape, divider):
    nr, nc = shape
    h, w = cards[0].shape
    result = np.full((nr*(h+1)-1, nc*(w+1)-1), divider, dtype=np.int8)
    for slot, i in enumerate(permutation):
        r, c = divmod(slot, nc)
        result[r*(h+1):r*(h+1)+h, c*(w+1):c*(w+1)+w] = cards[i]
    return result


def _make_pair(rng, n, axis, witness=None, source_tile=None):
    upper = int(rng.integers(3, 5)) if witness == "assembly" else int(rng.integers(2, 5))
    if source_tile is not None:
        upper = max(upper, source_tile.shape[0])
    band = int(rng.integers(6, 12) if n >= 5 else rng.integers(5, 8))
    h = min(20 if n == 5 else 14, upper + band + 3)
    w = 5 if n == 5 else int(rng.integers(5, 10))
    if source_tile is not None:
        w = max(w, source_tile.shape[1] + 2)
    if n == 5:
        w = 5
    masks = _network(n, h, w, upper, rng, witness)
    count = len(_components(_weld(masks, range(n))))
    _require(1 <= count <= 15, "thread budget")
    if source_tile is None:
        limit = min(w-2, 31//count-1, 5)
        width = int(rng.integers(3, min(4, limit)+1)) if witness else int(rng.integers(1, limit+1))
        height = int(rng.integers(3, upper+1)) if witness else int(rng.integers(1, upper+1))
        source_tile = _stamp(height, width, rng, witness=bool(witness))
    tile = source_tile.copy()
    colors = list(map(int, rng.permutation(10)))
    bg, divider, ink, primary, secondary = colors[:5]
    mode = (0 if witness == "assembly" else int(rng.integers(1, 3))
            if witness == "one" else int(rng.integers(3)))
    k = 1 if mode == 0 else int(rng.integers(2, min(n, 5)+1)) if mode == 1 else min(n, 5)
    wire_colors = colors[5:5+k]
    labels = list(range(k)) + list(map(int, rng.integers(k, size=n-k)))
    rng.shuffle(labels)
    cards = [np.where(m, wire_colors[labels[i]], bg).astype(np.int8) for i, m in enumerate(masks)]
    top = int(rng.integers(1, upper-tile.shape[0]+2))
    left = int(rng.integers(1, w-tile.shape[1]))
    cards[0][top:top+tile.shape[0], left:left+tile.shape[1]][tile] = ink
    # Swatch rectangles sit in the free upper band; only their axial order matters.
    budget = w-3
    pwidth = int(rng.integers(1, budget))
    zwidth = int(rng.integers(1, budget-pwidth+1))
    if witness == "assembly":
        pwidth, zwidth = _choose(rng, [(a, b) for a in range(1, budget)
                                     for b in range(1, budget-a+1)
                                     if a != count and b != count])
    start = int(rng.integers(1, w-1-pwidth-zwidth))
    gap = int(rng.integers(1, w-1-start-pwidth-zwidth+1))
    for x, width, color in ((start, pwidth, primary), (start+pwidth+gap, zwidth, secondary)):
        height = int(rng.integers(1, upper+1))
        top = int(rng.integers(1, upper-height+2))
        cards[-1][top:top+height, x:x+width] = color
    if axis == 0:
        cards = [p.T.copy() for p in cards]
    ph, pw = cards[0].shape
    layouts = [(r, n//r) for r in range(1, n+1) if n % r == 0
               and r*(ph+1)-1 <= 30 and (n//r)*(pw+1)-1 <= 30]
    shape = _choose(rng, layouts)
    permutation = list(map(int, rng.permutation(n)))
    if permutation == list(range(n)):
        permutation = permutation[1:] + permutation[:1]
    if witness == "assembly":
        # Reading-order assembly is an explicit ablation, not the intended rule.
        for _ in range(50):
            permutation = list(map(int, rng.permutation(n)))
            if _union_count(masks, permutation, overlap=True)[0] != count:
                break
        else:
            permutation = list(reversed(range(n)))
    colored = np.where(tile, primary, secondary).astype(np.int8)
    gap = np.full((tile.shape[0], 1), secondary, dtype=np.int8)
    parts = [part for i in range(count) for part in
             ((colored,) if i == count-1 else (colored, gap))]
    out = np.concatenate(parts, axis=1)
    return {"input": _layout(cards, permutation, shape, divider),
            "output": out if axis == 1 else out.T.copy()}, tile


def _fresh_palette(pair, seen):
    roles = list(map(int, np.unique(pair["input"])))
    for values in permutations(range(10), len(roles)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[roles] = values
        candidate = {k: mapping[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("finite palette space exhausted")


def _legal_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mixed = bool(rng.integers(2))
    base = int(rng.integers(2))
    train_axes = [base, 1-base if mixed else base] + [int(rng.integers(2)) if mixed else base
                                                   for _ in range(num_train-2)]
    puzzle, seen, reference = {"train": [], "test": []}, set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = ("assembly" if index == 0 else "one" if index == 1 else None) if split == "train" else None
            n = 4 if witness == "assembly" else 3 if witness == "one" else int(rng.integers(2, 5) if split == "train" else rng.integers(5, 7))
            axis = train_axes[index] if split == "train" else _choose(rng, sorted(set(train_axes)))
            pair, tile = _make_pair(rng, n, axis, witness, reference if witness == "one" else None)
            if witness == "assembly":
                reference = tile
            if _key(pair["input"]) in seen:
                pair = _fresh_palette(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _cardinality_counts(data):
    return {_union_count(data["wires"], order, rank=True)[0]
            for order in _orders(data["signatures"], by_size=True)}


def _analyze(inp, out):
    data = _decode(inp)
    virtual = _weld(data["wires"], data["order"])
    count = len(_components(virtual))
    union, local, contacts, redundant = _union_count(data["wires"], data["order"])
    _require(count == union == len(_components(virtual, True)), "global thread connectivity")
    _require(count == local - contacts + redundant, "fragment equivalence invariant")
    _require(np.array_equal(solve(inp), out), "wrong printed strip")
    tile = data["tile"]
    actual = out if data["axis"] == 1 else out.T
    _require(actual.shape == (tile.shape[0], count*(tile.shape[1]+1)-1), "output crop and count")
    _require(set(map(int, np.unique(out))) <= {data["primary"], data["secondary"]}, "output palette")
    for j in range(count):
        lo = j*(tile.shape[1]+1)
        _require(np.array_equal(actual[:, lo:lo+tile.shape[1]] == data["primary"], tile), "stamp preservation")
        if j < count-1:
            _require(np.all(actual[:, lo+tile.shape[1]] == data["secondary"]), "one-cell separator")
    data.update(count=count, local=local, contacts=contacts, redundant=redundant,
                kinds={_kind(s) for a, b in data["signatures"] for s in (a, b) if s},
                repeated=len({data["signatures"][i][1] for i in data["order"][:-1]}) < len(data["cards"])-1)
    return data


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _legal_counts(len(puzzle["train"]), len(puzzle["test"]))
    records, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            records[split].append(_analyze(inp, out))
    train, test = records["train"], records["test"]
    _require(len({d["count"] for d in train}) >= 2 and any(d["count"] == 1 for d in train), "count witnesses")
    _require(any(d["count"] >= 2 and not d["tile"].all() for d in train), "crop, recoloring and repetition")
    _require(any(d["redundant"] > 0 for d in train), "redundant contact witness")
    _require(any(_terminal_counts(d["wires"], d["order"])[2] < d["count"] for d in train),
             "teach counting threads that do not span both terminals")
    _require(any(d["count"] not in (len(_components(d["wires"][d["order"][0]])),
                                    len(_components(d["wires"][d["order"][-1]])))
                 for d in train), "terminal-local counters must not fit")
    _require(any(d["count"] == 1 and len(d["wire_colors"]) > 1 for d in train),
             "teach joining different wire colors")
    _require(any(_cardinality_counts(d) - {d["count"]} for d in train), "full positions must affect answer")
    _require(any(_union_count(d["wires"], range(len(d["cards"])), overlap=True)[0] != d["count"]
                 for d in train), "input-order shortcut must fail")
    _require(all(d["count"] < d["local"] for d in train+test), "welding must merge fragments")
    demonstrated = set.union(*(d["kinds"] for d in train))
    _require({"single", "thick", "separated"} <= demonstrated, "port category teaching")
    _require(set.union(*(d["kinds"] for d in test)) <= demonstrated, "untaught port category")
    _require(max(len(d["cards"]) for d in train) < min(len(d["cards"]) for d in test), "longer-chain shift")
    for k in range(1, 8):
        _require(any(not np.array_equal(d["tile"], _transform(d["tile"], k)) for d in train), "stencil orientation")
    for role in (0, 1):
        _require(any(d["swatch_shapes"][role][1] != d["count"] for d in train), "swatch-width shortcut")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _hand_cases():
    """Rule witnesses beyond the sampler's bands, never consulted by solve."""
    # The second thread exists only between the two internal cards. Counting
    # terminal-spanning threads would return one copy rather than two.
    masks = [np.zeros((9, 5), dtype=bool) for _ in range(4)]
    masks[0][5, 3:] = True
    masks[1][5] = True
    masks[1][7, 3:] = True
    masks[2][5] = True
    masks[2][7, :2] = True
    masks[3][5, :2] = True
    cards = [np.where(m, color, 0).astype(np.int8) for m, color in zip(masks, (1, 5, 6, 9))]
    cards[0][1, 1:3] = 2
    cards[0][2, 1] = 2
    cards[-1][1, 1], cards[-1][1, 3] = 3, 4
    inp = _layout(cards, [2, 0, 3, 1], (2, 2), 8)
    expected = np.array([[3, 3, 4, 3, 3], [3, 4, 4, 3, 4]], dtype=np.int8)
    for a, b in ((inp, expected), (inp.T.copy(), expected.T.copy())):
        _require(np.array_equal(solve(a), b), "internal thread counted")
        data = _analyze(a, b)
        _require(data["count"] == 2 and data["local"] == 6, "internal thread invariant")

    # Six cards, 30 local fragments and 15 seam identifications: 15 global
    # threads. Opposite-side stubs are separated by an empty column. This is
    # exercises the separate-side vocabulary at its maximum count and
    # satisfies both connectivity conventions.
    links = [(4, 6, 8), (5, 7, 9), (4, 7, 10), (5, 8, 11), (6, 9, 12)]
    cards = []
    for i in range(6):
        card = np.zeros((14, 5), dtype=np.int8)
        if i:
            for r in links[i-1]:
                card[r, :2] = 1
        if i < 5:
            for r in links[i]:
                card[r, 3:] = 1
        cards.append(card)
    cards[0][1, 1] = 2
    cards[-1][1, 1], cards[-1][1, 3] = 3, 4
    inp = _layout(cards, [4, 1, 5, 0, 3, 2], (2, 3), 8)
    expected = np.array([[3 if c % 2 == 0 else 4 for c in range(29)]], dtype=np.int8)
    for a, b in ((inp, expected), (inp.T.copy(), expected.T.copy())):
        _require(np.array_equal(solve(a), b), "15-thread budget boundary")
        _require(_analyze(a, b)["count"] == 15, "15-thread count")


def self_test():
    """Bounded 42-seed semantic/variation audit, plus counts and mutations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    for seed in (0, 17, 41):
        _require(_puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed)), "determinism")
    records, input_shapes, stamp_shapes, geometries = [], set(), set(), set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _analyze(pair["input"], pair["output"])
                records.append(data)
                input_shapes.add(pair["input"].shape)
                stamp_shapes.add(data["tile"].shape)
                for mask in data["wires"]:
                    geometries.add(_key(mask))
    _require({len(d["cards"]) for d in records} == set(range(2, 7)), "card count coverage")
    _require({d["axis"] for d in records} == {0, 1}, "both axes")
    _require(len(input_shapes) >= 35 and len(stamp_shapes) >= 10, "geometry variation")
    _require(len({d["cards"][0].shape for d in records}) >= 20, "card proportions")
    _require(len(geometries) >= 250, "wire geometry variation")
    _require(max(d["count"] for d in records) >= 4, "print count variation")
    _require(sum(d["repeated"] for d in records) >= 3, "repeated joins")
    _require(any(len(d["wire_colors"]) >= 3 for d in records), "wire color independence")
    _require({d["background"] for d in records} == set(range(10)), "background palette")
    _require({d["divider"] for d in records} == set(range(10)), "divider palette")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(573528609, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    corrupt = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupt["test"][0]["output"].flat[0] = (int(corrupt["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    # Ambiguous duplicate self-loops cannot be ordered by colors or input slots.
    a = (1, 3)
    _require(len(_orders([((), a), (a, a), (a, a), (a, ())])) == 2, "ambiguity detection")
    _hand_cases()
    return {"seeds": 42, "pairs": len(records), "input_shapes": len(input_shapes),
            "stamp_shapes": len(stamp_shapes), "wire_masks": len(geometries),
            "print_counts": sorted({d["count"] for d in records}),
            "repeated_join_pairs": sum(d["repeated"] for d in records)}
