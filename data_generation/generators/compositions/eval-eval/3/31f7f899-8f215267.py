"""Count-equivalence sorting of bar spans. Standard library and NumPy only."""

from collections import Counter, defaultdict

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid bounds")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "color bounds")
    return g


def _mode(a):
    counts = np.bincount(a.ravel(), minlength=10)
    best = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[best]) == 1, "ambiguous color mode")
    return best


def _components(g, bg):
    seen = np.zeros(g.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(g != bg):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        color = int(g[r, c])
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (0 <= yy < g.shape[0] and 0 <= xx < g.shape[1]
                        and not seen[yy, xx] and int(g[yy, xx]) == color):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append((color, tuple(cells)))
    return result


def _paint(shape, row, slots, spans, bg, carrier):
    g = np.full(shape, bg, dtype=np.int8)
    g[row, :] = carrier
    for (x, width, color, _, _), (lo, hi) in zip(slots, spans):
        g[row+lo:row+hi+1, x:x+width] = color
    return g


def _inspect(value):
    """Parse slots and loose objects; all conventions come from input cells."""
    original = _grid(value)
    bg = _mode(original)
    rows = np.flatnonzero(np.all(original != bg, axis=1))
    cols = np.flatnonzero(np.all(original != bg, axis=0))
    _require(len(rows) + len(cols) == 1, "one full carrier required")
    vertical = bool(len(cols))
    g = original.T if vertical else original
    row = int(cols[0] if vertical else rows[0])
    carrier = _mode(g[row])
    _require(carrier != bg and np.count_nonzero(g[row] == carrier) >= 3,
             "carrier exposure")
    _require(np.count_nonzero(g == carrier) == np.count_nonzero(g[row] == carrier),
             "carrier color outside carrier")
    slots, x = [], 0
    while x < g.shape[1]:
        color = int(g[row, x])
        end = x+1
        while end < g.shape[1] and int(g[row, end]) == color:
            end += 1
        if color != carrier:
            top = bottom = row
            while top > 0 and np.all(g[top-1, x:end] == color):
                top -= 1
            while bottom+1 < g.shape[0] and np.all(g[bottom+1, x:end] == color):
                bottom += 1
            slots.append((x, end-x, color, top-row, bottom-row))
        x = end
    _require(2 <= len(slots) <= 8, "slot count")
    colors = [s[2] for s in slots]
    _require(len(set(colors)) == len(colors), "slot colors must be distinct")
    templates = {}
    for _, _, _, lo, hi in slots:
        _require(templates.setdefault(hi-lo+1, (lo, hi)) == (lo, hi),
                 "equal lengths need identical intervals")
    skeleton = _paint(g.shape, row, slots, [s[3:] for s in slots], bg, carrier)
    occupied = skeleton != bg
    _require(np.array_equal(g[occupied], skeleton[occupied]), "broken bar")
    loose = np.where(occupied, bg, g)
    objects = _components(loose, bg)
    if objects:
        low = row + min(s[3] for s in slots)
        high = row + max(s[4] for s in slots)
        rr = [r for _, cells in objects for r, c in cells]
        _require(min(rr) >= high+2 or max(rr) <= low-2,
                 "loose objects must be beyond one side of the bar band")
    labels = np.full(g.shape, -1, dtype=int)
    for i, (_, cells) in enumerate(objects):
        _require(1 <= len(cells) <= 8, "loose object area")
        for r, c in cells:
            labels[r, c] = i
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = labels[max(0, dr):g.shape[0]+min(0, dr),
                   max(0, dc):g.shape[1]+min(0, dc)]
        b = labels[max(0, -dr):g.shape[0]+min(0, -dr),
                   max(0, -dc):g.shape[1]+min(0, -dc)]
        _require(not np.any((a >= 0) & (b >= 0) & (a != b)), "loose objects touch")
    count = Counter(color for color, _ in objects)
    area = Counter()
    for color, cells in objects:
        area[color] += len(cells)
    return dict(background=bg, carrier=carrier, vertical=vertical, row=row,
                shape=g.shape, slots=slots, objects=objects,
                counts=tuple(count[c] for c in colors),
                areas=tuple(area[c] for c in colors))


def _assign(slots, keys, metric="length"):
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    spans = [s[3:] for s in slots]
    for indices in groups.values():
        order = sorted(indices, key=lambda i: (slots[i][4]-slots[i][3]+1)
                       * (slots[i][1] if metric == "area" else 1))
        for dest, source in zip(indices, order):
            spans[dest] = slots[source][3:]
    return spans


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Sort complete spans within equal component-count classes, then erase field."""
    d = _inspect(input_grid)
    # Select the next shortest remaining interval separately for each class.
    # This also works for counts and group sizes beyond those in the sampler.
    pools = defaultdict(list)
    for slot, count in zip(d["slots"], d["counts"]):
        pools[count].append(slot[3:])
    assigned = []
    for count in d["counts"]:
        span = min(pools[count], key=lambda s: s[1]-s[0])
        pools[count].remove(span)
        assigned.append(span)
    out = _paint(d["shape"], d["row"], d["slots"], assigned,
                 d["background"], d["carrier"])
    return out.T.copy() if d["vertical"] else out


def _shape(rng, area=None):
    """Geometry is independent of how many objects a color has."""
    area = int(rng.integers(1, 9)) if area is None else area
    kind = int(rng.integers(4))
    if kind == 0:
        cells = {(0, i) for i in range(area)}
    elif kind == 1:
        width = int(rng.integers(1, min(area, 4)+1))
        cells = {(i//width, i % width) for i in range(area)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _pack(rng, shape, first_row, specs):
    h, w = shape
    taken = np.zeros(shape, dtype=bool)
    taken[:first_row] = True
    placements = []
    for color, cells in sorted(specs, key=lambda s: -len(s[1])):
        sh = max(r for r, c in cells)+1
        sw = max(c for r, c in cells)+1
        if sh > h-first_row or sw > w:
            return None
        legal = np.ones((h-sh+1, w-sw+1), dtype=bool)
        for dr, dc in cells:
            legal &= ~taken[dr:dr+h-sh+1, dc:dc+w-sw+1]
        yy, xx = np.nonzero(legal)
        if not len(yy):
            return None
        j = int(rng.integers(len(yy)))
        y, x = int(yy[j]), int(xx[j])
        placed = tuple((y+r, x+c) for r, c in cells)
        placements.append((color, placed))
        for r, c in placed:
            taken[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
    return placements


def _blueprint(rng, witness, test):
    forced = {}
    if witness == 1:
        counts = [0, 1, 0, 2, 1]
        values = [int(rng.integers(1, 3)), int(rng.integers(3, 5)), 5,
                  int(rng.integers(6, 8)), int(rng.integers(8, 10))]
        lengths = [values[i] for i in (4, 3, 0, 2, 1)]
        widths = [int(rng.integers(1, 4)), 1, int(rng.integers(1, 4)),
                  int(rng.integers(1, 4)), 3]
        # Same total area, different counts; same count, different total areas.
        forced = {1: [4], 3: [2, 2], 4: [1]}
    elif witness == 2:
        n = int(rng.integers(6, 9))
        counts = [2, 1, 2, 1, 2, 0] + [1, 2][:n-6]
        values = sorted(int(x) for x in rng.choice(np.arange(1, 12), 6, replace=False))
        lengths = [values[i] for i in (5, 4, 0, 3, 2, 1)]
        lengths += [int(rng.integers(1, 12)) for _ in range(n-6)]
        widths = [int(x) for x in rng.integers(1, 4, n)]
    elif test:
        n = int(rng.integers(4, 9))
        novel = int(rng.integers(3, 5))
        size = int(rng.integers(2, min(4 if novel == 3 else 3, n-1)+1))
        counts = [novel]*size + [int(x) for x in rng.integers(0, 3, n-size)]
        # Retain a zero class, with either one member or several members.
        counts[size] = 0
        while sum(counts) > 16:
            j = max(range(size, n), key=lambda i: counts[i])
            counts[j] -= 1
        lengths = [int(x) for x in rng.integers(1, 12, n)]
        if len(set(lengths[:size])) == 1:
            lengths[0] = lengths[0] % 11 + 1
        lengths[:size] = sorted(lengths[:size], reverse=True)
        widths = [int(x) for x in rng.integers(1, 4, n)]
    else:
        n = int(rng.integers(2, 9))
        counts = [int(x) for x in rng.integers(0, 3, n)]
        if rng.random() < .18:
            counts = [int(rng.integers(0, 3))]*n
        lengths = [int(x) for x in rng.integers(1, 12, n)]
        widths = [int(x) for x in rng.integers(1, 4, n)]
    n = len(counts)
    # Shuffle the spatial partition, preserving the teaching witnesses.
    for _ in range(100):
        p = [int(i) for i in rng.permutation(n)]
        cs, ls = [counts[i] for i in p], [lengths[i] for i in p]
        if witness:
            moved = [k for k in set(cs) if [ls[i] for i in range(n) if cs[i] == k]
                     != sorted(ls[i] for i in range(n) if cs[i] == k)]
            required = {0, 1} if witness == 1 else {1, 2}
            if not required.issubset(moved):
                continue
        elif test:
            novel_lengths = [l for c, l in zip(cs, ls) if c == max(cs)]
            if novel_lengths == sorted(novel_lengths):
                continue
        break
    else:
        p = list(range(n))
    return ([counts[i] for i in p], [lengths[i] for i in p],
            [widths[i] for i in p], {j: forced[i] for j, i in enumerate(p) if i in forced})


def _sample_pair(rng, witness, test, vertical):
    counts, lengths, widths, forced = _blueprint(rng, witness, test)
    n = len(counts)
    palette = [int(x) for x in rng.permutation(10)]
    bg, carrier, colors = palette[0], palette[1], palette[2:2+n]
    extra_colors = palette[2+n:]
    # At most 18 objects: the finite compact fallback can always fit this many.
    nd = int(rng.integers(0, min(3, 18-sum(counts))+1)) if extra_colors else 0
    if witness == 1:
        nd = max(1, nd)
    specs = []
    for i, (color, count) in enumerate(zip(colors, counts)):
        specs += [(color, _shape(rng, forced[i][j] if i in forced else None))
                  for j in range(count)]
    specs += [(int(rng.choice(extra_colors)), _shape(rng)) for _ in range(nd)]
    # All possible intervals fit in the same 11-cell perpendicular envelope.
    templates = {}
    for length in sorted(set(lengths)):
        lower = max(-5, 1-length)
        upper = min(0, 6-length)
        lo = int(rng.integers(lower, upper+1))
        templates[length] = (lo, lo+length-1)
    if witness == 1:
        # Reassign a visibly upper-biased long interval and a lower one.
        templates[max(lengths)] = (-5, max(lengths)-6)
        short = sorted(set(lengths))[1]
        templates[short] = (0, short-1)
    spans = [templates[l] for l in lengths]
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    row = int(rng.integers(1, 3))-lo
    field_top = row+hi+int(rng.integers(2, 5))
    min_width = max(15, sum(widths)+max(4, max(widths)+1))
    width = int(rng.integers(min_width, 31))
    height = int(rng.integers(max(field_top+9, 18), 31))
    placements = None
    for attempt in range(8):
        placements = _pack(rng, (height, width), field_top, specs)
        if placements is not None:
            break
        height, width = min(30, height+2), min(30, width+2)
    if placements is None:
        height = width = 30
        # Keep object areas and counts; simplify only failed random packing.
        placements = []
        for i, (color, cells) in enumerate(specs):
            y, x = field_top+4*(i//7), 4*(i % 7)
            compact = tuple((y+j//3, x+j % 3) for j in range(len(cells)))
            _require(max(r for r, c in compact) < height, "fallback packing bounds")
            placements.append((color, compact))
    gaps = [int(x) for x in rng.multinomial(width-sum(widths), [1/(n+1)]*(n+1))]
    slots, x = [], gaps[0]
    for i in range(n):
        slots.append((x, widths[i], colors[i], *spans[i]))
        x += widths[i]+gaps[i+1]
    inp = _paint((height, width), row, slots, spans, bg, carrier)
    for color, cells in placements:
        for r, c in cells:
            inp[r, c] = color
    assigned = _assign(slots, counts)
    out = _paint((height, width), row, slots, assigned, bg, carrier)
    if rng.random() < .5:
        inp, out = inp[::-1].copy(), out[::-1].copy()
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mixed = bool(rng.random() < .75)
    axis = bool(rng.integers(2))
    train_axes = [axis, not axis if mixed else axis]
    train_axes += [bool(rng.integers(2)) if mixed else axis for _ in range(nt-2)]
    test_axes = [bool(rng.integers(2)) if mixed else axis for _ in range(nq)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, axes in (("train", train_axes), ("test", test_axes)):
        for i, vertical in enumerate(axes):
            witness = i+1 if split == "train" and i < 2 else 0
            for _ in range(32):
                pair = _sample_pair(rng, witness, split == "test", vertical)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("distinct input retry limit")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "canvas changed")
    a, b = _inspect(inp), _inspect(out)
    for k in ("background", "carrier", "vertical", "row", "shape"):
        _require(a[k] == b[k], f"changed {k}")
    _require([s[:3] for s in a["slots"]] == [s[:3] for s in b["slots"]], "moved slot")
    _require(not b["objects"], "field not erased")
    # Relational validation: sortedness plus exact interval multiset per class.
    for count in set(a["counts"]):
        indices = [i for i, n in enumerate(a["counts"]) if n == count]
        before = [a["slots"][i][3:] for i in indices]
        after = [b["slots"][i][3:] for i in indices]
        _require(Counter(before) == Counter(after), "intervals crossed count classes")
        sizes = [hi-lo+1 for lo, hi in after]
        _require(sizes == sorted(sizes), "class lengths not increasing")
    return a, b


def _features(pair):
    a, b = _check_pair(pair)
    slots, counts = a["slots"], a["counts"]
    before, after = [s[3:] for s in slots], [s[3:] for s in b["slots"]]
    groups = Counter(counts)
    return dict(shape=tuple(np.asarray(pair["input"]).shape), vertical=a["vertical"],
                background=a["background"], carrier=a["carrier"], n=len(slots),
                counts=counts, areas=a["areas"], widths=tuple(s[1] for s in slots),
                lengths=tuple(s[4]-s[3]+1 for s in slots), spans=tuple(before),
                moved=before != after, global_diff=_assign(slots, [0]*len(slots)) != after,
                moved_counts=tuple(sorted({n for n, x, y in zip(counts, before, after) if x != y})),
                area_groups_diff=_assign(slots, a["areas"]) != after,
                area_sort_diff=_assign(slots, counts, "area") != after,
                reverse_diff=before[::-1] != after,
                zero_moved=any(n == 0 and x != y for n, x, y in zip(counts, before, after)),
                triple_moved=any(groups[n] >= 3 and x != y
                                 for n, x, y in zip(counts, before, after)),
                singleton=any(groups[n] == 1 for n in counts),
                interleaved=any(counts[i] == counts[k] != counts[j]
                                for i in range(len(counts)) for j in range(i+1, len(counts))
                                for k in range(j+1, len(counts))),
                masks=tuple(tuple(sorted((r-min(y for y, x in cells),
                                          c-min(x for y, x in cells)) for r, c in cells))
                            for _, cells in a["objects"]),
                distractors=sum(c not in {s[2] for s in slots} for c, _ in a["objects"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            features[split].append(_features(pair))
    tr = features["train"]
    for key in ("global_diff", "area_groups_diff", "area_sort_diff", "zero_moved",
                "triple_moved", "singleton", "interleaved"):
        _require(any(f[key] for f in tr), "missing demonstration: "+key)
    _require(sum(f["moved"] for f in tr) >= 2, "need two moved examples")
    _require(any(f["moved"] and f["reverse_diff"] for f in tr), "reversal shortcut")
    _require({0, 1, 2} <= {c for f in tr for c in f["counts"]}, "count evidence")
    _require(any(n == m and a != b for f in tr
                 for n, a in zip(f["counts"], f["areas"])
                 for m, b in zip(f["counts"], f["areas"])), "area independence")
    _require(any(n != m and a == b for f in tr
                 for n, a in zip(f["counts"], f["areas"])
                 for m, b in zip(f["counts"], f["areas"])), "area collision")
    shown_axes = {f["vertical"] for f in tr if f["moved"]}
    for f in features["test"]:
        _require(f["vertical"] in shown_axes, "undemonstrated carrier axis")
        _require(max(f["counts"]) > max(c for x in tr for c in x["counts"]),
                 "test must introduce a larger count value")
        _require(max(f["counts"]) in f["moved_counts"], "novel count class must exchange spans")


def self_test():
    """Bounded semantic, ablation, reproducibility, and 42-seed variation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fs, signatures = [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(p["input"]), _key(p["output"]))
                          for s in ("train", "test") for p in puzzle[s])
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solve mismatch")
                fs.append(_features(pair))
    _require(len(signatures) == 42, "seed repetition")
    _require(len({f["shape"] for f in fs}) >= 35, "canvas variation")
    _require({f["vertical"] for f in fs} == {False, True}, "axis variation")
    _require({f["n"] for f in fs} == set(range(2, 9)), "slot count variation")
    _require({f["background"] for f in fs} == set(range(10)), "background variation")
    _require({f["carrier"] for f in fs} == set(range(10)), "carrier variation")
    _require({c for f in fs for c in f["counts"]} == set(range(5)), "count variation")
    _require({w for f in fs for w in f["widths"]} == {1, 2, 3}, "width variation")
    _require(len({m for f in fs for m in f["masks"]}) >= 70, "shape vocabulary")
    _require(any(f["global_diff"] and f["interleaved"] for f in fs), "coupling absent")
    _require(any(len(set(f["lengths"])) < f["n"] for f in fs), "length ties absent")
    _require(any(not f["moved"] for f in fs), "no singleton/already-sorted cases")
    p0 = generate_puzzle(0)
    _require(all(np.array_equal(p[role], q[role]) for s in p0
                 for p, q in zip(p0[s], puzzles[0][s]) for role in p), "nondeterminism")
    # Mutations violate different defining invariants.
    base = puzzles[0]
    for mode in ("omit_A", "omit_B", "area_groups", "area_sort", "recenter"):
        rejected = False
        for pair in base["train"]:
            d = _inspect(pair["input"])
            slots = d["slots"]
            if mode == "omit_A":
                spans = [s[3:] for s in slots]
            elif mode == "omit_B":
                spans = _assign(slots, [0]*len(slots))
            elif mode == "area_groups":
                spans = _assign(slots, d["areas"])
            elif mode == "area_sort":
                spans = _assign(slots, d["counts"], "area")
            else:
                spans = [(-((hi-lo+1)//2), -((hi-lo+1)//2)+hi-lo)
                         for lo, hi in _assign(slots, d["counts"])]
            if any(d["row"]+lo < 0 or d["row"]+hi >= d["shape"][0] for lo, hi in spans):
                continue
            wrong = _paint(d["shape"], d["row"], slots, spans, d["background"], d["carrier"])
            if d["vertical"]:
                wrong = wrong.T.copy()
            if np.array_equal(wrong, pair["output"]):
                continue
            try:
                _check_pair({"input": pair["input"], "output": wrong})
            except AssertionError:
                rejected = True
                break
        _require(rejected, "accepted mutation: "+mode)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
