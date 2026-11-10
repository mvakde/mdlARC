"""Two-pin registration: intersect coordinate evidence with a motif's displacement.

Only the standard library and NumPy are used. No parent code or examples are
loaded. All semantic information is reconstructed from the visible input.
"""
from collections import Counter
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("a grid must be a 1..30 integer rectangle with colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        groups.append(sorted(cells))
    return groups


def _signature(cells):
    """The two singleton colors are pins; choose either as a computation origin."""
    counts = Counter(color for _, _, color in cells)
    pins = sorted(color for color, count in counts.items() if count == 1)
    if len(pins) != 2 or any(count < 2 for color, count in counts.items() if color not in pins):
        raise ValueError("each complete motif must have exactly two singleton pin colors")
    ar, ac = next((r, c) for r, c, color in cells if color == pins[0])
    sig = tuple(sorted((r-ar, c-ac, color) for r, c, color in cells))
    br, bc = next((r, c) for r, c, color in sig if color == pins[1])
    if br == 0 or bc == 0:
        raise ValueError("the distinct pins must occupy different rows and columns")
    return tuple(pins), sig


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    if any(int(a[r, c]) != bg for r, c in ((0, w-1), (h-1, 0), (h-1, w-1))):
        raise ValueError("the four canvas corners must show the exterior fill")
    candidates = []
    # The empty board interior is an isolated, uniformly colored rectangle.
    # This also works when interior and exterior fills have the same color.
    for fill in map(int, np.unique(a)):
        for comp in _components(a == fill):
            rs, cs = zip(*comp)
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1-r0+1 < 5 or c1-c0+1 < 5 or len(comp) != (r1-r0+1)*(c1-c0+1):
                continue
            if r0 < 2 or c0 < 2 or r1 > h-3 or c1 > w-3:
                continue
            box = (r0-1, c0-1, r1-r0+3, c1-c0+3)
            y, x, bh, bw = box
            b = a[y:y+bh, x:x+bw]
            frame = int(b[0, 0])
            if frame in (bg, fill) or any(int(b[r, c]) != frame for r, c in
                                        ((0, bw-1), (bh-1, 0), (bh-1, bw-1))):
                continue
            edge = np.concatenate((b[0], b[-1], b[1:-1, 0], b[1:-1, -1]))
            counts = np.bincount(edge, minlength=10)
            if np.any(edge == fill) or counts[frame] <= max(counts[c] for c in range(10) if c != frame):
                continue
            candidates.append((box, fill, frame))
    if len(candidates) != 1:
        raise ValueError("one empty coordinate board must be visible")
    box, fill, frame = candidates[0]
    y, x, bh, bw = box
    b = a[y:y+bh, x:x+bw]
    outside = a != bg
    outside[y:y+bh, x:x+bw] = False
    templates, repetitions = {}, Counter()
    for comp in _components(outside):
        if not 4 <= len(comp) <= 15:
            raise ValueError("reference motifs must have 4..15 occupied cells")
        rs, cs = zip(*comp)
        if max(rs)-min(rs)+1 > 5 or max(cs)-min(cs)+1 > 5:
            raise ValueError("reference motifs must fit a 5 by 5 box")
        pair, sig = _signature([(r, c, int(a[r, c])) for r, c in comp])
        if pair in templates and templates[pair] != sig:
            raise ValueError("conflicting definitions of a pin pair")
        templates[pair] = sig
        repetitions[pair] += 1
    if not 1 <= len(templates) <= 3:
        raise ValueError("one to three distinct motifs are required")
    keys = set().union(*map(set, templates))
    if keys & {bg, fill, frame}:
        raise ValueError("pin colors must be distinct from the structural colors")
    if any(color in keys for sig in templates.values() for _, _, color in sig
           if sum(v == color for _, _, v in sig) > 1):
        raise ValueError("a decoration color cannot also be a pin")
    rows, cols = {}, {}
    for color in keys:
        rows[color] = tuple(r for r in range(1, bh-1) if b[r, 0] == b[r, -1] == color)
        cols[color] = tuple(c for c in range(1, bw-1) if b[0, c] == b[-1, c] == color)
    # The exterior has no noise: every non-fill cell belongs to a prototype.
    return dict(a=a, board=b, box=box, bg=bg, fill=fill, frame=frame,
                templates=templates, repetitions=dict(repetitions), rows=rows, cols=cols)


def _placements(templates, rows, cols, shape, second_pin=True, centered=False):
    bh, bw = shape
    stamps = []
    for pair, sig in sorted(templates.items()):
        p, q = pair
        dr, dc = next((r, c) for r, c, color in sig if color == q)
        offset_r = offset_c = 0
        if centered:
            offset_r = (min(r for r, _, _ in sig) + max(r for r, _, _ in sig)) // 2
            offset_c = (min(c for _, c, _ in sig) + max(c for _, c, _ in sig)) // 2
        for r in rows.get(p, ()):
            for c in cols.get(p, ()):
                if second_pin and (r+dr not in rows.get(q, ()) or c+dc not in cols.get(q, ())):
                    continue
                cells = tuple((r+rr-offset_r, c+cc-offset_c, color) for rr, cc, color in sig)
                if not all(1 <= rr < bh-1 and 1 <= cc < bw-1 for rr, cc, _ in cells):
                    if second_pin and not centered:
                        raise ValueError("a matched motif would cross the board boundary")
                    continue
                stamps.append((pair, r, c, cells))
    return stamps


def _paint(board, stamps, *, opaque=False, fill=None):
    out = board.copy()
    occupied = {}
    for _, _, _, cells in stamps:
        if opaque:
            rs, cs, _ = zip(*cells)
            out[min(rs):max(rs)+1, min(cs):max(cs)+1] = fill
        for r, c, color in cells:
            if (r, c) in occupied:
                raise ValueError("accepted motif footprints must be disjoint")
            occupied[r, c] = color
            out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the completed, frame-inclusive board using only visible cells."""
    p = _parse(input_grid)
    stamps = _placements(p["templates"], p["rows"], p["cols"], p["board"].shape)
    return _paint(p["board"], stamps)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_shape(rng, extent, hole=False, require_extent=None):
    for _ in range(100):
        h, w = int(rng.integers(2, extent+1)), int(rng.integers(2, extent+1))
        size = int(rng.integers(4, min(15, h*w)+1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < size:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            # Branches, compact patches, and narrow paths all remain possible.
            cells.add(_pick(rng, frontier))
        cells = sorted(cells)
        pins = [(a, b) for a in cells for b in cells if a[0] != b[0] and a[1] != b[1]]
        rs, cs = zip(*cells)
        area = (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)
        actual_extent = max(max(rs)-min(rs)+1, max(cs)-min(cs)+1)
        if (not pins or hole and area == len(cells)
                or require_extent is not None and actual_extent != require_extent):
            continue
        a, b = _pick(rng, pins)
        if hole and (2*a[0] == min(rs)+max(rs) and 2*a[1] == min(cs)+max(cs)):
            continue
        return cells, a, b
    if require_extent is not None:
        # Connected long L: its two pins still differ along both axes.
        cells = [(r, 0) for r in range(require_extent)] + [(require_extent-1, 1)]
        return cells, (0, 0), (require_extent-1, 1)
    # A connected, asymmetric transparent fallback, with diagonal pin separation.
    return [(0, 0), (1, 0), (1, 1), (2, 0)], (0, 0), (1, 1)


def _key_pairs(rng, count, graph):
    if count == 1:
        return [(0, 1)], 2
    if graph == "disjoint":
        return [(2*i, 2*i+1) for i in range(count)], 2*count
    if graph == "path":
        return [(i, i+1) for i in range(count)], count+1
    if graph == "triangle" and count == 3:
        return [(0, 1), (1, 2), (0, 2)], 3
    return [(0, i+1) for i in range(count)], count+1


def _templates(rng, count, extent, graph, distinct_fill, hole, require_extent=None):
    pairs, key_count = _key_pairs(rng, count, graph)
    palette = list(map(int, rng.permutation(10)))
    bg, frame = palette[:2]
    fill = palette[2] if distinct_fill else bg
    available = [c for c in palette if c not in (bg, frame, fill)]
    keys = available[:key_count]
    decorations = [c for c in palette if c not in keys and c not in (bg, fill)]
    templates = {}
    for i, (pi, qi) in enumerate(pairs):
        cells, a, b = _make_shape(rng, extent, hole and i == 0,
                                 require_extent if i == 0 else None)
        colors = [_pick(rng, decorations)]
        if len(cells) >= 6 and len(decorations) >= 2 and rng.random() < .55:
            colors = list(map(int, rng.choice(decorations, 2, replace=False)))
        rest = [p for p in cells if p not in (a, b)]
        assigned = colors*2 + [_pick(rng, colors) for _ in range(len(rest)-2*len(colors))]
        rng.shuffle(assigned)
        raw = [(a[0], a[1], keys[pi]), (b[0], b[1], keys[qi])]
        raw += [(r, c, color) for (r, c), color in zip(rest, assigned)]
        pair, sig = _signature(raw)
        templates[pair] = sig
    return bg, fill, frame, templates


def _axis_matches(templates, marks, axis):
    result = []
    for pair, sig in sorted(templates.items()):
        p, q = pair
        delta = next((r, c)[axis] for r, c, color in sig if color == q)
        result.append(tuple(sorted(v for v in marks.get(p, ()) if v+delta in marks.get(q, ()))))
    return tuple(result)


def _reserve_axis(rng, templates, length, axis, repetitions):
    marks = {key: set() for pair in templates for key in pair}
    owners = {}
    for i, (pair, sig) in enumerate(sorted(templates.items())):
        p, q = pair
        coords = [(r, c)[axis] for r, c, _ in sig]
        low, high = 1-min(coords), length-2-max(coords)
        delta = next((r, c)[axis] for r, c, color in sig if color == q)
        chosen = []
        for _ in range(repetitions[i][axis]):
            options = [v for v in range(low, high+1)
                       if owners.get(v, p) == p and owners.get(v+delta, q) == q
                       and all(abs(v-u) >= max(coords)-min(coords)+2 for u in chosen)]
            if not options:
                raise ValueError("no coordinate room")
            v = int(_pick(rng, options))
            marks[p].add(v)
            marks[q].add(v+delta)
            owners[v], owners[v+delta] = p, q
            chosen.append(v)
    return marks, owners


def _decoys(rng, templates, marks, owners, length, axis, required=False):
    baseline = _axis_matches(templates, marks, axis)
    # Add unrelated row/column requests without secretly pairing whole points.
    for key in rng.permutation(sorted(marks)):
        key = int(key)
        want = 1 if required else int(rng.integers(0, 3))
        for _ in range(want):
            options = []
            for v in range(3, length-3):
                if v in owners:
                    continue
                marks[key].add(v)
                if _axis_matches(templates, marks, axis) == baseline:
                    options.append(v)
                marks[key].remove(v)
            if not options:
                if required:
                    raise ValueError("no incompatible coordinate room")
                break
            v = int(_pick(rng, options))
            marks[key].add(v)
            owners[v] = key


def _construct(rng, kind, curriculum):
    if kind == "first":
        count, extent, graph = 1, 4, "hub"
    elif kind == "second":
        count, extent, graph = 2, 3, "hub"
    elif kind == "test":
        count = 3 if curriculum == "network" else int(rng.integers(1, 3))
        extent = 4 if curriculum == "recombine" else 5
        graph = str(_pick(rng, ("path", "triangle"))) if curriculum == "network" else "hub"
    else:
        count, extent = int(rng.integers(1, 4)), int(rng.integers(3, 5))
        graph = "hub" if curriculum == "network" else str(_pick(rng, ("hub", "path", "triangle", "disjoint")))
    distinct_fill = kind in ("first", "second") or rng.random() < .65
    bg, fill, frame, templates = _templates(rng, count, extent, graph, distinct_fill, kind == "first",
                                          5 if kind == "test" and curriculum == "scale" else None)
    signatures = list(templates.values())
    bounds = [(max(r for r, _, _ in s)-min(r for r, _, _ in s)+1,
               max(c for _, c, _ in s)-min(c for _, c, _ in s)+1) for s in signatures]
    ref_counts = [2 if kind == "first" else 1 for _ in signatures]
    if kind != "first" and rng.random() < .4:
        ref_counts[int(rng.integers(count))] += 1
    ref_h = sum(bh*n for (bh, _), n in zip(bounds, ref_counts)) + sum(ref_counts)-1
    if ref_h > 26:
        ref_counts = [1]*count
        ref_h = sum(bh for bh, _ in bounds) + count-1
    strip_w = max(bw for _, bw in bounds)+2
    gap = int(rng.integers(1, 3))
    max_bw = 30-strip_w-gap-2
    bh = int(rng.integers(19 if kind == "second" else 14, 27))
    bw = int(rng.integers(17 if kind == "second" else 13, max_bw+1))
    if kind == "second":
        reps = [(2, 2), (1, 1)]
    elif kind == "first":
        reps = [(1, 2)]
    else:
        reps = [(1, 1) for _ in templates]
        if kind == "test" and curriculum == "recombine":
            reps[int(rng.integers(count))] = _pick(rng, [(3, 1), (1, 3)])
        elif count <= 2 and rng.random() < .65:
            reps[int(rng.integers(count))] = _pick(rng, [(2, 1), (1, 2), (2, 2)])
    rows, ro = _reserve_axis(rng, templates, bh, 0, reps)
    cols, co = _reserve_axis(rng, templates, bw, 1, reps)
    # Shared keys can yield additional legitimate combinations. They are never
    # suppressed: reconstruct all of them, then require disjoint footprints.
    _decoys(rng, templates, rows, ro, bh, 0, kind == "first")
    _decoys(rng, templates, cols, co, bw, 1, kind == "first")
    board = np.full((bh, bw), fill, dtype=np.int8)
    board[[0, -1], :] = frame
    board[:, [0, -1]] = frame
    for key, values in rows.items():
        for v in values:
            board[v, 0] = board[v, -1] = key
    for key, values in cols.items():
        for v in values:
            board[0, v] = board[-1, v] = key
    if kind == "first":
        # A one-sided SECOND-pin column aligned to a real first-pin decoy:
        # looking at only the top edge would create an extra complete motif.
        pair, sig = next(iter(templates.items()))
        p, q = pair
        dc = next(c for _, c, color in sig if color == q)
        lo = 1-min(c for _, c, _ in sig)
        hi = bw-2-max(c for _, c, _ in sig)
        choices = [c+dc for c in cols[p] if lo <= c <= hi and c+dc not in co]
        if not choices:
            raise ValueError("no one-sided pairing witness")
        v = int(_pick(rng, choices))
        board[0, v] = q
    else:
        # Some labels name no prototype pin. They are inert even with both axes.
        absent = [c for c in range(10) if c not in set(rows) and c not in (bg, fill, frame)]
        if absent and rng.random() < .45:
            color = int(_pick(rng, absent))
            rr = [r for r in range(1, bh-1) if r not in ro]
            cc = [c for c in range(1, bw-1) if c not in co]
            if rr and cc:
                r, c = int(_pick(rng, rr)), int(_pick(rng, cc))
                board[r, 0] = board[r, -1] = color
                board[0, c] = board[-1, c] = color
    # Ground truth below uses construction records, not the input parser.
    stamps = _placements(templates, rows, cols, board.shape)
    out = _paint(board, stamps)
    if not stamps or len({pair for pair, _, _, _ in stamps}) != count:
        raise ValueError("every sampled type must contribute")
    height = max(bh, ref_h)+2+int(rng.integers(0, max(1, 29-max(bh, ref_h))))
    width = strip_w+gap+bw+2
    a = np.full((height, width), bg, dtype=np.int8)
    by = int(rng.integers(1, height-bh))
    bx = strip_w+gap+1
    a[by:by+bh, bx:bx+bw] = board
    top = int(rng.integers(1, height-ref_h))
    # Irregular vertical offsets and horizontal positions make prototypes
    # reference objects rather than a fixed lookup-table layout.
    for sig, (mh, mw), repetitions in zip(signatures, bounds, ref_counts):
        lr, lc = min(r for r, _, _ in sig), min(c for _, c, _ in sig)
        for _ in range(repetitions):
            left = int(rng.integers(1, strip_w-mw+1))
            for r, c, color in sig:
                a[top+r-lr, left+c-lc] = color
            top += mh+1
    turns = int(rng.integers(4))
    a, out = np.rot90(a, turns).copy(), np.rot90(out, turns).copy()
    pair = {"input": a, "output": out}
    f = _features(pair)
    if kind == "first" and not all(f[name] for name in
                                  ("ignore_second_difference", "single_edge_difference",
                                   "center_difference", "transparent_holes")):
        raise ValueError("first demonstration failed to distinguish shortcuts")
    if kind == "second" and not (f["cartesian"] and f["shared_pin"]):
        raise ValueError("second demonstration failed to expose the shared-pin join")
    if kind != "test" and any(max(m) > 2 for m in f["products"]):
        raise ValueError("training must retain the demonstrated one/two-axis range")
    return pair


def _make_pair(rng, kind, curriculum):
    for _ in range(400):
        try:
            return _construct(rng, kind, curriculum)
        except ValueError:
            continue
    raise RuntimeError("bounded sampling could not produce a registration scene")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    curriculum = str(_pick(rng, ("network", "scale", "recombine")))
    result = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            kind = "test" if split == "test" else "first" if i == 0 else "second" if i == 1 else "other"
            for _ in range(20):
                pair = _make_pair(rng, kind, curriculum)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded sampling could not make distinct inputs")
            seen.add(key)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def _shortcut_output(p, mode):
    rows, cols = p["rows"], p["cols"]
    if mode == "pins_only":
        out = p["board"].copy()
        for key in rows:
            for r in rows[key]:
                for c in cols[key]:
                    out[r, c] = key
        return out
    if mode == "single_edge":
        b = p["board"]
        rows = {key: tuple(r for r in range(1, b.shape[0]-1) if b[r, 0] == key or b[r, -1] == key)
                for key in p["rows"]}
        cols = {key: tuple(c for c in range(1, b.shape[1]-1) if b[0, c] == key or b[-1, c] == key)
                for key in p["cols"]}
    try:
        stamps = _placements(p["templates"], rows, cols, p["board"].shape,
                             second_pin=mode != "ignore_second", centered=mode == "center")
        return _paint(p["board"], stamps, opaque=mode == "opaque", fill=p["bg"])
    except ValueError:
        # The wrong rule can also create conflicts or cross the frame. Such a
        # failure is distinguishable, but first-example witnesses below require
        # a literal extra/shifted output whenever the alternative is defined.
        return None


def _features(pair):
    p = _parse(pair["input"])
    stamps = _placements(p["templates"], p["rows"], p["cols"], p["board"].shape)
    predicted = _paint(p["board"], stamps)
    if not np.array_equal(predicted, np.asarray(pair["output"])):
        raise ValueError("incorrect registration output")
    row_matches = _axis_matches(p["templates"], p["rows"], 0)
    col_matches = _axis_matches(p["templates"], p["cols"], 1)
    candidates = sum(len(p["rows"][pair[0]])*len(p["cols"][pair[0]]) for pair in p["templates"])
    keys = [key for pair in p["templates"] for key in pair]
    holes = sum((max(r for r, _, _ in sig)-min(r for r, _, _ in sig)+1)
                *(max(c for _, c, _ in sig)-min(c for _, c, _ in sig)+1)-len(sig)
                for sig in p["templates"].values())
    y, x, bh, bw = p["box"]
    outside = p["a"] != p["bg"]
    outside[y:y+bh, x:x+bw] = False
    rr, cc = np.nonzero(outside)
    side = "top" if max(rr) < y else "bottom" if min(rr) >= y+bh else "left" if max(cc) < x else "right"
    result = dict(types=len(p["templates"]), pins=len(set(keys)), shared_pin=len(keys) > len(set(keys)),
                  accepted=len(stamps), rejected=candidates-len(stamps),
                  cartesian=any(len(r)>1 and len(c)>1 for r, c in zip(row_matches, col_matches)),
                  transparent_holes=holes > 0 and p["fill"] != p["bg"],
                  source_side=side, input_shape=p["a"].shape, output_shape=p["board"].shape,
                  equal_fills=p["bg"] == p["fill"], products=tuple((len(r), len(c)) for r, c in zip(row_matches, col_matches)),
                  geometries=tuple(tuple((r, c) for r, c, _ in sig) for sig in p["templates"].values()),
                  sizes=tuple(len(sig) for sig in p["templates"].values()),
                  extents=tuple((max(r for r, _, _ in sig)-min(r for r, _, _ in sig)+1,
                                 max(c for _, c, _ in sig)-min(c for _, c, _ in sig)+1) for sig in p["templates"].values()),
                  repeated_reference=any(n > 1 for n in p["repetitions"].values()),
                  multicolor_decoration=any(len(set(color for _, _, color in sig))-2 > 1 for sig in p["templates"].values()),
                  pin_degree_profile=tuple(sorted(Counter(keys).values())),
                  palette=tuple(map(int, np.unique(p["a"]))))
    for mode in ("ignore_second", "single_edge", "center", "opaque", "pins_only"):
        wrong = _shortcut_output(p, mode)
        result[mode+"_difference"] = wrong is not None and not np.array_equal(wrong, predicted)
    # Omitting A leaves no target anchor points: a B-only copying rule returns
    # the unchanged empty board. This must differ in the demonstrations.
    result["ignore_coordinates_difference"] = not np.array_equal(p["board"], predicted)
    return result


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must contain exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("both splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(ne, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("more training than test examples are required")
    seen, evidence, queries = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("each pair must contain exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            if _key(inp) in seen:
                raise ValueError("duplicate input")
            seen.add(_key(inp))
            f = _features({"input": inp, "output": out})
            if not f["accepted"] or not f["ignore_coordinates_difference"]:
                raise ValueError("each generated pair must actually register a motif")
            # The parser and exact prediction also check border preservation,
            # every colored offset, paired labels, strict fit, and disjointness.
            if split == "train":
                evidence.append(f)
            else:
                queries.append(f)
    for name in ("ignore_second_difference", "single_edge_difference", "center_difference",
                 "opaque_difference", "transparent_holes", "ignore_coordinates_difference",
                 "cartesian", "shared_pin", "repeated_reference", "pins_only_difference"):
        if not any(f[name] for f in evidence):
            raise ValueError("missing distinguishing training evidence: "+name)
    train_extent = max(max(e) for f in evidence for e in f["extents"])
    train_mult = max(max(m) for f in evidence for m in f["products"])
    train_graphs = {f["pin_degree_profile"] for f in evidence}
    for f in queries:
        if not (max(max(e) for e in f["extents"]) > train_extent
                or max(max(m) for m in f["products"]) > train_mult
                or f["types"] == 3 and f["pin_degree_profile"] not in train_graphs):
            raise ValueError("test lacks the intended visible structural shift")


def self_test():
    # A hand-constructed regression: expected coordinates are written directly,
    # without _placements, generation, or reference parsing. Both pins are
    # off-center, and the reference background must stay transparent.
    manual = np.zeros((12, 22), dtype=np.int8)
    board = np.full((10, 12), 3, dtype=np.int8)
    board[[0, -1], :] = 2
    board[:, [0, -1]] = 2
    board[[3, 6], 0] = board[[3, 6], -1] = 4
    board[4, 0] = board[4, -1] = 5
    board[0, [3, 7]] = board[-1, [3, 7]] = 4
    board[0, [4, 8]] = board[-1, [4, 8]] = 5
    manual[1:11, 9:21] = board
    manual[2, 2] = 4
    manual[3, 3] = 5
    manual[3, 2] = manual[4, 2] = manual[4, 3] = 6
    expected = board.copy()
    expected[3, [3, 7]] = 4
    expected[4, [4, 8]] = 5
    expected[4, [3, 7]] = 6
    expected[5, [3, 4, 7, 8]] = 6
    assert np.array_equal(solve(manual), expected)
    # Change A's visible row evidence while keeping B's exact motif fixed.
    moved_marks = manual.copy()
    moved_marks[5, [9, 20]] = 2  # old q row, in canvas coordinates
    moved_marks[8, [9, 20]] = 5  # new q row licenses p at board row 6
    moved_expected = board.copy()
    moved_expected[4, [0, 11]] = 2
    moved_expected[7, [0, 11]] = 5
    moved_expected[6, [3, 7]] = 4
    moved_expected[7, [4, 8]] = 5
    moved_expected[7, [3, 7]] = 6
    moved_expected[8, [3, 4, 7, 8]] = 6
    assert np.array_equal(solve(moved_marks), moved_expected)
    assert not np.array_equal(expected, moved_expected)
    # Change B's pin displacement, preserving the silhouette and all A marks.
    # Neither candidate row then registers both pins; an empty crop is valid.
    moved_pin = manual.copy()
    moved_pin[3, 3], moved_pin[4, 3] = 6, 5
    assert np.array_equal(solve(moved_pin), board)
    assert not np.array_equal(solve(moved_pin), expected)
    # Conflicting complete definitions of one pin pair must be rejected.
    conflicting = manual.copy()
    conflicting[7:10, 2:4] = moved_pin[2:5, 2:4]
    try:
        solve(conflicting)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting prototype definitions accepted")
    records, puzzles = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solve must not mutate its argument"
                records.append(_features(pair))
    assert len(set(puzzles)) == 42
    assert len({f["input_shape"] for f in records}) >= 35
    assert len({f["output_shape"] for f in records}) >= 35
    assert len({g for f in records for g in f["geometries"]}) >= 100
    assert {f["types"] for f in records} == {1, 2, 3}
    assert {f["source_side"] for f in records} == {"top", "bottom", "left", "right"}
    assert any(f["equal_fills"] for f in records)
    assert any(f["multicolor_decoration"] for f in records)
    assert any(max(e) == 5 for f in records for e in f["extents"])
    assert {(1, 1), (1, 2), (2, 1), (2, 2)} <= {m for f in records for m in f["products"]}
    assert all(any(f[name] for f in records) for name in
               ("rejected", "shared_pin", "cartesian", "opaque_difference", "center_difference"))
    first = generate_puzzle(9876, 6, 3)
    second = generate_puzzle(9876, 6, 3)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s])
               for k in ("input", "output"))
    # Check color and orientation covariance without changing the family rule.
    demo = first["train"][0]
    recolor = np.random.default_rng(122).permutation(10)
    assert np.array_equal(solve(recolor[demo["input"]]), recolor[demo["output"]])
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(demo["input"], turns)), np.rot90(demo["output"], turns))
    assert np.array_equal(solve(demo["input"][:, ::-1]), demo["output"][:, ::-1])
    # Mutations exercise validation, including the two inherited mechanisms.
    for mode in ("ignore_second", "single_edge", "center", "opaque", "pins_only"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        wrong = _shortcut_output(_parse(bad["train"][0]["input"]), mode)
        assert wrong is not None and not np.array_equal(wrong, bad["train"][0]["output"])
        bad["train"][0]["output"] = wrong
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("shortcut mutation accepted: "+mode)
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(400+nt+ne, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
