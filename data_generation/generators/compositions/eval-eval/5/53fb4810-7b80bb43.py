"""Periodic waves on selectively repaired orthogonal rail graphs.

One hub carries a complete color word. Its wave follows shortest rail distance,
including turns and branches. Only a bracketed three-cell break with a uniquely
attached two-cell diagonal is repaired; other breaks continue to block the wave.
No parent generator, stored puzzle, seed, or construction record is used by solve.
"""

from collections import deque
import itertools
import numpy as np


CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return g.astype(np.int8, copy=True)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _at(mask, p):
    return _inside(mask.shape, p) and bool(mask[p])


def _point(p, step, k=1):
    return p[0] + k * step[0], p[1] + k * step[1]


def _cells(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _components(mask, diagonal=False):
    pending = _cells(mask)
    steps = (tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if r or c) if diagonal else CARDINAL)
    result = []
    while pending:
        p = min(pending)
        pending.remove(p)
        group, queue = {p}, [p]
        while queue:
            p = queue.pop()
            for step in steps:
                q = _point(p, step)
                if q in pending:
                    pending.remove(q)
                    group.add(q)
                    queue.append(q)
        result.append(group)
    return result


def _strong(mask):
    """Membership in at least one intact straight run of three cells."""
    result = np.zeros_like(mask)
    for axis in (0, 1):
        m, out = (mask, result) if axis == 0 else (mask.T, result.T)
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for offset in range(3):
            out[:, offset:offset + triples.shape[1]] |= triples
    return result


def _thick(mask):
    return bool((mask[:-1, :-1] & mask[1:, :-1]
                 & mask[:-1, 1:] & mask[1:, 1:]).any())


def _hub_shape(h, w, kind):
    m = np.ones((h, w), dtype=bool)
    if kind == "clipped":
        m[0, 0] = m[0, -1] = m[-1, 0] = m[-1, -1] = False
    elif kind in ("cross", "cross_t"):
        m[:] = False
        m[1:4, :] = True
        m[:, 2] = True
        if kind == "cross_t":
            m = m.T
    return m


def _parse(g):
    values, counts = np.unique(g, return_counts=True)
    order = np.argsort(counts)
    if len(values) < 4 or counts[order[-1]] == counts[order[-2]]:
        raise ValueError("a unique dominant background and four roles are required")
    bg = int(values[order[-1]])
    blocks = [int(c) for c in values if c != bg and _thick(g == c)]
    if len(blocks) != 1:
        raise ValueError("exactly one hub color must contain a solid 2x2 block")
    hub = blocks[0]
    hm = g == hub
    rr, cc = np.nonzero(hm)
    top, left = int(rr.min()), int(cc.min())
    h, w = int(rr.max()) - top + 1, int(cc.max()) - left + 1
    body = hm[top:top+h, left:left+w]
    allowed = ["rectangle", "clipped"] + (["cross", "cross_t"] if h == w == 5 else [])
    if (h not in (3, 5) or w not in (3, 5)
            or not any(np.array_equal(body, _hub_shape(h, w, k)) for k in allowed)
            or len(_components(hm)) != 1):
        raise ValueError("unrecognized hub")
    others = [(int(np.count_nonzero(g == c)), int(c)) for c in values
              if c not in (bg, hub)]
    others.sort(reverse=True)
    if len(others) < 2 or others[0][0] == others[1][0]:
        raise ValueError("the rail color must be more frequent than each word color")
    rail = others[0][1]
    faces = ((top, left+w//2), (top+h-1, left+w//2),
             (top+h//2, left), (top+h//2, left+w-1))
    ports = []
    for p, step in zip(faces, CARDINAL):
        q = _point(p, step)
        if _inside(g.shape, q) and int(g[q]) not in (bg, hub, rail):
            ports.append((q, step))
    if len(ports) != 1:
        raise ValueError("the hub must have exactly one centered color-word port")
    source, direction = ports[0]
    points, word = [], []
    p = source
    while _inside(g.shape, p) and int(g[p]) not in (bg, hub, rail):
        points.append(p)
        word.append(int(g[p]))
        p = _point(p, direction)
    if not 1 <= len(word) <= 5 or not _inside(g.shape, p) or int(g[p]) != rail:
        raise ValueError("a complete straight word must join the rails")
    wm = np.zeros_like(g, dtype=bool)
    for p in points:
        wm[p] = True
    if not np.array_equal(wm, (g != bg) & (g != hub) & (g != rail)):
        raise ValueError("word colors may appear only in the source stem")
    # The hub box's corner cutouts are empty, and only the port touches its body.
    if not np.all((g[top:top+h, left:left+w] == hub)
                  | (g[top:top+h, left:left+w] == bg)):
        raise ValueError("material inside a hub cutout")
    touching = {q for p in _cells(hm) for step in CARDINAL
                if _inside(g.shape, q := _point(p, step)) and g[q] not in (bg, hub)}
    if touching != {source}:
        raise ValueError("only the word may touch the hub")
    return dict(bg=bg, hub=hub, rail=rail, hub_mask=hm, word_mask=wm,
                word=tuple(word), source=source, direction=direction,
                word_cells=tuple(points), hub_kind=next(k.replace("cross_t", "cross") for k in allowed
                    if np.array_equal(body, _hub_shape(h, w, k))), hub_size=(h, w))


def _clean_rails(mask, word_mask, hub_mask):
    """Input-only B evidence. Word cells are protected, not diagonal noise."""
    track = mask & ~word_mask
    strong = _strong(mask)
    repairs = []
    for step in ((0, 1), (1, 0)):
        perpendicular = (step[1], step[0])
        for anchor in sorted(_cells(track)):
            far = _point(anchor, step, 4)
            gap = tuple(_point(anchor, step, k) for k in (1, 2, 3))
            if not _at(track, far):
                continue
            if any(_at(mask | hub_mask, p) for p in gap):
                continue
            if not (_at(mask, _point(anchor, step, -1))
                    or _at(mask, _point(anchor, step, 5))):
                continue
            matches = []
            for endpoint, toward in ((anchor, 1), (far, -1)):
                for side in (-1, 1):
                    diagonal = (toward*step[0] + side*perpendicular[0],
                                toward*step[1] + side*perpendicular[1])
                    spur = tuple(_point(endpoint, diagonal, k) for k in (1, 2))
                    if (all(_at(track, p) and not _at(strong, p) for p in spur)
                            and not _at(track, _point(endpoint, diagonal, 3))):
                        matches.append(spur)
            if len(matches) > 1:
                raise ValueError("ambiguous diagonal evidence for a gap")
            if matches:
                repairs.append(dict(gap=gap, spur=matches[0],
                                    anchors=(anchor, far), axis="h" if step[1] else "v"))
    added, anchors = np.zeros_like(mask), np.zeros_like(mask)
    used_spurs = set()
    for repair in repairs:
        if used_spurs & set(repair["spur"]):
            raise ValueError("one spur supports multiple repairs")
        used_spurs.update(repair["spur"])
        for p in repair["gap"]:
            if added[p]:
                raise ValueError("overlapping repairs")
            added[p] = True
        for p in repair["anchors"]:
            anchors[p] = True
    weak = track & ~_strong(mask | added) & ~anchors
    removed = np.zeros_like(mask)
    for p in _cells(weak):
        if any(_at(weak, _point(p, step))
               for step in ((1, 1), (1, -1), (-1, 1), (-1, -1))):
            removed[p] = True
    return (mask | added) & ~removed, repairs, removed


def _distances(mask, source):
    distances = np.full(mask.shape, -1, dtype=np.int16)
    distances[source] = 0
    queue = deque([source])
    while queue:
        p = queue.popleft()
        for step in CARDINAL:
            q = _point(p, step)
            if _at(mask, q) and distances[q] < 0:
                distances[q] = distances[p] + 1
                queue.append(q)
    return distances


def _reconstruct(g, repair=True):
    roles = _parse(g)
    raw = (g == roles["rail"]) | roles["word_mask"]
    clean, repairs, removed = _clean_rails(raw, roles["word_mask"], roles["hub_mask"])
    if not repair:
        for record in repairs:
            for p in record["gap"]:
                clean[p] = False
    d = _distances(clean, roles["source"])
    for i, p in enumerate(roles["word_cells"]):
        if d[p] != i:
            raise ValueError("the word stem must be a single source corridor")
        degree = sum(_at(clean, _point(p, step)) for step in CARDINAL)
        if degree != (1 if i == 0 else 2):
            raise ValueError("extra connection to the word stem")
    out = g.copy()
    out[removed] = roles["bg"]
    out[clean & ~raw] = roles["rail"]
    reached = d >= 0
    word = np.asarray(roles["word"], dtype=np.int8)
    out[reached] = word[d[reached] % len(word)]
    return out, dict(roles=roles, raw=raw, clean=clean, repairs=repairs,
                     removed=removed, distances=d)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell solely from visible input geometry and colors."""
    return _reconstruct(_grid(input_grid))[0]


def _edge(a, b):
    return tuple(sorted((a, b)))


def _grow_graph(rng, rows, cols, root, target):
    active, edges = {root}, set()
    while len(active) < target:
        frontier = [(a, _point(a, s)) for a in sorted(active) for s in CARDINAL
                    if 0 <= _point(a, s)[0] < rows and 0 <= _point(a, s)[1] < cols
                    and _point(a, s) not in active]
        a, b = frontier[int(rng.integers(len(frontier)))]
        edges.add(_edge(a, b))
        active.add(b)
    return active, edges


def _segment(a, b):
    dr = int(np.sign(b[0]-a[0]))
    dc = int(np.sign(b[1]-a[1]))
    length = abs(b[0]-a[0]) + abs(b[1]-a[1])
    return tuple(_point(a, (dr, dc), k) for k in range(length+1))


def _label_construction(clean, source, word, hub_mask, palette):
    """Labels come from the intended clean drawing, before any corruption.

    Synchronous set wavefronts, rather than the input solver's repair/BFS path.
    This is construction evidence, not an independently authored solver.
    """
    bg, hub, rail = palette[:3]
    out = np.full(clean.shape, bg, dtype=np.int8)
    out[hub_mask] = hub
    out[clean] = rail
    unseen, front, tick = _cells(clean), {source}, 0
    while front:
        unseen.difference_update(front)
        for p in front:
            out[p] = word[tick % len(word)]
        front = {q for p in front for step in CARDINAL
                 if (q := _point(p, step)) in unseen}
        tick += 1
    return out


def _diagonal_group(cells):
    ordered = sorted(cells)
    return (len(ordered) in (2, 3)
            and all(ordered[i+1][0]-ordered[i][0] == 1 for i in range(len(ordered)-1))
            and all(ordered[i+1][1]-ordered[i][1] == ordered[1][1]-ordered[0][1]
                    for i in range(len(ordered)-1))
            and abs(ordered[1][1]-ordered[0][1]) == 1)


def _damage_options(clean, points, hub_mask):
    step = (int(np.sign(points[-1][0]-points[0][0])),
            int(np.sign(points[-1][1]-points[0][1])))
    perpendicular = (step[1], step[0])
    for start in range(2, len(points)-3):
        gap = points[start:start+3]
        anchors = (points[start-1], points[start+3])
        if not all(_at(clean, p) for p in gap+anchors):
            continue
        if any(_at(clean, _point(p, perpendicular, side))
               for p in gap for side in (-1, 1)):
            continue
        for endpoint, toward in zip(anchors, (1, -1)):
            for side in (-1, 1):
                diagonal = (toward*step[0] + side*perpendicular[0],
                            toward*step[1] + side*perpendicular[1])
                spur = tuple(_point(endpoint, diagonal, k) for k in (1, 2))
                if all(_inside(clean.shape, p) and not _at(clean | hub_mask, p) for p in spur):
                    yield gap, spur


def _make_scene(rng, palette, word, *, loops, cuts, repairs, contrast, large):
    """Sample a graph, cut its topology, then corrupt distinct straight edges."""
    hub_h, hub_w = int(rng.choice((3, 5))), int(rng.choice((3, 5)))
    kinds = ["rectangle"] + (["clipped"] if (hub_h, hub_w) != (3, 3) else [])
    if hub_h == hub_w == 5:
        kinds.append("cross")
    kind = str(rng.choice(kinds))
    hub_left, tail = int(rng.integers(3)), int(rng.integers(2, 5))
    source_x = hub_left + hub_w
    rail_x = source_x + len(word) + tail
    available_cols = [c for c in range(2, 5) if rail_x + 6*(c-1) + 1 < 30]
    cols = max(available_cols) if large else int(rng.choice(available_cols))
    rows = int(rng.choice((3, 4) if large else (2, 3)))
    xs, ys = [rail_x], [int(rng.integers(2, 4))]
    for i in range(cols-1):
        allowance = 29 - xs[-1] - 6*(cols-i-2)
        xs.append(xs[-1] + int(rng.integers(6, min(9, allowance)+1)))
    for i in range(rows-1):
        allowance = 26 - ys[-1] - 6*(rows-i-2)
        ys.append(ys[-1] + int(rng.integers(6, min(9, allowance)+1)))
    height = min(30, ys[-1] + int(rng.integers(2, 4)) + 1)
    width = min(30, xs[-1] + int(rng.integers(1, 4)) + 1)
    source_row = int(rng.integers(rows))
    root = (source_row, 0)
    hub_top = ys[source_row] - hub_h//2
    hm = np.zeros((height, width), dtype=bool)
    hm[hub_top:hub_top+hub_h, hub_left:hub_left+hub_w] = _hub_shape(hub_h, hub_w, kind)
    source = (ys[source_row], source_x)
    seed_cells = tuple(_point(source, (0, 1), k) for k in range(len(word)))
    wm = np.zeros_like(hm)
    for p in seed_cells:
        wm[p] = True
    total = rows*cols
    target = total if loops or large else int(rng.integers(max(4, total-3), total+1))
    active, edges = _grow_graph(rng, rows, cols, root, target)
    alternatives = sorted({_edge(a, _point(a, s)) for a in active for s in CARDINAL
                           if _point(a, s) in active} - edges)
    if len(alternatives) < loops:
        raise ValueError("not enough independent cycles")
    for i in rng.permutation(len(alternatives))[:loops]:
        edges.add(alternatives[int(i)])
    positions = {a: (ys[a[0]], xs[a[1]]) for a in active}
    segments = {e: _segment(positions[e[0]], positions[e[1]]) for e in sorted(edges)}
    clean = np.zeros_like(hm)
    for points in segments.values():
        for p in points:
            clean[p] = True
    clean[ys[source_row], source_x:rail_x+1] = True
    # Small terminal stubs alter branch shape without introducing graph shortcuts.
    for _ in range(int(rng.integers(4))):
        node = sorted(active)[int(rng.integers(len(active)))]
        step = CARDINAL[int(rng.integers(4))]
        extension = tuple(_point(positions[node], step, k)
                          for k in range(1, int(rng.integers(1, 4))+1))
        if all(_inside(clean.shape, p) and not _at(clean | hm, p) for p in extension):
            neighbors = {q for p in extension for s in CARDINAL
                         if _at(clean | hm, q := _point(p, s))}
            if neighbors == {positions[node]}:
                for p in extension:
                    clean[p] = True
    degrees = {a: sum(a in e for e in edges) for a in active}
    leaf_edges = [e for e in sorted(edges)
                  if any(degrees[a] == 1 and a != root for a in e)]
    if len(leaf_edges) < cuts:
        raise ValueError("not enough pendant edges for intentional barriers")
    cut_edges, cut_records = set(), []
    for i in rng.permutation(len(leaf_edges))[:cuts]:
        e = leaf_edges[int(i)]
        points = segments[e]
        start = int(rng.integers(2, len(points)-3))
        gap = points[start:start+3]
        for p in gap:
            clean[p] = False
        cut_edges.add(e)
        cut_records.append((gap, points[start-1], points[start+3]))
    if _thick(clean):
        raise ValueError("rails must remain one cell thick")
    raw, selected = clean.copy(), []
    options = []
    for e, points in segments.items():
        if e not in cut_edges:
            options.extend(_damage_options(clean, points, hm))
    for index in rng.permutation(len(options)):
        gap, spur = options[int(index)]
        if any(not raw[p] for p in gap) or any(raw[p] for p in spur):
            continue
        footprint = set(gap+spur)
        if any(abs(p[0]-q[0]) <= 1 and abs(p[1]-q[1]) <= 1
               for p in footprint for old in selected for q in old[0]+old[1]):
            continue
        trial = raw.copy()
        for p in gap:
            trial[p] = False
        for p in spur:
            trial[p] = True
        try:
            recovered, records, _ = _clean_rails(trial, wm, hm)
        except ValueError:
            continue
        if np.array_equal(recovered, clean) and len(records) == len(selected)+1:
            raw = trial
            selected.append((gap, spur))
            if len(selected) == repairs:
                break
    if len(selected) != repairs:
        raise ValueError("could not place separated repair evidence")
    # A diagonal near an intentional break whose extrapolated anchor is empty.
    near_miss = False
    for gap, a, b in cut_records:
        step = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
        perp = (step[1], step[0])
        possibilities = []
        for anchor, toward in ((a, 1), (b, -1)):
            for side in (-1, 1):
                possibilities.append(tuple((anchor[0]+k*toward*step[0]+(k+1)*side*perp[0],
                                            anchor[1]+k*toward*step[1]+(k+1)*side*perp[1])
                                           for k in (1, 2)))
        for i in rng.permutation(len(possibilities)):
            points = possibilities[int(i)]
            if not all(_inside(raw.shape, p) and not _at(raw | hm, p) for p in points):
                continue
            trial = raw.copy()
            for p in points:
                trial[p] = True
            try:
                recovered, records, removed = _clean_rails(trial, wm, hm)
            except ValueError:
                continue
            if (np.array_equal(recovered, clean) and len(records) == repairs
                    and all(removed[p] for p in points)
                    and all(_diagonal_group(c) for c in _components(removed, True))):
                raw, near_miss = trial, True
                break
        if near_miss:
            break
    if contrast and not near_miss:
        raise ValueError("the contrast witness needs a nonattaching diagonal")
    for _ in range(int(rng.integers(1, 5))):
        for _ in range(35):
            p = (int(rng.integers(height)), int(rng.integers(width)))
            step = (int(rng.choice((-1, 1))), int(rng.choice((-1, 1))))
            points = tuple(_point(p, step, k) for k in range(int(rng.choice((2, 3)))))
            if not all(_inside(raw.shape, q) and not _at(raw | hm, q) for q in points):
                continue
            # Detached clutter is separated from every original drawing cell.
            if any(_at(raw | hm, (q[0]+dr, q[1]+dc))
                   for q in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                continue
            trial = raw.copy()
            for q in points:
                trial[q] = True
            try:
                recovered, records, removed = _clean_rails(trial, wm, hm)
            except ValueError:
                continue
            if (np.array_equal(recovered, clean) and len(records) == repairs
                    and all(_diagonal_group(c) for c in _components(removed, True))):
                raw = trial
                break
    bg, hub, rail = palette[:3]
    inp = np.full(clean.shape, bg, dtype=np.int8)
    inp[raw] = rail
    inp[hm] = hub
    for p, color in zip(seed_cells, word):
        inp[p] = color
    out = _label_construction(clean, source, word, hm, palette)
    if not np.array_equal(solve(inp), out):
        raise ValueError("scene geometry creates unintended reconstruction")
    return {"input": inp, "output": out}


def _period(word):
    return next(p for p in range(1, len(word)+1)
                if len(word) % p == 0 and all(v == word[i % p] for i, v in enumerate(word)))


def _features(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    prediction, a = _reconstruct(inp)
    assert np.array_equal(out, prediction), "incomplete or incorrect output"
    r, d, clean = a["roles"], a["distances"], a["clean"]
    assert not _thick(clean), "thick rail block"
    removed_groups = _components(a["removed"], True)
    assert all(_diagonal_group(c) for c in removed_groups), "non-diagonal deletion"
    assert len(a["repairs"]) > 0, "a generated scene must demonstrate repair"
    added = clean & ~a["raw"]
    assert int(added.sum()) == 3*len(a["repairs"])
    assert all(a["removed"][p] for rec in a["repairs"] for p in rec["spur"])
    assert np.array_equal(inp[r["hub_mask"]], out[r["hub_mask"]])
    assert np.array_equal(inp[r["word_mask"]], out[r["word_mask"]])
    untouched = ~clean & ~a["removed"]
    assert np.array_equal(inp[untouched], out[untouched]), "paint escaped the rails"
    assert np.all(out[clean & (d < 0)] == r["rail"]), "unreachable rails were colored"
    for rec in a["repairs"]:
        # Every sampled repair has a meaningful effect beyond its own gap.
        assert all(d[p] >= 0 for p in rec["gap"]), "unreachable cosmetic repair"
    reached = _cells(d >= 0)
    edge_count = sum(_point(p, s) in reached for p in reached for s in ((1, 0), (0, 1)))
    cycles = edge_count - len(reached) + 1
    degrees = [sum(_point(p, s) in reached for s in CARDINAL) for p in reached]
    no_b, ablated = _reconstruct(inp, repair=False)
    distal = clean & ~added & ~r["word_mask"]
    b_effect = int(np.count_nonzero((out != no_b) & distal))
    shared_reach = distal & (d >= 0) & (ablated["distances"] >= 0)
    phase_b_effect = int(np.count_nonzero((out != no_b) & shared_reach))
    reach_b_effect = int(np.count_nonzero(distal & (d >= 0) & (ablated["distances"] < 0)))
    manhattan = np.indices(inp.shape)
    md = np.abs(manhattan[0]-r["source"][0]) + np.abs(manhattan[1]-r["source"][1])
    word = np.asarray(r["word"])
    wrong_distance = word[md % len(word)]
    manhattan_effect = int(np.count_nonzero((out != wrong_distance) & (d >= 0)))
    reset_effect = 0
    for p in reached - set(r["word_cells"]):
        neighbors = [q for s in CARDINAL if (q := _point(p, s)) in reached]
        turn = len(neighbors) == 2 and neighbors[0][0] != neighbors[1][0] \
            and neighbors[0][1] != neighbors[1][1]
        if len(neighbors) >= 3 or turn:
            reset_effect += sum(d[q] == d[p]+1 and out[q] != word[0] for q in neighbors)
    unsupported = 0
    for step in ((1, 0), (0, 1)):
        for p in sorted(_cells(clean & ~r["word_mask"])):
            if (_at(clean, _point(p, step, 4))
                    and all(_inside(inp.shape, q := _point(p, step, k))
                            and inp[q] == out[q] == r["bg"] for k in (1, 2, 3))):
                unsupported += 1
    return dict(shape=inp.shape, direction=r["direction"], word=r["word"],
                length=len(word), period=_period(word), cycles=cycles,
                junctions=sum(x >= 3 for x in degrees), maximum_degree=max(degrees),
                repairs=len(a["repairs"]), repair_axes=tuple(sorted({x["axis"] for x in a["repairs"]})),
                erased=len(removed_groups), extra_noise=len(removed_groups)-len(a["repairs"]),
                unsupported=unsupported, unseeded=int(np.count_nonzero(clean & (d < 0))),
                unseeded_components=len(_components(clean & (d < 0))),
                reached=int(np.count_nonzero(d >= 0)), b_effect=b_effect,
                phase_b_effect=phase_b_effect, reach_b_effect=reach_b_effect,
                reset_effect=reset_effect, manhattan_effect=manhattan_effect,
                roles=(r["bg"], r["hub"], r["rail"]),
                hub_kind=r["hub_kind"], hub_size=r["hub_size"],
                distances=tuple(sorted(set(d[d >= 0].tolist()))))


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(rng, palette, pattern, config, turns=None):
    word = tuple(palette[3+i] for i in pattern)
    for _ in range(160):
        try:
            pair = _make_scene(rng, palette, word, **config)
            f = _features(pair)
            if f["b_effect"] == 0:
                continue
            if config["contrast"] and (f["reach_b_effect"] < 5 or f["unseeded"] < 3):
                continue
            if config["loops"] and (f["cycles"] < config["loops"]
                    or f["manhattan_effect"] == 0 or f["phase_b_effect"] == 0
                    or f["reset_effect"] == 0):
                continue
            rotation = int(rng.integers(4)) if turns is None else turns
            mirror = bool(rng.integers(2))
            pair = {k: np.ascontiguousarray(np.rot90(v, rotation)) for k, v in pair.items()}
            if mirror:
                pair = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in pair.items()}
            return pair
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("bounded graph sampler exhausted its attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    n, t = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("cycles", "repairs", "barriers", "word", "orientation")))
    palette_mode = str(rng.choice(("fixed", "swap", "varied", "varied")))
    base_palette = tuple(map(int, rng.permutation(10)))

    def palette():
        if palette_mode == "varied":
            return tuple(map(int, rng.permutation(10)))
        if palette_mode == "swap" and rng.integers(2):
            return (base_palette[2], base_palette[1], base_palette[0], *base_palette[3:])
        return base_palette

    def pattern(length, primitive=False):
        colors = int(rng.integers(1 if not primitive else 2, 5))
        p = tuple(map(int, rng.integers(colors, size=length)))
        if primitive and _period(p) != length:
            p = (0,)*(length-1)+(1,)
        return p

    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n), ("test", t)):
        for i in range(count):
            config = dict(loops=0, cuts=1, repairs=1, contrast=False, large=False)
            turns = None
            if split == "train" and i == 0:
                p = (0, 1)
                config["contrast"] = True
            elif split == "train" and i == 1:
                p = (0, 1, 0)
                config["loops"], config["cuts"] = 1, 0
            elif split == "train":
                p = pattern(int(rng.integers(1, 4)))
                # Constant words need no Manhattan-distance contrast.
                config["loops"] = 0
                config["cuts"] = int(rng.integers(2))
            else:
                p = pattern(int(rng.integers(2, 5)), primitive=True)
                config["loops"] = int(rng.integers(2)) if _period(p) > 2 else 0
                config["cuts"] = int(rng.integers(2))
                if regime == "cycles":
                    p = pattern(3, primitive=True)
                    config.update(loops=2, cuts=0, large=True)
                elif regime == "repairs":
                    config.update(repairs=int(rng.integers(2, 4)), large=True)
                elif regime == "barriers":
                    config.update(loops=0, cuts=2, repairs=2, large=True)
                elif regime == "word":
                    p = pattern(int(rng.choice((4, 5))), primitive=True)
                    config.update(loops=0)
            if regime == "orientation":
                turns = int(rng.choice((0, 2) if split == "train" else (1, 3)))
            pair = _make_pair(rng, palette(), p, config, turns=turns)
            if _key(pair["input"]) in seen:
                # Finite role-preserving injective recoloring resolves rare duplicates.
                colors = sorted(map(int, np.unique(pair["input"])))
                for replacement in itertools.permutations(range(10), len(colors)):
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[colors] = replacement
                    candidate = {k: mapping[v] for k, v in pair.items()}
                    if _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
                else:
                    raise RuntimeError("finite recoloring space exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, t = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            f = _features(pair)
            assert f["b_effect"] > 0, "repair must affect a distal color or reachability"
            assert np.any(inp != out)
            if split == "train":
                train.append(f)
    assert any(f["unsupported"] and f["unseeded"] and f["extra_noise"] for f in train), \
        "show a preserved barrier, uncolored distal rails, and deleted near-miss clutter"
    assert any(f["reach_b_effect"] >= 5 and f["unsupported"] and f["unseeded"] >= 3 for f in train), \
        "show a repair connecting a substantial branch beside an intentional barrier"
    assert any(f["cycles"] and f["manhattan_effect"] and f["phase_b_effect"] and f["reset_effect"]
               for f in train), "show shortest graph distance, loop shortcuts, and phase continuation"
    assert any(len(f["word"]) == 3 and f["word"][0] == f["word"][2] != f["word"][1]
               for f in train), "show ABA as a complete cyclic word"
    assert any(f["period"] >= 2 for f in train), "show nonconstant phase"


def self_test() -> dict:
    """A bounded 42-seed sweep with structural measurements and negative labels."""
    # Hand-constructed semantic regression, independent of the random sampler:
    # ABA at (6,3), a loop, a returning branch, and an unsupported side barrier.
    fixture = np.zeros((18, 24), dtype=np.int8)
    fixture[5:8, 0:3] = 8
    fixture[6, 3:6] = (1, 2, 1)
    fixture[6, 6:19] = 5
    fixture[6:15, 8] = fixture[6:15, 14] = 5
    fixture[14, 2:15] = 5
    fixture[6:17, 18] = 5
    fixture[6, 10:13] = 0
    fixture[7, 10] = fixture[8, 11] = 5
    fixture[9:12, 18] = 0
    fixture[9, 20] = fixture[10, 21] = 5  # projected anchor (8,19) is empty
    answer = solve(fixture)
    hand_expected = {(6, 6): 1, (6, 7): 2, (6, 10): 2, (6, 11): 1,
                     (6, 12): 1, (6, 15): 1, (10, 14): 1, (14, 14): 2,
                     (14, 2): 2, (12, 18): 5, (16, 18): 5,
                     (7, 10): 0, (8, 11): 0, (9, 20): 0, (10, 21): 0}
    assert all(answer[p] == color for p, color in hand_expected.items())
    assert np.all(answer[9:12, 18] == 0)
    assert np.all(answer[5:8, 0:3] == 8)
    assert _reconstruct(fixture, repair=False)[0][6, 15] == 2  # shortcut changes phase
    assert (abs(14-6)+abs(2-3)) % 3 == 0 and answer[14, 2] == 2  # not Manhattan
    puzzles = [generate_puzzle(s) for s in range(42)]
    records = [_features(p) for puzzle in puzzles for split in ("train", "test")
               for p in puzzle[split]]
    signatures = {tuple(_key(p["input"]) for split in ("train", "test") for p in pu[split])
                  for pu in puzzles}
    assert len(signatures) == 42
    assert len({f["shape"] for f in records}) >= 35
    assert {f["direction"] for f in records} == set(CARDINAL)
    assert {f["length"] for f in records} >= {1, 2, 3, 4, 5}
    assert {f["repairs"] for f in records} >= {1, 2, 3}
    assert max(f["cycles"] for f in records) >= 2
    assert max(f["junctions"] for f in records) >= 3
    assert max(f["unseeded"] for f in records) >= 5
    assert max(f["unseeded_components"] for f in records) >= 2
    assert len({f["reached"] for f in records}) >= 30
    assert len({f["hub_size"] for f in records}) >= 4
    assert {f["hub_kind"] for f in records} == {"rectangle", "clipped", "cross"}
    assert {axis for f in records for axis in f["repair_axes"]} == {"h", "v"}
    assert len({f["roles"] for f in records}) >= 30
    for role in range(3):
        assert {f["roles"][role] for f in records} == set(range(10))
    assert any(f["period"] == 1 for f in records)
    for pu in puzzles:
        validate_generated_puzzle(pu)
        fs = [_features(p) for s in ("train", "test") for p in pu[s]]
        assert len({(f["junctions"], f["cycles"], f["reached"], f["unseeded"]) for f in fs}) >= 3
    assert tuple(_key(p[k]) for s in ("train", "test") for p in puzzles[0][s]
                 for k in ("input", "output")) == tuple(
        _key(p[k]) for s in ("train", "test") for p in generate_puzzle(0)[s]
        for k in ("input", "output"))
    # Color relabeling and all eight rigid symmetries preserve semantics.
    pair = puzzles[0]["train"][0]
    mapping = np.array([8, 2, 5, 0, 9, 3, 7, 1, 6, 4], dtype=np.int8)
    for turns in range(4):
        for flip in (False, True):
            inp, out = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
            if flip:
                inp, out = inp[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(solve(mapping[inp]), mapping[out])
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            pu = generate_puzzle(314159, n, t)
            assert len(pu["train"]) == n and len(pu["test"]) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Omitting an entire repair, including its spur, must fail full validation.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    pair = damaged["train"][0]
    _, info = _reconstruct(pair["input"])
    rec, roles = info["repairs"][0], info["roles"]
    for p in rec["gap"] + rec["spur"]:
        pair["output"][p] = pair["input"][p]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("omitted repair accepted")
    # A's omission: constant paint cannot replace a nonconstant word's wave.
    witness = next(p for p in puzzles[0]["train"] if _features(p)["period"] > 1)
    _, info = _reconstruct(witness["input"])
    constant = witness["output"].copy()
    constant[info["distances"] >= 0] = info["roles"]["word"][0]
    assert np.any(constant != witness["output"])
    return {"seeds": 42, "pairs": len(records), "shapes": len({f["shape"] for f in records}),
            "role_assignments": len({f["roles"] for f in records}),
            "word_lengths": sorted({f["length"] for f in records}),
            "repair_counts": sorted({f["repairs"] for f in records}),
            "cycles": sorted({f["cycles"] for f in records}),
            "reachable_counts": len({f["reached"] for f in records})}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
