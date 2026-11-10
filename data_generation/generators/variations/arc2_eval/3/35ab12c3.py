"""Topology-aware sparse drawing completion and visible unit translations.

A unique noncrossing full cycle takes precedence over a unique full path.
Colors have local roles; only non-unit links produce companion strokes.
"""

import numpy as np
import hashlib


_OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _validate_count_value(name, value, low, high):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= int(value) <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}] or None")


def _resolve_counts(seed, rng, num_train, num_test):
    _validate_count_value("num_train", num_train, 2, 10)
    _validate_count_value("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sign(v):
    return int(v > 0) - int(v < 0)


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _raster_segment(a, b):
    assert _aligned(a, b)
    dr, dc = b[0] - a[0], b[1] - a[1]
    return [(a[0] + i * _sign(dr), a[1] + i * _sign(dc))
            for i in range(max(abs(dr), abs(dc)) + 1)]


def _candidate_edges(points):
    landmarks = set(points)
    return [(i, j) for i, a in enumerate(points) for j, b in enumerate(points)
            if i < j and _aligned(a, b)
            and not (set(_raster_segment(a, b)[1:-1]) & landmarks)]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    # Continuous intersection includes diagonal half-cell crossings.
    if (max(a[0], b[0]) < min(c[0], d[0])
            or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1])
            or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _infer_order(points):
    """Infer topology without a template, output, seed or private vertex order."""
    n = len(points)
    assert n <= 12, "excessive landmark count"
    edges = _candidate_edges(points)
    if not edges:
        return None
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    assert all(adjacency), "partly linked color group"
    conflicts = [set() for _ in edges]
    for k, (a, b) in enumerate(edges):
        for l, (c, d) in enumerate(edges[:k]):
            if not {a, b} & {c, d} and _intersects(points[a], points[b], points[c], points[d]):
                conflicts[k].add(l)
                conflicts[l].add(k)

    def search(closed):
        solutions = {}
        budget = [0]

        def visit(order, used, selected):
            budget[0] += 1
            assert budget[0] <= 30000, "topology search budget exceeded"
            if len(solutions) > 1:
                return
            if len(order) == n:
                chosen = selected
                if closed:
                    closing = [e for v, e in adjacency[order[-1]] if v == order[0]]
                    if not closing or conflicts[closing[0]] & set(selected):
                        return
                    chosen = selected + [closing[0]]
                solutions[tuple(sorted(chosen))] = order
                return
            for neighbor, edge in adjacency[order[-1]]:
                if neighbor not in used and not conflicts[edge] & set(selected):
                    visit(order + [neighbor], used | {neighbor}, selected + [edge])

        for start in ([0] if closed else range(n)):
            visit([start], {start}, [])
            if len(solutions) > 1:
                break
        return solutions

    cycles = search(True) if n >= 3 else {}
    assert len(cycles) <= 1, "competing closed contours"
    solutions = cycles or search(False)
    assert len(solutions) == 1, "no unique full path or contour"
    signature, order = next(iter(solutions.items()))
    selected = [edges[k] for k in signature]
    pixels = set().union(*(set(_raster_segment(points[i], points[j])) for i, j in selected))
    long_pixels, lengths, directions = set(), [], set()
    for i, j in selected:
        a, b = points[i], points[j]
        length = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        lengths.append(length)
        directions.add("horizontal" if a[0] == b[0] else "vertical" if a[1] == b[1] else "diagonal")
        if length > 1:
            long_pixels.update(_raster_segment(a, b))
    assert long_pixels, "already complete group"
    turns = []
    if cycles:
        turns = [_sign(_cross(points[order[i-1]], points[order[i]], points[order[(i+1) % n]])) for i in range(n)]
    return {"points": points, "edges": selected, "order": order,
            "kind": "cycle" if cycles else "path", "pixels": pixels,
            "long_pixels": long_pixels, "lengths": lengths, "directions": directions,
            "chords": len(edges) - len(selected), "concave": -1 in turns and 1 in turns}


def _analyze_input(inp):
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert max(counts) > inp.size // 2
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(inp == c)]
              for c in values if c != background}
    sources = []
    for color, points in groups.items():
        info = _infer_order(points)
        if info is not None:
            info["color"] = color
            sources.append(info)
    assert sources
    source_colors = {s["color"] for s in sources}
    source_pixels = set()
    for source in sources:
        assert not source_pixels & source["pixels"], "sources collide"
        source_pixels.update(source["pixels"])
        assert any(inp[p] == background for p in source["pixels"])
    markers, marked, inert = [], set(), []
    for color, points in groups.items():
        if color in source_colors:
            continue
        associations = [(i, v, p) for p in points for i, s in enumerate(sources)
                        for v in s["points"] if max(abs(p[0]-v[0]), abs(p[1]-v[1])) == 1]
        if associations:
            assert len(points) == 1 and len(associations) == 1, "ambiguous marker"
            i, v, p = associations[0]
            assert i not in marked
            marked.add(i)
            offset = (p[0] - v[0], p[1] - v[1])
            copy = {(r + offset[0], c + offset[1]) for r, c in sources[i]["long_pixels"]}
            assert p in copy, "marker anchored only to a unit link"
            assert all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in copy), "clipped copy"
            markers.append({"color": color, "point": p, "source_index": i,
                            "offset": offset, "copy_pixels": copy})
        else:
            assert not _candidate_edges(points)
            assert not set(points) & source_pixels
            inert.extend(points)
    expected = inp.copy()
    claimed = {}
    for marker in markers:
        for p in marker["copy_pixels"]:
            if inp[p] == background and p not in source_pixels:
                assert p not in claimed, "unresolved copy/copy overlap"
                claimed[p] = marker["color"]
                expected[p] = marker["color"]
    for source in sources:
        for p in source["pixels"]:
            if inp[p] == background:
                expected[p] = source["color"]
    for marker in markers:
        marker["visible"] = int(np.count_nonzero((expected == marker["color"]) & (inp == background)))
        marker["overlap"] = len(marker["copy_pixels"] & source_pixels)
        marker["new_overlap"] = sum(inp[p] == background for p in marker["copy_pixels"] & source_pixels)
    return expected, {"background": background, "sources": sources, "markers": markers,
                      "inert": inert, "palette": len(values)}


def _analyze_pair(inp, out):
    expected, info = _analyze_input(inp)
    assert np.array_equal(expected, out), "incorrect completion or copy"
    assert np.array_equal(inp[inp != info["background"]], out[inp != info["background"]])
    assert np.any(inp != out)
    return info


def _unit_copy_difference(inp, info):
    """Cells that would differ if already complete unit-only stubs were copied."""
    source_pixels = set().union(*(s["pixels"] for s in info["sources"]))
    extra = set()
    for marker in info["markers"]:
        source = info["sources"][marker["source_index"]]
        dr,dc = marker["offset"]
        for r,c in source["pixels"] - source["long_pixels"]:
            p = r+dr,c+dc
            if (0 <= p[0] < inp.shape[0] and 0 <= p[1] < inp.shape[1]
                    and inp[p] == info["background"] and p not in source_pixels
                    and p not in marker["copy_pixels"]):
                extra.add(p)
    return len(extra)


def _normalize_points(points):
    mr, mc = min(p[0] for p in points), min(p[1] for p in points)
    return [(r-mr, c-mc) for r, c in points]


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return result


def _ordered_edges(points, closed):
    return [(i, i+1) for i in range(len(points)-1)] + ([(len(points)-1, 0)] if closed else [])


def _polygon(rng, size, concave=False):
    if rng.random() < .22:
        points, _ = _walk(rng,size)
        if _aligned(points[-1],points[0]):
            return points, True
    h, w = (int(rng.integers(3, size+1)) for _ in range(2))
    mode = int(rng.integers(3,5)) if concave else int(rng.integers(5))
    if mode == 0:
        w = h
    cuts = [0, 0, 0, 0]
    if mode >= 2:
        cuts = [int(rng.integers(0, max(2, min(h, w)//2))) for _ in range(4)]
    a, b, c, d = cuts
    points = [(0,a),(0,w-b),(b,w),(h-c,w),(h,w-c),(h,d),(h-d,0),(a,0)]
    points = [p for i,p in enumerate(points) if p != points[i-1]]
    if mode in (3,4):
        options = [(i, _raster_segment(p, points[(i+1) % len(points)]))
                   for i,p in enumerate(points)
                   if max(abs(p[0]-points[(i+1)%len(points)][0]), abs(p[1]-points[(i+1)%len(points)][1])) >= 4]
        if options:
            i, line = options[int(rng.integers(len(options)))]
            j = int(rng.integers(1, len(line)-2))
            k = int(rng.integers(j+1, len(line)-1))
            sr, sc = _sign(line[-1][0]-line[0][0]), _sign(line[-1][1]-line[0][1])
            depth = int(rng.integers(1, max(2, min(h,w)//2)))
            p, q = line[j], line[k]
            points[i+1:i+1] = [p, (p[0]+sc*depth,p[1]-sr*depth),
                               (q[0]+sc*depth,q[1]-sr*depth), q]
    if mode == 1:
        points = [(r+c, r-c) for r,c in points]
    return points, True


def _walk(rng, size, segment=False):
    n = 2 if segment else int(rng.integers(3, 9))
    points = [(0, 0)]
    diagonal_only = rng.random() < .25
    choices = [x for x in _OFFSETS if not diagonal_only or x[0]*x[1]]
    for _ in range(n-1):
        dr, dc = choices[int(rng.integers(len(choices)))]
        length = int(rng.integers(2 if segment else 1, size+1))
        points.append((points[-1][0]+dr*length, points[-1][1]+dc*length))
    return points, False


def _make_shape(rng, size, family=None, chord=False):
    selected = family or str(rng.choice(["segment", "walk", "polygon"], p=[.28,.30,.42]))
    require_concave = selected == "polygon" and size >= 4 and rng.random() < .32
    for attempt in range(140):
        points, closed = _polygon(rng,size,require_concave) if selected == "polygon" else _walk(rng,size,selected == "segment")
        if len(points) != len(set(points)) or len(points) > 12:
            continue
        points = _normalize_points(points)
        if max(max(r for r,c in points),max(c for r,c in points)) > min(24,size*2):
            continue
        try:
            info = _infer_order(points)
            intended = {tuple(sorted(e)) for e in _ordered_edges(points,closed)}
            if info is None or set(info["edges"]) != intended or (chord and not info["chords"]):
                continue
            if require_concave and not info["concave"]:
                continue
            return points, closed
        except AssertionError:
            continue
    if selected == "polygon" or chord:
        return [(0,0),(0,size),(size,size),(size,0)], True
    return [(0,0),(0,size)], False


def _make_spec(rng, points, closed, offsets, marked=False, witness=False):
    edges = _ordered_edges(points,closed)
    source = set().union(*(set(_raster_segment(points[i],points[j])) for i,j in edges))
    long_pixels = set().union(*(set(_raster_segment(points[i],points[j])) for i,j in edges
                              if len(_raster_segment(points[i],points[j])) > 2))
    candidates = []
    if marked:
        for dr,dc in offsets:
            copy = {(r+dr,c+dc) for r,c in long_pixels}
            for v in points:
                marker = (v[0]+dr,v[1]+dc)
                if marker in source or marker not in copy:
                    continue
                if sum(max(abs(marker[0]-q[0]),abs(marker[1]-q[1])) == 1 for q in points) != 1:
                    continue
                if witness and (not copy & (source-set(points)) or len(copy-source-{marker}) < 2):
                    continue
                candidates.append((marker,copy,(dr,dc)))
        if not candidates:
            return None
        marker, copy, offset = candidates[int(rng.integers(len(candidates)))]
    else:
        marker,copy,offset = None,set(),None
    return {"vertices":points,"closed":closed,"source_pixels":source,
            "copy_pixels":copy,"marker":marker,"offset":offset}


def _shift_spec(spec, dr, dc):
    item = dict(spec)
    for key in ("vertices","source_pixels","copy_pixels"):
        values = [(r+dr,c+dc) for r,c in spec[key]]
        item[key] = values if key == "vertices" else set(values)
    if spec["marker"] is not None:
        item["marker"] = (spec["marker"][0]+dr,spec["marker"][1]+dc)
    return item


def _transform_spec(spec, code):
    item = dict(spec)
    for key in ("vertices","source_pixels","copy_pixels"):
        values = _transform(spec[key],code)
        item[key] = values if key == "vertices" else set(values)
    if spec["marker"] is not None:
        item["marker"] = _transform([spec["marker"]],code)[0]
        item["offset"] = _transform([spec["offset"]],code)[0]
    return item


def _footprint(spec):
    return spec["source_pixels"] | spec["copy_pixels"]


def _halo(pixels):
    return {(r+dr,c+dc) for r,c in pixels for dr in (-1,0,1) for dc in (-1,0,1)}


def _pack(rng, specs, nested):
    placed,blocked = [],set()
    for i,spec in enumerate(specs):
        footprint = _footprint(spec)
        r0,c0 = min(r for r,c in footprint),min(c for r,c in footprint)
        h,w = max(r for r,c in footprint)-r0+1,max(c for r,c in footprint)-c0+1
        if h > 30 or w > 30:
            return None
        options = [(r-r0,c-c0) for r in range(31-h) for c in range(31-w)]
        rng.shuffle(options)
        if nested and i == 1:
            outer = placed[0]["source_pixels"]
            lo_r,hi_r = min(r for r,c in outer)+2,max(r for r,c in outer)-2
            lo_c,hi_c = min(c for r,c in outer)+2,max(c for r,c in outer)-2
            options = [(r,c) for r,c in options if all(lo_r<=a+r<=hi_r and lo_c<=b+c<=hi_c for a,b in footprint)]
        for dr,dc in options:
            pixels = {(r+dr,c+dc) for r,c in footprint}
            if not pixels & blocked:
                placed.append(_shift_spec(spec,dr,dc))
                blocked.update(_halo(pixels))
                break
        else:
            return None
    return placed


def _fit_canvas(rng, objects):
    footprint = set().union(*(_footprint(s) for s in objects))
    r0,c0 = min(r for r,c in footprint),min(c for r,c in footprint)
    h,w = max(r for r,c in footprint)-r0+1,max(c for r,c in footprint)-c0+1
    padding = []
    for length in (h,w):
        extra = 0 if rng.random() < .23 else int(rng.integers(0,31-length))
        before = int(rng.integers(extra+1))
        padding.append((extra,before))
    objects = [_shift_spec(s,padding[0][1]-r0,padding[1][1]-c0) for s in objects]
    return h+padding[0][0],w+padding[1][0],objects


def _draw(rng, objects, background):
    h,w,objects = _fit_canvas(rng,objects)
    landmark_count = sum(len(s["vertices"])+(s["marker"] is not None) for s in objects)
    while h*w <= 2*landmark_count:
        if h <= w and h < 30:
            h += 1
        else:
            w += 1
    inp = np.full((h,w),background,dtype=np.int8)
    palette = [int(c) for c in rng.permutation(10) if c != background]
    for spec in objects:
        spec["color"] = palette.pop()
        for p in spec["vertices"]:
            inp[p] = spec["color"]
        if spec["marker"] is not None:
            spec["marker_color"] = palette.pop()
            inp[spec["marker"]] = spec["marker_color"]
    blocked = _halo(set().union(*(_footprint(s) for s in objects)))
    candidates = [(r,c) for r in range(h) for c in range(w) if (r,c) not in blocked]
    rng.shuffle(candidates)
    inert_colors = palette[:int(rng.integers(1,min(3,len(palette))+1))] if palette else []
    groups = {color:[] for color in inert_colors}
    count = min(int(rng.integers(7)), (inp.size-1)//2-landmark_count) if inert_colors else 0
    for point in candidates:
        if count == 0:
            break
        choices = [c for c,ps in groups.items() if not any(_aligned(point,p) for p in ps)]
        if choices:
            c = choices[int(rng.integers(len(choices)))]
            groups[c].append(point)
            inp[point] = c
            count -= 1
    out = inp.copy()
    for spec in objects:
        if spec["marker"] is not None:
            for p in spec["copy_pixels"]:
                if inp[p] == background:
                    out[p] = spec["marker_color"]
    for spec in objects:
        for p in spec["source_pixels"]:
            if inp[p] == background:
                out[p] = spec["color"]
    return {"input":inp,"output":out}


def _build_pair(rng, background, offsets, size, count, witness=False, plain=False, unit_demo=False):
    for attempt in range(90):
        nested = not unit_demo and count >= 2 and rng.random() < .22
        specs = []
        for i in range(count):
            if unit_demo and i == 0:
                depth = int(rng.integers(2,6))
                length = depth+int(rng.integers(2,5))
                points,closed = [(0,0),(0,1),(0,length),(depth,length)],False
            elif nested and i == 0:
                h,w = (int(rng.integers(13,25)) for _ in range(2))
                if witness:
                    w = h
                points,closed = [(0,0),(0,w),(h,w),(h,0)],True
            else:
                local_size = min(size,5) if count >= 3 or nested else size
                family = "polygon" if witness and i == 0 else "walk" if plain and i == 0 else None
                points,closed = _make_shape(rng,local_size,family,witness and i == 0)
            marked = True if unit_demo and i == 0 else False if plain else (witness and i == 0) or rng.random() < .57
            code = int(rng.integers(8))
            # Find local offsets whose full geometric transform is permitted.
            local_offsets = [d for d in _OFFSETS if _transform([d],code)[0] in offsets]
            spec = _make_spec(rng,points,closed,local_offsets,marked,witness and i == 0)
            if spec is None:
                break
            specs.append(_transform_spec(spec,code))
        if len(specs) != count:
            continue
        placed = _pack(rng,specs,nested)
        if placed is None:
            continue
        pair = _draw(rng,placed,background)
        try:
            info = _analyze_pair(pair["input"],pair["output"])
            if witness:
                assert any(s["chords"] for s in info["sources"])
                assert any(m["visible"] >= 2 and m["new_overlap"] for m in info["markers"])
            if unit_demo:
                assert _unit_copy_difference(pair["input"],info)
            return pair
        except AssertionError:
            continue
    # Finite, spacious fallback; no legal request depends on lucky placement.
    specs = []
    for i in range(count):
        if unit_demo and i == 0:
            points,closed = [(0,0),(0,1),(0,5),(2,5)],False
            options = [d for d in offsets if d[0]]
        else:
            points = [(0,0),(0,5)] if plain and i == 0 else [(0,0),(0,5),(5,5),(5,0)]
            closed,options = len(points) > 2,offsets
        spec = _make_spec(rng,points,closed,options,(witness or unit_demo) and i == 0,witness and i == 0)
        assert spec is not None
        specs.append(_shift_spec(spec,2+14*(i//2),2+14*(i%2)))
    pair = _draw(rng,specs,background)
    _analyze_pair(pair["input"],pair["output"])
    return pair


def _grid_fingerprint(grid):
    return grid.shape, np.asarray(grid,dtype=np.int8).tobytes()


def _pair_digest(pair):
    digest = hashlib.sha256()
    for key in ("input","output"):
        grid = pair[key]
        digest.update(bytes(grid.shape))
        digest.update(np.asarray(grid,dtype=np.int8).tobytes())
    return digest.hexdigest()


def _inside(point, polygon):
    r,c = point
    inside = False
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        if (a[0] > r) != (b[0] > r):
            crossing = a[1] + (r-a[0])*(b[1]-a[1])/(b[0]-a[0])
            if c < crossing:
                inside = not inside
    return inside


def measure_pair(pair):
    """Review features reconstructed from returned grids, never sampled metadata."""
    info = _analyze_pair(pair["input"],pair["output"])
    sources,markers = info["sources"],info["markers"]
    h,w = pair["input"].shape
    nesting = 0
    for i,outer in enumerate(sources):
        if outer["kind"] != "cycle":
            continue
        polygon = [outer["points"][j] for j in outer["order"]]
        for j,inner in enumerate(sources):
            if i != j and all(_inside(p,polygon) for p in inner["pixels"]):
                nesting += 1
    touches = lambda pixels: any(r in (0,h-1) or c in (0,w-1) for r,c in pixels)
    rejected = 0
    for s in sources:
        all_pixels = set().union(*(set(_raster_segment(s["points"][i],s["points"][j]))
                                  for i,j in _candidate_edges(s["points"])))
        rejected += len(all_pixels-s["pixels"])
    shapes = []
    for s in sources:
        points = _normalize_points(s["points"])
        shapes.append((s["kind"],tuple(sorted(points)),tuple(sorted(s["edges"]))))
    return {
        "shape":[h,w], "background":info["background"], "palette":info["palette"],
        "sources":len(sources), "markers":len(markers), "inert":len(info["inert"]),
        "inert_singletons":sum(np.count_nonzero(pair["input"] == c) == 1
                                for c in np.unique(pair["input"])
                                if any(pair["input"][p] == c for p in info["inert"])),
        "vertices":[len(s["points"]) for s in sources],
        "lengths":[x for s in sources for x in s["lengths"]],
        "cycles":sum(s["kind"] == "cycle" for s in sources),
        "segments":sum(len(s["points"]) == 2 for s in sources),
        "segment_directions":[next(iter(s["directions"])) for s in sources if len(s["points"]) == 2],
        "squares":sum(s["kind"] == "cycle" and len(s["points"]) == 4
                      and s["directions"] == {"horizontal","vertical"}
                      and len(set(s["lengths"])) == 1 for s in sources),
        "diamonds":sum(s["kind"] == "cycle" and len(s["points"]) == 4
                       and s["directions"] == {"diagonal"} for s in sources),
        "bent_paths":sum(s["kind"] == "path" and len(s["points"]) > 2 for s in sources),
        "concave":sum(s["concave"] for s in sources),
        "unit_links":sum(s["lengths"].count(1) for s in sources),
        "chords":sum(s["chords"] for s in sources), "rejected_cells":rejected,
        "diagonal_only":sum(s["directions"] == {"diagonal"} for s in sources),
        "mixed_directions":sum("diagonal" in s["directions"] and len(s["directions"]) > 1 for s in sources),
        "offsets":[list(m["offset"]) for m in markers],
        "visible_copies":sum(m["visible"] > 0 for m in markers),
        "hidden_copies":sum(m["visible"] == 0 for m in markers),
        "partial_copies":sum(m["visible"] > 0 and m["overlap"] > 0 for m in markers),
        "clear_copies":sum(m["overlap"] == 0 for m in markers),
        "overlap_evidence":sum(int(m["new_overlap"]) for m in markers if m["visible"] > 0),
        "copied_kinds":[sources[m["source_index"]]["kind"] for m in markers],
        "copied_bends":sum(len(sources[m["source_index"]]["points"]) > 2 for m in markers),
        "unit_copy_difference":_unit_copy_difference(pair["input"],info),
        "nested":nesting,
        "source_boundary":any(touches(s["pixels"]) for s in sources),
        "copy_boundary":any(touches(m["copy_pixels"]) for m in markers),
        "geometry":repr(sorted(shapes)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen,infos = set(), {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype,np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert pair["input"].shape == pair["output"].shape
            fingerprint = _grid_fingerprint(pair["input"])
            assert fingerprint not in seen
            seen.add(fingerprint)
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            infos[split].append(_analyze_pair(pair["input"],pair["output"]))
    training = infos["train"]
    # Observable demonstrations establish the fixed convention even at (2,1).
    assert any(s["chords"] for i in training for s in i["sources"])
    assert any(s["kind"] == "path" for i in training for s in i["sources"])
    assert any(len(i["markers"]) < len(i["sources"]) for i in training)
    assert any(m["visible"] >= 2 and m["new_overlap"] > 0 for i in training for m in i["markers"])
    # These are pair properties, and must not collapse to puzzle constants.
    assert len({i["background"] for i in training}) >= 2
    assert len({len(i["sources"]) for i in training}) >= 2
    if any(_unit_copy_difference(p["input"],i) for p,i in zip(puzzle["test"],infos["test"])):
        assert any(_unit_copy_difference(p["input"],i) for p,i in zip(puzzle["train"],training)), "unit-only copy omission not demonstrated"
    # No numeric color, size, or offset is a hidden convention. Both regimes use
    # exactly the same topology and translation inference above, with no clipping.


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt,nq = _resolve_counts(seed,rng,num_train,num_test)
    # Presentation choices are sampled once; the transformation never varies.
    held = _OFFSETS[int(rng.integers(8))]
    offset_regime = int(rng.integers(3))  # familiar, held-out, unrestricted
    training_offsets = tuple(d for d in _OFFSETS if offset_regime != 1 or d != held)
    size_shift = int(rng.integers(-1,2))
    base_size = int(rng.integers(4,10))
    demo_indices = [int(x) for x in rng.choice(nt,2,replace=False)]
    witness_count = int(rng.integers(1,5))
    plain_count = int(rng.choice([c for c in range(1,5) if c != witness_count]))
    backgrounds = [int(x) for x in rng.integers(10,size=nt+nq)]
    backgrounds[demo_indices[1]] = (backgrounds[demo_indices[0]]+int(rng.integers(1,10))) % 10
    puzzle,seen = {"train":[],"test":[]},set()
    familiar = set()
    for regime,split,count in (("source_train","train",nt),("target_test","test",nq)):
        for index in range(count):
            witness = regime == "source_train" and index == demo_indices[0]
            plain = regime == "source_train" and index == demo_indices[1]
            construction_count = witness_count if witness else plain_count if plain else int(rng.integers(1,5))
            offsets = training_offsets if regime == "source_train" else (
                tuple(sorted(familiar)) if offset_regime == 0 else (held,) if offset_regime == 1 else _OFFSETS)
            size = int(np.clip(base_size+int(rng.integers(-2,3))+(size_shift*2 if regime == "target_test" else 0),3,12))
            pair_rng = np.random.default_rng(int(rng.integers(0,2**63-1)))
            background = backgrounds[index if regime == "source_train" else nt+index]
            for retry in range(30):
                pair = _build_pair(pair_rng,background,offsets,size,construction_count,witness,plain)
                fingerprint = _grid_fingerprint(pair["input"])
                if fingerprint not in seen and _pair_digest(pair) not in _SOURCE_DIGESTS:
                    break
            else:
                raise AssertionError("distinct-pair construction exhausted")
            seen.add(fingerprint)
            puzzle[split].append(pair)
            if split == "train":
                familiar.update(m["offset"] for m in _analyze_pair(pair["input"],pair["output"])["markers"])
    # A unit-only endpoint can distinguish copying the missing long links from
    # copying the entire drawing. Supply observable evidence if a query needs it.
    unit_diff = lambda p: _unit_copy_difference(p["input"],_analyze_pair(p["input"],p["output"]))
    if any(unit_diff(p) for p in puzzle["test"]) and not any(unit_diff(p) for p in puzzle["train"]):
        index = demo_indices[1]
        unit_count = int(rng.choice([c for c in (2,3,4) if c != witness_count]))
        replacement = _build_pair(rng,backgrounds[index],training_offsets,base_size,unit_count,plain=True,unit_demo=True)
        assert _grid_fingerprint(replacement["input"]) not in seen
        puzzle["train"][index] = replacement
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k],y[k]) for x,y in zip(a[s],b[s]) for k in ("input","output"))
        for s in ("train","test"))


def self_test() -> None:
    _source_regressions()
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    features = {s:[] for s in ("train","test")}
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        for split in features:
            features[split].extend(measure_pair(p) for p in puzzle[split])
    for split,rows in features.items():
        assert {x["sources"] for x in rows} == {1,2,3,4}
        assert {x["background"] for x in rows} == set(range(10))
        assert {tuple(d) for x in rows for d in x["offsets"]} == set(_OFFSETS)
        assert len({tuple(x["shape"]) for x in rows}) >= 30
        assert len({x["geometry"] for x in rows}) >= 50
        assert any(x["markers"] == 0 for x in rows)
        assert any(x["visible_copies"] >= 2 for x in rows)
        assert {d for x in rows for d in x["segment_directions"]} == {"horizontal","vertical","diagonal"}
        for metric in ("concave","unit_links","chords","diagonal_only","mixed_directions",
                       "nested","source_boundary","copy_boundary","copied_bends","hidden_copies",
                       "clear_copies","partial_copies","inert_singletons","squares","diamonds","unit_copy_difference"):
            assert sum(x[metric] for x in rows) > 0, (split,metric)
        assert {x["inert"] for x in rows} == set(range(7))
        assert min(min(x["shape"]) for x in rows) <= 6
        assert max(max(x["shape"]) for x in rows) == 30
        assert max(v for x in rows for v in x["vertices"]) >= 8
        assert any(x["concave"] and x["markers"] and x["sources"] >= 2 for x in rows)
    familiar,novel,mixed,bigger,smaller = 0,0,0,0,0
    for puzzle in automatic:
        train = [measure_pair(p) for p in puzzle["train"]]
        test = [measure_pair(p) for p in puzzle["test"]]
        seen_offsets = {tuple(d) for x in train for d in x["offsets"]}
        asked_offsets = {tuple(d) for x in test for d in x["offsets"]}
        familiar += bool(asked_offsets) and asked_offsets <= seen_offsets
        novel += bool(asked_offsets-seen_offsets)
        mixed += any(len({tuple(d) for d in x["offsets"]}) >= 2 for x in train+test)
        mean_train = np.mean([np.prod(x["shape"]) for x in train])
        mean_test = np.mean([np.prod(x["shape"]) for x in test])
        bigger += mean_test > mean_train
        smaller += mean_test < mean_train
    assert min(familiar,novel,mixed,bigger,smaller) >= 5
    a = generate_puzzle(314159,7,3)
    assert _puzzles_equal(a,generate_puzzle(314159,7,3))
    assert not _puzzles_equal(a,generate_puzzle(314160,7,3))
    cases = [(2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)]
    for i,(nt,nq) in enumerate(cases):
        p = generate_puzzle(9000+i,nt,nq)
        assert nt is None or len(p["train"]) == nt
        assert nq is None or len(p["test"]) == nq
        validate_generated_puzzle(p)
    for nt,nq in [(1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1),(2.0,1)]:
        try:
            generate_puzzle(2,nt,nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Missing and recolored completion cells must fail validation.
    original = automatic[0]
    for mutation in range(2):
        p = {s:[{k:v.copy() for k,v in pair.items()} for pair in original[s]] for s in original}
        pair = p["test"][0]
        location = tuple(np.argwhere(pair["input"] != pair["output"])[0])
        pair["output"][location] = pair["input"][location] if mutation == 0 else (int(pair["output"][location])+1)%10
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            continue
        raise AssertionError("wrong completion accepted")


# Exact source contents, not forbidden canvas dimensions.
_SOURCE_DIGESTS = frozenset({'183fb89121499a290a303924befe003dc56b71a39510a28cbfae22cc128b51d6', 'a9f00b6612db7009725b824f9c943778990e609de52cf2486bafb110f2c05a9f', '9a01323fd41f67ba117baacb81427e0e5917fd192f10efd225983f5b0da362b1', '850541f669d894bbe22930f6d383f04c9829983ae0b9f338e533ab7973ff7218'})

_SOURCE_FIXTURES = (('00000000000000000000/00000000000000000000/00000006000000000000/00000000000000000000/00000000000000000000/00070000000000000000/00060000000000018008/00000330000000000000/00000000000006000000/00000000000000000000/00300000030000000000/00000000000000000000/00000000000000000008/00000330000000000000/00000000000000008800/00000000000000000000/00000000000000000000/00000400000040000000/00000000000000000000/00000000000000000000', '00000000000000000000/00000007000000000000/00000076700000000000/00000760670000000000/00007600067000000000/00076000006700000000/00060000000670018888/00000330000067018018/00003003000006018018/00030000300000018018/00300000030000018018/00030000300000018018/00003003000000018018/00000330000000018180/00000000000000018800/00000000000000000000/00000000000000000000/00000444444440000000/00000000000000000000/00000000000000000000'), ('000000000000000000/000000000000000000/000000010000000000/000000008000000000/000000000040000000/000000000006060000/000008000000000000/000000000000000000/000000080000000060/000000000000000000/000000000000000000/000000000000000000/003000000006000000/000000000000000000/000003000000000000/000000000030000000/000000000000000000/000000000000000000', '000000000000000000/000000000000000000/000000010000000000/000000108000000000/000001080044400000/000010800046660000/000008000046006000/000000800046000600/000000080046000060/000000000046000000/000000000046000000/000000000046000000/003000000006000000/000000000000000000/000003000000000000/000000000030000000/000000000000000000/000000000000000000'), ('000000000000000/080000000000000/001000000000010/000000000000000/000070000070000/000000000007000/000000420200000/000000000000000/000000000000000/000000000000000/000070700000000/000000000000000/001000000000010/000000000000000/000000000000000', '000000000000000/088888888888800/081111111111110/081000000000810/081077777770810/081070000007810/081070422270810/081070000700810/081070007000810/081070070000810/081077700000810/081888888888810/001111111111110/000000000000000/000000000000000'), ('000000000000003000000/000000000000000000000/000600710000000000000/004000000000000003000/000000000000000000000/000000000300000000000/000000000030030000000/000000000000000000000/000000000000000303000/000000000000000000000/000000000000000000000/000000000000000000000/000000000000000000000/000000000080000000000/000000000000000000000/000000000000000080000/000000000000000000000/000600010000080000000/000000000000000000000/000000000000000000000/000000000000000000000', '000000000000003000000/000000000000030300000/000600710000300030000/004600710003000003000/004600710030000003000/004600710300000003000/004600710033330003000/004600710000003003000/004600710000000333000/004600710000000000000/004600710000000000000/004600710000000000000/004600710000000000000/004600710080000000000/004600710000000000000/004600710000000080000/004600710000000000000/004600710000080000000/004000000000000000000/000000000000000000000/000000000000000000000'))


def _source_regressions():
    for index, (inp_rows, out_rows) in enumerate(_SOURCE_FIXTURES):
        inp = np.array([[int(c) for c in row] for row in inp_rows.split("/")], dtype=np.int8)
        out = np.array([[int(c) for c in row] for row in out_rows.split("/")], dtype=np.int8)
        info = _analyze_pair(inp, out)
        for turns in range(4):
            for reflect in (False,True):
                a,b = np.rot90(inp,turns),np.rot90(out,turns)
                if reflect:
                    a,b = np.fliplr(a),np.fliplr(b)
                _analyze_pair(a,b)
        for source in info["sources"]:
            assert _infer_order(list(reversed(source["points"])))["pixels"] == source["pixels"]
        if index == 0:
            green = next(s for s in info["sources"] if s["color"] == 3)
            assert green["chords"] == 3 and green["lengths"].count(1) == 2
            all_pixels = set().union(*(set(_raster_segment(green["points"][i],green["points"][j])) for i,j in _candidate_edges(green["points"])))
            assert len(all_pixels-green["pixels"]) == 14
        if index == 2:
            orange = next(s for s in info["sources"] if s["color"] == 7)
            assert orange["chords"] >= 1 and 1 in orange["lengths"]
            assert len(set(_raster_segment((4,10),(10,4)))-orange["pixels"]) == 5
        if index == 3:
            assert any(s["concave"] and len(s["points"]) == 7 for s in info["sources"])
