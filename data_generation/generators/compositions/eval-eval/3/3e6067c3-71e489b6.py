"""Itinerary-controlled circuit audit: colored, ordered halos on a routed scaffold.

Only standard Python and NumPy are used. Solving decodes visible geometry and the
framed key; it never reads generation metadata, training pairs, or files.
"""
from __future__ import annotations

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int64, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                point = (y + dy, x + dx)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    group.append(point)
        groups.append(group)
    return groups


def _dilate(mask):
    h, w = mask.shape
    pad = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            result |= pad[dy:dy+h, dx:dx+w]
    return result


def _band(a, b, size, patch):
    m = (size - patch) // 2
    if a[0] == b[0] and abs(a[1] - b[1]) > size:
        return (a[0]+m, a[0]+m+patch,
                min(a[1], b[1])+size, max(a[1], b[1]))
    if a[1] == b[1] and abs(a[0] - b[0]) > size:
        return (min(a[0], b[0])+size, max(a[0], b[0]),
                a[1]+m, a[1]+m+patch)
    return None


def _slice(box):
    y, v, x, u = box
    return slice(y, v), slice(x, u)


def _graph(positions, size, patch, shape):
    stations = np.zeros(shape, dtype=bool)
    for y, x in positions:
        stations[y:y+size, x:x+size] = True
    graph = [[] for _ in positions]
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            if i == j:
                continue
            box = _band(a, b, size, patch)
            # Corruption pixels play no part in deciding geometric adjacency.
            if box is not None and not stations[_slice(box)].any():
                graph[i].append(j)
    return graph, stations


def _routes(graph, labels, key, limit=2):
    choices = [[i for i, c in enumerate(labels) if c == k] for k in key]
    answers = []

    def visit(path, used):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in choices[len(path)]:
            if i not in used and (not path or i in graph[path[-1]]):
                visit(path + [i], used | {i})

    visit([], set())
    return answers


def _decode(value):
    a = _grid(value)
    h, w = a.shape
    bg = int(a[0, 0])
    if not all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("input must have a uniform background frame")
    occupied = np.argwhere(a != bg)
    if not len(occupied):
        raise ValueError("missing key")
    bottom = int(occupied[:, 0].max())
    xs = np.flatnonzero(a[bottom] != bg)
    if bottom < 3 or not len(xs):
        raise ValueError("missing framed key")
    left, right = int(xs[0]), int(xs[-1])
    carrier = int(a[bottom, left])
    top = bottom - 2
    width = right - left + 1
    if (width < 7 or width % 2 != 1 or width > 25
            or not np.all(a[top, left:right+1] == carrier)
            or not np.all(a[bottom, left:right+1] == carrier)
            or a[top+1, left] != carrier or a[top+1, right] != carrier
            or not np.all(a[top+1, left+2:right:2] == bg)
            or not np.all(a[top-1] == bg)):
        raise ValueError("invalid framed key")
    key = tuple(map(int, a[top+1, left+1:right:2]))
    if any(c in (bg, carrier) for c in key) or len(set(key)) < 2:
        raise ValueError("key tokens must be label colors")
    key_mask = np.zeros_like(a, dtype=bool)
    key_mask[top:bottom+1, left:right+1] = True
    if np.any((a[top:] != bg) & ~key_mask[top:]):
        raise ValueError("unexpected cells beside or below key")

    label_mask = (a != bg) & (a != carrier) & ~key_mask
    stations, geometry = [], None
    for group in _components(label_mask):
        pts = np.array(group)
        y, x = map(int, pts.min(axis=0))
        v, u = map(int, pts.max(axis=0)+1)
        p = v-y
        color = int(a[y, x])
        if u-x != p or len(group) != p*p or not np.all(a[y:v, x:u] == color):
            raise ValueError("label patches must be intact squares")
        # Expand the intact carrier square until the first failed annulus.
        margin = 0
        for m in range(1, 15):
            if y-m < 1 or x-m < 1 or v+m > top-1 or u+m > w-1:
                break
            block = a[y-m:v+m, x-m:u+m]
            annulus = np.ones(block.shape, dtype=bool)
            annulus[m:m+p, m:m+p] = False
            if not np.all(block[annulus] == carrier):
                break
            margin = m
        size = p + 2*margin
        if not margin or not 3 <= size <= 7:
            raise ValueError("station has no readable carrier square")
        if geometry is not None and geometry != (size, p):
            raise ValueError("station geometry must be common within an input")
        geometry = size, p
        stations.append((y-margin, x-margin, color))
    stations.sort()
    if geometry is None or not 3 <= len(stations) <= 14:
        raise ValueError("expected 3..14 stations")
    size, patch = geometry
    positions = [s[:2] for s in stations]
    graph, station_mask = _graph(positions, size, patch, a.shape)
    if int(station_mask.sum()) != len(stations)*size*size:
        raise ValueError("stations overlap")
    solutions = _routes(graph, [s[2] for s in stations], key)
    if len(solutions) != 1:
        raise ValueError("key must have one distinct-station geometric route")
    route = solutions[0]
    owner = np.full(a.shape, -1, dtype=int)
    load = np.zeros_like(owner)
    bands = []
    for step, (i, j) in enumerate(zip(route, route[1:])):
        box = _band(positions[i], positions[j], size, patch)
        bands.append(box)
        region = _slice(box)
        owner[region] = step  # The last link owns shared corridor cells.
        load[region] += 1
    scaffold = station_mask | (owner >= 0)
    holes = (owner >= 0) & (a == bg)
    extras = (a == carrier) & ~scaffold & ~key_mask
    if np.any((a != bg) & ~scaffold & ~key_mask & ~extras):
        raise ValueError("unrecognized scene pixels")
    return dict(a=a, bg=bg, carrier=carrier, key=key, key_top=top,
                key_left=left, key_right=right, key_mask=key_mask,
                stations=stations, size=size, patch=patch, route=route,
                station_mask=station_mask, owner=owner, load=load,
                scaffold=scaffold, bands=bands, holes=holes, extras=extras)


def _prediction(info, *, uniform=False, targets=False, earliest=False, first_owner=False):
    a = info["a"].copy()
    a[info["extras"]] = info["bg"]
    holes, owner = info["holes"], info["owner"]
    if first_owner:
        owner = np.full_like(owner, -1)
        for step in reversed(range(len(info["bands"]))):
            owner[_slice(info["bands"][step])] = step
    order = range(len(info["bands"]))
    if earliest:
        order = reversed(tuple(order))
    for step in order:
        selected = holes & (owner == step)
        halo = _dilate(selected) & ~holes
        color = info["key"][0] if uniform else info["key"][step + int(targets)]
        a[halo] = color
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the itinerary scaffold, erase extras, and paint ordered hole halos."""
    return _prediction(_decode(input_grid))


def _crosses(a, b, c, d):
    if a[0] == b[0] and c[1] == d[1]:
        return (min(a[1], b[1]) < c[1] < max(a[1], b[1])
                and min(c[0], d[0]) < a[0] < max(c[0], d[0]))
    if a[1] == b[1] and c[0] == d[0]:
        return _crosses(c, d, a, b)
    return False


def _cross_count(positions, path):
    edges = [(positions[i], positions[j]) for i, j in zip(path, path[1:])]
    return sum(_crosses(*a, *b) for i, a in enumerate(edges) for b in edges[i+1:])


def _lattice_graph(cells):
    graph = [[] for _ in cells]
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            if i == j:
                continue
            if a[0] == b[0]:
                clear = not any(c[0] == a[0] and min(a[1], b[1]) < c[1] < max(a[1], b[1]) for c in cells)
            elif a[1] == b[1]:
                clear = not any(c[1] == a[1] and min(a[0], b[0]) < c[0] < max(a[0], b[0]) for c in cells)
            else:
                continue
            if clear:
                graph[i].append(j)
    return graph


def _find_path(rng, cells, graph, length, crossing):
    budget = 2400

    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            count = _cross_count(cells, path)
            return path if (count > 0 if crossing else count == 0) else None
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            if j not in path:
                found = visit(path + [j])
                if found is not None:
                    return found
        return None

    for start in rng.permutation(len(cells)):
        found = visit([int(start)])
        if found is not None:
            return found
        if budget < 0:
            break
    return None


def _labels(rng, graph, path, palette, repeated):
    length, n = len(path), len(graph)
    for _ in range(90):
        k = min(8, length) if not repeated else min(int(rng.integers(3, 7)), length-1)
        colors = list(map(int, rng.permutation(palette)[:k]))
        key = colors + [int(rng.choice(colors)) for _ in range(length-k)]
        key = list(map(int, rng.permutation(key)))
        if repeated:
            key[int(rng.integers(1, length))] = key[0]
        labels = [int(rng.choice(palette)) for _ in range(n)]
        for i, c in zip(path, key):
            labels[i] = c
        answers = _routes(graph, labels, key)
        if len(answers) == 1 and answers[0] == tuple(path):
            return labels, key
    return None


def _axis(rng, count, size, limit):
    gaps = [2]*(count-1)
    spare = limit - count*size - sum(gaps)
    for j in rng.permutation(len(gaps)):
        extra = int(rng.integers(0, min(6, spare)+1))
        gaps[int(j)] += extra
        spare -= extra
    result = [0]
    for gap in gaps:
        result.append(result[-1] + size + gap)
    return result


def _construct_scene(rng, cells, path, labels, key, size, patch, roles):
    rows, cols = max(r for r, c in cells)+1, max(c for r, c in cells)+1
    yy, xx = _axis(rng, rows, size, 24), _axis(rng, cols, size, 28)
    bh, bw = yy[-1]+size, xx[-1]+size
    kw = 2*len(key)+1
    minimum_h, minimum_w = bh+6, max(bw+2, kw+2)
    compact = rng.random() < .5
    h = int(rng.integers(minimum_h, min(30, minimum_h+2)+1 if compact else 31))
    w = int(rng.integers(minimum_w, min(30, minimum_w+2)+1 if compact else 31))
    top = int(rng.integers(1, h-bh-4))
    left = int(rng.integers(1, w-bw))
    positions = [(yy[r]+top, xx[c]+left) for r, c in cells]
    bg, carrier = roles
    clean = np.full((h, w), bg, dtype=int)
    station_mask = np.zeros((h, w), dtype=bool)
    margin = (size-patch)//2
    for (y, x), color in zip(positions, labels):
        clean[y:y+size, x:x+size] = carrier
        clean[y+margin:y+margin+patch, x+margin:x+margin+patch] = color
        station_mask[y:y+size, x:x+size] = True
    bands, owner = [], np.full((h, w), -1, dtype=int)
    load = np.zeros((h, w), dtype=int)
    for step, (i, j) in enumerate(zip(path, path[1:])):
        box = _band(positions[i], positions[j], size, patch)
        bands.append(box)
        clean[_slice(box)] = carrier
        owner[_slice(box)] = step
        load[_slice(box)] += 1
    ky = int(rng.integers(top+bh+1, h-3))
    kx = int(rng.integers(1, w-kw))
    clean[ky:ky+3, kx:kx+kw] = carrier
    clean[ky+1, kx+1:kx+kw-1] = bg
    clean[ky+1, kx+1:kx+kw-1:2] = key
    return dict(clean=clean, positions=positions, path=path, size=size, bands=bands, owner=owner,
                load=load, station_mask=station_mask, key=key, key_top=ky,
                bg=bg, carrier=carrier)


def _choose(rng, mask):
    points = np.argwhere(mask)
    if not len(points):
        return None
    return tuple(map(int, points[int(rng.integers(len(points)))]))


def _place_motif(rng, available, selected, offsets):
    pts = np.array(offsets, dtype=int)
    if rng.random() < .5:
        pts = pts[:, ::-1]
    pts *= rng.choice((-1, 1), size=2)
    pts -= pts.min(axis=0)
    hh, ww = pts.max(axis=0)+1
    h, w = available.shape
    good = np.ones((h-hh+1, w-ww+1), dtype=bool)
    for dy, dx in pts:
        good &= available[dy:dy+good.shape[0], dx:dx+good.shape[1]]
        good &= ~selected[dy:dy+good.shape[0], dx:dx+good.shape[1]]
    start = _choose(rng, good)
    if start is None:
        return False
    pts += start
    selected[pts[:, 0], pts[:, 1]] = True
    return True


def _defects(rng, scene, mode):
    owner, load = scene["owner"], scene["load"]
    key, station = scene["key"], scene["station_mask"]
    holes = np.zeros_like(station)
    reserved = np.zeros_like(station)
    if mode == "intro":
        if not _place_motif(rng, owner == 0, holes, ((0, 0), (1, 0), (1, 1))):
            return None
        # A hole next to a station shows that its halo crosses the road boundary.
        target = np.zeros_like(station)
        y, x = scene["positions"][scene["path"][2]]
        size = scene["size"]
        target[y:y+size, x:x+size] = True
        point = _choose(rng, (owner == 1) & _dilate(target))
        if point is None:
            return None
        holes[point] = True
    elif mode in ("cross", "test"):
        candidates = []
        for y, x in np.argwhere(load >= 2):
            late = int(owner[y, x])
            for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)):
                ny, nx = int(y+dy), int(x+dx)
                if 0 <= ny < owner.shape[0] and 0 <= nx < owner.shape[1]:
                    early = int(owner[ny, nx])
                    if 0 <= early < late and key[early] != key[late]:
                        candidates.append(((int(y), int(x)), (ny, nx)))
        if not candidates:
            return None
        a, b = candidates[int(rng.integers(len(candidates)))]
        holes[a] = holes[b] = True
        ma, mb = np.zeros_like(holes), np.zeros_like(holes)
        ma[a], mb[b] = True, True
        reserved = _dilate(ma) & _dilate(mb) & ~holes
    else:
        point = _choose(rng, owner >= 0)
        holes[point] = True
    budget = min(10, max(4, int((owner >= 0).sum())//4))
    wanted = int(holes.sum()) if mode == "intro" else int(rng.integers(max(2, int(holes.sum())), budget+1))
    if mode == "test":
        wanted = max(4, wanted)
    motifs = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
              ((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (2, 2)))
    for _ in range(30):
        remaining = wanted - int(holes.sum())
        if remaining <= 0:
            break
        options = [m for m in motifs if len(m) <= remaining]
        _place_motif(rng, (owner >= 0) & ~reserved, holes,
                     options[int(rng.integers(len(options)))])
    if holes.sum() > (owner >= 0).sum()/2:
        return None
    field = (scene["clean"] == scene["bg"])
    # Keep the key lane and station identification free of accidental markers.
    field[scene["key_top"]-1:] = False
    field[[0, -1]] = False
    field[:, [0, -1]] = False
    field &= ~_dilate(station)
    extras = np.zeros_like(holes)
    far = _choose(rng, field & ~_dilate(holes))
    if far is None:
        return None
    extras[far] = True
    near = _choose(rng, field & _dilate(holes))
    if near is not None and (mode in ("intro", "cross") or rng.random() < .6):
        extras[near] = True
    wanted_extras = int(rng.integers(max(1, int(extras.sum())), 7))
    for _ in range(18):
        if extras.sum() >= wanted_extras:
            break
        motif = ((0, 0), (1, 0)) if wanted_extras-extras.sum() >= 2 and rng.random() < .35 else ((0, 0),)
        _place_motif(rng, field, extras, motif)
    return holes, extras


def _paint_constructed(scene, holes, extras):
    """Labels from explicit construction masks; never calls solve or _prediction."""
    inp, out = scene["clean"].copy(), scene["clean"].copy()
    inp[holes], inp[extras] = scene["bg"], scene["carrier"]
    # Coordinate iteration is separate from the solver's mask dilation.
    points = sorted((int(scene["owner"][y, x]), int(y), int(x)) for y, x in np.argwhere(holes))
    h, w = out.shape
    for step, y, x in points:
        for ny in range(max(0, y-1), min(h, y+2)):
            for nx in range(max(0, x-1), min(w, x+2)):
                if not holes[ny, nx]:
                    out[ny, nx] = scene["key"][step]
    out[holes] = scene["bg"]
    return {"input": inp, "output": out}


def _sample_pair(rng, params, mode):
    for attempt in range(240):
        if mode == "intro":
            size, patch, length, crossing = params["demo_size"], params["wide"], 4, False
        elif mode == "cross":
            size, patch, length, crossing = params["demo_size"], params["narrow"], 6, True
        else:
            size = int(rng.choice((3, 4, 5))) if mode == "test" else int(rng.choice((3, 4, 5, 6, 7)))
            patch = int(rng.choice(list(range(1 if size % 2 else 2, size-1, 2))))
            length = int(rng.integers(7, 11)) if mode == "test" else int(rng.integers(3, 10))
            crossing = mode == "test" or bool(rng.random() < .5)
        options = [(r, c) for r in range(1, 5) for c in range(1, 6)
                   if r*size+(r-1)*2 <= 24 and c*size+(c-1)*2 <= 28
                   and length + int(crossing) <= r*c <= length+7
                   and (not crossing or min(r, c) >= 3)]
        if not options:
            continue
        rows, cols = options[int(rng.integers(len(options)))]
        unused = int(rng.integers(0, min(3, rows*cols-length-int(crossing))+1))
        n = length + unused
        universe = [(r, c) for r in range(rows) for c in range(cols)]
        cells = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
        # Trim unused coordinate levels so sampled canvas size reflects the scene.
        if len({r for r, c in cells}) != rows or len({c for r, c in cells}) != cols:
            continue
        graph = _lattice_graph(cells)
        path = _find_path(rng, cells, graph, length, crossing)
        if path is None:
            continue
        roles = params["roles"] if params["stable"] else tuple(map(int, rng.choice(10, 2, replace=False)))
        palette = [c for c in range(10) if c not in roles]
        repeated = mode == "cross" or (mode != "intro" and rng.random() < .65)
        assigned = _labels(rng, graph, path, palette, repeated)
        if assigned is None:
            continue
        labels, key = assigned
        scene = _construct_scene(rng, cells, path, labels, key, size, patch, roles)
        defect_mode = mode if mode in ("intro", "cross", "test") else "free"
        defects = _defects(rng, scene, defect_mode)
        if defects is None:
            continue
        pair = _paint_constructed(scene, *defects)
        info = _decode(pair["input"])
        if not np.array_equal(info["owner"], scene["owner"]):
            raise AssertionError("construction and input route disagree")
        if mode in ("cross", "test"):
            if (not _features(info)["ordered_collision"]
                    or np.array_equal(_prediction(info), _prediction(info, first_owner=True))):
                continue
        return pair
    raise RuntimeError("bounded scene search exhausted")


def _features(info):
    holes, owner = info["holes"], info["owner"]
    covers = np.zeros_like(owner)
    colors = np.full_like(owner, -1)
    ordered_collision = False
    for step in range(len(info["bands"])):
        halo = _dilate(holes & (owner == step)) & ~holes
        ordered_collision |= bool((halo & (covers > 0) & (colors != info["key"][step])).any())
        covers[halo] += 1
        colors[halo] = info["key"][step]
    cluster = bool((holes[:-1] & holes[1:]).any() or (holes[:, :-1] & holes[:, 1:]).any())
    diagonal = bool((holes[:-1, :-1] & holes[1:, 1:]).any() or (holes[:-1, 1:] & holes[1:, :-1]).any())
    route_positions = [s[:2] for s in info["stations"]]
    positions = tuple((y-min(s[0] for s in info["stations"]), x-min(s[1] for s in info["stations"])) for y, x, c in info["stations"])
    return dict(shape=info["a"].shape, size=info["size"], patch=info["patch"],
                length=len(info["key"]), stations=len(info["stations"]),
                unused=len(info["stations"])-len(info["key"]),
                holes=int(holes.sum()), extras=int(info["extras"].sum()),
                crossings=_cross_count(route_positions, info["route"]),
                cross_hole=bool((holes & (info["load"] > 1)).any()),
                ordered_collision=ordered_collision, orthogonal=cluster, diagonal=diagonal,
                boundary=bool((holes & _dilate(~(owner >= 0))).any()),
                station_halo=bool((_dilate(holes) & info["station_mask"]).any()),
                far_extras=int((info["extras"] & ~_dilate(holes)).sum()),
                adjacent_extras=int((info["extras"] & _dilate(holes)).sum()),
                repeated=len(set(info["key"])) < len(info["key"]),
                roles=(info["bg"], info["carrier"]), layout=positions,
                source_colors=len({info["key"][int(k)] for k in owner[holes]}))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            sig = a.shape, a.tobytes()
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            info = _decode(a)
            assert np.array_equal(out, _prediction(info)), "incorrect ordered audit"
            assert info["holes"].any() and info["holes"].sum() <= (info["owner"] >= 0).sum()/2
            assert info["extras"].sum() <= 6
            assert np.all(out[info["holes"]] == info["bg"]), "holes must stay empty"
            assert np.array_equal(a[info["key_mask"]], out[info["key_mask"]])
            assert np.array_equal(a[(a != info["bg"]) & (a != info["carrier"])],
                                  out[(a != info["bg"]) & (a != info["carrier"])])
            assert not np.array_equal(a, out)
            infos[split].append(info)
    train = [_features(a) for a in infos["train"]]
    test = [_features(a) for a in infos["test"]]
    assert any(a["size"] == b["size"] and a["patch"] != b["patch"] for a in train for b in train)
    for flag in ("orthogonal", "diagonal", "station_halo", "far_extras", "adjacent_extras", "repeated",
                 "cross_hole", "ordered_collision"):
        assert any(a[flag] for a in train), "training lacks " + flag
    assert any(a["source_colors"] >= 2 for a in train)
    assert any(not np.array_equal(_prediction(a, uniform=True), _prediction(a)) for a in infos["train"])
    assert any(not np.array_equal(_prediction(a, targets=True), _prediction(a)) for a in infos["train"])
    assert any(not np.array_equal(_prediction(a, earliest=True), _prediction(a)) for a in infos["train"])
    assert any(not np.array_equal(_prediction(a, first_owner=True), _prediction(a)) for a in infos["train"])
    for a in test:
        assert a["length"] > max(train[0]["length"], train[1]["length"])
        assert a["cross_hole"] and a["ordered_collision"]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(name + " must be a legal integer count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    size = int(rng.choice((5, 6)))
    params = dict(demo_size=size, wide=size-2, narrow=size-4,
                  roles=tuple(map(int, rng.choice(10, 2, replace=False))), stable=bool(rng.random() < .45))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mode = "test" if split == "test" else "intro" if i == 0 else "cross" if i == 1 else "free"
            for _ in range(20):
                pair = _sample_pair(rng, params, mode)
                signature = pair["input"].shape, pair["input"].tobytes()
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _encode(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
                       for s in ("train", "test")}, separators=(",", ":"))


def self_test():
    """Bounded 42-seed semantics, diversity, interface, and counterfactual checks."""
    shapes, layouts, signatures, roles = set(), set(), set(), set()
    metrics = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_encode(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                assert np.array_equal(solve(before.tolist()), pair["output"])
                assert np.array_equal(before, pair["input"])
                info = _decode(before)
                f = _features(info)
                metrics.append(f)
                shapes.add(f["shape"])
                layouts.add((f["size"], f["patch"], f["layout"]))
                roles.add(f["roles"])
                # Color equality alone defines roles; arbitrary bijections commute.
                if seed < 2:
                    perm = np.array([4, 7, 1, 9, 0, 8, 3, 5, 2, 6])
                    assert np.array_equal(solve(perm[before]), perm[pair["output"]])
                    # Reflect only the circuit, retaining the key's reading order.
                    reflected, expected = before.copy(), pair["output"].copy()
                    cut = info["key_top"]-1
                    reflected[:cut] = np.fliplr(reflected[:cut])
                    expected[:cut] = np.fliplr(expected[:cut])
                    assert np.array_equal(solve(reflected), expected)
        if seed < 2:
            assert _encode(puzzle) == _encode(generate_puzzle(seed))
        info = _decode(puzzle["train"][1]["input"])
        reversed_key = info["a"].copy()
        reversed_key[info["key_top"]+1, info["key_left"]+1:info["key_right"]:2] = info["key"][::-1]
        assert not np.array_equal(solve(reversed_key), puzzle["train"][1]["output"])
    assert len(signatures) == 42 and len(shapes) >= 25 and len(layouts) >= 80 and len(roles) >= 20
    assert {m["size"] for m in metrics} >= {3, 4, 5, 6, 7}
    assert {m["patch"] for m in metrics} >= {1, 2, 3, 4, 5}
    assert len({m["holes"] for m in metrics}) >= 6
    assert len({m["extras"] for m in metrics}) >= 4
    assert any(m["unused"] == 0 for m in metrics) and any(m["unused"] >= 2 for m in metrics)
    assert any(m["crossings"] >= 2 for m in metrics)
    assert any(m["adjacent_extras"] for m in metrics)
    # The input-only rule also covers an unsampled, single-hole/no-speck member.
    original = generate_puzzle(0)["train"][0]["input"]
    info = _decode(original)
    clean = original.copy()
    clean[info["extras"]] = info["bg"]
    clean[info["holes"]] = info["carrier"]
    y, x = map(int, np.argwhere(info["holes"] & (info["owner"] == 1))[0])
    clean[y, x] = info["bg"]
    expected = clean.copy()
    expected[max(0, y-1):y+2, max(0, x-1):x+2] = info["key"][1]
    expected[y, x] = info["bg"]
    assert np.array_equal(solve(clean), expected)
    for nt, nq in ((2, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(311470191, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    bad = generate_puzzle(0)
    info = _decode(bad["test"][0]["input"])
    point = tuple(np.argwhere(info["holes"])[0])
    bad["test"][0]["output"][point] = info["carrier"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError("filled hole was accepted")
