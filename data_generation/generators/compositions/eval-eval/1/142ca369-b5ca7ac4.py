"""Optical docks: route, relabel, align, and stably compact framed cargo."""
from __future__ import annotations

import numpy as np


EDGES = ("L", "R", "U", "D")


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=True)


def _components(g, bg):
    todo = set(map(tuple, np.argwhere(g != bg)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _decode(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg]*2 <= g.size:
        raise ValueError("background must occupy more than half of the input")
    cargo, sources, mirrors = [], {}, []
    for cells in _components(g, bg):
        rs, cs = zip(*cells)
        r, c, h, w = min(rs), min(cs), max(rs)-min(rs)+1, max(cs)-min(cs)+1
        patch = g[r:r+h, c:c+w]
        shade = int(g[cells[0]])
        if len(cells) == h*w and h >= 3 and w >= 3:
            if not (h % 2 and w % 2):
                raise ValueError("cargo dimensions must be odd")
            if not (np.all(patch[0] == shade) and np.all(patch[-1] == shade)
                    and np.all(patch[:, 0] == shade) and np.all(patch[:, -1] == shade)
                    and np.all(patch[1:-1, 1:-1] != shade)):
                raise ValueError("invalid cargo frame")
            cargo.append(dict(r=r, c=c, h=h, w=w, tag=shade, patch=patch.copy()))
            continue
        if any(int(g[p]) != shade for p in cells):
            raise ValueError("optical objects must be monochromatic and separated")
        if len(cells) == 3 and h == w == 2:
            absent = next(p for p in ((r,c), (r,c+1), (r+1,c), (r+1,c+1)) if p not in cells)
            elbow = (2*r+1-absent[0], 2*c+1-absent[1])
            if shade in sources:
                raise ValueError("source tags must be distinct")
            sources[shade] = dict(cells=cells, elbow=elbow,
                                  direction=(elbow[0]-absent[0], elbow[1]-absent[1]), color=shade)
        elif len(cells) % 2 and (h == 1 or w == 1):
            mirrors.append(dict(cells=cells, mid=(r+h//2, c+w//2), color=shade,
                                axis=None if len(cells) == 1 else 0 if h == 1 else 1))
        else:
            raise ValueError("unrecognized optical object")
    if not cargo or set(sources) != {o["tag"] for o in cargo}:
        raise ValueError("each source tag must address cargo, and every cargo needs a source")
    return bg, cargo, sources, mirrors


def _trace(shape, sources, mirrors):
    """Cargo is data, not an optical object; rays are implicit control signals."""
    occupied = {p for s in sources.values() for p in s["cells"]}
    occupied.update(p for m in mirrors for p in m["cells"])
    routes, used = {}, set()
    H, W = shape
    for tag, source in sources.items():
        p, d, color = source["elbow"], source["direction"], tag
        states, path, contacts = set(), [], []
        while True:
            q = (p[0]+d[0], p[1]+d[1])
            outside = (not 0 <= q[0] < H, not 0 <= q[1] < W)
            if any(outside):
                if all(outside) or not path:
                    raise ValueError("corner exits or zero-length routes are ambiguous")
                edge = ("U" if q[0] < 0 else "D") if outside[0] else ("L" if q[1] < 0 else "R")
                routes[tag] = dict(edge=edge, exit=p, color=color, path=tuple(path),
                                   contacts=tuple(contacts), initial=source["direction"])
                break
            state = (q, d)
            if state in states:
                raise ValueError("optical loop")
            states.add(state)
            if q in occupied:
                raise ValueError("ray struck an optical object away from its midpoint face")
            hits = []
            for i, m in enumerate(mirrors):
                for axis in (0, 1) if m["axis"] is None else (m["axis"],):
                    other = 1-axis
                    if q[other] == m["mid"][other] and m["mid"][axis]-q[axis] == d[axis]:
                        hits.append((i, axis))
            if len(hits) > 1:
                raise ValueError("ambiguous midpoint contact")
            if hits:
                i, axis = hits[0]
                color = mirrors[i]["color"]
                d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
                used.add(i)
                contacts.append((i, axis, q, color))
            path.append((q, color))
            p = q
    if used != set(range(len(mirrors))):
        raise ValueError("every mirror must participate in a route")
    return routes


def _overlap(a, b):
    return (max(a["r"], b["r"]) < min(a["r"]+a["h"], b["r"]+b["h"])
            and max(a["c"], b["c"]) < min(a["c"]+a["w"], b["c"]+b["w"]))


def _axis(o):
    return 1 if o["edge"] in ("L", "R") else 0


def _coordinate(o, axis):
    return o["c"] if axis else o["r"]


def _size(o, axis):
    return o["w"] if axis else o["h"]


def _set_coordinate(o, axis, value):
    o["c" if axis else "r"] = int(value)


def _span_overlap(a, b, axis):
    perp = 1-axis
    return max(_coordinate(a, perp), _coordinate(b, perp)) < min(
        _coordinate(a, perp)+_size(a, perp), _coordinate(b, perp)+_size(b, perp))


def _project(cargo, routes, shape):
    objects = []
    for original in cargo:
        o = dict(original)
        route = routes[o["tag"]]
        o.update(edge=route["edge"], final_color=route["color"])
        axis = _axis(o)
        perp = 1-axis
        coordinate = route["exit"][perp]-_size(o, perp)//2
        o["clamped"] = not 0 <= coordinate <= shape[perp]-_size(o, perp)
        _set_coordinate(o, perp, max(0, min(coordinate, shape[perp]-_size(o, perp))))
        o["patch"] = o["patch"].copy()
        o["patch"][0, :] = o["patch"][-1, :] = route["color"]
        o["patch"][:, 0] = o["patch"][:, -1] = route["color"]
        objects.append(o)
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            if a["edge"] == b["edge"] and _overlap(a, b):
                raise ValueError("same-edge cargo overlaps after transverse alignment")
    return objects


def _pack(objects, shape, *, unstable=False, ignore_blockers=False):
    result = [dict(o) for o in objects]
    for edge in EDGES:
        group = [o for o in result if o["edge"] == edge]
        axis, low = (1 if edge in ("L", "R") else 0), edge in ("L", "U")
        group.sort(key=lambda o: _coordinate(o, axis) if low else -_coordinate(o, axis)-_size(o, axis))
        settled = []
        for o in group:
            size = _size(o, axis)
            blockers = [] if ignore_blockers else [p for p in settled if _span_overlap(o, p, axis)]
            if unstable:
                positions = range(shape[axis]-size+1)
                if not low:
                    positions = reversed(positions)
                for position in positions:
                    candidate = dict(o)
                    _set_coordinate(candidate, axis, position)
                    if not any(_overlap(candidate, p) for p in settled):
                        break
                else:
                    raise ValueError("insufficient packing capacity")
            elif low:
                position = max((_coordinate(p, axis)+_size(p, axis) for p in blockers), default=0)
            else:
                position = min((_coordinate(p, axis) for p in blockers), default=shape[axis])-size
            if not 0 <= position <= shape[axis]-size:
                raise ValueError("packing chain exceeds the canvas")
            _set_coordinate(o, axis, position)
            settled.append(o)
    if not ignore_blockers and any(_overlap(a,b) for i,a in enumerate(result) for b in result[i+1:]):
        raise ValueError("final docks collide")
    return result


def _render(shape, bg, objects):
    out = np.full(shape, bg, dtype=np.int8)
    for o in objects:
        out[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]] = o["patch"]
    return out


def _analyze(input_grid):
    g = _grid(input_grid)
    bg, cargo, sources, mirrors = _decode(g)
    routes = _trace(g.shape, sources, mirrors)
    projected = _project(cargo, routes, g.shape)
    return g, bg, cargo, sources, mirrors, routes, projected


def solve(input_grid: np.ndarray) -> np.ndarray:
    g, bg, _, _, _, _, projected = _analyze(input_grid)
    return _render(g.shape, bg, _pack(projected, g.shape))


def _separated(cells, occupied, shape):
    H, W = shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if any(p in occupied for p in ((r,c),(r-1,c),(r+1,c),(r,c-1),(r,c+1))):
            return False
    return True


def _source_cells(elbow, direction):
    r,c = elbow
    dr,dc = direction
    return ((r,c), (r-dr,c), (r,c-dc))


def _optics(rng, shape, destinations, bounce_axes, tags, bg, final_colors=None):
    """Build backward from requested docks, then verify complete forward routes.

    This constructs small optical circuits, not parent puzzle outputs.
    """
    sources, mirrors, occupied = {}, [], set()
    routes = {}
    for channel, ((edge, transverse), axes, tag) in enumerate(zip(destinations, bounce_axes, tags)):
        found = False
        for _ in range(150):
            H,W = shape
            endpoint = {"L":(transverse,0), "R":(transverse,W-1),
                        "U":(0,transverse), "D":(H-1,transverse)}[edge]
            random_sign = int(rng.choice((-1,1)))
            d = {"L":(random_sign,-1), "R":(random_sign,1),
                 "U":(-1,random_sign), "D":(1,random_sign)}[edge]
            p, added, local = endpoint, [], set(occupied)
            valid = True
            for axis in reversed(axes):
                distance = int(rng.integers(2,9))
                contact = (p[0]-distance*d[0], p[1]-distance*d[1])
                incoming = (-d[0],d[1]) if axis == 0 else (d[0],-d[1])
                mid = list(contact)
                mid[axis] += incoming[axis]
                length = int(rng.choice((1,3,3,5)))
                radius = length//2
                cells = tuple((mid[0], mid[1]+k) if axis == 0 else (mid[0]+k, mid[1])
                              for k in range(-radius,radius+1))
                if not _separated(cells,local,shape):
                    valid = False
                    break
                color = int(rng.choice([v for v in range(10) if v != bg]))
                if not added and final_colors is not None and final_colors[channel] is not None:
                    color = final_colors[channel]
                added.append(dict(cells=cells,mid=tuple(mid),color=color,
                                  axis=None if length == 1 else axis))
                local.update(cells)
                p,d = contact,incoming
            if not valid:
                continue
            distance = int(rng.integers(2,9))
            elbow = (p[0]-distance*d[0],p[1]-distance*d[1])
            cells = _source_cells(elbow,d)
            if not _separated(cells,local,shape):
                continue
            source = dict(cells=cells,elbow=elbow,direction=d,color=tag)
            trial_sources = dict(sources)
            trial_sources[tag] = source
            try:
                trial = _trace(shape,trial_sources,mirrors+added)
            except ValueError:
                continue
            if (trial[tag]["edge"] != edge or trial[tag]["exit"] != endpoint
                    or len(trial[tag]["contacts"]) != len(axes)
                    or any(trial[t] != routes[t] for t in routes)):
                continue
            sources,mirrors,routes = trial_sources,mirrors+added,trial
            occupied = local | set(cells)
            found = True
            break
        if not found:
            return None
    return sources,mirrors,routes,occupied


def _appearance(rng, h, w, tag, final, bg, pattern_probability):
    allowed = [v for v in range(10) if v not in (tag,final,bg)]
    interior = np.full((h-2,w-2), int(rng.choice(allowed)), dtype=np.int8)
    if interior.size > 1 and rng.random() < pattern_probability:
        shades = rng.choice(allowed, int(rng.integers(2,min(4,len(allowed))+1)),replace=False)
        rr,cc = np.indices(interior.shape)
        mode = int(rng.integers(4))
        indices = rr if mode == 0 else cc if mode == 1 else rr+cc if mode == 2 else rng.integers(len(shades),size=interior.shape)
        interior = shades[indices % len(shades)].astype(np.int8)
    patch = np.full((h,w),tag,dtype=np.int8)
    patch[1:-1,1:-1] = interior
    return patch


def _make_scene(rng, style, purpose, anchor=None, require_transit=False):
    # Teaching scenes are rotated as whole geometric constructions. These are
    # sampling templates, not solving conventions or separate rules.
    teach = purpose in ("horizontal", "vertical")
    for attempt in range(240):
        bg = int(rng.choice([v for v in range(10) if v != anchor]))
        if teach:
            shape = (int(rng.integers(21,31)),int(rng.integers(25,31)))
            edge = str(rng.choice(("L","R")))
            opposite = "R" if edge == "L" else "L"
            chain_length = int(rng.choice((3,3,4)))
            thickness = int(rng.choice((3,5))) if purpose == "horizontal" else 5
            step = int(rng.integers((thickness+1)//2,thickness))
            max_base = shape[0]-7-thickness//2-(chain_length-1)*step
            if purpose == "horizontal":
                if max_base < thickness//2+1:
                    continue
                base = int(rng.integers(thickness//2+1,max_base+1))
            else:
                base = 1  # A demonstrably clipped finite-width dock.
            centers = [base+j*step for j in range(chain_length)]
            destinations = [(edge,t) for t in centers] + [(opposite,shape[0]-4)]
            core_axes = [(),(1,),(0,1)] if purpose == "horizontal" else [(0,),(0,0),()]
            axes = core_axes + ([tuple(int(x) for x in rng.integers(0,2,int(rng.integers(0,3))))]
                                if chain_length == 4 else [])
            axes += [(0,)] if purpose == "horizontal" else [(1,)]
            sizes = [(thickness,int(rng.choice((3,5)))) for _ in range(chain_length)] + [(3,3)]
            multiplicities = [1]*chain_length + [2 if purpose == "horizontal" else 1]
            total_width = sum(w for _,w in sizes[:-1])
            spare = shape[1]-total_width-(chain_length-1)
            gaps = rng.multinomial(spare,np.full(chain_length+1,1/(chain_length+1)))
            starts, position = [],int(gaps[0])
            for j,(_,w) in enumerate(sizes[:-1]):
                starts.append(position)
                position += w+1+int(gaps[j+1])
        else:
            shape = (int(rng.integers(13,31)),int(rng.integers(13,31)))
            channels = int(rng.integers(1,6))
            destinations,axes,sizes,multiplicities = [],[],[],[]
            for j in range(channels):
                edge = str(rng.choice(EDGES))
                D = shape[0] if edge in ("L","R") else shape[1]
                destinations.append((edge,int(rng.integers(1,D-1))))
                count = int(rng.integers(0,4 if purpose == "test" else 3))
                axes.append(tuple(int(x) for x in rng.integers(0,2,count)))
                choices = (3,5,7) if purpose == "test" else (3,5)
                sizes.append((int(rng.choice(choices)),int(rng.choice(choices))))
                multiplicities.append(int(rng.choice((1,1,1,2))))
            if purpose == "test":
                # A new transverse footprint changes dock spans and therefore
                # can change the packing graph, even with familiar optics.
                h,w = sizes[0]
                sizes[0] = (7,w) if destinations[0][0] in ("L","R") else (h,7)
        tags = [int(v) for v in rng.choice([v for v in range(10) if v != bg],len(destinations),replace=False)]
        if teach:
            if anchor in tags:
                tags.remove(anchor)
            tags = [anchor] + tags[:len(destinations)-1]
        final_colors = None
        if teach:
            final_colors = [None]*len(tags)
            if purpose == "horizontal":
                final_colors[1] = final_colors[-1] = tags[1]
            else:
                final_colors[0] = tags[1]
        optics = _optics(rng,shape,destinations,axes,tags,bg,final_colors)
        if optics is None:
            continue
        sources,mirrors,routes,occupied = optics
        if teach and {m["axis"] for m in mirrors if m["axis"] is not None} != {0,1}:
            continue
        if purpose == "horizontal" and not any(len(m["cells"]) == 1 for m in mirrors):
            continue
        objects = []
        success = True
        for j, (tag, (h,w), multiplicity) in enumerate(zip(tags,sizes,multiplicities)):
            for k in range(multiplicity):
                found = False
                for _ in range(160):
                    r = int(rng.integers(shape[0]-h+1))
                    c = int(rng.integers(shape[1]-w+1))
                    if teach and j < chain_length:
                        c = starts[j] if edge == "L" else shape[1]-starts[j]-w
                    cells = {(rr,cc) for rr in range(r,r+h) for cc in range(c,c+w)}
                    if not _separated(cells,occupied,shape):
                        continue
                    candidate = dict(r=r,c=c,h=h,w=w,tag=tag,
                                     patch=_appearance(rng,h,w,tag,routes[tag]["color"],bg,style))
                    try:
                        projected = _project(objects+[candidate],routes,shape)
                        _pack(projected,shape)
                    except ValueError:
                        continue
                    objects.append(candidate)
                    occupied.update(cells)
                    found = True
                    break
                if not found:
                    success = False
                    break
            if not success:
                break
        if not success:
            continue
        g = _render(shape,bg,objects)
        for source in sources.values():
            for p in source["cells"]:
                g[p] = source["color"]
        for mirror in mirrors:
            for p in mirror["cells"]:
                g[p] = mirror["color"]
        if np.count_nonzero(g == bg)*2 <= g.size:
            continue
        if purpose == "vertical":
            g = g.T.copy()
        try:
            details = _analyze(g)
            if require_transit and not any(g[point] != bg for route in details[5].values()
                                           for point,_ in route["path"]):
                continue
            projected = details[-1]
            settled = _pack(projected,g.shape)
            if teach and not _witness(projected,g.shape):
                continue
            out = _render(g.shape,bg,settled)
            return dict(input=g,output=out)
        except ValueError:
            continue
    raise RuntimeError(f"could not construct {purpose} scene in bounded attempts")


def _witness(projected, shape):
    stable = _pack(projected,shape)
    try:
        unstable = _pack(projected,shape,unstable=True)
    except ValueError:
        return True
    return any((a["r"],a["c"]) != (b["r"],b["c"]) for a,b in zip(stable,unstable))


def _counts(num_train, num_test):
    for value,name,lo,hi in ((num_train,"num_train",2,10),(num_test,"num_test",1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    style = float(rng.choice((0.0,0.35,0.8)))
    anchor = int(rng.integers(10))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",num_train),("test",num_test)):
        for i in range(count):
            purpose = ("horizontal" if i == 0 else "vertical" if i == 1 else "free") if split == "train" else "test"
            require_transit = False
            if split == "train" and i == 1:
                first = _analyze(puzzle["train"][0]["input"])
                require_transit = not any(first[0][point] != first[1] for route in first[5].values()
                                          for point,_ in route["path"])
            for _ in range(30):
                pair = _make_scene(rng,style,purpose,
                                   anchor if purpose in ("horizontal","vertical") else None,
                                   require_transit=require_transit)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(g):
    return g.shape,g.astype(np.int8).tobytes()


def _physical_pack(projected, shape):
    """Supplemental audit via unit steps, rather than the boundary recurrence."""
    objects = [dict(o) for o in projected]
    for edge in EDGES:
        group = [o for o in objects if o["edge"] == edge]
        axis = 1 if edge in ("L","R") else 0
        delta = -1 if edge in ("L","U") else 1
        for _ in range(shape[axis]*max(1,len(group))+1):
            moved = False
            for o in group:
                p = _coordinate(o,axis)+delta
                candidate = dict(o)
                _set_coordinate(candidate,axis,p)
                if (0 <= p <= shape[axis]-_size(o,axis)
                        and not any(_overlap(candidate,other) for other in group if other is not o)):
                    _set_coordinate(o,axis,p)
                    moved = True
            if not moved:
                break
        else:
            raise AssertionError("physical packing failed to settle")
    assert not any(_overlap(a,b) for i,a in enumerate(objects) for b in objects[i+1:])
    return objects


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,train_edges,train_axes,train_initial = set(),set(),set(),set()
    outcomes, final_edges = {}, {}
    unchanged_contact = multiple_cargo = cross_tag_blocker = cargo_transit = point_seen = False
    recolored = clamped = reordered = geometry_changed = False
    for split in ("train","test"):
        for pair_index,pair in enumerate(puzzle[split]):
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            g,out = _grid(pair["input"]),_grid(pair["output"])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            _,bg,cargo,sources,mirrors,routes,projected = _analyze(g)
            checked = _physical_pack(projected,g.shape)
            assert np.array_equal(out,_render(g.shape,bg,checked)), "incorrect cargo delivery"
            # Explicit interior preservation and border-only relabeling.
            for original,delivered in zip(cargo,checked):
                r,c,h,w = (delivered[k] for k in ("r","c","h","w"))
                patch = out[r:r+h,c:c+w]
                assert np.array_equal(patch[1:-1,1:-1],original["patch"][1:-1,1:-1])
                assert np.all(patch[0] == delivered["final_color"])
                assert np.all(patch[-1] == delivered["final_color"])
                assert np.all(patch[:,0] == delivered["final_color"])
                assert np.all(patch[:,-1] == delivered["final_color"])
            assert set(np.unique(out)) <= set(np.unique(g))
            assert not np.array_equal(g,out)
            if split == "train" and pair_index < 2:
                train_edges.update(o["edge"] for o in projected)
                train_axes.update(contact[1] for route in routes.values() for contact in route["contacts"]
                                  if mirrors[contact[0]]["axis"] is not None)
                multiple_cargo |= len(cargo) > len(sources)
                point_seen |= any(len(m["cells"]) == 1 for m in mirrors)
                cargo_transit |= any(g[point] != bg for route in routes.values() for point,_ in route["path"])
                cross_tag_blocker |= any(a["tag"] != b["tag"] and a["edge"] == b["edge"]
                    and _span_overlap(a,b,_axis(a)) for i,a in enumerate(projected) for b in projected[i+1:])
                for tag, route in routes.items():
                    outcomes.setdefault(tag,set()).add(route["color"])
                    final_edges.setdefault(route["color"],set()).add(route["edge"])
                    incoming = tag
                    for _,_,_,shade in route["contacts"]:
                        unchanged_contact |= incoming == shade
                        incoming = shade
                train_initial.update(s["direction"] for s in sources.values())
                recolored |= any(tag != route["color"] for tag,route in routes.items())
                clamped |= any(o["clamped"] for o in projected)
                reordered |= _witness(projected,g.shape)
                for tag,route in routes.items():
                    if route["contacts"]:
                        p,d = sources[tag]["elbow"],sources[tag]["direction"]
                        while 0 <= p[0]+d[0] < g.shape[0] and 0 <= p[1]+d[1] < g.shape[1]:
                            p = (p[0]+d[0],p[1]+d[1])
                        geometry_changed |= p != route["exit"]
    assert train_edges == set(EDGES), "training must demonstrate all four dock edges"
    assert train_axes == {0,1}, "training must demonstrate both reflection axes"
    assert len(train_initial) >= 2
    assert recolored and clamped and reordered and geometry_changed
    assert unchanged_contact and multiple_cargo and cross_tag_blocker and cargo_transit and point_seen
    assert any(len(values)>1 for values in outcomes.values()), "training must refute a fixed color successor"
    assert any(len(edges)>1 for edges in final_edges.values()), "handoff colors must not secretly encode an edge"


def measure_puzzle(puzzle):
    records = []
    for split in ("train","test"):
        for pair in puzzle[split]:
            g,bg,cargo,sources,mirrors,routes,projected = _analyze(pair["input"])
            packed = _pack(projected,g.shape)
            records.append(dict(split=split,shape=list(g.shape),background=bg,
                sources=len(sources),cargo=len(cargo),mirrors=len(mirrors),
                edges=sorted({r["edge"] for r in routes.values()}),
                bounces=[len(r["contacts"]) for r in routes.values()],
                axes=[c[1] for r in routes.values() for c in r["contacts"]],
                footprints=sorted({(o["h"],o["w"]) for o in cargo}),
                transverse_sizes=sorted({_size(o,1-_axis(o)) for o in projected}),
                bar_lengths=sorted({len(m["cells"]) for m in mirrors}),
                cargo_transit_cells=sum(int(g[point] != bg) for r in routes.values() for point,_ in r["path"]),
                recolored=sum(t != r["color"] for t,r in routes.items()),
                clamped=sum(o["clamped"] for o in projected),
                pattern=sum(len(np.unique(o["patch"][1:-1,1:-1]))>1 for o in cargo),
                perpendicular_moves=sum((o["r"],o["c"]) != (p["r"],p["c"]) for o,p in zip(cargo,projected)),
                overlap_edges=sum(a["edge"] == b["edge"] and _span_overlap(a,b,_axis(a))
                                  for i,a in enumerate(projected) for b in projected[i+1:]),
                stable_witness=_witness(projected,g.shape),
                positions=[[o["r"],o["c"]] for o in packed]))
    return records


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_key(p["input"]) for s in puzzle.values() for p in s) for puzzle in puzzles}) == 42
    assert all(np.array_equal(solve(p["input"].copy()),p["output"])
               for puzzle in puzzles for pairs in puzzle.values() for p in pairs)
    assert all(_key(a[k]) == _key(b[k]) for split in ("train","test")
               for a,b in zip(puzzles[0][split],generate_puzzle(0)[split]) for k in ("input","output"))
    records = [r for puzzle in puzzles for r in measure_puzzle(puzzle)]
    assert len({tuple(r["shape"]) for r in records}) >= 40
    assert {r["background"] for r in records} == set(range(10))
    assert {1,2,3,4} <= {r["sources"] for r in records}
    assert {0,1,2,3} <= {b for r in records for b in r["bounces"]}
    assert {1,3,5} <= {n for r in records for n in r["bar_lengths"]}
    assert any(r["pattern"] for r in records) and any(r["pattern"] == 0 for r in records)
    assert any(r["cargo"] > r["sources"] for r in records)
    assert any(len(r["edges"]) >= 3 for r in records)
    assert all(7 in r["transverse_sizes"] for r in records if r["split"] == "test")
    assert all(set(r["transverse_sizes"]) <= {3,5} for r in records if r["split"] == "train")
    _rule_regressions()
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            puzzle = generate_puzzle(80000+nt*10+ne,nt,ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s:[{k:np.array(v,copy=True) for k,v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")


def _rule_regressions():
    # A direct midpoint reflection routes a source that initially points left
    # to the bottom, changing the frame and retaining a nonuniform payload.
    grid = np.zeros((12,14),dtype=np.int8)
    grid[0,6] = grid[1,6] = grid[1,7] = 1
    grid[3:6,2] = 2
    patch = np.ones((3,5),dtype=np.int8)
    patch[1,1:-1] = (3,4,5)
    grid[7:10,2:7] = patch
    expected = np.zeros_like(grid)
    patch[0] = patch[-1] = 2
    patch[:,0] = patch[:,-1] = 2
    expected[9:12,8:13] = patch
    assert np.array_equal(solve(grid),expected)
    _,_,_,_,_,routes,_ = _analyze(grid)
    assert routes[1]["edge"] == "D" and routes[1]["exit"] == (11,10)
    assert routes[1]["path"][:3] == (((2,5),1),((3,4),1),((4,3),2))
    # Mixed-width indirect chain: the third cannot jump the middle blocker,
    # even though it is disjoint from the first object's perpendicular span.
    chain = [dict(r=r,c=c,h=5,w=w,edge="L")
             for r,c,w in ((1,2,3),(5,10,5),(9,18,3))]
    shape = (17,25)
    assert [o["c"] for o in _pack(chain,shape)] == [0,3,8]
    assert [o["c"] for o in _physical_pack(chain,shape)] == [0,3,8]
    assert [o["c"] for o in _pack(chain,shape,unstable=True)] == [0,3,0]
    reflected = [dict(o,c=shape[1]-o["c"]-o["w"],edge="R") for o in chain]
    assert [o["c"] for o in _pack(reflected,shape)] == [22,17,14]
    transposed = [dict(o,r=o["c"],c=o["r"],h=o["w"],w=o["h"],edge="U") for o in chain]
    assert [o["r"] for o in _pack(transposed,shape[::-1])] == [0,3,8]
    # Same-class projected collision and incompatible final edge groups are
    # rejected, rather than being resolved by hidden painting order.
    collision = [dict(r=1,c=1,h=3,w=5,edge="L"),dict(r=1,c=2,h=3,w=5,edge="R")]
    try:
        _pack(collision,(10,8))
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping final dock groups accepted")
    cargo = []
    for tag,col in ((1,1),(2,8)):
        patch = np.full((3,3),tag,dtype=np.int8)
        patch[1,1] = tag+3
        cargo.append(dict(r=1,c=col,h=3,w=3,tag=tag,patch=patch))
    routes = {tag:dict(edge="U",exit=(0,5),color=tag) for tag in (1,2)}
    try:
        _project(cargo,routes,(12,14))
    except ValueError:
        pass
    else:
        raise AssertionError("same-edge overlap after alignment accepted")
