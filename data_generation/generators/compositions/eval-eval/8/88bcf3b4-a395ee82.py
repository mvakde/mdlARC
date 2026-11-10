"""A sparse peg blueprint is made real only where a taut thread catches it.

Only Python's standard library and NumPy are used. All semantic parameters are
recovered from the query pixels; sampling choices never affect the rule.
"""
from __future__ import annotations

import numpy as np

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells, directions=DIR8):
    unseen, result = set(cells), []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _local(p, endpoint, forward, lateral):
    r, c = p[0] - endpoint[0], p[1] - endpoint[1]
    return r * forward[0] + c * forward[1], r * lateral[0] + c * lateral[1]


def _global(p, endpoint, forward, lateral):
    u, v = p
    return (endpoint[0] + u * forward[0] + v * lateral[0],
            endpoint[1] + u * forward[1] + v * lateral[1])


def _raster_hull(hull, length, outward=True):
    route, segment = [], 0
    tail_intercept = hull[-1][0] + hull[-1][1]
    for u in range(length):
        if u >= hull[-1][0]:
            v = tail_intercept - u
        else:
            while hull[segment + 1][0] < u:
                segment += 1
            a, b = hull[segment:segment + 2]
            den = b[0] - a[0]
            num = a[1] * den + (b[1] - a[1]) * (u - a[0])
            v = -((-num) // den) if outward else num // den
        route.append((u, v))
    return route


def _hull_route(obstacles, length):
    """Outward-ceiled upper hull, anchored at (0,1), leaving with slope -1.

    Clearance constraints are (u,v+1). Equal-slope interior hull vertices are
    removed, but their contacts remain detectable against the actual cells.
    Integer arithmetic makes rasterization exact, even on negative ordinates.
    """
    assert obstacles and min(u for u, v in obstacles) >= 2
    assert max(v - u for u, v in obstacles) <= 0
    profile = {}
    for u, v in obstacles:
        profile[u] = max(profile.get(u, -1000), v + 1)
    far = max(profile)
    tail_intercept = max(1, max(u + v + 1 for u, v in obstacles))
    points = [(0, 1)] + sorted(profile.items()) + [(far + 2, tail_intercept - far - 2)]
    hull = []
    for p in points:
        while len(hull) >= 2:
            a, b = hull[-2:]
            # Slopes must strictly decrease along the upper convex hull.
            if (b[1] - a[1]) * (p[0] - b[0]) > (p[1] - b[1]) * (b[0] - a[0]):
                break
            hull.pop()
        hull.append(p)
    route = _raster_hull(hull, length)
    assert route[0] == (0, 1)
    assert all(abs(b[1] - a[1]) <= 1 for a, b in zip(route, route[1:]))
    assert not set(route) & obstacles
    return route, hull


def _field(mask, origin, sites, anchor):
    h, w = mask.shape
    pixels = [tuple(map(int, p)) for p in np.argwhere(mask)]
    copies = {}
    for r, c in sites:
        top = origin[0] + (r - anchor[0]) * h
        left = origin[1] + (c - anchor[1]) * w
        copies[(r, c)] = {(top + dr, left + dc) for dr, dc in pixels}
    return copies


def _decode(inp):
    """Unique roles from components, pitch-two points and support attachment."""
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 5, "background plus four distinct foreground roles required"
    bg = int(values[np.argmax(counts)])
    assert counts.max() > sorted(map(int, counts))[-2], "background must be unique mode"
    cells = {int(v): {tuple(map(int, p)) for p in np.argwhere(inp == v)}
             for v in values if v != bg}
    comps = {v: _components(p) for v, p in cells.items()}
    glyph_colors = [v for v, cs in comps.items()
                    if len(cs) == 2 and sorted(map(len, cs))[0] == 1
                    and sorted(map(len, cs))[1] >= 3]
    assert len(glyph_colors) == 1, "glyph and exceptional point must be identifiable"
    ink = glyph_colors[0]
    glyph = next(p for p in comps[ink] if len(p) > 1)
    anchor_pixel = next(iter(next(p for p in comps[ink] if len(p) == 1)))
    top, left, bottom, right = _bbox(glyph)
    h, w = bottom - top + 1, right - left + 1
    assert 2 <= h <= 4 and 2 <= w <= 4
    assert len(_components(glyph, DIR4)) == 1, "glyph must be four-connected"
    mask = inp[top:bottom + 1, left:right + 1] == ink
    marker_colors = [v for v, cs in comps.items() if v != ink
                     and len(cs) >= 2 and all(len(p) == 1 for p in cs)]
    assert len(marker_colors) == 1, "ordinary blueprint points must be isolated"
    marker = marker_colors[0]
    points = cells[marker] | {anchor_pixel}
    pr, pc, er, ec = _bbox(points)
    assert all((r - pr) % 2 == 0 and (c - pc) % 2 == 0 for r, c in points)
    assert (bottom + 1 < pr or er + 1 < top or right + 1 < pc or ec + 1 < left), "blueprint/glyph need a blank band"
    sites = {(int(r - pr) // 2, int(c - pc) // 2) for r, c in points}
    anchor = (anchor_pixel[0] - pr) // 2, (anchor_pixel[1] - pc) // 2
    remaining = [v for v in cells if v not in (ink, marker)]
    candidates = []
    for sc in remaining:
        support = cells[sc]
        if len(comps[sc]) != 1 or len(support) < 3:
            continue
        r0, c0, r1, c1 = _bbox(support)
        if r0 == r1 and len(support) == c1 - c0 + 1:
            ends = [((r0, c0), (0, -1)), ((r1, c1), (0, 1))]
        elif c0 == c1 and len(support) == r1 - r0 + 1:
            ends = [((r0, c0), (-1, 0)), ((r1, c1), (1, 0))]
        else:
            continue
        tc = next(v for v in remaining if v != sc)
        slack = cells[tc]
        if len(comps[tc]) != 1 or len(slack) < 3:
            continue
        # An irregular two-dimensional strand cannot itself be a support.
        sr0, sl0, sr1, sl1 = _bbox(slack)
        if sr0 == sr1 or sl0 == sl1:
            continue
        for endpoint, forward in ends:
            for lateral in ((forward[1], -forward[0]), (-forward[1], forward[0])):
                attachment = _global((0, 1), endpoint, forward, lateral)
                local_slack = {_local(p, endpoint, forward, lateral) for p in slack}
                if attachment not in slack or any(u > 0 or v < 1 for u, v in local_slack):
                    continue
                candidates.append((sc, tc, endpoint, forward, lateral))
    assert len(candidates) == 1, "support direction and attachment side must be unique"
    sc, tc, endpoint, forward, lateral = candidates[0]
    copies = _field(mask, (top, left), sites, anchor)
    assert all(_inside(p, inp.shape) for ps in copies.values() for p in ps), "potential glyphs must fit"
    world_obstacles = set().union(*copies.values())
    assert sum(map(len, copies.values())) == len(world_obstacles), "copies cannot overlap"
    obstacles = {_local(p, endpoint, forward, lateral) for p in world_obstacles}
    route, hull = _hull_route(obstacles, len(cells[tc]))
    touches = {(u, v - 1) for u, v in route}
    active = {site for site, ps in copies.items()
              if any(_local(p, endpoint, forward, lateral) in touches for p in ps)}
    world_route = [_global(p, endpoint, forward, lateral) for p in route]
    return dict(bg=bg, ink=ink, marker=marker, support_color=sc, thread_color=tc,
                glyph=glyph, mask=mask, origin=(top, left), points=points,
                sites=sites, anchor=anchor, copies=copies, active=active,
                support=cells[sc], slack=cells[tc], endpoint=endpoint,
                forward=forward, lateral=lateral, obstacles=obstacles,
                route=route, world_route=world_route, hull=hull)


def _render(inp, info, active=None, route=None, invert=True):
    out = np.full_like(inp, info["bg"])
    for p in info["support"]:
        out[p] = info["support_color"]
    chosen = info["active"] if active is None else active
    for site in chosen:
        exceptional = site == info["anchor"]
        color = info["marker"] if exceptional == invert else info["ink"]
        for p in info["copies"][site]:
            out[p] = color
    world_route = info["world_route"] if route is None else route
    for p in world_route:
        if _inside(p, out.shape):
            assert out[p] == info["bg"], "thread must not collide with a peg or support"
            out[p] = info["thread_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer and solve from the input alone. No training, RNG or stored samples."""
    inp = _grid(input_grid)
    return _render(inp, _decode(inp))


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def _mask(rng, h, w):
    # Random opposite-corner spanning path, with independently sampled density.
    cells, r, c = {(0, 0)}, 0, 0
    steps = [0] * (h - 1) + [1] * (w - 1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        cells.add((r, c))
    target = int(rng.integers(len(cells), h * w + 1))
    while len(cells) < target:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIR4
                           if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
        cells.add(_pick(rng, frontier))
    mask = np.zeros((h, w), dtype=bool)
    for p in cells:
        mask[p] = True
    if rng.integers(2):
        mask = np.flipud(mask)
    if rng.integers(2):
        mask = np.fliplr(mask)
    return mask.copy()


def _slack(rng, length, diagonal=False):
    depth = int(rng.integers(2, 5))
    width = max(3, (length - 1 + depth - 1) // depth + 1)
    # A single diagonal link from the attachment is forced for one lesson.
    cells = {(0, 1)}
    if diagonal:
        cells.add((-1, 2))
        allowed = {(u, v) for u in range(-depth, 0) for v in range(2, width + 1)}
    else:
        cells.update({(-1, 1), (-1, 2)})
        allowed = {(u, v) for u in range(-depth, 1) for v in range(1, width + 1)}
    style = int(rng.integers(3))
    while len(cells) < length:
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in DIR8
                           if (u + du, v + dv) in allowed} - cells)
        assert frontier
        if style == 0:
            scores = np.array([1 + sum((u + du, v + dv) in cells for du, dv in DIR8)
                               for u, v in frontier], dtype=float)
        elif style == 1:
            scores = np.array([1 / (1 + sum((u + du, v + dv) in cells for du, dv in DIR8))
                               for u, v in frontier], dtype=float)
        else:
            scores = np.ones(len(frontier))
        cells.add(frontier[int(rng.choice(len(frontier), p=scores / scores.sum()))])
    return cells


def _contact_sites(copies, route):
    neighbors = {(u, v - 1) for u, v in route}
    return {site for site, cells in copies.items() if cells & neighbors}


def _make_pair(rng, lesson, map_limit, turn, clip, scale_shift=0):
    """Sample compatible structures, then place a blueprint within the canvas.

    Retries change geometry, not semantics. There is no bank of solved pairs.
    """
    for attempt in range(400):
        h, w = map(int, rng.integers(2, 4 if scale_shift < 0 else 5, size=2))
        if scale_shift > 0 and max(h, w) < 4:
            if rng.integers(2):
                h = 4
            else:
                w = 4
        mask = _mask(rng, h, w)
        if lesson in ("prefix", "long") and mask.all():
            continue
        if lesson == "prefix" and (h == w or any(np.array_equal(mask, other)
                                                  for other in (np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)))):
            continue
        mh, mw = map(int, rng.integers(2, map_limit + 1, size=2))
        if map_limit == 4 and max(mh, mw) < 4:
            if rng.integers(2):
                mh = 4
            else:
                mw = 4
        sites = {(r, c) for r in range(mh) for c in range(mw)
                 if rng.random() < float(rng.uniform(.35, .85))}
        # Actual map extent, not a padded generation window, determines indexing.
        sites.update({(0, int(rng.integers(mw))), (mh - 1, int(rng.integers(mw))),
                      (int(rng.integers(mh)), 0), (int(rng.integers(mh)), mw - 1)})
        if len(sites) < 4:
            continue
        # Whole blank lattice bands are not tested without demonstrations of
        # that extension. Local holes and disconnected layouts remain possible.
        if ({r for r, c in sites} != set(range(mh))
                or {c for r, c in sites} != set(range(mw))):
            continue
        if lesson in ("prefix", "long") and not all(any((r + (axis == 0), c + (axis == 1)) in sites
                                                       for r, c in sites) for axis in (0, 1)):
            continue
        raw_copies = _field(mask, (0, 0), sites, (0, 0))
        raw = set().union(*raw_copies.values())
        near = int(rng.integers(2, 6))
        box_tangent = max(c * w + w - 1 - r * h for r, c in sites)
        offset = near - box_tangent - int(rng.integers(0, 4))
        copies = {site: {(u + near, v + offset) for u, v in ps}
                  for site, ps in raw_copies.items()}
        obstacles = set().union(*copies.values())
        far = max(u for u, v in obstacles)
        if far > 21 or max(v for u, v in obstacles) < 2:
            continue
        long_length = min(25, far + int(rng.integers(3, 7)))
        long_route, hull = _hull_route(obstacles, long_length)
        full_active = _contact_sites(copies, long_route)
        # One field supplies exposed supports and genuinely buried candidates.
        if len(full_active) < 2 or len(full_active) == len(sites):
            continue
        first_contacts = sorted({u for u, v in long_route
                                 if (u, v - 1) in obstacles})
        if lesson == "prefix":
            lengths = [n for n in range(max(3, first_contacts[0] + 1), far + 1)
                       if 0 < len(_contact_sites(copies, long_route[:n])) < len(full_active)]
            if not lengths or mask.all():
                continue
            length = int(_pick(rng, lengths))
        elif lesson == "long":
            length = long_length
        else:
            lengths = list(range(far + 2 if clip else 3, long_length + 1))
            length = int(_pick(rng, lengths))
        route = long_route[:length]
        active = _contact_sites(copies, route)
        if not active:
            # Generated examples always show at least one peg; solve also
            # supports shorter valid family members with no activated peg.
            continue
        if lesson == "prefix":
            boxes = set().union(*({(u, v) for u in range(_bbox(ps)[0], _bbox(ps)[2] + 1)
                                  for v in range(_bbox(ps)[1], _bbox(ps)[3] + 1)}
                                 for ps in copies.values()))
            box_route, _ = _hull_route(boxes, length)
            cones = [(u, max(v + 1 - abs(u - t) for t, v in obstacles))
                     for u in range(length)]
            # Shape and collective bridging must both be observable, not just
            # theoretically present after the counted thread has already ended.
            if box_route == route or cones == route or _raster_hull(hull, length, False) == route:
                continue
        anchor_options = sorted(sites - active if lesson == "prefix" else active if lesson == "long" else sites)
        if lesson == "prefix":
            noncentral = [p for p in anchor_options if p != ((mh - 1) // 2, (mw - 1) // 2)]
            anchor_options = noncentral or anchor_options
        anchor = _pick(rng, anchor_options)
        glyph = copies[anchor]
        support_length = int(rng.integers(3, 6))
        support = {(-k, 0) for k in range(support_length)}
        slack = _slack(rng, length, diagonal=lesson == "prefix")
        input_body = support | slack | glyph
        base = input_body | obstacles
        if not clip:
            base |= set(route)
        bounds = _bbox(base)
        if bounds[2] - bounds[0] + 1 > 30 or bounds[3] - bounds[1] + 1 > 30:
            continue
        ph, pw = 2 * mh - 1, 2 * mw - 1
        glyph_box = _bbox(glyph)
        # Points can occupy future output locations, but never input material.
        forbidden = {(u + du, v + dv) for u, v in input_body for du, dv in DIR8 + ((0, 0),)}
        placements = []
        for br in range(bounds[0] - ph - 1, bounds[2] + 3):
            for bc in range(bounds[1] - pw - 1, bounds[3] + 3):
                r0, c0 = min(bounds[0], br), min(bounds[1], bc)
                r1, c1 = max(bounds[2], br + ph - 1), max(bounds[3], bc + pw - 1)
                if r1 - r0 >= 30 or c1 - c0 >= 30:
                    continue
                if clip and r1 >= length - 1:
                    continue  # outgoing clipping stays visible after placement
                if not (glyph_box[2] + 1 < br or br + ph < glyph_box[0]
                        or glyph_box[3] + 1 < bc or bc + pw < glyph_box[1]):
                    continue
                points = {(br + 2 * r, bc + 2 * c) for r, c in sites}
                if points & forbidden:
                    continue
                placements.append((br, bc, r0, c0, r1, c1))
        if not placements:
            continue
        # Favor compact placements without imposing a single diagram location.
        if rng.random() < .5:
            area = min((p[4] - p[2] + 1) * (p[5] - p[3] + 1) for p in placements)
            placements = [p for p in placements if (p[4] - p[2] + 1) * (p[5] - p[3] + 1) <= area + 30]
        br, bc, r0, c0, r1, c1 = _pick(rng, placements)
        margins = [0, 0, 0, 0]
        for axis, size in enumerate((r1 - r0 + 1, c1 - c0 + 1)):
            budget = min(30 - size, int(rng.integers(0, 5)))
            if axis == 0 and clip:
                margins[0] = budget
            else:
                before = int(rng.integers(budget + 1))
                margins[2 * axis:2 * axis + 2] = [before, budget - before]
        gh = r1 - r0 + 1 + margins[0] + margins[1]
        gw = c1 - c0 + 1 + margins[2] + margins[3]
        # Temporary role labels are permuted only after all geometry is built.
        inp = np.zeros((gh, gw), dtype=np.int8)
        shift = margins[0] - r0, margins[2] - c0
        for ps, color in ((support, 1), (slack, 2), (glyph, 3)):
            for u, v in ps:
                inp[u + shift[0], v + shift[1]] = color
        for r, c in sites:
            inp[br + 2 * r + shift[0], bc + 2 * c + shift[1]] = 3 if (r, c) == anchor else 4
        if turn is None:
            rotation = int(rng.integers(4))
        else:
            rotation = int(rng.choice((turn, turn + 2)))
        inp = np.rot90(inp, rotation)
        if rng.integers(2):
            inp = np.fliplr(inp)
        inp = inp.copy()
        try:
            info = _decode(inp)
        except AssertionError:
            continue
        out = _render(inp, info)
        if lesson == "long" and info["anchor"] not in info["active"]:
            continue
        if clip and not any(not _inside(p, inp.shape) for p in info["world_route"]):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded geometric construction exhausted")


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    train_turn = int(rng.integers(2)) if regime == 1 else None
    teach_clip = bool(rng.integers(2))
    fixed_palette = rng.random() < .2
    base_palette = np.array(rng.choice(10, 5, replace=False), dtype=np.int8)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = ("prefix" if index == 0 else "long" if index == 1 else "random") if split == "train" else "random"
            if split == "test" and regime == 0 and index == 0:
                lesson = "long"
            clip = (teach_clip and lesson == "long") if split == "train" else (teach_clip and rng.random() < .45)
            if split == "train" and index >= 2:
                clip = teach_clip and rng.random() < .3
            limit = 3 if split == "train" or regime != 0 else 4
            turn = train_turn if split == "train" or regime != 1 else 1 - train_turn
            scale_shift = (-1 if split == "train" else 1) if regime == 2 else 0
            for retry in range(30):
                pair = _make_pair(rng, lesson, limit, turn, clip, scale_shift)
                if split == "train" and index == 1:
                    previous = _decode(puzzle["train"][0]["input"])
                    current = _decode(pair["input"])
                    if (any(a == b for a, b in zip(previous["mask"].shape, current["mask"].shape))
                            or previous["anchor"] == current["anchor"]):
                        continue
                palette = base_palette if fixed_palette else np.array(rng.choice(10, 5, replace=False), dtype=np.int8)
                pair = {k: palette[a] for k, a in pair.items()}
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(inp, info=None):
    m = _decode(inp) if info is None else info
    far = max(u for u, v in m["obstacles"])
    full_route, _ = _hull_route(m["obstacles"], far + 3)
    local_copies = {s: {_local(p, m["endpoint"], m["forward"], m["lateral"]) for p in ps}
                    for s, ps in m["copies"].items()}
    full_active = _contact_sites(local_copies, full_route)
    boxes = set().union(*({(u, v) for u in range(_bbox(ps)[0], _bbox(ps)[2] + 1)
                          for v in range(_bbox(ps)[1], _bbox(ps)[3] + 1)}
                         for ps in local_copies.values()))
    box_route, _ = _hull_route(boxes, len(m["route"]))
    cones = [(u, max(v + 1 - abs(u - t) for t, v in m["obstacles"]))
             for u in range(len(m["route"]))]
    visible = sum(_inside(p, inp.shape) for p in m["world_route"])
    lost_edges = {edge for r, c in m["world_route"] for edge, flag in
                  (("top", r < 0), ("bottom", r >= inp.shape[0]),
                   ("left", c < 0), ("right", c >= inp.shape[1])) if flag}
    fraction = _raster_hull(m["hull"], len(m["route"]), False) != m["route"]
    return dict(shape=inp.shape, dimensions=m["mask"].shape, mask=tuple(map(int, m["mask"].flat)),
                map_shape=(max(r for r, c in m["sites"]) + 1, max(c for r, c in m["sites"]) + 1),
                sites=tuple(sorted(m["sites"])), anchor=m["anchor"], count=len(m["sites"]),
                anchor_active=m["anchor"] in m["active"],
                active=len(m["active"]), full_active=len(full_active), buried=len(m["sites"] - full_active),
                length=len(m["slack"]), visible=visible, clipped=visible < len(m["slack"]),
                edges=tuple(sorted(lost_edges)), forward=m["forward"], lateral=m["lateral"],
                diagonal_slack=len(_components(m["slack"], DIR4)) > 1,
                shape_matters=box_route != m["route"], bridges=cones != m["route"],
                fractional=fraction, palette=(m["bg"], m["support_color"], m["thread_color"], m["ink"], m["marker"]))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "repeated input"
            seen.add(_key(inp))
            m = _decode(inp)
            expected = _render(inp, m)
            assert np.array_equal(out, expected), "input-only semantic reconstruction failed"
            assert not np.array_equal(inp, out)
            assert all(out[p] == m["support_color"] for p in m["support"])
            assert out[_global((0, 1), m["endpoint"], m["forward"], m["lateral"])] == m["thread_color"]
            assert np.count_nonzero(out == m["thread_color"]) == sum(_inside(p, inp.shape) for p in m["world_route"])
            assert np.count_nonzero(out == m["ink"]) == len(m["active"] - {m["anchor"]}) * int(m["mask"].sum())
            assert np.count_nonzero(out == m["marker"]) == (m["anchor"] in m["active"]) * int(m["mask"].sum())
            # Cellwise inverse accounting distinguishes true glyph copying from
            # replacing masks by rectangles, regardless of joining components.
            for site, pixels in m["copies"].items():
                color = m["marker"] if site == m["anchor"] else m["ink"]
                assert all(out[p] == (color if site in m["active"] else m["bg"]) for p in pixels)
            metric = _measure(inp, m)
            assert 4 <= metric["count"] <= 16 and metric["active"] >= 1
            assert metric["buried"] >= 1, "field lacks a buried candidate"
            assert not np.array_equal(_render(inp, m, active=m["sites"], route=[]), out), "omitting A did not change answer"
            # Omitting B: route only around the actual source glyph, never
            # resolving the sparse map into the full potential field.
            single = {_local(p, m["endpoint"], m["forward"], m["lateral"]) for p in m["glyph"]}
            single_route, _ = _hull_route(single, len(m["slack"]))
            single_active = {m["anchor"]} if single & {(u, v - 1) for u, v in single_route} else set()
            alternative = _render(inp, m, active=single_active,
                                  route=[_global(p, m["endpoint"], m["forward"], m["lateral"]) for p in single_route])
            metric["B_matters"] = not np.array_equal(alternative, out)
            (training if split == "train" else testing).append(metric)
    first, second = training[:2]
    assert first["active"] < first["full_active"] and first["visible"] == first["length"]
    assert not first["anchor_active"] and second["anchor_active"]
    assert first["diagonal_slack"] and first["shape_matters"] and first["bridges"]
    assert first["fractional"]
    assert second["active"] == second["full_active"] and second["active"] >= 2
    assert all(a != b for a, b in zip(first["dimensions"], second["dimensions"]))
    assert first["anchor"] != second["anchor"]
    assert not all(first["mask"]) and not all(second["mask"])
    assert any(m["B_matters"] for m in training), "training does not expose B's contribution"
    assert any(m["clipped"] for m in training) or not any(m["clipped"] for m in testing), "untaught test clipping"


def self_test() -> None:
    """Bounded semantic regressions, 42-seed diversity, counts and corruption."""
    # A collective plateau bridges two separated pins; a buried middle pin is
    # not a contact. The tail departs immediately, without a far-face detour.
    obs = {(2, 2), (6, 2), (4, 0)}
    path, _ = _hull_route(obs, 9)
    assert path == [(0, 1), (1, 2), (2, 3), (3, 3), (4, 3),
                    (5, 3), (6, 3), (7, 2), (8, 1)]
    copies = {(0, 0): {(2, 2)}, (1, 0): {(4, 0)}, (2, 0): {(6, 2)}}
    assert _contact_sites(copies, path[:4]) == {(0, 0)}
    assert _contact_sites(copies, path) == {(0, 0), (2, 0)}
    # A fractional tangent is always rounded outward, including below zero.
    assert _hull_route({(3, 1)}, 8)[0] == [(0, 1), (1, 2), (2, 2), (3, 2),
                                         (4, 1), (5, 0), (6, -1), (7, -2)]
    assert _hull_route({(3, -5)}, 7)[0] == [(u, 1-u) for u in range(7)]
    puzzles, metrics = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                m = _measure(pair["input"])
                m["split"] = split
                metrics.append(m)
    signatures = {tuple(_key(pair[k]) for split in ("train", "test")
                        for pair in p[split] for k in ("input", "output")) for p in puzzles}
    assert len(signatures) == 42
    assert len({m["shape"] for m in metrics}) >= 45
    assert len({m["mask"] + m["dimensions"] for m in metrics}) >= 35
    assert len({(m["sites"], m["anchor"]) for m in metrics}) >= 65
    assert {m["forward"] for m in metrics} == {(-1, 0), (1, 0), (0, -1), (0, 1)}
    assert {m["dimensions"] for m in metrics} == {(h, w) for h in range(2, 5) for w in range(2, 5)}
    assert len({m["anchor"] for m in metrics}) >= 8
    assert len({m["count"] for m in metrics}) >= 6
    assert len({m["active"] for m in metrics}) >= 3
    assert len({m["length"] for m in metrics}) >= 12
    assert {m["palette"][0] for m in metrics} == set(range(10))
    assert {e for m in metrics for e in m["edges"]} == {"top", "bottom", "left", "right"}
    assert any(m["fractional"] for m in metrics if m["split"] == "train")
    assert any(m["map_shape"][0] == 4 or m["map_shape"][1] == 4 for m in metrics)
    again = generate_puzzle(9)
    assert all(np.array_equal(a[k], b[k]) for s in puzzles[9]
               for a, b in zip(puzzles[9][s], again[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            q = generate_puzzle(7000 + a * 10 + b, a, b)
            assert len(q["train"]) == a and len(q["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Reordering requests or solving unrelated examples cannot alter a query.
    query = puzzles[0]["test"][0]
    before = query["input"].copy()
    solve(puzzles[1]["test"][0]["input"])
    assert np.array_equal(solve(before), query["output"])
    assert np.array_equal(before, query["input"])
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in ps]
               for s, ps in puzzles[0].items()}
    m = _decode(damaged["train"][0]["input"])
    damaged["train"][0]["output"][min(m["support"])] = m["bg"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted support")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, every legal count combination)")
