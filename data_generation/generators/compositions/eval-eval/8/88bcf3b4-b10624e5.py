"""A taut thread reveals the scaled, reflected stencils it grazes.

The solver reads only pixels. The corner keys replace B's four-panel layout;
all latent stencils participate in ONE discrete convex roof from A's support.
"""
from __future__ import annotations

import numpy as np


D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return int(value)


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _connected(cells, directions=D8):
    if not cells:
        return False
    todo = [min(cells)]
    found = set(todo)
    while todo:
        r, c = todo.pop()
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in cells and p not in found:
                found.add(p)
                todo.append(p)
    return found == set(cells)


def _components(a, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    by_color = {}
    while remaining:
        p = min(remaining)
        color = int(a[p])
        cells, todo = {p}, [p]
        remaining.remove(p)
        while todo:
            r, c = todo.pop()
            for dr, dc in D8:
                q = r + dr, c + dc
                if q in remaining and int(a[q]) == color:
                    remaining.remove(q)
                    cells.add(q)
                    todo.append(q)
        by_color.setdefault(color, []).append(cells)
    return by_color


def _local(p, end, f, lateral):
    r, c = p[0] - end[0], p[1] - end[1]
    return r*f[0] + c*f[1], r*lateral[0] + c*lateral[1]


def _global(p, end, f, lateral):
    u, v = p
    return end[0] + u*f[0] + v*lateral[0], end[1] + u*f[1] + v*lateral[1]


def _square(cells):
    u0, v0 = min(p[0] for p in cells), min(p[1] for p in cells)
    u1, v1 = max(p[0] for p in cells), max(p[1] for p in cells)
    side = u1-u0+1
    if side != v1-v0+1 or len(cells) != side*side:
        return None
    return u0, v0, side


def _scaled(motif, source_side, target_side, flips):
    """Exact rational rasterization, including boundaries inside an anchor.

    Refine each source pixel into target_side^2 microcells and combine
    source_side^2 of those per target pixel. No color boundary is rounded.
    Coordinates are relative to the TOP LEFT of the respective anchor.
    """
    rs, cs = zip(*motif)
    top = min(rs)//source_side*source_side
    left = min(cs)//source_side*source_side
    bottom = (max(rs)//source_side+1)*source_side
    right = (max(cs)//source_side+1)*source_side
    # -1 is transparent background, independent of the literal palette.
    raw = np.full((bottom-top, right-left), -1, dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r-top, c-left] = color
    expanded = np.repeat(np.repeat(raw, target_side, 0), target_side, 1)
    h, w = expanded.shape[0]//source_side, expanded.shape[1]//source_side
    cells = expanded.reshape(h, source_side, w, source_side)
    low, high = cells.min((1, 3)), cells.max((1, 3))
    assert np.array_equal(low, high), "ambiguous fractional stencil boundary"
    offset_u = top*target_side//source_side
    offset_v = left*target_side//source_side
    result = {}
    for r, c in np.argwhere(low != -1):
        u, v = int(r)+offset_u, int(c)+offset_v
        if flips[0]:
            u = target_side-1-u
        if flips[1]:
            v = target_side-1-v
        result[u, v] = int(low[r, c])
    return result


def _roof(obstacles, length):
    """Least roof in the family's +1,0,-1 turning vocabulary.

    A >0 peak requires a diagonal incoming tangent (max(v-u)==0).
    Peak zero uses a parallel incoming tangent. The three support lines
    depend on the entire constellation, including unrevealed stencils.
    """
    assert obstacles and min(u for u, v in obstacles) >= 1
    peak = max(v for u, v in obstacles)
    tangent = max(v-u for u, v in obstacles)
    far_sum = max(u+v for u, v in obstacles)
    assert peak >= 0 and tangent <= 0 and (peak == 0 or tangent == 0)
    assert far_sum >= 2*peak
    return [(u, min(u+1 if peak else 1, peak+1, far_sum+1-u))
            for u in range(length)]


def _decode(a):
    """Find a unique support/strand plus square-and-corner-key grammar."""
    hist = np.bincount(a.ravel().astype(int), minlength=10)
    assert np.count_nonzero(hist == hist.max()) == 1
    bg = int(hist.argmax())
    comps = _components(a, bg)
    candidates = []
    for support_color, groups in comps.items():
        if len(groups) != 1:
            continue
        support = groups[0]
        if not 3 <= len(support) <= 8:
            continue
        if len({p[0] for p in support}) == 1:
            direction = (0, 1)
        elif len({p[1] for p in support}) == 1:
            direction = (1, 0)
        else:
            continue
        for end, f in ((max(support), direction),
                       (min(support), (-direction[0], -direction[1]))):
            for lateral in ((f[1], -f[0]), (-f[1], f[0])):
                attachment = _global((0, 1), end, f, lateral)
                for strand_color, strands in comps.items():
                    if strand_color == support_color or len(strands) != 1:
                        continue
                    strand = strands[0]
                    if attachment not in strand or len(strand) < 2:
                        continue
                    slack = {_local(p, end, f, lateral) for p in strand}
                    if any(u > 0 or v < 1 for u, v in slack):
                        continue
                    local_groups = {color: [{_local(p, end, f, lateral) for p in cells}
                                             for cells in parts]
                                    for color, parts in comps.items()
                                    if color not in (support_color, strand_color)}
                    for anchor_color, anchors in local_groups.items():
                        if not 2 <= len(anchors) <= 5:
                            continue
                        squares = [_square(cells) for cells in anchors]
                        if any(s is None or s[0] < 1 or s[2] > 6 for s in squares):
                            continue
                        for marker_color, dots in local_groups.items():
                            if marker_color == anchor_color or len(dots) != len(anchors):
                                continue
                            if any(len(dot) != 1 for dot in dots):
                                continue
                            remaining = {next(iter(dot)) for dot in dots}
                            sites = []
                            valid = True
                            for square, cells in zip(squares, anchors):
                                u, v, side = square
                                corners = [(u-1, v-1), (u-1, v+side),
                                           (u+side, v-1), (u+side, v+side)]
                                matches = [i for i, p in enumerate(corners) if p in remaining]
                                if len(matches) != 1:
                                    valid = False
                                    break
                                corner = matches[0]
                                remaining.remove(corners[corner])
                                sites.append({"top": (u, v), "side": side, "corner": corner,
                                              "dot": corners[corner], "anchor_cells": cells})
                            if not valid or remaining:
                                continue
                            detail = {p: color for color, groups2 in local_groups.items()
                                      if color not in (anchor_color, marker_color)
                                      for cells in groups2 for p in cells}
                            if not detail or len(set(detail.values())) > 3:
                                continue
                            refs = [i for i, site in enumerate(sites)
                                    if _connected(set(detail) | site["anchor_cells"], D4)]
                            if len(refs) != 1:
                                continue
                            ref = refs[0]
                            origin = sites[ref]["top"]
                            motif = {(u-origin[0], v-origin[1]): anchor_color
                                     for u, v in sites[ref]["anchor_cells"]}
                            motif.update({(u-origin[0], v-origin[1]): color
                                          for (u, v), color in detail.items()})
                            try:
                                shadows = []
                                for site in sites:
                                    xor = sites[ref]["corner"] ^ site["corner"]
                                    tile = _scaled(motif, sites[ref]["side"], site["side"],
                                                   (bool(xor & 2), bool(xor & 1)))
                                    shadows.append({(u+site["top"][0], v+site["top"][1]): color
                                                    for (u, v), color in tile.items()})
                                union = set().union(*(set(tile) for tile in shadows))
                                assert sum(map(len, shadows)) == len(union)
                                assert all(_connected(tile, D4) for tile in shadows)
                                path = _roof(union, len(strand))
                                assert not set(path) & union
                                assert not set(path) & {s["dot"] for s in sites}
                                assert not union & {s["dot"] for s in sites}
                            except AssertionError:
                                continue
                            candidates.append({"bg": bg, "support_color": support_color,
                                               "strand_color": strand_color, "anchor_color": anchor_color,
                                               "marker_color": marker_color, "support": support,
                                               "strand": strand, "slack": slack, "end": end,
                                               "f": f, "lateral": lateral, "sites": sites,
                                               "reference": ref, "motif": motif, "shadows": shadows,
                                               "union": union, "path": path})
    assert len(candidates) == 1, f"expected unique pixel grammar, found {len(candidates)}"
    return candidates[0]


def _contact(tile, path):
    return any((u, v-1) in tile for u, v in path)


def _bare_path(data, length):
    """Diagnostic: apply the roof formula to anchors and visible detail only."""
    bare = set().union(*(s["anchor_cells"] for s in data["sites"]))
    bare |= set(data["shadows"][data["reference"]])
    peak = max(0, max(v for u, v in bare))
    far_sum = max(0, max(u+v for u, v in bare))
    return [(u, min(u+1 if peak else 1, peak+1, far_sum+1-u)) for u in range(length)]


def _render(a, data, length=None, anchors_only=False, reveal_all=False):
    out = a.copy()
    for p in data["strand"]:
        out[p] = data["bg"]
    n = len(data["strand"]) if length is None else length
    path = _bare_path(data, n) if anchors_only else _roof(data["union"], n)
    active = []
    for i, tile in enumerate(data["shadows"]):
        touched = _contact(tile, path)
        if i == data["reference"] or (not anchors_only and (touched or reveal_all)):
            active.append(i)
            for p, color in tile.items():
                r, c = _global(p, data["end"], data["f"], data["lateral"])
                if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                    out[r, c] = color
    visible = []
    for p in path:
        r, c = _global(p, data["end"], data["f"], data["lateral"])
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            if length is None and not anchors_only:
                assert out[r, c] == data["bg"], "thread collides with preserved material"
            # Diagnostics are allowed to cross shadows, but never erase keys.
            if out[r, c] in (data["bg"], data["strand_color"]):
                out[r, c] = data["strand_color"]
            visible.append(p)
    return out, active, visible


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _array(input_grid)
    return _render(a, _decode(a))[0]


def _motif(rng, compact):
    """Small connected stencils with independent arms, indents and fine detail."""
    a = 1 if compact or rng.random() < .75 else 2
    tile = {(u, v): 0 for u in range(a) for v in range(a)}
    if compact:
        arm1, arm2 = int(rng.integers(1, 3)), 1
        for u in range(a, a+arm1):
            tile[u, 0] = 1
        for v in range(a, a+arm2):
            tile[0, v] = 2
    else:
        style = int(rng.integers(4))
        steps = int(rng.integers(2, 7))
        for j in range(steps):
            frontier = sorted({(u+du, v+dv) for u, v in tile for du, dv in D4
                               if -2 <= u+du <= a+2 and -2 <= v+dv <= a+2} - set(tile))
            if style == 0:
                frontier.sort(key=lambda p: (-p[0], abs(p[1])))
                p = frontier[int(rng.integers(min(3, len(frontier))))]
            else:
                p = frontier[int(rng.integers(len(frontier)))]
            tile[p] = 1 + j % 2
        if style == 2:
            for v in range(a, a+3):
                tile[a-1, v] = 2
    # Whole-stencil orientation is independent of the keys.
    for _ in range(int(rng.integers(4))):
        tile = {(v, a-1-u): color for (u, v), color in tile.items()}
    if rng.random() < .5:
        tile = {(u, a-1-v): color for (u, v), color in tile.items()}
    return tile, a


def _place_world(rng, count, witness=False):
    """Construct a jointly constrained constellation, never parent puzzle copies."""
    for attempt in range(500):
        motif, a = _motif(rng, compact=witness or attempt > 300)
        units = [int(rng.integers(1, 4 if a == 1 else 3)) for _ in range(count)]
        if witness:
            units = [1, 1, 2, 2]
            rng.shuffle(units)
        corners = list(map(int, rng.permutation(4)))
        corners += [int(rng.integers(4)) for _ in range(max(0, count-4))]
        sites, shadows, occupied = [], [], set()
        possible = True
        for i in range(count):
            unit, corner = units[i], corners[i]
            # motif is represented with NW as its reference key.
            tile = _scaled(motif, a, a*unit, (bool(corner & 2), bool(corner & 1)))
            for _ in range(90):
                origin = int(rng.integers(2, 17)), int(rng.integers(-5, 10))
                shifted = {(u+origin[0], v+origin[1]): color for (u, v), color in tile.items()}
                side = a*unit
                dot = (origin[0] + (side if corner & 2 else -1),
                       origin[1] + (side if corner & 1 else -1))
                footprint = set(shifted) | {dot}
                if min(u for u, v in footprint) < 1 or max(u for u, v in footprint) > 18:
                    continue
                # Separate even the invisible stencils and all visible keys.
                if footprint & occupied:
                    continue
                sites.append({"top": origin, "side": side, "corner": corner, "dot": dot})
                shadows.append(shifted)
                occupied.update((u+du, v+dv) for u, v in footprint
                                for du in (-1, 0, 1) for dv in (-1, 0, 1))
                break
            else:
                possible = False
                break
        if not possible:
            continue
        union = set().union(*(set(t) for t in shadows))
        # Tangency aligns the constellation, not the individual stations.
        delta = -max(v-u for u, v in union)
        shadows = [{(u, v+delta): color for (u, v), color in tile.items()} for tile in shadows]
        sites = [{**s, "top": (s["top"][0], s["top"][1]+delta),
                  "dot": (s["dot"][0], s["dot"][1]+delta)} for s in sites]
        union = set().union(*(set(t) for t in shadows))
        peak = max(v for u, v in union)
        if not 1 <= peak <= 11:
            continue
        path = _roof(union, 26)
        if set(path) & {s["dot"] for s in sites}:
            continue
        contacts = [min((u for u, v in path if (u, v-1) in tile), default=99)
                    for tile in shadows]
        touching = sorted((u, i) for i, u in enumerate(contacts) if u != 99)
        if witness:
            if len(touching) != 3 or len({u for u, i in touching}) != 3:
                continue
            if touching[2][0]-touching[1][0] < 2:
                continue
        elif not touching:
            continue
        # Full departing route must fit with at least three support cells.
        long_length = max(touching[-1][0]+3, max(u for u, v in union)+3)
        if long_length > 23:
            continue
        short_length = (touching[1][0]+1 if witness else
                        max(2, touching[0][0]+int(rng.integers(1, 4))))
        return {"sites": sites, "shadows": shadows, "contacts": contacts,
                "touching": touching, "long": long_length, "short": short_length,
                "a": a, "motif": motif}
    return _fallback_world(count)


def _fallback_world(count=4):
    """A small same-rule geometric fallback, not a bank of finished puzzles.

    Three outer stencils provide the three supporting lines. Remaining sites
    lie under that roof. The four-site version supplies the core curriculum.
    """
    motif = {(0, 0): 0, (1, 0): 1, (0, -1): 2, (0, -2): 2}
    definitions = [((6, 3), 2, 2), ((10, 5), 1, 1), ((15, 5), 1, 3),
                   ((10, -5), 2, 0), ((4, -5), 1, 0)]
    if count == 2:
        definitions = [definitions[0], definitions[2]]
    else:
        definitions = definitions[:count]
    sites, shadows = [], []
    for top, side, corner in definitions:
        dot = (top[0] + (side if corner & 2 else -1),
               top[1] + (side if corner & 1 else -1))
        sites.append({"top": top, "side": side, "corner": corner, "dot": dot})
        tile = _scaled(motif, 1, side, (bool(corner & 2), bool(corner & 1)))
        shadows.append({(u+top[0], v+top[1]): color for (u, v), color in tile.items()})
    union = set().union(*(set(t) for t in shadows))
    path = _roof(union, 26)
    contacts = [min((u for u, v in path if (u, v-1) in tile), default=99)
                for tile in shadows]
    touching = sorted((u, i) for i, u in enumerate(contacts) if u != 99)
    return {"sites": sites, "shadows": shadows, "contacts": contacts,
            "touching": touching, "long": 18, "short": 11, "a": 1, "motif": motif}


def _slack(rng, length, depth, width):
    # The attachment is linked diagonally to the rest: four-connectivity
    # cannot count the budget. The remaining mass can be sparse or dense.
    cells, order = {(0, 1), (-1, 2)}, [(0, 1), (-1, 2)]
    sparse = rng.random() < .5
    while len(cells) < length:
        frontier = sorted({(u+du, v+dv) for u, v in cells for du, dv in D8
                           if -depth <= u+du <= -1 and 2 <= v+dv <= width} - cells)
        assert frontier
        scores = [sum((u+du, v+dv) in cells for du, dv in D8) for u, v in frontier]
        weights = np.array([1/(s+1) if sparse else (s+1)**2 for s in scores])
        p = frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))]
        cells.add(p)
        order.append(p)
    return order


def _build_pair(rng, world, reference, length, palette, transform, clip=False):
    support_length = int(rng.integers(4 if clip else 3, min(8, 30-length)+1))
    depth = min(support_length-1, 5)
    width = max(4, (length+depth-1)//depth+1)
    slack = set(_slack(rng, length, depth, width))
    support = {(-u, 0) for u in range(support_length)}
    union = set().union(*(set(t) for t in world["shadows"]))
    path = _roof(union, length)
    material = support | slack | union | {s["dot"] for s in world["sites"]}
    full = material | set(path)
    u0, u1 = min(u for u, v in full), max(u for u, v in full)
    v0, v1 = min(v for u, v in full), max(v for u, v in full)
    if clip:
        # All stencils remain intact; only exterior thread clearance clips.
        v1 = max(v for u, v in material)
        assert any(v > v1 for u, v in path), "scene has no clippable roof clearance"
    assert u1-u0+1 <= 30 and v1-v0+1 <= 30
    margins = [int(rng.integers(0, 3)) for _ in range(4)]
    if clip:
        margins[3] = 0
    while u1-u0+1+margins[0]+margins[1] > 30:
        margins[0 if margins[0] else 1] -= 1
    while v1-v0+1+margins[2]+margins[3] > 30:
        margins[2 if margins[2] else 3] -= 1
    assert u1-u0+1 <= 30 and v1-v0+1 <= 30
    end = margins[0]-u0, margins[2]-v0
    inp = np.full((u1-u0+1+sum(margins[:2]), v1-v0+1+sum(margins[2:])), palette[0], dtype=np.int8)
    def put(p, role):
        q = p[0]+end[0], p[1]+end[1]
        assert inp[q] == palette[0]
        inp[q] = palette[role]
    for p in support:
        put(p, 1)
    for p in slack:
        put(p, 2)
    for i, site in enumerate(world["sites"]):
        top, side = site["top"], site["side"]
        for u in range(top[0], top[0]+side):
            for v in range(top[1], top[1]+side):
                put((u, v), 3)
        put(site["dot"], 4)
        if i == reference:
            for p, role in world["shadows"][i].items():
                if role:
                    put(p, 4+role)
    # Construction truth uses the world; it does not invoke solve/decoder.
    out = inp.copy()
    for u, v in slack:
        out[u+end[0], v+end[1]] = palette[0]
    for i, tile in enumerate(world["shadows"]):
        if i == reference or _contact(tile, path):
            for (u, v), role in tile.items():
                out[u+end[0], v+end[1]] = palette[3 if role == 0 else 4+role]
    for u, v in path:
        r, c = u+end[0], v+end[1]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            assert out[r, c] == palette[0]
            out[r, c] = palette[2]
    rotations, mirror = transform
    for _ in range(rotations):
        inp, out = np.rot90(inp), np.rot90(out)
    if mirror:
        inp, out = inp[:, ::-1], out[:, ::-1]
    return {"input": inp.copy(), "output": out.copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    train_transform = int(rng.integers(4)), bool(rng.integers(2))
    # Two demonstrations share latent geometry but move the visible source.
    # Three contacting keys give all three pairwise reflection relationships.
    for attempt in range(80):
        world = _fallback_world(4) if attempt == 79 else _place_world(rng, 4, witness=True)
        palette = list(map(int, rng.permutation(10)))
        late = world["touching"][-1][1]
        early = world["touching"][0][1]
        first = _build_pair(rng, world, late, world["long"], palette, train_transform)
        second = _build_pair(rng, world, early, world["short"], palette, train_transform)
        try:
            for pair in (first, second):
                d = _decode(pair["input"])
                assert np.array_equal(_render(pair["input"], d)[0], pair["output"])
            assert _witnesses([first, second])
        except AssertionError:
            continue
        puzzle["train"].extend((first, second))
        seen.update(_key(p["input"]) for p in (first, second))
        break
    else:
        raise AssertionError("bounded demonstration construction exhausted")
    # With >=3 demonstrations, teach clipping before permitting clipped tests.
    clip_taught = num_train >= 3 and rng.random() < .65
    for split, count in (("train", num_train-2), ("test", num_test)):
        for index in range(count):
            clip = (split == "train" and index == 0 and clip_taught) or (
                split == "test" and clip_taught and rng.random() < .6)
            for attempt in range(200):
                count_sites = int(rng.integers(2, 6))
                fallback = attempt >= 180
                w = _fallback_world(count_sites) if fallback else _place_world(rng, count_sites)
                reference = w["touching"][0][1] if fallback else int(rng.integers(count_sites))
                length = w["long"] if fallback or rng.random() < .7 or clip else w["short"]
                palette2 = list(map(int, rng.permutation(10)))
                if fallback:
                    # Twenty distinct backgrounds/anchor assignments outnumber
                    # the at most thirteen previous pairs. Other roles remain
                    # distinct, so a finite fallback cannot repeat an input.
                    bg = (attempt-180)//2
                    anchor = (bg+1+(attempt % 2)) % 10
                    rest = [c for c in range(10) if c not in (bg, anchor)]
                    palette2 = [bg, rest[0], rest[1], anchor, rest[2], rest[3], rest[4]]
                # Test shift: perpendicular support, with familiar keyed
                # reflections/scales recombined in a new constellation.
                transform = ((train_transform[0]+1+2*int(rng.integers(2))) % 4,
                             bool(rng.integers(2))) if split == "test" else train_transform
                try:
                    pair = _build_pair(rng, w, reference, length, palette2, transform, clip)
                    d = _decode(pair["input"])
                    predicted, active, visible = _render(pair["input"], d)
                    assert any(i != d["reference"] for i in active)
                    assert np.array_equal(predicted, pair["output"])
                    assert _key(pair["input"]) not in seen
                    assert (len(visible) < length) == clip
                except AssertionError:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("bounded pair construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(pairs):
    flips = set()
    long_witness = short_witness = shadow_witness = scaled = route_witness = False
    references = set()
    for pair in pairs:
        d = _decode(pair["input"])
        _, active, visible = _render(pair["input"], d)
        ref = d["sites"][d["reference"]]
        references.add(ref["corner"])
        route_witness |= _bare_path(d, len(d["strand"])) != d["path"]
        for i in active:
            if i != d["reference"]:
                flips.add(ref["corner"] ^ d["sites"][i]["corner"])
                scaled |= ref["side"] != d["sites"][i]["side"]
        far = max(u for u, v in d["union"])
        long_witness |= len(d["path"]) > far+1 and len(visible) == len(d["strand"])
        short_witness |= len(d["path"]) <= far and len(visible) == len(d["strand"])
        all_path = _roof(d["union"], 35)
        shadow_witness |= any(i != d["reference"] and not _contact(tile, all_path)
                              for i, tile in enumerate(d["shadows"]))
    return ({1, 2, 3} <= flips and len(references) >= 2 and scaled and long_witness
            and short_witness and shadow_witness and route_witness)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, clipping = set(), {"train": False, "test": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _array(pair["input"]), _array(pair["output"])
            assert a.shape == b.shape and not np.array_equal(a, b)
            assert _key(a) not in seen
            seen.add(_key(a))
            d = _decode(a)
            expected, active, visible = _render(a, d)
            assert np.array_equal(expected, b), "wrong contact-gated completion/roof"
            assert any(i != d["reference"] for i in active)
            clipping[split] |= len(visible) < len(d["strand"])
            # Preservation and exact count are checked from actual pixels.
            for p in d["support"]:
                assert b[p] == d["support_color"]
            for site in d["sites"]:
                for p in site["anchor_cells"]:
                    assert b[_global(p, d["end"], d["f"], d["lateral"])] == d["anchor_color"]
                assert b[_global(site["dot"], d["end"], d["f"], d["lateral"])] == d["marker_color"]
            assert int(np.count_nonzero(b == d["strand_color"])) == len(visible)
            assert (0, 1) in visible
            assert not set(d["path"]) & d["union"]
            slopes = [q[1]-p[1] for p, q in zip(d["path"], d["path"][1:])]
            assert set(slopes) <= {-1, 0, 1} and slopes == sorted(slopes, reverse=True)
    assert _witnesses(puzzle["train"]), "missing observable curriculum witnesses"
    assert not clipping["test"] or clipping["train"], "clipping not demonstrated"
    # Concrete causal ablations: count/gating and virtual stencil geometry.
    assert any(not np.array_equal(_render(_array(p["input"]), _decode(_array(p["input"])),
                                          length=35, reveal_all=True)[0], p["output"])
               for p in puzzle["train"])
    assert any(not np.array_equal(_render(_array(p["input"]), _decode(_array(p["input"])),
                                          anchors_only=True)[0], p["output"])
               for p in puzzle["train"])


def _puzzle_key(p):
    return tuple(tuple((_key(q["input"]), _key(q["output"])) for q in p[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed audit, causal checks, exact-scaling and corruption checks."""
    shapes, structures, site_counts, ratios, corners, lengths, motifs, axes = (set() for _ in range(8))
    backgrounds, clipped, hidden, palettes, puzzles = set(), 0, 0, set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(_puzzle_key(p))
        if seed < 3:
            assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                d = _decode(a)
                b, active, visible = _render(a, d)
                assert np.array_equal(solve(a.copy()), pair["output"])
                assert np.array_equal(a, pair["input"])
                shapes.add(a.shape)
                color_map = {}
                structures.add(tuple(color_map.setdefault(int(x), len(color_map)) for x in a.ravel()))
                site_counts.add(len(d["sites"]))
                ref = d["sites"][d["reference"]]
                ratios.update((s["side"], ref["side"]) for s in d["sites"])
                corners.update(s["corner"] for s in d["sites"])
                lengths.add(len(d["strand"]))
                motifs.add(tuple(sorted((u, v, 0 if c == d["anchor_color"] else 1)
                                        for (u, v), c in d["motif"].items())))
                axes.add((split, bool(d["f"][0])))
                backgrounds.add(d["bg"])
                palettes.add((d["bg"], d["support_color"], d["strand_color"], d["anchor_color"], d["marker_color"]))
                clipped += len(visible) < len(d["strand"])
                hidden += len(active) < len(d["sites"])
    assert len(puzzles) == 42 and len(shapes) >= 35 and len(structures) >= 180
    assert site_counts == {2, 3, 4, 5} and corners == {0, 1, 2, 3}
    assert len(ratios) >= 8 and len(lengths) >= 12 and len(motifs) >= 35
    assert len(axes) == 4 and len(backgrounds) == 10 and len(palettes) >= 100
    assert clipped >= 8 and hidden >= 84
    # All legal explicit counts, plus type and range errors.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            q = generate_puzzle(91+nt*4+nq, nt, nq)
            assert len(q["train"]) == nt and len(q["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Scaling rejects unresolved finer-than-target color boundaries.
    try:
        _scaled({(0, 0): 1, (0, 1): 2, (1, 0): 1, (1, 1): 1}, 2, 1, (False, False))
    except AssertionError:
        pass
    else:
        raise AssertionError("fractional boundary rounded")
    assert _roof({(2, 2), (5, 2)}, 8) == [(0, 1), (1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 2), (7, 1)]
    assert _roof({(2, 0)}, 6) == [(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -2)]
    # Exercise all fallback station counts and clipping explicitly.
    for count in range(2, 6):
        w = _fallback_world(count)
        for clip in (False, True):
            q = _build_pair(np.random.default_rng(121), w, w["touching"][0][1],
                            w["long"], list(range(10)), (1, True), clip)
            assert np.array_equal(solve(q["input"]), q["output"])
    w = _fallback_world(4)
    lessons = [_build_pair(np.random.default_rng(122), w, w["touching"][-1][1],
                           w["long"], list(range(10)), (0, False)),
               _build_pair(np.random.default_rng(123), w, w["touching"][0][1],
                           w["short"], list(range(10)), (0, False))]
    assert _witnesses(lessons)
    # Full input-only prediction for a parallel approach, not just a profile.
    parallel = _fallback_world(2)
    parallel["shadows"] = [{(u, v-7): c for (u, v), c in tile.items()}
                           for tile in parallel["shadows"]]
    parallel["sites"] = [{**s, "top": (s["top"][0], s["top"][1]-7),
                           "dot": (s["dot"][0], s["dot"][1]-7)} for s in parallel["sites"]]
    q = _build_pair(np.random.default_rng(0), parallel, 0, 18, list(range(10)), (0, False))
    assert np.array_equal(solve(q["input"]), q["output"])
    # A destination detail may clip while its anchor and key remain intact.
    # Contacts are conceptual: the hidden outgoing stencil still activates.
    q = _build_pair(np.random.default_rng(0), _fallback_world(2), 0, 18,
                    list(range(10)), (0, False))
    d = _decode(q["input"])
    cut = d["end"][1]+7  # preserve local columns through v=6, omit v>=7
    assert np.all(q["input"][:, cut:] == d["bg"])
    assert np.array_equal(solve(q["input"][:, :cut]), q["output"][:, :cut])
    damaged = generate_puzzle(83, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "shapes": len(shapes),
            "structures": len(structures), "motifs": len(motifs), "site_counts": sorted(site_counts),
            "anchor_size_pairs": sorted(ratios), "lengths": sorted(lengths),
            "clipped_pairs": clipped, "partly_revealed_pairs": hidden}
