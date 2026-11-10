"""Stamp a foreground inventory, then frequency-pack it in bottom-up columns.

The solver deliberately builds the intermediate stamped canvas. The generator's
labels use the equivalent inventory equation; validation checks both descriptions.
No parent modules, files, seeds, or generated examples are used by solve().
"""

from __future__ import annotations

import hashlib

import numpy as np


HOST_SEEDS = (
    0, 164867350, 22341117, 806554981, 406237529, 496168784,
    840147096, 979057464, 183745685, 386099143, 857147043, 547197277,
    41494507, 817115773, 117466506, 757741876, 917498976, 576731315,
    874038950, 321440708, 47142505, 640209368, 317338943, 413177581,
    611346663, 823657546, 219945401, 826524380, 777772698, 580525723,
    313288649, 981886313, 883095891, 212874983, 500433306, 506607648,
    950178198, 211374156, 523798527, 360885905, 733375587, 401373579,
)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(grid, color):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    component.append(p)
                    stack.append(p)
        result.append(component)
    return result


def _decode(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("three roles and a uniquely most frequent input background required")
    bg = int(colors[np.argmax(counts)])
    parts = {int(c): _components(grid, c) for c in colors if c != bg}
    candidates = [c for c, cs in parts.items()
                  if len(cs) == 2 and sorted(map(len, cs))[0] == 1
                  and sorted(map(len, cs))[1] >= 3]
    if len(candidates) != 1:
        raise ValueError("template must be one connected object plus one isolated anchor")
    ink = candidates[0]
    marker = next(c for c in parts if c != ink)
    if len(parts[marker]) < 2 or any(len(c) != 1 for c in parts[marker]):
        raise ValueError("at least two isolated ordinary sites required; ties are excluded")
    glyph = np.array(next(c for c in parts[ink] if len(c) > 1), dtype=int)
    anchor_pixel = next(c[0] for c in parts[ink] if len(c) == 1)
    tr, tc = map(int, glyph.min(axis=0))
    br, bc = map(int, glyph.max(axis=0))
    h, w = br-tr+1, bc-tc+1
    if min(h, w) < 2:
        raise ValueError("template must span at least two rows and two columns")
    mask = grid[tr:br+1, tc:bc+1] == ink
    points = [anchor_pixel] + [c[0] for c in parts[marker]]
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    if not (br+1 < pr or er+1 < tr or bc+1 < pc or ec+1 < tc):
        raise ValueError("map and template bounding boxes need a blank separating row or column")
    if any((r-pr) % 2 or (c-pc) % 2 for r, c in points):
        raise ValueError("all sites must lie on the fixed pitch-two lattice")
    mh, mw = (er-pr)//2+1, (ec-pc)//2+1
    layout = np.zeros((mh, mw), dtype=bool)
    for r, c in points:
        layout[(r-pr)//2, (c-pc)//2] = True
    anchor = ((anchor_pixel[0]-pr)//2, (anchor_pixel[1]-pc)//2)
    origin = tr-anchor[0]*h, tc-anchor[1]*w
    if not (0 <= origin[0] and origin[0]+mh*h <= grid.shape[0]
            and 0 <= origin[1] and origin[1]+mw*w <= grid.shape[1]):
        raise ValueError("the entire anchor-relative assembly must fit without clipping")
    return {"background": bg, "ink": ink, "marker": marker,
            "mask": mask, "layout": layout, "anchor": anchor,
            "template_top": (tr, tc), "origin": origin,
            "map_points": tuple(points), "population": int(mask.sum()),
            "sites": len(points)}


def _stamp(shape, meta):
    """A's exact unrotated, anchor-relative substitution, starting from blank."""
    out = np.full(shape, meta["background"], dtype=np.int8)
    h, w = meta["mask"].shape
    rr, cc = np.nonzero(meta["mask"])
    top, left = meta["origin"]
    for r, c in np.argwhere(meta["layout"]):
        color = meta["marker"] if (int(r), int(c)) == meta["anchor"] else meta["ink"]
        out[top+r*h+rr, left+c*w+cc] = color
    return out


def _pack_counts(shape, background, histogram):
    ranked = sorted(histogram.items(), key=lambda p: -p[1])
    if (not ranked or len({n for _, n in ranked}) != len(ranked)
            or any(n <= 0 or c == background for c, n in ranked)
            or sum(n for _, n in ranked) > shape[0]*shape[1]):
        raise ValueError("positive, distinct foreground counts fitting the canvas required")
    stream = np.full(shape[0]*shape[1], background, dtype=np.int8)
    pos = 0
    for color, count in ranked:
        stream[pos:pos+count] = color
        pos += count
    return stream.reshape(shape[1], shape[0])[:, ::-1].T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    meta = _decode(grid)
    stamped = _stamp(grid.shape, meta)
    colors, counts = np.unique(stamped[stamped != meta["background"]], return_counts=True)
    return _pack_counts(grid.shape, meta["background"],
                        {int(c): int(n) for c, n in zip(colors, counts)})


def _inventory(meta):
    m, n = meta["population"], meta["sites"]
    return {meta["ink"]: m*(n-1), meta["marker"]: m}


def _walk_pack(shape, background, histogram, major="column", reverse_major=False,
               reverse_minor=True, descending=True):
    """A direct cell walk for validation and explicit alternative-rule checks."""
    h, w = shape
    out = np.full(shape, background, dtype=np.int8)
    ranked = sorted(histogram.items(), key=lambda p: (-p[1] if descending else p[1], p[0]))
    stream = (color for color, n in ranked for _ in range(n))
    majors = list(range(w if major == "column" else h))
    minors = list(range(h if major == "column" else w))
    if reverse_major:
        majors.reverse()
    if reverse_minor:
        minors.reverse()
    cells = ((b, a) if major == "column" else (a, b) for a in majors for b in minors)
    for (r, c), color in zip(cells, stream):
        out[r, c] = color
    return out


def _connected_mask(rng, h, w, target):
    # A random monotone spanning path, followed by random frontier growth.
    result = np.zeros((h, w), dtype=bool)
    steps = [0]*(h-1) + [1]*(w-1)
    rng.shuffle(steps)
    r = c = 0
    result[r, c] = True
    for axis in steps:
        r += axis == 0
        c += axis == 1
        result[r, c] = True
    while int(result.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(result):
            for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= a < h and 0 <= b < w and not result[a, b]:
                    frontier.add((int(a), int(b)))
        result[_pick(rng, sorted(frontier))] = True
    if rng.integers(2):
        result = result[::-1, :]
    if rng.integers(2):
        result = result[:, ::-1]
    return result.copy()


def _sample_template(rng, low, high, non_solid=False, exclude=None):
    options = []
    for h in range(2, 6):
        for w in range(2, 6):
            populations = [m for m in range(max(low, h+w-1), min(high, h*w)+1)
                           if m != exclude and (not non_solid or m < h*w)]
            if populations:
                options.append((h, w, populations))
    h, w, populations = _pick(rng, options)
    return _connected_mask(rng, h, w, _pick(rng, populations))


def _sample_map(rng, mask, low, high, teaching=False, orientation=None, exclude=None):
    h, w = mask.shape
    connected = teaching or rng.random() < .55
    choices = []
    for mh in range(1, 8):
        for mw in range(1, 8):
            if mh*h > 26 or mw*w > 26 or (teaching and min(mh, mw) < 2):
                continue
            if orientation is not None and int(np.sign(mw-mh)) != orientation:
                continue
            minimum = max(low, mh+mw-1 if connected else 3)
            ns = [n for n in range(minimum, min(high, mh*mw)+1) if n != exclude]
            if ns:
                choices.append((mh, mw, ns))
    mh, mw, ns = _pick(rng, choices)
    n = _pick(rng, ns)
    if connected:
        layout = _connected_mask(rng, mh, mw, n)
    else:
        layout = np.zeros((mh, mw), dtype=bool)
        layout[0, 0] = layout[-1, -1] = True
        available = [(r, c) for r in range(mh) for c in range(mw) if not layout[r, c]]
        rng.shuffle(available)
        for p in available[:n-int(layout.sum())]:
            layout[p] = True
        if rng.integers(2):
            layout = layout[:, ::-1].copy()
    anchor = tuple(map(int, _pick(rng, np.argwhere(layout))))
    return layout, anchor


def _place(rng, mask, layout, anchor, roles, partial=False):
    h, w = mask.shape
    mh, mw = layout.shape
    ah, aw, ph, pw = mh*h, mw*w, 2*mh-1, 2*mw-1
    tr, tc = anchor[0]*h, anchor[1]*w
    options = []
    for pr in range(-ph-1, ah+2):
        for pc in range(-pw-1, aw+2):
            if not (pr+ph < tr or tr+h < pr or pc+pw < tc or tc+w < pc):
                continue
            top, left = min(0, pr), min(0, pc)
            bh, bw = max(ah, pr+ph)-top, max(aw, pc+pw)-left
            if bh <= (28 if partial else 30) and bw <= 30:
                options.append((pr, pc, top, left, bh, bw))
    if not options:
        raise ValueError("no unclipped placement for this recipe")
    if rng.random() < .5:
        best = min(p[4]*p[5] for p in options)
        options = [p for p in options if p[4]*p[5] <= best*1.25]
    pr, pc, top, left, bh, bw = _pick(rng, options)
    m, n = int(mask.sum()), int(layout.sum())
    shapes = [(gh, gw) for gh in range(max(4, bh), 31) for gw in range(bw, 31)
              if gh*gw-(m+n) > max(m+1, n-1)
              # If background were last by frequency, packing it would look
              # identical to leaving it blank. Expose that alternative in train.
              and (not partial or gh*gw-m*n > m)
              and (not partial or 2 <= m*(n-1) % gh < gh)]
    if not shapes:
        raise ValueError("no canvas revealing the required partial boundary")
    mode = int(rng.integers(4))
    if mode == 0:
        area = min(a*b for a, b in shapes)
        shapes = [s for s in shapes if s[0]*s[1] == area]
    elif mode == 1:
        shapes = [s for s in shapes if s[0] >= s[1]] or shapes
    elif mode == 2:
        shapes = [s for s in shapes if s[0] <= s[1]] or shapes
    gh, gw = _pick(rng, shapes)
    dr, dc = int(rng.integers(gh-bh+1))-top, int(rng.integers(gw-bw+1))-left
    bg, ink, marker = roles
    inp = np.full((gh, gw), bg, dtype=np.int8)
    rr, cc = np.nonzero(mask)
    inp[dr+tr+rr, dc+tc+cc] = ink
    for r, c in np.argwhere(layout):
        inp[dr+pr+2*r, dc+pc+2*c] = ink if (int(r), int(c)) == anchor else marker
    out = _pack_counts(inp.shape, bg, {ink: m*(n-1), marker: m})
    return {"input": inp, "output": out}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    orientation = _pick(rng, (-1, 1))
    base_roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    consistent_palette = rng.random() < .3
    puzzle, seen = {"train": [], "test": []}, set()
    first_meta = None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train" and i == 2:
                # The output is deliberately *not* reflected with its input.
                first = puzzle["train"][0]
                candidates = [np.fliplr(first["input"]), np.flipud(first["input"]),
                              np.rot90(first["input"], 2)]
                candidates = [a.copy() for a in candidates if _key(a) not in seen]
                inp = _pick(rng, candidates)
                pair = {"input": inp, "output": first["output"].copy()}
            else:
                teaching = split == "train" and i < 2
                if split == "train" and i == 0:
                    roles = base_roles
                elif split == "train" and i == 1:
                    roles = (base_roles[0], base_roles[2], base_roles[1])
                elif consistent_palette:
                    roles = base_roles if rng.integers(2) else (base_roles[0], base_roles[2], base_roles[1])
                else:
                    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
                pop_range = ((3, 8) if split == "train" else (9, 22)) if regime == 1 else (3, 25)
                n_range = ((3, 7) if split == "train" else (8, 20)) if regime == 0 else (3, 14)
                map_orientation = (orientation if split == "train" else -orientation) if regime == 2 else None
                previous = first_meta if teaching and i == 1 else None
                for _ in range(24):
                    mask = _sample_template(rng, *pop_range, non_solid=teaching and i == 0,
                                            exclude=previous["population"] if previous else None)
                    layout, anchor = _sample_map(rng, mask, *n_range, teaching=teaching,
                                                 orientation=map_orientation,
                                                 exclude=previous["sites"] if previous else None)
                    try:
                        pair = _place(rng, mask, layout, anchor, roles,
                                      partial=teaching and i == 0)
                    except ValueError:
                        continue
                    if _key(pair["input"]) not in seen:
                        break
                else:
                    raise RuntimeError("bounded recipe sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train" and i == 0:
                first_meta = _decode(pair["input"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _shortcut_outputs(grid, meta):
    bg, ink, marker = meta["background"], meta["ink"], meta["marker"]
    m, n = meta["population"], meta["sites"]
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    raw = {int(c): int(k) for c, k in zip(colors, counts)}
    area = meta["mask"].size
    return {
        "pack_raw_input": _walk_pack(grid.shape, bg, raw),
        "omit_inversion": _walk_pack(grid.shape, bg, {ink: m, marker: m*(n-1)}),
        "solid_bbox": _walk_pack(grid.shape, bg, {ink: area*(n-1), marker: area}),
        "one_cell_per_site": _walk_pack(grid.shape, bg, {ink: n-1, marker: 1}),
        "omit_packing": _stamp(grid.shape, meta),
        "ascending_counts": _walk_pack(grid.shape, bg, _inventory(meta), descending=False),
        "include_background": _walk_pack(grid.shape, bg,
            {**_inventory(meta), bg: grid.size-m*n}),
    }


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, metas, rejected = set(), {"train": [], "test": []}, set()
    rejected_traversals = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            meta = _decode(inp)
            metas[split].append(meta)
            m, n = meta["population"], meta["sites"]
            bg, ink, marker = meta["background"], meta["ink"], meta["marker"]
            assert m >= 3 and n >= 3 and m*n <= inp.size
            assert np.count_nonzero(inp == ink) == m+1
            assert np.count_nonzero(inp == marker) == n-1
            assert np.count_nonzero(out == ink) == m*(n-1)
            assert np.count_nonzero(out == marker) == m
            assert np.count_nonzero(out == bg) == inp.size-m*n
            assert set(map(int, np.unique(out))) <= {bg, ink, marker}
            assert np.array_equal(out, _walk_pack(inp.shape, bg, _inventory(meta)))
            assert np.array_equal(out, solve(inp))
            assert not np.array_equal(inp, out)
            stamped = _stamp(inp.shape, meta)
            assert np.count_nonzero(stamped == ink) == m*(n-1)
            assert np.count_nonzero(stamped == marker) == m
            tr, tc = meta["template_top"]
            h, w = meta["mask"].shape
            assert np.all(stamped[tr:tr+h, tc:tc+w][meta["mask"]] == marker)
            if split == "train":
                rejected.update(name for name, alt in _shortcut_outputs(inp, meta).items()
                                if not np.array_equal(alt, out))
                for major in ("column", "row"):
                    for rev_a in (False, True):
                        for rev_b in (False, True):
                            params = major, rev_a, rev_b
                            alt = _walk_pack(inp.shape, bg, _inventory(meta), *params)
                            if not np.array_equal(alt, out):
                                rejected_traversals.add(params)
    assert rejected == {"pack_raw_input", "omit_inversion", "solid_bbox", "one_cell_per_site",
                        "omit_packing", "ascending_counts", "include_background"}
    assert len(rejected_traversals) == 7 and ("column", False, True) not in rejected_traversals
    a, b = metas["train"][:2]
    assert a["population"] != b["population"] and a["sites"] != b["sites"]
    assert a["population"] < a["mask"].size
    assert (a["background"], a["ink"], a["marker"]) == (b["background"], b["marker"], b["ink"])
    height = np.asarray(puzzle["train"][0]["input"]).shape[0]
    assert 2 <= a["population"]*(a["sites"]-1) % height < height
    if nt >= 3:
        x, y = puzzle["train"][0], puzzle["train"][2]
        assert np.asarray(x["input"]).shape == np.asarray(y["input"]).shape
        assert np.array_equal(x["output"], y["output"])
        assert _inventory(a) == _inventory(metas["train"][2])
    assert any(_shift_flags(metas).values()), "missing structural train/test shift"


def _shift_flags(metas):
    train, test = metas["train"], metas["test"]
    orientations = [{int(np.sign(m["layout"].shape[1]-m["layout"].shape[0])) for m in group}
                    for group in (train, test)]
    return {
        "more_sites": min(m["sites"] for m in test) > max(m["sites"] for m in train),
        "more_template_ink": min(m["population"] for m in test) > max(m["population"] for m in train),
        "map_aspect_reversal": len(orientations[0]) == len(orientations[1]) == 1
            and next(iter(orientations[0]))*next(iter(orientations[1])) == -1,
    }


def self_test():
    """Bounded, in-memory checks over the 42 requested seeds; returns metrics."""
    signatures, shapes, masks, layouts, anchors, roles, pops, sites, remainders = (set() for _ in range(9))
    shifts = {"more_sites": 0, "more_template_ink": 0, "map_aspect_reversal": 0}
    pairs = lines = gaps = aligned = partial = equal_hist = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        metas = {s: [_decode(p["input"]) for p in puzzle[s]] for s in ("train", "test")}
        for name, active in _shift_flags(metas).items():
            shifts[name] += int(active)
        equal_hist += int(np.array_equal(puzzle["train"][0]["output"], puzzle["train"][2]["output"]))
        for split in ("train", "test"):
            for pair, meta in zip(puzzle[split], metas[split]):
                pairs += 1
                shapes.add(pair["input"].shape)
                masks.add(_key(meta["mask"]))
                layouts.add(_key(meta["layout"]))
                anchors.add(meta["anchor"])
                roles.add((meta["background"], meta["ink"], meta["marker"]))
                pops.add(meta["population"])
                sites.add(meta["sites"])
                remainder = meta["population"]*(meta["sites"]-1) % pair["input"].shape[0]
                remainders.add(remainder)
                aligned += remainder == 0
                partial += remainder > 1
                lines += min(meta["layout"].shape) == 1
                gaps += bool(np.any(~meta["layout"].any(axis=0)) or np.any(~meta["layout"].any(axis=1)))
                # Check omission witnesses on each actual example, not only seed metadata.
                alternatives = _shortcut_outputs(pair["input"], meta)
                for name in ("pack_raw_input", "omit_inversion", "one_cell_per_site", "ascending_counts"):
                    assert not np.array_equal(alternatives[name], pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 40
    assert len(masks) >= 50 and len(layouts) >= 50 and len(anchors) >= 15
    assert len(pops) >= 12 and len(sites) >= 12 and len(roles) >= 60
    assert {c for triple in roles for c in triple} == set(range(10))
    assert all(v > 0 for v in shifts.values())
    assert len(remainders) >= 12 and aligned > 0 and partial >= 42
    assert lines > 0 and gaps > 0 and equal_hist == 42
    legal_count_cases = 0
    for nt in range(2, 11):
        for nq in range(1, min(nt-1, 4)+1):
            p = generate_puzzle(1234567, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
            legal_count_cases += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None), (-1, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A controlled anchor swap changes A's spatial construction, but cannot
    # change the final multiset. This limitation is intrinsic to the proposal.
    fixture = np.zeros((22, 22), dtype=np.int8)
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    patch = fixture[8:10, 8:11]
    patch[mask] = 2
    for r, c in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 2)):
        fixture[1+2*r, 1+2*c] = 2 if (r, c) == (1, 1) else 8
    swapped = fixture.copy()
    swapped[3, 3], swapped[1, 1] = 8, 2
    assert np.array_equal(solve(fixture), solve(swapped))
    assert not np.array_equal(_stamp(fixture.shape, _decode(fixture)),
                              _stamp(swapped.shape, _decode(swapped)))
    # Removing an ordinary site changes exactly one template's inventory.
    fewer = fixture.copy()
    fewer[1, 5] = 0
    assert not np.array_equal(solve(fixture), solve(fewer))
    assert np.count_nonzero(solve(fixture)) - np.count_nonzero(solve(fewer)) == int(mask.sum())
    broken = generate_puzzle(0)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": pairs, "distinct_puzzles": len(signatures),
            "canvas_shapes": len(shapes), "template_masks": len(masks), "map_layouts": len(layouts),
            "anchor_indices": len(anchors), "ordered_palettes": len(roles),
            "template_populations": sorted(pops), "site_counts": sorted(sites),
            "dominant_boundary_remainders": sorted(remainders), "aligned_boundaries": aligned,
            "partial_boundaries_over_one": partial, "line_maps": lines, "gapped_maps": gaps,
            "equal_inventory_contrasts": equal_hist, "shift_puzzles": shifts,
            "legal_count_cases": legal_count_cases,
            "semantic_verification": "generator-owned checks; no independent solver or blind prediction"}
