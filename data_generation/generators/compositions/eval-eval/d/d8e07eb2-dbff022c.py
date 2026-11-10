"""Negative-space ballots: a directed color cast gates whole-vessel line fills.

Only standard-library modules and NumPy are used. Prediction reads pixels only;
construction labels use the intended atlas coordinates and carved silhouettes.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import itertools
import json

import numpy as np

_STEPS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_STEPS8 = _STEPS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer 2D grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, bool)
    groups = []
    steps = _STEPS8 if diagonal else _STEPS4
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            q = stack.pop()
            cells.append(q)
            for dr, dc in steps:
                rr, cc = q[0] + dr, q[1] + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        groups.append(tuple(cells))
    return groups


def _trim(mask):
    rr, cc = np.nonzero(mask)
    assert len(rr), "empty silhouette"
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _shape_key(mask):
    a = _trim(mask).astype(np.uint8)
    return a.shape, a.tobytes()


def _token_key(mask, color):
    return int(color), _shape_key(mask)


def _complete_line(coords, nr, nc):
    coords = list(coords)
    if len(set(coords)) != len(coords):
        return False
    selected = set(coords)
    return (any(selected == {(r, c) for c in range(nc)} for r in range(nr))
            or any(selected == {(r, c) for r in range(nr)} for c in range(nc)))


def _background(g):
    colors, counts = np.unique(g, return_counts=True)
    assert sum(counts == counts.max()) == 1, "ambiguous background"
    return int(colors[np.argmax(counts)])


def _edge_key(g, bg):
    h, w = g.shape
    candidates = []
    for cells in _components(g != bg, True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != sh * sw or len({int(g[q]) for q in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and 3 <= sw <= 5:
            if r0 == 0:
                lanes.append(("top", g[r0, c0:c1+1], g[r1, c0:c1+1]))
            if r1 == h-1:
                lanes.append(("bottom", g[r1, c0:c1+1], g[r0, c0:c1+1]))
        if sw == 2 and 3 <= sh <= 5:
            if c0 == 0:
                lanes.append(("left", g[r0:r1+1, c0], g[r0:r1+1, c1]))
            if c1 == w-1:
                lanes.append(("right", g[r0:r1+1, c1], g[r0:r1+1, c0]))
        for side, outer, inner in lanes:
            sources = list(map(int, outer))
            if len(set(sources)) == len(sources):
                candidates.append((side, dict(zip(sources, map(int, inner)))))
    assert len(candidates) == 1, "key must have one unambiguous edge direction"
    return candidates[0]


def _parse_scene(input_grid, cavity_diagonal=False):
    original = _grid(input_grid)
    bg = _background(original)
    side, mapping = _edge_key(original, bg)
    rotation = {"top": 0, "left": 3, "bottom": 2, "right": 1}[side]
    g = np.rot90(original, rotation).copy()
    assert np.all(g[2] == bg)
    marker = int(g[3, 0])
    assert marker != bg and marker not in set(mapping) | set(mapping.values())
    pw = 0
    while pw < g.shape[1] and g[3, pw] == marker:
        pw += 1
    end = 3
    while end < g.shape[0] and g[end, 0] == marker:
        end += 1
    panel = g[3:end, :pw]
    assert panel.size and np.all(g[3:end, pw:] == bg)
    rows = [r for r in range(len(panel)) if np.all(panel[r] == marker)]
    cols = [c for c in range(pw) if np.all(panel[:, c] == marker)]
    assert rows[0] == cols[0] == 0
    assert rows[-1] == len(panel)-1 and cols[-1] == pw-1
    nr, nc = len(rows)-1, len(cols)-1
    assert nr in (2, 3) and nc in (2, 3)
    assert len(set(np.diff(rows))) == len(set(np.diff(cols))) == 1
    assert int(rows[1]-rows[0]-1) in (2, 3)
    assert int(cols[1]-cols[0]-1) in (2, 3)
    lookup, atlas = {}, {}
    for r, c in itertools.product(range(nr), range(nc)):
        token = panel[rows[r]+1:rows[r+1], cols[c]+1:cols[c+1]]
        inks = np.unique(token[token != bg])
        assert len(inks) == 1 and int(inks[0]) != marker
        mask, ink = token != bg, int(inks[0])
        assert len(_components(mask)) == 1
        # No enclosed background islands in a token's own silhouette.
        padded = np.pad(mask, 1)
        assert len(_components(~padded)) == 1
        rr, cc = np.nonzero(mask)
        assert int(rr.min()) == (len(token)-(int(rr.max()-rr.min())+1))//2
        assert int(cc.min()) == (token.shape[1]-(int(cc.max()-cc.min())+1))//2
        key = _token_key(mask, ink)
        assert key not in lookup, "atlas tokens must be unique"
        lookup[key], atlas[(r, c)] = (r, c), key
    assert end + 1 < len(g) and np.all(g[end] == bg)
    body0 = end + 1
    body = g[body0:]
    assert not np.any(body == marker)
    ids = np.full(body.shape, -1, int)
    vessels = []
    for ink in sorted(map(int, np.unique(body))):
        if ink == bg:
            continue
        for cells in _components(body == ink, True):
            idx = len(vessels)
            for q in cells:
                ids[q] = idx
            vessels.append({"wall": ink, "ink": cells, "holes": []})
    assert vessels, "no vessels"
    exterior = []
    for cells in _components(body == bg, cavity_diagonal):
        if any(r in (0, body.shape[0]-1) or c in (0, body.shape[1]-1) for r, c in cells):
            exterior.extend(cells)
            continue
        boundary = set()
        for r, c in cells:
            for dr, dc in _STEPS4:
                rr, cc = r+dr, c+dc
                if body[rr, cc] != bg:
                    boundary.add((rr, cc))
        owners = {int(ids[q]) for q in boundary}
        assert len(owners) == 1 and -1 not in owners, "mixed cavity ownership"
        owner = next(iter(owners))
        rr, cc = zip(*cells)
        shape = np.zeros((max(rr)-min(rr)+1, max(cc)-min(cc)+1), bool)
        for r, c in cells:
            shape[r-min(rr), c-min(cc)] = True
        assert len(_components(~np.pad(shape, 1))) == 1, "internal islands are outside the family"
        vessels[owner]["holes"].append({"cells": cells, "shape": shape})
    return {"grid": g, "bg": bg, "marker": marker, "mapping": mapping,
            "lookup": lookup, "atlas": atlas, "nr": nr, "nc": nc,
            "body0": body0, "vessels": vessels, "exterior": exterior,
            "rotation": rotation, "side": side,
            "cell": (int(rows[1]-1), int(cols[1]-1))}


def _decisions(info, raw_wall=False):
    decisions = []
    for vessel in info["vessels"]:
        wall = vessel["wall"]
        fill = wall if raw_wall else info["mapping"].get(wall)
        matches = []
        if fill is not None:
            for hole in vessel["holes"]:
                matches.append(info["lookup"].get(_token_key(hole["shape"], fill)))
        win = (fill is not None and bool(matches) and None not in matches
               and _complete_line(matches, info["nr"], info["nc"]))
        decisions.append((fill, matches, win))
    return decisions


def _predict(info, ignore_line=False, raw_wall=False):
    out = info["grid"].copy()
    for vessel, (fill, _, win) in zip(info["vessels"], _decisions(info, raw_wall)):
        if fill is not None and (win or ignore_line):
            for hole in vessel["holes"]:
                for r, c in hole["cells"]:
                    out[r+info["body0"], c] = fill
    return np.rot90(out, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the atlas, one-step color casts, and container ballots from input."""
    return _predict(_parse_scene(input_grid))


@lru_cache(maxsize=4)
def _shape_bank(gh, gw):
    bank = {}
    for bits in range(1, 1 << (gh*gw)):
        a = np.array([(bits >> i) & 1 for i in range(gh*gw)], bool).reshape(gh, gw)
        if len(_components(a)) == 1 and len(_components(~np.pad(a, 1))) == 1:
            crop = _trim(a)
            bank[_shape_key(crop)] = crop
    return tuple(bank.values())


def _make_atlas(rng, nr, nc, gh, gw, main, other, axis):
    index = int(rng.integers(nr if axis == "row" else nc))
    line = ([(index, c) for c in range(nc)] if axis == "row"
            else [(r, index) for r in range(nr)])
    rng.shuffle(line)
    outside = [(r, c) for r, c in np.ndindex(nr, nc) if (r, c) not in line]
    extra_choices = [q for q in outside if not _complete_line(line[:-1]+[q], nr, nc)]
    extra = extra_choices[int(rng.integers(len(extra_choices)))]
    near = line[:-1]+[extra]
    foil_choices = [q for q in outside if q != extra]
    foil = foil_choices[int(rng.integers(len(foil_choices)))]
    bank = _shape_bank(gh, gw)
    # Make a nonrectangular winning hole an explicit enclosure/shape witness.
    irregular = [i for i, m in enumerate(bank) if not np.all(m)]
    chosen = [int(rng.choice(irregular))]
    remaining = [i for i in range(len(bank)) if i not in chosen]
    chosen.extend(map(int, rng.choice(remaining, len(line), replace=False)))
    atlas = {q: (bank[i].copy(), main) for q, i in zip(line+[extra], chosen)}
    atlas[foil] = (atlas[line[0]][0].copy(), other)
    used = {_token_key(m, ink) for m, ink in atlas.values()}
    line_shapes = {_shape_key(atlas[q][0]) for q in line}
    for q in np.ndindex(nr, nc):
        if q in atlas:
            continue
        choices = [(m, ink) for m in bank for ink in (main, other)
                   if _token_key(m, ink) not in used
                   and (ink == main or _shape_key(m) not in line_shapes)]
        m, ink = choices[int(rng.integers(len(choices)))]
        atlas[q] = (m.copy(), ink)
        used.add(_token_key(m, ink))
    return atlas, line, near


def _vessel(rng, shapes, opened=False, layout="strip", decorate=True):
    order = list(map(int, rng.permutation(len(shapes))))
    ordered = [shapes[i] for i in order]
    holes = []
    if layout == "strip":
        layout = "vertical" if rng.random() < .5 else "horizontal"
    if layout == "bent" and len(shapes) == 3:
        th = max(m.shape[0] for m in ordered)
        tw = max(m.shape[1] for m in ordered)
        h, w = 2*th+3, 2*tw+3
        positions = [(1, 1), (1, tw+2), (th+2, 1)]
        for m, (r, c) in zip(ordered, positions):
            holes.append(tuple((r+int(a), c+int(b)) for a, b in np.argwhere(m)))
    elif layout == "vertical":
        h = sum(m.shape[0] for m in ordered)+len(ordered)+1
        w = max(m.shape[1] for m in ordered)+2
        r = 1
        for m in ordered:
            c = 1+(w-2-m.shape[1])//2
            holes.append(tuple((r+int(a), c+int(b)) for a, b in np.argwhere(m)))
            r += m.shape[0]+1
    else:
        h = max(m.shape[0] for m in ordered)+2
        w = sum(m.shape[1] for m in ordered)+len(ordered)+1
        c = 1
        for m in ordered:
            r = 1+(h-2-m.shape[0])//2
            holes.append(tuple((r+int(a), c+int(b)) for a, b in np.argwhere(m)))
            c += m.shape[1]+1
    mask = np.ones((h, w), bool)
    for cells in holes:
        for q in cells:
            mask[q] = False
    # Diagonal corners retain four-connected closure, including small diamonds.
    for q in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)):
        if rng.random() < .7:
            mask[q] = False
    # A diagonal window demonstrates that diagonal background escape does
    # not invalidate four-connected enclosure. Its orthogonal neighbors stay ink.
    top = min(q for cells in holes for q in cells)
    assert top[0] == 1
    mask[0, top[1]-1] = False
    if decorate and rng.random() < .55:
        mask = np.pad(mask, ((0, 1), (0, 0)))
        mask[-1, int(rng.integers(1, w-1))] = True
    if opened:
        r, c = min(holes[0])
        mask[:r+1, c] = False
    # Change arrangement, not stamp orientation: exact matching is unrotated.
    assert len(_components(mask, True)) == 1
    return mask, holes, order


def _pack(rng, masks, panel_width, body_limit):
    # Random separated placement: no reliance on slots for vessel ownership.
    for attempt in range(24):
        width = int(rng.integers(max(panel_width, max(m.shape[1] for m in masks), 18), 31))
        height = body_limit if attempt >= 12 else int(rng.integers(max(m.shape[0] for m in masks), body_limit+1))
        blocked = np.zeros((height, width), bool)
        locations = [None]*len(masks)
        for i in sorted(range(len(masks)), key=lambda j: -masks[j].size):
            mh, mw = masks[i].shape
            positions = [(r, c) for r in range(height-mh+1) for c in range(width-mw+1)
                         if not np.any(blocked[r:r+mh, c:c+mw])]
            if not positions:
                break
            r, c = positions[int(rng.integers(len(positions)))]
            locations[i] = (r, c)
            blocked[max(0,r-1):min(height,r+mh+1), max(0,c-1):min(width,c+mw+1)] = True
        else:
            return height, width, locations
    return None


def _render_pair(rng, geometry, palette, axis, side_turn, contrast, count, cast_target):
    nr, nc, gh, gw = geometry
    bg, marker = map(int, palette[:2])
    k = int(rng.integers(3, 6))
    sources = list(map(int, palette[2:2+k]))
    destination_a, destination_b = sources[1], int(palette[2+k])
    mapping = {sources[0]: destination_a, sources[1]: destination_b,
               sources[2]: destination_a}
    for source in sources[3:]:
        mapping[source] = int(rng.choice(palette[2:]))
    main, other = ((destination_a, destination_b) if cast_target == 0
                   else (destination_b, destination_a))
    active_wall = sources[0] if cast_target == 0 else sources[1]
    wrong_wall = sources[1] if cast_target == 0 else sources[0]
    atlas, line, near = _make_atlas(rng, nr, nc, gh, gw, main, other, axis)
    specs = [(active_wall, line, False), (active_wall, near, False)]
    specs.append((active_wall, line, True) if contrast == "open"
                 else (destination_b, line, False))
    if count >= 4:
        # Identical negative shapes, different directed casts: the foil color
        # has only one of these shapes in the atlas, so the whole ballot fails.
        specs.append((wrong_wall, line, False))
    if count >= 5:
        specs.append((sources[2] if cast_target == 0 else active_wall, line, False))
    panel_h, panel_w = nr*(gh+1)+1, nc*(gw+1)+1
    body0 = 3+panel_h+1
    body_limit = 30-body0
    made = []
    for wall, coords, opened in specs:
        layout = "bent" if rng.random() < .25 and len(coords) == 3 else "strip"
        made.append(_vessel(rng, [atlas[q][0] for q in coords], opened, layout))
    if all(m[0].shape[0] <= body_limit for m in made):
        packing = _pack(rng, [m[0] for m in made], panel_w, body_limit)
    else:
        packing = None
    if packing is None:
        # Guaranteed two-per-shelf layout; five vessels are admitted only for
        # a short atlas and 2x2 token cells. Strip maxima are 5x10 in that case.
        made = []
        for wall, coords, opened in specs:
            mask, holes, order = _vessel(rng, [atlas[q][0] for q in coords], opened, "horizontal", False)
            made.append((mask, holes, order))
        locations, y, maxwidth = [], 0, panel_w
        for i in range(0, len(made), 2):
            row_h = max(made[j][0].shape[0] for j in range(i, min(i+2, len(made))))
            x = 0
            for j in range(i, min(i+2, len(made))):
                locations.append((y, x))
                x += made[j][0].shape[1]+1
            maxwidth = max(maxwidth, x-1)
            y += row_h+1
        assert y-1 <= body_limit and maxwidth <= 30, "fallback must fit"
        packing = (body_limit, max(24, maxwidth), locations)
    height, width, locations = packing
    g = np.full((body0+height, width), bg, np.int8)
    # Key offset does not alter boundary-to-inward direction.
    start = int(rng.integers(1, width-k+1))
    entry_order = list(map(int, rng.permutation(k)))
    for j, index in enumerate(entry_order):
        source = sources[index]
        g[0, start+j], g[1, start+j] = source, mapping[source]
    panel = g[3:3+panel_h, :panel_w]
    panel[::gh+1, :] = marker
    panel[:, ::gw+1] = marker
    for (r, c), (mask, ink) in atlas.items():
        a = 3+r*(gh+1)+1+(gh-mask.shape[0])//2
        b = c*(gw+1)+1+(gw-mask.shape[1])//2
        g[a:a+mask.shape[0], b:b+mask.shape[1]][mask] = ink
    out = g.copy()
    for (wall, coords, opened), (mask, holes, order), (r, c) in zip(specs, made, locations):
        r += body0
        g[r:r+mask.shape[0], c:c+mask.shape[1]][mask] = wall
        out[r:r+mask.shape[0], c:c+mask.shape[1]][mask] = wall
        surviving = [coords[order[i]] for i in range(len(holes)) if not (opened and i == 0)]
        fill = mapping.get(wall)
        win = (fill is not None and fill == main
               and _complete_line(surviving, nr, nc))
        if win:
            for cells in holes:
                for a, b in cells:
                    out[r+int(a), c+int(b)] = fill
    # Input background must be geometrically recognizable, even on dense scenes.
    if _background(g) != bg:
        raise AssertionError("background ceased to be dominant")
    return {"input": np.rot90(g, side_turn).copy(), "output": np.rot90(out, side_turn).copy()}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(name+" outside its permitted range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    nr, nc = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    gh, gw = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    geometry = nr, nc, gh, gw
    palette = rng.permutation(10)
    local_palette = bool(rng.integers(2))
    turns = list(map(int, rng.permutation(4)))
    axis0 = "row" if rng.random() < .5 else "col"
    train, test, seen = [], [], set()
    for i in range(nt+nq):
        training = i < nt
        axis = axis0 if i == 0 else ("col" if axis0 == "row" else "row") if i == 1 else str(rng.choice(["row", "col"]))
        contrast = "open" if i == 0 else "inert" if i == 1 else str(rng.choice(["open", "inert"]))
        turn = turns[i % 2] if training else turns[2+(i-nt) % 2]
        maximum = 5 if nr == 2 and gh == gw == 2 else 4
        count = 4 if i < 2 else int(rng.integers(3, maximum+1))
        cast_target = i if i < 2 else int(rng.integers(2))
        for retry in range(12):
            colors = rng.permutation(10) if local_palette else palette
            pair = _render_pair(rng, geometry, colors, axis, turn, contrast, count, cast_target)
            key = _fingerprint(pair["input"])
            if key not in seen:
                seen.add(key)
                (train if training else test).append(pair)
                break
        else:
            raise AssertionError("bounded unique-input construction exhausted")
    rng.shuffle(train)
    rng.shuffle(test)
    return {"train": train, "test": test}


def _fingerprint(a):
    a = np.asarray(a, np.uint8)
    return a.shape, a.tobytes()


def _features(info):
    decisions = _decisions(info)
    winners = [v for v, d in zip(info["vessels"], decisions) if d[2]]
    near = any(fill is not None and matches and None not in matches
               and len(set(matches)) == len(matches) and not win
               and len(matches) in (info["nr"], info["nc"])
               for fill, matches, win in decisions)
    shape_sensitive = any(
        key[1] == other[1] and key[0] != other[0]
        for v, (_, matches, win) in zip(info["vessels"], decisions) if win
        for p in matches for key in [info["atlas"][p]]
        for other in info["lookup"])
    chain = any(info["mapping"].get(v["wall"]) in info["mapping"]
                and info["mapping"][info["mapping"][v["wall"]]] != info["mapping"][v["wall"]]
                for v in winners)
    exterior = set(info["exterior"])
    open_active = False
    for v in info["vessels"]:
        if v["wall"] not in info["mapping"]:
            continue
        rr, cc = zip(*v["ink"])
        open_active |= any(min(rr) < r < max(rr) and min(cc) < c < max(cc)
                           for r, c in exterior)
    cast_contrast = any(
        fill is not None and fill != info["mapping"][winner["wall"]] and not win
        and sorted(_shape_key(h["shape"]) for h in vessel["holes"])
        == sorted(_shape_key(h["shape"]) for h in winner["holes"])
        for winner in winners
        for vessel, (fill, _, win) in zip(info["vessels"], decisions))
    return {"dimensions": (info["nr"], info["nc"]), "cell": info["cell"],
            "side": info["side"], "vessels": len(info["vessels"]),
            "winner_count": len(winners), "near": near, "color_foil": shape_sensitive,
            "chain": chain, "open_active": open_active, "cast_contrast": cast_contrast,
            "key_length": len(info["mapping"]),
            "background": info["bg"],
            "hole_counts": tuple(len(v["holes"]) for v in info["vessels"]),
            "irregular": any(not np.all(h["shape"]) for v in winners for h in v["holes"]),
            "inert": any(v["wall"] not in info["mapping"] and v["holes"] for v in info["vessels"]),
            "axes": {"row" if len({p[0] for p in matches}) == 1 else "col"
                     for _, matches, win in decisions if win}}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records, omit_a, omit_b, connectivity_witness = set(), [], False, False, False
    cast_mapping_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse_scene(inp)
            predicted = _predict(info)
            assert np.array_equal(predicted, out), "incorrect ballot transformation"
            assert np.any(inp != out), "scene must show a successful ballot"
            canonical_out = np.rot90(out, info["rotation"])
            assert np.array_equal(inp[inp != info["bg"]], out[inp != info["bg"]]), "original ink changed"
            for r, c in info["exterior"]:
                assert canonical_out[r+info["body0"], c] == info["bg"]
            feature = _features(info)
            records.append((split, feature))
            if split == "train":
                omit_a |= not np.array_equal(_predict(info, ignore_line=True), out)
                omit_b |= not np.array_equal(_predict(info, raw_wall=True), out)
                connectivity_witness |= not np.array_equal(_predict(_parse_scene(inp, True)), out)
                winning_fill = next(d[0] for d in _decisions(info) if d[2])
                collapsed = dict(info, mapping={c: winning_fill for c in info["mapping"]})
                cast_mapping_witness |= not np.array_equal(_predict(collapsed), out)
    trains = [f for s, f in records if s == "train"]
    assert omit_a and omit_b, "training must make each inherited concept necessary"
    assert all(f["near"] and f["winner_count"] for f in trains)
    assert any(f["chain"] for f in trains) and any(f["color_foil"] for f in trains)
    assert any(f["irregular"] for f in trains) and any(f["inert"] for f in trains)
    assert any(f["open_active"] for f in trains) and connectivity_witness
    assert any(f["cast_contrast"] for f in trains) and cast_mapping_witness
    assert set().union(*(f["axes"] for f in trains)) == {"row", "col"}
    assert len({f["side"] for f in trains}) >= 2


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _hand_fixture():
    """Explicit domain cases outside the sampler's normal teaching scenes."""
    a = np.array([[1, 0], [1, 1]], bool)
    h = np.ones((1, 2), bool)
    v = np.ones((2, 1), bool)
    s = np.ones((1, 1), bool)
    e = np.array([[0, 1], [1, 1]], bool)
    f = np.ones((2, 2), bool)
    atlas = [(a, 2), (h, 2), (v, 2), (a, 4), (h, 4), (v, 4),
             (s, 2), (e, 4), (f, 4)]
    inp = np.zeros((30, 30), np.int8)
    inp[:2, 11:15] = [[1, 2, 3, 4], [2, 4, 2, 4]]
    inp[3:13:3, :10] = 9
    inp[3:13, :10:3] = 9
    for i, (mask, ink) in enumerate(atlas):
        r, c = divmod(i, 3)
        r, c = 4+r*3+(2-mask.shape[0])//2, 1+c*3+(2-mask.shape[1])//2
        inp[r:r+mask.shape[0], c:c+mask.shape[1]][mask] = ink
    # Winner; complete line plus extra; duplicate; unmatched; self-map;
    # a different source with the same destination as the first winner.
    ballots = [(1, [a, h, v]), (1, [a, h, v, s]), (1, [a, a, h, v]),
               (1, [a, h, e]), (4, [a, h, v]), (3, [a, h, v])]
    expected = inp.copy()
    rng = np.random.default_rng(921)
    for i, (wall, masks) in enumerate(ballots):
        contour, holes, _ = _vessel(rng, masks, False, "horizontal", False)
        r, c = 14+(i//2)*5, (i % 2)*16
        inp[r:r+contour.shape[0], c:c+contour.shape[1]][contour] = wall
        expected[r:r+contour.shape[0], c:c+contour.shape[1]][contour] = wall
        if i in (0, 4, 5):
            fill = 4 if i == 4 else 2
            for cells in holes:
                for a0, b0 in cells:
                    expected[r+a0, c+b0] = fill
    return inp, expected


def self_test():
    """Bounded pixel-owned checks; returns compact descriptive diversity metrics."""
    signatures, normalized, canvas, atlas_dims, cells, key_lengths = set(), set(), set(), set(), set(), set()
    shapes, hole_counts, vessel_counts, backgrounds, sides = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse_scene(pair["input"])
                f = _features(info)
                canvas.add(pair["input"].shape)
                atlas_dims.add(f["dimensions"])
                cells.add(f["cell"])
                key_lengths.add(f["key_length"])
                vessel_counts.add(f["vessels"])
                backgrounds.add(f["background"])
                sides.add(f["side"])
                hole_counts.update(f["hole_counts"])
                shapes.update(_shape_key(h["shape"]) for v in info["vessels"] for h in v["holes"])
                roles = {}
                norm = tuple(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat)
                normalized.add((pair["input"].shape, norm))
    assert len(signatures) == 42 and len(normalized) >= 150
    assert len(canvas) >= 20 and len(shapes) >= 20
    assert atlas_dims == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert cells == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert key_lengths == {3, 4, 5} and sides == {"top", "left", "bottom", "right"}
    assert {3, 4} <= vessel_counts and {1, 2, 3} <= hole_counts
    assert len(backgrounds) >= 8
    fixture, answer = _hand_fixture()
    assert np.array_equal(solve(fixture), answer)
    assert sum(d[2] for d in _decisions(_parse_scene(fixture))) == 3
    assert _puzzle_digest(generate_puzzle(17)) == _puzzle_digest(generate_puzzle(17))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(810+nt*10+nq, nt, nq)
            validate_generated_puzzle(p)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    baseline = generate_puzzle(0)
    for turn in range(4):
        for pair in baseline["train"]+baseline["test"]:
            assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
    perm = np.random.default_rng(912).permutation(10)
    for pair in baseline["train"]+baseline["test"]:
        assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    for defect in ("ink", "fill", "duplicate"):
        broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in baseline[s]] for s in baseline}
        if defect == "duplicate":
            broken["train"][1] = {k: v.copy() for k, v in broken["train"][0].items()}
        else:
            pair = broken["test"][0]
            info = _parse_scene(pair["input"])
            if defect == "ink":
                q = tuple(np.argwhere(pair["input"] != info["bg"])[0])
            else:
                q = tuple(np.argwhere(pair["input"] != pair["output"])[0])
            pair["output"][q] = (int(pair["output"][q])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted "+defect)
    return {"seeds": 42, "distinct_puzzles": len(signatures), "normalized_inputs": len(normalized),
            "canvas_shapes": len(canvas), "cavity_shapes": len(shapes),
            "atlas_dimensions": sorted(atlas_dims), "cell_dimensions": sorted(cells),
            "vessel_counts": sorted(vessel_counts), "hole_counts": sorted(hole_counts),
            "backgrounds": sorted(backgrounds), "key_lengths": sorted(key_lengths)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
