"""Fault fossils: defect rings become fixed rivets in a removable block pile.

Only standard-library modules and NumPy are used.  The solver reads pixels and
the visible header; construction metadata is never consulted by solve().
"""
import json

import numpy as np


HOST_SEEDS = (0, 285256729, 868018912, 786832028, 733394099, 763892356,
              3670303, 603380716, 489093532, 324683699, 597861963, 392168199,
              391091030, 66298237, 392189883, 508218069, 865153977, 757253092,
              974905542, 453733984, 109472334, 108624046, 121025584, 535382329,
              347078048, 624251351, 270906067, 815900516, 715862882, 475781669,
              868642431, 131647051, 322564998, 237830380, 521490988, 473547957,
              921391739, 194390952, 703691700, 541268577, 427597113, 353085336)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer color grid of dimensions 1..30")
    return a.astype(np.int64, copy=True)


def _components(mask):
    """Four-connected components, in deterministic raster order."""
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, points = [start], [start]
        while todo:
            y, x = todo.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = (y + dy, x + dx)
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    points.append(p)
        result.append(points)
    return result


def _mask(shape, rect):
    y, x, h, w = rect[:4]
    m = np.zeros(shape, dtype=bool)
    m[y:y+h, x:x+w] = True
    return m


def _cells(rect):
    y, x, h, w = rect[:4]
    return frozenset((r, c) for r in range(y, y+h) for c in range(x, x+w))


def _shift(rect, d):
    return (rect[0] + d,) + tuple(rect[1:])


def _halo(holes):
    h, w = holes.shape
    p = np.pad(holes, 1)
    ring = np.zeros_like(holes)
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                ring |= p[dy:dy+h, dx:dx+w]
    # A defect center wins even when another defect's ring covers it.
    ring &= ~holes
    ring[:2] = False  # The header and separator are outside the physical scene.
    return ring


def _decode(inp):
    """Infer rectangles, holes, specks, selectors and ring identity from input."""
    h, w = inp.shape
    if h < 6 or w < 5:
        raise ValueError("insufficient room for the visible header and scene")
    bg, rivet = int(inp[1, 0]), int(inp[0, 0])
    if rivet == bg or not np.all(inp[1] == bg) or inp[0, 1] != bg:
        raise ValueError("invalid rivet swatch / blank separator")
    selectors = [int(c) for c in inp[0, 2:] if c != bg]
    if (not 1 <= len(selectors) <= 3 or len(set(selectors)) != len(selectors)
            or rivet in selectors or np.any(inp[2:] == rivet)):
        raise ValueError("invalid construction identities or command swatches")
    rects = []
    extra = np.zeros(inp.shape, dtype=bool)
    holes = np.zeros_like(extra)
    occupied = np.zeros_like(extra)
    for color in sorted(set(map(int, np.unique(inp[2:]))) - {bg}):
        m = inp == color
        m[:2] = False
        for points in _components(m):
            rr, cc = zip(*points)
            if len(points) <= 2:
                extra[rr, cc] = True
                continue
            if len(points) < 4:
                raise ValueError("a genuine block must retain at least four cells")
            y, x, bottom, right = min(rr), min(cc), max(rr), max(cc)
            rect = (y, x, bottom-y+1, right-x+1, color)
            corners = ((y, x), (y, right), (bottom, x), (bottom, right))
            if any(inp[p] != color for p in corners):
                raise ValueError("missing corner makes the rectangular extent ambiguous")
            box = _mask(inp.shape, rect)
            if (occupied & box).any() or np.any(box & (inp != color) & (inp != bg)):
                raise ValueError("overlapping or obscured block extents")
            occupied |= box
            holes |= box & (inp == bg)
            rects.append(tuple(map(int, rect)))
    if not rects:
        raise ValueError("scene must contain a recoverable block")
    if (extra & occupied).any():
        raise ValueError("a speck cannot lie inside another block")
    rects.sort()
    return bg, rivet, set(selectors), rects, holes, extra


def _prepare(shape, rects, holes, selectors, use_rivets=True):
    ring = _halo(holes) if use_rivets else np.zeros(shape, dtype=bool)
    pinned, mobile, removed = [], [], []
    static = ring.copy()
    for rect in rects:
        box = _mask(shape, rect)
        if rect[4] in selectors:
            removed.append(rect)
        elif (box & ring).any():
            pinned.append(rect)
            static |= box & ~holes
        else:
            mobile.append(rect)
    return ring, static, pinned, mobile, removed


def _settle(mobile, static):
    """Greatest synchronous set of rigid rectangles able to descend one row."""
    h = static.shape[0]
    fixed = frozenset(map(tuple, np.argwhere(static)))
    result = list(mobile)
    for _ in range(h * (len(result) + 1)):
        cells = [_cells(r) for r in result]
        targets = [frozenset((y+1, x) for y, x in c) for c in cells]
        moving = {i for i, t in enumerate(targets)
                  if all(y < h for y, x in t) and not t & fixed}
        while True:
            stopped = {i for i in moving
                       if any(targets[i] & cells[j]
                              for j in range(len(cells)) if j not in moving)}
            if not stopped:
                break
            moving -= stopped
        if not moving:
            return result
        result = [_shift(r, 1) if i in moving else r for i, r in enumerate(result)]
    raise AssertionError("gravity did not converge within the finite scene")


def _settle_direct(mobile, static):
    """Construction/check path: settle lower rectangles directly onto terrain."""
    h = static.shape[0]
    terrain = set(map(tuple, np.argwhere(static)))
    result = list(mobile)
    order = sorted(range(len(result)), key=lambda i: result[i][0]+result[i][2],
                   reverse=True)
    for i in order:
        r = result[i]
        d = 0
        while r[0] + r[2] + d < h and not _cells(_shift(r, d+1)) & terrain:
            d += 1
        result[i] = _shift(r, d)
        terrain.update(_cells(result[i]))
    return result


def _render(inp, bg, rivet, pinned, mobile, holes, ring):
    out = np.full(inp.shape, bg, dtype=np.int64)
    out[:2] = inp[:2]
    for r in pinned:
        m = _mask(inp.shape, r) & ~holes
        out[m] = r[4]
    for r in mobile:
        out[_mask(inp.shape, r)] = r[4]
    out[ring] = rivet
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions, material identities and geometry come from this input."""
    inp = _grid(input_grid)
    bg, rivet, commands, rects, holes, _ = _decode(inp)
    ring, static, pinned, mobile, _ = _prepare(inp.shape, rects, holes, commands)
    settled = _settle(mobile, static)
    return _render(inp, bg, rivet, pinned, settled, holes, ring)


def _add_at(geometry, shape, y, x, h, w):
    r = (int(y), int(x), int(h), int(w))
    if (y < 2 or x < 0 or y+h > shape[0] or x+w > shape[1] or h*w < 4
            or any(_cells(r) & _cells(q) for q in geometry)):
        return False
    geometry.append(r)
    return True


def _drop(geometry, shape, h, w, x):
    r = (2, int(x), int(h), int(w))
    terrain = set().union(*(_cells(q) for q in geometry)) if geometry else set()
    if h*w < 4 or 2+h > shape[0] or _cells(r) & terrain:
        return False
    d = 0
    while 2+h+d < shape[0] and not _cells(_shift(r, d+1)) & terrain:
        d += 1
    geometry.append(_shift(r, d))
    return True


def _color(rng, geometry, available, equal_groups=()):
    """Adjacent blocks differ; separated blocks may share a deletion identity."""
    groups = [{i} for i in range(len(geometry))]
    for ids in equal_groups:
        merged = set(ids)
        groups = [g for g in groups if not g & merged] + [merged]
    rng.shuffle(groups)
    occupied = [_cells(r) for r in geometry]
    neighbor = [{(y+dy, x+dx) for y, x in c
                 for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))} for c in occupied]
    assigned = {}
    for group in groups:
        if any(neighbor[i] & occupied[j] for i in group for j in group if i != j):
            return None
        forbidden = {assigned[j] for i in group for j in assigned
                     if neighbor[i] & occupied[j]}
        choices = [c for c in available if c not in forbidden]
        if not choices:
            return None
        c = int(rng.choice(choices))
        assigned.update((i, c) for i in group)
    return [(*r, assigned[i]) for i, r in enumerate(geometry)]


def _add_holes(rng, holes, rect, count, edge=False):
    y, x, h, w = rect[:4]
    if min(h, w) < 3:
        return
    corners = {(y, x), (y, x+w-1), (y+h-1, x), (y+h-1, x+w-1)}
    choices = [(a, b) for a, b in sorted(_cells(rect)) if (a, b) not in corners]
    rng.shuffle(choices)
    if edge:
        choices.sort(key=lambda p: not (p[0] in (y, y+h-1) or p[1] in (x, x+w-1)))
    box = _mask(holes.shape, rect)
    for p in choices:
        if count <= 0:
            break
        if holes[p]:
            continue
        holes[p] = True
        remaining = box & ~holes
        if remaining.sum() < 4 or len(_components(remaining)) != 1:
            holes[p] = False
        else:
            count -= 1


def _construct(rng, shape, rects, holes, commands, bg, rivet, extras=True):
    inp = np.full(shape, bg, dtype=np.int64)
    inp[0, 0] = rivet
    positions = rng.choice(np.arange(2, shape[1]), len(commands), replace=False)
    values = list(commands)
    rng.shuffle(values)
    for p, color in zip(positions, values):
        inp[0, p] = color
    for r in rects:
        inp[_mask(shape, r)] = r[4]
    inp[holes] = bg
    # Isolated single specks never change a true block's visible extent.
    if extras:
        wanted = int(rng.integers(1, 6))
        choices = np.argwhere((inp == bg) & ~holes)
        rng.shuffle(choices)
        count = 0
        for y, x in choices:
            if y < 2:
                continue
            color = int(rng.choice([r[4] for r in rects]))
            neighbors = [(a, b) for a, b in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                         if 2 <= a < shape[0] and 0 <= b < shape[1]]
            if any(inp[p] == color for p in neighbors):
                continue
            # Specks of any identity cannot intrude into a recovered rectangle.
            if any(_mask(shape, r)[y, x] for r in rects):
                continue
            inp[y, x] = color
            count += 1
            if count == wanted:
                break
    ring, static, pinned, mobile, _ = _prepare(shape, rects, holes, set(commands))
    out = _render(inp, bg, rivet, pinned, _settle_direct(mobile, static), holes, ring)
    decoded = _decode(inp)
    if set(decoded[3]) != set(rects) or not np.array_equal(decoded[4], holes):
        return None
    if not np.array_equal(solve(inp), out):
        return None
    return {"input": inp, "output": out}


def _anchor(rng, kind, bg, rivet, available):
    """Two small teaching networks, with randomized spans, placement and caps."""
    for _ in range(80):
        H, W = int(rng.integers(17, 29)), int(rng.integers(14, 29))
        shape, geom = (H, W), []
        holes = np.zeros(shape, dtype=bool)
        if kind == "fossil":
            ph, pw = int(rng.integers(6, min(11, H-6))), int(rng.integers(3, 6))
            bw = int(rng.integers(pw+3, min(W, pw+12)+1))
            bx = int(rng.integers(0, W-bw+1))
            px = int(rng.integers(bx+1, bx+bw-pw))
            top = H-ph
            _add_at(geom, shape, top, px, ph, pw)
            bh = int(rng.integers(1, 4))
            _add_at(geom, shape, top-bh, bx, bh, bw)
            # One hole establishes a fossil landing; a bottom-edge hole shows
            # clipping.  A diagonal pair optionally merges their surroundings.
            offset = int(rng.integers(2, ph-2))
            cx = px+int(rng.integers(1, pw-1))
            holes[top+offset, cx] = True
            holes[H-1, cx] = True
            if pw >= 4 and offset < ph-3 and rng.random() < .65:
                holes[top+offset+1, cx-1] = True
            # Unequal upper rectangles share a real support network.
            n = int(rng.integers(0, 4))
            for _ in range(n):
                rw = int(rng.integers(2, min(bw, 7)+1))
                rh = int(rng.integers(2, 5))
                _drop(geom, shape, rh, rw, int(rng.integers(bx, bx+bw-rw+1)))
            rects = _color(rng, geom, available)
            if rects is None:
                continue
            commands = [rects[0][4]]
        else:
            ph, pw = int(rng.integers(4, 8)), int(rng.integers(1, 4))
            cw, ch = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            left = int(rng.integers(0, 3))
            right = int(rng.integers(max(left+cw+2, W-cw-3), W-cw+1))
            top = H-ph
            _add_at(geom, shape, top, left+(cw-pw)//2, ph, pw)
            _add_at(geom, shape, top, right+(cw-pw)//2, ph, pw)
            _add_at(geom, shape, top-ch, left, ch, cw)
            _add_at(geom, shape, top-ch, right, ch, cw)
            # A shared shelf above both caps keeps this a connected network.
            _add_at(geom, shape, top-ch-1, left, 1, right+cw-left)
            # A top-edge defect reaches the shared shelf above it.  Removing
            # the defective cap leaves that otherwise intact shelf visibly
            # riveted in midair, teaching pinning of a neighboring block.
            holes[top-ch, left+cw//2] = True
            _add_holes(rng, holes, geom[2], int(rng.integers(0, 3)), edge=True)
            rects = _color(rng, geom, available, equal_groups=((0, 1),))
            if rects is None:
                continue
            # Both pedestals and the damaged cap disappear; the right cap
            # falls while the shared shelf remains attached to the fossil.
            commands = [rects[0][4], rects[2][4]]
            if (len(set(commands)) != 2 or rects[3][4] in commands
                    or rects[4][4] in commands):
                continue
            absent = [c for c in available if c not in {r[4] for r in rects}]
            if absent and rng.random() < .5:
                commands.append(int(rng.choice(absent)))
        pair = _construct(rng, shape, rects, holes, commands, bg, rivet)
        if pair is None:
            continue
        f = _features(pair)
        if kind == "fossil" and f["fossil_landing"] and f["moved"]:
            return pair
        if kind == "pin" and f["unsupported_pin"] and f["moved"] and f["erased_specks"]:
            return pair
    raise RuntimeError("failed to produce the required teaching network")


def _random_pair(rng, bg, rivet, available, rich=False):
    for _ in range(100):
        H = int(rng.integers(15 if rich else 10, 31))
        W = int(rng.integers(15 if rich else 8, 31))
        shape, geom = (H, W), []
        wanted = int(rng.integers(7, 17) if rich else rng.integers(3, 10))
        # Free deposition composes shelves, thick pillars, slabs and overhangs;
        # there is no fixed partition into slots or designated upper payload.
        for _ in range(wanted*6):
            if len(geom) >= wanted:
                break
            kind = int(rng.integers(3))
            if kind == 0:
                rh, rw = int(rng.integers(1, 4)), int(rng.integers(4, W+1))
            elif kind == 1:
                rh, rw = int(rng.integers(4, min(11, H-3)+1)), int(rng.integers(1, 5))
            else:
                rh, rw = int(rng.integers(2, 7)), int(rng.integers(2, min(8, W)+1))
            if rw <= W:
                _drop(geom, shape, rh, rw, int(rng.integers(0, W-rw+1)))
        if len(geom) < (7 if rich else 3):
            continue
        rects = _color(rng, geom, available)
        if rects is None:
            continue
        holes = np.zeros(shape, dtype=bool)
        candidates = [r for r in rects if min(r[2:4]) >= 3]
        rng.shuffle(candidates)
        for r in candidates[:int(rng.integers(1, min(len(candidates), 5)+1))] if candidates else []:
            _add_holes(rng, holes, r, int(rng.integers(1, min(7, r[2]*r[3]//5)+1)),
                       edge=bool(rng.random() < .5))
        colors = list(available)
        n = int(rng.integers(1, 4))
        commands = list(map(int, rng.choice(colors, n, replace=False)))
        pair = _construct(rng, shape, rects, holes, commands, bg, rivet)
        if pair is not None and holes.any():
            return pair
    raise RuntimeError("bounded block deposition exhausted")


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    # A palette convention is shared by a puzzle; role assignment is also
    # visibly encoded, so solve supports other palettes without any training.
    bg, rivet = palette[:2]
    available = palette[2:]
    puzzle = {"train": [], "test": []}
    seen = set()
    for kind in ("fossil", "pin"):
        p = _anchor(rng, kind, bg, rivet, available)
        puzzle["train"].append(p)
        seen.add(_signature(p["input"]))
    for split, count in (("train", int(num_train)-2), ("test", int(num_test))):
        for _ in range(count):
            for attempt in range(80):
                p = _random_pair(rng, bg, rivet, available, rich=split == "test")
                f = _features(p)
                if (_signature(p["input"]) not in seen
                        and (split == "train" or (f["removed"] and f["moved"]
                                                 and f["a_changes_positions"]))):
                    puzzle[split].append(p)
                    seen.add(_signature(p["input"]))
                    break
            else:
                raise RuntimeError("unable to construct a distinct interacting example")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp = np.asarray(pair["input"])
    bg, rivet, commands, rects, holes, extras = _decode(inp)
    ring, static, pinned, mobile, removed = _prepare(inp.shape, rects, holes, commands)
    final = _settle(mobile, static)
    distances = [b[0]-a[0] for a, b in zip(mobile, final)]
    _, bare, _, bare_mobile, _ = _prepare(inp.shape, rects, holes, commands, False)
    bare_final = _settle(bare_mobile, bare)
    bare_rows = {r: s[0] for r, s in zip(bare_mobile, bare_final)}
    position_changes = any(bare_rows[r] != r[0] for r in pinned) or any(
        bare_rows[r] != s[0] for r, s in zip(mobile, final))
    pinned_body = np.zeros(inp.shape, dtype=bool)
    for r in pinned:
        pinned_body |= _mask(inp.shape, r) & ~holes
    removed_body = np.zeros(inp.shape, dtype=bool)
    for r in removed:
        removed_body |= _mask(inp.shape, r)
    ghost = _halo(holes & removed_body) & ~holes & ~pinned_body
    ghost_cells = set(map(tuple, np.argwhere(ghost)))
    fossil_landing = any(d > 0 and any((y+1, x) in ghost_cells
                                      for y, x in _cells(s))
                         for d, s in zip(distances, final))
    unsupported_pin = any(bare_rows[r] > r[0] for r in pinned)
    secondary_pins = [r for r in pinned if not (_mask(inp.shape, r) & holes).any()]
    palette = {r[4] for r in rects}
    groups = _components(holes)
    overlap_counts = np.zeros(inp.shape, dtype=np.int16)
    for y, x in np.argwhere(holes):
        overlap_counts[max(2, y-1):y+2, max(0, x-1):x+2] += 1
    return {"shape": list(inp.shape), "blocks": len(rects), "holes": int(holes.sum()),
            "hole_groups": len(groups), "extras": int(extras.sum()),
            "erased_specks": int((extras & ~ring).sum()),
            "commands": len(commands), "removed": len(removed), "pinned": len(pinned),
            "moved": sum(d > 0 for d in distances),
            "fall_distances": sorted(set(distances)-{0}),
            "fossil_landing": fossil_landing, "unsupported_pin": unsupported_pin,
            "secondary_pins": len(secondary_pins),
            "unsupported_secondary_pin": any(bare_rows[r] > r[0] for r in secondary_pins),
            "a_changes_positions": position_changes, "ghost_cells": int(ghost.sum()),
            "overlap": bool(((overlap_counts > 1) & ring).any()),
            "clipping": bool(holes[-1].any() or holes[:, 0].any() or holes[:, -1].any()),
            "repeated_identity": len(palette) < len(rects),
            "absent_selector": bool(commands-palette),
            "widths": sorted({r[3] for r in rects}), "heights": sorted({r[2] for r in rects})}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions, train_features = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and np.array_equal(inp[:2], out[:2])
            key = _signature(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            bg, rivet, commands, rects, holes, extras = _decode(inp)
            conventions.add((bg, rivet))
            # Intact source rectangles already form a stable pile before damage.
            empty = np.zeros(inp.shape, dtype=bool)
            assert _settle(rects, empty) == rects, "input pile was not initially supported"
            ring, static, pinned, mobile, removed = _prepare(inp.shape, rects, holes, commands)
            final = _settle(mobile, static)
            assert final == _settle_direct(mobile, static), "settling implementations disagree"
            assert _settle(final, static) == final, "unsettled output"
            occupied = set(map(tuple, np.argwhere(static)))
            for before, after in zip(mobile, final):
                assert after[0] >= before[0] and after[1:] == before[1:]
                assert not occupied & _cells(after)
                occupied.update(_cells(after))
            assert np.array_equal(out, solve(inp))
            assert np.all(out[ring] == rivet) and np.all(out[holes] == bg)
            assert not np.array_equal(inp, out)
            f = _features(pair)
            if split == "train":
                train_features.append(f)
            else:
                assert f["moved"] and f["removed"] and f["a_changes_positions"]
    assert len(conventions) == 1, "generated puzzle must share background/rivet roles"
    assert any(f["fossil_landing"] for f in train_features), "teach fossil support"
    assert any(f["unsupported_pin"] for f in train_features), "teach anchoring"
    assert any(f["unsupported_secondary_pin"] for f in train_features), "teach neighbor anchoring"
    assert any(f["erased_specks"] for f in train_features), "teach visible speck cleanup"
    assert any(f["moved"] for f in train_features), "teach gravity"


def _fingerprint(puzzle):
    return tuple((_signature(p["input"]), _signature(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed audit plus count, symmetry and diagnostic regressions."""
    shapes, structures, fingerprints, palettes = set(), set(), set(), set()
    records = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            inp = pair["input"]
            f = _features(pair)
            records.append(f)
            shapes.add(inp.shape)
            bg, rivet, _, rects, holes, _ = _decode(inp)
            palettes.add((bg, rivet))
            clean = np.zeros(inp.shape, dtype=np.uint8)
            for r in rects:
                clean[_mask(inp.shape, r)] = 1
            # Shape-plus-occupancy ignores all literal color choices.
            structures.add(_signature(clean))
            reflected = np.fliplr(inp).copy()
            # Reflection reverses the scene and selectors while keeping the
            # two-cell legend in its fixed location.
            reflected[:2] = inp[:2]
            expected = np.fliplr(pair["output"]).copy()
            expected[:2] = pair["output"][:2]
            assert np.array_equal(solve(reflected), expected)
    assert len(fingerprints) == 42 and len(shapes) >= 40 and len(structures) >= 170
    assert len(palettes) >= 25
    assert len({f["blocks"] for f in records}) >= 8
    assert len({f["holes"] for f in records}) >= 10
    assert len({f["extras"] for f in records}) >= 4
    assert len({d for f in records for d in f["fall_distances"]}) >= 7
    assert any(len(f["fall_distances"]) >= 2 for f in records)
    for flag in ("fossil_landing", "unsupported_pin", "unsupported_secondary_pin", "overlap", "clipping",
                 "repeated_identity", "absent_selector", "a_changes_positions"):
        assert sum(bool(f[flag]) for f in records) >= 10, flag
    assert {f["commands"] for f in records} == {1, 2, 3}
    assert min(h for h, w in shapes) <= 12 and max(h for h, w in shapes) == 30
    assert _fingerprint(generate_puzzle(13579)) == _fingerprint(generate_puzzle(13579))
    assert _fingerprint(generate_puzzle(13579)) != _fingerprint(generate_puzzle(13580))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(37, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Hand-specified ring/support oracle, not a generated example.
    inp = np.zeros((12, 9), dtype=int)
    inp[0, 0], inp[0, 4] = 7, 2
    inp[5:, 3:6] = 2
    inp[4, 1:8] = 3
    inp[7, 4] = 0
    expected = np.zeros_like(inp)
    expected[:2] = inp[:2]
    expected[5, 1:8] = 3
    expected[6:9, 3:6] = 7
    expected[7, 4] = 0
    assert np.array_equal(solve(inp), expected)
    # A diagonal ring cell rivets an intact beam after both supports disappear.
    inp = np.zeros((12, 12), dtype=int)
    inp[0, 0], inp[0, 4], inp[0, 6] = 7, 2, 3
    inp[8:, 3] = 2
    inp[5:, 8] = 2
    inp[5:8, 2:5] = 3
    inp[5, 3] = 0
    inp[4, 4:10] = 4
    expected = np.zeros_like(inp)
    expected[:2] = inp[:2]
    expected[4, 4:10] = 4
    expected[4:7, 2:5] = 7
    expected[5, 3] = 0
    assert np.array_equal(solve(inp), expected)
    # L-shaped adjacent defects keep all three centers empty after unioning.
    inp = np.zeros((12, 9), dtype=int)
    inp[0, 0], inp[0, 4] = 7, 2
    inp[6:, 2:7] = 2
    inp[8, 3:5] = 0
    inp[9, 4] = 0
    expected = np.zeros_like(inp)
    expected[:2] = inp[:2]
    expected[7, 2:6] = 7
    expected[8, [2, 5]] = 7
    expected[9, [2, 3, 5]] = 7
    expected[10, 3:6] = 7
    assert np.array_equal(solve(inp), expected)
    # Deleting a pedestal cannot move a surviving block riveted through a hole.
    inp = np.zeros((12, 9), dtype=int)
    inp[0, 0], inp[0, 4] = 7, 2
    inp[8:, 4] = 2
    inp[4:8, 2:7] = 3
    inp[5, 4] = 0
    expected = inp.copy()
    expected[8:, 4] = 0
    expected[4:7, 3:6] = 7
    expected[5, 4] = 0
    assert np.array_equal(solve(inp), expected)
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "clean_structures": len(structures),
            "palettes": len(palettes),
            "block_counts": sorted({f["blocks"] for f in records}),
            "hole_counts": sorted({f["holes"] for f in records}),
            "fall_distances": sorted({d for f in records for d in f["fall_distances"]}),
            "effects": {k: sum(bool(f[k]) for f in records)
                        for k in ("fossil_landing", "unsupported_pin", "unsupported_secondary_pin",
                                  "overlap", "clipping")},
            "semantic_verification": "author-owned checks; no independent solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
