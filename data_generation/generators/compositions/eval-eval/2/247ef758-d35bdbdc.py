"""Two-step transfer atlases. Only standard-library modules and NumPy are used.

All geometry and conventions are recoverable from an individual input. No parent
generator is imported, and no sampled metadata is retained by solve().
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


_D8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D8:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    candidates = []
    # A constant divider followed immediately by a framed, empty rectangle.
    for d in range(3, w - 7):
        if not np.all(a[:, d] == a[0, d]):
            continue
        interior = a[1:-1, d + 2:-1]
        if not interior.size or not np.all(interior == interior[0, 0]):
            continue
        bg, divider, frame = int(interior[0, 0]), int(a[0, d]), int(a[0, d + 1])
        if len({bg, divider, frame}) != 3:
            continue
        if not all(a[r, c] == frame for r, c in ((0, d+1), (0, w-1), (h-1, d+1), (h-1, w-1))):
            continue
        if not np.array_equal(a[0, d+2:-1], a[-1, d+2:-1]):
            continue
        if not np.array_equal(a[1:-1, d+1], a[1:-1, -1]):
            continue
        stage = a[:, :d]
        if not np.any(stage != bg) or np.any((stage == divider) | (stage == frame)):
            continue
        edges = np.concatenate((a[0, d+1:], a[-1, d+1:], a[1:-1, d+1], a[1:-1, -1]))
        if np.any((edges == bg) | (edges == divider)):
            continue
        candidates.append((d, bg, divider, frame))
    assert len(candidates) == 1, "unique divider/frame geometry required"
    d, bg, divider, frame = candidates[0]
    rows, cols = {}, {}
    for r in range(1, h-1):
        color = int(a[r, d+1])
        if color != frame:
            rows.setdefault(color, []).append(r)
    for c in range(d+2, w-1):
        color = int(a[0, c])
        if color != frame:
            cols.setdefault(color, []).append(c)
    glyphs = []
    for cells in _components(map(tuple, np.argwhere(a[:, :d] != bg))):
        locations = np.array(sorted(cells), dtype=int)
        lo, hi = locations.min(axis=0), locations.max(axis=0)
        shape = tuple(map(int, hi-lo+1))
        assert all(s % 2 == 1 for s in shape) and len(cells) >= 3
        center = tuple(map(int, (lo+hi)//2))
        assert center in cells, "center is a visible instruction pixel"
        body_values = {int(a[p]) for p in cells if p != center}
        assert len(body_values) == 1
        body, value = body_values.pop(), int(a[center])
        assert body not in (bg, divider, frame) and value not in (bg, divider, frame)
        glyphs.append(dict(cells=cells, lo=tuple(map(int, lo)), center=center,
                           body=body, value=value, shape=shape,
                           offsets={(r-center[0], c-center[1]) for r, c in cells}))
    glyphs.sort(key=lambda t: t["lo"])
    assert glyphs and len({t["body"] for t in glyphs}) == len(glyphs), "one source per body identity"
    by_body = {t["body"]: t for t in glyphs}
    for t in glyphs:
        donor = by_body.get(t["value"])
        if donor is t:
            donor = None
        t["donor"] = donor
        t["final"] = None if donor is None else donor["value"]
        t["rows"] = rows.get(t["body"], [])
        t["cols"] = [] if donor is None else cols.get(t["final"], [])
        t["active"] = donor is not None and bool(t["rows"]) and bool(t["cols"])
        if t["active"]:
            for axis, positions in enumerate((t["rows"], t["cols"])):
                assert all(y-x >= t["shape"][axis] for x, y in zip(positions, positions[1:])), "copies of one glyph cannot overlap"
            for r, c in itertools.product(t["rows"], t["cols"]):
                for dr, dc in t["offsets"]:
                    assert 1 <= r+dr < h-1 and d+2 <= c+dc < w-1, "stamp must fit inside frame"
    return dict(grid=a, d=d, bg=bg, divider=divider, frame=frame,
                rows=rows, cols=cols, glyphs=glyphs)


def _render(p, mode="correct"):
    """Input interpretation, plus deliberately wrong rules for witness checks."""
    out = p["grid"].copy()
    if mode == "in_place":
        for t in p["glyphs"]:
            if t["active"]:
                out[t["center"]] = t["final"]
        return out
    selected = []
    for t in p["glyphs"]:
        final = t["final"]
        rs = t["rows"]
        cs = t["cols"]
        if mode == "direct":
            final = t["value"]
            cs = p["cols"].get(final, [])
        elif mode == "same_axes":
            cs = p["cols"].get(t["body"], []) if final is not None else []
        elif mode == "allow_self" and t["body"] == t["value"]:
            final = t["value"]
            cs = p["cols"].get(final, [])
        elif mode == "zip_axes":
            pass
        elif mode == "recursive" and final is not None:
            # Exactly one extra hop is already a wrong rule; no recursive loop.
            second = next((s for s in p["glyphs"] if s["body"] == final), None)
            if second is not None:
                final = second["value"]
                cs = p["cols"].get(final, [])
        if final is None or not rs or not cs:
            continue
        for cell in t["cells"]:
            out[cell] = p["bg"]
        centers = list(zip(rs, cs)) if mode == "zip_axes" else list(itertools.product(rs, cs))
        pixels = {}
        for r, c in centers:
            for dr, dc in t["offsets"]:
                q = r+dr, c+dc
                if 1 <= q[0] < out.shape[0]-1 and p["d"]+2 <= q[1] < out.shape[1]-1:
                    pixels[q] = final if (dr, dc) == (0, 0) else t["body"]
        selected.append((t, pixels))
    if mode == "destination_order":
        selected.sort(key=lambda item: (min(item[0]["rows"]), min(item[0]["cols"])))
    elif mode == "size_order":
        selected.sort(key=lambda item: item[0]["shape"][0]*item[0]["shape"][1])
    elif mode == "reverse_order":
        selected.reverse()
    occupied = set()
    for _, pixels in selected:
        for cell, color in pixels.items():
            if cell not in occupied:
                out[cell] = color
                occupied.add(cell)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the atlas using only the supplied visible input."""
    return _render(_parse(input_grid))


def _mask(rng, shape):
    h, w = shape
    r0, c0 = h//2, w//2
    m = np.zeros(shape, dtype=bool)
    kind = int(rng.integers(6))
    if min(shape) == 1 or kind == 0:
        m[:] = True
    elif kind == 1:
        m[[0, -1], :] = True
        m[:, [0, -1]] = True
        m[:r0+1, c0] = True
    elif kind == 2:
        m[r0, :] = True
        m[:, c0] = True
    elif kind == 3 and h == w:
        m = np.eye(h, dtype=bool)
    else:
        m[r0, c0] = True
        # Connect the center to four independently chosen boundary cells.
        targets = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                   (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        for tr, tc in targets:
            r, c = r0, c0
            while (r, c) != (tr, tc):
                dr, dc = int(np.sign(tr-r)), int(np.sign(tc-c))
                if dr and dc and rng.random() < .4:
                    if rng.random() < .5:
                        dr = 0
                    else:
                        dc = 0
                r, c = r+dr, c+dc
                m[r, c] = True
        m |= rng.random(shape) < float(rng.uniform(.05, .55))
    m[r0, c0] = True
    if rng.random() < .5:
        m = m[::-1]
    if rng.random() < .5:
        m = m[:, ::-1]
    # Remove any isolated random specks without changing the tight odd box.
    central = next(c for c in _components(map(tuple, np.argwhere(m))) if (r0, c0) in c)
    result = np.zeros(shape, dtype=bool)
    for q in central:
        result[q] = True
    assert np.any(result[0]) and np.any(result[-1]) and np.any(result[:, 0]) and np.any(result[:, -1])
    return result


def _pack(rng, masks, h, sw, priority=False):
    """Bounded free placement; a small deterministic packing search is fallback."""
    n = len(masks)
    for attempt in range(24):
        occupied = np.zeros((h, sw), dtype=bool)
        locations = [None]*n
        order = sorted(range(n), key=lambda i: -masks[i].size)
        good = True
        for i in order:
            bh, bw = masks[i].shape
            choices = [(r, c) for r in range(1, h-bh) for c in range(sw-bw+1)
                       if not occupied[max(0, r-1):r+bh+1, max(0, c-1):c+bw+1].any()]
            if not choices:
                good = False
                break
            r, c = choices[int(rng.integers(len(choices)))]
            occupied[r:r+bh, c:c+bw] = True
            locations[i] = (r, c)
        if good and (not priority or locations[0] < locations[1]):
            return masks, locations
    # General scenes may simplify their geometry to 3x3 after packing failures.
    if not priority:
        masks = [_mask(rng, (3, 3)) for _ in masks]
        locations = [(1+4*(i//2), (sw-3)*(i % 2)) for i in range(n)]
        return masks, locations
    # Priority scene has five glyphs: 5x5, 3x3, 3x3, 1x3, 1x3.
    locations = []
    top = 1
    for m in masks:
        locations.append((top, int(rng.integers(sw-m.shape[1]+1))))
        top += m.shape[0]+1
    assert top-1 < h
    return masks, locations


def _allocate_axis(rng, length, counts, spacing, fixed=None):
    """Choose named coordinates, reserving each border cell for one color."""
    fixed = {} if fixed is None else fixed
    result = {k: sorted(v) for k, v in fixed.items()}
    occupied = {v for values in result.values() for v in values}
    for key in sorted(counts, key=lambda k: -counts[k]):
        if key in result:
            continue
        wanted = counts[key]
        choices = [p for p in range(3, length-3) if p not in occupied]
        possibilities = [c for c in itertools.combinations(choices, wanted)
                         if all(b-a >= spacing.get(key, 1) for a, b in zip(c, c[1:]))]
        if not possibilities:
            return None
        coordinates = list(possibilities[int(rng.integers(len(possibilities)))])
        result[key] = coordinates
        occupied.update(coordinates)
    return result


def _model_pair(rng, regime, shift):
    """Construct expected pixels from sampled records, without calling solve."""
    special = regime in ("chain", "priority")
    n = (6 if regime == "chain" else 5) if special else int(rng.integers(3, 7))
    if regime == "test" and shift == "cycle":
        n = int(rng.integers(4, 7))
    sw = int(rng.integers(7, 10))
    if shift == "height":
        h = int(rng.integers(25, 31) if regime == "test" else rng.integers(20, 25))
    else:
        h = int(rng.integers(20, 31))
    fw = int(rng.integers(16 if special else 14, 30-sw))
    palette = list(map(int, rng.permutation(10)))
    bg, divider, frame = palette[:3]
    ids = palette[3:]
    bodies = ids[:n]
    values = list(map(int, rng.choice(ids, n)))
    graph_kind = str(rng.choice(["general", "cycle", "chain", "fan"]))
    if special:
        # E is an absent body; F is a monochrome donor in the chain scene.
        bodies = [ids[i] for i in ([0, 1, 2, 3, 5, 6] if regime == "chain" else [0, 1, 2, 3, 5])]
        values = ([ids[1], ids[2], ids[3], ids[4], ids[5], ids[5]] if regime == "chain"
                  else [ids[1], ids[0], ids[0], ids[4], ids[3]])
    elif graph_kind == "cycle" or regime == "test" and shift == "cycle":
        values = [bodies[(i+1) % n] for i in range(n)]
    elif graph_kind == "chain":
        values = [ids[i+1] for i in range(n)]
    else:
        values[0], values[1] = ids[1], ids[2]
        if graph_kind == "fan":
            values[2] = ids[1]
    shapes = [(1, 3), (3, 1), (3, 3), (3, 5), (5, 3), (5, 5)]
    masks = [_mask(rng, shapes[int(rng.choice(6, p=[.07, .07, .43, .16, .16, .11]))]) for _ in range(n)]
    if regime == "chain":
        masks[4:] = [_mask(rng, (1, 3)) for _ in range(2)]
    if regime == "priority":
        masks[0] = np.ones((5, 5), dtype=bool)
        masks[0][1:-1, 1:-1] = False
        masks[0][2, 2] = True  # 8-connected to the ring through this spoke.
        masks[0][1, 2] = True
        masks[1] = np.ones((3, 3), dtype=bool)
        masks[2] = _mask(rng, (3, 3))
        masks[3:] = [_mask(rng, (1, 3)) for _ in range(2)]
    masks, locations = _pack(rng, masks, h, sw, regime == "priority")
    lookup = dict(zip(bodies, values))
    finals = [lookup.get(v) if v != b else None for b, v in zip(bodies, values)]
    row_counts, col_counts = {}, {}
    if regime == "chain":
        row_counts = {ids[0]: 2, ids[1]: 1, ids[3]: 1, ids[5]: 1, ids[6]: 1}
        col_counts = {ids[2]: 2, ids[3]: 2, ids[1]: 1, ids[4]: 1, ids[5]: 1}
    elif regime == "priority":
        row_counts = {ids[0]: 1, ids[1]: 1, ids[3]: 1, ids[5]: 1}
        col_counts = {ids[2]: 1, ids[0]: 1, ids[1]: 1}
    else:
        # Axis availability is independent of source position and mask shape.
        for color in bodies:
            if rng.random() < .70:
                row_counts[color] = int(rng.choice([1, 2, 3], p=[.64, .29, .07]))
        for color in ids:
            if rng.random() < .58:
                col_counts[color] = int(rng.choice([1, 2, 3], p=[.64, .29, .07]))
        row_counts[ids[0]] = max(1, row_counts.get(ids[0], 0))
        col_counts[finals[0]] = max(1, col_counts.get(finals[0], 0))
    row_spacing = dict(zip(bodies, [m.shape[0] for m in masks]))
    col_spacing = {}
    for m, final in zip(masks, finals):
        if final is not None:
            col_spacing[final] = max(col_spacing.get(final, 1), m.shape[1])
    fixed_rows, fixed_cols = {}, {}
    if regime == "chain":
        fixed_rows = {ids[0]: [3, h-4], ids[1]: [h//2]}
    elif regime == "priority":
        r = int(rng.integers(5, h-5))
        c = int(rng.integers(5, fw-5))
        # Earlier, larger source lands lower and to the right than the next.
        fixed_rows = {ids[0]: [r+1], ids[1]: [r]}
        fixed_cols = {ids[0]: [c+1], ids[1]: [c]}
    rows = _allocate_axis(rng, h, row_counts, row_spacing, fixed_rows)
    cols = _allocate_axis(rng, fw, col_counts, col_spacing, fixed_cols)
    if rows is None or cols is None:
        return None
    w = sw+1+fw
    a = np.full((h, w), bg, dtype=np.int8)
    a[:, sw] = divider
    a[[0, -1], sw+1:] = frame
    a[:, [sw+1, w-1]] = frame
    for color, coordinates in rows.items():
        for r in coordinates:
            a[r, sw+1] = a[r, -1] = color
    for color, coordinates in cols.items():
        for c in coordinates:
            a[0, sw+1+c] = a[-1, sw+1+c] = color
    records = []
    for body, value, mask, (top, left), final in zip(bodies, values, masks, locations, finals):
        cells = [(top+int(r), left+int(c)) for r, c in np.argwhere(mask)]
        center = top+mask.shape[0]//2, left+mask.shape[1]//2
        for q in cells:
            a[q] = body
        a[center] = value
        records.append(dict(body=body, final=final, cells=cells, center=center, top=top, left=left))
    out = a.copy()
    active = [t for t in records if t["final"] is not None and rows.get(t["body"]) and cols.get(t["final"])]
    for t in active:
        for q in t["cells"]:
            out[q] = bg
    # Reverse painting with sampled records is separate from solver occupancy.
    for t in sorted(active, key=lambda x: (x["top"], x["left"]), reverse=True):
        for r, c in itertools.product(rows[t["body"]], cols[t["final"]]):
            for y, x in t["cells"]:
                q = r+y-t["center"][0], sw+1+c+x-t["center"][1]
                out[q] = t["final"] if (y, x) == t["center"] else t["body"]
    return dict(input=a, output=out)


def _features(a):
    p = _parse(a)
    correct = _render(p)
    ts = p["glyphs"]
    active = [t for t in ts if t["active"]]
    cover = []
    for t in active:
        cover.append({(r+dr, c+dc) for r, c in itertools.product(t["rows"], t["cols"])
                      for dr, dc in t["offsets"]})
    wrong = {mode: bool(np.any(correct != _render(p, mode))) for mode in
             ("direct", "same_axes", "zip_axes", "recursive", "destination_order", "size_order", "reverse_order", "allow_self", "in_place")}
    return dict(shape=a.shape, roles=(p["bg"], p["divider"], p["frame"]),
                count=len(ts), active=len(active),
                motifs=tuple((t["shape"], tuple(sorted(t["offsets"]))) for t in ts),
                products=tuple((len(t["rows"]), len(t["cols"])) for t in active),
                relations=tuple((t["body"], t["value"], t["final"]) for t in ts),
                conflicts=sum(len(a & b) for i, a in enumerate(cover) for b in cover[i+1:]),
                moving_donor=any(t["donor"]["active"] and t["donor"]["final"] != t["donor"]["value"] for t in active),
                remote_donor=any(not t["donor"]["active"] for t in active),
                failed_donor=any(t["donor"] is None and t["rows"] for t in ts),
                missing_rows=any(t["donor"] is not None and not t["rows"] and p["cols"].get(t["final"]) for t in ts),
                missing_cols=any(t["donor"] is not None and t["rows"] and not t["cols"] for t in ts),
                monochrome=any(t["body"] == t["value"] for t in ts),
                monochrome_donor=any(t["donor"]["body"] == t["donor"]["value"] for t in active),
                newly_solid=any(t["body"] == t["final"] and t["body"] != t["value"] for t in active),
                shared_donor=any(sum(t["donor"] is s for t in active) >= 2 for s in ts),
                inactive_first=not ts[0]["active"], wrong=wrong)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, ne = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not 2 <= nt <= 10 or not 1 <= ne <= 4 or nt <= ne:
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed % (1 << 64))
    shift = str(rng.choice(["height", "cycle"]))
    modes = ["chain", "priority"] + ["random"]*(nt-2)
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            mode = modes[i] if split == "train" else "test"
            for _ in range(96):
                pair = _model_pair(rng, mode, shift)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                p = _parse(pair["input"])
                assert np.array_equal(_render(p), pair["output"]), "construction/parser disagreement"
                f = _features(pair["input"])
                if mode == "chain" and not (f["moving_donor"] and f["remote_donor"] and f["missing_rows"] and
                                             f["wrong"]["direct"] and f["wrong"]["recursive"] and f["wrong"]["zip_axes"]):
                    continue
                if mode == "priority" and not all(f["wrong"][k] for k in ("destination_order", "size_order", "reverse_order")):
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("bounded construction attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check pixels, preservation, simultaneous reads, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            p = _parse(a)
            expected = _render(p)
            assert np.array_equal(expected, b), "wrong transfer, mask, center, priority, or preservation"
            assert not np.array_equal(a, b)
            d = p["d"]
            assert np.array_equal(a[:, d:d+2], b[:, d:d+2])
            assert np.array_equal(a[[0, -1], d+1:], b[[0, -1], d+1:])
            assert np.array_equal(a[:, -1], b[:, -1])
            for t in p["glyphs"]:
                if t["active"]:
                    assert all(b[q] == p["bg"] for q in t["cells"])
                else:
                    assert all(b[q] == a[q] for q in t["cells"])
            features[split].append(_features(a))
    train = features["train"]
    for shortcut in ("direct", "same_axes", "zip_axes", "recursive", "destination_order", "size_order", "reverse_order", "allow_self", "in_place"):
        assert any(f["wrong"][shortcut] for f in train), "missing distinguishing training: " + shortcut
    for witness in ("moving_donor", "remote_donor", "failed_donor", "missing_rows", "missing_cols", "monochrome_donor", "newly_solid"):
        assert any(f[witness] for f in train), "missing training witness: " + witness
    assert any(any(r >= 2 and c >= 2 for r, c in f["products"]) for f in train)


def self_test():
    """Streaming 42-seed audit: no files, traces, or saved puzzle collection."""
    shapes, motifs, roles, products, counts, signatures = set(), set(), set(), set(), set(), set()
    observed = {k: 0 for k in ("shared_donor", "monochrome", "missing_cols", "inactive_first", "conflicts")}
    within = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        local_counts = set()
        for split in ("train", "test"):
            for pair in p[split]:
                a = np.asarray(pair["input"])
                f = _features(a)
                digest.update(a.tobytes())
                shapes.add(a.shape)
                motifs.update(f["motifs"])
                roles.add(f["roles"])
                products.update(f["products"])
                counts.add(f["count"])
                local_counts.add(f["count"])
                for k in observed:
                    observed[k] += bool(f[k])
        signatures.add(digest.digest())
        within += len(local_counts) > 1
        if seed == 0:
            q = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
            # A nontrivial global recoloring checks that color roles are visible.
            permutation = np.array([6, 3, 8, 1, 9, 4, 0, 7, 2, 5])
            for pair in p["train"] + p["test"]:
                assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
            for mode in ("direct", "same_axes", "zip_axes", "recursive", "destination_order"):
                pair_index = next(i for i, x in enumerate(p["train"]) if _features(x["input"])["wrong"][mode])
                bad = {s: [dict(input=x["input"], output=x["output"]) for x in p[s]] for s in p}
                bad["train"][pair_index]["output"] = _render(_parse(p["train"][pair_index]["input"]), mode)
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("accepted wrong " + mode + " labels")
    assert len(signatures) == 42 and len(shapes) >= 35 and len(motifs) >= 70
    assert len(roles) >= 60 and {3, 4, 5, 6} <= counts and within >= 35
    assert {(1, 1), (1, 2), (2, 1), (2, 2)} <= products
    assert all(v >= 3 for v in observed.values()), observed
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "masks": len(motifs), "role_assignments": len(roles),
            "products": sorted(products), "observed": observed}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
