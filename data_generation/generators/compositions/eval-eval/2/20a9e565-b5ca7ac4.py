"""Contact-grown cargo: stable packing supplies the index of a growing motif.

No files, parent imports, latent state, or training labels are used by solve.
Generation uses boundary recurrences; input solving uses unit-step movement.
All experiments are streaming and bounded; no execution traces are saved.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


STYLES = ("comb", "ladder", "hook", "hollow", "solid")
HOST_SEEDS = (0, 131772470, 808102107, 579882797, 215963918, 204144554,
              992006217, 354450843, 343984494, 600477094, 853035317,
              549285911, 634446655, 968427377, 537902831, 948917903,
              860937484, 590916179, 964393473, 462524089, 93934931,
              965549594, 684971145, 332965631, 398756946, 138167995,
              3438851, 959244677, 747622281, 869938764, 368050011,
              116931820, 159564168, 745475527, 135353112, 101767268,
              532541953, 964451441, 901642325, 889376295, 386120329,
              873766904)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad shape"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= a) & (a <= 9)), "bad color"
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _band(a, b):
    return max(a["r"], b["r"]) < min(a["r"] + a["h"], b["r"] + b["h"])


def _overlap(a, b):
    return _band(a, b) and max(a["c"], b["c"]) < min(a["c"] + a["w"], b["c"] + b["w"])


def _corners(hole):
    t, b, l, r = hole
    return {(t, l), (t, l + 1), (t + 1, l),
            (b, r), (b, r - 1), (b - 1, r)}


def _hole(interior, background):
    candidates = []
    for raw in np.unique(interior):
        color = int(raw)
        if color == background:
            continue
        pos = np.argwhere(interior == color)
        if len(pos) != 6:
            continue
        t, l = map(int, pos.min(axis=0))
        b, r = map(int, pos.max(axis=0))
        if b - t < 2 or r - l < 2:
            continue
        bounds = (t, b, l, r)
        if set(map(tuple, pos)) != _corners(bounds):
            continue
        window = interior[t:b + 1, l:r + 1]
        if np.all((window == color) | (window == background)):
            candidates.append((bounds, color))
    assert len(candidates) <= 1, "ambiguous local annotations"
    return candidates[0] if candidates else (None, None)


def _decode(input_grid):
    """Edge rails encode axis and frame roles, independently in every input."""
    a = _grid(input_grid)
    orientations = []
    for transpose in (False, True):
        g = a.T if transpose else a
        if (g.shape[1] >= 7 and np.all(g[:, 0] == g[0, 0])
                and np.all(g[:, -1] == g[0, -1]) and g[0, 0] != g[0, -1]):
            orientations.append((transpose, g.copy()))
    assert len(orientations) == 1, "need two unambiguous opposite rails"
    transpose, canvas = orientations[0]
    frames = (int(canvas[0, 0]), int(canvas[0, -1]))
    scene = canvas[:, 1:-1].copy()
    counts = np.bincount(scene.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert bg not in frames and counts[bg] > max(counts[f] for f in frames)
    assert np.count_nonzero(counts == counts[bg]) == 1, "background tie"
    covered = np.zeros(scene.shape, dtype=bool)
    objects = []
    for r in range(scene.shape[0]):
        for c in range(scene.shape[1]):
            if covered[r, c] or scene[r, c] == bg:
                continue
            frame = int(scene[r, c])
            assert frame in frames, "pixel outside a complete crate"
            assert r + 2 < len(scene) and c + 2 < scene.shape[1]
            bottom, right = r + 2, c + 2
            assert scene[r + 1, c + 1] != frame
            while bottom < len(scene) and scene[bottom, c + 1] != frame:
                bottom += 1
            while right < scene.shape[1] and scene[r + 1, right] != frame:
                right += 1
            assert bottom < len(scene) and right < scene.shape[1], "open crate"
            patch = scene[r:bottom + 1, c:right + 1].copy()
            assert np.all(patch[0] == frame) and np.all(patch[-1] == frame)
            assert np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)
            assert min(patch.shape) >= 5 and not np.any(covered[r:bottom + 1, c:right + 1])
            assert not any(np.any(patch[1:-1, 1:-1] == f) for f in frames)
            hole, marker = _hole(patch[1:-1, 1:-1], bg)
            objects.append(dict(r=r, c=c, h=len(patch), w=patch.shape[1],
                                d=frames.index(frame), patch=patch, hole=hole, marker=marker))
            covered[r:bottom + 1, c:right + 1] = True
    assert objects and np.all(covered | (scene == bg))
    markers = {o["marker"] for o in objects if o["hole"] is not None}
    assert len(markers) == 1, "one annotation color and at least one erased window"
    marker = next(iter(markers))
    assert marker not in (*frames, bg)
    for o in objects:
        if o["hole"] is None:
            assert not np.any(o["patch"] == marker), "stray marker"
    return dict(transpose=transpose, canvas=canvas, scene=scene, bg=bg,
                frames=frames, marker=marker, objects=objects)


def _boundary_pack(objects, width):
    """Generation path: stable destination-side predecessor constraints."""
    result = [dict(o) for o in objects]
    for d in (0, 1):
        group = [o for o in result if o["d"] == d]
        group.sort(key=lambda o: (o["c"] if d == 0 else -o["c"] - o["w"], o["r"]))
        placed = []
        for o in group:
            blockers = [p for p in placed if _band(o, p)]
            o["c"] = (max((p["c"] + p["w"] for p in blockers), default=0) if d == 0
                      else min((p["c"] for p in blockers), default=width) - o["w"])
            contacts = [p for p in blockers if
                        (p["c"] + p["w"] == o["c"] if d == 0 else o["c"] + o["w"] == p["c"])]
            o["depth"] = 1 + max((p["depth"] for p in contacts), default=-1)
            if not 0 <= o["c"] <= width - o["w"]:
                raise ValueError("chain exceeds space")
            placed.append(o)
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:]):
        raise ValueError("final groups collide")
    return result


def _physical_pack(objects, shape):
    """Input-only path: one-cell moves cannot overtake same-color blockers."""
    result = [dict(o) for o in objects]
    for d, delta in ((0, -1), (1, 1)):
        group = [o for o in result if o["d"] == d]
        occupied = np.zeros(shape, dtype=bool)
        for o in group:
            occupied[o["r"]:o["r"] + o["h"], o["c"]:o["c"] + o["w"]] = True
        for _ in range(shape[1] * max(1, len(group)) + 1):
            moved = False
            for o in group:
                r, c, h, w = (o[k] for k in ("r", "c", "h", "w"))
                occupied[r:r + h, c:c + w] = False
                nc = c + delta
                if 0 <= nc <= shape[1] - w and not np.any(occupied[r:r + h, nc:nc + w]):
                    o["c"] = nc
                    moved = True
                occupied[r:r + h, o["c"]:o["c"] + w] = True
            if not moved:
                break
        else:
            raise AssertionError("movement bound exceeded")
        group.sort(key=lambda o: o["c"] if d == 0 else -o["c"] - o["w"])
        done = []
        for o in group:
            contacts = [p for p in done if _band(o, p) and
                        (p["c"] + p["w"] == o["c"] if d == 0 else o["c"] + o["w"] == p["c"])]
            at_edge = o["c"] == (0 if d == 0 else shape[1] - o["w"])
            assert at_edge or contacts, "not fully compacted"
            o["depth"] = 0 if at_edge else 1 + max(p["depth"] for p in contacts)
            done.append(o)
    assert not any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:])
    return result


def _symbol(style, h, w, color, bg, *, predicates=False):
    a = np.full((h, w), bg, dtype=np.int8)
    if predicates:
        r, c = np.indices((h, w))
        if style == "comb":
            mask = (c == 0) | ((h - 1 - r) % 2 == 0)
        elif style == "ladder":
            mask = (c == 0) | (c == w - 1) | ((h - 1 - r) % 2 == 0)
        elif style == "hook":
            mask = (c == 0) | (c == w - 1) | (r == h - 1)
        elif style == "hollow":
            mask = (c == 0) | (c == w - 1) | (r == 0) | (r == h - 1)
        else:
            mask = np.ones((h, w), dtype=bool)
        a[mask] = color
    else:
        if style == "solid":
            a[:] = color
        else:
            a[:, 0] = color
            if style != "comb":
                a[:, -1] = color
            if style in ("comb", "ladder"):
                for row in range(h - 1, -1, -2):
                    a[row] = color
            else:
                a[-1] = color
                if style == "hollow":
                    a[0] = color
    return a


def _interior(o, depth, model, bg, *, predicates=False):
    style, base, delta, glyph_width, ay, ax, palette = model
    h = base + delta * depth
    ih, iw = o["h"] - 2, o["w"] - 2
    assert 3 <= h <= ih and 2 <= glyph_width <= iw, "symbol does not fit crate"
    top = (ih - h) * ay // 2
    left = (iw - glyph_width) * ax // 2
    a = np.full((ih, iw), bg, dtype=np.int8)
    a[top:top + h, left:left + glyph_width] = _symbol(
        style, h, glyph_width, palette[depth % len(palette)], bg, predicates=predicates)
    return a


def _fit_model(objects, settled, bg, marker):
    visible = []
    for o, p in zip(objects, settled):
        if o["hole"] is not None:
            continue
        interior = o["patch"][1:-1, 1:-1]
        cells = np.argwhere(interior != bg)
        assert len(cells) and len(np.unique(interior[interior != bg])) == 1
        t, l = cells.min(axis=0)
        b, r = cells.max(axis=0)
        visible.append((p["depth"], int(b - t + 1), int(r - l + 1),
                        int(interior[tuple(cells[0])]), o))
    depths = {v[0] for v in visible}
    assert {0, 1, 2} <= depths, "need three exposed contact stages"
    heights = {d: h for d, h, _, _, _ in visible}
    base, delta = heights[0], heights[1] - heights[0]
    assert delta != 0 and all(h == base + d * delta for d, h, _, _, _ in visible)
    widths = {v[2] for v in visible}
    assert len(widths) == 1
    glyph_width = next(iter(widths))
    palette = None
    for period in (1, 2):
        mapping = {}
        for d, _, _, col, _ in visible:
            if d % period in mapping and mapping[d % period] != col:
                break
            mapping[d % period] = col
        else:
            if len(mapping) == period:
                palette = tuple(mapping[i] for i in range(period))
                break
    assert palette is not None and marker not in palette
    candidates = []
    for style in STYLES:
        for ay in (0, 1, 2):
            for ax in (0, 1, 2):
                model = (style, base, delta, glyph_width, ay, ax, palette)
                filled = []
                for o, p in zip(objects, settled):
                    try:
                        predicted = _interior(o, p["depth"], model, bg, predicates=True)
                    except AssertionError:
                        break
                    actual = o["patch"][1:-1, 1:-1]
                    known = np.ones(actual.shape, dtype=bool)
                    if o["hole"] is not None:
                        t, b, l, r = o["hole"]
                        known[t:b + 1, l:r + 1] = False
                    if not np.array_equal(actual[known], predicted[known]):
                        break
                    filled.append(predicted)
                if len(filled) == len(objects):
                    candidates.append((model, filled))
    assert candidates, "no consistent contact-growth model"
    expected = candidates[0][1]
    assert all(all(np.array_equal(a, b) for a, b in zip(expected, other))
               for _, other in candidates), "ambiguous erased continuation"
    return candidates[0]


def _analyze(input_grid):
    info = _decode(input_grid)
    objects = info["objects"]
    settled = _physical_pack(objects, info["scene"].shape)
    model, interiors = _fit_model(objects, settled, info["bg"], info["marker"])
    output = info["canvas"].copy()
    output[:, 1:-1] = info["bg"]
    for original, p, interior in zip(objects, settled, interiors):
        patch = original["patch"].copy()
        patch[1:-1, 1:-1] = interior
        r, c = p["r"], p["c"] + 1
        output[r:r + p["h"], c:c + p["w"]] = patch
    info.update(settled=settled, model=model)
    if info["transpose"]:
        output = output.T.copy()
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore local inclusive windows at contact-defined stages, and pack."""
    return _analyze(input_grid)[0]


def _render(shape, bg, frames, objects):
    a = np.full((shape[0], shape[1] + 2), bg, dtype=np.int8)
    a[:, 0], a[:, -1] = frames
    for o in objects:
        r, c = o["r"], o["c"] + 1
        a[r:r + o["h"], c:c + o["w"]] = o["patch"]
    return a


def _geometry(rng, n, purpose, max_height):
    """Create a spine, then vary bands, widths, independent roots and branches."""
    widths = [int(rng.integers(5, 8)) for _ in range(n)]
    if n == 5:
        widths = [5] * n
        if rng.random() < 0.6:
            widths[int(rng.integers(n))] += 1
    total = sum(widths)
    if total > 28:
        return None
    width = int(rng.integers(max(22, total), 29))
    free = width - total
    gaps = rng.multinomial(free, np.full(n + 1, 1 / (n + 1)))
    common_h = max_height + 2 + int(rng.integers(2))
    if purpose == "roots":
        step = 1
    elif purpose == "witness":
        step = max(1, (common_h + 2) // 3)
    else:
        step = int(rng.integers(1, max(2, min(common_h - 1, (29 - common_h) // (n - 1)) + 1)))
    span = common_h + (n - 1) * step
    height = span + (common_h + 1 if purpose == "roots" else 0)
    if height > 30:
        return None
    height += int(rng.integers(min(3, 30 - height) + 1))
    top = int(rng.integers(height - (span + (common_h + 1 if purpose == "roots" else 0)) + 1))
    objects = []
    c = int(gaps[0])
    for i, w in enumerate(widths):
        objects.append(dict(r=top + i * step, c=c, h=common_h, w=w, d=0))
        c += w + int(gaps[i + 1])
    if purpose == "roots":
        row = top + span + 1
        w0, w1 = int(rng.integers(5, 8)), int(rng.integers(5, 8))
        col = int(rng.integers(w1, min(width - w0, objects[-1]["c"] - 1) + 1))
        objects += [dict(r=row, c=col, h=common_h, w=w0, d=0),
                    dict(r=row, c=0, h=common_h, w=w1, d=1)]
    else:
        # Add the opposite class in a visible gap, if feasible. This need not
        # appear in every example: both rails always carry their own meaning.
        for _ in range(24):
            o = dict(r=int(rng.integers(height - common_h + 1)),
                     c=int(rng.integers(width - 4)), h=common_h, w=5, d=1)
            if not any(_overlap(o, p) for p in objects):
                try:
                    _boundary_pack(objects + [o], width)
                except ValueError:
                    continue
                objects.append(o)
                break
    # Optional additions generate shorter branches and genuine mixed sizes.
    for _ in range(int(rng.integers(3))):
        for _ in range(16):
            h = int(rng.integers(5, common_h + 1))
            w = int(rng.integers(5, 8))
            o = dict(r=int(rng.integers(height - h + 1)),
                     c=int(rng.integers(width - w + 1)), h=h, w=w, d=int(rng.integers(2)))
            if any(_overlap(o, p) for p in objects):
                continue
            try:
                packed = _boundary_pack(objects + [o], width)
            except ValueError:
                continue
            if max(p["depth"] for p in packed) <= n - 1:
                objects.append(o)
                break
    if rng.random() < 0.5:
        for o in objects:
            o["r"] = height - o["r"] - o["h"]
    if rng.random() < 0.5:
        for o in objects:
            o["c"] = width - o["c"] - o["w"]
            o["d"] = 1 - o["d"]
    return (height, width), objects


def _fork_geometry(rng, base, delta):
    """A fork rejoins paths of unequal depth at the erased final crate.

    One wide stage-1 branch reaches as far as two narrower serial crates.
    The stage-2 crate is disjoint from the root but overlaps its predecessor,
    so edge-first repacking incorrectly jumps it ahead. The merger touches
    stages 1 and 2: its symbol explicitly teaches 1+max, rather than 1+min.
    """
    widths = [int(rng.integers(5, 7)) for _ in range(4)]
    width = int(rng.integers(sum(widths) + 1, 29))
    gaps = rng.multinomial(width - sum(widths), np.full(5, 0.2))
    cols, c = [], int(gaps[0])
    for i, w in enumerate(widths):
        cols.append(c)
        c += w + int(gaps[i + 1])
    child_h = base + delta + 2
    root_h = max(child_h + 1, base + 2)
    middle_h = base + 2 * delta + 2
    merge_h = base + 3 * delta + 2
    height = max(root_h + middle_h, child_h * 2, child_h - 1 + merge_h)
    assert height <= 30
    height += int(rng.integers(min(3, 30 - height) + 1))
    used_height = max(root_h + middle_h, child_h * 2, child_h - 1 + merge_h)
    top = int(rng.integers(height - used_height + 1))
    objects = [dict(r=top, c=cols[0], h=root_h, w=widths[0], d=0),
               dict(r=top + child_h, c=cols[1], h=child_h, w=widths[1], d=0),
               dict(r=top + root_h, c=cols[2], h=middle_h, w=widths[2], d=0),
               dict(r=top + child_h - 1, c=cols[3], h=merge_h, w=widths[3], d=0),
               dict(r=top, c=cols[1], h=child_h, w=widths[1] + widths[2], d=0)]
    if rng.random() < 0.5:
        for o in objects:
            o["r"] = height - o["r"] - o["h"]
    if rng.random() < 0.5:
        for o in objects:
            o["c"] = width - o["c"] - o["w"]
            o["d"] = 1
    return (height, width), objects


def _make_pair(rng, params, index, test):
    style, delta, glyph_width, ay, ax, period, fixed_axis = params
    n = 5 if test else 4
    purpose = "roots" if not test and index == 0 else "witness" if not test and index == 1 else "free"
    for _ in range(160):
        roles = list(map(int, rng.permutation(10)))
        bg, marker, f0, f1 = roles[:4]
        palette = tuple(roles[4:4 + period])
        base = int(rng.integers(3, 5)) + (n - 1 if delta < 0 else 0)
        max_height = max(base, base + (n - 1) * delta)
        geometry = (_fork_geometry(rng, base, delta) if purpose == "witness"
                    else _geometry(rng, n, purpose, max_height))
        if geometry is None:
            continue
        shape, objects = geometry
        try:
            packed = _boundary_pack(objects, shape[1])
        except ValueError:
            continue
        if {p["depth"] for p in packed[:n]} != set(range(n)):
            continue
        model = (style, base, delta, glyph_width, ay, ax, palette)
        complete = []
        try:
            for o, p in zip(objects, packed):
                patch = np.full((o["h"], o["w"]), (f0, f1)[o["d"]], dtype=np.int8)
                patch[1:-1, 1:-1] = _interior(o, p["depth"], model, bg)
                complete.append(dict(o, patch=patch))
        except AssertionError:
            continue
        # One full-payload erasure teaches extrapolation. Later examples may
        # erase only a suffix or a transverse slice; its bounds stay local.
        masks = [n - 1]
        if index >= 2 and rng.random() < 0.3:
            masks.append(int(rng.integers(1, n - 1)))
        inputs = [dict(o, patch=o["patch"].copy()) for o in complete]
        for i in masks:
            interior = inputs[i]["patch"][1:-1, 1:-1]
            ih, iw = interior.shape
            hole = (0, ih - 1, 0, iw - 1)
            if index >= 2 and rng.random() < 0.5 and ih >= 5:
                hole = (int(rng.integers(ih - 2)), ih - 1, 0, iw - 1)
            t, b, l, r = hole
            if not np.any(interior[t:b + 1, l:r + 1] != bg):
                break
            interior[t:b + 1, l:r + 1] = bg
            for rr, cc in _corners(hole):
                interior[rr, cc] = marker
        else:
            out_objects = [dict(o, c=p["c"]) for o, p in zip(complete, packed)]
            a = _render(shape, bg, (f0, f1), inputs)
            b = _render(shape, bg, (f0, f1), out_objects)
            axis = int(rng.integers(2)) if fixed_axis is None else fixed_axis
            if axis:
                a, b = a.T.copy(), b.T.copy()
            try:
                predicted, info = _analyze(a)
            except AssertionError:
                continue
            if np.array_equal(predicted, b):
                metrics = _metrics(info)
                if purpose == "roots" and metrics["roots"] < 3:
                    continue
                if purpose == "witness" and not (metrics["stable_witness"] and metrics["fork"] and metrics["merge"]):
                    continue
                if not metrics["moving"]:
                    continue
                # A fully hidden leaf must defeat literal copying of the
                # nearest exposed stage, even in the smallest 2/1 puzzle.
                return {"input": a, "output": b}
    raise RuntimeError("bounded construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    style = STYLES[int(rng.integers(len(STYLES)))]
    glyph_width = int(rng.integers(2, 4)) if style == "solid" else 3
    params = (style, int(rng.choice([-1, 1, 1, 2])), glyph_width,
              int(rng.integers(3)), int(rng.integers(3)),
              int(rng.choice([1, 2, 2, 2])),
              None if rng.random() < 0.4 else int(rng.integers(2)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(16):
                pair = _make_pair(rng, params, index, split == "test")
                k = _key(pair["input"])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness retry exhausted")
    return puzzle


def _edge_first(objects, width):
    """Incorrect same-class overtaking rule, retained only as a counterfactual."""
    result = [dict(o) for o in objects]
    for d in (0, 1):
        group = [o for o in result if o["d"] == d]
        group.sort(key=lambda o: o["c"] if d == 0 else -o["c"] - o["w"])
        placed = []
        for o in group:
            columns = range(width - o["w"] + 1)
            if d:
                columns = reversed(columns)
            for c in columns:
                o["c"] = c
                if not any(_overlap(o, p) for p in placed):
                    break
            placed.append(o)
    return result


def _metrics(info):
    objects, packed = info["objects"], info["settled"]
    contacts = []
    for i, a in enumerate(packed):
        predecessors = [j for j, b in enumerate(packed) if i != j and a["d"] == b["d"] and _band(a, b)
                        and (b["c"] + b["w"] == a["c"] if a["d"] == 0 else a["c"] + a["w"] == b["c"])]
        contacts.append(predecessors)
    old = _edge_first(objects, info["scene"].shape[1])
    return dict(axis=int(info["transpose"]), style=info["model"][0], delta=info["model"][2],
                glyph_width=info["model"][3], period=len(info["model"][-1]),
                count=len(objects), max_depth=max(o["depth"] for o in packed),
                masked_depths=[p["depth"] for o, p in zip(objects, packed) if o["hole"] is not None],
                roots=sum(p["depth"] == 0 for p in packed),
                fork=any(sum(j in prev for prev in contacts) > 1 for j in range(len(objects))),
                merge=any(len(prev) > 1 for prev in contacts),
                unequal_merge=any(len({packed[j]["depth"] for j in prev}) > 1 for prev in contacts),
                mixed_sizes=len({(o["h"], o["w"]) for o in objects}) > 1,
                stable_witness=any(o["c"] != p["c"] for o, p in zip(old, packed)),
                moving=sum(o["c"] != p["c"] for o, p in zip(objects, packed)),
                partial=any(o["hole"] is not None and o["hole"] != (0, o["h"] - 3, 0, o["w"] - 3) for o in objects))


def _ablation_witnesses(info):
    """Actual-pixel counterfactuals, plus the incorrect global-ordinal index."""
    objects, packed = info["objects"], info["settled"]
    model, bg = info["model"], info["bg"]
    complete, copied = [], []
    rank = {}
    for d in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["d"] == d]
        indices.sort(key=lambda i: objects[i]["c"] if d == 0 else -objects[i]["c"] - objects[i]["w"])
        rank.update({i: j for j, i in enumerate(indices)})
    ordinal_wrong = False
    for i, (o, p) in enumerate(zip(objects, packed)):
        patch = o["patch"].copy()
        truth = _interior(o, p["depth"], model, bg, predicates=True)
        patch[1:-1, 1:-1] = truth
        complete.append(dict(o, patch=patch))
        wrong = patch.copy()
        if o["hole"] is not None:
            t, b, l, r = o["hole"]
            # Keep the correct target color; copy only the previous geometry.
            # This isolates extrapolation from color cycling and movement.
            previous = _interior(o, max(0, p["depth"] - 1), model, bg, predicates=True)
            previous[previous != bg] = model[-1][p["depth"] % len(model[-1])]
            wrong[1 + t:2 + b, 1 + l:2 + r] = previous[t:b + 1, l:r + 1]
            if rank[i] != p["depth"]:
                try:
                    alternative = _interior(o, rank[i], model, bg, predicates=True)
                except AssertionError:
                    ordinal_wrong = True
                else:
                    ordinal_wrong |= not np.array_equal(truth[t:b + 1, l:r + 1], alternative[t:b + 1, l:r + 1])
        copied.append(dict(o, c=p["c"], patch=wrong))
    correct = _render(info["scene"].shape, bg, info["frames"],
                      [dict(o, c=p["c"]) for o, p in zip(complete, packed)])
    no_motion = _render(info["scene"].shape, bg, info["frames"], complete)
    no_growth = _render(info["scene"].shape, bg, info["frames"], copied)
    return dict(no_growth_cells=int(np.count_nonzero(correct != no_growth)),
                no_packing_cells=int(np.count_nonzero(correct != no_motion)),
                ordinal_wrong=bool(ordinal_wrong))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            predicted, info = _analyze(a)
            assert np.array_equal(predicted, b), "incorrect reconstruction or packing"
            assert not np.any(b == info["marker"]), "annotation survived"
            assert set(map(int, np.unique(b))) <= set(map(int, np.unique(a)))
            recurrence = _boundary_pack(info["objects"], info["scene"].shape[1])
            assert [(p["c"], p["depth"]) for p in recurrence] == [(p["c"], p["depth"]) for p in info["settled"]]
            for o, p in zip(info["objects"], info["settled"]):
                # Check the actual output patch, including frames, unchanged
                # visible pixels, and the inclusive replacement of annotations.
                canonical = b.T if info["transpose"] else b
                actual = canonical[p["r"]:p["r"] + p["h"], p["c"] + 1:p["c"] + p["w"] + 1]
                known = np.ones(o["patch"].shape, dtype=bool)
                if o["hole"] is not None:
                    t, bottom, l, right = o["hole"]
                    known[t + 1:bottom + 2, l + 1:right + 2] = False
                    assert np.any(actual[~known] != info["bg"]), "empty erasure"
                assert np.array_equal(actual[known], o["patch"][known]), "payload changed outside erasure"
            records[split].append(_metrics(info))
            if split == "train" and len(records[split]) == 1:
                witnesses = _ablation_witnesses(info)
                assert witnesses["no_growth_cells"] > 0 and witnesses["no_packing_cells"] > 0
                assert witnesses["ordinal_wrong"], "global order must differ from contact stage"
    assert records["train"][0]["roots"] >= 3, "need independent-band and opposite-frame roots"
    assert records["train"][1]["stable_witness"], "need a same-class no-overtaking demonstration"
    assert records["train"][1]["fork"] and records["train"][1]["merge"], "need fork and merge demonstrations"
    assert records["train"][1]["unequal_merge"], "need max versus min at a join"
    assert all(r["max_depth"] <= 3 for r in records["train"])
    assert all(r["max_depth"] == 4 and 4 in r["masked_depths"] for r in records["test"])
    assert all(r["moving"] > 0 for rs in records.values() for r in rs)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for s in ("train", "test"):
        for p in puzzle[s]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 puzzles streamed; retain feature sets and hashes, never a corpus."""
    fingerprints, shapes, styles, axes, deltas, counts, periods = (set() for _ in range(7))
    totals = dict(forks=0, merges=0, unequal_merges=0, partial=0, mixed_sizes=0, stable_witnesses=0, pairs=0)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for s in ("train", "test"):
            for pair in puzzle[s]:
                _, info = _analyze(pair["input"])
                m = _metrics(info)
                shapes.add(pair["input"].shape)
                styles.add(m["style"])
                axes.add(m["axis"])
                deltas.add(m["delta"])
                counts.add(m["count"])
                periods.add(m["period"])
                for name, feature in (("forks", "fork"), ("merges", "merge"), ("unequal_merges", "unequal_merge"),
                                      ("partial", "partial"), ("mixed_sizes", "mixed_sizes"),
                                      ("stable_witnesses", "stable_witness")):
                    totals[name] += int(m[feature])
                totals["pairs"] += 1
    assert len(fingerprints) == 42 and len(shapes) >= 15
    assert styles == set(STYLES) and axes == {0, 1} and {-1, 1, 2} <= deltas and periods == {1, 2}
    assert len(counts) >= 4 and totals["partial"] > 0 and totals["mixed_sizes"] > 42
    assert totals["forks"] >= 42 and totals["merges"] >= 42 and totals["unequal_merges"] >= 42 and totals["stable_witnesses"] >= 42
    # Illegal types, extremes, and equal counts all reject explicitly.
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Reject a geometrically valid but overtaking-packed answer, using the
    # taught unequal-depth fork rather than only corrupting a rail pixel.
    broken = generate_puzzle(0, 2, 1)
    _, info = _analyze(broken["train"][1]["input"])
    wrong_objects = _edge_first(info["objects"], info["scene"].shape[1])
    for original, placed, true in zip(info["objects"], wrong_objects, info["settled"]):
        placed["patch"] = original["patch"].copy()
        placed["patch"][1:-1, 1:-1] = _interior(original, true["depth"], info["model"], info["bg"])
    wrong = _render(info["scene"].shape, info["bg"], info["frames"], wrong_objects)
    broken["train"][1]["output"] = wrong.T.copy() if info["transpose"] else wrong
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("same-class overtaking accepted")
    return dict(seeds=42, pairs=totals["pairs"], distinct_puzzles=len(fingerprints),
                input_shapes=len(shapes), styles=sorted(styles), axes=sorted(axes),
                deltas=sorted(deltas), object_counts=sorted(counts), periods=sorted(periods), **{k: v for k, v in totals.items() if k != "pairs"})


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
