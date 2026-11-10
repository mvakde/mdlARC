"""A color census controls the conserved opening of a launch gate.

The active tally contains n marks at stride two. Each mark commands one cell
of outward displacement for EACH half, hence an exact channel width of 2*n.
This includes a trailing blank in each two-cell tally unit; the occupied
first-to-last-mark span (2*n-1) is not the command width. The projectile keeps
its visible dimensions and must have transverse width <= 2*n.

Only standard library and NumPy are used. No files, examples, random state, or
construction metadata are consulted by solve. See task_spec.md for conventions.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_SINGLE = ((0, 0),)
_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "Grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid must contain integer ARC colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _rectangle(cells):
    r0, r1, c0, c1 = _bounds(cells)
    return len(cells) == (r1 - r0) * (c1 - c0)


def _normalize(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _panel_candidates(grid):
    """Normalize the visually encoded tally direction to rightward."""
    found = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        for field in map(int, np.unique(a)):
            points = np.argwhere(a == field)
            r0, r1, c0, c1 = _bounds(points.tolist())
            panel = a[r0:r1, c0:c1]
            if min(panel.shape) < 3:
                continue
            if not all(np.all(edge == field) for edge in
                       (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
                continue
            keys = np.argwhere(panel != field)
            if not 2 <= len(keys) <= 3 or len(set(keys[:, 1])) != 1:
                continue
            if not np.all(np.diff(keys[:, 0]) >= 2):
                continue
            colors = tuple(int(panel[r, c]) for r, c in keys)
            if len(set(colors)) != len(colors):
                continue
            inset = int(keys[0, 1])
            if not 0 < 2 * inset < panel.shape[1] - 1:
                continue
            outside = np.ones(a.shape, bool)
            outside[r0:r1, c0:c1] = False
            if not outside.any():
                continue
            frequencies = np.bincount(a[outside], minlength=10)
            bg = int(np.argmax(frequencies))
            if bg == field or bg in colors or frequencies[bg] <= .5 * outside.sum():
                continue
            ring = np.zeros(a.shape, bool)
            ring[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] = True
            ring[r0:r1, c0:c1] = False
            if not np.all(a[ring] == bg):
                continue
            found.append(dict(grid=a, turn=turn, field=field, bg=bg,
                              panel=(r0, r1, c0, c1), keys=colors,
                              positions=tuple((r0 + int(r), c0 + int(c)) for r, c in keys),
                              outside=outside))
    return found


def _parse(input_grid):
    grid = _grid(input_grid)
    panels = _panel_candidates(grid)
    _require(len(panels) == 1, "Expected one unambiguous oriented key panel")
    m = panels[0]
    a, outside = m["grid"], m["outside"]
    objects = []
    for color in sorted(set(map(int, a[outside])) - {m["bg"]}):
        for cells in _components((a == color) & outside):
            objects.append(dict(color=color, cells=cells, bounds=_bounds(cells)))
    candidates = []
    for li, launcher in enumerate(objects):
        lr0, lr1, lc0, lc1 = launcher["bounds"]
        if lc0 != 0 or launcher["color"] in m["keys"]:
            continue
        rows = [sorted(r for r, cc in launcher["cells"] if cc == c) for c in range(lc1)]
        if any(not row or row != list(range(row[0], row[-1] + 1)) for row in rows):
            continue
        widths = tuple(len(row) for row in rows)
        if any(w % 2 for w in widths) or any(x < y for x, y in zip(widths, widths[1:])):
            continue
        center2 = rows[0][0] + rows[0][-1] + 1
        if center2 % 2 or any(row[0] + row[-1] + 1 != center2 for row in rows):
            continue
        center = center2 // 2
        for pi, projectile in enumerate(objects):
            pr0, pr1, pc0, pc1 = projectile["bounds"]
            pw = pr1 - pr0
            if projectile["color"] not in m["keys"] or not _rectangle(projectile["cells"]):
                continue
            if pc0 != lc1 or pr0 + pr1 != center2 or pw % 2 or widths[-1] < pw:
                continue
            if pc1 >= a.shape[1] - (pc1 - pc0):
                continue
            for bi, barrier in enumerate(objects):
                br0, br1, bc0, bc1 = barrier["bounds"]
                if (bi == li or barrier["color"] == launcher["color"] or
                        barrier["color"] in m["keys"] or not _rectangle(barrier["cells"])):
                    continue
                span = br1 - br0
                if bc0 < pc1 or br0 + br1 != center2 or span % 2:
                    continue
                if span <= pw or span <= bc1 - bc0:
                    continue
                glyphs = [o for i, o in enumerate(objects) if i not in (li, pi, bi)]
                # Dedicated launcher/barrier colors cannot also be census glyphs.
                if any(o["color"] in (launcher["color"], barrier["color"]) for o in glyphs):
                    continue
                n = sum(o["color"] == projectile["color"] for o in glyphs)
                if n < 1 or 2 * n < pw or br0 - n < 0 or br1 + n > a.shape[0]:
                    continue
                key_index = m["keys"].index(projectile["color"])
                kr, kc = m["positions"][key_index]
                if kc + 2 * (n - 1) >= m["panel"][3] - 1:
                    continue
                candidate = dict(m, objects=objects, launcher=launcher, projectile=projectile,
                                 barrier=barrier, glyphs=glyphs, count=n, center=center,
                                 widths=widths, active=key_index)
                candidates.append(candidate)
    _require(len(candidates) == 1, "Expected one launcher/projectile/barrier interpretation")
    m = candidates[0]
    # All exterior objects are isolated, except the intended launcher/projectile contact.
    owner = np.full(a.shape, -1, int)
    for i, o in enumerate(objects):
        for r, c in o["cells"]:
            owner[r, c] = i
    permitted = ({id(m["launcher"]), id(m["projectile"])},
                 {id(m["projectile"]), id(m["barrier"])})
    for i, o in enumerate(objects):
        for r, c in o["cells"]:
            for dr, dc in _NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]:
                    j = int(owner[rr, cc])
                    _require(j < 0 or j == i or {id(o), id(objects[j])} in permitted,
                             "Unrelated exterior objects touch")
    # Every other key denotes present glyphs even though only the active row is extended.
    _require(all(any(g["color"] == color for g in m["glyphs"]) for color in m["keys"]),
             "A key has no exterior glyph")
    return m


def _render(m, shift=None, tally=True):
    """Render the rule; shift is an internal counterfactual-testing override."""
    a = m["grid"]
    out = a.copy()
    n = m["count"] if shift is None else shift
    pr0, pr1, pc0, pc1 = m["projectile"]["bounds"]
    br0, br1, bc0, bc1 = m["barrier"]["bounds"]
    center = m["center"]
    bg = m["bg"]
    out[pr0:pr1, pc0:pc1] = bg
    out[br0:br1, bc0:bc1] = bg
    # The entire swept corridor must be free after the barrier is removed.
    _require(np.all(out[pr0:pr1, pc0:] == bg), "Projectile corridor is obstructed")
    targets = ((br0 - n, center - n), (center + n, br1 + n))
    for r0, r1 in targets:
        _require(0 <= r0 < r1 <= a.shape[0] and np.all(out[r0:r1, bc0:bc1] == bg),
                 "Shifted barrier collides or leaves the grid")
        out[r0:r1, bc0:bc1] = m["barrier"]["color"]
    dest = a.shape[1] - (pc1 - pc0)
    _require(np.all(out[pr0:pr1, dest:] == bg), "Projectile does not fit the channel")
    out[pr0:pr1, dest:] = m["projectile"]["color"]
    if tally:
        kr, kc = m["positions"][m["active"]]
        out[kr, kc:kc + 2 * m["count"]:2] = m["projectile"]["color"]
    return np.rot90(out, -m["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role and parameter from this input alone."""
    return _render(_parse(input_grid))


def _shape(rng):
    """Bounded procedural eight-connected glyphs, with no shape/color dictionary."""
    if rng.random() < .28:
        cells = (_DIAMOND, _SQUARE, _SINGLE,
                 ((0, 0), (1, 1), (2, 2)),
                 ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)))[int(rng.integers(5))]
    else:
        occupied = {(1, 1)}
        target = int(rng.integers(1, 10))
        while len(occupied) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in occupied for dr, dc in _NEIGHBORS
                               if 0 <= r + dr < 3 and 0 <= c + dc < 3} - occupied)
            occupied.add(frontier[int(rng.integers(len(frontier)))])
        cells = tuple(occupied)
    return _normalize(cells)


def _layout(rng, count, width, k, endpoint, compact=False):
    """Produce a command bay beside the future barrier sweep, never in its path."""
    capacity = max(count, int(rng.integers(4, 6)))
    max_count = max(capacity, 2)
    inset = 1 if compact else int(rng.integers(1, 3))
    top = 1 if compact else int(rng.integers(1, 3))
    gap = 2 if compact else int(rng.integers(2, 4))
    key_rows = [top + gap * i for i in range(k)]
    ph = key_rows[-1] + 2
    pw = inset + 2 * max_count + 1 + (0 if compact else int(rng.integers(3)))
    w = 30 if compact else int(rng.integers(25, 31))
    bay = 12 if compact else int(rng.integers(max(11, ph + 1), 14))
    max_half = min(4, (30 - bay - 1 - 2 * capacity) // 2)
    if max_half < width // 2 + 1:
        return None
    half = int(rng.integers(width // 2 + 1, max_half + 1))
    margin_budget = 30 - bay - 1 - 2 * half - 2 * capacity
    low = int(rng.integers(min(2, margin_budget) + 1))
    high = int(rng.integers(min(2, margin_budget - low) + 1))
    h = bay + 1 + 2 * half + 2 * capacity + low + high
    center = bay + 1 + low + half + capacity
    panel_r = 0 if compact else int(rng.integers(bay - ph + 1))
    panel_c = 0 if compact else int(rng.integers(w - pw + 1))
    blocked = np.zeros((bay, w), bool)
    blocked[max(0, panel_r - 1):panel_r + ph + 1,
            max(0, panel_c - 1):panel_c + pw + 1] = True
    slots = [(r, c) for r in range(1, bay - 2, 4) for c in range(1, w - 2, 4)
             if not blocked[r:r + 3, c:c + 3].any()]
    rng.shuffle(slots)
    depth = int(rng.integers(1, 6))
    levels = sorted(map(int, rng.integers(width // 2, half + 1, size=depth)), reverse=True)
    profile = tuple(2 * x for x in levels)
    plen = int(rng.integers(1, 5))
    thick = int(rng.integers(1, min(4, 2 * half)))
    if endpoint == "edge":
        bc1 = w
    elif endpoint == "overlap" and plen > 1:
        bc1 = w - int(rng.integers(1, plen))
    else:
        bc1 = depth + plen + int(rng.integers(0, 8)) + thick
    if bc1 - thick < depth + plen or w - plen <= depth + plen:
        return None
    return dict(shape=(h, w), panel=(panel_r, panel_r + ph, panel_c, panel_c + pw),
                positions=tuple((panel_r + r, panel_c + inset) for r in key_rows),
                slots=slots, center=center, profile=profile,
                projectile=(center - width // 2, center + width // 2, depth, depth + plen),
                barrier=(center - half, center + half, bc1 - thick, bc1), bay=bay)


def _build_group(rng, active_counts, turn, *, twin=False, endpoint=None):
    """Twins share all geometry; one extra exterior glyph is the only input edit."""
    k = int(rng.integers(2, 4))
    palette = list(map(int, rng.permutation(10)))
    bg, field, launcher, barrier = palette[:4]
    keys = tuple(palette[4:4 + k])
    flip = bool(rng.integers(2))
    active = (int(rng.integers(k - 1)) if flip else int(rng.integers(1, k))) if twin else int(rng.integers(k))
    pcolor = keys[active]
    other_counts = {c: int(rng.integers(1, 3)) for c in keys if c != pcolor}
    ignored = palette[4 + k:]
    nd = int(rng.integers(0, 3))
    ignored_counts = {c: 1 for c in ignored[:nd]}
    if twin:
        # A present inactive category and an unkeyed distractor refute counting everything.
        other_counts[next(c for c in keys if c != pcolor)] = 2
        ignored_counts = {ignored[0]: 2}
    max_n = max(active_counts)
    width = 2 if twin else int(rng.choice([w for w in (2, 4, 6) if w <= 2 * max_n]))
    endpoint = endpoint or str(rng.choice(("clear", "edge", "overlap")))
    total = max_n + sum(other_counts.values()) + sum(ignored_counts.values())
    # Bounded layout attempts; fallback retains every requested count and witness.
    layout = None
    for attempt in range(13):
        candidate = _layout(rng, max_n, width, k, endpoint, compact=attempt == 12)
        if candidate is not None and len(candidate["slots"]) >= total:
            layout = candidate
            break
    if layout is None:
        # A width-6 projectile can need a lower command bay to fit. Width is a sampled
        # geometry choice, not a label; the fixed width-2 fallback preserves the rule.
        width = 2
        layout = _layout(rng, max_n, width, k, endpoint, compact=True)
    _require(layout is not None and len(layout["slots"]) >= total, "Bounded packing failed")
    h, w = layout["shape"]
    base = np.full((h, w), bg, np.int8)
    r0, r1, c0, c1 = layout["panel"]
    base[r0:r1, c0:c1] = field
    for pos, color in zip(layout["positions"], keys):
        base[pos] = color
    center = layout["center"]
    for c, span in enumerate(layout["profile"]):
        base[center - span // 2:center + span // 2, c] = launcher
    pr0, pr1, pc0, pc1 = layout["projectile"]
    br0, br1, bc0, bc1 = layout["barrier"]
    base[pr0:pr1, pc0:pc1] = pcolor
    base[br0:br1, bc0:bc1] = barrier
    specs = [(pcolor, i) for i in range(max_n)]
    specs += [(color, i) for color, n in other_counts.items() for i in range(n)]
    specs += [(color, i) for color, n in ignored_counts.items() for i in range(n)]
    glyphs = []
    for (color, i), (r, c) in zip(specs, layout["slots"]):
        shape = (_DIAMOND, _SQUARE, _SINGLE)[i] if twin and color == pcolor and i < 3 else _shape(rng)
        # Jitter small glyphs inside their separated 3x3 slots.
        rh = 1 + max(rr for rr, cc in shape)
        cw = 1 + max(cc for rr, cc in shape)
        r += int(rng.integers(4 - rh))
        c += int(rng.integers(4 - cw))
        cells = tuple((r + dr, c + dc) for dr, dc in shape)
        glyphs.append((color, i, cells))
    pairs = []
    for n in active_counts:
        inp = base.copy()
        out = base.copy()
        for color, i, cells in glyphs:
            if color == pcolor and i >= n:
                continue
            for r, c in cells:
                inp[r, c] = out[r, c] = color
        # Construction labels use sampled roles/counts, separately from solve's parser.
        out[pr0:pr1, pc0:pc1] = bg
        out[br0:br1, bc0:bc1] = bg
        out[br0 - n:center - n, bc0:bc1] = barrier
        out[center + n:br1 + n, bc0:bc1] = barrier
        out[pr0:pr1, w - (pc1 - pc0):] = pcolor
        kr, kc = layout["positions"][active]
        out[kr, kc:kc + 2 * n:2] = pcolor
        if flip:
            inp, out = np.flipud(inp), np.flipud(out)
        pairs.append({"input": np.rot90(inp, turn).copy(),
                      "output": np.rot90(out, turn).copy()})
    return pairs


def _validate_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if not lo <= value <= hi:
            raise ValueError(name + " is outside the legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _validate_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    turn = int(rng.integers(4))
    regime = int(rng.integers(3))  # count extrapolation, quarter turn, recombination
    puzzle = {"train": _build_group(rng, (3, 4), turn, twin=True), "test": []}
    seen = {_key(pair["input"]) for pair in puzzle["train"]}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for i in range(count):
            n = 5 if split == "test" and regime == 0 else int(rng.integers(1, 5))
            if regime == 1:
                rotation = (turn + (2 * (i % 2) if split == "train" else 1 + 2 * (i % 2))) % 4
            else:
                rotation = (turn + i + (1 if split == "train" else 2)) % 4
            for retry in range(12):
                pair = _build_group(rng, (n,), rotation)[0]
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Could not generate a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "Output must preserve the full canvas")
    m = _parse(inp)
    expected = _render(m)
    _require(np.array_equal(out, expected), "Wrong output")
    b = np.rot90(out, m["turn"])
    a = m["grid"]
    br0, br1, bc0, bc1 = m["barrier"]["bounds"]
    pr0, pr1, pc0, pc1 = m["projectile"]["bounds"]
    n, center = m["count"], m["center"]
    # Check masks and conservation without the output painter.
    barrier_cells = set(map(tuple, np.argwhere(b == m["barrier"]["color"]).tolist()))
    intended_barrier = {(r - n if r < center else r + n, c)
                        for r, c in m["barrier"]["cells"]}
    _require(barrier_cells == intended_barrier, "Barrier must retain both translated halves")
    _require(len(barrier_cells) == len(m["barrier"]["cells"]), "Barrier area changed")
    _require(all(b[r, c] == m["launcher"]["color"] for r, c in m["launcher"]["cells"]),
             "Launcher changed")
    for g in m["glyphs"]:
        _require(all(b[r, c] == g["color"] for r, c in g["cells"]), "Census glyph changed")
    _require(np.all(b[pr0:pr1, pc0:pc1] == m["bg"]), "Projectile trail remains")
    # Exact projectile mask: exterior glyphs and the panel share its color, so
    # counting that color globally would conflate three different roles.
    destination = a.shape[1] - (pc1 - pc0)
    expected_projectile_color = a == m["projectile"]["color"]
    expected_projectile_color[pr0:pr1, pc0:pc1] = False
    expected_projectile_color[pr0:pr1, destination:] = True
    kr, kc = m["positions"][m["active"]]
    expected_projectile_color[kr, kc:kc + 2 * n:2] = True
    _require(np.array_equal(b == m["projectile"]["color"], expected_projectile_color),
             "Projectile, glyph, or tally footprint changed incorrectly")
    # In the barrier slab there are exactly 2*n rows of clearance transversely.
    _require(np.all(b[center - n:center + n, bc0:bc1] != m["barrier"]["color"]),
             "Wrong channel width")
    before_counts = np.bincount(a.ravel(), minlength=10)
    after_counts = np.bincount(b.ravel(), minlength=10)
    delta = after_counts - before_counts
    intended_delta = np.zeros(10, int)
    intended_delta[m["projectile"]["color"]] = n - 1
    intended_delta[m["field"]] = -(n - 1)
    _require(np.array_equal(delta, intended_delta), "Non-tally material is not conserved")
    _require(not np.array_equal(inp, out), "Identity pair")
    return m


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    _validate_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for p in puzzle[split]:
            m = _check_pair(p)
            key = _key(p["input"])
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            if split == "train":
                training.append(m)
    _require({3, 4} <= {m["count"] for m in training}, "Missing count-control demonstrations")
    _require(any(m["count"] > (m["projectile"]["bounds"][1] - m["projectile"]["bounds"][0]) // 2
                 for m in training), "Projectile-only shortcut is not distinguished")
    _require(any(m["active"] != 0 for m in training), "First-key shortcut is not distinguished")
    witnesses = [{_normalize(g["cells"]) for g in m["glyphs"]
                  if g["color"] == m["projectile"]["color"]} for m in training]
    _require(any({_DIAMOND, _SQUARE, _SINGLE} <= w for w in witnesses),
             "Missing connectivity and unequal-area witnesses")
    # The mandatory pair differs only by one additional same-color glyph.
    twins = False
    for x in training:
        for y in training:
            if x["count"] != 3 or y["count"] != 4 or x["turn"] != y["turn"]:
                continue
            if x["grid"].shape != y["grid"].shape:
                continue
            diff = x["grid"] != y["grid"]
            if diff.any() and np.all(x["grid"][diff] == x["bg"]) and np.all(
                    y["grid"][diff] == x["projectile"]["color"]):
                twins = True
    _require(twins, "Missing matched geometry/count intervention")


def self_test():
    """Stream 42 seeds; retain signatures/features, never whole generated corpora."""
    signatures, shapes, palettes, directions, counts, widths = set(), set(), set(), set(), set(), set()
    vocabulary, geometries, edge_states, profile_types = set(), set(), set(), set()
    samples = 0
    ablation_witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed == 0:
            _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "Nondeterministic")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _parse(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "Solver disagrees")
                shapes.add(tuple(pair["input"].shape))
                directions.add(m["turn"])
                counts.add(m["count"])
                widths.add(m["projectile"]["bounds"][1] - m["projectile"]["bounds"][0])
                palettes.add((m["bg"], m["field"], m["projectile"]["color"]))
                vocabulary.update(_normalize(g["cells"]) for g in m["glyphs"])
                geometries.add((tuple(m["grid"].shape), m["panel"], m["barrier"]["bounds"], m["widths"]))
                edge_states.add(m["barrier"]["bounds"][3] == m["grid"].shape[1])
                profile_types.add(len(set(m["widths"])))
                samples += 1
                if split == "train" and m["count"] in (3, 4):
                    pw = m["projectile"]["bounds"][1] - m["projectile"]["bounds"][0]
                    if pw == 2:
                        # Parent B alone would open only a projectile-width gap.
                        _require(not np.array_equal(_render(m, shift=pw // 2), pair["output"]),
                                 "Count ablation did not change the answer")
                        # A stationary barrier obstructs the swept projectile.
                        pr0, pr1, _, _ = m["projectile"]["bounds"]
                        _, _, bc0, bc1 = m["barrier"]["bounds"]
                        _require(np.all(m["grid"][pr0:pr1, bc0:bc1] == m["barrier"]["color"]),
                                 "The barrier does not actually block passage")
                        ablation_witnesses += 1
                # Rotation/reflection and literal-color renaming are true family symmetries.
                if split == "test":
                    inp, out = pair["input"], pair["output"]
                    for turn in range(4):
                        _require(np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn)),
                                 "Rotation equivariance failed")
                    _require(np.array_equal(solve(np.flipud(inp)), np.flipud(out)), "Reflection failed")
                    permutation = np.roll(np.arange(10), 3)
                    _require(np.array_equal(solve(permutation[inp]), permutation[out]), "Palette failed")
    _require(len(signatures) == 42 and samples == 210, "Insufficient distinct puzzles")
    _require(ablation_witnesses >= 84, "Missing parent-necessity witnesses")
    _require(len(shapes) >= 30 and len(geometries) >= 100 and len(palettes) >= 80, "Limited geometry/palette")
    _require(directions == set(range(4)) and counts == set(range(1, 6)), "Missing count/direction coverage")
    _require({2, 4} <= widths and len(vocabulary) >= 35, "Limited projectile/glyph diversity")
    _require(edge_states == {True, False} and {1, 2} <= profile_types, "Missing endpoint/profile variation")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "Wrong explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # Explicit mutation evidence: punching out a channel without translating
    # conserved halves must be rejected, even when the projectile is delivered.
    punched = generate_puzzle(7)
    pair = next(p for p in punched["train"] if _parse(p["input"])["count"] == 3)
    m = _parse(pair["input"])
    b = np.rot90(pair["output"], m["turn"]).copy()
    b[b == m["barrier"]["color"]] = m["bg"]
    br0, br1, bc0, bc1 = m["barrier"]["bounds"]
    for r in range(br0, br1):
        if not m["center"] - m["count"] <= r < m["center"] + m["count"]:
            b[r, bc0:bc1] = m["barrier"]["color"]
    pair["output"] = np.rot90(b, -m["turn"]).copy()
    try:
        validate_generated_puzzle(punched)
    except ValueError:
        pass
    else:
        raise AssertionError("Erased barrier material accepted")
    # A separate single-cell corruption must also be detected.
    bad = generate_puzzle(7)
    cell = tuple(np.argwhere(bad["test"][0]["output"] != bad["test"][0]["input"])[0])
    bad["test"][0]["output"][cell] = bad["test"][0]["input"][cell]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    return {"seeds": 42, "pairs": samples, "distinct_puzzles": len(signatures),
            "grid_shapes": len(shapes), "geometries": len(geometries),
            "palettes": len(palettes), "glyph_shapes": len(vocabulary),
            "counts": sorted(counts), "projectile_widths": sorted(widths),
            "directions": sorted(directions), "parent_ablation_witnesses": ablation_witnesses,
            "legal_count_combinations": 30, "punching_mutation_rejected": True}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
