"""Cross-registered assemblies. Standard library and NumPy; no saved state.

No generated artifacts or execution traces are written by this module.
"""
from __future__ import annotations

import hashlib
import itertools
import numbers

import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer grid with sides 1..30 and colors 0..9")
    return a


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _centers(mask):
    found = np.zeros(mask.shape, dtype=bool)
    if min(mask.shape) >= 3:
        found[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                                    & mask[2:, 1:-1] & mask[1:-1, :-2]
                                    & mask[1:-1, 2:])
    return tuple(tuple(map(int, p)) for p in np.argwhere(found))


def _registration(mask):
    points = _centers(mask)
    if len(points) != 3:
        raise ValueError("the glyph must contain exactly three complete plus centers")
    rows, cols = sorted({r for r, c in points}), sorted({c for r, c in points})
    if len(rows) != 2 or len(cols) != 2:
        raise ValueError("the three plus centers must form an axis-aligned L")
    dy, dx = rows[1] - rows[0], cols[1] - cols[0]
    if min(dy, dx) < 2:
        raise ValueError("registration steps must be at least two cells")
    return dy, dx, points


def _destructive_centers(mask):
    remaining = mask.copy()
    result = []
    for r in range(1, mask.shape[0] - 1):
        for c in range(1, mask.shape[1] - 1):
            cells = [(r + dr, c + dc) for dr, dc in ((0, 0),) + _DIRS]
            if all(remaining[p] for p in cells):
                result.append((r, c))
                for p in cells:
                    remaining[p] = False
    return tuple(result)


def _components(mask):
    todo = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        seen, stack = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    seen.append(p)
                    stack.append(p)
        result.append(seen)
    return result


def _gap(a, b):
    """Half-open boxes separated by at least one complete blank row/column."""
    r, c, h, w = a
    s, t, j, k = b
    return r + h < s or s + j < r or c + w < t or t + k < c


def _decode(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    if len(values) != 3 or sorted(map(int, counts))[-1] == sorted(map(int, counts))[-2]:
        raise ValueError("three visible roles and a uniquely dominant background required")
    bg = int(values[int(np.argmax(counts))])
    candidates = []
    for color in map(int, values):
        if color == bg:
            continue
        parts = _components(a == color)
        large = [p for p in parts if len(p) > 1]
        small = [p for p in parts if len(p) == 1]
        if len(large) == 1 and len(small) == 1:
            candidates.append((color, large[0], small[0][0]))
    if len(candidates) != 1:
        raise ValueError("one connected glyph and one matching isolated anchor required")
    ink, cells, anchor_cell = candidates[0]
    marker = next(int(v) for v in values if v not in (bg, ink))
    marker_parts = _components(a == marker)
    if not marker_parts or any(len(p) != 1 for p in marker_parts):
        raise ValueError("ordinary map points must be isolated pixels")
    tr, tc = np.min(cells, axis=0).astype(int)
    br, bc = np.max(cells, axis=0).astype(int)
    tr, tc, br, bc = map(int, (tr, tc, br, bc))
    mask = (a[tr:br + 1, tc:bc + 1] == ink)
    dy, dx, crosses = _registration(mask)
    points = sorted([anchor_cell] + [p[0] for p in marker_parts])
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    if any((r - pr) % 2 or (c - pc) % 2 for r, c in points):
        raise ValueError("map points must share a pitch-two lattice")
    if not _gap((tr, tc, *mask.shape), (pr, pc, er - pr + 1, ec - pc + 1)):
        raise ValueError("glyph and map boxes must be separated")
    layout = tuple(((r - pr) // 2, (c - pc) // 2) for r, c in points)
    anchor = ((anchor_cell[0] - pr) // 2, (anchor_cell[1] - pc) // 2)
    mh, mw = (er - pr) // 2 + 1, (ec - pc) // 2 + 1
    h, w = mask.shape
    top, left = tr - anchor[0] * dy, tc - anchor[1] * dx
    if not (0 <= top and top + (mh - 1) * dy + h <= a.shape[0]
            and 0 <= left and left + (mw - 1) * dx + w <= a.shape[1]):
        raise ValueError("the complete registered assembly must fit without clipping")
    return dict(bg=bg, ink=ink, marker=marker, mask=mask, crosses=crosses,
                dy=dy, dx=dx, tr=tr, tc=tc, top=top, left=left,
                layout=layout, anchor=anchor, map_shape=(mh, mw), points=points)


def _render(shape, info, *, bbox_steps=False, use_anchor=True):
    """Mask holes are transparent; the registered glyph has final priority."""
    out = np.full(shape, info["bg"], dtype=np.int8)
    mask = info["mask"]
    dy, dx = mask.shape if bbox_steps else (info["dy"], info["dx"])
    ar, ac = info["anchor"] if use_anchor else (0, 0)
    mr, mc = np.nonzero(mask)
    for r, c in info["layout"]:
        rr, cc = info["tr"] + (r - ar) * dy + mr, info["tc"] + (c - ac) * dx + mc
        # Bounds are guaranteed for the intended rule. Clipping is used only
        # to quantify deliberately wrong ablation rules in validation.
        valid = (0 <= rr) & (rr < shape[0]) & (0 <= cc) & (cc < shape[1])
        out[rr[valid], cc[valid]] = info["ink"]
    out[info["tr"] + mr, info["tc"] + mc] = info["marker"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive palette roles, local registration, and every copy from input only."""
    a = _grid(input_grid)
    return _render(a.shape, _decode(a))


def _glyph(rng, dy, dx, avoid_padding=None):
    """Generate a connected arbitrary mask while preserving exactly three sockets."""
    # Small margins can become copied appendages; they are cropped if unused.
    options = list(itertools.product(range(2), repeat=2))
    row_options = [p for p in options if avoid_padding is None or sum(p) + 3 != avoid_padding[0]]
    col_options = [p for p in options if avoid_padding is None or sum(p) + 3 != avoid_padding[1]]
    vertical, horizontal = _pick(rng, row_options), _pick(rng, col_options)
    margins = [vertical[0], vertical[1], horizontal[0], horizontal[1]]
    h, w = dy + 3 + margins[0] + margins[1], dx + 3 + margins[2] + margins[3]
    r, c = 1 + margins[0], 1 + margins[2]
    mask = np.zeros((h, w), dtype=bool)
    for cr, cc in ((r, c), (r + dy, c), (r, c + dx)):
        for dr, dc in ((0, 0),) + _DIRS:
            mask[cr + dr, cc + dc] = True
    mask[r:r + dy + 1, c] = True
    mask[r, c:c + dx + 1] = True
    # Extend the outside arms to make every selected margin observable in the
    # tight box. These thin extensions introduce no additional plus centers.
    mask[:r, c] = True
    mask[r + dy + 1:, c] = True
    mask[r, :c] = True
    mask[r, c + dx + 1:] = True
    desired = set(_centers(mask))
    assert len(desired) == 3
    target = max(int(mask.sum()) + 3, int(mask.size * float(rng.uniform(.40, .78))))
    # Monotone growth means a rejected fourth-center cell can never become
    # admissible later. Each pixel is considered at most once.
    forbidden = set()
    for _ in range(mask.size):
        if mask.sum() >= target:
            break
        frontier = sorted({(int(a + dr), int(b + dc))
                           for a, b in np.argwhere(mask) for dr, dc in _DIRS
                           if 0 <= a + dr < h and 0 <= b + dc < w
                           and not mask[a + dr, b + dc]
                           and (int(a + dr), int(b + dc)) not in forbidden})
        if not frontier:
            break
        point = _pick(rng, frontier)
        mask[point] = True
        if set(_centers(mask)) != desired:
            mask[point] = False
            forbidden.add(point)
    # Reflections change the L's elbow and the copied asymmetric detail.
    if rng.integers(2):
        mask = np.flipud(mask)
    if rng.integers(2):
        mask = np.fliplr(mask)
    rr, cc = np.nonzero(mask)
    mask = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()
    assert len(_components(mask)) == 1
    assert _registration(mask)[:2] == (dy, dx)
    return mask


def _map(rng, dims, *, witness=False):
    h, w = dims
    # Vary occupancy independently of physical extent. Keep tight bounds.
    occupied = {(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)}
    if witness:
        # Both coordinate steps are explicit; the middle empty band remains.
        occupied.update(((0, 0), (1, 0), (0, 1), (h - 1, w - 1)))
        if h >= 4:
            occupied = {p for p in occupied if p[0] != 2}
        if w >= 4:
            occupied = {p for p in occupied if p[1] != 2}
    else:
        adjacent = [((r, c), (r + dr, c + dc)) for r in range(h) for c in range(w)
                    for dr, dc in ((1, 0), (0, 1)) if r + dr < h and c + dc < w]
        occupied.update(_pick(rng, adjacent))
    available = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied
                 and not (witness and ((h >= 4 and r == 2) or (w >= 4 and c == 2)))]
    rng.shuffle(available)
    target = int(rng.integers(max(3, len(occupied)), max(3, h * w) + 1))
    occupied.update(available[:max(0, target - len(occupied))])
    return tuple(sorted(occupied))


def _place(rng, mask, dy, dx, layout, anchor, colors):
    h, w = mask.shape
    mh, mw = max(r for r, c in layout) + 1, max(c for r, c in layout) + 1
    ah, aw = h + (mh - 1) * dy, w + (mw - 1) * dx
    ph, pw = 2 * mh - 1, 2 * mw - 1
    tr, tc = anchor[0] * dy, anchor[1] * dx
    # Enumerate a bounded set of positions relative to assembly top-left.
    choices = []
    for pr in range(-ph - 1, ah + 2):
        bh = max(ah, pr + ph) - min(0, pr)
        if bh > 30:
            continue
        for pc in range(-pw - 1, aw + 2):
            bw = max(aw, pc + pw) - min(0, pc)
            if bw <= 30 and _gap((tr, tc, h, w), (pr, pc, ph, pw)):
                choices.append((pr, pc, bh, bw))
    if not choices:
        raise ValueError("no map placement fits")
    if rng.random() < .60:
        minimum = min(v[2] * v[3] for v in choices)
        choices = [v for v in choices if v[2] * v[3] <= minimum * 1.25]
    pr, pc, bh, bw = _pick(rng, choices)
    gh, gw = min(30, bh + int(rng.integers(5))), min(30, bw + int(rng.integers(5)))
    if rng.random() < .10:
        gh = gw = 30
    # Ensure a uniquely dominant background even for unusually dense small inputs.
    ink_count, mark_count = int(mask.sum()) + 1, len(layout) - 1
    need = ink_count + mark_count + max(ink_count, mark_count) + 1
    while gh * gw < need:
        if gh <= gw and gh < 30:
            gh += 1
        elif gw < 30:
            gw += 1
        else:
            raise AssertionError("background bound exceeded")
    top = -min(0, pr) + int(rng.integers(gh - bh + 1))
    left = -min(0, pc) + int(rng.integers(gw - bw + 1))
    bg, ink, marker = colors
    a = np.full((gh, gw), bg, dtype=np.int8)
    mr, mc = np.nonzero(mask)
    a[top + tr + mr, left + tc + mc] = ink
    for r, c in layout:
        a[top + pr + 2 * r, left + pc + 2 * c] = ink if (r, c) == anchor else marker
    # Generation uses coordinate sets; solve uses indexed array stamps.
    ink_pixels = set()
    for r, c in layout:
        ink_pixels.update((top + r * dy + int(i), left + c * dx + int(j))
                          for i, j in zip(mr, mc))
    out = np.full_like(a, bg)
    for p in ink_pixels:
        out[p] = ink
    out[top + tr + mr, left + tc + mc] = marker
    return {"input": a, "output": out}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _fresh(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    # At most 13 earlier inputs; 720 bijective three-role palettes guarantee
    # an unused input. This finite fallback preserves every geometric rule.
    old = list(map(int, np.unique(pair["input"])))
    for colors in itertools.permutations(range(10), 3):
        lut = np.arange(10, dtype=np.int8)
        lut[old] = colors
        candidate = {k: lut[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise AssertionError("palette uniqueness bound exceeded")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Every puzzle teaches a gapped map before applying a longer extent on a
    # selected axis. This requires no hidden interpretation parameter.
    axis = int(rng.integers(2))
    stable = rng.random() < .20
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, used = {"train": [], "test": []}, set()
    first_anchor = first_steps = first_padding = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            teaching = split == "train" and index < 2
            if teaching and index == 0:
                # Overlap of the local five-cell motifs is visible in training.
                steps = (2, int(rng.integers(3, 6)))
                if rng.integers(2):
                    steps = steps[::-1]
            elif teaching:
                steps = _pick(rng, [(y, x) for y in range(2, 7) for x in range(2, 7)
                                    if y != first_steps[0] and x != first_steps[1]])
            else:
                steps = tuple(map(int, rng.integers(2, 7, size=2)))
                if split == "test" and steps[axis] == 6:
                    steps = tuple(int(rng.integers(2, 6)) if i == axis else v
                                  for i, v in enumerate(steps))
            dy, dx = steps
            mask = _glyph(rng, dy, dx, avoid_padding=first_padding if teaching and index == 1 else None)
            if teaching and index == 0:
                dims = _pick(rng, ((2, 3), (3, 2), (3, 3)))
            elif teaching:
                # Explicit empty lattice band plus both direct pitch witnesses.
                dims = (4, 4)
            elif split == "test":
                limits = ((30 - mask.shape[0]) // dy + 1, (30 - mask.shape[1]) // dx + 1)
                long = int(rng.integers(5, min(7, limits[axis]) + 1))
                short = int(rng.integers(1, 4))
                dims = (long, short) if axis == 0 else (short, long)
            else:
                limits = (min(4, (30 - mask.shape[0]) // dy + 1),
                          min(4, (30 - mask.shape[1]) // dx + 1))
                dims = tuple(int(rng.integers(1, n + 1)) for n in limits)
                if dims == (1, 1):
                    dims = (2, 1)
            layout = _map(rng, dims, witness=teaching and index == 1)
            anchors = list(layout)
            if teaching and index == 0:
                def candidates(points):
                    glyph_cells = set(zip(*np.nonzero(mask)))
                    h, w = mask.shape
                    result = []
                    for ar, ac in points:
                        if (ar, ac) == ((dims[0] - 1) // 2, (dims[1] - 1) // 2):
                            continue
                        if not any((ar + dr, ac + dc) in points for dr, dc in _DIRS):
                            continue
                        # The registered copy must visibly leave other ink in
                        # its transparent holes, even with only two examples.
                        filled = any(0 <= i + (r - ar) * dy < h and 0 <= j + (c - ac) * dx < w
                                     and (i + (r - ar) * dy, j + (c - ac) * dx) not in glyph_cells
                                     for r, c in points for i, j in glyph_cells)
                        if filled:
                            result.append((ar, ac))
                    return result
                noncentral = candidates(layout)
                if not noncentral:
                    layout = tuple(itertools.product(range(dims[0]), range(dims[1])))
                    noncentral = candidates(layout)
                assert noncentral
                anchor = _pick(rng, noncentral)
            elif teaching:
                changed = [p for p in anchors if p[0] != first_anchor[0] and p[1] != first_anchor[1]]
                anchor = _pick(rng, changed)
            else:
                anchor = _pick(rng, anchors)
            colors = palette if stable else tuple(map(int, rng.choice(10, 3, replace=False)))
            pair = _fresh(_place(rng, mask, dy, dx, layout, anchor, colors), used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if teaching and index == 0:
                first_anchor, first_steps = anchor, steps
                first_padding = (mask.shape[0] - dy, mask.shape[1] - dx)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_pair(pair):
    a, b = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == b.shape
    m = _decode(a)
    mask, dy, dx = m["mask"], m["dy"], m["dx"]
    # A direct neighborhood scan checks local recognition independently of
    # the vectorized center detector used to decode and generate glyphs.
    direct = tuple((r, c) for r in range(1, mask.shape[0] - 1)
                   for c in range(1, mask.shape[1] - 1)
                   if all(mask[r + dr, c + dc] for dr, dc in ((0, 0),) + _DIRS))
    assert direct == m["crosses"]
    # Invert every output coordinate into every candidate copy. This checks
    # mask transparency and anchor priority without using the stamp renderer.
    rr, cc = np.indices(a.shape)
    union = np.zeros(a.shape, dtype=bool)
    coverage = np.zeros(a.shape, dtype=np.uint8)
    ar, ac = m["anchor"]
    for r, c in m["layout"]:
        i, j = rr - m["tr"] - (r - ar) * dy, cc - m["tc"] - (c - ac) * dx
        inside = (i >= 0) & (i < mask.shape[0]) & (j >= 0) & (j < mask.shape[1])
        present = inside & mask[np.clip(i, 0, mask.shape[0] - 1), np.clip(j, 0, mask.shape[1] - 1)]
        union |= present
        coverage += present
    expected = np.full(a.shape, m["bg"], dtype=np.int8)
    expected[union] = m["ink"]
    i, j = rr - m["tr"], cc - m["tc"]
    inside = (i >= 0) & (i < mask.shape[0]) & (j >= 0) & (j < mask.shape[1])
    anchor_mask = inside & mask[np.clip(i, 0, mask.shape[0] - 1), np.clip(j, 0, mask.shape[1] - 1)]
    expected[anchor_mask] = m["marker"]
    assert np.array_equal(b, expected), "output is not the registered mask union"
    assert np.array_equal(solve(a), b)
    assert np.count_nonzero(b == m["marker"]) == mask.sum()
    assert np.count_nonzero(b != m["bg"]) == np.count_nonzero(union)
    assert np.count_nonzero(union) > mask.sum(), "the map must add material beyond the source glyph"
    assert not np.array_equal(a, b)
    mh, mw = m["map_shape"]
    layout = set(m["layout"])
    m.update(shape=a.shape, overlap=int(np.count_nonzero(coverage > 1)),
             overlap_at_anchor=int(np.count_nonzero((coverage > 1) & anchor_mask)),
             transparency=int(np.count_nonzero(inside & ~anchor_mask & (b == m["ink"]))),
             gaps=(mh - len({r for r, c in layout}), mw - len({c for r, c in layout})),
             witnesses=(any((r + 1, c) in layout for r, c in layout),
                        any((r, c + 1) in layout for r, c in layout)),
             bbox_ablation=not np.array_equal(_render(a.shape, m, bbox_steps=True), b),
             anchor_ablation=not np.array_equal(_render(a.shape, m, use_anchor=False), b),
             destructive_ablation=_destructive_centers(mask) != direct,
             local_overlap=any(abs(r - s) + abs(c - t) <= 2
                               for (r, c), (s, t) in itertools.combinations(direct, 2)))
    return m


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            metas[split].append(_audit_pair(pair))
    train = metas["train"]
    assert len({m["dy"] for m in train}) >= 2 and len({m["dx"] for m in train}) >= 2
    assert len({m["mask"].shape[0] - m["dy"] for m in train}) >= 2
    assert len({m["mask"].shape[1] - m["dx"] for m in train}) >= 2
    assert len({m["anchor"][0] for m in train}) >= 2 and len({m["anchor"][1] for m in train}) >= 2
    assert any(m["local_overlap"] for m in train)
    assert any(m["destructive_ablation"] for m in train), "simultaneous local detection needs a witness"
    assert any(m["overlap_at_anchor"] > 0 for m in train), "training must expose priority at a shared joint"
    assert any(m["transparency"] > 0 for m in train), "training must expose transparent mask holes"
    assert any(m["witnesses"] == (True, True) for m in train)
    assert all(m["bbox_ablation"] for m in train)
    assert any(m["anchor_ablation"] for m in train)
    for axis in (0, 1):
        if any(m["gaps"][axis] for m in metas["test"]):
            assert any(m["gaps"][axis] for m in train), "test gap lacks training evidence"
    return None


def self_test():
    """Bounded, in-memory 42-seed semantic and diversity checks; no file writes."""
    shapes, masks, layouts, steps, anchors, palettes, elbows = (set() for _ in range(7))
    counts, overlaps, signatures = set(), [], set()
    gap_axes, edges = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _audit_pair(pair)
                signature.update(repr(_key(pair["input"])).encode())
                shapes.add(m["shape"])
                masks.add(_key(m["mask"]))
                layouts.add((m["map_shape"], m["layout"]))
                steps.add((m["dy"], m["dx"]))
                anchors.add(m["anchor"])
                palettes.add((m["bg"], m["ink"], m["marker"]))
                counts.add(len(m["layout"]))
                overlaps.append(m["overlap"])
                for axis in (0, 1):
                    if m["gaps"][axis]:
                        gap_axes.add(axis)
                cr = m["crosses"]
                elbow = next(p for p in cr if sum(p[0] == q[0] for q in cr) == 2
                             and sum(p[1] == q[1] for q in cr) == 2)
                elbows.add((elbow[0] == min(r for r, c in cr), elbow[1] == min(c for r, c in cr)))
                span = (m["mask"].shape[0] + (m["map_shape"][0] - 1) * m["dy"],
                        m["mask"].shape[1] + (m["map_shape"][1] - 1) * m["dx"])
                for side, condition in enumerate((m["top"] == 0, m["left"] == 0,
                        m["top"] + span[0] == m["shape"][0], m["left"] + span[1] == m["shape"][1])):
                    if condition:
                        edges.add(side)
        signatures.add(signature.digest())
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(masks) >= 120 and len(layouts) >= 100
    assert len(steps) >= 20 and len(anchors) >= 15 and len(palettes) >= 80
    assert len(elbows) == 4 and gap_axes == {0, 1}
    assert len(counts) >= 10 and max(overlaps) > 50 and edges == {0, 1, 2, 3}
    # Determinism includes the output and supports every legal explicit count.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    p, q = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # All 3x3 neighborhoods, including clipped boundary candidates.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        assert bool(_centers(mask)) == all(mask[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)))
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, shapes=len(shapes), glyph_masks=len(masks), map_layouts=len(layouts),
                step_pairs=len(steps), anchor_sites=len(anchors), role_palettes=len(palettes),
                map_counts=len(counts), max_shared_pixels=max(overlaps))
