"""Stretched paint systems: contract identical vector runs, then supply holes.

Only NumPy and the standard library are used. Generation constructs geometric
ownership independently of the input-only solver. No files are read or written.
"""
from collections import deque
import hashlib

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
CUP_OFFSETS = frozenset(
    frozenset(((dr, 0), (0, dc), (dr, dc)))
    for dr in (-1, 1) for dc in (-1, 1)
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _contract(a):
    """Full-vector consecutive equality, computed from the unpainted input."""
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    ri, ci = np.flatnonzero(rows), np.flatnonzero(cols)
    return a[np.ix_(ri, ci)].copy(), ri, ci


def _components(mask, steps=D8):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    groups = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if labels[r, c] >= 0:
            continue
        ident = len(groups)
        labels[r, c] = ident
        queue = [(r, c)]
        for y, x in queue:
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and mask[ny, nx]
                        and labels[ny, nx] < 0):
                    labels[ny, nx] = ident
                    queue.append((ny, nx))
        groups.append(set(queue))
    return labels, groups


def _neighbors(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _terminal(a, cells, bg):
    """A singleton paint at the missing corner of a 2x2 string cap."""
    by_color = {}
    for p in cells:
        by_color.setdefault(int(a[p]), []).append(p)
    result = []
    for color, positions in by_color.items():
        if len(positions) != 1:
            continue
        p = positions[0]
        near = {q for q in _neighbors({p}, a.shape) if a[q] != bg}
        offsets = frozenset((q[0] - p[0], q[1] - p[1]) for q in near)
        if (offsets in CUP_OFFSETS and len({int(a[q]) for q in near}) == 1
                and all(a[q] != color for q in near)):
            result.append((p, color))
    if len(result) > 1:
        raise ValueError("Conflicting terminals in a single system")
    return result[0] if result else None


def _paint(a):
    """Analyze the compressed geometry; no generation metadata is consulted."""
    bg = int(a[0, 0])
    if not (np.all(a[[0, -1], :] == bg) and np.all(a[:, [0, -1]] == bg)):
        raise ValueError("The family has a uniform exterior border")
    labels, systems = _components(a != bg)
    terminals = [_terminal(a, s, bg) for s in systems]
    _, regions = _components(a == bg, D4)
    holes = []
    out = a.copy()
    for cells in regions:
        if any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
               for r, c in cells):
            continue
        owners = {int(labels[p]) for p in _neighbors(cells, a.shape, D4)
                  if labels[p] >= 0}
        if len(owners) != 1:
            raise ValueError("A cavity must have exactly one surrounding system")
        owner = next(iter(owners))
        holes.append((cells, owner))
        if terminals[owner] is not None:
            color = terminals[owner][1]
            for p in cells:
                out[p] = color
    return out, {"background": bg, "systems": systems, "labels": labels,
                 "terminals": terminals, "holes": holes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collapse maximal consecutive identical row/column runs, then paint."""
    compressed, _, _ = _contract(_grid(input_grid))
    return _paint(compressed)[0]


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values) - 1)]


def _move(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _contour(rng, small):
    """Four-connected row-interval interiors, enclosed by a minimal D4 shell."""
    kind = _pick(rng, ("box", "diamond", "slant", "steps", "waist", "bar"))
    h = _ri(rng, 1, 3 if small else 5)
    w = _ri(rng, 1, 3 if small else 5)
    if kind == "diamond":
        radius = _ri(rng, 1, 1 if small else 2)
        inner = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1) if abs(r) + abs(c) <= radius}
    else:
        if kind == "bar":
            h, w = 1, _ri(rng, 2, 4 if small else 7)
        inner = set()
        left = 0
        drift = _pick(rng, (-1, 1))
        previous = None
        for r in range(h):
            if r and kind in ("slant", "steps"):
                left += drift if kind == "slant" else _pick(rng, (-1, 0, 1))
            width = max(1, w - int(kind == "waist" and 0 < r < h - 1))
            if kind == "steps":
                width = max(1, w + _ri(rng, -1, 1))
            if previous is not None:
                # Adjacent row intervals overlap, so there is one cavity.
                left = min(max(left, previous[0] - width + 1), previous[1])
            inner.update((r, c) for c in range(left, left + width))
            previous = (left, left + width - 1)
    if rng.random() < .5:
        inner = {(c, r) for r, c in inner}
    shell = {(r + dr, c + dc) for r, c in inner for dr, dc in D4} - inner
    mr = min(r for r, c in shell)
    mc = min(c for r, c in shell)
    return _move(inner, -mr, -mc), _move(shell, -mr, -mc)


def _cap(rng):
    cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    source = _pick(rng, sorted(cells))
    return source, cells - {source}


def _path(rng, starts, goals, allowed, shape):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    prev = {p: None for p in starts}
    steps = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in steps:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in prev:
                prev[q] = p
                queue.append(q)
    if end is None:
        return None
    cells = set()
    while end is not None:
        cells.add(end)
        end = prev[end]
    return cells


def _active_nearest_wrong(info):
    """Count marked holes whose nearest terminal has a different paint value."""
    terminals = [t for t in info["terminals"] if t is not None]
    wrong = 0
    for cells, owner in info["holes"]:
        terminal = info["terminals"][owner]
        if terminal is None:
            continue
        distances = [(min(max(abs(r - t[0][0]), abs(c - t[0][1])) for r, c in cells), t[1])
                     for t in terminals]
        minimum = min(distance for distance, _ in distances)
        nearest_colors = {color for distance, color in distances if distance == minimum}
        wrong += terminal[1] not in nearest_colors
    return wrong


def _scene(rng, active, inactive, counts, background, paints, matched=False,
           require_nearest=False):
    """Free placement and route trees; return constructed input and paint mask."""
    nsys = active + inactive
    nloop = sum(counts)
    small = nloop >= 5
    low = 15 if nloop <= 3 else 19
    high = 23
    h, w = _ri(rng, low, high), _ri(rng, low, high)
    if nsys == 1 and nloop == 1:
        h, w = _ri(rng, 8, 18), _ri(rng, 10, 23)
    shape = (h, w)
    canvas = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    available = [int(v) for v in rng.permutation(10)
                 if int(v) not in [background] + paints]
    # Repeated wire colors across disconnected systems are legal.
    wires = available[:nsys]
    if nsys > 1 and rng.random() < .3:
        wires = [wires[0]] * nsys
    edges = available[nsys:nsys + _ri(rng, 1, 2)]
    occupied = set()
    loops = []
    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)
    matched_done = set()
    for owner in owners:
        matching_target = matched and owner not in matched_done
        if matching_target:
            # One supplied and one unsupplied target have identical geometry
            # and boundary color, defeating area/shape/color selection rules.
            inner = {(1, 1)}
            shell = {(0, 1), (1, 0), (1, 2), (2, 1)}
            matched_done.add(owner)
        else:
            inner, shell = _contour(rng, small)
        lh = max(r for r, c in shell) + 1
        lw = max(c for r, c in shell) + 1
        placed = False
        if lh > h - 2 or lw > w - 2:
            return None
        for _ in range(45):
            rr, cc = _ri(rng, 1, h - lh - 1), _ri(rng, 1, w - lw - 1)
            ins, edge = _move(inner, rr, cc), _move(shell, rr, cc)
            cells = ins | edge
            if cells & (occupied | _neighbors(occupied, shape)):
                continue
            loops.append((ins, edge, owner, edges[0] if matching_target else _pick(rng, edges)))
            occupied |= cells
            placed = True
            break
        if not placed:
            return None
    # End caps and their empty corner are protected before routing begins.
    caps, sources = [], []
    blocked = occupied | _neighbors(occupied, shape)
    for owner in range(nsys):
        p0, cap0 = _cap(rng)
        placed = False
        for _ in range(60):
            rr, cc = _ri(rng, 1, h - 3), _ri(rng, 1, w - 3)
            p = (p0[0] + rr, p0[1] + cc)
            cap = _move(cap0, rr, cc)
            cells = cap | {p}
            if cells & blocked:
                continue
            sources.append(p)
            caps.append(cap)
            blocked |= cells | _neighbors(cells, shape)
            placed = True
            break
        if not placed:
            return None
    loop_cells = set().union(*(ins | edge for ins, edge, _, _ in loops))
    wiresets = [set(cap) for cap in caps]
    cap_protection = set(sources) | _neighbors(set(sources), shape)
    for owner in [int(v) for v in rng.permutation(nsys)]:
        own = wiresets[owner]
        other = set().union(*(wire | {sources[i]} for i, wire in enumerate(wiresets)
                              if i != owner))
        protected = other | _neighbors(other, shape) | cap_protection
        tree = set(own)
        target_order = [int(v) for v in rng.permutation(len(loops))]
        for index in target_order:
            ins, edge, loop_owner, _ = loops[index]
            if loop_owner != owner:
                continue
            otherloops = loop_cells - (ins | edge)
            forbidden = loop_cells | _neighbors(otherloops, shape) | protected
            goals = (_neighbors(edge, shape) & canvas) - forbidden
            if not goals:
                return None
            allowed = (canvas - forbidden) | tree
            path = _path(rng, tree, goals, allowed, shape)
            if path is None:
                return None
            own |= path - loop_cells
            tree |= path | edge
    a = np.full(shape, background, dtype=np.int8)
    for ins, edge, owner, color in loops:
        for p in edge:
            a[p] = color
    for owner, cells in enumerate(wiresets):
        for p in cells:
            a[p] = wires[owner]
        if owner < active:
            a[sources[owner]] = paints[owner]
    b = a.copy()
    for ins, edge, owner, color in loops:
        if owner < active:
            for p in ins:
                b[p] = paints[owner]
    # Rotation and reflection apply to both geometric evidence and intended fill.
    k = _ri(rng, 0, 3)
    a, b = np.rot90(a, k), np.rot90(b, k)
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    canonical, ri, ci = _contract(a)
    intended = b[np.ix_(ri, ci)].copy()
    try:
        predicted, info = _paint(canonical)
    except ValueError:
        return None
    if (not np.array_equal(predicted, intended)
            or len(info["systems"]) != nsys
            or len(info["holes"]) != nloop
            or sum(t is not None for t in info["terminals"]) != active):
        return None
    # Every intended loop must still survive, owned by the intended route.
    observed_counts = sorted(sum(owner == i for _, owner in info["holes"])
                             for i in range(nsys))
    if observed_counts != sorted(counts):
        return None
    if require_nearest and not _active_nearest_wrong(info):
        return None
    return canonical, intended


def _axis_lengths(rng, length, forced, test, no_op=False):
    runs = np.ones(length, dtype=int)
    if no_op:
        return runs
    # Keep room for at least four copies of one vector in every test.
    index = forced[0]
    runs[index] = _ri(rng, 4, min(7, 31 - length)) if test else 2
    budget = 30 - int(runs.sum())
    indices = forced[1:] + [int(i) for i in rng.permutation(length)]
    extra = _ri(rng, 0, min(budget, 5 if test else 7))
    for i in indices:
        if budget <= 0 or extra <= 0:
            break
        amount = min(extra, budget, _ri(rng, 0, 2), max(0, (7 if test else 3) - runs[i]))
        runs[i] += amount
        budget -= amount
        extra -= amount
    return runs


def _stretched_pair(rng, canonical, intended, test, lesson):
    _, info = _paint(canonical)
    terminals = [t for t in info["terminals"] if t is not None]
    p = terminals[_ri(rng, 0, len(terminals) - 1)][0]
    cavity = sorted(next(cells for cells, owner in info["holes"]
                         if info["terminals"][owner] is not None))
    q = cavity[_ri(rng, 0, len(cavity) - 1)]
    no_op = None if lesson or rng.random() >= .18 else _ri(rng, 0, 1)
    rows = _axis_lengths(rng, canonical.shape[0], [p[0], q[0]], test, no_op == 0)
    cols = _axis_lengths(rng, canonical.shape[1], [p[1], q[1]], test, no_op == 1)
    a = np.repeat(np.repeat(canonical, rows, axis=0), cols, axis=1)
    return {"input": a, "output": intended.copy()}


def _check_counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(a):
    return a.shape, a.tobytes()


def _fallback_scene(rng, active, inactive, counts, background, paints):
    """Guaranteed sparse packing with the same counts and cap convention."""
    nsys = active + inactive
    palette = [int(c) for c in rng.permutation(10) if int(c) not in [background] + paints]
    wires, edge_color = palette[:nsys], palette[nsys]
    a = np.full((4 * nsys + 1, 23), background, dtype=np.int8)
    owned = []
    for owner, count in enumerate(counts):
        r = 1 + 4 * owner
        left = owner % 2 == 0
        source_c = 1 if left else 21
        other_c = source_c + 1 if left else source_c - 1
        a[r, other_c] = a[r + 1, source_c] = a[r + 1, other_c] = wires[owner]
        if owner < active:
            a[r, source_c] = paints[owner]
        centers = [12 + 4 * j for j in range(count)] if left else [4 + 4 * j for j in range(count)]
        for center in centers:
            for dr, dc in D4:
                a[r + 1 + dr, center + dc] = edge_color
            owned.append(((r + 1, center), owner))
        if left:
            a[r + 1, 3:centers[0] - 1] = wires[owner]
        else:
            a[r + 1, centers[-1] + 2:20] = wires[owner]
        for first, second in zip(centers, centers[1:]):
            a[r + 1, first + 2:second - 1] = wires[owner]
    b = a.copy()
    for point, owner in owned:
        if owner < active:
            b[point] = paints[owner]
    canonical, ri, ci = _contract(a)
    intended = b[np.ix_(ri, ci)].copy()
    predicted, info = _paint(canonical)
    assert np.array_equal(predicted, intended)
    assert len(info["systems"]) == nsys and len(info["holes"]) == sum(counts)
    assert sorted(sum(owner == i for _, owner in info["holes"]) for i in range(nsys)) == sorted(counts)
    return canonical, intended


def _fallback_pair(rng, active, inactive, counts, bg, paints, test, seen):
    canonical, intended = _fallback_scene(rng, active, inactive, counts, bg, paints)
    _, info = _paint(canonical)
    p = next(t[0] for t in info["terminals"] if t is not None)
    selectable = [r for r in range(canonical.shape[0]) if r != p[0]][:4]
    assert len(selectable) == 4
    # Sixteen distinct row-multiplicity patterns exceed the at-most-thirteen
    # previous inputs. All fit within 30 and preserve the split's shift.
    for serial in range(16):
        rows, cols = np.ones(canonical.shape[0], int), np.ones(canonical.shape[1], int)
        rows[p[0]] = cols[p[1]] = 4 if test else 2
        for bit, index in enumerate(selectable):
            rows[index] += (serial >> bit) & 1
        expanded = np.repeat(np.repeat(canonical, rows, axis=0), cols, axis=1)
        if _key(expanded) not in seen:
            return {"input": expanded, "output": intended}
    raise AssertionError("Sixteen fallback patterns cannot all be occupied")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_palette = [int(c) for c in rng.permutation(10)]
    fixed_background = bool(rng.integers(0, 2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, number in (("train", num_train), ("test", num_test)):
        test = split == "test"
        for index in range(number):
            lesson = not test and index < 2
            if lesson and index == 0:
                active, inactive, counts = 1, 1, [2, 1]
            elif lesson:
                active, inactive, counts = 2, 0, [1, 2]
            else:
                nsys = _pick(rng, (1, 2, 2, 3))
                active = _ri(rng, 1, nsys)
                inactive = nsys - active
                counts = [_ri(rng, 1, 3) for _ in range(nsys)]
                while sum(counts) > 6:
                    counts[int(np.argmax(counts))] -= 1
            if fixed_background:
                bg = base_palette[0]
            elif lesson:
                bg = base_palette[index]
            else:
                bg = _ri(rng, 0, 9)
            color_pool = [c for c in base_palette if c != bg]
            # The first lessons use different copied values, independently of geometry.
            if lesson:
                paints = [color_pool[(index + i + 1) % len(color_pool)] for i in range(active)]
            else:
                paints = [int(c) for c in rng.choice(color_pool, size=active, replace=False)]
                if active > 1 and rng.random() < .25:
                    paints = [paints[0]] * active
            pair = None
            for attempt in range(120):
                scene = _scene(rng, active, inactive, counts, bg, paints,
                               matched=lesson and index == 0,
                               require_nearest=lesson and index == 1)
                if scene is None:
                    continue
                candidate = _stretched_pair(rng, *scene, test=test, lesson=lesson)
                key = _key(candidate["input"])
                if key in seen:
                    continue
                pair = candidate
                seen.add(key)
                break
            if pair is None:
                pair = _fallback_pair(rng, active, inactive, counts, bg, paints, test, seen)
                seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _run_lengths(a, axis):
    vectors = a if axis == 0 else a.T
    lengths = [1]
    for i in range(1, len(vectors)):
        if np.array_equal(vectors[i], vectors[i - 1]):
            lengths[-1] += 1
        else:
            lengths.append(1)
    return tuple(lengths)


def _four_neighbor_shortcut(a, info):
    """Ablation: insist that a supplied cavity be connected by edges only."""
    labels, groups = _components(a != info["background"], D4)
    paints = {int(labels[t[0]]): t[1] for t in info["terminals"] if t is not None}
    out = a.copy()
    for cells, _ in info["holes"]:
        ids = {int(labels[p]) for p in _neighbors(cells, a.shape, D4) if labels[p] >= 0}
        if len(ids) == 1 and next(iter(ids)) in paints:
            for p in cells:
                out[p] = paints[next(iter(ids))]
    return out


def _features(pair):
    a, b = _grid(pair["input"]), _grid(pair["output"])
    c, _, _ = _contract(a)
    _, info = _paint(c)
    active = [i for i, t in enumerate(info["terminals"]) if t is not None]
    counts = [sum(owner == i for _, owner in info["holes"])
              for i in range(len(info["systems"]))]
    rows, cols = _run_lengths(a, 0), _run_lengths(a, 1)
    block_areas = [rows[t[0][0]] * cols[t[0][1]] for t in info["terminals"] if t is not None]
    matched = {False: set(), True: set()}
    for cells, owner in info["holes"]:
        r0, c0 = min(r for r, _ in cells), min(col for _, col in cells)
        shape = tuple(sorted((r - r0, col - c0) for r, col in cells))
        colors = tuple(sorted({int(c[p]) for p in _neighbors(cells, c.shape, D4)}))
        matched[info["terminals"][owner] is not None].add((shape, colors))
    return {"input_shape": a.shape, "output_shape": b.shape,
            "background": info["background"], "systems": len(counts),
            "active": len(active), "inactive": len(counts) - len(active),
            "holes": len(info["holes"]), "counts": tuple(sorted(counts)),
            "cavity_areas": [len(cells) for cells, _ in info["holes"]],
            "paints": sorted({t[1] for t in info["terminals"] if t is not None}),
            "row_runs": rows, "column_runs": cols, "terminal_block_areas": block_areas,
            "changed": int(np.count_nonzero(c != b)),
            "matched_distractor": bool(matched[False] & matched[True]),
            "wrong_nearest_active": _active_nearest_wrong(info),
            "four_differs": not np.array_equal(_four_neighbor_shortcut(c, info), b),
            "normalized_structure": _normalized_digest(c)}


def _normalized_digest(a):
    roles = {}
    values = bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)
    return hashlib.sha256(bytes(a.shape) + values).hexdigest()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    train_features = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "Inputs must be distinct"
            seen.add(_key(a))
            c, _, _ = _contract(a)
            predicted, info = _paint(c)
            assert np.array_equal(predicted, b), "Input-derived ownership disagrees"
            assert a.shape != c.shape and c.shape == b.shape
            assert np.any(c != b), "A marked cavity must actually fill"
            assert np.all(c[c != b] == info["background"])
            assert set(np.unique(b)) == set(np.unique(c)), "Copy existing terminal colors"
            assert np.array_equal(_contract(c)[0], c), "Contract all input runs"
            rows, cols = _run_lengths(a, 0), _run_lengths(a, 1)
            assert np.array_equal(np.repeat(np.repeat(c, rows, axis=0), cols, axis=1), a)
            terminals = [t for t in info["terminals"] if t is not None]
            assert 1 <= len(terminals) <= len(info["systems"]) <= 3
            assert 1 <= len(info["holes"]) <= 6
            assert any(rows[t[0][0]] * cols[t[0][1]] > 1 for t in terminals)
            assert all(sum(owner == i for _, owner in info["holes"]) <= 3
                       for i in range(len(info["systems"])))
            positions = {t[0] for t in terminals}
            structural_colors = {int(c[p]) for system in info["systems"]
                                 for p in system - positions}
            assert not structural_colors & {t[1] for t in terminals}
            assert all(len({int(c[p]) for p in _neighbors(cells, c.shape, D4)}) == 1
                       for cells, _ in info["holes"])
            assert all(any(owner == i for _, owner in info["holes"])
                       for i in range(len(info["systems"])))
            assert all(np.all(b[tuple(np.array(sorted(cells)).T)] ==
                              (info["background"] if info["terminals"][owner] is None
                               else info["terminals"][owner][1]))
                       for cells, owner in info["holes"])
            # Neither omitted parent is allowed to coincide with the right answer.
            assert not np.array_equal(c, b), "Omitting A leaves hollow cavities"
            assert a.shape != b.shape, "Omitting B retains duplicate vectors"
            f = _features(pair)
            if split == "train":
                assert max(rows + cols) <= 3
                if index < 2:
                    assert any(rows[t[0][0]] > 1 and cols[t[0][1]] > 1 for t in terminals)
                train_features.append(f)
            else:
                assert any(max(rows[t[0][0]], cols[t[0][1]]) >= 4 for t in terminals), (
                    "Test extrapolates terminal-vector multiplicity")
    assert any(f["inactive"] and f["holes"] > f["systems"] for f in train_features)
    assert any(f["matched_distractor"] for f in train_features), "Match supplied/unsupplied shapes"
    assert any(f["active"] >= 2 and len(f["paints"]) >= 2 for f in train_features)
    assert any(f["wrong_nearest_active"] for f in train_features), "Teach tracing over proximity"
    assert any(max(f["row_runs"]) > 1 and max(f["column_runs"]) > 1
               for f in train_features)
    assert len({paint for f in train_features for paint in f["paints"]}) >= 2
    assert any(f["four_differs"] for f in train_features), "Teach diagonal connectivity"


def self_test():
    """Bounded in-memory checks over 42 seeds; returns descriptive measurements."""
    shapes, output_shapes, structures, digests, backgrounds = set(), set(), set(), set(), set()
    systems, actives, holes, areas, inventories = set(), set(), set(), set(), set()
    one_axis = repeated_paint = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f["input_shape"])
                output_shapes.add(f["output_shape"])
                structures.add(f["normalized_structure"])
                backgrounds.add(f["background"])
                systems.add(f["systems"])
                actives.add(f["active"])
                holes.add(f["holes"])
                areas.update(f["cavity_areas"])
                inventories.add((f["systems"], f["active"], f["counts"]))
                one_axis += (max(f["row_runs"]) == 1) != (max(f["column_runs"]) == 1)
                c = _contract(pair["input"])[0]
                ts = [t[1] for t in _paint(c)[1]["terminals"] if t is not None]
                repeated_paint += len(ts) > len(set(ts))
                signature.update(bytes(pair["input"].shape) + pair["input"].tobytes())
                assert np.array_equal(solve(pair["input"]), pair["output"])
        digests.add(signature.digest())
    assert len(digests) == 42
    assert len(shapes) >= 40 and len(output_shapes) >= 40 and len(structures) >= 160
    assert len(backgrounds) >= 8 and systems == {1, 2, 3} and actives == {1, 2, 3}
    assert max(holes) >= 6 and len(areas) >= 8 and len(inventories) >= 12
    assert one_axis >= 5 and repeated_paint >= 2
    # Exercise the bounded fallback, including the full range of inventories.
    from itertools import product
    fallback_rng = np.random.default_rng(867)
    fallback_count = 0
    for nsys in range(1, 4):
        for counts in product(range(1, 4), repeat=nsys):
            if sum(counts) > 6:
                continue
            for active in range(1, nsys + 1):
                scene = _fallback_scene(fallback_rng, active, nsys - active, counts,
                                        0, list(range(1, active + 1)))
                assert np.array_equal(solve(scene[0]), scene[1])
                fallback_count += 1
    fallback_puzzle = {"train": [], "test": []}
    fallback_seen = set()
    for test, active, inactive, counts in ((False, 1, 1, [2, 1]),
                                           (False, 2, 0, [1, 2]),
                                           (True, 3, 0, [2, 2, 2])):
        pair = _fallback_pair(fallback_rng, active, inactive, counts, 0,
                              list(range(1, active + 1)), test, fallback_seen)
        fallback_seen.add(_key(pair["input"]))
        fallback_puzzle["test" if test else "train"].append(pair)
    validate_generated_puzzle(fallback_puzzle)
    assert _active_nearest_wrong(_paint(_contract(fallback_puzzle["train"][1]["input"])[0])[1])
    # Hold thirteen of its sixteen deterministic expansion patterns occupied.
    fallback_seen.clear()
    for _ in range(14):
        # Reset the local RNG so the canonical scene is identical every time.
        pair = _fallback_pair(np.random.default_rng(12), 2, 0, [1, 2], 0,
                              [1, 2], False, fallback_seen)
        fallback_seen.add(_key(pair["input"]))
    assert len(fallback_seen) == 14
    # All legal explicit counts, rejecting non-integers and booleans as well.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1700 + 10 * nt + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p, q = generate_puzzle(0), generate_puzzle(0)
    for split in p:
        for first, second in zip(p[split], q[split]):
            assert all(np.array_equal(first[k], second[k]) for k in ("input", "output"))
    # Changing one local source must affect exactly its own supplied cavities.
    pair = p["train"][0]
    canonical, _, _ = _contract(pair["input"])
    _, info = _paint(canonical)
    owner = next(i for i, t in enumerate(info["terminals"]) if t is not None)
    point, _ = info["terminals"][owner]
    fresh = next(color for color in range(10) if color not in canonical)
    altered = canonical.copy()
    expected = pair["output"].copy()
    altered[point] = expected[point] = fresh
    for cells, system in info["holes"]:
        if system == owner:
            for cell in cells:
                expected[cell] = fresh
    expanded = np.repeat(np.repeat(altered, _run_lengths(pair["input"], 0), axis=0),
                         _run_lengths(pair["input"], 1), axis=1)
    assert np.array_equal(solve(expanded), expected)
    # Reject a single omitted filled pixel.
    bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
    pair = bad["test"][0]
    c = _contract(pair["input"])[0]
    r, col = np.argwhere(c != pair["output"])[0]
    pair["output"][r, col] = c[r, col]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("An incomplete cavity fill passed validation")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "canonical_structures": len(structures), "backgrounds": sorted(backgrounds),
            "system_counts": sorted(systems), "active_counts": sorted(actives),
            "cavity_counts": sorted(holes), "cavity_areas": sorted(areas),
            "ownership_inventories": len(inventories), "one_axis_pairs": one_axis,
            "repeated_paint_pairs": repeated_paint, "fallback_inventories_checked": fallback_count}
