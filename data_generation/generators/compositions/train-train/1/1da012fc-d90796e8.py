"""Cut and rank: local contact cuts determine global palette correspondence.

Only Python's standard library and NumPy are used. `solve` reads grids only;
the carved-room constructor is deliberately separate from the solving rule.
"""
from itertools import combinations
import json

import numpy as np


ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in ORTHO:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    result.sort(key=_rank)
    return result


def _rank(cells):
    return min(r for r, _ in cells), min(c for _, c in cells)


def _contacts(grid):
    es = {tuple(map(int, p)) for p in np.argwhere(grid == 2)}
    ts = {tuple(map(int, p)) for p in np.argwhere(grid == 3)}
    matches = []
    for r, c in sorted(es):
        adjacent = [(r + dr, c + dc) for dr, dc in ORTHO
                    if (r + dr, c + dc) in ts]
        if len(adjacent) > 1:
            raise ValueError("eraser has multiple targets")
        if adjacent:
            matches.append(((r, c), adjacent[0]))
    if len({t for _, t in matches}) != len(matches):
        raise ValueError("target has multiple erasers")
    return matches, es, ts


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(int, copy=False)


def _parse(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1 or counts.max() * 2 <= a.size:
        raise ValueError("background must be a strict majority")
    bg = int(colors[np.argmax(counts)])
    if bg in (2, 3):
        raise ValueError("background cannot be an active marker")
    candidates = []
    for carrier in map(int, colors):
        if carrier in (bg, 2, 3):
            continue
        positions = np.argwhere(a == carrier)
        r0, c0 = map(int, positions.min(axis=0))
        r1, c1 = map(int, positions.max(axis=0))
        rect = a[r0:r1 + 1, c0:c1 + 1]
        if min(rect.shape) < 3 or np.mean(rect == carrier) < .70:
            continue
        if not (np.all(rect[0] == carrier) and np.all(rect[-1] == carrier)
                and np.all(rect[:, 0] == carrier) and np.all(rect[:, -1] == carrier)):
            continue
        if any(np.any(rect == v) for v in (bg, 2, 3)):
            continue
        label_points = [tuple(map(int, p)) for p in np.argwhere(rect != carrier)]
        if not 2 <= len(label_points) <= 8:
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1
               for (r, c), (rr, cc) in combinations(label_points, 2)):
            continue
        panel = np.zeros(a.shape, dtype=bool)
        panel[r0:r1 + 1, c0:c1 + 1] = True
        exterior_colors = set(map(int, np.unique(a[~panel]))) - {bg, 2, 3}
        if len(exterior_colors) != 1:
            continue
        body = exterior_colors.pop()
        labels = [int(rect[p]) for p in label_points]
        if body == carrier or body in labels or len(set(labels)) < 2:
            continue
        # A full background gap separates the palette from all material.
        expanded = np.zeros(a.shape, dtype=bool)
        expanded[max(0, r0 - 1):r1 + 2, max(0, c0 - 1):c1 + 2] = True
        if np.any(a[expanded & ~panel] != bg):
            continue
        material = (a != bg) & ~panel
        matches, es, ts = _contacts(a)
        removed = {e for e, _ in matches}
        activated = {t for _, t in matches}
        surviving = material.copy()
        for p in removed:
            surviving[p] = False
        comps = _components(surviving)
        if len(comps) != len(labels):
            continue
        if any(sum(int(a[p]) == body for p in comp) < 3 for comp in comps):
            continue
        if len({_rank(comp) for comp in comps}) != len(comps):
            continue
        candidates.append(dict(grid=a, bg=bg, body=body, carrier=carrier,
                               panel=panel, bbox=(r0, c0, r1, c1), labels=labels,
                               comps=comps, material=material, surviving=surviving,
                               before=_components(material), matches=matches,
                               removed=removed, activated=activated, es=es, ts=ts))
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous palette/material interpretation")
    return candidates[0]


def _paint(a, labels=None, groups=None):
    out = a["grid"].copy()
    for p in a["removed"]:
        out[p] = a["bg"]
    groups = a["comps"] if groups is None else groups
    labels = a["labels"] if labels is None else labels
    for comp, color in zip(groups, labels):
        for p in comp:
            if int(a["grid"][p]) == a["body"] or p in a["activated"]:
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the fixed rule using only information visible in the input."""
    return _paint(_parse(input_grid))


def _facts(a):
    unmatched_e = a["es"] - a["removed"]
    unmatched_t = a["ts"] - a["activated"]
    diagonal = any((r + dr, c + dc) in unmatched_t
                   for r, c in unmatched_e for dr, dc in DIAG)
    gap = any((r + 2 * dr, c + 2 * dc) in unmatched_t
              and int(a["grid"][r + dr, c + dc]) == a["bg"]
              for r, c in unmatched_e for dr, dc in ORTHO)
    # The midpoint is in bounds whenever both endpoints of an axial gap exist.
    essential = a["surviving"].copy()
    for p in unmatched_e:
        essential[p] = False
    nonsevering = 0
    for e in a["removed"]:
        one_cut = a["material"].copy()
        one_cut[e] = False
        nonsevering += len(_components(one_cut)) == len(a["before"])
    return dict(diagonal=diagonal, gap=gap,
                essential_retained=len(_components(essential)) > len(a["comps"]),
                nonsevering=nonsevering,
                horizontal=any(e[0] == t[0] for e, t in a["matches"]),
                vertical=any(e[1] == t[1] for e, t in a["matches"]),
                fragments=len(a["comps"]), initial=len(a["before"]),
                matches=len(a["matches"]), retained=len(unmatched_e))


def _no_split(a):
    """Ablation: local deletion, but one color for each original network."""
    colors = []
    for old in a["before"]:
        colors.append(next(color for comp, color in zip(a["comps"], a["labels"])
                           if comp & old))
    return _paint(a, colors, a["before"])


def _all_erasers(a):
    """Ablation: erase every 2, regardless of immediate target contact."""
    out = _paint(a)
    for p in a["es"]:
        out[p] = a["bg"]
    return out


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _lattice(rng, n):
    rooms = {(int(rng.integers(3)), int(rng.integers(3)))}
    while len(rooms) < n:
        frontier = sorted({(r + dr, c + dc) for r, c in rooms for dr, dc in ORTHO
                           if 0 <= r + dr < 3 and 0 <= c + dc < 3} - rooms)
        rooms.add(frontier[int(rng.integers(len(frontier)))])
    lowr, lowc = min(r for r, _ in rooms), min(c for _, c in rooms)
    return sorted((r - lowr, c - lowc) for r, c in rooms)


def _partitions(n, maximum=3, rows=3):
    if n == 0:
        return [()]
    if rows == 0:
        return []
    return [(first,) + rest for first in range(1, min(maximum, n) + 1)
            for rest in _partitions(n - first, maximum, rows - 1)]


def _field(rng, active_count, special, dent, complex_layout):
    """Build material in private symbolic colors: -1 empty, 1 body, 2/3 marks."""
    n = int(rng.integers(active_count + 2, min(9, active_count + 5)))
    rooms = _lattice(rng, n)
    roomset = set(rooms)
    nr, nc = max(r for r, _ in rooms) + 1, max(c for _, c in rooms) + 1
    heights = [int(x) for x in rng.integers(3, 6, nr)]
    widths = [int(x) for x in rng.integers(3, 6, nc)]
    tops = [sum(heights[:r]) + r for r in range(nr)]
    lefts = [sum(widths[:c]) + c for c in range(nc)]
    # Random spanning tree, plus optional cycle edges.
    reached = {rooms[int(rng.integers(n))]}
    tree = []
    while len(reached) < n:
        options = sorted((p, (p[0] + dr, p[1] + dc)) for p in reached
                         for dr, dc in ORTHO
                         if (p[0] + dr, p[1] + dc) in roomset - reached)
        p, q = options[int(rng.integers(len(options)))]
        tree.append(tuple(sorted((p, q))))
        reached.add(q)
    active_indices = set(map(int, rng.choice(len(tree), active_count, replace=False)))
    active = {e for i, e in enumerate(tree) if i in active_indices}
    edges = set(tree)
    if complex_layout:
        extras = sorted({tuple(sorted((p, (p[0] + dr, p[1] + dc))))
                         for p in rooms for dr, dc in ORTHO
                         if (p[0] + dr, p[1] + dc) in roomset} - edges)
        if extras and rng.random() < .60:
            edges.add(extras[int(rng.integers(len(extras)))])
        inactive = sorted(edges - active)
        if len(inactive) >= 2 and rng.random() < .30:
            edges.remove(inactive[int(rng.integers(len(inactive)))])
    inactive = sorted(edges - active)
    special_options = [e for e in inactive if special != "gap" or
                       (heights[e[0][0]] if e[0][0] == e[1][0]
                        else widths[e[0][1]]) >= 4]
    if not special_options:
        return None
    special_edge = special_options[int(rng.integers(len(special_options)))]
    occupied, protected, forbidden = set(), set(), set()
    erasers, targets = set(), set()
    room_cells = {}
    for r, c in rooms:
        cells = {(tops[r] + dr, lefts[c] + dc)
                 for dr in range(heights[r]) for dc in range(widths[c])}
        occupied.update(cells)
        room_cells[r, c] = cells
    for edge in sorted(edges):
        p, q = edge
        horizontal = p[0] == q[0]
        length = heights[p[0]] if horizontal else widths[p[1]]
        kind = "active" if edge in active else (special if edge == special_edge else "plain")
        if kind == "gap":
            offset = 1 if bool(rng.integers(2)) else length - 2
            delta = 2 if offset == 1 else -2
        else:
            offset = int(rng.integers(1, length - 1))
            delta = 1 if bool(rng.integers(2)) else -1
        if horizontal:
            e = (tops[p[0]] + offset, lefts[p[1]] + widths[p[1]])
            cross = (0, 1)
            along = (1, 0)
        else:
            e = (tops[p[0]] + heights[p[0]], lefts[p[1]] + offset)
            cross = (1, 0)
            along = (0, 1)
        erasers.add(e)
        protected.update((e[0] + sign * cross[0], e[1] + sign * cross[1])
                         for sign in (-1, 1))
        if kind == "plain":
            continue
        hook = 1 if bool(rng.integers(2)) else -1
        if kind == "diagonal":
            t = (e[0] + delta * along[0] + hook * cross[0],
                 e[1] + delta * along[1] + hook * cross[1])
            targets.add(t)
            protected.add(t)
        else:
            t = (e[0] + delta * along[0], e[1] + delta * along[1])
            targets.add(t)
            protected.add((t[0] + hook * cross[0], t[1] + hook * cross[1]))
            forbidden.add((t[0] - hook * cross[0], t[1] - hook * cross[1]))
    if protected & forbidden or not protected <= occupied:
        return None
    occupied -= forbidden
    h, w = sum(heights) + nr - 1, sum(widths) + nc - 1
    # Preserve each room's connectivity and one point on every available edge.
    for cells in room_cells.values():
        cells = cells & occupied
        if len(_components(_mask(cells, (h, w)))) != 1:
            return None
        for axis in (0, 1):
            for extremum in (min(p[axis] for p in cells), max(p[axis] for p in cells)):
                options = sorted(p for p in cells if p[axis] == extremum)
                protected.add(options[int(rng.integers(len(options)))])
        candidates = sorted(cells - protected)
        rng.shuffle(candidates)
        budget = int(rng.integers(0, max(2, len(cells) // 2)))
        for point in candidates[:budget]:
            trial = cells - {point}
            if len(trial) >= 6 and len(_components(_mask(trial, (h, w)))) == 1:
                cells = trial
                occupied.remove(point)
    field = np.full((h, w), -1, dtype=int)
    for p in occupied:
        field[p] = 1
    for p in erasers:
        field[p] = 2
    for p in targets:
        field[p] = 3
    if dent:
        before_count = len(_components(field != -1))
        options = []
        for p in sorted(occupied - targets - protected):
            for dr, dc in ORTHO:
                q = (p[0] + dr, p[1] + dc)
                if q not in occupied - targets - protected or q == p:
                    continue
                trial = field.copy()
                trial[p], trial[q] = 2, 3
                try:
                    _contacts(trial)
                except ValueError:
                    continue
                mask = trial != -1
                mask[p] = False
                if len(_components(mask)) == before_count:
                    options.append((p, q))
        if not options:
            return None
        p, q = options[int(rng.integers(len(options)))]
        field[p], field[q] = 2, 3
    try:
        matches, es, _ = _contacts(field)
    except ValueError:
        return None
    surviving = field != -1
    for e, _ in matches:
        surviving[e] = False
    comps = _components(surviving)
    if not 2 <= len(comps) <= 8 or any(sum(field[p] == 1 for p in c) < 3 for c in comps):
        return None
    if len(comps) <= len(_components(field != -1)):
        return None
    essential = surviving.copy()
    for p in es - {e for e, _ in matches}:
        essential[p] = False
    if len(_components(essential)) <= len(comps):
        return None
    return field


def _mask(cells, shape):
    mask = np.zeros(shape, dtype=bool)
    for p in cells:
        mask[p] = True
    return mask


def _make_pair(rng, roles, active_count, special, dent, complex_layout, direction):
    field = _field(rng, active_count, special, dent, complex_layout)
    if field is None:
        return None
    field = np.rot90(field, int(rng.integers(4)))
    if bool(rng.integers(2)):
        field = np.fliplr(field)
    matches, _, _ = _contacts(field)
    has_direction = any((e[0] == t[0]) == (direction == "H") for e, t in matches)
    if direction and not has_direction:
        field = field.T
        matches, _, _ = _contacts(field)
    survivor = field != -1
    for e, _ in matches:
        survivor[e] = False
    comps = _components(survivor)
    if len({_rank(c) for c in comps}) != len(comps):
        return None
    n = len(comps)
    bg, body, carrier = roles
    eligible = [c for c in range(10) if c not in {bg, body, carrier, 2, 3}]
    labels = [int(c) for c in rng.choice(eligible, n)]
    if len(set(labels)) < 2 or labels == labels[::-1]:
        return None
    core_two_cut = (active_count == 2 and special == "gap" and not dent
                    and not complex_layout and direction is not None)
    if core_two_cut:
        # Separate ordinal assignment from numeric and area sorting even when
        # the puzzle contains only the minimum two demonstrations.
        if (len(set(labels)) != n or labels == sorted(labels)
                or labels == sorted(labels, reverse=True)):
            return None
        if sorted(range(n), key=lambda i: (len(comps[i]), _rank(comps[i]))) == list(range(n)):
            return None
    packs = _partitions(n)
    bands = packs[int(rng.integers(len(packs)))]
    ph = max(4, 2 * len(bands) + 1) + int(rng.integers(2))
    pw = max(4, 2 * max(bands) + 1) + int(rng.integers(2))
    panel = np.full((ph, pw), carrier, dtype=int)
    k = 0
    for row, count in enumerate(bands):
        columns = [cols for cols in combinations(range(1, pw - 1), count)
                   if all(b - a >= 2 for a, b in zip(cols, cols[1:]))]
        cols = columns[int(rng.integers(len(columns)))]
        for col in cols:
            panel[2 * row + 1, col] = labels[k]
            k += 1
    fh, fw = field.shape
    side = int(rng.integers(4))
    if side < 2:
        h, w = max(fh, ph), fw + pw + 2
        fr, pr = int(rng.integers(h - fh + 1)), int(rng.integers(h - ph + 1))
        fc, pc = (0, fw + 2) if side == 0 else (pw + 2, 0)
    else:
        h, w = fh + ph + 2, max(fw, pw)
        fc, pc = int(rng.integers(w - fw + 1)), int(rng.integers(w - pw + 1))
        fr, pr = (0, fh + 2) if side == 2 else (ph + 2, 0)
    if max(h, w) > 30:
        return None
    foreground = np.count_nonzero(field != -1) + panel.size
    while h * w <= 2 * foreground:
        if h < 30 and (h <= w or w == 30):
            h += 1
        elif w < 30:
            w += 1
        else:
            return None
    height = int(rng.integers(h, min(30, h + 5) + 1))
    width = int(rng.integers(w, min(30, w + 5) + 1))
    min_used_h, min_used_w = max(fr + fh, pr + ph), max(fc + fw, pc + pw)
    top = int(rng.integers(height - min_used_h + 1))
    left = int(rng.integers(width - min_used_w + 1))
    grid = np.full((height, width), bg, dtype=int)
    block = np.where(field == -1, bg, np.where(field == 1, body, field))
    grid[top + fr:top + fr + fh, left + fc:left + fc + fw] = block
    grid[top + pr:top + pr + ph, left + pc:left + pc + pw] = panel
    try:
        a = _parse(grid)
    except ValueError:
        return None
    out = _paint(a)
    facts = _facts(a)
    if not facts[special] or (dent and not facts["nonsevering"]):
        return None
    if core_two_cut and len({int(out[p]) for p in a["activated"]}) < 2:
        return None
    if np.array_equal(out, _no_split(a)):
        return None
    return {"input": grid, "output": out}


def _key(grid):
    return tuple(grid.shape), tuple(map(int, grid.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _count(seed, "seed", 0, 2**128 - 1)
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    role_pool = [c for c in range(10) if c not in (2, 3)]
    base_roles = tuple(map(int, rng.choice(role_pool, 3, replace=False)))
    varied_roles = bool(rng.integers(2))
    first_direction = "H" if bool(rng.integers(2)) else "V"
    plans = [(1, "diagonal", True, False, first_direction),
             (2, "gap", False, False, "V" if first_direction == "H" else "H")]
    for _ in range(nt - 2):
        plans.append((int(rng.integers(1, 4)),
                      "diagonal" if bool(rng.integers(2)) else "gap",
                      rng.random() < .45, True, None))
    rng.shuffle(plans)
    test_plans = [(int(rng.choice([1, 2, 3, 4], p=[.08, .17, .45, .30])),
                   "diagonal" if bool(rng.integers(2)) else "gap",
                   rng.random() < .35, True, None) for _ in range(ns)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, split_plans in (("train", plans), ("test", test_plans)):
        for plan in split_plans:
            for _ in range(350):
                roles = (tuple(map(int, rng.choice(role_pool, 3, replace=False)))
                         if varied_roles else base_roles)
                pair = _make_pair(rng, roles, *plan)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct a valid unique example")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_facts = set(), []
    palette_witness = gate_witness = split_witness = False
    numeric_witness = size_witness = False
    target_colors = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _parse(inp)
            facts = _facts(a)
            assert facts["matches"] > 0 and facts["retained"] > 0
            assert facts["fragments"] > facts["initial"]
            assert facts["essential_retained"]
            assert np.array_equal(out, _paint(a))
            assert np.array_equal(out[a["panel"]], inp[a["panel"]])
            assert np.all(out[~a["material"] & ~a["panel"]] == a["bg"])
            for p in a["es"] - a["removed"] | (a["ts"] - a["activated"]):
                assert out[p] == inp[p]
            for p in a["removed"]:
                assert out[p] == a["bg"]
            for comp, label in zip(a["comps"], a["labels"]):
                for p in comp:
                    if inp[p] == a["body"] or p in a["activated"]:
                        assert out[p] == label
            if split == "train":
                train_facts.append(facts)
                palette_witness |= not np.array_equal(out, _paint(a, a["labels"][::-1]))
                gate_witness |= not np.array_equal(out, _all_erasers(a))
                split_witness |= not np.array_equal(out, _no_split(a))
                numeric_witness |= (not np.array_equal(out, _paint(a, sorted(a["labels"])))
                                    and not np.array_equal(out, _paint(a, sorted(a["labels"], reverse=True))))
                by_size = sorted(a["comps"], key=lambda c: (len(c), _rank(c)))
                size_witness |= not np.array_equal(out, _paint(a, groups=by_size))
                target_colors.update(int(out[p]) for p in a["activated"])
    assert palette_witness and gate_witness and split_witness
    assert numeric_witness and size_witness and len(target_colors) >= 2
    for feature in ("horizontal", "vertical", "diagonal", "gap", "nonsevering"):
        assert any(f[feature] for f in train_facts), feature


def self_test():
    """Bounded 42-seed structural/semantic checks, plus count boundaries."""
    shapes, layouts, roles, labels, signatures = set(), set(), set(), set(), set()
    facts = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _parse(pair["input"])
                f = _facts(a)
                facts.append(f)
                shapes.add(pair["input"].shape)
                layouts.add((f["fragments"], f["initial"], f["matches"], f["retained"],
                             tuple(sorted(len(c) for c in a["comps"]))))
                roles.add((a["bg"], a["body"], a["carrier"]))
                labels.add(tuple(a["labels"]))
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(layouts) >= 80 and len(roles) >= 30 and len(labels) >= 80
    assert any(f["fragments"] >= 5 for f in facts)
    assert any(f["initial"] >= 2 for f in facts)
    assert any(f["matches"] >= 4 for f in facts)
    assert any(h > w for h, w in shapes) and any(h < w for h, w in shapes)
    # The supplied verifier exhausts all legal explicit count combinations.
    # Check representative boundaries here without duplicating that full sweep.
    for nt, ns in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(12345 + nt * 5 + ns, nt, ns)
        assert len(p["train"]) == nt and len(p["test"]) == ns
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    one, two = generate_puzzle(909, 2, 1), generate_puzzle(909, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(one[s], two[s]) for k in ("input", "output"))
    # A hand-built, free-form network with edge contacts and a retained bridge.
    _semantic_test()
    return {"seeds": 42, "shapes": len(shapes), "layouts": len(layouts),
            "role_palettes": len(roles), "label_sequences": len(labels)}


def _semantic_test():
    # Its shape is independent of the room constructor; left and right regions
    # are linked by a cuttable 2, with a separate unactivated 2 inside the left.
    g = np.zeros((11, 17), dtype=int)
    g[0:4, 0:3] = 5
    g[1:4, 4:7] = 5
    g[0, 3], g[1, 3] = 3, 2
    g[2, 1] = 2
    g[5:10, 10:17] = 1
    g[6, 11], g[8, 15] = 4, 8
    out = solve(g)
    assert out[1, 3] == 0 and out[0, 3] == 4
    assert out[2, 1] == 2 and out[3, 6] == 8
    # Changing just an orthogonal contact to a diagonal is handled by a new
    # valid palette count only; malformed counts must not be silently guessed.
    bad = g.copy()
    bad[0, 3], bad[0, 2] = 0, 3
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted label/fragment mismatch")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
