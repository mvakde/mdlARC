"""Shared seam stamps: overlapping contact maps constrain a keyed square tiling.

Only standard-library modules and NumPy are used. No parent generator is called.
"""
from collections import Counter
from itertools import combinations
import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    stack.append(p)
        groups.append(found)
    return groups


def _normalize(cells):
    r0 = min(p[0] for p in cells)
    c0 = min(p[1] for p in cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}


def _dimensions(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _rotate(cells, turns):
    result = dict(cells)
    for _ in range(turns % 4):
        h, _ = _dimensions(result)
        result = {(c, h-1-r): v for (r, c), v in result.items()}
    return _normalize(result)


def _parse(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[counts.argmax()])
    _require(counts.max()*2 > grid.size, "background must occupy a strict majority")
    components = _groups((int(r), int(c)) for r, c in np.argwhere(grid != background))
    _require(len(components) == 4, "need four disconnected pieces")
    for a, b in combinations(components, 2):
        _require(all(max(abs(r-u), abs(c-v)) >= 2 for r, c in a for u, v in b),
                 "pieces need a one-cell background gap")
    legends = []
    for r in range(grid.shape[0]-1):
        for c in range(grid.shape[1]-1):
            patch = grid[r:r+2, c:c+2]
            if background not in patch and len(set(map(int, patch.flat))) == 4:
                legends.append((r, c, patch.copy()))
    _require(len(legends) == 1, "need one unambiguous four-color 2x2 legend")
    r0, c0, legend = legends[0]
    legend_points = {(r0+i, c0+j) for i in range(2) for j in range(2)}
    anchor = next(i for i, cells in enumerate(components) if legend_points <= cells)
    components = [components[anchor]] + [p for i, p in enumerate(components) if i != anchor]
    key_colors = set(map(int, legend.flat))
    parts = []
    for i, cells in enumerate(components):
        ink = {p for p in cells if int(grid[p]) in key_colors}
        if i == 0:
            ink -= legend_points
        material = cells - ink
        carrier_points = material - (legend_points if i == 0 else set())
        _require(len(carrier_points) >= 3, "piece needs visible material")
        carriers = {int(grid[p]) for p in carrier_points}
        _require(len(carriers) == 1 and carriers.isdisjoint(key_colors), "unclear carrier")
        stamps = _groups(ink)
        _require(stamps, "each piece must participate in registration")
        _require(all(2 <= len(s) <= 12 and len({int(grid[p]) for p in s}) == 2
                     for s in stamps), "stamps must be two-color connected patches")
        for a, b in combinations(stamps + ([legend_points] if i == 0 else []), 2):
            _require(all(max(abs(r-u), abs(c-v)) >= 2 for r, c in a for u, v in b),
                     "ink patches must have a carrier/background gap")
        rr, cc = min(r for r, c in cells), min(c for r, c in cells)
        local = {(r-rr, c-cc): int(grid[r, c]) for r, c in cells}
        local_material = {(r-rr, c-cc) for r, c in material}
        local_stamps = [{(r-rr, c-cc): int(grid[r, c]) for r, c in stamp} for stamp in stamps]
        parts.append({"cells": local, "material": local_material,
                      "stamps": local_stamps, "carrier": next(iter(carriers))})
    material_area = sum(len(p["material"]) for p in parts)
    ink_area = sum(sum(map(len, p["stamps"])) for p in parts)
    _require(ink_area % 2 == 0, "every seam stamp must occur twice")
    merged_area = material_area + ink_area//2
    side = int(merged_area**0.5)
    _require(side*side == merged_area and 4 <= side <= 30, "merged area must be a square")
    return {"parts": parts, "key": legend, "side": side, "background": background,
            "ink_area": ink_area, "material_area": material_area}


def _candidate(part, side, corner, turns):
    # Rotate the whole colored footprint; translate its bounding box to a corner.
    h, w = _dimensions(part["cells"])
    def transform(p):
        r, c = p
        hh, ww = h, w
        for _ in range(turns):
            r, c, hh, ww = c, hh-1-r, ww, hh
        return r, c
    cells = {transform(p): v for p, v in part["cells"].items()}
    height, width = _dimensions(cells)
    if height > side or width > side:
        return None
    dr = 0 if corner < 2 else side-height
    dc = 0 if corner % 2 == 0 else side-width
    def index(p):
        r, c = transform(p)
        return (r+dr)*side+c+dc
    footprint = sum(1 << index(p) for p in part["cells"])
    corners = (0, side-1, side*(side-1), side*side-1)
    if sum(bool(footprint & (1 << q)) for q in corners) != 1 or not footprint & (1 << corners[corner]):
        return None
    material = sum(1 << index(p) for p in part["material"])
    stamps = [tuple(sorted((index(p), v) for p, v in s.items())) for s in part["stamps"]]
    stamp_masks = [sum(1 << q for q, v in s) for s in stamps]
    return {"footprint": footprint, "material": material,
            "stamps": tuple(stamps), "stamp_masks": tuple(stamp_masks),
            "turns": turns, "corner": corner}


def _compatible(a, b, use_ink):
    overlap = a["footprint"] & b["footprint"]
    if overlap & (a["material"] | b["material"]):
        return False
    if not overlap:
        return True
    # Partial stamp matches are not a valid registration, even if colors agree.
    for masks in (a["stamp_masks"], b["stamp_masks"]):
        if any(mask & overlap and mask & overlap != mask for mask in masks):
            return False
    if use_ink:
        common = set(a["stamps"]) & set(b["stamps"])
        covered = 0
        for stamp in common:
            covered |= sum(1 << q for q, v in stamp)
        return covered == overlap
    return True


def _render_placement(analysis, placement):
    side = analysis["side"]
    out = np.full(side*side, -1, dtype=np.int16)
    for piece, candidate in sorted(placement.items()):
        color = int(analysis["key"].flat[candidate["corner"]])
        mask = candidate["material"]
        for q in range(side*side):
            if mask & (1 << q):
                out[q] = color
        for stamp in candidate["stamps"]:
            for q, v in stamp:
                # In a valid solution both copies agree. The first-copy policy
                # is used only to exhibit the deliberately ink-blind ablation.
                if out[q] < 0:
                    out[q] = v
    _require(bool(np.all(out >= 0)), "packing left a hole")
    return out.reshape(side, side).astype(np.int8)


def _pack(analysis, use_ink=True, limit=2):
    parts, side = analysis["parts"], analysis["side"]
    full = (1 << (side*side))-1
    anchor = _candidate(parts[0], side, 0, 0)
    if anchor is None:
        return []
    table = {}
    for piece in range(1, 4):
        for corner in range(1, 4):
            choices, seen = [], set()
            for turns in range(4):
                candidate = _candidate(parts[piece], side, corner, turns)
                if candidate is not None:
                    sig = candidate["material"], candidate["stamps"]
                    if sig not in seen:
                        seen.add(sig)
                        choices.append(candidate)
            table[piece, corner] = choices
    placement, solutions, answers = {0: anchor}, [], set()
    def search(corner, remaining, occupied):
        if len(solutions) >= limit:
            return
        if corner == 4:
            if occupied != full:
                return
            occurrences = Counter()
            adjacency = {i: set() for i in placement}
            for i, cand in placement.items():
                for stamp, mask in zip(cand["stamps"], cand["stamp_masks"]):
                    occurrences[stamp if use_ink else mask] += 1
                for j, other in placement.items():
                    if i != j and cand["footprint"] & other["footprint"]:
                        adjacency[i].add(j)
            if set(occurrences.values()) != {2}:
                return
            visited, pending = {0}, [0]
            while pending:
                i = pending.pop()
                for j in adjacency[i] - visited:
                    visited.add(j)
                    pending.append(j)
            if len(visited) != 4:
                return
            if use_ink:
                # A stamp is a contact map: its two colors name its two
                # incident corner pieces, and each color reaches that material.
                for stamp in occurrences:
                    partners = [p for p in placement.values() if stamp in p["stamps"]]
                    if {v for q, v in stamp} != {int(analysis["key"].flat[p["corner"]]) for p in partners}:
                        return
                    for partner in partners:
                        color = int(analysis["key"].flat[partner["corner"]])
                        if not any(partner["material"] & (1 << (rr*side+cc))
                                   for q, v in stamp if v == color for dr, dc in _DIRECTIONS
                                   for rr, cc in [(q//side+dr, q%side+dc)]
                                   if 0 <= rr < side and 0 <= cc < side):
                            return
            out = _render_placement(analysis, placement)
            signature = _key(out)
            if signature not in answers:
                answers.add(signature)
                solutions.append({"placement": dict(placement), "output": out})
            return
        for piece in remaining:
            for candidate in table[piece, corner]:
                if all(_compatible(candidate, p, use_ink) for p in placement.values()):
                    placement[piece] = candidate
                    search(corner+1, tuple(i for i in remaining if i != piece),
                           occupied | candidate["footprint"])
                    del placement[piece]
                    if len(solutions) >= limit:
                        return
    search(1, (1, 2, 3), anchor["footprint"])
    return solutions


def _analyze(input_grid):
    a = _parse(input_grid)
    solutions = _pack(a)
    _require(len(solutions) == 1, "need one colored solution to the joint stamp/packing constraints")
    a.update(solutions[0])
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from pixels and the fixed family conventions alone."""
    return _analyze(input_grid)["output"].copy()


def _owner_valid(owner):
    n = len(owner)
    if tuple(int(owner[p]) for p in ((0, 0), (0, n-1), (n-1, 0), (n-1, n-1))) != (0, 1, 2, 3):
        return False
    if not any(np.all(owner[r:r+2, c:c+2] == 0)
               for r in range(n-1) for c in range(n-1)):
        return False
    return all(np.count_nonzero(owner == i) >= 8 and
               len(_groups(map(tuple, np.argwhere(owner == i)))) == 1 for i in range(4))


def _partition(rng, side, kind):
    owner = np.full((side, side), -1, dtype=np.int8)
    if kind == 0:
        # Two wandering vertical cuts and independent top/bottom heights.
        split = int(rng.integers(2, side-2))
        top_cut = int(rng.integers(2, side-1))
        bottom_cut = int(rng.integers(1, side-1))
        for r in range(side):
            cut = top_cut if r < split else bottom_cut
            if r not in (0, split) and rng.random() < .7:
                cut = int(np.clip(cut + rng.integers(-1, 2), 1, side-1))
            if r < split:
                top_cut = cut
                owner[r, :cut], owner[r, cut:] = 0, 1
            else:
                bottom_cut = cut
                owner[r, :cut], owner[r, cut:] = 2, 3
        # Exchanges produce tabs across the horizontal seam too.
        for _ in range(side*2):
            r, c = int(rng.integers(1, side-1)), int(rng.integers(1, side-1))
            old = int(owner[r, c])
            dr, dc = _DIRECTIONS[int(rng.integers(4))]
            owner[r, c] = owner[r+dr, c+dc]
            if not _owner_valid(owner):
                owner[r, c] = old
    else:
        # Four connected growth fronts with independent, strongly unequal rates.
        weights = rng.dirichlet(np.full(4, .9 if kind == 1 else .35))
        fronts = [set() for _ in range(4)]
        starts = ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))
        for i, p in enumerate(starts):
            owner[p] = i
        def add_front(i, p):
            for dr, dc in _DIRECTIONS:
                q = p[0]+dr, p[1]+dc
                if 0 <= q[0] < side and 0 <= q[1] < side and owner[q] < 0:
                    fronts[i].add(q)
        for i, p in enumerate(starts):
            add_front(i, p)
        for _ in range(side*side-4):
            for front in fronts:
                front.difference_update(p for p in list(front) if owner[p] >= 0)
            choices = [i for i in range(4) if fronts[i]]
            probabilities = weights[choices]/weights[choices].sum()
            i = int(rng.choice(choices, p=probabilities))
            points = sorted(fronts[i])
            p = points[int(rng.integers(len(points)))]
            owner[p] = i
            add_front(i, p)
    return owner


def _separated(a, b):
    return all(max(abs(r-u), abs(c-v)) >= 2 for r, c in a for u, v in b)


def _stamp_options(owner, forbidden, rng, grow=True):
    n = len(owner)
    options = []
    for r in range(1, n-1):
        for c in range(1, n-1):
            for dr, dc in ((1, 0), (0, 1)):
                q = r+dr, c+dc
                if q[0] >= n-1 or q[1] >= n-1 or owner[r, c] == owner[q]:
                    continue
                labels = frozenset((int(owner[r, c]), int(owner[q])))
                cells = {(r, c), q}
                # Grow each contact independently: dominoes, strips, elbows,
                # rectangles, and asymmetric patches can share one example.
                target = int(rng.choice([2, 2, 3, 4, 5, 6, 8])) if grow else 2
                for _ in range(target-2):
                    frontier = sorted({(u+dy, v+dx) for u, v in cells for dy, dx in _DIRECTIONS
                                       if 1 <= u+dy < n-1 and 1 <= v+dx < n-1
                                       and int(owner[u+dy, v+dx]) in labels} - cells)
                    frontier = [p for p in frontier if _separated({p}, forbidden)]
                    if not frontier:
                        break
                    cells.add(frontier[int(rng.integers(len(frontier)))])
                if _separated(cells, forbidden):
                    options.append((labels, cells))
    return options


def _make_stamps(owner, rng, target, symmetric=False, repeat=False):
    n = len(owner)
    if symmetric:
        half = n//2
        stamps = [{(1, half-1), (1, half)}, {(n-2, half-1), (n-2, half)},
                  {(half-1, 1), (half, 1)}, {(half-1, n-2), (half, n-2)}]
        return (0, 0), stamps
    locations = [(r, c) for r in range(n-1) for c in range(n-1)
                 if np.all(owner[r:r+2, c:c+2] == 0)]
    rng.shuffle(locations)
    for r, c in locations[:18]:
        legend = {(r+i, c+j) for i in range(2) for j in range(2)}
        stamps, forbidden, reached = [], set(legend), {0}
        while len(reached) < 4:
            options = [(labels, cells) for labels, cells in _stamp_options(owner, forbidden, rng, not repeat)
                       if bool(labels & reached) and not labels <= reached]
            if not options:
                break
            labels, cells = options[int(rng.integers(len(options)))]
            stamps.append(cells)
            forbidden |= cells
            reached |= set(labels)
        if len(reached) != 4:
            continue
        if repeat:
            signatures = {_stamp_signature(owner, cells) for cells in stamps}
            options = [(labels, cells) for labels, cells in _stamp_options(owner, forbidden, rng, False)
                       if _stamp_signature(owner, cells) in signatures]
            if not options:
                continue
            labels, cells = options[int(rng.integers(len(options)))]
            stamps.append(cells)
            forbidden |= cells
        while len(stamps) < target:
            options = _stamp_options(owner, forbidden, rng)
            if not options:
                break
            labels, cells = options[int(rng.integers(len(options)))]
            stamps.append(cells)
            forbidden |= cells
        if len(stamps) == target:
            return (r, c), stamps
    return None


def _stamp_signature(owner, cells):
    # Same-orientation signatures suffice for the deliberately repeated join.
    return tuple(sorted(_normalize({p: int(owner[p]) for p in cells}).items()))


def _scatter(parts, background, rng):
    parts = [_normalize(p) for p in parts]
    dims = [_dimensions(p) for p in parts]
    area = sum(map(len, parts))
    # Free scattering is independent of solved corners and allows nested boxes.
    for _ in range(18):
        low_h, low_w = max(h for h, w in dims), max(w for h, w in dims)
        h = int(rng.integers(max(low_h, 12), 31))
        w = int(rng.integers(max(low_w, 12), 31))
        if h*w <= 2*area:
            continue
        grid = np.full((h, w), background, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        success = True
        for i in rng.permutation(4):
            ph, pw = dims[i]
            points = np.array(sorted(parts[i]), dtype=int)
            for _ in range(70):
                r0, c0 = int(rng.integers(h-ph+1)), int(rng.integers(w-pw+1))
                rr, cc = points[:, 0]+r0, points[:, 1]+c0
                if blocked[rr, cc].any():
                    continue
                for (r, c), value in parts[i].items():
                    r, c = r+r0, c+c0
                    grid[r, c] = value
                    blocked[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
                break
            else:
                success = False
                break
        if success:
            return grid
    # A shuffled two-shelf construction is guaranteed for side <= 12.
    order = list(map(int, rng.permutation(4)))
    top, bottom = order[:2], order[2:]
    heights = [max(dims[i][0] for i in group) for group in (top, bottom)]
    widths = [sum(dims[i][1] for i in group)+1 for group in (top, bottom)]
    h, w = sum(heights)+1, max(widths)
    # Add random margins, then enough empty space for a majority background.
    h = min(30, h+int(rng.integers(2, 6)))
    w = min(30, w+int(rng.integers(2, 6)))
    while h*w <= 2*area:
        if h <= w and h < 30:
            h += 1
        else:
            w += 1
        _require(h <= 30 and w <= 30, "scattering capacity exceeded")
    grid = np.full((h, w), background, dtype=np.int8)
    row = int(rng.integers(h-(sum(heights)+1)+1))
    for group, height, width in zip((top, bottom), heights, widths):
        col = int(rng.integers(w-width+1))
        for i in group:
            rr = row+int(rng.integers(height-dims[i][0]+1))
            for (r, c), value in parts[i].items():
                grid[r+rr, c+col] = value
            col += dims[i][1]+1
        row += height+1
    return grid


def _construct_pair(rng, side, target_stamps, key_colors, turns, symmetric=False, repeat=False):
    for attempt in range(140):
        if symmetric:
            half = side//2
            owner = np.empty((side, side), dtype=np.int8)
            owner[:half, :half], owner[:half, half:] = 0, 1
            owner[half:, :half], owner[half:, half:] = 2, 3
        else:
            owner = _partition(rng, side, int(rng.integers(3)))
            if not _owner_valid(owner):
                continue
        chosen = _make_stamps(owner, rng, target_stamps, symmetric, repeat)
        if chosen is None:
            continue
        (kr, kc), stamps = chosen
        palette = [int(c) for c in range(10) if c not in key_colors]
        background = int(rng.choice(palette))
        available = [c for c in palette if c != background]
        count = int(rng.integers(1, 5))
        carriers = list(map(int, rng.choice(available, count, replace=False)))
        carriers += [int(rng.choice(carriers)) for _ in range(4-count)]
        rng.shuffle(carriers)
        stamp_cells = set().union(*stamps)
        parts = []
        for i in range(4):
            part = {tuple(map(int, p)): carriers[i] for p in np.argwhere(owner == i)
                    if tuple(p) not in stamp_cells}
            for stamp in stamps:
                labels = {int(owner[p]) for p in stamp}
                if i in labels:
                    part.update({p: int(key_colors[int(owner[p])]) for p in stamp})
            if i == 0:
                for j in range(4):
                    part[kr+j//2, kc+j%2] = int(key_colors[j])
            parts.append(_rotate(_normalize(part), turns[i]))
        inp = _scatter(parts, background, rng)
        expected = np.asarray(key_colors, dtype=np.int8)[owner]
        try:
            analysis = _analyze(inp)
        except AssertionError:
            continue
        if np.array_equal(analysis["output"], expected):
            if symmetric:
                blind = _pack(analysis, use_ink=False, limit=3)
                if not any(not np.array_equal(s["output"], expected) for s in blind):
                    continue
            return {"input": inp, "output": expected}
    raise RuntimeError("bounded construction did not find a valid shared-stamp square")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    growth = bool(rng.integers(2))
    puzzle, seen, previous_key = {"train": [], "test": []}, set(), None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            symmetric = split == "train" and i == 0
            if symmetric:
                side = 8 if growth else int(rng.choice([8, 10]))
                stamps, turns = 4, [0, 0, 0, 0]
            elif split == "train":
                side = int(rng.integers(8, 10 if growth else 12))
                stamps = 5 if i == 1 else int(rng.integers(3, 6))
                turns = [0, 1, 2, 3] if i == 1 else [0]+list(map(int, rng.integers(0, 4, 3)))
            else:
                side = int(rng.integers(10 if growth else 8, 13))
                stamps = int(rng.integers(6, 8))
                side = max(side, stamps+3)
                turns = [0]+list(map(int, rng.integers(0, 4, 3)))
            for _ in range(12):
                key_colors = list(map(int, rng.choice(10, 4, replace=False)))
                if split == "train" and i == 1 and key_colors == previous_key:
                    key_colors = key_colors[1:]+key_colors[:1]
                try:
                    pair = _construct_pair(rng, side, stamps, key_colors, turns, symmetric,
                                           repeat=split == "train" and i == 1)
                except RuntimeError:
                    # Retain the contact-count shift while easing scarce space.
                    side = min(12, side+1)
                    continue
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    previous_key = key_colors
                    break
            else:
                raise RuntimeError("could not make a distinct, unambiguous example")
    return puzzle


def _measure(analysis):
    placements = analysis["placement"]
    stamps = {}
    for piece, cand in placements.items():
        for stamp in cand["stamps"]:
            stamps.setdefault(stamp, []).append(piece)
    edges = Counter(tuple(sorted(v)) for v in stamps.values())
    footprints = [p["material"] for p in placements.values()]
    ordered_stamps = sorted(stamps)
    signatures = [tuple(sorted(_normalize({(q//analysis["side"], q%analysis["side"]): v
                                          for q, v in s}).items())) for s in ordered_stamps]
    return {"side": analysis["side"], "stamps": len(stamps),
            "stamp_sizes": tuple(len(s) for s in ordered_stamps),
            "stamp_shapes": tuple((max(q//analysis["side"] for q, v in s)-min(q//analysis["side"] for q, v in s)+1,
                                   max(q%analysis["side"] for q, v in s)-min(q%analysis["side"] for q, v in s)+1) for s in ordered_stamps),
            "repeated_signature": len(set(signatures)) < len(signatures),
            "contacts": len(edges), "repeated_contact": max(edges.values()) > 1,
            "areas": tuple(mask.bit_count() for mask in footprints),
            "carriers": len({p["carrier"] for p in analysis["parts"]}),
            "turns": tuple(placements[i]["turns"] for i in range(4)),
            "background": analysis["background"]}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(out, a["output"]), "incorrect keyed contact-map output")
            _require(set(map(int, np.unique(out))) == set(map(int, a["key"].flat)), "output must use the four key colors")
            placement = a["placement"]
            # Local stamp colors must describe the two corner-colored materials
            # to which that contact map belongs, rather than decorative ink.
            for cand in placement.values():
                for stamp in cand["stamps"]:
                    partners = [p for p in placement.values() if stamp in p["stamps"]]
                    expected_colors = {int(a["key"].flat[p["corner"]]) for p in partners}
                    _require({v for q, v in stamp} == expected_colors, "stamp does not label its two contacting pieces")
                    for q, color in stamp:
                        matching = next(p for p in partners if int(a["key"].flat[p["corner"]]) == color)
                        same_ink = {qq for qq, vv in stamp if vv == color}
                        _require(any((matching["material"] & (1 << (rr*a["side"]+cc)))
                                     for qq in same_ink for dr, dc in _DIRECTIONS
                                     for rr, cc in [(qq//a["side"]+dr, qq%a["side"]+dc)]
                                     if 0 <= rr < a["side"] and 0 <= cc < a["side"]),
                                 "stamp color must touch the material it labels")
            analyses[split].append(a)
    # The smallest legal training set already teaches the key's variability,
    # all quarter turns, and an ambiguity that shape-only packing cannot resolve.
    _require(not np.array_equal(analyses["train"][0]["key"], analyses["train"][1]["key"]), "key variability not demonstrated")
    _require(any(any(not np.array_equal(s["output"], a["output"]) for s in _pack(a, False, 3))
                 for a in analyses["train"]), "training must defeat ink-blind geometric packing")
    _require({p["turns"] for a in analyses["train"] for p in a["placement"].values()} == {0, 1, 2, 3},
             "training must show every quarter turn")
    _require(any(_measure(a)["repeated_signature"] for a in analyses["train"]),
             "training must demonstrate distinct contacts with identical stamps")


def self_test():
    """Bounded pixel checks and observed structural coverage over 42 seeds."""
    records, puzzles, input_shapes, normalized_outputs = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                m = _measure(a)
                m["split"] = split
                records.append(m)
                input_shapes.add(pair["input"].shape)
                key = list(map(int, a["key"].flat))
                normalized = np.array([[key.index(int(v)) for v in row] for row in pair["output"]], dtype=np.int8)
                normalized_outputs.add(_key(normalized))
        # Input position and carrier colors are semantically irrelevant.
        a = _analyze(puzzle["test"][0]["input"])
        parts = [p["cells"] for p in a["parts"]]
        repacked = _scatter(parts, a["background"], np.random.default_rng(1000+seed))
        _require(np.array_equal(solve(repacked), a["output"]), "scatter changed answer")
    _require(len(set(puzzles)) == 42, "seeds collapsed")
    _require(len(input_shapes) >= 35 and len(normalized_outputs) >= 100, "structural variation collapsed")
    _require({r["background"] for r in records} == set(range(10)), "background roles lack variation")
    _require({r["carriers"] for r in records} == {1, 2, 3, 4}, "shared/distinct carriers missing")
    _require({r["stamps"] for r in records} >= {3, 4, 5, 6, 7}, "contact count variation missing")
    _require(len({r["areas"] for r in records}) >= 100, "material geometry variation missing")
    _require(any(r["repeated_contact"] for r in records), "multiple registrations of one contact missing")
    _require(any(h == 1 or w == 1 for r in records for h, w in r["stamp_shapes"]), "thin signatures missing")
    _require(any(h > 1 and w > 1 for r in records for h, w in r["stamp_shapes"]), "2D signatures missing")
    _require(any(n < h*w for r in records for n, (h, w) in zip(r["stamp_sizes"], r["stamp_shapes"])), "irregular signatures missing")
    _require(any(max(r["areas"])/min(r["areas"]) >= 3 for r in records), "unequal pieces missing")
    for split in ("train", "test"):
        subset = [r for r in records if r["split"] == split]
        _require({turn for r in subset for turn in r["turns"]} == {0, 1, 2, 3}, "rotation coverage missing")
    a, b = generate_puzzle(123, 3, 1), generate_puzzle(123, 3, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p), "nondeterminism")
    for nt, nq in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(10000+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return {"seeds": 42, "pairs": len(records), "input_shapes": len(input_shapes),
            "normalized_outputs": len(normalized_outputs), "stamp_counts": sorted({r["stamps"] for r in records})}


if __name__ == "__main__":
    print(self_test())
