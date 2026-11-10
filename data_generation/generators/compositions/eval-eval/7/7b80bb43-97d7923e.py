"""Repair-gated route selection on a shared orthogonal tree.

Only Python/NumPy are required. Generation keeps one puzzle at a time in memory;
no samples, logs, caches, execution traces, or temporary files are written here.
"""
from collections import deque
import hashlib
import json

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(((0 <= a) & (a <= 9)).all()), "invalid colors")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in STEPS
            if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]]


def _at(mask, p):
    return 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and bool(mask[p])


def _cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _adjacent(mask):
    result = np.zeros_like(mask)
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def _halo(mask):
    a = np.pad(mask, 1)
    return np.logical_or.reduce([a[r:r + mask.shape[0], c:c + mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        a = mask if dc else mask.T
        m = a[:, :-4] & a[:, 4:] & ~(a[:, 1:-3] | a[:, 2:-2] | a[:, 3:-1])
        for y, x in np.argwhere(m):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _repair(mask, protected, fillable=None):
    """One simultaneous, input-only pass: exact two-cell isolated deviations.

    Protected cells are wire cells beside a root or terminal marker. Bracket
    anchors are also retained. Uncued gaps are never closed by connectivity.
    """
    isolated = mask & ~_adjacent(mask) & ~protected
    repairs, anchors = [], np.zeros_like(mask)
    for r, c, dr, dc in _gaps(mask):
        if fillable is not None and not all(fillable[r + k * dr, c + k * dc] for k in (1, 2, 3)):
            continue
        if not (_at(mask, (r - dr, c - dc)) or _at(mask, (r + 5 * dr, c + 5 * dc))):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint * dr, c + endpoint * dc
            direction = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                vr, vc = direction * dr + side * dc, direction * dc + side * dr
                spur = tuple((ar + k * vr, ac + k * vc) for k in (1, 2))
                if all(_at(isolated, p) for p in spur) and not _at(mask, (ar + 3 * vr, ac + 3 * vc)):
                    matches.append(spur)
        if matches:
            _require(len(matches) == 1, "ambiguous repair cue")
            gap = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
            repairs.append((gap, matches[0]))
            anchors[r, c] = anchors[r + 4 * dr, c + 4 * dc] = True
    diagonal_neighbor = np.zeros_like(mask)
    diagonal_neighbor[1:, 1:] |= mask[:-1, :-1]
    diagonal_neighbor[:-1, :-1] |= mask[1:, 1:]
    diagonal_neighbor[1:, :-1] |= mask[:-1, 1:]
    diagonal_neighbor[:-1, 1:] |= mask[1:, :-1]
    removed = isolated & diagonal_neighbor & ~anchors
    clean = mask & ~removed
    used = set()
    for gap, spur in repairs:
        footprint = set(gap + spur)
        _require(not used.intersection(footprint), "overlapping repairs")
        used.update(footprint)
        for p in gap:
            clean[p] = True
        for p in spur:
            _require(removed[p], "deviation not removable")
    return clean, repairs, removed


def _roles(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.sum(counts == counts[bg]) == 1, "background must be uniquely modal")
    squares = []
    for color in np.flatnonzero(counts):
        if color == bg:
            continue
        m = a == color
        if (m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]).any():
            squares.append(int(color))
    _require(len(squares) == 1 and counts[squares[0]] == 4, "source must be one monochrome 2x2")
    active = squares[0]
    remaining = [int(c) for c in np.flatnonzero(counts) if c not in (bg, active)]
    _require(len(remaining) >= 2, "missing wire or family")
    wire = max(remaining, key=lambda c: counts[c])
    _require(all(counts[wire] > counts[c] for c in remaining if c != wire), "wire role must be unique")
    families = sorted(c for c in remaining if c != wire)
    _require(1 <= len(families) <= 3, "one to three terminal families required")
    return bg, wire, active, families


def _walk(mask, source):
    distance, previous = {source: 1}, {source: None}
    queue = deque([source])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, mask.shape):
            if mask[q] and q not in distance:
                distance[q] = distance[p] + 1
                previous[q] = p
                queue.append(q)
    return distance, previous


def _path(previous, p):
    result = []
    while p is not None:
        result.append(p)
        p = previous[p]
    return tuple(reversed(result))


def _routes(mask, source, terminals, ranks, metric="wire"):
    distances, previous = _walk(mask, source)
    grouped = {c: [] for c in ranks}
    for cap, p in terminals:
        adjacent = [q for q in _neighbors(p, mask.shape) if mask[q]]
        _require(len(adjacent) == 1, "terminal needs exactly one wire neighbor")
        q = adjacent[0]
        if q in distances:
            length = distances[q]
            key = length if metric == "wire" else abs(q[0] - source[0]) + abs(q[1] - source[1]) + 1
            grouped[cap].append({"terminal": p, "length": length, "metric": key,
                                 "path": _path(previous, q)})
    selected = []
    for cap, members in grouped.items():
        _require(len(members) >= ranks[cap], "requested reachable rank does not exist")
        _require(len({m["length"] for m in members}) == len(members), "tied route lengths in family")
        ordered = sorted(members, key=lambda m: (-m["metric"], m["terminal"]))
        selected.append((cap, ordered[ranks[cap] - 1]))
    return grouped, selected


def _decode(a):
    bg, wire, active, families = _roles(a)
    raw, root = a == wire, a == active
    entry = _cells(raw & _adjacent(root))
    _require(len(entry) == 1, "source needs exactly one wire connection")
    source = next(iter(entry))
    ranks, terminals, keys = {}, [], {}
    wire_adjacent = _adjacent(raw)
    for cap in families:
        end_cells = _cells((a == cap) & wire_adjacent)
        key_cells = _cells((a == cap) & ~wire_adjacent)
        _require(1 <= len(key_cells) <= 3, "unary key must have one to three cells")
        rs, cs = {p[0] for p in key_cells}, {p[1] for p in key_cells}
        _require(len(rs) == 1 or len(cs) == 1, "unary key must be straight")
        axis = sorted(p[1] if len(rs) == 1 else p[0] for p in key_cells)
        _require(axis == list(range(axis[0], axis[0] + len(axis))), "unary key must be contiguous")
        root_links = sum(sum(root[q] for q in _neighbors(p, a.shape)) for p in key_cells)
        _require(root_links == 1, "unary key must attach to the source at one end")
        _require(len(end_cells) >= 2, "each color needs competing terminals")
        keys[cap] = key_cells
        ranks[cap] = len(key_cells)
        terminals.extend((cap, p) for p in sorted(end_cells))
    protected = raw & _adjacent((a != bg) & ~raw)
    clean, repairs, removed = _repair(raw, protected, a == bg)
    _require(not (clean[:-1, :-1] & clean[1:, :-1] & clean[:-1, 1:] & clean[1:, 1:]).any(), "thick wire")
    # Every component of the wire forest is acyclic. The source square is not
    # included in this graph; it identifies the single entry point only.
    unseen = _cells(clean)
    while unseen:
        component, _ = _walk(clean, min(unseen))
        vertices = set(component)
        edges = sum(len([q for q in _neighbors(p, a.shape) if clean[q]]) for p in vertices) // 2
        _require(edges == len(vertices) - 1, "wire graph must be a forest")
        unseen.difference_update(vertices)
    grouped, selected = _routes(clean, source, terminals, ranks)
    _require(all(len(v) >= 2 for v in grouped.values()), "each family needs two reachable competitors")
    for _, terminal in terminals:
        q = next(q for q in _neighbors(terminal, a.shape) if clean[q])
        _require(sum(clean[p] for p in _neighbors(q, a.shape)) <= 1, "terminal must end a twig")
    return {"bg": bg, "wire": wire, "active": active, "families": families,
            "raw": raw, "clean": clean, "source": source, "terminals": terminals,
            "keys": keys, "ranks": ranks, "repairs": repairs, "removed": removed,
            "grouped": grouped, "selected": selected}


def _render(a, info, selected=None):
    out = a.copy()
    out[info["raw"] & ~info["clean"]] = info["bg"]
    out[info["clean"]] = info["wire"]
    for _, chosen in info["selected"] if selected is None else selected:
        for p in chosen["path"]:
            out[p] = info["active"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, repairs, reachability, path lengths, and unary ranks from pixels."""
    a = _grid(input_grid)
    return _render(a, _decode(a))


def _tree(rng, nr, nc):
    edges = [((r, c), (r + dr, c + dc)) for r in range(nr) for c in range(nc)
             for dr, dc in ((1, 0), (0, 1)) if r + dr < nr and c + dc < nc]
    parent = {(r, c): (r, c) for r in range(nr) for c in range(nc)}

    def find(p):
        while parent[p] != p:
            p = parent[p]
        return p

    selected = []
    for i in rng.permutation(len(edges)):
        a, b = edges[int(i)]
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            selected.append((a, b))
    return selected


def _segment(a, b):
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    length = abs(b[0] - a[0]) + abs(b[1] - a[1])
    return tuple((a[0] + k * dr, a[1] + k * dc) for k in range(length + 1))


def _geometry(rng):
    sr, sc = int(rng.integers(4, 6)), int(rng.integers(4, 6))
    nr, nc = int(rng.integers(4, 7 if sr == 4 else 6)), int(rng.integers(4, 7 if sc == 4 else 6))
    h, w = 5 + sr * (nr - 1) + 4, 7 + sc * (nc - 1) + 3
    mask = np.zeros((h, w), bool)
    origin = lambda p: (5 + sr * p[0], 7 + sc * p[1])
    coarse = _tree(rng, nr, nc)
    degrees = {(r, c): 0 for r in range(nr) for c in range(nc)}
    edges = []
    for a, b in coarse:
        degrees[a] += 1
        degrees[b] += 1
        cells = _segment(origin(a), origin(b))
        edges.append(cells)
        for p in cells:
            mask[p] = True
    mask[5, 5:8] = True
    root = np.zeros_like(mask)
    root[4:6, 3:5] = True
    leaves = [p for p, degree in degrees.items() if degree == 1 and p != (0, 0)]
    _require(len(leaves) >= 6, "not enough leaf routes")
    rng.shuffle(leaves)
    terminals = []
    occupied = mask | root
    for leaf in leaves:
        a = origin(leaf)
        options = [(dr, dc, n) for dr, dc in STEPS for n in (1, 2)]
        rng.shuffle(options)
        for dr, dc, n in options:
            cells = tuple((a[0] + k * dr, a[1] + k * dc) for k in range(1, n + 1))
            if not all(0 <= p[0] < h and 0 <= p[1] < w and not occupied[p] for p in cells):
                continue
            if any(any(occupied[q] and q != a for q in _neighbors(p, mask.shape)) for p in cells):
                continue
            for p in cells[:-1]:
                mask[p] = occupied[p] = True
            occupied[cells[-1]] = True
            edge = next(e for e in edges if a in (e[0], e[-1]))
            terminals.append({"point": cells[-1], "edge": edge})
            break
        else:
            raise ValueError("cannot place terminal")
    return mask, root, edges, terminals


def _assign(rng, mask, terminals, family_count, ranks):
    dist, _ = _walk(mask, (5, 5))
    lengths = []
    for terminal in terminals:
        q = next(q for q in _neighbors(terminal["point"], mask.shape) if mask[q])
        lengths.append(dist[q])
    # Disconnect a longest leaf, then give it a color whose valid selected route
    # is shorter. Filling every gap would therefore alter an ordinal answer.
    maximum = max(lengths)
    blocked = int(rng.choice([i for i, length in enumerate(lengths) if length == maximum]))
    available = [i for i in range(len(terminals)) if i != blocked]
    minimum = [max(2, rank) for rank in ranks]
    if ranks[0] == 2:
        minimum[0] = 3
    _require(sum(minimum) <= len(available), "not enough competitors for this scene")
    for _ in range(64):
        order = list(available)
        rng.shuffle(order)
        assignment, cursor = {}, 0
        for family, count in enumerate(minimum):
            for i in order[cursor:cursor + count]:
                assignment[i] = family
            cursor += count
        for i in order[cursor:]:
            assignment[i] = int(rng.integers(family_count))
        if any(len({lengths[i] for i, f in assignment.items() if f == family}) !=
               sum(f == family for f in assignment.values()) for family in range(family_count)):
            continue
        possible = [f for f in range(family_count)
                    if all(lengths[i] < maximum for i, g in assignment.items() if g == f)]
        if not possible:
            continue
        assignment[blocked] = int(rng.choice(possible))
        return assignment, blocked
    raise ValueError("cannot assign untied color families")


def _damage_options(mask, root, marker_mask, edges, selected_paths, shared_only):
    protected = mask & _adjacent(root | marker_mask)
    options = []
    base_blocked = _halo(root | marker_mask)
    for edge in [e[start:start + 5] for e in edges for start in range(len(e) - 4)]:
        if not all(mask[p] for p in edge):
            continue
        gap = edge[1:4]
        if not any(set(gap).intersection(path) for path in selected_paths):
            continue
        # A shared-edge shift must affect at least two selected family routes.
        sharing = sum(bool(set(gap).intersection(path)) for path in selected_paths)
        if shared_only and sharing < 2:
            continue
        if any(base_blocked[p] for p in gap):
            continue
        for anchor, toward in ((edge[0], edge[-1]), (edge[-1], edge[0])):
            dr = int(np.sign(toward[0] - anchor[0]))
            dc = int(np.sign(toward[1] - anchor[1]))
            for side in (-1, 1):
                spur = tuple((anchor[0] + k * (dr + side * dc),
                              anchor[1] + k * (dc + side * dr)) for k in (1, 2))
                if not all(0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1]
                           and not mask[p] and not base_blocked[p] for p in spur):
                    continue
                trial = mask.copy()
                for p in gap:
                    trial[p] = False
                for p in spur:
                    trial[p] = True
                try:
                    recovered, evidence, _ = _repair(trial, protected)
                except ValueError:
                    continue
                if len(evidence) == 1 and np.array_equal(recovered, mask):
                    options.append((gap, spur, sharing))
    return options


def _selection_ids(selected):
    return {chosen["terminal"] for _, chosen in selected}


def _metric_contrast(info):
    _, alternative = _routes(info["clean"], info["source"], info["terminals"], info["ranks"], "manhattan")
    # A strict contrast, without a tie-breaking explanation for the shortcut.
    for cap, chosen in info["selected"]:
        members = info["grouped"][cap]
        metrics = [abs(m["path"][-1][0] - info["source"][0]) +
                   abs(m["path"][-1][1] - info["source"][1]) + 1 for m in members]
        alt = next(m for c, m in alternative if c == cap)
        if len(set(metrics)) == len(metrics) and alt["terminal"] != chosen["terminal"]:
            return True
    return False


def _make_pair(rng, palette, family_count, ranks, rotation, shared, contrast, single_repair=False):
    for _ in range(256):
        try:
            full, root, edges, terminals = _geometry(rng)
            assignment, blocked = _assign(rng, full, terminals, family_count, ranks)
            bg, wire, active = map(int, palette[:3])
            caps = list(map(int, palette[3:3 + family_count]))
            marker_mask = np.zeros_like(full)
            a = np.full(full.shape, bg, dtype=np.int8)
            a[full], a[root] = wire, active
            visible_terminals = []
            for i, terminal in enumerate(terminals):
                point, cap = terminal["point"], caps[assignment[i]]
                a[point] = cap
                marker_mask[point] = True
                visible_terminals.append((cap, point))
            # Keys are unary arms attached directly to the source square.
            arms = [((4, 3), (-1, 0)), ((4, 3), (0, -1)), ((5, 3), (1, 0))]
            rng.shuffle(arms)
            for cap, rank, (anchor, step) in zip(caps, ranks, arms):
                for k in range(1, rank + 1):
                    p = (anchor[0] + k * step[0], anchor[1] + k * step[1])
                    a[p], marker_mask[p] = cap, True
            clean = full.copy()
            permanent_gap = terminals[blocked]["edge"][1:4]
            for p in permanent_gap:
                clean[p], a[p] = False, bg
            rank_map = dict(zip(caps, ranks))
            grouped, selected = _routes(clean, (5, 5), visible_terminals, rank_map)
            selected_paths = [set(chosen["path"]) for _, chosen in selected]
            options = _damage_options(clean, root, marker_mask, edges, selected_paths, shared)
            _require(bool(options), "no eligible route repair")
            first = options[int(rng.integers(len(options)))]
            raw = clean.copy()
            damage = [first]
            for p in first[0]:
                raw[p] = False
            for p in first[1]:
                raw[p] = True
            protected = clean & _adjacent(root | marker_mask)
            # Add up to three more compatible repairs. Every accepted damaged
            # scene must reconstruct exactly the independently built forest.
            target = 1 if single_repair else int(rng.integers(2 if shared else 1, 5))
            for index in rng.permutation(len(options)):
                if len(damage) >= target:
                    break
                gap, spur, sharing = options[int(index)]
                old = {p for g, s, _ in damage for p in g + s}
                if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1
                       for p in gap + spur for q in old):
                    continue
                trial = raw.copy()
                for p in gap:
                    trial[p] = False
                for p in spur:
                    trial[p] = True
                try:
                    recovered, evidence, _ = _repair(trial, protected)
                except ValueError:
                    continue
                if len(evidence) == len(damage) + 1 and np.array_equal(recovered, clean):
                    raw = trial
                    damage.append((gap, spur, sharing))
            if shared:
                _require(len(damage) >= 2, "shared shift needs a combined repair set")
            # Optional free diagonal clutter. It has no orthogonal contact and
            # cannot create an extra repair, delete a bracket, or touch a marker.
            for _ in range(int(rng.integers(3))):
                for _ in range(20):
                    r, c = int(rng.integers(a.shape[0])), int(rng.integers(a.shape[1]))
                    dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
                    cells = tuple((r + k * dr, c + k * dc) for k in range(int(rng.integers(2, 4))))
                    forbidden = _halo(raw | root | marker_mask | clean)
                    if not all(0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]
                               and not forbidden[p] for p in cells):
                        continue
                    trial = raw.copy()
                    for p in cells:
                        trial[p] = True
                    recovered, evidence, _ = _repair(trial, protected)
                    if len(evidence) == len(damage) and np.array_equal(recovered, clean):
                        raw = trial
                        break
            a[full] = bg
            a[raw] = wire
            info = _decode(a)
            _require(np.array_equal(info["clean"], clean), "reconstruction differs from construction")
            _require(_selection_ids(info["selected"]) == _selection_ids(selected), "route mismatch")
            if contrast:
                _require(_metric_contrast(info), "needs a bent-path length contrast")
            out = a.copy()
            out[raw & ~clean] = bg
            out[clean] = wire
            for _, chosen in selected:
                for p in chosen["path"]:
                    out[p] = active
            if bool(rng.integers(2)):
                a, out = a[:, ::-1], out[:, ::-1]
            # Small translations change margins without exceeding 30.
            dh, dw = int(rng.integers(31 - a.shape[0])), int(rng.integers(31 - a.shape[1]))
            top, left = int(rng.integers(dh + 1)), int(rng.integers(dw + 1))
            pad = ((top, dh - top), (left, dw - left))
            a, out = np.pad(a, pad, constant_values=bg), np.pad(out, pad, constant_values=bg)
            return {"input": np.ascontiguousarray(np.rot90(a, rotation)),
                    "output": np.ascontiguousarray(np.rot90(out, rotation))}
        except ValueError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    _require(n > t, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    varied_palette = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    shift = str(rng.choice(["orientation", "shared_break", "families"]))
    max_rank = int(rng.integers(2, 4))
    anchor_indices = list(map(int, rng.choice(n, 2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n), ("test", t)):
        for index in range(count):
            anchor = anchor_indices.index(index) + 1 if split == "train" and index in anchor_indices else None
            f = 2 if anchor else int(rng.integers(1, 4))
            if shift == "families":
                f = 3 if split == "test" else (2 if anchor else int(rng.integers(1, 3)))
            ranks = [int(rng.integers(1, max_rank + 1)) for _ in range(f)]
            if anchor:
                ranks[0] = anchor
                if max_rank == 3 and anchor == 1:
                    ranks[1] = 3
            pair_palette = palette if anchor or not varied_palette else rng.permutation(10)
            rotation = (base_rotation + (split == "test")) % 4 if shift == "orientation" else int(rng.integers(4))
            shared = shift == "shared_break" and split == "test"
            if shared:
                f = max(2, f)
                if len(ranks) < f:
                    ranks.append(int(rng.integers(1, max_rank + 1)))
            for _ in range(8):
                pair = _make_pair(rng, pair_palette, f, ranks, rotation, shared, anchor == 1,
                                  shift == "shared_break" and split == "train")
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    a, out = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == out.shape, "dimensions must be preserved")
    info = _decode(a)
    _require(np.array_equal(_render(a, info), out), "incorrect repair or route highlighting")
    _require(bool(info["repairs"]) and bool(info["removed"].any()), "missing A mechanism")
    _require(bool((out == info["active"]).sum() > 4), "missing B mechanism")
    _require(bool((info["raw"] & info["clean"] & (out == info["wire"])).any()), "no unselected wire")
    _require(len(info["terminals"]) > sum(map(len, info["grouped"].values())), "missing unreachable terminal")
    _require(any(set(gap).intersection(chosen["path"]) for gap, _ in info["repairs"]
                 for _, chosen in info["selected"]), "repairs do not affect selected routes")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n, t = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= t <= 4 and n > t, "invalid pair counts")
    seen, train, test = set(), [], []
    for split, results in (("train", train), ("test", test)):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            signature = _key(_grid(pair["input"]))
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            results.append(info)
    histories = {}
    for info in train:
        for cap, rank in info["ranks"].items():
            histories.setdefault(cap, set()).add(rank)
    _require(any({1, 2} <= values for values in histories.values()), "missing unary-rank teaching contrast")
    _require(any(len(x["families"]) >= 2 for x in train), "missing cap grouping demonstration")
    _require(any(1 < x["ranks"][cap] < len(members) for x in train for cap, members in x["grouped"].items()),
             "missing non-extreme rank demonstration")
    _require(any(_metric_contrast(x) for x in train), "Manhattan shortcut survives training")
    shown = {rank for x in train for rank in x["ranks"].values()}
    _require(all(rank in shown for x in test for rank in x["ranks"].values()), "test ordinal not demonstrated")


def self_test():
    """42-seed streaming sweep plus semantic/metamorphic and mutation checks."""
    shapes, structures, ranks, counts, repairs, signatures = set(), set(), set(), set(), set(), set()
    backgrounds, wire_colors, roots, reachable_lengths, turns = set(), set(), set(), set(), set()
    max_overlap, disconnection_changes, fill_all_changes = 0, 0, 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                info = _analyze_pair(pair)
                assert np.array_equal(solve(a), out)
                digest.update(a.tobytes())
                shapes.add(a.shape)
                backgrounds.add(info["bg"])
                wire_colors.add(info["wire"])
                roots.add(info["active"])
                ranks.update(info["ranks"].values())
                counts.add(len(info["families"]))
                repairs.add(len(info["repairs"]))
                roles = {info["bg"]: 0, info["wire"]: 1, info["active"]: 2}
                roles.update({c: 3 + i for i, c in enumerate(info["families"])})
                normalized = bytes(roles[int(v)] for v in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
                overlap = {}
                for _, chosen in info["selected"]:
                    path = chosen["path"]
                    for p in path:
                        overlap[p] = overlap.get(p, 0) + 1
                    directions = [(b[0] - a0[0], b[1] - a0[1]) for a0, b in zip(path, path[1:])]
                    turns.add(sum(x != y for x, y in zip(directions, directions[1:])))
                max_overlap = max(max_overlap, max(overlap.values()))
                for members in info["grouped"].values():
                    reachable_lengths.update(m["length"] for m in members)
                raw_clean = info["raw"] & ~info["removed"]
                try:
                    _, alt = _routes(raw_clean, info["source"], info["terminals"], info["ranks"])
                    disconnection_changes += _selection_ids(alt) != _selection_ids(info["selected"])
                except ValueError:
                    disconnection_changes += 1
                all_closed = info["clean"].copy()
                for r, c, dr, dc in _gaps(info["clean"]):
                    for k in (1, 2, 3):
                        all_closed[r + k * dr, c + k * dc] = True
                try:
                    _, alt = _routes(all_closed, info["source"], info["terminals"], info["ranks"])
                    fill_all_changes += _selection_ids(alt) != _selection_ids(info["selected"])
                except ValueError:
                    fill_all_changes += 1
        signatures.add(digest.digest())
        if seed == 0:
            first = puzzle
    assert len(signatures) == 42 and len(shapes) >= 30 and len(structures) >= 180
    assert counts == {1, 2, 3} and ranks == {1, 2, 3} and max(repairs) >= 3
    assert len(backgrounds) == len(wire_colors) == len(roots) == 10
    assert len(reachable_lengths) >= 30 and max(turns) >= 6 and max_overlap == 3
    assert disconnection_changes >= 100 and fill_all_changes >= 100
    # All eight rigid motions and an arbitrary bijective palette renaming.
    pair = first["train"][0]
    palette = np.array([8, 3, 9, 1, 6, 0, 7, 4, 2, 5], dtype=np.int8)
    for rotation in range(4):
        for reflect in (False, True):
            x, y = pair["input"], pair["output"]
            if reflect:
                x, y = x[:, ::-1], y[:, ::-1]
            x, y = palette[np.rot90(x, rotation)], palette[np.rot90(y, rotation)]
            assert np.array_equal(solve(x), y)
    # Full-output validation catches partial repairs, wrong highlights, and
    # leaving diagonal clutter even when other required edits are correct.
    for kind in ("repair", "highlight", "diagonal"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        p = bad["train"][0]
        info = _decode(p["input"])
        if kind == "repair":
            q = info["repairs"][0][0][0]
            p["output"][q] = info["bg"]
        elif kind == "highlight":
            q = info["selected"][0][1]["path"][0]
            p["output"][q] = info["wire"]
        else:
            q = next(iter(_cells(info["removed"])))
            p["output"][q] = info["wire"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("incorrect output accepted")
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(113, n, t)
            assert len(puzzle["train"]) == n and len(puzzle["test"]) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    same = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], same[s]) for k in a)
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "structures": len(structures),
            "families": sorted(counts), "ranks": sorted(ranks), "repair_counts": sorted(repairs),
            "route_lengths": len(reachable_lengths), "max_bends": max(turns),
            "max_selected_overlap": max_overlap, "omit_repairs_changes_selection": disconnection_changes,
            "close_all_gaps_changes_selection": fill_all_changes}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
