"""Switchyard cargo courier. Standard library and NumPy only.

The decoder reconstructs geometry and all semantic roles from a single input.
Generation metadata and parent generators are never used by solve().
"""
from __future__ import annotations

from itertools import product

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _facing(a, b):
    delta = (b[0] - a[0], b[1] - a[1])
    _require(delta in _DIRS, "path edges must be unit orthogonal steps")
    return _DIRS.index(delta)


def _body(mask, tip, facing):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, facing - 1) for p in mask)}


def _in_bounds(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _pointed(cells):
    """Find the symmetric transverse axis and its unique projecting tip."""
    possibilities = []
    for tip in sorted(cells):
        for facing in range(4):
            offsets = {_rotate((r - tip[0], c - tip[1]), 1 - facing)
                       for r, c in cells}
            if any(c > 0 for r, c in offsets):
                continue
            if {p for p in offsets if p[1] == 0} != {(0, 0)}:
                continue
            if max(abs(r) for r, c in offsets) == 0:
                continue
            if {(-r, c) for r, c in offsets} != offsets:
                continue
            possibilities.append((tuple(sorted(offsets)), tip, facing))
    _require(len(possibilities) == 1, "courier needs one geometrically unique tip")
    # Diagonal wings are allowed, but the whole mask is eight-connected.
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in product((-1, 0, 1), repeat=2):
            q = (r + dr, c + dc)
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    _require(seen == cells, "courier must be eight-connected")
    return possibilities[0]


def _paths(track_mask):
    """Return ordered simple components; reject branches, cycles, isolates."""
    cells = {tuple(map(int, p)) for p in np.argwhere(track_mask)}
    adj = {p: [q for dr, dc in _DIRS
               if (q := (p[0] + dr, p[1] + dc)) in cells] for p in cells}
    _require(all(len(ns) in (1, 2) for ns in adj.values()), "branch or isolated rail")
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        component, todo = {start}, [start]
        while todo:
            for q in adj[todo.pop()]:
                if q not in component:
                    component.add(q)
                    todo.append(q)
        ends = sorted(p for p in component if len(adj[p]) == 1)
        _require(len(ends) == 2, "rail component must have two ends, not a cycle")
        ordered, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            choices = [q for q in adj[current] if q != previous]
            _require(len(choices) == 1, "ambiguous path")
            previous, current = current, choices[0]
            ordered.append(current)
        _require(len(ordered) == len(component), "incomplete path")
        result.append(ordered)
        remaining -= component
    return sorted(result, key=lambda p: (p[0], p[-1]))


def _switch(base, centers, track, gate):
    """Simultaneous quarter-turn writes, including compatible tile overlap."""
    result, writes, overlap = base.copy(), {}, 0
    for r, c in centers:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                value = gate if cc == c else track
                if (rr, cc) in writes:
                    _require(writes[rr, cc] == value, "conflicting switch tiles")
                    overlap += 1
                writes[rr, cc] = value
    for p, value in writes.items():
        result[p] = value
    return result, writes, overlap


def _interpret_horizontal(a, bg, track):
    positions = np.argwhere(a == track)
    _require(len(positions) > 0, "missing rail")
    r0, c0 = map(int, positions.min(axis=0))
    r1, c1 = map(int, positions.max(axis=0))
    rows = sorted(set(map(int, positions[:, 0])))
    _require(len(rows) >= 2 and rows == list(range(r0, r1 + 1, 2)), "parallel lanes")
    _require(c1 - c0 + 1 >= 7, "rail bank width")
    core = a[r0:r1 + 1, c0:c1 + 1]
    separators = set(map(int, np.unique(core[1::2]))) - {bg}
    _require(len(separators) == 1, "one switch-bar color")
    gate = next(iter(separators))
    _require(gate != track, "switch and rail roles distinct")
    outside = a.copy()
    outside[r0:r1 + 1, c0:c1 + 1] = bg
    outside_colors = set(map(int, np.unique(outside))) - {bg}
    _require(len(outside_colors) == 1, "one courier outside the bank")
    ink = next(iter(outside_colors))
    rest = set(map(int, np.unique(a))) - {bg, track, gate, ink}
    _require(len({bg, track, gate, ink}) == 4 and len(rest) == 1, "five distinct roles")
    bead = next(iter(rest))
    ink_cells = {tuple(map(int, p)) for p in np.argwhere(a == ink)}
    mask, tip, initial = _pointed(ink_cells)
    _require(tip[0] in rows and tip[1] in (c0, c1), "tip at an outer rail end")
    _require(initial == (1 if tip[1] == c0 else 3), "courier points into the bank")
    _require({p for p in ink_cells if r0 <= p[0] <= r1 and c0 <= p[1] <= c1} == {tip},
             "input body must be outside the bank except for its tip")
    base = a.copy()
    base[base == ink] = bg
    base[tip] = track
    bead_cells = {tuple(map(int, p)) for p in np.argwhere(a == bead)}
    _require(all(r in rows and c0 < c < c1 for r, c in bead_cells), "beads inside input lanes, not at outer ports")
    _require(all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1
                 for p in bead_cells for q in bead_cells if p < q), "initial beads are singletons")
    base[base == bead] = track
    check_outside = base.copy()
    check_outside[r0:r1 + 1, c0:c1 + 1] = bg
    _require(bool(np.all(check_outside == bg)), "unrelated outside objects")
    _require(bool(np.all(np.isin(base[r0:r1 + 1:2, c0:c1 + 1], (bg, track)))), "lane cells")
    centers, breaks = [], []
    for r in rows:
        _require(base[r, c0] == track and base[r, c1] == track, "exposed ends")
        xs = [c for c in range(c0, c1 + 1) if base[r, c] == bg]
        _require(all(c0 + 2 <= c <= c1 - 2 for c in xs), "breaks away from bank ends")
        _require(all(b - x >= 3 for x, b in zip(xs, xs[1:])), "separated breaks")
        breaks.extend((r, c) for c in xs)
    for r in range(r0 + 1, r1, 2):
        xs = [c for c in range(c0, c1 + 1) if base[r, c] == gate]
        runs = []
        for c in xs:
            if not runs or c != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(c)
        for run in runs:
            _require(len(run) == 3 and c0 + 1 <= run[0] and run[-1] <= c1 - 1,
                     "three-cell interior switch bars")
            c = run[1]
            _require(bool(np.all(base[r - 1, c - 1:c + 2] == track)) and
                     bool(np.all(base[r + 1, c - 1:c + 2] == track)), "unbroken switch tile")
            centers.append((r, c))
    _require(bool(centers), "at least one switch")
    structural, writes, overlap = _switch(base, centers, track, gate)
    paths = _paths(structural == track)
    for path in paths:
        for r, c in (path[0], path[-1]):
            _require(r in rows and (c in (c0, c1) or base[r, c - 1] == bg or base[r, c + 1] == bg),
                     "path ends must be exposed")
    _require(all(structural[p] == track for p in bead_cells), "a switch cannot destroy a bead")
    active = [p for p in paths if tip in (p[0], p[-1])]
    _require(len(active) == 1, "one path begins at the tip")
    path = active[0] if active[0][0] == tip else active[0][::-1]
    _require(_facing(path[0], path[1]) == initial, "initial motion follows the tip")
    active_beads = bead_cells & set(path)
    _require(len(active_beads) >= 2, "at least two reachable checkpoints")
    indices = [i for i, p in enumerate(path) if p in active_beads]
    last = max(indices)
    _require(min(indices) >= 2, "checkpoint clear of the initial tip")
    final = _facing(path[last - 1], path[last])
    final_body = _body(mask, path[last], final)
    for i, p in enumerate(path[:last + 1]):
        direction = initial if i == 0 else _facing(path[i - 1], p)
        _require(_in_bounds(_body(mask, p, direction), a.shape), "body out of bounds during travel")
    passive = bead_cells - active_beads
    _require(not (passive & final_body), "final body cannot hide uncollected beads")
    available = [p for p in reversed(path[:last + 1]) if p not in final_body]
    _require(len(available) >= len(active_beads), "not enough room for cargo behind the body")
    cargo = set(available[:len(active_beads)])
    # No marker is deleted. Reachable checkpoints become a compact cargo load.
    answer = structural.copy()
    for p in passive | cargo:
        answer[p] = bead
    for p in final_body:
        answer[p] = ink
    info = dict(bg=bg, track=track, gate=gate, ink=ink, bead=bead,
                box=(r0, r1, c0, c1), mask=mask, tip=tip, initial=initial,
                centers=centers, breaks=breaks, paths=paths, path=path,
                structural=structural, base=base, writes=writes, overlap=overlap,
                beads=bead_cells, active=active_beads, passive=passive, indices=indices,
                last=last, final=final, body=final_body, cargo=cargo)
    return answer, info


def _decode(grid):
    a = _grid(grid)
    bg = int(a[0, 0])
    _require(bool(np.all(a[0] == bg) and np.all(a[-1] == bg) and
                  np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)), "background border")
    colors = set(map(int, np.unique(a)))
    _require(len(colors) == 5, "exactly five visible role colors")
    possibilities = []
    for transpose in (False, True):
        board = a.T if transpose else a
        for track in sorted(colors - {bg}):
            try:
                out, info = _interpret_horizontal(board, bg, track)
            except ValueError:
                continue
            possibilities.append((out.T.copy() if transpose else out, info, transpose))
    _require(len(possibilities) == 1, "input must have a unique geometric interpretation")
    out, info, transpose = possibilities[0]
    info["transpose"] = transpose
    info["shape"] = a.shape
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input using only its visible railwork, courier, and beads."""
    return _decode(input_grid)[0]


def _arrow(span, extension, solid, tail):
    cells = {(0, -i) for i in range(extension)}
    for k in range(1, span + 1):
        transverse = range(-k, k + 1) if solid else (-k, k)
        cells.update((r, -extension + 1 - k) for r in transverse)
    for k in range(1, tail + 1):
        cells.update((r, -extension + 1 - span - k) for r in (-span, span))
    return tuple(sorted(cells))


_MASKS = tuple(dict.fromkeys(_arrow(s, e, solid, t)
                           for s, e, solid, t in product((1, 2), (1, 2), (False, True), (0, 1))))


def _railwork(rng, lanes, width):
    """Sample switches across the bank, then cut unwanted return segments."""
    centers, previous = [], []
    richness = float(rng.uniform(.2, 1.0))
    for j in range(lanes - 1):
        chosen = []
        target = int(rng.integers(0, min(5, (width - 1) // 4) + 1))
        if rng.random() > richness:
            target = min(target, 1)
        for raw in rng.permutation(np.arange(2, width - 2)):
            c = int(raw)
            if len(chosen) == target:
                break
            if all(abs(c - x) >= 4 for x in chosen) and all(abs(c - x) >= 2 for x in previous):
                chosen.append(c)
        previous = sorted(chosen)
        centers.extend((2 * j + 1, c) for c in previous)
    if not centers:
        centers = [(2 * int(rng.integers(lanes - 1)) + 1, int(rng.integers(2, width - 2)))]
    breaks = []
    for r in range(0, 2 * lanes - 1, 2):
        ports = sorted((c, rr) for rr, c in centers if abs(rr - r) == 1)
        for (a, side), (b, other) in zip(ports, ports[1:]):
            if side == other:
                breaks.append((r, int(rng.integers(a + 2, b - 1))))
    # Additional cuts create interior exits and passive components.
    for _ in range(int(rng.integers(0, 5))):
        choices = [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(2, width - 2)
                   if all(abs(c - cc) >= 2 for rr, cc in centers if abs(rr - r) == 1)
                   and all(abs(c - cc) >= 3 for rr, cc in breaks if rr == r)]
        if not choices:
            break
        breaks.append(choices[int(rng.integers(len(choices)))])
    base = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    base[::2] = 1
    for r, c in centers:
        base[r, c - 1:c + 2] = 2
    for p in breaks:
        base[p] = 0
    return base, centers


def _separated_subset(rng, candidates, count, required, witness=None):
    chosen = [required] if witness is None else [required, witness]
    if len(chosen) == count:
        return set(chosen)
    for raw in rng.permutation(len(candidates)):
        p = candidates[int(raw)]
        if p not in chosen and all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in chosen):
            chosen.append(p)
            if len(chosen) == count:
                return set(chosen)
    return None


def _make_pair(rng, mask, load, quarter, passive_required, palette, anti_distance=False):
    depth = -min(c for r, c in mask)
    span = max(abs(r) for r, c in mask)
    margin = max(depth, span) + 1
    room = 30 - 2 * margin
    for attempt in range(100):
        lanes = int(rng.integers(2, min(11, (room + 1) // 2) + 1))
        width = int(rng.integers(max(9, 2 * load + 1), room + 1))
        raw, centers = _railwork(rng, lanes, width)
        turned, _, _ = _switch(raw, centers, 1, 2)
        try:
            paths = _paths(turned == 1)
        except ValueError:
            continue
        options = []
        for original in paths:
            for path in (original, original[::-1]):
                tip = path[0]
                if tip[1] not in (0, width - 1):
                    continue
                initial = 1 if tip[1] == 0 else 3
                for i in range(max(6, 2 * load), len(path) - 2):
                    q = path[i]
                    if raw[q] != 1 or q[0] == tip[0]:
                        continue
                    final = _facing(path[i - 1], q)
                    if quarter is True and final % 2 == initial % 2:
                        continue
                    if quarter is False and final % 2 != initial % 2:
                        continue
                    body = _body(mask, q, final)
                    if len([p for p in path[:i] if p not in body]) < load:
                        continue
                    candidates = [p for j, p in enumerate(path[:i + 1])
                                  if j >= 2 and raw[p] == 1]
                    if len(candidates) >= load:
                        options.append((path, i, body, candidates, initial))
        rng.shuffle(options)
        for path, last, body, candidates, initial in options[:100]:
            witness = None
            if anti_distance:
                distance = lambda p: abs(p[0] - path[0][0]) + abs(p[1] - path[0][1])
                farther = [p for p in candidates if distance(p) > distance(path[last]) and
                           max(abs(p[0] - path[last][0]), abs(p[1] - path[last][1])) > 1]
                if not farther:
                    continue
                witness = farther[int(rng.integers(len(farther)))]
            active = _separated_subset(rng, candidates, load, path[last], witness)
            if active is None:
                continue
            pool = [p for p in sorted(set(map(tuple, np.argwhere((raw == 1) & (turned == 1)))) - set(path))
                    if p not in body and 0 < p[1] < width - 1]
            rng.shuffle(pool)
            passive = set()
            wanted = int(rng.integers(1 if passive_required else 0, 7))
            for p in pool:
                if len(passive) >= wanted:
                    break
                if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in active | passive):
                    passive.add(p)
            if passive_required and not passive:
                continue
            # Pad enough for every intermediate rigid pose, plus variable margins.
            pads = []
            for size in raw.shape:
                slack = min(3, 30 - size - 2 * margin)
                extra = int(rng.integers(slack + 1))
                before = int(rng.integers(extra + 1))
                pads.append((margin + before, margin + extra - before))
            bg, track, gate, ink, bead = palette
            a = np.asarray((bg, track, gate), dtype=np.int8)[raw]
            for p in active | passive:
                a[p] = bead
            a = np.pad(a, pads, constant_values=bg)
            tip = (path[0][0] + pads[0][0], path[0][1] + pads[1][0])
            for p in _body(mask, tip, initial):
                a[p] = ink
            # Pair rotations are safe: the movement command is the visible tip.
            a = np.rot90(a, int(rng.integers(4))).copy()
            out, info = _decode(a)
            assert len(info["active"]) == load
            return {"input": a, "output": out}
    raise RuntimeError("bounded sampler could not place the requested cargo demonstration")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    fixed_palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    shared_palette = bool(rng.random() < .25)
    masks = rng.permutation(len(_MASKS))
    first_mask = _MASKS[int(masks[0])]
    # A second readable mask must actually occupy the rail cell behind its tip,
    # so training rules out cargo packing that ignores the body's footprint.
    filled_back = [m for m in _MASKS if m != first_mask and (0, -1) in m]
    second_mask = filled_back[int(rng.integers(len(filled_back)))]
    special = list(map(int, rng.permutation(num_train)[:2]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == special[0]:
                mask, load, quarter, passive = first_mask, 2, True, True
            elif split == "train" and index == special[1]:
                mask, load, quarter, passive = second_mask, 4, False, False
            else:
                mask = _MASKS[int(rng.integers(len(_MASKS)))]
                load = int(rng.integers(2, 5) if split == "train" else rng.integers(5, 8))
                quarter, passive = None, False
            for _ in range(12):
                palette = fixed_palette if shared_palette else tuple(map(int, rng.choice(10, 5, replace=False)))
                pair = _make_pair(rng, mask, load, quarter, passive, palette,
                                  anti_distance=(split == "train" and index == special[1]))
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _without_switches(info):
    """Counterfactual: the courier's original lane, before any rewiring."""
    path = next(p for p in _paths(info["base"] == info["track"])
                if info["tip"] in (p[0], p[-1]))
    if path[0] != info["tip"]:
        path = path[::-1]
    reachable = [p for p in path if p in info["beads"]]
    return reachable[-1] if reachable else info["tip"], set(reachable)


def _validate(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "preserve full canvas")
            key = _key(inp)
            _require(key not in seen, "distinct inputs")
            seen.add(key)
            expected, info = _decode(inp)
            _require(bool(np.array_equal(expected, out)), "incorrect switched courier and cargo state")
            _require(int(np.count_nonzero(out == info["bead"])) == len(info["beads"]), "bead conservation")
            _require(int(np.count_nonzero(out == info["ink"])) == len(info["mask"]), "rigid body area")
            normalized_out = out.T if info["transpose"] else out
            _require({tuple(map(int, p)) for p in np.argwhere(normalized_out == info["ink"])} == info["body"],
                     "exact body geometry and pose")
            _require(all(normalized_out[p] == info["bead"] for p in info["passive"]), "passive beads fixed")
            cargo_scan = [p for p in reversed(info["path"][:info["last"] + 1]) if p not in info["body"]]
            _require(info["cargo"] == set(cargo_scan[:len(info["active"])]), "compact packing behind body")
            _require(info["last"] <= len(info["path"]) - 3, "stop before rail endpoint")
            no_a_tip, no_a_beads = _without_switches(info)
            _require(no_a_tip != info["path"][info["last"]] and no_a_beads != info["active"],
                     "switch connectivity must affect destination and pickup")
            analyses[split].append(info)
    train, test = analyses["train"], analyses["test"]
    _require(len({a["mask"] for a in train}) >= 2, "demonstrate actual shape preservation")
    _require({2, 4} <= {len(a["active"]) for a in train}, "demonstrate two different cargo loads")
    _require(any(a["initial"] % 2 != a["final"] % 2 for a in train), "demonstrate quarter-turn arrival")
    _require(any(a["passive"] for a in train), "demonstrate selective pickup")
    _require(all(2 <= len(a["active"]) <= 4 for a in train) and
             all(5 <= len(a["active"]) <= 7 for a in test), "cargo-load train/test shift")
    _require(any(_body(a["mask"], a["path"][a["last"]], a["initial"]) != a["body"] for a in train),
             "omitting body rotation changes the answer")
    _require(any(a["cargo"] != set(a["path"][a["last"] - len(a["active"]):a["last"]]) for a in train),
             "demonstrate packing that depends on the actual body")
    _require(any(any(abs(p[0] - a["tip"][0]) + abs(p[1] - a["tip"][1]) >
                         abs(a["path"][a["last"]][0] - a["tip"][0]) +
                         abs(a["path"][a["last"]][1] - a["tip"][1]) for p in a["active"])
                 for a in train), "route order must be distinguished from spatial distance")
    return analyses


def validate_generated_puzzle(puzzle) -> None:
    """Validate schema, topology, exact pose, selective pickup and cargo shift."""
    try:
        _validate(puzzle)
    except (ValueError, TypeError, IndexError, StopIteration) as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def _structure(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def self_test():
    """Bounded 42-seed audit plus geometry, determinism and corruption checks."""
    signatures, structures, shapes, masks, networks, loads, turns = set(), set(), set(), set(), set(), set(), set()
    palettes, overlaps, breaks, passive_counts = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        analyses = _validate(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair, a in zip(puzzle[split], analyses[split]):
                structures.add(_structure(pair["input"]))
                shapes.add(pair["input"].shape)
                masks.add(a["mask"])
                networks.add((a["base"].shape, (a["base"] == a["track"]).tobytes(),
                              tuple(a["centers"])))
                loads.add(len(a["active"]))
                turns.add((a["final"] - a["initial"]) % 4)
                palettes.add(tuple(a[k] for k in ("bg", "track", "gate", "ink", "bead")))
                overlaps.add(a["overlap"])
                breaks.add(len(a["breaks"]))
                passive_counts.add(len(a["passive"]))
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 40 and len(networks) >= 150 and len(masks) >= 12
    assert loads == set(range(2, 8)) and turns == set(range(4))
    assert len(palettes) >= 60 and max(overlaps) >= 1 and len(breaks) >= 5
    assert len(passive_counts) >= 4
    for role in range(5):
        assert {p[role] for p in palettes} == set(range(10))
    first, again = generate_puzzle(901, 3, 2), generate_puzzle(901, 3, 2)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1200 + 10 * nt + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    cycle = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    try:
        _paths(cycle)
    except ValueError:
        pass
    else:
        raise AssertionError("cycle accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal explicit counts)")
