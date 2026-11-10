"""Branching stencil packets: pattern-controlled rays on a 3x3 block lattice.

Only NumPy and the standard library are used. No parent code is imported.
solve() reads visible input roles, geometry, and the packet mask only.
"""

from __future__ import annotations

import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid must be 2D with dimensions 1..30")
    if not np.issubdtype(a.dtype, np.integer) or not np.all((0 <= a) & (a <= 9)):
        raise ValueError("grid must have integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _blocks(a):
    h, w = a.shape
    return a.reshape(h // 3, 3, w // 3, 3).transpose(0, 2, 1, 3)


def _unblock(b):
    h, w = b.shape[:2]
    return b.transpose(0, 2, 1, 3).reshape(h * 3, w * 3).copy()


def _analyze(input_grid):
    a = _grid(input_grid)
    if any(n % 3 or not 3 <= n // 3 <= 10 for n in a.shape):
        raise ValueError("expected 3..10 aligned blocks on each axis")
    blocks = _blocks(a)
    h, w = blocks.shape[:2]
    uniform = np.all(blocks == blocks[:, :, :1, :1], axis=(2, 3))
    source = np.argwhere(~uniform)
    if source.shape != (1, 2):
        raise ValueError("expected exactly one patterned source block")
    r, c = map(int, source[0])
    edges = [(r == 0, 0), (c == 0, 1), (r == h - 1, 2), (c == w - 1, 3)]
    turns = [k for yes, k in edges if yes]
    if len(turns) != 1:
        raise ValueError("source must touch a unique edge, excluding corners")
    k = turns[0]
    canonical = np.rot90(a, -k).copy()
    b = _blocks(canonical)
    h, w = b.shape[:2]
    source = np.argwhere(~np.all(b == b[:, :, :1, :1], axis=(2, 3)))[0]
    r, c = map(int, source)
    if r != 0 or not 0 < c < w - 1:
        raise ValueError("ambiguous inward direction")
    t = b[0, c].copy()
    carrier = int(t[1, 1])
    colors = set(map(int, np.unique(t)))
    if len(colors) != 2:
        raise ValueError("source must have exactly carrier and detail colors")
    detail = next(x for x in colors if x != carrier)
    if not (np.any(t[0, [0, 2]] == detail) and np.any(t[2, [0, 2]] == detail)):
        raise ValueError("both orientations need at least one forward gate")
    u = np.ones((h, w), dtype=bool)
    u[0, c] = False
    values, counts = np.unique(b[:, :, 0, 0][u], return_counts=True)
    if len(values) != 2 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("need obstacle blocks and a unique background mode")
    background = int(values[np.argmax(counts)])
    obstacle = next(int(x) for x in values if x != background)
    if len({background, obstacle, carrier, detail}) != 4:
        raise ValueError("the four visible color roles must be distinct")
    walls = (b[:, :, 0, 0] == obstacle) & u
    return dict(input=a, canonical=canonical, blocks=b, turns=k, tile=t,
                carrier=carrier, detail=detail, background=background,
                obstacle=obstacle, walls=walls, source=c, h=h, w=w)


def _view(tile, operation):
    # Eight global square symmetries, used only for explicit alternative checks.
    if operation < 4:
        return np.rot90(tile, operation)
    return np.rot90(tile[:, ::-1], operation - 4)


def _runs(row):
    result = []
    c = 0
    while c < len(row):
        if not row[c]:
            c += 1
            continue
        a = c
        while c + 1 < len(row) and row[c + 1]:
            c += 1
        result.append((a, c))
        c += 1
    return result


def _trace(info, variant="correct", operation=2):
    """Record state sets independently of prints; bit 1/2 is orientation 0/1.

    Alternative modes are intentionally incorrect rules for witness checking.
    They never supply generated answers.
    """
    h, w = info["h"], info["w"]
    walls, s = info["walls"], info["source"]
    tiles = (info["tile"], _view(info["tile"], operation))
    detail = info["detail"]
    prints = np.zeros((h, w), dtype=np.uint8)
    prints[0, s] = 1
    active = {s: 1}
    if variant == "all_obstacles":
        # The incorrect rule activates even boundary obstacles with no incident
        # packet. There is no upstream layer here, but fictitious flanks grow.
        for a, b in _runs(walls[0]):
            for x, corner in ((a - 1, 0), (b + 1, 2)):
                if 0 <= x < w and tiles[1][2, corner] == detail:
                    active[x] = active.get(x, 0) | 2
                    prints[0, x] |= 2
    events, incoming = [], []
    stats = dict(simultaneous=0, multi_hit=0, clipped=0, suppressed=0,
                 double_branch=0, interrupted=0, state_mergers=0)
    reached = set()
    for r in range(1, h):
        incoming.append((r, tuple(sorted(active.items()))))
        following, hits = {}, {}
        runs = _runs(walls[r])
        for c, bits in sorted(active.items()):
            if walls[r, c] and variant != "ignore_obstacles":
                run = next((a, b) for a, b in runs if a <= c <= b)
                hits.setdefault(run, []).append((c, bits))
            else:
                following[c] = following.get(c, 0) | bits
                if not walls[r, c]:
                    prints[r, c] |= bits
        if variant == "all_obstacles":
            for run in runs:
                hits.setdefault(run, [(run[0], 1)])
        stats["simultaneous"] += len(hits) > 1
        if variant == "one_per_layer" and hits:
            hits = {min(hits): hits[min(hits)]}
        for (a, b), packets in sorted(hits.items()):
            bits = 0
            for _, state in packets:
                bits |= state
            reached.add((r, a, b))
            stats["multi_hit"] += len(packets) > 1
            stats["clipped"] += a == 0 or b == w - 1
            span = range(max(0, a - 1), min(w, b + 2))
            stats["interrupted"] += any(walls[r - 1, c] for c in span)
            for c in span:
                if not walls[r - 1, c]:
                    prints[r - 1, c] |= bits
            emitted = []
            for p in (0, 1):
                if not bits & (1 << p):
                    continue
                q = 1 if variant == "turn_once" else 1 - p
                sampled = tiles[p if variant == "sample_before_turn" else q]
                enabled = 0
                for x, corner in ((a - 1, 0), (b + 1, 2)):
                    if not 0 <= x < w:
                        continue
                    value = sampled[2, 1 if variant == "sample_middle" else corner]
                    gate = variant == "all_gates" or value == detail
                    if not gate:
                        stats["suppressed"] += 1
                        continue
                    enabled += 1
                    new = 1 << q
                    previous = following.get(x, 0)
                    stats["state_mergers"] += previous != 0 and not previous & new
                    following[x] = previous | new
                    prints[r, x] |= new
                    emitted.append((x, q))
                stats["double_branch"] += enabled == 2
            events.append((r, a, b, bits, tuple(emitted)))
        if variant == "live_crosspieces":
            # Incorrectly treat every upstream print as a new forward packet.
            for c in range(w):
                if prints[r - 1, c] and not walls[r, c]:
                    following[c] = following.get(c, 0) | int(prints[r - 1, c])
                    prints[r, c] |= prints[r - 1, c]
        active = following
    all_runs = {(r, a, b) for r in range(h) for a, b in _runs(walls[r])}
    stats.update(events=events, incoming=incoming, reached=len(reached),
                 inert=len(all_runs - reached), run_count=len(all_runs),
                 union_blocks=int(np.count_nonzero((prints == 3) & ~walls)),
                 reached_lengths=tuple(sorted(b - a + 1 for _, a, b in reached)),
                 all_lengths=tuple(sorted(b - a + 1 for _, a, b in all_runs)),
                 source=s, shape=(h, w), orientation=info["turns"],
                 mask=tuple(map(int, (info["tile"] == detail).flat)))
    return prints, stats


def _render(info, prints, operation=2, drop_union=False):
    b = info["blocks"].copy()
    tile = info["tile"]
    tiles = (tile, _view(tile, operation))
    for r, c in np.argwhere(prints):
        r, c = int(r), int(c)
        if info["walls"][r, c] or (r == 0 and c == info["source"]):
            continue
        bits = int(prints[r, c])
        if bits == 1 or (bits == 3 and drop_union):
            b[r, c] = tiles[0]
        elif bits == 2:
            b[r, c] = tiles[1]
        else:
            b[r, c] = np.where((tiles[0] == info["detail"]) |
                               (tiles[1] == info["detail"]),
                               info["detail"], info["carrier"])
    return np.ascontiguousarray(np.rot90(_unblock(b), info["turns"]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles and packet states exclusively from this input."""
    info = _analyze(input_grid)
    prints, _ = _trace(info)
    return _render(info, prints)


_FAULTS = ("ignore_obstacles", "all_gates", "turn_once", "sample_before_turn", "sample_middle",
           "all_obstacles", "one_per_layer", "drop_union", "live_crosspieces")


def _alternative(info, fault=None, operation=2):
    prints, _ = _trace(info, "correct" if fault == "drop_union" or fault is None else fault,
                       operation)
    return _render(info, prints, operation, fault == "drop_union")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _signature(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def _source_mask(rng, kind):
    t = np.zeros((3, 3), dtype=np.int8)
    if kind == "gates":
        t[0, [0, 2]] = 1
        t[2, int(rng.choice([0, 2]))] = 1
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        t[edges[int(rng.integers(4))]] = 1
    elif kind == "merge":
        t[0, [0, 2]] = t[2, [0, 2]] = 1
        t[[(0, 1), (1, 0), (1, 2), (2, 1)][int(rng.integers(4))]] = 1
    else:
        # Arbitrary masks with at least one corner on each forward/backward edge.
        for row in (0, 2):
            selected = rng.choice([0, 2], size=int(rng.integers(1, 3)), replace=False)
            t[row, selected] = 1
        for point in ((0, 1), (1, 0), (1, 2), (2, 1)):
            t[point] = rng.random() < float(rng.choice([.15, .45, .8]))
        # Retain a second carrier cell as well as the fixed center.
        if t.sum() == 8:
            t[[(0, 1), (1, 0), (1, 2), (2, 1)][int(rng.integers(4))]] = 0
    return t


def _input(walls, source, mask, palette, direction):
    background, obstacle, carrier, detail = palette
    h, w = walls.shape
    b = np.full((h, w, 3, 3), background, dtype=np.int8)
    b[walls] = obstacle
    b[0, source] = np.where(mask, detail, carrier)
    return np.ascontiguousarray(np.rot90(_unblock(b), direction))


def _witness_geometry(rng, kind, cap):
    """Sample coordinates around local causal contrasts, not whole puzzle banks."""
    if kind == "gates":
        h, w = int(rng.integers(6, cap + 1)), int(rng.integers(6, cap + 1))
        walls = np.zeros((h, w), dtype=bool)
        a = int(rng.integers(2, w - 3))
        b = int(rng.integers(a, w - 2))
        source = int(rng.integers(a, b + 1))
        r1 = int(rng.integers(1, h - 3))
        r2 = int(rng.integers(r1 + 2, h - 1))
        walls[r1, a:b + 1] = True
        # A second interior collision on just one daughter, followed by a free layer.
        left = a - 1
        lo = int(rng.integers(1, left + 1))
        hi = int(rng.integers(left, b + 1))
        walls[r2, lo:hi + 1] = True
        # An equally ordinary inert singleton beside the source establishes causality.
        walls[0, int(rng.choice([0, w - 1]))] = True
        # Extra causal/inert bars can be combined beyond the decisive two collisions.
        if r2 + 2 < h and rng.random() < .55:
            r = int(rng.integers(r2 + 2, h))
            lo = int(rng.integers(w))
            hi = int(rng.integers(lo, w))
            walls[r, lo:hi + 1] = True
    else:
        h, w = int(rng.integers(8, cap + 1)), int(rng.integers(7, cap + 1))
        walls = np.zeros((h, w), dtype=bool)
        source = int(rng.integers(3, w - 2))
        r1 = int(rng.integers(1, h - 6))
        r2 = r1 + int(rng.integers(1, 3))
        r3 = r2 + 2
        r4 = int(rng.integers(r3 + 1, h))
        walls[r1, source] = True
        # The right daughter passes this bar. The other's new right flank joins it
        # after one extra collision, so both packet orientations arrive together.
        walls[r2, source - 1:source + 1] = True
        walls[r3, source - 2] = walls[r3, source + 1] = True
        # The smallest ray coordinate here is source-3; an edge run clips one flank.
        walls[r4, :source - 2] = True
        walls[0, 0] = True
        if rng.random() < .5:
            # Connected prefix obstacle interrupts an upstream print while staying inert.
            walls[max(0, r1 - 1), 0] = True
    return walls, source


def _next_states(row, active, mask):
    """Generation-only scheduling helper; solve never calls this function."""
    following, hits = {}, {}
    runs = _runs(row)
    for c, bits in sorted(active.items()):
        if row[c]:
            run = next((a, b) for a, b in runs if a <= c <= b)
            hits[run] = hits.get(run, 0) | bits
        else:
            following[c] = following.get(c, 0) | bits
    for (a, b), bits in hits.items():
        for p in (0, 1):
            if bits & (1 << p):
                q = 1 - p
                tile = mask if q == 0 else mask[::-1, ::-1]
                for c, corner in ((a - 1, 0), (b + 1, 2)):
                    if 0 <= c < len(row) and tile[2, corner]:
                        following[c] = following.get(c, 0) | (1 << q)
    return following


def _causal_run(rng, row, pivot):
    """Vary length and impact offset while separating existing maximal runs."""
    if row[pivot]:
        return
    lo = hi = pivot
    while lo > 0 and not row[lo - 1] and (lo < 2 or not row[lo - 2]):
        lo -= 1
    while hi + 1 < len(row) and not row[hi + 1] and (hi + 2 == len(row) or not row[hi + 2]):
        hi += 1
    capacity = hi - lo + 1
    style = rng.random()
    if style < .35:
        length = 1
    elif style < .8:
        length = int(rng.integers(1, min(4, capacity) + 1))
    else:
        length = int(rng.integers(1, capacity + 1))
    left = int(rng.integers(max(lo, pivot - length + 1), min(pivot, hi - length + 1) + 1))
    row[left:left + length] = True


def _random_geometry(rng, cap, mask, scale_test=False):
    h, w = map(int, rng.integers(4, cap + 1, size=2))
    if scale_test:
        if rng.random() < .5:
            h = 10
        else:
            w = 10
    source = int(rng.integers(1, w - 1))
    walls = np.zeros((h, w), dtype=bool)
    for _ in range(int(rng.integers(0, h + 1))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        length = int(rng.integers(1, min(w, 4) + 1)) if rng.random() < .8 else int(rng.integers(1, w + 1))
        lo = int(rng.integers(w - length + 1))
        kind = int(rng.integers(4))
        if kind < 2:
            walls[r, lo:lo + length] = True
        else:
            depth = int(rng.integers(1, min(4, h - r) + 1))
            walls[r:r + depth, c] = True
            if kind == 3:
                # Bent connected object; the struck transverse layer remains the unit.
                walls[r + depth - 1, min(c, lo):max(c, lo) + 1] = True
    walls[0, source] = False
    # Ensure background is the strict mode without coupling run length to reachability.
    if walls.sum() >= (h * w - 1) / 2:
        rows = list(map(int, rng.permutation(h)))
        for r in rows:
            walls[r] = False
            if walls.sum() < (h * w - 1) / 2:
                break
    # Schedule varied collisions on actual active packets, allowing simultaneous
    # events and same-run multi-hits. Canvas size does not prescribe event count.
    active = {source: 1}
    probability = float(rng.choice([.25, .5, .8]))
    event_rows = set(range(1, h)) if rng.random() < .7 else {int(rng.integers(1, h))}
    for r in range(1, h):
        if r in event_rows and active and rng.random() < probability:
            available = [c for c in sorted(active) if not walls[r, c]]
            rng.shuffle(available)
            quota = int(rng.integers(1, min(3, len(available)) + 1)) if available else 0
            for c in available[:quota]:
                proposal = walls[r].copy()
                _causal_run(rng, proposal, c)
                if walls.sum() - walls[r].sum() + proposal.sum() < (h * w - 1) / 2:
                    walls[r] = proposal
        active = _next_states(walls[r], active, mask)
    # If the initial column contains no obstacle, no packet can have branched.
    # Add a guaranteed reached singleton, possibly joining an existing run.
    if not np.any(walls[1:, source]):
        walls[int(rng.integers(1, h)), source] = True
    return walls, source


def _choose_palette(rng, settings):
    if settings["fixed_palette"]:
        return settings["palette"]
    return tuple(map(int, rng.choice(10, 4, replace=False)))


def _choose_direction(rng, settings, split, index):
    if settings["shift"] == "orientation":
        if split == "test":
            return settings["held_out"]
        return (settings["held_out"] + 1 + index % 3) % 4
    if settings["fixed_direction"]:
        return settings["direction"]
    return int(rng.integers(4))


def _make_pair(rng, kind, settings, split, index, seen):
    cap = 9 if settings["shift"] == "scale" and split == "train" else 10
    direction = _choose_direction(rng, settings, split, index)
    palette = _choose_palette(rng, settings)
    for _ in range(160):
        mask = _source_mask(rng, kind)
        if kind in ("gates", "merge"):
            walls, source = _witness_geometry(rng, kind, cap)
        else:
            walls, source = _random_geometry(rng, cap, mask, settings["shift"] == "scale" and split == "test")
        a = _input(walls, source, mask, palette, direction)
        if _key(a) in seen:
            continue
        try:
            info = _analyze(a)
        except ValueError:
            continue
        prints, stats = _trace(info)
        out = _render(info, prints)
        if stats["reached"] == 0 or np.array_equal(a, out):
            continue
        if kind == "gates":
            if not (stats["suppressed"] and stats["double_branch"] and stats["reached"] >= 2):
                continue
            if any(np.array_equal(out, _alternative(info, fault)) for fault in
                   ("ignore_obstacles", "all_gates", "sample_before_turn")):
                continue
            if any(np.array_equal(out, _alternative(info, operation=op)) for op in range(8) if op != 2):
                continue
        if kind == "merge":
            if not (stats["state_mergers"] and stats["simultaneous"] and stats["clipped"]):
                continue
            if any(np.array_equal(out, _alternative(info, fault)) for fault in
                   ("drop_union", "one_per_layer")):
                continue
        return {"input": a, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def _training_ambiguities(train):
    """Named finite alternative family; comparisons read training outputs only."""
    infos = [_analyze(p["input"]) for p in train]
    alternatives = [(fault, 2) for fault in _FAULTS]
    alternatives += [(None, op) for op in range(8) if op != 2]
    return [(fault, op) for fault, op in alternatives
            if all(np.array_equal(_alternative(info, fault, op), p["output"])
                   for info, p in zip(infos, train))]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    settings = dict(palette=tuple(map(int, rng.choice(10, 4, replace=False))),
                    fixed_palette=bool(rng.random() < .4),
                    fixed_direction=bool(rng.random() < .45),
                    direction=int(rng.integers(4)), held_out=int(rng.integers(4)),
                    shift=str(rng.choice(["composition", "orientation", "scale"],
                                         p=[.55, .25, .20])))
    puzzle = {"train": [], "test": []}
    seen = set()
    for i in range(nt):
        kind = ("gates", "merge")[i] if i < 2 else "random"
        pair = _make_pair(rng, kind, settings, "train", i, seen)
        seen.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    rng.shuffle(puzzle["train"])
    surviving = _training_ambiguities(puzzle["train"])
    for i in range(nq):
        for _ in range(100):
            pair = _make_pair(rng, "random", settings, "test", i, seen)
            info = _analyze(pair["input"])
            if all(np.array_equal(_alternative(info, fault, op), pair["output"])
                   for fault, op in surviving):
                break
        else:
            # A previously observed input pattern is always compatible with
            # training-consistent motif transforms. Fresh geometry remains required.
            raise RuntimeError("bounded training coverage construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_prints(info, output, prints):
    """Scalar cell check of preserved blocks and additive prints.

    This does not call the renderer. The event trace is generator-owned, so this
    is targeted consistency checking rather than an independent semantic solver.
    """
    out = _blocks(np.rot90(output, -info["turns"]))
    tile, detail = info["tile"], info["detail"]
    for r in range(info["h"]):
        for c in range(info["w"]):
            immutable = info["walls"][r, c] or (r == 0 and c == info["source"])
            bits = int(prints[r, c])
            if immutable or not bits:
                _require(np.array_equal(out[r, c], info["blocks"][r, c]),
                         "source, wall, or unvisited background changed")
                continue
            for y in range(3):
                for x in range(3):
                    marked = ((bits & 1 and tile[y, x] == detail) or
                              (bits & 2 and tile[2 - y, 2 - x] == detail))
                    expected = detail if marked else info["carrier"]
                    _require(out[r, c, y, x] == expected, "wrong half-turn or detail union")
            _require(out[r, c, 1, 1] == info["carrier"], "print lost carrier center")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training_stats = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "changed dimensions")
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(a)
            prints, stats = _trace(info)
            _check_prints(info, out, prints)
            _require(not np.array_equal(a, out), "unchanged pair")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(a))), "introduced color")
            if split == "train":
                training_stats.append(stats)
            else:
                _require(stats["reached"] > 0, "test lacks a causal encounter")
    for feature in ("suppressed", "double_branch", "state_mergers", "simultaneous", "clipped", "inert"):
        _require(any(s[feature] for s in training_stats), f"training lacks {feature} witness")
    infos = [_analyze(p["input"]) for p in puzzle["train"]]
    # Parental necessity and added superposition must be visibly taught.
    for fault in ("ignore_obstacles", "all_gates", "turn_once", "sample_before_turn",
                  "all_obstacles", "drop_union", "one_per_layer"):
        _require(any(not np.array_equal(_alternative(info, fault), p["output"])
                     for info, p in zip(infos, puzzle["train"])), f"training admits {fault}")
    _require(any(not np.array_equal(_alternative(info, operation=0), p["output"])
                 for info, p in zip(infos, puzzle["train"])), "training omits tile turning")
    for fault, op in _training_ambiguities(puzzle["train"]):
        for pair in puzzle["test"]:
            info = _analyze(pair["input"])
            _require(np.array_equal(_alternative(info, fault, op), pair["output"]),
                     "training leaves a tested query-changing alternative")
    return None


def _normalized(a):
    labels = {}
    return tuple(labels.setdefault(int(x), len(labels)) for x in a.flat)


def self_test():
    """Bounded 42-seed variation, symmetry, corruption, and count checks."""
    puzzles = []
    shapes, structures, masks, source_positions, signatures = set(), set(), set(), set(), set()
    palettes, directions, lengths, corner_patterns, marks = set(), set(), set(), set(), set()
    observed = {name: 0 for name in ("suppressed", "double_branch", "state_mergers",
                                   "simultaneous", "multi_hit", "clipped", "interrupted")}
    test_observed = {name: 0 for name in observed}
    shift_orientation = shift_scale = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        train_directions, test_directions, train_sizes, test_sizes = set(), set(), [], []
        for split in ("train", "test"):
            for pair in p[split]:
                a, out = pair["input"], pair["output"]
                _require(np.array_equal(solve(a.copy()), out), "solve changed with copied input")
                _require(np.array_equal(solve(a.tolist()), out), "nested-list solve mismatch")
                info = _analyze(a)
                prints, stats = _trace(info)
                structures.add((a.shape, _normalized(a)))
                shapes.add(a.shape)
                masks.add(stats["mask"])
                source_positions.add((stats["shape"], stats["source"]))
                palettes.add((info["background"], info["obstacle"], info["carrier"], info["detail"]))
                directions.add(info["turns"])
                lengths.update(stats["all_lengths"])
                mask = info["tile"] == info["detail"]
                corner_patterns.add(tuple(bool(mask[y, x]) for y, x in ((0, 0), (0, 2), (2, 0), (2, 2))))
                marks.add(int(mask.sum()))
                for feature in observed:
                    observed[feature] += stats[feature] > 0
                    if split == "test":
                        test_observed[feature] += stats[feature] > 0
                (train_directions if split == "train" else test_directions).add(info["turns"])
                (train_sizes if split == "train" else test_sizes).append(max(a.shape))
                # Half-turn, coordinate sampling, and union must commute with scene rotation.
                for k in (1, 2, 3):
                    _require(np.array_equal(solve(np.rot90(a, k)), np.rot90(out, k)),
                             "rotation covariance failed")
                _require(np.array_equal(solve(a[:, ::-1]), out[:, ::-1]), "reflection covariance failed")
                # Arbitrary literal-color permutation preserves visible role semantics.
                mapping = np.array([7, 2, 9, 4, 1, 8, 0, 6, 3, 5], dtype=np.int8)
                _require(np.array_equal(solve(mapping[a]), mapping[out]), "palette covariance failed")
        shift_orientation += bool(test_directions - train_directions)
        shift_scale += min(test_sizes) > max(train_sizes)
    _require(len(signatures) == 42, "seed diversity collapsed")
    _require(len(shapes) >= 30 and len(structures) >= 180, "geometric structure variation collapsed")
    _require(len(masks) >= 35 and len(source_positions) >= 50, "mask/entry variation collapsed")
    _require(len(palettes) >= 80 and directions == set(range(4)), "palette/direction variation collapsed")
    _require(len(corner_patterns) >= 7 and {2, 3, 4, 5, 6, 7} <= marks, "gate/motif variation collapsed")
    _require({1, 2, 3, 4, 5, 6, 7, 8, 9, 10} <= lengths, "run-length variation collapsed")
    _require(all(value > 0 for value in observed.values()), "interaction coverage collapsed")
    _require(all(value > 0 for value in test_observed.values()), "query interaction coverage collapsed")
    _require(shift_orientation >= 3 and shift_scale >= 3, "train/test shift variation collapsed")
    # Exercise every legal explicit count combination once, without a seed cross product.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 100 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, False), (2.0, 1), (None, 1), (3, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_signature(generate_puzzle(0)) == _signature(puzzles[0]), "seed nondeterminism")
    # Smallest family dimensions and a dense mask outside the ordinary sampler.
    # Exact destinations below are specified directly, without a trace call.
    walls = np.zeros((3, 3), dtype=bool)
    walls[1, 1] = True
    dense = np.ones((3, 3), dtype=np.int8)
    dense[1, 1] = 0
    a = _input(walls, 1, dense, (7, 2, 6, 9), 0)
    expected = _blocks(a).copy()
    tile = np.where(dense, 9, 6)
    for r, c in ((0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)):
        expected[r, c] = tile
    _require(np.array_equal(solve(a), _unblock(expected)), "small dense-mask fixture failed")
    # Clipping can remove the sole enabled side and leave no descendant packet.
    walls[1, 0] = True
    mask = np.zeros((3, 3), dtype=np.int8)
    mask[0, 2] = mask[2, 0] = 1
    a = _input(walls, 1, mask, (7, 2, 6, 9), 0)
    expected = _blocks(a).copy()
    expected[0, 0] = expected[0, 2] = np.where(mask, 9, 6)
    _require(np.array_equal(solve(a), _unblock(expected)), "clipped extinction fixture failed")
    # Corrupt a source, wall, free print, and an unvisited background cell in an answer.
    base = puzzles[0]
    info = _analyze(base["train"][0]["input"])
    prints, _ = _trace(info)
    candidates = [(0, info["source"]), tuple(map(int, np.argwhere(info["walls"])[0]))]
    candidates += [tuple(map(int, np.argwhere((prints > 0) & ~info["walls"] &
                                              (np.indices(prints.shape)[0] > 0))[0]))]
    empty = np.argwhere((prints == 0) & ~info["walls"])
    if len(empty):
        candidates.append(tuple(map(int, empty[0])))
    for r, c in candidates:
        broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in base[s]]
                  for s in ("train", "test")}
        answer = np.rot90(broken["train"][0]["output"], -info["turns"]).copy()
        answer[3 * r + 1, 3 * c + 1] = (int(answer[3 * r + 1, 3 * c + 1]) + 1) % 10
        broken["train"][0]["output"] = np.rot90(answer, info["turns"])
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted output was accepted")
    # Compact reproducible descriptive evidence, useful to the later reviewer.
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in p),
                puzzles=len(signatures), shapes=len(shapes), normalized_structures=len(structures),
                masks=len(masks), source_layouts=len(source_positions), palettes=len(palettes),
                gate_patterns=len(corner_patterns), mark_counts=sorted(marks),
                run_lengths=sorted(lengths), interactions=observed, test_interactions=test_observed,
                orientation_shifts=shift_orientation, scale_shifts=shift_scale,
                independent_solver=False, blind_prediction=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
