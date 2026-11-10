"""Bead-powered rotary forks: input-visible roles, conserved mass, causal gates.

Only the standard library and NumPy are used.  No parent module or evidence is
loaded by this implementation.  All coordinates in the internal representation
have the stock-bearing entry boundary at the top.
"""
from collections import defaultdict
import json

import numpy as np


_ROLE_NAMES = ("background", "channel", "bead", "gate", "tip")
_SHORTCUTS = ("duplicate", "all_gates", "no_rotation", "no_pool", "fixed_left")


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _parse_canonical(a, bg, channel, bead, gate, tip):
    """Recognize reservoirs and every tipped bar, without generated metadata."""
    h, w = a.shape
    if h < 9 or w < 5:
        return None
    entries = list(map(int, np.flatnonzero(a[0] == channel)))
    if not entries or any(b - c < 2 for c, b in zip(entries, entries[1:])):
        return None
    if np.any((a[0] != bg) & (a[0] != channel)):
        return None
    source_mask = np.zeros(a.shape, dtype=bool)
    caps = np.zeros(a.shape, dtype=bool)
    beads = np.zeros(a.shape, dtype=bool)
    stocks = {}
    for c in entries:
        n = 0
        while n + 1 < h and a[n + 1, c] == bead:
            n += 1
        if not n or n + 1 >= h or a[n + 1, c] != channel:
            return None
        stocks[c] = n
        source_mask[:n + 2, c] = True
        caps[0, c] = caps[n + 1, c] = True
        beads[1:n + 1, c] = True
    if not np.array_equal(a == channel, caps) or not np.array_equal(a == bead, beads):
        return None
    remaining = (a != bg) & ~source_mask
    if np.any(remaining & ~np.isin(a, [gate, tip])):
        return None
    layers = {}
    for r in range(h):
        cols = list(map(int, np.flatnonzero(remaining[r])))
        if not cols:
            continue
        bars = []
        j = 0
        while j < len(cols):
            start = cols[j]
            if (j + 2 >= len(cols) or cols[j:j + 3] != [start, start + 1, start + 2]
                    or (j + 3 < len(cols) and cols[j + 3] == start + 3)):
                return None
            triple = tuple(map(int, a[r, start:start + 3]))
            if triple == (tip, gate, gate):
                preference = -1
            elif triple == (gate, gate, tip):
                preference = 1
            else:
                return None
            bars.append((start + 1, preference))
            j += 3
        if any(d - c < 4 for (c, _), (d, _) in zip(bars, bars[1:])):
            return None
        layers[r] = bars
    rows = sorted(layers)
    total = sum(stocks.values())
    if (not rows or rows[0] < max(stocks.values()) + 3
            or any(b - r < 4 for r, b in zip(rows, rows[1:]))
            or rows[-1] > h - total - 2):
        return None
    return {"stocks": stocks, "layers": layers, "total": total,
            "palette": (bg, channel, bead, gate, tip)}


def _recognize(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 5 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("input must have five visible roles and a unique dominant background")
    bg = int(colors[np.argmax(counts)])
    rest = [int(c) for c in colors if c != bg]
    choices = []
    for turns in range(4):
        canonical = np.rot90(a, turns).copy()
        for channel in rest:
            if not np.any(canonical[0] == channel):
                continue
            for bead in rest:
                if bead == channel:
                    continue
                others = [c for c in rest if c not in (channel, bead)]
                for gate, tip in (others, others[::-1]):
                    info = _parse_canonical(canonical, bg, channel, bead, gate, tip)
                    if info is not None:
                        choices.append((canonical, turns, info))
    if len(choices) != 1:
        raise ValueError("reservoirs, tipped gates and inward direction must be unambiguous")
    return choices[0]


def _division(mass, preference):
    half = mass // 2
    return (half + (mass % 2 if preference == -1 else 0),
            half + (mass % 2 if preference == 1 else 0))


def _simulate(a, info, shortcut=None):
    """Forward weighted propagation; gate layers are simultaneous macro events.

    A gate occupies rows r-1..r+1 after turning.  Its connecting crosspiece is
    consequently at r-2.  Gates on different layers have disjoint local writes.
    Wrong-rule modes are used only for demonstration checks, never by solve.
    """
    bg, channel, bead, gate, tip = info["palette"]
    h, w = a.shape
    out = a.copy()
    out[out == bead] = channel
    active = dict(info["stocks"])
    cursor = 0
    events, profiles, dormant = [], [], []
    for r, bars in sorted(info["layers"].items()):
        incoming = dict(active)
        for c in active:
            line = out[cursor:r + 2, c]
            original = a[cursor:r + 2, c]
            line[np.isin(original, [bg, channel, bead])] = channel
        hits = []
        following = dict(active)
        for center, preference in bars:
            packets = [(c, active[c]) for c in range(center - 1, center + 2) if c in active]
            mass = sum(n for _, n in packets)
            if not mass:
                dormant.append((r, center))
                if shortcut != "all_gates":
                    continue
            for c, _ in packets:
                following.pop(c, None)
            used_preference = -1 if shortcut == "fixed_left" else preference
            left, right = _division(mass, used_preference)
            if shortcut == "duplicate":
                left = right = mass
            elif shortcut == "no_pool":
                parts = [_division(n, used_preference) for _, n in packets]
                left = sum(n[0] for n in parts)
                right = sum(n[1] for n in parts)
            if shortcut == "no_rotation":
                # The unturned bar has width three: a B-style detour would
                # exit outside the original endpoints instead of using the
                # two channels opened by rotation.
                flanks = (center - 2, center + 2)
                span = slice(max(0, center - 2), min(w, center + 3))
                line = out[r - 1, span]
                line[a[r - 1, span] == bg] = channel
                if flanks[0] < 0:
                    right += left
                    left = 0
                if flanks[1] >= w:
                    left += right
                    right = 0
                for c, n in zip(flanks, (left, right)):
                    if n and 0 <= c < w:
                        out[r:r + 2, c] = channel
            else:
                flanks = (center - 1, center + 1)
                out[r - 2, center - 1:center + 2] = channel
                out[r - 1:r + 2, center - 1:center + 2] = channel
                out[r - 1:r + 2, center] = gate
                # Clockwise rigid rotation: left tip -> upstream; right tip
                # -> downstream.  The whole grid may have any orientation.
                out[r + preference, center] = tip
            for c, n in zip(flanks, (left, right)):
                if n and 0 <= c < w:
                    following[c] = following.get(c, 0) + n
            if mass:
                event = {"row": r, "center": center, "preference": preference,
                         "mass": mass, "packets": packets, "outgoing": (left, right)}
                events.append(event)
                hits.append(event)
        active = following
        if shortcut != "duplicate":
            _require(sum(active.values()) == info["total"], "mass lost at a gate layer")
        _require(all(n > 0 for n in active.values()), "an empty packet propagated")
        profiles.append({"row": r, "incoming": incoming, "outgoing": dict(active),
                         "hits": len(hits)})
        cursor = r + 2
    for c, n in active.items():
        out[cursor:, c] = channel
        # Clamping applies only to the intentionally mass-duplicating mutation.
        out[max(0, h - n):, c] = bead
    if shortcut is None:
        _require(np.count_nonzero(out == bead) == info["total"], "beads not conserved")
    stats = {"events": events, "profiles": profiles, "dormant": dormant,
             "terminal_stocks": dict(active), "total": info["total"],
             "layers": len(info["layers"]),
             "gates": sum(len(b) for b in info["layers"].values()),
             "two_sided": sum(min(e["outgoing"]) > 0 for e in events),
             "single": sum(e["mass"] == 1 for e in events),
             "pooling": sum(len(e["packets"]) > 1 for e in events),
             "simultaneous": sum(p["hits"] > 1 for p in profiles),
             "odd_left": sum(e["mass"] % 2 and e["preference"] == -1 for e in events),
             "odd_right": sum(e["mass"] % 2 and e["preference"] == 1 for e in events),
             "even": sum(e["mass"] % 2 == 0 for e in events)}
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from visible input geometry and bead counts."""
    canonical, turns, info = _recognize(input_grid)
    answer, _ = _simulate(canonical, info)
    return np.rot90(answer, -turns).copy()


def _blank(h, w, stocks):
    a = np.zeros((h, w), dtype=np.int8)
    for c, n in stocks.items():
        a[0, c] = a[n + 1, c] = 1
        a[1:n + 1, c] = 2
    return a


def _bar(a, r, c, preference):
    a[r, c - 1:c + 2] = 3
    a[r, c + preference] = 4


def _normal_info(a):
    info = _parse_canonical(a, 0, 1, 2, 3, 4)
    _require(info is not None, "invalid constructed geometry")
    return info


def _witness(rng, kind):
    """Two scalable local mechanisms, not a bank of complete fixed puzzles."""
    w = int(rng.integers(9, 21))
    center = int(rng.integers(3, w - 3))
    rows = [int(rng.integers(6, 9))]
    rows.append(rows[-1] + int(rng.integers(4, 6)))
    rows.append(rows[-1] + int(rng.integers(4, 6)))
    if kind == "extinction":
        stocks = {center: 3}
        s = int(rng.choice([-1, 1]))
        bars = [(rows[0], center, -s), (rows[1], center + 2 * s, s),
                (rows[2], center + s, int(rng.choice([-1, 1])))]
    else:
        # Odd + odd must be pooled before rounding.  The inward-facing middle
        # forks and final pooling keep the no-pool error visible at the exits.
        stocks = {center - 1: 1, center + 1: 3}
        if rng.random() < .5:
            stocks = {center - 1: 3, center + 1: 1}
        bars = [(rows[0], center, int(rng.choice([-1, 1]))),
                (rows[1], center - 2, 1), (rows[1], center + 2, -1),
                (rows[2], center, int(rng.choice([-1, 1])))]
    total = sum(stocks.values())
    h = min(30, rows[-1] + total + 2 + int(rng.integers(0, 4)))
    a = _blank(h, w, stocks)
    for r, c, preference in bars:
        _bar(a, r, c, preference)
    # Add unobstructive, optionally reached scenery, keeping the witnessed
    # central construction intact.  Same-sized dormant bars are not size cues.
    for r in rows:
        occupied = [c for y, c, _ in bars if y == r]
        if rng.random() < .5:
            options = [c for c in range(1, w - 1) if all(abs(c - d) >= 4 for d in occupied)
                       and abs(c - center) >= 5]
            if options:
                c = int(rng.choice(options))
                _bar(a, r, c, int(rng.choice([-1, 1])))
    return a


def _random_scene(rng, split, shift):
    """Sample independent stock, lateral geometry, layer spacing and density."""
    if split == "test" and shift == "depth":
        w, h = int(rng.integers(7, 31)), int(rng.integers(27, 31))
        entry_count = int(rng.integers(1, 3))
        budget = int(rng.integers(4, 6))
        cap_total = 5
    else:
        w, h = int(rng.integers(7, 31)), int(rng.integers(18, 31))
        entry_count = int(rng.integers(1, 3))
        if split == "test" and shift == "sources":
            entry_count = int(rng.integers(3, min(4, (w + 1) // 2) + 1))
        budget = int(rng.integers(1, 4))
        cap_total = min(9, h - 10)
    slots = list(map(int, rng.permutation(w)))
    entries = []
    for c in slots:
        if all(abs(c - d) >= 2 for d in entries):
            entries.append(c)
            if len(entries) == entry_count:
                break
    # Four separated entries always fit on width >=7; shuffled greedy placement
    # can leave fewer slots, so use a translated even lattice in that rare case.
    if len(entries) < entry_count:
        entries = list(range(0, 2 * entry_count, 2))
    stocks = {c: 1 for c in sorted(entries)}
    for _ in range(int(rng.integers(0, cap_total - len(entries) + 1))):
        candidates = [c for c in stocks if stocks[c] < 4]
        if candidates:
            stocks[int(rng.choice(candidates))] += 1
    total = sum(stocks.values())
    first = max(stocks.values()) + 3 + int(rng.integers(0, 3))
    h = max(h, first + total + 2)
    last = h - total - 2
    if split == "test" and shift == "depth":
        # At least four layers, each 4 cells apart, fit by construction.
        first = max(stocks.values()) + 3
    rows = [first]
    while len(rows) < budget:
        gap = 4 if split == "test" and shift == "depth" else int(rng.integers(4, 7))
        if rows[-1] + gap > last:
            break
        rows.append(rows[-1] + gap)
    a = _blank(h, w, stocks)
    active = dict(stocks)
    for r in rows:
        selected = []
        # Most layers have a causal gate; one can be entirely dormant.  For
        # depth transfer every layer has a collision, making the shift real.
        causal = split == "test" and shift == "depth" or rng.random() < .85
        if causal:
            for c in rng.permutation(sorted(active)):
                pivot = int(c)
                options = [d for d in range(max(1, pivot - 1), min(w - 2, pivot + 1) + 1)
                           if all(abs(d - q) >= 4 for q in selected)]
                if options:
                    selected.append(int(rng.choice(options)))
                if len(selected) >= int(rng.integers(1, 4)):
                    break
        quota = int(rng.integers(0, min(6, (w + 1) // 4) + 1))
        for c in rng.permutation(np.arange(1, w - 1)):
            c = int(c)
            if len(selected) >= max(1, quota):
                break
            if all(abs(c - d) >= 4 for d in selected):
                selected.append(c)
        following = dict(active)
        for c in sorted(selected):
            preference = int(rng.choice([-1, 1]))
            _bar(a, r, c, preference)
            mass = sum(active.get(d, 0) for d in range(c - 1, c + 2))
            if mass:
                for d in range(c - 1, c + 2):
                    following.pop(d, None)
                for d, n in zip((c - 1, c + 1), _division(mass, preference)):
                    if n:
                        following[d] = following.get(d, 0) + n
        active = following
    return a


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or high is not None and value > high:
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = np.array(rng.choice(10, 5, replace=False), dtype=np.int8)
    mixed_palette = bool(rng.random() < .55)
    rotation = int(rng.integers(4))
    shift = str(rng.choice(["depth", "sources", "orientation"]))
    witnesses = list(map(int, rng.permutation(nt)[:2]))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            kind = None
            if split == "train" and i in witnesses:
                kind = "extinction" if i == witnesses[0] else "pooling"
            for _ in range(60):
                a = _witness(rng, kind) if kind else _random_scene(rng, split, shift)
                info = _normal_info(a)
                o, stats = _simulate(a, info)
                if kind:
                    required = ("single", "odd_left", "odd_right") if kind == "extinction" else ("pooling", "simultaneous", "even")
                    if not all(stats[k] for k in required):
                        continue
                    if kind == "extinction" and not stats["dormant"]:
                        continue
                    if kind == "pooling" and np.array_equal(_simulate(a, info, "no_pool")[0], o):
                        continue
                colors = np.array(rng.choice(10, 5, replace=False), dtype=np.int8) if mixed_palette else palette
                turn = rotation
                if split == "test" and shift == "orientation":
                    turn = (rotation + int(rng.integers(1, 4))) % 4
                inp = np.rot90(colors[a], turn).copy()
                out = np.rot90(colors[o], turn).copy()
                key = (inp.shape, inp.tobytes())
                if key in seen:
                    continue
                canonical, undo, decoded = _recognize(inp)
                _require(np.array_equal(np.rot90(_simulate(canonical, decoded)[0], -undo), out),
                         "palette/orientation changed the rule")
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("bounded construction could not find a distinct valid scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, visible conventions, mass, local rotation and teaching."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, coverage = set(), defaultdict(int)
    rejected = {name: False for name in _SHORTCUTS}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, o = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == o.shape and not np.array_equal(a, o), "invalid output dimensions or identity")
            key = (a.shape, a.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            canonical, turns, info = _recognize(a)
            expected, stats = _simulate(canonical, info)
            actual = np.rot90(o, turns)
            _require(np.array_equal(actual, expected), "incorrect weighted rotary propagation")
            bg, channel, bead, gate, tip = info["palette"]
            _require(np.count_nonzero(actual == bead) == info["total"], "bead total changed")
            for color in (gate, tip):
                _require(np.count_nonzero(actual == color) == np.count_nonzero(canonical == color),
                         "gate material not preserved")
            _require(np.all(actual[canonical == channel] == channel), "source channel caps changed")
            _require(np.all(actual[canonical == bead] == channel), "reservoir beads were not released")
            for c, n in stats["terminal_stocks"].items():
                _require(np.all(actual[-n:, c] == bead), "stock not packed against downstream boundary")
                _require(actual[-n - 1, c] == channel, "packed stock lacks its feeding channel")
            reached = {(e["row"], e["center"]) for e in stats["events"]}
            for r, bars in info["layers"].items():
                for c, preference in bars:
                    if (r, c) in reached:
                        _require(actual[r + preference, c] == tip and actual[r, c] == gate,
                                 "rigid clockwise rotation incorrect")
                        _require(np.all(actual[r - 1:r + 2, [c - 1, c + 1]] == channel),
                                 "side channels were not opened")
                    else:
                        _require(np.array_equal(actual[r, c - 1:c + 2], canonical[r, c - 1:c + 2]),
                                 "unreached gate changed")
            _require(all(sum(p["incoming"].values()) == sum(p["outgoing"].values()) == info["total"]
                         for p in stats["profiles"]), "layer mass balance failed")
            if split == "train":
                for feature in ("two_sided", "single", "pooling", "simultaneous", "odd_left", "odd_right", "even"):
                    coverage[feature] += stats[feature]
                coverage["dormant"] += len(stats["dormant"])
                for name in rejected:
                    rejected[name] |= not np.array_equal(_simulate(canonical, info, name)[0], actual)
    _require(all(coverage.values()) and len(coverage) == 8,
             "training lacks a required rotary/weighted/causal witness")
    _require(all(rejected.values()), "training admits a known test-changing shortcut")


def _signature(puzzle):
    return tuple((split, p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed coverage plus explicit, manually specified regressions."""
    # One 3-bead packet: marked-right fork gives one left and two right.
    a = _blank(13, 7, {3: 3})
    _bar(a, 6, 3, 1)
    expected = a.copy()
    expected[:7, 3] = 1
    expected[4:8, 2:5] = 1
    expected[5:8, 3] = [3, 3, 4]
    expected[8:, [2, 4]] = 1
    expected[12, 2] = 2
    expected[11:, 4] = 2
    _require(np.array_equal(solve(a), expected), "manual fork/rotation/packing regression")
    # Input-only scale and role/orientation equivariance, including edge gates.
    b = _blank(12, 5, {0: 1, 2: 1})
    _bar(b, 5, 1, -1)
    ob = solve(b)
    _require(ob[-1, 0] == ob[-1, 2] == 2 and np.count_nonzero(ob == 2) == 2,
             "odd packets were rounded separately rather than pooled")
    # Valid grammar beyond the sampler's per-reservoir stock cap and source
    # count cap; solving must not depend on those generation ranges.
    large_stock = _blank(20, 7, {3: 6})
    _bar(large_stock, 9, 3, -1)
    large_answer = solve(large_stock)
    _require(np.all(large_answer[-3:, [2, 4]] == 2)
             and np.count_nonzero(large_answer == 2) == 6,
             "solver imposed the sampler's stock cap")
    many_sources = _blank(18, 9, {0: 1, 2: 1, 4: 1, 6: 1, 8: 1})
    _bar(many_sources, 6, 1, -1)
    _bar(many_sources, 6, 7, 1)
    _bar(many_sources, 10, 4, -1)
    many_answer = solve(many_sources)
    _require(set(map(int, np.flatnonzero(many_answer[-1] == 2))) == {0, 2, 3, 6, 8}
             and np.count_nonzero(many_answer == 2) == 5,
             "solver imposed the sampler's source cap")
    for rotation in range(4):
        mapping = np.array([9, 0, 7, 1, 4], dtype=np.int8)
        _require(np.array_equal(solve(np.rot90(mapping[a], rotation)),
                               np.rot90(mapping[expected], rotation)), "input-visible equivariance")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "seed diversity collapsed")
    shapes, normalized, palettes, orientations = set(), set(), set(), set()
    masses, source_counts, gate_counts, layer_counts, gaps = set(), set(), set(), set(), set()
    totals = {s: defaultdict(int) for s in ("train", "test")}
    shifted_orientation = shifted_depth = shifted_sources = 0
    for puzzle in puzzles:
        summaries = {}
        for split in ("train", "test"):
            summaries[split] = []
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                untouched = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(inp, untouched), "solve mismatch or mutation")
                canonical, turns, info = _recognize(inp)
                _, stats = _simulate(canonical, info)
                normal = np.zeros(canonical.shape, dtype=np.int8)
                for i, color in enumerate(info["palette"]):
                    normal[canonical == color] = i
                normalized.add((normal.shape, normal.tobytes()))
                shapes.add(inp.shape)
                palettes.add(info["palette"])
                orientations.add(turns)
                masses.add(info["total"])
                source_counts.add(len(info["stocks"]))
                gate_counts.add(stats["gates"])
                layer_counts.add(stats["layers"])
                rows = sorted(info["layers"])
                gaps.update(b - r for r, b in zip(rows, rows[1:]))
                totals[split]["edge_gate"] += any(c in (1, canonical.shape[1] - 2)
                                                   for bars in info["layers"].values() for c, _ in bars)
                totals[split]["dormant"] += len(stats["dormant"])
                for key in ("two_sided", "single", "pooling", "simultaneous", "odd_left", "odd_right", "even"):
                    totals[split][key] += stats[key]
                summaries[split].append((turns, stats["layers"], len(info["stocks"])))
        train, test = summaries["train"], summaries["test"]
        shifted_orientation += bool({x[0] for x in test} - {x[0] for x in train})
        shifted_depth += min(x[1] for x in test) > max(x[1] for x in train)
        shifted_sources += min(x[2] for x in test) > max(x[2] for x in train)
    _require(len(shapes) >= 55 and len(normalized) >= 180, "geometric diversity collapsed")
    _require(len(palettes) >= 60 and orientations == set(range(4)), "presentation diversity collapsed")
    _require(source_counts == {1, 2, 3, 4} and len(masses) >= 7, "stock variation collapsed")
    _require(len(gate_counts) >= 10 and {1, 2, 3, 4} <= layer_counts and {4, 5, 6} <= gaps,
             "fork forest variation collapsed")
    _require(all(v > 0 for d in totals.values() for v in d.values()), "missing structural feature in a split")
    _require(min(shifted_orientation, shifted_depth, shifted_sources) >= 3, "train/test shifts collapsed")
    _require(_signature(generate_puzzle(173, 6, 3)) == _signature(generate_puzzle(173, 6, 3)), "nondeterministic")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(101, nt, nq)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "explicit counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "normalized_inputs": len(normalized),
            "palettes": len(palettes), "stock_totals": sorted(masses), "source_counts": sorted(source_counts),
            "gate_counts": sorted(gate_counts), "layer_counts": sorted(layer_counts), "layer_gaps": sorted(gaps),
            "feature_events": {s: dict(d) for s, d in totals.items()},
            "shift_puzzles": {"orientation": shifted_orientation, "depth": shifted_depth, "sources": shifted_sources}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
