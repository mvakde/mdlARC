"""Chromatic rendezvous: identity agreement inside a branching transport network.

Only standard-library modules and NumPy are used. The solver reads all roles
and the network from its argument; it never reads a seed or construction data.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _read(input_grid):
    """Rotate the unique token edge to the top and decode the entry collar."""
    inp = _grid(input_grid)
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, 0])
        if np.all(g[0] == token) and np.count_nonzero(g == token) == g.shape[1]:
            candidates.append((g, k, token))
    if len(candidates) != 1:
        raise ValueError("input needs one unique full-edge token strip")
    g, rotation, token = candidates[0]
    h, w = g.shape
    if h < 6 or w < 5:
        raise ValueError("network is too small for the entry collar")
    bg = int(g[1, 0])
    if bg == token or not np.all(g[1] == bg):
        raise ValueError("the approach row must be uniform background")
    if not np.array_equal(g[2], g[3]):
        raise ValueError("guide studs must be two aligned cells long")
    obstacles = set(map(int, np.unique(g[4:]))) - {bg}
    if len(obstacles) != 1:
        raise ValueError("interior needs exactly one obstacle identity")
    barrier = next(iter(obstacles))
    guides = set(map(int, np.unique(g[2]))) - {bg}
    if not guides or guides & {token, barrier}:
        raise ValueError("guide, token, background and obstacle roles must differ")
    stud_cols = list(map(int, np.flatnonzero(g[2] != bg)))
    if any(b - a < 2 for a, b in zip(stud_cols, stud_cols[1:])):
        raise ValueError("guide studs must have background between them")
    matched, mismatched = {}, {}
    for c in range(1, w - 1):
        if g[2, c] != bg:
            continue
        left, right = int(g[2, c - 1]), int(g[2, c + 1])
        if left in guides and right in guides:
            if left == right:
                matched[c] = left
            else:
                mismatched[c] = (left, right)
    if not matched:
        raise ValueError("network needs at least one matching entrance")
    return dict(grid=g.copy(), rotation=rotation, token=token, background=bg,
                barrier=barrier, guides=guides, matched=matched,
                mismatched=mismatched)


def _runs(row, barrier):
    result = []
    c = 0
    while c < len(row):
        if row[c] != barrier:
            c += 1
            continue
        start = c
        while c + 1 < len(row) and row[c + 1] == barrier:
            c += 1
        result.append((start, c))
        c += 1
    return result


def _step(row, active, barrier, shortcut=None):
    """One simultaneous layer, using only immutable obstacle cells.

    First collect every incoming signal on each struck obstacle. Then combine
    pass-through and detour arrivals at free cells. Neither stage is ordered by
    source, color, or obstacle position. Wrong variants are QA ablations only.
    """
    w = len(row)
    runs = _runs(row, barrier)
    membership = {c: (a, b) for a, b in runs for c in range(a, b + 1)}
    hits, arrivals = defaultdict(list), defaultdict(list)
    for c, identity in sorted(active.items()):
        if c in membership:
            hits[membership[c]].append((c, identity))
        else:
            arrivals[c].append(identity)
    events = []
    for (a, b), incoming in sorted(hits.items()):
        identities = {identity for _, identity in incoming}
        agreed = len(identities) == 1
        enabled = agreed or shortcut in ("ignore_agreement", "ignore_bar_agreement")
        winner = min(identities)
        if shortcut == "bar_majority":
            frequencies = {identity: sum(v == identity for _, v in incoming) for identity in identities}
            winner = max(frequencies, key=frequencies.get)
            enabled = frequencies[winner] * 2 > len(incoming)
        if shortcut == "no_fork":
            enabled = False
        flanks = [c for c in (a - 1, b + 1) if 0 <= c < w]
        if enabled:
            identity = winner  # Exactly one identity in the true rule.
            if shortcut == "one_flank":
                flanks = flanks[:1]
            for c in flanks:
                arrivals[c].append(identity)
        events.append(dict(run=(a, b), incoming=tuple(incoming), agreed=agreed,
                           clipped=a == 0 or b == w - 1,
                           flanks=tuple(flanks), enabled=enabled))
    following, meetings = {}, []
    for c, values in sorted(arrivals.items()):
        identities = set(values)
        agreed = len(identities) == 1
        enabled = agreed or shortcut in ("ignore_agreement", "ignore_free_agreement")
        winner = min(identities)
        if shortcut == "free_majority":
            frequencies = {identity: values.count(identity) for identity in identities}
            winner = max(frequencies, key=frequencies.get)
            enabled = frequencies[winner] * 2 > len(values)
        if enabled:
            following[c] = winner
        if len(values) >= 2:
            meetings.append(dict(column=c, identities=tuple(sorted(identities)),
                                 arrivals=len(values), agreed=agreed))
    return following, events, meetings, runs


def _engine(analysis, shortcut=None):
    g = analysis["grid"]
    token, bg, barrier = (analysis[n] for n in ("token", "background", "barrier"))
    sources = dict(analysis["matched"])
    if shortcut == "ignore_entry":
        sources.update({c: colors[0] for c, colors in analysis["mismatched"].items()})
    out = g.copy()
    out[0, list(sources)] = bg
    active = dict(sources)
    history, reached, all_runs = [], set(), set()
    for r in range(4, g.shape[0]):
        following, events, meetings, runs = _step(g[r], active, barrier, shortcut)
        reached.update((r, *e["run"]) for e in events)
        all_runs.update((r, *run) for run in runs)
        history.append(dict(row=r, incoming=dict(active), events=events,
                            meetings=meetings, outgoing=dict(following)))
        active = following
    for c in active:
        _require(g[-1, c] == bg, "frontier must be free")
        out[-1, c] = token
    stats = dict(frontier=dict(active), sources=sources, history=history,
                 reached=reached, all_runs=all_runs,
                 untouched=all_runs - reached,
                 causal_layers=sum(bool(layer["events"]) for layer in history),
                 active_count=len(sources), inactive_count=len(analysis["mismatched"]))
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique answer from the input, without construction state."""
    a = _read(input_grid)
    out, _ = _engine(a)
    return np.ascontiguousarray(np.rot90(out, -a["rotation"]), dtype=np.int8)


def _features(stats):
    events = [e for layer in stats["history"] for e in layer["events"]]
    meetings = [m for layer in stats["history"] for m in layer["meetings"]]
    return {name for name, present in (
        ("fork", any(e["agreed"] and len(e["flanks"]) == 2 for e in events)),
        ("clipping", any(e["agreed"] and e["clipped"] for e in events)),
        ("same_bar", any(e["agreed"] and len(e["incoming"]) > 1 for e in events)),
        ("mixed_bar", any(not e["agreed"] for e in events)),
        ("equal_free", any(m["agreed"] for m in meetings)),
        ("mixed_free", any(not m["agreed"] for m in meetings)),
        ("simultaneous", any(len(layer["events"]) > 1 for layer in stats["history"])),
        ("singleton", any(e["run"][0] == e["run"][1] for e in events)),
        ("unreached", bool(stats["untouched"])),
        ("free_pass", any(layer["events"] and any(
            all(not e["run"][0] <= c <= e["run"][1] for e in layer["events"])
            for c in layer["incoming"]) for layer in stats["history"])),
    ) if present}


def _blank(h, w, roles):
    token, bg, _, _ = roles
    g = np.full((h, w), bg, dtype=np.int8)
    g[0] = token
    return g


def _put_gate(g, center, left, right):
    bg = g[1, 0]
    _require(np.all(g[2:4, [center - 1, center + 1]] == bg), "overlapping gates")
    g[2:4, center - 1] = left
    g[2:4, center + 1] = right


def _spaced_centers(rng, w, n):
    # Uniform combinations after reserving a minimum separation of four cells.
    choices = rng.choice(np.arange(2, w - 2 - 3 * (n - 1)), n, replace=False)
    return np.sort(choices) + np.arange(n) * 3


def _witness(rng, roles, kind, compact=False, extra_layer=False):
    """Two variable geometric witnesses, rather than a bank of exact puzzles."""
    token, bg, barrier, palette = roles
    if kind == 0:
        h = int(rng.integers(11, 18 if compact else 23))
        w = int(rng.integers(17, 25 if compact else 29))
        g = _blank(h, w, roles)
        p, q, t = map(int, _spaced_centers(rng, w, 3))
        x, y = map(int, rng.choice(palette, 2, replace=False))
        _put_gate(g, p, x, x)
        _put_gate(g, q, x, x)
        _put_gate(g, t, x, y)
        r1 = int(rng.integers(5, h - 4))
        r2 = int(rng.integers(r1 + 2, h - 1))
        left = int(rng.integers(1, p + 1))
        right = int(rng.integers(q, t - 1))
        a, b = left - 1, right + 1
        middle = int(rng.integers(a + 1, b))
        end = int(rng.integers(b, w - 1))
        g[r1, left:right + 1] = barrier
        g[r2, :middle] = barrier
        g[r2, middle + 1:end + 1] = barrier
        # This obstacle is geometrically ordinary but lies on an inactive ray.
        g[4, t:int(rng.integers(t + 1, w + 1))] = barrier
    else:
        h = int(rng.integers(12, 18 if compact else 24))
        w = int(rng.integers(21, 26 if compact else 30))
        g = _blank(h, w, roles)
        centers = _spaced_centers(rng, w - 2, 4) + np.array([1, 1, 3, 3])
        u, p, q, t = map(int, centers)
        x, y = map(int, rng.choice(palette, 2, replace=False))
        for center, identity in ((u, y), (p, x), (q, y)):
            _put_gate(g, center, identity, identity)
        _put_gate(g, t, x, y)
        # Two additional visible sources make 2:1 disagreements at BOTH kinds
        # of rendezvous. Shared studs teach unanimity rather than majority vote.
        v = u - 2
        g[2:4, u - 3] = y
        middle = p + 2
        g[2:4, p + 3] = x
        r1 = int(rng.integers(5, h - 6))
        r2 = int(rng.integers(r1 + 2, h - 4))
        r3 = int(rng.integers(r2 + 2, h - 1))
        left = int(rng.integers(u + 2, p + 1))
        right = int(rng.integers(q, t - 1))
        a, b = left - 1, right + 1
        g[r1, left:middle] = barrier
        g[r1, middle + 1:right + 1] = barrier
        # The two inward detours disagree at 'middle'. On the next layer,
        # the left outer detour and untouched sources u/v disagree on a bar.
        lo = int(rng.integers(0, v + 1))
        hi = int(rng.integers(a, middle - 1))
        g[r2, lo:hi + 1] = barrier
        g[r3, b] = barrier
        g[4, t:int(rng.integers(t + 1, w + 1))] = barrier
    if extra_layer:
        # Used only as a bounded spare for a deeper test network.
        # Insert a layer before the empty final edge, keeping dimensions legal.
        if h < 30:
            g = np.insert(g, h - 1, np.full(w, bg, dtype=np.int8), axis=0)
        a0 = _read(g)
        _, s0 = _engine(a0)
        pivot = sorted(s0["frontier"])[-1]
        g[-2, pivot] = barrier
    return g


def _random_gates(rng, g, palette):
    w = g.shape[1]
    m = int(rng.integers(4, min(12, (w + 1) // 2) + 1))
    slack = w - (2 * m - 1)
    gaps = np.full(m - 1, 2, dtype=int)
    protected = set(map(int, rng.choice(m - 1, 3, replace=False)))
    # Some rails share neighboring entrances; others are separated into groups.
    spare = [i for i in range(m - 1) if i not in protected]
    for i in rng.permutation(spare):
        delta = int(rng.integers(0, min(4, slack) + 1))
        gaps[int(i)] += delta
        slack -= delta
    start = int(rng.integers(0, slack + 1))
    positions = [start]
    for gap in gaps:
        positions.append(positions[-1] + int(gap))
    eligible = [i for i, gap in enumerate(gaps) if gap == 2]
    flags = {i: bool(rng.random() < .58) for i in range(m - 1)}
    ordered = list(map(int, rng.permutation(eligible)))
    flags[ordered[0]] = flags[ordered[1]] = True
    flags[ordered[2]] = False
    identity = int(rng.choice(palette))
    colors = [identity]
    for i in range(m - 1):
        if not flags[i]:
            identity = int(rng.choice([c for c in palette if c != identity]))
        colors.append(identity)
    for c, identity in zip(positions, colors):
        g[2:4, c] = identity


def _paint_interval(rng, row, pivot, barrier):
    w = len(row)
    # Overlap and touching are allowed: the visible maximal run is authoritative.
    probability = rng.random()
    if probability < .3:
        n = 1
    elif probability < .65:
        n = int(rng.integers(2, min(w, 6) + 1))
    elif probability < .85:
        n = int(rng.integers(max(1, w // 2), w + 1))
    else:
        n = int(rng.integers(1, w + 1))
    lo = int(rng.integers(max(0, pivot - n + 1), min(pivot, w - n) + 1))
    row[lo:lo + n] = barrier


def _random_scene(rng, roles, shift, is_test, compact):
    token, bg, barrier, palette = roles
    if is_test and shift == "scale":
        h, w = int(rng.integers(24, 31)), int(rng.integers(22, 31))
    else:
        h = int(rng.integers(12 if compact else 10, 19 if compact else 31))
        w = int(rng.integers(13, 26 if compact else 31))
    if is_test and shift == "cascade":
        h = max(h, 13)
    g = _blank(h, w, roles)
    _random_gates(rng, g, palette)
    active = dict(_read_collar(g, roles))
    # Leave a clear far edge. Event count is not prescribed by canvas size.
    budget = int(rng.integers(4 if is_test and shift == "cascade" else 1,
                              min(9, h - 5) + 1))
    event_rows = set(map(int, rng.choice(np.arange(4, h - 1), budget, replace=False)))
    for r in range(4, h - 1):
        if r in event_rows and active:
            pivots = list(map(int, rng.permutation(sorted(active))))
            quota = int(rng.integers(1, min(3, len(pivots)) + 1))
            for c in pivots[:quota]:
                _paint_interval(rng, g[r], c, barrier)
        # Independent scenery sometimes contacts active rays; its true causal
        # status is determined by the finished input, never by this sampling name.
        if rng.random() < .22:
            _paint_interval(rng, g[r], int(rng.integers(w)), barrier)
        active, _, _, _ = _step(g[r], active, barrier)
    # Connected forward stems and elbows can be reached or inert; adding them
    # does not consult intended labels. All outputs are subsequently re-read.
    for _ in range(int(rng.integers(0, 4))):
        c = int(rng.integers(w))
        r = int(rng.integers(4, h - 1))
        end = int(rng.integers(r + 1, min(h, r + 5)))
        g[r:end, c] = barrier
        if rng.random() < .5:
            delta = int(rng.choice([-1, 1]))
            other = min(w - 1, max(0, c + delta * int(rng.integers(1, 5))))
            g[end - 1, min(c, other):max(c, other) + 1] = barrier
    return g


def _read_collar(g, roles):
    # Constructor convenience; it has no role in solve and uses no latent gates.
    _, bg, _, palette = roles
    return {c: int(g[2, c - 1]) for c in range(1, g.shape[1] - 1)
            if g[2, c] == bg and int(g[2, c - 1]) in palette
            and g[2, c - 1] == g[2, c + 1]}


def _observable_a(g):
    a = _read(g)
    correct, stats = _engine(a)
    wrong, _ = _engine(a, "no_fork")
    return stats["causal_layers"] > 0 and not np.array_equal(correct, wrong)


def _make_pair(rng, roles, kind, shift, compact, rotation, seen):
    for attempt in range(160):
        if kind in (0, 1):
            g = _witness(rng, roles, kind, compact)
        else:
            g = _random_scene(rng, roles, shift, kind == "test", compact)
        if rng.random() < .5:
            g = np.fliplr(g).copy()
        try:
            a = _read(g)
        except ValueError:
            continue
        correct, stats = _engine(a)
        if not a["mismatched"] or not _observable_a(g):
            continue
        if kind == "test" and shift == "cascade" and stats["causal_layers"] < 4:
            continue
        inp = np.ascontiguousarray(np.rot90(g, rotation), dtype=np.int8)
        if _signature(inp) in seen:
            continue
        return {"input": inp,
                "output": np.ascontiguousarray(np.rot90(correct, rotation), dtype=np.int8)}
    # Finite spare procedure, still governed by the same rule. A random witness
    # already has vast variability, and a fourth causal layer can be inserted.
    for serial in range(64):
        if kind == "test" and shift == "cascade":
            base_kind = 1
        else:
            base_kind = int(rng.integers(0, 2)) if kind not in (0, 1) else kind
        g = _witness(rng, roles, base_kind, compact=False,
                     extra_layer=kind == "test" and shift == "cascade")
        if kind == "test" and shift == "scale":
            # Padding keeps the source edge and its collar intact.
            g = np.vstack([g[:-1], np.full((30 - g.shape[0], g.shape[1]),
                                          roles[1], dtype=np.int8), g[-1:]])
            if g.shape[1] < 22:
                g = np.pad(g, ((0, 0), (0, 22 - g.shape[1])),
                           constant_values=roles[1])
                g[0] = roles[0]
        if serial % 2:
            g = np.fliplr(g).copy()
        inp = np.ascontiguousarray(np.rot90(g, rotation), dtype=np.int8)
        if _signature(inp) not in seen:
            a = _read(inp)
            correct, stats = _engine(a)
            if kind == "test" and shift == "cascade" and stats["causal_layers"] < 4:
                continue
            return {"input": inp, "output": np.rot90(correct, -a["rotation"]).copy()}
    raise RuntimeError("bounded distinct-network construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["cascade", "scale", "orientation"]))
    mixed_palette = bool(rng.integers(2))
    colors = tuple(map(int, rng.permutation(10)))
    size = int(rng.integers(2, 8))
    rotation = int(rng.integers(4))
    mixed_orientation = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            local = tuple(map(int, rng.permutation(10))) if mixed_palette else colors
            local_size = int(rng.integers(2, 8)) if mixed_palette else size
            roles = (local[0], local[1], local[2], local[3:3 + local_size])
            k = rotation
            if shift == "orientation":
                k = (rotation + (i % 2 if split == "train" else 2 + i % 2)) % 4
            elif mixed_orientation:
                k = (rotation + i + int(split == "test")) % 4
            kind = i if split == "train" and i < 2 else split
            pair = _make_pair(rng, roles, kind, shift, shift == "scale" and split == "train",
                              k, seen)
            puzzle[split].append(pair)
            seen.add(_signature(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, covered = set(), set()
    discriminated = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape preservation")
            signature = _signature(inp)
            _require(signature not in seen, "duplicate inputs")
            seen.add(signature)
            a = _read(inp)
            g = a["grid"]
            canonical_out = np.rot90(out, a["rotation"])
            expected, stats = _engine(a)
            _require(np.array_equal(canonical_out, expected), "incorrect agreement network answer")
            _require(a["matched"] and a["mismatched"], "match/mismatch contrast required")
            _require(_observable_a(inp), "A's obstacle detours must change the answer")
            _require(np.array_equal(canonical_out[1:-1], g[1:-1]), "interior must be preserved")
            _require(np.all(canonical_out[-1, g[-1] != a["background"]] ==
                            g[-1, g[-1] != a["background"]]), "last-layer obstacles preserved")
            _require(np.count_nonzero(inp != out) == len(a["matched"]) + len(stats["frontier"]),
                     "only selected sources and surviving terminals change")
            _require(np.count_nonzero(out == a["token"]) ==
                     g.shape[1] - len(a["matched"]) + len(stats["frontier"]),
                     "token count must equal inactive strip plus unique surviving frontier")
            wrong, _ = _engine(a, "ignore_entry")
            _require(not np.array_equal(wrong, expected), "B's rail equality must change the answer")
            features = _features(stats)
            if split == "train":
                covered.update(features)
                for shortcut in ("no_fork", "one_flank", "ignore_entry", "ignore_agreement",
                                 "ignore_bar_agreement", "ignore_free_agreement",
                                 "bar_majority", "free_majority"):
                    wrong, _ = _engine(a, shortcut)
                    if not np.array_equal(wrong, expected):
                        discriminated.add(shortcut)
            else:
                _require(features <= covered, "test introduces an untaught interaction")
    _require({"fork", "clipping", "same_bar", "mixed_bar", "equal_free", "mixed_free",
              "simultaneous", "singleton", "unreached", "free_pass"} <= covered,
             "training lacks a required network witness")
    _require({"no_fork", "one_flank", "ignore_entry", "ignore_agreement",
              "ignore_bar_agreement", "ignore_free_agreement", "bar_majority",
              "free_majority"} <= discriminated,
             "training admits a simpler tested alternative")


def measurements(puzzle):
    """Actual-grid measurements for bounded self-tests and scratch inspection."""
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _read(pair["input"])
            _, s = _engine(a)
            events = [e for layer in s["history"] for e in layer["events"]]
            records.append(dict(split=split, shape=list(a["grid"].shape),
                                rotation=a["rotation"],
                                roles=[a["token"], a["background"], a["barrier"]],
                                guide_count=len(a["guides"]), active=s["active_count"],
                                inactive=s["inactive_count"], terminals=len(s["frontier"]),
                                causal_layers=s["causal_layers"], reached=len(s["reached"]),
                                untouched=len(s["untouched"]),
                                hit_lengths=[e["run"][1] - e["run"][0] + 1 for e in events],
                                features=sorted(_features(s))))
    return records


def _fixture(mixed):
    roles = (9, 0, 5, (2, 3))
    g = _blank(11, 19 if mixed else 17, roles)
    if mixed:
        for c, identity in ((1, 3), (6, 2), (10, 3)):
            _put_gate(g, c, identity, identity)
        _put_gate(g, 14, 2, 3)
        g[5, 5:8] = g[5, 9:12] = 5
        g[7, 1:5] = 5
        frontier = {12}
        sources = {1, 6, 10}
    else:
        _put_gate(g, 4, 2, 2)
        _put_gate(g, 8, 2, 2)
        _put_gate(g, 12, 2, 3)
        g[5, 3:10] = 5
        g[7, :6] = g[7, 7:15] = 5
        frontier = {6, 15}
        # Shared matching studs also visibly select centers 6 and 10.
        sources = {4, 6, 8, 10}
    expected = g.copy()
    expected[0, sorted(sources)] = 0
    expected[-1, sorted(frontier)] = 9
    return g, expected


def self_test():
    """Bounded 42-seed variation, exact local fixtures and metamorphic checks."""
    records, puzzles, signatures, normalized = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        records.extend(measurements(puzzle))
        sig = tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s])
        signatures.add(sig)
        again = generate_puzzle(seed)
        _require(all(np.array_equal(p[k], q[k]) for s in puzzle
                     for p, q in zip(puzzle[s], again[s]) for k in ("input", "output")), "determinism")
        for s in ("train", "test"):
            for pair in puzzle[s]:
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "input-only solve")
                _require(np.array_equal(before, pair["input"]), "solver mutated its argument")
                a = _read(before)
                names = {a["background"]: 0, a["token"]: 1, a["barrier"]: 2}
                rows = tuple(tuple(names.setdefault(int(c), len(names)) for c in row)
                             for row in a["grid"])
                normalized.add(rows)
    _require(len(signatures) == 42, "42 seeds must yield distinct puzzles")
    _require(len(normalized) >= 150, "structural diversity collapsed")
    _require(len({tuple(r["shape"]) for r in records}) >= 50, "shape diversity collapsed")
    _require({r["rotation"] for r in records} == set(range(4)), "orientation coverage")
    for i in range(3):
        _require({r["roles"][i] for r in records} == set(range(10)), "literal role coverage")
    for field in ("active", "inactive", "terminals", "causal_layers", "guide_count"):
        _require(len({r[field] for r in records}) >= 4, f"{field} diversity collapsed")
    _require(len({n for r in records for n in r["hit_lengths"]}) >= 15, "run lengths collapsed")
    for mixed in (False, True):
        inp, expected = _fixture(mixed)
        _require(np.array_equal(solve(inp.tolist()), expected), "exact rendezvous fixture")
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(inp, k)), np.rot90(expected, k)), "rotation equivariance")
        permutation = np.random.default_rng(880 + int(mixed)).permutation(10)
        _require(np.array_equal(solve(permutation[inp]), permutation[expected]), "color equivariance")
    # One entrance identity can change a distant collector without moving a bar.
    inp, _ = _fixture(True)
    inp[2:4, [0, 2]] = 2
    expected = inp.copy()
    expected[0, [1, 6, 10]] = 0
    expected[-1, [0, 5, 12]] = 9
    _require(np.array_equal(solve(inp), expected), "identity handoff at a collector")
    # Unanimity is not a numerical vote, at either kind of collector.
    row = np.zeros(11, dtype=np.int8)
    row[2:7] = 5
    active, events, _, _ = _step(row, {2: 2, 4: 2, 6: 3}, 5)
    _require(not active and not events[0]["agreed"], "majority must not pass a bar")
    row[:] = 0
    row[2:5] = row[6:9] = 5
    active, _, meetings, _ = _step(row, {3: 2, 5: 2, 7: 3}, 5)
    _require(active == {1: 2, 9: 3} and not meetings[0]["agreed"],
             "majority must not pass a free rendezvous")
    # General solver support beyond the sampler's clear far edge.
    inp, _ = _fixture(False)
    inp[-1, 6] = 5
    expected = inp.copy()
    expected[0, [4, 6, 8, 10]] = 0
    expected[-1, [5, 7, 15]] = 9
    _require(np.array_equal(solve(inp), expected), "final-row singleton detour")
    inp[-1] = 5
    expected = inp.copy()
    expected[0, [4, 6, 8, 10]] = 0
    _require(np.array_equal(solve(inp), expected), "spanning run and total extinction")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(913, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a false terminal pattern")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                normalized_structures=len(normalized),
                canonical_shapes=len({tuple(r["shape"]) for r in records}),
                active_counts=sorted({r["active"] for r in records}),
                terminal_counts=sorted({r["terminals"] for r in records}),
                causal_depths=sorted({r["causal_layers"] for r in records}),
                guide_counts=sorted({r["guide_count"] for r in records}))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
