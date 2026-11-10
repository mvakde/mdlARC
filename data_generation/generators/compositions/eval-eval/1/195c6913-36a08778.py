"""Phase-keyed shutters: cyclic distance changes causal branching geometry.

Only NumPy and the standard library are used. No parent modules or saved puzzles
are imported. Canonical coordinates put a framed program tape above the arena.
"""
from __future__ import annotations

import numpy as np


def _ensure(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _ensure(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _ensure(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _ensure(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _orient(a, code):
    return np.rot90(a[:, ::-1] if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (b[:, ::-1] if code >= 4 else b).copy()


def _runs(arena, background):
    runs = []
    for r in range(1, arena.shape[0]):
        c = 0
        while c < arena.shape[1]:
            if arena[r, c] == background:
                c += 1
                continue
            left = c
            color = int(arena[r, c])
            while c + 1 < arena.shape[1] and arena[r, c + 1] != background:
                c += 1
            _ensure(bool(np.all(arena[r, left:c + 1] == color)), "shutters must be monochrome")
            runs.append((r, left, c, color))
            c += 1
    return runs


def _decode(value):
    """The L frame, rather than color frequency or a seed, defines orientation."""
    grid = _grid(value)
    candidates = []
    for code in range(8):
        a = _unorient(grid, code)
        h, w = a.shape
        if h < 9 or w < 8:
            continue
        frame, bg = int(a[4, 0]), int(a[0, 1])
        if frame == bg or not np.all(a[4] == frame) or not np.all(a[:5, 0] == frame):
            continue
        if not np.all(a[0, 1:] == bg) or not np.all(a[3, 1:] == bg):
            continue
        if not np.array_equal(a[1, 1:], a[2, 1:]):
            continue
        word, swatches = [], []
        c = 1
        valid = True
        while c < w:
            if a[1, c] == bg:
                c += 1
                continue
            if (c + 2 >= w or a[1, c] == frame or a[1, c + 1] != a[1, c]
                    or a[1, c + 2] != bg or c < 2 or a[1, c - 1] != bg):
                valid = False
                break
            word.append(int(a[1, c]))
            swatches.append(c)
            c += 2
        if not valid or not 2 <= len(word) <= 6 or not 2 <= len(set(word)) <= 5:
            continue
        arena = a[5:]
        if frame in arena or not set(map(int, np.unique(arena))) <= set(word) | {bg}:
            continue
        sources = tuple(map(int, np.flatnonzero(arena[0] != bg)))
        if not 1 <= len(sources) <= 4 or any(arena[0, s] != word[0] for s in sources):
            continue
        if any(b - x < 2 for x, b in zip(sources, sources[1:])):
            continue
        try:
            runs = _runs(arena, bg)
        except ValueError:
            continue
        if not runs:
            continue
        candidates.append(dict(canonical=a, arena=arena, background=bg, frame=frame,
                               word=tuple(word), sources=sources, swatches=swatches,
                               runs=runs, orientation=code))
    _ensure(len(candidates) == 1, "input must have exactly one framed program interpretation")
    return candidates[0]


def _trace(arena, background, word, sources, conflict_color, mode="correct"):
    """Forward layers carry phase states; transverse legs carry their own delays.

    Incorrect modes are used only to check the information in demonstrations.
    A rejected construction can return multiple phases at a cell; legal family
    inputs require exactly one. No propagation decision reads painted output.
    """
    if mode == "constant":
        word = (word[0],)
    elif mode == "deduplicate":
        word = tuple(dict.fromkeys(word))
    n = len(word)
    h, w = arena.shape
    obstacles = arena != background
    obstacles[0] = False
    run_at = {}
    all_runs = _runs(arena, background)
    for r, a, b, color in all_runs:
        for c in range(a, b + 1):
            run_at[r, c] = (a, b, color)
    phases, incoming, events = {}, [], []
    active = {(s, 0) for s in sources}
    merges = 0
    blocked = False

    def paint(r, c, phase):
        phases.setdefault((r, c), set()).add(phase % n)

    for r in range(h):
        incoming.append(tuple(sorted(active)))
        following = set()

        def forward(c, p):
            nonlocal merges
            state = (c, p % n)
            merges += state in following
            following.add(state)

        for c, p in sorted(active):
            paint(r, c, p)
            if r + 1 == h:
                continue
            q = (p + 1) % n
            if not obstacles[r + 1, c]:
                forward(c, q)
                continue
            a, b, color = run_at[r + 1, c]
            transmit = word[q] == color and mode != "always_split"
            events.append((r + 1, a, b, c, q, "pass" if transmit else "split"))
            if transmit:
                forward(c, q)
                continue
            if mode == "stop":
                continue
            # The blocked forward cell consumes no phase. Each fork starts
            # from the already-painted incident elbow, counted just once.
            base = p + (mode == "consume_collision")
            for end, step in ((a - 1, -1), (b + 1, 1)):
                for x in range(c + step, end + step, step):
                    if not 0 <= x < w:
                        break
                    distance = abs(x - c)
                    side_phase = base + distance
                    if mode == "reset_turn":
                        side_phase = distance - 1
                    if obstacles[r, x]:
                        blocked = True
                    paint(r, x, side_phase)
                if 0 <= end < w:
                    next_phase = base + abs(end - c) + 1
                    if mode == "reset_turn":
                        next_phase = 0
                    elif mode == "no_lateral_delay":
                        next_phase = p + 1
                    forward(end, next_phase)
        active = following

    out = arena.copy()
    path = np.zeros(arena.shape, dtype=bool)
    conflict = any(len(ps) != 1 for ps in phases.values())
    for (r, c), ps in phases.items():
        if not obstacles[r, c]:
            out[r, c] = word[next(iter(ps))] if len(ps) == 1 else conflict_color
            path[r, c] = True
    reached = {(r, a, b) for r, a, b, _, _, _ in events}
    passed = {(r, a, b) for r, a, b, _, _, kind in events if kind == "pass"}
    split = {(r, a, b) for r, a, b, _, _, kind in events if kind == "split"}
    stats = dict(valid=not conflict and not blocked, conflict=conflict, blocked=blocked,
                 phases=phases, path=path, incoming=incoming, events=events,
                 reached=reached, passed=passed, split=split, merges=int(merges),
                 clipped=any(a == 0 or b == w - 1 for _, a, b in split),
                 layers=len({r for r, _, _ in reached}),
                 inert={(r, a, b) for r, a, b, _ in all_runs} - reached,
                 max_distance=max((r + c for r, c in phases), default=0))
    return out, stats


def _analyze(input_grid, mode="correct"):
    info = _decode(input_grid)
    arena_out, stats = _trace(info["arena"], info["background"], info["word"],
                              info["sources"], info["frame"], mode)
    out = info["canonical"].copy()
    for c in info["swatches"]:
        out[1:3, c:c + 2] = info["background"]
    out[5:] = arena_out
    return _orient(out, info["orientation"]), info, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    output, _, stats = _analyze(input_grid)
    _ensure(stats["valid"], "family excludes conflicting phases and blocked transverse detours")
    return output


def _evidence(info, stats, arena_out):
    evidence = set()
    if stats["passed"]:
        evidence.add("transmission")
    if stats["split"]:
        evidence.add("branching")
    if stats["inert"]:
        evidence.add("causality")
    # A source's unobstructed opening is sampled long enough to show reuse.
    first = min((r for r, _, _ in stats["reached"]), default=0)
    if first >= 2 * len(info["word"]):
        evidence.add("cycling")
    for mode in ("constant", "deduplicate", "always_split", "stop",
                 "reset_turn", "consume_collision", "no_lateral_delay"):
        wrong, ws = _trace(info["arena"], info["background"], info["word"],
                           info["sources"], info["frame"], mode)
        if not np.array_equal(wrong, arena_out):
            evidence.add(mode)
        if not np.array_equal(ws["path"], stats["path"]):
            evidence.add(mode + "_geometry")
    return evidence


_REQUIRED = {"transmission", "branching", "causality", "cycling", "constant_geometry",
             "deduplicate", "always_split_geometry", "stop_geometry", "reset_turn",
             "consume_collision", "no_lateral_delay_geometry"}


def _word(rng, repeat):
    length = int(rng.integers(3 if repeat else 2, 7))
    distinct = int(rng.integers(2, min(5, length - int(repeat)) + 1))
    word = list(range(distinct)) + list(map(int, rng.integers(distinct, size=length - distinct)))
    rng.shuffle(word)
    return tuple(word)


def _interval(rng, width, pivot):
    p = rng.random()
    if p < .2:
        length = 1
    elif p < .65:
        length = int(rng.integers(2, min(width, 7) + 1))
    elif p < .8:
        length = int(rng.integers(max(2, width - 5), width + 1))
    else:
        length = int(rng.integers(1, width + 1))
    low, high = max(0, pivot - length + 1), min(pivot, width - length)
    left = int(rng.integers(low, high + 1))
    return left, left + length - 1


def _construct(rng, word, test, witness, allowed):
    n = len(word)
    layers = int(rng.integers(4, 7) if test else rng.integers(2, 4))
    # Long openings teach actual repetition; subsequent spacings are irregular.
    opening = 2 * n if witness == "keys" else int(rng.integers(2, n + 3))
    depth_low = max(2 * layers + opening, 16 if test else 11)
    if depth_low > 25:
        return None
    depth = int(rng.integers(depth_low, 26 if test else max(depth_low + 1, 22)))
    width = int(rng.integers(max(11, 3 * n + 2), 31))
    arena = np.full((depth, width), -1, dtype=np.int8)
    count = 1 if witness == "phase" else int(rng.choice([1, 1, 1, 2, 2, 3]))
    slots = list(map(int, rng.permutation(width)))
    sources = []
    for c in slots:
        if all(abs(c - x) >= 3 for x in sources):
            sources.append(c)
            if len(sources) == count:
                break
    sources = tuple(sorted(sources))
    arena[0, list(sources)] = word[0]
    pool = np.arange(opening, depth - layers + 1)
    if len(pool) < layers:
        return None
    rows = np.sort(rng.choice(pool, size=layers, replace=False)) + np.arange(layers)
    trace = lambda a: _trace(a, -1, word, sources, -2)
    for index, row_value in enumerate(rows):
        row = int(row_value)
        _, before = trace(arena)
        states = list(before["incoming"][row - 1])
        if not states:
            return None
        rng.shuffle(states)
        quota = 1 if witness else int(rng.integers(1, min(3, len(states)) + 1))
        placed = 0
        for c, phase in states:
            if placed == quota:
                break
            for attempt in range(28):
                left, right = _interval(rng, width, c)
                if np.any(arena[row, max(0, left - 1):min(width, right + 2)] != -1):
                    continue
                match = (index == 1) if witness else rng.random() < .48
                if witness and index > 1:
                    match = rng.random() < .55
                key = word[(phase + 1) % n]
                color = key if match else int(rng.choice([x for x in sorted(set(word)) if x != key]))
                candidate = arena.copy()
                candidate[row, left:right + 1] = color
                _, after = trace(candidate)
                if not after["valid"]:
                    continue
                if allowed is not None and ((after["clipped"] and "clipped" not in allowed)
                                            or (after["merges"] and "merges" not in allowed)):
                    continue
                arena = candidate
                placed += 1
                break
        if not placed:
            return None
    # Equal-looking but unreached shutters teach causal rather than global fill.
    desired = int(rng.integers(1 if witness else 0, 4))
    for _ in range(desired):
        for attempt in range(45):
            row = int(rng.choice(rows))
            _, before = trace(arena)
            c = int(rng.integers(width))
            left, right = _interval(rng, width, c)
            if np.any(arena[row, max(0, left - 1):min(width, right + 2)] != -1):
                continue
            if any(left <= x <= right for x, _ in before["incoming"][row - 1]):
                continue
            color = int(rng.choice(word))
            candidate = arena.copy()
            candidate[row, left:right + 1] = color
            _, after = trace(candidate)
            if after["valid"]:
                arena = candidate
                break
    out, stats = trace(arena)
    if not stats["valid"] or stats["layers"] != layers:
        return None
    return arena, out, sources, stats


def _make_pair(rng, test, witness, orientation, palette, allowed=None):
    for attempt in range(260):
        word = _word(rng, repeat=bool(witness))
        scene = _construct(rng, word, test, witness, allowed)
        if scene is None:
            continue
        arena, arena_out, sources, stats = scene
        info = dict(arena=arena, background=-1, word=word, sources=sources, frame=-2)
        evidence = _evidence(info, stats, arena_out)
        needed = ({"transmission", "branching", "cycling", "constant_geometry", "always_split_geometry"}
                  if witness == "keys" else
                  {"causality", "deduplicate", "reset_turn", "consume_collision",
                   "no_lateral_delay_geometry", "stop_geometry"} if witness == "phase" else set())
        if not needed <= evidence:
            continue
        if test:
            if not {"transmission", "branching", "no_lateral_delay_geometry"} <= evidence:
                continue
            if len({r for r, _, _ in stats["split"]}) < 2:
                continue
        h, w = arena.shape
        bg, frame = palette[:2]
        mapping = palette[2:]
        grid = np.full((h + 5, w), bg, dtype=np.int8)
        grid[4] = frame
        grid[:5, 0] = frame
        slack = w - (3 * len(word) + 2)
        left = 2 + int(rng.integers(min(4, slack) + 1))
        positions = []
        for i in range(len(word)):
            positions.append(left)
            left += 3
        remaining = w - 1 - (positions[-1] + 2)
        # Expand selected gaps without changing reading order or swatch size.
        for i in range(1, len(positions)):
            extra = int(rng.integers(min(2, max(0, remaining)) + 1))
            for j in range(i, len(positions)):
                positions[j] += extra
            remaining -= extra
        for c, symbol in zip(positions, word):
            grid[1:3, c:c + 2] = mapping[symbol]
        for r, c in np.argwhere(arena != -1):
            grid[r + 5, c] = mapping[int(arena[r, c])]
        output = grid.copy()
        for c in positions:
            output[1:3, c:c + 2] = bg
        output[5:] = bg
        for r, c in np.argwhere(arena_out != -1):
            output[r + 5, c] = mapping[int(arena_out[r, c])]
        pair = {"input": _orient(grid, orientation), "output": _orient(output, orientation)}
        # Verify parsing on the actual visible palette and frame, not metadata.
        decoded_out, _, observed = _analyze(pair["input"])
        _ensure(observed["valid"] and np.array_equal(decoded_out, pair["output"]), "encoding defect")
        return pair, stats, evidence
    raise RuntimeError("bounded scene construction exhausted")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _ensure(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _ensure(_integer(num_train) and _integer(num_test), "counts must be integers")
    _ensure(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test,
            "counts must satisfy 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    base_orientation = int(rng.integers(8))
    mixed_orientation = rng.random() < .45
    mixed_palette = rng.random() < .65
    base_palette = tuple(map(int, rng.permutation(10)))
    witness_order = ["keys", "phase"]
    rng.shuffle(witness_order)
    puzzle = {"train": [], "test": []}
    seen, covered, allowed = set(), set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = witness_order[i] if split == "train" and i < 2 else None
            orientation = int(rng.integers(8)) if mixed_orientation else base_orientation
            palette = tuple(map(int, rng.permutation(10))) if mixed_palette else base_palette
            for retry in range(12):
                pair, stats, evidence = _make_pair(rng, split == "test", witness, orientation,
                                                  palette, allowed if split == "test" else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                covered |= evidence
                if stats["clipped"]:
                    allowed.add("clipped")
                if stats["merges"]:
                    allowed.add("merges")
    _ensure(_REQUIRED <= covered, "insufficient training evidence")
    return puzzle


def validate_generated_puzzle(puzzle):
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _ensure(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _ensure(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, evidence, allowed = set(), set(), set()
    train_layers = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _ensure(inp.shape == out.shape, "output shape changed")
            _ensure(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            expected, info, stats = _analyze(inp)
            _ensure(stats["valid"], "conflicting phases or blocked crosspiece")
            _ensure(np.array_equal(expected, out), "output disagrees with input-only rule")
            canonical_out = _unorient(out, info["orientation"])
            a = info["canonical"]
            # Every scene object and frame pixel survives; only blank arena
            # cells are painted, and only program swatches are erased.
            objects = info["arena"] != info["background"]
            _ensure(np.array_equal(canonical_out[5:][objects], info["arena"][objects]), "object overwritten")
            _ensure(np.array_equal(canonical_out[a == info["frame"]], a[a == info["frame"]]), "frame changed")
            _ensure(np.all(np.isin(canonical_out[5:], [info["background"], *info["word"]])), "unexpected paint")
            ev = _evidence(info, stats, canonical_out[5:])
            if split == "train":
                evidence |= ev
                train_layers.append(stats["layers"])
                if stats["clipped"]:
                    allowed.add("clipped")
                if stats["merges"]:
                    allowed.add("merges")
            else:
                _ensure(stats["layers"] > max(train_layers), "test must extend causal depth")
                _ensure(not stats["clipped"] or "clipped" in allowed, "clipping lacks demonstration")
                _ensure(not stats["merges"] or "merges" in allowed, "merging lacks demonstration")
                _ensure(bool(stats["passed"] and stats["split"]), "test needs transmission and branching")
                _ensure(len({r for r, _, _ in stats["split"]}) >= 2, "test needs repeated branching")
                _ensure("no_lateral_delay_geometry" in ev, "test must couple phase to later geometry")
    _ensure(_REQUIRED <= evidence, "demonstrations admit a tested simpler alternative")
    return True


def self_test():
    """Bounded semantic fixtures, 42-seed variation, and API/count regressions."""
    # Hand-calculated fork: row 2's A mismatches next B in AAB. A blocked
    # attempted cell costs nothing; the right flank is one step farther away.
    arena = np.full((8, 11), -1, dtype=np.int8)
    arena[0, 4] = 0
    arena[2, 3:7] = 0
    out, s = _trace(arena, -1, (0, 0, 1), (4,), -2)
    # At the elbow (1,4), phase 1; left flank col 2 has phase 1 at row 2,
    # while right flank col 7 has phase 2. Original shutter cells survive.
    assert s["valid"] and out[2, 2] == 0 and out[2, 7] == 1
    assert np.all(out[2, 3:7] == 0) and not s["passed"]
    assert s["phases"][2, 2] == {1} and s["phases"][2, 7] == {2}
    # The delayed left descendant meets another A on row 5 and transmits.
    # Ignoring sideways distance makes that same shutter opaque instead.
    arena[5, 1:4] = 0
    out, s = _trace(arena, -1, (0, 0, 1), (4,), -2)
    wrong, ws = _trace(arena, -1, (0, 0, 1), (4,), -2, "no_lateral_delay")
    assert s["valid"] and (5, 1, 3) in s["passed"]
    assert out[6, 2] == 1 and out[6, 0] == -1
    assert wrong[6, 2] == -1 and wrong[6, 0] == 0
    assert not np.array_equal(s["path"], ws["path"])
    # Make this a matching B shutter: it transmits at col 4, with no detour.
    arena[5, 1:4] = -1
    arena[2, 3:7] = 1
    out, s = _trace(arena, -1, (0, 0, 1), (4,), -2)
    assert s["passed"] and not s["split"] and out[3, 4] == 0 and out[3, 2] == -1
    # Two descendants strike distinct singleton shutters together. Their
    # common inner flank has the same phase, so it merges without priority.
    arena = np.full((8, 11), -1, dtype=np.int8)
    arena[0, 3] = 0
    arena[2, 3] = 1
    arena[4, [2, 4]] = 0
    out, s = _trace(arena, -1, (0, 1), (3,), -2)
    assert s["valid"] and s["merges"] == 1
    assert s["phases"][4, 3] == {0} and out[5, 3] == 1
    # An edge shutter clips only its out-of-bounds descendant.
    arena = np.full((7, 9), -1, dtype=np.int8)
    arena[0, 2] = 0
    arena[2, :5] = 1
    out, s = _trace(arena, -1, (0, 1), (2,), -2)
    assert s["valid"] and s["clipped"] and out[1, 0] == 1
    assert s["incoming"][3] == ((5, 0),) and out[3, 5] == 0

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, orientations, patterns, lengths, source_counts, normalized = set(), set(), set(), set(), set(), set()
    role_colors = [set(), set(), set()]
    flags = {"clipped": 0, "merges": 0, "simultaneous": 0, "singleton": 0, "long": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                expected = solve(pair["input"])
                assert np.array_equal(expected, pair["output"])
                _, stats = _trace(info["arena"], info["background"], info["word"], info["sources"], info["frame"])
                shapes.add(pair["input"].shape)
                orientations.add(info["orientation"])
                labels = {}
                pattern = tuple(labels.setdefault(x, len(labels)) for x in info["word"])
                patterns.add(pattern)
                lengths.add(len(info["word"]))
                source_counts.add(len(info["sources"]))
                role_colors[0].add(info["background"])
                role_colors[1].add(info["frame"])
                role_colors[2].update(info["word"])
                normalized.add((info["arena"].shape, (info["arena"] != info["background"]).tobytes()))
                flags["clipped"] += bool(stats["clipped"])
                flags["merges"] += bool(stats["merges"])
                flags["simultaneous"] += any(sum(e[0] == r for e in stats["events"]) > 1
                                              for r, _, _ in stats["reached"])
                flags["singleton"] += any(b == a for _, a, b, _ in info["runs"])
                flags["long"] += any(b - a + 1 >= info["arena"].shape[1] * .7 for _, a, b, _ in info["runs"])
    assert len(shapes) >= 35 and len(normalized) >= 180
    assert orientations == set(range(8)) and lengths == set(range(2, 7))
    assert len(patterns) >= 20 and source_counts >= {1, 2, 3}
    assert all(colors == set(range(10)) for colors in role_colors)
    assert all(flags.values()), flags
    # All legal explicit combinations are exercised by verify.py; pin extremes
    # here without multiplying the bounded 42-seed semantic exercise.
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        a = generate_puzzle(3107, nt, nq)
        b = generate_puzzle(3107, nt, nq)
        validate_generated_puzzle(a)
        assert all(_key(x[k]) == _key(y[k]) for split in a
                   for x, y in zip(a[split], b[split]) for k in ("input", "output"))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "arena_structures": len(normalized),
            "word_patterns": len(patterns), "orientations": len(orientations), "features": flags}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
