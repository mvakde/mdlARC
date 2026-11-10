"""Radial echoes: periodic corner signals compete for concentric frames.

No files, parent modules, RNG state, or training examples enter solve().
Generation paints borders from sampled plans; solve uses input-derived depth
indexing. All diagnostics are bounded and self_test streams its 42 seeds.
"""

import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _corners(h, w, depth):
    d = np.arange(depth)
    return ((d, d), (d, w - 1 - d),
            (h - 1 - d, w - 1 - d), (h - 1 - d, d))


def _parse(value):
    g = _grid(value)
    h, w = g.shape
    if min(h, w) < 6 or min(h, w) % 2:
        raise ValueError("the smaller canvas dimension must be even and at least six")
    depth = min(h, w) // 2
    background = int(g[0, 1])  # This cell belongs to no corner track.
    track_mask = np.zeros(g.shape, dtype=bool)
    signals = []
    for corner, coordinates in enumerate(_corners(h, w, depth)):
        track_mask[coordinates] = True
        code = g[coordinates]
        occupied = np.flatnonzero(code != background)
        if not len(occupied):
            continue
        if len(occupied) not in (2, 3):
            raise ValueError("an active corner needs two references and at most one terminal")
        colors, counts = np.unique(code[occupied], return_counts=True)
        repeated = colors[counts == 2]
        if len(repeated) != 1:
            raise ValueError("each track must have exactly one equal-colored reference pair")
        reference = int(repeated[0])
        a, b = map(int, np.flatnonzero(code == reference))
        p = b - a
        others = [int(d) for d in occupied if int(d) not in (a, b)]
        terminal = others[0] if others else None
        if terminal is not None and ((terminal - a) % p or a < terminal < b):
            raise ValueError("a terminal must lie beyond the references on their phase")
        signals.append({"corner": corner, "a": a, "b": b, "t": terminal,
                        "ref": reference,
                        "paint": reference if terminal is None else int(code[terminal])})
    if np.any(g[~track_mask] != background) or not signals:
        raise ValueError("only corner tracks may contain non-background beads")
    return g, background, depth, signals


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all four radial signals and return the uniquely supported shells."""
    g, background, depth, signals = _parse(input_grid)
    counts = np.zeros(depth, dtype=int)
    colors = np.full(depth, background, dtype=np.int8)
    for s in signals:
        p = s["b"] - s["a"]
        positions = np.arange(s["a"] % p, depth, p)
        if s["t"] is not None:
            positions = positions[(positions >= min(s["a"], s["t"]))
                                  & (positions <= max(s["b"], s["t"]))]
        counts[positions] += 1
        colors[positions] = s["paint"]
    colors[counts != 1] = background
    h, w = g.shape
    r = np.minimum(np.arange(h), np.arange(h)[::-1])
    c = np.minimum(np.arange(w), np.arange(w)[::-1])
    return colors[np.minimum(r[:, None], c[None, :])].copy()


def _support(s, depth):
    """Construction-side set of depths, separate from solve's vector indexing."""
    p = s["b"] - s["a"]
    lo, hi = (0, depth - 1) if s["t"] is None else (
        min(s["a"], s["t"]), max(s["b"], s["t"]))
    return {d for d in range(lo, hi + 1) if (d - s["a"]) % p == 0}


def _ownership(signals, depth):
    owners = [[] for _ in range(depth)]
    for i, s in enumerate(signals):
        for d in _support(s, depth):
            owners[d].append(i)
    return owners


def _border_paint(shape, layers):
    out = np.empty(shape, dtype=np.int8)
    for d, color in enumerate(layers):
        # The last rectangle has thickness two along its shorter dimension.
        out[d, d:shape[1] - d] = color
        out[shape[0] - 1 - d, d:shape[1] - d] = color
        out[d:shape[0] - d, d] = color
        out[d:shape[0] - d, shape[1] - 1 - d] = color
    return out


def _render(shape, background, signals):
    depth = min(shape) // 2
    inp = np.full(shape, background, dtype=np.int8)
    tracks = _corners(*shape, depth)
    for s in signals:
        rr, cc = tracks[s["corner"]]
        for d in (s["a"], s["b"]):
            inp[rr[d], cc[d]] = s["ref"]
        if s["t"] is not None:
            d = s["t"]
            inp[rr[d], cc[d]] = s["paint"]
    layers = [signals[o[0]]["paint"] if len(o) == 1 else background
              for o in _ownership(signals, depth)]
    return {"input": inp, "output": _border_paint(shape, layers)}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _geometry(rng, depth, terminal=None, direction=None, period=None, both_sides=False):
    if terminal is None:
        terminal = bool(rng.integers(2))
    if period is None:
        period = int(rng.integers(1, (depth - 1) // 2 + 1))
    options = []
    for a in range(depth - period):
        b = a + period
        phase = list(range(a % period, depth, period))
        if len(phase) < 3:
            continue
        if not terminal:
            if not both_sides or (phase[0] < a and phase[-1] > b):
                options.append((a, b, None))
        else:
            for t in phase:
                if ((t < a and direction in (None, -1))
                        or (t > b and direction in (None, 1))):
                    options.append((a, b, t))
    if not options:
        return None
    a, b, t = _pick(rng, options)
    return {"a": a, "b": b, "t": t}


def _plan_ok(plans, depth, witness):
    if any(s is None for s in plans):
        return False
    owners = _ownership(plans, depth)
    unique = [d for d, o in enumerate(owners) if len(o) == 1]
    if not unique:
        return False
    if witness is None:
        return True
    multiplicities = {len(o) for o in owners}
    unique_sources = {owners[d][0] for d in unique}
    if witness == "two":
        terminal_support = _support(plans[0], depth)
        seeds = {s[k] for s in plans for k in ("a", "b", "t") if s[k] is not None}
        # Same-color output must be asymmetric and visibly erase an original bead.
        mask = [len(o) == 1 for o in owners]
        return (2 in multiplicities and unique_sources == {0, 1}
                and len(terminal_support) < depth
                and any(len(owners[d]) == 2 for d in seeds)
                and mask != mask[::-1])
    return ({2, 3} <= multiplicities and len(unique_sources) >= 2 and 0 in unique_sources
            and len(owners[-1]) == 1 and len(_support(plans[0], depth)) < depth)


def _plans(rng, depth, count, witness=None):
    # Random proposals use complete feasible coordinate ranges. A finite
    # constructive fallback makes success independent of lucky retries.
    for _ in range(128):
        if witness == "two":
            p = int(rng.integers(2, (depth - 1) // 3 + 1))
            plans = [_geometry(rng, depth, True, 1, 1),
                     _geometry(rng, depth, False, period=p, both_sides=True)]
        elif witness == "three":
            plans = [_geometry(rng, depth, True, -1, int(rng.integers(2, (depth - 1) // 2 + 1))),
                     _geometry(rng, depth, False, period=int(rng.integers(2, 4))),
                     _geometry(rng, depth, False, period=int(rng.integers(2, 5)))]
        else:
            plans = [_geometry(rng, depth) for _ in range(count)]
        if _plan_ok(plans, depth, witness):
            return plans
    return _fallback_plans(depth, count, witness)


def _fallback_plans(depth, count, witness):
    """Finite valid constructions; also exercised directly by self_test."""
    if witness == "two":
        plans = [{"a": 2, "b": 3, "t": depth - 2},
                 {"a": 2, "b": 4, "t": None}]
        if _plan_ok(plans, depth, witness):
            return plans
        # The symmetric special case is avoided by moving the terminal inward.
        plans[0]["t"] = depth - 3
        if _plan_ok(plans, depth, witness):
            return plans
    elif witness == "three":
        for terminal in range(depth - 4):
            for p in (2, 3, 4):
                for q in (2, 3, 4):
                    for phase in range(min(p, depth - 2 * p)):
                        for other in range(min(q, depth - 2 * q)):
                            plans = [{"a": terminal + 2, "b": terminal + 4, "t": terminal},
                                     {"a": phase, "b": phase + p, "t": None},
                                     {"a": other, "b": other + q, "t": None}]
                            if _plan_ok(plans, depth, witness):
                                return plans
    else:
        # One full-depth signal and 0..3 bounded signals leave the center unique.
        return ([{"a": 0, "b": 1, "t": None}]
                + [{"a": 0, "b": 1, "t": 2} for _ in range(count - 1)])
    raise AssertionError("constructive witness fallback failed")


def _make_pair(rng, depth, count, witness=None):
    plans = _plans(rng, depth, count, witness)
    background = int(rng.integers(10))
    foreground = [c for c in range(10) if c != background]
    minimum = 3 if witness == "three" else 2
    k = int(rng.integers(minimum, 10))
    palette = [int(c) for c in rng.choice(foreground, k, replace=False)]
    corners = [int(c) for c in rng.choice(4, count, replace=False)]
    paints = ([_pick(rng, palette)] * count if witness == "two" else
              [int(c) for c in rng.choice(palette, count, replace=False)]
              if witness == "three" else [_pick(rng, palette) for _ in plans])
    signals = []
    for plan, corner, paint in zip(plans, corners, paints):
        reference = paint if plan["t"] is None else _pick(rng, [c for c in palette if c != paint])
        signals.append(dict(plan, corner=corner, ref=reference, paint=paint))
    shorter = 2 * depth
    longer = int(rng.integers(shorter, 31))
    shape = (shorter, longer) if rng.integers(2) else (longer, shorter)
    return _render(shape, background, signals)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError("num_train must be 2..10 and num_test 1..4 integers")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    first = int(rng.integers(8, 12))
    depths = [first, _pick(rng, [d for d in range(9, 12) if d != first])]
    depths += [int(rng.integers(6, 12)) for _ in range(num_train - 2)]
    shift = int(rng.integers(2))
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = ("two", "three")[i] if split == "train" and i < 2 else None
            depth = depths[i] if split == "train" else int(
                rng.integers(12, 16) if shift else rng.integers(4, 6))
            tracks = 2 if witness == "two" else 3 if witness == "three" else int(rng.integers(1, 5))
            pair = _make_pair(rng, depth, tracks, witness)
            # There are ten different backgrounds under cyclic relabeling.
            # At most nine previous training inputs have this depth; test
            # depths are disjoint and have at most three previous inputs.
            for delta in range(10):
                candidate = {role: ((g.astype(int) + delta) % 10).astype(np.int8)
                             for role, g in pair.items()}
                key = _key(candidate["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(candidate)
                    break
            else:
                raise AssertionError("ten-background uniqueness bound violated")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(g, mode):
    """Explicit simpler hypotheses, used only to audit demonstration evidence."""
    g, bg, depth, signals = _parse(g)
    supports = []
    paints = []
    for s in signals:
        p = s["b"] - s["a"]
        positions = _support(s, depth)
        if mode == "raw_beads":
            positions = {s[k] for k in ("a", "b", "t") if s[k] is not None}
        elif mode == "unbounded_terminal":
            positions = set(range(s["a"] % p, depth, p))
        elif mode == "solid_intervals":
            positions = set(range(depth)) if s["t"] is None else set(
                range(min(s["a"], s["t"]), max(s["b"], s["t"]) + 1))
        elif mode == "solid_terminal" and s["t"] is not None:
            positions = set(range(min(s["a"], s["t"]), max(s["b"], s["t"]) + 1))
        supports.append(positions)
        paints.append(s["ref"] if mode == "no_recolor" else s["paint"])
    layers = []
    for d in range(depth):
        ids = [i for i, support in enumerate(supports) if d in support]
        keep = len(ids) == 1
        if mode == "union":
            keep = bool(ids)
        elif mode == "parity":
            keep = len(ids) % 2 == 1
        elif mode == "same_color_union":
            keep = bool(ids) and len({paints[i] for i in ids}) == 1
        layers.append(paints[ids[0]] if keep else bg)
    if mode == "reverse_depth":
        layers.reverse()
    out = _border_paint(g.shape, layers)
    if mode == "keep_beads":
        out[g != bg] = g[g != bg]
    elif mode == "no_frames":
        out[:] = bg
        for rr, cc in _corners(*g.shape, depth):
            out[rr, cc] = layers
    return out


_ALTERNATIVES = ("raw_beads", "unbounded_terminal", "solid_intervals", "solid_terminal", "no_recolor",
                 "union", "parity", "same_color_union", "reverse_depth",
                 "keep_beads", "no_frames")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_depths, test_depths = set(), set(), set()
    multiplicities, modes, directions, periods = set(), set(), set(), set()
    dotted_terminal = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, bg, depth, signals = _parse(pair["input"])
            out = _grid(pair["output"])
            assert out.shape == g.shape and _key(g) not in seen
            seen.add(_key(g))
            (train_depths if split == "train" else test_depths).add(depth)
            owners = _ownership(signals, depth)
            assert any(len(o) == 1 for o in owners), "all echoes canceled"
            # Check every complete border, including erased source beads and
            # the rectangular center. This is not just a call to solve().
            for d, ids in enumerate(owners):
                color = signals[ids[0]]["paint"] if len(ids) == 1 else bg
                assert np.all(out[d, d:out.shape[1] - d] == color)
                assert np.all(out[-1 - d, d:out.shape[1] - d] == color)
                assert np.all(out[d:out.shape[0] - d, d] == color)
                assert np.all(out[d:out.shape[0] - d, -1 - d] == color)
            assert np.array_equal(solve(g), out)
            assert not np.array_equal(g, out)
            if split == "train":
                multiplicities.update(len(o) for o in owners)
                for s in signals:
                    periods.add(s["b"] - s["a"])
                    modes.add(s["t"] is None)
                    if s["t"] is not None:
                        directions.add(-1 if s["t"] < s["a"] else 1)
                        dotted_terminal |= s["b"] - s["a"] > 1
    assert {1, 2, 3} <= multiplicities
    assert modes == {False, True} and directions == {-1, 1}
    assert 1 in periods and max(periods) >= 2
    assert dotted_terminal
    assert len(train_depths) >= 2 and train_depths.isdisjoint(test_depths)
    for mode in _ALTERNATIVES:
        assert any(not np.array_equal(_alternative(p["input"], mode), p["output"])
                   for p in puzzle["train"]), "training fails to reject " + mode


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, count boundaries, counterfactuals and symmetries."""
    signatures, shapes, structures, colors, backgrounds = set(), set(), set(), set(), set()
    track_counts, periods, terminal_distances, shifts, overlaps = set(), set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, bg, depth, signals = _parse(pair["input"])
                pairs += 1
                shapes.add(g.shape)
                backgrounds.add(bg)
                colors.update(map(int, np.unique(g)))
                track_counts.add(len(signals))
                overlaps.update(map(len, _ownership(signals, depth)))
                if split == "test":
                    shifts.add("larger" if depth >= 12 else "smaller")
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in g.flat)
                structures.add((g.shape, hashlib.sha256(normalized).hexdigest()))
                for s in signals:
                    p = s["b"] - s["a"]
                    periods.add(p)
                    if s["t"] is not None:
                        terminal_distances.add(min(abs(s["t"] - s["a"]), abs(s["t"] - s["b"])) // p)
                # Geometric and color covariance directly test absence of
                # hidden corner priority and fixed color commands.
                if split == "test":
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(g, k)), np.rot90(pair["output"], k))
                    assert np.array_equal(solve(g.T), pair["output"].T)
                    assert np.array_equal(solve(9 - g), 9 - pair["output"])
                    before = g.copy()
                    solve(g)
                    assert np.array_equal(g, before)
    assert len(signatures) == 42 and len(shapes) >= 45 and len(structures) >= 180
    assert backgrounds == colors == set(range(10))
    assert track_counts == {1, 2, 3, 4} and {0, 1, 2, 3, 4} <= overlaps
    assert {1, 2, 3, 4, 5} <= periods and max(terminal_distances) >= 5
    assert shifts == {"smaller", "larger"}
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exercise the constructive witness fallbacks' entire intended depth domain.
    for depth in range(8, 12):
        for witness, count in (("two", 2), ("three", 3)):
            if witness == "two" or depth >= 9:
                assert _plan_ok(_fallback_plans(depth, count, witness), depth, witness)
    # Exact layer codes for the two worked examples in task_spec.md.
    for depth, signals, layers in (
        (10, [{"a": 2, "b": 3, "t": 7, "ref": 2, "paint": 6, "corner": 0},
              {"a": 4, "b": 6, "t": None, "ref": 6, "paint": 6, "corner": 2}],
         [6, 0, 0, 6, 0, 6, 0, 6, 6, 0]),
        (9, [{"a": 6, "b": 7, "t": 1, "ref": 2, "paint": 4, "corner": 0},
             {"a": 0, "b": 2, "t": None, "ref": 6, "paint": 6, "corner": 1},
             {"a": 1, "b": 4, "t": None, "ref": 8, "paint": 8, "corner": 3}],
         [6, 0, 0, 4, 0, 4, 0, 0, 6])):
        pair = _render((2 * depth, 2 * depth + 1), 0, signals)
        assert np.array_equal(solve(pair["input"]), _border_paint(pair["input"].shape, layers))
    # Hand-computed boundary fixture: N=3, two identical echoes cancel
    # even though they have the same paint color. Then add a third echo;
    # triple occupancy stays erased rather than reappearing by parity.
    signals = [{"a": 0, "b": 1, "t": None, "ref": 2, "paint": 2, "corner": i}
               for i in range(3)]
    for count in (1, 2, 3):
        pair = _render((6, 9), 7, signals[:count])
        expected = np.full((6, 9), 2 if count == 1 else 7, dtype=np.int8)
        assert np.array_equal(solve(pair["input"]), expected)
    # This checks that the validator notices a broken shell, rather than only
    # accepting its own generation metadata.
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 1] = (int(bad["test"][0]["output"][0, 1]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted shell accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_inputs": len(structures),
            "track_counts": sorted(track_counts), "periods": sorted(periods),
            "overlap_counts": sorted(overlaps), "shifts": sorted(shifts),
            "backgrounds": sorted(backgrounds), "max_terminal_steps": max(terminal_distances)}
