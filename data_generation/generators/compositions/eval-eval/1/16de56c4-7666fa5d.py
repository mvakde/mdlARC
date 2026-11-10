"""Synchronized diagonal loom: input-only modular bracketing.

Only NumPy and the standard library are needed. No parent code, file reads,
training labels, seed, or generation state is used by solve.
"""

import numpy as np


ALTERNATIVES = (
    "no_rhythm", "no_bounds", "no_recolor", "no_phase", "no_extrapolate",
    "forward", "backward", "beads_only", "nearest_terminal", "same_period",
    "consecutive", "any_pair", "all_agree", "outer_physical",
    "majority_reference", "no_geometry",
)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=False)


def _key(g):
    return (g.shape, g.tobytes())


def _inspect(value):
    """Infer geometry, roles, and clocks entirely from foreground pixels."""
    g = _grid(value)
    hues, counts = np.unique(g, return_counts=True)
    bg = int(hues[np.argmax(counts)])
    if counts.max() <= g.size // 2:
        raise ValueError("background must occupy a strict majority")
    fg = g[g != bg]
    hues, counts = np.unique(fg, return_counts=True)
    if not len(hues) or counts.max() <= len(fg) // 2:
        raise ValueError("foreground needs a majority scaffold color")
    scaffold = int(hues[np.argmax(counts)])
    candidates = []
    for s in (-1, 1):
        lines = {}
        for r, c in np.argwhere(g != bg):
            r, c = int(r), int(c)
            lines.setdefault(c - s*r, []).append((r, c))
        qs = sorted(lines)
        if (len(qs) < 2 or any(b-a < 4 or (b-a) % 2
                               for a, b in zip(qs, qs[1:]))):
            continue
        rails = []
        valid = True
        for q in qs:
            cells = sorted(lines[q])
            rows = [r for r, _ in cells]
            vals = [int(g[r, c]) for r, c in cells]
            if (len(rows) < 5 or rows != list(range(rows[0], rows[-1]+1))
                    or vals.count(scaffold) <= len(vals)//2):
                valid = False
                break
            beads = [(r+s*c, int(g[r, c])) for r, c in cells
                     if g[r, c] != scaffold]
            bead_counts = {}
            for _, color in beads:
                bead_counts[color] = bead_counts.get(color, 0) + 1
            repeated = [color for color, n in bead_counts.items() if n >= 2]
            if len(repeated) != 1 or bead_counts[repeated[0]] != 2:
                valid = False
                break
            ref = repeated[0]
            a, b = sorted(t for t, color in beads if color == ref)
            step = b-a
            if step <= 0 or step % 2:
                valid = False
                break
            extras = [(t, color) for t, color in beads if color != ref]
            aligned = [(t, color) for t, color in extras if (t-a) % step == 0]
            if len(aligned) > 1 or (aligned and a < aligned[0][0] < b):
                valid = False
                break
            terminal = aligned[0] if aligned else None
            ts = [r+s*c for r, c in cells]
            rails.append(dict(q=q, lo=min(ts), hi=max(ts), a=a, b=b,
                              p=step//2, ref=ref, terminal=terminal,
                              extras=extras, length=len(rows)))
        if valid:
            candidates.append(dict(background=bg, scaffold=scaffold,
                                   slope=s, rails=rails))
    if len(candidates) != 1:
        raise ValueError("input must have one recognizable parallel rail family")
    return g, candidates[0]


def _clock(rail, alternative=""):
    lo, hi, ink = rail["lo"], rail["hi"], rail["ref"]
    term = rail["terminal"]
    if alternative == "nearest_terminal" and rail["extras"]:
        term = min(rail["extras"], key=lambda x: min(abs(x[0]-rail["a"]),
                                                    abs(x[0]-rail["b"])))
    if term is not None:
        ink = term[1]
        if alternative != "no_bounds":
            lo, hi = min(rail["a"], term[0]), max(rail["b"], term[0])
        if alternative == "no_recolor":
            ink = rail["ref"]
    elif alternative == "no_extrapolate":
        lo, hi = rail["a"], rail["b"]
    if alternative == "forward" and term is None:
        lo = max(lo, rail["a"])
    if alternative == "backward" and term is None:
        hi = min(hi, rail["b"])
    phase = rail["lo"] if alternative == "no_phase" else rail["a"]
    return lo, hi, ink, phase


def _fires(rail, level, alternative=""):
    lo, hi, _, phase = _clock(rail, alternative)
    if alternative == "no_rhythm":
        return rail["lo"] <= level <= rail["hi"]
    if not lo <= level <= hi:
        return False
    if alternative == "beads_only":
        return level in (rail["a"], rail["b"],
                         None if rail["terminal"] is None else rail["terminal"][0])
    return (level-phase) % (2*rail["p"]) == 0


def _prediction(value, alternative=""):
    g, info = _inspect(value)
    out = g.copy()
    rr, cc = np.indices(g.shape)
    q = cc-info["slope"]*rr
    t = rr+info["slope"]*cc
    rails = info["rails"]
    ref_counts = {}
    for rail in rails:
        ref_counts[rail["ref"]] = ref_counts.get(rail["ref"], 0)+1
    majority_reference = min(ref_counts, key=lambda color: (-ref_counts[color], color))
    field = g == info["background"]
    for level in range(int(t.min()), int(t.max())+1):
        active = [i for i, rail in enumerate(rails) if _fires(rail, level, alternative)]
        if len(active) < 2:
            continue
        if alternative == "all_agree" and len({_clock(rails[i])[2] for i in active}) > 1:
            continue
        if alternative == "outer_physical" and (0 not in active or len(rails)-1 not in active):
            continue
        if alternative == "consecutive":
            pairs = [(i, i+1) for i in range(len(rails)-1)
                     if i in active and i+1 in active]
        elif alternative == "any_pair":
            pairs = [(i, j) for n, i in enumerate(active) for j in active[n+1:]]
        else:
            pairs = [(active[0], active[-1])]
        for i, j in pairs:
            a, b = rails[i], rails[j]
            ink_a, ink_b = _clock(a, alternative)[2], _clock(b, alternative)[2]
            if ink_a != ink_b or (alternative == "same_period" and a["p"] != b["p"]):
                continue
            mask = field & (t == level)
            if alternative != "no_geometry":
                mask &= (q > a["q"]) & (q < b["q"])
            out[mask] = majority_reference if alternative == "majority_reference" else ink_a
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover visible clocks and draw synchronized, agreeing crossbars."""
    return _prediction(input_grid)


def _label(g, specs, slope, background):
    """Construction labels: a direct cellwise calculation on sampled specs.

    Unlike solve, this neither segments input nor enumerates scanline masks.
    It remains a self-authored consistency check, not an independent solver.
    """
    out = g.copy()
    for r, c in np.argwhere(g == background):
        r, c = int(r), int(c)
        t, q = r+slope*c, c-slope*r
        active = []
        for rail in specs:
            term = rail["terminal"]
            low = rail["lo"] if term is None else min(rail["a"], term[0])
            high = rail["hi"] if term is None else max(rail["b"], term[0])
            if low <= t <= high and (t-rail["a"]) % (2*rail["p"]) == 0:
                active.append((rail["q"], rail["ref"] if term is None else term[1]))
        if len(active) < 2:
            continue
        left, right = min(active), max(active)
        if left[1] == right[1] and left[0] < q < right[0]:
            out[r, c] = left[1]
    return out


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _spec(q, lo, hi, a, p, ref, terminal=None, extras=()):
    return dict(q=q, lo=lo, hi=hi, a=a, b=a+2*p, p=p,
                ref=ref, terminal=terminal, extras=list(extras))


def _teaching_specs(rng, kind, colors):
    """Variable constructive witnesses, never a fixed pixel template."""
    ink, other, ornament = colors[:3]
    m = int(rng.integers(7, 10))
    count = 3 if kind == 0 else 4
    qs = [0]
    for _ in range(count-1):
        qs.append(qs[-1] + int(_pick(rng, (4, 6, 8))))
    # Independently offset endpoints; clocks are not anchored at these ends.
    intervals = [(-2*m+2*int(rng.integers(0, 2)),
                   2*m-2*int(rng.integers(0, 2))) for _ in qs]
    direction = int(_pick(rng, (-1, 1)))
    if kind == 0:
        pa, pb = _pick(rng, ((2, 3), (3, 2), (2, 4), (4, 2)))
        a = 2*pa*int(_pick(rng, (-1, 0)))
        shared_step = int(np.lcm(pa, pb))
        right_a, right_terminal = -2*shared_step, 2*shared_step
        middle_a, terminal_t = (-4, 4) if direction > 0 else (2, -4)
        specs = [
            _spec(qs[0], *intervals[0], a, pa, ink),
            _spec(qs[1], *intervals[1], middle_a, 1, other, (terminal_t, ink)),
            _spec(qs[2], *intervals[2], right_a, pb, other, (right_terminal, ink)),
        ]
    else:
        p = int(_pick(rng, (2, 4)))
        terminal_a, terminal_t = (-6, 6) if direction > 0 else (4, -6)
        specs = [
            _spec(qs[0], *intervals[0], -2*p, p, ink),
            _spec(qs[1], *intervals[1], terminal_a, 1, ink, (terminal_t, other)),
            _spec(qs[2], *intervals[2], -6, 3, ink),
            _spec(qs[3], *intervals[3], -2, 1, ink),
        ]
    # Off-phase ornament deliberately appears on an otherwise unbounded rail.
    rail = specs[0]
    choices = [t for t in range(rail["lo"], rail["hi"]+1, 2)
               if (t-rail["a"]) % (2*rail["p"])]
    rail["extras"] = [(int(_pick(rng, choices)), ornament)]
    # A reflection of longitudinal geometry changes terminal direction and
    # reference placement without changing the concept or the carrier slope.
    if rng.random() < .5:
        for rail in specs:
            rail["lo"], rail["hi"] = -rail["hi"], -rail["lo"]
            rail["a"], rail["b"] = -rail["b"], -rail["a"]
            if rail["terminal"] is not None:
                t, ink_value = rail["terminal"]
                rail["terminal"] = (-t, ink_value)
            rail["extras"] = [(-t, color) for t, color in rail["extras"]]
    return specs


def _random_specs(rng, count, colors):
    qs = [0]
    for _ in range(count-1):
        qs.append(qs[-1] + int(_pick(rng, (4, 4, 6, 8, 10))))
    if qs[-1] > 36:
        raise ValueError("transverse span too large")
    mode = int(rng.integers(4))
    all_terminal = bool(rng.random() < .18)
    no_terminal = bool(rng.random() < .2)
    specs = []
    for i, q in enumerate(qs):
        length = int(rng.integers(7, 19))
        if mode == 0:
            before = int(rng.integers(2, length-2))
            lo = -2*before
        elif mode == 1:
            lo = -12
        elif mode == 2:
            lo = -2*(length-1)+10
        else:
            lo = -16+4*i+2*int(rng.integers(-2, 3))
        hi = lo+2*(length-1)
        period = int(rng.integers(1, min(6, (length-1)//2)+1))
        phases = [offset for offset in range(period)
                  if len(range(lo+2*offset, hi+1, 2*period)) >= 3]
        positions = list(range(lo+2*int(_pick(rng, phases)), hi+1, 2*period))
        ref_index = int(rng.integers(len(positions)-1))
        a = positions[ref_index]
        ref = int(_pick(rng, colors[:3] if rng.random() < .35 else colors[:1]))
        candidates = [t for t in positions if t < a or t > a+2*period]
        terminal = None
        if candidates and not no_terminal and (all_terminal or rng.random() < .48):
            color = int(_pick(rng, [c for c in colors[:3] if c != ref]))
            terminal = (int(_pick(rng, candidates)), color)
        extras = []
        off_phase = [t for t in range(lo, hi+1, 2) if (t-a) % (2*period)]
        available = [c for c in colors if c != ref and (terminal is None or c != terminal[1])]
        maximum_noise = min(2, len(off_phase), len(available),
                            (length-1)//2-2-int(terminal is not None))
        if maximum_noise > 0 and rng.random() < .45:
            n = int(rng.integers(1, maximum_noise+1))
            points = rng.choice(off_phase, n, replace=False)
            inks = rng.choice(available, n, replace=False)
            extras = [(int(t), int(c)) for t, c in zip(points, inks)]
        specs.append(_spec(q, lo, hi, a, period, ref, terminal, extras))
    return specs


def _render(rng, specs, slope, palette):
    background, scaffold = palette[:2]
    cells = []
    for rail in specs:
        for t in range(rail["lo"], rail["hi"]+1, 2):
            q = rail["q"]
            cells.append(((t-slope*q)//2, (q+slope*t)//2))
    rmin, cmin = min(r for r, _ in cells), min(c for _, c in cells)
    height = max(r for r, _ in cells)-rmin+1
    width = max(c for _, c in cells)-cmin+1
    if max(height, width) > 30:
        raise ValueError("rails do not fit ARC dimensions")
    dimensions = []
    offsets = []
    for span in (height, width):
        extra = int(rng.integers(0, min(6, 30-span)+1))
        if rng.random() < .12:
            extra = int(rng.integers(0, 31-span))
        dimensions.append(span+extra)
        offsets.append(int(rng.integers(extra+1)))
    dr, dc = offsets[0]-rmin, offsets[1]-cmin
    g = np.full(tuple(dimensions), background, dtype=np.int8)
    translated = []
    for original in specs:
        rail = dict(original)
        q_shift, t_shift = dc-slope*dr, dr+slope*dc
        rail["q"] += q_shift
        for key in ("lo", "hi", "a", "b"):
            rail[key] += t_shift
        if rail["terminal"] is not None:
            t, color = rail["terminal"]
            rail["terminal"] = (t+t_shift, color)
        rail["extras"] = [(t+t_shift, color) for t, color in rail["extras"]]
        translated.append(rail)
        q = rail["q"]
        for t in range(rail["lo"], rail["hi"]+1, 2):
            r, c = (t-slope*q)//2, (q+slope*t)//2
            g[r, c] = scaffold
        beads = [(rail["a"], rail["ref"]), (rail["b"], rail["ref"])]
        if rail["terminal"] is not None:
            beads.append(rail["terminal"])
        beads += rail["extras"]
        for t, color in beads:
            r, c = (t-slope*q)//2, (q+slope*t)//2
            g[r, c] = color
    _, info = _inspect(g)
    if info["slope"] != slope or len(info["rails"]) != len(specs):
        raise ValueError("wrong visible rail interpretation")
    out = _label(g, translated, slope, background)
    if not np.array_equal(out, solve(g)):
        raise AssertionError("sampled-geometry label and input reconstruction disagree")
    return dict(input=g, output=out)


def _differences(pair):
    return {alt for alt in ALTERNATIVES
            if not np.array_equal(pair["output"], _prediction(pair["input"], alt))}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    stable = bool(rng.random() < .3)
    first_slope = int(_pick(rng, (-1, 1)))
    puzzle = {"train": [], "test": []}
    seen = set()

    def make(kind, slope, count=None):
        for _ in range(192):
            colors = palette if stable else [int(c) for c in rng.permutation(10)]
            try:
                bead_colors = colors[2:]
                if kind == 1:
                    # Palette stability must not make the output a fixed hue:
                    # swap local bead roles in the second demonstration.
                    bead_colors = [bead_colors[1], bead_colors[0]]+bead_colors[2:]
                specs = (_teaching_specs(rng, kind, bead_colors) if kind is not None
                         else _random_specs(rng, count, colors[2:]))
                pair = _render(rng, specs, slope, colors)
            except ValueError:
                continue
            if _key(pair["input"]) not in seen:
                return pair
        raise RuntimeError("bounded rail construction exhausted")

    required = set(ALTERNATIVES)
    # Tests first: determine what their actual returned geometry requires.
    for _ in range(num_test):
        for _attempt in range(192):
            pair = make(None, int(_pick(rng, (-1, 1))), int(rng.integers(5, 8)))
            differences = _differences(pair)
            if {"no_rhythm", "no_geometry"} <= differences:
                required |= differences
                puzzle["test"].append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            raise RuntimeError("bounded test witness construction exhausted")

    # Even the legal minimum of two demonstrations teaches every named rule
    # distinction. Their geometry/palettes/placements are freely varied.
    for _attempt in range(128):
        a = make(0, first_slope)
        b = make(1, -first_slope)
        added_inks = {int(c) for pair in (a, b)
                      for c in np.unique(pair["output"][pair["input"] != pair["output"]])}
        if (_key(a["input"]) != _key(b["input"])
                and len(added_inks) >= 2
                and required <= (_differences(a) | _differences(b))):
            puzzle["train"] += [a, b]
            seen.update((_key(a["input"]), _key(b["input"])))
            break
    else:
        raise RuntimeError("bounded demonstration coverage exhausted")
    for _ in range(num_train-2):
        pair = make(None, int(_pick(rng, (-1, 1))), int(rng.integers(2, 8)))
        puzzle["train"].append(pair)
        seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, coverage, test_dependencies, training_inks = set(), set(), set(), set()
    train_slopes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            _, info = _inspect(g)
            assert np.array_equal(solve(g), out), "incorrect modular bracket label"
            occupied = g != info["background"]
            assert np.array_equal(g[occupied], out[occupied]), "rails/beads must survive"
            changed = g != out
            assert np.all(g[changed] == info["background"])
            effective_inks = {_clock(s)[2] for s in info["rails"]}
            assert set(int(c) for c in np.unique(out[changed])) <= effective_inks
            # Verify every added pixel really has the two matching extremes.
            for r, c in np.argwhere(changed):
                t, q = int(r+info["slope"]*c), int(c-info["slope"]*r)
                active = [s for s in info["rails"] if _fires(s, t)]
                assert len(active) >= 2
                assert active[0]["q"] < q < active[-1]["q"]
                assert _clock(active[0])[2] == _clock(active[-1])[2] == int(out[r, c])
            differences = _differences(dict(input=g, output=out))
            if split == "train":
                coverage |= differences
                train_slopes.add(info["slope"])
                training_inks.update(int(c) for c in np.unique(out[changed]))
            else:
                assert {"no_rhythm", "no_geometry"} <= differences, "test must use both parents"
                test_dependencies |= differences
    assert train_slopes == {-1, 1}, "both diagonal presentations must be demonstrated"
    assert len(training_inks) >= 2, "training must show that crossbar ink is not fixed"
    assert set(ALTERNATIVES) <= coverage, "missing alternative-rule teaching witness"
    assert test_dependencies <= coverage, "test depends on an untaught distinction"


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _features(pair):
    g, info = _inspect(pair["input"])
    rails = info["rails"]
    q0, t0 = min(s["q"] for s in rails), min(s["lo"] for s in rails)
    layout = tuple((s["q"]-q0, s["lo"]-t0, s["hi"]-t0,
                    s["a"]-t0, s["p"], None if s["terminal"] is None
                    else s["terminal"][0]-t0) for s in rails)
    return dict(shape=g.shape, slope=info["slope"], background=info["background"],
                scaffold=info["scaffold"], count=len(rails), layout=layout,
                periods=tuple(s["p"] for s in rails),
                lengths=tuple(s["length"] for s in rails),
                gaps=tuple(b["q"]-a["q"] for a, b in zip(rails, rails[1:])),
                terminals=sum(s["terminal"] is not None for s in rails),
                noise=sum(len(s["extras"])-int(s["terminal"] is not None) for s in rails),
                directions=tuple(-1 if s["terminal"][0] < s["a"] else 1
                                 for s in rails if s["terminal"] is not None),
                inks=tuple(sorted({_clock(s)[2] for s in rails})),
                changed=int(np.sum(g != pair["output"])))


def self_test():
    """Bounded semantic fixtures, 42-seed variation, and all legal counts."""
    # Analytic fixture: outer q=0,8 rails synchronize at t=-12,0,12;
    # q=4 has an offset phase and shares t=-6,6 only with the right rail.
    # No solve call defines the expected cell set.
    specs = [_spec(0, -12, 12, -4, 2, 3),
             _spec(4, -12, 12, -2, 2, 3),
             _spec(8, -12, 12, -6, 3, 3)]
    fixture_rng = np.random.default_rng(77)
    pair = _render(fixture_rng, specs, 1, [0, 5, 3, 7, 9])
    inp, info = _inspect(pair["input"])
    shift = info["rails"][0]["a"]+4
    qshift = info["rails"][0]["q"]
    expected = inp.copy()
    for r, c in np.argwhere(inp == 0):
        t, q = int(r+c)-shift, int(c-r)-qshift
        # Left/right synchronize every 12 t units; middle never shares their
        # multiples of four, but coincides with right at t=-6 and t=6.
        if ((t in (-12, 0, 12) and 0 < q < 8)
                or (t in (-6, 6) and 4 < q < 8)):
            expected[r, c] = 3
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(pair["output"], expected)
    assert not np.array_equal(_prediction(inp, "consecutive"), expected)
    assert not np.array_equal(_prediction(inp, "outer_physical"), expected)

    # Terminal ink and inclusive stopping each matter: only -4,0,4 draw.
    specs = [_spec(0, -12, 12, -4, 2, 3),
             _spec(8, -12, 12, -4, 2, 7, (4, 3))]
    stopped = _render(fixture_rng, specs, 1, [0, 5, 3, 7, 9])
    inp, info = _inspect(stopped["input"])
    shift = info["rails"][0]["a"]+4
    qshift = info["rails"][0]["q"]
    expected = inp.copy()
    for r, c in np.argwhere(inp == 0):
        t, q = int(r+c)-shift, int(c-r)-qshift
        if t in (-4, 0, 4) and 0 < q < 8:
            expected[r, c] = 3
    assert np.array_equal(solve(inp), expected)
    assert not np.array_equal(_prediction(inp, "no_bounds"), expected)
    assert not np.array_equal(_prediction(inp, "no_recolor"), expected)
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(inp, turns)), np.rot90(expected, turns))
        assert np.array_equal(solve(9-np.rot90(inp, turns)), 9-np.rot90(expected, turns))

    # A conflicting firing outer rail suppresses all matching interior pairs.
    specs = [_spec(q, -12, 12, -4, 2, 3) for q in (0, 4, 8)]
    specs.append(_spec(12, -12, 12, -4, 2, 3, (4, 7)))
    conflict = _render(fixture_rng, specs, -1, [0, 5, 3, 7, 9])
    inp, info = _inspect(conflict["input"])
    shift = info["rails"][0]["a"]+4
    qshift = info["rails"][0]["q"]
    expected = inp.copy()
    for r, c in np.argwhere(inp == 0):
        t, q = int(r-c)-shift, int(c+r)-qshift
        if t in (-12, -8, 8, 12) and 0 < q < 8:
            expected[r, c] = 3
    assert np.array_equal(solve(inp), expected)
    assert not np.array_equal(_prediction(inp, "any_pair"), expected)

    # Different-ink interior pulses are ignored when the two firing extremes
    # agree. At +/-6 the left rail is silent, exposing a conflicting extreme.
    specs = [_spec(0, -12, 12, -4, 2, 3),
             _spec(4, -12, 12, -6, 1, 3, (6, 7)),
             _spec(8, -12, 12, -6, 3, 3),
             _spec(12, -12, 12, -2, 1, 3)]
    interior = _render(fixture_rng, specs, 1, [0, 5, 3, 7, 9])
    inp, info = _inspect(interior["input"])
    shift, qshift = info["rails"][0]["a"]+4, info["rails"][0]["q"]
    expected = inp.copy()
    for r, c in np.argwhere(inp == 0):
        t, q = int(r+c)-shift, int(c-r)-qshift
        if t in (-12, -8, -4, 0, 4, 8, 12) and 0 < q < 12:
            expected[r, c] = 3
    assert np.array_equal(solve(inp), expected)
    assert not np.array_equal(_prediction(inp, "all_agree"), expected)
    assert not np.array_equal(_prediction(inp, "any_pair"), expected)

    # Solve must exceed the sampler's period-six range. Here period eight and
    # period four synchronize at -16,0,16 on long visible rails.
    specs = [_spec(0, -24, 24, -16, 8, 3),
             _spec(8, -24, 24, -8, 4, 3)]
    wide = _render(fixture_rng, specs, 1, [0, 5, 3, 7, 9])
    inp, info = _inspect(wide["input"])
    shift, qshift = info["rails"][0]["a"]+16, info["rails"][0]["q"]
    expected = inp.copy()
    for r, c in np.argwhere(inp == 0):
        t, q = int(r+c)-shift, int(c-r)-qshift
        if t in (-16, 0, 16) and 0 < q < 8:
            expected[r, c] = 3
    assert np.array_equal(solve(inp), expected)

    # Correct labels can still be an insufficient teaching set. Two simple
    # brackets do not explain the conflict fixture or the full family.
    simple_specs = [_spec(0, -12, 12, -4, 2, 3),
                    _spec(8, -12, 12, -4, 2, 3)]
    insufficient = {"train": [_render(fixture_rng, simple_specs, s, [0, 5, 3, 7, 9])
                              for s in (-1, 1)], "test": [conflict]}
    try:
        validate_generated_puzzle(insufficient)
    except AssertionError as exc:
        assert "training must show" in str(exc) or "teaching witness" in str(exc)
    else:
        raise AssertionError("correctly labeled but inadequate training accepted")

    fingerprints, layouts, shapes, records = set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                records.append(f)
                shapes.add(f["shape"])
                layouts.add(f["layout"])
    assert len(fingerprints) == 42 and len(layouts) >= 150 and len(shapes) >= 60
    assert {r["slope"] for r in records} == {-1, 1}
    assert {2, 3, 4, 5, 6, 7} <= {r["count"] for r in records}
    assert {1, 2, 3, 4, 5, 6} <= {p for r in records for p in r["periods"]}
    assert len({r["background"] for r in records}) == 10
    assert len({r["scaffold"] for r in records}) == 10
    assert {-1, 1} <= {d for r in records for d in r["directions"]}
    assert any(r["noise"] == 0 for r in records) and any(r["noise"] >= 3 for r in records)
    assert any(r["terminals"] == 0 for r in records)
    assert any(r["terminals"] == r["count"] for r in records)
    assert any(len(r["inks"]) >= 3 for r in records)
    assert max(max(r["shape"]) for r in records) == 30
    assert len({r["gaps"] for r in records}) >= 20
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(314159, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Output corruption is rejected, not merely a wrong schema or palette.
    puzzle = generate_puzzle(999, 2, 1)
    g = puzzle["test"][0]["output"]
    g[0, 0] = (int(g[0, 0])+1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
