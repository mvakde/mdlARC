"""An intrinsic clock on a route described by anchored miniature strokes.

Only NumPy and the standard library are used. No files, examples, or latent
vocabulary enter solve(). Generation and all diagnostics are bounded.
"""
from functools import lru_cache

import numpy as np


# (dr, dc, length); the first pixel of EVERY stroke is below the endpoint.
CATALOG = tuple((dr, dc, n)
                for dr, dc in ((0, -1), (0, 1), (1, 0), (1, -1), (1, 1))
                for n in (range(2, 5) if dr == 0 else range(2, 4)))
HOST_SEEDS = (0, 439832247, 176835301, 771212246, 920012184, 96564914,
              203479782, 538590117, 548216222, 278689224, 240967242,
              940646452, 789748277, 986935809, 550283274, 790246863,
              884235550, 998937217, 943003792, 577473355, 585565080,
              71768823, 91797980, 441297128, 718188007, 243192860,
              87980564, 6928003, 892640852, 381784674, 758875515,
              834704916, 762946467, 813049609, 475347793, 257438448,
              600055473, 727623585, 964943064, 769495389, 614205966,
              970040179)


def _integer(name, value, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(name + " is out of range")
    return value


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid grid color")
    return a


def _offsets(command):
    dr, dc, n = command
    return [(1 + dr * k, dc * k) for k in range(n)]


def _trace(program, start, height, width):
    """Ordered coordinates, command identities, and command start indices."""
    r, c = 0, start
    route, ids, starts = [], [], []
    for j, command in enumerate(program):
        starts.append(len(route))
        cells = [(r + dy, c + dx) for dy, dx in _offsets(command)]
        _need(all(0 <= y < height and 0 <= x < width for y, x in cells),
              "program leaves the pane")
        route.extend(cells)
        ids.extend([j] * len(cells))
        r, c = cells[-1]
    _need(len(set(route)) == len(route), "route intersects itself")
    return route, ids, starts


def _read(input_grid):
    """Recover geometry and all color roles from this input alone."""
    inp = _grid(input_grid)
    bg = int(inp[0, 0])
    dividers = [c for c in range(1, inp.shape[1] - 1)
                if inp[0, c] != bg and np.all(inp[:, c] == inp[0, c])]
    _need(len(dividers) == 1, "divider must be unique")
    sep = dividers[0]
    divider = int(inp[0, sep])
    panel, pane = inp[:, :sep], inp[:, sep + 1:]
    top_marks = np.flatnonzero(pane[0] != bg)
    _need(len(top_marks) == 1, "pane needs one row-zero seed")
    start = int(top_marks[0])
    marker = int(pane[0, start])
    _need(marker != divider, "seed and divider must differ")
    anchors_x = [int(c) for c in np.flatnonzero(panel[0] == marker)]
    _need(1 <= len(anchors_x) <= 3 and anchors_x[0] == 3,
          "invalid lane anchors")
    _need(anchors_x[-1] + 4 == sep, "invalid panel width")
    xgaps = {b - a - 7 for a, b in zip(anchors_x, anchors_x[1:])}
    _need(not xgaps or (len(xgaps) == 1 and xgaps <= {1, 2}), "invalid lane gaps")
    anchors_y = [int(r) for r in np.flatnonzero(panel[:, 3] == marker)]
    _need(2 <= len(anchors_y) <= 5 and anchors_y[0] == 0,
          "invalid band anchors")
    ygaps = {b - a - 4 for a, b in zip(anchors_y, anchors_y[1:])}
    _need(len(ygaps) == 1 and ygaps <= {1, 2}, "invalid band gaps")
    _need(anchors_y[-1] + 4 <= inp.shape[0], "truncated glyph")
    expected_panel = np.full(panel.shape, bg, dtype=inp.dtype)
    program, inks = [], []
    for ax in anchors_x:
        for ay in anchors_y:
            block = panel[ay:ay + 4, ax - 3:ax + 4]
            _need(block.shape == (4, 7) and block[0, 3] == marker,
                  "missing glyph anchor")
            ink_cells = (block != bg) & (block != marker)
            ink_values = np.unique(block[ink_cells])
            _need(len(ink_values) == 1 and int(ink_values[0]) != divider,
                  "glyph needs one nonstructural ink color")
            ink = int(ink_values[0])
            coords = set((int(y), int(x) - 3) for y, x in np.argwhere(ink_cells))
            matches = [command for command in CATALOG if set(_offsets(command)) == coords]
            _need(len(matches) == 1, "unknown miniature stroke")
            program.append(matches[0])
            inks.append(ink)
            expected_panel[ay, ax] = marker
            for dy, dx in _offsets(matches[0]):
                expected_panel[ay + dy, ax + dx] = ink
    _need(np.array_equal(panel, expected_panel), "unexpected program marks")
    route, ids, starts = _trace(program, start, *pane.shape)
    index_at = {pos: i for i, pos in enumerate(route)}
    pins = {}
    for y, x in np.argwhere(pane != bg):
        pos = int(y), int(x)
        if pos == (0, start):
            continue
        _need(pos in index_at, "pin is outside the route")
        color = int(pane[pos])
        _need(color not in (marker, divider), "pin uses a structural color")
        pins[index_at[pos]] = color
    _need(2 <= len(pins) <= 5, "invalid pin count")
    counts = {c: list(pins.values()).count(c) for c in set(pins.values())}
    repeated = [c for c, n in counts.items() if n > 1]
    _need(len(repeated) == 1 and counts[repeated[0]] == 2,
          "reference color must occur exactly twice")
    ref = repeated[0]
    a, b = sorted(i for i, c in pins.items() if c == ref)
    period = b - a
    aligned = [i for i, c in pins.items() if c != ref and (i - a) % period == 0]
    _need(len(aligned) <= 1, "multiple terminals")
    terminal = aligned[0] if aligned else None
    _need(terminal is None or terminal < a or terminal > b,
          "terminal is between references")
    return {"pane": pane.copy(), "bg": bg, "divider": divider, "marker": marker,
            "start": start, "program": tuple(program), "inks": tuple(inks),
            "lanes": len(anchors_x), "bands": len(anchors_y),
            "gaps": (next(iter(ygaps)), next(iter(xgaps)) if xgaps else 0),
            "route": route, "ids": ids, "starts": starts, "pins": pins,
            "a": a, "b": b, "period": period, "ref": ref, "terminal": terminal,
            "noise": tuple(i for i, c in pins.items() if c != ref and i != terminal)}


def _paint(info, *, reset=False, solid=False):
    out = info["pane"].copy()
    a, b, p, t = (info[k] for k in ("a", "b", "period", "terminal"))
    low, high = (0, len(info["route"]) - 1) if t is None else (min(a, t), max(b, t))
    color = info["ref"] if t is None else info["pins"][t]
    for i, pos in enumerate(info["route"]):
        if solid:
            if out[pos] == info["bg"]:
                out[pos] = info["ref"]
            continue
        coordinate = i - info["starts"][info["ids"][i]] if reset else i
        if low <= i <= high and (coordinate - a) % p == 0:
            out[pos] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Unfold the glyph-defined ruler, complete its clock, fold it back."""
    return _paint(_read(input_grid))


def _vocabulary(rng):
    directions = [(0, -1), (0, 1), (1, 0)]
    if rng.random() < .7:
        directions += [(1, -1), (1, 1)]
    result = [(dr, dc, int(rng.integers(2, 5 if dr == 0 else 4)))
              for dr, dc in directions]
    if len({c[2] for c in result}) == 1:
        result[2] = (1, 0, 3 if result[2][2] == 2 else 2)
    extra_count = int(rng.integers(0, min(2, 6 - len(result)) + 1))
    allowed = [c for c in CATALOG if c[:2] in directions and c not in result]
    for j in rng.permutation(len(allowed))[:extra_count]:
        result.append(allowed[int(j)])
    return tuple(result)


def _program(rng, vocabulary, count, width, min_pixels, require_all):
    """Finite exact reachability, followed by randomized feasible choices."""
    costs = [1 + dr * (n - 1) for dr, dc, n in vocabulary]
    shifts = [dc * (n - 1) for dr, dc, n in vocabulary]
    lengths = [n for dr, dc, n in vocabulary]
    missing = (1 << len(vocabulary)) - 1 if require_all else 0

    @lru_cache(maxsize=None)
    def possible(left, col, room, needed, pixels):
        if left == 0:
            return needed == 0 and pixels == 0
        if left < needed.bit_count() or room < left or pixels > 4 * left:
            return False
        extra_cost = sum(costs[j] - 1 for j in range(len(vocabulary)) if needed & (1 << j))
        if room < left + extra_cost:
            return False
        for j in range(len(vocabulary)):
            end = col + shifts[j]
            if (0 <= end < width and costs[j] <= room
                    and possible(left - 1, end, room - costs[j], needed & ~(1 << j),
                                 max(0, pixels - lengths[j]))):
                return True
        return False

    try:
        starts = [int(x) for x in rng.permutation(width)]
        start = next((c for c in starts if possible(count, c, 29, missing, min_pixels)), None)
        if start is None:
            return None
        sequence = []
        left, col, room, needed, pixels = count, start, 29, missing, min_pixels
        while left:
            legal = []
            for j in range(len(vocabulary)):
                end = col + shifts[j]
                if (0 <= end < width and costs[j] <= room
                        and possible(left - 1, end, room - costs[j], needed & ~(1 << j),
                                     max(0, pixels - lengths[j]))):
                    legal.append(j)
            j = legal[int(rng.integers(len(legal)))]
            sequence.append(vocabulary[j])
            col += shifts[j]
            room -= costs[j]
            needed &= ~(1 << j)
            pixels = max(0, pixels - lengths[j])
            left -= 1
        return tuple(sequence), start
    finally:
        possible.cache_clear()


def _geometry(rng, size, witness):
    if witness == "full":
        a = int(rng.integers(1, size - 2))
        return a, a + 1, None
    if witness == "stop":
        periods = range(2, min(6, (size - 1) // 4) + 1)
        terminal_mode = True
    else:
        periods = range(1, min(9, (size - 1) // 2) + 1)
        terminal_mode = bool(rng.integers(2))
    by_period = {}
    for p in periods:
        options = []
        for a in range(size - p):
            b = a + p
            phase = list(range(a % p, size, p))
            if len(phase) < 3:
                continue
            if not terminal_mode:
                options.append((a, b, None))
                continue
            for t in phase:
                if a <= t <= b:
                    continue
                low, high = min(a, t), max(b, t)
                if not any(i not in (a, b, t) and low <= i <= high for i in phase):
                    continue  # Every bounded clock adds a beat, besides recoloring.
                if witness == "stop" and not any(i < low or i > high for i in phase):
                    continue
                options.append((a, b, t))
        if options:
            by_period[p] = options
    if not by_period:
        return None
    p = list(by_period)[int(rng.integers(len(by_period)))]
    options = by_period[p]
    return options[int(rng.integers(len(options)))]


def _alternative_order(info, order):
    if order == "row":
        program = tuple(info["program"][c * info["bands"] + r]
                        for r in range(info["bands"]) for c in range(info["lanes"]))
    else:
        program = info["program"][::-1]
    try:
        route, _, _ = _trace(program, info["start"], *info["pane"].shape)
    except ValueError:
        return None
    out = info["pane"].copy()
    for pos in route:
        out[pos] = info["ref"]
    return out


def _unchained(info):
    """Ablation: independently stamp each glyph at the original seed."""
    out = info["pane"].copy()
    for command in info["program"]:
        for dy, dx in _offsets(command):
            y, x = dy, info["start"] + dx
            if 0 <= y < out.shape[0] and 0 <= x < out.shape[1] and out[y, x] == info["bg"]:
                out[y, x] = info["ref"]
    return out


def _make_pair(rng, vocabulary, split, ordinal):
    witness = "full" if split == "train" and ordinal == 0 else (
        "stop" if split == "train" and ordinal == 1 else None)
    if witness == "full":
        lanes, bands = 2, 3
    elif witness == "stop":
        layouts = [(2, 3), (3, 2)]
        if len(vocabulary) <= 5:
            layouts.append((1, 5))
        lanes, bands = layouts[int(rng.integers(len(layouts)))]
    elif split == "train":
        layouts = [(1, 4), (1, 5), (2, 2), (2, 3), (3, 2)]
        lanes, bands = layouts[int(rng.integers(len(layouts)))]
    else:
        layouts = [(2, 4), (2, 5), (3, 3), (3, 4)]
        lanes, bands = layouts[int(rng.integers(len(layouts)))]
    ygap, xgap = (int(x) for x in rng.integers(1, 3, size=2))
    panel_w = 7 * lanes + (lanes - 1) * xgap
    max_width = min(18, 29 - panel_w)
    min_width = 8 if witness == "full" else 4
    if max_width < min_width:
        return None
    width = int(rng.integers(min_width, max_width + 1))
    sampled = _program(rng, vocabulary, lanes * bands, width,
                       12 if witness == "stop" else 0,
                       witness is not None or split == "test")
    if sampled is None:
        return None
    program, start = sampled
    route, ids, starts = _trace(program, start, 30, width)
    if witness == "stop" and len({c[2] for c in program}) < 2:
        return None
    height = max(4 * bands + (bands - 1) * ygap, route[-1][0] + 1)
    height += int(rng.integers(0, min(2, 30 - height) + 1))
    geometry = _geometry(rng, len(route), witness)
    if geometry is None:
        return None
    a, b, terminal = geometry
    period = b - a
    roles = [int(c) for c in rng.permutation(10)]
    bg, divider, marker = roles[:3]
    palette = roles[3:]
    ref = palette[0]
    pins = {a: ref, b: ref}
    if terminal is not None:
        pins[terminal] = palette[1]
    noise_options = [i for i in range(len(route)) if (i - a) % period]
    if witness == "stop":
        # A geometrically nearer third color is deliberately inert.
        noise_options = list(range(a + 1, b))
    max_noise = min(2, len(noise_options))
    noise_count = 1 if witness == "stop" else int(rng.integers(max_noise + 1))
    if witness == "full":
        noise_count = 0
    for i, color in zip(rng.choice(noise_options, noise_count, replace=False), palette[2:]):
        pins[int(i)] = color
    inp = np.full((height, panel_w + 1 + width), bg, dtype=np.int8)
    inp[:, panel_w] = divider
    inp[0, panel_w + 1 + start] = marker
    for j, command in enumerate(program):
        lane, band = divmod(j, bands)
        ay, ax = band * (4 + ygap), lane * (7 + xgap) + 3
        inp[ay, ax] = marker
        ink = palette[int(rng.integers(len(palette)))]
        for dy, dx in _offsets(command):
            inp[ay + dy, ax + dx] = ink
    for i, color in pins.items():
        y, x = route[i]
        inp[y, panel_w + 1 + x] = color
    # Labels are constructed from sampled command and index geometry, not solve.
    output = inp[:, panel_w + 1:].copy()
    if terminal is None:
        chosen_indices = range(a % period, len(route), period)
        paint = ref
    else:
        low, high = min(a, terminal), max(b, terminal)
        chosen_indices = range(low, high + 1, period)
        paint = pins[terminal]
    for i in chosen_indices:
        output[route[i]] = paint
    info = _read(inp)
    if witness == "full":
        for order in ("row", "reverse"):
            alternative = _alternative_order(info, order)
            if alternative is not None and np.array_equal(alternative, output):
                return None
    if witness == "stop":
        phase_indices = [i for i in chosen_indices]
        if len({ids[i] for i in phase_indices}) < 2:
            return None
        if not any(program[ids[i]][0] == 0 for i in phase_indices):
            return None
        if np.array_equal(_paint(info, reset=True), output):
            return None
    return {"input": inp, "output": output}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    vocabulary = _vocabulary(rng)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for ordinal in range(count):
            for _ in range(96):
                pair = _make_pair(rng, vocabulary, split, ordinal)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("finite construction budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read(inp)
            expected = _paint(info)
            assert out.shape == info["pane"].shape and np.array_equal(out, expected), "incorrect clock"
            assert np.any((info["pane"] == info["bg"]) & (out != info["bg"])), "no new beat"
            assert out[0, info["start"]] == info["marker"], "seed was changed"
            for i in info["noise"]:
                assert out[info["route"][i]] == info["pins"][i], "off-phase pin changed"
            route_mask = np.zeros(out.shape, dtype=bool)
            for pos in info["route"]:
                route_mask[pos] = True
            assert np.array_equal(out[~route_mask], info["pane"][~route_mask])
            analyses[split].append(info)
    first, second = analyses["train"][:2]
    vocabulary = set(first["program"])
    assert 3 <= len(vocabulary) <= 6
    assert {(0, -1), (0, 1), (1, 0)} <= {c[:2] for c in vocabulary}
    assert first["period"] == 1 and first["terminal"] is None
    assert 0 < first["a"] and first["b"] < len(first["route"]) - 1
    assert set(second["program"]) == vocabulary
    assert second["terminal"] is not None and second["period"] > 1 and second["noise"]
    assert len({c[2] for c in second["program"]}) > 1
    low, high = min(second["a"], second["terminal"]), max(second["b"], second["terminal"])
    phase = list(range(second["a"] % second["period"], len(second["route"]), second["period"]))
    assert any(i < low or i > high for i in phase), "missing stopping witness"
    chosen = [i for i in phase if low <= i <= high]
    assert len({second["ids"][i] for i in chosen}) >= 2
    assert any(second["program"][second["ids"][i]][0] == 0 for i in chosen)
    assert not np.array_equal(_paint(second, reset=True), _paint(second)), "phase-reset ambiguity"
    assert not np.array_equal(_paint(second, solid=True), _paint(second)), "B omission invisible"
    assert not np.array_equal(_unchained(first), _paint(first)), "A omission invisible"
    for order in ("row", "reverse"):
        alternative = _alternative_order(first, order)
        assert alternative is None or not np.array_equal(alternative, _paint(first)), "traversal ambiguity"
    for split in analyses:
        for info in analyses[split]:
            assert set(info["program"]) <= vocabulary, "new glyph in puzzle"
    assert min(len(i["program"]) for i in analyses["test"]) > max(
        len(i["program"]) for i in analyses["train"]), "missing longer-program shift"
    return None


def _puzzle_key(p):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for s in ("train", "test") for pair in p[s])


def self_test():
    """Stream 42 seeds; retain only compact fingerprints and feature sets."""
    signatures, structures, shapes, vocabularies = set(), set(), set(), set()
    periods, backgrounds, gaps, layouts, directions, lengths, modes = (set() for _ in range(7))
    all_turns, max_route, noise_count = 0, 0, 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(hash(_puzzle_key(p)))
        vocabulary = set(_read(p["train"][0]["input"])["program"])
        vocabularies.add(tuple(sorted(vocabulary)))
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                i = _read(inp)
                assert np.array_equal(solve(inp.copy()), out)
                roles, normalized = {}, []
                for v in inp.flat:
                    normalized.append(roles.setdefault(int(v), len(roles)))
                structures.add((inp.shape, bytes(normalized)))
                shapes.add(inp.shape)
                periods.add(i["period"])
                backgrounds.add(i["bg"])
                gaps.add(i["gaps"])
                layouts.add((i["lanes"], i["bands"]))
                directions.update(c[:2] for c in i["program"])
                lengths.update(c[2] for c in i["program"])
                modes.add(0 if i["terminal"] is None else (-1 if i["terminal"] < i["a"] else 1))
                all_turns += sum(a[:2] != b[:2] for a, b in zip(i["program"], i["program"][1:]))
                max_route = max(max_route, len(i["route"]))
                noise_count += len(i["noise"])
    assert len(signatures) == 42 and len(vocabularies) >= 20
    assert len(structures) >= 200 and len(shapes) >= 60
    assert backgrounds == set(range(10)) and {1, 2, 3, 4} <= periods
    assert len(gaps) >= 5 and len(layouts) >= 8
    assert directions == {(0, -1), (0, 1), (1, 0), (1, -1), (1, 1)}
    assert lengths == {2, 3, 4} and modes == {-1, 0, 1}
    assert all_turns > 500 and max_route >= 30 and noise_count > 100
    a = generate_puzzle(12345, 10, 4)
    assert _puzzle_key(a) == _puzzle_key(generate_puzzle(12345, 10, 4))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(17, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 2.5, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # Color-equivariance includes black foreground on a nonblack background.
    p = generate_puzzle(2718, 2, 1)
    permutation = np.array([9, 0, 7, 6, 4, 2, 8, 1, 3, 5], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Detect label damage and an out-of-route pin, using returned grids.
    for damage in ("label", "pin"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad["test"][0]
        info = _read(pair["input"])
        if damage == "label":
            y, x = info["route"][info["a"]]
            pair["output"][y, x] = info["bg"]
        else:
            pane_start = pair["input"].shape[1] - info["pane"].shape[1]
            outside = next((y, x) for y in range(1, info["pane"].shape[0])
                           for x in range(info["pane"].shape[1]) if (y, x) not in set(info["route"]))
            pair["input"][outside[0], pane_start + outside[1]] = info["ref"]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("semantic mutation accepted")
    _intrinsic_phase_regression()
    return None


def _intrinsic_phase_regression():
    """Same pins, shorter early detour: terminal becomes off-phase."""
    # Five commands, one lane. Shortening an out-and-back pair of strokes
    # removes two clock ticks while the routes rejoin at the same endpoint.
    # The common first R4 stroke supplies references at indices 0 and 3.
    programs = (((0, 1, 4), (0, 1, 3), (0, -1, 3), (1, 0, 3), (0, 1, 3)),
                ((0, 1, 4), (0, 1, 2), (0, -1, 2), (1, 0, 3), (0, 1, 3)))
    inputs, outputs = [], []
    for program in programs:
        inp = np.zeros((25, 18), dtype=np.int8)
        inp[:, 7] = 8
        inp[0, 8 + 3] = 9
        for j, command in enumerate(program):
            inp[5 * j, 3] = 9
            for dy, dx in _offsets(command):
                inp[5 * j + dy, 3 + dx] = 7
        for pos, color in (((1, 3), 1), ((1, 6), 1), ((6, 6), 2)):
            inp[pos[0], 8 + pos[1]] = color
        inputs.append(inp)
        outputs.append(solve(inp))
    # Explicit expected cells, independent of route/phase helper functions.
    expected = [np.zeros((25, 10), dtype=np.int8) for _ in programs]
    for out in expected:
        out[0, 3] = 9
    for pos in ((1, 3), (1, 6), (2, 8), (3, 6), (6, 6)):
        expected[0][pos] = 2
    for pos in ((1, 3), (1, 6), (3, 7), (5, 6), (7, 7)):
        expected[1][pos] = 1
    expected[1][6, 6] = 2
    assert all(np.array_equal(a, b) for a, b in zip(outputs, expected))
    assert not np.array_equal(outputs[0], outputs[1])
    assert _read(inputs[0])["terminal"] is not None and _read(inputs[1])["terminal"] is None
