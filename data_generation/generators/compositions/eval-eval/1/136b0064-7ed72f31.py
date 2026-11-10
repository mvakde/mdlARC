"""ARC echo programs: a colored drawing history and its cursor share symmetries.

Only standard-library modules and NumPy are used. No parent generator is imported.
The solver reads the program and its color-addressed gates entirely from input.
"""
from __future__ import annotations

import itertools
import numpy as np


_MASK_ROWS = (
    ("110", "101", "010"),
    ("101", "101", "111"),
    ("101", "010", "010"),
    ("111", "010", "101"),
    ("100", "010", "001"),
    ("001", "010", "100"),
)
_MASKS = tuple(np.array([[int(c) for c in row] for row in rows], dtype=bool)
               for rows in _MASK_ROWS)
_STROKES = ((0, 1, 3), (0, -1, 2), (1, 0, 2), (0, -1, 4),
            (1, 1, 3), (1, -1, 3))
_MASK_LOOKUP = {tuple(map(int, mask.flat)): k for k, mask in enumerate(_MASKS)}
_ALTERNATIVES = ("no_strokes", "no_echo", "no_cursor", "recent_only",
                 "deferred", "translation", "row_major")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(1 <= d <= 30 for d in g.shape)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("grids must be integer 2D arrays, size 1..30, colors 0..9")
    return g.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, group = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in cells:
                        cells.remove(q)
                        stack.append(q)
                        group.append(q)
        groups.append(sorted(group))
    return groups


def _bbox(cells):
    cells = list(cells)
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_at_divider(grid, divider, bg):
    panel, canvas = grid[:, :divider], grid[:, divider + 1:]
    sep = int(grid[0, divider])
    _require(panel.shape[1] >= 3 and canvas.shape[1] >= 3, "panes too narrow")
    tokens = []
    for cells in _components(panel != bg):
        colors = {int(panel[q]) for q in cells}
        _require(len(colors) == 1, "a glyph must be monochrome")
        color = colors.pop()
        _require(color != sep, "divider color in program")
        r0, r1, c0, c1 = _bbox(cells)
        if len(cells) == 1:
            top, left, code = r0 - 1, c0 - 1, None
        else:
            _require((r1 - r0, c1 - c0) == (2, 2), "unknown glyph extent")
            bits = tuple(map(int, (panel[r0:r1 + 1, c0:c1 + 1] != bg).flat))
            _require(bits in _MASK_LOOKUP, "unknown stroke glyph")
            top, left, code = r0, c0, _MASK_LOOKUP[bits]
        _require(0 <= top <= grid.shape[0] - 3 and 0 <= left <= divider - 3,
                 "instruction slot outside panel")
        tokens.append({"top": top, "left": left, "color": color, "code": code})
    _require(bool(tokens) and any(t["code"] is not None for t in tokens), "empty program")
    tokens.sort(key=lambda t: (t["left"], t["top"]))
    # Full 3x3 slots must have a blank-cell separation, including singleton calls.
    for a, b in itertools.combinations(tokens, 2):
        _require(abs(a["left"] - b["left"]) >= 4 or abs(a["top"] - b["top"]) >= 4,
                 "instruction slots touch")
    calls = [t["color"] for t in tokens if t["code"] is None]
    _require(1 <= len(calls) <= 2 and len(set(calls)) == len(calls), "calls need unique addresses")
    _require(not any(t["color"] in calls for t in tokens if t["code"] is not None),
             "stroke and address colors overlap")
    gates = {}
    for color in calls:
        cells = list(map(tuple, np.argwhere(canvas == color).tolist()))
        _require(bool(cells), "missing called gate")
        r0, r1, c0, c1 = _bbox(cells)
        if len(cells) == 1:
            kind = "P"
        elif 2 <= len(cells) <= 7 and r0 == r1 and c1 - c0 + 1 == len(cells):
            kind = "H"
        elif 2 <= len(cells) <= 7 and c0 == c1 and r1 - r0 + 1 == len(cells):
            kind = "V"
        else:
            raise ValueError("gate must be a point or contiguous straight line")
        gates[color] = {"kind": kind, "cells": cells, "a": r0, "b": c0}
    marks = [(int(r), int(c)) for r, c in np.argwhere(canvas != bg)
             if int(canvas[r, c]) not in gates]
    _require(len(marks) == 1, "canvas must have one seed besides its gates")
    seed = marks[0]
    seed_color = int(canvas[seed])
    _require(seed_color != sep and seed_color not in {t["color"] for t in tokens},
             "seed role is not distinct")
    return {"background": bg, "separator": sep, "canvas": canvas.copy(),
            "seed": seed, "seed_color": seed_color, "tokens": tokens, "gates": gates}


def _read_input(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background tied")
    bg = int(values[counts.argmax()])
    candidates = []
    for c in range(3, grid.shape[1] - 3):
        if grid[0, c] != bg and np.all(grid[:, c] == grid[0, c]):
            try:
                candidates.append(_read_at_divider(grid, c, bg))
            except ValueError:
                pass
    _require(len(candidates) == 1, "divider/role interpretation must be unique")
    return candidates[0]


def _transform(q, gate):
    r, c = q
    kind, a, b = gate["kind"], gate["a"], gate["b"]
    if kind == "H":
        return 2 * a - r, c
    if kind == "V":
        return r, 2 * b - c
    return 2 * a - r, 2 * b - c


def _translation(source, gate):
    r0, _, c0, _ = _bbox(source)
    d0, _, e0, _ = _bbox(_transform(q, gate) for q in source)
    return d0 - r0, e0 - c0


def _orientation_witness(source, gate):
    dr, dc = _translation(source, gate)
    translated = {(r + dr, c + dc): color for (r, c), color in source.items()}
    turned = {_transform(q, gate): color for q, color in source.items()}
    return translated != turned


def _execute(info, alternative=None, with_trace=False):
    """Execute from observed roles; alternatives are validation-only ablations."""
    out = info["canvas"].copy()
    bg = info["background"]
    cursor = info["seed"]
    ink = {cursor: info["seed_color"]}
    recent = dict(ink)
    controls = {q for gate in info["gates"].values() for q in gate["cells"]}
    tokens = list(info["tokens"])
    if alternative == "deferred":
        tokens = [t for t in tokens if t["code"] is not None] + [
            t for t in tokens if t["code"] is None]
    elif alternative == "row_major":
        tokens.sort(key=lambda t: (t["top"], t["left"]))
    events = []

    def add(cells):
        for q, color in cells.items():
            r, c = q
            _require(0 <= r < out.shape[0] and 0 <= c < out.shape[1], "ink leaves canvas")
            _require(q not in controls, "ink collides with a control marker")
            if alternative is None:
                _require(q not in ink, "new ink collides with previous ink")
        ink.update(cells)

    for token in tokens:
        code, color = token["code"], token["color"]
        if code is not None:
            if alternative == "no_strokes":
                continue
            r, c = cursor
            dr, dc, n = _STROKES[code]
            coords = [(r + 1 + j * dr, c + j * dc) for j in range(n)]
            cells = {q: color for q in coords}
            add(cells)
            recent.update(cells)
            if with_trace:
                events.append({"op": "stroke", "code": code, "color": color,
                               "cells": cells, "before": cursor, "after": coords[-1]})
            cursor = coords[-1]
        else:
            if alternative == "no_echo":
                continue
            gate = info["gates"][color]
            source = dict(recent if alternative == "recent_only" else ink)
            before = cursor
            if alternative == "translation":
                dr, dc = _translation(source, gate)
                copied = {(r + dr, c + dc): v for (r, c), v in source.items()}
                mapped_cursor = before[0] + dr, before[1] + dc
            else:
                copied = {_transform(q, gate): v for q, v in source.items()}
                mapped_cursor = _transform(before, gate)
            add(copied)
            if alternative != "no_cursor":
                cursor = mapped_cursor
            recent = {cursor: ink[cursor]}
            if with_trace:
                events.append({"op": "echo", "gate": gate, "color": color,
                               "source": source, "copied": copied,
                               "before": before, "after": cursor})
    for q, color in ink.items():
        out[q] = color
    return (out, events) if with_trace else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive output from the input alone, without training data or latent state."""
    return _execute(_read_input(input_grid))


def _disagrees(info, out, alternative):
    try:
        return not np.array_equal(_execute(info, alternative), out)
    except ValueError:
        # An alternative demanding an out-of-bounds cell or writing over a visible
        # control also contradicts the supplied clear-destination demonstration.
        return True


def _valid_disagrees(info, out, alternative):
    """Prefer a visible wrong output over refutation solely by a boundary."""
    try:
        return not np.array_equal(_execute(info, alternative), out)
    except ValueError:
        return False


def _geometry(rng, codes, kinds):
    """Independent relative-coordinate label construction; no call to solve."""
    ink = {(0, 0): -1}  # -1 is the seed; other labels are stroke occurrence IDs.
    cursor = (0, 0)
    controls, gates, program = set(), [], []
    cuts = sorted(int(v) for v in rng.choice(range(1, len(codes)), len(kinds), replace=False))
    for i, code in enumerate(codes):
        dr, dc, n = _STROKES[code]
        r, c = cursor
        cells = [(r + 1 + j * dr, c + j * dc) for j in range(n)]
        if any(q in ink or q in controls for q in cells):
            return None
        ink.update({q: i for q in cells})
        cursor = cells[-1]
        program.append(("stroke", code, i))
        if i + 1 in cuts:
            kind = kinds[cuts.index(i + 1)]
            r0, r1, c0, c1 = _bbox(ink)
            gap = int(rng.choice([1, 2], p=[.85, .15]))
            a = r1 + gap if rng.random() < .5 else r0 - gap
            b = c1 + gap if rng.random() < .5 else c0 - gap
            length = int(rng.integers(2, 8))
            if kind == "H":
                left = int(rng.integers(c0 - 2, c1 + 2))
                marker = [(a, left + j) for j in range(length)]
                copied = {(2 * a - r, c): v for (r, c), v in ink.items()}
                next_cursor = (2 * a - cursor[0], cursor[1])
            elif kind == "V":
                top = int(rng.integers(r0 - 2, r1 + 2))
                marker = [(top + j, b) for j in range(length)]
                copied = {(r, 2 * b - c): v for (r, c), v in ink.items()}
                next_cursor = (cursor[0], 2 * b - cursor[1])
            else:
                marker = [(a, b)]
                copied = {(2 * a - r, 2 * b - c): v for (r, c), v in ink.items()}
                next_cursor = (2 * a - cursor[0], 2 * b - cursor[1])
            if any(q in ink or q in controls for q in marker):
                return None
            if any(q in ink or q in controls or q in marker for q in copied):
                return None
            gate_id = len(gates)
            gates.append({"kind": kind, "cells": marker})
            program.append(("echo", gate_id, gate_id))
            controls.update(marker)
            ink.update(copied)
            cursor = next_cursor
        r0, r1, c0, c1 = _bbox(set(ink) | controls)
        if r1 - r0 + 1 > 30 or c1 - c0 + 1 > 22:
            return None
    r0, r1, c0, c1 = _bbox(set(ink) | controls)
    return {"ink": ink, "gates": gates, "program": program,
            "bounds": (r0, r1, c0, c1)}


def _render(rng, geometry, roles, codes, first_down_color, is_first, recolor_down):
    bg, sep, seed_color = roles[:3]
    address_pool, ink_pool = list(roles[3:5]), list(roles[5:])
    rng.shuffle(address_pool)
    palette_size = 2 if is_first else int(rng.integers(2, 6))
    palette = [int(v) for v in rng.choice(ink_pool, palette_size, replace=False)]
    colors = [palette[i % len(palette)] for i in range(len(codes))]
    rng.shuffle(colors)
    down_positions = [i for i, code in enumerate(codes) if code == 2]
    if recolor_down:
        choices = [c for c in ink_pool if c != first_down_color]
        for i in down_positions:
            colors[i] = int(rng.choice(choices))
    r0, r1, c0, c1 = geometry["bounds"]
    height, width = r1 - r0 + 1, c1 - c0 + 1
    length = len(geometry["program"])
    layouts = []
    for lanes in range(1 if not is_first else 2, 4):
        bands = (length + lanes - 1) // lanes
        for ygap in (1, 2):
            for xgap in (1, 2):
                ph, pw = 3 * bands + ygap * (bands - 1), 3 * lanes + xgap * (lanes - 1)
                if ph <= 30 and pw + 1 + width <= 30:
                    layouts.append((lanes, bands, ygap, xgap, ph, pw))
    if not layouts:
        return None
    lanes, bands, ygap, xgap, ph, pw = layouts[int(rng.integers(len(layouts)))]
    extra_h = int(rng.integers(min(5, 30 - height) + 1))
    extra_w = int(rng.integers(min(5, 29 - pw - width) + 1))
    h, w = max(ph, height + extra_h), width + extra_w
    top = int(rng.integers(h - height + 1))
    left = int(rng.integers(w - width + 1))
    dy, dx = top - r0, left - c0

    def shift(q):
        return q[0] + dy, q[1] + dx

    canvas = np.full((h, w), bg, dtype=np.int8)
    canvas[shift((0, 0))] = seed_color
    for j, gate in enumerate(geometry["gates"]):
        for q in gate["cells"]:
            canvas[shift(q)] = address_pool[j]
    output = canvas.copy()
    for q, occurrence in geometry["ink"].items():
        output[shift(q)] = seed_color if occurrence == -1 else colors[occurrence]
    inp = np.full((h, pw + 1 + w), bg, dtype=np.int8)
    inp[:, pw] = sep
    inp[:, pw + 1:] = canvas
    for i, (op, value, occurrence) in enumerate(geometry["program"]):
        lane, band = divmod(i, bands)
        y, x = band * (3 + ygap), lane * (3 + xgap)
        if op == "echo":
            inp[y + 1, x + 1] = address_pool[value]
        else:
            inp[y:y + 3, x:x + 3][_MASKS[value]] = colors[occurrence]
    return {"input": inp, "output": output}


def _required_codes(rng, active):
    order = [int(v) for v in rng.permutation(active)]
    cut = (len(order) + 1) // 2
    parts = [set(order[:cut]) | {2}, set(order[cut:]) | {2}]
    for part in parts:
        for k in order:
            if len(part) >= 3:
                break
            part.add(k)
    return [sorted(part) for part in parts]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(int(c) for c in rng.permutation(10))
    vocabulary_size = int(rng.integers(3, 7))
    horizontal = int(rng.choice([0, 1, 3]))
    active = [2, horizontal]
    remaining = [k for k in range(6) if k not in active]
    active += [int(v) for v in rng.choice(remaining, vocabulary_size - 2, replace=False)]
    required = _required_codes(rng, active)
    kind_subsets = [("H",), ("V",), ("P",), ("H", "V"), ("H", "P"),
                    ("V", "P"), ("H", "V", "P")]
    kinds = kind_subsets[int(rng.integers(len(kind_subsets)))]
    schedule = list(rng.permutation(kinds))
    while len(schedule) < 4:
        schedule.append(str(rng.choice(kinds)))
    rng.shuffle(schedule)
    gate_plans = [schedule[:2], schedule[2:]]
    puzzle = {"train": [], "test": []}
    seen, first_down_color = set(), None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            is_first = split == "train" and index == 0
            recolor_down = split == "train" and index == 1
            n = (4 if index == 0 else 5 if index == 1 else int(rng.integers(3, 6))) if split == "train" else int(rng.integers(6, 9))
            needed = required[index] if split == "train" and index < 2 else []
            pair_kinds = gate_plans[index] if split == "train" and index < 2 else [
                str(rng.choice(kinds)) for _ in range(2 if split == "test" else int(rng.integers(1, 3)))]
            for attempt in range(2048):
                # Some very short, one-sided programs cannot continue after a
                # particular pair of folds without hitting their own copies.
                # Change the proposal as well as its coordinates, rather than
                # spending the entire search budget on an impossible schedule.
                if attempt and attempt % 32 == 0:
                    if split == "train" and index < 2:
                        rng.shuffle(pair_kinds)
                    else:
                        n = int(rng.integers(3, 6) if split == "train" else rng.integers(6, 9))
                        pair_kinds = [str(rng.choice(kinds)) for _ in range(
                            int(rng.integers(1, 3)) if split == "train" else 2)]
                codes = list(needed) + [int(rng.choice(active)) for _ in range(n - len(needed))]
                rng.shuffle(codes)
                geometry = _geometry(rng, codes, pair_kinds)
                if geometry is None:
                    continue
                pair = _render(rng, geometry, roles, codes, first_down_color, is_first, recolor_down)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                info = _read_input(pair["input"])
                expected, trace = _execute(info, with_trace=True)
                if not np.array_equal(expected, pair["output"]):
                    raise AssertionError("independent construction disagrees with execution")
                if split == "train" and index < 2:
                    if not all(_orientation_witness(e["source"], e["gate"])
                               for e in trace if e["op"] == "echo"):
                        continue
                if is_first and not all(_disagrees(info, expected, alt) for alt in _ALTERNATIVES):
                    continue
                if is_first and not all(_valid_disagrees(info, expected, alt)
                                        for alt in ("no_echo", "no_cursor")):
                    continue
                break
            else:
                raise RuntimeError("bounded clear-destination scene search exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if is_first:
                first_down_color = next(t["color"] for t in info["tokens"] if t["code"] == 2)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles = set(), set()
    train_codes, train_kinds, witnessed = set(), set(), set()
    training_lengths, test_lengths = [], []
    colors_by_code, codes_by_color = {}, {}
    refuted = {alt: False for alt in _ALTERNATIVES}
    visible_refuted = {alt: False for alt in ("no_strokes", "no_echo", "no_cursor", "recent_only")}
    first_traces = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read_input(inp)
            roles.add((info["background"], info["separator"], info["seed_color"]))
            expected, trace = _execute(info, with_trace=True)
            assert np.array_equal(out, expected), "incorrect echo-program output"
            assert out.shape == info["canvas"].shape
            initial = info["canvas"]
            assert np.array_equal(out[initial != info["background"]], initial[initial != info["background"]])
            echoes = [e for e in trace if e["op"] == "echo"]
            strokes = [e for e in trace if e["op"] == "stroke"]
            assert len(echoes) in (1, 2) and strokes
            # Every clear echo doubles the current ink; the seed is never a stroke color.
            assert np.count_nonzero(out == info["seed_color"]) == 2 ** len(echoes)
            for event in echoes:
                assert set(event["source"]).isdisjoint(event["copied"])
                assert len(event["source"]) == len(event["copied"])
                assert sorted(event["source"].values()) == sorted(event["copied"].values())
                assert event["after"] == _transform(event["before"], event["gate"])
            assert trace[0]["op"] == trace[-1]["op"] == "stroke"
            assert all(a["op"] != "echo" or b["op"] != "echo" for a, b in zip(trace, trace[1:]))
            codes = {e["code"] for e in strokes}
            kinds = {e["gate"]["kind"] for e in echoes}
            if split == "train":
                training_lengths.append(len(strokes))
                train_codes.update(codes)
                train_kinds.update(kinds)
                witnessed.update(e["gate"]["kind"] for e in echoes if _orientation_witness(e["source"], e["gate"]))
                for event in strokes:
                    colors_by_code.setdefault(event["code"], set()).add(event["color"])
                    codes_by_color.setdefault(event["color"], set()).add(event["code"])
                for alt in refuted:
                    refuted[alt] |= _disagrees(info, out, alt)
                for alt in visible_refuted:
                    visible_refuted[alt] |= _valid_disagrees(info, out, alt)
                if index < 2:
                    assert len(echoes) == 2
                    first_traces.append(trace)
            else:
                test_lengths.append(len(strokes))
                assert codes <= train_codes and kinds <= train_kinds
    assert len(roles) == 1, "puzzle role convention changed"
    assert 3 <= len(train_codes) <= 6 and 2 in train_codes
    assert train_kinds <= witnessed, "training hides a symmetry behind an unchanged silhouette"
    assert max(training_lengths) <= 5 < min(test_lengths) <= max(test_lengths) <= 8
    assert min(training_lengths) >= 3 and training_lengths[:2] == [4, 5]
    assert len(colors_by_code[2]) >= 2, "shared down glyph never recolored"
    assert any(len(codes) >= 2 for codes in codes_by_color.values()), "colors encode glyph meanings"
    assert all(refuted.values()), f"training permits alternatives: {refuted}"
    assert all(visible_refuted.values()), f"ablations lack a visible in-bounds witness: {visible_refuted}"
    early_codes = {e["code"] for tr in first_traces for e in tr if e["op"] == "stroke"}
    early_kinds = {e["gate"]["kind"] for tr in first_traces for e in tr if e["op"] == "echo"}
    assert train_codes <= early_codes and train_kinds <= early_kinds


def self_test():
    """Bounded 42-seed semantic, variation, count-contract and mutation checks."""
    shapes, vocabularies, mixtures, programs, normalized = set(), set(), set(), set(), set()
    lengths, marker_lengths, sides, call_times = set(), set(), set(), set()
    palettes, overhangs, border_pairs = set(), 0, 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        v, kinds = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                predicted, trace = _execute(info, with_trace=True)
                assert np.array_equal(predicted, pair["output"])
                shapes.add((pair["input"].shape, pair["output"].shape))
                palettes.add((info["background"], info["separator"], info["seed_color"]))
                codes = tuple(t["code"] if t["code"] is not None else info["gates"][t["color"]]["kind"] for t in info["tokens"])
                programs.add(codes)
                lengths.add(sum(t["code"] is not None for t in info["tokens"]))
                call_times.add(tuple(i for i, t in enumerate(info["tokens"]) if t["code"] is None))
                v.update(t["code"] for t in info["tokens"] if t["code"] is not None)
                kinds.update(g["kind"] for g in info["gates"].values())
                color_ids = {}
                normalized.add((pair["input"].shape, tuple(color_ids.setdefault(int(c), len(color_ids)) for c in pair["input"].flat)))
                border_pairs += bool(np.any(predicted[0] != info["background"]) or np.any(predicted[-1] != info["background"]) or np.any(predicted[:, 0] != info["background"]) or np.any(predicted[:, -1] != info["background"]))
                for event in trace:
                    if event["op"] != "echo":
                        continue
                    gate, source = event["gate"], event["source"]
                    marker_lengths.add(len(gate["cells"]))
                    r0, r1, c0, c1 = _bbox(source)
                    if gate["kind"] == "H":
                        sides.add(("H", int(np.sign(gate["a"] - (r0 + r1) / 2))))
                        overhangs += c0 < min(c for r, c in gate["cells"]) or c1 > max(c for r, c in gate["cells"])
                    elif gate["kind"] == "V":
                        sides.add(("V", int(np.sign(gate["b"] - (c0 + c1) / 2))))
                        overhangs += r0 < min(r for r, c in gate["cells"]) or r1 > max(r for r, c in gate["cells"])
                    else:
                        sides.add(("P", int(np.sign(gate["a"] - (r0 + r1) / 2)), int(np.sign(gate["b"] - (c0 + c1) / 2))))
                    # Independent cropped-mask flip/rotation audit of exact colors.
                    tile = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
                    for (r, c), color in source.items():
                        tile[r - r0, c - c0] = color
                    if gate["kind"] == "H":
                        transformed, y, x = np.flipud(tile), 2 * gate["a"] - r1, c0
                    elif gate["kind"] == "V":
                        transformed, y, x = np.fliplr(tile), r0, 2 * gate["b"] - c1
                    else:
                        transformed, y, x = np.rot90(tile, 2), 2 * gate["a"] - r1, 2 * gate["b"] - c1
                    checked = {(y + int(r), x + int(c)): int(transformed[r, c]) for r, c in np.argwhere(transformed >= 0)}
                    assert checked == event["copied"]
        vocabularies.add(tuple(sorted(v)))
        mixtures.add(tuple(sorted(kinds)))
    assert len(shapes) >= 50 and len(programs) >= 100 and len(normalized) >= 190
    assert len(vocabularies) >= 10 and len(mixtures) == 7
    assert lengths == set(range(3, 9)) and marker_lengths == set(range(1, 8))
    assert len(call_times) >= 12 and len(palettes) >= 30
    assert {("H", -1), ("H", 1), ("V", -1), ("V", 1)} <= sides
    assert {("P", r, c) for r in (-1, 1) for c in (-1, 1)} <= sides
    assert overhangs >= 30 and border_pairs >= 20
    assert all(_key(a[role]) == _key(b[role]) for split in ("train", "test")
               for a, b in zip(puzzles[0][split], generate_puzzle(0)[split]) for role in ("input", "output"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(71023 + 17 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt a label, then corrupt a copied seed: both must fail semantic validation.
    for damage in ("stroke", "seed_copy"):
        bad = {split: [{key: value.copy() for key, value in pair.items()} for pair in puzzles[0][split]] for split in ("train", "test")}
        pair = bad["train"][0]
        info = _read_input(pair["input"])
        if damage == "stroke":
            q = next(q for q in map(tuple, np.argwhere(pair["output"] != info["canvas"]).tolist()) if pair["output"][q] != info["seed_color"])
        else:
            q = next(q for q in map(tuple, np.argwhere(pair["output"] == info["seed_color"]).tolist()) if q != info["seed"])
        pair["output"][q] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("semantic corruption accepted")
    # A full finite axis acts beyond its endpoints, and same-coordinate copies
    # preserve colors under each operation. Rotation never rotates later strokes.
    source = {(0, 0): 8, (1, 0): 4, (1, 1): 4, (1, 2): 4}
    gate = {"kind": "H", "a": 3, "b": 1, "cells": [(3, 1), (3, 2)]}
    assert {_transform(q, gate): c for q, c in source.items()} == {(6, 0): 8, (5, 0): 4, (5, 1): 4, (5, 2): 4}
    assert _transform((1, 2), gate) == (5, 2)
    return {"seeds": 42, "pairs": 210, "shape_pairs": len(shapes),
            "programs": len(programs), "vocabularies": len(vocabularies),
            "symmetry_subsets": len(mixtures), "normalized_inputs": len(normalized),
            "overhanging_echoes": int(overhangs), "output_border_pairs": border_pairs}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
