"""A cyclic glyph loom transports a separate cyclic color word along a ribbon.

Only standard-library modules and NumPy are used. No parent module is imported.
All semantic roles, orientation, programs and words are reconstructed from pixels.
"""
from __future__ import annotations

import itertools
import numpy as np


# The source A silhouettes are retained, but each stroke pixel becomes a
# transported cross-section: forward progress is made on EVERY slice.
MASKS = (
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
)
SHIFTS = ((0, 1, 1), (0, -1), (0, 0), (0, -1, -1, -1))


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(value):
    a = np.asarray(value, dtype=np.int8)
    return a.shape, a.tobytes()


def _period(word):
    """A complete visible word; non-dividing prefix periods are not permitted."""
    n = len(word)
    return next(p for p in range(1, n + 1) if n % p == 0 and
                all(np.array_equal(word[i], word[i % p]) for i in range(n)))


def _read_down(g, bg, frame_color):
    """Try the orientation in which the unique attached seed projects downward."""
    ys, xs = np.nonzero(g == frame_color)
    assert len(ys)
    t, b, l, r = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    fh, fw = b - t + 1, r - l + 1
    assert fh >= 5 and fw >= 5 and (fh - 1) % 4 == (fw - 1) % 4 == 0
    bands, lanes = (fh - 1) // 4, (fw - 1) // 4
    assert 1 <= bands * lanes <= 6
    expected_frame = np.zeros(g.shape, dtype=bool)
    expected_frame[t, l:r + 1] = expected_frame[b, l:r + 1] = True
    expected_frame[t:b + 1, l] = expected_frame[t:b + 1, r] = True
    assert np.array_equal(g == frame_color, expected_frame)
    assert b + 1 < g.shape[0]
    positions = np.flatnonzero(g[b + 1] != bg)
    assert 1 <= len(positions) <= 3
    left, width = int(positions[0]), len(positions)
    assert np.array_equal(positions, np.arange(left, left + width))
    assert l < left and left + width - 1 < r
    # The port is visibly centered; even widths use the local left half-cell.
    assert left == l + (fw - width) // 2
    claimed = expected_frame.copy()
    program, inks = [], set()
    for lane in range(lanes):
        for band in range(bands):
            y, x = t + 1 + 4 * band, l + 1 + 4 * lane
            block = g[y:y + 3, x:x + 3]
            occupancy = block != bg
            matches = [i for i, m in enumerate(MASKS) if np.array_equal(occupancy, m)]
            assert len(matches) == 1
            colors = np.unique(block[occupancy])
            assert len(colors) == 1 and int(colors[0]) != frame_color
            inks.add(int(colors[0]))
            program.append(matches[0])
            claimed[y:y + 3, x:x + 3] |= occupancy
    word = []
    end = b
    for y in range(b + 1, g.shape[0]):
        row = g[y, left:left + width]
        if np.all(row == bg):
            break
        assert np.all(row != bg)
        word.append(row.copy())
        claimed[y, left:left + width] = True
        end = y
    assert word
    word = np.array(word, dtype=np.int8)
    payload = set(int(x) for x in word.flat)
    assert not payload.intersection({bg, frame_color} | inks)
    assert np.array_equal(claimed, g != bg), "unattached material or interior noise"
    complete = end == g.shape[0] - 1 or left == 0 or left + width == g.shape[1]
    return dict(grid=g, bg=bg, frame=frame_color, box=(t, l, fh, fw),
                bands=bands, lanes=lanes, program=tuple(program), word=word,
                left=left, width=width, end=end, complete=complete,
                period=_period(word), inks=inks)


def _read_input(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "background tie"
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for color in colors:
        if color == bg:
            continue
        for turns in range(4):
            canonical = np.rot90(g, turns).copy()
            try:
                info = _read_down(canonical, bg, int(color))
            except AssertionError:
                continue
            info["turns"] = turns
            candidates.append(info)
    assert len(candidates) == 1, "missing or ambiguous loom/port orientation"
    return candidates[0]


def _trajectory(shape, end, left, width, program, *, once=False):
    """Yield (row, left edge, command index, within-command index, cycle)."""
    row, col, cycle = end, left, 0
    if row == shape[0] - 1 or col == 0 or col + width == shape[1]:
        return
    while True:  # At most 29 slices: forward coordinate increases each time.
        for j, command in enumerate(program):
            for k, dc in enumerate(SHIFTS[command]):
                row, col = row + 1, col + dc
                assert 0 <= row < shape[0] and 0 <= col <= shape[1] - width
                yield row, col, j, k, cycle
                if row == shape[0] - 1 or col == 0 or col + width == shape[1]:
                    return
        if once:
            return
        cycle += 1


def _predict(info, variant="correct"):
    out = info["grid"].copy()
    program = info["program"]
    if variant == "row_major":
        n = info["bands"]
        program = tuple(program[c * n + r] for r in range(n) for c in range(info["lanes"]))
    if variant == "straight":
        program = (2,)
    phase, last_command, last_cycle = 0, None, None
    word = info["word"]
    if variant == "prefix_period":
        # Intentionally wrong ABA -> AB hypothesis, used only for diagnostics.
        p = next(p for p in range(1, len(word) + 1) if
                 all(np.array_equal(word[i], word[i % p]) for i in range(len(word))))
        word = word[:p]
    for row, col, j, k, cycle in _trajectory(out.shape, info["end"], info["left"],
                                            info["width"], program,
                                            once=variant == "once"):
        if variant == "reset_command" and last_command != (cycle, j):
            phase = 0
        if variant == "reset_program" and last_cycle != cycle:
            phase = 0
        values = word[phase % len(word)]
        if variant == "constant":
            values = word[0]
        if variant == "reverse_slice":
            values = values[::-1]
        if variant == "reverse_word":
            values = word[::-1][phase % len(word)]
        assert np.all(out[row, col:col + info["width"]] == info["bg"])
        out[row, col:col + info["width"]] = values
        phase += 1
        last_command, last_cycle = (cycle, j), cycle
    return np.ascontiguousarray(np.rot90(out, -info["turns"]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire output from the one input, with fixed family semantics."""
    return _predict(_read_input(input_grid))


def _layout_options(n):
    return [(b, n // b) for b in range(1, n + 1) if n % b == 0
            and 4 * b + 1 <= 17 and 4 * (n // b) + 1 <= 25]


def _word(rng, period, width, payload, kind=None):
    tile = rng.choice(payload, (period, width)).astype(np.int8)
    if kind == "aba":
        tile[:, :] = payload[0]
        tile[1, :] = payload[1]
    elif kind == "asymmetric":
        tile[0, 0], tile[0, -1] = payload[0], payload[1]
        tile[1] = tile[0][::-1]
        tile[1, width // 2] = payload[1]
    if _period(tile) != period:
        tile[:] = payload[0]
        tile[-1, -1] = payload[1]
    return tile


def _paint(rng, shape, top, frame_left, bands, lanes, program, word, rotation, bg):
    available = [int(x) for x in rng.permutation(10) if int(x) != bg]
    frame, ink_a, ink_b = available[:3]
    payload = available[3:]
    # Construction word uses role indexes; actual colors vary per input.
    colored_word = np.array(payload, dtype=np.int8)[word]
    h, w = 4 * bands + 1, 4 * lanes + 1
    g = np.full(shape, bg, dtype=np.int8)
    g[top, frame_left:frame_left + w] = frame
    g[top + h - 1, frame_left:frame_left + w] = frame
    g[top:top + h, frame_left] = frame
    g[top:top + h, frame_left + w - 1] = frame
    for j, command in enumerate(program):
        lane, band = divmod(j, bands)
        y, x = top + 1 + band * 4, frame_left + 1 + lane * 4
        block = g[y:y + 3, x:x + 3]
        block[MASKS[command]] = int(rng.choice((ink_a, ink_b)))
    width = word.shape[1]
    left = frame_left + (w - width) // 2
    start = top + h
    g[start:start + len(word), left:left + width] = colored_word
    out = g.copy()
    # Generator labels use explicit construction values, not the input parser.
    for phase, (row, col, _, _, _) in enumerate(
            _trajectory(shape, start + len(word) - 1, left, width, program)):
        out[row, col:col + width] = colored_word[phase % len(word)]
    return {"input": np.ascontiguousarray(np.rot90(g, rotation)),
            "output": np.ascontiguousarray(np.rot90(out, rotation))}


def _make_pair(rng, kind, regime, rotation, bg):
    """Finite geometry search; witness constraints apply only to two anchors."""
    if kind == "clock_witness":
        bands, lanes, width, period, copies = 2, 2, 1, 3, 1
        choices = []
        for program in itertools.permutations(range(4)):
            trace = list(_trajectory((30, 30), 11, 4, 1, program))
            if len(trace) >= 13 and trace[-1][1] == 0:
                choices.append(program)
        program = choices[int(rng.integers(len(choices)))]
        trace = list(_trajectory((30, 30), 11, 4, 1, program))
        # Leave a row beyond the side-hit slice so the side boundary is an
        # observable stopping cause, rather than a simultaneous corner hit.
        top = int(rng.integers(0, min(2, 17 - len(trace)) + 1))
        minimum = top + 13 + len(trace)
        shape = (int(rng.integers(minimum, 31)), int(rng.integers(16, 31)))
        frame_left = 0
        word = _word(rng, period, width, [0, 1], "aba")
    elif kind == "vector_witness":
        bands, lanes, width, period = 1, 4, 3, 2
        copies = int(rng.integers(1, 3))
        program = tuple(int(x) for x in rng.permutation(4))
        shape = (int(rng.integers(26, 31)), int(rng.integers(22, 31)))
        top = int(rng.integers(0, 3))
        feasible = []
        for frame_left in range(shape[1] - 17 + 1):
            trace = list(_trajectory(shape, top + 5 + period * copies - 1,
                                     frame_left + 7, width, program))
            if len(trace) >= 12 and trace[-1][0] == shape[0] - 1:
                feasible.append(frame_left)
        assert feasible
        frame_left = int(rng.choice(feasible))
        word = _word(rng, period, width, [0, 1, 2], "asymmetric")
    else:
        if kind == "test" and regime == 1:
            n, period, width = 6, int(rng.integers(2, 4)), int(rng.integers(1, 4))
        elif kind == "test" and regime == 0:
            n, period, width = int(rng.integers(1, 5)), int(rng.integers(4, 6)), int(rng.integers(1, 4))
        elif kind == "test":
            n, period, width = int(rng.integers(1, 4)), 3, 3
        else:
            n, period, width = int(rng.integers(1, 5)), int(rng.integers(1, 4)), int(rng.integers(1, 4))
            if regime == 2 and width == period == 3:
                width = 2
        copies = int(rng.integers(1, 3))
        minimum_extension = max(4, period + 1) if kind == "test" else 1
        options = [(b, l) for b, l in _layout_options(n)
                   if 4 * b + 1 + copies * period + minimum_extension <= 30]
        if not options:
            copies = 1
            options = [(b, l) for b, l in _layout_options(n)
                       if 4 * b + 1 + period + minimum_extension <= 30]
        bands, lanes = options[int(rng.integers(len(options)))]
        fh, fw = 4 * bands + 1, 4 * lanes + 1
        shape = (int(rng.integers(max(14, fh + copies * period + minimum_extension), 31)),
                 int(rng.integers(max(14, fw + 2), 31)))
        top = int(rng.integers(0, shape[0] - fh - copies * period - minimum_extension + 1))
        commands = [int(x) for x in rng.integers(0, 4, n)]
        if kind == "test":
            commands[0] = int(rng.choice((0, 1, 3)))
        program = tuple(commands)
        feasible = []
        for candidate_left in range(shape[1] - fw + 1):
            trace = list(_trajectory(shape, top + fh + copies * period - 1,
                                     candidate_left + (fw - width) // 2, width, program))
            if len(trace) >= minimum_extension:
                feasible.append(candidate_left)
        assert feasible
        frame_left = int(rng.choice(feasible))
        palette_count = int(rng.integers(2, 7))
        word = _word(rng, period, width, list(range(palette_count)))
    word = np.tile(word, (copies, 1))
    return _paint(rng, shape, top, frame_left, bands, lanes, program, word, rotation, bg)


def _unique(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    colors = [int(c) for c in np.unique(pair["input"])]
    # More injective palette assignments than the maximum 14 inputs.
    for values in itertools.permutations(range(10), len(colors)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = values
        candidate = {k: mapping[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("finite palette repair exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    backgrounds = [int(c) for c in rng.permutation(10)]
    rotation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + nq):
        split = "train" if i < nt else "test"
        kind = "clock_witness" if i == 0 else "vector_witness" if i == 1 else split
        pair = _make_pair(rng, kind, regime, (rotation + i) % 4, backgrounds[i % 10])
        pair = _unique(pair, seen)
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    # Order is not an instruction; the two required witnesses may appear anywhere.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    trace = list(_trajectory(info["grid"].shape, info["end"], info["left"],
                             info["width"], info["program"]))
    return dict(shape=info["grid"].shape, layout=(info["bands"], info["lanes"]),
                width=info["width"], period=info["period"], visible=len(info["word"]),
                program=info["program"], command_period=sum(len(SHIFTS[k]) for k in info["program"]),
                extension=len(trace), laps=max((x[4] for x in trace), default=-1) + 1,
                boundary="none" if not trace else "forward" if trace[-1][0] == info["grid"].shape[0] - 1 else "side",
                bg=info["bg"], turns=info["turns"], frame=info["frame"],
                position=info["box"][:2], complete=info["complete"])


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    rejected = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read_input(inp)
            assert np.array_equal(out, _predict(info)), "wrong loom output"
            # Reconstruct the exact writable region separately from color prediction.
            canonical_out = np.rot90(out, info["turns"])
            writable = np.zeros(info["grid"].shape, dtype=bool)
            trace = list(_trajectory(writable.shape, info["end"], info["left"],
                                     info["width"], info["program"]))
            for row, col, j, k, cycle in trace:
                writable[row, col:col + info["width"]] = True
            assert np.all(info["grid"][writable] == info["bg"])
            assert np.array_equal(canonical_out[~writable], info["grid"][~writable])
            assert np.count_nonzero(out != inp) == len(trace) * info["width"]
            infos[split].append(info)
            if split == "test":
                assert len(trace) >= max(4, info["period"] + 1)
                assert not np.array_equal(out, _predict(info, "straight")), "test ignores A"
                assert not np.array_equal(out, _predict(info, "constant")), "test ignores B's cycle"
            if split == "train":
                for variant in ("straight", "constant", "once", "row_major", "reset_command",
                                "reset_program", "prefix_period", "reverse_slice", "reverse_word"):
                    if not np.array_equal(out, _predict(info, variant)):
                        rejected.add(variant)
    assert rejected == {"straight", "constant", "once", "row_major", "reset_command",
                        "reset_program", "prefix_period", "reverse_slice", "reverse_word"}, rejected
    training = infos["train"]
    assert set(k for i in training for k in i["program"]) == set(range(4))
    assert any(i["width"] == 1 and i["period"] == 3 and _features(i)["laps"] >= 2 for i in training)
    assert any(i["width"] == 3 and i["period"] == 2 for i in training)
    assert { _features(i)["boundary"] for i in training } >= {"side", "forward"}
    assert len({i["bg"] for i in training}) >= 2
    tf, qf = [_features(i) for i in training], [_features(i) for i in infos["test"]]
    # Actual pixels must expose at least one of the three documented shifts.
    longer = min(x["period"] for x in qf) > max(x["period"] for x in tf)
    more_commands = min(len(x["program"]) for x in qf) > max(len(x["program"]) for x in tf)
    combined = all(x["width"] == 3 and x["period"] == 3 for x in qf) and not any(
        x["width"] == 3 and x["period"] == 3 for x in tf)
    assert longer or more_commands or combined, "missing train/test shift"


def self_test():
    """Bounded semantic regressions, 42-seed diversity, counts and mutation tests."""
    assert _period(np.array([[2], [3], [2]])) == 3
    assert _period(np.array([[2], [3], [2], [3]])) == 2
    records, keys, normalized = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        keys.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _read_input(pair["input"])
                records.append(_features(info))
                # Normalize semantic roles and payloads in raster order.
                renaming = {}
                normalized.add((info["grid"].shape, tuple(
                    renaming.setdefault(int(c), len(renaming)) for c in info["grid"].flat)))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    assert len(keys) == 42 and len(normalized) >= 160
    assert len({r["shape"] for r in records}) >= 70
    assert {r["width"] for r in records} == {1, 2, 3}
    assert {r["period"] for r in records} == {1, 2, 3, 4, 5}
    assert {r["turns"] for r in records} == {0, 1, 2, 3}
    assert {r["bg"] for r in records} == set(range(10))
    assert len({r["program"] for r in records}) >= 70
    assert len({r["layout"] for r in records}) >= 8
    assert len({r["position"] for r in records}) >= 25
    assert len({r["extension"] for r in records}) >= 18
    assert any(r["visible"] > r["period"] for r in records)
    assert any(r["laps"] >= 3 for r in records)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(91, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(5, 2, 1), generate_puzzle(5, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    # Equivariance: port-relative glyphs and vector slices rotate together.
    pair = p["test"][0]
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns))
    # A complete original seed, even with an irregular tail, is authoritative.
    rng = np.random.default_rng(7)
    complete = _paint(rng, (14, 14), 0, 4, 1, 1, (0,),
                      np.array([[0], [1], [0], [1], [0], [1], [0], [1], [2]]), 0, 9)
    assert np.array_equal(complete["input"], complete["output"])
    assert np.array_equal(solve(complete["input"]), complete["input"])
    # Analytical fixture: right-three repeated with ABA, carrying phase across
    # command boundaries. Independent coordinate list, no generated labels.
    fixture = _paint(rng, (16, 16), 0, 3, 1, 1, (0,), np.array([[0], [1], [0]]), 0, 9)
    info = _read_input(fixture["input"])
    expected = fixture["input"].copy()
    columns = [5, 6, 7, 7, 8, 9, 9, 10]
    for j, col in enumerate(columns):
        expected[8 + j, col] = info["word"][j % 3, 0]
    assert np.array_equal(solve(fixture["input"]), expected)
    # All four commands, independent clock phases, second lap, and a side stop
    # with blank rows beyond it. This is the worked example in task_spec.md.
    fixture = _paint(rng, (28, 16), 0, 0, 2, 2, (3, 0, 1, 2),
                     np.array([[0], [1], [0]]), 0, 9)
    info = _read_input(fixture["input"])
    expected = fixture["input"].copy()
    columns = [4, 3, 2, 1, 1, 2, 3, 3, 2, 2, 2, 2, 1, 0]
    color_indices = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    for j, (col, color_index) in enumerate(zip(columns, color_indices)):
        expected[12 + j, col] = info["word"][color_index, 0]
    assert np.array_equal(solve(fixture["input"]), expected)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({r["shape"] for r in records}),
            "programs": len({r["program"] for r in records}),
            "normalized_structures": len(normalized)}


if __name__ == "__main__":
    print(self_test())
