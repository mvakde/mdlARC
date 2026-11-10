"""Periodic quality-control loom: settle only phase-compatible candidates.

Only the standard library and NumPy are used.  The solver reads geometry and
periodic witnesses from its argument; it has no access to construction state.
"""

from __future__ import annotations

import json
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _run_length(row):
    longest = current = 0
    for cell in row:
        current = current + 1 if cell else 0
        longest = max(longest, current)
    return longest


def _axis_period(a, axis):
    for d in range(1, a.shape[axis] // 2 + 1):
        first = a[:-d] if axis == 0 else a[:, :-d]
        second = a[d:] if axis == 0 else a[:, d:]
        if np.array_equal(first, second):
            return d
    return a.shape[axis]


def _layout_candidates(inp):
    """Explain every nonbackground cell by aligned three-layer groups."""
    bg = int(inp[0, 0])
    if not (np.all(inp[0] == bg) and np.all(inp[-1] == bg)
            and np.all(inp[:, 0] == bg) and np.all(inp[:, -1] == bg)):
        return []
    found = []
    for orientation in range(4):
        a = np.rot90(inp, orientation)
        h, w = a.shape
        possible_receivers = []
        for row in range(2, h - 1):
            loc = np.flatnonzero(a[row] != bg)
            if len(loc) >= 6 and int(loc[-1] - loc[0] + 1) == len(loc):
                possible_receivers.append((row, int(loc[0]), int(loc[-1]) + 1))
        for structure in map(int, np.unique(a)):
            if structure == bg:
                continue
            bars = []
            for row, left, right in possible_receivers:
                holes = a[row, left:right] == structure
                if not 1 <= int(holes.sum()) <= right - left - 2:
                    continue
                support = np.full(w, bg, dtype=np.int8)
                source_mask = np.zeros(w, dtype=bool)
                source_mask[left:right] = holes
                support[source_mask] = structure
                if not np.array_equal(a[row - 1], support):
                    continue
                if not np.array_equal(a[row - 2] != bg, source_mask):
                    continue
                if np.any(a[row - 2, source_mask] == structure):
                    continue
                bars.append((row, left, right))
            if len(bars) < 2:
                continue
            if len({(left, right) for _, left, right in bars}) != 1:
                continue
            if any(bars[i + 1][0] - bars[i][0] < 4 for i in range(len(bars) - 1)):
                continue
            expected = np.full(a.shape, bg, dtype=np.int8)
            for row, _, _ in bars:
                expected[row - 2:row + 1] = a[row - 2:row + 1]
            if not np.array_equal(expected, a):
                continue
            left, right = bars[0][1:]
            rows = tuple(row for row, _, _ in bars)
            matrix = a[list(rows), left:right].copy()
            cargo = a[[row - 2 for row in rows], left:right].copy()
            visible = set(map(int, np.unique(matrix[matrix != structure])))
            if len(visible) < 2 or bg in visible:
                continue
            if not set(map(int, np.unique(cargo[cargo != bg]))) <= visible:
                continue
            found.append(dict(grid=a, orientation=orientation, background=bg,
                              structure=structure, rows=rows, left=left,
                              right=right, matrix=matrix, cargo=cargo))
    return found


def _infer_matrix(matrix, structure):
    """Enumerate all fully witnessed period interpretations, rejecting ties."""
    holes = matrix == structure
    axis_options = []
    for axis in range(2):
        choices = []
        for d in range(1, matrix.shape[axis] // 2 + 1):
            a = matrix[:-d] if axis == 0 else matrix[:, :-d]
            b = matrix[d:] if axis == 0 else matrix[:, d:]
            reliable = (a != structure) & (b != structure)
            if np.array_equal(a[reliable], b[reliable]):
                choices.append(d)
        axis_options.append(choices)
    completions = []
    for p in axis_options[0]:
        for q in axis_options[1]:
            complete = matrix.copy()
            valid = True
            for r in range(p):
                for c in range(q):
                    phase = matrix[r::p, c::q]
                    reliable = phase[phase != structure]
                    if reliable.size < 2 or not np.all(reliable == reliable[0]):
                        valid = False
                        break
                    block = complete[r::p, c::q]
                    block[holes[r::p, c::q]] = reliable[0]
                if not valid:
                    break
            if valid:
                completions.append(((p, q), complete))
    if not completions:
        raise ValueError("no fully witnessed periodic completion")
    completions.sort(key=lambda item: (item[0][0] * item[0][1], *item[0]))
    period, full = completions[0]
    if any(not np.array_equal(full, other) for _, other in completions[1:]):
        raise ValueError("periodic completion is ambiguous")
    return full, period


def _analyze(inp):
    layouts = _layout_candidates(inp)
    if len(layouts) != 1:
        raise ValueError("source/support/receiver geometry is not unique")
    a = layouts[0]
    full, period = _infer_matrix(a["matrix"], a["structure"])
    holes = a["matrix"] == a["structure"]
    accepted = holes & (a["cargo"] == full)
    result = a["grid"].copy()
    for i, row in enumerate(a["rows"]):
        cols = a["left"] + np.flatnonzero(accepted[i])
        result[row, cols] = a["cargo"][i, accepted[i]]
        result[row - 2, cols] = a["background"]
    a.update(full=full, period=period, holes=holes, accepted=accepted,
             canonical_output=result,
             output=np.rot90(result, -a["orientation"]).copy())
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the selective settling result, using the input alone."""
    return _analyze(_grid(input_grid))["output"]


def _tile(rng, p, q, palette):
    colors = np.asarray(palette, dtype=np.int8)
    k = len(colors)
    style = str(rng.choice(("random", "weave", "separator", "mirror")))
    for attempt in range(200):
        active = style if attempt < 80 else "random"
        if active == "mirror" and p >= 3 and q >= 3 and k <= ((p + 1) // 2) * ((q + 1) // 2):
            quadrant = rng.choice(colors, size=((p + 1) // 2, (q + 1) // 2))
            rr = np.minimum(np.arange(p), p - 1 - np.arange(p))
            cc = np.minimum(np.arange(q), q - 1 - np.arange(q))
            tile = quadrant[rr[:, None], cc[None, :]].copy()
        elif active == "weave":
            rr = rng.integers(0, k, size=p)
            cc = rng.integers(0, k, size=q)
            tile = colors[(rr[:, None] + cc[None, :]) % k].copy()
        elif active == "separator":
            tile = rng.choice(colors, size=(p, q))
            tile[0] = colors[0]
        else:
            tile = rng.choice(colors, size=(p, q))
            tile.flat[rng.choice(p * q, k, replace=False)] = rng.permutation(colors)
        if set(map(int, tile.flat)) != set(map(int, colors)):
            continue
        if any(np.array_equal(tile, np.roll(tile, -d, axis=0)) for d in range(1, p)):
            continue
        if any(np.array_equal(tile, np.roll(tile, -d, axis=1)) for d in range(1, q)):
            continue
        return tile.astype(np.int8)
    raise RuntimeError("motif sampling exhausted")


def _mask(rng, shape, period, short_runs, wide_run):
    r_count, c_count = shape
    p, q = period
    for _ in range(100):
        mask = np.zeros(shape, dtype=bool)
        # Probe H cannot be reconstructed within its own row.
        h_row = int(rng.integers(r_count))
        h_phase = int(rng.integers(q))
        h_cols = np.arange(h_phase, c_count, q)
        mask[h_row, h_cols] = True
        h_probe = (h_row, int(rng.choice(h_cols)))
        # Probe V cannot be reconstructed within its own column.  A different
        # row phase keeps the two deliberate erasure patterns independent.
        v_phase = int(rng.choice([r for r in range(p) if r != h_row % p]))
        v_col = int(rng.integers(c_count))
        v_rows = np.arange(v_phase, r_count, p)
        mask[v_rows, v_col] = True
        v_probe = (int(rng.choice(v_rows)), v_col)
        if wide_run:
            row = int(rng.integers(r_count))
            length = int(rng.integers(q + 1, min(2 * q + 1, c_count - 2) + 1))
            start = int(rng.integers(c_count - length + 1))
            mask[row, start:start + length] = True

        def legal():
            if np.any(mask.sum(axis=1) > c_count - 2):
                return False
            if short_runs and any(_run_length(row) > 2 for row in mask):
                return False
            return all(np.count_nonzero(~mask[r::p, c::q]) >= 2
                       for r in range(p) for c in range(q))

        if not legal():
            continue
        for row in range(r_count):
            if not mask[row].any():
                for col in rng.permutation(c_count):
                    mask[row, col] = True
                    if legal():
                        break
                    mask[row, col] = False
        if not np.all(mask.any(axis=1)):
            continue
        target = max(int(mask.sum()), int(round(rng.uniform(0.14, 0.34) * mask.size)))
        for flat in rng.permutation(mask.size):
            if mask.sum() >= target:
                break
            row, col = divmod(int(flat), c_count)
            if mask[row, col]:
                continue
            mask[row, col] = True
            if not legal():
                mask[row, col] = False
        return mask, (h_probe, v_probe)
    raise RuntimeError("mask sampling exhausted")


def _candidates(rng, full, holes, probes, palette, bg):
    coords = [tuple(map(int, rc)) for rc in np.argwhere(holes)]
    forced_accept = set(probes)
    color_a = int(full[probes[0]])
    if len({int(full[rc]) for rc in forced_accept}) < 2:
        choices = [rc for rc in coords if rc not in forced_accept and full[rc] != color_a]
        if not choices:
            return None
        forced_accept.add(choices[int(rng.integers(len(choices)))])
    choices = [rc for rc in coords if rc not in forced_accept and full[rc] != color_a]
    if not choices:
        return None
    same_color_reject = choices[int(rng.integers(len(choices)))]
    others = [rc for rc in coords if rc not in forced_accept and rc != same_color_reject]
    if not others:
        return None
    forced_reject = {same_color_reject, others[int(rng.integers(len(others)))]}
    acceptance_rate = float(rng.uniform(0.35, 0.72))
    cargo = np.full(full.shape, bg, dtype=np.int8)
    for rc in coords:
        accept = rc in forced_accept or (rc not in forced_reject and rng.random() < acceptance_rate)
        if accept:
            cargo[rc] = full[rc]
        else:
            cargo[rc] = rng.choice([c for c in palette if c != full[rc]])
    cargo[same_color_reject] = color_a
    return cargo


def _build_pair(rng, shift, split, orientation, shared_roles):
    for _ in range(180):
        if shift == "longer_vertical":
            p = 2 if split == "train" else 3
        else:
            p = int(rng.integers(2, 4))
        if shift == "longer_horizontal":
            q = int(rng.integers(2, 5)) if split == "train" else int(rng.integers(5, 7))
        else:
            q = int(rng.integers(2, 7))
        r_count = int(rng.integers(2 * p, 8))
        c_count = int(rng.integers(max(8, 3 * q), 29))
        if shared_roles is None:
            colors = list(map(int, rng.permutation(10)))
            bg, structure = colors[:2]
            palette = colors[2:2 + int(rng.integers(2, min(5, p * q) + 1))]
        else:
            bg, structure, palette = shared_roles
        tile = _tile(rng, p, q, palette)
        rr = (np.arange(r_count) + int(rng.integers(p))) % p
        cc = (np.arange(c_count) + int(rng.integers(q))) % q
        full = tile[rr[:, None], cc[None, :]].copy()
        if (_axis_period(full, 0), _axis_period(full, 1)) != (p, q):
            continue
        holes, probes = _mask(rng, full.shape, (p, q),
                               shift == "wider_holes" and split == "train",
                               shift == "wider_holes" and split == "test")
        cargo = _candidates(rng, full, holes, probes, palette, bg)
        if cargo is None:
            continue
        receiver = full.copy()
        receiver[holes] = structure
        try:
            recovered, inferred = _infer_matrix(receiver, structure)
        except ValueError:
            continue
        if inferred != (p, q) or not np.array_equal(recovered, full):
            continue
        # Allocate free height among margins and inter-strip gaps.  Their
        # physical spacing need not itself be periodic.
        spare = 30 - (4 * r_count + 1)
        extra = int(rng.integers(min(spare, 5) + 1))
        extras = rng.multinomial(extra, np.full(r_count + 1, 1 / (r_count + 1)))
        front, back = 1 + int(extras[0]), 1 + int(extras[1])
        gaps = [1 + int(v) for v in extras[2:]]
        left = int(rng.integers(1, min(3, 30 - c_count - 1) + 1))
        right = int(rng.integers(1, min(3, 30 - c_count - left) + 1))
        height = front + back + 3 * r_count + sum(gaps)
        inp = np.full((height, left + c_count + right), bg, dtype=np.int8)
        out = inp.copy()
        source = front
        for i in range(r_count):
            inp[source + 2, left:left + c_count] = receiver[i]
            cols = left + np.flatnonzero(holes[i])
            inp[source + 1, cols] = structure
            inp[source, cols] = cargo[i, holes[i]]
            out[source:source + 3] = inp[source:source + 3]
            good = holes[i] & (cargo[i] == full[i])
            dest = left + np.flatnonzero(good)
            out[source, dest] = bg
            out[source + 2, dest] = full[i, good]
            if i < r_count - 1:
                source += 3 + gaps[i]
        inp = np.rot90(inp, -orientation).copy()
        out = np.rot90(out, -orientation).copy()
        try:
            decoded = _analyze(inp)
        except ValueError:
            continue
        if np.array_equal(decoded["output"], out):
            return {"input": inp, "output": out}
    raise RuntimeError(f"unable to construct a readable {split} pair for {shift}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("wider_holes", "longer_horizontal", "longer_vertical")))
    direction = int(rng.integers(4))
    directions = [direction]
    if rng.random() < 0.55:
        directions.append((direction + int(rng.integers(1, 4))) % 4)
    train_directions = directions + [int(rng.choice(directions))
                                     for _ in range(num_train - len(directions))]
    test_directions = [int(rng.choice(directions)) for _ in range(num_test)]
    shared_roles = None
    if rng.random() < 0.25:
        colors = list(map(int, rng.permutation(10)))
        shared_roles = (colors[0], colors[1], colors[2:5])
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, orientations in (("train", train_directions), ("test", test_directions)):
        for orientation in orientations:
            for _ in range(20):
                pair = _build_pair(rng, shift, split, orientation, shared_roles)
                fingerprint = _key(pair["input"])
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    a = _analyze(inp)
    assert np.array_equal(a["output"], out)
    physical_in = a["grid"]
    physical_out = np.rot90(out, a["orientation"])
    matrix, full, holes, accepted = (a[k] for k in ("matrix", "full", "holes", "accepted"))
    rejected = holes & ~accepted
    p, q = a["period"]
    r_count, c_count = matrix.shape
    assert 4 <= r_count <= 7 and 8 <= c_count <= 28
    assert p in (2, 3) and 2 <= q <= 6
    assert r_count >= 2 * p and c_count >= 3 * q
    assert 2 <= len(np.unique(full)) <= 5
    assert np.all(holes.any(axis=1)) and np.all((~holes).sum(axis=1) >= 2)
    assert accepted.sum() >= 2 and rejected.sum() >= 2
    good_colors = set(map(int, a["cargo"][accepted]))
    bad_colors = set(map(int, a["cargo"][rejected]))
    assert len(good_colors) >= 2 and good_colors & bad_colors
    assert np.array_equal(matrix[~holes], full[~holes])
    for r in range(p):
        for c in range(q):
            observed = matrix[r::p, c::q]
            witnesses = observed[observed != a["structure"]]
            assert len(witnesses) >= 2 and np.all(witnesses == witnesses[0])
    # Correct candidates that need witnesses outside their own row/column.
    assert any(holes[r, c % q::q].all() for r, c in np.argwhere(accepted))
    assert any(holes[r % p::p, c].all() for r, c in np.argwhere(accepted))

    expected_changes = np.zeros(inp.shape if a["orientation"] % 2 == 0
                                else inp.T.shape, dtype=bool)
    move_all = physical_in.copy()
    restore_only = physical_in.copy()
    for i, row in enumerate(a["rows"]):
        all_cols = a["left"] + np.flatnonzero(holes[i])
        good_cols = a["left"] + np.flatnonzero(accepted[i])
        bad_cols = a["left"] + np.flatnonzero(rejected[i])
        expected_changes[row, good_cols] = True
        expected_changes[row - 2, good_cols] = True
        assert np.all(physical_out[row - 2, good_cols] == a["background"])
        assert np.array_equal(physical_out[row, good_cols], physical_in[row - 2, good_cols])
        assert np.array_equal(physical_out[row - 2, bad_cols], physical_in[row - 2, bad_cols])
        assert np.all(physical_out[row, bad_cols] == a["structure"])
        assert np.array_equal(physical_in[row - 1], physical_out[row - 1])
        move_all[row - 2, all_cols] = a["background"]
        move_all[row, all_cols] = a["cargo"][i, holes[i]]
        restore_only[row, all_cols] = full[i, holes[i]]
    assert np.array_equal(physical_in != physical_out, expected_changes)
    assert np.count_nonzero(expected_changes) == 2 * int(accepted.sum())
    assert not np.array_equal(move_all, physical_out), "omitting A must change the answer"
    assert not np.array_equal(restore_only, physical_out), "omitting B must change the answer"
    a["max_run"] = max(_run_length(row) for row in holes)
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            analyses[split].append(a)
    train, test = analyses["train"], analyses["test"]
    taught = {a["orientation"] for a in train}
    assert all(a["orientation"] in taught for a in test)
    horizontal_shift = all(a["period"][1] <= 4 for a in train) and all(a["period"][1] >= 5 for a in test)
    vertical_shift = all(a["period"][0] == 2 for a in train) and all(a["period"][0] == 3 for a in test)
    run_shift = all(a["max_run"] <= 2 for a in train) and all(a["max_run"] > a["period"][1] for a in test)
    assert horizontal_shift or vertical_shift or run_shift, "missing the promised visible test shift"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test():
    """Bounded structural, counterfactual, metamorphic and 42-seed checks."""
    shapes, periods, palettes, roles, directions = set(), set(), set(), set(), set()
    normalized, puzzles, hole_counts, good_counts, row_gaps = set(), set(), set(), set(), set()
    mirrors = 0
    shifts = set()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        metadata = {s: [_analyze(_grid(pair["input"])) for pair in puzzle[s]]
                    for s in ("train", "test")}
        if all(a["period"][1] <= 4 for a in metadata["train"]) and all(a["period"][1] >= 5 for a in metadata["test"]):
            shifts.add("longer_horizontal")
        if all(a["period"][0] == 2 for a in metadata["train"]) and all(a["period"][0] == 3 for a in metadata["test"]):
            shifts.add("longer_vertical")
        if all(max(_run_length(row) for row in a["holes"]) <= 2 for a in metadata["train"]) and all(max(_run_length(row) for row in a["holes"]) > a["period"][1] for a in metadata["test"]):
            shifts.add("wider_holes")
        for split in ("train", "test"):
            for pair, a in zip(puzzle[split], metadata[split]):
                pair_count += 1
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                periods.add(a["period"])
                directions.add(a["orientation"])
                roles.add((a["background"], a["structure"]))
                palettes.add(tuple(sorted(map(int, np.unique(a["full"])))))
                hole_counts.add(int(a["holes"].sum()))
                good_counts.add(int(a["accepted"].sum()))
                row_gaps.update(a["rows"][i + 1] - a["rows"][i] for i in range(len(a["rows"]) - 1))
                color_roles = {}
                normalized.add((pair["input"].shape,
                                tuple(color_roles.setdefault(int(c), len(color_roles))
                                      for c in pair["input"].flat)))
                p, q = a["period"]
                tile = a["full"][:p, :q]
                mirrors += (any(np.array_equal(tile, np.roll(tile[::-1], d, axis=0)) for d in range(p))
                            and any(np.array_equal(tile, np.roll(tile[:, ::-1], d, axis=1)) for d in range(q)))
    assert len(puzzles) == 42
    assert len(shapes) >= 45 and len(normalized) >= 190
    assert len(periods) >= 9 and len(palettes) >= 25 and len(roles) >= 25
    assert directions == {0, 1, 2, 3} and len(hole_counts) >= 20 and len(good_counts) >= 15
    assert len(row_gaps) >= 2 and mirrors >= 5
    assert shifts == {"wider_holes", "longer_horizontal", "longer_vertical"}

    # Exercise every legal explicit count combination, including 2/1 and 10/4.
    legal_count_calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ne)
            validate_generated_puzzle(puzzle)
            legal_count_calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{ne}")
    assert _puzzle_key(generate_puzzle(17, 6, 3)) == _puzzle_key(generate_puzzle(17, 6, 3))

    # Geometry/color covariance and a changed hypothesis, with no regeneration.
    example = generate_puzzle(0)["train"][0]
    permutation = np.random.default_rng(9001).permutation(10)
    for rotation in range(4):
        inp = permutation[np.rot90(example["input"], rotation)]
        expected = permutation[np.rot90(example["output"], rotation)]
        assert np.array_equal(solve(inp), expected)
    a = _analyze(example["input"])
    r, c = map(int, np.argwhere(a["holes"] & ~a["accepted"])[0])
    changed = a["grid"].copy()
    changed[a["rows"][r] - 2, a["left"] + c] = a["full"][r, c]
    expected = a["canonical_output"].copy()
    expected[a["rows"][r] - 2, a["left"] + c] = a["background"]
    expected[a["rows"][r], a["left"] + c] = a["full"][r, c]
    assert np.array_equal(solve(np.rot90(changed, -a["orientation"])),
                          np.rot90(expected, -a["orientation"]))
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(puzzles),
                grid_shapes=len(shapes), color_normalized_inputs=len(normalized),
                periods=sorted(periods), palettes=len(palettes),
                role_assignments=len(roles), directions=sorted(directions),
                hole_counts=len(hole_counts), accepted_counts=len(good_counts),
                row_gaps=sorted(row_gaps), reflected_motifs=mirrors,
                shifts=sorted(shifts), legal_count_calls=legal_count_calls)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
