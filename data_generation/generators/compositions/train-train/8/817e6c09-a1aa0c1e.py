"""Broken ladder necklaces: rung multiplicity, contact groups, and parity carry.

Only NumPy and the standard library are used. No parent files are loaded.
"""
from __future__ import annotations

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _path(left, right, y0, y1, elbow):
    """Ordered cable cells, excluding the two colored platforms."""
    cells = [(y0, x) for x in range(left + 3, elbow + 1)]
    if y1 != y0:
        step = 1 if y1 > y0 else -1
        cells += [(y, elbow) for y in range(y0 + step, y1 + step, step)]
    cells += [(y1, x) for x in range(elbow + 1, right)]
    return cells


def _components(a, background, accent):
    """Actual four-neighbor components, independently of cable templates."""
    labels = np.full(a.shape, -1, dtype=int)
    active = (a != background) & (a != accent)
    label = 0
    for sr, sc in np.argwhere(active):
        sr, sc = int(sr), int(sc)
        if labels[sr, sc] >= 0:
            continue
        labels[sr, sc] = label
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if (0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]
                        and active[rr, cc] and labels[rr, cc] < 0):
                    labels[rr, cc] = label
                    stack.append((rr, cc))
        label += 1
    return labels


def _inspect(input_grid):
    a = _grid(input_grid)
    hist = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(hist))
    _require(hist[bg] > a.size / 2, "background must be a strict majority")
    singletons = np.flatnonzero(hist == 1)
    _require(len(singletons) == 1, "one singleton terminal token required")
    accent = int(singletons[0])
    tr, tc = map(int, np.argwhere(a == accent)[0])
    _require(tc >= 2, "terminal lacks its two-cell cable tail")
    cable = int(a[tr, tc - 1])
    _require(cable not in (bg, accent) and a[tr, tc - 2] == cable,
             "terminal must follow two cable cells")
    colors = [int(c) for c in np.flatnonzero(hist) if c not in (bg, accent, cable)]
    _require(2 <= len(colors) <= 5, "two to five ladders required")
    towers = []
    for color in colors:
        cells = np.argwhere(a == color)
        top, left = map(int, cells.min(axis=0))
        base, last_col = map(int, cells.max(axis=0))
        _require(last_col == left + 2, "each ladder is three cells wide")
        flags = []
        for row in range(top, base + 1):
            columns = np.flatnonzero(a[row] == color).tolist()
            _require(columns in ([left, left + 2], [left, left + 1, left + 2]),
                     "invalid ladder row")
            flags.append(len(columns) == 3)
        _require(flags[-1] and not flags[0] and len(flags) >= 4 and not flags[-2],
                 "ladder needs a bottom platform and rail endpoints")
        pattern = flags[:-1]  # The bottom platform is not a rung.
        count = sum(pattern)
        _require(1 <= count <= 3, "one to three crossbars required")
        runs, run = [], 0
        for full in pattern:
            if full:
                _require(run > 0, "crossbars cannot touch vertically")
                runs.append(run)
                run = 0
            else:
                run += 1
        runs.append(run)
        _require(all(1 <= x <= 3 for x in runs), "rail runs must have length one to three")
        towers.append({"left": left, "base": base, "top": top, "color": color,
                       "count": count, "rail_runs": tuple(runs)})
    towers.sort(key=lambda t: t["left"])
    n = len(towers)
    _require(all(towers[i + 1]["left"] - towers[i]["left"] >= 6 for i in range(n - 1)),
             "ladder spans need at least three gutter columns")
    _require((tr, tc) == (towers[-1]["base"], towers[-1]["left"] + 5),
             "terminal must be just right of the rightmost platform")
    tail = {(tr, tc - 2), (tr, tc - 1)}
    actual_cable = {tuple(map(int, p)) for p in np.argwhere(a == cable)}
    accounted = set(tail)
    modes = []
    for i in range(n - 1):
        p, q = towers[i:i + 2]
        first, last = p["left"] + 3, q["left"] - 1
        observed = {cell for cell in actual_cable if first <= cell[1] <= last}
        accounted.update(observed)
        if not observed:
            modes.append("absent")
            continue
        match = None
        for elbow in range(first + 1, last):
            path = _path(p["left"], q["left"], p["base"], q["base"], elbow)
            positions = set(path)
            if not observed <= positions:
                continue
            missing = [j for j, cell in enumerate(path) if cell not in observed]
            if not missing:
                match = "intact"
                break
            if len(missing) <= 2 and missing == list(range(missing[0], missing[-1] + 1)):
                match = "gap"
        _require(match is not None, "cable is not a single elbow path with an optional gap")
        modes.append(match)
    _require(accounted == actual_cable, "stray cable cells")
    labels = _components(a, bg, accent)
    ids = [int(labels[t["base"], t["left"]]) for t in towers]
    groups, start = [], 0
    for i in range(1, n):
        if ids[i] != ids[i - 1]:
            groups.append(tuple(range(start, i)))
            start = i
    groups.append(tuple(range(start, n)))
    _require(len({ids[g[0]] for g in groups}) == len(groups), "interleaved contact groups")
    _require(all((ids[i] == ids[i + 1]) == (modes[i] == "intact") for i in range(n - 1)),
             "unintended contact bypasses a cable gap")
    counts = tuple(t["count"] for t in towers)
    width = 3 * sum(counts) + 2 * len(groups) - 3
    _require(width <= 30, "necklace would exceed ARC width")
    return {"background": bg, "cable": cable, "accent": accent, "towers": towers,
            "counts": counts, "colors": tuple(t["color"] for t in towers),
            "groups": tuple(groups), "modes": tuple(modes), "width": width}


def _bead_runs(m, *, groups=None, local_reset=False, left_anchor=False, paint=True):
    groups = m["groups"] if groups is None else groups
    result = []
    for group in groups:
        beads = []
        for i in group:
            beads.extend((m["colors"][i], i, j) for j in range(m["counts"][i]))
        run = []
        for j, (color, i, local) in enumerate(beads):
            distance = (m["counts"][i] - 1 - local if local_reset
                        else j if left_anchor else len(beads) - 1 - j)
            run.append(m["accent"] if paint and distance % 2 == 0 else color)
        result.append(tuple(run))
    return tuple(result)


def _render(m, **kwargs):
    runs = _bead_runs(m, **kwargs)
    width = 3 * sum(len(run) for run in runs) + 2 * len(runs) - 3
    out = np.full((2, width), m["background"], dtype=np.int8)
    col = 0
    for gi, run in enumerate(runs):
        if gi:
            col += 3
        for j, color in enumerate(run):
            if j:
                col += 1
            out[:, col:col + 2] = color
            col += 2
    assert col == width
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role, rung, and connected group from this input alone."""
    return _render(_inspect(input_grid))


def _pattern(rng, count, equal_height=False):
    if equal_height:
        # Counts 1, 2 and 3 all get seven ladder rows: height cannot encode count.
        runs = [1] * (count + 1)
        for _ in range(7 - count - len(runs)):
            eligible = [i for i, x in enumerate(runs) if x < 3]
            runs[int(rng.choice(eligible))] += 1
    else:
        runs = [int(x) for x in rng.integers(1, 4, size=count + 1)]
    pattern = [False] * runs[0]
    for i in range(count):
        pattern += [True] + [False] * runs[i + 1]
    return pattern


def _build_scene(rng, counts, roles, palette, cuts, *, equal_height=False, gap_size=None):
    """Return the scene plus planned cable paths, for the matched demonstration."""
    n = len(counts)
    bg, cable, accent = roles
    colors = tuple(int(x) for x in rng.choice(palette, n, replace=False))
    patterns = [_pattern(rng, c, equal_height) for c in counts]
    # A three-column gutter has a center elbow safely away from both ladders.
    slack = 30 - (6 * n)  # n*3 + (n-1)*3 + three terminal columns
    gaps = [3] * (n - 1)
    left_margin = 0
    right_margin = 0
    for _ in range(int(rng.integers(0, slack + 1))):
        choice = int(rng.integers(0, n + 1))
        if choice < n - 1:
            gaps[choice] += 1
        elif choice == n - 1:
            left_margin += 1
        else:
            right_margin += 1
    lefts = [left_margin]
    for gap in gaps:
        lefts.append(lefts[-1] + 3 + gap)
    width = lefts[-1] + 6 + right_margin
    reference = max(map(len, patterns)) + int(rng.integers(0, 4))
    bases = [reference + int(rng.integers(0, 5)) for _ in counts]
    height = max(bases) + 1 + int(rng.integers(0, 4))
    a = np.full((height, width), bg, dtype=np.int8)
    for left, base, color, pattern in zip(lefts, bases, colors, patterns):
        a[base, left:left + 3] = color
        for row, full in enumerate(pattern, start=base - len(pattern)):
            a[row, left] = a[row, left + 2] = color
            if full:
                a[row, left + 1] = color
    paths = []
    for i in range(n - 1):
        elbow = int(rng.integers(lefts[i] + 4, lefts[i + 1] - 1))
        path = _path(lefts[i], lefts[i + 1], bases[i], bases[i + 1], elbow)
        paths.append(path)
        mode = cuts.get(i, "intact")
        if mode == "absent":
            continue
        missing = set()
        if mode == "gap":
            size = gap_size if gap_size is not None else int(rng.integers(1, 3))
            start = int(rng.integers(0, len(path) - size + 1))
            missing = set(range(start, start + size))
        for j, cell in enumerate(path):
            if j not in missing:
                a[cell] = cable
    a[bases[-1], lefts[-1] + 3:lefts[-1] + 5] = cable
    a[bases[-1], lefts[-1] + 5] = accent
    # A very short two-ladder scene can otherwise have more ink than background.
    # Preserve the visible majority convention by adding ordinary bottom margin.
    while np.count_nonzero(a == bg) * 2 <= a.size:
        assert a.shape[0] < 30
        a = np.vstack((a, np.full((1, width), bg, dtype=np.int8)))
    m = _inspect(a)
    assert m["counts"] == tuple(counts) and m["colors"] == colors
    assert tuple(i for i, mode in enumerate(m["modes"]) if mode != "intact") == tuple(sorted(cuts))
    return a, paths


def _random_pair(rng, roles, palette, mode, *, n_choices=(2, 3, 4, 5), forbidden_counts=(), small=False):
    for _ in range(300):
        n = int(rng.choice(n_choices))
        counts = tuple(int(x) for x in rng.integers(1, 4, size=n))
        if counts in forbidden_counts:
            continue
        cuts = {}
        if mode in ("gap", "absent"):
            cuts[int(rng.integers(0, n - 1))] = mode
        elif mode == "multi":
            if n < 3:
                continue
            indices = [int(x) for x in rng.choice(n - 1, int(rng.integers(2, n)), replace=False)]
            for j, i in enumerate(indices):
                cuts[i] = "gap" if j % 2 == 0 else "absent"
        groups = len(cuts) + 1
        if 3 * sum(counts) + 2 * groups - 3 > 30:
            continue
        a, _ = _build_scene(rng, counts, roles, palette, cuts, gap_size=1 if small else None)
        return {"input": a, "output": solve(a)}
    raise RuntimeError("bounded pair construction exhausted")


def _legal_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    permutation = [int(x) for x in rng.permutation(10)]
    roles, palette = tuple(permutation[:3]), tuple(permutation[3:])
    n = int(rng.integers(3, 6))
    counts = [1] * n
    two, three = [int(x) for x in rng.choice(n, 2, replace=False)]
    counts[two], counts[three] = 2, 3
    if counts[-1] % 2 == 0:
        odd_at = int(rng.choice([i for i, count in enumerate(counts) if count % 2 == 1]))
        counts[-1], counts[odd_at] = counts[odd_at], counts[-1]
    a, paths = _build_scene(rng, counts, roles, palette, {}, equal_height=True)
    candidates = [i for i in range(n - 1) if sum(counts[i + 1:]) % 2 == 1 and sum(counts[:i + 1]) >= 2]
    cut_at = int(rng.choice(candidates))
    b = a.copy()
    b[paths[cut_at][int(rng.integers(len(paths[cut_at])))]] = roles[0]
    puzzle = {"train": [{"input": a, "output": solve(a)}, {"input": b, "output": solve(b)}], "test": []}
    seen = {_key(a), _key(b)}

    def add(split, mode, **kwargs):
        for _ in range(50):
            pair = _random_pair(rng, roles, palette, mode, **kwargs)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError("unique pair construction exhausted")

    if num_train >= 3:
        add("train", "absent")
    if num_train >= 4:
        add("train", "multi", n_choices=(3, 4, 5))
    for _ in range(4, num_train):
        add("train", str(rng.choice(("all", "gap", "absent", "multi"))))
    trained = [_inspect(p["input"]) for p in puzzle["train"]]
    forbidden = {m["counts"] for m in trained}
    n_choices = tuple(sorted({len(m["counts"]) for m in trained}))
    test_modes = ("all", "gap") if num_train == 2 else (("all", "gap", "absent") if num_train == 3
                                                        else ("all", "gap", "absent", "multi"))
    for _ in range(num_test):
        mode = str(rng.choice(test_modes))
        add("test", mode, n_choices=n_choices, forbidden_counts=forbidden, small=num_train == 2)
        forbidden.add(_inspect(puzzle["test"][-1]["input"])["counts"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check scene geometry, physical contact, parity witnesses and split coverage."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _legal_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, roles, meta = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            m = _inspect(a)
            assert _key(a) not in seen
            seen.add(_key(a))
            roles.add((m["background"], m["cable"], m["accent"]))
            assert np.array_equal(out, _render(m))
            assert out.shape == (2, m["width"]) and np.array_equal(out[0], out[1])
            # Each run contains exactly its rung total in whole square beads.
            col = 0
            for gi, (group, run) in enumerate(zip(m["groups"], _bead_runs(m))):
                if gi:
                    assert np.all(out[:, col:col + 3] == m["background"])
                    col += 3
                assert len(run) == sum(m["counts"][i] for i in group)
                assert run[-1] == m["accent"]
                for j, color in enumerate(run):
                    if j:
                        assert np.all(out[:, col] == m["background"])
                        col += 1
                    assert np.all(out[:, col:col + 2] == color)
                    assert (color == m["accent"]) == ((len(run) - 1 - j) % 2 == 0)
                    col += 2
            assert col == out.shape[1]
            assert not np.array_equal(out, _render(m, paint=False)), "A ablation must change output"
            meta[split].append(m)
    assert len(roles) == 1
    first, second = meta["train"][:2]
    a, b = (np.asarray(p["input"]) for p in puzzle["train"][:2])
    assert a.shape == b.shape and np.count_nonzero(a != b) == 1
    changed = a != b
    assert np.all(a[changed] == first["cable"]) and np.all(b[changed] == first["background"])
    assert first["counts"] == second["counts"] and first["colors"] == second["colors"]
    assert set(first["counts"]) == {1, 2, 3}
    assert len(first["groups"]) == 1 and len(second["groups"]) == 2
    assert all(t["base"] - t["top"] == 7 for t in first["towers"])
    assert sum(second["counts"][i] for i in second["groups"][-1]) % 2 == 1
    left_size = sum(second["counts"][i] for i in second["groups"][0])
    # Compare actual colors before the new output gap, not just output width.
    assert _bead_runs(first)[0][:left_size] != _bead_runs(second)[0]
    assert _bead_runs(first) != _bead_runs(first, local_reset=True)
    assert any(len(run) % 2 == 0 for m in meta["train"] for run in _bead_runs(m))
    assert any(_bead_runs(m) != _bead_runs(m, left_anchor=True) for m in meta["train"])
    # Ignoring the B contact contribution both removes a group gap and mispaints beads.
    one_group = (tuple(range(len(second["counts"]))),)
    assert not np.array_equal(_render(second), _render(second, groups=one_group))
    trained_n = {len(m["counts"]) for m in meta["train"]}
    trained_counts = {m["counts"] for m in meta["train"]}
    shown_modes = {mode for m in meta["train"] for mode in m["modes"]}
    if len(meta["train"]) >= 3:
        assert "absent" in shown_modes
    if len(meta["train"]) >= 4:
        assert any(len(m["groups"]) >= 3 for m in meta["train"])
    test_counts = set()
    for m in meta["test"]:
        assert len(m["counts"]) in trained_n
        assert m["counts"] not in trained_counts | test_counts
        test_counts.add(m["counts"])
        assert set(m["modes"]) <= shown_modes
        if len(meta["train"]) < 4:
            assert len(m["groups"]) <= 2
    return None


def self_test():
    """A bounded 42-seed diversity audit plus all legal count combinations."""
    # Hand-specified semantic fixtures, including the example in task_spec.md.
    rows = ("303000404000505000", "303000444000555000", "303000404000505000",
            "333000444000555000", "303000404000505000", "303000404000555000",
            "303000404000505000", "333888444888555882")
    fixture = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    expected = np.array([[int(c) for c in "33022044022055022"]] * 2, dtype=np.int8)
    assert np.array_equal(solve(fixture), expected)
    fixture[7, 10] = 0
    expected = np.array([[int(c) for c in "2204402200022055022"]] * 2, dtype=np.int8)
    assert np.array_equal(solve(fixture), expected)
    short_rows = ("303000404000", "333000444000", "303000404000",
                  "333888444882", "000000000000", "000000000000")
    short = np.array([[int(c) for c in row] for row in short_rows], dtype=np.int8)
    assert np.array_equal(solve(short), np.array([[3, 3, 0, 2, 2]] * 2))
    shapes, layouts, count_vectors, group_sizes, palettes, signatures = set(), set(), set(), set(), set(), set()
    shifted_bases = absent = stretched = gapped = 0
    test_patterns = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
        signatures.add(tuple(_key(pair["input"]) for s in p for pair in p[s]))
        for split in p:
            for pair in p[split]:
                m = _inspect(pair["input"])
                shapes.add(tuple(pair["input"].shape))
                counts, groups = m["counts"], m["groups"]
                count_vectors.add(counts)
                group_sizes.add(len(groups))
                palettes.add((m["background"], m["cable"], m["accent"]))
                layout = tuple((t["left"], t["base"], t["top"], t["rail_runs"]) for t in m["towers"])
                layouts.add((layout, m["modes"]))
                shifted_bases += len({t["base"] for t in m["towers"]}) > 1
                absent += "absent" in m["modes"]
                gapped += "gap" in m["modes"]
                stretched += any(max(t["rail_runs"]) > 1 for t in m["towers"])
                if split == "test":
                    test_patterns.add((counts, tuple(tuple(g) for g in groups)))
    assert len(signatures) == 42 and len(shapes) >= 25 and len(layouts) >= 120
    assert len(count_vectors) >= 40 and group_sizes >= {1, 2, 3}
    assert len(palettes) >= 30 and len(test_patterns) >= 25
    assert shifted_bases >= 150 and stretched >= 150 and absent >= 42 and gapped >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(104729 + a * 10 + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "layouts": len(layouts), "count_vectors": len(count_vectors),
            "group_counts": sorted(group_sizes), "role_palettes": len(palettes),
            "test_patterns": len(test_patterns), "nonlevel_platform_scenes": shifted_bases,
            "stretched_rail_scenes": stretched, "absent_cable_scenes": absent,
            "gapped_cable_scenes": gapped}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
