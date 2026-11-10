"""Detour loom: stretch a carrier along a routed ribbon, tile its stencil.

Only NumPy and the standard library are used. The solver reads a single input;
palette choices and sampling regimes never participate in the transformation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DIRECTIONS = ("down", "right", "up", "left")
REQUIRED_EVIDENCE = frozenset({
    "straight", "fixed_clearance", "missing_plus_one", "closest_endpoint",
    "ignore_stencil", "tile_carrier", "stretch_stencil", "opaque_zero",
    "reset_at_bends", "row_phase", "column_phase", "plane_phase",
    "visible_carrier", "visible_overwrite",
})
INTERACTIVE_EVIDENCE = frozenset({
    "ignore_stencil", "tile_carrier", "stretch_stencil", "visible_carrier",
    "visible_overwrite", "reset_at_bends",
})


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= x <= 30 for x in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _from_forward(a, direction):
    if direction == "down":
        return a.copy()
    if direction == "up":
        return a[::-1].copy()
    if direction == "right":
        return a.T.copy()
    return a[::-1].T.copy()


def _to_forward(a, direction):
    if direction == "down":
        return a.copy()
    if direction == "up":
        return a[::-1].copy()
    if direction == "right":
        return a.T.copy()
    return a.T[::-1].copy()


def _walk(vertices):
    """Ordered cells of an orthogonal polyline; count each bend just once."""
    points = [tuple(vertices[0])]
    for end in vertices[1:]:
        r, c = points[-1]
        er, ec = map(int, end)
        if r != er and c != ec:
            raise ValueError("nonorthogonal segment")
        dr, dc = int(np.sign(er - r)), int(np.sign(ec - c))
        while (r, c) != (er, ec):
            r, c = r + dr, c + dc
            points.append((r, c))
    if len(set(points)) != len(points):
        raise ValueError("route repeats a cell")
    return np.asarray(points, dtype=np.int64)


def _decode(input_grid):
    a = _grid(input_grid)
    if a.shape[0] < 8:
        raise ValueError("missing map or legend")
    # The removable legend always has exactly three rows, in display coordinates.
    scene, separator, carrier_row, stencil_row = a[:-3], a[-3], a[-2], a[-1]
    divider = int(separator[0])
    if divider == 0 or not np.all(separator == divider):
        raise ValueError("legend needs a uniform nonzero separator")
    occupied = np.flatnonzero(carrier_row)
    k = len(occupied)
    if not 2 <= k <= 7 or k >= a.shape[1] or not np.array_equal(occupied, np.arange(k)):
        raise ValueError("carrier must be a nonzero prefix of length 2..7")
    if np.any(stencil_row[k:]):
        raise ValueError("stencil extends past the carrier")
    carrier, stencil = carrier_row[:k].copy(), stencil_row[:k].copy()
    if np.all(stencil != 0):
        raise ValueError("stencil must have at least one transparent cell")
    colors, frequencies = np.unique(scene[scene != 0], return_counts=True)
    if len(colors) != 3 or np.count_nonzero(frequencies == 1) != 1:
        raise ValueError("map must have three roles and one singleton seed role")
    seed = int(colors[frequencies == 1][0])
    others = [(int(n), int(c)) for c, n in zip(colors, frequencies) if c != seed]
    others.sort()
    if others[0][0] < 2 or others[0][0] == others[1][0]:
        raise ValueError("marker and permanent-bar roles are not distinguishable")
    marker, permanent = others[0][1], others[1][1]
    controls = {seed, marker, permanent, divider}
    if len(controls) != 4 or np.any(scene == divider):
        raise ValueError("separator and map controls must be distinct")
    payload = set(map(int, np.unique(np.concatenate((carrier, stencil))))) - {0}
    if controls & payload:
        raise ValueError("strip ink must be distinct from control colors")
    r, c = map(int, np.argwhere(scene == seed)[0])
    h, w = scene.shape
    matches = (r == 0, c == 0, r == h - 1, c == w - 1)
    if sum(matches) != 1:
        raise ValueError("seed must be on exactly one map edge")
    direction = DIRECTIONS[matches.index(True)]
    forward = _to_forward(scene, direction)
    height, width = forward.shape
    initial = int(np.argwhere(forward == seed)[0, 1])
    structure = (forward == permanent) | (forward == marker)
    rows = np.flatnonzero(structure.any(axis=1))
    if not 1 <= len(rows) <= 5:
        raise ValueError("expected 1..5 bars")
    vertices, bars = [(0, initial)], []
    current, previous = initial, -1
    for raw in rows:
        f = int(raw)
        cells = np.flatnonzero(structure[f])
        lo, hi = int(cells[0]), int(cells[-1])
        markers = tuple(map(int, np.flatnonzero(forward[f] == marker)))
        n = len(markers)
        if (not 1 <= n <= 4 or len(cells) != hi - lo + 1
                or hi - lo + 1 < 2 * n + 1):
            raise ValueError("invalid contiguous asymmetric bar")
        low = markers == tuple(lo + 2 * j for j in range(n))
        high = markers == tuple(hi - 2 * j for j in reversed(range(n)))
        if low == high:
            raise ValueError("marker endpoint must be unique")
        sign = 1 if low else -1
        d = n + 1
        target = hi + d if sign == 1 else lo - d
        turn = f - d
        if (not 0 < f < height - 1 or turn < 1 or turn <= previous
                or not lo <= current <= hi or not 0 <= target < width):
            raise ValueError("bar command is not feasible")
        vertices.extend(((turn, current), (turn, target)))
        bars.append((f, lo, hi, n, sign))
        current, previous = target, f
    vertices.append((height - 1, current))
    path = _walk(vertices)
    if np.any(structure[path[:, 0], path[:, 1]]):
        raise ValueError("route intersects a bar")
    if len(path) % k:
        raise ValueError("ribbon length must be divisible by carrier length")
    return dict(scene=scene, forward=forward, direction=direction, seed=seed,
                marker=marker, permanent=permanent, divider=divider,
                initial=initial, bars=bars, vertices=vertices, path=path,
                carrier=carrier, stencil=stencil, k=k, q=len(path) // k)


def _print_sequence(carrier, stencil, length):
    # Integral replication is used by all valid family inputs. The floor form
    # also lets diagnostics render counterfactual paths of a different length.
    k = len(carrier)
    times = np.arange(length)
    base = carrier[times * k // length]
    tile = stencil[times % k]
    return np.where(tile == 0, base, tile)


def _paint(decoded, sequence, path=None):
    path = decoded["path"] if path is None else path
    out = decoded["forward"].copy()
    out[out == decoded["marker"]] = decoded["permanent"]
    out[path[:, 0], path[:, 1]] = sequence
    return _from_forward(out, decoded["direction"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone, including decoding all variable color roles."""
    d = _decode(input_grid)
    return _paint(d, _print_sequence(d["carrier"], d["stencil"], len(d["path"])))


def _span_plan(rng, width, counts):
    """Finite search over lateral states; repeated bypass sides are permitted."""
    failed = set()

    def visit(i, current):
        if i == len(counts):
            return []
        if (i, current) in failed:
            return None
        n = counts[i]
        options = []
        for lo in range(current + 1):
            for hi in range(max(current, lo + 2 * n), width):
                for sign, target in ((1, hi + n + 1), (-1, lo - n - 1)):
                    if 0 <= target < width:
                        options.append((lo, hi, target, sign))
        for ix in rng.permutation(len(options)):
            item = options[int(ix)]
            rest = visit(i + 1, item[2])
            if rest is not None:
                return [item] + rest
        failed.add((i, current))
        return None

    for initial in rng.permutation(np.arange(1, width - 1)):
        result = visit(0, int(initial))
        if result is not None:
            return int(initial), result
    return None


def _carrier(rng, k, payload):
    count = 1 if rng.random() < .12 else int(rng.integers(2, min(k, len(payload)) + 1))
    inks = rng.choice(payload, count, replace=False)
    kind = int(rng.integers(4))
    if count == 1:
        labels = np.zeros(k, dtype=int)
    elif kind == 0:
        labels = (np.arange(k) + int(rng.integers(count))) % count
    elif kind == 1:
        cuts = sorted(rng.choice(np.arange(1, k), count - 1, replace=False))
        labels = rng.permutation(count)[np.searchsorted(cuts, np.arange(k), side="right")]
    elif kind == 2:
        labels = rng.integers(count, size=k)
        labels[rng.choice(k, count, replace=False)] = rng.permutation(count)
    else:
        labels = np.zeros(k, dtype=int)
        locations = rng.choice(k, count - 1, replace=False)
        labels[locations] = np.arange(1, count)
    return inks[labels].astype(np.int64)


def _stencil(rng, k, payload):
    result = np.zeros(k, dtype=np.int64)
    if rng.random() < .10:
        return result
    count = int(rng.integers(1, k))
    positions = rng.choice(k, count, replace=False)
    if rng.random() < .30:
        start = int(rng.integers(k))
        positions = (start + np.arange(count)) % k
    ink_count = 1 if rng.random() < .55 else int(rng.integers(1, min(3, count, len(payload)) + 1))
    inks = rng.choice(payload, ink_count, replace=False)
    values = rng.choice(inks, count)
    values[:ink_count] = inks
    rng.shuffle(values)
    result[positions] = values
    return result


def _sample_pair(rng, controls, payload, directions, maximum):
    seed, permanent, marker, divider = controls
    for _ in range(80):
        count = int(rng.choice(np.arange(1, 6), p=(.16, .26, .25, .20, .13)))
        ns = list(map(int, rng.integers(1, maximum + 1, size=count)))
        if count == 1 and ns[0] == 1:
            ns[0] = 2  # Keep the unique boundary singleton unambiguous.
        while sum(ns) + 2 * count + 2 > 27:
            choices = [i for i, n in enumerate(ns) if n > 1]
            ns[int(rng.choice(choices))] -= 1
        minimum = sum(ns) + 2 * count + 2
        width = int(rng.integers(8, 31))
        plan = _span_plan(rng, width, ns)
        if plan is None:
            width, plan = 30, _span_plan(rng, 30, ns)
        if plan is None:
            continue
        initial, spans = plan
        lateral = 0
        current = initial
        for _, _, target, _ in spans:
            lateral += abs(target - current)
            current = target
        # Route length is H + lateral movement, independently of bar row gaps.
        choices = [(k, h) for k in range(2, min(7, width - 1) + 1)
                   for h in range(minimum, 28) if (h + lateral) % k == 0]
        if not choices:
            continue
        k, height = choices[int(rng.integers(len(choices)))]
        slack = height - minimum
        gaps = rng.multinomial(slack, np.full(count + 1, 1 / (count + 1)))
        if rng.random() < .25:
            gaps[:] = 0
            gaps[int(rng.integers(count + 1))] = slack
        scene = np.zeros((height, width), dtype=np.int64)
        scene[0, initial] = seed
        vertices, current, previous = [(0, initial)], initial, -1
        for i, (n, (lo, hi, target, sign)) in enumerate(zip(ns, spans)):
            turn = (1 if i == 0 else previous + 1) + int(gaps[i])
            f = turn + n + 1
            scene[f, lo:hi + 1] = permanent
            for j in range(n):
                scene[f, lo + 2 * j if sign == 1 else hi - 2 * j] = marker
            vertices.extend(((turn, current), (turn, target)))
            current, previous = target, f
        vertices.append((height - 1, current))
        path = _walk(vertices)
        assert len(path) == height + lateral and len(path) % k == 0
        carrier, stencil = _carrier(rng, k, payload), _stencil(rng, k, payload)
        # Construct labels from the sampled plan, without calling solve/decode.
        base = np.repeat(carrier, len(path) // k)
        tile = np.tile(stencil, len(path) // k)
        inks = np.where(tile == 0, base, tile)
        out = scene.copy()
        out[out == marker] = permanent
        out[path[:, 0], path[:, 1]] = inks
        direction = str(rng.choice(directions))
        physical, output = _from_forward(scene, direction), _from_forward(out, direction)
        legend = np.zeros((3, physical.shape[1]), dtype=np.int64)
        # In horizontal routes the physical width is the forward height.
        if k >= physical.shape[1]:
            continue
        legend[0] = divider
        legend[1, :k], legend[2, :k] = carrier, stencil
        if physical.shape[0] + 3 > 30:
            continue
        return {"input": np.vstack((physical, legend)), "output": output}
    raise RuntimeError("bounded geometric sampler found no valid pair")


def _alternate_path(d, distance):
    vertices, current = [(0, d["initial"])], d["initial"]
    for f, lo, hi, n, sign in d["bars"]:
        delta = distance(n)
        target = hi + delta if sign == 1 else lo - delta
        vertices.extend(((f - delta, current), (f - delta, target)))
        current = target
    vertices.append((d["forward"].shape[0] - 1, current))
    return _walk(vertices)


def _evidence(pair):
    d = _decode(pair["input"])
    out = np.asarray(pair["output"])
    path, k, q = d["path"], d["k"], d["q"]
    t = np.arange(len(path))
    base, tile = d["carrier"][t // q], d["stencil"][t % k]
    actual = np.where(tile == 0, base, tile)
    covered = set()

    def compare(name, sequence):
        if np.any(actual != sequence):
            covered.add(name)

    compare("ignore_stencil", base)
    compare("tile_carrier", np.where(tile == 0, d["carrier"][t % k], tile))
    stretched = d["stencil"][t // q]
    compare("stretch_stencil", np.where(stretched == 0, base, stretched))
    compare("opaque_zero", tile)
    moves = np.diff(path, axis=0)
    bends = np.flatnonzero(np.any(moves[1:] != moves[:-1], axis=1)) + 1
    resets = np.zeros(len(path), dtype=int)
    for bend in bends:
        resets[bend:] = bend
    reset_tile = d["stencil"][(t - resets) % k]
    compare("reset_at_bends", np.where(reset_tile == 0, base, reset_tile))
    for name, phase in (("row_phase", path[:, 0]), ("column_phase", path[:, 1]),
                        ("plane_phase", path.sum(axis=1))):
        ink = d["stencil"][phase % k]
        compare(name, np.where(ink == 0, base, ink))
    if np.any(tile == 0) and len(np.unique(base[tile == 0])) >= 2:
        covered.add("visible_carrier")
    if np.any((tile != 0) & (tile != base)):
        covered.add("visible_overwrite")
    for name, distance in (("fixed_clearance", lambda n: 2),
                           ("missing_plus_one", lambda n: n)):
        alternate = _alternate_path(d, distance)
        hypothetical = _paint(d, _print_sequence(d["carrier"], d["stencil"], len(alternate)), alternate)
        if not np.array_equal(out, hypothetical):
            covered.add(name)
    straight = np.array([(r, d["initial"]) for r in range(d["forward"].shape[0])])
    if not np.array_equal(out, _paint(d, _print_sequence(d["carrier"], d["stencil"], len(straight)), straight)):
        covered.add("straight")
    current, width = d["initial"], d["forward"].shape[1]
    for _, lo, hi, n, sign in d["bars"]:
        left, right = lo - n - 1, hi + n + 1
        actual_target, other = (right, left) if sign == 1 else (left, right)
        if 0 <= other < width and abs(other - current) < abs(actual_target - current):
            covered.add("closest_endpoint")
        current = actual_target
    return covered


def _teaching_ok(pairs, maximum):
    ds = [_decode(p["input"]) for p in pairs]
    evidence = set().union(*(_evidence(p) for p in pairs))
    ns = {bar[3] for d in ds for bar in d["bars"]}
    signs = {bar[4] for d in ds for bar in d["bars"]}
    repeated = any(any(x[4] == y[4] for x, y in zip(d["bars"], d["bars"][1:])) for d in ds)
    # A same-count/different-length witness rejects using bar length as distance.
    lengths = {}
    for d in ds:
        for _, lo, hi, n, _ in d["bars"]:
            lengths.setdefault(n, set()).add(hi - lo + 1)
    return (REQUIRED_EVIDENCE <= evidence and set(range(1, maximum + 1)) <= ns
            and signs == {-1, 1} and repeated and any(len(v) > 1 for v in lengths.values())
            and len({d["direction"] for d in ds}) >= 2
            and len({d["k"] for d in ds}) >= 2 and len({d["q"] for d in ds}) >= 2)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(np.arange(1, 10))
    controls = tuple(map(int, permutation[:4]))
    payload = permutation[4:4 + int(rng.integers(3, 6))]
    maximum = int(rng.integers(2, 5))
    dirs = list(rng.permutation(DIRECTIONS))
    train_dirs, heldout = dirs[:3], dirs[3]
    candidates, candidate_keys, covers = [], set(), []
    training = None
    # Two informative examples suffice even at the smallest legal count. The
    # selected pair is evidence-driven, rather than a fixed opening picture.
    for _ in range(120):
        p = _sample_pair(rng, controls, payload, train_dirs, maximum)
        if _key(p["input"]) in candidate_keys:
            continue
        cover = _evidence(p)
        for j in rng.permutation(len(candidates)):
            j = int(j)
            if REQUIRED_EVIDENCE <= covers[j] | cover and _teaching_ok([candidates[j], p], maximum):
                training = [candidates[j], p]
                break
        candidates.append(p)
        candidate_keys.add(_key(p["input"]))
        covers.append(cover)
        if training is not None:
            break
    if training is None:
        training = _fallback_training(controls, payload, train_dirs)
    puzzle = {"train": training, "test": []}
    seen = {_key(p["input"]) for p in training}
    for split, total in (("train", num_train), ("test", num_test)):
        while len(puzzle[split]) < total:
            directions = train_dirs if split == "train" else [heldout]
            for _ in range(80):
                pair = _sample_pair(rng, controls, payload, directions, maximum)
                first_test_ok = (split != "test" or puzzle["test"]
                                 or INTERACTIVE_EVIDENCE <= _evidence(pair))
                if _key(pair["input"]) not in seen and first_test_ok:
                    break
            else:
                raise RuntimeError("bounded sampler did not find a distinct input")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _fallback_training(controls, payload, directions):
    """Two finite constructive teaching witnesses, only if pool selection fails."""
    seed, bar, marker, divider = controls
    a, b, c = map(int, payload[:3])
    cases = (
        ((12, 14), 4, [(4, 3, 7, 1, 1), (9, 5, 10, 2, -1)],
         [a, b, c], [0, b, 0]),
        ((21, 27), 4, [(3, 1, 6, 1, 1), (7, 5, 11, 2, 1),
                      (12, 9, 19, 3, -1), (18, 0, 8, 4, 1)],
         [a, b, a, c], [c, 0, b, 0]),
    )
    pairs = []
    for direction, (shape, initial, bars, carrier, stencil) in zip(directions, cases):
        scene = np.zeros(shape, dtype=np.int64)
        scene[0, initial] = seed
        for f, lo, hi, n, sign in bars:
            scene[f, lo:hi + 1] = bar
            for j in range(n):
                scene[f, lo + 2 * j if sign == 1 else hi - 2 * j] = marker
        scene = _from_forward(scene, direction)
        legend = np.zeros((3, scene.shape[1]), dtype=np.int64)
        legend[0] = divider
        legend[1, :len(carrier)], legend[2, :len(stencil)] = carrier, stencil
        inp = np.vstack((scene, legend))
        pairs.append({"input": inp, "output": solve(inp)})
    assert _teaching_ok(pairs, 4)
    return pairs


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, controls, decoded = set(), set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            d = _decode(a)
            assert b.shape == d["scene"].shape
            assert np.array_equal(solve(a.copy()), b), "wrong ribbon output"
            # Family-specific properties, checked from actual grids.
            f = _to_forward(b, d["direction"])
            mask = np.zeros(d["forward"].shape, dtype=bool)
            mask[d["path"][:, 0], d["path"][:, 1]] = True
            bars = (d["forward"] == d["permanent"]) | (d["forward"] == d["marker"])
            assert not np.any(mask & bars)
            assert np.all(f[bars] == d["permanent"])
            assert np.all(f[~mask & ~bars] == 0)
            assert np.all(f[mask] != 0)
            assert len(d["path"]) == d["forward"].shape[0] + sum(
                abs(d["vertices"][i + 1][1] - d["vertices"][i][1])
                for i in range(len(d["vertices"]) - 1))
            assert d["marker"] not in set(map(int, np.unique(b)))
            controls.add((d["seed"], d["permanent"], d["marker"], d["divider"]))
            decoded[split].append(d)
    assert len(controls) == 1, "control conventions must agree within the puzzle"
    train_ns = {b[3] for d in decoded["train"] for b in d["bars"]}
    test_ns = {b[3] for d in decoded["test"] for b in d["bars"]}
    assert test_ns <= train_ns
    assert _teaching_ok(puzzle["train"], max(train_ns)), "insufficient demonstrations"
    assert any(INTERACTIVE_EVIDENCE <= _evidence(p) for p in puzzle["test"]), "test lacks an interaction witness"
    train_dirs = {d["direction"] for d in decoded["train"]}
    test_dirs = {d["direction"] for d in decoded["test"]}
    assert not train_dirs & test_dirs, "expected the held-out-heading shift"


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s])
                 for s in ("train", "test"))


def _literal_regression():
    # A literal two-bar map: marker counts 1 and 2; bends (2,4)->(2,9),
    # then (6,9)->(6,2). Length=24, so carrier [2,3,4] has eight-cell runs.
    scene = np.zeros((12, 14), dtype=np.int64)
    scene[0, 4] = 1
    scene[4, 3:8] = [9, 5, 5, 5, 5]
    scene[9, 5:11] = [5, 5, 5, 9, 5, 9]
    legend = np.zeros((3, 14), dtype=np.int64)
    legend[0] = 8
    legend[1, :3], legend[2, :3] = [2, 3, 4], [0, 6, 0]
    path = ([(0, 4), (1, 4), (2, 4)] + [(2, c) for c in range(5, 10)]
            + [(r, 9) for r in range(3, 7)] + [(6, c) for c in range(8, 1, -1)]
            + [(r, 2) for r in range(7, 12)])
    assert len(path) == 24
    values = [2, 6, 2, 2, 6, 2, 2, 6, 3, 3, 6, 3,
              3, 6, 3, 3, 6, 4, 4, 6, 4, 4, 6, 4]
    expected = scene.copy()
    expected[expected == 9] = 5
    for point, ink in zip(path, values):
        expected[point] = ink
    # The legend stays in screen coordinates under rotation/reflection of map.
    for direction in DIRECTIONS:
        physical = _from_forward(scene, direction)
        header = np.zeros((3, physical.shape[1]), dtype=np.int64)
        header[0] = 8
        header[1, :3], header[2, :3] = [2, 3, 4], [0, 6, 0]
        assert np.array_equal(solve(np.vstack((physical, header))), _from_forward(expected, direction))
    # Normalizing controls is independent of payload colors and preserves zero.
    recolor = np.array([0, 7, 8, 1, 2, 6, 4, 5, 9, 3])
    assert np.array_equal(solve(recolor[np.vstack((scene, legend))]), recolor[expected])
    # Enumerate all transparent/opaque masks on each valid strip length for this
    # literal route. Ink sequences need not match any generation motif.
    inks = [2, 6, 3, 7, 4, 2]
    for k in (2, 3, 4, 6):
        for bits in range(2**k - 1):
            carrier = inks[:k]
            stencil = [inks[(i + 2) % len(inks)] if bits & (1 << i) else 0 for i in range(k)]
            header = np.zeros((3, 14), dtype=np.int64)
            header[0] = 8
            header[1, :k], header[2, :k] = carrier, stencil
            target = scene.copy()
            target[target == 9] = 5
            for t, point in enumerate(path):
                target[point] = stencil[t % k] or carrier[t // (24 // k)]
            assert np.array_equal(solve(np.vstack((scene, header))), target)
    invalid = legend.copy()
    invalid[1, :5] = [2, 3, 4, 6, 7]  # Five does not divide 24.
    try:
        solve(np.vstack((scene, invalid)))
    except ValueError:
        pass
    else:
        raise AssertionError("nonintegral ribbon replication accepted")
    # Moving only the later bar shortens the route from 24 to 21. The same
    # length-three carrier then has runs of seven rather than eight cells, so
    # an earlier corner changes ink before the modified obstacle is reached.
    coupled_legend = legend.copy()
    coupled_legend[2, :3] = [0, 0, 6]
    moved = scene.copy()
    moved[9] = 0
    moved[9, 8:14] = [5, 5, 5, 9, 5, 9]
    old = solve(np.vstack((scene, coupled_legend)))
    new = solve(np.vstack((moved, coupled_legend)))
    assert old[2, 9] == 2 and new[2, 9] == 3


def self_test() -> dict:
    _literal_regression()
    for first in DIRECTIONS:
        for second in DIRECTIONS:
            if first != second:
                assert _teaching_ok(_fallback_training((1, 5, 9, 8), [2, 3, 4], [first, second]), 4)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[7]) == _puzzle_key(generate_puzzle(7))
    assert _puzzle_key(generate_puzzle(52, 7, 3)) == _puzzle_key(generate_puzzle(52, 7, 3))
    structures, shapes, directions, counts, strips, scales, roles = set(), set(), set(), set(), set(), set(), set()
    masks, carriers, transitions, combinations = set(), set(), set(), set()
    first_tests = set()
    for p in puzzles:
        validate_generated_puzzle(p)
        first_tests.add(_normalized(p["test"][0]["output"]))
        for split in p:
            for pair in p[split]:
                d = _decode(pair["input"])
                structures.add(_normalized(pair["input"]))
                shapes.add(tuple(pair["output"].shape))
                directions.add((d["direction"], split))
                counts.add(len(d["bars"]))
                strips.add(d["k"])
                scales.add(d["q"])
                roles.add((d["seed"], d["permanent"], d["marker"], d["divider"]))
                masks.add(tuple(d["stencil"] != 0))
                carriers.add(_normalized(d["carrier"]))
                signs = [b[4] for b in d["bars"]]
                transitions.update(zip(signs, signs[1:]))
                combinations.add((d["direction"], len(d["bars"]), d["k"], d["q"], split))
    assert len(structures) >= 190 and len(shapes) >= 60 and len(first_tests) == 42
    assert directions == {(d, s) for d in DIRECTIONS for s in ("train", "test")}
    assert counts == {1, 2, 3, 4, 5} and strips == set(range(2, 8))
    assert len(scales) >= 12 and len(roles) >= 38
    assert len(masks) >= 35 and len(carriers) >= 35 and len(combinations) >= 150
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(10000 + 10 * train + test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in x.items()} for x in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=210, normalized_inputs=len(structures),
                output_shapes=len(shapes), first_test_structures=len(first_tests),
                carrier_structures=len(carriers), stencil_masks=len(masks),
                replication_factors=len(scales), control_palettes=len(roles),
                joint_combinations=len(combinations), independent_solver=False)


if __name__ == "__main__":
    metrics = self_test()
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(metrics, separators=(",", ":")) + "\n")
    print(json.dumps(metrics, separators=(",", ":")))
