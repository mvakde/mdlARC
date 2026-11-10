"""Switch-routed cargo: stable compaction on newly connected track paths.

Only visible pixels determine structural roles, routes, and cargo order.
No parent generator is imported or executed by this module.
"""
from itertools import permutations
import numpy as np


_DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))
HOST_SEEDS = (0, 926636861, 661976806, 919339308, 337885903, 796069084,
              611760678, 697165169, 964774465, 26278361, 552547737,
              418228152, 547833197, 739706404, 768163169, 54888233,
              546771718, 922182554, 331066219, 771754628, 889873591,
              380671220, 281695704, 760906637, 266539452, 495030513,
              933319986, 267000485, 161694040, 589602236, 269390341,
              756237740, 716469336, 88011523, 982717862, 429692084,
              43771835, 558382922, 163142547, 19063352, 992199897, 578996877)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with axes 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _trace(mask):
    """Recover all actual four-neighbor chains, including empty-cargo chains."""
    cells = {tuple(map(int, q)) for q in np.argwhere(mask)}
    adj = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
               if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    if any(len(v) not in (1, 2) for v in adj.values()):
        raise ValueError("Tracks must have degree one or two")
    remaining, paths = set(cells), []
    while remaining:
        component = {min(remaining)}
        queue = list(component)
        for p in queue:
            for q in adj[p]:
                if q not in component:
                    component.add(q)
                    queue.append(q)
        ends = sorted(p for p in component if len(adj[p]) == 1)
        if len(ends) != 2:
            raise ValueError("Closed tracks are not permitted")
        path = [ends[0]]
        while path[-1] != ends[1]:
            next_cells = [q for q in adj[path[-1]]
                          if len(path) == 1 or q != path[-2]]
            if len(next_cells) != 1:
                raise ValueError("Ambiguous route")
            path.append(next_cells[0])
        if len(path) != len(component):
            raise ValueError("Incomplete route")
        # In canonical coordinates, rows index parallel lanes. This family
        # excludes physical gravity traps, but allows either horizontal turn.
        row_steps = {b[0] - a[0] for a, b in zip(path, path[1:]) if a[0] != b[0]}
        if len(row_steps) > 1:
            raise ValueError("A route reverses across the lane axis")
        paths.append(path)
        remaining.difference_update(component)
    return sorted(paths, key=lambda p: (p[0], p[-1]))


def _structure(core, bg, track, divider):
    """Interpret horizontal lanes; all switch writes happen simultaneously."""
    h, w = core.shape
    if h < 3 or h % 2 != 1 or w < 7:
        raise ValueError("Invalid lane canvas")
    if np.any(core[::2] == divider) or np.any(core[::2, [0, w - 1]] == bg):
        raise ValueError("Invalid lane endpoints or lane color")
    centers, guard, writes = [], set(), {}
    for r in range(1, h, 2):
        if np.any(~np.isin(core[r], [bg, divider])):
            raise ValueError("Cargo cannot be in separator rows")
        xs = np.flatnonzero(core[r] == divider).tolist()
        starts = [c for c in xs if c - 1 not in xs]
        for first in starts:
            c = first + 1
            if (first < 1 or first + 3 >= w
                    or xs[xs.index(first):xs.index(first) + 3] != [first, first + 1, first + 2]
                    or first + 3 in xs):
                raise ValueError("Switches must be separate three-cell bars")
            if (not np.all(core[r - 1, c - 1:c + 2] == track)
                    or not np.all(core[r + 1, c - 1:c + 2] == track)):
                raise ValueError("Switch neighborhoods must be empty track")
            centers.append((r, c))
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    p, value = (rr, cc), divider if cc == c else track
                    if p in writes and writes[p] != value:
                        raise ValueError("Conflicting simultaneous switch writes")
                    guard.add(p)
                    writes[p] = value
    if not centers:
        raise ValueError("At least one switch is required")
    breaks = []
    for r in range(0, h, 2):
        xs = np.flatnonzero(core[r] == bg).tolist()
        if (any(c < 2 or c > w - 3 for c in xs)
                or any(b - a < 3 for a, b in zip(xs, xs[1:]))):
            raise ValueError("Breaks must leave usable track segments")
        breaks.extend((r, c) for c in xs)
    base = np.full(core.shape, bg, dtype=np.int8)
    base[::2] = np.where(core[::2] == bg, bg, track)
    for p, value in writes.items():
        base[p] = value
    paths = _trace(base == track)
    for path in paths:
        for r, c in (path[0], path[-1]):
            if (r % 2 != 0 or (c not in (0, w - 1)
                    and core[r, c - 1] != bg and core[r, c + 1] != bg)):
                raise ValueError("Unexposed terminal")
    return {"base": base, "paths": paths, "centers": centers,
            "guard": guard, "breaks": breaks, "roles": (bg, track, divider)}


def _decode(value):
    """Infer background, track, switch, margins, and lane axis from geometry.

    In particular, background need not be the most frequent input color.
    The six empty track cells around each bar identify the track color.
    """
    a = _grid(value)
    candidates = []
    colors = sorted(map(int, np.unique(a)))
    for axis in (0, 1):
        view = a if axis == 0 else a.T
        for bg in colors:
            occupied = np.argwhere(view != bg)
            if not len(occupied):
                continue
            r0, c0 = map(int, occupied.min(axis=0))
            r1, c1 = map(int, occupied.max(axis=0) + 1)
            core = view[r0:r1, c0:c1]
            if core.shape[0] < 3 or core.shape[0] % 2 != 1 or core.shape[1] < 7:
                continue
            bars = set(map(int, np.unique(core[1::2]))) - {bg}
            if len(bars) != 1:
                continue
            divider = bars.pop()
            t_values = set()
            for r in range(1, core.shape[0], 2):
                xs = np.flatnonzero(core[r] == divider).tolist()
                for c in xs:
                    if c - 1 not in xs and c + 2 < core.shape[1]:
                        t_values.update(map(int, core[r - 1, c:c + 3]))
                        t_values.update(map(int, core[r + 1, c:c + 3]))
            if len(t_values) != 1:
                continue
            track = t_values.pop()
            if track in (bg, divider):
                continue
            try:
                info = _structure(core, bg, track, divider)
            except ValueError:
                continue
            info.update(core=core, axis=axis, box=(r0, r1, c0, c1), shape=a.shape)
            candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("Input must have exactly one geometric role interpretation")
    return candidates[0]


def _cargo(core, path, roles):
    return [(p, int(core[p])) for p in path if int(core[p]) not in roles]


def _packed(core, info, ordering="path", settle=True):
    result = info["base"].copy()
    for path in info["paths"]:
        cargo = _cargo(core, path, info["roles"])
        if path[0][0] == path[-1][0] or not settle:
            for p, color in cargo:
                result[p] = color
            continue
        sequence = [c for _, c in cargo]
        if ordering == "reverse":
            sequence.reverse()
        elif ordering == "scan":
            sequence = [c for _, c in sorted(cargo)]
        elif ordering == "sort":
            sequence.sort()
        elif ordering != "path":
            raise ValueError("Unknown diagnostic ordering")
        if sequence:
            for p, color in zip(path[-len(sequence):], sequence):
                result[p] = color
    return result


def _restore(core_output, info):
    bg = info["roles"][0]
    view_shape = info["shape"] if info["axis"] == 0 else info["shape"][::-1]
    view = np.full(view_shape, bg, dtype=np.int8)
    r0, r1, c0, c1 = info["box"]
    view[r0:r1, c0:c1] = core_output
    return view if info["axis"] == 0 else view.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Rotate bars; stably pack cargo toward increasing lane index."""
    info = _decode(input_grid)
    return _restore(_packed(info["core"], info), info)


def _features(info):
    core, roles = info["core"], info["roles"]
    out = _packed(core, info)
    result = {"ordered": False, "deep": False, "level_gaps": False,
              "recurrence": False, "interior": False, "multi_lane": False}
    loads = []
    for path in info["paths"]:
        cargo = _cargo(core, path, roles)
        seq = [c for _, c in cargo]
        loads.append(len(seq))
        result["interior"] |= any(p not in (path[0], path[-1]) for p, _ in cargo)
        result["multi_lane"] |= len({p[0] for p, _ in cargo}) >= 2
        runs = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
        result["recurrence"] |= len(runs) > len(set(runs))
        if path[0][0] == path[-1][0]:
            positions = [i for i, p in enumerate(path) if int(core[p]) not in roles]
            result["level_gaps"] |= (len(positions) >= 2
                                     and positions[-1] - positions[0] + 1 > len(positions))
        elif seq:
            tail = path[-len(seq):]
            result["deep"] |= len({p[0] for p in tail}) >= 2
            scan = [c for _, c in sorted(cargo)]
            result["ordered"] |= (seq != seq[::-1] and seq != scan
                                  and len(runs) > len(set(runs))
                                  and len({p[0] for p, _ in cargo}) >= 2)
    result.update(loads=loads, moved=not np.array_equal(out, _packed(core, info, settle=False)),
                  reverse_diff=not np.array_equal(out, _packed(core, info, "reverse")),
                  scan_diff=not np.array_equal(out, _packed(core, info, "scan")),
                  sort_diff=not np.array_equal(out, _packed(core, info, "sort")))
    return result


def _empty_layout(lanes, width, centers, breaks, roles):
    bg, track, divider = roles
    core = np.full((2 * lanes - 1, width), bg, dtype=np.int8)
    core[::2] = track
    for p in breaks:
        core[p] = bg
    for r, c in centers:
        core[r, c - 1:c + 2] = divider
    return core


def _sample_layout(rng, lanes, width, busy, roles):
    centers = []
    for r in range(1, 2 * lanes - 1, 2):
        wanted = int(rng.integers(0, min(6, (width - 1) // 4) + 1)) if busy else int(rng.choice([0, 1, 1]))
        row = []
        for c in map(int, rng.permutation(np.arange(2, width - 2))):
            if len(row) == wanted:
                break
            if (all(abs(c - q) >= 4 for q in row)
                    and all(abs(c - q) >= 2 for rr, q in centers if rr == r - 2)):
                row.append(c)
        centers.extend((r, c) for c in sorted(row))
    if not centers:
        centers = [(2 * int(rng.integers(lanes - 1)) + 1, int(rng.integers(2, width - 2)))]
    breaks = []
    for r in range(0, 2 * lanes - 1, 2):
        ports = sorted((c, rr) for rr, c in centers if abs(r - rr) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                breaks.append((r, int(rng.integers(a + 2, b - 1))))
    for _ in range(int(rng.integers(0, min(8, lanes) + 1))):
        available = [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(2, width - 2)
                     if all(abs(c - q) >= 2 for rr, q in centers if abs(rr - r) == 1)
                     and all(abs(c - q) >= 3 for rr, q in breaks if rr == r)]
        if available:
            breaks.append(available[int(rng.integers(len(available)))])
    core = _empty_layout(lanes, width, centers, breaks, roles)
    return core, _structure(core, *roles)


def _witness_paths(info):
    result = []
    for path in info["paths"]:
        available = [i for i, p in enumerate(path) if p not in info["guard"]]
        horizontal = {b[1] - a[1] for a, b in zip(path, path[1:]) if a[1] != b[1]}
        low_run = sum(p[0] == path[-1][0] for p in path)
        if (path[0][0] != path[-1][0] and len(available) >= max(4, low_run + 1)
                and horizontal == {-1, 1}):
            result.append((path, available, low_run))
    return result


def _load(rng, core, info, palette, probability, special, forced_colors=None):
    core = core.copy()
    # Payload colors and run boundaries are chosen independently of bends.
    for path in info["paths"]:
        previous = int(rng.choice(palette))
        for p in path:
            if p in info["guard"]:
                continue
            if rng.random() < probability:
                if rng.random() < .6:
                    previous = int(rng.choice(palette))
                core[p] = previous
    if special in ("order", "level", "fallback"):
        options = _witness_paths(info)
        if not options:
            return None
        choice = 0 if special == "fallback" else int(rng.integers(len(options)))
        path, available, low_run = options[choice]
        for p in path:
            if p not in info["guard"]:
                core[p] = info["roles"][1]
        count = max(4, low_run + 1)
        # A repeated color interrupted by another color, plus a distinct tail,
        # gives visible order and recurrence evidence even with two demos.
        a, b, c = (list(map(int, forced_colors)) if forced_colors is not None
                   else list(map(int, rng.choice(palette, 3, replace=False))))
        seq = [a, b, a] + [c] * (count - 3)
        for _ in range(32):
            indices = sorted(map(int, rng.choice(available, count, replace=False)))
            if len({path[i][0] for i in indices}) < 2:
                continue
            trial = core.copy()
            for i, color in zip(indices, seq):
                trial[path[i]] = color
            check = dict(info, core=trial)
            f = _features(check)
            if f["ordered"] and f["deep"] and f["sort_diff"] and f["moved"]:
                core = trial
                break
        else:
            return None
    if special in ("level", "fallback"):
        levels = [p for p in info["paths"] if p[0][0] == p[-1][0]
                  and len(p) >= 3 and all(q not in info["guard"] for q in p)]
        if not levels:
            return None
        path = levels[int(rng.integers(len(levels)))]
        for p in path:
            core[p] = info["roles"][1]
        core[path[0]], core[path[-1]] = int(palette[0]), int(palette[1])
    if not any(_cargo(core, p, info["roles"]) for p in info["paths"]):
        p = next(p for path in info["paths"] for p in path if p not in info["guard"])
        core[p] = int(palette[0])
    return core


def _safe_layout(mode, target, roles):
    lanes, width = (7, 11) if mode == 0 and target else (3, 11)
    if mode == 3 and not target:
        lanes, width = 6, 14
    centers, cuts = [(1, 3), (3, 5)], [(0, 7)]
    if mode == 2 and target:
        lanes, width = 4, 15
        centers, cuts = [(1, 3), (1, 9), (3, 5)], [(0, 6)]
    core = _empty_layout(lanes, width, centers, cuts, roles)
    return core, _structure(core, *roles)


def _pair(rng, roles, axis, mode, target, special):
    palette = [c for c in range(10) if c not in roles]
    rng.shuffle(palette)
    palette = palette[:int(rng.integers(3, 8))] if special else palette[:int(rng.integers(1, 8))]
    for _ in range(70):
        lo, hi, wlo, whi = 2, 12, 9, 30
        if mode == 0:
            lo, hi = (7, 15) if target else (2, 6)
        elif mode == 3:
            lo, hi, wlo, whi = (3, 5, 9, 18) if target else (6, 15, 14, 30)
        lanes, width = int(rng.integers(lo, hi + 1)), int(rng.integers(wlo, whi + 1))
        busy = (target if mode == 2 else bool(rng.random() < .5))
        try:
            core, info = _sample_layout(rng, lanes, width, busy, roles)
        except ValueError:
            continue
        probability = float(rng.uniform(.55, .85) if target and mode == 1 else rng.uniform(.08, .40))
        loaded = _load(rng, core, info, palette, probability, special)
        if loaded is not None:
            if mode == 2 and target:
                row_counts = [sum(rr == r for rr, _ in info["centers"])
                              for r in range(1, core.shape[0], 2)]
                if max(row_counts) < 2:
                    continue
            return _present(rng, loaded, info, axis)
    # The fallback is a valid composition witness, not a parent example.
    core, info = _safe_layout(mode, target, roles)
    if len(palette) < 3:
        palette = [c for c in range(10) if c not in roles]
    loaded = _load(rng, core, info, palette, .8 if mode == 1 and target else .2, "fallback")
    if loaded is None:
        raise RuntimeError("Constructive witness failed")
    return _present(rng, loaded, info, axis)


def _present(rng, core, info, axis):
    out = _packed(core, info)
    if rng.random() < .5:
        reflected = np.fliplr(core)
        before = _features(dict(info, core=core))
        after = _features(_decode(reflected))
        # Reflection can accidentally make row scanning coincide with a
        # witness's color order. Preserve demonstrated counterexamples.
        if not before["ordered"] or after["ordered"]:
            core, out = reflected, np.fliplr(out)
    margins = []
    for n in core.shape:
        total = int(rng.integers(min(4, 30 - n) + 1))
        before = int(rng.integers(total + 1))
        margins.append((before, total - before))
    bg = info["roles"][0]
    core, out = np.pad(core, margins, constant_values=bg), np.pad(out, margins, constant_values=bg)
    if axis:
        core, out = core.T, out.T
    # Geometric role uniqueness is checked after presentation too.
    _decode(core)
    return {"input": core.copy(), "output": out.copy()}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    fixed_roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    variable_roles, axis, mode = bool(rng.random() < .65), int(rng.integers(2)), int(rng.integers(4))
    demonstration = list(map(int, rng.permutation(num_train)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            special = "order" if split == "test" or i == demonstration[0] else ""
            if split == "train" and i == demonstration[1]:
                special = "level"
            for _ in range(40):
                roles = tuple(map(int, rng.choice(10, 3, replace=False))) if variable_roles else fixed_roles
                try:
                    pair = _pair(rng, roles, axis, mode, split == "test", special)
                except ValueError:
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                # At most 13 earlier inputs exist; 210 distinct cargo triples
                # on one fixed layout guarantee an unused bounded alternative.
                roles = fixed_roles
                palette = [c for c in range(10) if c not in roles]
                for triple in permutations(palette, 3):
                    core, info = _safe_layout(mode, split == "test", roles)
                    probability = 1 if mode == 1 and split == "test" else 0
                    loaded = _load(rng, core, info, palette, probability, "fallback", triple)
                    if loaded is None:
                        continue
                    pair = {"input": loaded.T.copy() if axis else loaded,
                            "output": _packed(loaded, info).T.copy() if axis else _packed(loaded, info)}
                    if _key(pair["input"]) not in seen:
                        break
                else:
                    raise RuntimeError("No distinct bounded fallback")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible graph, per-component conservation, stable suffix and teaching."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features, axes = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            axes.add(info["axis"])
            r0, r1, c0, c1 = info["box"]
            out_view = out if info["axis"] == 0 else out.T
            observed = out_view[r0:r1, c0:c1]
            roles = info["roles"]
            for path in info["paths"]:
                old = _cargo(info["core"], path, roles)
                new = _cargo(observed, path, roles)
                assert [c for _, c in old] == [c for _, c in new], "Cargo order/count changed"
                if path[0][0] == path[-1][0]:
                    assert old == new, "Level cargo moved"
                elif old:
                    assert [p for p, _ in new] == path[-len(old):], "Cargo is not a packed suffix"
                for p in path:
                    if p not in {q for q, _ in new}:
                        assert int(observed[p]) == roles[1], "An empty slot is not track"
            expected = _restore(_packed(info["core"], info), info)
            assert np.array_equal(out, expected), "Switch writes or margins changed"
            features[split].append(_features(info))
    assert len(axes) == 1, "Generated puzzles teach one lane orientation"
    for split in ("train", "test"):
        assert any(f["ordered"] and f["deep"] and f["moved"] and f["sort_diff"]
                   for f in features[split]), "Routing and sequence evidence is missing"
    assert any(f["level_gaps"] for f in features["train"]), "Level retention must be taught"
    assert any(f["interior"] for f in features["train"])


def self_test():
    """Bounded 42-seed variation audit, count boundaries, mutations and fixtures."""
    puzzles, shapes, layouts, roles, loads, bars, breaks, axes = [], set(), set(), set(), set(), set(), set(), set()
    records = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        puzzles.append(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"].copy()
                info = _decode(inp)
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, pair["input"]), "solve mutated its input"
                core = info["core"]
                shapes.add(inp.shape)
                # Binary geometry excludes cargo colors and margins.
                layouts.add(_key(np.where(core == info["roles"][0], 0,
                                          np.where(core == info["roles"][2], 2, 1))))
                roles.add(info["roles"])
                bars.add(len(info["centers"]))
                breaks.add(len(info["breaks"]))
                axes.add(info["axis"])
                f = _features(info)
                loads.update(f["loads"])
                records.append((seed, split, len(info["centers"]), core.shape, f))
    assert len(set(puzzles)) == 42
    assert len(layouts) >= 100 and len(shapes) >= 60 and len(roles) >= 60
    assert len(bars) >= 10 and len(breaks) >= 7 and len(loads) >= 15
    assert axes == {0, 1}
    for role in range(3):
        assert {r[role] for r in roles} == set(range(10))
    assert any(shape[0] == 29 for _, _, _, shape, _ in records)
    assert max(loads) > 30, "The retained-canvas rule should permit long queues"
    # Observe size and routing transfer from actual arrays, not mode names.
    larger = smaller = routing = 0
    for seed in HOST_SEEDS:
        train = [r for r in records if r[0] == seed and r[1] == "train"]
        test = [r for r in records if r[0] == seed and r[1] == "test"]
        larger += min(r[3][0] for r in test) > max(r[3][0] for r in train)
        smaller += max(r[3][0] for r in test) < min(r[3][0] for r in train)
        # The complete audit measures separator counts; this quick check also
        # detects more total switches than any single-switch training row can hold.
        routing += any(r[2] > (r[3][0] - 1) // 2 for r in test)
    assert larger >= 5 and smaller >= 2 and routing >= 5
    # Host also exercises all legal counts; these include the hardest boundaries.
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(12345, a, b)
        q = generate_puzzle(12345, a, b)
        assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # Explicit small fixture: compatible corner overlap and a gapped level path.
    fixture_roles = (0, 8, 5)
    core = _empty_layout(3, 11, [(1, 3), (3, 5)], [(0, 7)], fixture_roles)
    info = _structure(core, *fixture_roles)
    core = _load(np.random.default_rng(101), core, info, [1, 2, 3], 0, "fallback")
    assert core is not None
    expected = _packed(core, info)
    assert np.array_equal(solve(core), expected)
    assert np.array_equal(solve(core.T), expected.T)
    f = _features(dict(info, core=core))
    assert all(f[k] for k in ("ordered", "deep", "level_gaps", "recurrence", "moved", "scan_diff", "reverse_diff", "sort_diff"))
    # Literal, hand-worked answer: source order 1,2,1,3 packs through (1,2),
    # and the level queue at (0,8)..(0,10) retains its empty middle slot.
    def digits(rows):
        return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    hand_input = digits(("12888880182", "00555000000", "31888888888",
                         "00005550000", "88888888888"))
    hand_output = digits(("88858880182", "00158000000", "31258588888",
                          "00008580000", "88888588888"))
    assert np.array_equal(solve(hand_input), hand_output)
    assert np.array_equal(solve(hand_input.T), hand_output.T)
    for mode in range(4):
        for target in (False, True):
            safe, structure = _safe_layout(mode, target, fixture_roles)
            loaded = _load(np.random.default_rng(12), safe, structure, [1, 2, 3], 0, "fallback")
            assert loaded is not None
            for axis in (0, 1):
                pair = _present(np.random.default_rng(34), loaded, structure, axis)
                assert np.array_equal(solve(pair["input"]), pair["output"])
    for centers, cut in (([(1, 3), (3, 3)], []), ([(1, 3), (3, 4)], []),
                         ([(1, 3), (1, 7)], [])):
        bad = _empty_layout(3 if max(r for r, _ in centers) == 3 else 2, 11, centers, cut, fixture_roles)
        try:
            _structure(bad, *fixture_roles)
        except ValueError:
            continue
        raise AssertionError("Malformed topology accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted label accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": len(records), "shapes": len(shapes),
            "binary_layouts": len(layouts), "role_palettes": len(roles),
            "divider_counts": sorted(bars), "break_counts": sorted(breaks),
            "per_path_loads": sorted(loads), "lane_axes": sorted(axes)}
