"""First-fault circuits: rotated switches route probes to negative defects.

Only standard-library Python and NumPy are used. No files, seeds, examples or
generation metadata are consulted by solve(). See task_spec.md for the rule.
"""
from itertools import permutations

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_TAUGHT_EFFECTS = ("overlapping_halos", "adjacent_holes", "halo_on_divider")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _rewire(height, width, centers, track, switch):
    board = np.zeros((height, width), dtype=np.int8)
    board[::2] = track
    writes = {}
    for r, c in centers:
        for y in range(r - 1, r + 2):
            for x in range(c - 1, c + 2):
                color = switch if x == c else track
                if (y, x) in writes and writes[y, x] != color:
                    raise ValueError("conflicting switch neighborhoods")
                writes[y, x] = color
    for cell, color in writes.items():
        board[cell] = color
    return board


def _graph(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    graph = {p: tuple((p[0] + dy, p[1] + dx) for dy, dx in _DIRS
                      if (p[0] + dy, p[1] + dx) in cells) for p in cells}
    if any(len(v) not in (1, 2) for v in graph.values()):
        raise ValueError("wires must form paths or closed loops")
    remaining = set(graph)
    paths, loops = [], []
    while remaining:
        start = min(remaining)
        component, stack = set(), [start]
        while stack:
            p = stack.pop()
            if p in component:
                continue
            component.add(p)
            stack.extend(q for q in graph[p] if q not in component)
        remaining.difference_update(component)
        ends = sorted(p for p in component if len(graph[p]) == 1)
        if not ends:
            loops.append(component)
            continue
        if len(ends) != 2:
            raise ValueError("a path needs two terminals")
        ordered, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            next_cell, = (q for q in graph[current] if q != previous)
            ordered.append(next_cell)
            previous, current = current, next_cell
        paths.append(ordered)
    paths.sort(key=lambda p: (p[0], p[-1]))
    return graph, paths, loops


def _parse_core(a, track, switch, probe):
    h, w = a.shape
    if h < 3 or h % 2 != 1 or w < 9:
        raise ValueError("invalid lane dimensions")
    if not np.isin(a[::2], [0, track, probe]).all():
        raise ValueError("unexpected lane color")
    if not np.isin(a[1::2], [0, track, switch]).all():
        raise ValueError("unexpected separator color")
    if not np.all(np.count_nonzero(a[::2], axis=1) * 4 >= 3 * w):
        raise ValueError("too little intact lane evidence")
    probe_cells = set(map(tuple, np.argwhere(a == probe)))
    if not probe_cells or any(r % 2 or c not in (0, w - 1) for r, c in probe_cells):
        raise ValueError("probes must be lane terminals")
    centers = []
    for r in range(1, h, 2):
        if np.count_nonzero(a[r] == track) > max(1, w // 6):
            raise ValueError("too many separator specks")
        xs = np.flatnonzero(a[r] == switch).tolist()
        while xs:
            start = xs[0]
            if xs[:3] != [start, start + 1, start + 2] or start + 3 in xs:
                raise ValueError("switches must be isolated three-cell bars")
            c = start + 1
            if not 2 <= c <= w - 3:
                raise ValueError("switch too near a terminal")
            centers.append((r, c))
            xs = xs[3:]
    if not centers:
        raise ValueError("at least one switch is required")
    board = _rewire(h, w, centers, track, switch)
    holes = set(map(tuple, np.argwhere((a == 0) & (np.indices(a.shape)[0] % 2 == 0))))
    if any(board[p] != track for p in holes):
        raise ValueError("a defect cannot occupy a cell replaced by a divider")
    graph, paths, loops = _graph(board == track)
    if any(len(graph[p]) != 1 for p in probe_cells):
        raise ValueError("a probe must be an exposed terminal")
    ends = {p for path in paths for p in (path[0], path[-1])}
    if any(r % 2 or c not in (0, w - 1) for r, c in ends):
        raise ValueError("unexpected internal terminal")
    extras = set(map(tuple, np.argwhere((a == track) & (np.indices(a.shape)[0] % 2 == 1))))
    return {"a": a, "track": track, "switch": switch, "probe": probe,
            "centers": centers, "board": board, "holes": holes,
            "probes": probe_cells, "extras": extras, "graph": graph,
            "paths": paths, "loops": loops}


def _inspect(value):
    a = _grid(value)
    colors = set(map(int, np.unique(a)))
    if 0 not in colors or len(colors) != 4:
        raise ValueError("the field and three distinct visible roles are required")
    candidates = []
    for transposed in (False, True):
        core = a.T if transposed else a
        if core.shape[0] % 2 != 1 or core.shape[0] < 3 or core.shape[1] < 9:
            continue
        for track, switch, probe in permutations(sorted(colors - {0})):
            try:
                info = _parse_core(core, track, switch, probe)
            except ValueError:
                continue
            info["transposed"] = transposed
            candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("lane axis and roles must be uniquely visible")
    return candidates[0]


def _walk_hits(info):
    """Input solver: a separate inward walk from each visible source."""
    active = set()
    for source in sorted(info["probes"]):
        previous, current = None, source
        while True:
            if current in info["holes"]:
                active.add(current)
                break
            choices = [q for q in info["graph"][current] if q != previous]
            if not choices:
                break
            next_cell, = choices
            previous, current = current, next_cell
    return active


def _halo(cells, shape):
    result = np.zeros(shape, dtype=bool)
    for r, c in cells:
        result[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return result


def _paint(info, active):
    out = info["board"].copy()
    for p in info["holes"]:
        out[p] = 0
    for p in info["probes"]:
        out[p] = info["probe"]
    halo = _halo(active, out.shape)
    for p in info["holes"]:
        halo[p] = False
    out[halo] = info["probe"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _inspect(input_grid)
    out = _paint(info, _walk_hits(info))
    return (out.T if info["transposed"] else out).copy()


def _ordered_hits(paths, holes, probes):
    """Construction labels use ordered component positions, not probe walks."""
    active = set()
    for path in paths:
        defects = [p for p in path if p in holes]
        if defects:
            if path[0] in probes:
                active.add(defects[0])
            if path[-1] in probes:
                active.add(defects[-1])
    return active


def _unturned_hits(info):
    active = set()
    for r, c in info["probes"]:
        row_holes = sorted(p for p in info["holes"] if p[0] == r)
        if row_holes:
            active.add(row_holes[0] if c == 0 else row_holes[-1])
    return active


def _nearest_hits(info):
    active = set()
    for r, c in info["probes"]:
        if info["holes"]:
            distances = {p: (r - p[0]) ** 2 + (c - p[1]) ** 2 for p in info["holes"]}
            best = min(distances.values())
            active.update(p for p, d in distances.items() if d == best)
    return active


def _features(info):
    active = _walk_hits(info)
    out = _paint(info, active)
    hole_mask = np.zeros(out.shape, dtype=bool)
    for p in info["holes"]:
        hole_mask[p] = True
    coverage = np.zeros(out.shape, dtype=np.int16)
    for p in active:
        coverage += _halo({p}, out.shape)
    source_paths = [p for p in info["paths"] if p[0] in info["probes"] or p[-1] in info["probes"]]
    return {"lanes": (out.shape[0] + 1) // 2, "length": out.shape[1],
            "switches": len(info["centers"]), "holes": len(info["holes"]),
            "extras": len(info["extras"]), "probes": len(info["probes"]),
            "active": len(active), "dark": len(info["holes"] - active),
            "loops": len(info["loops"]),
            "multi_switch_row": max(sum(r == y for y, _ in info["centers"])
                                    for r, _ in info["centers"]) >= 2,
            "overlapping_halos": bool(np.any((coverage > 1) & ~hole_mask)),
            "halo_on_divider": bool(np.any((coverage > 0) & (info["board"] == info["switch"]))),
            "clipped": any(r in (0, out.shape[0] - 1) or c in (0, out.shape[1] - 1) for r, c in active),
            "adjacent_holes": any((r + dy, c + dx) in info["holes"]
                                  for r, c in info["holes"] for dy, dx in _DIRS),
            "clean_probe_path": any(not (set(p) & info["holes"]) for p in source_paths),
            "far_erasure": any(not _halo(active, out.shape)[p] and out[p] == 0 for p in info["extras"]),
            "routing_witness": not np.array_equal(out, _paint(info, _unturned_hits(info))),
            "shielding_witness": not np.array_equal(out, _paint(info, info["holes"])),
            "distance_witness": not np.array_equal(out, _paint(info, _nearest_hits(info)))}


def _layout(rng, lanes, width, dense):
    centers, previous = [], []
    for r in range(1, 2 * lanes - 1, 2):
        wanted = int(rng.integers(0, min(4, (width - 1) // 4) + 1)) if dense else int(rng.choice([0, 1, 1, 1]))
        chosen = []
        for raw in rng.permutation(np.arange(2, width - 2)):
            c = int(raw)
            if len(chosen) == wanted:
                break
            if all(abs(c - p) >= 2 for p in previous) and all(abs(c - p) >= 4 for p in chosen):
                chosen.append(c)
        centers.extend((r, c) for c in sorted(chosen))
        previous = chosen
    if not centers:
        centers = [(int(rng.integers(lanes - 1)) * 2 + 1, int(rng.integers(2, width - 2)))]
    return centers


def _make_pair(rng, palette, kind, split, regime, allowed_effects=None):
    track, switch, probe = palette
    for attempt in range(160):
        if regime == 0:
            low, high = (3, 6) if split == "train" else (6, 16)
        else:
            low, high = 3, 11
        lanes = int(rng.integers(low, high))
        width = int(rng.integers(12, 31))
        h = 2 * lanes - 1
        centers = _layout(rng, lanes, width, rng.random() < .6)
        board = _rewire(h, width, centers, track, switch)
        graph, paths, loops = _graph(board == track)
        eligible = {p for p in graph if p[0] % 2 == 0}
        interior = eligible - {p for path in paths for p in (path[0], path[-1])}
        holes, probes = set(), set()
        protected = set()
        if kind == "route":
            choices = [p for p in paths if abs(p[0][0] - p[-1][0]) >= 4 and p[0][1] != p[-1][1]]
            if not choices:
                continue
            path = choices[int(rng.integers(len(choices))) ]
            path = path if rng.random() < .5 else path[::-1]
            source, target = path[0], path[-1]
            decoys = sorted(p for p in interior - set(path) if p[0] == source[0])
            if not decoys:
                continue
            decoy = decoys[int(rng.integers(len(decoys)))]
            holes.update((target, decoy))
            probes.add(source)
            protected.update(path)
            # A second, undamaged circuit demonstrates that probes themselves
            # do not automatically produce halos or move to the other end.
            clean = [p for p in paths if p is not path and not (set(p) & (set(path) | holes))]
            if clean:
                q = clean[int(rng.integers(len(clean)))]
                probes.add(q[0])
                protected.update(q)
        elif kind == "shield":
            choices = [p for p in paths if len([q for q in p if q in interior]) >= 7]
            if not choices:
                continue
            path = choices[int(rng.integers(len(choices))) ]
            choices = [q for q in path if q in interior]
            # Three spaced defects: each end sees one and the middle is hidden.
            ids = sorted(map(int, rng.choice(len(choices), 3, replace=False)))
            if min(np.diff(ids)) < 2:
                continue
            holes.update(choices[i] for i in ids)
            probes.update((path[0], path[-1]))
        else:
            for path in paths:
                for p in (path[0], path[-1]):
                    if rng.random() < .4:
                        probes.add(p)
        if not probes:
            probes.add(paths[0][0])
        candidates = sorted(eligible - probes - protected - holes)
        rng.shuffle(candidates)
        wanted = int(rng.integers(8, 17) if regime == 2 and split == "test" else rng.integers(3, 9))
        if kind == "shield":
            wanted = max(wanted, 5)
        for p in candidates:
            if len(holes) >= wanted:
                break
            if sum(q[0] == p[0] for q in holes) < width // 4:
                holes.add(p)
                # Occasional short clusters, still bounded by lane evidence.
                neighbor = (p[0], p[1] + 1)
                if (rng.random() < .22 and neighbor in eligible - probes - protected - holes
                        and sum(q[0] == p[0] for q in holes) < width // 4):
                    holes.add(neighbor)
        if kind != "route":
            for path in paths:
                for p in (path[0], path[-1]):
                    if p not in holes and rng.random() < .2:
                        probes.add(p)
        inp = np.zeros((h, width), dtype=np.int8)
        inp[::2] = track
        for r, c in centers:
            inp[r, c - 1:c + 2] = switch
        for p in holes:
            inp[p] = 0
        for p in probes:
            inp[p] = probe
        extra_candidates = [(r, c) for r in range(1, h, 2) for c in range(width) if inp[r, c] == 0]
        rng.shuffle(extra_candidates)
        extra_count = int(rng.integers(1 if kind == "shield" else 0, min(9, lanes + 3)))
        extras = set()
        for p in extra_candidates:
            if len(extras) >= extra_count:
                break
            if sum(q[0] == p[0] for q in extras) < max(1, width // 6):
                extras.add(p)
                inp[p] = track
        info = _parse_core(inp, *palette)
        f = _features(info)
        if not f["active"]:
            continue
        if allowed_effects is not None and any(f[k] and k not in allowed_effects for k in _TAUGHT_EFFECTS):
            continue
        if kind == "shield" and not (f["shielding_witness"] and f["far_erasure"]):
            continue
        if kind == "route" and not (f["routing_witness"] and f["distance_witness"] and f["clean_probe_path"]):
            continue
        if regime == 2 and split == "test" and f["holes"] < 8:
            continue
        active = _ordered_hits(paths, holes, probes)
        # Construction paints from explicit masks; solve independently recovers
        # the input scaffold and walks each source before painting its answer.
        out = board.copy()
        for p in probes:
            out[p] = probe
        for r, c in active:
            out[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = probe
        for p in holes:
            out[p] = 0
        rotation = int(rng.integers(4))
        if regime == 1:
            rotation = int(rng.choice([0, 2] if split == "train" else [1, 3]))
        if rng.random() < .5:
            inp, out = np.fliplr(inp), np.fliplr(out)
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        try:
            parsed = _inspect(inp)
        except ValueError:
            continue
        if not np.array_equal(_paint(parsed, _walk_hits(parsed)), out.T if parsed["transposed"] else out):
            raise AssertionError("explicit construction disagrees with input reconstruction")
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct the required circuit witnesses in 160 attempts")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    fixed = tuple(map(int, rng.choice(np.arange(1, 10), 3, replace=False)))
    vary_palette = bool(rng.integers(2))
    anchors = list(map(int, rng.permutation(num_train)[:2]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        taught = None if split == "train" else {k for k in _TAUGHT_EFFECTS
                    if any(_features(_inspect(p["input"]))[k] for p in puzzle["train"])}
        for i in range(count):
            kind = "shield" if split == "train" and i == anchors[0] else "route" if split == "train" and i == anchors[1] else "general"
            for _ in range(12):
                palette = tuple(map(int, rng.choice(np.arange(1, 10), 3, replace=False))) if vary_palette else fixed
                pair = _make_pair(rng, palette, kind, split, regime, taught)
                a = pair["input"]
                signature = (a.shape, a.tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and not np.array_equal(a, b)
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            info = _inspect(a)
            active = _walk_hits(info)
            canonical_out = b.T if info["transposed"] else b
            assert np.array_equal(canonical_out, _paint(info, active))
            assert all(canonical_out[p] == 0 for p in info["holes"])
            assert all(canonical_out[p] == info["probe"] for p in info["probes"])
            assert active <= info["holes"] and active
            assert _ordered_hits(info["paths"], info["holes"], info["probes"]) == active
            assert all(not (loop & info["probes"]) for loop in info["loops"])
            if split == "train":
                training.append(_features(info))
            else:
                testing.append(_features(info))
    for feature in ("routing_witness", "shielding_witness", "distance_witness", "far_erasure", "clipped", "clean_probe_path"):
        assert any(f[feature] for f in training), f"training must expose {feature}"
    for feature in _TAUGHT_EFFECTS:
        assert not any(f[feature] for f in testing) or any(f[feature] for f in training), feature


def self_test():
    # Hand-solvable circuit with one switch. The left probe sees the top gap
    # first; a farther gap on the same circuit remains dark. A right-hand
    # defect belongs to a different, unprobed circuit, and a speck is erased.
    a = np.array([[3, 0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 2, 2, 2, 0, 0, 1],
                  [1, 1, 0, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
    expected = np.array([[3, 0, 3, 1, 2, 1, 1, 1, 0],
                         [3, 3, 3, 1, 2, 1, 0, 0, 0],
                         [1, 1, 0, 1, 2, 1, 1, 1, 1]], dtype=np.int8)
    assert np.array_equal(solve(a), expected)
    # Two staggered switches connect the upper-right probe to a lower-left
    # boundary fault. A geometrically closer upper-left fault is unreachable.
    a = np.ones((5, 9), dtype=np.int8)
    a[1::2] = 0
    a[1, 1:4] = a[3, 3:6] = 2
    a[0, 8], a[0, 0], a[4, 0], a[1, 8] = 3, 0, 0, 1
    info = _inspect(a)
    assert _walk_hits(info) == {(4, 0)}
    assert _unturned_hits(info) == _nearest_hits(info) == {(0, 0)}
    assert solve(a)[1, 8] == 0
    # An unprobed closed loop cannot light a fault merely because the loop
    # would be connected after filling the gap. Its two external paths remain
    # ordinary circuits. This also exercises a two-lane family member.
    a = np.ones((3, 11), dtype=np.int8)
    a[1] = 0
    a[1, 1:4] = a[1, 5:8] = 2
    a[0, 0], a[2, 1], a[0, 4] = 3, 0, 0
    info = _inspect(a)
    assert len(info["loops"]) == 1
    assert _walk_hits(info) == {(2, 1)}
    assert solve(a)[0, 4] == 0
    shapes, layouts, palettes, signatures = set(), set(), set(), set()
    records = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = []
        local_layouts = set()
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                info = _inspect(inp)
                f = _features(info)
                records.append(f)
                shapes.add(inp.shape)
                palettes.add((info["track"], info["switch"], info["probe"]))
                normalized = np.zeros(inp.shape, dtype=np.int8)
                for code, color in enumerate((info["track"], info["switch"], info["probe"]), 1):
                    normalized[inp == color] = code
                key = (normalized.shape, normalized.tobytes())
                layouts.add(key)
                local_layouts.add(key)
                signature.append((inp.shape, inp.tobytes(), out.tobytes()))
                assert np.array_equal(solve(inp.copy()), out)
                if seed < 3:
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                        assert np.array_equal(solve(np.fliplr(np.rot90(inp, k))), np.fliplr(np.rot90(out, k)))
        assert len(local_layouts) == 5
        signatures.add(tuple(signature))
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    assert len(signatures) == 42 and len(layouts) >= 190
    assert len(shapes) >= 65 and len(palettes) >= 30
    for feature in ("loops", "multi_switch_row", "overlapping_halos", "adjacent_holes"):
        assert any(f[feature] for f in records), feature
    assert len({f["switches"] for f in records}) >= 12
    assert len({f["holes"] for f in records}) >= 10
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
