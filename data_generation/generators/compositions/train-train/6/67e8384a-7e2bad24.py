"""A single ray whose fourfold reflected prints are live collision barriers.

Only the standard library and NumPy are used. solve() parses geometric roles
from its argument; it never consults a generator, seed, or training examples.
"""

from collections import Counter
import numpy as np


HOST_SEEDS = (
    0, 273987630, 438043939, 896575581, 168512918, 888314536, 152816966,
    148120527, 364015768, 816153401, 86122371, 123570277, 750950057,
    814556509, 253496821, 764575816, 95133773, 745566505, 343985667,
    119038344, 403538221, 403913644, 561721416, 787982065, 640426566,
    630365639, 460151328, 535799522, 654476842, 965670000, 722442671,
    876582616, 185864281, 171389925, 662425323, 432393303, 804779288,
    906472053, 723626197, 555943256, 737846024, 52785249,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must lie in 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must lie in 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, component = [first], [first]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    component.append(q)
                    stack.append(q)
        result.append(component)
    return result


def _seed_order(points, shape):
    """Return an inward-ordered, straight diagonal, or None."""
    if not 2 <= len(points) <= 5:
        return None
    ordered = sorted(points)
    d = (ordered[1][0] - ordered[0][0], ordered[1][1] - ordered[0][1])
    if abs(d[0]) != 1 or abs(d[1]) != 1:
        return None
    if any((b[0]-a[0], b[1]-a[1]) != d
           for a, b in zip(ordered, ordered[1:])):
        return None
    h, w = shape
    boundary = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
    if sum(map(boundary, ordered)) != 1:
        return None
    if boundary(ordered[-1]):
        ordered.reverse()
    if not boundary(ordered[0]):
        return None
    # A non-corner emitter makes its entry edge unambiguous.
    r, c = ordered[0]
    if r in (0, h-1) and c in (0, w-1):
        return None
    return ordered


def _analyse(a):
    h, w = a.shape
    _require(h % 2 == w % 2 == 0 and min(h, w) >= 14,
             "plate must have even sides in 14..30")
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[int(np.argmax(counts))])
    _require(int(counts.max()) > a.size // 2, "background must occupy over half the plate")
    positions = {int(v): [tuple(map(int, p)) for p in np.argwhere(a == v)]
                 for v in colors if int(v) != bg}
    candidates = [(v, _seed_order(p, a.shape)) for v, p in positions.items()]
    candidates = [(v, p) for v, p in candidates if p is not None]
    _require(len(candidates) == 1, "exactly one diagonal boundary seed is required")
    ink, seed = candidates[0]
    bare = a.copy()
    bare[a == ink] = bg
    _require(np.array_equal(bare, bare[:, ::-1]) and
             np.array_equal(bare, bare[::-1, :]), "material plate must have both symmetries")
    # Each seed occupies only one open quadrant. Its three initial images
    # therefore cannot overlap it or any material.
    _require(len({(r // (h//2), c // (w//2)) for r, c in seed}) == 1,
             "seed must remain in one quadrant")
    mirrors, glass, orientation = None, None, {}
    bar_lengths, dot_count = [], 0
    for color, points in positions.items():
        if color == ink:
            continue
        comps = _components(points)
        if all(len(comp) == 1 for comp in comps):
            _require(glass is None, "at most one glass color")
            glass, dot_count = color, len(points)
        else:
            _require(mirrors is None, "at most one solid-mirror color")
            mirrors = color
            for comp in comps:
                rs, cs = {p[0] for p in comp}, {p[1] for p in comp}
                _require(len(comp) >= 3 and (len(rs) == 1 or len(cs) == 1),
                         "solid mirrors must be straight segments of at least three cells")
                axis = "H" if len(rs) == 1 else "V"
                for point in comp:
                    orientation[point] = axis
                bar_lengths.append(len(comp))
    _require(mirrors is not None or glass is not None, "plate needs an optical material")
    start = seed[0]
    side = ("top" if start[0] == 0 else "bottom" if start[0] == h-1
            else "left" if start[1] == 0 else "right")
    return dict(background=bg, ink=ink, mirror=mirrors, glass=glass,
                seed=seed, orientation=orientation, side=side,
                bar_lengths=bar_lengths, dots=dot_count)


def _orbit(r, c, h, w, printing="both"):
    rows = (r, h-1-r) if printing in ("both", "vertical") else (r,)
    cols = (c, w-1-c) if printing in ("both", "horizontal") else (c,)
    return {(rr, cc) for rr in rows for cc in cols}


def _trace(a, info, printing="both", reflect=True):
    """Trace on live ink; static input supplies the orientation of solid bars.

    Nondefault modes are explicit counterfactuals for validation only.
    """
    h, w = a.shape
    out = a.copy()
    ink, mirror, glass = (info[k] for k in ("ink", "mirror", "glass"))
    seed = info["seed"]
    physical = set(seed)
    printed = set()
    for r, c in seed:
        printed.update(_orbit(r, c, h, w, printing))
    for point in printed:
        _require(int(a[point]) in (info["background"], ink), "seed image hits material")
        out[point] = ink
    r, c = seed[-1]
    dr, dc = r-seed[-2][0], c-seed[-2][1]
    states, events, path = set(), [], []
    while True:
        state = r, c, dr, dc
        _require(state not in states, "cyclic zero-motion mirror trap")
        states.add(state)
        nr, nc = r+dr, c+dc
        if not (0 <= nr < h and 0 <= nc < w):
            reason = "edge"
            break
        if (nr, nc) in printed:
            reason = "self" if (nr, nc) in physical else "echo"
            break
        value = int(out[nr, nc])
        if mirror is not None and value == mirror and reflect:
            axis = info["orientation"][(nr, nc)]
            events.append(("R", axis, nr, nc))
            if axis == "H":
                dr = -dr
            else:
                dc = -dc
            continue
        if glass is not None and value == glass:
            events.append(("T", "dot", nr, nc))
        # Symmetry of the stationary scene ensures no print can cover a
        # mirror unless this is the deliberately no-reflection ablation.
        images = _orbit(nr, nc, h, w, printing)
        for point in images:
            out[point] = ink
        printed.update(images)
        physical.add((nr, nc))
        path.append((nr, nc))
        r, c = nr, nc
    return out, dict(events=events, path=path, reason=reason,
                     stop=(nr, nc), printed=printed, physical=physical)


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _trace(a, _analyse(a))[0]


def _late_print(a, info):
    """Tempting wrong rule: finish a single ordinary ray, then mirror its ink."""
    ordinary, _ = _trace(a, info, printing="none")
    out = a.copy()
    h, w = a.shape
    for r, c in np.argwhere(ordinary == info["ink"]):
        for point in _orbit(int(r), int(c), h, w):
            out[point] = info["ink"]
    return out


def _expand_quarter(q):
    top = np.concatenate((q, q[:, ::-1]), axis=1)
    return np.concatenate((top, top[::-1, :]), axis=0)


def _material_board(rng, large=False):
    low = 9 if large else 7
    h, w = map(int, rng.integers(low, 16, size=2))
    if rng.random() < 0.18:
        w = h
    q = np.zeros((h, w), dtype=np.int8)
    occupied = np.zeros_like(q, dtype=bool)
    target = int(rng.integers(2, 8))
    # Try more placements than needed; spacing the *entire dashed footprint*
    # keeps each gate legible rather than interlacing independent gates.
    for _ in range(target * 4):
        if target == 0:
            break
        horizontal = bool(rng.integers(2))
        material = int(rng.choice((2, 3), p=(0.57, 0.43)))
        axis = w if horizontal else h
        length = int(rng.integers(3, min(axis-2, 11)+1))
        if material == 3 and length % 2 == 0:
            length -= 1
        r = int(rng.integers(1, h-(0 if horizontal else length)))
        c = int(rng.integers(1, w-(length if horizontal else 0)))
        cells = [(r, c+i) if horizontal else (r+i, c) for i in range(length)]
        r0, c0 = cells[0]
        r1, c1 = cells[-1]
        if occupied[max(0, r0-1):r1+2, max(0, c0-1):c1+2].any():
            continue
        for i, point in enumerate(cells):
            occupied[point] = True
            if material == 2 or i % 2 == 0:
                q[point] = material
        target -= 1
    return _expand_quarter(q)


def _candidate(rng, requirement, wanted_axis=None, large=False):
    """Rejection sampling is bounded; there is no seed-dependent rule."""
    for _ in range(2400):
        field = _material_board(rng, large)
        h, w = field.shape
        # Several source positions on one random field reduce placement cost.
        for _source in range(5):
            length = int(rng.integers(2, 6))
            dc = int(rng.choice((-1, 1)))
            low = 1 + (length-1 if dc < 0 else 0)
            high = w//2-1 - (length-1 if dc > 0 else 0)
            if low >= high:
                continue
            c0 = int(rng.integers(low, high+1))
            seed = [(r, c0+dc*r) for r in range(length)]
            if any(field[p] != 0 for p in seed):
                continue
            a = field.copy()
            for point in seed:
                a[point] = 1
            try:
                info = _analyse(a)
                out, stats = _trace(a, info)
            except ValueError:
                continue
            events = stats["events"]
            kinds = {e[0] for e in events}
            axes = {e[1] for e in events if e[0] == "R"}
            quadrants = {(r//(h//2), c//(w//2)) for r, c in seed+stats["path"]}
            if len(stats["path"]) < 2 or not events:
                continue
            if requirement == "coupled":
                if kinds != {"R", "T"} or stats["reason"] != "echo":
                    continue
                # Require a genuinely older remote print, not just the
                # immediately preceding point's image across both seams.
                if stats["stop"] in _orbit(*stats["path"][-1], h, w):
                    continue
                try:
                    late = _late_print(a, info)
                except ValueError:
                    continue
                if np.array_equal(late, out):
                    continue
            elif requirement == "exit":
                if not axes or stats["reason"] != "edge":
                    continue
            elif requirement == "mixed":
                if axes != {"H", "V"} or len(quadrants) < 2:
                    continue
            elif requirement != "any":
                raise ValueError("unknown sampling requirement")
            if wanted_axis is not None:
                if len(axes) != 1:
                    continue
                first_axis = next(e[1] for e in events if e[0] == "R")
                if first_axis != wanted_axis:
                    a = a.T.copy()
            elif rng.random() < 0.5:
                a = a.T.copy()
            if rng.random() < 0.5:
                a = a[::-1, :].copy()
            if rng.random() < 0.5:
                a = a[:, ::-1].copy()
            return a
    raise RuntimeError(f"bounded search exhausted for {requirement}")


def _count(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    # None is an optional convenience; the specified 4/1 defaults remain literal.
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test+1, 11))
    elif train is None:
        train = int(rng.integers(test+1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train-1)+1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    palette = np.asarray(rng.choice(10, size=4, replace=False), dtype=np.int8)
    axis = str(rng.choice(("H", "V")))
    plans = [("coupled", axis), ("exit", "V" if axis == "H" else "H")]
    plans.extend(("any", None) for _ in range(train-2))
    rng.shuffle(plans)
    puzzle, used = {"train": [], "test": []}, set()
    for section, count in (("train", train), ("test", test)):
        for i in range(count):
            if section == "train":
                requirement, wanted_axis = plans[i]
            else:
                # The first test combines the two demonstrated reflection
                # orientations on one physical route across a mirror seam.
                requirement = "mixed" if i == 0 else str(rng.choice(("any", "mixed", "coupled")))
                wanted_axis = None
            for _ in range(40):
                a = _candidate(rng, requirement, wanted_axis, large=section == "test")
                a = palette[a]
                if _key(a) in used:
                    continue
                # Shape variation is evidence that the symmetry axes are
                # the plate's midlines, not fixed numeric coordinates.
                if section == "train" and i == 1 and a.shape == puzzle["train"][0]["input"].shape:
                    continue
                used.add(_key(a))
                puzzle[section].append({"input": a, "output": solve(a)})
                break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used, roles, train_axes, train_kinds = set(), None, set(), set()
    coupling_witness = ordinary_exit = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in used
            used.add(_key(a))
            info = _analyse(a)
            these_roles = tuple(info[k] for k in ("background", "ink", "mirror", "glass"))
            if roles is None:
                roles = these_roles
            # A material may be absent in a pair; present roles stay consistent.
            assert all(x is None or y is None or x == y for x, y in zip(roles, these_roles))
            roles = tuple(x if x is not None else y for x, y in zip(roles, these_roles))
            expected, stats = _trace(a, info)
            assert np.array_equal(out, expected)
            assert np.array_equal(out, out[::-1, :])
            assert np.array_equal(out, out[:, ::-1])
            changed = a != out
            assert np.any(changed) and np.all(out[changed] == info["ink"])
            assert np.all(out[a == info["ink"]] == info["ink"])
            if info["mirror"] is not None:
                assert np.all(out[a == info["mirror"]] == info["mirror"])
            assert len(stats["path"]) >= 2 and stats["events"]
            # Every final ink cell has the whole four-element orbit, and
            # the physical tip never enters a previously printed location.
            assert len(stats["printed"]) % 4 == 0
            assert len(stats["physical"]) == len(info["seed"]) + len(stats["path"])
            assert len(stats["printed"]) == 4 * len(stats["physical"])
            if stats["reason"] != "edge":
                assert stats["stop"] in stats["printed"]
            if section == "train":
                train_axes.update(e[1] for e in stats["events"] if e[0] == "R")
                train_kinds.update(e[0] for e in stats["events"])
                ordinary_exit |= stats["reason"] == "edge"
                last = stats["path"][-1]
                if (stats["reason"] == "echo" and
                        stats["stop"] not in _orbit(*last, *a.shape)):
                    try:
                        coupling_witness |= not np.array_equal(_late_print(a, info), out)
                    except ValueError:
                        pass
    assert train_axes == {"H", "V"} and train_kinds == {"R", "T"}
    assert coupling_witness and ordinary_exit
    assert len({_grid(p["input"]).shape for p in puzzle["train"]}) >= 2
    assert None not in roles


def _signature(puzzle):
    return tuple(tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural checks and semantic counterfactuals."""
    shapes, palettes, layouts, sides, lengths = set(), set(), set(), set(), set()
    orders, mirror_counts, gate_counts, signatures = set(), set(), set(), set()
    metrics = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed in HOST_SEEDS[:3]:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        local = Counter()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a, out = pair["input"], pair["output"]
                info = _analyse(a)
                _, stats = _trace(a, info)
                shapes.add(a.shape)
                palettes.add(tuple(info[k] for k in ("background", "ink", "mirror", "glass")))
                mapping = {}
                layouts.add((a.shape, tuple(mapping.setdefault(int(v), len(mapping)) for v in a.flat)))
                sides.add(info["side"])
                lengths.add(len(info["seed"]))
                orders.add(tuple(e[0] for e in stats["events"]))
                mirror_counts.add(len(info["bar_lengths"]))
                gate_counts.add(info["dots"])
                axes = {e[1] for e in stats["events"] if e[0] == "R"}
                local["mixed_"+section] += axes == {"H", "V"}
                metrics[stats["reason"]] += 1
                metrics["square"] += a.shape[0] == a.shape[1]
                metrics["rectangle"] += a.shape[0] != a.shape[1]
                metrics["near_limit"] += 30 in a.shape
                metrics["single_material"] += info["mirror"] is None or info["glass"] is None
                metrics["multi_reflect"] += sum(e[0] == "R" for e in stats["events"]) >= 3
                if stats["reason"] == "echo":
                    metrics["older_echo"] += stats["stop"] not in _orbit(*stats["path"][-1], *a.shape)
                    metrics["erased_glass_stop"] += int(a[stats["stop"]]) == info["glass"]
                if section == "train":
                    # Each omitted mechanism has a visible wrong prediction.
                    for mode in ("none", "horizontal", "vertical"):
                        wrong, _ = _trace(a, info, printing=mode)
                        local["omit_"+mode] += not np.array_equal(wrong, out)
                    no_reflect, _ = _trace(a, info, reflect=False)
                    local["omit_reflect"] += not np.array_equal(no_reflect, out)
                    for kind, _, r, c in stats["events"]:
                        if kind == "T":
                            assert out[r, c] == info["ink"] != a[r, c]
                            local["transmission"] += 1
                # Cell-level orbit formula, separate from set-based tracing.
                h, w = a.shape
                for r, c in stats["physical"]:
                    for rr, cc in ((r, c), (h-1-r, c), (r, w-1-c), (h-1-r, w-1-c)):
                        assert out[rr, cc] == info["ink"]
        assert local["mixed_test"] >= 1
        assert all(local[k] > 0 for k in ("omit_none", "omit_horizontal", "omit_vertical",
                                         "omit_reflect", "transmission"))
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(palettes) >= 35 and len(layouts) >= 190
    assert sides == {"top", "bottom", "left", "right"}
    assert lengths == {2, 3, 4, 5} and len(orders) >= 8
    assert len(mirror_counts) >= 4 and len(gate_counts) >= 5
    assert all(metrics[k] > 0 for k in ("edge", "echo", "square", "rectangle",
                                       "near_limit", "single_material", "multi_reflect",
                                       "erased_glass_stop"))
    assert metrics["older_echo"] >= 42
    # All legal explicit counts, plus partial counts and type validation.
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(71, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((None, None), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(919, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                        (3, 4), (True, 1), (4, False), (4.0, 1), (4, "1")):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(101)
    out = damaged["train"][0]["output"]
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=210, shapes=len(shapes), palettes=len(palettes),
                normalized_inputs=len(layouts), event_orders=len(orders),
                mirror_counts=sorted(mirror_counts), gate_dot_counts=sorted(gate_counts),
                source_lengths=sorted(lengths), entry_sides=sorted(sides),
                **dict(metrics))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
