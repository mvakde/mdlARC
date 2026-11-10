"""The shuttle clock: contour-sensitive sensor sequencing during rigid docking.

Only NumPy and the standard library are used. No parent generator is imported.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _box(points):
    p = np.asarray(points)
    return tuple(map(int, (p[:, 0].min(), p[:, 0].max(),
                           p[:, 1].min(), p[:, 1].max())))


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    component.append(q)
                    stack.append(q)
        result.append(np.array(sorted(component), dtype=int))
    return result


def _geometry(body, stop):
    t, b, l, r = _box(body)
    st, sb, sl, sr = _box(stop)
    row_sep, col_sep = b < st or sb < t, r < sl or sr < l
    assert row_sep != col_sep, "The boxes must separate on exactly one axis"
    if r < sl:
        direction, gap = (0, 1), sl - r - 1
    elif sr < l:
        direction, gap = (0, -1), l - sr - 1
    elif b < st:
        direction, gap = (1, 0), st - b - 1
    else:
        direction, gap = (-1, 0), t - sb - 1
    assert gap >= 1
    return direction, int(gap)


def _read_input(grid):
    """Recognize roles by component geometry, then read the stop's clock."""
    a = _grid(grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert counts.max() * 2 > a.size
    pieces = _components(a != bg)
    large = [p for p in pieces if len(p) > 1]
    assert len(large) == 2
    stops, bodies = [], []
    for p in large:
        t, b, l, r = _box(p)
        rectangle = len(p) == (b - t + 1) * (r - l + 1)
        colors = np.unique(a[p[:, 0], p[:, 1]])
        if rectangle and len(colors) == 3:
            stops.append(p)
        elif not rectangle and len(colors) == 1 and len(p) >= 5:
            bodies.append(p)
    assert len(stops) == len(bodies) == 1
    body, stop = bodies[0], stops[0]
    direction, gap = _geometry(body, stop)
    dr, dc = direction
    t, b, l, r = _box(stop)
    patch = a[t:b + 1, l:r + 1]
    # Each stripe is uniform across the transverse thickness. Read near to far.
    if dc:
        assert np.all(patch == patch[0:1, :])
        line = patch[0, ::dc]
    else:
        assert np.all(patch == patch[:, 0:1])
        line = patch[::dr, 0]
    assert len(line) % 3 == 0
    band = len(line) // 3
    phases = tuple(int(line[i * band]) for i in range(3))
    assert len(set(phases)) == 3
    assert np.array_equal(line, np.repeat(phases, band))
    # Motion-parallel slices are intervals; hence (time, transverse) has no ties.
    primary = 1 if dc else 0
    transverse = 1 - primary
    for level in np.unique(body[:, transverse]):
        occupied = body[body[:, transverse] == level, primary]
        assert len(occupied) == occupied.max() - occupied.min() + 1
    sensors = np.array([p[0] for p in pieces if len(p) == 1], dtype=int).reshape(-1, 2)
    assert len(sensors) >= 1
    # Single cells really are isolated sensors, including diagonal separation.
    all_large = np.concatenate((body, stop))
    for i, q in enumerate(sensors):
        assert np.all(np.max(np.abs(all_large - q), axis=1) > 1)
        if i:
            assert np.all(np.max(np.abs(sensors[:i] - q), axis=1) > 1)
    final = body + np.array(direction) * gap
    assert np.all(final >= 0) and np.all(final < np.array(a.shape))
    fixed = set(map(tuple, np.concatenate((stop, sensors)).tolist()))
    assert not fixed.intersection(map(tuple, final.tolist()))
    return {"grid": a, "bg": bg, "body": body, "stop": stop,
            "sensors": sensors, "direction": direction, "gap": gap,
            "phases": phases, "band": band,
            "body_color": int(a[tuple(body[0])])}


def _event_order(body, sensors, direction, gap):
    """First actual occupied-cell encounter, not bounding-box entry."""
    cells = set(map(tuple, body.tolist()))
    dr, dc = direction
    events = []
    for r0, c0 in sensors:
        r, c = int(r0), int(c0)
        for time in range(1, gap + 1):
            if (r - dr * time, c - dc * time) in cells:
                events.append((time, dr * c - dc * r, r, c))
                break
    events.sort()
    return events


def _paint(info, events):
    answer = info["grid"].copy()
    body = info["body"]
    final = body + np.array(info["direction"]) * info["gap"]
    answer[body[:, 0], body[:, 1]] = info["bg"]
    answer[final[:, 0], final[:, 1]] = info["body_color"]
    for index, (_, _, r, c) in enumerate(events):
        answer[r, c] = info["phases"][index % 3]
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    return _paint(info, _event_order(info["body"], info["sensors"],
                                    info["direction"], info["gap"]))


def _static_events(info, points):
    """A plausible shortcut: scan positions along the direction of travel."""
    dr, dc = info["direction"]
    return sorted((dr * int(r) + dc * int(c), dr * int(c) - dc * int(r),
                   int(r), int(c)) for r, c in points)


def _bbox_events(info):
    """Ablation that replaces the contour by its bounding rectangle."""
    t, b, l, r = _box(info["body"])
    rectangle = np.array([(y, x) for y in range(t, b + 1)
                          for x in range(l, r + 1)], dtype=int)
    # Rectangle slices are intervals too, so the event sorter is applicable.
    return _event_order(rectangle, info["sensors"], info["direction"], info["gap"])


def _facts(first, second):
    info = _read_input(first)
    out = _grid(second)
    assert info["grid"].shape == out.shape
    events = _event_order(info["body"], info["sensors"],
                          info["direction"], info["gap"])
    assert len({event[:2] for event in events}) == len(events)
    expected = _paint(info, events)
    assert np.array_equal(expected, out), "Wrong movement, activation, or clock phase"
    body, stop = info["body"], info["stop"]
    final = body + np.array(info["direction"]) * info["gap"]
    # Check adjacency explicitly (_geometry intentionally accepts positive gaps only).
    bt, bb, bl, br = _box(final)
    st, sb, sl, sr = _box(stop)
    dr, dc = info["direction"]
    assert ((dc == 1 and br + 1 == sl) or (dc == -1 and sr + 1 == bl)
            or (dr == 1 and bb + 1 == st) or (dr == -1 and sb + 1 == bt))
    assert np.all(out[stop[:, 0], stop[:, 1]] == info["grid"][stop[:, 0], stop[:, 1]])
    assert np.all(out[final[:, 0], final[:, 1]] == info["body_color"])
    assert np.count_nonzero(out == info["body_color"]) == len(body)
    event_points = {(r, c) for _, _, r, c in events}
    misses = [tuple(map(int, q)) for q in info["sensors"] if tuple(q) not in event_points]
    assert all(out[q] == info["grid"][q] for q in misses)
    distances = np.max(np.abs(final[:, None, :] - stop[None, :, :]), axis=2)
    clear = bool(np.all(distances > 1))
    static = _paint(info, _static_events(info, [(r, c) for _, _, r, c in events]))
    bbox_answer = _paint(info, _bbox_events(info))
    leave_sensors = _paint(info, [])
    colors_at_events = [int(info["grid"][r, c]) for _, _, r, c in events]
    repeated_cross_phase = any(colors_at_events[i] == colors_at_events[j] and i % 3 != j % 3
                               for i in range(len(events)) for j in range(i))
    ties = any(n > 1 for n in Counter(event[0] for event in events).values())
    later = _event_order(body, info["sensors"], info["direction"], info["gap"] + 3)
    # A late bead in the swept bounding rectangle, but beyond the contour's reach.
    late = bool({e[2:] for e in later} - event_points)
    return dict(info, events=events, active=len(events), misses=len(misses), clear=clear,
                ties=ties, chronological=not np.array_equal(static, out),
                contour_matters=not np.array_equal(bbox_answer, out),
                cycle_matters=not np.array_equal(leave_sensors, out),
                repeated_cross_phase=repeated_cross_phase, late=late,
                silhouette=(bb - bt + 1, br - bl + 1,
                            tuple(map(tuple, (body - body.min(axis=0)).tolist()))))


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _palette(rng):
    return tuple(map(int, rng.choice(10, 5, replace=False)))


def _mask(rng, clear, thin=False):
    """A connected variable-width ribbon; each horizontal slice is an interval."""
    h = int(rng.integers(10, 15) if thin else rng.integers(6, 11))
    w = int(rng.integers(6, 9))
    backbone = int(rng.integers(0, 2))
    left = rng.integers(0, backbone + 1, size=h)
    right = rng.integers(max(1, backbone), 4 if thin else w, size=h)
    left[int(rng.integers(h))] = 0
    if clear:
        right[0] = int(rng.integers(max(2, backbone), w - 3))
        right[1] = int(rng.integers(max(2, backbone), w - 2))
        right[int(rng.integers(3, h))] = w - 1
    else:
        short_row = int(rng.integers(h))
        long_row = int(rng.choice([r for r in range(h) if r != short_row]))
        right[short_row] = 2
        right[long_row] = w - 1
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        mask[r, int(left[r]):int(right[r]) + 1] = True
    return mask


def _make_pair(rng, rotation, palette, band, calibration=False, clear=False, testing=False):
    """Construct in rightward coordinates, then rotate the complete scene."""
    bg, body_color, *phases = palette
    for _attempt in range(160):
        short_run = not calibration and rng.random() < 0.25
        mask = _mask(rng, clear, short_run)
        h, w = mask.shape
        thickness = int(rng.integers(2, 4))
        gap = int(rng.integers(6, 9) if short_run else
                  rng.integers(9, 15 if testing else 14))
        depth = 3 * band
        if clear:
            offset = -thickness + 1
        elif rng.random() < 0.7:
            facing_row = int(rng.choice(np.flatnonzero(mask[:, -1])))
            offset = facing_row - int(rng.integers(thickness))
        else:
            offset = int(rng.integers(-thickness + 1, h))
        min_row, max_row = min(0, offset), max(h, offset + thickness)
        extent_h, extent_w = max_row - min_row, w + gap + depth
        if extent_w > 30:
            continue
        gh = int(rng.integers(extent_h + 3, min(30, extent_h + 14) + 1))
        if rng.random() < 0.15:
            gh = 30
        gw = int(rng.integers(extent_w, min(30, extent_w + 6) + 1))
        rp = int(rng.integers(gh - extent_h + 1)) - min_row
        cp = int(rng.integers(gw - extent_w + 1))
        body = np.argwhere(mask) + (rp, cp)
        stop = np.array([(r, c) for r in range(rp + offset, rp + offset + thickness)
                         for c in range(cp + w + gap, cp + w + gap + depth)], dtype=int)
        final = body + (0, gap)
        # Sensor input pixels must be isolated from the two large components.
        forbidden = set()
        for r, c in np.concatenate((body, stop)):
            forbidden.update((int(r + dr), int(c + dc))
                             for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        forbidden.update(map(tuple, final.tolist()))
        available = {(r, c) for r in range(gh) for c in range(gw)
                     if (r, c) not in forbidden}
        sweep, future = {}, {}
        for t in range(1, gap + 4):
            for r, c in body:
                q = (int(r), int(c + t))
                if q in available:
                    future.setdefault(q, t)
                    if t <= gap:
                        sweep.setdefault(q, t)
        candidates = sorted(sweep)
        if len(candidates) < 7:
            continue
        # Some ordinary scenes intentionally agree with a spatial scan: expose
        # the mechanism on a common front, then contrast it with calibrations.
        if not calibration and rng.random() < 0.35:
            fronts = {int(r): int(body[body[:, 0] == r, 1].max())
                      for r in np.unique(body[:, 0])}
            groups = {}
            for q in candidates:
                groups.setdefault(fronts[q[0]], []).append(q)
            aligned = max(groups.values(), key=len)
            if len(aligned) >= 12:
                candidates = aligned
        untied = not calibration and not short_run and rng.random() < 0.45
        # An unhit dot inside the bbox sweep contradicts rectangular sweeping.
        bt, bb, bl, br = _box(body)
        late_pool = [q for q, t in future.items() if t > gap
                     and bt <= q[0] <= bb and bl <= q[1] <= br + gap]
        if not late_pool:
            continue
        late = late_pool[int(rng.integers(len(late_pool)))]
        chosen = []
        blocked = set()

        def add(q):
            chosen.append(q)
            blocked.update((q[0] + dr, q[1] + dc)
                           for dr in (-1, 0, 1) for dc in (-1, 0, 1))

        add(late)
        if calibration:
            buckets = {}
            for q in candidates:
                if q not in blocked:
                    buckets.setdefault(sweep[q], []).append(q)
            pairs = [(a, b) for bucket in buckets.values()
                     for i, a in enumerate(bucket) for b in bucket[i + 1:]
                     if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1]
            if not pairs:
                continue
            a, b = pairs[int(rng.integers(len(pairs)))]
            add(a)
            add(b)
        target = int(rng.integers(6, 9 if short_run else (11 if testing else 10)))
        if untied:
            possible_times = len({sweep[q] for q in candidates if q not in blocked})
            if possible_times < 6:
                continue
            target = min(target, possible_times)
        active_now = len(chosen) - 1
        used_times = {sweep[q] for q in chosen if q in sweep}
        for idx in rng.permutation(len(candidates)):
            q = candidates[int(idx)]
            if q not in blocked and (not untied or sweep[q] not in used_times):
                add(q)
                used_times.add(sweep[q])
                active_now += 1
                if active_now >= target:
                    break
        if active_now < target:
            continue
        # One to three further misses, including dots outside the swept bbox.
        miss_pool = sorted(available - set(sweep) - blocked)
        target_misses = int(rng.integers(2, 5))
        misses_now = 1
        for idx in rng.permutation(len(miss_pool)):
            q = miss_pool[int(idx)]
            if q not in blocked:
                add(q)
                misses_now += 1
                if misses_now >= target_misses:
                    break
        if misses_now < 2:
            continue
        active = sorted((sweep[q], -q[0], *q) for q in chosen if q in sweep)
        static = sorted(active, key=lambda e: (e[3], -e[2]))
        rank = {e[2:]: i for i, e in enumerate(active)}
        if calibration and not any(i % 3 != rank[e[2:]] % 3 for i, e in enumerate(static)):
            continue
        grid = np.full((gh, gw), bg, dtype=np.int8)
        grid[body[:, 0], body[:, 1]] = body_color
        for r, c in stop:
            grid[r, c] = phases[(c - cp - w - gap) // band]
        sensor_palette = list(map(int, rng.choice([c for c in range(10)
                                                  if c not in (bg, body_color)],
                                                 int(rng.integers(1, 5)), replace=False)))
        for q in chosen:
            grid[q] = sensor_palette[int(rng.integers(len(sensor_palette)))]
        # Incoming color cannot explain the three-phase labels.
        i = int(rng.integers(len(active)))
        j = int(rng.choice([j for j in range(len(active)) if j % 3 != i % 3]))
        grid[active[i][2:]] = grid[active[j][2:]] = sensor_palette[0]
        # Ensure the late miss visibly refutes a rectangular sweep, not by chance
        # having the color that the rectangular shortcut would assign there.
        rectangle = np.array([(r, c) for r in range(bt, bb + 1)
                              for c in range(bl, br + 1)], dtype=int)
        rect_events = _event_order(rectangle, np.array(chosen), (0, 1), gap)
        late_rank = next(i for i, e in enumerate(rect_events) if e[2:] == late)
        grid[late] = int(rng.choice([c for c in range(10)
                                    if c not in (bg, body_color, phases[late_rank % 3])]))
        # Labels are built from the construction's sweep, not by calling solve.
        answer = grid.copy()
        answer[body[:, 0], body[:, 1]] = bg
        answer[final[:, 0], final[:, 1]] = body_color
        for i, (_, _, r, c) in enumerate(active):
            answer[r, c] = phases[i % 3]
        pair = {"input": np.rot90(grid, rotation).copy(),
                "output": np.rot90(answer, rotation).copy()}
        facts = _facts(pair["input"], pair["output"])
        if calibration and not (facts["ties"] and facts["chronological"]):
            continue
        assert facts["contour_matters"] and facts["cycle_matters"] and facts["late"]
        assert not clear or facts["clear"]
        return pair
    raise RuntimeError("Bounded shuttle construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_palette = bool(rng.integers(2))
    palette = _palette(rng)
    band = int(rng.integers(1, 3))
    held_out = int(rng.integers(4))
    # Two perpendicular calibrations suffice even for the minimum 2/1 API.
    plans = [((held_out + 1) % 4, True, True),
             ((held_out + 2) % 4, True, False)]
    for _ in range(nt - 2):
        plans.append(((held_out + int(rng.integers(1, 4))) % 4,
                      False, bool(rng.integers(2))))
    rng.shuffle(plans)
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            if split == "train":
                rotation, calibration, clear = plans[i]
            else:
                rotation = held_out if i == 0 else int(rng.integers(4))
                calibration, clear = False, bool(rng.integers(2))
            for _ in range(12):
                pair = _make_pair(rng, rotation, palette if fixed_palette else _palette(rng),
                                  band, calibration, clear, split == "test")
                signature = _signature(pair["input"])
                if signature not in used:
                    used.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            signature = _signature(_grid(pair["input"]))
            assert signature not in seen
            seen.add(signature)
            f = _facts(pair["input"], pair["output"])
            assert f["active"] >= 6 and f["misses"] >= 2
            assert f["contour_matters"] and f["cycle_matters"] and f["late"]
            assert f["repeated_cross_phase"]
            assert not np.array_equal(pair["input"], pair["output"])
            facts[split].append(f)
    assert len({f["band"] for split in facts.values() for f in split}) == 1
    train = facts["train"]
    for axis in (0, 1):
        assert any(f["direction"][axis] and f["ties"] and f["chronological"] for f in train)
    assert any(f["clear"] for f in train)
    assert facts["test"][0]["direction"] not in {f["direction"] for f in train}


def _puzzle_signature(puzzle):
    return tuple((_signature(p["input"]), _signature(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded checks of 42 seeds, counterfactuals, geometry, and the count API."""
    shapes, silhouettes, directions, palettes, gaps, active_counts, bands = [set() for _ in range(7)]
    signatures, backgrounds = set(), set()
    clear_count = touching_count = edge_count = overlap_count = 0
    untied_count = spatial_agreement_count = 0
    fixed_count = variable_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        local_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _facts(pair["input"], pair["output"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                silhouettes.add(f["silhouette"])
                directions.add(f["direction"])
                palettes.add((f["bg"], f["body_color"], *f["phases"]))
                local_palettes.add((f["bg"], f["body_color"], *f["phases"]))
                backgrounds.add(f["bg"])
                gaps.add(f["gap"])
                active_counts.add(f["active"])
                bands.add(f["band"])
                clear_count += f["clear"]
                touching_count += not f["clear"]
                untied_count += not f["ties"]
                spatial_agreement_count += not f["chronological"]
                b = f["body"]
                final = b + np.array(f["direction"]) * f["gap"]
                overlap_count += bool(set(map(tuple, b.tolist())) & set(map(tuple, final.tolist())))
                foreground = np.argwhere(pair["input"] != f["bg"])
                edge_count += bool(np.any(foreground == 0) or
                                   np.any(foreground == np.array(pair["input"].shape) - 1))
                # Rigid rotations preserve the whole input-derived transformation.
                if seed < 3:
                    for k in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(pair["input"], k)),
                                              np.rot90(pair["output"], k))
        fixed_count += len(local_palettes) == 1
        variable_count += len(local_palettes) > 1
    assert len(signatures) == 42
    assert len(shapes) >= 70 and len(silhouettes) >= 160
    assert len(palettes) >= 60 and backgrounds == set(range(10))
    assert directions == {(0, 1), (0, -1), (1, 0), (-1, 0)}
    assert gaps == set(range(6, 15)) and active_counts.issuperset(range(6, 11))
    assert bands == {1, 2}
    assert min(clear_count, touching_count, edge_count, fixed_count, variable_count) >= 5
    assert overlap_count >= 2, "Short runs must exercise overlapping old/new footprints"
    assert untied_count >= 5 and spatial_agreement_count >= 5
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4242 + nt * 10 + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    assert _puzzle_signature(generate_puzzle(123, 3, 2)) == _puzzle_signature(generate_puzzle(123, 3, 2))
    damaged = generate_puzzle(41, 2, 1)
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    f = _read_input(damaged["test"][0]["input"])
    e = _event_order(f["body"], f["sensors"], f["direction"], f["gap"])[0]
    damaged["test"][0]["output"][e[2], e[3]] = f["phases"][1]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt clock phase accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal count combinations)")
