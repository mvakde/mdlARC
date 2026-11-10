"""Paired sluices: cap endpoint geometry changes hydraulic connectivity.

Only standard Python and NumPy are used. No parent module is imported.
"""
from __future__ import annotations

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _diamond(center, radius, interior=False):
    r, c = center
    return [(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (abs(dr) + abs(dc) < radius if interior
                else abs(dr) + abs(dc) == radius)]


def _cap(center, radius, side):
    r, c = center
    return [(r + k + 1, c + side * (radius - k + 1))
            for k in range(radius + 1)]


def _component(mask, start):
    seen = np.zeros(mask.shape, dtype=bool)
    if not mask[start]:
        return seen
    todo = [start]
    seen[start] = True
    while todo:
        r, c = todo.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if _inside(p, mask.shape) and mask[p] and not seen[p]:
                seen[p] = True
                todo.append(p)
    return seen


def _components(mask):
    remaining = mask.copy()
    parts = []
    while remaining.any():
        p = tuple(map(int, np.argwhere(remaining)[0]))
        part = _component(remaining, p)
        parts.append(part)
        remaining[part] = False
    return parts


def _analyze(grid):
    """Read all roles, caps, first hits and paired gate matches from pixels."""
    h, w = grid.shape
    background = int(grid[0, 0])
    if grid[0, -1] != background:
        raise ValueError("the two upper corners must be background")
    candidates = []
    for radius in (2, 3, 4):
        for r in range(radius, h - radius):
            for c in range(radius, w - radius):
                center = (r, c)
                outline = _diamond(center, radius)
                structural = int(grid[outline[0]])
                if structural == background:
                    continue
                if not all(grid[p] == structural for p in outline):
                    continue
                interior = _diamond(center, radius, True)
                if not all(grid[p] == background for p in interior):
                    continue
                caps = {}
                for side in (-1, 1):
                    cells = _cap(center, radius, side)
                    if not all(_inside(p, grid.shape) for p in cells):
                        continue
                    color = int(grid[cells[0]])
                    if color not in (background, structural) and all(grid[p] == color for p in cells):
                        caps[side] = color
                if caps:
                    candidates.append(dict(center=center, radius=radius,
                                           structural=structural, caps=caps))
    # This family uses one controller operating a shared reservoir, not
    # separate little puzzles. The geometric parser supports every radius.
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous hollow diamond controller")
    emitter = candidates[0]
    solid = emitter["structural"]
    cap_colors = set(emitter["caps"].values())
    if len(cap_colors) != 1:
        raise ValueError("caps must identify one liquid color")
    water = cap_colors.pop()
    remaining_colors = set(map(int, np.unique(grid))) - {background, solid, water}
    if len(remaining_colors) != 1:
        raise ValueError("expected one distinct plug color")
    plug = remaining_colors.pop()
    cap_mask = np.zeros(grid.shape, dtype=bool)
    for side in emitter["caps"]:
        for p in _cap(emitter["center"], emitter["radius"], side):
            cap_mask[p] = True
    gates = []
    for part in _components(grid == plug):
        points = np.argwhere(part)
        cols = np.unique(points[:, 1])
        if len(cols) != 1 or len(points) < 2:
            raise ValueError("plugs must be separate vertical segments")
        col = int(cols[0])
        top, bottom = int(points[:, 0].min()), int(points[:, 0].max())
        if len(points) != bottom - top + 1 or col in (0, w - 1):
            raise ValueError("invalid plug segment")
        # A plug is visibly embedded in a full-height dam.
        if not np.all(np.isin(grid[:, col], (solid, plug))):
            raise ValueError("a plug must lie in a solid dam")
        gates.append((top, bottom, col))
    blockers = (grid == solid) | (grid == plug)
    opened = np.zeros(grid.shape, dtype=bool)
    hits_by_side = {}
    matched = []
    for side in emitter["caps"]:
        cells = _cap(emitter["center"], emitter["radius"], side)
        hits = []
        for endpoint in (cells[0], cells[-1]):
            r, c = endpoint
            hit = None
            while True:
                r, c = r + 1, c + side
                if not _inside((r, c), grid.shape):
                    break
                if blockers[r, c]:
                    hit = (r, c)
                    break
            hits.append(hit)
        hits_by_side[side] = tuple(hits)
        for top, bottom, col in gates:
            if set(hits) == {(top, col), (bottom, col)}:
                opened[top:bottom + 1, col] = True
                matched.append((top, bottom, col))
    if not matched:
        raise ValueError("at least one complete paired sluice must open")
    terrain = (blockers | cap_mask) & ~opened
    moving = (grid == water) & ~cap_mask
    if not moving.any():
        raise ValueError("missing movable liquid")
    return dict(background=background, solid=solid, water=water, plug=plug,
                emitter=emitter, caps=cap_mask, gates=gates, matched=matched,
                hits=hits_by_side, opened=opened, terrain=terrain, moving=moving)


def _pour(terrain, source, reverse=False, top_first=False):
    """Downward unit pours with lateral motion only at a supported row.

    Generated volumes end on complete horizontal levels. Two tie orders and
    two source orders must agree; those orders are verification probes, not a
    hidden convention of the task.
    """
    h, w = terrain.shape
    water = np.zeros(terrain.shape, dtype=bool)
    points = sorted(map(tuple, np.argwhere(source)),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    for rr, cc in points:
        r, c = int(rr), int(cc)
        while r > 0 and water[r, c]:
            r -= 1
        if terrain[r, c] or water[r, c]:
            raise ValueError("supply is trapped under a ceiling")
        for _ in range(h + 1):
            if r + 1 < h and not (terrain[r + 1, c] or water[r + 1, c]):
                r += 1
                continue
            left = right = c
            while left > 0 and not (terrain[r, left - 1] or water[r, left - 1]):
                left -= 1
            while right + 1 < w and not (terrain[r, right + 1] or water[r, right + 1]):
                right += 1
            exits = [] if r + 1 == h else [x for x in range(left, right + 1)
                if not (terrain[r + 1, x] or water[r + 1, x])]
            if not exits:
                water[r, c] = True
                break
            c = min(exits, key=lambda x: (abs(x - c), -x if reverse else x))
            r += 1
        else:
            raise AssertionError("pour failed to descend")
    return water


def _render_output(grid, analysis, settled):
    out = grid.copy()
    out[analysis["moving"] | analysis["opened"]] = analysis["background"]
    out[settled] = analysis["water"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    analysis = _analyze(grid)
    settled = _pour(analysis["terrain"], analysis["moving"])
    return _render_output(grid, analysis, settled)


def _profile(rng, width, lower, height, style):
    """Solid ground with variable floors and joined lower pockets."""
    if style == 0:
        values = np.full(width, int(rng.integers(lower, height + 1)))
    elif style == 1:
        values = np.rint(np.linspace(rng.integers(lower, height + 1),
                                    rng.integers(lower, height + 1), width)).astype(int)
    elif style == 2:
        values = rng.integers(lower, height + 1, size=width)
    elif style == 3:
        values = np.full(width, height, dtype=int)
        # A raised internal ridge joins two deeper pockets above its crest.
        values[max(1, width // 2):max(1, width // 2) + 1] = lower
        if width > 5:
            values[1] = lower + 1
    else:
        values = np.empty(width, dtype=int)
        level = int(rng.integers(lower, height + 1))
        for i in range(width):
            level = int(np.clip(level + rng.integers(-1, 2), lower, height))
            values[i] = level
    if rng.integers(2):
        values = values[::-1].copy()
    return values


def _make_pair(rng, radius, kind, test=False):
    """Construct basin capacities first, then an equal-volume visible supply.

    Output masks are derived from vessel levels, not from solve(). All four
    pouring orders are checked before the pair is returned.
    """
    for attempt in range(80):
        double = kind in ("witness_spill", "witness_equal") or bool(rng.integers(2))
        dmin = 2 * radius + 1
        if double:
            distances = list(map(int, rng.choice(np.arange(dmin, dmin + 3), 2, replace=False)))
            source_width = sum(distances) - 1
            spare = 30 - (source_width + 2 + 6)
            widths = [3, 3]
            for _ in range(int(rng.integers(spare + 1))):
                widths[int(rng.integers(2))] += 1
            left_dam = widths[0]
            right_dam = left_dam + source_width + 1
            width = right_dam + 1 + widths[1]
            center_col = left_dam + distances[0]
            chambers = [list(range(left_dam)), list(range(left_dam + 1, right_dam)),
                        list(range(right_dam + 1, width))]
            source_index = 1
            dams = {-1: left_dam, 1: right_dam}
            receivers = {-1: 0, 1: 2}
            distance = {-1: distances[0], 1: distances[1]}
            sides = [-1, 1]
        else:
            dist = int(rng.integers(dmin, dmin + 3))
            source_width = dist + radius + 2 + int(rng.integers(3))
            receiver_width = int(rng.integers(3, min(8, 29 - source_width) + 1))
            width = source_width + 1 + receiver_width
            center_col = source_width - dist
            chambers = [list(range(source_width)), list(range(source_width + 1, width))]
            source_index = 0
            dams = {1: source_width}
            receivers = {1: 1}
            distance = {1: dist}
            sides = [1]
        center_row = int(rng.integers(radius + 1,
                           min(radius + 3, 30 - max(distance.values()) - radius - 7) + 1))
        center = (center_row, center_col)
        expected_gates = {s: (center_row + distance[s] - radius,
                             center_row + distance[s] + radius, dams[s]) for s in sides}
        invalid = None
        if double and kind == "witness_spill":
            invalid = int(rng.choice(sides))
        elif double and kind != "witness_equal" and rng.random() < .42:
            invalid = int(rng.choice(sides))
        gates = dict(expected_gates)
        if invalid is not None:
            top, bottom, col = gates[invalid]
            shift = int(rng.choice([-1, 1]))
            gates[invalid] = (top + shift, bottom + shift, col)
        max_bottom = max(b for _, b, _ in gates.values())
        height = int(rng.integers(max_bottom + 7, 31))
        lower_floor = max_bottom + 3
        palette = list(map(int, rng.choice(10, 4, replace=False)))
        background, solid, liquid, plug = palette
        grid = np.full((height, width), background, dtype=np.int8)
        for col in dams.values():
            grid[:, col] = solid
        profiles = []
        for index, columns in enumerate(chambers):
            style = 3 if kind == "witness_equal" and index == source_index else int(rng.integers(5))
            profile = _profile(rng, len(columns), lower_floor, height, style)
            profiles.append(profile)
            for col, floor in zip(columns, profile):
                grid[int(floor):, col] = solid
        for top, bottom, col in gates.values():
            grid[top:bottom + 1, col] = plug
        for p in _diamond(center, radius):
            grid[p] = solid
        cap_mask = np.zeros(grid.shape, dtype=bool)
        for side in sides:
            for p in _cap(center, radius, side):
                grid[p] = liquid
                cap_mask[p] = True
        active = [s for s in sides if s != invalid]
        # Differing dam distances ensure a unique first (lower) outlet.
        first = max(active, key=lambda s: expected_gates[s][1])
        spill_bottom = expected_gates[first][1]
        regime = ("spill" if kind == "witness_spill" else
                  "equal" if kind == "witness_equal" else
                  "retain" if kind == "retain" else str(rng.choice(["retain", "spill", "equal"])))
        if test:
            regime = str(rng.choice(["spill", "equal"]))
        target = np.zeros(grid.shape, dtype=bool)

        def fill_chamber(index, surface):
            for col, floor in zip(chambers[index], profiles[index]):
                target[surface:int(floor), col] = True

        if regime == "retain":
            surface = int(rng.integers(spill_bottom + 2, int(profiles[source_index].min())))
            fill_chamber(source_index, surface)
        elif regime == "spill":
            fill_chamber(source_index, spill_bottom + 1)
            receiver = receivers[first]
            surface = int(rng.integers(spill_bottom + 2, int(profiles[receiver].min())))
            fill_chamber(receiver, surface)
        else:
            # Above the highest aperture's bottom, all selected vessels
            # communicate. Closed shifted plugs still exclude their vessel.
            lowest_bottom = min(expected_gates[s][1] for s in active)
            surface = max(center_row + radius + 2,
                          lowest_bottom - int(rng.integers(4)))
            fill_chamber(source_index, surface)
            for side in active:
                fill_chamber(receivers[side], surface)
                top, bottom, col = expected_gates[side]
                target[max(surface, top):bottom + 1, col] = True
        # This independent capacity construction stays below controller/caps.
        assert not np.any(target & cap_mask)
        volume = int(target.sum())
        source_area = np.zeros(grid.shape, dtype=bool)
        source_area[:, chambers[source_index]] = True
        source_area &= grid == background
        source_area[0] = False  # both upper corners visibly identify background
        # An absent side must remain visibly absent; supply is never allowed
        # to accidentally impersonate another uniform side cap.
        for side in (-1, 1):
            if side not in sides:
                for p in _cap(center, radius, side):
                    if _inside(p, grid.shape):
                        source_area[p] = False
        # Pick a free point below all controller pixels to exclude its sealed
        # hollow interior from the source's connected reservoir.
        anchor = (lower_floor - 1, chambers[source_index][0])
        eligible = _component(source_area, anchor)
        points = np.argwhere(eligible)
        if len(points) <= volume or volume == 0:
            continue
        jitter = rng.random(len(points))
        style = "low" if test else str(rng.choice(["compact", "scatter", "band"]))
        if style == "low":
            score = -points[:, 0] * 2 + jitter
        elif style == "compact":
            chosen_anchor = points[int(rng.integers(len(points)))]
            score = np.abs(points - chosen_anchor).sum(axis=1) + jitter
        elif style == "band":
            row = int(rng.integers(0, lower_floor))
            score = np.abs(points[:, 0] - row) * 3 + jitter
        else:
            score = jitter
        chosen = points[np.argsort(score)[:volume]].copy()
        outside = points[~target[points[:, 0], points[:, 1]]]
        if not len(outside):
            continue
        if np.all(target[chosen[:, 0], chosen[:, 1]]):
            chosen[-1] = outside[int(rng.integers(len(outside)))]
        moving = np.zeros(grid.shape, dtype=bool)
        moving[chosen[:, 0], chosen[:, 1]] = True
        grid[moving] = liquid
        opened = np.zeros(grid.shape, dtype=bool)
        for side in active:
            top, bottom, col = expected_gates[side]
            opened[top:bottom + 1, col] = True
        terrain = ((grid == solid) | (grid == plug) | cap_mask) & ~opened
        if not all(np.array_equal(_pour(terrain, moving, reverse, order), target)
                   for reverse in (False, True) for order in (False, True)):
            continue
        out = grid.copy()
        out[moving | opened] = background
        out[target] = liquid
        # Reflection changes SW/SE geometry while preserving downward gravity.
        if rng.integers(2):
            grid, out = grid[:, ::-1].copy(), out[:, ::-1].copy()
            cap_mask, opened = cap_mask[:, ::-1].copy(), opened[:, ::-1].copy()
        try:
            analysis = _analyze(grid)
        except ValueError:
            continue
        if not np.array_equal(analysis["opened"], opened) or not np.array_equal(analysis["caps"], cap_mask):
            continue
        return {"input": grid, "output": out}
    raise RuntimeError("bounded capacity/source construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not low <= int(value) <= high:
            raise ValueError("illegal pair count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    maximum_radius = int(rng.integers(2, 5))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            kind = ("witness_spill" if index == 0 else "witness_equal" if index == 1
                    else "retain" if index == 2 else "varied") if split == "train" else "varied"
            radius = maximum_radius if kind.startswith("witness") else int(rng.integers(2, maximum_radius + 1))
            for _ in range(30):
                pair = _make_pair(rng, radius, kind, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape and np.any(inp != out)
    a = _analyze(inp)
    water = (out == a["water"]) & ~a["caps"]
    assert int(water.sum()) == int(a["moving"].sum()) > 0, "volume changed"
    assert not np.any(water & a["terrain"]), "liquid entered a solid"
    fixed = ~(a["moving"] | a["opened"] | water)
    assert np.array_equal(inp[fixed], out[fixed]), "stationary material changed"
    assert np.array_equal(out[a["caps"]], inp[a["caps"]]), "a cap moved"
    assert np.all(np.isin(out[a["opened"]], (a["background"], a["water"])))
    assert np.all(out[(inp == a["plug"]) & ~a["opened"]] == a["plug"])
    surfaces = []
    for pool in _components(water):
        surface = int(np.argwhere(pool)[:, 0].min())
        surfaces.append(surface)
        closure = _component(~a["terrain"] & (np.arange(inp.shape[0])[:, None] >= surface),
                             tuple(map(int, np.argwhere(pool)[0])))
        assert np.array_equal(pool, closure), "incomplete level or leaking basin"
    for reverse in (False, True):
        for order in (False, True):
            assert np.array_equal(_pour(a["terrain"], a["moving"], reverse, order), water), "pour-order ambiguity"
    assert np.array_equal(solve(inp), out)
    # Ablation A: opening sluices alone does not settle the finite supply.
    gate_only = inp.copy()
    gate_only[a["opened"]] = a["background"]
    assert not np.array_equal(gate_only, out), "no demonstrated fluid motion"
    # Ablation B: keeping plugs changes hydraulic results in teaching spill /
    # equalization examples. Retention examples intentionally need no transfer.
    closed_terrain = (inp == a["solid"]) | (inp == a["plug"]) | a["caps"]
    closed_water = _pour(closed_terrain, a["moving"])
    liquid_needs_gate = not np.array_equal(closed_water, water)
    target_points = np.argwhere(water)
    source_points = np.argwhere(a["moving"])
    assert target_points[:, 0].sum() > source_points[:, 0].sum(), "liquid did not descend"
    return dict(shape=inp.shape, radius=a["emitter"]["radius"],
                sides=tuple(sorted(a["emitter"]["caps"])),
                gates=len(a["gates"]), opened=len(a["matched"]),
                unmatched=len(a["gates"]) - len(a["matched"]),
                surfaces=tuple(sorted(surfaces)), pools=len(surfaces),
                volume=int(water.sum()), liquid_needs_gate=liquid_needs_gate,
                roles=(a["background"], a["solid"], a["water"], a["plug"]),
                retained=int((water & a["moving"]).sum()),
                floor_shape=tuple(int(np.count_nonzero(inp[:, c] == a["solid"]))
                                  for c in range(inp.shape[1])),
                a=a)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_stats, test_stats = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            stats = _inspect_pair(pair)
            (train_stats if split == "train" else test_stats).append(stats)
    # Teaching obligations, checked from returned pixels rather than metadata.
    assert any(s["unmatched"] and s["liquid_needs_gate"] and len(set(s["surfaces"])) > 1
               for s in train_stats), "missing spill/decoy witness"
    assert any(s["opened"] == 2 and s["pools"] == 1 for s in train_stats), "missing communicating-vessel witness"
    assert {-1, 1} <= {side for s in train_stats for side in s["sides"]}
    assert max(s["radius"] for s in train_stats) >= max(s["radius"] for s in test_stats)
    assert all(s["liquid_needs_gate"] for s in test_stats), "test must require hydraulic transfer"


def self_test():
    """Bounded 42-seed structural, hydraulic, ablation and mutation checks."""
    signatures, shapes, radii, volumes, roles, layouts = set(), set(), set(), set(), set(), set()
    records = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                stats = _inspect_pair(pair)
                records.append(stats)
                local.append(stats)
                shapes.add(stats["shape"])
                radii.add(stats["radius"])
                volumes.add(stats["volume"])
                roles.add(stats["roles"])
                a = stats["a"]
                normalized = np.full(pair["input"].shape, 0, dtype=np.int8)
                normalized[pair["input"] == a["solid"]] = 1
                normalized[pair["input"] == a["plug"]] = 2
                normalized[a["moving"]] = 3
                normalized[a["caps"]] = 4
                layouts.add(_key(normalized))
        assert len({s["shape"] for s in local}) >= 2
        assert len({s["roles"] for s in local}) >= 2
        assert any(s["unmatched"] for s in local) and any(s["opened"] == 2 for s in local)
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 40 and radii == {2, 3, 4}
    assert len(volumes) >= 40 and len(roles) >= 80
    assert {s["gates"] for s in records} == {1, 2}
    assert any(not s["liquid_needs_gate"] for s in records)
    assert any(s["sides"] == (-1,) for s in records)
    assert any(s["sides"] == (1,) for s in records)
    assert sum(s["retained"] > 0 for s in records) >= 100
    # Repeatability, every legal explicit count, and malformed count types.
    reference = generate_puzzle(0)
    again = generate_puzzle(0)
    assert all(_key(p[k]) == _key(q[k]) for s in ("train", "test")
               for p, q in zip(reference[s], again[s]) for k in ("input", "output"))
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * n + t, n, t)
            assert len(puzzle["train"]) == n and len(puzzle["test"]) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in reference[s]]
               for s in ("train", "test")}
    a = _analyze(damaged["train"][0]["input"])
    cell = tuple(map(int, np.argwhere((damaged["train"][0]["output"] == a["water"]) & ~a["caps"])[0]))
    damaged["train"][0]["output"][cell] = a["background"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted lost liquid")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal counts)")
