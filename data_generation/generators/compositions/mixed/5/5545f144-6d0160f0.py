"""A terminal mover tip addresses a rigidly transported 3x3 payload panel.

Motion colors are fixed: body 1, tip 2, waypoints 3. Background, lattice,
and inert data colors are read from each input. No files or parent modules
are needed by generation, solving, validation, or self_test.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json

import numpy as np


BODY, TIP, WAYPOINT = 1, 2, 3
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
POSITIONS = tuple((r, c) for r in range(3) for c in range(3))
DATA_COLORS = (0, 4, 5, 6, 7, 8, 9)
# Body offsets relative to a right-facing tip. The tip itself is (0,0).
MASKS = (
    ((0, -1),),
    ((0, -1), (1, -1)),
    ((-1, -1), (0, -1)),
    ((-1, -1), (1, -1)),
    ((-1, -1), (0, -1), (1, -1)),
)


def _rotate(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _body(mask, tip, direction):
    return {(tip[0] + dr, tip[1] + dc)
            for dr, dc in (_rotate(p, direction - 1) for p in mask)}


def _footprint(mask, tip, direction):
    return _body(mask, tip, direction) | {tip}


def _inside(cells):
    return all(0 <= r < 3 and 0 <= c < 3 for r, c in cells)


def _direction(a, b):
    if a[0] == b[0] and a != b:
        return 1 if b[1] > a[1] else 3
    if a[1] == b[1] and a != b:
        return 2 if b[0] > a[0] else 0
    return None


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _panel(position):
    r, c = position
    return slice(4*r, 4*r+3), slice(4*c, 4*c+3)


def _lattice_mask():
    mask = np.zeros((11, 11), dtype=bool)
    mask[[3, 7], :] = True
    mask[:, [3, 7]] = True
    return mask


def _blank(background, divider):
    grid = np.full((11, 11), background, dtype=np.int8)
    grid[_lattice_mask()] = divider
    return grid


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and grid.shape == (11, 11), "expected an 11x11 lattice"
    assert np.issubdtype(grid.dtype, np.integer), "integer cells required"
    assert np.all((0 <= grid) & (grid <= 9)), "colors must be in 0..9"
    return grid


def _read_input(input_grid):
    """Decode and check the visible motion, without consulting the catalog."""
    grid = _grid(input_grid)
    lattice = _lattice_mask()
    dividers = np.unique(grid[lattice])
    assert len(dividers) == 1, "the lattice must have one color"
    divider = int(dividers[0])
    assert divider in DATA_COLORS and not np.any(grid[~lattice] == divider)
    counts = np.bincount(grid[~lattice].astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert background in DATA_COLORS and counts[background] > 40, "background must be a strict panel-cell majority"
    assert background != divider
    tips = np.argwhere(grid == TIP)
    assert tips.shape == (1, 2), "one visibly marked tip is required"
    r, c = map(int, tips[0])
    source = (r // 4, c // 4)
    tip = (r % 4, c % 4)
    payload = grid[_panel(source)].copy()
    assert tip in POSITIONS
    for color in (BODY, WAYPOINT):
        assert np.count_nonzero(grid == color) == np.count_nonzero(payload == color), "all motion stays in the payload"
    body = set(map(tuple, np.argwhere(payload == BODY).tolist()))
    matches = [(mask, d) for mask in MASKS for d in range(4)
               if _body(mask, tip, d) == body]
    assert len(matches) == 1, "the marked mover must have a unique facing"
    mask, initial = matches[0]
    waypoints = set(map(tuple, np.argwhere(payload == WAYPOINT).tolist()))
    assert 1 <= len(waypoints) <= 4
    cargo = set(map(tuple, np.argwhere(~np.isin(payload, (background, BODY, TIP, WAYPOINT))).tolist()))
    assert cargo, "a complete payload includes inert data"

    points, directions, footprints = [tip], [], [_footprint(mask, tip, initial)]
    remaining = set(waypoints)
    while remaining:
        rays = {}
        for q in sorted(remaining):
            d = _direction(points[-1], q)
            if d is not None:
                rays.setdefault(d, []).append(q)
        assert len(rays) == 1, "exactly one orthogonal ray may contain unvisited waypoints"
        d, candidates = next(iter(rays.items()))
        if not directions:
            assert d == initial, "the first leg follows the visible tip"
        q = min(candidates, key=lambda p: _distance(points[-1], p))
        cells = _footprint(mask, q, d)
        assert _inside(cells), "the whole arriving mover must fit"
        points.append(q)
        directions.append(d)
        footprints.append(cells)
        remaining.remove(q)
    for i, cells in enumerate(footprints):
        assert not cells.intersection(points[i+1:]), "future markers must remain visible"
        assert not cells.intersection(cargo), "inert cargo cannot be overwritten at a stop"
    settled = payload.copy()
    settled[np.isin(settled, (BODY, TIP, WAYPOINT))] = background
    for cell in _body(mask, points[-1], directions[-1]):
        settled[cell] = BODY
    settled[points[-1]] = TIP
    return dict(background=background, divider=divider, source=source,
                payload=payload, mask=mask, initial=initial, points=tuple(points),
                directions=tuple(directions), cargo=tuple(sorted(cargo)),
                settled=settled, destination=points[-1])


def _place(info, destination=None, payload=None):
    output = _blank(info["background"], info["divider"])
    output[_panel(info["destination"] if destination is None else destination)] = (
        info["settled"] if payload is None else payload)
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Settle the payload's journey, clear interiors, then place the whole panel."""
    return _place(_read_input(input_grid))


@lru_cache(maxsize=1)
def _route_catalog():
    """Exhaust the tiny geometry; never store or load a path/template file.

    An appended future point cannot introduce a competing ray at any past tip,
    cannot cover a previous mover footprint, and must leave room for cargo.
    The catalog has 156 paths, not a randomly selected list of route scripts.
    """
    records = []

    def extend(mi, points, directions, occupied, initial):
        if directions:
            records.append((mi, tuple(points), tuple(directions),
                            tuple(p for p in POSITIONS if p not in occupied)))
        if len(directions) == 4:
            return
        for q in POSITIONS:
            d = _direction(points[-1], q)
            if d is None or q in occupied or (not directions and d != initial):
                continue
            # Future q must be beyond the nearest next stop on the SAME ray
            # at each earlier tip, or be unaligned with that earlier tip.
            if any(_direction(p, q) is not None and
                   (_direction(p, q) != _direction(p, nxt) or
                    _distance(p, q) <= _distance(p, nxt))
                   for p, nxt in zip(points, points[1:])):
                continue
            cells = _footprint(MASKS[mi], q, d)
            if not _inside(cells):
                continue
            used = occupied | cells
            if len(used) <= 8:
                extend(mi, points + [q], directions + [d], used, initial)

    for mi, mask in enumerate(MASKS):
        for initial in range(4):
            for tip in POSITIONS:
                footprint = _footprint(mask, tip, initial)
                if _inside(footprint):
                    extend(mi, [tip], [], footprint, initial)
    return tuple(records)


def _motion_key(source, record):
    mi, points, ds, _ = record
    return (source, tuple(sorted(_body(MASKS[mi], points[0], ds[0]))),
            points[0], tuple(sorted(points[1:])))


def _choose(rng, choices):
    assert choices, "construction has no compatible candidate"
    return choices[int(rng.integers(len(choices)))]


def _plan(rng, num_train, num_test):
    catalog = _route_catalog()
    source_order = [POSITIONS[int(i)] for i in rng.permutation(9)]
    held_out, train_sources = source_order[0], source_order[1:]
    # This witness defeats endpoint guesses based on distance or row-major
    # extrema. In a three-cell-wide arena it requires four waypoints.
    first = _choose(rng, [r for r in catalog
                         if _distance(r[1][0], r[1][-1]) <
                         max(_distance(r[1][0], q) for q in r[1][1:])
                         and r[1][-1] not in (min(r[1][1:]), max(r[1][1:]))])
    second = _choose(rng, [r for r in catalog if len(r[2]) <= 2 and r[0] != first[0]
                           and r[1][-1][0] != first[1][-1][0]
                           and r[1][-1][1] != first[1][-1][1]
                           and any(t[0] != t[1] for t in (r[1][-1], first[1][-1]))])
    if train_sources[0] == first[1][-1] and train_sources[1] == second[1][-1]:
        train_sources[0], train_sources[1] = train_sources[1], train_sources[0]
    train = [(train_sources[0], first), (train_sources[1], second)]
    used = {_motion_key(s, r) for s, r in train}
    for i in range(2, num_train):
        source = train_sources[i % 8]
        rows = {r[1][-1][0] for _, r in train}
        cols = {r[1][-1][1] for _, r in train}
        options = [r for r in catalog if _motion_key(source, r) not in used]
        # Fill missing address coordinates before adding redundant combinations.
        scores = [int(r[1][-1][0] not in rows) + int(r[1][-1][1] not in cols)
                  for r in options]
        best = max(scores)
        options = [r for r, score in zip(options, scores) if score == best]
        if i == 2:
            longer = [r for r in options if len(r[2]) >= 3]
            options = longer or options
        record = _choose(rng, options)
        train.append((source, record))
        used.add(_motion_key(source, record))
    rows = {r[1][-1][0] for _, r in train}
    cols = {r[1][-1][1] for _, r in train}
    seen_addresses = {r[1][-1] for _, r in train}
    taught_collinear = any(any(a == b for a, b in zip(r[2], r[2][1:])) for _, r in train)
    test, test_addresses = [], set()
    for i in range(num_test):
        source = held_out if i == 0 else _choose(rng, list(POSITIONS))
        options = [r for r in catalog if r[1][-1][0] in rows and r[1][-1][1] in cols
                   and _motion_key(source, r) not in used]
        if not taught_collinear:
            options = [r for r in options if all(a != b for a, b in zip(r[2], r[2][1:]))]
        distinct = [r for r in options if r[1][-1] not in test_addresses]
        options = distinct or options
        if i == 0:
            novel = [r for r in options if r[1][-1] not in seen_addresses]
            options = novel or options
            longer = [r for r in options if len(r[2]) >= 3]
            if rng.random() < 0.60:
                options = longer or options
        record = _choose(rng, options)
        test.append((source, record))
        test_addresses.add(record[1][-1])
        used.add(_motion_key(source, record))
    # Pair order conveys no command or fixed source/address schedule.
    rng.shuffle(train)
    return train, test


def _draw_pair(rng, source, record):
    mi, points, directions, free = record
    mask = MASKS[mi]
    background, divider = [int(v) for v in rng.choice(DATA_COLORS, 2, replace=False)]
    available = [c for c in DATA_COLORS if c not in (background, divider)]
    palette = [int(v) for v in rng.choice(available, int(rng.integers(1, 6)), replace=False)]
    payload = np.full((3, 3), background, dtype=np.int8)
    for cell in _body(mask, points[0], directions[0]):
        payload[cell] = BODY
    payload[points[0]] = TIP
    for cell in points[1:]:
        payload[cell] = WAYPOINT
    cargo_count = int(rng.integers(1, len(free) + 1))
    cargo_cells = [free[int(i)] for i in rng.choice(len(free), cargo_count, replace=False)]
    for cell in cargo_cells:
        payload[cell] = _choose(rng, palette)
    grid = _blank(background, divider)
    grid[_panel(source)] = payload
    distractors = [(4*r+dr, 4*c+dc) for r, c in POSITIONS if (r, c) != source
                   for dr, dc in POSITIONS]
    payload_marks = int(np.count_nonzero(payload != background))
    count = int(rng.integers(6, min(32, 40-payload_marks) + 1))
    chosen = []
    if rng.random() < 0.50:
        focus = _choose(rng, [p for p in POSITIONS if p != source])
        focus_cells = [(4*focus[0]+r, 4*focus[1]+c) for r, c in POSITIONS]
        n = min(count, int(rng.integers(5, 10)))
        chosen = [focus_cells[int(i)] for i in rng.choice(9, n, replace=False)]
    remaining = [p for p in distractors if p not in chosen]
    chosen += [remaining[int(i)] for i in rng.choice(len(remaining), count-len(chosen), replace=False)]
    for cell in chosen:
        grid[cell] = _choose(rng, palette)
    # A cargo color also occurs outside the source: retaining an entire color
    # instead of identifying the addressed payload is observably incorrect.
    grid[chosen[0]] = payload[cargo_cells[0]]
    settled = payload.copy()
    settled[np.isin(settled, (BODY, TIP, WAYPOINT))] = background
    for cell in _body(mask, points[-1], directions[-1]):
        settled[cell] = BODY
    settled[points[-1]] = TIP
    output = _blank(background, divider)
    output[_panel(points[-1])] = settled
    return {"input": grid, "output": output}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    train, test = _plan(rng, num_train, num_test)
    puzzle = {split: [_draw_pair(rng, source, record) for source, record in plans]
              for split, plans in (("train", train), ("test", test))}
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _inspect_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    lattice = _lattice_mask()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read_input(inp)
            assert np.array_equal(out, _place(info)), "incorrect completed-panel relocation"
            assert np.array_equal(inp[lattice], out[lattice]), "lattice changed"
            assert np.count_nonzero(out == BODY) == len(info["mask"])
            assert np.count_nonzero(out == TIP) == 1 and not np.any(out == WAYPOINT)
            settled = out[_panel(info["destination"])]
            for cell in info["cargo"]:
                assert settled[cell] == info["payload"][cell], "cargo must keep its local position and color"
            assert any(np.any(inp[_panel(p)] != info["background"])
                       for p in POSITIONS if p != info["source"]), "distractors are required"
            assert not np.array_equal(inp, out)
            # A-ablation: the initial tip would give the wrong destination.
            assert not np.array_equal(out, _place(info, info["points"][0]))
            # A-ablation: copying the untouched initial panel is also wrong.
            assert not np.array_equal(out, _place(info, payload=info["payload"]))
            analyses[split].append(info)
    train, test = analyses["train"], analyses["test"]
    assert len({a["source"] for a in train}) >= 2
    assert len({a["destination"][0] for a in train}) >= 2
    assert len({a["destination"][1] for a in train}) >= 2
    assert len({a["mask"] for a in train}) >= 2
    assert len({len(a["directions"]) for a in train}) >= 2
    assert any(len(a["directions"]) >= 2 and a["initial"] != a["directions"][-1] for a in train)
    assert any(_distance(a["points"][0], a["destination"]) <
               max(_distance(a["points"][0], q) for q in a["points"][1:])
               and a["destination"] not in (min(a["points"][1:]), max(a["points"][1:]))
               for a in train), "training must defeat distance and row-major endpoint guesses"
    assert any(a["destination"][0] != a["destination"][1] for a in train), "distinguish row/column reversal"
    assert any(a["source"] != a["destination"] for a in train), "whole-panel relocation must matter"
    assert test[0]["source"] not in {a["source"] for a in train}, "held-out source shift missing"
    rows = {a["destination"][0] for a in train}
    cols = {a["destination"][1] for a in train}
    assert all(a["destination"][0] in rows and a["destination"][1] in cols for a in test)
    taught_collinear = any(any(x == y for x, y in zip(a["directions"], a["directions"][1:]))
                           for a in train)
    assert taught_collinear or all(all(x != y for x, y in zip(a["directions"], a["directions"][1:]))
                                  for a in test), "collinear-stop tests need a demonstration"
    return analyses


def validate_generated_puzzle(puzzle):
    """Check labels, actual rigid geometry, cargo, addressing, and teaching."""
    try:
        _inspect_puzzle(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def _normalized_route(points):
    variants = []
    for mirror in (1, -1):
        for k in range(4):
            variants.append(tuple(_rotate((r-points[0][0], mirror*(c-points[0][1])), k)
                                  for r, c in points))
    return min(variants)


def self_test():
    """Bounded 42-seed checks; returns compact, grid-derived diversity counts."""
    catalogs = _route_catalog()
    assert len(catalogs) == 156
    counts = Counter(len(r[2]) for r in catalogs)
    assert dict(counts) == {1: 36, 2: 48, 3: 40, 4: 32}
    catalog_rng = np.random.default_rng(9876)
    for i, record in enumerate(catalogs):
        pair = _draw_pair(catalog_rng, POSITIONS[i % 9], record)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    endpoints, sources, masks, paths, roles, cargo_counts, profiles = (set() for _ in range(7))
    keys, initial_dirs, final_dirs, distances, palette_roles = (set() for _ in range(5))
    ablation_a = ablation_b = same_source = mixed_turns = dense_decoys = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        keys.add(_puzzle_key(puzzle))
        analyses = _inspect_puzzle(puzzle)
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], analyses[split]):
                pairs += 1
                out = np.asarray(pair["output"])
                before = pair["input"].copy()
                assert np.array_equal(solve(before), out)
                assert np.array_equal(before, pair["input"]), "solve modified its input"
                endpoints.add(info["destination"])
                sources.add(info["source"])
                masks.add(info["mask"])
                paths.add(_normalized_route(info["points"]))
                roles.add((info["background"], info["divider"]))
                cargo_counts.add(len(info["cargo"]))
                profiles.add(len(info["directions"]))
                initial_dirs.add(info["initial"])
                final_dirs.add(info["directions"][-1])
                palette_roles.update((info["background"], info["divider"]))
                ds = info["directions"]
                turns = {(b-a) % 4 for a, b in zip(ds, ds[1:]) if a != b}
                mixed_turns += 1 in turns and 3 in turns
                distances.update(_distance(a, b) for a, b in zip(info["points"], info["points"][1:]))
                ablation_a += not np.array_equal(out, _place(info, info["points"][0]))
                ablation_b += not np.array_equal(out, _place(info, info["source"]))
                same_source += info["source"] == info["destination"]
                dense_decoys += any(np.count_nonzero(pair["input"][_panel(p)] != info["background"]) >= 5
                                    for p in POSITIONS if p != info["source"])
    assert len(keys) == 42 and sources == set(POSITIONS)
    assert endpoints == set(POSITIONS) - {(1, 1)}
    assert len(masks) == 5 and profiles == {1, 2, 3, 4}
    assert len(paths) == 10 and len(roles) >= 30 and len(cargo_counts) >= 2
    assert initial_dirs == final_dirs == set(range(4)) and distances == {1, 2}
    assert palette_roles == set(DATA_COLORS)
    assert ablation_a == pairs and ablation_b >= pairs//2
    assert min(same_source, mixed_turns, dense_decoys) >= 1
    # All 30 legal explicit count combinations, in memory one at a time.
    legal_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(1000 + 10*a + b, a, b)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
            legal_calls += 1
    assert legal_calls == 30
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {a!r}/{b!r}")
    ambiguous = _blank(0, 8)
    ambiguous[0, 0], ambiguous[0, 1] = BODY, TIP
    ambiguous[0, 2] = ambiguous[2, 1] = WAYPOINT
    ambiguous[2, 0] = 5
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted competing waypoint rays")
    # Corrupt a carried data cell and a preserved divider, respectively.
    puzzle = generate_puzzle(123)
    info = _read_input(puzzle["test"][0]["input"])
    for kind in ("cargo", "divider"):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        cell = (3, 0) if kind == "divider" else (
            4*info["destination"][0] + info["cargo"][0][0],
            4*info["destination"][1] + info["cargo"][0][1])
        broken["test"][0]["output"][cell] = (int(broken["test"][0]["output"][cell])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted changed {kind}")
    return dict(seeds=42, pairs=pairs, catalog_paths=len(catalogs),
                catalog_waypoint_counts=dict(sorted(counts.items())), legal_count_calls=legal_calls,
                source_slots=len(sources), destination_slots=len(endpoints), mover_masks=len(masks),
                normalized_routes=len(paths), background_divider_pairs=len(roles),
                cargo_counts=sorted(cargo_counts), waypoint_counts=sorted(profiles),
                initial_directions=sorted(initial_dirs), final_directions=sorted(final_dirs),
                leg_lengths=sorted(distances), a_ablation_changes=ablation_a,
                b_ablation_changes=ablation_b, stationary_panel_moves=same_source,
                mixed_turn_routes=mixed_turns, examples_with_dense_decoys=dense_decoys,
                independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
