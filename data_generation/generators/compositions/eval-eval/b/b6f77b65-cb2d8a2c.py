"""Escape routes around the swept space of selectively released rigid rods.

Only horizontal shelves are fixed fixtures. Vertical striped rods are free
rigid pieces; gravity is always down, independently of the seed's heading.
No parent generator, training labels, or sampling state is used by solve.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _decode(value):
    a = _grid(value)
    h, w = a.shape
    assert h >= 6 and w >= 6
    bg = int(a[0, 0])
    assert np.all(a[0] == bg), "the top border identifies background"
    seeds = [(r, c) for c in (0, w - 1) for r in range(1, h - 1)
             if a[r, c] != bg]
    assert len(seeds) == 1, "one seed on a vertical boundary is required"
    seed_row, seed_col = seeds[0]
    reverse = seed_col == w - 1
    if reverse:
        a = a[:, ::-1].copy()
    route = int(a[seed_row, 0])
    horizontal = np.zeros(a.shape, dtype=bool)
    shelves = []
    for r in range(1, h):
        c = 1
        while c < w - 1:
            end = c + 1
            while end < w - 1 and a[r, end] == a[r, c]:
                end += 1
            if a[r, c] != bg and end - c >= 2:
                shelves.append((r, c, end - 1, int(a[r, c])))
                horizontal[r, c:end] = True
            c = end
    explained = horizontal.copy()
    explained[seed_row, 0] = True
    rods = []
    for c in range(1, w - 1):
        rows = np.flatnonzero((a[:, c] != bg) & ~horizontal[:, c])
        if not len(rows):
            continue
        top, bottom = int(rows[0]), int(rows[-1])
        assert len(rows) == bottom - top + 1 and top >= 1
        colors = set(map(int, a[top:bottom + 1, c]))
        assert len(colors) == 2 and route in colors
        body = (colors - {route}).pop()
        markers = tuple(map(int, np.flatnonzero(a[top:bottom + 1, c] == route)))
        n = len(markers)
        assert 1 <= n <= 4 and len(rows) >= 2 * n + 1
        low = markers == tuple(2 * j for j in range(n))
        high = markers == tuple(bottom - top - 2 * j for j in reversed(range(n)))
        assert low != high, "a rod must mark exactly one end"
        assert bottom == h - 1 or any(r == bottom + 1 and lo <= c <= hi
                                     for r, lo, hi, color in shelves)
        remaining = [r for r, lo, hi, color in shelves
                     if color != route and r > bottom and lo <= c <= hi]
        final_bottom = min(remaining) - 1 if remaining else h - 1
        assert final_bottom >= bottom
        rods.append(dict(col=c, top=top, bottom=bottom, final_bottom=final_bottom,
                         fall=final_bottom - bottom, body=body, n=n,
                         side=1 if low else -1))
        explained[top:bottom + 1, c] = True
    assert rods and np.array_equal(explained, a != bg)
    assert not np.any(a[:, -1] != bg)
    return a, bg, route, seed_row, reverse, shelves, rods


def _trace(shape, seed_row, rods, envelope="sweep", unit=False, wrong_side=False):
    """Trace in physical coordinates; alternative envelopes are audit probes."""
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    row, start = seed_row, 0
    events = []
    for rod in rods:
        top, bottom = rod["top"], rod["final_bottom"]
        if envelope == "initial":
            bottom = rod["bottom"]
        elif envelope == "final":
            top += rod["fall"]
        hit = top <= row <= bottom
        event = dict(rod, hit=hit, incoming=row, sweep_top=top, sweep_bottom=bottom)
        if hit:
            d = 1 if unit else rod["n"] + 1
            side = -rod["side"] if wrong_side else rod["side"]
            target = bottom + d if side == 1 else top - d
            turn = rod["col"] - d
            assert 0 <= target < h and start <= turn < rod["col"]
            for c in range(start, turn + 1):
                mask[row, c] = True
            step = 1 if target > row else -1
            for r in range(row, target + step, step):
                mask[r, turn] = True
            row, start = target, turn
            event.update(clearance=d, target=target, turn=turn)
        events.append(event)
    mask[row, start:] = True
    return mask, events


def _render_result(decoded):
    a, bg, route, seed_row, reverse, shelves, rods = decoded
    b = np.full(a.shape, bg, dtype=np.int8)
    for r, lo, hi, color in shelves:
        if color != route:
            b[r, lo:hi + 1] = color
    sweep = np.zeros(a.shape, dtype=bool)
    for rod in rods:
        c = rod["col"]
        b[rod["top"] + rod["fall"]:rod["final_bottom"] + 1, c] = rod["body"]
        sweep[rod["top"]:rod["final_bottom"] + 1, c] = True
    path, events = _trace(a.shape, seed_row, rods)
    assert not np.any(path & sweep), "escape route intersects a swept rod"
    assert not np.any(path & (b != bg)), "escape route intersects a fixed shelf"
    b[path] = route
    return (b[:, ::-1].copy() if reverse else b), path, events


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive deletion, settlement, swept obstacles, and routing from input only."""
    return _render_result(_decode(input_grid))[0]


def _options(h, current, n, side, motion, hit):
    options = []
    for length in range(2 * n + 1, min(12, h - 2) + 1):
        for top in range(1, h - length):
            bottom = top + length - 1
            for fall in range(0 if motion != "moving" else 1,
                              1 if motion == "still" else min(7, h - 1 - bottom) + 1):
                final = bottom + fall
                target = final + n + 1 if side == 1 else top - n - 1
                # A skipped rod needs a blank row between its sweep and the ray.
                intercepts = top <= current <= final
                if hit != intercepts:
                    continue
                if not hit and not (current < top - 1 or current > final + 2):
                    continue
                if hit and not 0 <= target < h:
                    continue
                options.append((top, bottom, fall))
    return options


def _make_pair(rng, ns, direction, sides=None, motions=None, hits=None, core=False):
    count = len(ns)
    for attempt in range(100):
        h = int(rng.integers(20 if core else max(12, 2 * max(ns) + 6), 31))
        first = ns[0] + 2
        gaps = [n + 4 for n in ns[1:]]
        minimum_width = first + sum(gaps) + 4
        assert minimum_width <= 30
        w = int(rng.integers(minimum_width, 31))
        allocation = rng.multinomial(w - minimum_width, np.full(count + 1, 1 / (count + 1)))
        cols = [first + int(allocation[0])]
        for i, gap in enumerate(gaps):
            cols.append(cols[-1] + gap + int(allocation[i + 1]))
        colors = list(map(int, rng.permutation(10)))
        bg, route = colors[:2]
        body_palette = colors[2:2 + int(rng.integers(1, 4))]
        shelf_palette = colors[5:5 + int(rng.integers(1, 4))]
        a = np.full((h, w), bg, dtype=np.int8)
        seed_row = int(rng.integers(3, h - 3))
        a[seed_row, 0] = route
        current = seed_row
        failed = False
        for i, (c, n) in enumerate(zip(cols, ns)):
            hit = True if hits is None else hits[i]
            motion = "any" if motions is None else motions[i]
            side_order = [int(rng.choice((-1, 1)))] if sides is None else [sides[i]]
            if sides is None:
                side_order += [-side_order[0]]
            candidates = []
            for side in side_order:
                candidates = _options(h, current, n, side, motion, hit)
                if candidates:
                    break
            if not candidates:
                failed = True
                break
            top, bottom, fall = candidates[int(rng.integers(len(candidates)))]
            final = bottom + fall
            body = int(rng.choice(body_palette))
            a[top:bottom + 1, c] = body
            for j in range(n):
                a[top + 2 * j if side == 1 else bottom - 2 * j, c] = route

            def shelf(r, color):
                left = int(rng.integers(0, min(n, 2) + 1))
                right = int(rng.integers(1, 3))
                a[r, c - left:c + right + 1] = color

            if fall:
                shelf(bottom + 1, route)
                # Extra erased ledges are real competing landing surfaces.
                if fall >= 3 and rng.random() < .65:
                    shelf(int(rng.integers(bottom + 2, final + 1)), route)
            if final < h - 1:
                shelf(final + 1, int(rng.choice(shelf_palette)))
            # Harmless matching shelves can disappear below a stable landing.
            if final + 3 < h and rng.random() < .35:
                shelf(int(rng.integers(final + 3, h)), route)
            if hit:
                current = final + n + 1 if side == 1 else top - n - 1
        if failed:
            continue
        if direction == "left":
            a = a[:, ::-1].copy()
        try:
            decoded = _decode(a)
            out, path, events = _render_result(decoded)
            assert len(events) == count and [e["hit"] for e in events] == (
                [True] * count if hits is None else hits)
            return {"input": a, "output": out}
        except AssertionError:
            continue
    raise AssertionError("bounded construction failed")


def _checked_integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_integer(seed, "seed", 0)
    num_train = _checked_integer(num_train, "num_train", 2, 10)
    num_test = _checked_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    maximum = int(rng.choice((2, 3)))
    shift = str(rng.choice(("reflection", "longer")))
    direction = str(rng.choice(("right", "left")))
    sign = int(rng.choice((-1, 1)))
    puzzle = {"train": [], "test": []}
    seen = set()

    def add(split, ns, d, **kwargs):
        for _ in range(30):
            pair = _make_pair(rng, ns, d, **kwargs)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise AssertionError("unique-input construction exhausted")

    # These two demonstrations suffice even for the legal 2/1 request.
    add("train", [1, maximum], direction, sides=[sign, -sign],
        motions=["moving", "moving"], core=True)
    add("train", [max(2, maximum - 1), 1, int(rng.integers(1, maximum + 1))],
        direction, sides=[-sign, sign, sign], motions=["still", "any", "any"],
        hits=[True, True, False], core=True)
    for _ in range(num_train - 2):
        count = int(rng.integers(1, 4 if shift == "longer" else 5))
        ns = list(map(int, rng.integers(1, maximum + 1, size=count)))
        d = direction if shift == "reflection" else str(rng.choice(("left", "right")))
        add("train", ns, d)
    for _ in range(num_test):
        count = 4 if shift == "longer" else int(rng.integers(1, 5))
        ns = list(map(int, rng.integers(1, maximum + 1, size=count)))
        d = ("left" if direction == "right" else "right") if shift == "reflection" else str(rng.choice(("left", "right")))
        hits = [True] * count
        if count >= 3 and rng.random() < .5:
            hits[-1] = False
        add("test", ns, d, hits=hits)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    decoded = _decode(pair["input"])
    a, bg, route, seed, reverse, shelves, rods = decoded
    expected, path, events = _render_result(decoded)
    assert np.array_equal(expected, np.asarray(pair["output"]))

    def differs(**kwargs):
        try:
            other, _ = _trace(a.shape, seed, rods, **kwargs)
            return not np.array_equal(path, other)
        except AssertionError:
            return True

    return dict(shape=a.shape, direction="left" if reverse else "right",
                background=bg, route=route, rod_count=len(rods),
                lengths=[r["bottom"] - r["top"] + 1 for r in rods],
                falls=[r["fall"] for r in rods],
                counts=[r["n"] for r in events if r["hit"]],
                sides=[r["side"] for r in events if r["hit"]],
                moving_sides=[r["side"] for r in events if r["hit"] and r["fall"]],
                skipped=sum(not r["hit"] for r in events),
                still=sum(r["fall"] == 0 for r in rods),
                sweep_only=sum(r["hit"] and r["incoming"] > r["bottom"] for r in events),
                floor=sum(r["final_bottom"] == a.shape[0] - 1 for r in rods),
                erased=sum(color == route for _, _, _, color in shelves),
                shelf_lengths=[hi - lo + 1 for _, lo, hi, _ in shelves],
                initial_diff=differs(envelope="initial"),
                final_diff=differs(envelope="final"),
                unit_diff=differs(unit=True), side_diff=differs(wrong_side=True))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            record = _features(pair)
            records[split].append(record)
            decoded = _decode(a)
            canonical, bg, route, seed, reverse, shelves, rods = decoded
            out = b[:, ::-1] if reverse else b
            assert out[seed, 0] == route
            for r, lo, hi, color in shelves:
                if color != route:
                    assert np.all(out[r, lo:hi + 1] == color)
            for rod in rods:
                t, bottom, c = rod["top"] + rod["fall"], rod["final_bottom"], rod["col"]
                assert bottom - t == rod["bottom"] - rod["top"]
                assert np.all(out[t:bottom + 1, c] == rod["body"])
                assert bottom == a.shape[0] - 1 or any(
                    r == bottom + 1 and lo <= c <= hi and color != route
                    for r, lo, hi, color in shelves)
    train = records["train"]
    assert {s for f in train for s in f["moving_sides"]} == {-1, 1}
    assert {1, 2} <= {n for f in train for n in f["counts"]}
    assert any(f["still"] for f in train) and any(f["skipped"] for f in train)
    assert any(f["initial_diff"] for f in train) and any(f["final_diff"] for f in train)
    assert any(f["unit_diff"] and f["side_diff"] for f in train)
    assert {n for f in records["test"] for n in f["counts"]} <= {n for f in train for n in f["counts"]}


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    records, shapes, structures, palettes = [], set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            original = pair["input"].copy()
            assert np.array_equal(solve(original), pair["output"])
            assert np.array_equal(original, pair["input"])
            f = _features(pair)
            records.append(f)
            shapes.add(f["shape"])
            # Normalize literal colors while retaining all marker/support geometry.
            roles = {}
            normalized = np.array([roles.setdefault(int(c), len(roles))
                                   for c in original.flat], dtype=np.int8).reshape(original.shape)
            structures.add(_key(normalized))
            palettes.add((f["background"], f["route"]))
            # Horizontal reflection keeps downward gravity and reverses routing.
            assert np.array_equal(solve(original[:, ::-1]), pair["output"][:, ::-1])
            # Any one-to-one palette permutation preserves the complete answer.
            perm = np.array([7, 2, 9, 0, 5, 1, 8, 4, 6, 3], dtype=np.int8)
            assert np.array_equal(solve(perm[original]), perm[pair["output"]])
    assert len(shapes) >= 45 and len(structures) >= 180 and len(palettes) >= 35
    assert {f["rod_count"] for f in records} == {1, 2, 3, 4}
    assert {f["direction"] for f in records} == {"left", "right"}
    assert {n for f in records for n in f["counts"]} == {1, 2, 3}
    assert len({d for f in records for d in f["falls"]}) >= 7
    assert {s for f in records for s in f["sides"]} == {-1, 1}
    assert any(f["sweep_only"] for f in records) and any(f["floor"] for f in records)
    assert any(a == b for f in records for a, b in zip(f["sides"], f["sides"][1:]))
    # Independently labeled diagnostic: moving rod, bottom markers, old-top bypass.
    a = np.zeros((16, 12), dtype=np.int8)
    a[7, 0] = 3
    a[5:10, 6] = 2
    a[9, 6] = 3
    a[10, 6:8] = 3
    a[13, 6:9] = 4
    expected = np.zeros_like(a)
    expected[8:13, 6] = 2
    expected[13, 6:9] = 4
    expected[7, :5] = 3
    expected[3:8, 4] = 3
    expected[3, 4:] = 3
    assert np.array_equal(solve(a), expected)
    # Count four is supported although routine puzzles teach counts 1..2 or 1..3.
    a = np.zeros((24, 13), dtype=np.int8)
    a[12, 0] = 6
    a[7:16, 7] = 9
    a[[7, 9, 11, 13], 7] = 6
    a[16, 7:9] = 6
    a[19, 7:10] = 4
    expected = np.zeros_like(a)
    expected[10:19, 7] = 9
    expected[19, 7:10] = 4
    expected[12, :3] = 6
    expected[12:24, 2] = 6
    expected[23, 2:] = 6
    assert np.array_equal(solve(a), expected)
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(920 + train * 10 + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in puzzles[0]}
    damaged["train"][0]["output"][0, 0] = (int(damaged["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len(shapes),
            "normalized_structures": len(structures), "palettes": len(palettes),
            "rod_counts": sorted({f["rod_count"] for f in records}),
            "fall_distances": sorted({d for f in records for d in f["falls"]}),
            "swept_only_encounters": sum(f["sweep_only"] for f in records),
            "floor_landings": sum(f["floor"] for f in records),
            "directions": dict(Counter(f["direction"] for f in records))}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
