"""Collision clock: cyclic ink commands demolition in a changing scaffold.

Only NumPy and the standard library are used. No files, examples, seed state,
or generation parameters are consulted by solve(). See task_spec.md.
"""
from __future__ import annotations

import json

import numpy as np


# A stick is (orientation, top row, left column, length, color).
def _cells(stick):
    orientation, row, col, length, _ = stick
    return frozenset((row + (k if orientation == "V" else 0),
                      col + (k if orientation == "H" else 0))
                     for k in range(length))


def _shift(stick, distance):
    return stick[0], stick[1] + distance, stick[2], stick[3], stick[4]


def _settle(sticks, height):
    """Greatest simultaneously movable set, followed by one-cell translation."""
    result = list(sticks)
    for _ in range(height + 1):
        occupied = [_cells(p) for p in result]
        targets = [{(r + 1, c) for r, c in cells} for cells in occupied]
        moving = {i for i, cells in enumerate(targets)
                  if all(r < height for r, c in cells)}
        while True:
            blocked = {i for i in moving for j in range(len(result))
                       if j not in moving and targets[i] & occupied[j]}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_shift(p, 1) if i in moving else p
                  for i, p in enumerate(result)]
    raise AssertionError("gravity failed to converge")


def _settle_direct(sticks, height):
    """Authored cross-check: settle lowest pieces by their full free distance."""
    result = list(sticks)
    for _ in range(height * (len(sticks) + 1)):
        changed = False
        order = sorted(range(len(result)),
                       key=lambda i: max(r for r, c in _cells(result[i])),
                       reverse=True)
        for i in order:
            others = set().union(*[_cells(q) for j, q in enumerate(result) if j != i])
            distance = 0
            for d in range(1, height + 1):
                target = _cells(_shift(result[i], d))
                if any(r >= height for r, c in target) or target & others:
                    break
                distance = d
            if distance:
                result[i] = _shift(result[i], distance)
                changed = True
        if not changed:
            return result
    raise AssertionError("direct gravity failed to converge")


def _components(grid, background):
    seen = set()
    result = []
    for r, c in np.argwhere(grid != background):
        p = int(r), int(c)
        if p in seen:
            continue
        color = int(grid[p])
        stack, cells = [p], set()
        seen.add(p)
        while stack:
            a, b = stack.pop()
            cells.add((a, b))
            for q in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if (0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1]
                        and q not in seen and int(grid[q]) == color):
                    seen.add(q)
                    stack.append(q)
        result.append((color, cells))
    return result


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _decode(value):
    grid = _grid(value)
    h, w = grid.shape
    assert h >= 7
    bg = int(grid[2, 0])
    assert np.all(grid[2:4] == bg)
    assert np.array_equal(grid[0], grid[1])
    word = []
    c = 0
    while c < w:
        if grid[0, c] == bg:
            c += 1
            continue
        color = int(grid[0, c])
        assert c + 1 < w and grid[0, c + 1] == color
        assert c + 2 == w or grid[0, c + 2] == bg
        word.append(color)
        c += 2
    assert 2 <= len(word) <= 6 and len(set(word)) >= 2
    starts = [(r, 0) for r in range(4, h) if grid[r, 0] != bg]
    assert len(starts) == 1
    start = starts[0]
    assert int(grid[start]) == word[0] and grid[start[0], 1] == bg
    scene = grid.copy()
    scene[:4] = bg
    scene[start] = bg
    sticks = []
    for color, cells in _components(scene, bg):
        rows = {r for r, c in cells}
        cols = {c for r, c in cells}
        assert len(cells) >= 2 and (len(rows) == 1 or len(cols) == 1)
        r, c = min(rows), min(cols)
        orientation = "H" if len(rows) == 1 else "V"
        stick = orientation, r, c, len(cells), color
        assert _cells(stick) == cells and c >= 1
        sticks.append(stick)
    assert sticks
    sticks.sort(key=lambda p: (p[1], p[2], p[0], p[4]))
    return dict(grid=grid, background=bg, word=tuple(word), start=start,
                sticks=sticks)


def _render(info, sticks, path=()):
    grid = np.full(info["grid"].shape, info["background"], dtype=np.int8)
    grid[:4] = info["grid"][:4]
    for p, color in path:
        grid[p] = color
    occupied = set()
    for stick in sticks:
        cells = _cells(stick)
        assert not cells & occupied
        occupied |= cells
        for p in cells:
            grid[p] = stick[4]
    return grid


def _execute(info, alternative=None, cross_check=False):
    """Input-derived execution. Alternative models are diagnostics only.

    A cursor turns AFTER commanding deletion and completing the ensuing fall.
    Its ink is an annotation, never a physical support. Valid scenes keep the
    accumulated ink clear after every complete settling event.
    """
    word = info["word"]
    if alternative == "deduplicate":
        word = tuple(dict.fromkeys(word))
    h, w = info["grid"].shape
    active = dict(enumerate(info["sticks"]))
    initial_occupancy = {p: i for i, s in active.items() for p in _cells(s)}
    cursor, phase, direction = info["start"], 1, 0
    path = [(cursor, word[0])]
    legs, events, clear = [1], [], True
    directions = ((0, 1), (1, 0))
    for _ in range(2 * (h + w) + 2):
        occupancy = {p: i for i, s in active.items() for p in _cells(s)}
        obstacles = initial_occupancy if alternative == "frozen_geometry" else occupancy
        dr, dc = directions[direction]
        ahead = cursor[0] + dr, cursor[1] + dc
        if not (0 <= ahead[0] < h and 0 <= ahead[1] < w):
            terminal = "edge"
            break
        if ahead not in obstacles:
            cursor = ahead
            path.append((cursor, word[phase % len(word)]))
            phase += 1
            legs[-1] += 1
            continue
        hit = obstacles[ahead]
        contact_color = info["sticks"][hit][4]
        command = word[phase % len(word)]
        if alternative == "contact_color":
            command = contact_color
        deleted = [i for i, p in active.items() if p[4] == command]
        if alternative == "local_deletion":
            deleted = [i for i in deleted if i == hit]
        if alternative == "no_deletion":
            deleted = []
        active = {i: p for i, p in active.items() if i not in deleted}
        ids = sorted(active)
        before = [active[i] for i in ids]
        after = before if alternative == "no_gravity" else _settle(before, h)
        if cross_check:
            assert after == _settle_direct(before, h)
        falls = {i: q[1] - p[1] for i, p, q in zip(ids, before, after)}
        active = dict(zip(ids, after))
        occupied_after = set().union(*[_cells(p) for p in after])
        clear = clear and not bool(occupied_after & {p for p, color in path})
        events.append(dict(position=ahead, command=command, contact=contact_color,
                           deleted=tuple(deleted), falls=falls))
        direction = 1 - direction
        if alternative == "reset_bends":
            phase = 0
        elif alternative == "consume_collision":
            phase += 1
        dr, dc = directions[direction]
        turn = cursor[0] + dr, cursor[1] + dc
        new_obstacles = (initial_occupancy if alternative == "frozen_geometry"
                         else {p for s in active.values() for p in _cells(s)})
        if not (0 <= turn[0] < h and 0 <= turn[1] < w) or turn in new_obstacles:
            terminal = "blocked_turn"
            break
        legs.append(0)
    else:
        raise AssertionError("monotone cursor exceeded its distance bound")
    output = _render(info, list(active.values()), path)
    return dict(output=output, path=path, legs=tuple(legs), events=events,
                terminal=terminal, clear=clear, remaining=active)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only this input's visible legend, cursor and rigid sticks."""
    return _execute(_decode(input_grid))["output"]


_ALTERNATIVES = ("deduplicate", "reset_bends", "consume_collision", "contact_color",
                 "local_deletion", "no_gravity", "frozen_geometry", "no_deletion")


def _coverage(info, result):
    evidence = {name for name in _ALTERNATIVES
                if not np.array_equal(result["output"],
                                      _execute(info, name)["output"])}
    if len(result["path"]) >= 2 * len(info["word"]):
        evidence.add("cycling")
    if any(len(e["deleted"]) >= 2 for e in result["events"]):
        evidence.add("global_command")
    if any(e["deleted"] and e["command"] != e["contact"] for e in result["events"]):
        evidence.add("remote_command")
    if any(sum(d > 0 for d in e["falls"].values()) >= 2 for e in result["events"]):
        evidence.add("cascade")
    static = _execute(info, "frozen_geometry")
    if [p for p, c in result["path"]] != [p for p, c in static["path"]]:
        evidence.add("route_feedback")
    return evidence


_REQUIRED = set(_ALTERNATIVES) | {"cycling", "global_command", "remote_command",
                                       "cascade", "route_feedback"}


def _encode(shape, background, word, start, sticks, rng):
    h, w = shape
    grid = np.full(shape, background, dtype=np.int8)
    slack = w - (3 * len(word) - 1)
    assert slack >= 0
    left = int(rng.integers(0, slack + 1))
    gaps = [1] * (len(word) - 1)
    spare = slack - left
    for i in rng.permutation(len(gaps)):
        add = int(rng.integers(0, min(2, spare) + 1))
        gaps[int(i)] += add
        spare -= add
    c = left
    for i, color in enumerate(word):
        grid[:2, c:c + 2] = color
        if i < len(gaps):
            c += 2 + gaps[i]
    occupied = set()
    for stick in sticks:
        cells = _cells(stick)
        assert not cells & occupied and start not in cells
        assert all(4 <= r < h and 1 <= c < w for r, c in cells)
        occupied |= cells
        for p in cells:
            grid[p] = stick[4]
    grid[start] = word[0]
    info = _decode(grid)
    assert set(info["sticks"]) == set(sticks)
    return info


def _pattern(word):
    roles = {}
    return tuple(roles.setdefault(c, len(roles)) for c in word)


def _word(rng, palette, minimum=2, maximum=6):
    n = int(rng.integers(minimum, maximum + 1))
    distinct = int(rng.integers(2, min(n, len(palette)) + 1))
    symbols = list(map(int, rng.choice(palette, distinct, replace=False)))
    result = symbols + list(map(int, rng.choice(symbols, n - distinct)))
    rng.shuffle(result)
    return tuple(result)


def _witness(rng, large=False):
    """A randomized interacting bridge, carrying all essential counterexamples.

    The first command removes a tall post AND a redundant shorter post.
    An overhanging shelf falls into the cursor's future leg. Its collision
    samples a different color, removing the last support of the lower shelf.
    Both shelves then fall together. This is a single changing construction.
    """
    for attempt in range(160):
        palette = tuple(map(int, rng.permutation(10)))
        bg, inactive = palette[:2]
        word = _word(rng, palette[2:6], 4, 6)
        if len(set(word)) < 3 or len(set(word)) == len(word):
            continue
        h = int(rng.integers(24 if large else 19, 31))
        w = int(rng.integers(max(20, 3 * len(word) - 1), 31))
        choices = [length for length in range(5, min(15, w - 9))
                   if word[length % len(word)] != word[(length + 1) % len(word)]]
        if not choices:
            continue
        length = int(rng.choice(choices))
        a, c = word[length % len(word)], word[(length + 1) % len(word)]
        b = int(rng.choice([x for x in set(word) if x not in (a, c)]))
        row = int(rng.integers(8, h - 7))
        left = int(rng.integers(1, length - 1))
        right = int(rng.integers(length + 2, min(length + 6, w - 5)))
        lower_left = int(rng.integers(length + 1, right + 1))
        lower_right = int(rng.integers(right + 4, w))
        support_c = int(rng.integers(right + 1, lower_right - 1))
        support_a = int(rng.integers(support_c + 2, lower_right + 1))
        sticks = [("V", row - 1, length, h - row + 1, a),
                  ("H", row - 2, left, right - left + 1, b),
                  ("H", row + 3, lower_left, lower_right - lower_left + 1, inactive),
                  ("V", row + 4, support_c, h - row - 4, c),
                  ("V", row + 4, support_a, h - row - 4, a)]
        # Cargo and extra supports vary geometry and support depth, not the rule.
        if rng.random() < (.9 if large else .5):
            cargo_n = int(rng.integers(2, min(5, row - 6) + 1))
            cargo_x = int(rng.integers(length + 1, right + 1))
            cargo_color = int(rng.choice([x for x in palette[2:] if x not in (a, c, b)]))
            sticks.append(("V", row - 2 - cargo_n, cargo_x, cargo_n, cargo_color))
        if large or rng.random() < .3:
            extra_x = int(rng.integers(lower_left, right + 1))
            if all(p[0] != "V" or abs(p[2] - extra_x) >= 2 for p in sticks):
                sticks.append(("V", row + 4, extra_x, h - row - 4, c))
        try:
            info = _encode((h, w), bg, word, (row, 0), sticks, rng)
            assert _settle(info["sticks"], h) == info["sticks"]
            result = _execute(info)
            assert result["clear"]
            if not _REQUIRED <= _coverage(info, result):
                continue
        except AssertionError:
            continue
        return {"input": info["grid"], "output": result["output"]}
    raise AssertionError("bounded witness construction exhausted")


def _drop(sticks, orientation, col, length, height):
    candidate = orientation, 4, col, length, 0
    occupied = set().union(*[_cells(p) for p in sticks])
    if any(r >= height for r, c in _cells(candidate)) or _cells(candidate) & occupied:
        return
    for d in range(height):
        next_cells = _cells(_shift(candidate, d + 1))
        if any(r >= height for r, c in next_cells) or next_cells & occupied:
            sticks.append(_shift(candidate, d))
            return


def _random_pair(rng, large=False, require_fall=False):
    """Dropped sticks form arbitrary acyclic support networks, without slots."""
    for attempt in range(150):
        h = int(rng.integers(21 if large else 13, 31))
        w = int(rng.integers(19 if large else 12, 31))
        palette = tuple(map(int, rng.permutation(10)))
        bg = palette[0]
        word = _word(rng, palette[1:6], 2, min(6, (w + 1) // 3))
        sticks = []
        target = int(rng.integers(8 if large else 3, 18 if large else 13))
        for _ in range(target * 3):
            if len(sticks) >= target:
                break
            orientation = "H" if rng.random() < .55 else "V"
            n = int(rng.integers(2, min(w - 2, 18) + 1)) if orientation == "H" else int(rng.integers(2, min(h - 5, 14) + 1))
            col = int(rng.integers(1, w - n + 1)) if orientation == "H" else int(rng.integers(1, w))
            _drop(sticks, orientation, col, n, h)
        if len(sticks) < 3:
            continue
        # Different colors at initial contacts make the rigid units recoverable.
        colored = []
        for p in sticks:
            cells = _cells(p)
            neighbors = cells | {(r + dr, c + dc) for r, c in cells
                                 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
            forbidden = {q[4] for q in colored if _cells(q) & neighbors}
            choices = [x for x in palette[1:8] if x not in forbidden]
            if not choices:
                break
            colored.append((*p[:4], int(rng.choice(choices))))
        if len(colored) != len(sticks):
            continue
        rows = rng.permutation(np.arange(4, h))
        for row_value in rows[:min(10, len(rows))]:
            row = int(row_value)
            if any((row, 1) in _cells(p) for p in colored):
                continue
            try:
                info = _encode((h, w), bg, word, (row, 0), colored, rng)
                result = _execute(info)
            except AssertionError:
                continue
            events = result["events"]
            if (not result["clear"] or not events
                    or not any(e["deleted"] for e in events)
                    or (require_fall and not any(any(d > 0 for d in e["falls"].values()) for e in events))):
                continue
            if large and (len(events) < 2 or len(result["remaining"]) < 2):
                continue
            return {"input": info["grid"], "output": result["output"]}
    # Same semantic rule; randomized bridges ensure bounded termination.
    return _witness(rng, large)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    result, used = {"train": [], "test": []}, set()
    witness_slot = int(rng.integers(num_train))
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(12):
                if split == "train" and i == witness_slot:
                    pair = _witness(rng)
                else:
                    # Tests recombine deeper/denser networks; both kinds are
                    # already visible in training, including the bridge witness.
                    large = split == "test" or (i > 1 and rng.random() < .3)
                    pair = _random_pair(rng, large, require_fall=(split == "test"))
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    result[split].append(pair)
                    break
            else:
                raise AssertionError("distinct inputs exhausted")
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _decode(inp)
            assert _settle(info["sticks"], inp.shape[0]) == info["sticks"]
            result = _execute(info)
            assert result["clear"], "settled rods must not cover accumulated ink"
            assert np.array_equal(result["output"], out)
            assert np.array_equal(inp[:4], out[:4])
            assert not np.array_equal(inp, out)
            assert len({p for p, color in result["path"]}) == len(result["path"])
            assert sum(result["legs"]) == len(result["path"])
            assert len(result["path"]) <= sum(inp.shape)
            for event in result["events"]:
                assert all(d >= 0 for d in event["falls"].values())
                assert all(info["sticks"][i][4] == event["command"] for i in event["deleted"])
            assert all(p[0] == info["sticks"][i][0] and p[2:] == info["sticks"][i][2:]
                       for i, p in result["remaining"].items())
            assert _settle(list(result["remaining"].values()), inp.shape[0]) == list(result["remaining"].values())
            if split == "train":
                evidence |= _coverage(info, result)
    assert _REQUIRED <= evidence, sorted(_REQUIRED - evidence)


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """42 seeds, all legal counts, mutation checks, and bounded physics tests."""
    signatures, shapes, layouts, patterns, piece_counts, distances = set(), set(), set(), set(), set(), set()
    support_counts, backgrounds, terminals = set(), set(), set()
    max_events, moved, total = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                result = _execute(info, cross_check=True)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(info["grid"].shape)
                layouts.add((info["grid"].shape, (info["grid"][4:] != info["background"]).tobytes()))
                patterns.add(_pattern(info["word"]))
                piece_counts.add(len(info["sticks"]))
                backgrounds.add(info["background"])
                terminals.add(result["terminal"])
                max_events = max(max_events, len(result["events"]))
                for event in result["events"]:
                    distances.update(d for d in event["falls"].values() if d)
                    moved += sum(d > 0 for d in event["falls"].values())
                cells = [_cells(p) for p in info["sticks"]]
                for i, p in enumerate(info["sticks"]):
                    below = {(r + 1, c) for r, c in cells[i]}
                    support_counts.add(sum(bool(below & q) for j, q in enumerate(cells) if i != j))
                total += 1
    assert len(signatures) == 42 and total == 210
    assert len(shapes) >= 50 and len(layouts) >= 150
    assert len(patterns) >= 15 and len(piece_counts) >= 10
    assert len(backgrounds) == 10 and len(distances) >= 8
    assert max_events >= 3 and support_counts >= {0, 1, 2}
    assert terminals == {"edge", "blocked_turn"} and moved >= 100
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(72019, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3, 1.5), (None, 1), (3, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    # Two indispensable mechanics in literal, small, non-generated fixtures.
    rods = [("H", 5, 1, 7, 1), ("V", 6, 1, 4, 2),
            ("V", 6, 7, 4, 3), ("V", 8, 4, 2, 4)]
    assert _settle(rods, 10) == rods
    assert _settle([rods[0], rods[2], rods[3]], 10)[0] == rods[0]
    assert _settle([rods[0], rods[3]], 10)[0][1] == 7
    assert _settle([rods[0]], 10)[0][1] == 9
    # An unobstructed canvas exit must not delete a color or turn downward.
    grid = np.zeros((10, 10), dtype=np.int8)
    grid[:2, 1:3], grid[:2, 4:6] = 1, 2
    grid[4, 0], grid[9, 7:10] = 1, 2
    result = _execute(_decode(grid))
    assert not result["events"] and result["terminal"] == "edge"
    assert [p for p, c in result["path"]] == [(4, c) for c in range(10)]
    return {"seeds": 42, "pairs": total, "shapes": len(shapes),
            "layouts": len(layouts), "word_patterns": len(patterns),
            "piece_counts": sorted(piece_counts), "fall_distances": sorted(distances),
            "max_collisions": max_events, "moving_stick_events": moved}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
