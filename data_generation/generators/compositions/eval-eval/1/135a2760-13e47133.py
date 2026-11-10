"""Contour-dragged periodic weaves. Standard library and NumPy only.

No files, random state, demonstrations, or construction metadata enter solve().
Generation and all diagnostics are bounded; this module writes no artifacts.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("a grid must be an integer rectangle of colors 0..9, size 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        todo = [(r, c)]
        seen[r, c] = True
        for r, c in todo:
            for dr, dc in _D4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    todo.append((nr, nc))
        part = np.zeros(mask.shape, dtype=bool)
        rows, cols = zip(*todo)
        part[rows, cols] = True
        result.append(part)
    return result


def _depth(mask, diagonal=True):
    """Repeated simultaneous erosion; walls, holes and off-canvas are outside."""
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    h, w = mask.shape
    level = 0
    while np.any(current):
        padded = np.pad(current, 1, constant_values=False)
        interior = current.copy()
        for dr, dc in (_D8 if diagonal else _D4):
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~interior] = level
        current = interior
        level += 1
    return depth


def _bfs_depth(mask):
    """A second depth calculation for self-test, using boundary-source BFS."""
    h, w = mask.shape
    d = np.full(mask.shape, -1, dtype=np.int16)
    queue = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc])
               for dr, dc in _D8):
            d[r, c] = 0
            queue.append((r, c))
    for r, c in queue:
        for dr, dc in _D8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and d[nr, nc] < 0:
                d[nr, nc] = d[r, c] + 1
                queue.append((nr, nc))
    return d


def _positions(mask, side):
    rr, cc = np.nonzero(mask)
    if side == "left":
        t = cc - cc.min()
    elif side == "right":
        t = cc.max() - cc
    elif side == "top":
        t = rr - rr.min()
    else:
        t = rr.max() - rr
    return t.astype(int), int(t.max()) + 1


def _parse(a):
    if min(a.shape) < 5:
        raise ValueError("the family requires an outer frame and nonempty rooms")
    wall = int(a[0, 0])
    if any(int(a[r, c]) != wall for r, c in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("all four frame corners must agree")
    border = np.zeros(a.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    ports = np.argwhere(border & (a != wall))
    if len(ports) != 1:
        raise ValueError("exactly one frame cell is a launch marker")
    r, c = map(int, ports[0])
    marker = int(a[r, c])
    if np.any(a[~border] == marker):
        raise ValueError("the launch color is reserved for its frame cell")
    side = "top" if r == 0 else "bottom" if r == a.shape[0] - 1 else "left" if c == 0 else "right"
    walls = (a == wall) | border
    rooms = _components(~walls)
    if not rooms:
        raise ValueError("there must be an open room")
    return wall, marker, side, walls, rooms


def _recover(values, coordinate, span):
    """All admissible periodic explanations must agree on the complete repair."""
    colors = np.unique(values)
    if len(colors) != 2:
        raise ValueError("each room must visibly use exactly two weave colors")
    candidates = []
    budget = max(1, len(values) // 8)
    for p in range(2, min(6, span // 3) + 1):
        word = np.empty(p, dtype=np.int8)
        errors = 0
        for phase in range(p):
            sample = values[coordinate % p == phase]
            if len(sample) < 3:
                break
            n0 = int(np.count_nonzero(sample == colors[0]))
            n1 = len(sample) - n0
            if n0 == n1:
                break
            word[phase] = colors[int(n1 > n0)]
            errors += min(n0, n1)
        else:
            if errors <= budget and len(np.unique(word)) == 2:
                candidates.append((word[coordinate % p], word, errors))
    if not candidates:
        raise ValueError("no supported contour-dragged rhythm")
    prediction, word, errors = candidates[0]
    if any(not np.array_equal(prediction, x[0]) for x in candidates[1:]):
        raise ValueError("competing rhythms imply different repairs")
    return prediction, word, errors


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    _, _, side, _, rooms = _parse(a)
    out = a.copy()
    for room in rooms:
        t, span = _positions(room, side)
        u = t - _depth(room)[room]
        out[room] = _recover(a[room], u, span)[0]
    return out


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _partition(rng, h, w, target):
    """Compose a floor plan; retain one full-span room for long rhythms."""
    wall = np.zeros((h + 2, w + 2), dtype=bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    rects = [(1, h, 1, w)]
    while len(rects) < target:
        choices = []
        full = sum(c == 1 and d == w for a, b, c, d in rects)
        for i, (a, b, c, d) in enumerate(rects):
            if b - a + 1 >= 7:
                choices.append((i, 0))
            if d - c + 1 >= 19 and not (c == 1 and d == w and full == 1):
                choices.append((i, 1))
        if not choices:
            break
        i, axis = choices[int(rng.integers(len(choices)))]
        a, b, c, d = rects.pop(i)
        if axis == 0:
            cut = int(rng.integers(a + 3, b - 2))
            wall[cut, c - 1:d + 2] = True
            rects.extend(((a, cut - 1, c, d), (cut + 1, b, c, d)))
        else:
            cut = int(rng.integers(c + 9, d - 8))
            wall[a - 1:b + 2, cut] = True
            rects.extend(((a, b, c, cut - 1), (a, b, cut + 1, d)))
    return wall, rects


def _add_obstacle(rng, wall, rect, island):
    a, b, c, d = rect
    old_count = len(_components(~wall))
    for _ in range(18):
        proposal = wall.copy()
        if island:
            if b - a < 5 or d - c < 5:
                return False
            r = int(rng.integers(a + 2, b - 1))
            col = int(rng.integers(c + 2, d - 1))
            length = int(rng.integers(1, 5))
            if rng.integers(2):
                proposal[r:min(b - 1, r + length), col] = True
            else:
                proposal[r, col:min(d - 1, col + length)] = True
            new = proposal & ~wall
            padded = np.pad(wall, 1)
            touching = np.zeros(wall.shape, dtype=bool)
            for dr, dc in _D8:
                touching |= padded[1 + dr:1 + dr + wall.shape[0], 1 + dc:1 + dc + wall.shape[1]]
            if np.any(touching & new):
                continue
        else:
            edge = int(rng.integers(4))
            if edge < 2:
                if b - a < 5:
                    continue
                r = int(rng.integers(a + 2, b - 1))
                length = int(rng.integers(2, min(6, d - c - 2)))
                if edge == 0:
                    proposal[r, c:c + length] = True
                else:
                    proposal[r, d - length + 1:d + 1] = True
            else:
                if b - a < 5:
                    continue
                col = int(rng.integers(c + 2, d - 1))
                length = int(rng.integers(2, min(6, b - a - 2)))
                if edge == 2:
                    proposal[a:a + length, col] = True
                else:
                    proposal[b - length + 1:b + 1, col] = True
        block = proposal[:-1, :-1] & proposal[1:, :-1] & proposal[:-1, 1:] & proposal[1:, 1:]
        if np.any(block) or len(_components(~proposal)) != old_count:
            continue
        wall[:] = proposal
        return True
    return False


def _wave_majority(values, u):
    result = values.copy()
    for k in np.unique(u):
        cells = u == k
        colors, counts = np.unique(values[cells], return_counts=True)
        if len(colors) == 1 or counts[0] != counts[1]:
            result[cells] = colors[int(np.argmax(counts))]
    return result


def _paint_room(rng, room, colors, period, require_witness):
    t, span = _positions(room, "left")
    depth = _depth(room)[room]
    u = t - depth
    budget = max(1, int(room.sum()) // 8)
    groups = [int(k) for k in np.unique(u) if 1 <= np.count_nonzero(u == k) < budget]
    for attempt in range(96):
        # Enumerated nonconstant words provide a finite fallback after random words.
        if attempt < 64:
            bits = rng.integers(0, 2, size=period)
        else:
            n = 1 + (attempt - 64) % ((1 << period) - 2)
            bits = np.array([(n >> j) & 1 for j in range(period)])
        if len(np.unique(bits)) != 2:
            continue
        word = np.asarray(colors, dtype=np.int8)[bits]
        clean = word[u % period]
        noisy = clean.copy()
        chosen = np.zeros(len(clean), dtype=bool)
        if groups:
            k = groups[int(rng.integers(len(groups)))]
            chosen |= u == k
        elif require_witness:
            continue
        # Force an interior error that actually depends on contour drag.
        informative = np.flatnonzero((depth > 0) & (word[t % period] != clean) & ~chosen)
        if len(informative) and chosen.sum() < budget:
            chosen[int(rng.choice(informative))] = True
        available = np.flatnonzero(~chosen)
        extra = min(int(rng.integers(0, 4)), budget - int(chosen.sum()))
        if extra:
            chosen[rng.choice(available, size=extra, replace=False)] = True
        if not np.any(chosen):
            chosen[int(rng.integers(len(chosen)))] = True
        noisy[chosen] = np.where(clean[chosen] == colors[0], colors[1], colors[0])
        try:
            repaired, recovered, _ = _recover(noisy, u, span)
        except ValueError:
            continue
        if len(recovered) != period or not np.array_equal(repaired, clean):
            continue
        if require_witness:
            if np.array_equal(_wave_majority(noisy, u), clean):
                continue
            if not np.any(chosen & (word[t % period] != clean)):
                continue
            rr, cc = np.nonzero(room)
            box = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
            d4 = _depth(room, diagonal=False)[room]
            if (not np.any(clean != word[(t - box) % period])
                    or not np.any(clean != word[(t - d4) % period])):
                continue
        return noisy, clean
    raise ValueError("bounded word/noise construction exhausted")


def _make_pair(rng, index, test, roles, turns):
    wall_color, launch, palette = roles
    for attempt in range(28):
        if index == 0 and not test:
            h, w = int(rng.integers(13, 23)), int(rng.integers(17, 26))
            target = 1
        elif index == 1 and not test:
            h, w = int(rng.integers(16, 27)), int(rng.integers(16, 29))
            target = int(rng.integers(2, 5))
        elif test:
            h, w = int(rng.integers(12, 29)), int(rng.integers(20, 29))
            if rng.random() < 0.2:
                h = w = 28
            target = int(rng.integers(1, 7))
        else:
            h, w = int(rng.integers(8, 29)), int(rng.integers(12, 29))
            target = int(rng.integers(1, 6))
        walls, rects = _partition(rng, h, w, target)
        # First demonstration has a slit, second an island; tests combine them.
        ordered = sorted(rects, key=lambda q: (q[1] - q[0]) * (q[3] - q[2]), reverse=True)
        if index == 0 and not test:
            if not _add_obstacle(rng, walls, ordered[0], False):
                continue
        elif index == 1 and not test:
            if not _add_obstacle(rng, walls, ordered[0], True):
                continue
        obstacle_count = int(rng.integers(2 if test else 0, 6 if test else 4))
        for j in range(obstacle_count):
            rect = ordered[int(rng.integers(len(ordered)))]
            _add_obstacle(rng, walls, rect, bool((j + index) % 2))
        rooms = _components(~walls)
        inp = np.full(walls.shape, wall_color, dtype=np.int8)
        out = inp.copy()
        try:
            big = max(range(len(rooms)), key=lambda i: _positions(rooms[i], "left")[1])
            for i, room in enumerate(rooms):
                _, span = _positions(room, "left")
                cap = min(6, span // 3)
                if test and i == big:
                    p = int(rng.integers(5, cap + 1))
                elif index == 0 and not test:
                    p = int(rng.choice((3, 4)))
                elif index == 1 and not test and i < 2:
                    p = 2 + i
                else:
                    p = int(rng.integers(2, min(cap, 5 if not test else 6) + 1))
                colors = rng.choice(palette, size=2, replace=False)
                noisy, clean = _paint_room(rng, room, colors, p, index == 0 and not test)
                inp[room], out[room] = noisy, clean
            row = int(rng.integers(1, h + 1))
            inp[row, 0] = out[row, 0] = launch
            inp = np.rot90(inp, turns).copy()
            out = np.rot90(out, turns).copy()
            if not np.array_equal(solve(inp), out):
                continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded layout construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count("num_train", num_train, 2, 10)
    ne = _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    first_turn = int(rng.integers(4))
    # Perpendicular demonstrations expose both axes before the tests.
    turns = [first_turn, (first_turn + int(rng.choice((1, 3)))) % 4]
    turns += [int(rng.integers(4)) for _ in range(nt + ne - 2)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for serial in range(nt + ne):
        split = "train" if serial < nt else "test"
        index = serial if split == "train" else serial - nt
        shift = serial % 10
        local = palette[shift:] + palette[:shift]
        roles = (local[0], local[1], local[2:])
        for _ in range(4):
            pair = _make_pair(rng, index, split == "test", roles, turns[serial])
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    a, b = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == b.shape
    wall, launch, side, walls, rooms = _parse(a)
    assert np.array_equal(a[walls], b[walls]), "walls and launch marker changed"
    assert np.array_equal(solve(a), b), "output disagrees with input-only solver"
    records = []
    for room in rooms:
        d = _depth(room)
        t, span = _positions(room, side)
        u = t - d[room]
        repaired, word, errors = _recover(a[room], u, span)
        assert np.array_equal(repaired, b[room])
        assert 1 <= errors <= max(1, int(room.sum()) // 8)
        changed = a[room] != b[room]
        assert set(a[room]) == set(b[room]) and len(set(b[room])) == 2
        rr, cc = np.nonzero(room)
        box = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        records.append({
            "period": len(word), "span": span, "errors": errors,
            "area": int(room.sum()), "depth": int(d.max()),
            "full_wave_error": not np.array_equal(_wave_majority(a[room], u), repaired),
            "drag_error": bool(np.any(changed & (word[t % len(word)] != repaired))),
            "box_witness": bool(np.any(word[(t - box) % len(word)] != repaired)),
            "l1_witness": bool(np.any(word[(t - _depth(room, False)[room]) % len(word)] != repaired)),
            "reverse_witness": bool(np.any(word[(t + d[room]) % len(word)] != repaired)),
            "room": room,
        })
    return {"wall": wall, "launch": launch, "side": side, "walls": walls,
            "shape": a.shape, "rooms": records}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = _grid(pair["input"])
            key = (a.shape, a.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(pair)
            assert 1 <= len(info["rooms"]) <= 6
            assert not np.any(info["walls"][:-1, :-1] & info["walls"][1:, :-1]
                              & info["walls"][:-1, 1:] & info["walls"][1:, 1:])
            analyses[split].append(info)
    first, second = analyses["train"][:2]
    assert len(first["rooms"]) == 1 and len(second["rooms"]) >= 2
    assert {x["period"] for x in second["rooms"]} >= {2, 3}
    assert (first["side"] in ("top", "bottom")) != (second["side"] in ("top", "bottom"))
    for flag in ("full_wave_error", "drag_error", "box_witness", "l1_witness", "reverse_witness"):
        assert any(r[flag] for r in first["rooms"]), f"missing training witness: {flag}"
    assert len({a["wall"] for a in analyses["train"] + analyses["test"]}) >= 3
    for info in analyses["test"]:
        assert any(r["period"] >= 5 for r in info["rooms"])


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 puzzles streamed one at a time, plus bounded API and mutation checks."""
    shapes, layouts, periods, sides, counts, fingerprints = (set() for _ in range(6))
    error_sizes, depths = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair)
                shapes.add(info["shape"])
                layouts.add((info["shape"], info["walls"].tobytes()))
                sides.add(info["side"])
                counts.add(len(info["rooms"]))
                for r in info["rooms"]:
                    periods.add(r["period"])
                    error_sizes.add(r["errors"])
                    depths.add(r["depth"])
                    assert np.array_equal(_depth(r["room"]), _bfs_depth(r["room"]))
        # Input purity and determinism without retaining 42 grid payloads.
        original = puzzle["test"][0]["input"].copy()
        solve(puzzle["test"][0]["input"])
        assert np.array_equal(original, puzzle["test"][0]["input"])
    assert len(fingerprints) == 42 and len(shapes) >= 65 and len(layouts) >= 150
    assert periods == {2, 3, 4, 5, 6} and len(sides) == 4
    assert counts >= {1, 2, 3, 4, 5} and len(error_sizes) >= 10 and len(depths) >= 6
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    assert _signature(generate_puzzle(771, 2, 1)) == _signature(generate_puzzle(771, 2, 1))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(170, 2, 1)
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted a changed frame")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "shapes": len(shapes),
            "wall_layouts": len(layouts), "periods": sorted(periods), "directions": sorted(sides),
            "room_counts": sorted(counts), "error_counts": sorted(error_sizes), "depths": sorted(depths)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
