"""Complete a two-fold wall stencil, then paint its independently keyed rooms.

Only standard-library modules and NumPy are used. Generation writes no files.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _roles(a):
    counts = np.bincount(a.ravel(), minlength=10)
    order = sorted(range(10), key=lambda c: (-int(counts[c]), c))
    if not counts[order[0]] > counts[order[1]] > counts[order[2]]:
        raise ValueError("neutral and wall colors must be the unique two most frequent")
    return order[0], order[1]


def _closure(mask):
    return mask | mask[::-1, :] | mask[:, ::-1] | mask[::-1, ::-1]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    result = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        component = np.zeros_like(mask)
        queue = deque([(sr, sc)])
        seen[sr, sc] = True
        while queue:
            r, c = queue.popleft()
            component[r, c] = True
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(component)
    return result


def _depth(mask):
    """Distance from the eight-neighbor boundary; canvas exterior is outside."""
    h, w = mask.shape
    result = np.full(mask.shape, -1, dtype=np.int16)
    padded = np.pad(mask, 1)
    inner = mask.copy()
    for dr, dc in DIR8:
        inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    queue = deque()
    for r, c in np.argwhere(mask & ~inner):
        r, c = int(r), int(c)
        result[r, c] = 0
        queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and result[nr, nc] < 0:
                result[nr, nc] = result[r, c] + 1
                queue.append((nr, nc))
    return result


def _erosion(mask, diagonal=True):
    """Separate iterative erosion implementation for validation."""
    h, w = mask.shape
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int16)
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in DIR8 if diagonal else DIR4:
            inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~inner] = level
        current = inner
        level += 1
    return result


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _paths(mask, depths):
    result = []
    for r, c in np.argwhere(mask & (depths == 0)):
        r, c = int(r), int(c)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (_inside(mask, r - dr, c) or _inside(mask, r, c - dc)
                        or not _inside(mask, r + dr, c)
                        or not _inside(mask, r, c + dc)):
                    continue
                n = 0
                while _inside(mask, r + n * dr, c + n * dc) and depths[r + n * dr, c + n * dc] == n:
                    n += 1
                if n:
                    result.append((r, c, dr, dc, n))
    return result


def _read_key(a, mask, depths, neutral):
    cues = [(int(r), int(c)) for r, c in np.argwhere(mask & (a != neutral))]
    if not cues:
        return (neutral,), ()
    length = max(int(depths[p]) for p in cues) + 1
    candidates = []
    for r, c, dr, dc, capacity in _paths(mask, depths):
        if capacity >= length and all(p == (r + int(depths[p]) * dr, c + int(depths[p]) * dc) for p in cues):
            period = tuple(int(a[r + d * dr, c + d * dc]) for d in range(length))
            candidates.append((period, (dr, dc)))
    periods = {p for p, _ in candidates}
    if len(periods) != 1:
        raise ValueError("a room must specify one unambiguous inward diagonal period")
    return next(iter(periods)), tuple(sorted({d for _, d in candidates}))


def _decode(a):
    neutral, wall_color = _roles(a)
    visible = a == wall_color
    walls = _closure(visible)
    if np.any(walls & ~visible & (a != neutral)):
        raise ValueError("a palette cue cannot occupy a completed wall")
    result = np.full(a.shape, neutral, dtype=np.int8)
    result[walls] = wall_color
    rooms = []
    for mask in _components(~walls):
        depths = _depth(mask)
        period, directions = _read_key(a, mask, depths, neutral)
        result[mask] = np.asarray(period, dtype=np.int8)[depths[mask] % len(period)]
        rooms.append((mask, depths, period, directions))
    return result, (neutral, wall_color, visible, walls, rooms)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive everything from this grid; never consult a seed or demonstrations."""
    return _decode(_grid(input_grid))[0]


def _orbit(r, c, h, w):
    return tuple(sorted({(r, c), (h - 1 - r, c), (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))


def _wall_orbits(walls):
    h, w = walls.shape
    return [_orbit(r, c, h, w) for r in range((h + 1) // 2)
            for c in range((w + 1) // 2) if walls[r, c]]


def _line(mask, p, q):
    r, c = p
    rr, cc = q
    assert r == rr or c == cc
    mask[min(r, rr):max(r, rr) + 1, min(c, cc):max(c, cc) + 1] = True


def _primitive(rng, walls, kind):
    # Draw in a randomly transposed coordinate frame; closure is taken later.
    a = walls.T if rng.integers(2) else walls
    h, w = a.shape
    mh, mw = h // 2, w // 2
    if kind == 0:  # A border-to-fold dogleg, reflected into spanning stepped walls.
        c = int(rng.integers(4, mw - 1))
        row = int(rng.integers(3, mh - 1))
        shift = int(rng.choice([-2, -1, 1, 2]))
        cc = int(np.clip(c + shift, 3, mw - 1))
        _line(a, (0, c), (row, c))
        _line(a, (row, c), (row, cc))
        _line(a, (row, cc), (mh, cc))
    elif kind == 1:  # A centered enclosure, possibly crossed by later strokes.
        r, c = int(rng.integers(2, mh - 2)), int(rng.integers(2, mw - 2))
        for p, q in (((r, c), (r, w - 1 - c)), ((r, c), (h - 1 - r, c))):
            _line(a, p, q)
    elif kind == 2:  # An edge-attached pocket in a quarter of the canvas.
        c = int(rng.integers(2, mw - 4))
        cc = int(rng.integers(c + 4, mw + 1))
        r = int(rng.integers(4, mh + 1))
        _line(a, (0, c), (r, c))
        _line(a, (r, c), (r, cc))
        _line(a, (r, cc), (0, cc))
    elif kind == 3:  # Free orthogonal branches, which can attach or intersect.
        r, c = int(rng.integers(2, mh)), int(rng.integers(2, mw))
        rr, cc = int(rng.integers(r + 1, mh + 1)), int(rng.integers(c + 1, mw + 1))
        _line(a, (r, c), (rr, c))
        _line(a, (rr, c), (rr, cc))
    elif kind == 4:  # A full straight cut, with no required central cross.
        c = int(rng.integers(3, mw))
        a[:, c] = True
    else:  # An independent off-center loop and its reflected copies.
        r, c = int(rng.integers(1, mh - 4)), int(rng.integers(1, mw - 4))
        rr, cc = int(rng.integers(r + 4, mh + 1)), int(rng.integers(c + 4, mw + 1))
        for p, q in (((r, c), (r, cc)), ((r, c), (rr, c)),
                     ((rr, c), (rr, cc)), ((r, cc), (rr, cc))):
            _line(a, p, q)


def _fallback_walls():
    # Deep, concave six-room termination stencil, not a normal layout bank.
    a = np.zeros((29, 29), dtype=bool)
    _line(a, (0, 8), (7, 8))
    _line(a, (7, 8), (7, 10))
    _line(a, (7, 10), (14, 10))
    _line(a, (14, 0), (14, 28))
    return _closure(a)


def _shape_and_walls(rng, phase, fallback=False):
    if fallback:
        return _fallback_walls()
    h, w = (int(x) for x in rng.integers(18, 31, size=2))
    if rng.random() < 0.18:
        w = h
    a = np.zeros((h, w), dtype=bool)
    n = int(rng.integers(1, 4 if phase != "test" else 6))
    for _ in range(n):
        _primitive(rng, a, int(rng.integers(6)))
    return _closure(a)


def _box_depth(mask):
    rr, cc = np.nonzero(mask)
    return np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))


def _contour_witness(rooms):
    for mask, depths, period, _ in rooms:
        p = np.asarray(period)
        if len(p) < 2:
            continue
        values = p[depths[mask] % len(p)]
        if (np.any(values != p[_box_depth(mask) % len(p)])
                and np.any(values != p[_erosion(mask, False)[mask] % len(p)])):
            return True
    return False


def _paint_keys(rng, walls, neutral, wall_color, phase):
    masks = _components(~walls)
    if not 2 <= len(masks) <= 13 or not 0.04 <= walls.mean() <= 0.34:
        return None
    depths = [_depth(m) for m in masks]
    paths = [_paths(m, d) for m, d in zip(masks, depths)]
    capacities = [max((p[4] for p in ps), default=0) for ps in paths]
    big = max(range(len(masks)), key=lambda i: capacities[i])
    if capacities[big] < 4:
        return None
    forced = {big: 3}
    if phase == "first":
        twos = [i for i, cap in enumerate(capacities) if i != big and cap >= 2]
        if not twos:
            return None
        two = twos[int(rng.integers(len(twos)))]
        ones = [i for i, cap in enumerate(capacities) if i not in (big, two) and cap >= 1]
        if not ones:
            return None
        forced[two] = 2
        forced[ones[int(rng.integers(len(ones)))]] = 1
    payload = [c for c in range(10) if c not in (neutral, wall_color)]
    inp = np.full(walls.shape, neutral, dtype=np.int8)
    out = inp.copy()
    inp[walls] = out[walls] = wall_color
    rooms = []
    for i, (m, d, ps, capacity) in enumerate(zip(masks, depths, paths, capacities)):
        length = forced[i] if i in forced else (0 if not capacity or rng.random() < 0.12 else int(rng.integers(1, min(3, capacity) + 1)))
        if length == 0:
            rooms.append((m, d, (neutral,), ()))
            continue
        options = [p for p in ps if p[4] >= length]
        preferred = [p for p in options if p[2] == (-1 if phase == "second" else 1)]
        if phase in ("first", "second") and preferred:
            options = preferred
        r, c, dr, dc, _ = options[int(rng.integers(len(options)))]
        period = [int(x) for x in rng.choice(payload, length, replace=False)]
        if (phase == "second" and i == big) or (length >= 2 and rng.random() < 0.22):
            period[1 if length == 3 else 0] = neutral
        for j, color in enumerate(period):
            inp[r + j * dr, c + j * dc] = color
        p = np.asarray(period, dtype=np.int8)
        out[m] = p[d[m] % length]
        rooms.append((m, d, tuple(period), ((dr, dc),)))
    if not _contour_witness(rooms):
        return None
    if phase == "first" and (np.array_equal(out, out[::-1]) or np.array_equal(out, out[:, ::-1])):
        return None
    if phase == "second" and not any(dr == -1 for _, _, _, ds in rooms for dr, _ in ds):
        return None
    return inp, out, rooms


def _fragment(rng, walls, phase):
    orbits = _wall_orbits(walls)
    four = [i for i, o in enumerate(orbits) if len(o) == 4]
    if len(four) < 3:
        return None
    visible = np.zeros_like(walls)
    # Order witness positions using random horizontal/vertical cuts, then
    # add/remove partners orbitwise. At least one always survives.
    h, w = walls.shape
    cuts = [int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))]
    for orbit in orbits:
        scores = [int(r >= cuts[0]) + 2 * int(c >= cuts[1]) for r, c in orbit]
        priority = int(rng.integers(4))
        order = sorted(range(len(orbit)), key=lambda i: ((scores[i] - priority) % 4, i))
        if phase == "test":
            keep = 1 if rng.random() < 0.84 else min(2, len(orbit))
        elif phase == "first":
            keep = min(len(orbit), int(rng.integers(2, 5)))
        else:
            keep = min(len(orbit), int(rng.integers(1, 4)))
        for i in order[:keep]:
            visible[orbit[i]] = True
    selected = [four[int(i)] for i in rng.choice(len(four), 3, replace=False)]
    # Explicitly distinguish the two direct folds, and (except the easiest
    # first demonstration) require a diagonal-only witness as well.
    for j, index in enumerate(selected):
        orbit = orbits[index]
        survivor = orbit[int(rng.integers(4))]
        for point in orbit:
            visible[point] = (point[0] == survivor[0] if j == 0 else
                              point[1] == survivor[1] if j == 1 else
                              point == survivor if phase != "first" else True)
    return visible


def _interaction_witness(visible, rooms):
    incomplete_depth = _depth(~visible)
    for mask, depths, period, _ in rooms:
        p = np.asarray(period)
        if np.any(p[depths[mask] % len(p)] != p[incomplete_depth[mask] % len(p)]):
            return True
    return False


def _make_pair(rng, phase, force_fallback=False):
    # All trials are bounded and immediately discarded; no retries are logged.
    for attempt in range(160):
        walls = _shape_and_walls(rng, phase, fallback=force_fallback or attempt >= 128)
        neutral, wall_color = (int(x) for x in rng.choice(10, 2, replace=False))
        painted = _paint_keys(rng, walls, neutral, wall_color, phase)
        if painted is None:
            continue
        inp, out, rooms = painted
        visible = _fragment(rng, walls, phase)
        if visible is None:
            continue
        inp[walls & ~visible] = neutral
        try:
            if _roles(inp) != (neutral, wall_color):
                continue
            prediction, _ = _decode(inp)
        except ValueError:
            continue
        if not np.array_equal(prediction, out) or not _interaction_witness(visible, rooms):
            continue
        ratio = float(visible.sum() / walls.sum())
        if phase == "test" and not ratio < 0.40:
            continue
        if phase != "test" and not ratio > 0.40:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded stencil construction exhausted")


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, 2, 10, "num_train")
    ne = _integer(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + ne):
        phase = "test" if i >= nt else ("first" if i == 0 else "second" if i == 1 else "other")
        for _ in range(16):
            pair = _make_pair(rng, phase)
            key = _signature(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["test" if i >= nt else "train"].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
    return puzzle


def _signature(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _features(a, out):
    predicted, analysis = _decode(a)
    neutral, wall_color, visible, walls, rooms = analysis
    assert np.array_equal(predicted, out), "incorrect stencil or contour fill"
    assert np.array_equal(out == wall_color, walls), "wall geometry changed"
    assert np.array_equal(a[a != neutral], out[a != neutral]), "visible ink changed"
    assert np.any(walls & ~visible), "no wall completion"
    assert _interaction_witness(visible, rooms), "wall completion must change non-wall contour colors"
    assert _contour_witness(rooms), "must distinguish contour depth from box and Manhattan bands"
    assert any(len(p) == 3 and int(d.max()) >= 3 for _, d, p, _ in rooms), "show a complete repeated cycle"
    for mask, depths, _, _ in rooms:
        assert np.array_equal(depths, _erosion(mask)), "erosion/BFS mismatch"
    lr = visible | visible[:, ::-1]
    tb = visible | visible[::-1, :]
    assert np.any(walls & ~lr) and np.any(walls & ~tb), "both folds must matter"
    triple = sum(len(o) == 4 and sum(bool(visible[p]) for p in o) == 1 for o in _wall_orbits(walls))
    return {"shape": a.shape, "neutral": neutral, "wall": wall_color,
            "rooms": len(rooms), "wall_cells": int(walls.sum()),
            "visible_fraction": float(visible.sum() / walls.sum()), "triple_orbits": triple,
            "lengths": sorted({len(p) for _, _, p, _ in rooms}),
            "neutral_entry": any(neutral in p and len(p) > 1 for _, _, p, _ in rooms),
            "unmarked": any(p == (neutral,) for _, _, p, _ in rooms),
            "directions": sorted({direction for _, _, _, ds in rooms for direction in ds}),
            "concave_rooms": int(sum(int(m.sum()) < (np.ptp(np.nonzero(m)[0]) + 1) * (np.ptp(np.nonzero(m)[1]) + 1) for m, _, _, _ in rooms)),
            "asymmetric": not np.array_equal(out, out[::-1]) and not np.array_equal(out, out[:, ::-1]),
            "geometry": hashlib.sha256(walls.tobytes()).hexdigest()}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            key = _signature(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            f = _features(a, out)
            if split == "train":
                assert f["visible_fraction"] > 0.40
                train.append(f)
            else:
                assert f["visible_fraction"] < 0.40 and f["triple_orbits"] > 0
    assert any(f["asymmetric"] for f in train), "show independent inks in reflected rooms"
    assert {1, 2, 3} <= {n for f in train for n in f["lengths"]}, "show variable key lengths"
    assert any(f["neutral_entry"] for f in train), "show a neutral key entry"
    assert any(f["triple_orbits"] > 0 for f in train), "demonstrate diagonal-only wall recovery"
    assert any(any(dr == -1 for dr, _ in f["directions"]) for f in train)
    return True


def self_test():
    """42-seed coverage, independent depth checks, determinism and corruptions.

    Retain only small hashes/counters, never all generated grids or traces.
    """
    shapes, roles, geometries, room_counts, directions, lengths = (set() for _ in range(6))
    signatures = set()
    records = {"train": [], "test": []}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                f = _features(a, out)
                digest.update(bytes(a.shape) + a.tobytes() + out.tobytes())
                shapes.add(f["shape"])
                roles.add((f["neutral"], f["wall"]))
                geometries.add((f["shape"], f["geometry"]))
                room_counts.add(f["rooms"])
                directions.update(f["directions"])
                lengths.update(f["lengths"])
                records[split].append(f)
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 40 and len(geometries) >= 100
    assert len(roles) >= 40 and len(room_counts) >= 5
    assert directions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert lengths == {1, 2, 3}
    assert any(f["unmarked"] for f in records["train"])
    assert any(30 in s for s in shapes) and any(h != w for h, w in shapes)
    for phase in ("first", "second", "test"):
        pair = _make_pair(np.random.default_rng(710), phase, force_fallback=True)
        _features(pair["input"], pair["output"])
    for nt, ne in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(1019, nt, ne)
        q = generate_puzzle(1019, nt, ne)
        validate_generated_puzzle(p)
        assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(210, 2, 1)
    for kind in ("wall", "paint", "cue"):
        bad = {s: [{k: a.copy() for k, a in p.items()} for p in original[s]] for s in original}
        pair = bad["test"][0]
        a, out = pair["input"], pair["output"]
        neutral, wall_color = _roles(a)
        eligible = out == wall_color if kind == "wall" else out != wall_color
        if kind == "cue":
            eligible &= a != neutral
        point = tuple(np.argwhere(eligible)[0])
        out[point] = (int(out[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupt output accepted")
    summary = {"seeds": 42, "pairs": sum(map(len, records.values())), "shapes": len(shapes),
               "role_pairs": len(roles), "geometries": len(geometries), "room_counts": sorted(room_counts),
               "directions": sorted(directions), "period_lengths": sorted(lengths),
               "train_visible_range": [min(f["visible_fraction"] for f in records["train"]), max(f["visible_fraction"] for f in records["train"])],
               "test_visible_range": [min(f["visible_fraction"] for f in records["test"]), max(f["visible_fraction"] for f in records["test"])]}
    return summary


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
