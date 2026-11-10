"""Shape-linked reservoirs: rotation classes are communicating water networks.

Only Python's standard library and NumPy are used. No files are read by solving
or generation. See task_spec.md for the full-level and enclosure conventions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json

import numpy as np


WATER = 1
D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)
ALTERNATIVES = ("independent", "global", "color", "area", "reflection",
                "identity", "half_turn", "exactly_two", "equal_share",
                "relative_level", "unchanged")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or a.dtype.kind not in "iu" or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, directions=D4):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in directions:
                p = (y + dy, x + dx)
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _crop(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ValueError("empty shape")
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def _mask_key(mask):
    a = _crop(mask)
    return (a.shape[0], a.shape[1], a.tobytes())


def _canonical(mask):
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _hole_mask(shell):
    """Recover bounded free space without consulting the liquid pattern."""
    padded = np.pad(shell, 1)
    outside = np.zeros(padded.shape, dtype=bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    while stack:
        y, x = stack.pop()
        for dy, dx in D4:
            ny, nx = y + dy, x + dx
            if (0 <= ny < padded.shape[0] and 0 <= nx < padded.shape[1]
                    and not padded[ny, nx] and not outside[ny, nx]):
                outside[ny, nx] = True
                stack.append((ny, nx))
    return (~padded & ~outside)[1:-1, 1:-1]


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if (bg == WATER or np.any(g[0] != bg) or np.any(g[-1] != bg)
            or np.any(g[:, 0] != bg) or np.any(g[:, -1] != bg)):
        raise ValueError("a nonblue background border is required")
    tanks = []
    all_interiors = np.zeros(g.shape, dtype=bool)
    all_walls = set()
    for color in sorted(set(map(int, np.unique(g))) - {bg, WATER}):
        for cells in _components(g == color, D8):
            top = min(y for y, x in cells)
            left = min(x for y, x in cells)
            bottom = max(y for y, x in cells)
            right = max(x for y, x in cells)
            wall = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
            for y, x in cells:
                wall[y - top, x - left] = True
            hole = _hole_mask(wall)
            if len(_components(hole)) != 1:
                raise ValueError("each tank wall must enclose exactly one connected cavity")
            interior = np.zeros(g.shape, dtype=bool)
            interior[top:bottom + 1, left:right + 1] = hole
            if (np.any(all_interiors & interior)
                    or np.any((g[interior] != bg) & (g[interior] != WATER))):
                raise ValueError("tank interiors must be disjoint and contain only background/liquid")
            if any((y + dy, x + dx) in all_walls
                   for y, x in cells for dy, dx in D8):
                raise ValueError("different tanks must have background clearance")
            all_walls.update(cells)
            all_interiors |= interior
            tanks.append({"color": color, "wall": wall, "top": top, "left": left,
                          "cells": cells, "interior": interior, "key": _canonical(wall)})
    if not tanks or np.any((g == WATER) & ~all_interiors):
        raise ValueError("all liquid must lie inside a tank")
    return g, bg, tanks


def _groups(tanks, mode="rotation"):
    frequencies = Counter(t["key"] for t in tanks)
    groups = defaultdict(list)
    for i, t in enumerate(tanks):
        wall = t["wall"]
        if mode == "independent":
            key = i
        elif mode == "global":
            key = 0
        elif mode == "color":
            key = t["color"]
        elif mode == "area":
            key = int(wall.sum())
        elif mode == "reflection":
            key = min(t["key"], _canonical(np.fliplr(wall)))
        elif mode == "identity":
            key = _mask_key(wall)
        elif mode == "half_turn":
            key = min(_mask_key(wall), _mask_key(np.rot90(wall, 2)))
        elif mode == "exactly_two":
            key = ("pair", t["key"]) if frequencies[t["key"]] == 2 else ("single", i)
        else:
            key = t["key"]
        groups[key].append(i)
    return list(groups.values())


def _union(tanks, indices):
    mask = np.zeros(tanks[0]["interior"].shape, dtype=bool)
    for i in indices:
        mask |= tanks[i]["interior"]
    return mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the result solely from the supplied grid, with no learned state."""
    g, bg, tanks = _parse(input_grid)
    out = g.copy()
    out[out == WATER] = bg
    for indices in _groups(tanks):
        domain = _union(tanks, indices)
        volume = int(np.count_nonzero((g == WATER) & domain))
        cells = sorted(map(tuple, np.argwhere(domain)), key=lambda p: (-p[0], p[1]))
        if 0 < volume < len(cells) and cells[volume - 1][0] == cells[volume][0]:
            raise ValueError("volume splits a tied horizontal level")
        for y, x in cells[:volume]:
            out[y, x] = WATER
    return out


def _partition_fits(g, out, tanks, mode):
    """Could the shown output be a lowest-first filling of this partition?

    Even partial tied rows are allowed here, so rejection does not depend on an
    arbitrary left/right tie break in a competing explanation.
    """
    if mode == "unchanged":
        return np.array_equal(g, out)
    if mode == "equal_share":
        for indices in _groups(tanks):
            counts = [int(((out == WATER) & tanks[i]["interior"]).sum()) for i in indices]
            if max(counts) - min(counts) > 1:
                return False
        return True
    if mode == "relative_level":
        for indices in _groups(tanks):
            filled_depths, empty_depths = [], []
            for i in indices:
                interior = tanks[i]["interior"]
                bottom = int(np.nonzero(interior)[0].max())
                filled_depths.extend(bottom - int(y) for y in np.nonzero(interior & (out == WATER))[0])
                empty_depths.extend(bottom - int(y) for y in np.nonzero(interior & (out != WATER))[0])
            if filled_depths and empty_depths and max(filled_depths) > min(empty_depths):
                return False
        return True
    for indices in _groups(tanks, mode):
        domain = _union(tanks, indices)
        filled = domain & (out == WATER)
        empty = domain & (out != WATER)
        if int(((g == WATER) & domain).sum()) != int(filled.sum()):
            return False
        if filled.any() and empty.any():
            if int(np.nonzero(filled)[0].min()) < int(np.nonzero(empty)[0].max()):
                return False
    return True


def _shell(inner):
    inside = np.pad(_crop(inner), 1)
    dilated = inside.copy()
    for y, x in np.argwhere(inside):
        dilated[y - 1:y + 2, x - 1:x + 2] = True
    wall = dilated & ~inside
    if (len(_components(wall, D8)) != 1
            or not np.array_equal(_hole_mask(wall), inside)):
        raise ValueError("shell would contain an unintended cavity")
    return wall


def _random_inner(rng):
    h, w = map(int, rng.integers(2, 6, size=2))
    kind = int(rng.integers(5))
    if kind == 0:
        return np.ones((h, w), dtype=bool)
    if kind == 1:
        a = np.zeros((h, w), dtype=bool)
        a[:, 0] = True
        a[-1, :] = True
        return a
    if kind == 2:
        a = np.zeros((h, w), dtype=bool)
        a[int(rng.integers(h)), :] = True
        a[:, int(rng.integers(w))] = True
        return _crop(a)
    if kind == 3:
        a = np.zeros((h, w), dtype=bool)
        for y in range(h):
            a[y, :int(rng.integers(1, w + 1))] = True
        return _crop(a)
    cells = {(0, 0)}
    target = int(rng.integers(3, 18))
    for _ in range(target - 1):
        candidates = set()
        for y, x in sorted(cells):
            for dy, dx in D4:
                p = y + dy, x + dx
                trial = cells | {p}
                ys, xs = zip(*trial)
                if p not in cells and max(ys) - min(ys) < 5 and max(xs) - min(xs) < 5:
                    candidates.add(p)
        options = sorted(candidates)
        if not options:
            break
        cells.add(options[int(rng.integers(len(options)))])
    ys, xs = zip(*cells)
    a = np.zeros((max(ys) - min(ys) + 1, max(xs) - min(xs) + 1), dtype=bool)
    for y, x in cells:
        a[y - min(ys), x - min(xs)] = True
    return a


def _sample_wall(rng, forbidden, chiral=False, ordinary=False):
    for _ in range(100):
        try:
            wall = _shell(_random_inner(rng))
        except ValueError:
            continue
        key, mirror = _canonical(wall), _canonical(np.fliplr(wall))
        if (key in forbidden or (chiral and key == mirror)
                or (ordinary and mirror in forbidden)):
            continue
        return wall
    # A finite deterministic vocabulary also provides chiral shapes.
    for h in range(2, 6):
        for w in range(2, 6):
            inner = np.zeros((h, w), dtype=bool)
            inner[:, 0] = True
            inner[-1, :] = True
            wall = _shell(inner)
            key, mirror = _canonical(wall), _canonical(np.fliplr(wall))
            if key not in forbidden and (not chiral or key != mirror) and (not ordinary or mirror not in forbidden):
                return wall
    raise AssertionError("bounded wall vocabulary exhausted")


def _objects(rng, profile, witness, bg):
    bases, used, objects = [], set(), []
    for group, count in enumerate(profile):
        if witness == "triplet" and group == 1:
            base = np.fliplr(bases[0]).copy()
        elif group > 0 and count > 1 and rng.random() < .30:
            reflected = np.fliplr(bases[0]).copy()
            base = reflected if _canonical(reflected) not in used else _sample_wall(rng, used)
        else:
            base = _sample_wall(rng, used, chiral=group == 0,
                                ordinary=count == 1 and witness != "triplet")
        key = _canonical(base)
        if key in used:
            raise AssertionError("class collision")
        used.add(key)
        bases.append(base)
        first_turn = int(rng.integers(4))
        for member in range(count):
            turn = (first_turn + 1) % 4 if group == 0 and member == 1 else int(rng.integers(4))
            if group == 0 and member == 0:
                turn = first_turn
            wall = np.rot90(base, turn).copy()
            objects.append({"wall": wall, "group": group})
    palette = [c for c in range(10) if c not in (bg, WATER)]
    rng.shuffle(palette)
    palette_size = int(rng.integers(1, min(len(objects), len(palette)) + 1))
    for obj in objects:
        obj["color"] = int(rng.choice(palette[:palette_size]))
    if witness == "triplet":
        # Different colors within the triple; one reused by its reflected singleton.
        objects[0]["color"], objects[1]["color"] = palette[0], palette[1]
        objects[3]["color"] = palette[0]
    return objects


def _pack(rng, objects, force_fallback=False):
    if not force_fallback:
        for attempt in range(12):
            low = 17 if attempt < 5 else 25
            h, w = map(int, rng.integers(low, 31, size=2))
            occupied = np.zeros((h, w), dtype=bool)
            anchors, placements = {}, {}
            order = sorted(range(len(objects)), key=lambda i: -objects[i]["wall"].size)
            for i in order:
                obj = objects[i]
                th, tw = obj["wall"].shape
                for trial in range(70):
                    if trial < 45 and obj["group"] in anchors:
                        y = int(np.clip(anchors[obj["group"]] + int(rng.integers(-3, 4)), 1, h - th - 1))
                    else:
                        y = int(rng.integers(1, h - th))
                    x = int(rng.integers(1, w - tw))
                    if occupied[y - 1:y + th + 1, x - 1:x + tw + 1].any():
                        continue
                    occupied[y:y + th, x:x + tw] = True
                    placements[i] = (y, x)
                    anchors.setdefault(obj["group"], y)
                    break
                else:
                    break
            if len(placements) == len(objects):
                return h, w, placements, False
    # Seven <=7x7 walls fit into seven of nine separated 9x9 slots.
    slots = list(map(int, rng.permutation(9)[:len(objects)]))
    placements = {}
    for i, slot in enumerate(slots):
        th, tw = objects[i]["wall"].shape
        placements[i] = (1 + 9 * (slot // 3) + int(rng.integers(8 - th)),
                         1 + 9 * (slot % 3) + int(rng.integers(8 - tw)))
    return 30, 30, placements, True


def _target(rng, tanks, require_shared):
    result = np.zeros(tanks[0]["interior"].shape, dtype=bool)
    rows = np.arange(result.shape[0])[:, None]
    shared_seen = False
    for indices in _groups(tanks):
        domain = _union(tanks, indices)
        options = []
        for level in sorted(set(map(int, np.nonzero(domain)[0]))):
            wet = domain & (rows >= level)
            volume = int(wet.sum())
            if not 1 < volume < int(domain.sum()):
                continue
            partial = sum(0 < int((wet & tanks[i]["interior"]).sum())
                          < int(tanks[i]["interior"].sum()) for i in indices)
            options.append((level, partial))
        if not options:
            raise ValueError("no nontrivial complete level")
        shared = [option for option in options if option[1] >= 2]
        if len(indices) > 1 and shared:
            options = shared if require_shared or rng.random() < .80 else options
        level, partial = options[int(rng.integers(len(options)))]
        shared_seen |= partial >= 2
        result |= domain & (rows >= level)
    if require_shared and not shared_seen:
        raise ValueError("no shared partial meniscus")
    return result


def _supply(rng, tanks, target):
    source = np.zeros(target.shape, dtype=bool)
    for indices in _groups(tanks):
        domain = _union(tanks, indices)
        points = np.argwhere(domain)
        volume = int((target & domain).sum())
        target_counts = [int((target & tanks[i]["interior"]).sum()) for i in indices]
        for attempt in range(12):
            style = int(rng.integers(4))
            jitter = rng.random(len(points))
            if style == 0:
                scores = jitter
            elif style == 1:
                donor = tanks[indices[int(rng.integers(len(indices)))]]
                scores = ~donor["interior"][points[:, 0], points[:, 1]] * 10 + jitter
            elif style == 2:
                scores = points[:, 0] + jitter
            else:
                scores = (~target[points[:, 0], points[:, 1]]) * 2 + jitter
            chosen = points[np.argsort(scores)[:volume]]
            supply = np.zeros(target.shape, dtype=bool)
            supply[chosen[:, 0], chosen[:, 1]] = True
            changed_counts = target_counts != [int((supply & tanks[i]["interior"]).sum()) for i in indices]
            if not np.array_equal(supply, target & domain) and (len(indices) == 1 or changed_counts):
                break
        else:
            # A single swap across members guarantees a visible network transfer.
            supply = target & domain
            swapped = False
            for a in indices:
                filled = np.argwhere(supply & tanks[a]["interior"])
                for b in indices:
                    empty = np.argwhere(~supply & tanks[b]["interior"])
                    if (a != b or len(indices) == 1) and len(filled) and len(empty):
                        supply = supply.copy()
                        supply[tuple(filled[0])] = False
                        supply[tuple(empty[0])] = True
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                raise ValueError("cannot construct a visible transfer")
        source |= supply
    return source


def _make_pair(rng, bg, profile, witness):
    for _ in range(80):
        objects = _objects(rng, profile, witness, bg)
        h, w, placements, _ = _pack(rng, objects)
        inp = np.full((h, w), bg, dtype=np.int8)
        for i, obj in enumerate(objects):
            y, x = placements[i]
            wall = obj["wall"]
            part = inp[y:y + wall.shape[0], x:x + wall.shape[1]]
            part[wall] = obj["color"]
        _, _, tanks = _parse(inp)
        try:
            target = _target(rng, tanks, require_shared=witness in ("triplet", "networks"))
            source = _supply(rng, tanks, target)
        except ValueError:
            continue
        inp[source] = WATER
        out = inp.copy()
        out[source] = bg
        out[target] = WATER
        if witness == "triplet":
            if any(_partition_fits(inp, out, tanks, mode) for mode in ALTERNATIVES):
                continue
        assert np.array_equal(solve(inp), out)
        return {"input": inp, "output": out}
    raise AssertionError("bounded reservoir construction exhausted")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    stable_background = int(rng.choice([c for c in range(10) if c != WATER])) if rng.random() < .4 else None
    train_profiles = ((2, 1), (2, 1, 1), (3, 1), (2, 2), (2, 2, 1),
                      (3, 1, 1), (2, 2, 1, 1), (1, 1, 1), (3,))
    test_profiles = ((3, 2, 1), (3, 3, 1))
    requirements = ["triplet", "networks"] + ["extra"] * (int(num_train) - 2)
    rng.shuffle(requirements)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            witness = requirements[index] if split == "train" else "test"
            if witness == "triplet":
                profile = (3, 1, 1) if rng.random() < .35 else (3, 1)
            elif witness == "networks":
                profile = (2, 2, 1, 1) if rng.random() < .35 else (2, 2, 1)
            else:
                profiles = test_profiles if split == "test" else train_profiles
                profile = profiles[int(rng.integers(len(profiles)))]
            for _ in range(12):
                bg = stable_background if stable_background is not None else int(rng.choice([c for c in range(10) if c != WATER]))
                pair = _make_pair(rng, bg, profile, witness)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise AssertionError("distinct-input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(pair):
    g, bg, tanks = _parse(pair["input"])
    out = _grid(pair["output"])
    assert out.shape == g.shape
    wall_cells = (g != bg) & (g != WATER)
    assert np.array_equal(g[wall_cells], out[wall_cells]), "wall changed"
    assert np.all((out == bg) | (out == WATER) | wall_cells), "new wall appeared"
    all_inside = _union(tanks, range(len(tanks)))
    assert np.all(out[~all_inside] == g[~all_inside]), "outside of tank changed"
    assert np.array_equal(solve(g), out), "solver and label disagree"
    assert not np.array_equal(g, out), "no transformation"
    groups = _groups(tanks)
    sizes, levels, transfers, shared = [], [], 0, 0
    keys = {t["key"] for t in tanks}
    frequencies = Counter(t["key"] for t in tanks)
    mirror_negative, ordinary, quarter, color_partner, color_cross = False, False, False, False, False
    for indices in groups:
        domain = _union(tanks, indices)
        wet = domain & (out == WATER)
        empty = domain & (out != WATER)
        sizes.append(len(indices))
        assert int(((g == WATER) & domain).sum()) == int(wet.sum()), "class volume changed"
        if wet.any() and empty.any():
            assert int(np.nonzero(wet)[0].min()) > int(np.nonzero(empty)[0].max()), "incomplete or unsupported level"
        level = int(np.nonzero(wet)[0].min()) if wet.any() else -1
        levels.append(level)
        before = [int(((g == WATER) & tanks[i]["interior"]).sum()) for i in indices]
        after = [int(((out == WATER) & tanks[i]["interior"]).sum()) for i in indices]
        if len(indices) > 1:
            assert before != after, "repeated class does not exchange volume"
            transfers += 1
        shared += sum(0 < n < int(tanks[i]["interior"].sum()) for i, n in zip(indices, after)) >= 2
        for a in indices:
            for b in indices:
                if a == b:
                    continue
                ka, kb = _mask_key(tanks[a]["wall"]), _mask_key(tanks[b]["wall"])
                quarter |= ka != kb and kb in (_mask_key(np.rot90(tanks[a]["wall"], 1)), _mask_key(np.rot90(tanks[a]["wall"], 3)))
                color_partner |= tanks[a]["color"] != tanks[b]["color"]
    for t in tanks:
        mirror = _canonical(np.fliplr(t["wall"]))
        if frequencies[t["key"]] == 1:
            ordinary |= mirror == t["key"] or mirror not in keys
            mirror_negative |= mirror != t["key"] and mirror in keys
        color_cross |= any(t["color"] == s["color"] and t["key"] != s["key"] for s in tanks)
    return {"background": bg, "shape": g.shape, "tanks": len(tanks), "sizes": sorted(sizes),
            "levels": sorted(set(levels)), "transfers": transfers, "shared": shared,
            "mirror_negative": mirror_negative, "ordinary": ordinary, "quarter": quarter,
            "color_partner": color_partner, "color_cross": color_cross,
            "areas": [int(t["interior"].sum()) for t in tanks],
            "volume": int((g == WATER).sum()),
            "rejected": [m for m in ALTERNATIVES if not _partition_fits(g, out, tanks, m)]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            try:
                for value in pair.values():
                    _grid(value)
                feature = _features(pair)
            except ValueError as error:
                raise AssertionError(str(error)) from error
            key = _fingerprint(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert 3 <= feature["tanks"] <= 7 and max(feature["sizes"]) <= 3
            features[split].append(feature)
    train, test = features["train"], features["test"]
    assert any(3 in f["sizes"] and f["mirror_negative"] for f in train)
    assert any(f["sizes"].count(2) >= 2 and f["ordinary"] for f in train)
    assert any(f["quarter"] and f["color_partner"] and f["color_cross"] for f in train)
    assert any(f["shared"] for f in train)
    assert set(ALTERNATIVES) <= {m for f in train for m in f["rejected"]}, "teaching shortcuts survive"
    assert all(not (3 in f["sizes"] and f["transfers"] >= 2) for f in train)
    assert all(f["sizes"] in ([1, 2, 3], [1, 3, 3]) for f in test), "missing recombination shift"
    return None


def self_test():
    """42 puzzles, bounded streaming metrics, hand checks, and count checks."""
    signatures, structures, shapes, backgrounds, profiles, volumes, areas = (set() for _ in range(7))
    pairs = transfers = shared = reflected = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                g = np.asarray(pair["input"])
                signature.update(bytes(g.shape) + g.tobytes() + np.asarray(pair["output"]).tobytes())
                normalized = np.where(g == f["background"], 0, np.where(g == WATER, 1, 2)).astype(np.int8)
                structures.add((g.shape, sha256(normalized.tobytes()).digest()))
                shapes.add(g.shape)
                backgrounds.add(f["background"])
                profiles.add(tuple(f["sizes"]))
                volumes.add(f["volume"])
                areas.update(f["areas"])
                pairs += 1
                transfers += f["transfers"]
                shared += bool(f["shared"])
                reflected += f["mirror_negative"]
        signatures.add(signature.digest())
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(np.array_equal(puzzle[s][i][r], again[s][i][r])
                       for s in puzzle for i in range(len(puzzle[s])) for r in ("input", "output"))
            bad = {s: [{r: np.asarray(v).copy() for r, v in p.items()} for p in puzzle[s]] for s in puzzle}
            bad["test"][0]["output"][0, 0] = WATER
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("corrupted output accepted")
    assert len(signatures) == 42 and len(structures) >= 190
    assert len(shapes) >= 35 and len(backgrounds) >= 8
    assert len(profiles) >= 8 and len(volumes) >= 35 and len(areas) >= 12
    assert transfers >= 220 and shared >= 70 and reflected >= 42
    # Direct small reference fixture: two rectangular chambers with unequal elevations.
    g = np.zeros((12, 16), dtype=np.int8)
    wall = _shell(np.ones((3, 3), dtype=bool))
    for top, left, color in ((1, 1, 2), (3, 9, 3)):
        patch = g[top:top + 5, left:left + 5]
        patch[wall] = color
    g[2:5, 2:5] = WATER  # nine cells in the higher tank, three in the lower
    g[4, 10:13] = WATER
    expected = g.copy()
    expected[expected == WATER] = 0
    expected[4, 2:5] = WATER
    expected[4:7, 10:13] = WATER
    assert np.array_equal(solve(g), expected)
    # An inadmissible row tie is rejected, not silently resolved by column order.
    g[2, 2] = 0
    try:
        solve(g)
    except ValueError:
        pass
    else:
        raise AssertionError("partial tied row accepted")
    # Exercise the deterministic packing fallback separately, preserving all walls.
    rng = np.random.default_rng(321)
    objects = _objects(rng, (3, 3, 1), "test", 0)
    h, w, placements, fallback = _pack(rng, objects, force_fallback=True)
    assert fallback and (h, w) == (30, 30)
    fallback_grid = np.zeros((h, w), dtype=np.int8)
    for i, obj in enumerate(objects):
        y, x = placements[i]
        patch = fallback_grid[y:y + obj["wall"].shape[0], x:x + obj["wall"].shape[1]]
        patch[obj["wall"]] = obj["color"]
    assert len(_parse(fallback_grid)[2]) == 7
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(700 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "structures": len(structures), "shapes": len(shapes),
            "backgrounds": len(backgrounds), "class_profiles": len(profiles),
            "volumes": len(volumes), "cavity_areas": len(areas),
            "transferring_networks": transfers, "shared_level_pairs": shared,
            "reflection_negative_pairs": reflected}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
