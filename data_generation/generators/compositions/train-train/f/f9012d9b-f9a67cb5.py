"""Invisible periodic sluices: infer erased geometry, draw only its signal trail.

The solver uses input evidence only.  Generation constructs a periodic substrate
and a path independently of the solver, then hides a square of that substrate.
"""
from __future__ import annotations

import json
from numbers import Integral

import numpy as np


HOST_SEEDS = (0, 836526702, 728436947, 37118486, 997579812, 672408230,
              428131219, 222676582, 34961649, 232473677, 364317600,
              168578679, 17052155, 977539001, 621814649, 113762238,
              914262928, 393819005, 993858859, 615059564, 954761363,
              905732898, 685753936, 29454209, 590066803, 311891089,
              581439474, 614147036, 138944006, 442425671, 440319006,
              855669715, 703353823, 471508060, 848525573, 166763380,
              929855395, 176811838, 298942629, 638053001, 943499096,
              975558589)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _orient(a, direction, inverse=False):
    # Reflections and transposes preserve the physical lateral distances.
    if direction == 0:  # top
        return a.copy()
    if direction == 1:  # bottom
        return a[::-1].copy()
    if direction == 2:  # left
        return a.T.copy()
    return (a.T[:, ::-1] if inverse else a[:, ::-1].T).copy()


def _key(a):
    return a.shape, a.tobytes()


def _square(a, color):
    points = np.argwhere(a == color)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    size = int(hi[0] - lo[0])
    if size >= 2 and size == hi[1] - lo[1] and len(points) == size * size:
        return int(lo[0]), int(lo[1]), size
    return None


def _consistent_periods(a, known, axis):
    """Necessary one-axis consistency, including observations beyond a gap."""
    rr, cc = np.nonzero(known)
    vals = a[known]
    answer = []
    for p in range(1, a.shape[axis] // 2 + 1):
        if axis == 0:
            ids, n = (rr % p) * a.shape[1] + cc, p * a.shape[1]
        else:
            ids, n = rr * p + cc % p, a.shape[0] * p
        low, high = np.full(n, 10), np.full(n, -1)
        np.minimum.at(low, ids, vals)
        np.maximum.at(high, ids, vals)
        seen = high >= 0
        if np.all(low[seen] == high[seen]):
            answer.append(p)
    return answer


def _recover(a, known):
    """Fully witnessed rectangular periods with two repeats on each axis.

    Every admissible period must agree on the ENTIRE reconstruction.  We do
    not choose a period by a seed-dependent or training-dependent convention.
    """
    rr, cc = np.nonzero(known)
    vals = a[known]
    choices = []
    for pr in _consistent_periods(a, known, 0):
        for pc in _consistent_periods(a, known, 1):
            ids = (rr % pr) * pc + cc % pc
            low, high = np.full(pr * pc, 10), np.full(pr * pc, -1)
            np.minimum.at(low, ids, vals)
            np.maximum.at(high, ids, vals)
            if np.any(high < 0) or np.any(low != high):
                continue
            tile = high.reshape(pr, pc).astype(np.int8)
            full = tile[np.arange(a.shape[0])[:, None] % pr,
                        np.arange(a.shape[1])[None, :] % pc]
            choices.append((pr * pc, pr, pc, full, tile))
    if not choices:
        raise ValueError("no fully witnessed periodic substrate")
    choices.sort(key=lambda item: item[:3])
    full = choices[0][3]
    if any(not np.array_equal(full, item[3]) for item in choices[1:]):
        raise ValueError("ambiguous periodic substrate")
    return full.copy(), choices[0][1:3], choices[0][4].copy()


def _inspect(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4:
        raise ValueError("four distinct roles are required")
    seeds = []
    for color, count in zip(colors, counts):
        if count != 1:
            continue
        r, c = map(int, np.argwhere(a == color)[0])
        edges = (r == 0, r == a.shape[0] - 1,
                 c == 0, c == a.shape[1] - 1)
        if sum(edges) == 1:
            seeds.append((int(color), edges.index(True)))
    if len(seeds) != 1:
        raise ValueError("one non-corner edge seed is required")
    signal, direction = seeds[0]
    masks = [(int(c), _square(a, c)) for c in colors if c != signal]
    masks = [(c, box) for c, box in masks if box is not None]
    if len(masks) != 1:
        raise ValueError("one solid square erasure is required")
    erased = masks[0][0]
    canonical = _orient(a, direction)
    box = _square(canonical, erased)
    known = (canonical != erased) & (canonical != signal)
    content = [int(c) for c in colors if c not in (signal, erased)]
    edge_content = set(map(int, canonical[0][known[0]]))
    if len(edge_content) != 1:
        raise ValueError("the entry edge must be an open corridor")
    background = edge_content.pop()
    wall = next(c for c in content if c != background)
    full, periods, tile = _recover(canonical, known)
    walls = tuple(map(int, np.flatnonzero(np.any(full == wall, axis=1))))
    if (len(walls) < 2 or walls[0] < 2 or walls[-1] > len(full) - 2
            or any(b - a < 2 for a, b in zip(walls, walls[1:]))):
        raise ValueError("walls must be separated with open entry and exit")
    for r in walls:
        holes = np.flatnonzero(full[r] == background)
        if not 1 <= len(holes) <= full.shape[1] // 2:
            raise ValueError("a wall must have sparse apertures")
    seed_col = int(np.argwhere(canonical == signal)[0, 1])
    if full[0, seed_col] != background:
        raise ValueError("the seed must replace open substrate")
    return dict(input=a, canonical=canonical, full=full, periods=periods,
                tile=tile, roles=(background, wall, signal, erased),
                direction=direction, walls=walls, seed=seed_col, box=box)


def _trace(full, wall, seed, walls, lateral=True):
    """Layerwise directed routing; a set merges branches without multiplicity."""
    h, w = full.shape
    path = np.zeros((h, w), dtype=bool)
    path[0, seed] = True
    active, previous = {seed}, 0
    events = {"collisions": [], "direct": [], "redirected": [],
              "splits": 0, "mergers": 0, "deaths": 0, "active": []}
    for row in walls:
        outgoing = set()
        for col in sorted(active):
            path[previous + 1:row, col] = True
            if full[row, col] != wall:
                events["direct"].append((row, col))
                path[row, col] = True
                events["mergers"] += col in outgoing
                outgoing.add(col)
                continue
            events["collisions"].append((row, col))
            if not lateral:
                continue
            exits = 0
            for step in (-1, 1):
                x = col + step
                while 0 <= x < w:
                    path[row - 1, x] = True
                    if full[row, x] != wall:
                        path[row, x] = True
                        events["redirected"].append((row, x))
                        events["mergers"] += x in outgoing
                        outgoing.add(x)
                        exits += 1
                        break
                    x += step
                else:
                    events["deaths"] += 1
            events["splits"] += exits == 2
        active, previous = outgoing, row
        events["active"].append(len(active))
    for col in active:
        path[previous + 1:, col] = True
    return path, events


def _render(info, path):
    result = info["canonical"].copy()
    result[path] = info["roles"][2]
    return _orient(result, info["direction"], inverse=True)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _inspect(input_grid)
    path, _ = _trace(info["full"], info["roles"][1], info["seed"], info["walls"])
    return _render(info, path)


def _diagnose(info, path, events):
    mask = info["canonical"] == info["roles"][3]
    full, wall = info["full"], info["roles"][1]
    opened = full.copy()
    opened[mask] = info["roles"][0]
    no_a, _ = _trace(opened, wall, info["seed"], info["walls"])
    no_b, _ = _trace(full, wall, info["seed"], info["walls"], lateral=False)
    return {
        "hidden_trail": int(np.count_nonzero(path & mask)),
        "hidden_wall": int(np.count_nonzero((full == wall) & mask)),
        "hidden_unreached_open": int(np.count_nonzero((full != wall) & mask & ~path)),
        "a_difference": int(np.count_nonzero((path != no_a) & mask)),
        "b_difference": int(np.count_nonzero((path != no_b) & mask)),
        "hidden_collision": sum(bool(mask[r, c] and mask[r - 1, c])
                                for r, c in events["collisions"]),
        "hidden_direct": sum(bool(mask[r, c]) for r, c in events["direct"]),
        "hidden_redirected": sum(bool(mask[r, c]) for r, c in events["redirected"]),
        "hidden_layers": sum(bool(np.any(mask[r])) for r in info["walls"]),
    }


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _substrate(rng, profile):
    if profile == "repeat":
        pr = int(rng.integers(3, 6))
        offsets = [int(rng.integers(0, pr))]
    else:
        pr = int(rng.integers(5, 10))
        offsets = [0, int(rng.integers(2, pr - 1))]
        if profile == "free" and pr >= 8 and rng.random() < .4:
            offsets = [0, 3, 6]
    pc = int(rng.integers(3, 8))
    tile = np.zeros((pr, pc), dtype=np.int8)
    used = set()
    for r in offsets:
        # Disjoint aperture phases in adjacent wall types force rerouting.
        available = [c for c in range(pc) if c not in used]
        if not available:
            available = list(range(pc))
        count = 1 if profile == "repeat" else int(rng.integers(1, min(2, len(available), pc // 2) + 1))
        holes = set(map(int, rng.choice(available, count, replace=False)))
        tile[r] = 1
        tile[r, sorted(holes)] = 0
        used = holes
    # Only crop phases with two empty rows before the first wall are used.
    phases = [p for p in range(pr) if not np.any(tile[p])
              and not np.any(tile[(p + 1) % pr])]
    if not phases:
        return None
    phase = int(rng.choice(phases))
    depths = [h for h in range(max(12, 2 * pr + 1), 31)
              if not np.any(tile[(h - 1 + phase) % pr])]
    h = int(rng.choice(depths))
    w = int(rng.integers(max(13, 2 * pc + 1), 31))
    phase_c = int(rng.integers(pc))
    full = tile[(np.arange(h)[:, None] + phase) % pr,
                (np.arange(w)[None, :] + phase_c) % pc]
    walls = tuple(map(int, np.flatnonzero(np.any(full == 1, axis=1))))
    if len(walls) < 3:
        return None
    seed = int(rng.integers(1, w - 1))
    if profile == "repeat" and full[walls[0], seed] == 0:
        blocked = np.flatnonzero(full[walls[0]] == 1)
        blocked = blocked[(blocked > 0) & (blocked < w - 1)]
        seed = int(rng.choice(blocked))
    return full, walls, seed, pr, pc


def _make_pair(rng, profile, roles, direction, regime, shift, forbidden):
    bg, wall, signal, erased = roles
    for _ in range(350):
        substrate = _substrate(rng, profile)
        if substrate is None:
            continue
        binary, walls, seed, pr, _ = substrate
        full = np.where(binary == 0, bg, wall).astype(np.int8)
        path, events = _trace(full, wall, seed, walls)
        if not events["collisions"]:
            continue
        if profile == "repeat":
            first_holes = np.flatnonzero(full[walls[0]] != wall)
            left, right = first_holes[first_holes < seed], first_holes[first_holes > seed]
            # These conditions do not depend on erasure placement.  Reject
            # the substrate once instead of trying 28 impossible windows.
            if (not len(left) or not len(right)
                    or seed - left[-1] == right[0] - seed
                    or not np.any(~path[walls[0], first_holes])):
                continue
        h, w = full.shape
        if profile == "repeat":
            upper = min(7 if shift == "scale" and regime == "train" else 9, pr + 4)
            size = int(rng.integers(pr + 2, upper + 1))
        elif regime == "test" and shift == "scale":
            size = int(rng.integers(8, 11))
        else:
            size = int(rng.integers(4, 8))
        size = min(size, h - 2, w - 1)
        for _ in range(28):
            row, col = events["collisions"][int(rng.integers(len(events["collisions"])))]
            rmin, rmax = max(0, row - size + 1), min(row - 1, h - size)
            cmin, cmax = max(0, col - size + 1), min(col, w - size)
            if rmin > rmax or cmin > cmax:
                continue
            r0 = int(rng.integers(rmin, rmax + 1))
            c0 = int(rng.integers(cmin, cmax + 1))
            # Location shift holds out lateral-boundary erasures from training.
            if shift == "edge":
                if regime == "train" and (r0 == 0 or c0 == 0 or r0 + size == h or c0 + size == w):
                    continue
                if regime == "test":
                    c0 = 0 if col < w // 2 else w - size
            inp = full.copy()
            inp[r0:r0 + size, c0:c0 + size] = erased
            if inp[0, seed] == erased:
                continue
            inp[0, seed] = signal
            rotated = _orient(inp, direction, inverse=True)
            if _key(rotated) in forbidden:
                continue
            # Check the actual visible input, not construction metadata.
            try:
                info = _inspect(rotated)
            except ValueError:
                continue
            if not np.array_equal(info["full"], full):
                continue
            d = _diagnose(info, path, events)
            if not (d["hidden_trail"] >= 4 and d["hidden_wall"] >= 2
                    and d["hidden_unreached_open"] >= 2
                    and d["a_difference"] >= 2 and d["b_difference"] >= 2
                    and d["hidden_collision"] >= 1 and d["hidden_redirected"] >= 1):
                continue
            if profile == "repeat" and not (d["hidden_layers"] >= 2 and
                                              d["hidden_direct"] >= 1 and
                                              events["splits"] >= 1):
                continue
            out = inp.copy()
            out[path] = signal
            return {"input": rotated, "output": _orient(out, direction, inverse=True)}
    raise RuntimeError(f"bounded construction failed: {profile}/{regime}/{shift}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(10, 4, replace=False)))
    variable_palette = bool(rng.integers(2))
    shift = str(rng.choice(["scale", "edge", "orientation"]))
    anchor_direction = int(rng.integers(4))
    profiles = ["repeat", "stagger"] + [str(rng.choice(["repeat", "stagger", "free"]))
                                            for _ in range(nt - 2)]
    rng.shuffle(profiles)
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            profile = profiles[index] if split == "train" else str(rng.choice(["stagger", "free"]))
            if shift == "orientation":
                direction = anchor_direction if split == "train" else (anchor_direction + 1 + index) % 4
            else:
                direction = int(rng.integers(4))
            roles = tuple(map(int, rng.permutation(base))) if variable_palette else base
            if variable_palette and split == "train" and index < 2:
                roles = base if index == 0 else base[1:] + base[:1]
            pair = _make_pair(rng, profile, roles, direction, split, shift, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, demonstrations = set(), []
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in used
            used.add(_key(a))
            info = _inspect(a)
            infos[split].append(info)
            bg, wall, signal, erased = info["roles"]
            path, events = _trace(info["full"], wall, info["seed"], info["walls"])
            assert np.array_equal(out, _render(info, path))
            assert not np.any(path & (info["full"] == wall))
            changed = out != a
            assert np.all(out[changed] == signal)
            assert np.all((a[changed] == bg) | (a[changed] == erased))
            assert np.count_nonzero(out == signal) > 1
            assert np.count_nonzero(out == erased) >= 4
            d = _diagnose(info, path, events)
            assert d["a_difference"] >= 2 and d["b_difference"] >= 2
            assert d["hidden_trail"] >= 4 and d["hidden_unreached_open"] >= 2
            assert d["hidden_wall"] >= 2 and d["hidden_collision"] >= 1
            assert d["hidden_redirected"] >= 1
            if split == "train":
                demonstrations.append((info, events, d))
    def clear_two_sided_demo(info, events, d):
        holes = np.flatnonzero(info["full"][info["walls"][0]] != info["roles"][1])
        seed = info["seed"]
        left, right = holes[holes < seed], holes[holes > seed]
        path, _ = _trace(info["full"], info["roles"][1], seed, info["walls"])
        return (d["hidden_layers"] >= 2 and d["hidden_direct"] >= 1
                and events["splits"] >= 1 and len(left) and len(right)
                and seed - left[-1] != right[0] - seed
                and np.any(~path[info["walls"][0], holes]))
    assert any(clear_two_sided_demo(info, e, d) for info, e, d in demonstrations)
    # Staggered wall types add at least one repeated change of aperture phase.
    assert any(len({tuple(info["full"][r]) for r in info["walls"]}) >= 2
               for info, _, _ in demonstrations)
    train, test = infos["train"], infos["test"]
    scale_shift = min(i["box"][2] for i in test) > max(i["box"][2] for i in train)
    def interior(info):
        r, c, s = info["box"]
        h, w = info["full"].shape
        return r > 0 and c > 0 and r + s < h and c + s < w
    edge_shift = (all(interior(i) for i in train)
                  and all(i["box"][1] == 0 or i["box"][1] + i["box"][2] == i["full"].shape[1]
                          for i in test))
    train_dirs = {i["direction"] for i in train}
    orientation_shift = len(train_dirs) == 1 and test[0]["direction"] not in train_dirs
    assert scale_shift or edge_shift or orientation_shift
    if len({i["roles"] for i in train}) == 1:
        assert len({i["roles"] for i in train + test}) == 1


def _fingerprint(puzzle):
    return tuple((split, tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                              for p in puzzle[split])) for split in ("train", "test"))


def self_test():
    """Bounded checks; returns compact, reproducible diversity measurements."""
    # A small hand-worked route checks nearest-on-each-side, asymmetric split,
    # merging, boundary death, and a direct crossing that must not spread.
    fixture = np.zeros((10, 11), dtype=np.int8)
    for row, holes in ((2, [2, 7, 9]), (5, [4]), (8, [4, 8])):
        fixture[row] = 1
        fixture[row, holes] = 0
    expected = np.zeros_like(fixture, dtype=bool)
    expected[0, 4] = True
    expected[1, 2:8] = True
    expected[2:4, [2, 7]] = True
    expected[4] = True
    expected[5:, 4] = True
    actual, events = _trace(fixture, 1, 4, (2, 5, 8))
    assert np.array_equal(actual, expected)
    assert events["mergers"] == 1 and events["deaths"] == 2
    # These transformations change no geometric convention.  They exercise
    # the complete input-only role and direction inference, not just tracing.
    permutation = np.array([7, 0, 9, 2, 6, 1, 8, 4, 5, 3], dtype=np.int8)
    for pair in generate_puzzle(2026, 2, 1)["train"]:
        for k in range(4):
            inp = permutation[np.rot90(pair["input"], k)]
            out = permutation[np.rot90(pair["output"], k)]
            assert np.array_equal(solve(inp), out)
    shapes, periods, masks, directions, roles, layouts, signatures = (set() for _ in range(7))
    mergers = deaths = tests_larger = tests_edge = tests_new_direction = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        train_infos = [_inspect(p["input"]) for p in puzzle["train"]]
        max_size = max(i["box"][2] for i in train_infos)
        train_dirs = {i["direction"] for i in train_infos}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect(pair["input"])
                path, events = _trace(info["full"], info["roles"][1], info["seed"], info["walls"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(info["input"].shape)
                periods.add(info["periods"])
                masks.add(info["box"][2])
                directions.add(info["direction"])
                roles.add(info["roles"])
                layouts.add((info["full"].shape, info["periods"], info["box"],
                             info["seed"], tuple(map(tuple, info["tile"] == info["roles"][1]))))
                mergers += events["mergers"] > 0
                deaths += events["deaths"] > 0
                if split == "test":
                    r, c, s = info["box"]
                    tests_larger += s > max_size
                    tests_edge += c == 0 or c + s == info["full"].shape[1]
                    tests_new_direction += info["direction"] not in train_dirs
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 50 and len(periods) >= 15 and len(roles) >= 35
    assert len(masks) >= 6 and directions == {0, 1, 2, 3}
    assert mergers >= 30 and deaths >= 30
    assert tests_larger >= 5 and tests_edge >= 5 and tests_new_direction >= 5
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(0, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=210, shapes=len(shapes), periods=len(periods),
                mask_sizes=sorted(masks), palettes=len(roles), layouts=len(layouts),
                merging_pairs=mergers, boundary_termination_pairs=deaths,
                larger_mask_tests=tests_larger, lateral_edge_tests=tests_edge,
                new_direction_tests=tests_new_direction)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
