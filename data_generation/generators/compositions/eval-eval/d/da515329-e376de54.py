"""Three pinwheel runners synchronize to the runner opposite the vacant gate.

All conventions are visible in the input. No parent files or saved examples are
read by this module. Running this file executes a bounded in-memory self-test.
"""
from __future__ import annotations

from collections import Counter, deque
from itertools import islice, permutations
from math import isqrt

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer 2D color grid, sides 1..30")
    return a


def _route(radius, arm, hand, length):
    """An ordered prefix of one infinite-plane parent-A pinwheel phase.

    Positions are counted once each, excluding the hollow center. Segments
    continue outside any canvas; fitting is checked separately, never clipped.
    """
    y = x = 0
    dy, dx = DIRS[arm]
    cells = []

    def walk(vy, vx, steps):
        nonlocal y, x
        for _ in range(min(steps, length - len(cells))):
            y, x = y + vy, x + vx
            cells.append((y, x))

    walk(dy, dx, radius)
    dy, dx = hand * dx, -hand * dy
    stage = 0
    while len(cells) < length:
        walk(dy, dx, 2 + 4 * stage)
        walk(hand * dx, -hand * dy, radius - 2)
        walk(dy, dx, radius)
        dy, dx = hand * dx, -hand * dy
        stage += 1
    return cells


def _formula_route(radius, arm, hand, length):
    """Index-to-cell cross-check using closed-form starts, not a path walk."""
    result = []
    for index in range(length):
        if index < radius:
            y, x = -index - 1, 0
        else:
            offset = index - radius
            # Stage k starts after 2*k*(k+radius-1) post-spoke steps.
            k = (isqrt((radius - 1) ** 2 + 2 * offset) - radius + 1) // 2
            step = offset - 2 * k * (k + radius - 1) + 1
            q, phase = divmod(k, 4)
            y, x = ((-radius - 8*q, -8*q),
                    (-2 - 8*q, radius + 8*q + 2),
                    (radius + 8*q + 4, 8*q + 4),
                    (8*q + 6, -radius - 8*q - 6))[phase]
            dy, dx = ((0, 1), (1, 0), (0, -1), (-1, 0))[phase]
            forward, jog = 2 + 4*k, radius - 2
            along = min(step, forward) + max(0, step - forward - jog)
            across = min(max(0, step - forward), jog)
            y, x = y + dy*along + dx*across, x + dx*along - dy*across
        x *= hand
        for _ in range(arm):
            y, x = x, -y
        result.append((y, x))
    return result


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, cells = deque([start]), {start}
        while queue:
            r, c = queue.popleft()
            for dy, dx in DIRS:
                p = r + dy, c + dx
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    queue.append(p)
        result.append(cells)
    return result


def _ordered_component(cells, start):
    ordered, previous, current = [], None, start
    while current is not None:
        ordered.append(current)
        r, c = current
        next_cells = [(r+dy, c+dx) for dy, dx in DIRS
                      if (r+dy, c+dx) in cells and (r+dy, c+dx) != previous]
        if len(next_cells) > 1 or len(ordered) > len(cells):
            raise ValueError("runner must be a simple path")
        previous, current = current, next_cells[0] if next_cells else None
    if len(ordered) != len(cells):
        raise ValueError("runner is not a single path")
    return ordered


def _analyze(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() <= a.size // 2:
        raise ValueError("background must occupy more than half the input")
    components = _components(a != bg)
    if len(components) != 3:
        raise ValueError("expected exactly three disconnected runners")
    endpoints = {}
    colors = []
    for component_id, cells in enumerate(components):
        palette = {int(a[p]) for p in cells}
        ends = []
        for r, c in cells:
            degree = sum((r+dy, c+dx) in cells for dy, dx in DIRS)
            if degree == 1:
                ends.append((r, c))
            elif degree != 2:
                raise ValueError("runner branches or is a singleton")
        if len(palette) != 1 or len(ends) != 2:
            raise ValueError("runner must be monochrome with two endpoints")
        colors.append(palette.pop())
        endpoints.update((p, component_id) for p in ends)
    possible_hubs = {(r+dy, c+dx) for r, c in endpoints for dy, dx in DIRS}
    candidates = []
    for cy, cx in sorted(possible_hubs):
        if not (0 <= cy < a.shape[0] and 0 <= cx < a.shape[1]) or a[cy, cx] != bg:
            continue
        neighbors = [(cy+dy, cx+dx) for dy, dx in DIRS]
        ids = [endpoints.get(p) for p in neighbors]
        active = [arm for arm, component_id in enumerate(ids) if component_id is not None]
        if len(active) != 3 or len({ids[arm] for arm in active}) != 3:
            continue
        missing = next(arm for arm in range(4) if arm not in active)
        p = neighbors[missing]
        if not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]) or a[p] != bg:
            continue
        paths, radii, hands, payload = {}, set(), set(), {}
        valid = True
        for arm in active:
            path = _ordered_component(components[ids[arm]], neighbors[arm])
            dy, dx = DIRS[arm]
            radius = 0
            while radius < len(path) and path[radius] == (cy+(radius+1)*dy, cx+(radius+1)*dx):
                radius += 1
            if not 2 <= radius <= 5 or len(path) < radius + 3:
                valid = False
                break
            bend = (path[radius][0]-path[radius-1][0], path[radius][1]-path[radius-1][1])
            if bend == (dx, -dy):
                hand = 1
            elif bend == (-dx, dy):
                hand = -1
            else:
                valid = False
                break
            expected = [(cy+y, cx+x) for y, x in _route(radius, arm, hand, len(path))]
            if path != expected:
                valid = False
                break
            paths[arm], payload[arm] = path, colors[ids[arm]]
            radii.add(radius)
            hands.add(hand)
        if valid and len(radii) == len(hands) == 1:
            reference = (missing + 2) % 4
            candidates.append({"background": bg, "hub": (cy, cx), "missing": missing,
                               "radius": radii.pop(), "hand": hands.pop(),
                               "paths": paths, "colors": payload, "reference": reference,
                               "lengths": {arm: len(path) for arm, path in paths.items()},
                               "target": len(paths[reference])})
    if len(candidates) != 1:
        raise ValueError("input must reveal one hollow hub, radius and handedness")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Equalize route distances to the ordinal middle, using the input alone."""
    a = _grid(input_grid)
    info = _analyze(a)
    out = np.full(a.shape, info["background"], dtype=np.int8)
    cy, cx = info["hub"]
    for arm in info["paths"]:
        for y, x in _route(info["radius"], arm, info["hand"], info["target"]):
            r, c = cy+y, cx+x
            if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
                raise ValueError("normalized runner would leave the canvas")
            if out[r, c] != info["background"]:
                raise ValueError("normalized routes intersect")
            out[r, c] = info["colors"][arm]
    return out


def _capacity(radius):
    # Quarter turns/reflection do not change the maximum bounding-box side.
    routes = [_route(radius, arm, 1, 180) for arm in (0, 1, 2)]
    low, high = np.array([0, 0]), np.array([0, 0])
    for index in range(180):
        points = np.array([route[index] for route in routes])
        low, high = np.minimum(low, points.min(axis=0)), np.maximum(high, points.max(axis=0))
        if max(high - low + 1) > 30:
            return index
    raise AssertionError("capacity bound too small")


def _target_between(rng, low, high, radius, avoid=()):
    candidates = []
    route = _route(radius, 0, 1, high+1)
    for length in range(low, high+1):
        # A reference ending inside a run distinguishes cell distance from
        # merely copying a number of turns or stopping at the next corner.
        incoming = np.subtract(route[length-1], route[length-2])
        outgoing = np.subtract(route[length], route[length-1])
        if length not in avoid and np.array_equal(incoming, outgoing):
            candidates.append(length)
    if not candidates:
        raise AssertionError("target interval has no interior-run length")
    return int(rng.choice(candidates))


def _make_pair(rng, radius, hand, missing, target, ceiling, mode, padding, mono, teach):
    floor = radius + 3
    reference = (missing + 2) % 4
    others = [arm for arm in range(4) if arm not in (missing, reference)]
    if rng.integers(2):
        others.reverse()
    third_stage = radius + 6*(radius+2)

    def short(early=False):
        return int(rng.integers(floor, min(target-1, 3*radius) + 1 if early else target))

    def long():
        return int(rng.integers(max(target+1, third_stage+2) if teach else target+1, ceiling+1))

    lengths = {reference: target, others[0]: target, others[1]: target}
    if mode == "mixed":
        lengths[others[0]], lengths[others[1]] = short(teach), long()
    elif mode == "extend":
        lengths[others[0]], lengths[others[1]] = short(), short()
    elif mode == "trim":
        lengths[others[0]], lengths[others[1]] = long(), long()
    elif mode == "one_extend":
        lengths[others[0]] = short()
    elif mode == "one_trim":
        lengths[others[0]] = long()
    else:
        raise AssertionError("unknown correction mode")
    input_paths = {arm: _route(radius, arm, hand, length) for arm, length in lengths.items()}
    output_paths = {arm: _formula_route(radius, arm, hand, target) for arm in lengths}
    union = np.array([(0, 0)] + [p for paths in (input_paths, output_paths)
                               for path in paths.values() for p in path])
    low, high = union.min(axis=0), union.max(axis=0)
    box = high - low + 1
    margins = [int(rng.integers(min(padding, 30-int(side)), min(8, 30-int(side))+1))
               if padding else 0 for side in box]
    shape = tuple(int(side+extra) for side, extra in zip(box, margins))
    offset = np.array([int(rng.integers(extra+1)) for extra in margins]) - low
    palette = [int(c) for c in rng.permutation(10)]
    bg = palette[0]
    active = sorted(lengths)
    if mono:
        colors = dict.fromkeys(active, palette[1])
    elif rng.random() < .5:
        colors = {arm: palette[1+i] for i, arm in enumerate(active)}
    else:
        assignment = [palette[1], palette[1], palette[2]]
        rng.shuffle(assignment)
        colors = dict(zip(active, assignment))
    pair = {}
    for role, paths in (("input", input_paths), ("output", output_paths)):
        grid = np.full(shape, bg, dtype=np.int8)
        for arm, path in paths.items():
            for y, x in path:
                r, c = int(y+offset[0]), int(x+offset[1])
                assert grid[r, c] == bg
                grid[r, c] = colors[arm]
        pair[role] = grid
    return pair


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _ensure_unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    roles = [int(c) for c in np.unique(pair["input"])]
    # More injective recolorings than previously used inputs: finite fallback,
    # no random retry loop, and at most 15 small candidates alive sequentially.
    for palette in islice(permutations(range(10), len(roles)), len(used)+1):
        mapping = np.arange(10, dtype=np.int8)
        mapping[roles] = palette
        candidate = {role: mapping[grid] for role, grid in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise AssertionError("injective recoloring pigeonhole failure")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    radii = [int(rng.integers(2, 6))]
    radii.append(int(rng.choice([r for r in range(2, 6) if r != radii[0]])))
    radii += [int(rng.integers(2, 6)) for _ in range(ntrain-2)]
    base_missing, base_hand = int(rng.integers(4)), int(rng.choice([-1, 1]))
    mixed_hands = bool(rng.integers(2))
    shift = str(rng.choice(("progress", "rotate", "crop")))
    train_ceiling = int(rng.integers(60, 73))
    puzzle, used, train_info = {"train": [], "test": []}, set(), []
    for index in range(ntrain+ntest):
        training = index < ntrain
        local = index if training else index-ntrain
        radius = radii[index] if training else int(rng.choice(radii))
        hand = base_hand * (-1 if mixed_hands and local % 2 else 1)
        missing = (base_missing + (2*(local % 2) if training or shift == "rotate" else int(rng.integers(4)))) % 4
        if not training and shift == "rotate":
            missing = (base_missing + 1 + 2*(local % 2)) % 4
        mode = ("mixed" if local == 0 else str(rng.choice(("extend", "trim"))) if local == 1
                else str(rng.choice(("mixed", "extend", "trim", "one_extend", "one_trim"))))
        ceiling = train_ceiling if training else _capacity(radius)
        lower = radius + 2*(2*(radius+1)) + 2  # two stages, then at least two cells
        if training and local == 1 and mode == "extend":
            lower = radius + 6*(radius+2) + 2
        avoid = (train_info[0]["target"],) if training and local == 1 else ()
        if not training and shift == "progress":
            lower = max(max(a["lengths"].values()) for a in train_info) + 4
        upper = ceiling-4
        if not training and shift != "progress":
            upper = min(upper, train_ceiling-4)
        target = _target_between(rng, lower, upper, radius, avoid)
        pair = _make_pair(rng, radius, hand, missing, target, ceiling, mode,
                          4 if training and shift == "crop" else 0 if not training and shift == "crop" else 1,
                          mono=(training and local == 0) or (local >= 2 and rng.random() < .3),
                          teach=training and local < 2)
        pair = _ensure_unique(pair, used)
        used.add(_key(pair["input"]))
        puzzle["train" if training else "test"].append(pair)
        if training:
            train_info.append(_analyze(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train/test keys")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            bg, target = info["background"], info["target"]
            cy, cx = info["hub"]
            assert b[cy, cx] == bg and np.count_nonzero(b != bg) == 3*target
            assert set(np.unique(b)) == set(np.unique(a))
            assert not np.array_equal(a, b)
            out_info = _analyze(b)
            for field in ("background", "hub", "missing", "radius", "hand", "colors"):
                assert out_info[field] == info[field], field
            assert set(out_info["lengths"].values()) == {target}
            for arm, old_path in info["paths"].items():
                new_path = out_info["paths"][arm]
                common = min(len(old_path), target)
                assert old_path[:common] == new_path[:common]
                assert all(a[p] == b[p] == info["colors"][arm] for p in old_path[:common])
                assert all(b[p] == bg for p in old_path[target:])
                formula = [(cy+y, cx+x) for y, x in _formula_route(info["radius"], arm, info["hand"], target)]
                assert new_path == formula, "incorrect winding continuation"
            ref_path = info["paths"][info["reference"]]
            assert out_info["paths"][info["reference"]] == ref_path
            infos[split].append(info)
    train = infos["train"]
    assert train[0]["radius"] != train[1]["radius"]
    assert train[0]["target"] != train[1]["target"]
    first_lengths, first_target = train[0]["lengths"].values(), train[0]["target"]
    assert min(first_lengths) < first_target < max(first_lengths)
    assert any(sorted(i["lengths"].values())[1] != i["target"] for i in train)
    assert len(set(train[0]["colors"].values())) == 1
    assert len(set(train[1]["colors"].values())) >= 2
    assert all(i["radius"] in {t["radius"] for t in train} for i in infos["test"])
    # Each returned puzzle has an observable held-out demand: more route
    # progress than any training runner, a new vacant-gate axis, or zero spare
    # canvas around the input/output union versus padded demonstrations.
    max_train = max(max(i["lengths"].values()) for i in train)
    missing_train = {i["missing"] for i in train}
    def tight(pair):
        i = _analyze(pair["input"])
        points = np.argwhere((np.asarray(pair["input"]) != i["background"]) |
                             (np.asarray(pair["output"]) != i["background"]))
        return tuple(points.max(axis=0)-points.min(axis=0)+1) == np.asarray(pair["input"]).shape
    assert (all(i["target"] > max_train for i in infos["test"])
            or all(i["missing"] not in missing_train for i in infos["test"])
            or (all(tight(p) for p in puzzle["test"]) and not any(tight(p) for p in puzzle["train"])))


def self_test():
    """42 streamed seeds, all legal counts, metamorphic and negative checks.

    No artifacts or traces are written. Only one puzzle and its comparison
    copy are retained; arrays contain at most 30*30 cells each.
    """
    from hashlib import sha256

    for radius in range(2, 6):
        for arm in range(4):
            for hand in (-1, 1):
                assert _route(radius, arm, hand, 180) == _formula_route(radius, arm, hand, 180)
    boundary_members = 0
    control_rng = np.random.default_rng(11083)
    for radius in range(2, 6):
        for hand in (-1, 1):
            for vacancy in range(4):
                for target in (radius+3, 3*radius, 5*radius+4):
                    pair = _make_pair(control_rng, radius, hand, vacancy, target,
                                      target+7, "trim", 1, False, False)
                    assert np.array_equal(solve(pair["input"]), pair["output"])
                    assert np.array_equal(solve(pair["output"]), pair["output"])
                    boundary_members += 1
    shapes, structural, targets, palettes = set(), set(), set(), set()
    radii, hands, missing, modes, shifts = set(), set(), set(), Counter(), Counter()
    signatures = set()
    pair_count = palette_checks = symmetry_checks = reference_checks = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = sha256()
        tinfo = [_analyze(p["input"]) for p in puzzle["train"]]
        vinfo = [_analyze(p["input"]) for p in puzzle["test"]]
        max_train = max(max(i["lengths"].values()) for i in tinfo)
        shifts["progress"] += int(all(i["target"] > max_train for i in vinfo))
        shifts["rotate"] += int(all(i["missing"] not in {t["missing"] for t in tinfo} for i in vinfo))
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                a, b = pair["input"], pair["output"]
                info = _analyze(a)
                assert np.array_equal(solve(a), b)
                assert np.array_equal(solve(b), b), "normalization must be idempotent"
                radius, hand = info["radius"], info["hand"]
                shapes.add(a.shape)
                radii.add(radius)
                hands.add(hand)
                missing.add(info["missing"])
                targets.add(info["target"])
                palettes.add((info["background"], tuple(info["colors"].values())))
                order = [(info["missing"]+j) % 4 for j in (1, 2, 3)]
                lengths = tuple(info["lengths"][arm] for arm in order)
                structural.add((radius, hand, info["missing"], lengths))
                delta = [info["target"]-length for length in lengths]
                modes["mixed" if min(delta) < 0 < max(delta) else "extend" if max(delta) > 0 else "trim"] += 1
                modes["one_change"] += int(sum(d != 0 for d in delta) == 1)
                pair_count += 1
                digest.update(repr(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                if index == 0:
                    for turns in range(4):
                        for reflect in (False, True):
                            left, right = np.rot90(a, turns), np.rot90(b, turns)
                            if reflect:
                                left, right = np.fliplr(left), np.fliplr(right)
                            assert np.array_equal(solve(left), right)
                            symmetry_checks += 1
                    palette = np.random.default_rng(seed+831).permutation(10)
                    assert np.array_equal(solve(palette[a]), palette[b])
                    palette_checks += 1
        # Both omission tests use the actual first demonstration.
        info = tinfo[0]
        assert len(set(info["lengths"].values())) > 1  # keeping A, omit B = leave prefixes
        short_arm = min(info["lengths"], key=info["lengths"].get)
        short = info["paths"][short_arm]
        tangent = np.subtract(short[-1], short[-2])
        added = _analyze(puzzle["train"][0]["output"])["paths"][short_arm][len(short):]
        straight = [tuple(np.array(short[-1])+step*tangent) for step in range(1, len(added)+1)]
        assert added != straight  # keeping B, omit A = straight free-end extension
        altered = puzzle["train"][0]["input"].copy()
        reference_path = info["paths"][info["reference"]]
        altered[reference_path[-1]] = info["background"]
        changed = solve(altered)
        assert _analyze(changed)["target"] == info["target"]-1
        assert not np.array_equal(changed, puzzle["train"][0]["output"])
        reference_checks += 1
        signatures.add(digest.hexdigest())
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in p)
            del again
        del puzzle, pair, a, b, left, right, altered, changed
    assert len(signatures) == 42 and len(structural) >= 180
    assert len(shapes) >= 50 and len(targets) >= 30 and len(palettes) >= 100
    assert radii == {2, 3, 4, 5} and hands == {-1, 1} and missing == {0, 1, 2, 3}
    assert all(modes[k] > 0 for k in ("mixed", "extend", "trim", "one_change"))
    assert shifts["progress"] > 0 and shifts["rotate"] > 0
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            p = generate_puzzle(1901+ntrain*10+ntest, ntrain, ntest)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
            del p
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0)
    bg = _analyze(bad["test"][0]["input"])["background"]
    point = tuple(np.argwhere(bad["test"][0]["output"] != bg)[0])
    bad["test"][0]["output"][point] = bg
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "route_configurations": len(structural),
            "targets": len(targets), "palettes": len(palettes),
            "radii": sorted(radii), "hands": sorted(hands), "vacancies": sorted(missing),
            "patterns": dict(modes), "observed_shifts": dict(shifts),
            "palette_checks": palette_checks, "symmetry_checks": symmetry_checks,
            "reference_perturbations": reference_checks,
            "short_and_corner_members": boundary_members,
            "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
