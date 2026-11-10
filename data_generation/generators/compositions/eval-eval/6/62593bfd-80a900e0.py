"""Shape portals: transfer side caps through rotation-equivalent badge frames.

Only Python's standard library and NumPy are used. Generation retains one puzzle
at a time, writes no files, and has bounded placement/uniqueness searches.
"""
from hashlib import sha256

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 603733519, 452690152, 363224456, 717407278, 895262372,
              879011449, 797587436, 927651515, 926490610, 556096725,
              412479986, 728814841, 441589838, 294744478, 583384971,
              871031885, 602141105, 270317873, 1905494, 668571653,
              179369080, 668203795, 70713516, 532906849, 364212074,
              530248709, 362369251, 458248465, 889364357, 369248338,
              944840797, 140446466, 324159156, 316016808, 594002991,
              150584605, 513130707, 743558124, 906557708, 902778131,
              683263880)


def _bits(mask):
    return sum(int(v) << i for i, v in enumerate(mask.flat))


def _mask(bits):
    return np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)


def _canonical(mask):
    return min(_bits(np.rot90(mask, k)) for k in range(4))


def _pose(mask):
    key = _canonical(mask)
    turns = [k for k in range(4) if np.array_equal(np.rot90(_mask(key), k), mask)]
    assert len(turns) == 1, "a badge must have a unique rotational pose"
    return key, turns[0]


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {min(cells)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == cells


def _legal_badge(mask):
    return (mask.shape == (3, 3) and 4 <= int(mask.sum()) <= 7
            and mask.any(axis=0).all() and mask.any(axis=1).all()
            and _connected(mask)
            and len({_bits(np.rot90(mask, k)) for k in range(4)}) == 4)


BADGES = tuple(b for b in range(1, 512)
               if _legal_badge(_mask(b)) and _canonical(_mask(b)) == b)
CHIRAL = tuple(b for b in BADGES if _canonical(np.fliplr(_mask(b))) != b)


def _turn(direction, k):
    # np.rot90 maps (dr,dc) to (-dc,dr), i.e. NW to SW.
    return (int(direction) - int(k)) % 4


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _ring(center, radius):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius}


def _cap(center, radius, direction):
    dr, dc = DIRECTIONS[direction]
    r, c = center
    return [(r + dr * (k + 1), c + dc * (radius - k + 1))
            for k in range(radius + 1)]


def _paths(center, radius, direction, shape):
    cap = _cap(center, radius, direction)
    dr, dc = DIRECTIONS[direction]
    paths = []
    for r, c in (cap[0], cap[-1]):
        cells = []
        r, c = r + dr, c + dc
        while _inside((r, c), shape):
            cells.append((r, c))
            r, c = r + dr, c + dc
        paths.append(cells)
    return paths


def _background(shape, colors):
    return np.array(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _infer_background(grid):
    parity = np.indices(grid.shape).sum(axis=0) % 2
    colors = []
    for p in (0, 1):
        values = grid[parity == p]
        counts = np.bincount(values.astype(int), minlength=10)
        color = int(counts.argmax())
        assert int(counts[color]) * 2 > len(values), "checkerboard parity must have a majority"
        colors.append(color)
    assert colors[0] != colors[1]
    return tuple(colors)


def _grid(value):
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape)
    assert np.issubdtype(array.dtype, np.integer)
    assert np.all((0 <= array) & (array <= 9))
    return array


def _analyze(input_grid):
    """Recover checkerboard, housings, badge poses and sources from pixels."""
    grid = _grid(input_grid)
    h, w = grid.shape
    colors = _infer_background(grid)
    base = _background(grid.shape, colors)
    objects = []
    for radius in (3, 4):
        if min(h, w) < 2 * radius + 1:
            continue
        shape = h - 2 * radius, w - 2 * radius
        offsets = sorted(_ring((0, 0), radius))
        dr, dc = offsets[0]
        candidate_colors = grid[radius + dr:radius + dr + shape[0],
                                radius + dc:radius + dc + shape[1]]
        valid = (candidate_colors != colors[0]) & (candidate_colors != colors[1])
        for dr, dc in offsets[1:]:
            valid &= grid[radius + dr:radius + dr + shape[0],
                          radius + dc:radius + dc + shape[1]] == candidate_colors
        for rr, cc in np.argwhere(valid):
            r, c = int(rr) + radius, int(cc) + radius
            patch = grid[r - 1:r + 2, c - 1:c + 2]
            local_base = base[r - 1:r + 2, c - 1:c + 2]
            badge = patch != local_base
            assert _legal_badge(badge), "invalid badge inside housing"
            badge_colors = set(map(int, patch[badge]))
            assert len(badge_colors) == 1
            badge_color = badge_colors.pop()
            housing = int(candidate_colors[rr, cc])
            assert badge_color not in (*colors, housing)
            key, pose = _pose(badge)
            obj = dict(center=(r, c), radius=radius, housing=housing,
                       badge=badge.copy(), badge_color=badge_color,
                       key=key, pose=pose, caps={})
            for d in range(4):
                cells = _cap((r, c), radius, d)
                if not all(_inside(p, grid.shape) for p in cells):
                    continue
                values = {int(grid[p]) for p in cells}
                if len(values) == 1:
                    color = values.pop()
                    if color not in (*colors, housing, badge_color):
                        obj["caps"][d] = color
            objects.append(obj)
    objects.sort(key=lambda o: o["center"])
    assert objects, "no portal housings"
    covered = set()
    for i, obj in enumerate(objects):
        for other in objects[:i]:
            distance = max(abs(a - b) for a, b in zip(obj["center"], other["center"]))
            assert distance > obj["radius"] + other["radius"] + 1, "housing boxes must be separated"
        r, c = obj["center"]
        cells = _ring((r, c), obj["radius"])
        cells |= {(r - 1 + int(y), c - 1 + int(x)) for y, x in np.argwhere(obj["badge"])}
        for d in obj["caps"]:
            cells.update(_cap((r, c), obj["radius"], d))
        assert not covered & cells
        covered.update(cells)
    assert covered == set(map(tuple, np.argwhere(grid != base))), "unexplained foreground"
    groups = {}
    for obj in objects:
        groups.setdefault(obj["key"], []).append(obj)
    for members in groups.values():
        count = sum(bool(obj["caps"]) for obj in members)
        assert (count == 1 if len(members) > 1 else count <= 1), "one source per repeated badge class"
    return dict(background=colors, base=base, objects=objects, groups=groups)


def _arrivals(analysis):
    for members in analysis["groups"].values():
        source = next((obj for obj in members if obj["caps"]), None)
        if source is None:
            continue
        for receiver in members:
            if receiver is source:
                continue
            delta = (receiver["pose"] - source["pose"]) % 4
            for direction, color in sorted(source["caps"].items()):
                yield source, receiver, _turn(direction, delta), color


def _predict(grid, analysis):
    output = grid.copy()
    claimed = set()
    for source, receiver, direction, color in _arrivals(analysis):
        assert color not in (receiver["housing"], receiver["badge_color"], *analysis["background"])
        cells = _cap(receiver["center"], receiver["radius"], direction)
        assert all(_inside(p, grid.shape) for p in cells), "received caps must be complete"
        for path in _paths(receiver["center"], receiver["radius"], direction, grid.shape):
            cells += path
        assert not claimed.intersection(cells), "signals may not overlap"
        for p in cells:
            assert grid[p] == analysis["base"][p], "signal hits original foreground"
            output[p] = color
        claimed.update(cells)
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; neither seed nor generation state is read."""
    grid = _grid(input_grid)
    return _predict(grid, _analyze(grid))


def _closed_form(grid, analysis):
    """A second geometry check, sharing parsing but not endpoint walking."""
    rows, cols = np.indices(grid.shape)
    output = grid.copy()
    used = np.zeros(grid.shape, dtype=bool)
    for _, receiver, direction, color in _arrivals(analysis):
        dr, dc = DIRECTIONS[direction]
        u = (rows - receiver["center"][0]) * dr
        v = (cols - receiver["center"][1]) * dc
        cap = (u >= 1) & (v >= 1) & (u + v == receiver["radius"] + 2)
        rays = (np.abs(u - v) == receiver["radius"]) & (np.minimum(u, v) >= 2)
        mask = cap | rays
        assert not np.any(used & mask)
        assert np.all(grid[mask] == analysis["base"][mask])
        output[mask] = color
        used |= mask
    return output


def _choose_badge(rng, forbidden=(), chiral=False):
    choices = [b for b in (CHIRAL if chiral else BADGES) if b not in forbidden]
    assert choices, "badge vocabulary exhausted"
    return int(rng.choice(choices))


def _specs(rng, kind, taught_directions=None, forbidden=()):
    """Describe classes and local frames, without selecting any positions."""
    if kind == "mirror":
        key = _choose_badge(rng, forbidden, chiral=True)
        keys = [key, key, _canonical(np.fliplr(_mask(key)))]
        source_turn = int(rng.integers(4))
        target_turn = (source_turn + int(rng.choice([1, 3]))) % 4
        turns = [source_turn, target_turn, int(rng.integers(4))]
        group_ids = [0, 0, 1]
        source_dirs = [_turn(d, -target_turn) for d in taught_directions]
        # A blank reflected singleton is an explicit negative receiver.
        signals = {0: source_dirs}
        radii = [3, 4, int(rng.integers(3, 5))]
        if rng.random() < .5:
            radii[0], radii[1] = radii[1], radii[0]
    elif kind == "triple":
        key = _choose_badge(rng, forbidden)
        mirror = _canonical(np.fliplr(_mask(key)))
        single = _choose_badge(rng, set(forbidden) | {key, mirror})
        keys, group_ids = [key, key, key, single], [0, 0, 0, 1]
        canonical_side = int(rng.integers(4))
        turns = [int(rng.integers(4))]
        turns += [(canonical_side - d) % 4 for d in taught_directions]
        turns += [int(rng.integers(4))]
        signals = {0: [canonical_side], 3: list(map(int, rng.choice(4, int(rng.integers(1, 4)), replace=False)))}
        radii = [int(rng.integers(3, 5)), 3, 4, int(rng.integers(3, 5))]
        if rng.random() < .5:
            radii[1], radii[2] = radii[2], radii[1]
    else:
        profile = [2, 2] if kind == "two_pairs" else [list(x) for x in
            ((2,), (2, 1), (3,), (2, 2), (3, 1), (2, 1, 1))][int(rng.integers(6))]
        keys, turns, group_ids, signals = [], [], [], {}
        banned = set(forbidden)
        for group, size in enumerate(profile):
            key = _choose_badge(rng, banned)
            banned.add(key)
            start = len(keys)
            keys += [key] * size
            turns += list(map(int, rng.integers(4, size=size)))
            group_ids += [group] * size
            count = int(rng.integers(1, (5 if size == 2 else 3)))
            if size == 1 and rng.random() < .3:
                count = 0
            signals[start] = list(map(int, rng.choice(4, count, replace=False)))
        radii = list(map(int, rng.integers(3, 5, size=len(keys))))
    specs = []
    for i, (key, turn, radius, group) in enumerate(zip(keys, turns, radii, group_ids)):
        specs.append(dict(key=key, turn=turn, radius=radius, group=group,
                          badge=np.rot90(_mask(key), turn).copy(),
                          canonical_caps={d: None for d in signals.get(i, [])},
                          incoming=[]))
    return specs


def _colors(rng, specs, teaching):
    palette = list(map(int, rng.permutation(10)))
    background = tuple(palette[:2])
    nh = int(rng.integers(1, 3))
    nb = 2 if teaching else int(rng.integers(1, 4))
    housing_bank = palette[2:2 + nh]
    badge_bank = palette[2 + nh:2 + nh + nb]
    payload_bank = palette[2 + nh + nb:]
    for i, obj in enumerate(specs):
        obj["housing"] = int(rng.choice(housing_bank))
        obj["badge_color"] = int(rng.choice(badge_bank))
    if teaching:
        specs[0]["badge_color"], specs[1]["badge_color"] = badge_bank
        specs[-1]["badge_color"] = badge_bank[0]
    first_payload = int(rng.choice(payload_bank))
    for i, obj in enumerate(specs):
        bank = [int(rng.choice(payload_bank))] if rng.random() < .45 else payload_bank
        obj["canonical_caps"] = {d: int(rng.choice(bank)) for d in obj["canonical_caps"]}
        if teaching and obj["canonical_caps"]:
            first = next(iter(obj["canonical_caps"]))
            obj["canonical_caps"][first] = first_payload
        if teaching and i == 0 and len(obj["canonical_caps"]) >= 2:
            # Distinguish carrying each cap's color from using one source color.
            sides = list(obj["canonical_caps"])
            obj["canonical_caps"][sides[1]] = next(c for c in payload_bank if c != first_payload)
        obj["caps"] = {_turn(d, obj["turn"]): color for d, color in obj["canonical_caps"].items()}
    for source in specs:
        if not source["caps"]:
            continue
        for target in specs:
            if target is not source and target["group"] == source["group"]:
                target["incoming"] = [(_turn(d, target["turn"]), color)
                                      for d, color in source["canonical_caps"].items()]
    return background


def _footprint(obj, center, shape, teaching):
    r, c = center
    cells = _ring(center, obj["radius"])
    cells |= {(r - 1 + int(y), c - 1 + int(x)) for y, x in np.argwhere(obj["badge"])}
    rays = set()
    for direction in obj["caps"]:
        cells.update(_cap(center, obj["radius"], direction))
    probes = set()
    for direction, _ in obj["incoming"]:
        cap = _cap(center, obj["radius"], direction)
        cells.update(cap)
        paths = _paths(center, obj["radius"], direction, shape)
        if teaching and min(map(len, paths)) < 2:
            return None
        for path in paths:
            rays.update(path)
        if teaching:
            dr, dc = DIRECTIONS[direction]
            probes.update((y + dr * step, x + dc * step)
                          for y, x in cap[1:-1] for step in (1, 2))
    if not all(_inside(p, shape) for p in cells | probes):
        return None
    return cells, rays, probes


def _place(rng, specs, shape, teaching):
    """Free-center bounded backtracking with exact complete signal footprints."""
    candidates = []
    for obj in specs:
        radius = obj["radius"]
        options = []
        for r in range(radius + 1, shape[0] - radius - 1):
            for c in range(radius + 1, shape[1] - radius - 1):
                result = _footprint(obj, (r, c), shape, teaching)
                if result is not None:
                    options.append(((r, c), result))
        if not options:
            return False
        order = rng.permutation(len(options))
        candidates.append([options[int(i)] for i in order])
    order = sorted(range(len(specs)), key=lambda i: (-len(specs[i]["incoming"]), -specs[i]["radius"]))
    budget = [14000]
    selected = {}

    def visit(step, occupied, forbidden_probes):
        if step == len(order):
            return True
        i = order[step]
        for center, (cells, rays, probes) in candidates[i]:
            budget[0] -= 1
            if budget[0] < 0:
                return False
            if (cells | rays) & (occupied | forbidden_probes) or probes & occupied:
                continue
            if any(max(abs(a - b) for a, b in zip(center, pos)) <=
                   specs[i]["radius"] + specs[j]["radius"] + 1
                   for j, pos in selected.items()):
                continue
            selected[i] = center
            if visit(step + 1, occupied | cells | rays, forbidden_probes | probes):
                return True
            selected.pop(i)
        return False

    if not visit(0, set(), set()):
        return False
    for i, obj in enumerate(specs):
        obj["center"] = selected[i]
    return True


def _render(shape, background, specs):
    inp = _background(shape, background)
    for obj in specs:
        r, c = obj["center"]
        for p in _ring((r, c), obj["radius"]):
            inp[p] = obj["housing"]
        for y, x in np.argwhere(obj["badge"]):
            inp[r - 1 + y, c - 1 + x] = obj["badge_color"]
        for direction, color in obj["caps"].items():
            for p in _cap((r, c), obj["radius"], direction):
                inp[p] = color
    out = inp.copy()
    # Labels come from the sampled class/pose description, not solve().
    for obj in specs:
        for direction, color in obj["incoming"]:
            cells = _cap(obj["center"], obj["radius"], direction)
            for path in _paths(obj["center"], obj["radius"], direction, shape):
                cells += path
            for p in cells:
                out[p] = color
    return {"input": inp, "output": out}


def _make_pair(rng, kind, directions=None, forbidden=()):
    teaching = kind in ("mirror", "triple")
    for attempt in range(24):
        specs = _specs(rng, kind, directions, forbidden)
        background = _colors(rng, specs, teaching)
        n = len(specs)
        low = 18 if n == 2 else 22 if n == 3 else 25
        if attempt < 3:
            shape = tuple(map(int, rng.integers(low, 31, size=2)))
        else:
            shape = (30, 30)
        if not _place(rng, specs, shape, teaching):
            continue
        if kind == "triple":
            distances = [sum((a - b) ** 2 for a, b in zip(specs[0]["center"], target["center"]))
                         for target in specs[1:3]]
            if distances[0] == distances[1]:
                continue
        pair = _render(shape, background, specs)
        try:
            analysis = _analyze(pair["input"])
            assert np.array_equal(_predict(pair["input"], analysis), pair["output"])
        except AssertionError:
            continue
        return pair
    raise RuntimeError("bounded portal placement exhausted")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(name + " is outside its legal range")
    return int(value)


def _fingerprint(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    dirs = list(map(int, rng.permutation(4)))
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    witnesses = {slots[0]: ("mirror", dirs[:2]), slots[1]: ("triple", dirs[2:])}
    puzzle = {"train": [], "test": []}
    seen, train_badges = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            kind, directions = witnesses.get(index, ("random", None)) if split == "train" else ("two_pairs", None)
            for _ in range(12):
                pair = _make_pair(rng, kind, directions, train_badges if split == "test" else ())
                fingerprint = _fingerprint(pair["input"])
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_badges.update(_analyze(pair["input"])["groups"])
                    break
            else:
                raise RuntimeError("distinct-input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair, analysis=None):
    inp, out = (np.asarray(pair[k]) for k in ("input", "output"))
    analysis = _analyze(inp) if analysis is None else analysis
    groups = analysis["groups"]
    arrivals = list(_arrivals(analysis))
    taught, rotations, size_changes, lengths = set(), set(), set(), []
    for source, target, direction, _ in arrivals:
        rotations.add((target["pose"] - source["pose"]) % 4)
        size_changes.add((source["radius"], target["radius"]))
        paths = _paths(target["center"], target["radius"], direction, inp.shape)
        lengths.extend(map(len, paths))
        dr, dc = DIRECTIONS[direction]
        probes = [(r + dr * step, c + dc * step)
                  for r, c in _cap(target["center"], target["radius"], direction)[1:-1]
                  for step in (1, 2)]
        if min(map(len, paths)) >= 2 and all(_inside(p, inp.shape) and inp[p] == out[p] == analysis["base"][p] for p in probes):
            taught.add(direction)
    singletons = [members[0] for members in groups.values() if len(members) == 1]
    mirror_negative = any(_canonical(np.fliplr(o["badge"])) != o["key"] and
                          len(groups.get(_canonical(np.fliplr(o["badge"])), [])) > 1 for o in singletons)
    ordinary = any(_canonical(np.fliplr(o["badge"])) not in groups or
                   _canonical(np.fliplr(o["badge"])) == o["key"] for o in singletons)
    different_colors = any(a["badge_color"] != b["badge_color"] for a, b, _, _ in arrivals)
    payload_binding = any(len(set(o["caps"].values())) > 1 and len(groups[o["key"]]) > 1
                          for o in analysis["objects"])
    unrelated_reuse = any(a["badge_color"] == b["badge_color"] and a["key"] != b["key"]
                          for a in analysis["objects"] for b in analysis["objects"])
    return dict(count=len(analysis["objects"]), sizes=tuple(sorted(map(len, groups.values()))),
                keys=set(groups), background=analysis["background"],
                shape=inp.shape, radii={o["radius"] for o in analysis["objects"]},
                rotations=rotations, size_changes=size_changes, lengths=lengths,
                taught=taught, mirror_negative=mirror_negative, ordinary=ordinary,
                different_colors=different_colors, unrelated_reuse=unrelated_reuse,
                payload_binding=payload_binding,
                cap_counts=[len(o["caps"]) for o in analysis["objects"] if o["caps"]],
                areas={int(o["badge"].sum()) for o in analysis["objects"]},
                changed=int(np.count_nonzero(inp != out)))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_grid(pair[k]) for k in ("input", "output"))
            assert inp.shape == out.shape
            fingerprint = _fingerprint(inp)
            assert fingerprint not in seen
            seen.add(fingerprint)
            analysis = _analyze(inp)
            assert 2 <= len(analysis["objects"]) <= 4
            assert np.array_equal(_predict(inp, analysis), out), "incorrect cap transfer or endpoint rays"
            assert np.array_equal(_closed_form(inp, analysis), out)
            foreground = inp != analysis["base"]
            assert np.array_equal(inp[foreground], out[foreground]), "original objects must be preserved"
            assert not np.array_equal(inp, out)
            features[split].append(_features(pair, analysis))
    train = features["train"]
    assert set.union(*(f["taught"] for f in train)) == set(range(4)), "four directions need endpoint/interior witnesses"
    assert any(f["mirror_negative"] for f in train), "reflection shortcut not contradicted"
    assert any(f["ordinary"] for f in train), "ordinary singleton not shown"
    assert any(3 in f["sizes"] for f in train), "all receivers of a triplet must be taught"
    assert any(f["rotations"] & {1, 3} for f in train)
    assert any(f["different_colors"] and f["unrelated_reuse"] for f in train)
    assert any(f["payload_binding"] for f in train), "per-cap color binding not demonstrated"
    assert any(a != b for f in train for a, b in f["size_changes"]), "local housing size adaptation not shown"
    trained_keys = set.union(*(f["keys"] for f in train))
    for f in features["test"]:
        assert f["sizes"] == (2, 2), "test recombines two simultaneously active classes"
        assert not f["keys"] & trained_keys, "test badges must have novel shapes"


def self_test():
    """Bounded streaming 42-seed tests; no files or execution traces are saved."""
    summaries = []
    hashes, structures, shapes, backgrounds, keys = set(), set(), set(), set(), set()
    counts, cap_counts, radii, rotations, size_changes, areas = set(), set(), set(), set(), set(), set()
    max_ray = 0
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        if first is None:
            first = puzzle
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                feature = _features(pair)
                digest.update(bytes(grid.shape) + grid.tobytes() + pair["output"].tobytes())
                shapes.add(grid.shape)
                backgrounds.add(feature["background"])
                keys.update(feature["keys"])
                counts.add(feature["count"])
                cap_counts.update(feature["cap_counts"])
                radii.update(feature["radii"])
                rotations.update(feature["rotations"])
                size_changes.update(feature["size_changes"])
                areas.update(feature["areas"])
                max_ray = max(max_ray, max(feature["lengths"], default=0))
                roles = {}
                structures.add((grid.shape, bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)))
                summaries.append(feature["changed"])
        hashes.add(digest.hexdigest())
    assert len(hashes) == 42 and len(structures) >= 180
    assert len(shapes) >= 20 and len(backgrounds) >= 25 and len(keys) >= 25
    assert counts == {2, 3, 4} and radii == {3, 4}
    assert rotations == set(range(4)) and cap_counts == {1, 2, 3, 4}
    assert size_changes == {(3, 3), (3, 4), (4, 3), (4, 4)}
    assert areas == {4, 5, 6, 7} and max_ray >= 15
    repeat = generate_puzzle(HOST_SEEDS[0])
    assert all(_fingerprint(p[r]) == _fingerprint(q[r]) for s in first
               for p, q in zip(first[s], repeat[s]) for r in ("input", "output"))
    # All legal count pairs, including two teaching examples and four tests.
    legal_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
            legal_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{r: g.copy() for r, g in p.items()} for p in first[s]] for s in first}
    inp, out = (bad["test"][0][r] for r in ("input", "output"))
    cell = tuple(np.argwhere(inp != out)[0])
    out[cell] = inp[cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a missing arrival cell")
    return dict(seeds=42, pairs=len(summaries), legal_count_calls=legal_calls,
                distinct_puzzles=len(hashes), structures=len(structures), shapes=len(shapes),
                backgrounds=len(backgrounds), badge_classes=len(keys),
                object_counts=sorted(counts), source_cap_counts=sorted(cap_counts),
                radii=sorted(radii), relative_rotations=sorted(rotations),
                radius_transfers=sorted(size_changes), badge_areas=sorted(areas),
                max_endpoint_length=max_ray, changed_cells_range=[min(summaries), max(summaries)])
