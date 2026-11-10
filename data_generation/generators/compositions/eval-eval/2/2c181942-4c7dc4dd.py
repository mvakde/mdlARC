"""Reference-calibrated comb-key assembly.

A fitted port demonstrates a constant total tooth depth. Loose keys are
selected by BOTH local label color and complementary tooth profile, rotated
without reflection, and inserted into the remaining locks. No parent code,
files, sampled parameters, or demonstrations are consulted by solve().

Generation and self_test do not write files. All retries and memory are bounded.
"""
from __future__ import annotations

import hashlib
from itertools import product

import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_WIDTHS = (4, 6, 8)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8, copy=False).tobytes()


def _port_cells(hub, side, depth, lane):
    """Top uses left-to-right lanes; other ports are its clockwise rotations."""
    row, col, width = hub
    r, c = -1 - np.asarray(depth), 1 + np.asarray(lane)
    for _ in range(side):
        r, c = c, width + 1 - r
    r, c = np.broadcast_arrays(r, c)
    return np.column_stack((r.ravel() + row, c.ravel() + col)).astype(int)


def _bar_cells(hub, side):
    return _port_cells(hub, side, -1, np.arange(hub[2]))


def _paint(grid, cells, color):
    grid[cells[:, 0], cells[:, 1]] = color


def _inside(cells, shape):
    return bool(np.all((cells >= 0) & (cells < np.array(shape))))


def _plate(grid, hub, stone, labels):
    r, c, m = hub
    grid[r:r + m + 2, c:c + m + 2] = stone
    for side, label in enumerate(labels):
        _paint(grid, _bar_cells(hub, side), label)


def _find_plate(grid, background):
    found = []
    height, width = grid.shape
    for m in _WIDTHS:
        n = m + 2
        if min(height, width) < n:
            continue
        tl = grid[:height - n + 1, :width - n + 1]
        tr = grid[:height - n + 1, n - 1:]
        bl = grid[n - 1:, :width - n + 1]
        br = grid[n - 1:, n - 1:]
        for r, c in np.argwhere((tl == tr) & (tl == bl) & (tl == br) & (tl != background)):
            r, c = int(r), int(c)
            block = grid[r:r + n, c:c + n]
            stone = int(block[0, 0])
            bars = (block[0, 1:-1], block[1:-1, -1],
                    block[-1, 1:-1], block[1:-1, 0])
            labels = tuple(int(b[0]) for b in bars)
            if len(set(labels)) != 4 or stone in labels or background in labels:
                continue
            if not all(np.all(b == label) for b, label in zip(bars, labels)):
                continue
            if not np.all(block[1:-1, 1:-1] == stone):
                continue
            found.append(((r, c, m), stone, labels))
    _require(len(found) == 1, "expected one solid plate with four distinct colored bars")
    return found[0]


def _mask_from_profile(profile, capacity, handle):
    teeth = capacity - np.asarray(profile, dtype=int)
    _require(bool(np.all(teeth >= 1)), "nonpositive key tooth")
    m, depth = len(teeth), int(teeth.max())
    mask = np.zeros((handle + depth, m), dtype=bool)
    mask[:handle, m // 2 - 1:m // 2 + 1] = True
    mask[handle:] = np.arange(depth)[:, None] < teeth[None, :]
    return mask


def _key_options(mask):
    """Recognize the neck/spine/teeth geometrically, not by a mask catalog."""
    options = {}
    for turns in range(4):
        a = np.rot90(mask, turns).copy()
        h, m = a.shape
        if m not in _WIDTHS or h < 2:
            continue
        neck = np.zeros(m, dtype=bool)
        neck[m // 2 - 1:m // 2 + 1] = True
        handle = 0
        while handle < h and np.array_equal(a[handle], neck):
            handle += 1
        if not 1 <= handle <= 3 or handle == h or not np.all(a[handle]):
            continue
        body = a[handle:]
        teeth = body.sum(axis=0).astype(int)
        expected = np.arange(body.shape[0])[:, None] < teeth[None, :]
        if np.any(teeth == 0) or not np.array_equal(body, expected):
            continue
        signature = _key(a)
        if signature not in options:
            options[signature] = dict(mask=a, handle=handle, teeth=teeth, turns=[])
        options[signature]["turns"].append(turns)
    return list(options.values())


def _canonical_key(mask):
    options = _key_options(mask)
    _require(len(options) == 1, "missing or competing key-handle interpretations")
    return options[0]


def _insert_cells(hub, side, mask, handle, capacity):
    r, c = np.nonzero(mask)
    return _port_cells(hub, side, capacity + handle - 1 - r, c)


def _components(grid, background, protected):
    remaining = set(map(tuple, np.argwhere((grid != background) & ~protected)))
    pieces = []
    while remaining:
        start = min(remaining)
        color = int(grid[start])
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                point = (r + dr, c + dc)
                if point in remaining and int(grid[point]) == color:
                    remaining.remove(point)
                    stack.append(point)
        coords = np.array(cells, dtype=int)
        low, high = coords.min(axis=0), coords.max(axis=0)
        mask = np.zeros(tuple(high - low + 1), dtype=bool)
        mask[coords[:, 0] - low[0], coords[:, 1] - low[1]] = True
        pieces.append(dict(color=color, cells=coords, **_canonical_key(mask)))
    return pieces


def _inspect(input_grid):
    grid = _grid(input_grid)
    background = int(np.bincount(grid.ravel(), minlength=10).argmax())
    _require(np.count_nonzero(grid == background) > grid.size // 2,
             "background must occupy more than half of the input")
    hub, stone, labels = _find_plate(grid, background)
    m = hub[2]
    protected = np.zeros(grid.shape, dtype=bool)
    r, c, _ = hub
    protected[r:r + m + 2, c:c + m + 2] = True
    base = np.full_like(grid, background)
    _plate(base, hub, stone, labels)
    locks, completed = {}, []

    def at(side, d, j):
        cell = _port_cells(hub, side, d, j)[0]
        return int(grid[tuple(cell)]) if _inside(cell[None, :], grid.shape) else background

    for side in range(4):
        profile, key_runs, lock_cells, fitted_cells = [], [], [], []
        for j in range(m):
            length = 0
            while length < 30 and at(side, length, j) == stone:
                lock_cells.append(tuple(_port_cells(hub, side, length, j)[0]))
                length += 1
            profile.append(length)
            end = length
            while end < 30 and at(side, end, j) == labels[side]:
                fitted_cells.append(tuple(_port_cells(hub, side, end, j)[0]))
                end += 1
            key_runs.append(end - length)
        if not any(profile):
            _require(not any(key_runs), "inactive port cannot have a fitted key")
            continue
        _require(min(profile) >= 1 and len(set(profile)) > 1, "invalid lock profile")
        locks[side] = np.array(profile, dtype=int)
        cells = np.array(lock_cells, dtype=int)
        _paint(base, cells, stone)
        protected[cells[:, 0], cells[:, 1]] = True
        if any(key_runs):
            _require(min(key_runs) >= 1, "partly fitted reference key")
            ends = np.array(profile) + np.array(key_runs)
            outer = [j for j in range(m) if j not in (m // 2 - 1, m // 2)]
            capacity = int(ends[outer[0]])
            handle = int(ends[m // 2] - capacity)
            _require(3 <= capacity <= 27 and 1 <= handle <= 3, "invalid reference depth")
            expected_ends = np.full(m, capacity)
            expected_ends[m // 2 - 1:m // 2 + 1] += handle
            _require(np.array_equal(ends, expected_ends), "reference must fill a rectangle plus handle")
            mask = _mask_from_profile(profile, capacity, handle)
            _canonical_key(mask)
            cells = np.array(fitted_cells, dtype=int)
            _paint(base, cells, labels[side])
            protected[cells[:, 0], cells[:, 1]] = True
            completed.append(dict(side=side, capacity=capacity, handle=handle, cells=cells))
    _require(len(completed) == 1 and 2 <= len(locks) <= 4,
             "one completed port and at least one unfinished lock are required")
    reference = completed[0]
    capacity = reference["capacity"]
    pieces = _components(grid, background, protected)
    _require(bool(pieces), "missing loose keys")
    # Every loose object has a full eight-neighbor gap from the fixed assembly
    # and every other loose object, even when its label color is reused.
    owners = np.full(grid.shape, -1, dtype=int)
    owners[protected] = -2
    for index, p in enumerate(pieces):
        _require(p["color"] not in (background, stone), "invalid loose-key color")
        for rr, cc in p["cells"]:
            window = owners[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2]
            _require(not np.any(window != -1), "loose keys touch another object")
        owners[p["cells"][:, 0], p["cells"][:, 1]] = index

    answer = base.copy()
    selected, targets = [], sorted(set(locks) - {reference["side"]})
    for side in targets:
        fits = [p for p in pieces if p["color"] == labels[side]
                and len(p["teeth"]) == m
                and np.all(locks[side] + p["teeth"] == capacity)]
        _require(len(fits) == 1, "each unfinished lock needs exactly one matching colored complement")
        p = fits[0]
        cells = _insert_cells(hub, side, p["mask"], p["handle"], capacity)
        _require(_inside(cells, grid.shape), "assembled key would cross the boundary")
        _require(bool(np.all(answer[cells[:, 0], cells[:, 1]] == background)), "key overlaps assembly")
        _paint(answer, cells, p["color"])
        selected.append(dict(side=side, piece=p, cells=cells))
    return answer, dict(background=background, stone=stone, hub=hub, labels=labels,
                        locks=locks, reference=reference, pieces=pieces, selected=selected,
                        targets=targets, base=base)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the entire assembled scene from the visible input alone."""
    return _inspect(input_grid)[0]


def _profile(rng, m, capacities):
    limit = min(capacities)
    for _ in range(80):
        profile = rng.integers(1, limit, size=m)
        if rng.integers(2):
            profile[0], profile[-1] = 1, int(rng.integers(2, limit))
        else:
            profile[-1], profile[0] = 1, int(rng.integers(2, limit))
        if all(len(_key_options(_mask_from_profile(profile, cap, handle))) == 1
               for cap, handle in product(capacities, (1, 2))):
            return profile
    # This asymmetric comb has one geometrically recognizable handle for all
    # sampled capacities (4..7). It is not consulted by input-only solving.
    return np.array([1] + [2] * (m - 2) + [3], dtype=int)


def _new_template(rng, capacities, m, target_count, twins=False):
    palette = list(map(int, rng.permutation(10)))
    background, stone, labels = palette[0], palette[1], tuple(palette[2:6])
    sides = list(map(int, rng.permutation(4)))
    reference_side, targets = sides[0], sides[1:1 + target_count]
    profiles = {s: _profile(rng, m, capacities) for s in (reference_side, *targets)}
    if twins:
        profiles[targets[1]] = profiles[targets[0]].copy()
        if np.array_equal(profiles[reference_side], profiles[targets[0]]):
            profiles[reference_side] = profiles[targets[0]][::-1].copy()
    pool = []

    def add(side, cap, color=None, forced=None, forced_handle=None):
        handle = int(rng.integers(1, 3)) if forced_handle is None else forced_handle
        mask = _mask_from_profile(profiles[side], cap, handle)
        turn = int(rng.integers(4)) if forced is None else int(forced)
        pool.append(dict(color=labels[side] if color is None else color,
                         mask=mask, handle=handle, turn=turn, side=side))

    if twins:
        for index, side in enumerate(targets):
            for cap_index, cap in enumerate(capacities):
                add(side, cap, forced=(-side - index - 1) % 4,
                    forced_handle=1 + (index + cap_index) % 2)
        # This is an exact silhouette duplicate of a legitimate key, so a
        # geometric fit alone cannot supply the correct visible color.
        original = pool[0]
        pool.append({**original, "color": palette[6], "turn": int(rng.integers(4))})
    else:
        actual, alternative = capacities
        for side in targets:
            add(side, actual)
        add(targets[0], alternative)
        max_pool = 4 if m == 6 else 6
        if len(pool) < max_pool and rng.random() < .75:
            original = pool[int(rng.integers(target_count))]
            pool.append({**original, "color": palette[6], "turn": int(rng.integers(4))})
        if len(pool) < max_pool and target_count > 1 and rng.random() < .45:
            add(targets[-1], alternative)
    return dict(background=background, stone=stone, labels=labels, m=m,
                profiles=profiles, reference_side=reference_side, targets=targets,
                ref_handle=int(rng.integers(1, 3)), pool=pool,
                capacities=tuple(map(int, capacities)))


def _reserve(template, shape, hub):
    reserve = np.zeros(shape, dtype=bool)
    r, c, m = hub
    reserve[r:r + m + 2, c:c + m + 2] = True
    extent = max(template["capacities"]) + 2
    d, j = np.indices((extent, m))
    for side in template["profiles"]:
        cells = _port_cells(hub, side, d, j)
        _require(_inside(cells, shape), "invalid packing envelope")
        reserve[cells[:, 0], cells[:, 1]] = True
    blocked = reserve.copy()
    for r, c in np.argwhere(reserve):
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return blocked


def _pack(template, rng):
    """Try bounded random packing, then a guaranteed separated corner layout."""
    m = template["m"]
    extent, n = max(template["capacities"]) + 2, m + 2
    for attempt in range(28):
        h, w = (map(int, rng.integers(24 if m == 4 else 26, 31, size=2))
                if attempt < 12 else (30, 30))
        low, high = [0, 0], [h - n, w - n]
        for side in template["profiles"]:
            if side == 0:
                low[0] = extent
            elif side == 1:
                high[1] = w - n - extent
            elif side == 2:
                high[0] = h - n - extent
            else:
                low[1] = extent
        if any(a > b for a, b in zip(low, high)):
            continue
        pos = [int(rng.integers(a, b + 1)) for a, b in zip(low, high)]
        if rng.random() < .3:
            axis = int(rng.integers(2))
            pos[axis] = int(rng.choice((low[axis], high[axis])))
        hub = (*pos, m)
        blocked = _reserve(template, (h, w), hub)
        grid = np.full((h, w), template["background"], dtype=np.int8)
        order = list(map(int, rng.permutation(len(template["pool"]))))
        order.sort(key=lambda k: -template["pool"][k]["mask"].size)
        success = True
        for index in order:
            p = template["pool"][index]
            a = np.rot90(p["mask"], p["turn"])
            ph, pw = a.shape
            rr, cc = np.nonzero(a)
            ys, xs = np.indices((h - ph + 1, w - pw + 1))
            rows, cols = ys.ravel()[:, None] + rr, xs.ravel()[:, None] + cc
            choices = np.flatnonzero(~np.any(blocked[rows, cols], axis=1))
            if not len(choices):
                success = False
                break
            k = int(rng.choice(choices))
            cells = np.column_stack((rows[k], cols[k]))
            _paint(grid, cells, p["color"])
            for y, x in cells:
                blocked[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
        if success:
            return grid, hub, False
    return _corner_pack(template)


def _corner_pack(template):
    """Guaranteed fallback; each non-four-lane key gets an 8x8 corner."""
    m = template["m"]
    n = m + 2
    hub = ((30 - n) // 2, (30 - n) // 2, m)
    grid = np.full((30, 30), template["background"], dtype=np.int8)
    if m == 4:
        slots = ((0, 0), (0, 5), (0, 21), (0, 26),
                 (22, 0), (22, 5), (22, 21), (22, 26))
    else:
        slots = ((0, 0), (0, 22), (22, 0), (22, 22))
    _require(len(template["pool"]) <= len(slots), "too many keys for fallback slots")
    for p, (y, x) in zip(template["pool"], slots):
        turn = (0 if p["side"] in (1, 2, 3) else 2) if m == 4 else p["turn"]
        a = np.rot90(p["mask"], turn)
        rr, cc = np.nonzero(a)
        grid[y + rr, x + cc] = p["color"]
    return grid, hub, True


def _render(template, loose, hub, capacity):
    """Construct labels independently from the pixel-reading solving path."""
    inp = loose.copy()
    out = np.full_like(loose, template["background"])
    for a in (inp, out):
        _plate(a, hub, template["stone"], template["labels"])
        for side, profile in template["profiles"].items():
            for j, length in enumerate(profile):
                cells = _port_cells(hub, side, np.arange(length), j)
                _paint(a, cells, template["stone"])
    side = template["reference_side"]
    handle = template["ref_handle"]
    mask = _mask_from_profile(template["profiles"][side], capacity, handle)
    cells = _insert_cells(hub, side, mask, handle, capacity)
    for a in (inp, out):
        _paint(a, cells, template["labels"][side])
    for side in template["targets"]:
        desired = capacity - template["profiles"][side]
        candidates = [p for p in template["pool"] if p["color"] == template["labels"][side]
                      and np.array_equal(_canonical_key(p["mask"])["teeth"], desired)]
        _require(len(candidates) == 1, "construction lost a unique key")
        p = candidates[0]
        cells = _insert_cells(hub, side, p["mask"], p["handle"], capacity)
        _paint(out, cells, p["color"])
    return {"input": inp, "output": out}


def _counts(a, b):
    for x, low, high in ((a, 2, 10), (b, 1, 4)):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
            raise ValueError("example counts must be integers")
        if not low <= x <= high:
            raise ValueError("example count outside supported range")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_caps = tuple(map(int, rng.choice((4, 5, 6, 7), 2, replace=False)))
    test_caps = tuple(c for c in (4, 5, 6, 7) if c not in train_caps)
    # The first two demonstrations are matched counterfactuals: same unfinished
    # locks, labels and loose pool, but a different already-fitted reference.
    template = _new_template(rng, train_caps, 4, 2, twins=True)
    loose, hub, _ = _pack(template, rng)
    train = [_render(template, loose, hub, cap) for cap in train_caps]
    seen = {_key(p["input"]) for p in train}
    puzzle = {"train": train, "test": []}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for _ in range(count):
            for attempt in range(12):
                cap = int(rng.choice(train_caps if split == "train" else test_caps))
                alt = int(rng.choice([x for x in (4, 5, 6, 7) if x != cap]))
                m = int(rng.choice((4, 6)))
                target_count = int(rng.integers(1, 4))
                template = _new_template(rng, (cap, alt), m, target_count)
                loose, hub, _ = _pack(template, rng)
                pair = _render(template, loose, hub, cap)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    answer, a = _inspect(pair["input"])
    assert np.array_equal(answer, pair["output"]), "incorrect interlocking assembly"
    cap, m = a["reference"]["capacity"], a["hub"][2]
    wrong_color_fit = False
    alternative_depths = set()
    for side in a["targets"]:
        for p in a["pieces"]:
            if len(p["teeth"]) != m:
                continue
            summed = a["locks"][side] + p["teeth"]
            if len(set(map(int, summed))) == 1:
                if p["color"] == a["labels"][side] and int(summed[0]) != cap:
                    alternative_depths.add(int(summed[0]))
                if p["color"] not in a["labels"] and int(summed[0]) == cap:
                    wrong_color_fit = True
    return dict(capacity=cap, lanes=m, targets=len(a["targets"]),
                reference_side=a["reference"]["side"], background=a["background"],
                shape=tuple(np.asarray(pair["input"]).shape),
                handles=tuple(p["handle"] for p in a["pieces"]),
                profiles=tuple(tuple(map(int, p)) for p in a["locks"].values()),
                key_masks=tuple(_key(p["mask"]) for p in a["pieces"]),
                rotations=tuple((t - s["side"]) % 4 for s in a["selected"]
                                for t in s["piece"]["turns"]),
                wrong_color_fit=wrong_color_fit, alternative_depths=tuple(sorted(alternative_depths)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            features[split].append(_features({"input": inp, "output": out}))
    train = features["train"]
    train_caps = {f["capacity"] for f in train}
    assert len(train_caps) == 2, "training must teach two reference depths"
    assert not train_caps & {f["capacity"] for f in features["test"]}, "missing held-out depth shift"
    assert any(f["wrong_color_fit"] for f in train), "missing color-selection witness"
    assert any(f["alternative_depths"] for f in train), "reference could be bypassed"
    assert any(any(r != 0 for r in f["rotations"]) for f in train), "missing rotation witness"
    # The twin demonstrations differ only in their fitted reference key. Their
    # query locks and loose candidates are identical; selected target keys differ.
    a, b = (_inspect(puzzle["train"][i]["input"])[1] for i in (0, 1))
    assert a["hub"] == b["hub"] and a["labels"] == b["labels"]
    assert len(a["targets"]) == 2
    x, y = a["targets"]
    assert np.array_equal(a["locks"][x], a["locks"][y])
    assert not np.array_equal(a["locks"][x], a["locks"][a["reference"]["side"]])
    assert {s["piece"]["handle"] for s in a["selected"]} == {1, 2}
    assert {s["piece"]["handle"] for s in b["selected"]} == {1, 2}
    assert any(p["color"] not in a["labels"]
               and any(_key(p["mask"]) == _key(s["piece"]["mask"]) for s in a["selected"])
               for p in a["pieces"]), "missing exact-shape wrong-color witness"
    stripped = []
    for pair, scene in zip(puzzle["train"][:2], (a, b)):
        g = _grid(pair["input"])
        _paint(g, scene["reference"]["cells"], scene["background"])
        stripped.append(g)
    assert np.array_equal(*stripped), "matched reference witnesses were changed"
    assert a["reference"]["capacity"] != b["reference"]["capacity"]
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])


def _puzzle_signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _extension_self_test():
    checks = []
    for m, cap, handle in ((4, 3, 3), (8, 8, 3), (6, 9, 1), (4, 23, 1)):
        profile = np.array([1] + [2] * (m - 2) + [min(3, cap - 1)])
        mask = _mask_from_profile(profile, cap, handle)
        template = dict(background=0, stone=8, labels=(1, 2, 3, 4), m=m,
                        profiles={0: profile[::-1].copy(), 1: profile},
                        reference_side=0, targets=[1], ref_handle=handle,
                        pool=[dict(mask=mask, handle=handle, color=2, turn=0, side=1)])
        hub = (24, 0, m) if cap == 23 else (12, 8, m)
        loose = np.zeros((30, 30), dtype=np.int8)
        y, x = (0, 12) if cap == 23 else (0, 22 if m < 8 else 21)
        rr, cc = np.nonzero(mask)
        loose[y + rr, x + cc] = 2
        pair = _render(template, loose, hub, cap)
        assert np.array_equal(solve(pair["input"]), pair["output"])
        checks.append(dict(lanes=m, capacity=cap, handle=handle))
    tiny = np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
    assert len(_key_options(tiny)) == 1
    symmetric = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)
    assert len(_canonical_key(symmetric)["turns"]) == 2
    ambiguous = np.array([[0, 1, 1, 0], [1, 1, 1, 1],
                          [0, 0, 1, 1], [0, 0, 1, 0]], dtype=bool)
    try:
        _canonical_key(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("competing handle interpretations accepted")
    # Exercise the corner fallback directly, including quarter-turned six-lane
    # pieces whose width can be eight, and the crowded matched-twin pool.
    for m, count, twin in ((4, 2, True), (6, 3, False)):
        template = _new_template(np.random.default_rng(818 + m), (7, 6), m, count, twins=twin)
        if m == 6:
            for p in template["pool"]:
                teeth = _canonical_key(p["mask"])["teeth"]
                p["mask"] = _mask_from_profile(8 - teeth, 8, 2)
                p["handle"], p["turn"] = 2, 1
        loose, hub, used_fallback = _corner_pack(template)
        assert used_fallback
        for cap in ((7, 6) if twin else (7,)):
            pair = _render(template, loose, hub, cap)
            assert np.array_equal(solve(pair["input"]), pair["output"])
    return checks


def self_test():
    """42 seeds, streamed measurements, counterfactuals and negative checks.

    Streams the seed sweep; retains only a few boundary/counterfactual fixtures.
    It returns small aggregate measurements and never saves scenes or traces.
    """
    signatures, shapes, profiles, masks, capacities, lanes = set(), set(), set(), set(), set(), set()
    ref_sides, rotations, counts, handles, backgrounds = set(), set(), set(), set(), set()
    pair_count, color_witnesses, depth_witnesses = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                pair_count += 1
                shapes.add(f["shape"])
                profiles.update(f["profiles"])
                masks.update(f["key_masks"])
                capacities.add(f["capacity"])
                lanes.add(f["lanes"])
                ref_sides.add(f["reference_side"])
                rotations.update(f["rotations"])
                counts.add(f["targets"])
                handles.update(f["handles"])
                backgrounds.add(f["background"])
                color_witnesses += f["wrong_color_fit"]
                depth_witnesses += bool(f["alternative_depths"])
        if seed == 0:
            # A's color routing: two geometrically identical unfinished locks
            # exchange labels, with every loose key and reference unchanged.
            pair = puzzle["train"][0]
            _, scene = _inspect(pair["input"])
            x, y = scene["targets"]
            assert np.array_equal(scene["locks"][x], scene["locks"][y])
            swapped = pair["input"].copy()
            _paint(swapped, _bar_cells(scene["hub"], x), scene["labels"][y])
            _paint(swapped, _bar_cells(scene["hub"], y), scene["labels"][x])
            other = solve(swapped)
            for selected in scene["selected"]:
                s = selected["side"]
                other_side = y if s == x else x
                other_label = scene["labels"][other_side]
                moved = next(q["piece"] for q in scene["selected"] if q["side"] == other_side)
                cells = _insert_cells(scene["hub"], s, moved["mask"], moved["handle"],
                                      scene["reference"]["capacity"])
                assert np.all(other[cells[:, 0], cells[:, 1]] == other_label)
            # Keep all fixed geometry and the reference unchanged, and shorten
            # one selected loose handle. The fitted answer must shorten too.
            chosen = next(s for s in scene["selected"] if s["piece"]["handle"] == 2)
            p = chosen["piece"]
            canonical_tip = np.zeros_like(p["mask"])
            canonical_tip[0, scene["hub"][2] // 2 - 1:scene["hub"][2] // 2 + 1] = True
            raw_tip = np.rot90(canonical_tip, -p["turns"][0])
            low = p["cells"].min(axis=0)
            tip = np.argwhere(raw_tip) + low
            shortened = pair["input"].copy()
            _paint(shortened, tip, scene["background"])
            expected = pair["output"].copy()
            d = scene["reference"]["capacity"] + 1
            tip = _port_cells(scene["hub"], chosen["side"], d,
                              np.arange(scene["hub"][2] // 2 - 1, scene["hub"][2] // 2 + 1))
            _paint(expected, tip, scene["background"])
            assert np.array_equal(solve(shortened), expected)
            # B's calibration: erase the completed exemplar from both outputs;
            # the remaining assemblies still differ, despite identical queries.
            erased = []
            for twin in puzzle["train"][:2]:
                _, info = _inspect(twin["input"])
                g = twin["output"].copy()
                _paint(g, info["reference"]["cells"], info["background"])
                erased.append(g)
            assert not np.array_equal(*erased)
            # The rule is invariant to a global quarter-turn and color renaming.
            assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
            rename = np.array([9, 3, 5, 1, 6, 0, 8, 2, 4, 7], dtype=np.int8)
            assert np.array_equal(solve(rename[pair["input"]]), rename[pair["output"]])
            before = pair["input"].copy()
            solve(pair["input"])
            assert np.array_equal(before, pair["input"]), "solver mutated its input"
    assert len(signatures) == 42
    assert len(shapes) >= 15 and len(profiles) >= 50 and len(masks) >= 80
    assert capacities == {4, 5, 6, 7} and lanes == {4, 6}
    assert ref_sides == rotations == set(range(4))
    assert counts == {1, 2, 3} and handles == {1, 2} and len(backgrounds) >= 8
    assert color_witnesses >= 42 and depth_witnesses >= 42
    extension_checks = _extension_self_test()
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    first = generate_puzzle(991, 2, 1)
    assert _puzzle_signature(first) == _puzzle_signature(generate_puzzle(991, 2, 1))
    assert _puzzle_signature(first) != _puzzle_signature(generate_puzzle(992, 2, 1))
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(signatures),
                canvas_shapes=len(shapes), lock_profiles=len(profiles), key_shapes=len(masks),
                capacities=sorted(capacities), lane_counts=sorted(lanes),
                reference_sides=sorted(ref_sides), required_rotations=sorted(rotations),
                target_counts=sorted(counts), color_witnesses=color_witnesses,
                reference_depth_witnesses=depth_witnesses,
                extension_checks=extension_checks, corner_fallback_checks=3,
                independent_solver=False, blind_prediction=False)
