"""Contour broadcast scanners. All conventions are visible in each input.

No parent module, saved puzzle, or generation state is used by solve().
"""
from collections import deque
import json
from pathlib import Path

import numpy as np


DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 42407094, 517078733, 199749059, 751026520, 448988375,
              787278063, 376930149, 181357415, 252681578, 649284179,
              474525884, 221526078, 868229915, 774172855, 26925219,
              922116039, 780572759, 426011520, 782004145, 674526299,
              111336969, 613193231, 947465807, 125623134, 380864078,
              450359019, 325927250, 799479425, 763207349, 808905842,
              30705877, 83450702, 29024644, 546715868, 744931346,
              875243303, 292289583, 802719944, 515683906, 930144808, 225283786)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int16, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        root = min(todo)
        todo.remove(root)
        q, cells = [root], []
        for r, c in q:
            cells.append((r, c))
            for dr, dc in DIR4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    q.append(p)
        component = np.zeros(mask.shape, dtype=bool)
        rr, cc = np.array(cells).T
        component[rr, cc] = True
        result.append(component)
    return result


def _depth(mask, neighbors=DIR8):
    """Repeated erosion. Layer zero touches any outside neighbor."""
    current = mask.copy()
    d = np.full(mask.shape, -1, dtype=np.int16)
    h, w = mask.shape
    level = 0
    while current.any():
        pad = np.pad(current, 1)
        inside = current.copy()
        for dr, dc in neighbors:
            inside &= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        d[current & ~inside] = level
        current = inside
        level += 1
    return d


def _depth_bfs(mask):
    """Separate boundary-distance check, without iterative erosion."""
    h, w = mask.shape
    d = np.full(mask.shape, -1, dtype=np.int16)
    q = deque()
    for r, c in np.argwhere(mask):
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                        and mask[r + dr, c + dc]) for dr, dc in DIR8):
            d[r, c] = 0
            q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and d[rr, cc] < 0:
                d[rr, cc] = d[r, c] + 1
                q.append((rr, cc))
    return d


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _corners(mask, depth, length):
    paths = []
    for r, c in np.argwhere(mask & (depth == 0)):
        r, c = int(r), int(c)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (_inside(mask, r - dr, c) or _inside(mask, r, c - dc)
                        or not _inside(mask, r + dr, c)
                        or not _inside(mask, r, c + dc)):
                    continue
                points = tuple((r + k * dr, c + k * dc) for k in range(length))
                if all(_inside(mask, rr, cc) and depth[rr, cc] == k
                       for k, (rr, cc) in enumerate(points)):
                    paths.append(points)
    return paths


def _gate_distance(shape, gate):
    rr, cc = np.indices(shape)
    dist = np.full(shape, 100, dtype=np.int16)
    for r, c in gate:
        dist = np.minimum(dist, np.maximum(abs(rr - r), abs(cc - c)))
    return dist


def _normal(mask, gate):
    normals = []
    rr, cc = np.array(gate).T
    for dr, dc in DIR4:
        if dr and len(set(rr.tolist())) != 1:
            continue
        if dc and len(set(cc.tolist())) != 1:
            continue
        if all(not _inside(mask, r + dr, c + dc)
               and _inside(mask, r - dr, c - dc) for r, c in gate):
            normals.append((dr, dc))
    assert len(normals) == 1, "bar must have one outward normal"
    return normals[0]


def _parse(inp):
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    assert len(set(border.tolist())) == 1, "complete monochrome wall frame"
    wall = int(border[0])
    counts = np.bincount(inp.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert background != wall and counts[background] > inp.size / 2
    colors = [c for c in range(10) if counts[c] and c not in (background, wall)]
    assert len(colors) >= 3
    foreground = max(colors, key=lambda c: int(counts[c]))
    assert all(counts[foreground] > counts[c] for c in colors if c != foreground)
    codes = [c for c in colors if c != foreground]
    repeated = [c for c in codes if counts[c] > 1]
    assert len(repeated) == 1
    marker = repeated[0]
    singleton_colors = [c for c in codes if c != marker]
    assert all(counts[c] == 1 for c in singleton_colors)
    cues = set(map(tuple, np.argwhere(np.isin(inp, singleton_colors)).tolist()))
    regions = _components(inp != wall)
    active_options = [m for m in regions if all(m[r, c] for r, c in cues)]
    assert len(active_options) == 1
    active = active_options[0]
    depth = _depth(active)
    length = len(cues) + 1
    assert 2 <= length <= 4
    assert {int(depth[r, c]) for r, c in cues} == set(range(1, length))
    paths = [p for p in _corners(active, depth, length)
             if inp[p[0]] == marker and set(p[1:]) == cues]
    assert len(paths) == 1, "one uniquely decodable corner key"
    key = paths[0]
    period = tuple(int(inp[p]) for p in key)
    assert len(set(period)) == length
    gate = tuple(map(tuple, np.argwhere(inp == marker).tolist()))
    gate = tuple(p for p in gate if p != key[0])
    assert 2 <= len(gate) <= 9
    rr, cc = np.array(gate).T
    assert ((len(set(rr.tolist())) == 1 and np.array_equal(np.sort(cc), np.arange(cc.min(), cc.max() + 1)))
            or (len(set(cc.tolist())) == 1 and np.array_equal(np.sort(rr), np.arange(rr.min(), rr.max() + 1))))
    assert all(active[p] and depth[p] == 0 for p in gate)
    assert min(max(abs(r - rr), abs(c - cc)) for r, c in gate for rr, cc in key) >= 3
    normal = _normal(active, gate)
    lane = active & (_gate_distance(inp.shape, gate) == depth)
    return dict(background=background, wall=wall, foreground=foreground,
                marker=marker, period=period, key=key, gate=gate, normal=normal,
                active=active, depth=depth, lane=lane, regions=regions)


def _paint(inp, info, depth=None, lane=None, broadcast=True, full_bands=False):
    d = info["depth"] if depth is None else depth
    fan = info["lane"] if lane is None else lane
    active, period = info["active"], np.array(info["period"], dtype=np.int16)
    out = inp.copy()
    for p in info["key"]:
        out[p] = info["background"]
    even = active & (d % 2 == 0)
    flash = even & ((d // 2) % len(period) == len(period) - 1)
    if full_bands:
        out[active] = info["background"]
        out[even] = period[(d[even] // 2) % len(period)]
    else:
        out[fan] = info["background"]
        ticks = fan & even
        out[ticks] = period[(d[ticks] // 2) % len(period)]
    if broadcast:
        out[flash & (inp == info["foreground"])] = period[-1]
    return out.astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode geometry, key, bar and objects solely from this input."""
    inp = _grid(input_grid)
    try:
        info = _parse(inp)
    except AssertionError as exc:
        raise ValueError("input is outside the contour-scanner family: " + str(exc)) from exc
    return _paint(inp, info)


def _straight_fields(info):
    """A parent-B-only alternative: parallel Cartesian strip and inward distance."""
    rr, cc = np.indices(info["active"].shape)
    gr, gc = np.array(info["gate"]).T
    dr, dc = info["normal"]
    if dr:
        d = (rr - gr[0]) * -dr
        lane = (cc >= gc.min()) & (cc <= gc.max()) & (d >= 0)
    else:
        d = (cc - gc[0]) * -dc
        lane = (rr >= gr.min()) & (rr <= gr.max()) & (d >= 0)
    return d, lane & info["active"]


def _evidence(inp, info):
    m, d, lane = info["active"], info["depth"], info["lane"]
    fg = inp == info["foreground"]
    L = len(info["period"])
    flash = m & (d % 2 == 0) & ((d // 2) % L == L - 1)
    ordinary = m & (d > 0) & (d % 2 == 0) & ~flash
    bg = inp == info["background"]
    actual = _paint(inp, info)
    rr, cc = np.nonzero(m)
    box = np.full(m.shape, -1, dtype=np.int16)
    box[m] = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
    manhattan = _depth(m, DIR4)
    straight_d, straight_lane = _straight_fields(info)
    def differs(depth):
        fan = m & (_gate_distance(inp.shape, info["gate"]) == depth)
        return bool(np.any(actual != _paint(inp, info, depth=depth, lane=fan)))
    # A tie is a fan point also nearest to a boundary cell outside the gate.
    other_boundary = m & (d == 0)
    for p in info["gate"]:
        other_boundary[p] = False
    other_dist = _gate_distance(inp.shape, np.argwhere(other_boundary))
    ties = lane & (d > 0) & (other_dist == d)
    remote_component = mixed_object = False
    for obj in _components(fg & m):
        remote_component |= bool(np.any(obj & flash) and not np.any(obj & lane))
        mixed_object |= bool(np.any(obj & flash & ~lane) and np.any(obj & ~flash & ~lane))
    inactive_flash = False
    for region in info["regions"]:
        if np.array_equal(region, m):
            continue
        rd = _depth(region)
        inactive_flash |= bool(np.any(region & fg & (rd % 2 == 0) & ((rd // 2) % L == L - 1)))
    return dict(clear=bool(np.any(fg & lane & (d % 2 == 1))),
                overwrite=bool(np.any(fg & lane & ordinary)),
                flash_overwrite=bool(np.any(fg & lane & flash)),
                remote_flash=bool(np.any(fg & ~lane & flash)),
                preserved=bool(np.any(fg & m & ~lane & ~flash)),
                empty_flash=bool(np.any(bg & m & ~lane & flash)),
                repeated=bool(np.any(lane & (d >= 2 * L) & (d % (2 * L) == 0))),
                fanning=bool(np.any(lane & ~straight_lane & (d > 0) & (d % 2 == 0))),
                ties=bool(np.any(ties & (d % 2 == 0))),
                concave=bool(m.sum() < (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)),
                box_diff=differs(box), manhattan_diff=differs(manhattan),
                no_A_diff=bool(np.any(actual != _paint(inp, info, straight_d, straight_lane))),
                no_B_diff=bool(np.any(actual != _paint(inp, info, full_bands=True))),
                no_broadcast_diff=bool(np.any(actual != _paint(inp, info, broadcast=False))),
                remote_component=remote_component, mixed_object=mixed_object,
                inactive_flash=inactive_flash)


def _layout(rng, shape, reserve_side, rich):
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    rooms = [(1, h - 2, 1, w - 2)]
    target = int(rng.integers(1, 7))
    for _ in range(target - 1):
        choices = []
        for i, (a, b, c, e) in enumerate(rooms):
            for axis, low, high in ((0, a, b), (1, c, e)):
                for q in range(low + 3, high - 2):
                    children = ([(a, q - 1, c, e), (q + 1, b, c, e)] if axis == 0 else
                                [(a, b, c, q - 1), (a, b, q + 1, e)])
                    proposal = rooms[:i] + rooms[i + 1:] + children
                    if any(min(bb - aa + 1, ee - cc + 1) >= reserve_side
                           for aa, bb, cc, ee in proposal):
                        choices.append((i, axis, q, children))
        if not choices:
            break
        i, axis, q, children = choices[int(rng.integers(len(choices)))]
        a, b, c, e = rooms.pop(i)
        if axis == 0:
            wall[q, c - 1:e + 2] = True
        else:
            wall[a - 1:b + 2, q] = True
        rooms.extend(children)
    # U-shaped bays bend a region's contours and create small neighboring rooms.
    big = max(rooms, key=lambda r: min(r[1] - r[0], r[3] - r[2]))
    budgets = int(rng.integers(1 if rich else 0, 4))
    for _ in range(budgets):
        a, b, c, e = big
        transpose = bool(rng.integers(2))
        work = wall.T if transpose else wall
        aa, bb, cc, ee = (c, e, a, b) if transpose else big
        if bb - aa < 8 or ee - cc < 8:
            continue
        near_left = bool(rng.integers(2))
        base = cc - 1 if near_left else ee + 1
        sign = 1 if near_left else -1
        top = int(rng.integers(aa + 1, bb - 5))
        bottom = min(bb - 1, top + int(rng.integers(4, 9)))
        far = base + sign * int(rng.integers(3, min(6, ee - cc - 3)))
        lo, hi = sorted((base, far))
        if work[top - 1:bottom + 2, lo + 1:hi].any():
            continue
        work[top, lo:hi + 1] = True
        work[bottom, lo:hi + 1] = True
        work[top:bottom + 1, far] = True
    # Optional closed internal wall islands add genuine holes and enclosed rooms.
    for _ in range(int(rng.integers(0, 3))):
        for _attempt in range(12):
            bh, bw = int(rng.integers(4, 8)), int(rng.integers(4, 8))
            if bh + 4 >= h or bw + 4 >= w:
                continue
            r, c = int(rng.integers(2, h - bh - 1)), int(rng.integers(2, w - bw - 1))
            if wall[r - 1:r + bh + 1, c - 1:c + bw + 1].any():
                continue
            wall[r, c:c + bw] = wall[r + bh - 1, c:c + bw] = True
            wall[r:r + bh, c] = wall[r:r + bh, c + bw - 1] = True
            break
    return wall


def _gate_options(mask, depth, key, normal):
    dr, dc = normal
    boundary = mask & (depth == 0)
    eligible = np.zeros(mask.shape, dtype=bool)
    for r, c in np.argwhere(boundary):
        if (not _inside(mask, r + dr, c + dc) and _inside(mask, r - dr, c - dc)
                and min(max(abs(r - kr), abs(c - kc)) for kr, kc in key) >= 3):
            eligible[r, c] = True
    options = []
    # Components here are straight runs because the fixed outward normal fixes axis.
    work = eligible if dr else eligible.T
    for r in range(work.shape[0]):
        positions = np.flatnonzero(work[r])
        start = 0
        for end in range(1, len(positions) + 1):
            if end < len(positions) and positions[end] == positions[end - 1] + 1:
                continue
            run = positions[start:end]
            if len(run) >= 2:
                options.append(tuple((r, int(c)) if dr else (int(c), r) for c in run))
            start = end
    return options


def _objects(rng, allowed):
    """Sample varied natural rectangles, walks and grown masses, then test evidence."""
    h, w = allowed.shape
    target = int(allowed.sum() * rng.uniform(.09, .30))
    mask = np.zeros_like(allowed)
    points = np.argwhere(allowed)
    if not len(points):
        return mask
    count = int(rng.integers(2, 13))
    sizes = np.maximum(3, (rng.dirichlet(np.ones(count) * 1.5) * target).astype(int))
    for size in sizes:
        kind = int(rng.integers(4))
        if kind == 0:
            bh = min(h - 2, max(2, int(np.sqrt(size) * rng.uniform(.5, 1.8))))
            bw = min(w - 2, max(2, int(size) // bh))
            r, c = int(rng.integers(1, h - bh)), int(rng.integers(1, w - bw))
            patch = np.ones((bh, bw), dtype=bool)
            if bh >= 5 and bw >= 5 and rng.random() < .5:
                patch[2:-1, 2:-1] = False
            mask[r:r + bh, c:c + bw] |= patch & allowed[r:r + bh, c:c + bw]
        else:
            root = tuple(points[int(rng.integers(len(points)))])
            cells, queued = [root], {root}
            for _ in range(int(size)):
                if not cells:
                    break
                i = len(cells) - 1 if kind == 1 else int(rng.integers(len(cells)))
                r, c = cells.pop(i)
                mask[r, c] = True
                for j in rng.permutation(4):
                    dr, dc = DIR4[int(j)]
                    p = r + dr, c + dc
                    if _inside(allowed, *p) and p not in queued:
                        cells.append(p)
                        queued.add(p)
    return mask & allowed


def _sample_pair(rng, length, normal, teaching, first=False, test=False):
    need_depth = 6 if teaching or (test and length == 4) else length - 1
    lower = max(11, 2 * need_depth + 5)
    for _attempt in range(180):
        h, w = int(rng.integers(lower, 31)), int(rng.integers(lower, 31))
        if test and rng.random() < .15:
            h = w = 30
        walls = _layout(rng, (h, w), 2 * need_depth + 1, rich=first or test)
        candidates = []
        regions = _components(~walls)
        depths = [_depth(region) for region in regions]
        for ri, (region, depth) in enumerate(zip(regions, depths)):
            if depth.max() >= need_depth:
                # The two-entry demonstration also teaches that a neighboring
                # room with foreground on depth two does not receive the flash.
                if teaching and not first and not any(j != ri and rd.max() >= 2
                                                     for j, rd in enumerate(depths)):
                    continue
                for key in _corners(region, depth, length):
                    options = _gate_options(region, depth, key, normal)
                    if options:
                        candidates.append((region, depth, key, options))
        if not candidates:
            continue
        region, depth, key, options = candidates[int(rng.integers(len(candidates)))]
        run = options[int(rng.integers(len(options)))]
        bar_length = int(rng.integers(2, min(9, len(run)) + 1))
        start = int(rng.integers(len(run) - bar_length + 1))
        gate = run[start:start + bar_length]
        if first:
            fan = region & (_gate_distance((h, w), gate) == depth)
            if not np.any(fan & (depth >= 2 * length) & (depth % (2 * length) == 0)):
                continue
        palette = [int(c) for c in rng.permutation(10)]
        background, wall, foreground = palette[:3]
        period = tuple(palette[3:3 + length])
        allowed = ~walls
        for p in key + gate:
            allowed[p] = False
        for _scene in range(12):
            objects = _objects(rng, allowed)
            if objects.sum() <= bar_length + 1:
                continue
            inp = np.full((h, w), background, dtype=np.int8)
            inp[walls] = wall
            inp[objects] = foreground
            for p, color in zip(key, period):
                inp[p] = color
            for p in gate:
                inp[p] = period[0]
            if np.count_nonzero(inp == background) <= inp.size / 2:
                continue
            try:
                info = _parse(inp)
            except AssertionError:
                continue
            if teaching:
                evidence = _evidence(inp, info)
                required = ("clear", "overwrite", "remote_flash", "preserved", "empty_flash",
                            "fanning", "no_A_diff", "no_B_diff", "no_broadcast_diff")
                if first:
                    required += ("flash_overwrite", "repeated", "concave", "box_diff", "manhattan_diff", "ties",
                                 "remote_component", "mixed_object")
                else:
                    required += ("inactive_flash",)
                if not all(evidence[k] for k in required):
                    continue
            return {"input": inp, "output": _paint(inp, info)}
    raise RuntimeError("bounded search exhausted while constructing teaching geometry")


def _integer(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ns = _integer("num_train", num_train, 2, 10), _integer("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    order = rng.permutation(4).tolist()
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            if split == "train":
                length = 3 if index == 0 else 2 if index == 1 else int(rng.integers(2, 4))
                side = order[index % 4]
            else:
                length = 4 if index == 0 else int(rng.integers(2, 5))
                side = order[2] if index == 0 else int(rng.integers(4))
            for _duplicate in range(20):
                pair = _sample_pair(rng, length, DIR4[side], teaching=split == "train" and index < 2,
                                    first=split == "train" and index == 0, test=split == "test")
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("unable to construct distinct inputs")
            seen.add(signature)
            puzzle[split].append(pair)
    return puzzle


def _pixel_check(inp, info):
    """Coordinate-by-coordinate validator, including consumption of the key."""
    d = _depth_bfs(info["active"])
    assert np.array_equal(d, info["depth"])
    out = inp.copy()
    L = len(info["period"])
    for r, c in np.ndindex(inp.shape):
        if not info["active"][r, c]:
            continue
        level = int(d[r, c])
        nearest_gate = min(max(abs(r - gr), abs(c - gc)) for gr, gc in info["gate"])
        if (r, c) in info["key"]:
            out[r, c] = info["background"]
        if nearest_gate == level:
            out[r, c] = (info["background"] if level % 2 else info["period"][(level // 2) % L])
        if level % 2 == 0 and (level // 2) % L == L - 1 and inp[r, c] == info["foreground"]:
            out[r, c] = info["period"][-1]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(out, _pixel_check(inp, info)), "incorrect scanner output"
            assert np.array_equal(out, solve(inp))
            assert not np.array_equal(inp, out)
            assert np.array_equal(inp == info["wall"], out == info["wall"]), "wall changed"
            assert np.array_equal(out[~info["active"]], inp[~info["active"]]), "inactive room changed"
            assert set(np.unique(out)).issubset(set(np.unique(inp)))
            records[split].append((info, _evidence(inp, info)))
    train = records["train"]
    assert {2, 3}.issubset({len(i["period"]) for i, _ in train}), "teach variable clock lengths"
    for feature in ("clear", "overwrite", "flash_overwrite", "remote_flash", "preserved",
                    "empty_flash", "repeated", "fanning", "ties", "concave", "box_diff",
                    "manhattan_diff", "no_A_diff", "no_B_diff", "no_broadcast_diff",
                    "remote_component", "mixed_object", "inactive_flash"):
        assert any(e[feature] for _, e in train), "missing demonstration: " + feature
    assert len({i["normal"] for i, _ in train}) >= 2
    return True


def _puzzle_key(puzzle):
    return tuple((s, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]))
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed invariant/variation suite plus counts and metamorphic checks."""
    records, puzzles, shapes, geometry, colors = [], set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                record = dict(split=split, shape=pair["input"].shape,
                              regions=len(info["regions"]), period=len(info["period"]),
                              bar=len(info["gate"]), side=DIR4.index(info["normal"]),
                              depth=int(info["depth"].max()), objects=int(np.sum(pair["input"] == info["foreground"])),
                              object_components=len(_components(pair["input"] == info["foreground"])),
                              lane_cells=int(info["lane"].sum()),
                              wall_components=len(_components(pair["input"] == info["wall"])),
                              core_components=max(len(_components(info["active"] & (info["depth"] >= d)))
                                                  for d in range(int(info["depth"].max()) + 1)),
                              **_evidence(pair["input"], info))
                records.append(record)
                shapes.add(pair["input"].shape)
                geometry.add(_key((pair["input"] == info["wall"]).astype(np.int8)))
                colors.add((info["background"], info["wall"], info["foreground"], info["period"]))
    assert len(puzzles) == 42
    assert len(shapes) >= 60 and len(geometry) >= 180 and len(colors) >= 180
    assert {r["period"] for r in records} == {2, 3, 4}
    assert len({r["regions"] for r in records}) >= 5
    assert len({r["bar"] for r in records}) >= 6
    assert len({r["object_components"] for r in records}) >= 6
    assert all({r["side"] for r in records if r["split"] == s} == set(range(4)) for s in ("train", "test"))
    assert any(r["shape"] == (30, 30) for r in records)
    # Every legal count combination, including two-example teaching sets.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9381 + 31 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
            validate_generated_puzzle(p)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    assert _puzzle_key(p) == _puzzle_key(generate_puzzle(0))
    assert _puzzle_key(p) != _puzzle_key(generate_puzzle(1))
    # Rotations, reflections, arbitrary palette permutations and list inputs.
    for pair in p["train"] + p["test"]:
        for k in range(4):
            a, b = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
            assert np.array_equal(solve(a), b)
            assert np.array_equal(solve(a[:, ::-1]), b[:, ::-1])
        palette = np.array([8, 4, 9, 0, 2, 7, 6, 1, 5, 3])
        assert np.array_equal(solve(palette[pair["input"]].tolist()), palette[pair["output"]])
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted wall accepted")
    # Report actual geometry and interactions rather than only normalized hashes.
    report = dict(seeds=list(HOST_SEEDS), puzzles=len(puzzles), pairs=len(records),
                  shapes=len(shapes), divider_geometries=len(geometry), palettes=len(colors),
                  region_counts=sorted({r["regions"] for r in records}),
                  bar_lengths=sorted({r["bar"] for r in records}),
                  object_component_counts=sorted({r["object_components"] for r in records}),
                  records=records, verification="generator-owned checks; no independent semantic approval")
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    return report
