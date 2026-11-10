"""A finite stencil printer whose counters also cast shadows.

No parent modules or files are read. All solving information is in the input.
Generation keeps only one puzzle and bounded candidate grids in memory; it
creates no files, logs, execution traces, or persistent seed-dependent state.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np


CHECK_SEEDS = (0, 63826907, 853678722, 438255123, 897655050, 925779457,
               289841269, 648025424, 249239778, 532632061, 563935660,
               169081730, 618879706, 658039216, 106186830, 782979810,
               408064632, 474795731, 299773699, 739865987, 995995049,
               609922278, 602197295, 639002092, 763821371, 82160123,
               304061578, 544931283, 139527086, 309668819, 950198786,
               975226011, 585262445, 38360227, 407800405, 640969912,
               483177603, 872511638, 177893678, 786250030, 21305484, 346278731)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer color grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(a):
    return (a.shape, a.astype(np.int8, copy=False).tobytes())


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (diagonal or abs(dy) + abs(dx) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                point = (y + dy, x + dx)
                if point in remaining:
                    remaining.remove(point)
                    group.append(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _fit_rail(values, sign):
    """Fit the longest terminal stair, including its digital boundary phase."""
    ys = np.asarray(values, dtype=int)
    xs = np.arange(len(ys)) - len(ys) + 1
    candidates = []
    for period in (1, 2, 3):
        for phase in range(period):
            predicted = ys[-1] + sign * ((xs + phase) // period)
            length = 0
            for same in (predicted == ys)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2 * period + 1 and abs(int(ys[-1] - ys[-length])) >= 2:
                candidates.append((length, -period, phase))
    if not candidates:
        raise ValueError("two complete outward stair steps must be visible")
    length, negative_period, phase = max(candidates)
    return (-negative_period, phase, int(ys[-1]), length)


def _decode(input_grid):
    """Recognize one closed reservoir, its stencil and its open terminal rails."""
    grid = _grid(input_grid)
    background = int(np.bincount(grid.ravel(), minlength=10).argmax())
    _need(np.count_nonzero(grid == background) > grid.size // 2,
          "background must occupy more than half the input")
    matches = []
    for rotation in range(4):
        a = np.rot90(grid, -rotation)
        for color in np.unique(a):
            wall = int(color)
            if wall == background:
                continue
            points = np.argwhere(a == wall)
            if len(points) < 13:
                continue
            rear, mouth = int(points[:, 1].min()), int(points[:, 1].max())
            if mouth - rear < 5 or mouth >= a.shape[1] - 2:
                continue
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 5 or np.any(np.diff(cap) != 1):
                continue
            upper, lower, valid = [], [], True
            for x in range(rear + 1, mouth + 1):
                rows = np.flatnonzero(a[:, x] == wall)
                gaps = np.flatnonzero(np.diff(rows) > 1)
                if len(gaps) != 1:
                    valid = False
                    break
                j = int(gaps[0])
                upper.append(int(rows[j]))
                lower.append(int(rows[j + 1]))
            if not valid:
                continue
            try:
                fits = (_fit_rail(upper, -1), _fit_rail(lower, 1))
            except ValueError:
                continue
            if len(_components(a == wall, True)) != 1:
                continue
            chamber = np.zeros(a.shape, dtype=bool)
            for x, (lo, hi) in enumerate(zip(upper, lower), rear + 1):
                chamber[lo + 1:hi, x] = True
            markers = np.unique(a[chamber & (a != background) & (a != wall)])
            if len(markers) != 1:
                continue
            ink = int(markers[0])
            source = a == ink
            if np.any(source & ~chamber) or len(_components(source)) != 1:
                continue
            source_points = np.argwhere(source)
            y0, x0 = map(int, source_points.min(axis=0))
            y1, x1 = map(int, source_points.max(axis=0) + 1)
            tile = source[y0:y1, x0:x1].copy()
            cone = np.zeros(a.shape, dtype=bool)
            for x in range(mouth + 1, a.shape[1]):
                limits = [value + sign * ((x - mouth + phase) // period)
                          for (period, phase, value, _), sign in zip(fits, (-1, 1))]
                cone[:, x] = ((np.arange(a.shape[0]) > limits[0])
                              & (np.arange(a.shape[0]) < limits[1]))
            screens = (a != background) & (a != wall) & ~source
            objects = _components(screens)
            # Touching colors form one object; diagonal-only contacts are excluded.
            if len(objects) != len(_components(screens, True)):
                continue
            qualifying = [obj for obj in objects if any(cone[y, x] for y, x in obj)]
            if not qualifying:
                continue
            effective = screens & cone
            shadow = np.zeros(a.shape, dtype=bool)
            # Later screens do not widen a previous screen's lane shadow.
            for y in range(a.shape[0]):
                xs = np.flatnonzero(effective[y])
                if len(xs):
                    shadow[y, int(xs[0]) + 1:] = True
            matches.append(dict(a=a, rotation=rotation, background=background,
                                wall=wall, ink=ink, rear=rear, mouth=mouth,
                                fits=fits, chamber=chamber, tile=tile, y0=y0,
                                source_origin=(y0, x0), cone=cone, screens=screens,
                                objects=objects, qualifying=qualifying,
                                count=len(qualifying), effective=effective, shadow=shadow))
    _need(len(matches) == 1, "input must have exactly one emitter/stencil interpretation")
    return matches[0]


def _render(d, count=None, ignore_cone=False, ignore_shadow=False,
            solid=False, transform=0):
    """Input-only modular rendering; alternatives are used for witness checks."""
    a, tile = d["a"], d["tile"]
    if transform == 1:
        tile = tile[:, ::-1]
    elif transform == 2:
        tile = tile[::-1]
    elif transform == 3:
        tile = tile[::-1, ::-1]
    n = d["count"] if count is None else count
    h, w = tile.shape
    out = a.copy()
    for x in range(d["mouth"] + 1, a.shape[1]):
        offset = x - d["mouth"] - 1
        if offset >= n * (w + 1) - 1:
            continue
        phase = offset % (w + 1)
        if phase == w and not solid:
            continue
        for row in range(h):
            y = d["y0"] + row
            if not 0 <= y < a.shape[0]:
                continue
            if (solid or tile[row, phase]) and a[y, x] == d["background"]:
                if (ignore_cone or d["cone"][y, x]) and (ignore_shadow or not d["shadow"][y, x]):
                    out[y, x] = d["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input only; never uses a seed, examples or generation state."""
    d = _decode(input_grid)
    return np.rot90(_render(d), d["rotation"]).copy()


def _motif(h, w, rng, witness):
    if w == 1:
        return np.ones((h, w), dtype=bool)
    mode = int(rng.integers(4 if witness else 5))
    mask = np.ones((h, w), dtype=bool)
    if mode == 0:
        target = max(h + w - 1, int(h * w * rng.uniform(.35, .8)))
        for flat in rng.permutation(h * w):
            if mask.sum() <= target:
                break
            y, x = divmod(int(flat), w)
            mask[y, x] = False
            if (not mask[0].any() or not mask[-1].any()
                    or not mask[:, 0].any() or not mask[:, -1].any()
                    or len(_components(mask)) != 1):
                mask[y, x] = True
    elif mode == 1:
        mask[:] = False
        spine = int(rng.integers(w))
        mask[:, spine] = True
        for y in range(h):
            end = int(rng.integers(w))
            mask[y, min(end, spine):max(end, spine) + 1] = True
        mask[int(rng.integers(h)), :] = True
    elif mode == 2:
        mask[1:-1, 1:-1] = False
        if h > 5:
            mask[int(rng.integers(2, h - 2)), :] = True
    elif mode == 3:
        mask[:] = False
        mask[::2, :] = True
        side = int(rng.integers(2))
        for y in range(1, h, 2):
            mask[y, -1 if side else 0] = True
            side = 1 - side
    return mask


@lru_cache(maxsize=24)
def _geometry_options(n, w):
    # Fixed tiny cache of geometry choices, not examples or solutions.
    result = []
    for p in range(1, 4):
        for q in range(1, 4):
            for rear in range(3):
                for left in range(3):
                    for right in range(3):
                        for shoulder in (1, 2):
                            mouth = rear + w + left + right + shoulder + 2 * max(p, q) + 1
                            minimum_width = mouth + n * (w + 1) + w + 2
                            if minimum_width <= 30:
                                result.append((p, q, rear, left, right, shoulder, mouth, minimum_width))
    return tuple(result)


def _make_world(rng, n, h, w, palette, witness):
    bg, wall, ink, _, _ = palette
    options = _geometry_options(n, w)
    if not options:
        return None
    p, q, rear, left, right, shoulder, mouth, minimum_width = options[int(rng.integers(len(options)))]
    width = int(rng.integers(minimum_width, min(30, minimum_width + 3) + 1))
    top = int(rng.integers(3, 6))
    height = top + h + int(rng.integers(3, 6))
    center = top + (h - 1) // 2 + int(rng.integers(-2, 3))
    neck_upper, neck_lower = center - int(rng.integers(1, 3)), center + int(rng.integers(1, 3))
    bay_length = w + left + right
    upper, lower = [], []
    for _ in range(bay_length):
        upper.append(top - int(rng.integers(1, min(top, 3) + 1)))
        lower.append(top + h - 1 + int(rng.integers(1, min(height - top - h, 3) + 1)))
    for i in range(1, shoulder + 1):
        upper.append(int(round(upper[bay_length - 1] + (neck_upper - upper[bay_length - 1]) * i / shoulder)))
        lower.append(int(round(lower[bay_length - 1] + (neck_lower - lower[bay_length - 1]) * i / shoulder)))
    phase_top, phase_bottom = int(rng.integers(p)), int(rng.integers(q))
    flare_start = rear + len(upper) + 1
    for t in range(2 * max(p, q) + 1):
        upper.append(neck_upper - (t + phase_top) // p)
        lower.append(neck_lower + (t + phase_bottom) // q)
    if min(upper) < 0 or max(lower) >= height:
        return None
    a = np.full((height, width), bg, dtype=np.int8)
    a[upper[0]:lower[0] + 1, rear] = wall
    for index, (lo, hi) in enumerate(zip(upper, lower)):
        x = rear + 1 + index
        a[lo, x] = a[hi, x] = wall
        if index and x < flare_start:
            old_lo, old_hi = upper[index - 1], lower[index - 1]
            # Reservoir corners are connected on the exterior side. Their inner
            # faces stay exactly lo/hi. The periodic flare itself is always thin.
            col = x if lo >= old_lo else x - 1
            a[min(lo, old_lo):max(lo, old_lo) + 1, col] = wall
            col = x if hi <= old_hi else x - 1
            a[min(hi, old_hi):max(hi, old_hi) + 1, col] = wall
    tile = _motif(h, w, rng, witness)
    source_x = rear + 1 + left
    region = a[top:top + h, source_x:source_x + w]
    if np.any(region != bg):
        return None
    region[tile] = ink
    cone = np.zeros(a.shape, dtype=bool)
    chamber = np.zeros(a.shape, dtype=bool)
    for x, (lo, hi) in enumerate(zip(upper, lower), rear + 1):
        chamber[lo + 1:hi, x] = True
    for x in range(mouth + 1, width):
        t = x - flare_start
        lo = neck_upper - (t + phase_top) // p
        hi = neck_lower + (t + phase_bottom) // q
        cone[:, x] = (np.arange(height) > lo) & (np.arange(height) < hi)
    protected = [top + r for r in range(h) if tile[r, -1] and cone[top + r, mouth + 1]]
    if not protected:
        return None
    protected_row = protected[int(rng.integers(len(protected)))]
    return dict(a=a, cone=cone, chamber=chamber, tile=tile, top=top, mouth=mouth, n=n,
                protected=protected_row, palette=palette)


_SHAPES = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 0)),
           ((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (2, 0)),
           ((0, 0), (1, 0), (1, 1)),
           ((0, 0), (0, 1), (1, 0), (1, 1)),
           ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)))


def _place_screens(world, rng, witness):
    a, cone = world["a"], world["cone"]
    bg, wall, ink, c1, c2 = world["palette"]
    n, w = world["n"], world["tile"].shape[1]
    start = world["mouth"] + 1
    end = start + n * (w + 1) - 2
    occupied = np.zeros(a.shape, dtype=bool)

    def legal(points, qualifying=True):
        for y, x in points:
            if not (0 <= y < a.shape[0] and 0 <= x < a.shape[1]) or a[y, x] != bg:
                return False
            if y == world["protected"]:
                return False
            if not qualifying and world["chamber"][y, x]:
                return False
            if occupied[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any():
                return False
        return any(cone[y, x] for y, x in points) == qualifying

    def put(points, index, multicolor=False):
        for j, (y, x) in enumerate(points):
            a[y, x] = (c2 if multicolor and j % 2 else (c1 if index % 2 == 0 else c2))
            occupied[y, x] = True

    placed = 0
    if witness and n == 2:
        # An L and a bar have six cells in total. The second object lies in
        # an already shadowed lane, but must still buy another copy.
        rows = [world["top"] + r for r in range(world["tile"].shape[0])
                if world["tile"][r, -1]]
        rng.shuffle(rows)
        for y in rows:
            first = [(y, start), (y + 1, start), (y + 1, start + 1)]
            second = [(y, start + 4 + k) for k in range(3)]
            if cone[y, start] and legal(first) and legal(second):
                put(first, 0)
                if not legal(second):
                    return False
                # Same color: color counting is not object counting.
                put(second, 0)
                placed = 2
                break
        if placed != 2:
            return False
    while placed < n:
        shape_options = _SHAPES[1:3] if witness else _SHAPES
        shape = shape_options[int(rng.integers(len(shape_options)))]
        candidates = []
        for x in range(start, end + 1):
            for y in range(a.shape[0]):
                points = [(y + dy, x + dx) for dy, dx in shape]
                if max(xx for _, xx in points) <= end and legal(points):
                    candidates.append(points)
        if not candidates:
            # A singleton is a bounded fallback only for non-teaching objects.
            if witness:
                return False
            shape = _SHAPES[0]
            candidates = [[(int(y), int(x))] for y, x in np.argwhere(cone)
                          if start <= x <= end and legal([(int(y), int(x))])]
            if not candidates:
                return False
        points = candidates[int(rng.integers(len(candidates)))]
        put(points, placed, multicolor=(witness and placed == 0) or rng.random() < .25)
        placed += 1
    # Both teaching cases have an additional inert domino. It has no effect on
    # either the count or the shadows, even when it shares an effective color.
    inert_count = 1 if witness else int(rng.integers(3))
    for index in range(inert_count):
        shape = _SHAPES[1] if witness else _SHAPES[int(rng.integers(3))]
        candidates = []
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                points = [(y + dy, x + dx) for dy, dx in shape]
                if legal(points, False):
                    candidates.append(points)
        if not candidates:
            return False
        put(candidates[int(rng.integers(len(candidates)))], index)
    return True


def _construct_output(world):
    """Construction uses explicit copies and the union of blocker rays."""
    a, tile, cone = world["a"], world["tile"], world["cone"]
    bg, wall, ink, _, _ = world["palette"]
    screens = (a != bg) & (a != wall) & (a != ink)
    blocked = np.zeros(a.shape, dtype=bool)
    for y, x in np.argwhere(screens & cone):
        blocked[y, x + 1:] = True
    out = a.copy()
    h, w = tile.shape
    for k in range(world["n"]):
        x0 = world["mouth"] + 1 + k * (w + 1)
        for row, col in np.argwhere(tile):
            y, x = world["top"] + int(row), x0 + int(col)
            if x < a.shape[1] and cone[y, x] and not blocked[y, x] and a[y, x] == bg:
                out[y, x] = ink
    return out


def _facts(d):
    expected = _render(d)
    start, w = d["mouth"] + 1, d["tile"].shape[1]
    changes = expected != d["a"]
    per_copy = [int(changes[:, start + k * (w + 1):start + k * (w + 1) + w].sum())
                for k in range(d["count"])]
    centered_twice = d["fits"][0][2] + d["fits"][1][2] - d["tile"].shape[0] + 1
    centered_origins = {centered_twice // 2, (centered_twice + 1) // 2}
    registration = all(not np.array_equal(expected, _render(dict(d, y0=y)))
                       for y in centered_origins)
    object_lanes = [{y for y, x in obj if d["cone"][y, x]} for obj in d["qualifying"]]
    repeated_objects = any(object_lanes[i] & object_lanes[j]
                           for i in range(len(object_lanes)) for j in range(i))
    multicolor = any(len({int(d["a"][y, x]) for y, x in obj}) > 1 for obj in d["qualifying"])
    return dict(clipping=not np.array_equal(expected, _render(d, ignore_cone=True)),
                shadow=not np.array_equal(expected, _render(d, ignore_shadow=True)),
                stop=not np.array_equal(expected, _render(d, count=d["count"] + 1)),
                holes=not np.array_equal(expected, _render(d, solid=True)),
                flips=tuple(not np.array_equal(expected, _render(d, transform=k)) for k in (1, 2, 3)),
                per_copy=per_copy,
                area=sum(map(len, d["qualifying"])),
                inert=len(d["objects"]) - d["count"],
                count_all=not np.array_equal(expected, _render(d, count=len(d["objects"]))),
                count_pixels=not np.array_equal(expected, _render(d, count=int(d["effective"].sum()))),
                count_lanes=not np.array_equal(expected, _render(d, count=int(np.any(d["effective"], axis=1).sum()))),
                registration=registration, repeated_objects=repeated_objects, multicolor=multicolor)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shared_palette = rng.random() < .2
    shared_direction = rng.random() < .2
    base_palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    base_rotation = int(rng.integers(4))
    shift = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", num_train), ("test", num_test)):
        for index in range(amount):
            witness = split == "train" and index < 2
            for attempt in range(96):
                if witness:
                    n, h, w = index + 2, int(rng.integers(9, 13)), 3
                elif split == "train":
                    n, h, w = int(rng.integers(1, 4)), int(rng.integers(7, 13)), int(rng.integers(1, 4))
                else:
                    # Each query extrapolates one structural quantity, while
                    # the cone, phase and shadow physics remain unchanged.
                    mode = (shift + index) % 3
                    n = int(rng.integers(4, 6)) if mode == 0 else int(rng.integers(1, 4))
                    h = int(rng.integers(13, 16)) if mode == 1 else int(rng.integers(7, 13))
                    w = 4 if mode == 2 else int(rng.integers(1, 3 if mode == 0 else 4))
                    if mode == 2:
                        n = int(rng.integers(1, 3))
                palette = base_palette if shared_palette else tuple(map(int, rng.choice(10, 5, replace=False)))
                world = _make_world(rng, n, h, w, palette, witness)
                if world is None or not _place_screens(world, rng, witness):
                    continue
                canonical_out = _construct_output(world)
                rotation = base_rotation if shared_direction else int(rng.integers(4))
                pair = {"input": np.rot90(world["a"], rotation).copy(),
                        "output": np.rot90(canonical_out, rotation).copy()}
                if _key(pair["input"]) in seen:
                    continue
                try:
                    d = _decode(pair["input"])
                    f = _facts(d)
                except (AssertionError, ValueError):
                    continue
                if (d["count"] != n or not f["stop"] or not all(f["per_copy"])
                        or not np.array_equal(_render(d), np.rot90(pair["output"], -d["rotation"]))):
                    continue
                if witness and not (f["clipping"] and f["shadow"] and f["holes"]
                                    and all(f["flips"]) and f["area"] == 6
                                    and f["count_all"] and f["count_pixels"]
                                    and f["registration"] and (index == 0 or f["count_lanes"])):
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair["input"]))
                break
            else:
                raise RuntimeError(f"bounded construction exhausted for {split} pair {index}")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape and _key(inp) not in seen, "shape mismatch or duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            expected = np.rot90(_render(d), d["rotation"])
            _need(np.array_equal(expected, out), "output disagrees with input-only rule")
            aout = np.rot90(out, -d["rotation"])
            changed = aout != d["a"]
            _need(changed.any(), "identity pair")
            _need(np.all(d["a"][changed] == d["background"]) and np.all(aout[changed] == d["ink"]),
                  "only blank cells may acquire source ink")
            _need(not np.any(changed & (~d["cone"] | d["shadow"])), "ink escaped cone or crossed a shadow")
            _need(not np.any(changed[:, :d["mouth"] + 1]), "source reservoir must be preserved")
            _need(1 <= d["count"] <= 5 and 7 <= d["tile"].shape[0] <= 15
                  and 1 <= d["tile"].shape[1] <= 4, "outside sampled family bounds")
            f = _facts(d)
            _need(f["stop"] and all(f["per_copy"]), "count has no observable finite-print effect")
            analyses[split].append((d, f))
    train = analyses["train"]
    _need(any(a[0]["count"] != b[0]["count"] and a[1]["area"] == b[1]["area"] == 6
              for a in train for b in train), "missing equal-area/different-object-count demonstrations")
    for name in ("clipping", "shadow", "holes", "count_all", "count_pixels", "count_lanes", "registration",
                 "repeated_objects", "multicolor"):
        _need(any(f[name] for _, f in train), "missing training witness: " + name)
    for transform in range(3):
        _need(any(f["flips"][transform] for _, f in train), "ambiguous stencil reflection")
    maxima = (max(d["count"] for d, _ in train),
              max(d["tile"].shape[0] for d, _ in train),
              max(d["tile"].shape[1] for d, _ in train))
    for d, _ in analyses["test"]:
        values = (d["count"], *d["tile"].shape)
        _need(any(v > m for v, m in zip(values, maxima)), "query has no intended structural extrapolation")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.astype(np.int8, copy=False).tobytes())
    return digest.hexdigest()


def _hand_checks():
    """Hand-specified cells, independent of sampled profiles/output construction."""
    a = np.zeros((17, 26), dtype=np.int8)
    a[1:14, 1] = 8
    a[1, 2:6] = a[13, 2:6] = 8
    a[1:7, 6] = a[8:14, 6] = 8
    for x, lo, hi in ((7, 6, 8), (8, 5, 9), (9, 4, 10)):
        a[lo, x] = a[hi, x] = 8
    rows = ("110", "100", "111", "010", "110", "100", "111", "001", "011")
    for y, row in enumerate(rows, 3):
        for x, value in enumerate(row, 2):
            if value == "1":
                a[y, x] = 3
    for y, x in ((6, 10), (7, 10), (7, 11), (6, 14), (6, 15), (6, 16), (0, 10), (0, 11)):
        a[y, x] = 5
    out = solve(a)
    _need(out[3, 10] == 0 and out[3, 11] == 3,
          "thin virtual boundary must stay blank, then admit the next phase")
    _need(out[6, 11] == 0 and out[7, 14] == 0 and out[9, 14] == 3,
          "screen shadows must be straight and lane-specific")
    _need(np.all(out[:, 13] == 0) and out[9, 16] == 3 and out[9, 18] == 0,
          "three-column tiles need one separator and exactly two copies")
    _need(np.array_equal(out[a != 0], a[a != 0]), "existing material must survive")
    # Split a three-cell bar into a singleton and a domino: the occupied area
    # stays six, but there are now three qualifying objects, so a third copy.
    split = a.copy()
    split[6, 15], split[6, 17] = 0, 5
    _need(solve(split)[9, 18] == 3, "grouping, not area, determines print length")
    # One object can have two colors; only that preserved input cell changes.
    mixed = a.copy()
    mixed[6, 10] = 6
    expected = out.copy()
    expected[6, 10] = 6
    _need(np.array_equal(solve(mixed), expected), "object connectivity must ignore screen color")
    # A boundary-straddling object counts once. Its outer boundary cell cannot
    # shadow a row that enters the widening footprint one column later.
    straddle = a.copy()
    straddle[3:5, 10] = 6
    projected = solve(straddle)
    _need(projected[9, 18] == 3 and projected[3, 11] == 3 and projected[4, 14] == 0,
          "only footprint-intersecting cells may initiate shadows")


def self_test():
    """42 seeds, streaming metrics, counts, metamorphic checks and corruption."""
    _hand_checks()
    signatures, stencils, outlines, shapes = set(), set(), set(), set()
    directions, counts, periods, palettes, object_shapes = set(), set(), set(), set(), set()
    totals = {k: 0 for k in ("pairs", "clipping", "shadow", "inert", "repeated_lane", "multicolor", "straddling")}
    sample = None
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed in (0, CHECK_SEEDS[-1]):
            _need(_signature(puzzle) == _signature(generate_puzzle(seed)), "seed is not deterministic")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                f = _facts(d)
                shapes.add(pair["input"].shape)
                stencils.add(_key(d["tile"]))
                wall_points = np.argwhere(d["a"] == d["wall"])
                lo, hi = wall_points.min(axis=0), wall_points.max(axis=0) + 1
                outlines.add(_key(d["a"][lo[0]:hi[0], lo[1]:hi[1]] == d["wall"]))
                directions.add((split, d["rotation"]))
                counts.add(d["count"])
                periods.update(fit[0] for fit in d["fits"])
                palettes.add((d["background"], d["wall"], d["ink"]))
                totals["pairs"] += 1
                for name in ("clipping", "shadow"):
                    totals[name] += int(f[name])
                totals["inert"] += int(f["inert"] > 0)
                totals["repeated_lane"] += int(f["repeated_objects"])
                for obj in d["qualifying"]:
                    points = np.asarray(obj)
                    normalized = points - points.min(axis=0)
                    object_shapes.add(tuple(sorted(map(tuple, normalized))))
                    totals["multicolor"] += int(len({int(d["a"][y, x]) for y, x in obj}) > 1)
                    totals["straddling"] += int(any(not d["cone"][y, x] for y, x in obj))
        if seed == 0:
            sample = puzzle
    _need(len(signatures) == 42 and len(stencils) >= 80 and len(outlines) >= 80, "structural diversity collapsed")
    _need(len(shapes) >= 35 and len(palettes) >= 40 and len(object_shapes) >= 6, "geometry/palette diversity collapsed")
    _need(counts == {1, 2, 3, 4, 5} and periods == {1, 2, 3}, "count or cadence coverage missing")
    _need(directions == {(s, k) for s in ("train", "test") for k in range(4)}, "orientation coverage missing")
    _need(totals["multicolor"] > 0 and totals["straddling"] > 0, "object boundary cases missing")
    # All 30 legal explicit count combinations. Each puzzle is discarded before
    # the next; no suite of full grid payloads is retained.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(314159 + nt * 7 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    original = sample["train"][0]
    # Rotation and palette permutation preserve all semantic roles.
    mapping = np.array([9, 7, 4, 0, 6, 1, 8, 2, 5, 3], dtype=np.int8)
    for k in range(4):
        inp = mapping[np.rot90(original["input"], k)]
        expected = mapping[np.rot90(original["output"], k)]
        _need(np.array_equal(solve(inp), expected), "rotation/palette metamorphism failed")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in sample[s]] for s in sample}
    pair = damaged["test"][0]
    y, x = np.argwhere(pair["input"] != pair["output"])[0]
    pair["output"][y, x] = pair["input"][y, x]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), stencils=len(stencils),
                outlines=len(outlines), input_shapes=len(shapes), palettes=len(palettes),
                screen_shapes=len(object_shapes), counts=sorted(counts), periods=sorted(periods),
                orientations=len(directions), **totals)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
