"""Chamber-addressed component routing. Only NumPy and the standard library.

Source segmentation defines a component-relative silhouette. Its color and
enclosed four-region count jointly select a visible pipped receptacle.
"""
from __future__ import annotations

import copy
import numpy as np

FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "bad grid colors")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in EIGHT if diagonal else FOUR:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _mask_of(cells):
    r0, r1, c0, c1 = _box(cells)
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _holes(mask, diagonal=False):
    empty = ~np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = empty.shape
    return [{(r - 1, c - 1) for r, c in region}
            for region in _components(empty, diagonal)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in region)]


def _body_row(row, background):
    nz = np.flatnonzero(row != background)
    if len(nz) < 12:
        return None
    left, right = int(nz[0]), int(nz[-1])
    rail, route = int(row[left]), int(row[left + 1])
    if (right - left + 1 >= 12 and rail == int(row[right])
            and route not in (background, rail)
            and np.all(row[left + 1:right] == route)):
        return left, right, route, rail
    return None


def _canonical_analysis(a, bg):
    h, w = a.shape
    rows = [_body_row(row, bg) for row in a]
    bands, covered = [], set()
    r = 1
    while r < h - 1:
        if rows[r] is None:
            r += 1
            continue
        left, right, route, rail = rows[r]
        top = r
        while r + 1 < h and rows[r + 1] == rows[top]:
            r += 1
        bottom = r
        header = a[top - 1]
        _require(np.all(header[:left] == bg) and np.all(header[right + 1:] == bg),
                 "header extends beyond its body")
        strip = header[left:right + 1]
        _require(strip[0] == strip[-1] == rail and np.all((strip == rail) | (strip == bg)),
                 "missing pipped header")
        pip_mask = strip == bg
        _require(not np.any(pip_mask[:-1] & pip_mask[1:]), "header pips touch")
        k = int(pip_mask.sum())
        bands.append(dict(header=top - 1, top=top, bottom=bottom, left=left,
                          right=right, route=route, rail=rail, count=k))
        covered.update(range(top - 1, bottom + 1))
        r += 1
    _require(bands, "no receptacles")
    first = bands[0]["header"]
    _require(first >= 3 and np.all(a[first - 1] == bg), "missing source separator")
    for x, y in zip(bands, bands[1:]):
        _require(y["header"] >= x["bottom"] + 2, "receptacles touch")
    _require(all(r in covered or np.all(a[r] == bg) for r in range(first, h)),
             "unexplained destination pixels")
    lookup = {(b["route"], b["count"]): b for b in bands}
    _require(len(lookup) == len(bands), "duplicate destination address")
    parts = []
    stage = a[:first]
    for color in map(int, np.unique(stage)):
        if color == bg:
            continue
        for cells in _components(stage == color):
            mask = _mask_of(cells)
            _require(mask.shape[1] <= 10, "source component is too wide")
            holes = _holes(mask)
            key = color, len(holes)
            _require(key in lookup, "source address has no destination")
            band = lookup[key]
            delta = band["bottom"] - max(r for r, c in cells)
            target = {(r + delta, c) for r, c in cells}
            _require(all(band["top"] <= r <= band["bottom"] and
                         band["left"] < c < band["right"] for r, c in target),
                     "stamp crosses destination boundary")
            parts.append(dict(color=color, count=len(holes), cells=cells, mask=mask,
                              holes=holes, band=band, target=target))
    _require(parts, "empty staging field")
    _require({(p["color"], p["count"]) for p in parts} == set(lookup), "unused address")
    occupied = set()
    for p in parts:
        _require(not occupied & p["target"], "target cell collision")
        occupied.update(p["target"])
    return dict(grid=a, background=bg, bands=bands, parts=parts, first=first)


def _analyze(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg) and
             np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "nonuniform perimeter")
    candidates = []
    for turn in range(4):
        try:
            info = _canonical_analysis(np.rot90(a, turn), bg)
        except ValueError:
            continue
        info["turn"] = turn
        candidates.append(info)
    _require(len(candidates) == 1, "ambiguous or missing scene direction")
    return candidates[0]


def _render(info, alternative=None):
    a, parts, bands = info["grid"], info["parts"], info["bands"]
    out = a.copy()
    for p in parts:
        for cell in p["cells"]:
            out[cell] = info["background"]
    for p in parts:
        b = p["band"]
        bottom = max(r for r, c in p["cells"])
        if alternative == "color_only":
            b = max((x for x in bands if x["route"] == p["color"]),
                    key=lambda x: (x["bottom"] - x["top"], x["top"]))
        elif alternative == "count_only":
            b = max((x for x in bands if x["count"] == p["count"]),
                    key=lambda x: (x["bottom"] - x["top"], x["top"]))
        elif alternative == "shared_shift":
            bottom = max(r for q in parts if (q["color"], q["count"]) ==
                         (p["color"], p["count"]) for r, c in q["cells"])
        for r, c in p["cells"]:
            rr = r + b["bottom"] - bottom
            if 0 <= rr < len(out):
                out[rr, c] = b["rail"]
    return np.rot90(out, -info["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All geometry, addresses, colors and direction come from this input."""
    return _render(_analyze(input_grid))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _trim(mask):
    r0, r1, c0, c1 = _box(set(map(tuple, np.argwhere(mask).tolist())))
    return mask[r0:r1, c0:c1].copy()


def _shape(rng, k, maxh=4, maxw=7):
    """Connected positive masks; enclosed regions can grow independently."""
    if k == 0:
        h, w = int(rng.integers(1, maxh + 1)), int(rng.integers(1, maxw + 1))
        if rng.random() < .35:
            return np.ones((h, w), dtype=bool)
        mask = np.zeros((h, w), dtype=bool)
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        goal = int(rng.integers(1, h * w + 1))
        for _ in range(h * w):
            frontier = sorted({(r + dr, c + dc) for r, c in np.argwhere(mask)
                               for dr, dc in FOUR if 0 <= r + dr < h and
                               0 <= c + dc < w and not mask[r + dr, c + dc]})
            if int(mask.sum()) >= goal or not frontier:
                break
            p = _pick(rng, frontier)
            mask[p] = True
            if _holes(mask):
                mask[p] = False
        return _trim(mask)
    eligible = [(h, w) for h in range(3, maxh + 1) for w in range(3, maxw + 1)
                if ((h - 2) * (w - 2) + 1) // 2 >= k]
    _require(eligible, "insufficient cavity capacity")
    for _ in range(40):
        h, w = _pick(rng, eligible)
        mask = np.ones((h, w), dtype=bool)
        if k == 1 and rng.random() < .4:
            mask[1:-1, 1:-1] = False
            return mask
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        for i in rng.permutation(len(cells)):
            r, c = cells[int(i)]
            if any(not mask[r + dr, c + dc] for dr, dc in FOUR):
                continue
            mask[r, c] = False
            if len(_components(mask)) != 1:
                mask[r, c] = True
            if len(_holes(mask)) == k:
                break
        if len(_holes(mask)) != k:
            continue
        goal = int(rng.integers(k, (h - 2) * (w - 2) + 1))
        for i in rng.permutation(len(cells)):
            if int((~mask).sum()) >= goal:
                break
            p = cells[int(i)]
            if not mask[p]:
                continue
            mask[p] = False
            if len(_holes(mask)) != k or len(_components(mask)) != 1:
                mask[p] = True
        # Corner clipping changes the outline without altering topology.
        for p in rng.permutation(4)[:int(rng.integers(3))]:
            corner = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)][int(p)]
            if rng.random() < .4:
                mask[corner] = False
                if len(_holes(mask)) != k or len(_components(mask)) != 1:
                    mask[corner] = True
        return mask
    # A one-row chamber comb is feasible for every sampled count (max four).
    mask = np.ones((3, 2 * k + 1), dtype=bool)
    mask[1, 1:-1:2] = False
    return mask


def _part(mask, key):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    _require(len(_components(mask)) == 1 and len(_holes(_mask_of(cells))) == key[1],
             "shape constructor violated address")
    return dict(key=key, cells=cells)


def _topology_groups(rng, colors):
    x, y = colors[:2]
    h, w = int(rng.integers(4, 7)), int(rng.integers(4, 8))
    ring = np.ones((h, w), dtype=bool)
    ring[1:-1, 1:-1] = False
    fill = ~ring
    two = np.ones((4, 5 + int(rng.integers(2))), dtype=bool)
    two[1, 1] = two[2, 2] = two[2, 3] = False
    two = np.rot90(two, int(rng.choice([0, 2]))).copy()
    return [[_part(ring, (x, 1)), _part(fill, (y, 0))], [_part(two, (x, 2))]]


def _alignment_groups(rng, colors):
    x, y = colors[:2]
    shared = _shape(rng, 0, 3, 4)
    second = _shape(rng, 0, 3, 3)
    groups = [[_part(shared, (x, 0))], [_part(second, (x, 0))],
              [_part(shared, (y, 0))]]
    if rng.random() < .55:
        groups.append([_part(_shape(rng, 1, 3, 5), (x, 1))])
    return groups


def _ordinary_groups(rng, colors, curriculum, is_test):
    x, y, z = colors
    if is_test and curriculum == "scale":
        h, w = int(rng.integers(7, 9)), int(rng.integers(8, 11))
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
        return [[_part(mask, (x, 1))], [_part(_shape(rng, 0, 3, 5), (x, 0))]]
    n = int(rng.choice([2, 2, 3, 3, 4]))
    cap = 4 if is_test and curriculum == "counts" else 2
    ks = list(map(int, rng.choice(cap + 1, 2, replace=False)))
    if is_test and curriculum == "counts":
        ks[1] = int(rng.integers(3, 5))
        ks[0] = int(rng.integers(0, 3))
    keys = [(x, ks[0]), (x, ks[1])]
    if n >= 3:
        keys.append((y, int(rng.choice([0, ks[0], ks[1]]))))
    if n == 4:
        available = [(c, k) for c in (x, y, z) for k in range(cap + 1)
                     if (c, k) not in keys]
        keys.append(_pick(rng, available))
    maxh = 3 if n == 4 else 4 if n == 3 else int(rng.integers(4, 6))
    maxw = 10 if cap == 4 else 7
    masks = [_shape(rng, k, maxh, maxw) for c, k in keys]
    groups = [[_part(mask, key)] for mask, key in zip(masks, keys)]
    # Nesting comes from actual cavities, not a fixed frame template.
    candidates = [(i, j) for i, key in enumerate(keys) if key[1] > 0
                  for j, other in enumerate(keys) if other[1] == 0 and other[0] != key[0]]
    if candidates and rng.random() < .5:
        i, j = _pick(rng, candidates)
        cavity = _pick(rng, _holes(masks[i]))
        occupant = np.zeros(masks[i].shape, dtype=bool)
        for p in cavity:
            occupant[p] = True
        groups[i].append(_part(occupant, keys[j]))
        groups[j] = []
    groups = [g for g in groups if g]
    if rng.random() < .5:
        i = int(rng.integers(n))
        groups.append([_part(_shape(rng, keys[i][1], maxh, maxw), keys[i])])
    return groups


def _assemble(rng, groups, bg, turn, need_independence=False):
    keys = sorted({p["key"] for g in groups for p in g})
    req = {key: max(_mask_of(p["cells"]).shape[0] for g in groups for p in g
                    if p["key"] == key) for key in keys}
    shapes = [_box(set().union(*(p["cells"] for p in g))) for g in groups]
    maxh = max(b - a for a, b, c, d in shapes)
    n = len(keys)
    stage_cap = 30 - sum(req.values()) - 2 * n - 2
    if stage_cap < maxh + int(need_independence):
        return None
    stage = int(rng.integers(maxh + int(need_independence), min(stage_cap, maxh + 5) + 1))
    width = int(rng.integers(max(16, max(d - c for a, b, c, d in shapes) + 7), 31))
    gaps = [1] * n
    depths = dict(req)
    far = 1
    # Spend a random portion of remaining room on spacing and body depth.
    spare = 30 - (stage + sum(req.values()) + 2 * n + 2)
    for _ in range(int(rng.integers(spare + 1))):
        choice = int(rng.integers(2 * n + 1))
        if choice < n:
            depths[keys[choice]] += 1
        elif choice < 2 * n:
            gaps[choice - n] += 1
        else:
            far += 1
    height = 1 + stage + sum(gaps) + n + sum(depths.values()) + far
    grid = np.full((height, width), bg, dtype=np.int8)
    bands, cursor = {}, 1 + stage
    for i in rng.permutation(n):
        key = keys[int(i)]
        route, count = key
        rail = int(rng.choice([c for c in range(10) if c not in (bg, route)]))
        left, right = int(rng.integers(1, 3)), width - 1 - int(rng.integers(1, 3))
        cursor += gaps[int(i)]
        header = cursor
        grid[header, left:right + 1] = rail
        choices = list(range(left + 1, right))
        pips = []
        for j in rng.permutation(len(choices)):
            c = choices[int(j)]
            if all(abs(c - x) > 1 for x in pips):
                pips.append(c)
                if len(pips) == count:
                    break
        if count:
            if len(pips) < count:
                return None
            grid[header, pips[:count]] = bg
        top, bottom = header + 1, header + depths[key]
        grid[top:bottom + 1, left:right + 1] = route
        grid[top:bottom + 1, [left, right]] = rail
        bands[key] = dict(top=top, bottom=bottom, left=left, right=right, rail=rail)
        cursor = bottom + 1
    boxes, placed, occupied = [], [], set()
    order = list(map(int, rng.permutation(len(groups))))
    order.sort(key=lambda i: -sum(len(p["cells"]) for p in groups[i]))
    for i in order:
        g = groups[i]
        a, b, c, d = shapes[i]
        gh, gw = b - a, d - c
        positions = [(r, col) for r in range(1, 2 + stage - gh)
                     for col in range(3, width - 2 - gw)]
        found = False
        for j in rng.permutation(len(positions)):
            r, col = positions[int(j)]
            if any(not (r + gh < rr or rr + hh < r or col + gw < cc or cc + ww < col)
                   for rr, cc, hh, ww in boxes):
                continue
            trial, target = [], set()
            for p in g:
                cells = {(r + y - a, col + x - c) for y, x in p["cells"]}
                band = bands[p["key"]]
                delta = band["bottom"] - max(y for y, x in cells)
                dest = {(y + delta, x) for y, x in cells}
                if (target & dest or occupied & dest or
                        any(not (band["top"] <= y <= band["bottom"] and
                                 band["left"] < x < band["right"]) for y, x in dest)):
                    break
                trial.append(dict(key=p["key"], cells=cells, target=dest, rail=band["rail"]))
                target.update(dest)
            else:
                placed.extend(trial)
                occupied.update(target)
                boxes.append((r, col, gh, gw))
                found = True
                break
        if not found:
            return None
    if need_independence and not any(p["key"] == q["key"] and
            max(r for r, c in p["cells"]) != max(r for r, c in q["cells"])
            for i, p in enumerate(placed) for q in placed[i + 1:]):
        return None
    for p in placed:
        for cell in p["cells"]:
            grid[cell] = p["key"][0]
    # Labels use the construction's visible masks and independently computed
    # placements. solve() separately rediscovers all addresses from the input.
    out = grid.copy()
    for p in placed:
        for cell in p["cells"]:
            out[cell] = bg
    for p in placed:
        for cell in p["target"]:
            out[cell] = p["rail"]
    return {"input": np.rot90(grid, turn).copy(), "output": np.rot90(out, turn).copy()}


def _counts(nt, ne):
    for name, value, lo, hi in (("num_train", nt, 2, 10), ("num_test", ne, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"illegal {name}")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return int(nt), int(ne)


def _key(grid):
    return grid.shape, grid.tobytes()


def _reserve_pair(rng, lesson, curriculum, is_test, bg, colors, turn, serial):
    """Finite constructive reserve: 14+ distinct, geometrically proven scenes.

    It follows the same rule and curriculum. Random packing never turns an
    otherwise legal count request into a probabilistic exhaustion error.
    """
    x, y = colors[:2]
    if lesson == "topology":
        ring = np.ones((4, 4), dtype=bool)
        ring[1:-1, 1:-1] = False
        two = np.ones((4, 5), dtype=bool)
        two[1, 1] = two[2, 2] = two[2, 3] = False
        groups = [[_part(ring, (x, 1)), _part(~ring, (y, 0))], [_part(two, (x, 2))]]
        positions = [(1 + serial // 7, 3 + serial % 7), (2, 20)]
    elif lesson == "alignment":
        shared = np.ones((2, 2), dtype=bool)
        groups = [[_part(shared, (x, 0))], [_part(np.ones((1, 2), dtype=bool), (x, 0))],
                  [_part(shared, (y, 0))]]
        positions = [(1 + serial // 7, 3 + serial % 7), (7, 15), (2, 22)]
    else:
        if is_test and curriculum == "scale":
            mask = np.ones((7, 8), dtype=bool)
            mask[1:-1, 1:-1] = False
            count = 1
            first_top = 1 + (serial // 7) % 2
        elif is_test and curriculum == "counts":
            mask = np.ones((3, 7), dtype=bool)
            mask[1, 1:-1:2] = False
            count, first_top = 3, 1 + serial // 7
        else:
            mask = np.ones((4, 5), dtype=bool)
            mask[1:-1, 1:-1] = False
            count, first_top = 1, 1 + serial // 7
        groups = [[_part(mask, (x, count))], [_part(np.ones((1, 2), dtype=bool), (x, 0))]]
        positions = [(first_top, 3 + serial % 7), (7, 22)]
    keys = sorted({p["key"] for g in groups for p in g})
    depths = {key: 1 + max(_mask_of(p["cells"]).shape[0] for g in groups for p in g
                          if p["key"] == key) for key in keys}
    n = len(keys)
    height = 8 + sum(depths.values()) + 2 * n + 2
    grid = np.full((height, 29), bg, dtype=np.int8)
    bands, cursor = {}, 9
    for i in rng.permutation(n):
        key = keys[int(i)]
        route, count = key
        rail = int(rng.choice([c for c in range(10) if c not in (bg, route)]))
        header = cursor + 1
        top, bottom = header + 1, header + depths[key]
        grid[header, 1:28] = rail
        grid[header, list(range(3, 3 + 2 * count, 2))] = bg
        grid[top:bottom + 1, 1:28] = route
        grid[top:bottom + 1, [1, 27]] = rail
        bands[key] = dict(bottom=bottom, rail=rail)
        cursor = bottom + 1
    placed = []
    for group, (r0, c0) in zip(groups, positions):
        for p in group:
            cells = {(r0 + r, c0 + c) for r, c in p["cells"]}
            for cell in cells:
                grid[cell] = p["key"][0]
            band = bands[p["key"]]
            shift = band["bottom"] - max(r for r, c in cells)
            placed.append((cells, {(r + shift, c) for r, c in cells}, band["rail"]))
    out = grid.copy()
    for cells, target, rail in placed:
        for cell in cells:
            out[cell] = bg
    for cells, target, rail in placed:
        for cell in target:
            out[cell] = rail
    return {"input": np.rot90(grid, turn).copy(), "output": np.rot90(out, turn).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    curriculum = _pick(rng, ["counts", "scale", "direction"])
    turns = list(map(int, rng.permutation(4)))
    mixed = curriculum == "direction" or rng.random() < .4
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            turn = (turns[index % 2] if split == "train" else turns[2 + index % 2]) if mixed else turns[0]
            for _ in range(160):
                palette = list(map(int, rng.permutation(10)))
                bg, colors = palette[0], palette[1:4]
                if split == "train" and index == 0:
                    groups = _topology_groups(rng, colors)
                elif split == "train" and index == 1:
                    groups = _alignment_groups(rng, colors)
                else:
                    groups = _ordinary_groups(rng, colors, curriculum, split == "test")
                pair = _assemble(rng, groups, bg, turn, split == "train" and index == 1)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                try:
                    analysis = _analyze(pair["input"])
                except ValueError:
                    continue
                if not np.array_equal(_render(analysis), pair["output"]):
                    raise AssertionError("construction and input-derived transform disagree")
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                lesson = "topology" if split == "train" and index == 0 else (
                    "alignment" if split == "train" and index == 1 else None)
                # At most 13 inputs precede this pair. Scale reserves have 14
                # distinct source-position variants; all other reserves have 28.
                for serial in range(14 if split == "test" and curriculum == "scale" else 28):
                    pair = _reserve_pair(rng, lesson, curriculum, split == "test", bg, colors, turn, serial)
                    if _key(pair["input"]) not in seen:
                        _require(np.array_equal(solve(pair["input"]), pair["output"]), "reserve labeling error")
                        seen.add(_key(pair["input"]))
                        puzzle[split].append(pair)
                        break
                else:
                    raise AssertionError("finite reserve uniqueness bound violated")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    a, bg, parts = info["grid"], info["background"], info["parts"]
    inversion = any(p["color"] == q["color"] and p["count"] < q["count"] and
                    sum(map(len, p["holes"])) > sum(map(len, q["holes"]))
                    for p in parts for q in parts)
    diagonal = any(len(_holes(p["mask"], True)) < p["count"] for p in parts)
    nested = False
    for p in parts:
        r0, r1, c0, c1 = _box(p["cells"])
        hole_cells = {(r0 + r, c0 + c) for hole in p["holes"] for r, c in hole}
        if hole_cells and all(a[cell] != bg for cell in hole_cells):
            nested = True
    independent = any(p["color"] == q["color"] and p["count"] == q["count"] and
                      max(r for r, c in p["cells"]) != max(r for r, c in q["cells"])
                      for p in parts for q in parts)
    color_tie = any(p["color"] != q["color"] and p["count"] == q["count"] and
                    np.array_equal(p["mask"], q["mask"]) for p in parts for q in parts)
    orientation = any(not np.array_equal(p["mask"], np.flipud(p["mask"])) and
                      not np.array_equal(p["mask"], np.fliplr(p["mask"])) for p in parts)
    return dict(inversion=inversion, diagonal=diagonal, nested=nested,
                independent=independent, color_tie=color_tie, asymmetric=orientation)


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train, test = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            _require(np.array_equal(_render(info), out), "incorrect routing or preserved cells")
            for p in info["parts"]:
                cells8 = next(region for region in _components(
                    info["grid"][:info["first"]] == p["color"], True) if p["cells"] <= region)
                _require(cells8 == p["cells"], "diagonal-only source grouping")
            if split == "train":
                train.append(info)
            else:
                test.append(info)
    features = [_features(info) for info in train]
    for name in ("inversion", "diagonal", "nested", "independent", "color_tie", "asymmetric"):
        _require(any(f[name] for f in features), "missing training evidence: " + name)
    for alternative in ("color_only", "count_only", "shared_shift"):
        _require(any(not np.array_equal(_render(info), _render(info, alternative)) for info in train),
                 "training does not refute " + alternative)
    train_counts = {b["count"] for info in train for b in info["bands"]}
    train_turns = {info["turn"] for info in train}
    tallest = max(p["mask"].shape[0] for info in train for p in info["parts"])
    widest = max(p["mask"].shape[1] for info in train for p in info["parts"])
    for info in test:
        new_count = max(b["count"] for b in info["bands"]) > max(train_counts)
        larger_frame = any(p["mask"].shape[0] > tallest and p["mask"].shape[1] > widest
                           for p in info["parts"])
        _require(new_count or larger_frame or info["turn"] not in train_turns,
                 "test has no structural curriculum shift")


def self_test():
    """Bounded grid-derived variation audit and family-specific regressions."""
    signatures, shapes, structures, masks, sizes, layouts, counts, directions = (set() for _ in range(8))
    totals = dict(pairs=0, nested=0, inversion=0, diagonal=0, independent=0, rail_reuse=0)
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for p in puzzle[split]:
                info = _analyze(p["input"])
                _require(np.array_equal(solve(p["input"].tolist()), p["output"]), "list-input solve failed")
                shapes.add(p["input"].shape)
                roles = {}
                structures.add(tuple(roles.setdefault(int(c), len(roles)) for c in p["input"].flat))
                bands, parts = info["bands"], info["parts"]
                layouts.add(tuple((b["bottom"] - b["top"] + 1, b["right"] - b["left"] + 1)
                                  for b in bands))
                counts.add(tuple(sorted(b["count"] for b in bands)))
                directions.add(info["turn"])
                for part in parts:
                    masks.add((part["mask"].shape, part["mask"].tobytes()))
                    sizes.update(map(len, part["holes"]))
                totals["pairs"] += 1
                for name, present in _features(info).items():
                    if name in totals:
                        totals[name] += int(present)
                routes = {b["route"] for b in bands}
                totals["rail_reuse"] += any(b["rail"] in routes for b in bands)
        validate_generated_puzzle(generate_puzzle(seed, 2, 1))
    _require(len(signatures) == 42, "cross-seed duplicate puzzle")
    _require(len(shapes) >= 20 and len(structures) >= 150 and len(masks) >= 75,
             "insufficient geometric variation")
    _require(len(layouts) >= 35 and len(counts) >= 8 and directions == set(range(4)),
             "insufficient layout/count/direction variation")
    _require(max(sizes) >= 25 and all(totals[k] >= 42 for k in
             ("nested", "inversion", "diagonal", "independent")), "missing topology coverage")
    # Determinism, legal extremes, invalid types/counts.
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p, q = generate_puzzle(1901, nt, ne), generate_puzzle(1901, nt, ne)
        _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                     for k in x), "seed is not deterministic")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Permutations and whole-scene rotations preserve all semantic color roles.
    rng = np.random.default_rng(77)
    lookup = rng.permutation(10)
    for p in puzzles[0]["train"] + puzzles[0]["test"]:
        for turn in range(4):
            inp, out = np.rot90(lookup[p["input"]], turn), np.rot90(lookup[p["output"]], turn)
            _require(np.array_equal(solve(inp), out), "rotation/palette equivariance failed")
    # Relocating pips preserves the address and the entire body stamp.
    p = puzzles[0]["train"][0]
    info = _analyze(p["input"])
    changed = info["grid"].copy()
    expected = np.rot90(p["output"], info["turn"]).copy()
    for b in info["bands"]:
        changed[b["header"], b["left"]:b["right"] + 1] = b["rail"]
        pips = list(range(b["left"] + 1, b["right"], 2))[:b["count"]]
        changed[b["header"], pips] = info["background"]
        expected[b["header"]] = changed[b["header"]]
    _require(np.array_equal(solve(np.rot90(changed, -info["turn"])),
                           np.rot90(expected, -info["turn"])), "pip positions affected transport")
    # An unsampled five-chamber case checks that solving has no count catalog.
    five = np.ones((5, 5), dtype=bool)
    five[1, 1] = five[1, 3] = five[2, 2] = five[3, 1] = five[3, 3] = False
    groups = [[_part(five, (2, 5))], [_part(np.ones((1, 2), dtype=bool), (3, 0))]]
    for _ in range(20):
        higher = _assemble(rng, groups, 0, 0)
        if higher is not None:
            break
    _require(higher is not None and np.array_equal(solve(higher["input"]), higher["output"]),
             "input-only solving failed outside the sampled count range")
    # Directly test reserve scenes even if ordinary rejection sampling succeeds.
    for curriculum in ("counts", "scale", "direction"):
        for turn in range(4):
            sample = {"train": [
                _reserve_pair(rng, "topology", curriculum, False, 0, [2, 3, 4], turn, 27),
                _reserve_pair(rng, "alignment", curriculum, False, 1, [3, 4, 5], (turn + 1) % 4, 27)],
                "test": [_reserve_pair(rng, None, curriculum, True, 2, [4, 5, 6], (turn + 2) % 4, 13)]}
            validate_generated_puzzle(sample)
    broken = copy.deepcopy(puzzles[0])
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    # Removing the topology demonstration leaves correct pairs but insufficient teaching.
    minimal = generate_puzzle(1, 2, 1)
    alignment = next(p for p in minimal["train"] if _features(_analyze(p["input"]))["independent"])
    deficient = copy.deepcopy(minimal)
    deficient["train"] = [alignment, {k: (9 - v).copy() for k, v in alignment.items()}]
    try:
        validate_generated_puzzle(deficient)
    except ValueError:
        pass
    else:
        raise AssertionError("insufficient training accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                normalized_structures=len(structures), component_masks=len(masks),
                depth_span_layouts=len(layouts), count_sets=len(counts), cavity_areas=sorted(sizes),
                directions=sorted(directions), **totals)
