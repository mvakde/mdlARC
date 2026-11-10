"""Self-counting glyph carousel: keyed traversal with component lifetimes.

Only visible input geometry is used by solve.  Generation keeps source masks
and their scan order to construct labels; decoding rediscovers both from pixels.
"""

from itertools import combinations, product
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape),
             "grids must have dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
             "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group, stack = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    group.append(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _crop(mask):
    p = np.argwhere(mask)
    _require(len(p) > 0, "empty glyph")
    lo, hi = p.min(axis=0), p.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy(), tuple(map(int, lo))


def _normalize(a):
    """Rotate the fast vector right; keep the slow vector's handedness."""
    candidates = []
    for rotation in range(4):
        g = np.rot90(a, rotation)
        if min(g.shape) < 7:
            continue
        for up in (False, True):
            row = g[-1] if up else g[0]
            bg, fast = int(row[2]), int(row[1])
            side = g[:-1, 0] if up else g[1:, 0]
            slow, origin = int(side[0]), int(row[0])
            if len({bg, fast, slow, origin}) != 4:
                continue
            pattern = np.where(np.arange(1, g.shape[1]) % 2, fast, bg)
            opposite = g[0, 1:] if up else g[-1, 1:]
            far_side = g[:-1, -1] if up else g[1:, -1]
            if (np.array_equal(row[1:], pattern) and np.all(side == slow)
                    and np.all(opposite == bg) and np.all(far_side == bg)):
                candidates.append((g.copy(), up, bg, rotation, (origin, fast, slow)))
    _require(len(candidates) == 1, "expected one dashed/solid L key")
    return candidates[0]


def _decode(input_grid):
    a = _grid(input_grid)
    g, up, bg, rotation, guides = _normalize(a)
    body = g.copy()
    body[-1 if up else 0, :] = bg
    body[:, 0] = bg
    candidates = []
    for color in map(int, np.unique(body)):
        if color == bg:
            continue
        for group in _components(body == color):
            points = np.array(group)
            r0, c0 = map(int, points.min(axis=0))
            r1, c1 = map(int, points.max(axis=0) + 1)
            h, w = r1-r0, c1-c0
            if min(h, w) < 5 or len(group) != 2*h + 2*w - 4:
                continue
            frame = body[r0:r1, c0:c1]
            if (np.all(frame[[0, -1], :] == color)
                    and np.all(frame[:, [0, -1]] == color)
                    and not np.any(frame[1:-1, 1:-1] == color)):
                candidates.append((r0, r1, c0, c1, color))
    # A rectangular ring inside a glyph is still stencil ink.  Enclosure assigns
    # it to the outermost frame rather than inventing a second module.
    boxes = [b for b in candidates if not any(
        q != b and q[0] < b[0] and b[1] < q[1] and q[2] < b[2] and b[3] < q[3]
        for q in candidates)]
    _require(1 <= len(boxes) <= 5, "expected 1..5 outer frames")
    covered = np.zeros(g.shape, dtype=bool)
    lanes, entries = {}, []
    for r0, r1, c0, c1, color in boxes:
        _require(2 <= c0 < c1 < g.shape[1], "frame lacks space beside the slow key")
        _require((1 <= r0 < r1 < g.shape[0]-1 if up else
                  2 <= r0 < r1 < g.shape[0]), "frame lacks space beside key or far edge")
        _require(not covered[r0:r1, c0:c1].any(), "overlapping frames")
        covered[r0:r1, c0:c1] = True
        inner = g[r0+1:r1-1, c0+1:c1-1]
        inks = set(map(int, np.unique(inner))) - {bg}
        _require(len(inks) == 1 and color not in inks, "glyph needs one local ink")
        ink = next(iter(inks))
        mask = inner == ink
        _require(not mask[0].any() and not mask[-1].any()
                 and not mask[:, 0].any() and not mask[:, -1].any(),
                 "glyph must have background space inside its frame")
        tile, offset = _crop(mask)
        count = len(_components(tile))
        _require(1 <= count <= 4 and count == len(_components(tile, True)),
                 "glyph components must be separate under both connectivities")
        _require(1 <= tile.shape[0] <= 5 and 1 <= tile.shape[1] <= 23,
                 "glyph exceeds family dimensions")
        entry = {"mask": tile, "count": count, "color": color, "ink": ink,
                 "box": (r0, r1, c0, c1), "inner": mask, "offset": offset}
        lanes.setdefault((r0, r1), []).append(entry)
    _require(np.all((body == bg) | covered), "unowned marks outside frames")
    lane_bands = sorted(lanes, reverse=up)
    physical_bands = sorted(lanes)
    _require(len(lanes) <= 3 and all(b[0]-a[1] >= 1 for a, b in
             zip(physical_bands, physical_bands[1:])), "staggered or touching lanes")
    for band in lane_bands:
        lane = sorted(lanes[band], key=lambda e: e["box"][2])
        _require(all(b["box"][2]-a["box"][3] >= 1 for a, b in zip(lane, lane[1:])),
                 "frames must be separated in each lane")
        entries.extend(lane)
    _require(len({e["mask"].shape[1] % 2 for e in entries}) == 1,
             "glyph widths need common parity for exact centering")
    height = sum(e["count"]*(e["mask"].shape[0]+1) for e in entries)-1
    _require(height <= 30, "carousel exceeds output budget")
    return {"entries": entries, "background": bg, "up": up,
            "rotation": rotation, "guides": guides, "lanes": lane_bands,
            "normalized": g}


def _render(entries, bg, mode="carousel"):
    """Center each emitted crop; single secondary/background row between all."""
    if mode == "once":
        counts = [1]*len(entries)
    elif mode == "area":
        counts = [int(e["mask"].sum()) for e in entries]
    elif mode in ("columns", "rows"):
        counts = []
        for e in entries:
            active = np.any(e["mask"], axis=0 if mode == "columns" else 1)
            counts.append(int(np.count_nonzero(active & ~np.r_[False, active[:-1]])))
    else:
        counts = [e["count"] for e in entries]
    if mode == "contiguous":
        sequence = [i for i, n in enumerate(counts) for _ in range(n)]
    else:
        sequence = [i for round_number in range(max(counts))
                    for i, n in enumerate(counts) if round_number < n]
    width = max(e["mask"].shape[1] for e in entries)
    blocks = []
    for i in sequence:
        e = entries[i]
        tile = np.where(e["mask"], e["color"], bg).astype(np.int8)
        block = np.full((tile.shape[0], width), bg, dtype=np.int8)
        left = (width-tile.shape[1])//2
        block[:, left:left+tile.shape[1]] = tile
        if blocks:
            blocks.append(np.full((1, width), bg, dtype=np.int8))
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, training examples, cache, or generation metadata is consulted."""
    d = _decode(input_grid)
    return _render(d["entries"], d["background"])


def _connected(rng, h, w):
    if min(h, w) == 1:
        return np.ones((h, w), dtype=bool)
    mode = int(rng.integers(5))
    if mode == 0:
        return np.ones((h, w), dtype=bool)
    if mode == 1:
        m = np.zeros((h, w), dtype=bool)
        m[[0, -1], :] = True
        m[:, [0, -1]] = True
        return m
    m = np.ones((h, w), dtype=bool)
    target = max(h+w-1, int(rng.integers(1, h*w)))
    for flat in rng.permutation(h*w):
        if m.sum() <= target:
            break
        r, c = divmod(int(flat), w)
        m[r, c] = False
        if (not m.any() or _crop(m)[0].shape != (h, w)
                or len(_components(m)) != 1):
            m[r, c] = True
    return m


def _exact_two_rows(rng, width, area):
    """Random spanning walk plus growth: exact area, one component, tight box."""
    _require(width+1 <= area <= 2*width, "infeasible two-row object")
    seam = int(rng.integers(width))
    m = np.zeros((2, width), dtype=bool)
    m[0, :seam+1] = True
    m[1, seam:] = True
    if rng.integers(2):
        m = m[::-1].copy()
    for _ in range(area-int(m.sum())):
        adjacent = np.zeros_like(m)
        adjacent[1:] |= m[:-1]
        adjacent[:-1] |= m[1:]
        adjacent[:, 1:] |= m[:, :-1]
        adjacent[:, :-1] |= m[:, 1:]
        sites = np.argwhere(adjacent & ~m)
        r, c = sites[int(rng.integers(len(sites)))]
        m[r, c] = True
    return m


def _asymmetric(m):
    return all(not np.array_equal(m, x) for x in (m[::-1], m[:, ::-1], m[::-1, ::-1]))


def _witness_glyph(rng, count, area, parity):
    areas = np.full(count, 2, dtype=int)
    for _ in range(area-2*count):
        areas[int(rng.integers(count))] += 1
    if count == 1:
        choices = [w for w in range((area+1)//2, min(7, area-1)+1) if w % 2 == parity]
        width = max(choices)
        for _ in range(20):
            mask = _exact_two_rows(rng, width, area)
            if _asymmetric(mask):
                return mask
        mask = np.zeros((2, width), dtype=bool)
        mask[0] = True
        mask[1, :area-width] = True
        return mask
    widths = [(int(a)+1)//2 for a in areas]
    gaps = [1]*(count-1)
    minimum = sum(widths)+sum(gaps)
    limit = 9 if count == 2 else 11
    choices = [w for w in range(minimum, limit+1) if w % 2 == parity]
    total = int(rng.choice(choices))
    for _ in range(total-minimum):
        options = [("width", i) for i in range(count) if widths[i] < int(areas[i])-1]
        options += [("gap", i) for i in range(count-1)]
        kind, index = options[int(rng.integers(len(options)))]
        (widths if kind == "width" else gaps)[index] += 1
    mask, start = np.zeros((2, total), dtype=bool), 0
    for i, (w, a) in enumerate(zip(widths, areas)):
        mask[:, start:start+w] = _exact_two_rows(rng, w, int(a))
        start += w+(gaps[i] if i < len(gaps) else 0)
    return mask


def _axis_regions(rng, length, number):
    sizes = [1]*number
    gaps = [1]*(number-1)
    for _ in range(length-(2*number-1)):
        index = int(rng.integers(2*number-1))
        (sizes if index < number else gaps)[index if index < number else index-number] += 1
    ranges, start = [], 0
    for i, size in enumerate(sizes):
        ranges.append((start, start+size))
        start += size+(gaps[i] if i < number-1 else 0)
    return ranges


def _glyph(rng, count, h, w):
    # Independent row/column region partitions admit vertical, horizontal and
    # two-dimensional collections. Projection runs are not the component count.
    options = []
    for nr in range(1, min(count, (h+1)//2)+1):
        for nc in range(1, min(count, (w+1)//2)+1):
            if count <= nr*nc <= count+1:
                sites = list(product(range(nr), range(nc)))
                for occupied in combinations(sites, count):
                    if ({r for r, _ in occupied} == set(range(nr))
                            and {c for _, c in occupied} == set(range(nc))):
                        options.append((nr, nc, occupied))
    nr, nc, occupied = options[int(rng.integers(len(options)))]
    rows, cols = _axis_regions(rng, h, nr), _axis_regions(rng, w, nc)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in occupied:
        r0, r1 = rows[r]
        c0, c1 = cols[c]
        mask[r0:r1, c0:c1] = _connected(rng, r1-r0, c1-c0)
    return mask


def _partitions(widths, max_lanes):
    """All contiguous lane partitions that fit, at most 16 for five modules."""
    result = []
    for cuts in product((False, True), repeat=len(widths)-1):
        groups, start = [], 0
        for i, cut in enumerate(cuts, 1):
            if cut:
                groups.append(list(range(start, i)))
                start = i
        groups.append(list(range(start, len(widths))))
        if len(groups) <= max_lanes and all(
            sum(widths[i] for i in group)+len(group)-1 <= 27 for group in groups):
            result.append(groups)
    return result


def _gaps(rng, occupied, n, extent):
    extra = extent-(occupied+n+1)
    _require(extra >= 0, "packing budget exceeded")
    weights = rng.dirichlet(np.full(n+1, float(rng.choice([.4, 1., 3.]))))
    return (1+rng.multinomial(extra, weights)).tolist()


def _scene(rng, masks, counts, groups, margins, frame_height, configuration,
           witness=False):
    # Every role is local and visible. Frame swatches can reuse guide colors.
    bg, origin, fast, slow = map(int, rng.choice(10, 4, replace=False))
    pool = [c for c in range(10) if c != bg]
    if witness or rng.random() < .65:
        colors = list(map(int, rng.choice(pool, len(masks), replace=False)))
    else:
        colors = list(map(int, rng.choice(pool, len(masks), replace=True)))
    entries, frames = [], []
    for mask, count, color, (left, right) in zip(masks, counts, colors, margins):
        ink = int(rng.choice([c for c in pool if c != color]))
        fw = mask.shape[1]+left+right+2
        frame = np.full((frame_height, fw), bg, dtype=np.int8)
        frame[[0, -1], :] = color
        frame[:, [0, -1]] = color
        spare = frame_height-mask.shape[0]-2
        top = int(rng.integers(1, spare))
        region = frame[top+1:top+1+mask.shape[0], left+1:left+1+mask.shape[1]]
        region[mask] = ink
        frames.append(frame)
        entries.append({"mask": mask.copy(), "count": count, "color": color})
    minimum_u = max(sum(frames[i].shape[1] for i in group)+len(group)+1 for group in groups)
    minimum_v = len(groups)*frame_height+len(groups)+1
    extent_u = int(rng.integers(minimum_u, 30))
    extent_v = int(rng.integers(minimum_v, 30))
    base = np.full((extent_v+1, extent_u+1), bg, dtype=np.int8)
    rotation, up = configuration
    key_row = extent_v if up else 0
    base[:, 0] = slow
    base[key_row, 1:] = np.where(np.arange(1, extent_u+1) % 2, fast, bg)
    base[key_row, 0] = origin
    vgaps = _gaps(rng, frame_height*len(groups), len(groups), extent_v)
    v, ordered = vgaps[0], []
    for lane_index, group in enumerate(groups):
        ugaps = _gaps(rng, sum(frames[i].shape[1] for i in group), len(group), extent_u)
        u = 1 if witness and lane_index == 1 else ugaps[0]
        for j, i in enumerate(group):
            frame = frames[i]
            r = extent_v-v-frame_height if up else v+1
            c = u+1
            base[r:r+frame_height, c:c+frame.shape[1]] = frame
            ordered.append(entries[i])
            u += frame.shape[1]+ugaps[j+1]
        v += frame_height+vgaps[lane_index+1]
    return {"input": np.rot90(base, rotation).copy(), "output": _render(ordered, bg)}


def _witness_pair(rng, configuration, two_dimensional=False):
    area = int(rng.choice([6, 8, 10]))
    parity = int(rng.integers(2))
    counts = [1, 2, 3]
    masks = [_witness_glyph(rng, n, area, parity) for n in counts]
    h = 2
    if two_dimensional:
        h = 3
        # A different layout of the same lifetimes. Two or three components
        # overlap in their column projections, separated by a blank middle row.
        widths = [int(rng.choice([w for w in range(3, limit+1) if w % 2 == parity]))
                  for limit in (7, 9, 11)]
        masks = [_connected(rng, 3, widths[0]), np.zeros((3, widths[1]), dtype=bool),
                 np.zeros((3, widths[2]), dtype=bool)]
        if not _asymmetric(masks[0]):
            masks[0][:] = False
            masks[0][0, :] = masks[0][:, 0] = True
            masks[0][-1, 1] = True
        masks[1][[0, 2], :] = True
        masks[2][2, :] = True
        gap = int(rng.integers(1, widths[2]-1))
        masks[2][0, :gap] = masks[2][0, gap+1:] = True
    if len({m.shape[1] for m in masks}) == 1:
        # Preserve lifetimes while exposing symmetric padding.
        m = masks[2]
        if two_dimensional:
            masks[2] = np.pad(m, ((0, 0), (0, 2)))
            masks[2][[0, 2], -2:] = True
        else:
            blank = int(np.flatnonzero(~m.any(axis=0))[0])
            masks[2] = np.concatenate((m[:, :blank], np.zeros((h, 2), dtype=bool), m[:, blank:]), axis=1)
    # Count one can share a lane with either longer motif; no fixed first count.
    partner = int(rng.choice([1, 2]))
    first_lane = [0, partner]
    rng.shuffle(first_lane)
    groups = [first_lane, [3-partner]]
    return _scene(rng, masks, counts, groups, [(1, 1)]*3,
                  int(rng.integers(h+4, h+7)), configuration, True)


def _ordinary_pair(rng, configuration, split, shift=None):
    h = 4 if shift == "size" else int(rng.integers(1, 4 if split == "train" else 6))
    copy_budget = 31//(h+1)
    number = int(rng.integers(1, min(5, copy_budget)+1))
    if shift == "count":
        h = int(rng.integers(1, 4))
        copy_budget = 31//(h+1)
        number = int(rng.integers(1, min(5, copy_budget-3)+1))
    counts = [1]*number
    if shift == "count":
        counts[int(rng.integers(number))] = 4
    maximum_count = 3 if split == "train" else 4
    extra = int(rng.integers(copy_budget-sum(counts)+1))
    for _ in range(extra):
        eligible = [i for i, n in enumerate(counts) if n < maximum_count]
        if not eligible:
            break
        counts[int(rng.choice(eligible))] += 1
    parity = int(rng.integers(2))
    widths = [2*n-(1 if parity else 0) for n in counts]
    frame_height = h+4+int(rng.integers(3))
    max_lanes = min(3, 28//(frame_height+1))
    options = _partitions([w+4 for w in widths], max_lanes)
    _require(bool(options), "minimum modules cannot fit")
    groups = options[int(rng.integers(len(options)))]
    margins = [[1, 1] for _ in counts]
    for group in groups:
        spare = 27-sum(widths[i]+4 for i in group)-len(group)+1
        spend = int(rng.integers(spare+1))
        while spend:
            choices = [(i, "width") for i in group if spend >= 2 and widths[i] <= 21]
            choices += [(i, side) for i in group for side in (0, 1) if margins[i][side] < 3]
            if not choices:
                break
            i, kind = choices[int(rng.integers(len(choices)))]
            if kind == "width":
                widths[i] += 2
                spend -= 2
            else:
                margins[i][kind] += 1
                spend -= 1
    masks = [_glyph(rng, n, h, w) for n, w in zip(counts, widths)]
    return _scene(rng, masks, counts, groups, margins, frame_height, configuration)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    rotation, up = int(rng.integers(4)), bool(rng.integers(2))
    configurations = [(r, u) for r in range(4) for u in (False, True)]
    witness_configs = [(rotation, up), ((rotation+1) % 4, not up)]
    shift = str(rng.choice(["configuration", "count", "size"]))
    reserved = [c for c in configurations if c not in witness_configs]
    held_config = reserved[int(rng.integers(len(reserved)))]
    train_pool = [c for c in configurations if shift != "configuration" or c != held_config]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(number):
            witness = split == "train" and index < 2
            if witness:
                config = witness_configs[index]
            elif split == "test" and index == 0 and shift == "configuration":
                config = held_config
            else:
                pool = train_pool if split == "train" else configurations
                config = pool[int(rng.integers(len(pool)))]
            for _ in range(32):
                pair = (_witness_pair(rng, config, index == 1) if witness else
                        _ordinary_pair(rng, config, split,
                                       shift if split == "test" and index == 0 else None))
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("unique input retries exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(descriptors):
    for mode in ("once", "area", "columns", "rows", "contiguous"):
        _require(any(not np.array_equal(_render(d["entries"], d["background"]),
                                       _render(d["entries"], d["background"], mode))
                     for d in descriptors), f"training does not distinguish {mode}")
    _require(any(any(a["mask"].sum() == b["mask"].sum() and a["count"] != b["count"]
                     for a in d["entries"] for b in d["entries"]) for d in descriptors),
             "need equal-area different-lifetime evidence")
    _require(any(len({e["mask"].shape[1] for e in d["entries"]}) > 1 for d in descriptors),
             "need centering evidence")
    _require({d["up"] for d in descriptors} == {False, True}, "need both slow handednesses")
    _require(len({d["rotation"] % 2 for d in descriptors}) == 2, "need both fast axes")
    for kind in ("reverse_fast", "reverse_slow", "fast_major", "reflect_rows", "reflect_cols"):
        differs = False
        for d in descriptors:
            es = d["entries"]
            if kind == "reverse_fast":
                order = sorted(es, key=lambda e:
                    ((-e["box"][0] if d["up"] else e["box"][0]), -e["box"][2]))
            elif kind == "reverse_slow":
                order = sorted(es, key=lambda e: ((e["box"][0] if d["up"] else
                                                  -e["box"][0]), e["box"][2]))
            elif kind == "fast_major":
                order = sorted(es, key=lambda e: (e["box"][2], -e["box"][0]
                                                  if d["up"] else e["box"][0]))
            else:
                order = [dict(e, mask=e["mask"][::-1] if kind == "reflect_rows" else
                              e["mask"][:, ::-1]) for e in es]
            differs |= not np.array_equal(_render(es, d["background"]), _render(order, d["background"]))
        _require(differs, f"training leaves {kind} unresolved")


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad example counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate inputs")
            seen.add(_key(a))
            d = _decode(a)
            _require(np.array_equal(out, _render(d["entries"], d["background"])), "incorrect output")
            _require(not np.array_equal(a, out), "identity pair")
            (training if split == "train" else testing).append(d)
    _training_evidence(training)
    train_config = {(d["rotation"], d["up"]) for d in training}
    max_count = max(e["count"] for d in training for e in d["entries"])
    max_height = max(e["mask"].shape[0] for d in training for e in d["entries"])
    _require(any((d["rotation"], d["up"]) not in train_config
                 or max(e["count"] for e in d["entries"]) > max_count
                 or max(e["mask"].shape[0] for e in d["entries"]) > max_height
                 for d in testing), "missing compositional or structural query shift")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> dict:
    """Bounded tests, including actual input-derived diversity over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity collapsed")
    structures, input_shapes, output_shapes, configs, counts, widths, heights = (set() for _ in range(7))
    pairs, multi_lane, repeated_colors, holes = 0, 0, 0, 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                pairs += 1
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                configs.add((d["rotation"], d["up"]))
                multi_lane += len(d["lanes"]) > 1
                repeated_colors += len({e["color"] for e in d["entries"]}) < len(d["entries"])
                for e in d["entries"]:
                    counts.add(e["count"])
                    heights.add(e["mask"].shape[0])
                    widths.add(e["mask"].shape[1])
                    structures.add(_key(e["mask"]))
                    holes += len(_components(~np.pad(e["mask"], 1))) > 1
    _require(len(configs) == 8 and counts == {1, 2, 3, 4}, "directions/counts missing")
    _require(len(structures) >= 100 and len(input_shapes) >= 50 and len(output_shapes) >= 25,
             "insufficient geometric variation")
    _require(min(widths) <= 2 and max(widths) >= 15 and len(heights) == 5,
             "size range collapsed")
    _require(multi_lane >= 84 and repeated_colors > 0 and holes > 0, "structural modes missing")
    for seed in (0, 11, 41):
        _require(_puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
    # All 30 legal explicit count combinations, plus invalid types/ranges.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(3100+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.2), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Proper rotation of the whole scene leaves the normalized answer unchanged.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(pair["input"], k)), pair["output"]),
                     "whole-scene rotation failure")
    # Supported family members beyond the sampler's usual common crop height:
    # a short three-piece glyph, vertically separated pieces, and a nested ring.
    ring = np.ones((5, 7), dtype=bool)
    ring[1:-1, 1:-1] = False
    masks = [np.array([[1, 0, 1, 0, 1]], dtype=bool),
             np.array([[1]*5, [0]*5, [1]*5], dtype=bool), ring]
    for rotation in range(4):
        for up in (False, True):
            p = _scene(np.random.default_rng(872+rotation+4*up), masks, [3, 2, 1],
                       [[0, 1], [2]], [(1, 1)]*3, 9, (rotation, up))
            _require(np.array_equal(solve(p["input"]), p["output"]),
                     "mixed heights or nested glyph ring failure")
    # Smallest crop and largest supported width, including a lifetime of one.
    for mask in (np.ones((1, 1), dtype=bool), np.ones((5, 23), dtype=bool)):
        p = _scene(np.random.default_rng(909), [mask], [1], [[0]], [(1, 1)],
                   mask.shape[0]+4, (0, False))
        _require(np.array_equal(solve(p["input"]), p["output"]), "crop boundary failure")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0] = bad["train"][0]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate input accepted")
    # A diagonal pair has two edge-components but only one eight-component.
    p = _scene(np.random.default_rng(916), [np.eye(2, dtype=bool)], [2], [[0]],
               [(1, 1)], 6, (1, True))
    try:
        solve(p["input"])
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous component connectivity accepted")
    damaged = puzzles[0]["train"][0]["input"].copy()
    d = _decode(damaged)
    normalized = d["normalized"].copy()
    r0, _, c0, _ = d["entries"][0]["box"]
    normalized[r0, c0] = d["background"]
    try:
        solve(normalized)
    except ValueError:
        pass
    else:
        raise AssertionError("broken frame accepted")
    return {"seeds": 42, "pairs": pairs, "glyph_structures": len(structures),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "configurations": len(configs), "component_counts": sorted(counts),
            "glyph_widths": sorted(widths), "glyph_heights": sorted(heights),
            "multi_lane_pairs": multi_lane, "repeated_frame_color_pairs": repeated_colors,
            "glyphs_with_holes": holes}
