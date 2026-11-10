"""Room ballots on four interleaved clockwise pinwheel paths.

Only NumPy and the standard library are used. No artifacts are written by this
module. Generation retains at most one puzzle and one candidate scene at a time.
"""
from collections import Counter
import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _cross(shape, center, length):
    m = np.zeros(shape, dtype=bool)
    y, x = center
    for d in range(1, length + 1):
        m[y-d, x] = m[y+d, x] = m[y, x-d] = m[y, x+d] = True
    return m


def _seed_phases(shape, center, length):
    result = np.zeros((4, *shape), dtype=bool)
    y, x = center
    for arm, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
        for d in range(1, length+1):
            result[arm, y+d*dy, x+d*dx] = True
    return result


def _detect_cross(a, color):
    m = a == color
    sites = np.argwhere(m)
    if not len(sites) or len(sites) % 4:
        return None
    length = len(sites) // 4
    if not 1 <= length <= 6:
        return None
    lo, hi = sites.min(axis=0), sites.max(axis=0)
    if not np.all(hi - lo == 2 * length):
        return None
    center = tuple(map(int, (hi + lo) // 2))
    if not np.array_equal(m, _cross(a.shape, center, length)):
        return None
    return length, center


def _phases(length, shape, center):
    """Four disjoint B-type paths; draw clipped segments, never stop at exit.

    North starts eastward at radius R=max(2,L). At stage k: forward 4k+2,
    right R-2, forward R, then turn the stage direction right. The remaining
    paths are quarter rotations. Arms are ordered N,E,S,W only for bookkeeping.
    """
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    masks = np.zeros((4, h, w), dtype=bool)
    distance = max(cy, cx, h-1-cy, w-1-cx)
    stages = 4 * (distance // 8 + 2)
    for arm, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
        def line(y0, x0, y1, x1):
            y0, y1, x0, x1 = y0+cy, y1+cy, x0+cx, x1+cx
            if y0 == y1 and 0 <= y0 < h:
                low, high = max(0, min(x0, x1)), min(w-1, max(x0, x1))
                if low <= high:
                    masks[arm, y0, low:high+1] = True
            elif x0 == x1 and 0 <= x0 < w:
                low, high = max(0, min(y0, y1)), min(h-1, max(y0, y1))
                if low <= high:
                    masks[arm, low:high+1, x0] = True
        y, x = radius*dy, radius*dx
        line(dy, dx, y, x)
        dy, dx = dx, -dy
        for k in range(stages):
            for vy, vx, steps in ((dy, dx, 4*k+2), (dx, -dy, radius-2),
                                  (dy, dx, radius)):
                ny, nx = y+steps*vy, x+steps*vx
                line(y, x, ny, nx)
                y, x = ny, nx
            dy, dx = dx, -dy
    return masks


def _interval_phases(length, shape, center):
    """Closed-form cell membership cross-check, with no evolving path state."""
    radius = max(2, length)
    y, x = np.indices(shape)
    y, x = y-center[0], x-center[1]
    bound = max(int(np.abs(y).max()), int(np.abs(x).max()))
    result = []
    for _ in range(4):
        mask = (x == 0) & (y >= -radius) & (y <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, ((yy, xx), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (y-yy)*dy + (x-xx)*dx
                across = (y-yy)*dx - (x-xx)*dy
                run = 16*q + 4*s + 2
                mask |= (across == 0) & (along >= 0) & (along <= run)
                mask |= (along == run) & (across >= 0) & (across <= radius-2)
                mask |= (across == radius-2) & (along >= run) & (along <= run+radius)
        result.append(mask)
        y, x = -x, y
    return np.array(result)


def _walls(shape, rooms):
    """Room tuples describe their wall coordinates, inclusive."""
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in rooms:
        mask[t, l:r+1] = mask[b, l:r+1] = True
        mask[t:b+1, l] = mask[t:b+1, r] = True
    return mask


def _enclosed_rooms(mask):
    """Return rectangular positive-area interiors, excluding the outside."""
    free = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros_like(free)
    h, w = free.shape
    rooms = []
    for yy, xx in np.argwhere(free):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        stack, cells, outside = [(yy, xx)], [], False
        seen[yy, xx] = True
        while stack:
            y, x = stack.pop()
            cells.append((y-1, x-1))
            outside |= y in (0, h-1) or x in (0, w-1)
            for r, c in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= r < h and 0 <= c < w and free[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    stack.append((r, c))
        if not outside:
            a = np.array(cells)
            lo, hi = a.min(axis=0), a.max(axis=0)
            if len(cells) != int(np.prod(hi-lo+1)):
                raise ValueError("room interiors must be rectangles")
            rooms.append(tuple(map(int, (lo[0]-1, lo[1]-1, hi[0]+1, hi[1]+1))))
    return rooms


def _recover(a, color, background):
    """Repair supported isolated overwritten pixels, not open background."""
    visible = a == color
    repaired = visible.copy()
    gap = (a != background) & ~visible
    horizontal = np.zeros_like(visible)
    vertical = np.zeros_like(visible)
    horizontal[:, 1:-1] = visible[:, :-2] & visible[:, 2:]
    vertical[1:-1, :] = visible[:-2, :] & visible[2:, :]
    repaired |= gap & (horizontal | vertical)
    sites = np.argwhere(visible)
    if len(sites) < 8:
        return None
    t, l = map(int, sites.min(axis=0))
    b, r = map(int, sites.max(axis=0))
    if min(b-t, r-l) < 2:
        return None
    if not (repaired[t, l:r+1].all() and repaired[b, l:r+1].all()
            and repaired[t:b+1, l].all() and repaired[t:b+1, r].all()):
        return None
    try:
        local_rooms = _enclosed_rooms(repaired[t:b+1, l:r+1])
    except ValueError:
        return None
    rooms = [(a+t, c+l, b+t, d+l) for a, c, b, d in local_rooms]
    if not rooms or not np.array_equal(repaired, _walls(a.shape, rooms)):
        return None
    return {"color": int(color), "mask": repaired, "rooms": rooms, "box": (t, l, b, r)}


def _decode(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    if len(values) not in (5, 6) or counts.max() <= a.size // 2:
        raise ValueError("a scene needs a majority background and 2..3 networks")
    background = int(values[int(np.argmax(counts))])
    crosses = [(int(c), _detect_cross(a, int(c))) for c in values if c != background]
    crosses = [(c, info) for c, info in crosses if info is not None]
    if len(crosses) != 1:
        raise ValueError("a scene needs one uniquely colored hollow cross")
    neutral, (length, center) = crosses[0]
    if a[center] != background:
        raise ValueError("the cross center must be background")
    networks, leftovers = [], []
    for color in values:
        if color in (background, neutral):
            continue
        recovered = _recover(a, int(color), background)
        if recovered is None:
            leftovers.append(int(color))
        else:
            networks.append(recovered)
    if len(leftovers) != 1 or not 2 <= len(networks) <= 3:
        raise ValueError("exactly one noise color and 2..3 closed networks required")
    noise = leftovers[0]
    # Explicit family conditions: disjoint boxes; every repaired cell was noise.
    for i, n in enumerate(networks):
        if np.any(n["mask"] & ~np.isin(a, (n["color"], noise))):
            raise ValueError("a wall can only be overwritten by noise")
        for other in networks[:i]:
            t, l, b, r = n["box"]
            u, v, d, e = other["box"]
            if not (b < u or d < t or r < v or e < l):
                raise ValueError("network boxes must be disjoint")
    return a, background, neutral, noise, length, center, networks


def _votes(networks, phases, pixel_votes=False):
    scores = np.zeros((len(networks), 4), dtype=int)
    for i, n in enumerate(networks):
        for t, l, b, r in n["rooms"]:
            portion = phases[:, t+1:b, l+1:r]
            scores[i] += portion.sum(axis=(1, 2)) if pixel_votes else portion.any(axis=(1, 2))
    return scores


def _winners(scores, networks, neutral):
    winners = []
    for arm in range(4):
        col = scores[:, arm]
        best = np.flatnonzero(col == col.max())
        winners.append(networks[int(best[0])]["color"] if len(best) == 1 and col.max() > 0 else neutral)
    return winners


def _paint(shape, background, phases, winners):
    out = np.full(shape, background, dtype=np.int8)
    for mask, color in zip(phases, winners):
        out[mask] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, neutral, _, length, center, networks = _decode(input_grid)
    phases = _phases(length, a.shape, center)
    return _paint(a.shape, bg, phases, _winners(_votes(networks, phases), networks, neutral))


def _capacity(box):
    t, l, b, r = box
    return ((b-t)//2) * ((r-l)//2)


def _partition(rng, box, count, style):
    if style == "strip":
        t, l, b, r = box
        axes = [a for a, span in ((0, b-t), (1, r-l)) if span >= 2*count]
        if axes:
            axis = int(rng.choice(axes))
            start, end = (t, b) if axis == 0 else (l, r)
            gaps = np.full(count, 2)
            for _ in range(end-start-2*count):
                gaps[int(rng.integers(count))] += 1
            coords = [start] + list(map(int, start+np.cumsum(gaps)))
            return [(coords[i], l, coords[i+1], r) if axis == 0 else
                    (t, coords[i], b, coords[i+1]) for i in range(count)]
    rooms = [box]
    first_axis = None
    for step in range(count-1):
        choices = []
        for i, (t, l, b, r) in enumerate(rooms):
            for axis, start, end in ((0, t, b), (1, l, r)):
                if style == "partial" and step == 1 and axis == first_axis:
                    continue
                for cut in range(start+2, end-1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else (
                        (t, l, b, cut), (t, cut, b, r))
                    capacity = sum(_capacity(q) for j, q in enumerate(rooms) if j != i)
                    if capacity + sum(_capacity(q) for q in halves) >= count:
                        choices.append((i, axis, halves))
        if not choices:
            return None
        i, axis, halves = choices[int(rng.integers(len(choices)))]
        first_axis = axis if step == 0 else first_axis
        rooms[i:i+1] = halves
    return rooms


def _partial(n):
    mask = n["mask"]
    t, l, b, r = n["box"]
    ys = {y for a, _, c, _ in n["rooms"] for y in (a, c)}
    xs = {x for _, a, _, c in n["rooms"] for x in (a, c)}
    return any(not mask[y, l:r+1].all() for y in ys) or any(
        not mask[t:b+1, x].all() for x in xs)


def _evidence(networks, phases, neutral):
    votes = _votes(networks, phases)
    winners = _winners(votes, networks, neutral)
    pixels = _winners(_votes(networks, phases, True), networks, neutral)
    totals = np.repeat(np.array([len(n["rooms"]) for n in networks])[:, None], 4, axis=1)
    total_winners = _winners(totals, networks, neutral)
    flat = [{**n, "rooms": [n["box"]]} for n in networks]
    flat_winners = _winners(_votes(flat, phases), networks, neutral)
    positive_tie = any(votes[:, arm].max() > 0 and
                       np.count_nonzero(votes[:, arm] == votes[:, arm].max()) > 1
                       for arm in range(4))
    top_tie = positive_tie and any(votes[:, arm].max() > votes[:, arm].min() and
                                  np.count_nonzero(votes[:, arm] == votes[:, arm].max()) > 1
                                  for arm in range(4))
    return {"votes": votes, "winners": winners,
            "pixel_shortcut": pixels != winners, "total_shortcut": total_winners != winners,
            "flat_shortcut": flat_winners != winners, "positive_tie": positive_tie,
            "top_tie": top_tie,
            "colored_arms": sum(c != neutral for c in winners),
            "winning_networks": len(set(c for c in winners if c != neutral)),
            "tie": neutral in winners, "partial": any(_partial(n) for n in networks)}


def _candidate(rng, length, room_cap, palettes, teaching, requirement):
    bg, neutral, noise, *colors = palettes
    number = len(colors)
    h, w = map(int, rng.integers(19 if teaching else 17, 31, size=2))
    shape = h, w
    networks = []
    for color in colors:
        accepted = None
        for _ in range(55):
            bh, bw = map(int, rng.integers(5, min(16, min(shape))+1, size=2))
            t, l = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
            box = (t, l, t+bh-1, l+bw-1)
            if any(not (box[2]+1 < n["box"][0] or n["box"][2]+1 < t or
                        box[3]+1 < n["box"][1] or n["box"][3]+1 < l) for n in networks):
                continue
            count = int(rng.integers(1, min(room_cap, _capacity(box))+1))
            style = str(rng.choice(("partial", "partial", "random", "strip")))
            rooms = _partition(rng, box, count, style)
            if rooms is None:
                continue
            accepted = {"color": color, "rooms": rooms, "mask": _walls(shape, rooms), "box": box}
            break
        if accepted is None:
            return None
        networks.append(accepted)
    union = np.logical_or.reduce([n["mask"] for n in networks])
    margin = length + (2 if teaching else 0)
    centers = []
    for y in range(margin, h-margin):
        for x in range(margin, w-margin):
            if not np.any(union & _cross(shape, (y, x), length)) and not union[y, x]:
                centers.append((y, x))
    if not centers:
        return None
    # Tests favor boundary-near centers; some remain central. No hidden direction.
    if not teaching:
        tight = [p for p in centers if min(p[0]-length, p[1]-length,
                 h-1-p[0]-length, w-1-p[1]-length) <= 1]
        if not tight:
            return None
        centers = tight
    center = centers[int(rng.integers(len(centers)))]
    phases = _phases(length, shape, center)
    evidence = _evidence(networks, phases, neutral)
    if evidence["colored_arms"] < 2:
        return None
    if requirement and not (evidence[requirement] and evidence["partial"]):
        return None
    if requirement == "pixel_shortcut" and not evidence["positive_tie"]:
        return None
    if requirement == "total_shortcut" and teaching:
        if len(set(c for c in evidence["winners"] if c != neutral)) < 2:
            return None
    a = np.full(shape, bg, dtype=np.int8)
    for n in networks:
        a[n["mask"]] = n["color"]
    seed_mask = _cross(shape, center, length)
    a[seed_mask] = neutral
    # Only isolated supported straight-wall damage; protect corners/junctions.
    damaged = np.zeros(shape, dtype=bool)
    for n in networks:
        m = n["mask"]
        horiz, vert = np.zeros(shape, bool), np.zeros(shape, bool)
        horiz[1:-1, 1:-1] = (m[1:-1, :-2] & m[1:-1, 2:] &
                                      ~m[:-2, 1:-1] & ~m[2:, 1:-1])
        vert[1:-1, 1:-1] = (m[:-2, 1:-1] & m[2:, 1:-1] &
                                     ~m[1:-1, :-2] & ~m[1:-1, 2:])
        safe = np.argwhere(m & (horiz | vert))
        rng.shuffle(safe)
        amount = int(rng.integers(0, min(5, len(safe))+1))
        if teaching and len(safe):
            amount = max(1, amount)
        for y, x in safe:
            if not amount:
                break
            if damaged[max(0,y-1):y+2, max(0,x-1):x+2].any():
                continue
            a[y, x] = noise
            damaged[y, x] = True
            amount -= 1
    near = union | seed_mask
    excluded = near.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0,dy), min(h,h+dy))
            xs = slice(max(0,dx), min(w,w+dx))
            excluded[ys, xs] |= near[max(0,-dy):min(h,h-dy), max(0,-dx):min(w,w-dx)]
    excluded[center] = True
    sites = np.argwhere(~excluded)
    if len(sites) == 0:
        return None
    for index in rng.choice(len(sites), size=min(len(sites), int(rng.integers(2, 8))), replace=False):
        y, x = sites[int(index)]
        a[y, x] = noise
    if np.count_nonzero(a == bg) <= a.size // 2:
        return None
    try:
        _, _, _, _, _, _, recovered = _decode(a)
    except ValueError:
        return None
    intended = {n["color"]: n for n in networks}
    if any(not np.array_equal(n["mask"], intended[n["color"]]["mask"]) for n in recovered):
        return None
    out = _paint(shape, bg, phases, evidence["winners"])
    return {"input": a, "output": out}


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    nt = _checked_count(num_train, "num_train", 2, 10)
    ne = _checked_count(num_test, "num_test", 1, 4)
    if ne is None:
        ne = int(rng.integers(1, min(4, (nt or 10)-1)+1))
    if nt is None:
        nt = int(rng.integers(max(2, ne+1), 11))
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    fixed_palette = bool(rng.random() < .25)
    palette = list(map(int, rng.permutation(10)))
    lengths = list(map(int, rng.choice((2, 3, 4), size=2, replace=False)))
    train_lengths = lengths + [int(rng.choice(lengths)) for _ in range(nt-2)]
    room_cap = int(rng.integers(4, 7))
    shift = str(rng.choice(("topology", "count", "placement")))
    puzzle, used = {"train": [], "test": []}, set()
    for index in range(nt+ne):
        teaching = index < nt
        length = train_lengths[index] if teaching else int(rng.choice(lengths))
        # Two demonstrations together defeat room-total and pixel-total shortcuts.
        requirement = ("pixel_shortcut" if index == 0 else
                       "total_shortcut" if index == 1 else None)
        if not teaching:
            requirement = "total_shortcut" if shift == "placement" else None
        cap = room_cap+1 if not teaching and shift == "count" else room_cap
        number = 2 if (teaching and index < 2) else int(rng.choice((2, 2, 3)))
        if number == 3:
            if teaching:
                requirement = "top_tie"
            elif not any(len(_decode(pair["input"])[-1]) == 3 for pair in puzzle["train"]):
                number = 2
        roles = palette[:3+number] if fixed_palette else list(map(int, rng.choice(10, size=3+number, replace=False)))
        for _ in range(1500):
            pair = _candidate(rng, length, cap, roles, teaching, requirement)
            if pair is not None and _key(pair["input"]) not in used:
                used.add(_key(pair["input"]))
                puzzle["train" if teaching else "test"].append(pair)
                break
        else:
            raise RuntimeError("bounded scene search exhausted")
    rng.shuffle(puzzle["train"])
    return puzzle


def _cycle_count(mask):
    # Connected planar pixel graph, subtracting filled 2x2 pixel faces.
    edges = int((mask[:-1] & mask[1:]).sum() + (mask[:, :-1] & mask[:, 1:]).sum())
    tiny = int((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).sum())
    return edges-int(mask.sum())+1-tiny


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence, demonstrated, demonstrated_network_counts = set(), [], set(), set()
    seed_shortcut = damage_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            a, bg, neutral, noise, length, center, networks = _decode(a)
            phases = _interval_phases(length, a.shape, center)
            assert np.all(phases.sum(axis=0) <= 1)
            assert np.array_equal(phases, _phases(length, a.shape, center))
            scores = _votes(networks, phases)
            winners = _winners(scores, networks, neutral)
            assert np.array_equal(out, _paint(a.shape, bg, phases, winners))
            assert out[center] == bg and not np.any(out == noise)
            assert np.array_equal(out != bg, phases.any(axis=0))
            for n in networks:
                assert _cycle_count(n["mask"]) == len(n["rooms"])
            facts = _evidence(networks, phases, neutral)
            assert facts["colored_arms"] >= 2
            margin = min(center[0]-length, center[1]-length,
                         a.shape[0]-1-center[0]-length, a.shape[1]-1-center[1]-length)
            if split == "train":
                assert margin >= 2
                demonstrated.add(length)
                demonstrated_network_counts.add(len(networks))
                evidence.append(facts)
                if len(networks) == 3:
                    assert facts["top_tie"]
                seed_scores = _votes(networks, _seed_phases(a.shape, center, length))
                seed_shortcut |= _winners(seed_scores, networks, neutral) != winners
                damage_witness |= any(np.any(n["mask"] & (a == noise)) for n in networks)
            else:
                assert margin <= 1
                assert length in demonstrated
                assert len(networks) in demonstrated_network_counts
    assert len(demonstrated) >= 2
    assert any(e["pixel_shortcut"] and e["partial"] for e in evidence)
    assert any(e["total_shortcut"] and e["partial"] for e in evidence)
    assert any(e["positive_tie"] for e in evidence)
    assert any(e["flat_shortcut"] for e in evidence)
    assert any(e["winning_networks"] >= 2 for e in evidence)
    assert seed_shortcut and damage_witness


def measure_puzzle(puzzle):
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a, bg, neutral, noise, length, center, networks = _decode(pair["input"])
            phases = _phases(length, a.shape, center)
            e = _evidence(networks, phases, neutral)
            records.append({"split": split, "shape": list(a.shape), "length": length,
                            "center": list(center), "palette": [bg, neutral, noise]+[n["color"] for n in networks],
                            "rooms": [len(n["rooms"]) for n in networks],
                            "partial": e["partial"], "pixel_shortcut": e["pixel_shortcut"],
                            "total_shortcut": e["total_shortcut"], "tie": e["tie"],
                            "flat_shortcut": e["flat_shortcut"], "positive_tie": e["positive_tie"],
                            "top_tie": e["top_tie"],
                            "winners": e["winners"], "votes": e["votes"].tolist(),
                            "damaged": int(sum(np.count_nonzero(n["mask"] & (a == noise)) for n in networks))})
    return records


def self_test():
    """Bounded, streaming 42-seed checks; no puzzle bank or diagnostic files."""
    counts, shapes, palettes, layouts, signatures = Counter(), set(), set(), set(), set()
    features = Counter()
    first = None
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                for role in ("input", "output"):
                    digest.update(bytes(pair[role].shape)+pair[role].tobytes())
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
        signatures.add(digest.hexdigest())
        for r in measure_puzzle(p):
            shapes.add(tuple(r["shape"]))
            palettes.add(tuple(r["palette"]))
            layouts.add((tuple(r["rooms"]), r["length"], tuple(r["center"])))
            counts.update(r["rooms"])
            for name in ("partial", "pixel_shortcut", "total_shortcut", "tie"):
                features[name] += r[name]
            features["three_networks"] += len(r["rooms"]) == 3
        if seed == 0:
            first = p
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(palettes) >= 25 and len(layouts) >= 100
    assert set(range(1, 6)) <= set(counts)
    assert all(features[k] > 0 for k in ("partial", "pixel_shortcut", "total_shortcut", "tie", "three_networks"))
    repeated = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], repeated[s]) for k in a)
    # Changing literal roles or rotating a scene must transport its answer.
    permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
    for pair in first["train"] + first["test"]:
        original = pair["input"].copy()
        assert np.array_equal(solve(np.rot90(original)), np.rot90(pair["output"]))
        assert np.array_equal(solve(permutation[original]), permutation[pair["output"]])
        assert np.array_equal(original, pair["input"]), "solve mutated its input"
    # Geometric checks of every supported cross scale, boundary exits/re-entry,
    # and the radius-two short-spoke convention; no stored execution traces.
    for length in range(1, 7):
        shape, center = (29, 30), (length, 29-length)
        phases = _phases(length, shape, center)
        assert np.array_equal(phases, _interval_phases(length, shape, center))
        big = _phases(length, (70, 70), (35, 35))
        y, x = 35-center[0], 35-center[1]
        assert np.array_equal(phases, big[:, y:y+shape[0], x:x+shape[1]])
    assert np.array_equal(_phases(1, (30, 30), (15, 15)), _phases(2, (30, 30), (15, 15)))
    # Input-only solving at the less frequently sampled scales, not just masks.
    rng = np.random.default_rng(391031)
    for length in (1, 5, 6):
        for _ in range(300):
            pair = _candidate(rng, length, 6, [0, 8, 5, 1, 2], False, None)
            if pair is not None:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                break
        else:
            raise AssertionError("could not exercise a supported spoke length")
    # Every legal explicit pair-count request; also partial-count convenience.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(2000+10*nt+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
            validate_generated_puzzle(p)
    for nt, ne in ((None, None), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(8000, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, np.bool_(True)), ("4", 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # Mutation protection: one wrong output pixel cannot pass the validator.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed output accepted")
