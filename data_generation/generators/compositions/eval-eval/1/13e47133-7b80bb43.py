"""Folded color seals: a repaired wall is also a contour-color program.

Only standard library and NumPy. No parent imports, files, or generation state
are used by solve(). Experiments are bounded and self_test retains no grids.
"""
from __future__ import annotations

from collections import deque
import hashlib
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _at(a, r, c):
    return 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and bool(a[r, c])


def _roles(a):
    counts = np.bincount(a.ravel(), minlength=10)
    order = sorted(range(10), key=lambda k: (-int(counts[k]), k))
    bg, wall = order[:2]
    if not counts[bg] > counts[wall] > counts[order[2]] > 0:
        raise ValueError("Background and wall must be strictly first and second in frequency")
    return bg, wall


def _regions(wall):
    """Label four-connected non-wall regions, including the exterior."""
    h, w = wall.shape
    labels = np.zeros((h, w), dtype=np.int16)
    regions = {}
    for sr, sc in np.argwhere(~wall):
        sr, sc = int(sr), int(sc)
        if labels[sr, sc]:
            continue
        k = len(regions) + 1
        labels[sr, sc] = k
        queue = deque([(sr, sc)])
        cells, closed = [], True
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            if r in (0, h - 1) or c in (0, w - 1):
                closed = False
            for dr, dc in _D4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and not labels[rr, cc]:
                    labels[rr, cc] = k
                    queue.append((rr, cc))
        regions[k] = {"mask": labels == k, "closed": closed, "area": len(cells)}
    return labels, regions


def _erosion_depth(mask, diagonal=True):
    """Contour depth; off-canvas positions count as outside."""
    current = mask.copy()
    depth = np.full(mask.shape, -1, np.int16)
    h, w = mask.shape
    d = 0
    while current.any():
        padded = np.pad(current, 1)
        inside = current.copy()
        for dr, dc in (_D8 if diagonal else _D4):
            inside &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~inside] = d
        current = inside
        d += 1
    return depth


def _construction_depth(mask):
    """BFS construction, checked against the solver's iterative erosion."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, np.int16)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not _at(mask, r + dr, c + dc) for dr, dc in _D8):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and depth[rr, cc] < 0:
                depth[rr, cc] = depth[r, c] + 1
                queue.append((rr, cc))
    return depth


def _gaps(wall, intact=False):
    """Bracketed spans, as (row, column, row_direction, column_direction)."""
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = wall if dc else wall.T
        found = m[:, :-4] & m[:, 4:]
        middle = (m[:, 1:-3], m[:, 2:-2], m[:, 3:-1])
        found &= (middle[0] & middle[1] & middle[2]) if intact else ~(middle[0] | middle[1] | middle[2])
        for y, x in np.argwhere(found):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            if _at(wall, r - dr, c - dc) or _at(wall, r + 5 * dr, c + 5 * dc):
                result.append((r, c, dr, dc))
    return result


def _seal(r, c, dr, dc, endpoint, side):
    ar, ac = r + endpoint * dr, c + endpoint * dc
    toward = 1 if endpoint == 0 else -1
    vr, vc = toward * dr + side * dc, toward * dc + side * dr
    return {"span": (r, c, dr, dc),
            "gap": tuple((r + k * dr, c + k * dc) for k in (1, 2, 3)),
            "beads": ((ar + vr, ac + vc), (ar + 2 * vr, ac + 2 * vc)),
            "third": (ar + 3 * vr, ac + 3 * vc),
            "endpoint": endpoint, "side": side}


def _read_seals(a, bg, wall_color):
    wall = a == wall_color
    beads = (a != bg) & ~wall
    result = []
    for r, c, dr, dc in _gaps(wall):
        if any(a[r + k * dr, c + k * dc] != bg for k in (1, 2, 3)):
            continue
        matches = []
        for endpoint in (0, 4):
            for side in (-1, 1):
                seal = _seal(r, c, dr, dc, endpoint, side)
                if all(_at(beads, *p) for p in seal["beads"]) and not _at(beads, *seal["third"]):
                    seal["period"] = tuple(int(a[p]) for p in seal["beads"])
                    matches.append(seal)
        if len(matches) > 1:
            raise ValueError("A gap has conflicting diagonal attachments")
        result.extend(matches)
    used = [p for seal in result for p in seal["beads"]]
    if len(used) != len(set(used)):
        raise ValueError("A bead belongs to multiple seals")
    return result


def _incident(seal, labels):
    r, c, dr, dc = seal["span"]
    ids = set()
    for k in (1, 2, 3):
        for sign in (-1, 1):
            rr, cc = r + k * dr + sign * dc, c + k * dc + sign * dr
            if 0 <= rr < labels.shape[0] and 0 <= cc < labels.shape[1] and labels[rr, cc]:
                ids.add(int(labels[rr, cc]))
    return ids


def _paint(wall, seals, bg, wall_color, construction=False):
    labels, regions = _regions(wall)
    programs = {}
    for seal in seals:
        for k in _incident(seal, labels):
            if regions[k]["closed"]:
                period = seal["period"]
                if k in programs and programs[k] != period:
                    raise ValueError("An enclosure has incompatible color seals")
                programs[k] = period
    out = np.full(wall.shape, bg, np.int8)
    out[wall] = wall_color
    for k, period in programs.items():
        mask = regions[k]["mask"]
        depth = (_construction_depth if construction else _erosion_depth)(mask)
        out[mask] = np.asarray(period, np.int8)[depth[mask] % 2]
        regions[k]["depth"] = depth
    return out, labels, regions, programs


def _decode(a):
    bg, wall_color = _roles(a)
    seals = _read_seals(a, bg, wall_color)
    wall = a == wall_color
    for seal in seals:
        for p in seal["gap"]:
            wall[p] = True
    out, labels, regions, programs = _paint(wall, seals, bg, wall_color)
    return out, {"bg": bg, "wall_color": wall_color, "wall": wall, "seals": seals,
                 "labels": labels, "regions": regions, "programs": programs}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only reconstruction, closure detection, and contour propagation."""
    return _decode(_grid(input_grid))[0]


def _closed_ids(regions):
    return {k for k, info in regions.items() if info["closed"]}


def _polygon(rng, first=False, rectangle=False):
    """Compose a silhouette from coarse tiles; only its exposed edges are walls."""
    if rectangle:
        h, w = map(int, rng.integers(24, 31, size=2))
        t, l, b, r = map(int, rng.integers(1, 3, size=4))
        wall = np.zeros((h, w), bool)
        wall[t, l:w-r] = wall[h-b-1, l:w-r] = True
        wall[t:h-b, l] = wall[t:h-b, w-r-1] = True
        return wall
    nr, nc = ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.integers(4))]
    target = int(rng.integers(3 if first else 4, nr * nc + (0 if first else 1)))
    tiles = {(int(rng.integers(nr)), int(rng.integers(nc)))}
    while len(tiles) < target:
        options = sorted({(r + dr, c + dc) for r, c in tiles for dr, dc in _D4
                          if 0 <= r + dr < nr and 0 <= c + dc < nc} - tiles)
        tiles.add(options[int(rng.integers(len(options)))])
    rows, cols = zip(*tiles)
    r0, c0 = min(rows), min(cols)
    tiles = {(r - r0, c - c0) for r, c in tiles}
    nr, nc = max(r for r, c in tiles) + 1, max(c for r, c in tiles) + 1
    heights = list(map(int, rng.integers(6, 10, size=nr)))
    widths = list(map(int, rng.integers(6, 10, size=nc)))
    for sizes in (heights, widths):
        while sum(sizes) > 24:
            sizes[int(np.argmax(sizes))] -= 1
    t, l, b, r = map(int, rng.integers(1, 4, size=4))
    ys, xs = [t], [l]
    for n in heights:
        ys.append(ys[-1] + n)
    for n in widths:
        xs.append(xs[-1] + n)
    wall = np.zeros((ys[-1] + b + 1, xs[-1] + r + 1), bool)
    # Reduce a margin when a 24-cell silhouette would exceed the ARC limit.
    wall = wall[:30, :30]
    for row, col in sorted(tiles):
        a, z, x, y = ys[row], ys[row + 1], xs[col], xs[col + 1]
        if (row - 1, col) not in tiles:
            wall[a, x:y+1] = True
        if (row + 1, col) not in tiles:
            wall[z, x:y+1] = True
        if (row, col - 1) not in tiles:
            wall[a:z+1, x] = True
        if (row, col + 1) not in tiles:
            wall[a:z+1, y] = True
    labels, regions = _regions(wall)
    assert len(_closed_ids(regions)) == 1
    assert not _thick(wall)
    return wall


def _thick(wall):
    return bool((wall[:-1, :-1] & wall[1:, :-1] & wall[:-1, 1:] & wall[1:, 1:]).any())


def _subdivide(rng, wall, target):
    for _ in range(target - 1):
        labels, regions = _regions(wall)
        options = []
        for k in sorted(_closed_ids(regions)):
            mask = regions[k]["mask"]
            for transpose in (False, True):
                m, w = (mask.T, wall.T) if transpose else (mask, wall)
                for r in range(m.shape[0]):
                    cs = np.flatnonzero(m[r])
                    if len(cs) < 5:
                        continue
                    # Each contiguous interval can be an independent chord.
                    for run in np.split(cs, np.flatnonzero(np.diff(cs) != 1) + 1):
                        lo, hi = int(run[0]), int(run[-1])
                        if len(run) >= 5 and lo > 0 and hi + 1 < m.shape[1] and w[r, lo-1] and w[r, hi+1]:
                            options.append((transpose, r, lo, hi))
        rng.shuffle(options)
        for transpose, r, lo, hi in options[:64]:
            trial = wall.copy()
            work = trial.T if transpose else trial
            work[r, lo-1:hi+2] = True
            _, candidate_regions = _regions(trial)
            closed = _closed_ids(candidate_regions)
            if (len(closed) == len(_closed_ids(regions)) + 1 and not _thick(trial)
                    and all(candidate_regions[k]["area"] >= 25
                            and int(_erosion_depth(candidate_regions[k]["mask"]).max()) >= 2 for k in closed)):
                wall = trial
                break
        else:
            break
    return wall


def _candidates(wall):
    labels, regions = _regions(wall)
    closed = _closed_ids(regions)
    result = []
    for r, c, dr, dc in _gaps(wall, intact=True):
        if any(_at(wall, r+k*dr+s*dc, c+k*dc+s*dr) for k in (1, 2, 3) for s in (-1, 1)):
            continue  # Never remove a junction or a parallel wall.
        for endpoint in (0, 4):
            for side in (-1, 1):
                seal = _seal(r, c, dr, dc, endpoint, side)
                if not all(0 <= rr < wall.shape[0] and 0 <= cc < wall.shape[1] and not wall[rr, cc]
                           for rr, cc in seal["beads"]):
                    continue
                seal["ids"] = _incident(seal, labels) & closed
                result.append(seal)
    return result, regions


def _footprint(seal, with_beads=True):
    r, c, dr, dc = seal["span"]
    cells = {(r + k * dr, c + k * dc) for k in range(5)}
    return cells | set(seal["beads"]) if with_beads else cells


def _separated(seal, occupied, with_beads=True):
    return all(max(abs(r-rr), abs(c-cc)) > 1 for r, c in _footprint(seal, with_beads)
               for rr, cc in occupied)


def _choose(rng, candidates, predicate, occupied):
    options = [s for s in candidates if predicate(s) and _separated(s, occupied)]
    assert options
    return dict(options[int(rng.integers(len(options)))])


def _contour_witness(info):
    for k, period in info["programs"].items():
        if period[0] == period[1]:
            continue
        region = info["regions"][k]
        mask, depth = region["mask"], region["depth"]
        rr, cc = np.nonzero(mask)
        box = np.minimum.reduce((rr-rr.min(), rr.max()-rr, cc-cc.min(), cc.max()-cc))
        palette = np.asarray(period)
        actual = palette[depth[mask] % 2]
        if (region["area"] < (rr.max()-rr.min()+1)*(cc.max()-cc.min()+1)
                and np.any(actual != palette[box % 2])
                and np.any(actual != palette[_erosion_depth(mask, False)[mask] % 2])):
            return True
    return False


def _seal_signature(seals):
    return sorted((s["span"], s["beads"], s["period"]) for s in seals)


def _make_pair(rng, bg, wall_color, profile):
    rectangle = profile == 1 or (profile > 1 and rng.random() < .25)
    wall = _polygon(rng, first=profile == 0, rectangle=rectangle)
    target = 1 if profile == 0 else 5 if profile == 1 else int(rng.integers(4 if profile == 3 else 1, 7 if profile == 3 else 6))
    wall = _subdivide(rng, wall, target)
    candidates, regions = _candidates(wall)
    closed = _closed_ids(regions)
    required = len(closed) == 1 if profile == 0 else len(closed) == 5 if profile == 1 else len(closed) >= (4 if profile == 3 else 1)
    assert required
    rng.shuffle(candidates)
    seals, openings, occupied, assigned = [], [], set(), set()
    if profile == 1:
        shared = _choose(rng, candidates, lambda s: len(s["ids"]) == 2, occupied)
        seals.append(shared)
        occupied |= _footprint(shared)
        assigned |= shared["ids"]
        single = _choose(rng, candidates, lambda s: len(s["ids"]) == 1 and not (s["ids"] & assigned), occupied)
        seals.append(single)
        occupied |= _footprint(single)
        assigned |= single["ids"]
        opening = _choose(rng, candidates, lambda s: len(s["ids"]) == 1 and not (s["ids"] & assigned), occupied)
        openings.append(opening)
        occupied |= _footprint(opening)
    else:
        wanted = 1 if profile == 0 else int(rng.integers(3 if profile == 3 else 1, min(4 if profile == 3 else 2, len(closed)) + 1))
        shared_first = profile > 1 and len(closed) >= 3 and rng.random() < .4
        for index in range(wanted):
            incident_count = 2 if index == 0 and shared_first else 1
            options = [s for s in candidates if len(s["ids"]) == incident_count and not (s["ids"] & assigned)
                       and _separated(s, occupied)]
            if not options and incident_count == 2:
                options = [s for s in candidates if len(s["ids"]) == 1 and not (s["ids"] & assigned)
                           and _separated(s, occupied)]
            if not options:
                break
            seal = dict(options[int(rng.integers(len(options)))])
            seals.append(seal)
            occupied |= _footprint(seal)
            assigned |= seal["ids"]
        assert len(seals) >= (3 if profile == 3 else 1)
        if profile != 0:
            options = [s for s in candidates if len(s["ids"]) == 1 and not (s["ids"] & assigned)
                       and _separated(s, occupied)]
            if options and rng.random() < .65:
                opening = dict(options[int(rng.integers(len(options)))])
                openings.append(opening)
                occupied |= _footprint(opening)
    palette = [int(c) for c in rng.permutation(10) if c not in (bg, wall_color)]
    for i, seal in enumerate(seals):
        a, b = palette[(2*i) % 8], palette[(2*i+1) % 8]
        if (profile == 1 and i == 1) or (profile > 1 and i > 0 and rng.random() < .25):
            b = a
        seal["period"] = (a, b)
    # An intentional opening is part of the intended output geometry.
    for opening in openings:
        for p in opening["gap"]:
            wall[p] = False
    inp = np.full(wall.shape, bg, np.int8)
    inp[wall] = wall_color
    for seal in seals:
        for p in seal["gap"]:
            inp[p] = bg
        for p, color in zip(seal["beads"], seal["period"]):
            assert inp[p] == bg
            inp[p] = color
    expected_signature = _seal_signature(seals)
    assert _seal_signature(_read_seals(inp, bg, wall_color)) == expected_signature
    # Detached/incorrectly attached colored diagonals are discarded. Accept them
    # only when the complete visible scene still identifies exactly these seals.
    clutter_count = 0
    for _ in range(int(rng.integers(1, 4))):
        for _attempt in range(48):
            r, c = map(int, rng.integers([0, 0], inp.shape))
            dr, dc = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
            length = int(rng.choice((2, 3)))
            cells = [(r+k*dr, c+k*dc) for k in range(length)]
            existing_beads = np.argwhere((inp != bg) & (inp != wall_color))
            if not all(0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1] and inp[rr, cc] == bg
                       and all(max(abs(rr-int(ar)), abs(cc-int(ac))) > 1 for ar, ac in existing_beads)
                       for rr, cc in cells):
                continue
            trial = inp.copy()
            for i, p in enumerate(cells):
                trial[p] = palette[(i + int(rng.integers(8))) % 8]
            try:
                if _seal_signature(_read_seals(trial, bg, wall_color)) != expected_signature:
                    continue
            except ValueError:
                continue
            inp = trial
            clutter_count += 1
            break
    assert clutter_count
    # Labels come from the clean construction, never from solve's return value.
    out, _, _, _ = _paint(wall, seals, bg, wall_color, construction=True)
    predicted, info = _decode(inp)
    assert np.array_equal(predicted, out)
    assert info["programs"] and any(p[0] != p[1] for p in info["programs"].values())
    if profile == 0:
        assert _contour_witness(info)
        assert max(int(r["depth"].max()) for r in info["regions"].values() if "depth" in r) >= 3
    if profile == 1:
        assert len(info["programs"]) == 3
        assert _closed_ids(info["regions"]) - info["programs"].keys()
        assert len(_gaps(wall)) >= 1
        assert any(len(_incident(s, info["labels"]) & _closed_ids(info["regions"])) == 2 for s in info["seals"])
    # Independent spatial symmetries preserve the same attachment and depth rule.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, turns), np.rot90(out, turns)
    if reflect:
        inp, out = inp[:, ::-1], out[:, ::-1]
    return {"input": inp.copy(), "output": out.copy()}


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if n <= t:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    shift = int(rng.integers(1, 10))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(n + t):
        split = "train" if i < n else "test"
        profile = i if i < 2 else 2 if split == "train" else 3
        bg, wall_color = colors[i % 10], colors[(i + shift) % 10]
        # At most 64 small construction attempts, one pair live at a time.
        for _attempt in range(64):
            try:
                pair = _make_pair(rng, bg, wall_color, profile)
                assert _key(pair["input"]) not in seen
                break
            except (AssertionError, ValueError):
                continue
        else:
            raise RuntimeError("Bounded construction could not realize the requested geometry")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    predicted, info = _decode(inp)
    assert np.array_equal(predicted, out), "Output differs from complete input-only reconstruction"
    bg, wall_color = info["bg"], info["wall_color"]
    assert not _thick(info["wall"])
    assert info["seals"] and info["programs"]
    assert np.all(out[inp == wall_color] == wall_color)
    assert np.array_equal(out == wall_color, info["wall"])
    used_beads = {p for s in info["seals"] for p in s["beads"]}
    colored = set(map(tuple, np.argwhere((inp != bg) & (inp != wall_color)).tolist()))
    assert colored - used_beads, "No irrelevant diagonal evidence to discriminate"
    remaining = colored.copy()
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, stroke = [start], []
        while queue:
            r, c = queue.pop()
            stroke.append((r, c))
            for dr, dc in _D8:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        stroke.sort()
        assert len(stroke) in (2, 3), "Colored stroke length is outside the family"
        deltas = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(stroke, stroke[1:])]
        assert len(set(deltas)) == 1 and deltas[0][0] == 1 and abs(deltas[0][1]) == 1
    assert all(int(out[p]) == wall_color for s in info["seals"] for p in s["gap"])
    assert set(np.unique(out)) <= set(np.unique(inp))
    for k, period in info["programs"].items():
        region = info["regions"][k]
        assert region["closed"] and set(period).isdisjoint((bg, wall_color))
        assert np.array_equal(_construction_depth(region["mask"]), region["depth"])
        assert np.array_equal(out[region["mask"]], np.asarray(period)[region["depth"][region["mask"]] % 2])
    for k, region in info["regions"].items():
        if k not in info["programs"]:
            assert np.all(out[region["mask"]] == bg), "Unsealed or unprogrammed region was colored"
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, t = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _pair_info(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    assert any(_contour_witness(info) for info in train), "No nonrectangular erosion demonstration"
    assert any(len(set(info["programs"].values())) >= 2 for info in train), "No local-program contrast"
    assert any(any(p[0] == p[1] for p in info["programs"].values()) for info in train)
    assert any(any(len(_incident(s, info["labels"]) & _closed_ids(info["regions"])) == 2 for s in info["seals"]) for info in train)
    assert any(_gaps(info["wall"]) for info in train), "No preserved unsupported gap"
    assert any(_closed_ids(info["regions"]) - info["programs"].keys() for info in train), "No blank unprogrammed enclosure"
    # A omission: clean wall alone misses the propagated bands.
    # B omission: keeping broken geometry changes non-wall cells, not just gaps.
    for pair, info in zip(puzzle["train"], train):
        inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
        no_a = np.where(info["wall"], info["wall_color"], info["bg"])
        assert np.any(no_a != out)
        no_b = _paint(inp == info["wall_color"], info["seals"], info["bg"], info["wall_color"])[0]
        witness = ~info["wall"] & (inp == info["bg"])
        assert np.any((no_b != out) & witness), "Repair has no downstream contour consequence"
    assert all(1 <= len(info["seals"]) <= 2 for info in train)
    assert all(3 <= len(info["seals"]) <= 4 for info in infos["test"])


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for field in ("input", "output"):
                a = np.asarray(pair[field], np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 puzzles streamed in memory; no files, logs, or retained payloads."""
    shapes, geometry, palettes, fingerprints, rooms, seals, directions = (set() for _ in range(7))
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_digest(puzzle))
        if seed == 0:
            first = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _pair_info(pair)
                shapes.add(pair["input"].shape)
                geometry.add(hashlib.sha256(bytes(info["wall"].shape) + info["wall"].tobytes()).hexdigest())
                palettes.add((info["bg"], info["wall_color"]))
                rooms.add(len(_closed_ids(info["regions"])))
                seals.add(len(info["seals"]))
                directions.update((s["span"][2], s["span"][3], s["endpoint"], s["side"]) for s in info["seals"])
        # Symmetries apply to attachment order and contour depth together.
        pair = puzzle["test"][0]
        for turns in range(4):
            a, b = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
            assert np.array_equal(solve(a), b)
            assert np.array_equal(solve(a[:, ::-1]), b[:, ::-1])
    assert len(fingerprints) == 42 and len(shapes) >= 40 and len(geometry) >= 100
    assert len(palettes) >= 40 and len(rooms) >= 4 and len(seals) >= 3 and len(directions) == 8
    assert _digest(first) == _digest(generate_puzzle(0))
    for n, t in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(213, n, t))
    for n, t in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal example counts accepted")
    pair = first["train"][0]
    _, info = _decode(pair["input"])
    seal = info["seals"][0]
    swapped = pair["input"].copy()
    p, q = seal["beads"]
    swapped[p], swapped[q] = swapped[q], swapped[p]
    assert not np.array_equal(solve(swapped), pair["output"]), "Bead order has no effect"
    recolor = np.array([3, 7, 1, 8, 2, 6, 9, 0, 5, 4], np.int8)
    assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
    # Negative regression: full-label validation rejects missing bands, an
    # omitted repair, and retained irrelevant debris separately.
    edits = []
    edits.append((seal["gap"][0], info["bg"]))
    k = next(iter(info["programs"]))
    cell = tuple(map(int, np.argwhere(info["regions"][k]["mask"])[0]))
    edits.append((cell, info["bg"]))
    unused = next(tuple(map(int, p)) for p in np.argwhere((pair["input"] != info["bg"]) & (pair["input"] != info["wall_color"]))
                  if tuple(p) not in {p for s in info["seals"] for p in s["beads"]})
    edits.append((unused, int(pair["input"][unused]) if pair["output"][unused] != pair["input"][unused] else info["wall_color"]))
    for cell, color in edits:
        broken_pair = {"input": pair["input"], "output": pair["output"].copy()}
        broken_pair["output"][cell] = color
        try:
            _pair_info(broken_pair)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupt output was accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "wall_geometries": len(geometry), "role_pairs": len(palettes),
            "closed_region_counts": sorted(rooms), "seal_counts": sorted(seals),
            "attachment_orientations": len(directions)}
