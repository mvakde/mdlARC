"""Complete a launcher network, then classify whole rows by color equality.

Only standard library and NumPy are used. No parent module, file, example, or
generation state is consulted by solve. The structural-ring color labels true;
the background labels false. See task_spec.md for the supported domain.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


PORTS = {"NW": (-1, -1), "N0": (-1, 0), "N1": (-1, 1), "NE": (-1, 2),
         "W0": (0, -1), "E0": (0, 2), "W1": (1, -1), "E1": (1, 2),
         "SW": (2, -1), "S0": (2, 0), "S1": (2, 1), "SE": (2, 2)}
HOST_SEEDS = (0, 225543960, 659006586, 256374015, 164353041, 667009235,
              791557535, 666172770, 876655683, 876357998, 323854563,
              603376884, 26385114, 405087697, 195146732, 434447639,
              808342945, 747383544, 13905426, 443185748, 680699622,
              75766857, 978533043, 625521230, 591655377, 199052106,
              562581622, 508930392, 186094280, 6548327, 773637463,
              859858392, 647881095, 564387179, 429217310, 453399993,
              764920577, 594100864, 93473217, 865486793, 405160826,
              889327142)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or a.dtype.kind not in "iu" or
            not all(1 <= n <= 30 for n in a.shape) or
            np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _direction(offset):
    return tuple(-1 if x == -1 else 1 if x == 2 else 0 for x in offset)


def _parse(a):
    """Find a unique exact cover by cores and incomplete common-color rings."""
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[bg]) == 1, "Background frequency tie")
    candidates = {}
    h, w = a.shape
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(a[r, c])
            if color == bg or not np.all(a[r:r+2, c:c+2] == color):
                continue
            ring = [((r+dr, c+dc), (dr, dc)) for dr, dc in PORTS.values()
                    if _inside((r+dr, c+dc), a.shape)]
            solid = [int(a[p]) for p, _ in ring if int(a[p]) != bg]
            if len(solid) < 3 or len(solid) == len(ring) or len(set(solid)) != 1:
                continue
            structure = solid[0]
            if structure == color:
                continue
            footprint = {(rr, cc) for rr in range(r-1, r+3)
                         for cc in range(c-1, c+3) if _inside((rr, cc), a.shape)}
            ports = [(p, _direction(offset)) for p, offset in ring if int(a[p]) == bg]
            candidates.setdefault(structure, []).append(
                {"core": (r, c), "color": color, "ports": ports, "footprint": footprint})
    actual = set(zip(*np.where(a != bg)))
    solutions = []
    for structure, launchers in candidates.items():
        occupied, covered = set(), set()
        for launcher in launchers:
            footprint = launcher["footprint"]
            if footprint & occupied:
                break
            occupied |= footprint
            covered |= {p for p in footprint if int(a[p]) != bg}
        else:
            if covered == actual:
                solutions.append((structure, launchers))
    _require(len(solutions) == 1, "Launcher roles or exact cover are ambiguous")
    structure, launchers = solutions[0]
    return bg, structure, sorted(launchers, key=lambda x: x["core"])


def _walk(start, velocity, shape, raster="floor"):
    if raster == "sign":
        velocity = tuple(int(np.sign(x)) for x in velocity)
    scale = max(map(abs, velocity))
    path = []
    for tick in range(3 * max(shape) + 1):
        p = tuple(start[k] + (1 if velocity[k] >= 0 else -1) *
                  ((tick * abs(velocity[k]) + (scale-1 if raster == "ceil" else 0)) // scale)
                  for k in (0, 1))
        if not _inside(p, shape):
            break
        path.append(p)
    return path


def _hit(aa, bb):
    hits = []
    for ia, a in enumerate(aa):
        cells = {p: i for i, p in enumerate(a)}
        mids = {(p[0]+q[0], p[1]+q[1]): i
                for i, (p, q) in enumerate(zip(a, a[1:]))
                if p[0] != q[0] and p[1] != q[1]}
        for ib, b in enumerate(bb):
            for j, p in enumerate(b):
                if p in cells:
                    hits.append((cells[p]-1, j-1, ia, ib, "cell"))
            for j, (p, q) in enumerate(zip(b, b[1:])):
                midpoint = (p[0]+q[0], p[1]+q[1])
                if p[0] != q[0] and p[1] != q[1] and midpoint in mids:
                    i = mids[midpoint]
                    if {a[i], a[i+1]} != {p, q}:
                        hits.append((i, j, ia, ib, "between"))
    return min(hits, key=lambda x: (x[0]+x[1], max(x[:2]), x[2], x[3])) if hits else None


def _network(input_grid, raster="floor"):
    a = _grid(input_grid)
    bg, structure, launchers = _parse(a)
    groups, colors, owners = [], [], []
    for launcher_id, launcher in enumerate(launchers):
        for p, v in launcher["ports"]:
            sid = len(colors)
            colors.append(launcher["color"])
            owners.append(launcher_id)
            groups.append({"v": v, "members": [(sid, p)], "depth": 0})
    out = a.copy()
    occupied, last, segments = {}, {}, set()
    events = []

    def paint(sid, path):
        for i, p in enumerate(path):
            _require(int(a[p]) == bg, "Ray overwrites a launcher")
            _require(p not in occupied or
                     (i == 0 and occupied[p] == sid and last.get(sid) == p),
                     "Ray overlap or self-intersection")
            occupied[p] = sid
            last[sid] = p
            out[p] = colors[sid]
        for p, q in zip(path, path[1:]):
            if p[0] != q[0] and p[1] != q[1]:
                midpoint = (p[0]+q[0], p[1]+q[1])
                sign = 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1
                _require((midpoint, -sign) not in segments, "Unresolved crossing")
                segments.add((midpoint, sign))

    for _ in range(len(colors) + 1):
        forecasts = [[_walk(p, g["v"], a.shape, raster) for _, p in g["members"]] for g in groups]
        hits = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                hit = _hit(forecasts[i], forecasts[j])
                if hit is not None:
                    hits.append((i, j, hit))
        if not hits:
            for group, paths in zip(groups, forecasts):
                for (sid, _), path in zip(group["members"], paths):
                    paint(sid, path)
            break
        nearest = {}
        for i, j, hit in hits:
            nearest[i] = min(nearest.get(i, 1000), hit[0])
            nearest[j] = min(nearest.get(j, 1000), hit[1])
        ready = [(i, j, hit) for i, j, hit in hits
                 if hit[0] == nearest[i] and hit[1] == nearest[j]]
        _require(ready, "Cyclic encounter ordering")
        i, j, hit = min(ready, key=lambda x: (max(x[2][:2]), sum(x[2][:2]), x[0], x[1]))
        for k, l, other in hits:
            if (k, l) != (i, j):
                _require(not ((k in (i, j) and other[0] == nearest[k]) or
                              (l in (i, j) and other[1] == nearest[l])), "Multiway encounter")
        u, v = groups[i]["v"], groups[j]["v"]
        _require(u != v and max(map(abs, u)) == max(map(abs, v)) == 1,
                 "Parallel overlap or an unsupported staircase re-encounter")
        sums = tuple(u[k]+v[k] for k in (0, 1))
        divisor = math.gcd(*map(abs, sums))
        _require(divisor > 0, "Head-on cancellation")
        kind = ("perpendicular" if sum(map(abs, u)) == sum(map(abs, v)) == 1 else
                "opposing_diagonals" if sum(map(abs, u)) == sum(map(abs, v)) == 2 else
                "staircase")
        if kind == "staircase":
            _require(sum(u[k]*v[k] for k in (0, 1)) > 0, "Obtuse encounter")
        _require(min(hit[:2]) >= 0, "Encounter at a strand origin")
        members = []
        for index, cut in ((i, hit[0]), (j, hit[1])):
            for (sid, _), path in zip(groups[index]["members"], forecasts[index]):
                _require(cut < len(path)-1, "Bundle clipped before encounter")
                paint(sid, path[:cut+1])
                members.append((sid, path[cut]))
        velocity = tuple(x // divisor for x in sums)
        depth = 1 + max(groups[i]["depth"], groups[j]["depth"])
        events.append({"kind": kind, "intersection": hit[4], "incoming": (u, v),
                       "outgoing": velocity, "depth": depth,
                       "approach_distances": tuple(cut + (1 if hit[4] == "cell" else .5)
                                                   for cut in hit[:2]),
                       "last": tuple(p for _, p in members),
                       "colors": tuple(colors[s] for s, _ in members),
                       "launchers": tuple(sorted({owners[s] for s, _ in members}))})
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append({"v": velocity, "members": members, "depth": depth})
    else:
        raise AssertionError("Too many merges")
    return out, {"background": bg, "structure": structure,
                 "launchers": launchers, "events": events}


def _row_mask(a):
    return np.all(a == a[:, :1], axis=1)


def _label(a, positive, negative):
    return np.repeat(np.where(_row_mask(a), positive, negative)[:, None],
                     a.shape[1], axis=1).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    network, info = _network(input_grid)
    return _label(network, info["structure"], info["background"])


def _build(shape, specs):
    a = np.zeros(shape, dtype=np.int8)
    occupied = set()
    for r, c, color, ports in specs:
        _require(0 <= r < shape[0]-1 and 0 <= c < shape[1]-1, "Core out of bounds")
        footprint = {(rr, cc) for rr in range(r-1, r+3) for cc in range(c-1, c+3)
                     if _inside((rr, cc), shape)}
        _require(not (footprint & occupied), "Footprints overlap")
        occupied |= footprint
        port_cells = {(r+PORTS[n][0], c+PORTS[n][1]) for n in ports}
        _require(all(_inside(p, shape) for p in port_cells), "Clipped port")
        ring = {(r+dr, c+dc) for dr, dc in PORTS.values()
                if _inside((r+dr, c+dc), shape)}
        _require(len(ring-port_cells) >= 3, "Not enough ring")
        for p in ring-port_cells:
            a[p] = 8
        a[r:r+2, c:c+2] = color
    return a


def _mirror_specs(specs, shape, flip_rows=False, flip_cols=False):
    reverse = {offset: name for name, offset in PORTS.items()}
    result = []
    for r, c, color, names in specs:
        nr = shape[0]-2-r if flip_rows else r
        nc = shape[1]-2-c if flip_cols else c
        offsets = [PORTS[name] for name in names]
        names = tuple(reverse[(1-dr if flip_rows else dr, 1-dc if flip_cols else dc)]
                      for dr, dc in offsets)
        result.append((nr, nc, color, names))
    return result


def _active_specs(rng, shape, kind, late, colors, short_staircase=False):
    """Sample a boundary emitter and an interior ray aimed at it."""
    h, w = shape
    slot = int(rng.integers(2))
    max_distance = min(5, h-8)
    _require(max_distance >= 3, "Active scene too short")
    distance = int(rng.integers(3, max_distance+1))
    y = int(rng.integers(4, h-distance-3))
    r = y-slot
    if kind == "perpendicular":
        gaps = [gap for gap in range(2, min(y-1, w-5)+1)
                if w-gap-2 != distance]
        gap = int(rng.choice(gaps))
        k = w-gap
        name = str(rng.choice(("N0", "N1")))
        ar, ac = y+distance+1, k-(name == "N1")
    else:
        gap = (1 if late else 3 if short_staircase else
               int(rng.integers(3, min(8, y+1, w-distance-4)+1)))
        if gap == 3:
            # The completed source row is fully occupied by two different
            # payloads. Keep its first core slot so a separate blank row also
            # distinguishes spatial bending from either crossing priority.
            slot = 0
            r = y
        k = w-gap
        name = "NE"
        ar, ac = y+distance+1, k-distance-2
    specs = [(r, 0, colors[0], ("E0" if slot == 0 else "E1",)),
             (ar, ac, colors[1], (name,))]
    if bool(rng.integers(2)):
        specs = _mirror_specs(specs, shape, flip_cols=True)
    return specs


def _free_spec(rng, row, width, color):
    both = bool(rng.integers(2))
    ports = ("E0", "E1") if both else (str(rng.choice(("E0", "E1"))),)
    specs = [(row, 0, color, ports)]
    if bool(rng.integers(2)):
        specs = _mirror_specs(specs, (row+3, width), flip_cols=True)
    return specs[0]


def _scene(rng, modes, width, free_count=1, short_staircase=False):
    """Place all emitters before tracing; no whole parent puzzle is generated."""
    if not modes:
        n = int(rng.integers(1, 4))
        gap = int(rng.integers(4, 7))
        top = int(rng.integers(0, 3))
        h = top + gap*(n-1) + int(rng.integers(3, 6))
        specs = [_free_spec(rng, top+i*gap, width, 1 if rng.random() < .3 else i+1)
                 for i in range(n)]
    elif len(modes) == 1:
        active_h = int(rng.integers(14, 20))
        kind, late = modes[0]
        specs = _active_specs(rng, (active_h, width), kind, late, (1, 2), short_staircase)
        free_count = max(1, free_count)
        for i in range(free_count):
            color = 1 if rng.random() < .3 else 3+i
            specs.append(_free_spec(rng, active_h+1+i*4, width, color))
        h = active_h+1+4*(free_count-1)+int(rng.integers(3, 5))
    else:
        top_h, bottom_h = (int(x) for x in rng.integers(11, 13, 2))
        middle_h = int(rng.integers(5, 7))
        h = top_h+middle_h+bottom_h
        kind, late = modes[0]
        specs = _active_specs(rng, (top_h, width), kind, late, (1, 2), short_staircase)
        kind, late = modes[1]
        bottom = _active_specs(rng, (bottom_h, width), kind, late, (3, 4), short_staircase)
        bottom = _mirror_specs(bottom, (bottom_h, width), flip_rows=True)
        specs += [(r+top_h+middle_h, c, color, ports) for r, c, color, ports in bottom]
        specs.append(_free_spec(rng, top_h+1, width, 1 if rng.random() < .3 else 5))
    _require(h <= 30, "Scene exceeds ARC size")
    if bool(rng.integers(2)):
        specs = _mirror_specs(specs, (h, width), flip_rows=True)
    a = _build((h, width), specs)
    network, info = _network(a)
    _require(len(info["events"]) == len(modes), "Unplanned encounter")
    _require(sorted(e["kind"] for e in info["events"]) == sorted(m[0] for m in modes),
             "Wrong encounter kind")
    return a, network


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _signature(a):
    return a.shape, a.tobytes()


def _straight(a, info, reverse=False):
    """A deliberately wrong pass-through interpretation, for contrast checks."""
    out = a.copy()
    launchers = info["launchers"][::-1] if reverse else info["launchers"]
    for launcher in launchers:
        for p, v in launcher["ports"]:
            for q in _walk(p, v, a.shape):
                if int(a[q]) == info["background"]:
                    out[q] = launcher["color"]
    return out


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    ne = _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    if nt >= 4:
        curriculum = [[], [("perpendicular", False)], [("staircase", False)], [("staircase", True)]]
        kinds = ("perpendicular", "staircase")
    else:
        curriculum = [[("staircase", False)], [("staircase", True)]]
        kinds = ("staircase",)
    while len(curriculum) < nt:
        kind = str(rng.choice(kinds))
        curriculum.append([(kind, kind == "staircase" and bool(rng.integers(2)))])
    rng.shuffle(curriculum)
    puzzle = {"train": [], "test": []}
    seen, test_masks = set(), set()
    uniform_colors = set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            if split == "train":
                modes = curriculum[index]
            else:
                # Shift: longer canvas and/or a composition of two taught encounters.
                count_events = 2 if rng.random() < .65 else 1
                modes = []
                for _ in range(count_events):
                    kind = str(rng.choice(kinds))
                    modes.append((kind, kind == "staircase" and bool(rng.integers(2))))
            for attempt in range(100):
                width = int(rng.integers(13, 24)) if split == "train" else int(rng.integers(24, 31))
                try:
                    a, network = _scene(rng, modes, width,
                                        free_count=int(rng.integers(1, 3)),
                                        short_staircase=split == "train")
                    palette = np.asarray(rng.permutation(10), dtype=np.int8)
                    # Role assignment is visible, not a convention learned from a seed.
                    a, network = palette[a], palette[network]
                    bg, positive = int(palette[0]), int(palette[8])
                    out = _label(network, positive, bg)
                    sig = _signature(a)
                    mask_sig = (a.shape, _row_mask(network).tobytes())
                    if sig in seen or (split == "test" and mask_sig in test_masks):
                        continue
                    full_colors = {int(row[0]) for row in network
                                   if np.all(row == row[0]) and int(row[0]) != bg}
                    if split == "train" and index == 1 and full_colors <= uniform_colors:
                        continue
                    # Early bends must affect the mask with either crossing priority.
                    # Late bends are a deliberate contrast: a collided row can complete.
                    if any(not late for _, late in modes):
                        _, info = _network(a)
                        _require(all(not np.array_equal(_row_mask(network), _row_mask(_straight(a, info, rev)))
                                     for rev in (False, True)), "Invisible collision")
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("Bounded scene construction exhausted")
            seen.add(sig)
            if split == "test":
                test_masks.add(mask_sig)
            else:
                uniform_colors |= full_colors
            puzzle[split].append({"input": a, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Illegal counts")
    seen, taught, needed, full_colors = set(), set(), set(), set()
    contrasts = {"input": False, "empty_only": False, "endpoints": False,
                 "occupancy": False, "binary_occupancy": False,
                 "straight": False, "reverse_straight": False}
    phases = set()
    raster_contrasts = {"ceil": False, "sign": False}
    train_width, test_width = 0, 31
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "Output shape changed")
            sig = _signature(a)
            _require(sig not in seen, "Repeated input")
            seen.add(sig)
            network, info = _network(a)
            bg, positive = info["background"], info["structure"]
            mask = _row_mask(network)
            _require(np.array_equal(out, _label(network, positive, bg)), "Incorrect row mask")
            _require(np.any(mask) and np.any(~mask), "Both row classes must appear")
            _require(np.array_equal(a[a != bg], network[a != bg]), "Launcher changed")
            nonbg_full = {int(row[0]) for row in network if np.all(row == row[0]) and int(row[0]) != bg}
            _require(nonbg_full, "No completed payload row")
            kinds = {e["kind"] for e in info["events"]}
            _require(all(e["approach_distances"][0] != e["approach_distances"][1]
                         for e in info["events"]), "Equal-clock shortcut is not contradicted")
            if split == "test":
                needed |= kinds
                test_width = min(test_width, a.shape[1])
                _require(kinds, "Test has no collision")
            else:
                taught |= kinds
                train_width = max(train_width, a.shape[1])
                full_colors |= nonbg_full
                alternatives = {"input": _row_mask(a), "empty_only": np.all(network == bg, axis=1),
                                "endpoints": network[:, 0] == network[:, -1],
                                "occupancy": np.all(network != bg, axis=1),
                                "binary_occupancy": (np.all(network == bg, axis=1) |
                                                     np.all(network != bg, axis=1)),
                                "straight": _row_mask(_straight(a, info)),
                                "reverse_straight": _row_mask(_straight(a, info, True))}
                for name, alt in alternatives.items():
                    contrasts[name] |= not np.array_equal(mask, alt)
                if "staircase" in kinds:
                    for raster in raster_contrasts:
                        try:
                            alternative, _ = _network(a, raster)
                            raster_contrasts[raster] |= not np.array_equal(mask, _row_mask(alternative))
                        except AssertionError:
                            raster_contrasts[raster] = True
                for launcher_id, launcher in enumerate(info["launchers"]):
                    r, c = launcher["core"]
                    for p, v in launcher["ports"]:
                        if v[0] == 0 and (c == 0 or c == a.shape[1]-2):
                            affected = any(launcher_id in e["launchers"] for e in info["events"])
                            if affected:
                                phases.add(("staircase" in kinds, bool(mask[p[0]])))
    _require(needed <= taught, "Untaught collision type")
    _require(len(full_colors) >= 2, "Training fails color-independent full-row contrast")
    _require(all(contrasts.values()), "Training permits a simple alternative: " + str(contrasts))
    _require(train_width < test_width, "Missing long-row test shift")
    if "staircase" in needed:
        _require((True, True) in phases and (True, False) in phases,
                 "Staircase tests need both completing and diverted demonstrations")
        _require(all(raster_contrasts.values()), "Training leaves staircase phase or slope unresolved")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded: stream 42 seeds; retain statistics and digests, no files."""
    cardinal = _build((17, 17), [(7, 0, 1, ("E0",)), (13, 12, 2, ("N0",))])
    cardinal_network, cardinal_info = _network(cardinal)
    _require(cardinal_info["events"][0]["outgoing"] == (-1, 1), "Perpendicular heading fixture failed")
    _require([int(cardinal_network[p]) for p in ((6, 12), (5, 13), (4, 14), (3, 15), (2, 16))] == [1]*5,
             "Perpendicular first strand fixture failed")
    _require([int(cardinal_network[p]) for p in ((7, 13), (6, 14), (5, 15), (4, 16))] == [2]*4,
             "Perpendicular partner fixture failed")
    _require(cardinal_info["events"][0]["approach_distances"] == (10, 5),
             "Unequal-distance spatial encounter fixture failed")
    # Hand-derived paths: the major-axis step occurs before the diagonal step.
    for collision_col in (12, 16):
        specs = [(6, 0, 1, ("E0",)), (11, collision_col-6, 2, ("NE",))]
        a = _build((15, 17), specs)
        network, info = _network(a)
        _require(len(info["events"]) == 1 and info["events"][0]["outgoing"] == (-1, 2),
                 "Staircase heading fixture failed")
        _require(bool(_row_mask(network)[6]) == (collision_col == 16), "Late-completion fixture failed")
        if collision_col == 12:
            _require([int(network[p]) for p in ((6, 12), (5, 13), (5, 14), (4, 15), (4, 16))] == [1]*5,
                     "Major-axis-first phase fixture failed")
            _require([int(network[p]) for p in ((7, 12), (6, 13), (6, 14), (5, 15), (5, 16))] == [2]*5,
                     "Partner strand fixture failed")
        for rows in (False, True):
            for cols in (False, True):
                mirrored = _build(a.shape, _mirror_specs(specs, a.shape, rows, cols))
                expected = network[::-1] if rows else network
                expected = expected[:, ::-1] if cols else expected
                _require(np.array_equal(_network(mirrored)[0], expected), "Reflection changed ray law")
    shapes, palettes, layouts, masks, hashes = set(), set(), set(), set(), set()
    event_counts, launcher_counts, positive_colors = set(), set(), set()
    completed = diverted = late = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        hashes.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                network, info = _network(a)
                shapes.add(a.shape)
                palettes.add(tuple(int(c) for c in np.unique(a)))
                masks.add((a.shape, _row_mask(network).tobytes()))
                positive_colors.add(info["structure"])
                event_counts.add(len(info["events"]))
                launcher_counts.add(len(info["launchers"]))
                layouts.add((a.shape, tuple((l["core"], tuple(l["ports"])) for l in info["launchers"])))
                completed += int(np.count_nonzero(_row_mask(network) & np.any(network != info["background"], axis=1)))
                diverted += int(not np.array_equal(_row_mask(a), _row_mask(network)))
                late += any(e["kind"] == "staircase" for e in info["events"])
    _require(len(hashes) == 42 and len(shapes) >= 40 and len(layouts) >= 150, "Weak geometry variation")
    _require(len(palettes) >= 40 and len(positive_colors) >= 8, "Weak palette variation")
    _require({0, 1, 2} <= event_counts and {1, 2, 3, 4, 5} <= launcher_counts, "Weak count variation")
    _require(len(masks) >= 150 and completed >= 210 and diverted >= 150 and late >= 42,
             "Weak causal or answer variation")
    _require(_fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0)), "Nondeterminism")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(20261023, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "Counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Input-only solve accepts nested lists too and does not mutate its argument.
    p = generate_puzzle(91, 2, 1)
    a = p["test"][0]["input"].copy()
    _require(np.array_equal(solve(a.tolist()), p["test"][0]["output"]), "List solving failed")
    _require(np.array_equal(a, p["test"][0]["input"]), "Input mutated")
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted label accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "layouts": len(layouts),
            "palettes": len(palettes), "answer_masks": len(masks),
            "positive_colors": len(positive_colors), "completed_payload_rows": completed,
            "pairs_changed_by_network": diverted, "staircase_pairs": late}
