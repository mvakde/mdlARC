"""Complete marker symmetries, then highlight every empty clipped 3x3 center.

The highlight uses the marker color. Roles are recovered from input geometry;
there is no seed, training-label, or generation-state dependency in solve().
No files, logs, caches, or execution traces are written by this module.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 881736552, 144615942, 506155694, 441938806, 90214400,
              820464890, 712573730, 539054290, 948978633, 584907793,
              874281748, 601302261, 406491837, 799395899, 425447023,
              961609630, 114002742, 780633041, 710706595, 850329839,
              908704317, 295733420, 102836232, 215217841, 743005811,
              32996550, 925712275, 856358112, 551421643, 310619110,
              786812361, 426802761, 222227034, 326113415, 815473217,
              858075560, 475239406, 433473543, 262000588, 493066667,
              776272849)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((0 <= a) & (a <= 9))), "grid colors must be integers 0..9")
    return a


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(sorted(cells))
    return result


def _bbox(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max()),
            int(a[:, 1].min()), int(a[:, 1].max()))


def _kind(marker):
    r0, r1, c0, c1 = _bbox(marker)
    if len(marker) == 1:
        return "P"
    if r0 == r1 and len(marker) == c1 - c0 + 1:
        return "H"
    if c0 == c1 and len(marker) == r1 - r0 + 1:
        return "V"
    raise ValueError("marker is not a straight segment or singleton")


def _read_with_marker(a, bg, color, regions):
    groups, used = [], set()
    for region in regions:
        marker = [q for q in region if a[q] == color]
        payload = [q for q in region if a[q] != color]
        _require(marker and payload, "each region needs marker and payload")
        kind = _kind(marker)
        _require(len({int(a[q]) for q in payload}) == 1, "payload must be monochrome")
        pmask = np.zeros(a.shape, bool)
        for q in payload:
            pmask[q] = True
        _require(len(_components(pmask)) == 1, "payload must be eight-connected")
        r0, r1, c0, c1 = _bbox(payload)
        ar, ac = marker[0]
        if kind == "H":
            _require(r1 < ar or r0 > ar, "payload crosses horizontal axis")
            dest = [(2 * ar - r, c) for r, c in payload]
        elif kind == "V":
            _require(c1 < ac or c0 > ac, "payload crosses vertical axis")
            dest = [(r, 2 * ac - c) for r, c in payload]
        else:
            _require((r1 <= ar or r0 >= ar) and (c1 <= ac or c0 >= ac),
                     "point payload must occupy a closed quadrant")
            dest = [(2 * ar - r, 2 * ac - c) for r, c in payload]
        _require(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1]
                     and a[r, c] == bg for r, c in dest),
                 "copy must fit in clear background")
        _require(not used.intersection(dest), "copies collide")
        used.update(dest)
        groups.append({"kind": kind, "marker": marker, "payload": payload,
                       "destination": dest})
    return groups


def _read_input(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    _require(counts.max() > a.size // 2, "input needs a strict majority background")
    bg = int(colors[counts.argmax()])
    regions = _components(a != bg)
    _require(regions, "input needs a foreground group")
    possibilities = []
    for color in colors:
        if color == bg:
            continue
        try:
            groups = _read_with_marker(a, bg, int(color), regions)
        except ValueError:
            continue
        possibilities.append((int(color), groups))
    _require(len(possibilities) == 1, "marker role must be uniquely readable from input")
    marker, groups = possibilities[0]
    return a, bg, marker, groups


def _expand(a, groups):
    expanded = a.copy()
    for g in groups:
        for source, dest in zip(g["payload"], g["destination"]):
            expanded[dest] = a[source]
    return expanded


def _clear(obstacles, diagonal=True):
    """No out-of-grid obstacles: the neighborhood is clipped, not blocked."""
    h, w = obstacles.shape
    padded = np.pad(obstacles, 1, constant_values=False)
    near = np.zeros((h, w), bool)
    for dr in range(3):
        for dc in range(3):
            if diagonal or abs(dr - 1) + abs(dc - 1) <= 1:
                near |= padded[dr:dr + h, dc:dc + w]
    return ~near


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read input roles, add all copies once, and scan that frozen obstacle field."""
    a, bg, marker, groups = _read_input(input_grid)
    expanded = _expand(a, groups)
    selected = _clear(expanded != bg)
    out = expanded.copy()
    out[selected] = marker
    return out


def _orientation_witness(g):
    r0, _, c0, _ = _bbox(g["payload"])
    a0, _, b0, _ = _bbox(g["destination"])
    translated = {(r + a0 - r0, c + b0 - c0) for r, c in g["payload"]}
    return translated != set(g["destination"])


def _trim(mask):
    r, c = np.nonzero(mask)
    return mask[r.min():r.max() + 1, c.min():c.max() + 1].copy()


def _shape(rng, scale, hole=False, witness=False, large=False):
    if hole:
        # Its 3x3 interior has exactly one clear center. A missing outer corner
        # makes the symmetry orientation visible without opening the clear core.
        mask = np.ones((5, 5), bool)
        mask[1:4, 1:4] = False
        mask[-1, -1] = False
        return mask
    h, w = (6, 6) if large else (int(rng.integers(2, scale + 1)) for _ in range(2))
    family = int(rng.integers(7))
    mask = np.zeros((h, w), bool)
    if family == 0:
        mask[:] = True
    elif family == 1:
        mask[0, :] = mask[:, 0] = True
        mask[-1, :int(rng.integers(1, w))] = True
    elif family == 2:
        c = int(rng.integers(w))
        for r in range(h):
            mask[r, c] = True
            c = int(np.clip(c + int(rng.choice([-1, 1])), 0, w - 1))
    elif family == 3:
        mask[0, 0] = True
        for _ in range(int(rng.integers(2, h * w + 1))):
            frontier = sorted({(int(r + dr), int(c + dc))
                               for r, c in np.argwhere(mask)
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r + dr < h and 0 <= c + dc < w
                               and not mask[r + dr, c + dc]})
            if not frontier:
                break
            mask[frontier[int(rng.integers(len(frontier)))]] = True
    elif family == 4:
        mask[:] = True
        mask[:int(rng.integers(1, h)), :int(rng.integers(1, w))] = False
    elif family == 5:
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    else:
        for r in range(h):
            mask[r, min(r, w - 1)] = True
        rr, cc = np.argwhere(mask)[len(np.argwhere(mask)) // 2]
        mask[rr, max(0, int(cc) - 1):min(w, int(cc) + 2)] = True
    mask = _trim(mask)
    if witness:
        # A missing corner is asymmetric under H, V, and P and is not a line.
        mask = np.ones((h, w), bool)
        mask[-1, -1] = False
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask.copy()


def _prototype(rng, kind, scale, signs, hole=False, witness=False, large=False):
    mask = _shape(rng, scale, hole, witness and not hole, large)
    if kind == "P":
        # Closed-quadrant centers can share a row or column with the payload.
        dr, dc = (1, 1) if witness or hole else [(1, 1), (0, 1), (1, 0)][int(rng.integers(3))]
        cells = {(int(r + dr), int(c + dc)) for r, c in np.argwhere(mask)}
        # A short diagonal path attaches a sparse shape to the marker.
        target = min(cells, key=lambda q: (max(q), q))
        r, c = dr, dc
        while (r, c) != target:
            cells.add((r, c))
            r += int(np.sign(target[0] - r))
            c += int(np.sign(target[1] - c))
        cells.add((dr, dc))
        payload = sorted((signs[0] * r, signs[1] * c) for r, c in cells)
        marker = [(0, 0)]
        dest = [(-r, -c) for r, c in payload]
    else:
        # Construct horizontally and transpose for V. The finite marker may
        # overhang the payload or stop before an endpoint; it still marks an axis.
        contact = int(rng.choice(np.flatnonzero(mask[0])))
        start = int(rng.integers(-1, contact + 1))
        end = int(rng.integers(max(start + 1, contact), max(start + 2, mask.shape[1] + 2)))
        if hole:
            # An exposed axis endpoint also demonstrates that markers count as
            # obstacles during clearance, independently of the two payloads.
            end = mask.shape[1] + 1
        marker = [(0, c) for c in range(start, end + 1)]
        payload = [(signs[0] * (int(r) + 1), int(c)) for r, c in np.argwhere(mask)]
        dest = [(-r, c) for r, c in payload]
        if kind == "V":
            marker, payload, dest = [[(c, r) for r, c in points]
                                     for points in (marker, payload, dest)]
    g = {"kind": kind, "marker": marker, "payload": payload, "destination": dest}
    r0, _, c0, _ = _bbox(marker + payload + dest)
    for role in ("marker", "payload", "destination"):
        g[role] = [(r - r0, c - c0) for r, c in g[role]]
    return g


def _pack(rng, prototypes, h, w, style):
    blocked = np.zeros((h, w), bool)
    placed = []
    anchor = (int(rng.integers(h)), int(rng.integers(w)))
    for index in sorted(range(len(prototypes)),
                        key=lambda i: -len(sum((prototypes[i][k] for k in
                                               ("marker", "payload", "destination")), []))):
        g = prototypes[index]
        footprint = np.array(sorted(set(g["marker"] + g["payload"] + g["destination"])))
        gh, gw = footprint.max(axis=0) + 1
        if gh > h or gw > w:
            return None
        origins = [(r, c) for r in range(h - int(gh) + 1) for c in range(w - int(gw) + 1)]
        if style != "scan":
            rng.shuffle(origins)
        if style == "edge":
            origins.sort(key=lambda q: min(q[0], q[1], h - gh - q[0], w - gw - q[1]))
        elif style == "cluster":
            origins.sort(key=lambda q: abs(q[0] + gh / 2 - anchor[0]) +
                         abs(q[1] + gw / 2 - anchor[1]))
        for r, c in origins:
            rr, cc = footprint[:, 0] + r, footprint[:, 1] + c
            if not blocked[rr, cc].any():
                moved = {"kind": g["kind"], "index": index}
                for role in ("marker", "payload", "destination"):
                    moved[role] = [(x + r, y + c) for x, y in g[role]]
                placed.append(moved)
                for x, y in zip(rr, cc):
                    blocked[max(0, x - 1):min(h, x + 2), max(0, y - 1):min(w, y + 2)] = True
                break
        else:
            return None
    return placed


def _pair(rng, marker, kinds, required, hole, shift, test, side_plan):
    if shift == "count":
        count = int(rng.integers(4, 7)) if test else int(rng.integers(1, 4))
        scale = 3
    elif shift == "scale":
        count = int(rng.integers(1, 3))
        scale = 6 if test else 3
    else:
        count, scale = int(rng.integers(1, 5)), int(rng.integers(3, 6))
    count = max(count, len(required), 1)
    specs = list(required)
    while len(specs) < count:
        specs.append(str(rng.choice(kinds)))
    rng.shuffle(specs)
    # Every scene has an orientation witness, preventing marker-role ambiguity.
    witness_at = int(rng.integers(count))
    style = str(rng.choice(["free", "cluster", "edge"]))
    h, w = int(rng.integers(15, 31)), int(rng.integers(15, 31))
    if count >= 4 or hole:
        h, w = max(h, 25), max(w, 25)
    if test and shift == "scale":
        h, w = max(h, 27), max(w, 27)
    bg = int(rng.choice([c for c in range(10) if c != marker]))
    pool = [c for c in range(10) if c not in (bg, marker)]
    # Reuse and multiple palettes are both possible; neither signals ownership.
    rng.shuffle(pool)
    pool = pool[:int(rng.integers(1, min(5, count) + 1))]
    for attempt in range(24):
        # Large groups retain their scale for most attempts. Final retries use
        # smaller footprints on a full canvas; the rule never changes.
        cap = scale if attempt < 12 else max(2, scale - (attempt - 10) // 3)
        if attempt in (6, 12, 18):
            h, w = min(30, h + 3), min(30, w + 3)
        if attempt == 23:
            h = w = 30
        prototypes = []
        for j, kind in enumerate(specs):
            signs = side_plan[kind]
            if shift == "side" and test:
                signs = tuple(-s for s in signs)
            elif shift != "side":
                signs = tuple(int(rng.choice([-1, 1])) for _ in range(2))
            g = _prototype(rng, kind, cap, signs, hole and j == witness_at,
                           j == witness_at or kind in required,
                           test and shift == "scale" and j == witness_at)
            prototypes.append(g)
        placed = _pack(rng, prototypes, h, w, "scan" if attempt == 23 else style)
        if placed is None:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        for g in placed:
            for q in g["marker"]:
                inp[q] = marker
            for q in g["payload"]:
                inp[q] = pool[g["index"] % len(pool)]
        try:
            _, found_bg, found_marker, groups = _read_input(inp)
        except ValueError:
            continue
        if found_bg != bg or found_marker != marker:
            continue
        if not set(required) <= {g["kind"] for g in groups if _orientation_witness(g)}:
            continue
        if not any(_orientation_witness(g) for g in groups):
            continue
        expanded = _expand(inp, groups)
        clear = _clear(expanded != bg)
        # Causal evidence away from copied pixels: expansion actually changes B.
        changed_windows = (expanded == bg) & _clear(inp != bg) & ~clear
        if not clear.any() or not changed_windows.any():
            continue
        out = expanded.copy()
        out[clear] = marker
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    _require(low <= value <= high, f"{name} out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    _require(nt > ns, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    subsets = [("H",), ("V",), ("P",), ("H", "V"),
               ("H", "P"), ("V", "P"), ("H", "V", "P")]
    kinds = subsets[int(rng.integers(len(subsets)))]
    shift = str(rng.choice(["count", "scale", "side"]))
    side_plan = {k: tuple(int(rng.choice([-1, 1])) for _ in range(2)) for k in kinds}
    plans = [[] for _ in range(nt)]
    for j, kind in enumerate(rng.permutation(kinds)):
        plans[j % nt].append(str(kind))
    rng.shuffle(plans)
    hole_at = int(rng.integers(nt))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            for _ in range(12):
                pair = _pair(rng, marker, kinds, plans[index] if split == "train" else [],
                             split == "train" and index == hole_at, shift,
                             split == "test", side_plan)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "bad pair counts")
    seen, marker_colors, training_kinds, witnessed = set(), set(), set(), set()
    evidence = {name: False for name in ("edge", "corner", "diagonal", "tiny", "markers")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _, bg, marker, groups = _read_input(inp)
            marker_colors.add(marker)
            expanded = _expand(inp, groups)
            clear = _clear(expanded != bg)
            expected = expanded.copy()
            expected[clear] = marker
            _require(np.array_equal(expected, out), "output violates the composed rule")
            _require(np.array_equal(inp[inp != bg], out[inp != bg]), "original foreground changed")
            _require(clear.any() and (expanded != inp).any(), "both stages must be nonempty")
            _require(bool(np.any((expanded == bg) & _clear(inp != bg) & ~clear)),
                     "symmetry does not affect the subsequent background scan")
            _require(np.any((expanded == bg) & ~clear), "background fringe disappeared")
            if split == "train":
                training_kinds.update(g["kind"] for g in groups)
                witnessed.update(g["kind"] for g in groups if _orientation_witness(g))
                evidence["edge"] |= bool(clear[0].any() or clear[-1].any() or
                                         clear[:, 0].any() or clear[:, -1].any())
                evidence["corner"] |= bool(clear[0, 0] or clear[0, -1] or
                                           clear[-1, 0] or clear[-1, -1])
                evidence["diagonal"] |= bool(np.any(_clear(expanded != bg, False) & ~clear))
                evidence["tiny"] |= any(len(c) < 5 for c in _components(clear, False))
                marker_mask = inp == marker
                evidence["markers"] |= bool(np.any((expanded == bg) &
                                                   _clear((expanded != bg) & ~marker_mask) & ~clear))
            else:
                _require({g["kind"] for g in groups} <= training_kinds,
                         "test introduces an undemonstrated operation")
    _require(len(marker_colors) == 1, "marker/highlight convention changed")
    _require(training_kinds <= witnessed, "training hides a symmetry behind invariant shapes")
    _require(all(evidence.values()), f"training missing diagnostic evidence: {evidence}")


def self_test():
    """Bounded: 42 puzzles streamed in memory, a few count/mutation checks, no disk."""
    structures, shapes, markers, bgs, counts, silhouettes = set(), set(), set(), set(), set(), set()
    kinds_seen, pair_count, tiny, close = set(), 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                inp = pair["input"]
                before = inp.copy()
                out = solve(inp)
                assert np.array_equal(out, pair["output"]) and np.array_equal(inp, before)
                _, bg, marker, groups = _read_input(inp)
                expanded = _expand(inp, groups)
                # Check local scanning directly with clipped slices, separately
                # from the padded union implementation used in solve().
                slow = expanded.copy()
                for r in range(inp.shape[0]):
                    for c in range(inp.shape[1]):
                        if np.all(expanded[max(0, r - 1):r + 2, max(0, c - 1):c + 2] == bg):
                            slow[r, c] = marker
                assert np.array_equal(slow, out)
                # Check symmetry coordinates against actual cropped array flips.
                for g in groups:
                    r0, r1, c0, c1 = _bbox(g["payload"])
                    dr, er, dc, ec = _bbox(g["destination"])
                    target = np.zeros((er - dr + 1, ec - dc + 1), bool)
                    for r, c in g["destination"]:
                        target[r - dr, c - dc] = True
                    # A point marker can be on the payload bounding row/column;
                    # exclude it from the cropped payload tile before flipping.
                    pmask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
                    for r, c in g["payload"]:
                        pmask[r - r0, c - c0] = True
                    turned = (np.flipud(pmask) if g["kind"] == "H" else
                              np.fliplr(pmask) if g["kind"] == "V" else np.rot90(pmask, 2))
                    assert np.array_equal(turned, target)
                    silhouettes.add((pmask.shape, pmask.tobytes()))
                norm = np.full(inp.shape, 0, np.int8)
                norm[inp == marker] = 1
                norm[(inp != bg) & (inp != marker)] = 2
                structures.add(hashlib.sha256(norm.tobytes() + str(inp.shape).encode()).hexdigest())
                shapes.add(inp.shape)
                markers.add(marker)
                bgs.add(bg)
                counts.add(len(groups))
                kinds_seen.add(tuple(sorted({g["kind"] for g in groups})))
                tiny += any(len(c) < 5 for c in _components(_clear(expanded != bg), False))
                if len(groups) > 1:
                    full = [g["payload"] + g["marker"] + g["destination"] for g in groups]
                    close += any(min(max(abs(r - a), abs(c - b)) for r, c in x for a, b in y) == 2
                                 for j, x in enumerate(full) for y in full[j + 1:])
    assert pair_count == 210 and len(structures) == 210
    assert len(shapes) >= 35 and len(silhouettes) >= 35
    assert len(markers) >= 8 and len(bgs) >= 8 and len(counts) >= 5
    assert len(kinds_seen) == 7 and tiny >= 42 and close >= 5
    for nt, ns in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(27182, nt, ns)
        assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(12345), generate_puzzle(12345)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in
               zip(first[s], second[s]) for k in ("input", "output"))
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    q = bad["train"][0]
    r, c = np.argwhere(q["input"] != q["output"])[0]
    q["output"][r, c] = q["input"][r, c]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted corrupted output")
    return {"seeds": 42, "pairs": pair_count, "structures": len(structures),
            "shapes": len(shapes), "silhouettes": len(silhouettes),
            "marker_colors": len(markers), "background_colors": len(bgs),
            "group_counts": sorted(counts), "operation_mixtures": len(kinds_seen),
            "pairs_with_small_clear_components": tiny, "pairs_with_meeting_fringes": close}
