"""Registered common-core gauges. Only NumPy and the standard library are used."""
from collections import defaultdict, deque
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a, bg):
    todo = set(zip(*np.nonzero(a != bg)))
    parts = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        r0, c0 = min(r for r, c in group), min(c for r, c in group)
        h = max(r for r, c in group)-r0+1
        w = max(c for r, c in group)-c0+1
        part = np.full((h, w), bg, dtype=np.int8)
        for r, c in group:
            part[r-r0, c-c0] = a[r, c]
        parts.append(part)
    return parts


def _parse_card(part, bg):
    """Rail endpoints and patterned teeth identify the axis, without color IDs."""
    candidates = []
    for vertical in (False, True):
        a = part.T if vertical else part
        for row in np.flatnonzero(np.all(a != bg, axis=1)):
            row = int(row)
            rail = int(a[row, 0])
            if a[row, -1] != rail or rail == bg:
                continue
            if np.any(a[np.arange(a.shape[0]) != row] == rail):
                continue
            records = []
            x = 0
            while x < a.shape[1]:
                color = int(a[row, x])
                end = x+1
                while end < a.shape[1] and a[row, end] == color:
                    end += 1
                if color != rail:
                    records.append([x, end-x, color])
                x = end
            if not records:
                continue
            rebuilt = np.full_like(a, bg)
            rebuilt[row] = rail
            slots = []
            good = True
            for i, (x, width, color) in enumerate(records):
                rows = np.flatnonzero(np.any(a[:, x:x+width] != bg, axis=1))
                lo, hi = int(rows[0]), int(rows[-1])
                if not np.all(a[lo:hi+1, x:x+width] == color):
                    good = False
                    break
                stop = records[i+1][0] if i+1 < len(records) else a.shape[1]
                patch = a[:, x+width:stop]
                rr, cc = np.nonzero(patch != bg)
                # The rail is scaffolding, not part of a key.
                off = rr != row
                rr, cc = rr[off], cc[off]
                if not len(rr):
                    good = False
                    break
                top, bottom, left, right = min(rr), max(rr), min(cc), max(cc)
                motif = patch[top:bottom+1, left:right+1]
                if (top != row+1 or left < 1 or motif.size < 3
                        or np.any(motif == bg) or len(np.unique(motif)) != 2
                        or rail in motif or color in motif):
                    good = False
                    break
                kx = x+width+int(left)
                signature = _key(motif)
                rebuilt[lo:hi+1, x:x+width] = color
                rebuilt[top:bottom+1, kx:kx+motif.shape[1]] = motif
                slots.append({"x": x, "width": width, "color": color,
                              "span": (lo-row, hi-row), "key": signature,
                              "key_x": kx, "motif": motif.copy()})
            if (good and np.array_equal(a, rebuilt)
                    and len({s["key"] for s in slots}) == len(slots)):
                candidates.append({"vertical": vertical, "rail": rail,
                                   "row": row, "width": a.shape[1],
                                   "slots": slots})
    if len(candidates) != 1:
        raise ValueError("card must have one readable rail and complete patterned keys")
    return candidates[0]


def _analyze(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg]*2 <= a.size:
        raise ValueError("background must occupy more than half the input")
    cards = [_parse_card(p, bg) for p in _components(a, bg)]
    if len(cards) < 2:
        raise ValueError("need at least two separated observations")
    if len({c["vertical"] for c in cards}) != 1 or len({c["rail"] for c in cards}) != 1:
        raise ValueError("all cards must share their axis and rail role")
    occurrences = defaultdict(list)
    for i, card in enumerate(cards):
        for slot in card["slots"]:
            occurrences[slot["key"]].append((i, slot))
    if any(len(v) < 2 for v in occurrences.values()):
        raise ValueError("each gauge must be observed at least twice")
    edges = [[] for _ in cards]
    for copies in occurrences.values():
        i, s = copies[0]
        for j, t in copies[1:]:
            delta = (cards[i]["row"]-cards[j]["row"], s["key_x"]-t["key_x"])
            edges[i].append((j, delta))
            edges[j].append((i, (-delta[0], -delta[1])))
    offsets = {0: (0, 0)}
    queue = deque([0])
    while queue:
        i = queue.popleft()
        for j, (dr, dc) in edges[i]:
            proposed = offsets[i][0]+dr, offsets[i][1]+dc
            if j in offsets:
                if offsets[j] != proposed:
                    raise ValueError("registration keys disagree")
            else:
                offsets[j] = proposed
                queue.append(j)
    if len(offsets) != len(cards):
        raise ValueError("registration graph is disconnected")
    rail_row = cards[0]["row"]
    slots = []
    for copies in occurrences.values():
        placed = []
        for i, s in copies:
            dr, dc = offsets[i]
            if cards[i]["row"]+dr != rail_row:
                raise ValueError("registered rails are not collinear")
            placed.append((s["x"]+dc, s["width"], s["color"], s["key_x"]+dc))
        if len(set(placed)) != 1:
            raise ValueError("same key must describe the same fixed slot")
        x, width, color, key_x = placed[0]
        spans = [s["span"] for i, s in copies]
        core = max(lo for lo, hi in spans), min(hi for lo, hi in spans)
        if not core[0] <= 0 <= core[1]:
            raise ValueError("common interval must cross the rail")
        slots.append({"x": x, "width": width, "color": color,
                      "key_x": key_x, "motif": copies[0][1]["motif"],
                      "core": core, "observations": spans})
    slots.sort(key=lambda s: s["x"])
    if not 2 <= len(slots) <= 6:
        raise ValueError("family has two to six fixed slots")
    if len({s["color"] for s in slots}) < 2:
        raise ValueError("at least two bar colors are required")
    templates = {}
    for s in slots:
        length = s["core"][1]-s["core"][0]+1
        if templates.setdefault(length, s["core"]) != s["core"]:
            raise ValueError("equal core lengths must have identical relative intervals")
    left = min(offsets[i][1] for i in range(len(cards)))
    right = max(offsets[i][1]+c["width"]-1 for i, c in enumerate(cards))
    coverage = set()
    for i, c in enumerate(cards):
        coverage.update(range(offsets[i][1], offsets[i][1]+c["width"]))
    if coverage != set(range(left, right+1)):
        raise ValueError("registered rail must be continuous")
    for s, t in zip(slots, slots[1:]):
        if s["key_x"]+s["motif"].shape[1] >= t["x"]:
            raise ValueError("slot keys require a gap before the next gauge")
    key_colors = {int(v) for s in slots for v in s["motif"].ravel()}
    if len(key_colors) != 2 or key_colors & {s["color"] for s in slots}:
        raise ValueError("two key colors and bar colors must be disjoint")
    return {"bg": bg, "rail": cards[0]["rail"], "vertical": cards[0]["vertical"],
            "slots": slots, "left": left, "right": right,
            "cards": cards, "offsets": offsets}


def _render(info, spans):
    slots = info["slots"]
    low = min([0]+[lo for lo, hi in spans])
    high = max([s["motif"].shape[0] for s in slots]+[hi for lo, hi in spans])
    out = np.full((high-low+1, info["right"]-info["left"]+1), info["bg"], dtype=np.int8)
    out[-low] = info["rail"]
    for s, (lo, hi) in zip(slots, spans):
        x, kx = s["x"]-info["left"], s["key_x"]-info["left"]
        out[lo-low:hi-low+1, x:x+s["width"]] = s["color"]
        h, w = s["motif"].shape
        out[1-low:1-low+h, kx:kx+w] = s["motif"]
    if max(out.shape) > 30:
        raise ValueError("assembled output exceeds ARC dimensions")
    return out.T.copy() if info["vertical"] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute registrations, interval intersections, and fixed-slot length ranks."""
    info = _analyze(input_grid)
    spans = sorted((s["core"] for s in info["slots"]), key=lambda x: x[1]-x[0])
    return _render(info, spans)


def _check_pair(pair):
    """Check interval conservation and scaffold geometry directly in output cells."""
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _analyze(inp)
    a = out.T if info["vertical"] else out
    slots = info["slots"]
    cores = [s["core"] for s in slots]
    low = min([0]+[x[0] for x in cores])
    high = max([s["motif"].shape[0] for s in slots]+[x[1] for x in cores])
    assert a.shape == (high-low+1, info["right"]-info["left"]+1), "incorrect tight crop"
    rebuilt = np.full_like(a, info["bg"])
    rebuilt[-low] = info["rail"]
    remaining = cores.copy()
    previous = 0
    for s in slots:
        x, kx = s["x"]-info["left"], s["key_x"]-info["left"]
        rr = np.flatnonzero(np.any(a[:, x:x+s["width"]] != info["bg"], axis=1))
        assert len(rr), "missing output bar"
        top, bottom = int(rr[0]), int(rr[-1])
        assert np.all(a[top:bottom+1, x:x+s["width"]] == s["color"]), "slot changed color or width"
        span = top+low, bottom+low
        length = bottom-top+1
        assert span in remaining, "output must conserve complete common intervals"
        assert length >= previous, "common intervals must increase along rail"
        remaining.remove(span)
        previous = length
        rebuilt[top:bottom+1, x:x+s["width"]] = s["color"]
        h, w = s["motif"].shape
        rebuilt[1-low:1-low+h, kx:kx+w] = s["motif"]
    assert not remaining and np.array_equal(rebuilt, a), "changed key, rail, or unexplained cells"
    return info


def _pack(parts, bg, rng):
    """Random placements use a foreground halo, not positions that encode order."""
    shapes = [p.shape for p in parts]
    min_h, min_w = max(h for h, w in shapes), max(w for h, w in shapes)
    for attempt in range(35):
        if attempt < 12:
            h = int(rng.integers(max(min_h, 14), 31))
            w = int(rng.integers(max(min_w, 14), 31))
        else:
            h = w = 30
        grid = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(parts)), key=lambda i: -(parts[i].size+float(rng.random())*35))
        success = True
        for i in order:
            part = parts[i]
            ph, pw = part.shape
            mask = part != bg
            found = None
            positions = rng.permutation((h-ph+1)*(w-pw+1))
            for pos in positions:
                r, c = divmod(int(pos), w-pw+1)
                if not np.any(blocked[r:r+ph, c:c+pw] & mask):
                    found = r, c
                    break
            if found is None:
                success = False
                break
            r, c = found
            view = grid[r:r+ph, c:c+pw]
            view[mask] = part[mask]
            for rr, cc in zip(*np.nonzero(mask)):
                y, x = r+int(rr), c+int(cc)
                blocked[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] = True
        if success and np.count_nonzero(grid == bg)*2 > grid.size:
            return grid
    raise ValueError("could not pack the observation cards")


def _views(rng, n, mode):
    if mode == "windows":
        return [list(range(n-1)), list(range(1, n)), [0], [n-1]]
    if mode == "full":
        return [list(range(n)), list(range(n))]
    if mode == "chain":
        return [[i, i+1] for i in range(n-1)]+[[0], [n-1]]
    # Two differently divided covers create a connected four-card graph.
    a, b = sorted(int(v) for v in rng.choice(range(1, n), 2, replace=False))
    return [list(range(a)), list(range(a, n)), list(range(b)), list(range(b, n))]


def _make_pair(rng, config, n, vertical, witness=0, test=False):
    if config["fixed_roles"]:
        bg, rail, ka, kb = config["roles"]
    else:
        bg, rail, ka, kb = [int(x) for x in rng.choice(10, 4, replace=False)]
    accents = [c for c in range(10) if c not in (bg, rail, ka, kb)]
    colors = [int(x) for x in rng.choice(accents, n, replace=True)]
    if len(set(colors)) == 1:
        colors[-1] = next(c for c in accents if c != colors[0])
    motifs, signatures = [], set()
    for i in range(n):
        shapes = [(1, 3), (2, 2), (3, 1), (2, 3), (3, 2)] if n <= 4 else [(2, 2), (3, 1), (3, 2)]
        for _ in range(80):
            h, w = shapes[int(rng.integers(len(shapes)))]
            bits = rng.integers(0, 2, size=(h, w))
            if len(np.unique(bits)) != 2:
                continue
            motif = np.where(bits, ka, kb).astype(np.int8)
            if _key(motif) not in signatures:
                break
        else:
            raise ValueError("key sampling exhausted")
        motifs.append(motif)
        signatures.add(_key(motif))
    widths = [int(x) for x in rng.integers(1, 4, size=n)]
    gaps = [int(x) for x in rng.integers(1, 3, size=n-1)]
    key_gaps = [int(x) for x in rng.integers(1, 3, size=n)]
    if witness == 1:
        lengths = [4, 1, 3]
    elif witness == 2:
        lengths = [3, 4, 2, 3]
        widths[:2] = [3, 1]
    else:
        lengths = [int(x) for x in rng.integers(1, 7 if not test else 8, size=n)]
        rng.shuffle(lengths)
    templates = {}
    for length in set(lengths):
        lo = -(length//2)
        if length % 2 == 0:
            lo += int(rng.integers(2))
        elif length >= 3 and config["free_offsets"] and witness != 1:
            lo += int(rng.choice([-1, 0, 1]))
        templates[length] = (lo, lo+length-1)
    if witness == 1:
        templates[4] = (-2, 1)
    if witness == 2:
        # Demonstrate both even biases, or both directions of strong offsets.
        if config["free_offsets"]:
            templates[2], templates[3], templates[4] = (0, 1), (0, 2), (-3, 0)
        else:
            templates[4] = (-1, 2)
    left_margin, right_margin = [int(x) for x in rng.integers(1, 3, size=2)]
    def total_width():
        return left_margin+right_margin+sum(widths)+sum(key_gaps)+sum(m.shape[1] for m in motifs)+sum(gaps)
    while total_width() > 30:
        eligible = [i for i in range(n) if widths[i] > (3 if witness == 2 and i == 0 else 1)]
        if eligible:
            widths[int(rng.choice(eligible))] -= 1
        elif 2 in key_gaps:
            key_gaps[key_gaps.index(2)] = 1
        elif 2 in gaps:
            gaps[gaps.index(2)] = 1
        elif left_margin > 1:
            left_margin = 1
        elif right_margin > 1:
            right_margin = 1
        else:
            raise ValueError("slot layout too wide")
    x, slots = left_margin, []
    for i in range(n):
        kx = x+widths[i]+key_gaps[i]
        slots.append({"x": x, "width": widths[i], "color": colors[i],
                      "key_x": kx, "motif": motifs[i], "core": templates[lengths[i]]})
        x = kx+motifs[i].shape[1]+(gaps[i] if i < n-1 else right_margin)
    mode = "windows" if witness == 1 else "split" if witness == 2 else str(rng.choice(["windows", "full", "chain", "split"]))
    views = _views(rng, n, mode)
    if witness == 2 or (not witness and ((test and rng.random() < .7) or rng.random() < .22)):
        # Repeated complete observations are explicitly allowed and demonstrated.
        views.append(views[int(rng.integers(len(views)))].copy())
    occurrences = defaultdict(list)
    for v, indices in enumerate(views):
        for i in indices:
            occurrences[i].append(v)
    observations = {}
    for i in range(n):
        lo, hi = slots[i]["core"]
        vs = occurrences[i]
        for j, v in enumerate(vs):
            extension = int(rng.integers(0, 3))
            if witness == 1:
                extension = 4 if lengths[i] == 1 else 0
            if j == 0:
                span = lo, hi+extension
            elif j == 1:
                span = lo-extension, hi
            else:
                span = lo-int(rng.integers(3)), hi+int(rng.integers(3))
            observations[v, i] = span
    if witness == 2:
        # The extra card is identical, so it cannot alter any common interval.
        original = next(v for v in range(len(views)-1) if views[v] == views[-1])
        for i in views[-1]:
            observations[len(views)-1, i] = observations[original, i]
    parts = []
    for v, indices in enumerate(views):
        first, last = indices[0], indices[-1]
        start = 0 if first == 0 else slots[first]["x"]-1
        end = x-1 if last == n-1 else slots[last]["key_x"]+motifs[last].shape[1]
        info = {"bg": bg, "rail": rail, "vertical": False,
                "slots": [dict(slots[i], x=slots[i]["x"]-start, key_x=slots[i]["key_x"]-start) for i in indices],
                "left": 0, "right": end-start}
        parts.append(_render(info, [observations[v, i] for i in indices]))
    inp = _pack(parts, bg, rng)
    intended = {"bg": bg, "rail": rail, "vertical": vertical,
                "slots": slots, "left": 0, "right": x-1}
    out = _render(intended, sorted((s["core"] for s in slots), key=lambda z: z[1]-z[0]))
    if vertical:
        inp = inp.T.copy()
    pair = {"input": inp, "output": out}
    recovered = _check_pair(pair)
    # Compare input-derived quantities with construction, not just final pixels.
    assert [s["core"] for s in recovered["slots"]] == [s["core"] for s in slots]
    return pair


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    config = {"fixed_roles": bool(rng.random() < .5),
              "roles": [int(x) for x in rng.choice(10, 4, replace=False)],
              "free_offsets": bool(rng.random() < .65)}
    axes = str(rng.choice(["horizontal", "vertical", "mixed"]))
    train_axes = ([False, True]+[bool(rng.integers(2)) for _ in range(nt-2)] if axes == "mixed"
                  else [axes == "vertical"]*nt)
    test_axes = ([bool(rng.integers(2)) for _ in range(nq)] if axes == "mixed"
                 else [axes == "vertical"]*nq)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count, directions in (("train", nt, train_axes), ("test", nq, test_axes)):
        for i in range(count):
            witness = i+1 if split == "train" and i < 2 else 0
            n = (3 if witness == 1 else 4 if witness == 2 else 5
                 if split == "test" and i == 0 else int(rng.integers(3, 5 if split == "train" else 6)))
            for _ in range(40):
                try:
                    pair = _make_pair(rng, config, n, directions[i], witness, split == "test")
                except (ValueError, AssertionError):
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction failed")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    ss = info["slots"]
    cores = [s["core"] for s in ss]
    ordered = sorted(cores, key=lambda z: z[1]-z[0])
    union = [(min(lo for lo, hi in s["observations"]), max(hi for lo, hi in s["observations"])) for s in ss]
    minimum_lengths = [min(hi-lo+1 for lo, hi in s["observations"]) for s in ss]
    min_order = [cores[i] for i in sorted(range(len(ss)), key=lambda i: minimum_lengths[i])]
    area_order = [s["core"] for s in sorted(ss, key=lambda s: s["width"]*(s["core"][1]-s["core"][0]+1))]
    return {"count": len(ss), "cards": len(info["cards"]), "axis": info["vertical"],
            "changed": cores != ordered, "non_reversal": cores[::-1] != ordered,
            "union_differs": sorted(union, key=lambda z: z[1]-z[0]) != ordered,
            "min_rank_differs": min_order != ordered, "area_differs": area_order != ordered,
            "offset": any(lo+hi != 0 for lo, hi in cores),
            "ties": len(set(cores)) < len(cores),
            "redundancy": max(len(s["observations"]) for s in ss),
            "widths": tuple(s["width"] for s in ss),
            "lengths": tuple(hi-lo+1 for lo, hi in cores),
            "bg": info["bg"], "rail": info["rail"]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _grid(pair["output"])
            k = _key(_grid(pair["input"]))
            assert k not in seen, "duplicate input"
            seen.add(k)
            features[split].append(_features(_check_pair(pair)))
    train = features["train"]
    assert any(f["changed"] and f["non_reversal"] for f in train), "sorting witness missing"
    assert any(f["min_rank_differs"] and f["union_differs"] for f in train), "registration/intersection witness missing"
    assert any(f["area_differs"] for f in train), "length-versus-area witness missing"
    assert len({f["count"] for f in train}) >= 2, "count variation missing"
    assert max(f["redundancy"] for f in train) >= 3, "redundant observation not demonstrated"
    assert {f["axis"] for f in features["test"]} <= {f["axis"] for f in train}
    assert max(f["count"] for f in features["test"]) > max(f["count"] for f in train), "count extrapolation missing"


def self_test():
    """A bounded 42-seed sweep, relational mutations, and API boundary checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, input_shapes, output_shapes, structures = set(), set(), set(), set()
    measured = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for split in ("train", "test") for p in puzzle[split] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                info = _analyze(pair["input"])
                measured.append(_features(info))
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                roles = {}
                normalized = tuple(roles.setdefault(int(v), len(roles)) for v in pair["input"].ravel())
                structures.add((pair["input"].shape, normalized))
    assert len(signatures) == 42 and len(structures) >= 190
    assert len(input_shapes) >= 35 and len(output_shapes) >= 30
    assert {f["axis"] for f in measured} == {False, True}
    assert {f["count"] for f in measured} >= {3, 4, 5}
    assert len({f["cards"] for f in measured}) >= 4
    assert {f["bg"] for f in measured} == set(range(10))
    assert {w for f in measured for w in f["widths"]} >= {1, 2, 3}
    assert {l for f in measured for l in f["lengths"]} >= set(range(1, 8))
    assert any(f["ties"] for f in measured) and any(f["offset"] for f in measured)
    p = generate_puzzle(0)
    q = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for split in p for a, b in zip(p[split], q[split]) for k in a)
    rejected = set()
    for pair in p["train"]:
        info = _analyze(pair["input"])
        ss = info["slots"]
        candidates = {
            "omit_sort": [s["core"] for s in ss],
            "union": sorted([(min(lo for lo, hi in s["observations"]), max(hi for lo, hi in s["observations"])) for s in ss], key=lambda z: z[1]-z[0]),
            "area": [s["core"] for s in sorted(ss, key=lambda s: s["width"]*(s["core"][1]-s["core"][0]+1))],
            "reversal": [s["core"] for s in ss[::-1]],
        }
        for name, spans in candidates.items():
            bad = _render(info, spans)
            if np.array_equal(bad, pair["output"]):
                continue
            try:
                _check_pair({"input": pair["input"], "output": bad})
            except AssertionError:
                rejected.add(name)
            else:
                raise AssertionError("incorrect alternative accepted")
    assert rejected == {"omit_sort", "union", "area", "reversal"}
    # Translation/repacking and palette bijections cannot change registration.
    pair = p["train"][0]
    info = _analyze(pair["input"])
    parts = _components(pair["input"], info["bg"])
    moved = _pack(parts, info["bg"], np.random.default_rng(819))
    assert np.array_equal(solve(moved), pair["output"])
    mapping = np.array([8, 6, 4, 2, 0, 9, 7, 5, 3, 1], dtype=np.int8)
    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    # Duplicating a complete observation is idempotent under intersection.
    doubled = _pack(parts+[parts[0].copy()], info["bg"], np.random.default_rng(820))
    assert np.array_equal(solve(doubled), pair["output"])
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(771+nt, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(measured), "input_shapes": len(input_shapes),
            "output_shapes": len(output_shapes), "structures": len(structures),
            "slot_counts": sorted({f["count"] for f in measured}),
            "card_counts": sorted({f["cards"] for f in measured}),
            "mutation_checks": sorted(rejected)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
