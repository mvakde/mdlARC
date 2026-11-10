"""Switching barriers: collision exchanges a local template before routing.

Only the standard library and NumPy are used. The reader obtains direction,
colors, templates, anchors and scales from the input, not from sampling state.
"""

from math import gcd
import numpy as np


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _bounds(points):
    return (int(min(y for y, x in points)), int(min(x for y, x in points)),
            int(max(y for y, x in points)), int(max(x for y, x in points)))


def _sig(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _groups(mask):
    pending = set(zip(*np.nonzero(mask)))
    result = []
    while pending:
        p = min(pending)
        pending.remove(p)
        group, stack = {p}, [p]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = y + dy, x + dx
                    if q in pending:
                        pending.remove(q)
                        group.add(q)
                        stack.append(q)
        result.append(group)
    return result


def _primitive(crop):
    scale = 0
    for a in (crop, crop.T):
        for row in a:
            cuts = [0] + (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist() + [len(row)]
            for a, b in zip(cuts, cuts[1:]):
                scale = gcd(scale, b - a)
    reduced = crop[::scale, ::scale]
    _require(np.array_equal(crop, np.repeat(np.repeat(reduced, scale, 0), scale, 1)),
             "nonuniform pixel scale")
    return scale, reduced


def _separate(a, b):
    # Inclusive bounding rectangles, separated by at least one background line.
    ay, ax, by, bx = a
    cy, cx, dy, dx = b
    return by + 1 < cy or dy + 1 < ay or bx + 1 < cx or dx + 1 < ax


def _rect(templates, obj):
    y0, x0, y1, x1 = _bounds(set(templates[0]) | set(templates[1]))
    y, x, s = obj["y"], obj["x"], obj["scale"]
    return y + y0*s, x + x0*s, y + (y1+1)*s-1, x + (x1+1)*s-1


def _beam(template):
    xs = sorted(x for y, x in template if y == 0)
    _require(xs and xs == list(range(xs[0], xs[-1]+1)), "leading beam is not contiguous")
    _require(min(y for y, x in template) == 0, "missing leading beam")
    _require(all(xs[0] <= x <= xs[-1] for y, x in template), "material outside beam span")
    return xs[0], xs[-1]


def _span(template, obj):
    lo, hi = _beam(template)
    return obj["x"] + lo*obj["scale"], obj["x"] + (hi+1)*obj["scale"]-1


def _render(grid, template, obj):
    s = obj["scale"]
    for (y, x), c in template.items():
        yy, xx = obj["y"] + y*s, obj["x"] + x*s
        grid[yy:yy+s, xx:xx+s] = c


def _infer_objects(a, bg, guide):
    objects = []
    for cells in _groups((a != bg) & (a != guide)):
        y0, x0, y1, x1 = _bounds(cells)
        crop = a[y0:y1+1, x0:x1+1]
        _require(np.count_nonzero((crop != bg) & (crop != guide)) == len(cells),
                 "interleaved motifs")
        _require(not np.any(crop == guide), "guide intersects motif")
        s, small = _primitive(crop)
        mapping = {(int(y), int(x)): int(small[y, x])
                   for y, x in zip(*np.nonzero(small != bg))}
        objects.append({"box": (y0, x0), "scale": s, "signature": _sig(mapping)})
    signatures = sorted({o["signature"] for o in objects})
    _require(len(signatures) == 2, "exactly two visible normalized states required")
    reference = min(o["scale"] for o in objects)
    _require({o["signature"] for o in objects if o["scale"] == reference} == set(signatures),
             "both states need a common reference scale")
    maps = [{(y, x): c for y, x, c in sig} for sig in signatures]
    shifts = set()
    for c in set(maps[0].values()) & set(maps[1].values()):
        p = {q for q, color in maps[0].items() if color == c}
        q = {q for q, color in maps[1].items() if color == c}
        py, px, _, _ = _bounds(p)
        qy, qx, _, _ = _bounds(q)
        shift = py-qy, px-qx
        if p == {(y+shift[0], x+shift[1]) for y, x in q}:
            shifts.add(shift)
    alignments = []
    for dy, dx in sorted(shifts):
        shifted = {(y+dy, x+dx): c for (y, x), c in maps[1].items()}
        common = {p for p, c in maps[0].items() if shifted.get(p) == c}
        if len(common) >= 2:
            alignments.append((len(common), (dy, dx), shifted, common))
    _require(alignments, "no shared colored scaffold")
    score = max(item[0] for item in alignments)
    best = [item for item in alignments if item[0] == score]
    _require(len(best) == 1, "tied scaffold registration")
    _, (dy, dx), aligned, common = best[0]
    oy, ox, _, _ = _bounds(common)
    templates = [{(y-oy, x-ox): c for (y, x), c in m.items()} for m in (maps[0], aligned)]
    beams = [_beam(t) for t in templates]
    _require(beams[0] != beams[1], "exchange must change beam endpoints")
    _require(max(beams[0][0], beams[1][0]) <= min(beams[0][1], beams[1][1]),
             "beam states must overlap")
    placements, rectangles = [], []
    for o in objects:
        state = signatures.index(o["signature"])
        s = o["scale"]
        sy, sx = (dy, dx) if state else (0, 0)
        obj = {"y": o["box"][0] + (oy-sy)*s,
               "x": o["box"][1] + (ox-sx)*s, "scale": s, "state": state}
        rect = _rect(templates, obj)
        y0, x0, y1, x1 = rect
        _require(1 <= y0 <= y1 < a.shape[0] and 0 <= x0 <= x1 < a.shape[1],
                 "replacement clipped or on entry boundary")
        _require(all(_separate(rect, other) for other in rectangles), "overlapping replacement regions")
        _require(not np.any(a[y0:y1+1, x0:x1+1] == guide), "stub crosses replacement region")
        rectangles.append(rect)
        placements.append(obj)
    reconstructed = np.full(a.shape, bg, dtype=int)
    reconstructed[a == guide] = guide
    for obj in placements:
        _render(reconstructed, templates[obj["state"]], obj)
    _require(np.array_equal(reconstructed, a), "motif reconstruction failed")
    return templates, placements, reference


def _read(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg]*2 > grid.size, "background must occupy a strict majority")
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation).copy()
        h, w = a.shape
        for guide in map(int, np.unique(a)):
            if guide == bg:
                continue
            entries = np.flatnonzero(a[0] == guide).tolist()
            if not entries or any(b-a < 2 for a, b in zip(entries, entries[1:])):
                continue
            mask = np.zeros(a.shape, bool)
            lengths = []
            for c in entries:
                rows = np.flatnonzero(a[:, c] == guide)
                if not (2 <= len(rows) < h and np.array_equal(rows, np.arange(len(rows)))):
                    break
                lengths.append(len(rows))
                mask[:len(rows), c] = True
            else:
                if not np.array_equal(mask, a == guide):
                    continue
                try:
                    templates, objects, reference = _infer_objects(a, bg, guide)
                except AssertionError:
                    continue
                candidates.append({"grid": a, "background": bg, "guide": guide,
                                   "rotation": rotation, "entries": entries, "stub_lengths": lengths,
                                   "templates": templates, "objects": objects, "reference_scale": reference})
    _require(len(candidates) == 1, "entry color/direction or templates are ambiguous")
    return candidates[0]


def _run(info, shortcut=None):
    """Front-to-back event simulation. Geometry is exchanged BEFORE routing.

    All incoming rays striking one gate are a single event. For simultaneous
    gates, triggers use the same incoming-layer snapshot. New material absorbs
    any additional incoming ray in its span. Coincident outgoing rays merge.
    """
    a, bg, guide = info["grid"], info["background"], info["guide"]
    templates, objects = info["templates"], info["objects"]
    out = a.copy()
    h, w = a.shape
    active = set(info["entries"])
    by_row = {}
    for index, obj in enumerate(objects):
        by_row.setdefault(obj["y"], []).append(index)
    reached, events, incoming = set(), [], {}
    simultaneous = multi_hit = mergers = clipped = interrupted = newly_absorbed = 0
    for r in range(1, h):
        incoming[r] = sorted(active)
        triggers = []
        for index in by_row.get(r, []):
            obj = objects[index]
            old = _span(templates[obj["state"]], obj)
            hits = sorted(c for c in active if old[0] <= c <= old[1])
            if hits:
                triggers.append((index, obj, old, hits))
        simultaneous += len(triggers) > 1
        absorbed, emitted = set(), set()
        for index, obj, old, hits in triggers:
            reached.add(index)
            opposite = 1-obj["state"]
            if shortcut == "no_toggle" or (shortcut == "per_ray" and len(hits) % 2 == 0):
                opposite = obj["state"]
            new = _span(templates[opposite], obj)
            routing = old if shortcut == "old_ends" else new
            caught = {c for c in active if min(old[0], routing[0]) <= c <= max(old[1], routing[1])}
            if shortcut == "old_absorb":
                caught = set(hits)
            absorbed.update(caught)
            y0, x0, y1, x1 = _rect(templates, obj)
            out[y0:y1+1, x0:x1+1] = bg
            _render(out, templates[opposite], obj)
            flanks = [routing[0]-1, routing[1]+1]
            start, stop = min(flanks + list(caught)), max(flanks + list(caught))
            cross = out[r-1, max(0, start):min(w, stop+1)]
            interrupted += bool(np.any((cross != bg) & (cross != guide)))
            cross[cross == bg] = guide
            for c in flanks:
                if 0 <= c < w:
                    mergers += c in emitted or (c in active and c not in caught)
                    emitted.add(c)
                else:
                    clipped += 1
            multi_hit += len(hits) > 1
            newly_absorbed += len(caught) - len(hits)
            events.append({"object": index, "row": r, "old": old, "new": new,
                           "hits": hits, "caught": sorted(caught), "flanks": flanks})
        active = (active - absorbed) | emitted
        for c in sorted(active):
            if out[r, c] == bg:
                out[r, c] = guide
            elif out[r, c] != guide and shortcut is None:
                raise AssertionError("ray met material below an unhandled leading beam")
    if shortcut == "toggle_all":
        for index, obj in enumerate(objects):
            if index not in reached:
                y0, x0, y1, x1 = _rect(templates, obj)
                out[y0:y1+1, x0:x1+1] = bg
                _render(out, templates[1-obj["state"]], obj)
    stats = {"reached": sorted(reached), "events": events, "incoming": incoming,
             "simultaneous": int(simultaneous), "multi_hit": int(multi_hit),
             "mergers": int(mergers), "clipped": int(clipped), "interrupted": int(interrupted),
             "newly_absorbed": int(newly_absorbed),
             "layers": len({e["row"] for e in events})}
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    out, _ = _run(info)
    return np.rot90(out, -info["rotation"]).copy()


def _design(rng):
    # Both beams contain the same core; state zero reaches farther left and
    # state one farther right. Lengths, proportions and lower structure vary.
    while True:
        width, height = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        left, right = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        if width+left+right <= 8:
            break
    short_left, short_right = int(rng.integers(left)), int(rng.integers(right))
    stem = int(rng.integers(width))
    core = {(0, x) for x in range(width)} | {(y, stem) for y in range(1, height)}
    for y in range(1, height):
        if rng.random() < .6:
            lo = int(rng.integers(stem+1))
            hi = int(rng.integers(stem, width))
            core.update((y, x) for x in range(lo, hi+1))
    common = {p: 1 for p in core}
    interior = sorted(p for p in core if p[0] > 0)
    rng.shuffle(interior)
    changing = interior.pop() if len(interior) >= 2 and rng.random() < .65 else None
    if interior and rng.random() < .6:
        common[interior[0]] = 2
    templates = []
    for state, (l, r) in enumerate(((left, short_right), (short_left, right))):
        t = dict(common)
        accent = 3 if state == 0 else 5
        if changing is not None:
            t[changing] = accent
        xs = list(range(-l, 0)) + list(range(width, width+r))
        for x in xs:
            t[0, x] = accent
            # Unequal hanging decorations remain under the state's own beam.
            if rng.random() < .45:
                for y in range(1, int(rng.integers(1, height)) + 1):
                    t[y, x] = accent + int(rng.random() < .45)
        templates.append(t)
    palette = rng.permutation(10).tolist()
    colored = [{p: int(palette[c]) for p, c in t.items()} for t in templates]
    return colored, int(palette[0]), int(palette[9]), width


def _known_info(shape, templates, objects, entries, lengths, bg, guide):
    a = np.full(shape, bg, dtype=int)
    for obj in objects:
        _render(a, templates[obj["state"]], obj)
    for c, length in zip(entries, lengths):
        a[:length, c] = guide
    return {"grid": a, "templates": templates, "objects": objects,
            "entries": entries, "background": bg, "guide": guide}


def _place(rng, templates, objects, width, row, state, scale, targets=None, avoid=()):
    candidates = []
    for x in range(-30, width+30):
        obj = {"y": row, "x": x, "scale": scale, "state": state}
        rect = _rect(templates, obj)
        if not (rect[0] >= 3 and rect[2] < 30 and 0 <= rect[1] <= rect[3] < width):
            continue
        if not all(_separate(rect, _rect(templates, other)) for other in objects):
            continue
        lo, hi = _span(templates[state], obj)
        if targets is not None and not any(lo <= c <= hi for c in targets):
            continue
        if any(lo <= c <= hi for c in avoid):
            continue
        candidates.append(obj)
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _make_pair(rng, kind, shift, seen):
    # Attempts sample fresh procedural geometry, not stored examples. All
    # budgets are bounded. Mandatory witnesses are checked from returned pixels.
    for attempt in range(160):
        templates, bg, guide, core_width = _design(rng)
        union_y0, union_x0, union_y1, union_x1 = _bounds(set(templates[0]) | set(templates[1]))
        uw, uh = union_x1-union_x0+1, union_y1-union_y0+1
        fork = kind == "fork" or (kind == "test" and rng.random() < .45)
        scale = 2 if kind == "fork" else 1
        if kind == "test":
            scale = 3 if shift == "scale" else int(rng.integers(1, 3))
        if fork and (scale+1)*uw+1 > 30:
            fork = False
        minimum_width = (scale+1)*uw+1 if fork else max(scale*uw, 2*uw+2, 14)
        if minimum_width > 30:
            continue
        width = int(rng.integers(minimum_width, 31))
        row = int(rng.integers(3, 6))
        objects, entries = [], []
        if fork:
            gap = 1 if rng.random() < .7 else 2
            if (scale+1)*uw+gap > width:
                gap = 1
            left = int(rng.integers(width-(scale+1)*uw-gap+1))
            first = {"y": row, "x": left-union_x0*scale, "scale": scale, "state": 0}
            second = {"y": row, "x": left+uw*scale+gap-union_x0, "scale": 1, "state": 1}
            objects.extend([first, second])
            # One gate is hit twice; the other is hit concurrently. With gap
            # one the replacement beams' facing flanks are exactly coincident.
            entries = [first["x"], first["x"]+max(2, scale), second["x"]]
            # A ray beside the old beam is covered by its replacement. Show
            # this explicitly instead of introducing widened capture in test.
            entries.append(_span(templates[0], first)[1]+1)
        else:
            first = _place(rng, templates, objects, width, row, 0, scale)
            if first is None:
                continue
            objects.append(first)
            entries = [first["x"]+int(rng.integers(core_width*scale))]
        entries = sorted(set(entries))
        if any(b-a < 2 for a, b in zip(entries, entries[1:])):
            continue
        lengths = [int(rng.integers(2, min(4, row)+1)) for _ in entries]
        if any(length > row for length in lengths):
            continue
        next_row = row+uh*scale+int(rng.integers(1, 3))
        levels = 3 if kind == "test" and shift == "depth" else 2
        success = True
        for level in range(1, levels):
            if next_row+uh > 30:
                success = False
                break
            partial = _known_info((30, width), templates, objects, entries, lengths, bg, guide)
            _, stats = _run(partial)
            _, old_stats = _run(partial, "old_ends")
            targets = stats["incoming"][next_row]
            # The simple demonstration explicitly puts a gate on a route that
            # the old endpoints do not reach. That makes feedback observable.
            avoid = old_stats["incoming"][next_row] if kind == "chain" and level == 1 else ()
            state = 0 if fork and level == 1 else level % 2
            obj = _place(rng, templates, objects, width, next_row, state, 1, targets, avoid)
            if obj is None:
                success = False
                break
            objects.append(obj)
            next_row += uh+int(rng.integers(1, 4))
        if not success:
            continue
        # Add at least one genuinely unreached motif, while supplying a unit
        # reference of any state absent at scale one. Its role is in the same
        # scene, rather than in a separate key or calibration panel.
        wanted = {0, 1} - {o["state"] for o in objects if o["scale"] == 1}
        inert_states = sorted(wanted) or [int(rng.integers(2))]
        inert_states += [int(rng.integers(2)) for _ in range(int(rng.integers(0, 4)))]
        for state in inert_states:
            partial = _known_info((30, width), templates, objects, entries, lengths, bg, guide)
            _, stats = _run(partial)
            slots = list(range(3, 31-uh))
            rng.shuffle(slots)
            chosen = None
            for r in slots:
                obj = _place(rng, templates, objects, width, r, state, 1,
                             avoid=stats["incoming"].get(r, []))
                if obj is not None:
                    # A flank from a same-layer event may reach this candidate
                    # only below its leading bar; reject any such interference.
                    trial = _known_info((30, width), templates, objects+[obj], entries, lengths, bg, guide)
                    try:
                        _, ts = _run(trial)
                    except AssertionError:
                        continue
                    if len(objects) not in ts["reached"]:
                        chosen = obj
                        break
            if chosen is None:
                if state in wanted or len(objects) <= levels+int(fork):
                    success = False
                break
            objects.append(chosen)
        if not success:
            continue
        lowest = max(_rect(templates, o)[2] for o in objects)+1
        height = int(rng.integers(max(lowest, min(30, next_row-1)), 31))
        info = _known_info((height, width), templates, objects, entries, lengths, bg, guide)
        try:
            out, stats = _run(info)
        except AssertionError:
            continue
        if np.count_nonzero(info["grid"] == bg)*2 <= info["grid"].size:
            continue
        rotation = int(rng.integers(4))
        inp, out = np.rot90(info["grid"], rotation).copy(), np.rot90(out, rotation).copy()
        if rng.random() < .5:
            inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
        if _key(inp) in seen:
            continue
        pair = {"input": inp, "output": out}
        try:
            parsed = _read(inp)
            predicted, actual = _run(parsed)
            _require(np.array_equal(np.rot90(predicted, -parsed["rotation"]), out), "roundtrip failed")
            _require(len(actual["reached"]) < len(parsed["objects"]), "missing inert motif")
            reached_states = {parsed["objects"][i]["state"] for i in actual["reached"]}
            _require(reached_states == {0, 1}, "both exchange directions must be reached")
            if kind == "chain":
                _, wrong = _run(parsed, "old_ends")
                _require(set(wrong["reached"]) != set(actual["reached"]), "feedback witness is redundant")
            if kind == "fork":
                _require(actual["simultaneous"] and actual["multi_hit"], "missing simultaneous/multi-hit witness")
                _require(not np.array_equal(_run(parsed, "per_ray")[0], predicted), "multi-hit witness redundant")
                _require(actual["newly_absorbed"] and
                         not np.array_equal(_run(parsed, "old_absorb")[0], predicted),
                         "widening-absorption witness redundant")
            if kind == "test" and shift == "depth":
                _require(actual["layers"] >= 3, "missing deeper cascade")
        except AssertionError:
            continue
        return pair
    raise AssertionError(f"bounded scene construction exhausted for {kind}/{shift}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = "scale" if rng.random() < .5 else "depth"
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            kind = "test" if split == "test" else "chain" if index % 2 == 0 else "fork"
            pair = _make_pair(rng, kind, shift, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "output dimensions changed")
    info = _read(a)
    expected, stats = _run(info)
    canonical_out = np.rot90(b, info["rotation"])
    _require(np.array_equal(expected, canonical_out), "incorrect exchange or routing")
    _require(not np.array_equal(a, b), "unchanged example")
    _require(np.all(canonical_out[info["grid"] == info["guide"]] == info["guide"]), "stub changed")
    # Check whole replacement regions independently of the trace: foreground
    # equals the registered selected template; all other region cells are only
    # background or signal. Unreached material is unchanged; a trace may pass
    # through currently empty cells of its reserved opposite-state footprint.
    scaffold = {p: c for p, c in info["templates"][0].items()
                if info["templates"][1].get(p) == c}
    for index, obj in enumerate(info["objects"]):
        final_state = 1-obj["state"] if index in stats["reached"] else obj["state"]
        material = np.full(expected.shape, info["background"], dtype=int)
        _render(material, info["templates"][final_state], obj)
        y0, x0, y1, x1 = _rect(info["templates"], obj)
        region = canonical_out[y0:y1+1, x0:x1+1]
        m = material[y0:y1+1, x0:x1+1]
        _require(np.all(region[m != info["background"]] == m[m != info["background"]]), "wrong scaled material")
        _require(np.all(np.isin(region[m == info["background"]], [info["background"], info["guide"]])), "unerased decoration")
        for (y, x), c in scaffold.items():
            yy, xx, s = obj["y"]+y*obj["scale"], obj["x"]+x*obj["scale"], obj["scale"]
            _require(np.all(canonical_out[yy:yy+s, xx:xx+s] == c), "scaffold moved")
        if index not in stats["reached"]:
            original = info["grid"][y0:y1+1, x0:x1+1]
            _require(np.array_equal(region[original != info["background"]],
                                    original[original != info["background"]]), "inert motif changed")
    return info, stats


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal example counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info, stats = _analyze(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                training.append((info, stats))
    _require(any(len(s["reached"]) < len(i["objects"]) for i, s in training), "no reachability demonstration")
    _require(any({i["objects"][j]["state"] for j in s["reached"]} == {0, 1}
                 for i, s in training), "one swap direction missing")
    _require(any(set(_run(i, "old_ends")[1]["reached"]) != set(s["reached"])
                 for i, s in training), "no state-dependent downstream witness")
    _require(any(s["simultaneous"] and s["multi_hit"] and
                 not np.array_equal(_run(i, "per_ray")[0], _run(i)[0])
                 for i, s in training), "no simultaneous/grouped-activation witness")
    _require(any(any(i["objects"][j]["scale"] > 1 for j in s["reached"])
                 for i, s in training), "no enlarged activated motif in training")
    for shortcut in ("no_toggle", "old_ends", "toggle_all", "per_ray", "old_absorb"):
        _require(any(not np.array_equal(_run(i, shortcut)[0], _run(i)[0]) for i, s in training),
                 f"training fails to distinguish {shortcut}")


def self_test():
    """Bounded 42-seed semantic invariants, diversity and negative checks."""
    # A hand-enumerated feedback regression. This expectation neither invokes
    # the event simulator nor copies generated labels.
    a = np.zeros((12, 15), dtype=int)
    a[:2, 4] = 9
    a[3, 2:6] = [2, 2, 1, 1]
    a[4, 4] = 1
    a[7, 8:12] = [1, 1, 3, 3]
    a[8, 8] = 1
    a[3, 9:13] = [2, 2, 1, 1]
    a[4, 11] = 1
    expected = a.copy()
    expected[3, 2:8] = [0, 0, 1, 1, 3, 3]
    expected[7, 6:12] = [2, 2, 1, 1, 0, 0]
    expected[2, 3:9] = 9
    expected[3:6, [3, 8]] = 9
    expected[6, 3] = 9
    expected[6, 5:11] = 9
    for r in range(7, 12):
        for c in (3, 5, 10):
            expected[r, c] = 9
    _require(np.array_equal(solve(a), expected), "manual geometric-feedback regression")
    # Larger-than-sampled scales, including an edge-clipped new exit. Unit
    # references occur below the enlarged gate; their occupied cells stay put.
    templates = [{(0, 0): 1, (0, 1): 1, (1, 0): 1, (0, -2): 2, (0, -1): 2},
                 {(0, 0): 1, (0, 1): 1, (1, 0): 1, (0, 2): 3, (0, 3): 3}]
    for scale in (4, 5):
        objects = [{"y": 3, "x": 2*scale, "scale": scale, "state": 0},
                   {"y": 16, "x": 5, "scale": 1, "state": 0},
                   {"y": 16, "x": 16, "scale": 1, "state": 1}]
        known = _known_info((30, 30), templates, objects, [2*scale], [2], 0, 9)
        wanted = known["grid"].copy()
        wanted[3:3+2*scale, :6*scale] = 0
        _render(wanted, templates[1], objects[0])
        wanted[2, 2*scale-1:min(30, 6*scale+1)] = 9
        for r in range(3, 30):
            for c in (2*scale-1, 6*scale):
                if c < 30:
                    wanted[r, c] = 9
        _require(np.array_equal(solve(known["grid"]), wanted), "large-scale boundary regression")
    # Missing second state and a present state whose opposite would be clipped.
    missing = a.copy()
    missing[7:9, 6:12] = 0
    missing[7, 6:10] = [2, 2, 1, 1]
    missing[8, 8] = 1
    clipped_input = a.copy()
    clipped_input[7:9, 6:12] = 0
    clipped_input[7, 1:5] = [1, 1, 3, 3]
    clipped_input[8, 1] = 1
    for malformed in (missing, clipped_input):
        try:
            solve(malformed)
        except AssertionError:
            pass
        else:
            raise AssertionError("malformed motif convention accepted")
    shapes, palettes, designs, counts, schedules, fingerprints = set(), set(), set(), set(), set(), set()
    directions, layers, offsets, beam_sizes, backgrounds = set(), set(), set(), set(), set()
    scales, clipped, merges, interrupted, shifted_routes, widening = set(), 0, 0, 0, 0, 0
    within_design = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = tuple(_key(p[r]) for split in puzzle for p in puzzle[split] for r in ("input", "output"))
        fingerprints.add(fingerprint)
        local_designs = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info, stats = _analyze(pair)
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "input-only solve mismatch")
                _require(np.array_equal(before, pair["input"]), "solve mutated input")
                shapes.add(pair["input"].shape)
                palettes.add(tuple(sorted(map(int, np.unique(pair["input"])))))
                # Color-normalize the aligned two-state vocabulary.
                roles = {}
                design = tuple(tuple((y, x, roles.setdefault(c, len(roles))) for y, x, c in _sig(t))
                               for t in info["templates"])
                designs.add(design)
                local_designs.add(design)
                counts.add(len(info["objects"]))
                schedule = tuple(sorted(o["scale"] for o in info["objects"]))
                schedules.add(schedule)
                scales.update(schedule)
                directions.add(info["rotation"])
                backgrounds.add(info["background"])
                layers.add(stats["layers"])
                clipped += stats["clipped"] > 0
                merges += stats["mergers"] > 0
                interrupted += stats["interrupted"] > 0
                widening += stats["newly_absorbed"] > 0
                shifted_routes += set(_run(info, "old_ends")[1]["reached"]) != set(stats["reached"])
                for e in stats["events"]:
                    offsets.update(c-e["old"][0] for c in e["hits"])
                    beam_sizes.update((e["old"][1]-e["old"][0]+1, e["new"][1]-e["new"][0]+1))
        within_design += len(local_designs) > 1
    _require(len(fingerprints) == 42, "puzzles collapsed")
    _require(len(shapes) >= 35 and len(designs) >= 90 and len(palettes) >= 35, "structural/palette diversity collapsed")
    _require(within_design == 42, "pair-local template variation collapsed")
    _require(len(counts) >= 4 and len(schedules) >= 8, "count/scale schedules collapsed")
    _require(scales == {1, 2, 3} and directions == set(range(4)), "scale/orientation coverage missing")
    _require(len(backgrounds) == 10 and len(beam_sizes) >= 10 and len(offsets) >= 6, "role/beam/hit diversity missing")
    _require(clipped > 0 and merges >= 20 and shifted_routes >= 42 and widening >= 42,
             "interaction coverage missing")
    _require({2, 3} <= layers, "cascade-depth shift missing")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(903, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "count request ignored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(a[r], b[r]) for s in p for a, b in zip(p[s], q[s])
                 for r in ("input", "output")), "nondeterminism")
    # Reject a changed scaffold pixel, and a label produced by sequential old-
    # geometry routing. These are semantic failures rather than schema errors.
    info = _read(p["train"][0]["input"])
    wrong, _ = _run(info, "old_ends")
    bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in p[s]] for s in p}
    bad["train"][0]["output"] = np.rot90(wrong, -info["rotation"]).copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("sequential shortcut label accepted")
    bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in p[s]] for s in p}
    output = bad["test"][0]["output"]
    output[0, 0] = (int(output[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "template_vocabularies": len(designs),
            "palette_sets": len(palettes), "motif_counts": sorted(counts),
            "scale_schedules": len(schedules), "beam_widths": sorted(beam_sizes),
            "clipped_pairs": int(clipped), "merged_pairs": int(merges),
            "interrupted_pairs": int(interrupted), "state_dependent_reachability_pairs": int(shifted_routes),
            "widening_capture_pairs": int(widening)}
