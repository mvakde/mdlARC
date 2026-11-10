"""Complementary broadcast valves. Standard library and NumPy only.

Generation renders labels from a construction plan. The input-only reader
discovers frame/ink roles, reconstructs components, and infers both pipe states.
No parent code, seed, generated example, or output label is used by solve.
"""
from __future__ import annotations

import heapq
import numpy as np

D = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid cells")
    return a.astype(int, copy=True)


def _components(grid):
    ids = np.full(grid.shape, -1, dtype=int)
    groups, colors = [], []
    for p in np.ndindex(grid.shape):
        if ids[p] >= 0:
            continue
        i, c = len(groups), int(grid[p])
        ids[p] = i
        stack, group = [p], set()
        while stack:
            r, x = stack.pop()
            group.add((r, x))
            for dr, dx in D:
                q = r + dr, x + dx
                if _inside(q, grid.shape) and ids[q] < 0 and grid[q] == c:
                    ids[q] = i
                    stack.append(q)
        groups.append(group)
        colors.append(c)
    return ids, colors, groups


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _bounds(points):
    return min(r for r, c in points), max(r for r, c in points) + 1, min(c for r, c in points), max(c for r, c in points) + 1


def _frame(h, w):
    return {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}


def _port(h, w, direction, length, offset, ink, accent):
    dr, dc = D[direction]
    if direction == 0:
        center, root = (-1 - length, offset), (-1, offset)
    elif direction == 1:
        center, root = (offset, w + length), (offset, w)
    elif direction == 2:
        center, root = (h + length, offset), (h, offset)
    else:
        center, root = (offset, -1 - length), (offset, -1)
    pr, pc = -dc, dr
    points = {center: ink, (center[0] + dr, center[1] + dc): ink,
              (center[0] + pr, center[1] + pc): ink, (center[0] - pr, center[1] - pc): ink}
    for k in range(length):
        points[root[0] + k * dr, root[1] + k * dc] = accent
    return dict(points=points, direction=direction, center=center, length=length, offset=offset)


def _stamp(grid, mapping, anchor, scale):
    for (r, c), value in mapping.items():
        y, x = anchor[0] + r * scale, anchor[1] + c * scale
        _need(0 <= y and y + scale <= grid.shape[0] and 0 <= x and x + scale <= grid.shape[1], "clipped valve")
        grid[y:y + scale, x:x + scale] = value


def _socket(grid, anchor, scale, h, w, value):
    y, x = anchor
    grid[y + scale:y + (h - 1) * scale, x + scale:x + (w - 1) * scale] = value


def _ray(ids, anchor, scale, port, source):
    r, c = port["center"]
    dr, dc = D[port["direction"]]
    # An enlarged nozzle has a scale-wide beam, rather than an arbitrarily
    # chosen middle pixel. Every parallel ray must agree. This convention is
    # invariant under reflection even for even-sized pixel blocks.
    recipients = set()
    for lane in range(scale):
        y = anchor[0] + r * scale + (lane if dc else 0)
        x = anchor[1] + c * scale + (lane if dr else 0)
        if dr == 1:
            y = anchor[0] + (r + 2) * scale
        elif dr == -1:
            y = anchor[0] + (r - 1) * scale - 1
        elif dc == 1:
            x = anchor[1] + (c + 2) * scale
        else:
            x = anchor[1] + (c - 1) * scale - 1
        target = None
        while _inside((y, x), ids.shape):
            if ids[y, x] != source:
                target = int(ids[y, x])
                break
            y, x = y + dr, x + dc
        recipients.add(target)
    _need(len(recipients) == 1, "beam has ambiguous recipients")
    return recipients.pop()


def _render(clean, ids, occurrences, ports, carrier, h, w):
    assignments, edges = {}, []
    for o in occurrences:
        new = ports[1 - o["state"]]
        target = _ray(ids, o["anchor"], o["scale"], new, o["source"])
        edges.append((o["source"], target, o["payload"]))
        if target is not None:
            _need(target not in assignments or assignments[target] == o["payload"], "conflicting broadcasts")
            assignments[target] = o["payload"]
    out = clean.copy()
    for target, value in assignments.items():
        out[ids == target] = value
    for o in occurrences:
        anchor, s = o["anchor"], o["scale"]
        host = int(out[anchor])
        _stamp(out, {p: carrier for p in _frame(h, w)}, anchor, s)
        _socket(out, anchor, s, h, w, o["payload"] if o["explicit"] else host)
        _stamp(out, ports[1 - o["state"]]["points"], anchor, s)
    return out, assignments, edges


def _read_with_roles(grid, carrier, ink, groups, colors):
    frames = []
    for cells, color in zip(groups, colors):
        if color != carrier:
            continue
        y0, y1, x0, x1 = _bounds(cells)
        hh, ww = y1 - y0, x1 - x0
        patch = grid[y0:y1, x0:x1]
        s = next((k for k in range(1, min(hh, ww)) if patch[k, ww // 2] != carrier), 0)
        _need(s and hh % s == 0 and ww % s == 0, "bad frame thickness")
        h, w = hh // s, ww // s
        _need(3 <= h <= 5 and 3 <= w <= 5, "bad logical frame")
        logical_mask = np.array([[r in (0, h - 1) or c in (0, w - 1) for c in range(w)] for r in range(h)])
        mask = np.repeat(np.repeat(logical_mask, s, 0), s, 1)
        _need(np.array_equal(patch == carrier, mask), "not a rectangular frame")
        corners = [(y0 - 1, x0 - 1), (y0 - 1, x1), (y1, x0 - 1), (y1, x1)]
        _need(all(_inside(p, grid.shape) for p in corners), "missing frame apron")
        host = int(grid[corners[0]])
        _need(all(grid[p] == host for p in corners) and host not in (carrier, ink), "ambiguous host")
        chamber = patch[s:-s, s:-s]
        _need(np.unique(chamber).size == 1, "socket is not uniform")
        payload = int(chamber[0, 0])
        _need(payload not in (carrier, ink), "socket uses annotation ink")
        frames.append(dict(anchor=(y0, x0), scale=s, host=host, payload=payload,
                           explicit=payload != host, h=h, w=w))
    _need(len(frames) >= 2 and len({(o["h"], o["w"]) for o in frames}) == 1, "inconsistent frames")
    h, w = frames[0]["h"], frames[0]["w"]
    ink_cells = {p for cells, c in zip(groups, colors) if c == ink for p in cells}
    used, templates = set(), []
    for o in frames:
        y, x = o["anchor"]
        s = o["scale"]
        candidates = []
        for direction in range(4):
            for length in range(3):
                for offset in range(1, (w if direction % 2 == 0 else h) - 1):
                    port = _port(h, w, direction, length, offset, ink, ink)
                    logical_ink = [p for p, v in port["points"].items()]
                    # The first four points are the T; later points are the stem.
                    t = logical_ink[:4]
                    pixels = {(y + r * s + i, x + c * s + j) for r, c in t for i in range(s) for j in range(s)}
                    if not pixels <= ink_cells:
                        continue
                    mapping = {p: ink for p in t}
                    stem_colors = set()
                    valid = True
                    for r, c in logical_ink[4:]:
                        yy, xx = y + r * s, x + c * s
                        if not (0 <= yy and yy + s <= grid.shape[0] and 0 <= xx and xx + s <= grid.shape[1]):
                            valid = False
                            break
                        block = grid[yy:yy + s, xx:xx + s]
                        if np.unique(block).size != 1:
                            valid = False
                            break
                        value = int(block[0, 0])
                        stem_colors.add(value)
                        mapping[r, c] = value
                    if valid and (not length or (len(stem_colors) == 1 and not stem_colors & {o["host"], o["payload"], ink, carrier})):
                        port["points"] = mapping
                        candidates.append((port, pixels))
        _need(len(candidates) == 1, "no unique nozzle for frame")
        port, pixels = candidates[0]
        _need(not used & pixels, "shared nozzle")
        used.update(pixels)
        signature = tuple(sorted(port["points"].items()))
        if signature not in [t[0] for t in templates]:
            templates.append((signature, port))
        o["state"] = [t[0] for t in templates].index(signature)
    _need(used == ink_cells and len(templates) == 2, "exactly two explained pipe states required")
    ports = [t[1] for t in templates]
    _need(ports[0]["direction"] != ports[1]["direction"], "states must route in different directions")
    minimum = min(o["scale"] for o in frames)
    _need({o["state"] for o in frames if o["scale"] == minimum} == {0, 1}, "both states need a common reference scale")
    annotation_colors = {carrier, ink} | {v for p in ports for v in p["points"].values()}
    _need(all(o["host"] not in annotation_colors and o["payload"] not in annotation_colors for o in frames), "pigment and annotations must contrast")
    clean, occupied = grid.copy(), set()
    for o in frames:
        a, s = o["anchor"], o["scale"]
        support = _frame(h, w) | set(ports[o["state"]]["points"]) | {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
        for r, c in support:
            pixels = {(a[0] + r * s + i, a[1] + c * s + j) for i in range(s) for j in range(s)}
            _need(not pixels & occupied, "overlapping valves")
            occupied.update(pixels)
        _stamp(clean, {p: o["host"] for p in support}, a, s)
    _need(not set(np.unique(clean)) & annotation_colors, "unexplained annotation colors")
    ids, fills, territories = _components(clean)
    reserved = []
    for o in frames:
        o["source"] = int(ids[o["anchor"]])
        # Reserve the union rectangle, plus a one-logical-pixel apron, in one
        # original component. This checks unseen replacement cells as well.
        union = _frame(h, w) | set(ports[0]["points"]) | set(ports[1]["points"])
        r0, r1, c0, c1 = _bounds(union)
        y, x = o["anchor"]
        s = o["scale"]
        box = (y + (r0 - 1) * s, y + (r1 + 1) * s, x + (c0 - 1) * s, x + (c1 + 1) * s)
        _need(0 <= box[0] < box[1] <= grid.shape[0] and 0 <= box[2] < box[3] <= grid.shape[1], "union apron clipped")
        _need(np.all(ids[box[0]:box[1], box[2]:box[3]] == o["source"]), "replacement leaves original host")
        _need(all(box[1] <= b[0] or b[1] <= box[0] or box[3] <= b[2] or b[3] <= box[2] for b in reserved), "overlapping replacement reserves")
        reserved.append(box)
    out, assignments, edges = _render(clean, ids, frames, ports, carrier, h, w)
    old_targets = [_ray(ids, o["anchor"], o["scale"], ports[o["state"]], o["source"]) for o in frames]
    final = [assignments.get(i, c) for i, c in enumerate(fills)]
    old_assignments, old_valid = {}, True
    for o, target in zip(frames, old_targets):
        if target is not None:
            if target in old_assignments and old_assignments[target] != o["payload"]:
                old_valid = False
            old_assignments[target] = o["payload"]
    old_final = [old_assignments.get(i, c) for i, c in enumerate(fills)]
    return out, dict(clean=clean, ids=ids, fills=fills, territories=territories,
                     occurrences=frames, ports=ports, carrier=carrier, ink=ink, h=h, w=w,
                     assignments=assignments, edges=edges,
                     escape=any(t is None for _, t, _ in edges),
                     routing=any(t != e[1] for t, e in zip(old_targets, edges)),
                     old_routing_differs=old_valid and final != old_final,
                     broadcast_change=final != fills,
                     multiple_incoming=len([e for e in edges if e[1] is not None]) > len(assignments),
                     explicit=any(o["explicit"] and e[1] is not None and o["payload"] != o["host"] for o, e in zip(frames, edges)),
                     simultaneous=any(not o["explicit"] and o["source"] in assignments and assignments[o["source"]] != o["host"] and e[1] is not None for o, e in zip(frames, edges)),
                     locality=any(fills[i] == fills[j] and final[i] != final[j] for i in range(len(fills)) for j in range(i)))


def _read(input_grid):
    grid = _grid(input_grid)
    _, colors, groups = _components(grid)
    answers = []
    # A frame color has only hollow rectangular components; T ink has only
    # four-logical-pixel components. Cheap geometric filtering precedes decoding.
    possible_frames, possible_inks = [], []
    for color in sorted(set(colors)):
        cells = [g for g, c in zip(groups, colors) if c == color]
        if len(cells) < 2:
            continue
        boxes = [_bounds(g) for g in cells]
        if all((b[1] - b[0]) >= 3 and (b[3] - b[2]) >= 3 and len(g) < (b[1] - b[0]) * (b[3] - b[2]) for g, b in zip(cells, boxes)):
            possible_frames.append(color)
        if all(len(g) == 4 * int(round((len(g) / 4) ** .5)) ** 2 for g in cells):
            possible_inks.append(color)
    for carrier in possible_frames:
        for ink in possible_inks:
            if carrier == ink:
                continue
            try:
                answers.append(_read_with_roles(grid, carrier, ink, groups, colors))
            except (ValueError, IndexError):
                continue
    _need(len(answers) == 1, "input has no unique valve interpretation")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _read(input_grid)[0].copy()


def _pack(rng, sizes):
    options = []
    for width in range(max(w for h, w in sizes), 31):
        for order in (sorted(range(len(sizes)), key=lambda i: sizes[i][0], reverse=True), list(rng.permutation(len(sizes)))):
            rects = [None] * len(sizes)
            y = x = shelf = 0
            for i in order:
                h, w = sizes[i]
                if x and x + w > width:
                    y, x, shelf = y + shelf + 1, 0, 0
                rects[i] = y, x, h, w
                x += w + 1
                shelf = max(shelf, h)
            if max(y + h for y, x, h, w in rects) <= 30:
                options.append(rects)
    if not options:
        return None
    rects = options[int(rng.integers(len(options)))]
    h0, w0 = max(y + h for y, x, h, w in rects), max(x + w for y, x, h, w in rects)
    shape = int(rng.integers(h0, min(30, h0 + 6) + 1)), int(rng.integers(w0, min(30, w0 + 6) + 1))
    for _ in range(30):
        i = int(rng.integers(len(rects)))
        _, _, hh, ww = rects[i]
        candidate = int(rng.integers(shape[0] - hh + 1)), int(rng.integers(shape[1] - ww + 1)), hh, ww
        y, x, hh, ww = candidate
        if all(j == i or y + hh < yy or yy + h < y or x + ww < xx or xx + w < x for j, (yy, xx, h, w) in enumerate(rects)):
            rects[i] = candidate
    return shape, rects


def _grow(rng, shape, rects, extra):
    labels = np.full(shape, -1, dtype=int)
    for i, (y, x, h, w) in enumerate(rects):
        labels[y:y + h, x:x + w] = i
    free = list(zip(*np.nonzero(labels < 0)))
    if len(free) < extra:
        return None
    for j in range(extra):
        p = free.pop(int(rng.integers(len(free))))
        labels[p] = len(rects) + j
    queue, serial = [], 0
    for p in np.ndindex(shape):
        if labels[p] >= 0:
            for dr, dc in D:
                q = p[0] + dr, p[1] + dc
                if _inside(q, shape) and labels[q] < 0:
                    heapq.heappush(queue, (float(rng.uniform(.5, 2)), serial, q, int(labels[p])))
                    serial += 1
    while queue:
        distance, _, p, label = heapq.heappop(queue)
        if labels[p] >= 0:
            continue
        labels[p] = label
        for dr, dc in D:
            q = p[0] + dr, p[1] + dc
            if _inside(q, shape) and labels[q] < 0:
                heapq.heappush(queue, (distance + float(rng.uniform(.5, 2)), serial, q, label))
                serial += 1
    return labels


def _pair(rng, kind, allow_escape, allow_locality, fallback=False):
    palette = [int(c) for c in rng.permutation(10)]
    carrier, ink, a0, a1 = palette[:4]
    h, w = (3, 3) if fallback else (int(rng.integers(3, 5)), int(rng.integers(3, 5)))
    directions = [1, 3] if fallback else [int(v) for v in rng.choice(4, 2, replace=False)]
    ports = [_port(h, w, d, 0 if fallback else int(rng.choice([0, 0, 1, 2])), int(rng.integers(1, (w if d % 2 == 0 else h) - 1)), ink, a) for d, a in zip(directions, (a0, a1))]
    union = _frame(h, w) | set(ports[0]["points"]) | set(ports[1]["points"])
    r0, r1, c0, c1 = _bounds(union)
    bh, bw = r1 - r0 + 2, c1 - c0 + 2
    if fallback:
        # Two unequal bands, two common-scale references and one enlargement.
        # Opposite horizontal states create a guaranteed simultaneous swap.
        split = int(rng.integers(18, 22))
        shape = int(rng.integers(17, 31)), split + 9
        labels = np.ones(shape, dtype=int)
        labels[:, :split] = 0
        rects = [(0, split, 5, 9), (11, 0, 5, 9), (0, 0, 10, 18)]
        scales, states, sources = [1, 1, 2], [0, 1, 1], [1, 0, 0]
        count = 2
    else:
        n = int(rng.integers(3, 5 if kind == "implicit" else 6))
        scales = [1, 1] + [1] * (n - 2)
        if kind in ("implicit", "test") or rng.random() < .5:
            scales[2] = 3 if kind == "test" and rng.random() < .35 else 2
        packed = _pack(rng, [(bh * s, bw * s) for s in scales])
        if packed is None:
            return None
        shape, rects = packed
        extra = int(rng.integers(0, min(2, 5 - n) + 1))
        labels = _grow(rng, shape, rects, extra)
        if labels is None:
            return None
        states, sources = [0, 1] + [int(rng.integers(2)) for _ in range(n - 2)], list(range(n))
        count = n + extra
    fills = palette[4:4 + count]
    if allow_locality and not fallback and rng.random() < .65:
        neighbors = set()
        for left, right in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
            neighbors.update(tuple(sorted((int(a), int(b)))) for a, b in zip(left[left != right], right[left != right]))
        pairs = [(a, b) for a in range(count) for b in range(a) if (b, a) not in neighbors]
        if pairs:
            a, b = pairs[int(rng.integers(len(pairs)))]
            fills[a] = fills[b]
    clean = np.asarray(fills, dtype=int)[labels]
    occurrences = []
    for (y, x, _, _), s, state, source in zip(rects, scales, states, sources):
        occurrences.append(dict(anchor=(y + (1 - r0) * s, x + (1 - c0) * s), scale=s, state=state, source=source,
                                host=fills[source], payload=fills[source], explicit=False))
    if kind != "implicit":
        try:
            eligible = [i for i, o in enumerate(occurrences) if _ray(labels, o["anchor"], o["scale"], ports[1 - o["state"]], o["source"]) is not None]
        except ValueError:
            return None
        if not eligible:
            return None
        i = eligible[int(rng.integers(len(eligible)))]
        chosen = occurrences[i]
        choices = [c for c in palette[4:] if c != chosen["host"] and (kind != "explicit" or c not in fills)]
        if not choices:
            return None
        payload = int(rng.choice(choices))
        chosen.update(payload=payload, explicit=True)
        if fallback:
            # Duplicate source broadcasts must still agree.
            for o in occurrences:
                if o["source"] == chosen["source"]:
                    o.update(payload=payload, explicit=True)
    inp = clean.copy()
    for o in occurrences:
        _stamp(inp, {p: carrier for p in _frame(h, w)}, o["anchor"], o["scale"])
        _socket(inp, o["anchor"], o["scale"], h, w, o["payload"])
        _stamp(inp, ports[o["state"]]["points"], o["anchor"], o["scale"])
    try:
        out, _, _ = _render(clean, labels, occurrences, ports, carrier, h, w)
        predicted, info = _read(inp)
    except ValueError:
        return None
    if not np.array_equal(out, predicted) or not info["routing"] or not info["broadcast_change"] or (kind == "implicit" and not (info["simultaneous"] and info["old_routing_differs"])) or (kind == "explicit" and not info["explicit"]) or (not allow_escape and info["escape"]):
        return None
    if not allow_locality and len(info["fills"]) != len(set(info["fills"])):
        return None
    # Whole-scene dihedral changes preserve the interpretation; templates remain
    # pair-local. The decoder also accepts these fresh nozzle orientations.
    turn, reflect = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, turn), np.rot90(out, turn)
    if reflect:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return dict(input=inp.copy(), output=out.copy())


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("require integer counts 2..10 / 1..4 with num_train > num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            kind = ("implicit" if i == 0 else "explicit" if i == 1 else str(rng.choice(["implicit", "explicit"]))) if split == "train" else "test"
            taught = [_read(p["input"])[1] for p in puzzle["train"]]
            escape = split == "train" or any(t["escape"] for t in taught)
            locality = split == "train" or any(t["locality"] for t in taught)
            for attempt in range(90):
                pair = _pair(rng, kind, escape, locality, fallback=attempt >= 70)
                if pair is not None:
                    key = _key(pair["input"])
                    if key not in seen:
                        seen.add(key)
                        puzzle[split].append(pair)
                        break
            else:
                raise RuntimeError("bounded valve construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal counts")
    infos, seen = {"train": [], "test": []}, set()
    for split in puzzle:
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape and _key(inp) not in seen, "shape mismatch or duplicate input")
            seen.add(_key(inp))
            predicted, info = _read(inp)
            _need(np.array_equal(predicted, out), "wrong routing, state exchange, socket, or fill")
            _need(np.array_equal(inp == info["carrier"], out == info["carrier"]), "scaffold moved")
            _need(info["routing"] and not np.array_equal(inp, out), "missing state/routing contrast")
            _need(info["broadcast_change"], "all generated broadcasts are color no-ops")
            # Check untouched territory pixels outside old and new valve ink.
            for source, target, payload in info["edges"]:
                if target is not None:
                    _need(info["assignments"][target] == payload, "inconsistent recipient")
            infos[split].append(info)
    train = infos["train"]
    _need(any(t["simultaneous"] for t in train), "training omits original-color dependencies")
    _need(any(t["old_routing_differs"] and t["broadcast_change"] for t in train), "training fails to distinguish omission of A or B")
    _need(any(t["explicit"] for t in train), "training omits pigment override")
    _need(any(len({o["scale"] for o in t["occurrences"]}) > 1 for t in train), "training omits enlargement")
    for t in infos["test"]:
        _need(not t["escape"] or any(a["escape"] for a in train), "untaught escaping ray")
        _need(len(set(t["fills"])) == len(t["fills"]) or any(a["locality"] for a in train), "untaught repeated-fill locality")


def self_test():
    shapes, partitions, designs, schedules, counts, carriers, fingerprints = set(), set(), set(), set(), set(), set(), set()
    features = {k: 0 for k in ("explicit", "simultaneous", "escape", "locality")}
    directions, stem_lengths, sizes = set(), set(), set()
    incoming_case = None
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprint = tuple(_key(pair[r]) for s in p for pair in p[s] for r in ("input", "output"))
        fingerprints.add(fingerprint)
        if seed < 2:
            q = generate_puzzle(seed)
            _need(fingerprint == tuple(_key(pair[r]) for s in q for pair in q[s] for r in ("input", "output")), "nondeterminism")
        for split in p:
            for pair in p[split]:
                out, info = _read(pair["input"])
                _need(np.array_equal(out, pair["output"]) and np.array_equal(solve(pair["input"].copy()), out), "input-only solve mismatch")
                shapes.add(out.shape)
                partitions.add(_key(info["ids"]))
                schedules.add(tuple(sorted(o["scale"] for o in info["occurrences"])))
                counts.add(len(info["occurrences"]))
                carriers.add(info["carrier"])
                sizes.add((info["h"], info["w"]))
                design = tuple(sorted((port["direction"], port["length"], port["offset"]) for port in info["ports"]))
                designs.add((info["h"], info["w"], design))
                directions.update(port["direction"] for port in info["ports"])
                stem_lengths.update(port["length"] for port in info["ports"])
                for k in features:
                    features[k] += bool(info[k])
                if incoming_case is None and info["multiple_incoming"]:
                    incoming_case = pair["input"].copy(), info
    _need(len(fingerprints) == 42 and len(partitions) >= 100 and len(shapes) >= 25, "insufficient structural variation")
    _need(len(designs) >= 25 and len(schedules) >= 5 and len(counts) >= 3 and len(sizes) >= 3, "collapsed valves/scales/counts")
    _need(directions == set(range(4)) and stem_lengths == {0, 1, 2} and len(carriers) == 10, "missing palette/geometry variation")
    _need(all(features[k] >= 3 for k in features), "missing mechanism coverage")
    _need(incoming_case is not None, "no consistent repeated broadcast witness")
    bad, info = incoming_case
    target = next(t for _, t, _ in info["edges"] if t is not None and sum(e[1] == t for e in info["edges"]) > 1)
    index = next(i for i, edge in enumerate(info["edges"]) if edge[1] == target)
    o = info["occurrences"][index]
    forbidden = {info["carrier"], info["ink"], o["payload"]} | {v for port in info["ports"] for v in port["points"].values()}
    _socket(bad, o["anchor"], o["scale"], info["h"], info["w"], min(set(range(10)) - forbidden))
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting incoming pigments accepted")
    # Exercise the bounded fallback even when ordinary sampling never needs it.
    for index, kind in enumerate(("implicit", "explicit", "test")):
        pair = _pair(np.random.default_rng(31000 + index), kind, True, True, fallback=True)
        _need(pair is not None and np.array_equal(solve(pair["input"]), pair["output"]), "fallback violates the family")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(60000 + 10 * nt + ns, nt, ns)
            _need(len(p["train"]) == nt and len(p["test"]) == ns, "ignored explicit counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{r: pair[r].copy() for r in pair} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, shapes=len(shapes), partitions=len(partitions), designs=len(designs), scale_schedules=len(schedules), valve_counts=sorted(counts), mechanisms=features)
