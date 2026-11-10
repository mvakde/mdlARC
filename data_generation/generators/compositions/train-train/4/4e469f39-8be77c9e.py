"""Folded delivery: notched receivers fetch and mirror-double nearby packets.

Only NumPy and the standard library are used. All semantic information is in
the input. In particular, neither the frame ink nor the payload is seed-coded.
"""

import json
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.tobytes()


def _basic(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _ring(r, c, h, w):
    return ({(r, x) for x in range(c, c + w)}
            | {(r + h - 1, x) for x in range(c, c + w)}
            | {(y, c) for y in range(r, r + h)}
            | {(y, c + w - 1) for y in range(r, r + h)})


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for rr, cc in np.argwhere(mask):
        p = int(rr), int(cc)
        if seen[p]:
            continue
        seen[p] = True
        stack, cells = [p], set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = r + dr, c + dc
                if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                        and mask[q] and not seen[q]):
                    seen[q] = True
                    stack.append(q)
        result.append(cells)
    return result


def _separate(a, b):
    r, c, h, w = a["box"]
    y, x, hh, ww = b["box"]
    return r + h < y or y + hh < r or c + w < x or x + ww < c


def _opening(box, gap):
    r, c, h, w = box
    y, x = gap
    if y == r and c < x < c + w - 1:
        side, offset, length = "top", x - c, w
        start, step = (y - 1, x), (0, 1 if offset == 1 else -1)
        normal, tangent = h - 2, w - 2
    elif y == r + h - 1 and c < x < c + w - 1:
        side, offset, length = "bottom", x - c, w
        start, step = (y + 1, x), (0, 1 if offset == 1 else -1)
        normal, tangent = h - 2, w - 2
    elif x == c and r < y < r + h - 1:
        side, offset, length = "left", y - r, h
        start, step = (y, x - 1), (1 if offset == 1 else -1, 0)
        normal, tangent = w - 2, h - 2
    elif x == c + w - 1 and r < y < r + h - 1:
        side, offset, length = "right", y - r, h
        start, step = (y, x + 1), (1 if offset == 1 else -1, 0)
        normal, tangent = w - 2, h - 2
    else:
        raise ValueError("gap must be on a side, not a corner")
    if length < 4 or offset not in (1, length - 2) or normal < 2 or normal % 2 or tangent < 2:
        raise ValueError("receiver must have a doubled cavity and an off-center notch")
    return side, start, step


def _inside(p, box):
    r, c, h, w = box
    return r <= p[0] < r + h and c <= p[1] < c + w


def _connections(grid, ink, objects):
    """Trace real input rays; frame bounds are also obstacles to empty pixels."""
    owner = np.full(grid.shape, -1, dtype=int)
    for i, obj in enumerate(objects):
        r, c, h, w = obj["box"]
        owner[r:r + h, c:c + w] = i
    links = []
    for obj in objects:
        if obj["kind"] != "receiver":
            continue
        side, p, step = _opening(obj["box"], obj["gap"])
        path = []
        while 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]:
            index = int(owner[p])
            if index >= 0:
                source = objects[index]
                if source["kind"] != "source" or int(grid[p]) != ink:
                    raise ValueError("a delivery ray must first hit a closed packet")
                break
            if grid[p] != 0:
                raise ValueError("delivery path is not empty")
            path.append(p)
            p = p[0] + step[0], p[1] + step[1]
        else:
            raise ValueError("delivery ray reaches the boundary without a packet")
        if not path:
            raise ValueError("delivery needs at least one exterior cell")
        r, c, h, w = source["box"]
        payload = grid[r + 1:r + h - 1, c + 1:c + w - 1]
        _, _, rh, rw = obj["box"]
        expected = ((2 * payload.shape[0], payload.shape[1])
                    if side in ("top", "bottom")
                    else (payload.shape[0], 2 * payload.shape[1]))
        if expected != (rh - 2, rw - 2):
            raise ValueError("packet does not fit the receiver after reflection")
        links.append({"receiver": obj, "source": source, "side": side,
                      "path": path, "hit": p, "payload": payload.copy()})
    return links


def _parse_as(grid, ink):
    objects = []
    for cells in _components(grid == ink):
        ys, xs = zip(*cells)
        r, c = min(ys), min(xs)
        h, w = max(ys) - r + 1, max(xs) - c + 1
        if min(h, w) < 3:
            raise ValueError("not a rectangular frame")
        border = _ring(r, c, h, w)
        if not cells <= border:
            raise ValueError("frame ink occurs inside a packet")
        missing = border - cells
        obj = {"box": (r, c, h, w)}
        if not missing:
            obj["kind"] = "source"
        elif len(missing) == 1:
            obj.update(kind="receiver", gap=next(iter(missing)))
            _opening(obj["box"], obj["gap"])
            if np.any(grid[r + 1:r + h - 1, c + 1:c + w - 1] != 0):
                raise ValueError("receiver cavity must be empty")
        else:
            raise ValueError("not a closed packet or a singly notched receiver")
        objects.append(obj)
    if not any(o["kind"] == "source" for o in objects) or not any(o["kind"] == "receiver" for o in objects):
        raise ValueError("need both kinds of frame")
    allowed = np.zeros(grid.shape, dtype=bool)
    for i, obj in enumerate(objects):
        if any(not _separate(obj, other) for other in objects[:i]):
            raise ValueError("frames need a background cell between their boxes")
        r, c, h, w = obj["box"]
        allowed[r:r + h, c:c + w] = True
    if np.any(grid[~allowed] != 0):
        raise ValueError("unframed marks")
    return objects, _connections(grid, ink, objects)


def _parse(grid):
    candidates = []
    for ink in (int(x) for x in np.unique(grid) if x != 0):
        try:
            objects, links = _parse_as(grid, ink)
        except ValueError:
            continue
        candidates.append((ink, objects, links))
    if len(candidates) != 1:
        raise ValueError("input must have exactly one frame-ink interpretation")
    return candidates[0]


def _doubled(payload, side):
    if side == "top":
        return np.concatenate((payload, payload[::-1]), axis=0)
    if side == "bottom":
        return np.concatenate((payload[::-1], payload), axis=0)
    if side == "left":
        return np.concatenate((payload, payload[:, ::-1]), axis=1)
    return np.concatenate((payload[:, ::-1], payload), axis=1)


def _deliver(grid, ink, links):
    out = grid.copy()
    for link in links:
        obj = link["receiver"]
        r, c, h, w = obj["box"]
        out[r + 1:r + h - 1, c + 1:c + w - 1] = _doubled(link["payload"], link["side"])
        out[obj["gap"]] = ink
        for p in link["path"]:
            out[p] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _basic(input_grid)
    ink, _, links = _parse(grid)
    return _deliver(grid, ink, links)


def _paint(grid, ink, obj, payload=None):
    r, c, h, w = obj["box"]
    for p in _ring(r, c, h, w):
        grid[p] = ink
    if obj["kind"] == "source":
        grid[r + 1:r + h - 1, c + 1:c + w - 1] = payload
    else:
        grid[obj["gap"]] = 0


def _pattern(rng, h, w, ink, informative=False):
    pool = [x for x in range(10) if x != ink]
    nonzero = [x for x in pool if x != 0]
    colors = [0] + [int(x) for x in rng.choice(nonzero, size=int(rng.integers(1, 5)), replace=False)]
    a, b = colors[1], colors[-1]
    style = int(rng.integers(8))
    out = np.zeros((h, w), dtype=np.int8)
    if style == 0:
        out[:] = rng.choice(colors, size=(h, w))
    elif style == 1:
        out[int(rng.integers(h)):, int(rng.integers(w)):] = a
    elif style == 2:
        out[:] = rng.choice(colors, size=(h, 1))
    elif style == 3:
        out[:] = rng.choice(colors, size=(1, w))
    elif style == 4:
        out.flat[int(rng.integers(out.size))] = a
    elif style == 5:
        out[:] = a
        out.flat[int(rng.integers(out.size))] = b if b != a else 0
    elif style == 6:
        out[:] = rng.choice(colors, size=(h, w))
        out[(h + 1) // 2:] = out[:h // 2][::-1]
    else:
        out[:] = int(rng.choice(colors))
    if informative:
        # A visible zero margin and two unequal corner values distinguish row
        # reversal, duplication, rotation, and cropping to nonzero cells.
        assert h >= 2 and w >= 2
        out[:] = 0
        out[0, 0], out[-1, 0] = a, b
        if a == b:
            out[-1, 0] = next(x for x in nonzero if x != a)
    return out


def _contrast(rng):
    """One scene, two notch positions; a farther same-ray decoy is present."""
    ink = int(rng.integers(1, 10))
    ph, pw = int(rng.integers(2, 4)), 2
    fw, dh, rh = pw + 2, ph + 2, 2 * ph + 2
    spaces = [int(rng.integers(1, 3)) for _ in range(3)]
    margin_l, margin_r = int(rng.integers(3)), int(rng.integers(3))
    xs = [margin_l]
    for distance in spaces:
        xs.append(xs[-1] + fw + distance)
    width = xs[-1] + fw + margin_r
    ray = dh - 1 + int(rng.integers(2))
    height = ray + 1 + rh + int(rng.integers(3))
    grid = np.zeros((height, width), dtype=np.int8)
    objects = []
    payloads = []
    for index, x in enumerate((xs[0], xs[2], xs[3])):
        top = int(rng.integers(max(0, ray - dh + 1), ray + 1))
        # Receiver is horizontally separate; donors may meet the ray at any
        # border level, including their first or last row.
        obj = {"kind": "source", "box": (top, x, dh, fw)}
        for _ in range(64):
            payload = _pattern(rng, ph, pw, ink, informative=True)
            if index == 1:
                # The right-hand delivery also occupies the other column, so
                # copying a colored column alone cannot fit both examples.
                extra = [v for v in range(1, 10)
                         if v not in (ink, int(payload[0, 0]), int(payload[-1, 0]))]
                payload[0, -1] = int(rng.choice(extra))
            if not any(np.array_equal(payload, old) for old in payloads):
                break
        else:
            # At most two previous pictures exclude fewer than the 56 ordered
            # two-color margin patterns; make even this retry loop bounded.
            for a in range(1, 10):
                for b in range(1, 10):
                    if a == b or ink in (a, b):
                        continue
                    candidate = np.zeros((ph, pw), dtype=np.int8)
                    candidate[0, 0], candidate[-1, 0] = a, b
                    if not any(np.array_equal(candidate, old) for old in payloads):
                        payload = candidate
                        break
                else:
                    continue
                break
        objects.append(obj)
        payloads.append(payload)
        _paint(grid, ink, obj, payload)
    pairs = []
    for sign in (1, -1):
        inp = grid.copy()
        receiver = {"kind": "receiver", "box": (ray + 1, xs[1], rh, fw),
                    "gap": (ray + 1, xs[1] + (1 if sign == 1 else pw))}
        _paint(inp, ink, receiver)
        links = _connections(inp, ink, objects + [receiver])
        pairs.append({"input": inp, "output": _deliver(inp, ink, links)})
    return pairs


def _free_scene(rng, count, is_test, used):
    """Scatter packets first, then place receivers using actual first-hit rays.

    No bands or per-receiver panels are reserved. Packets can be unused or
    shared; receivers may align or have overlapping vertical spans.
    """
    for attempt in range(100):
        height = int(rng.integers(19 if count >= 3 else 13, 31))
        width = int(rng.integers(21 if count >= 3 else 15, 31))
        if attempt > 60:
            height = width = 30
        ink = int(rng.integers(1, 10))
        grid = np.zeros((height, width), dtype=np.int8)
        objects = []
        source_count = int(rng.integers(2, 7))
        for _ in range(source_count):
            ph = int(rng.integers(1, 5 if is_test else 4))
            pw = int(rng.integers(2, 7 if is_test else 6))
            for _ in range(70):
                r = int(rng.integers(max(1, height - 2 * ph - 3)))
                c = int(rng.integers(width - pw - 1))
                obj = {"kind": "source", "box": (r, c, ph + 2, pw + 2)}
                if all(_separate(obj, old) for old in objects):
                    _paint(grid, ink, obj, _pattern(rng, ph, pw, ink))
                    objects.append(obj)
                    break
        if len(objects) < 2:
            continue
        source_objects = list(objects)
        paths = set()
        for _ in range(count):
            candidates = []
            # Sample source and placement order rather than selecting a fixed
            # earliest position, which would collapse layout variety.
            for si in rng.permutation(len(source_objects)):
                source = source_objects[int(si)]
                sr, sc, sh, sw = source["box"]
                rh, rw = 2 * (sh - 2) + 2, sw
                ray_rows = list(range(sr, sr + sh))
                rng.shuffle(ray_rows)
                for ray in ray_rows:
                    top = ray + 1
                    if top + rh > height:
                        continue
                    for left0 in rng.permutation(width - rw + 1):
                        left = int(left0)
                        box = top, left, rh, rw
                        obj = {"kind": "receiver", "box": box}
                        if not all(_separate(obj, old) for old in objects):
                            continue
                        if any(_inside(p, box) for p in paths):
                            continue
                        for sign in (1, -1):
                            obj2 = dict(obj, gap=(top, left + (1 if sign == 1 else rw - 2)))
                            # Tracing a new receiver requires no repainting:
                            # its exterior path is tested against all boxes.
                            try:
                                link = _connections(grid, ink, source_objects +
                                                    [o for o in objects if o["kind"] == "receiver"] + [obj2])[-1]
                            except ValueError:
                                continue
                            candidates.append((obj2, link))
                    if candidates:
                        break
                if candidates:
                    break
            if not candidates:
                break
            obj, link = candidates[int(rng.integers(len(candidates)))]
            objects.append(obj)
            _paint(grid, ink, obj)
            paths.update(link["path"])
        if sum(o["kind"] == "receiver" for o in objects) != count:
            continue
        key = _key(grid)
        if key in used:
            continue
        links = _connections(grid, ink, objects)
        return {"input": grid, "output": _deliver(grid, ink, links)}
    raise RuntimeError("bounded scene placement exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    train = _contrast(rng)
    used = {_key(p["input"]) for p in train}
    for _ in range(nt - 2):
        pair = _free_scene(rng, int(rng.integers(1, 3)), False, used)
        used.add(_key(pair["input"]))
        train.append(pair)
    max_train_count = max(len(_parse(p["input"])[2]) for p in train)
    test = []
    for _ in range(ne):
        # Every test has one more receiver than any training example. Packet
        # size/motif, source sharing, and geometry still follow the same rule.
        pair = _free_scene(rng, max_train_count + 1, True, used)
        used.add(_key(pair["input"]))
        test.append(pair)
    rng.shuffle(train)
    puzzle = {"train": train, "test": test}
    for split in puzzle.values():
        for pair in split:
            for role in pair:
                pair[role] = np.rot90(pair[role], rotation).copy()
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(inp, out):
    """Check pixel provenance independently of the concatenation renderer."""
    ink, objects, links = _parse(inp)
    assert inp.shape == out.shape
    changed_allowed = np.zeros(inp.shape, dtype=bool)
    for obj in objects:
        r, c, h, w = obj["box"]
        if obj["kind"] == "source":
            assert np.array_equal(inp[r:r + h, c:c + w], out[r:r + h, c:c + w])
        for p in _ring(r, c, h, w):
            if p != obj.get("gap"):
                assert out[p] == ink
    a_witness = b_witness = crop_witness = nearest_witness = False
    for link in links:
        receiver, payload, side = link["receiver"], link["payload"], link["side"]
        r, c, h, w = receiver["box"]
        dh, dw = payload.shape
        changed_allowed[r + 1:r + h - 1, c + 1:c + w - 1] = True
        changed_allowed[receiver["gap"]] = True
        assert out[receiver["gap"]] == ink
        for p in link["path"]:
            changed_allowed[p] = True
            assert inp[p] == 0 and out[p] == ink
        for y in range(h - 2):
            for x in range(w - 2):
                if side == "top":
                    sy, sx = (y if y < dh else 2 * dh - 1 - y), x
                elif side == "bottom":
                    sy, sx = (dh - 1 - y if y < dh else y - dh), x
                elif side == "left":
                    sy, sx = y, (x if x < dw else 2 * dw - 1 - x)
                else:
                    sy, sx = y, (dw - 1 - x if x < dw else x - dw)
                assert out[r + 1 + y, c + 1 + x] == payload[sy, sx]
        axis = 0 if side in ("top", "bottom") else 1
        duplicate = np.concatenate((payload, payload), axis=axis)
        actual = out[r + 1:r + h - 1, c + 1:c + w - 1]
        b_witness |= not np.array_equal(duplicate, actual)
        crop_witness |= bool(np.any(np.all(payload == 0, axis=axis)))
        # Reverse just the notch: when another compatible packet is reached,
        # compare full interiors, not merely the visible connector.
        opposite = inp.copy()
        opposite[receiver["gap"]] = ink
        gy, gx = receiver["gap"]
        ng = ((gy, 2 * c + w - 1 - gx) if axis == 0
              else (2 * r + h - 1 - gy, gx))
        opposite[ng] = 0
        try:
            _, _, reversed_links = _parse(opposite)
            other = next(v for v in reversed_links if v["receiver"]["box"] == receiver["box"])
            a_witness |= not np.array_equal(payload, other["payload"])
        except ValueError:
            pass
        # Are there additional packet borders beyond the chosen one? They
        # cannot affect the answer, even if their contents differ.
        _, _, step = _opening(receiver["box"], receiver["gap"])
        p = link["hit"]
        while 0 <= p[0] < inp.shape[0] and 0 <= p[1] < inp.shape[1]:
            for obj in objects:
                if obj["kind"] == "source" and obj is not link["source"] and _inside(p, obj["box"]):
                    sr, sc, sh, sw = obj["box"]
                    alt = inp[sr + 1:sr + sh - 1, sc + 1:sc + sw - 1]
                    nearest_witness |= alt.shape == payload.shape and not np.array_equal(alt, payload)
            p = p[0] + step[0], p[1] + step[1]
    assert np.array_equal(out[~changed_allowed], inp[~changed_allowed])
    assert np.array_equal(solve(inp), out)
    assert not np.array_equal(inp, out)
    return ink, objects, links, (a_witness, b_witness, crop_witness, nearest_witness)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_counts, test_counts, sides = set(), set(), set(), set()
    witnesses = np.zeros(4, dtype=bool)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _basic(pair["input"]), _basic(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            _, _, links, flags = _check_pair(inp, out)
            sides.update(v["side"] for v in links)
            if split == "train":
                train_counts.add(len(links))
                witnesses |= flags
            else:
                test_counts.add(len(links))
    assert len(sides) == 1  # generation policy; solve also accepts mixed sides
    assert min(test_counts) > max(train_counts)
    assert all(witnesses), "training must expose notch selection, reflection, margins, and nearest hit"


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    # Hand-built mixed-side scene, outside the sampler's single-side schedule.
    # Explicit expected pixels also check that the bottom original half is not
    # accidentally rotated. Rotation covariance covers both horizontal sides.
    mixed = np.zeros((12, 26), dtype=np.int8)
    s1 = {"kind": "source", "box": (1, 0, 4, 4)}
    s2 = {"kind": "source", "box": (7, 22, 4, 4)}
    r1 = {"kind": "receiver", "box": (3, 6, 6, 4), "gap": (3, 8)}
    r2 = {"kind": "receiver", "box": (2, 16, 6, 4), "gap": (7, 17)}
    _paint(mixed, 5, s1, np.array([[2, 0], [3, 4]], dtype=np.int8))
    _paint(mixed, 5, s2, np.array([[6, 7], [0, 8]], dtype=np.int8))
    _paint(mixed, 5, r1)
    _paint(mixed, 5, r2)
    expected = mixed.copy()
    expected[4:8, 7:9] = [[2, 0], [3, 4], [3, 4], [2, 0]]
    expected[3:7, 17:19] = [[0, 8], [6, 7], [6, 7], [0, 8]]
    expected[2, 4:9] = 5
    expected[3, 8] = 5
    expected[8, 17:22] = 5
    expected[7, 17] = 5
    for k in range(4):
        assert np.array_equal(solve(np.rot90(mixed, k)), np.rot90(expected, k))
    larger = np.zeros((16, 15), dtype=np.int8)
    payload = (np.arange(15).reshape(5, 3) % 8).astype(np.int8)
    _paint(larger, 9, {"kind": "source", "box": (1, 0, 7, 5)}, payload)
    _paint(larger, 9, {"kind": "receiver", "box": (3, 8, 12, 5), "gap": (3, 11)})
    delivered = solve(larger)
    assert np.array_equal(delivered[4:9, 9:12], payload)
    assert np.array_equal(delivered[9:14, 9:12], payload[::-1])
    shapes, sides, inks, payload_shapes, counts = set(), set(), set(), set(), set()
    fingerprints, structures = set(), set()
    shared = overlaps = multicolor = sparse = symmetric = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        fingerprints.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                ink, objects, links, _ = _check_pair(inp, out)
                shapes.add(inp.shape)
                inks.add(ink)
                counts.add(len(links))
                mapping = {0: 0, ink: 1}
                normalized = tuple(mapping.setdefault(int(x), len(mapping)) for x in inp.flat)
                structures.add((inp.shape, normalized))
                source_boxes = [v["source"]["box"] for v in links]
                shared += len(set(source_boxes)) < len(source_boxes)
                receivers = [v["receiver"] for v in links]
                overlaps += any(a["box"][0] < b["box"][0] + b["box"][2]
                                and b["box"][0] < a["box"][0] + a["box"][2]
                                for i, a in enumerate(receivers) for b in receivers[:i])
                for v in links:
                    sides.add(v["side"])
                    p = v["payload"]
                    payload_shapes.add(p.shape)
                    multicolor += len(np.unique(p)) >= 3
                    sparse += np.count_nonzero(p) <= p.size / 3
                    symmetric += np.array_equal(p, p[::-1]) or np.array_equal(p, p[:, ::-1])
    assert len(fingerprints) == 42 and len(structures) >= 150
    assert len(shapes) >= 35 and len(payload_shapes) >= 15
    assert sides == {"top", "bottom", "left", "right"} and len(inks) == 9
    assert {1, 2, 3} <= counts
    assert shared and overlaps and multicolor and sparse and symmetric
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(1000 + 10 * n + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "normalized_inputs": len(structures),
            "payload_shapes": len(payload_shapes), "shared_source_pairs": shared,
            "overlapping_receiver_spans": int(overlaps), "multicolor_payloads": int(multicolor),
            "sparse_payloads": int(sparse), "symmetric_payloads": int(symmetric)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
