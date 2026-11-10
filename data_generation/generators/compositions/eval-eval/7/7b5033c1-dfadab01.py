"""Printing trail: input-encoded stencils, ordered path traversal, transparent ink.

Only NumPy and the standard library are used. No files are read by generation
or solving. See task_spec.md for the full family, including unsupported inputs.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _components(cells, directions=DIR8):
    remaining = set(cells)
    result = []
    while remaining:
        q = min(remaining)
        remaining.remove(q)
        stack, component = [q], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(component)
    return result


def _trace(cells):
    cells = set(cells)
    adj = {q: [(q[0] + dr, q[1] + dc) for dr, dc in DIR4
               if (q[0] + dr, q[1] + dc) in cells] for q in cells}
    ends = sorted(q for q in cells if len(adj[q]) == 1)
    _require(len(ends) == 2 and all(len(v) in (1, 2) for v in adj.values()), "simple chain required")
    _require(ends[0][0] != ends[1][0], "endpoint rows must differ")
    current, previous, path = ends[0], None, []
    visited = set()
    while current is not None:
        _require(current not in visited, "path cycle")
        visited.add(current)
        path.append(current)
        choices = [q for q in adj[current] if q != previous]
        previous, current = current, choices[0] if choices else None
    _require(len(path) == len(cells), "disconnected chain")
    return path


def _parse(value):
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(int(counts[bg]) > a.size // 2, "background must be a strict majority")
    components = _components(map(tuple, np.argwhere(a != bg)))
    trails, keys = [], []
    for cells in components:
        colors = {int(a[q]) for q in cells}
        if len(colors) >= 3:
            trails.append(cells)
        else:
            keys.append(cells)
    _require(len(trails) == 1 and 2 <= len(keys) <= 3, "one trail and two or three keys required")
    book, key_boxes = {}, {}
    for cells in keys:
        r, c = min(q[0] for q in cells), min(q[1] for q in cells)
        h, w = max(q[0] for q in cells) - r + 1, max(q[1] for q in cells) - c + 1
        _require(3 <= h <= 4 and 3 <= w <= 5, "key dimensions")
        _require((r, c) in cells, "key needs its upper-left code")
        code = int(a[r, c])
        colors = {int(a[q]) for q in cells}
        _require(len(colors) == 2 and sum(a[q] == code for q in cells) == 1, "key code must be unique")
        ink = (colors - {code}).pop()
        mask = a[r:r+h, c:c+w] == ink
        _require(mask.sum() >= 4 and mask[0, 1] and mask[1, 0], "connected blank-corner key convention")
        _require(not mask[0, 0] and mask[-1].any() and mask[:, -1].any(), "tight key bounds")
        _require(len(_components(map(tuple, np.argwhere(mask)))) == 1, "stencil must be eight-connected")
        _require(code not in book, "duplicate key code")
        book[code], key_boxes[code] = (ink, mask.copy()), (r, c, h, w)
    inks = {ink for ink, _ in book.values()}
    _require(len(inks) == len(book) and not (inks & set(book)), "disjoint code and ink colors")
    _require(len({_array_key(mask) for _, mask in book.values()}) == len(book), "distinct stencil masks required")
    trail_colors = {int(a[q]) for q in trails[0]}
    wire = trail_colors - set(book)
    _require(len(wire) == 1 and not (wire & inks), "one distinct connector color")
    _require(len(trail_colors & set(book)) >= 2, "at least two active codes")
    _require(bg not in set(book) | inks | wire, "palette collision")
    path = _trace(trails[0])
    events = [(r, c, int(a[r, c])) for r, c in path if int(a[r, c]) in book]
    return {"array": a, "background": bg, "wire": wire.pop(), "book": book,
            "path": path, "events": events, "key_boxes": key_boxes}


def _render(shape, bg, events, book, *, opaque=False, solid=False, points=False):
    out = np.full(shape, bg, dtype=np.int8)
    for r, c, code in events:
        ink, mask = book[code]
        if points:
            out[r, c] = ink
            continue
        h, w = min(mask.shape[0], shape[0] - r), min(mask.shape[1], shape[1] - c)
        block = out[r:r+h, c:c+w]
        if opaque:
            block[:] = bg
        if solid:
            block[:] = ink
        else:
            block[mask[:h, :w]] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, stencil dictionary, path order and painting from this input."""
    info = _parse(input_grid)
    return _render(info["array"].shape, info["background"], info["events"], info["book"])


def _mask(rng, h, w):
    mask = rng.random((h, w)) < float(rng.uniform(.35, .72))
    mask[0, 0] = False
    mask[0, 1] = mask[1, 0] = mask[-1, -1] = True
    # Join any scattered ink by shortest eight-neighbor bridges. Never fill the
    # marked corner, which remains the transparent anchor of every stencil.
    for _ in range(h * w):
        parts = _components(map(tuple, np.argwhere(mask)))
        if len(parts) == 1:
            break
        p, q = min(((a, b) for a in parts[0] for part in parts[1:] for b in part),
                   key=lambda pair: max(abs(pair[0][0] - pair[1][0]), abs(pair[0][1] - pair[1][1])))
        r, c = p
        while (r, c) != q:
            r += int(np.sign(q[0] - r))
            c += int(np.sign(q[1] - c))
            mask[r, c] = True
    mask[0, 0] = False
    return mask


def _walk(rng, length):
    path, occupied, previous = [(0, 0)], {(0, 0)}, None
    persistence = float(rng.uniform(1.3, 6.5))
    for _ in range(length - 1):
        choices, weights = [], []
        for d in DIR4:
            q = path[-1][0] + d[0], path[-1][1] + d[1]
            if q in occupied:
                continue
            if sum((q[0] + dr, q[1] + dc) in occupied for dr, dc in DIR4) != 1:
                continue
            nr = max(q[0], max(r for r, _ in path)) - min(q[0], min(r for r, _ in path)) + 1
            nc = max(q[1], max(c for _, c in path)) - min(q[1], min(c for _, c in path)) + 1
            if nr > 21 or nc > 22:
                continue
            choices.append((q, d))
            weights.append(persistence if d == previous else 1.)
        if not choices:
            return None
        p = np.asarray(weights, dtype=float)
        q, previous = choices[int(rng.choice(len(choices), p=p / p.sum()))]
        occupied.add(q)
        path.append(q)
    if path[0][0] == path[-1][0]:
        return None
    if path[0][0] > path[-1][0]:
        path.reverse()
    lo_r, lo_c = min(r for r, _ in path), min(c for _, c in path)
    return [(r - lo_r, c - lo_c) for r, c in path]


def _fallback_path(rng):
    """Spacious four-lane serpentine, with randomized lengths and reflection."""
    width = int(rng.integers(12, 19))
    gaps = [int(rng.integers(4, 7)) for _ in range(3)]
    path = [(0, 0)]
    direction = 1
    for lane in range(4):
        for _ in range(width - 1):
            path.append((path[-1][0], path[-1][1] + direction))
        if lane < 3:
            for _ in range(gaps[lane]):
                path.append((path[-1][0] + 1, path[-1][1]))
        direction *= -1
    if rng.random() < .5:
        path = [(r, width - 1 - c) for r, c in path]
    return path


def _rectangle(q, h, w, shape):
    return {(r, c) for r in range(q[0], min(shape[0], q[0] + h))
            for c in range(q[1], min(shape[1], q[1] + w))}


def _choose_events(rng, path, shape, book, cluster_size, train, clip_axis):
    codes = list(book)
    mh = max(m.shape[0] for _, m in book.values())
    mw = max(m.shape[1] for _, m in book.values())
    full = [i for i, (r, c) in enumerate(path) if r + mh <= shape[0] and c + mw <= shape[1]]
    starts = list(range(len(path) - cluster_size + 1))
    if clip_axis is not None:
        axis = clip_axis
        edge = max(q[axis] for q in path)
        starts = [i for i in starts if any(path[j][axis] == edge
                  for j in range(i, i + cluster_size))]
    rng.shuffle(starts)
    for start in starts[:60]:
        cluster = list(range(start, start + cluster_size))
        occupied = set().union(*(_rectangle(path[i], mh, mw, shape) for i in cluster))
        isolated = []
        candidates = list(full)
        rng.shuffle(candidates)
        for i in candidates:
            rect = _rectangle(path[i], mh, mw, shape)
            if not (rect & occupied):
                isolated.append(i)
                occupied |= rect
                if len(isolated) == len(codes):
                    break
        if train and len(isolated) < len(codes):
            continue
        if not train:
            # Dense query clusters plus spatially separated events, with no
            # obligation to leave dictionary demonstrations unobscured.
            isolated = isolated[:len(codes)]
            available = [i for i in range(len(path)) if i not in cluster + isolated]
            rng.shuffle(available)
            isolated += available[:max(0, len(codes) - len(isolated))]
        labels = {i: codes[j] for j, i in enumerate(isolated)}
        labels.update({i: int(rng.choice(codes)) for i in cluster})
        # Adjacent different-code prints make chronology observable.
        labels[cluster[0]], labels[cluster[1]] = map(int, rng.choice(codes, 2, replace=False))
        events = [(path[i][0], path[i][1], labels[i]) for i in sorted(labels)]
        bg = -1  # temporary background cannot collide with ink
        correct = _render(shape, bg, events, book)
        rivals = (events[::-1], sorted(events), sorted(events, key=lambda e: (e[2], e[0], e[1])))
        if any(np.array_equal(correct, _render(shape, bg, e, book)) for e in rivals):
            continue
        if np.array_equal(correct, _render(shape, bg, events, book, opaque=True)):
            continue
        return labels
    return None


def _place_keys(rng, inp, output_occupied, bg, book):
    reserved = set(output_occupied) | set(map(tuple, np.argwhere(inp != bg)))
    # Keep keys visibly separate from the trail and outside all print windows.
    blocked = {(r + dr, c + dc) for r, c in reserved for dr, dc in ((0, 0),) + DIR8}
    boxes = {}
    codes = list(book)
    rng.shuffle(codes)
    for code in codes:
        ink, mask = book[code]
        h, w = mask.shape
        locations = [(r, c) for r in range(inp.shape[0] - h + 1)
                     for c in range(inp.shape[1] - w + 1)]
        rng.shuffle(locations)
        for r, c in locations:
            rect = _rectangle((r, c), h, w, inp.shape)
            if rect & blocked:
                continue
            inp[r:r+h, c:c+w][mask] = ink
            inp[r, c] = code
            boxes[code] = (r, c, h, w)
            blocked.update((a + dr, b + dc) for a, b in rect for dr, dc in ((0, 0),) + DIR8)
            break
        else:
            return None
    return boxes


def _make_pair(rng, book, wire, bg, index, test, clip_axis, allow_above=True):
    # At most 96 independent geometries, then spacious constructive geometries.
    # No attempt writes to disk; only the current candidate is retained.
    for attempt in range(128):
        if attempt < 96:
            path = _walk(rng, int(rng.integers(42, 68) if not test else rng.integers(54, 83)))
            if path is None:
                continue
        else:
            path = _fallback_path(rng)
        if not allow_above and min(r for r, _ in path) < path[0][0]:
            continue
        bh, bw = max(r for r, _ in path) + 1, max(c for _, c in path) + 1
        H = 30 if attempt >= 96 else int(rng.integers(max(18, bh), 31))
        W = 30 if attempt >= 96 else int(rng.integers(max(18, bw), 31))
        top, left = int(rng.integers(H - bh + 1)), int(rng.integers(W - bw + 1))
        force_clip = not test and index == 1
        if force_clip:
            if clip_axis == 0:
                top = H - bh
            else:
                left = W - bw
        elif test and rng.random() < .55:
            if rng.random() < .5:
                top = H - bh
            else:
                left = W - bw
        path = [(r + top, c + left) for r, c in path]
        cluster = int(rng.integers(6, 9) if test else rng.integers(3, 5))
        active_book = book
        if test and len(book) == 3 and rng.random() < .5:
            selected = set(map(int, rng.choice(list(book), 2, replace=False)))
            active_book = {c: v for c, v in book.items() if c in selected}
        labels = _choose_events(rng, path, (H, W), active_book, cluster, not test,
                                clip_axis if force_clip else None)
        if labels is None:
            continue
        inp = np.full((H, W), bg, dtype=np.int8)
        for i, q in enumerate(path):
            inp[q] = labels.get(i, wire)
        events = [(path[i][0], path[i][1], code) for i, code in sorted(labels.items())]
        occupied = set().union(*(_rectangle((r, c), *book[code][1].shape, inp.shape)
                                 for r, c, code in events))
        if _place_keys(rng, inp, occupied, bg, book) is None:
            continue
        if np.count_nonzero(inp == bg) <= inp.size // 2:
            continue
        out = _render(inp.shape, bg, events, book)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded trail construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 4))
    palette = list(map(int, rng.permutation(10)))
    wire = palette[1]
    book = {}
    signatures = set()
    for j in range(k):
        for _ in range(40):
            mask = _mask(rng, int(rng.integers(3, 5)), int(rng.integers(3, 6)))
            key = mask.shape, mask.tobytes()
            if key not in signatures:
                break
        else:
            raise RuntimeError("could not construct distinct stencils")
        signatures.add(key)
        book[palette[2 + j]] = (palette[2 + k + j], mask)
    backgrounds = [palette[0]] + palette[2 + 2*k:]
    clip_axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + nq):
        test = i >= nt
        index = i - nt if test else i
        bg = backgrounds[i % len(backgrounds)] if i < 2 else int(rng.choice(backgrounds))
        for _ in range(8):
            allow_above = not test or any(_features(_parse(p["input"]))["above"] for p in puzzle["train"])
            pair = _make_pair(rng, book, wire, bg, index, test, clip_axis, allow_above)
            signature = _array_key(pair["input"])
            if signature not in seen:
                break
        else:
            raise RuntimeError("distinct input construction exhausted")
        seen.add(signature)
        puzzle["test" if test else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _array_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _book_key(book):
    return tuple((code, ink, _array_key(mask)) for code, (ink, mask) in sorted(book.items()))


def _features(info):
    a, path, events, book = (info[k] for k in ("array", "path", "events", "book"))
    shape, bg = a.shape, info["background"]
    expected = _render(shape, bg, events, book)
    coverage = np.zeros(shape, dtype=np.int8)
    for r, c, code in events:
        mask = book[code][1]
        h, w = min(mask.shape[0], shape[0] - r), min(mask.shape[1], shape[1] - c)
        coverage[r:r+h, c:c+w] += mask[:h, :w]
    isolated_codes = set()
    rectangles = [_rectangle((r, c), *book[code][1].shape, shape) for r, c, code in events]
    for i, (r, c, code) in enumerate(events):
        mask = book[code][1]
        if r + mask.shape[0] <= shape[0] and c + mask.shape[1] <= shape[1]:
            if all(i == j or not (rectangles[i] & rectangles[j]) for j in range(len(events))):
                isolated_codes.add(code)
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return {"events": len(events), "length": len(path), "shape": shape,
            "isolated_codes": isolated_codes, "depth": int(coverage.max()),
            "turns": int(sum(a != b for a, b in zip(steps, steps[1:]))),
            "above": bool(min(r for r, _ in path) < path[0][0]),
            "clip": any(r + book[c][1].shape[0] > shape[0] or
                        col + book[c][1].shape[1] > shape[1] for r, col, c in events),
            "reverse_diff": not np.array_equal(expected, _render(shape, bg, events[::-1], book)),
            "spatial_diff": not np.array_equal(expected, _render(shape, bg, sorted(events), book)),
            "group_diff": not np.array_equal(expected, _render(shape, bg,
                                  sorted(events, key=lambda e: (e[2], e[0], e[1])), book)),
            "opaque_diff": not np.array_equal(expected, _render(shape, bg, events, book, opaque=True)),
            "points_diff": not np.array_equal(expected, _render(shape, bg, events, book, points=True))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(nq, "num_test", 1, 4)
    _require(nt > nq, "counts")
    seen, dictionary, wire = set(), None, None
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output shape must be preserved")
            signature = _array_key(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _parse(inp)
            key = _book_key(info["book"])
            if dictionary is None:
                dictionary, wire = key, info["wire"]
            _require(key == dictionary and wire == info["wire"], "puzzle conventions changed")
            expected = _render(inp.shape, info["background"], info["events"], info["book"])
            _require(np.array_equal(out, expected), "incorrect ordered stencil output")
            # Keys are material for reading, never independent printing events.
            for r, c, h, w in info["key_boxes"].values():
                _require(bool(np.all(out[r:r+h, c:c+w] == info["background"])), "printed over a key")
            f = _features(info)
            _require(all(f[k] for k in ("reverse_diff", "spatial_diff", "group_diff", "opaque_diff", "points_diff")),
                     "missing concept-disambiguation witness")
            if split == "train":
                _require(f["isolated_codes"] == set(info["book"]), "unobscured stencil demonstrations required")
            records[split].append(f)
    _require(any(f["clip"] for f in records["train"]), "clipping must be demonstrated")
    _require(min(f["events"] for f in records["test"]) > max(f["events"] for f in records["train"]),
             "query must extrapolate print count")
    if any(f["above"] for f in records["test"]):
        _require(any(f["above"] for f in records["train"]), "higher-endpoint convention needs a witness")


def self_test():
    """Bounded, in-memory tests of 42 seeds plus mutations and interface edges."""
    signatures, shapes, geometries, books = set(), set(), set(), set()
    lengths, turns, backgrounds, event_counts, mask_shapes, depths = set(), set(), set(), set(), set(), set()
    clipped = above = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                f = _features(info)
                for name in ("input", "output"):
                    a = pair[name]
                    digest.update(bytes(a.shape) + a.tobytes())
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                shapes.add(f["shape"])
                books.add(_book_key(info["book"]))
                origin = info["path"][0]
                geometries.add(tuple((r-origin[0], c-origin[1]) for r, c in info["path"]))
                lengths.add(f["length"])
                turns.add(f["turns"])
                backgrounds.add(info["background"])
                event_counts.add(f["events"])
                mask_shapes.update(m.shape for _, m in info["book"].values())
                depths.add(f["depth"])
                clipped += f["clip"]
                above += f["above"]
        signatures.add(digest.hexdigest())
        if seed == 0:
            first = puzzle
    _require(len(signatures) == 42 and len(books) == 42, "seed diversity")
    _require(len(shapes) >= 35 and len(geometries) >= 100, "geometry diversity")
    _require(len(lengths) >= 20 and len(turns) >= 8, "path diversity")
    _require(len(backgrounds) == 10 and len(mask_shapes) == 6, "role and stencil size diversity")
    _require(len(event_counts) >= 6 and len(depths) >= 3 and clipped >= 42, "printing diversity")
    again = generate_puzzle(0)
    for s in first:
        for a, b in zip(first[s], again[s]):
            _require(all(np.array_equal(a[k], b[k]) for k in a), "determinism")
    # Input-only stencil response: change one key's ink, without changing the
    # trail, seed or any output supplied to solve. Every occurrence must follow.
    query = first["test"][0]
    info = _parse(query["input"])
    code = info["events"][0][2]
    old_ink, mask = info["book"][code]
    unused = next(c for c in range(10) if c not in set(map(int, np.unique(query["input"]))))
    altered = query["input"].copy()
    r, c, h, w = info["key_boxes"][code]
    altered[r:r+h, c:c+w][mask] = unused
    expected = query["output"].copy()
    expected[expected == old_ink] = unused
    _require(np.array_equal(solve(altered), expected), "input-only key response")
    # Change stencil geometry in an input key, with all route cells held fixed.
    # An isolated demonstration gives a directly observable pixel consequence.
    example = first["train"][0]
    info = _parse(example["input"])
    geometry_checked = False
    for r, c, code in info["events"]:
        ink, mask = info["book"][code]
        h, w = mask.shape
        rectangle = _rectangle((r, c), h, w, info["array"].shape)
        others = [_rectangle((a, b), *info["book"][k][1].shape, info["array"].shape)
                  for a, b, k in info["events"] if (a, b) != (r, c)]
        if len(rectangle) != h*w or any(rectangle & other for other in others):
            continue
        kr, kc, _, _ = info["key_boxes"][code]
        for dr in range(h):
            for dc in range(w):
                if (dr, dc) in ((0, 0), (0, 1), (1, 0)):
                    continue
                changed = example["input"].copy()
                new_color = info["background"] if mask[dr, dc] else ink
                changed[kr+dr, kc+dc] = new_color
                try:
                    _parse(changed)
                except ValueError:
                    continue
                predicted = solve(changed)
                _require(int(predicted[r+dr, c+dc]) == new_color, "input-only mask response")
                _require(not np.array_equal(predicted, example["output"]), "stencil geometry had no effect")
                geometry_checked = True
                break
            if geometry_checked:
                break
        if geometry_checked:
            break
    _require(geometry_checked, "no stencil geometry mutation tested")
    for cells in (((0, 0), (0, 1), (0, 2)),
                  ((0, 1), (1, 0), (1, 1), (1, 2)),
                  ((0, 0), (0, 1), (1, 0), (1, 1)),
                  ((0, 0), (1, 0), (4, 4))):
        try:
            _trace(cells)
        except ValueError:
            continue
        raise AssertionError("malformed or tied-endpoint trail accepted")
    # Whole-output corruptions must be rejected, even with unchanged inputs.
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    i = _parse(corrupt["test"][0]["input"])
    corrupt["test"][0]["output"] = _render(i["array"].shape, i["background"], i["events"][::-1], i["book"])
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("reverse-order corruption accepted")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(908, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "dictionaries": len(books), "shapes": len(shapes), "path_geometries": len(geometries),
            "path_lengths": len(lengths), "turn_counts": len(turns), "background_colors": len(backgrounds),
            "stencil_sizes": len(mask_shapes), "print_counts": sorted(event_counts),
            "overlap_depths": sorted(depths), "clipped_pairs": clipped, "above_endpoint_pairs": above}
