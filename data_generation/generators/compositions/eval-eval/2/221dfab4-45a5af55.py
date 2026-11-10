"""Echoing layers: a periodic probe activates complete geometric stamp depths.

No filesystem, network, stored examples, latent solver state, or parent imports.
All random choices belong to the local default_rng. Diagnostics are streamed.
"""
from __future__ import annotations

import hashlib
import numpy as np


_SIDES = ("top", "right", "bottom", "left")
_WITNESSES = ("odd_clear", "ordinary_overwrite", "signal_contact",
              "remote_echo", "unhit_on_signal_row", "selective_repeat",
              "preserved_layer", "cycle")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(isinstance(value, (int, np.integer))
             and not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), name + " out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid color out of range")
    return a.astype(np.int8, copy=False)


def _canonical(grid, side):
    if side == "top":
        return grid
    if side == "bottom":
        return grid[::-1]
    if side == "left":
        return grid.T
    return grid.T[::-1]


def _orient(grid, side):
    if side in ("top", "bottom"):
        return _canonical(grid, side).copy()
    if side == "left":
        return grid.T.copy()
    return grid[::-1].T.copy()


def _depths(h, w):
    r = np.minimum(np.arange(h), np.arange(h)[::-1])
    c = np.minimum(np.arange(w), np.arange(w)[::-1])
    return np.minimum(r[:, None], c[None, :])


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _parse(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) > grid.size / 2, "background must be a strict majority")
    count = dict(zip(map(int, values), map(int, counts)))
    keys = []
    for marker in values:
        marker = int(marker)
        if marker == bg or count[marker] not in (2, 4, 6, 8):
            continue
        for side in _SIDES:
            view = _canonical(grid, side)
            points = np.argwhere(view == marker)
            if not np.all(points[:, 0] == 0):
                continue
            pos = np.sort(points[:, 1])
            start, end = int(pos[0]), int(pos[-1]) + 1
            length = end - start
            if length != len(points) + 1 or length not in (3, 5, 7, 9):
                continue
            mid = start + length // 2
            expected = np.array([i for i in range(start, end) if i != mid])
            signal = int(view[0, mid])
            if (np.array_equal(pos, expected) and signal not in (bg, marker)
                    and count.get(signal) == 1):
                keys.append((side, start, length, marker, signal))
    _require(len(keys) == 1, "need exactly one unambiguous beaded edge bar")
    side, start, length, marker, signal = keys[0]
    g = _canonical(grid, side)
    fg = (g != bg) & (g != marker) & (g != signal)
    boxes = []
    for cells in _components(fg):
        rr, cc = np.asarray(cells).T
        r, c = int(rr.min()), int(cc.min())
        h, w = int(rr.max()) - r + 1, int(cc.max()) - c + 1
        _require(h >= 4 and w >= 4 and len(cells) == h * w, "stamps must be filled rectangles")
        local = g[r:r + h, c:c + w]
        depth = _depths(h, w)
        code = []
        for d in range(int(depth.max()) + 1):
            colors = np.unique(local[depth == d])
            _require(len(colors) == 1, "each geometric depth must be monochromatic")
            code.append(int(colors[0]))
        boxes.append((r, c, h, w, tuple(code)))
    _require(1 <= len(boxes) <= 4, "need one to four stamps")
    for i, (r, c, h, w, _) in enumerate(boxes):
        for rr, cc, hh, ww, _ in boxes[i + 1:]:
            _require(r + h < rr or rr + hh < r or c + w < cc or cc + ww < c,
                     "stamps require a one-cell moat including diagonals")
    return dict(grid=g, background=bg, marker=marker, signal=signal, side=side,
                start=start, length=length, boxes=boxes)


def _active(box, start, length):
    r, c, h, w = box[:4]
    depth = _depths(h, w)
    rows = (np.arange(r, r + h) % 6 == 4)[:, None]
    cols = ((np.arange(c, c + w) >= start)
            & (np.arange(c, c + w) < start + length))[None, :]
    return np.unique(depth[rows & cols])


def _track(g, info, signal_phase=4):
    start, length = info["start"], info["length"]
    for d in range(len(g)):
        color = info["background"]
        if d % 2 == 0:
            color = info["signal"] if d % 6 == signal_phase else info["marker"]
        g[d, start:start + length] = color


def _prediction(info, mode="layers"):
    g = info["grid"]
    out = g.copy()
    start, length, signal = info["start"], info["length"], info["signal"]
    touched_colors = set()
    if mode == "colors":
        for box in info["boxes"]:
            touched_colors.update(box[4][d] for d in _active(box, start, length))
    for box in info["boxes"]:
        r, c, h, w, code = box
        depth = _depths(h, w)
        active = _active(box, start, length)
        if mode == "layers":
            selected = np.isin(depth, active)
        elif mode == "objects":
            selected = np.full((h, w), len(active) > 0)
        elif mode == "colors":
            selected = np.isin(g[r:r + h, c:c + w], tuple(touched_colors))
        elif mode == "rows":
            selected = np.broadcast_to((np.arange(r, r + h) % 6 == 4)[:, None], (h, w))
        elif mode == "none":
            selected = np.zeros((h, w), dtype=bool)
        else:
            raise ValueError("unknown diagnostic mode")
        out[r:r + h, c:c + w][selected] = signal
    _track(out, info)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, contact and layer from this input alone; never mutate it."""
    info = _parse(input_grid)
    return _orient(_prediction(info), info["side"])


def _frame_paint(info):
    """Authored cross-check: enumerate tick contacts, then draw four frame sides.

    This uses neither the solver's vector activation nor its distance-map painting.
    It is not an externally supplied reference solver.
    """
    out = info["grid"].copy()
    start, end = info["start"], info["start"] + info["length"]
    for r, c, h, w, _ in info["boxes"]:
        depths = set()
        for tick in range(4, len(out), 6):
            for col in range(start, end):
                if r <= tick < r + h and c <= col < c + w:
                    depths.add(min(tick - r, r + h - 1 - tick, col - c, c + w - 1 - col))
        for d in depths:
            a, b = r + d, r + h - 1 - d
            x, y = c + d, c + w - 1 - d
            out[a, x:y + 1] = out[b, x:y + 1] = info["signal"]
            out[a:b + 1, x] = out[a:b + 1, y] = info["signal"]
    for d in range(len(out)):
        out[d, start:end] = (info["background"] if d % 2 else
                             info["signal"] if d % 6 == 4 else info["marker"])
    return _orient(out, info["side"])


def _evidence(info):
    start, end = info["start"], info["start"] + info["length"]
    e = {name: False for name in _WITNESSES}
    e["cycle"] = len(info["grid"]) >= 13
    e["multi"] = len(info["boxes"]) > 1
    e["partial_stamp"] = False
    for box in info["boxes"]:
        r, c, h, w, code = box
        depth = _depths(h, w)
        active = _active(box, start, end - start)
        on = np.isin(depth, active)
        rows = np.broadcast_to(np.arange(r, r + h)[:, None], (h, w))
        inside = np.broadcast_to(((np.arange(c, c + w) >= start)
                                  & (np.arange(c, c + w) < end))[None, :], (h, w))
        e["odd_clear"] |= bool(np.any(inside & (rows % 2 == 1)))
        e["ordinary_overwrite"] |= bool(np.any(inside & np.isin(rows % 6, (0, 2))))
        e["signal_contact"] |= bool(len(active))
        e["remote_echo"] |= bool(np.any(on & ~inside & (rows % 6 != 4)))
        e["unhit_on_signal_row"] |= bool(np.any(~on & ~inside & (rows % 6 == 4)))
        e["preserved_layer"] |= bool(np.any(~on & ~inside))
        e["partial_stamp"] |= bool(np.any(on & ~inside) and np.any(~on & ~inside))
        for d in range(len(code) - 1):
            if (code[d] == code[d + 1] and ((d in active) != (d + 1 in active))
                    and np.any((depth == d) & ~inside)
                    and np.any((depth == d + 1) & ~inside)):
                e["selective_repeat"] = True
    return e


def _box_geometry(rng, depth, span, count, start, length, teaching):
    """At most 96 geometry proposals, each with at most 96 placements per box."""
    area_limit = int(depth * span * .46) - length
    for _ in range(96):
        boxes, area = [], 0
        for index in range(count):
            for _ in range(96):
                low = 8 if teaching and index == 0 else 4
                high_h = min(depth - 1, 20 if count == 1 else 16 if index == 0 else 11)
                high_w = min(span, 22 if count == 1 else 18 if index == 0 else 12)
                h = int(rng.integers(low, high_h + 1))
                w = h if rng.random() < .28 and h <= high_w else int(rng.integers(low, high_w + 1))
                r, c = int(rng.integers(1, depth - h + 1)), int(rng.integers(span - w + 1))
                if area + h * w > area_limit:
                    continue
                if any(not (r + h < rr or rr + hh < r or c + w < cc or cc + ww < c)
                       for rr, cc, hh, ww in boxes):
                    continue
                candidate = (r, c, h, w)
                if teaching and index == 0:
                    dummy = candidate + (tuple(range((min(h, w) + 1) // 2)),)
                    info = dict(grid=np.empty((depth, span), dtype=np.int8), start=start,
                                length=length, boxes=[dummy])
                    witness = _evidence(info)
                    # Repeated colors are assigned later; require the geometry that
                    # permits a selectively activated adjacent pair of depths.
                    active = set(map(int, _active(dummy, start, length)))
                    split = any((d in active) != (d + 1 in active)
                                for d in range(len(dummy[4]) - 1))
                    required = [k for k in _WITNESSES if k != "selective_repeat"]
                    if not split or not all(witness[k] for k in required):
                        continue
                boxes.append(candidate)
                area += h * w
                break
            else:
                break
        if len(boxes) == count:
            return boxes
    return None


def _color_scene(rng, depth, span, start, length, boxes, bg, marker, signal, teaching):
    g = np.full((depth, span), bg, dtype=np.int8)
    palette = [int(c) for c in rng.permutation([i for i in range(10) if i not in (bg, marker, signal)])]
    for index, (r, c, h, w) in enumerate(boxes):
        n = (min(h, w) + 1) // 2
        k = int(rng.integers(1 if not teaching or index else 2, min(7, n) + 1))
        colors = palette[:k]
        code = [int(rng.choice(colors))]
        for _ in range(n - 1):
            code.append(code[-1] if rng.random() < .35 else int(rng.choice(colors)))
        if teaching and index == 0:
            active = set(map(int, _active((r, c, h, w), start, length)))
            choices = [d for d in range(n - 1) if ((d in active) != (d + 1 in active))
                       and np.any((_depths(h, w) == d) &
                                  ~np.broadcast_to(((np.arange(c, c + w) >= start) &
                                                    (np.arange(c, c + w) < start + length))[None, :], (h, w)))
                       and np.any((_depths(h, w) == d + 1) &
                                  ~np.broadcast_to(((np.arange(c, c + w) >= start) &
                                                    (np.arange(c, c + w) < start + length))[None, :], (h, w)))]
            if not choices:
                return None
            d = int(rng.choice(choices))
            code[d] = code[d + 1] = colors[0]
            other = next(i for i in range(n) if i not in (d, d + 1))
            code[other] = colors[1]
        g[r:r + h, c:c + w] = np.asarray(code, dtype=np.int8)[_depths(h, w)]
    g[0, start:start + length] = marker
    g[0, start + length // 2] = signal
    return g


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _make_pair(rng, bg, side, count, teaching, seen, excluded_signals=(), echo_required=False):
    allowed = [c for c in range(10) if c != bg]
    for _ in range(24):
        depth = int(rng.integers(18 if count <= 2 else 22, 31))
        if teaching and depth % 6 == 3:
            continue
        span = int(rng.integers(18 if count <= 2 else 22, 31))
        length = int(rng.choice((3, 5, 7) if teaching else (3, 5, 7, 9)))
        start = int(rng.integers(span - length + 1))
        marker, signal = map(int, rng.choice(allowed, size=2, replace=False))
        if signal in excluded_signals:
            continue
        boxes = _box_geometry(rng, depth, span, count, start, length, teaching)
        if boxes is None:
            continue
        g = _color_scene(rng, depth, span, start, length, boxes, bg, marker, signal, teaching)
        if g is None:
            continue
        inp = _orient(g, side)
        if _key(inp) in seen:
            continue
        info = _parse(inp)
        if teaching and not all(_evidence(info)[k] for k in _WITNESSES):
            continue
        if echo_required:
            e = _evidence(info)
            if not (e["remote_echo"] and e["partial_stamp"]):
                continue
        return {"input": inp, "output": _frame_paint(info)}
    # Finite fallback: four available disjoint sites, a selective anchor at the
    # first. At least 56 marker/signal assignments remain after excluding one
    # signal, more than the <=13 preceding inputs. Each assignment changes input.
    sites = [(2, 2, 10, 10), (14, 1, 6, 6), (14, 10, 6, 6), (14, 19, 6, 6)]
    roles = [(m, s) for m in allowed for s in allowed if m != s and s not in excluded_signals]
    for i in rng.permutation(len(roles)):
        marker, signal = roles[int(i)]
        g = _color_scene(rng, 24, 28, 6, 3, sites[:count], bg, marker, signal, teaching)
        if g is None:
            continue
        inp = _orient(g, side)
        if _key(inp) not in seen:
            info = _parse(inp)
            return {"input": inp, "output": _frame_paint(info)}
    raise RuntimeError("finite uniqueness fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    _require(nt > ns, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    bg, base = int(rng.integers(10)), int(rng.integers(4))
    puzzle, seen, first_signal = {"train": [], "test": []}, set(), None
    for split, n in (("train", nt), ("test", ns)):
        for i in range(n):
            teaching = split == "train" and i < 2
            side = _SIDES[(base + 2 * (i % 2) + (split == "test")) % 4]
            count = (i + 1 if teaching else int(rng.integers(1, 3))) if split == "train" else int(rng.integers(3, 5))
            pair = _make_pair(rng, bg, side, count, teaching, seen,
                              (first_signal,) if teaching and i == 1 else (),
                              echo_required=split == "test")
            if teaching and i == 0:
                first_signal = _parse(pair["input"])["signal"]
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    # Presentation order carries no meaning.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen, evidence, signals = set(), [], set()
    defeated = {mode: False for mode in ("none", "rows", "objects", "colors")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas size must be preserved")
            key = _key(inp)
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            info = _parse(inp)
            expected = _orient(_prediction(info), info["side"])
            _require(np.array_equal(out, expected), "wrong layer/track pixels")
            _require(np.array_equal(out, _frame_paint(info)), "frame cross-check disagrees")
            if split == "train":
                evidence.append(_evidence(info))
                signals.add(info["signal"])
                for mode in defeated:
                    defeated[mode] |= not np.array_equal(out, _orient(_prediction(info, mode), info["side"]))
            else:
                e = _evidence(info)
                _require(e["remote_echo"] and e["partial_stamp"],
                         "held-out examples must retain a visible layer interaction")
    _require(all(any(e[k] for e in evidence) for k in _WITNESSES), "incomplete teaching evidence")
    _require(any(e["multi"] for e in evidence), "multiple stamps must be demonstrated")
    _require(len(signals) >= 2, "training must reveal varying signal roles")
    _require(all(defeated.values()), "a documented shortcut fits all demonstrations")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, no artifact writes; return descriptive measurements for recording."""
    signatures, shapes, structures = set(), set(), set()
    sides = {s: set() for s in ("train", "test")}
    counts, lengths, layer_counts, colors, aspects, parities = set(), set(), set(), set(), set(), set()
    activated, untouched, repeated, returned, boundary = 0, 0, 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        if seed < 3:
            assert _signature(p) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _parse(pair["input"])
                shapes.add(pair["input"].shape)
                sides[split].add(info["side"])
                counts.add(len(info["boxes"]))
                lengths.add(info["length"])
                colors.update(map(int, np.unique(pair["input"])))
                roles = {}
                normalized = bytes(roles.setdefault(int(x), len(roles)) for x in pair["input"].flat)
                structures.add((pair["input"].shape, hashlib.sha256(normalized).hexdigest()))
                for box in info["boxes"]:
                    r, c, h, w, code = box
                    active = _active(box, info["start"], info["length"])
                    activated += bool(len(active))
                    untouched += not bool(len(active))
                    layer_counts.add(len(code))
                    aspects.add(int(np.sign(h - w)))
                    parities.add((h % 2, w % 2))
                    repeated += any(a == b for a, b in zip(code, code[1:]))
                    runs = [a for i, a in enumerate(code) if i == 0 or code[i - 1] != a]
                    returned += len(runs) > len(set(runs))
                    boundary += r + h == len(info["grid"]) or c == 0 or c + w == info["grid"].shape[1]
        if seed < 4:
            for pair in p["train"] + p["test"]:
                inp, out = pair["input"], pair["output"]
                saved = inp.copy()
                for k in range(4):
                    assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                perm = np.random.default_rng(seed + 900).permutation(10)
                assert np.array_equal(solve(perm[inp]), perm[out])
                assert np.array_equal(inp, saved)
    assert len(signatures) == 42 and len(structures) >= 200 and len(shapes) >= 35
    assert all(sides[s] == set(_SIDES) for s in sides)
    assert counts == {1, 2, 3, 4} and lengths == {3, 5, 7, 9}
    assert colors == set(range(10)) and aspects == {-1, 0, 1}
    assert parities == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(layer_counts) >= 6 and activated > 100 and untouched > 15
    assert repeated > 100 and returned > 20 and boundary > 15
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Boundary fixtures: corner bar, canvas-edge stamp, odd rectangular center,
    # no signal contact, and both truncated and complete final clock periods.
    for depth in (18, 19, 20, 21, 22, 23):
        g = np.zeros((depth, 24), dtype=np.int8)
        g[0, :3] = (1, 3, 1)
        g[-7:, -9:] = np.asarray([2, 2, 4, 2], dtype=np.int8)[_depths(7, 9)]
        for side in _SIDES:
            inp = _orient(g, side)
            info = _parse(inp)
            assert not len(_active(info["boxes"][0], 0, 3))
            assert np.array_equal(solve(inp), _frame_paint(info))
    # Numeric semantic fixture: only depths 1 and 2 contact signal ticks.
    # Equal original colors at depths 0 and 1 must acquire different answers.
    g = np.zeros((18, 18), dtype=np.int8)
    g[0, 6:9] = (1, 3, 1)
    g[2:12, 2:12] = np.asarray([2, 2, 4, 2, 5], dtype=np.int8)[_depths(10, 10)]
    expected = solve(g)
    assert expected[4, 2] == 2  # unhit depth on a signal cross-section
    assert expected[3, 3] == 3  # remote echo away from every signal cross-section
    assert expected[4, 4] == 3 and expected[5, 5] == 2
    assert expected[3, 6] == 0 and expected[2, 6] == 1  # overwrite beats echo
    assert expected[4, 6] == 3 and expected[0, 7] == 1
    for side in _SIDES:
        assert np.array_equal(solve(_orient(g, side)), _orient(expected, side))
    p = generate_puzzle(25, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), shapes=len(shapes),
                color_normalized_inputs=len(structures), stamp_counts=sorted(counts),
                bar_lengths=sorted(lengths), layer_counts=sorted(layer_counts),
                activated_stamps=activated, untouched_stamps=untouched,
                adjacent_repeat_stamps=repeated, returning_color_stamps=returned,
                boundary_stamps=boundary, independent_solver=False)
