"""Gap-guided relief printing: rigid fitting decodes a registered color stencil.

Only standard-library and NumPy dependencies. No files, parent generators,
training examples, cached answers, or random state are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer) and bool(np.all((0 <= a) & (a <= 9))),
           "grids must contain integer ARC colors")
    return a


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    todo.append(p)
        result.append(frozenset(component))
    return result


def _cells(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _rectangle(cells):
    r0, r1, c0, c1 = _box(cells)
    return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def _shift(cells, origin):
    return frozenset((r + origin[0], c + origin[1]) for r, c in cells)


def _turn(p, op):
    r, c = p
    if op >= 4:
        c = -c
    for _ in range(op % 4):
        r, c = c, -r
    return r, c


def _transform(regions, op):
    raw = [frozenset(_turn(p, op) for p in region) for region in regions]
    points = frozenset().union(*raw)
    r0, _, c0, _ = _box(points)
    return tuple(_shift(region, (-r0, -c0)) for region in raw)


def _canonical(cells):
    return min(tuple(sorted(_transform((cells,), op)[0])) for op in range(8))


def _registered_signature(offsets):
    # Keep the origin; normalizing by the mask box would lose registration.
    return min(tuple(sorted(_turn(p, op) for p in offsets)) for op in range(8))


def _joint(plug, payload):
    contacts = {p for p in plug if any((p[0] + dr, p[1] + dc) in payload for dr, dc in D4)}
    _check(len(contacts) == 1, "a plug must have one contact cell with its payload")
    return next(iter(contacts))


def _mode(a):
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    _check(np.count_nonzero(counts == counts.max()) == 1, "background must be the unique mode")
    return int(counts.argmax())


def _decode(a):
    """Read plates, mold candidates and a unique simultaneous docking plan."""
    a = _grid(a)
    bg = _mode(a)
    keys, mono = [], []
    for component in _components(_cells(a != bg)):
        colors = tuple(sorted({int(a[p]) for p in component}))
        _check(len(colors) <= 2, "instruction and dot components must be separable")
        if len(colors) == 2:
            regions = tuple(frozenset(p for p in component if int(a[p]) == color) for color in colors)
            _check(all(len(_components(x)) == 1 for x in regions), "disconnected key color")
            keys.append({"cells": component, "colors": colors, "regions": regions})
        else:
            mono.append((colors[0], component))
    _check(bool(keys), "no printing plate")
    ink_colors = {c for key in keys for c in key["colors"]}
    neutral_colors = {c for c, component in mono if c not in ink_colors}
    _check(len(neutral_colors) == 1, "molds need one distinct neutral color")
    neutral = next(iter(neutral_colors))
    fragments = [component for c, component in mono if c == neutral]
    _check(len(fragments) == 2 * len(keys) and all(len(x) >= 2 for x in fragments),
           "each plate requires two nontrivial mold fragments")
    targets = []
    for i, first in enumerate(fragments):
        for j in range(i + 1, len(fragments)):
            fixed = first | fragments[j]
            rectangle = _rectangle(fixed)
            if any(int(a[p]) != (neutral if p in fixed else bg) for p in rectangle):
                continue
            slot = rectangle - fixed
            if slot and len(_components(slot)) == 1:
                targets.append({"parts": frozenset((i, j)), "rectangle": rectangle,
                                "slot": slot, "fixed": fixed})
    options = []
    for index, key in enumerate(keys):
        candidates, seen = [], set()
        for target in targets:
            slot = target["slot"]
            sr, _, sc, _ = _box(slot)
            for pi in (0, 1):
                try:
                    _joint(key["regions"][pi], key["regions"][1 - pi])
                except ValueError:
                    continue
                for op in range(8):
                    regions = _transform(key["regions"], op)
                    pr, _, pc, _ = _box(regions[pi])
                    placed = tuple(_shift(region, (sr - pr, sc - pc)) for region in regions)
                    if placed[pi] != slot:
                        continue
                    payload = placed[1 - pi]
                    if payload & target["rectangle"]:
                        continue
                    if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1])
                           or int(a[r, c]) != bg for r, c in payload):
                        continue
                    q = _joint(slot, payload)
                    offsets = tuple(sorted((r - q[0], c - q[1]) for r, c in payload))
                    source = key["colors"][pi ^ int(op >= 4)]
                    color = key["colors"][(1 - pi) ^ int(op >= 4)]
                    signature = (target["parts"], q, offsets, source, color)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    candidates.append(dict(target, key=index, pi=pi, op=op, q=q,
                                           offsets=offsets, source=source, target=color,
                                           payload=payload, full=slot | payload))
        _check(bool(candidates), "a plate has no legal mold")
        options.append(candidates)
    solutions, signatures = [], set()

    def search(index, used, rectangles, full, chosen):
        if len(solutions) > 1:
            return
        if index == len(keys):
            if len(used) != len(fragments):
                return
            if len({m["source"] for m in chosen}) != len(chosen):
                return
            signature = tuple((m["key"], tuple(sorted(m["parts"])), m["q"],
                               m["offsets"], m["source"], m["target"]) for m in chosen)
            if signature not in signatures:
                signatures.add(signature)
                solutions.append(tuple(chosen))
            return
        for m in options[index]:
            if m["parts"] & used or m["rectangle"] & rectangles or m["full"] & full:
                continue
            if m["payload"] & rectangles or m["rectangle"] & full:
                continue
            search(index + 1, used | m["parts"], rectangles | m["rectangle"],
                   full | m["full"], chosen + [m])

    search(0, frozenset(), frozenset(), frozenset(), [])
    _check(len(solutions) == 1, "docking plan is missing or ambiguous")
    instruction_cells = _cells(a == neutral) | frozenset().union(*(key["cells"] for key in keys))
    info = {"background": bg, "neutral": neutral, "keys": keys,
            "instruction_cells": instruction_cells, "fragments": fragments}
    return info, solutions[0]


def _render(a, info, records):
    out = a.copy()
    bg, instructions = info["background"], info["instruction_cells"]
    for p in instructions:
        out[p] = bg
    all_anchors, paints, visits = {}, {}, {}
    for m in records:
        anchors = sorted(_cells(a == m["source"]) - instructions)
        _check(bool(anchors), "every decoded source needs an original dot")
        all_anchors[m["source"]] = anchors
        for p in anchors:
            out[p] = bg
    for m in records:
        for anchor in all_anchors[m["source"]]:
            for dr, dc in m["offsets"]:
                p = anchor[0] + dr, anchor[1] + dc
                if not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]):
                    continue
                _check(p not in paints or paints[p] == m["target"], "conflicting printed colors")
                paints[p] = m["target"]
                visits[p] = visits.get(p, 0) + 1
    for p, color in paints.items():
        out[p] = color
    return out, all_anchors, paints, visits


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input without a seed or examples; never mutate the input."""
    a = _grid(input_grid)
    info, moves = _decode(a)
    return _render(a, info, moves)[0]


def _choose(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _sample_shape(rng, small=False, asymmetric=False):
    """A variable separator and a procedurally grown exterior relief."""
    for _ in range(100):
        h = int(rng.integers(3, 6 if small else 7))
        w = int(rng.integers(3, 6 if small else 8))
        y = int(rng.integers(1, h - 1))
        slot = {(y, 0)}
        for c in range(1, w):
            ny = int(np.clip(y + rng.integers(-1, 2), 1, h - 2))
            slot.update((r, c) for r in range(min(y, ny), max(y, ny) + 1))
            y = ny
        for _ in range(int(rng.integers(0, w))):
            frontier = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in D4
                               if 1 <= r + dr < h - 1 and 0 <= c + dc < w} - slot)
            if frontier and rng.random() < .55:
                slot.add(_choose(rng, frontier))
        rectangle = {(r, c) for r in range(h) for c in range(w)}
        fixed = rectangle - slot
        if len(_components(fixed)) != 2 or _rectangle(fixed) != rectangle:
            continue
        if asymmetric and len({tuple(sorted(_transform((slot,), op)[0])) for op in range(8)}) != 8:
            continue
        q = _choose(rng, sorted(p for p in slot if p[1] in (0, w - 1)))
        start = q[0], -1 if q[1] == 0 else w
        payload = {start}
        relation = int(rng.integers(3))
        size = (int(rng.integers(2, len(slot))) if relation == 0 else
                len(slot) if relation == 1 else len(slot) + int(rng.integers(1, 7)))
        size = min(size, 19 if small else 25)
        style, tip = int(rng.integers(3)), start
        for _ in range(size - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in payload for dr, dc in D4
                               if -4 <= r + dr < h + 4 and -4 <= c + dc < w + 4}
                              - rectangle - payload)
            frontier = [p for p in frontier if all(
                (p[0] + dr, p[1] + dc) not in slot or (p[0] + dr, p[1] + dc) == q
                for dr, dc in D4)]
            if not frontier:
                break
            weights = []
            for p in frontier:
                degree = sum((p[0] + dr, p[1] + dc) in payload for dr, dc in D4)
                distance = abs(p[0] - start[0]) + abs(p[1] - start[1])
                weights.append(degree ** 2 / (1 + distance) if style == 0 else
                               (8 if abs(p[0] - tip[0]) + abs(p[1] - tip[1]) == 1 else .2) / degree ** 2
                               if style == 1 else 1 / degree)
            weights = np.asarray(weights, dtype=float)
            tip = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
            payload.add(tip)
        if len(payload) == size:
            return _transform((fixed, slot, payload), int(rng.integers(8)))
    return _safe_shape(int(rng.integers(3)))


def _safe_shape(index):
    w = 4 + index
    rectangle = {(r, c) for r in range(4) for c in range(w)}
    slot = {(1, w - 2), (1, w - 1)} | {(2, c) for c in range(w)}
    payload = {(2, w), (2, w + 1), (3, w + 1)}
    return tuple(frozenset(x) for x in (rectangle - slot, slot, payload))


def _pack(rng, pieces, shape):
    occupied, origins = set(), [None] * len(pieces)
    order = sorted(range(len(pieces)), key=lambda i: -len(_rectangle(pieces[i])) + rng.random())
    for i in order:
        _, r1, _, c1 = _box(pieces[i])
        if r1 >= shape[0] or c1 >= shape[1]:
            return None
        choices = [(r, c) for r in range(shape[0] - r1) for c in range(shape[1] - c1)]
        rng.shuffle(choices)
        if rng.random() < .3:
            choices.sort(key=lambda p: p[0] not in (0, shape[0] - r1 - 1)
                         and p[1] not in (0, shape[1] - c1 - 1))
        for origin in choices:
            box = _shift(_rectangle(pieces[i]), origin)
            if box & occupied:
                continue
            origins[i] = origin
            occupied.update((r + dr, c + dc) for r, c in box for dr, dc in ((0, 0),) + D4)
            break
        else:
            return None
    return origins


def _mapping(rng, colors, count, mode=None):
    colors = [int(c) for c in rng.permutation(colors)]
    mode = (_choose(rng, ("independent", "shared")) if mode == "nonchaining" else
            mode or _choose(rng, ("independent", "chain", "cycle", "shared")))
    if count == 1:
        return [(colors[0], colors[1])]
    if mode == "chain":
        return [(colors[i], colors[i + 1]) for i in range(count)]
    if mode == "cycle":
        return [(colors[i], colors[(i + 1) % count]) for i in range(count)]
    if mode == "shared":
        return [(colors[i], colors[count]) for i in range(count)]
    return [(colors[2 * i], colors[2 * i + 1]) for i in range(count)]


def _footprint(offsets, p, shape):
    return {(p[0] + dr, p[1] + dc) for dr, dc in offsets
            if 0 <= p[0] + dr < shape[0] and 0 <= p[1] + dc < shape[1]}


def _add_dots(rng, a, bg, reserved, records, teaching, clipped):
    """Construct anchors and ground truth from sampled registered reliefs."""
    instructions = _cells(a != bg)
    guard = {(r + dr, c + dc) for r, c in reserved for dr, dc in ((0, 0),) + D4}
    all_cells = [(r, c) for r in range(a.shape[0]) for c in range(a.shape[1])]
    dots, paint, anchors = {}, {}, []

    def place(m, p, requirement=None):
        if p in guard or p in dots or int(a[p]) != bg:
            return False
        if any(dots.get((p[0] + dr, p[1] + dc), m["source"]) != m["source"] for dr, dc in D4):
            return False
        footprint = _footprint(m["offsets"], p, a.shape)
        if not footprint or footprint & reserved:
            return False
        if requirement == "full" and len(footprint) != len(m["offsets"]):
            return False
        if requirement == "clip" and len(footprint) == len(m["offsets"]):
            return False
        if teaching and (p in paint or footprint & set(dots)):
            return False
        if teaching and not (footprint - set(paint)):
            return False
        if any(q in paint and paint[q] != m["target"] for q in footprint):
            return False
        a[p] = m["source"]
        dots[p] = m["source"]
        paint.update({q: m["target"] for q in footprint})
        return True

    for i, m in enumerate(records):
        candidates = list(all_cells)
        rng.shuffle(candidates)
        positions = []
        # A complete copy makes the actual visible shape and registration readable.
        for p in candidates:
            if place(m, p, "full"):
                positions.append(p)
                break
        else:
            raise ValueError("no complete print fits")
        if teaching:
            # Connected source dots must each trigger a print. The joint is
            # absent from the relief, so a tangent neighbor can stay clear.
            r, c = positions[0]
            neighbors = [(r + dr, c + dc) for dr, dc in D4]
            rng.shuffle(neighbors)
            for p in neighbors:
                if 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1] and place(m, p, "full"):
                    positions.append(p)
                    break
            else:
                raise ValueError("no connected two-dot teaching print fits")
        if clipped and i == 0:
            for p in candidates:
                if place(m, p, "clip"):
                    positions.append(p)
                    break
            else:
                raise ValueError("no partial print fits")
        desired = int(rng.integers(2 if teaching else 1, 5 if teaching else 8))
        desired = max(desired, len(positions))
        style = _choose(rng, ("scatter", "scatter", "run", "cluster"))
        step = _choose(rng, D4)
        while len(positions) < desired:
            nearby = []
            if not teaching and style != "scatter":
                nearby = ([(positions[-1][0] + step[0], positions[-1][1] + step[1])]
                          if style == "run" else [(r + dr, c + dc) for r, c in positions for dr, dc in D4])
            for p in nearby + candidates:
                if 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1] and place(m, p):
                    positions.append(p)
                    break
            else:
                break
        _check(len(positions) >= (2 if teaching else 1), "insufficient printing dots")
        anchors.append(positions)
    sources = {m["source"] for m in records}
    inactive = sorted({m["target"] for m in records} - sources)
    for color in inactive:
        candidates = list(all_cells)
        rng.shuffle(candidates)
        required = 2 if teaching else int(rng.integers(1, 5))
        positions = []
        for p in candidates:
            if p in guard or p in dots or p in paint or int(a[p]) != bg:
                continue
            if any(dots.get((p[0] + dr, p[1] + dc), color) != color for dr, dc in D4):
                continue
            a[p] = color
            dots[p] = color
            positions.append(p)
            if len(positions) == required:
                break
        _check(len(positions) == required, "no room for inactive ink dots")
    out = a.copy()
    for p in instructions | set(p for group in anchors for p in group):
        out[p] = bg
    for p, color in paint.items():
        out[p] = color
    return out


def _scene(rng, palette, count, ops, *, witness=None, teaching=False,
           clipped=False, forbidden=frozenset(), safe=False, mapping_mode=None):
    bg, neutral = palette[:2]
    mapping = _mapping(rng, palette[2:], count, mapping_mode)
    shapes = [(_safe_shape(i) if safe else _sample_shape(rng, count >= 3)) for i in range(count)]
    if witness is not None:
        shapes[0] = witness[0]
        pc, hc = witness[1]
        mapping[0] = (pc, hc) if ops[0] < 4 else (hc, pc)
    pieces, systems = [], []
    for shape, op, (source, color) in zip(shapes, ops, mapping):
        fixed, slot, payload = _transform(shape, op)
        q = _joint(slot, payload)
        offsets = tuple(sorted((r - q[0], c - q[1]) for r, c in payload))
        if _registered_signature(offsets) in forbidden:
            raise ValueError("test relief repeats training registration")
        key = _transform(shape[1:], 0)
        keycolors = (source, color) if op < 4 else (color, source)
        pieces.extend((fixed | slot | payload, key[0] | key[1]))
        systems.append((fixed, slot, payload, key, keycolors, source, color, offsets))
    origins = None
    if safe:
        shape = (30, 30)
        tiles = [(r * 10, c * 15) for r in range(3) for c in range(2)]
        rng.shuffle(tiles)
        origins = tiles[:len(pieces)]
    else:
        for attempt in range(10):
            low = 12 if count == 1 else 19 if count == 2 else 26
            shape = ((int(rng.integers(low, 31)), int(rng.integers(low, 31)))
                     if attempt < 7 else (30, 30))
            origins = _pack(rng, pieces, shape)
            if origins is not None:
                break
    _check(origins is not None, "plates do not pack")
    a = np.full(shape, bg, dtype=np.int8)
    reserved, records = set(), []
    for i, (fixed, slot, payload, key, keycolors, source, color, offsets) in enumerate(systems):
        origin, ko = origins[2 * i:2 * i + 2]
        reserved |= _shift(_rectangle(pieces[2 * i]), origin)
        reserved |= _shift(_rectangle(pieces[2 * i + 1]), ko)
        for p in _shift(fixed, origin):
            a[p] = neutral
        for region, ink in zip(key, keycolors):
            for p in _shift(region, ko):
                a[p] = ink
        records.append({"source": source, "target": color, "offsets": offsets,
                        "key_cells": _shift(key[0] | key[1], ko)})
    out = _add_dots(rng, a, bg, reserved, records, teaching, clipped)
    _check(_mode(a) == bg, "wrong input background")
    info, moves = _decode(a)
    actual = {(m["source"], m["target"], m["offsets"], info["keys"][m["key"]]["cells"]) for m in moves}
    expected = {(m["source"], m["target"], m["offsets"], m["key_cells"]) for m in records}
    _check(actual == expected, "visible docking differs from sampled construction")
    _check(np.array_equal(_render(a, info, moves)[0], out), "printing disagrees with construction")
    return {"input": a, "output": out}


def _integer(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(name + " is outside its legal integer range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    _check(num_train > num_test, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_palette = tuple(int(c) for c in rng.permutation(10))
    stable = rng.random() < .35
    # Unequal regions make the teaching plug identity observable; the rest
    # of the family also samples equal-sized plug and relief regions.
    witness_shape = _safe_shape(0)
    for _ in range(20):
        candidate = _sample_shape(rng, asymmetric=True)
        if len(candidate[1]) != len(candidate[2]):
            witness_shape = candidate
            break
    witness = witness_shape, base_palette[2:4]
    puzzle, seen, familiar = {"train": [], "test": []}, set(), set()
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            teaching = split == "train" and i < 2
            count = 1 if teaching else int(rng.integers(1, 4))
            if split == "train" and i == 2:
                count = int(rng.integers(2, 4))
            ops = [int(rng.integers(8)) for _ in range(count)]
            if teaching:
                ops[0] = int(rng.integers(1, 4)) if i == 0 else int(rng.integers(4, 8))
            palette = (base_palette if teaching or stable else tuple(int(c) for c in rng.permutation(10)))
            # Teaching includes a full print, and the mirrored witness also
            # includes a partial one. Other examples mix boundary behavior.
            clipped = i == 1 if teaching else rng.random() < .45
            for attempt in range(45):
                try:
                    pair = _scene(rng, palette, count, ops, witness=witness if teaching else None,
                                  teaching=teaching, clipped=clipped,
                                  forbidden=familiar if split == "test" else frozenset(),
                                  safe=attempt >= 40 and not teaching,
                                  mapping_mode=("chain" if split == "train" and i == 2 else
                                                "nonchaining" if split == "test" and num_train == 2 else None))
                    key = (pair["input"].shape, pair["input"].tobytes())
                    if key in seen:
                        continue
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        _, moves = _decode(pair["input"])
                        familiar.update(_registered_signature(m["offsets"]) for m in moves)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded scene construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _color_policy_output(a, info, moves, policy):
    records = []
    for m in moves:
        colors = info["keys"][m["key"]]["colors"]
        swap = policy[int(m["op"] >= 4)]
        records.append(dict(m, source=colors[m["pi"] ^ swap], target=colors[(1 - m["pi"]) ^ swap]))
    _check(len({m["source"] for m in records}) == len(records), "alternative sources conflict")
    return _render(a, info, records)[0]


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _integer(nt, 2, 10, "training length")
    _integer(ne, 1, 4, "test length")
    _check(nt > ne, "train/test count relationship")
    seen, train_ops, train_shapes, test_shapes = set(), set(), set(), []
    compatible = {(0, 0), (0, 1), (1, 0), (1, 1)}
    geometry_witness, repetition_witness, inactive_witness = False, False, False
    component_witness, empty_origin_witness, train_chain_witness, test_chain = False, False, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _check(a.shape == out.shape and not np.array_equal(a, out), "changed same-size output required")
            key = a.shape, a.astype(np.int8).tobytes()
            _check(key not in seen, "duplicate input")
            seen.add(key)
            info, moves = _decode(a)
            expected, anchors, paints, visits = _render(a, info, moves)
            _check(np.array_equal(expected, out), "incorrect printed output")
            sources = {m["source"] for m in moves}
            chaining = any(m["target"] in sources for m in moves)
            original_anchors = set(p for positions in anchors.values() for p in positions)
            unaffected = (_cells(a != info["background"]) - info["instruction_cells"]
                          - original_anchors - set(paints))
            _check(all(int(out[p]) == int(a[p]) for p in unaffected), "inactive dot was changed")
            _check(all(int(out[p]) == info["background"] for p in info["instruction_cells"] - set(paints)),
                   "reader instruction not erased")
            for m in moves:
                _check((0, 0) not in m["offsets"], "relief cannot occupy the plug registration")
                _check(len(_components(m["slot"])) == 1 and len(_components(m["payload"])) == 1,
                       "disconnected slot or relief")
                _check(_joint(m["slot"], m["payload"]) == m["q"], "wrong joint registration")
                _check(len(_components(m["fixed"])) == 2 and _rectangle(m["fixed"]) == m["rectangle"],
                       "mold does not define a tight two-fragment rectangle")
                signature = _registered_signature(m["offsets"])
                if split == "train":
                    train_shapes.add(signature)
                    train_ops.add(int(m["op"] >= 4))
                else:
                    test_shapes.append(signature)
            if split == "train":
                for policy in list(compatible):
                    try:
                        if not np.array_equal(_color_policy_output(a, info, moves, policy), out):
                            compatible.remove(policy)
                    except ValueError:
                        compatible.remove(policy)
                repetition_witness |= any(len(positions) >= 2 for positions in anchors.values())
                inactive_witness |= bool(unaffected) and any(int(a[p]) not in sources for p in unaffected)
                component_witness |= any(any((r + dr, c + dc) in set(positions)
                                             for r, c in positions for dr, dc in D4)
                                         for positions in anchors.values())
                empty_origin_witness |= any(int(out[p]) == info["background"] for p in original_anchors)
                train_chain_witness |= chaining and any(color in sources for color in paints.values())
                unturned = []
                for m in moves:
                    key_plate = info["keys"][m["key"]]
                    plug, payload = key_plate["regions"][m["pi"]], key_plate["regions"][1 - m["pi"]]
                    q = _joint(plug, payload)
                    unturned.append(dict(m, offsets=tuple(sorted((r - q[0], c - q[1]) for r, c in payload))))
                try:
                    geometry_witness |= not np.array_equal(_render(a, info, unturned)[0], out)
                except ValueError:
                    geometry_witness = True
            else:
                test_chain |= chaining
    _check(train_ops == {0, 1} and compatible == {(0, 1)}, "training does not identify reflection color exchange")
    _check(geometry_witness and repetition_witness and inactive_witness and component_witness and empty_origin_witness,
           "training must expose fitted orientation, per-pixel copying, empty origins, and inactive colors")
    _check(not test_chain or train_chain_witness, "test color chaining lacks simultaneous training evidence")
    _check(all(sig not in train_shapes for sig in test_shapes), "test must introduce registered relief geometry")


def _features(pair):
    a = np.asarray(pair["input"])
    info, moves = _decode(a)
    _, anchors, paints, visits = _render(a, info, moves)
    active = set(p for group in anchors.values() for p in group)
    sources = {m["source"] for m in moves}
    targets = {m["target"] for m in moves}
    graph = ("single" if len(moves) == 1 else "shared" if len(targets) == 1 else
             "cycle" if sources == targets else "chain" if sources & targets else "independent")
    return {"shape": a.shape, "background": info["background"], "neutral": info["neutral"],
            "count": len(moves), "operations": [m["op"] for m in moves], "graph": graph,
            "slots": [_canonical(m["slot"]) for m in moves],
            "reliefs": [_registered_signature(m["offsets"]) for m in moves],
            "relations": [int(np.sign(len(m["slot"]) - len(m["payload"]))) for m in moves],
            "anchor_counts": [len(anchors[m["source"]]) for m in moves],
            "overlap": sum(v > 1 for v in visits.values()),
            "covered_anchors": len(active & set(paints)),
            "clipped": sum(len(_footprint(m["offsets"], p, a.shape)) < len(m["offsets"])
                           for m in moves for p in anchors[m["source"]]),
            "inactive": len(_cells((a != info["background"]) & (a != info["neutral"]))
                            - info["instruction_cells"] - active),
            "symmetric_slots": sum(len({tuple(sorted(_transform((m["slot"],), op)[0]))
                                         for op in range(8)}) < 8 for m in moves)}


def _manual_regression(reflected=False):
    a = np.zeros((18, 18), dtype=np.int8)
    fixed, plug, relief = _safe_shape(0)
    if reflected:
        # Reflect about the rectangle's vertical midline, allowing exterior
        # relief columns to become negative. Input plate stays unchanged.
        fixed = {(r, 3 - c) for r, c in fixed}
    for r, c in fixed:
        a[r + 2, c + (5 if reflected else 2)] = 2
    for r, c in plug:
        a[r + 10, c + 2] = 3
    for r, c in relief:
        a[r + 10, c + 2] = 4
    source, target = (4, 3) if reflected else (3, 4)
    # The ordinary fixture also prints across erased mold cells, exercising
    # valid behavior beyond the sampler's generous reader clearance.
    points = [(1, 12), (5, 12), (8, 1)] if reflected else [(1, 12), (5, 12), (8, 16), (3, 0)]
    for p in points:
        a[p] = source
    a[16, 12] = target
    expected = np.zeros_like(a)
    expected[16, 12] = target
    for r, c in points:
        for dr, dc in ((0, -1), (0, -2), (1, -2)) if reflected else ((0, 1), (0, 2), (1, 2)):
            if 0 <= r + dr < 18 and 0 <= c + dc < 18:
                expected[r + dr, c + dc] = target
    _check(np.array_equal(solve(a), expected), "manual relief/registration regression failed")
    return a, expected


def self_test():
    """42 sequential seeds, aggregate diagnostics only, no disk payloads."""
    for reflected in (False, True):
        _manual_regression(reflected)
    dimensions, slots, reliefs, operations, backgrounds, neutrals = set(), set(), set(), set(), set(), set()
    graphs, counts, relations, signatures, anchor_counts = set(), set(), set(), set(), set()
    totals = {k: 0 for k in ("pairs", "overlap_pairs", "clipped_pairs", "inactive_pairs",
                             "covered_anchor_pairs", "symmetric_slot_pairs")}
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                f = _features(pair)
                digest.update(bytes(pair["input"].shape))
                digest.update(pair["input"].tobytes())
                dimensions.add(f["shape"])
                backgrounds.add(f["background"])
                neutrals.add(f["neutral"])
                slots.update(f["slots"])
                reliefs.update(f["reliefs"])
                operations.update(f["operations"])
                relations.update(f["relations"])
                counts.add(f["count"])
                graphs.add(f["graph"])
                anchor_counts.update(f["anchor_counts"])
                totals["pairs"] += 1
                for name, feature in (("overlap_pairs", "overlap"), ("clipped_pairs", "clipped"),
                                      ("inactive_pairs", "inactive"), ("covered_anchor_pairs", "covered_anchors"),
                                      ("symmetric_slot_pairs", "symmetric_slots")):
                    totals[name] += bool(f[feature])
        signatures.add(digest.hexdigest())
    _check(len(signatures) == 42 and len(dimensions) >= 35, "insufficient puzzle/layout variation")
    _check(len(slots) >= 35 and len(reliefs) >= 60, "insufficient structural variation")
    _check(operations == set(range(8)) and counts == {1, 2, 3} and relations == {-1, 0, 1},
           "missing operation/count/region-size regimes")
    _check(len(backgrounds) >= 8 and len(neutrals) >= 8 and max(anchor_counts) >= 5,
           "missing palette or dot-count variation")
    _check(graphs == {"single", "chain", "cycle", "shared", "independent"}, "missing local mapping regimes")
    _check(all(totals[k] > 0 for k in totals), "missing overlap/clipping/preservation/symmetry witnesses")
    first = generate_puzzle(19, 2, 1)
    same = generate_puzzle(19, 2, 1)
    _check(all(np.array_equal(x[k], y[k]) for s in first for x, y in zip(first[s], same[s])
               for k in ("input", "output")), "nondeterministic puzzle")
    for nt, ne in ((10, 4), (5, 3), (3, 2)):
        validate_generated_puzzle(generate_puzzle(73, nt, ne))
    rng = np.random.default_rng(211)
    for count in (1, 2, 3):
        pair = _scene(rng, tuple(range(10)), count, list(range(count)), safe=True,
                      mapping_mode="chain")
        _check(np.array_equal(solve(pair["input"]), pair["output"]), "compact fallback failed")
    for split in first:
        for pair in first[split]:
            frozen = pair["input"].copy()
            frozen.flags.writeable = False
            _check(np.array_equal(solve(frozen), pair["output"]), "read-only input failed")
            _check(np.array_equal(solve(frozen.tolist()), pair["output"]), "nested-list input failed")
    validate_generated_puzzle({s: [{k: v.tolist() for k, v in pair.items()} for pair in first[s]]
                               for s in first})
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    ambiguous = np.zeros((16, 20), dtype=np.int8)
    ambiguous[2, 2:6] = ambiguous[4, 2:6] = 2
    ambiguous[10, 4:8] = 3
    ambiguous[10:12, 8] = 4
    ambiguous[0, 15], ambiguous[14, 15] = 3, 4
    try:
        solve(ambiguous)
    except ValueError as exc:
        _check("ambiguous" in str(exc), "wrong ambiguity rejection")
    else:
        raise AssertionError("ambiguous mirrored printer accepted")
    return dict(totals, seeds=42, distinct_puzzles=len(signatures), dimensions=len(dimensions),
                slot_shapes=len(slots), registered_reliefs=len(reliefs), operations=sorted(operations),
                printer_counts=sorted(counts), region_size_relations=sorted(relations),
                backgrounds=len(backgrounds), neutral_colors=len(neutrals), graphs=sorted(graphs),
                anchor_count_range=[min(anchor_counts), max(anchor_counts)])


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
