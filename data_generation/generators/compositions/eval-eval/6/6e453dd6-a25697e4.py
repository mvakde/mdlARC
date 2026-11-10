"""Mirror Foundry: eligible mold cavities recruit rigid two-colored keys.

Pure input-only solve; no evidence imports, files, caches, or generation metadata.
See task_spec.md for the intended rule and explicit geometric domain.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "ARC color bounds")
    return a.astype(np.int8, copy=True)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(frozenset(part))
    return result


def _cells(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _moat(cells):
    return set(cells) | {(r + dr, c + dc) for r, c in cells for dr, dc in DIRS}


def _transform(regions, op):
    """Reflect columns if op>=4, then rotate; normalize the complete object."""
    transformed = []
    for region in regions:
        new = []
        for r, c in region:
            if op >= 4:
                c = -c
            for _ in range(op % 4):
                r, c = -c, r
            new.append((r, c))
        transformed.append(new)
    all_points = [p for region in transformed for p in region]
    r0, _, c0, _ = _bbox(all_points)
    return tuple(_shift(region, -r0, -c0) for region in transformed)


def _signature(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted(_shift(cells, -r0, -c0)))


def _orbit(cells):
    return {_signature(_transform((cells,), op)[0]) for op in range(8)}


def _canonical(cells):
    return min(_orbit(cells))


def _negative_regions(cells):
    r0, r1, c0, c1 = _bbox(cells)
    empty = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} - set(cells)
    result = []
    for pocket in _components(empty):
        closed = not any(r in (r0, r1) or c in (c0, c1) for r, c in pocket)
        active = closed and any(c == c1 - 1 for r, c in pocket)
        if active:
            _require({r for r, c in pocket} == {r for r, c in pocket if c == c1 - 1},
                     "partial-contact active hole is outside the family")
        result.append({"cells": pocket, "kind": "active" if active else "recessed" if closed else "open"})
    return result


def _scene(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    order = np.argsort(counts)
    _require(len(order) >= 4 and counts[order[-1]] > counts[order[-2]], "unique modal background")
    bg = int(colors[order[-1]])
    candidates = []
    for turn in range(4):
        g = np.rot90(a, turn).copy()
        h, w = g.shape
        for q in range(1, w - 1):
            rail = int(g[0, q])
            if rail == bg or not np.all(g[:, q] == rail) or np.count_nonzero(g == rail) != h:
                continue
            left = _components(_cells(g[:, :q] != bg))
            right = _components(_shift(_cells(g[:, q + 1:] != bg), 0, q + 1))
            if not left or not right:
                continue
            if any(len({int(g[p]) for p in part}) != 1 for part in left):
                continue
            if any(len({int(g[p]) for p in part}) != 2 for part in right):
                continue
            molds, keys = [], []
            for part in left:
                r0, r1, c0, c1 = _bbox(part)
                molds.append({"cells": part, "color": int(g[min(part)]), "bbox": (r0, r1, c0, c1),
                              "shift": q - 1 - c1, "pockets": _negative_regions(part)})
            good = True
            for part in right:
                palette = tuple(sorted({int(g[p]) for p in part}))
                regions = tuple(frozenset(p for p in part if int(g[p]) == color) for color in palette)
                if any(len(_components(region)) != 1 for region in regions):
                    good = False
                    break
                keys.append({"cells": part, "colors": palette, "regions": regions})
            if good:
                candidates.append({"grid": g, "turn": turn, "bg": bg, "rail": rail,
                                   "q": q, "molds": molds, "keys": keys})
    _require(len(candidates) == 1, "unique rail and component-defined sides required")
    return candidates[0]


def _slots(info, activation="normal"):
    slots = []
    for mi, mold in enumerate(info["molds"]):
        for pocket in mold["pockets"]:
            kind = pocket["kind"]
            if kind != "active" and not (activation == "enclosed" and kind == "recessed") and activation != "all":
                continue
            shadow = frozenset((r, 2 * info["q"] - c - mold["shift"]) for r, c in pocket["cells"])
            slots.append({"cells": shadow, "mold": mi, "kind": kind})
    return slots


def _fit(info, activation="normal", swap_policy=(0, 1)):
    """Enumerate a unique full colored output, including conserved spare keys."""
    g, bg, q = info["grid"], info["bg"], info["q"]
    h, w = g.shape
    keys, slots = info["keys"], _slots(info, activation)
    _require(len(slots) <= len(keys), "not enough keys")
    possibilities = []
    for si, slot in enumerate(slots):
        target = slot["cells"]
        sr0, sr1, sc0, sc1 = _bbox(target)
        options, seen = [], set()
        for ki, key in enumerate(keys):
            for op in range(8):
                regions = _transform(key["regions"], op)
                for pi in (0, 1):
                    if len(regions[pi]) != len(target):
                        continue
                    r0, _, c0, _ = _bbox(regions[pi])
                    placed = tuple(_shift(region, sr0 - r0, sc0 - c0) for region in regions)
                    if placed[pi] != target:
                        continue
                    other = placed[1 - pi]
                    if any(sr0 <= r <= sr1 and sc0 <= c <= sc1 for r, c in other):
                        continue
                    full = placed[0] | placed[1]
                    if any(not (0 <= r < h and q < c < w) for r, c in full):
                        continue
                    if any(int(g[p]) != bg for p in full):
                        continue
                    palette = key["colors"][::-1] if swap_policy[int(op >= 4)] else key["colors"]
                    paint = tuple(sorted((r, c, color) for region, color in zip(placed, palette) for r, c in region))
                    signature = (ki, paint)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    options.append({"slot": si, "key": ki, "op": op, "plug_index": pi,
                                    "full": full, "paint": paint, "plug": placed[pi]})
        _require(bool(options), "socket has no legal whole-key fit")
        possibilities.append(options)
    base = g.copy()
    for mold in info["molds"]:
        for p in mold["cells"]:
            base[p] = bg
    for mold in info["molds"]:
        for p in _shift(mold["cells"], 0, mold["shift"]):
            _require(base[p] == bg, "docked molds overlap")
            base[p] = mold["color"]
    order = sorted(range(len(slots)), key=lambda i: len(possibilities[i]))
    answers = {}

    def search(depth, used, occupied, moves):
        if len(answers) > 1:
            return
        if depth == len(order):
            result = base.copy()
            for ki in used:
                for p in keys[ki]["cells"]:
                    result[p] = bg
            for move in moves:
                for r, c, color in move["paint"]:
                    result[r, c] = color
            answers.setdefault(result.tobytes(), (result, list(moves)))
            return
        for move in possibilities[order[depth]]:
            if move["key"] not in used and not (move["full"] & occupied):
                search(depth + 1, used | {move["key"]}, occupied | move["full"], moves + [move])

    search(0, set(), frozenset(), [])
    _require(len(answers) == 1, "complete colored key assignment must be unique")
    result, moves = next(iter(answers.values()))
    return result, moves, slots


def _derive(value, activation="normal", swap_policy=(0, 1)):
    info = _scene(value)
    result, moves, slots = _fit(info, activation, swap_policy)
    return np.rot90(result, -info["turn"]).copy(), info, moves, slots


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the visible input and fixed family rule."""
    return _derive(input_grid)[0]


def _pocket(rng, kind, forbidden, target_area=None, small=False):
    for _ in range(200):
        height = int(rng.integers(2, 4 if small else 5))
        widths = list(map(int, rng.integers(1, 4 if small else 5, size=height)))
        cells = {(r, c) for r, width in enumerate(widths) for c in range(1 - width, 1)}
        if kind == "open":
            cells.add((int(rng.integers(height)), 1))
        if target_area is not None and len(cells) != target_area:
            continue
        if target_area is None and not 4 <= len(cells) <= (8 if small else 13):
            continue
        signature = _canonical(cells)
        if len(_orbit(cells)) == 8 and signature not in forbidden:
            return {"kind": kind, "cells": frozenset(cells), "height": height, "width": max(widths)}
    raise ValueError("pocket silhouette sampling exhausted")


def _mold(rng, pockets, tail=False):
    width = max(p["width"] + (4 if p["kind"] == "recessed" else 2) for p in pockets)
    width += int(rng.integers(2))
    gap = int(rng.integers(1, 3))
    height = 2 + sum(p["height"] for p in pockets) + gap * (len(pockets) - 1)
    mask = np.ones((height, width), dtype=bool)
    result, row = [], 1
    for pocket in pockets:
        right = width - (3 + int(rng.integers(2)) if pocket["kind"] == "recessed" else 2)
        cells = _shift(pocket["cells"], row, right)
        for p in cells:
            mask[p] = False
        result.append({**pocket, "cells": cells})
        row += pocket["height"] + gap
    if tail and len(pockets) == 1 and pockets[0]["kind"] == "recessed":
        # A thin protrusion rather than a thick wall defines the leading extreme.
        end = max(c for r, c in result[0]["cells"]) + 2
        mask[:, end:] = False
        mask[int(rng.choice([0, height - 1])), end:] = True
    elif rng.random() < .45:
        # Safe exterior erosion, disjoint from pockets and the leading wall.
        mask[0, :int(rng.integers(1, 3))] = False
        if rng.random() < .5:
            mask[-1, 0] = False
    occupied = _cells(mask)
    _require(len(_components(occupied)) == 1, "constructed mold disconnected")
    actual = {p["cells"]: p["kind"] for p in _negative_regions(occupied)}
    _require(all(actual.get(p["cells"]) == p["kind"] for p in result), "constructed cavity classification")
    return mask, result


def _handle(rng, slot, all_slots, occupied, shape, q, size):
    h, w = shape
    r0, r1, c0, c1 = _bbox(slot)
    blocked = _moat(occupied | (all_slots - set(slot)))

    def legal(p):
        r, c = p
        return (0 <= r < h and q < c < w and p not in blocked
                and not (r0 <= r <= r1 and c0 <= c <= c1)
                and r0 - 4 <= r <= r1 + 4 and c0 - 3 <= c <= c1 + 6)

    starts = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in DIRS if legal((r + dr, c + dc))})
    if not starts:
        raise ValueError("no exterior attachment")
    start = starts[int(rng.integers(len(starts)))]
    handle, tip = {start}, start
    style = int(rng.integers(3))
    while len(handle) < size:
        frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in DIRS
                           if legal((r + dr, c + dc))} - handle)
        if not frontier:
            raise ValueError("attachment growth blocked")
        weights = []
        for r, c in frontier:
            neighbors = sum((r + dr, c + dc) in handle for dr, dc in DIRS)
            dist = abs(r - start[0]) + abs(c - start[1])
            weights.append(neighbors ** 2 / (1 + dist) if style == 0 else
                           (8 if abs(r - tip[0]) + abs(c - tip[1]) == 1 else .2) / neighbors ** 2
                           if style == 1 else 1 / neighbors)
        probs = np.asarray(weights, dtype=float)
        tip = frontier[int(rng.choice(len(frontier), p=probs / probs.sum()))]
        handle.add(tip)
    return frozenset(handle)


def _pack(rng, regions, shape, q, reserved):
    h, w = shape
    occupied = _moat(reserved)
    origins = [None] * len(regions)
    order = sorted(range(len(regions)), key=lambda i: -len(regions[i]))
    for i in order:
        cells = regions[i]
        _, r1, _, c1 = _bbox(cells)
        choices = [(r, c) for r in range(h - r1) for c in range(q + 1, w - c1)]
        rng.shuffle(choices)
        for r, c in choices:
            placed = _shift(cells, r, c)
            if not placed & occupied:
                origins[i] = (r, c)
                occupied.update(_moat(placed))
                break
        else:
            raise ValueError("loose keys do not pack")
    return origins


def _build_pair(rng, witness, is_test, palette, turn, small=False):
    if witness == 0:
        groups = [["active"], ["open"]]
    elif witness == 1:
        groups = [["active", "recessed"]]
    elif is_test:
        groups = [["active", "active", "recessed"]]
        if rng.random() < (.15 if small else .5):
            groups.append(["open"])
    else:
        groups = [["active"]]
        if rng.random() < .5:
            groups.append(["active"])
        for _ in range(int(rng.integers(1, 3))):
            kind = str(rng.choice(["open", "recessed"]))
            if rng.random() < .5:
                groups[int(rng.integers(len(groups)))].append(kind)
            else:
                groups.append([kind])
    forbidden, molds, pocket_order = set(), [], []
    first_area = None
    for group in groups:
        pockets = []
        for kind in group:
            target_area = first_area if witness == 0 and kind == "open" else None
            p = _pocket(rng, kind, forbidden, target_area, small)
            if first_area is None:
                first_area = len(p["cells"])
            forbidden.add(_canonical(p["cells"]))
            pockets.append(p)
        mask, pockets = _mold(rng, pockets, tail=witness not in (0, 1) and rng.random() < .5)
        molds.append((mask, pockets))
    if witness is None and not is_test and len(molds) < 3 and rng.random() < .18:
        mask = np.ones((int(rng.integers(2, 4)), int(rng.integers(2, 5))), dtype=bool)
        molds.append((mask, []))
    rng.shuffle(molds)
    base_h = sum(m.shape[0] for m, _ in molds) + len(molds) - 1
    if base_h > 27:
        raise ValueError("molds too tall")
    height = int(rng.integers(max(14, base_h + 1), 31))
    widest = max(m.shape[1] for m, _ in molds)
    q = widest + int(rng.integers(1, 4))
    width = int(rng.integers(max(24, q + 15), 31))
    bg, rail = palette[:2]
    object_colors = palette[2:]
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[:, q] = rail
    out = inp.copy()
    extras = rng.multinomial(height - base_h, np.ones(len(molds) + 1) / (len(molds) + 1))
    row = int(extras[0])
    for i, (mask, pockets) in enumerate(molds):
        shift = int(rng.integers(q - mask.shape[1] + 1))
        if witness in (0, 1) and any(p["kind"] == "active" for p in pockets):
            shift = int(rng.integers(1, q - mask.shape[1] + 1))
        color = int(rng.choice(object_colors))
        col = q - mask.shape[1] - shift
        for r, c in _cells(mask):
            inp[row + r, col + c] = color
            out[row + r, col + c + shift] = color
        for p in pockets:
            # Direct construction of the final reflected negative-space shape.
            slot = frozenset((row + r, 2 * q - (col + c + shift)) for r, c in p["cells"])
            pocket_order.append({**p, "slot": slot})
        row += mask.shape[0] + 1 + int(extras[i + 1])
    all_slots = set().union(*(p["slot"] for p in pocket_order))
    final_occupied, key_data = set(), []
    for pocket in pocket_order:
        slot = pocket["slot"]
        area = len(slot)
        if witness == 0 and pocket["kind"] == "active":
            size = area + int(rng.integers(1, 4))
        elif witness == 1 and pocket["kind"] == "active":
            size = int(rng.integers(2, area))
        else:
            relationship = int(rng.integers(3))
            size = (int(rng.integers(2, area)) if relationship == 0 else area if relationship == 1
                    else area + int(rng.integers(1, 4)))
        handle = _handle(rng, slot, all_slots, final_occupied, inp.shape, q, size)
        final_occupied.update(slot | handle)
        op = (int(rng.integers(1, 4)) if witness == 0 and pocket["kind"] == "active" else
              int(rng.integers(4, 8)) if witness == 1 and pocket["kind"] == "active" else
              int(rng.integers(8)))
        source = _transform((slot, handle), op)
        colors = tuple(map(int, rng.choice(object_colors, 2, replace=False)))
        key_data.append({"source": source, "colors": colors, "slot": slot, "handle": handle,
                         "op": op, "active": pocket["kind"] == "active"})
    origins = _pack(rng, [k["source"][0] | k["source"][1] for k in key_data], inp.shape, q, final_occupied)
    for key, (r, c) in zip(key_data, origins):
        for region, color in zip(key["source"], key["colors"]):
            for p in _shift(region, r, c):
                inp[p] = color
                if not key["active"]:
                    out[p] = color
        if key["active"]:
            colors = key["colors"][::-1] if key["op"] >= 4 else key["colors"]
            for region, color in zip((key["slot"], key["handle"]), colors):
                for p in region:
                    out[p] = color
    return {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}


def _array_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, train > test")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.integers(2))
    palette = tuple(map(int, rng.permutation(10)))
    orientation = int(rng.integers(2))
    base_turn = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            witness = index if split == "train" and index < 2 else None
            turn = base_turn if orientation == 0 else int(rng.integers(4))
            colors = palette if stable else tuple(map(int, rng.permutation(10)))
            for attempt in range(80):
                try:
                    pair = _build_pair(rng, witness, split == "test", colors, turn, small=attempt >= 40)
                    signature = _array_key(pair["input"])
                    if signature in seen:
                        continue
                    predicted = solve(pair["input"])
                    if not np.array_equal(predicted, pair["output"]):
                        continue
                    # The two teaching witnesses must have usable concrete
                    # counterfactual arrangements, not just missing-fit errors.
                    if witness is not None:
                        alt = _derive(pair["input"], "all" if witness == 0 else "enclosed")[0]
                        if np.array_equal(alt, pair["output"]):
                            continue
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
                except (AssertionError, ValueError):
                    continue
            else:
                raise RuntimeError("bounded scene construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    predicted, info, moves, slots = _derive(pair["input"])
    molds, keys = info["molds"], info["keys"]
    used = {m["key"] for m in moves}
    active_per_mold = Counter(s["mold"] for s in slots)
    centers = [np.mean(list(s["cells"]), axis=0) for s in slots]
    misleading = 0
    for m in moves:
        center = np.mean(list(keys[m["key"]]["cells"]), axis=0)
        own = float(np.abs(center - centers[m["slot"]]).sum())
        misleading += any(float(np.abs(center - c).sum()) < own for i, c in enumerate(centers) if i != m["slot"])
    return {"prediction": predicted, "info": info, "moves": moves, "slots": slots,
            "shape": tuple(pair["input"].shape), "turn": info["turn"], "background": info["bg"],
            "molds": len(molds), "keys": len(keys), "moved": len(moves), "spares": len(keys) - len(used),
            "shifts": tuple(m["shift"] for m in molds), "misleading_proximity": misleading,
            "kinds": Counter(p["kind"] for mold in molds for p in mold["pockets"]),
            "max_active_in_mold": max(active_per_mold.values(), default=0),
            "relations": tuple(int(np.sign(len(keys[m["key"]]["regions"][m["plug_index"]]) -
                                              len(keys[m["key"]]["regions"][1 - m["plug_index"]]))) for m in moves)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "same-size transformation")
            signature = _array_key(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            f = _features({"input": inp, "output": out})
            _require(np.array_equal(f["prediction"], out), "output differs from unique input-only solution")
            _require(not np.array_equal(inp, out), "generation must show a change")
            info = f["info"]
            gout = np.rot90(out, info["turn"])
            _require(np.all(gout[:, info["q"]] == info["rail"]), "rail changed")
            last = -2
            for mold in info["molds"]:
                r0, r1, _, _ = mold["bbox"]
                _require(r0 >= last + 2, "mold bands must be separated")
                last = r1
                for p in _shift(mold["cells"], 0, mold["shift"]):
                    _require(gout[p] == mold["color"], "mold shape/color not preserved")
            used = {m["key"] for m in f["moves"]}
            for ki, key in enumerate(info["keys"]):
                for p in key["cells"]:
                    _require(gout[p] == (info["bg"] if ki in used else info["grid"][p]), "spare/source-key conservation")
            _require(set(map(int, np.unique(inp))) == set(map(int, np.unique(out))), "palette conservation")
            if split == "train":
                train_features.append(f)
    _require({int(m["op"] >= 4) for f in train_features for m in f["moves"]} == {0, 1}, "training parity coverage")
    _require(any(f["kinds"]["open"] for f in train_features), "open contrast missing")
    _require(any(f["kinds"]["recessed"] for f in train_features), "recessed contrast missing")
    _require(any(max(f["shifts"]) > 0 for f in train_features), "docking evidence missing")
    _require({-1, 1} <= {r for f in train_features for r in f["relations"]}, "plug-size contrasts missing")
    for activation, policy in (("all", (0, 1)), ("enclosed", (0, 1)), ("normal", (0, 0)), ("normal", (1, 1))):
        differs = False
        for pair in puzzle["train"]:
            try:
                alternative = _derive(pair["input"], activation, policy)[0]
            except AssertionError:
                continue
            if not np.array_equal(alternative, pair["output"]):
                differs = True
                break
        _require(differs, "training fails a parent-concept omission check")


def self_test():
    """Bounded 42-seed audit; no files, serialized puzzles, or retained traces."""
    structures, shapes, cavities, handles, palettes, hashes = set(), set(), set(), set(), set(), set()
    counts, parities, operations, relations, turns, shifts = set(), set(), set(), set(), set(), set()
    total = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                info = f["info"]
                digest.update(pair["input"].tobytes())
                shapes.add(f["shape"])
                colors = {}
                normalized = bytes(colors.setdefault(int(c), len(colors)) for c in pair["input"].flat)
                structures.add((f["shape"], hashlib.sha256(normalized).digest()))
                palettes.add((info["bg"], info["rail"]))
                counts.add((f["molds"], f["keys"], f["moved"]))
                turns.add(info["turn"])
                shifts.update(f["shifts"])
                relations.update(f["relations"])
                total["pairs"] += 1
                total["spares"] += f["spares"]
                total["open"] += f["kinds"]["open"]
                total["recessed"] += f["kinds"]["recessed"]
                total["misleading_proximity"] += f["misleading_proximity"]
                total["zero_shift"] += 0 in f["shifts"]
                if split == "test":
                    _require(f["max_active_in_mold"] >= 2, "grouping shift absent")
                else:
                    _require(f["max_active_in_mold"] <= 1, "training consumed grouping shift")
                for slot in f["slots"]:
                    cavities.add(_canonical(slot["cells"]))
                for m in f["moves"]:
                    operations.add(m["op"])
                    parities.add(int(m["op"] >= 4))
                    key = info["keys"][m["key"]]
                    handles.add(_canonical(key["regions"][1 - m["plug_index"]]))
        hashes.add(digest.digest())
    _require(len(hashes) == 42 and len(structures) >= 190, "cross-seed structural diversity")
    _require(len(shapes) >= 35 and len(cavities) >= 40 and len(handles) >= 60, "geometry diversity")
    _require(len(palettes) >= 30 and len(counts) >= 7, "palette/count diversity")
    _require(turns == set(range(4)) and operations == set(range(8)), "direction/operation diversity")
    _require(relations == {-1, 0, 1} and parities == {0, 1} and len(shifts) >= 5, "role/shift diversity")
    _require(total["zero_shift"] > 10 and total["misleading_proximity"] > 0, "aligned/proximity cases")
    p = generate_puzzle(987654, 2, 1)
    same = generate_puzzle(987654, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], same[s])
                 for k in ("input", "output")), "determinism")
    pair = p["test"][0]
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn)), "rotation equivariance")
    lookup = np.array([7, 4, 9, 1, 6, 0, 8, 2, 5, 3], dtype=np.int8)
    _require(np.array_equal(solve(lookup[pair["input"]]), lookup[pair["output"]]), "palette equivariance")
    # Two identical source keys in different places yield two different
    # conserved-spare/erased-source outputs. Neither is an admissible tie-break.
    ambiguous = np.zeros((22, 29), dtype=np.int8)
    ambiguous[:, 7] = 1
    body = {(r, c) for r in range(2, 7) for c in range(1, 6)}
    hole = {(3 + r, c) for r, width in enumerate((1, 3, 2)) for c in range(5 - width, 5)}
    for point in body - hole:
        ambiguous[point] = 2
    shadow = frozenset((r, 14 - (c + 1)) for r, c in hole)
    extra = {(4, 12), (4, 13)}
    source = _transform((shadow, extra), 1)
    for dr, dc in ((10, 16), (16, 21)):
        for region, color in zip(source, (3, 4)):
            for point in _shift(region, dr, dc):
                ambiguous[point] = color
    try:
        solve(ambiguous)
    except AssertionError as error:
        _require("assignment must be unique" in str(error), "ambiguity fixture failed for another reason")
    else:
        raise AssertionError("competing colored outputs accepted")
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        q = generate_puzzle(2026, nt, ne)
        _require((len(q["train"]), len(q["test"])) == (nt, ne), "explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad label accepted")
    return {"seeds": 42, "pairs": total["pairs"], "structures": len(structures), "shapes": len(shapes),
            "cavity_shapes": len(cavities), "handle_shapes": len(handles), "palette_roles": len(palettes),
            "count_combinations": len(counts), "operations": sorted(operations), "measurements": dict(total),
            "independent_solver": False}
