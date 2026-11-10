"""Rigid seals, a bounded color inventory, and conditional container dissolution.

No parent imports, external assets, stored grids, or generation state are used
by solve. All random choices use the caller's local NumPy Generator.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")
    return int(a), int(b)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
    return g


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, part = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    part.add(p)
        result.append(frozenset(part))
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _rect(box):
    r0, r1, c0, c1 = box
    return frozenset((r, c) for r in range(r0, r1) for c in range(c0, c1))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _transform(regions, op):
    """Reflect left/right first for op >= 4, then rotate, then normalize jointly."""
    result = []
    for region in regions:
        part = []
        for r, c in region:
            if op >= 4:
                c = -c
            for _ in range(op % 4):
                r, c = c, -r
            part.append((r, c))
        result.append(part)
    r0, _, c0, _ = _bbox([p for part in result for p in part])
    return tuple(_shift(part, -r0, -c0) for part in result)


def _shape_key(cells):
    return min(tuple(sorted(_transform((cells,), op)[0])) for op in range(8))


def _corner(g, corner):
    if corner & 1:
        g = g[:, ::-1]
    if corner & 2:
        g = g[::-1, :]
    return g


def _legend(g):
    candidates = []
    if min(g.shape) < 5:
        raise AssertionError("no room for a legend")
    for corner in range(4):
        v = _corner(g, corner)
        material = int(v[3, 3])
        if not (np.all(v[3, :4] == material) and np.all(v[:4, 3] == material)):
            continue
        colors, counts = np.unique(v[:3, :3], return_counts=True)
        pads = colors[counts == 6]
        if len(pads) != 1:
            continue
        bg = int(pads[0])
        query = frozenset(int(c) for c in colors if c != bg)
        if (len(query) == 3 and material not in query and material != bg
                and np.all(v[4, :5] == bg) and np.all(v[:4, 4] == bg)):
            candidates.append((corner, bg, material, query))
    assert len(candidates) == 1, "legend must be unique"
    return candidates[0]


def _read(g):
    corner, bg, material, query = _legend(g)
    v = _corner(g, corner)
    mask = v != bg
    mask = mask.copy()
    mask[:4, :4] = False
    parts = _components(tuple(map(tuple, np.argwhere(mask))))
    fragments, keys = [], []
    for cells in parts:
        colors = sorted({int(v[p]) for p in cells})
        if material in colors:
            fragments.append(cells)
        else:
            assert len(colors) == 2, "loose keys have exactly two colors"
            regions = tuple(frozenset(p for p in cells if v[p] == color) for color in colors)
            assert all(len(_components(s)) == 1 for s in regions)
            keys.append({"cells": cells, "colors": tuple(colors), "regions": regions})
    assert 1 <= len(keys) <= 4 and len(fragments) == 2 * len(keys)
    records = []
    for i, a in enumerate(fragments):
        for j in range(i + 1, len(fragments)):
            body = a | fragments[j]
            box = _bbox(body)
            rectangle = _rect(box)
            if rectangle & _rect((0, 4, 0, 4)):
                continue
            if frozenset(p for p in rectangle if v[p] != bg) != body:
                continue
            slot = rectangle - body
            if len(slot) < 3 or len(_components(slot)) != 1:
                continue
            records.append({"parts": frozenset((i, j)), "box": box,
                            "rectangle": rectangle, "body": body, "slot": slot})
    return {"view": v, "corner": corner, "bg": bg, "material": material,
            "query": query, "fragments": fragments, "keys": keys, "records": records}


def _assembly(g):
    """Enumerate exact whole-key fits and require a unique colored assembly.

    Membership never chooses between geometric fits. Ambiguity is rejected
    before any rectangle is erased, even if erasure would hide that ambiguity.
    """
    info = _read(g)
    v, bg = info["view"], info["bg"]
    choices = []
    for ki, key in enumerate(info["keys"]):
        possible, signatures = [], set()
        for record in info["records"]:
            sr, _, sc, _ = _bbox(record["slot"])
            for op in range(8):
                regions = _transform(key["regions"], op)
                for pi in (0, 1):
                    pr, _, pc, _ = _bbox(regions[pi])
                    placed = tuple(_shift(s, sr - pr, sc - pc) for s in regions)
                    if placed[pi] != record["slot"]:
                        continue
                    outside = placed[1 - pi]
                    if outside & record["rectangle"]:
                        continue
                    if any(not (0 <= r < v.shape[0] and 0 <= c < v.shape[1])
                           or v[r, c] != bg or (r < 4 and c < 4) for r, c in outside):
                        continue
                    colors = key["colors"][::-1] if op >= 4 else key["colors"]
                    paint = tuple(sorted((p, color) for region, color in zip(placed, colors) for p in region))
                    signature = record["box"], paint
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    possible.append({**record, "key": ki, "op": op, "plug_index": pi,
                                     "regions": placed, "colors": colors, "paint": paint,
                                     "outside": outside, "full": placed[0] | placed[1]})
        assert possible, "a loose key has no legal placement"
        choices.append(possible)
    solutions = {}
    order = sorted(range(len(choices)), key=lambda k: len(choices[k]))

    def visit(n, used_parts, rectangles, occupied, moves):
        if len(solutions) > 1:
            return
        if n == len(order):
            assert len(used_parts) == len(info["fragments"])
            signature = tuple(sorted((m["box"], m["paint"]) for m in moves))
            solutions.setdefault(signature, tuple(moves))
            return
        for m in choices[order[n]]:
            if (m["parts"] & used_parts or m["rectangle"] & rectangles
                    or m["full"] & occupied or m["full"] & rectangles
                    or m["rectangle"] & occupied):
                continue
            visit(n + 1, used_parts | m["parts"], rectangles | m["rectangle"],
                  occupied | m["full"], moves + [m])

    visit(0, frozenset(), frozenset(), frozenset(), [])
    assert len(solutions) == 1, "complete colored assembly must be unique"
    moves = list(next(iter(solutions.values())))
    assembled = v.copy()
    for key in info["keys"]:
        for p in key["cells"]:
            assembled[p] = bg
    for move in moves:
        for p, color in move["paint"]:
            assembled[p] = color
        counts = Counter(int(assembled[p]) for p in move["rectangle"])
        labels = frozenset(set(counts) - {bg, info["material"]})
        move["counts"] = counts
        move["labels"] = labels
        move["positive"] = info["query"] <= labels
        move["body_labels"] = frozenset(int(v[p]) for p in move["body"]) - {info["material"]}
    info["moves"] = moves
    info["assembled"] = assembled
    return info


def _finish(info, *, erase=True, count_outside=False, swap_policy=None):
    out = info["assembled"].copy()
    if swap_policy is not None:
        # Only used for explicit semantic ablations in validation/self_test.
        for m in info["moves"]:
            colors = info["keys"][m["key"]]["colors"]
            if swap_policy[int(m["op"] >= 4)]:
                colors = colors[::-1]
            for s, color in zip(m["regions"], colors):
                for p in s:
                    out[p] = color
    if erase:
        for m in info["moves"]:
            scope = m["rectangle"] | m["outside"] if count_outside else m["rectangle"]
            labels = {int(out[p]) for p in scope} - {info["bg"], info["material"]}
            if info["query"] <= labels:
                for p in m["rectangle"]:
                    out[p] = info["bg"]
    return _corner(out, info["corner"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input cells only; preserve its dimensions and legend."""
    return _finish(_assembly(_grid(input_grid)))


def _fallback_shape(index):
    w = 4 + index
    rectangle = _rect((0, 4, 0, w))
    slot = frozenset({(1, w - 2), (1, w - 1)} | {(2, c) for c in range(w)})
    handle = frozenset(((2, w), (3, w), (3, w + 1)))
    return rectangle - slot, slot, handle


def _sample_shape(rng, forbidden, capacity, fallback_index=None):
    if fallback_index is not None:
        return _fallback_shape(fallback_index)
    for _ in range(180):
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        row = int(rng.integers(1, h - 1))
        slot = {(row, 0)}
        for c in range(1, w):
            new = int(np.clip(row + rng.integers(-1, 2), 1, h - 2))
            slot.update((r, c) for r in range(min(row, new), max(row, new) + 1))
            row = new
        # Optional wider cuts and small branches; the component check below
        # enforces a real separating crack rather than an arbitrary hole.
        for _ in range(int(rng.integers(3))):
            r, c = sorted(slot)[int(rng.integers(len(slot)))]
            dr, dc = DIRS[int(rng.integers(4))]
            if 0 < r + dr < h - 1 and 0 <= c + dc < w:
                slot.add((r + dr, c + dc))
        rectangle = _rect((0, h, 0, w))
        target = rectangle - slot
        parts = _components(target)
        if (len(parts) != 2 or min(map(len, parts)) < 3 or len(target) - 2 < capacity
                or _bbox(target) != (0, h, 0, w) or _shape_key(slot) in forbidden):
            continue
        starts = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in DIRS
                         if (r + dr, c + dc) not in rectangle})
        handle = {starts[int(rng.integers(len(starts)))]}
        relation = int(rng.integers(3))
        size = (int(rng.integers(2, max(3, len(slot)))) if relation == 0 else
                len(slot) if relation == 1 else len(slot) + int(rng.integers(1, 5)))
        size = min(size, 13)
        compact = bool(rng.integers(2))
        for _ in range(size - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in DIRS
                               if -3 <= r + dr < h + 3 and -3 <= c + dc < w + 3}
                              - rectangle - handle)
            if not frontier:
                break
            weights = np.array([sum((r + dr, c + dc) in handle for dr, dc in DIRS)
                                for r, c in frontier], dtype=float)
            weights = weights ** 2 if compact else 1 / weights ** 2
            handle.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
        regions = _transform((target, slot, handle), 0)
        box = _bbox(set().union(*regions))
        if len(handle) == size and max(box[1], box[3]) <= 10:
            return regions
    raise AssertionError("shape sampling bound exhausted")


def _case(rng, query, extras, kind):
    q, x = list(query), list(extras)
    if kind.startswith("neg"):
        missing = int(kind[-1])
        marks = [q[i] for i in range(3) if i != missing] + [x[0], x[1], x[0], x[1]]
        return marks, (x[0], q[missing])
    if kind == "exact":
        return q[1:], (q[0], x[0])
    if kind == "extra":
        return [q[0], q[1], x[0], x[1], q[0], x[0]], (q[2], x[1])
    if kind == "empty":
        return [], (x[0], x[1])
    positive = bool(rng.integers(2))
    if positive:
        missing = int(rng.integers(4))
        marks = [q[i] for i in range(3) if i != missing]
        plug = q[missing] if missing < 3 else int(rng.choice(x))
    else:
        take = int(rng.integers(3))
        marks = [int(c) for c in rng.permutation(q)[:take]]
        plug = int(rng.choice(x + marks))
    marks += [int(c) for c in rng.permutation(x)[:int(rng.integers(len(x) + 1))]]
    if marks:
        marks += [int(rng.choice(marks)) for _ in range(int(rng.integers(4)))]
    handle = int(rng.choice([c for c in q + x if c != plug]))
    return marks, (plug, handle)


def _pack(rng, sizes):
    area = sum((h + 1) * (w + 1) for h, w in sizes) + 25
    for attempt in range(18):
        if attempt < 12:
            h, w = int(rng.integers(18, 31)), int(rng.integers(18, 31))
            if h * w < area * 1.25:
                continue
        else:
            h = w = 30
        boxes, origins = [(0, 5, 0, 5)], [None] * len(sizes)
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1] + rng.random())
        for i in order:
            ph, pw = sizes[i]
            options = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            rng.shuffle(options)
            if rng.random() < .4:
                options.sort(key=lambda p: not (p[0] in (0, h - ph) or p[1] in (0, w - pw)))
            for r, c in options:
                box = r, r + ph, c, c + pw
                if all(box[1] < b[0] or b[1] < box[0] or box[3] < b[2] or b[3] < box[2]
                       for b in boxes):
                    boxes.append(box)
                    origins[i] = (r, c)
                    break
            else:
                break
        else:
            return h, w, origins
    raise AssertionError("packing bound exhausted")


def _make_pair(rng, palette, package, count, corner, force_fallback=False):
    bg, material = palette[:2]
    query = palette[2:5]
    extras = palette[5:]
    kinds = (["exact", "neg1", "neg2"] if package == 0 else
             ["neg0", "extra", "empty"] if package == 1 else ["free"] * count)
    parities = ([0, 1, 0] if package == 0 else [1, 1, 0] if package == 1 else
                [int(rng.integers(2)) for _ in kinds])
    for attempt in range(45):
        try:
            systems, sizes, forbidden = [], [], set()
            for i, (kind, parity) in enumerate(zip(kinds, parities)):
                marks, colors = _case(rng, query, extras, kind)
                shape = _sample_shape(rng, forbidden, len(marks), i if attempt >= 35 or force_fallback else None)
                forbidden.add(_shape_key(shape[1]))
                dest_op = int(rng.integers(8))
                source_op = int(rng.integers(4)) + 4 * ((dest_op >= 4) ^ bool(parity))
                target, slot, handle = _transform(shape, dest_op)
                key_regions = _transform(shape[1:], source_op)
                incoming_colors = colors[::-1] if parity else colors
                final_cells = target | slot | handle
                key_cells = key_regions[0] | key_regions[1]
                tb, kb = _bbox(final_cells), _bbox(key_cells)
                sizes.extend(((tb[1], tb[3]), (kb[1], kb[3])))
                systems.append((target, slot, handle, key_regions, marks, colors, incoming_colors))
            h, w, origins = _pack(rng, sizes)
            inp = np.full((h, w), bg, dtype=np.int8)
            inp[3, :4] = material
            inp[:4, 3] = material
            for pos, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
                inp[int(pos) // 3, int(pos) % 3] = color
            out = inp.copy()
            for i, (target, slot, handle, key, marks, colors, incoming) in enumerate(systems):
                ar, ac = origins[2 * i]
                kr, kc = origins[2 * i + 1]
                for p in _shift(target, ar, ac):
                    inp[p] = out[p] = material
                # Retain material in both fragments, so their role is visible.
                reserved = {sorted(part)[int(rng.integers(len(part)))] for part in _components(target)}
                available = sorted(target - reserved)
                assert len(available) >= len(marks)
                for index, color in zip(rng.choice(len(available), len(marks), replace=False), marks):
                    r, c = available[int(index)]
                    inp[r + ar, c + ac] = out[r + ar, c + ac] = color
                for region, color in zip(key, incoming):
                    for p in _shift(region, kr, kc):
                        inp[p] = color
                for region, color in zip((slot, handle), colors):
                    for p in _shift(region, ar, ac):
                        out[p] = color
                # Truth is constructed from the actual interior identities,
                # independently of solve's search and inferred source palette.
                if set(query) <= set(marks) | {colors[0]}:
                    tr0, tr1, tc0, tc1 = _bbox(target | slot)
                    out[ar + tr0:ar + tr1, ac + tc0:ac + tc1] = bg
            inp, out = _corner(inp, corner).copy(), _corner(out, corner).copy()
            info = _assembly(inp)
            if package == 0:
                # Enforce the count-shortcut witness from returned cells,
                # including rare seeds with unusually long positive cracks.
                marked = lambda m: sum(n for c, n in m["counts"].items() if c != material)
                if max(marked(m) for m in info["moves"] if not m["positive"]) < min(
                        marked(m) for m in info["moves"] if m["positive"]):
                    continue
            if np.array_equal(_finish(info), out):
                return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise AssertionError("bounded scene construction exhausted")


def _signature(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _teaching(infos):
    records = [(info, m) for info in infos for m in info["moves"]]
    positives = [(i, m) for i, m in records if m["positive"]]
    negatives = [(i, m) for i, m in records if not m["positive"]]
    assert positives and negatives
    assert any(m["labels"] == i["query"] for i, m in positives)
    assert any(m["labels"] - i["query"] for i, m in positives)
    assert any(not m["body_labels"] for i, m in negatives)
    for qindex in range(3):
        assert any(len(i["query"] - m["labels"]) == 1
                   and sorted(i["query"])[qindex] not in m["labels"] for i, m in negatives)
    assert any(i["query"] <= m["labels"] | set(m["colors"])
               for i, m in negatives), "external handles must not count"
    assert max(len(m["labels"]) for i, m in negatives) > min(len(m["labels"]) for i, m in positives)
    def marks(i, m):
        return sum(n for c, n in m["counts"].items() if c != i["material"])
    assert max(marks(i, m) for i, m in negatives) >= min(marks(i, m) for i, m in positives)
    assert {int(m["op"] >= 4) for i, m in records} == {0, 1}
    assert any(m["positive"] and not i["query"] <= m["body_labels"] for i, m in records)
    for policy in ((0, 0), (1, 1), (1, 0)):
        assert any(not np.array_equal(_finish(i, swap_policy=policy), _finish(i)) for i in infos)
        assert any(any(m["positive"] and any(wrong[p] != i["bg"] for p in m["rectangle"])
                       for m in i["moves"])
                   for i in infos
                   for wrong in [_corner(_finish(i, swap_policy=policy), i["corner"])]), (
                       "wrong key parity must change an actual dissolution decision")
    # Both simplifications produce different real training answers.
    assert any(not np.array_equal(_finish(i, erase=False), _finish(i)) for i in infos)
    assert any(not np.array_equal(_finish(i, count_outside=True), _finish(i)) for i in infos)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    stable = bool(rng.integers(2))
    active = int(rng.integers(7, 11))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            package = index if split == "train" and index < 2 else 2
            n = 3 if package < 2 else (4 if split == "test" and index == 0 else int(rng.integers(2, 4)))
            if stable:
                colors = palette[:active]
            elif package < 2:
                # The two teaching examples ask the same literal query, so
                # their three omission witnesses remain directly comparable.
                query = palette[2:5]
                remaining = [int(c) for c in rng.permutation(10) if c not in query]
                colors = remaining[:2] + query + remaining[2:int(rng.integers(4, 8))]
            else:
                colors = [int(c) for c in rng.permutation(10)[:int(rng.integers(7, 11))]]
            for _ in range(8):
                pair = _make_pair(rng, colors, package, n, int(rng.integers(4)))
                sig = _signature(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct-input bound exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            sig = _signature(inp)
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            info = _assembly(inp)
            assert np.array_equal(_finish(info), out), "incorrect assembly/dissolution"
            canonical_out = _corner(out, info["corner"])
            assert np.array_equal(canonical_out[:4, :4], info["view"][:4, :4])
            for key in info["keys"]:
                assert all(canonical_out[p] == info["bg"] for p in key["cells"])
            for m in info["moves"]:
                assert len(_components(m["body"])) == 2 and _bbox(m["body"]) == m["box"]
                assert all(info["view"][p] == info["bg"] for p in m["slot"])
                assert all(canonical_out[p] == info["assembled"][p] for p in m["outside"])
                if m["positive"]:
                    assert all(canonical_out[p] == info["bg"] for p in m["rectangle"])
                else:
                    assert all(canonical_out[p] == info["assembled"][p] for p in m["rectangle"])
            if split == "train":
                training.append(info)
    _teaching(training)


def self_test():
    """Forty-two seeds streamed in memory; no sample/trace files are written."""
    values = {k: set() for k in ("shapes", "slots", "handles", "operations", "counts", "corners",
                                 "materials", "backgrounds", "relationships", "structures", "puzzles")}
    stats = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                info = _assembly(g)
                values["shapes"].add(g.shape)
                values["corners"].add(info["corner"])
                values["materials"].add(info["material"])
                values["backgrounds"].add(info["bg"])
                values["counts"].add(len(info["moves"]))
                digest.update(bytes(g.shape) + g.tobytes() + pair["output"].tobytes())
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in g.flat)
                values["structures"].add(hashlib.sha256(bytes(g.shape) + normalized).hexdigest())
                for m in info["moves"]:
                    key = info["keys"][m["key"]]
                    plug, handle = key["regions"][m["plug_index"]], key["regions"][1 - m["plug_index"]]
                    values["slots"].add(_shape_key(plug))
                    values["handles"].add(_shape_key(handle))
                    values["operations"].add(m["op"])
                    values["relationships"].add(int(np.sign(len(plug) - len(handle))))
                    stats["deleted" if m["positive"] else "retained"] += 1
                    stats["extra_positive"] += bool(m["positive"] and m["labels"] - info["query"])
                    stats["handle_trap"] += bool(not m["positive"] and info["query"] <= m["labels"] | set(m["colors"]))
                    stats["symmetric_slot"] += len({tuple(sorted(_transform((plug,), op)[0])) for op in range(8)}) < 8
                    r0, r1, c0, c1 = m["box"]
                    stats["edge"] += r0 == 0 or c0 == 0 or r1 == info["view"].shape[0] or c1 == info["view"].shape[1]
                stats["pairs"] += 1
        values["puzzles"].add(digest.hexdigest())
        # Counterfactual query: with the keys and records unchanged, ask for
        # three identities of a hard negative. It must now dissolve.
        for pair in puzzle["train"]:
            info = _assembly(pair["input"])
            hard = next((m for m in info["moves"] if not m["positive"] and len(m["labels"]) >= 3), None)
            if hard is not None:
                changed = pair["input"].copy()
                v = _corner(changed, info["corner"])
                positions = np.argwhere(v[:3, :3] != info["bg"])
                for p, color in zip(positions, sorted(hard["labels"])[:3]):
                    v[tuple(p)] = color
                predicted = _corner(solve(changed), info["corner"])
                assert all(predicted[p] == info["bg"] for p in hard["rectangle"])
                break
        else:
            raise AssertionError("missing counterfactual query witness")
        if seed < 4:
            pair = puzzle["train"][0]
            assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
            permutation = np.random.default_rng(100 + seed).permutation(10)
            assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
            info = _assembly(pair["input"])
            moved, expected = pair["input"].copy(), pair["output"].copy()
            for g in (moved, expected):
                v = _corner(g, info["corner"])
                v[:3, :3] = np.roll(v[:3, :3].copy(), 1, axis=0)
            assert np.array_equal(solve(moved), expected)
    assert len(values["puzzles"]) == 42 and len(values["structures"]) >= 180
    assert len(values["shapes"]) >= 35 and len(values["slots"]) >= 45 and len(values["handles"]) >= 45
    assert values["operations"] == set(range(8)) and values["counts"] == {2, 3, 4}
    assert values["corners"] == set(range(4)) and values["relationships"] == {-1, 0, 1}
    assert len(values["materials"]) == 10 and len(values["backgrounds"]) == 10
    assert all(stats[k] > 0 for k in ("deleted", "retained", "extra_positive", "handle_trap", "symmetric_slot", "edge"))
    for package, n in ((0, 3), (1, 3), (2, 4)):
        pair = _make_pair(np.random.default_rng(600 + package), list(range(10)), package, n,
                          package, force_fallback=True)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(42, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p, q = generate_puzzle(19, 2, 1), generate_puzzle(19, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # Output corruption must not pass the family validator.
    broken = {s: [{k: g.copy() for k, g in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "measurements": {k: int(v) for k, v in stats.items()},
            "distinct": {k: len(v) for k, v in values.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
