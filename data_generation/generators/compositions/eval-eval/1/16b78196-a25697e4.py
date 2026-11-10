"""Shared-apron locks: rigid two-color keys jointly complete a band rectangle.

Only NumPy and the standard library are used. Generation paints constructed
answers; the input-only enumerator separately recovers and checks them. No
parent module, saved examples, seed, or generation metadata enters solve().
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(cells):
    remaining, parts = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(frozenset(part))
    return parts


def _cells(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _transform(regions, op):
    """0..3 rotations, 4..7 reflection followed by a rotation; normalize jointly."""
    transformed = []
    for region in regions:
        result = set()
        for r, c in region:
            if op >= 4:
                c = -c
            for _ in range(op % 4):
                r, c = c, -r
            result.add((r, c))
        transformed.append(result)
    all_cells = set().union(*transformed)
    r0, _, c0, _ = _bbox(all_cells)
    return tuple(_shift(region, -r0, -c0) for region in transformed)


def _bits(cells, width):
    result = 0
    for r, c in cells:
        result |= 1 << (r * width + c)
    return result


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _inspect(input_grid):
    grid = _grid(input_grid)
    hist = np.bincount(grid.ravel(), minlength=10)
    bg = int(hist.argmax())
    if np.count_nonzero(hist == hist[bg]) != 1:
        raise ValueError("background must be the unique most frequent color")
    objects = _components(_cells(grid != bg))
    bands = []
    for cells in objects:
        if len({int(grid[p]) for p in cells}) != 1:
            continue
        for axis in (0, 1):
            if len({p[1 - axis] for p in cells}) == grid.shape[1 - axis]:
                bands.append((cells, axis, int(grid[min(cells)])))
    if len(bands) != 1:
        raise ValueError("expected one monochrome spanning band")
    _, axis, band = bands[0]
    g = grid if axis == 0 else grid.T
    r0, r1, _, _ = _bbox(_cells(g == band))
    if np.any((g[r0:r1] != band) & (g[r0:r1] != bg)):
        raise ValueError("keys must start outside the band envelope")
    raw_slots = _components(_cells(g[r0:r1] == bg))
    if len(raw_slots) < 2:
        raise ValueError("at least two notches must share the apron")
    sides = set()
    for slot in raw_slots:
        touches = {s for s, row in ((0, 0), (1, r1 - r0 - 1))
                   if any(r == row for r, c in slot)}
        if len(touches) != 1 or any(c in (0, g.shape[1] - 1) for r, c in slot):
            raise ValueError("each notch opens only on one long band edge")
        sides.update(touches)
    if len(sides) != 1:
        raise ValueError("all notches must open on the same long edge")
    flipped = sides == {1}
    if flipped:
        g = g[::-1]
    fixed = _cells(g == band)
    top, bottom, _, _ = _bbox(fixed)
    slots = [_shift(s, top, 0) for s in _components(_cells(g[top:bottom] == bg))]
    keys = []
    for obj in _components(_cells(g != bg)):
        if obj == fixed:
            continue
        colors = tuple(sorted({int(g[p]) for p in obj}))
        if len(colors) != 2 or band in colors:
            raise ValueError("each detached key needs two colors distinct from the band")
        regions = tuple(frozenset(p for p in obj if int(g[p]) == c) for c in colors)
        if any(len(_components(region)) != 1 for region in regions):
            raise ValueError("both key regions must be connected")
        if any(top <= r < bottom for r, c in obj):
            raise ValueError("key intersects the band envelope")
        keys.append({"cells": obj, "colors": colors, "regions": regions})
    if len(keys) != len(slots):
        raise ValueError("one whole key is required per notch")
    base = np.full_like(g, bg)
    for p in fixed:
        base[p] = band
    return {"grid": g, "background": bg, "band": band, "fixed": fixed,
            "top": top, "bottom": bottom, "slots": slots, "keys": keys,
            "base": base, "axis": axis, "flipped": flipped}


def _placements(info):
    """Enumerate BOTH regions and ALL symmetries, before rectangular completion."""
    h, w = info["grid"].shape
    options = [[] for _ in info["slots"]]
    for ki, key in enumerate(info["keys"]):
        for si, slot in enumerate(info["slots"]):
            sr, _, sc, _ = _bbox(slot)
            seen = set()
            for op in range(8):
                regions = _transform(key["regions"], op)
                for plug in (0, 1):
                    if len(regions[plug]) != len(slot):
                        continue
                    pr, _, pc, _ = _bbox(regions[plug])
                    moved = tuple(_shift(region, sr - pr, sc - pc) for region in regions)
                    if moved[plug] != slot:
                        continue
                    handle = moved[1 - plug]
                    if any(not (0 <= r < info["top"] and 0 <= c < w) for r, c in handle):
                        continue
                    colors = key["colors"] if op < 4 else key["colors"][::-1]
                    painted = tuple(sorted((r, c, colors[j]) for j in (0, 1) for r, c in moved[j]))
                    if painted in seen:
                        continue
                    seen.add(painted)
                    options[si].append({"key": ki, "slot": si, "op": op, "plug": plug,
                                        "regions": moved, "handle": handle,
                                        "bits": _bits(handle, w), "bbox": _bbox(handle),
                                        "painted": painted})
    return options


def _render(info, moves, policy=(False, True)):
    out = info["base"].copy()
    for move in moves:
        key = info["keys"][move["key"]]
        colors = key["colors"][::-1] if policy[move["op"] >= 4] else key["colors"]
        for region, color in zip(move["regions"], colors):
            for p in region:
                if out[p] != info["background"]:
                    raise ValueError("overlapping keys")
                out[p] = color
    return out


def _restore(out, info):
    if info["flipped"]:
        out = out[::-1]
    if info["axis"]:
        out = out.T
    return out.copy()


def _search(input_grid):
    info = _inspect(input_grid)
    options = _placements(info)
    if any(not row for row in options):
        raise ValueError("a notch has no locally fitting whole key")
    area = sum(len(k["cells"]) for k in info["keys"]) - sum(map(len, info["slots"]))
    if area <= 0:
        raise ValueError("empty apron")
    left = min(c for slot in info["slots"] for r, c in slot)
    right = max(c for slot in info["slots"] for r, c in slot) + 1
    width = info["grid"].shape[1]
    solutions = {}
    for apron_width in range(right - left, width + 1):
        if area % apron_width:
            continue
        apron_height = area // apron_width
        if not 1 <= apron_height <= info["top"]:
            continue
        for c0 in range(max(0, right - apron_width), min(left, width - apron_width) + 1):
            rect = (info["top"] - apron_height, info["top"], c0, c0 + apron_width)
            eligible = [[m for m in row if m["bbox"][0] >= rect[0]
                         and m["bbox"][1] <= rect[1] and m["bbox"][2] >= rect[2]
                         and m["bbox"][3] <= rect[3]] for row in options]

            def visit(remaining, used_keys, occupied, moves):
                if len(solutions) > 1:
                    return
                if not remaining:
                    # Equal area, containment, and disjointness prove exact cover.
                    answer = _render(info, moves)
                    solutions.setdefault(answer.tobytes(), (answer, tuple(moves), rect))
                    return
                si, candidates = min(((s, [m for m in eligible[s]
                                          if not used_keys & (1 << m["key"])
                                          and not occupied & m["bits"]]) for s in remaining),
                                     key=lambda item: len(item[1]))
                for move in candidates:
                    visit(remaining - {si}, used_keys | (1 << move["key"]),
                          occupied | move["bits"], moves + [move])

            visit(set(range(len(options))), 0, 0, [])
            if len(solutions) > 1:
                raise ValueError("multiple distinct colored rectangular assemblies")
    if len(solutions) != 1:
        raise ValueError("no rectangular assembly")
    answer, moves, rect = next(iter(solutions.values()))
    info.update({"moves": moves, "rectangle": rect, "options": options})
    return _restore(answer, info), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique assembly using only the supplied input."""
    return _search(input_grid)[0]


def _ablation_witness(info):
    """A complete, collision-free notch fitting that fails rectangular closure."""
    chosen = list(info["moves"])
    expected = _render(info, chosen)
    for i, old in enumerate(chosen):
        other_bits = 0
        for j, m in enumerate(chosen):
            if i != j:
                other_bits |= m["bits"]
        for alt in info["options"][old["slot"]]:
            if alt["key"] != old["key"] or alt["bits"] & other_bits:
                continue
            candidate = chosen.copy()
            candidate[i] = alt
            out = _render(info, candidate)
            if not np.array_equal(out != info["background"], expected != info["background"]):
                return candidate
    return None


def _palette(rng):
    values = tuple(map(int, rng.permutation(10)))
    return values[0], values[1], values[2:]


def _kit(rng, count, safe=False):
    """Connected interlocking strips, each with its own attached notch region."""
    deep_socket = not safe and rng.random() < .3
    height = 3 if safe or deep_socket else int(rng.integers(3, 7))
    widths = [4] * count if safe else list(map(int, rng.integers(3, 6, size=count)))
    widths[0] = 4 if safe else int(rng.integers(4, 6))
    total = sum(widths)
    borders = np.zeros((height, count + 1), dtype=int)
    borders[:, -1] = total
    borders[-1] = np.cumsum([0] + widths)
    for row in range(height - 2, -1, -1):
        for _ in range(100):
            current = borders[row + 1].copy()
            current[1:-1] += rng.integers(-1, 2, size=count - 1)
            current[1] = int(np.clip(current[1], 3, widths[0]))
            if np.all(np.diff(current) >= 2):
                borders[row] = current
                break
        else:
            borders[row] = borders[row + 1]
    if np.all(borders[:, 1] == widths[0]):
        borders[0, 1] -= 1
    if safe:
        for i in range(1, count):
            borders[0, i] = 4 * i - 1
        borders[1] = borders[-1]
    core = 2 if safe else int(rng.integers(2, 4))
    max_depth = (30 - 2 * height - 2 - core) // 3
    depth = min(5, max_depth) if safe or deep_socket else int(rng.integers(1, min(5, max_depth) + 1))
    regions = []
    previous_end = -1
    for i, size in enumerate(widths):
        start = int(borders[-1, i])
        end = int(borders[-1, i + 1])
        if i == 0:
            slot_width, opening = 2, 0
            profile = np.full(2, 2 if safe else depth, dtype=int)
        else:
            slot_width = 2 if safe else size - 1 if deep_socket and i == count - 1 else int(rng.integers(2, size))
            low = max(start, previous_end + 1)
            high = end - slot_width
            opening = high if safe and i == count - 1 else int(rng.integers(low, high + 1))
            if safe:
                profile = np.full(slot_width, i + 2, dtype=int)
            elif (deep_socket and i == count - 1) or rng.random() < .45:
                profile = np.full(slot_width, depth, dtype=int)
            else:
                profile = rng.integers(1, depth + 1, size=slot_width)
        previous_end = opening + slot_width
        plug = frozenset((height + r, opening + c)
                         for c in range(slot_width) for r in range(int(profile[c])))
        handle = frozenset((r, c) for r in range(height)
                           for c in range(int(borders[r, i]), int(borders[r, i + 1])))
        regions.append((plug, handle))
    depth = max(r - height + 1 for plug, handle in regions for r, c in plug)
    return {"regions": regions, "height": height, "width": total, "depth": depth, "core": core}


def _dilate(mask):
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dc), min(w, w - dc)
            out[r0 + dr:r1 + dr, c0 + dc:c1 + dc] |= mask[r0:r1, c0:c1]
    return out


def _scatter(rng, base, keys, colors, top, bottom, bg):
    h, w = base.shape
    for _ in range(32):
        inp = base.copy()
        forbidden = np.zeros_like(inp, dtype=bool)
        forbidden[max(0, top - 1):min(h, bottom + 1)] = True
        order = sorted(range(len(keys)), key=lambda i: -len(keys[i][0] | keys[i][1]) + rng.random())
        for i in order:
            regions = keys[i]
            shape = regions[0] | regions[1]
            _, kh, _, kw = _bbox(shape)
            if kh > h or kw > w:
                break
            valid = np.ones((h - kh + 1, w - kw + 1), dtype=bool)
            for r, c in shape:
                valid &= ~forbidden[r:r + valid.shape[0], c:c + valid.shape[1]]
            choices = np.argwhere(valid)
            if not len(choices):
                break
            dr, dc = map(int, choices[int(rng.integers(len(choices)))])
            mask = np.zeros_like(forbidden)
            for region, color in zip(regions, colors[i]):
                for p in _shift(region, dr, dc):
                    inp[p] = color
                    mask[p] = True
            forbidden |= _dilate(mask)
        else:
            return inp
    raise ValueError("bounded scatter exhausted")


def _scene(rng, kit, source_ops, palette, key_colors, turn, mirror=False, layout=None):
    bg, band, payload = palette
    key_shapes = [_transform(regions, op) for regions, op in zip(kit["regions"], source_ops)]
    max_source_h = max(_bbox(a | b)[1] for a, b in key_shapes)
    clearance = max(kit["height"] + 1, max_source_h + 1)
    band_depth = kit["depth"] + kit["core"]
    if layout is None:
        minimum_h = max(17, 2 * clearance + band_depth)
        minimum_w = kit["width"] + 8
        if minimum_h > 30 or minimum_w > 30:
            raise ValueError("kit too large")
        h = int(rng.integers(minimum_h, 31))
        w = int(rng.integers(max(18, minimum_w), 31))
        top = int(rng.integers(clearance, h - band_depth - clearance + 1))
        c0 = int(rng.integers(4, w - kit["width"] - 3))
        layout = (h, w, top, c0)
    h, w, top, c0 = layout
    bottom = top + band_depth
    base = np.full((h, w), bg, dtype=np.int8)
    base[top:bottom] = band
    destination = []
    for plug, handle in kit["regions"]:
        regions = tuple(_shift(region, top - kit["height"], c0) for region in (plug, handle))
        destination.append(regions)
        for p in regions[0]:
            base[p] = bg
    if mirror:
        base = base[:, ::-1].copy()
        destination = [tuple(frozenset((r, w - 1 - c) for r, c in region) for region in pair)
                       for pair in destination]
    out = base.copy()
    for regions, op, colors in zip(destination, source_ops, key_colors):
        colors = colors[::-1] if bool(op >= 4) != mirror else colors
        for region, color in zip(regions, colors):
            for p in region:
                if out[p] != bg:
                    raise ValueError("construction overlap")
                out[p] = color
    inp = _scatter(rng, base, key_shapes, key_colors, top, bottom, bg)
    if np.count_nonzero(inp == bg) <= max(np.count_nonzero(inp == c) for c in range(10) if c != bg):
        raise ValueError("ambiguous background")
    inp, out = np.rot90(inp, turn).copy(), np.rot90(out, turn).copy()
    predicted, info = _search(inp)
    if not np.array_equal(predicted, out) or _ablation_witness(info) is None:
        raise ValueError("construction needs unique global closure and a nonrectangular alternative")
    return {"input": inp, "output": out}, layout


def _key_colors(rng, palette, count):
    payload = palette[2]
    if rng.random() < .5:
        pair = tuple(map(int, rng.choice(payload, 2, replace=False)))
        return [pair] * count
    return [tuple(map(int, rng.choice(payload, 2, replace=False))) for _ in range(count)]


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = _palette(rng)
    stable = bool(rng.random() < .3)
    turn = int(rng.integers(4))
    # The first two demonstrations reuse the actual kit and input-key geometry:
    # mirroring the dock forces a color exchange, without an invisible role map.
    for attempt in range(80):
        try:
            kit = _kit(rng, 2, safe=attempt >= 64)
            ops = list(map(int, rng.integers(4, size=2)))
            colors = _key_colors(rng, palette, 2)
            first, layout = _scene(rng, kit, ops, palette, colors, turn)
            second, _ = _scene(rng, kit, ops, palette, colors, turn, mirror=True, layout=layout)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("bounded witness construction exhausted")
    puzzle = {"train": [first, second], "test": []}
    seen = {_signature(first["input"]), _signature(second["input"])}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for index in range(count):
            n = (3 if index == 0 else int(rng.integers(2, 4))) if split == "train" else (3 if num_train == 2 else 4)
            current_palette = palette if stable else _palette(rng)
            scene_turn = turn if split == "train" else (turn + 1) % 4
            if split == "train" and index > 0 and rng.random() < .4:
                scene_turn = int(rng.integers(4))
            for attempt in range(80):
                try:
                    kit = _kit(rng, n, safe=attempt >= 64)
                    # Fresh scenes include both parities, decoupled from key position.
                    parity = [0, 1] + list(map(int, rng.integers(2, size=n - 2)))
                    rng.shuffle(parity)
                    ops = [int(rng.integers(4)) + 4 * p for p in parity]
                    colors = _key_colors(rng, current_palette, n)
                    pair, _ = _scene(rng, kit, ops, current_palette, colors, scene_turn)
                    signature = _signature(pair["input"])
                    if signature in seen:
                        continue
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded scene construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, training_parities = set(), set()
    witnesses = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            expected, info = _search(inp)
            assert np.array_equal(expected, out)
            canonical = out if info["axis"] == 0 else out.T
            if info["flipped"]:
                canonical = canonical[::-1]
            assert np.array_equal(info["grid"] == info["band"], canonical == info["band"])
            assert np.count_nonzero(inp != info["background"]) == np.count_nonzero(out != info["background"])
            used, handle_union = set(), set()
            for move in info["moves"]:
                key = info["keys"][move["key"]]
                assert move["key"] not in used
                used.add(move["key"])
                plug = move["plug"]
                assert move["regions"][plug] == info["slots"][move["slot"]]
                assert not handle_union & move["handle"]
                handle_union.update(move["handle"])
                transformed = _transform(key["regions"], move["op"])
                ar, _, ac, _ = _bbox(set().union(*move["regions"]))
                assert tuple(_shift(r, ar, ac) for r in transformed) == move["regions"]
                if split == "train":
                    training_parities.add(int(move["op"] >= 4))
            r0, r1, c0, c1 = info["rectangle"]
            assert handle_union == {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
            assert used == set(range(len(info["keys"])))
            # A is necessary even after all B-like notch fits and non-overlap.
            assert _ablation_witness(info) is not None
            witnesses[split] += 1
    assert training_parities == {0, 1}
    # Check all four parity/color policies against the same uniquely fitted moves.
    compatible = []
    for policy in ((False, False), (False, True), (True, False), (True, True)):
        if all(np.array_equal(_restore(_render(info, info["moves"], policy), info), pair["output"])
               for pair in puzzle["train"] for _, info in [_search(pair["input"])]):
            compatible.append(policy)
    assert compatible == [(False, True)], "demonstrations must distinguish reflection-local exchange"
    return None


def _features(pair):
    _, info = _search(pair["input"])
    relations, slot_shapes, handle_shapes = [], [], []
    for move in info["moves"]:
        key = info["keys"][move["key"]]
        plug = key["regions"][move["plug"]]
        handle = key["regions"][1 - move["plug"]]
        relations.append(int(np.sign(len(plug) - len(handle))))
        slot_shapes.append(min(tuple(sorted(_transform((plug,), op)[0])) for op in range(8)))
        handle_shapes.append(min(tuple(sorted(_transform((handle,), op)[0])) for op in range(8)))
    r0, r1, c0, c1 = info["rectangle"]
    return {"shape": tuple(pair["input"].shape), "keys": len(info["keys"]),
            "orientation": info["axis"], "side": int(info["flipped"]),
            "roles": (info["background"], info["band"]),
            "operations": tuple(m["op"] for m in info["moves"]),
            "relations": tuple(relations), "slots": tuple(slot_shapes),
            "handles": tuple(handle_shapes), "apron": (r1 - r0, c1 - c0),
            "shared_palette": len({k["colors"] for k in info["keys"]}) < len(info["keys"]),
            "local_options": tuple(len(row) for row in info["options"])}


def self_test():
    """Bounded, no files: 42 seeds, semantic ablations, mutations, determinism."""
    shapes, slots, handles, roles, operations, counts, relations = (set() for _ in range(7))
    structures, puzzle_hashes, orientations, sides, aprons = set(), set(), set(), set(), set()
    reuse = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for pair in puzzle["train"] + puzzle["test"]:
            feature = _features(pair)
            shapes.add(feature["shape"])
            slots.update(feature["slots"])
            handles.update(feature["handles"])
            roles.add(feature["roles"])
            operations.update(feature["operations"])
            counts.add(feature["keys"])
            relations.update(feature["relations"])
            orientations.add(feature["orientation"])
            sides.add(feature["side"])
            aprons.add(feature["apron"])
            reuse += feature["shared_palette"]
            mapping = {}
            normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
            structures.add((pair["input"].shape, normalized))
            digest.update(pair["input"].tobytes())
            # solve must accept lists and never mutate the caller's array.
            before = pair["input"].copy()
            assert np.array_equal(solve(before.tolist()), pair["output"])
            assert np.array_equal(before, pair["input"])
        puzzle_hashes.add(digest.digest())
    assert len(puzzle_hashes) == 42 and len(structures) >= 200
    assert len(shapes) >= 35 and len(slots) >= 20 and len(handles) >= 80
    assert len(roles) >= 20 and len(aprons) >= 25
    assert operations == set(range(8)) and counts == {2, 3, 4}
    assert relations == {-1, 0, 1} and orientations == {0, 1} and sides == {0, 1}
    assert reuse >= 40
    original = generate_puzzle(0, 2, 1)
    duplicate = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in original
               for a, b in zip(original[s], duplicate[s]) for k in ("input", "output"))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for policy in ((False, False), (True, False), (True, True)):
        assert any(not np.array_equal(_restore(_render(info, info["moves"], policy), info), pair["output"])
                   for pair in original["train"] for _, info in [_search(pair["input"])])
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong answer accepted")
    # One key has reflection-symmetric region masks, so geometry permits two
    # different colored answers even after perfect rectangular completion.
    ambiguous = np.zeros((20, 20), dtype=np.int8)
    ambiguous[10:14] = 8
    ambiguous[10:12, 5:8] = 0
    ambiguous[10:12, 9:12] = 0
    ambiguous[0:2, 0:4] = 2
    ambiguous[2:4, 0:3] = 3
    ambiguous[1:3, 14:17] = 4
    ambiguous[3:5, 14:17] = 5
    try:
        solve(ambiguous)
    except ValueError as error:
        assert "multiple distinct colored" in str(error)
    else:
        raise AssertionError("ambiguous colored assembly accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(puzzle_hashes),
            "input_structures": len(structures), "canvas_shapes": len(shapes),
            "slot_shapes_up_to_D4": len(slots), "handle_shapes_up_to_D4": len(handles),
            "apron_dimensions": len(aprons), "palette_role_pairs": len(roles),
            "operations": sorted(operations), "plug_minus_handle_signs": sorted(relations)}
