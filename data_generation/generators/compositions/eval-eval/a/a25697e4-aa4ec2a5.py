"""Keyed chambers: rigid seals route colors through a topological change.

Only NumPy and the Python standard library are used. No parent code is imported.
The fixed marker/outline color is 8; all other roles are visible in each input.
"""

from itertools import product

import numpy as np


MARKER = 8
D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CHECK_SEEDS = (
    0, 628852782, 544943959, 139110122, 836817508, 768349574,
    232662781, 439838298, 525662291, 486187196, 21892444, 488720894,
    747639130, 410471238, 873543551, 450048580, 60364863, 676073225,
    166851, 26726616, 32613448, 518576115, 321288319, 73435684,
    224746381, 776395989, 203979369, 154455118, 730685781, 551826666,
    311008315, 399741899, 823982270, 388573172, 236125892, 481946732,
    207381820, 881470179, 963755184, 308691479, 434202275, 821534053,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(seed, nt, ne):
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, low, high in ((nt, 2, 10), (ne, 1, 4)):
        if not _integer(value) or not low <= value <= high:
            raise ValueError("illegal example count")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return int(nt), int(ne)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(((0 <= a) & (a <= 9)).all()), "invalid colors")
    return a


def _components(cells, neighbors=D4):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    comp.add(p)
                    stack.append(p)
        result.append(frozenset(comp))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _box_cells(cells):
    r0, r1, c0, c1 = _bbox(cells)
    return frozenset(product(range(r0, r1 + 1), range(c0, c1 + 1)))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _transform(regions, operation):
    """Mirror left/right first, then rotate; 0..3 preserve orientation."""
    transformed = []
    for region in regions:
        dest = []
        for r, c in region:
            if operation >= 4:
                c = -c
            for _ in range(operation % 4):
                r, c = c, -r
            dest.append((r, c))
        transformed.append(dest)
    points = [p for region in transformed for p in region]
    r0, _, c0, _ = _bbox(points)
    return tuple(_shift(region, -r0, -c0) for region in transformed)


def _canonical(cells):
    return min(tuple(sorted(_transform((cells,), op)[0])) for op in range(8))


def _holes(wall, neighbors=D4):
    """Enclosures of one barrier, with explicit exterior padding at all edges."""
    r0, r1, c0, c1 = _bbox(wall)
    domain = set(product(range(r0 - 1, r1 + 2), range(c0 - 1, c1 + 2)))
    unseen = domain - set(wall)
    start = r0 - 1, c0 - 1
    unseen.remove(start)
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            p = r + dr, c + dc
            if p in unseen:
                unseen.remove(p)
                stack.append(p)
    return frozenset(unseen)


def _signature(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.uint8).tobytes()


def _inspect(inp):
    values, counts = np.unique(inp, return_counts=True)
    _require(len(values) >= 4, "scene needs walls, guides and keys")
    maximum = int(counts.max())
    _require(np.count_nonzero(counts == maximum) == 1, "background must be uniquely dominant")
    bg = int(values[np.argmax(counts)])
    _require(bg != MARKER, "marker cannot be background")
    comps = _components(map(tuple, np.argwhere(inp != bg).tolist()))
    _require(len(comps) == len(_components(map(tuple, np.argwhere(inp != bg).tolist()), D8)),
             "distinct objects must not touch diagonally")
    targets, keys = [], []
    for cells in comps:
        colors = sorted({int(inp[p]) for p in cells})
        _require(len(colors) == 2, "each object has exactly two colors")
        if MARKER in colors:
            slot = frozenset(p for p in cells if int(inp[p]) == MARKER)
            body = cells - slot
            _require(len(_components(slot)) == 1 and 1 <= len(_components(body)) <= 2,
                     "connected guide must link one or two wall fragments")
            targets.append({"body": body, "slot": slot, "cells": cells,
                            "color": next(c for c in colors if c != MARKER),
                            "box": _box_cells(cells)})
        else:
            regions = tuple(frozenset(p for p in cells if int(inp[p]) == color) for color in colors)
            _require(all(len(_components(region)) == 1 for region in regions),
                     "key regions must be connected")
            keys.append({"regions": regions, "colors": tuple(colors), "cells": cells})
    _require(1 <= len(keys) == len(targets) <= 3, "one key per chamber, up to three")
    boxes = [t["box"] for t in targets]
    _require(all(not a & b for i, a in enumerate(boxes) for b in boxes[i + 1:]),
             "chamber boxes must not interlock")
    boxes += [_box_cells(k["cells"]) for k in keys]
    for i, box in enumerate(boxes):
        moat = {(r + dr, c + dc) for r, c in box for dr, dc in ((0, 0),) + D8}
        _require(all(not moat & other for other in boxes[i + 1:]),
                 "object boxes require a one-cell background moat")
    return bg, targets, keys


def _match(inp):
    """Global, input-only fit search; deduplicate complete colored placements."""
    bg, targets, keys = _inspect(inp)
    possibilities = []
    for ki, key in enumerate(keys):
        options = {}
        for ti, target in enumerate(targets):
            sr0, _, sc0, _ = _bbox(target["slot"])
            for op in range(8):
                regions = _transform(key["regions"], op)
                for pi in (0, 1):
                    pr0, _, pc0, _ = _bbox(regions[pi])
                    placed = tuple(_shift(region, sr0 - pr0, sc0 - pc0) for region in regions)
                    if placed[pi] != target["slot"]:
                        continue
                    probe = placed[1 - pi]
                    if not probe <= target["box"]:
                        continue
                    if any(int(inp[p]) != bg for p in probe):
                        continue
                    colors = key["colors"] if op < 4 else key["colors"][::-1]
                    signature = (ti, tuple(sorted((p, color) for reg, color in zip(placed, colors) for p in reg)))
                    options[signature] = {"target": ti, "key": ki, "op": op,
                                          "plug": placed[pi], "probe": probe,
                                          "plug_color": colors[pi], "probe_color": colors[1 - pi],
                                          "original_plug_color": key["colors"][pi],
                                          "original_probe_color": key["colors"][1 - pi]}
        _require(bool(options), "key has no legal rigid placement")
        possibilities.append(list(options.values()))
    answers = {}

    def search(index, used, occupied, moves):
        if len(answers) > 1:
            return
        if index == len(keys):
            signature = tuple(sorted((m["target"], p, color) for m in moves
                                     for reg, color in ((m["plug"], m["plug_color"]),
                                                        (m["probe"], m["probe_color"])) for p in reg))
            answers.setdefault(signature, tuple(moves))
            return
        for move in possibilities[index]:
            cells = move["plug"] | move["probe"]
            if move["target"] not in used and not cells & occupied:
                search(index + 1, used | {move["target"]}, occupied | cells, moves + [move])

    search(0, set(), set(), [])
    _require(len(answers) == 1, "competing or absent complete colored fits")
    moves = next(iter(answers.values()))
    return bg, targets, keys, moves


def _paint(shape, bg, targets, moves, *, swap=True, topology="assembled", outline=True):
    """Paint known placements; optional policies are used only as counterfactuals."""
    out = np.full(shape, bg, dtype=np.int8)
    all_walls, all_holes, all_probes = set(), set(), set()
    inspections = []
    for m in moves:
        t = targets[m["target"]]
        wall = t["body"] | m["plug"]
        holes = _holes(t["body"] if topology == "before" else wall)
        _require(holes == _holes(t["body"] if topology == "before" else wall, D8),
                 "diagonal-only topology is excluded")
        pc = m["plug_color"] if swap else m["original_plug_color"]
        hc = m["probe_color"] if swap else m["original_probe_color"]
        hollow = bool(holes) if topology not in ("never", "always") else topology == "always"
        for p in t["body"]:
            out[p] = pc if hollow else t["color"]
        for p in m["plug"]:
            out[p] = pc
        if topology != "never":
            for p in holes:
                out[p] = hc
        for p in m["probe"]:
            out[p] = hc
        all_walls.update(wall)
        all_holes.update(holes)
        all_probes.update(m["probe"])
        inspections.append({"wall": wall, "holes": holes, "body": t["body"],
                            "slot": m["plug"], "probe": m["probe"],
                            "hollow": hollow, "pc": pc, "hc": hc,
                            "wall_color": t["color"], "move": m})
    halo = {(r + dr, c + dc) for r, c in all_walls for dr, dc in D8
            if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]}
    halo -= all_walls | all_holes | all_probes
    if outline:
        for p in halo:
            out[p] = MARKER
    return out, inspections, frozenset(halo)


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    bg, targets, _, moves = _match(inp)
    return _paint(inp.shape, bg, targets, moves)[0]


def _prototype(rng, pockets, vent, forbidden=(), small=False, fallback_index=None):
    """Carve pockets and a shaped gate into wall fragments, then grow a probe.

    A vent is a genuinely open orthogonal tunnel, never an inferred class flag.
    The sampling labels are discarded before reconstruction and validation.
    """
    if fallback_index is not None:
        pockets = min(pockets, 2)
    for attempt in range(180):
        fallback = fallback_index is not None
        if fallback:
            h, w, top, bottom = 7, 8, 3, 1
        else:
            h = int(rng.integers(6, 10 if not small else 9))
            w = int(rng.integers(7 if pockets <= 1 else 8, 12 if not small else 10))
            top = int(rng.integers(2, min(5, h - 2)))
            bottom = int(rng.integers(1, min(3, h - top - 1)))
        rectangle = set(product(range(h), range(w)))
        if pockets <= 1:
            intervals = [(1, w - 1)]
        elif pockets == 2:
            division = 5 if fallback else int(rng.integers(3, w - 2))
            intervals = [(1, division), (division + 1, w - 1)]
        else:
            # Three pockets require sufficient room and are an occasional extension.
            if w < 10:
                continue
            intervals = [(1, 3), (4, 6), (7, w - 1)]
        candidates = [i for i, (a, b) in enumerate(intervals) if b - a >= 2]
        if not candidates:
            continue
        main = 0 if fallback else int(rng.choice(candidates))
        air, main_air = set(), set()
        for i, (left, right) in enumerate(intervals):
            pocket_top = top if i == main else top + (1 if fallback else int(rng.integers(0, 2)))
            pocket_bottom = h - bottom
            if pocket_bottom <= pocket_top:
                pocket_top = top
            pocket = set(product(range(pocket_top, pocket_bottom), range(left, right)))
            # Off-center or stepped pockets without an ambiguous diagonal leak.
            if (not fallback and len(pocket) >= 6 and right - left >= 2
                    and pocket_bottom - pocket_top >= 2 and rng.random() < .65):
                corner = (pocket_bottom - 1, left if rng.random() < .5 else right - 1)
                pocket.remove(corner)
            air.update(pocket)
            if i == main:
                main_air = pocket
        left, right = intervals[main]
        if fallback:
            end = 3 + fallback_index
            slot = {(0, 2), (1, 2)} | {(2, c) for c in range(2, end + 1)}
        else:
            col = int(rng.integers(left, right))
            slot = {(0, col)}
            for row in range(1, top):
                new_col = int(np.clip(col + rng.integers(-1, 2), left, right - 1))
                slot.update((row, c) for c in range(min(col, new_col), max(col, new_col) + 1))
                col = new_col
            # Unequal branches vary gate area and D4 symmetry, beyond thin paths.
            row = int(rng.integers(top))
            col = int(rng.choice(sorted(c for r, c in slot if r == row)))
            reach = int(rng.integers(1, 4))
            direction = -1 if rng.random() < .5 else 1
            slot.update((row, c) for c in range(min(col, int(np.clip(col + direction * reach, left, right - 1))),
                                               max(col, int(np.clip(col + direction * reach, left, right - 1))) + 1))
        if _canonical(slot) in forbidden or len(slot) < 3:
            continue
        closed_wall = rectangle - air
        if vent:
            # An independent door from the main bay through the opposite wall.
            columns = sorted(c for r, c in main_air if r == h - bottom - 1)
            vent_col = columns[len(columns) // 2] if fallback else int(rng.choice(columns))
            closed_wall -= {(r, vent_col) for r in range(h - bottom, h)}
        body = closed_wall - slot
        if not body or not 1 <= len(_components(body)) <= 2:
            continue
        if not fallback:
            # Connected distant extensions teach whole-body classification.
            if rng.random() < .65:
                anchor_cols = sorted(c for r, c in body if r == h - 1)
                if anchor_cols:
                    c = int(rng.choice(anchor_cols))
                    length = int(rng.integers(1, 3 if small else 5))
                    tail = {(r, c) for r in range(h, h + length)}
                    if rng.random() < .65:
                        direction = -1 if rng.random() < .5 else 1
                        tail.update((h + length - 1, c + direction * k)
                                    for k in range(1, int(rng.integers(2, 4))))
                    body |= tail
            # Shave outer corners, retaining the intended topology and connectivity.
            for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
                if corner in body and rng.random() < .35:
                    candidate = body - {corner}
                    if (len(_components(candidate)) == len(_components(body))
                            and _holes(candidate | slot) == _holes(body | slot)):
                        body = candidate
        final_wall = body | slot
        holes = _holes(final_wall)
        if holes != _holes(final_wall, D8):
            continue
        expected_holes = max(1, pockets) - int(vent)
        if len(_components(holes)) != expected_holes:
            continue
        starts = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in D4} & main_air)
        if not starts:
            continue
        start = starts[-1] if fallback else starts[int(rng.integers(len(starts)))]
        probe = {start}
        if fallback:
            desired = min(2, len(main_air))
        else:
            relation = int(rng.integers(3))
            desired = (int(rng.integers(1, max(2, len(slot)))) if relation == 0 else
                       len(slot) if relation == 1 else len(slot) + int(rng.integers(1, 7)))
            desired = min(desired, len(main_air) - int(not vent), 14)
        if not vent:
            desired = min(desired, len(main_air) - 1)
        style = int(rng.integers(2))
        while len(probe) < desired:
            frontier = sorted({(r + dr, c + dc) for r, c in probe for dr, dc in D4} & main_air - probe)
            if not frontier:
                break
            weights = np.array([1 + (3 if style == 0 else -0.15) *
                                sum((r + dr, c + dc) in probe for dr, dc in D4) for r, c in frontier])
            p = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
            probe.add(p)
        if len(probe) != desired or _holes(probe) or _canonical(slot) == _canonical(probe):
            continue
        initial_op = 0 if fallback else int(rng.integers(8))
        return _transform((body, slot, probe), initial_op)
    if fallback_index is None:
        for index in range(3):
            try:
                return _prototype(rng, pockets, vent, forbidden, small, fallback_index=index)
            except AssertionError:
                continue
    raise AssertionError("bounded chamber construction exhausted")


def _palette(rng, bg):
    available = [c for c in range(10) if c not in (bg, MARKER)]
    return tuple(map(int, rng.choice(available, 3, replace=False)))


def _pack(rng, footprints, shape):
    h, w = shape
    sizes = [(_bbox(s)[1] + 1, _bbox(s)[3] + 1) for s in footprints]
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1] + rng.random())
    occupied = np.zeros(shape, dtype=bool)
    origins = [None] * len(sizes)
    for i in order:
        mh, mw = sizes[i]
        if mh > h or mw > w:
            return None
        prefix = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = prefix[mh:, mw:] - prefix[:-mh, mw:] - prefix[mh:, :-mw] + prefix[:-mh, :-mw]
        candidates = np.argwhere(sums == 0)
        if not len(candidates):
            return None
        if rng.random() < .45:
            edge = candidates[(candidates[:, 0] == 0) | (candidates[:, 1] == 0)
                              | (candidates[:, 0] == h - mh) | (candidates[:, 1] == w - mw)]
            if len(edge):
                candidates = edge
        r, c = map(int, candidates[int(rng.integers(len(candidates)))])
        origins[i] = r, c
        occupied[max(0, r - 1):min(h, r + mh + 1), max(0, c - 1):min(w, c + mw + 1)] = True
    return origins


def _build_pair(rng, bg, prototypes, colors, parities, fallback=False):
    targets, keys, moves, footprints = [], [], [], []
    for i, ((body, slot, probe), palette, parity) in enumerate(zip(prototypes, colors, parities)):
        op = int(rng.integers(4)) + 4 * parity
        dst_body, dst_slot, dst_probe = _transform((body, slot, probe), op)
        src_slot, src_probe = _transform((slot, probe), 0)
        targets.append({"body": dst_body, "slot": dst_slot, "cells": dst_body | dst_slot,
                        "color": palette[0], "box": _box_cells(dst_body | dst_slot)})
        keys.append((src_slot, src_probe))
        footprints.extend((_box_cells(dst_body | dst_slot), _box_cells(src_slot | src_probe)))
        pc, hc = palette[1:] if parity == 0 else palette[1:][::-1]
        moves.append({"target": i, "key": i, "op": op, "plug": dst_slot, "probe": dst_probe,
                      "plug_color": pc, "probe_color": hc,
                      "original_plug_color": palette[1], "original_probe_color": palette[2]})
    if fallback:
        shape = 30, 30
        origins = [(1 + 9 * i, c) for i in range(len(targets)) for c in (1, 16)]
    else:
        area = sum(len(f) for f in footprints)
        origins = None
        for attempt in range(24):
            shape = ((30, 30) if attempt >= 16 else
                     (int(rng.integers(10, 31)), int(rng.integers(10, 31))))
            if shape[0] * shape[1] < area * 1.35:
                continue
            origins = _pack(rng, footprints, shape)
            if origins is not None:
                break
        if origins is None:
            raise AssertionError("packing failed")
    inp = np.full(shape, bg, dtype=np.int8)
    for i, (target, key, palette, m) in enumerate(zip(targets, keys, colors, moves)):
        tr, tc = origins[2 * i]
        kr, kc = origins[2 * i + 1]
        for name in ("body", "slot", "cells", "box"):
            target[name] = _shift(target[name], tr, tc)
        for name in ("plug", "probe"):
            m[name] = _shift(m[name], tr, tc)
        for p in target["body"]:
            inp[p] = palette[0]
        for p in target["slot"]:
            inp[p] = MARKER
        for region, color in zip(key, palette[1:]):
            for p in _shift(region, kr, kc):
                inp[p] = color
    # Labels are constructed from chosen rigid placements, then independently
    # reconstructed by _match from pixels. Never accept a hidden pairing.
    out = _paint(shape, bg, targets, moves)[0]
    if not fallback and rng.random() < .8:
        used = set(map(tuple, np.argwhere((inp != bg) | (out != bg)).tolist()))
        r0, r1, c0, c1 = _bbox(used)
        margins = list(map(int, rng.integers(0, 4, size=4)))
        r0, r1 = max(0, r0 - margins[0]), min(shape[0] - 1, r1 + margins[1])
        c0, c1 = max(0, c0 - margins[2]), min(shape[1] - 1, c1 + margins[3])
        cropped = inp[r0:r1 + 1, c0:c1 + 1]
        if np.mean(cropped == bg) > .50:
            inp, out = cropped.copy(), out[r0:r1 + 1, c0:c1 + 1].copy()
    _require(np.mean(inp == bg) > .5, "background too dense")
    fitted_bg, fitted_targets, _, fitted_moves = _match(inp)
    _require(sorted(int(m["op"] >= 4) for m in fitted_moves) == sorted(parities),
             "parity witness is obscured by interchangeable key regions")
    _require(np.array_equal(_paint(inp.shape, fitted_bg, fitted_targets, fitted_moves)[0], out),
             "constructed and reconstructed fits disagree")
    return {"input": inp, "output": out}


def _scene_prototypes(rng, n, anchor=False, fallback=False):
    shapes, forbidden = [], set()
    for i in range(n):
        pockets = (2 if i in (0, 2) else 1) if anchor else int(rng.choice([1, 1, 1, 2, 2, 3]))
        if n >= 3 and pockets == 3:
            pockets = 2
        vent = i in (1, 2) if anchor else bool(rng.random() < .45)
        shape = _prototype(rng, pockets, vent, forbidden, n >= 3,
                           fallback_index=i if fallback else None)
        shapes.append(shape)
        forbidden.add(_canonical(shape[1]))
    return shapes


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    bg0 = int(rng.choice([c for c in range(10) if c != MARKER]))
    variable_palette = bool(rng.random() < .7)
    anchor_shapes = _scene_prototypes(rng, 3, anchor=True)
    anchor_colors = [_palette(rng, bg0) for _ in range(3)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            anchor = split == "train" and index < 2
            n = 2 + index if anchor else int(rng.integers(1 if split == "train" else 2, 4))
            if split == "train":
                parity = index if anchor else int(rng.integers(2))
                parities = [parity] * n
            else:
                parities = [0, 1] + [int(rng.integers(2)) for _ in range(n - 2)]
                rng.shuffle(parities)
            bg = bg0 if anchor or not variable_palette else int(rng.choice([c for c in range(10) if c != MARKER]))
            for attempt in range(90):
                fallback = attempt >= 75
                prototypes = (anchor_shapes[:n] if anchor and not fallback else
                              _scene_prototypes(rng, n, anchor, fallback))
                colors = anchor_colors[:n] if anchor else [_palette(rng, bg) for _ in range(n)]
                try:
                    pair = _build_pair(rng, bg, prototypes, colors, parities, fallback)
                except AssertionError:
                    continue
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded unique-scene construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _policy_output(inp, bg, targets, moves, policy):
    altered = []
    for m in moves:
        m = dict(m)
        colors = m["original_plug_color"], m["original_probe_color"]
        if policy[int(m["op"] >= 4)]:
            colors = colors[::-1]
        m["plug_color"], m["probe_color"] = colors
        altered.append(m)
    return _paint(inp.shape, bg, targets, altered)[0]


def _pair_analysis(inp):
    bg, targets, keys, moves = _match(inp)
    expected, zones, halo = _paint(inp.shape, bg, targets, moves)
    return bg, targets, keys, moves, expected, zones, halo


def validate_generated_puzzle(puzzle):
    """Validate returned pixels, not sampled topology labels or source pairings."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "bad counts")
    seen, teaching, policies = set(), set(), {(0, 0), (0, 1), (1, 0), (1, 1)}
    counterexamples = {"before": False, "always": False, "never": False, "no_outline": False}
    multiple_train, hollow_open_train = False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            bg, targets, keys, moves, expected, zones, halo = _pair_analysis(inp)
            _require(np.array_equal(out, expected), "wrong keyed-chamber output")
            # Independently check pixel zones and topology before added outlines.
            all_walls = set().union(*(z["wall"] for z in zones))
            all_holes = set().union(*(z["holes"] for z in zones))
            all_probes = set().union(*(z["probe"] for z in zones))
            _require(_holes(all_walls) == all_holes, "collective or nested enclosure")
            _require(_holes(all_walls, D8) == all_holes, "ambiguous diagonal exterior")
            _require(bool(halo), "outline absent")
            for z in zones:
                _require(len(_components(z["wall"])) == 1, "assembled barrier disconnected")
                _require(not z["probe"] & z["wall"], "probe is not wall material")
                _require(all(int(out[p]) == z["pc"] for p in z["slot"]), "wrong seal color")
                body_color = z["pc"] if z["hollow"] else z["wall_color"]
                _require(all(int(out[p]) == body_color for p in z["body"]), "partial wall recoloring")
                _require(all(int(out[p]) == z["hc"] for p in z["holes"] | z["probe"]), "missing cavity/probe color")
                _require(z["pc"] != z["hc"] and z["wall_color"] not in (z["pc"], z["hc"], bg, MARKER),
                         "local semantic colors collide")
                if split == "train":
                    teaching.add((int(z["move"]["op"] >= 4), z["hollow"]))
                    multiple_train |= len(_components(z["holes"])) >= 2
                    hollow_open_train |= z["hollow"] and bool(z["probe"] - z["holes"])
            # Form dilation by array shifts, rather than using the paint loops.
            wall_mask = np.zeros(inp.shape, dtype=bool)
            for p in all_walls:
                wall_mask[p] = True
            padded = np.pad(wall_mask, 1)
            dilation = np.logical_or.reduce([padded[1 + dr:1 + dr + inp.shape[0],
                                                      1 + dc:1 + dc + inp.shape[1]] for dr, dc in D8])
            for p in all_walls | all_holes | all_probes:
                dilation[p] = False
            actual_halo = frozenset(map(tuple, np.argwhere(dilation).tolist()))
            _require(actual_halo == halo and all(int(out[p]) == MARKER for p in halo), "incorrect exterior halo")
            changed_zones = all_walls | all_holes | all_probes | set(halo)
            _require(all(int(out[r, c]) == bg for r in range(inp.shape[0]) for c in range(inp.shape[1])
                         if (r, c) not in changed_zones), "source key or distant background retained")
            _require(all(all(int(out[p]) == bg for p in k["cells"]) for k in keys), "source key not erased")
            parities = {int(m["op"] >= 4) for m in moves}
            if split == "train":
                _require(len(parities) == 1, "training scenes must demonstrate one parity at a time")
                policies = {p for p in policies if np.array_equal(_policy_output(inp, bg, targets, moves, p), out)}
                for alternative in ("before", "always", "never"):
                    alt = _paint(inp.shape, bg, targets, moves, topology=alternative)[0]
                    counterexamples[alternative] |= not np.array_equal(alt, out)
                counterexamples["no_outline"] |= not np.array_equal(
                    _paint(inp.shape, bg, targets, moves, outline=False)[0], out)
            else:
                _require(parities == {0, 1}, "test must recombine both geometric parities")
    _require(teaching == {(0, False), (0, True), (1, False), (1, True)},
             "training must cross enclosure with both parities")
    _require(multiple_train, "training must demonstrate filling all pockets")
    _require(hollow_open_train, "training must highlight a wall whose probe bay stays open")
    _require(policies == {(0, 1)}, "training does not distinguish parity-to-color policies")
    _require(all(counterexamples.values()), "training permits a simpler alternative: " + str(counterexamples))


def _features(inp):
    bg, targets, keys, moves, out, zones, halo = _pair_analysis(inp)
    records = []
    for z, t in zip(zones, (targets[m["target"]] for m in moves)):
        m = z["move"]
        source = keys[m["key"]]
        initial_holes = _holes(t["body"])
        r0, r1, c0, c1 = _bbox(z["wall"])
        hole_regions = _components(z["holes"])
        irregular = any(len(h) != len(_box_cells(h)) for h in hole_regions)
        records.append({"parity": int(m["op"] >= 4), "operation": m["op"],
                        "holes": len(hole_regions), "initial_holes": len(_components(initial_holes)),
                        "irregular": irregular, "wall_shape": _canonical(z["wall"]),
                        "slot_shape": _canonical(z["slot"]), "probe_shape": _canonical(z["probe"]),
                        "slot_orbit": len({tuple(sorted(_transform((z["slot"],), op)[0])) for op in range(8)}),
                        "relation": int(np.sign(len(z["slot"]) - len(z["probe"]))),
                        "edge": r0 == 0 or c0 == 0 or r1 == inp.shape[0] - 1 or c1 == inp.shape[1] - 1,
                        "bay_stays_open": bool(z["probe"] - z["holes"]),
                        "source_center": tuple(np.mean(np.array(sorted(source["cells"])), axis=0)),
                        "target_center": ((r0 + r1) / 2, (c0 + c1) / 2),
                        "colors": (bg, t["color"], m["plug_color"], m["probe_color"])})
    all_holes = set().union(*(z["holes"] for z in zones))
    occupied = set().union(*(z["wall"] for z in zones))
    later_holes = _holes(occupied | halo)
    # The outlined open door can appear closed, but this never changes classification.
    outline_closes_bay = any(p in later_holes - all_holes for z in zones if not z["hollow"] for p in z["probe"])
    wrong_nearest = False
    for i, a in enumerate(records):
        distance = lambda b: sum(abs(x - y) for x, y in zip(a["source_center"], b["target_center"]))
        wrong_nearest |= any(distance(b) < distance(a) for j, b in enumerate(records) if j != i)
    return {"shape": inp.shape, "count": len(zones), "background": bg,
            "systems": records, "wrong_nearest": wrong_nearest,
            "outline_closes_bay": outline_closes_bay,
            "structure": _signature(inp != bg)}


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded pixel-derived diversity and negative checks across all 42 host seeds."""
    dimensions, structures, walls, slots, probes, palettes = (set() for _ in range(6))
    operations, counts, hole_counts, relations, backgrounds = (set() for _ in range(5))
    observed = {k: 0 for k in ("irregular", "edge", "symmetric_gate", "new_enclosure",
                              "hollow_open_bay", "misleading_proximity", "outline_closes_bay")}
    pairs = 0
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                f = _features(pair["input"])
                dimensions.add(f["shape"])
                structures.add(f["structure"])
                counts.add(f["count"])
                backgrounds.add(f["background"])
                observed["misleading_proximity"] += f["wrong_nearest"]
                observed["outline_closes_bay"] += f["outline_closes_bay"]
                for a in f["systems"]:
                    walls.add(a["wall_shape"])
                    slots.add(a["slot_shape"])
                    probes.add(a["probe_shape"])
                    palettes.add(a["colors"])
                    operations.add(a["operation"])
                    relations.add(a["relation"])
                    hole_counts.add(a["holes"])
                    observed["irregular"] += a["irregular"]
                    observed["edge"] += a["edge"]
                    observed["symmetric_gate"] += a["slot_orbit"] < 8
                    observed["new_enclosure"] += a["holes"] > a["initial_holes"]
                    observed["hollow_open_bay"] += a["holes"] > 0 and a["bay_stays_open"]
    _require(len(dimensions) >= 35 and len(structures) >= 180, "insufficient layout diversity")
    _require(len(walls) >= 100 and len(slots) >= 20 and len(probes) >= 35, "insufficient shape diversity")
    _require(len(palettes) >= 60 and len(backgrounds) >= 8, "insufficient local palette diversity")
    _require(operations == set(range(8)) and counts == {1, 2, 3}, "operation/count diversity")
    _require({0, 1, 2, 3} <= hole_counts and relations == {-1, 0, 1}, "cavity/region size diversity")
    _require(all(observed.values()), "missing structural interaction: " + str(observed))
    # Full explicit-count coverage belongs to verify.py; exercise boundaries
    # here without duplicating its complete call matrix.
    for nt, ne in ((2, 1), (10, 4), (10, 1), (5, 4), (3, 2)):
        p = generate_puzzle(12345, nt, ne)
        _require(len(p["train"]) == nt, "wrong train count")
        _require(len(p["test"]) == ne, "wrong test count")
    p = generate_puzzle(771, 2, 1)
    _require(_equal_puzzles(p, generate_puzzle(771, 2, 1)), "nondeterminism")
    _require(not _equal_puzzles(p, generate_puzzle(772, 2, 1)), "cross-seed repetition")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, False), (4, 1.5),
                   (None, 1), (4, None), (None, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt each meaningful output zone, requiring rejection by the validator.
    p = generate_puzzle(99, 2, 1)
    for zone in ("body", "slot", "holes", "probe", "halo", "source"):
        q = {s: [{k: v.copy() for k, v in a.items()} for a in p[s]] for s in p}
        changed = False
        for pair in q["train"]:
            _, _, keys, _, _, zones, halo = _pair_analysis(pair["input"])
            cells = (halo if zone == "halo" else keys[0]["cells"] if zone == "source" else
                     set().union(*(z[zone] for z in zones)))
            if cells:
                cell = min(cells)
                pair["output"][cell] = (int(pair["output"][cell]) + 1) % 10
                changed = True
                break
        _require(changed, "corruption zone absent")
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted corrupt " + zone)
    # Exercise safe packing fallback and different parity combinations directly.
    rng = np.random.default_rng(14)
    for n in (1, 2, 3):
        shapes = _scene_prototypes(rng, n, anchor=n == 2, fallback=True)
        colors = [_palette(rng, 0) for _ in range(n)]
        parities = [i % 2 for i in range(n)]
        pair = _build_pair(rng, 0, shapes, colors, parities, fallback=True)
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "fallback failure")
    # A straight guide with an off-center probe admits left and right reflected
    # fits in the same bay. Different parity colors make this truly ambiguous.
    ambiguous = np.zeros((18, 18), dtype=np.int8)
    rectangle = set(product(range(7), range(9)))
    air = set(product(range(2, 6), range(1, 8)))
    slot = {(0, 4), (1, 4)}
    for p in _shift(rectangle - air - slot, 2, 1):
        ambiguous[p] = 2
    for p in _shift(slot, 2, 1):
        ambiguous[p] = MARKER
    for p in _shift({(0, 0), (1, 0)}, 12, 13):
        ambiguous[p] = 3
    for p in _shift({(2, 0), (2, 1), (3, 1)}, 12, 13):
        ambiguous[p] = 4
    try:
        solve(ambiguous)
    except AssertionError as error:
        _require("competing" in str(error), "ambiguous fixture rejected for wrong reason")
    else:
        raise AssertionError("competing reflected colored fits accepted")
    return {"seeds_checked": len(CHECK_SEEDS), "pairs_checked": pairs,
            "canvas_shapes": len(dimensions), "normalized_layouts": len(structures),
            "wall_shapes_up_to_D4": len(walls), "gate_shapes_up_to_D4": len(slots),
            "probe_shapes_up_to_D4": len(probes), "local_palettes": len(palettes),
            "operations": sorted(operations), "system_counts": sorted(counts),
            "cavity_counts": sorted(hole_counts), "size_relations": sorted(relations),
            "interactions": observed, "independent_semantic_certification": False}
