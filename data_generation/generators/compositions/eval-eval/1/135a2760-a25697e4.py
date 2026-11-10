"""Woven locks: periodic texture resolves mirror-ambiguous rigid key fitting.

Standard library and NumPy only. No evidence files are accessed at runtime.
"""
from __future__ import annotations

import json
from itertools import combinations
import numpy as np

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PROFILES = ((3, 1), (3, 1, 1), (3, 3, 1), (5, 1), (5, 3), (5, 3, 1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(x):
    a = np.asarray(x)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(frozenset(part))
    return result


def _cells(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _raw(r, c, op):
    # Four orientation-preserving maps followed by four reversing maps.
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (r, -c), (-c, -r), (-r, c), (c, r))[op]


def _transform(regions, op):
    raw = [frozenset(_raw(r, c, op) for r, c in region) for region in regions]
    r0, _, c0, _ = _bbox(frozenset().union(*raw))
    return tuple(_shift(region, -r0, -c0) for region in raw)


def _signature(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _canonical(cells):
    return min(_signature(_transform((cells,), op)[0]) for op in range(8))


def _key(a):
    return a.shape, a.tobytes()


def _recover(motif, bg):
    """Compare all supported periods, ignoring erasures, with strict majorities."""
    colors = tuple(int(c) for c in np.unique(motif) if c != bg)
    if len(colors) != 2:
        raise ValueError("each ribbon must visibly contain two texture colors")
    h, length = motif.shape
    known = motif != bg
    error_limit = max(1, int(known.sum()) // 12)
    answers = {}
    for period in range(1, length // 4 + 1):
        unit = np.empty((h, period), dtype=np.int8)
        errors, supported = 0, True
        for r in range(h):
            for phase in range(period):
                values = motif[r, phase::period]
                values = values[values != bg]
                n0 = int(np.count_nonzero(values == colors[0]))
                n1 = len(values) - n0
                if len(values) < 3 or n0 == n1:
                    supported = False
                    break
                unit[r, phase] = colors[int(n1 > n0)]
                errors += min(n0, n1)
            if not supported:
                break
        if not supported or errors > error_limit:
            continue
        clean = unit[:, np.arange(length) % period]
        if set(int(c) for c in np.unique(clean)) != set(colors):
            continue
        signature = clean.tobytes()
        if signature not in answers:
            answers[signature] = (clean, period, errors)
    if len(answers) != 1:
        raise ValueError("periodic repair is absent or has competing answers")
    return next(iter(answers.values()))


def _parse(inp):
    bg = int(inp[0, 0])
    if not (np.all(inp[0] == bg) and np.all(inp[-1] == bg)
            and np.all(inp[:, 0] == bg) and np.all(inp[:, -1] == bg)):
        raise ValueError("a one-cell background perimeter is required")
    interpretations = []
    for marker in (int(c) for c in np.unique(inp) if c != bg):
        bars = _components(_cells(inp == marker))
        if len(bars) not in (2, 4):
            continue
        descriptions = []
        for bar in bars:
            r0, r1, c0, c1 = _bbox(bar)
            if c0 == c1 and 3 <= len(bar) <= 6:
                descriptions.append(("h", r0, r1, c0, c1))
            elif r0 == r1 and 3 <= len(bar) <= 6:
                descriptions.append(("v", r0, r1, c0, c1))
            else:
                break
        if len(descriptions) != len(bars):
            continue
        pairings = []
        for i, j in combinations(range(len(bars)), 2):
            a, b = descriptions[i], descriptions[j]
            if a[0] != b[0]:
                continue
            if a[0] == "h" and a[1:3] == b[1:3]:
                c0, c1 = sorted((a[3], b[3]))
                rect = (a[1], a[2], c0 + 1, c1 - 1)
            elif a[0] == "v" and a[3:5] == b[3:5]:
                r0, r1 = sorted((a[1], b[1]))
                rect = (r0 + 1, r1 - 1, a[3], a[4])
            else:
                continue
            r0, r1, c0, c1 = rect
            length = c1 - c0 + 1 if a[0] == "h" else r1 - r0 + 1
            if 12 <= length <= 26:
                pairings.append((i, j, a[0], rect))
        # Rails must have a unique complete pairing; no order convention.
        possibilities = []
        for selected in combinations(pairings, len(bars) // 2):
            if sorted(k for p in selected for k in p[:2]) == list(range(len(bars))):
                possibilities.append(selected)
        for pairing in possibilities:
            try:
                fields, holes, field_cells = [], [], set()
                for i, j, axis, rect in pairing:
                    r0, r1, c0, c1 = rect
                    region = frozenset((r, c) for r in range(r0, r1 + 1)
                                       for c in range(c0, c1 + 1))
                    if region & field_cells or region & _cells(inp == marker):
                        raise ValueError("overlapping rails or fields")
                    field_cells.update(region)
                    motif = inp[r0:r1 + 1, c0:c1 + 1]
                    normalized = motif if axis == "h" else motif.T
                    clean, period, errors = _recover(normalized, bg)
                    clean_absolute = clean if axis == "h" else clean.T
                    parts = _components(p for p in region if inp[p] == bg)
                    if not 1 <= len(parts) <= 3:
                        raise ValueError("a ribbon needs one to three connected tears")
                    colors = tuple(int(c) for c in np.unique(clean))
                    field_index = len(fields)
                    for part in parts:
                        wanted = {int(clean_absolute[r - r0, c - c0]) for r, c in part}
                        if len(wanted) != 1:
                            raise ValueError("a tear must be monochrome after texture recovery")
                        if axis == "h":
                            touches = {r for r, c in part if r in (r0, r1)}
                        else:
                            touches = {c for r, c in part if c in (c0, c1)}
                        if len(touches) != 1:
                            raise ValueError("tear must reach exactly one open ribbon edge")
                        holes.append({"cells": part, "field": field_index,
                                      "wanted": next(iter(wanted)), "colors": colors})
                    fields.append({"axis": axis, "rect": rect, "cells": region,
                                   "clean": clean_absolute, "period": period,
                                   "errors": errors, "colors": colors})
                outside = _cells(inp != bg) - frozenset(field_cells) - _cells(inp == marker)
                keys = []
                for component in _components(outside):
                    colors = tuple(sorted({int(inp[p]) for p in component}))
                    if len(colors) != 2:
                        raise ValueError("a loose key must have two colors")
                    regions = tuple(frozenset(p for p in component if inp[p] == color)
                                    for color in colors)
                    if any(len(_components(region)) != 1 for region in regions):
                        raise ValueError("both key color regions must be connected")
                    keys.append({"cells": component, "regions": regions, "colors": colors})
                if len(keys) != len(holes) or not 1 <= len(keys) <= 4:
                    raise ValueError("one loose key is needed per tear, at most four")
                fixed = frozenset(field_cells) | _cells(inp == marker)
                interpretations.append({"input": inp, "bg": bg, "marker": marker,
                                        "fields": fields, "holes": holes, "keys": keys,
                                        "fixed": fixed})
            except ValueError:
                continue
    if len(interpretations) != 1:
        raise ValueError("input does not have one unambiguous ribbon interpretation")
    return interpretations[0]


def _options(a, use_texture=True, policy=(0, 1), label=None):
    inp, bg = a["input"], a["bg"]
    by_key = []
    for ki, key in enumerate(a["keys"]):
        candidates, seen = [], set()
        for hi, hole in enumerate(a["holes"]):
            if key["colors"] != hole["colors"]:
                continue
            hr0, _, hc0, _ = _bbox(hole["cells"])
            for op in range(8):
                regions = _transform(key["regions"], op)
                colors = key["colors"][::-1] if policy[int(op >= 4)] else key["colors"]
                for pi in (0, 1):
                    pr0, _, pc0, _ = _bbox(regions[pi])
                    placed = tuple(_shift(region, hr0 - pr0, hc0 - pc0) for region in regions)
                    if placed[pi] != hole["cells"]:
                        continue
                    if use_texture and colors[pi] != hole["wanted"]:
                        continue
                    handle = placed[1 - pi]
                    if (handle & a["fixed"] or any(
                            not (1 <= r < inp.shape[0] - 1 and 1 <= c < inp.shape[1] - 1)
                            or int(inp[r, c]) != bg for r, c in handle)):
                        continue
                    paint = tuple(sorted((p, colors[j]) for j in (0, 1) for p in placed[j]))
                    if label is not None and any(int(label[p]) != color for p, color in paint):
                        continue
                    signature = (hi, paint)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    candidates.append({"key": ki, "hole": hi, "op": op, "plug_index": pi,
                                       "regions": placed, "handle": handle,
                                       "paint": paint, "full": placed[0] | placed[1]})
        by_key.append(candidates)
    return by_key


def _assemblies(a, use_texture=True, policy=(0, 1), label=None, limit=2):
    """Enumerate complete colored answers, deduplicating identical descriptions."""
    base = a["input"].copy()
    for key in a["keys"]:
        for p in key["cells"]:
            base[p] = a["bg"]
    for field in a["fields"]:
        r0, r1, c0, c1 = field["rect"]
        base[r0:r1 + 1, c0:c1 + 1] = field["clean"]
    by_key = _options(a, use_texture, policy, label)
    order = sorted(range(len(by_key)), key=lambda i: len(by_key[i]))
    answers = {}

    def search(depth, used_holes, occupied, moves):
        if len(answers) >= limit:
            return
        if depth == len(order):
            out = base.copy()
            for move in moves:
                for p, color in move["paint"]:
                    out[p] = color
            if label is None or np.array_equal(out, label):
                answers.setdefault(_key(out), (out, tuple(moves)))
            return
        for move in by_key[order[depth]]:
            if move["hole"] in used_holes or move["full"] & occupied:
                continue
            search(depth + 1, used_holes | {move["hole"]},
                   occupied | move["full"], moves + [move])

    search(0, set(), frozenset(), [])
    return list(answers.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _parse(_grid(input_grid))
    answers = _assemblies(a)
    if len(answers) != 1:
        raise ValueError("rigid assembly does not have a unique colored answer")
    return answers[0][0]


def _cap(profile):
    width = profile[0]
    return frozenset((r, c) for r, n in enumerate(profile)
                     for c in range((width - n) // 2, (width + n) // 2))


def _handle(rng, plug, reach):
    width = max(c for r, c in plug) + 1
    start_c = int(rng.choice((0, width - 1)))
    handle = {(-1, start_c)}
    capacity = reach * 4
    relation = int(rng.integers(3))
    target = (int(rng.integers(2, max(3, len(plug)))) if relation == 0 else
              len(plug) if relation == 1 else len(plug) + int(rng.integers(1, 4)))
    target = min(target, capacity, 10)
    for _ in range(target - 1):
        frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in DIR4
                           if -reach <= r + dr < 0 and start_c - 1 <= c + dc <= start_c + 2}
                          - handle)
        if not frontier:
            raise ValueError("handle growth exhausted")
        # Walk-like and compact attachments both arise from this growth.
        p = frontier[int(rng.integers(len(frontier)))]
        handle.add(p)
    if frozenset((r, width - 1 - c) for r, c in handle) == frozenset(handle):
        raise ValueError("handle must expose the chosen handedness")
    return frozenset(handle)


def _make_field(rng, count, profiles, palette, period, length, thickness, parities,
                witness=False):
    bg, marker, color0, color1 = palette
    reach = int(rng.integers(1, 4))
    side = int(rng.integers(2))  # all tears in this ribbon use one of its open edges
    unit = rng.integers(0, 2, size=(thickness, period), dtype=np.int8)
    restrictions, holes, handles = {}, [], []
    reserved = set()
    neighbor_cells = set()
    for i, profile in enumerate(profiles):
        cap = _cap(profile)
        w, depth = profile[0], len(profile)
        if depth >= thickness:
            raise ValueError("a tear must leave texture beneath it")
        choices = list(range(2, length - w - 1))
        rng.shuffle(choices)
        for x in choices:
            slot = frozenset((r if side == 0 else thickness - 1 - r, c + x) for r, c in cap)
            moat = {(r + dr, c + dc) for r, c in slot for dr, dc in ((0, 0),) + DIR4}
            if moat & reserved:
                continue
            bit = int(rng.integers(2))
            wanted = {(r, c % period): bit for r, c in slot}
            if witness and i == 0:
                edge = 0 if side == 0 else thickness - 1
                for c in (x - 1, x + w):
                    group = (edge, c % period)
                    if group in wanted and wanted[group] != 1 - bit:
                        break
                    wanted[group] = 1 - bit
                else:
                    pass
                if any(wanted[(r, c % period)] != bit for r, c in slot):
                    continue
            if any(group in restrictions and restrictions[group] != value
                   for group, value in wanted.items()):
                continue
            restrictions.update(wanted)
            reserved.update(slot)
            if witness and i == 0:
                neighbor_cells.update(((0 if side == 0 else thickness - 1, x - 1),
                                       (0 if side == 0 else thickness - 1, x + w)))
            handle = _handle(rng, cap, reach)
            handle = frozenset((r if side == 0 else thickness - 1 - r, c + x)
                               for r, c in handle)
            holes.append((slot, bit, int(parities[i])))
            handles.append(handle)
            break
        else:
            raise ValueError("could not carve compatible monochrome tears")
    for (r, phase), bit in restrictions.items():
        unit[r, phase] = bit
    clean_bits = unit[:, np.arange(length) % period]
    clean = np.where(clean_bits == 0, color0, color1).astype(np.int8)
    noisy = clean.copy()
    erased = frozenset().union(*(slot for slot, bit, parity in holes))
    for p in erased:
        noisy[p] = bg
    if any(np.count_nonzero(noisy[r, phase::period] != bg) < 3
           for r in range(thickness) for phase in range(period)):
        raise ValueError("too few surviving repetitions")
    error_groups = [(r, phase) for r in range(thickness) for phase in range(period)]
    rng.shuffle(error_groups)
    flipped = 0
    target_errors = int(rng.integers(1, 4))
    for r, phase in error_groups:
        positions = [c for c in range(phase, length, period)
                     if (r, c) not in erased and (r, c) not in neighbor_cells]
        if len(positions) < 3:
            continue
        c = int(rng.choice(positions))
        noisy[r, c] = color1 if noisy[r, c] == color0 else color0
        flipped += 1
        if flipped == target_errors:
            break
    repaired, recovered_period, _ = _recover(noisy, bg)
    if not flipped or recovered_period != period or not np.array_equal(repaired, clean):
        raise ValueError("template does not expose its intended primitive period")
    top = reach if side == 0 else 0
    field_in = np.full((thickness + reach, length + 2), bg, dtype=np.int8)
    field_out = field_in.copy()
    field_in[top:top + thickness, 1:-1] = noisy
    field_out[top:top + thickness, 1:-1] = clean
    field_in[top:top + thickness, (0, -1)] = marker
    field_out[top:top + thickness, (0, -1)] = marker
    keys, destination_buffer = [], set()
    for (slot, bit, parity), handle in zip(holes, handles):
        plug_color = (color0, color1)[bit]
        other_color = (color1, color0)[bit]
        absolute = (_shift(slot, top, 1), _shift(handle, top, 1))
        for p in absolute[1]:
            field_out[p] = other_color
        # Reserve both mirror-compatible handle destinations, not just the answer.
        r0, r1, c0, c1 = _bbox(absolute[0])
        destination_buffer.update(absolute[1])
        destination_buffer.update((r, c0 + c1 - c) for r, c in absolute[1])
        op = int(rng.integers(4)) + 4 * parity
        incoming = _transform(absolute, op)
        colors = (plug_color, other_color) if parity == 0 else (other_color, plug_color)
        cells = incoming[0] | incoming[1]
        _, kr1, _, kc1 = _bbox(cells)
        key = np.full((kr1 + 1, kc1 + 1), bg, dtype=np.int8)
        for region, color in zip(incoming, colors):
            for p in region:
                key[p] = color
        keys.append(key)
    # The rectangular packing footprint contains both possible handles.
    if any(not (0 <= r < field_in.shape[0] and 0 <= c < field_in.shape[1])
           for r, c in destination_buffer):
        raise ValueError("attachment exceeds the reserved field footprint")
    return field_in, field_out, keys


def _pack(rng, items):
    """Independent rectangle packing inside a guaranteed background perimeter."""
    order = sorted(range(len(items)), key=lambda i: -items[i].size + float(rng.random()))
    for _ in range(20):
        occupied = np.zeros((30, 30), dtype=bool)
        positions = [None] * len(items)
        for i in order:
            h, w = items[i].shape
            options = [(r, c) for r in range(1, 30 - h) for c in range(1, 30 - w)]
            rng.shuffle(options)
            for r, c in options:
                if occupied[r:r + h, c:c + w].any():
                    continue
                positions[i] = (r, c)
                occupied[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
                break
            else:
                break
        else:
            return positions
    # Bounded shelf fallback; randomized shelves and key order preserve variety.
    fields = [i for i in order if items[i].shape[1] >= 14]
    keys = [i for i in order if i not in fields]
    rng.shuffle(fields)
    rng.shuffle(keys)
    shelves = [[i] for i in fields] + [keys]
    rng.shuffle(shelves)
    positions = [None] * len(items)
    r = 1
    for shelf in shelves:
        c = 1
        if not shelf:
            continue
        shelf_h = max(items[i].shape[0] for i in shelf)
        for i in shelf:
            h, w = items[i].shape
            if r + h >= 30 or c + w >= 30:
                raise ValueError("shelf fallback cannot fit these objects")
            positions[i] = (r, c)
            c += w + 1
        r += shelf_h + 1
    return positions


def _build_pair(rng, palettes, fields_count, holes_count, first_parity, test=False):
    bg, marker = palettes[:2]
    counts = ([holes_count] if fields_count == 1 else
              [2, 2] if holes_count == 4 else [1, holes_count - 1])
    # Unique slot silhouettes force association by shape even with shared palettes.
    profiles = [(3, 1)]
    # Other tears on the witness ribbon must fit its isolated three-cell
    # color run. Wider caps remain available on the other ribbon.
    narrow = list(PROFILES[1:3])
    rng.shuffle(narrow)
    profiles.extend(narrow[:counts[0] - 1])
    rest = [profile for profile in PROFILES if profile not in profiles]
    rng.shuffle(rest)
    profiles.extend(rest[:holes_count - len(profiles)])
    textures, keys = [], []
    cursor = 0
    for fi, n in enumerate(counts):
        selected = profiles[cursor:cursor + n]
        depth = max(map(len, selected))
        thickness = int(rng.integers(depth + 1, 5 if fields_count == 2 else 7))
        period_choices = [p for p in (4, 5, 6) if (n + 3) * p <= 26]
        period = int(rng.choice(period_choices))
        if fields_count == 2 and fi == 1:
            period = int(rng.choice([p for p in period_choices if p != textures[0][3]]))
        min_length = max((n + 3) * period, 20 if test else 16)
        length = int(rng.integers(min_length, 27))
        colors = palettes[2:4] if fi == 0 or palettes[4] == 0 else palettes[5:7]
        parities = [int(rng.integers(2)) for _ in range(n)]
        if fi == 0:
            parities[0] = first_parity
        if test and fi == 1:
            parities[0] = 1 - first_parity
        field_in, field_out, loose = _make_field(
            rng, n, selected, (bg, marker, *colors), period, length, thickness,
            parities, witness=fi == 0)
        textures.append((field_in, field_out, loose, period))
        keys.extend(loose)
        cursor += n
    items = [f[0] for f in textures] + keys
    positions = _pack(rng, items)
    inp = np.full((30, 30), bg, dtype=np.int8)
    out = inp.copy()
    for i, (r, c) in enumerate(positions):
        h, w = items[i].shape
        inp[r:r + h, c:c + w] = items[i]
        if i < fields_count:
            out[r:r + h, c:c + w] = textures[i][1]
    used = _cells((inp != bg) | (out != bg))
    r0, r1, c0, c1 = _bbox(used)
    # Keep at least one background cell around every side. Optional extra margins.
    margins = [int(rng.integers(1, 4)) for _ in range(4)]
    r0, r1 = max(0, r0 - margins[0]), min(29, r1 + margins[1])
    c0, c1 = max(0, c0 - margins[2]), min(29, c1 + margins[3])
    inp, out = inp[r0:r1 + 1, c0:c1 + 1].copy(), out[r0:r1 + 1, c0:c1 + 1].copy()
    # Incoming keys have independent orientations; the entire scene may rotate.
    turns = int(rng.integers(4))
    return {"input": np.rot90(inp, turns).copy(), "output": np.rot90(out, turns).copy()}


def _pair(rng, palette, fields, holes, parity, test):
    for _ in range(180):
        try:
            pair = _build_pair(rng, palette, fields, holes, parity, test)
            a = _parse(pair["input"])
            answers = _assemblies(a)
            if len(answers) != 1 or not np.array_equal(answers[0][0], pair["output"]):
                continue
            if len(_assemblies(a, use_texture=False)) != 2:
                continue
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded woven-lock construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("counts must satisfy 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    stable = bool(rng.integers(3) == 0)
    base_palette = tuple(int(c) for c in rng.permutation(10))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            palette = base_palette if stable else tuple(int(c) for c in rng.permutation(10))
            # Explicit palette roles: bg, rail, first binary pair, reuse flag,
            # second binary pair. Numeric identities have no solving significance.
            roles = (*palette[:4], int(rng.integers(2)), *palette[4:6])
            if split == "train" and i == 0:
                fields, holes, parity = 1, 1, 0
            elif split == "train" and i == 1:
                fields, holes, parity = 2, 2, 1
            elif split == "test":
                fields, holes, parity = 2, int(rng.integers(3, 5)), int(rng.integers(2))
            else:
                fields = int(rng.integers(1, 3))
                holes = int(rng.integers(fields, min(3, fields + 1) + 1))
                parity = int(rng.integers(2))
            for _ in range(8):
                pair = _pair(rng, roles, fields, holes, parity, split == "test")
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _compatible_policies(train):
    compatible = []
    for policy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if all(_assemblies(_parse(_grid(pair["input"])), policy=policy,
                           label=_grid(pair["output"]), limit=1) for pair in train):
            compatible.append(policy)
    return compatible


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train/test keys")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    if not (2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
            and len(puzzle["train"]) > len(puzzle["test"])):
        raise ValueError("illegal pair counts")
    seen, train_parities, train_fields, test_fields = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pairs must have exactly input/output keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            if inp.shape != out.shape or _key(inp) in seen:
                raise ValueError("duplicate inputs or changed dimensions")
            seen.add(_key(inp))
            a = _parse(inp)
            answers = _assemblies(a)
            if len(answers) != 1 or not np.array_equal(answers[0][0], out):
                raise ValueError("output is not the unique input-derived answer")
            moves = answers[0][1]
            if len(_assemblies(a, use_texture=False)) != 2:
                raise ValueError("geometry alone must leave competing colored answers")
            allowed = set()
            for field in a["fields"]:
                if field["errors"] < 1:
                    raise ValueError("each generated ribbon must demonstrate sparse flip repair")
                r0, r1, c0, c1 = field["rect"]
                if not np.array_equal(out[r0:r1 + 1, c0:c1 + 1], field["clean"]):
                    raise ValueError("completed ribbon must be periodic")
                allowed.update(field["cells"])
            for key in a["keys"]:
                if any(out[p] != a["bg"] for p in key["cells"]):
                    raise ValueError("loose key source was not erased")
                allowed.update(key["cells"])
            for move in moves:
                key = a["keys"][move["key"]]
                if sum(map(len, move["regions"])) != len(key["cells"]):
                    raise ValueError("key area must be conserved")
                allowed.update(move["full"])
                if split == "train":
                    train_parities.add(int(move["op"] >= 4))
            if any(tuple(p) not in allowed for p in np.argwhere(inp != out)):
                raise ValueError("changed rails or unrelated background")
            (train_fields if split == "train" else test_fields).add(len(a["fields"]))
            if split == "test" and (len(a["fields"]) != 2 or len(a["holes"]) < 3):
                raise ValueError("test must combine two textures and at least three keys")
    if train_parities != {0, 1} or train_fields != {1, 2} or test_fields != {2}:
        raise ValueError("demonstrations must expose both parities and both ribbon counts")
    if _compatible_policies(puzzle["train"]) != [(0, 1)]:
        raise ValueError("training must distinguish all four parity-to-swap conventions")


def self_test():
    shapes, normalized, periods, slot_shapes, handle_shapes = set(), set(), set(), set(), set()
    axes, key_counts, relations, operations, palettes, signatures = set(), set(), set(), set(), set(), set()
    partial, shared, island_witnesses = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                a = _parse(inp)
                out, moves = _assemblies(a)[0]
                assert np.array_equal(out, pair["output"])
                # Omitting B's protruding rigid region gives a different answer.
                without_handles = out.copy()
                for move in moves:
                    for p in move["handle"]:
                        without_handles[p] = a["bg"]
                assert not np.array_equal(without_handles, out)
                assert len(_assemblies(a, use_texture=False)) == 2
                shapes.add(inp.shape)
                roles = {}
                normalized.add((inp.shape, tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                palettes.add(tuple(int(c) for c in np.unique(inp)))
                key_counts.add(len(moves))
                shared += len(a["fields"]) == 2 and a["fields"][0]["colors"] == a["fields"][1]["colors"]
                for field in a["fields"]:
                    axes.add(field["axis"])
                    periods.add(field["period"])
                    length = field["clean"].shape[1 if field["axis"] == "h" else 0]
                    partial += length % field["period"] != 0
                for move in moves:
                    operations.add(move["op"])
                    key = a["keys"][move["key"]]
                    pi = move["plug_index"]
                    slot_shapes.add(_canonical(key["regions"][pi]))
                    handle_shapes.add(_canonical(key["regions"][1 - pi]))
                    relations.add(int(np.sign(len(key["regions"][pi]) - len(key["regions"][1 - pi]))))
                    hole = a["holes"][move["hole"]]
                    field = a["fields"][hole["field"]]
                    coords = hole["cells"] if field["axis"] == "h" else frozenset((c, r) for r, c in hole["cells"])
                    r0, r1, c0, c1 = _bbox(coords)
                    for r in (r0, r1):
                        if all((r, c) in coords for c in range(c0, c1 + 1)):
                            neighbors = ((r, c0 - 1), (r, c1 + 1))
                            if field["axis"] == "v":
                                neighbors = tuple((c, r) for r, c in neighbors)
                            island_witnesses += all(inp[p] not in (a["bg"], hole["wanted"], a["marker"])
                                                    for p in neighbors)
    assert len(signatures) == 42 and len(normalized) >= 200
    assert len(shapes) >= 25 and len(palettes) >= 10
    assert axes == {"h", "v"} and periods == {4, 5, 6}
    assert key_counts == {1, 2, 3, 4} and operations == set(range(8))
    assert len(slot_shapes) == len(PROFILES) and len(handle_shapes) >= 20
    assert relations == {-1, 0, 1} and partial > 50 and shared > 20
    assert island_witnesses >= 42
    p = generate_puzzle(2027, 2, 1)
    q = generate_puzzle(2027, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
               for k in ("input", "output"))
    reordered = {"train": list(reversed(p["train"])), "test": p["test"]}
    validate_generated_puzzle(reordered)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generated = generate_puzzle(31415 + 10 * nt + ne, nt, ne)
            assert len(generated["train"]) == nt and len(generated["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(7, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("bad output accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
