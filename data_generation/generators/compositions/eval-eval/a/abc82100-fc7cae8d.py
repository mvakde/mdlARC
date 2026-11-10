"""Registered interference stencils: input-local frames and simultaneous XOR ink.

Only NumPy and Python's standard library are used. No parent code is imported.
"""
from __future__ import annotations

import copy
import json
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
DIRECTIONS = tuple(NORMAL.values())
MATRICES = (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)),
    ((0, 1), (-1, 0)), ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)), ((0, -1), (-1, 0)),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(a):
    return tuple(a.shape), a.astype(np.uint8).tobytes()


def _move(matrix, vector):
    r, c = vector
    return (int(matrix[0][0] * r + matrix[0][1] * c),
            int(matrix[1][0] * r + matrix[1][1] * c))


def _side_image(matrix, side):
    normal = _move(matrix, NORMAL[side])
    return next(s for s in SIDES if NORMAL[s] == normal)


def _frame_matrix(inner, outer):
    possibilities = [m for m in MATRICES
                     if all(outer.get(_side_image(m, s)) == color
                            for s, color in inner.items())]
    if len(possibilities) != 1:
        raise ValueError("side identities do not define one symmetry")
    return possibilities[0]


def _side_line(a, side):
    return {"T": a[0, 1:-1], "R": a[1:-1, -1],
            "B": a[-1, 1:-1], "L": a[1:-1, 0]}[side]


def _guide_positions(bounds, side):
    r, c, h, w = bounds
    if side == "T":
        return [(r - 1, j) for j in range(c, c + w)]
    if side == "B":
        return [(r + h, j) for j in range(c, c + w)]
    if side == "L":
        return [(i, c - 1) for i in range(r, r + h)]
    return [(i, c + w) for i in range(r, r + h)]


def _components(a, color):
    remaining = {tuple(map(int, p)) for p in np.argwhere(a == color)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _read_panel(a, bounds, guides):
    r, c, h, w = bounds
    panel = a[r:r + h, c:c + w]
    colors = set(map(int, np.unique(panel))) - {0}
    if len(colors) != 3 or colors & set(guides.values()):
        raise ValueError("a stencil rectangle needs neutral cells and two labels")
    neutral_candidates = [v for v in colors if np.count_nonzero(panel == v) >= 2]
    if len(neutral_candidates) != 1:
        raise ValueError("neutral stencil color is ambiguous")
    neutral = neutral_candidates[0]
    mask = {tuple(map(int, p)) for p in np.argwhere(panel == neutral)}
    labels = [tuple(map(int, p)) for p in np.argwhere((panel != 0) & (panel != neutral))]
    if len(labels) != 2 or panel[labels[0]] == panel[labels[1]]:
        raise ValueError("a stencil needs exactly two distinct label pixels")
    readings = []
    for source_position, ink_position in (labels, labels[::-1]):
        dr = ink_position[0] - source_position[0]
        dc = ink_position[1] - source_position[1]
        if (dr, dc) not in DIRECTIONS:
            continue
        q = (ink_position[0] + dr, ink_position[1] + dc)
        if not (0 <= q[0] < h and 0 <= q[1] < w):
            continue
        if panel[q] not in (0, neutral):
            continue
        if min(max(abs(x - q[0]), abs(y - q[1])) for x, y in mask) > 1:
            continue
        readings.append(dict(source=int(panel[source_position]), ink=int(panel[ink_position]),
                             q=(r + q[0], c + q[1]), direction=(dr, dc),
                             offsets=tuple(sorted((x - q[0], y - q[1]) for x, y in mask))))
    if len(readings) != 1:
        raise ValueError("source/ink order or registration is ambiguous")
    return dict(readings[0], neutral=neutral, bounds=bounds, guides=guides)


def _decode(input_grid):
    a = _grid(input_grid)
    H, W = a.shape
    if min(H, W) < 10 or any(a[p] != 0 for p in ((0, 0), (0, W-1), (H-1, 0), (H-1, W-1))):
        raise ValueError("work frame needs four blank corners and room for stencils")
    outer = {}
    for side in SIDES:
        line = _side_line(a, side)
        if np.all(line == 0):
            continue
        if not np.all(line == line[0]) or line[0] == 0:
            raise ValueError("outer controls must be complete straight lines")
        outer[side] = int(line[0])
    if (len(outer) != 2 or len(set(outer.values())) != 2
            or not any(set(outer) == {SIDES[i], SIDES[(i + 1) % 4]} for i in range(4))):
        raise ValueError("expected two differently colored adjacent outer controls")
    bars = []
    for color in outer.values():
        for component in _components(a, color):
            if any(r in (0, H-1) or c in (0, W-1) for r, c in component):
                # No inward protrusions or connected interior control fragments.
                if any(0 < r < H-1 and 0 < c < W-1 for r, c in component):
                    raise ValueError("an inner guide touches the outer controls")
                continue
            rows = sorted({r for r, c in component})
            cols = sorted({c for r, c in component})
            if len(rows) == 1 and len(component) == cols[-1] - cols[0] + 1 and len(component) >= 4:
                bars.append(dict(axis="H", fixed=rows[0], start=cols[0], end=cols[-1],
                                 color=color, cells=component))
            elif len(cols) == 1 and len(component) == rows[-1] - rows[0] + 1 and len(component) >= 4:
                bars.append(dict(axis="V", fixed=cols[0], start=rows[0], end=rows[-1],
                                 color=color, cells=component))
            else:
                raise ValueError("inner guides must be isolated straight bars of length >=4")
    records, used_bars = [], set()
    for hi, horizontal in enumerate(bars):
        if horizontal["axis"] != "H":
            continue
        for vi, vertical in enumerate(bars):
            if vertical["axis"] != "V" or horizontal["color"] == vertical["color"]:
                continue
            hr, hs, he = horizontal["fixed"], horizontal["start"], horizontal["end"]
            vc, vs, ve = vertical["fixed"], vertical["start"], vertical["end"]
            hside = "T" if vs == hr + 1 else "B" if ve == hr - 1 else None
            vside = "L" if vc == hs - 1 else "R" if vc == he + 1 else None
            if hside is None or vside is None:
                continue
            bounds = (vs, hs, ve - vs + 1, he - hs + 1)
            guides = {hside: horizontal["color"], vside: vertical["color"]}
            try:
                record = _read_panel(a, bounds, guides)
            except ValueError:
                continue
            if hi in used_bars or vi in used_bars:
                raise ValueError("a guide participates in competing rectangles")
            used_bars.update((hi, vi))
            record["matrix"] = _frame_matrix(guides, outer)
            records.append(record)
    if not records or len(used_bars) != len(bars):
        raise ValueError("every inner guide must identify one stencil rectangle")
    records.sort(key=lambda x: x["bounds"])
    if len({x["neutral"] for x in records}) != 1 or len({x["ink"] for x in records}) != 1:
        raise ValueError("all local stencils must share one neutral color and one ink color")
    neutral, ink = records[0]["neutral"], records[0]["ink"]
    sources = {x["source"] for x in records}
    if len(sources) != len(records) or sources & {ink, neutral, 0, *outer.values()}:
        raise ValueError("sources must be distinct and disjoint from other roles")
    if ink in {neutral, *outer.values()}:
        raise ValueError("ink and control roles must be distinct")
    reserved, mask_cells, guide_cells = set(), set(), set()
    for record in records:
        r, c, h, w = record["bounds"]
        if r < 2 or c < 2 or r+h > H-2 or c+w > W-2:
            raise ValueError("stencils must be inside the work frame")
        box = {(i, j) for i in range(r-1, r+h+1) for j in range(c-1, c+w+1)}
        if reserved & box:
            raise ValueError("stencil rectangles or their clear rings overlap")
        reserved |= box
        for side in SIDES:
            for p in _guide_positions(record["bounds"], side):
                expected = record["guides"].get(side, 0)
                if a[p] != expected:
                    raise ValueError("broken guides or nonblank unused panel sides")
                if expected:
                    guide_cells.add(p)
        if any(a[p] != 0 for p in ((r-1, c-1), (r-1, c+w), (r+h, c-1), (r+h, c+w))):
            raise ValueError("guide corners must be blank")
        mask_cells |= {(record["q"][0]+dr, record["q"][1]+dc) for dr, dc in record["offsets"]}
    if {tuple(map(int, p)) for p in np.argwhere(a == neutral)} != mask_cells:
        raise ValueError("neutral cells occur outside stencil rectangles")
    for record in records:
        record["anchors"] = tuple(tuple(map(int, p)) for p in np.argwhere(a == record["source"])
                                  if tuple(p) not in reserved)
        if not record["anchors"]:
            raise ValueError("each stencil needs at least one external source pixel")
    return dict(grid=a, outer=outer, records=records, neutral=neutral, ink=ink,
                reserved=reserved, guide_cells=guide_cells)


def _coverage(info, native=False, pixels_only=False, component_only=False):
    H, W = info["grid"].shape
    visits = np.zeros((H-2, W-2), dtype=np.int16)
    clipped = 0
    for record in info["records"]:
        offsets = ((0, 0),) if pixels_only else record["offsets"]
        offsets = tuple(_move(MATRICES[0] if native else record["matrix"], p) for p in offsets)
        anchors = record["anchors"]
        if component_only:
            remaining = set(anchors)
            selected = []
            while remaining:
                start = min(remaining)
                remaining.remove(start)
                selected.append(start)
                stack = [start]
                while stack:
                    r, c = stack.pop()
                    for dr, dc in DIRECTIONS:
                        p = (r+dr, c+dc)
                        if p in remaining:
                            remaining.remove(p)
                            stack.append(p)
            anchors = selected
        for ar, ac in anchors:
            for dr, dc in offsets:
                r, c = ar + dr, ac + dc
                if 1 <= r < H-1 and 1 <= c < W-1:
                    visits[r-1, c-1] += 1
                else:
                    clipped += 1
    return visits, clipped


def _base(info):
    a = info["grid"].copy()
    for record in info["records"]:
        r, c, h, w = record["bounds"]
        a[r-1:r+h+1, c-1:c+w+1] = 0
        for p in record["anchors"]:
            a[p] = 0
    return a[1:-1, 1:-1].copy()


def _render(info, native=False, pixels_only=False, component_only=False, union=False):
    result = _base(info)
    visits, _ = _coverage(info, native, pixels_only, component_only)
    if not native and np.any((visits > 0) & (result != 0) & (result != info["ink"])):
        raise ValueError("a footprint hits inert non-ink content")
    if union:
        result[visits > 0] = info["ink"]
    else:
        active = (visits % 2) == 1
        result[active] = np.where(result[active] == info["ink"], 0, info["ink"])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read local labels and frames; count toggles from original input anchors."""
    return _render(_decode(input_grid))


def _geometry_key(offsets):
    variants = []
    for matrix in MATRICES:
        cells = [_move(matrix, p) for p in offsets]
        lo = (min(r for r, c in cells), min(c for r, c in cells))
        variants.append(tuple(sorted((r-lo[0], c-lo[1]) for r, c in cells)))
    return min(variants)


def _registered_mask(rng, teaching=False, hole=False):
    for _ in range(240):
        h, w = int(rng.integers(4, 9)), int(rng.integers(4, 9))
        style = _pick(rng, ("grow", "grow", "bend", "teeth", "outline", "stairs", "line"))
        r0, c0 = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        if style == "grow":
            cells = {(r0, c0)}
            for _ in range(int(rng.integers(3, max(4, h*w//2)))):
                r, c = _pick(rng, sorted(cells))
                dr, dc = _pick(rng, DIRECTIONS)
                if 0 <= r+dr < h and 0 <= c+dc < w:
                    cells.add((r+dr, c+dc))
        elif style == "bend":
            cells = {(r0, c) for c in range(c0, w)} | {(r, c0) for r in range(r0, h)}
            cells.add((h-1, min(w-1, c0+1)))
        elif style == "teeth":
            cells = {(r0, c) for c in range(c0, w)}
            cells |= {(min(h-1, r0+int(rng.integers(1, h-r0))), c)
                      for c in range(c0, w, 2)}
        elif style == "outline":
            cells = {(r, w-1) for r in range(r0, h)}
            cells |= {(r0, c) for c in range(c0, w)} | {(h-1, c) for c in range(c0+1, w)}
        elif style == "stairs":
            cells = {(min(h-1, r0+j//2), min(w-1, c0+(j+1)//2))
                     for j in range(int(rng.integers(3, h+w)))}
        else:
            cells = {(r0, c) for c in range(c0, w)}
        if len(cells) < 2:
            continue
        if teaching and len(_geometry_key(cells)) < 4:
            continue
        candidates = []
        for qr in range(h):
            for qc in range(w):
                q = (qr, qc)
                if teaching and ((q not in cells) != hole):
                    continue
                if min(max(abs(r-qr), abs(c-qc)) for r, c in cells) > 1:
                    continue
                for dr, dc in DIRECTIONS:
                    sl, tl = (qr-2*dr, qc-2*dc), (qr-dr, qc-dc)
                    reverse = (qr-3*dr, qc-3*dc)
                    if not all(0 <= r < h and 0 <= c < w for r, c in (sl, tl)):
                        continue
                    if sl in cells or tl in cells:
                        continue
                    if (0 <= reverse[0] < h and 0 <= reverse[1] < w
                            and min(max(abs(r-reverse[0]), abs(c-reverse[1])) for r, c in cells) <= 1):
                        continue
                    offsets = tuple(sorted((r-qr, c-qc) for r, c in cells))
                    if teaching and len({tuple(sorted(_move(m, p) for p in offsets)) for m in MATRICES}) != 8:
                        continue
                    candidates.append((q, sl, tl, offsets))
        if candidates:
            q, sl, tl, offsets = _pick(rng, candidates)
            return (h, w), cells, q, sl, tl, offsets
    raise RuntimeError("bounded registered-mask sampling exhausted")


def _footprint(record, anchor, shape):
    H, W = shape
    return {(anchor[0]+r, anchor[1]+c)
            for r, c in (_move(record["matrix"], p) for p in record["offsets"])
            if 1 <= anchor[0]+r < H-1 and 1 <= anchor[1]+c < W-1}


def _construct(rng, count, teaching, held_geometry=(), mode="mixed"):
    H, W = int(rng.integers(21 if count <= 2 else 25, 31)), int(rng.integers(21 if count <= 2 else 25, 31))
    colors = list(map(int, rng.permutation(np.arange(1, 10))))
    markers, neutral, ink = colors[:2], colors[2], colors[3]
    sources = colors[4:4+count]
    outer_sides = _pick(rng, [(SIDES[i], SIDES[(i+1) % 4]) for i in range(4)])
    outer = dict(zip(outer_sides, markers))
    a = np.zeros((H, W), dtype=np.int8)
    for s, color in outer.items():
        _side_line(a, s)[:] = color
    records, reserved = [], set()
    for index, source in enumerate(sources):
        for _ in range(160):
            shape, cells, q, sl, tl, offsets = _registered_mask(rng, teaching, hole=(index == 0))
            if held_geometry and _geometry_key(offsets) in held_geometry:
                continue
            h, w = shape
            positions = [(r, c) for r in range(3, H-h-2) for c in range(3, W-w-2)]
            rng.shuffle(positions)
            for r, c in positions:
                box = {(i, j) for i in range(r-2, r+h+2) for j in range(c-2, c+w+2)}
                if box & reserved:
                    continue
                inner_sides = _pick(rng, [(SIDES[i], SIDES[(i+1) % 4]) for i in range(4)])
                if teaching and index == 0:
                    matrix = _pick(rng, MATRICES[1:2] + MATRICES[3:4])
                    inverse = tuple(zip(*matrix))
                    guides = {_side_image(inverse, s): color for s, color in outer.items()}
                else:
                    guides = dict(zip(inner_sides, markers if rng.random() < .5 else markers[::-1]))
                    matrix = _frame_matrix(guides, outer)
                bounds = (r, c, h, w)
                for x, y in cells:
                    a[r+x, c+y] = neutral
                a[r+sl[0], c+sl[1]], a[r+tl[0], c+tl[1]] = source, ink
                for side, color in guides.items():
                    for p in _guide_positions(bounds, side):
                        a[p] = color
                record = dict(source=source, ink=ink, neutral=neutral, bounds=bounds,
                              guides=guides, q=(r+q[0], c+q[1]), offsets=offsets,
                              direction=(tl[0]-sl[0], tl[1]-sl[1]), matrix=matrix, anchors=[])
                records.append(record)
                reserved |= box
                break
            else:
                continue
            break
        else:
            raise RuntimeError("stencil packing exhausted")
    # Content is placed outside all instructional rectangles and their moats.
    free = [(r, c) for r in range(1, H-1) for c in range(1, W-1) if (r, c) not in reserved]
    rng.shuffle(free)
    anchors, protected = set(), set()

    def place(i, p, complete=False, protect=False, overlap=None, clipped=False):
        if p not in free_set or p in anchors:
            return False
        rec = records[i]
        fp = _footprint(rec, p, a.shape)
        if not fp or fp & protected:
            return False
        full = len(fp) == len(rec["offsets"])
        if complete and not full or clipped and full:
            return False
        if overlap is not None and not fp & overlap:
            return False
        a[p] = rec["source"]
        anchors.add(p)
        rec["anchors"].append(p)
        if protect:
            protected.update(fp)
        return True

    free_set = set(free)
    if teaching:
        # Isolated complete stamps expose exact shape and registration.
        for i in range(count):
            if not any(place(i, p, complete=True, protect=True) for p in free):
                raise RuntimeError("no isolated complete teaching stamp")
        # An extra complete stamp and a second source intersect on the same canvas.
        cross_anchor = next((p for p in free if place(0, p, complete=True)), None)
        if cross_anchor is None:
            raise RuntimeError("no teaching overlap anchor")
        fp = _footprint(records[0], cross_anchor, a.shape)
        if not any(place(1, p, complete=True, overlap=fp) for p in free):
            raise RuntimeError("no cross-source teaching overlap")
        neighbors = [(cross_anchor[0]+dr, cross_anchor[1]+dc) for dr, dc in DIRECTIONS]
        rng.shuffle(neighbors)
        if not any(place(0, p) for p in neighbors):
            raise RuntimeError("no adjacent source-pixel witness")
        current = dict(grid=a, records=records)
        current_visits, _ = _coverage(current)
        if not np.any((current_visits >= 3) & (current_visits % 2 == 1)):
            double = {(int(r)+1, int(c)+1) for r, c in np.argwhere(current_visits == 2)}
            if not double or not any(place(i, p, complete=True, overlap=double)
                                     for i in range(count) for p in free):
                raise RuntimeError("no odd triple-overlap teaching witness")
        if not any(place(0, p, clipped=True) for p in free):
            raise RuntimeError("no clipped teaching stamp")
    for i in range(count):
        wanted = int(rng.integers(2, 8 if mode == "cluster" else 6))
        if mode == "edge":
            candidates = sorted(free, key=lambda p: min(p[0]-1, p[1]-1, H-2-p[0], W-2-p[1]))
        else:
            candidates = free
        for p in candidates:
            if len(records[i]["anchors"]) >= wanted:
                break
            if records[i]["anchors"] and rng.random() < .5:
                last = records[i]["anchors"][-1]
                near = [(last[0]+dr, last[1]+dc) for dr, dc in DIRECTIONS]
                rng.shuffle(near)
                for q in near:
                    if place(i, q):
                        break
            if len(records[i]["anchors"]) < wanted:
                place(i, p)
        if not records[i]["anchors"]:
            raise RuntimeError("a stencil has no anchor")
    info = dict(grid=a, outer=outer, records=records, neutral=neutral, ink=ink)
    visits, _ = _coverage(info)
    # Preprinted ink includes an erasing witness and untouched ink, with no
    # perturbation of isolated complete demonstration stamps.
    pre_candidates = [p for p in free if p not in anchors and p not in protected]
    erase = [p for p in pre_candidates if visits[p[0]-1, p[1]-1] % 2 == 1]
    untouched = [p for p in pre_candidates if visits[p[0]-1, p[1]-1] == 0]
    if teaching and (not erase or not untouched):
        raise RuntimeError("preprinted-ink teaching witnesses do not fit")
    for pool in (erase, untouched):
        if pool:
            a[_pick(rng, pool)] = ink
    for p in pre_candidates[:int(rng.integers(0, 7))]:
        a[p] = ink
    unused = colors[4+count:]
    if unused and untouched and rng.random() < .6:
        inert_positions = [p for p in untouched if a[p] == 0]
        for p in inert_positions[:int(rng.integers(1, 5))]:
            a[p] = unused[0]
    return {"input": a, "output": _render(info)}, records


def _swap_guides(pair, record):
    a = pair["input"].copy()
    colors = list(record["guides"].values())
    for side, color in record["guides"].items():
        replacement = colors[1] if color == colors[0] else colors[0]
        for p in _guide_positions(record["bounds"], side):
            a[p] = replacement
    return {"input": a, "output": solve(a)}


def _count(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a reproducible puzzle with an exact two-example frame contrast."""
    nt, ne = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    for _ in range(150):
        try:
            pair, records = _construct(rng, 2, True)
            contrast = _swap_guides(pair, records[0])
            candidate = {"train": [pair, contrast], "test": []}
            _teaching_checks([_decode(x["input"]) for x in candidate["train"]],
                             [x["output"] for x in candidate["train"]])
            puzzle["train"] = candidate["train"]
            break
        except (ValueError, AssertionError, RuntimeError):
            continue
    else:
        raise RuntimeError("bounded teaching construction exhausted")
    seen = {_key(x["input"]) for x in puzzle["train"]}
    regime = _pick(rng, ("mixed", "cluster", "edge"))
    for split, number in (("train", nt-2), ("test", ne)):
        for index in range(number):
            held = {_geometry_key(rec["offsets"]) for p in puzzle["train"]
                    for rec in _decode(p["input"])["records"]} if split == "test" else set()
            count = int(rng.integers(1, 4))
            for _ in range(100):
                try:
                    pair, _ = _construct(rng, count, False, held, regime if split == "test" else "mixed")
                    decoded = _decode(pair["input"])
                    if _key(pair["input"]) in seen or not np.array_equal(_render(decoded), pair["output"]):
                        continue
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
                except (ValueError, RuntimeError):
                    continue
            else:
                raise RuntimeError("bounded ordinary pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _teaching_checks(infos, outputs):
    _require(any(not np.array_equal(_render(i, pixels_only=True), o) for i, o in zip(infos, outputs)),
             "training must distinguish stencils from anchor recoloring")
    _require(any(not np.array_equal(_render(i, component_only=True), o) for i, o in zip(infos, outputs)),
             "training must distinguish source pixels from source components")
    _require(any(not np.array_equal(_render(i, union=True), o) for i, o in zip(infos, outputs)),
             "training must distinguish toggling from overpainting")
    _require(any(not np.array_equal(_render(i, native=True), o) for i, o in zip(infos, outputs)),
             "training must use marker orientation")
    _require(any((0, 0) not in r["offsets"] for i in infos for r in i["records"]),
             "training must show an empty registration point")
    _require(any(_coverage(i)[1] for i in infos), "training must show boundary clipping")
    _require(any(np.any((_base(i) == i["ink"]) & (_coverage(i)[0] % 2 == 1)) for i in infos),
             "training must show erasure of existing ink")
    _require(any(np.any((_base(i) == i["ink"]) & (_coverage(i)[0] == 0)) for i in infos),
             "training must preserve untouched ink")
    _require(any(np.any((_coverage(i)[0] >= 2) & (_coverage(i)[0] % 2 == 0)
                       & (_base(i) == 0)) for i in infos), "training must show actual cancellation")
    _require(any(np.any((_coverage(i)[0] >= 3) & (_coverage(i)[0] % 2 == 1)
                       & (_base(i) == 0)) for i in infos), "training must show odd higher-order overlap")
    _require(any(all(any(len(_footprint(r, a, i["grid"].shape)) == len(r["offsets"])
                            and all(_coverage(i)[0][p[0]-1, p[1]-1] == 1
                                    and _base(i)[p[0]-1, p[1]-1] == 0
                                    for p in _footprint(r, a, i["grid"].shape))
                            for a in r["anchors"]) for r in i["records"])
                 for i in infos), "training must show each stencil as an isolated complete stamp")
    cross = False
    for info in infos:
        per = [_coverage(dict(info, records=[r]))[0] for r in info["records"]]
        cross |= bool(np.any(np.sum([p > 0 for p in per], axis=0) >= 2))
    _require(cross, "training must show interaction between different stencils")
    contrast = False
    for j, first in enumerate(infos):
        for k in range(j+1, len(infos)):
            second = infos[k]
            if first["grid"].shape != second["grid"].shape or first["outer"] != second["outer"]:
                continue
            if [(r["bounds"], r["q"], r["offsets"], r["anchors"]) for r in first["records"]] != [
                    (r["bounds"], r["q"], r["offsets"], r["anchors"]) for r in second["records"]]:
                continue
            different = first["grid"] != second["grid"]
            differences = set(map(tuple, np.argwhere(different)))
            if (differences and differences <= first["guide_cells"]
                    and not np.array_equal(outputs[j], outputs[k])):
                contrast = True
    _require(contrast, "training must have a guide-identity-only contrast with different answers")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal example counts")
    seen, training_infos, training_outputs = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _decode(a)
            _require(b.shape == (a.shape[0]-2, a.shape[1]-2), "output must be the entire work interior")
            _require(np.array_equal(_render(info), b), "incorrect registered interference result")
            _require(not set(map(int, np.unique(b))) & {info["neutral"], *info["outer"].values(),
                                                       *(r["source"] for r in info["records"])},
                     "instruction or source colors survive")
            base = _base(info)
            visits, _ = _coverage(info)
            _require(np.array_equal(b[visits == 0], base[visits == 0]), "untouched content changed")
            _require(np.array_equal(b[visits % 2 == 0], base[visits % 2 == 0]), "even visits failed to cancel")
            if split == "train":
                training_infos.append(info)
                training_outputs.append(b)
    _teaching_checks(training_infos, training_outputs)


def _manual_case():
    a = np.zeros((14, 15), dtype=np.int8)
    a[0, 1:-1], a[1:-1, 0] = 3, 2
    a[2, 3:8], a[3:8, 2] = 2, 3
    a[5, 3], a[5, 4] = 1, 4
    for p in ((5, 6), (4, 6), (6, 6), (6, 7)):
        a[p] = 8
    for p in ((10, 4), (10, 5), (11, 12)):
        a[p] = 1
    a[11, 3], a[9, 10], a[9, 2] = 4, 4, 6
    expected = np.zeros((12, 13), dtype=np.int8)
    for p in ((10, 5), (11, 4), (11, 5), (11, 10), (11, 11), (11, 12), (8, 9)):
        expected[p] = 4
    expected[8, 1] = 6
    return a, expected


def self_test():
    """Bounded semantic witnesses, metamorphisms, counts, and 42-seed audit."""
    a, expected = _manual_case()
    _require(np.array_equal(solve(a), expected), "manual hole/transpose/XOR/clipping regression")
    for turn in range(4):
        for flip in (False, True):
            x, y = np.rot90(a, turn), np.rot90(expected, turn)
            if flip:
                x, y = np.fliplr(x), np.fliplr(y)
            _require(np.array_equal(solve(x), y), "coordinate equivariance failed")
    palette = np.array([0, 9, 7, 6, 5, 4, 3, 2, 1, 8], dtype=np.int8)
    _require(np.array_equal(solve(palette[a]), palette[expected]), "local color permutation failed")
    shapes, geometries, frames, counts, masks, palettes, clipped, overlaps = set(), set(), set(), set(), set(), set(), 0, 0
    signatures = set()
    for seed in range(42):
        nt = 2 + seed % 9
        ne = 1 + (seed // 9) % min(4, nt-1)
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        if seed < 3:
            again = generate_puzzle(seed, nt, ne)
            _require(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                         for k in ("input", "output")), "nondeterministic seed")
        train_geometry = {_geometry_key(r["offsets"]) for p in puzzle["train"]
                          for r in _decode(p["input"])["records"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solve failed")
                shapes.add(pair["input"].shape)
                counts.add(len(info["records"]))
                palettes.add((info["neutral"], info["ink"], tuple(info["outer"].values())))
                local_geometries = set()
                for record in info["records"]:
                    geometries.add(_geometry_key(record["offsets"]))
                    local_geometries.add(_geometry_key(record["offsets"]))
                    frames.add(record["matrix"])
                    masks.add(len(record["offsets"]))
                if split == "test":
                    _require(local_geometries.isdisjoint(train_geometry), "test reused training mask geometry")
                visits, clip = _coverage(info)
                clipped += clip > 0
                overlaps += bool(np.any(visits >= 2))
    _require(len(signatures) == 42, "42 seeds did not give distinct puzzles")
    _require(len(shapes) >= 45 and len(geometries) >= 65 and len(palettes) >= 60, "insufficient structural diversity")
    _require(frames == set(MATRICES) and counts == {1, 2, 3} and len(masks) >= 12, "missing family variations")
    _require(clipped >= 50 and overlaps >= 50, "missing clipping/overlap variation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000 + 10*nt + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "explicit count ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0)
    bad = copy.deepcopy(puzzle)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("wrong output accepted")
    bad = copy.deepcopy(puzzle)
    info = _decode(bad["test"][0]["input"])
    rec = info["records"][0]
    side = next(iter(rec["guides"]))
    positions = _guide_positions(rec["bounds"], side)
    bad["test"][0]["input"][positions[len(positions)//2]] = 0
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("broken guide accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "mask_geometries": len(geometries),
            "role_palettes": len(palettes), "symmetries": len(frames),
            "mask_sizes": sorted(masks), "clipped_pairs": clipped, "overlapping_pairs": overlaps}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
