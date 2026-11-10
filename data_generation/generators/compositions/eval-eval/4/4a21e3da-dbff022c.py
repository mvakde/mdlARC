"""Seam-cut casts: marker geometry changes the topology of keyed molds.

Only the returned input supplies the solver's state. Construction checks safe
transport and has a central-band fallback; solve() assumes no source layout.
"""
from collections import defaultdict
import copy
import json

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "right", "bottom", "left")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("a grid must be an integer 1..30 by 1..30 array in 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask)))
    steps = STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _regions(grid, bg, diagonal=False):
    h, w = grid.shape
    regions = []
    for cells in _components(grid == bg, diagonal):
        boundary = {q for r, c in cells for dr, dc in STEPS
                    if 0 <= (q := (r + dr, c + dc))[0] < h
                    and 0 <= q[1] < w and grid[q] != bg}
        edge = any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        regions.append((cells, boundary, edge))
    return regions


def _read_input(value):
    grid = _grid(value)
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    if sum(counts == counts.max()) != 1:
        raise ValueError("background must be the unique most frequent input color")
    bg = int(colors[np.argmax(counts)])
    ink_components = _components(grid != bg, True)
    candidates = []
    for cells in ink_components:
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != sh * sw or len({int(grid[p]) for p in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and 2 <= sw <= 7 and 0 < c0 <= c1 < w - 1:
            if r0 == 0:
                lanes.append(("top", grid[0, c0:c1+1], grid[1, c0:c1+1]))
            if r1 == h - 1:
                lanes.append(("bottom", grid[-1, c0:c1+1], grid[-2, c0:c1+1]))
        if sw == 2 and 2 <= sh <= 7 and 0 < r0 <= r1 < h - 1:
            if c0 == 0:
                lanes.append(("left", grid[r0:r1+1, 0], grid[r0:r1+1, 1]))
            if c1 == w - 1:
                lanes.append(("right", grid[r0:r1+1, -1], grid[r0:r1+1, -2]))
        for side, outer, inner in lanes:
            sources, fills = list(map(int, outer)), list(map(int, inner))
            if len(set(sources)) == len(sources):
                candidates.append((cells, side, dict(zip(sources, fills))))
    if len(candidates) != 1:
        raise ValueError("input must contain one unambiguous isolated edge key")
    key, key_side, mapping = candidates[0]
    border = {(r, c) for r in range(h) for c in range(w)
              if r in (0, h-1) or c in (0, w-1)}
    marker_cells = {p for p in border - key if grid[p] != bg}
    if len(marker_cells) not in (1, 2):
        raise ValueError("expected one or two border markers")
    marker_values = {int(grid[p]) for p in marker_cells}
    if len(marker_values) != 1:
        raise ValueError("all guide markers must share a color")
    marker = marker_values.pop()
    if marker in set(mapping) | set(mapping.values()) or set(map(tuple, np.argwhere(grid == marker))) != marker_cells:
        raise ValueError("guide color must be exclusive to markers")
    markers = {}
    for r, c in sorted(marker_cells):
        sides = [s for s, yes in zip(SIDES, (r == 0, c == w-1, r == h-1, c == 0)) if yes]
        if len(sides) != 1 or sides[0] in markers:
            raise ValueError("markers cannot be at corners or repeat a side")
        if next(cells for cells in ink_components if (r, c) in cells) != {(r, c)}:
            raise ValueError("markers must be isolated")
        markers[sides[0]] = (r, c)
    vertical = set(markers) & {"top", "bottom"}
    horizontal = set(markers) & {"left", "right"}
    if len(markers) == 2 and (len(vertical) != 1 or len(horizontal) != 1):
        raise ValueError("two markers must lie on adjacent sides")
    ac = markers[next(iter(vertical))][1] if vertical else None
    ar = markers[next(iter(horizontal))][0] if horizontal else None
    objects = [cells for cells in ink_components if cells != key and not cells & marker_cells]
    contours = set().union(*objects) if objects else set()
    if not contours:
        raise ValueError("no molds")
    for cells in objects:
        if cells & border or len({int(grid[p]) for p in cells}) != 1:
            raise ValueError("molds must be separated monochrome interior objects")
    paths = {}
    for side, (r, c) in markers.items():
        dr, dc = {"top": (1, 0), "right": (0, -1), "bottom": (-1, 0), "left": (0, 1)}[side]
        ray = []
        while 0 <= r < h and 0 <= c < w:
            ray.append((r, c))
            r, c = r + dr, c + dc
        hits = [i for i, p in enumerate(ray) if p in contours]
        if not hits:
            raise ValueError("each guide must intersect mold ink")
        path = ray[:max(hits) + 1]
        if set(path) & key:
            raise ValueError("key cannot obstruct a guide")
        paths[side] = path
    return dict(grid=grid, bg=bg, key=key, key_side=key_side, mapping=mapping,
                marker=marker, markers=markers, ar=ar, ac=ac,
                contours=contours, objects=objects, paths=paths)


def _corner(p, info):
    r, c = p
    ar, ac = info["ar"], info["ac"]
    if r == ar or c == ac:
        return None
    v = ("top" if r < ar else "bottom") if ar is not None else next(iter(info["markers"]))
    s = ("left" if c < ac else "right") if ac is not None else next(iter(info["markers"]))
    return (v, s)


def _derive(info, *, original_topology=False, recursive=False, individual=False,
            nearest=False, strict=True):
    grid, bg, marker = info["grid"], info["bg"], info["marker"]
    sealed = grid.copy()
    for path in info["paths"].values():
        if nearest:
            path = path[:next(i for i, p in enumerate(path) if p in info["contours"]) + 1]
        for p in path:
            if grid[p] == bg:
                sealed[p] = marker
    topology = grid if original_topology else sealed
    pockets = []
    all_void = set()
    for cells, boundary, edge in _regions(topology, bg):
        if edge:
            continue
        if boundary & info["key"]:
            raise ValueError("key cannot bound a cavity")
        owners = {int(grid[p]) for p in boundary if int(topology[p]) != marker}
        if len(owners) != 1:
            if strict:
                raise ValueError("every sealed cavity must have one mold color")
            continue
        color = owners.pop()
        all_void |= cells
        pockets.append((cells, color))
    groups = defaultdict(dict)
    dropped = 0
    for cells, color in pockets:
        if color not in info["mapping"]:
            continue
        fill = info["mapping"][color]
        if recursive:
            visited = {color}
            while fill in info["mapping"] and fill not in visited:
                visited.add(fill)
                fill = info["mapping"][fill]
        for p in cells:
            corner = _corner(p, info)
            if corner is None:
                continue
            if set(corner) & set(info["markers"]):
                groups[corner][p] = fill
            else:
                dropped += 1
    out = sealed.copy()
    casts = {}
    shifts = {}
    for corner, colored_cells in sorted(groups.items()):
        chunks = (_components(_mask(grid.shape, colored_cells)) if individual else [set(colored_cells)])
        for cells in chunks:
            rr, cc = zip(*cells)
            dr = -min(rr) if corner[0] == "top" else grid.shape[0] - 1 - max(rr)
            dc = -min(cc) if corner[1] == "left" else grid.shape[1] - 1 - max(cc)
            shifts[corner] = (dr, dc)
            for p in sorted(cells):
                q = (p[0] + dr, p[1] + dc)
                if strict and (out[q] != bg or q in all_void):
                    if q in info["key"]:
                        raise ValueError("key collides with a cast")
                    raise ValueError("transported casts must not collide or refill source cavities")
                out[q] = colored_cells[p]
                casts[q] = colored_cells[p]
    return out, dict(sealed=sealed, pockets=pockets, groups=dict(groups),
                     casts=casts, shifts=shifts, dropped=dropped)


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete seams, extract keyed negative space, and transport it by sector."""
    return _derive(_read_input(input_grid))[0]


def _features(info, state):
    grid, bg, mapping = info["grid"], info["bg"], info["mapping"]
    original_regions = _regions(grid, bg)
    sealed_regions = _regions(state["sealed"], bg)
    original_closed = set().union(*(cells for cells, _, edge in original_regions if not edge))
    eight_closed = set().union(*(cells for cells, _, edge in _regions(state["sealed"], bg, True) if not edge))
    retained = set().union(*(set(cells) for cells in state["groups"].values()))
    active = [(cells, color) for cells, color in state["pockets"] if color in mapping]
    irregular = any(len(cells) != (max(r for r,c in cells)-min(r for r,c in cells)+1)
                    * (max(c for r,c in cells)-min(c for r,c in cells)+1) for cells, _ in active)
    original_objects_closed = {int(grid[next(iter(boundary))]) for _, boundary, edge in original_regions
                               if not edge and len({int(grid[p]) for p in boundary}) == 1}
    open_active = 0
    for obj in info["objects"]:
        color = int(grid[next(iter(obj))])
        has_pocket = any(bool(boundary & obj) for _, boundary, edge in sealed_regions if not edge)
        open_active += color in mapping and not has_pocket
    return dict(shape=list(grid.shape), key_side=info["key_side"], key_length=len(mapping),
                background=bg, marker=info["marker"], sides=sorted(info["markers"]),
                arity=len(info["markers"]), object_count=len(info["objects"]),
                contour_sizes=sorted(map(len, info["objects"])),
                cavity_count=len(state["pockets"]), cast_cells=len(retained), dropped=state["dropped"],
                seam_created=any(cells - original_closed for cells, color in active
                                 if cells & retained),
                diagonal=bool(retained - eight_closed), irregular=irregular,
                collective=any(len(_components(_mask(grid.shape, cells))) > 1
                               for cells in state["groups"].values()),
                multicolor_group=any(len(set(cells.values())) > 1 for cells in state["groups"].values()),
                chain=any(mapping[color] in mapping and mapping[mapping[color]] != mapping[color]
                          and bool(cells & retained) for cells, color in active),
                self_map=any(mapping[color] == color and bool(cells & retained) for cells, color in active),
                inert_closed=any(color not in mapping for _, color in state["pockets"]),
                open_active=int(open_active),
                internal_gap=any(any(grid[p] == bg for p in path[min(hits):max(hits)+1])
                                 for path in info["paths"].values()
                                 if (hits := [i for i,p in enumerate(path) if p in info["contours"]])),
                original_closed_colors=sorted(original_objects_closed))


def _ring(rng, kind, maxh, maxw, core=False, double=False, small=False):
    """Build actual mold ink, including diagonal contours and irregular voids."""
    if kind == "diamond":
        radius = 3 if double else int(rng.integers(2, 4))
        if 2 * radius + 1 > min(maxh, maxw):
            kind = "profile"
        else:
            rr, cc = np.indices((2*radius+1, 2*radius+1))
            wall = abs(rr-radius) + abs(cc-radius) == radius
            return wall, radius, radius
    h = (7 if double else 5) if core and small else int(rng.integers(7 if core and double else (5 if core else 3), maxh + 1))
    w = (7 if double else 7) if core and small else int(rng.integers(7 if double else (5 if core else 3), maxw + 1))
    wall = np.ones((h, w), dtype=bool)
    wall[1:-1, 1:-1] = False
    ar, ac = h // 2, w // 2
    if kind == "profile":
        # Notches and unequal walls change the cavity, not just its silhouette.
        wall[1, 1] = True
        if h >= 7 and not small and rng.random() < .7:
            wall[2, 1] = True
        if w >= 9 and rng.random() < .6:
            wall[-2, -2:-4:-1] = True
        wall[0, 0] = False
        wall[-1, 0] = False
        wall[0, -1] = False
    elif kind == "multi":
        # A transverse divider leaves at least one cavity on each side.
        wall[ar, :] = True
        if h >= 8 and rng.random() < .5:
            wall[ar-2, :] = True
    return wall, ar, ac


def _satellite(rng, kind, small):
    if small:
        wall = np.ones((3, 3), dtype=bool)
        wall[1, 1] = False
        if kind == "open":
            wall[0, 1] = False
        return wall
    if kind == "diamond":
        rr, cc = np.indices((5, 5))
        return abs(rr-2) + abs(cc-2) == 2
    wall, _, ac = _ring(rng, "profile" if kind == "profile" else "rectangle", 6, 6)
    if kind == "open":
        wall[0, ac] = False
    return np.rot90(wall, int(rng.integers(4))).copy()


def _key_coords(shape, side, start, n):
    h, w = shape
    if side == "top":
        return [((0, start+i), (1, start+i)) for i in range(n)]
    if side == "bottom":
        return [((h-1, start+i), (h-2, start+i)) for i in range(n)]
    if side == "left":
        return [((start+i, 0), (start+i, 1)) for i in range(n)]
    return [((start+i, w-1), (start+i, w-2)) for i in range(n)]


def _pack_scene(rng, shape, double, lesson, object_count, small=False):
    h, w = shape
    margin = 2 if not small and rng.random() < .55 else h // 3
    band = h - 2 * margin
    if lesson == 0:
        kind = "multi"
    elif lesson == 1:
        kind = "profile"
    else:
        kind = str(rng.choice(["rectangle", "profile", "diamond", "multi"]))
    core, local_ar, local_ac = _ring(rng, kind, min(band, 9), min(w-10, 11), True, double, small)
    ch, cw = core.shape
    r0 = int(rng.integers(margin, h-margin-ch+1))
    c0 = int(rng.integers(3, w-cw-2))
    if small:
        r0 = margin + (band-ch)//2
        c0 = (w-cw)//2
    ar, ac = r0+local_ar, c0+local_ac
    # Opening on a seam is the central interaction: the seam repairs the mold.
    opened = lesson in (0, 1) or rng.random() < .6
    if opened and double and rng.random() < .35:
        core[local_ar, local_ac+1:] = False
    elif opened:
        core[:local_ar, local_ac] = False
    grid = np.zeros(shape, dtype=np.int8)
    grid[r0:r0+ch, c0:c0+cw][core] = 1
    boxes = [(r0, c0, ch, cw)]
    specs = []
    if lesson == 0:
        specs = [("inert", "rectangle")]
    elif lesson == 1:
        specs = [("mapped", "rectangle"), ("mapped", "open")]
    while len(specs) < object_count - 1:
        specs.append((str(rng.choice(["mapped", "mapped", "inert"])),
                      str(rng.choice(["rectangle", "profile", "diamond", "open"]))))
    placed = []
    for index, (role, kind) in enumerate(specs):
        wall = _satellite(rng, kind, small)
        mh, mw = wall.shape
        positions = []
        for r in range(margin, h-margin-mh+1):
            for c in range(2, w-mw-1):
                if any(not (r+mh < br or br+bh < r or c+mw < bc or bc+bw < c)
                       for br, bc, bh, bw in boxes):
                    continue
                if kind == "open" and (c <= ac < c+mw or (double and r <= ar < r+mh)):
                    continue
                positions.append((r, c))
        if not positions:
            return None
        if small:
            # Alternate side lanes; two three-cell molds can share a seven-row band.
            preferences = [(margin, 2), (h-margin-mh, w-mw-2),
                           (h-margin-mh, 2), (margin, w-mw-2)]
            r, c = next((p for p in preferences if p in positions), positions[0])
        else:
            r, c = positions[int(rng.integers(len(positions)))]
        if small and kind == "open" and double and r > ar:
            wall = np.rot90(wall, 2).copy()
        boxes.append((r, c, mh, mw))
        placed.append((role, kind, wall, r, c))
    return grid, ar, ac, placed


def _make_pair(rng, shape, double, lesson, object_count, turn, allowed_key_sides=None):
    for attempt in range(40):
        small = attempt >= 28
        packed = _pack_scene(rng, shape, double, lesson, object_count, small)
        if packed is None:
            continue
        grid, ar, ac, placed = packed
        k = int(rng.integers(2, 6))
        sources = list(range(1, k+1))
        fills = list(map(int, rng.choice(np.arange(1, 9), k)))
        if lesson == 0:
            fills[0], fills[1] = 2, k+1
        elif lesson == 1:
            fills[0], fills[1] = 1, k+1
        else:
            mode = int(rng.integers(4))
            if mode == 0:
                fills = sources[1:] + [k+1]
            elif mode == 1:
                fills = [k+1] * k
            elif mode == 2:
                fills[0] = 1
        mapping = dict(zip(sources, fills))
        for index, (role, kind, wall, r, c) in enumerate(placed):
            if role == "inert":
                color = int(rng.integers(k+1, 9))
            elif lesson == 1:
                color = 2
            else:
                color = int(rng.choice(sources))
            grid[r:r+wall.shape[0], c:c+wall.shape[1]][wall] = color
        grid[0, ac] = 9
        if double:
            grid[ar, -1] = 9
        order = list(map(int, rng.permutation(k)))
        entries = [(sources[i], mapping[sources[i]]) for i in order]
        choices = [(side, start) for side in SIDES
                   for start in range(2, (shape[1] if side in ("top", "bottom") else shape[0])-k-1)
                   if allowed_key_sides is None or SIDES[(SIDES.index(side)-turn) % 4] in allowed_key_sides]
        rng.shuffle(choices)
        for side, start in choices:
            cells = _key_coords(shape, side, start, k)
            occupied = {p for pair in cells for p in pair}
            if any(grid[q] != 0 for r,c in occupied for dr in (-1,0,1) for dc in (-1,0,1)
                   if 0 <= (q := (r+dr,c+dc))[0] < shape[0] and 0 <= q[1] < shape[1]):
                continue
            candidate = grid.copy()
            for (outer, inner), (source, fill) in zip(cells, entries):
                candidate[outer], candidate[inner] = source, fill
            try:
                info = _read_input(candidate)
                output, state = _derive(info)
                # Keep deposited shapes visually separate from the apparatus.
                # This is a sampling preference, not a layout assumption in solve.
                halo_ink = {q for r,c in state["casts"] for dr in (-1,0,1) for dc in (-1,0,1)
                            if 0 <= (q := (r+dr,c+dc))[0] < shape[0] and 0 <= q[1] < shape[1]
                            and state["sealed"][q] != 0}
                if halo_ink - info["key"]:
                    break
                if halo_ink:
                    continue
                f = _features(info, state)
                if not f["cast_cells"]:
                    break
                if lesson == 0 and not (f["seam_created"] and f["collective"] and f["chain"]
                                       and f["inert_closed"] and f["internal_gap"]):
                    break
                if lesson == 1 and not (f["seam_created"] and f["dropped"] and f["self_map"]
                                       and f["open_active"] and f["irregular"] and f["diagonal"]
                                       and f["multicolor_group"]):
                    break
            except ValueError as error:
                if str(error) == "key collides with a cast":
                    continue
                break
            # Palette is local, fully input-visible, including black in other roles.
            palette = np.asarray(rng.permutation(10), dtype=np.int8)
            candidate, output = palette[candidate], palette[output]
            candidate, output = [np.rot90(a, turn).copy() for a in (candidate, output)]
            return {"input": candidate, "output": output}
    raise RuntimeError("bounded mold packing failed")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(4))  # grow, compressed, rotated, combined multiplicity
    base_turn = int(rng.integers(4))
    train_turns = [base_turn, (base_turn+1) % 4]
    train_turns += [int(rng.choice(train_turns[:2])) if shift == 2 else int(rng.integers(4))
                    for _ in range(ntrain-2)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", ntrain), ("test", ntest)):
        for index in range(n):
            train = split == "train"
            lesson = index if train and index < 2 else -1
            double = lesson == 1 or (lesson < 0 and bool(rng.integers(2)))
            if shift == 0:
                low, high = (17, 24) if train else (25, 30)
            elif shift == 1:
                low, high = (25, 30) if train else (17, 22)
            else:
                low, high = 17, 30
            h, w = int(rng.integers(low, high+1)), int(rng.integers(max(19, low), max(19, high)+1))
            # A seven-row source band supports the smallest diagonal/cut witness.
            if h == 18:
                h = 19
            object_count = 2 if lesson == 0 else 3 if lesson == 1 else int(rng.integers(1, 6))
            if shift == 3:
                object_count = (2 if lesson == 0 else 3) if train else int(rng.integers(4, 6))
            turn = train_turns[index] if train else ((base_turn+2+index%2) % 4 if shift == 2 else int(rng.integers(4)))
            allowed_key_sides = None
            if train and index == 1:
                previous_side = _read_input(puzzle["train"][0]["input"])["key_side"]
                allowed_key_sides = ({"left", "right"} if previous_side in ("top", "bottom")
                                     else {"top", "bottom"})
            for _ in range(12):
                pair = _make_pair(rng, (h, w), double, lesson, object_count, turn, allowed_key_sides)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(grid):
    a = _grid(grid)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = _fingerprint(inp)
            assert signature not in seen
            seen.add(signature)
            info = _read_input(inp)
            expected, state = _derive(info)
            assert np.array_equal(expected, out), "incorrect casts or seams"
            assert np.array_equal(inp[inp != info["bg"]], out[inp != info["bg"]]), "input ink changed"
            # Exact transport and one-step mapped colors, checked cell by cell.
            for corner, cells in state["groups"].items():
                dr, dc = state["shifts"][corner]
                for (r, c), fill in cells.items():
                    assert out[r+dr, c+dc] == fill
                    assert state["sealed"][r, c] == info["bg"]
            added = set(map(tuple, np.argwhere((out != inp) & (out != info["marker"]))))
            assert added == set(state["casts"]), "cast area not conserved"
            assert added, "empty transformation"
            features[split].append(_features(info, state))
    train, test = features["train"], features["test"]
    assert {f["arity"] for f in test} <= {f["arity"] for f in train}
    # This family always teaches the interaction, rather than only separate pieces.
    assert any(f["seam_created"] and f["chain"] and f["collective"] for f in train)
    assert any(f["arity"] == 2 and f["dropped"] and f["seam_created"] for f in train)
    assert any(f["inert_closed"] for f in train)
    assert any(f["open_active"] for f in train)
    assert any(f["internal_gap"] for f in train)
    assert any(f["multicolor_group"] for f in train)
    assert ({f["key_side"] for f in train} & {"top", "bottom"}
            and {f["key_side"] for f in train} & {"left", "right"})
    for feature in ("self_map", "irregular", "diagonal"):
        assert not any(f[feature] for f in test) or any(f[feature] for f in train), "missing " + feature + " witness"
    return None


def self_test():
    """Bounded API, semantic-ablation, symmetry, mutation and 42-seed checks."""
    # New hand-derived fixtures, unrelated to either parent's stored grids.
    fixture = np.zeros((15, 19), dtype=np.int8)
    fixture[0, 9] = 9
    fixture[5:10, 6:13] = 1
    fixture[6:9, 7:12] = 0
    fixture[7, 6:13] = 1
    fixture[5, 9] = 0
    ring = np.array([[1,1,1], [1,0,1], [1,1,1]], dtype=np.int8)
    fixture[5:8, 2:5] = ring * 4
    fixture[5:8, 14:17] = ring * 2
    fixture[5, 15] = 0
    fixture[14, 8:10] = [1, 2]
    fixture[13, 8:10] = [2, 5]
    expected = fixture.copy()
    for r in range(10):
        if expected[r, 9] == 0:
            expected[r, 9] = 9
    for r in (0, 2):
        for c in (0, 1, 17, 18):
            expected[r, c] = 2
    assert np.array_equal(solve(fixture), expected)
    two = fixture.copy()
    two[7, 18] = 9
    expected_two = fixture.copy()
    for r in range(10):
        if expected_two[r, 9] == 0:
            expected_two[r, 9] = 9
    for c in range(2, 19):
        if expected_two[7, c] == 0:
            expected_two[7, c] = 9
    for r, columns in ((0, (0, 1, 17, 18)), (14, (17, 18))):
        for c in columns:
            expected_two[r, c] = 2
    assert np.array_equal(solve(two), expected_two)
    for a, b in ((fixture, expected), (two, expected_two)):
        assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
        assert np.array_equal(solve(a.tolist()), b)
    assert expected[6, 3] == expected[6, 15] == 0  # inert and still-open molds
    activated = fixture.copy()
    activated[14, 9] = 4  # listing this mold changes the joint stencil's extent
    activated_result = solve(activated)
    assert activated_result[0, 0] == 5 and activated_result[0, 4] == 2
    assert activated_result[2, 5] == 2 and activated_result[2, 0] == 0
    signatures, shapes, structures = set(), set(), set()
    sides, keys, backgrounds, counts, cavity_counts = set(), set(), set(), set(), set()
    witnesses = defaultdict(int)
    records = []
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        encode = lambda task: json.dumps({s: [{k: np.asarray(v).tolist() for k,v in pair.items()}
                                             for pair in task[s]] for s in ("train", "test")}, separators=(",", ":"))
        assert encode(p) == encode(q)
        signatures.add(encode(p))
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                info = _read_input(inp)
                expected, state = _derive(info)
                f = _features(info, state)
                records.append(f)
                shapes.add(inp.shape)
                sides.add(tuple(f["sides"]))
                keys.add(f["key_side"])
                backgrounds.add(f["background"])
                counts.add(f["object_count"])
                cavity_counts.add(f["cavity_count"])
                roles, normalized = {}, []
                for c in inp.flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                structures.add((inp.shape, tuple(normalized)))
                assert np.array_equal(expected, out)
                for name in ("seam_created", "collective", "chain", "self_map", "inert_closed", "open_active", "irregular", "diagonal", "multicolor_group"):
                    witnesses[name] += bool(f[name])
                if split == "train":
                    for name, args in (("no_seam_topology", {"original_topology": True}),
                                       ("recursive_lookup", {"recursive": True}),
                                       ("separate_pockets", {"individual": True}),
                                       ("nearest_hit", {"nearest": True})):
                        mutant = _derive(info, strict=False, **args)[0]
                        witnesses[name] += not np.array_equal(mutant, out)
        # Symmetries exercise solve beyond the construction's particular palette.
        pair = p["test"][0]
        for turn in range(4):
            inp, out = [np.rot90(a, turn) for a in pair.values()]
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(solve((inp+3) % 10), (out+3) % 10)
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 50 and max(max(s) for s in shapes) == 30
    assert len(sides) == 8 and keys == set(SIDES) and backgrounds == set(range(10))
    assert counts == set(range(1, 6)) and len(cavity_counts) >= 7
    assert all(witnesses[name] > 0 for name in witnesses)
    assert min(witnesses[name] for name in ("no_seam_topology", "recursive_lookup", "separate_pockets", "nearest_hit")) >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(100+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1,1), (11,1), (2,2), (4,0), (6,5), (True,1), (4,1.5), (None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = copy.deepcopy(generate_puzzle(0))
    info = _read_input(bad["train"][0]["input"])
    p = min(info["key"])
    bad["train"][0]["output"][p] = info["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("key mutation accepted")
    return dict(seeds=42, pairs=len(records), shapes=len(shapes), structures=len(structures),
                witnesses=dict(witnesses))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
