"""Rotational consensus: infer a pooling operation from an in-input example.

Only standard-library modules and NumPy are used. No parent code, examples,
files, random state, or generation metadata are consulted by solve().
"""
from collections import defaultdict
from itertools import combinations

import numpy as np


_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))
_D4 = ((0, 1), (0, -1), (1, 0), (-1, 0))
_MODES = ("union", "intersection")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _trim(a):
    rows, cols = np.nonzero(a)
    if not len(rows):
        raise ValueError("empty silhouette")
    return a[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()


def _mask_key(a):
    return a.shape[0], a.shape[1], tuple(map(int, a.flat))


def _alignment(mask, reflections=False):
    choices = []
    for flip in range(2 if reflections else 1):
        base = np.fliplr(mask) if flip else mask
        for turn in range(4):
            rotated = np.rot90(base, turn)
            choices.append((_mask_key(rotated), flip, turn))
    return min(choices)


def _canonical(mask):
    return _alignment(mask)[0]


def _connected(mask, directions=_D8):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    visited, todo = {min(cells)}, [min(cells)]
    while todo:
        r, c = todo.pop()
        for dr, dc in directions:
            p = r+dr, c+dc
            if p in cells and p not in visited:
                visited.add(p)
                todo.append(p)
    return visited == cells


def _objects(panel, bg, accent):
    """Segment occupancy, rather than splitting an object at its internal colors."""
    cells = set(map(tuple, np.argwhere(panel != bg)))
    objects = []
    while cells:
        start = min(cells)
        cells.remove(start)
        todo, found = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in _D8:
                p = r+dr, c+dc
                if p in cells:
                    cells.remove(p)
                    todo.append(p)
                    found.append(p)
        top = min(p[0] for p in found)
        left = min(p[1] for p in found)
        shape = max(p[0] for p in found)-top+1, max(p[1] for p in found)-left+1
        mask = np.zeros(shape, dtype=bool)
        marks = np.zeros(shape, dtype=bool)
        for r, c in found:
            mask[r-top, c-left] = True
            marks[r-top, c-left] = panel[r, c] == accent
        body = {int(panel[p]) for p in found} - {accent}
        if len(body) != 1 or not marks.any():
            raise ValueError("each before object needs one body role and visible marks")
        if len({_mask_key(np.rot90(mask, k)) for k in range(4)}) != 4:
            raise ValueError("rotationally symmetric silhouettes have ambiguous mark alignment")
        objects.append(dict(top=top, left=left, mask=mask, marks=marks,
                            body=next(iter(body)), key=_canonical(mask)))
    if not objects:
        raise ValueError("empty before panel")
    return objects


def _consensus(panel, bg, accent, mode, matching="rotations"):
    """Pixel-reading implementation, also used for documented counterfactuals."""
    if mode not in _MODES:
        raise ValueError("unknown consensus operation")
    objects = _objects(panel, bg, accent)
    groups = defaultdict(list)
    for i, obj in enumerate(objects):
        if matching == "identity":
            key, flip, turn = _mask_key(obj["mask"]), 0, 0
        else:
            key, flip, turn = _alignment(obj["mask"], matching == "reflections")
        if matching == "color":
            key = (key, obj["body"])
        elif matching == "singletons":
            key = (key, i)
        marks = np.fliplr(obj["marks"]) if flip else obj["marks"]
        obj["flip"], obj["turn"] = flip, turn
        obj["aligned_marks"] = np.rot90(marks, turn)
        groups[key].append(obj)
    output = np.full_like(panel, bg)
    for group in groups.values():
        pooled = (np.logical_or.reduce([o["aligned_marks"] for o in group])
                  if mode == "union" else
                  np.logical_and.reduce([o["aligned_marks"] for o in group]))
        for obj in group:
            aligned = (obj["aligned_marks"] if matching == "exactly_two" and len(group) != 2
                       else pooled)
            marks = np.rot90(aligned, -obj["turn"])
            if obj["flip"]:
                marks = np.fliplr(marks)
            r, c = obj["top"], obj["left"]
            h, w = obj["mask"].shape
            region = output[r:r+h, c:c+w]
            region[obj["mask"]] = obj["body"]
            region[marks] = accent
    return output


def _panels(grid):
    """Find rectangular frames with matching colored corners and uniform sides."""
    h, w = grid.shape
    bg = int(grid[0, 0])
    if any(int(grid[p]) != bg for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))):
        raise ValueError("external corners must expose the background")
    found = []
    # Minimum side seven; generated objects have extent at most five. This also
    # works after rotating the whole scene, and does not name any literal color.
    for top in range(1, h-6):
        for left in range(1, w-6):
            accent, edge = int(grid[top, left]), int(grid[top, left+1])
            if accent == bg or edge in (bg, accent):
                continue
            right = left+1
            while right < w and grid[top, right] == edge:
                right += 1
            if right >= w-1 or right-left < 6 or grid[top, right] != accent:
                continue
            for bottom in range(top+6, h-1):
                if (grid[bottom, left] != accent or grid[bottom, right] != accent
                        or not np.all(grid[bottom, left+1:right] == edge)
                        or not np.all(grid[top+1:bottom, left] == edge)
                        or not np.all(grid[top+1:bottom, right] == edge)):
                    continue
                inner = grid[top+1:bottom, left+1:right].copy()
                if np.any(inner == edge):
                    continue
                found.append(dict(box=(top, left, bottom, right), edge=edge,
                                  accent=accent, grid=inner))
    if len(found) != 3:
        raise ValueError(f"expected three unambiguous frames, found {len(found)}")
    for a, b in combinations(found, 2):
        r, c, y, x = a["box"]
        br, bc, by, bx = b["box"]
        if not (y < br or by < r or x < bc or bx < c):
            raise ValueError("frames must be disjoint")
    accent = found[0]["accent"]
    if any(p["accent"] != accent for p in found):
        raise ValueError("frame corners must agree on the mark role")
    ordered = sorted(found, key=lambda p: p["grid"].size)
    a, b, target = ordered
    if a["grid"].shape != b["grid"].shape or target["grid"].size <= b["grid"].size:
        raise ValueError("need an equal-size reference pair and larger target")
    before = [p for p in (a, b) if p["edge"] == target["edge"]]
    if len(before) != 1:
        raise ValueError("target border must identify exactly one reference-before")
    before = before[0]
    after = b if before is a else a
    reserved = {bg, accent, before["edge"], after["edge"]}
    if len(reserved) != 4:
        raise ValueError("background, marks, and both borders must differ")
    if any(np.isin(p["grid"], [before["edge"], after["edge"]]).any() for p in found):
        raise ValueError("border colors are reserved from interiors")
    return dict(bg=bg, accent=accent, before=before, after=after, target=target)


def _mode(scene):
    before, after = scene["before"]["grid"], scene["after"]["grid"]
    choices = [mode for mode in _MODES
               if np.array_equal(_consensus(before, scene["bg"], scene["accent"], mode), after)]
    if len(choices) != 1:
        raise ValueError("reference must uniquely select union or intersection")
    return choices[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the cropped target answer entirely from this input."""
    scene = _panels(_grid(input_grid))
    return _consensus(scene["target"]["grid"], scene["bg"], scene["accent"], _mode(scene))


def _sample_mask(rng, forbidden, limit, chiral=False, diagonal=False, extent=None,
                 ordinary=False):
    def permitted(mask):
        key = _canonical(mask)
        mirror = _canonical(np.fliplr(mask))
        return (5 <= int(mask.sum()) <= 16 and max(mask.shape) <= limit
                and key not in forbidden
                and (not ordinary or mirror not in forbidden)
                and len({_mask_key(np.rot90(mask, k)) for k in range(4)}) == 4
                and (not chiral or key != mirror)
                and (not diagonal or not _connected(mask, _D4))
                and (extent is None or max(mask.shape) == extent)
                and _connected(mask))

    for _ in range(240):
        kind = int(rng.integers(4))
        if extent is not None:
            # Long backbone with independently placed branches, not enlargement
            # of a training mask. Its exact size forms a new shape class.
            mask = np.zeros((extent, int(rng.integers(2, limit+1))), bool)
            mask[:, 0] = True
            for r in rng.choice(extent, int(rng.integers(1, extent+1)), replace=False):
                mask[r, :int(rng.integers(2, mask.shape[1]+1))] = True
        elif kind == 0 and limit >= 3:
            hh, ww = int(rng.integers(3, limit+1)), int(rng.integers(3, limit+1))
            mask = np.ones((hh, ww), bool)
            mask[1:-1, 1:-1] = False
            mask[0, int(rng.integers(ww))] = False
        elif kind == 1:
            hh, ww = int(rng.integers(2, limit+1)), int(rng.integers(2, limit+1))
            mask = np.zeros((hh, ww), bool)
            mask[:, 0], mask[-1, :] = True, True
            mask[int(rng.integers(hh)), :int(rng.integers(1, ww+1))] = True
        else:
            cells = {(0, 0)}
            dirs = _D8 if diagonal or kind == 3 else _D4
            count = int(rng.integers(5, min(16, limit*limit-1)+1))
            for _ in range(count-1):
                candidates = set()
                for r, c in sorted(cells):
                    for dr, dc in dirs:
                        p = r+dr, c+dc
                        if p in cells:
                            continue
                        rs = [q[0] for q in cells] + [p[0]]
                        cs = [q[1] for q in cells] + [p[1]]
                        if max(rs)-min(rs) < limit and max(cs)-min(cs) < limit:
                            candidates.add(p)
                if not candidates:
                    break
                candidates = sorted(candidates)
                cells.add(candidates[int(rng.integers(len(candidates)))])
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            mask = np.zeros((max(r for r, c in cells)-r0+1,
                             max(c for r, c in cells)-c0+1), bool)
            for r, c in cells:
                mask[r-r0, c-c0] = True
        mask = _trim(mask)
        if permitted(mask):
            return mask
    # Bounded vocabulary fallback, still satisfying all requested properties.
    if extent is None:
        for bits in range(1, 512):
            mask = _trim(np.array([(bits >> i) & 1 for i in range(9)], bool).reshape(3, 3))
            if permitted(mask):
                return mask
    else:
        for width in range(2, limit+1):
            for row in range(extent):
                mask = np.zeros((extent, width), bool)
                mask[:, 0], mask[row, :] = True, True
                if permitted(mask):
                    return mask
    raise RuntimeError("asymmetric shape vocabulary exhausted")


def _mark_sets(rng, mask, count):
    """Purposeful zero/partial/all frequency witnesses plus random extra marks."""
    cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
    rng.shuffle(cells)
    marks = [set() for _ in range(count)]
    if count == 1:
        n = int(rng.integers(1, len(cells)))
        marks[0].update(cells[:n])
        return marks
    # First cell never marked, second marked by all, third by exactly one.
    for m in marks:
        m.add(cells[1])
    one = int(rng.integers(count))
    marks[one].add(cells[2])
    # With triplets, a fourth coordinate occurs in exactly two copies.
    missing = int(rng.integers(count))
    for i, m in enumerate(marks):
        if i != missing:
            m.add(cells[3])
    for p in cells[4:]:
        for m in marks:
            if rng.random() < rng.uniform(.15, .85):
                m.add(p)
    return marks


def _make_class(rng, base, count, class_id, bodies, force_turn=False):
    # Constructor coordinates are the canonical silhouette; marks are sampled
    # once there, then different orientations and colors are independently used.
    _, _, canonical_turn = _alignment(base)
    base = np.rot90(base, canonical_turn).copy()
    sets = _mark_sets(rng, base, count)
    turns = [int(rng.integers(4)) for _ in range(count)]
    if force_turn and count >= 2:
        turns[1] = (turns[0]+1) % 4
    result = []
    for i, turn in enumerate(turns):
        mark_mask = np.zeros_like(base)
        for p in sets[i]:
            mark_mask[p] = True
        result.append(dict(mask=np.rot90(base, turn).copy(),
                           marks=np.rot90(mark_mask, turn).copy(),
                           canonical_marks=sets[i], canonical_shape=base.shape,
                           class_id=class_id, turn=turn, body=int(bodies[i])))
    return result


def _paint_objects(objects, shape, bg, accent, mode=None):
    """Construction path: sampled class labels, not the input-reading solver."""
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj["class_id"]].append(obj)
    pools = {}
    for key, group in grouped.items():
        pools[key] = (set.union(*(o["canonical_marks"] for o in group)) if mode == "union"
                      else set.intersection(*(o["canonical_marks"] for o in group)))
    out = np.full(shape, bg, np.int8)
    for obj in objects:
        if mode is None:
            marks = obj["marks"]
        else:
            marks = np.zeros(obj["canonical_shape"], bool)
            for p in pools[obj["class_id"]]:
                marks[p] = True
            marks = np.rot90(marks, obj["turn"])
        r, c = obj["top"], obj["left"]
        h, w = obj["mask"].shape
        region = out[r:r+h, c:c+w]
        if np.any(region[obj["mask"]] != bg):
            raise RuntimeError("construction overlap")
        region[obj["mask"]] = obj["body"]
        region[marks] = accent
    return out


def _reference(rng, palette):
    bg, accent = map(int, palette[:2])
    base = _sample_mask(rng, set(), 3)
    objects = _make_class(rng, base, 3, 0, palette[4:7], force_turn=True)
    widths = [o["mask"].shape[1] for o in objects]
    needed = sum(widths)+2
    w = int(rng.integers(needed, 12))
    h = int(rng.integers(5, 8))
    slack = w-needed
    extras = rng.multinomial(slack, [1/4]*4)
    left = int(extras[0])
    for i, obj in enumerate(objects):
        obj["left"] = left
        obj["top"] = int(rng.integers(h-obj["mask"].shape[0]+1))
        left += widths[i]+1+int(extras[i+1])
    return (_paint_objects(objects, (h, w), bg, accent),
            {m: _paint_objects(objects, (h, w), bg, accent, m) for m in _MODES})


def _target_objects(rng, palette, teaching, test):
    forbidden, objects = set(), []
    if teaching:
        sizes, singles, mirror = [2, 3], 2, True
    else:
        profiles = [([2], 1), ([3], 1), ([2, 2], 1), ([2, 3], 2),
                    ([3, 3], 1), ([2], 3), ([], 3), ([2, 2], 0)]
        sizes, singles = profiles[int(rng.integers(len(profiles)))]
        sizes = list(sizes)
        if rng.random() < .45:
            singles = min(3, max(0, singles+int(rng.choice([-1, 1]))))
        while sum(sizes)+singles < 3:
            singles += 1
        mirror = bool(sizes and singles and rng.random() < .25)
    limit = 3 if teaching else (5 if test else int(rng.choice([3, 4])))
    body_pool = list(map(int, palette[4:]))
    palette_size = int(rng.integers(2 if teaching else 1, 7))
    chosen = body_pool[:palette_size]
    bases = []
    for i, count in enumerate(sizes):
        base = _sample_mask(rng, forbidden, limit, chiral=mirror and i == 0,
                            diagonal=teaching and i == 1,
                            extent=5 if test and i == 0 else None)
        bases.append(base)
        forbidden.update((_canonical(base), _canonical(np.fliplr(base))))
        bodies = [int(rng.choice(chosen)) for _ in range(count)]
        if teaching and i == 0:
            bodies[:2] = chosen[:2]
        objects.extend(_make_class(rng, base, count, i, bodies, force_turn=teaching))
    for i in range(singles):
        if mirror and i == 0:
            base = np.fliplr(bases[0]).copy()
        else:
            base = _sample_mask(rng, forbidden, limit, ordinary=True,
                                extent=5 if test and not sizes and i == 0 else None)
        forbidden.update((_canonical(base), _canonical(np.fliplr(base))))
        body = chosen[0] if teaching else int(rng.choice(chosen))
        objects.extend(_make_class(rng, base, 1, len(sizes)+i, [body]))
    return objects


def _place_target(rng, objects, max_height):
    # Two independent shelves guarantee a finite fit even for eight 5x5 masks.
    order = [objects[int(i)] for i in rng.permutation(len(objects))]
    cut = int(rng.integers(max(1, len(order)//2-1), min(len(order)-1, len(order)//2+1)+1))
    rows = [order[:cut], order[cut:]]
    if any(sum(o["mask"].shape[1] for o in row)+len(row)-1 > 26 for row in rows):
        cut = len(order)//2
        rows = [order[:cut], order[cut:]]
    gaps = [[int(rng.integers(1, 3)) for _ in range(len(row)-1)] for row in rows]
    widths = [sum(o["mask"].shape[1] for o in row)+sum(g) for row, g in zip(rows, gaps)]
    if max(widths) > 26:
        gaps = [[1]*(len(row)-1) for row in rows]
        widths = [sum(o["mask"].shape[1] for o in row)+len(row)-1 for row in rows]
    heights = [max(o["mask"].shape[0] for o in row) for row in rows]
    row_gap = int(rng.integers(1, 3))
    needed_h = sum(heights)+row_gap
    if needed_h > max_height:
        row_gap, needed_h = 1, sum(heights)+1
    h = int(rng.integers(max(8, needed_h), max_height+1))
    w = int(rng.integers(max(12, max(widths)), 27))
    extras_y = rng.multinomial(h-needed_h, [1/3]*3)
    tops = [int(extras_y[0]), int(extras_y[0])+heights[0]+row_gap+int(extras_y[1])]
    for row, gap, needed_w, height, top in zip(rows, gaps, widths, heights, tops):
        extra = rng.multinomial(w-needed_w, np.full(len(row)+1, 1/(len(row)+1)))
        left = int(extra[0])
        for i, obj in enumerate(row):
            obj["left"] = left
            obj["top"] = top+int(rng.integers(height-obj["mask"].shape[0]+1))
            left += obj["mask"].shape[1]
            if i < len(gap):
                left += gap[i]+int(extra[i+1])
    return h, w


def _frame(grid, interior, position, edge, accent):
    r, c = position
    h, w = interior.shape
    grid[r, c:c+w+2] = edge
    grid[r+h+1, c:c+w+2] = edge
    grid[r:r+h+2, c] = edge
    grid[r:r+h+2, c+w+1] = edge
    for p in ((r, c), (r, c+w+1), (r+h+1, c), (r+h+1, c+w+1)):
        grid[p] = accent
    grid[r+1:r+h+1, c+1:c+w+1] = interior


def _scene_bundle(rng, palette, teaching, test, turn):
    bg, accent, before_edge, after_edge = map(int, palette[:4])
    before, afters = _reference(rng, palette)
    gap_y = int(rng.integers(1, 3))
    max_height = min(16, 24-before.shape[0]-gap_y)
    for _ in range(30):
        objects = _target_objects(rng, palette, teaching, test)
        shape = _place_target(rng, objects, max_height)
        target = _paint_objects(objects, shape, bg, accent)
        answers = {m: _paint_objects(objects, shape, bg, accent, m) for m in _MODES}
        if not teaching or any(not np.array_equal(_consensus(target, bg, accent, m, "reflections"),
                                                  answers[m]) for m in _MODES):
            break
    else:
        raise RuntimeError("could not expose reflection-only distractor")
    ref_h, ref_w = before.shape
    target_h, target_w = target.shape
    gap_x = int(rng.integers(1, min(3, 30-2*(ref_w+2)-2)+1))
    row_width = 2*(ref_w+2)+gap_x
    min_h, min_w = ref_h+target_h+6+gap_y, max(row_width, target_w+2)+2
    h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    y_extra = rng.multinomial(h-min_h, [1/3]*3)
    ref_top = 1+int(y_extra[0])
    target_top = ref_top+ref_h+2+gap_y+int(y_extra[1])
    ref_left = int(rng.integers(1, w-row_width))
    target_left = int(rng.integers(1, w-target_w-2))
    reference_positions = [(ref_top, ref_left), (ref_top, ref_left+ref_w+2+gap_x)]
    if rng.integers(2):
        reference_positions.reverse()
    pairs = {}
    for mode in _MODES:
        grid = np.full((h, w), bg, np.int8)
        _frame(grid, before, reference_positions[0], before_edge, accent)
        _frame(grid, afters[mode], reference_positions[1], after_edge, accent)
        _frame(grid, target, (target_top, target_left), before_edge, accent)
        pairs[mode] = dict(input=np.rot90(grid, turn).copy(),
                           output=np.rot90(answers[mode], turn).copy())
    return pairs


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_palette = rng.permutation(10)
    stable_palette, stable_turn = rng.random() < .22, rng.random() < .22
    preferred_turn = int(rng.integers(4))

    def bundle(teaching=False, test=False):
        palette = base_palette if stable_palette else rng.permutation(10)
        turn = preferred_turn if stable_turn else int(rng.integers(4))
        return _scene_bundle(rng, palette, teaching, test, turn)

    twins = bundle(teaching=True)
    puzzle = {"train": [twins[m] for m in _MODES], "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, count in (("train", a-2), ("test", b)):
        for _ in range(count):
            for _ in range(30):
                pair = bundle(test=split == "test")[_MODES[int(rng.integers(2))]]
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
    order = rng.permutation(a)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    return puzzle


def _features(scene, mode):
    objects = _objects(scene["target"]["grid"], scene["bg"], scene["accent"])
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj["key"]].append(obj)
    repeated = [o for o in objects if len(grouped[o["key"]]) > 1]
    singles = [o for o in objects if len(grouped[o["key"]]) == 1]
    mirror_negative = any(_canonical(np.fliplr(o["mask"])) in grouped
                          and _canonical(np.fliplr(o["mask"])) != o["key"] for o in singles)
    return dict(mode=mode, objects=len(objects), sizes=tuple(sorted(map(len, grouped.values()))),
                extents=tuple(max(o["mask"].shape) for o in objects),
                shapes=tuple(sorted(grouped)), bodies=len({o["body"] for o in objects}),
                diagonal=any(not _connected(o["mask"], _D4) for o in objects),
                mirror_negative=mirror_negative,
                ordinary_singleton=any(_canonical(np.fliplr(o["mask"])) not in grouped
                                       or _canonical(np.fliplr(o["mask"])) == o["key"] for o in singles),
                different_color_match=any(len({o["body"] for o in group}) > 1
                                          for group in grouped.values() if len(group) > 1),
                cross_class_color=any(a["body"] == b["body"] and a["key"] != b["key"]
                                      for a, b in combinations(objects, 2)),
                mixed=bool(repeated and singles))


def _twin_key(inp, scene):
    redacted = inp.copy()
    r, c, y, x = scene["after"]["box"]
    redacted[r+1:y, c+1:x] = scene["bg"]
    return _key(redacted)


def validate_generated_puzzle(puzzle):
    """Schema, pixel invariants, teaching witnesses, and parental ablations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, twins = set(), defaultdict(list)
    features = {"train": [], "test": []}
    alternatives = {k: False for k in ("identity", "color", "reflections", "exactly_two", "singletons")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            scene, mode = _panels(inp), None
            mode = _mode(scene)
            target = scene["target"]["grid"]
            assert out.shape == target.shape
            assert np.array_equal(out != scene["bg"], target != scene["bg"]), "occupancy changed"
            assert np.array_equal(out, solve(inp)), "wrong consensus output"
            # Every member retains only its own body role and the mark role.
            for obj in _objects(target, scene["bg"], scene["accent"]):
                r, c = obj["top"], obj["left"]
                h, w = obj["mask"].shape
                colors = set(map(int, out[r:r+h, c:c+w][obj["mask"]]))
                assert colors <= {obj["body"], scene["accent"]}, "body-role transfer error"
            f = _features(scene, mode)
            features[split].append(f)
            if split == "train":
                assert max(f["extents"]) <= 4, "training scale exceeds intended shift"
                twins[_twin_key(inp, scene)].append((mode, _key(out)))
                for matching in alternatives:
                    alternatives[matching] |= not np.array_equal(
                        _consensus(target, scene["bg"], scene["accent"], mode, matching), out)
            else:
                assert max(f["extents"]) == 5, "test must introduce extent five"
    assert any({m for m, _ in group} == set(_MODES) and len({o for _, o in group}) == 2
               for group in twins.values()), "no informative reference-after twins"
    assert all(alternatives.values()), f"training permits a matching shortcut: {alternatives}"
    train = features["train"]
    assert any(2 in f["sizes"] and 3 in f["sizes"] for f in train)
    for key in ("mirror_negative", "ordinary_singleton", "different_color_match",
                "cross_class_color", "diagonal", "mixed"):
        assert any(f[key] for f in train), f"missing training witness: {key}"


def _puzzle_key(puzzle):
    return tuple(_key(np.asarray(pair[role])) for split in ("train", "test")
                 for pair in puzzle[split] for role in ("input", "output"))


def self_test():
    """Bounded 42-seed variation audit; returns descriptive, reproducible metrics."""
    puzzles, shapes, target_shapes, silhouettes, structures, palettes = set(), set(), set(), set(), set(), set()
    profiles, object_counts, modes, extents = set(), set(), set(), set()
    diagonal = mirror_negative = all_singleton = all_repeated = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                scene = _panels(inp)
                f = _features(scene, _mode(scene))
                shapes.add(inp.shape)
                target_shapes.add(pair["output"].shape)
                silhouettes.update(f["shapes"])
                profiles.add(f["sizes"])
                object_counts.add(f["objects"])
                modes.add((split, f["mode"]))
                extents.update(f["extents"])
                palettes.add((scene["bg"], scene["accent"], scene["before"]["edge"], scene["after"]["edge"]))
                roles = {}
                normalized = np.array([roles.setdefault(int(x), len(roles)) for x in inp.flat], np.uint8).reshape(inp.shape)
                structures.add(_key(normalized))
                diagonal += f["diagonal"]
                mirror_negative += f["mirror_negative"]
                all_singleton += max(f["sizes"]) == 1
                all_repeated += min(f["sizes"]) > 1
    assert len(puzzles) == 42
    assert len(shapes) >= 20 and len(target_shapes) >= 30
    assert len(silhouettes) >= 60 and len(structures) >= 150 and len(palettes) >= 25
    assert {3, 4, 5, 6, 7} <= object_counts and len(profiles) >= 8
    assert modes == {(s, m) for s in ("train", "test") for m in _MODES}
    assert 5 in extents and min(diagonal, mirror_negative, all_singleton, all_repeated) > 0
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(generate_puzzle(0))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(8000+a*10+b, a, b)
        validate_generated_puzzle(p)
        assert len(p["train"]) == a and len(p["test"]) == b
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(8000+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = generate_puzzle(79)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzles), canvas_shapes=len(shapes),
                target_shapes=len(target_shapes), silhouette_classes=len(silhouettes),
                normalized_inputs=len(structures), role_palettes=len(palettes),
                class_profiles=len(profiles), object_counts=sorted(object_counts),
                diagonal_scenes=int(diagonal), mirror_negative_scenes=int(mirror_negative),
                all_singleton_scenes=int(all_singleton), all_repeated_scenes=int(all_repeated),
                semantic_verification="generator-owned checks only")
