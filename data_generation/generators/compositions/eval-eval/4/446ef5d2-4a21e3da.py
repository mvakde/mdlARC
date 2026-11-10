"""Ray-batched jigsaw packets. All solution conventions are input-visible.

Border rays define rigid packets, including disconnected fragments. Retained
packets tile one rectangle, using exact seam colors and an external L anchor.
No parent generator, source fixture, seed, or generation metadata is used by solve.
"""
from __future__ import annotations

from itertools import permutations
import json
import numpy as np


_CORNERS = ("TL", "TR", "BR", "BL")
_SIDES = ("T", "R", "B", "L")
HOST_SEEDS = (0, 37560473, 805330885, 186051989, 959708222, 658805456,
              549366570, 140093350, 998118937, 779010508, 809783637,
              776150764, 29603603, 756782621, 226629160, 901977012,
              193066371, 80817651, 707412911, 403799895, 950650937,
              728667584, 237797141, 472391284, 810179749, 253938433,
              562843096, 914477782, 978021550, 84772470, 248006033,
              765453101, 738352851, 864157370, 529138582, 630621708,
              566139318, 282684239, 499724251, 15731636, 144673074,
              198094435)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, queue = [start], [start]
        for r, c in queue:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    queue.append(p)
                    component.append(p)
        result.append(component)
    return result


def _piece(grid, cells):
    points = np.array(cells, dtype=int)
    r, c = points.min(axis=0)
    h, w = points.max(axis=0) - (r, c) + 1
    matrix = np.full((h, w), -1, dtype=np.int8)
    for y, x in cells:
        matrix[y-r, x-c] = grid[y, x]
    return matrix, (int(r), int(c))


def _l_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((r+dy, c), (r, c+dx), (r+dy, c+dx))


def _extract(grid, split_fragments=False, retain_all=False):
    """Identify roles, full conceptual axes, selected regions, and L anchor."""
    h, w = grid.shape
    border = np.zeros((h, w), bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    rim = grid[border]
    values, counts = np.unique(rim, return_counts=True)
    bg = int(values[counts.argmax()])
    if int(counts.max()) <= len(rim)//2:
        raise ValueError("background must dominate the border")
    markers = [tuple(map(int, p)) for p in np.argwhere(border & (grid != bg))]
    if len(markers) not in (1, 2):
        raise ValueError("one or two border markers required")
    guide = int(grid[markers[0]])
    if any(grid[p] != guide for p in markers) or np.count_nonzero(grid == guide) != len(markers):
        raise ValueError("guide color occurs only at the border markers")
    sides, row_axis, col_axis = set(), None, None
    for r, c in markers:
        flags = [s for s, yes in (("T", r == 0), ("B", r == h-1),
                                 ("L", c == 0), ("R", c == w-1)) if yes]
        if len(flags) != 1 or flags[0] in sides:
            raise ValueError("markers must lie on distinct noncorner sides")
        side = flags[0]
        sides.add(side)
        if side in "TB":
            col_axis = c
        else:
            row_axis = r
    if len(sides) == 2 and (row_axis is None or col_axis is None):
        raise ValueError("two marked sides must be adjacent")
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        q = np.argwhere(grid == color)
        if color in (bg, guide) or len(q) != 3:
            continue
        lo, hi = q.min(0), q.max(0)
        if tuple(hi-lo) != (1, 1):
            continue
        missing = {(int(lo[0]), int(lo[1])), (int(lo[0]), int(hi[1])),
                   (int(hi[0]), int(lo[1])), (int(hi[0]), int(hi[1]))} - set(map(tuple, q))
        point = missing.pop()
        if grid[point] in (bg, guide, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 0): "BR", (0, 1): "BL"}[tuple(np.array(point)-lo)]
        candidates.append((color, point, corner))
    if len(candidates) != 1:
        raise ValueError("a unique external three-cell L is required")
    anchor_color, point, corner = candidates[0]
    payload = (grid != bg) & (grid != guide) & (grid != anchor_color)
    if np.any(payload & border):
        raise ValueError("payload must be strictly inside the canvas")
    if ((row_axis is not None and np.any(payload[row_axis]))
            or (col_axis is not None and np.any(payload[:, col_axis]))):
        raise ValueError("the conceptual guide axes must be empty of payload")
    groups = {}
    for raw_r, raw_c in np.argwhere(payload):
        r, c = int(raw_r), int(raw_c)
        key = (("T" if r < row_axis else "B") if row_axis is not None else "",
               ("L" if c < col_axis else "R") if col_axis is not None else "")
        groups.setdefault(key, []).append((r, c))
    if len(groups) != (4 if len(sides) == 2 else 2):
        raise ValueError("every ray-defined region must contain payload")
    retained, discarded = {}, []
    for key, cells in sorted(groups.items()):
        if len(sides) == 1 or set(key) & sides or retain_all:
            retained[key] = cells
        else:
            discarded.extend(cells)
    pieces, anchor = [], None
    fragment_counts = []
    for cells in retained.values():
        components = _components(cells)
        fragment_counts.append(len(components))
        for batch in (components if split_fragments else [cells]):
            if point in batch:
                anchor = len(pieces)
            pieces.append(_piece(grid, batch))
    if anchor is None:
        raise ValueError("the L must anchor a retained packet")
    matrix, (r, c) = pieces[anchor]
    expected = (r + (matrix.shape[0]-1 if corner[0] == "B" else 0),
                c + (matrix.shape[1]-1 if corner[1] == "R" else 0))
    if point != expected:
        raise ValueError("the L must mark an occupied bounding-box corner")
    return dict(bg=bg, guide=guide, anchor_color=anchor_color, point=point,
                corner=corner, sides=sides, axes=(row_axis, col_axis),
                pieces=pieces, anchor=anchor, discarded=discarded,
                fragment_counts=fragment_counts, retained=retained)


def _assemblies(info, canvas):
    """Complete translation-only mask packing; small packet count bounds search.

    Each placement at the earliest uncovered cell aligns the new packet's first
    occupied cell. Earlier occupied cells cannot be used, so this ordering is
    complete for disconnected masks too. No source position except the anchor
    constrains packing. All contacts between distinct packets must agree.
    """
    pieces = info["pieces"]
    n, anchor = len(pieces), info["anchor"]
    area = sum(int(np.count_nonzero(m >= 0)) for m, _ in pieces)
    point, corner, bg = info["point"], info["corner"], info["bg"]
    results, geometry_count, rejected, nodes = {}, 0, 0, 0
    for h in range(1, canvas[0]+1):
        if area % h:
            continue
        w = area//h
        oy = point[0] - (h-1 if corner[0] == "B" else 0)
        ox = point[1] - (w-1 if corner[1] == "R" else 0)
        if w > canvas[1] or min(oy, ox) < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m, _ in pieces):
            continue
        configs = []
        for m, _ in pieces:
            cells = [(int(y), int(x)) for y, x in np.argwhere(m >= 0)]
            bits = sum(1 << (y*w+x) for y, x in cells)
            configs.append((m.shape, cells, cells[0][1], bits))
        ar, ac = pieces[anchor][1][0]-oy, pieces[anchor][1][1]-ox
        ah, aw = pieces[anchor][0].shape
        if ar < 0 or ac < 0 or ar+ah > h or ac+aw > w:
            continue
        positions = [None]*n
        positions[anchor] = (ar, ac)
        full = (1 << area)-1
        occupied = configs[anchor][3] << (ar*w+ac)

        def visit(occ, remaining):
            nonlocal geometry_count, rejected, nodes
            nodes += 1
            if nodes > 250000:
                raise ValueError("packing search exceeded its explicit budget")
            if not remaining:
                geometry_count += 1
                picture = np.full((h, w), -1, np.int8)
                labels = np.full((h, w), -1, int)
                for i, (y, x) in enumerate(positions):
                    for dy, dx in configs[i][1]:
                        picture[y+dy, x+dx] = pieces[i][0][dy, dx]
                        labels[y+dy, x+dx] = i
                assert np.all(picture >= 0)
                across = labels[:, 1:] != labels[:, :-1]
                down = labels[1:] != labels[:-1]
                mismatch = np.any(across & (picture[:, 1:] != picture[:, :-1])) or np.any(down & (picture[1:] != picture[:-1]))
                if mismatch:
                    rejected += 1
                    return
                out = np.full(canvas, bg, np.int8)
                out[oy:oy+h, ox:ox+w] = picture
                key = out.tobytes()
                results.setdefault(key, (out, tuple((y+oy, x+ox) for y, x in positions), labels))
                return
            first = ((full ^ occ) & -(full ^ occ)).bit_length()-1
            y, x = divmod(first, w)
            for i in remaining:
                (ph, pw), _, start_col, bits = configs[i]
                xx = x-start_col
                if xx < 0 or y+ph > h or xx+pw > w:
                    continue
                placed = bits << (y*w+xx)
                if not occ & placed:
                    positions[i] = (y, xx)
                    visit(occ | placed, tuple(j for j in remaining if j != i))

        visit(occupied, tuple(i for i in range(n) if i != anchor))
    return results, geometry_count, rejected


def _infer(grid):
    info = _extract(grid)
    candidates, count, rejected = _assemblies(info, grid.shape)
    if len(candidates) != 1:
        raise ValueError(f"expected one supported output, found {len(candidates)}")
    out, positions, labels = next(iter(candidates.values()))
    return out, info, positions, labels, count, rejected


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the supplied input; never mutate it."""
    return _infer(_grid(input_grid))[0].copy()


def _target_labels(rng, n, kind):
    if kind == "pattern":
        s = int(rng.integers(3, 6))
        labels = np.zeros((s, 2*s), int)
        labels[:, s:] = 1
        if rng.integers(2):
            labels = labels.T.copy()
        return labels
    h, w = int(rng.integers(5, 12)), int(rng.integers(5, 12))
    labels = np.zeros((h, w), int)
    cut = int(rng.integers(2, w-1))
    if n == 2:
        labels[:, cut:] = 1
    elif rng.random() < .5:
        cut2 = int(rng.integers(cut+1, w))
        labels[:, cut:cut2] = 1
        labels[:, cut2:] = 2
    else:
        split = int(rng.integers(2, h-1))
        labels[:split, cut:] = 1
        labels[split:, cut:] = 2
    if kind in ("steps", "islands"):
        # Move a seam cell, preserving complementary occupancy rather than
        # changing the final rectangular area.
        y = int(rng.integers(1, h-1))
        labels[y, cut] = 0
    if kind == "islands":
        # A remote cell creates a genuinely detached companion fragment.
        if w-cut < 3:
            labels = np.pad(labels, ((0, 0), (0, 3)), mode="edge")
            w = labels.shape[1]
        y, x = int(rng.integers(1, h-1)), w-2
        labels[y, x] = 0
        if h >= 8 and rng.random() < .5:
            other = 1 if y > h//2 else h-2
            if abs(other-y) >= 3:
                labels[other, x] = 0
    if rng.integers(2):
        labels = labels.T.copy()
    if rng.integers(2):
        labels = labels[::-1].copy()
    if rng.integers(2):
        labels = labels[:, ::-1].copy()
    return labels


def _picture(rng, labels, palette):
    h, w = labels.shape
    carrier = palette[0]
    pic = np.full((h, w), carrier, np.int8)
    # Coherent regions, outlines, and bars precede the partition seam smoothing.
    style = int(rng.integers(3))
    for k in range(int(rng.integers(3, 8))):
        y, x = int(rng.integers(h-1)), int(rng.integers(w-1))
        hh, ww = int(rng.integers(2, h-y+1)), int(rng.integers(2, w-x+1))
        color = palette[1+k % (len(palette)-1)]
        if style == 0:
            pic[y:y+hh, x:x+ww] = color
        elif style == 1:
            pic[y:y+hh, x] = color
            pic[y:y+hh, x+ww-1] = color
            pic[y, x:x+ww] = color
            pic[y+hh-1, x:x+ww] = color
        else:
            pic[y:y+min(hh, 2), x:x+ww] = color
    parents = list(range(h*w))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for y in range(h):
        for x in range(w):
            for yy, xx in ((y+1, x), (y, x+1)):
                if yy < h and xx < w and labels[y, x] != labels[yy, xx]:
                    parents[root(yy*w+xx)] = root(y*w+x)
    groups = {}
    for i in range(h*w):
        groups.setdefault(root(i), []).append(i)
    for cells in groups.values():
        values, counts = np.unique(pic.ravel()[cells], return_counts=True)
        color = int(rng.choice(values[counts == counts.max()]))
        pic.ravel()[cells] = color
    return pic


def _fragment_witness(rng, corner, palette):
    """Two equal panels; a companion dot locks their order as one packet.

    If that dot moves independently, both panel orders match all seams and give
    different pictures. The packet spacing is the essential extra constraint.
    """
    s, a, b = int(rng.integers(4, 6)), int(rng.integers(2, 4)), 6
    labels = np.zeros((2*s, a+b), int)
    labels[:s, a:] = 1
    labels[s:, a:] = 2
    labels[s+1, a+b-2] = 0
    pic = np.full(labels.shape, palette[0], np.int8)
    pic[1:3, a+1:a+3] = palette[1]
    pic[s+1:s+3, a+1:a+3] = palette[2]
    if rng.integers(2):
        labels, pic = labels[::-1].copy(), pic[::-1].copy()
    if corner[1] == "R":
        labels, pic = labels[:, ::-1].copy(), pic[:, ::-1].copy()
    return labels, pic


def _regions(shape, axes):
    h, w = shape
    row, col = axes
    rows = [("", 1, h-2)] if row is None else [("T", 1, row-1), ("B", row+1, h-2)]
    cols = [("", 1, w-2)] if col is None else [("L", 1, col-1), ("R", col+1, w-2)]
    return {(v, s): (r0, r1, c0, c1) for v, r0, r1 in rows for s, c0, c1 in cols}


def _scatter(rng, labels, pic, corner, sides, roles):
    bg, guide, anchor_color = roles
    n = int(labels.max())+1
    target = pic.shape
    pieces = []
    for i in range(n):
        pieces.append(_piece(np.where(labels == i, pic, bg), [tuple(map(int, p)) for p in np.argwhere(labels == i)]))
    mh = max(m.shape[0] for m, _ in pieces)
    mw = max(m.shape[1] for m, _ in pieces)
    vertical = bool(set(sides) & set("TB"))
    horizontal = bool(set(sides) & set("LR"))
    hmin = 2*mh+3 if horizontal else max(mh, target[0])+4
    wmin = 2*mw+3 if vertical else max(mw, target[1])+4
    if hmin > 30 or wmin > 30:
        return None
    h = int(rng.integers(hmin, min(30, hmin+7)+1))
    w = int(rng.integers(wmin, min(30, wmin+7)+1))
    row = int(rng.integers(mh+1, h-mh-1)) if horizontal else None
    col = int(rng.integers(mw+1, w-mw-1)) if vertical else None
    regions = _regions((h, w), (row, col))
    active = [key for key in regions if len(sides) == 1 or set(key) & set(sides)]
    rng.shuffle(active)
    local_point = (target[0]-1 if corner[0] == "B" else 0,
                   target[1]-1 if corner[1] == "R" else 0)
    anchor = int(labels[local_point])
    assignments = list(permutations(active))
    rng.shuffle(assignments)
    for assignment in assignments:
        positions = []
        for i, (m, _) in enumerate(pieces):
            r0, r1, c0, c1 = regions[assignment[i]]
            ph, pw = m.shape
            choices = []
            for y in range(r0, r1-ph+2):
                for x in range(c0, c1-pw+2):
                    if i == anchor:
                        point = (y+(ph-1 if corner[0] == "B" else 0),
                                 x+(pw-1 if corner[1] == "R" else 0))
                        l_cells = _l_cells(point, corner)
                        if not all(r0 <= yy <= r1 and c0 <= xx <= c1 for yy, xx in l_cells):
                            continue
                        oy, ox = point[0]-local_point[0], point[1]-local_point[1]
                        if oy < 0 or ox < 0 or oy+target[0] > h or ox+target[1] > w:
                            continue
                    choices.append((y, x))
            if not choices:
                break
            positions.append(choices[int(rng.integers(len(choices)))])
        if len(positions) == n:
            break
    else:
        return None
    grid = np.full((h, w), bg, np.int8)
    for (m, _), (y, x) in zip(pieces, positions):
        sub = grid[y:y+m.shape[0], x:x+m.shape[1]]
        sub[m >= 0] = m[m >= 0]
    if len(sides) == 2:
        inactive = next(key for key in regions if key not in active)
        r0, r1, c0, c1 = regions[inactive]
        dh, dw = int(rng.integers(2, min(5, r1-r0+1)+1)), int(rng.integers(2, min(5, c1-c0+1)+1))
        y, x = int(rng.integers(r0, r1-dh+2)), int(rng.integers(c0, c1-dw+2))
        decoy = rng.choice(np.unique(pic), size=(dh, dw)).astype(np.int8)
        grid[y:y+dh, x:x+dw] = decoy
        if dh >= 3 and dw >= 3 and rng.integers(2):
            grid[y+1, x+1] = bg
    y, x = positions[anchor]
    ph, pw = pieces[anchor][0].shape
    point = (y+(ph-1 if corner[0] == "B" else 0), x+(pw-1 if corner[1] == "R" else 0))
    for p in _l_cells(point, corner):
        assert grid[p] == bg
        grid[p] = anchor_color
    for s in sides:
        p = {"T": (0, col), "B": (h-1, col), "L": (row, 0), "R": (row, w-1)}[s]
        grid[p] = guide
    out = np.full_like(grid, bg)
    oy, ox = point[0]-local_point[0], point[1]-local_point[1]
    out[oy:oy+target[0], ox:ox+target[1]] = pic
    return {"input": grid, "output": out}


def _make_pair(rng, corner, orientation, roles, palette, kind):
    sides = (_SIDES[orientation % 4],) if orientation < 4 else (_SIDES[orientation % 4], _SIDES[(orientation+1) % 4])
    n = 2 if len(sides) == 1 else 3
    for _ in range(100):
        if kind == "fragments":
            labels, pic = _fragment_witness(rng, corner, palette)
        else:
            labels = _target_labels(rng, n, kind)
            pic = _picture(rng, labels, palette)
        pair = _scatter(rng, labels, pic, corner, sides, roles)
        if pair is None:
            continue
        try:
            predicted, info, _, _, geometry_count, rejected = _infer(pair["input"])
        except ValueError:
            continue
        if not np.array_equal(predicted, pair["output"]):
            continue
        if kind == "pattern" and not (geometry_count > 1 and rejected > 0):
            continue
        if kind == "fragments":
            alt, _, _ = _assemblies(_extract(pair["input"], split_fragments=True), pair["input"].shape)
            if len(alt) < 2:
                continue
        if kind == "islands" and max(info["fragment_counts"]) < 2:
            continue
        return pair
    raise RuntimeError("bounded construction exhausted valid packet candidates")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    order = list(map(int, rng.permutation(10)))
    stable = bool(rng.random() < .25)
    start = int(rng.integers(4))
    train_corners = (_CORNERS[start], _CORNERS[(start+2) % 4])
    test_corners = (_CORNERS[(start+1) % 4], _CORNERS[(start+3) % 4])
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            if split == "train" and i < 2:
                corner = train_corners[i]
                kind = "pattern" if i == 0 else "fragments"
                orientation = int(rng.integers(4)) + (4 if i else 0)
            else:
                corner = str(rng.choice(train_corners if split == "train" else test_corners))
                kind = "islands" if split == "test" else str(rng.choice(("rectangles", "steps", "islands")))
                orientation = int(rng.integers(8))
            for _ in range(20):
                colors = order if stable else list(map(int, rng.permutation(10)))
                roles = tuple(colors[:3])
                size = 3 if stable or kind == "fragments" else int(rng.integers(2, 6))
                palette = colors[3:3+size]
                pair = _make_pair(rng, corner, orientation, roles, palette, kind)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate schema, exact conservation, rigid packets, and training evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            pred, info, positions, labels, geom, rejected = _infer(inp)
            assert np.array_equal(pred, out)
            bg = info["bg"]
            assert not np.any(out == info["guide"]) and not np.any(out == info["anchor_color"])
            assert positions[info["anchor"]] == info["pieces"][info["anchor"]][1]
            occupied = np.zeros_like(out, bool)
            for (m, _), (y, x) in zip(info["pieces"], positions):
                mask = m >= 0
                assert not np.any(occupied[y:y+m.shape[0], x:x+m.shape[1]][mask])
                occupied[y:y+m.shape[0], x:x+m.shape[1]][mask] = True
                assert np.array_equal(out[y:y+m.shape[0], x:x+m.shape[1]][mask], m[mask])
            q = np.argwhere(occupied)
            lo, hi = q.min(0), q.max(0)+1
            assert occupied[lo[0]:hi[0], lo[1]:hi[1]].all()
            assert np.count_nonzero(out != bg) == int(occupied.sum())
            for color in range(10):
                if color not in (bg, info["guide"], info["anchor_color"]):
                    removed = sum(int(inp[p] == color) for p in info["discarded"])
                    assert np.count_nonzero(inp == color) - removed == np.count_nonzero(out == color)
            # Normalize region membership, not physical connected components.
            fragment_witness = False
            if max(info["fragment_counts"]) > 1 and sum(info["fragment_counts"]) <= 5:
                alternatives, _, _ = _assemblies(_extract(inp, split_fragments=True), inp.shape)
                fragment_witness = any(k != out.tobytes() for k in alternatives)
            features[split].append(dict(corner=info["corner"], sides=tuple(sorted(info["sides"])),
                                        arity=len(info["sides"]), fragments=max(info["fragment_counts"]),
                                        fragment_witness=fragment_witness,
                                        pattern_witness=geom > 1 and rejected > 0,
                                        discarded=len(info["discarded"]),
                                        packet_count=len(info["pieces"]),
                                        shape=tuple(out.shape), target=tuple(hi-lo),
                                        palette=tuple(map(int, np.unique(inp))),
                                        roles=(bg, info["guide"], info["anchor_color"])))
    tr, te = features["train"], features["test"]
    assert any(f["pattern_witness"] for f in tr), "missing pattern-vs-shape evidence"
    assert any(f["fragment_witness"] for f in tr), "missing rigid-fragment evidence"
    assert {f["arity"] for f in tr} == {1, 2}
    assert any(f["discarded"] > 0 for f in tr), "missing selective-deletion evidence"
    assert {f["corner"] for f in tr}.isdisjoint({f["corner"] for f in te})
    assert all(f["fragments"] >= 2 for f in te)
    return features


def _fingerprint(puzzle):
    return tuple((np.asarray(p["input"]).shape, np.asarray(p["input"]).tobytes(),
                  np.asarray(p["output"]).tobytes()) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded, returned-grid measurements over exactly the 42 supplied seeds."""
    fingerprints, shapes, targets, structures, roles = set(), set(), set(), set(), set()
    orientations = {"train": set(), "test": set()}
    corners = set()
    pairs = fragment_cases = pattern_cases = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        f = validate_generated_puzzle(p)
        fingerprints.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair, feature in zip(p[split], f[split]):
                inp = np.asarray(pair["input"])
                copy = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, copy)
                pairs += 1
                shapes.add(feature["shape"])
                targets.add(feature["target"])
                roles.add(feature["roles"])
                orientations[split].add(feature["sides"])
                corners.add(feature["corner"])
                fragment_cases += feature["fragments"] > 1
                pattern_cases += feature["pattern_witness"]
                info = _extract(inp)
                structures.add(tuple((m.shape, (m >= 0).tobytes()) for m, _ in info["pieces"]))
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(targets) >= 25 and len(structures) >= 60
    assert corners == set(_CORNERS)
    assert len(roles) >= 30
    assert all(len(s) == 8 for s in orientations.values())
    assert fragment_cases >= 84 and pattern_cases >= 42
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    # Geometry and visible color roles must commute with whole-scene symmetries.
    for pair in generate_puzzle(0)["train"][:2]:
        for turns in range(4):
            assert np.array_equal(solve(np.rot90(pair["input"], turns)),
                                  np.rot90(pair["output"], turns))
        assert np.array_equal(solve(pair["input"][:, ::-1]), pair["output"][:, ::-1])
        rename = np.array([7, 4, 9, 1, 6, 0, 8, 3, 5, 2], dtype=np.int8)
        assert np.array_equal(solve(rename[pair["input"]]), rename[pair["output"]])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(0, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validation must reject broken conservation and missing learning evidence.
    p = generate_puzzle(0)
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    info = _extract(damaged["test"][0]["input"])
    damaged["test"][0]["output"][info["point"]] = info["guide"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=pairs, canvas_shapes=len(shapes), target_shapes=len(targets),
                packet_structures=len(structures), role_triples=len(roles),
                fragment_cases=fragment_cases, pattern_cases=pattern_cases,
                orientations={s: len(v) for s, v in orientations.items()})


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
