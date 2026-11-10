"""Exact-cover band tabs determine the union of clear clipped 3x3 windows.

Standalone: only Python's standard library and NumPy are used. No parent files,
saved examples, random state or generation parameters are consulted by solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


HIGHLIGHT = 3
HOST_SEEDS = (0,64751823,553994709,853793838,470345507,869888548,467703626,
              275179003,720799688,64512024,953415109,692850688,43205827,
              302566860,245029666,602579765,42264307,180931807,640969898,
              938837387,462549157,164329347,59201471,523044301,886671309,
              543286206,616506814,267339855,285173991,321784118,721999098,
              982781663,350449012,276361637,856723503,672928648,688510603,
              233114194,594250068,407454577,8276703,697934679)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= p[dr:dr+h, dc:dc+w]
    return out


def _window_masks(grid, background):
    """Clear centers, then union of their complete IN-GRID window footprints."""
    centers = ~_dilate(grid != background)
    return centers, _dilate(centers) & (grid == background)


def _literal_windows(grid, background, clipped=True):
    """Separate, deliberately literal oracle for validation; at most 900 windows."""
    h, w = grid.shape
    union = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not clipped and (r == 0 or c == 0 or r == h-1 or c == w-1):
                continue
            rs, cs = slice(max(0, r-1), min(h, r+2)), slice(max(0, c-1), min(w, c+2))
            if np.all(grid[rs, cs] == background):
                union[rs, cs] = True
    return union


def _components(grid, background):
    remaining = set(map(tuple, np.argwhere(grid != background)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        color, todo, cells = int(grid[first]), [first], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining and grid[q] == color:
                    remaining.remove(q)
                    todo.append(q)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _scene(grid):
    if np.any(grid == HIGHLIGHT):
        raise ValueError("3 is reserved for the output highlight")
    histogram = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(histogram))
    if histogram[bg] <= grid.size // 2:
        raise ValueError("background must occupy strictly more than half the input")
    components = _components(grid, bg)
    spanning = [(i, axis) for i, (cells, _) in enumerate(components)
                for axis in (0, 1)
                if len(np.unique(cells[:, 1-axis])) == grid.shape[1-axis]]
    if len(spanning) != 1:
        raise ValueError("expected one monochrome spanning band")
    band_index, axis = spanning[0]
    band = components[band_index][1]
    canonical = grid if axis == 0 else grid.T
    band_mask = canonical == band
    edges = []
    for c in range(canonical.shape[1]):
        rows = np.flatnonzero(band_mask[:, c])
        if not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[-1]+1)):
            raise ValueError("band must be an interval in each cross-section")
        edges.append((int(rows[0]), int(rows[-1])+1))
    pieces, occupied = [], band_mask.copy()
    for i, (cells, color) in enumerate(components):
        if i == band_index:
            continue
        if color == band:
            raise ValueError("band color is reserved for the band")
        cells = cells if axis == 0 else cells[:, ::-1]
        if np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]]):
            raise ValueError("input pieces need one-cell separation")
        occupied[cells[:, 0], cells[:, 1]] = True
        pieces.append((cells, color))
    if not pieces:
        raise ValueError("expected movable pieces")
    return canonical, bg, band, axis, np.array(edges), pieces


def _piece_profile(cells, sign):
    cross = cells[:, 1]-cells[:, 1].min()
    outward = sign*cells[:, 0]
    lo, hi = [], []
    for c in range(int(cross.max())+1):
        values = np.sort(outward[cross == c])
        if not len(values) or not np.array_equal(values, np.arange(values[0], values[-1]+1)):
            raise ValueError("pieces must be full-width and column-convex along the band")
        lo.append(int(values[0]))
        hi.append(int(values[-1])+1)
    origin = min(lo)
    return (tuple(x-origin for x in lo), tuple(x-origin for x in hi),
            np.column_stack((outward-origin, cross)))


def _cell_bits(cells, width):
    bits = 0
    for r, c in cells:
        bits |= 1 << (int(r)*width+int(c))
    return bits


def _packings(grid):
    """Search every side/alignment/subset/order, then cover all source objects.

    A tab is a full-width chain of column-convex pieces, closed by a flat outer
    edge. Neither maximal notches nor width/color dictionaries choose its site.
    Stop only after two DIFFERENT colored completed grids have been found.
    """
    a, bg, band, axis, edges, pieces = _scene(grid)
    h, w = a.shape
    base = np.full_like(a, bg)
    base[a == band] = band
    candidates = []
    local_starts = set()
    for sign in (-1, 1):
        descriptions = [_piece_profile(cells, sign) for cells, _ in pieces]
        lookup = {}
        for i, (lo, _, _) in enumerate(descriptions):
            lookup.setdefault(lo, []).append(i)
        boundary = 1-edges[:, 0] if sign == -1 else edges[:, 1]
        limit = 1 if sign == -1 else h
        for width in sorted({len(d[0]) for d in descriptions}):
            for start in range(w-width+1):
                root = tuple(map(int, boundary[start:start+width]))

                def extend(profile, used, placements, occupied):
                    if used and len(set(profile)) == 1:
                        candidates.append((used, occupied, placements,
                                           (sign, start, width, root, profile[0])))
                    offset = min(profile)
                    normalized = tuple(x-offset for x in profile)
                    for i in lookup.get(normalized, ()):
                        if used & (1 << i):
                            continue
                        _, upper, local = descriptions[i]
                        next_profile = tuple(offset+x for x in upper)
                        if max(next_profile) > limit:
                            continue
                        cells = np.column_stack((sign*(local[:, 0]+offset), local[:, 1]+start))
                        if np.any((cells[:, 0] < 0) | (cells[:, 0] >= h)):
                            continue
                        if np.any(base[cells[:, 0], cells[:, 1]] != bg):
                            continue
                        if set(map(tuple, cells)) == set(map(tuple, pieces[i][0])):
                            continue
                        cell_bits = _cell_bits(cells, w)
                        if occupied & cell_bits:
                            continue
                        if not used:
                            local_starts.add((i, sign, start))
                        extend(next_profile, used | (1 << i), placements+[(i, cells)],
                               occupied | cell_bits)

                extend(root, 0, [], 0)
    by_piece = [[] for _ in pieces]
    for candidate in candidates:
        for i in range(len(pieces)):
            if candidate[0] & (1 << i):
                by_piece[i].append(candidate)
    full = (1 << len(pieces))-1
    answers = {}

    def cover(used, occupied, chosen):
        if len(answers) >= 2:
            return
        if used == full:
            answer = base.copy()
            for candidate in chosen:
                for i, cells in candidate[2]:
                    answer[cells[:, 0], cells[:, 1]] = pieces[i][1]
            answers.setdefault(answer.tobytes(), (answer, chosen))
            return
        options = min(([c for c in by_piece[i] if not c[0] & used and not c[1] & occupied]
                       for i in range(len(pieces)) if not used & (1 << i)), key=len)
        for candidate in options:
            cover(used | candidate[0], occupied | candidate[1], chosen+[candidate])

    cover(0, 0, [])
    info = {"background": bg, "band": band, "axis": axis, "pieces": pieces,
            "edges": edges, "local_starts": len(local_starts),
            "candidate_count": len(candidates)}
    return list(answers.values()), info


def _completed(grid):
    answers, info = _packings(grid)
    if len(answers) != 1:
        raise ValueError("input must have one distinct colored packing")
    a, tabs = answers[0]
    info["tabs"] = tabs
    return (a if info["axis"] == 0 else a.T).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    completed, info = _completed(_grid(input_grid))
    _, union = _window_masks(completed, info["background"])
    out = completed.copy()
    out[union] = HIGHLIGHT
    return out


def _contour(rng, width, amplitude):
    mode = int(rng.integers(4))
    if mode == 0:
        p = rng.integers(0, amplitude+1, width)
    elif mode == 1:
        p = np.cumsum(rng.integers(-1, 2, width))
        p = np.clip(p-p.min(), 0, amplitude)
    elif mode == 2:
        p = np.floor(np.linspace(0, amplitude, width)).astype(int)
        if rng.integers(2):
            p = p[::-1]
    else:
        p = np.zeros(width, dtype=int)
        cut = int(rng.integers(1, width))
        p[:cut] = int(rng.integers(1, amplitude+1))
    p -= p.min()
    if not p.max():
        p[int(rng.integers(width))] = 1
    return p


def _cuts(rng, width, count):
    cuts = [_contour(rng, width, int(rng.integers(1, 3)))]
    for _ in range(count-1):
        shape = _contour(rng, width, int(rng.integers(1, 4)))
        previous = cuts[-1]
        # Positive column thickness AND neighboring interval overlap.
        offset = max(int(np.max(previous+1-shape)),
                     int(np.max(np.maximum(previous[:-1], previous[1:])+1
                                -np.minimum(shape[:-1], shape[1:]))))
        cuts.append(shape+offset+int(rng.integers(2)))
    cuts.append(np.full(width, int(cuts[-1].max())+int(rng.integers(1, 4))))
    return cuts


def _scatter(rng, base, pieces, bg):
    h, w = base.shape
    for _ in range(12):
        a = base.copy()
        blocked = _dilate(base != bg)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0])+rng.random()*5)
        for i in order:
            cells, color = pieces[i]
            origin = cells.min(axis=0)
            shape = cells-origin
            ph, pw = shape.max(axis=0)+1
            if ph > h or pw > w:
                break
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in shape:
                valid &= ~blocked[r:r+valid.shape[0], c:c+valid.shape[1]]
            valid[tuple(origin)] = False
            locations = np.argwhere(valid)
            if not len(locations):
                break
            target = shape+locations[int(rng.integers(len(locations)))]
            a[target[:, 0], target[:, 1]] = color
            blocked = _dilate(a != bg)
        else:
            return a
    raise ValueError("scatter clearance")


def _palette(rng):
    bg = 0 if rng.random() < .55 else int(rng.choice([c for c in range(10) if c != HIGHLIGHT]))
    band = int(rng.choice([c for c in range(10) if c not in (bg, HIGHLIGHT)]))
    colors = tuple(map(int, rng.permutation([c for c in range(10) if c not in (bg, band, HIGHLIGHT)])))
    return bg, band, colors


def _diagnostics(inp, completed, bg, band):
    centers, union = _window_masks(completed, bg)
    _, premature = _window_masks(inp, bg)
    common = (inp == bg) & (completed == bg)
    bare_band = np.full_like(completed, bg)
    bare_band[completed == band] = band
    _, bare_union = _window_masks(bare_band, bg)
    return {"highlight": int(union.sum()),
            "residual": int(((completed == bg) & ~union).sum()),
            "union_not_centers": int((union & ~centers).sum()),
            "order_witness": int(((union != premature) & common).sum()),
            "packing_witness": int(((union != bare_union) & (completed == bg)).sum())}


def _finish_candidate(inp, target):
    completed, info = _completed(inp)
    if not np.array_equal(completed, target):
        raise ValueError("construction disagrees with unique exact cover")
    d = _diagnostics(inp, completed, info["background"], info["band"])
    if not all(d[k] for k in ("highlight", "residual", "union_not_centers", "order_witness")):
        raise ValueError("missing causal/negative-space diagnostic")
    out = completed.copy()
    out[_window_masks(completed, info["background"])[1]] = HIGHLIGHT
    return {"input": inp.copy(), "output": out}


def _make_pair(rng, vertical, tag, palette, large=False):
    bg, band, colors = palette
    ntabs = (1 if tag == "edge" else 2 if tag == "bay" else
             int(rng.choice([1, 3])) if tag == "challenge" else
             int(rng.choice([1, 2])) if tag == "practice" else
             int(rng.choice([1, 2, 3], p=[.2, .5, .3])))
    widths = [int(rng.integers(3, 9)) for _ in range(ntabs)]
    if ntabs > 1 and rng.random() < .55:
        widths = [widths[0]]*ntabs
    sides = [1]*ntabs if tag in ("edge", "bay") else list(map(int, rng.choice([-1, 1], ntabs)))
    if tag == "edge":
        counts = [int(rng.integers(2, 4))]
    elif tag == "challenge" and ntabs == 1:
        counts = [4]
    elif tag == "bay":
        counts = [int(rng.integers(2, 4)) for _ in sides]
    elif tag == "practice":
        counts = [int(rng.choice([1, 2, 3], p=[.15, .5, .35])) for _ in sides]
    else:
        counts = [int(rng.choice([1, 2, 3, 4], p=[.12, .48, .30, .10])) for _ in sides]
    if large and tag != "challenge":
        widths, counts = [5]*ntabs, [2]*ntabs
    if sum(counts) > 9:
        raise ValueError("too many pieces")
    gap = int(rng.integers(1, 3)) if tag == "bay" else 1
    minimum_cross = max(16, max(sum(w for w, s in zip(widths, sides) if s == side)
                               +gap*max(0, sides.count(side)-1)+6 for side in (-1, 1)))
    if minimum_cross > 30:
        raise ValueError("tab spans too wide")
    cross = 30 if large or rng.random() < .18 else int(rng.integers(minimum_cross, 31))
    if tag == "bay":
        start = int(rng.integers(3, cross-sum(widths)-gap-1))
        starts = [start, start+widths[0]+gap]
    elif tag == "edge":
        starts = [int(rng.integers(4, cross-widths[0]-1))]
    else:
        starts, reserved = [], {-1: set(), 1: set()}
        for width, side in zip(widths, sides):
            options = [x for x in range(cross-width+1)
                       if not set(range(x-1, x+width+1)) & reserved[side]]
            if not options:
                raise ValueError("no separated tab span")
            x = options[int(rng.integers(len(options)))]
            starts.append(x)
            reserved[side].update(range(x, x+width))
    profiles = [_cuts(rng, w, n) for w, n in zip(widths, counts)]
    edge = {-1: np.zeros(cross, dtype=int), 1: np.zeros(cross, dtype=int)}
    for side in (-1, 1):
        for _ in range(int(rng.integers(2, 7))):
            c = int(rng.integers(cross))
            edge[side][c:min(cross, c+int(rng.integers(1, 6)))] = int(rng.integers(3))
    for s, x, width, cuts in zip(sides, starts, widths, profiles):
        edge[s][x:x+width] = cuts[0]
    if tag == "edge":
        edge[-1][:] = 0
        edge[1][:3] = (2, 0, 2)  # An unused thin recess, unchanged by window union.
    core = int(rng.integers(2, 5))
    extents = {s: max([int(edge[s].max())]+[int(c[-1][0]) for c, t in zip(profiles, sides) if t == s])
               for s in (-1, 1)}
    minimum_h = max(16, core+extents[-1]+extents[1]+(4 if tag == "edge" else 0))
    if minimum_h > 30:
        raise ValueError("tabs too deep")
    h = 30 if large or rng.random() < .18 else int(rng.integers(minimum_h, 31))
    center = 2 if tag == "edge" else extents[-1]+int(rng.integers(h-core-extents[-1]-extents[1]+1))
    base = np.full((h, cross), bg, dtype=np.int8)
    for c in range(cross):
        base[center-edge[-1][c]:center+core+edge[1][c], c] = band
    target, pieces = base.copy(), []
    palette_size = int(rng.integers(1 if rng.random() < .35 else min(3, sum(counts)), min(7, sum(counts))+1))
    color_order = rng.permutation(sum(counts))
    for side, start, width, cuts in zip(sides, starts, widths, profiles):
        for lo, hi in zip(cuts, cuts[1:]):
            cells = np.array([(center-1-u if side == -1 else center+core+u, start+c)
                              for c in range(width) for u in range(int(lo[c]), int(hi[c]))])
            color = colors[int(color_order[len(pieces)]) % palette_size]
            target[cells[:, 0], cells[:, 1]] = color
            pieces.append((cells, color))
    if np.count_nonzero(target == bg) <= target.size // 2:
        raise ValueError("need a majority background")
    inp = _scatter(rng, base, pieces, bg)
    if rng.integers(2):
        inp, target = inp[::-1], target[::-1]
    if rng.integers(2):
        inp, target = inp[:, ::-1], target[:, ::-1]
    if vertical:
        inp, target = inp.T, target.T
    pair = _finish_candidate(inp, target)
    if tag == "edge":
        union = pair["output"] == HIGHLIGHT
        if not np.any(union & ~_literal_windows(target, bg, clipped=False)):
            raise ValueError("need a clipped-boundary witness")
    return pair


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} outside {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    tr, te = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode, phase = int(rng.integers(5)), int(rng.integers(2))
    stable = bool(rng.random() < .22)
    shared = _palette(rng)
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", tr), ("test", te)):
        for index in range(count):
            if mode < 2:
                vertical = bool(mode)
            elif mode < 4:
                vertical = (mode == 2) == (split == "test")
            else:
                vertical = bool((index+phase) % 2) if split == "train" else bool(rng.integers(2))
            if split == "train":
                tag = "edge" if index == 0 else "bay" if index == 1 else "practice"
            else:
                tag = "free" if mode in (2, 3) else "challenge"
            palette = shared if stable else _palette(rng)
            for attempt in range(96):
                try:
                    pair = _make_pair(rng, vertical, tag, palette, large=attempt >= 64)
                except ValueError:
                    continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    return result


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    completed, info = _completed(inp)
    bg, band = info["background"], info["band"]
    centers, union = _window_masks(completed, bg)
    assert np.array_equal(union, _literal_windows(completed, bg))
    expected = completed.copy()
    expected[union] = HIGHLIGHT
    assert np.array_equal(out, expected)
    assert np.array_equal(inp == band, out == band)
    assert np.all(out[completed != bg] == completed[completed != bg])
    assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                          np.bincount(completed.ravel(), minlength=10))
    canonical = completed if info["axis"] == 0 else completed.T
    canonical_out = out if info["axis"] == 0 else out.T
    used = set()
    boxes = []
    for candidate in info["tabs"]:
        sign, start, width, root, end = candidate[3]
        lo, hi = (min(root), end) if sign == 1 else (1-end, 1-min(root))
        boxes.append((sign, start, width, lo, hi))
        assert np.all(canonical[lo:hi, start:start+width] != bg)
        for i, cells in candidate[2]:
            assert i not in used
            used.add(i)
            source, color = info["pieces"][i]
            delta = cells.min(axis=0)-source.min(axis=0)
            assert np.any(delta)
            assert set(map(tuple, source+delta)) == set(map(tuple, cells))
            assert np.all(canonical[cells[:, 0], cells[:, 1]] == color)
    assert used == set(range(len(info["pieces"])))
    diagnostics = _diagnostics(inp, completed, bg, band)
    assert all(diagnostics[k] for k in ("highlight", "residual", "union_not_centers", "order_witness"))
    narrow_bay = 0
    for i, left in enumerate(boxes):
        for right in boxes[i+1:]:
            if left[0] != right[0]:
                continue
            first, second = sorted((left, right), key=lambda b: b[1])
            gap_start, gap_end = first[1]+first[2], second[1]
            if 1 <= gap_end-gap_start <= 2:
                lo, hi = max(first[3], second[3]), min(first[4], second[4])
                narrow_bay += int(np.sum(canonical_out[lo:hi, gap_start:gap_end] == bg))
    diagnostics["narrow_bay_witness"] = narrow_bay
    diagnostics["clipping_witness"] = int((union & ~_literal_windows(completed, bg, False)).sum())
    diagnostics.update({"shape": tuple(inp.shape), "axis": info["axis"],
                        "roles": (bg, band), "tabs": len(info["tabs"]),
                        "widths": tuple(t[3][2] for t in info["tabs"]),
                        "counts": tuple(len(t[2]) for t in info["tabs"]),
                        "sides": tuple(t[3][0] for t in info["tabs"]),
                        "piece_count": len(info["pieces"]),
                        "color_reuse": len({c for _, c in info["pieces"]}) < len(info["pieces"]),
                        "local_starts": info["local_starts"]})
    return diagnostics


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            features = _validate_pair(pair)
            inp = np.asarray(pair["input"], dtype=np.int8)
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            if split == "train":
                training.append(features)
    assert any(f["clipping_witness"] for f in training)
    assert any(f["narrow_bay_witness"] for f in training)
    assert any(f["packing_witness"] for f in training)
    assert any(f["tabs"] >= 2 for f in training)
    assert any(max(f["counts"]) >= 2 for f in training)


def _signature(puzzle):
    digest = hashlib.sha256()
    for s in ("train", "test"):
        for pair in puzzle[s]:
            for k in ("input", "output"):
                a = np.asarray(pair[k], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded in-memory suite: 42 specified seeds, no files or execution traces."""
    # The center-only rule marks one cell; the specified union marks all nine.
    fixture = np.ones((7, 7), dtype=np.int8)
    fixture[2:5, 2:5] = 0
    centers, union = _window_masks(fixture, 0)
    assert centers.sum() == 1 and union.sum() == 9
    # Two-row boundary strip qualifies ONLY through clipped windows.
    fixture[:] = 1
    fixture[:2] = 0
    assert _window_masks(fixture, 0)[1].sum() == 14
    assert not _literal_windows(fixture, 0, False).any()
    # Interior two-column slit and a diagonal pinched cell do not qualify.
    fixture[:] = 1
    fixture[1:6, 2:4] = 0
    assert not _window_masks(fixture, 0)[1].any()
    fixture[:] = 1
    fixture[2:5, 2:5] = 0
    fixture[2, 2] = 1
    assert not _window_masks(fixture, 0)[0][3, 3]
    assert not _window_masks(fixture, 0)[1].any()
    fixture[:] = 1
    fixture[:2, :2] = 0
    assert _window_masks(fixture, 0)[1].sum() == 4
    # Global uniqueness counts colored outputs, including repeated colors.
    roots = np.zeros((16, 18), dtype=np.int8)
    roots[4:7] = 8
    roots[7, [3, 10]] = 8
    cap = np.array([(r, c) for c, lo in enumerate((0, 1, 0)) for r in range(lo, 3)])
    roots[cap[:, 0], cap[:, 1]] = 2
    assert len(_packings(roots)[0]) == 2
    roots[cap[:, 0], cap[:, 1]+7] = 4
    assert len(_packings(roots)[0]) == 2
    roots[roots == 4] = 2
    assert len(_packings(roots)[0]) == 1
    shapes, structures, signatures, features, modes = set(), set(), set(), [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        split_axes = []
        for s in ("train", "test"):
            axes = set()
            for pair in puzzle[s]:
                f = _validate_pair(pair)
                features.append(f)
                shapes.add(f["shape"])
                axes.add(f["axis"])
                roles = {}
                a = pair["input"]
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, hashlib.sha256(norm).digest()))
                original = a.copy()
                assert np.array_equal(solve(a), pair["output"])
                assert np.array_equal(a, original)
            split_axes.append(tuple(sorted(axes)))
        modes.add(tuple(split_axes))
    assert len(signatures) == 42 and len(structures) >= 205
    assert len(shapes) >= 35 and (30, 30) in shapes
    assert {f["axis"] for f in features} == {0, 1}
    assert len(modes) >= 4
    assert {f["tabs"] for f in features} >= {1, 2, 3}
    assert {c for f in features for c in f["counts"]} >= {1, 2, 3, 4}
    assert len({f["roles"] for f in features}) >= 20
    assert sum(f["color_reuse"] for f in features) >= 20
    assert sum(f["residual"] > 0 for f in features) == 210
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1000+10*tr+te, tr, te)
            assert len(p["train"]) == tr and len(p["test"]) == te
            validate_generated_puzzle(p)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Output mutation must be rejected even if schema/colors remain valid.
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"] = p["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("mislabeled output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
