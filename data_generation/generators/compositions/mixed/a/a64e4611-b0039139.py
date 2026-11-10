"""Clipped-clear-center stencil stamping, followed by component-count repetition.

Only NumPy and the standard library are used. Importing this module does no I/O.
The generator does not import, execute, or chain either parent generator.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("a grid must be an integer 2D array, dimensions 1..30, colors 0..9")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _components(mask, diagonal=False):
    """Return components in deterministic scan order."""
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        group, todo = [], [(r, c)]
        seen[r, c] = True
        while todo:
            y, x = todo.pop()
            group.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        groups.append(group)
    return groups


def _crop(mask):
    coords = np.argwhere(mask)
    if not len(coords):
        raise ValueError("the stencil must contain foreground")
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy(), tuple(map(int, lo))


def _clear_centers(foreground):
    """All centers simultaneously; off-grid positions contain no obstacle."""
    h, w = foreground.shape
    padded = np.pad(foreground, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr+h, dc:dc+w]
    return ~near


def _decode(input_grid):
    """Find one valid ordered four-panel parse using input alone.

    Canonical coordinates have the panels from left to right; vertical inputs
    are transposed only for calculation, then transposed back on output.
    """
    grid = _grid(input_grid)
    candidates = []
    for vertical in (False, True):
        g = grid.T if vertical else grid
        for divider in map(int, np.unique(g)):
            cuts = np.flatnonzero(np.all(g == divider, axis=0))
            if len(cuts) != 3 or int(np.sum(g == divider)) != 3*g.shape[0]:
                continue
            a, b, c = map(int, cuts)
            if min(a, b-a-1, c-b-1, g.shape[1]-c-1) < 1:
                continue
            s, q, p, z = g[:, :a], g[:, a+1:b], g[:, b+1:c], g[:, c+1:]
            primary, secondary = int(p[0, 0]), int(z[0, 0])
            if primary == secondary or not np.all(p == primary) or not np.all(z == secondary):
                continue
            sc, qc = set(map(int, np.unique(s))), set(map(int, np.unique(q)))
            if len(sc) != 2 or len(qc) != 2 or len(sc & qc) != 1:
                continue
            bg = next(iter(sc & qc))
            ink, counter_ink = next(iter(sc - {bg})), next(iter(qc - {bg}))
            if len({bg, divider, ink, counter_ink}) != 4:
                continue
            foreground, (top, left) = _crop(s == ink)
            counter = q == counter_ink
            count = len(_components(counter))
            candidates.append({
                "vertical": vertical, "foreground": foreground, "counter": counter,
                "count": count, "primary": primary, "secondary": secondary,
                "palette": (bg, divider, ink, counter_ink),
                "lengths": (s.shape[1], q.shape[1], p.shape[1], z.shape[1]),
                "margins": (top, s.shape[0]-top-foreground.shape[0],
                            left, s.shape[1]-left-foreground.shape[1]),
            })
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous four-panel parse")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the ordered panels and the original cropped stencil."""
    d = _decode(input_grid)
    foreground, n = d["foreground"], d["count"]
    mask = foreground | _clear_centers(foreground)
    h, a = mask.shape
    length = n*(a+1)-1
    if length > 30:
        raise ValueError("repeated output exceeds ARC bounds")
    out = np.full((h, length), d["secondary"], dtype=np.int8)
    # Modular placement differs from the concatenation used by generation.
    for x in range(length):
        offset = x % (a+1)
        if offset < a:
            out[mask[:, offset], x] = d["primary"]
    return out.T.copy() if d["vertical"] else out


def _count_arg(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _partition(length, pieces, rng):
    sizes = np.ones(pieces, dtype=int)
    gaps = np.ones(pieces-1, dtype=int)
    for _ in range(length - (2*pieces-1)):
        k = int(rng.integers(2*pieces-1))
        if k < pieces:
            sizes[k] += 1
        else:
            gaps[k-pieces] += 1
    spans, start = [], 0
    for i, size in enumerate(sizes):
        spans.append((start, start+int(size)))
        start += int(size) + (int(gaps[i]) if i < pieces-1 else 0)
    return spans


def _grow(h, w, area, rng):
    result = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        result[r, c] = True
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not result[y, x]:
                frontier.add((y, x))
    return result


def _counter(h, w, count, rng, area=None):
    """Connected shapes in separated boxes: four/eight counts always agree."""
    layouts = [(nr, nc) for nr in range(1, min(count, (h+1)//2)+1)
               for nc in range(1, min(count, (w+1)//2)+1)
               if count <= nr*nc <= count+2]
    if not layouts:
        raise ValueError("counter does not fit")
    for attempt in range(20):
        nr, nc = layouts[int(rng.integers(len(layouts)))]
        if attempt == 19 and area is not None:
            # Training uses h >= 13, w >= 5 and count <= 3, so this
            # balanced row layout guarantees at least 55 available cells.
            nr, nc = count, 1
            available = h-count+1
            sizes = [available//count + (i < available % count) for i in range(count)]
            rows, offset = [], 0
            for size in sizes:
                rows.append((offset, offset+size))
                offset += size+1
            cols = [(0, w)]
        else:
            rows, cols = _partition(h, nr, rng), _partition(w, nc, rng)
        boxes = [(r0, r1, c0, c1) for r0, r1 in rows for c0, c1 in cols]
        rng.shuffle(boxes)
        if area is not None:
            boxes.sort(key=lambda b: (b[1]-b[0])*(b[3]-b[2]), reverse=True)
        boxes = boxes[:count]
        capacities = [(r1-r0)*(c1-c0) for r0, r1, c0, c1 in boxes]
        if area is None or sum(capacities) >= area:
            break
    _require(area is None or sum(capacities) >= area, "counter ink area does not fit")
    areas = [1]*count
    if area is not None:
        for _ in range(area-count):
            choices = [i for i in range(count) if areas[i] < capacities[i]]
            k = choices[int(rng.integers(len(choices)))]
            areas[k] += 1
    result = np.zeros((h, w), dtype=bool)
    singletons = area is None and rng.random() < .18
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        bh, bw = r1-r0, c1-c0
        if area is not None:
            obj = _grow(bh, bw, areas[i], rng)
        elif singletons:
            obj = np.ones((1, 1), dtype=bool)
        else:
            oh, ow = int(rng.integers(1, bh+1)), int(rng.integers(1, bw+1))
            mode = int(rng.integers(4))
            if mode == 0:
                obj = np.ones((oh, ow), dtype=bool)
            elif mode == 1:
                obj = _grow(oh, ow, int(rng.integers(1, oh*ow+1)), rng)
            elif mode == 2:
                obj = np.zeros((oh, ow), dtype=bool)
                obj[:, 0] = obj[-1, :] = True
            else:
                obj = np.ones((oh, ow), dtype=bool)
                if min(oh, ow) >= 3:
                    obj[1:-1, 1:-1] = False
        oh, ow = obj.shape
        rr, cc = r0+int(rng.integers(bh-oh+1)), c0+int(rng.integers(bw-ow+1))
        result[rr:rr+oh, cc:cc+ow] = obj
    if result.all():
        # Only possible for a single component; retain visible background.
        result[0, 0] = False
    _require(len(_components(result)) == count == len(_components(result, True)),
             "counter connectivity error")
    return result


def _stencil(h, w, rng, diagnostic=False):
    """Variable negative space, bounded by the original ink's exact crop."""
    for attempt in range(40):
        mode = int(rng.integers(4))
        if mode == 0:
            fg = rng.random((h, w)) < rng.uniform(.15, .55)
        elif mode == 1:
            fg = np.zeros((h, w), dtype=bool)
            # Open walls and branching barriers; no opposite-corner path.
            for _ in range(int(rng.integers(2, 6))):
                if rng.random() < .5:
                    r = int(rng.integers(h))
                    lo, hi = sorted(map(int, rng.choice(w+1, 2, replace=False)))
                    fg[r, lo:hi] = True
                else:
                    c = int(rng.integers(w))
                    lo, hi = sorted(map(int, rng.choice(h+1, 2, replace=False)))
                    fg[lo:hi, c] = True
        elif mode == 2:
            fg = np.ones((h, w), dtype=bool)
        else:
            fg = np.zeros((h, w), dtype=bool)
            fg[0, :] = fg[:, 0] = True
            if rng.random() < .5:
                fg[-1, :] = True
            if rng.random() < .5:
                fg[:, -1] = True
            for _ in range(int(rng.integers(0, 5))):
                fg[int(rng.integers(h)), int(rng.integers(w))] = True
        # Blank pockets, steps and edge-opening bays vary in size and number.
        for _ in range(int(rng.integers(1, 5))):
            ph, pw = int(rng.integers(3, min(h, 9)+1)), int(rng.integers(2, w+1))
            r, c = int(rng.integers(h-ph+1)), int(rng.integers(w-pw+1))
            fg[r:r+ph, c:c+pw] = False
        # Tightness uses independently placed side anchors, not corner paths.
        fg[0, int(rng.integers(w))] = True
        fg[-1, int(rng.integers(w))] = True
        fg[int(rng.integers(h)), 0] = True
        fg[int(rng.integers(h)), -1] = True
        if diagnostic:
            # Isolated 3x3 cavity: ONLY its center qualifies, with no size cutoff.
            col = int(rng.integers(w-4))
            fg[1:6, col:col+5] = True
            fg[2:5, col+1:col+4] = False
            # Clipped boundary: two clear centers touch the left crop edge.
            fg[7:13, :3] = True
            fg[8:12, :2] = False
            # A diagonal obstacle rejects (9,4) despite clear axial neighbors.
            fg[8:11, 3:6] = False
            fg[8, 3] = True
        if rng.random() < .5:
            fg = fg[::-1].copy()
        if rng.random() < .5:
            fg = fg[:, ::-1].copy()
        crop, _ = _crop(fg)
        centers = _clear_centers(fg)
        marked = fg | centers
        if (crop.shape == (h, w) and np.any(centers) and np.any(~marked)
                and (not diagnostic or (int(centers.sum()) >= 3 and len(_components(fg)) > 1
                     and all(not np.array_equal(marked, t) for t in
                             (marked[::-1], marked[:, ::-1], marked[::-1, ::-1]))))):
            return fg
    # This fallback is only for ordinary geometry draws. It remains part of
    # the same local rule and does not change any rule or output convention.
    if diagnostic:
        raise RuntimeError("diagnostic geometry retries exhausted")
    fg = np.zeros((h, w), dtype=bool)
    fg[0, :] = fg[:, 0] = True
    return fg


def _render_pair(rng, fg, count, vertical, palette, swatches, area=None, witness=False):
    bg, divider, ink, counter_ink = palette
    h, a = fg.shape
    canvas_h = min(30, max(h+int(rng.integers(0, 5)), 2*count-1))
    pad = int(rng.integers(1 if witness else 0, 4))
    stencil_w = a+pad
    counter_min = 5 if witness else 3
    counter_max = min(9, 30-3-stencil_w-2)
    counter_w = int(rng.integers(counter_min, counter_max+1))
    budget = 30-3-stencil_w-counter_w
    total = int(rng.integers(2, budget+1))
    pwidth = int(rng.integers(1, total))
    zwidth = total-pwidth
    if witness:
        choices = [(p, z) for p in range(1, budget)
                   for z in range(1, budget-p+1) if p != count and z != count]
        pwidth, zwidth = choices[int(rng.integers(len(choices)))]
    top, left = int(rng.integers(canvas_h-h+1)), int(rng.integers(pad+1))
    stencil = np.full((canvas_h, stencil_w), bg, dtype=np.int8)
    stencil[top:top+h, left:left+a][fg] = ink
    counter = _counter(canvas_h, counter_w, count, rng, area)
    primary, secondary = swatches
    panels = [stencil, np.where(counter, counter_ink, bg).astype(np.int8),
              np.full((canvas_h, pwidth), primary, dtype=np.int8),
              np.full((canvas_h, zwidth), secondary, dtype=np.int8)]
    line = np.full((canvas_h, 1), divider, dtype=np.int8)
    inp = np.concatenate((panels[0], line, panels[1], line, panels[2], line, panels[3]), axis=1)
    # Generation uses direct per-center slice tests, not solve's padded test.
    mask = fg.copy()
    for r, c in np.argwhere(~fg):
        if not fg[max(0, r-1):r+2, max(0, c-1):c+2].any():
            mask[r, c] = True
    tile = np.where(mask, primary, secondary).astype(np.int8)
    separator = np.full((h, 1), secondary, dtype=np.int8)
    pieces = []
    for i in range(count):
        if i:
            pieces.append(separator)
        pieces.append(tile)
    out = np.concatenate(pieces, axis=1)
    return {"input": inp.T.copy() if vertical else inp,
            "output": out.T.copy() if vertical else out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """One seed controls all geometry, counters, layouts and visible palettes."""
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt = _count_arg(num_train, "num_train", 2, 10)
    nq = _count_arg(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palettes = [tuple(map(int, rng.choice(10, 4, replace=False)))]
    shift = int(rng.integers(1, 10))
    palettes.append(tuple((c+shift) % 10 for c in palettes[0]))
    swatches = [tuple(map(int, rng.choice([c for c in range(10) if c != p[1]], 2, replace=False)))
                for p in palettes]
    first_vertical = bool(rng.integers(2))
    counts = list(map(int, rng.permutation([2, 3])))
    area = int(rng.integers(12, 21))
    puzzle, used = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for index in range(amount):
            witness = split == "train" and index < 2
            for attempt in range(24):
                if witness:
                    n = counts[index]
                    h, a = int(rng.integers(13, 17)), int(rng.integers(6, 9))
                    vertical = first_vertical if index == 0 else not first_vertical
                    palette_index = index
                else:
                    n = int(rng.integers(1, 5 if split == "train" else 7))
                    a = int(rng.integers(4, min(14, 31//n-1)+1))
                    h = (int(rng.integers(6, 17)) if split == "train"
                         else int(rng.integers(18, 31)))
                    vertical, palette_index = bool(rng.integers(2)), int(rng.integers(2))
                fg = _stencil(h, a, rng, diagnostic=witness and index == 0)
                palette = palettes[palette_index]
                pair_swatches = swatches[palette_index]
                if not witness and rng.random() < .5:
                    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
                    pair_swatches = tuple(map(int, rng.choice(
                        [c for c in range(10) if c != palette[1]], 2, replace=False)))
                pair = _render_pair(rng, fg, n, vertical, palette, pair_swatches,
                                    area=area if witness else None, witness=witness)
                if _key(pair["input"]) in used:
                    continue
                _decode(pair["input"])
                used.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("unique pair retries exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _diagnostics(fg):
    clear = _clear_centers(fg)
    orthogonal_near = fg.copy()
    orthogonal_near[1:] |= fg[:-1]
    orthogonal_near[:-1] |= fg[1:]
    orthogonal_near[:, 1:] |= fg[:, :-1]
    orthogonal_near[:, :-1] |= fg[:, 1:]
    edge = np.zeros(fg.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return {"centers": int(clear.sum()), "edge_centers": int(np.sum(clear & edge)),
            "isolated_center": any(len(g) == 1 for g in _components(clear)),
            "diagonal_block": bool(np.any(~orthogonal_near & ~clear)),
            "fringe": int(np.sum(~(fg | clear)))}


def validate_generated_puzzle(puzzle):
    """Check labels, panel evidence, and the two-parent necessity witnesses.

    This checks generated teaching puzzles. solve also accepts valid individual
    family inputs without the generator's teaching/shift restrictions.
    """
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split types")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate inputs")
            seen.add(_key(inp))
            d = _decode(inp)
            fg, n = d["foreground"], d["count"]
            _require(n == len(_components(d["counter"], True)), "ambiguous counter connectivity")
            diag = _diagnostics(fg)
            _require(diag["centers"] > 0 and diag["fringe"] > 0, "empty local-rule evidence")
            expected = solve(inp)
            _require(np.array_equal(out, expected), "wrong output")
            canonical = out.T if d["vertical"] else out
            h, a = fg.shape
            _require(canonical.shape == (h, n*(a+1)-1), "copy count or trailing separator")
            tile = np.where(fg | _clear_centers(fg), d["primary"], d["secondary"])
            for i in range(n):
                _require(np.array_equal(canonical[:, i*(a+1):i*(a+1)+a], tile), "tile altered")
                if i < n-1:
                    _require(np.all(canonical[:, i*(a+1)+a] == d["secondary"]), "separator ink")
            # A omission changes actual tile cells; B omission changes its extent.
            _require(np.any(tile != np.where(fg, d["primary"], d["secondary"])), "A is inert")
            d["diagnostics"] = diag
            (training if split == "train" else tests).append(d)
    _require({d["vertical"] for d in training} == {False, True}, "axis training witness")
    for role in range(4):
        _require(len({d["palette"][role] for d in training}) >= 2, "fixed structural color role")
    _require(len({d["count"] for d in training}) >= 2, "count variation missing")
    _require(any(d["count"] > 1 for d in training), "B is inert")
    _require(any(a["count"] != b["count"] and a["counter"].sum() == b["counter"].sum()
                 for a in training for b in training), "ink area can substitute for count")
    _require(any(d["diagnostics"]["centers"] >= 3 and d["diagnostics"]["edge_centers"]
                 and d["diagnostics"]["isolated_center"] and d["diagnostics"]["diagonal_block"]
                 and len(_components(d["foreground"])) > 1
                 for d in training), "missing local predicate diagnostics")
    _require(any(sum(d["margins"]) > 0 and d["count"] > 1 for d in training), "crop witness")
    for panel in (2, 3):
        _require(any(d["lengths"][panel] != d["count"] for d in training), "swatch-width shortcut")
    for reflected in (lambda x: x[::-1], lambda x: x[:, ::-1], lambda x: x[::-1, ::-1]):
        _require(any(not np.array_equal(d["foreground"] | _clear_centers(d["foreground"]),
                                       reflected(d["foreground"] | _clear_centers(d["foreground"])))
                     for d in training), "orientation not exposed")
    _require(min(d["foreground"].shape[0] for d in tests)
             > max(d["foreground"].shape[0] for d in training), "missing cross-axis size shift")


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(p[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, streaming 42-seed tests; returns compact aggregate measurements."""
    hashes, shapes, masks, counts, palettes, axes = set(), set(), set(), set(), set(), set()
    edge = isolated = diagonal = disconnected = one = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_puzzle_hash(puzzle))
        if seed in (0, 7, 41):
            _require(_puzzle_hash(puzzle) == _puzzle_hash(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                d = _decode(pair["input"])
                fg = d["foreground"]
                shapes.add(pair["input"].shape)
                masks.add(_key(fg))
                counts.add(d["count"])
                palettes.add(d["palette"])
                axes.add(d["vertical"])
                diag = _diagnostics(fg)
                edge += diag["edge_centers"] > 0
                isolated += diag["isolated_center"]
                diagonal += diag["diagonal_block"]
                disconnected += len(_components(fg)) > 1
                one += d["count"] == 1
    _require(len(hashes) == 42 and len(masks) >= 150, "structure diversity")
    _require(len(shapes) >= 50 and len(palettes) >= 40, "layout/palette diversity")
    _require(counts == set(range(1, 7)) and axes == {False, True}, "count/axis diversity")
    _require(edge >= 42 and isolated >= 42 and diagonal >= 42 and disconnected > 10 and one > 0,
             "local/counter feature diversity")
    # Every legal explicit combination, including smallest teaching sets.
    explicit = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(314159+nt*10+nq, nt, nq)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "counts ignored")
            explicit += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exact tiny predicate: one interior center, diagonal veto, clipped edge.
    ring = np.ones((5, 5), dtype=bool)
    ring[1:4, 1:4] = False
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    _require(np.array_equal(_clear_centers(ring), expected), "whole-window or size-cutoff error")
    sparse = np.zeros((4, 4), dtype=bool)
    sparse[0, 0] = True
    _require(not _clear_centers(sparse)[1, 1] and _clear_centers(sparse)[3, 3], "diagonal/clipping error")
    # The solver supports legal inputs beyond the sampler's dimensions/counts.
    for vertical in (False, True):
        fg = np.zeros((7, 2), dtype=bool)
        fg[0, :] = fg[-1, 0] = True
        pair = _render_pair(np.random.default_rng(888), fg, 8, vertical,
                            (0, 1, 2, 3), (4, 5))
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "narrow-tile support")
        solid = np.ones((2, 3), dtype=bool)
        pair = _render_pair(np.random.default_rng(889), solid, 1, vertical,
                            (0, 1, 2, 3), (4, 5), witness=True)
        _require(np.array_equal(solve(pair["input"]), pair["output"])
                 and pair["output"].size == solid.size, "inactive predicate/count-one support")
    damaged = generate_puzzle(0)
    damaged["test"][0]["output"].flat[0] = (int(damaged["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(hashes),
            "input_shapes": len(shapes), "stencil_structures": len(masks),
            "structural_palettes": len(palettes), "counts": sorted(counts),
            "edge_center_pairs": edge, "isolated_center_pairs": isolated,
            "diagonal_veto_pairs": diagonal, "fragmented_stencil_pairs": disconnected,
            "one_copy_pairs": one, "legal_count_combinations": explicit}
