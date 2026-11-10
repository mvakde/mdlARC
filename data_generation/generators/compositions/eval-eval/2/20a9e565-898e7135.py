"""Growing moulds: extrapolate negative space, select material, enlarge the crop.

Only NumPy and the standard library are used. Generation paints latent geometry;
the solver independently segments visible cavities and fits terminal extrusion.
No parent module, seed metadata, output dimensions or stored example is consulted.
"""

from collections import Counter
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _N4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normalise(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _size(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _orientations(shape, reflection=True):
    result = set()
    for flip in range(2 if reflection else 1):
        a = [(r, -c if flip else c) for r, c in shape]
        for _ in range(4):
            result.add(_normalise(a))
            a = [(c, -r) for r, c in a]
    return tuple(sorted(result))


def _key(shape):
    return _orientations(_normalise(shape))[0]


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _checked_grid(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, sides 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _marker_cells(bounds):
    r0, r1, c0, c1 = bounds  # exclusive upper bounds
    return {(r0, c0), (r0, c0+1), (r0+1, c0),
            (r1-1, c1-1), (r1-1, c1-2), (r1-2, c1-1)}


def _find_markers(a):
    found = []
    for raw in np.unique(a):
        cells = tuple(map(tuple, np.argwhere(a == raw)))
        if len(cells) != 6:
            continue
        bounds = _bbox(cells)
        if min(bounds[1]-bounds[0], bounds[3]-bounds[2]) < 2:
            continue
        if set(cells) == _marker_cells(bounds) and len(_components(a == raw)) == 2:
            found.append((int(raw), bounds))
    if len(found) != 1:
        raise ValueError("one pair of disconnected canonical L markers is required")
    return found[0]


def _scene(a):
    marker, bounds = _find_markers(a)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= a.size // 2 or bg == marker:
        raise ValueError("the canvas background must occupy more than half the input")
    q0, q1, p0, p1 = bounds
    if not np.all((a[q0:q1, p0:p1] == bg) | (a[q0:q1, p0:p1] == marker)):
        raise ValueError("the whole marked window is erased")
    boards = []
    for structural in np.unique(a):
        if structural in (bg, marker):
            continue
        for cells in _components(a == structural):
            r0, r1, c0, c1 = _bbox(cells)
            if not (r0 < q0 < q1 < r1 and c0 < p0 < p1 < c1):
                continue
            board = a[r0:r1, c0:c1]
            if (np.all(board[0] == structural) and np.all(board[-1] == structural)
                    and np.all(board[:, 0] == structural) and np.all(board[:, -1] == structural)
                    and np.all(np.isin(board, (bg, marker, structural)))):
                boards.append((int(structural), (r0, r1, c0, c1), board.copy()))
    if len(boards) != 1:
        raise ValueError("one intact plate border containing the window is required")
    structural, bb, board = boards[0]
    r0, r1, c0, c1 = bb
    qb = q0-r0, q1-r0, p0-c0, p1-c0
    visible_mask = board == bg
    visible_mask[qb[0]:qb[1], qb[2]:qb[3]] = False
    cavities = _components(visible_mask)
    if not 3 <= len(cavities) <= 6:
        raise ValueError("three to six complete visible moulds are required")
    outside = np.ones(a.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    exterior = [(int(color), cells) for color in np.unique(a) if color != bg
                for cells in _components((a == color) & outside)]
    if any(color in (structural, marker) for color, cells in exterior):
        raise ValueError("plate and annotation colors cannot be loose material")
    if len(_components((a != bg) & outside)) != len(exterior):
        raise ValueError("loose objects must be separated, even when colors differ")
    large = [(color, cells) for color, cells in exterior if len(cells) >= 4]
    if not 2 <= len(large) <= 6:
        raise ValueError("two to six block-scaled inventory objects are required")
    return bg, marker, structural, bb, board, qb, cavities, large, exterior


def _linear(values):
    delta = values[1] - values[0]
    return delta if all(x == values[0]+i*delta for i, x in enumerate(values)) else None


def _extrude(shape, height, tail_col):
    """Fit-side continuation: extend or shorten only singleton terminal rows."""
    h, _ = _size(shape)
    if height < 2:
        return None
    if height >= h:
        return tuple(sorted(set(shape) | {(r, tail_col) for r in range(h, height)}))
    if any(c != tail_col for r, c in shape if r >= height):
        return None
    result = tuple((r, c) for r, c in shape if r < height)
    if not result or (height-1, tail_col) not in result:
        return None
    return result


def _transform(a, trans, fr, fc):
    b = a.T if trans else a
    return b[::-1 if fr else 1, ::-1 if fc else 1].copy()


def _untransform(a, trans, fr, fc):
    b = a[::-1 if fr else 1, ::-1 if fc else 1]
    return b.T.copy() if trans else b.copy()


def _geometric_models(board, qb, bg):
    """Find an axis and an observed, arithmetic one-arm extrusion sequence."""
    query = np.zeros(board.shape, dtype=bool)
    query[qb[0]:qb[1], qb[2]:qb[3]] = True
    observed = (board == bg) & ~query
    for trans in (False, True):
        for fr in (False, True):
            for fc in (False, True):
                mask = _transform(observed, trans, fr, fc)
                target = _transform(query, trans, fr, fc)
                q = _bbox(tuple(map(tuple, np.argwhere(target))))
                objects = sorted(_components(mask), key=lambda s: _bbox(s)[2])
                boxes = [_bbox(s) for s in objects]
                starts = [b[2] for b in boxes]
                tops = [b[0] for b in boxes]
                heights = [b[1]-b[0] for b in boxes]
                widths = [b[3]-b[2] for b in boxes]
                if len(set(widths)) != 1 or any(b[3] > q[2] for b in boxes):
                    continue
                step, drift, dh = _linear(starts), _linear(tops), _linear(heights)
                w = widths[0]
                if step is None or drift is None or dh is None or step <= w:
                    continue
                shapes = [_normalise(s) for s in objects]
                shortest = shapes[int(np.argmin(heights))]
                hmin, _ = _size(shortest)
                terminal = [c for r, c in shortest if r == hmin-1]
                if len(terminal) != 1:
                    continue
                tail_col = terminal[0]
                if any(_extrude(shortest, h, tail_col) != s for h, s in zip(heights, shapes)):
                    continue
                predicted = []
                static = np.zeros((q[1]-q[0], q[3]-q[2]), dtype=bool)
                for i in range(len(objects), 30):
                    x = starts[0] + i*step
                    if x >= q[3]:
                        break
                    y = tops[0] + i*drift
                    h = heights[0] + i*dh
                    shape = _extrude(shortest, h, tail_col)
                    if shape is None:
                        predicted = []
                        break
                    cells = [(r+y, c+x) for r, c in shape]
                    clipped = [(r-q[0], c-q[2]) for r, c in cells
                               if q[0] <= r < q[1] and q[2] <= c < q[3]]
                    if clipped:
                        predicted.append((i, shape, tuple(clipped)))
                    for r, c in shapes[-1]:
                        rr, cc = r+y-q[0], c+x-q[2]
                        if 0 <= rr < static.shape[0] and 0 <= cc < static.shape[1]:
                            static[rr, cc] = True
                if not predicted:
                    continue
                yield dict(trans=trans, fr=fr, fc=fc, q=q, targets=predicted,
                           dh=dh, drift=drift, gap=step-w, visible=len(objects),
                           shapes=tuple(shapes), static=static,
                           direction=("up" if fc else "down") if trans else ("left" if fc else "right"))


def _decode(cells, k):
    r0, r1, c0, c1 = _bbox(cells)
    if (r1-r0) % k or (c1-c0) % k or len(cells) % (k*k):
        return None
    counts = Counter(((r-r0)//k, (c-c0)//k) for r, c in cells)
    if any(n != k*k for n in counts.values()):
        return None
    return tuple(sorted(counts))


def _recover(input_grid):
    a = _checked_grid(input_grid)
    bg, marker, structure, bb, board, qb, cavities, loose, exterior = _scene(a)
    models = list(_geometric_models(board, qb, bg))
    if not models:
        raise ValueError("visible cavities do not establish one-arm arithmetic continuation")
    answers = []
    for k in range(2, 6):
        if max(qb[1]-qb[0], qb[3]-qb[2])*k > 30:
            continue
        inventory = {}
        for color, cells in loose:
            shape = _decode(cells, k)
            if shape is None:
                break
            inventory.setdefault(_key(shape), []).append((color, shape))
        else:
            # Same-class duplicates are interchangeable only with the same color.
            if any(len({color for color, s in group}) != 1 for group in inventory.values()):
                raise ValueError("congruent differently colored inventory pieces are ambiguous")
            for model in models:
                if any(_key(shape) not in inventory for i, shape, cells in model["targets"]):
                    continue
                q = model["q"]
                logical = np.full((q[1]-q[0], q[3]-q[2]), structure, dtype=np.int8)
                holemask = np.zeros(logical.shape, dtype=bool)
                reflection = rotation = 0
                target_keys = set()
                partial = False
                for i, shape, cells in model["targets"]:
                    key = _key(shape)
                    target_keys.add(key)
                    partial |= len(cells) < len(shape)
                    color, piece = inventory[key][0]
                    reflection += shape not in _orientations(piece, False)
                    rotation += shape != piece and shape in _orientations(piece, False)
                    for r, c in cells:
                        logical[r, c] = color
                        holemask[r, c] = True
                oriented = _untransform(logical, model["trans"], model["fr"], model["fc"])
                answer = np.repeat(np.repeat(oriented, k, 0), k, 1)
                equal_area_decoy = any(key not in target_keys and len(key) == len(t)
                                       and group[0][0] != inventory[t][0][0]
                                       for key, group in inventory.items() for t in target_keys)
                info = dict(background=bg, marker=marker, structural=structure,
                            scale=k, board_shape=board.shape, bounds=qb,
                            growth=model["dh"], drift=model["drift"], gap=model["gap"],
                            direction=model["direction"], visible=model["visible"],
                            target_count=len(model["targets"]),
                            partial=partial,
                            skipped=model["targets"][0][0]-model["visible"],
                            reflection=reflection, rotation=rotation,
                            a_witness=not np.array_equal(holemask, model["static"]),
                            equal_area_decoy=equal_area_decoy,
                            unused_classes=len(set(inventory)-target_keys),
                            classes=tuple(sorted(inventory)),
                            noise_sizes=tuple(len(c) for color, c in exterior if len(c) < 4))
                answers.append((answer, info))
    if not answers:
        raise ValueError("no common scale and inventory assignment can fill the predicted moulds")
    if any(not np.array_equal(answers[0][0], out) for out, info in answers[1:]):
        raise ValueError("competing interpretations predict different marked crops")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover everything from this input alone, without training or latent state."""
    return _recover(input_grid)[0]


def _connected(cells):
    if not cells:
        return False
    reached = {next(iter(cells))}
    stack = list(reached)
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            p = r+dr, c+dc
            if p in cells and p not in reached:
                reached.add(p)
                stack.append(p)
    return reached == cells


def _core(rng, witness):
    if witness:
        return ((0, 0), (0, 1), (1, 0)), 0
    for _ in range(80):
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        tail = int(rng.integers(w))
        cells = {(h-1, tail)}
        area = int(rng.integers(h+w-1, h*w+1))
        while len(cells) < area:
            choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _N4
                              if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            cells.add(choices[int(rng.integers(len(choices)))])
        if _bbox(cells) == (0, h, 0, w):
            return tuple(sorted(cells)), tail
    return ((0, 0), (0, 1), (1, 0)), 0


def _latent_shape(core, tail, length):
    """Generation-side ground truth: a core plus an explicitly drawn arm."""
    h, _ = _size(core)
    return tuple(sorted(core + tuple((h+j, tail) for j in range(length))))


def _decoy(rng, shape, forbidden):
    h, w = _size(shape)
    occupied = set(shape)
    variants = []
    for old in sorted(occupied):
        for new in [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]:
            cells = (occupied - {old}) | {new}
            if (_connected(cells) and _bbox(cells) == (0, h, 0, w)
                    and _key(tuple(sorted(cells))) not in forbidden):
                variants.append(tuple(sorted(cells)))
    if not variants:
        return None
    return variants[int(rng.integers(len(variants)))]


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    acc = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = acc[h:, w:] - acc[:-h, w:] - acc[h:, :-w] + acc[:-h, :-w]
    return np.argwhere(sums == 0)


def _pack(rng, board, qb, shapes, colors, k, bg, marker, noise_color):
    objects = [board]
    for shape, color in zip(shapes, colors):
        h, w = _size(shape)
        obj = np.full((h*k, w*k), bg, dtype=np.int8)
        for r, c in shape:
            obj[r*k:(r+1)*k, c*k:(c+1)*k] = color
        objects.append(obj)
    for attempt in range(70):
        h, w = ((int(rng.integers(23, 31)), int(rng.integers(23, 31)))
                if attempt < 30 else (30, 30))
        canvas = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        placements = {}
        for i in order:
            obj = objects[i]
            hh, ww = obj.shape
            positions = _free_origins(blocked, hh, ww)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            canvas[r:r+hh, c:c+ww] = obj
            blocked[max(0, r-1):min(h, r+hh+1), max(0, c-1):min(w, c+ww+1)] = True
            placements[i] = (r, c)
        else:
            # Optional tiny scraps are rejected by geometry, not a fixed color.
            for _ in range(int(rng.integers(0, 6))):
                scrap = ((0, 0),) if rng.random() < .6 else ((0, 0), (0, 1))
                sh, sw = _size(scrap)
                pos = _free_origins(blocked, sh, sw)
                if not len(pos):
                    break
                r, c = map(int, pos[int(rng.integers(len(pos)))])
                color = int(rng.choice(colors)) if rng.random() < .4 else noise_color
                for rr, cc in scrap:
                    canvas[r+rr, c+cc] = color
                blocked[max(0, r-1):min(h, r+sh+1), max(0, c-1):min(w, c+sw+1)] = True
            if np.count_nonzero(canvas == bg) <= canvas.size // 2:
                continue
            r, c = placements[0]
            bounds = qb[0]+r, qb[1]+r, qb[2]+c, qb[3]+c
            # Crop only empty external margins; the plate encloses the query.
            if rng.random() < .55:
                cells = np.argwhere(canvas != bg)
                r0, c0 = map(int, cells.min(0))
                r1, c1 = map(int, cells.max(0)+1)
                r0, c0 = max(0, r0-int(rng.integers(3))), max(0, c0-int(rng.integers(3)))
                r1, c1 = min(h, r1+int(rng.integers(3))), min(w, c1+int(rng.integers(3)))
                cropped = canvas[r0:r1, c0:c1]
                if np.count_nonzero(cropped == bg) > cropped.size // 2:
                    canvas = cropped.copy()
                    bounds = bounds[0]-r0, bounds[1]-r0, bounds[2]-c0, bounds[3]-c0
            for rr, cc in _marker_cells(bounds):
                canvas[rr, cc] = marker
            return canvas
    return None


def _make_pair(rng, *, axis, k, skip, span, witness, partial):
    for attempt in range(160):
        core, tail = _core(rng, witness == "reflection")
        bh, w = _size(core)
        n = int(rng.integers(3, 5))
        dh = 1 if witness == "reflection" else int(rng.choice([-1, 1]))
        last = n+skip+span-1
        length0 = int(rng.integers(1, 3)) + (last if dh < 0 else 0)
        shapes = [_latent_shape(core, tail, length0+i*dh) for i in range(last+1)]
        if max(_size(s)[0] for s in shapes[n+skip:])*k > 22:
            continue
        if witness == "reflection" and len(_orientations(shapes[n+skip])) != 8:
            continue
        gap = int(rng.integers(1, 4))
        drift = int(rng.choice([-1, 0, 0, 1]))
        positions = [(i*drift, i*(w+gap)) for i in range(last+1)]
        targets = list(range(n+skip, last+1))
        low = min(positions[i][0] for i in targets)
        high = max(positions[i][0]+_size(shapes[i])[0] for i in targets)
        c0 = positions[targets[0]][1]
        c1 = positions[targets[-1]][1] + w
        r0, r1 = low, high
        if partial:
            if rng.random() < .5 and r1-r0 >= 5:
                r0 += int(rng.integers(1, r1-r0-2))
            elif span > 1:
                c0 += int(rng.integers(1, w))
            if dh < 0:
                # A crop ending at the shortened arm would hide the difference
                # from a longer copied arm. Show newly restored metal as evidence.
                r1 += int(rng.integers(1, 3))
        else:
            r0 -= int(rng.integers(2))
            r1 += int(rng.integers(2))
        # At least 2x2, with disconnected Ls; a metal fringe is optional.
        if c1-c0 < 3:
            c1 += 3-(c1-c0)
        if r1-r0 < 3:
            r0 = r1-3
        if max(r1-r0, c1-c0)*k > 30:
            continue
        ymin = min([r0] + [positions[i][0] for i in range(n)])
        ymax = max([r1] + [positions[i][0]+_size(shapes[i])[0] for i in range(n)])
        height, width = ymax-ymin+2, c1+2
        if max(height, width) > 30:
            continue
        roles = list(map(int, rng.permutation(10)))
        bg, structural, marker, noise = roles[:4]
        palette = roles[4:]
        qb = r0-ymin+1, r1-ymin+1, c0+1, c1+1
        board = np.full((height, width), structural, dtype=np.int8)
        for i in range(n):
            y, x = positions[i]
            for rr, cc in shapes[i]:
                board[rr+y-ymin+1, cc+x+1] = bg
        board[qb[0]:qb[1], qb[2]:qb[3]] = bg
        needed = [shapes[i] for i in targets]
        forbidden = {_key(s) for s in needed}
        equal_decoy = _decoy(rng, needed[0], forbidden)
        if equal_decoy is None:
            continue
        # The first demo includes the last observed shape as a concrete wrong
        # constant-repeat choice, in addition to a same-area wrong geometry.
        bank = needed + [equal_decoy]
        if witness == "reflection" and _key(shapes[n-1]) not in {_key(s) for s in bank}:
            bank.append(shapes[n-1])
        colors = [palette[i % len(palette)] for i in range(len(bank))]
        if rng.random() < .25 and len(needed) > 1:
            colors[1] = colors[0]
        loose = []
        for j, shape in enumerate(bank):
            orientations = _orientations(shape)
            if j == 0 and witness == "reflection":
                orientations = tuple(s for s in orientations if s not in _orientations(shape, False))
            elif j == 0 and witness == "rotation":
                orientations = tuple(s for s in _orientations(shape, False) if s != shape) or orientations
            loose.append(orientations[int(rng.integers(len(orientations)))])
        logical = np.full((r1-r0, c1-c0), structural, dtype=np.int8)
        for j, i in enumerate(targets):
            y, x = positions[i]
            for rr, cc in shapes[i]:
                gr, gc = rr+y, cc+x
                if r0 <= gr < r1 and c0 <= gc < c1:
                    logical[gr-r0, gc-c0] = colors[j]
        # Sequence axis and continuation direction are visible, never latent.
        fr, fc = bool(rng.integers(2)), bool(rng.integers(2))
        qmask = np.zeros(board.shape, dtype=bool)
        qmask[qb[0]:qb[1], qb[2]:qb[3]] = True
        board = _transform(board, axis, fr, fc)
        qmask = _transform(qmask, axis, fr, fc)
        qb = _bbox(tuple(map(tuple, np.argwhere(qmask))))
        logical = _transform(logical, axis, fr, fc)
        inp = _pack(rng, board, qb, loose, colors, k, bg, marker, noise)
        if inp is None:
            continue
        out = np.repeat(np.repeat(logical, k, 0), k, 1)
        try:
            recovered, info = _recover(inp)
        except ValueError:
            continue
        if (not np.array_equal(recovered, out) or not info["equal_area_decoy"]
                or not info["a_witness"] or (witness and not info[witness])):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded geometry/packing construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # direction, distance, or resolution shift
    train_axis = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_classes = None
    for split, count in (("train", num_train), ("test", num_test)):
        for j in range(count):
            test = split == "test"
            axis = (not train_axis) if test and mode == 0 else train_axis
            if mode == 1 and j >= 2:
                axis = bool(rng.integers(2))
            k = 3 if test and mode == 2 else 2
            if mode == 0 and j >= 2:
                k = int(rng.choice([2, 3]))
            skip = int(rng.integers(1, 3)) if test and mode == 1 else 0
            span = 1 if (not test and j == 0) or k == 3 else int(rng.choice([1, 2]))
            if not test and j == 1:
                span = 2
            witness = "reflection" if not test and j == 0 else "rotation" if not test and j == 1 else None
            for _ in range(12):
                pair = _make_pair(rng, axis=axis, k=k, skip=skip, span=span,
                                  witness=witness, partial=test or (j >= 2 and rng.random() < .5))
                key = _grid_key(pair["input"])
                classes = _recover(pair["input"])[1]["classes"]
                if not test and j == 1 and classes == first_classes:
                    continue
                if key not in seen:
                    if not test and j == 0:
                        first_classes = classes
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Inspect returned grids, including growth, scale and decoy diagnostics."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _checked_grid(pair["input"]), _checked_grid(pair["output"])
            key = _grid_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), "incorrect output"
            assert info["marker"] not in out and info["background"] not in out
            assert info["growth"] != 0 and info["a_witness"], "constant-copy shortcut not distinguished"
            assert info["equal_area_decoy"] and info["unused_classes"] >= 1, "shape matching not distinguished"
            k = info["scale"]
            logical = out[::k, ::k]
            assert np.array_equal(out, np.repeat(np.repeat(logical, k, 0), k, 1))
            assert np.any(out == info["structural"]) and np.any(out != info["structural"])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            infos[split].append(info)
    assert infos["train"][0]["reflection"] > 0
    assert infos["train"][1]["rotation"] > 0
    assert infos["train"][1]["target_count"] == 2
    assert len({i["classes"] for i in infos["train"]}) >= 2
    train = infos["train"]
    train_axes = {"vertical" if x["direction"] in ("up", "down") else "horizontal" for x in train}
    assert all(("vertical" if t["direction"] in ("up", "down") else "horizontal") not in train_axes
               or t["skipped"] > max(x["skipped"] for x in train)
               or t["scale"] > max(x["scale"] for x in train) for t in infos["test"])
    return None


def _puzzle_key(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed suite, legal-count boundaries and negative diagnostics."""
    keys, infos, dimensions = set(), [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        keys.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _recover(pair["input"])[1]
                infos.append((split, info))
                dimensions.add((pair["input"].shape, pair["output"].shape))
    assert len(keys) == 42 and len(dimensions) >= 80
    all_info = [i for s, i in infos]
    assert {i["direction"] for i in all_info} == {"up", "down", "left", "right"}
    assert {i["scale"] for i in all_info} == {2, 3}
    assert {i["growth"] for i in all_info} >= {-1, 1}
    assert len({shape for i in all_info for shape in i["classes"]}) >= 35
    assert len({i["board_shape"] for i in all_info}) >= 25
    assert len({i["background"] for i in all_info}) == 10
    assert len({i["marker"] for i in all_info}) == 10
    assert {i["visible"] for i in all_info} == {3, 4}
    assert {i["gap"] for i in all_info} >= {1, 2, 3}
    assert {i["drift"] for i in all_info} >= {-1, 0, 1}
    assert any(i["skipped"] > 0 for s, i in infos if s == "test")
    assert any(i["reflection"] for s, i in infos if s == "test")
    assert any(i["rotation"] for s, i in infos if s == "test")
    assert any(i["partial"] for s, i in infos if s == "train")
    assert any(i["partial"] for s, i in infos if s == "test")
    first = generate_puzzle(0, 2, 1)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(0, 2, 1))
    for nt, ne in ((10, 4), (5, 4), (10, 1)):
        validate_generated_puzzle(generate_puzzle(nt*100+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    inp = first["train"][0]["input"].copy()
    scene = _scene(inp)
    color, cells = scene[7][0]
    r, c = cells[0]
    inp[r, c] = scene[0]
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("broken macrocell accepted")
    # Replace a same-area surplus object with a differently colored duplicate of
    # the selected class. Such inventory cannot supply a unique material color.
    inp = first["train"][0]["input"].copy()
    info = _recover(inp)[1]
    scene = _scene(inp)
    k = info["scale"]
    model = next(_geometric_models(scene[4], scene[5], scene[0]))
    wanted = model["targets"][0][1]
    candidates = [(color, cells, _decode(cells, k)) for color, cells in scene[7]]
    selected_color = next(color for color, cells, shape in candidates if _key(shape) == _key(wanted))
    wrong_color, wrong_cells, wrong_shape = next((color, cells, shape) for color, cells, shape in candidates
        if len(shape) == len(wanted) and _key(shape) != _key(wanted) and color != selected_color)
    r0, r1, c0, c1 = _bbox(wrong_cells)
    replacement = next(s for s in _orientations(wanted) if _size(s) == ((r1-r0)//k, (c1-c0)//k))
    inp[r0:r1, c0:c1] = scene[0]
    for r, c in replacement:
        inp[r0+r*k:r0+(r+1)*k, c0+c*k:c0+(c+1)*k] = wrong_color
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting congruent stock colors accepted")
    inp[inp == wrong_color] = selected_color
    assert np.array_equal(solve(inp), first["train"][0]["output"])
    return None
