"""A queried complete line resolves the colors of an ambiguous scaled jigsaw.

Only the standard library and NumPy are used. The renderer creates labels from
the construction model; solve reconstructs the model's meaning from pixels.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normal(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _rotate(shape, turns):
    for _ in range(turns % 4):
        shape = tuple((c, -r) for r, c in shape)
    return _normal(shape)


@lru_cache(maxsize=1024)
def _orientations(shape, reflection=True):
    shape = _normal(shape)
    bases = (shape, tuple((r, -c) for r, c in shape)) if reflection else (shape,)
    return tuple(sorted({_rotate(s, q) for s in bases for q in range(4)}))


def _class(shape):
    return _orientations(_normal(shape))[0]


def _size(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _mask(shape):
    out = np.zeros(_size(shape), dtype=bool)
    for r, c in shape:
        out[r, c] = True
    return out


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _DIRS:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


@lru_cache(maxsize=4)
def _bank(h, w):
    """Enumerate connected polyomino classes, rather than a fixed shape list."""
    found = set()
    for code in range(1, 1 << (h*w)):
        cells = tuple((i//w, i % w) for i in range(h*w) if code & (1 << i))
        if len(cells) <= 7 and len(_components(_mask(_normal(cells)))) == 1:
            found.add(_class(cells))
    return tuple(sorted(found, key=lambda s: (len(s), s)))


def _lines(rows, cols):
    return ([tuple(r*cols+c for c in range(cols)) for r in range(rows)] +
            [tuple(r*cols+c for r in range(rows)) for c in range(cols)])


def _compatible_lines(classes, query_classes, rows, cols):
    want = sorted(query_classes)
    return [line for line in _lines(rows, cols)
            if len(line) == len(want) and sorted(classes[p] for p in line) == want]


def _compatible_diagonals(classes, query_classes, rows, cols):
    if rows != cols or len(query_classes) != rows:
        return []
    diagonals = [tuple(r*cols+r for r in range(rows)),
                 tuple(r*cols+cols-1-r for r in range(rows))]
    return [line for line in diagonals if sorted(classes[p] for p in line) == sorted(query_classes)]


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _as_grid(grid):
    a = np.asarray(grid)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _board_geometry(board, bg, structure):
    choices = []
    for rows, cols, h, w in itertools.product((2, 3), repeat=4):
        if board.shape != (rows*(h+1)+1, cols*(w+1)+1):
            continue
        if not (np.all(board[::h+1] == structure) and np.all(board[:, ::w+1] == structure)):
            continue
        holes = []
        for r, c in itertools.product(range(rows), range(cols)):
            top, left = r*(h+1)+1, c*(w+1)+1
            cells = tuple(map(tuple, np.argwhere(board[top:top+h, left:left+w] == bg)))
            if not cells or len(_components(_mask(_normal(cells)))) != 1:
                break
            holes.append(tuple((int(x)+top, int(y)+left) for x, y in cells))
        else:
            choices.append((rows, cols, h, w, holes))
    _require(len(choices) == 1, "blueprint must have one regular socket lattice")
    return choices[0]


def _decode_macro(cells, k):
    r0, r1, c0, c1 = _bbox(cells)
    if (r1-r0) % k or (c1-c0) % k or len(cells) % (k*k):
        return None
    blocks = Counter(((r-r0)//k, (c-c0)//k) for r, c in cells)
    if any(n != k*k for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _parse_tray(grid, bg, structure, marker, bounds):
    r0, r1, c0, c1 = bounds
    tray = grid[r0:r1, c0:c1]
    _require(marker not in (bg, structure), "tray frame roles must be distinct")
    _require(np.all(tray[[0, -1]] == marker) and np.all(tray[:, [0, -1]] == marker), "incomplete tray frame")
    h = len(tray)-2
    _require(h in (2, 3), "query chamber height must be 2 or 3")
    choices = []
    for q, w in itertools.product((2, 3), (2, 3)):
        if tray.shape[1] != q*(w+1)+1 or not np.all(tray[:, ::w+1] == marker):
            continue
        tokens = []
        for j in range(q):
            chamber = tray[1:-1, j*(w+1)+1:j*(w+1)+1+w]
            ink = np.unique(chamber[chamber != bg])
            if len(ink) != 1 or int(ink[0]) in (structure, marker):
                break
            shape = _normal(tuple(map(tuple, np.argwhere(chamber != bg))))
            if len(_components(_mask(shape))) != 1:
                break
            tokens.append((shape, int(ink[0])))
        else:
            choices.append(tokens)
    _require(len(choices) == 1, "query tray segmentation is ambiguous")
    return choices[0]


def _analyze(input_grid):
    """Input-only scene parsing and joint line/assignment inference."""
    grid = _as_grid(input_grid)
    bg = int(grid[0, 0])
    _require(np.all(grid[[0, -1]] == bg) and np.all(grid[:, [0, -1]] == bg), "canvas needs a background margin")
    colored = [(color, comp) for color in range(10) if color != bg
               for comp in _components(grid == color)]
    board_candidates = []
    for structure, comp in colored:
        bounds = _bbox(comp)
        r0, r1, c0, c1 = bounds
        board = grid[r0:r1, c0:c1]
        if min(board.shape) < 7:
            continue
        if not (np.all(board[[0, -1]] == structure) and np.all(board[:, [0, -1]] == structure)):
            continue
        if not np.all((board == bg) | (board == structure)):
            continue
        try:
            geometry = _board_geometry(board, bg, structure)
        except ValueError:
            continue
        board_candidates.append((structure, bounds, board.copy(), geometry))
    _require(len(board_candidates) == 1, "expected exactly one blueprint")
    structure, bounds, board, (rows, cols, h, w, holes) = board_candidates[0]
    r0, r1, c0, c1 = bounds
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    external = [(color, comp) for color in range(10) if color != bg
                for comp in _components((grid == color) & outside)]
    trays = []
    for marker, comp in external:
        box = _bbox(comp)
        try:
            tokens = _parse_tray(grid, bg, structure, marker, box)
        except ValueError:
            continue
        trays.append((marker, box, tokens))
    _require(len(trays) == 1, "expected exactly one query tray")
    marker, tray_bounds, query = trays[0]
    tr0, tr1, tc0, tc1 = tray_bounds
    outside[tr0:tr1, tc0:tc1] = False
    pieces = [(color, comp) for color in range(10) if color != bg
              for comp in _components((grid == color) & outside)]
    _require(len(pieces) == rows*cols, "one loose piece is required per socket")
    _require(all(color not in (structure, marker) for color, comp in pieces), "piece ink overlaps frame roles")
    _require(len(_components((grid != bg) & outside)) == len(pieces), "loose pieces must be separated")
    # Verify the entire scene, including separation of the two framed objects.
    for box in (bounds, tray_bounds):
        a, b, c, d = box
        moat = grid[a-1:b+1, c-1:d+1].copy()
        moat[1:-1, 1:-1] = bg
        _require(np.all(moat == bg), "framed objects need a background moat")
    hole_classes = [_class(_normal(cells)) for cells in holes]
    hole_counts = Counter(hole_classes)
    interpretations = []
    for k in range(2, 5):
        if max(board.shape)*k > 30:
            continue
        decoded = []
        for color, comp in pieces:
            shape = _decode_macro(comp, k)
            if shape is None:
                break
            decoded.append((shape, color))
        else:
            if Counter(_class(s) for s, ink in decoded) != hole_counts:
                continue
            # B's exact colored lookup is in the loose orientation, before D4 fitting.
            query_indices = []
            for token in query:
                hits = [i for i, p in enumerate(decoded) if p == token]
                if len(hits) != 1:
                    break
                query_indices.append(hits[0])
            else:
                if len(set(query_indices)) != len(query) or len({_class(s) for s, ink in query}) != len(query):
                    continue
                qclasses = [_class(s) for s, ink in query]
                lines = _compatible_lines(hole_classes, qclasses, rows, cols)
                if len(lines) != 1:
                    continue
                line = lines[0]
                queried = {_class(s): ink for s, ink in query}
                residual = defaultdict(list)
                for i, (shape, ink) in enumerate(decoded):
                    if i not in query_indices:
                        residual[_class(shape)].append(ink)
                if any(len(set(inks)) != 1 for inks in residual.values()):
                    continue
                colors = [queried[key] if p in line else residual[key][0]
                          for p, key in enumerate(hole_classes)]
                reflection = rotation = 0
                for p in line:
                    key = hole_classes[p]
                    shape = next(s for s, ink in query if _class(s) == key)
                    dest = _normal(holes[p])
                    reflection += dest not in _orientations(shape, False)
                    rotation += dest != shape and dest in _orientations(shape, False)
                ambiguous = sum(key in residual and residual[key][0] != queried[key]
                                for key in queried)
                color_decoy = any(_class(s) != key and ink == col
                                  for key, col in queried.items()
                                  for i, (s, ink) in enumerate(decoded) if i not in query_indices)
                info = dict(grid=grid, bg=bg, structure=structure, marker=marker,
                            board=board, board_bounds=bounds, tray_bounds=tray_bounds,
                            rows=rows, cols=cols, h=h, w=w, holes=holes,
                            classes=hole_classes, colors=colors, query=query,
                            decoded=decoded, pieces=pieces, query_indices=query_indices,
                            scale=k, line=line, reflection=int(reflection), rotation=int(rotation),
                            ambiguous_classes=int(ambiguous), color_decoy=color_decoy)
                interpretations.append(info)
    _require(len(interpretations) == 1, "expected one scale and one uniquely constrained assembly")
    return interpretations[0]


def _render_answer(info, highlight=True):
    logical = info["board"].copy()
    for cells, color in zip(info["holes"], info["colors"]):
        for r, c in cells:
            logical[r, c] = color
    if highlight:
        line, cols = info["line"], info["cols"]
        if len({p//cols for p in line}) == 1:
            r = line[0]//cols*(info["h"]+1)
            band = logical[r:r+info["h"]+2, :]
        else:
            c = line[0] % cols*(info["w"]+1)
            band = logical[:, c:c+info["w"]+2]
        band[band == info["structure"]] = info["marker"]
    return np.repeat(np.repeat(logical, info["scale"], 0), info["scale"], 1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render_answer(_analyze(input_grid))


_LAYOUTS = {k: tuple((r, c, h, w) for r, c, h, w in itertools.product((2, 3), repeat=4)
                     if max(r*(h+1)+1, c*(w+1)+1)*k <= 30) for k in (2, 3, 4)}


def _random_fit(rng, key, h, w):
    variants = [s for s in _orientations(key) if _size(s)[0] <= h and _size(s)[1] <= w]
    return variants[int(rng.integers(len(variants)))]


def _model(rng, k, lesson=False, query_count=None):
    layouts = [v for v in _LAYOUTS[k]
               if (not lesson or v[2:] == (3, 3) and v[0] == v[1])
               and (query_count is None or query_count in v[:2])]
    rows, cols, h, w = layouts[int(rng.integers(len(layouts)))]
    bank = _bank(h, w)
    # Small and larger shapes coexist. Avoid filling a scene with nine bulky boxes.
    small = [s for s in bank if len(s) <= 4]
    pool = bank if rows*cols <= 6 else tuple(s for s in bank if len(s) <= 5)
    axes = [axis for axis in (0, 1) if query_count is None or (cols if axis == 0 else rows) == query_count]
    axis = axes[int(rng.integers(len(axes)))]
    line = _lines(rows, cols)[int(rng.integers(rows)) if axis == 0 else rows+int(rng.integers(cols))]
    q = len(line)
    selected = [pool[int(i)] for i in rng.choice(len(pool), q, replace=False)]
    chiral = _class(((0, 0), (1, 0), (2, 0), (2, 1)))
    if lesson:
        if chiral in selected:
            j = selected.index(chiral)
            selected[0], selected[j] = selected[j], selected[0]
        else:
            selected[0] = chiral
        # Equal area, different shape: reflection and rotation cannot be
        # replaced by matching areas. The second shape is a T or a zigzag.
        possibilities = [s for s in small if len(s) == 4 and s not in [selected[0]]+selected[2:]
                         and len(_orientations(s, False)) > 1]
        selected[1] = possibilities[int(rng.integers(len(possibilities)))]
    other = [s for s in pool if s not in selected]
    outside = [p for p in range(rows*cols) if p not in line]
    for _ in range(256):
        classes = [None]*(rows*cols)
        for p, key in zip(line, selected):
            classes[p] = key
        # Guarantee geometric/color ambiguity and a same-ink different-shape decoy.
        order = list(map(int, rng.permutation(outside)))
        classes[order[0]] = selected[int(rng.integers(q))]
        classes[order[1]] = other[int(rng.integers(len(other)))]
        for p in order[2:]:
            source = selected if rng.random() < .45 else other
            classes[p] = source[int(rng.integers(len(source)))]
        if lesson:
            diagonal = (tuple(r*cols+r for r in range(rows)) if rng.random() < .5 else
                        tuple(r*cols+cols-1-r for r in range(rows)))
            intersection = next(p for p in diagonal if p in line)
            remaining_keys = [key for key in selected if key != classes[intersection]]
            rng.shuffle(remaining_keys)
            for p, key in zip([p for p in diagonal if p not in line], remaining_keys):
                classes[p] = key
        if (_compatible_lines(classes, selected, rows, cols) == [line]
                and any(classes[p] in selected for p in outside)
                and any(classes[p] not in selected for p in outside)):
            break
    else:
        # Guaranteed unique: one selected class repeats directly perpendicular
        # to itself; all remaining nonline cells have a fresh blocker class.
        blocker = other[0]
        classes = [blocker]*(rows*cols)
        for p, key in zip(line, selected):
            classes[p] = key
        anchor = line[0]
        target = ((anchor+cols) % (rows*cols) if axis == 0 else
                  (anchor//cols)*cols+(anchor % cols+1) % cols)
        classes[target] = selected[0]
        order = [target] + [p for p in outside if p != target]
        assert _compatible_lines(classes, selected, rows, cols) == [line]
        if lesson:
            # Explicit fallback with a full compatible diagonal and blockers.
            classes = [blocker]*(rows*cols)
            for p, key in zip(line, selected):
                classes[p] = key
            diagonal = tuple(r*cols+r for r in range(rows))
            intersection = next(p for p in diagonal if p in line)
            remaining_keys = [key for key in selected if key != classes[intersection]]
            for p, key in zip([p for p in diagonal if p not in line], remaining_keys):
                classes[p] = key
            assert _compatible_lines(classes, selected, rows, cols) == [line]
    palette = list(map(int, rng.permutation(10)))
    bg, structure, marker = palette[:3]
    inks = palette[3:3+int(rng.integers(2, 8))]
    selected_colors = {s: inks[int(rng.integers(len(inks)))] for s in selected}
    residual_colors = {}
    for key in sorted({classes[p] for p in outside}):
        available = [c for c in inks if c != selected_colors.get(key)]
        residual_colors[key] = available[int(rng.integers(len(available)))]
    # One nonqueried shape shares a queried ink, teaching full token identity.
    blocker = next(classes[p] for p in outside if classes[p] not in selected)
    residual_colors[blocker] = selected_colors[selected[0]]
    board = np.full((rows*(h+1)+1, cols*(w+1)+1), structure, np.int8)
    holes, colors, pieces, query = [], [], [], []
    for p, key in enumerate(classes):
        shape = _random_fit(rng, key, h, w)
        sh, sw = _size(shape)
        dr, dc = int(rng.integers(h-sh+1)), int(rng.integers(w-sw+1))
        r, c = divmod(p, cols)
        cells = tuple((r*(h+1)+1+dr+x, c*(w+1)+1+dc+y) for x, y in shape)
        ink = selected_colors[key] if p in line else residual_colors[key]
        variants = _orientations(key)
        if lesson and p in line and key == chiral:
            variants = tuple(s for s in variants if shape not in _orientations(s, False))
        elif lesson and p == line[1]:
            variants = tuple(s for s in _orientations(shape, False) if s != shape)
        loose = variants[int(rng.integers(len(variants)))]
        holes.append(cells)
        colors.append(ink)
        pieces.append((loose, ink))
        if p in line:
            query.append((loose, ink))
        for x, y in cells:
            board[x, y] = bg
    rng.shuffle(pieces)
    rng.shuffle(query)
    return dict(bg=bg, structure=structure, marker=marker, board=board,
                rows=rows, cols=cols, h=h, w=w, holes=holes, colors=colors,
                pieces=pieces, query=query, scale=k, line=line)


def _transpose_model(model):
    m = dict(model)
    m.update(rows=model["cols"], cols=model["rows"], h=model["w"], w=model["h"],
             board=model["board"].T.copy())
    permutation = [r*model["cols"]+c for c in range(model["cols"]) for r in range(model["rows"])]
    m["holes"] = [tuple(sorted((c, r) for r, c in model["holes"][p])) for p in permutation]
    m["colors"] = [model["colors"][p] for p in permutation]
    m["line"] = tuple(sorted((p % model["cols"])*model["rows"]+p//model["cols"] for p in model["line"]))
    return m


def _free_positions(blocked, h, w):
    if h > len(blocked) or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _tray(model):
    query = model["query"]
    h = max(_size(s)[0] for s, ink in query)
    w = max(_size(s)[1] for s, ink in query)
    h, w = max(2, h), max(2, w)
    q = len(query)
    a = np.full((h+2, q*(w+1)+1), model["bg"], np.int8)
    a[[0, -1], :] = model["marker"]
    a[:, ::w+1] = model["marker"]
    for j, (shape, ink) in enumerate(query):
        sh, sw = _size(shape)
        for r, c in shape:
            a[1+(h-sh)//2+r, j*(w+1)+1+(w-sw)//2+c] = ink
    return a


def _pack(rng, model):
    bg, k = model["bg"], model["scale"]
    objects = [model["board"], _tray(model)]
    for shape, ink in model["pieces"]:
        piece = np.full(_size(shape), bg, np.int8)
        piece[_mask(shape)] = ink
        objects.append(np.repeat(np.repeat(piece, k, 0), k, 1))
    order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, -max(objects[i].shape)))
    for attempt in range(100):
        height, width = ((int(rng.integers(22, 31)), int(rng.integers(22, 31)))
                         if attempt < 16 else (30, 30))
        canvas = np.full((height, width), bg, np.int8)
        blocked = np.zeros(canvas.shape, bool)
        blocked[[0, -1], :] = True
        blocked[:, [0, -1]] = True
        for i in order:
            obj = objects[i]
            oh, ow = obj.shape
            available = _free_positions(blocked, oh, ow)
            if not len(available):
                break
            if attempt >= 32:
                # Boundary-biased trials leave larger contiguous free regions.
                distances = np.minimum(available[:, 0]-1, height-oh-1-available[:, 0]) + np.minimum(
                    available[:, 1]-1, width-ow-1-available[:, 1])
                best = available[distances == distances.min()]
                top, left = map(int, best[int(rng.integers(len(best)))])
            else:
                top, left = map(int, available[int(rng.integers(len(available)))])
            canvas[top:top+oh, left:left+ow] = obj
            blocked[top-1:top+oh+1, left-1:left+ow+1] = True
        else:
            # Vary margins and aspect ratio without ever removing the moat.
            rr, cc = np.where(canvas != bg)
            a, b = max(0, int(rr.min())-int(rng.integers(1, 4))), min(height, int(rr.max())+int(rng.integers(2, 5)))
            c, d = max(0, int(cc.min())-int(rng.integers(1, 4))), min(width, int(cc.max())+int(rng.integers(2, 5)))
            return canvas[a:b, c:d].copy()
    return None


def _make_pair(rng, model):
    inp = _pack(rng, model)
    if inp is None:
        return None
    return {"input": inp, "output": _render_answer(model)}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    # Two contrast demonstrations reuse exact loose/query tokens, while the
    # transposed blueprint changes their only legal line and their coloring.
    initial_scale = int(rng.choice((2, 3)))
    for _ in range(80):
        anchor = _model(rng, initial_scale, lesson=True)
        first = _make_pair(rng, anchor)
        second = _make_pair(rng, _transpose_model(anchor))
        if first is not None and second is not None:
            break
    else:
        raise RuntimeError("could not pack teaching contrast")
    puzzle = {"train": [first, second], "test": []}
    seen = {_fingerprint(first["input"]), _fingerprint(second["input"])}
    trained_scales = {initial_scale}
    trained_inventories = {tuple(sorted(anchor["pieces"]))}
    regime = int(rng.integers(3))  # fixed, familiar mixed, or new-scale test
    for split, count in (("train", nt-2), ("test", nq)):
        for index in range(count):
            if regime == 0:
                k = initial_scale
            elif regime == 1:
                k = int(rng.choice((2, 3) if split == "train" else sorted(trained_scales)))
            elif split == "train":
                k = initial_scale
            else:
                k = int(rng.choice([s for s in (2, 3, 4) if s != initial_scale]))
            desired_q = None
            if split == "train" and index == 0:
                # The first fresh lesson exposes the other supported line length.
                desired_q = 5-len(anchor["query"])
            elif split == "test" and nt == 2:
                desired_q = len(anchor["query"])
                if k == 4 and desired_q == 3:
                    k = 3
            for _ in range(80):
                model = _model(rng, k, query_count=desired_q)
                if split == "test" and tuple(sorted(model["pieces"])) in trained_inventories:
                    continue
                pair = _make_pair(rng, model)
                if pair is not None and _fingerprint(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct feasible scene")
            seen.add(_fingerprint(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                trained_scales.add(k)
                trained_inventories.add(tuple(sorted(model["pieces"])))
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    puzzle["test"] = [puzzle["test"][int(i)] for i in rng.permutation(nq)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Schema plus joint-assignment, training-evidence, and rendering invariants."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(nq, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _as_grid(pair["input"]), _as_grid(pair["output"])
            key = _fingerprint(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(inp)
            _require(np.array_equal(_render_answer(info), out), "wrong output")
            _require(info["ambiguous_classes"] >= 1, "line must resolve a genuine color ambiguity")
            _require(info["color_decoy"], "a same-ink different-shape decoy is required")
            # Test B ablation: geometry alone admits a different colored board.
            queried = {_class(s): ink for s, ink in info["query"]}
            witness = next(key for key in queried if any(_class(s) == key and ink != queried[key]
                                                        for s, ink in info["decoded"]))
            destinations = [p for p, key in enumerate(info["classes"]) if key == witness]
            _require(len({info["colors"][p] for p in destinations}) >= 2, "B ablation is ineffective")
            on = next(p for p in destinations if p in info["line"])
            off = next(p for p in destinations if p not in info["line"])
            ablated = dict(info, colors=list(info["colors"]))
            ablated["colors"][on], ablated["colors"][off] = ablated["colors"][off], ablated["colors"][on]
            _require(not np.array_equal(_render_answer(ablated), out), "geometry-only color swap must change answer")
            plain = _render_answer(info, highlight=False)
            _require(not np.array_equal(plain, out), "winning-band highlighting is ineffective")
            # A's scale and cavity fitting must have observable effects.
            _require(out.shape == tuple(n*info["scale"] for n in info["board"].shape), "wrong enlargement")
            _require(np.all(out.reshape(info["board"].shape[0], info["scale"],
                                        info["board"].shape[1], info["scale"]) ==
                            out[::info["scale"], ::info["scale"]][:, None, :, None]), "nonuniform macrocells")
            analyses[split].append(info)
    train = analyses["train"]
    _require({len({p//i["cols"] for p in i["line"]}) == 1 for i in train} == {True, False},
             "training must demonstrate both line axes")
    _require(any(i["reflection"] for i in train), "training needs a reflection witness")
    _require(any(i["rotation"] for i in train), "training needs a rotation witness")
    _require(any(_compatible_diagonals(i["classes"], [_class(s) for s, ink in i["query"]],
                                      i["rows"], i["cols"]) for i in train),
             "training needs to distinguish diagonal completion")
    _require({len(i["query"]) for i in analyses["test"]} <= {len(i["query"]) for i in train},
             "tests must use demonstrated line lengths")
    _require(all(tuple(sorted(i["decoded"])) not in {tuple(sorted(t["decoded"])) for t in train}
                 for i in analyses["test"]), "tests must have fresh piece inventories")
    # Recover the intentional same-token / changed-blueprint contrast from pixels.
    contrast = False
    for a, b in itertools.combinations(train, 2):
        if (sorted(a["decoded"]) == sorted(b["decoded"]) and sorted(a["query"]) == sorted(b["query"])
                and a["marker"] == b["marker"] and a["scale"] == b["scale"]
                and (a["board"].shape != b["board"].shape or not np.array_equal(a["board"], b["board"]))
                and (len({p//a["cols"] for p in a["line"]}) == 1) !=
                    (len({p//b["cols"] for p in b["line"]}) == 1)):
            contrast = True
    _require(contrast, "training must expose lookup against the current geometry")


def self_test():
    """Bounded semantic, variation, metamorphic, count, and corruption checks."""
    fingerprints, shapes, layouts, selections, classes, palettes, scales = (set() for _ in range(7))
    ambiguous, reflected, rotated = [], [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_fingerprint(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                layouts.add((info["rows"], info["cols"], info["h"], info["w"]))
                selections.add((info["rows"], info["cols"], info["line"]))
                classes.update(info["classes"])
                palettes.add((info["bg"], info["structure"], info["marker"]))
                scales.add(info["scale"])
                ambiguous.append(info["ambiguous_classes"])
                reflected.append(info["reflection"])
                rotated.append(info["rotation"])
    assert len(fingerprints) == 42
    assert len(shapes) >= 25 and len(layouts) >= 10 and len(selections) >= 14
    assert len(classes) >= 15 and len(palettes) >= 70 and scales == {2, 3, 4}
    assert min(ambiguous) >= 1 and max(ambiguous) >= 2
    assert any(reflected) and any(rotated)
    baseline = generate_puzzle(0)
    repeat = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in baseline for a, b in zip(baseline[s], repeat[s]) for k in a)
    pair = baseline["test"][0]
    # Palette bijections preserve all geometric and exact-token relations.
    permutation = np.array([9, 4, 1, 7, 2, 8, 0, 6, 3, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Swap exact query-token order in the tray; its output must be identical.
    info = _analyze(pair["input"])
    a, b, c, d = info["tray_bounds"]
    tray = pair["input"][a:b, c:d].copy()
    q = len(info["query"])
    stride = (d-c-1)//q
    reordered = pair["input"].copy()
    for j in range(q):
        reordered[a+1:b-1, c+1+j*stride:c+(j+1)*stride] = tray[1:-1, 1+(q-1-j)*stride:(q-j)*stride]
    assert np.array_equal(solve(reordered), pair["output"])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42+nt*10+nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in baseline[s]] for s in baseline}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    # Broken macrocell: erasing one loose-piece pixel cannot be repaired by the line rule.
    broken = pair["input"].copy()
    r, c = info["pieces"][0][1][0]
    broken[r, c] = info["bg"]
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("partial macrocell accepted")
    # Dense legal cavities beyond the sampler's resource caps, including an
    # enclosed structural island. Also reject a scene with two candidate rows.
    a_shape = ((0, 0), (1, 0), (2, 0), (2, 1))
    b_shape = ((0, 0), (0, 1), (0, 2), (1, 1))
    dense_rng = np.random.default_rng(907)
    for case in ("solid", "ring", "tie"):
        large = (tuple((r, c) for r, c in itertools.product(range(3), repeat=2)
                       if case == "solid" or (r, c) != (1, 1)) if case != "tie" else b_shape)
        masks = (a_shape, b_shape, a_shape, large)
        logical = np.full((9, 9), 5, np.int8)
        cavities = []
        for p, shape in enumerate(masks):
            r, c = divmod(p, 2)
            cells = tuple((r*4+1+x, c*4+1+y) for x, y in shape)
            cavities.append(cells)
            for x, y in cells:
                logical[x, y] = 0
        a_loose = _normal(tuple((r, -c) for r, c in a_shape))
        b_loose = _rotate(b_shape, 1)
        colors = [1, 2, 3, 4 if case == "tie" else 1]
        model = dict(bg=0, structure=5, marker=6, board=logical, rows=2, cols=2,
                     h=3, w=3, holes=cavities, colors=colors, scale=2, line=(0, 1),
                     query=[(a_loose, 1), (b_loose, 2)],
                     pieces=[(a_loose, 1), (b_loose, 2), (a_shape, 3), (large, colors[3])])
        inp = _pack(dense_rng, model)
        assert inp is not None
        if case == "tie":
            try:
                solve(inp)
            except ValueError:
                pass
            else:
                raise AssertionError("tied lines accepted")
        else:
            assert np.array_equal(solve(inp), _render_answer(model))


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, semantic invariants)")
