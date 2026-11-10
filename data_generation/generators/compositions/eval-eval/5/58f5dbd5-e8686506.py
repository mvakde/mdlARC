"""Fracture and fit: subtract same-color stencils, then assemble every fragment.

No parent code is imported. All randomness is local to generate_puzzle. The
input-only parser and exact-cover solver use neither labels nor construction
metadata. Diagnostics are returned as small summaries, never written here.
"""
from collections import Counter
from functools import lru_cache
import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _normal(cells):
    y0, x0 = min(y for y, x in cells), min(x for y, x in cells)
    return tuple(sorted((y-y0, x-x0) for y, x in cells))


def _shape_size(shape):
    return max(y for y, x in shape)+1, max(x for y, x in shape)+1


def _mirror(shape):
    return _normal([(y, -x) for y, x in shape])


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [
        np.rot90(np.fliplr(mask), k) for k in range(4)]


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _bbox(cells):
    yy, xx = zip(*cells)
    return min(yy), min(xx), max(yy)+1, max(xx)+1


def _perimeter(a, color):
    return (np.all(a[0] == color) and np.all(a[-1] == color)
            and np.all(a[:, 0] == color) and np.all(a[:, -1] == color))


def _parse(input_grid):
    """Recover stock, cuts, fragments and the unique visibly framed mold."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(_perimeter(a, bg), "Input needs a uniform background border")
    colors = sorted(set(map(int, np.unique(a))) - {bg})
    boards = []
    for color in colors:
        box = _bbox([tuple(p) for p in np.argwhere(a == color)])
        y, x, b, r = box
        board = a[y:b, x:r]
        if (min(board.shape) >= 4 and _perimeter(board, color)
                and np.any(board == bg) and np.all((board == bg) | (board == color))):
            boards.append((color, box, board.copy()))
    _require(len(boards) == 1, "Exactly one framed incomplete mold is required")
    structural, box, board = boards[0]
    _require(np.array_equal(board, board[:, ::-1])
             or np.array_equal(board, board[::-1]), "Mold scaffold must be bilateral")
    stock_colors = [c for c in colors if c != structural]
    _require(2 <= len(stock_colors) <= 3, "Expected two or three stock colors")
    stocks, pieces, boxes = [], [], [box]
    for color in stock_colors:
        comps = _components(a == color)
        largest = max(comps, key=len)
        y, x, b, r = _bbox(largest)
        sh, sw = b-y, r-x
        _require(min(sh, sw) >= 3 and len(largest) == sh*sw,
                 "Each stock must be a solid rectangle")
        outside = a == color
        outside = outside.copy()
        outside[y:b, x:r] = False
        cells = [tuple(p) for p in np.argwhere(outside)]
        _require(bool(cells), "Missing same-color stencil")
        sy, sx, sb, sr = _bbox(cells)
        _require((sb-sy, sr-sx) == (sh, sw), "Stencil must span its stock register")
        register = a[sy:sb, sx:sr]
        _require(np.all((register == bg) | (register == color)), "Overlapping registers")
        cut = register == color
        _require(0 < int(cut.sum()) < cut.size, "Stencil must cut some but not all stock")
        material = ~cut
        fragments = [_normal(c) for c in _components(material)]
        pieces.extend((color, shape) for shape in fragments)
        stocks.append({"color": color, "shape": (sh, sw), "stock_box": (y, x, b, r),
                       "stencil_box": (sy, sx, sb, sr), "cut": cut,
                       "material": material, "fragments": fragments})
        boxes.extend(((y, x, b, r), (sy, sx, sb, sr)))
    # Entire register boxes, including their empty cells, must be separated.
    for i, p in enumerate(boxes):
        for q in boxes[i+1:]:
            _require(p[2] < q[0] or q[2] < p[0] or p[3] < q[1] or q[3] < p[1],
                     "Objects need a background cell between register boxes")
    _require(sum(len(s) for c, s in pieces) == np.count_nonzero(board == bg),
             "Surviving stock area must equal mold gap area")
    return {"background": bg, "structural": structural, "box": box,
            "board": board, "stocks": stocks, "pieces": pieces}


class _SearchLimit(Exception):
    pass


def _assembly(board, bg, pieces, node_limit=None, allowed=None):
    """Find up to two colored answers. Identical-piece permutations collapse.

    The cache has a fixed memory bound. Generation imposes a node limit and
    rejects a timeout; exported solve does not truncate a proof of uniqueness.
    """
    holes = [tuple(map(int, p)) for p in np.argwhere(board == bg)]
    if sum(len(s) for c, s in pieces) != len(holes):
        return [], {"nodes": 0, "placements": ()}
    index = {p: i for i, p in enumerate(holes)}
    inventory = Counter((int(c), _normal(s)) for c, s in pieces)
    types = sorted(inventory)
    by_cell = [[] for _ in holes]
    placements = []
    for t, (color, shape) in enumerate(types):
        sh, sw = _shape_size(shape)
        for y in range(board.shape[0]-sh+1):
            for x in range(board.shape[1]-sw+1):
                cells = [(y+dy, x+dx) for dy, dx in shape]
                if not all(p in index for p in cells):
                    continue
                if allowed is not None and any(allowed[p] != color for p in cells):
                    continue
                ids = [index[p] for p in cells]
                mask = sum(1 << i for i in ids)
                paint = sum((color+1) << (4*i) for i in ids)
                move = (t, mask, paint)
                placements.append((color, tuple(cells)))
                for i in ids:
                    by_cell[i].append(move)
    nodes = 0

    @lru_cache(maxsize=4096)
    def visit(remaining, counts):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise _SearchLimit("Exact cover exceeded construction budget")
        if not remaining:
            return (0,) if not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            choices = [m for m in by_cell[bit.bit_length()-1]
                       if counts[m[0]] and m[1] & remaining == m[1]]
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint in best:
            rest = list(counts)
            rest[t] -= 1
            for suffix in visit(remaining ^ mask, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) >= 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = visit((1 << len(holes))-1, tuple(inventory[t] for t in types))
    answers = []
    for code in codes:
        answer = board.copy()
        for p, i in index.items():
            answer[p] = ((code >> (4*i)) & 15)-1
        answers.append(answer)
    return answers, {"nodes": nodes, "placements": tuple(placements)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    answers, _ = _assembly(info["board"], info["background"], info["pieces"])
    _require(len(answers) == 1, "Inventory must determine one completed colored mold")
    return answers[0].copy()


def _orientation_options(info, answer):
    """Which D4 transformations of every stencil could explain this label?

    Used only to select/check demonstrations. Exported solve has no labels and
    always uses the displayed stencil orientation. Conditioning placements on
    the proposed answer checks feasibility even when an alternative is ambiguous.
    """
    options = set()
    for t in range(8):
        pieces = []
        for stock in info["stocks"]:
            cut = _transforms(stock["cut"])[t]
            if cut.shape != stock["shape"]:
                break
            pieces.extend((stock["color"], _normal(comp)) for comp in _components(~cut))
        else:
            answers, _ = _assembly(info["board"], info["background"], pieces,
                                   node_limit=6000, allowed=answer)
            if answers:
                options.add(t)
    return options


def _compound_fits(stock, answer):
    """Could all surviving material move as one decal, without fracture?"""
    cells = _normal([tuple(map(int, p)) for p in np.argwhere(stock["material"])])
    h, w = _shape_size(cells)
    return any(all(answer[y+dy, x+dx] == stock["color"] for dy, dx in cells)
               for y in range(answer.shape[0]-h+1) for x in range(answer.shape[1]-w+1))


def _swapped_stencils_explain(info, answer):
    """Counterfactual: exchange the first two color associations, keeping ink."""
    stocks = info["stocks"]
    if stocks[0]["shape"] != stocks[1]["shape"]:
        return False
    order = [1, 0] + list(range(2, len(stocks)))
    pieces = [(stock["color"], _normal(comp)) for stock, j in zip(stocks, order)
              for comp in _components(~stocks[j]["cut"])]
    answers, _ = _assembly(info["board"], info["background"], pieces,
                           node_limit=6000, allowed=answer)
    return bool(answers)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _grow(rng):
    """Procedural oriented polyominoes, rather than a finite mask dictionary."""
    h, w = int(rng.integers(2, 5)), int(rng.integers(2, 4))
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    target = int(rng.integers(2, min(7, h*w)+1))
    while len(cells) < target:
        neighbors = sorted({(y+dy, x+dx) for y, x in cells
                            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                            if 0 <= y+dy < h and 0 <= x+dx < w} - cells)
        cells.add(_pick(rng, neighbors))
    shape = _normal(cells)
    for _ in range(int(rng.integers(4))):
        shape = _normal([(x, -y) for y, x in shape])
    return shape


def _source(rng, h, w, shapes):
    for _ in range(25):
        material = np.zeros((h, w), dtype=bool)
        blocked = np.zeros((h, w), dtype=bool)
        for i in sorted(range(len(shapes)), key=lambda i: -len(shapes[i])):
            shape = shapes[i]
            sh, sw = _shape_size(shape)
            choices = [(y, x) for y in range(h-sh+1) for x in range(w-sw+1)
                       if all(not blocked[y+dy, x+dx] for dy, dx in shape)]
            if not choices:
                break
            y, x = _pick(rng, choices)
            for dy, dx in shape:
                yy, xx = y+dy, x+dx
                material[yy, xx] = True
                for ny, nx in ((yy, xx), (yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                    if 0 <= ny < h and 0 <= nx < w:
                        blocked[ny, nx] = True
        else:
            cut = ~material
            if all((cut[0].any(), cut[-1].any(), cut[:, 0].any(), cut[:, -1].any())):
                return material
    return None


def _inventory(rng, n, stock_size, rich):
    h, w = stock_size
    groups, used = [], set()
    for color_id in range(n):
        for _ in range(60):
            own = []
            count = 2 if rich and rng.random() < .35 else 1
            for _ in range(count):
                p = _grow(rng)
                own.append((p, _mirror(p)))
            sigs = {s for group in own for s in group}
            if sigs & used or sum(len(s) for g in own for s in g) > h*w*.66:
                continue
            used.update(sigs)
            groups.extend((color_id, color_id, group) for group in own)
            break
        else:
            return None
    # Exchange the color owners of two opposite-oriented partners. Geometry
    # remains mirror-balanced, but colors and each stock's inventory need not be.
    # This prevents symmetry alone from locating colors or justifying reflection.
    exchange = [(i, j) for i in range(len(groups)) for j in range(i+1, len(groups))
                if groups[i][0] != groups[j][0] and groups[i][2][0] != groups[i][2][1]
                and groups[j][2][0] != groups[j][2][1]]
    if exchange:
        i, j = _pick(rng, exchange)
        ci, _, gi = groups[i]
        cj, _, gj = groups[j]
        groups[i], groups[j] = (ci, cj, gi), (cj, ci, gj)
    shapes_by_color = [[] for _ in range(n)]
    for left, right, (p, q) in groups:
        shapes_by_color[left].append(p)
        shapes_by_color[right].append(q)
    owners = {}
    for color_id, shapes in enumerate(shapes_by_color):
        for shape in shapes:
            if shape in owners and owners[shape] != color_id:
                return None
            owners[shape] = color_id
    materials = [_source(rng, h, w, shapes) for shapes in shapes_by_color]
    if any(m is None for m in materials):
        return None
    return groups, materials


def _mold(rng, groups, shape, contact):
    h, w = shape
    for _ in range(30):
        painted = np.full((h, w), -1, dtype=np.int8)
        used = set()
        order = sorted(range(len(groups)), key=lambda i: (-len(groups[i][2][0]), float(rng.random())))
        for i in order:
            left_color, right_color, (p, q) = groups[i]
            sh, sw = _shape_size(p)
            choices = []
            for y in range(1, h-sh):
                for x in range(1, w-sw):
                    first = {(y+dy, x+dx) for dy, dx in p}
                    second = {(yy, w-1-xx) for yy, xx in first}
                    cells = first | second
                    if first & second or cells & used:
                        continue
                    paint = {cell: left_color for cell in first}
                    paint.update((cell, right_color) for cell in second)
                    adj = sum(painted[ny, nx] >= 0 and painted[ny, nx] != paint[yy, xx]
                              for yy, xx in paint
                              for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)))
                    choices.append((adj, y, x, cells))
            if not choices:
                break
            if contact and max(c[0] for c in choices) > 0:
                choices = [c for c in choices if c[0] > 0]
            elif not contact:
                isolated = [c for c in choices if c[0] == 0]
                if isolated:
                    choices = isolated
            _, y, x, cells = _pick(rng, choices)
            for cell in cells:
                painted[cell] = left_color if cell in {(y+dy, x+dx) for dy, dx in p} else right_color
            used.update(cells)
        else:
            if len(used) <= .74*(h-2)*(w-2):
                return painted
    return None


def _scene(rng, board, materials, colors, bg):
    """Pack separated register boxes; positions carry no pairing information."""
    items = [("board", board)]
    for color, material in zip(colors, materials):
        solid = np.full(material.shape, color, dtype=np.int8)
        stencil = np.where(material, bg, color).astype(np.int8)
        items.extend((("stock", solid), ("stencil", stencil)))
    # Random scatter first; all occupied bounding boxes include a clear halo.
    for attempt in range(10):
        h, w = (int(rng.integers(24, 31)), int(rng.integers(24, 31))) if attempt < 6 else (30, 30)
        canvas = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(items)), key=lambda i: (-items[i][1].size, float(rng.random())))
        for i in order:
            a = items[i][1]
            ah, aw = a.shape
            choices = [(y, x) for y in range(1, h-ah) for x in range(1, w-aw)
                       if not blocked[y:y+ah, x:x+aw].any()]
            if not choices:
                break
            y, x = _pick(rng, choices)
            canvas[y:y+ah, x:x+aw] = a
            blocked[y-1:y+ah+1, x-1:x+aw+1] = True
        else:
            return canvas
    # Constructive shelf packing is bounded and always fits supported sizes.
    order = sorted(range(len(items)), key=lambda i: (-items[i][1].shape[0], -items[i][1].shape[1]))
    canvas = np.full((30, 30), bg, dtype=np.int8)
    y, x, row_h = 1, 1, 0
    for i in order:
        a = items[i][1]
        ah, aw = a.shape
        if x+aw > 29:
            y, x, row_h = y+row_h+1, 1, 0
        if y+ah > 29:
            return None
        canvas[y:y+ah, x:x+aw] = a
        x += aw+1
        row_h = max(row_h, ah)
    return canvas


def _features(info, answer, search):
    bg, board = info["background"], info["board"]
    gap_comps = _components(board == bg)
    shared = sum(len({int(answer[p]) for p in comp}) > 1 for comp in gap_comps)
    cross = any(any(answer[p] != color for p in cells)
                for color, cells in search["placements"])
    counts = Counter(info["pieces"])
    return {"shared_gaps": shared, "cross_color_placement": cross,
            "pieces": len(info["pieces"]), "stock_count": len(info["stocks"]),
            "stock_size": info["stocks"][0]["shape"], "board_shape": board.shape,
            "repeated_piece": max(counts.values()) > 1,
            "several_shapes_same_color": any(len(set(s["fragments"])) > 1 for s in info["stocks"]),
            "asymmetric_cut": any(len({(m.shape, m.tobytes()) for m in _transforms(s["cut"])}) == 8
                                  for s in info["stocks"]),
            "fracture_needed": any(not _compound_fits(s, answer) for s in info["stocks"]),
            "ink_area": sum(int(s["cut"].sum()) for s in info["stocks"]),
            "material_area": sum(len(s) for c, s in info["pieces"]),
            "nodes": search["nodes"]}


def _make_pair(rng, n, stock_size, rotation, coupled, rich, orientation_witness=False,
               single_axis=False):
    for _ in range(100):
        inventory = _inventory(rng, n, stock_size, rich)
        if inventory is None:
            continue
        groups, materials = inventory
        area = sum(int(m.sum()) for m in materials)
        dims = [(h, w) for h in range(6, 13) for w in range(6, 13)
                if area <= .7*(h-2)*(w-2)]
        if not dims:
            continue
        painted = _mold(rng, groups, _pick(rng, dims), coupled)
        if painted is None:
            continue
        if single_axis and np.array_equal(painted >= 0, (painted >= 0)[::-1]):
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, structural, colors = palette[0], palette[1], palette[2:2+n]
        board = np.full(painted.shape, structural, dtype=np.int8)
        board[painted >= 0] = bg
        output = np.full(painted.shape, structural, dtype=np.int8)
        for i, color in enumerate(colors):
            output[painted == i] = color
        pieces = [(colors[i], p) for i, j, (p, q) in groups] + [
            (colors[j], q) for i, j, (p, q) in groups]
        try:
            answers, search = _assembly(board, bg, pieces, node_limit=6000)
        except _SearchLimit:
            continue
        if len(answers) != 1 or not np.array_equal(answers[0], output):
            continue
        if coupled:
            # At least one initial legal placement would paint a wrong color;
            # exact inventory and the remaining placements exclude that choice.
            gap_comps = _components(board == bg)
            if not any(len({int(output[p]) for p in comp}) > 1 for comp in gap_comps):
                continue
            if not any(any(output[p] != c for p in cells) for c, cells in search["placements"]):
                continue
        # Different ink/material counts distinguish subtraction from ink transfer.
        if sum(m.size-int(m.sum()) for m in materials) == area:
            continue
        # Include a source with a fully asymmetric cut; stock positions vanish
        # after fracture, but orientation of every fragment is retained.
        if orientation_witness and not any(
                len({(m.shape, m.tobytes()) for m in _transforms(~material)}) == 8
                for material in materials):
            continue
        if orientation_witness:
            info = {"board": board, "background": bg,
                    "stocks": [{"color": c, "cut": ~m, "shape": m.shape}
                               for c, m in zip(colors, materials)]}
            if _orientation_options(info, output) != {0}:
                continue
        canvas = _scene(rng, board, materials, colors, bg)
        if canvas is None:
            continue
        return {"input": np.rot90(canvas, rotation).copy(),
                "output": np.rot90(output, rotation).copy()}
    raise RuntimeError("Bounded construction exhausted; no ambiguous fallback is returned")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # count, scale, or whole-scene orientation shift
    base_rotation = int(rng.integers(4))
    puzzle, seen, orientation_options = {"train": [], "test": []}, set(), set(range(8))
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            n = (2 if split == "train" else 3) if regime == 0 else int(rng.integers(2, 4))
            sizes = range(4, 6) if regime == 1 and split == "train" else range(5, 8)
            if regime != 1:
                sizes = range(4, 8)
            stock_size = (int(_pick(rng, list(sizes))), int(_pick(rng, list(sizes))))
            if regime == 1 and split == "test" and max(stock_size) == 5:
                stock_size = (int(rng.integers(6, 8)), stock_size[1])
            witness = split == "train" and i == 0
            if witness:
                stock_size = (max(5, stock_size[0]), max(5, stock_size[1]))
            rotation = ((base_rotation + (0 if split == "train" else 1)
                         + 2*int(rng.integers(2))) % 4 if regime == 2
                        else int(rng.integers(4)))
            coupled = split == "test" or i == 1 or rng.random() < .45
            for attempt in range(8):
                pair = _make_pair(rng, n, stock_size, rotation, coupled, rich=True,
                                  orientation_witness=witness, single_axis=regime == 2)
                key = _key(pair["input"])
                if key not in seen:
                    if split == "train":
                        options = _orientation_options(_parse(pair["input"]), pair["output"])
                        if i == 1 and orientation_options & options != {0}:
                            continue
                        orientation_options &= options
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, orientation_options = set(), [], set(range(8))
    pairing_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            info = _parse(a)
            answers, search = _assembly(info["board"], info["background"], info["pieces"])
            _require(len(answers) == 1 and np.array_equal(answers[0], out), "Incorrect or ambiguous answer")
            fixed = info["board"] != info["background"]
            _require(np.array_equal(info["board"][fixed], out[fixed]), "Fixed mold cells changed")
            _require(not np.any(out == info["background"]), "Unfilled mold")
            for stock in info["stocks"]:
                _require(np.count_nonzero(out == stock["color"]) == stock["material"].sum(),
                         "Surviving material not conserved by color")
                _require(len(stock["fragments"]) >= 2, "Generated stock must demonstrate fracture")
            features = _features(info, out, search)
            _require(features["ink_area"] != features["material_area"], "Ink-copy shortcut not distinguished")
            if split == "train":
                training.append(features)
                orientation_options &= _orientation_options(info, out)
                pairing_witness |= not _swapped_stencils_explain(info, out)
    _require(any(f["shared_gaps"] and f["cross_color_placement"] for f in training),
             "Training needs a genuinely coupled assembly witness")
    _require(any(f["asymmetric_cut"] for f in training), "Training lacks stencil orientation evidence")
    _require(orientation_options == {0}, "Training leaves a competing stencil orientation feasible")
    _require(any(f["fracture_needed"] for f in training), "Training does not require separating stock fragments")
    _require(pairing_witness, "Training does not distinguish color-based stencil pairing")


def self_test():
    """Bounded 42-seed geometry, conservation, ambiguity and API regression checks."""
    metrics = {name: set() for name in ("puzzles", "inputs", "outputs", "stocks", "shapes", "counts", "backgrounds")}
    shared = cross = repeated = multi_shape = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        metrics["puzzles"].add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                answers, search = _assembly(info["board"], info["background"], info["pieces"])
                _require(len(answers) == 1 and np.array_equal(answers[0], pair["output"]), "Solver disagreement")
                f = _features(info, pair["output"], search)
                metrics["inputs"].add(pair["input"].shape)
                metrics["outputs"].add(pair["output"].shape)
                metrics["stocks"].update(s["shape"] for s in info["stocks"])
                metrics["shapes"].update(s for c, s in info["pieces"])
                metrics["counts"].add(len(info["pieces"]))
                metrics["backgrounds"].add(info["background"])
                shared += f["shared_gaps"] > 0
                cross += f["cross_color_placement"]
                repeated += f["repeated_piece"]
                multi_shape += f["several_shapes_same_color"]
    _require(len(metrics["puzzles"]) == 42, "Seeds must produce distinct puzzles")
    _require(len(metrics["inputs"]) >= 20 and len(metrics["outputs"]) >= 20, "Insufficient geometry variation")
    _require(len(metrics["stocks"]) >= 10 and len(metrics["shapes"]) >= 35, "Insufficient stock/fragment variation")
    _require(len(metrics["counts"]) >= 3 and metrics["backgrounds"] == set(range(10)), "Insufficient roles/counts")
    _require(min(shared, cross, repeated, multi_shape) >= 20, "Missing coupled or repeated inventories")
    a = generate_puzzle(123, 2, 1)
    _require(_digest(a) == _digest(generate_puzzle(123, 2, 1)), "Nondeterminism")
    # Exact finite inventory, colored-answer uniqueness, orientation preservation.
    board = np.array([[0, 8, 0]], dtype=np.int8)
    _require(len(_assembly(board, 0, [(1, ((0, 0),)), (2, ((0, 0),))])[0]) == 2, "Missed color ambiguity")
    _require(len(_assembly(board, 0, [(1, ((0, 0),)), (1, ((0, 0),))])[0]) == 1, "Identical pieces miscounted")
    slot = np.array([[0, 0, 0], [8, 0, 8]], dtype=np.int8)
    _require(not _assembly(slot, 0, [(1, ((0, 1), (1, 0), (1, 1), (1, 2)))])[0], "Illegally reflected piece")
    _require(not _assembly(board, 0, [(1, ((0, 0),))])[0], "Invented missing material")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42+nt*10+ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "Explicit counts ignored")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted invalid counts")
    bad = {s: [{k: np.array(v).copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Validator accepted a damaged rim")
    # Cut away one additional surviving cell: the same gaps cannot then be
    # filled. This exercises the connection between stencil topology and stock.
    bad = {s: [{k: np.array(v).copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    pair = bad["test"][0]
    info = _parse(pair["input"])
    stock = info["stocks"][0]
    dy, dx = map(int, np.argwhere(stock["material"])[0])
    sy, sx, _, _ = stock["stencil_box"]
    pair["input"][sy+dy, sx+dx] = stock["color"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Validator accepted missing surviving material")
    return {"seeds": 42, "diversity": {k: len(v) for k, v in metrics.items()},
            "shared_gap_pairs": shared, "cross_color_pairs": cross,
            "repeated_piece_pairs": repeated, "multiple_shapes_one_color_pairs": multi_shape}
