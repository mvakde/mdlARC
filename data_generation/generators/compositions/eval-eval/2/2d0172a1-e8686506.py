"""Unfold containment diagrams into a finite inventory of skyline pieces.

Only Python's standard library and NumPy are used. No parent module, example
lookup, seed, or construction metadata is consulted by solve().
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, dtype=bool)
    stack = [(0, 0)]
    outside[0, 0] = True
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not padded[y, x] and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    return (~padded & ~outside)[1:-1, 1:-1]


def _bbox(cells):
    a = np.asarray(cells)
    low, high = a.min(axis=0), a.max(axis=0)+1
    return tuple(map(int, (*low, *high)))


def _mask(cells, shape):
    a = np.zeros(shape, dtype=bool)
    rr, cc = zip(*cells)
    a[rr, cc] = True
    return a


def _shape_key(mask):
    return tuple(map(tuple, np.argwhere(mask).tolist()))


def _profile_mask(heights):
    out = np.zeros((max(heights), len(heights)), dtype=bool)
    for c, height in enumerate(heights):
        out[-height:, c] = True
    return out


def _read_input(value):
    """Decode board, stitches, actual containment, and oriented inventory."""
    a = np.asarray(value)
    _require(a.ndim == 2 and np.issubdtype(a.dtype, np.integer), "Expected an integer grid")
    _require(all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9)), "ARC bounds")
    colors, counts = np.unique(a, return_counts=True)
    _require(sum(counts == counts.max()) == 1, "Background must be uniquely dominant")
    bg = int(colors[np.argmax(counts)])
    boards = []
    for color in colors:
        if color == bg:
            continue
        top, left, bottom, right = _bbox(np.argwhere(a == color))
        board = a[top:bottom, left:right]
        if min(board.shape) < 5 or not np.any(board == bg):
            continue
        if (np.all(board[0] == color) and np.all(board[-1] == color)
                and np.all(board[:, 0] == color) and np.all(board[:, -1] == color)):
            boards.append((top, left, bottom, right))
    _require(len(boards) == 1, "Expected one uniform complete scaffold rim")
    top, left, bottom, right = boards[0]
    board = a[top:bottom, left:right].copy()
    outside = a.copy()
    outside[top:bottom, left:right] = bg
    ext_colors, ext_counts = np.unique(outside[outside != bg], return_counts=True)
    # Every drawing color has at least a contour's perimeter. Only the stitch
    # color consists of 2 or 3 isolated single cells outside the board.
    candidates = [int(c) for c, n in zip(ext_colors, ext_counts) if 2 <= n <= 3]
    _require(len(candidates) == 1, "Expected one stitch color")
    stitch = candidates[0]
    marks = [tuple(map(int, p)) for p in np.argwhere(outside == stitch)]
    _require(all(len(c) == 1 for c in _components(outside == stitch)), "Stitches must be isolated")
    payload_colors = set(map(int, ext_colors)) - {stitch}
    _require(payload_colors and not (payload_colors | {stitch}) & (set(map(int, np.unique(board))) - {bg}),
             "Scaffold and drawing palettes must be disjoint")
    assigned = {c: [] for c in payload_colors}
    for r, c in marks:
        neighbors = [int(outside[y, x]) for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                     if 0 <= y < a.shape[0] and 0 <= x < a.shape[1] and outside[y, x] != bg]
        _require(len(neighbors) == 2 and neighbors[0] == neighbors[1] and neighbors[0] in payload_colors,
                 "Stitch must replace a straight outer-contour cell")
        assigned[neighbors[0]].append((r, c))
    inventory = []
    for color in sorted(payload_colors):
        drawing = outside == color
        for p in assigned[color]:
            drawing[p] = True
        records = []
        for cells in _components(drawing):
            cm = _mask(cells, a.shape)
            hole = _holes(cm)
            _require(not hole.any() or len(_components(hole)) == 1, "Contour must have one hole")
            records.append(dict(cells=cells, mask=cm, hole=hole, box=_bbox(cells)))
        parents = []
        for i, child in enumerate(records):
            enclosing = []
            for j, parent in enumerate(records):
                if i == j:
                    continue
                inside = [bool(parent['hole'][p]) for p in child['cells']]
                _require(all(inside) or not any(inside), "Crossing objects")
                if all(inside):
                    enclosing.append(j)
            parents.append(min(enclosing, key=lambda j: int(records[j]['hole'].sum())) if enclosing else None)
        children = {i: [] for i in range(len(records))}
        for i, parent in enumerate(parents):
            if parent is not None:
                children[parent].append(i)
        roots = [i for i, p in enumerate(parents) if p is None]
        _require(len(roots) == len(assigned[color]), "Every packet needs exactly one stitch")
        for root in roots:
            rec = records[root]
            _require(rec['hole'].any(), "Packet root must be a closed contour")
            own_marks = [p for p in assigned[color] if rec['mask'][p]]
            _require(len(own_marks) == 1, "One stitch per root")
            r0, c0, r1, c1 = rec['box']
            mark = np.zeros((r1-r0, c1-c0), dtype=bool)
            mr, mc = own_marks[0]
            mark[mr-r0, mc-c0] = True
            rotations = []
            for k in range(4):
                rotated = np.rot90(mark, -k)
                y, x = np.argwhere(rotated)[0]
                if y == rotated.shape[0]-1 and 0 < x < rotated.shape[1]-1:
                    rotations.append(k)
            _require(len(rotations) == 1, "Stitch must identify one base side")
            rotation = rotations[0]
            leaves, depths, blob_areas, contour_boxes = [], [], [], []
            descendants = []

            def visit(i, depth):
                descendants.append(i)
                obj = records[i]
                if obj['hole'].any():
                    _require(children[i], "Empty contours are not material diagrams")
                    y0, x0, y1, x1 = obj['box']
                    contour_boxes.append((y1-y0, x1-x0))
                    for child in children[i]:
                        visit(child, depth+1)
                else:
                    _require(not children[i], "Blob cannot enclose another object")
                    leaf = np.rot90(obj['mask'][r0:r1, c0:c1], -rotation)
                    y0, x0, y1, x1 = _bbox(np.argwhere(leaf))
                    _require(leaf.sum() >= .4*(y1-y0)*(x1-x0), "Noncompact terminal")
                    leaves.append((x0, x1, y0, y1, depth))
                    blob_areas.append(len(obj['cells']))
            visit(root, 0)
            leaves.sort()
            _require(2 <= len(leaves) <= 5, "Each packet has 2..5 terminals")
            _require(max(p[2] for p in leaves) < min(p[3] for p in leaves), "Terminals need a common transverse line")
            _require(all(x[1] < y[0] for x, y in zip(leaves, leaves[1:])), "Terminal projections must be separated")
            depths = tuple(p[4] for p in leaves)
            _require(all(1 <= d <= 3 for d in depths), "Nesting depth must be 1..3")
            silhouette = np.rot90(_profile_mask(tuple(d+1 for d in depths)), rotation).copy()
            inventory.append(dict(color=color, mask=silhouette, depths=depths, rotation=rotation,
                                  root_box=rec['box'], mark=own_marks[0], blob_areas=tuple(blob_areas),
                                  contour_boxes=tuple(contour_boxes), components=len(descendants)))
    _require(2 <= len(inventory) <= 3 and len(inventory) == len(marks), "Inventory must have 2..3 packets")
    _require(sum(int(p['mask'].sum()) for p in inventory) == int(np.count_nonzero(board == bg)),
             "Unfolded inventory area must equal the board's missing area")
    return dict(background=bg, stitch=stitch, bbox=boards[0], board=board, pieces=inventory)


def _assembly(board, bg, pieces):
    """Exact cover by translations; distinct colored completions count as answers."""
    holes = [tuple(map(int, p)) for p in np.argwhere(board == bg)]
    index = {p: i for i, p in enumerate(holes)}
    types = Counter((int(p['color']), _shape_key(p['mask'])) for p in pieces)
    kinds = sorted(types)
    by_cell = [[] for _ in holes]
    initial = []
    for t, (color, cells) in enumerate(kinds):
        sh, sw = max(r for r, c in cells)+1, max(c for r, c in cells)+1
        count = 0
        for r in range(board.shape[0]-sh+1):
            for c in range(board.shape[1]-sw+1):
                translated = [(r+y, c+x) for y, x in cells]
                if not all(p in index for p in translated):
                    continue
                ids = [index[p] for p in translated]
                mask = sum(1 << i for i in ids)
                paint = sum((color+1) << (4*i) for i in ids)
                move = (t, mask, paint)
                for i in ids:
                    by_cell[i].append(move)
                count += 1
        initial.append(count)
    nodes = 0

    @lru_cache(maxsize=4096)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if not remaining:
            return (0,) if not any(counts) else ()
        options = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length()-1
            viable = [m for m in by_cell[i] if counts[m[0]] and m[1] & remaining == m[1]]
            if not viable:
                return ()
            if options is None or len(viable) < len(options):
                options = viable
                if len(viable) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint in options:
            rest = list(counts)
            rest[t] -= 1
            for suffix in search(remaining ^ mask, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) >= 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))
    codes = search((1 << len(holes))-1, tuple(types[k] for k in kinds))
    answers = []
    for code in codes:
        out = board.copy()
        for i, p in enumerate(holes):
            out[p] = ((code >> (4*i)) & 15)-1
        answers.append(out)
    return answers, nodes, tuple(initial)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    answers, _, _ = _assembly(info['board'], info['background'], info['pieces'])
    _require(len(answers) == 1, "The supplied inventory must have one colored completion")
    return answers[0]


def _depths(tree, depth=0):
    if tree is None:
        return (depth,)
    return sum((_depths(child, depth+1) for child in tree), ())


def _sample_tree(rng, n, mixed=False, deep=False):
    children = [None]*n
    if mixed or rng.random() < .8:
        length = int(rng.integers(1, n if mixed else n+1))
        start = int(rng.integers(n-length+1))
        nested = [None]*length
        if deep:
            at = int(rng.integers(length))
            nested[at] = (None,)
        children[start:start+length] = [tuple(nested)]
        if len(children) >= 3 and rng.random() < .3:
            free = [i for i, x in enumerate(children) if x is None]
            if len(free) >= 2:
                at = int(rng.choice(free))
                children[at] = (None,)
    return tuple(children)


def _tree_for_depths(depths, level=1):
    """A metric-redraw test helper; distinct trees can encode the same profile."""
    children, i = [], 0
    while i < len(depths):
        if depths[i] == level:
            children.append(None)
            i += 1
        else:
            end = i+1
            while end < len(depths) and depths[end] > level:
                end += 1
            children.append(_tree_for_depths(depths[i:end], level+1))
            i = end
    return tuple(children)


def _outline(rng, h, w, pads, root):
    """A connected stepped perimeter around a safe rectangular child corridor."""
    top, bottom, left, right = pads
    domain = np.ones((h, w), dtype=bool)
    if rng.random() < .85:
        # Clipped corners give genuine displaced perimeter cells. An eight-
        # neighbor boundary retains orthogonal bridges on the stepped diagonals.
        for flip_r, flip_c in ((False, False), (False, True), (True, False), (True, True)):
            cut = int(rng.integers(0, min(top, bottom, left, right)))
            for y in range(cut):
                for x in range(cut-y):
                    domain[h-1-y if flip_r else y, w-1-x if flip_c else x] = False
        # Independently movable shallow side bays, outside the child moat.
        if top >= 3 and w >= 7 and rng.random() < .65:
            a = int(rng.integers(2, w-3))
            domain[0, a:a+2] = False
        if left >= 3 and h >= 7 and rng.random() < .5:
            a = int(rng.integers(2, h-3))
            domain[a:a+2, 0] = False
        if right >= 3 and h >= 7 and rng.random() < .5:
            a = int(rng.integers(2, h-3))
            domain[a:a+2, -1] = False
    eroded = domain.copy()
    padded = np.pad(domain, 1)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy+h, dx:dx+w]
    outline = domain & ~eroded
    # Central bottom cells remain straight for the stitch, even on a jagged loop.
    if root:
        outline[-1, w//2-1:w//2+2] = True
    return outline


def _draw_tree(rng, tree, compact=True, root=True):
    """Metric drawing only. Material heights are never encoded by pixel sizes."""
    if tree is None:
        h = int(rng.choice([1, 1, 2]))
        w = int(rng.choice([1, 1, 2, 3]))
        mask = np.ones((h, w), dtype=bool)
        if h == 2 and w >= 2 and rng.random() < .5:
            mask[int(rng.integers(h)), int(rng.integers(w))] = False
        return mask, (h-1)//2
    parts = [_draw_tree(rng, child, compact, False) for child in tree]
    anchor = max(y for m, y in parts)
    below = max(m.shape[0]-1-y for m, y in parts)
    gaps = [int(rng.integers(1, 2 if compact else 4)) for _ in parts[:-1]]
    height = anchor+below+1
    width = sum(m.shape[1] for m, y in parts)+sum(gaps)
    content = np.zeros((height, width), dtype=bool)
    col = 0
    for i, (mask, y) in enumerate(parts):
        row = anchor-y
        content[row:row+mask.shape[0], col:col+mask.shape[1]] |= mask
        col += mask.shape[1]+(gaps[i] if i < len(gaps) else 0)
    pads = [int(rng.choice([2, 2, 2, 3])) for _ in range(4)] if compact else [int(rng.integers(2, 5)) for _ in range(4)]
    top, bottom, left, right = pads
    out = _outline(rng, height+top+bottom, width+left+right, pads, root)
    out[top:top+height, left:left+width] |= content
    return out, anchor+top


def _packet(rng, tree, rotation, color, bg, stitch, compact):
    mask, _ = _draw_tree(rng, tree, compact)
    tile = np.where(mask, color, bg).astype(np.int8)
    tile[-1, tile.shape[1]//2] = stitch
    return np.rot90(tile, rotation).copy()


def _make_board(rng, pieces, bg, fixed, secondary, shared):
    for _ in range(80):
        h = int(rng.integers(max(6, max(p['mask'].shape[0] for p in pieces)+2), 14))
        w = int(rng.integers(max(6, max(p['mask'].shape[1] for p in pieces)+2), 14))
        used = set()
        placements = []
        order = list(map(int, rng.permutation(len(pieces))))
        for i in order:
            shape = pieces[i]['mask']
            cells = _shape_key(shape)
            candidates = []
            for y in range(1, h-shape.shape[0]):
                for x in range(1, w-shape.shape[1]):
                    pos = {(y+r, x+c) for r, c in cells}
                    if pos & used:
                        continue
                    touch = any((r+dr, c+dc) in used for r, c in pos
                                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                    if used and shared and not touch:
                        continue
                    candidates.append(pos)
            if not candidates:
                break
            chosen = candidates[int(rng.integers(len(candidates)))]
            used |= chosen
            placements.append((i, chosen))
        else:
            if len(used) > .66*h*w:
                continue
            board = np.full((h, w), fixed, dtype=np.int8)
            if secondary is not None:
                cut_r = int(rng.integers(1, h-1))
                cut_c = int(rng.integers(1, w-1))
                style = int(rng.integers(3))
                for r in range(1, h-1):
                    for c in range(1, w-1):
                        if ((r >= cut_r) if style == 0 else (c >= cut_c) if style == 1 else ((r >= cut_r) != (c >= cut_c))):
                            board[r, c] = secondary
            out = board.copy()
            for i, cells in placements:
                for p in cells:
                    board[p] = bg
                    out[p] = pieces[i]['color']
            answers, _, initial = _assembly(board, bg, pieces)
            if len(answers) != 1 or not np.array_equal(answers[0], out):
                continue
            if shared and max(initial) <= 1:
                continue
            return board, out
    return None


def _pack(rng, tiles, bg):
    """Bounded guillotine rectangle packing; no panel boundaries are drawn."""
    n = len(tiles)
    dp = {}
    for i, tile in enumerate(tiles):
        if max(tile.shape) > 30:
            return None
        dp[1 << i] = {tile.shape: [(i, 0, 0)]}
    for size in range(2, n+1):
        for subset in range(1, 1 << n):
            if subset.bit_count() != size:
                continue
            options = {}
            a = (subset-1) & subset
            while a:
                b = subset ^ a
                if b and a & (subset & -subset):
                    for (ha, wa), pa in dp.get(a, {}).items():
                        for (hb, wb), pb in dp.get(b, {}).items():
                            for axis in (0, 1):
                                gap = int(rng.integers(1, 3))
                                h, w = (ha+hb+gap, max(wa, wb)) if axis == 0 else (max(ha, hb), wa+wb+gap)
                                if max(h, w) > 30:
                                    continue
                                aa, bb = list(pa), list(pb)
                                if axis == 0:
                                    xa, xb = int(rng.integers(w-wa+1)), int(rng.integers(w-wb+1))
                                    aa = [(i, y, x+xa) for i, y, x in aa]
                                    bb = [(i, y+ha+gap, x+xb) for i, y, x in bb]
                                else:
                                    ya, yb = int(rng.integers(h-ha+1)), int(rng.integers(h-hb+1))
                                    aa = [(i, y+ya, x) for i, y, x in aa]
                                    bb = [(i, y+yb, x+wa+gap) for i, y, x in bb]
                                positions = aa+bb
                                if rng.random() < .5:
                                    # Exchange the children along the join axis.
                                    if axis == 0:
                                        positions = [(i, y+hb+gap, x) for i, y, x in aa]+[(i, y-ha-gap, x) for i, y, x in bb]
                                    else:
                                        positions = [(i, y, x+wb+gap) for i, y, x in aa]+[(i, y, x-wa-gap) for i, y, x in bb]
                                if (h, w) not in options or rng.random() < .2:
                                    options[h, w] = positions
                a = (a-1) & subset
            # At most 24 footprints per subproblem, bounded independently of seed.
            footprints = sorted(options, key=lambda s: (s[0]*s[1], max(s)))[:24]
            dp[subset] = {s: options[s] for s in footprints}
    options = dp.get((1 << n)-1, {})
    if not options:
        return None
    shapes = list(options)
    h, w = shapes[int(rng.integers(len(shapes)))]
    extra_h, extra_w = int(rng.integers(min(3, 31-h))), int(rng.integers(min(3, 31-w)))
    y0, x0 = int(rng.integers(extra_h+1)), int(rng.integers(extra_w+1))
    out = np.full((h+extra_h, w+extra_w), bg, dtype=np.int8)
    for i, y, x in options[h, w]:
        th, tw = tiles[i].shape
        out[y+y0:y+y0+th, x+x0:x+x0+tw] = tiles[i]
    return out


def _make_pair(rng, palette, kind, rotation_base, allow_deep=True):
    bg, fixed, stitch, second = map(int, palette[:4])
    for attempt in range(240):
        n = 2 if kind == 'nesting' else 3 if kind == 'inventory' else int(rng.integers(2, 4))
        trees, rotations = [], []
        for i in range(n):
            leaves = (5 if kind == 'test' and i == 0 else
                      int(rng.integers(3, 5)) if kind == 'nesting' and i == 0 else int(rng.integers(2, 5)))
            # Three levels are compatible but expensive; they occur only when
            # the complete diagram and board can both be packed within ARC bounds.
            deep = allow_deep and kind not in ('nesting', 'inventory') and rng.random() < .12
            trees.append(_sample_tree(rng, leaves, mixed=(i == 0 or kind == 'nesting'), deep=deep))
            rotations.append((rotation_base+i) % 4 if kind == 'nesting'
                             else (rotation_base+2+i) % 4 if kind == 'inventory'
                             else int(rng.integers(4)))
        if kind == 'inventory':
            # An actual second copy is supplied and used; it is never inferred.
            trees[1] = trees[0]
            rotations[1] = rotations[0]
            rotations[2] = (rotation_base+3) % 4
        elif n == 3 and rng.random() < .2:
            source = int(rng.integers(2))
            trees[2], rotations[2] = trees[source], rotations[source]
        if kind == 'nesting':
            first = _depths(trees[0])
            if first in (tuple(sorted(first)), tuple(sorted(first, reverse=True))):
                continue
            if all(_depths(t) == _depths(t)[::-1] for t in trees):
                continue
        pieces, color_of_shape = [], {}
        for i, (tree, rotation) in enumerate(zip(trees, rotations)):
            depths = _depths(tree)
            mask = np.rot90(_profile_mask(tuple(d+1 for d in depths)), rotation).copy()
            sig = _key(mask)
            color = color_of_shape.setdefault(sig, int(palette[4+i]))
            pieces.append(dict(color=color, mask=mask))
        if kind != 'inventory' and n == 3 and rng.random() < .35:
            # Color does not replace segmentation: one color can describe two
            # different shapes, which remain separately supplied inventory items.
            if _key(pieces[2]['mask']) != _key(pieces[0]['mask']):
                pieces[2]['color'] = pieces[0]['color']
        if len({p['color'] for p in pieces}) < 2:
            continue
        shared = kind in ('nesting', 'inventory', 'test') or rng.random() < .65
        board_result = _make_board(rng, pieces, bg, fixed, second if rng.random() < .5 else None, shared)
        if board_result is None:
            continue
        board, expected = board_result
        compact = attempt >= 5 or rng.random() < .8
        tiles = [board]+[_packet(rng, tree, rotation, p['color'], bg, stitch, compact)
                         for tree, rotation, p in zip(trees, rotations, pieces)]
        inp = _pack(rng, tiles, bg)
        if inp is None:
            continue
        try:
            info = _read_input(inp)
            recovered = Counter((p['color'], _key(p['mask'])) for p in info['pieces'])
            intended = Counter((p['color'], _key(p['mask'])) for p in pieces)
            if recovered != intended or not np.array_equal(solve(inp), expected):
                continue
        except ValueError:
            continue
        return dict(input=inp, output=expected)
    raise RuntimeError('Bounded construction exhausted; no invalid fallback is returned')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name+' must be an integer in '+str((low, high)))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    exchange = bool(rng.integers(2))
    rotation_base = int(rng.integers(4))
    result, seen = {'train': [], 'test': []}, set()
    observed_deep = False
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            kind = 'test' if split == 'test' else ('nesting' if i == 0 else 'inventory' if i == 1 else 'ordinary')
            current = rng.permutation(10) if exchange else palette
            for _ in range(8):
                pair = _make_pair(rng, current, kind, rotation_base,
                                  allow_deep=(split == 'train' or observed_deep))
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('Could not generate distinct inputs')
            seen.add(_key(pair['input']))
            result[split].append(pair)
            if split == 'train':
                observed_deep |= any(max(p['depths']) == 3 for p in _read_input(pair['input'])['pieces'])
        rng.shuffle(result[split])
    return result


def _analyze_pair(pair):
    inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
    info = _read_input(inp)
    bg, board, pieces = info['background'], info['board'], info['pieces']
    answers, nodes, initial = _assembly(board, bg, pieces)
    _require(len(answers) == 1 and np.array_equal(out, answers[0]), 'Not the unique input-only completion')
    fixed = board != bg
    _require(out.shape == board.shape and np.array_equal(out[fixed], board[fixed]), 'Fixed cells or crop changed')
    _require(not np.any(out == bg), 'Unfilled material gap')
    expected_counts = Counter()
    for p in pieces:
        expected_counts[p['color']] += int(p['mask'].sum())
    _require(Counter(map(int, out[~fixed])) == expected_counts, 'Material was copied, discarded, or recolored')
    gaps = _components(~fixed)
    shared = sum(len({int(out[p]) for p in gap}) > 1 for gap in gaps)
    types = Counter((p['color'], _key(p['mask'])) for p in pieces)
    flat_area = sum(2*len(p['depths']) for p in pieces)
    return dict(info=info, shared=shared, choices=max(initial), nodes=nodes,
                repeated=max(types.values()) > 1, nested=any(max(p['depths']) > 1 for p in pieces),
                nonuniform=any(len(set(p['depths'])) > 1 for p in pieces),
                flat_fails=flat_area != int(np.count_nonzero(~fixed)))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Pair keys')
            for value in pair.values():
                a = np.asarray(value)
                _require(a.ndim == 2 and np.issubdtype(a.dtype, np.integer), 'Integer grids required')
                _require(all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9)), 'ARC bounds')
            key = _key(pair['input'])
            _require(key not in seen, 'Duplicate input')
            seen.add(key)
            analysis = _analyze_pair(pair)
            if split == 'train':
                training.append(analysis)
    _require(any(a['nonuniform'] and a['flat_fails'] for a in training), 'Nesting demonstration missing')
    train_profiles = [p['depths'] for a in training for p in a['info']['pieces']]
    _require(any(d != tuple(sorted(d)) and d != tuple(sorted(d, reverse=True)) for d in train_profiles),
             'Terminal order demonstration missing')
    _require(any(d != d[::-1] for d in train_profiles), 'Asymmetric profile demonstration missing')
    _require(any(a['shared'] and a['choices'] > 1 for a in training), 'Global assembly demonstration missing')
    _require(any(a['repeated'] for a in training), 'Finite repeated inventory demonstration missing')
    _require({p['rotation'] for a in training for p in a['info']['pieces']} == set(range(4)), 'Base directions missing')
    _require(max(len(p['depths']) for a in training for p in a['info']['pieces']) <= 4, 'Training terminal regime')
    for pair in puzzle['test']:
        info = _read_input(pair['input'])
        _require(any(len(p['depths']) == 5 for p in info['pieces']), 'Longer terminal sequence shift missing')
        _require(max(max(p['depths']) for p in info['pieces']) <= max(max(d) for d in train_profiles),
                 'Test introduces undemonstrated nesting depth')


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, data = _key(pair[role])
                h.update(bytes(shape))
                h.update(data)
    return h.hexdigest()


def self_test():
    """42 streamed seeds, semantic invariants, metamorphisms, and count bounds.

    No files, images, traces, or per-seed payload copies are stored here.
    """
    metrics = dict(seeds=42, pairs=0, shared=0, repeated=0, nonuniform=0, flat_ablation_failures=0,
                   redraws_checked=0, order_ablation_puzzles=0, component_lookup_ablation_puzzles=0)
    signatures, canvases, board_shapes, profiles, layouts, palettes, depths, areas = (set() for _ in range(8))
    rng = np.random.default_rng(21067)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        order_witness = component_witness = False
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                info = a['info']
                metrics['pairs'] += 1
                for k in ('shared', 'repeated', 'nonuniform'):
                    metrics[k] += bool(a[k])
                metrics['flat_ablation_failures'] += bool(a['flat_fails'])
                canvases.add(np.asarray(pair['input']).shape)
                board_shapes.add(np.asarray(pair['output']).shape)
                layouts.add(_key(info['board'] == info['background']))
                palettes.add(tuple(sorted(map(int, np.unique(pair['input'])))))
                for p in info['pieces']:
                    profiles.add((p['depths'], p['rotation']))
                    depths.update(p['depths'])
                    areas.update(p['blob_areas'])
                if split == 'train':
                    component_witness |= a['shared'] > 0
                    # With two differently colored pieces, changing a piece's
                    # nonmonotonic profile cannot preserve its colored region.
                    if len(info['pieces']) == 2 and len({p['color'] for p in info['pieces']}) == 2:
                        nonmonotonic = any(d != tuple(sorted(d)) and d != tuple(sorted(d, reverse=True))
                                           for d in (p['depths'] for p in info['pieces']))
                        if nonmonotonic:
                            for reverse in (False, True):
                                altered = [dict(color=p['color'], mask=np.rot90(
                                    _profile_mask(tuple(d+1 for d in sorted(p['depths'], reverse=reverse))), p['rotation']))
                                           for p in info['pieces']]
                                wrong_answers, _, _ = _assembly(info['board'], info['background'], altered)
                                _require(not any(np.array_equal(answer, pair['output']) for answer in wrong_answers),
                                         'Depth sorting reproduced the ordered witness')
                            order_witness = True
        metrics['order_ablation_puzzles'] += order_witness
        metrics['component_lookup_ablation_puzzles'] += component_witness
        if seed < 6:
            pair = puzzle['test'][0]
            inp, out = pair['input'], pair['output']
            permutation = rng.permutation(10)
            _require(np.array_equal(solve(permutation[inp]), permutation[out]), 'Palette dependence')
            for k in (1, 2, 3):
                _require(np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k)), 'Whole-scene rotation failed')
            info = _read_input(inp)
            redraws = 0
            for _ in range(24):
                # Rebuild topology from visible depths, redraw every component,
                # and scatter afresh while retaining the same board/inventory.
                tiles = [info['board']]+[_packet(rng, _tree_for_depths(p['depths']), p['rotation'], p['color'],
                                                 info['background'], info['stitch'], True)
                                         for p in info['pieces']]
                redrawn = _pack(rng, tiles, info['background'])
                if redrawn is None:
                    continue
                try:
                    answer = solve(redrawn)
                except ValueError:
                    continue
                _require(np.array_equal(answer, out), 'Metric geometry or source placement changed the answer')
                _require(_key(redrawn) != _key(inp), 'Redraw did not vary geometry')
                redraws += 1
                if redraws == 2:
                    break
            _require(redraws == 2, 'Could not exercise independent metric redraws')
            metrics['redraws_checked'] += redraws
            # Delete a material cell in the label: validation must reject it.
            bad = {s: [dict(p) for p in puzzle[s]] for s in puzzle}
            bad['test'][0] = {k: v.copy() for k, v in pair.items()}
            bg = _read_input(inp)['background']
            hole = tuple(np.argwhere(_read_input(inp)['board'] == bg)[0])
            bad['test'][0]['output'][hole] = bg
            try:
                validate_generated_puzzle(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Validator accepted missing material')
            missing_stitch = inp.copy()
            missing_stitch[info['pieces'][0]['mark']] = info['background']
            try:
                solve(missing_stitch)
            except ValueError:
                pass
            else:
                raise AssertionError('An unmarked/open packet was accepted')
    _require(len(signatures) == 42 and len(canvases) >= 35 and len(board_shapes) >= 25, 'Geometric diversity collapsed')
    _require(len(profiles) >= 65 and len(layouts) >= 180 and len(palettes) >= 12, 'Structural diversity collapsed')
    _require({1, 2, 3} <= depths and len(areas) >= 4, 'Nesting/metric variety collapsed')
    _require(metrics['shared'] >= 180 and metrics['repeated'] >= 42 and metrics['nonuniform'] >= 180,
             'Semantic coverage collapsed')
    _require(metrics['order_ablation_puzzles'] == 42 and metrics['component_lookup_ablation_puzzles'] == 42,
             'A parent-free shortcut survived the training witnesses')
    _require(_puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0)), 'Nondeterminism')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(901+10*nt+ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, 'Ignored explicit counts')
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted illegal counts')
    # A genuine differently colored exchange is ambiguous. Equal pieces are
    # an inventory multiplicity, not multiple colored answers.
    board = np.array([[0, 8, 0]], dtype=np.int8)
    unit = np.ones((1, 1), dtype=bool)
    _require(len(_assembly(board, 0, [dict(color=1, mask=unit), dict(color=2, mask=unit)])[0]) == 2,
             'Missed colored ambiguity')
    _require(len(_assembly(board, 0, [dict(color=1, mask=unit), dict(color=1, mask=unit)])[0]) == 1,
             'Identical-piece exchange counted twice')
    shape = _profile_mask((2, 3, 2, 2))
    orientation_board = np.full((5, 6), 8, dtype=np.int8)
    orientation_board[1:4, 1:5][shape] = 0
    _require(len(_assembly(orientation_board, 0, [dict(color=1, mask=shape)])[0]) == 1,
             'Supplied orientation did not fit')
    for wrong in (np.rot90(shape), shape[:, ::-1]):
        _require(not _assembly(orientation_board, 0, [dict(color=1, mask=wrong)])[0],
                 'A rotated or reflected substitute filled an asymmetric slot')
    metrics.update(distinct_puzzles=len(signatures), canvas_shapes=len(canvases), board_shapes=len(board_shapes),
                   oriented_depth_profiles=len(profiles), gap_layouts=len(layouts), palettes=len(palettes),
                   observed_depths=sorted(depths), blob_areas=sorted(areas))
    return metrics


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
