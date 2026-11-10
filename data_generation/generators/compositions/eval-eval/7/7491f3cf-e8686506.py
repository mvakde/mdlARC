"""Territorial jigsaw: exact inventory assembly under a marker-selected cut.

Only standard library and NumPy are used. No files, traces, caches across calls,
parent generators, seeds, or training labels participate in solve().
"""
from collections import Counter
from functools import lru_cache
import hashlib
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _normal(cells):
    mr, mc = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - mr, c - mc) for r, c in cells))


def _components(mask, diagonal=False):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _cut_info(cut):
    """Orthogonal, unbranched guide; its complement has exactly two regions."""
    h, w = cut.shape
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {p: sorted(q for q in ((p[0]-1, p[1]), (p[0]+1, p[1]),
                                  (p[0], p[1]-1), (p[0], p[1]+1)) if q in cells)
             for p in cells}
    ends = sorted(p for p in graph if len(graph[p]) == 1)
    _require(len(cells) >= 5 and len(ends) == 2 and
             all(1 <= len(v) <= 2 for v in graph.values()), "guide is not a simple path")
    border = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
    _require(all(border(p) for p in ends) and
             all(not border(p) for p in cells - set(ends)), "bad guide endpoints")
    order, prev, p = [], None, ends[0]
    while p is not None and p not in order:
        order.append(p)
        following = [q for q in graph[p] if q != prev]
        prev, p = p, following[0] if following else None
    _require(len(order) == len(cells), "disconnected guide")
    regions = _components(~cut)
    _require(len(regions) == 2, "guide must separate two regions")
    labels = np.full(cut.shape, -1, dtype=np.int8)
    sites = []
    for i, region in enumerate(regions):
        for p in region:
            labels[p] = i
        sites.append([p for p in region if all(max(abs(p[0]-r), abs(p[1]-c)) >= 2
                                               for r, c in cells)])
    _require(all(sites), "no separated marker site")
    return labels, sites, order


def _decode(inp):
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(np.count_nonzero(inp == bg) > inp.size // 2, "background must dominate")
    candidates = []
    for fixed in map(int, colors):
        if fixed == bg:
            continue
        rr, cc = np.where(inp == fixed)
        top, left, bottom, right = int(rr.min()), int(cc.min()), int(rr.max())+1, int(cc.max())+1
        h, w = bottom-top, right-left
        if h < 5 or w < 5:
            continue
        board = inp[top:bottom, left:right]
        values = set(map(int, np.unique(board))) - {bg, fixed}
        if len(values) != 1:
            continue
        guide = next(iter(values))
        border = np.concatenate((board[0], board[-1], board[1:-1, 0], board[1:-1, -1]))
        if not np.all((border == fixed) | (border == guide)):
            continue
        # Exactly two boundary guide cells are the ports, never the marker.
        if np.count_nonzero(border == guide) != 2:
            continue
        comps = sorted(_components(board == guide, diagonal=True), key=len)
        if len(comps) != 2 or len(comps[0]) != 1:
            continue
        marker = comps[0][0]
        if marker[0] in (0, h-1) or marker[1] in (0, w-1):
            continue
        cut = np.zeros((h, w), dtype=bool)
        for p in comps[1]:
            cut[p] = True
        try:
            labels, sites, path = _cut_info(cut)
        except AssertionError:
            continue
        selected = int(labels[marker])
        if selected < 0 or marker not in sites[selected]:
            continue
        outside = inp.copy()
        outside[top:bottom, left:right] = bg
        if fixed in outside or guide in outside:
            continue
        if np.any(outside[top:bottom] != bg):
            continue
        inventory, actual = [], []
        readable = True
        for bank, slab, offset in ((0, outside[:top], 0), (1, outside[bottom:], bottom)):
            bank_pieces = _components(slab != bg)
            if not bank_pieces:
                readable = False
                break
            for comp in bank_pieces:
                palette = {int(slab[p]) for p in comp}
                if len(palette) != 1:
                    readable = False
                    break
                color = next(iter(palette))
                inventory.append((bank, color, _normal(comp)))
                actual.append(tuple((r+offset, c) for r, c in comp))
        if not readable:
            continue
        palettes = [{color for b, color, shape in inventory if b == bank} for bank in (0, 1)]
        if palettes[0] & palettes[1]:
            continue
        clean = board.copy()
        clean[cut] = bg
        clean[marker] = fixed
        candidates.append(dict(bg=bg, fixed=fixed, guide=guide,
                               bbox=(top, left, h, w), board=clean, cut=cut,
                               labels=labels, selected=selected, marker=marker,
                               sites=sites, path=path, inventory=inventory,
                               exterior=actual, palettes=palettes))
    _require(len(candidates) == 1, "expected one recognizable board and two banks")
    return candidates[0]


class _SearchLimit(Exception):
    pass


def _assembly(d, ignore_regions=False):
    """Enumerate up to two distinct colored answers, using every piece once.

    Pieces are counted by bank, color and oriented shape, so identical-piece
    permutations cannot manufacture ambiguity. The bounded memo is local only.
    """
    board, bg = d['board'], d['bg']
    h, w = board.shape
    holes = [tuple(map(int, p)) for p in np.argwhere(board == bg)]
    inventory = Counter(d['inventory'])
    if sum(len(shape)*n for (bank, color, shape), n in inventory.items()) != len(holes):
        return [], dict(nodes=0, placements=0)
    types = sorted(inventory)
    indices = {p: i for i, p in enumerate(holes)}
    by_cell = [[] for _ in holes]
    initial_moves = 0
    for t, (bank, color, shape) in enumerate(types):
        sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
        wanted = d['selected'] if bank == 0 else 1-d['selected']
        for r in range(h-sh+1):
            for c in range(w-sw+1):
                cells = [(r+dr, c+dc) for dr, dc in shape]
                if not all(p in indices for p in cells):
                    continue
                sides = {int(d['labels'][p]) for p in cells} - {-1}
                if not ignore_regions and sides != {wanted}:
                    continue
                ids = [indices[p] for p in cells]
                mask = sum(1 << i for i in ids)
                paint = sum((color+1) << (4*i) for i in ids)
                move = t, mask, paint
                initial_moves += 1
                for i in ids:
                    by_cell[i].append(move)
    nodes = 0

    @lru_cache(maxsize=6000)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if nodes > 6000:
            raise _SearchLimit("assembly proof exceeded 6000 states")
        if not remaining:
            return (0,) if not any(counts) else ()
        best, bits = None, remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length()-1
            moves = [m for m in by_cell[i] if counts[m[0]] and m[1] & remaining == m[1]]
            if not moves:
                return ()
            if best is None or len(moves) < len(best):
                best = moves
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint in best:
            rest = list(counts)
            rest[t] -= 1
            for suffix in search(remaining ^ mask, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) == 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(holes))-1, tuple(inventory[t] for t in types))
    results = []
    for code in codes:
        result = board.copy()
        for i, p in enumerate(holes):
            result[p] = ((code >> (4*i)) & 15)-1
        results.append(result)
    search.cache_clear()
    return results, dict(nodes=nodes, placements=initial_moves)


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    d = _decode(inp)
    answers, _ = _assembly(d)
    _require(len(answers) == 1, "inventory must have one colored territorial assembly")
    return answers[0]


def _make_cut(rng, h, w, family):
    cut = np.zeros((h, w), dtype=bool)
    if family == 'vertical':
        cut[:, int(rng.integers(4, w-4))] = True
    elif family == 'horizontal':
        cut[int(rng.integers(3, h-3)), :] = True
    elif family == 'dogleg':
        a = int(rng.integers(4, w-5))
        b = min(w-4, a+int(rng.integers(1, 4)))
        r = int(rng.integers(3, h-3))
        cut[:r+1, a] = True
        cut[r, a:b+1] = True
        cut[r:, b] = True
    elif family == 'elbow':
        r, c = int(rng.integers(4, h-2)), int(rng.integers(4, w-5))
        cut[:r+1, c] = True
        cut[r, c:] = True
    else:
        raise ValueError("unknown cut family")
    if rng.random() < .5:
        cut = cut[:, ::-1].copy()
    if rng.random() < .5:
        cut = cut[::-1].copy()
    return cut


def _neighbors(cells):
    return {(r+dr, c+dc) for r, c in cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - set(cells)


def _layout(rng, cut, balanced):
    """Construct the inventory by removing actual tiles from a complete board."""
    labels, sites, path = _cut_info(cut)
    h, w = cut.shape
    used, tiles = set(), []
    cursor = 0
    side_start = int(rng.integers(2))
    # Every seam tile reaches exactly one region. Adjacent seam tiles share a
    # connected gap; they cannot be solved by a one-hole/one-template lookup.
    while cursor < len(path):
        left = len(path)-cursor
        length = left if left <= 4 else int(rng.integers(2, 5))
        if left-length == 1:
            length += 1
        cells = set(path[cursor:cursor+length])
        side = (len(tiles)+side_start) % 2
        for _ in range(int(rng.integers(1, 4))):
            choices = sorted(p for p in _neighbors(cells) if 0 < p[0] < h-1 and 0 < p[1] < w-1
                             and labels[p] == side and p not in used)
            if not choices:
                if not any(labels[p] == side for p in cells):
                    return None
                break
            cells.add(choices[int(rng.integers(len(choices)))])
        if not any(labels[p] == side for p in cells) or cells & used:
            return None
        tiles.append((side, frozenset(cells)))
        used.update(cells)
        cursor += length
    if len(tiles) > 8:
        return None
    seam_tiles = tiles[:]

    def place(shape, side, separated):
        sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
        candidates = []
        for r in range(1, h-sh):
            for c in range(1, w-sw):
                cells = frozenset((r+dr, c+dc) for dr, dc in shape)
                if cells & used or any(labels[p] != side for p in cells):
                    continue
                if separated and _neighbors(cells) & used:
                    continue
                candidates.append(cells)
        if not candidates:
            return False
        cells = candidates[int(rng.integers(len(candidates)))]
        tiles.append((side, cells))
        used.update(cells)
        return True

    # Equal shape multisets in the introductory twins permit a marker-only
    # reversal. Other examples retain at least one swappable cross-bank shape.
    for i, (side, cells) in enumerate(seam_tiles):
        if balanced or i == 0 or rng.random() < .3:
            if not place(_normal(cells), 1-side, balanced):
                return None
    extras = int(rng.integers(1, 3))
    for _ in range(extras):
        family = int(rng.integers(4))
        length = int(rng.integers(2, 5))
        if family == 0:
            shape = tuple((0, c) for c in range(length))
        elif family == 1:
            shape = tuple((r, 0) for r in range(length))
        elif family == 2:
            shape = ((0, 0), (0, 1), (1, 0), (1, 1))
        else:
            shape = ((0, 0), (1, 0), (1, 1))
        if balanced:
            if not place(shape, 0, True) or not place(shape, 1, True):
                return None
        else:
            place(shape, int(rng.integers(2)), rng.random() < .4)
    if len(tiles) > 18:
        return None
    types = [{_normal(cells) for s, cells in tiles if s == side} for side in (0, 1)]
    if min(map(len, types)) < 2:
        return None
    # The marker must overwrite scaffold, not conceal a missing tile.
    sites = [[p for p in choices if p not in used and 0 < p[0] < h-1 and 0 < p[1] < w-1]
             for choices in sites]
    if not all(sites):
        return None
    return labels, sites, tiles


def _scatter_bank(rng, inventory, width):
    """Bounded shelf packing with randomized order, offsets and blank rows."""
    items = list(inventory)
    rng.shuffle(items)
    items.sort(key=lambda x: -(max(r for r, c in x[1])+1))
    rows, row, col, shelf = [], 0, int(rng.integers(0, 3)), 0
    for color, shape in items:
        h, w = max(r for r, c in shape)+1, max(c for r, c in shape)+1
        if col+w > width:
            row += shelf+1
            col, shelf = int(rng.integers(0, 3)), 0
        rows.append((row, col, color, shape))
        shelf = max(shelf, h)
        col += w+int(rng.integers(1, 4))
    height = row+shelf
    if height > 7:
        return None
    slack = int(rng.integers(0, min(2, 7-height)+1))
    result = np.full((height+slack, width), -1, dtype=np.int8)
    shift = int(rng.integers(slack+1))
    for r, c, color, shape in rows:
        for dr, dc in shape:
            result[r+shift+dr, c+dc] = color
    return result


def _paint_scene(rng, cut, layout, palette, selected, balanced):
    bg, fixed, guide, a0, a1, b0, b1 = map(int, palette[:7])
    labels, sites, tiles = layout
    h, w = cut.shape
    shape_colors = {}
    for bank in (0, 1):
        side = selected if bank == 0 else 1-selected
        signatures = sorted({_normal(cells) for s, cells in tiles if s == side})
        rng.shuffle(signatures)
        colors = (a0, a1) if bank == 0 else (b0, b1)
        for i, shape in enumerate(signatures):
            shape_colors[bank, shape] = colors[i % 2]
    inventories = [[], []]
    output = np.full((h, w), fixed, dtype=np.int8)
    holes = np.zeros((h, w), dtype=bool)
    for side, cells in tiles:
        bank = 0 if side == selected else 1
        shape = _normal(cells)
        color = shape_colors[bank, shape]
        inventories[bank].append((color, shape))
        for p in cells:
            holes[p] = True
            output[p] = color
    board = np.full((h, w), fixed, dtype=np.int8)
    board[holes] = bg
    board[cut] = guide
    marker = sites[selected][int(rng.integers(len(sites[selected])))]
    board[marker] = guide
    width = int(rng.integers(max(24, w+2), 31))
    banks = [_scatter_bank(rng, inv, width) for inv in inventories]
    if any(bank is None for bank in banks):
        return None
    # Optional outside padding stays bounded and is not a cue to piece identity.
    gap0, gap1 = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    height = len(banks[0])+gap0+h+gap1+len(banks[1])
    if height > 30:
        return None
    canvas = np.full((height, width), bg, dtype=np.int8)
    for bank, start in ((banks[0], 0), (banks[1], height-len(banks[1]))):
        mask = bank >= 0
        canvas[start:start+len(bank)][mask] = bank[mask]
    top, left = len(banks[0])+gap0, int(rng.integers(width-w+1))
    canvas[top:top+h, left:left+w] = board
    if np.count_nonzero(canvas == bg) <= canvas.size//2:
        return None
    pair = dict(input=canvas, output=output)
    twin = None
    if balanced:
        twin_input = canvas.copy()
        twin_board = twin_input[top:top+h, left:left+w]
        twin_board[marker] = fixed
        other_marker = sites[1-selected][int(rng.integers(len(sites[1-selected])))]
        twin_board[other_marker] = guide
        twin_output = np.full((h, w), fixed, dtype=np.int8)
        for side, cells in tiles:
            new_bank = 0 if side == 1-selected else 1
            for p in cells:
                twin_output[p] = shape_colors[new_bank, _normal(cells)]
        twin = dict(input=twin_input, output=twin_output)
    return pair, twin


def _new_scene(rng, family, palette, balanced=False, large=False):
    for _ in range(240):
        h = 13 if large else int(rng.integers(9, 13))
        w = int(rng.integers(14 if large else 13, 18))
        cut = _make_cut(rng, h, w, family)
        layout = _layout(rng, cut, balanced)
        if layout is None:
            continue
        result = _paint_scene(rng, cut, layout, palette, int(rng.integers(2)), balanced)
        if result is None:
            continue
        pair, twin = result
        try:
            d = _decode(pair['input'])
            answers, _ = _assembly(d)
            if len(answers) != 1 or not np.array_equal(answers[0], pair['output']):
                continue
            if twin is not None and not np.array_equal(solve(twin['input']), twin['output']):
                continue
        except (AssertionError, _SearchLimit):
            continue
        return pair, twin
    raise RuntimeError("could not construct a unique territorial jigsaw in 240 proposals")


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError("illegal example count")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 1)+1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1)+1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    nt, nq = _counts(rng, num_train, num_test)
    palette = rng.permutation(10)
    change_palette = bool(rng.integers(2))
    regime = int(rng.integers(3))
    families = ('vertical', 'horizontal', 'dogleg', 'elbow')
    # Introductory twins isolate marker semantics with a readable straight cut.
    # Bent guides are available in the remaining demonstrations and tests.
    first_family = str(rng.choice(families[:2]))
    first, twin = _new_scene(rng, first_family, palette, balanced=True)
    puzzle = dict(train=[first, twin], test=[])
    seen = {_key(first['input']), _key(twin['input'])}
    for split, n in (('train', nt-2), ('test', nq)):
        for i in range(n):
            if split == 'test' and regime == 0:
                family = 'dogleg'
            elif split == 'test' and regime == 2:
                family = 'horizontal' if first_family == 'vertical' else 'vertical'
            elif split == 'train' and regime == 0:
                family = str(rng.choice(('vertical', 'horizontal', 'elbow')))
            elif split == 'train' and regime == 2:
                family = first_family
            else:
                family = str(rng.choice(families))
            pair_palette = rng.permutation(10) if change_palette else palette
            for _ in range(8):
                pair, _ = _new_scene(rng, family, pair_palette, large=split == 'test' and regime == 1)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("could not separate inputs")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair, ablate=False):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    d = _decode(inp)
    answers, stats = _assembly(d)
    _require(len(answers) == 1 and np.array_equal(answers[0], out), "wrong or ambiguous output")
    board, bg = d['board'], d['bg']
    fixed = board != bg
    _require(out.shape == board.shape and np.array_equal(out[fixed], board[fixed]), "fixed cells changed")
    _require(not np.any(out == bg) and not np.any(out == d['guide']), "unfilled board or retained guide")
    expected = Counter()
    for bank, color, shape in d['inventory']:
        expected[color] += len(shape)
    actual = Counter(map(int, out[~fixed]))
    _require(expected == actual, "inventory colors not conserved")
    for bank in (0, 1):
        side = d['selected'] if bank == 0 else 1-d['selected']
        visible = set(map(int, out[(d['labels'] == side) & ~fixed]))
        _require(len(visible) >= 2 and visible <= d['palettes'][bank], "bank has no shape-dependent colors")
    _require(all(np.any(np.isin(out[d['cut']], list(p))) for p in d['palettes']), "seam needs both banks")
    common = ({shape for bank, color, shape in d['inventory'] if bank == 0} &
              {shape for bank, color, shape in d['inventory'] if bank == 1})
    _require(common, "missing geometry-only color-swap alternative")
    if ablate:
        unconstrained, _ = _assembly(d, ignore_regions=True)
        _require(len(unconstrained) == 2, "removing A should make colored assembly ambiguous")
    d['stats'] = stats
    return d


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, training_sides = set(), set()
    kinds, heights = {'train': set(), 'test': set()}, {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp = _grid(pair['input'])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _inspect_pair(pair)
            kinds[split].add(_family(d))
            heights[split].append(d['board'].shape[0])
            if split == 'train':
                training_sides.add(d['selected'])
    _require(training_sides == {0, 1}, "training must show both effective marker choices")
    bend_shift = kinds['train'] <= {'vertical', 'horizontal', 'elbow'} and kinds['test'] == {'dogleg'}
    size_shift = min(heights['test']) > max(heights['train'])
    orientation_shift = ((kinds['train'], kinds['test']) in
                         (({'vertical'}, {'horizontal'}), ({'horizontal'}, {'vertical'})))
    _require(bend_shift or size_shift or orientation_shift, "missing geometric train/test shift")


def _family(d):
    path = d['path']
    dr = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    bends = sum(a != b for a, b in zip(dr, dr[1:]))
    return ('horizontal' if path[0][0] == path[-1][0] else 'vertical') if bends == 0 else (
        'elbow' if bends == 1 else 'dogleg')


def self_test():
    """Bounded 42-seed suite; all large puzzle payloads are released per seed."""
    signatures, canvases, boards, layouts, shapes = set(), set(), set(), set(), set()
    families, backgrounds, counts, palettes = set(), set(), set(), set()
    pair_count, shared_gaps, unequal, worst_nodes = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _inspect_pair(pair, ablate=True)
                inp, out = pair['input'], pair['output']
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                canvases.add(inp.shape)
                boards.add(out.shape)
                layouts.add((out.shape, hashlib.sha256((d['board'] == d['bg']).tobytes()).digest()))
                shapes.update(shape for bank, color, shape in d['inventory'])
                families.add(_family(d))
                backgrounds.add(d['bg'])
                counts.add(len(d['inventory']))
                palettes.add((d['bg'], d['fixed'], d['guide'], tuple(sorted(d['palettes'][0])),
                              tuple(sorted(d['palettes'][1]))))
                unequal += sum(bank == 0 for bank, color, shape in d['inventory']) != sum(
                    bank == 1 for bank, color, shape in d['inventory'])
                shared_gaps += any(len(set(map(int, out[tuple(np.array(comp).T)]))) > 1
                                   for comp in _components(d['board'] == d['bg']))
                worst_nodes = max(worst_nodes, d['stats']['nodes'])
                pair_count += 1
        signatures.add(digest.digest())
        # The introductory pair isolates the causal role of the marker.
        a, b = puzzle['train'][:2]
        _require(np.count_nonzero(a['input'] != b['input']) == 2, "twins change more than marker")
        _require(not np.array_equal(a['output'], b['output']), "marker reversal has no effect")
        if seed < 4:
            repeat = generate_puzzle(seed)
            _require(all(np.array_equal(p[g], q[g]) for s in puzzle for p, q in zip(puzzle[s], repeat[s])
                         for g in ('input', 'output')), "nondeterminism")
    _require(len(signatures) == 42, "duplicate puzzles")
    _require(len(canvases) >= 25 and len(boards) >= 15 and len(layouts) >= 100, "geometry collapsed")
    _require(len(shapes) >= 45 and len(counts) >= 7, "inventory variation collapsed")
    _require(families == {'vertical', 'horizontal', 'elbow', 'dogleg'}, "guide vocabulary collapsed")
    _require(backgrounds == set(range(10)) and len(palettes) >= 40, "semantic palette variation collapsed")
    _require(shared_gaps == pair_count and unequal >= 20, "shared assembly/unequal inventories missing")
    for a, b in ((2, 1), (10, 4), (None, 1), (5, None)):
        generate_puzzle(1234, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0, 2, 1)
    for kind in ('output', 'inventory'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        pair = bad['test'][0]
        if kind == 'output':
            pair['output'][0, 0] = (int(pair['output'][0, 0])+1) % 10
        else:
            d = _decode(pair['input'])
            pair['input'][d['exterior'][0][0]] = d['bg']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corruption was accepted")
    # Focused orientation and inventory semantics, independent of sampling.
    board = np.array([[0, 0, 0], [8, 0, 8]], dtype=np.int8)
    d = dict(board=board, bg=0, selected=0, labels=np.zeros_like(board),
             inventory=[(0, 2, ((0, 1), (1, 0), (1, 1), (1, 2)))])
    _require(not _assembly(d)[0], "piece reflection was silently allowed")
    return dict(seeds=42, pairs=pair_count, canvas_shapes=len(canvases), board_shapes=len(boards),
                hole_layouts=len(layouts), oriented_piece_shapes=len(shapes), piece_counts=sorted(counts),
                shared_multicolor_gaps=shared_gaps, unequal_bank_counts=unequal, max_solver_states=worst_nodes)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
