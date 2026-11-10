"""Finite portable stencils assembled by translation into a board's gaps.

Only NumPy and the Python standard library are used. No parent generator is
imported. Parsing and solving use visible input cells, never construction state.
"""
from functools import lru_cache
import json
import numpy as np


class SearchLimit(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "Invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        todo, component = [start], []
        while todo:
            r, c = todo.pop()
            component.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in cells:
                    cells.remove(p)
                    todo.append(p)
        result.append(component)
    return result


def _box(cells):
    x = np.asarray(cells)
    return int(x[:, 0].min()), int(x[:, 1].min()), int(x[:, 0].max())+1, int(x[:, 1].max())+1


def _tight(mask):
    r, c, e, d = _box(np.argwhere(mask))
    return mask[r:e, c:d].copy()


def _perimeter(a, color):
    return (np.all(a[0] == color) and np.all(a[-1] == color)
            and np.all(a[:, 0] == color) and np.all(a[:, -1] == color))


def _card_read(tile, bg, marker):
    _require(min(tile.shape) >= 4 and _perimeter(tile, marker), "Malformed legend card")
    inside = tile[1:-1, 1:-1]
    _require(not np.any(inside == marker), "Legend marker appears inside card")
    keys = []
    for cells in _components(inside != bg):
        values, counts = np.unique([inside[p] for p in cells], return_counts=True)
        if len(values) < 2:
            continue
        r, c, e, d = _box(cells)
        _require((e-r, d-c) == (2, 2) and len(cells) == 4
                 and len(values) == 2 and sorted(map(int, counts)) == [1, 3], "Malformed directed key")
        source = int(values[np.argmin(counts)])
        target = int(values[np.argmax(counts)])
        keys.append((r, c, e, d, source, target))
    _require(len(keys) == 1, "Card needs one two-color key")
    r, c, e, d, source, target = keys[0]
    rest = inside != bg
    rest[r:e, c:d] = False
    _require(rest.any() and np.all(inside[rest] == source), "Card needs one source-colored footprint")
    a, b, z, y = _box(np.argwhere(rest))
    footprint = rest[a:z, b:y].copy()
    _require(2 <= int(footprint.sum()) and max(footprint.shape) <= 10, "Unsupported footprint size")
    return source, target, footprint


def _parse(input_grid):
    a = _grid(input_grid)
    colors, amounts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(amounts)])
    _require(int(amounts.max()) > a.size/2, "Background must occupy more than half the canvas")
    card_sets = []
    for marker in colors:
        if marker == bg:
            continue
        components = _components(a == marker)
        if not 2 <= len(components) <= 4:
            continue
        boxes, entries = [], {}
        try:
            for cells in components:
                r, c, e, d = _box(cells)
                h, w = e-r, d-c
                _require(min(h, w) >= 4 and len(cells) == 2*h+2*w-4, "Not card rings")
                tile = a[r:e, c:d]
                s, t, mask = _card_read(tile, bg, int(marker))
                _require(s not in entries and int(marker) not in (s, t), "Ambiguous card association")
                entries[s] = (t, mask)
                boxes.append((r, c, e, d))
            card_sets.append((int(marker), boxes, entries))
        except ValueError:
            continue
    candidates = []
    for marker, cards, entries in card_sets:
        sources = set(entries)
        targets = {t for t, _ in entries.values()}
        for frame in colors:
            frame = int(frame)
            if frame in sources | targets | {bg, marker}:
                continue
            r, c, e, d = _box(np.argwhere(a == frame))
            board = a[r:e, c:d]
            if min(board.shape) < 5 or not _perimeter(board, frame):
                continue
            if any(not (z <= r or e <= x or y <= c or d <= b) for x, b, z, y in cards):
                continue
            fixed = set(map(int, np.unique(board))) - {bg}
            if fixed & (sources | targets | {marker}) or not np.any(board == bg):
                continue
            remaining = a != bg
            remaining[r:e, c:d] = False
            for x, y, z, q in cards:
                remaining[x:z, y:q] = False
            tokens, counts = [], {s: 0 for s in sources}
            try:
                for cells in _components(remaining):
                    x, y, z, q = _box(cells)
                    _require((z-x, q-y) == (2, 2) and len(cells) == 4, "Inventory token is not a solid 2x2 square")
                    values = {int(a[p]) for p in cells}
                    _require(len(values) == 1 and next(iter(values)) in sources, "Unknown inventory token")
                    s = next(iter(values))
                    counts[s] += 1
                    tokens.append((x, y, z, q, s))
                _require(all(counts.values()), "Every card must have inventory")
                _require(sum(counts[s]*int(entries[s][1].sum()) for s in sources)
                         == int(np.sum(board == bg)), "Decoded inventory area does not equal gap area")
                candidates.append(dict(grid=a, bg=bg, marker=marker, frame=frame,
                                       box=(r, c, e, d), board=board.copy(), cards=cards,
                                       masks={s: m for s, (_, m) in entries.items()},
                                       mapping={s: t for s, (t, _) in entries.items()},
                                       counts=counts, tokens=tokens))
            except ValueError:
                continue
    _require(len(candidates) == 1, "Input must identify exactly one board and inventory")
    return candidates[0]


def _assemble(board, bg, masks, counts, mapping, unlimited=False, limit=40000):
    """Enumerate at most two distinct colored exact covers, ignoring token order.

    A timeout is an error, never evidence of uniqueness. Bounding rectangles
    can overlap: only active footprint cells occupy a gap.
    """
    holes = list(map(tuple, np.argwhere(board == bg)))
    n = len(holes)
    sources = sorted(masks)
    if not unlimited and sum(counts[s]*int(masks[s].sum()) for s in sources) != n:
        return [], 0
    indices = {p: i for i, p in enumerate(holes)}
    by_cell = [[] for _ in holes]
    for k, s in enumerate(sources):
        mask = masks[s]
        sh, sw = mask.shape
        offsets = list(map(tuple, np.argwhere(mask)))
        for r in range(board.shape[0]-sh+1):
            for c in range(board.shape[1]-sw+1):
                cells = [(r+x, c+y) for x, y in offsets]
                if not all(p in indices for p in cells):
                    continue
                ids = [indices[p] for p in cells]
                bits = sum(1 << i for i in ids)
                paint = sum((int(mapping[s])+1) << (4*i) for i in ids)
                move = k, bits, paint
                for i in ids:
                    by_cell[i].append(move)
    nodes = 0

    @lru_cache(maxsize=None)
    def search(remaining, inventory):
        nonlocal nodes
        nodes += 1
        if nodes > limit:
            raise SearchLimit("Exact-cover proof exceeded its node bound")
        if not remaining:
            return (0,) if unlimited or not any(inventory) else ()
        best, bits = None, remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length()-1
            options = [m for m in by_cell[i]
                       if (unlimited or inventory[m[0]]) and m[1] & remaining == m[1]]
            if not options:
                return ()
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for k, covered, paint in best:
            rest = list(inventory)
            if not unlimited:
                rest[k] -= 1
            for suffix in search(remaining ^ covered, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) >= 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << n)-1, tuple(counts[s] for s in sources))
    answers = []
    for code in codes:
        result = board.copy()
        for i, p in enumerate(holes):
            result[p] = ((code >> (4*i)) & 15)-1
        answers.append(result)
    return answers, nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode cards and solve the finite assembly using this input alone."""
    a = _parse(input_grid)
    answers, _ = _assemble(a['board'], a['bg'], a['masks'], a['counts'], a['mapping'])
    _require(len(answers) == 1, "Inventory must have one colored completion")
    return answers[0].copy()


def _grow(rng, area):
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = sorted({(r+x, c+y) for r, c in cells
                           for x, y in ((-1, 0), (1, 0), (0, -1), (0, 1))}-cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    r, c, e, d = _box(list(cells))
    mask = np.zeros((e-r, d-c), bool)
    for x, y in cells:
        mask[x-r, y-c] = True
    return mask


def _base(rng, scale=False, rectangular=False):
    if scale:
        return np.ones((1, 4) if rng.random() < .5 else (4, 1), bool)
    if rectangular or rng.random() < .55:
        h, w = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2)][int(rng.integers(5))]
        return np.ones((h, w), bool)
    for _ in range(30):
        mask = _grow(rng, int(rng.integers(3, 7)))
        if max(mask.shape) <= 3:
            return mask
    return np.array([[1, 1, 1], [0, 1, 0]], bool)


def _double(mask, axis):
    return np.concatenate((mask, mask), axis=axis)


def _complement(rng, mask, kind):
    borders = {'elbow': (1, 0, 1, 0), 'cup': (1, 0, 1, 1),
               'ring': (1, 1, 1, 1), 'split': (1, 1, 0, 0)}
    top, bottom, left, right = borders[kind]
    h, w = mask.shape
    other = np.ones((h+top+bottom, w+left+right), bool)
    other[top:top+h, left:left+w] = ~mask
    return other, (top, left)


def _free(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = summed[h:, w:]-summed[:-h, w:]-summed[h:, :-w]+summed[:-h, :-w]
    return np.argwhere(totals == 0)


def _reserve(blocked, r, c, h, w):
    blocked[max(0, r-1):min(blocked.shape[0], r+h+1),
            max(0, c-1):min(blocked.shape[1], c+w+1)] = True


def _make_board(rng, masks, counts, sources, colors):
    bg, frame, secondary = colors
    small, large = sources[:2]
    units = []
    # Every small-footprint token has its own isolated anchor gap. These anchors
    # exhaust the small inventory before any larger region can use extra copies.
    for _ in range(counts[small]):
        units.append((masks[small].shape, [(small, 0, 0)]))
    used_large = 0
    if len(sources) >= 3:
        wrap = sources[2]
        for _ in range(counts[wrap]):
            wh, ww = masks[wrap].shape
            lh, lw = masks[large].shape
            matches = [(r, c) for r in range(wh-lh+1) for c in range(ww-lw+1)
                       if np.array_equal(~masks[wrap][r:r+lh, c:c+lw], masks[large])
                       and int(masks[wrap].sum())+int(masks[large].sum()) == wh*ww]
            if not matches:
                return None
            r, c = matches[int(rng.integers(len(matches)))]
            units.append(((wh, ww), [(wrap, 0, 0), (large, r, c)]))
            used_large += 1
    for _ in range(counts[large]-used_large):
        units.append((masks[large].shape, [(large, 0, 0)]))
    for source in sources[3:]:
        for _ in range(counts[source]):
            units.append((masks[source].shape, [(source, 0, 0)]))
    min_h = max(shape[0] for shape, _ in units)+2
    min_w = max(shape[1] for shape, _ in units)+2
    area = sum((h+1)*(w+1) for (h, w), _ in units)
    for attempt in range(65):
        if attempt > 45:
            h = w = 15
        else:
            h = int(rng.integers(max(7, min_h), 16))
            w = int(rng.integers(max(7, min_w), 16))
        if (h-2)*(w-2) < area*.8:
            continue
        blocked = np.zeros((h-2, w-2), bool)
        placements = []
        order = sorted(range(len(units)), key=lambda i: (-np.prod(units[i][0]), float(rng.random())))
        for i in order:
            (uh, uw), parts = units[i]
            choices = _free(blocked, uh, uw)
            if not len(choices):
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            _reserve(blocked, r, c, uh, uw)
            placements.extend((s, r+1+x, c+1+y) for s, x, y in parts)
        else:
            board = np.full((h, w), frame, np.int8)
            if secondary is not None:
                y, x = np.indices((h-2, w-2))
                style = int(rng.integers(3))
                pattern = y >= (h-2)//2 if style == 0 else x >= (w-2)//2 if style == 1 else (x+y)%3 == 0
                board[1:-1, 1:-1][pattern] = secondary
            for s, r, c in placements:
                sh, sw = masks[s].shape
                board[r:r+sh, c:c+sw][masks[s]] = bg
            return board, placements
    return None


def _card(source, target, mask, bg, marker):
    h, w = mask.shape
    tile = np.full((h+5, max(w, 2)+2), bg, np.int8)
    tile[0] = tile[-1] = marker
    tile[:, 0] = tile[:, -1] = marker
    tile[1:3, 1:3] = [[source, target], [target, target]]
    tile[4:4+h, 1:1+w][mask] = source
    return tile


def _scatter(rng, board, cards, counts, bg):
    tiles = [board] + list(cards) + [np.full((2, 2), s, np.int8)
                                     for s, n in counts.items() for _ in range(n)]
    for attempt in range(85):
        h = int(rng.integers(24, 31)) if attempt < 45 else 30
        w = int(rng.integers(24, 31)) if attempt < 45 else 30
        canvas = np.full((h, w), bg, np.int8)
        blocked = np.zeros((h, w), bool)
        positions = {}
        # Board first; independently shuffled card/token positions carry no map.
        order = [0] + sorted(range(1, len(tiles)), key=lambda i: (-tiles[i].size, float(rng.random())))
        for i in order:
            tile = tiles[i]
            th, tw = tile.shape
            choices = _free(blocked, th, tw)
            if not len(choices):
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            canvas[r:r+th, c:c+tw] = tile
            positions[i] = r, c
            _reserve(blocked, r, c, th, tw)
        else:
            if np.sum(canvas == bg) <= canvas.size/2:
                continue
            # Trim variable unused margins only when dominance stays visible.
            r, c, e, d = _box(np.argwhere(canvas != bg))
            r, c = max(0, r-int(rng.integers(4))), max(0, c-int(rng.integers(4)))
            e, d = min(h, e+int(rng.integers(4))), min(w, d+int(rng.integers(4)))
            trimmed = canvas[r:e, c:d]
            if np.sum(trimmed == bg) > trimmed.size/2:
                canvas = trimmed.copy()
            return canvas
    return None


def _mapping(rng, sources, external, teach=False):
    k = len(sources)
    if teach:
        result = {sources[0]: sources[1], sources[1]: external[0]}
        result.update({s: external[i % len(external)] for i, s in enumerate(sources[2:])})
        return result
    kind = int(rng.integers(4))
    if kind == 0:
        result = {s: external[i % len(external)] for i, s in enumerate(sources)}
    elif kind == 1:
        order = list(map(int, rng.permutation(sources)))
        result = {s: order[(i+1) % k] for i, s in enumerate(order)}
    elif kind == 2:
        order = list(map(int, rng.permutation(sources)))
        result = {s: order[i+1] if i+1 < k else external[0] for i, s in enumerate(order)}
    else:
        palette = sources + external
        result = {s: int(rng.choice([t for t in palette if t != s])) for s in sources}
    if result[sources[0]] == result[sources[1]]:
        options = [t for t in sources+external if t != sources[1] and t != result[sources[0]]]
        result[sources[1]] = int(rng.choice(options))
    return result


def _pair(rng, training, index, style, palette, rotation):
    for _ in range(180):
        scale = not training and style == 'scale'
        fragment = not training and style == 'fragmentation'
        k = (2 if index == 0 else int(rng.integers(3, 5))) if training and index < 2 else int(rng.integers(2, 5))
        if scale:
            k = int(rng.integers(2, 4))
        if fragment or (not training and style == 'repetition'):
            k = 3
        bg, frame, marker = map(int, palette[:3])
        sources = list(map(int, palette[3:3+k]))
        external = list(map(int, palette[3+k:9]))
        secondary = int(palette[9]) if rng.random() < .45 else None
        base = _base(rng, scale, rectangular=style == 'fragmentation')
        axis = int(np.argmax(base.shape)) if scale else int(rng.integers(2))
        masks = {sources[0]: base, sources[1]: _double(base, axis)}
        if k >= 3:
            kind = 'split' if fragment else 'elbow' if training and index == 1 else str(rng.choice(['elbow', 'cup', 'ring']))
            masks[sources[2]], _ = _complement(rng, masks[sources[1]], kind)
            if masks[sources[2]].sum() <= base.sum():
                continue
        if k == 4:
            extra = _grow(rng, int(rng.integers(int(base.sum())+2, int(base.sum())+6)))
            if max(extra.shape) > 5:
                continue
            masks[sources[3]] = extra
        high = not training and style == 'repetition'
        counts = {sources[0]: 3 if high else int(rng.integers(1, 3)),
                  sources[1]: 4 if high else int(rng.integers(1, 3))}
        if k >= 3:
            counts[sources[2]] = 2 if high else 1
        if k == 4:
            counts[sources[3]] = int(rng.integers(1, 3))
        mapping = _mapping(rng, sources, external, teach=training and index == 0)
        layout = _make_board(rng, masks, counts, sources, (bg, frame, secondary))
        if layout is None:
            continue
        board, placements = layout
        expected = board.copy()
        for s, r, c in placements:
            h, w = masks[s].shape
            expected[r:r+h, c:c+w][masks[s]] = mapping[s]
        try:
            answers, _ = _assemble(board, bg, masks, counts, mapping)
            if len(answers) != 1 or not np.array_equal(answers[0], expected):
                continue
        except SearchLimit:
            continue
        cards = [_card(s, mapping[s], masks[s], bg, marker) for s in rng.permutation(sources)]
        canvas = _scatter(rng, board, cards, counts, bg)
        if canvas is None:
            continue
        canvas = np.rot90(canvas, rotation).copy()
        expected = np.rot90(expected, rotation).copy()
        try:
            if np.array_equal(solve(canvas), expected):
                return {'input': canvas, 'output': expected}
        except ValueError:
            continue
    raise RuntimeError("Could not construct a unique portable-stencil assembly")


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    style = str(rng.choice(['scale', 'fragmentation', 'repetition', 'orientation']))
    palette = rng.permutation(10)
    exchange = bool(rng.random() < .75)
    rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            training = split == 'train'
            rot = (rotation+1) % 4 if not training and style == 'orientation' else rotation
            if style != 'orientation' and index >= 2:
                rot = int(rng.integers(4))
            local_palette = rng.permutation(10) if exchange else palette
            for _ in range(8):
                pair = _pair(rng, training, index, style, local_palette, rot)
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not produce distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "Duplicate inputs")
            seen.add(_key(inp))
            a = _parse(inp)
            board, bg = a['board'], a['bg']
            _require(out.shape == board.shape, "Output is not the inclusive board crop")
            fixed = board != bg
            _require(np.array_equal(board[fixed], out[fixed]), "Fixed scaffold changed")
            _require(not np.any(out == bg), "Board has unfilled gaps")
            expected_counts = {}
            for s, n in a['counts'].items():
                t = a['mapping'][s]
                expected_counts[t] = expected_counts.get(t, 0)+n*int(a['masks'][s].sum())
            _require({int(c): int(n) for c, n in zip(*np.unique(out[~fixed], return_counts=True))}
                     == expected_counts, "Decoded footprint/color inventory was not conserved")
            answers, _ = _assemble(board, bg, a['masks'], a['counts'], a['mapping'])
            _require(len(answers) == 1 and np.array_equal(out, answers[0]), "Not the unique colored exact cover")
            if split == 'train' and index < 2:
                reusable, _ = _assemble(board, bg, a['masks'], a['counts'], a['mapping'], unlimited=True)
                _require(len(reusable) == 2, "Demonstration does not distinguish finite inventory from reusable templates")
                unpainted, _ = _assemble(board, bg, a['masks'], a['counts'], {s: s for s in a['masks']})
                _require(not unpainted or any(not np.array_equal(out, x) for x in unpainted), "Legend paint mapping is redundant")
                raw = {s: np.ones((2, 2), bool) for s in a['masks']}
                raw_answers, _ = _assemble(board, bg, raw, a['counts'], a['mapping'])
                _require(not raw_answers or all(not np.array_equal(out, x) for x in raw_answers), "Stencil decoding is redundant")
    return True


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(p['input'], np.int8)), _key(np.asarray(p['output'], np.int8)))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed coverage, semantic checks, and input-only metamorphisms."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, inputs, outputs, masks, palettes, counts = set(), set(), set(), set(), set(), set()
    fragmented = overlapping_maps = shared = 0
    for puzzle in puzzles:
        signatures.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                a = _parse(p['input'])
                inputs.add(p['input'].shape)
                outputs.add(p['output'].shape)
                masks.update(_key(m.astype(np.int8)) for m in a['masks'].values())
                palettes.add((a['bg'], a['frame'], a['marker']))
                counts.add(sum(a['counts'].values()))
                fragmented += any(len(_components(m)) > 1 for m in a['masks'].values())
                overlapping_maps += bool(set(a['mapping']) & set(a['mapping'].values()))
                shared += any(len(set(map(int, p['output'][tuple(np.asarray(comp).T)]))) > 1
                              for comp in _components(a['board'] == a['bg']))
                before = p['input'].copy()
                _require(np.array_equal(solve(before), p['output']) and np.array_equal(before, p['input']), "Solve mutated its input")
    _require(len(signatures) == 42, "Seeds fail to vary whole puzzles")
    _require(len(inputs) >= 15 and len(outputs) >= 25 and len(masks) >= 60, "Insufficient geometry variation")
    _require(len(palettes) >= 25 and len(counts) >= 5, "Insufficient palette/inventory variation")
    _require(fragmented >= 10 and overlapping_maps >= 40 and shared >= 42, "Missing family mechanisms")
    _require(_puzzle_key(generate_puzzle(17)) == _puzzle_key(puzzles[17]), "Seed is not reproducible")
    # All legal explicit counts, plus malformed values, are bounded and checked.
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1000+tr*10+te, tr, te)
            _require(len(p['train']) == tr and len(p['test']) == te, "Count request ignored")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Whole-scene rotations and arbitrary recoloring preserve the rule, including
    # glyph singleton direction, displayed footprint orientation, and the crop.
    rng = np.random.default_rng(932)
    for puzzle in puzzles[:3]:
        for p in puzzle['train'][:2]+puzzle['test']:
            for turn in (1, 2, 3):
                _require(np.array_equal(solve(np.rot90(p['input'], turn)), np.rot90(p['output'], turn)), "Scene-rotation failure")
            permutation = rng.permutation(10)
            _require(np.array_equal(solve(permutation[p['input']]), permutation[p['output']]), "Palette-role failure")
    # Inventory deletion and fixed-cell corruption must fail, and ambiguous
    # color swaps must differ from harmless exchanges of identical tokens.
    p = puzzles[0]['train'][0]
    a = _parse(p['input'])
    altered = p['input'].copy()
    r, c, e, d, _ = a['tokens'][0]
    altered[r:e, c:d] = a['bg']
    try:
        solve(altered)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing inventory token was accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    bad['train'][0]['output'][0, 0] = a['bg']
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Fixed-cell corruption was accepted')
    board = np.array([[0, 9, 0]], np.int8)
    unit = np.ones((1, 1), bool)
    _require(len(_assemble(board, 0, {1: unit, 2: unit}, {1: 1, 2: 1}, {1: 3, 2: 4})[0]) == 2, "Color-swap ambiguity missed")
    _require(len(_assemble(board, 0, {1: unit}, {1: 2}, {1: 3})[0]) == 1, "Identical-token exchanges treated as ambiguity")
    return {'seeds': 42, 'pairs': 210, 'distinct_puzzles': len(signatures),
            'input_shapes': len(inputs), 'output_shapes': len(outputs),
            'footprints': len(masks), 'palette_roles': len(palettes),
            'inventory_counts': sorted(counts), 'fragmented_pairs': fragmented,
            'overlapping_map_pairs': overlapping_maps, 'shared_multicolor_gap_pairs': shared}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
