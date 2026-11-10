"""Wave molds: marked shape matching selects a periodic geodesic fill.

No parent modules, files, labels, seeds, or generation records are used by solve.
The construction computes its labels before input packing; recovery reads pixels.
"""

from collections import Counter, deque
from itertools import permutations
from math import isqrt
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_L = ((0, 0), (1, 0), (2, 0), (2, 1))
_T = ((0, 0), (1, 0), (2, 0), (1, 1))
_DETOUR = ((0, 0), (1, 0), (2, 0), (3, 0),
           (3, 1), (3, 2), (2, 2), (1, 2))
_LONG_U = ((0, 0), (1, 0), (2, 0), (3, 0),
           (3, 1), (3, 2), (2, 2), (1, 2), (0, 2))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(1 <= n <= 30 for n in g.shape)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("grids must be integer 2D arrays, sides 1..30, colors 0..9")
    return g.astype(np.int8, copy=False)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells, queue = {first}, [first]
        while queue:
            r, c = queue.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    queue.append(p)
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _normal(shape, anchor):
    r0, c0, _, _ = _box(shape)
    return (tuple(sorted((r - r0, c - c0) for r, c in shape)),
            (anchor[0] - r0, anchor[1] - c0))


def _variants(shape, anchor, reflect=True):
    found = set()
    for flip in range(2 if reflect else 1):
        cells = [(r, -c if flip else c) for r, c in shape]
        a = (anchor[0], -anchor[1] if flip else anchor[1])
        for _ in range(4):
            found.add(_normal(cells, a))
            cells = [(c, -r) for r, c in cells]
            a = (a[1], -a[0])
    return tuple(sorted(found))


def _class(shape, anchor):
    return _variants(shape, anchor)[0]


def _distances(shape, anchor):
    cells = set(shape)
    if anchor not in cells:
        raise ValueError("inlet outside mold")
    distances, queue = {anchor: 0}, deque([anchor])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            p = (r + dr, c + dc)
            if p in cells and p not in distances:
                distances[p] = distances[(r, c)] + 1
                queue.append(p)
    if len(distances) != len(cells):
        raise ValueError("disconnected mold")
    return distances


def _cyclic_word(word):
    """Reduce only complete copies. ABA remains a three-step cycle."""
    word = tuple(map(int, word))
    for p in range(1, len(word) + 1):
        if len(word) % p == 0 and all(v == word[i % p] for i, v in enumerate(word)):
            return word[:p]
    raise ValueError("empty code")


def _block_grid(crop, scale):
    h, w = crop.shape
    if h % scale or w % scale:
        raise ValueError("partial macrocell")
    blocks = crop.reshape(h // scale, scale, w // scale, scale)
    logical = blocks[:, 0, :, 0]
    if not np.all(blocks == logical[:, None, :, None]):
        raise ValueError("nonuniform macrocell")
    return logical


def _parse_key(g, cells, bg, marker, hub, scale):
    r0, c0, r1, c1 = _box(cells)
    logical = _block_grid(g[r0:r1, c0:c1], scale)
    anchors = np.argwhere(logical == marker)
    if len(anchors) != 1:
        raise ValueError("each key has one macrocell inlet")
    a = tuple(map(int, anchors[0]))
    shape = set(map(tuple, np.argwhere((logical == hub) | (logical == marker)).tolist()))
    if len(shape) < 3 or len(_components((logical == hub) | (logical == marker))) != 1:
        raise ValueError("invalid key body")
    tail = set(map(tuple, np.argwhere((logical != bg) & (logical != hub)
                                    & (logical != marker)).tolist()))
    if not tail:
        raise ValueError("missing repeat code")
    choices = []
    for direction, (dr, dc) in enumerate(_DIRS):
        expected = {(a[0] + i * dr, a[1] + i * dc) for i in range(1, len(tail) + 1)}
        if tail != expected:
            continue
        # The inlet is on an extreme face; the code projects away from the body.
        if any((r - a[0]) * dr + (c - a[1]) * dc > 0 for r, c in shape):
            continue
        word = tuple(int(logical[a[0] + i * dr, a[1] + i * dc])
                     for i in range(1, len(tail) + 1))
        choices.append((direction, word))
    if len(choices) != 1:
        raise ValueError("code must be a single outward strip")
    direction, visible = choices[0]
    shape, a = _normal(shape, a)
    return dict(shape=shape, anchor=a, cls=_class(shape, a),
                word=_cyclic_word(visible), visible=visible,
                direction=direction, bounds=(r0, c0, r1, c1))


def _recover(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= g.size // 2:
        raise ValueError("background must occupy more than half the canvas")
    boards = []
    for structural in range(10):
        if structural == bg:
            continue
        for cells in _components(g == structural):
            r0, c0, r1, c1 = _box(cells)
            b = g[r0:r1, c0:c1]
            if min(b.shape) < 4:
                continue
            if not (np.all(b[0] == structural) and np.all(b[-1] == structural)
                    and np.all(b[:, 0] == structural) and np.all(b[:, -1] == structural)):
                continue
            other = set(map(int, np.unique(b))) - {bg, structural}
            if len(other) != 1:
                continue
            marker = next(iter(other))
            holes = _components(b != structural)
            if len(holes) < 2 or any(sum(b[p] == marker for p in h) != 1 for h in holes):
                continue
            outside = np.ones(g.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            moat = np.zeros(g.shape, dtype=bool)
            moat[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] = True
            if np.any((g != bg) & moat & outside):
                continue
            objects = _components((g != bg) & outside)
            if len(objects) != len(holes):
                continue
            boards.append((structural, marker, (r0, c0, r1, c1), b.copy(), holes, objects))
    if len(boards) != 1:
        raise ValueError("expected one marked blueprint")
    structural, marker, bounds, board, holes, objects = boards[0]
    scales = []
    for cells in objects:
        anchors = {p for p in cells if g[p] == marker}
        if not anchors:
            raise ValueError("unmarked loose object")
        k = isqrt(len(anchors))
        ar0, ac0, ar1, ac1 = _box(anchors)
        if k < 2 or (ar1 - ar0, ac1 - ac0) != (k, k) or len(anchors) != k * k:
            raise ValueError("loose inlet must be a solid square macrocell")
        scales.append(k)
    if len(set(scales)) != 1:
        raise ValueError("keys must share a common scale")
    scale = scales[0]
    if max(board.shape) * scale > 30:
        raise ValueError("enlarged board exceeds 30")
    cavities = []
    for cells in holes:
        anchor = next(p for p in cells if board[p] == marker)
        shape, a = _normal(cells, anchor)
        cavities.append(dict(cells=cells, anchor=anchor, shape=shape, local_anchor=a,
                             cls=_class(shape, a), distances=_distances(cells, anchor)))
    common_colors = set.intersection(*(set(int(g[p]) for p in cells) - {marker}
                                       for cells in objects)) - {bg, structural}
    accepted = []
    for hub in sorted(common_colors):
        try:
            keys = [_parse_key(g, cells, bg, marker, hub, scale) for cells in objects]
            if Counter(k['cls'] for k in keys) != Counter(c['cls'] for c in cavities):
                continue
            words = {}
            for key in keys:
                cls, word = key['cls'], key['word']
                if cls in words and words[cls] != word:
                    raise ValueError("interchangeable keys have conflicting codes")
                words[cls] = word
            accepted.append((hub, keys, words))
        except ValueError:
            continue
    if len(accepted) != 1:
        raise ValueError("ambiguous or missing shape/code assignment")
    hub, keys, words = accepted[0]
    logical = board.copy()
    for cavity in cavities:
        word = words[cavity['cls']]
        for p, d in cavity['distances'].items():
            logical[p] = word[d % len(word)]
    output = np.repeat(np.repeat(logical, scale, 0), scale, 1)
    return output, dict(background=bg, structural=structural, marker=marker, hub=hub,
                        scale=scale, bounds=bounds, board=board, cavities=cavities,
                        keys=keys, words=words, logical=logical)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer local roles, scale, matching, and geodesic phase from this input."""
    return _recover(input_grid)[0]


def _grow(rng, area, span=5):
    cells = {(0, 0)}
    style = int(rng.integers(3))
    for _ in range(area - 1):
        options = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS} - cells)
        options = [p for p in options if max(_box(cells | {p})[2] - _box(cells | {p})[0],
                                            _box(cells | {p})[3] - _box(cells | {p})[1]) <= span]
        if not options:
            break
        weights = []
        for r, c in options:
            degree = sum((r + dr, c + dc) in cells for dr, dc in _DIRS)
            weights.append(3.0 ** degree if style == 0 else 1.0 / degree if style == 1 else 1.0)
        weights = np.asarray(weights) / sum(weights)
        cells.add(options[int(rng.choice(len(options), p=weights))])
    shape, _ = _normal(cells, min(cells))
    h, w = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
    tips = [p for p in shape if (p[0] in (0, h - 1) or p[1] in (0, w - 1))
            and max(_distances(shape, p).values()) >= 2]
    anchor = tips[int(rng.integers(len(tips)))]
    return shape, anchor


def _random_orientation(rng, shape, anchor):
    vs = _variants(shape, anchor)
    return vs[int(rng.integers(len(vs)))]


def _place_holes(rng, shapes, h, w):
    for _ in range(18):
        occupied, placed = set(), [None] * len(shapes)
        order = sorted(range(len(shapes)), key=lambda i: (-len(shapes[i][0]), float(rng.random())))
        for i in order:
            shape, anchor = _random_orientation(rng, *shapes[i])
            sh, sw = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
            origins = [(r, c) for r in range(1, h - sh) for c in range(1, w - sw)]
            for j in rng.permutation(len(origins)):
                r0, c0 = origins[int(j)]
                cells = {(r + r0, c + c0) for r, c in shape}
                if any(p in occupied or any((p[0] + dr, p[1] + dc) in occupied for dr, dc in _DIRS)
                       for p in cells):
                    continue
                placed[i] = (cells, (anchor[0] + r0, anchor[1] + c0))
                occupied.update(cells)
                break
            if placed[i] is None:
                break
        else:
            return placed
    return None


def _render_key(rng, shape, anchor, word, palette, scale):
    bg, structural, hub, marker = palette[:4]
    h, w = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
    directions = [i for i, (dr, dc) in enumerate(_DIRS)
                  if not any((r - anchor[0]) * dr + (c - anchor[1]) * dc > 0 for r, c in shape)]
    direction = directions[int(rng.integers(len(directions)))]
    dr, dc = _DIRS[direction]
    tail = [(anchor[0] + i * dr, anchor[1] + i * dc) for i in range(1, len(word) + 1)]
    r0, c0, r1, c1 = _box(set(shape) | set(tail))
    logical = np.full((r1 - r0, c1 - c0), bg, dtype=np.int8)
    for r, c in shape:
        logical[r - r0, c - c0] = hub
    logical[anchor[0] - r0, anchor[1] - c0] = marker
    for (r, c), color in zip(tail, word):
        logical[r - r0, c - c0] = color
    return np.repeat(np.repeat(logical, scale, 0), scale, 1)


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    s = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return np.argwhere(s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w] == 0)


def _pack(rng, board, keys, bg):
    objects = [board] + keys
    for attempt in range(24):
        min_h = max(14, max(a.shape[0] for a in objects))
        min_w = max(14, max(a.shape[1] for a in objects))
        if max(min_h, min_w) > 30:
            return None
        h, w = ((int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31)))
                if attempt < 12 else (30, 30))
        canvas, blocked = np.full((h, w), bg, dtype=np.int8), np.zeros((h, w), bool)
        order = [0] + sorted(range(1, len(objects)), key=lambda i: -objects[i].size)
        for i in order:
            a = objects[i]
            ah, aw = a.shape
            origins = _free_origins(blocked, ah, aw)
            if not len(origins):
                break
            r, c = map(int, origins[int(rng.integers(len(origins)))])
            canvas[r:r + ah, c:c + aw] = a
            blocked[max(0, r - 1):r + ah + 1, max(0, c - 1):c + aw + 1] = True
        else:
            if np.count_nonzero(canvas == bg) > canvas.size // 2:
                return canvas
    # A finite rectangle-packing escape path. Random placement is the normal path.
    for order in permutations(range(len(objects))):
        rows, current, used, row_h = [], [], 0, 0
        for i in order:
            ah, aw = objects[i].shape
            if used and used + 1 + aw > 30:
                rows.append((current, used, row_h))
                current, used, row_h = [], 0, 0
            c = used + (1 if used else 0)
            current.append((i, c))
            used, row_h = c + aw, max(row_h, ah)
        rows.append((current, used, row_h))
        if sum(row[2] for row in rows) + len(rows) - 1 > 30:
            continue
        canvas = np.full((30, 30), bg, dtype=np.int8)
        r = int(rng.integers(31 - sum(row[2] for row in rows) - len(rows) + 1))
        for row, used, row_h in rows:
            shift = int(rng.integers(31 - used))
            for i, c in row:
                a = objects[i]
                canvas[r:r + a.shape[0], shift + c:shift + c + a.shape[1]] = a
            r += row_h + 1
        if np.count_nonzero(canvas == bg) > canvas.size // 2:
            return canvas
    return None


def _alternative(info, mode):
    logical = info['board'].copy()
    cavities = info['cavities']
    words = [info['words'][c['cls']] for c in cavities]
    if mode == 'position':
        words = [k['word'] for k in sorted(info['keys'], key=lambda k: k['bounds'][:2])]
        cavities = sorted(cavities, key=lambda c: min(c['cells']))
    for cavity, word in zip(cavities, words):
        if mode == 'prefix':
            p = next(p for p in range(1, len(word) + 1)
                     if all(v == word[i % p] for i, v in enumerate(word)))
            word = word[:p]
        for point, distance in cavity['distances'].items():
            if mode == 'flat':
                distance = 0
            elif mode == 'manhattan':
                distance = sum(abs(x - y) for x, y in zip(point, cavity['anchor']))
            logical[point] = word[distance % len(word)]
    k = info['scale']
    return np.repeat(np.repeat(logical, k, 0), k, 1)


def _make_pair(rng, palette, scale, count, witness=None, large=False, max_area=9):
    bg, structural, hub, marker = map(int, palette[:4])
    payload = list(map(int, palette[4:]))
    for attempt in range(80):
        simple = attempt >= 35 or count >= 4 or scale == 4
        if witness == 'matching':
            shapes = [(_L, (0, 0)), (_T, (0, 0))]
        elif witness == 'word':
            shapes = [(_DETOUR, (0, 0)), (_L, (0, 0))]
        elif large:
            extra = int(rng.integers(4))
            long_shape = _LONG_U + tuple((-i, 0) for i in range(1, extra + 1))
            long_shape, inlet = _normal(long_shape, (-extra, 0))
            shapes = [(long_shape, inlet)] + [_grow(rng, int(rng.integers(4, 7))) for _ in range(count - 1)]
        else:
            shapes = [_grow(rng, int(rng.integers(4, 6 if simple else max_area + 1)), 3 if simple else 5)
                      for _ in range(count)]
        if not witness and rng.random() < .25 and len(shapes) >= 3:
            shapes[-1] = shapes[0]  # interchangeable keys must carry the same code
        if attempt >= 60 and not witness:
            shapes = [(_LONG_U if large and i == 0 else _L if i % 2 == 0 else _T, (0, 0))
                      for i in range(count)]
        limit = 30 // scale
        low = max(7, 9 if count >= 3 else 7)
        if low > limit:
            continue
        h, w = int(rng.integers(low, limit + 1)), int(rng.integers(low, limit + 1))
        if rng.random() < .18:
            w = h
        placements = _place_holes(rng, shapes, h, w)
        if placements is None:
            continue
        codes = {}
        for i, (shape, anchor) in enumerate(shapes):
            cls = _class(shape, anchor)
            if cls in codes:
                continue
            if witness == 'matching':
                word = (payload[2 * i], payload[2 * i + 1])
            elif witness == 'word' and i == 0:
                word = (payload[0], payload[1], payload[0])
            elif i == 0:
                reach = max(_distances(shape, anchor).values())
                length = int(rng.integers(2, min(2 if simple else 4, reach) + 1))
                colors = payload[:int(rng.integers(2, 7))]
                word = list(map(int, rng.choice(colors, size=length)))
                word[1] = int(rng.choice([v for v in colors if v != word[0]]))
                word = tuple(word)
            else:
                length = int(rng.integers(1, 3 if simple else 5))
                colors = payload[:int(rng.integers(2, 7))]
                word = tuple(map(int, rng.choice(colors, size=length)))
            codes[cls] = word
        board, logical = np.full((h, w), structural, dtype=np.int8), np.full((h, w), structural, dtype=np.int8)
        key_grids = []
        for i, ((cells, anchor), (shape, local_anchor)) in enumerate(zip(placements, shapes)):
            word = codes[_class(shape, local_anchor)]
            for p in cells:
                board[p] = bg
            board[anchor] = marker
            for p, distance in _distances(cells, anchor).items():
                logical[p] = word[distance % len(word)]
            dest, dest_a = _normal(cells, anchor)
            orientations = _variants(dest, dest_a)
            if witness == 'matching' and i == 0:
                rotations = _variants(dest, dest_a, False)
                reflected = [v for v in orientations if v not in rotations]
                orientations = reflected or orientations
            loose_shape, loose_anchor = orientations[int(rng.integers(len(orientations)))]
            key_grids.append(_render_key(rng, loose_shape, loose_anchor, word, palette, scale))
        canvas = _pack(rng, board, key_grids, bg)
        if canvas is None:
            continue
        output = np.repeat(np.repeat(logical, scale, 0), scale, 1)
        try:
            recovered, info = _recover(canvas)
        except ValueError:
            continue
        if not np.array_equal(recovered, output):
            raise AssertionError("construction and recovery disagree")
        if witness == 'matching' and np.array_equal(_alternative(info, 'position'), output):
            continue
        if witness == 'word' and any(np.array_equal(_alternative(info, m), output)
                                     for m in ('manhattan', 'prefix')):
            continue
        return dict(input=canvas, output=output)
    raise RuntimeError("bounded wave-mold construction exhausted")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime = int(rng.integers(4))
    backgrounds = list(map(int, rng.permutation(10)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for index in range(n):
            serial = index if split == 'train' else num_train + index
            palette = list(map(int, rng.permutation(10)))
            bg = backgrounds[serial % 10]
            palette.remove(bg)
            palette.insert(0, bg)
            witness = ('matching' if index == 0 else 'word' if index == 1 else None) if split == 'train' else None
            if regime == 0:  # new common scale
                scale, count, large = (2 if split == 'train' else 3), 2, False
            elif regime == 1:  # larger geodesic detours, same arithmetic
                scale, count, large = 2, int(rng.integers(2, 4)), split == 'test'
            elif regime == 2:  # more independently matched emitters
                scale, count, large = 2, (2 if split == 'train' else int(rng.integers(3, 5))), False
            else:  # two visible scales; test generalizes to scale four
                scale, count, large = (2 + index % 2 if split == 'train' else 4), 2, False
            if witness:
                count = 2
            for _ in range(10):
                max_area = 8 if regime == 1 and split == 'train' else 9
                pair = _make_pair(rng, palette, scale, count, witness, large, max_area)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            predicted, info = _recover(inp)
            assert np.array_equal(predicted, out), "wrong wave fill"
            assert 2 <= info['scale'] <= 4 and 2 <= len(info['cavities']) <= 4
            assert all(4 <= len(c['cells']) <= 12 for c in info['cavities'])
            assert all(1 <= len(k['visible']) <= 4 for k in info['keys'])
            assert len({info[r] for r in ('background', 'structural', 'marker', 'hub')}) == 4
            role_colors = {info[r] for r in ('background', 'structural', 'marker', 'hub')}
            assert all(not (set(k['visible']) & role_colors) for k in info['keys'])
            assert info['background'] not in np.unique(out) and info['marker'] not in np.unique(out)
            assert not np.array_equal(_alternative(info, 'flat'), out), "each example must expose A"
            assert any(len(info['words'][c['cls']]) > 1
                       and max(c['distances'].values()) >= len(info['words'][c['cls']])
                       for c in info['cavities']), "show repetition beyond the visible code"
            k, board = info['scale'], info['board']
            assert out.shape == (board.shape[0] * k, board.shape[1] * k)
            compressed = _block_grid(out, k)
            assert np.all(compressed[board == info['structural']] == info['structural'])
            for c in info['cavities']:
                distances = c['distances']
                assert distances[c['anchor']] == 0
                assert all(abs(distances[p] - distances[q]) <= 1 for p in distances for dr, dc in _DIRS
                           if (q := (p[0] + dr, p[1] + dc)) in distances)
            if split == 'train':
                training.append((info, out))
    # These are instructional witnesses, not latent constraints used by solve.
    for mode in ('flat', 'position', 'manhattan', 'prefix'):
        assert any(not np.array_equal(_alternative(info, mode), out) for info, out in training), mode
    assert any(len(x['cells']) == len(y['cells']) and x['cls'] != y['cls']
               and sorted((_box(x['shape'])[2], _box(x['shape'])[3]))
               == sorted((_box(y['shape'])[2], _box(y['shape'])[3]))
               for info, _ in training for i, x in enumerate(info['cavities']) for y in info['cavities'][i + 1:])
    assert any((c['shape'], c['local_anchor']) not in _variants(key['shape'], key['anchor'], False)
               for info, _ in training for c in info['cavities'] for key in info['keys'] if c['cls'] == key['cls'])
    assert len({info['background'] for info, _ in training}) >= 2


def _features(info):
    cavities, keys = info['cavities'], info['keys']
    return dict(scale=info['scale'], board_shape=tuple(info['board'].shape),
                count=len(cavities), classes=tuple(sorted(c['cls'] for c in cavities)),
                areas=tuple(sorted(len(c['cells']) for c in cavities)),
                directions=tuple(sorted(k['direction'] for k in keys)),
                periods=tuple(sorted(len(k['word']) for k in keys)),
                max_distance=max(max(c['distances'].values()) for c in cavities),
                repeated=len({c['cls'] for c in cavities}) < len(cavities),
                branched=any(sum((p[0] + dr, p[1] + dc) in c['cells'] for dr, dc in _DIRS) >= 3
                             for c in cavities for p in c['cells']))


def self_test():
    """Bounded 42-seed coverage, diagnostics, count contracts, determinism."""
    assert _cyclic_word((2, 3, 2)) == (2, 3, 2)
    assert _cyclic_word((2, 3, 2, 3)) == (2, 3)
    signatures, features, shapes = set(), [], set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        split_infos = {s: [_recover(p['input'])[1] for p in puzzle[s]] for s in ('train', 'test')}
        train, test = split_infos['train'], split_infos['test']
        train_scales, test_scales = {i['scale'] for i in train}, {i['scale'] for i in test}
        train_max_area = max(len(c['cells']) for i in train for c in i['cavities'])
        test_min_max_area = min(max(len(c['cells']) for c in i['cavities']) for i in test)
        assert ((train_scales == {2} and test_scales == {3})
                or (train_max_area <= 8 and test_min_max_area >= 9)
                or (max(len(i['cavities']) for i in train) == 2
                    and min(len(i['cavities']) for i in test) >= 3)
                or (train_scales == {2, 3} and test_scales == {4})), "missing visible shift"
        signature = tuple((_key(p['input']), _key(p['output']))
                          for s in ('train', 'test') for p in puzzle[s])
        signatures.add(signature)
        if seed == 0:
            first = puzzle
            again = generate_puzzle(seed)
            assert signature == tuple((_key(p['input']), _key(p['output']))
                                      for s in ('train', 'test') for p in again[s])
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                predicted, info = _recover(pair['input'])
                assert np.array_equal(predicted, pair['output'])
                features.append(_features(info))
                shapes.add(pair['input'].shape)
    assert len(signatures) == 42
    assert len(shapes) >= 35
    assert {f['scale'] for f in features} == {2, 3, 4}
    assert {f['count'] for f in features} >= {2, 3, 4}
    assert {d for f in features for d in f['directions']} == set(range(4))
    assert {p for f in features for p in f['periods']} >= {1, 2, 3, 4}
    assert len({f['board_shape'] for f in features}) >= 30
    assert len({c for f in features for c in f['classes']}) >= 45
    assert any(f['repeated'] for f in features) and any(f['branched'] for f in features)
    assert max(f['max_distance'] for f in features) >= 8
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(321, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    inp = first['train'][0]['input'].copy()
    _, info = _recover(inp)
    marker = info['marker']
    key = info['keys'][0]
    r0, c0, r1, c1 = key['bounds']
    r, c = map(int, np.argwhere(inp[r0:r1, c0:c1] == marker)[0])
    inp[r0 + r, c0 + c] = info['background']
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("partial inlet macrocell accepted")
    # Two congruent marked molds cannot receive two different repeat codes.
    board = np.ones((7, 8), dtype=np.int8)
    for left in (1, 5):
        for r, c in _L:
            board[r + 1, c + left] = 0
        board[1, left] = 3
    diagnostic_rng = np.random.default_rng(19)
    palette = list(range(10))
    key_a = _render_key(diagnostic_rng, _L, (0, 0), (4, 5), palette, 2)
    key_b = _render_key(diagnostic_rng, _L, (0, 0), (6, 7), palette, 2)
    diagnostic = _pack(diagnostic_rng, board, [key_a, key_b], 0)
    assert diagnostic is not None
    try:
        solve(diagnostic)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting codes on interchangeable keys accepted")
    diagnostic[diagnostic == 6] = 4
    diagnostic[diagnostic == 7] = 5
    assert solve(diagnostic).shape == (14, 16)


if __name__ == '__main__':
    self_test()
    print('wave-mold self_test passed (42 seeds)')
