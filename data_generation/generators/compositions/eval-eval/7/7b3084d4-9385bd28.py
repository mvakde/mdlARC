"""Jigsaw printing plates: rigid material determines overlapping ink envelopes.

No parent imports, files, demonstrations, or random state are used by solve.
All diagnostics are in memory; this module writes no files.
"""
from __future__ import annotations

from math import isqrt
import numpy as np

_EDGE = ((-1, 0), (0, -1), (0, 1), (1, 0))
_NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'Invalid grid size')
    _check(np.issubdtype(a.dtype, np.integer), 'Integer cells required')
    _check(bool(np.all((a >= 0) & (a <= 9))), 'Invalid color')
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask, neighbors=_NEAR):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _tight(board, mask):
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0) + 1
    return np.where(mask[r0:r1, c0:c1], board[r0:r1, c0:c1], -1).astype(np.int8)


def _rotations(piece):
    result, seen = [], set()
    for k in range(4):
        p = np.rot90(piece, k).copy()
        key = _key(p)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _read(input_grid):
    """The compact lookup card is the only multicarrier component."""
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _check(int(np.sum(counts == counts[bg])) == 1, 'Background is ambiguous')
    comps = _components(a != bg)
    candidates = []
    for cells in comps:
        pts = np.array(cells)
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0) + 1)
        if not (3 <= r1-r0 <= 5 and c1-c0 == 2):
            continue
        card = a[r0:r1, c0:c1]
        cues = list(map(int, card[:, 0]))
        if bg in cues or len(set(cues)) != len(cues):
            continue
        rest = [x for x in comps if x is not cells]
        if len(rest) != len(cues):
            continue
        scene = a.copy()
        scene[r0:r1, c0:c1] = bg
        extra = set(map(int, np.unique(scene))) - {bg, *cues}
        if len(extra) != 1:
            continue
        marker = extra.pop()
        if int(np.sum(scene == marker)) != 1 or marker in card:
            continue
        pieces, carriers, okay = [], [], True
        for obj in rest:
            mask = np.zeros(a.shape, dtype=bool)
            for r, c in obj:
                mask[r, c] = True
            vals = set(map(int, a[mask])) - {marker}
            if len(vals) != 1 or next(iter(vals)) not in cues or len(obj) < 3:
                okay = False
                break
            carriers.append(vals.pop())
            pieces.append(_tight(a, mask))
        area = sum(int(np.sum(p >= 0)) for p in pieces)
        side = isqrt(area)
        if not okay or set(carriers) != set(cues) or side*side != area:
            continue
        candidates.append(dict(background=bg, marker=marker, pieces=pieces,
                               carriers=carriers, side=side, cues=cues,
                               fills=list(map(int, card[:, 1])),
                               card=(r0, c0, r1, c1), input=a))
    _check(len(candidates) == 1, 'Require a unique lookup card and rigid pieces')
    return candidates[0]


def _assemble(pieces, side, marker, allow_rotations=True):
    """Exact cover; return at most two distinct colored square assemblies."""
    options, marked = [], []
    for i, piece in enumerate(pieces):
        is_anchor = bool(np.any(piece == marker))
        if is_anchor:
            marked.append(i)
        seen, placements = set(), []
        for p in (_rotations(piece) if allow_rotations else [piece]):
            h, w = p.shape
            if h > side or w > side:
                continue
            if is_anchor and p[0, 0] != marker:
                continue
            coords = [(int(r)*side+int(c), int(p[r, c])) for r, c in np.argwhere(p >= 0)]
            base = sum(1 << pos for pos, _ in coords)
            offsets = [0] if is_anchor else [r*side+c for r in range(side-h+1) for c in range(side-w+1)]
            for offset in offsets:
                mask = base << offset
                if mask not in seen:
                    seen.add(mask)
                    placements.append((mask, offset, coords))
        if not placements:
            return []
        options.append(placements)
    if len(marked) != 1:
        return []
    full = (1 << (side*side))-1
    chosen, results, signatures = {}, [], set()

    def search(used, remaining):
        if len(results) == 2:
            return
        if not remaining:
            if used != full:
                return
            result = np.empty((side, side), dtype=np.int8)
            for _, offset, coords in chosen.values():
                for pos, color in coords:
                    result.flat[pos+offset] = color
            key = _key(result)
            if key not in signatures:
                signatures.add(key)
                results.append(result)
            return
        bit = (full ^ used) & -(full ^ used)
        for i in remaining:
            tail = tuple(j for j in remaining if j != i)
            for p in options[i]:
                if p[0] & bit and not p[0] & used:
                    chosen[i] = p
                    search(used | p[0], tail)
                    if len(results) == 2:
                        return
            chosen.pop(i, None)

    anchor = marked[0]
    for p in options[anchor]:
        chosen[anchor] = p
        search(p[0], tuple(i for i in range(len(pieces)) if i != anchor))
        if len(results) == 2:
            break
    return results


def _layers(board, facts):
    marker = facts['marker']
    anchor_piece = next(p for p in facts['pieces'] if np.any(p == marker))
    anchor = next(int(c) for c in np.unique(anchor_piece) if c >= 0 and c != marker)
    result = []
    for cue, fill in zip(facts['cues'], facts['fills']):
        mask = (board == cue) | ((board == marker) if cue == anchor else False)
        pts = np.argwhere(mask)
        lo, hi = pts.min(axis=0), pts.max(axis=0)+1
        result.append((mask, tuple(map(int, (*lo, *hi))), fill))
    return result


def _print_board(board, facts, reverse=False, transparent_background=False):
    """Solver traversal chooses the foremost footprint separately per pixel."""
    layers = _layers(board, facts)
    if reverse:
        layers = layers[::-1]
    result = board.copy()
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            for mask, (r0, c0, r1, c1), fill in layers:
                if r0 <= r < r1 and c0 <= c < c1:
                    if mask[r, c]:
                        result[r, c] = board[r, c]
                        break
                    if transparent_background and fill == facts['background']:
                        continue
                    result[r, c] = fill
                    break
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read(input_grid)
    boards = _assemble(facts['pieces'], facts['side'], facts['marker'])
    _check(len(boards) == 1, 'Rigid material must have one anchored assembly')
    return _print_board(boards[0], facts)


def _grow(rng, side, count, style):
    labels = np.full((side, side), -1, dtype=np.int8)
    if style == 0 and count <= 4:
        corners = [0, side-1, side*(side-1), side*side-1]
        seeds = corners[:count]
    else:
        seeds = [0] + list(map(int, rng.choice(np.arange(1, side*side), count-1, replace=False)))
    weights = np.exp(rng.uniform(-.7, .7, count))
    frontiers = [set() for _ in range(count)]
    sizes = [1]*count
    for i, pos in enumerate(seeds):
        labels.flat[pos] = i
    def neighbors(pos):
        r, c = divmod(pos, side)
        return [rr*side+cc for dr, dc in _EDGE
                if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side]
    for i, pos in enumerate(seeds):
        frontiers[i].update(p for p in neighbors(pos) if labels.flat[p] < 0)
    for _ in range(side*side-count):
        available = [i for i in range(count) if frontiers[i]]
        i = min(available, key=lambda j: sizes[j]/weights[j] + float(rng.random())*.8)
        choices = sorted(frontiers[i])
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for f in frontiers:
            f.discard(pos)
        frontiers[i].update(p for p in neighbors(pos) if labels.flat[p] < 0)
    return labels


def _cut(rng, side, count):
    """Stepped interfaces supplement free-growth interfaces."""
    rectangles = [(0, 0, side, side)]
    while len(rectangles) < count:
        choices = []
        for i, (r, c, h, w) in enumerate(rectangles):
            choices.extend((i, 0, k) for k in range(1, h) if min(k, h-k)*w >= 4)
            choices.extend((i, 1, k) for k in range(1, w) if min(k, w-k)*h >= 4)
        if not choices:
            return None
        i, axis, k = choices[int(rng.integers(len(choices)))]
        r, c, h, w = rectangles.pop(i)
        rectangles.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                          [(r, c, h, k), (r, c+k, h, w-k)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r+h, c:c+w] = i
    for _ in range(side*4):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = _EDGE[int(rng.integers(4))]
        rr, cc = r+dr, c+dc
        if (r, c) == (0, 0) or not (0 <= rr < side and 0 <= cc < side):
            continue
        old, new = int(labels[r, c]), int(labels[rr, cc])
        if old == new or np.sum(labels == old) <= 4:
            continue
        labels[r, c] = new
        if len(_components(labels == old, _EDGE)) != 1:
            labels[r, c] = old
    return labels


def _scatter(rng, pieces, bg, marker, anchor_rotation, card):
    oriented = []
    for p in pieces:
        k = anchor_rotation if np.any(p == marker) else int(rng.integers(4))
        oriented.append(np.rot90(p, k).copy())
    oriented.append(np.where(card != bg, card, -1).astype(np.int8))
    masks = []
    for i, p in enumerate(oriented):
        # Reserve the whole card, including background swatches, so permuting
        # rows cannot join the card to a neighboring plate.
        occupied = np.ones(p.shape, dtype=bool) if i == len(pieces) else p >= 0
        masks.append([(int(r), int(c)) for r, c in np.argwhere(occupied)])
    for attempt in range(12):
        canvas = np.full((30, 30), bg, dtype=np.int8)
        blocked, maxr, maxc = 0, 0, 0
        order = list(map(int, rng.permutation(len(oriented))))
        if attempt == 0:
            order.sort(key=lambda i: -oriented[i].size)
        good = True
        for i in order:
            p, cells = oriented[i], masks[i]
            h, w = p.shape
            base = sum(1 << (r*30+c) for r, c in cells)
            options = []
            for pos in rng.permutation((31-h)*(31-w))[:220]:
                r, c = divmod(int(pos), 31-w)
                mask = base << (r*30+c)
                if mask & blocked:
                    continue
                hr, wc = max(maxr, r+h), max(maxc, c+w)
                score = hr*wc + abs(hr-wc)*2 + float(rng.random())*30
                options.append((score, r, c))
            if not options:
                good = False
                break
            _, r, c = min(options)
            for pr, pc in cells:
                rr, cc = r+pr, c+pc
                canvas[rr, cc] = bg if p[pr, pc] < 0 else p[pr, pc]
                for nr in range(max(0, rr-1), min(30, rr+2)):
                    for nc in range(max(0, cc-1), min(30, cc+2)):
                        blocked |= 1 << (nr*30+nc)
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not good:
            continue
        pts = np.argwhere(canvas != bg)
        lo, hi = pts.min(axis=0), pts.max(axis=0)+1
        crop = canvas[lo[0]:hi[0], lo[1]:hi[1]]
        h, w = crop.shape
        H, W = h+int(rng.integers(min(4, 30-h)+1)), w+int(rng.integers(min(4, 30-w)+1))
        counts = np.bincount(crop.ravel(), minlength=10)
        while H*W-int(np.sum(crop != bg)) <= max(counts[c] for c in range(10) if c != bg):
            if H < 30 and (H <= W or W == 30):
                H += 1
            elif W < 30:
                W += 1
            else:
                good = False
                break
        if not good:
            continue
        a = np.full((H, W), bg, dtype=np.int8)
        r, c = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
        a[r:r+h, c:c+w] = crop
        return a
    return None


def _construction_output(board, cues, fills, marker, anchor):
    """Generation uses bottom-to-top whole-array stamping, unlike solve."""
    answer = board.copy()
    for cue, fill in reversed(list(zip(cues, fills))):
        mask = board == cue
        if cue == anchor:
            mask = mask.copy()
            mask[0, 0] = True
        # Generation knows the partition's anchor identity; solve derives it
        # from the marked rigid object's visible cells.
        pts = np.argwhere(mask)
        lo, hi = pts.min(axis=0), pts.max(axis=0)+1
        view = answer[lo[0]:hi[0], lo[1]:hi[1]]
        view[:] = fill
        answer[mask] = board[mask]
    return answer


def _witnesses(board, facts, output):
    layers = _layers(board, facts)
    visible_inks, covered_carriers = set(), set()
    retained = 0
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            for mask, (r0, c0, r1, c1), fill in layers:
                if r0 <= r < r1 and c0 <= c < c1:
                    if mask[r, c] and output[r, c] == board[r, c]:
                        retained += 1
                    if not mask[r, c] and output[r, c] == fill:
                        visible_inks.add(fill)
                        if output[r, c] != board[r, c]:
                            covered_carriers.add(int(board[r, c]))
                    break
    return dict(inks=len(visible_inks), occluded=len(covered_carriers), retained=retained,
                printing=not np.array_equal(board, output),
                priority=not np.array_equal(output, _print_board(board, facts, reverse=True)),
                opaque=not np.array_equal(output, _print_board(board, facts, transparent_background=True)),
                anchor_matters=not np.array_equal(output, _print_board(np.rot90(board), facts)),
                rotation=not bool(_assemble(facts['pieces'], facts['side'], facts['marker'], False)))


def _make_pair(rng, side, count, bg, marker, rotation, require_opaque=False):
    palette = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, marker)])))
    carriers = palette[:count]
    inks = palette[count:]
    for attempt in range(400):
        style = int(rng.integers(3))
        labels = _cut(rng, side, count) if style == 2 else _grow(rng, side, count, style)
        if labels is None or min(int(np.sum(labels == i)) for i in range(count)) < 4:
            continue
        anchor_label = int(labels[0, 0])
        board = np.empty((side, side), dtype=np.int8)
        for i, cue in enumerate(carriers):
            board[labels == i] = cue
        board[0, 0] = marker
        pieces = [_tight(board, labels == i) for i in range(count)]
        if min(pieces[anchor_label].shape) < 2:
            continue
        solutions = _assemble(pieces, side, marker)
        if len(solutions) != 1 or not np.array_equal(solutions[0], board):
            continue
        order = list(map(int, rng.permutation(count)))
        cues = [carriers[i] for i in order]
        fills = list(map(int, rng.choice(inks[:2]+[bg], count)))
        if require_opaque:
            fills[int(rng.integers(max(1, count-1)))] = bg
        facts = dict(marker=marker, pieces=pieces, side=side, cues=cues, fills=fills, background=bg)
        output = _construction_output(board, cues, fills, marker, carriers[anchor_label])
        if not np.array_equal(output, _print_board(board, facts)):
            raise AssertionError('Construction and pixel traversal disagree')
        witness = _witnesses(board, facts, output)
        if not (witness['inks'] >= 2 and witness['priority'] and witness['anchor_matters']):
            continue
        if require_opaque and not witness['opaque']:
            continue
        card = np.array(list(zip(cues, fills)), dtype=np.int8)
        a = _scatter(rng, pieces, bg, marker, rotation, card)
        if a is None:
            continue
        parsed = _read(a)
        _check(np.array_equal(solve(a), output), 'Scattered input lost its intended meaning')
        return dict(input=a, output=output)
    raise RuntimeError('Bounded construction exhausted; no ambiguous fallback is used')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _count(num_train, 2, 10, 'num_train')
    ne = _count(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    held_rotation = int(rng.integers(4))
    train_rotations = [k for k in (1, 2, 3) if k != held_rotation]
    first_rotation = int(rng.choice(train_rotations))
    base_count = int(rng.integers(3, 6))
    first_side = int(rng.integers(6, 9))
    bg = int(rng.choice([c for c in range(10) if c != marker]))
    first = _make_pair(rng, first_side, base_count, bg, marker, first_rotation, True)
    f = _read(first['input'])
    r0, c0, r1, c1 = f['card']
    second_input = first['input'].copy()
    second_input[r0:r1, c0:c1] = second_input[r0:r1, c0:c1][::-1]
    second = dict(input=second_input, output=solve(second_input))
    puzzle = dict(train=[first, second], test=[])
    seen = {_key(first['input']), _key(second_input)}
    larger = bool(rng.integers(2))
    observed_counts = {base_count}
    for split, n in (('train', nt-2), ('test', ne)):
        for index in range(n):
            if split == 'train':
                count = 3+(base_count-3+index+1) % 3
                observed_counts.add(count)
                side = int(rng.integers(5, 10))
                rotation = int(rng.choice([k for k in range(4) if k != held_rotation]))
            else:
                count = int(rng.choice(sorted(observed_counts)))
                side = int(rng.integers(9, 11)) if larger else int(rng.integers(5, 10))
                rotation = held_rotation
            bg = int(rng.choice([c for c in range(10) if c != marker]))
            for _ in range(10):
                pair = _make_pair(rng, side, count, bg, marker, rotation)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('Could not obtain distinct inputs')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'Illegal counts')
    seen, markers, train_witnesses = set(), set(), []
    corners, counts = {'train': set(), 'test': set()}, {'train': set(), 'test': set()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _check(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            f = _read(inp)
            markers.add(f['marker'])
            marked = next(p for p in f['pieces'] if np.any(p == f['marker']))
            _check(min(marked.shape) >= 2, 'Generated anchor has two-dimensional extent')
            mr, mc = map(int, np.argwhere(marked == f['marker'])[0])
            _check(mr in (0, marked.shape[0]-1) and mc in (0, marked.shape[1]-1), 'Anchor is at a material bounding corner')
            corners[split].add((mr == 0, mc == 0))
            counts[split].add(len(f['pieces']))
            _check(3 <= len(f['pieces']) <= 5, 'Plate count')
            _check(4 <= f['side'] <= 12, 'Supported assembly side')
            boards = _assemble(f['pieces'], f['side'], f['marker'])
            _check(len(boards) == 1, 'Assembly must be unique before printing')
            board = boards[0]
            scene_counts = np.bincount(inp.ravel(), minlength=10)
            r0, c0, r1, c1 = f['card']
            scene_counts -= np.bincount(inp[r0:r1, c0:c1].ravel(), minlength=10)
            for color in f['carriers']+[f['marker']]:
                _check(scene_counts[color] == np.sum(board == color), 'Assembly must conserve physical colors')
            _check(out.shape == board.shape, 'Output size follows material area')
            _check(np.array_equal(_print_board(board, f), out), 'Incorrect print')
            _check(np.array_equal(solve(inp), out), 'Input-only solve disagrees')
            w = _witnesses(board, f, out)
            _check(w['printing'] and w['priority'], 'Both assembly and priority must affect answer')
            _check(w['retained'] >= 3 and w['occluded'] > 0, 'Retained material and overprinted foreign material must be visible')
            if split == 'train':
                train_witnesses.append(w)
    _check(len(markers) == 1, 'Consistent anchor marker role')
    _check(len(corners['test']) == 1 and not corners['test'] & corners['train'], 'Tests hold out one marker corner')
    _check(counts['test'] <= counts['train'], 'Test counts must have demonstrations')
    _check(train_witnesses[0]['opaque'] and train_witnesses[0]['inks'] >= 2, 'First demonstration teaches ink and opaque clearing')
    for feature in ('rotation', 'opaque', 'anchor_matters'):
        _check(any(w[feature] for w in train_witnesses), f'Missing teaching witness: {feature}')
    # Controlled contrast: same physical pieces, only the ordered table changes.
    p, q = puzzle['train'][:2]
    f, g = _read(p['input']), _read(q['input'])
    _check(f['card'] == g['card'], 'Controlled contrast card position')
    r0, c0, r1, c1 = f['card']
    x, y = f['input'].copy(), g['input'].copy()
    x[r0:r1, c0:c1] = f['background']
    y[r0:r1, c0:c1] = g['background']
    _check(np.array_equal(x, y), 'First two examples isolate priority')
    _check(dict(zip(f['cues'], f['fills'])) == dict(zip(g['cues'], g['fills'])), 'Contrast preserves lookup')
    _check(not np.array_equal(p['output'], q['output']), 'Priority contrast must be visible')


def self_test():
    """42 seeds, streamed metrics only; no files, caches or retained puzzles."""
    sides, counts, shapes, backgrounds, signatures, structures = set(), set(), set(), set(), set(), set()
    opaque = rotation = occlusion = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(x['input']) for s in ('train', 'test') for x in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                f = _read(pair['input'])
                board = _assemble(f['pieces'], f['side'], f['marker'])[0]
                w = _witnesses(board, f, pair['output'])
                sides.add(f['side'])
                counts.add(len(f['pieces']))
                shapes.add(pair['input'].shape)
                backgrounds.add(f['background'])
                opaque += w['opaque']
                rotation += w['rotation']
                occlusion += w['occluded'] > 0
                roles = {}
                structures.add((board.shape, tuple(roles.setdefault(int(c), len(roles)) for c in board.flat)))
    _check(len(signatures) == 42 and len(structures) >= 140, 'Structural variation across seeds')
    _check(counts == {3, 4, 5} and len(sides) >= 5 and len(shapes) >= 30, 'Geometric variation')
    _check(backgrounds == set(range(10)), 'All backgrounds')
    _check(opaque >= 42 and rotation >= 100 and occlusion >= 150, 'Semantic variation')
    p, q = generate_puzzle(803, 2, 1), generate_puzzle(803, 2, 1)
    _check(all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x), 'Determinism')
    # Move and independently rotate the same complete physical objects.
    # The card and all semantic roles stay fixed, so printing must be invariant.
    f = _read(p['train'][0]['input'])
    card = np.array(list(zip(f['cues'], f['fills'])), dtype=np.int8)
    moved = _scatter(np.random.default_rng(9001), f['pieces'], f['background'],
                     f['marker'], 0, card)
    _check(moved is not None, 'Directed rescattering must fit')
    _check(np.array_equal(solve(moved), p['train'][0]['output']), 'Scattered layout and initial angles must not affect printing')
    # Non-sampled but compatible visible swatches: another carrier and self-ink.
    altered = p['train'][0]['input'].copy()
    r0, c0, r1, c1 = f['card']
    new_fills = [f['cues'][1], f['cues'][1]]+[f['background']]*(len(f['cues'])-2)
    altered[r0:r1, c1-1] = new_fills
    board = _assemble(f['pieces'], f['side'], f['marker'])[0]
    anchor_piece = next(x for x in f['pieces'] if np.any(x == f['marker']))
    anchor = next(int(c) for c in np.unique(anchor_piece) if c >= 0 and c != f['marker'])
    expected = _construction_output(board, f['cues'], new_fills, f['marker'], anchor)
    before = altered.copy()
    _check(np.array_equal(solve(altered), expected), 'Compatible carrier-valued inks')
    _check(np.array_equal(before, altered), 'solve must not mutate its argument')
    validate_generated_puzzle(generate_puzzle(63556605, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(100+nt*5+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Output corruption accepted')
