"""Restore a cropped-witness mosaic from a finite inventory of reference tiles.

All references, crop colors, silhouette cells and the panel delimiter are visible
in the input. No reference, seed, or construction placement is kept by solve().
"""

from collections import Counter
import json

import numpy as np


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _d4(a, k):
    return np.rot90(a if k < 4 else np.fliplr(a), k % 4).copy()


def _variants(a):
    seen = set()
    for k in range(8):
        for swap in (0, 1):
            v = _d4(a, k) ^ swap
            key = _key(v)
            if key not in seen:
                seen.add(key)
                yield v, k, swap


def _canonical(a):
    return min(_key(v) for v, _, _ in _variants(a))


def _components(mask, with_cells=False):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    H, W = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(int(r), int(c))], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        ys, xs = zip(*cells)
        record = (min(ys), min(xs), max(ys) + 1, max(xs) + 1, len(cells))
        result.append(record + (cells,) if with_cells else record)
    return result


def _rectangles(panel, background):
    rects = _components(panel != background)
    if not rects:
        raise ValueError("No reference rectangles")
    result = []
    for r, c, y, x, area in rects:
        tile = panel[r:y, c:x]
        if tile.size != area or np.any(tile == background):
            raise ValueError("References must be filled, separate rectangles")
        result.append(tile.copy())
    return result


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape):
        raise ValueError("Grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("Grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _interpretations(grid):
    """Enumerate visible delimiters and local color roles, without output labels."""
    for axis in (0, 1):
        a = grid if axis == 0 else grid.T
        H, W = a.shape
        for cut in range(1, H - 1):
            marker = int(a[cut, 0])
            if not np.all(a[cut] == marker) or np.count_nonzero(a == marker) != W:
                continue
            panels = (a[:cut], a[cut + 1:])
            if axis == 1:
                panels = tuple(p.T for p in panels)
            for board, tray in (panels, panels[::-1]):
                tc, bc = set(map(int, np.unique(tray))), set(map(int, np.unique(board)))
                if len(tc) != 3 or len(bc) != 4:
                    continue
                for erased in sorted(tc & bc):
                    try:
                        tiles = _rectangles(tray, erased)
                    except ValueError:
                        continue
                    source_colors = sorted(tc - {erased})
                    if any(set(map(int, np.unique(t))) != set(source_colors) for t in tiles):
                        continue
                    # The board background is absent from the entire source tray.
                    for background in sorted(bc - tc):
                        payload = sorted(bc - {background, erased})
                        if len(payload) != 2:
                            continue
                        known = (board == payload[0]) | (board == payload[1])
                        patches = []
                        valid = True
                        for r, c, y, x, area in _components(known):
                            patch = board[r:y, c:x]
                            if patch.size != area or set(map(int, np.unique(patch))) != set(payload):
                                valid = False
                                break
                            patches.append((r, c, (patch == payload[1]).astype(np.int8)))
                        if not valid or not patches or not np.any(board == background) or not np.any(board == erased):
                            continue
                        refs = [(t == source_colors[1]).astype(np.int8) for t in tiles]
                        yield dict(board=board, tray=tray, erased=erased, background=background,
                                   payload=payload, source_colors=source_colors, refs=refs,
                                   patches=patches, known=known, support=board != background,
                                   axis=axis, marker=marker, cut=cut,
                                   side=('top' if axis == 0 else 'left') if board is panels[0]
                                   else ('bottom' if axis == 0 else 'right'))


def _placements(a):
    """Joint crop/orientation/role/offset candidates for each inventory group."""
    groups, lookup = [], {}
    for ref in a['refs']:
        key = _key(ref)
        if key in lookup:
            groups[lookup[key]]['count'] += 1
        else:
            lookup[key] = len(groups)
            groups.append(dict(ref=ref, count=1))
    support, known = a['support'], a['known']
    H, W = support.shape
    candidates, by_patch = [], [[] for _ in a['patches']]
    for g, group in enumerate(groups):
        for v, orientation, swap in _variants(group['ref']):
            h, w = v.shape
            for pi, (pr, pc, patch) in enumerate(a['patches']):
                ph, pw = patch.shape
                if ph > h or pw > w or patch.size == v.size:
                    continue
                windows = np.lib.stride_tricks.sliding_window_view(v, (ph, pw))
                offsets = np.argwhere(np.all(windows == patch, axis=(-2, -1)))
                for dr, dc in offsets:
                    r, c = pr - int(dr), pc - int(dc)
                    if r < 0 or c < 0 or r + h > H or c + w > W:
                        continue
                    if not np.all(support[r:r + h, c:c + w]):
                        continue
                    # One whole crop per tile; never split another surviving crop.
                    if np.count_nonzero(known[r:r + h, c:c + w]) != patch.size:
                        continue
                    bits = 0
                    for y in range(r, r + h):
                        for x in range(c, c + w):
                            bits |= 1 << (y * W + x)
                    candidate = dict(group=g, patch=pi, r=r, c=c, pattern=v,
                                     bits=bits, orientation=orientation, swap=swap,
                                     offset=(int(dr), int(dc)))
                    by_patch[pi].append(len(candidates))
                    candidates.append(candidate)
    return groups, candidates, by_patch


def _search(a, unlimited=False, max_nodes=100000):
    groups, candidates, by_patch = _placements(a)
    counts = [g['count'] for g in groups]
    need = 0
    H, W = a['support'].shape
    for r, c in zip(*np.nonzero(a['support'])):
        need |= 1 << (int(r) * W + int(c))
    answers, dead = {}, set()
    path = []
    nodes = 0

    def search(covered, assigned):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("Unresolved mosaic search exceeded its node budget")
        if len(answers) >= 2:
            return True
        if covered == need:
            if assigned != (1 << len(by_patch)) - 1:
                return False
            out = a['board'].copy()
            for ci in path:
                p = candidates[ci]
                r, c, v = p['r'], p['c'], p['pattern']
                out[r:r + v.shape[0], c:c + v.shape[1]] = np.take(a['payload'], v)
            answers.setdefault(_key(out), (out, tuple(path)))
            return True
        state = (covered, assigned, tuple(counts))
        if state in dead:
            return False
        best = None
        for pi, choices in enumerate(by_patch):
            if assigned & (1 << pi):
                continue
            options = [ci for ci in choices if not (covered & candidates[ci]['bits'])
                       and (unlimited or counts[candidates[ci]['group']] > 0)]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
        if best is None:
            dead.add(state)
            return False
        found = False
        for ci in best:
            p = candidates[ci]
            g = p['group']
            counts[g] -= 1
            path.append(ci)
            found |= search(covered | p['bits'], assigned | (1 << p['patch']))
            path.pop()
            counts[g] += 1
            if len(answers) >= 2:
                break
        if not found:
            dead.add(state)
        return found

    search(0, 0)
    return answers, groups, candidates, by_patch, nodes


def _analyze(input_grid):
    grid = _grid(input_grid)
    interpretations = []
    for a in _interpretations(grid):
        answers, groups, candidates, by_patch, nodes = _search(a)
        if not answers:
            continue
        if len(answers) != 1:
            raise ValueError("The mosaic has more than one rendered completion")
        output, path = next(iter(answers.values()))
        a.update(output=output, path=path, groups=groups, candidates=candidates,
                 by_patch=by_patch, nodes=nodes)
        interpretations.append(a)
    if len(interpretations) != 1:
        raise ValueError("Input must identify one board, inventory, and completion")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the complete mosaic solely from the supplied grid."""
    return _analyze(input_grid)['output'].copy()


def _pattern(rng, h, w):
    """Small structured, asymmetric binary references, without a template bank."""
    for _ in range(100):
        a = np.zeros((h, w), dtype=np.int8)
        mode = int(rng.integers(5))
        if mode == 0:
            a[int(rng.integers(h)), :] = 1
            a[:, int(rng.integers(w))] = 1
        elif mode == 1:
            r, c = int(rng.integers(h - 1)), int(rng.integers(w - 1))
            a[r:, c] = 1
            a[r, c:] = 1
        elif mode == 2:
            a[::int(rng.integers(2, 4)), :] = 1
        elif mode == 3:
            a[0, :] = a[-1, :] = 1
            a[:, 0] = a[:, -1] = 1
        else:
            for r in range(h):
                c = min(w - 1, r * w // h)
                a[r, :c + 1] = 1
        # Local marks interrupt repetition and distinguish orientation and offsets.
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            a[r, c] ^= 1
        if .15 < a.mean() < .85 and len(list(_variants(a))) == 16:
            return a
    # Still procedural and bounded; this extremely rare branch uses a random
    # matrix only after the structured attempts, never a stored reference.
    for _ in range(100):
        a = rng.integers(0, 2, size=(h, w), dtype=np.int8)
        if len(list(_variants(a))) == 16:
            return a
    raise RuntimeError("Could not sample an asymmetric reference")


def _twins(rng):
    for _ in range(100):
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        if h == w:
            continue
        x = _pattern(rng, h, w)
        ph, pw = 2, int(rng.choice([2, min(3, w - 1)]))
        r, c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
        weak = (r, c, ph, pw)
        if len(np.unique(x[r:r + ph, c:c + pw])) != 2:
            continue
        y = x.copy()
        outside = [(rr, cc) for rr in range(h) for cc in range(w)
                   if not (r <= rr < r + ph and c <= cc < c + pw)]
        rng.shuffle(outside)
        for rr, cc in outside[:int(rng.integers(2, min(5, len(outside)) + 1))]:
            y[rr, cc] ^= 1
        if len(list(_variants(y))) != 16 or _canonical(x) == _canonical(y):
            continue
        # Find a proper large crop of X not recognizable in any Y variant,
        # and with only one occurrence among X's geometric/color variants.
        strongs = []
        for dh, dw in ((1, 0), (0, 1), (1, 1)):
            for rr in range(dh + 1):
                for cc in range(dw + 1):
                    patch = x[rr:rr + h - dh, cc:cc + w - dw]
                    if len(np.unique(patch)) != 2:
                        continue
                    matches = []
                    for gi, ref in enumerate((x, y)):
                        for v, k, s in _variants(ref):
                            if any(t > u for t, u in zip(patch.shape, v.shape)):
                                continue
                            wins = np.lib.stride_tricks.sliding_window_view(v, patch.shape)
                            matches.extend((gi, k, s, *pos) for pos in np.argwhere(np.all(wins == patch, axis=(-2, -1))))
                    if len(matches) == 1:
                        strongs.append((rr, cc, h - dh, w - dw))
        if strongs:
            return [x, y], strongs[int(rng.integers(len(strongs)))], weak
    raise RuntimeError("Could not build two references with a shared crop")


def _grow(rng, shapes, disconnected):
    rects = [(0, 0, *shapes[0])]
    for h, w in shapes[1:]:
        choices = []
        for r, c, rh, rw in rects:
            for side in range(4):
                gap = int(rng.integers(1, 3)) if disconnected and rng.random() < .3 else 0
                if side < 2:
                    rr = r - h - gap if side == 0 else r + rh + gap
                    positions = [(rr, cc) for cc in range(c - w + 1, c + rw)]
                else:
                    cc = c - w - gap if side == 2 else c + rw + gap
                    positions = [(rr, cc) for rr in range(r - h + 1, r + rh)]
                for rr, cc in positions:
                    if any(rr < y + th and rr + h > y and cc < x + tw and cc + w > x
                           for y, x, th, tw in rects):
                        continue
                    all_rects = rects + [(rr, cc, h, w)]
                    H = max(y + th for y, x, th, tw in all_rects) - min(y for y, x, th, tw in all_rects)
                    W = max(x + tw for y, x, th, tw in all_rects) - min(x for y, x, th, tw in all_rects)
                    if H <= 17 and W <= 17:
                        # Weight compact attachments, while permitting branches.
                        choices.append((H * W + int(rng.integers(35)), rr, cc))
        if not choices:
            return None
        choices.sort()
        _, r, c = choices[int(rng.integers(min(12, len(choices))))]
        rects.append((r, c, h, w))
    r0, c0 = min(r for r, c, h, w in rects), min(c for r, c, h, w in rects)
    top, left, bottom, right = map(int, rng.integers(0, 3, size=4))
    rects = [(r - r0 + top, c - c0 + left, h, w) for r, c, h, w in rects]
    H, W = max(r + h for r, c, h, w in rects) + bottom, max(c + w for r, c, h, w in rects) + right
    if top + left + bottom + right == 0:
        H += 1
    return rects, (H, W)


def _crop_choices(rng, ref, sparse):
    H, W = ref.shape
    result = []
    for h in range(2, H + 1):
        for w in range(2, W + 1):
            if h * w == H * W or (sparse and h * w > max(6, .6 * H * W)):
                continue
            for r in range(H - h + 1):
                for c in range(W - w + 1):
                    if len(np.unique(ref[r:r + h, c:c + w])) == 2:
                        result.append((r, c, h, w))
    rng.shuffle(result)
    return result


def _scatter(rng, tiles, H, W, erased):
    if H < max(t.shape[0] for t in tiles) or W < max(t.shape[1] for t in tiles):
        return None
    for _ in range(32):
        occupied = np.zeros((H, W), dtype=bool)
        placements = []
        order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
        for i in order:
            h, w = tiles[i].shape
            choices = []
            for r in range(H - h + 1):
                for c in range(W - w + 1):
                    if not np.any(occupied[max(0, r - 1):min(H, r + h + 1), max(0, c - 1):min(W, c + w + 1)]):
                        choices.append((r, c))
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            occupied[r:r + h, c:c + w] = True
            placements.append((i, r, c))
        if len(placements) == len(tiles):
            tray = np.full((H, W), erased, dtype=np.int8)
            for i, r, c in placements:
                h, w = tiles[i].shape
                tray[r:r + h, c:c + w] = tiles[i]
            return tray
    return None


def _inventory_witness(a):
    """An alternate same-footprint completion becomes legal if copies are infinite."""
    used = Counter(a['candidates'][ci]['group'] for ci in a['path'])
    for ci in a['path']:
        p = a['candidates'][ci]
        for di in a['by_patch'][p['patch']]:
            q = a['candidates'][di]
            if q['bits'] == p['bits'] and q['group'] != p['group'] and not np.array_equal(q['pattern'], p['pattern']):
                if used[q['group']] == a['groups'][q['group']]['count']:
                    return True
    return False


def _geometry_witness(a):
    """A locally matching footprint would overlap the cover or leave a gap."""
    return any(any(a['candidates'][di]['bits'] != a['candidates'][ci]['bits']
                   for di in a['by_patch'][a['candidates'][ci]['patch']])
               for ci in a['path'])


def _build_pair(rng, roles, count, sparse, teaching, disconnected, seen):
    erased, background, marker, s0, s1, q0, q1 = roles
    for attempt in range(160):
        refs, strong, weak = _twins(rng)
        signatures = {_canonical(t) for t in refs}
        for _ in range(100):
            if len(refs) == count:
                break
            h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
            if h * w < 8:
                continue
            t = _pattern(rng, h, w)
            if _canonical(t) not in signatures:
                signatures.add(_canonical(t))
                refs.append(t)
        if len(refs) != count:
            continue
        orientations = list(map(int, rng.integers(0, 8, size=count)))
        if teaching == 0:
            orientations[:3] = [0, 0, 1]
        elif teaching == 1:
            orientations[:3] = [1, 1, 4]
        swaps = list(map(int, rng.integers(0, 2, size=count)))
        if teaching >= 0:
            swaps[:2] = [0, 1]
        fulls = [_d4(t, k) ^ swap for t, k, swap in zip(refs, orientations, swaps)]
        grown = _grow(rng, [v.shape for v in fulls], disconnected)
        if grown is None:
            continue
        rects, shape = grown
        board = np.full(shape, background, dtype=np.int8)
        output = board.copy()
        known = np.zeros(shape, dtype=bool)
        failed = False
        for i, ((r, c, h, w), ref, v, k, swap) in enumerate(zip(rects, refs, fulls, orientations, swaps)):
            board[r:r + h, c:c + w] = erased
            output[r:r + h, c:c + w] = np.where(v == 0, q0, q1)
            choices = [strong] if i == 0 else [weak] if i == 1 else _crop_choices(rng, ref, sparse)
            placed = False
            for cr, cc, ph, pw in choices:
                mask = np.zeros(ref.shape, dtype=bool)
                mask[cr:cr + ph, cc:cc + pw] = True
                mask = _d4(mask, k)
                ys, xs = np.nonzero(mask)
                rr, col = r + int(ys.min()), c + int(xs.min())
                yy, xx = r + int(ys.max()) + 1, c + int(xs.max()) + 1
                if np.any(known[max(0, rr - 1):min(shape[0], yy + 1), max(0, col - 1):min(shape[1], xx + 1)]):
                    continue
                known[rr:yy, col:xx] = True
                board[rr:yy, col:xx] = output[rr:yy, col:xx]
                placed = True
                break
            if not placed:
                failed = True
                break
        if failed:
            continue
        tiles = [np.where(t == 0, s0, s1).astype(np.int8) for t in refs]
        # Random tray geometry is independent of destination order and positions.
        sides = list(map(int, rng.permutation(4)))
        for side in sides:
            H, W = shape
            vertical = side < 2
            th, tw = (H, 29 - W) if vertical else (29 - H, W)
            tray = _scatter(rng, tiles, th, tw, erased)
            if tray is None:
                continue
            ys, xs = np.nonzero(tray != erased)
            if vertical:
                end = int(rng.integers(int(xs.max()) + 1, tw + 1))
                tray = tray[:, :end]
                line = np.full((H, 1), marker, dtype=np.int8)
                panels = (board, line, tray) if side == 0 else (tray, line, board)
                inp = np.concatenate(panels, axis=1)
            else:
                end = int(rng.integers(int(ys.max()) + 1, th + 1))
                tray = tray[:end, :]
                line = np.full((1, W), marker, dtype=np.int8)
                panels = (board, line, tray) if side == 2 else (tray, line, board)
                inp = np.concatenate(panels, axis=0)
            if _key(inp) in seen:
                continue
            try:
                a = _analyze(inp)
            except ValueError:
                continue
            if not np.array_equal(a['output'], output) or not _inventory_witness(a):
                continue
            if teaching == 1 and not _geometry_witness(a):
                continue
            seen.add(_key(inp))
            return dict(input=inp.copy(), output=output.copy())
    raise RuntimeError("Bounded construction could not find a unique mosaic")


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = rng.random() < .35
    colors = tuple(map(int, rng.permutation(10)[:7]))
    # Palette stability is a sampling choice, never a solving convention.
    disconnected = rng.random() < .3
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(nt + nq):
        training = i < nt
        roles = colors if stable else tuple(map(int, rng.permutation(10)[:7]))
        # Later pairs can reuse the source colors as destination colors. The first
        # two visibly demonstrate color-role renaming with disjoint palettes.
        if i >= 2 and rng.random() < .3:
            roles = roles[:5] + roles[3:5]
        count = 3 if i == 0 else int(rng.integers(3, 5)) if training else int(rng.integers(4, 7))
        pair = _build_pair(rng, roles, count, sparse=not training or i == 1,
                           teaching=i if i < 2 else -1, disconnected=disconnected and i > 0, seen=seen)
        puzzle['train' if training else 'test'].append(pair)
    return puzzle


def _features(a):
    choices = [a['candidates'][ci] for ci in a['path']]
    owners = np.full(a['support'].shape, -1, dtype=int)
    for i, p in enumerate(choices):
        r, c = p['r'], p['c']
        h, w = p['pattern'].shape
        owners[r:r + h, c:c + w] = i
    component_tiles = []
    for r, c, y, x, area, cells in _components(a['support'], with_cells=True):
        component_tiles.append(len({owners[rr, cc] for rr, cc in cells}))
    return dict(side=a['side'], board_shape=a['board'].shape, tiles=len(choices),
                component_tiles=max(component_tiles), components=len(component_tiles),
                coverage=float(a['support'].mean()),
                observed=float(a['known'].sum() / a['support'].sum()),
                orientations=tuple(p['orientation'] for p in choices),
                offsets=tuple(p['offset'] for p in choices),
                shapes=tuple(p['pattern'].shape for p in choices),
                crop_shapes=tuple(p.shape for _, _, p in a['patches']),
                locally_ambiguous=sum(len(a['by_patch'][i]) > 1 for i in range(len(a['patches']))),
                nodes=a['nodes'], inventory_witness=_inventory_witness(a),
                geometry_witness=_geometry_witness(a))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, feats = set(), []
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        inp, out = _grid(pair['input']), _grid(pair['output'])
        assert _key(inp) not in seen
        seen.add(_key(inp))
        a = _analyze(inp)
        assert np.array_equal(a['output'], out), "Output differs from the unique input-derived restoration"
        assert np.all(out[~a['support']] == a['background'])
        assert np.array_equal(out[a['known']], a['board'][a['known']])
        assert not np.any(out == a['erased']) and not np.any(out == a['marker'])
        assert set(map(int, np.unique(out))) == {a['background'], *a['payload']}
        # Generated members use every physical tile, with one proper crop each.
        used = Counter(a['candidates'][ci]['group'] for ci in a['path'])
        assert all(used[g] == group['count'] for g, group in enumerate(a['groups']))
        assert sum(t.size for t in a['refs']) == np.count_nonzero(a['support'])
        assert len(a['refs']) == len(a['patches'])
        assert _inventory_witness(a), "Finite copies must change the set of possible rendered answers"
        for ci in a['path']:
            p = a['candidates'][ci]
            patch = a['patches'][p['patch']][2]
            assert 0 < patch.size < p['pattern'].size
            # Construct the omission-of-A counterexample on actual output pixels.
            # It retains this footprint and the same physical source consumption.
            alternative = out.copy()
            r, c, v = p['r'], p['c'], p['pattern']
            alternative[r:r + v.shape[0], c:c + v.shape[1]] = np.take(a['payload'], 1 - v)
            pr, pc, observed = a['patches'][p['patch']]
            ph, pw = observed.shape
            assert np.all(alternative[pr:pr + ph, pc:pc + pw] != out[pr:pr + ph, pc:pc + pw])
            assert np.array_equal(alternative[~a['support']], out[~a['support']])
        feats.append(_features(a))
    training = feats[:len(puzzle['train'])]
    assert any(f['component_tiles'] >= 3 for f in training)
    assert all(f['locally_ambiguous'] > 0 and f['inventory_witness'] for f in training[:2])
    assert training[1]['geometry_witness'], "Demonstrations must reject a locally matching wrong footprint"
    # At least one non-square dimension exchange is visibly taught.
    assert any(p['pattern'].shape == tuple(reversed(a['groups'][p['group']]['ref'].shape))
               and p['pattern'].shape[0] != p['pattern'].shape[1]
               for pair in puzzle['train'][:2] for a in [_analyze(pair['input'])]
               for p in (a['candidates'][ci] for ci in a['path']))


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded semantic checks and measured structural variation on 42 seeds."""
    features, signatures, refs, palettes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for pair in puzzle['train'] + puzzle['test']:
            a = _analyze(pair['input'])
            features.append(_features(a))
            refs.update(_canonical(t) for t in a['refs'])
            palettes.add(tuple(a['source_colors'] + a['payload']))
    assert len(signatures) == 42 and len(refs) >= 100
    assert len({f['board_shape'] for f in features}) >= 30
    assert len({s for f in features for s in f['shapes']}) >= 12
    assert len({s for f in features for s in f['crop_shapes']}) >= 12
    assert len({o for f in features for o in f['offsets']}) >= 8
    assert {f['side'] for f in features} == {'left', 'right', 'top', 'bottom'}
    assert {k for f in features for k in f['orientations']} == set(range(8))
    assert {f['tiles'] for f in features} >= {3, 4, 5, 6}
    assert max(f['component_tiles'] for f in features) >= 5
    assert any(f['components'] > 1 for f in features)
    assert len(palettes) >= 30
    assert all(f['inventory_witness'] and f['locally_ambiguous'] for f in features)
    p = generate_puzzle(2026, 2, 1)
    assert _signature(p) == _signature(generate_puzzle(2026, 2, 1))
    validate_generated_puzzle(generate_puzzle(23, 10, 4))
    # Explicit unlimited-copy counterexample on actual generated input.
    a = _analyze(p['train'][1]['input'])
    assert len(_search(a, unlimited=True)[0]) == 2
    # Removing an actually used reference destroys feasibility or uniqueness.
    removed = dict(a)
    removed['refs'] = a['refs'][1:]
    assert not _search(removed)[0]
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Validator accepted a corrupted label")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                reference_classes=len(refs), board_shapes=len({f['board_shape'] for f in features}),
                tile_counts=sorted({f['tiles'] for f in features}),
                sides=sorted({f['side'] for f in features}),
                observed_fraction_range=[min(f['observed'] for f in features), max(f['observed'] for f in features)])


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
