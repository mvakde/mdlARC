"""Phase-registered reconstruction from a finite inventory of rectangles.

All inference is from the input.  The frame (9), pilot bar (5), and empty
canvas/vacancies (0) are fixed; board and probe colors are local visible roles.
"""

from collections import Counter
import numpy as np


FRAME, PILOT, EMPTY = 9, 5, 0
HOST_SEEDS = (0, 882665187, 509528309, 830699452, 610400475, 852002396,
              798131955, 288578389, 522745039, 188926170, 411469946,
              411078006, 220622146, 905603378, 769735592, 152493913,
              73944017, 979686921, 42485263, 761884240, 315900551,
              385171435, 391641631, 350842050, 808445057, 984340310,
              325191925, 18653864, 292617788, 228944605, 374959916,
              68550389, 909696419, 793852017, 955440922, 584515856,
              991370624, 482184720, 469298481, 859795131, 418883512,
              975803020)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        root = min(pending)
        pending.remove(root)
        todo, cells = [root], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in pending:
                    pending.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _bbox(mask):
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("Missing frame")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _parse(input_grid):
    """Rotate the entire input, including its tiles, into pilot-at-top coordinates."""
    original = _grid(input_grid)
    y, x, yy, xx = _bbox(original == FRAME)
    board0 = original[y:yy, x:xx]
    points = np.argwhere(board0 == PILOT)
    if not len(points) or np.count_nonzero(original == PILOT) != len(points):
        raise ValueError("Pilot must occur only on the frame")
    rotations = []
    for k in range(4):
        b = np.rot90(board0, k)
        pp = np.argwhere(b == PILOT)
        if np.all(pp[:, 0] == 0):
            cols = np.sort(pp[:, 1])
            if (cols[0] > 0 and cols[-1] < b.shape[1] - 1
                    and np.array_equal(cols, np.arange(cols[0], cols[-1] + 1))):
                rotations.append(k)
    if len(rotations) != 1:
        raise ValueError("Pilot has no unique inward direction")
    k = rotations[0]
    canvas = np.rot90(original, k).copy()
    y, x, yy, xx = _bbox(canvas == FRAME)
    board = canvas[y:yy, x:xx].copy()
    h, w = board.shape
    if h < 7 or w < 7:
        raise ValueError("Frame too small")
    columns = np.flatnonzero(board[0] == PILOT)
    start, length = int(columns[0]), len(columns)
    border = np.zeros_like(board, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    expected_border = np.full(board.shape, FRAME, dtype=np.int8)
    expected_border[0, start:start + length] = PILOT
    if (not np.array_equal(board[border], expected_border[border])
            or np.any(board[1:-1, 1:-1] == FRAME)
            or np.any(board[1:-1, 1:-1] == PILOT)):
        raise ValueError("The frame is not a single bordered rectangle")
    # The one-cell lining inside the frame makes the board role explicit.
    base = int(board[1, 1])
    if base in (EMPTY, PILOT, FRAME):
        raise ValueError("Invalid board color")
    lining = np.zeros_like(board, dtype=bool)
    lining[1, 1:-1] = lining[-2, 1:-1] = True
    lining[1:-1, 1] = lining[1:-1, -2] = True
    if not np.all(board[lining] == base):
        raise ValueError("Missing board-colored lining")
    interior = ~border
    extra = set(map(int, np.unique(board[interior]))) - {EMPTY, base}
    if len(extra) != 1:
        raise ValueError("Exactly one visible probe color is required")
    probe = extra.pop()
    if probe in (PILOT, FRAME):
        raise ValueError("Invalid probe color")
    mask = interior & ((board == EMPTY) | (board == probe))
    scanned = np.broadcast_to(np.arange(h)[:, None] % 6 == 4, board.shape)
    if not mask.any() or np.any((board == probe) & ~scanned):
        raise ValueError("Probe observations must be at distance 4 modulo 6")
    external = canvas.copy()
    external[y:yy, x:xx] = EMPTY
    tiles = []
    positions = []
    for cells in _components(external != EMPTY):
        rr, cc = np.array(cells).T
        a, b, aa, bb = int(rr.min()), int(cc.min()), int(rr.max() + 1), int(cc.max() + 1)
        tile = external[a:aa, b:bb].copy()
        if (tile.size != len(cells) or np.any(tile == EMPTY)
                or np.any(np.isin(tile, (FRAME, PILOT, base)))):
            raise ValueError("Inventory objects must be separated filled rectangles")
        tiles.append(tile)
        positions.append((a, b))
    if not tiles:
        raise ValueError("Missing rectangle inventory")
    return dict(canvas=canvas, board=board, mask=mask, scanned=scanned, probe=probe,
                base=base, tiles=tiles, positions=positions, start=start, length=length,
                rotation=k, bbox=(y, x, yy, xx))


def _pack(mask, tiles, probe, clues=None, scanned=None, node_limit=30000):
    """Enumerate up to two distinct full renderings, with finite copy counts.

    Neither source order nor construction placements influence the search.
    Interchangeable copies are grouped. Failed suffixes may be memoized; successful
    ones may not, because their earlier pixels can differ.
    """
    coords = np.argwhere(mask)
    n = len(coords)
    if not n:
        return {}, dict(nodes=0, dead_ends=0, exhaustive=True)
    index = np.full(mask.shape, -1, dtype=int)
    index[mask] = np.arange(n)
    patterns, counts, lookup = [], [], {}
    for tile in tiles:
        key = _key(tile)
        if key not in lookup:
            lookup[key] = len(patterns)
            patterns.append(tile)
            counts.append(0)
        counts[lookup[key]] += 1
    initial = tuple(counts)
    placements, by_cell = [], [[] for _ in range(n)]
    h, w = mask.shape
    for g, tile in enumerate(patterns):
        th, tw = tile.shape
        for r in range(h - th + 1):
            for c in range(w - tw + 1):
                if not np.all(mask[r:r + th, c:c + tw]):
                    continue
                if clues is not None:
                    measured = scanned[r:r + th, c:c + tw]
                    observations = clues[r:r + th, c:c + tw]
                    if not np.array_equal((tile == probe)[measured], observations[measured]):
                        continue
                indices = tuple(map(int, index[r:r + th, c:c + tw].ravel()))
                bits = sum(1 << i for i in indices)
                pi = len(placements)
                placements.append((g, bits, indices, tuple(map(int, tile.flat)), r, c))
                for i in indices:
                    by_cell[i].append(pi)
    full, rendered = (1 << n) - 1, [-1] * n
    answers, dead, path = {}, set(), []
    nodes, failures, exhaustive = 0, 0, True

    def visit(covered):
        nonlocal nodes, failures, exhaustive
        if len(answers) >= 2 or not exhaustive:
            return False
        nodes += 1
        if nodes > node_limit:
            exhaustive = False
            return False
        if covered == full:
            key = tuple(rendered)
            if key not in answers:
                answers[key] = dict(placements=tuple(path),
                                    used=tuple(a - b for a, b in zip(initial, counts)))
            return True
        state = covered, tuple(counts)
        if state in dead:
            return False
        best = None
        for i in range(n):
            if covered & (1 << i):
                continue
            options = [p for p in by_cell[i] if counts[placements[p][0]]
                       and not covered & placements[p][1]]
            if not options:
                dead.add(state)
                failures += 1
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        success = False
        for pi in best:
            g, bits, indices, colors, r, c = placements[pi]
            counts[g] -= 1
            for i, color in zip(indices, colors):
                rendered[i] = color
            path.append((g, r, c))
            success |= visit(covered | bits)
            path.pop()
            for i in indices:
                rendered[i] = -1
            counts[g] += 1
            if len(answers) >= 2 or not exhaustive:
                break
        if not success and exhaustive and len(answers) < 2:
            dead.add(state)
        return success

    visit(0)
    return answers, dict(nodes=nodes, dead_ends=failures, exhaustive=exhaustive,
                         patterns=patterns, counts=initial)


def _reconstruct(info):
    outcomes, stats = _pack(info['mask'], info['tiles'], info['probe'],
                            info['board'] == info['probe'], info['scanned'])
    if not stats['exhaustive'] or len(outcomes) != 1:
        raise ValueError("Input does not give one exhaustively verified reconstruction")
    pixels, record = next(iter(outcomes.items()))
    complete = info['board'].copy()
    complete[info['mask']] = pixels
    return complete, record, stats


def _track(complete, info):
    out = complete.copy()
    a, b = info['start'], info['start'] + info['length']
    out[:, a:b] = info['base']
    for d in range(len(out)):
        if d % 6 in (0, 2):
            out[d, a:b] = PILOT
        elif d % 6 == 4:
            out[d, a:b] = info['probe']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    complete, _, _ = _reconstruct(info)
    return np.rot90(_track(complete, info), -info['rotation']).copy()


def _grow_rectangles(rng, h, w, count, repeat, testing):
    mask = np.zeros((h, w), dtype=bool)
    rects = []
    connected = rng.random() < .83
    for i in range(count):
        if i == 1 or (i == 2 and repeat):
            th, tw = rects[0][2:]
        else:
            th = int(rng.integers(3, 6)) if i == 0 else int(rng.integers(1, 6))
            tw = int(rng.integers(2, 6)) if i == 0 else int(rng.integers(1, 6))
        options = []
        for _ in range(90):
            if not rects:
                if testing:
                    r = int(rng.integers(max(2, 23 - th), min(23, h - 1 - th)))
                else:
                    r = int(rng.integers(2, h - 1 - th))
                c = int(rng.integers(2, w - 1 - tw))
            elif not connected and rng.random() < .30:
                r, c = int(rng.integers(2, h - 1 - th)), int(rng.integers(2, w - 1 - tw))
            else:
                r0, c0, rh, rw = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    r = r0 - th if side == 0 else r0 + rh
                    c = c0 + int(rng.integers(1 - tw, rw))
                else:
                    c = c0 - tw if side == 2 else c0 + rw
                    r = r0 + int(rng.integers(1 - th, rh))
            if (r < 2 or c < 2 or r + th > h - 2 or c + tw > w - 2
                    or np.any(mask[r:r + th, c:c + tw])):
                continue
            options.append((r, c))
        if not options:
            return None
        r, c = options[int(rng.integers(len(options)))]
        rects.append((r, c, th, tw))
        mask[r:r + th, c:c + tw] = True
    return rects


def _pattern(rng, shape, probe, body):
    h, w = shape
    yy, xx = np.indices(shape)
    mode = int(rng.integers(5))
    tile = np.full(shape, int(rng.choice(body)), dtype=np.int8)
    if len(body) > 1:
        if mode == 1:
            tile[yy % 2 == 1] = body[1]
        elif mode == 2:
            tile[xx % 2 == 1] = body[1]
        elif mode == 3:
            tile[(xx + yy) % 2 == 1] = body[1]
        elif mode == 4:
            tile = rng.choice(body, size=shape).astype(np.int8)
    sensed = rng.random(shape) < float(rng.uniform(.18, .48))
    if not sensed.any():
        sensed[int(rng.integers(h)), int(rng.integers(w))] = True
    tile[sensed] = probe
    return tile


def _scatter(rng, board, tiles):
    """The framed board can float; source rectangles can surround several sides."""
    bh, bw = board.shape
    ch = int(rng.integers(max(26, bh), 31))
    cw = int(rng.integers(max(26, bw), 31))
    for _ in range(12):
        by, bx = int(rng.integers(ch - bh + 1)), int(rng.integers(cw - bw + 1))
        canvas = np.zeros((ch, cw), dtype=np.int8)
        blocked = np.zeros_like(canvas, dtype=bool)
        blocked[max(0, by - 1):min(ch, by + bh + 1), max(0, bx - 1):min(cw, bx + bw + 1)] = True
        canvas[by:by + bh, bx:bx + bw] = board
        order = sorted(rng.permutation(len(tiles)).tolist(), key=lambda i: -tiles[i].size)
        for i in order:
            tile = tiles[i]
            h, w = tile.shape
            options = [(r, c) for r in range(ch - h + 1) for c in range(cw - w + 1)
                       if not blocked[r:r + h, c:c + w].any()]
            if not options:
                break
            r, c = options[int(rng.integers(len(options)))]
            canvas[r:r + h, c:c + w] = tile
            blocked[max(0, r - 1):min(ch, r + h + 1), max(0, c - 1):min(cw, c + w + 1)] = True
        else:
            return canvas
    return None


def _make_board(shape, base, rects, tiles, probe, start, length):
    complete = np.full(shape, base, dtype=np.int8)
    complete[0] = complete[-1] = FRAME
    complete[:, 0] = complete[:, -1] = FRAME
    complete[0, start:start + length] = PILOT
    mask = np.zeros(shape, dtype=bool)
    for (r, c, h, w), tile in zip(rects, tiles):
        mask[r:r + h, c:c + w] = True
        complete[r:r + h, c:c + w] = tile
    measured = np.broadcast_to(np.arange(shape[0])[:, None] % 6 == 4, shape)
    stencil = complete.copy()
    stencil[mask] = EMPTY
    stencil[mask & measured & (complete == probe)] = probe
    return stencil, complete, mask, measured


def _evidence(info, complete, record, stats):
    mask, scan = info['mask'], info['scanned']
    strip = np.zeros_like(mask)
    strip[:, info['start']:info['start'] + info['length']] = True
    d = np.arange(len(mask))[:, None]
    body = mask & (complete != info['probe'])
    probe = mask & (complete == info['probe'])
    clue_rows = tuple(map(int, np.flatnonzero(np.any(info['board'] == info['probe'], axis=1))))
    labels = np.full(mask.shape, -1, dtype=int)
    comps = _components(mask)
    for i, comp in enumerate(comps):
        for r, c in comp:
            labels[r, c] = i
    component_counts = Counter(int(labels[r, c]) for _, r, c in record['placements'])
    return dict(clue_rows=clue_rows, odd_clear=bool(np.any(body & strip & (d % 2 == 1))),
                ordinary_overwrite=bool(np.any(mask & strip & (d > 0) & ((d % 6 == 0) | (d % 6 == 2)))),
                remote_probe=bool(np.any(probe & scan & ~strip)),
                unmeasured_probe=bool(np.any(probe & ~scan & ~strip)),
                scanned_body=bool(np.any(body & scan & ~strip)),
                repeated=any(n >= 2 for n in record['used']),
                asymmetric=any(n and not np.array_equal(t, t[::-1, ::-1])
                               for t, n in zip(stats['patterns'], record['used'])),
                merged=max(component_counts.values(), default=0),
                surplus=len(info['tiles']) - sum(record['used']),
                tile_count=len(info['tiles']), used=sum(record['used']),
                shape=tuple(info['board'].shape), rotation=info['rotation'],
                base=info['base'], probe=info['probe'], components=len(comps),
                start=info['start'], length=info['length'])


def _acceptable(info, complete, record, stats, force_scan, force_repeat, testing):
    e = _evidence(info, complete, record, stats)
    if force_repeat and not e['repeated']:
        return False
    if testing and 22 not in e['clue_rows']:
        return False
    if force_scan:
        if not ({4, 10} <= set(e['clue_rows']) and e['merged'] >= 3 and e['asymmetric']
                and all(e[k] for k in ('odd_clear', 'ordinary_overwrite', 'remote_probe',
                                      'unmeasured_probe', 'scanned_body'))):
            return False
        bare, bare_stats = _pack(info['mask'], info['tiles'], info['probe'], node_limit=4000)
        if not bare_stats['exhaustive'] or len(bare) < 2:
            return False
        if not _visible_ambiguity(info, bare):
            return False
    return True


def _build_pair(rng, rotation, base, probe, force_scan, force_repeat, testing, surplus):
    body = [c for c in range(1, 9) if c not in (PILOT, base, probe)]
    rng.shuffle(body)
    body = body[:int(rng.integers(1, len(body) + 1))]
    for _ in range(48):
        shape = (int(rng.integers(25, 28) if testing else rng.integers(14, 21)),
                 int(rng.integers(11, 19)))
        count = int(rng.integers(4 if force_repeat or testing else 3, 8 if testing else 7))
        rects = _grow_rectangles(rng, *shape, count, force_repeat, testing)
        if rects is None:
            continue
        tiles = [_pattern(rng, (h, w), probe, body) for r, c, h, w in rects]
        if np.array_equal(tiles[0], tiles[1]):
            continue
        if force_repeat:
            tiles[2] = tiles[0].copy()
        length = int(rng.integers(1, min(5, shape[1] - 3)))
        start = int(rng.integers(1, shape[1] - length))
        stencil, expected, mask, measured = _make_board(shape, base, rects, tiles, probe, start, length)
        if not np.any(stencil == probe):
            continue
        outcomes, stats = _pack(mask, tiles, probe, stencil == probe, measured, node_limit=6000)
        wanted = tuple(map(int, expected[mask]))
        if not stats['exhaustive'] or set(outcomes) != {wanted}:
            continue
        record = outcomes[wanted]
        info = dict(board=stencil, mask=mask, scanned=measured, tiles=tiles, base=base,
                    probe=probe, start=start, length=length, rotation=rotation)
        if not _acceptable(info, expected, record, stats, force_scan, force_repeat, testing):
            continue
        if surplus:
            # A competing pattern of a needed size can be rejected by the scan,
            # rather than being an automatically impossible oversized distractor.
            accepted = False
            for _ in range(8):
                decoy = _pattern(rng, tiles[0].shape, probe, body)
                answers, extra_stats = _pack(mask, tiles + [decoy], probe,
                                             stencil == probe, measured, node_limit=6000)
                if extra_stats['exhaustive'] and set(answers) == {wanted}:
                    tiles.append(decoy)
                    accepted = True
                    break
            if not accepted:
                continue
        canvas = _scatter(rng, stencil, tiles)
        if canvas is None:
            continue
        info['tiles'] = tiles
        inp, out = np.rot90(canvas, -rotation).copy(), np.rot90(_track(expected, info), -rotation).copy()
        return {'input': inp, 'output': out}
    return _fallback(rng, rotation, base, probe, force_repeat, testing, surplus)


def _fallback(rng, rotation, base, probe, repeat, testing, surplus):
    """Certified same-rule construction, not a different transformation.

    Two equal-sized lobes joined by a forced narrow spine. Their distinct probe
    columns break the geometric exchange symmetry. An optional third lobe is an
    identical copy; an optional wrong-probe copy is surplus.
    """
    w = int(rng.integers(3, 6))
    x = int(rng.integers(2, 5))
    offset = 12 if testing else 0
    h = int(rng.integers(26, 28) if testing else rng.integers(14, 21))
    width = max(2 * w + x + 3, int(rng.integers(12, 19)))
    colors = [c for c in range(1, 9) if c not in (PILOT, base, probe)]
    a, b = rng.choice(colors, 2, replace=False).tolist()
    p = np.full((4, w), a, dtype=np.int8)
    q = np.full((4, w), b, dtype=np.int8)
    p[:, 0], q[:, -1] = probe, probe
    p[1, -1], q[2, 0] = b, a  # asymmetric copied payload
    spine = rng.choice([a, b, probe], size=(6, 1)).astype(np.int8)
    rects = [(2 + offset, x, 4, w), (8 + offset, x, 4, w), (3 + offset, x + w, 6, 1)]
    tiles = [p, q, spine]
    if repeat:
        rects.append((8 + offset, x + w + 1, 4, w))
        tiles.append(p.copy())
    stencil, expected, mask, measured = _make_board((h, width), base, rects, tiles, probe, x + 1, 1)
    if surplus:
        # It fits either lobe geometrically, but its all-probe scan signature fails.
        tiles.append(np.full(p.shape, probe, dtype=np.int8))
    outcomes, stats = _pack(mask, tiles, probe, stencil == probe, measured)
    assert stats['exhaustive'] and set(outcomes) == {tuple(map(int, expected[mask]))}
    canvas = _scatter(rng, stencil, tiles)
    if canvas is None:
        # Side tray guarantees space even if the scattered proposals were unlucky.
        # Width is at most 18; all pieces fit in the remaining side strip.
        cw, ch = 30, max(28, h)
        canvas = np.zeros((ch, cw), dtype=np.int8)
        canvas[:h, :width] = stencil
        occupied = np.zeros_like(canvas, dtype=bool)
        occupied[:, :width + 1] = True
        for tile in sorted(tiles, key=lambda t: -t.size):
            th, tw = tile.shape
            choices = [(r, c) for r in range(ch - th + 1) for c in range(width + 1, cw - tw + 1)
                       if not occupied[r:r + th, c:c + tw].any()]
            assert choices, 'Fallback tray capacity'
            r, c = choices[0]
            canvas[r:r + th, c:c + tw] = tile
            occupied[max(0, r - 1):min(ch, r + th + 1), max(0, c - 1):min(cw, c + tw + 1)] = True
    info = dict(base=base, probe=probe, start=x + 1, length=1)
    return dict(input=np.rot90(canvas, -rotation).copy(),
                output=np.rot90(_track(expected, info), -rotation).copy())


def _visible_ambiguity(info, outcomes):
    """The scan witness must matter even after the track overwrites its corridor."""
    rendered = set()
    for pixels in outcomes:
        full = info['board'].copy()
        full[info['mask']] = pixels
        rendered.add(_key(_track(full, info)))
    return len(rendered) >= 2


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(name + ' outside legal range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    rotations = rng.permutation(4).tolist()
    regime = int(rng.integers(3))  # consistent / quarter turn / mixed with held-out sides
    allow_surplus = bool(rng.integers(2))
    result, seen, previous = {'train': [], 'test': []}, set(), None
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            testing = split == 'test'
            if regime == 0:
                rotation = rotations[0]
            elif regime == 1:
                rotation = rotations[1 if testing else 0]
            else:
                rotation = rotations[(2 + i % 2) if testing else (i % 2)]
            choices = [(a, b) for a in range(1, 9) for b in range(1, 9)
                       if a != b and a != PILOT and b != PILOT and (a, b) != previous]
            base, probe = choices[int(rng.integers(len(choices)))]
            previous = base, probe
            surplus = allow_surplus and (i == 0 or rng.random() < .4)
            for _ in range(8):
                pair = _build_pair(rng, rotation, base, probe, not testing and i == 0,
                                   not testing and i == 1, testing, surplus)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('Bounded distinct-input construction exhausted')
            seen.add(_key(pair['input']))
            result[split].append(pair)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, features, infos = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'Duplicate input'
            seen.add(_key(inp))
            info = _parse(inp)
            complete, record, stats = _reconstruct(info)
            expected = np.rot90(_track(complete, info), -info['rotation'])
            assert np.array_equal(out, expected), 'Incorrect reconstruction or track'
            # Validate the projection both positively and negatively.
            assert np.array_equal((complete == info['probe'])[info['mask'] & info['scanned']],
                                  (info['board'] == info['probe'])[info['mask'] & info['scanned']])
            assert not np.any(complete[info['mask']] == EMPTY), 'Unfilled vacancy'
            assert all(n <= c for n, c in zip(record['used'], stats['counts'])), 'Reused tile'
            features.append(_evidence(info, complete, record, stats))
            infos.append(info)
    train, test = features[:nt], features[nt:]
    assert any({4, 10} <= set(f['clue_rows']) for f in train), 'Repeated scan phase untaught'
    assert all(any(f[k] for f in train) for k in ('odd_clear', 'ordinary_overwrite', 'remote_probe',
                                               'unmeasured_probe', 'scanned_body', 'asymmetric', 'repeated'))
    assert any(f['merged'] >= 3 for f in train), 'Merged packing untaught'
    assert not any(f['surplus'] for f in test) or any(f['surplus'] for f in train)
    assert all(f['shape'][0] <= 20 for f in train) and all(f['shape'][0] >= 25 for f in test)
    assert all(22 in f['clue_rows'] for f in test), 'Missing extrapolation witness'
    ambiguity_witness = False
    for info in infos[:nt]:
        outcomes, stats = _pack(info['mask'], info['tiles'], info['probe'], node_limit=6000)
        if stats['exhaustive'] and _visible_ambiguity(info, outcomes):
            ambiguity_witness = True
            break
    assert ambiguity_witness, 'Ignoring the scan must change/underdetermine a demonstration'
    return None


def self_test():
    """Bounded structural, semantic, counterfactual, and 42-seed diversity checks."""
    records, signatures = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                full, record, stats = _reconstruct(info)
                f = _evidence(info, full, record, stats)
                f['split'] = split
                records.append(f)
                assert np.array_equal(solve(pair['input']), pair['output'])
    assert len(signatures) == 42
    assert len({f['shape'] for f in records}) >= 25
    assert len({(f['shape'], f['start'], f['length'], f['used'], f['components']) for f in records}) >= 100
    for split in ('train', 'test'):
        rr = [f for f in records if f['split'] == split]
        assert {f['rotation'] for f in rr} == {0, 1, 2, 3}
        assert len({f['used'] for f in rr}) >= 3
        assert len({(f['base'], f['probe']) for f in rr}) >= 12
    assert any(f['surplus'] for f in records) and any(not f['surplus'] for f in records)
    assert any(f['components'] > 1 for f in records)
    a, b = generate_puzzle(71, 2, 1), generate_puzzle(71, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # The host generates every legal combination; keep this self-test bounded
    # while exercising both extremes and the largest complete puzzle.
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100 + nt * 5 + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Finite inventory is essential even when probes agree with both copies.
    mask = np.ones((1, 4), dtype=bool)
    tile = np.array([[2, 3]], dtype=np.int8)
    clues = np.array([[True, False, True, False]])
    scan = np.ones_like(mask)
    assert not _pack(mask, [tile], 2, clues, scan)[0]
    assert len(_pack(mask, [tile, tile], 2, clues, scan)[0]) == 1
    # Same-size matrices have ambiguous geometry but probe observations resolve it.
    other = np.array([[3, 2]], dtype=np.int8)
    assert len(_pack(mask, [tile, other], 2)[0]) == 2
    mixed_clues = np.array([[True, False, False, True]])
    assert len(_pack(mask, [tile, other], 2, mixed_clues, scan)[0]) == 1
    # Positive observations alone accept a false extra probe in the second cell.
    small = np.ones((1, 3), dtype=bool)
    p = np.array([[2, 3, 3]], dtype=np.int8)
    q = np.array([[2, 2, 3]], dtype=np.int8)
    observations = np.array([[True, False, False]])
    assert len(_pack(small, [p, q], 2, observations, small)[0]) == 1
    assert len(_pack(small, [p, q], 2, observations, observations)[0]) == 2
    # Exercise certified fallback for every orientation and surplus convention.
    rng = np.random.default_rng(718)
    for k in range(4):
        for surplus in (False, True):
            train = [_fallback(rng, k, 1, 2, False, False, surplus),
                     _fallback(rng, k, 3, 4, True, False, False)]
            test = [_fallback(rng, k, 6, 7, True, True, surplus)]
            validate_generated_puzzle(dict(train=train, test=test))
    # One wrong copied pixel is rejected; negative scan evidence is also binding.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][1, 1] = (int(bad['test'][0]['output'][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output accepted')
    return {'seeds_checked': 42, 'pairs_checked': len(records),
            'board_shapes': len({f['shape'] for f in records}),
            'structural_signatures': len({(f['shape'], f['start'], f['length'], f['used'], f['components'])
                                          for f in records})}
