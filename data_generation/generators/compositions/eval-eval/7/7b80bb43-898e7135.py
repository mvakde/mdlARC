"""Restore uncertain seams using a scaled inventory of cavity-shaped patches.

The solver searches visible repair cues and exact D4 shape matches jointly.
Generation constructs labels from walls and chosen gates, not from solve().
"""
from collections import Counter
from functools import lru_cache
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    answer = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, found = [start], []
        while stack:
            r, c = stack.pop()
            found.append((r, c))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in cells:
                    cells.remove(p)
                    stack.append(p)
        answer.append(tuple(sorted(found)))
    return answer


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells) + 1,
            min(c for _, c in cells), max(c for _, c in cells) + 1)


def _normalise(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _rotate(shape, turns):
    for _ in range(turns % 4):
        shape = tuple((c, -r) for r, c in shape)
    return _normalise(shape)


@lru_cache(maxsize=8192)
def _orientations(shape, reflection=True):
    bases = (shape, tuple((r, -c) for r, c in shape)) if reflection else (shape,)
    return tuple(sorted({_rotate(s, q) for s in bases for q in range(4)}))


def _canonical(shape):
    return _orientations(_normalise(shape))[0]


def _mask(shape):
    a = np.zeros((max(r for r, _ in shape)+1, max(c for _, c in shape)+1), bool)
    for r, c in shape:
        a[r, c] = True
    return a


def _at(a, r, c):
    return 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and bool(a[r, c])


def _strong(mask):
    """Membership in a horizontal or vertical run of at least three cells."""
    result = np.zeros_like(mask)
    for m, s in ((mask, result), (mask.T, result.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for offset in range(3):
            s[:, offset:offset+triples.shape[1]] |= triples
    return result


def _gaps(mask):
    answer = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        bracket = m[:, :-4] & m[:, 4:]
        bracket &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(bracket):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            answer.append((r, c, dr, dc))
    return answer


def _trace_geometry(raw):
    """Erase diagonal-only strokes and identify OPTIONAL three-cell repairs.

    All decisions here use the unmodified drawing. The inventory, examined
    later, determines which eligible repairs actually take place.
    """
    strong = _strong(raw)
    candidates, anchors = [], np.zeros_like(raw)
    for r, c, dr, dc in _gaps(raw):
        # A bracket must have an endpoint supported by real straight linework.
        if not (strong[r, c] or strong[r+4*dr, c+4*dc]):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r+endpoint*dr, c+endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                if (all(_at(raw, rr, cc) and not strong[rr, cc] for rr, cc in spur)
                        and not _at(raw, ar+3*sr, ac+3*sc)):
                    matches.append(spur)
        if matches:
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            candidates.append((gap, tuple(matches)))
            anchors[r, c] = anchors[r+4*dr, c+4*dc] = True
    weak = raw & ~strong & ~anchors
    erase = np.zeros_like(raw)
    for dr, dc in ((1, 1), (1, -1)):
        for r, c in np.argwhere(weak):
            r, c = int(r), int(c)
            if _at(weak, r+dr, c+dc):
                erase[r, c] = erase[r+dr, c+dc] = True
    return raw & ~erase, candidates, erase


def _decode(cells, scale):
    r0, r1, c0, c1 = _bbox(cells)
    if (r1-r0) % scale or (c1-c0) % scale:
        return None
    blocks = Counter(((r-r0)//scale, (c-c0)//scale) for r, c in cells)
    if any(n != scale*scale for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _recover(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= inp.size//2:
        raise ValueError("the background must occupy more than half the input")
    solutions, seen_boards = [], set()
    for wall in range(10):
        if wall == bg:
            continue
        for component in _components(inp == wall):
            bounds = _bbox(component)
            if (wall, bounds) in seen_boards:
                continue
            seen_boards.add((wall, bounds))
            r0, r1, c0, c1 = bounds
            board = inp[r0:r1, c0:c1]
            if not (5 <= min(board.shape) and max(board.shape) <= 15
                    and np.all(board[0] == wall) and np.all(board[-1] == wall)
                    and np.all(board[:, 0] == wall) and np.all(board[:, -1] == wall)
                    and np.all((board == wall) | (board == bg))):
                continue
            base, cues, erased = _trace_geometry(board == wall)
            if not 1 <= len(cues) <= 6 or not erased.any():
                continue
            if any(len(matches) != 1 or not all(erased[p] for p in matches[0])
                   for _, matches in cues):
                continue
            outside = np.ones(inp.shape, bool)
            outside[r0:r1, c0:c1] = False
            # The complete frame is the only object using the wall color.
            if np.any(outside & (inp == wall)):
                continue
            moat = np.zeros(inp.shape, bool)
            moat[max(0, r0-1):min(inp.shape[0], r1+1),
                 max(0, c0-1):min(inp.shape[1], c1+1)] = True
            if np.any(moat & outside & (inp != bg)):
                continue
            external = [(color, cells) for color in range(10) if color != bg
                        for cells in _components(outside & (inp == color))]
            if len(external) != len(_components(outside & (inp != bg))):
                continue
            large = [(color, cells) for color, cells in external if len(cells) >= 4]
            noise = [(color, cells) for color, cells in external if len(cells) < 4]
            if not 1 <= len(large) <= 9:
                continue
            for scale in range(2, min(3, 30//max(board.shape))+1):
                pieces = [(color, _decode(cells, scale), cells) for color, cells in large]
                if any(shape is None for _, shape, _ in pieces):
                    continue
                groups = {}
                for color, shape, cells in pieces:
                    groups.setdefault(_canonical(shape), []).append((color, shape, cells))
                if any(len({p[0] for p in group}) != 1 for group in groups.values()):
                    continue
                wanted = Counter({key: len(group) for key, group in groups.items()})
                areas = sorted(len(shape) for _, shape, _ in pieces)
                coarse = 0
                local = []
                for bits in range(1 << len(cues)):
                    mask = base.copy()
                    accepted = tuple(i for i in range(len(cues)) if bits & (1 << i))
                    for i in accepted:
                        for p in cues[i][0]:
                            mask[p] = True
                    holes = _components(~mask)
                    if len(holes) != len(pieces) or sorted(map(len, holes)) != areas:
                        continue
                    coarse += 1
                    if Counter(_canonical(h) for h in holes) != wanted:
                        continue
                    logical = np.full(board.shape, wall, np.int8)
                    reflection = rotation = 0
                    for cells in holes:
                        key = _canonical(cells)
                        for p in cells:
                            logical[p] = groups[key][0][0]
                        if len(groups[key]) == 1:
                            destination, loose = _normalise(cells), groups[key][0][1]
                            if destination not in _orientations(loose, False):
                                reflection += 1
                            elif destination != loose:
                                rotation += 1
                    expected = np.repeat(np.repeat(logical, scale, 0), scale, 1)
                    info = dict(background=bg, wall=wall, scale=scale, bounds=bounds,
                                board_shape=tuple(board.shape), base=base, final=mask,
                                cues=cues, erased=erased, accepted=accepted, holes=holes,
                                pieces=pieces, noise=noise, reflection=reflection,
                                rotation=rotation, classes=tuple(sorted(groups)),
                                repairs=len(accepted), rejected=len(cues)-len(accepted),
                                piece_count=len(pieces), areas=tuple(areas))
                    local.append((expected, info))
                for expected, info in local:
                    info['coarse_matches'] = coarse
                    solutions.append((expected, info))
    if len(solutions) != 1:
        raise ValueError("expected one frame, scale, repair subset and colored assembly")
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the complete answer from this input alone."""
    return _recover(input_grid)[0].copy()


def _chambers(heights, widths):
    ys, xs = [0], [0]
    for h in heights:
        ys.append(ys[-1]+h+1)
    for w in widths:
        xs.append(xs[-1]+w+1)
    mask = np.zeros((ys[-1]+1, xs[-1]+1), bool)
    mask[ys, :] = True
    mask[:, xs] = True
    edges = [('v', i, j) for i in range(len(heights)) for j in range(1, len(widths))]
    edges += [('h', i, j) for i in range(1, len(heights)) for j in range(len(widths))]
    return mask, ys, xs, edges


def _door(edge, ys, xs, offset):
    axis, i, j = edge
    if axis == 'v':
        r, c, dr, dc = ys[i]+offset, xs[j], 1, 0
    else:
        r, c, dr, dc = ys[i], xs[j]+offset, 0, 1
    return (r, c, dr, dc), tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))


def _add_stubs(rng, mask, ys, xs):
    """Small wall extensions and occasional structural islands diversify faces."""
    for i in range(len(ys)-1):
        for j in range(len(xs)-1):
            if rng.random() >= .48:
                continue
            r0, r1, c0, c1 = ys[i]+1, ys[i+1], xs[j]+1, xs[j+1]
            options = []
            for length in (1, 2):
                for r in range(r0, r1):
                    options += [tuple((r, c0+k) for k in range(length)),
                                tuple((r, c1-1-k) for k in range(length))]
                for c in range(c0, c1):
                    options += [tuple((r0+k, c) for k in range(length)),
                                tuple((r1-1-k, c) for k in range(length))]
            if r1-r0 >= 3 and c1-c0 >= 3 and rng.random() < .22:
                options.append(((int(rng.integers(r0+1, r1-1)),
                                 int(rng.integers(c0+1, c1-1))),))
            for index in rng.permutation(len(options)):
                trial = mask.copy()
                for p in options[int(index)]:
                    trial[p] = True
                room = ~trial[r0:r1, c0:c1]
                if room.any() and len(_components(room)) == 1:
                    mask = trial
                    break
    return mask


def _draw_traces(rng, base, descriptions):
    proposals = []
    for r, c, dr, dc in descriptions:
        choices = []
        for endpoint in (0, 4):
            ar, ac = r+endpoint*dr, c+endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr+side*dc, toward*dc+side*dr
                cells = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                if all(0 < rr < base.shape[0]-1 and 0 < cc < base.shape[1]-1
                       and not base[rr, cc] for rr, cc in cells):
                    choices.append(cells)
        if not choices:
            return None
        choices = [choices[int(i)] for i in rng.permutation(len(choices))]
        gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
        proposals.append((gap, choices))
    proposals.sort(key=lambda p: len(p[1]))
    visited = 0

    def place(index, raw, planned):
        nonlocal visited
        if index == len(proposals):
            return raw
        gap, choices = proposals[index]
        for cells in choices:
            visited += 1
            if visited > 350:
                return None
            if any(raw[p] for p in cells):
                continue
            trial = raw.copy()
            for p in cells:
                trial[p] = True
            clean, cues, erased = _trace_geometry(trial)
            if (np.array_equal(clean, base) and {g for g, _ in cues} == planned | {gap}
                    and all(len(matches) == 1 for _, matches in cues)
                    and all(_straight_diagonal(c) for c in _components(erased, True))):
                answer = place(index+1, trial, planned | {gap})
                if answer is not None:
                    return answer
        return None

    raw = place(0, base.copy(), set())
    if raw is None:
        return None
    planned = {g for g, _ in proposals}
    # Optional detached strokes are erased by the same geometric rule.
    for _ in range(int(rng.integers(3))):
        for _ in range(12):
            r = int(rng.integers(1, raw.shape[0]-2))
            c = int(rng.integers(1, raw.shape[1]-2))
            dc = int(rng.choice([-1, 1]))
            length = int(rng.choice([2, 3]))
            cells = [(r+k, c+k*dc) for k in range(length)]
            if not all(0 < rr < raw.shape[0]-1 and 0 < cc < raw.shape[1]-1
                       and not raw[rr, cc] for rr, cc in cells):
                continue
            trial = raw.copy()
            for p in cells:
                trial[p] = True
            b, cs, removed = _trace_geometry(trial)
            if (np.array_equal(b, base) and {gap for gap, _ in cs} == planned
                    and all(len(matches) == 1 for _, matches in cs)
                    and all(_straight_diagonal(c) for c in _components(removed, True))):
                raw = trial
                break
    return raw


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, pieces, scale, bg, wall, boundary):
    masks = [np.ones(board.shape, bool)]
    masks += [np.repeat(np.repeat(_mask(shape), scale, 0), scale, 1)
              for _, shape in pieces]
    for attempt in range(55):
        shape = (int(rng.integers(24, 31)), int(rng.integers(24, 31))) if attempt < 20 else (30, 30)
        canvas, blocked = np.full(shape, bg, np.int8), np.zeros(shape, bool)
        order = sorted(range(len(masks)), key=lambda i: (-masks[i].size, float(rng.random())))
        for index in order:
            mask = masks[index]
            h, w = mask.shape
            options = _free_origins(blocked, h, w)
            if boundary and index == 0:
                options = np.array([p for p in options if p[0] == 0 or p[1] == 0
                                    or p[0]+h == shape[0] or p[1]+w == shape[1]], int).reshape(-1, 2)
            if len(options) == 0:
                break
            r, c = map(int, options[int(rng.integers(len(options)))])
            view = canvas[r:r+h, c:c+w]
            if index == 0:
                view[:] = board
            else:
                view[mask] = pieces[index-1][0]
            blocked[max(0, r-1):min(shape[0], r+h+1),
                    max(0, c-1):min(shape[1], c+w+1)] = True
        else:
            for _ in range(int(rng.integers(7))):
                # Noise uses piece colors, so filtering by a special noise color fails.
                cells = ((0, 0),) if rng.random() < .55 else ((0, 0), (0, 1))
                if rng.random() < .2:
                    cells = ((0, 0), (0, 1), (1, 0))
                noise_mask = _mask(_rotate(cells, int(rng.integers(4))))
                h, w = noise_mask.shape
                options = _free_origins(blocked, h, w)
                if not len(options):
                    break
                r, c = map(int, options[int(rng.integers(len(options)))])
                canvas[r:r+h, c:c+w][noise_mask] = pieces[int(rng.integers(len(pieces)))][0]
                blocked[max(0, r-1):min(shape[0], r+h+1),
                        max(0, c-1):min(shape[1], c+w+1)] = True
            if np.count_nonzero(canvas == bg) <= canvas.size//2:
                continue
            if rng.random() < .7:
                r0, r1, c0, c1 = _bbox(np.argwhere(canvas != bg))
                r0, c0 = max(0, r0-int(rng.integers(4))), max(0, c0-int(rng.integers(4)))
                r1, c1 = min(shape[0], r1+int(rng.integers(4))), min(shape[1], c1+int(rng.integers(4)))
                cropped = canvas[r0:r1, c0:c1]
                if np.count_nonzero(cropped == bg) > cropped.size//2:
                    canvas = cropped.copy()
            return canvas
    return None


def _make_pair(rng, scale, kind, boundary=False):
    for attempt in range(240):
        witness = kind == 'witness'
        if witness:
            heights, widths = [6], [3, 4, 3]
        elif scale == 3:
            heights, widths = [3, 3], [3, 3]
            if rng.random() < .4:
                (heights if rng.integers(2) else widths)[int(rng.integers(2))] = 4
        else:
            layouts = [(2, 2)] if kind == 'junction' else ([(2, 3), (3, 2)] if kind == 'rich'
                                                        else [(1, 3), (2, 2), (2, 3), (3, 2), (3, 3)])
            nr, nc = layouts[int(rng.integers(len(layouts)))]
            heights, widths = [3]*nr, [3]*nc
            for sizes in (heights, widths):
                for i in range(len(sizes)):
                    if sum(sizes)+len(sizes)+1 < 15 and rng.random() < .5:
                        sizes[i] += int(rng.integers(1, min(3, 15-sum(sizes)-len(sizes))))
        full, ys, xs, edges = _chambers(heights, widths)
        if not witness:
            full = _add_stubs(rng, full, ys, xs)
        if max(full.shape)*scale > 30:
            continue
        if witness:
            selected, cue_ids, closed = edges, [0, 1], [int(rng.integers(2))]
            offsets = [0, 1]
            if rng.integers(2):
                offsets.reverse()
        else:
            total = int(rng.integers(2, min(8, len(edges))+1))
            selected = [edges[int(i)] for i in rng.permutation(len(edges))[:total]]
            ncues = int(rng.integers(2, min(6, total)+1))
            cue_ids = list(map(int, rng.permutation(total)[:ncues]))
            # Repairs and intentionally open traced gates coexist.
            nclose = int(rng.integers(1, ncues))
            if scale == 3:
                nclose = max(nclose, ncues-1)
            closed = list(map(int, rng.choice(cue_ids, nclose, replace=False)))
            offsets = [int(rng.integers((heights[i] if axis == 'v' else widths[j])-2))
                       for axis, i, j in selected]
        descriptions, gaps = zip(*[_door(edge, ys, xs, offset)
                                   for edge, offset in zip(selected, offsets)])
        base = full.copy()
        for gap in gaps:
            for p in gap:
                base[p] = False
        if not np.array_equal(_trace_geometry(base)[0], base):
            continue
        final = base.copy()
        for index in closed:
            for p in gaps[index]:
                final[p] = True
        holes = _components(~final)
        if not 2 <= len(holes) <= 9 or any(not h for h in holes):
            continue
        raw = _draw_traces(rng, base, [descriptions[i] for i in cue_ids])
        if raw is None:
            continue
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        raw, final = np.rot90(raw, turns), np.rot90(final, turns)
        if flip:
            raw, final = raw[:, ::-1], final[:, ::-1]
        holes = _components(~final)
        classes = list(dict.fromkeys(_canonical(h) for h in holes))
        palette = list(map(int, rng.permutation(10)))
        bg, wall = palette[:2]
        cardinality = len(classes) if witness else int(rng.integers(1, min(8, len(classes))+1))
        colors = {key: palette[2+i % cardinality] for i, key in enumerate(classes)}
        logical = np.full(final.shape, wall, np.int8)
        loose, reflected = [], False
        for hole in holes:
            shape, key = _normalise(hole), _canonical(hole)
            options = _orientations(shape)
            mirrors = tuple(s for s in options if s not in _orientations(shape, False))
            if witness and mirrors and not reflected:
                options, reflected = mirrors, True
            loose.append((colors[key], options[int(rng.integers(len(options)))]))
            for p in hole:
                logical[p] = colors[key]
        if witness and not reflected:
            continue
        board = np.where(raw, wall, bg).astype(np.int8)
        canvas = _pack(rng, board, loose, scale, bg, wall, boundary)
        if canvas is None:
            continue
        output = np.repeat(np.repeat(logical, scale, 0), scale, 1)
        try:
            recovered, info = _recover(canvas)
        except ValueError:
            continue
        if (np.array_equal(recovered, output) and info['repairs'] >= 1 and info['rejected'] >= 1
                and (not witness or info['coarse_matches'] >= 2 and info['reflection'] >= 1)):
            return {'input': canvas, 'output': output}
    raise RuntimeError(f"bounded construction exhausted for {kind}, scale {scale}")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    return int(value)


def _grid_key(grid):
    a = _grid(grid)
    return tuple(a.shape), a.tobytes()


def _puzzle_key(puzzle):
    return tuple((_grid_key(p['input']), _grid_key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['topology', 'scale', 'boundary']))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for index in range(count):
            if split == 'train':
                kind = 'witness' if index == 0 else (
                    'junction' if index == 1 or shift == 'topology' else 'mixed')
                scale = 2 if shift in ('scale', 'topology') or index == 0 else int(rng.choice([2, 3]))
            else:
                kind = 'rich' if shift == 'topology' else 'junction'
                scale = 3 if shift == 'scale' else 2
            for _ in range(12):
                pair = _make_pair(rng, scale, kind, split == 'test' and shift == 'boundary')
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _straight_diagonal(cells):
    cells = sorted(cells)
    return (len(cells) in (2, 3) and all(cells[i+1][0]-cells[i][0] == 1
            for i in range(len(cells)-1)) and len({cells[i+1][1]-cells[i][1]
            for i in range(len(cells)-1)}) == 1 and abs(cells[1][1]-cells[0][1]) == 1)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), 'incorrect or incomplete assembly'
            assert info['repairs'] >= 1 and info['rejected'] >= 1
            assert 2 <= info['piece_count'] <= 9 and info['scale'] in (2, 3)
            assert all(len(cells) <= 3 for _, cells in info['noise'])
            assert all(_straight_diagonal(c) for c in _components(info['erased'], True))
            assert np.count_nonzero(info['final'] & ~info['base']) == 3*info['repairs']
            assert out.shape == tuple(d*info['scale'] for d in info['board_shape'])
            # No input-derived structure is lost to enlargement or assembly.
            small = out[::info['scale'], ::info['scale']]
            assert np.array_equal(small == info['wall'], info['final'])
            assert np.array_equal(np.repeat(np.repeat(small, info['scale'], 0), info['scale'], 1), out)
            if split == 'train':
                train.append(info)
    assert any(i['coarse_matches'] >= 2 for i in train), 'training must refute area-only repair choice'
    assert any(i['reflection'] for i in train), 'training must demonstrate reflection'
    assert len({i['classes'] for i in train}) >= 2


def self_test():
    """Bounded diversity, counterfactual and mutation checks on 42 fixed seeds."""
    puzzles, infos, normalized = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, info = _recover(pair['input'])
                assert np.array_equal(predicted, pair['output'])
                infos.append(info)
                roles = {}
                normalized.add((pair['input'].shape, tuple(roles.setdefault(int(c), len(roles))
                                                          for c in pair['input'].ravel())))
                # Omitting A leaves every eligible repair open; B's inventory fails.
                raw_holes = Counter(_canonical(h) for h in _components(~info['base']))
                target = Counter(_canonical(s) for _, s, _ in info['pieces'])
                assert raw_holes != target
                # Omitting B and closing all spurred gaps makes wrong walls.
                indiscriminate = info['base'].copy()
                for gap, _ in info['cues']:
                    for p in gap:
                        indiscriminate[p] = True
                assert not np.array_equal(indiscriminate, info['final'])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(normalized) >= 180
    assert len({i['board_shape'] for i in infos}) >= 20
    assert len({i['areas'] for i in infos}) >= 35
    assert len({key for i in infos for key in i['classes']}) >= 80
    assert {i['scale'] for i in infos} == {2, 3}
    assert max(len(i['cues']) for i in infos) >= 5
    assert max(i['piece_count'] for i in infos) >= 5
    assert max(i['repairs'] for i in infos) >= 3
    assert any(len(_components(i['final'])) > 1 for i in infos)
    assert len({(i['background'], i['wall']) for i in infos}) >= 40
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    # Rotation and reflection of the WHOLE input also transform the cropped answer.
    pair = puzzles[0]['train'][0]
    for turns in range(4):
        for flip in (False, True):
            inp, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
            if flip:
                inp, out = inp[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(inp), out)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(592761805, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
           for s in ('train', 'test')}
    _, info = _recover(bad['train'][0]['input'])
    # Deliberately omit one required three-cell wall in an otherwise solved label.
    scale = info['scale']
    for r, c in info['cues'][info['accepted'][0]][0]:
        bad['train'][0]['output'][r*scale:(r+1)*scale, c*scale:(c+1)*scale] = info['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('omitted wall repair accepted')
    # Corrupt a single macro-pixel inside a genuine loose piece. No smaller
    # component/color heuristic may silently discard the now-damaged patch.
    broken = puzzles[0]['train'][0]['input'].copy()
    _, info = _recover(broken)
    color, _, cells = info['pieces'][0]
    broken[cells[0]] = info['background']
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete macrocell accepted')
    # Equal gate offsets make the two possible single repairs mirror-related.
    # Their inventories agree, but their final wall locations differ: reject.
    rng = np.random.default_rng(1097)
    frame, ys, xs, edges = _chambers([6], [3, 4, 3])
    descriptions, gaps = zip(*[_door(edge, ys, xs, 0) for edge in edges])
    base = frame.copy()
    for gap in gaps:
        for p in gap:
            base[p] = False
    raw = _draw_traces(rng, base, descriptions)
    assert raw is not None
    final = base.copy()
    for p in gaps[0]:
        final[p] = True
    pieces = [(2+i, _normalise(hole)) for i, hole in enumerate(_components(~final))]
    ambiguous = _pack(rng, np.where(raw, 5, 0).astype(np.int8), pieces, 2, 0, 5, False)
    assert ambiguous is not None
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous repair locations accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, geometry variation and counterfactual checks')
