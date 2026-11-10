"""Cavity relays: negative-space lookup and board-line feedback.

Only the standard library and NumPy are used.  The renderer knows construction
coordinates; solve reconstructs everything from the input pixels.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            p, q = stack.pop()
            cells.append((p, q))
            for x, y in ((p - 1, q), (p + 1, q), (p, q - 1), (p, q + 1)):
                if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    stack.append((x, y))
        result.append(cells)
    return result


def _trim(mask):
    rr, cc = np.nonzero(mask)
    _require(len(rr) > 0, "empty glyph")
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _key(mask, ink):
    m = _trim(mask)
    return int(ink), m.shape, m.tobytes()


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _scene(grid):
    """Canonicalize to objects on the left and the framed board on the right."""
    candidates = []
    for turn in range(4):
        g = np.rot90(grid, turn)
        bg = int(g[0, 0])
        if not all(int(g[r, c]) == bg for r, c in
                   ((0, 0), (0, g.shape[1] - 1), (g.shape[0] - 1, 0),
                    (g.shape[0] - 1, g.shape[1] - 1))):
            continue
        for col in range(1, g.shape[1] - 1):
            rail = int(g[0, col])
            if rail == bg or not np.all(g[:, col] == rail):
                continue
            if np.any(g[:, :col] == rail) or not np.any(g[:, :col] != bg):
                continue
            if not np.any(g[:, col + 1:] == rail):
                continue
            candidates.append((g, turn, bg, rail, col))
    _require(len(candidates) == 1, "ambiguous rail/board orientation")
    return candidates[0]


def _analyze(input_grid):
    g, turn, bg, rail, col = _scene(_grid(input_grid))
    rr, cc = np.where(g[:, col + 1:] == rail)
    r0, r1 = int(rr.min()), int(rr.max())
    c0, c1 = int(cc.min()) + col + 1, int(cc.max()) + col + 1
    _require(0 < r0 < r1 < g.shape[0] - 1 and col + 1 < c0 < c1 < g.shape[1] - 1,
             "board must have exterior margins and a beam gutter")
    row_lines = [r for r in range(r0, r1 + 1) if np.all(g[r, c0:c1 + 1] == rail)]
    col_lines = [c for c in range(c0, c1 + 1) if np.all(g[r0:r1 + 1, c] == rail)]
    n = len(row_lines) - 1
    _require(n in (3, 4) and len(col_lines) == n + 1, "board must be square")
    _require(row_lines[0] == r0 and row_lines[-1] == r1 and
             col_lines[0] == c0 and col_lines[-1] == c1, "unclosed board")
    dh, dw = row_lines[1] - r0, col_lines[1] - c0
    _require(dh in (3, 4) and dw in (3, 4), "unsupported cell canvas")
    _require(all(b - a == dh for a, b in zip(row_lines, row_lines[1:])) and
             all(b - a == dw for a, b in zip(col_lines, col_lines[1:])), "irregular lattice")
    board, lookup = [], {}
    expected = np.full(g.shape, bg, np.int8)
    expected[:, col] = rail
    expected[r0:r1 + 1, c0:c1 + 1] = g[r0:r1 + 1, c0:c1 + 1]
    for br, bc in itertools.product(range(n), repeat=2):
        rs = slice(row_lines[br] + 1, row_lines[br + 1])
        cs = slice(col_lines[bc] + 1, col_lines[bc + 1])
        block = g[rs, cs]
        inks = np.unique(block[block != bg])
        _require(len(inks) == 1 and int(inks[0]) != rail, "cell must contain one glyph ink")
        mask = block != bg
        k = _key(mask, int(inks[0]))
        _require(k not in lookup, "duplicate colored glyph")
        lookup[k] = br * n + bc
        board.append(dict(key=k, mask=_trim(mask), ink=int(inks[0]), rs=rs, cs=cs))
    for r in row_lines:
        _require(np.all(g[r, c0:c1 + 1] == rail), "broken horizontal divider")
    for c in col_lines:
        _require(np.all(g[r0:r1 + 1, c] == rail), "broken vertical divider")
    objects, active, inactive, open_rows = [], [], [], set()
    previous_end = -1
    for cells in _components(g[:, :col] != bg):
        rows, cols = np.array(cells).T
        a, b, x, y = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
        _require(a > 0 and b < g.shape[0] - 1, "object must leave top/bottom margins")
        _require(a > previous_end + 1, "objects require separated row bands")
        previous_end = b
        inks = {int(g[r, c]) for r, c in cells}
        _require(len(inks) == 1, "carrier must be monochrome")
        ink = inks.pop()
        m = np.zeros((b - a + 1, y - x + 1), bool)
        m[rows - a, cols - x] = True
        holes, enclosed = [], set()
        for region in _components(~m):
            if any(r in (0, m.shape[0] - 1) or c in (0, m.shape[1] - 1) for r, c in region):
                continue
            enclosed.update(region)
            hr, hc = np.array(region).T
            hm = np.zeros(m.shape, bool)
            hm[hr, hc] = True
            k = _key(hm, ink)
            _require(k in lookup, "every enclosed cavity needs an exact board copy")
            item = dict(key=k, pos=lookup[k], rows={a + int(r) for r in hr},
                        contact_rows={a + int(r) for r, c in region if c == m.shape[1] - 2},
                        active=int(hc.max()) == m.shape[1] - 2, shape=_trim(hm))
            # The generated and intended family avoids partial-contact ambiguity.
            if item['active']:
                _require(item['rows'] == item['contact_rows'], "partial extreme contact excluded")
            holes.append(item)
            (active if item['active'] else inactive).append(item)
        if m.shape[1] >= 3:
            for r in range(m.shape[0]):
                if not m[r, -2] and m[r, -1] and (r, m.shape[1] - 2) not in enclosed:
                    open_rows.add(a + r)
        shift = col - 1 - y
        _require(shift >= 0, "carrier crosses rail")
        objects.append(dict(cells=cells, mask=m, ink=ink, shift=shift, holes=holes))
        for r, c in cells:
            expected[r, c] = g[r, c]
    _require(np.array_equal(g, expected), "ink outside the rail, carriers, or board")
    positions = {h['pos'] for h in active}
    _require(1 <= len(active) <= n and len(positions) == len(active), "invalid or repeated active keys")
    return dict(grid=g, turn=turn, bg=bg, rail=rail, col=col, n=n, dh=dh, dw=dw,
                board=board, board_bounds=(r0, r1, c0, c1), objects=objects,
                active=active, inactive=inactive, open_rows=open_rows, positions=positions)


def _line_kind(positions, n):
    p = set(positions)
    if len(p) == n and len({x // n for x in p}) == 1:
        return 'row'
    if len(p) == n and len({x % n for x in p}) == 1:
        return 'col'
    return None


def _predict(info, positions=None, force_beams=False):
    """Input-derived output; optional arguments are used only for ablation tests."""
    g, bg, rail, col = (info[k] for k in ('grid', 'bg', 'rail', 'col'))
    selected = info['positions'] if positions is None else set(positions)
    out = g.copy()
    out[:, :col] = bg
    for o in info['objects']:
        for r, c in o['cells']:
            out[r, c + o['shift']] = o['ink']
    for p in selected:
        cell = info['board'][p]
        block = out[cell['rs'], cell['cs']]
        block[block == bg] = rail
    if force_beams or _line_kind(selected, info['n']):
        for hole in info['active']:
            for r in hole['rows']:
                # Obstacles are read from the input, so highlighting cannot
                # change beam lengths. The rail itself is crossed, not painted.
                for c in range(col + 1, g.shape[1]):
                    if g[r, c] != bg:
                        break
                    out[r, c] = rail
    return np.rot90(out, -info['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone, including the visible signal-color convention."""
    return _predict(_analyze(input_grid))


@lru_cache(maxsize=8)
def _banks(gh, gw):
    bank = {}
    for code in range(1, 1 << (gh * gw)):
        m = np.array([(code >> i) & 1 for i in range(gh * gw)], bool).reshape(gh, gw)
        m = _trim(m)
        bank[(m.shape, m.tobytes())] = m
    all_masks = list(bank.values())
    # Connected, right-justified rows all meet the extreme leading wall.
    apertures = [m for m in all_masks if all(
        row[-1] and np.all(row[np.flatnonzero(row)[0]:]) for row in m)]
    return all_masks, apertures


def _inventory(rng, n, gh, gw, inks):
    bank, apertures = _banks(gh, gw)
    chosen = list(map(int, rng.choice(len(apertures), n, replace=False)))
    tall = [i for i, m in enumerate(apertures) if m.shape[0] >= 2]
    if not any(apertures[i].shape[0] >= 2 for i in chosen):
        chosen[0] = int(rng.choice(tall))
    primary = int(rng.choice(inks))
    secondary = int(rng.choice([c for c in inks if c != primary]))
    colors = [primary] * n
    if rng.random() < .45:
        colors[int(rng.integers(n))] = secondary
    tokens = [(apertures[i].copy(), c) for i, c in zip(chosen, colors)]
    keys = {_key(m, c) for m, c in tokens}
    small = [m for m in apertures if m.shape[0] <= 2]
    choices = [(m, int(c)) for m in small for c in inks if _key(m, c) not in keys]
    rng.shuffle(choices)
    for m, c in choices[:2]:
        tokens.append((m.copy(), c))
        keys.add(_key(m, c))
    # Every possible active key has a same-shape, different-color decoy,
    # including keys chosen by a later one-cavity partial query.
    for m, c in tokens[:n]:
        clone_ink = secondary if c != secondary else primary
        if _key(m, clone_ink) not in keys:
            tokens.append((m.copy(), clone_ink))
            keys.add(_key(m, clone_ink))
    for ink in sorted(set(colors)):
        if not any(c == ink for _, c in tokens[n:]):
            choices = [m for m in bank if _key(m, ink) not in keys]
            m = choices[int(rng.integers(len(choices)))]
            tokens.append((m.copy(), ink))
            keys.add(_key(m, ink))
    remainder = [(m, int(c)) for m in bank for c in inks if _key(m, c) not in keys]
    rng.shuffle(remainder)
    for m, c in remainder[:n * n - len(tokens)]:
        tokens.append((m.copy(), c))
    _require(len(tokens) == n * n, "insufficient vocabulary")
    return tokens


def _positions(rng, n, kind, count):
    axis = kind if kind in ('row', 'col') else str(rng.choice(['row', 'col']))
    index = int(rng.integers(n))
    line = [index * n + i if axis == 'row' else i * n + index for i in range(n)]
    missing = int(rng.integers(n))
    if kind in ('row', 'col'):
        return line, None
    if kind == 'near':
        extra = int(rng.choice([p for p in range(n * n) if p not in line]))
        return line[:missing] + line[missing + 1:] + [extra], line[missing]
    options = [list(p) for p in itertools.combinations(range(n * n), count)
               if not _line_kind(p, n) and set(p) not in
               ({i * n + i for i in range(n)}, {i * n + n - 1 - i for i in range(n)})]
    return options[int(rng.integers(len(options)))], None


def _active_carrier(rng, tokens, style):
    ink = tokens[0][1]
    _require(all(c == ink for _, c in tokens), "mixed-ink carrier")
    width = max(m.shape[1] for m, _ in tokens) + 2 + int(rng.integers(2))
    height = sum(m.shape[0] + 1 for m, _ in tokens) + 1
    mask = np.ones((height, width), bool)
    rows, start = set(), 1
    for m, _ in tokens:
        h, w = m.shape
        block = mask[start:start + h, width - w - 1:width - 1]
        block[m] = False
        rows.update(range(start, start + h))
        start += h + 1
    if style == 'lobes' and len(tokens) > 1:
        start = 1
        for i, (m, _) in enumerate(tokens[:-1]):
            start += m.shape[0]
            # Cut only outside both neighboring chambers' back walls. A
            # one-column connector alone would open or merge the apertures.
            left_wall = width - max(m.shape[1], tokens[i + 1][0].shape[1]) - 2
            mask[start, :left_wall] = False
            start += 1
    # A back extension changes the silhouette without changing the leading face.
    if rng.random() < .35 and width > 3:
        trial = mask.copy()
        trial[0, 0] = False
        if len(_components(trial)) == 1:
            mask = trial
    return dict(mask=mask, ink=ink, active_rows=rows, category='active')


def _decoy_carrier(rng, token, kind):
    m, ink = token
    h, w = m.shape
    mask = np.ones((h + 2, w + 2), bool)
    mask[1:1 + h, 1:1 + w][m] = False
    if kind == 'open':
        row = int(rng.integers(h))
        left = int(np.flatnonzero(m[row])[0])
        mask[row + 1, :left + 1] = False
    else:
        extension = int(rng.integers(1, 3))
        extended = np.zeros((h + 2, w + 2 + extension), bool)
        extended[:, :w + 2] = mask
        extended[int(rng.choice([0, h + 1])), w + 2:] = True
        mask = extended
    return dict(mask=mask, ink=ink, active_rows=set(), category=kind)


def _group_keys(rng, selected, tokens, mode):
    order = list(selected)
    rng.shuffle(order)
    if mode == 'single':
        return [[p] for p in order]
    groups = []
    for ink in sorted({tokens[p][1] for p in selected}):
        same = [p for p in order if tokens[p][1] == ink]
        if mode == 'mixed' and len(same) >= 3:
            groups.extend([same[:1], same[1:]])
        else:
            groups.append(same)
    rng.shuffle(groups)
    return groups


def _render(rng, config, tokens, kind, count, grouping, test):
    n, gh, gw, bg, rail = (config[k] for k in ('n', 'gh', 'gw', 'bg', 'rail'))
    selected = sorted(map(int, rng.choice(n, count, replace=False)))
    positions, missing = _positions(rng, n, kind, count)
    board_map = [None] * (n * n)
    for p, key_index in zip(positions, selected):
        board_map[p] = key_index
    # A recessed key fills the gap in a near miss: ignoring A changes marking.
    if missing is not None:
        board_map[missing] = n
    rest = [i for i in range(n * n) if i not in board_map]
    rng.shuffle(rest)
    for p in range(n * n):
        if board_map[p] is None:
            board_map[p] = rest.pop()
    groups = _group_keys(rng, selected, tokens, grouping)
    objects = [_active_carrier(rng, [tokens[p] for p in group],
                               str(rng.choice(['solid', 'lobes']))) for group in groups]
    objects += [_decoy_carrier(rng, tokens[n], 'recessed'),
                _decoy_carrier(rng, tokens[n + 1], 'open')]
    # If singleton chambers exhaust the height budget, merge equal-colored
    # active keys. This only changes grouping, never a key or its eligibility.
    def minimum_height():
        return sum(o['mask'].shape[0] for o in objects) + len(objects) - 1 + 2
    if minimum_height() > 30:
        groups = _group_keys(rng, selected, tokens, 'compound')
        objects = [_active_carrier(rng, [tokens[p] for p in group], 'lobes') for group in groups]
        objects += [_decoy_carrier(rng, tokens[n], 'recessed'),
                    _decoy_carrier(rng, tokens[n + 1], 'open')]
    rng.shuffle(objects)
    board_h, board_w = n * (gh + 1) + 1, n * (gw + 1) + 1
    minimum = max(minimum_height(), board_h + 2)
    _require(minimum <= 30, "carriers do not fit")
    height = int(rng.integers(minimum, min(30, minimum + 4) + 1))
    widest = max(o['mask'].shape[1] for o in objects)
    minimum_work = widest + (0 if test else 1)
    max_gutter = min(3, 30 - minimum_work - board_w - 2)
    gutter = int(rng.integers(1, max_gutter + 1))
    margin = int(rng.integers(1, min(2, 30 - minimum_work - board_w - gutter - 1) + 1))
    max_work = 30 - board_w - gutter - margin - 1
    work = int(rng.integers(minimum_work, min(max_work, widest + 5) + 1))
    width = work + 1 + gutter + board_w + margin
    board_r = int(rng.integers(1, height - board_h))
    board_c = work + 1 + gutter
    inp = np.full((height, width), bg, np.int8)
    inp[:, work] = rail
    for i in range(n + 1):
        inp[board_r + i * (gh + 1), board_c:board_c + board_w] = rail
        inp[board_r:board_r + board_h, board_c + i * (gw + 1)] = rail
    for p, key_index in enumerate(board_map):
        br, bc = divmod(p, n)
        m, ink = tokens[key_index]
        h, w = m.shape
        r = board_r + br * (gh + 1) + 1 + (gh - h) // 2
        c = board_c + bc * (gw + 1) + 1 + (gw - w) // 2
        inp[r:r + h, c:c + w][m] = ink
    spare = height - minimum_height()
    gaps = rng.multinomial(spare, np.full(len(objects) + 1, 1 / (len(objects) + 1)))
    shifts = [int(rng.integers(work - o['mask'].shape[1] + 1)) for o in objects]
    if not test and not any(shifts):
        movable = [i for i, o in enumerate(objects) if o['mask'].shape[1] < work]
        shifts[int(rng.choice(movable))] = 1
    row, beam_rows, placed = 1 + int(gaps[0]), set(), []
    for i, o in enumerate(objects):
        m = o['mask']
        shift = shifts[i]
        c = work - m.shape[1] - shift
        inp[row:row + m.shape[0], c:c + m.shape[1]][m] = o['ink']
        beam_rows.update(row + r for r in o['active_rows'])
        placed.append((o, row, c, shift))
        row += m.shape[0] + 1 + int(gaps[i + 1])
    # Construction renderer: no solve(), parser, or recovered cavity classification.
    out = inp.copy()
    out[:, :work] = bg
    for o, r, c, shift in placed:
        m = o['mask']
        out[r:r + m.shape[0], c + shift:c + shift + m.shape[1]][m] = o['ink']
    for p in positions:
        br, bc = divmod(p, n)
        r, c = board_r + br * (gh + 1) + 1, board_c + bc * (gw + 1) + 1
        block = out[r:r + gh, c:c + gw]
        block[block == bg] = rail
    if kind in ('row', 'col'):
        for r in beam_rows:
            for c in range(work + 1, width):
                if inp[r, c] != bg:
                    break
                out[r, c] = rail
    turn = config['turn']
    if config['orientation_mode'] == 'mixed':
        turn = int(rng.integers(4))
    elif config['orientation_mode'] == 'shift' and test:
        turn = (turn + 2) % 4
    if config['flip']:
        inp, out = np.flipud(inp), np.flipud(out)
    return dict(input=np.rot90(inp, turn).copy(), output=np.rot90(out, turn).copy())


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside its permitted range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    colors = list(map(int, rng.permutation(10)))
    n = int(rng.choice([3, 4]))
    gh = 2 if n == 4 else int(rng.choice([2, 3]))
    config = dict(n=n, gh=gh, gw=int(rng.choice([2, 3])), bg=colors[0], rail=colors[1],
                  inks=colors[2:2 + int(rng.integers(2, 5))], turn=int(rng.integers(4)),
                  orientation_mode=str(rng.choice(['fixed', 'mixed', 'shift'], p=[.55, .30, .15])),
                  flip=bool(rng.integers(2)))
    # Two-example puzzles retain one orientation, and test only the taught axis.
    if nt == 2:
        config['orientation_mode'] = 'fixed'
    inventory_mode = str(rng.choice(['rearranged', 'fresh']))
    anchor = _inventory(rng, n, gh, config['gw'], config['inks'])
    first_axis = str(rng.choice(['row', 'col']))
    other_axis = 'col' if first_axis == 'row' else 'row'
    train_schedule = [(first_axis, n, 'single'), ('near', n, 'compound')]
    if nt >= 3:
        train_schedule.append((other_axis, n, 'mixed'))
    if nt >= 4:
        train_schedule.append(('partial', int(rng.integers(1, n)), 'mixed'))
    while len(train_schedule) < nt:
        kind = str(rng.choice(['row', 'col', 'near', 'scatter', 'partial']))
        count = int(rng.integers(1, n)) if kind == 'partial' else n
        train_schedule.append((kind, count, str(rng.choice(['single', 'compound', 'mixed']))))
    test_schedule = []
    for _ in range(nq):
        kinds = [first_axis, other_axis] if nt >= 3 else [first_axis]
        kinds += ['near', 'scatter'] + (['partial'] if nt >= 4 else [])
        kind = str(rng.choice(kinds))
        count = int(rng.integers(1, n)) if kind == 'partial' else n
        test_schedule.append((kind, count, 'mixed'))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, schedule in (('train', train_schedule), ('test', test_schedule)):
        for index, (kind, count, grouping) in enumerate(schedule):
            # The first two demonstrations deliberately reuse all colored keys:
            # rearranging the same query into a near miss disables its beams.
            reuse = (split == 'train' and index < 2) or inventory_mode == 'rearranged'
            tokens = anchor if reuse else _inventory(rng, n, gh, config['gw'], config['inks'])
            for _ in range(32):
                pair = _render(rng, config, tokens, kind, count, grouping, split == 'test')
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness attempts exhausted')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "invalid split")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "dimensions changed")
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _analyze(inp)
            _require(np.array_equal(out, _predict(info)), "output violates input-derived rule")
            canonical_out = np.rot90(out, info['turn'])
            g, col, rail = info['grid'], info['col'], info['rail']
            _require(np.array_equal(canonical_out[:, col], g[:, col]), "rail changed")
            for cell in info['board']:
                before, after = g[cell['rs'], cell['cs']], canonical_out[cell['rs'], cell['cs']]
                _require(np.array_equal(before[before != info['bg']], after[before != info['bg']]),
                         "board ink changed")
            for o in info['objects']:
                _require(len(_components(o['mask'])) == 1, "disconnected carrier")
            # Same-shape/different-color and same-color/different-shape distractors
            # must be visible for a selected key, not merely present somewhere.
            keys = [b['key'] for b in info['board']]
            _require(any(keys[p][1:] == keys[q][1:] and keys[p][0] != keys[q][0]
                         for p in info['positions'] for q in range(info['n'] ** 2)
                         if q not in info['positions']), "no color-sensitive selection")
            _require(any(keys[p][0] == keys[q][0] and keys[p][1:] != keys[q][1:]
                         for p in info['positions'] for q in range(info['n'] ** 2)
                         if q not in info['positions']), "no shape-sensitive selection")
            infos[split].append(info)
    training = infos['train']
    _require(len({(i['bg'], i['rail'], i['n'], i['dh'], i['dw']) for s in infos for i in infos[s]}) == 1,
             "puzzle conventions changed")
    wins = { _line_kind(i['positions'], i['n']) for i in training }
    _require(bool(wins & {'row', 'col'}) and None in wins, "both outcomes must be demonstrated")
    _require(any(len(i['active']) == i['n'] and not _line_kind(i['positions'], i['n']) for i in training),
             "full-count failure is required")
    _require(any(i['inactive'] for i in training), "recessed-hole rejection not taught")
    _require(any(i['open_rows'] for i in training), "enclosure not taught")
    _require(any(o['shift'] > 0 for i in training for o in i['objects']), "alignment not taught")
    _require(any(len(h['rows']) >= 2 for i in training for h in i['active']), "multirow aperture not taught")
    _require(any(sum(h['active'] for h in o['holes']) >= 2 for i in training for o in i['objects']),
             "multiple apertures per carrier not taught")
    # Removing A admits a recessed key; removing B emits from every active key.
    _require(any(not np.array_equal(_predict(i), _predict(i, i['positions'] | {h['pos'] for h in i['inactive']}))
                 for i in training), "A ablation leaves all answers unchanged")
    _require(any(not np.array_equal(_predict(i), _predict(i, force_beams=True)) for i in training),
             "B ablation leaves all answers unchanged")
    # An intentional current-board contrast, observable without generation state.
    _require(any({h['key'] for h in a['active']} == {h['key'] for h in b['active']} and
                 bool(_line_kind(a['positions'], a['n'])) != bool(_line_kind(b['positions'], b['n']))
                 for a, b in itertools.combinations(training, 2)), "no board-rearrangement contrast")
    for i in infos['test']:
        kind = _line_kind(i['positions'], i['n'])
        _require(kind is None or kind in wins, "unwitnessed winning axis")
        if len(i['active']) < i['n']:
            _require(any(len(t['active']) < t['n'] for t in training), "partial queries not taught")


def self_test():
    """Bounded pixel, ablation, diversity, metamorphic, and negative checks."""
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train', 'test'):
            records.extend(_analyze(pair['input']) for pair in p[split])
        again = generate_puzzle(seed)
        _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], again[s])
                     for k in ('input', 'output')), "nondeterministic seed")
    _require(len({tuple(_signature(q['input']) for s in ('train', 'test') for q in p[s])
                  for p in puzzles}) == 42, "duplicate puzzles across seeds")
    _require({i['n'] for i in records} == {3, 4}, "board sizes collapsed")
    _require(len({i['grid'].shape for i in records}) >= 12, "geometry diversity collapsed")
    _require(len({(i['bg'], i['rail']) for i in records}) >= 20, "palette diversity collapsed")
    _require(len({b['key'][1:] for i in records for b in i['board']}) >= 35, "glyph diversity collapsed")
    _require({_line_kind(i['positions'], i['n']) for i in records} == {None, 'row', 'col'},
             "outcome diversity collapsed")
    _require({len(i['active']) for i in records} >= {1, 2, 3, 4}, "query counts collapsed")
    _require({len(i['objects']) for i in records} >= {3, 4, 5}, "carrier counts collapsed")
    _require(any(o['shift'] == 0 for i in records for o in i['objects']) and
             any(o['shift'] >= 3 for i in records for o in i['objects']), "travel diversity collapsed")
    _require(any(len(_components(b['mask'])) > 1 for i in records for b in i['board']),
             "disconnected distractor glyphs absent")
    base = puzzles[0]
    # Full-scene transformations rotate both positive glyphs and negative keys.
    for pair in base['train'] + base['test']:
        for turn in range(4):
            for flip in (False, True):
                inp = np.fliplr(pair['input']) if flip else pair['input']
                out = np.fliplr(pair['output']) if flip else pair['output']
                _require(np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn)),
                         "scene symmetry changed rule")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(2000 + nt * 10 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    for defect in ('label', 'duplicate', 'ink'):
        damaged = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in base[s]] for s in base}
        if defect == 'duplicate':
            damaged['test'][0] = {k: v.copy() for k, v in damaged['train'][0].items()}
        elif defect == 'label':
            damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
        else:
            info = _analyze(damaged['test'][0]['input'])
            out = np.rot90(damaged['test'][0]['output'], info['turn'])
            b = info['board'][0]
            block = out[b['rs'], b['cs']]
            r, c = np.argwhere(info['grid'][b['rs'], b['cs']] != info['bg'])[0]
            block[r, c] = info['bg']
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            continue
        raise AssertionError('validator accepted corruption: ' + defect)


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal counts, symmetries, and ablations)')
