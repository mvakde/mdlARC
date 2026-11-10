"""Shadow Ballots: local panel votes, directional beams, and opaque receivers.

All interpretation and voting in solve comes from input pixels. Construction
uses the same public rule for labels; the checks below are generator-owned,
not an independent semantic certification.
"""
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np


_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _ring(r, c, h, w):
    """The theoretical one-cell ring, with its outward compass headings."""
    result = []
    for x in range(r - 1, r + h + 1):
        for y in range(c - 1, c + w + 1):
            if r <= x < r + h and c <= y < c + w:
                continue
            dr = -1 if x < r else 1 if x >= r + h else 0
            dc = -1 if y < c else 1 if y >= c + w else 0
            result.append(((x, y), (dr, dc)))
    return result


def _interpret(a, bg, frame, receiver, signals):
    panels = _components(np.isin(a, signals))
    _require(2 <= len(panels) <= 6, "need 2..6 panels")
    covered, footprints, records = set(), set(), []
    for cells in panels:
        rr, cc = zip(*cells)
        r, c = min(rr), min(cc)
        h, w = max(rr) - r + 1, max(cc) - c + 1
        _require(h >= 2 and w >= 2 and len(cells) == h * w, "panel must be a solid rectangle")
        counts = [sum(int(a[p]) == s for p in cells) for s in signals]
        _require(min(counts) > 0 and counts[0] != counts[1], "panel needs a mixed strict majority")
        border = [(p, d) for p, d in _ring(r, c, h, w) if _inside(p, a.shape)]
        _require(all(int(a[p]) in (bg, frame) for p, _ in border), "invalid panel ring")
        gaps = [(p, d) for p, d in border if a[p] == bg]
        solid = {p for p, _ in border if a[p] == frame}
        _require(gaps and len(solid) >= 3, "ring needs gaps and at least three frame cells")
        footprint = cells | {p for p, _ in border}
        _require(not footprints & footprint, "panel footprints overlap")
        footprints |= footprint
        covered |= cells | solid
        records.append(dict(box=(r, c, h, w), cells=cells, gaps=gaps,
                            color=signals[int(counts[1] > counts[0])], counts=counts))
    targets = _components(a == receiver)
    _require(1 <= len(targets) <= 6 and all(len(t) >= 3 for t in targets), "invalid receivers")
    target_ids = {p: i for i, cells in enumerate(targets) for p in cells}
    actual = set(zip(*np.nonzero((a != bg) & (a != receiver))))
    _require(covered == actual, "unaccounted frame or panel pixels")
    return dict(background=bg, frame=frame, receiver=receiver, signals=signals,
                panels=records, targets=targets, target_ids=target_ids)


def _parse(a):
    values, counts = np.unique(a, return_counts=True)
    _require(len(values) == 5 and np.count_nonzero(counts == counts.max()) == 1,
             "input needs five roles and a unique most frequent background")
    bg = int(values[np.argmax(counts)])
    others = [int(v) for v in values if v != bg]
    interpretations = []
    for signals in combinations(others, 2):
        remainder = [v for v in others if v not in signals]
        for frame, receiver in (remainder, remainder[::-1]):
            try:
                interpretations.append(_interpret(a, bg, frame, receiver, signals))
            except AssertionError:
                pass
    _require(len(interpretations) == 1, "input must have one geometric role interpretation")
    return interpretations[0]


def _trace(a, start, direction, bg):
    """Trace through input background only; output trails are transparent."""
    cells, p = [], start
    while _inside(p, a.shape):
        if a[p] != bg:
            return cells, p
        cells.append(p)
        p = (p[0] + direction[0], p[1] + direction[1])
    return cells, None


def _render(a, info):
    trails = defaultdict(Counter)
    votes = [Counter() for _ in info['targets']]
    beams = []
    for pid, panel in enumerate(info['panels']):
        for start, direction in panel['gaps']:
            cells, hit = _trace(a, start, direction, info['background'])
            color = panel['color']
            tid = info['target_ids'].get(hit)
            if tid is not None:
                votes[tid][color] += 1
            for p in cells:
                trails[p][color] += 1
            beams.append(dict(panel=pid, start=start, direction=direction,
                              color=color, cells=cells, target=tid, hit=hit))
    out = a.copy()
    s0, s1 = info['signals']
    for p, ballot in trails.items():
        n0, n1 = ballot[s0], ballot[s1]
        out[p] = info['frame'] if n0 == n1 else s0 if n0 > n1 else s1
    for cells, ballot in zip(info['targets'], votes):
        n0, n1 = ballot[s0], ballot[s1]
        if n0 != n1:
            color = s0 if n0 > n1 else s1
            for p in cells:
                out[p] = color
    return out, dict(info, votes=votes, beams=beams, trails=trails)


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _render(a, _parse(a))[0]


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _neighbors(cells, shape, diagonal=False):
    dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc] if diagonal else _ORTH
    return {(r + dr, c + dc) for r, c in cells for dr, dc in dirs
            if _inside((r + dr, c + dc), shape)}


def _pattern(rng, h, w, color, other, strength=None):
    area = h * w
    if strength == 'close':
        minority = (area - 1) // 2
    elif strength == 'strong':
        minority = int(rng.integers(1, max(2, area // 5)))
    else:
        minority = int(rng.integers(1, (area - 1) // 2 + 1))
    flat = np.full(area, color, dtype=np.int8)
    style = int(rng.integers(3))
    if style == 0:
        order = rng.permutation(area)
    elif style == 1:
        rr, cc = np.indices((h, w))
        x, y = int(rng.integers(h)), int(rng.integers(w))
        order = np.argsort((abs(rr - x) + abs(cc - y) + rng.random((h, w)) / 3).ravel())
    else:
        order = np.arange(area).reshape(h, w)
        if rng.integers(2):
            order = order.T
        if rng.integers(2):
            order = order[::-1]
        order = order.ravel()
    flat[order[:minority]] = other
    return flat.reshape(h, w)


def _silhouette(rng, a, color):
    """A connected, often concave receiver with independently sampled geometry."""
    h, w = a.shape
    th, tw = int(rng.integers(4, 8)), int(rng.integers(4, 8))
    r = int(rng.integers(5, h - th - 3))
    c = int(rng.integers(5, w - tw - 3))
    style = int(rng.integers(4))
    if style == 0:
        cells = {(x, y) for x in range(r, r + th) for y in range(c, c + tw)}
        # Remove some corner cells while retaining a substantial central block.
        for x, y in [(r, c), (r, c + tw - 1), (r + th - 1, c), (r + th - 1, c + tw - 1)]:
            if rng.integers(2):
                cells.remove((x, y))
    elif style == 1:
        cells = {(x, y) for x in range(r, r + th) for y in range(c, c + tw)
                 if x in (r, r + th - 1) or y in (c, c + tw - 1)}
        # Hollow shapes collect at the first encountered rim, not at the far rim.
    elif style == 2:
        cells = {(x, y) for x in range(r, r + th) for y in range(c, c + tw)
                 if x in (r + th // 2, r + th // 2 - 1) or y in (c + tw // 2, c + tw // 2 - 1)}
    else:
        cells = {(x, y) for x in range(r, r + th) for y in range(c, c + tw)
                 if x < r + 2 or y < c + 2 or x >= r + th - 2}
    for p in cells:
        a[p] = color
    return cells


def _place_panel(rng, a, target, roles, color, wanted, witness=False):
    bg, frame, receiver, s0, s1 = roles
    other = s1 if color == s0 else s0
    target_list = sorted(target)
    occupied = set(zip(*np.nonzero(a != bg)))
    blocked = occupied | _neighbors(occupied, a.shape, diagonal=True)
    for _ in range(700):
        if witness and wanted == 3:
            ph, pw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            if ph * pw > 12:
                continue
        elif witness:
            ph, pw = int(rng.integers(3, 6)), int(rng.integers(4, 7))
        else:
            ph, pw = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        if ph * pw < wanted * 2:
            continue
        offsets = _ring(0, 0, ph, pw)
        offset, direction = offsets[int(rng.integers(len(offsets)))]
        q = target_list[int(rng.integers(len(target_list)))]
        distance = int(rng.integers(3, 11))
        port = (q[0] - distance * direction[0], q[1] - distance * direction[1])
        r, c = port[0] - offset[0], port[1] - offset[1]
        if not (0 <= r and 0 <= c and r + ph <= a.shape[0] and c + pw <= a.shape[1]):
            continue
        border = [(p, d) for p, d in _ring(r, c, ph, pw) if _inside(p, a.shape)]
        cells = {(x, y) for x in range(r, r + ph) for y in range(c, c + pw)}
        footprint = cells | {p for p, _ in border}
        if footprint & blocked:
            continue
        usable = []
        for p, d in border:
            trail, hit = _trace(a, p, d, bg)
            if hit in target and len(trail) >= 2:
                usable.append((p, d))
        if len(usable) < wanted or len(border) - wanted < 3:
            continue
        chosen = [usable[int(i)] for i in rng.choice(len(usable), wanted, replace=False)]
        gaps = {p for p, _ in chosen}
        a[r:r + ph, c:c + pw] = _pattern(rng, ph, pw, color, other,
            'close' if witness and wanted == 3 else 'strong' if witness else None)
        for p, _ in border:
            a[p] = bg if p in gaps else frame
        return (r, c, ph, pw)
    return None


def _add_extra_gaps(rng, a, protected_votes=None, seek_crossing=False):
    """Extra openings are admitted by their visible effect, not by a layout bank."""
    info = _parse(a)
    _, before = _render(a, info)
    opportunities = [(p, d) for panel in info['panels']
                     for p, d in _ring(*panel['box'])
                     if _inside(p, a.shape) and a[p] == info['frame']]
    rng.shuffle(opportunities)
    candidates = [(x,) for x in opportunities]
    if seek_crossing:
        candidates = [tuple(opportunities[int(i)] for i in rng.choice(len(opportunities), 2, replace=False))
                      for _ in range(180)] + candidates
    limit = int(rng.integers(1, 5))
    accepted = 0
    for changes in candidates:
        b = a.copy()
        for p, _ in changes:
            b[p] = info['background']
        try:
            _, after = _render(b, _parse(b))
        except AssertionError:
            continue
        if protected_votes is not None and after['votes'] != before['votes']:
            continue
        if seek_crossing and not any(len(v) == 2 for v in after['trails'].values()):
            continue
        a[:] = b
        before = after
        accepted += 1
        if seek_crossing or accepted >= limit:
            return True
    return not seek_crossing


def _add_shadow(rng, a, target, color=None, exact_one=False):
    _, info = _render(a, _parse(a))
    tid = next(i for i, t in enumerate(info['targets']) if t == target)
    eligible = [b for b in info['beams'] if b['target'] == tid and
                (color is None or b['color'] == color) and len(b['cells']) >= 6]
    rng.shuffle(eligible)
    for beam in eligible:
        options = list(range(2, len(beam['cells']) - 2))
        rng.shuffle(options)
        for k in options:
            p = beam['cells'][k]
            dr, dc = beam['direction']
            # A narrow connected piece can block just one of parallel ballots.
            cells = {p, (p[0] + dr, p[1] + dc), (p[0] - dr, p[1] - dc)}
            if dr and dc:
                cells |= {(p[0] + dr, p[1]), (p[0] - dr, p[1])}
            if any(not _inside(q, a.shape) for q in cells):
                continue
            if any(a[q] != info['background'] for q in cells | _neighbors(cells, a.shape, True)):
                continue
            affected = [b for b in info['beams'] if set(b['cells']) & cells]
            if exact_one and (len(affected) != 1 or affected[0]['color'] != color):
                continue
            b = a.copy()
            for q in cells:
                b[q] = info['receiver']
            try:
                _, after = _render(b, _parse(b))
            except AssertionError:
                continue
            if after['votes'][next(i for i, t in enumerate(after['targets']) if t == target)] == info['votes'][tid]:
                continue
            a[:] = b
            return cells
    return None


def _add_unreached(rng, a):
    _, info = _render(a, _parse(a))
    bg, receiver = info['background'], info['receiver']
    for _ in range(240):
        r, c = int(rng.integers(1, a.shape[0] - 3)), int(rng.integers(1, a.shape[1] - 3))
        style = int(rng.integers(3))
        cells = {(r, c), (r + 1, c), (r + 2, c)} if style == 0 else (
            {(r, c), (r, c + 1), (r + 1, c)} if style == 1 else
            {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)})
        if any(a[q] != bg for q in cells | _neighbors(cells, a.shape, True)):
            continue
        if any(p in info['trails'] for p in cells):
            continue
        for p in cells:
            a[p] = receiver
        return cells
    return None


def _measure(a):
    _, info = _render(a, _parse(a))
    s0, s1 = info['signals']
    weighting = False
    raw_conflict = False
    for tid, vote in enumerate(info['votes']):
        if vote[s0] == vote[s1]:
            continue
        winner = s0 if vote[s0] > vote[s1] else s1
        loser = s1 if winner == s0 else s0
        contributing = {b['panel'] for b in info['beams'] if b['target'] == tid}
        labels = [info['panels'][i]['color'] for i in contributing]
        if labels.count(winner) < labels.count(loser):
            weighting = True
            raw_conflict |= int(np.sum(a == winner)) < int(np.sum(a == loser))
    mixed = sum(len(v) == 2 for v in info['trails'].values())
    ties = sum(v[s0] == v[s1] and bool(v) for v in info['votes'])
    silent = sum(not v for v in info['votes'])
    diagonal = any(all(d != 0 for d in b['direction']) for b in info['beams'])
    local_conflict = any(
        a[(beam['start'][0] - beam['direction'][0],
           beam['start'][1] - beam['direction'][1])] != beam['color']
        for beam in info['beams'] if beam['target'] is not None)
    shadow = False
    for tid, cells in enumerate(info['targets']):
        b = a.copy()
        for p in cells:
            b[p] = info['background']
        # Keep the original target index map to compare votes to surviving shapes.
        _, without = _render(b, info)
        if any(without['votes'][j] != info['votes'][j] for j in range(len(info['targets'])) if j != tid):
            shadow = True
    return dict(info, weighting=weighting, raw_conflict=raw_conflict, mixed=mixed,
                ties=ties, silent=silent, diagonal=diagonal, shadow=shadow,
                local_conflict=local_conflict)


def _make_pair(rng, roles, kind, bigger=False):
    bg, frame, receiver, s0, s1 = roles
    for _ in range(100):
        h, w = map(int, rng.integers(22 if bigger else 19, 31 if bigger else 29, 2))
        a = np.full((h, w), bg, dtype=np.int8)
        target = _silhouette(rng, a, receiver)
        winner = s0 if rng.integers(2) else s1
        other = s1 if winner == s0 else s0
        witness = kind in ('weight', 'shadow')
        labels = [winner, other, other] if witness else [s0, s1] + [int(rng.choice((s0, s1))) for _ in range(int(rng.integers(0, 4)))]
        weights = [3, 1, 1] if witness else list(map(int, rng.integers(1, 5, len(labels))))
        success = True
        for color, wanted in zip(labels, weights):
            if _place_panel(rng, a, target, roles, color, wanted, witness) is None:
                success = False
                break
        if not success:
            continue
        try:
            _parse(a)
            if kind == 'shadow':
                if _add_shadow(rng, a, target, winner, exact_one=True) is None:
                    continue
            elif kind == 'general' and rng.random() < 0.8:
                _add_shadow(rng, a, target)
            # Preserve the teaching tallies while seeking transparent crossings.
            crossed = _add_extra_gaps(rng, a, True if witness else None, seek_crossing=kind == 'shadow')
            if kind == 'shadow' and not crossed:
                continue
            if _add_unreached(rng, a) is None:
                continue
            if bigger:
                for _ in range(int(rng.integers(1, 3))):
                    if _add_shadow(rng, a, target) is None:
                        _add_unreached(rng, a)
            out, info = _render(a, _parse(a))
            m = _measure(a)
            if kind == 'weight' and not (m['weighting'] and m['raw_conflict'] and m['local_conflict']):
                continue
            if kind == 'shadow' and not (m['shadow'] and m['ties'] and m['mixed'] and m['diagonal']):
                continue
            if not any(len(v) == 2 and v[s0] != v[s1] for v in info['votes']) and kind == 'general':
                # General examples still include a contested strict vote.
                continue
            if bigger and not (m['shadow'] and sum(bool(v) for v in m['votes']) >= 2 and len(m['targets']) >= 3):
                continue
            rotation = int(rng.integers(4))
            a, out = np.rot90(a, rotation), np.rot90(out, rotation)
            if rng.integers(2):
                a, out = np.fliplr(a), np.fliplr(out)
            return {'input': np.ascontiguousarray(a), 'output': np.ascontiguousarray(out)}
        except AssertionError:
            continue
    raise RuntimeError('bounded scene construction exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    varying = bool(rng.integers(2))
    train_kinds = ['weight', 'shadow'] + ['general'] * (nt - 2)
    rng.shuffle(train_kinds)
    result, seen = {'train': [], 'test': []}, set()
    for split, kinds in (('train', train_kinds), ('test', ['general'] * ne)):
        for kind in kinds:
            roles = tuple(map(int, rng.choice(10, 5, replace=False))) if varying else palette
            for _ in range(12):
                pair = _make_pair(rng, roles, kind, bigger=split == 'test')
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('could not make distinct inputs')
            seen.add(_key(pair['input']))
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'invalid counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and _key(a) not in seen, 'shape change or duplicate input')
            seen.add(_key(a))
            expected, info = _render(a, _parse(a))
            _require(np.array_equal(expected, b), 'wrong solution')
            _require(np.any(a != b), 'unchanged example')
            preserve = (a != info['background']) & (a != info['receiver'])
            _require(np.array_equal(a[preserve], b[preserve]), 'panels or frames changed')
            for cells in info['targets']:
                _require(len({int(b[p]) for p in cells}) == 1, 'receiver must recolor as a whole object')
            # Each beam has one ballot, even if it touches a long rim or crosses
            # several cells of the same receiver's bounding rectangle.
            _require(sum(sum(v.values()) for v in info['votes']) == sum(beam['target'] is not None for beam in info['beams']), 'ballots not conserved')
            _require(all(a[p] == info['background'] for beam in info['beams'] for p in beam['cells']), 'beam passed through input foreground')
            if split == 'train':
                training.append(_measure(a))
    _require(any(m['weighting'] and m['raw_conflict'] and m['local_conflict'] for m in training), 'missing local/port-versus-panel/raw-count witness')
    _require(any(m['shadow'] and m['ties'] and m['mixed'] and m['diagonal'] for m in training), 'missing shadow/tie/crossing/diagonal witness')
    _require(any(m['silent'] for m in training), 'unreached receiver behavior not taught')


def self_test():
    # Hand-calculated fixture: three horizontal ballots versus one vertical,
    # plus a diagonal crossing which continues and eventually hits a panel.
    fixture = np.zeros((13, 14), dtype=np.int8)
    fixture[4:9, 0:4] = 8
    fixture[5:8, 1:3] = [[3, 4], [3, 3], [3, 4]]
    fixture[5:8, 3] = 0
    fixture[0:4, 6:11] = 8
    fixture[1:3, 7:10] = [[4, 3, 4], [4, 4, 4]]
    fixture[3, 8] = fixture[3, 6] = 0
    fixture[5:8, 7:10] = 5
    expected = fixture.copy()
    expected[5:8, 3:7] = 3
    expected[3:5, 8] = 4
    for p in ((3, 6), (4, 5)):
        expected[p] = 4
    expected[5, 4] = expected[6, 3] = 8
    expected[5:8, 7:10] = 3
    _require(np.array_equal(solve(fixture), expected), 'hand-calculated beam/port/crossing fixture failed')
    # Intercept two horizontal votes; the downstream shape now ties 1:1.
    shadowed = fixture.copy()
    for p in ((6, 4), (7, 4), (6, 5)):
        shadowed[p] = 5
    shadow_expected = expected.copy()
    shadow_expected[6, 5:7] = 0
    shadow_expected[7, 5:7] = 0
    for p in ((6, 4), (7, 4), (6, 5)):
        shadow_expected[p] = 3
    shadow_expected[5:8, 7:10] = 5
    _require(np.array_equal(solve(shadowed), shadow_expected), 'first-hit shadow/tie fixture failed')
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(shadowed, k)), np.rot90(shadow_expected, k)), 'rotation changed rule')
    _require(np.array_equal(solve(np.fliplr(shadowed)), np.fliplr(shadow_expected)), 'reflection changed rule')

    shapes, structures, panel_counts, target_counts, headings, areas, gaps = (set() for _ in range(7))
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                m = _measure(a)
                shapes.add(a.shape)
                panel_counts.add(len(m['panels']))
                target_counts.add(len(m['targets']))
                headings.update(b['direction'] for b in m['beams'])
                areas.update(len(p['cells']) for p in m['panels'])
                gaps.update(len(p['gaps']) for p in m['panels'])
                roles = {m['background']: 0, m['frame']: 1, m['receiver']: 2, m['signals'][0]: 3, m['signals'][1]: 4}
                structures.add((a.shape, bytes(roles[int(v)] for v in a.flat)))
                _require(np.array_equal(solve(a.copy()), pair['output']), 'solve mismatch')
    _require(len(fingerprints) == 42 and len(structures) >= 180, 'structural diversity collapsed')
    _require(len(shapes) >= 35 and len(areas) >= 12 and len(gaps) >= 5, 'geometry or port diversity collapsed')
    _require({2, 3, 4, 5} <= panel_counts and {2, 3, 4} <= target_counts and len(headings) == 8, 'counts or headings collapsed')
    a, b = generate_puzzle(903, 2, 1), generate_puzzle(903, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), 'nondeterminism')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1000 + nt * 10 + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    m = _measure(corrupt['test'][0]['input'])
    p = next(iter(m['targets'][0]))
    corrupt['test'][0]['output'][p] = m['frame']
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted receiver accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                color_normalized_structures=len(structures), panel_counts=sorted(panel_counts),
                receiver_counts=sorted(target_counts), headings=sorted(headings),
                panel_areas=sorted(areas), port_counts=sorted(gaps),
                semantic_verification='generator-owned checks and hand-calculated fixtures only')
