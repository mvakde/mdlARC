"""Keyed strip blueprints: joint docking and partial-print inference.

Only standard Python and NumPy are used. No parent code or stored puzzle is
consulted by generation or solving. The small strips remain where they were;
their uniquely selected, abstracted assembly is printed as a square mosaic.
"""
from collections import Counter
from math import gcd

import numpy as np


N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    left = set(map(tuple, np.argwhere(mask).tolist()))
    answer = []
    while left:
        q = min(left)
        left.remove(q)
        found, todo = {q}, [q]
        while todo:
            r, c = todo.pop()
            for dr, dc in N4:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    found.add(p)
                    todo.append(p)
        answer.append(found)
    return answer


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells) + 1, max(c for r, c in cells) + 1)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _signature(pins):
    return tuple(c - pins[0] for c in pins) if pins else ()


def _parse_horizontal(a, bg, components, squares):
    """Try a horizontal interpretation; derive pin hue from geometry, not hue."""
    square_cells = set().union(*(s[4] for s in squares))
    pieces = [c for c in components if not c <= square_cells]
    if not 4 <= len(pieces) <= 6:
        return None
    colors = set.intersection(*({int(a[q]) for q in p} for p in pieces))
    parses = []
    for marker in sorted(colors):
        strips = []
        for cells in pieces:
            body = sorted(q for q in cells if a[q] != marker)
            if not body or len({r for r, c in body}) != 1:
                break
            row, left = body[0]
            width = len(body)
            if not 3 <= width <= 7 or body != [(row, left + k) for k in range(width)]:
                break
            pins = cells - set(body)
            if any(r not in (row - 1, row + 1) or not left <= c < left + width
                   for r, c in pins):
                break
            top = tuple(sorted(c - left for r, c in pins if r == row - 1))
            bottom = tuple(sorted(c - left for r, c in pins if r == row + 1))
            if not (top or bottom) or any(p and not 2 <= len(p) <= 3 for p in (top, bottom)):
                break
            strips.append(dict(body=tuple(int(a[q]) for q in body), top=top,
                               bottom=bottom, cells=cells, position=(row, left)))
        else:
            if (sum(not p['top'] for p in strips) == 1
                    and sum(not p['bottom'] for p in strips) == 1
                    and marker not in {s[3] for s in squares}):
                parses.append((marker, strips))
    return parses[0] if len(parses) == 1 else None


def _parse(input_grid):
    original = _grid(input_grid)
    counts = np.bincount(original.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= np.partition(counts, -2)[-2]:
        raise ValueError("background must be uniquely dominant")
    for transposed in (False, True):
        a = original.T.copy() if transposed else original
        cc = _components(a != bg)
        squares = []
        for cells in cc:
            t, l, b, r = _box(cells)
            if (b - t == r - l >= 2 and len(cells) == (b - t) ** 2
                    and len({int(a[q]) for q in cells}) == 1):
                squares.append((t, l, b - t, int(a[t, l]), cells))
        if not 3 <= len(squares) <= 6:
            continue
        parsed = _parse_horizontal(a, bg, cc, squares)
        if parsed is None:
            continue
        marker, strips = parsed
        sides = {s[2] for s in squares}
        if len(sides) != 1:
            continue
        side = sides.pop()
        ys, xs = sorted({s[0] for s in squares}), sorted({s[1] for s in squares})
        if len(ys) < 2 or len(xs) < 2:
            continue
        dy, dx = 0, 0
        for x, y in zip(ys, ys[1:]):
            dy = gcd(dy, y - x)
        for x, y in zip(xs, xs[1:]):
            dx = gcd(dx, y - x)
        if dy != dx or not side < dx <= side + 3:
            continue
        pitch = dx
        clue = {(int((y - ys[0]) // pitch), int((x - xs[0]) // pitch)): color
                for y, x, size, color, cells in squares}
        if len(clue) != len(squares):
            continue
        return dict(grid=a, bg=bg, marker=marker, strips=strips,
                    side=side, pitch=pitch, clue=clue,
                    seed_origin=(ys[0], xs[0]), transposed=transposed)
    raise ValueError("not a keyed-strip blueprint input")


def _plans(strips, bg, area_only=False):
    """All complete docking chains, modulo translation. No print labels used."""
    root = next(i for i, p in enumerate(strips) if not p['top'])
    result = []

    def walk(order, offsets):
        i = order[-1]
        p = strips[i]
        if not p['bottom']:
            if len(order) != len(strips):
                return
            left = min(offsets)
            offsets = tuple(c - left for c in offsets)
            width = max(c + len(strips[j]['body']) for j, c in zip(order, offsets))
            plan = np.full((len(order), width), bg, dtype=np.int8)
            for r, (j, c) in enumerate(zip(order, offsets)):
                plan[r, c:c + len(strips[j]['body'])] = strips[j]['body']
            result.append(dict(plan=plan, order=tuple(order), offsets=offsets))
            return
        for j, q in enumerate(strips):
            if j in order or not q['top']:
                continue
            compatible = (len(p['bottom']) == len(q['top']) if area_only else
                          _signature(p['bottom']) == _signature(q['top']))
            if compatible:
                shift = offsets[-1] + p['bottom'][0] - q['top'][0]
                walk(order + [j], offsets + [shift])

    walk([root], [0])
    return result


def _matches(plan, clue):
    mh = max(r for r, c in clue) + 1
    mw = max(c for r, c in clue) + 1
    return [(r, c) for r in range(plan.shape[0] - mh + 1)
            for c in range(plan.shape[1] - mw + 1)
            if all(int(plan[r + y, c + x]) == v for (y, x), v in clue.items())]


def _abstract_answers(plans, clue, bg):
    """Distinct colored plans relative to the motif; independent of layout."""
    answers = set()
    for p in plans:
        plan = p['plan']
        for r, c in _matches(plan, clue):
            answers.add(tuple((y - r, x - c, int(plan[y, x]))
                              for y, x in zip(*np.nonzero(plan != bg))))
    return answers


def _answers(meta, area_only=False):
    a, bg, side, pitch = (meta[k] for k in ('grid', 'bg', 'side', 'pitch'))
    small = set().union(*(p['cells'] for p in meta['strips']))
    answers = {}
    for record in _plans(meta['strips'], bg, area_only):
        plan = record['plan']
        h, w = plan.shape
        for match in _matches(plan, meta['clue']):
            y = meta['seed_origin'][0] - match[0] * pitch
            x = meta['seed_origin'][1] - match[1] * pitch
            b, r = y + (h - 1) * pitch + side, x + (w - 1) * pitch + side
            if not (0 <= y < b <= a.shape[0] and 0 <= x < r <= a.shape[1]):
                continue
            if any(y <= t < b and x <= l < r for t, l in small):
                continue
            out = a.copy()
            for (t, l), color in np.ndenumerate(plan):
                if color != bg:
                    yy, xx = y + t * pitch, x + l * pitch
                    region = a[yy:yy + side, xx:xx + side]
                    # The motif is the only supplied material in this rectangle.
                    if not np.all((region == bg) | (region == color)):
                        break
                    out[yy:yy + side, xx:xx + side] = color
            else:
                answers.setdefault(_key(out), (out, dict(record, match=match, base=(y, x))))
    return list(answers.values())


def _recover(input_grid):
    meta = _parse(input_grid)
    answers = _answers(meta)
    if len(answers) != 1:
        raise ValueError(f"joint docking/print constraints admit {len(answers)} answers")
    out, selected = answers[0]
    meta.update(selected)
    return (out.T.copy() if meta['transposed'] else out), meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input only, including its visible scale and docking keys."""
    return _recover(input_grid)[0]


def _check_assembly(meta):
    """Redraw the latent miniature with rigid markers; check two-owner joints."""
    payload, owners = {}, Counter()
    for row, (i, col) in enumerate(zip(meta['order'], meta['offsets'])):
        p = meta['strips'][i]
        for c, color in enumerate(p['body']):
            q = 2 * row, col + c
            assert q not in payload
            payload[q] = color
        for dr, pins in ((-1, p['top']), (1, p['bottom'])):
            for c in pins:
                owners[2 * row + dr, col + c] += 1
    assert owners and set(owners.values()) == {2}, "incomplete geometric docking"
    assert not set(owners) & set(payload)
    assert sum(len(p['top']) + len(p['bottom']) for p in meta['strips']) == 2 * len(owners)
    assert len(_components_from_cells(set(owners) | set(payload))) == 1


def _components_from_cells(cells):
    t, l, b, r = _box(cells)
    mask = np.zeros((b - t, r - l), dtype=bool)
    for y, x in cells:
        mask[y - t, x - l] = True
    return _components(mask)


def _features(meta):
    plans = _plans(meta['strips'], meta['bg'])
    keys = {_signature(p[s]) for p in meta['strips'] for s in ('top', 'bottom') if p[s]}
    return dict(count=len(meta['strips']), side=meta['side'], pitch=meta['pitch'],
                shape=meta['grid'].shape, plan_shape=meta['plan'].shape,
                holes=int(np.count_nonzero(meta['plan'] == meta['bg'])),
                keys=tuple(sorted(keys)), key_sizes=tuple(sorted({len(k) for k in keys})),
                offsets=meta['offsets'], match=meta['match'], base=meta['base'],
                marker=meta['marker'], background=meta['bg'],
                vertical=meta['transposed'], clue_count=len(meta['clue']),
                clue_height=max(r for r, c in meta['clue']) + 1,
                docking_plans=len({_key(p['plan']) for p in plans}),
                area_answers=len(_answers(meta, area_only=True)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, meta = _recover(inp)
            assert np.array_equal(expected, out), "incorrect print or altered strip"
            assert not np.array_equal(inp, out)
            assert np.all(out[inp != meta['bg']] == inp[inp != meta['bg']])
            assert np.count_nonzero(inp == meta['marker']) == np.count_nonzero(out == meta['marker'])
            _check_assembly(meta)
            f = _features(meta)
            assert f['docking_plans'] >= 2, "print evidence must select among docking assemblies"
            assert len(set(meta['offsets'])) > 1, "keys must determine real row staggering"
            # Scatter cannot encode the correct row order by top-to-bottom sorting.
            source_order = tuple(sorted(range(f['count']), key=lambda i: meta['strips'][i]['position']))
            assert source_order != meta['order']
            features[split].append(f)
    tr = features['train']
    assert len({f['count'] for f in tr}) >= 2
    assert len({f['side'] for f in tr}) >= 2
    assert {f['vertical'] for f in tr} == {False, True}
    assert any(f['area_answers'] >= 2 for f in tr), "geometry ablation must destroy uniqueness"
    assert any(f['match'][0] > 0 for f in tr) and any(f['match'][0] == 0 for f in tr)
    assert all(f['clue_height'] == 3 for f in features['test'])
    return None


def _counts(rng, num_train, num_test):
    for name, v, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if v is not None and (isinstance(v, (bool, np.bool_))
                              or not isinstance(v, (int, np.integer)) or not low <= v <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _sample_strips(rng, n, palette, geometry):
    # A repeated key gives exchangeable internal strips. A second equal-area
    # key forces a transition that cannot be inferred from pin count.
    cardinality = int(rng.choice([2, 3], p=[.78, .22]))
    if cardinality == 2:
        spans = rng.choice(np.arange(1, 5), 2, replace=False)
        ka, kb = (0, int(spans[0])), (0, int(spans[1]))
    else:
        bank = [(0, 1, 3), (0, 2, 3), (0, 1, 4), (0, 2, 4), (0, 3, 4)]
        ia, ib = rng.choice(len(bank), 2, replace=False)
        ka, kb = bank[int(ia)], bank[int(ib)]
    edge_keys = [ka] * (n - 1)
    if geometry:
        edge_keys[-1] = kb
    strips = []
    for i in range(n):
        top = edge_keys[i - 1] if i else ()
        bottom = edge_keys[i] if i < n - 1 else ()
        minimum = max(3, max(top + bottom, default=0) + 1)
        length = int(rng.integers(minimum, 8))
        ports = []
        for key in (top, bottom):
            offset = int(rng.integers(length - key[-1])) if key else 0
            ports.append(tuple(offset + c for c in key))
        strips.append(dict(body=tuple(int(v) for v in rng.choice(palette, length)),
                           top=ports[0], bottom=ports[1]))
    return strips


def _motifs(rng, plan, bg, span, location):
    rows = [location] if location is not None else list(range(plan.shape[0] - span + 1))
    choices = []
    for r in rows:
        for c in range(plan.shape[1] - 1):
            if np.any(plan[r:r + span, c:c + 2] == bg):
                continue
            # A full rectangle or a rectangle with one absent clue square.
            # An absent clue square says nothing about the eventual plan cell.
            for missing in [None] + list(range(span * 2)):
                clue = {(y, x): int(plan[r + y, c + x])
                        for y in range(span) for x in range(2)
                        if missing != y * 2 + x}
                choices.append((clue, (r, c)))
    rng.shuffle(choices)
    return choices


def _scatter(rng, shape, strips, bg, marker, rect):
    h, w = shape
    y, x, b, r = rect
    for restart in range(12):
        blocked = np.zeros(shape, dtype=bool)
        blocked[max(0, y - 1):min(h, b + 1), max(0, x - 1):min(w, r + 1)] = True
        a = np.full(shape, bg, dtype=np.int8)
        order = list(rng.permutation(len(strips)))
        for i in order:
            p = strips[int(i)]
            center = int(bool(p['top']))
            height = 1 + int(bool(p['top'])) + int(bool(p['bottom']))
            width = len(p['body'])
            candidates = [(t, l) for t in range(h - height + 1)
                          for l in range(w - width + 1)]
            rng.shuffle(candidates)
            for t, l in candidates:
                if blocked[t:t + height, l:l + width].any():
                    continue
                a[t + center, l:l + width] = p['body']
                for dr, pins in ((-1, p['top']), (1, p['bottom'])):
                    for c in pins:
                        a[t + center + dr, l + c] = marker
                blocked[max(0, t - 1):min(h, t + height + 1),
                        max(0, l - 1):min(w, l + width + 1)] = True
                break
            else:
                break
        else:
            return a
    return None


def _make_pair(rng, kind, side=None, vertical=None):
    if kind == 'selection':
        n = 4
    elif kind == 'geometry':
        n = 5
    elif kind == 'test':
        n = int(rng.integers(5, 7))
    else:
        n = int(rng.integers(4, 6))
    geometry = n >= 5
    span = 3 if kind == 'test' else 2
    for attempt in range(300):
        colors = [int(v) for v in rng.permutation(10)]
        bg, marker = colors[:2]
        palette = colors[2:2 + int(rng.integers(3, 7))]
        strips = _sample_strips(rng, n, palette, geometry)
        plans = _plans(strips, bg)
        if len({_key(p['plan']) for p in plans}) < 2:
            continue
        possible = [p for p in plans if p['plan'].shape[1] <= 7
                    and len(set(p['offsets'])) > 1]
        rng.shuffle(possible)
        for record in possible[:8]:
            plan = record['plan']
            location = n - 2 if kind == 'selection' else 0 if kind == 'geometry' else None
            motifs = _motifs(rng, plan, bg, span, location)
            chosen = None
            area_plans = _plans(strips, bg, area_only=True) if kind == 'geometry' else []
            for clue, match in motifs[:100]:
                if len(_abstract_answers(plans, clue, bg)) != 1:
                    continue
                if kind == 'geometry' and len(_abstract_answers(area_plans, clue, bg)) < 2:
                    continue
                chosen = clue, match
                break
            if chosen is None:
                continue
            clue, match = chosen
            ss = side if side is not None else int(rng.choice([2, 3]))
            pitch = ss + int(rng.integers(1, 3))
            bh, bw = (n - 1) * pitch + ss, (plan.shape[1] - 1) * pitch + ss
            if max(bh, bw) > 25:
                continue
            shape = tuple(int(rng.integers(max(20, size + 3), 31)) for size in (bh, bw))
            y, x = int(rng.integers(shape[0] - bh + 1)), int(rng.integers(shape[1] - bw + 1))
            inp = _scatter(rng, shape, strips, bg, marker, (y, x, y + bh, x + bw))
            if inp is None:
                continue
            for (r, c), color in clue.items():
                yy, xx = y + (r + match[0]) * pitch, x + (c + match[1]) * pitch
                inp[yy:yy + ss, xx:xx + ss] = color
            out = inp.copy()
            for (r, c), color in np.ndenumerate(plan):
                if color != bg:
                    yy, xx = y + r * pitch, x + c * pitch
                    out[yy:yy + ss, xx:xx + ss] = color
            if rng.random() < .5:
                inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
            if kind not in ('selection', 'geometry') and rng.random() < .5:
                inp, out = inp[::-1].copy(), out[::-1].copy()
            use_vertical = bool(vertical) if vertical is not None else bool(rng.integers(2))
            if use_vertical:
                inp, out = inp.T.copy(), out.T.copy()
            try:
                predicted, meta = _recover(inp)
                f = _features(meta)
            except ValueError:
                continue
            if not np.array_equal(predicted, out):
                continue
            if len(set(meta['offsets'])) < 2:
                continue
            source_order = tuple(sorted(range(n), key=lambda i: meta['strips'][i]['position']))
            if source_order == meta['order'] or (kind == 'geometry' and f['area_answers'] < 2):
                continue
            return {'input': inp, 'output': out}
    raise RuntimeError("bounded keyed-blueprint sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    side_order = list(rng.permutation([2, 3]))
    vertical = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            kind = ('selection' if i == 0 else 'geometry' if i == 1 else 'ordinary') if split == 'train' else 'test'
            for retry in range(20):
                pair = _make_pair(rng, kind, int(side_order[i]) if split == 'train' and i < 2 else None,
                                  (vertical if i == 0 else not vertical) if split == 'train' and i < 2 else None)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded semantic, counterfactual, diversity and API checks on 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, features = set(), []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s]
                             for k in ('input', 'output')))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                out, meta = _recover(p['input'].copy())
                assert np.array_equal(out, p['output'])
                features.append(dict(_features(meta), split=split))
    assert len(signatures) == 42
    for name, minimum in (('shape', 25), ('plan_shape', 8), ('offsets', 30),
                          ('keys', 12), ('base', 40), ('match', 6), ('background', 9), ('marker', 9)):
        assert len({f[name] for f in features}) >= minimum, name
    assert {f['count'] for f in features} == {4, 5, 6}
    assert {f['side'] for f in features} == {2, 3}
    assert {f['pitch'] - f['side'] for f in features} == {1, 2}
    assert {s for f in features for s in f['key_sizes']} == {2, 3}
    assert any(f['holes'] >= 4 for f in features)
    assert any(f['clue_count'] == 3 for f in features)
    assert any(f['clue_count'] == 6 for f in features)
    assert any(f['docking_plans'] >= 6 for f in features)
    for f in features:
        if f['split'] == 'test':
            assert f['clue_height'] == 3 and f['count'] >= 5
    assert tuple(_key(p[k]) for s in ('train', 'test') for p in puzzles[0][s] for k in ('input', 'output')) == tuple(_key(p[k]) for s in ('train', 'test') for p in generate_puzzle(0)[s] for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(10000 + 10 * a + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        generate_puzzle(8181, a, b)
    for a, b in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Preserve the squares but erase all top/bottom keys: the print alone has
    # no spatially ordered blueprint and must not yield a guessed answer.
    inp = puzzles[0]['test'][0]['input'].copy()
    meta = _parse(inp)
    canon = meta['grid'].copy()
    canon[canon == meta['marker']] = meta['bg']
    try:
        solve(canon.T if meta['transposed'] else canon)
    except ValueError:
        pass
    else:
        raise AssertionError("missing docking evidence accepted")
    # The print is also necessary: without it, even intact keys leave multiple
    # colored plans and neither scale nor absolute print position is visible.
    meta = _parse(puzzles[0]['test'][0]['input'])
    canon = meta['grid'].copy()
    small = set().union(*(p['cells'] for p in meta['strips']))
    for q in np.ndindex(canon.shape):
        if q not in small:
            canon[q] = meta['bg']
    try:
        solve(canon.T if meta['transposed'] else canon)
    except ValueError:
        pass
    else:
        raise AssertionError("missing print evidence accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, semantic ablations and legal counts)')
