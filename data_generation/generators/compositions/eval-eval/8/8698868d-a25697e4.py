"""Perforated double-ended keys: topology chooses a rigidly fitted plug.

No parent modules, source files, random state, or training examples are used by
solve. Generation paints constructed answers; validation reconstructs from input.
"""
from itertools import permutations
import json
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = DIR4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(cells, diagonal=False):
    remaining = set(cells)
    result = []
    while remaining:
        point = min(remaining)
        remaining.remove(point)
        stack, part = [point], {point}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR8 if diagonal else DIR4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(frozenset(part))
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _rectangle(box):
    a, b, c, d = box
    return frozenset((r, x) for r in range(a, b) for x in range(c, d))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _raw(point, op):
    r, c = point
    if op >= 4:
        c = -c
    for _ in range(op % 4):
        r, c = -c, r
    return r, c


def _transform(cells, op):
    return frozenset(_raw(p, op) for p in cells)


def _normalize(cells):
    a, _, c, _ = _box(cells)
    return _shift(cells, -a, -c)


def _shape(cells):
    return tuple(sorted(_normalize(cells)))


def _canonical(cells):
    return min(_shape(_transform(cells, op)) for op in range(8))


def _orbit(cells):
    return {_shape(_transform(cells, op)) for op in range(8)}


def _enclosure(foreground):
    """Fill complement components that cannot reach a padded exterior."""
    a, b, c, d = _box(foreground)
    box = (a - 1, b + 1, c - 1, d + 1)
    complement = _rectangle(box) - foreground
    cavities = [part for part in _components(complement)
                if not any(r in (a - 1, b) or x in (c - 1, d) for r, x in part)]
    holes = frozenset().union(*cavities)
    return foreground | holes, holes, cavities


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('grid must be a 1..30 integer matrix with colors 0..9')
    return a.astype(np.int8, copy=True)


def _inspect(inp):
    colors, freq = np.unique(inp, return_counts=True)
    assert sum(freq == freq.max()) == 1, 'background must be uniquely modal'
    bg = int(colors[np.argmax(freq)])
    mono, keys = [], []
    for component in _components(map(tuple, np.argwhere(inp != bg).tolist())):
        palette = tuple(sorted({int(inp[p]) for p in component}))
        assert len(palette) in (1, 2), 'unexpected foreground component'
        if len(palette) == 1:
            mono.append({'cells': component, 'color': palette[0]})
        else:
            regions = []
            for color in palette:
                fg = frozenset(p for p in component if int(inp[p]) == color)
                assert len(_components(fg)) == 1, 'disconnected key region'
                env, holes, cavities = _enclosure(fg)
                assert all(int(inp[p]) == bg for p in holes), 'cavity contains another color'
                regions.append({'fg': fg, 'env': env, 'holes': holes,
                                'count': len(cavities), 'cavities': cavities})
            assert not regions[0]['env'] & regions[1]['env']
            keys.append({'cells': component, 'colors': palette, 'regions': regions})
    mono.sort(key=lambda v: min(v['cells']))
    keys.sort(key=lambda v: min(v['cells']))
    assert 1 <= len(keys) <= 2 and len(mono) == 2 * len(keys)
    targets = []
    for i in range(len(mono)):
        for j in range(i + 1, len(mono)):
            if mono[i]['color'] != mono[j]['color']:
                continue
            fixed = mono[i]['cells'] | mono[j]['cells']
            box = _box(fixed)
            rect = _rectangle(box)
            if frozenset(p for p in rect if int(inp[p]) != bg) != fixed:
                continue
            parts = _components(rect - fixed)
            large = [p for p in parts if len(p) > 1]
            if len(large) != 1:
                continue
            slot = large[0]
            clues = frozenset().union(*(p for p in parts if len(p) == 1))
            a, b, c, d = box
            if any(r in (a, b - 1) or x in (c, d - 1) for r, x in clues):
                continue
            lr = any(x == c for r, x in slot) and any(x == d - 1 for r, x in slot)
            tb = any(r == a for r, x in slot) and any(r == b - 1 for r, x in slot)
            if not (lr or tb):
                continue
            targets.append({'parts': (i, j), 'rect': rect, 'slot': slot,
                            'clues': clues, 'count': len(clues),
                            'color': mono[i]['color'], 'box': box})
    assert targets, 'no locks found'
    return {'bg': bg, 'mono': mono, 'keys': keys, 'targets': targets}


def _placements(key, target, shape, *, use_counts=True, policy=(0, 1)):
    sr, _, sc, _ = _box(target['slot'])
    result, signatures = [], set()
    for pi, plug in enumerate(key['regions']):
        if use_counts and plug['count'] != target['count']:
            continue
        if target['color'] in key['colors']:
            continue
        for op in range(8):
            env = _transform(plug['env'], op)
            er, _, ec, _ = _box(env)
            dr, dc = sr - er, sc - ec
            if _shift(env, dr, dc) != target['slot']:
                continue
            moved = [{name: _shift(_transform(region[name], op), dr, dc)
                      for name in ('fg', 'env', 'holes')} for region in key['regions']]
            if moved[1 - pi]['env'] & target['rect']:
                continue
            full = moved[0]['env'] | moved[1]['env']
            if any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in full):
                continue
            palette = key['colors'][::-1] if policy[int(op >= 4)] else key['colors']
            paint = {p: color for region, color in zip(moved, palette) for p in region['fg']}
            paint.update({p: target['color'] for p in moved[pi]['holes']})
            signature = (pi, tuple(sorted(paint.items())), tuple(sorted(full)))
            if signature in signatures:
                continue
            signatures.add(signature)
            result.append({'target': target, 'pi': pi, 'op': op, 'full': full,
                           'paint': paint, 'moved': moved})
    return result


def _answers(inp, *, use_counts=True, policy=(0, 1), limit=2):
    analysis = _inspect(inp)
    candidates = []
    for ki, key in enumerate(analysis['keys']):
        moves = []
        for target in analysis['targets']:
            for move in _placements(key, target, inp.shape, use_counts=use_counts, policy=policy):
                move['ki'] = ki
                moves.append(move)
        candidates.append(moves)
    answers, seen = [], set()

    def search(ki, used_parts, chosen):
        if len(answers) >= limit:
            return
        if ki == len(candidates):
            if used_parts != set(range(len(analysis['mono']))):
                return
            out = np.full(inp.shape, analysis['bg'], dtype=np.int8)
            for move in chosen:
                target = move['target']
                for p in target['rect'] - target['slot']:
                    out[p] = target['color']
            for move in chosen:
                for p, color in move['paint'].items():
                    out[p] = color
            signature = out.tobytes()
            if signature not in seen:
                seen.add(signature)
                answers.append((out, tuple(chosen)))
            return
        for move in candidates[ki]:
            target = move['target']
            parts = set(target['parts'])
            if used_parts & parts:
                continue
            if any(target['rect'] & old['target']['rect']
                   or move['full'] & old['full']
                   or move['full'] & old['target']['rect']
                   or old['full'] & target['rect'] for old in chosen):
                continue
            search(ki + 1, used_parts | parts, chosen + [move])

    search(0, set(), [])
    return answers, analysis


def _decode(inp):
    answers, analysis = _answers(inp)
    assert len(answers) == 1, 'input needs a unique complete colored answer'
    out, moves = answers[0]
    analysis['moves'] = moves
    return out, analysis


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the unique answer using only visible input geometry."""
    inp = _array(input_grid)
    try:
        return _decode(inp)[0]
    except AssertionError as error:
        raise ValueError(str(error)) from error


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _silhouette(rng, h, w):
    cap = min(2, (h - 4) // 2)
    upper = rng.integers(0, cap + 1, size=w)
    lower = rng.integers(0, cap + 1, size=w)
    # Full extent, but no enforced axis or perimeter color marker.
    upper[int(rng.integers(w))] = 0
    lower[int(rng.integers(w))] = 0
    return frozenset((r, c) for c in range(w)
                     for r in range(int(upper[c]), h - int(lower[c])))


def _punch(rng, env, count, witness=False):
    inner = sorted(p for p in env if all((p[0] + dr, p[1] + dc) in env for dr, dc in DIR4))
    if count == 0:
        return frozenset()
    for attempt in range(50):
        holes = set()
        if witness and count == 2:
            pairs = [(p, (p[0] + 1, p[1] + dc)) for p in inner for dc in (-1, 1)
                     if (p[0] + 1, p[1] + dc) in inner]
            rng.shuffle(pairs)
            for pair in pairs:
                if len(_components(env - set(pair))) == 1:
                    holes.update(pair)
                    break
        else:
            for index in rng.permutation(len(inner)):
                p = inner[int(index)]
                if any((p[0] + dr, p[1] + dc) in holes for dr, dc in DIR4):
                    continue
                if len(_components(env - holes - {p})) != 1:
                    continue
                holes.add(p)
                if len(holes) == count:
                    break
        if len(holes) != count:
            continue
        desired = (2 if witness and count == 2 else
                   int(rng.integers(3, min(12, len(inner)) + 1)) if witness else
                   int(rng.integers(count, len(inner) + 1)))
        order = list(inner)
        for _ in range(len(inner)):
            if len(holes) >= desired:
                break
            rng.shuffle(order)
            grew = False
            for p in order:
                if p in holes or not any((p[0] + dr, p[1] + dc) in holes for dr, dc in DIR4):
                    continue
                trial = holes | {p}
                if len(_components(trial)) == count and len(_components(env - trial)) == 1:
                    holes = trial
                    grew = True
                    break
            if not grew:
                break
        filled, recovered, _ = _enclosure(env - holes)
        if filled == env and recovered == holes and (not witness or count != 1 or len(holes) >= 3):
            return frozenset(holes)
    raise ValueError('cavity capacity exhausted')


def _key_template(rng, counts, *, witness=False, forbidden=()):
    for _ in range(160):
        try:
            if witness:
                h, w = int(rng.integers(6, 9)), 5
                env0 = _silhouette(rng, h, w)
                if len(_orbit(env0)) != 8 or _canonical(env0) in forbidden:
                    continue
                env1 = _shift(env0, 0, w)
            else:
                h, w = int(rng.integers(6, 9)), int(rng.integers(5, 8))
                env0 = _silhouette(rng, h, w)
                if rng.random() < .45:
                    env1 = _shift(env0, 0, w)
                else:
                    h1, w1 = int(rng.integers(6, 9)), int(rng.integers(5, 8))
                    env1 = _shift(_silhouette(rng, h1, w1), int(rng.integers(-1, 2)), w)
            holes0 = _punch(rng, env0, counts[0], witness)
            holes1 = _punch(rng, env1, counts[1], witness)
            fg0, fg1 = env0 - holes0, env1 - holes1
            if len(_components(fg0 | fg1)) != 1:
                continue
            regions = [{'fg': fg, 'env': env, 'holes': holes, 'count': k}
                       for fg, env, holes, k in zip((fg0, fg1), (env0, env1), (holes0, holes1), counts)]
            op = int(rng.integers(8))
            transformed = [{name: _transform(region[name], op) for name in ('fg', 'env', 'holes')}
                           | {'count': region['count']} for region in regions]
            a, _, c, _ = _box(transformed[0]['env'] | transformed[1]['env'])
            for region in transformed:
                for name in ('fg', 'env', 'holes'):
                    region[name] = _shift(region[name], -a, -c)
            return transformed
        except ValueError:
            continue
    raise RuntimeError('bounded key construction exhausted')


def _clues(rng, rect, slot, count):
    if not count:
        return frozenset()
    a, b, c, d = _box(rect)
    fixed = rect - slot
    options = [p for p in sorted(fixed) if a < p[0] < b - 1 and c < p[1] < d - 1
               and all((p[0] + dr, p[1] + dc) not in slot for dr, dc in DIR4)]
    for _ in range(60):
        holes = set()
        for i in rng.permutation(len(options)):
            p = options[int(i)]
            if any((p[0] + dr, p[1] + dc) in holes for dr, dc in DIR4):
                continue
            if len(_components(fixed - holes - {p})) == 2:
                holes.add(p)
                if len(holes) == count:
                    return frozenset(holes)
    raise ValueError('not enough isolated clue locations')


def _lock(rng, regions, pi, op, colors, reserve_alternatives=False):
    plug, other = regions[pi], regions[1 - pi]
    a, b, c, d = _box(plug['env'])
    oa, ob, oc, od = _box(other['env'])
    horizontal = oc >= d or od <= c
    last_error = None
    for _ in range(15):
        pad0, pad1 = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        box = (a - pad0, b + pad1, c, d) if horizontal else (a, b, c - pad0, d + pad1)
        rect = _rectangle(box)
        slot = plug['env']
        try:
            assert len(_components(rect - slot)) == 2
            clues = _clues(rng, rect, slot, plug['count'])
            break
        except (AssertionError, ValueError) as error:
            last_error = error
    else:
        raise ValueError('lock construction failed') from last_error
    moved = [{name: _transform(region[name], op) for name in ('fg', 'env', 'holes')}
             | {'count': region['count']} for region in regions]
    target = {'rect': _transform(rect, op), 'slot': _transform(slot, op),
              'clues': _transform(clues, op), 'color': colors[0], 'count': plug['count'], 'parts': ()}
    key = {'regions': regions, 'colors': colors[1:]}
    reserved = target['rect'] | moved[0]['env'] | moved[1]['env']
    if reserve_alternatives:
        # Translate to open positive space solely to reuse the geometric fitter.
        shift0 = 30
        shifted_target = {**target, 'rect': _shift(target['rect'], shift0, shift0),
                          'slot': _shift(target['slot'], shift0, shift0)}
        for alt in _placements(key, shifted_target, (90, 90), use_counts=False):
            reserved |= _shift(alt['full'], -shift0, -shift0)
    ar, br, ac, bc = _box(reserved)
    target = {**target, **{name: _shift(target[name], -ar, -ac) for name in ('rect', 'slot', 'clues')}}
    moved = [{name: _shift(region[name], -ar, -ac) for name in ('fg', 'env', 'holes')}
             | {'count': region['count']} for region in moved]
    return {'target': target, 'moved': moved, 'pi': pi, 'op': op, 'colors': colors,
            'shape': (br - ar, bc - ac)}


def _shelf_fallback(rng, shapes):
    """Enumerate shelf partitions of at most four small bounding rectangles."""
    choices = []
    n = len(shapes)
    for order in permutations(range(n)):
        for cuts in range(1 << (n - 1)):
            rows, row = [], []
            for j, index in enumerate(order):
                row.append(index)
                if j == n - 1 or (cuts >> j) & 1:
                    rows.append(row)
                    row = []
            heights = [max(shapes[i][0] for i in row) for row in rows]
            widths = [sum(shapes[i][1] for i in row) + len(row) - 1 for row in rows]
            height, width = sum(heights) + len(rows) - 1, max(widths)
            if height <= 30 and width <= 30:
                choices.append((rows, heights, widths, height, width))
    assert choices, 'no shelf packing within 30'
    rows, heights, widths, height, width = _pick(rng, choices)
    origins = [None] * n
    top = int(rng.integers(31 - height))
    for row, rh, rw in zip(rows, heights, widths):
        left = int(rng.integers(31 - rw))
        for index in row:
            h, w = shapes[index]
            origins[index] = (top + int(rng.integers(rh - h + 1)), left)
            left += w + 1
        top += rh + 1
    return origins


def _pack(rng, shapes):
    order = sorted(range(len(shapes)), key=lambda i: -shapes[i][0] * shapes[i][1])
    for _ in range(20):
        origins = [None] * len(shapes)
        used = []
        for i in order:
            h, w = shapes[i]
            options = [(r, c) for r in range(31 - h) for c in range(31 - w)]
            rng.shuffle(options)
            for r, c in options:
                if all(r + h < a or b < r or c + w < x or y < c for a, b, x, y in used):
                    origins[i] = (r, c)
                    used.append((r, r + h, c, c + w))
                    break
            else:
                break
        if all(p is not None for p in origins):
            return origins
    return _shelf_fallback(rng, shapes)


def _colors(rng, stable_palette, n):
    palette = stable_palette if stable_palette is not None else tuple(map(int, rng.permutation(10)))
    bg, pool = palette[0], palette[1:]
    if stable_palette is not None:
        triples = [(pool[0], pool[1], pool[2])] * n
    else:
        richness = int(rng.integers(3, 10))
        triples = [tuple(map(int, rng.choice(pool[:richness], 3, replace=False))) for _ in range(n)]
    return bg, triples


def _build_pair(rng, templates, plugs, parities, bg, triples, witness=False):
    systems = [_lock(rng, regions, pi, int(rng.integers(4)) + 4 * parity, colors, witness)
               for regions, pi, parity, colors in zip(templates, plugs, parities, triples)]
    source_shapes = []
    for regions in templates:
        a, b, c, d = _box(regions[0]['env'] | regions[1]['env'])
        assert a == c == 0
        source_shapes.append((b, d))
    shapes = [system['shape'] for system in systems] + source_shapes
    origins = _pack(rng, shapes)
    inp = np.full((30, 30), bg, dtype=np.int8)
    out = inp.copy()
    for i, (system, regions) in enumerate(zip(systems, templates)):
        dr, dc = origins[i]
        target = system['target']
        tc, c0, c1 = system['colors']
        for p in _shift(target['rect'] - target['slot'] - target['clues'], dr, dc):
            inp[p] = tc
        for p in _shift(target['rect'] - target['slot'], dr, dc):
            out[p] = tc
        palette = (c1, c0) if system['op'] >= 4 else (c0, c1)
        for region, color in zip(system['moved'], palette):
            for p in _shift(region['fg'], dr, dc):
                out[p] = color
        for p in _shift(system['moved'][system['pi']]['holes'], dr, dc):
            out[p] = tc
        sr, sc = origins[len(systems) + i]
        for region, color in zip(regions, (c0, c1)):
            for p in _shift(region['fg'], sr, sc):
                inp[p] = color
    if rng.random() < .92:
        # Keep the entire reserved box, including training's alternative fits.
        lowr = min(p[0] for p in origins)
        lowc = min(p[1] for p in origins)
        highr = max(p[0] + s[0] for p, s in zip(origins, shapes))
        highc = max(p[1] + s[1] for p, s in zip(origins, shapes))
        lowr = max(0, lowr - int(rng.integers(4)))
        lowc = max(0, lowc - int(rng.integers(4)))
        highr = min(30, highr + int(rng.integers(4)))
        highc = min(30, highc + int(rng.integers(4)))
        cropped = inp[lowr:highr, lowc:highc]
        colors, freq = np.unique(cropped, return_counts=True)
        if colors[np.argmax(freq)] == bg and sum(freq == freq.max()) == 1:
            inp, out = cropped.copy(), out[lowr:highr, lowc:highc].copy()
    expected, analysis = _decode(inp)
    assert np.array_equal(expected, out), 'constructed answer disagrees with input'
    if witness:
        assert len(_answers(inp, use_counts=False)[0]) == 2, 'count witness lost'
    return {'input': inp, 'output': out}


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if (not _is_int(num_train) or not _is_int(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('counts require 2..10 train, 1..4 test, and train > test')
    return int(num_train), int(num_test)


def _signature(array):
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _is_int(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    novel_policy = _pick(rng, ('zero', 'higher', 'mixed'))
    novel_counts = {'zero': (0,), 'higher': (4, 5), 'mixed': (0, 4, 5)}[novel_policy]
    stable = tuple(map(int, rng.permutation(10))) if rng.random() < .25 else None
    witnesses = []
    forbidden = set()
    for counts in ((1, 2), (1, 3)):
        key = _key_template(rng, counts, witness=True, forbidden=forbidden)
        forbidden.add(_canonical(key[0]['env']))
        witnesses.append(key)
    # Keep both witness keys, local colors, and background unchanged: only
    # chosen end, destination orientation, dots, and arrangement change.
    witness_bg, witness_triples = _colors(rng, stable, 2)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', nt), ('test', ne)):
        for i in range(number):
            for attempt in range(70):
                try:
                    witness = split == 'train' and i < 2
                    if witness:
                        templates = witnesses
                        plugs = [i, i]
                        parities = [i, 1 - i]
                        bg, triples = witness_bg, witness_triples
                    else:
                        n = int(rng.integers(1, 3))
                        templates = []
                        for j in range(n):
                            counts = tuple(map(int, rng.choice((1, 2, 3) if split == 'train' else 6,
                                                              2, replace=False)))
                            if split == 'test' and j == 0:
                                counts = (_pick(rng, novel_counts), _pick(rng, (1, 2, 3)))
                            templates.append(_key_template(rng, counts))
                        plugs = list(map(int, rng.integers(2, size=n)))
                        if split == 'test':
                            plugs[0] = 0
                        parities = list(map(int, rng.integers(2, size=n)))
                        bg, triples = _colors(rng, stable, n)
                    pair = _build_pair(rng, templates, plugs, parities, bg, triples, witness)
                    sig = _signature(pair['input'])
                    if sig in seen:
                        continue
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
                except (AssertionError, ValueError):
                    continue
            else:
                raise RuntimeError('bounded scene construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(analysis):
    regions = [r for key in analysis['keys'] for r in key['regions']]
    return {
        'parities': {int(m['op'] >= 4) for m in analysis['moves']},
        'area_inversion': any(a['count'] < b['count'] and len(a['holes']) > len(b['holes'])
                              for key in analysis['keys'] for a in key['regions'] for b in key['regions']),
        'diagonal': any(len(_components(r['holes'], True)) < r['count'] for r in regions),
        'equal_envelopes': any(_canonical(k['regions'][0]['env']) == _canonical(k['regions'][1]['env'])
                               and k['regions'][0]['count'] != k['regions'][1]['count'] for k in analysis['keys']),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, parities, count_parities = set(), [], set(), set()
    count_ambiguity = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            sig = _signature(inp)
            assert sig not in seen, 'duplicate input'
            seen.add(sig)
            expected, a = _decode(inp)
            assert np.array_equal(expected, out), 'incorrect assembly output'
            for move in a['moves']:
                target, key = move['target'], a['keys'][move['ki']]
                assert len(_components(target['rect'] - target['slot'] - target['clues'])) == 2
                assert key['regions'][move['pi']]['count'] == target['count']
                assert move['moved'][move['pi']]['env'] == target['slot']
                assert not move['moved'][1 - move['pi']]['env'] & target['rect']
                assert all(int(out[p]) == target['color'] for p in target['clues'])
                assert all(int(out[p]) == target['color'] for p in move['moved'][move['pi']]['holes'])
                assert all(int(out[p]) == a['bg'] for p in move['moved'][1 - move['pi']]['holes'])
                # Generated packing keeps destinations clear of every source.
                assert all(int(out[p]) == a['bg'] for p in key['cells'])
                assert all(int(out[p]) == int(inp[p]) for p in target['rect'] - target['slot'] - target['clues'])
                if split == 'train':
                    parity = int(move['op'] >= 4)
                    parities.add(parity)
                    count_parities.add((target['count'], parity))
            if split == 'train':
                training.append(_features(a))
                count_ambiguity |= len(_answers(inp, use_counts=False)[0]) > 1
    assert parities == {0, 1}, 'training must expose both color operations'
    assert {(1, 0), (1, 1)} <= count_parities, 'count/parity confounding'
    assert any(f['parities'] == {0, 1} for f in training), 'local color exchange not demonstrated'
    assert count_ambiguity, 'shape-only shortcut not contradicted'
    assert any(f['area_inversion'] for f in training), 'area shortcut not contradicted'
    assert any(f['diagonal'] for f in training), 'four/eight connectivity not distinguished'
    assert any(f['equal_envelopes'] for f in training), 'no topological role witness'


def self_test():
    """Bounded semantic checks and structural measurements on 42 fixed seeds."""
    shapes, contours, holes, operations, counts, palettes, normalized, fingerprints = (set() for _ in range(8))
    relations, plug_relations, shifts = set(), set(), set()
    reuse, zero, varying, misleading_proximity = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        parts, dimensions, train_counts = [], set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                original = inp.copy()
                assert np.array_equal(solve(inp.tolist()), out)
                assert np.array_equal(solve(inp.copy()), out)
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(inp, original), 'solve mutated its input'
                parts.append(_signature(inp))
                shapes.add(inp.shape)
                dimensions.add(inp.shape)
                roles = {}
                normalized.add((inp.shape, tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                palettes.add(tuple(map(int, np.unique(inp))))
                _, a = _decode(inp)
                targets = {m['target']['color'] for m in a['moves']}
                accents = {c for k in a['keys'] for c in k['colors']}
                reuse += bool(targets & accents)
                for move in a['moves']:
                    operations.add(move['op'])
                    counts.add(move['target']['count'])
                    key = a['keys'][move['ki']]
                    plug_relations.add(int(np.sign(len(key['regions'][move['pi']]['fg'])
                                                   - len(key['regions'][1 - move['pi']]['fg']))))
                    kr0, kr1, kc0, kc1 = _box(key['cells'])
                    center = ((kr0 + kr1) / 2, (kc0 + kc1) / 2)
                    def distance(target):
                        r0, r1, c0, c1 = target['box']
                        return abs(center[0] - (r0 + r1) / 2) + abs(center[1] - (c0 + c1) / 2)
                    misleading_proximity += any(distance(other['target']) < distance(move['target'])
                                                for other in a['moves'] if other is not move)
                    if split == 'train':
                        train_counts.add(move['target']['count'])
                if split == 'test':
                    novel = {m['target']['count'] for m in a['moves']} - train_counts
                    assert novel, 'test has no intended count shift'
                    if 0 in novel:
                        shifts.add('zero')
                    if novel & {4, 5}:
                        shifts.add('higher')
                for key in a['keys']:
                    regions = key['regions']
                    relations.add(int(np.sign(len(regions[0]['env']) - len(regions[1]['env']))))
                    for region in regions:
                        contours.add(_canonical(region['env']))
                        holes.add((_canonical(region['env']), _canonical(region['holes']) if region['holes'] else ()))
                        zero += region['count'] == 0
        varying += len(dimensions) > 1
        fingerprints.add(tuple(parts))
    assert len(fingerprints) == 42 and len(normalized) >= 180
    assert len(shapes) >= 25 and len(contours) >= 50 and len(holes) >= 100
    assert len(palettes) >= 20 and operations == set(range(8))
    assert counts == set(range(6)) and relations == {-1, 0, 1}
    assert plug_relations == {-1, 0, 1} and shifts == {'zero', 'higher'}
    assert reuse > 0 and zero > 0 and varying == 42 and misleading_proximity > 0
    # Verify all explicit legal requests, determinism, and illegal types/counts.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    a, b = generate_puzzle(81, 2, 1), generate_puzzle(81, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    validate_generated_puzzle({'train': list(reversed(a['train'])), 'test': a['test']})
    # Concrete ablations from the minimum-count puzzle.
    analyses = [_decode(p['input'])[1] for p in a['train']]
    assert any(len(_answers(p['input'], use_counts=False)[0]) == 2 for p in a['train'])
    assert all(not np.array_equal(_answers(p['input'], policy=(0, 0))[0][0][0], p['output'])
               for p in a['train'])
    assert all(not np.array_equal(_answers(p['input'], policy=(1, 1))[0][0][0], p['output'])
               for p in a['train']), 'scene-wide swapping explains training'
    assert any(len(r['holes']) > r['count'] for d in analyses for k in d['keys'] for r in k['regions'])
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # An orbit-1 square envelope: the exterior arm and canvas boundary, rather
    # than artificial plug asymmetry, select the only legal colored placement.
    symmetric = np.zeros((10, 20), dtype=np.int8)
    symmetric[1:10, 0:3] = 2
    symmetric[4:7, 0:3] = 0
    symmetric[2, 1] = 0
    symmetric[4:7, 12:15] = 3
    symmetric[5, 13] = 0
    symmetric[0:6, 15] = 4
    symmetric[5, 16] = 4
    expected = np.zeros_like(symmetric)
    expected[1:10, 0:3] = 2
    expected[4:7, 0:3] = 3
    expected[5, 1] = 2
    expected[0:6, 3] = 4
    expected[5, 4] = 4
    assert np.array_equal(solve(symmetric), expected)
    assert len(_orbit(_decode(symmetric)[1]['moves'][0]['target']['slot'])) == 1
    # More space admits the vertically reversed arm, which exchanges colors.
    # Such an input is outside the unique-answer family and must be rejected.
    ambiguous = np.zeros((25, 20), dtype=np.int8)
    ambiguous[8:17, 0:3] = 2
    ambiguous[11:14, 0:3] = 0
    ambiguous[9, 1] = 0
    ambiguous[4:7, 12:15] = 3
    ambiguous[5, 13] = 0
    ambiguous[0:6, 15] = 4
    ambiguous[5, 16] = 4
    assert len(_answers(ambiguous)[0]) == 2
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous colored outputs accepted')
    return {'seeds': 42, 'grid_shapes': len(shapes), 'envelope_shapes': len(contours),
            'cavity_configurations': len(holes), 'color_normalized_inputs': len(normalized),
            'palettes': len(palettes), 'operations': sorted(operations), 'plug_counts': sorted(counts),
            'cross_role_reuse_pairs': reuse, 'misleading_proximity_keys': misleading_proximity,
            'plug_handle_size_relations': sorted(plug_relations), 'test_shifts': sorted(shifts)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
