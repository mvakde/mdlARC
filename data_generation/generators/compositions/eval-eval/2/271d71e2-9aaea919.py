"""Refueling relay: spatially gated token addition drives a unary-fuel vehicle.

All solving conventions are visible in the query. No files, examples, RNG, or
generation metadata are consulted by solve(). See task_spec.md for the domain.
"""
from collections import defaultdict
import hashlib
import json

import numpy as np


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((0 <= g) & (g <= 9))), "invalid colors")
    return g.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        point = min(todo)
        todo.remove(point)
        stack, cells = [point], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bounds(cells):
    ys, xs = zip(*cells)
    return min(ys), max(ys), min(xs), max(xs)


def _find_frame(g, bg):
    candidates = []
    for color in np.unique(g):
        if int(color) == bg:
            continue
        for cells in _components(g == color):
            t, b, l, r = _bounds(cells)
            if b - t < 2 or r - l < 2:
                continue
            if len(cells) == 2 * (b - t + r - l) and all(
                    y in (t, b) or x in (l, r) for y, x in cells):
                candidates.append((int(color), t, b, l, r))
    _need(len(candidates) == 1, "need one rectangular ring")
    return candidates[0]


def _read_scene(value):
    original = _grid(value)
    bg = int(original[0, 0])
    _need(all(np.all(edge == bg) for edge in
              (original[0], original[-1], original[:, 0], original[:, -1])),
          "outer border must identify the background")
    f, t, b, l, r = _find_frame(original, bg)
    # Turning by k puts the attached bar above the rectangle.
    sides = ((0, original[t - 1, l:r + 1]),
             (2, original[b + 1, l:r + 1]),
             (3, original[t:b + 1, l - 1]),
             (1, original[t:b + 1, r + 1]))
    leads = [(k, int(v[0])) for k, v in sides
             if len(v) and int(v[0]) not in (bg, f) and np.all(v == v[0])]
    _need(len(leads) == 1, "ambiguous leading side")
    k, m = leads[0]
    g = np.rot90(original, k).copy()
    _, t, b, l, r = _find_frame(g, bg)
    other = set(map(int, np.unique(g))) - {bg, f, m}
    _need(len(other) == 1, "ready fuel must match the token color")
    ready = other.pop()
    front = t - 1
    destinations = [y for y in range(front) if np.all(g[y, l:r + 1] == m)]
    _need(len(destinations) == 1, "need one destination bar")
    dest = destinations[0]
    _need(np.all(g[dest + 1:front, l:r + 1] == bg), "obstructed corridor")
    _need(np.all(g[front, l:r + 1] == m), "incomplete leading bar")
    values = g[t + 1:b, l + 1:r].ravel().tolist()
    spent = values.count(m)
    _need(values == [m] * spent + [ready] * (len(values) - spent),
          "counter must be a travel-relative spent prefix")
    cover = np.zeros(g.shape, dtype=np.int8)
    cover[t:b + 1, l:r + 1] = 1
    cover[front, l:r + 1] += 1
    cover[dest, l:r + 1] += 1
    token_mask = g == ready
    token_mask[t:b + 1, l:r + 1] = False
    groups = defaultdict(list)
    common_mask = None
    for cells in _components(token_mask):
        y0, y1, x0, x1 = _bounds(cells)
        _need(x1 <= l - 4 or x0 >= r + 4, "token inside the travel lane")
        side = -1 if x1 < l else 1
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
        for y, x in cells:
            mask[y - y0, x - x0] = True
        _need(mask.shape[0] in (1, 3) and 1 <= mask.shape[1] <= 3,
              "unsupported token dimensions")
        if common_mask is None:
            common_mask = mask
        _need(np.array_equal(common_mask, mask), "tokens must share a silhouette")
        groups[(y0, y1, side)].append((x0, x1, cells))
    _need(bool(groups), "need token stacks")
    stacks, pitches = [], set()
    for (y0, y1, side), tokens in sorted(groups.items()):
        tokens.sort(key=lambda a: a[0])
        n = len(tokens)
        w = tokens[0][1] - tokens[0][0] + 1
        _need(1 <= n <= 4, "stack count must be 1..4")
        if side == -1:
            _need(tokens[-1][1] == l - 4, "left stack is not anchored")
            control_x = l - 2
        else:
            _need(tokens[0][0] == r + 4, "right stack is not anchored")
            control_x = r + 2
        gaps = [z[0] - a[0] for a, z in zip(tokens, tokens[1:])]
        _need(not gaps or len(set(gaps)) == 1 and gaps[0] > w,
              "irregular stack pitch")
        pitches.update(gaps)
        control = [(y, control_x) for y in range(y0, y1 + 1)]
        control_values = [int(g[p]) for p in control]
        marked = control_values == [f] * len(control)
        _need(marked or control_values == [bg] * len(control), "invalid source control")
        p = front - (y0 + y1) // 2
        _need(1 <= p < front - dest, "stations must be strictly inside the journey")
        cells = [cell for token in tokens for cell in token[2]]
        for cell in cells + (control if marked else []):
            cover[cell] += 1
        stacks.append(dict(p=p, side=side, n=n, marked=marked,
                           cells=cells, control=control, area=len(cells)))
    _need(len(pitches) <= 1, "one token pitch per scene")
    _need(any(s['marked'] for s in stacks), "need a marked source")
    levels = sorted({s['p'] for s in stacks})
    _need(all(z - a >= common_mask.shape[0] + 1 for a, z in zip(levels, levels[1:])),
          "station levels must have a separating row")
    _need(np.array_equal(cover, (g != bg).astype(np.int8)), "unexplained foreground")
    scene = dict(bg=bg, frame=f, marker=m, ready=ready, top=t, bottom=b,
                 left=l, right=r, front=front, distance=front - dest,
                 capacity=len(values), fuel=len(values) - spent, stacks=stacks,
                 token_shape=common_mask.shape,
                 token_signature=(common_mask.shape, common_mask.tobytes()),
                 pitch=next(iter(pitches), None))
    return g, k, scene


def _walk(scene, rival=None, skip_level=None):
    """Pixel-reader's unit-step dynamics; only a small state tuple is retained."""
    cap, fuel, distance = scene['capacity'], scene['fuel'], scene['distance']
    stacks = scene['stacks']
    if rival == 'preload':
        fuel = min(cap, fuel + sum(s['n'] for s in stacks if s['marked']))
    pos, picked, saturated = 0, set(), 0
    while pos < distance and fuel > 0:
        pos += 1
        fuel -= 1
        at = [i for i, s in enumerate(stacks) if s['p'] == pos and
              (s['marked'] or rival == 'all_stacks')]
        if at:
            amounts = [stacks[i]['n'] for i in at]
            amount = sum(amounts)
            if rival in ('no_refill', 'preload') or pos == skip_level:
                amount = 0
            elif rival == 'bars':
                amount = len(at)
            elif rival == 'max_stack':
                amount = max(amounts)
            elif rival == 'pixels':
                amount = sum(stacks[i]['area'] for i in at)
            saturated += int(fuel + amount > cap)
            fuel = fuel + amount if rival == 'no_capacity' else min(cap, fuel + amount)
            picked.update(i for i in at if stacks[i]['marked'])
    return pos, fuel, frozenset(picked), saturated


def _event_walk(scene):
    """Construction-side calculation jumps between events, rather than cells."""
    pos, fuel, picked, saturated = 0, scene['fuel'], set(), 0
    events = defaultdict(list)
    for i, source in enumerate(scene['stacks']):
        if source['marked']:
            events[source['p']].append(i)
    for p, indices in sorted(events.items()):
        if p - pos > fuel:
            break
        fuel -= p - pos
        pos = p
        total = sum(scene['stacks'][i]['n'] for i in indices)
        saturated += int(fuel + total > scene['capacity'])
        fuel = min(scene['capacity'], fuel + total)
        picked.update(indices)
    advance = min(fuel, scene['distance'] - pos)
    return pos + advance, fuel - advance, frozenset(picked), saturated


def _paint_output(g, scene, result):
    pos, fuel, picked, _ = result
    t, b, l, r = (scene[key] for key in ('top', 'bottom', 'left', 'right'))
    out = g.copy()
    out[t:b + 1, l:r + 1] = scene['bg']
    out[t - 1, l:r + 1] = scene['bg']
    out[t - pos:b + 1 - pos, l:r + 1] = scene['frame']
    counter = np.full(scene['capacity'], scene['ready'], dtype=np.int8)
    counter[:scene['capacity'] - fuel] = scene['marker']
    out[t + 1 - pos:b - pos, l + 1:r] = counter.reshape((b - t - 1, r - l - 1))
    out[t - 1 - pos, l:r + 1] = scene['marker']
    for i in picked:
        for cell in scene['stacks'][i]['cells']:
            out[cell] = scene['marker']
        for cell in scene['stacks'][i]['control']:
            out[cell] = scene['bg']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive geometry, palette, fuel, events, and final state from input only."""
    g, k, scene = _read_scene(input_grid)
    return np.rot90(_paint_output(g, scene, _walk(scene)), -k).copy()


def _stack(p, side, n, marked=True):
    return dict(p=int(p), side=int(side), n=int(n), marked=bool(marked))


def _lesson_a(rng, h):
    # An empty-on-arrival rescue, unequal simultaneous operands, then a trap.
    p = int(rng.integers(2, 6))
    operands = [(a, b) for a in range(1, 4) for b in range(1, 4)
                if a != b and a + b <= 5]
    a, b = operands[int(rng.integers(len(operands)))]
    total = a + b
    dims = [(y, x) for y in range(2, 5) for x in range(2, 6)
            if y * x > max(p, total)]
    ih, iw = dims[int(rng.integers(len(dims)))]
    future = p + total + int(rng.integers(max(1, h + 1 - total), 4))
    distance = future + int(rng.integers(2, 5))
    side = int(rng.choice((-1, 1)))
    stacks = [_stack(p, -1, a), _stack(p, 1, b),
              _stack(future, side, int(rng.integers(1, 4)))]
    if rng.random() < .5:
        stacks.append(_stack(future, -side, int(rng.integers(1, 4)), False))
    return dict(ih=ih, iw=iw, fuel=p, distance=distance, stacks=stacks)


def _lesson_b(rng, h):
    # A full tank clips the first refill; it then reaches a second source empty.
    dims = [(y, x) for y in range(1, 4) for x in range(1, 5) if 4 <= y * x <= 9]
    ih, iw = dims[int(rng.integers(len(dims)))]
    cap = ih * iw
    p = int(rng.integers(1, 4))
    q = p + cap
    second = int(rng.integers(2, min(4, cap - 1) + 1))
    distance = q + int(rng.integers(1, min(3, second) + 1))
    side = int(rng.choice((-1, 1)))
    second_side = int(rng.choice((-1, 1)))
    stacks = [_stack(p, side, p + 1), _stack(q, second_side, second),
              _stack(q, -second_side, int(rng.integers(1, 5)), False)]
    return dict(ih=ih, iw=iw, fuel=cap, distance=distance, stacks=stacks)


def _test_route(rng, h):
    dims = [(y, x) for y in range(1, 5) for x in range(1, 5) if 6 <= y * x <= 12]
    ih, iw = dims[int(rng.integers(len(dims)))]
    cap = ih * iw
    p = int(rng.integers(1, 4))
    levels = [p]
    for _ in range(2):
        levels.append(levels[-1] + int(rng.integers(h + 1, 6)))
    initial = p + int(rng.integers(0, min(2, levels[1] - p - 1) + 1))
    fuel, last, stacks = initial, 0, []
    for i, p in enumerate(levels):
        fuel -= p - last
        needed = levels[i + 1] - p if i < 2 else 1
        choices = [(a,) for a in range(1, 5)] + [(a, b) for a in range(1, 5) for b in range(1, 5)]
        # Limit carry-over so removing either early pickup blocks the next one.
        if i == 0:
            upper = needed + min(2, levels[2] - levels[1] - 1)
        elif i == 1:
            upper = needed + 2
        else:
            upper = cap
        choices = [amounts for amounts in choices
                   if needed <= min(cap, fuel + sum(amounts)) <= upper]
        amounts = choices[int(rng.integers(len(choices)))]
        side = int(rng.choice((-1, 1)))
        stacks.append(_stack(p, side, amounts[0]))
        if len(amounts) == 2:
            stacks.append(_stack(p, -side, amounts[1]))
        elif rng.random() < .5:
            stacks.append(_stack(p, -side, int(rng.integers(1, 5)), False))
        before_pickup = fuel
        fuel = min(cap, fuel + sum(amounts))
        last = p
    # Both destination and fuel stops occur when the actual geometry fits.
    gap = max(1, h + 1 - fuel)
    trap = last + fuel + gap
    if trap + 2 <= 25 - ih and rng.random() < .45:
        distance = trap + 2
        stacks.append(_stack(trap, int(rng.choice((-1, 1))), int(rng.integers(1, 5))))
    else:
        distance = last + int(rng.integers(before_pickup + 1, min(4, fuel) + 1))
    return dict(ih=ih, iw=iw, fuel=initial, distance=distance, stacks=stacks)


def _extra_route(rng, h):
    ih, iw = int(rng.integers(1, 5)), int(rng.integers(1, 6))
    distance = int(rng.integers(5, min(20, 25 - ih) + 1))
    positions = list(range(1, distance))
    rng.shuffle(positions)
    levels = []
    for p in positions:
        if all(abs(p - q) >= h + 1 for q in levels):
            levels.append(p)
        if len(levels) >= int(rng.integers(1, 5)):
            break
    stacks = []
    for p in sorted(levels):
        side = int(rng.choice((-1, 1)))
        stacks.append(_stack(p, side, int(rng.integers(1, 5))))
        if rng.random() < .55:
            stacks.append(_stack(p, -side, int(rng.integers(1, 5)), rng.random() < .55))
    spec = dict(ih=ih, iw=iw, fuel=int(rng.integers(ih * iw + 1)),
                distance=distance, stacks=stacks)
    basic = dict(capacity=ih * iw, **{key: spec[key] for key in ('fuel', 'distance', 'stacks')})
    result = _event_walk(basic)
    if len({stacks[i]['p'] for i in result[2]}) > 2:
        return _lesson_a(rng, h)
    return spec


def _token_shape(rng, h, w):
    if h == 1 or w == 1 or rng.random() < .3:
        return np.ones((h, w), dtype=bool)
    if w == 3 and rng.random() < .4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0], mask[-1, :] = True, True
    for y in range(h - 1):
        mask[y, :int(rng.integers(1, w + 1))] = True
    return np.rot90(mask, 2).copy() if rng.random() < .5 else mask


def _render(rng, spec, h, palette, rotation, teaching=False):
    bg, f, m, ready = palette
    ih, iw, distance = spec['ih'], spec['iw'], spec['distance']
    maxima = {side: max((s['n'] for s in spec['stacks'] if s['side'] == side), default=0)
              for side in (-1, 1)}
    options = []
    for w in range(1, 4):
        for pitch in range(w + 1, w + 3):
            extent = {side: w + 3 + (n - 1) * pitch if n else 0 for side, n in maxima.items()}
            if sum(extent.values()) + iw + 4 <= 30:
                options.append((w, pitch, extent))
    _need(bool(options), "no feasible token geometry")
    if teaching:
        informative = [option for option in options if h * option[0] > 1]
        if informative:
            options = informative
        else:
            # Anchor event separations are at least four rows even when h=1.
            h = 3
    w, pitch, extent = options[int(rng.integers(len(options)))]
    mask = _token_shape(rng, h, w)
    base_width = sum(extent.values()) + iw + 4
    slack = int(rng.integers(0, min(4, 30 - base_width) + 1))
    left_margin = 1 + int(rng.integers(slack + 1))
    width = base_width + slack
    l = left_margin + extent[-1]
    r = l + iw + 1
    base_height = distance + ih + 5
    _need(base_height <= 30, "journey too tall")
    slack_y = int(rng.integers(0, min(3, 30 - base_height) + 1))
    top_margin = 1 + int(rng.integers(slack_y + 1))
    front = top_margin + distance
    t, b = front + 1, front + ih + 2
    g = np.full((base_height + slack_y, width), bg, dtype=np.int8)
    g[t:b + 1, l:r + 1] = f
    cap = ih * iw
    counter = np.array([m] * (cap - spec['fuel']) + [ready] * spec['fuel'], dtype=np.int8)
    g[t + 1:b, l + 1:r] = counter.reshape((ih, iw))
    g[front, l:r + 1] = m
    g[top_margin, l:r + 1] = m
    stacks = []
    for source in spec['stacks']:
        p, side, n = (source[key] for key in ('p', 'side', 'n'))
        y0 = front - p - h // 2
        control_x = l - 2 if side == -1 else r + 2
        control = [(y0 + y, control_x) for y in range(h)]
        cells = []
        for index in range(n):
            x0 = l - 3 - w - index * pitch if side == -1 else r + 4 + index * pitch
            for y, x in np.argwhere(mask):
                cell = (y0 + int(y), x0 + int(x))
                g[cell] = ready
                cells.append(cell)
        if source['marked']:
            for cell in control:
                g[cell] = f
        stacks.append(dict(source, cells=cells, control=control, area=len(cells)))
    scene = dict(bg=bg, frame=f, marker=m, ready=ready, top=t, bottom=b, left=l, right=r,
                 front=front, capacity=cap, fuel=spec['fuel'], distance=distance, stacks=stacks)
    expected = _paint_output(g, scene, _event_walk(scene))
    return {'input': np.rot90(g, rotation).copy(), 'output': np.rot90(expected, rotation).copy()}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(g):
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    varying_structures = bool(rng.integers(2))
    rotations = rng.permutation(4).tolist()
    lessons = rng.permutation(2).tolist()
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for index in range(count):
            for _ in range(64):
                h = int(rng.choice((1, 3)))
                if split == 'test':
                    spec = _test_route(rng, h)
                elif index < 2:
                    spec = (_lesson_a if lessons[index] == 0 else _lesson_b)(rng, h)
                else:
                    spec = _extra_route(rng, h)
                local = palette
                if varying_structures:
                    bg, f = map(int, rng.choice([c for c in range(10) if c not in palette[2:]], 2, replace=False))
                    local = (bg, f, *palette[2:])
                rotation = rotations[index] if split == 'train' and index < 2 else int(rng.integers(4))
                pair = _render(rng, spec, h, local, rotation, split == 'train' and index < 2)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(scene, result):
    pos, fuel, picked, saturated = result
    stacks = scene['stacks']
    return dict(distance=scene['distance'], capacity=scene['capacity'], initial=scene['fuel'],
                position=pos, fuel=fuel, saturated=saturated,
                visits=len({stacks[i]['p'] for i in picked}), sources=sum(s['marked'] for s in stacks),
                ignored=sum(not s['marked'] for s in stacks),
                unreached=sum(s['marked'] and i not in picked for i, s in enumerate(stacks)),
                mask=scene.get('token_shape'), pitch=scene.get('pitch'))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, states, train_records = set(), set(), []
    rivals = {name: False for name in ('no_refill', 'preload', 'bars', 'max_stack',
                                      'pixels', 'all_stacks', 'no_capacity')}
    training_directions = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "shape changed")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            g, k, scene = _read_scene(inp)
            states.add((scene['marker'], scene['ready']))
            result = _walk(scene)
            _need(result == _event_walk(scene), "step/event disagreement")
            expected = np.rot90(_paint_output(g, scene, result), -k)
            _need(np.array_equal(expected, out), "incorrect movement, counter, or source state")
            # Conservation includes discarded excess fuel, rather than hiding clipping.
            picked = result[2]
            supplied = sum(scene['stacks'][i]['n'] for i in picked)
            discarded = scene['fuel'] + supplied - result[0] - result[1]
            _need(discarded >= 0 and (discarded > 0) == (result[3] > 0), "fuel accounting failed")
            feature = _features(scene, result)
            if split == 'train':
                training_directions.add(k)
                train_records.append(feature)
                _need(feature['visits'] <= 2, "training depth contract violated")
                for name in rivals:
                    rivals[name] |= _walk(scene, name)[:3] != result[:3]
            else:
                _need(feature['visits'] >= 3, "test must extend the causal relay")
                levels = sorted({scene['stacks'][i]['p'] for i in picked})
                _need(_walk(scene, skip_level=levels[0])[0] < levels[1] and
                      _walk(scene, skip_level=levels[1])[0] < levels[2] and
                      _walk(scene, skip_level=levels[2])[0] < result[0],
                      "test pickups must be causally necessary")
    _need(len(states) == 1, "counter states changed within a puzzle")
    _need(len(training_directions) >= 2, "training must show rotated counter order")
    _need(any(f['position'] < f['distance'] and f['unreached'] for f in train_records),
          "training must show an unreachable marked supply")
    _need(any(f['position'] == f['distance'] for f in train_records), "training lacks destination stop")
    _need(any(f['saturated'] for f in train_records), "training lacks saturation")
    # Pixel-counting is indistinguishable when every token is a single cell.
    # The informative lessons therefore always use tokens of area > 1.
    _need(all(rivals.values()), "training fails to distinguish a documented shortcut: " +
          ','.join(name for name, exposed in rivals.items() if not exposed))


def self_test():
    """42 seeded puzzles, API boundaries, family coverage, and deliberate errors.

    Streaming aggregates only: no puzzle archive, execution traces, or file IO.
    Returned compact-serializable statistics are useful for the local report.
    """
    shapes, masks, pitches, capacities, distances, directions = set(), set(), set(), set(), set(), set()
    fingerprints, role_sets, source_counts, visits, stops = set(), set(), set(), set(), set()
    stationary = saturated = traps = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                fingerprint.update(pair['input'].tobytes())
                shapes.add(pair['input'].shape)
                g, k, scene = _read_scene(pair['input'])
                result = _walk(scene)
                f = _features(scene, result)
                role_sets.add((scene['bg'], scene['frame'], scene['marker'], scene['ready']))
                masks.add(scene['token_signature'])
                pitches.add(scene['pitch'])
                capacities.add(scene['capacity'])
                distances.add(scene['distance'])
                directions.add(k)
                source_counts.add(f['sources'])
                visits.add(f['visits'])
                stops.add(result[0] == scene['distance'])
                stationary += result[0] == 0
                saturated += f['saturated'] > 0
                traps += f['unreached'] > 0
                _need(np.array_equal(solve(pair['input']), pair['output']), "solver mismatch")
        fingerprints.add(fingerprint.hexdigest())
        if seed in (0, 17):
            repeat = generate_puzzle(seed)
            _need(all(np.array_equal(a[key], b[key]) for split in p
                      for a, b in zip(p[split], repeat[split]) for key in ('input', 'output')),
                  "nondeterministic generation")
    _need(len(fingerprints) == 42 and len(shapes) >= 40, "insufficient scene variation")
    _need(len(role_sets) >= 35 and len(capacities) >= 10 and len(distances) >= 10,
          "insufficient role/counter/route variation")
    _need(len(masks) >= 6 and len(pitches - {None}) >= 3, "insufficient token geometry")
    _need(directions == {0, 1, 2, 3} and {0, 1, 2, 3} <= visits and len(source_counts) >= 4,
          "missing orientations or causal/count variation")
    _need(stops == {False, True} and stationary and saturated and traps, "missing stopping cases")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(12991, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Direct family members beyond the sampler: a five-pickup one-sided route,
    # the same route with no starting fuel, and a one-cell saturated tank.
    probes = [
        (3, dict(ih=2, iw=3, fuel=1, distance=21,
                 stacks=[_stack(p, -1, 4) for p in (1, 5, 9, 13, 17)])),
        (3, dict(ih=2, iw=3, fuel=0, distance=21,
                 stacks=[_stack(p, -1, 4) for p in (1, 5, 9, 13, 17)])),
        (1, dict(ih=1, iw=1, fuel=1, distance=5,
                 stacks=[_stack(1, 1, 3), _stack(4, 1, 2)])),
    ]
    probe_rng = np.random.default_rng(3091)
    for h, spec in probes:
        for rotation in range(4):
            pair = _render(probe_rng, spec, h, (0, 5, 8, 3), rotation)
            before = pair['input'].copy()
            _need(np.array_equal(solve(pair['input']), pair['output']), "general-family probe failed")
            _need(np.array_equal(before, pair['input']), "solver mutated its input")
    original = generate_puzzle(31, 2, 1)
    # Family-specific corruptions: wrong tank cell, source color, control, stop.
    for defect in ('counter', 'source', 'control', 'position'):
        broken = {split: [{key: value.copy() for key, value in pair.items()}
                          for pair in pairs] for split, pairs in original.items()}
        pair = broken['train'][0]
        g, k, scene = _read_scene(pair['input'])
        result = _walk(scene)
        out = np.rot90(pair['output'], k).copy()
        source = scene['stacks'][min(result[2])]
        if defect == 'counter':
            cell = (scene['top'] + 1 - result[0], scene['left'] + 1)
            out[cell] = scene['ready'] if out[cell] == scene['marker'] else scene['marker']
        elif defect == 'source':
            out[source['cells'][0]] = scene['ready']
        elif defect == 'control':
            out[source['control'][0]] = scene['frame']
        else:
            out[scene['top'] - 1 - result[0], scene['left']] = scene['bg']
        pair['output'] = np.rot90(out, -k).copy()
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + defect)
    return dict(seeds=42, pairs=210, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                token_geometries=len(masks), capacities=sorted(capacities),
                distances=sorted(distances), directions=sorted(directions),
                source_counts=sorted(source_counts), visits=sorted(visits),
                stationary_pairs=stationary, saturation_pairs=saturated, trap_pairs=traps,
                direct_family_probes=12, legal_count_combinations=30,
                checks='generator-owned; no independent solver or blind prediction')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
