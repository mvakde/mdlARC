"""First-complete harvest: cumulative set completion controls unary transfer.

Only NumPy and the standard library are used. Solving parses the actual pixels;
generation paints its expected result from independently chosen cut lengths.
"""
from __future__ import annotations

import numpy as np


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _connected(mask):
    remaining = set(zip(*np.nonzero(mask)))
    if not remaining:
        return False
    queue = [remaining.pop()]
    for y, x in queue:
        for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if point in remaining:
                remaining.remove(point)
                queue.append(point)
    return not remaining


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid, sides 1..30, colors 0..9")
    return a


def _is_header(block, background):
    if block.shape[0] != 4 or block.shape[1] < 4:
        return False
    b = block[:, :4]
    frame = int(b[3, 3])
    values, counts = np.unique(b[:3, :3], return_counts=True)
    return (frame != background and np.all(b[:, 3] == frame)
            and np.all(b[3] == frame)
            and np.count_nonzero(b[:3, :3] == background) == 6
            and len(values) == 4 and frame not in values
            and sorted(counts.tolist()) == [1, 1, 1, 6]
            and np.all(block[:, 4:] == background))


def _parse(input_grid):
    """Recover lanes, selector, local pitch, and bottom-first source records."""
    a = _array(input_grid)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg), "A blank top row identifies background"
    spans = _runs(np.any(a != bg, axis=0))
    assert 2 <= len(spans) <= 6
    lanes, headers, pitches, baselines = [], [], set(), set()
    for x, end in spans:
        assert 4 <= end - x <= 6
        groups = _runs(np.any(a[:, x:end] != bg, axis=1))
        assert groups
        header = None
        if _is_header(a[groups[-1][0]:groups[-1][1], x:end], bg):
            header = groups.pop()
            headers.append((len(lanes), header))
        assert groups
        blocks = [a[y:stop, x:end] for y, stop in groups]
        assert all(2 <= b.shape[0] <= 4 for b in blocks)
        assert all(b.shape == blocks[0].shape for b in blocks)
        starts = [y for y, _ in groups]
        pitches.update(b - c for c, b in zip(starts, starts[1:]))
        baselines.add(groups[-1][1])
        lanes.append(dict(x=x, end=end, groups=groups, blocks=blocks,
                          header=header, height=len(groups)))
    assert len(headers) == 1 and len(baselines) == 1
    assert len(pitches) == 1, "Pitch must be visible in this input"
    pitch = pitches.pop()
    dst_index, (header_y, header_stop) = headers[0]
    dst = lanes[dst_index]
    h, w = dst['blocks'][0].shape
    assert all(l['blocks'][0].shape == (h, w) for l in lanes)
    assert h < pitch <= h + 2
    baseline = baselines.pop()
    assert 1 <= header_y - baseline <= 3
    assert np.all(a[baseline:header_y] == bg)
    assert np.all(a[header_stop:] == bg)
    b = a[header_y:header_stop, dst['x']:dst['x'] + 4]
    frame = int(b[3, 3])
    query = frozenset(int(v) for v in b[:3, :3].flat if v != bg)
    assert len(query) == 3 and np.count_nonzero(a == frame) == 7
    mask = dst['blocks'][0] != bg
    assert _connected(mask) and np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))
    colors = set(int(v) for v in dst['blocks'][0][mask])
    assert len(colors) == 1
    color = colors.pop()
    assert color not in query | {bg, frame}
    for block in dst['blocks']:
        assert np.array_equal(block != bg, mask)
        assert np.all(block[mask] == color)
    sources, papers = [], set()
    for index, lane in enumerate(lanes):
        if index == dst_index:
            continue
        cards = []
        for (y, stop), block in reversed(list(zip(lane['groups'], lane['blocks']))):
            assert np.all(block != bg), "Source records are solid rectangles"
            paper = int(block[0, 0])
            assert np.all(block[:, 0] == paper)
            papers.add(paper)
            assert not set(int(v) for v in block.flat) & {frame, color}
            labels = frozenset(int(v) for v in block.flat if v != paper)
            cards.append(dict(y=y, stop=stop, labels=labels,
                              marks=int(np.count_nonzero(block != paper)),
                              counts={int(v): int(n) for v, n in
                                      zip(*np.unique(block, return_counts=True))}))
        collected, cut = set(), 0
        for k, card in enumerate(cards, 1):
            collected.update(card['labels'])
            if query <= collected:
                cut = k
                break
        sources.append(dict(x=lane['x'], end=lane['end'], cards=cards,
                            height=len(cards), cut=cut,
                            labels=frozenset().union(*(c['labels'] for c in cards))))
    assert len(papers) == 1
    paper = papers.pop()
    assert paper not in query | {bg, frame, color}
    total = sum(s['cut'] for s in sources)
    bottom_y = dst['groups'][-1][0]
    assert bottom_y - (dst['height'] + total - 1) * pitch >= 1
    return dict(background=bg, frame=frame, paper=paper, query=query,
                color=color, mask=mask, pitch=pitch, h=h, w=w, dst=dst,
                sources=sources, total=total, header_y=header_y,
                bottom_y=bottom_y, shape=a.shape)


def _paint_token(grid, y, x, color, mask):
    h, w = mask.shape
    assert y >= 1 and y + h <= grid.shape[0] and x + w <= grid.shape[1]
    grid[y:y + h, x:x + w][mask] = color


def _from_parsed(a, info):
    result = a.copy()
    for source in info['sources']:
        for card in source['cards'][:source['cut']]:
            result[card['y']:card['stop'], source['x']:source['end']] = info['background']
    dst = info['dst']
    for k in range(dst['height'], dst['height'] + info['total']):
        _paint_token(result, info['bottom_y'] - k * info['pitch'], dst['x'],
                     info['color'], info['mask'])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, examples, hidden conventions, or generation state are needed."""
    a = _array(input_grid)
    return _from_parsed(a, _parse(a))


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _geometry(rng, teaching=False, small_pitch=False):
    h = 2 if small_pitch else int(rng.integers(2, 5))
    w = int(rng.integers(4, 6 if teaching else 7))
    gap = 1 if small_pitch or (teaching and h == 4) else int(rng.integers(1, 3))
    pitch = h + gap
    capacity = 1 + (24 - h) // pitch
    mask = np.ones((h, w), bool)
    kind = int(rng.integers(5))
    if kind == 1:
        mask[:-1, 1:] = False
    elif kind == 2:
        mask[1:, :] = False
        mask[:, w // 2] = True
    elif kind == 3:
        for y in range(h - 1):
            mask[y, int(rng.integers(1, w + 1)):] = False
    elif kind == 4:
        mask[:] = False
        mask[h // 2, :] = True
        mask[:, 1:w - 1] = True
    if rng.integers(2):
        mask = mask[:, ::-1]
    if rng.integers(2):
        mask = mask[::-1]
    return dict(h=h, w=w, pitch=pitch, capacity=capacity, mask=mask.copy())


def _random_labels(rng, allowed, slots, empty_chance=.18):
    if not allowed or rng.random() < empty_chance:
        return []
    identities = rng.choice(allowed, int(rng.integers(1, len(allowed) + 1)), replace=False).tolist()
    target = int(rng.integers(len(identities), slots + 1))
    return identities + rng.choice(identities, target - len(identities)).tolist()


def _complete(rng, q, extras, n, cut, slots, style='free'):
    """Withhold one query identity until the intended first-completion card."""
    cards = [[] for _ in range(cut)]
    if cut == 1:
        cards[0] = list(q)
        if style != 'exact':
            chosen = list(extras[:int(rng.integers(len(extras) + 1))])
            cards[0] += chosen
            if len(cards[0]) < slots and rng.random() < .7:
                cards[0].append(int(rng.choice(cards[0])))
    else:
        blank = int(rng.integers(cut - 1)) if style == 'blank' else -1
        earlier = [i for i in range(cut - 1) if i != blank]
        for color in q[:2]:
            cards[int(rng.choice(earlier))].append(color)
        cards[-1] = [q[2]]
        # No individual card contains the entire query, even the stopping card.
        for i in range(cut):
            if i == blank:
                continue
            optional = list(extras) if style != 'blank' else []
            if optional and rng.random() < .75:
                cards[i].append(int(rng.choice(optional)))
            if cards[i] and style != 'blank' and len(cards[i]) < slots:
                cards[i].append(cards[i][0])
        if style != 'blank' and not any(set(c) & set(extras) for c in cards):
            cards[-1].append(extras[0])
    cards += [_random_labels(rng, list(q) + list(extras), slots) for _ in range(n - cut)]
    return dict(cards=cards, cut=cut)


def _incomplete(rng, q, extras, n, slots, missing=None, style='free'):
    if style == 'empty':
        return dict(cards=[[] for _ in range(n)], cut=0)
    if missing is None:
        missing = int(rng.integers(3))
    allowed = [v for j, v in enumerate(q) if j != missing] + list(extras)
    cards = [_random_labels(rng, allowed, slots) for _ in range(n)]
    if style == 'rich':
        cards[0] = [v for j, v in enumerate(q) if j != missing] + list(extras[:2])
        cards[0] += [cards[0][0], cards[0][-1]]
    elif style == 'matched':
        cards[0] = [v for j, v in enumerate(q) if j != missing] + [extras[0]]
    return dict(cards=cards, cut=0)


def _teaching_choices(rng, g1, g2):
    choices = []
    for a in (3, 4):
        for da in range(1, 4):
            for b in (3, 4):
                for db in range(1, 4):
                    if (a + da <= g1['capacity'] and b + db <= g2['capacity']
                            and a != b and da != db and a * db != b * da
                            and a + da != b + db):
                        choices.append((a, da, b - 1, db))
    assert choices
    return choices[int(rng.integers(len(choices)))]


def _sources(rng, geometry, q, extras, package, choice=None, shift=None, train_info=None):
    capacity = geometry['capacity']
    slots = geometry['h'] * (geometry['w'] - 1)
    max_sources = 31 // (geometry['w'] + 1) - 1
    if package == 'prefix':
        cut, dest = choice
        n = int(rng.integers(cut + 1, capacity + 1))
        positive = _complete(rng, q, extras, n, cut, slots, 'blank')
        # The tail contains a complete query but survives: collection stops
        # permanently at the first completed bottom prefix. Keeping this
        # stack's union exactly Q also defeats distinct-color thresholds.
        positive['cards'][cut] = list(q)
        for j in range(cut + 1, n):
            positive['cards'][j] = _random_labels(rng, list(q), slots)
        sources = [positive]
        for missing in range(3):
            if missing == 0:
                # Equal height, mark count, and distinct-color count, with a
                # single identity substituted throughout the whole stack.
                cards = [[extras[0] if v == q[0] else v for v in card]
                         for card in positive['cards']]
                sources.append(dict(cards=cards, cut=0))
            else:
                count = int(rng.integers(1, capacity + 1))
                sources.append(_incomplete(rng, q, extras, count, slots, missing, 'rich'))
    elif package == 'aggregation':
        cut, dest = choice
        n1 = int(rng.integers(2, capacity + 1))
        n2 = int(rng.integers(cut + 1, capacity + 1))
        sources = [_complete(rng, q, extras, n1, 1, slots, 'exact'),
                   _complete(rng, q, extras, n2, cut, slots)]
        negative = _incomplete(rng, q, extras, int(rng.integers(2, capacity + 1)),
                               slots, style='matched')
        negative['cards'][1] = []
        sources.append(negative)
        if max_sources >= 4 and rng.random() < .8:
            sources.append(_incomplete(rng, q, extras, int(rng.integers(1, capacity + 1)),
                                       slots, style='empty'))
    else:
        if package == 'test' and shift == 'more_sources':
            contributors = int(rng.integers(3, min(4, max_sources, capacity - 2) + 1))
            cuts = [1] * contributors
            cuts[int(rng.integers(contributors))] = 2
            spare = capacity - sum(cuts) - 1
            if spare and rng.random() < .5:
                cuts[int(rng.integers(contributors))] += int(rng.integers(1, spare + 1))
            dest = int(rng.integers(1, capacity - sum(cuts) + 1))
        elif package == 'test' and shift == 'deeper_scan':
            cut = int(rng.integers(train_info['max_cut'] + 1, capacity))
            cuts = [cut]
            dest = int(rng.integers(1, capacity - cut + 1))
        elif package == 'test' and shift == 'larger_receiver':
            dest = int(rng.integers(train_info['max_dest'] + 1, capacity))
            total = int(rng.integers(1, min(3, capacity - dest) + 1))
            cuts = [total] if total == 1 or rng.random() < .5 else [1, total - 1]
        else:
            dest = int(rng.integers(1, min(3, capacity - 1) + 1))
            contributors = int(rng.integers(0, min(2, max_sources, capacity - dest) + 1))
            cuts = []
            for i in range(contributors):
                maximum = min(4, capacity - dest - sum(cuts) - (contributors - i - 1))
                cuts.append(int(rng.integers(1, maximum + 1)))
        count = int(rng.integers(max(1, len(cuts)), max_sources + 1))
        sources = [_complete(rng, q, extras, int(rng.integers(k, capacity + 1)), k, slots)
                   for k in cuts]
        for _ in range(count - len(cuts)):
            sources.append(_incomplete(rng, q, extras, int(rng.integers(1, capacity + 1)), slots,
                                       style='empty' if rng.random() < .2 else 'free'))
    if dest == 1 and max(len(s['cards']) for s in sources) == 1:
        # A harmless tail makes pitch visible even when every operand is one.
        sources[0]['cards'].append([])
    assert len(sources) <= max_sources
    return sources, dest


def _layout(rng, count, w):
    minimum = count * w + count - 1
    width = int(rng.integers(minimum, 31))
    extras = rng.multinomial(width - minimum, np.full(count + 1, 1 / (count + 1)))
    xs, x = [], int(extras[0])
    for i in range(count):
        xs.append(x)
        x += w + (1 + int(extras[i + 1]) if i + 1 < count else 0)
    return width, xs


def _draw_pair(rng, geometry, colors, sources, dest_height):
    bg, frame, paper, dest_color = colors[:4]
    q = colors[4:7]
    h, w, pitch, mask = (geometry[k] for k in ('h', 'w', 'pitch', 'mask'))
    total = sum(s['cut'] for s in sources)
    tallest = max(dest_height + total, max(len(s['cards']) for s in sources))
    minimum = 6 + h + (tallest - 1) * pitch
    assert minimum <= 30
    spare = 30 - minimum
    gap = int(rng.integers(1, min(3, spare + 1) + 1))
    spare -= gap - 1
    bottom_margin = int(rng.integers(0, min(3, spare) + 1))
    minimum += gap - 1 + bottom_margin
    height = int(rng.integers(minimum, 31))
    width, xs = _layout(rng, len(sources) + 1, w)
    header_y = height - bottom_margin - 4
    bottom_y = header_y - gap - h
    a = np.full((height, width), bg, dtype=np.int8)
    b = a.copy()
    dest_index = int(rng.integers(len(xs)))
    dest_x = xs.pop(dest_index)
    rng.shuffle(sources)
    for x, source in zip(xs, sources):
        for k, marks in enumerate(source['cards']):
            y = bottom_y - k * pitch
            block = np.full((h, w), paper, dtype=np.int8)
            positions = rng.choice(h * (w - 1), len(marks), replace=False)
            for pos, label in zip(positions, rng.permutation(marks)):
                block[int(pos) // (w - 1), 1 + int(pos) % (w - 1)] = label
            a[y:y + h, x:x + w] = block
            if k >= source['cut']:
                b[y:y + h, x:x + w] = block
    for k in range(dest_height):
        _paint_token(a, bottom_y - k * pitch, dest_x, dest_color, mask)
    for k in range(dest_height + total):
        _paint_token(b, bottom_y - k * pitch, dest_x, dest_color, mask)
    header = np.full((4, 4), bg, dtype=np.int8)
    header[:, 3] = frame
    header[3, :] = frame
    for pos, label in zip(rng.choice(9, 3, replace=False), rng.permutation(q)):
        header[int(pos) // 3, int(pos) % 3] = label
    for grid in (a, b):
        grid[header_y:header_y + 4, dest_x:dest_x + 4] = header
    return dict(input=a, output=b)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(['more_sources', 'deeper_scan', 'larger_receiver']))
    varied_palette = bool(rng.integers(2))
    base_colors = [int(v) for v in rng.permutation(10)]
    g1, g2 = _geometry(rng, teaching=True), _geometry(rng, teaching=True)
    if max(g1['capacity'], g2['capacity']) < 6:
        # Different operands, ratios AND resulting heights need capacity six
        # in at least one witness. Smaller tokens supply visible extra room.
        g2 = _geometry(rng, teaching=True, small_pitch=True)
    a, da, k, db = _teaching_choices(rng, g1, g2)
    puzzle, seen = dict(train=[], test=[]), set()
    train_info = None
    for index in range(num_train + num_test):
        package = 'prefix' if index == 0 else 'aggregation' if index == 1 else 'free' if index < num_train else 'test'
        geometry = g1 if index == 0 else g2 if index == 1 else _geometry(
            rng, small_pitch=package == 'test' and shift != 'more_sources')
        # Three sources plus a receiver require width at most six; at the
        # largest token height, pitch five still fits four added tokens.
        if package == 'test' and shift == 'more_sources':
            geometry['pitch'] = geometry['h'] + 1
            geometry['capacity'] = 1 + (24 - geometry['h']) // geometry['pitch']
        choice = (a, da) if index == 0 else (k, db) if index == 1 else None
        if index == num_train:
            infos = [_parse(p['input']) for p in puzzle['train']]
            train_info = dict(max_cut=max(s['cut'] for i in infos for s in i['sources']),
                              max_dest=max(i['dst']['height'] for i in infos))
        for attempt in range(32):
            colors = [int(v) for v in rng.permutation(10)] if varied_palette else list(base_colors)
            if not varied_palette:
                labels = list(rng.permutation(base_colors[4:]))
                colors[4:] = [int(v) for v in labels]
            extras = colors[7:7 + int(rng.integers(2 if package in ('prefix', 'aggregation') else 1, 4))]
            sources, dest = _sources(rng, geometry, colors[4:7], extras, package,
                                      choice, shift, train_info)
            pair = _draw_pair(rng, geometry, colors, sources, dest)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError('Bounded distinct-input sampling exhausted')
        seen.add(key)
        puzzle['train' if index < num_train else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(infos):
    records = [(i, s) for i in infos for s in i['sources']]
    consumed = [(i, c) for i, s in records for c in s['cards'][:s['cut']]]
    retained = [(i, c) for i, s in records for c in s['cards'][s['cut']:]]
    assert any(s['cut'] >= 3 and any(not c['labels'] for c in s['cards'][:s['cut']])
               and not any(i['query'] <= c['labels'] for c in s['cards'][:s['cut']])
               for i, s in records), 'Teach cumulative completion and counting empty cards'
    assert any(s['cut'] and s['height'] > s['cut'] for _, s in records), 'Teach first completion'
    assert any(all(any(s['cut'] == 0 and s['labels'] & i['query'] == i['query'] - {q}
                          for s in i['sources']) for q in i['query']) for i in infos)
    assert any(s['cut'] == 0 and len(s['labels']) >= 4 for _, s in records)
    assert any(p['cut'] and not n['cut'] and p['height'] == n['height']
               and len(p['labels']) == len(n['labels'])
               and sum(c['marks'] for c in p['cards']) == sum(c['marks'] for c in n['cards'])
               for i in infos for p in i['sources'] for n in i['sources']), 'Matched stack identity contrast'
    assert any(not c['labels'] for _, c in retained)
    assert any(c['marks'] == 3 and c['labels'] == i['query'] for i, c in consumed)
    assert any(c['marks'] == 3 and len(c['labels']) == 3
               and len(c['labels'] & i['query']) == 2 for i, c in retained)
    assert any(c['labels'] - i['query'] for i, c in consumed)
    assert any(any(c['counts'].get(q, 0) > 1 for q in i['query']) for i, c in consumed)
    assert any(len([s for s in i['sources'] if s['cut']]) >= 2
               and len({s['cut'] for s in i['sources'] if s['cut']}) > 1 for i in infos)
    active = [i for i in infos if i['total']]
    assert len({i['total'] for i in active}) >= 2
    assert len({i['dst']['height'] for i in active}) >= 2
    assert len({i['dst']['height'] + i['total'] for i in active}) >= 2
    assert any(a['total'] * b['dst']['height'] != b['total'] * a['dst']['height']
               for a in active for b in active)
    rivals = [lambda i: 2 * i['dst']['height'], lambda i: i['total'],
              lambda i: i['dst']['height'] * i['total'],
              lambda i: i['dst']['height'] + sum(bool(s['cut']) for s in i['sources']),
              lambda i: i['dst']['height'] + max(s['cut'] for s in i['sources']),
              lambda i: i['dst']['height'] + sum(s['height'] for s in i['sources'] if s['cut'])]
    for rival in rivals:
        assert any(rival(i) != i['dst']['height'] + i['total'] for i in active)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    infos, seen = dict(train=[], test=[]), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _array(pair['input']), _array(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _parse(a)
            assert np.array_equal(_from_parsed(a, info), b), 'Wrong cut, growth, or preservation'
            # State family-specific preservation and conservation explicitly.
            x, y = info['dst']['x'], info['header_y']
            assert np.array_equal(a[y:y + 4, x:x + 4], b[y:y + 4, x:x + 4])
            erased_area = info['total'] * info['h'] * info['w']
            added_area = info['total'] * int(info['mask'].sum())
            assert np.count_nonzero(a != info['background']) - np.count_nonzero(b != info['background']) == erased_area - added_area
            infos[split].append(info)
    _training_evidence(infos['train'])
    # The sampler promises a test shift in contributing count, scan depth,
    # or initial receiver height, rather than a new operation or convention.
    tr = infos['train']
    max_contributors = max(sum(s['cut'] > 0 for s in i['sources']) for i in tr)
    max_cut = max(s['cut'] for i in tr for s in i['sources'])
    max_dest = max(i['dst']['height'] for i in tr)
    for i in infos['test']:
        assert (sum(s['cut'] > 0 for s in i['sources']) > max_contributors
                or max(s['cut'] for s in i['sources']) > max_cut
                or i['dst']['height'] > max_dest)


def _puzzle_key(p):
    return tuple(tuple((_key(v['input']), _key(v['output'])) for v in p[s])
                 for s in ('train', 'test'))


def self_test() -> dict:
    """Bounded finite evidence; no claim of independent semantic approval."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    features = {k: set() for k in ('shapes', 'geometry', 'positions', 'roles', 'cuts',
                                  'source_counts', 'receiver_counts', 'mask', 'queries')}
    blank_consumed = no_match = all_match = 0
    for p in puzzles:
        for split in ('train', 'test'):
            for pair in p[split]:
                i = _parse(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                features['shapes'].add(i['shape'])
                features['geometry'].add((i['h'], i['w'], i['pitch']))
                features['positions'].add((i['dst']['x'], tuple(s['x'] for s in i['sources'])))
                features['roles'].add(tuple(i[k] for k in ('background', 'frame', 'paper', 'color')))
                features['cuts'].update(s['cut'] for s in i['sources'])
                features['source_counts'].add(len(i['sources']))
                features['receiver_counts'].add(i['dst']['height'])
                features['mask'].add((i['mask'].shape, i['mask'].tobytes()))
                features['queries'].add(tuple(sorted(i['query'])))
                blank_consumed += sum(not c['labels'] for s in i['sources'] for c in s['cards'][:s['cut']])
                no_match += i['total'] == 0
                all_match += all(s['cut'] for s in i['sources'])
                # Permute positions of the three selector cells: decisions
                # are unchanged while the header itself is copied faithfully.
                moved = pair['input'].copy()
                x, y = i['dst']['x'], i['header_y']
                moved[y:y + 3, x:x + 3] = np.roll(moved[y:y + 3, x:x + 3].copy(), 1, axis=0)
                expected = pair['output'].copy()
                expected[y:y + 4, x:x + 4] = moved[y:y + 4, x:x + 4]
                assert np.array_equal(solve(moved), expected)
        # Isolate a rich negative and the receiver from a demonstration, then
        # change only the selector. This gives a capacity-safe counterfactual
        # within the same family, with no geometry or count changes between
        # the two inputs being compared.
        found = False
        for pair in p['train']:
            i = _parse(pair['input'])
            if i['total'] == 0:
                continue
            for source in i['sources']:
                if (not source['cut'] and source['height'] > 1
                        and len(source['cards'][0]['labels']) >= 3):
                    labels = sorted(source['cards'][0]['labels'])[:3]
                    if frozenset(labels) == i['query']:
                        continue
                    base = pair['input'].copy()
                    for other in i['sources']:
                        if other['x'] != source['x']:
                            base[:, other['x']:other['end']] = i['background']
                    changed = base.copy()
                    x, y = i['dst']['x'], i['header_y']
                    header = changed[y:y + 3, x:x + 3]
                    for point, c in zip(np.argwhere(header != i['background']), labels):
                        header[tuple(point)] = c
                    new = _parse(changed)
                    selected = next(s for s in new['sources'] if s['x'] == source['x'])
                    assert selected['cut'] == 1
                    old_output, new_output = solve(base), solve(changed)
                    c = source['cards'][0]
                    assert np.array_equal(old_output[c['y']:c['stop'], source['x']:source['end']],
                                          base[c['y']:c['stop'], source['x']:source['end']])
                    assert np.all(new_output[c['y']:c['stop'], source['x']:source['end']] == i['background'])
                    found = True
                    break
            if found:
                break
        assert found
        # A matching tail moved to the bottom completes immediately. Height,
        # full-stack color union, and all mark counts stay identical; only the
        # order changes. This targets the added prefix mechanism directly.
        found_order = False
        for pair in p['train']:
            i = _parse(pair['input'])
            for source in i['sources']:
                if source['cut'] < 3 or source['cut'] == source['height']:
                    continue
                tail = source['cards'][source['cut']]
                if not i['query'] <= tail['labels']:
                    continue
                bottom = source['cards'][0]
                changed = pair['input'].copy()
                x, end = source['x'], source['end']
                first = changed[bottom['y']:bottom['stop'], x:end].copy()
                changed[bottom['y']:bottom['stop'], x:end] = changed[tail['y']:tail['stop'], x:end]
                changed[tail['y']:tail['stop'], x:end] = first
                new = _parse(changed)
                selected = next(s for s in new['sources'] if s['x'] == x)
                assert selected['cut'] == 1 and selected['labels'] == source['labels']
                expected_count = i['dst']['height'] + i['total'] - source['cut'] + 1
                assert np.count_nonzero(solve(changed) == i['color']) == expected_count * i['mask'].sum()
                found_order = True
                break
            if found_order:
                break
        assert found_order
    assert len(features['shapes']) >= 30 and len(features['geometry']) >= 10
    assert len(features['positions']) >= 60 and len(features['roles']) >= 20
    assert {0, 1, 2, 3, 4, 5} <= features['cuts']
    assert {1, 2, 3, 4, 5} <= features['source_counts']
    assert max(features['receiver_counts']) >= 4 and len(features['mask']) >= 15
    assert len(features['queries']) >= 20 and blank_consumed >= 42
    assert no_match and all_match
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Output mutations target whole-card erasure, exact copied silhouettes,
    # untouched tails, and selector preservation.
    for defect in ('erasure', 'growth', 'tail', 'header'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
        for pair in bad['train']:
            i = _parse(pair['input'])
            source = next((s for s in i['sources'] if 0 < s['cut'] < s['height']), None)
            if source is not None:
                break
        if defect == 'erasure':
            pair['output'][source['cards'][0]['y'], source['x']] = i['paper']
        elif defect == 'growth':
            y = i['bottom_y'] - i['dst']['height'] * i['pitch']
            yy, xx = np.argwhere(i['mask'])[0]
            pair['output'][y + yy, i['dst']['x'] + xx] = i['background']
        elif defect == 'tail':
            pair['output'][source['cards'][source['cut']]['y'], source['x']] = i['background']
        else:
            pair['output'][i['header_y'] + 3, i['dst']['x']] = i['background']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Corrupt output accepted: ' + defect)
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in p),
                distinct={k: len(v) for k, v in features.items()},
                blank_cards_consumed=blank_consumed, no_match=no_match,
                all_match=all_match, independent_semantic_verification=False)
